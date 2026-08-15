import base64
import json
import requests
import time
import io
import numpy as np
import cv2
import multiprocessing
import re
import ast
import threading
import os
from contextlib import contextmanager
from PIL import Image, ImageOps
from threading import Semaphore

# 线程锁，用于打印输出，避免混乱
print_lock = threading.Lock()

# 全局信号量，限制GPT API并发调用数（避免429错误）
# 注意：这个信号量在多进程环境下不共享，所以每个进程独立限制
# 如果要跨进程限制，需要用multiprocessing.Semaphore
_api_semaphore = Semaphore(1)  # 改为1，确保串行调用API


@contextmanager
def _cross_process_api_lock():
    """Optionally serialize API calls across independent scene processes."""
    lock_path = os.environ.get("IMAGINARIUM_GPT_LOCK_FILE", "").strip()
    if not lock_path:
        yield
        return

    # Imported lazily because fcntl is available on the Linux inference host,
    # but not on Windows development machines.
    import fcntl

    lock_parent = os.path.dirname(lock_path)
    if lock_parent:
        os.makedirs(lock_parent, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

# 全局变量，用于在工作进程中存储 agent 实例
worker_agent = None

IMAGE_PLACEHOLDER = '<image-placeholder>'

def safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    with print_lock:
        print(*args, **kwargs, flush=True)

def init_worker(agent_params):
    """初始化工作进程，创建 agent 实例"""
    global worker_agent
    worker_agent = GPTApi(**agent_params)
        
class BaseApi:
    def __init__(self) -> None:
        pass

    @staticmethod
    def encode_image(image_input, max_size=768, jpeg_quality=75):
        if isinstance(image_input, str):
            img = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input[..., ::-1] if image_input.shape[-1]==3 else image_input)
        else:
            img = image_input

        # 缩放以降低 GPT API payload 大小，减少超时
        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)

        if img.mode == 'RGBA':
            bg = Image.new('RGB', img.size, (255,255,255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        elif img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')

        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def get_response(self, prompt, image=None, **kwargs):
        raise NotImplementedError


class GPTApi(BaseApi):
    """
    GPT API Wrapper for Imaginarium.

    适配 lightspeed API:
      - POST {GPT_ENDPOINT}  -> 202 {"task_id": ...} + 轮询 (async mode)
      - POST /api/v1/gemini/stream -> SSE stream (sync mode, fast)
    endpoint 为 /gpt/call 时使用 OpenAI Responses 格式。
    endpoint 为 /gemini/call 时使用 Gemini generateContent 格式。
    endpoint 为 /gemini/stream 时使用 Gemini SSE stream (同步，低延迟)。
    """
    def __init__(self, model, GPT_KEY, GPT_ENDPOINT) -> None:
        self.GPT_KEY = GPT_KEY
        self.GPT_ENDPOINT = GPT_ENDPOINT
        self.is_stream = "/gemini/stream" in GPT_ENDPOINT
        if self.is_stream:
            self.api_family = "gemini-stream"
        elif "/gemini/" in GPT_ENDPOINT:
            self.api_family = "gemini"
        else:
            self.api_family = "gpt"
        # 由 endpoint 推导任务查询地址 (async only)
        if "/gpt/call" in GPT_ENDPOINT:
            self.TASK_URL = GPT_ENDPOINT.replace("/gpt/call", "/tasks")
        elif "/gemini/call" in GPT_ENDPOINT:
            self.TASK_URL = GPT_ENDPOINT.replace("/gemini/call", "/tasks")
        elif "/gemini/stream" in GPT_ENDPOINT:
            self.TASK_URL = GPT_ENDPOINT.replace("/gemini/stream", "/tasks")
        else:
            self.TASK_URL = GPT_ENDPOINT.rsplit("/", 1)[0] + "/tasks"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.GPT_KEY}",
        }
        self.current_model = model

    @staticmethod
    def _extract_text_from_task(task_data):
        """
        从 lightspeed 任务结果中提取助手文本。
        Responses API 的 data.output 是一个列表，可能包含 reasoning 项 (content=None)
        和 message 项 (content=[{type:'output_text', text:...}])。
        Gemini API 的 data.candidates[].content.parts[].text 也在这里兼容。
        必须遍历挑出 message 项里的 output_text，不能简单取 output[0]。
        """
        data = task_data.get("data", {}) or {}
        candidates = data.get("candidates", []) or []
        gemini_texts = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content") or {}
            for part in (content.get("parts") or []):
                if isinstance(part, dict) and part.get("text"):
                    gemini_texts.append(part["text"])
        if gemini_texts:
            return "".join(gemini_texts)

        output = data.get("output", []) or []
        texts = []
        # 优先取 type == 'message' 的项
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for c in (item.get("content") or []):
                    if isinstance(c, dict) and c.get("type") in ("output_text", "text") and c.get("text"):
                        texts.append(c["text"])
        # 兜底: 任何带 text 的 content
        if not texts:
            for item in output:
                if not isinstance(item, dict):
                    continue
                for c in (item.get("content") or []):
                    if isinstance(c, dict) and c.get("text"):
                        texts.append(c["text"])
        # 再兜底: 顶层 output_text 字段 (部分实现提供)
        if not texts and isinstance(data.get("output_text"), str):
            texts.append(data["output_text"])
        return "".join(texts)

    def _submit_and_poll(self, payload, max_wait=600, poll_interval=2, max_429_retries=5):
        """提交异步任务到 lightspeed API，轮询直到完成并返回文本。"""
        # 处理429并发限制：指数退避重试
        for attempt in range(max_429_retries):
            try:
                resp = requests.post(self.GPT_ENDPOINT, headers=self.headers, json=payload, timeout=60)
                if resp.status_code == 429:
                    wait_time = (2 ** attempt) * 10  # 10s, 20s, 40s, 80s, 160s
                    print(f"[{self.api_family.upper()} API] 429 concurrency limit, retrying in {wait_time}s (attempt {attempt+1}/{max_429_retries})")
                    time.sleep(wait_time)
                    continue
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_429_retries - 1:
                    raise Exception(f"Failed to submit task after {max_429_retries} retries: {e}")
                time.sleep(10)
        else:
            raise Exception(f"Failed to submit task: persistent 429 errors")
        
        task_info = resp.json()
        task_id = task_info.get("task_id")
        if not task_id:
            raise Exception(f"Failed to get task_id: {task_info}")

        start_time = time.time()
        while True:
            if time.time() - start_time > max_wait:
                raise TimeoutError(f"Task {task_id} timed out after {max_wait}s")
            task_resp = requests.get(f"{self.TASK_URL}/{task_id}", headers=self.headers, timeout=60)
            
            # 处理轮询时的429错误
            if task_resp.status_code == 429:
                time.sleep(poll_interval * 2)
                continue
            
            try:
                task_data = task_resp.json()
            except Exception:
                time.sleep(poll_interval)
                continue
            status = task_data.get("status")
            if status == "success":
                return self._extract_text_from_task(task_data)
            elif status in ("failed", "error"):
                raise Exception(f"Task {task_id} failed: {json.dumps(task_data, ensure_ascii=False)[:800]}")
            time.sleep(poll_interval)

    def _build_input_content(self, prompt, image):
        """构建单条 user 消息的 content 列表 (Responses 格式)。"""
        input_content = []
        if isinstance(image, list):
            if not isinstance(prompt, list):
                prompts, images = prompt.split(IMAGE_PLACEHOLDER), image
            else:
                prompts, images = prompt, image
            if images is None:
                images = [None] * (len(prompts) - 1 if len(prompts) > 0 else 0)
            input_content.append({"type": "input_text", "text": prompts[0]})
            for prompt_part, image_part in zip(prompts[1:], images):
                if image_part is not None:
                    encoded = BaseApi.encode_image(image_part)
                    input_content.append({
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{encoded}"
                    })
                input_content.append({"type": "input_text", "text": prompt_part})
            # 若 prompt 占位符少于图片数，剩余图片仍然附加
            for image_part in images[len(prompts) - 1:]:
                if image_part is not None:
                    encoded = BaseApi.encode_image(image_part)
                    input_content.append({
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{encoded}"
                    })
        elif isinstance(image, np.ndarray):
            if image is not None and image.size > 0:
                encoded = BaseApi.encode_image(image)
                input_content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encoded}"
                })
            input_content.append({"type": "input_text", "text": prompt})
        elif image is not None:
            encoded = BaseApi.encode_image(image)
            input_content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{encoded}"
            })
            input_content.append({"type": "input_text", "text": prompt})
        else:
            input_content.append({"type": "input_text", "text": prompt})
        return input_content

    def _build_gemini_parts(self, prompt, image):
        """构建 Gemini generateContent 的 parts，保持和 Responses 路径同样的图文顺序。"""
        parts = []

        def add_text(text):
            if text is not None and str(text):
                parts.append({"text": str(text)})

        def add_image(image_part):
            if image_part is None:
                return
            if isinstance(image_part, np.ndarray) and image_part.size == 0:
                return
            encoded = BaseApi.encode_image(image_part)
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": encoded,
                }
            })

        if isinstance(image, list):
            if not isinstance(prompt, list):
                prompts, images = str(prompt).split(IMAGE_PLACEHOLDER), image
            else:
                prompts, images = prompt, image
            if images is None:
                images = [None] * (len(prompts) - 1 if len(prompts) > 0 else 0)
            add_text(prompts[0] if prompts else "")
            for prompt_part, image_part in zip(prompts[1:], images):
                add_image(image_part)
                add_text(prompt_part)
            for image_part in images[len(prompts) - 1:]:
                add_image(image_part)
        else:
            add_image(image)
            add_text(prompt)
        return parts

    def _build_gemini_payload(self, prompt, image=None, **kwargs):
        max_output_tokens = kwargs.get("max_tokens", 16384)
        generation_config = {
            "maxOutputTokens": max_output_tokens,
            "temperature": kwargs.get("temperature", 0),
        }
        response_mime_type = os.environ.get("IMAGINARIUM_GEMINI_RESPONSE_MIME_TYPE", "").strip()
        if response_mime_type:
            generation_config["responseMimeType"] = response_mime_type

        return {
            "model": self.current_model,
            "systemInstruction": {
                "parts": [{
                    "text": (
                        "You are a precise visual AI assistant. "
                        "When the user asks for JSON output, respond with ONLY valid JSON or the requested Python-style literal. "
                        "No markdown code blocks, no explanations outside the requested structure, no extra text. "
                        "Use double quotes for JSON keys and string values. "
                        "Do not wrap the output in ```json markers."
                    )
                }]
            },
            "contents": [{
                "role": "user",
                "parts": self._build_gemini_parts(prompt, image),
            }],
            "generationConfig": generation_config,
        }

    def _gemini_stream(self, payload, max_wait=300):
        """Send Gemini SSE stream request, collect full text.
        Returns the complete text or raises on error."""
        # 多进程串扰：每个进程在第一次请求前加随机 jitter (0-4s)
        # 这样 4 个 worker 自然错开 ~5s 的调用窗口，避免同时触发 429
        import random as _random
        time.sleep(_random.random() * 4)
        
        # 处理429并发限制：指数退避重试
        max_429_retries = 5
        for attempt in range(max_429_retries):
            try:
                resp = requests.post(self.GPT_ENDPOINT, headers=self.headers, json=payload,
                                   stream=True, timeout=max_wait)
                if resp.status_code == 429:
                    wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s, 40s, 80s
                    print(f"[GEMINI-STREAM] 429 concurrency limit, retrying in {wait_time}s (attempt {attempt+1}/{max_429_retries})")
                    time.sleep(wait_time)
                    continue
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_429_retries - 1:
                    raise Exception(f"Failed to submit stream task after {max_429_retries} retries: {e}")
                time.sleep(5)
        else:
            raise Exception("Persistent 429 errors on stream endpoint")
        
        full_text = ""
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode('utf-8')
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                    candidates = data.get("candidates", []) or []
                    for c in candidates:
                        parts = (c.get("content") or {}).get("parts") or []
                        for p in parts:
                            full_text += p.get("text", "")
                except json.JSONDecodeError:
                    pass
        
        if not full_text:
            raise Exception("Gemini stream returned empty text")
        return full_text

    def get_response(self, prompt, image=None, history=None, return_history=None, only_return_request=False, **kwargs):
        if self.is_stream:
            # Gemini stream mode: sync SSE, fastest
            payload = self._build_gemini_payload(prompt, image=image, **kwargs)
            start_time = time.time()
            with _cross_process_api_lock():
                full_text = self._gemini_stream(payload)
            print(f"GEMINI-STREAM Time cost {time.time() - start_time:.1f}s.")
            if only_return_request:
                return payload
            if return_history:
                return full_text, [
                    payload["contents"][0],
                    {"role": "model", "parts": [{"text": full_text}]}
                ]
            return full_text
        elif self.api_family == "gemini":
            payload = self._build_gemini_payload(prompt, image=image, **kwargs)
            input_list = None
        else:
            # 1. 构建 input 列表
            input_content = self._build_input_content(prompt, image)
            cur_user_msg = {"role": "user", "content": input_content}

            if history:
                input_list = list(history) + [cur_user_msg]
            else:
                input_list = [
                    {"role": "system", "content": [
                        {"type": "input_text",
                         "text": "You are a precise visual AI assistant. "
                                "When the user asks for JSON output, respond with ONLY valid JSON — "
                                "no markdown code blocks, no explanations outside the JSON, no extra text. "
                                "Use double quotes for all keys and string values. "
                                "Do not wrap the JSON in ```json markers."}
                    ]},
                    cur_user_msg,
                ]

            max_output_tokens = kwargs.get("max_tokens", 16384)
            payload = {
                "model": self.current_model,
                "input": input_list,
                "max_output_tokens": max_output_tokens,
                "reasoning": {"effort": "low", "summary": "auto"},
            }
            # 注意: gpt-5.x reasoning 模型不接受自定义 temperature, 故此处不传 temperature

        if only_return_request:
            return payload

        # 2. 异步提交 + 重试（使用信号量限制并发）
        max_wait = int(kwargs.get("max_wait", os.environ.get("IMAGINARIUM_GPT_MAX_WAIT", 600)))
        max_retries = int(kwargs.get("max_retries", os.environ.get("IMAGINARIUM_GPT_MAX_RETRIES", 5)))
        for retry in range(max_retries):
            try:
                start_time = time.time()
                # 获取信号量，限制并发API调用
                with _cross_process_api_lock():
                    _api_semaphore.acquire()
                    try:
                        full_text = self._submit_and_poll(payload, max_wait=max_wait)
                    finally:
                        _api_semaphore.release()
                
                print(f"{self.api_family.upper()} Time cost {time.time() - start_time:.1f}s.")
                if full_text:
                    if not return_history:
                        return full_text
                    else:
                        if input_list is None:
                            input_list = [payload["contents"][0], {"role": "model", "parts": [{"text": full_text}]}]
                        else:
                            cur_answer = {"role": "assistant", "content": [
                                {"type": "output_text", "text": full_text}
                            ]}
                            input_list.append(cur_answer)
                        return full_text, input_list
            except Exception as e:
                print(f"Error when sending request to the server (Retry {retry+1}/{max_retries}): {e}", flush=True)
                if retry < max_retries - 1:
                    time.sleep(2)
        return None


def _fallback_result(return_list, return_json, return_dict):
    if return_list:
        return []
    if return_json or return_dict:
        return {}
    return None


def parallel_processing_requests(agent_params,all_image_list, all_prompt_list, return_list, return_json, return_dict, num_processes=1, timeout=None):
    """并行处理GPT请求；慢尾请求超时后返回保守空结果，避免整场景卡死。

    注意这里使用整批 deadline，而不是按结果顺序逐个 get(timeout)。
    否则排在前面的慢请求会挡住后面已经完成的结果，导致 dense scene
    的部分区域无谓丢失。
    """
    args_list = [(prompt, image_list) for prompt, image_list in zip(all_prompt_list, all_image_list)]
    if not args_list:
        return []

    if timeout is None:
        timeout = int(os.environ.get(
            "IMAGINARIUM_PARALLEL_GPT_TIMEOUT",
            os.environ.get("IMAGINARIUM_GPT_MAX_WAIT", "600")
        ))

    process_cap_env = os.environ.get("IMAGINARIUM_PARALLEL_GPT_PROCESSES", "").strip()
    if process_cap_env:
        try:
            num_processes = min(num_processes, max(1, int(process_cap_env)))
        except ValueError:
            safe_print(f"Ignoring invalid IMAGINARIUM_PARALLEL_GPT_PROCESSES={process_cap_env!r}")
    worker_count = min(len(args_list), num_processes)
    fallback = _fallback_result(return_list, return_json, return_dict)
    pool = multiprocessing.Pool(processes=worker_count, initializer=init_worker, initargs=(agent_params,))
    try:
        if return_list:
            worker_fn = process_single_request_and_return_list
        elif return_json:
            worker_fn = process_single_request_and_return_json
        elif return_dict:
            worker_fn = process_single_request_and_return_dict
        else:
            raise ValueError("return_list, return_json, or return_dict must be selected")

        async_results = [pool.apply_async(worker_fn, (args,)) for args in args_list]
        all_results = [None] * len(async_results)
        pending = set(range(len(async_results)))
        deadline = time.time() + timeout
        timed_out = False

        while pending:
            made_progress = False
            for idx in list(pending):
                async_result = async_results[idx]
                if not async_result.ready():
                    continue
                try:
                    all_results[idx] = async_result.get(timeout=0)
                except Exception as e:
                    safe_print(f"GPT request {idx + 1}/{len(args_list)} failed: {e}; using fallback.")
                    all_results[idx] = fallback
                pending.remove(idx)
                made_progress = True

            if not pending:
                break
            if time.time() >= deadline:
                timed_out = True
                safe_print(
                    f"{len(pending)}/{len(args_list)} GPT requests timed out after {timeout}s "
                    f"(pool workers={worker_count}); using fallback for unfinished requests: "
                    f"{[i + 1 for i in sorted(pending)]}"
                )
                for idx in pending:
                    all_results[idx] = fallback
                pool.terminate()
                break
            if not made_progress:
                time.sleep(0.25)

        if not timed_out:
            pool.close()
        pool.join()
    except Exception:
        pool.terminate()
        pool.join()
        raise

    return all_results

def process_single_request_and_return_list(args):
    prompt, image_list = args
    final_res = 'error'
    for _ in range(3):
        res = worker_agent.get_response(prompt, image=image_list)
        final_res = extract_list_with_re(res)
        if final_res!='error': break
    return final_res

def process_single_request_and_return_json(args):
    prompt, image_list = args
    final_res = 'error'
    for _ in range(3):
        res = worker_agent.get_response(prompt, image=image_list)
        final_res = extract_json_with_re(res)
        if final_res!='error': break
    return final_res

def process_single_request_and_return_dict(args):
    prompt, image_list = args
    final_res = 'error'
    for _ in range(3):
        res = worker_agent.get_response(prompt, image=image_list)
        final_res = extract_dict_with_re(res)
        if final_res!='error': break
    return final_res

def extract_list_with_re(output):
    if output is None or not isinstance(output, str):
        return []
    try:
        list_pattern = r'\[.*?\]'
        list_matches = re.findall(list_pattern, output, re.DOTALL)
        if list_matches:
            list_str = list_matches[-1]
            try:
                list_data = ast.literal_eval(list_str)
                if isinstance(list_data, list):
                    return list_data
            except (SyntaxError, ValueError):
                pass
        return []
    except Exception:
        return 'error'
    
def extract_dict_with_re(output):
    if output is None or not isinstance(output, str):
        return 'error'
    try:
        dict_pattern = r'\{.*\}'
        dict_match = re.search(dict_pattern, output, re.DOTALL)
        if dict_match:
            dict_str = dict_match.group(0)
            dict_str = dict_str.replace('None', 'None')
            dict_data = ast.literal_eval(dict_str)
            return dict_data
        else:
            raise ValueError("No dict found")
    except Exception:
        return 'error'
    
def extract_json_with_re(output):
    if output is None or not isinstance(output, str):
        return None
    json_match = re.search(r'\{[\s\S]*\}', output)
    if json_match:
        json_str = json_match.group()
        json_str = re.sub(r'//.*', '', json_str)
        json_str = json_str.replace('False', 'false').replace('True', 'true').replace('None', 'null')
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {}
    else:
        return 'error'
