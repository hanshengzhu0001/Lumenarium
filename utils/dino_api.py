import base64
import json
from PIL import Image
from io import BytesIO
import os

import numpy as np
from pycocotools import mask as mask_utils
from typing import Dict, List
from typing import Tuple
import concurrent.futures
import tempfile
import numpy as np
from PIL import Image
from easydict import EasyDict
import re

def string2rle(rle_str: str) -> List[int]:
    p = 0
    cnts = []

    while p < len(rle_str) and rle_str[p]:
        x = 0
        k = 0
        more = 1

        while more:
            c = ord(rle_str[p]) - 48
            x |= (c & 0x1f) << 5 * k
            more = c & 0x20
            p += 1
            k += 1

            if not more and (c & 0x10):
                x |= -1 << 5 * k

        if len(cnts) > 2:
            x += cnts[len(cnts) - 2]
        cnts.append(x)
    return cnts


def rle2mask(rle: Dict, size: Tuple[int, int], label=1):
    h, w = size
    img = np.zeros((h, w), dtype=np.uint8)

    ps = 0
    cnts = rle
    for i in range(0, len(cnts) -1, 2):
        ps += cnts[i]

        for j in range(cnts[i + 1]):
            x = (ps + j) % w
            y = (ps + j) // w

            if y < h and x < w:
                img[y, x] = label
            else:
                break

        ps += cnts[i + 1]

    return img

def rle2rgba(mask_obj) -> Image.Image:
    """
    Convert the compressed RLE string of mask object to png image object.

    :param mask_obj: The :class:`Mask <dds_cloudapi_sdk.tasks.ivp.IVPObjectMask>` object detected by this task
    """
    mask_array = mask_utils.decode(mask_obj)

    # convert the array to a 4-channel RGBA image
    mask_alpha = np.where(mask_array == 1, 255, 0).astype(np.uint8)
    mask_rgba = np.stack((255 * np.ones_like(mask_alpha),
                            255 * np.ones_like(mask_alpha),
                            255 * np.ones_like(mask_alpha),
                            mask_alpha),
                            axis=-1)
    image = Image.fromarray(mask_rgba, "RGBA")
    return image


def postprocess(result, task, return_mask):
    """Postprocess the result from the API call

    Args:
        result (TaskResult): Task result with the following keys:
            - objects (List[DetectionObject]): Each DetectionObject has the following keys:
                - bbox (List[float]): Box in xyxy format
                - category (str): Detection category
                - score (float): Detection score
                - mask (DetectionObjectMask): Use mask.counts to parse RLE mask 
        task (DetectionTask): The task object
        return_mask (bool): Whether to return mask

    Returns:
        (Dict): Return dict in format:
            {
                "scores": (List[float]): A list of scores for each object
                "categorys": (List[str]): A list of categorys for each object
                "boxes": (List[List[int]]): A list of boxes for each object
                "masks": (List[PIL.Image]): A list of masks in the format of PIL.Image
            }
    """
    def process_object_with_mask(object):
        box = object.bbox
        score = object.score
        category = object.category
        # import pdb; pdb.set_trace();
        mask = rle2rgba(object.mask)

        # Crop mask with bbox as per user's suggestion
        mask_array = np.array(mask)
        x0, y0, x1, y1 = [int(c) for c in box]
        h, w, _ = mask_array.shape
        
        # Ensure coordinates are within image bounds
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)

        # Create a new blank mask and copy the cropped part
        cropped_mask_array = np.zeros_like(mask_array)
        if y1 > y0 and x1 > x0:
            cropped_mask_array[y0:y1, x0:x1] = mask_array[y0:y1, x0:x1]
        
        mask = Image.fromarray(cropped_mask_array, "RGBA")
        return box, score, category, mask
    
    def process_object_without_mask(object):
        box = object.bbox
        score = object.score
        category = object.category
        mask = None
        return box, score, category, mask
    
    boxes, scores, categorys, masks = [], [], [], []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        if return_mask:
            process_object = process_object_with_mask
        else:
            process_object = process_object_without_mask
        futures = [executor.submit(process_object, obj) for obj in result.objects]
        for future in concurrent.futures.as_completed(futures):
            box, score, category, mask = future.result()
            boxes.append(box)
            scores.append(score)
            categorys.append(category)
            if mask is not None:
                masks.append(mask)

    return dict(boxes=boxes, categorys=categorys, scores=scores, masks=masks)


def array_to_base64(image_array):
    # 将numpy数组转换为PIL Image
    if isinstance(image_array, np.ndarray):
        img = Image.fromarray(image_array)
    else:
        img = image_array  # 如果已经是PIL Image
    # 转换为 RGB 或 RGBA（保持 PNG 支持透明度）
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    # 将图像转为二进制数据
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    # 转换为 Base64 并添加前缀
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"

def path_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_str = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"

def dino_api(prompts, token=None):
    """Detection + segmentation. Uses SAM3 text-based when IMAGINARIUM_USE_SAM3_DETECTION=1."""
    if os.environ.get("IMAGINARIUM_USE_SAM3_DETECTION", "0") == "1":
        return _sam3_text_detection(prompts)
    return _gd_detection(prompts)


def _gd_detection(prompts):
    """Original GroundingDINO v2 detection path."""
    import torch
    import numpy as np
    from PIL import Image
    from groundingdino.util.inference import load_model, preprocess_caption
    from groundingdino.util.utils import get_phrases_from_posmap
    import groundingdino, os
    from transformers import BertModel
    
    # Monkey-patch: transformers>=5.x removed get_head_mask from BertModel
    if not hasattr(BertModel, 'get_head_mask'):
        def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                head_mask = [head_mask] * num_hidden_layers
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask
        BertModel.get_head_mask = _get_head_mask
    
    global _gd_model
    if '_gd_model' not in globals():
        print("[GD] Loading local GroundingDINO model...")
        cfg_path = os.path.join(os.path.dirname(groundingdino.__file__), 'config/GroundingDINO_SwinB_cfg.py')
        weight_cache_dir = os.environ.get("IMAGINARIUM_WEIGHT_CACHE_DIR", "").strip()
        cached_ckpt = os.path.join(weight_cache_dir, "groundingdino_swinb_cogcoor.pth") if weight_cache_dir else ""
        ckpt_path = cached_ckpt if cached_ckpt and os.path.exists(cached_ckpt) else 'weights/groundingdino_swinb_cogcoor.pth'
        _gd_model = load_model(cfg_path, ckpt_path, device='cuda')
        _gd_model = _gd_model.cuda()
        print("[GD] Model loaded.")
    
    image_data = prompts['image']
    prompt_text = prompts['prompt']
    
    if isinstance(image_data, str):
        img = Image.open(image_data).convert('RGB')
    elif isinstance(image_data, np.ndarray):
        img = Image.fromarray(image_data)
    else:
        img = image_data.convert('RGB')
    
    w, h = img.size
    
    import torchvision.transforms as T
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(img).unsqueeze(0).cuda()
    
    caption = preprocess_caption(str(prompt_text))
    with torch.no_grad():
        outputs = _gd_model(image_tensor, captions=[caption])
    
    pred_logits = outputs['pred_logits'].sigmoid()[0]  # [N, 256]
    pred_boxes = outputs['pred_boxes'][0]              # [N, 4] cxcywh
    
    # Filter by score
    max_scores, _ = pred_logits.max(dim=1)
    mask = max_scores > 0.15
    pred_boxes = pred_boxes[mask]
    scores = max_scores[mask]
    pred_logits = pred_logits[mask]
    
    boxes_xyxy = []
    for box in pred_boxes:
        cx, cy, bw, bh = box.tolist()
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        boxes_xyxy.append([max(0,x1), max(0,y1), min(w,x2), min(h,y2)])
    
    scores_list = scores.tolist()
    
    # Categories from prompt. GroundingDINO logits are token positions, not
    # prompt-category indices, so decode phrases through the model tokenizer.
    prompt_tokens = str(prompt_text).replace(' . ', '.').split('.')
    prompt_tokens = [t.strip() for t in prompt_tokens if t.strip()]

    def normalize_label(value):
        return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')

    normalized_prompt_tokens = {normalize_label(t): t for t in prompt_tokens}

    def map_phrase_to_prompt_token(phrase):
        normalized_phrase = normalize_label(phrase.replace('.', ''))
        if not normalized_phrase:
            return 'object'
        if normalized_phrase in normalized_prompt_tokens:
            return normalized_prompt_tokens[normalized_phrase]

        phrase_parts = set(normalized_phrase.split('_'))
        best_token = None
        best_score = 0.0
        for token in prompt_tokens:
            token_norm = normalize_label(token)
            if not token_norm:
                continue
            if token_norm in normalized_phrase or normalized_phrase in token_norm:
                score = min(len(token_norm), len(normalized_phrase)) / max(len(token_norm), len(normalized_phrase))
            else:
                token_parts = set(token_norm.split('_'))
                score = len(phrase_parts & token_parts) / max(len(phrase_parts), len(token_parts))
            if score > best_score:
                best_score = score
                best_token = token
        return best_token if best_score >= 0.5 and best_token else 'object'

    tokenizer = _gd_model.tokenizer
    tokenized = tokenizer(caption)
    text_threshold = 0.15
    categorys = []
    for logit in pred_logits:
        phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        categorys.append(map_phrase_to_prompt_token(phrase))
    
    # --- SAM3 refinement: replace bbox-rectangle masks with fine instance masks ---
    masks = _refine_masks_with_sam3(img, boxes_xyxy, h, w)
    
    return dict(boxes=boxes_xyxy, categorys=categorys, scores=scores_list, masks=masks)


# ─── SAM3 Tracker singleton + refinement ──────────────────────────────────
_sam3_tracker = None

def _load_sam3():
    """Lazy-load SAM3 Tracker (drop-in SAM2 replacement, bbox→fine mask)."""
    global _sam3_tracker
    if _sam3_tracker is not None:
        return _sam3_tracker
    import torch
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    print("[SAM3] Loading Tracker from facebook/sam3 ...")
    _sam3_tracker = {
        "model": Sam3TrackerModel.from_pretrained(
            "facebook/sam3", torch_dtype=torch.float16
        ).cuda().eval(),
        "processor": Sam3TrackerProcessor.from_pretrained("facebook/sam3"),
    }
    print(f"[SAM3] Ready. GPU: {torch.cuda.memory_allocated()//1024//1024} MB")
    return _sam3_tracker


def _refine_masks_with_sam3(pil_image, boxes_xyxy, h, w):
    """Run SAM3 Tracker with GD bboxes as box-prompt; return fine RGBA masks.

    Batches all boxes into one forward pass for speed.
    """
    import torch
    import numpy as np
    from PIL import Image
    
    if not boxes_xyxy:
        return []
    
    tracker = _load_sam3()
    model = tracker["model"]
    processor = tracker["processor"]
    
    # prepare batched box input: [[x1,y1,x2,y2], ...] per image
    valid_boxes = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = [max(0, int(v)) for v in box]
        if x2 > x1 and y2 > y1:
            valid_boxes.append([x1, y1, x2, y2])
    
    if not valid_boxes:
        # all invalid → fallback to empty masks
        return [Image.fromarray(np.zeros((h, w, 4), dtype=np.uint8), 'RGBA')
                for _ in boxes_xyxy]
    
    # SAM3 Tracker processes all boxes in one call per image
    inputs = processor(
        images=pil_image,
        input_boxes=[valid_boxes],  # batch=1, list of boxes
        return_tensors="pt",
    )
    inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # post_process_masks returns list[N_tensors] → [B, N, H, W] binary
    refined = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"],
    )[0]  # batch 0 → [N, H, W] float
    
    # convert back to RGBA PIL images matching original box order
    refined_idx = 0
    masks = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = [max(0, int(v)) for v in box]
        if x2 <= x1 or y2 <= y1:
            masks.append(Image.fromarray(np.zeros((h, w, 4), dtype=np.uint8), 'RGBA'))
            continue
        mask_bool = (refined[refined_idx] > 0.5).numpy()  # (H, W) bool
        mask_arr = np.zeros((h, w, 4), dtype=np.uint8)
        mask_arr[mask_bool, :] = [255, 255, 255, 255]
        masks.append(Image.fromarray(mask_arr, 'RGBA'))
        refined_idx += 1
    
    return masks


# ─── SAM3 Text-based detection (replaces GD entirely) ─────────────────────
_sam3_text = None

def _load_sam3_text_model():
    """Lazy-load SAM3 for text-based open-vocabulary instance segmentation."""
    global _sam3_text
    if _sam3_text is not None:
        return _sam3_text
    import torch
    from transformers import Sam3Model, Sam3Processor
    print("[SAM3-Text] Loading from facebook/sam3 ...")
    _sam3_text = {
        "model": Sam3Model.from_pretrained(
            "facebook/sam3", torch_dtype=torch.float16
        ).cuda().eval(),
        "processor": Sam3Processor.from_pretrained("facebook/sam3"),
    }
    print(f"[SAM3-Text] Ready. GPU: {torch.cuda.memory_allocated()//1024//1024} MB")
    return _sam3_text


def _sam3_text_detection(prompts):
    """SAM3 text-based: one query per category, instance masks directly, cross-category NMS."""
    import torch
    import numpy as np
    from PIL import Image

    image_data = prompts['image']
    prompt_text = prompts['prompt']

    if isinstance(image_data, str):
        img = Image.open(image_data).convert('RGB')
    elif isinstance(image_data, np.ndarray):
        img = Image.fromarray(image_data)
    else:
        img = image_data.convert('RGB')
    w, h = img.size

    # deduplicate categories from prompt
    categories = list(dict.fromkeys(
        t.strip() for t in prompt_text.split('.') if t.strip()
    ))
    # skip structural elements (handled by dedicated wall/floor detection)
    categories = [c for c in categories if c not in ('floor', 'wall', 'ceiling')]

    sam3 = _load_sam3_text_model()
    model = sam3["model"]
    processor = sam3["processor"]

    all_boxes, all_masks, all_cats, all_scores = [], [], [], []

    for cat in categories:
        text_prompt = cat.replace('_', ' ')
        try:
            inputs = processor(images=img, text=text_prompt, return_tensors="pt")
        except Exception:
            continue
        inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs, threshold=0.3, mask_threshold=0.3,
            target_sizes=inputs["original_sizes"].tolist()
        )[0]  # batch 0

        for i in range(len(result["masks"])):
            mask_t = result["masks"][i]
            mask_np = mask_t.cpu().numpy() if torch.is_tensor(mask_t) else np.array(mask_t)
            box_t = result["boxes"][i]
            box = box_t.tolist() if torch.is_tensor(box_t) else list(box_t)
            score_t = result["scores"][i]
            score = score_t.item() if torch.is_tensor(score_t) else float(score_t)

            mask_arr = np.zeros((h, w, 4), dtype=np.uint8)
            mask_arr[mask_np > 0.5, :] = [255, 255, 255, 255]
            all_masks.append(Image.fromarray(mask_arr, 'RGBA'))
            all_boxes.append([int(max(0, v)) for v in box])
            all_cats.append(cat)
            all_scores.append(score)

    # cross-category mask NMS
    if len(all_masks) > 1:
        keep = _mask_nms(all_masks, all_scores, iou_thresh=0.6)
        all_masks = [all_masks[i] for i in keep]
        all_boxes = [all_boxes[i] for i in keep]
        all_cats = [all_cats[i] for i in keep]
        all_scores = [all_scores[i] for i in keep]

    return dict(boxes=all_boxes, categorys=all_cats, scores=all_scores, masks=all_masks)


def _mask_nms(masks, scores, iou_thresh=0.6):
    """Mask-based NMS: remove masks with high IoU, keep higher-scored."""
    import numpy as np
    n = len(masks)
    if n <= 1:
        return list(range(n))
    idxs = np.argsort(scores)[::-1]
    keep = []
    # precompute bool arrays for speed
    bool_masks = [np.array(m)[:, :, 3] > 0 for m in masks]
    while len(idxs) > 0:
        best = idxs[0]
        keep.append(best)
        remaining = []
        best_mask = bool_masks[best]
        best_area = np.sum(best_mask)
        for i in idxs[1:]:
            cur_mask = bool_masks[i]
            inter = np.sum(best_mask & cur_mask)
            union = best_area + np.sum(cur_mask) - inter
            iou = inter / union if union > 0 else 0
            if iou < iou_thresh:
                remaining.append(i)
        idxs = np.array(remaining)
    return keep
