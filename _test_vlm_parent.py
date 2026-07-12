"""Quick test: which VLM outputs parent field most consistently?"""
import json, time, base64, requests
from pathlib import Path

# Load prompt template
prompts_path = Path('prompts/used_prompts.py')
with open(prompts_path) as f:
    exec(f.read())

# Minimal prompt for a single region
prompt = GENERATE_SCENE_GRAPH_PROMPT_FLOOR_WALL.format(
    all_items_list='wall_0, wall_1, bathtub_0, bench_0, display_cabinet_0, carpet_0, single_sofa_chair_0',
    items_in_region='["bathtub_0", "bench_0", "display_cabinet_0", "carpet_0", "single_sofa_chair_0"]',
    wall_color_name=''
)

# Load image
img_path = 'demo/bathroom_01_v3.png'
with open(img_path, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiJmZTQ1NjZlNi1jNGRkLTRjYTgtYTRlNS1hZGU1MThmMmU5OWMiLCJwcm9qZWN0X25hbWUiOiJQQ0ciLCJzZXJ2aWNlcyI6eyJnZW1pbmkiOjJ9LCJleHBpcmVzX2F0IjoiMjA3Ni0wNS0wOFQwNjoyMzozNi4wMTQ1NTcrMDA6MDBaIiwiaXNzdWVkX2F0IjoiMjAyNi0wNS0yMVQwNjoyMzozNi4wMTYxMzcrMDA6MDAifQ.8MmvBhbTdzUBotmciVp7MKh9wPxIA8IwCl7uDFzRyWc"
base = "https://lightaiapi.lightspeed.qq.com"
headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

def test_gpt_call(name, model, timeout=180):
    print(f"\n=== {name} ({model}) ===")
    t0 = time.time()
    payload = {
        "model": model,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"}
        ]}]
    }
    try:
        r = requests.post(f"{base}/api/v1/gpt/call", json=payload, headers=headers, timeout=60)
        print(f"  Submit: {r.status_code}, {time.time()-t0:.1f}s")
        if r.status_code not in (200, 202):
            print(f"  Error: {r.text[:200]}")
            return None
        
        task = r.json()
        task_id = task.get('task_id', '')
        if not task_id:
            print(f"  No task_id: {task}")
            return None
        
        # Poll
        for attempt in range(timeout // 5):
            time.sleep(5)
            sr = requests.get(f"{base}/api/v1/tasks/{task_id}", headers=headers, timeout=30)
            if sr.status_code != 200:
                continue
            st = sr.json()
            status = st.get('status', st.get('state', ''))
            if status in ('completed', 'success', 'succeeded'):
                elapsed = time.time() - t0
                result = st.get('output') or st.get('result') or st.get('data')
                if isinstance(result, list) and len(result) > 0:
                    text = result[0].get('content', [{}])[0].get('text', '')
                else:
                    text = str(result)
                # Count parent field usage
                has_parent = 'parent' in text
                parent_count = text.count('"parent"')
                obj_count = text.count('"isOnFloor"') + text.count('"isHangingOnWall"')
                print(f"  Done: {elapsed:.1f}s, parent_mentions={parent_count}, objects_approx={obj_count}")
                print(f"  Preview: {text[:300]}")
                return {"model": model, "elapsed": elapsed, "parent_count": parent_count, "obj_count": obj_count, "has_parent": has_parent}
            elif status in ('failed', 'error'):
                print(f"  Failed: {st}")
                return None
        
        print(f"  Timeout after {timeout}s")
        return None
    except Exception as e:
        print(f"  Exception: {e}")
        return None

# Test all
print(f"Prompt length: {len(prompt)} chars")
print(f"Image size: {len(img_b64)//1024} KB")
results = {}
for name, model in [("GPT-5.4", "gpt-5.4"), ("GPT-5.2", "gpt-5.2")]:
    r = test_gpt_call(name, model)
    if r:
        results[name] = r

print("\n=== SUMMARY ===")
for name, r in results.items():
    print(f"{name}: {r['elapsed']:.0f}s, parent={r['parent_count']}, objects≈{r['obj_count']}, has_parent={r['has_parent']}")
