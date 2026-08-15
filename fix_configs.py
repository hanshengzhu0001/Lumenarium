import os, glob, re

config_dir = r"C:\Users\hansenzhu\Desktop\ppt_images\Lumenarium\config"
skip = {"config-example.yaml", "config_wallsoft_full302_retry3.yaml"}

GPT_KEY_PAT = re.compile(r'gpt_key:\s*"eyJ[a-zA-Z0-9_\-\.]+"')
DINO_PAT = re.compile(r'ground_dino_token:\s*"[a-f0-9]+"')
ENDPOINT_PAT = re.compile(r'gpt_endpoint:\s*"https://lightaiapi\.lightspeed\.qq\.com/api/v1/gemini/call"')
MODEL_PAT = re.compile(r'gpt_model:\s*"gemini-3\.\d+-pro-preview"')

fixed = 0
for f in glob.glob(os.path.join(config_dir, "*.yaml")):
    name = os.path.basename(f)
    if name in skip:
        continue
    with open(f, "r", encoding="utf-8") as fh:
        content = fh.read()
    changed = False
    if GPT_KEY_PAT.search(content):
        content = GPT_KEY_PAT.sub('gpt_key: ${GPT_API_KEY}', content)
        changed = True
    if DINO_PAT.search(content):
        content = DINO_PAT.sub('ground_dino_token: ${GROUND_DINO_TOKEN}', content)
        changed = True
    if ENDPOINT_PAT.search(content):
        content = ENDPOINT_PAT.sub('gpt_endpoint: ${GPT_ENDPOINT:https://lightaiapi.lightspeed.qq.com/api/v1/gemini/call}', content)
        changed = True
    if MODEL_PAT.search(content):
        content = MODEL_PAT.sub('gpt_model: ${GPT_MODEL:gemini-3.1-pro-preview}', content)
        changed = True
    if changed:
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(content)
        fixed += 1

print(f"Fixed {fixed} files")
