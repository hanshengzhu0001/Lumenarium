#!/usr/bin/env python3
"""Monitor GPUs 2-7 and launch extra batch_eval when free."""
import subprocess, time, os, sys
from pathlib import Path

LOG = "batch_logs/paper30_v3_final.log"

def free_gpus():
    r = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader'],
                       capture_output=True, text=True)
    free = []
    for line in r.stdout.strip().split('\n'):
        idx, mem = line.split(', ')
        gpu = int(idx)
        used_mb = int(mem.split()[0])
        if gpu >= 2 and used_mb < 5000:  # GPUs 2-7 with <5GB
            free.append(gpu)
    return free

def progress():
    done = int(subprocess.getoutput(f'grep "✅" {LOG} 2>/dev/null | wc -l') or '0')
    fail = int(subprocess.getoutput(f'grep "❌" {LOG} 2>/dev/null | wc -l') or '0')
    return done, fail

print("Monitoring GPUs 2-7 for extra batch_eval...")
extra_launched = False

while True:
    free = free_gpus()
    d, f = progress()
    print(f"[{time.strftime('%H:%M:%S')}] paper30: {d}✅ {f}❌ | Free GPUs 2-7: {free}", flush=True)
    
    if len(free) >= 2 and not extra_launched and Path('eval_sample_extra20.txt').exists():
        print(f"  >>> Launching extra batch_eval on GPUs {free[:2]}!")
        os.system(f"""
            cd /ssd/kevinzyz/imaginarium/Imaginarium-repo && \
            nohup env IMAGINARIUM_GPU_IDS={free[0]},{free[1]} IMAGINARIUM_PARALLEL_GPT_PROCESSES=1 \
            IMAGINARIUM_S1_PARENT_AWARE=1 IMAGINARIUM_S3_STACK_AWARE=1 IMAGINARIUM_S4_STACK_AWARE=1 \
            IMAGINARIUM_S4_USE_TORCH_VOXELS=1 IMAGINARIUM_S4_SKIP_RENDER=1 \
            IMAGINARIUM_S1_LOWCAT_PASS=1 IMAGINARIUM_S1_RELABEL_ANONYMOUS=1 \
            IMAGINARIUM_WEIGHT_CACHE_DIR=/mnt/kevinzyz/artifacts/Imaginarium-repo/weights_cache \
            PYTHONUNBUFFERED=1 \
            /ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python batch_eval.py \
            --run-name paper30_extra --config-template config/config_gpt54_sync.yaml \
            --output-root saved_results_paper30_extra --scenes eval_sample_extra20.txt \
            --v3-only --gpu-count 2 --timeout 7200 --gpt-max-wait 900 --gpt-max-retries 10 \
            > batch_logs/paper30_extra.log 2>&1 &
        """)
        extra_launched = True
        print("  >>> Extra batch_eval launched!")
    
    if d >= 30:
        print("Paper30 complete!")
        break
    
    # Watch log for new output
    last_lines = subprocess.getoutput(f'tail -3 {LOG}')
    if '❌' in last_lines:
        print(f"  FAILURE: {last_lines.split(chr(10))[0]}", flush=True)
    
    time.sleep(30)
