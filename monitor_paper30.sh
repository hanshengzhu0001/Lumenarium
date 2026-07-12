#!/bin/bash
# Continuous monitor for paper30 Gemini + GPT runs
# Reports progress, detects stalls, auto-restarts failures

cd /ssd/kevinzyz/imaginarium/Imaginarium-repo
PROJECT_DIR="/ssd/kevinzyz/imaginarium/Imaginarium-repo"
BATCH_LOG_DIR="/mnt/kevinzyz/artifacts/Imaginarium-repo/batch_logs"

gem_log="${BATCH_LOG_DIR}/synmap_paper30_gemini.out"
gpt_log="${BATCH_LOG_DIR}/synmap_paper30_gpt54.out"
SCENE_LIST="eval_sample_paper30_v3.txt"

log() { echo "[$(date +%H:%M:%S)] $*"; }

round=0
while true; do
    round=$((round+1))
    log "=== Round $round ==="

    # ── Gemini status ──
    gem_ok=0; gem_fail=0
    if [ -f "$gem_log" ]; then
        gem_ok=$(grep -c "✅.*ok" "$gem_log" 2>/dev/null || echo 0)
        gem_fail=$(grep -c "❌.*fail" "$gem_log" 2>/dev/null || echo 0)
    fi
    gem_ok=${gem_ok//[^0-9]/}; gem_fail=${gem_fail//[^0-9]/}
    gem_ok=${gem_ok:-0}; gem_fail=${gem_fail:-0}
    gem_total=$((gem_ok + gem_fail))
    gem_hung=false; gem_age=0
    if [ -f "$gem_log" ]; then
        gem_age=$(( $(date +%s) - $(stat -c %Y "$gem_log" 2>/dev/null || echo "$(date +%s)") ))
        if [ $gem_age -gt 600 ] && [ $gem_total -lt 30 ]; then
            gem_hung=true
        fi
    fi
    log "  Gemini: ✅$gem_ok ❌$gem_fail total=$gem_total/30 hung=$gem_hung (log age=${gem_age}s)"

    # ── GPT status ──
    gpt_ok=0; gpt_fail=0
    if [ -f "$gpt_log" ]; then
        gpt_ok=$(grep -c "✅.*ok" "$gpt_log" 2>/dev/null || echo 0)
        gpt_fail=$(grep -c "❌.*fail" "$gpt_log" 2>/dev/null || echo 0)
    fi
    gpt_ok=${gpt_ok//[^0-9]/}; gpt_fail=${gpt_fail//[^0-9]/}
    gpt_ok=${gpt_ok:-0}; gpt_fail=${gpt_fail:-0}
    gpt_total=$((gpt_ok + gpt_fail))
    gpt_hung=false; gpt_age=0
    if [ -f "$gpt_log" ]; then
        gpt_age=$(( $(date +%s) - $(stat -c %Y "$gpt_log" 2>/dev/null || echo "$(date +%s)") ))
        if [ $gpt_age -gt 600 ] && [ $gpt_total -lt 30 ]; then
            gpt_hung=true
        fi
    fi
    log "  GPT54: ✅$gpt_ok ❌$gpt_fail total=$gpt_total/30 hung=$gpt_hung (log age=${gpt_age}s)"

    # ── GPU memory ──
    gpu_info=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | head -4)
    log "  GPU: $(echo $gpu_info | tr '\n' ' ')"

    # ── Auto-restart hung runs ──
    if [ "$gem_hung" = true ] && [ $gpt_total -lt 30 ]; then
        log "  ⚠️ Gemini hung! Restarting..."
        pkill -f "synmap_paper30_gemini" 2>/dev/null
        sleep 3
        IMAGINARIUM_GPU_IDS=0,1 nohup /ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python batch_eval.py \
            --run-name synmap_paper30_gemini --config-template config/config_gemini_sync.yaml \
            --scenes "$SCENE_LIST" --v3-only --gpu-count 2 --timeout 7200 --gpt-max-wait 900 --gpt-max-retries 10 \
            > "$gem_log" 2>&1 &
        log "  Gemini restarted (pid=$!)"
    fi

    if [ "$gpt_hung" = true ] && [ $gpt_total -lt 30 ]; then
        log "  ⚠️ GPT54 hung! Restarting..."
        pkill -f "synmap_paper30_gpt54" 2>/dev/null
        sleep 3
        IMAGINARIUM_GPU_IDS=2,3 nohup /ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python batch_eval.py \
            --run-name synmap_paper30_gpt54 --config-template config/config_gpt54_sync.yaml \
            --scenes "$SCENE_LIST" --v3-only --gpu-count 2 --timeout 7200 --gpt-max-wait 900 --gpt-max-retries 10 \
            > "$gpt_log" 2>&1 &
        log "  GPT54 restarted (pid=$!)"
    fi

    # ── Check if v3 complete for both ──
    if [ $gem_total -ge 30 ] && [ $gpt_total -ge 30 ]; then
        log "🎉 V3 complete for both Gemini and GPT!"
        gem_final=$(grep "完成:" "$gem_log" 2>/dev/null | tail -1)
        gpt_final=$(grep "完成:" "$gpt_log" 2>/dev/null | tail -1)
        log "Gemini: $gem_final"
        log "GPT54: $gpt_final"

        # Retry failures
        gem_fail_count=$(grep -c "❌.*fail" "$gem_log" 2>/dev/null || echo 0)
        if [ "$gem_fail_count" -gt 0 ]; then
            log "Retrying $gem_fail_count Gemini failures..."
            grep "❌.*fail" "$gem_log" | grep -oP '\[v3\] \K\w+' > /tmp/gem_fail_scenes.txt
            IMAGINARIUM_GPU_IDS=0,1 nohup /ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python batch_eval.py \
                --run-name synmap_paper30_gemini_retry --config-template config/config_gemini_sync.yaml \
                --scenes /tmp/gem_fail_scenes.txt --v3-only --gpu-count 2 --timeout 7200 --gpt-max-wait 900 --gpt-max-retries 10 \
                > "${BATCH_LOG_DIR}/synmap_paper30_gemini_retry.out" 2>&1 &
            log "Gemini retry launched"
        fi

        gpt_fail_count=$(grep -c "❌.*fail" "$gpt_log" 2>/dev/null || echo 0)
        if [ "$gpt_fail_count" -gt 0 ]; then
            log "Retrying $gpt_fail_count GPT failures..."
            grep "❌.*fail" "$gpt_log" | grep -oP '\[v3\] \K\w+' > /tmp/gpt_fail_scenes.txt
            IMAGINARIUM_GPU_IDS=2,3 nohup /ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python batch_eval.py \
                --run-name synmap_paper30_gpt54_retry --config-template config/config_gpt54_sync.yaml \
                --scenes /tmp/gpt_fail_scenes.txt --v3-only --gpu-count 2 --timeout 7200 --gpt-max-wait 900 --gpt-max-retries 10 \
                > "${BATCH_LOG_DIR}/synmap_paper30_gpt54_retry.out" 2>&1 &
            log "GPT retry launched"
        fi

        # Move to v1
        log "🚀 Launching V1 for Gemini..."
        IMAGINARIUM_GPU_IDS=0,1 nohup /ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python batch_eval.py \
            --run-name synmap_paper30_gemini_v1 --config-template config/config_gemini_sync.yaml \
            --scenes "$SCENE_LIST" --gpu-count 2 --timeout 7200 --gpt-max-wait 900 --gpt-max-retries 10 \
            > "${BATCH_LOG_DIR}/synmap_paper30_gemini_v1.out" 2>&1 &
        log "Gemini v1 launched"

        log "🚀 Launching V1 for GPT-5.4..."
        IMAGINARIUM_GPU_IDS=2,3 nohup /ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python batch_eval.py \
            --run-name synmap_paper30_gpt54_v1 --config-template config/config_gpt54_sync.yaml \
            --scenes "$SCENE_LIST" --gpu-count 2 --timeout 7200 --gpt-max-wait 900 --gpt-max-retries 10 \
            > "${BATCH_LOG_DIR}/synmap_paper30_gpt54_v1.out" 2>&1 &
        log "GPT-5.4 v1 launched"

        break  # exit monitoring after launching v1
    fi

    sleep 120  # check every 2 minutes
done

log "=== Monitor phase 1 complete (v3 done, v1 launched) ==="
log "Starting phase 2: wait for v1 completion..."

while true; do
    round=$((round+1))
    log "=== Phase2 Round $round ==="
    
    gem_v1_ok=$(grep -c "✅.*ok" "${BATCH_LOG_DIR}/synmap_paper30_gemini_v1.out" 2>/dev/null || echo 0)
    gpt_v1_ok=$(grep -c "✅.*ok" "${BATCH_LOG_DIR}/synmap_paper30_gpt54_v1.out" 2>/dev/null || echo 0)
    gem_v1_fail=$(grep -c "❌.*fail" "${BATCH_LOG_DIR}/synmap_paper30_gemini_v1.out" 2>/dev/null || echo 0)
    gpt_v1_fail=$(grep -c "❌.*fail" "${BATCH_LOG_DIR}/synmap_paper30_gpt54_v1.out" 2>/dev/null || echo 0)
    
    log "  Gemini v1: ✅$gem_v1_ok ❌$gem_v1_fail | GPT-5.4 v1: ✅$gpt_v1_ok ❌$gpt_v1_fail"
    
    gem_v1_total=$((gem_v1_ok + gem_v1_fail))
    gpt_v1_total=$((gpt_v1_ok + gpt_v1_fail))
    
    if [ $gem_v1_total -ge 30 ] && [ $gpt_v1_total -ge 30 ]; then
        log "🎉 V1 complete for both!"

        # Retry v1 failures
        gem_v1_need_retry=$(grep -c "❌.*fail" "${BATCH_LOG_DIR}/synmap_paper30_gemini_v1.out" 2>/dev/null || echo 0)
        gpt_v1_need_retry=$(grep -c "❌.*fail" "${BATCH_LOG_DIR}/synmap_paper30_gpt54_v1.out" 2>/dev/null || echo 0)
        
        if [ "$gem_v1_need_retry" -gt 0 ]; then
            grep "❌.*fail" "${BATCH_LOG_DIR}/synmap_paper30_gemini_v1.out" | grep -oP '\[v1\] \K\w+' > /tmp/gem_v1_fail.txt
            IMAGINARIUM_GPU_IDS=0,1 nohup /ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python batch_eval.py \
                --run-name synmap_paper30_gemini_v1_retry --config-template config/config_gemini_sync.yaml \
                --scenes /tmp/gem_v1_fail.txt --gpu-count 2 --timeout 7200 --gpt-max-wait 900 --gpt-max-retries 10 \
                > "${BATCH_LOG_DIR}/synmap_paper30_gemini_v1_retry.out" 2>&1 &
        fi
        if [ "$gpt_v1_need_retry" -gt 0 ]; then
            grep "❌.*fail" "${BATCH_LOG_DIR}/synmap_paper30_gpt54_v1.out" | grep -oP '\[v1\] \K\w+' > /tmp/gpt_v1_fail.txt
            IMAGINARIUM_GPU_IDS=2,3 nohup /ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python batch_eval.py \
                --run-name synmap_paper30_gpt54_v1_retry --config-template config/config_gpt54_sync.yaml \
                --scenes /tmp/gpt_v1_fail.txt --gpu-count 2 --timeout 7200 --gpt-max-wait 900 --gpt-max-retries 10 \
                > "${BATCH_LOG_DIR}/synmap_paper30_gpt54_v1_retry.out" 2>&1 &
        fi
        
        # ─── ALL DONE → EVAL + KEEP GPUs BUSY ───
        log "🚀 ALL DONE. Running eval..."
        /ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python eval_gt_metrics.py \
            --saved-results saved_results_synmap_paper30_gemini --versions v3 \
            --metrics-out eval_paper30_gemini_v3.json 2>&1 | tail -5 >> "${BATCH_LOG_DIR}/paper30_final.log"
        /ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python eval_gt_metrics.py \
            --saved-results saved_results_synmap_paper30_gpt54 --versions v3 \
            --metrics-out eval_paper30_gpt54_v3.json 2>&1 | tail -5 >> "${BATCH_LOG_DIR}/paper30_final.log"

        log "Eval complete. Keeping GPUs occupied with occupy_gpus.py..."
        # Launch GPU placeholder to keep GPUs until morning
        pkill -f "occupy_gpus" 2>/dev/null
        CUDA_VISIBLE_DEVICES=0,1 nohup /ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python /home/kevinzyz/occupy_gpus.py \
            > /tmp/occupy_01.out 2>&1 &
        CUDA_VISIBLE_DEVICES=2,3 nohup /ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python /home/kevinzyz/occupy_gpus.py \
            > /tmp/occupy_23.out 2>&1 &
        log "GPU occupiers launched. Exiting."
        break
    fi
    sleep 120
done

echo "[$(date)] MONITOR COMPLETE" >> "${BATCH_LOG_DIR}/paper30_final.log"
