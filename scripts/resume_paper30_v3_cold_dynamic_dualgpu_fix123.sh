#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Lumenarium"

python="$HOME/.venvs/lumenarium-py311/bin/python"
root="$HOME/Lumenarium/a10_reusable_results/paper30"
manifest="$root/manifest.txt"
version=v3_cold_paper30_fix123
audit="$root/sceneba_audit/v5_sceneproof_fast_visual_paper30_fix121"
config="$audit/config_v3_cold_fix123.yaml"
log_root="$HOME/Lumenarium/logs/paper30_v3_cold_fix123"
queue="$log_root/dynamic_queue"
mkdir -p "$queue/claims" "$queue/done"

run_worker() {
  local gpu="$1" scene result image claim free started ended elapsed rc status batch attempt log backup
  local runtime="$log_root/runtime_gpu${gpu}.jsonl"
  touch "$runtime"
  while IFS= read -r scene || test -n "$scene"; do
    scene="${scene%$'\r'}"; test -n "$scene" || continue
    result="$root/${scene}_${version}_result"
    if compgen -G "$result/S4_layout_refinement/*_placement_info_s4.json" >/dev/null; then
      mkdir -p "$queue/done/$scene"; continue
    fi
    claim="$queue/claims/$scene"
    # Atomic, persistent scene claim. Never clear claims globally: another
    # coordinator or a surviving worker may still own the scene.
    if test -s "$claim/owner_pid"; then
      owner="$(cat "$claim/owner_pid" 2>/dev/null || true)"
      if test -n "$owner" && ! kill -0 "$owner" 2>/dev/null; then
        rm -rf -- "$claim"
      fi
    fi
    mkdir "$claim" 2>/dev/null || continue
    printf '%s\n' "$$" > "$claim/owner_pid"
    if compgen -G "$result/S4_layout_refinement/*_placement_info_s4.json" >/dev/null; then
      mkdir -p "$queue/done/$scene"; continue
    fi
    image="demo/${scene}_${version}.png"
    test -s "$image" || cp "demo/${scene}_v1.png" "$image"
    test -e "$result" && mv "$result" "${result}.partial_dynamic_$(date +%s)"
    while true; do
      free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
      test -n "$free" && test "$free" -ge 20000 && break
      echo "WAIT_GPU scene=$scene gpu=$gpu free=${free:-unknown}MiB $(date)"; sleep 30
    done
    started="$(date +%s%N)"; rc=1; attempt=0
    # batch=4 retry once for transient HTTP/S4 failures, then batch=2 for OOM.
    for batch in 4 4 2; do
      attempt=$((attempt + 1))
      test -e "$result" && mv "$result" "${result}.attempt${attempt}_$(date +%s)"
      log="$log_root/${scene}_gpu${gpu}_batch${batch}.log"
      echo "START_DYNAMIC scene=$scene gpu=$gpu batch=$batch attempt=$attempt $(date)"
      set +e
      timeout "${IMAGINARIUM_SCENE_TIMEOUT:-14400}" env \
        CUDA_VISIBLE_DEVICES="$gpu" IMAGINARIUM_S3_MAX_UNIQUE_FEATURES_PER_BATCH="$batch" \
        IMAGINARIUM_FLOOR_VERIFY_V2=1 IMAGINARIUM_S3_STACK_AWARE=1 \
        IMAGINARIUM_S4_STACK_AWARE=1 IMAGINARIUM_S1_LOWCAT_PASS=1 \
        IMAGINARIUM_USE_SAM3_DETECTION=1 IMAGINARIUM_PARALLEL_GPT_PROCESSES=1 \
        IMAGINARIUM_GPT_LOCK_FILE=/tmp/lumenarium_v3_fix123_gpt.lock \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
        LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
        "$python" -u run_imaginarium_I2Layout_v3.py "$image" --config "$config" --clean \
        > "$log" 2>&1
      rc=$?; set -e
      if test "$rc" -eq 0 && compgen -G "$result/S4_layout_refinement/*_placement_info_s4.json" >/dev/null; then break; fi
      if grep -aqE 'Response ended prematurely|ChunkedEncodingError|ProtocolError|Blender finished without S4 placement output|key "(wall|floor|ceiling)_[0-9]+" not found' "$log" \
         && test "$attempt" -lt 3; then
        echo "RETRY_TRANSIENT_OR_STRUCTURAL scene=$scene gpu=$gpu attempt=$attempt"; sleep 10; continue
      fi
      if test "$batch" -eq 4 && grep -aqE 'CUDA out of memory|OutOfMemoryError' "$log"; then
        echo "RETRY_LOW_MEMORY scene=$scene gpu=$gpu batch=4->2"; continue
      fi
      break
    done
    ended="$(date +%s%N)"; elapsed="$($python -c "print(($ended-$started)/1e9)")"
    status=fail
    test "$rc" -eq 0 && compgen -G "$result/S4_layout_refinement/*_placement_info_s4.json" >/dev/null && status=ok
    printf '{"scene":"%s","gpu":%s,"elapsed_seconds":%.6f,"status":"%s","return_code":%s,"successful_batch":%s,"attempts":%s}\n' \
      "$scene" "$gpu" "$elapsed" "$status" "$rc" "$batch" "$attempt" >> "$runtime"
    echo "DONE_DYNAMIC scene=$scene gpu=$gpu batch=$batch elapsed=$elapsed status=$status"
    if test "$status" = ok; then
      mkdir -p "$queue/done/$scene"
      rm -rf -- "$claim"
    else
      rm -rf -- "$claim"
      echo "FAILED_SCENE_CONTINUING scene=$scene gpu=$gpu"
    fi
  done < "$manifest"
}

run_worker 0 > "$log_root/dynamic_gpu0.log" 2>&1 & p0=$!
run_worker 1 > "$log_root/dynamic_gpu1.log" 2>&1 & p1=$!
echo "DYNAMIC_GPU0_PID=$p0"
echo "DYNAMIC_GPU1_PID=$p1"
status=0; wait "$p0" || status=1; wait "$p1" || status=1
count="$(find "$root" -path '*_v3_cold_paper30_fix123_result/S4_layout_refinement/*_placement_info_s4.json' -type f | wc -l)"
echo "V3_DYNAMIC_FINISHED completed=$count/30 status=$status"
test "$status" -eq 0 && test "$count" -eq 30
