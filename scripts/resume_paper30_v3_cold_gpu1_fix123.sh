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
runtime="$log_root/runtime_gpu1.jsonl"
list="$(mktemp /tmp/v3_fix123_gpu1_resume_XXXXXX)"
trap 'rm -f -- "$list"' EXIT
awk 'NR % 2 == 0' "$manifest" > "$list"
touch "$runtime"

while IFS= read -r scene || test -n "$scene"; do
  scene="${scene%$'\r'}"; test -n "$scene" || continue
  result="$root/${scene}_${version}_result"
  if compgen -G "$result/S4_layout_refinement/*_placement_info_s4.json" >/dev/null; then
    echo "CACHED_V3_GPU1 scene=$scene"
    continue
  fi
  image="demo/${scene}_${version}.png"
  test -s "$image" || cp "demo/${scene}_v1.png" "$image"
  if test -e "$result"; then
    mv "$result" "${result}.partial_before_gpu1_resume_$(date +%s)"
  fi
  while true; do
    free="$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    test -n "$free" && test "$free" -ge 20000 && break
    echo "WAIT_GPU1 scene=$scene free=${free:-unknown}MiB $(date)"; sleep 30
  done
  started="$(date +%s%N)"; rc=1; attempt=0
  for batch in 4 2; do
    attempt=$((attempt + 1))
    test -e "$result" && mv "$result" "${result}.attempt${attempt}_$(date +%s)"
    log="$log_root/${scene}_gpu1_batch${batch}.log"
    echo "START_V3_GPU1_RESUME scene=$scene batch=$batch attempt=$attempt $(date)"
    set +e
    timeout "${IMAGINARIUM_SCENE_TIMEOUT:-14400}" env \
      CUDA_VISIBLE_DEVICES=1 \
      IMAGINARIUM_S3_MAX_UNIQUE_FEATURES_PER_BATCH="$batch" \
      IMAGINARIUM_FLOOR_VERIFY_V2=1 IMAGINARIUM_S3_STACK_AWARE=1 \
      IMAGINARIUM_S4_STACK_AWARE=1 IMAGINARIUM_S1_LOWCAT_PASS=1 \
      IMAGINARIUM_USE_SAM3_DETECTION=1 IMAGINARIUM_PARALLEL_GPT_PROCESSES=1 \
      IMAGINARIUM_GPT_LOCK_FILE=/tmp/lumenarium_v3_fix123_gpt.lock \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
      LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
      "$python" -u run_imaginarium_I2Layout_v3.py "$image" --config "$config" --clean \
      > "$log" 2>&1
    rc=$?; set -e
    cp "$log" "$log_root/${scene}_gpu1.log"
    if test "$rc" -eq 0 && compgen -G "$result/S4_layout_refinement/*_placement_info_s4.json" >/dev/null; then break; fi
    if test "$batch" -eq 4 && grep -qE 'CUDA out of memory|OutOfMemoryError' "$log"; then
      echo "RETRY_LOW_MEMORY scene=$scene batch=4->2"; continue
    fi
    break
  done
  ended="$(date +%s%N)"; elapsed="$($python -c "print(($ended-$started)/1e9)")"
  status=fail
  test "$rc" -eq 0 && compgen -G "$result/S4_layout_refinement/*_placement_info_s4.json" >/dev/null && status=ok
  printf '{"scene":"%s","gpu":1,"elapsed_seconds":%.6f,"status":"%s","return_code":%s,"successful_batch":%s,"attempts":%s}\n' \
    "$scene" "$elapsed" "$status" "$rc" "$batch" "$attempt" >> "$runtime"
  echo "DONE_V3_GPU1_RESUME scene=$scene batch=$batch elapsed=$elapsed status=$status"
  test "$status" = ok || exit 1
done < "$list"

echo "V3_GPU1_RESUME_FINISHED status=0"
