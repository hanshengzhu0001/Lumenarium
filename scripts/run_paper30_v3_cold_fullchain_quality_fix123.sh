#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
python="$HOME/.venvs/lumenarium-py311/bin/python"
root="$HOME/Lumenarium/a10_reusable_results/paper30"
manifest="$root/manifest.txt"
version=v3_cold_paper30_fix123
target=v5_sceneproof_fast_visual_paper30_fix121
audit="$root/sceneba_audit/$target"
log_root="$HOME/Lumenarium/logs/paper30_v3_cold_fix123"
config="$audit/config_v3_cold_fix123.yaml"
runtime="$audit/v3_cold_fix123_runtime.json"
llm_config="config/config_a10_paper30_v4_deepsearch.yaml"
test -s "$llm_config" || llm_config="config/config_sam3_gemini.yaml"
mkdir -p "$audit/physical_native" "$log_root"

"$python" - "config/config_paper30_v3_final.yaml" "$llm_config" "$config" "$root" <<'PY'
import re, sys
from pathlib import Path
source, llm_source, target, root = map(Path, sys.argv[1:])
text = source.read_text(encoding="utf-8")
llm_text = llm_source.read_text(encoding="utf-8")
text, n = re.subn(
    r'(?m)^(\s*save_parent_folder\s*:\s*).+$',
    lambda m: f'{m.group(1)}"{root.as_posix()}"', text, count=1)
if n != 1: raise SystemExit("cannot rewrite v3 save_parent_folder")
for key in ("gpt_key", "gpt_endpoint", "gpt_model", "ground_dino_token"):
    match = re.search(rf'(?m)^\s*{key}\s*:\s*(.+)$', llm_text)
    if not match: raise SystemExit(f"missing {key} in {llm_source}")
    value = match.group(1).strip()
    if key == "gpt_key" and value == "${oc.env:GPT_API_KEY}" and not __import__("os").environ.get("GPT_API_KEY"):
        value = "${oc.env:GPT_API_KEY,}"
    text, count = re.subn(rf'(?m)^(\s*{key}\s*:\s*).+$',
                          lambda m, value=value: m.group(1) + value, text, count=1)
    if count != 1: raise SystemExit(f"cannot rewrite {key}")
text = text.replace(
    "/mnt/kevinzyz/artifacts/Imaginarium-repo/weights_cache/", "weights/")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(text, encoding="utf-8")
print(f"V3_LLM_CONFIG_SOURCE={llm_source}")
PY

list0="$(mktemp /tmp/v3_cold_fix123_gpu0_XXXXXX)"
list1="$(mktemp /tmp/v3_cold_fix123_gpu1_XXXXXX)"
trap 'rm -f -- "$list0" "$list1"' EXIT
awk 'NR % 2 == 1' "$manifest" > "$list0"
awk 'NR % 2 == 0' "$manifest" > "$list1"

run_worker() {
  local gpu="$1" list="$2" scene image result backup started ended elapsed rc status free batch attempt log
  local runtime_jsonl="$log_root/runtime_gpu${gpu}.jsonl"
  : > "$runtime_jsonl"
  while IFS= read -r scene || test -n "$scene"; do
    scene="${scene%$'\r'}"; test -n "$scene" || continue
    image="demo/${scene}_${version}.png"
    test -s "$image" || cp "demo/${scene}_v1.png" "$image"
    result="$root/${scene}_${version}_result"
    if test -e "$result"; then
      backup="${result}.pre_rerun_$(date +%s)"
      mv "$result" "$backup"
      echo "PRESERVED_PREVIOUS scene=$scene path=$backup"
    fi
    while true; do
      free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
      test -n "$free" && test "$free" -ge "${IMAGINARIUM_GPU_FREE_FLOOR_MB:-20000}" && break
      echo "WAIT_GPU scene=$scene gpu=$gpu free=${free:-unknown}MiB $(date)"
      sleep 30
    done
    started="$(date +%s%N)"
    rc=1; attempt=0
    for batch in 4 2; do
      attempt=$((attempt + 1))
      log="$log_root/${scene}_gpu${gpu}_batch${batch}.log"
      if test -e "$result"; then
        backup="${result}.attempt${attempt}_$(date +%s)"
        mv "$result" "$backup"
      fi
      echo "START_V3_COLD scene=$scene gpu=$gpu batch=$batch attempt=$attempt $(date)"
      set +e
      timeout "${IMAGINARIUM_SCENE_TIMEOUT:-14400}" env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        IMAGINARIUM_S3_MAX_UNIQUE_FEATURES_PER_BATCH="$batch" \
        IMAGINARIUM_FLOOR_VERIFY_V2=1 \
        IMAGINARIUM_S3_STACK_AWARE=1 \
        IMAGINARIUM_S4_STACK_AWARE=1 \
        IMAGINARIUM_S1_LOWCAT_PASS=1 \
        IMAGINARIUM_USE_SAM3_DETECTION=1 \
        IMAGINARIUM_PARALLEL_GPT_PROCESSES=1 \
        IMAGINARIUM_GPT_LOCK_FILE=/tmp/lumenarium_v3_fix123_gpt.lock \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONUNBUFFERED=1 \
        LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
        "$python" -u run_imaginarium_I2Layout_v3.py "$image" \
        --config "$config" --clean > "$log" 2>&1
      rc=$?
      set -e
      cp "$log" "$log_root/${scene}_gpu${gpu}.log"
      if test "$rc" -eq 0 && \
          compgen -G "$result/S4_layout_refinement/*_placement_info_s4.json" >/dev/null; then
        break
      fi
      if test "$batch" -eq 4 && grep -qE 'CUDA out of memory|OutOfMemoryError' "$log"; then
        echo "RETRY_LOW_MEMORY scene=$scene gpu=$gpu batch=4->2"
        continue
      fi
      break
    done
    ended="$(date +%s%N)"
    elapsed="$($python -c "print(($ended-$started)/1e9)")"
    status=fail
    compgen -G "$result/S4_layout_refinement/*_placement_info_s4.json" >/dev/null \
      && test "$rc" -eq 0 && status=ok
    printf '{"scene":"%s","gpu":%s,"elapsed_seconds":%.6f,"status":"%s","return_code":%s,"successful_batch":%s,"attempts":%s}\n' \
      "$scene" "$gpu" "$elapsed" "$status" "$rc" "$batch" "$attempt" >> "$runtime_jsonl"
    echo "DONE_V3_COLD scene=$scene gpu=$gpu batch=$batch attempts=$attempt elapsed=$elapsed status=$status"
    test "$status" = ok || return 1
  done < "$list"
}

full_started="$(date +%s%N)"
run_worker 0 "$list0" > "$log_root/gpu0.log" 2>&1 & p0=$!
run_worker 1 "$list1" > "$log_root/gpu1.log" 2>&1 & p1=$!
echo "V3_COLD_GPU0_PID=$p0"
echo "V3_COLD_GPU1_PID=$p1"
status=0; wait "$p0" || status=1; wait "$p1" || status=1
full_ended="$(date +%s%N)"
test "$status" -eq 0 || { echo "V3_COLD_STOP reason=worker_failure"; exit 1; }

"$python" - "$runtime" "$version" "$full_started" "$full_ended" \
  "$log_root/runtime_gpu0.jsonl" "$log_root/runtime_gpu1.jsonl" <<'PY'
import json, statistics, sys
from pathlib import Path
out, version, started, ended, *paths = sys.argv[1:]
rows = []
for path in paths:
    rows += [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
values = [x["elapsed_seconds"] for x in rows]
ordered = sorted(values)
p90 = ordered[min(len(ordered)-1, int(0.9 * len(ordered)))]
record = {
    "schema_version": "lumenarium_fullchain_runtime_v1",
    "version": version, "scenes": len(rows), "a10_devices": 2,
    "wall_clock_seconds": (int(ended)-int(started))/1e9,
    "gpu_total_seconds": sum(values),
    "mean_scene_seconds": statistics.mean(values),
    "median_scene_seconds": statistics.median(values),
    "p90_scene_seconds": p90,
}
Path(out).write_text(json.dumps(record, indent=2) + "\n")
print("V3_COLD_RUNTIME=" + json.dumps(record, sort_keys=True))
PY

versions="v1,$version,v4_deepsearch,$target"
labels="v1,v3,v4-deepsearch,v5-fast"
"$python" eval_gt_metrics.py --saved-results "$root" --scenes "$manifest" \
  --versions "$versions" --min-visible-mask-area 8000 --min-visible-bbox-size 0 \
  --batch-logs logs --metrics-out "$audit/gt_8000_with_v3_fix123.json" \
  --manifest-out "$audit/gt_manifest_8000_with_v3_fix123.json"

"$python" eval_physical_realizability.py --saved-results "$root" \
  --scenes "$manifest" --versions "$version" --geometry-version "$version" \
  --baseline-version "$version" --collision-policy legacy \
  --metrics-out "$audit/physical_native/$version.json" \
  --scene-csv "$audit/physical_native/$version.scenes.csv" \
  --object-csv "$audit/physical_native/$version.objects.csv" \
  --report-out "$audit/physical_native/$version.txt"

"$python" sceneproof_cross_version_quality_dashboard.py \
  --gt "$audit/gt_8000_with_v3_fix123.json" \
  --physical-dir "$audit/physical_native" \
  --versions "$versions" --labels "$labels" --runtime-json "$runtime" \
  --out-json "$audit/cross_version_quality_with_v3_fix123.json" \
  --out-csv "$audit/cross_version_quality_with_v3_fix123.csv" \
  --out-txt "$audit/cross_version_quality_with_v3_fix123.txt"

echo "V3_COLD_FIX123_FINISHED status=0"
echo "V3_COLD_RUNTIME=$runtime"
echo "V3_COLD_QUALITY=$audit/cross_version_quality_with_v3_fix123.json"
