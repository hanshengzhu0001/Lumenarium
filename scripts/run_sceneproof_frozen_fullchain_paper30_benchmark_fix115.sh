#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
python="${IMAGINARIUM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
base_manifest="${SCENEPROOF_FULLCHAIN_MANIFEST:-a10_reusable_results/paper30/manifest.txt}"
base_config="${SCENEPROOF_FULLCHAIN_CONFIG:-config/config_a10_paper30_v4_deepsearch.yaml}"
run_id="${SCENEPROOF_FULLCHAIN_RUN_ID:-fix124_v5_fast_cold_paper30}"
root="${SCENEPROOF_FULLCHAIN_RESULTS_ROOT:-$HOME/Lumenarium/a10_reusable_results/$run_id}"
log_root="${SCENEPROOF_FULLCHAIN_LOG_ROOT:-$HOME/Lumenarium/logs/$run_id}"
manifest="$root/manifest.txt"
config="$root/config_frozen.yaml"
source="v4_deepsearch"
control="v5_sceneproof_fix43_smooth_${run_id}"
guarded="v5_sceneproof_collision_partial_commit_${run_id}"
baseline="v5_sceneproof_collision_partial_commit_certified_${run_id}"
target="v5_sceneproof_vertical_support_visual_${run_id}"
deepsearch_workers="${SCENEPROOF_DEEPSEARCH_WORKERS_PER_GPU:-4}"
deepsearch_url="${OMNIVERSE_DEEPSEARCH_URL:-${SCENEPROOF_DEEPSEARCH_URL:-https://miller-unshapeable-melany.ngrok-free.dev/search}}"
scene_timeout="${IMAGINARIUM_SCENE_TIMEOUT:-14400}"
gpu_floor="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}"

test -x "$python" || { echo "Missing Python: $python" >&2; exit 2; }
test -s "$base_manifest" || { echo "Missing manifest: $base_manifest" >&2; exit 2; }
test -s "$base_config" || { echo "Missing config: $base_config" >&2; exit 2; }
mkdir -p "$root" "$log_root/s03"
probe="$(curl --fail --show-error --silent --max-time 30 \
  --request POST --header 'Content-Type: application/json' \
  --data '{"description":"house","limit":2}' "$deepsearch_url")" || {
  echo "FULLCHAIN_STOP ngrok DeepSearch unavailable: $deepsearch_url" >&2
  exit 3
}
printf '%s' "$probe" | "$python" -c \
  'import json,sys; x=json.load(sys.stdin); assert isinstance(x,list) and x and isinstance(x[0].get("url"),str)' || {
  echo "FULLCHAIN_STOP ngrok DeepSearch returned invalid JSON" >&2
  exit 3
}
echo "FULLCHAIN_DEEPSEARCH_URL=$deepsearch_url WORKERS_PER_GPU=$deepsearch_workers TOTAL_WORKERS=$((deepsearch_workers * 2))"
cp "$base_manifest" "$manifest"

"$python" - "$base_config" "$config" "$root" <<'PY'
import re, sys
from pathlib import Path
source, target, root = map(Path, sys.argv[1:])
text = source.read_text(encoding="utf-8")
text, count = re.subn(
    r'(?m)^(\s*save_parent_folder\s*:\s*).+$',
    lambda match: f'{match.group(1)}"{root.as_posix()}"',
    text,
    count=1,
)
if count != 1:
    raise SystemExit("Could not freeze save_parent_folder in config")
target.write_text(text, encoding="utf-8")
PY

list0="$(mktemp /tmp/sceneproof_fix115_gpu0_XXXXXX)"
list1="$(mktemp /tmp/sceneproof_fix115_gpu1_XXXXXX)"
trap 'rm -f -- "$list0" "$list1"' EXIT
awk 'NR % 2 == 1' "$manifest" > "$list0"
awk 'NR % 2 == 0' "$manifest" > "$list1"

now_ns() { date +%s%N; }
elapsed_s() { "$python" -c "print(($(now_ns)-$1)/1e9)"; }

run_s03_worker() {
  local gpu="$1" list="$2" scene image result placement started ended elapsed rc status batch attempt scene_log
  local runtime="$log_root/s03/runtime_gpu${gpu}.jsonl"
  touch "$runtime"
  while IFS= read -r scene || test -n "$scene"; do
    scene="${scene%$'\r'}"; test -n "$scene" || continue
    image="demo/${scene}_${source}.png"
    result="$root/${scene}_${source}_result"
    placement="$(find "$result/S3_pose_inference" -maxdepth 1 -type f -name '*_placement_info.json' -print -quit 2>/dev/null || true)"
    if test -s "$placement"; then
      echo "CACHED_S03 scene=$scene gpu=$gpu"
      continue
    fi
    test -s "$image" || { echo "FAIL_S03 scene=$scene reason=missing_image path=$image"; return 1; }
    started="$(now_ns)"; rc=1; attempt=0; batch=16
    # Optimistically start at 16 on A10. Large scenes may exceed memory, so
    # fall back deterministically through 8, 4, and 2. A transient network
    # failure retries the same batch once without changing the memory policy.
    batches=(16 8 4 2)
    batch_index=0
    transient_retry=0
    while test "$batch_index" -lt "${#batches[@]}"; do
      batch="${batches[$batch_index]}"
      attempt=$((attempt + 1))
      scene_log="$log_root/s03/${scene}_gpu${gpu}_attempt${attempt}_batch${batch}.log"
      echo "START_S03 scene=$scene gpu=$gpu workers=$deepsearch_workers batch=$batch attempt=$attempt $(date)"
      set +e
      timeout "$scene_timeout" env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        IMAGINARIUM_STOP_AFTER_STAGE=S3 \
        IMAGINARIUM_S3_MAX_UNIQUE_FEATURES_PER_BATCH="$batch" \
        IMAGINARIUM_PARALLEL_GPT_PROCESSES=1 \
        IMAGINARIUM_GPT_LOCK_FILE=/tmp/lumenarium_fix124_gemini.lock \
        OMNIVERSE_DEEPSEARCH_URL="$deepsearch_url" \
        OMNIVERSE_DEEPSEARCH_WORKERS="$deepsearch_workers" \
        OMNIVERSE_DEEPSEARCH_MAX_ATTEMPTS=8 \
        OMNIVERSE_DEEPSEARCH_TIMEOUT=120 \
        OMNIVERSE_DEEPSEARCH_RETRY_DELAY=2 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONUNBUFFERED=1 \
        LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
        "$python" -u run_imaginarium_I2Layout_v4_deepsearch.py "$image" --config "$config" \
        > "$scene_log" 2>&1
      rc=$?
      set -e
      placement="$(find "$result/S3_pose_inference" -maxdepth 1 -type f -name '*_placement_info.json' -print -quit 2>/dev/null || true)"
      test "$rc" -eq 0 && test -s "$placement" && break
      if grep -aqE 'Response ended prematurely|ChunkedEncodingError|ProtocolError|ConnectionError|Read timed out|RemoteDisconnected' "$scene_log" \
         && test "$transient_retry" -lt 1; then
        transient_retry=$((transient_retry + 1))
        echo "RETRY_S03_TRANSIENT scene=$scene gpu=$gpu batch=$batch attempt=$attempt"; sleep 10; continue
      fi
      if grep -aqE 'CUDA out of memory|OutOfMemoryError' "$scene_log" && test "$batch" -gt 2; then
        echo "RETRY_S03_OOM scene=$scene gpu=$gpu batch=$batch next=${batches[$((batch_index + 1))]}"
        batch_index=$((batch_index + 1)); transient_retry=0; sleep 10; continue
      fi
      break
    done
    ended="$(now_ns)"
    elapsed="$($python -c "print(($ended-$started)/1e9)")"
    placement="$(find "$result/S3_pose_inference" -maxdepth 1 -type f -name '*_placement_info.json' -print -quit 2>/dev/null || true)"
    status=fail; test "$rc" -eq 0 && test -s "$placement" && status=ok
    printf '{"scene":"%s","gpu":%s,"elapsed_seconds":%.6f,"status":"%s","return_code":%s,"attempts":%s,"successful_batch":%s}\n' \
      "$scene" "$gpu" "$elapsed" "$status" "$rc" "$attempt" "$batch" >> "$runtime"
    echo "DONE_S03 scene=$scene gpu=$gpu elapsed=$elapsed status=$status attempts=$attempt batch=$batch"
    test "$status" = ok || echo "$scene" >> "$log_root/s03/failed_scenes.txt"
  done < "$list"
}

full_started="$(now_ns)"
s03_started="$(now_ns)"
run_s03_worker 0 "$list0" > "$log_root/s03/gpu0.log" 2>&1 & p0=$!
run_s03_worker 1 "$list1" > "$log_root/s03/gpu1.log" 2>&1 & p1=$!
echo "S03_GPU0_PID=$p0"
echo "S03_GPU1_PID=$p1"
echo "S03_LOGS=$log_root/s03/gpu0.log,$log_root/s03/gpu1.log"
status=0; wait "$p0" || status=1; wait "$p1" || status=1
missing_s03="$($python - "$root" "$manifest" "$source" <<'PY'
import sys
from pathlib import Path
root, manifest, version = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
scenes=[x.strip() for x in manifest.read_text().splitlines() if x.strip()]
missing=[]
for scene in scenes:
    folder=root/f"{scene}_{version}_result"/"S3_pose_inference"
    if not any(folder.glob("*_placement_info.json")): missing.append(scene)
print(",".join(missing))
PY
)"
test -z "$missing_s03" || { echo "FULLCHAIN_STOP stage=S03 missing=$missing_s03"; exit 1; }
s03_ended="$(now_ns)"

scenelm_started="$(now_ns)"
env \
  SCENEPROOF_RESULTS_ROOT="$root" \
  SCENEPROOF_SOURCE_VERSION="$source" \
  SCENEPROOF_FIX43_SOURCE_MANIFEST="$manifest" \
  SCENEPROOF_FIX43_MANIFEST="$root/fix61_manifest.txt" \
  SCENEPROOF_FIX43_EXPECTED_SCENES=30 \
  SCENEPROOF_FIX43_MINIMUM_NONZERO_SCENES=18 \
  SCENEPROOF_FIX43_CONTROL_VERSION="$control" \
  SCENEPROOF_FIX43_GUARDED_VERSION="$guarded" \
  SCENEPROOF_FIX43_CERTIFIED_VERSION="$baseline" \
  SCENEPROOF_FIX43_SKIP_RENDER=1 \
  SCENEPROOF_FIX43_SKIP_FORMAL_EVAL=1 \
  IMAGINARIUM_GPU_FREE_FLOOR_MB="$gpu_floor" \
  bash scripts/run_sceneproof_fix43_inloop_fullstack_smoke5_fix56.sh \
  > "$log_root/scenelm_sceneproof_fix61.log" 2>&1
scenelm_ended="$(now_ns)"

proof_started="$(now_ns)"
env \
  SCENEPROOF_RESULTS_ROOT="$root" \
  SCENEPROOF_MANIFEST="$manifest" \
  SCENEPROOF_FIX114_SOURCE_VERSION="$source" \
  SCENEPROOF_FIX114_BASELINE_VERSION="$baseline" \
  SCENEPROOF_FIX114_TARGET_VERSION="$target" \
  SCENEPROOF_FIX114_SKIP_RENDER=1 \
  SCENEPROOF_FIX114_SKIP_COMPARISON_ARCHIVE=1 \
  IMAGINARIUM_GPU_FREE_FLOOR_MB="$gpu_floor" \
  bash scripts/run_sceneproof_vertical_support_final_paper30_fix114.sh \
  > "$log_root/sceneproof_fix114.log" 2>&1
proof_ended="$(now_ns)"

render_started="$(now_ns)"
env \
  SCENEPROOF_RESULTS_ROOT="$root" \
  SCENEPROOF_MANIFEST="$manifest" \
  SCENEPROOF_RENDER_SOURCE_VERSION="$source" \
  SCENEPROOF_CERTIFIED_VERSION="$target" \
  SCENEPROOF_RENDER_LOG_ROOT="$log_root/final_render" \
  SCENEPROOF_RENDER_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}" \
  IMAGINARIUM_GPU_FREE_FLOOR_MB="$gpu_floor" \
  bash scripts/render_sceneproof_certified_paper30.sh \
  > "$log_root/final_render.log" 2>&1
render_ended="$(now_ns)"
full_ended="$(now_ns)"

audit="$root/sceneba_audit/$target"
"$python" - "$root/benchmark.json" "$run_id" "$root" "$source" "$baseline" "$target" \
  "$s03_started" "$s03_ended" "$scenelm_started" "$scenelm_ended" \
  "$proof_started" "$proof_ended" "$render_started" "$render_ended" \
  "$full_started" "$full_ended" "$audit/final_eval.json" <<'PY'
import json, sys
from pathlib import Path
(out, run_id, root, source, baseline, target, *values, final_eval) = sys.argv[1:]
numbers = list(map(int, values))
names = ["s0_s3_deepsearch", "scenelm_fix61", "sceneproof_fix114", "final_render", "end_to_end"]
timing = {}
for index, name in enumerate(names):
    start, end = numbers[index * 2:index * 2 + 2]
    timing[name] = (end - start) / 1e9
record = {
    "schema_version": "sceneproof_frozen_fullchain_benchmark_v1",
    "run_id": run_id,
    "results_root": root,
    "source_version": source,
    "fix61_version": baseline,
    "fix114_version": target,
    "scenes": 30,
    "a10_devices": 2,
    "timing_seconds": timing,
    "timing_without_final_render_seconds": timing["end_to_end"] - timing["final_render"],
    "two_a10_wall_hours": timing["end_to_end"] / 3600.0,
    "two_a10_wall_hours_without_final_render": (timing["end_to_end"] - timing["final_render"]) / 3600.0,
    "fix114_eval": json.load(open(final_eval)),
}
Path(out).write_text(json.dumps(record, indent=2) + "\n")
print(f"FULLCHAIN_SCENES=30/30 A10_DEVICES=2")
for name, seconds in timing.items(): print(f"{name.upper()}_SECONDS={seconds:.3f}")
print(f"FIX114_EVAL={final_eval}")
print(f"FULLCHAIN_BENCHMARK={out}")
PY

echo "FULLCHAIN_FINISHED status=0"
echo "FULLCHAIN_RESULTS_ROOT=$root"
echo "FULLCHAIN_BENCHMARK=$root/benchmark.json"
echo "FULLCHAIN_PHYSICAL=$audit/physical.json"
echo "FULLCHAIN_GT=$audit/gt_8000.json"
