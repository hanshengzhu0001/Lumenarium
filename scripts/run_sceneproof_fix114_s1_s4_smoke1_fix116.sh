#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
python="${IMAGINARIUM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
scene="${SCENEPROOF_FIX116_SCENE:-bedroom_01}"
gpu="${SCENEPROOF_FIX116_GPU:-0}"
source="v4_deepsearch"
run_id="${SCENEPROOF_FIX116_RUN_ID:-fix116_s1_s4_smoke1}"
base_root="${SCENEPROOF_FIX116_BASE_ROOT:-$HOME/Lumenarium/a10_reusable_results/paper30}"
root="${SCENEPROOF_FIX116_RESULTS_ROOT:-$HOME/Lumenarium/a10_reusable_results/$run_id}"
log_root="${SCENEPROOF_FIX116_LOG_ROOT:-$HOME/Lumenarium/logs/$run_id}"
base_config="${SCENEPROOF_FIX116_CONFIG:-config/config_a10_paper30_v4_deepsearch.yaml}"
manifest="$root/manifest.txt"
config="$root/config.yaml"
image="demo/${scene}_${source}.png"
source_result="$base_root/${scene}_${source}_result"
result="$root/${scene}_${source}_result"
control="v5_sceneproof_fix43_smooth_${run_id}"
guarded="v5_sceneproof_collision_partial_commit_${run_id}"
baseline="v5_sceneproof_collision_partial_commit_certified_${run_id}"
target="v5_sceneproof_vertical_support_visual_${run_id}"
deepsearch_url="${OMNIVERSE_DEEPSEARCH_URL:-${SCENEPROOF_DEEPSEARCH_URL:-https://miller-unshapeable-melany.ngrok-free.dev/search}}"
deepsearch_workers="${SCENEPROOF_FIX116_DEEPSEARCH_WORKERS:-8}"

test -x "$python" || { echo "Missing Python: $python" >&2; exit 2; }
test -s "$image" || { echo "Missing input: $image" >&2; exit 2; }
test -s "$base_config" || { echo "Missing config: $base_config" >&2; exit 2; }
test -d "$source_result/S0_geometry_pred_results" || {
  echo "Missing frozen S0: $source_result/S0_geometry_pred_results" >&2; exit 2;
}
mkdir -p "$root" "$log_root" "$result"
probe="$(curl --fail --show-error --silent --max-time 30 \
  --request POST --header 'Content-Type: application/json' \
  --data '{"description":"house","limit":2}' "$deepsearch_url")" || {
  echo "FIX116_STOP ngrok DeepSearch unavailable: $deepsearch_url" >&2
  exit 3
}
printf '%s' "$probe" | "$python" -c \
  'import json,sys; x=json.load(sys.stdin); assert isinstance(x,list) and x and isinstance(x[0].get("url"),str)' || {
  echo "FIX116_STOP ngrok DeepSearch returned invalid JSON" >&2
  exit 3
}
echo "FIX116_DEEPSEARCH_URL=$deepsearch_url WORKERS=$deepsearch_workers"
printf '%s\n' "$scene" > "$manifest"
if ! test -e "$result/S0_geometry_pred_results"; then
  ln -s "$(realpath "$source_result/S0_geometry_pred_results")" \
    "$result/S0_geometry_pred_results"
fi

"$python" - "$base_config" "$config" "$root" <<'PY'
import re, sys
from pathlib import Path
source, target, root = map(Path, sys.argv[1:])
text = source.read_text(encoding="utf-8")
text, count = re.subn(
    r'(?m)^(\s*save_parent_folder\s*:\s*).+$',
    lambda match: f'{match.group(1)}"{root.as_posix()}"', text, count=1,
)
if count != 1: raise SystemExit("save_parent_folder missing")
target.write_text(text, encoding="utf-8")
PY

now_ns() { date +%s%N; }
elapsed() { "$python" -c "print(($2-$1)/1e9)"; }
full_start="$(now_ns)"

s13_start="$(now_ns)"
if [[ "${SCENEPROOF_FIX116_REUSE_S13:-0}" == "1" ]] && \
   find "$result/S3_pose_inference" -maxdepth 1 -type f -name '*_placement_info.json' -print -quit 2>/dev/null | grep -q .; then
  echo "FIX116_STAGE=S1_S3 REUSED $(date)"
else
  echo "FIX116_STAGE=S1_S3 START $(date)"
  timeout "${SCENEPROOF_FIX116_S13_TIMEOUT:-14400}" env \
  CUDA_VISIBLE_DEVICES="$gpu" \
  IMAGINARIUM_STOP_AFTER_STAGE=S3 \
  IMAGINARIUM_S3_MAX_UNIQUE_FEATURES_PER_BATCH=8 \
  IMAGINARIUM_PARALLEL_GPT_PROCESSES=1 \
  IMAGINARIUM_GPT_LOCK_FILE=/tmp/lumenarium_fix116_gemini.lock \
  OMNIVERSE_DEEPSEARCH_URL="$deepsearch_url" \
  OMNIVERSE_DEEPSEARCH_WORKERS="$deepsearch_workers" \
  OMNIVERSE_DEEPSEARCH_MAX_ATTEMPTS=6 \
  OMNIVERSE_DEEPSEARCH_TIMEOUT=120 \
  OMNIVERSE_DEEPSEARCH_RETRY_DELAY=2 \
  PYTHONUNBUFFERED=1 \
  LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
  "$python" -u run_imaginarium_I2Layout_v4_deepsearch.py "$image" --config "$config" \
    > "$log_root/s1_s3.log" 2>&1
fi
s13_end="$(now_ns)"
source_pose="$(find "$result/S3_pose_inference" -maxdepth 1 -type f -name '*_placement_info.json' -print -quit 2>/dev/null || true)"
test -s "$source_pose" || { echo "FIX116_STOP missing S3 pose" >&2; exit 1; }
echo "FIX116_STAGE=S1_S3 PASS elapsed=$(elapsed "$s13_start" "$s13_end")"

fix61_start="$(now_ns)"
echo "FIX116_STAGE=SCENELM_FIX61 START $(date)"
env \
  SCENEPROOF_RESULTS_ROOT="$root" \
  SCENEPROOF_SOURCE_VERSION="$source" \
  SCENEPROOF_FIX43_SOURCE_MANIFEST="$manifest" \
  SCENEPROOF_FIX43_MANIFEST="$root/fix61_manifest.txt" \
  SCENEPROOF_FIX43_EXPECTED_SCENES=1 \
  SCENEPROOF_FIX43_MINIMUM_NONZERO_SCENES=0 \
  SCENEPROOF_FIX43_CONTROL_VERSION="$control" \
  SCENEPROOF_FIX43_GUARDED_VERSION="$guarded" \
  SCENEPROOF_FIX43_CERTIFIED_VERSION="$baseline" \
  SCENEPROOF_FIX43_SKIP_RENDER=1 \
  SCENEPROOF_FIX43_SKIP_FORMAL_EVAL=1 \
  IMAGINARIUM_GPU0_ID="$gpu" IMAGINARIUM_GPU1_ID="$gpu" \
  IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-8000}" \
  bash scripts/run_sceneproof_fix43_inloop_fullstack_smoke5_fix56.sh \
  > "$log_root/fix61.log" 2>&1
fix61_end="$(now_ns)"
test -s "$root/${scene}_${baseline}_result/S4_layout_refinement/${scene}_${baseline}_placement_info_s4.json"
echo "FIX116_STAGE=SCENELM_FIX61 PASS elapsed=$(elapsed "$fix61_start" "$fix61_end")"

fix114_start="$(now_ns)"
echo "FIX116_STAGE=SCENEPROOF_FIX114 START $(date)"
env \
  SCENEPROOF_RESULTS_ROOT="$root" \
  SCENEPROOF_MANIFEST="$manifest" \
  SCENEPROOF_FIX114_SOURCE_VERSION="$source" \
  SCENEPROOF_FIX114_GEOMETRY_VERSION="$control" \
  SCENEPROOF_FIX114_BASELINE_VERSION="$baseline" \
  SCENEPROOF_FIX114_TARGET_VERSION="$target" \
  SCENEPROOF_FIX114_SKIP_RENDER=1 \
  SCENEPROOF_FIX114_SKIP_COMPARISON_ARCHIVE=1 \
  IMAGINARIUM_GPU0_ID="$gpu" IMAGINARIUM_GPU1_ID="$gpu" \
  IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-8000}" \
  bash scripts/run_sceneproof_vertical_support_final_paper30_fix114.sh \
  > "$log_root/fix114.log" 2>&1
fix114_end="$(now_ns)"
echo "FIX116_STAGE=SCENEPROOF_FIX114 PASS elapsed=$(elapsed "$fix114_start" "$fix114_end")"

render_start="$(now_ns)"
echo "FIX116_STAGE=FINAL_RENDER START $(date)"
env \
  SCENEPROOF_RESULTS_ROOT="$root" \
  SCENEPROOF_MANIFEST="$manifest" \
  SCENEPROOF_RENDER_SOURCE_VERSION="$source" \
  SCENEPROOF_CERTIFIED_VERSION="$target" \
  SCENEPROOF_RENDER_LOG_ROOT="$log_root/render" \
  SCENEPROOF_RENDER_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}" \
  SCENEPROOF_RENDER_FORCE=1 \
  IMAGINARIUM_GPU0_ID="$gpu" IMAGINARIUM_GPU1_ID="$gpu" \
  IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-8000}" \
  bash scripts/render_sceneproof_certified_paper30.sh \
  > "$log_root/render.log" 2>&1
render_end="$(now_ns)"; full_end="$(now_ns)"
final_render="$root/${scene}_${target}_result/S4_layout_refinement/${scene}_${target}_render_simu.png"
test -s "$final_render"
echo "FIX116_STAGE=FINAL_RENDER PASS elapsed=$(elapsed "$render_start" "$render_end")"

audit="$root/sceneba_audit/$target"
"$python" - "$root/smoke_result.json" "$scene" "$source" "$baseline" "$target" \
  "$s13_start" "$s13_end" "$fix61_start" "$fix61_end" \
  "$fix114_start" "$fix114_end" "$render_start" "$render_end" \
  "$full_start" "$full_end" "$audit/final_eval.json" "$final_render" <<'PY'
import json, sys
from pathlib import Path
out, scene, source, baseline, target, *rest = sys.argv[1:]
timestamps = list(map(int, rest[:10])); evaluation, render = rest[10:]
names = ["s1_s3", "scenelm_fix61", "sceneproof_fix114", "render", "end_to_end"]
timing = {name:(timestamps[2*i+1]-timestamps[2*i])/1e9 for i,name in enumerate(names)}
record = {
  "schema_version":"sceneproof_fix114_s1_s4_smoke_v1", "passed":True,
  "scene":scene, "source":source, "baseline":baseline, "target":target,
  "timing_seconds":timing, "evaluation":json.load(open(evaluation)),
  "render":str(Path(render).resolve()),
}
Path(out).write_text(json.dumps(record,indent=2)+"\n")
print(f"FIX116_PASSED=True SCENE={scene}")
for key,value in timing.items(): print(f"{key.upper()}_SECONDS={value:.3f}")
print(f"FIX116_EVAL={evaluation}")
print(f"FIX116_RENDER={render}")
print(f"FIX116_RESULT={out}")
PY
