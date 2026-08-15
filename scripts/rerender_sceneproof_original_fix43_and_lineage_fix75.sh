#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"

root="${SCENEPROOF_RESULTS_ROOT:-a10_reusable_results/paper30}"
scene="bedroom_01"
source_version="v4_deepsearch"
target="v5_sceneproof_visual_rollback_smoke1_fix43"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"
samples="${SCENEPROOF_RENDER_SAMPLES:-256}"
gpu="${SCENEPROOF_RENDER_GPU:-0}"
gpu_floor="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}"
timeout_seconds="${SCENEPROOF_RENDER_TIMEOUT:-1800}"

source_json="$(find "$root/${scene}_${source_version}_result/S3_pose_inference" \
  -maxdepth 1 -type f -name '*_placement_info.json' -print -quit)"
target_dir="$root/${scene}_${target}_result/S4_layout_refinement"
placement="$(find "$target_dir" -maxdepth 1 -type f -name '*_placement_info_s4.json' -print -quit)"
old_render="$(find "$target_dir" -maxdepth 1 -type f -name '*_render_simu.png' -print -quit)"
rerender="$HOME/fix43_original_bedroom_rerender_fix75.png"
camera_audit="$HOME/fix43_original_bedroom_rerender_fix75.camera.json"
fix68="$HOME/fix68_bedroom_rerender_fix73.png"
image_comparison="$root/sceneba_audit/sceneproof_original_fix43_roundtrip_fix75.json"
lineage="$root/sceneba_audit/sceneproof_pillow_pose_lineage_fix75.json"
log="$HOME/Lumenarium/logs/sceneproof_original_fix43_rerender_fix75.log"
tmp_output="/tmp/sceneproof_original_fix43_rerender_${scene}_$$"

test -x "$blender" || { echo "Missing Blender: $blender" >&2; exit 2; }
test -s "$source_json" || { echo "Missing S3 source: $source_json" >&2; exit 2; }
test -s "$placement" || { echo "Missing original Fix43 placement: $placement" >&2; exit 2; }
test -s "$old_render" || { echo "Missing original Fix43 render: $old_render" >&2; exit 2; }
test -s "$fix68" || { echo "Missing Fix73 render: $fix68" >&2; exit 2; }
mkdir -p "$tmp_output" "$(dirname "$lineage")" "$(dirname "$log")"
trap 'rm -rf -- "$tmp_output"' EXIT

while true; do
  free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if test -n "$free" && test "$free" -ge "$gpu_floor"; then break; fi
  echo "WAIT_GPU_FIX75 gpu=$gpu free=${free:-unknown}MiB"
  sleep 30
done

timeout "$timeout_seconds" env \
  CUDA_VISIBLE_DEVICES="$gpu" \
  IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$placement" \
  IMAGINARIUM_S4_RENDER_ONLY_OUTPUT="$rerender" \
  IMAGINARIUM_S4_RENDER_ONLY_AUDIT="$camera_audit" \
  IMAGINARIUM_S4_RENDER_ONLY_SAMPLES="$samples" \
  PYTHONUNBUFFERED=1 \
  LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
  "$blender" --background \
    --python modules/S4_blender_layout_and_corr.py -- \
    --obj_placement_info_json_path "$source_json" \
    --output_folder "$tmp_output" \
    > "$log" 2>&1 < /dev/null

test -s "$rerender" || { echo "Original Fix43 rerender was not created" >&2; exit 1; }
test -s "$camera_audit" || { echo "Original Fix43 camera audit was not created" >&2; exit 1; }

"$HOME/.venvs/lumenarium-py311/bin/python" \
  sceneproof_render_roundtrip_compare.py \
  --old-smooth "$old_render" \
  --rerender-smooth "$rerender" \
  --fix68 "$fix68" \
  --out "$image_comparison"

"$HOME/.venvs/lumenarium-py311/bin/python" \
  sceneproof_pillow_pose_lineage_fix75.py \
  --saved-results "$root" \
  --scene "$scene" \
  --out "$lineage"

echo "FIX75_ORIGINAL_RENDER=$(readlink -f "$old_render")"
echo "FIX75_RERENDER=$(readlink -f "$rerender")"
echo "FIX75_IMAGE_COMPARISON=$(readlink -f "$image_comparison")"
echo "FIX75_POSE_LINEAGE=$(readlink -f "$lineage")"
echo "FIX75_LOG=$(readlink -f "$log")"
