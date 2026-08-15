#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
root="${SCENEPROOF_RESULTS_ROOT:-a10_reusable_results/paper30}"
scene="${SCENEPROOF_IDENTITY_SCENE:-bedroom_01}"
source="v4_deepsearch"
target="${SCENEPROOF_IDENTITY_VERSION:-v5_sceneproof_com_scoped_rollback_paper30_fix68}"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"
source_json="$(find "$root/${scene}_${source}_result/S3_pose_inference" -maxdepth 1 -type f -name '*_placement_info.json' -print -quit)"
placement="$root/${scene}_${target}_result/S4_layout_refinement/${scene}_${target}_placement_info_s4.json"
audit_root="$root/sceneba_audit/$target/render_identity_smoke1_fix72"
log_root="logs/sceneproof_render_identity_smoke1_fix72"
mkdir -p "$audit_root" "$log_root"

test -s "$source_json" || { echo "Missing source: $source_json" >&2; exit 2; }
test -s "$placement" || { echo "Missing placement: $placement" >&2; exit 2; }

tmp_output="/tmp/sceneproof_render_identity_fix71_${scene}_$$"
trap 'rm -rf -- "$tmp_output"' EXIT
mkdir -p "$tmp_output"

env \
  CUDA_VISIBLE_DEVICES="${SCENEPROOF_IDENTITY_GPU:-0}" \
  IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$placement" \
  IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER=1 \
  IMAGINARIUM_SCENEPROOF_RENDER_IDENTITY_AUDIT_OUTPUT="$audit_root/${scene}.json" \
  IMAGINARIUM_SCENEPROOF_RENDER_IDENTITY_COLOR_OUTPUT="$audit_root/${scene}_color_ids.png" \
  IMAGINARIUM_SCENEPROOF_RENDER_IDENTITY_ANNOTATED_OUTPUT="$audit_root/${scene}_annotated_ids.png" \
  IMAGINARIUM_SCENEPROOF_RENDER_IDENTITY_RESOLUTION=512 \
  PYTHONUNBUFFERED=1 \
  LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
  "$blender" --background \
    --python modules/S4_blender_layout_and_corr.py -- \
    --obj_placement_info_json_path "$source_json" \
    --output_folder "$tmp_output" \
    2>&1 | tee "$log_root/${scene}.log"

echo "FIX71_AUDIT=$(readlink -f "$audit_root/${scene}.json")"
echo "FIX71_COLOR_IDS=$(readlink -f "$audit_root/${scene}_color_ids.png")"
echo "FIX71_ANNOTATED_IDS=$(readlink -f "$audit_root/${scene}_annotated_ids.png")"
