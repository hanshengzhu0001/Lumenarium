#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
root="$HOME/Lumenarium/a10_reusable_results/paper30"
python="$HOME/.venvs/lumenarium-py311/bin/python"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"
scene="bedroom_01"
baseline="v5_sceneproof_collision_partial_commit_certified_paper30_fix61"
source="v4_deepsearch"
audit_version="${SCENEPROOF_OVERHANG_AUDIT_VERSION:-v5_sceneproof_support_overhang_projection_smoke1_fix111}"
audit_root="$root/sceneba_audit/$audit_version"
com_audit="$audit_root/${scene}__${baseline}_true_mesh_com.json"
placement="$root/${scene}_${baseline}_result/S4_layout_refinement/${scene}_${baseline}_placement_info_s4.json"
source_json="$(find "$root/${scene}_${source}_result/S3_pose_inference" -maxdepth 1 -type f -name '*_placement_info.json' -print -quit)"
log_root="$HOME/Lumenarium/logs/$audit_version"
mkdir -p "$audit_root" "$log_root"

test -s "$placement"
test -s "$source_json"
tmp_output="$(mktemp -d /tmp/sceneproof_overhang_fix111_XXXXXX)"
trap 'rm -rf -- "$tmp_output"' EXIT

env \
  CUDA_VISIBLE_DEVICES=0 \
  IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$placement" \
  IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER=1 \
  IMAGINARIUM_SCENEPROOF_TRUE_MESH_COM_AUDIT_OUTPUT="$com_audit" \
  PYTHONUNBUFFERED=1 \
  LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
  "$blender" --background --python modules/S4_blender_layout_and_corr.py -- \
    --obj_placement_info_json_path "$source_json" \
    --output_folder "$tmp_output" \
    > "$log_root/${scene}_com.log" 2>&1 < /dev/null

"$python" sceneproof_overhang_screen_fix94.py \
  --scene "$scene" --com-audit "$com_audit" --placement "$placement" \
  --margin-threshold-m 0.005 --target-margin-m 0.01 \
  --translate-budget-m 0.15 --top-k 20 --require-consistent-com \
  --out-report "$audit_root/projection_candidates.json" \
  2>&1 | tee "$log_root/${scene}_screen.log"

echo "FIX111_POLICY=minimum_tangent_translation_full_so3_and_height_frozen"
echo "FIX111_COM_AUDIT=$(readlink -f "$com_audit")"
echo "FIX111_CANDIDATES=$(readlink -f "$audit_root/projection_candidates.json")"
