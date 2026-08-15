#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
scene="bedroom_01"
version="${SCENEPROOF_FIX76_VERSION:-v5_sceneproof_pose_serialization_smoke1_fix76}"
manifest="/tmp/sceneproof_pose_serialization_smoke1_fix76.txt"
log_root="logs/$version"
coordinator_log="logs/${version}_coordinator.log"
printf '%s\n' "$scene" > "$manifest"

env \
  IMAGINARIUM_PAPER30_MANIFEST="$manifest" \
  IMAGINARIUM_S4_SOURCE_VERSION=v4_deepsearch \
  IMAGINARIUM_S4_SOURCE_STAGE=S3_pose_inference \
  IMAGINARIUM_S4_SOURCE_PATTERN='*_placement_info.json' \
  IMAGINARIUM_S4_TARGET_VERSION="$version" \
  IMAGINARIUM_S4_ENGINE=layoutvlm \
  IMAGINARIUM_LAYOUTVLM_STAGE=full \
  IMAGINARIUM_LAYOUTVLM_SOLVER=v5_scenelm \
  IMAGINARIUM_LAYOUTVLM_ITERATIONS=2 \
  IMAGINARIUM_LAYOUTVLM_ACTIVE_SET_ROUTER=0 \
  IMAGINARIUM_SCENEPROOF_PROGRAM_IR=1 \
  IMAGINARIUM_SCENEPROOF_REQUIRE_FACTOR_PARITY=1 \
  IMAGINARIUM_SCENEPROOF_REQUIRE_BINDING_AUDIT=1 \
  IMAGINARIUM_SCENEPROOF_SHADOW_JACOBIAN_OWNERSHIP=1 \
  IMAGINARIUM_SCENEPROOF_STABLE_LINEARIZATIONS=2 \
  IMAGINARIUM_SCENEPROOF_FULL_SO3_GUARDED_SCHUR=1 \
  IMAGINARIUM_SCENEPROOF_WARM_START_ANCHORED_PLANE_TRANSLATION=1 \
  IMAGINARIUM_SCENEPROOF_PLANE_ANCHOR_NORMAL_LIMIT_M=0.02 \
  IMAGINARIUM_SCENEPROOF_PLANE_PROXY_ABSTAIN_GAP_M=0.02 \
  IMAGINARIUM_SCENEPROOF_PLANE_ATTACH_REQUIRES_WITNESS=1 \
  IMAGINARIUM_SCENEPROOF_MATERIALIZED_WARM_START=1 \
  IMAGINARIUM_SCENEPROOF_PLANE_SIBLING_TANGENT_PROJECTION=1 \
  IMAGINARIUM_SCENEPROOF_PLANE_SIBLING_MAX_SHIFT_M=0.35 \
  IMAGINARIUM_SCENEPROOF_PLANE_COMPONENT_IMAGE_GAUGE=0 \
  IMAGINARIUM_SCENELM_KINEMATIC_BACKSUB=0 \
  IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
  IMAGINARIUM_S4_SCENE_TIMEOUT=3600 \
  IMAGINARIUM_S4_WORKER_LOG_ROOT="$log_root" \
  bash scripts/run_paper30_v4_s4_only_dual_gpu.sh \
  2>&1 | tee "$coordinator_log"

target_dir="$root/${scene}_${version}_result/S4_layout_refinement"
placement="$target_dir/${scene}_${version}_placement_info_s4.json"
inprocess="$target_dir/${scene}_${version}_render_simu.png"
reference_dir="$root/${scene}_v5_sceneproof_visual_rollback_smoke1_fix43_result/S4_layout_refinement"
reference="$(find "$reference_dir" -maxdepth 1 -type f -name '*_placement_info_s4.json' -print -quit)"
inprocess_copy="$HOME/fix76_inprocess_bedroom.png"
roundtrip="$HOME/fix76_roundtrip_bedroom.png"
roundtrip_audit="$HOME/fix76_roundtrip_bedroom.camera.json"
audit="$root/sceneba_audit/$version/pose_serialization_roundtrip.json"

test -s "$placement" || { echo "Missing Fix76 placement: $placement" >&2; exit 1; }
test -s "$inprocess" || { echo "Missing Fix76 in-process render: $inprocess" >&2; exit 1; }
test -s "$reference" || { echo "Missing original Fix43 placement" >&2; exit 2; }
cp -f -- "$inprocess" "$inprocess_copy"

SCENEPROOF_MANIFEST="$manifest" \
SCENEPROOF_CERTIFIED_VERSION="$version" \
SCENEPROOF_RENDER_LOG_ROOT="logs/${version}_roundtrip_render" \
SCENEPROOF_RENDER_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}" \
SCENEPROOF_RENDER_FORCE=1 \
IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
  bash scripts/render_sceneproof_certified_paper30.sh

cp -f -- "$inprocess" "$roundtrip"
default_camera_audit="$target_dir/${scene}_${version}_render_camera.json"
cp -f -- "$default_camera_audit" "$roundtrip_audit"

"$HOME/.venvs/lumenarium-py311/bin/python" \
  sceneproof_pose_serialization_roundtrip_fix76.py \
  --reference-placement "$reference" \
  --placement "$placement" \
  --inprocess-render "$inprocess_copy" \
  --roundtrip-render "$roundtrip" \
  --pipeline-log "$log_root/${scene}_gpu0.log" \
  --out "$audit"

echo "FIX76_INPROCESS=$(readlink -f "$inprocess_copy")"
echo "FIX76_ROUNDTRIP=$(readlink -f "$roundtrip")"
echo "FIX76_PLACEMENT=$(readlink -f "$placement")"
echo "FIX76_AUDIT=$(readlink -f "$audit")"
echo "FIX76_LOG=$(readlink -f "$log_root/${scene}_gpu0.log")"
