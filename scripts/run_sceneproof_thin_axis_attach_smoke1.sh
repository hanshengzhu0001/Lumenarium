#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
manifest="/tmp/sceneproof_thin_axis_attach_smoke1.txt"
version="${SCENEPROOF_THIN_AXIS_VERSION:-v5_sceneproof_thin_axis_attach_smoke1_fix28}"
printf '%s\n' bedroom_01 > "$manifest"

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
  IMAGINARIUM_SCENEPROOF_FULL_SO3_GUARDED_SCHUR=0 \
  IMAGINARIUM_SCENEPROOF_IN_LOOP_GUARDED_SCHUR=0 \
  IMAGINARIUM_SCENELM_KINEMATIC_BACKSUB=0 \
  IMAGINARIUM_SCENEPROOF_THIN_AXIS_ATTACH_RATIO=0.25 \
  IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
  IMAGINARIUM_S4_SCENE_TIMEOUT=3600 \
  IMAGINARIUM_S4_WORKER_LOG_ROOT="logs/$version" \
  bash scripts/run_paper30_v4_s4_only_dual_gpu.sh

env \
  SCENEPROOF_MANIFEST="$manifest" \
  SCENEPROOF_CERTIFIED_VERSION="$version" \
  SCENEPROOF_RENDER_LOG_ROOT="logs/${version}_locked_camera_render" \
  SCENEPROOF_RENDER_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}" \
  IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
  bash scripts/render_sceneproof_certified_paper30.sh

echo "THIN_AXIS_SMOKE1_FINISHED version=$version"
echo "PLACEMENT=$HOME/Lumenarium/a10_reusable_results/paper30/bedroom_01_${version}_result/S4_layout_refinement/bedroom_01_${version}_placement_info_s4.json"
echo "RENDER=$HOME/Lumenarium/a10_reusable_results/paper30/bedroom_01_${version}_result/S4_layout_refinement/bedroom_01_${version}_render_simu.png"
