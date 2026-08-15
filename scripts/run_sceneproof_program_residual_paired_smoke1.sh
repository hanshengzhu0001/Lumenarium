#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
manifest="${SCENEPROOF_RESIDUAL_MANIFEST:-/tmp/sceneproof_program_residual_smoke1.txt}"
legacy="${SCENEPROOF_RESIDUAL_LEGACY_VERSION:-v5_sceneproof_residual_legacy_smoke1}"
program="${SCENEPROOF_RESIDUAL_PROGRAM_VERSION:-v5_sceneproof_residual_program_smoke1}"
iterations="${SCENEPROOF_RESIDUAL_ITERATIONS:-30}"

if [[ ! -s "$manifest" ]]; then
  printf '%s\n' bedroom_01 > "$manifest"
fi

run_one() {
  local version="$1"
  local use_program="$2"
  env \
    IMAGINARIUM_PAPER30_MANIFEST="$manifest" \
    IMAGINARIUM_S4_SOURCE_VERSION=v4_deepsearch \
    IMAGINARIUM_S4_SOURCE_STAGE=S3_pose_inference \
    IMAGINARIUM_S4_SOURCE_PATTERN='*_placement_info.json' \
    IMAGINARIUM_S4_TARGET_VERSION="$version" \
    IMAGINARIUM_S4_ENGINE=layoutvlm \
    IMAGINARIUM_LAYOUTVLM_STAGE=full \
    IMAGINARIUM_LAYOUTVLM_SOLVER=v5_scenelm \
    IMAGINARIUM_LAYOUTVLM_ITERATIONS="$iterations" \
    IMAGINARIUM_LAYOUTVLM_ACTIVE_SET_ROUTER=0 \
    IMAGINARIUM_SCENEPROOF_PROGRAM_IR=1 \
    IMAGINARIUM_SCENEPROOF_REQUIRE_FACTOR_PARITY=1 \
    IMAGINARIUM_SCENEPROOF_SHADOW_RESIDUAL_PARITY=1 \
    IMAGINARIUM_SCENEPROOF_USE_PROGRAM_RESIDUALS="$use_program" \
    IMAGINARIUM_SCENEPROOF_RESIDUAL_FALLBACK=1 \
    IMAGINARIUM_SCENELM_KINEMATIC_BACKSUB=0 \
    IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
    IMAGINARIUM_S4_SCENE_TIMEOUT=3600 \
    IMAGINARIUM_S4_WORKER_LOG_ROOT="logs/${version}" \
    bash scripts/run_paper30_v4_s4_only_dual_gpu.sh
}

echo "===== START legacy residual $(date) ====="
run_one "$legacy" 0
echo "===== START program residual $(date) ====="
run_one "$program" 1
echo "SCENEPROOF_RESIDUAL_PAIRED_FINISHED legacy=$legacy program=$program $(date)"

