#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/Lumenarium"

manifest="${SCENEBA_REPAIR_MANIFEST:-/tmp/sceneba_repair_smoke5.txt}"
iterations="${SCENEBA_REPAIR_ITERATIONS:-30}"
gpu_floor="${SCENEBA_REPAIR_GPU_FREE_FLOOR_MB:-16000}"

if [[ ! -s "${manifest}" ]]; then
  printf '%s\n' \
    bedroom_01 \
    livingroom_10 \
    casino_01 \
    official_01 \
    streelitter_01 > "${manifest}"
fi

run_arm() {
  local tag="$1"
  local enabled="$2"
  local yaws="$3"
  local max_translation="$4"
  local target="v4_repair_${tag}"
  echo "===== START ${target} $(date) ====="
  env \
    IMAGINARIUM_PAPER30_MANIFEST="${manifest}" \
    IMAGINARIUM_S4_SOURCE_VERSION=v4_deepsearch \
    IMAGINARIUM_S4_SOURCE_STAGE=S3_pose_inference \
    IMAGINARIUM_S4_SOURCE_PATTERN="*_placement_info.json" \
    IMAGINARIUM_S4_REFERENCE_VERSION=v4 \
    IMAGINARIUM_S4_REFERENCE_STAGE=S4_layout_refinement \
    IMAGINARIUM_S4_REFERENCE_PATTERN="*_placement_info_s4.json" \
    IMAGINARIUM_S4_TARGET_VERSION="${target}" \
    IMAGINARIUM_LAYOUTVLM_STAGE=depth \
    IMAGINARIUM_LAYOUTVLM_ITERATIONS="${iterations}" \
    IMAGINARIUM_LAYOUTVLM_DEPTH_WEIGHT=0 \
    IMAGINARIUM_LAYOUTVLM_DEPTH_TRUST_WEIGHT=0 \
    IMAGINARIUM_LAYOUTVLM_DEPTH_FREEZE_YAW=1 \
    IMAGINARIUM_LAYOUTVLM_DEPTH_MIN_PIXELS=800 \
    IMAGINARIUM_SCENEBA_DISCRETE_REPAIR="${enabled}" \
    IMAGINARIUM_SCENEBA_REPAIR_YAWS="${yaws}" \
    IMAGINARIUM_SCENEBA_REPAIR_MAX_TRANSLATION="${max_translation}" \
    IMAGINARIUM_SCENEBA_REPAIR_MIN_RELATIVE_GAIN=0.08 \
    IMAGINARIUM_SCENEBA_REPAIR_MIN_ABSOLUTE_GAIN=0.001 \
    IMAGINARIUM_SCENEBA_REPAIR_MIN_MARGIN=0.0002 \
    IMAGINARIUM_GPU_FREE_FLOOR_MB="${gpu_floor}" \
    IMAGINARIUM_S4_SCENE_TIMEOUT=3600 \
    IMAGINARIUM_S4_WORKER_LOG_ROOT="logs/${target}_smoke5" \
    bash scripts/run_paper30_v4_s4_only_dual_gpu.sh
  echo "===== FINISH ${target} $(date) ====="
}

# Identical 30-iteration backend for every arm.  Only the bounded proposal
# set changes, so gains cannot be attributed to a larger optimization budget.
run_arm control 0 "0" 0.000001
run_arm yaw_only 1 "0,90,180,270" 0.000001
run_arm translation_only 1 "0" 0.5
run_arm joint 1 "0,90,180,270" 0.5

