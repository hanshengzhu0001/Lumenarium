#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
manifest="${SCENEPROOF_BINDING_MANIFEST:-/tmp/sceneproof_factor_binding_smoke1.txt}"
version="${SCENEPROOF_BINDING_VERSION:-v5_sceneproof_factor_binding_audit_smoke1}"

if [[ ! -s "$manifest" ]]; then
  printf '%s\n' bedroom_01 > "$manifest"
fi

env \
  IMAGINARIUM_PAPER30_MANIFEST="$manifest" \
  IMAGINARIUM_S4_SOURCE_VERSION=v4_deepsearch \
  IMAGINARIUM_S4_SOURCE_STAGE=S3_pose_inference \
  IMAGINARIUM_S4_SOURCE_PATTERN='*_placement_info.json' \
  IMAGINARIUM_S4_TARGET_VERSION="$version" \
  IMAGINARIUM_S4_ENGINE=layoutvlm \
  IMAGINARIUM_LAYOUTVLM_STAGE=full \
  IMAGINARIUM_LAYOUTVLM_SOLVER=adam \
  IMAGINARIUM_LAYOUTVLM_ITERATIONS=1 \
  IMAGINARIUM_LAYOUTVLM_ACTIVE_SET_ROUTER=0 \
  IMAGINARIUM_SCENEPROOF_PROGRAM_IR=1 \
  IMAGINARIUM_SCENEPROOF_REQUIRE_FACTOR_PARITY=1 \
  IMAGINARIUM_SCENEPROOF_REQUIRE_BINDING_AUDIT=1 \
  IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
  IMAGINARIUM_S4_SCENE_TIMEOUT=3600 \
  IMAGINARIUM_S4_WORKER_LOG_ROOT="logs/${version}" \
  bash scripts/run_paper30_v4_s4_only_dual_gpu.sh

