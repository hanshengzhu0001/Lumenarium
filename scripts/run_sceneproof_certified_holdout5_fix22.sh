#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
source_manifest="a10_reusable_results/paper30/manifest.txt"
manifest="${SCENEPROOF_HOLDOUT_MANIFEST:-/tmp/sceneproof_certified_holdout5_fix22.txt}"
control="${SCENEPROOF_INCUMBENT_VERSION:-v5_sceneproof_smooth_control_holdout5_fix22}"
candidate="${SCENEPROOF_CANDIDATE_VERSION:-v5_sceneproof_guarded_hybrid_holdout5_fix22}"
target="${SCENEPROOF_CERTIFIED_VERSION:-v5_sceneproof_postsim_component_certified_holdout5_fix22}"

if [[ ! -s "$manifest" ]]; then
  while IFS= read -r scene; do
    scene="${scene%$'\r'}"
    [[ -n "$scene" ]] || continue
    case "$scene" in
      bedroom_01|livingroom_10|casino_01|official_01|streelitter_01) continue ;;
    esac
    hash="$(printf 'sceneproof-fix22:%s' "$scene" | sha256sum | awk '{print $1}')"
    printf '%s %s\n' "$hash" "$scene"
  done < "$source_manifest" | sort | awk 'NR <= 5 {print $2}' > "$manifest"
fi

if [[ "$(wc -l < "$manifest")" -ne 5 ]]; then
  echo "Holdout manifest must contain exactly five scenes: $manifest" >&2
  exit 2
fi

echo "FROZEN_HOLDOUT_MANIFEST=$manifest"
cat "$manifest"

env \
  IMAGINARIUM_PAPER30_MANIFEST="$manifest" \
  IMAGINARIUM_S4_SOURCE_VERSION=v4_deepsearch \
  IMAGINARIUM_S4_SOURCE_STAGE=S3_pose_inference \
  IMAGINARIUM_S4_SOURCE_PATTERN='*_placement_info.json' \
  IMAGINARIUM_S4_TARGET_VERSION="$control" \
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
  IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
  IMAGINARIUM_S4_SCENE_TIMEOUT=3600 \
  IMAGINARIUM_S4_WORKER_LOG_ROOT="logs/$control" \
  bash scripts/run_paper30_v4_s4_only_dual_gpu.sh

SCENEPROOF_SCHUR_MANIFEST="$manifest" \
SCENEPROOF_SCHUR_VERSION="$candidate" \
IMAGINARIUM_SCENEPROOF_IN_LOOP_GUARDED_SCHUR=1 \
  bash scripts/run_sceneproof_full_so3_guarded_schur_smoke5.sh

SCENEPROOF_MANIFEST="$manifest" \
SCENEPROOF_INCUMBENT_VERSION="$control" \
SCENEPROOF_CANDIDATE_VERSION="$candidate" \
SCENEPROOF_CERTIFIED_VERSION="$target" \
  bash scripts/eval_sceneproof_postsim_component_certificate_fix21.sh

echo "SCENEPROOF_CERTIFIED_HOLDOUT5_FINISHED target=$target"
