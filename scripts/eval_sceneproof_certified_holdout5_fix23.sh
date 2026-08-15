#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"

SCENEPROOF_MANIFEST="${SCENEPROOF_HOLDOUT_MANIFEST:-/tmp/sceneproof_certified_holdout5_fix22.txt}" \
SCENEPROOF_LEGACY_VERSION=v4_legacy_sa5000_bench \
SCENEPROOF_LEGACY_LOG_ROOT=logs/paper30_s4_benchmark/legacy_sa5000 \
SCENEPROOF_INCUMBENT_VERSION=v5_sceneproof_smooth_control_holdout5_fix22 \
SCENEPROOF_CANDIDATE_VERSION=v5_sceneproof_guarded_hybrid_holdout5_fix22 \
SCENEPROOF_CERTIFIED_VERSION=v5_sceneproof_postsim_component_certified_holdout5_fix22 \
  bash scripts/eval_sceneproof_postsim_component_certificate_fix21.sh
