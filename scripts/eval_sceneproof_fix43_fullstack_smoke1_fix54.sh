#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
manifest="/tmp/sceneproof_fix43_fullstack_smoke1.txt"
printf '%s\n' bedroom_01 > "$manifest"

SCENEPROOF_MANIFEST="$manifest" \
SCENEPROOF_LEGACY_VERSION=v4_legacy_sa5000_bench \
SCENEPROOF_INCUMBENT_VERSION=v5_sceneproof_fix43_smooth_smoke1_fix53 \
SCENEPROOF_CANDIDATE_VERSION=v5_sceneproof_fix43_guarded_smoke1_fix53 \
SCENEPROOF_CERTIFIED_VERSION=v5_sceneproof_fix43_certified_smoke1_fix53 \
SCENEPROOF_INCLUDE_CERTIFICATE_RUNTIME=1 \
SCENEPROOF_REUSE_CERTIFICATE=1 \
SCENEPROOF_ALLOW_SAFE_ABSTAIN=1 \
SCENEPROOF_RENDER_LOG_ROOT=logs/v5_sceneproof_fix43_certified_smoke1_fix53_locked_camera_render \
  bash scripts/eval_sceneproof_postsim_component_certificate_fix21.sh
