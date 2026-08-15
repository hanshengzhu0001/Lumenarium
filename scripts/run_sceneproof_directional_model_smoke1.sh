#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
manifest="/tmp/sceneproof_directional_model_smoke1.txt"
version="v5_sceneproof_directional_model_smoke1_fix11"
printf '%s\n' livingroom_10 > "$manifest"

SCENEPROOF_SCHUR_MANIFEST="$manifest" \
SCENEPROOF_SCHUR_VERSION="$version" \
IMAGINARIUM_SCENEPROOF_IN_LOOP_GUARDED_SCHUR=1 \
  bash scripts/run_sceneproof_full_so3_guarded_schur_smoke5.sh
