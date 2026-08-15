#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
manifest="/tmp/sceneproof_nonsmooth_poll_smoke2.txt"
version="v5_sceneproof_nonsmooth_poll_smoke2_fix13"
printf '%s\n' livingroom_10 official_01 > "$manifest"

SCENEPROOF_SCHUR_MANIFEST="$manifest" \
SCENEPROOF_SCHUR_VERSION="$version" \
IMAGINARIUM_SCENEPROOF_IN_LOOP_GUARDED_SCHUR=1 \
  bash scripts/run_sceneproof_full_so3_guarded_schur_smoke5.sh
