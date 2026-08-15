#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
manifest="/tmp/sceneproof_global_trust_descent_smoke2.txt"
SCENEPROOF_SCHUR_MANIFEST="$manifest" \
SCENEPROOF_SCHUR_VERSION=v5_sceneproof_normalized_consistency_smoke2_fix9 \
  bash scripts/eval_sceneproof_full_so3_guarded_schur_smoke5.sh
