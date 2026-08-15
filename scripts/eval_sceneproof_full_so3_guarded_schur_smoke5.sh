#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
manifest="${SCENEPROOF_SCHUR_MANIFEST:-/tmp/sceneproof_full_so3_schur_smoke5.txt}"
version="${SCENEPROOF_SCHUR_VERSION:-v5_sceneproof_stationarity_schur_smoke5_fix5}"
out="a10_reusable_results/paper30/sceneba_audit/${version}.json"

"$PY" sceneproof_full_so3_schur_smoke5_audit.py \
  --saved-results a10_reusable_results/paper30 \
  --scenes "$manifest" \
  --version "$version" \
  --out "$out"

echo "FULL_SO3_SCHUR_SMOKE5_AUDIT=$out"
