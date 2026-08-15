#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
version="${SCENEPROOF_SCHUR_VERSION:-v5_sceneproof_dtype_stationarity_smoke1_fix5}"
reference="${SCENEPROOF_SCHUR_REFERENCE:-v5_sceneproof_jacobian_ownership_smoke1}"
out="a10_reusable_results/paper30/sceneba_audit/${version}.json"

"$PY" sceneproof_full_so3_schur_audit.py \
  --saved-results a10_reusable_results/paper30 \
  --scene bedroom_01 \
  --version "$version" \
  --reference-version "$reference" \
  --out "$out"

echo "FULL_SO3_SCHUR_AUDIT=$out"
