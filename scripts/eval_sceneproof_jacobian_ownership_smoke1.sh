#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
version="${SCENEPROOF_JACOBIAN_VERSION:-v5_sceneproof_jacobian_ownership_smoke1}"
out="a10_reusable_results/paper30/sceneba_audit/${version}.json"

"$PY" sceneproof_jacobian_ownership_audit.py \
  --saved-results a10_reusable_results/paper30 \
  --scene bedroom_01 \
  --version "$version" \
  --out "$out"

echo "JACOBIAN_OWNERSHIP_AUDIT=$out"
