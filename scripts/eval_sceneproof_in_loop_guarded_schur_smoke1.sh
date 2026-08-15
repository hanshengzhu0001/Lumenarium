#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
manifest="/tmp/sceneproof_in_loop_guarded_schur_smoke1.txt"
version="v5_sceneproof_in_loop_guarded_schur_smoke1_fix10"
root="a10_reusable_results/paper30/sceneba_audit"
aggregate="$root/${version}.json"
gate="$root/${version}_gate.json"

"$PY" sceneproof_full_so3_schur_smoke5_audit.py \
  --saved-results a10_reusable_results/paper30 \
  --scenes "$manifest" \
  --version "$version" \
  --out "$aggregate"

"$PY" sceneproof_in_loop_schur_audit.py \
  --aggregate "$aggregate" \
  --out "$gate"

echo "IN_LOOP_SCHUR_AUDIT=$aggregate"
echo "IN_LOOP_SCHUR_GATE=$gate"
