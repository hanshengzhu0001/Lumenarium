#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
manifest="/tmp/sceneproof_nonsmooth_poll_smoke2.txt"
version="v5_sceneproof_nonsmooth_poll_smoke2_fix13"
root="a10_reusable_results/paper30/sceneba_audit"
aggregate="$root/${version}.json"
diagnostic="$root/${version}_cross_scene.json"

SCENEPROOF_SCHUR_MANIFEST="$manifest" \
SCENEPROOF_SCHUR_VERSION="$version" \
  bash scripts/eval_sceneproof_full_so3_guarded_schur_smoke5.sh

"$PY" sceneproof_nonsmooth_poll_smoke2_audit.py \
  --aggregate "$aggregate" \
  --out "$diagnostic"

echo "NONSMOOTH_SMOKE2_AGGREGATE=$aggregate"
echo "NONSMOOTH_SMOKE2_DIAGNOSTIC=$diagnostic"
