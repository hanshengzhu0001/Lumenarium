#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
manifest="/tmp/sceneproof_nonsmooth_poll_smoke1.txt"
version="v5_sceneproof_nonsmooth_poll_smoke1_fix12"
root="a10_reusable_results/paper30/sceneba_audit"
aggregate="$root/${version}.json"
diagnostic="$root/${version}_poll.json"

SCENEPROOF_SCHUR_MANIFEST="$manifest" \
SCENEPROOF_SCHUR_VERSION="$version" \
  bash scripts/eval_sceneproof_full_so3_guarded_schur_smoke5.sh

"$PY" sceneproof_nonsmooth_poll_audit.py \
  --aggregate "$aggregate" \
  --out "$diagnostic"

echo "NONSMOOTH_AGGREGATE=$aggregate"
echo "NONSMOOTH_DIAGNOSTIC=$diagnostic"
