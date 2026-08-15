#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
control="v5_sceneproof_smooth_control_smoke5_fix15"
candidate="v5_sceneproof_guarded_hybrid_smoke5_fix15"
root="a10_reusable_results/paper30"
audit="$root/sceneba_audit/$candidate"

"$PY" sceneproof_postsim_responsibility_audit.py \
  --saved-results "$root" \
  --physical "$audit/physical.json" \
  --aggregate "$root/sceneba_audit/${candidate}.json" \
  --object-csv "$audit/physical_objects.csv" \
  --control-version "$control" \
  --candidate-version "$candidate" \
  --margin 0.005 \
  --out "$audit/postsim_responsibility.json" \
  --report "$audit/postsim_responsibility.txt"

cat "$audit/postsim_responsibility.txt"
