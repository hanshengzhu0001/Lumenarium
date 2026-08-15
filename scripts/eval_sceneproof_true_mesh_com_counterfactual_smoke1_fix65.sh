#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

saved="$HOME/Lumenarium/a10_reusable_results/paper30"
incumbent="v5_sceneproof_fix43_smooth_paper30_fix61"
candidate="v5_sceneproof_collision_partial_commit_certified_paper30_fix61"
audit="$saved/sceneba_audit/$candidate/true_mesh_com_smoke1_fix64_1"
out="$audit/counterfactual_oracle_fix65.json"
python_bin="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

"$python_bin" sceneproof_true_mesh_com_counterfactual_oracle.py \
  --saved-results "$saved" \
  --scene bedroom_01 \
  --incumbent-version "$incumbent" \
  --candidate-version "$candidate" \
  --com-audit-root "$audit" \
  --responsibility "$audit/responsibility.json" \
  --out "$out"

echo "COUNTERFACTUAL_ORACLE=$(readlink -f "$out")"
