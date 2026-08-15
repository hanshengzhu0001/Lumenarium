#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

saved="$HOME/Lumenarium/a10_reusable_results/paper30"
manifest="${SCENEPROOF_COM_MANIFEST:-$saved/manifest.txt}"
incumbent="v5_sceneproof_fix43_smooth_paper30_fix61"
candidate="v5_sceneproof_collision_partial_commit_certified_paper30_fix61"
audit="$saved/sceneba_audit/$candidate/true_mesh_com_paper30_fix66"
logs="$HOME/Lumenarium/logs/sceneproof_true_mesh_com_paper30_fix66"
python_bin="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

env \
  SCENEPROOF_COM_MANIFEST="$manifest" \
  SCENEPROOF_COM_AUDIT_ROOT="$audit" \
  SCENEPROOF_COM_LOG_ROOT="$logs" \
  bash scripts/run_sceneproof_true_mesh_com_paper30_fix64.sh

mkdir -p "$audit/oracles"
while IFS= read -r scene || test -n "$scene"; do
  scene="${scene%$'\r'}"
  test -n "$scene" || continue
  "$python_bin" sceneproof_true_mesh_com_counterfactual_oracle.py \
    --saved-results "$saved" \
    --scene "$scene" \
    --incumbent-version "$incumbent" \
    --candidate-version "$candidate" \
    --com-audit-root "$audit" \
    --responsibility "$audit/responsibility.json" \
    --out "$audit/oracles/$scene.json"
done < "$manifest"

"$python_bin" sceneproof_true_mesh_com_counterfactual_protocol.py \
  --manifest "$manifest" \
  --oracle-root "$audit/oracles" \
  --out "$audit/protocol.json"

echo "COM_PAPER30_RESPONSIBILITY=$(readlink -f "$audit/responsibility.json")"
echo "COM_PAPER30_PROTOCOL=$(readlink -f "$audit/protocol.json")"
