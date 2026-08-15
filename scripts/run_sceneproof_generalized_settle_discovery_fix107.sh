#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
saved="$HOME/Lumenarium/a10_reusable_results/paper30"
baseline="v5_sceneproof_collision_partial_commit_certified_paper30_fix61"
audit_root="${SCENEPROOF_TRUE_MESH_AUDIT_ROOT:-$saved/sceneba_audit/$baseline/true_mesh_com_paper30_fix66}"
physical_objects="${SCENEPROOF_FIX61_PHYSICAL_OBJECTS:-$saved/sceneba_audit/$baseline/physical_objects.csv}"
out_root="$saved/sceneba_audit/v5_sceneproof_generalized_settle_paper30_fix107"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

mkdir -p "$out_root"
"$python" sceneproof_generalized_settle_candidates.py \
  --saved-results "$saved" \
  --manifest "$saved/manifest.txt" \
  --baseline-version "$baseline" \
  --true-mesh-audit-root "$audit_root" \
  --physical-objects "$physical_objects" \
  --minimum-gap-m "${SCENEPROOF_SETTLE_MINIMUM_GAP_M:-0.005}" \
  --maximum-gap-m "${SCENEPROOF_SETTLE_MAXIMUM_GAP_M:-0.5}" \
  --max-representatives "${SCENEPROOF_SETTLE_MAX_REPRESENTATIVES:-30}" \
  --out "$out_root/candidate_discovery.json" \
  --all-report "$out_root/candidates_all.tsv" \
  --report "$out_root/candidate_representatives.tsv"

echo "GENERALIZED_SETTLE_DISCOVERY=$(readlink -f "$out_root/candidate_discovery.json")"
echo "GENERALIZED_SETTLE_CANDIDATES_ALL=$(readlink -f "$out_root/candidates_all.tsv")"
echo "GENERALIZED_SETTLE_REPRESENTATIVES=$(readlink -f "$out_root/candidate_representatives.tsv")"
