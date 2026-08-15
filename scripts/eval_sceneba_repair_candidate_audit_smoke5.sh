#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/Lumenarium"

PY="${SCENEBA_PYTHON:-${HOME}/.venvs/lumenarium-py311/bin/python}"
MANIFEST="${SCENEBA_REPAIR_MANIFEST:-/tmp/sceneba_repair_smoke5.txt}"
ROOT="a10_reusable_results/paper30"
OUT="${ROOT}/sceneba_audit/repair_candidate_oracle_smoke5.json"
GT="${ROOT}/sceneba_audit/repair_smoke5_gt_8000.json"

test -s "${GT}" || {
  echo "Missing existing Smoke5 GT metrics: ${GT}" >&2
  exit 2
}

"${PY}" sceneba_repair_oracle.py \
  --saved-results "${ROOT}" \
  --dataset-dir asset_data/imaginarium_3d_scene_layout_dataset \
  --scenes "${MANIFEST}" \
  --gt-metrics "${GT}" \
  --match-version v4 \
  --reference-version v4 \
  --audit-version v4_repair_candidate_audit \
  --out "${OUT}" \
  2>&1 | tee logs/sceneba_repair_candidate_oracle_smoke5.log

echo "ORACLE=${OUT}"
