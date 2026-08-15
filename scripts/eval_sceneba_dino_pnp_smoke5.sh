#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/Lumenarium"

PY="${SCENEBA_PYTHON:-${HOME}/.venvs/lumenarium-py311/bin/python}"
MANIFEST="${SCENEBA_DINO_PNP_MANIFEST:-/tmp/sceneba_repair_smoke5.txt}"
ROOT="a10_reusable_results/paper30"
TARGET_VERSION="${SCENEBA_DINO_PNP_TARGET_VERSION:-v4_dino_pnp_oracle}"
OUT="${ROOT}/sceneba_audit/${TARGET_VERSION}_smoke5.json"
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
  --audit-version "${TARGET_VERSION}" \
  --out "${OUT}" \
  2>&1 | tee "logs/${TARGET_VERSION}_smoke5.log"

echo "ORACLE=${OUT}"
