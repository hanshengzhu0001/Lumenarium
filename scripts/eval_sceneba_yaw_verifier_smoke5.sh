#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/Lumenarium"

PY="${SCENEBA_PYTHON:-${HOME}/.venvs/lumenarium-py311/bin/python}"
MANIFEST="${SCENEBA_YAW_MANIFEST:-/tmp/sceneba_repair_smoke5.txt}"
ROOT="a10_reusable_results/paper30"
TARGET_VERSION="${SCENEBA_YAW_TARGET_VERSION:-v4_yaw_verifier}"
AUDIT="${ROOT}/sceneba_audit/${TARGET_VERSION}_smoke5.json"
PHYSICAL="${ROOT}/sceneba_audit/${TARGET_VERSION}_physical_smoke5.json"
GT="${ROOT}/sceneba_audit/repair_smoke5_gt_8000.json"

test -s "${GT}" || {
  echo "Missing existing Smoke5 GT metrics: ${GT}" >&2
  exit 2
}

"${PY}" sceneba_yaw_oracle.py \
  --saved-results "${ROOT}" \
  --dataset-dir asset_data/imaginarium_3d_scene_layout_dataset \
  --scenes "${MANIFEST}" \
  --gt-metrics "${GT}" \
  --match-version v4 \
  --reference-version v4 \
  --audit-version "${TARGET_VERSION}" \
  --out "${AUDIT}" \
  2>&1 | tee "logs/${TARGET_VERSION}_oracle_smoke5.log"

"${PY}" eval_physical_realizability.py \
  --saved-results "${ROOT}" \
  --scenes "${MANIFEST}" \
  --versions "v4,${TARGET_VERSION}" \
  --geometry-version v4_deepsearch \
  --baseline-version v4 \
  --metrics-out "${PHYSICAL}" \
  --scene-csv "${ROOT}/sceneba_audit/${TARGET_VERSION}_physical_scenes.csv" \
  --object-csv "${ROOT}/sceneba_audit/${TARGET_VERSION}_physical_objects.csv" \
  --report-out "${ROOT}/sceneba_audit/${TARGET_VERSION}_physical.txt" \
  2>&1 | tee "logs/${TARGET_VERSION}_physical_smoke5.log"

"${PY}" sceneba_finalize_yaw_gates.py \
  --yaw-audit "${AUDIT}" \
  --physical-metrics "${PHYSICAL}" \
  --baseline-version v4 \
  --audit-version "${TARGET_VERSION}" \
  2>&1 | tee "logs/${TARGET_VERSION}_final_gates_smoke5.log"

echo "YAW_AUDIT=${AUDIT}"
echo "PHYSICAL=${PHYSICAL}"
