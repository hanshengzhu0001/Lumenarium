#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/Lumenarium"

PY="${SCENEBA_PYTHON:-${HOME}/.venvs/lumenarium-py311/bin/python}"
MANIFEST="${SCENEBA_REPAIR_MANIFEST:-/tmp/sceneba_repair_smoke5.txt}"
ROOT="a10_reusable_results/paper30"
VERSIONS="v4,v4_repair_control,v4_repair_yaw_only,v4_repair_translation_only,v4_repair_joint"
REPAIR_VERSIONS="v4_repair_control,v4_repair_yaw_only,v4_repair_translation_only,v4_repair_joint"
OUT="${ROOT}/sceneba_audit"
mkdir -p "${OUT}"

"${PY}" sceneba_repair_summary.py \
  --saved-results "${ROOT}" \
  --scenes "${MANIFEST}" \
  --versions "${REPAIR_VERSIONS}" \
  2>&1 | tee logs/sceneba_repair_smoke5_decisions.log

"${PY}" eval_gt_metrics.py \
  --saved-results "${ROOT}" \
  --scenes "${MANIFEST}" \
  --versions "${VERSIONS}" \
  --min-visible-mask-area 8000 \
  --min-visible-bbox-size 0 \
  --batch-logs logs \
  --metrics-out "${OUT}/repair_smoke5_gt_8000.json" \
  --manifest-out "${OUT}/repair_smoke5_gt_manifest_8000.json" \
  2>&1 | tee logs/sceneba_repair_smoke5_gt_8000.log

"${PY}" eval_physical_realizability.py \
  --saved-results "${ROOT}" \
  --scenes "${MANIFEST}" \
  --versions "${VERSIONS}" \
  --geometry-version v4_deepsearch \
  --baseline-version v4_repair_control \
  --metrics-out "${OUT}/repair_smoke5_physical.json" \
  --scene-csv "${OUT}/repair_smoke5_physical_scenes.csv" \
  --object-csv "${OUT}/repair_smoke5_physical_objects.csv" \
  --report-out "${OUT}/repair_smoke5_physical.txt" \
  2>&1 | tee logs/sceneba_repair_smoke5_physical.log

echo "GT=${OUT}/repair_smoke5_gt_8000.json"
echo "PHYSICAL=${OUT}/repair_smoke5_physical.json"
