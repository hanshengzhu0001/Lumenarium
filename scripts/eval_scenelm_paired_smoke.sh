#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"

PY="${SCENELM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
manifest="${SCENELM_MANIFEST:-/tmp/scenelm_smoke1.txt}"
root="${SCENELM_RESULTS_ROOT:-a10_reusable_results/paper30}"
reference="${SCENELM_REFERENCE_VERSION:-v4_scenelm_adam400_control}"
candidate="${SCENELM_VERSION:-v4_scenelm_v1}"
reference_logs="${SCENELM_REFERENCE_LOG_ROOT:-logs/${reference}}"
candidate_logs="${SCENELM_LOG_ROOT:-logs/${candidate}}"
audit="${SCENELM_AUDIT_DIR:-$root/sceneba_audit/${candidate}}"
versions="$reference,$candidate"
mkdir -p "$audit"

"$PY" eval_physical_realizability.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --versions "$versions" \
  --geometry-version v4_deepsearch \
  --baseline-version "$reference" \
  --runtime-log "$reference=$reference_logs" \
  --runtime-log "$candidate=$candidate_logs" \
  --metrics-out "$audit/physical.json" \
  --scene-csv "$audit/physical_scenes.csv" \
  --object-csv "$audit/physical_objects.csv" \
  --report-out "$audit/physical.ascii"

"$PY" eval_gt_metrics.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --versions "$versions" \
  --min-visible-mask-area 8000 \
  --min-visible-bbox-size 0 \
  --batch-logs logs \
  --metrics-out "$audit/gt_8000.json" \
  --manifest-out "$audit/gt_manifest_8000.json"

"$PY" scenelm_audit.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --physical "$audit/physical.json" \
  --gt "$audit/gt_8000.json" \
  --reference-version "$reference" \
  --scenelm-version "$candidate" \
  --out "$audit/final_gates.json"

cat "$audit/physical.ascii"
