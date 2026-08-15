#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"

PY="${SCENEBA_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
manifest="${SCENEBA_ACTIVE_ROUTER_MANIFEST:-/tmp/sceneba_compute_smoke5.txt}"
root="${SCENEBA_COMPUTE_RESULTS_ROOT:-a10_reusable_results/paper30}"
reference="${SCENEBA_ACTIVE_ROUTER_REFERENCE:-v4_router_b400}"
router="${SCENEBA_ACTIVE_ROUTER_VERSION:-v4_active_router_v1}"
router_logs="${SCENEBA_ACTIVE_ROUTER_LOG_ROOT:-logs/sceneba_active_router_v1_smoke5}"
reference_logs="${SCENEBA_ACTIVE_ROUTER_REFERENCE_LOG_ROOT:-logs/sceneba_compute_frontier_smoke5/v4_router_b400}"
audit="${SCENEBA_ACTIVE_ROUTER_AUDIT_DIR:-$root/sceneba_audit/active_router_v1_smoke5}"
versions="$reference,$router"
mkdir -p "$audit"

"$PY" eval_physical_realizability.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --versions "$versions" \
  --geometry-version v4_deepsearch \
  --baseline-version "$reference" \
  --runtime-log "$reference=$reference_logs" \
  --runtime-log "$router=$router_logs" \
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

"$PY" sceneba_active_router_audit.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --physical "$audit/physical.json" \
  --gt "$audit/gt_8000.json" \
  --reference-version "$reference" \
  --router-version "$router" \
  --out "$audit/final_gates.json"

cat "$audit/physical.ascii"
