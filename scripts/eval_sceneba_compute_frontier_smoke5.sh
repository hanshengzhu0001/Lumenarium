#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"

PY="${SCENEBA_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
manifest="${SCENEBA_COMPUTE_MANIFEST:-/tmp/sceneba_compute_smoke5.txt}"
root="${SCENEBA_COMPUTE_RESULTS_ROOT:-a10_reusable_results/paper30}"
audit="$root/sceneba_audit/compute_frontier_smoke5"
mkdir -p "$audit"

versions="v4_router_b30,v4_router_b100,v4_router_b400"

"$PY" eval_physical_realizability.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --versions "$versions" \
  --geometry-version v4_deepsearch \
  --baseline-version v4_router_b400 \
  --runtime-log v4_router_b30=logs/sceneba_compute_frontier_smoke5/v4_router_b30 \
  --runtime-log v4_router_b100=logs/sceneba_compute_frontier_smoke5/v4_router_b100 \
  --runtime-log v4_router_b400=logs/sceneba_compute_frontier_smoke5/v4_router_b400 \
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

"$PY" sceneba_compute_frontier.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --physical-object-csv "$audit/physical_objects.csv" \
  --source-version v4_deepsearch \
  --budget-version 30=v4_router_b30 \
  --budget-version 100=v4_router_b100 \
  --budget-version 400=v4_router_b400 \
  --reference-budget 400 \
  --out "$audit/frontier.json" \
  --object-csv-out "$audit/frontier_objects.csv"

"$PY" sceneba_materialize_compute_oracle.py \
  --saved-results "$root" \
  --frontier "$audit/frontier.json" \
  --target-version v4_router_oracle_mixed

mixed_versions="$versions,v4_router_oracle_mixed"
"$PY" eval_physical_realizability.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --versions "$mixed_versions" \
  --geometry-version v4_deepsearch \
  --baseline-version v4_router_b400 \
  --metrics-out "$audit/mixed_physical.json" \
  --scene-csv "$audit/mixed_physical_scenes.csv" \
  --object-csv "$audit/mixed_physical_objects.csv" \
  --report-out "$audit/mixed_physical.ascii"

"$PY" eval_gt_metrics.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --versions "$mixed_versions" \
  --min-visible-mask-area 8000 \
  --min-visible-bbox-size 0 \
  --batch-logs logs \
  --metrics-out "$audit/mixed_gt_8000.json" \
  --manifest-out "$audit/mixed_gt_manifest_8000.json"

"$PY" sceneba_finalize_compute_frontier.py \
  --frontier "$audit/frontier.json" \
  --physical "$audit/mixed_physical.json" \
  --gt "$audit/mixed_gt_8000.json" \
  --reference-version v4_router_b400 \
  --mixed-version v4_router_oracle_mixed \
  --out "$audit/final_gates.json"

"$PY" - \
  "$audit/physical.json" \
  "$audit/gt_8000.json" \
  "$audit/mixed_physical.json" \
  "$audit/mixed_gt_8000.json" <<'PY'
import json, sys
for path in sys.argv[1:]:
    data = json.load(open(path))
    failures = data.get("failures", [])
    print(path, "FAILURES=", len(failures))
    if failures:
        print(*failures[:10], sep="\n")
        raise SystemExit(1)
PY

cat "$audit/physical.ascii"
cat "$audit/mixed_physical.ascii"
