#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

manifest="${IMAGINARIUM_PAPER30_MANIFEST:-a10_reusable_results/paper30/manifest.txt}"
results_root="${IMAGINARIUM_PAPER30_RESULTS_ROOT:-a10_reusable_results/paper30}"
suite_root="${IMAGINARIUM_S4_BENCHMARK_LOG_ROOT:-logs/paper30_s4_benchmark}"
legacy_version="${IMAGINARIUM_S4_LEGACY_BENCHMARK_VERSION:-v4_legacy_sa5000_bench}"
layout_version="${IMAGINARIUM_S4_LAYOUT_BENCHMARK_VERSION:-v4_layoutvlm400_bench}"
depth_version="${IMAGINARIUM_S4_DEPTH_BENCHMARK_VERSION:-v4_layoutvlm400_depthcenter_bench}"
python_bin="${IMAGINARIUM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
versions="${legacy_version},${layout_version},${depth_version}"

test -f "$manifest"
expected="$(grep -cve '^[[:space:]]*$' "$manifest")"

for version in "$legacy_version" "$layout_version" "$depth_version"; do
    completed="$(
        find "$results_root" \
            -path "*_${version}_result/S4_layout_refinement/*_placement_info_s4.json" \
            -type f | wc -l
    )"
    test "$completed" -eq "$expected" || {
        echo "S4 preflight failed: $version=$completed/$expected" >&2
        exit 2
    }
done

geometry_ready="$(
    find "$results_root" \
        -path "*_${legacy_version}_result/S4_layout_refinement/*_placement_info_s3.json" \
        -type f | wc -l
)"
test "$geometry_ready" -eq "$expected" || {
    echo "Geometry preflight failed: $legacy_version pre-S4 snapshots=$geometry_ready/$expected" >&2
    exit 2
}

echo "===== PHYSICAL + RUNTIME RE-EVALUATION $(date) ====="
"$python_bin" eval_physical_realizability.py \
    --saved-results "$results_root" \
    --scenes "$manifest" \
    --versions "$versions" \
    --geometry-version "$legacy_version" \
    --baseline-version "$legacy_version" \
    --runtime-log "$legacy_version=$suite_root/legacy_sa5000" \
    --runtime-log "$layout_version=$suite_root/layoutvlm400" \
    --runtime-log "$depth_version=$suite_root/depth_center" \
    --runtime-composite "$depth_version=$layout_version+$depth_version" \
    --metrics-out "$results_root/eval_physical_s4_benchmark.json" \
    --scene-csv "$results_root/eval_physical_s4_benchmark_scenes.csv" \
    --object-csv "$results_root/eval_physical_s4_benchmark_objects.csv" \
    --report-out "$results_root/EVAL_PHYSICAL_S4_BENCHMARK.ascii"
"$python_bin" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); f=d.get("failures",[]); print("PHYSICAL_FAILURES=",len(f)); print(*f[:10],sep="\n"); sys.exit(bool(f))' \
    "$results_root/eval_physical_s4_benchmark.json"

echo "===== 8000PX GT POSE RE-EVALUATION $(date) ====="
"$python_bin" eval_gt_metrics.py \
    --saved-results "$results_root" \
    --scenes "$manifest" \
    --versions "$versions" \
    --min-visible-mask-area 8000 \
    --min-visible-bbox-size 0 \
    --batch-logs logs \
    --metrics-out "$results_root/eval_gt_metrics_s4_benchmark_8000.json" \
    --manifest-out "$results_root/eval_freeze_manifest_s4_benchmark_8000.json"
"$python_bin" -c \
    'import collections,json,sys; d=json.load(open(sys.argv[1])); f=d.get("failures",[]); print("GT_FAILURES=",len(f)); print(*collections.Counter(x["error"] for x in f).most_common(10),sep="\n"); sys.exit(bool(f))' \
    "$results_root/eval_gt_metrics_s4_benchmark_8000.json"

echo "===== RE-EVALUATION COMPLETE $(date) ====="
cat "$results_root/EVAL_PHYSICAL_S4_BENCHMARK.ascii"
