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
runner="scripts/run_paper30_v4_s4_only_dual_gpu.sh"
python_bin="${IMAGINARIUM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

test -f "$manifest"
test -f "$runner"
grep -Eq 'max_iterations[[:space:]]*=[[:space:]]*5000' \
    modules/S4_blender_layout_and_corr.py || {
    echo "Legacy SA5000 constant was not found; refusing a mislabeled benchmark." >&2
    exit 2
}
mkdir -p "$suite_root"

check_fresh() {
    local version="$1"
    local found
    found="$(
        find "$results_root" \
            -path "*_${version}_result/S4_layout_refinement/*_placement_info_s4.json" \
            -type f 2>/dev/null | head -1
    )"
    if test -n "$found" && test "${IMAGINARIUM_S4_BENCHMARK_ALLOW_RESUME:-0}" != "1"; then
        echo "Refusing a mixed cached/fresh timing run; output exists: $found" >&2
        echo "Use new version names or set IMAGINARIUM_S4_BENCHMARK_ALLOW_RESUME=1 (timing will be incomplete)." >&2
        exit 2
    fi
}

for version in "$legacy_version" "$layout_version" "$depth_version"; do
    check_fresh "$version"
done

echo "===== LEGACY SA5000 $(date) ====="
env \
    IMAGINARIUM_PAPER30_MANIFEST="$manifest" \
    IMAGINARIUM_PAPER30_RESULTS_ROOT="$results_root" \
    IMAGINARIUM_S4_ENGINE=legacy \
    IMAGINARIUM_LAYOUTVLM_STAGE=legacy_sa5000 \
    IMAGINARIUM_LAYOUTVLM_ITERATIONS=5000 \
    IMAGINARIUM_S4_SOURCE_VERSION=v4_deepsearch \
    IMAGINARIUM_S4_TARGET_VERSION="$legacy_version" \
    IMAGINARIUM_S4_WORKER_LOG_ROOT="$suite_root/legacy_sa5000" \
    IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
    IMAGINARIUM_S4_SCENE_TIMEOUT="${IMAGINARIUM_S4_SCENE_TIMEOUT:-14400}" \
    bash "$runner"

echo "===== LAYOUTVLM FULL400 $(date) ====="
env \
    IMAGINARIUM_PAPER30_MANIFEST="$manifest" \
    IMAGINARIUM_PAPER30_RESULTS_ROOT="$results_root" \
    IMAGINARIUM_S4_ENGINE=layoutvlm \
    IMAGINARIUM_LAYOUTVLM_STAGE=full \
    IMAGINARIUM_LAYOUTVLM_ITERATIONS=400 \
    IMAGINARIUM_S4_SOURCE_VERSION=v4_deepsearch \
    IMAGINARIUM_S4_TARGET_VERSION="$layout_version" \
    IMAGINARIUM_S4_WORKER_LOG_ROOT="$suite_root/layoutvlm400" \
    IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
    IMAGINARIUM_S4_SCENE_TIMEOUT="${IMAGINARIUM_S4_SCENE_TIMEOUT:-14400}" \
    bash "$runner"

echo "===== LAYOUTVLM DEPTH-CENTER TRANSLATION PASS $(date) ====="
env \
    IMAGINARIUM_PAPER30_MANIFEST="$manifest" \
    IMAGINARIUM_PAPER30_RESULTS_ROOT="$results_root" \
    IMAGINARIUM_S4_ENGINE=layoutvlm \
    IMAGINARIUM_LAYOUTVLM_STAGE=depth \
    IMAGINARIUM_LAYOUTVLM_ITERATIONS=400 \
    IMAGINARIUM_S4_SOURCE_VERSION=v4_deepsearch \
    IMAGINARIUM_S4_REFERENCE_VERSION="$layout_version" \
    IMAGINARIUM_S4_TARGET_VERSION="$depth_version" \
    IMAGINARIUM_LAYOUTVLM_DEPTH_WEIGHT=0.01 \
    IMAGINARIUM_LAYOUTVLM_DEPTH_CENTER_WEIGHT=1 \
    IMAGINARIUM_LAYOUTVLM_DEPTH_SIZE_WEIGHT=0 \
    IMAGINARIUM_LAYOUTVLM_DEPTH_METRIC_WEIGHT=0 \
    IMAGINARIUM_LAYOUTVLM_DEPTH_FREEZE_YAW=1 \
    IMAGINARIUM_LAYOUTVLM_DEPTH_TRUST_WEIGHT=0 \
    IMAGINARIUM_LAYOUTVLM_DEPTH_MIN_PIXELS=800 \
    IMAGINARIUM_S4_WORKER_LOG_ROOT="$suite_root/depth_center" \
    IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
    IMAGINARIUM_S4_SCENE_TIMEOUT="${IMAGINARIUM_S4_SCENE_TIMEOUT:-14400}" \
    bash "$runner"

versions="${legacy_version},${layout_version},${depth_version}"

echo "===== PHYSICAL + RUNTIME AUDIT $(date) ====="
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
    'import json,sys; d=json.load(open(sys.argv[1])); f=d.get("failures",[]); print("PHYSICAL_FAILURES=",len(f)); sys.exit(bool(f))' \
    "$results_root/eval_physical_s4_benchmark.json"

echo "===== 8000PX GT POSE AUDIT $(date) ====="
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
    'import json,sys; d=json.load(open(sys.argv[1])); f=d.get("failures",[]); print("GT_FAILURES=",len(f)); sys.exit(bool(f))' \
    "$results_root/eval_gt_metrics_s4_benchmark_8000.json"

echo "===== BENCHMARK COMPLETE $(date) ====="
cat "$results_root/EVAL_PHYSICAL_S4_BENCHMARK.ascii"
