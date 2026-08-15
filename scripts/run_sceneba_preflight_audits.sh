#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python_bin="${IMAGINARIUM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
results_root="${IMAGINARIUM_PAPER30_RESULTS_ROOT:-a10_reusable_results/paper30}"
manifest="${IMAGINARIUM_PAPER30_MANIFEST:-$results_root/manifest.txt}"
dataset_dir="${IMAGINARIUM_DATASET_DIR:-asset_data/imaginarium_3d_scene_layout_dataset}"
baseline="${SCENEBA_BASELINE_VERSION:-v4_legacy_sa5000_bench}"
candidate="${SCENEBA_CANDIDATE_VERSION:-v4_layoutvlm400_bench}"
match_version="${SCENEBA_MATCH_VERSION:-$candidate}"
retrieval_version="${SCENEBA_RETRIEVAL_VERSION:-v4_deepsearch}"
pose_version="${SCENEBA_POSE_VERSION:-v4_deepsearch}"
geometry_version="${SCENEBA_GEOMETRY_VERSION:-$candidate}"
gt_metrics="${SCENEBA_GT_METRICS:-$results_root/eval_gt_metrics_s4_benchmark_8000.json}"
physical_metrics="${SCENEBA_PHYSICAL_METRICS:-$results_root/eval_physical_s4_benchmark.json}"
out_dir="${SCENEBA_AUDIT_OUT_DIR:-$results_root/sceneba_audit}"

for path in "$manifest" "$gt_metrics" "$physical_metrics"; do
    test -f "$path" || {
        echo "Missing required input: $path" >&2
        exit 2
    }
done
mkdir -p "$out_dir"

echo "===== PAIRED BOOTSTRAP $(date) ====="
"$python_bin" sceneba_paired_audit.py \
    --gt-metrics "$gt_metrics" \
    --physical-metrics "$physical_metrics" \
    --baseline "$baseline" \
    --candidate "$candidate" \
    --samples 10000 \
    --confidence 0.95 \
    --rotation-margin -0.01 \
    --translation-margin -0.005 \
    --out "$out_dir/paired_bootstrap_10000.json" \
    2>&1 | tee "$out_dir/paired_bootstrap_10000.log"

echo "===== TOP-K / PARENT / POSE-MODE ORACLE $(date) ====="
"$python_bin" sceneba_topk_oracle.py \
    --saved-results "$results_root" \
    --dataset-dir "$dataset_dir" \
    --scenes "$manifest" \
    --gt-metrics "$gt_metrics" \
    --match-version "$match_version" \
    --retrieval-version "$retrieval_version" \
    --pose-version "$pose_version" \
    --geometry-version "$geometry_version" \
    --yaw-offsets 0,90,180,270 \
    --out "$out_dir/topk_oracle.json" \
    2>&1 | tee "$out_dir/topk_oracle.log"

"$python_bin" - "$out_dir/topk_oracle.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
failures = data.get("failures", [])
print("ORACLE_FAILURES=", len(failures))
print(*failures[:10], sep="\n")
if failures:
    raise SystemExit(1)
print("GO_NO_GO=", json.dumps(data["go_no_go"], indent=2))
PY

echo "===== SCENEBA PREFLIGHT COMPLETE $(date) ====="
echo "Outputs: $out_dir"
