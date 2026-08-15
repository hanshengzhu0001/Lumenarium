#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

manifest="${DEEPSEARCH_S2_MANIFEST:-/tmp/sceneba_repair_smoke5.txt}"
results_root="${DEEPSEARCH_S2_RESULTS_ROOT:-a10_reusable_results/paper30}"
source_version="${DEEPSEARCH_S2_SOURCE_VERSION:-v4_deepsearch}"
config="${DEEPSEARCH_S2_CONFIG:-config/config_a10_paper30_v4_deepsearch.yaml}"
python_bin="${IMAGINARIUM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
worker_values="${DEEPSEARCH_S2_WORKERS:-1 8}"
audit_root="${DEEPSEARCH_S2_AUDIT_ROOT:-$results_root/sceneba_audit/deepsearch_s2_concurrency_smoke5}"
log_root="${DEEPSEARCH_S2_LOG_ROOT:-logs/deepsearch_s2_concurrency_smoke5}"

test -f "$manifest" || { echo "Missing manifest: $manifest" >&2; exit 2; }
test -f "$config" || { echo "Missing config: $config" >&2; exit 2; }
test -x "$python_bin" || { echo "Missing Python: $python_bin" >&2; exit 2; }
mkdir -p "$audit_root" "$log_root"

for workers in $worker_values; do
    case "$workers" in
        ''|*[!0-9]*) echo "Invalid worker count: $workers" >&2; exit 2 ;;
    esac
    test "$workers" -ge 1 || { echo "Worker count must be >=1" >&2; exit 2; }
    version="v4_deepsearch_s2w${workers}_audit"
    runtime_file="$audit_root/runtime_w${workers}.jsonl"
    : > "$runtime_file"

    while IFS= read -r scene || test -n "$scene"; do
        scene="${scene%$'\r'}"
        test -n "$scene" || continue
        source_dir="$results_root/${scene}_${source_version}_result"
        target_dir="$results_root/${scene}_${version}_result"
        source_image="demo/${scene}_${source_version}.png"
        target_image="demo/${scene}_${version}.png"
        scene_log="$log_root/${scene}_w${workers}.log"

        test -d "$source_dir/S0_geometry_pred_results" || {
            echo "Missing cached S0: $source_dir" >&2; exit 3;
        }
        test -d "$source_dir/S1_scene_parsing_results" || {
            echo "Missing cached S1: $source_dir" >&2; exit 3;
        }
        test -f "$source_image" || {
            echo "Missing source image: $source_image" >&2; exit 3;
        }

        mkdir -p "$target_dir"
        rm -rf -- "$target_dir/S2_3d_retrieval_results"
        test -d "$target_dir/S0_geometry_pred_results" || \
            cp -a "$source_dir/S0_geometry_pred_results" "$target_dir/"
        test -d "$target_dir/S1_scene_parsing_results" || \
            cp -a "$source_dir/S1_scene_parsing_results" "$target_dir/"
        cp -f "$source_image" "$target_image"

        echo "START scene=$scene workers=$workers $(date)"
        started="$(date +%s.%N)"
        set +e
        env \
            IMAGINARIUM_USE_DEEPSEARCH=1 \
            IMAGINARIUM_SKIP_S2_VLM=1 \
            IMAGINARIUM_STOP_AFTER_STAGE=S2 \
            OMNIVERSE_DEEPSEARCH_WORKERS="$workers" \
            OMNIVERSE_DEEPSEARCH_MAX_ATTEMPTS=6 \
            OMNIVERSE_DEEPSEARCH_TIMEOUT=120 \
            OMNIVERSE_DEEPSEARCH_RETRY_DELAY=2 \
            PYTHONUNBUFFERED=1 \
            LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
            "$python_bin" -u run_imaginarium_I2Layout_v4_deepsearch.py \
            "$target_image" --config "$config" > "$scene_log" 2>&1
        rc=$?
        set -e
        ended="$(date +%s.%N)"
        elapsed="$($python_bin -c "print(float('$ended')-float('$started'))")"
        status="fail"
        test "$rc" -eq 0 && status="ok"
        printf '{"scene":"%s","workers":%s,"elapsed_seconds":%.6f,"status":"%s","return_code":%s}\n' \
            "$scene" "$workers" "$elapsed" "$status" "$rc" >> "$runtime_file"
        echo "END scene=$scene workers=$workers elapsed=$elapsed status=$status"
        test "$rc" -eq 0 || exit "$rc"
    done < "$manifest"
done

"$python_bin" tools/summarize_deepsearch_s2_concurrency.py \
    --audit-root "$audit_root" \
    --out "$audit_root/summary.json"

