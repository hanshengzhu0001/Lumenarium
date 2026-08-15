#!/usr/bin/env bash
set -u

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

manifest="${IMAGINARIUM_PAPER30_MANIFEST:-a10_reusable_results/paper30/manifest.txt}"
output_root="${IMAGINARIUM_PAPER30_OUTPUT_ROOT:-a10_reusable_results/paper30}"
config="${IMAGINARIUM_PAPER30_V4_CONFIG:-config/config_a10_paper30_v4_deepsearch.yaml}"
python_bin="${IMAGINARIUM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
gpu_free_floor_mb="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-18000}"
s3_batch="${IMAGINARIUM_S3_MAX_UNIQUE_FEATURES_PER_BATCH:-8}"
scene_timeout="${IMAGINARIUM_SCENE_TIMEOUT:-14400}"
gemini_lock="${IMAGINARIUM_GPT_LOCK_FILE:-/tmp/lumenarium_gemini.lock}"

mkdir -p logs
test -f "$manifest" || { echo "Missing manifest: $manifest" >&2; exit 2; }
test -f "$config" || { echo "Missing config: $config" >&2; exit 2; }
test -x "$python_bin" || { echo "Missing Python: $python_bin" >&2; exit 2; }

list0="/tmp/lumenarium_paper30_v4_gpu0.$$.txt"
list1="/tmp/lumenarium_paper30_v4_gpu1.$$.txt"
awk 'NR % 2 == 1' "$manifest" > "$list0"
awk 'NR % 2 == 0' "$manifest" > "$list1"

cleanup() {
    rm -f -- "$list0" "$list1"
}
trap cleanup EXIT

run_worker() {
    local gpu="$1"
    local scene_list="$2"
    local scene tag out free rc

    while IFS= read -r scene || test -n "$scene"; do
        scene="${scene%$'\r'}"
        test -n "$scene" || continue
        tag="${scene}_v4_deepsearch"
        out="${output_root}/${tag}_result"

        if compgen -G "$out/S4_layout_refinement/*_placement_info_s4.json" >/dev/null; then
            echo "CACHED $tag"
            continue
        fi

        while true; do
            free="$(
                nvidia-smi -i "$gpu" \
                    --query-gpu=memory.free \
                    --format=csv,noheader,nounits 2>/dev/null |
                    head -1 | tr -d ' '
            )"
            if test -n "$free" && test "$free" -ge "$gpu_free_floor_mb"; then
                break
            fi
            echo "WAIT_GPU $tag gpu=$gpu free=${free:-unknown}MiB $(date)"
            sleep 60
        done

        echo "START $tag gpu=$gpu free=${free}MiB S3_BATCH=$s3_batch $(date)"
        timeout "$scene_timeout" env \
            CUDA_VISIBLE_DEVICES="$gpu" \
            IMAGINARIUM_S3_MAX_UNIQUE_FEATURES_PER_BATCH="$s3_batch" \
            IMAGINARIUM_PARALLEL_GPT_PROCESSES=1 \
            IMAGINARIUM_GPT_LOCK_FILE="$gemini_lock" \
            OMNIVERSE_DEEPSEARCH_MAX_ATTEMPTS=6 \
            OMNIVERSE_DEEPSEARCH_TIMEOUT=120 \
            OMNIVERSE_DEEPSEARCH_RETRY_DELAY=2 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            PYTHONUNBUFFERED=1 \
            LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
            "$python_bin" -u run_imaginarium_I2Layout_v4_deepsearch.py \
            "demo/${tag}.png" \
            --config "$config"
        rc=$?

        if compgen -G "$out/S4_layout_refinement/*_placement_info_s4.json" >/dev/null; then
            echo "OK $tag gpu=$gpu"
        else
            echo "FAIL $tag gpu=$gpu rc=$rc"
        fi
        sleep 15
    done < "$scene_list"
}

worker_pids=()
stop_workers() {
    if ((${#worker_pids[@]})); then
        kill "${worker_pids[@]}" 2>/dev/null || true
    fi
}
trap 'stop_workers; cleanup; exit 130' INT TERM

run_worker 0 "$list0" > logs/paper30_v4_deepsearch_gpu0.log 2>&1 &
worker_pids+=("$!")
sleep 15
run_worker 1 "$list1" > logs/paper30_v4_deepsearch_gpu1.log 2>&1 &
worker_pids+=("$!")

echo "GPU0_WORKER_PID=${worker_pids[0]}"
echo "GPU1_WORKER_PID=${worker_pids[1]}"
echo "GEMINI_LOCK=$gemini_lock"
echo "S3_BATCH=$s3_batch GPU_FREE_FLOOR_MB=$gpu_free_floor_mb"

status=0
for pid in "${worker_pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
