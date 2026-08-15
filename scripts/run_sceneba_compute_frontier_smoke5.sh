#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"

manifest="${SCENEBA_COMPUTE_MANIFEST:-/tmp/sceneba_compute_smoke5.txt}"
results_root="${SCENEBA_COMPUTE_RESULTS_ROOT:-a10_reusable_results/paper30}"
log_root="${SCENEBA_COMPUTE_LOG_ROOT:-logs/sceneba_compute_frontier_smoke5}"
runner="scripts/run_paper30_v4_s4_only_dual_gpu.sh"
source_version="${SCENEBA_COMPUTE_SOURCE_VERSION:-v4_deepsearch}"

test -f "$manifest" || {
  printf '%s\n' \
    bedroom_01 \
    livingroom_10 \
    casino_01 \
    official_01 \
    streelitter_01 > "$manifest"
}
mkdir -p "$log_root"

for spec in \
  30:v4_router_b30 \
  100:v4_router_b100 \
  400:v4_router_b400
do
  IFS=: read -r budget version <<< "$spec"
  echo "===== START budget=$budget version=$version $(date) ====="
  env \
    IMAGINARIUM_PAPER30_MANIFEST="$manifest" \
    IMAGINARIUM_PAPER30_RESULTS_ROOT="$results_root" \
    IMAGINARIUM_S4_ENGINE=layoutvlm \
    IMAGINARIUM_LAYOUTVLM_STAGE=full \
    IMAGINARIUM_LAYOUTVLM_ITERATIONS="$budget" \
    IMAGINARIUM_S4_SOURCE_VERSION="$source_version" \
    IMAGINARIUM_S4_TARGET_VERSION="$version" \
    IMAGINARIUM_S4_WORKER_LOG_ROOT="$log_root/$version" \
    IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
    IMAGINARIUM_S4_SCENE_TIMEOUT="${IMAGINARIUM_S4_SCENE_TIMEOUT:-3600}" \
    bash "$runner"
  echo "===== FINISH budget=$budget version=$version $(date) ====="
done

echo "COMPUTE_FRONTIER_RUNS_COMPLETE $(date)"
