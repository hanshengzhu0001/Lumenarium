#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
root="$HOME/Lumenarium/a10_reusable_results/fix116_s1_s4_smoke1"
scene=bedroom_01
source=v4_deepsearch
geometry=v5_sceneproof_fix43_smooth_fix116_s1_s4_smoke1
baseline=v5_sceneproof_collision_partial_commit_certified_fix116_s1_s4_smoke1
target="${SCENEPROOF_FIX117_TARGET_VERSION:-v5_sceneproof_vertical_support_com_projection_fix117_1}"
manifest="$root/fix117_manifest.txt"
log="$HOME/Lumenarium/logs/sceneproof_fix116_support_transaction_smoke1_fix117"
printf '%s\n' "$scene" > "$manifest"
mkdir -p "$log"

env \
  SCENEPROOF_RESULTS_ROOT="$root" \
  SCENEPROOF_MANIFEST="$manifest" \
  SCENEPROOF_FIX114_SOURCE_VERSION="$source" \
  SCENEPROOF_FIX114_GEOMETRY_VERSION="$geometry" \
  SCENEPROOF_FIX114_BASELINE_VERSION="$baseline" \
  SCENEPROOF_FIX114_TARGET_VERSION="$target" \
  SCENEPROOF_FIX114_SKIP_RENDER=1 \
  SCENEPROOF_FIX114_SKIP_COMPARISON_ARCHIVE=1 \
  IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-8000}" \
  bash scripts/run_sceneproof_vertical_support_final_paper30_fix114.sh \
  > "$log/transaction.log" 2>&1

env \
  SCENEPROOF_RESULTS_ROOT="$root" \
  SCENEPROOF_MANIFEST="$manifest" \
  SCENEPROOF_RENDER_SOURCE_VERSION="$source" \
  SCENEPROOF_CERTIFIED_VERSION="$target" \
  SCENEPROOF_RENDER_LOG_ROOT="$log/render" \
  SCENEPROOF_RENDER_SAMPLES=256 \
  SCENEPROOF_RENDER_FORCE=1 \
  IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-8000}" \
  bash scripts/render_sceneproof_certified_paper30.sh \
  > "$log/render.log" 2>&1

audit="$root/sceneba_audit/$target"
render="$root/${scene}_${target}_result/S4_layout_refinement/${scene}_${target}_render_simu.png"
echo "FIX117_EVAL=$audit/final_eval.json"
echo "FIX117_TRANSACTION=$audit/transactions/${scene}.json"
echo "FIX117_RENDER=$render"
