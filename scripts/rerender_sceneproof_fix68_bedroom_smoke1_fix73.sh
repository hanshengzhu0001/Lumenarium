#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
manifest="/tmp/sceneproof_fix68_bedroom_rerender_fix73.txt"
target="v5_sceneproof_com_scoped_rollback_paper30_fix68"
render="a10_reusable_results/paper30/bedroom_01_${target}_result/S4_layout_refinement/bedroom_01_${target}_render_simu.png"
printf '%s\n' bedroom_01 > "$manifest"

SCENEPROOF_MANIFEST="$manifest" \
SCENEPROOF_CERTIFIED_VERSION="$target" \
SCENEPROOF_RENDER_LOG_ROOT="logs/sceneproof_fix68_bedroom_rerender_fix73" \
SCENEPROOF_RENDER_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}" \
SCENEPROOF_RENDER_FORCE=1 \
IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
  bash scripts/render_sceneproof_certified_paper30.sh

cp -f -- "$render" "$HOME/fix68_bedroom_rerender_fix73.png"
echo "FIX73_RENDER=$(readlink -f "$render")"
echo "FIX73_DOWNLOAD=$(readlink -f "$HOME/fix68_bedroom_rerender_fix73.png")"
