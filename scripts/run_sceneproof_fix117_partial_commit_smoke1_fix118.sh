#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Lumenarium"
root="$HOME/Lumenarium/a10_reusable_results/fix116_s1_s4_smoke1"
scene=bedroom_01
manifest="$root/fix118_manifest.txt"
baseline=v5_sceneproof_collision_partial_commit_certified_fix116_s1_s4_smoke1
visual=v5_sceneproof_vertical_support_com_projection_fix117_1
target=v5_sceneproof_vertical_support_partial_certified_fix118
geometry=v5_sceneproof_fix43_smooth_fix116_s1_s4_smoke1
audit="$root/sceneba_audit/$target"
printf '%s\n' "$scene" > "$manifest"
mkdir -p "$audit"
"$HOME/.venvs/lumenarium-py311/bin/python" sceneproof_fix114_partial_commit.py \
  --saved-results "$root" --scenes "$manifest" --geometry-version "$geometry" \
  --baseline-version "$baseline" --visual-version "$visual" --target-version "$target" \
  --transactions "$root/sceneba_audit/$visual/transactions" --out "$audit/partial_commit.json"
SCENEPROOF_RESULTS_ROOT="$root" SCENEPROOF_MANIFEST="$manifest" \
SCENEPROOF_RENDER_SOURCE_VERSION=v4_deepsearch SCENEPROOF_CERTIFIED_VERSION="$target" \
SCENEPROOF_RENDER_LOG_ROOT="$HOME/Lumenarium/logs/$target/render" \
SCENEPROOF_RENDER_FORCE=1 bash scripts/render_sceneproof_certified_paper30.sh
echo "FIX118_REPORT=$audit/partial_commit.json"
echo "FIX118_RENDER=$root/${scene}_${target}_result/S4_layout_refinement/${scene}_${target}_render_simu.png"
