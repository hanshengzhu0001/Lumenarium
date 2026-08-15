#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Lumenarium"
root="$HOME/Lumenarium/a10_reusable_results/fix116_s1_s4_smoke1"
manifest="$root/fix118_manifest.txt"
geometry=v5_sceneproof_fix43_smooth_fix116_s1_s4_smoke1
baseline=v5_sceneproof_collision_partial_commit_certified_fix116_s1_s4_smoke1
visual=v5_sceneproof_vertical_support_com_projection_fix117_1
partial=v5_sceneproof_vertical_support_partial_certified_fix118
audit="$root/sceneba_audit/relation_conditioned_eval_fix119"
mkdir -p "$audit"
bash scripts/freeze_sceneproof_fix117_visual_fix119.sh
"$HOME/.venvs/lumenarium-py311/bin/python" eval_physical_realizability.py \
  --saved-results "$root" --scenes "$manifest" \
  --versions "$baseline,$visual,$partial" --geometry-version "$geometry" \
  --baseline-version "$baseline" --metrics-out "$audit/physical.json" \
  --scene-csv "$audit/scenes.csv" --object-csv "$audit/objects.csv" \
  --collision-pairs-csv "$audit/collision_pairs.csv" \
  --report-out "$audit/physical.txt"
echo "FIX119_RELATION_EVAL=$audit/physical.json"
echo "FIX119_COLLISION_PAIRS=$audit/collision_pairs.csv"
