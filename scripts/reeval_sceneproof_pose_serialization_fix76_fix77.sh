#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
scene="bedroom_01"
version="v5_sceneproof_pose_serialization_smoke1_fix76"
target_dir="$root/${scene}_${version}_result/S4_layout_refinement"
placement="$target_dir/${scene}_${version}_placement_info_s4.json"
reference_dir="$root/${scene}_v5_sceneproof_visual_rollback_smoke1_fix43_result/S4_layout_refinement"
reference="$(find "$reference_dir" -maxdepth 1 -type f -name '*_placement_info_s4.json' -print -quit)"
inprocess="$HOME/fix76_inprocess_bedroom.png"
roundtrip="$HOME/fix76_roundtrip_bedroom.png"
pipeline_log="logs/$version/${scene}_gpu0.log"
out="$root/sceneba_audit/$version/pose_serialization_roundtrip_fix77.json"

"$HOME/.venvs/lumenarium-py311/bin/python" \
  sceneproof_pose_serialization_roundtrip_fix76.py \
  --reference-placement "$reference" \
  --placement "$placement" \
  --inprocess-render "$inprocess" \
  --roundtrip-render "$roundtrip" \
  --pipeline-log "$pipeline_log" \
  --out "$out"

echo "FIX77_STATUS_NOTE=$HOME/Lumenarium/SCENEPROOF_CURRENT_STATUS_2026-08-06.md"
echo "FIX77_AUDIT=$(readlink -f "$out")"
