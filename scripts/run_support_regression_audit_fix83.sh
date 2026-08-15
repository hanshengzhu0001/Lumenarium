#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
incumbent="v5_sceneproof_pose_serialization_smoke1_fix76"
candidate="v5_sceneproof_local_settle_candidate_smoke1_fix82"
audit="$root/sceneba_audit/$candidate"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

incumbent_placement="$root/bedroom_01_${incumbent}_result/S4_layout_refinement/bedroom_01_${incumbent}_placement_info_s4.json"
candidate_placement="$root/bedroom_01_${candidate}_result/S4_layout_refinement/bedroom_01_${candidate}_placement_info_s4.json"
probe="$audit/probes/single_sofa_chair_1.json"
object_csv="$audit/physical_objects.csv"

test -s "$object_csv" || { echo "Missing: $object_csv" >&2; exit 2; }
test -s "$incumbent_placement" || { echo "Missing: $incumbent_placement" >&2; exit 2; }
test -s "$candidate_placement" || { echo "Missing: $candidate_placement" >&2; exit 2; }
test -s "$probe" || { echo "Missing: $probe" >&2; exit 2; }

"$python" sceneproof_support_regression_audit_fix83.py \
  --object-csv "$object_csv" \
  --incumbent-placement "$incumbent_placement" \
  --candidate-placement "$candidate_placement" \
  --probe "$probe" \
  2>&1 | tee "$audit/support_regression_audit_fix83.txt"

echo ""
echo "FIX83_AUDIT=$audit/support_regression_audit_fix83.txt"
