#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
root="${SCENEPROOF_RESULTS_ROOT:-a10_reusable_results/paper30}"
scene="${SCENEPROOF_IDENTITY_SCENE:-bedroom_01}"
smooth="v5_sceneproof_fix43_smooth_paper30_fix61"
candidate="v5_sceneproof_collision_partial_commit_certified_paper30_fix61"
final="v5_sceneproof_com_scoped_rollback_paper30_fix68"
source="v4_deepsearch"
com_root="$root/sceneba_audit/$candidate/true_mesh_com_paper30_fix66"
out_root="$root/sceneba_audit/$final/support_visual_identity_smoke1_fix70"
mkdir -p "$out_root" logs

"$PY" sceneproof_support_visual_identity_audit.py \
  --saved-results "$root" --scene "$scene" \
  --smooth-version "$smooth" --candidate-version "$candidate" \
  --final-version "$final" --source-version "$source" \
  --com-audit "$com_root/${scene}__${candidate}.json" \
  --out "$out_root/${scene}.json" \
  --atlas "$out_root/${scene}_s1_support_identity_atlas.png"

echo "FIX70_AUDIT=$(readlink -f "$out_root/${scene}.json")"
echo "FIX70_ATLAS=$(readlink -f "$out_root/${scene}_s1_support_identity_atlas.png")"
