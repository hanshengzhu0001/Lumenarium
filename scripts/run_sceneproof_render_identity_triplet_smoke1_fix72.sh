#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
root="${SCENEPROOF_RESULTS_ROOT:-a10_reusable_results/paper30}"
scene="${SCENEPROOF_IDENTITY_SCENE:-bedroom_01}"
smooth="v5_sceneproof_fix43_smooth_paper30_fix61"
candidate="v5_sceneproof_collision_partial_commit_certified_paper30_fix61"
final="v5_sceneproof_com_scoped_rollback_paper30_fix68"

for version in "$smooth" "$candidate" "$final"; do
  echo "===== RENDER IDENTITY version=$version ====="
  SCENEPROOF_IDENTITY_VERSION="$version" \
    bash scripts/run_sceneproof_render_identity_smoke1_fix71.sh
done

out_root="$root/sceneba_audit/$final/render_identity_triplet_smoke1_fix72"
mkdir -p "$out_root"
"$PY" sceneproof_render_identity_triplet_compare.py \
  --smooth "$root/sceneba_audit/$smooth/render_identity_smoke1_fix72/${scene}.json" \
  --candidate "$root/sceneba_audit/$candidate/render_identity_smoke1_fix72/${scene}.json" \
  --final "$root/sceneba_audit/$final/render_identity_smoke1_fix72/${scene}.json" \
  --fix70 "$root/sceneba_audit/$final/support_visual_identity_smoke1_fix70/${scene}.json" \
  --out "$out_root/${scene}.json"

echo "FIX72_COMPARE=$(readlink -f "$out_root/${scene}.json")"
for version in "$smooth" "$candidate" "$final"; do
  echo "FIX72_ANNOTATED_${version}=$(readlink -f "$root/sceneba_audit/$version/render_identity_smoke1_fix72/${scene}_annotated_ids.png")"
done
