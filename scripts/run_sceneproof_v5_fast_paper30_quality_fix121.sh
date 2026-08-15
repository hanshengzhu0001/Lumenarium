#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Lumenarium"
python="$HOME/.venvs/lumenarium-py311/bin/python"
root="$HOME/Lumenarium/a10_reusable_results/paper30"
manifest="$root/manifest.txt"
source=v4_deepsearch
baseline=v5_sceneproof_collision_partial_commit_certified_paper30_fix61
target=v5_sceneproof_fast_visual_paper30_fix121
audit="$root/sceneba_audit/$target"
versions="v1,v4_deepsearch,$target"
labels="v1,v4-deepsearch,v5-fast"
mkdir -p "$audit/physical_native"

echo "V5_FAST_PAPER30_START $(date)"
"$python" - "$root" "$manifest" "$baseline" <<'PY'
import sys
from pathlib import Path
from eval_gt_metrics import s4_path
from eval_physical_realizability import find_geometry_snapshot

root, manifest, baseline = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
scenes = [row.strip() for row in manifest.read_text().splitlines() if row.strip()]
for scene in scenes:
    for version in ("v1", "v4_deepsearch", baseline):
        s4_path(root, scene, version)
    for version in ("v1", "v4_deepsearch"):
        find_geometry_snapshot(root, scene, version)
print(f"PREFLIGHT scenes={len(scenes)} cached=v1,v4_deepsearch,fix61 ready=1")
PY

env SCENEPROOF_RESULTS_ROOT="$root" SCENEPROOF_MANIFEST="$manifest" \
  SCENEPROOF_FIX114_SOURCE_VERSION="$source" \
  SCENEPROOF_FIX114_GEOMETRY_VERSION="$source" \
  SCENEPROOF_FIX114_BASELINE_VERSION="$baseline" \
  SCENEPROOF_FIX114_TARGET_VERSION="$target" \
  SCENEPROOF_FIX114_SKIP_RENDER=1 SCENEPROOF_FIX114_SKIP_COMPARISON_ARCHIVE=1 \
  SCENEPROOF_FIX114_SKIP_GT_EVAL=1 \
  bash scripts/run_sceneproof_vertical_support_final_paper30_fix114.sh

"$python" eval_gt_metrics.py --saved-results "$root" --scenes "$manifest" \
  --versions "$versions" --min-visible-mask-area 8000 --min-visible-bbox-size 0 \
  --batch-logs logs --metrics-out "$audit/gt_8000.json" \
  --manifest-out "$audit/gt_manifest_8000.json"

for version in v1 v4_deepsearch "$target"; do
  geometry="$version"
  [[ "$version" == "$target" ]] && geometry="$source"
  "$python" eval_physical_realizability.py --saved-results "$root" \
    --scenes "$manifest" --versions "$version" --geometry-version "$geometry" \
    --baseline-version "$version" --collision-policy legacy \
    --metrics-out "$audit/physical_native/$version.json" \
    --scene-csv "$audit/physical_native/$version.scenes.csv" \
    --object-csv "$audit/physical_native/$version.objects.csv" \
    --report-out "$audit/physical_native/$version.txt"
done

mkdir -p "$audit/physical_relation_program"
"$python" eval_physical_realizability.py --saved-results "$root" \
  --scenes "$manifest" --versions "$target" --geometry-version "$source" \
  --baseline-version "$target" --collision-policy relation_program \
  --metrics-out "$audit/physical_relation_program/$target.json" \
  --scene-csv "$audit/physical_relation_program/$target.scenes.csv" \
  --object-csv "$audit/physical_relation_program/$target.objects.csv" \
  --report-out "$audit/physical_relation_program/$target.txt"

"$python" sceneproof_cross_version_quality_dashboard.py \
  --gt "$audit/gt_8000.json" --physical-dir "$audit/physical_native" \
  --versions "$versions" --labels "$labels" \
  --out-json "$audit/cross_version_quality.json" \
  --out-csv "$audit/cross_version_quality.csv" \
  --out-txt "$audit/cross_version_quality.txt"

echo "V5_FAST_PAPER30_FINISHED target=$target"
echo "V5_FAST_QUALITY=$audit/cross_version_quality.json"
echo "V5_RELATION_DIAGNOSTIC=$audit/physical_relation_program/$target.json"
