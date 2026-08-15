#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"

PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
root="a10_reusable_results/paper30"
manifest="${SCENEPROOF_FIX43_MANIFEST:-/tmp/sceneproof_fix43_vs_fix25_smoke1.txt}"
legacy="v4_legacy_sa5000_bench"
smooth="v5_sceneproof_smooth_control_paper30_fix25"
guarded="v5_sceneproof_guarded_hybrid_paper30_fix25"
certified="v5_sceneproof_postsim_component_certified_paper30_fix25"
fix43="v5_sceneproof_visual_rollback_smoke1_fix43"
audit="$root/sceneba_audit/fix43_vs_fix25_smoke1"
fix25_audit="$root/sceneba_audit/$certified"

printf '%s\n' bedroom_01 > "$manifest"
mkdir -p "$audit"

for version in "$legacy" "$smooth" "$fix43" "$certified"; do
  test -s "$(find "$root/bedroom_01_${version}_result/S4_layout_refinement" \
    -maxdepth 1 -name '*_placement_info_s4.json' -print -quit)" || {
    echo "Missing cached placement: bedroom_01 $version" >&2
    exit 2
  }
done

"$PY" eval_physical_realizability.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --versions "$legacy,$smooth,$fix43,$certified" \
  --geometry-version v4_deepsearch \
  --baseline-version "$smooth" \
  --runtime-log "$legacy=logs/paper30_s4_benchmark/legacy_sa5000" \
  --runtime-log "$smooth=logs/$smooth" \
  --runtime-log "$guarded=logs/$guarded" \
  --runtime-log "$fix43=logs/$fix43" \
  --runtime-log "${certified}_certificate=$fix25_audit/certificate_runtime.jsonl" \
  --runtime-log "${certified}_render=logs/${certified}_locked_camera_render" \
  --runtime-composite "$certified=$smooth+$guarded+${certified}_certificate+${certified}_render" \
  --metrics-out "$audit/physical.json" \
  --scene-csv "$audit/physical_scenes.csv" \
  --object-csv "$audit/physical_objects.csv" \
  --report-out "$audit/physical.ascii"

"$PY" eval_gt_metrics.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --versions "$legacy,$smooth,$fix43,$certified" \
  --min-visible-mask-area 8000 \
  --min-visible-bbox-size 0 \
  --batch-logs logs \
  --metrics-out "$audit/gt_8000.json" \
  --manifest-out "$audit/gt_manifest_8000.json"

cat "$audit/physical.ascii"
echo "FIX43_COMPARISON_DIR=$HOME/Lumenarium/$audit"
