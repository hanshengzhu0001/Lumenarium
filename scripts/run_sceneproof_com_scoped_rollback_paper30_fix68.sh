#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
root="a10_reusable_results/paper30"
manifest="$root/manifest.txt"
incumbent="v5_sceneproof_fix43_smooth_paper30_fix61"
candidate="v5_sceneproof_collision_partial_commit_certified_paper30_fix61"
target="v5_sceneproof_com_scoped_rollback_paper30_fix68"
source_audit="$root/sceneba_audit/$candidate"
oracle="$source_audit/true_mesh_com_paper30_fix66/protocol.json"
audit="$root/sceneba_audit/$target"
mkdir -p "$audit"

"$PY" sceneproof_com_scoped_rollback_materialize.py \
  --saved-results "$root" --manifest "$manifest" \
  --incumbent-version "$incumbent" --candidate-version "$candidate" \
  --target-version "$target" --protocol "$oracle" \
  --out "$audit/materialization.json" \
  --runtime-jsonl "$audit/materialization_runtime.jsonl"

"$PY" eval_physical_realizability.py \
  --saved-results "$root" --scenes "$manifest" \
  --versions "$incumbent,$candidate,$target" \
  --geometry-version v4_deepsearch --baseline-version "$incumbent" \
  --metrics-out "$audit/physical.json" \
  --scene-csv "$audit/physical_scenes.csv" \
  --object-csv "$audit/physical_objects.csv" \
  --report-out "$audit/physical.txt"

"$PY" eval_gt_metrics.py \
  --saved-results "$root" --scenes "$manifest" \
  --versions "$incumbent,$candidate,$target" \
  --min-visible-mask-area 8000 --min-visible-bbox-size 0 \
  --batch-logs logs --metrics-out "$audit/gt_8000.json" \
  --manifest-out "$audit/gt_manifest_8000.json"

"$PY" sceneba_paired_audit.py \
  --gt-metrics "$audit/gt_8000.json" \
  --physical-metrics "$audit/physical.json" \
  --baseline "$incumbent" --candidate "$target" \
  --samples 10000 --rotation-margin -0.01 --translation-margin -0.005 \
  --out "$audit/paired_bootstrap_10000.json"

"$PY" sceneproof_com_scoped_rollback_gate.py \
  --materialization "$audit/materialization.json" \
  --protocol "$oracle" --physical "$audit/physical.json" \
  --gt "$audit/gt_8000.json" \
  --fix61-final-gates "$source_audit/final_gates.json" \
  --incumbent-version "$incumbent" --candidate-version "$candidate" \
  --target-version "$target" --out "$audit/final_gates.json"

cat "$audit/physical.txt"
echo "FIX68_MATERIALIZATION=$(readlink -f "$audit/materialization.json")"
echo "FIX68_FINAL_GATES=$(readlink -f "$audit/final_gates.json")"
echo "FIX68_BOOTSTRAP=$(readlink -f "$audit/paired_bootstrap_10000.json")"
