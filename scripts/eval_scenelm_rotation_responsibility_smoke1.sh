#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"

PY="${SCENELM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
root="${SCENELM_RESULTS_ROOT:-a10_reusable_results/paper30}"
dataset="${SCENELM_DATASET_DIR:-asset_data/imaginarium_3d_scene_layout_dataset}"
manifest="${SCENELM_MANIFEST:-/tmp/v5_scenelm_smoke1.txt}"
candidate="${SCENELM_RESPONSIBILITY_CANDIDATE:-v5_scenelm_collision_witness_fix21}"
reference="${SCENELM_RESPONSIBILITY_REFERENCE:-v4_adam400_exact_collision_fix18_control}"
prefix="${SCENELM_RESPONSIBILITY_PREFIX:-v5_scenelm_rotation_resp_audit1}"
audit="${SCENELM_RESPONSIBILITY_AUDIT_DIR:-$root/sceneba_audit/$prefix}"

mkdir -p "$audit"

# First obtain an exact object-level physical responsibility table for the
# frozen candidate/reference pair.  This is CPU-only and does not rerun S4.
"$PY" eval_physical_realizability.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --versions "$reference,$candidate" \
  --geometry-version v4_deepsearch \
  --baseline-version "$candidate" \
  --metrics-out "$audit/baseline_physical.json" \
  --scene-csv "$audit/baseline_physical_scenes.csv" \
  --object-csv "$audit/baseline_physical_objects.csv" \
  --report-out "$audit/baseline_physical.ascii"

"$PY" scenelm_rotation_responsibility_audit.py \
  --saved-results "$root" \
  --dataset-dir "$dataset" \
  --scenes "$manifest" \
  --candidate-version "$candidate" \
  --reference-version "$reference" \
  --physical-objects "$audit/baseline_physical_objects.csv" \
  --output-prefix "$prefix" \
  --out "$audit/responsibility.json" \
  --object-csv "$audit/responsibility_objects.csv"

versions="$candidate"
for policy in \
  oracle_all \
  oracle_observed \
  oracle_ambiguous \
  oracle_released \
  oracle_collision_offender \
  oracle_support_child \
  oracle_free_root
do
  versions="$versions,${prefix}_${policy}"
done

"$PY" eval_gt_metrics.py \
  --dataset-dir "$dataset" \
  --saved-results "$root" \
  --scenes "$manifest" \
  --versions "$versions" \
  --min-visible-mask-area 8000 \
  --min-visible-bbox-size 0 \
  --batch-logs logs \
  --metrics-out "$audit/gt_8000.json" \
  --manifest-out "$audit/gt_manifest_8000.json"

"$PY" eval_physical_realizability.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --versions "$versions" \
  --geometry-version v4_deepsearch \
  --baseline-version "$candidate" \
  --metrics-out "$audit/physical.json" \
  --scene-csv "$audit/physical_scenes.csv" \
  --object-csv "$audit/physical_objects.csv" \
  --report-out "$audit/physical.ascii"

"$PY" scenelm_rotation_responsibility_finalize.py \
  --responsibility "$audit/responsibility.json" \
  --gt "$audit/gt_8000.json" \
  --physical "$audit/physical.json" \
  --candidate-version "$candidate" \
  --out "$audit/final_gates.json"

cat "$audit/physical.ascii"
echo "ROTATION_RESPONSIBILITY_AUDIT=$audit"
