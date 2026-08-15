#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"

PY="${SCENELM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
root="${SCENELM_RESULTS_ROOT:-a10_reusable_results/paper30}"
manifest="${SCENELM_MANIFEST:-/tmp/v5_scenelm_smoke1.txt}"
reference="${SCENELM_REFERENCE_VERSION:-v4_adam400_exact_collision_fix18_control}"
candidate="${SCENELM_VERSION:-v5_scenelm_collision_witness_polish45_diag1}"
freeze_all="${SCENELM_YAW_FREEZE_ALL_VERSION:-v5_scenelm_yaw_freeze_all_audit1}"
structural="${SCENELM_YAW_STRUCTURAL_VERSION:-v5_scenelm_yaw_structural_audit1}"
audit="${SCENELM_YAW_AUDIT_DIR:-$root/sceneba_audit/scenelm_yaw_noharm_audit1}"
versions="$reference,$candidate,$freeze_all,$structural"

mkdir -p "$audit"

"$PY" scenelm_yaw_noharm_materialize.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --source-version v4_deepsearch \
  --candidate-version "$candidate" \
  --freeze-all-version "$freeze_all" \
  --structural-version "$structural"

"$PY" eval_gt_metrics.py \
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
  --baseline-version "$reference" \
  --metrics-out "$audit/physical.json" \
  --scene-csv "$audit/physical_scenes.csv" \
  --object-csv "$audit/physical_objects.csv" \
  --report-out "$audit/physical.ascii"

cat "$audit/physical.ascii"
echo "YAW_NOHARM_AUDIT=$audit"
