#!/usr/bin/env bash
set -euo pipefail

cd "${SCENEBA_REPO_ROOT:-$HOME/Lumenarium}"

python_bin="${SCENEBA_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
manifest="${SCENEBA_YAW_VLM_MANIFEST:-/tmp/sceneba_yaw_vlm_smoke5.txt}"
target_version="${SCENEBA_YAW_VLM_TARGET_VERSION:-v4_yaw_vlm_contact_v1}"
results_root="${SCENEBA_RESULTS_ROOT:-a10_reusable_results/paper30}"
dataset_dir="${SCENEBA_DATASET_DIR:-asset_data/imaginarium_3d_scene_layout_dataset}"
gt_metrics="${SCENEBA_GT_METRICS:-a10_reusable_results/paper30/sceneba_audit/repair_smoke5_gt_8000.json}"
audit_root="${results_root}/sceneba_audit"

mkdir -p "$audit_root" logs

"$python_bin" sceneba_yaw_oracle.py \
  --saved-results "$results_root" \
  --dataset-dir "$dataset_dir" \
  --scenes "$manifest" \
  --gt-metrics "$gt_metrics" \
  --match-version v4 \
  --reference-version v4 \
  --audit-version "$target_version" \
  --out "${audit_root}/${target_version}_smoke5.json" \
  2>&1 | tee "logs/${target_version}_oracle_smoke5.log"

"$python_bin" sceneba_yaw_vlm_audit.py \
  --yaw-audit "${audit_root}/${target_version}_smoke5.json" \
  --out "${audit_root}/${target_version}_vlm_audit_smoke5.json" \
  2>&1 | tee "logs/${target_version}_vlm_audit_smoke5.log"

"$python_bin" eval_physical_realizability.py \
  --saved-results "$results_root" \
  --scenes "$manifest" \
  --versions "v4,${target_version}" \
  --geometry-version v4_deepsearch \
  --baseline-version v4 \
  --metrics-out "${audit_root}/${target_version}_physical_smoke5.json" \
  --scene-csv "${audit_root}/${target_version}_physical_scenes.csv" \
  --object-csv "${audit_root}/${target_version}_physical_objects.csv" \
  --report-out "${audit_root}/${target_version}_physical.txt" \
  2>&1 | tee "logs/${target_version}_physical_smoke5.log"

"$python_bin" sceneba_finalize_yaw_vlm_gates.py \
  --vlm-audit "${audit_root}/${target_version}_vlm_audit_smoke5.json" \
  --physical "${audit_root}/${target_version}_physical_smoke5.json" \
  --baseline-version v4 \
  --audit-version "$target_version" \
  --out "${audit_root}/${target_version}_final_gates_smoke5.json" \
  2>&1 | tee "logs/${target_version}_final_gates_smoke5.log"
