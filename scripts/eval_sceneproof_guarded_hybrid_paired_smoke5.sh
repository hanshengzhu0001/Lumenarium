#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
manifest="/tmp/sceneproof_guarded_hybrid_paired_smoke5.txt"
control="v5_sceneproof_smooth_control_smoke5_fix15"
candidate="v5_sceneproof_guarded_hybrid_smoke5_fix15"
root="a10_reusable_results/paper30"
audit="$root/sceneba_audit/$candidate"
mkdir -p "$audit"

SCENEPROOF_SCHUR_MANIFEST="$manifest" \
SCENEPROOF_SCHUR_VERSION="$candidate" \
  bash scripts/eval_sceneproof_full_so3_guarded_schur_smoke5.sh

"$PY" sceneproof_guarded_hybrid_smoke5_audit.py \
  --aggregate "$root/sceneba_audit/${candidate}.json" \
  --out "$audit/protocol.json"

"$PY" eval_physical_realizability.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --versions "$control,$candidate" \
  --geometry-version v4_deepsearch \
  --baseline-version "$control" \
  --runtime-log "$control=logs/$control" \
  --runtime-log "$candidate=logs/$candidate" \
  --metrics-out "$audit/physical.json" \
  --scene-csv "$audit/physical_scenes.csv" \
  --object-csv "$audit/physical_objects.csv" \
  --report-out "$audit/physical.ascii"

"$PY" eval_gt_metrics.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --versions "$control,$candidate" \
  --min-visible-mask-area 8000 \
  --min-visible-bbox-size 0 \
  --batch-logs logs \
  --metrics-out "$audit/gt_8000.json" \
  --manifest-out "$audit/gt_manifest_8000.json"

"$PY" scenelm_audit.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --physical "$audit/physical.json" \
  --gt "$audit/gt_8000.json" \
  --reference-version "$control" \
  --scenelm-version "$candidate" \
  --physical-margin 0.005 \
  --physical-component-margin 0.005 \
  --rotation-margin 0.01 \
  --translation-margin 0.005 \
  --minimum-speedup 0.6666667 \
  --out "$audit/paired_gates.json"

"$PY" sceneproof_guarded_hybrid_final_gate.py \
  --protocol "$audit/protocol.json" \
  --paired "$audit/paired_gates.json" \
  --out "$audit/final_gates.json"

cat "$audit/physical.ascii"
echo "FINAL_GATES=$audit/final_gates.json"
