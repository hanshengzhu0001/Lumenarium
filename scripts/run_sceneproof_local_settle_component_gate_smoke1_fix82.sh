#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
scene="bedroom_01"
source_version="v4_deepsearch"
incumbent="v5_sceneproof_pose_serialization_smoke1_fix76"
candidate="v5_sceneproof_local_settle_candidate_smoke1_fix82"
fix80_root="$root/sceneba_audit/$incumbent/true_mesh_com_smoke1_fix79/local_settle_oracle_fix80"
fix81="$fix80_root/oracle_fix81.json"
audit="$root/sceneba_audit/$candidate"
log_root="logs/sceneproof_local_settle_component_gate_smoke1_fix82"
manifest="/tmp/sceneproof_local_settle_component_gate_smoke1_fix82.txt"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
incumbent_placement="$root/${scene}_${incumbent}_result/S4_layout_refinement/${scene}_${incumbent}_placement_info_s4.json"
candidate_dir="$root/${scene}_${candidate}_result/S4_layout_refinement"
candidate_placement="$candidate_dir/${scene}_${candidate}_placement_info_s4.json"
source_json="$(find "$root/${scene}_${source_version}_result/S3_pose_inference" -maxdepth 1 -type f -name '*_placement_info.json' -print -quit)"

test -s "$fix81" || { echo "Missing Fix81 audit: $fix81" >&2; exit 2; }
test -s "$incumbent_placement" || { echo "Missing Fix76 placement: $incumbent_placement" >&2; exit 2; }
test -s "$source_json" || { echo "Missing source S3 placement" >&2; exit 2; }
test -x "$blender" || { echo "Missing Blender: $blender" >&2; exit 2; }
mkdir -p "$audit/probes" "$log_root" "$candidate_dir"
printf '%s\n' "$scene" > "$manifest"

object_id="$($python - "$fix81" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1]))
rows = d.get("summary", {}).get("locally_promising_object_ids", [])
if len(rows) != 1:
    raise SystemExit(f"expected exactly one locally promising object, got {rows}")
print(rows[0])
PY
)"
probe="$audit/probes/${object_id}.json"
scene_log="$log_root/${object_id}.log"
tmp_output="/tmp/sceneproof_local_settle_fix82_${object_id}_$$"
mkdir -p "$tmp_output"

timeout "${SCENEPROOF_LOCAL_SETTLE_TIMEOUT:-1800}" env \
  CUDA_VISIBLE_DEVICES=0 \
  IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$incumbent_placement" \
  IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER=1 \
  IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_OBJECT_ID="$object_id" \
  IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_DURATION_SECONDS="${SCENEPROOF_LOCAL_SETTLE_DURATION_SECONDS:-1.0}" \
  IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_AUDIT_OUTPUT="$probe" \
  PYTHONUNBUFFERED=1 \
  LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
  "$blender" --background \
    --python modules/S4_blender_layout_and_corr.py -- \
    --obj_placement_info_json_path "$source_json" \
    --output_folder "$tmp_output" \
    > "$scene_log" 2>&1 < /dev/null
rm -rf -- "$tmp_output"
test -s "$probe" || { echo "Missing local settle probe: $probe" >&2; exit 1; }

"$python" sceneproof_local_settle_materialize_fix82.py \
  --incumbent "$incumbent_placement" \
  --probe "$probe" \
  --out "$candidate_placement"

"$python" eval_physical_realizability.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --versions "$incumbent,$candidate" \
  --geometry-version v4_deepsearch \
  --baseline-version "$incumbent" \
  --metrics-out "$audit/physical.json" \
  --scene-csv "$audit/physical_scenes.csv" \
  --object-csv "$audit/physical_objects.csv" \
  --report-out "$audit/physical.txt"

"$python" eval_gt_metrics.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --versions "$incumbent,$candidate" \
  --min-visible-mask-area 8000 \
  --min-visible-bbox-size 0 \
  --batch-logs logs \
  --metrics-out "$audit/gt_8000.json" \
  --manifest-out "$audit/gt_manifest_8000.json"

"$python" sceneproof_local_settle_component_gate_fix82.py \
  --probe "$probe" \
  --physical "$audit/physical.json" \
  --gt "$audit/gt_8000.json" \
  --incumbent-version "$incumbent" \
  --candidate-version "$candidate" \
  --out "$audit/component_gates.json"

cat "$audit/physical.txt"
echo "FIX82_PROBE=$(readlink -f "$probe")"
echo "FIX82_CANDIDATE=$(readlink -f "$candidate_placement")"
echo "FIX82_PHYSICAL=$(readlink -f "$audit/physical.json")"
echo "FIX82_GT=$(readlink -f "$audit/gt_8000.json")"
echo "FIX82_GATES=$(readlink -f "$audit/component_gates.json")"
echo "FIX82_LOG=$(readlink -f "$scene_log")"
