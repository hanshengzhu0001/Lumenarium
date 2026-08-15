#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
root="$HOME/Lumenarium/a10_reusable_results/paper30"
python="$HOME/.venvs/lumenarium-py311/bin/python"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"
scene="bedroom_01"
baseline="v5_sceneproof_collision_partial_commit_certified_paper30_fix61"
source="v4_deepsearch"
target="v5_sceneproof_sequential_vertical_transaction_smoke1_fix113"
fix112="$root/sceneba_audit/v5_sceneproof_vertical_first_contact_smoke1_fix112/projection_candidates.json"
audit="$root/sceneba_audit/$target"
output_dir="$root/${scene}_${target}_result/S4_layout_refinement"
placement="$root/${scene}_${baseline}_result/S4_layout_refinement/${scene}_${baseline}_placement_info_s4.json"
candidate="$output_dir/${scene}_${target}_placement_info_s4.json"
render="$output_dir/${scene}_${target}_render_simu.png"
source_json="$(find "$root/${scene}_${source}_result/S3_pose_inference" -maxdepth 1 -type f -name '*_placement_info.json' -print -quit)"
log_root="$HOME/Lumenarium/logs/$target"
mkdir -p "$audit" "$output_dir" "$log_root"
test -s "$fix112" && test -s "$placement" && test -s "$source_json"

ordered="$($python - "$fix112" <<'PY'
import json, sys
data=json.load(open(sys.argv[1]))
rows=[]
for row in data.get('selected', []):
    drop=row.get('vertical_first_contact_candidate')
    if row.get('recommended_action') == 'vertical_first_contact_drop' and drop:
        rows.append((float(drop['drop_m']), row['object_id']))
print(','.join(object_id for _, object_id in sorted(rows)))
PY
)"
test -n "$ordered" || { echo "No Fix112 vertical candidates" >&2; exit 2; }
echo "FIX113_ORDER=$ordered"

tmp_output="$(mktemp -d /tmp/sceneproof_fix113_XXXXXX)"
trap 'rm -rf -- "$tmp_output"' EXIT
env \
  CUDA_VISIBLE_DEVICES=0 \
  IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$placement" \
  IMAGINARIUM_SCENEPROOF_VERTICAL_TRANSACTION_OBJECT_IDS="$ordered" \
  IMAGINARIUM_SCENEPROOF_VERTICAL_TRANSACTION_AUDIT_OUTPUT="$audit/transaction.json" \
  IMAGINARIUM_SCENEPROOF_VERTICAL_TRANSACTION_PLACEMENT_OUTPUT="$candidate" \
  IMAGINARIUM_SCENEPROOF_VERTICAL_VISIBILITY_TOLERANCE=0.005 \
  IMAGINARIUM_SCENEPROOF_VERTICAL_VISIBILITY_RESOLUTION=256 \
  IMAGINARIUM_S4_RENDER_ONLY_OUTPUT="$render" \
  IMAGINARIUM_S4_RENDER_ONLY_SAMPLES=256 \
  IMAGINARIUM_S4_RENDER_ONLY_AUDIT="$audit/camera.json" \
  PYTHONUNBUFFERED=1 \
  LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
  "$blender" --background --python modules/S4_blender_layout_and_corr.py -- \
    --obj_placement_info_json_path "$source_json" \
    --output_folder "$tmp_output" \
    > "$log_root/${scene}_gpu0.log" 2>&1 < /dev/null

printf '%s\n' "$scene" > "$audit/manifest.txt"
"$python" eval_physical_realizability.py \
  --saved-results "$root" --scenes "$audit/manifest.txt" \
  --versions "$baseline,$target" --geometry-version "$source" \
  --baseline-version "$baseline" --metrics-out "$audit/physical.json" \
  --scene-csv "$audit/physical_scenes.csv" --object-csv "$audit/physical_objects.csv" \
  --report-out "$audit/physical.txt" > "$audit/physical.log" 2>&1
"$python" eval_gt_metrics.py \
  --saved-results "$root" --scenes "$audit/manifest.txt" \
  --versions "$baseline,$target" --min-visible-mask-area 8000 \
  --min-visible-bbox-size 0 --batch-logs logs \
  --metrics-out "$audit/gt_8000.json" --manifest-out "$audit/gt_manifest_8000.json" \
  > "$audit/gt.log" 2>&1

"$python" - "$audit/transaction.json" "$audit/physical.json" "$audit/gt_8000.json" "$baseline" "$target" <<'PY'
import json, sys
transaction, physical, gt = (json.load(open(path)) for path in sys.argv[1:4])
baseline, target = sys.argv[4:6]
b=physical['versions'][baseline]['aggregate']; t=physical['versions'][target]['aggregate']
deltas={}
for family in ('collision','support','plane','semantic'):
    x=b.get('families',{}).get(family,{}).get('score'); y=t.get('families',{}).get(family,{}).get('score')
    deltas[family]=None if x is None or y is None else y-x
print('ACCEPTED_OBJECTS=' + ','.join(transaction['accepted_object_ids']))
print('TRANSACTIONS=' + str(len(transaction['transactions'])))
print('FAMILY_DELTAS=' + json.dumps(deltas, sort_keys=True))
for row in transaction['transactions']:
    failed=[name for name, passed in row.get('gates',{}).items() if not passed]
    print(row['object_id'], 'accepted='+str(row['accepted']), 'drop_m='+str(row.get('drop_m')), 'supporter='+str(row.get('supporter_id')), 'failed='+','.join(failed))
PY

echo "FIX113_TRANSACTION=$(readlink -f "$audit/transaction.json")"
echo "FIX113_PLACEMENT=$(readlink -f "$candidate")"
echo "FIX113_RENDER=$(readlink -f "$render")"
echo "FIX113_PHYSICAL=$(readlink -f "$audit/physical.json")"
echo "FIX113_GT=$(readlink -f "$audit/gt_8000.json")"
