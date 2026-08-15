#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
root="${SCENEPROOF_RESULTS_ROOT:-$HOME/Lumenarium/a10_reusable_results/paper30}"
python="$HOME/.venvs/lumenarium-py311/bin/python"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"
baseline="${SCENEPROOF_FIX114_BASELINE_VERSION:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
source="${SCENEPROOF_FIX114_SOURCE_VERSION:-v4_deepsearch}"
geometry="${SCENEPROOF_FIX114_GEOMETRY_VERSION:-$source}"
target="${SCENEPROOF_FIX114_TARGET_VERSION:-v5_sceneproof_vertical_support_visual_paper30_fix114}"
manifest="${SCENEPROOF_MANIFEST:-$root/manifest.txt}"
audit="$root/sceneba_audit/$target"
log_root="$HOME/Lumenarium/logs/$target"
gpu0_id="${IMAGINARIUM_GPU0_ID:-0}"
gpu1_id="${IMAGINARIUM_GPU1_ID:-1}"
mkdir -p "$audit/transactions" "$log_root"

list0="$(mktemp /tmp/sceneproof_fix114_gpu0_XXXXXX)"
list1="$(mktemp /tmp/sceneproof_fix114_gpu1_XXXXXX)"
trap 'rm -f -- "$list0" "$list1"' EXIT
awk 'NR % 2 == 1' "$manifest" > "$list0"
awk 'NR % 2 == 0' "$manifest" > "$list1"

worker() {
  local gpu="$1" list="$2" scene source_json placement output_dir candidate transaction tmp rc
  while IFS= read -r scene; do
    test -n "$scene" || continue
    source_json="$(find "$root/${scene}_${source}_result/S3_pose_inference" -maxdepth 1 -type f -name '*_placement_info.json' -print -quit 2>/dev/null || true)"
    placement="$root/${scene}_${baseline}_result/S4_layout_refinement/${scene}_${baseline}_placement_info_s4.json"
    output_dir="$root/${scene}_${target}_result/S4_layout_refinement"
    candidate="$output_dir/${scene}_${target}_placement_info_s4.json"
    transaction="$audit/transactions/${scene}.json"
    mkdir -p "$output_dir"
    if test -s "$candidate" && test -s "$transaction"; then
      echo "CACHED scene=$scene gpu=$gpu"
      continue
    fi
    if ! test -s "$source_json" || ! test -s "$placement"; then
      echo "FAIL_INPUT scene=$scene gpu=$gpu"
      continue
    fi
    tmp="$(mktemp -d /tmp/sceneproof_fix114_${scene}_XXXXXX)"
    echo "START scene=$scene gpu=$gpu $(date)"
    set +e
    timeout 3600 env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$placement" \
      IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER=1 \
      IMAGINARIUM_SCENEPROOF_VERTICAL_TRANSACTION_AUTO_DISCOVER=1 \
      IMAGINARIUM_SCENEPROOF_VERTICAL_MAX_CANDIDATES_PER_SCENE=5 \
      IMAGINARIUM_SCENEPROOF_VERTICAL_OVERHANG_MARGIN_M=0.005 \
      IMAGINARIUM_SCENEPROOF_VERTICAL_TRANSACTION_AUDIT_OUTPUT="$transaction" \
      IMAGINARIUM_SCENEPROOF_VERTICAL_TRANSACTION_PLACEMENT_OUTPUT="$candidate" \
      IMAGINARIUM_SCENEPROOF_VERTICAL_VISIBILITY_TOLERANCE=0.005 \
      IMAGINARIUM_SCENEPROOF_VERTICAL_VISIBILITY_RESOLUTION=256 \
      PYTHONUNBUFFERED=1 \
      LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
      "$blender" --background --python modules/S4_blender_layout_and_corr.py -- \
        --obj_placement_info_json_path "$source_json" --output_folder "$tmp" \
        > "$log_root/${scene}_gpu${gpu}.log" 2>&1 < /dev/null
    rc=$?
    set -e
    rm -rf -- "$tmp"
    echo "DONE scene=$scene gpu=$gpu rc=$rc $(date)"
  done < "$list"
}

echo "FIX114_PAPER30_START baseline=$baseline target=$target"
worker "$gpu0_id" "$list0" > "$log_root/gpu0.log" 2>&1 & p0=$!
worker "$gpu1_id" "$list1" > "$log_root/gpu1.log" 2>&1 & p1=$!
echo "FIX114_GPU0_PID=$p0"
echo "FIX114_GPU1_PID=$p1"
echo "FIX114_WORKER_LOGS=$log_root/gpu0.log,$log_root/gpu1.log"
status=0
wait "$p0" || status=1
wait "$p1" || status=1

"$python" - "$manifest" "$audit/transactions" "$audit/affected_scenes.txt" "$audit/summary.json" <<'PY'
import json, sys
from pathlib import Path
manifest, root, affected, summary = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
scenes=[x.strip() for x in manifest.read_text().splitlines() if x.strip()]
rows={}; failures=[]
unresolved={}
for scene in scenes:
    path=root/f'{scene}.json'
    if not path.is_file(): failures.append(scene); continue
    audit=json.load(open(path)); accepted=audit.get('accepted_object_ids', [])
    rejected=[row.get('object_id') for row in audit.get('transactions',[]) if not row.get('accepted')]
    rejected=[x for x in rejected if isinstance(x,str)]
    if rejected: unresolved[scene]=rejected
    rows[scene]={'accepted_object_ids':accepted,'transactions':len(audit.get('transactions',[])),'unresolved_object_ids':rejected}
affected.write_text('\n'.join(scene for scene,row in rows.items() if row['accepted_object_ids'])+'\n')
summary.write_text(json.dumps({'schema_version':'sceneproof_vertical_support_paper30_v2','scenes':rows,'failures':failures,'accepted_objects':sum(len(x['accepted_object_ids']) for x in rows.values()),'physical_unresolved':unresolved,'physical_unresolved_objects':sum(map(len,unresolved.values()))},indent=2)+'\n')
print(f"SCENES={len(rows)}/{len(scenes)} FAILURES={len(failures)} ACCEPTED_OBJECTS={sum(len(x['accepted_object_ids']) for x in rows.values())} AFFECTED_SCENES={sum(bool(x['accepted_object_ids']) for x in rows.values())} PHYSICAL_UNRESOLVED={sum(map(len,unresolved.values()))}")
if failures: raise SystemExit(2)
PY

"$python" eval_physical_realizability.py \
  --saved-results "$root" --scenes "$manifest" --versions "$baseline,$target" \
  --geometry-version "$geometry" --baseline-version "$baseline" \
  --metrics-out "$audit/physical.json" --scene-csv "$audit/physical_scenes.csv" \
  --object-csv "$audit/physical_objects.csv" --report-out "$audit/physical.txt" \
  > "$audit/physical.log" 2>&1 || status=1
if [[ "${SCENEPROOF_FIX114_SKIP_GT_EVAL:-0}" == "1" ]]; then
  printf '{}\n' > "$audit/gt_8000.json"
else
  "$python" eval_gt_metrics.py \
    --saved-results "$root" --scenes "$manifest" --versions "$baseline,$target" \
    --min-visible-mask-area 8000 --min-visible-bbox-size 0 --batch-logs logs \
    --metrics-out "$audit/gt_8000.json" --manifest-out "$audit/gt_manifest_8000.json" \
    > "$audit/gt.log" 2>&1 || status=1
fi

if test -s "$audit/affected_scenes.txt" && [[ "${SCENEPROOF_FIX114_SKIP_RENDER:-0}" != "1" ]]; then
  SCENEPROOF_MANIFEST="$audit/affected_scenes.txt" \
  SCENEPROOF_CERTIFIED_VERSION="$target" \
  SCENEPROOF_RENDER_LOG_ROOT="$log_root/render" \
  SCENEPROOF_RENDER_SAMPLES=256 \
  bash scripts/render_sceneproof_certified_paper30.sh \
    > "$log_root/render.log" 2>&1 || status=1
fi

"$python" - "$audit/physical.json" "$audit/gt_8000.json" "$audit/summary.json" "$baseline" "$target" "$audit/final_eval.json" <<'PY'
import json, sys
physical,gt,summary=(json.load(open(p)) for p in sys.argv[1:4]); baseline,target=sys.argv[4:6]
b=physical['versions'][baseline]['aggregate']; t=physical['versions'][target]['aggregate']
d={}
for f in ('collision','support','plane','semantic'):
 x=b.get('families',{}).get(f,{}).get('score'); y=t.get('families',{}).get(f,{}).get('score'); d[f]=None if x is None or y is None else y-x
unresolved=summary.get('physical_unresolved',{})
before_macro=b.get('headline_macro_realizability')
after_macro=t.get('headline_macro_realizability')
macro_delta=None if before_macro is None or after_macro is None else after_macro-before_macro
out={'schema_version':'sceneproof_vertical_support_final_eval_v2','baseline':baseline,'target':target,'accepted_objects':summary['accepted_objects'],'physical_family_deltas':d,'physical_macro_delta':macro_delta,'failures':summary['failures'],'physical_unresolved':unresolved,'physical_unresolved_objects':summary.get('physical_unresolved_objects',0),'passed':not summary['failures'] and not unresolved,'decision':'pass' if not summary['failures'] and not unresolved else 'visibility_or_physical_unresolved','positioning':'qualitative_true_mesh_support_variant; Fix61 remains quantitative baseline'}
open(sys.argv[6],'w').write(json.dumps(out,indent=2)+'\n')
macro_text='n/a' if macro_delta is None else f'{macro_delta:+.9f}'
print(f"ACCEPTED_OBJECTS={out['accepted_objects']} PHYSICAL_MACRO_DELTA={macro_text} FAMILY_DELTAS={json.dumps(d,sort_keys=True)}")
print(f"PHYSICAL_UNRESOLVED={out['physical_unresolved_objects']} PASSED={out['passed']} DECISION={out['decision']}")
PY

archive="${SCENEPROOF_FIX114_RENDER_ARCHIVE:-$HOME/sceneproof_vertical_support_visual_paper30_fix114_renders.tar.gz}"
if [[ "${SCENEPROOF_FIX114_SKIP_COMPARISON_ARCHIVE:-0}" != "1" ]]; then
  collection="$root/sceneproof_vertical_support_visual_paper30_fix114_renders"
  mkdir -p "$collection"
  while IFS= read -r scene; do
    test -n "$scene" || continue
    mkdir -p "$collection/$scene"
    cp "$root/${scene}_${baseline}_result/S4_layout_refinement/${scene}_${baseline}_render_simu.png" "$collection/$scene/00_fix61.png"
    cp "$root/${scene}_${target}_result/S4_layout_refinement/${scene}_${target}_render_simu.png" "$collection/$scene/01_vertical_support.png"
  done < "$audit/affected_scenes.txt"
  cp "$audit/final_eval.json" "$collection/final_eval.json"
  tar -czf "$archive" -C "$root" "$(basename "$collection")"
fi

echo "FIX114_FINISHED status=$status"
echo "FIX114_FINAL_EVAL=$(readlink -f "$audit/final_eval.json")"
echo "FIX114_AFFECTED_SCENES=$(readlink -f "$audit/affected_scenes.txt")"
if [[ "${SCENEPROOF_FIX114_SKIP_COMPARISON_ARCHIVE:-0}" != "1" ]]; then
  echo "FIX114_RENDER_ARCHIVE=$(readlink -f "$archive")"
fi
exit "$status"
