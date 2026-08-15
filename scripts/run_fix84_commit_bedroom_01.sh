#!/usr/bin/env bash
# Promote the Fix84-gated single_sofa_chair_1 candidate to a scoped pose commit.
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
root="a10_reusable_results/paper30"
scene="bedroom_01"
incumbent_version="v5_sceneproof_pose_serialization_smoke1_fix76"
candidate_version="v5_sceneproof_local_settle_candidate_smoke1_fix82"
commit_version="v5_sceneproof_fix84_com_commit_${scene}"
blender="${IMAGINARIUM_BLENDER:-third_party/blender-4.3.2-linux-x64/blender}"

OUT="$root/${scene}_${commit_version}_result/S4_layout_refinement"
COMMIT="$OUT/${scene}_${commit_version}_placement_info_s4.json"
mkdir -p "$OUT"

incumbent="$root/${scene}_${incumbent_version}_result/S4_layout_refinement/${scene}_${incumbent_version}_placement_info_s4.json"
candidate="$root/${scene}_${candidate_version}_result/S4_layout_refinement/${scene}_${candidate_version}_placement_info_s4.json"
gates="$root/sceneba_audit/$candidate_version/component_gates_fix84.json"

for f in "$incumbent" "$candidate" "$gates"; do
    test -s "$f" || { echo "MISSING: $f" >&2; exit 2; }
done

# The prior run printed this foot line — double-check rather than trust memory.
passed="$("$PY" -c "
import json
with open('$gates',encoding='utf-8') as handle:
    print(int(json.load(handle).get('PASSED',False)))
" 2>/dev/null || echo 0)"
if [ "$passed" != "1" ]; then
    echo "FATAL Fix84 gates PASSED=$passed" >&2
    exit 3
fi
echo "=== COMMIT $scene: single_sofa_chair_1 ==="

# Promote candidate → committed placement.
"$PY" -c "
import json,pathlib,sys
candidate=json.loads(pathlib.Path('$candidate').read_text(encoding='utf-8'))
sc=candidate.get('sceneproof_local_settle_candidate')
if not isinstance(sc,dict):
    sys.exit('missing sceneproof_local_settle_candidate metadata')
sc['promoted']=True
sc['policy']='single_object_scoped_pose_commit'
sc['commit_version']='$commit_version'
p=pathlib.Path('$COMMIT')
p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(candidate,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(f'Wrote {p.resolve()}')
"
echo "OBJECT=single_sofa_chair_1 PROMOTED=True"

# Render beauty.
render="$OUT/${scene}_${commit_version}_render_beauty_256s.png"
env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$COMMIT" \
    IMAGINARIUM_S4_RENDER_ONLY_OUTPUT="$render" \
    IMAGINARIUM_S4_RENDER_ONLY_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}" \
    LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
    "$blender" --background \
        --python modules/S4_blender_layout_and_corr.py -- \
        --obj_placement_info_json_path "$candidate" \
        --output_folder /tmp/fix84_commit \
        > "$OUT/render.log" 2>&1 < /dev/null
if [ -s "$render" ]; then
    echo "RENDER $(stat --format=%s "$render") bytes $(readlink -f "$render")"
else
    echo "RENDER FAILED — tail of log:"
    tail -20 "$OUT/render.log"
    exit 4
fi

echo ""
echo "========================================"
echo "COMMIT COMPLETE  $(date)"
echo "COMMIT_PLACEMENT=$(readlink -f "$COMMIT")"
echo "COMMIT_RENDER=$(readlink -f "$render")"
echo ""
echo "To re-evaluate physical metrics, add '$commit_version' to the version list"
echo "and re-run the batch eval pipeline.  The candidate's metrics are already"
echo "known from Fix82Fix84; the commit placement is a byte-identical promotion of"
echo "that candidate plus promoted=True in metadata."
