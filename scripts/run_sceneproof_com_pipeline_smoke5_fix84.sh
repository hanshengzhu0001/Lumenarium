#!/usr/bin/env bash
# SceneProof COM pipeline: Fix79→Fix84 across 5 scenes.
# Stage 1: COM factorization (per-scene: which objects are unstable?)
# Stage 2: local gravity settle probes (per-object, dual GPU)
# Stage 3: oracle re-evaluation (dtype-aware Fix81)
# Stage 4: component gates + witnessed exemption (Fix82+Fix84)
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"

# ---- configuration ----
SMOKE5_SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE_VERSION="${SCENEPROOF_BASELINE_VERSION:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
SOURCE_VERSION="${SCENEPROOF_SOURCE_VERSION:-v4_deepsearch}"
PHYSICAL_OBJECTS="$root/sceneba_audit/$BASELINE_VERSION/physical_objects.csv"
COM_TIMEOUT="${SCENEPROOF_COM_TIMEOUT:-3600}"
SETTLE_DURATION="${SCENEPROOF_LOCAL_SETTLE_DURATION_SECONDS:-1.0}"
SETTLE_TIMEOUT="${SCENEPROOF_LOCAL_SETTLE_TIMEOUT:-600}"
MANIFEST="/tmp/smoke5_com_pipeline_fix84.txt"
: > /tmp/smoke5_failures.txt
printf '%s\n' $SMOKE5_SCENES > "$MANIFEST"

# ---- helpers ----
com_audit_root() {  # scene candidate
    echo "$root/sceneba_audit/$2/true_mesh_com_smoke5_fix79/$1"
}
settle_probe_root() {  # scene candidate
    echo "$(com_audit_root "$1" "$2")/local_settle_oracle_fix80"
}

# ---- stage 1: COM factorization per scene ----
echo "=== STAGE 1: COM factorization (Fix79) ==="
declare -A SCENE_CANDIDATE_VERSION
declare -A SCENE_ACTION_AUDIT
declare -A SCENE_SETTLE_PROBE_ROOT
for scene in $SMOKE5_SCENES; do
    candidate=""
    for try in \
        "v5_sceneproof_pose_serialization_smoke1_fix76" \
        "v5_sceneproof_collision_partial_commit_certified_paper30_fix61" \
        "v4_deepsearch"; do
        if test -f "$root/${scene}_${try}_result/S4_layout_refinement/${scene}_${try}_placement_info_s4.json"; then
            candidate="$try"; break
        fi
    done
    if test -z "$candidate"; then
        echo "SKIP $scene: no placement found" >&2; continue
    fi
    SCENE_CANDIDATE_VERSION[$scene]="$candidate"
    com_root="$(com_audit_root "$scene" "$candidate")"
    mkdir -p "$com_root"

    echo "  $scene: candidate=$candidate"
    scene_manifest="/tmp/smoke5_com_fix79_${scene}.txt"
    printf '%s\n' "$scene" > "$scene_manifest"

    if test -s "$com_root/${scene}__${candidate}.json"; then
        echo "  $scene: COM audit cached"
    else
        timeout "$COM_TIMEOUT" env \
          SCENEPROOF_COM_MANIFEST="$scene_manifest" \
          SCENEPROOF_COM_SMOOTH_VERSION="$BASELINE_VERSION" \
          SCENEPROOF_COM_FINAL_VERSION="$candidate" \
          SCENEPROOF_COM_AUDIT_ROOT="$com_root" \
          SCENEPROOF_COM_LOG_ROOT="logs/com_smoke5_fix79_$scene" \
          SCENEPROOF_COM_PHYSICAL_OBJECTS="$PHYSICAL_OBJECTS" \
          IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
          bash scripts/run_sceneproof_true_mesh_com_paper30_fix64.sh \
          || echo "FAIL_COM $scene" >> /tmp/smoke5_failures.txt
    fi

    candidate_audit="$com_root/${scene}__${candidate}.json"
    if test -s "$candidate_audit"; then
        action_audit="$com_root/com_factorization_audit.json"
        "$python" sceneproof_com_action_audit_fix78.py \
          --baseline "$candidate_audit" \
          --candidate "$candidate_audit" \
          --out "$action_audit" \
          || echo "FAIL_CLASSIFY $scene" >> /tmp/smoke5_failures.txt
        SCENE_ACTION_AUDIT[$scene]="$action_audit"
    fi
    SCENE_SETTLE_PROBE_ROOT[$scene]="$(settle_probe_root "$scene" "$candidate")"
    rm -f "$scene_manifest"
done

# ---- stage 2: collect actionable objects + local settle ----
echo ""
echo "=== STAGE 2: local gravity settle (Fix80) ==="
ALL_OBJECTS="/tmp/smoke5_settle_objects.txt"
: > "$ALL_OBJECTS"
for scene in $SMOKE5_SCENES; do
    action_audit="${SCENE_ACTION_AUDIT[$scene]:-}"
    test -s "$action_audit" || continue
    candidate="${SCENE_CANDIDATE_VERSION[$scene]}"
    "$python" - "$action_audit" "$scene" "$candidate" >> "$ALL_OBJECTS" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
scene = sys.argv[2]
candidate = sys.argv[3]
for row in data.get("actionable_objects", []):
    # Only settle objects where the COM is explicitly measured unstable.
    # Objects with STABILITY=None have no mesh contact and cannot be
    # audited for improvement — gravity settle is wasted Blender time.
    if row.get("action") != "local_gravity_settle_probe_candidate":
        continue
    if row.get("stability_class") != "unstable":
        continue
    print(f"{scene}\t{candidate}\t{row['object_id']}")
PY
done

total=$(wc -l < "$ALL_OBJECTS")
echo "Total actionable gravity-settle probes: $total"
test "$total" -gt 0 || { echo "No actionable objects across 5 scenes. Done."; exit 0; }

# Group objects by (scene, candidate) for batch processing.
# Each scene gets ONE Blender invocation (not one per object).
# Objects within a scene are settled sequentially inside Blender,
# reusing the loaded scene.  Between objects, poses are restored.
declare -A SCENE_BATCH
while IFS=$'\t' read -r scene candidate object_id || test -n "$scene"; do
    scene="${scene%$'\r'}"; test -n "$scene" || continue
    key="${scene}|${candidate}"
    if test -n "${SCENE_BATCH[$key]:-}"; then
        SCENE_BATCH[$key]="${SCENE_BATCH[$key]},${object_id}"
    else
        SCENE_BATCH[$key]="${object_id}"
    fi
done < "$ALL_OBJECTS"

echo "Batch scenes: ${#SCENE_BATCH[@]}"
for key in "${!SCENE_BATCH[@]}"; do
    scene="${key%%|*}"; candidate="${key##*|}"
    echo "  $scene ($candidate): $(echo "${SCENE_BATCH[$key]}" | tr ',' ' ' | wc -w) objects"
done

echo ""
echo "=== STAGE 2: batch gravity settle (Fix80 batch mode) ==="
for key in "${!SCENE_BATCH[@]}"; do
    scene="${key%%|*}"; candidate="${key##*|}"
    objects="${SCENE_BATCH[$key]}"
    probe_root="$root/sceneba_audit/$candidate/true_mesh_com_smoke5_fix79/$scene/local_settle_oracle_fix80"
    mkdir -p "$probe_root"

    placement="$root/${scene}_${candidate}_result/S4_layout_refinement/${scene}_${candidate}_placement_info_s4.json"
    source_json="$(find "$root/${scene}_${SOURCE_VERSION}_result/S3_pose_inference" -maxdepth 1 -type f -name '*_placement_info.json' -print -quit)"
    log_dir="logs/smoke5_settle_fix80_batch"; mkdir -p "$log_dir"
    scene_log="$log_dir/${scene}.log"
    tmp_output="/tmp/smoke5_settle_batch_${scene}_$$"
    mkdir -p "$tmp_output"

    count=$(echo "$objects" | tr ',' ' ' | wc -w)
    echo "BATCH_SETTLE scene=$scene objects=$count $(date)"
    timeout "$SETTLE_TIMEOUT" env \
        CUDA_VISIBLE_DEVICES=0 \
        IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$placement" \
        IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER=1 \
        IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_OBJECT_IDS="$objects" \
        IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_DURATION_SECONDS="$SETTLE_DURATION" \
        IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_AUDIT_OUTPUT="$probe_root" \
        PYTHONUNBUFFERED=1 \
        LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
        "$blender" --background \
          --python modules/S4_blender_layout_and_corr.py -- \
          --obj_placement_info_json_path "$source_json" \
          --output_folder "$tmp_output" \
          > "$scene_log" 2>&1 < /dev/null && echo "BATCH_DONE $scene" || {
        echo "FAIL_BATCH $scene" >&2
        grep -E 'Local gravity settle|Traceback|Error:|FAIL' "$scene_log" | tail -20 || true
    }
    rm -rf -- "$tmp_output"
done

# ---- stage 3: Fix81 oracle re-evaluation per scene ----
echo ""
echo "=== STAGE 3: oracle re-evaluation (Fix81) ==="
declare -A SCENE_ORACLE
declare -A SCENE_PROMISING
for scene in $SMOKE5_SCENES; do
    candidate="${SCENE_CANDIDATE_VERSION[$scene]:-}"
    test -n "$candidate" || continue
    probe_root="$(settle_probe_root "$scene" "$candidate")"
    com_root="$(com_audit_root "$scene" "$candidate")"
    action_audit="$com_root/com_factorization_audit.json"
    oracle="$probe_root/oracle_fix81.json"
    report="$probe_root/oracle_fix81.txt"
    test -s "$action_audit" || continue

    "$python" sceneproof_local_settle_oracle_fix80.py \
      --action-audit "$action_audit" \
      --probe-root "$probe_root" \
      --out "$oracle" \
      --report "$report" || continue
    SCENE_ORACLE[$scene]="$oracle"

    promising=$("$python" - "$oracle" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ids = d.get("summary", {}).get("locally_promising_object_ids", [])
print(len(ids), *ids)
PY
)
    count=$(echo "$promising" | awk '{print $1}')
    if test "$count" -gt 0; then
        echo "  $scene: $count locally promising: $(echo "$promising" | cut -d' ' -f2-)"
        SCENE_PROMISING[$scene]="$promising"
    else
        echo "  $scene: 0 locally promising"
    fi
done

# ---- stage 4: Fix82 materialize + Fix84 gates ----
echo ""
echo "=== STAGE 4: component gates + witnessed exemption (Fix82+Fix84) ==="
SUMMARY="/tmp/smoke5_fix84_summary.json"
results="{}"
for scene in $SMOKE5_SCENES; do
    promising="${SCENE_PROMISING[$scene]:-}"
    test -n "$promising" || continue
    candidate="${SCENE_CANDIDATE_VERSION[$scene]}"
    gate_version="v5_sceneproof_local_settle_candidate_smoke5_fix84"
    object_list=$(echo "$promising" | cut -d' ' -f2-)
    probe_root="$(settle_probe_root "$scene" "$candidate")"

    for object_id in $object_list; do
        probe="$probe_root/${object_id}.json"
        audit="$root/sceneba_audit/${gate_version}_${scene}/${object_id}"
        gates_out="$audit/component_gates_fix84.json"
        test -s "$probe" || { echo "  SKIP $scene/$object_id: probe missing"; continue; }

        incumbent_placement="$root/${scene}_${candidate}_result/S4_layout_refinement/${scene}_${candidate}_placement_info_s4.json"
        cand_dir="$root/${scene}_${gate_version}_result/S4_layout_refinement"
        cand_placement="$cand_dir/${scene}_${gate_version}_placement_info_s4.json"
        mkdir -p "$cand_dir" "$audit"

        "$python" sceneproof_local_settle_materialize_fix82.py \
          --incumbent "$incumbent_placement" \
          --probe "$probe" \
          --out "$cand_placement" || continue

        scene_manifest="/tmp/smoke5_fix84_${scene}_${object_id}.txt"
        printf '%s\n' "$scene" > "$scene_manifest"
        "$python" eval_physical_realizability.py \
          --saved-results "$root" --scenes "$scene_manifest" \
          --versions "$candidate,$gate_version" \
          --geometry-version "$SOURCE_VERSION" \
          --baseline-version "$candidate" \
          --metrics-out "$audit/physical.json" \
          --scene-csv "$audit/physical_scenes.csv" \
          --object-csv "$audit/physical_objects.csv" \
          --report-out "$audit/physical.txt" 2>/dev/null || continue

        "$python" eval_gt_metrics.py \
          --saved-results "$root" --scenes "$scene_manifest" \
          --versions "$candidate,$gate_version" \
          --min-visible-mask-area 8000 --min-visible-bbox-size 0 \
          --batch-logs logs \
          --metrics-out "$audit/gt_8000.json" \
          --manifest-out "$audit/gt_manifest_8000.json" 2>/dev/null || continue

        "$python" sceneproof_local_settle_component_gate_fix84.py \
          --probe "$probe" \
          --physical "$audit/physical.json" \
          --physical-objects "$audit/physical_objects.csv" \
          --gt "$audit/gt_8000.json" \
          --incumbent-version "$candidate" \
          --candidate-version "$gate_version" \
          --allow-support-proxy-exemption \
          --out "$gates_out" 2>&1 || true

        passed=$("$python" -c "import json; d=json.load(open('$gates_out')); print(d.get('passed',False))")
        echo "  $scene/$object_id: passed=$passed"
        results=$(echo "$results" | "$python" -c "
import json, sys
r = json.loads(sys.stdin.read())
r['${scene}/${object_id}'] = {'passed': $passed, 'gates': '$gates_out'}
print(json.dumps(r))
")
        rm -f "$scene_manifest"
    done
done

echo ""
echo "================================================"
echo "SMOKE5 COM PIPELINE SUMMARY (Fix79→Fix84)"
echo "================================================"
echo "$results" | "$python" -m json.tool
echo "FAILURES: $(cat /tmp/smoke5_failures.txt 2>/dev/null || echo none)"
