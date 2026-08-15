#!/usr/bin/env bash
# SceneProof Fix94b — produce the true-mesh COM audit for Smoke5, then screen.
#
# Fix94 found no Fix62 COM audit on disk for any Smoke5 scene, so the measurement
# has to be made before the screen can read it.  This script is the minimal path
# to that: one Blender session per scene, read-only, no render, no simulation, no
# pose mutation.  It is the same Blender invocation Fix62 used, minus the double
# GPU fan-out, the responsibility audit and the physical_objects.csv dependency,
# none of which the screen needs.
#
# It also measures something not yet known: how long a Blender session takes to
# start, build the scene and run a whole-scene true-mesh COM audit.  That number
# decides whether the downstream tip-and-settle step fits the one-minute-per-scene
# budget, so it is printed per scene rather than hidden.
#
# Caching is by output file, so a re-run costs nothing for scenes already done.
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"

SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
SOURCE="${SCENEPROOF_GEOMETRY_VERSION:-v4_deepsearch}"
COM_AUDIT_ROOT="${SCENEPROOF_COM_AUDIT_ROOT:-$root/sceneba_audit/${BASELINE}/true_mesh_com_fix62}"
GPU="${SCENEPROOF_COM_GPU:-0}"
COM_TIMEOUT="${SCENEPROOF_COM_TIMEOUT:-1800}"
TOP_K="${SCENEPROOF_OVERHANG_TOP_K:-3}"
MARGIN="${SCENEPROOF_OVERHANG_MARGIN_M:-0.005}"
TRANSLATE_BUDGET="${SCENEPROOF_TRANSLATE_BUDGET_M:-0.15}"

AUDIT_ROOT="$root/sceneba_audit/v5_sceneproof_overhang_screen_fix94"
LOG_ROOT="logs/sceneproof_overhang_com_fix94b"
mkdir -p "$AUDIT_ROOT" "$COM_AUDIT_ROOT" "$LOG_ROOT"

test -x "$blender" || { echo "Missing Blender: $blender" >&2; exit 2; }

echo "=== SCENEPROOF FIX94b: TRUE-MESH COM AUDIT THEN OVERHANG SCREEN ==="
echo "Scenes:    $SCENES"
echo "Baseline:  $BASELINE"
echo "Geometry:  $SOURCE"
echo "COM audit: $COM_AUDIT_ROOT"
echo "Start:     $(date)"

for scene in $SCENES; do
    echo ""
    echo "--- $scene ---"
    audit="$AUDIT_ROOT/$scene"
    mkdir -p "$audit"

    placement="$root/${scene}_${BASELINE}_result/S4_layout_refinement/${scene}_${BASELINE}_placement_info_s4.json"
    com_audit="$COM_AUDIT_ROOT/${scene}__${BASELINE}.json"
    source_json="$(find "$root/${scene}_${SOURCE}_result/S3_pose_inference" \
        -maxdepth 1 -type f -name '*_placement_info.json' -print -quit 2>/dev/null || true)"

    if ! test -s "$placement"; then
        echo "SKIP $scene: missing baseline placement"
        continue
    fi

    if test -s "$com_audit"; then
        echo "  [1/2] COM audit already on disk, reusing it"
    else
        if test -z "$source_json" || ! test -s "$source_json"; then
            echo "SKIP $scene: no S3 pose inference input under" \
                "$root/${scene}_${SOURCE}_result/S3_pose_inference"
            continue
        fi
        echo "  [1/2] measuring true-mesh COM in Blender (read-only, no render) ..."
        tmp_output="$(mktemp -d "/tmp/sceneproof_com_${scene}_XXXXXX")"
        scene_log="$LOG_ROOT/${scene}.log"
        started="$(date +%s)"
        set +e
        timeout "$COM_TIMEOUT" env \
            CUDA_VISIBLE_DEVICES="$GPU" \
            IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$placement" \
            IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER=1 \
            IMAGINARIUM_SCENEPROOF_TRUE_MESH_COM_AUDIT_OUTPUT="$com_audit" \
            PYTHONUNBUFFERED=1 \
            LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
            "$blender" --background \
                --python modules/S4_blender_layout_and_corr.py -- \
                --obj_placement_info_json_path "$source_json" \
                --output_folder "$tmp_output" \
                > "$scene_log" 2>&1 < /dev/null
        rc=$?
        set -e
        elapsed=$(( $(date +%s) - started ))
        rm -rf -- "$tmp_output"
        if test "$rc" -ne 0 || ! test -s "$com_audit"; then
            echo "  FAIL COM audit rc=$rc after ${elapsed}s, see $scene_log"
            grep -E 'True-mesh COM|Traceback|RuntimeError|ValueError|Error:' \
                "$scene_log" | tail -20|| true
            continue
        fi
        echo "  BLENDER_COM_AUDIT_SECONDS=$elapsed   (budget probe for the tip step)"
    fi

    echo "  [2/2] screening overhang candidates ..."

    # Diagnostic pass: every candidate, with the consistency indicators shown.
    if ! "$python" sceneproof_overhang_screen_fix94.py \
        --scene "$scene" \
        --com-audit "$com_audit" \
        --placement "$placement" \
        --margin-threshold-m "$MARGIN" \
        --translate-budget-m "$TRANSLATE_BUDGET" \
        --top-k "$TOP_K" \
        --out-report "$audit/overhang_screen.json" \
        > "$audit/screen.log" 2>&1; then
        echo "  FAIL screen, see $audit/screen.log"
        continue
    fi
    sed -n '/^OVERHANG/,$p' "$audit/screen.log" | sed 's/^/     /'

    # Repair list: drop only those whose COM measurement contradicts their own
    # serialized geometry, because their overhang distance is unusable.
    if ! "$python" sceneproof_overhang_screen_fix94.py \
        --scene "$scene" \
        --com-audit "$com_audit" \
        --placement "$placement" \
        --margin-threshold-m "$MARGIN" \
        --translate-budget-m "$TRANSLATE_BUDGET" \
        --top-k "$TOP_K" \
        --require-consistent-com \
        --out-report "$audit/overhang_repair_list.json" \
        > "$audit/repair_list.log" 2>&1; then
        echo "  FAIL repair list, see $audit/repair_list.log"
        continue
    fi
    echo "     --- actionable repair list ---"
    sed -n '/^OVERHANG/,$p' "$audit/repair_list.log" | sed 's/^/     /'
done

echo ""
echo "========================================"
echo "FIX94b COMPLETE  $(date)"
echo "AUDIT_ROOT=$AUDIT_ROOT"
echo "COM_AUDIT_ROOT=$COM_AUDIT_ROOT"
echo ""
echo "Reading guide:"
echo "  1. Uniform density is assumed throughout.  Its consequences are checked in"
echo "     the before-and-after render, not guessed at beforehand: two geometric"
echo "     proxies were tried and withdrawn, because a fill-ratio floor fires on"
echo "     every legged furniture item and a COM-height ceiling split three"
echo "     identical pillows apart at an arbitrary line."
echo "  2. The only automatic filter is an impossibility test on the data: a mesh"
echo "     larger than its own bounding box (fill_ratio above 1), or a centre of"
echo "     mass outside it (com_height_ratio outside 0 to 1).  Those come from the"
echo "     voxel resolution floor thickening thin objects, and they make the"
echo "     overhang distance unusable, so those objects are abstained from."
echo "  3. overhang= is what decides whether the repair is worth building.  Under"
echo "     about 1 cm it is invisible in a render; several centimetres is not."
echo "  4. BLENDER_COM_AUDIT_SECONDS is a one-off: Fix61 poses are frozen, so the"
echo "     audit is cached from here on."
