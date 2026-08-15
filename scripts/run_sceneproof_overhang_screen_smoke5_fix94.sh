#!/usr/bin/env bash
# SceneProof Fix94 — which objects actually hang off their support, and how badly?
#
# The visible defect in the frozen Fix61 baseline: some objects, pillows above
# all, rest with their centre of mass past the edge of the surface they sit on.
# Physically they would tip; on screen they read as floating.
#
# This run needs NO Blender and NO simulation, because the measurement already
# exists.  The Fix62 true-mesh COM audit recorded, per object,
#
#     com_signed_margin_msigned distance from the COM's vertical projection
#                              to the support polygon boundary (negative = outside)
#     stability_class          stable / marginal / unstable
#     support_polygon_area_m2  the measured contact region
#
# so the screen is a pure read plus geometry, seconds per scene.
#
# WHAT TO READ FIRST: the overhang distances.  This decides whether the repair is
# worth building at all.  A pillow whose COM is 5 mm past the mattress edge is
# invisible in a render and not worth touching.  One that is 8 cm past is the
# defect you can see.  The tool prints the distance and the height above the
# floor for every candidate, so the threshold is a judgement made on data rather
# than a number I picked.
#
# It also decides, per object, which of two repairs applies, from geometry alone:
#
#   translate  shortest horizontal nudge that brings the COM back inside the
#              support polygon.  Asserts "position error, it was meant to sit
#              here".  Keeps height and orientation, so the support contact gap is
#              unchanged by construction and no rotation-induced bounding-box
#              proxy artefact can arise.  Proposed only when the travel is within
#              budget and the nudge adds no new overlap.
#   tip        let it rotate off the edge and settle, possibly ending up flat,
#              leaning, or on the next surface down.  Asserts "physically
#              impossible position".  Needs simulation, handled in the next step.
#
# Nothing is moved here.  This is selection and routing only.
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
COM_AUDIT_ROOT="${SCENEPROOF_COM_AUDIT_ROOT:-$root/sceneba_audit/${BASELINE}/true_mesh_com_fix62}"
TOP_K="${SCENEPROOF_OVERHANG_TOP_K:-3}"
MARGIN="${SCENEPROOF_OVERHANG_MARGIN_M:-0.005}"
TRANSLATE_BUDGET="${SCENEPROOF_TRANSLATE_BUDGET_M:-0.15}"

AUDIT_ROOT="$root/sceneba_audit/v5_sceneproof_overhang_screen_fix94"
mkdir -p "$AUDIT_ROOT"

echo "=== SCENEPROOF FIX94: OVERHANG SCREEN (no Blender, no simulation) ==="
echo "Scenes:        $SCENES"
echo "Baseline:      $BASELINE"
echo "COM audit:     $COM_AUDIT_ROOT"
echo "top-K:         ${TOP_K}"
echo "Thresholds:    visibility ${MARGIN} m, translate budget ${TRANSLATE_BUDGET} m"
echo "Start:         $(date)"

missing=0
for scene in $SCENES; do
    echo ""
    echo "--- $scene ---"
    audit="$AUDIT_ROOT/$scene"
    mkdir -p "$audit"

    placement="$root/${scene}_${BASELINE}_result/S4_layout_refinement/${scene}_${BASELINE}_placement_info_s4.json"
    com_audit="$COM_AUDIT_ROOT/${scene}__${BASELINE}.json"
    if ! test -s "$placement"; then
        echo "SKIP $scene: missing baseline placement"
        continue
    fi
    if ! test -s "$com_audit"; then
        echo "SKIP $scene: missing Fix62 COM audit at $com_audit"
        echo "       run scripts/run_sceneproof_true_mesh_com_paper30_fix62.sh first"
        missing=$((missing + 1))
        continue
    fi

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
done

echo ""
echo "========================================"
echo "FIX94 COMPLETE  $(date)"
echo "AUDIT_ROOT=$AUDIT_ROOT"
if test "$missing" -gt 0; then
    echo "NOTE: $missing scene(s) lacked a Fix62 COM audit and were skipped."
fi
echo ""
echo "Reading guide:"
echo "  1. overhang= is the number that decides whether to build the repair."
echo "     Under about 1 cm it is not visible in a render; several centimetres is"
echo "     the defect you can see.If every candidate is under 1 cm, stop here."
echo "  2. action= splits the work.  translate is closed form and needs no"
echo "     simulation at all, so those candidates cost nothing.  Only tip needs"
echo "     Blender, and only for the objects listed."
echo "  3. excluded= should be dominated by com_inside_support_polygon.  A large"
echo "     com_abstained or com_uncertified count means the COM audit could not"
echo "     measure those objects, which is a separate problem from hanging."
