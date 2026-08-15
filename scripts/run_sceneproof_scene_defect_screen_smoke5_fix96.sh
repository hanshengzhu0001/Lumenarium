#!/usr/bin/env bash
# SceneProof Fix96 — find the defects the camera actually sees, and name the stage.
#
# The rendered Fix61 baseline settled what the overhang screen could not.  In four
# of five Smoke5 scenes the overhang candidates are invisible, for three different
# reasons: cropped outside the frame (livingroom_10), a top-down view where a nine
# centimetre overhang projects to a few pixels behind the table edge (official_01),
# or an arrangement that is simply plausible (a bin liner on a bin lid,
# streelitter_01).  The one visible case is a leaning floor lamp, which is exactly
# the one that should not be tipped.So gravity is not where the visible quality
# is lost.
#
# What is glaring is of three kinds, none of them fixable by moving an object:
#   wrong size     a sofa filling two thirds of the frame, a duct larger than the
#                  alley, a chair ten times its peers
#   wrong shape    casino chairs as thin curved sheets, a chair as a hollow frame
#   stray rod      a thin vertical pole, the same signature in two unrelated scenes
#
# This screen finds the first kind and locates the third, and it attributes them
# from the scaling chain the placement document already stores:
#
#   pcd_obb_size -> scale -> length
#
# A scale far from one means the factor was driven off a bad depth estimate, which
# is S3.  A sane scale with an absurd length means the asset's own dimensions are
# wrong.  One scale value shared across three or more objects is the clamp in
# estimate_scale_factors_for_object firing, which is direct evidence of a runaway
# estimate rather than a coincidence.
#
# It CANNOT see the second kind.  A chair retrieved as a curved sheet has plausible
# dimensions and the wrong geometry; only a look at the mesh or the render reveals
# that.  The report says so rather than implying full coverage.
#
# Ranking is by fraction of the frame occupied, computed through the actual scene
# camera.  The overhang screen ranked by overhang times height, which silently
# assumed a side view and put four invisible candidates on top.
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
TOP_K="${SCENEPROOF_DEFECT_TOP_K:-8}"

AUDIT_ROOT="$root/sceneba_audit/v5_sceneproof_scene_defect_screen_fix96"
mkdir -p "$AUDIT_ROOT"

echo "=== SCENEPROOF FIX96: VISIBLE DEFECT SCREEN (no Blender, no simulation) ==="
echo "Scenes:   $SCENES"
echo "Baseline: $BASELINE"
echo "Camera:   lens 30 mm, sensor 36 mm, 1024 px (matches setup_camera/render_scene)"
echo "Start:    $(date)"

for scene in $SCENES; do
    echo ""
    echo "--- $scene ---"
    audit="$AUDIT_ROOT/$scene"
    mkdir -p "$audit"

    placement="$root/${scene}_${BASELINE}_result/S4_layout_refinement/${scene}_${BASELINE}_placement_info_s4.json"
    if ! test -s "$placement"; then
        echo "SKIP $scene: missing baseline placement"
        continue
    fi

    if ! "$python" sceneproof_scene_defect_screen_fix96.py \
        --scene "$scene" \
        --placement "$placement" \
        --top-k "$TOP_K" \
        --out-report "$audit/scene_defect_screen.json" \
        > "$audit/screen.log" 2>&1; then
        echo "  FAIL screen, see $audit/screen.log"
        continue
    fi
    sed -n '/^DEFECTS/,$p' "$audit/screen.log" | sed 's/^/     /'
done

echo ""
echo "========================================"
echo "FIX96 COMPLETE  $(date)"
echo "AUDIT_ROOT=$AUDIT_ROOT"
echo ""
echo "Reading guide:"
echo "  1. screen= is the fraction of the frame the object occupies.  Anything"
echo "     above a few percent with a defect reason is a figure-ruining defect;"
echo "     the livingroom block should be tens of percent."
echo "  2. stage= is the actionable part.  s3_depth_driven_scaling means the fix is"
echo "     in depth estimation or in how pcd_obb_size drives the scale factor."
echo "     asset_dimensions_or_retrieval means the asset itself is wrong and the"
echo "     fix is upstream of S3."
echo "  3. scale_component_looks_clamped means three or more objects share one"
echo "     scale value, which is the SCALE_THRESHOLD clamp firing.  Those objects"
echo "     had runaway estimates that were merely capped, not corrected."
echo "  4. rod_like_extreme_aspect should catch the vertical pole in livingroom_10"
echo "     and official_01.  If both name the same retrieved_asset, one asset or"
echo "     one scaling path explains both scenes."
echo "  5. This screen cannot see wrong shape at the right size, such as a chair"
echo "     retrieved as a curved sheet.  Those need the mesh or the render."
