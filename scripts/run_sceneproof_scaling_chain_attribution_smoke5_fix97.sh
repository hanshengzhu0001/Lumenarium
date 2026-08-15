#!/usr/bin/env bash
# SceneProof Fix97 — attribute every visible size defect by arithmetic.
#
# Why Fix96 needed replacing rather than extending
# ------------------------------------------------
# Fix96's camera ranking was right and is kept: the two worst offenders in the whole
# Smoke5 set are pillow_16 at 81.6 per cent of the frame and paper_cup_1 at 46.8 per
# cent.  Its stage= column was wrong on exactly those two, in opposite directions,
# and its third detector fired on 80to 90 per cent of every scene.
#
#   gaming_table_2   called an asset problem.  Its rendered 5.80 x 4.19 x 0.92
#                    reproduces its observed box 5.79 x 4.19 x 0.92, so scaling
#                    honoured depth and the observed box is the wrong quantity.
#   paper_cup_1      called a depth-driven scaling problem.  Its observed box is
#                    0.11 x 0.04 x 0.03, a correct paper cup, and its scale is
#                    exactly one, so no scaling decision was made at all.
#   the clamp filter defined as "three or more objects share one scale value".  That
#                    detects scene-graph group membership, because lines 7099 to
#                    7126 overwrite every group member with the group's most
#                    frequent scale.  Withdrawn.
#
# The identity this run is built on
# ---------------------------------
# Line 7277 stores length = obj.dimensions, and a Blender object's dimensions is its
# local bounding box times its scale.  So
#
#     retrieved asset's native size  =  length / scale
#
# exactly.  Three boxes can then be compared per object, and the odd one out names
# the stage without any guessing:
#
#     pcd_obb_size     what depth observed
#     length / scale   what the asset natively is
#     length           what was rendered
#
# On the five known cases this reads: pillow_16's asset is natively 2.76 x 1.30 x
# 0.69, a real sofa, while depth observed 1.07 x 0.25 x 0.25, a correct pillow;
# paper_cup_1's asset is natively a 2 m duct; pen_0's is a 1.81 m lamp, which is the
# vertical rod in the render; stack_of_chips_3's is a 0.98 m garbage carton that
# should have shrunk by thirtyfold and instead grew by 1.37.
#
# Flag rates are printed beside every count, so a detector that fires on most of the
# scene is visible without having to be looked for.  That is the lesson from Fix96.
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
TOP_K="${SCENEPROOF_CHAIN_TOP_K:-6}"

AUDIT_ROOT="$root/sceneba_audit/v5_sceneproof_scaling_chain_attribution_fix97"
mkdir -p "$AUDIT_ROOT"

echo "=== SCENEPROOF FIX97: SCALING CHAIN ATTRIBUTION (no Blender, no simulation) ==="
echo "Scenes:   $SCENES"
echo "Baseline: $BASELINE"
echo "Chain:    pcd_obb_size  ->  length/scale (asset native)  ->  length (rendered)"
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

    if ! "$python" sceneproof_scaling_chain_attribution_fix97.py \
        --scene "$scene" \
        --placement "$placement" \
        --top-k "$TOP_K" \
        --out-report "$audit/scaling_chain_attribution.json" \
        > "$audit/chain.log" 2>&1; then
        echo "  FAIL attribution, see $audit/chain.log"
        continue
    fi
    sed -n '/^CHAIN/,$p' "$audit/chain.log" | sed 's/^/     /'
done

echo ""
echo "========================================"
echo "FIX97 COMPLETE  $(date)"
echo "AUDIT_ROOT=$AUDIT_ROOT"
echo ""
echo "Reading guide:"
echo "  1. Read the three boxes on the second line of each entry.  observed= is what"
echo "     depth saw, native= is what the retrieved asset actually is, rendered= is"
echo "     what went into the image.  Whichever disagrees with the other two is the"
echo "     stage at fault, and set_by= names it."
echo "  2. native/observed is the retrieval size error on its own.  A value near 1"
echo "     means retrieval returned an asset of about the right size, so any defect"
echo "     is in the scale factor.  A value of 5 means retrieval returned something"
echo "     five times too big and no scale factor could have rescued the frame."
echo "  3. branch= is recomputed from the production predicate at line 6410, not"
echo "     guessed.  small_object_pixel_bbox_path means the observed box was never"
echo "     used: that path matches the mask's pixel extent, which cannot tell a"
echo "     small nearby object from a large distant one.  Every object on that path"
echo "     inherits its size from whatever asset retrieval happened to return."
echo "  4. Every reason now prints its share of the scene.  Anything above roughly"
echo "     half is not a triage signal and should be read as a property of the"
echo "     pipeline rather than a list of things to fix."
echo "  5. Still invisible here: wrong shape at plausible size, such as a chair"
echo "     retrieved as a curved sheet.  That needs the mesh or the render."
