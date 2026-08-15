#!/usr/bin/env bash
# SceneProof Fix98 — the same scaling-chain attribution, with its four defects fixed.
#
# The exact identity it rests on
# -----------------------------
# Line 7277 stores length = obj.dimensions, and a Blender object's dimensions is its
# local bounding box times its scale.  So
#
#     retrieved asset's native size  =  length / scale
#
# exactly, and three comparable boxes follow per object:
#
#     pcd_obb_size     what depth observed
#     length / scale   what the asset natively is
#     length           what was rendered
#
# What the first run got wrong, found by reading its own output
# ------------------------------------------------------------
# 1. gaming_table_2 occupies 55 per cent of the casino frame and vanished from the
#    report, because every test was one of internal consistency and its chain is
#    self-consistent: rendered 5.80 x 4.19 x 0.92 reproduces observed 5.79 x 4.19 x
#    0.92.  It is consistently wrong.  Dropping Fix96's peer tests had removed the
#    only external reference.  Restored more sharply: objects sharing one
#    retrieved_asset have the same native size BY CONSTRUCTION, so their rendered
#    sizes are directly comparable.  gaming_table_2 is 2.22times the median of its
#    two same-asset peers.  Fix96 compared against a scene median volume instead,
#    which mixes bottles with sofas.
#
# 2. "native/observed = 5 means retrieval returned something five times too big"
#    was wrong.  ceiling_fan_0: observed 0.27 x 0.45 x 0.39, asset 1.46 x 1.46 x
#    0.52, rendered 1.27 x 1.27 x 0.29.  The 1.27 m fan is right; the 0.45 m
#    observation is wrong.  The ratio names a disagreement between two references
#    and never says which one is correct.  It is now a number, not a reason.
#
# 3. Reasons fired on objects the scale had already fixed.  trash_bin_0 retrieved a
#    1.72 m bin asset and the scale brought it to 0.47 m against an observed 0.48 m.
#    Every size reason is now stated about the rendered size, which is what reaches
#    the image.
#
# 4. The category-versus-asset-name test matched one way only, so bookshelf_0
#    retrieving 0_SM_Shelf_2 was flagged although Shelf is a correct asset for a
#    bookshelf.  Matching is bidirectional now.
#
# Why the stage taxonomy changed
#------------------------------
# Naming the mechanism that last touched the number put 82 per cent of the casino
# into scale_overwritten_by_group_consistency, which is true and useless.  Since
# rendered = native x scale is exact, there are only three decidable outcomes:
#
#     rendered_size_followed_the_observation    the render agrees with depth
#     rendered_size_followed_the_assetthe scale left the asset roughly as-is
#     rendered_size_followed_neither            the scale invented a third size
#
# Mechanisms - abstention, clamp bound, group overwrite, production branch - are kept
# as separate annotations, since an object can carry several at once.
#
# What the previous run already established
# -----------------------------------------
# For 41 per cent of livingroom_10 the scale is exactly one, meaning no scaling
# decision was made at all and the asset's native size went straight into the image.
# Four objects labelled bookshelf have observed boxes of 2 to 17 cm and each renders
# a1.9 m shelf, together about a fifth of the frame.  paper_cup_1 is a 2 m duct at
# 46.8 per cent of the street frame.  All of these are one mechanism, now counted.
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
TOP_K="${SCENEPROOF_CHAIN_TOP_K:-6}"

AUDIT_ROOT="$root/sceneba_audit/v5_sceneproof_scaling_chain_attribution_fix98"
mkdir -p "$AUDIT_ROOT"

echo "=== SCENEPROOF FIX98: SCALING CHAIN ATTRIBUTION (no Blender, no simulation) ==="
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
echo "FIX98 COMPLETE  $(date)"
echo "AUDIT_ROOT=$AUDIT_ROOT"
echo ""
echo "Reading guide:"
echo "  1. Read the three boxes.  observed= is what depth saw, native= is what the"
echo "     retrieved asset actually is, rendered= is what went into the image."
echo "  2. set_by= says which of the two references the rendered size followed, and"
echo "     to_observation= / to_asset= are the two distances behind that choice, as"
echo "     factors that are always at least 1.  followed_neither is the worst case:"
echo "     the scale agreed with nothing, which is pillow_16."
echo "  3. asset/observed= is a DISAGREEMENT, not a verdict.  It does not say which"
echo "     reference is wrong.  ceiling_fan_0 has 3.77 and its render is correct."
echo "  4. peer_median= is the sharp signal, because same-asset peers share a native"
echo "     size by construction.  Expect gaming_table_2 back in the casino list at"
echo "     about 2.2 times its peers; it was missing entirely from the last run."
echo "  5. branch= is recomputed from the production predicate at line 6410, not"
echo "     guessed.  small_object_pixel_bbox_path means the observed box was never"
echo "     used: that path matches the mask's pixel extent, which cannot tell a"
echo "     small nearby object from a large distant one."
echo "  6. The MECHANISM line counts objects whose size the asset determined against"
echo "     a disagreeing observation, with the frame area they cover.  It is a"
echo "     mechanism count, NOT an error count:ceiling_fan_0 belongs to it and is"
echo "     correct.  Deciding who deserved to win needs a reference this tool lacks."
echo "  7. Every reason still prints its share of the scene.  Anything above roughly"
echo "     half is a property of the pipeline, not a list of things to fix."
echo "  8. Still invisible here: wrong shape at plausible size, such as a chair"
echo "     retrieved as a curved sheet.  That needs the mesh or the render."
