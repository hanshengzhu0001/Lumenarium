#!/usr/bin/env bash
# SceneProof Fix100 — recover a defect Fix99 lost, and stop the corpus accusing the
# instances that are right.
#
# What Fix99 got right, kept unchanged
# ------------------------------------
# The thin-axis artefacts are gone.  curtain_0 and curtain_1 disappeared from every
# defect list, sign_2 lost its spurious size flags, and the bedroom mechanism count
# fell from six objects covering 18.9 per cent of the frame to zero, which means that
# figure had been entirely an artefact of aggregating by the worst edge.  The
# cross-scene corpus worked on its first run: a_SM_Frame04 appears 37 times across the
# 30 scenes and renders near 1.0 m, so the three bedroom frames at 3.51 m came out at
# 3.49 times the norm, the first time anything in this tool could see them.
#
# The regression this fixes
# -------------------------
# Measuring size by the cube root of the volume ratio has a blind spot: a rod and a
# blob of similar volume score close to one.  pen_0 renders 0.05 x 0.05 x 1.81, the
# vertical rod standing through the whole bedroom image, against an observed
# 0.06 x 0.44 x 0.08.  Their volumes differ by 2.14, so the factor was 1.29 and pen_0
# vanished from the report entirely.
#
# Size is now measured two independent ways and a defect is either of them:
#
#   volume factor       cube root of the volume ratio, catches uniformly wrong size
#   longest edge ratio  catches wrong extent at a similar volume
#
# An aspect ratio was tried for the second view and rejected: curtain_0's rendered
# aspect is 24.7 against an observed 3.58, so it brings the thin-axis artefact
# straight back, whereas its longest edges are equal.  Both views were checked against
# all twelve known cases before shipping; the union is correct on every one.
#
#   curtain_0     vol 1.90longest 1.00  -> silent, correctly
#   sign_2vol 1.56  longest 1.00  -> silent, correctly
#   trash_bin_0   vol 1.32  longest 0.98  -> silent, correctly
#   pen_0         vol 1.29  longest 4.11  -> FLAGGED, recovered
#   bookshelf_0   vol 3.41  longest 2.39  -> FLAGGED, kept
#   chair_2vol 0.52  longest 0.03  -> FLAGGED as far too small
#
# A consequence worth stating: ceiling_fan_0 returns to the mechanism list, since its
# longest edges differ by 3.24and the asset did determine its size.  That is correct.
# The 1.27 m fan is right and the 0.45 m observation is wrong, and this is exactly why
# the rollup is labelled a mechanism count and not an error count.  Fix99 had it drop
# out, which looked tidier but came from the same measure that lost pen_0.
#
# The corpus inversion this fixes
# -------------------------------
# Both the in-scene peer test and the corpus are MAJORITY references: they detect that
# an object differs from the others, not that it is wrong.  Two inversions appeared:
#
#   wall_mounted_picture_frame_3 renders at 1.47 m, the only correct frame in its
#   scene, and its in-scene peer median is 0.42because the other three are 3.51 m.
#   The corpus resolved this one correctly, at 1.46.
#
#   a_SM_CartonGarbage05 serves stack_of_chips in the casino at 1.34 m and
#   discarded_wooden_board in the street scene at 0.36 m.  Pooled by asset over17
#   instances the median is dominated by the casino's oversized chips, so the street
#   scene's correctly sized boards were flagged at 0.37 and sign_0 at 0.18.
#
# The corpus is therefore conditioned on the asset AND the category, which separates
# the two populations with no hand-written table.  The pooled figure is still reported
# as rendered_over_asset_only_corpus_median so the contamination stays visible.
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
TOP_K="${SCENEPROOF_CHAIN_TOP_K:-6}"

AUDIT_ROOT="$root/sceneba_audit/v5_sceneproof_scaling_chain_attribution_fix100"
CORPUS="$AUDIT_ROOT/asset_size_corpus.json"
mkdir -p "$AUDIT_ROOT"

echo "=== SCENEPROOF FIX100: SCALING CHAIN ATTRIBUTION (no Blender, no simulation) ==="
echo "Scenes:   $SCENES"
echo "Baseline: $BASELINE"
echo "Measure:  volume factor AND longest-edge ratio; corpus keyed on asset+category"
echo "Start:    $(date)"

echo ""
echo "--- building the cross-scene same-asset-and-category size corpus ---"
mapfile -t corpus_files < <(
    ls "$root"/*_"${BASELINE}"_result/S4_layout_refinement/*_placement_info_s4.json 2>/dev/null || true
)
echo "corpus placements found: ${#corpus_files[@]}"
if test "${#corpus_files[@]}" -eq 0; then
    echo "WARN no corpus placements; the cross-scene reference will abstain"
    rm -f "$CORPUS"
else
    "$python" sceneproof_scaling_chain_attribution_fix97.py \
        --emit-asset-corpus "$CORPUS" \
        --corpus-placement "${corpus_files[@]}"
fi

corpus_arg=()
if test -s "$CORPUS"; then
    corpus_arg=(--asset-corpus "$CORPUS")
fi

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
        "${corpus_arg[@]}" \
        --out-report "$audit/scaling_chain_attribution.json" \
        > "$audit/chain.log" 2>&1; then
        echo "  FAIL attribution, see $audit/chain.log"
        continue
    fi
    sed -n '/^CHAIN/,$p' "$audit/chain.log" | sed 's/^/     /'
done

echo ""
echo "========================================"
echo "FIX100 COMPLETE  $(date)"
echo "AUDIT_ROOT=$AUDIT_ROOT"
echo "CORPUS=$CORPUS"
echo ""
echo "Reading guide:"
echo "  1. to_observation now prints both views: (vol) is the cube root of the volume"
echo "     ratio and (longest) is the longest-edge ratio.  A defect is either one"
echo "     past the threshold.  Where (vol) is near 1 and (longest) is large the"
echo "     object has the right bulk and the wrong extent, which is pen_0."
echo "  2. pen_0 must be back in the bedroom list.  If it is not, the recovery"
echo "     failed and nothing else in this run should be trusted."
echo "  3. ceiling_fan_0 is expected back in the MECHANISM list at 3.24.  That is"
echo "     correct and is the standing example of the mechanism producing the right"
echo "     answer: the 1.27 m fan is right, the 0.45 m observation is wrong."
echo "  4. corpus_median= is now keyed on asset AND category, with n= the sample"
echo "     count for that pair.  discarded_wooden_board_6/7 and sign_0 in the street"
echo "     scene should NO LONGER be flagged by it; they were accused because the"
echo "     casino's oversized chips share their asset."
echo "  5. Both peer_median and corpus_median are majority references.  They say an"
echo "     object differs from the others, never that it is wrong.  Where the"
echo "     majority is wrong they accuse the minority that is right, which is"
echo "     wall_mounted_picture_frame_3 at 0.42."
echo "  6. size_was_set_by=undetermined_scaling_chain_incomplete counts objects with"
echo "     a zero or negative edge in a recorded box.  They are now abstained rather"
echo "     than given a meaningless factor; one per dense scene so far."
echo "  7. Still invisible here: wrong shape at plausible size and at plausible"
echo "     extent, such as a chair retrieved as a curved sheet of the same height."
