#!/usr/bin/env bash
# SceneProof Fix99 — same attribution, with a corrected size measure and a cross-scene
# reference for families that are uniformly wrong.
#
# The measurement error this fixes
# --------------------------------
# Fix98 aggregated a box disagreement by taking the maximum over edges, which hands
# the verdict to the thinnest axis.  That axis is where depth estimation is least
# reliable and where the visual consequence is smallest, so the two largest bedroom
# and livingroom entries were false positives:
#
#   curtain_0   observed 1.62 x 0.62 x 2.22, rendered 1.62 x 0.09 x 2.22, 7.86% of
#               the frame.  The two long edges agree to three decimal places.  The
#               whole 6.62x came from 9 cm of curtain thickness against 62 cm.
#   curtain_1   the same, at 11.86% of the bedroom frame.
#   sign_2      3.78x, entirely 31 cm of sign thickness against an observed 8 cm.
#
# The replacement is the equivalent linear factor: the cube root of the volume ratio,
# which is dimensionally what a scale factor is and cannot be dominated by one axis.
# It is also exactly consistent with the chain, since rendered = native x scale means
# the render-to-asset factor is the cube root of the product of the scale components.
# Verified on every known case before shipping:
#
#   curtain_0        6.62x -> 1.90x   no longer flagged, correctly
#   sign_2           3.78x -> 1.57x   no longer flagged, correctly
#   ceiling_fan_0    3.27x -> 2.14x   no longer flagged, and it was always correct
#   bookshelf_0      5.82x -> 3.39x   STILL flagged, correctly
#   paper_cup_1     32.59x -> 23.4x   still flagged
#   stack_of_chips_9 96.76x -> 38.5x  still flagged
#   trash_bin_0      1.99x -> 1.32x   still not flagged
#
# A measure restricted to the two longest edges was tried first and rejected, because
# it scores bookshelf_0 at 2.86 and misses a genuine defect.
#
# The blind spot this closes
# --------------------------
# All three bedroom picture frames render at 3.51 m with an in-scene same-asset peer
# median of exactly 1.00.  A 3.5 m picture frame is grossly wrong and no internal
# consistency test and no in-scene peer test can see it, because the whole family is
# uniformly wrong.  Since one retrieved_asset has one native size everywhere, its
# rendered sizes across the other scenes of the corpus are a legitimate external
# reference, and it needs no hand-written size table.  This run therefore builds the
# corpus over every paper30 placement first, then screens each scene against it.
#
# The detector this demotes
# -------------------------
# retrieved_asset_name_does_not_name_its_category stayed at 15 to 48 per cent after
# the bidirectional fix, which is above the rate at which a signal can be triaged,
# and it has known false positives from plurals such as 0_steel_frame_shelves for a
# bookshelf and from uninformative names such as a_SM_Decor_6.  Its precision is not
# established, so it is now reported as a field and never counted as a defect.  Its
# stronger sibling, naming a different category present in the same scene, sits at 8
# to 12 per cent and is kept.
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
TOP_K="${SCENEPROOF_CHAIN_TOP_K:-6}"

AUDIT_ROOT="$root/sceneba_audit/v5_sceneproof_scaling_chain_attribution_fix99"
CORPUS="$AUDIT_ROOT/asset_size_corpus.json"
mkdir -p "$AUDIT_ROOT"

echo "=== SCENEPROOF FIX99: SCALING CHAIN ATTRIBUTION (no Blender, no simulation) ==="
echo "Scenes:   $SCENES"
echo "Baseline: $BASELINE"
echo "Measure:  equivalent linear factor = cube root of the volume ratio"
echo "Start:    $(date)"

echo ""
echo "--- building the cross-scene same-asset size corpus ---"
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
echo "FIX99 COMPLETE  $(date)"
echo "AUDIT_ROOT=$AUDIT_ROOT"
echo "CORPUS=$CORPUS"
echo ""
echo "Reading guide:"
echo "  1. observed= what depth saw, native= what the asset natively is, rendered="
echo "     what went into the image.  native is length/scale, which is exact."
echo "  2. to_observation= and to_asset= are equivalent linear factors, so a thin"
echo "     axis alone can no longer decide anything.  worst_axis= is the old"
echo "     measure, kept beside them: where worst_axis is large and to_observation"
echo "     is near 1, the disagreement is confined to one thin axis and is not a"
echo "     defect.  Expect exactly that on curtain_0, curtain_1 and sign_2."
echo "  3. corpus_median= is the new reference, with n= its sample count across all"
echo "     paper30 scenes.  The three bedroom picture frames render at 3.51 m with"
echo "     an in-scene peer median of1.00; if the corpus disagrees, that is the"
echo "     first time anything in this tool can see them."
echo "  4. peer_median= remains the sharpest in-scene signal at 1 to 3 per cent:"
echo "     gaming_table_2 at 2.22, chandelier_2 at 5.13, small_potted_plant_0 at"
echo "     2.25.  It is silent by construction on a uniformly wrong family."
echo "  5. The MECHANISM line is a mechanism count, NOT an error count.  Deciding"
echo "     whether the asset deserved to win needs a reference this tool lacks,"
echo "     which is precisely what the corpus starts to supply."
echo "  6. The weak asset-name verdict is now printed separately and is NOT part of"
echo "     any defect list, because it sits at 15 to 48 per cent and its precision"
echo "     is not established."
echo "  7. Still invisible here: wrong shape at plausible size."
