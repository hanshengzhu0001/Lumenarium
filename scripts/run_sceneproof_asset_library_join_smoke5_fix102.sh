#!/usr/bin/env bash
# SceneProof Fix102 — asset library join, with the two defects Fix101 exposed fixed.
#
# Fix101 measured: join 100% on all five scenes, and zero abstentions across 364
# objects, so scene categories and the library's class_en are one aligned vocabulary of
# 498 classes.  Two of its own numbers were wrong.
#
# 1. A flat 2% identity tolerance is not defensible.  bbx is recorded to three decimal
#    places, so an authored 0.004 stands for [0.0035, 0.0045], plus or minus 12.5%.
#    mouse_1 printed identical computed and authored boxes and was still failed at
#    1.127, entirely inside the rounding noise.  The comparison is now against the
#    rounding interval.  Genuine failures survive: the casino table at 1.336, the
#    curtains, the sculpture, and desk_0 at 3% where rounding is only 0.06%.
#
# 2. A contradiction rate of 30to 70% conflated two different things.  pen_holder
#    against Desktop_pen_holder is one object at a finer granularity; pen against
#    Chandelier is a substitution.  Only labels sharing no token are now counted as a
#    defect, and the three relation counts are printed so the split is visible.
#    Two known imperfections are pinned in tests rather than patched with a hand list:
#    stack_of_chips against Stack_of_poker_cards shares 'stack' and is called one
#    family; wardrobe against Storage_locker shares nothing and is called a
#    substitution.
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
ASSET_CSV="${SCENEPROOF_ASSET_CSV:-asset_data/imaginarium_asset_info.csv}"

AUDIT_ROOT="$root/sceneba_audit/v5_sceneproof_asset_library_join_fix102"
mkdir -p "$AUDIT_ROOT"

echo "=== SCENEPROOF FIX102: ASSET LIBRARY JOIN ==="
echo "Scenes:$SCENES"
echo "Library:  $ASSET_CSV"
echo "Start:    $(date)"

test -s "$ASSET_CSV" || { echo "FATAL library not found at $ASSET_CSV"; exit 1; }

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

    if ! "$python" sceneproof_asset_library_join_fix101.py \
        --scene "$scene" \
        --placement "$placement" \
        --asset-info-csv "$ASSET_CSV" \
        --top-k "${SCENEPROOF_JOIN_TOP_K:-6}" \
        --out-report "$audit/asset_library_join.json" \
        > "$audit/join.log" 2>&1; then
        echo "  FAIL join, see $audit/join.log"
        continue
    fi
    sed -n '/^JOIN/,$p' "$audit/join.log" | sed 's/^/     /'
done

echo ""
echo "========================================"
echo "FIX102 COMPLETE  $(date)"
echo "AUDIT_ROOT=$AUDIT_ROOT"
echo ""
echo "Reading guide:"
echo "  1. IDENTITY is the number that decides whether length/scale = authored size"
echo "     can be stated in the paper unqualified.  Fix101 read 71-97% with a wrong"
echo "     tolerance; this run is the honest figure."
echo "  2. Read the listed identity failures for clustering.  If they share one"
echo "     scaling_strategy or one asset source the cause is systematic."
echo "  3. SUBSTITUTIONS is the actionable list and the headline defect rate.  It is"
echo "     the subset of contradictions where the two curated labels share no token."
echo "  4. contradiction_label_relations shows the split.  A large shares_a_token or"
echo "     shares_the_head_noun count means Fix101's 30-70% was mostly granularity."
echo "  5. label_relation is a heuristic over curated labels, not an exact test.  The"
echo "     two known imperfections are named in the module docstring."
echo "  6. Size against the authored asset is narrow by design: it fires only where"
echo "     the scale ran away, such as the bedroom picture frames at 5.01."
