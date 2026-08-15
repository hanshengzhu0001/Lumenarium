#!/usr/bin/env bash
# SceneProof Fix103 — the last audit-only round.  Two questions, both of which have to
# be answered before any pipeline line is touched.
#
# 1. Is the authored size usable as a prior?  Fix102 read 78.8 to 100% agreement between
#    length/scale and bbx.  Pooling those failures into one rate hides that they have
#    two different causes, so each is now classified by the shape of its disagreement:
#    one scalar on all three axes means a transform the scale field does not record,
#    a single wrong axis means one authored number.  Only the first would corrupt a
#    prior derived from bbx.
#
# 2. Is line 6414 worth changing?  When the pixel-bbox estimator reports an anisotropy
#    above five it returns [1,1,1].  That reads as a neutral default and is in fact the
#    assertion'this object is exactly as large as its asset'.  The library is what turns
#    that assertion into metres.  The audit prints the upper bound of objects bearing the
#    signature and, within it, the subset where the unscaled asset exceeds its own depth
#    evidence by more than three times.  The second number decides.
#
# Also folded in: the asset identifier as a second retrieval witness, because Fix102
# convicted wardrobe_0 for retrieving a_SM_Wardrobe_01.  This only corrects a reported
# rate and changes no pipeline behaviour.
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
ASSET_CSV="${SCENEPROOF_ASSET_CSV:-asset_data/imaginarium_asset_info.csv}"

AUDIT_ROOT="$root/sceneba_audit/v5_sceneproof_asset_library_join_fix103"
mkdir -p "$AUDIT_ROOT"

echo "=== SCENEPROOF FIX103: IDENTITY SHAPES + FALLBACK HITS (AUDIT ONLY) ==="
echo "Scenes:   $SCENES"
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
        --top-k "${SCENEPROOF_JOIN_TOP_K:-12}" \
        --fallback-evidence-factor "${SCENEPROOF_FALLBACK_FACTOR:-3.0}" \
        --out-report "$audit/asset_library_join.json" \
        > "$audit/join.log" 2>&1; then
        echo "  FAIL join, see $audit/join.log"
        continue
    fi
    sed -n '/^JOIN/,$p' "$audit/join.log" | sed 's/^/     /'
done

echo ""
echo "========================================"
echo "FIX103 COMPLETE$(date)"
echo "AUDIT_ROOT=$AUDIT_ROOT"
echo ""
echo "Reading guide:"
echo "  1. FALLBACK is the decision.  The candidate count is an UPPER BOUND: a unit"
echo "     scale is also what a correct estimate of one produces.  The 'far larger than"
echo "     the depth evidence' subset is the actionable number.  Under about 5% of a"
echo "     scene and line 6414 is not worth touching; the reported 41% of livingroom at"
echo "     scale=[1,1,1] predicts it is far above that."
echo "  2. shape of the disagreement decides whether bbx can serve as a size prior."
echo "     uniform_scalar_offset is the only shape that would corrupt one, because it"
echo "     means a scale exists that the scale field does not record.  Fix102's listing"
echo "     predicts four such cases, all in official_01."
echo "  3. by scaling_strategy prints failed/checked, so a strategy with many objects"
echo "     and few failures cannot be mistaken for the cause."
echo "  4. EXCUSED is the count Fix102 wrongly convicted.  Expect wardrobe_0, the three"
echo "     signs, the fruit-to-tomato family, the shelf-bucketed bookshelves and the"
echo "     paper-to-Map family; SUBSTITUTIONS should fall to roughly 15/30/33/34/32%."
echo "  5. 'resting on an opaque asset name' is the part of that rate with no evidence"
echo "     either way, kept as a defect and counted apart rather than quietly dropped."
