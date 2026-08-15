#!/usr/bin/env bash
# SceneProof Fix93 — the collision family has two separate defects, both found by
# the Fix92 export, and neither is "the layout is bad".
#
# DEFECT A: my own classifier was too strict, so it filed the biggest artefact as
# real.  On casino_01 the five worst "lateral_interpenetration" pairs were
#
#   gaming_table_2 vs casino_chair_3  / _14 / _15 / _23 / _24
#   fraction 0.9102volume 0.396212 m3   z_overlap 0.9167 m   -- IDENTICAL
#
# Identical to six decimals across five different chairs is not five independent
# faults, it is one seating geometry repeated.  Reconstructing it: the chair's
# whole footprint lies inside the table's footprint hull (intersection area
# 0.4322 m2 versus chair footprint 0.4225 m2) and the two share a base while the
# chair back rises above the table top.  A gaming table is oval, so its
# rectangular hull swallows chairs that stand outside the actual table.  The first
# classifier demanded FULL vertical containment, which the chair back breaks, so
# all of them fell through to "real".  Fixed by adding
# ``partial_vertical_enclosure``: same base, smaller rises above, footprint still
# substantially inside.
#
# DEFECT B, the serious one: the normalization ranks overlaps in the wrong order.
#
#   book_1 vs book_2           volume 0.000009 m3  fraction 0.5067  -> FULLY PENALISED
#   trash_bin_1 vs paper_cup_1 volume 0.061128 m3  fraction 0.0359  -> NOT PENALISED
#
# Nine cubic millimetres is punished, sixty-one litres is free.  The cause is the
# definition: fraction = intersection / min(volume_a, volume_b), and that
# denominator spans five orders of magnitude within one scene, from a book at
# ~1.8e-5 m3 to a bin at ~1.7 m3.  So the score tracks the inverse size of the
# smaller object, not the severity of the overlap.  A solver optimising it would
# chase millimetre overlaps between props and ignore litre-scale interpenetration
# between furniture.
#
# This run reports the same scenes under three normalizations computed from the
# same exported pairs: today's fraction, exact penetration depth in metres, and
# absolute intersection volume.  Penetration depth is the exact minimum
# translation distance for these prisms, not an approximation, because a prism is
# a convex polygon crossed with an interval and the distance to the boundary of a
# product set is the smaller of the factors' distances.
#
# Nothing here changes an evaluator score.  Fixing the estimand comes before
# improving its precision: a true-mesh pass would only compute a badly normalised
# quantity more accurately.
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
SOURCE="${SCENEPROOF_GEOMETRY_VERSION:-v4_deepsearch}"
EVAL_TIMEOUT="${SCENEPROOF_EVAL_TIMEOUT:-1800}"

AUDIT_ROOT="$root/sceneba_audit/v5_sceneproof_collision_normalization_fix93"
mkdir -p "$AUDIT_ROOT"

echo "=== SCENEPROOF FIX93: COLLISION CLASSES AND NORMALIZATION ==="
echo "Scenes:   $SCENES"
echo "Baseline: $BASELINE"
echo "Geometry: $SOURCE"
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
    manifest="$audit/scene_manifest.txt"
    printf '%s\n' "$scene" > "$manifest"

    echo "  [1/3] re-export pairs, now with exact penetration depth ..."
    if ! timeout "$EVAL_TIMEOUT" "$python" eval_physical_realizability.py \
        --saved-results "$root" --scenes "$manifest" \
        --versions "$BASELINE" --geometry-version "$SOURCE" \
        --baseline-version "$BASELINE" \
        --metrics-out "$audit/asis_physical.json" \
        --scene-csv "$audit/asis_scenes.csv" \
        --object-csv "$audit/asis_objects.csv" \
        --report-out "$audit/asis_physical.txt" \
        --collision-pairs-csv "$audit/collision_pairs.csv" \
        > "$audit/eval.log" 2>&1; then
        echo "  FAIL scoring, see $audit/eval.log"
        rm -f "$manifest"
        continue
    fi

    echo "  [2/3] re-classify with the partial-enclosure class..."
    if ! "$python" sceneproof_collision_measurement_audit_fix92.py \
        --scene "$scene" --version "$BASELINE" \
        --collision-pairs-csv "$audit/collision_pairs.csv" \
        --objects-csv "$audit/asis_objects.csv" \
        --out-report "$audit/collision_measurement_audit.json" \
        > "$audit/classes.log" 2>&1; then
        echo "  FAIL classification, see $audit/classes.log"
        rm -f "$manifest"
        continue
    fi
    sed -n '/^COLLISION/,$p' "$audit/classes.log" | sed 's/^/     /'

    echo "  [3/3] compare the three normalizations ..."
    if ! "$python" sceneproof_collision_normalization_audit_fix93.py \
        --scene "$scene" --version "$BASELINE" \
        --collision-pairs-csv "$audit/collision_pairs.csv" \
        --objects-csv "$audit/asis_objects.csv" \
        --out-report "$audit/collision_normalization_audit.json" \
        > "$audit/normalization.log" 2>&1; then
        echo "  FAIL normalization audit, see $audit/normalization.log"
        rm -f "$manifest"
        continue
    fi
    sed -n '/^NORMALIZATION/,$p' "$audit/normalization.log" | sed 's/^/     /'
    rm -f "$manifest"
done

echo ""
echo "========================================"
echo "FIX93 COMPLETE  $(date)"
echo "AUDIT_ROOT=$AUDIT_ROOT"
echo ""
echo "Reading guide:"
echo "  1. Class table: how many casino pairs moved out of"
echo "     lateral_interpenetration into partial_vertical_enclosure.  If the"
echo "     gaming_table-versus-chair pairs moved, defect A is confirmed and the"
echo "     earlier 'the deficit is real' reading of casino_01 was my error."
echo "  2. Normalization table: the three scores side by side.  The number that"
echo "     matters is the rank correlation and the flipped-verdict count.  Low"
echo "     correlation means fraction and depth disagree about WHICH objects are"
echo "     bad, which is what an ill-posed normalization means operationally."
echo "  3. The inversion line: smallest fully penalised overlap versus largest"
echo "     unpenalised one.  If the ratio is far above 1, the current metric is"
echo "     ordering overlaps by object size rather than by severity."
echo ""
echo "  No evaluator score changed in this run.  Decide the estimand first; a"
echo "  true-mesh pass would otherwise just compute a badly normalised quantity"
echo "  more accurately."
