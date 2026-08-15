#!/usr/bin/env bash
# SceneProof Fix91 — how much of the backfill gain is real?
#
# Fix90 measured the ground slab out of the constructed scene and confirmed the
# hypothesis exactly: length [10.0, 10.0, 0.04], z from -0.02 to +0.02, and every
# floor child starting at +0.02.  The constant 0.020000 m contact gap was the
# slab's half thickness and nothing else, so those objects were resting perfectly
# all along.
#
# But the same measurement exposed a second problem.  The slab is exactly 10 m by
# 10 m regardless of the room, so it is a construction placeholder, not the floor
# of a 4 to 6 m room.  Backfilling it therefore does two different things at once:
#
#   contact gap   becomes CORRECT.  The slab's top face is the surface the
#                 pipeline actually placed objects on.Recovering this summand is
#                 a genuine correction, worth up to 1/3 of a support term.
#   containment   become TRIVIALLY SATISFIED.  Every object in the room lies well
#   and overlap   inside a 10 m slab, so these summands stop discriminating.
#                 Recovering them inflates the score without measuring the layout.
#
# Three scorings plus an exact decomposition:
#   asis         what the frozen numbers report today
#   sidecar      measured slab backfilled, all three summands active.  Comparable
#                to asis by denominator, but partly vacuous.  DO NOT QUOTE ALONE.
#   restricted   measured slab backfilled, but the lateral extent of structural
#                placeholders is declared unmeasurable.  The contact gap is still
#                measured against the real top face.Denominator is preserved,
#                yet the estimand changes, so this is flagged incomparable.
#
# The decomposition splits the asis-to-sidecar support delta into the legitimate
# contact-gap part and the vacuous lateral part, in closed form, per object.
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
SOURCE="${SCENEPROOF_GEOMETRY_VERSION:-v4_deepsearch}"
EVAL_TIMEOUT="${SCENEPROOF_EVAL_TIMEOUT:-1800}"
SIDECAR_ROOT="${SCENEPROOF_SIDECAR_ROOT:-$root/sceneba_audit/v5_sceneproof_geometry_backfill_fix90}"

AUDIT_ROOT="$root/sceneba_audit/v5_sceneproof_gain_decomposition_fix91"
mkdir -p "$AUDIT_ROOT"

echo "=== SCENEPROOF FIX91: LEGITIMATE VS VACUOUS GAIN ==="
echo "Scenes:$SCENES"
echo "Baseline:     $BASELINE"
echo "Geometry:     $SOURCE"
echo "Sidecar root: $SIDECAR_ROOT"
echo "Start:        $(date)"

score_variant() {
    variant_name="$1"
    variant_audit="$2"
    variant_manifest="$3"
    shift 3
    timeout "$EVAL_TIMEOUT" "$python" eval_physical_realizability.py \
        --saved-results "$root" --scenes "$variant_manifest" \
        --versions "$BASELINE" --geometry-version "$SOURCE" \
        --baseline-version "$BASELINE" \
        --metrics-out "$variant_audit/${variant_name}_physical.json" \
        --scene-csv "$variant_audit/${variant_name}_scenes.csv" \
        --object-csv "$variant_audit/${variant_name}_objects.csv" \
        --report-out "$variant_audit/${variant_name}_physical.txt" \
        "$@" > "$variant_audit/${variant_name}_eval.log" 2>&1
}

for scene in $SCENES; do
    echo ""
    echo "--- $scene ---"
    audit="$AUDIT_ROOT/$scene"
    mkdir -p "$audit"

    placement="$root/${scene}_${BASELINE}_result/S4_layout_refinement/${scene}_${BASELINE}_placement_info_s4.json"
    sidecar="$SIDECAR_ROOT/$scene/structural_geometry.json"
    if ! test -s "$placement"; then
        echo "SKIP $scene: missing baseline placement"
        continue
    fi
    if ! test -s "$sidecar"; then
        echo "SKIP $scene: missing measured geometry sidecar at $sidecar"
        echo "       run scripts/run_sceneproof_geometry_backfill_smoke5_fix90.sh first"
        continue
    fi
    manifest="$audit/scene_manifest.txt"
    printf '%s\n' "$scene" > "$manifest"

    ok=1
    echo "  [1/4] score as-is ..."
    score_variant asis "$audit" "$manifest" || ok=0
    echo "  [2/4] score with measured slab backfilled ..."
    score_variant sidecar "$audit" "$manifest" \
        --structural-geometry-sidecar "$sidecar" || ok=0
    echo "  [3/4] score with placeholder lateral extent declared unmeasurable ..."
    score_variant restricted "$audit" "$manifest" \
        --structural-geometry-sidecar "$sidecar" \
        --placeholder-structural-lateral-extent || ok=0
    if test "$ok" -ne 1; then
        echo "  FAIL one or more scorings, see $audit/*_eval.log"
        continue
    fi

    echo "  [4/4] decompose the asis-to-sidecar support gain ..."
    if ! "$python" sceneproof_support_gain_decomposition_fix91.py \
        --scene "$scene" --version "$BASELINE" \
        --placement "$placement" \
        --baseline-objects-csv "$audit/asis_objects.csv" \
        --backfilled-objects-csv "$audit/sidecar_objects.csv" \
        --out-report "$audit/gain_decomposition.json" \
        > "$audit/decomposition.log" 2>&1; then
        echo "  FAIL decomposition, see $audit/decomposition.log"
        continue
    fi
    sed -n '/^SUPPORT_DELTA/,$p' "$audit/decomposition.log" | sed 's/^/       /'

    "$python" - "$audit" "$scene" "$BASELINE" <<'PY'
import json, sys
from pathlib import Path

audit, scene, version = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
variants = ("asis", "sidecar", "restricted")
FAMILIES = ("collision", "support", "plane", "boundary", "semantic")

blocks = {}
for name in variants:
    document = json.loads((audit / f"{name}_physical.json").read_text())
    blocks[name] = document["versions"][version]["scenes"][scene]

print("       {:<10}".format("family") + "".join(f"{n:>18}" for n in variants))
for family in FAMILIES:
    cells = []
    for name in variants:
        entry = blocks[name]["families"].get(family, {})
        score = entry.get("score")
        cells.append(
            "{:>18}".format(
                "n/a (n=0)" if score is None else f"{score:.4f} (n={entry.get('n')})"
            )
        )
    print("       {:<10}".format(family) + "".join(cells))

for name in variants:
    block = blocks[name]
    flags = []
    if block.get("estimand_changed_by_abstention"):
        flags.append("abstention")
    if block.get("estimand_changed_by_partial_summands"):
        flags.append(
            "partial summands x{}".format(
                block.get("partial_summand_support_term_count")
            )
        )
    verdict = "comparable" if not flags else "NOT COMPARABLE (" + ", ".join(flags) + ")"
    print(f"       {name}: {verdict}")

decomposition = json.loads((audit / "gain_decomposition.json").read_text())
fraction = decomposition["legitimate_fraction"]
print(
    "       support gain asis->sidecar: total={:+.4f}  legitimate={:+.4f}  "
    "vacuous={:+.4f}  legitimate share={}".format(
        decomposition["total_support_delta"],
        decomposition["legitimate_contact_gap_delta"],
        decomposition["vacuous_lateral_delta"],
        "n/a" if fraction is None else f"{fraction:.1%}",
    )
)
print(
    "       honest support estimate = asis + legitimate = {:.4f}".format(
        blocks["asis"]["families"]["support"]["score"]
        + decomposition["legitimate_contact_gap_delta"]
    )
)
PY
    rm -f "$manifest"
done

echo ""
echo "========================================"
echo "FIX91 COMPLETE  $(date)"
echo "AUDIT_ROOT=$AUDIT_ROOT"
echo ""
echo "Reading guide:"
echo "  The number to trust is 'honest support estimate': the as-is score plus"
echo "  only the contact-gap correction.  The sidecar column's remaining gain is"
echo "  containment and overlap against a 10 m placeholder slab and measures"
echo "  nothing about the layout."
echo "  The boundary family has no honest value at all here: answering it needs"
echo "  the room's extent, which no artefact represents."
