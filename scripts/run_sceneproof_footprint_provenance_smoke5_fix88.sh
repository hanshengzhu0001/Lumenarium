#!/usr/bin/env bash
# SceneProof Fix88 — is the residual support deficit a scene defect or a
# measurement artefact?
#
# The Fix87 screen reported ``parent=floor_0 parent_vertices=1`` on every Smoke5
# scene: the floor's XY footprint has collapsed to a single point.  From the
# evaluator's own code that single defect produces four effects, none of which is
# a property of the layout:
#
#   * every child resting on the floor gets an infinite containment error, so its
#     containment summand is zero;
#   * the same children get zero footprint overlap, so that summand is zero too,
#     capping their support term at one third;
#   * the boundary family measures every object against the floor polygon, so it
#     scores a constant zero, which matches the exactly 0.0 boundary delta seen
#     in all five scenes of the Fix85 run;
#   * the contact gap is inflated by the slab's half thickness, because a
#     collapsed slab reports its centre as its top surface.  Every
#     floor-supported object reported a gap of exactly 0.020000 m and the ground
#     slab is built 0.04 m thick.
#
# This script measures the size of that artefact instead of arguing about it.
# Per scene it scores the frozen baseline twice, identically except that the
# second run derives a collapsed object's footprint from the geometry carried by
# the layout itself.  The difference is the part of the reported deficit that was
# never a scene defect.
#
# Nothing is promoted and no pose is modified.  Pure Python, seconds per scene.
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
SOURCE="${SCENEPROOF_GEOMETRY_VERSION:-v4_deepsearch}"
EVAL_TIMEOUT="${SCENEPROOF_EVAL_TIMEOUT:-1800}"

AUDIT_ROOT="$root/sceneba_audit/v5_sceneproof_footprint_provenance_fix88"
mkdir -p "$AUDIT_ROOT"

echo "=== SCENEPROOF FIX88: FOOTPRINT PROVENANCE AND ARTEFACT SIZE ==="
echo "Scenes:$SCENES"
echo "Baseline:  $BASELINE"
echo "Geometry:  $SOURCE"
echo "Start:     $(date)"

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

    echo "  [1/3] footprint provenance audit ..."
    if ! "$python" sceneproof_footprint_provenance_audit_fix88.py \
        --scene "$scene" \
        --saved-results "$root" \
        --geometry-version "$SOURCE" \
        --placement "$placement" \
        --out-report "$audit/provenance.json" > "$audit/provenance.log" 2>&1; then
        echo "  FAIL provenance audit, see $audit/provenance.log"
        continue
    fi
    sed -n '/^OBJECTS=/,$p' "$audit/provenance.log" | sed 's/^/       /'

    echo "  [2/3] score baseline as-is ..."
    if ! timeout "$EVAL_TIMEOUT" "$python" eval_physical_realizability.py \
        --saved-results "$root" --scenes "$manifest" \
        --versions "$BASELINE" --geometry-version "$SOURCE" \
        --baseline-version "$BASELINE" \
        --metrics-out "$audit/asis_physical.json" \
        --scene-csv "$audit/asis_scenes.csv" \
        --object-csv "$audit/asis_objects.csv" \
        --report-out "$audit/asis_physical.txt" > "$audit/asis_eval.log" 2>&1; then
        echo "  FAIL as-is evaluation, see $audit/asis_eval.log"
        continue
    fi

    echo "  [3/3] score baseline with footprint repair ..."
    if ! timeout "$EVAL_TIMEOUT" "$python" eval_physical_realizability.py \
        --saved-results "$root" --scenes "$manifest" \
        --versions "$BASELINE" --geometry-version "$SOURCE" \
        --baseline-version "$BASELINE" \
        --repair-degenerate-footprints \
        --metrics-out "$audit/repaired_physical.json" \
        --scene-csv "$audit/repaired_scenes.csv" \
        --object-csv "$audit/repaired_objects.csv" \
        --report-out "$audit/repaired_physical.txt" > "$audit/repaired_eval.log" 2>&1; then
        echo "  FAIL repaired evaluation, see $audit/repaired_eval.log"
        continue
    fi

    "$python" - "$audit/asis_physical.json" "$audit/repaired_physical.json" \
        "$scene" "$BASELINE" <<'PY'
import json, sys

asis = json.load(open(sys.argv[1]))
repaired = json.load(open(sys.argv[2]))
scene, version = sys.argv[3], sys.argv[4]


def block(document):
    return document["versions"][version]["scenes"][scene]


left, right = block(asis), block(repaired)
print("       degenerate_footprints={} -> repaired={} (remaining={})".format(
    left["degenerate_footprint_count"],
    right["repaired_footprint_count"],
    right["degenerate_footprint_count"],
))
if right["repaired_footprint_object_ids"]:
    print("       repaired: " + ", ".join(right["repaired_footprint_object_ids"][:8]))
for family in ("collision", "support", "plane", "boundary", "semantic"):
    a = left["families"].get(family, {})
    b = right["families"].get(family, {})
    if a.get("score") is None and b.get("score") is None:
        continue
    delta = (
        None
        if a.get("score") is None or b.get("score") is None
        else b["score"] - a["score"]
    )
    print(
        "       {:<10} as_is={} repaired={} delta={} n={}->{}".format(
            family,
            None if a.get("score") is None else round(a["score"], 4),
            None if b.get("score") is None else round(b["score"], 4),
            None if delta is None else format(delta, "+.4f"),
            a.get("n"),
            b.get("n"),
        )
    )
print("       headline_macro {} -> {}".format(
    None if left.get("headline_macro_realizability") is None
    else round(left["headline_macro_realizability"], 4),
    None if right.get("headline_macro_realizability") is None
    else round(right["headline_macro_realizability"], 4),
))
PY
    rm -f "$manifest"
done

echo ""
echo "========================================"
echo "FIX88 COMPLETE  $(date)"
echo "AUDIT_ROOT=$AUDIT_ROOT"
echo "Per-scene evidence: \$AUDIT_ROOT/<scene>/{provenance.json,asis_physical.json,repaired_physical.json}"
echo""
echo "Reading guide:"
echo "  A large positive support/boundary delta means that much of the reported"
echo "  deficit was a collapsed-footprint artefact, not a layout defect, and the"
echo "  next fix belongs in the geometry snapshot rather than in more physics."
