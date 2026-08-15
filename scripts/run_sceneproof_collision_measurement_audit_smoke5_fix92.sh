#!/usr/bin/env bash
# SceneProof Fix92 — audit the collision instrument before explaining its score.
#
# Where we are.  The support family's deficit is now accounted for: three quarters
# of it was a measurement artefact of one unserialized floor slab, and the honest
# score is 0.59 to 0.67 indoors.  The boundary family has no honest value at all.
# That leaves collision as the largest remaining deficit on Smoke5:
#
#     casino_010.2605   livingroom_10 0.2751   streelitter_01 0.3548
#     official_01 0.5691bedroom_01 0.6571
#
# Before explaining those numbers as bad layout, audit the instrument, exactly as
# should have been done for support.
#
# What the evaluator measures.  Each object is its oriented bounding box treated
# as a SOLID PRISM: footprint hull extruded over [z_min, z_max].  Two consequences
# hold by construction:
#
#   1. The prism contains the true mesh.  So non-intersecting prisms imply
#      non-intersecting meshes: the reported pairs are a COMPLETE SUPERSET of the
#      realones, and every collision score is a LOWER BOUND.  Finer geometry can
#      only raise it.  This also means a true-mesh pass need only visit the pairs
#      reported here, never all O(N^2).
#   2. The cavity under a table top, inside a bookshelf, or between chair legs is
#      SOLID.  A chair correctly tucked under a table registers a large
#      intersection.  With collision_fraction_tolerance = 0.05, five percent of
#      the smaller object's bounding volume is enough to zero its term.
#
# The falsifiable reading.  If low collision scores were real interpenetration,
# there is no reason casino_01 should be 2.5x worse than bedroom_01 with the same
# retrieval and the same solver.  If instead they are the tucked-chair artefact,
# the deficit should concentrate in the enclosure-shaped class and in the dense
# seating scenes.  This run decides between those.
#
# It does NOT license quoting the exempt scores.  The classification is a
# heuristic over bounding boxes: it proposes which pairs deserve true-mesh
# scrutiny.  Only a true-mesh intersection can certify a pair as collision-free.
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"

SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
SOURCE="${SCENEPROOF_GEOMETRY_VERSION:-v4_deepsearch}"
EVAL_TIMEOUT="${SCENEPROOF_EVAL_TIMEOUT:-1800}"

AUDIT_ROOT="$root/sceneba_audit/v5_sceneproof_collision_measurement_fix92"
mkdir -p "$AUDIT_ROOT"

echo "=== SCENEPROOF FIX92: COLLISION MEASUREMENT AUDIT ==="
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

    echo "  [1/2] score as-is and export every reported collision pair ..."
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

    echo "  [2/2] classify the reported pairs ..."
    if ! "$python" sceneproof_collision_measurement_audit_fix92.py \
        --scene "$scene" --version "$BASELINE" \
        --collision-pairs-csv "$audit/collision_pairs.csv" \
        --objects-csv "$audit/asis_objects.csv" \
        --out-report "$audit/collision_measurement_audit.json" \
        > "$audit/audit.log" 2>&1; then
        echo "  FAIL audit, see $audit/audit.log"
        rm -f "$manifest"
        continue
    fi
    sed -n '/^COLLISION/,$p' "$audit/audit.log" | sed 's/^/     /'
    rm -f "$manifest"
done

echo ""
echo "========================================"
echo "FIX92 COMPLETE  $(date)"
echo "AUDIT_ROOT=$AUDIT_ROOT"
echo ""
echo "Reading guide:"
echo "  as_is is the number the paper currently reports, and it is a LOWER BOUND:"
echo "  the bounding-box prism contains the true mesh, so finer geometry can only"
echo "  raise it."
echo "  Read the class table as attribution, not as a score.  If most pairs and"
echo "  most sole-cause objects sit in enclosure_shaped, the deficit is the"
echo "  tucked-chair artefact and the next step is a true-mesh pass over the"
echo "  reported pairs only.  If they sit in lateral_interpenetration, the"
echo "  deficit is real and the next step is the solver."
echo "  contact_band_only means a missing support edge, not a collision, and is"
echo "  fixed upstream in S1, not by any physics."
