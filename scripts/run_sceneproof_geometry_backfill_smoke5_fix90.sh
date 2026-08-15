#!/usr/bin/env bash
# SceneProof Fix90 — measure the floor instead of assuming or deriving it.
#
# Where this stands.
#   * One object per Smoke5 scene has a collapsed XY footprint and it is always
#     floor_0.  Root cause is a single line,
#     modules/S4_blender_layout_and_corr.py:7274, which excludes the ground from
#     bbox and length serialization.  No artefact records the floor's extent.
#   * Consequences: every child resting on the floor has an unmeasurable
#     containment error and footprint overlap, the boundary family is measured
#     against a point and reports a constant zero, and the contact gap of every
#     floor child is a constant 0.020000 m.
#   * Deriving the floor plane from the walls was tried and is WITHDRAWN.  It put
#     the floor's top face 2.87 m to 3.15 m below the floor origin against a
#     predicted +0.02 m, because the walls are 10 m construction panels rather
#     than room-sized surfaces, and it produced boundary scores from 0.298 to
#     1.000 across scenes.  An arbitrary reference is not a measurement.
#   * The 10x10x0.04 construction constant is also refused: a fixed 10 m slab
#     makes containment and boundary vacuous, which inflates the metric.
#
# What this does instead: read the slab back out of the constructed Blender
# scene.  That is a measurement of the pipeline's own geometry, it needs no
# constant and no derivation, and it keeps every family denominator intact.
#
# Three scorings per scene, identical except for how the missing floor is handled:
#   asis      what the frozen numbers report today
#   abstain   omit unmeasurable terms and count them.  Honest, but it CHANGES THE
#             ESTIMAND: the omitted terms are the floor children, which are
#             systematically the worst scoring ones, so the remaining mean is not
#             comparable to asis and must never be quoted as an improvement.
#   sidecar   backfill the measured slab geometry.  Denominators are unchanged, so
#             these scores ARE comparable to asis.
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"

SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
SOURCE="${SCENEPROOF_GEOMETRY_VERSION:-v4_deepsearch}"
DUMP_TIMEOUT="${SCENEPROOF_DUMP_TIMEOUT:-1800}"
EVAL_TIMEOUT="${SCENEPROOF_EVAL_TIMEOUT:-1800}"

AUDIT_ROOT="$root/sceneba_audit/v5_sceneproof_geometry_backfill_fix90"
mkdir -p "$AUDIT_ROOT"

echo "=== SCENEPROOF FIX90: MEASURED FLOOR GEOMETRY ==="
echo "Scenes:$SCENES"
echo "Baseline:  $BASELINE"
echo "Geometry:  $SOURCE"
echo "Start:     $(date)"

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
    if ! test -s "$placement"; then
        echo "SKIP $scene: missing baseline placement"
        continue
    fi
    manifest="$audit/scene_manifest.txt"
    printf '%s\n' "$scene" > "$manifest"

    echo "  [1/5] provenance audit ..."
    if ! "$python" sceneproof_footprint_provenance_audit_fix88.py \
        --scene "$scene" --saved-results "$root" \
        --geometry-version "$SOURCE" --placement "$placement" \
        --out-report "$audit/provenance.json" > "$audit/provenance.log" 2>&1; then
        echo "  FAIL provenance audit, see $audit/provenance.log"
        continue
    fi
    sed -n '/^OBJECTS=/,$p' "$audit/provenance.log" | sed 's/^/       /'

    echo "  [2/5] measure structural geometry from the constructed scene ..."
    sidecar="$audit/structural_geometry.json"
    if ! test -s "$sidecar"; then
        source_json=$(find "$root/${scene}_${SOURCE}_result/S3_pose_inference" -maxdepth 1 -name '*_placement_info.json' -print -quit)
        if test -z "$source_json"; then
            echo "  SKIP $scene: no S3 placement for $SOURCE"
            continue
        fi
        timeout "$DUMP_TIMEOUT" env \
            CUDA_VISIBLE_DEVICES=0 \
            IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$placement" \
            IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER=1 \
            IMAGINARIUM_SCENEPROOF_STRUCTURAL_GEOMETRY_DUMP="$sidecar" \
            LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
            "$blender" --background --python modules/S4_blender_layout_and_corr.py -- \
              --obj_placement_info_json_path "$source_json" \
              --output_folder "/tmp/fix90_dump_${scene}_$$" \
              > "$audit/dump_blender.log" 2>&1 < /dev/null \
            || echo "       WARN: dump returned non-zero, see $audit/dump_blender.log"
    else
        echo "       cached sidecar, reusing"
    fi
    if ! test -s "$sidecar"; then
        echo "  FAIL $scene: no structural geometry dump produced"
        echo "       tail of $audit/dump_blender.log:"
        tail -n 15 "$audit/dump_blender.log" 2>/dev/null | sed 's/^/         /'
        continue
    fi
    "$python" - "$sidecar"<<'PY'
import json, sys
document = json.load(open(sys.argv[1]))
info = document.get("obj_info", {})
floors = {k: v for k, v in info.items() if k.startswith(("floor_", "ground_"))}
print(f"       measured {len(info)} objects; floor entries: {len(floors)}")
for name, entry in sorted(floors.items()):
    length = entry.get("length")
    bbox = entry.get("bbox") or []
    zs = [point[2] for point in bbox] if bbox else []
    print(
        "       {}: length={} z_min={} z_max={}".format(
            name,
            [round(v, 6) for v in length] if length else None,
            round(min(zs), 6) if zs else None,
            round(max(zs), 6) if zs else None,
        )
    )
PY

    ok=1
    echo "  [3/5] score as-is ..."
    score_variant asis "$audit" "$manifest" || ok=0
    echo "  [4/5] score with abstention ..."
    score_variant abstain "$audit" "$manifest" \
        --abstain-on-unmeasurable-footprints || ok=0
    echo "  [5/5] score with measured geometry backfill ..."
    score_variant sidecar "$audit" "$manifest" \
        --structural-geometry-sidecar "$sidecar" || ok=0
    if test "$ok" -ne 1; then
        echo "  FAIL one or more scorings, see $audit/*_eval.log"
        continue
    fi

    "$python" - "$audit" "$scene" "$BASELINE" <<'PY'
import json, sys
from pathlib import Path

audit, scene, version = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
variants = ("asis", "abstain", "sidecar")
FAMILIES = ("collision", "support", "plane", "boundary", "semantic")

blocks = {}
for name in variants:
    document = json.loads((audit / f"{name}_physical.json").read_text())
    blocks[name] = document["versions"][version]["scenes"][scene]

sidecar_block = blocks["sidecar"]
print(
    "       backfilled: {} {}".format(
        sidecar_block["structural_geometry_backfilled_count"],
        sidecar_block["structural_geometry_backfilled_object_ids"],
    )
)
print(
    "       degenerate footprints: asis={} sidecar={}".format(
        blocks["asis"]["degenerate_footprint_count"],
        sidecar_block["degenerate_footprint_count"],
    )
)

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
    counts = {k: v for k, v in block.get("abstained_counts", {}).items() if v}
    comparable = block.get("scores_comparable_to_non_abstained_runs")
    note = "comparable" if comparable else "ESTIMAND CHANGED, NOT COMPARABLE"
    detail = (", ".join(f"{k}={v}" for k, v in sorted(counts.items()))) or "none"
    print(f"       {name}: abstentions={detail}  -> {note}")

print(
    "       headline_macro: "
    + "  ".join(
        "{}={}{}".format(
            name,
            None
            if blocks[name].get("headline_macro_realizability") is None
            else round(blocks[name]["headline_macro_realizability"], 4),
            ""
            if blocks[name].get("scores_comparable_to_non_abstained_runs")
            else " (incomparable)",
        )
        for name in variants
    )
)
PY
    rm -f "$manifest"
done

echo""
echo "========================================"
echo "FIX90 COMPLETE  $(date)"
echo "AUDIT_ROOT=$AUDIT_ROOT"
echo ""
echo "Reading guide:"
echo "  Compare only asis against sidecar.  Both score every declared relation,"
echo "  so their denominators match and the difference is exactly the part of the"
echo "  reported deficit that was unmeasurability rather than a layout defect."
echo "  The abstain column is a diagnostic: it drops the floor children, which"
echo "  are systematically the worst terms, so its higher scores are a selection"
echo "  effect and are marked incomparable on purpose."
echo "  In the sidecar column, a floor child contact gap near zero confirms that"
echo "  the constant 0.020000 m gap was the slab's half thickness all along."
