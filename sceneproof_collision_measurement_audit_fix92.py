#!/usr/bin/env python3
"""SceneProof Fix92: audit the collision measurement before explaining its score.

Why this comes before any attempt to improve the score
------------------------------------------------------
The support family taught an expensive lesson: three quarters of its deficit was
a measurement artefact of one missing geometry record, and every hypothesis about
"the layout is bad" was wrong until the measurement itself was audited.  The
collision family is now the largest remaining deficit on Smoke5,ranging from
0.2605 on ``casino_01`` to 0.6571 on ``bedroom_01``, so it gets the same
treatment: audit the instrument first.

What the evaluator actually measures
------------------------------------
Each object is represented by its oriented bounding box treated as a *solid
prism*: the convex hull of its footprint extruded over ``[z_min, z_max]``.  Two
facts follow by construction, not by observation.

* The prism contains the object's true mesh.  Therefore a pair whose prisms do
  not intersect cannot have intersecting meshes, so the reported pairs are a
  **complete superset** of the truly interpenetrating pairs, and the measured
  intersection volume is an **upper bound** on the true one.  Consequently every
  reported ``collision`` score is a **lower bound** on what a true-mesh
  measurement would report.  Finer geometry can only raise it.
* The cavity beneath a table top, inside a bookshelf, or between the legs of a
  chair is *solid* in this representation.  A chair correctly tucked under a
  table therefore registers a large intersection.  With
  ``collision_fraction_tolerance = 0.05`` a mere 5% of the smaller object's
  bounding volume drives its term to zero, so a single correct tuck is enough.

What this tool does and does not claim
--------------------------------------
It classifies every reported pair using only quantities measured from the same
prisms, and reports what the family score would be if each class were exempt.
That is an **attribution**, not a new metric: the classification is a geometric
heuristic over bounding boxes, so it can propose which pairs deserve true-mesh
scrutiny but it cannot certify that any pair is collision-free.  Only a true-mesh
intersection can do that, and the lower-bound property above guarantees that
running it on the reported pairs alone is sufficient.

The classes are mutually exclusive and exhaustive:

``contact_band_only``
    The prisms overlap vertically by no more than ``--contact-band-m``.  This is
    a surface resting on another surface with no declared support relation, so it
    is a gap in the support graph, not interpenetration.
``enclosure_shaped``
    The smaller object's vertical span lies inside the larger object's span, and
    the footprint intersection covers at least ``--cavity-area-fraction`` of the
    smaller footprint.  This is the signature of an object occupying a cavity:
    tucked under, shelved inside, or slotted into the larger object.
``partial_vertical_enclosure``
    The two objects stand on the same surface and the smaller one rises above the
    larger, while its footprint still lies substantially inside the larger one.
    This is the same cavity artefact seen from the common case the first Smoke5
    run exposed: a chair tucked under a table with its back rising above the table
    top fails full vertical containment, and the first version of this classifier
    wrongly filed all fifty of those casino pairs as real interpenetration.
``lateral_interpenetration``
    Everything else: the prisms cut into each other from the side or straddle each
    other vertically without either enclosure pattern.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

CLASSES = (
    "contact_band_only",
    "enclosure_shaped",
    "partial_vertical_enclosure",
    "lateral_interpenetration",
)
ENCLOSURE_CLASSES = ("enclosure_shaped", "partial_vertical_enclosure")


def optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def linear_score(error: float | None, tolerance: float) -> float:
    if error is None:
        return 0.0
    return float(max(0.0, 1.0 - error / tolerance))


def classify(
    row: dict[str, Any],
    *,
    contact_band_m: float,
    cavity_area_fraction: float,
    span_tolerance_m: float,
) -> str:
    z_overlap = optional_float(row.get("z_overlap_m")) or 0.0
    if z_overlap <= contact_band_m:
        return "contact_band_only"
    smaller_min = optional_float(row.get("smaller_z_min"))
    smaller_max = optional_float(row.get("smaller_z_max"))
    larger_min = optional_float(row.get("larger_z_min"))
    larger_max = optional_float(row.get("larger_z_max"))
    area_ratio = optional_float(row.get("intersection_over_smaller_footprint"))
    if None in (smaller_min, smaller_max, larger_min, larger_max):
        # An unmeasurable span must never be filed as an artefact.
        return "lateral_interpenetration"
    if area_ratio is None or area_ratio < cavity_area_fraction:
        # The footprints only graze each other, so neither enclosure pattern
        # applies whatever the vertical spans do.
        return "lateral_interpenetration"
    if (
        smaller_min >= larger_min - span_tolerance_m
        and smaller_max <= larger_max + span_tolerance_m
    ):
        return "enclosure_shaped"
    if (
        abs(smaller_min - larger_min) <= span_tolerance_m
        and smaller_max > larger_max + span_tolerance_m
    ):
        # Both stand on the same surface and the smaller one rises above the
        # larger: a chair tucked under a table with its back showing.
        return "partial_vertical_enclosure"
    return "lateral_interpenetration"


def rescore(
    rows: list[dict[str, Any]],
    object_ids: list[str],
    exempt: set[str],
    *,
    fraction_tolerance: float,
) -> float | None:
    """Recompute the family score with the given classes treated as non-collisions.

    The denominator is every non-structural object, exactly as in the evaluator,
    so the result is comparable to the as-is score.  Only the numerator changes.
    """
    if not object_ids:
        return None
    worst = {object_id: 0.0 for object_id in object_ids}
    for row in rows:
        if row["_class"] in exempt:
            continue
        fraction = optional_float(row.get("overlap_fraction")) or 0.0
        for key in ("first_id", "second_id"):
            object_id = row.get(key)
            if object_id in worst:
                worst[object_id] = max(worst[object_id], fraction)
    return sum(
        linear_score(value, fraction_tolerance) for value in worst.values()
    ) / len(worst)


def audit_scene(
    rows: list[dict[str, Any]],
    object_ids: list[str],
    *,
    fraction_tolerance: float,
    contact_band_m: float,
    cavity_area_fraction: float,
    span_tolerance_m: float,
) -> dict[str, Any]:
    for row in rows:
        row["_class"] = classify(
            row,
            contact_band_m=contact_band_m,
            cavity_area_fraction=cavity_area_fraction,
            span_tolerance_m=span_tolerance_m,
        )
    as_is = rescore(rows, object_ids, set(), fraction_tolerance=fraction_tolerance)
    classes: dict[str, Any] = {}
    for name in CLASSES:
        member_rows = [row for row in rows if row["_class"] == name]
        touched = sorted(
            {
                row[key]
                for row in member_rows
                for key in ("first_id", "second_id")
                if row.get(key) in set(object_ids)
            }
        )
        # Objects whose every reported pair belongs to this class: these are the
        # objects whose zero score rests entirely on this one explanation.
        sole = sorted(
            object_id
            for object_id in touched
            if all(
                row["_class"] == name
                for row in rows
                if object_id in (row.get("first_id"), row.get("second_id"))
            )
        )
        classes[name] = {
            "pair_count": len(member_rows),
            "objects_touched": len(touched),
            "objects_explained_solely_by_this_class": len(sole),
            "object_ids_explained_solely_by_this_class": sole,
            "score_if_exempt": rescore(
                rows, object_ids, {name}, fraction_tolerance=fraction_tolerance
            ),
            "max_overlap_fraction": max(
                (optional_float(row.get("overlap_fraction")) or 0.0)
                for row in member_rows
            )
            if member_rows
            else None,
            "total_intersection_volume_m3": sum(
                (optional_float(row.get("intersection_volume_m3")) or 0.0)
                for row in member_rows
            ),
        }
    worst_real = sorted(
        (row for row in rows if row["_class"] == "lateral_interpenetration"),
        key=lambda row: -(optional_float(row.get("overlap_fraction")) or 0.0),
    )
    return {
        "object_count": len(object_ids),
        "reported_pair_count": len(rows),
        "collision_score_as_is": as_is,
        "score_if_only_lateral_interpenetration_counts": rescore(
            rows,
            object_ids,
            {"contact_band_only", *ENCLOSURE_CLASSES},
            fraction_tolerance=fraction_tolerance,
        ),
        "score_if_both_enclosure_classes_exempt": rescore(
            rows,
            object_ids,
            set(ENCLOSURE_CLASSES),
            fraction_tolerance=fraction_tolerance,
        ),
        "classes": classes,
        "worst_lateral_pairs": [
            {
                "first_id": row.get("first_id"),
                "second_id": row.get("second_id"),
                "overlap_fraction": optional_float(row.get("overlap_fraction")),
                "intersection_volume_m3": optional_float(
                    row.get("intersection_volume_m3")
                ),
                "penetration_depth_m": optional_float(row.get("penetration_depth_m")),
                "z_overlap_m": optional_float(row.get("z_overlap_m")),
                "smaller_footprint_inside_larger": row.get(
                    "smaller_footprint_inside_larger"
                ),
            }
            for row in worst_real[:10]
        ],
        "measurement_model": {
            "geometry": "oriented_bounding_box_solid_prism",
            "prism_contains_true_mesh": True,
            "reported_pairs_are_complete_superset_of_true_pairs": True,
            "as_is_score_is_a_lower_bound": True,
            "exempt_scores_are_attribution_not_measurement": True,
            "certification_requires_true_mesh_intersection": True,
        },
    }


def load_pairs(path: Path, scene: str, version: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("scene") == scene and row.get("version") == version
        ]


def load_object_ids(path: Path, scene: str, version: str) -> list[str]:
    """Read the collision denominator from the evaluator's own object CSV."""
    ids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("scene") != scene or row.get("version") != version:
                continue
            if optional_float(row.get("collision_term")) is None:
                continue
            object_id = row.get("object_id")
            if object_id:
                ids.append(object_id)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--collision-pairs-csv", type=Path, required=True)
    parser.add_argument("--objects-csv", type=Path, required=True)
    parser.add_argument("--collision-fraction-tolerance", type=float, default=0.05)
    parser.add_argument("--contact-band-m", type=float, default=0.02)
    parser.add_argument("--cavity-area-fraction", type=float, default=0.25)
    parser.add_argument("--span-tolerance-m", type=float, default=0.01)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()

    object_ids = load_object_ids(args.objects_csv, args.scene, args.version)
    if not object_ids:
        raise SystemExit(
            f"no collision terms for scene={args.scene} version={args.version} in "
            f"{args.objects_csv}"
        )
    rows = load_pairs(args.collision_pairs_csv, args.scene, args.version)
    result = audit_scene(
        rows,
        object_ids,
        fraction_tolerance=args.collision_fraction_tolerance,
        contact_band_m=args.contact_band_m,
        cavity_area_fraction=args.cavity_area_fraction,
        span_tolerance_m=args.span_tolerance_m,
    )
    report = {
        "schema_version": "sceneproof_collision_measurement_audit_v1",
        "scene": args.scene,
        "version": args.version,
        "thresholds": {
            "collision_fraction_tolerance": args.collision_fraction_tolerance,
            "contact_band_m": args.contact_band_m,
            "cavity_area_fraction": args.cavity_area_fraction,
            "span_tolerance_m": args.span_tolerance_m,
        },
        **result,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out_report.resolve()}")

    def show(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    print(
        "COLLISION objects={} pairs={} as_is={}".format(
            result["object_count"],
            result["reported_pair_count"],
            show(result["collision_score_as_is"]),
        )
    )
    for name in CLASSES:
        block = result["classes"][name]
        print(
            "  {:<26} pairs={:<5} objects={:<4} sole_cause={:<4} "
            "score_if_exempt={}".format(
                name,
                block["pair_count"],
                block["objects_touched"],
                block["objects_explained_solely_by_this_class"],
                show(block["score_if_exempt"]),
            )
        )
    print(
        "  score if both enclosure classes exempt = {}".format(
            show(result["score_if_both_enclosure_classes_exempt"])
        )
    )
    print(
        "  score if only lateral_interpenetration counts = {}".format(
            show(result["score_if_only_lateral_interpenetration_counts"])
        )
    )
    print(
        "  (as-is is a lower bound; exempt scores are attribution, "
        "not measurement)"
    )
    for pair in result["worst_lateral_pairs"][:5]:
        print(
            "    worst lateral: {} vs {} fraction={:.4f} volume={:.6f}m3 "
            "depth={:.4f}m z_overlap={:.4f}m footprint_inside={}".format(
                pair["first_id"],
                pair["second_id"],
                pair["overlap_fraction"] or 0.0,
                pair["intersection_volume_m3"] or 0.0,
                pair["penetration_depth_m"] or 0.0,
                pair["z_overlap_m"] or 0.0,
                pair["smaller_footprint_inside_larger"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
