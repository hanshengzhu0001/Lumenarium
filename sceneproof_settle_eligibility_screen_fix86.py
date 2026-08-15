#!/usr/bin/env python3
"""SceneProof eligibility screen for gravity settling (Fix86, corrected by Fix87).

What this decides
-----------------
Which objects a gravity simulation is allowed to touch, computed from artefacts
already on disk, with no simulation.

Fix86 established the semantic exclusions.  Applying gravity to an object that is
not free to fall cannot improve any score and can only destroy one:

* a child whose declared support parent is a wall or a ceiling is attached, not
  resting, and gravity slides it off the wall;
* a child declared ``inside`` a container is held by containment semantics, so
  free fall pulls it out of its container;
* a support parent is dynamic at the same time as its children, so the stack
  collapses instead of settling.

Fix87 corrects the remaining rule, which was wrong in a way the Smoke5 screen
exposed immediately: it admitted 286 targets, including objects already in
nanometre-scale contact with their parent.

The correction: gate on the actionable component, not the total
--------------------------------------------------------------
The evaluator's resting-support term is an explicit closed form,

    s =( L(gap, contact_tol) + L(containment, containment_tol)
          + min(1, overlap / overlap_tol) ) / 3,L(e, t) = max(0, 1 - e/t)

and a vertical gravity drop can only move the first summand.  The second and
third are set by the horizontal placement and by the parent's own footprint
geometry.Gating on the total``s`` therefore admits objects with nothing to
gain.  ``discarded_wooden_board_12`` in``streelitter_01`` is the witness: its
contact gap is 3.7e-9 m, so it is already resting exactly, yet it scores0.3333
because its containment error is 0.70 m and its footprint overlap is 0.

The attainable gain from settling alone is therefore exactly

    delta_max = (1 - L(gap, contact_tol)) / 3 = min(1, gap / contact_tol) / 3

which this screen evaluates in closed form.  No simulation is needed to know
whether an object is worth simulating.

A second finding this screen now reports
----------------------------------------
``outside_distance`` returns infinity only when the parent's footprint polygon
has fewer than three vertices, that is, when the parent's XY projection has
collapsed to a line or a point.  An infinite containment error is therefore not a
pose defect at all: it says the declared support parent has a degenerate
footprint.  Such an object is capped at a support term of 1/3 no matter what any
optimizer or simulator does, because two of the three summands are unreachable.

The report splits the scene's residual support deficit into the part gravity can
recover and the part that is bounded by the parent's geometry, which is the
quantity that decides whether more physics is worth running at all.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

import eval_physical_realizability as evaluator

SCHEMA = "sceneproof_settle_eligibility_v2"

EXCLUSION_ORDER = (
    "absent_from_layout",
    "structural",
    "no_support_term",
    "missing_support_parent",
    "held_by_containment",
    "is_support_parent",
    "already_in_contact",
    "attainable_gain_below_floor",
)

STRUCTURAL = re.compile(r"^(floor|ground|wall|ceiling|carpet|rug)_\d+$")


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def optional_float(raw: Any) -> float | None:
    """Parse a CSV cell, preserving infinities as infinities."""
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


def support_id(value: Any) -> str | None:
    if isinstance(value, (list, tuple)):
        value = next((item for item in value if item), None)
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def structural(name: str) -> bool:
    return bool(STRUCTURAL.match(name))


def linear_score(error: float | None, tolerance: float) -> float:
    """Mirror the evaluator's linear score, including its treatment of inf."""
    if error is None or not np.isfinite(error):
        return 0.0
    return float(max(0.0, 1.0 - error / tolerance))


def attainable_settle_gain(gap: float | None, contact_tolerance: float) -> float:
    """Support-term gain available from closing the contact gap alone."""
    if gap is None or not np.isfinite(gap):
        return 0.0
    return float(max(0.0, 1.0 - linear_score(gap, contact_tolerance)) / 3.0)


def load_baseline_terms(
    object_csv: Path,
    scene: str,
    version: str,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with Path(object_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "support_term" not in reader.fieldnames:
            raise SystemExit(
                f"{object_csv} has no 'support_term' column; re-run "
                "eval_physical_realizability.py with the term export"
            )
        for row in reader:
            if row.get("scene") != scene or row.get("version") != version:
                continue
            object_id = row.get("object_id")
            if object_id:
                rows[object_id] = row
    return rows


def footprint_diagnostics(
    source_info: dict[str, Any],
    target_info: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Per-object footprint facts, using the evaluator's own geometry builder.

    The vertex count is the decisive diagnostic: fewer than three vertices is the
    only condition under which the evaluator's containment error is infinite.
    """
    geometries = evaluator.build_geometries(source_info, target_info)
    facts: dict[str, dict[str, Any]] = {}
    for name, geometry in geometries.items():
        polygon = geometry.polygon
        facts[name] = {
            "footprint_vertex_count": int(len(polygon)),
            "footprint_area_m2": float(evaluator.polygon_area(polygon)),
            "footprint_degenerate": bool(len(polygon) < 3),
            "z_min_m": float(geometry.z_min),
            "z_max_m": float(geometry.z_max),
        }
    return facts


def screen(
    placement: dict[str, Any],
    baseline_rows: dict[str, dict[str, Any]],
    footprints: dict[str, dict[str, Any]],
    *,
    contact_tolerance: float,
    containment_tolerance: float,
    overlap_tolerance: float,
    minimum_gap_m: float,
    minimum_attainable_gain: float,
) -> dict[str, Any]:
    obj_info = placement.get("obj_info", {})

    declared_parents: set[str] = set()
    for name, info in obj_info.items():
        if not isinstance(info, dict):
            continue
        parent_id = support_id(info.get("supported"))
        if parent_id:
            declared_parents.add(parent_id)

    eligible: list[dict[str, Any]] = []
    excluded: dict[str, list[str]] = {reason: [] for reason in EXCLUSION_ORDER}
    parent_bound: list[dict[str, Any]] = []

    deficit_total = 0.0
    deficit_gap = 0.0
    deficit_containment = 0.0
    deficit_overlap = 0.0
    support_contributors = 0

    for object_id, row in sorted(baseline_rows.items()):
        support_term = optional_float(row.get("support_term"))
        gap = optional_float(row.get("support_contact_gap_m"))
        containment = optional_float(row.get("support_containment_error_m"))
        overlap = optional_float(row.get("support_footprint_overlap_ratio"))
        inside_error = optional_float(row.get("inside_containment_error_m"))

        # Residual support deficit, split by summand.  Only resting-support
        # objects have three summands, so containment-only objects are counted
        # against the total but not against the per-summand split.
        if support_term is not None:
            support_contributors += 1
            deficit_total += 1.0 - support_term
            if inside_error is None and gap is not None:
                deficit_gap += (1.0 - linear_score(gap, contact_tolerance)) / 3.0
                deficit_containment += (
                    1.0 - linear_score(containment, containment_tolerance)
                ) / 3.0
                observed_overlap = 0.0 if overlap is None else overlap
                deficit_overlap += (
                    1.0 - min(1.0, observed_overlap / overlap_tolerance)
                ) / 3.0

        info = obj_info.get(object_id)
        if not isinstance(info, dict):
            excluded["absent_from_layout"].append(object_id)
            continue
        if structural(object_id):
            excluded["structural"].append(object_id)
            continue
        if support_term is None:
            excluded["no_support_term"].append(object_id)
            continue

        parent_id = support_id(info.get("supported"))
        if not parent_id or parent_id not in obj_info:
            excluded["missing_support_parent"].append(object_id)
            continue
        if str(info.get("SpatialRel", "")).strip().lower() == "inside" or (
            inside_error is not None
        ):
            excluded["held_by_containment"].append(object_id)
            continue
        if object_id in declared_parents:
            excluded["is_support_parent"].append(object_id)
            continue

        parent_facts = footprints.get(parent_id, {})
        degenerate_parent = bool(
            parent_facts.get("footprint_degenerate")
            or (containment is not None and not np.isfinite(containment))
        )
        # A degenerate parent footprint permanently caps the support term at the
        # gap summand alone.  Record the ceiling; gravity may still recover that
        # summand, so this is a diagnostic rather than an exclusion.
        support_ceiling = 1.0
        if degenerate_parent:
            support_ceiling = 1.0 / 3.0
        elif overlap is not None and overlap < overlap_tolerance:
            support_ceiling = (
                1.0
                + linear_score(containment, containment_tolerance)
                + min(1.0, overlap / overlap_tolerance)
            ) / 3.0

        gain = attainable_settle_gain(gap, contact_tolerance)
        entry = {
            "object_id": object_id,
            "support_parent_id": parent_id,
            "support_term": support_term,
            "support_contact_gap_m": gap,
            "support_containment_error_m": containment,
            "support_footprint_overlap_ratio": overlap,
            "attainable_settle_gain": gain,
            "support_term_ceiling": support_ceiling,
            "parent_footprint_vertex_count": parent_facts.get(
                "footprint_vertex_count"
            ),
            "parent_footprint_area_m2": parent_facts.get("footprint_area_m2"),
            "parent_footprint_degenerate": degenerate_parent,
        }
        if degenerate_parent or support_ceiling < 1.0 - 1e-12:
            parent_bound.append(entry)

        if gap is None or not np.isfinite(gap) or gap <= minimum_gap_m:
            excluded["already_in_contact"].append(object_id)
            continue
        if gain < minimum_attainable_gain:
            excluded["attainable_gain_below_floor"].append(object_id)
            continue
        eligible.append(entry)

    # Largest attainable gain first.
    eligible.sort(
        key=lambda item: (-item["attainable_settle_gain"], item["object_id"])
    )
    denominator = max(support_contributors, 1)
    return {
        "eligible": eligible,
        "excluded": {
            reason: excluded[reason] for reason in EXCLUSION_ORDER if excluded[reason]
        },
        "excluded_counts": {
            reason: len(excluded[reason]) for reason in EXCLUSION_ORDER
        },
        "scored_object_count": len(baseline_rows),
        "support_contributor_count": support_contributors,
        "eligible_count": len(eligible),
        "declared_support_parent_count": len(declared_parents),
        "parent_geometry_bound": parent_bound,
        "parent_geometry_bound_count": len(parent_bound),
        "parent_footprint_degenerate_count": sum(
            1 for entry in parent_bound if entry["parent_footprint_degenerate"]
        ),
        # Scene-level attribution of the residual support deficit.  Each figure
        # is expressed in the same units as the family score, so it can be read
        # directly as "how much of the support score is recoverable".
        "support_deficit": {
            "total": deficit_total / denominator,
            "contact_gap_component": deficit_gap / denominator,
            "containment_component": deficit_containment / denominator,
            "footprint_overlap_component": deficit_overlap / denominator,
            "recoverable_by_settling": sum(
                entry["attainable_settle_gain"] for entry in eligible
            )
            / denominator,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument(
        "--baseline-objects-csv",
        type=Path,
        required=True,
        help="physical_objects.csv containing the baseline version rows",
    )
    parser.add_argument("--baseline-version", required=True)
    parser.add_argument(
        "--saved-results", type=Path, default=Path("a10_reusable_results/paper30")
    )
    parser.add_argument("--geometry-version", required=True)
    parser.add_argument("--contact-tolerance", type=float, default=0.05)
    parser.add_argument("--containment-tolerance", type=float, default=0.05)
    parser.add_argument("--support-overlap-tolerance", type=float, default=0.9)
    parser.add_argument(
        "--minimum-gap-m",
        type=float,
        default=0.002,
        help="objects closer than this to their parent are already resting",
    )
    parser.add_argument(
        "--minimum-attainable-gain",
        type=float,
        default=0.01,
        help=(
            "minimum closed-form support-term gain from closing the contact gap; "
            "below this a simulation cannot pay for itself"
        ),
    )
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--out-ids", type=Path, default=None)
    parser.add_argument("--max-targets", type=int, default=0)
    args = parser.parse_args()

    placement = load_json(args.incumbent)
    baseline_rows = load_baseline_terms(
        args.baseline_objects_csv, args.scene, args.baseline_version
    )
    if not baseline_rows:
        raise SystemExit(
            f"{args.baseline_objects_csv} has no rows for scene={args.scene} "
            f"version={args.baseline_version}"
        )

    geometry_path = evaluator.find_geometry_snapshot(
        args.saved_results, args.scene, args.geometry_version
    )
    geometry = load_json(geometry_path)
    evaluator.validate_geometry_snapshot(geometry, geometry_path)
    footprints = footprint_diagnostics(
        geometry.get("obj_info", {}), placement.get("obj_info", {})
    )

    result = screen(
        placement,
        baseline_rows,
        footprints,
        contact_tolerance=args.contact_tolerance,
        containment_tolerance=args.containment_tolerance,
        overlap_tolerance=args.support_overlap_tolerance,
        minimum_gap_m=args.minimum_gap_m,
        minimum_attainable_gain=args.minimum_attainable_gain,
    )
    selected = result["eligible"]
    if args.max_targets and len(selected) > args.max_targets:
        result["truncated_to"] = args.max_targets
        selected = selected[: args.max_targets]
    identifiers = [entry["object_id"] for entry in selected]

    report = {
        "schema_version": SCHEMA,
        "scene": args.scene,
        "baseline_version": args.baseline_version,
        "geometry_snapshot": str(geometry_path.resolve()),
        "incumbent": str(args.incumbent.resolve()),
        "thresholds": {
            "contact_tolerance": args.contact_tolerance,
            "containment_tolerance": args.containment_tolerance,
            "support_overlap_tolerance": args.support_overlap_tolerance,
            "minimum_gap_m": args.minimum_gap_m,
            "minimum_attainable_gain": args.minimum_attainable_gain,
        },
        "target_object_ids": identifiers,
        "target_count": len(identifiers),
        **result,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out_report.resolve()}")

    if args.out_ids is not None:
        args.out_ids.parent.mkdir(parents=True, exist_ok=True)
        args.out_ids.write_text(",".join(identifiers) + "\n", encoding="utf-8")
        print(f"Wrote {args.out_ids.resolve()}")

    deficit = result["support_deficit"]
    print(
        f"SCORED={result['scored_object_count']} "
        f"SUPPORT_TERMS={result['support_contributor_count']} "
        f"ELIGIBLE={result['eligible_count']} TARGETS={len(identifiers)}"
    )
    print(
        "SUPPORT_DEFICIT total={:.4f} gap={:.4f} containment={:.4f} overlap={:.4f} "
        "recoverable_by_settling={:.4f}".format(
            deficit["total"],
            deficit["contact_gap_component"],
            deficit["containment_component"],
            deficit["footprint_overlap_component"],
            deficit["recoverable_by_settling"],
        )
    )
    print(
        f"PARENT_GEOMETRY_BOUND={result['parent_geometry_bound_count']} "
        f"DEGENERATE_PARENT_FOOTPRINT={result['parent_footprint_degenerate_count']}"
    )
    for reason in EXCLUSION_ORDER:
        count = result["excluded_counts"].get(reason, 0)
        if count:
            print(f"  excluded/{reason}: {count}")
    for entry in selected[:15]:
        print(
            f"  target {entry['object_id']}: gap={entry['support_contact_gap_m']:.6f} "
            f"gain={entry['attainable_settle_gain']:.4f} "
            f"ceiling={entry['support_term_ceiling']:.4f} "
            f"parent={entry['support_parent_id']} "
            f"parent_vertices={entry['parent_footprint_vertex_count']}"
        )
    return 0 if identifiers else 4


if __name__ == "__main__":
    raise SystemExit(main())
