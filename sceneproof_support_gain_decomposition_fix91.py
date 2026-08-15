#!/usr/bin/env python3
"""SceneProof Fix91: split a support gain into legitimate and vacuous parts.

Why this is needed
------------------
Backfilling the measured floor geometry raised the Smoke5 support scores sharply,
for example 0.2591 to 0.9361 on ``streelitter_01``.  The measurement is correct
but it is not all meaningful, because the measured slab is exactly 10 m by 10 m
by 0.04 m: a procedural construction placeholder, not the room.

The resting-support term has three summands,

    s = ( L(gap, tol) + L(containment, tol) + min(1, overlap/tol) ) / 3,

and backfilling the slab affects all three, for very different reasons.

*``gap`` becomes correct.  The slab's top face is the surface the pipeline
  actually placed objects on, and the measurement confirms every floor child was
  resting exactly on it: the slab spans z in [-0.02, +0.02] and the children
  start at +0.02.  The previously reported 0.020000 m gap was the slab's half
  thickness, nothing else.  Recovering this summand is a genuine correction.
* ``containment`` and ``overlap`` become *trivially satisfied*.Any object in a
  4 to 6 m room lies well inside a 10 m slab, so these summands stop
  discriminating.  Recovering them inflates the score without measuring
  anything about the layout.

This tool reports the two contributions separately so the legitimate correction
is never quoted together with the vacuous one.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

STRUCTURAL_PREFIXES = ("floor_", "ground_", "wall_", "ceiling_", "carpet_", "rug_")


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
    if error is None or not np.isfinite(error):
        return 0.0
    return float(max(0.0, 1.0 - error / tolerance))


def overlap_score(ratio: float | None, tolerance: float) -> float:
    if ratio is None or not np.isfinite(ratio):
        return 0.0
    return float(min(1.0, max(0.0, ratio) / tolerance))


def format_metres(value: float | None) -> str:
    """Render a length for logs without hiding a missing measurement as zero."""
    if value is None:
        return "unmeasured"
    if not np.isfinite(value):
        return "inf"
    return f"{float(value):.6f}m"


def load_rows(path: Path, scene: str, version: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("scene") != scene or row.get("version") != version:
                continue
            object_id = row.get("object_id")
            if object_id:
                rows[object_id] = row
    return rows


def support_id(value: Any) -> str | None:
    if isinstance(value, (list, tuple)):
        value = next((item for item in value if item), None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def decompose(
    baseline_rows: dict[str, dict[str, str]],
    backfilled_rows: dict[str, dict[str, str]],
    placement: dict[str, Any],
    *,
    contact_tolerance: float,
    containment_tolerance: float,
    overlap_tolerance: float,
) -> dict[str, Any]:
    obj_info = placement.get("obj_info", {})
    contributors = [
        object_id
        for object_id in sorted(baseline_rows)
        if optional_float(baseline_rows[object_id].get("support_term")) is not None
    ]
    denominator = len(contributors)

    gap_gain = 0.0
    containment_gain = 0.0
    overlap_gain = 0.0
    structural_gap_gain = 0.0
    structural_lateral_gain = 0.0
    per_object: list[dict[str, Any]] = []

    for object_id in contributors:
        before, after = baseline_rows[object_id], backfilled_rows.get(object_id)
        if after is None:
            continue
        info = obj_info.get(object_id)
        parent_id = support_id(info.get("supported")) if isinstance(info, dict) else None
        parent_is_structural = bool(
            parent_id and str(parent_id).startswith(STRUCTURAL_PREFIXES)
        )

        pieces = {}
        for key, scorer, tolerance in (
            ("support_contact_gap_m", linear_score, contact_tolerance),
            ("support_containment_error_m", linear_score, containment_tolerance),
            ("support_footprint_overlap_ratio", overlap_score, overlap_tolerance),
        ):
            left = scorer(optional_float(before.get(key)), tolerance)
            right = scorer(optional_float(after.get(key)), tolerance)
            pieces[key] = (right - left) / 3.0

        gap_delta = pieces["support_contact_gap_m"]
        lateral_delta = (
            pieces["support_containment_error_m"]
            + pieces["support_footprint_overlap_ratio"]
        )
        gap_gain += gap_delta
        containment_gain += pieces["support_containment_error_m"]
        overlap_gain += pieces["support_footprint_overlap_ratio"]
        if parent_is_structural:
            structural_gap_gain += gap_delta
            structural_lateral_gain += lateral_delta

        if abs(gap_delta) > 1e-12 or abs(lateral_delta) > 1e-12:
            per_object.append(
                {
                    "object_id": object_id,
                    "support_parent_id": parent_id,
                    "parent_is_structural_placeholder": parent_is_structural,
                    "support_term_before": optional_float(
                        before.get("support_term")
                    ),
                    "support_term_after": optional_float(after.get("support_term")),
                    "contact_gap_before_m": optional_float(
                        before.get("support_contact_gap_m")
                    ),
                    "contact_gap_after_m": optional_float(
                        after.get("support_contact_gap_m")
                    ),
                    "legitimate_gap_gain": gap_delta,
                    "vacuous_lateral_gain": lateral_delta,
                }
            )

    scale = max(denominator, 1)
    per_object.sort(key=lambda item: -abs(item["vacuous_lateral_gain"]))
    total = (gap_gain + containment_gain + overlap_gain) / scale
    return {
        "support_term_count": denominator,
        "total_support_delta": total,
        "legitimate_contact_gap_delta": gap_gain / scale,
        "vacuous_containment_delta": containment_gain / scale,
        "vacuous_overlap_delta": overlap_gain / scale,
        "vacuous_lateral_delta": (containment_gain + overlap_gain) / scale,
        "legitimate_fraction": (
            (gap_gain / (gap_gain + containment_gain + overlap_gain))
            if abs(gap_gain + containment_gain + overlap_gain) > 1e-12
            else None
        ),
        "structural_parent_contact_gap_delta": structural_gap_gain / scale,
        "structural_parent_lateral_delta": structural_lateral_gain / scale,
        "objects_changed": len(per_object),
        "per_object": per_object,
    }


def format_per_object_line(entry: dict[str, Any]) -> str:
    """Render one per-object line with every value bound to its own label.

    An earlier version passed nine arguments to a format string holding eight
    fields, which shifted every value one slot to the left and printed the
    boolean ``parent_is_structural_placeholder`` as ``term 1.0000``.  The
    aggregate numbers never went through this path and were unaffected, but the
    per-object line was misleading, so the binding is now by keyword.
    """
    return (
        "  {object_id}: parent={parent} placeholder={placeholder} "
        "term {term_before:.4f}->{term_after:.4f} "
        "gap {gap_before}->{gap_after} "
        "legit={legit:+.4f} vacuous={vacuous:+.4f}".format(
            object_id=entry["object_id"],
            parent=entry["support_parent_id"],
            placeholder=entry["parent_is_structural_placeholder"],
            term_before=entry["support_term_before"] or 0.0,
            term_after=entry["support_term_after"] or 0.0,
            gap_before=format_metres(entry["contact_gap_before_m"]),
            gap_after=format_metres(entry["contact_gap_after_m"]),
            legit=entry["legitimate_gap_gain"],
            vacuous=entry["vacuous_lateral_gain"],
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--baseline-objects-csv", type=Path, required=True)
    parser.add_argument("--backfilled-objects-csv", type=Path, required=True)
    parser.add_argument("--contact-tolerance", type=float, default=0.05)
    parser.add_argument("--containment-tolerance", type=float, default=0.05)
    parser.add_argument("--support-overlap-tolerance", type=float, default=0.9)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--max-listed", type=int, default=10)
    args = parser.parse_args()

    with args.placement.open("r", encoding="utf-8") as handle:
        placement = json.load(handle)
    baseline_rows = load_rows(args.baseline_objects_csv, args.scene, args.version)
    backfilled_rows = load_rows(args.backfilled_objects_csv, args.scene, args.version)
    if not baseline_rows or not backfilled_rows:
        raise SystemExit(
            f"no rows for scene={args.scene} version={args.version} in one of the "
            "object CSVs"
        )

    result = decompose(
        baseline_rows,
        backfilled_rows,
        placement,
        contact_tolerance=args.contact_tolerance,
        containment_tolerance=args.containment_tolerance,
        overlap_tolerance=args.support_overlap_tolerance,
    )
    report = {
        "schema_version": "sceneproof_support_gain_decomposition_v1",
        "scene": args.scene,
        "version": args.version,
        "baseline_objects_csv": str(args.baseline_objects_csv.resolve()),
        "backfilled_objects_csv": str(args.backfilled_objects_csv.resolve()),
        **result,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out_report.resolve()}")
    fraction = result["legitimate_fraction"]
    print(
        "SUPPORT_DELTA total={:+.4f} legitimate_gap={:+.4f} "
        "vacuous_lateral={:+.4f} legitimate_fraction={}".format(
            result["total_support_delta"],
            result["legitimate_contact_gap_delta"],
            result["vacuous_lateral_delta"],
            "n/a" if fraction is None else f"{fraction:.1%}",
        )
    )
    print(
        "  of which against structural placeholders: gap={:+.4f} lateral={:+.4f}".format(
            result["structural_parent_contact_gap_delta"],
            result["structural_parent_lateral_delta"],
        )
    )
    for entry in result["per_object"][: args.max_listed]:
        print(format_per_object_line(entry))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
