#!/usr/bin/env python3
"""SceneProof Fix93: is the collision normalization a well-posed measurement?

The evidence that forced this question
--------------------------------------
The first Smoke5 pair export contains two pairs that, taken together, show the
current normalization ranks overlaps in the wrong order:

    livingroom_10  book_1 vs book_2          volume 0.000009 m3fraction 0.5067
    streelitter_01 trash_bin_1 vs paper_cup_1 volume 0.061128 m3   fraction 0.0359

The first is nine cubic *millimetres*, a cube two millimetres on a side, and it
drives both books' collision terms to zero.  The second is roughly six thousand
times more material passing through material, and it is not penalised at all
because ``collision_fraction_tolerance`` is 0.05.  Two more of the same kind:
``book_5 vs cup_6`` at 63 mm3 scores 0.4046, while
``discarded_wooden_board_11 vs empty_can_2`` at 38 mm3 escapes at 0.0446.

The cause is the definition, not the data:

    fraction = intersection_volume / min(volume_a, volume_b)

The denominator spans five orders of magnitude across a single scene, from a book
at about 1.8e-5 m3 to a bin at about 1.7 m3.  So the fraction answers "what share
of the smaller object is overlapped", which scales with the *inverse size of the
smaller object*, not "how badly do these two objects interpenetrate".  A metric
built on it rewards a solver for cleaning up millimetre overlaps between small
props whileignoring litre-scale interpenetration between large furniture.

What this tool does
-------------------
It reports the same scene under three normalizations, all computed from the same
exported pairs, so the disagreement is visible rather than asserted:

``fraction``
    Today's definition.  Reported unchanged, so the paper's current number is
    reproduced exactly.
``depth``
    ``L(max_penetration_depth, --depth-tolerance)``.  The depth is the exact
    minimum translation distance for the two prisms, so it is in metres and is
    scale-correct: a millimetre reads as a millimetre for a book and for a bin.
``volume``
    ``L(max_intersection_volume, --volume-tolerance)``.  Absolute material
    overlap, the quantity a fabricator or a renderer would care about.

It then measures how much the three disagree: the rank correlation between the
per-object terms, and the objects whose verdict flips.  A metric and its
scale-corrected counterpart disagreeing on *which objects are bad* is the
operational meaning of an ill-posed normalization.

This tool does not change any score.  It produces the evidence needed to decide
what the paper should report, and that decision is left explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

NORMALIZATIONS = ("fraction", "depth", "volume")


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
        return 1.0
    return float(max(0.0, 1.0 - error / tolerance))


def spearman(left: list[float], right: list[float]) -> float | None:
    """Rank correlation with midranks for ties, without a SciPy dependency."""
    if len(left) != len(right) or len(left) < 2:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while (
                end + 1 < len(order)
                and values[order[end + 1]] == values[order[position]]
            ):
                end += 1
            midrank = (position + end) / 2.0 + 1.0
            for index in range(position, end + 1):
                result[order[index]] = midrank
            position = end + 1
        return result

    first, second = ranks(left), ranks(right)
    mean_first = sum(first) / len(first)
    mean_second = sum(second) / len(second)
    covariance = sum(
        (a - mean_first) * (b - mean_second) for a, b in zip(first, second)
    )
    variance_first = sum((a - mean_first) ** 2 for a in first)
    variance_second = sum((b - mean_second) ** 2 for b in second)
    if variance_first <= 0.0 or variance_second <= 0.0:
        # One of the two assigns the same term to every object, so there is no
        # ranking to compare.  That is itself informative and is reported as None.
        return None
    return float(covariance / (variance_first * variance_second) ** 0.5)


def worst_per_object(
    rows: list[dict[str, Any]], object_ids: list[str], column: str
) -> dict[str, float]:
    worst = {object_id: 0.0 for object_id in object_ids}
    for row in rows:
        value = optional_float(row.get(column)) or 0.0
        for key in ("first_id", "second_id"):
            object_id = row.get(key)
            if object_id in worst:
                worst[object_id] = max(worst[object_id], value)
    return worst


def audit_normalizations(
    rows: list[dict[str, Any]],
    object_ids: list[str],
    *,
    fraction_tolerance: float,
    depth_tolerance: float,
    volume_tolerance: float,
) -> dict[str, Any]:
    columns = {
        "fraction": ("overlap_fraction", fraction_tolerance),
        "depth": ("penetration_depth_m", depth_tolerance),
        "volume": ("intersection_volume_m3", volume_tolerance),
    }
    terms: dict[str, dict[str, float]] = {}
    scores: dict[str, float | None] = {}
    for name, (column, tolerance) in columns.items():
        worst = worst_per_object(rows, object_ids, column)
        terms[name] = {
            object_id: linear_score(value, tolerance)
            for object_id, value in worst.items()
        }
        scores[name] = (
            sum(terms[name].values()) / len(object_ids) if object_ids else None
        )

    agreement: dict[str, Any] = {}
    for index, left in enumerate(NORMALIZATIONS):
        for right in NORMALIZATIONS[index + 1 :]:
            left_values = [terms[left][object_id] for object_id in object_ids]
            right_values = [terms[right][object_id] for object_id in object_ids]
            flipped = sorted(
                object_id
                for object_id in object_ids
                if (terms[left][object_id] <= 0.0) != (terms[right][object_id] <= 0.0)
            )
            agreement[f"{left}_vs_{right}"] = {
                "spearman_rank_correlation": spearman(left_values, right_values),
                "objects_with_flipped_verdict": len(flipped),
                "object_ids_with_flipped_verdict": flipped[:20],
            }

    # The pairs that make the disagreement concrete: smallest absolute overlap
    # that is nonetheless fully penalised, and largest that escapes entirely.
    penalised = [
        row
        for row in rows
        if (optional_float(row.get("overlap_fraction")) or 0.0) > fraction_tolerance
    ]
    escaped = [
        row
        for row in rows
        if (optional_float(row.get("overlap_fraction")) or 0.0) <= fraction_tolerance
    ]

    def summarize(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "first_id": row.get("first_id"),
            "second_id": row.get("second_id"),
            "intersection_volume_m3": optional_float(
                row.get("intersection_volume_m3")
            ),
            "penetration_depth_m": optional_float(row.get("penetration_depth_m")),
            "overlap_fraction": optional_float(row.get("overlap_fraction")),
        }

    smallest_penalised = min(
        penalised,
        key=lambda row: optional_float(row.get("intersection_volume_m3")) or 0.0,
        default=None,
    )
    largest_escaped = max(
        escaped,
        key=lambda row: optional_float(row.get("intersection_volume_m3")) or 0.0,
        default=None,
    )
    inversion = None
    if smallest_penalised is not None and largest_escaped is not None:
        small = optional_float(smallest_penalised.get("intersection_volume_m3")) or 0.0
        large = optional_float(largest_escaped.get("intersection_volume_m3")) or 0.0
        inversion = {
            "smallest_fully_penalised_pair": summarize(smallest_penalised),
            "largest_unpenalised_pair": summarize(largest_escaped),
            "unpenalised_over_penalised_volume_ratio": (
                float(large / small) if small > 0.0 else None
            ),
            "ordering_is_inverted": bool(large > small),
        }
    return {
        "object_count": len(object_ids),
        "reported_pair_count": len(rows),
        "scores": scores,
        "tolerances": {
            "fraction": fraction_tolerance,
            "depth_m": depth_tolerance,
            "volume_m3": volume_tolerance,
        },
        "agreement": agreement,
        "absolute_scale_inversion": inversion,
        "interpretation": {
            "fraction_denominator_is_the_smaller_object_volume": True,
            "fraction_is_not_scale_invariant": True,
            "depth_is_exact_minimum_translation_for_prisms": True,
            "no_score_here_replaces_the_evaluator_score": True,
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
    parser.add_argument("--depth-tolerance", type=float, default=0.01)
    parser.add_argument("--volume-tolerance", type=float, default=0.001)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()

    object_ids = load_object_ids(args.objects_csv, args.scene, args.version)
    if not object_ids:
        raise SystemExit(
            f"no collision terms for scene={args.scene} version={args.version} in "
            f"{args.objects_csv}"
        )
    rows = load_pairs(args.collision_pairs_csv, args.scene, args.version)
    if rows and "penetration_depth_m" not in rows[0]:
        raise SystemExit(
            "the pair CSV predates the penetration-depth column; re-run "
            "eval_physical_realizability.py with --collision-pairs-csv"
        )
    result = audit_normalizations(
        rows,
        object_ids,
        fraction_tolerance=args.collision_fraction_tolerance,
        depth_tolerance=args.depth_tolerance,
        volume_tolerance=args.volume_tolerance,
    )
    report = {
        "schema_version": "sceneproof_collision_normalization_audit_v1",
        "scene": args.scene,
        "version": args.version,
        **result,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out_report.resolve()}")

    def show(value: float | None, digits: int = 4) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    print(
        "NORMALIZATION objects={} pairs={}".format(
            result["object_count"], result["reported_pair_count"]
        )
    )
    for name in NORMALIZATIONS:
        print(
            "  {:<10} score={}   tolerance={}".format(
                name,
                show(result["scores"][name]),
                result["tolerances"][
                    {"fraction": "fraction", "depth": "depth_m", "volume": "volume_m3"}[
                        name
                    ]
                ],
            )
        )
    for key, block in result["agreement"].items():
        print(
            "  {:<20} spearman={:<9} flipped_objects={}".format(
                key,
                show(block["spearman_rank_correlation"], 3),
                block["objects_with_flipped_verdict"],
            )
        )
    inversion = result["absolute_scale_inversion"]
    if inversion:
        small = inversion["smallest_fully_penalised_pair"]
        large = inversion["largest_unpenalised_pair"]
        print(
            "  smallest penalised: {} vs {} volume={:.9f}m3 depth={}m "
            "fraction={:.4f}".format(
                small["first_id"],
                small["second_id"],
                small["intersection_volume_m3"] or 0.0,
                show(small["penetration_depth_m"], 6),
                small["overlap_fraction"] or 0.0,
            )
        )
        print(
            "  largest unpenalised: {} vs {} volume={:.9f}m3 depth={}m "
            "fraction={:.4f}".format(
                large["first_id"],
                large["second_id"],
                large["intersection_volume_m3"] or 0.0,
                show(large["penetration_depth_m"], 6),
                large["overlap_fraction"] or 0.0,
            )
        )
        print(
            "  ordering inverted={}  unpenalised/penalised volume ratio={}".format(
                inversion["ordering_is_inverted"],
                show(inversion["unpenalised_over_penalised_volume_ratio"], 1),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
