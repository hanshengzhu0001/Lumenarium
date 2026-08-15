#!/usr/bin/env python3
"""GT-free selector for repeated Lumenarium cold starts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from eval_physical_realizability import evaluate_scene, validate_geometry_snapshot


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def physical_args() -> SimpleNamespace:
    return SimpleNamespace(
        collision_volume_tolerance=1e-6,
        collision_fraction_tolerance=0.05,
        contact_tolerance=0.05,
        containment_tolerance=0.05,
        support_overlap_tolerance=0.9,
        plane_tolerance=0.05,
        plane_orientation_tolerance=15.0,
        boundary_tolerance=0.05,
        semantic_angle_tolerance=20.0,
        distance_tolerance=0.1,
        collision_pairs_csv="in_memory",
    )


def relation_counts(document: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    programs = document.get("sceneproof_relation_programs", {}).get("programs", [])
    for program in programs:
        kind = str(program.get("kind", "UNKNOWN")).upper()
        result[kind] = result.get(kind, 0) + 1
    return result


def fast_proxy(document: dict[str, Any]) -> dict[str, Any]:
    objects = document.get("obj_info", {})
    rows = []
    missing_parent = 0
    invalid_size = 0
    volumes = []
    for object_id, info in objects.items():
        if not isinstance(info, dict) or object_id == "scene_camera":
            continue
        size = info.get("pcd_obb_size", info.get("length"))
        try:
            size = np.asarray(size, dtype=np.float64)
            volume = float(np.prod(size))
        except (TypeError, ValueError):
            invalid_size += 1
            continue
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            invalid_size += 1
            continue
        volumes.append(volume)
        parent = info.get("supported")
        if isinstance(parent, str) and parent not in objects:
            missing_parent += 1
        rows.append((object_id, volume, float(np.max(size))))
    median = float(np.median(volumes)) if volumes else 0.0
    extreme = [
        object_id for object_id, volume, maximum in rows
        if maximum > 4.5 or (median > 0 and volume > max(2.0, 100.0 * median))
    ]
    return {
        "object_count": len(rows), "invalid_size_count": invalid_size,
        "missing_support_parent_count": missing_parent,
        "extreme_geometry_object_ids": extreme,
    }


def high_measure(candidate: dict[str, Any]) -> dict[str, Any]:
    geometry_path = Path(candidate["geometry_path"])
    placement_path = Path(candidate["placement_path"])
    geometry = load(geometry_path)
    placement = load(placement_path)
    validate_geometry_snapshot(geometry, geometry_path)
    metrics, _ = evaluate_scene(geometry, placement, physical_args())
    pairs = metrics.pop("collision_pair_details", [])
    severe = [
        row for row in pairs
        if float(row.get("overlap_fraction", 0.0)) >= 0.8
        or float(row.get("penetration_depth_m", 0.0)) >= 0.25
    ]
    kinds = relation_counts(placement)
    attachment_programs = sum(
        kinds.get(kind, 0) for kind in ("PLANE_ATTACH", "CEILING_ATTACH", "HANG")
    )
    return {
        "metrics": metrics,
        "relation_program_kinds": kinds,
        "relation_program_count": sum(kinds.values()),
        "attachment_program_count": attachment_programs,
        "severe_collision_pair_count": len(severe),
        "severe_collision_pairs": [
            {
                "first_id": row.get("first_id"), "second_id": row.get("second_id"),
                "overlap_fraction": row.get("overlap_fraction"),
                "penetration_depth_m": row.get("penetration_depth_m"),
            }
            for row in severe
        ],
    }


def rank(row: dict[str, Any], mode: str) -> tuple[Any, ...]:
    if mode == "fast":
        proxy = row["fast"]
        return (
            -len(proxy["extreme_geometry_object_ids"]),
            -proxy["missing_support_parent_count"],
            -proxy["invalid_size_count"],
        )
    high = row["high"]
    metrics = high["metrics"]
    critical = float(metrics.get("headline_critical_realizability") or 0.0)
    macro = float(metrics.get("headline_macro_realizability") or 0.0)
    hard_pass = critical > 0.0 and high["severe_collision_pair_count"] <= 2
    return (
        int(hard_pass),
        -high["severe_collision_pair_count"],
        critical,
        macro,
        -int(metrics.get("unintended_collision_pairs", 0)),
        high["relation_program_count"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--mode", choices=("fast", "high"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    specification = load(args.candidates)
    rows = []
    failures = []
    for candidate in specification.get("candidates", []):
        row = {"candidate_id": candidate["candidate_id"]}
        try:
            if args.mode == "fast":
                row["fast"] = fast_proxy(load(candidate["s3_path"]))
            else:
                row["high"] = high_measure(candidate)
            row["rank"] = list(rank(row, args.mode))
            rows.append(row)
        except Exception as error:
            failures.append({"candidate_id": candidate.get("candidate_id"), "error": repr(error)})
    selected = max(rows, key=lambda row: tuple(row["rank"])) if rows else None
    report = {
        "schema_version": "sceneproof_cold_start_selector_v1",
        "mode": args.mode, "gt_free": True,
        "candidate_count": len(specification.get("candidates", [])),
        "completed": len(rows), "failures": failures,
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "candidates": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"MODE={args.mode} CANDIDATES={report['candidate_count']} COMPLETED={len(rows)} FAILURES={len(failures)} SELECTED={report['selected_candidate_id']}")
    for row in sorted(rows, key=lambda item: tuple(item["rank"]), reverse=True):
        print(f"{row['candidate_id']} rank={row['rank']}")
    print(f"COLD_START_SELECTOR={args.out.resolve()}")


if __name__ == "__main__":
    main()
