#!/usr/bin/env python3
"""Locate the first saved SceneProof run that changed each pillow pose."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


DEFAULT_VERSIONS = [
    "v5_sceneproof_visual_rollback_smoke1_fix43",
    "v5_sceneproof_fix43_smooth_smoke1_fix53",
    "v5_sceneproof_fix43_inloop_guarded_smoke1_fix55",
    "v5_sceneproof_fix43_inloop_certified_smoke1_fix55",
    "v5_sceneproof_fix43_smooth_smoke5_fix56",
    "v5_sceneproof_fix43_inloop_guarded_smoke5_fix56",
    "v5_sceneproof_fix43_inloop_certified_smoke5_fix56",
    "v5_sceneproof_fix43_smooth_paper30_fix61",
    "v5_sceneproof_collision_partial_commit_paper30_fix61",
    "v5_sceneproof_collision_partial_commit_certified_paper30_fix61",
    "v5_sceneproof_com_scoped_rollback_paper30_fix68",
]


def find_placement(root: Path, scene: str, version: str) -> Path | None:
    folder = root / f"{scene}_{version}_result" / "S4_layout_refinement"
    matches = sorted(folder.glob("*_placement_info_s4.json"))
    if len(matches) > 1:
        raise RuntimeError(f"multiple placements for {version}: {matches}")
    return matches[0] if matches else None


def matrix_metrics(first, second) -> dict[str, float]:
    a = [[float(value) for value in row] for row in first]
    b = [[float(value) for value in row] for row in second]
    if len(a) != 4 or len(b) != 4 or any(len(row) != 4 for row in a + b):
        raise ValueError("pose must be a pair of 4x4 matrices")
    difference = [[b[i][j] - a[i][j] for j in range(4)] for i in range(4)]
    translation = math.sqrt(sum(difference[i][3] ** 2 for i in range(3)))
    linear = math.sqrt(sum(difference[i][j] ** 2 for i in range(3) for j in range(3)))
    return {
        "max_abs": max(abs(value) for row in difference for value in row),
        "translation_norm_m": translation,
        "linear_frobenius": linear,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-results", type=Path, required=True)
    parser.add_argument("--scene", default="bedroom_01")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--versions", nargs="*", default=DEFAULT_VERSIONS)
    parser.add_argument("--change-tolerance", type=float, default=1e-7)
    args = parser.parse_args()

    runs = []
    for version in args.versions:
        path = find_placement(args.saved_results, args.scene, version)
        if path is None:
            runs.append({"version": version, "available": False})
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        poses = {
            object_id: row.get("pose_matrix_for_blender")
            for object_id, row in document.get("obj_info", {}).items()
            if object_id.startswith("pillow_")
            and isinstance(row, dict)
            and row.get("pose_matrix_for_blender") is not None
        }
        runs.append(
            {
                "version": version,
                "available": True,
                "placement": str(path.resolve()),
                "mtime_ns": path.stat().st_mtime_ns,
                "poses": poses,
            }
        )

    available = [run for run in runs if run["available"]]
    if not available:
        raise RuntimeError("none of the requested placement versions exist")
    baseline = available[0]
    pillow_ids = sorted(
        {object_id for run in available for object_id in run["poses"]}
    )
    transitions = []
    previous = None
    first_change_by_object = {object_id: None for object_id in pillow_ids}
    for run in available:
        if previous is None:
            previous = run
            continue
        changed = {}
        for object_id in pillow_ids:
            if object_id not in previous["poses"] or object_id not in run["poses"]:
                continue
            metrics = matrix_metrics(previous["poses"][object_id], run["poses"][object_id])
            if metrics["max_abs"] > args.change_tolerance:
                changed[object_id] = metrics
            if (
                first_change_by_object[object_id] is None
                and object_id in baseline["poses"]
                and object_id in run["poses"]
                and matrix_metrics(baseline["poses"][object_id], run["poses"][object_id])["max_abs"]
                > args.change_tolerance
            ):
                first_change_by_object[object_id] = {
                    "version": run["version"],
                    "previous_available_version": previous["version"],
                    "delta_from_original_fix43": matrix_metrics(
                        baseline["poses"][object_id], run["poses"][object_id]
                    ),
                }
        transitions.append(
            {
                "from": previous["version"],
                "to": run["version"],
                "changed_pillow_count": len(changed),
                "changed_pillows": changed,
            }
        )
        previous = run

    report = {
        "schema_version": "sceneproof_pillow_pose_lineage_v1",
        "scene": args.scene,
        "change_tolerance": args.change_tolerance,
        "original_fix43_version": baseline["version"],
        "runs": [
            {key: value for key, value in run.items() if key != "poses"}
            for run in runs
        ],
        "pillow_ids": pillow_ids,
        "transitions": transitions,
        "first_change_from_original_fix43_by_object": first_change_by_object,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out.resolve()}")
    print(f"ORIGINAL_FIX43={baseline['version']}")
    for transition in transitions:
        names = ",".join(transition["changed_pillows"])
        print(
            f"TRANSITION={transition['from']}->{transition['to']} "
            f"CHANGED={transition['changed_pillow_count']} OBJECTS={names or '-'}"
        )
    for object_id, first in first_change_by_object.items():
        if first is None:
            print(f"FIRST_CHANGE {object_id}=unchanged")
        else:
            metrics = first["delta_from_original_fix43"]
            print(
                f"FIRST_CHANGE {object_id}={first['version']} "
                f"translation_m={metrics['translation_norm_m']:.9g} "
                f"linear={metrics['linear_frobenius']:.9g}"
            )
    print(f"FIX75_LINEAGE={args.out.resolve()}")


if __name__ == "__main__":
    main()
