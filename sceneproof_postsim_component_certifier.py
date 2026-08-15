#!/usr/bin/env python3
"""Materialize a GT-free, fail-closed post-simulation SceneProof result.

The guarded Schur trial is speculative.  This tool compares its serialized
post-simulation state with the smooth incumbent using the exact frozen-geometry
physical evaluator, rolls back only witness relation components, and falls
back to the complete incumbent when scoped rollback cannot certify every
headline family independently.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import time
from types import SimpleNamespace
from typing import Any, Iterable

STRUCTURAL = re.compile(r"^(floor|ground|wall|ceiling|carpet|rug)_\d+$")
HEADLINE_FAMILIES = ("collision", "support", "plane", "semantic")


def evaluator_args() -> SimpleNamespace:
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
    )


def matrix_values(info: dict[str, Any]) -> tuple[float, ...] | None:
    try:
        rows = info.get("pose_matrix_for_blender")
        if not isinstance(rows, list) or len(rows) != 4:
            return None
        value = tuple(float(item) for row in rows for item in row)
    except (TypeError, ValueError):
        return None
    if len(value) != 16 or not all(math.isfinite(item) for item in value):
        return None
    return value


def changed_objects(
    incumbent: dict[str, Any], candidate: dict[str, Any], tolerance: float = 1e-7
) -> set[str]:
    result: set[str] = set()
    first = incumbent.get("obj_info", {})
    second = candidate.get("obj_info", {})
    for object_id in set(first) & set(second):
        a = matrix_values(first[object_id])
        b = matrix_values(second[object_id])
        delta = (
            math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
            if a is not None and b is not None
            else 0.0
        )
        if delta > tolerance:
            result.add(object_id)
    return result


def relation_neighborhoods(obj_info: dict[str, Any]) -> dict[str, set[str]]:
    """Build local components without joining everything through floor/walls."""
    adjacency = {object_id: set() for object_id in obj_info}

    def connect(first: str, second: Any) -> None:
        if (
            first not in adjacency
            or not isinstance(second, str)
            or second not in adjacency
            or first == second
            or STRUCTURAL.match(second)
        ):
            return
        adjacency[first].add(second)
        adjacency[second].add(first)

    for object_id, info in obj_info.items():
        connect(object_id, info.get("supported"))
        for key in (
            "directlyFacing",
            "point_towards",
            "pointTowards",
            "alignWith",
            "align_with",
        ):
            value = info.get(key)
            entries = value if isinstance(value, list) else [value]
            for entry in entries:
                if isinstance(entry, str):
                    connect(object_id, entry)
                elif isinstance(entry, dict):
                    for target_key in ("target", "object", "id", "reference"):
                        if isinstance(entry.get(target_key), str):
                            connect(object_id, entry[target_key])
                            break
                elif isinstance(entry, (tuple, list)) and entry:
                    connect(object_id, entry[0])
    return adjacency


def scoped_component(
    witness: str,
    adjacency: dict[str, set[str]],
    changed: set[str],
) -> set[str]:
    """Return the changed witness component and unchanged separator boundary."""
    component = {witness}
    frontier = [witness]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency.get(current, ()):
            if neighbor in component:
                continue
            component.add(neighbor)
            if neighbor in changed:
                frontier.append(neighbor)
    return component


def family_gates(
    incumbent: dict[str, Any],
    candidate: dict[str, Any],
    margin: float,
) -> tuple[bool, dict[str, dict[str, Any]]]:
    rows: dict[str, dict[str, Any]] = {}
    for family in HEADLINE_FAMILIES:
        first = incumbent["families"].get(family, {}).get("score")
        second = candidate["families"].get(family, {}).get("score")
        evaluable = first is not None and second is not None
        delta = float(second) - float(first) if evaluable else None
        rows[family] = {
            "incumbent": first,
            "candidate": second,
            "delta": delta,
            "passed": bool(not evaluable or delta >= -margin),
        }
    first_macro = incumbent.get("headline_macro_realizability")
    second_macro = candidate.get("headline_macro_realizability")
    delta = float(second_macro) - float(first_macro)
    rows["physical_macro"] = {
        "incumbent": first_macro,
        "candidate": second_macro,
        "delta": delta,
        "passed": delta >= -margin,
    }
    return all(row["passed"] for row in rows.values()), rows


def local_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["object_id"]): row for row in rows}


def finite_delta(
    incumbent: dict[str, Any], candidate: dict[str, Any], field: str
) -> float | None:
    try:
        first, second = float(incumbent[field]), float(candidate[field])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(first) or not math.isfinite(second):
        return None
    return second - first


def witness_harm(
    incumbent: dict[str, Any], candidate: dict[str, Any], failed: set[str]
) -> float:
    harms: list[float] = []
    if "collision" in failed:
        delta = finite_delta(incumbent, candidate, "collision_overlap_fraction")
        if delta is not None:
            harms.append(delta)
    if "support" in failed:
        for field in ("support_contact_gap_m", "support_containment_error_m"):
            delta = finite_delta(incumbent, candidate, field)
            if delta is not None:
                harms.append(delta)
        delta = finite_delta(
            incumbent, candidate, "support_footprint_overlap_ratio"
        )
        if delta is not None:
            harms.append(-delta)
    if "plane" in failed:
        for field in ("plane_contact_gap_m", "plane_orientation_error_deg"):
            delta = finite_delta(incumbent, candidate, field)
            if delta is not None:
                harms.append(delta)
    if "semantic" in failed:
        delta = finite_delta(incumbent, candidate, "semantic_error")
        if delta is not None:
            harms.append(delta)
    local_delta = finite_delta(incumbent, candidate, "local_realizability")
    if local_delta is not None:
        harms.append(-local_delta)
    return max(harms, default=0.0)


def rollback_poses(
    selected: dict[str, Any], incumbent: dict[str, Any], object_ids: Iterable[str]
) -> None:
    for object_id in object_ids:
        if object_id not in selected.get("obj_info", {}):
            continue
        if object_id not in incumbent.get("obj_info", {}):
            continue
        pose = incumbent["obj_info"][object_id].get("pose_matrix_for_blender")
        if pose is not None:
            selected["obj_info"][object_id]["pose_matrix_for_blender"] = copy.deepcopy(pose)


def certify_scene(
    source: dict[str, Any],
    incumbent: dict[str, Any],
    candidate: dict[str, Any],
    *,
    margin: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from eval_physical_realizability import evaluate_scene

    args = evaluator_args()
    incumbent_metrics, incumbent_local_values = evaluate_scene(source, incumbent, args)
    selected = copy.deepcopy(candidate)
    initial_changed = changed_objects(incumbent, candidate)
    adjacency = relation_neighborhoods(selected.get("obj_info", {}))
    incumbent_local = local_rows(incumbent_local_values)
    rolled_back: set[str] = set()
    records: list[dict[str, Any]] = []

    for iteration in range(max(len(initial_changed), 1) + 1):
        selected_metrics, selected_local_values = evaluate_scene(source, selected, args)
        passed, gates = family_gates(incumbent_metrics, selected_metrics, margin)
        records.append(
            {
                "iteration": iteration,
                "rolled_back_objects": sorted(rolled_back),
                "gates": gates,
            }
        )
        if passed:
            return selected, {
                "accepted": True,
                "full_incumbent_fallback": False,
                "initial_changed_objects": sorted(initial_changed),
                "rolled_back_objects": sorted(rolled_back),
                "retained_changed_objects": sorted(initial_changed - rolled_back),
                "records": records,
            }

        failed = {
            family
            for family, row in gates.items()
            if family in HEADLINE_FAMILIES and not row["passed"]
        }
        selected_local = local_rows(selected_local_values)
        ranked = sorted(
            (
                (
                    witness_harm(incumbent_local[object_id], row, failed),
                    object_id,
                )
                for object_id, row in selected_local.items()
                if object_id in incumbent_local
                and object_id in initial_changed
                and object_id not in rolled_back
            ),
            reverse=True,
        )
        if not ranked or ranked[0][0] <= 0.0:
            break
        _, witness = ranked[0]
        component = scoped_component(witness, adjacency, initial_changed)
        rollback = (component & initial_changed) - rolled_back
        if not rollback:
            break
        rollback_poses(selected, incumbent, rollback)
        rolled_back.update(rollback)

    # The certificate is fail-closed.  The exact incumbent is always safe by
    # construction and avoids accepting a macro improvement over a failed
    # component.
    selected = copy.deepcopy(incumbent)
    selected_metrics, _ = evaluate_scene(source, selected, args)
    passed, gates = family_gates(incumbent_metrics, selected_metrics, margin)
    if not passed:
        raise RuntimeError("exact incumbent failed its own post-sim certificate")
    records.append(
        {
            "iteration": len(records),
            "rolled_back_objects": sorted(initial_changed),
            "gates": gates,
            "state": "full_incumbent_fallback",
        }
    )
    return selected, {
        "accepted": False,
        "full_incumbent_fallback": True,
        "initial_changed_objects": sorted(initial_changed),
        "rolled_back_objects": sorted(initial_changed),
        "retained_changed_objects": [],
        "records": records,
    }


def main() -> None:
    from eval_physical_realizability import (
        find_geometry_snapshot,
        find_s4,
        load_json,
        validate_geometry_snapshot,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-results", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--geometry-version", default="v4_deepsearch")
    parser.add_argument("--incumbent-version", required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--target-version", required=True)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--runtime-jsonl",
        type=Path,
        help="Optional per-scene CPU certificate timing log.",
    )
    args = parser.parse_args()

    scenes = [
        line.strip()
        for line in args.scenes.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report: dict[str, Any] = {
        "schema_version": "sceneproof_postsim_component_certificate_v1",
        "incumbent_version": args.incumbent_version,
        "candidate_version": args.candidate_version,
        "target_version": args.target_version,
        "requested_geometry_version": args.geometry_version,
        "margin": args.margin,
        "scenes": {},
        "failures": [],
    }
    runtime_rows: list[dict[str, Any]] = []
    for scene in scenes:
        start = time.perf_counter()
        try:
            geometry_version = args.geometry_version
            try:
                source_path = find_geometry_snapshot(
                    args.saved_results, scene, geometry_version
                )
            except FileNotFoundError as requested_error:
                # A clean S1--S4 run stops the source branch after S3.  The
                # incumbent SceneLM branch then materializes the required
                # imported-asset bbox/length snapshot immediately before S4.
                # This is still frozen pre-optimization geometry; it is not an
                # incumbent or candidate pose result.
                geometry_version = args.incumbent_version
                try:
                    source_path = find_geometry_snapshot(
                        args.saved_results, scene, geometry_version
                    )
                except FileNotFoundError:
                    raise requested_error
            source = load_json(source_path)
            validate_geometry_snapshot(source, source_path)
            incumbent_path = find_s4(
                args.saved_results, scene, args.incumbent_version
            )
            candidate_path = find_s4(
                args.saved_results, scene, args.candidate_version
            )
            incumbent = load_json(incumbent_path)
            candidate = load_json(candidate_path)
            selected, certificate = certify_scene(
                source, incumbent, candidate, margin=args.margin
            )
            selected["sceneproof_postsim_component_certificate"] = certificate
            output_dir = (
                args.saved_results
                / f"{scene}_{args.target_version}_result"
                / "S4_layout_refinement"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{scene}_{args.target_version}_placement_info_s4.json"
            output_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
            report["scenes"][scene] = {
                **certificate,
                "output": str(output_path),
                "geometry_snapshot_version": geometry_version,
                "geometry_snapshot_path": str(source_path),
                "geometry_snapshot_stage": "pre_s4_imported_asset_geometry",
            }
            elapsed = time.perf_counter() - start
            report["scenes"][scene]["certificate_seconds"] = elapsed
            runtime_rows.append(
                {
                    "scene": scene,
                    "version": f"{args.target_version}_certificate",
                    "engine": "sceneproof_certificate",
                    "stage": "postsim_component_certificate",
                    "gpu": None,
                    "elapsed_seconds": elapsed,
                    "status": "ok",
                    "return_code": 0,
                }
            )
            print(
                f"{scene} accepted={certificate['accepted']} "
                f"fallback={certificate['full_incumbent_fallback']} "
                f"changed={len(certificate['initial_changed_objects'])} "
                f"rolled_back={len(certificate['rolled_back_objects'])} "
                f"retained={len(certificate['retained_changed_objects'])}"
            )
        except Exception as error:  # fail closed and preserve the diagnostic
            elapsed = time.perf_counter() - start
            report["failures"].append({"scene": scene, "error": repr(error)})
            runtime_rows.append(
                {
                    "scene": scene,
                    "version": f"{args.target_version}_certificate",
                    "engine": "sceneproof_certificate",
                    "stage": "postsim_component_certificate",
                    "gpu": None,
                    "elapsed_seconds": elapsed,
                    "status": "fail",
                    "return_code": 1,
                }
            )
            print(f"FAIL scene={scene} error={error!r}")
    report["completed"] = len(report["scenes"])
    report["accepted"] = sum(
        int(row["accepted"]) for row in report["scenes"].values()
    )
    report["fallbacks"] = sum(
        int(row["full_incumbent_fallback"])
        for row in report["scenes"].values()
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.runtime_jsonl is not None:
        args.runtime_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.runtime_jsonl.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in runtime_rows
            ),
            encoding="utf-8",
        )
    print(f"Wrote {args.out}")
    print(
        f"SCENES={report['completed']}/{len(scenes)} "
        f"FAILURES={len(report['failures'])} "
        f"ACCEPTED={report['accepted']} FALLBACKS={report['fallbacks']}"
    )
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
