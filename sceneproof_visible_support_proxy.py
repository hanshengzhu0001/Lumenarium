"""Cheap, fail-closed visible support preprocessing for online SceneProof.

This module deliberately uses only saved OBBs, relation programs, and the
existing color-ID visibility audit.  It never invokes Blender, BVH, voxels, or
rigid-body simulation.  The resulting certificate is therefore a proxy
certificate, not a true-mesh physical proof.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


STRUCTURAL_PREFIXES = ("floor_", "wall_", "ceiling_", "room_")


def _parents(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _bbox_corners(info: dict[str, Any]) -> list[list[float]] | None:
    bbox = info.get("bbox") or info.get("bounding_box")
    if not isinstance(bbox, list):
        return None
    if len(bbox) == 2 and all(isinstance(row, list) and len(row) >= 3 for row in bbox):
        lo, hi = bbox
        return [
            [x, y, z]
            for x in (float(lo[0]), float(hi[0]))
            for y in (float(lo[1]), float(hi[1]))
            for z in (float(lo[2]), float(hi[2]))
        ]
    if len(bbox) >= 8 and all(isinstance(row, list) and len(row) >= 3 for row in bbox):
        return [[float(row[0]), float(row[1]), float(row[2])] for row in bbox]
    return None


def _world_aabb(info: dict[str, Any]) -> list[float] | None:
    matrix = info.get("pose_matrix_for_blender")
    corners = _bbox_corners(info)
    if not corners or not isinstance(matrix, list) or len(matrix) < 4:
        return None
    points = []
    try:
        for x, y, z in corners:
            points.append([
                float(matrix[row][0]) * x
                + float(matrix[row][1]) * y
                + float(matrix[row][2]) * z
                + float(matrix[row][3])
                for row in range(3)
            ])
    except (IndexError, TypeError, ValueError):
        return None
    return [
        min(point[0] for point in points), max(point[0] for point in points),
        min(point[1] for point in points), max(point[1] for point in points),
        min(point[2] for point in points), max(point[2] for point in points),
    ]


def _xy_overlap(a: list[float], b: list[float]) -> float:
    x = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    y = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    child_area = max((a[1] - a[0]) * (a[3] - a[2]), 1e-9)
    return x * y / child_area


def _overlap_volume(a: list[float], b: list[float]) -> float:
    return (
        max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
        * max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
        * max(0.0, min(a[5], b[5]) - max(a[4], b[4]))
    )


def _program_attachment_objects(data: dict[str, Any]) -> set[str]:
    attached: set[str] = set()
    bundle = data.get("sceneproof_relation_programs", {})
    for program in bundle.get("programs", []) if isinstance(bundle, dict) else []:
        if program.get("kind") != "PLANE_ATTACH":
            continue
        participants = program.get("participants", [])
        ids = [p.get("object_id") for p in participants if isinstance(p, dict)]
        attached.update(item for item in ids if isinstance(item, str))
    return attached


def _visible_ids(data: dict[str, Any], minimum_pixels: int) -> set[str]:
    audit = data.get("sceneproof_mesh_visibility_audit", {})
    records = audit.get("objects", {}) if isinstance(audit, dict) else {}
    if not isinstance(records, dict) or not records:
        return set()
    return {
        object_id
        for object_id, row in records.items()
        if isinstance(row, dict)
        and row.get("status") == "measured"
        and int(row.get("rendered_visible_pixels", 0)) >= minimum_pixels
    }


def apply_visible_support_proxy(
    data: dict[str, Any], *, maximum_shift_m: float = 0.5,
    contact_tolerance_m: float = 0.05, minimum_visible_pixels: int = 16,
    minimum_xy_overlap: float = 0.05, repair_enabled: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = copy.deepcopy(data)
    objects = output.get("obj_info", {})
    if not isinstance(objects, dict):
        raise ValueError("placement obj_info is missing")
    visible = _visible_ids(output, minimum_visible_pixels)
    visibility_available = bool(visible)
    if not visibility_available:
        visible = {
            object_id for object_id in objects
            if object_id != "scene_camera"
            and not object_id.startswith(STRUCTURAL_PREFIXES)
        }
    program_attached = _program_attachment_objects(output)
    bounds = {key: _world_aabb(value) for key, value in objects.items()}
    for key, info in objects.items():
        if bounds[key] is not None or not key.startswith(("floor_", "ceiling_")):
            continue
        matrix = info.get("pose_matrix_for_blender") if isinstance(info, dict) else None
        try:
            height = float(matrix[2][3])
        except (IndexError, TypeError, ValueError):
            continue
        bounds[key] = [-1e6, 1e6, -1e6, 1e6, height, height]
    rows: dict[str, dict[str, Any]] = {}
    unresolved: list[str] = []
    repaired: list[str] = []

    for object_id in sorted(visible):
        if object_id == "scene_camera" or object_id.startswith(STRUCTURAL_PREFIXES):
            continue
        info = objects.get(object_id, {})
        child = bounds.get(object_id)
        if child is None:
            rows[object_id] = {"status": "unresolved", "reason": "missing_obb"}
            unresolved.append(object_id)
            continue
        declared = _parents(info.get("supported"))
        semantic_planes = _parents(info.get("againstWall"))
        declared_planes = [p for p in declared if p.startswith(("wall_", "ceiling_"))]
        plane_signal = bool(semantic_planes or declared_planes)
        program_signal = object_id in program_attached

        # Wall attachment is a known false-positive source.  It exempts an
        # object from gravity only when two independent semantic signals agree.
        if plane_signal or program_signal:
            if plane_signal and program_signal:
                rows[object_id] = {
                    "status": "proxy_certified",
                    "reason": "explicit_plane_attachment_double_witness",
                    "witnesses": sorted(set(semantic_planes + declared_planes)),
                }
            else:
                rows[object_id] = {
                    "status": "unresolved",
                    "reason": "semantic_attachment_ambiguous",
                    "plane_signal": plane_signal,
                    "program_signal": program_signal,
                }
                unresolved.append(object_id)
            continue

        parent_id = next((p for p in declared if p in objects), None)
        inferred = False
        if parent_id is None:
            candidates = []
            for candidate_id, supporter in bounds.items():
                if candidate_id == object_id or supporter is None:
                    continue
                gap = child[4] - supporter[5]
                overlap = _xy_overlap(child, supporter)
                if -contact_tolerance_m <= gap <= maximum_shift_m and overlap >= minimum_xy_overlap:
                    candidates.append((supporter[5], overlap, candidate_id, gap))
            if candidates:
                _, _, parent_id, _ = max(candidates)
                inferred = True
        if parent_id is None:
            rows[object_id] = {
                "status": "unresolved", "reason": "no_support_witness"
            }
            unresolved.append(object_id)
            continue
        parent = bounds.get(parent_id)
        if parent is None:
            rows[object_id] = {
                "status": "unresolved", "reason": "support_parent_missing_obb",
                "parent_id": parent_id,
            }
            unresolved.append(object_id)
            continue
        gap = child[4] - parent[5]
        if abs(gap) <= contact_tolerance_m:
            rows[object_id] = {
                "status": "proxy_certified", "reason": "obb_contact_within_tolerance",
                "parent_id": parent_id, "inferred_parent": inferred, "gap_m": gap,
            }
            continue
        shift = -gap
        if abs(shift) > maximum_shift_m:
            rows[object_id] = {
                "status": "unresolved", "reason": "contact_shift_exceeds_budget",
                "parent_id": parent_id, "gap_m": gap, "required_shift_m": shift,
            }
            unresolved.append(object_id)
            continue
        if not repair_enabled:
            rows[object_id] = {
                "status": "unresolved", "reason": "obb_contact_repair_available",
                "parent_id": parent_id, "gap_m": gap, "required_shift_m": shift,
            }
            unresolved.append(object_id)
            continue
        candidate = child.copy()
        candidate[4] += shift
        candidate[5] += shift
        blockers = []
        for other_id, other in bounds.items():
            if other_id in {object_id, parent_id} or other is None:
                continue
            before = _overlap_volume(child, other)
            after = _overlap_volume(candidate, other)
            if after > before + 1e-6:
                blockers.append(other_id)
        if blockers:
            rows[object_id] = {
                "status": "unresolved", "reason": "new_obb_overlap",
                "parent_id": parent_id, "gap_m": gap, "blockers": blockers,
            }
            unresolved.append(object_id)
            continue
        matrix = info.get("pose_matrix_for_blender")
        matrix[2][3] = float(matrix[2][3]) + shift
        bounds[object_id] = candidate
        repaired.append(object_id)
        rows[object_id] = {
            "status": "proxy_repaired", "reason": "deterministic_z_contact_projection",
            "parent_id": parent_id, "inferred_parent": inferred,
            "gap_before_m": gap, "delta_z_m": shift,
            "xy_frozen": True, "so3_frozen": True,
        }

    certificate = {
        "schema_version": "sceneproof_visible_support_proxy_v1",
        "policy": "relation_conditioned_obb_only_no_true_mesh_no_physics",
        "certificate_strength": "proxy",
        "visibility_audit_available": visibility_available,
        "repair_enabled": repair_enabled,
        "visible_objects_checked": len(rows),
        "repaired_object_ids": repaired,
        "unresolved_object_ids": unresolved,
        "objects": rows,
        "status": "proxy_certified" if not unresolved else "unresolved",
        "passed": not unresolved,
        "thresholds": {
            "maximum_shift_m": maximum_shift_m,
            "contact_tolerance_m": contact_tolerance_m,
            "minimum_visible_pixels": minimum_visible_pixels,
            "minimum_xy_overlap": minimum_xy_overlap,
        },
    }
    output["sceneproof_visible_support_proxy"] = certificate
    return output, certificate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--certificate", required=True, type=Path)
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    output, certificate = apply_visible_support_proxy(
        data, repair_enabled=args.repair
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.certificate.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    args.certificate.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(
        f"VISIBLE_SUPPORT_STATUS={certificate['status']} "
        f"CHECKED={certificate['visible_objects_checked']} "
        f"REPAIRED={len(certificate['repaired_object_ids'])} "
        f"UNRESOLVED={len(certificate['unresolved_object_ids'])}"
    )


if __name__ == "__main__":
    main()
