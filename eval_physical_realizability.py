#!/usr/bin/env python3
"""Auditable physical/semantic S4 evaluation for Imaginarium.

The evaluator compares multiple S4 variants using identical geometry recovered
from a frozen S3 result.  It reports raw SI-unit violations, per-family pass
rates, per-object local scores, scene/global macro scores, and optional runtime
statistics.  No rendered-image or GT pose information is used here; this is
complementary to ``eval_gt_metrics.py``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable

import numpy as np

HEADLINE_FAMILIES = ("collision", "support", "plane", "semantic")

from modules._s4_layoutvlm_relations import build_semantic_relation_specs


STRUCTURAL = re.compile(r"^(floor|ground|wall|ceiling|carpet|rug)_\d+$")
CAMERA = re.compile(r"scene_camera", re.IGNORECASE)


@dataclass
class Geometry:
    name: str
    info: dict[str, Any]
    matrix: np.ndarray
    local_corners: np.ndarray
    world_corners: np.ndarray
    polygon: np.ndarray
    z_min: float
    z_max: float
    volume: float


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def matrix(info: dict[str, Any]) -> np.ndarray | None:
    value = info.get("pose_matrix_for_blender")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return result if result.shape == (4, 4) and np.isfinite(result).all() else None


def transform(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def convex_hull(points: np.ndarray) -> np.ndarray:
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) <= 2:
        return np.asarray(unique, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    return float(
        0.5
        * abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
    )


def cross2(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _inside(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> bool:
    return cross2(second - first, point - first) >= -1e-10


def _line_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> np.ndarray:
    first = p2 - p1
    second = q2 - q1
    denominator = cross2(first, second)
    if abs(denominator) <= 1e-12:
        return p2
    parameter = cross2(q1 - p1, second) / denominator
    return p1 + parameter * first


def convex_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = [point.copy() for point in subject]
    if len(subject) < 3 or len(clip) < 3:
        return np.empty((0, 2), dtype=np.float64)
    for index, first in enumerate(clip):
        second = clip[(index + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            current_inside = _inside(current, first, second)
            previous_inside = _inside(previous, first, second)
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(previous, current, first, second)
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    _line_intersection(previous, current, first, second)
                )
            previous = current
    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def interval_separation_distance(
    first_min: float, first_max: float, second_min: float, second_max: float
) -> float:
    """Shortest translation along one axis that makes two intervals disjoint.

    Note this is *not* the overlap length ``min(max) - max(min)``.  For a nested
    pair the overlap length equals the inner interval's own width, and translating
    by that much leaves the pair still overlapping.  The distance that actually
    separates them is the smaller of the two escape directions.
    """
    forward = float(first_max) - float(second_min)
    backward = float(second_max) - float(first_min)
    if forward <= 0.0 or backward <= 0.0:
        return 0.0
    return min(forward, backward)


def polygon_penetration_depth_2d(first: np.ndarray, second: np.ndarray) -> float:
    """Exact minimum translation distance separating two convex polygons.

    Both polygons come from ``convex_hull`` and are convex, so the
    separating-axis theorem applies: the minimum translation vector lies along one
    of the edge normals.  Returns 0.0 when the polygons are disjoint or when
    either is degenerate.
    """
    if len(first) < 3 or len(second) < 3:
        return 0.0
    best = float("inf")
    for polygon in (first, second):
        count = len(polygon)
        for index in range(count):
            edge = polygon[(index + 1) % count] - polygon[index]
            length = float(np.hypot(edge[0], edge[1]))
            if length <= 1e-12:
                continue
            axis = np.asarray((edge[1], -edge[0]), dtype=np.float64) / length
            left, right = first @ axis, second @ axis
            separation = interval_separation_distance(
                left.min(), left.max(), right.min(), right.max()
            )
            if separation <= 0.0:
                return 0.0
            best = min(best, separation)
    return 0.0 if not np.isfinite(best) else best


def prism_penetration_depth(
    first: "Geometry",
    second: "Geometry",
) -> tuple[float, str]:
    """Exact minimum translation distance separating two vertical convex prisms.

    A prism is a convex polygon crossed with a vertical interval, so its Minkowski
    difference with another prism is the product of the polygons' difference and
    the intervals' difference.  For such a product set the distance from the origin
    to the boundary is the smaller of the two factors' distances, which makes the
    composition below exact rather than a bound.

    The value answers "how far must one object move to stop overlapping", which is
    scale-correct: a millimetre of overlap reads as a millimetre regardless of how
    large or small the two objects are.
    """
    vertical = interval_separation_distance(
        first.z_min, first.z_max, second.z_min, second.z_max
    )
    lateral = polygon_penetration_depth_2d(first.polygon, second.polygon)
    if lateral <= 0.0 or vertical <= 0.0:
        return 0.0, "disjoint"
    if vertical <= lateral:
        return float(vertical), "vertical"
    return float(lateral), "lateral"


def point_segment_distance(point, first, second) -> float:
    edge = second - first
    denominator = float(np.dot(edge, edge))
    if denominator <= 1e-12:
        return float(np.linalg.norm(point - first))
    parameter = float(np.clip(np.dot(point - first, edge) / denominator, 0, 1))
    return float(np.linalg.norm(point - (first + parameter * edge)))


def outside_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return float("inf")
    signed = []
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        signed.append(cross2(second - first, point - first))
    if min(signed) >= -1e-9:
        return 0.0
    return min(
        point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    )


def angle_degrees(first: np.ndarray, second: np.ndarray, unsigned=False) -> float:
    first = first / max(float(np.linalg.norm(first)), 1e-12)
    second = second / max(float(np.linalg.norm(second)), 1e-12)
    cosine = float(np.clip(np.dot(first, second), -1, 1))
    if unsigned:
        cosine = abs(cosine)
    return math.degrees(math.acos(np.clip(cosine, -1, 1)))


def linear_score(error: float, tolerance: float) -> float:
    if not np.isfinite(error):
        return 0.0
    return float(max(0.0, 1.0 - error / tolerance))


def summarize(values: Iterable[float]) -> dict[str, Any]:
    values = [float(value) for value in values if np.isfinite(value)]
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    ordered = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": float(np.mean(ordered)),
        "median": float(np.median(ordered)),
        "p90": float(np.quantile(ordered, 0.9)),
        "max": float(np.max(ordered)),
    }


def find_s4(root: Path, scene: str, version: str) -> Path:
    folder = root / f"{scene}_{version}_result" / "S4_layout_refinement"
    preferred = folder / f"{scene}_{version}_placement_info_s4.json"
    if preferred.is_file():
        return preferred
    matches = sorted(folder.glob("*_placement_info_s4.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one S4 JSON in {folder}, found {len(matches)}")
    return matches[0]


def find_s3(root: Path, scene: str, version: str) -> Path:
    folder = root / f"{scene}_{version}_result" / "S3_pose_inference"
    matches = [
        path
        for path in sorted(folder.glob("*_placement_info.json"))
        if not path.name.endswith("_s4.json")
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one S3 JSON in {folder}, found {len(matches)}")
    return matches[0]


def find_geometry_snapshot(root: Path, scene: str, version: str) -> Path:
    """Find the frozen, asset-scaled geometry before S4 optimization.

    ``S3_pose_inference/*_placement_info.json`` contains poses and observed
    dimensions such as ``pcd_obb_size``, but it does not contain the imported
    asset's world ``bbox``/``length``.  S4 writes that required geometry to its
    own ``*_placement_info_s3.json`` immediately before refinement.
    """
    folder = root / f"{scene}_{version}_result" / "S4_layout_refinement"
    preferred = folder / f"{scene}_{version}_placement_info_s3.json"
    if preferred.is_file():
        return preferred
    matches = sorted(folder.glob("*_placement_info_s3.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one frozen S4 geometry snapshot in {folder}, "
            f"found {len(matches)}"
        )
    return matches[0]


def validate_geometry_snapshot(data: dict[str, Any], path: Path) -> None:
    objects = [
        info
        for name, info in data.get("obj_info", {}).items()
        if (
            isinstance(info, dict)
            and not CAMERA.search(name)
            and matrix(info) is not None
        )
    ]
    usable = 0
    for info in objects:
        try:
            bbox = np.asarray(info.get("bbox"), dtype=np.float64)
            length = np.asarray(info.get("length"), dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if (
            bbox.shape == (8, 3)
            and np.isfinite(bbox).all()
            and length.shape == (3,)
            and np.isfinite(length).all()
            and np.all(length > 0)
        ):
            usable += 1
    if not objects or usable / len(objects) < 0.9:
        raise ValueError(
            f"invalid frozen geometry snapshot {path}: "
            f"usable_bbox_and_length={usable}/{len(objects)}; "
            "use S4_layout_refinement/*_placement_info_s3.json, not the "
            "S3_pose_inference placement JSON"
        )


def support_id(value: Any) -> str | None:
    if isinstance(value, (list, tuple)):
        value = next((item for item in value if item), None)
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def fallback_local_corners(info: dict[str, Any], pose: np.ndarray) -> np.ndarray:
    length = np.asarray(info.get("length", [0, 0, 0]), dtype=np.float64)
    scale = np.linalg.norm(pose[:3, :3], axis=0)
    local_length = length / np.maximum(scale, 1e-8)
    minimum = -0.5 * local_length
    maximum = 0.5 * local_length
    return np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )


def local_geometry(info: dict[str, Any]) -> np.ndarray | None:
    pose = matrix(info)
    if pose is None:
        return None
    try:
        bbox = np.asarray(info.get("bbox"), dtype=np.float64)
    except (TypeError, ValueError):
        bbox = np.empty((0, 3))
    if bbox.shape == (8, 3) and np.isfinite(bbox).all():
        homogeneous = np.concatenate((bbox, np.ones((8, 1))), axis=1)
        return (homogeneous @ np.linalg.inv(pose).T)[:, :3]
    return fallback_local_corners(info, pose)


def build_geometries(
    source_info: dict[str, Any],
    target_info: dict[str, Any],
    *,
    repair_degenerate_footprints: bool = False,
) -> dict[str, Geometry]:
    """Recover world geometry for every object from the frozen source geometry.

    Local corners come from``source_info`` so that all compared variants share
    identical shape, and only the pose varies.  When the frozen snapshot carries
    no usable ``bbox`` or ``length`` for an object, the derived footprint can
    collapse to a line or a point.  That is not a scene defect but a measurement
    defect: a collapsed parent footprint makes the containment error infinite and
    the footprint overlap zero for every child, and a collapsed floor footprint
    makes every boundary error infinite.

    ``repair_degenerate_footprints`` retries such an object with the geometry
    carried by the layout itself.  It is off by default so frozen baseline
    numbers never move silently; the repaired count is reported so the effect is
    always visible.
    """
    result = {}
    for name, target in target_info.items():
        if CAMERA.search(name) or not isinstance(target, dict):
            continue
        pose = matrix(target)
        source = source_info.get(name, target)
        corners = local_geometry(source)
        if pose is None or corners is None:
            continue
        world = transform(corners, pose)
        polygon = convex_hull(world[:, :2])
        if repair_degenerate_footprints and len(polygon) < 3 and source is not target:
            alternative = local_geometry(target)
            if alternative is not None:
                alternative_world = transform(alternative, pose)
                alternative_polygon = convex_hull(alternative_world[:, :2])
                if len(alternative_polygon) >= 3:
                    corners = alternative
                    world = alternative_world
                    polygon = alternative_polygon
        z_min = float(world[:, 2].min())
        z_max = float(world[:, 2].max())
        volume = polygon_area(polygon) * max(0.0, z_max - z_min)
        result[name] = Geometry(
            name=name,
            info=target,
            matrix=pose,
            local_corners=corners,
            world_corners=world,
            polygon=polygon,
            z_min=z_min,
            z_max=z_max,
            volume=volume,
        )
    return result


def degenerate_footprint_object_ids(geometries: dict[str, Geometry]) -> list[str]:
    """Objects whose XY footprint collapsed to fewer than three hull vertices."""
    return sorted(
        name for name, geometry in geometries.items() if len(geometry.polygon) < 3
    )


WALL = re.compile(r"^wall_\d+$")


def resolve_floor_id(
    target: dict[str, Any],
    geometries: dict[str, Geometry],
) -> str | None:
    """Identify the object the boundary family measures against."""
    reference = target.get("reference_obj")
    if reference and str(reference) in geometries:
        return str(reference)
    return next(
        (
            name
            for name in geometries
            if name.startswith(("floor_", "ground_"))
        ),
        None,
    )


# A floor reconstruction from wall geometry was implemented and then withdrawn.
# It assumed the walls were room-sized panels standing on the floor, so that the
# hull of their footprints bounded the room and their lowest extent marked the
# floor's top face.  Measured on Smoke5 the derived top face landed 2.87 m to
# 3.15 m *below* the floor origin, against a predicted +0.02 m, because the walls
# are 10 m by 10 m construction panels far larger than the room and extending
# well below it.  The derived boundary score also ranged from 0.298 to 1.000
# across scenes, which is the signature of an arbitrary reference rather than a
# measurement.  The hypothesis is falsified and the code is gone rather than
# tuned.  The floor's true extent is recovered by measuring it, see
# ``--structural-geometry-sidecar``.


def load_structural_geometry_sidecar(path: Path) -> dict[str, dict[str, Any]]:
    """Load geometry measured from the pipeline's own scene.

    The ground slab is created procedurally and excluded from geometry
    serialization, so no artefact records its extent.  The sidecar carries the
    world bounding box and dimensions read back from the constructed Blender
    scene, which is a measurement of the pipeline's actual geometry rather than
    an assumed constant.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    entries = document.get("obj_info", document)
    result: dict[str, dict[str, Any]] = {}
    for name, info in entries.items():
        if not isinstance(info, dict):
            continue
        try:
            bbox = np.asarray(info.get("bbox"), dtype=np.float64)
            length = np.asarray(info.get("length"), dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if (
            bbox.shape == (8, 3)
            and np.isfinite(bbox).all()
            and length.shape == (3,)
            and np.isfinite(length).all()
        ):
            result[name] = {"bbox": bbox.tolist(), "length": length.tolist()}
    return result


def apply_structural_geometry_sidecar(
    source_info: dict[str, Any],
    target_info: dict[str, Any],
    sidecar: dict[str, dict[str, Any]],
) -> list[str]:
    """Fill in missing geometry from the sidecar, never overriding what exists."""
    backfilled: list[str] = []
    for name, geometry in sidecar.items():
        if name not in target_info or CAMERA.search(name):
            continue
        for info in (source_info.get(name), target_info.get(name)):
            if not isinstance(info, dict):
                continue
            corners = local_geometry(info)
            pose = matrix(info)
            if pose is None:
                continue
            usable = False
            if corners is not None:
                world = transform(corners, pose)
                usable = len(convex_hull(world[:, :2])) >= 3
            if usable:
                continue
            info["bbox"] = geometry["bbox"]
            info["length"] = geometry["length"]
            if name not in backfilled:
                backfilled.append(name)
    return sorted(backfilled)


def collision_pair_detail(
    first: "Geometry",
    second: "Geometry",
    intersection: np.ndarray,
    z_overlap: float,
    volume: float,
    fraction: float,
    penetration_depth: float,
    penetration_axis: str,
) -> dict[str, Any]:
    """Record one reported collision pair with the quantities that explain it.

    Only measurements are stored, never a verdict.  Whether a pair is a real
    interpenetration or an artefact of treating each bounding box as solid is
    decided downstream, so the classification rule stays auditable and can be
    changed without re-running the evaluator.
    """
    smaller, larger = (
        (first, second) if first.volume <= second.volume else (second, first)
    )
    smaller_footprint = polygon_area(smaller.polygon)
    intersection_area = polygon_area(intersection)
    return {
        "first_id": first.name,
        "second_id": second.name,
        "intersection_volume_m3": float(volume),
        "overlap_fraction": float(fraction),
        "penetration_depth_m": float(penetration_depth),
        "penetration_depth_axis": penetration_axis,
        "z_overlap_m": float(z_overlap),
        "intersection_area_m2": float(intersection_area),
        "first_volume_m3": float(first.volume),
        "second_volume_m3": float(second.volume),
        "smaller_object_id": smaller.name,
        "larger_object_id": larger.name,
        "smaller_volume_m3": float(smaller.volume),
        "smaller_z_min": float(smaller.z_min),
        "smaller_z_max": float(smaller.z_max),
        "larger_z_min": float(larger.z_min),
        "larger_z_max": float(larger.z_max),
        "smaller_footprint_area_m2": float(smaller_footprint),
        "larger_footprint_area_m2": float(polygon_area(larger.polygon)),
        "intersection_over_smaller_footprint": (
            float(intersection_area / smaller_footprint)
            if smaller_footprint > 1e-9
            else None
        ),
        # A chair tucked under an oval table sits inside the table's rectangular
        # footprint hull while being outside the table itself.  Recording this
        # separately keeps that case distinguishable from a real side-on impact.
        "smaller_footprint_inside_larger": (
            bool(intersection_area >= 0.99 * smaller_footprint)
            if smaller_footprint > 1e-9
            else None
        ),
        "shares_vertical_base_with_larger": bool(
            abs(smaller.z_min - larger.z_min) <= 0.01
        ),
        "smaller_rises_above_larger": bool(smaller.z_max > larger.z_max + 0.01),
        "smaller_supported": support_id(smaller.info.get("supported")),
        "larger_supported": support_id(larger.info.get("supported")),
        "smaller_spatial_rel": str(smaller.info.get("SpatialRel", "")).lower() or None,
        "larger_spatial_rel": str(larger.info.get("SpatialRel", "")).lower() or None,
    }


def evaluate_scene(
    source: dict[str, Any],
    target: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_info = source.get("obj_info", {})
    target_info = target.get("obj_info", {})
    sidecar = getattr(args, "_structural_geometry_sidecar", None) or {}
    backfilled_ids = (
        apply_structural_geometry_sidecar(source_info, target_info, sidecar)
        if sidecar
        else []
    )
    repair = bool(getattr(args, "repair_degenerate_footprints", False))
    geometries = build_geometries(
        source_info, target_info, repair_degenerate_footprints=repair
    )
    # Footprint provenance is reported unconditionally.  A collapsed footprint is
    # a measurement defect, and silently scoring zero for a quantity that cannot
    # be measured would conflate "known bad" with "unmeasurable".
    degenerate_now = degenerate_footprint_object_ids(geometries)
    if repair:
        unrepaired_geometries = build_geometries(
            source_info, target_info, repair_degenerate_footprints=False
        )
        degenerate_before = degenerate_footprint_object_ids(unrepaired_geometries)
    else:
        degenerate_before = degenerate_now
    repaired_ids = sorted(set(degenerate_before) - set(degenerate_now))

    floor_id = resolve_floor_id(target, geometries)
    abstain = bool(getattr(args, "abstain_on_unmeasurable_footprints", False))
    placeholder_lateral = bool(
        getattr(args, "placeholder_structural_lateral_extent", False)
    )
    abstentions: dict[str, list[str]] = {"support": [], "boundary": []}
    partial_summand_terms: list[str] = []
    object_ids = [name for name in geometries if not STRUCTURAL.match(name)]
    direct_support = {
        frozenset((name, support_id(geometry.info.get("supported"))))
        for name, geometry in geometries.items()
        if support_id(geometry.info.get("supported"))
    }
    relation_programs = target.get("sceneproof_relation_programs", {}).get(
        "programs", []
    )
    program_support: set[frozenset[str]] = set()
    program_attachment: set[frozenset[str]] = set()
    program_collision: set[frozenset[str]] = set()
    for program in relation_programs:
        if not isinstance(program, dict):
            continue
        participants = [
            row.get("object_id")
            for row in program.get("participants", [])
            if isinstance(row, dict) and isinstance(row.get("object_id"), str)
        ]
        if len(participants) < 2:
            continue
        pair = frozenset(participants[:2])
        if len(pair) != 2:
            continue
        kind = str(program.get("kind", "")).upper()
        if kind in {"SUPPORT", "STACK"}:
            program_support.add(pair)
        elif kind in {"PLANE_ATTACH", "CEILING_ATTACH", "HANG"}:
            program_attachment.add(pair)
        elif kind == "COLLISION_EXCLUSION":
            program_collision.add(pair)
    requested_collision_policy = str(
        getattr(args, "collision_policy", "auto")
    ).lower()
    relation_conditioned_collision = bool(relation_programs) and (
        requested_collision_policy != "legacy"
    )
    if relation_conditioned_collision:
        allowed_relation_contact = (
            direct_support | program_support | program_attachment
        )
    else:
        # Cross-version comparisons must not let the mere availability of a
        # Relation Program change which overlaps are scored.  Legacy mode uses
        # only the support edges serialized by every compared version.
        allowed_relation_contact = direct_support
    local: dict[str, dict[str, Any]] = {
        name: {
            "object_id": name,
            "scores": [],
            "collision_overlap_fraction": 0.0,
            "support_contact_gap_m": None,
            "support_containment_error_m": None,
            "support_footprint_overlap_ratio": None,
            "inside_containment_error_m": None,
            "plane_contact_gap_m": None,
            "plane_orientation_error_deg": None,
            "boundary_error_m": None,
            "semantic_error": None,
            # Per-object contribution to each family mean.``None`` means the
            # object contributes no term to that family, so the family mean is
            # exactly the average of the non-``None`` values below.  These are
            # written verbatim by the evaluator so downstream attribution never
            # has to re-derive a score from raw errors and tolerances.
            "collision_term": None,
            "support_term": None,
            "plane_term": None,
            "boundary_term": None,
            "semantic_term": None,
            # Scale-correct companions to ``collision_overlap_fraction``, reported
            # unconditionally but not scored.  The fraction divides by the smaller
            # object's bounding volume, so it answers "what share of the smaller
            # object is overlapped" rather than "how badly do these two overlap".
            # These two answer the latter in absolute units and make the
            # normalization auditable.
            "collision_max_penetration_depth_m": 0.0,
            "collision_max_intersection_volume_m3": 0.0,
        }
        for name in object_ids
    }
    family_scores: dict[str, list[float]] = {
        "collision": [],
        "support": [],
        "plane": [],
        "boundary": [],
        "semantic": [],
    }
    family_passes: dict[str, list[bool]] = {
        family: [] for family in family_scores
    }
    raw: dict[str, list[float]] = {
        "collision_fraction": [],
        "collision_volume_m3": [],
        "collision_penetration_depth_m": [],
        "support_contact_gap_m": [],
        "support_containment_error_m": [],
        "support_footprint_overlap_ratio": [],
        "inside_containment_error_m": [],
        "plane_contact_gap_m": [],
        "plane_orientation_error_deg": [],
        "boundary_error_m": [],
        "semantic_error": [],
    }

    # Unintended pairwise overlap. Direct support and structural pairs are not
    # collisions; their validity is measured by contact/plane constraints.
    #
    # The geometry used here is each object's oriented bounding box, treated as a
    # solid prism: the convex hull of its footprint extruded over [z_min, z_max].
    # That prism contains the object's true mesh, which has two consequences that
    # hold by construction rather than by observation.  First, a pair whose prisms
    # do not intersect cannot have intersecting meshes, so the pairs recorded below
    # are a complete superset of the truly interpenetrating pairs.  Second, the
    # measured intersection volume is an upper bound on the true one, so every
    # ``collision_term`` is a lower bound on the score a true-mesh measurement
    # would give.  The family score can therefore only rise under finer geometry.
    #
    # The corollary is that a legal arrangement can score zero here: a chair
    # tucked under a table has its seat inside the table's prism, because the
    # cavity beneath the table top is solid in this representation.  Pair details
    # are exported so that artefact can be told apart from real interpenetration.
    collision_pairs = 0
    eligible_pairs = 0
    collect_pairs = bool(getattr(args, "collision_pairs_csv", None))
    collision_pair_details: list[dict[str, Any]] = []
    for first_index, first_id in enumerate(object_ids):
        first = geometries[first_id]
        for second_id in object_ids[first_index + 1 :]:
            pair = frozenset((first_id, second_id))
            if pair in allowed_relation_contact:
                continue
            if relation_conditioned_collision and pair not in program_collision:
                continue
            eligible_pairs += 1
            second = geometries[second_id]
            z_overlap = min(first.z_max, second.z_max) - max(first.z_min, second.z_min)
            if z_overlap <= 1e-6:
                continue
            intersection = convex_intersection(first.polygon, second.polygon)
            volume = polygon_area(intersection) * z_overlap
            if volume <= args.collision_volume_tolerance:
                continue
            fraction = volume / max(min(first.volume, second.volume), 1e-9)
            depth, depth_axis = prism_penetration_depth(first, second)
            collision_pairs += 1
            raw["collision_fraction"].append(fraction)
            raw["collision_volume_m3"].append(volume)
            raw["collision_penetration_depth_m"].append(depth)
            for object_id in (first_id, second_id):
                local[object_id]["collision_overlap_fraction"] = max(
                    local[object_id]["collision_overlap_fraction"], fraction
                )
                local[object_id]["collision_max_penetration_depth_m"] = max(
                    local[object_id]["collision_max_penetration_depth_m"], depth
                )
                local[object_id]["collision_max_intersection_volume_m3"] = max(
                    local[object_id]["collision_max_intersection_volume_m3"], volume
                )
            if collect_pairs:
                collision_pair_details.append(
                    collision_pair_detail(
                        first,
                        second,
                        intersection,
                        z_overlap,
                        volume,
                        fraction,
                        depth,
                        depth_axis,
                    )
                )
    for object_id in object_ids:
        score = linear_score(
            local[object_id]["collision_overlap_fraction"],
            args.collision_fraction_tolerance,
        )
        local[object_id]["scores"].append(score)
        local[object_id]["collision_term"] = score
        family_scores["collision"].append(score)
        family_passes["collision"].append(
            local[object_id]["collision_overlap_fraction"]
            <= args.collision_fraction_tolerance
        )

    support_missing = 0
    for child_id in object_ids:
        child = geometries[child_id]
        parent_id = support_id(child.info.get("supported"))
        if not parent_id:
            continue
        parent = geometries.get(parent_id)
        if parent is None:
            support_missing += 1
            local[child_id]["scores"].append(0.0)
            local[child_id]["support_term"] = 0.0
            family_scores["support"].append(0.0)
            family_passes["support"].append(False)
            continue
        parent_kind = parent_id.split("_", 1)[0].lower()
        spatial = str(child.info.get("SpatialRel", "")).lower()
        if parent_kind in {"wall", "ceiling"}:
            parent_inverse = np.linalg.inv(parent.matrix)
            child_h = np.concatenate(
                (child.world_corners, np.ones((8, 1))), axis=1
            )
            child_local = (child_h @ parent_inverse.T)[:, :3]
            parent_min = parent.local_corners.min(axis=0)
            parent_max = parent.local_corners.max(axis=0)
            spans = parent_max - parent_min
            axis = int(np.argmin(spans))
            center = float(child_local[:, axis].mean())
            midpoint = 0.5 * (parent_min[axis] + parent_max[axis])
            if center >= midpoint:
                face = parent_max[axis]
                child_near = child_local[:, axis].min()
            else:
                face = parent_min[axis]
                child_near = child_local[:, axis].max()
            gap = abs(float(child_near - face))
            local[child_id]["plane_contact_gap_m"] = gap
            raw["plane_contact_gap_m"].append(gap)
            scores = [linear_score(gap, args.plane_tolerance)]
            if parent_kind == "wall":
                parent_normal = parent.matrix[:3, axis]
                child_spans = child.local_corners.max(axis=0) - child.local_corners.min(axis=0)
                child_axis = int(np.argmin(child_spans))
                child_normal = child.matrix[:3, child_axis]
                orientation = angle_degrees(child_normal, parent_normal, unsigned=True)
                local[child_id]["plane_orientation_error_deg"] = orientation
                raw["plane_orientation_error_deg"].append(orientation)
                scores.append(
                    linear_score(orientation, args.plane_orientation_tolerance)
                )
            score = float(np.mean(scores))
            local[child_id]["scores"].append(score)
            local[child_id]["plane_term"] = score
            family_scores["plane"].append(score)
            family_passes["plane"].append(
                gap <= args.plane_tolerance
                and (
                    parent_kind != "wall"
                    or orientation <= args.plane_orientation_tolerance
                )
            )
            continue

        if spatial == "inside":
            parent_inverse = np.linalg.inv(parent.matrix)
            child_h = np.concatenate(
                (child.world_corners, np.ones((8, 1))), axis=1
            )
            child_local = (child_h @ parent_inverse.T)[:, :3]
            minimum = parent.local_corners.min(axis=0)
            maximum = parent.local_corners.max(axis=0)
            violation = np.maximum(minimum - child_local, 0) + np.maximum(
                child_local - maximum, 0
            )
            error = float(np.linalg.norm(violation, axis=1).max())
            local[child_id]["inside_containment_error_m"] = error
            raw["inside_containment_error_m"].append(error)
            score = linear_score(error, args.containment_tolerance)
            local[child_id]["scores"].append(score)
            local[child_id]["support_term"] = score
            family_scores["support"].append(score)
            family_passes["support"].append(
                error <= args.containment_tolerance
            )
            continue

        if abstain and len(parent.polygon) < 3:
            # The parent's extent is absent from the artefacts, so its top face,
            # its containment region and its footprint overlap are all
            # unmeasurable.  Scoring zero here would report "known bad" for a
            # quantity that is merely unknown, so the term is omitted and the
            # abstention is counted instead.
            local[child_id]["support_unmeasurable_parent_id"] = parent_id
            abstentions["support"].append(child_id)
            continue

        gap = abs(child.z_min - parent.z_max)
        # The floor and the walls are procedural placeholders: measurement shows
        # the ground slab is exactly 10 m by 10 m by 0.04 m and the wall panels
        # are exactly 10 m tall, regardless of the room.  Their *top face* is real
        # -- it is the surface the pipeline placed objects on -- but their lateral
        # extent carries no information about the room, so containment and
        # footprint overlap measured against them are trivially satisfied rather
        # than verified.  Under this flag those two summands are declared
        # unmeasurable and the term is the mean of what remains.
        lateral_unmeasurable = bool(
            placeholder_lateral and STRUCTURAL.match(parent_id)
        )
        summands = [linear_score(gap, args.contact_tolerance)]
        local[child_id]["support_contact_gap_m"] = gap
        raw["support_contact_gap_m"].append(gap)
        if lateral_unmeasurable:
            local[child_id]["support_lateral_extent_unmeasurable_parent_id"] = (
                parent_id
            )
            partial_summand_terms.append(child_id)
            containment = None
            support_overlap = None
        else:
            containment = max(
                outside_distance(point, parent.polygon)
                for point in child.polygon
            )
            child_area = polygon_area(child.polygon)
            support_overlap = (
                polygon_area(convex_intersection(child.polygon, parent.polygon))
                / max(child_area, 1e-9)
            )
            support_overlap = float(np.clip(support_overlap, 0.0, 1.0))
            local[child_id]["support_containment_error_m"] = containment
            local[child_id]["support_footprint_overlap_ratio"] = support_overlap
            raw["support_containment_error_m"].append(containment)
            raw["support_footprint_overlap_ratio"].append(support_overlap)
            summands.append(
                linear_score(containment, args.containment_tolerance)
            )
            summands.append(
                min(1.0, support_overlap / args.support_overlap_tolerance)
            )
        local[child_id]["support_summand_count"] = len(summands)
        score = float(np.mean(summands))
        local[child_id]["scores"].append(score)
        local[child_id]["support_term"] = score
        family_scores["support"].append(score)
        family_passes["support"].append(
            gap <= args.contact_tolerance
            and (
                lateral_unmeasurable
                or (
                    containment <= args.containment_tolerance
                    and support_overlap >= args.support_overlap_tolerance
                )
            )
        )

    floor = geometries.get(str(floor_id)) if floor_id else None
    if floor is not None and abstain and len(floor.polygon) < 3:
        # Every boundary term is measured against this one polygon, so a
        # collapsed floor makes the whole family unmeasurable.  Reporting a
        # constant zero for it would be a fabricated score.
        abstentions["boundary"] = list(object_ids)
        floor = None
    if (
        floor is not None
        and placeholder_lateral
        and floor_id is not None
        and STRUCTURAL.match(str(floor_id))
    ):
        # The boundary family asks whether an object lies inside the room, but the
        # only candidate reference is a fixed 10 m construction slab.  The room's
        # spatial extent is not represented in any artefact, so this question is
        # not answerable rather than answered well or badly.
        abstentions["boundary"] = list(object_ids)
        floor = None
    if floor is not None:
        for object_id in object_ids:
            geometry = geometries[object_id]
            error = max(
                outside_distance(point, floor.polygon)
                for point in geometry.polygon
            )
            local[object_id]["boundary_error_m"] = error
            raw["boundary_error_m"].append(error)
            score = linear_score(error, args.boundary_tolerance)
            local[object_id]["scores"].append(score)
            local[object_id]["boundary_term"] = score
            family_scores["boundary"].append(score)
            family_passes["boundary"].append(
                error <= args.boundary_tolerance
            )

    ordered = list(geometries)
    warm_matrices = [source_info.get(name, {}).get("pose_matrix_for_blender", geometries[name].matrix.tolist()) for name in ordered]
    footprints = [
        [
            float(np.ptp(geometries[name].polygon[:, 0])),
            float(np.ptp(geometries[name].polygon[:, 1])),
        ]
        for name in ordered
    ]
    specs = build_semantic_relation_specs(source_info, ordered, warm_matrices, footprints)
    semantic_by_object: dict[str, list[float]] = {name: [] for name in object_ids}
    for (source_index, target_index), offset in zip(
        specs["align_pairs"], specs["align_offsets"]
    ):
        source_id, target_id = ordered[source_index], ordered[target_index]
        source_front = -geometries[source_id].matrix[:2, 1]
        target_front = -geometries[target_id].matrix[:2, 1]
        cosine, sine = math.cos(-offset), math.sin(-offset)
        rotated = np.asarray(
            [cosine * source_front[0] - sine * source_front[1],
             sine * source_front[0] + cosine * source_front[1]]
        )
        error = angle_degrees(rotated, target_front)
        semantic_by_object.setdefault(source_id, []).append(
            error / args.semantic_angle_tolerance
        )
    for (source_index, target_index), offset in zip(
        specs["point_pairs"], specs["point_offsets"]
    ):
        source_id, target_id = ordered[source_index], ordered[target_index]
        source = geometries[source_id]
        target = geometries[target_id]
        front = -source.matrix[:2, 1]
        cosine, sine = math.cos(-offset), math.sin(-offset)
        front = np.asarray(
            [cosine * front[0] - sine * front[1],
             sine * front[0] + cosine * front[1]]
        )
        direction = target.polygon.mean(axis=0) - source.polygon.mean(axis=0)
        error = angle_degrees(front, direction)
        semantic_by_object.setdefault(source_id, []).append(
            error / args.semantic_angle_tolerance
        )
    for pair, minimum, maximum in zip(
        specs["distance_pairs"],
        specs["distance_minimum"],
        specs["distance_maximum"],
    ):
        source_id, target_id = ordered[pair[0]], ordered[pair[1]]
        distance = float(
            np.linalg.norm(
                geometries[source_id].matrix[:2, 3]
                - geometries[target_id].matrix[:2, 3]
            )
        )
        error = max(minimum - distance, distance - maximum, 0.0)
        semantic_by_object.setdefault(source_id, []).append(
            error / args.distance_tolerance
        )
    for object_id, normalized in semantic_by_object.items():
        if object_id not in local or not normalized:
            continue
        error = float(max(normalized))
        local[object_id]["semantic_error"] = error
        raw["semantic_error"].append(error)
        score = max(0.0, 1.0 - error)
        local[object_id]["scores"].append(score)
        local[object_id]["semantic_term"] = score
        family_scores["semantic"].append(score)
        family_passes["semantic"].append(error <= 1.0)

    local_rows = []
    for object_id, values in local.items():
        values["local_realizability"] = (
            float(np.mean(values["scores"])) if values["scores"] else None
        )
        values.pop("scores")
        local_rows.append(values)

    families = {}
    available_scores = []
    for family, scores in family_scores.items():
        if not scores:
            families[family] = {"n": 0, "score": None, "pass_rate": None}
            continue
        score = float(np.mean(scores))
        families[family] = {
            "n": len(scores),
            "score": score,
            "pass_rate": float(np.mean(family_passes[family])),
        }
        available_scores.append(score)
    local_values = [
        row["local_realizability"]
        for row in local_rows
        if row["local_realizability"] is not None
    ]
    scene = {
        "object_count": len(object_ids),
        "eligible_collision_pairs": eligible_pairs,
        "unintended_collision_pairs": collision_pairs,
        "collision_relation_policy": (
            "relation_program_conditioned_candidates_and_contact_exemptions"
            if relation_conditioned_collision
            else "legacy_all_pairs_with_direct_support_exemptions"
        ),
        "relation_program_collision_pairs": len(program_collision),
        "relation_program_allowed_contact_pairs": len(allowed_relation_contact),
        # The prism representation contains each true mesh, so this pair list is a
        # complete superset of the truly interpenetrating pairs and the family
        # score is a lower bound on what finer geometry would report.  These two
        # keys are statements about the measurement model, not scores, and are
        # reported unconditionally.  The per-pair details are carried only when
        # requested and are removed from the metrics document once written to CSV,
        # so the metrics file does not grow with the pair count.
        "collision_geometry_model": "oriented_bounding_box_solid_prism",
        "collision_score_is_lower_bound_under_finer_geometry": True,
        **(
            {"collision_pair_details": collision_pair_details}
            if collect_pairs
            else {}
        ),
        "missing_support_parents": support_missing,
        # Footprint provenance.  A collapsed footprint forces the containment
        # error to infinity and the footprint overlap to zero for every child of
        # that object, and a collapsed floor forces every boundary error to
        # infinity, so these counts bound how much of the reported deficit is a
        # measurement artefact rather than a scene defect.
        "degenerate_footprint_object_ids": degenerate_now,
        "degenerate_footprint_count": len(degenerate_now),
        "footprint_repair_enabled": repair,
        "repaired_footprint_object_ids": repaired_ids,
        "repaired_footprint_count": len(repaired_ids),
        "floor_reconstruction": {
            "attempted": False,
            "withdrawn": "wall_derived_floor_plane_falsified_on_smoke5",
        },
        "structural_geometry_backfilled_object_ids": backfilled_ids,
        "structural_geometry_backfilled_count": len(backfilled_ids),
        "abstention_enabled": abstain,
        "abstained_object_ids": {
            family: sorted(ids) for family, ids in abstentions.items() if ids
        },
        "abstained_counts": {
            family: len(ids) for family, ids in abstentions.items()
        },
        "placeholder_structural_lateral_extent": placeholder_lateral,
        "partial_summand_support_terms": sorted(partial_summand_terms),
        "partial_summand_support_term_count": len(partial_summand_terms),
        # Abstention removes terms that are unmeasurable, and those terms are not
        # missing at random: they are exactly the children of an object with no
        # recorded extent, which are systematically the worst scoring ones.  The
        # remaining mean therefore answers a different question than the as-is
        # mean, so the two must never be compared or quoted as an improvement.
        # Declaring a summand unmeasurable changes the estimand in the same way,
        # even though it keeps the denominator intact.
        "estimand_changed_by_abstention": bool(any(abstentions.values())),
        "estimand_changed_by_partial_summands": bool(partial_summand_terms),
        "scores_comparable_to_non_abstained_runs": not bool(
            any(abstentions.values()) or partial_summand_terms
        ),
        "families": families,
        "macro_realizability": float(np.mean(available_scores)) if available_scores else None,
        "critical_realizability": float(min(available_scores)) if available_scores else None,
        "headline_families": list(HEADLINE_FAMILIES),
        "headline_macro_realizability": float(
            np.mean(
                [
                    families[name]["score"]
                    for name in HEADLINE_FAMILIES
                    if families.get(name, {}).get("score") is not None
                ]
            )
        )
        if any(families.get(name, {}).get("score") is not None for name in HEADLINE_FAMILIES)
        else None,
        "headline_critical_realizability": float(
            min(
                families[name]["score"]
                for name in HEADLINE_FAMILIES
                if families.get(name, {}).get("score") is not None
            )
        )
        if any(families.get(name, {}).get("score") is not None for name in HEADLINE_FAMILIES)
        else None,
        "mean_local_realizability": (
            float(np.mean(local_values)) if local_values else None
        ),
        "p10_local_realizability": (
            float(np.quantile(local_values, 0.1)) if local_values else None
        ),
        "min_local_realizability": (
            float(np.min(local_values)) if local_values else None
        ),
        "raw": {key: summarize(values) for key, values in raw.items()},
        "semantic_skipped": specs["skipped"],
    }
    return scene, local_rows


def parse_runtime_jsonl(path: Path) -> dict[str, float]:
    result = {}
    files = [path] if path.is_file() else sorted(path.rglob("runtime_gpu*.jsonl"))
    for file_path in files:
        for line in file_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "ok":
                result[str(record["scene"])] = float(record["elapsed_seconds"])
    return result


def restrict_runtime_to_scenes(
    runtime: dict[str, float], scenes: Iterable[str]
) -> dict[str, float]:
    """Keep runtime accounting paired to the evaluated scene manifest."""
    allowed = set(scenes)
    return {
        scene: seconds
        for scene, seconds in runtime.items()
        if scene in allowed
    }


def aggregate_scenes(scenes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    family_names = sorted(
        {family for scene in scenes.values() for family in scene["families"]}
    )
    families = {}
    for family in family_names:
        entries = [
            scene["families"][family]
            for scene in scenes.values()
            if scene["families"][family]["n"]
        ]
        total = sum(entry["n"] for entry in entries)
        families[family] = {
            "n": total,
            "score": (
                sum(entry["score"] * entry["n"] for entry in entries) / total
                if total else None
            ),
            "pass_rate": (
                sum(entry["pass_rate"] * entry["n"] for entry in entries) / total
                if total else None
            ),
        }
    macro = [scene["macro_realizability"] for scene in scenes.values()]
    critical = [scene["critical_realizability"] for scene in scenes.values()]
    headline_macro = [
        scene.get("headline_macro_realizability")
        for scene in scenes.values()
        if scene.get("headline_macro_realizability") is not None
    ]
    headline_critical = [
        scene.get("headline_critical_realizability")
        for scene in scenes.values()
        if scene.get("headline_critical_realizability") is not None
    ]
    local = [scene["mean_local_realizability"] for scene in scenes.values()]
    local_p10 = [scene["p10_local_realizability"] for scene in scenes.values()]
    local_min = [scene["min_local_realizability"] for scene in scenes.values()]
    return {
        "scene_count": len(scenes),
        "object_count": sum(scene["object_count"] for scene in scenes.values()),
        "unintended_collision_pairs": sum(
            scene["unintended_collision_pairs"] for scene in scenes.values()
        ),
        "missing_support_parents": sum(
            scene["missing_support_parents"] for scene in scenes.values()
        ),
        "macro_realizability": float(np.mean(macro)) if macro else None,
        "critical_realizability": float(np.mean(critical)) if critical else None,
        "headline_families": list(HEADLINE_FAMILIES),
        "headline_macro_realizability": (
            float(np.mean(headline_macro)) if headline_macro else None
        ),
        "headline_critical_realizability": (
            float(np.mean(headline_critical)) if headline_critical else None
        ),
        "mean_local_realizability": float(np.mean(local)) if local else None,
        "mean_scene_p10_local_realizability": (
            float(np.mean(local_p10)) if local_p10 else None
        ),
        "mean_scene_min_local_realizability": (
            float(np.mean(local_min)) if local_min else None
        ),
        "families": families,
    }


def parse_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected VERSION=PATH")
    return tuple(value.split("=", 1))  # type: ignore[return-value]


def runtime_summary(
    runtime: dict[str, float],
    *,
    mode: str = "stage",
    components: list[str] | None = None,
    stage_only: dict[str, float] | None = None,
) -> dict[str, Any]:
    values = list(runtime.values())
    result = {
        "mode": mode,
        "components": components or [],
        "scene_count": len(values),
        "mean_seconds": statistics.mean(values) if values else None,
        "median_seconds": statistics.median(values) if values else None,
        "p90_seconds": float(np.quantile(values, 0.9)) if values else None,
        "total_gpu_seconds": sum(values) if values else None,
        "per_scene": runtime,
    }
    if stage_only is not None:
        result["stage_only"] = runtime_summary(stage_only)
    return result


def write_report(path: Path, report: dict[str, Any], baseline: str | None) -> None:
    def headline(aggregate: dict[str, Any], *, critical: bool) -> float | None:
        key = (
            "headline_critical_realizability"
            if critical
            else "headline_macro_realizability"
        )
        if aggregate.get(key) is not None:
            return float(aggregate[key])
        values = [
            aggregate.get("families", {}).get(name, {}).get("score")
            for name in HEADLINE_FAMILIES
        ]
        values = [float(value) for value in values if value is not None]
        if not values:
            return None
        return min(values) if critical else float(np.mean(values))

    def formatted(value: float | None, width: int, precision: int = 4) -> str:
        if value is None:
            return f"{'n/a':>{width}}"
        return f"{float(value):>{width}.{precision}f}"

    lines = [
        "S4 GEOMETRIC/SEMANTIC REALIZABILITY + RUNTIME",
        "=" * 78,
        "Scores are [0,1]. Headline macro/critical use collision, support,",
        "plane, and semantic only. Boundary remains diagnostic and is reported N/A",
        "until it has non-degenerate scene coverage.",
        "Geometry is frozen from the common S3 source; this is an auditable proxy",
        "metric, complementary to GT pose metrics and optional PhysX rollouts.",
        "",
        f"{'version':34} {'macro':>8} {'critical':>9} {'local':>8} {'local10':>8} {'coll':>8} {'support':>8} {'plane':>8} {'bound':>8} {'semantic':>9} {'mean_s':>9} {'speedup':>8}",
        "-" * 142,
    ]
    baseline_data = report["versions"].get(baseline) if baseline else None
    for version, data in report["versions"].items():
        aggregate = data["aggregate"]
        families = aggregate["families"]
        runtime = data.get("runtime", {})
        def family(name):
            value = families.get(name, {}).get("score")
            return "n/a" if value is None else f"{value:.4f}"
        seconds = runtime.get("mean_seconds")
        baseline_seconds = (
            baseline_data.get("runtime", {}).get("mean_seconds")
            if baseline_data else None
        )
        speedup = (
            baseline_seconds / seconds
            if baseline_seconds is not None and seconds not in (None, 0)
            else None
        )
        macro_value = headline(aggregate, critical=False)
        critical_value = headline(aggregate, critical=True)
        lines.append(
            f"{version:34} {formatted(macro_value, 8)} "
            f"{formatted(critical_value, 9)} "
            f"{formatted(aggregate.get('mean_local_realizability'), 8)} "
            f"{formatted(aggregate.get('mean_scene_p10_local_realizability'), 8)} "
            f"{family('collision'):>8} {family('support'):>8} "
            f"{family('plane'):>8} {'n/a':>8} "
            f"{family('semantic'):>9} "
            f"{'n/a' if seconds is None else f'{seconds:.2f}':>9} "
            f"{'n/a' if speedup is None else f'{speedup:.2f}x':>8}"
        )
        if baseline_data and version != baseline:
            base = baseline_data["aggregate"]
            base_macro = headline(base, critical=False)
            base_critical = headline(base, critical=True)
            macro_delta = (
                None
                if macro_value is None or base_macro is None
                else macro_value - base_macro
            )
            critical_delta = (
                None
                if critical_value is None or base_critical is None
                else critical_value - base_critical
            )
            lines.append(
                f"  delta vs {baseline}: macro="
                f"{'n/a' if macro_delta is None else f'{macro_delta:+.5f}'}, "
                f"critical={'n/a' if critical_delta is None else f'{critical_delta:+.5f}'}"
            )
    lines.extend(
        [
            "",
            "Threshold pass rates:",
            f"{'version':34} {'collision':>11} {'support':>11} {'plane':>11} {'boundary':>11} {'semantic':>11}",
            "-" * 94,
        ]
    )
    for version, data in report["versions"].items():
        families = data["aggregate"]["families"]
        def pass_rate(name):
            value = families.get(name, {}).get("pass_rate")
            return "n/a" if value is None else f"{100.0 * value:.1f}%"
        lines.append(
            f"{version:34} {pass_rate('collision'):>11} "
            f"{pass_rate('support'):>11} {pass_rate('plane'):>11} "
            f"{'n/a':>11} {pass_rate('semantic'):>11}"
        )
    lines.extend(
        [
            "",
            "Runtime details (wall-clock seconds measured around each Blender process):",
            f"{'version':34} {'mode':>10} {'scenes':>7} {'mean':>10} {'median':>10} {'p90':>10} {'gpu_total':>12}",
            "-" * 101,
        ]
    )
    for version, data in report["versions"].items():
        runtime = data.get("runtime", {})
        values = [
            runtime.get("mean_seconds"),
            runtime.get("median_seconds"),
            runtime.get("p90_seconds"),
            runtime.get("total_gpu_seconds"),
        ]
        formatted = ["n/a" if value is None else f"{value:.2f}" for value in values]
        lines.append(
            f"{version:34} {runtime.get('mode', 'n/a'):>10} "
            f"{runtime.get('scene_count', 0):7d} "
            f"{formatted[0]:>10} {formatted[1]:>10} {formatted[2]:>10} "
            f"{formatted[3]:>12}"
        )
        if runtime.get("mode") == "composite":
            stage = runtime.get("stage_only", {})
            stage_mean = stage.get("mean_seconds")
            lines.append(
                f"  components={'+'.join(runtime.get('components', []))}; "
                f"extra depth pass mean="
                f"{'n/a' if stage_mean is None else f'{stage_mean:.2f}s'}"
            )
    lines.extend(["", "Thresholds:", json.dumps(report["thresholds"], indent=2)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-results", default="a10_reusable_results/paper30")
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--versions", required=True, help="Comma-separated S4 versions")
    parser.add_argument("--geometry-version", default="v4_deepsearch")
    parser.add_argument("--baseline-version")
    parser.add_argument("--runtime-log", action="append", default=[], type=parse_assignment)
    parser.add_argument(
        "--runtime-composite",
        action="append",
        default=[],
        type=parse_assignment,
        help="TARGET=VERSION_A+VERSION_B; replaces TARGET runtime with per-scene sum",
    )
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--scene-csv", required=True)
    parser.add_argument("--object-csv", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--collision-volume-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--collision-policy", choices=("auto", "legacy", "relation_program"),
        default="auto",
        help="Use one common legacy all-pairs policy or Relation Program edges.",
    )
    parser.add_argument("--collision-fraction-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--collision-pairs-csv",
        help=(
            "Write one row per reported collision pair.  The bounding-box prism "
            "contains the true mesh, so these pairs are a complete superset of "
            "the truly interpenetrating ones; the rows carry the measurements "
            "needed to tell a real interpenetration from the artefact of a solid "
            "prism, such as a chair tucked under a table."
        ),
    )
    parser.add_argument("--contact-tolerance", type=float, default=0.05)
    parser.add_argument("--containment-tolerance", type=float, default=0.05)
    parser.add_argument("--support-overlap-tolerance", type=float, default=0.9)
    parser.add_argument("--plane-tolerance", type=float, default=0.05)
    parser.add_argument("--plane-orientation-tolerance", type=float, default=15.0)
    parser.add_argument("--boundary-tolerance", type=float, default=0.05)
    parser.add_argument("--semantic-angle-tolerance", type=float, default=20.0)
    parser.add_argument("--distance-tolerance", type=float, default=0.1)
    parser.add_argument(
        "--repair-degenerate-footprints",
        action="store_true",
        help=(
            "when the frozen geometry snapshot yields a collapsed XY footprint "
            "for an object, retry with the geometry carried by the layout "
            "itself. Off by default so frozen baseline numbers never move "
            "silently; the repaired object ids are always reported."
        ),
    )
    parser.add_argument(
        "--abstain-on-unmeasurable-footprints",
        action="store_true",
        help=(
            "omit a term whose reference geometry is absent instead of scoring "
            "it zero.  Scoring zero reports 'known bad' for a quantity that is "
            "merely unknown; the omitted objects are counted per family."
        ),
    )
    parser.add_argument(
        "--floor-from-walls",
        action="store_true",
        help=(
            "WITHDRAWN and inert.  The wall-derived floor plane was falsified on "
            "Smoke5: the derived top face landed 2.87 m to 3.15 m below the floor "
            "origin against a predicted +0.02 m, because the walls are 10 m "
            "construction panels rather than room-sized surfaces.  Use "
            "--structural-geometry-sidecar instead."
        ),
    )
    parser.add_argument(
        "--structural-geometry-sidecar",
        type=Path,
        default=None,
        help=(
            "JSON carrying bbox and length measured back from the constructed "
            "scene for objects whose geometry was never serialized.  Only fills "
            "gaps; it never overrides geometry that already yields a usable "
            "footprint."
        ),
    )
    parser.add_argument(
        "--placeholder-structural-lateral-extent",
        action="store_true",
        help=(
            "treat the lateral extent of floors, walls and other procedural "
            "structural objects as unmeasurable while still using their top face. "
            "Measurement shows the ground slab is exactly 10x10x0.04 m and the "
            "wall panels exactly 10 m tall regardless of the room, so containment "
            "and footprint overlap against them are trivially satisfied rather "
            "than verified, and the room's extent is not represented anywhere. "
            "Use together with --structural-geometry-sidecar."
        ),
    )
    args = parser.parse_args()
    if args.floor_from_walls:
        raise SystemExit(
            "--floor-from-walls is withdrawn: the wall-derived floor plane was "
            "falsified on Smoke5 (derived top face 2.87-3.15 m below the floor "
            "origin, predicted +0.02 m; boundary score ranged 0.298-1.000 across "
            "scenes). Use --structural-geometry-sidecar."
        )
    args._structural_geometry_sidecar = (
        load_structural_geometry_sidecar(args.structural_geometry_sidecar)
        if args.structural_geometry_sidecar is not None
        else {}
    )

    root = Path(args.saved_results)
    scenes = [
        line.strip()
        for line in Path(args.scenes).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    versions = [value.strip() for value in args.versions.split(",") if value.strip()]
    runtime_paths = dict(args.runtime_log)
    stage_runtime = {
        version: restrict_runtime_to_scenes(
            parse_runtime_jsonl(Path(path)), scenes
        )
        for version, path in runtime_paths.items()
    }
    effective_runtime = dict(stage_runtime)
    composite_sources: dict[str, list[str]] = {}
    for target, expression in args.runtime_composite:
        components = [part.strip() for part in expression.split("+") if part.strip()]
        if len(components) < 2:
            raise SystemExit(f"runtime composite requires at least two versions: {target}={expression}")
        missing = [part for part in components if part not in stage_runtime]
        if missing:
            raise SystemExit(f"runtime composite {target} has missing runtime logs: {missing}")
        common = set(stage_runtime[components[0]])
        for component in components[1:]:
            common &= set(stage_runtime[component])
        effective_runtime[target] = {
            scene: sum(stage_runtime[component][scene] for component in components)
            for scene in sorted(common)
        }
        composite_sources[target] = components
    report: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "geometry_version": args.geometry_version,
        "methodology": {
            "geometry": (
                "asset-scaled local OBB frozen from the geometry version's "
                "S4 pre-optimization placement_info_s3 snapshot, transformed "
                "by each candidate S4 pose"
            ),
            "aggregation": "equal family weight per scene, then equal scene weight",
            "localization": "one row per object in object_csv",
            "scope": (
                "deterministic geometric/semantic proxy; does not replace dynamic "
                "rigid-body stability or human perceptual evaluation"
            ),
        },
        "thresholds": {
            "collision_volume_m3": args.collision_volume_tolerance,
            "collision_fraction": args.collision_fraction_tolerance,
            "contact_gap_m": args.contact_tolerance,
            "containment_error_m": args.containment_tolerance,
            "support_footprint_overlap_ratio": args.support_overlap_tolerance,
            "plane_gap_m": args.plane_tolerance,
            "plane_orientation_deg": args.plane_orientation_tolerance,
            "boundary_error_m": args.boundary_tolerance,
            "semantic_angle_deg": args.semantic_angle_tolerance,
            "distance_error_m": args.distance_tolerance,
        },
        "failures": [],
        "versions": {},
    }
    scene_rows = []
    object_rows = []
    collision_pair_rows = []
    source_cache = {}
    for scene in scenes:
        try:
            geometry_path = find_geometry_snapshot(
                root, scene, args.geometry_version
            )
            geometry = load_json(geometry_path)
            validate_geometry_snapshot(geometry, geometry_path)
            source_cache[scene] = geometry
        except Exception as exc:
            report["failures"].append(
                {"scene": scene, "version": args.geometry_version, "error": str(exc)}
            )
    for version in versions:
        version_scenes = {}
        version_local_rows = []
        for scene in scenes:
            if scene not in source_cache:
                continue
            try:
                target = load_json(find_s4(root, scene, version))
                metrics, local_rows = evaluate_scene(source_cache[scene], target, args)
                # Pair details are moved out of the metrics document so the JSON
                # size stays independent of the pair count; the CSV is their only
                # persistent home.
                for detail in metrics.pop("collision_pair_details", []):
                    collision_pair_rows.append(
                        {"version": version, "scene": scene, **detail}
                    )
                version_scenes[scene] = metrics
                scene_rows.append(
                    {
                        "version": version,
                        "scene": scene,
                        "macro_realizability": metrics["macro_realizability"],
                        "critical_realizability": metrics["critical_realizability"],
                        "mean_local_realizability": metrics["mean_local_realizability"],
                        "p10_local_realizability": metrics["p10_local_realizability"],
                        "min_local_realizability": metrics["min_local_realizability"],
                        "unintended_collision_pairs": metrics["unintended_collision_pairs"],
                        **{
                            f"{family}_score": values["score"]
                            for family, values in metrics["families"].items()
                        },
                    }
                )
                for row in local_rows:
                    output_row = {"version": version, "scene": scene, **row}
                    object_rows.append(output_row)
                    version_local_rows.append(output_row)
            except Exception as exc:
                report["failures"].append(
                    {"scene": scene, "version": version, "error": str(exc)}
                )
        runtime = effective_runtime.get(version, {})
        report["versions"][version] = {
            "aggregate": aggregate_scenes(version_scenes),
            "runtime": runtime_summary(
                runtime,
                mode="composite" if version in composite_sources else "stage",
                components=composite_sources.get(version),
                stage_only=stage_runtime.get(version)
                if version in composite_sources else None,
            ),
            "worst_objects": sorted(
                version_local_rows,
                key=lambda row: (
                    row["local_realizability"] is None,
                    row["local_realizability"]
                    if row["local_realizability"] is not None else float("inf"),
                ),
            )[:20],
            "scenes": version_scenes,
        }

    Path(args.metrics_out).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for path, rows in [
        (Path(args.scene_csv), scene_rows),
        (Path(args.object_csv), object_rows),
    ]:
        fieldnames = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    if args.collision_pairs_csv:
        # Written even when empty, so an absent file always means the flag was not
        # passed rather than that no pair was found.
        path = Path(args.collision_pairs_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in collision_pair_rows for key in row}) or [
            "version",
            "scene",
            "first_id",
            "second_id",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(collision_pair_rows)
        print(f"Wrote {path} ({len(collision_pair_rows)} collision pairs)")
    write_report(Path(args.report_out), report, args.baseline_version)
    print(f"Wrote {args.metrics_out}")
    print(f"Wrote {args.scene_csv}")
    print(f"Wrote {args.object_csv}")
    print(f"Wrote {args.report_out}")
    print(f"Failures: {len(report['failures'])}")
    for version, data in report["versions"].items():
        aggregate = data["aggregate"]
        macro = aggregate.get("macro_realizability")
        critical = aggregate.get("critical_realizability")
        local = aggregate.get("mean_local_realizability")
        print(
            version,
            f"macro={'n/a' if macro is None else f'{macro:.6f}'}",
            f"critical={'n/a' if critical is None else f'{critical:.6f}'}",
            f"local={'n/a' if local is None else f'{local:.6f}'}",
            f"collisions={aggregate['unintended_collision_pairs']}",
            f"runtime_mean={data['runtime']['mean_seconds']}",
        )


if __name__ == "__main__":
    main()
