"""Pure geometry and scoring helpers for SceneProof COM stability audits.

This module deliberately has no Blender dependency.  Blender is responsible
only for extracting true transformed mesh triangles; the certificate math is
kept here so it can be unit-tested independently.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def convex_hull_2d(points: Iterable[Iterable[float]], epsilon: float = 1e-9) -> np.ndarray:
    values = np.asarray(list(points), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if not np.isfinite(values).all():
        raise ValueError("points must be finite")
    ordered = sorted({(float(row[0]), float(row[1])) for row in values})
    if len(ordered) < 3:
        raise ValueError("a support polygon requires at least three points")

    def cross(origin, first, second):
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= epsilon:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= epsilon:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        raise ValueError("support polygon is degenerate")
    return hull


def polygon_area(polygon: np.ndarray) -> float:
    polygon = np.asarray(polygon, dtype=np.float64)
    return 0.5 * abs(
        float(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
    )


def _inside_edge(point, start, end, epsilon=1e-9) -> bool:
    return bool(
        (end[0] - start[0]) * (point[1] - start[1])
        - (end[1] - start[1]) * (point[0] - start[0])
        >= -epsilon
    )


def _line_intersection(first, second, clip_first, clip_second) -> np.ndarray:
    direction = second - first
    clip_direction = clip_second - clip_first
    denominator = (
        direction[0] * clip_direction[1]
        - direction[1] * clip_direction[0]
    )
    if abs(float(denominator)) <= 1e-12:
        return second.copy()
    offset = clip_first - first
    scale = (
        offset[0] * clip_direction[1]
        - offset[1] * clip_direction[0]
    ) / denominator
    return first + scale * direction


def convex_polygon_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Intersect two CCW convex polygons with Sutherland-Hodgman clipping."""
    subject = [row.copy() for row in convex_hull_2d(first)]
    clip = convex_hull_2d(second)
    for index, clip_first in enumerate(clip):
        clip_second = clip[(index + 1) % len(clip)]
        input_points = subject
        subject = []
        if not input_points:
            break
        previous = input_points[-1]
        for current in input_points:
            current_inside = _inside_edge(current, clip_first, clip_second)
            previous_inside = _inside_edge(previous, clip_first, clip_second)
            if current_inside:
                if not previous_inside:
                    subject.append(
                        _line_intersection(
                            previous, current, clip_first, clip_second
                        )
                    )
                subject.append(current)
            elif previous_inside:
                subject.append(
                    _line_intersection(previous, current, clip_first, clip_second)
                )
            previous = current
    if len(subject) < 3:
        return np.empty((0, 2), dtype=np.float64)
    result = convex_hull_2d(subject)
    if polygon_area(result) <= 1e-12:
        return np.empty((0, 2), dtype=np.float64)
    return result


def signed_margin_to_convex_polygon(point: Iterable[float], polygon: np.ndarray) -> float:
    """Positive inside distance and negative outside distance in metres."""
    point = np.asarray(tuple(point), dtype=np.float64)
    polygon = convex_hull_2d(polygon)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("point must be a finite 2-vector")
    margins = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length <= 1e-12:
            continue
        # CCW polygons have their interior on the left of every edge.
        margins.append(
            float(edge[0] * (point[1] - start[1]) - edge[1] * (point[0] - start[0]))
            / length
        )
    if not margins:
        raise ValueError("support polygon has no valid edges")
    return min(margins)


def erode_convex_polygon(polygon: np.ndarray, margin_m: float) -> np.ndarray:
    """Inset a CCW convex polygon by a metric margin on every edge."""
    polygon = convex_hull_2d(polygon)
    if not math.isfinite(margin_m) or margin_m < 0:
        raise ValueError("erosion margin must be finite and non-negative")
    if margin_m == 0:
        return polygon.copy()
    shifted_points = []
    directions = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1e-12:
            raise ValueError("support polygon has a degenerate edge")
        inward = np.asarray([-direction[1], direction[0]]) / length
        shifted_points.append(start + margin_m * inward)
        directions.append(direction)
    result = []
    for index in range(len(polygon)):
        previous = (index - 1) % len(polygon)
        first = shifted_points[previous]
        first_direction = directions[previous]
        second = shifted_points[index]
        second_direction = directions[index]
        denominator = float(
            first_direction[0] * second_direction[1]
            - first_direction[1] * second_direction[0]
        )
        if abs(denominator) <= 1e-12:
            raise ValueError("eroded support polygon is degenerate")
        offset = second - first
        scale = float(
            (offset[0] * second_direction[1] - offset[1] * second_direction[0])
            / denominator
        )
        result.append(first + scale * first_direction)
    eroded = convex_hull_2d(result)
    if polygon_area(eroded) <= 1e-12:
        raise ValueError("erosion removes the complete support polygon")
    return eroded


def minimum_translation_into_convex_polygon(
    point: Iterable[float], polygon: np.ndarray, *, margin_m: float = 0.0
) -> np.ndarray:
    """Minimum Euclidean translation placing a point in an eroded polygon."""
    point = np.asarray(tuple(point), dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("point must be a finite 2-vector")
    feasible = erode_convex_polygon(polygon, margin_m)
    if signed_margin_to_convex_polygon(point, feasible) >= -1e-12:
        return np.zeros(2, dtype=np.float64)
    best = None
    best_distance = math.inf
    for index, start in enumerate(feasible):
        end = feasible[(index + 1) % len(feasible)]
        edge = end - start
        parameter = float((point - start) @ edge / max(float(edge @ edge), 1e-18))
        candidate = start + min(1.0, max(0.0, parameter)) * edge
        distance = float(np.linalg.norm(candidate - point))
        if distance < best_distance:
            best, best_distance = candidate, distance
    if best is None:
        raise ValueError("support projection has no feasible point")
    return np.asarray(best - point, dtype=np.float64)


def stability_class(margin_m: float, tolerance_m: float = 0.005) -> str:
    if not math.isfinite(margin_m):
        return "abstained"
    if margin_m < -tolerance_m:
        return "unstable"
    if margin_m < tolerance_m:
        return "marginal"
    return "stable"


def voxel_heightfield_contact_points(
    child_points: np.ndarray,
    supporter_points: np.ndarray,
    *,
    grid_pitch_m: float,
    contact_tolerance_m: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Find XY contact cells from child-bottom/supporter-top voxel envelopes."""
    child = np.asarray(child_points, dtype=np.float64)
    supporter = np.asarray(supporter_points, dtype=np.float64)
    if (
        child.ndim != 2
        or supporter.ndim != 2
        or child.shape[1:] != (3,)
        or supporter.shape[1:] != (3,)
    ):
        raise ValueError("voxel points must have shape (N, 3)")
    if grid_pitch_m <= 0 or contact_tolerance_m <= 0:
        raise ValueError("voxel contact tolerances must be positive")
    if not np.isfinite(child).all() or not np.isfinite(supporter).all():
        raise ValueError("voxel contact points must be finite")

    def envelope(points: np.ndarray, minimum: bool) -> dict[tuple[int, int], float]:
        cells = np.floor(points[:, :2] / grid_pitch_m + 0.5).astype(np.int64)
        result: dict[tuple[int, int], float] = {}
        for cell, z_value in zip(cells, points[:, 2]):
            key = (int(cell[0]), int(cell[1]))
            if key not in result:
                result[key] = float(z_value)
            elif minimum:
                result[key] = min(result[key], float(z_value))
            else:
                result[key] = max(result[key], float(z_value))
        return result

    child_bottom = envelope(child, True)
    supporter_top = envelope(supporter, False)
    common = sorted(set(child_bottom) & set(supporter_top))
    accepted = []
    gaps = []
    for cell in common:
        gap = child_bottom[cell] - supporter_top[cell]
        if abs(gap) <= contact_tolerance_m:
            accepted.append(
                [cell[0] * grid_pitch_m, cell[1] * grid_pitch_m]
            )
            gaps.append(float(gap))
    points = np.asarray(accepted, dtype=np.float64)
    if not accepted:
        points = np.empty((0, 2), dtype=np.float64)
    return points, {
        "candidate_common_cells": float(len(common)),
        "accepted_contact_cells": float(len(accepted)),
        "minimum_gap_m": min(gaps) if gaps else math.nan,
        "maximum_gap_m": max(gaps) if gaps else math.nan,
    }


def voxel_top_surface_component(
    supporter_points: np.ndarray,
    *,
    query_xy: Iterable[float],
    reference_z_m: float,
    grid_pitch_m: float,
    height_tolerance_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the connected top-surface voxel component nearest ``query_xy``."""
    points = np.asarray(supporter_points, dtype=np.float64)
    query = np.asarray(tuple(query_xy), dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("supporter voxel points must have shape (N, 3)")
    if query.shape != (2,) or not np.isfinite(query).all():
        raise ValueError("query_xy must be a finite 2-vector")
    if not np.isfinite(points).all() or not math.isfinite(reference_z_m):
        raise ValueError("voxel surface inputs must be finite")
    if grid_pitch_m <= 0 or height_tolerance_m <= 0:
        raise ValueError("voxel surface tolerances must be positive")
    cells = np.floor(points[:, :2] / grid_pitch_m + 0.5).astype(np.int64)
    top: dict[tuple[int, int], float] = {}
    for cell, z_value in zip(cells, points[:, 2]):
        key = int(cell[0]), int(cell[1])
        top[key] = max(top.get(key, -math.inf), float(z_value))
    eligible = {
        cell for cell, z_value in top.items()
        if abs(z_value - reference_z_m) <= height_tolerance_m
    }
    components: list[set[tuple[int, int]]] = []
    unseen = set(eligible)
    while unseen:
        seed = unseen.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            x, y = frontier.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor = x + dx, y + dy
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        frontier.append(neighbor)
        components.append(component)
    valid = [component for component in components if len(component) >= 3]
    if not valid:
        raise ValueError("no connected voxel top surface at the child height")
    def distance(component):
        values = np.asarray(list(component), dtype=np.float64) * grid_pitch_m
        return float(np.min(np.linalg.norm(values - query[None, :], axis=1)))
    selected = min(valid, key=distance)
    selected_points = np.asarray(sorted(selected), dtype=np.float64) * grid_pitch_m
    hull = convex_hull_2d(selected_points)
    return hull, {
        "eligible_top_cells": len(eligible),
        "connected_components": len(components),
        "selected_component_cells": len(selected),
        "selected_component_distance_m": distance(selected),
        "grid_pitch_m": float(grid_pitch_m),
        "height_tolerance_m": float(height_tolerance_m),
    }


def voxel_vertical_first_contact(
    child_points: np.ndarray,
    supporter_points: np.ndarray,
    *,
    grid_pitch_m: float,
    maximum_drop_m: float,
    penetration_tolerance_m: float,
    contact_band_m: float,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Find the first downward translation creating voxel surface contact.

    XY and orientation are fixed.  A supporter already penetrating the child
    fails closed rather than being used as a drop target.
    """
    child = np.asarray(child_points, dtype=np.float64)
    supporter = np.asarray(supporter_points, dtype=np.float64)
    if child.ndim != 2 or supporter.ndim != 2 or child.shape[1:] != (3,) or supporter.shape[1:] != (3,):
        raise ValueError("voxel points must have shape (N, 3)")
    if not np.isfinite(child).all() or not np.isfinite(supporter).all():
        raise ValueError("voxel points must be finite")
    if min(grid_pitch_m, maximum_drop_m, penetration_tolerance_m, contact_band_m) <= 0:
        raise ValueError("first-contact tolerances must be positive")

    def envelope(points: np.ndarray, minimum: bool) -> dict[tuple[int, int], float]:
        cells = np.floor(points[:, :2] / grid_pitch_m + 0.5).astype(np.int64)
        result: dict[tuple[int, int], float] = {}
        for cell, z_value in zip(cells, points[:, 2]):
            key = int(cell[0]), int(cell[1])
            if key not in result:
                result[key] = float(z_value)
            elif minimum:
                result[key] = min(result[key], float(z_value))
            else:
                result[key] = max(result[key], float(z_value))
        return result

    child_bottom = envelope(child, True)
    supporter_top = envelope(supporter, False)
    common = sorted(set(child_bottom) & set(supporter_top))
    if not common:
        raise ValueError("child footprint does not overlap supporter voxels")
    gaps = {cell: child_bottom[cell] - supporter_top[cell] for cell in common}
    penetrating = [gap for gap in gaps.values() if gap < -penetration_tolerance_m]
    if penetrating:
        raise ValueError("supporter already penetrates the child footprint")
    eligible = {cell: gap for cell, gap in gaps.items() if -penetration_tolerance_m <= gap <= maximum_drop_m}
    if not eligible:
        raise ValueError("no support surface within the vertical trust region")
    drop = max(0.0, min(eligible.values()))
    contact_cells = [cell for cell, gap in eligible.items() if gap <= drop + contact_band_m]
    if len(contact_cells) < 3:
        raise ValueError("first-contact patch has fewer than three voxel cells")
    contact_points = np.asarray(contact_cells, dtype=np.float64) * grid_pitch_m
    hull = convex_hull_2d(contact_points)
    return drop, hull, {
        "common_footprint_cells": len(common),
        "eligible_drop_cells": len(eligible),
        "first_contact_cells": len(contact_cells),
        "minimum_gap_m": float(min(gaps.values())),
        "maximum_gap_m": float(max(gaps.values())),
        "drop_m": float(drop),
        "grid_pitch_m": float(grid_pitch_m),
    }


def strongly_connected_components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan SCCs used to reject mutually self-supporting certificates."""
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    result: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for neighbor in adjacency.get(node, set()):
            if neighbor not in adjacency:
                continue
            if neighbor not in indices:
                visit(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], indices[neighbor])
        if lowlink[node] == indices[node]:
            component = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            result.append(sorted(component))

    for node in sorted(adjacency):
        if node not in indices:
            visit(node)
    return result


def ungrounded_cyclic_components(
    adjacency: dict[str, set[str]], grounded_nodes: set[str]
) -> tuple[list[list[str]], list[list[str]]]:
    """Split cyclic SCCs by whether their condensation DAG reaches ground."""
    components = strongly_connected_components(adjacency)
    component_of = {
        node: index
        for index, component in enumerate(components)
        for node in component
    }
    edges: dict[int, set[int]] = {index: set() for index in range(len(components))}
    direct_ground = {index: False for index in range(len(components))}
    cyclic = set()
    for index, component in enumerate(components):
        if len(component) > 1 or any(
            node in adjacency.get(node, set()) for node in component
        ):
            cyclic.add(index)
        for node in component:
            for supporter in adjacency.get(node, set()):
                if supporter in grounded_nodes:
                    direct_ground[index] = True
                elif supporter in component_of:
                    target = component_of[supporter]
                    if target != index:
                        edges[index].add(target)

    memo: dict[int, bool] = {}

    def reaches_ground(index: int) -> bool:
        if index in memo:
            return memo[index]
        # Condensation edges form a DAG, but assign a provisional value to
        # guard malformed input and keep the verifier fail-closed.
        memo[index] = False
        memo[index] = direct_ground[index] or any(
            reaches_ground(target) for target in edges[index]
        )
        return memo[index]

    grounded_cycles = [
        components[index] for index in sorted(cyclic) if reaches_ground(index)
    ]
    ungrounded_cycles = [
        components[index] for index in sorted(cyclic) if not reaches_ground(index)
    ]
    return ungrounded_cycles, grounded_cycles


def physical_support_score(row: dict[str, Any]) -> float | None:
    """Reproduce the evaluator's object-level support score from its CSV row."""
    try:
        gap = float(row["support_contact_gap_m"])
        containment = float(row["support_containment_error_m"])
        overlap = float(row["support_footprint_overlap_ratio"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (gap, containment, overlap)):
        return None
    linear = lambda value, tolerance: max(0.0, 1.0 - value / tolerance)
    return (
        linear(gap, 0.05)
        + linear(containment, 0.05)
        + min(1.0, overlap / 0.9)
    ) / 3.0
