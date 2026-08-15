"""Build self-consistent LayoutVLM semantic constraints from Imaginarium S1.

LayoutVLM treats ``distance``, ``align_with`` and ``point_towards`` as
directed relations.  Imaginarium currently exposes the equivalent information
through a mixture of S1 fields:

* ``directlyFacing`` maps to ``point_towards``;
* explicit ``alignWith``/``align_with`` fields map to ``align_with``;
* repeated-object ``group`` labels provide conservative, warm-start-consistent
  alignment constraints;

The paper retains only relations that agree with the numerical warm start.
This module applies the same policy and keeps the conversion independent of
Blender and PyTorch so that it is easy to audit and unit test.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence


def _unit_xy(vector: Sequence[float]) -> tuple[float, float] | None:
    length = math.hypot(float(vector[0]), float(vector[1]))
    if length <= 1e-8:
        return None
    return float(vector[0]) / length, float(vector[1]) / length


def _front_xy(matrix: Sequence[Sequence[float]]) -> tuple[float, float] | None:
    """Return Imaginarium's asset front (local -Y) in world XY."""
    return _unit_xy((-float(matrix[0][1]), -float(matrix[1][1])))


def _center_xy(matrix: Sequence[Sequence[float]]) -> tuple[float, float]:
    return float(matrix[0][3]), float(matrix[1][3])


def _angle_radians(value: Any, default: float = 0.0) -> float:
    try:
        angle = float(value)
    except (TypeError, ValueError):
        return default
    # S1/LLM fields conventionally use degrees.  Small values are also useful
    # as explicit radian offsets, matching LayoutVLM's internal API.
    if abs(angle) > 2.0 * math.pi + 1e-6:
        angle = math.radians(angle)
    return angle


def _relation_entries(value: Any) -> list[tuple[str, float, Mapping[str, Any]]]:
    """Normalize string/dict/list S1 relations into target/angle/metadata."""
    if value is None:
        return []
    if isinstance(value, str):
        return [(value, 0.0, {})]
    if isinstance(value, Mapping):
        target = next(
            (
                value[key]
                for key in ("target", "object", "id", "reference")
                if isinstance(value.get(key), str)
            ),
            None,
        )
        if target is None:
            return []
        return [
            (
                target,
                _angle_radians(value.get("angle", value.get("theta", 0.0))),
                value,
            )
        ]
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], str):
            angle = _angle_radians(value[1]) if len(value) >= 2 else 0.0
            metadata: dict[str, Any] = {}
            if len(value) >= 3:
                metadata = {"min": value[1], "max": value[2]}
            return [(value[0], angle, metadata)]
        entries: list[tuple[str, float, Mapping[str, Any]]] = []
        for item in value:
            entries.extend(_relation_entries(item))
        return entries
    return []


def _rotate_xy(
    vector: tuple[float, float],
    angle: float,
) -> tuple[float, float]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        cosine * vector[0] - sine * vector[1],
        sine * vector[0] + cosine * vector[1],
    )


def _cosine(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    return max(-1.0, min(1.0, first[0] * second[0] + first[1] * second[1]))


def _wrapped_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def build_semantic_relation_specs(
    obj_info: Mapping[str, Mapping[str, Any]],
    ordered_ids: Sequence[str],
    warmstart_matrices: Sequence[Sequence[Sequence[float]]],
    footprint_sizes: Sequence[Sequence[float]],
    *,
    point_min_cosine: float = 0.5,
    align_tolerance_degrees: float = 20.0,
    distance_margin_ratio: float = 0.1,
    distance_margin_min: float = 0.1,
) -> dict[str, Any]:
    """Return directed, warm-start-consistent semantic relation arrays.

    The returned lists use indices into ``ordered_ids`` and can be converted
    directly to tensors by the Blender adapter.
    """
    count = len(ordered_ids)
    if len(warmstart_matrices) != count:
        raise ValueError("warmstart_matrices must match ordered_ids")
    if len(footprint_sizes) != count:
        raise ValueError("footprint_sizes must match ordered_ids")
    if not -1.0 <= point_min_cosine <= 1.0:
        raise ValueError("point_min_cosine must lie in [-1, 1]")
    if align_tolerance_degrees < 0:
        raise ValueError("align_tolerance_degrees must be non-negative")
    if distance_margin_ratio < 0 or distance_margin_min < 0:
        raise ValueError("distance margins must be non-negative")

    id_to_index = {object_id: index for index, object_id in enumerate(ordered_ids)}
    centers = [_center_xy(matrix) for matrix in warmstart_matrices]
    fronts = [_front_xy(matrix) for matrix in warmstart_matrices]
    skipped: list[dict[str, str]] = []

    point_pairs: list[tuple[int, int]] = []
    point_offsets: list[float] = []
    distance_pairs: list[tuple[int, int]] = []
    distance_minimum: list[float] = []
    distance_maximum: list[float] = []
    align_pairs: list[tuple[int, int]] = []
    align_offsets: list[float] = []
    seen_point: set[tuple[int, int]] = set()
    seen_distance: set[tuple[int, int]] = set()
    seen_align: set[tuple[int, int]] = set()
    facing_sources: set[str] = set()

    def valid_pair(
        source_id: str,
        target_id: str,
        relation: str,
    ) -> tuple[int, int] | None:
        if source_id not in id_to_index or target_id not in id_to_index:
            skipped.append(
                {
                    "relation": relation,
                    "source": source_id,
                    "target": target_id,
                    "reason": "missing optimization object",
                }
            )
            return None
        if source_id == target_id:
            skipped.append(
                {
                    "relation": relation,
                    "source": source_id,
                    "target": target_id,
                    "reason": "self relation",
                }
            )
            return None
        return id_to_index[source_id], id_to_index[target_id]

    def add_distance(
        pair: tuple[int, int],
        explicit_minimum: Any = None,
        explicit_maximum: Any = None,
    ) -> None:
        if pair in seen_distance:
            return
        source, target = pair
        delta_x = centers[source][0] - centers[target][0]
        delta_y = centers[source][1] - centers[target][1]
        warm_distance = math.hypot(delta_x, delta_y)
        source_short = min(
            abs(float(footprint_sizes[source][0])),
            abs(float(footprint_sizes[source][1])),
        )
        target_short = min(
            abs(float(footprint_sizes[target][0])),
            abs(float(footprint_sizes[target][1])),
        )
        physical_minimum = 0.5 * (source_short + target_short)
        margin = max(
            distance_margin_min,
            distance_margin_ratio * max(warm_distance, physical_minimum),
        )
        try:
            requested_minimum = float(explicit_minimum)
        except (TypeError, ValueError):
            requested_minimum = physical_minimum
        try:
            requested_maximum = float(explicit_maximum)
        except (TypeError, ValueError):
            requested_maximum = max(warm_distance, physical_minimum)

        # LayoutVLM's sandbox expands constraints to include the numerical
        # initialization.  Keep the same invariant while preserving a useful
        # physical lower bound whenever the warm start allows it.
        minimum = max(0.0, min(requested_minimum, warm_distance) - margin)
        maximum = max(requested_maximum, warm_distance, minimum) + margin
        distance_pairs.append(pair)
        distance_minimum.append(minimum)
        distance_maximum.append(maximum)
        seen_distance.add(pair)

    point_keys = ("point_towards", "pointTowards", "directlyFacing")
    for source_id in ordered_ids:
        info = obj_info.get(source_id, {})
        for key in point_keys:
            for target_id, angle, _ in _relation_entries(info.get(key)):
                pair = valid_pair(source_id, target_id, key)
                if pair is None or pair in seen_point:
                    continue
                source, target = pair
                source_front = fronts[source]
                target_direction = _unit_xy(
                    (
                        centers[target][0] - centers[source][0],
                        centers[target][1] - centers[source][1],
                    )
                )
                if source_front is None or target_direction is None:
                    skipped.append(
                        {
                            "relation": "point_towards",
                            "source": source_id,
                            "target": target_id,
                            "reason": "degenerate warm-start direction",
                        }
                    )
                    continue
                rotated_front = _rotate_xy(source_front, -angle)
                if _cosine(rotated_front, target_direction) < point_min_cosine:
                    skipped.append(
                        {
                            "relation": "point_towards",
                            "source": source_id,
                            "target": target_id,
                            "reason": "not self-consistent with warm start",
                        }
                    )
                    continue
                point_pairs.append(pair)
                point_offsets.append(angle)
                seen_point.add(pair)
                facing_sources.add(source_id)
                # Imaginarium has no explicit S1 distance band yet.  The
                # warm-start-centered band supplies LayoutVLM's paired
                # distance primitive without inventing a new absolute layout.
                add_distance(pair)

    for source_id in ordered_ids:
        info = obj_info.get(source_id, {})
        for key in ("distance", "distanceTo", "distance_to"):
            for target_id, _, metadata in _relation_entries(info.get(key)):
                pair = valid_pair(source_id, target_id, key)
                if pair is None:
                    continue
                add_distance(
                    pair,
                    metadata.get("min", metadata.get("dmin")),
                    metadata.get("max", metadata.get("dmax")),
                )

    align_tolerance = math.radians(align_tolerance_degrees)

    def add_align(
        source_id: str,
        target_id: str,
        angle: float,
        relation: str,
    ) -> None:
        pair = valid_pair(source_id, target_id, relation)
        if pair is None or pair in seen_align:
            return
        source, target = pair
        source_front = fronts[source]
        target_front = fronts[target]
        if source_front is None or target_front is None:
            skipped.append(
                {
                    "relation": "align_with",
                    "source": source_id,
                    "target": target_id,
                    "reason": "degenerate warm-start front",
                }
            )
            return
        error = math.acos(
            _cosine(_rotate_xy(source_front, -angle), target_front)
        )
        if error > align_tolerance:
            skipped.append(
                {
                    "relation": "align_with",
                    "source": source_id,
                    "target": target_id,
                    "reason": "not self-consistent with warm start",
                }
            )
            return
        align_pairs.append(pair)
        align_offsets.append(angle)
        seen_align.add(pair)

    for source_id in ordered_ids:
        info = obj_info.get(source_id, {})
        for key in ("align_with", "alignWith"):
            for target_id, angle, _ in _relation_entries(info.get(key)):
                add_align(source_id, target_id, angle, key)

    # Repeated-object groups are an Imaginarium-specific adapter.  Do not
    # align table-facing chairs with one another: their point_towards relation
    # intentionally gives them different headings.
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for object_id in ordered_ids:
        group = obj_info.get(object_id, {}).get("group")
        if group is not None and object_id not in facing_sources:
            groups[str(group)].append(object_id)
    for members in groups.values():
        if len(members) < 2:
            continue
        anchor_id = members[0]
        anchor_index = id_to_index[anchor_id]
        anchor_front = fronts[anchor_index]
        if anchor_front is None:
            continue
        anchor_angle = math.atan2(anchor_front[1], anchor_front[0])
        for source_id in members[1:]:
            source_index = id_to_index[source_id]
            source_front = fronts[source_index]
            if source_front is None:
                continue
            source_angle = math.atan2(source_front[1], source_front[0])
            relative = _wrapped_angle(source_angle - anchor_angle)
            snapped = round(relative / (math.pi / 2.0)) * (math.pi / 2.0)
            if abs(_wrapped_angle(relative - snapped)) <= align_tolerance:
                add_align(source_id, anchor_id, snapped, "group")

    return {
        "point_pairs": point_pairs,
        "point_offsets": point_offsets,
        "distance_pairs": distance_pairs,
        "distance_minimum": distance_minimum,
        "distance_maximum": distance_maximum,
        "align_pairs": align_pairs,
        "align_offsets": align_offsets,
        "skipped": skipped,
    }
