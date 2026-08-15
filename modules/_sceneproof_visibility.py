"""Pure helpers for SceneProof rendered-mesh visibility audits."""

from __future__ import annotations

import numpy as np


def decode_color_id_image(rgb_image, colors, *, tolerance=0.12):
    """Decode a flat RGB color-ID render into integer labels (0=unknown)."""
    image = np.asarray(rgb_image, dtype=np.float32)
    palette = np.asarray(colors, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb_image must have shape (H, W, 3)")
    if palette.ndim != 2 or palette.shape[1] != 3 or not len(palette):
        raise ValueError("colors must have shape (N, 3) with N > 0")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    distances = np.linalg.norm(
        image[:, :, None, :] - palette[None, None, :, :], axis=3
    )
    closest = distances.argmin(axis=2)
    minimum = distances.min(axis=2)
    labels = np.zeros(image.shape[:2], dtype=np.int32)
    matched = minimum <= float(tolerance)
    labels[matched] = closest[matched] + 1
    return labels


def binary_mask_metrics(rendered_mask, observed_mask):
    """Return overlap metrics for two equally sized boolean masks."""
    rendered = np.asarray(rendered_mask, dtype=bool)
    observed = np.asarray(observed_mask, dtype=bool)
    if rendered.shape != observed.shape:
        raise ValueError(
            "rendered and observed masks must have the same shape: "
            f"{rendered.shape} != {observed.shape}"
        )
    rendered_pixels = int(rendered.sum())
    observed_pixels = int(observed.sum())
    intersection = int(np.logical_and(rendered, observed).sum())
    union = int(np.logical_or(rendered, observed).sum())
    return {
        "rendered_visible_pixels": rendered_pixels,
        "observed_mask_pixels": observed_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": float(intersection / union) if union else 1.0,
        "precision": (
            float(intersection / rendered_pixels)
            if rendered_pixels
            else 0.0
        ),
        "recall": (
            float(intersection / observed_pixels)
            if observed_pixels
            else 0.0
        ),
    }


def classify_visibility(full_pixels, isolated_pixels, *, minimum_pixels=1):
    """Classify whether geometry is absent, occluded, or scene-visible."""
    full_pixels = int(full_pixels)
    isolated_pixels = int(isolated_pixels)
    minimum_pixels = int(minimum_pixels)
    if minimum_pixels < 1:
        raise ValueError("minimum_pixels must be positive")
    if isolated_pixels < minimum_pixels:
        return "outside_view_or_degenerate"
    if full_pixels < minimum_pixels:
        return "fully_occluded"
    if full_pixels < isolated_pixels:
        return "partially_occluded"
    return "visible"


def attribute_occluders(
    full_labels,
    isolated_mask,
    label_to_object,
    *,
    target_label,
):
    """Count frontmost labeled objects over one target's isolated silhouette."""
    labels = np.asarray(full_labels, dtype=np.int32)
    isolated = np.asarray(isolated_mask, dtype=bool)
    if labels.shape != isolated.shape:
        raise ValueError("full_labels and isolated_mask must have equal shape")
    target_label = int(target_label)
    counts = {}
    unknown = 0
    target_visible = 0
    for label, count in zip(*np.unique(labels[isolated], return_counts=True)):
        label = int(label)
        count = int(count)
        if label == target_label:
            target_visible += count
        elif label == 0 or label not in label_to_object:
            unknown += count
        else:
            object_id = str(label_to_object[label])
            counts[object_id] = counts.get(object_id, 0) + count
    total = int(isolated.sum())
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "isolated_pixels": total,
        "target_visible_pixels": target_visible,
        "unknown_or_background_pixels": unknown,
        "occluder_pixels_by_object": dict(ordered),
        "dominant_occluder": ordered[0][0] if ordered else None,
        "dominant_occluder_pixels": ordered[0][1] if ordered else 0,
        "dominant_occluder_fraction": (
            float(ordered[0][1] / total) if ordered and total else 0.0
        ),
    }


def convex_hull_2d(points, *, epsilon=1e-9):
    """Return a counter-clockwise convex hull without SciPy."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    unique = sorted({(float(x), float(y)) for x, y in values})
    if len(unique) < 3:
        raise ValueError("a finite plane patch needs at least 3 points")

    def cross(origin, first, second):
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(
            lower[-2], lower[-1], point
        ) <= epsilon:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(
            upper[-2], upper[-1], point
        ) <= epsilon:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        raise ValueError("plane patch projection is degenerate")
    return hull


def minimum_translation_into_convex_polygon(
    container_points,
    child_points,
    *,
    tolerance=1e-9,
):
    """Find the minimum 2D translation containing all child points.

    The finite host patch is replaced by its convex hull.  The resulting
    two-variable convex QP is solved exactly by enumerating single active
    edges and pairs of active edges.  No pose is mutated by this helper.
    """
    hull = convex_hull_2d(container_points, epsilon=tolerance)
    child = np.asarray(child_points, dtype=np.float64)
    if child.ndim != 2 or child.shape[1] != 2 or not len(child):
        raise ValueError("child_points must have shape (N, 2), N > 0")
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= tolerance):
        raise ValueError("plane patch hull has a degenerate edge")
    # A CCW polygon's inward normal is the left normal of each edge.
    inward = np.column_stack((-edges[:, 1], edges[:, 0])) / lengths[:, None]
    margins = np.min(
        np.sum(
            (child[:, None, :] - hull[None, :, :])
            * inward[None, :, :],
            axis=2,
        ),
        axis=0,
    )
    bounds = -margins

    def feasible(candidate):
        return bool(np.all(inward @ candidate >= bounds - tolerance))

    candidates = []
    zero = np.zeros(2, dtype=np.float64)
    if feasible(zero):
        candidates.append(zero)
    for normal, bound in zip(inward, bounds):
        candidate = normal * bound
        if feasible(candidate):
            candidates.append(candidate)
    for first in range(len(inward)):
        for second in range(first + 1, len(inward)):
            system = np.stack((inward[first], inward[second]))
            determinant = float(np.linalg.det(system))
            if abs(determinant) <= tolerance:
                continue
            candidate = np.linalg.solve(
                system,
                np.asarray((bounds[first], bounds[second])),
            )
            if feasible(candidate):
                candidates.append(candidate)
    maximum_violation = float(np.maximum(bounds, 0.0).max(initial=0.0))
    if not candidates:
        return {
            "feasible": False,
            "contained": False,
            "maximum_outside_distance": maximum_violation,
            "translation": None,
            "translation_norm": None,
            "host_hull": hull.tolist(),
        }
    translation = min(candidates, key=lambda value: float(value @ value))
    norm = float(np.linalg.norm(translation))
    return {
        "feasible": True,
        "contained": bool(maximum_violation <= tolerance),
        "maximum_outside_distance": maximum_violation,
        "translation": translation.tolist(),
        "translation_norm": norm,
        "host_hull": hull.tolist(),
    }
