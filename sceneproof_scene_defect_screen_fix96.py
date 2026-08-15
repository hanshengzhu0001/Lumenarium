#!/usr/bin/env python3
"""SceneProof Fix96: find the defects that are actually visible, and attribute them.

Why this replaces the overhang line as the priority
---------------------------------------------------
Rendering the frozen Fix61 baseline settled the question the overhang screen could
not answer.  Of the five Smoke5 scenes, the overhang candidates are invisible in
four of them, for three different reasons: the object is cropped outside the frame
(livingroom_10), the view is top-down so a nine-centimetre overhang projects to a
few pixels behind the table edge (official_01), or the arrangement is simply
plausible (a bin liner sitting on a bin lid, streelitter_01).  The one visible case,
a leaning floor lamp, is exactly the one that should not be tipped.

What is glaring instead falls into three kinds, none of which gravity can fix:

* an object of the wrong size, such as a sofa rendered as a featureless block
  filling two thirds of the frame, or a duct larger than the alley it sits in;
* an object of the wrong shape, such as casino chairs rendered as thin curved
  sheets, or a chair rendered as a hollow patterned frame;
* a thin rod, appearing in two unrelated scenes with the same signature, standing
  vertically through the whole image.

Why the attribution is determined rather than guessed
-----------------------------------------------------
The placement document carries all three quantities in the scaling chain:

``pcd_obb_size``   the box size observed by depth estimation, in camera frame
``scale``          the factor applied to the retrieved asset
``length``         the resulting world dimensions

so the layer at fault follows from the data:

* ``scale`` far from one means the scaling was driven off a bad ``pcd_obb_size``,
  which is a depth-estimation problem in S3, not a retrieval problem;
* ``scale`` near one with absurd ``length`` means the asset's own dimensions are
  wrong, which is an asset-library problem;
* both plausible but the shape wrong means the wrong asset was retrieved, which is
  an S1/S2 problem and shows up here only as a name that does not match the
  semantic category.

``estimate_scale_factors_for_object`` clamps the factor to ``SCALE_THRESHOLD``, so
a factor sitting exactly on a bound is direct evidence of a runaway estimate; that
shows up as several objects sharing an identical factor.

Why theranking uses the camera
-------------------------------
The overhang screen ranked by ``overhang * elevation``, which silently assumed a
side view: a gap under an object is visible from the side and invisible from
above.  Here every candidate is projected through the actual scene camera, so the
ranking is by how much of the frame the defect occupies.  That is what decides
whether a defect matters to a figure in the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_physical_realizability import STRUCTURAL, CAMERA  # noqa: E402

# Defaults match``setup_camera`` in modules/S4_blender_layout_and_corr.py and the
# render resolution used by ``render_scene``.
DEFAULT_LENS_MM = 30.0
DEFAULT_SENSOR_MM = 36.0
DEFAULT_RESOLUTION = 1024

CATEGORY = re.compile(r"^(.*?)_\d+$")


def category_of(object_id: str) -> str:
    match = CATEGORY.match(object_id)
    return match.group(1) if match else object_id


def as_vector(value: Any, size: int = 3) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.shape != (size,) or not np.isfinite(array).all():
        return None
    return array


def as_matrix(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != (4, 4) or not np.isfinite(array).all():
        return None
    return array


def world_corners(info: dict[str, Any]) -> np.ndarray | None:
    try:
        array = np.asarray(info.get("bbox"), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != (8, 3) or not np.isfinite(array).all():
        return None
    return array


def project_to_pixels(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    focal_px: float,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points through a Blender camera looking down its local -Z.

    Returns pixel coordinates and a mask of points in front of the camera.
    """
    inverse = np.linalg.inv(camera_matrix)
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    local = (homogeneous @ inverse.T)[:, :3]
    depth = -local[:, 2]
    in_front = depth > 1e-6
    safe_depth = np.where(in_front, depth, 1.0)
    pixels = np.stack(
        (
            resolution / 2.0 + focal_px * local[:, 0] / safe_depth,
            resolution / 2.0 - focal_px * local[:, 1] / safe_depth,
        ),
        axis=1,
    )
    return pixels, in_front


def framed_pixel_area(
    pixels: np.ndarray, in_front: np.ndarray, resolution: int
) -> dict[str, Any]:
    """Screen footprint of a projected bounding box, clipped to the frame."""
    if not in_front.any():
        return {
            "in_front_of_camera": False,
            "pixel_area_fraction": 0.0,
            "fully_outside_frame": True,
            "pixel_bbox": None,
        }
    visible = pixels[in_front]
    low = visible.min(axis=0)
    high = visible.max(axis=0)
    clipped_low = np.maximum(low, 0.0)
    clipped_high = np.minimum(high, float(resolution))
    extent = np.maximum(clipped_high - clipped_low, 0.0)
    area = float(extent[0] * extent[1])
    return {
        "in_front_of_camera": bool(in_front.all()),
        "pixel_area_fraction": area / float(resolution * resolution),
        "fully_outside_frame": area <= 0.0,
        "pixel_bbox": [
            float(low[0]),
            float(low[1]),
            float(high[0]),
            float(high[1]),
        ],
    }


def shape_signatures(
    length: np.ndarray | None,
    *,
    rod_aspect: float,
    sheet_aspect: float,
) -> tuple[list[str], dict[str, Any]]:
    if length is None or (length <= 0).any():
        return [], {"sorted_edges_m": None, "rod_aspect": None, "sheet_aspect": None}
    edges = np.sort(length)[::-1]
    rod = float(edges[0] / max(edges[1], 1e-9))
    sheet = float(edges[2] / max(edges[1], 1e-9))
    reasons = []
    if rod > rod_aspect:
        reasons.append("rod_like_extreme_aspect")
    if sheet < sheet_aspect:
        reasons.append("sheet_like_extreme_aspect")
    return reasons, {
        "sorted_edges_m": edges.tolist(),
        "rod_aspect": rod,
        "sheet_aspect": sheet,
    }


def screen_scene(
    placement: dict[str, Any],
    *,
    lens_mm: float,
    sensor_mm: float,
    resolution: int,
    rod_aspect: float,
    sheet_aspect: float,
    scale_high: float,
    scale_low: float,
    scale_anisotropy: float,
    volume_outlier_factor: float,
    category_outlier_factor: float,
    top_k: int,
) -> dict[str, Any]:
    obj_info = placement.get("obj_info", {})
    focal_px = lens_mm / sensor_mm * resolution

    camera_matrix = None
    for name, info in obj_info.items():
        if CAMERA.search(name) and isinstance(info, dict):
            camera_matrix = as_matrix(info.get("pose_matrix_for_blender"))
            break

    entries: list[dict[str, Any]] = []
    for object_id, info in sorted(obj_info.items()):
        if not isinstance(info, dict) or CAMERA.search(object_id):
            continue
        if STRUCTURAL.match(object_id):
            continue
        length = as_vector(info.get("length"))
        entries.append(
            {
                "object_id": object_id,
                "category": category_of(object_id),
                "retrieved_asset": info.get("retrieved_asset"),
                "length_m": None if length is None else length.tolist(),
                "volume_m3": None if length is None else float(np.prod(length)),
                "pcd_obb_size_m": (
                    vector.tolist()
                    if (vector := as_vector(info.get("pcd_obb_size"))) is not None
                    else None
                ),
                "scale": (
                    vector.tolist()
                    if (vector := as_vector(info.get("scale"))) is not None
                    else None
                ),
                "corners": world_corners(info),
            }
        )

    volumes = [e["volume_m3"] for e in entries if e["volume_m3"]]
    scene_median_volume = float(np.median(volumes)) if volumes else None
    by_category: dict[str, list[float]] = {}
    for item in entries:
        if item["volume_m3"]:
            by_category.setdefault(item["category"], []).append(item["volume_m3"])
    category_median = {
        name: float(np.median(values))
        for name, values in by_category.items()
        if len(values) >= 3
    }

    # A clamped scale factor is direct evidence of a runaway estimate, and shows up
    # as the same value appearing across several *objects*.  Counting components
    # instead of objects would flag every uniformly scaled object, since its three
    # identical components would reach the threshold on their own.
    objects_per_scale_value: dict[float, set[str]] = {}
    for item in entries:
        for component in item["scale"] or []:
            key = round(float(component), 9)
            objects_per_scale_value.setdefault(key, set()).add(item["object_id"])
    repeated_scales = {
        value for value, owners in objects_per_scale_value.items() if len(owners) >= 3
    }

    findings: list[dict[str, Any]] = []
    for item in entries:
        length = as_vector(item["length_m"])
        reasons, shape = shape_signatures(
            length, rod_aspect=rod_aspect, sheet_aspect=sheet_aspect
        )
        scale = as_vector(item["scale"])
        scale_report: dict[str, Any] = {
            "scale_max": None,
            "scale_min": None,
            "scale_anisotropy": None,
            "scale_components_shared_with_others": [],
        }
        if scale is not None:
            scale_max = float(scale.max())
            scale_min = float(scale.min())
            scale_report["scale_max"] = scale_max
            scale_report["scale_min"] = scale_min
            scale_report["scale_anisotropy"] = float(
                scale_max / max(abs(scale_min), 1e-9)
            )
            shared = [
                float(component)
                for component in scale
                if round(float(component), 9) in repeated_scales
            ]
            scale_report["scale_components_shared_with_others"] = shared
            if scale_max > scale_high:
                reasons.append("scale_factor_far_above_one")
            if scale_min < scale_low:
                reasons.append("scale_factor_far_below_one")
            if scale_report["scale_anisotropy"] > scale_anisotropy:
                reasons.append("scale_grossly_anisotropic")
            if shared:
                reasons.append("scale_component_looks_clamped")

        volume = item["volume_m3"]
        if volume and scene_median_volume:
            if volume > volume_outlier_factor * scene_median_volume:
                reasons.append("volume_far_above_scene_median")
        peer_median = category_median.get(item["category"])
        if volume and peer_median:
            ratio = volume / peer_median
            if ratio > category_outlier_factor or ratio < 1.0 / category_outlier_factor:
                reasons.append("volume_disagrees_with_same_category_peers")

        projection = {
            "in_front_of_camera": None,
            "pixel_area_fraction": None,
            "fully_outside_frame": None,
            "pixel_bbox": None,
        }
        corners = item["corners"]
        if camera_matrix is not None and corners is not None:
            pixels, in_front = project_to_pixels(
                corners, camera_matrix, focal_px=focal_px, resolution=resolution
            )
            projection = framed_pixel_area(pixels, in_front, resolution)

        # The attribution follows from which quantities are wrong, so the report
        # names a pipeline stage instead of a symptom.
        if any(
            reason.startswith("scale_") for reason in reasons
        ):
            attribution = "s3_depth_driven_scaling"
        elif reasons:
            attribution = "asset_dimensions_or_retrieval"
        else:
            attribution = None

        findings.append(
            {
                **{k: v for k, v in item.items() if k != "corners"},
                **shape,
                **scale_report,
                **projection,
                "defect_reasons": reasons,
                "likely_stage": attribution,
                "screen_salience": (
                    (projection["pixel_area_fraction"] or 0.0) if reasons else 0.0
                ),
            }
        )

    flagged = [item for item in findings if item["defect_reasons"]]
    flagged.sort(key=lambda item: -item["screen_salience"])
    reason_counts: dict[str, int] = {}
    for item in flagged:
        for reason in item["defect_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "object_count": len(findings),
        "flagged_count": len(flagged),
        "camera_available": camera_matrix is not None,
        "focal_px": focal_px,
        "scene_median_volume_m3": scene_median_volume,
        "reason_counts": dict(sorted(reason_counts.items())),
        "stage_counts": {
            stage: sum(1 for item in flagged if item["likely_stage"] == stage)
            for stage in ("s3_depth_driven_scaling", "asset_dimensions_or_retrieval")
        },
        "worst_by_screen_area": flagged[: max(top_k, 0)],
        "all_flagged": flagged,
        "policy": {
            "ranking_uses_the_actual_scene_camera": True,
            "attribution_reads_the_scaling_chain_not_a_guess": True,
            "wrong_shape_is_not_detectable_here_only_wrong_size": True,
            "no_pose_or_asset_is_modified_by_this_tool": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--lens-mm", type=float, default=DEFAULT_LENS_MM)
    parser.add_argument("--sensor-mm", type=float, default=DEFAULT_SENSOR_MM)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--rod-aspect", type=float, default=20.0)
    parser.add_argument("--sheet-aspect", type=float, default=0.02)
    parser.add_argument("--scale-high", type=float, default=5.0)
    parser.add_argument("--scale-low", type=float, default=0.2)
    parser.add_argument("--scale-anisotropy", type=float, default=10.0)
    parser.add_argument("--volume-outlier-factor", type=float, default=50.0)
    parser.add_argument("--category-outlier-factor", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()

    with args.placement.open("r", encoding="utf-8") as handle:
        placement = json.load(handle)

    result = screen_scene(
        placement,
        lens_mm=args.lens_mm,
        sensor_mm=args.sensor_mm,
        resolution=args.resolution,
        rod_aspect=args.rod_aspect,
        sheet_aspect=args.sheet_aspect,
        scale_high=args.scale_high,
        scale_low=args.scale_low,
        scale_anisotropy=args.scale_anisotropy,
        volume_outlier_factor=args.volume_outlier_factor,
        category_outlier_factor=args.category_outlier_factor,
        top_k=args.top_k,
    )
    report = {
        "schema_version": "sceneproof_scene_defect_screen_v1",
        "scene": args.scene,
        "thresholds": {
            "rod_aspect": args.rod_aspect,
            "sheet_aspect": args.sheet_aspect,
            "scale_high": args.scale_high,
            "scale_low": args.scale_low,
            "scale_anisotropy": args.scale_anisotropy,
            "volume_outlier_factor": args.volume_outlier_factor,
            "category_outlier_factor": args.category_outlier_factor,
        },
        "camera": {
            "lens_mm": args.lens_mm,
            "sensor_mm": args.sensor_mm,
            "resolution": args.resolution,
        },
        **result,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out_report.resolve()}")

    def show(value, digits=3):
        return "n/a" if value is None else f"{value:.{digits}f}"

    print(
        "DEFECTS objects={} flagged={} camera={} median_volume={}".format(
            result["object_count"],
            result["flagged_count"],
            result["camera_available"],
            show(result["scene_median_volume_m3"], 4),
        )
    )
    for item in result["worst_by_screen_area"]:
        print(
            "  {}: asset={} screen={:.2%} volume={}m3 edges={} stage={}".format(
                item["object_id"],
                item["retrieved_asset"],
                item["pixel_area_fraction"] or 0.0,
                show(item["volume_m3"], 4),
                "n/a"
                if item["sorted_edges_m"] is None
                else "[" + ",".join(f"{e:.2f}" for e in item["sorted_edges_m"]) + "]",
                item["likely_stage"],
            )
        )
        print(
            "      scale={} pcd_obb={} reasons={}".format(
                "n/a" if item["scale"] is None else
                "[" + ",".join(f"{s:.2f}" for s in item["scale"]) + "]",
                "n/a" if item["pcd_obb_size_m"] is None else
                "[" + ",".join(f"{s:.2f}" for s in item["pcd_obb_size_m"]) + "]",
                ",".join(item["defect_reasons"]),
            )
        )
    if result["reason_counts"]:
        print(
            "  reasons: "
            + ", ".join(
                f"{name}={count}" for name, count in result["reason_counts"].items()
            )
        )
    print(
        "  stages: "
        + ", ".join(f"{name}={count}" for name, count in result["stage_counts"].items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
