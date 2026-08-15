#!/usr/bin/env python3
"""SceneProof Fix94: find the objects that visibly hang off their support.

The problem this addresses
--------------------------
In the frozen Fix61 baseline some objects, pillows above all, rest with their
centre of mass beyond the edge of the surface they sit on.Physically they would
tip; visually they read as floating.  This is a rendering-visible defect, so it is
worth fixing even though the aggregate metrics barely move.

What this tool does, and why it needs no simulation
---------------------------------------------------
The Fix62 true-mesh centre-of-mass audit already measured, per object,

    com_signed_margin_m        signed distance from the COM's vertical projection
                               to the boundary of the measured support polygon;
                               negative means the projection falls outside
    support_polygon_area_m2    the measured contact region
    stability_class            stable / marginal / unstable

so "which objects hang off their support" is answered by data already on disk.
This tool selects andranks them, and decides which of two repairs each one needs.

Two repairs, and the choice between them is measured rather than assumed
------------------------------------------------------------------------
``translate``
    Move the object horizontally by the shortest vector that brings the COM
    projection inside the support polygon.  This asserts "the placement had a
    position error; the object was meant to sit here".  It keeps height and
    orientation, so the support contact gap is unchanged by construction and no
    rotation-induced bounding-box proxy artefact can occur.
``tip``
    Let it rotate about the contact edge and settle, which may end up lying flat,
    leaning against something, or dropping to the next surface.  This asserts "the
    placement is physically impossible; the object would leave this surface".
    Requires simulation and is handled downstream.

The choice is not a guess.  ``translate`` is only proposed when it is *feasible*:
the required travel is small, the object still has support area under it
afterwards, and the translated bounding box does not newly overlap anything.  All
three are decided from geometry already in the placement document.  Everything
else is routed to ``tip``.

Why an unstable verdict is not always actionable
-----------------------------------------------
The mass properties behind ``com_signed_margin_m`` come from filling the true mesh
with uniform density, which the audit itself labels
``filled_voxel_mass_properties_unproven``.  That assumption is kept as the single
global assumption of this repair, and its consequences are checked where they are
actually visible, in the before-and-after render.  No geometric proxy for "the
real object probably has ballast" is used, because whether uniform density holds
is knowledge about real mass distribution and geometry does not contain it.  An
earlier version tried two such proxies, a fill ratio floor and a centre-of-mass
heightceiling; both were withdrawn after Smoke5 showed the fill ratio fires on
every legged furniture item, including a chair whose settle Fix84 had already
verified, and that the height ceiling split three identical pillows apart at an
arbitrary line.

What is filtered instead: self-contradictory measurements
---------------------------------------------------------
The voxel filling has a resolution floor, so a thin object gets thickened.  That
shows up as values which cannot be true of any object:

``fill_ratio``true-mesh volume over bounding-box volume.  Above one means
                     the mesh is reported larger than the box that contains it.
``com_height_ratio`` where the centre of mass sits within the object's own height.
                     Outside the zero-to-one band means it is reported outside the
                     object's own bounding box.

A mouse pad measured at a fill ratio of 1.99 and a file folder at a relative COM
height of -0.12 are not top-heavy objects; they are objects whose COM measurement
disagrees with their serialized geometry.  Their overhang distance is unusable, so
they are abstained from rather than repaired.  This is a validity check on the
data, not a guess about the world.
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

from eval_physical_realizability import (  # noqa: E402
    convex_hull,
    convex_intersection,
    polygon_area,
    support_id,
)
from modules._sceneproof_support_stability import (  # noqa: E402
    minimum_translation_into_convex_polygon,
)

STRUCTURAL = re.compile(r"^(floor|ground|wall|ceiling|carpet|rug)_\d+$")
ATTACHMENT_PARENT = re.compile(r"^(wall|ceiling)_\d+$")


def optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def point_in_convex_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    if len(polygon) < 3:
        return False
    signs = []
    for index in range(len(polygon)):
        first = polygon[index]
        second = polygon[(index + 1) % len(polygon)]
        edge = second - first
        signs.append(edge[0] * (point[1] - first[1]) - edge[1] * (point[0] - first[0]))
    return all(value >= -1e-12 for value in signs) or all(
        value <= 1e-12 for value in signs
    )


def closest_point_in_convex_polygon(
    point: np.ndarray, polygon: np.ndarray
) -> np.ndarray | None:
    """Nearest point of a convex polygon to``point``, or None if degenerate."""
    if len(polygon) < 3:
        return None
    if point_in_convex_polygon(point, polygon):
        return point.astype(np.float64)
    best = None
    best_distance = float("inf")
    for index in range(len(polygon)):
        first = polygon[index]
        second = polygon[(index + 1) % len(polygon)]
        edge = second - first
        length_squared = float(edge @ edge)
        if length_squared <= 1e-18:
            candidate = first
        else:
            parameter = float((point - first) @ edge / length_squared)
            parameter = min(1.0, max(0.0, parameter))
            candidate = first + parameter * edge
        distance = float(np.linalg.norm(point - candidate))
        if distance < best_distance:
            best_distance, best = distance, candidate
    return None if best is None else best.astype(np.float64)


def footprint_and_span(info: dict[str, Any]) -> tuple[np.ndarray, float, float] | None:
    """Footprint hull and vertical span from a placement entry's world bbox."""
    bbox = info.get("bbox")
    if bbox is None:
        return None
    corners = np.asarray(bbox, dtype=np.float64)
    if corners.shape != (8, 3) or not np.isfinite(corners).all():
        return None
    return (
        convex_hull(corners[:, :2]),
        float(corners[:, 2].min()),
        float(corners[:, 2].max()),
    )


def prism_overlap_volume(
    first: tuple[np.ndarray, float, float],
    second: tuple[np.ndarray, float, float],
) -> float:
    first_polygon, first_min, first_max = first
    second_polygon, second_min, second_max = second
    z_overlap = min(first_max, second_max) - max(first_min, second_min)
    if z_overlap <= 1e-6:
        return 0.0
    return polygon_area(convex_intersection(first_polygon, second_polygon)) * z_overlap


def declared_support_parents(obj_info: dict[str, Any]) -> set[str]:
    return {
        parent
        for info in obj_info.values()
        if isinstance(info, dict) and (parent := support_id(info.get("supported")))
    }


def com_measurement_consistency(
    record: dict[str, Any],
    info: dict[str, Any],
    z_min: float,
    z_max: float,
    *,
    fill_ratio_ceiling: float,
    com_height_band: float,
) -> dict[str, Any]:
    """Check the COM measurement against the object's own serialized geometry.

    Both indicators are impossibility tests, not judgements about the world: a mesh
    cannot be larger than the box containing it, and a centre of mass cannot lie
    outside the object.  Either failing means the voxel filling and the serialized
    bounding box disagree, which happens when a thin object is thickened by the
    voxel resolution floor.  The overhang distance derived from such a COM is
    unusable.
    """
    reasons: list[str] = []
    length = np.asarray(info.get("length") or [], dtype=np.float64).reshape(-1)
    box_volume = (
        float(np.prod(length))
        if length.shape == (3,) and np.isfinite(length).all() and (length > 0).all()
        else None
    )
    mesh_volume = optional_float(record.get("mesh_volume_m3"))
    fill_ratio = (
        float(mesh_volume / box_volume)
        if mesh_volume is not None and box_volume and box_volume > 1e-12
        else None
    )
    if fill_ratio is not None and fill_ratio > fill_ratio_ceiling:
        reasons.append("mesh_volume_exceeds_its_bounding_box")

    com = np.asarray(
        record.get("center_of_mass_world_m") or [], dtype=np.float64
    ).reshape(-1)
    height = z_max - z_min
    com_height_ratio = (
        float((com[2] - z_min) / height)
        if com.shape == (3,) and height > 1e-9
        else None
    )
    if com_height_ratio is not None and not (
        -com_height_band <= com_height_ratio <= 1.0 + com_height_band
    ):
        reasons.append("com_outside_its_bounding_box")

    return {
        "fill_ratio": fill_ratio,
        "com_height_ratio": com_height_ratio,
        "inconsistency_reasons": reasons,
        "com_measurement_consistent": not reasons,
    }


def screen_scene(
    com_audit: dict[str, Any],
    placement: dict[str, Any],
    *,
    margin_threshold_m: float,
    translate_budget_m: float,
    target_margin_m: float = 0.0,
    new_overlap_tolerance_m3: float,
    elevation_saturation_m: float,
    top_k: int,
    fill_ratio_ceiling: float = 1.05,
    com_height_band: float = 0.02,
    require_consistent_com: bool = False,
) -> dict[str, Any]:
    obj_info = placement.get("obj_info", {})
    records = com_audit.get("objects", {})
    parents = declared_support_parents(obj_info)
    geometry = {
        name: shape
        for name, info in obj_info.items()
        if isinstance(info, dict) and (shape := footprint_and_span(info)) is not None
    }

    candidates: list[dict[str, Any]] = []
    excluded: dict[str, int] = {}

    def exclude(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for object_id, record in sorted(records.items()):
        info = obj_info.get(object_id)
        if not isinstance(info, dict):
            exclude("absent_from_placement")
            continue
        no_contact_witness = bool(
            record.get("reason") == "no_mesh_or_voxel_horizontal_contact_patch"
            and record.get("center_of_mass_world_m") is not None
            and record.get("declared_parent_surface_polygon_world_xy_m") is not None
        )
        if record.get("status") != "measured" and not no_contact_witness:
            exclude(f"com_{record.get('status', 'missing')}")
            continue
        if not no_contact_witness and record.get("certificate_status") != "certified":
            exclude("com_uncertified")
            continue
        margin_key = (
            "declared_parent_surface_margin_m"
            if no_contact_witness
            else "com_signed_margin_m"
        )
        margin = optional_float(record.get(margin_key))
        if margin is None:
            exclude("margin_unmeasured")
            continue
        if margin >= -margin_threshold_m:
            exclude("com_inside_support_polygon")
            continue
        parent_id = support_id(info.get("supported"))
        if not parent_id:
            exclude("no_declared_support_parent")
            continue
        if ATTACHMENT_PARENT.match(parent_id):
            # Wall and ceiling children are plane attachments; gravity says
            # nothing useful about them.
            exclude("plane_attachment")
            continue
        if str(info.get("SpatialRel", "")).lower() == "inside":
            exclude("containment_held")
            continue
        if object_id in parents:
            # Moving it would move the ground under its own children, turning a
            # local repair into a stack collapse.
            exclude("is_a_declared_support_parent")
            continue
        shape = geometry.get(object_id)
        if shape is None:
            exclude("no_usable_geometry")
            continue

        overhang = -margin
        elevation = shape[1]
        salience = overhang * min(
            1.0, max(0.0, elevation) / max(elevation_saturation_m, 1e-9)
        )
        density = com_measurement_consistency(
            record,
            info,
            shape[1],
            shape[2],
            fill_ratio_ceiling=fill_ratio_ceiling,
            com_height_band=com_height_band,
        )
        actionable = density["com_measurement_consistent"]
        if require_consistent_com and not actionable:
            exclude("com_measurement_inconsistent")
            continue
        polygon_key = (
            "declared_parent_surface_polygon_world_xy_m"
            if no_contact_witness
            else "support_polygon_world_xy_m"
        )
        polygon = np.asarray(record.get(polygon_key) or [], dtype=np.float64).reshape(-1, 2)
        com = np.asarray(
            record.get("center_of_mass_world_m") or [], dtype=np.float64
        ).reshape(-1)

        translation = None
        translate_blockers: list[str] = []
        collided: list[str] = []
        if len(polygon) >= 3 and com.shape == (3,):
            try:
                delta = minimum_translation_into_convex_polygon(
                    com[:2], polygon, margin_m=target_margin_m
                )
            except ValueError:
                translate_blockers.append("eroded_support_polygon_infeasible")
            else:
                translation = delta.tolist()
                travel = float(np.linalg.norm(delta))
                if travel > translate_budget_m:
                    translate_blockers.append("travel_exceeds_budget")
                moved = (
                    np.asarray(shape[0], dtype=np.float64)
                    + np.asarray(translation, dtype=np.float64),
                    shape[1],
                    shape[2],
                )
                collided = []
                for other_id, other in geometry.items():
                    if other_id == object_id or other_id == parent_id:
                        continue
                    if STRUCTURAL.match(other_id):
                        continue
                    before = prism_overlap_volume(shape, other)
                    after = prism_overlap_volume(moved, other)
                    # The test is "does the nudge make things worse", not "is the
                    # object currently clear".  An overlap that already exists is
                    # the collision family's problem and must not veto a repair
                    # that leaves it unchanged.
                    if after > before + new_overlap_tolerance_m3:
                        collided.append(other_id)
                if collided:
                    translate_blockers.append("new_overlap")
        else:
            translate_blockers.append("support_polygon_unavailable")

        candidates.append(
            {
                "object_id": object_id,
                "support_parent_id": parent_id,
                "com_signed_margin_m": margin,
                "overhang_distance_m": overhang,
                "elevation_m": elevation,
                "visual_salience": salience,
                "stability_class": record.get("stability_class"),
                "support_witness_mode": (
                    "declared_parent_surface_without_current_contact"
                    if no_contact_witness
                    else "measured_contact_region"
                ),
                "post_projection_contact_must_be_certified": True,
                "support_polygon_area_m2": optional_float(
                    record.get("support_polygon_area_m2")
                ),
                "mesh_volume_m3": optional_float(record.get("mesh_volume_m3")),
                "fill_ratio": density["fill_ratio"],
                "com_height_ratio": density["com_height_ratio"],
                "com_measurement_consistent": density["com_measurement_consistent"],
                "com_inconsistency_reasons": density["inconsistency_reasons"],
                "actionable": actionable,
                "proposed_translation_xy_m": translation,
                "translate_feasible": bool(
                    translation is not None and not translate_blockers
                ),
                "translate_blockers": translate_blockers,
                "vertical_first_contact_candidate": record.get(
                    "vertical_first_contact_candidate"
                ),
                "recommended_action": (
                    "translate"
                    if translation is not None and not translate_blockers
                    else (
                        "vertical_first_contact_drop"
                        if record.get("vertical_first_contact_candidate")
                        else "tip"
                    )
                ),
                "new_overlap_object_ids": sorted(collided),
            }
        )

    candidates.sort(key=lambda item: -item["visual_salience"])
    selected = candidates[: max(top_k, 0)]

    # Two targets that could touch each other must not be dropped in the same
    # simulation, or one becomes the other's moving ground.  The test is
    # deliberately conservative: overlapping horizontal footprints are enough to
    # group them, because a tipping object sweeps sideways.  Each group is one
    # simulation inside the same Blender session, so the cost is seconds, not a
    # restart.
    groups: list[list[str]] = []
    for entry in selected:
        object_id = entry["object_id"]
        footprint = geometry[object_id][0]
        touching, disjoint = [object_id], []
        for group in groups:
            if any(
                polygon_area(convex_intersection(footprint, geometry[member][0])) > 0.0
                for member in group
            ):
                touching.extend(group)
            else:
                disjoint.append(group)
        groups = disjoint + [sorted(set(touching))]

    return {
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "excluded_counts": dict(sorted(excluded.items())),
        "selected": selected,
        "simulation_groups": groups,
        "all_candidates": candidates,
        "require_consistent_com": require_consistent_com,
        "policy": {
            "selection_uses_only_measurements_on_disk": True,
            "visual_salience_is_a_ranking_heuristic_not_a_measurement": True,
            "translate_feasibility_is_decided_from_geometry": True,
            "translation_is_minimum_norm_into_eroded_true_mesh_support": True,
            "rotation_and_height_are_frozen": True,
            "no_pose_is_modified_by_this_tool": True,
            # Uniform density is the single global assumption behind every
            # stability verdict.  It is not second-guessed by geometric proxies;
            # its consequences are checked in the before-and-after render, where
            # they are actually visible.
            "uniform_density_is_assumed_and_checked_by_rendering": True,
            # The only automatic filter is an impossibility test on the data: a
            # mesh larger than its own box, or a centre of mass outside it.
            "consistency_filter_is_a_data_validity_test_not_a_world_model": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--com-audit", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument(
        "--margin-threshold-m",
        type=float,
        default=0.005,
        help="Only flag an object whose COM falls this far outside the support "
        "polygon; below it the overhang is not visible.",
    )
    parser.add_argument("--translate-budget-m", type=float, default=0.15)
    parser.add_argument("--target-margin-m", type=float, default=0.01)
    parser.add_argument("--new-overlap-tolerance-m3", type=float, default=1e-6)
    parser.add_argument("--elevation-saturation-m", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--fill-ratio-ceiling",
        type=float,
        default=1.05,
        help="A true-mesh volume above this multiple of the bounding-box volume is "
        "impossible, so the COM measurement disagrees with the geometry.",
    )
    parser.add_argument(
        "--com-height-band",
        type=float,
        default=0.02,
        help="Tolerance for the centre of mass lying within the object's own "
        "height; outside it the measurement places the COM outside the object.",
    )
    parser.add_argument(
        "--require-consistent-com",
        action="store_true",
        help="Emit only candidates whose COM measurement is consistent with their "
        "serialized geometry.  This is the list a repair step should act on.",
    )
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()

    with args.com_audit.open("r", encoding="utf-8") as handle:
        com_audit = json.load(handle)
    with args.placement.open("r", encoding="utf-8") as handle:
        placement = json.load(handle)

    result = screen_scene(
        com_audit,
        placement,
        margin_threshold_m=args.margin_threshold_m,
        translate_budget_m=args.translate_budget_m,
        target_margin_m=args.target_margin_m,
        new_overlap_tolerance_m3=args.new_overlap_tolerance_m3,
        elevation_saturation_m=args.elevation_saturation_m,
        top_k=args.top_k,
        fill_ratio_ceiling=args.fill_ratio_ceiling,
        com_height_band=args.com_height_band,
        require_consistent_com=args.require_consistent_com,
    )
    report = {
        "schema_version": "sceneproof_overhang_screen_v3",
        "scene": args.scene,
        "thresholds": {
            "margin_threshold_m": args.margin_threshold_m,
            "translate_budget_m": args.translate_budget_m,
            "target_margin_m": args.target_margin_m,
            "new_overlap_tolerance_m3": args.new_overlap_tolerance_m3,
            "elevation_saturation_m": args.elevation_saturation_m,
            "top_k": args.top_k,
            "fill_ratio_ceiling": args.fill_ratio_ceiling,
            "com_height_band": args.com_height_band,
        },
        **result,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out_report.resolve()}")
    print(
        "OVERHANG candidates={} selected={} groups={}".format(
            result["candidate_count"],
            result["selected_count"],
            len(result["simulation_groups"]),
        )
    )
    for entry in result["selected"]:
        print(
            "  {}: parent={} overhang={:.4f}m elevation={:.3f}m "
            "stability={} action={}{}".format(
                entry["object_id"],
                entry["support_parent_id"],
                entry["overhang_distance_m"],
                entry["elevation_m"],
                entry["stability_class"],
                entry["recommended_action"],
                ""
                if not entry["translate_blockers"]
                else " blockers=" + ",".join(entry["translate_blockers"]),
            )
        )
        drop = entry.get("vertical_first_contact_candidate")
        if drop:
            print(
                "      first_contact_supporter={} drop_z={:.4f}m".format(
                    drop["supporter_id"], -float(drop["drop_m"])
                )
            )
        fill = entry["fill_ratio"]
        ratio = entry["com_height_ratio"]
        print(
            "      fill_ratio={} com_height_ratio={} actionable={}{}".format(
                "n/a" if fill is None else f"{fill:.3f}",
                "n/a" if ratio is None else f"{ratio:.3f}",
                entry["actionable"],
                ""
                if entry["actionable"]
                else " inconsistent=" + ",".join(entry["com_inconsistency_reasons"]),
            )
        )
    if result["excluded_counts"]:
        print(
            "  excluded: "
            + ", ".join(
                f"{reason}={count}"
                for reason, count in result["excluded_counts"].items()
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
