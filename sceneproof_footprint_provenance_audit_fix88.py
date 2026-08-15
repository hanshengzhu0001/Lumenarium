#!/usr/bin/env python3
"""SceneProof Fix88: footprint provenance audit.

The question this answers
-------------------------
The Fix87 screen reported ``parent=floor_0 parent_vertices=1`` on every Smoke5
scene.  The floor's XY footprint convex hull has a single vertex, so the floor has
collapsed to a point in the horizontal plane.  Three consequences follow directly
from the evaluator's code, and all three are measurement artefacts rather than
scene defects:

1. ``outside_distance`` returns infinity when a polygon has fewer than three
   vertices, so every child declared as resting on the floor has an infinite
   containment error and therefore a zero containment summand.
2. ``convex_intersection`` returns nothing when either polygon has fewer than
   three vertices, so the same children have a zero footprint-overlap summand.
   Together with (1) this caps their support term at one third.
3. The boundary family measures every object against the floor polygon, so a
   collapsed floor makes every boundary error infinite and the whole family
   scores a constant zero.  That is consistent with the Fix85 run, where the
   measured boundary delta was exactly 0.0 in all five scenes.

The contact gap is affected too.  ``gap = |child.z_min - parent.z_max|``, and a
collapsed floor puts ``z_max`` at the slab's centre instead of its top surface,
so the reported gap is inflated by the slab's half thickness.  Every
floor-supported object in every Smoke5 scene reported a gap of exactly
0.020000 m, and the ground slab is built 0.04 m thick, so this audit tests the
half-thickness hypothesis explicitly.

What it reports
---------------
For every object: whether the frozen geometry snapshot yields a sound footprint,
whether the layout itself carries geometry that would yield a sound one, and how
many children a degenerate parent contaminates.  That determines which repair
applies: reading the layout's own geometry, or regenerating the snapshot
upstream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import eval_physical_realizability as evaluator


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def geometry_fields(info: Any) -> dict[str, Any]:
    """Describe the raw geometry fields the evaluator would read."""
    if not isinstance(info, dict):
        return {"present": False}
    try:
        bbox = np.asarray(info.get("bbox"), dtype=np.float64)
    except (TypeError, ValueError):
        bbox = np.empty((0, 3))
    try:
        length = np.asarray(info.get("length"), dtype=np.float64)
    except (TypeError, ValueError):
        length = np.empty((0,))
    bbox_usable = bool(bbox.shape == (8, 3) and np.isfinite(bbox).all())
    unique_bbox_points = (
        int(len({tuple(point) for point in np.round(bbox, 9)})) if bbox_usable else 0
    )
    length_usable = bool(
        length.shape == (3,) and np.isfinite(length).all() and np.all(length > 0)
    )
    return {
        "present": True,
        "bbox_shape": list(bbox.shape),
        "bbox_usable": bbox_usable,
        "bbox_unique_points": unique_bbox_points,
        "length": length.tolist() if length.size else None,
        "length_usable": length_usable,
        "has_pose": evaluator.matrix(info) is not None,
    }


def footprint_from(info: Any, pose_info: Any) -> dict[str, Any]:
    """Footprint the evaluator would derive from ``info`` at ``pose_info``'s pose."""
    pose = evaluator.matrix(pose_info) if isinstance(pose_info, dict) else None
    corners = evaluator.local_geometry(info) if isinstance(info, dict) else None
    if pose is None or corners is None:
        return {"derivable": False}
    world = evaluator.transform(corners, pose)
    polygon = evaluator.convex_hull(world[:, :2])
    return {
        "derivable": True,
        "vertex_count": int(len(polygon)),
        "degenerate": bool(len(polygon) < 3),
        "area_m2": float(evaluator.polygon_area(polygon)),
        "z_min_m": float(world[:, 2].min()),
        "z_max_m": float(world[:, 2].max()),
        "z_extent_m": float(world[:, 2].max() - world[:, 2].min()),
    }


def audit_scene(
    geometry: dict[str, Any],
    placement: dict[str, Any],
) -> dict[str, Any]:
    source_info = geometry.get("obj_info", {})
    target_info = placement.get("obj_info", {})

    children_of: dict[str, list[str]] = {}
    for name, info in target_info.items():
        if not isinstance(info, dict):
            continue
        parent_id = evaluator.support_id(info.get("supported"))
        if parent_id:
            children_of.setdefault(parent_id, []).append(name)

    reference = placement.get("reference_obj")
    if not reference:
        reference = next(
            (
                name
                for name in target_info
                if str(name).startswith(("floor_", "ground_"))
            ),
            None,
        )

    objects: dict[str, Any] = {}
    degenerate: list[str] = []
    repairable: list[str] = []
    unrepairable: list[str] = []
    for name, target in target_info.items():
        if evaluator.CAMERA.search(name) or not isinstance(target, dict):
            continue
        source = source_info.get(name)
        from_source = footprint_from(
            source if source is not None else target, target
        )
        from_layout = footprint_from(target, target)
        entry = {
            "object_id": name,
            "in_geometry_snapshot": source is not None,
            "snapshot_fields": geometry_fields(source),
            "layout_fields": geometry_fields(target),
            "footprint_from_snapshot": from_source,
            "footprint_from_layout": from_layout,
            "declared_child_count": len(children_of.get(name, [])),
            "declared_children": sorted(children_of.get(name, []))[:40],
            "is_reference_object": name == reference,
            "structural": bool(evaluator.STRUCTURAL.match(name)),
        }
        if from_source.get("degenerate"):
            degenerate.append(name)
            if from_layout.get("derivable") and not from_layout.get("degenerate"):
                entry["classification"] = "repairable_from_layout"
                repairable.append(name)
            else:
                entry["classification"] = "unrepairable_needs_upstream_snapshot"
                unrepairable.append(name)
        else:
            entry["classification"] = "sound"
        objects[name] = entry

    contaminated_children = sorted(
        {
            child
            for parent in degenerate
            for child in children_of.get(parent, [])
        }
    )
    reference_entry = objects.get(str(reference), {}) if reference else {}
    boundary_dead = bool(
        reference_entry.get("footprint_from_snapshot", {}).get("degenerate")
    )

    # Half-thickness hypothesis: a collapsed slab reports z_max at its centre, so
    # the phantom gap should equal half the slab's true vertical extent.  When the
    # layout carries no extent either, this check correctly predicts nothing and
    # the wall-derived reconstruction below is the only remaining evidence.
    hypothesis = []
    for parent in degenerate:
        layout = objects[parent]["footprint_from_layout"]
        if not layout.get("derivable"):
            continue
        predicted = 0.5 * float(layout.get("z_extent_m") or 0.0)
        hypothesis.append(
            {
                "parent_id": parent,
                "layout_z_extent_m": layout.get("z_extent_m"),
                "predicted_phantom_gap_m": predicted,
                "declared_child_count": len(children_of.get(parent, [])),
            }
        )

    # Wall geometry facts.  A previous version derived the floor plane from these
    # and it was falsified: the derived top face landed 2.87 m to 3.15 m below the
    # floor origin because the walls are 10 m construction panels, not room-sized
    # surfaces.  The facts are still reported, because they are what falsified the
    # hypothesis, but no floor plane is derived from them.
    geometries = evaluator.build_geometries(source_info, target_info)
    walls = {
        name: geometry
        for name, geometry in geometries.items()
        if evaluator.WALL.match(name)
    }
    usable_walls = {
        name: geometry
        for name, geometry in walls.items()
        if len(geometry.polygon) >= 3
    }
    wall_facts: dict[str, Any] = {
        "wall_count": len(walls),
        "usable_wall_count": len(usable_walls),
        "floor_plane_derivation": "withdrawn_falsified_on_smoke5",
    }
    if usable_walls:
        bottoms = [geometry.z_min for geometry in usable_walls.values()]
        heights = [
            geometry.z_max - geometry.z_min for geometry in usable_walls.values()
        ]
        collapsed_top = (
            geometries[str(reference)].z_max
            if reference and str(reference) in geometries
            else None
        )
        wall_facts.update(
            {
                "wall_bottom_min_m": float(min(bottoms)),
                "wall_bottom_median_m": float(np.median(bottoms)),
                "wall_bottom_max_m": float(max(bottoms)),
                "wall_height_median_m": float(np.median(heights)),
                "collapsed_floor_top_z_m": collapsed_top,
                # If the walls stood on the floor this would be about +0.02m.
                # On Smoke5 it is strongly negative, which is the falsification.
                "wall_bottom_minus_floor_origin_m": (
                    None
                    if collapsed_top is None
                    else float(np.median(bottoms)) - collapsed_top
                ),
                "walls_extend_below_floor_origin": (
                    None if collapsed_top is None else bool(min(bottoms) < collapsed_top)
                ),
            }
        )

    return {
        "reference_object": reference,
        "object_count": len(objects),
        "degenerate_footprint_object_ids": sorted(degenerate),
        "degenerate_footprint_count": len(degenerate),
        "repairable_from_layout": sorted(repairable),
        "repairable_from_layout_count": len(repairable),
        "unrepairable_object_ids": sorted(unrepairable),
        "unrepairable_count": len(unrepairable),
        "children_contaminated_by_degenerate_parent": contaminated_children,
        "children_contaminated_count": len(contaminated_children),
        "boundary_family_dead": boundary_dead,
        "phantom_gap_hypothesis": hypothesis,
        "wall_geometry_facts": wall_facts,
        "objects": objects,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--saved-results", type=Path, default=Path("a10_reusable_results/paper30")
    )
    parser.add_argument("--geometry-version", required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--max-listed", type=int, default=12)
    args = parser.parse_args()

    geometry_path = evaluator.find_geometry_snapshot(
        args.saved_results, args.scene, args.geometry_version
    )
    geometry = load_json(geometry_path)
    placement = load_json(args.placement)
    result = audit_scene(geometry, placement)
    report = {
        "schema_version": "sceneproof_footprint_provenance_v1",
        "scene": args.scene,
        "geometry_snapshot": str(geometry_path.resolve()),
        "placement": str(args.placement.resolve()),
        **result,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out_report.resolve()}")
    print(
        f"OBJECTS={result['object_count']} "
        f"DEGENERATE={result['degenerate_footprint_count']} "
        f"REPAIRABLE_FROM_LAYOUT={result['repairable_from_layout_count']} "
        f"UNREPAIRABLE={result['unrepairable_count']}"
    )
    print(
        f"REFERENCE_OBJECT={result['reference_object']} "
        f"BOUNDARY_FAMILY_DEAD={result['boundary_family_dead']} "
        f"CONTAMINATED_CHILDREN={result['children_contaminated_count']}"
    )
    for entry in result["phantom_gap_hypothesis"]:
        print(
            f"  phantom-gap check {entry['parent_id']}: "
            f"layout_z_extent={entry['layout_z_extent_m']} "
            f"predicted_gap={entry['predicted_phantom_gap_m']:.6f} "
            f"children={entry['declared_child_count']}"
        )
    preview = result["wall_geometry_facts"]
    print(
        "  walls: total={} usable={} floor_plane_derivation={}".format(
            preview["wall_count"],
            preview["usable_wall_count"],
            preview["floor_plane_derivation"],
        )
    )
    if preview.get("usable_wall_count"):
        print(
            "  wall bottoms: min={} median={} max={} height_median={} "
            "offset_vs_floor_origin={} extends_below_floor={}".format(
                preview.get("wall_bottom_min_m"),
                preview.get("wall_bottom_median_m"),
                preview.get("wall_bottom_max_m"),
                preview.get("wall_height_median_m"),
                preview.get("wall_bottom_minus_floor_origin_m"),
                preview.get("walls_extend_below_floor_origin"),
            )
        )
    for name in result["degenerate_footprint_object_ids"][: args.max_listed]:
        entry = result["objects"][name]
        snapshot = entry["snapshot_fields"]
        layout = entry["layout_fields"]
        print(
            f"  {name}: class={entry['classification']} "
            f"children={entry['declared_child_count']} "
            f"snapshot(bbox_usable={snapshot.get('bbox_usable')},"
            f"unique_pts={snapshot.get('bbox_unique_points')},"
            f"length={snapshot.get('length')}) "
            f"layout(bbox_usable={layout.get('bbox_usable')},"
            f"length={layout.get('length')}) "
            f"layout_vertices={entry['footprint_from_layout'].get('vertex_count')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
