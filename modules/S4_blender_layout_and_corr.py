import bpy
import numpy as np
from mathutils import Matrix, Vector, Quaternion, Euler
import os
import math
import json
import re
import pandas as pd
import mathutils
import copy
import argparse
import sys
import torch
import trimesh
import scipy
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
from bpy_extras.object_utils import world_to_camera_view
import pyassimp
import functools
import pickle
import tempfile
import random
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed


def _configure_lumenarium_trial_seed():
    raw = os.environ.get("LUMENARIUM_TRIAL_SEED", "").strip()
    if not raw:
        return None
    seed = int(raw) & 0x7FFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        bpy.context.scene.cycles.seed = seed
    except Exception:
        pass
    print(f"[Lumenarium] Trial seed: {seed}", flush=True)
    return seed


_LUMENARIUM_TRIAL_SEED = _configure_lumenarium_trial_seed()


class _NumPyCompatUnpickler(pickle.Unpickler):
    """Translate NumPy 2 private pickle paths for Blender's NumPy 1.x."""

    def find_class(self, module, name):
        if module == "numpy._core":
            module = "numpy.core"
        elif module.startswith("numpy._core."):
            module = "numpy.core." + module[len("numpy._core."):]
        return super().find_class(module, name)


def load_numpy_compatible_pickle(stream):
    return _NumPyCompatUnpickler(stream).load()


def _convex_hull_indices_2d(points, epsilon=1e-9):
    """Return counter-clockwise source indices of a 2D convex hull."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    ordered = sorted(
        (float(point[0]), float(point[1]), index)
        for index, point in enumerate(points)
    )
    unique = []
    for x_value, y_value, index in ordered:
        if unique and abs(x_value - unique[-1][0]) <= epsilon and abs(
            y_value - unique[-1][1]
        ) <= epsilon:
            continue
        unique.append((x_value, y_value, index))
    if len(unique) <= 1:
        return [entry[2] for entry in unique]

    def cross(origin, first, second):
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower = []
    for entry in unique:
        while len(lower) >= 2 and cross(
            lower[-2], lower[-1], entry
        ) <= epsilon:
            lower.pop()
        lower.append(entry)
    upper = []
    for entry in reversed(unique):
        while len(upper) >= 2 and cross(
            upper[-2], upper[-1], entry
        ) <= epsilon:
            upper.pop()
        upper.append(entry)
    return [entry[2] for entry in lower[:-1] + upper[:-1]]

# 添加项目根目录到 Python 路径
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent  # 项目根目录（scripts的父目录）
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.drop_sim_script import run_drop_simulation
from modules._s4_layoutvlm_ops import (
    convex_polygon_halfspaces,
    depth_aware_reprojection_loss,
    gate_support_containment_pairs,
    identity_reprojection_error,
    initialize_pose_variables,
    local_box_corners,
    oriented_penetration_loss,
    optimize_collision_stage,
    optimize_contact_stage,
    optimize_plane_stage,
    optimize_semantic_stage,
    pair_index_tensor,
    project_support_footprints_,
    reproject_pose_matrices,
    select_confident_discrete_pose_repairs,
    stack_pose_matrices,
    support_contact_loss,
    support_planar_containment_loss,
    room_boundary_loss,
    transform_points,
)
from modules._s4_layoutvlm_relations import build_semantic_relation_specs
from modules._s4_settle import (
    DEFAULT_MAX_SETTLE_GAP_M,
    resolve_settle_policy,
    rotation_explained_horizontal_motion,
    shortest_rotation_angle,
    settle_after_simulation_enabled,
    settle_delta_z,
)
from modules._sceneproof_compile import (
    audit_live_factor_parity,
    compile_legacy_relation_programs,
)
from modules._sceneproof_factor_binding import (
    audit_factor_semantics_and_ownership,
    build_runtime_factor_rows,
)
from modules._sceneproof_visibility import (
    attribute_occluders,
    binary_mask_metrics,
    classify_visibility,
    decode_color_id_image,
    minimum_translation_into_convex_polygon,
)
from modules._sceneproof_support_stability import (
    convex_hull_2d,
    convex_polygon_intersection,
    polygon_area,
    signed_margin_to_convex_polygon,
    minimum_translation_into_convex_polygon as minimum_com_translation_into_support,
    stability_class,
    strongly_connected_components,
    ungrounded_cyclic_components,
    voxel_heightfield_contact_points,
    voxel_top_surface_component,
    voxel_vertical_first_contact,
)

# ===== 重要：让所有print自动flush，避免输出被缓冲 =====
print = functools.partial(print, flush=True)

# 确保环境干净 (在业务逻辑开始前释放可能残留的显存)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

eps = 1e-3
LAYOUTVLM_STAGES = (
    "reproject",
    "collision",
    "contact",
    "wall",
    "semantic",
    "boundary",
    "full",
    "depth",
)


def _sceneproof_mesh_hierarchy(root):
    """Return every mesh owned by one imported instance root."""
    meshes = []
    stack = [root]
    visited = set()
    while stack:
        current = stack.pop()
        pointer = current.as_pointer()
        if pointer in visited:
            continue
        visited.add(pointer)
        if current.type == "MESH":
            meshes.append(current)
        stack.extend(list(current.children))
    return meshes


def _sceneproof_world_trimesh(root):
    """Build one true, evaluated, world-space mesh for an imported instance."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    components = []
    for mesh_object in _sceneproof_mesh_hierarchy(root):
        evaluated = mesh_object.evaluated_get(depsgraph)
        evaluated_mesh = evaluated.to_mesh()
        try:
            evaluated_mesh.calc_loop_triangles()
            vertices = np.asarray(
                [
                    tuple(evaluated.matrix_world @ vertex.co)
                    for vertex in evaluated_mesh.vertices
                ],
                dtype=np.float64,
            )
            faces = np.asarray(
                [tuple(triangle.vertices) for triangle in evaluated_mesh.loop_triangles],
                dtype=np.int64,
            )
            if vertices.size and faces.size:
                components.append(
                    trimesh.Trimesh(
                        vertices=vertices,
                        faces=faces,
                        process=False,
                    )
                )
        finally:
            evaluated.to_mesh_clear()
    if not components:
        raise ValueError(f"no evaluated mesh triangles for {root.name}")
    mesh = trimesh.util.concatenate(components)
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f"non-finite evaluated mesh for {root.name}")
    return mesh


def _sceneproof_voxel_mass_properties(
    mesh,
    *,
    target_longest_axis_voxels=64,
    minimum_pitch_m=0.002,
    maximum_pitch_m=0.03,
):
    """Return deterministic filled-voxel mass properties for an open mesh.

    Imported production assets are frequently non-watertight.  Rasterizing
    their real triangles, closing one-voxel seams, and filling enclosed cells
    gives a bounded, auditable mass witness without substituting an OBB/bbox.
    """
    extents = np.asarray(mesh.extents, dtype=np.float64)
    longest = float(extents.max()) if extents.size else 0.0
    if not math.isfinite(longest) or longest <= 0:
        raise ValueError("mesh has invalid extents for voxel mass properties")
    pitch = float(
        np.clip(
            longest / float(target_longest_axis_voxels),
            minimum_pitch_m,
            maximum_pitch_m,
        )
    )
    voxel_grid = mesh.voxelized(pitch=pitch, method="subdivide")
    surface = np.asarray(voxel_grid.matrix, dtype=bool)
    if surface.ndim != 3 or not surface.any():
        raise ValueError("mesh voxelization produced no occupied cells")
    closed = ndimage.binary_closing(
        surface,
        structure=ndimage.generate_binary_structure(3, 1),
        iterations=1,
    )
    # Preserve boundary samples removed by closing; they are still real mesh
    # witnesses and must remain part of the filled body.
    closed |= surface
    filled = ndimage.binary_fill_holes(closed)
    indices = np.argwhere(filled)
    if len(indices) < 8:
        raise ValueError("filled voxel body has fewer than eight cells")
    points = np.asarray(
        voxel_grid.indices_to_points(indices), dtype=np.float64
    )
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("filled voxel mass points are invalid")
    center_mass = points.mean(axis=0)
    surface_count = int(surface.sum())
    filled_count = int(filled.sum())
    return {
        "center_mass": center_mass,
        "volume": float(filled_count * pitch ** 3),
        # Internal geometric witness.  Callers must remove this array before
        # persisting the public JSON audit.
        "occupied_points": points,
        "pitch_m": pitch,
        "surface_voxels": surface_count,
        "filled_voxels": filled_count,
        "interior_voxels": max(filled_count - int(closed.sum()), 0),
        "fill_expansion_ratio": float(filled_count / max(surface_count, 1)),
        "grid_shape": [int(value) for value in surface.shape],
    }


def _sceneproof_support_program_ids(document):
    result = defaultdict(list)
    bundle = document.get("sceneproof_relation_programs", {})
    for program in bundle.get("programs", []):
        if program.get("kind") != "SUPPORT":
            continue
        participants = program.get("participants", [])
        child = next(
            (
                participant.get("object_id")
                for participant in participants
                if participant.get("role") in {"child", "object"}
            ),
            None,
        )
        if child is None and participants:
            child = participants[0].get("object_id")
        if isinstance(child, str):
            result[child].append(str(program.get("program_id", program.get("id"))))
    return result


def audit_sceneproof_true_mesh_com_support(
    placement_document,
    *,
    contact_tolerance_m=0.05,
    surface_band_m=0.01,
    normal_cosine=0.7,
    stability_tolerance_m=0.005,
    scoped_object_ids=None,
):
    """Audit true-mesh COM support without mutating poses or rigid bodies.

    The support polygon is derived from the intersection of actual upward and
    downward mesh faces near a common horizontal contact plane.  All objects
    at that plane are considered potential simultaneous supporters.  Invalid
    mass properties and non-horizontal contacts fail closed to ABSTAIN.
    """
    if contact_tolerance_m <= 0 or surface_band_m <= 0:
        raise ValueError("COM audit tolerances must be positive")
    if not 0 < normal_cosine <= 1:
        raise ValueError("COM audit normal cosine must be in (0, 1]")
    obj_info = placement_document.get("obj_info", {})
    roots = {
        object_id: bpy.data.objects.get(object_id)
        for object_id in obj_info
        if object_id != "scene_camera"
    }
    pose_before = {
        object_id: np.asarray(root.matrix_world, dtype=np.float64)
        for object_id, root in roots.items()
        if root is not None
    }
    meshes = {}
    mesh_errors = {}
    for object_id, root in roots.items():
        if root is None:
            mesh_errors[object_id] = "missing_blender_object"
            continue
        try:
            mesh = _sceneproof_world_trimesh(root)
            repaired = mesh.copy()
            if repaired.is_watertight and not repaired.is_winding_consistent:
                trimesh.repair.fix_normals(repaired)
            if repaired.is_watertight and float(repaired.volume) < 0:
                repaired.invert()
            meshes[object_id] = repaired
        except (TypeError, ValueError, RuntimeError) as error:
            mesh_errors[object_id] = str(error)

    voxel_bodies = {}
    voxel_errors = {}

    def voxel_body(object_id):
        if object_id in voxel_bodies:
            return voxel_bodies[object_id]
        if object_id in voxel_errors:
            raise ValueError(voxel_errors[object_id])
        try:
            raw = _sceneproof_voxel_mass_properties(meshes[object_id])
            body = dict(raw)
            body["center_mass"] = np.asarray(raw["center_mass"], dtype=np.float64)
            body["occupied_points"] = np.asarray(
                raw["occupied_points"], dtype=np.float64
            )
            voxel_bodies[object_id] = body
            return body
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            voxel_errors[object_id] = str(error)
            raise ValueError(str(error)) from error

    def public_voxel_audit(body):
        return {
            key: value
            for key, value in body.items()
            if key not in {"center_mass", "volume", "occupied_points"}
        }

    program_ids = _sceneproof_support_program_ids(placement_document)
    records = {}
    target_ids = (
        set(scoped_object_ids) if scoped_object_ids else None
    )
    for child_id, info in obj_info.items():
        if target_ids is not None and child_id not in target_ids:
            continue
        parent_id = info.get("supported") if isinstance(info, dict) else None
        spatial = info.get("SpatialRel") if isinstance(info, dict) else None
        if (
            child_id == "scene_camera"
            or not isinstance(parent_id, str)
            or spatial == "inside"
            or re.match(r"^(wall|ceiling)_\d+$", parent_id)
        ):
            continue
        record = {
            "object_id": child_id,
            "declared_parent_id": parent_id,
            "support_program_ids": program_ids.get(child_id, []),
            "pose_matrix_for_blender": info.get("pose_matrix_for_blender"),
            "status": "abstained",
            "mutates_placement": False,
        }
        child = meshes.get(child_id)
        if child is None:
            record["reason"] = mesh_errors.get(child_id, "missing_child_mesh")
            records[child_id] = record
            continue
        mass_audit = {
            "mesh_watertight": bool(child.is_watertight),
            "mesh_is_volume": bool(child.is_volume),
        }
        if child.is_watertight and child.is_volume:
            center_mass = np.asarray(child.center_mass, dtype=np.float64)
            volume = abs(float(child.volume))
            mass_method = "exact_closed_mesh_uniform_density"
        else:
            try:
                voxel_mass = voxel_body(child_id)
            except (TypeError, ValueError, RuntimeError) as error:
                record.update(
                    {
                        "reason": "filled_voxel_mass_properties_unproven",
                        **mass_audit,
                        "error": str(error),
                    }
                )
                records[child_id] = record
                continue
            center_mass = voxel_mass["center_mass"]
            volume = float(voxel_mass["volume"])
            mass_method = "true_mesh_filled_voxel_uniform_density"
            mass_audit["voxel_mass_properties"] = public_voxel_audit(
                voxel_mass
            )
        if (
            center_mass.shape != (3,)
            or not np.isfinite(center_mass).all()
            or not math.isfinite(volume)
            or volume <= 1e-12
        ):
            record["reason"] = "child_mesh_mass_properties_nonfinite"
            records[child_id] = record
            continue

        triangles = np.asarray(child.triangles, dtype=np.float64)
        normals = np.asarray(child.face_normals, dtype=np.float64)
        downward = normals[:, 2] <= -normal_cosine
        child_contact_hull = None
        child_contact_z = float(child.bounds[0, 2])
        if downward.any():
            downward_triangles = triangles[downward]
            downward_centers = downward_triangles.mean(axis=1)[:, 2]
            child_contact_z = float(downward_centers.min())
            child_contact_triangles = downward_triangles[
                downward_centers <= child_contact_z + surface_band_m
            ]
            try:
                child_contact_hull = convex_hull_2d(
                    child_contact_triangles[:, :, :2].reshape(-1, 2)
                )
            except ValueError:
                child_contact_hull = None

        contact_polygons = []
        supporters = []
        contact_gaps = {}
        contact_methods = {}
        declared_child_contact_polygon = None
        declared_parent_surface_polygon = None
        declared_parent_surface_method = None
        declared_parent_surface_audit = None
        child_bounds = np.asarray(child.bounds, dtype=np.float64)
        for supporter_id, supporter in meshes.items():
            if supporter_id == child_id or re.match(
                r"^(wall|ceiling)_\d+$", supporter_id
            ):
                continue
            supporter_bounds = np.asarray(supporter.bounds, dtype=np.float64)
            xy_overlap = np.all(
                np.minimum(child_bounds[1, :2], supporter_bounds[1, :2])
                >= np.maximum(child_bounds[0, :2], supporter_bounds[0, :2])
            )
            # A supporter may contain a tall headboard/backrest. Its global
            # maximum Z is not the local surface under the child. Only require
            # the child's bottom plane to intersect the supporter's vertical
            # extent; the local face/heightfield test determines the gap.
            child_bottom_z = float(child_bounds[0, 2])
            vertical_extent_relevant = (
                float(supporter_bounds[0, 2]) - contact_tolerance_m
                <= child_bottom_z
                <= float(supporter_bounds[1, 2]) + contact_tolerance_m
            )
            if not xy_overlap or not vertical_extent_relevant:
                continue
            if child_contact_hull is None:
                continue
            supporter_triangles = np.asarray(
                supporter.triangles, dtype=np.float64
            )
            supporter_normals = np.asarray(
                supporter.face_normals, dtype=np.float64
            )
            upward = supporter_normals[:, 2] >= normal_cosine
            if not upward.any():
                continue
            upward_triangles = supporter_triangles[upward]
            upward_centers = upward_triangles.mean(axis=1)[:, 2]
            closest_index = int(
                np.argmin(np.abs(upward_centers - child_contact_z))
            )
            gap = abs(float(upward_centers[closest_index] - child_contact_z))
            if gap > contact_tolerance_m:
                continue
            support_z = float(upward_centers[closest_index])
            local_faces = upward_triangles[
                np.abs(upward_centers - support_z) <= surface_band_m
            ]
            try:
                supporter_hull = convex_hull_2d(
                    local_faces[:, :, :2].reshape(-1, 2)
                )
                contact = convex_polygon_intersection(
                    child_contact_hull, supporter_hull
                )
            except ValueError:
                continue
            if len(contact) < 3 or polygon_area(contact) <= 1e-10:
                continue
            supporters.append(supporter_id)
            contact_gaps[supporter_id] = gap
            contact_methods[supporter_id] = "exact_horizontal_mesh_faces"
            contact_polygons.append(contact)
            if supporter_id == parent_id:
                declared_child_contact_polygon = child_contact_hull.copy()
                declared_parent_surface_polygon = supporter_hull.copy()
                declared_parent_surface_method = "exact_horizontal_mesh_faces"

        # Curved/open assets (pillows, bowls, soft furnishings) commonly have
        # no near-horizontal triangles.  Their real evaluated mesh is already
        # voxelized for mass, so use its bottom heightfield against each
        # supporter's top heightfield as a discrete contact witness.
        if not contact_polygons:
            try:
                child_voxels = voxel_body(child_id)
            except ValueError:
                child_voxels = None
            if child_voxels is not None:
                for supporter_id, supporter in meshes.items():
                    if supporter_id == child_id or re.match(
                        r"^(wall|ceiling)_\d+$", supporter_id
                    ):
                        continue
                    supporter_bounds = np.asarray(
                        supporter.bounds, dtype=np.float64
                    )
                    xy_overlap = np.all(
                        np.minimum(
                            child_bounds[1, :2], supporter_bounds[1, :2]
                        )
                        >= np.maximum(
                            child_bounds[0, :2], supporter_bounds[0, :2]
                        )
                    )
                    child_bottom_z = float(child_bounds[0, 2])
                    vertical_extent_relevant = (
                        float(supporter_bounds[0, 2]) - contact_tolerance_m
                        <= child_bottom_z
                        <= float(supporter_bounds[1, 2])
                        + contact_tolerance_m
                    )
                    if (
                        not xy_overlap
                        or not vertical_extent_relevant
                    ):
                        continue
                    try:
                        supporter_voxels = voxel_body(supporter_id)
                        pitch = max(
                            float(child_voxels["pitch_m"]),
                            float(supporter_voxels["pitch_m"]),
                        )
                        points, heightfield_audit = (
                            voxel_heightfield_contact_points(
                                child_voxels["occupied_points"],
                                supporter_voxels["occupied_points"],
                                grid_pitch_m=pitch,
                                contact_tolerance_m=contact_tolerance_m,
                            )
                        )
                        contact = convex_hull_2d(points)
                    except (TypeError, ValueError, RuntimeError):
                        continue
                    if len(contact) < 3 or polygon_area(contact) <= 1e-10:
                        continue
                    supporters.append(supporter_id)
                    contact_gaps[supporter_id] = max(
                        abs(float(heightfield_audit["minimum_gap_m"])),
                        abs(float(heightfield_audit["maximum_gap_m"])),
                    )
                    contact_methods[supporter_id] = (
                        "true_mesh_voxel_heightfield"
                    )
                    contact_polygons.append(contact)

        # A missing child contact patch must not erase knowledge of the real
        # declared-parent surface.  Recover the connected parent top-surface
        # voxel component nearest the child's COM at the child's current bottom
        # height.  This is a candidate witness only; a repair must subsequently
        # establish actual child-parent contact before it can be certified.
        if declared_parent_surface_polygon is None and parent_id in meshes:
            try:
                parent_voxels = voxel_body(parent_id)
                declared_parent_surface_polygon, declared_parent_surface_audit = (
                    voxel_top_surface_component(
                        parent_voxels["occupied_points"],
                        query_xy=center_mass[:2],
                        reference_z_m=float(child_bounds[0, 2]),
                        grid_pitch_m=float(parent_voxels["pitch_m"]),
                        height_tolerance_m=contact_tolerance_m,
                    )
                )
                declared_parent_surface_method = (
                    "true_mesh_voxel_top_surface_component"
                )
            except (KeyError, TypeError, ValueError, RuntimeError):
                declared_parent_surface_polygon = None
                declared_parent_surface_audit = None

        if not contact_polygons:
            vertical_drop_candidates = []
            try:
                child_drop_voxels = voxel_body(child_id)
            except ValueError:
                child_drop_voxels = None
            if child_drop_voxels is not None:
                for supporter_id in meshes:
                    if supporter_id == child_id:
                        continue
                    supporter_info = obj_info.get(supporter_id, {})
                    same_support_component = bool(
                        supporter_id == parent_id
                        or (
                            isinstance(supporter_info, dict)
                            and supporter_info.get("supported") == parent_id
                        )
                    )
                    if not same_support_component:
                        continue
                    try:
                        supporter_drop_voxels = voxel_body(supporter_id)
                        pitch = max(
                            float(child_drop_voxels["pitch_m"]),
                            float(supporter_drop_voxels["pitch_m"]),
                        )
                        drop_m, patch, drop_audit = voxel_vertical_first_contact(
                            child_drop_voxels["occupied_points"],
                            supporter_drop_voxels["occupied_points"],
                            grid_pitch_m=pitch,
                            maximum_drop_m=0.5,
                            penetration_tolerance_m=max(0.005, pitch),
                            contact_band_m=max(surface_band_m, 1.5 * pitch),
                        )
                    except (KeyError, TypeError, ValueError, RuntimeError) as error:
                        vertical_drop_candidates.append(
                            {
                                "supporter_id": supporter_id,
                                "status": "abstained",
                                "reason": str(error),
                            }
                        )
                        continue
                    vertical_drop_candidates.append(
                        {
                            "supporter_id": supporter_id,
                            "status": "measured",
                            "drop_m": float(drop_m),
                            "contact_polygon_world_xy_m": patch.tolist(),
                            "audit": drop_audit,
                        }
                    )
            measured_drops = [
                candidate
                for candidate in vertical_drop_candidates
                if candidate.get("status") == "measured"
            ]
            vertical_drop = (
                min(measured_drops, key=lambda candidate: candidate["drop_m"])
                if measured_drops
                else None
            )
            parent_surface_margin = None
            if declared_parent_surface_polygon is not None:
                try:
                    parent_surface_margin = signed_margin_to_convex_polygon(
                        center_mass[:2], declared_parent_surface_polygon
                    )
                except ValueError:
                    parent_surface_margin = None
            record.update(
                {
                    "reason": "no_mesh_or_voxel_horizontal_contact_patch",
                    "center_of_mass_world_m": center_mass.tolist(),
                    "mesh_volume_m3": volume,
                    "mass_property_method": mass_method,
                    "declared_parent_contact_present": False,
                    "declared_parent_surface_margin_m": parent_surface_margin,
                    "declared_parent_surface_polygon_world_xy_m": (
                        declared_parent_surface_polygon.tolist()
                        if declared_parent_surface_polygon is not None
                        else None
                    ),
                    "declared_parent_surface_method": declared_parent_surface_method,
                    "declared_parent_surface_audit": declared_parent_surface_audit,
                    "vertical_first_contact_candidate": vertical_drop,
                    "vertical_first_contact_candidate_audit": vertical_drop_candidates,
                    "vertical_first_contact_policy": {
                        "xy_frozen": True,
                        "so3_frozen": True,
                        "maximum_drop_m": 0.5,
                        "supporters": "declared_parent_or_same_parent_sibling",
                        "mutates_placement": False,
                    },
                    **mass_audit,
                }
            )
            records[child_id] = record
            continue
        try:
            support_hull = convex_hull_2d(
                np.concatenate(contact_polygons, axis=0)
            )
            margin = signed_margin_to_convex_polygon(
                center_mass[:2], support_hull
            )
        except ValueError as error:
            record["reason"] = "degenerate_combined_support_region"
            record["error"] = str(error)
            records[child_id] = record
            continue
        intrinsic_contact_margin = None
        parent_surface_margin = None
        if declared_child_contact_polygon is not None:
            try:
                intrinsic_contact_margin = signed_margin_to_convex_polygon(
                    center_mass[:2], declared_child_contact_polygon
                )
            except ValueError:
                intrinsic_contact_margin = None
        if declared_parent_surface_polygon is not None:
            try:
                parent_surface_margin = signed_margin_to_convex_polygon(
                    center_mass[:2], declared_parent_surface_polygon
                )
            except ValueError:
                parent_surface_margin = None
        record.update(
            {
                "status": "measured",
                "reason": "true_mesh_horizontal_support_measured",
                "center_of_mass_world_m": center_mass.tolist(),
                "mesh_volume_m3": volume,
                "mass_property_method": mass_method,
                **mass_audit,
                "supporter_ids": sorted(supporters),
                "declared_parent_contact_present": parent_id in supporters,
                "contact_gap_by_supporter_m": contact_gaps,
                "contact_method_by_supporter": contact_methods,
                "support_polygon_world_xy_m": support_hull.tolist(),
                "support_polygon_area_m2": polygon_area(support_hull),
                "com_signed_margin_m": margin,
                "intrinsic_child_contact_margin_m": intrinsic_contact_margin,
                "declared_parent_surface_margin_m": parent_surface_margin,
                "intrinsic_child_contact_polygon_world_xy_m": (
                    declared_child_contact_polygon.tolist()
                    if declared_child_contact_polygon is not None
                    else None
                ),
                "declared_parent_surface_polygon_world_xy_m": (
                    declared_parent_surface_polygon.tolist()
                    if declared_parent_surface_polygon is not None
                    else None
                ),
                "declared_parent_surface_method": declared_parent_surface_method,
                "declared_parent_surface_audit": declared_parent_surface_audit,
                "stability_class": stability_class(
                    margin, stability_tolerance_m
                ),
                "stability_tolerance_m": stability_tolerance_m,
                "legacy_contact_is_not_stability_proof": True,
            }
        )
        records[child_id] = record

    # A cyclic support component has no grounded causal ordering.  Preserve
    # its measurements for diagnosis but never promote it to a certificate.
    adjacency = {}
    for object_id, row in records.items():
        if row.get("status") != "measured":
            continue
        edges = set(row.get("supporter_ids", []))
        declared_parent = row.get("declared_parent_id")
        if isinstance(declared_parent, str):
            edges.add(declared_parent)
        adjacency[object_id] = edges
    grounded_nodes = {
        object_id
        for object_id in obj_info
        if re.match(r"^(floor|ground)_\d+$", object_id)
    }
    cyclic_components, grounded_cyclic_components = (
        ungrounded_cyclic_components(adjacency, grounded_nodes)
    )
    for component in cyclic_components:
        for object_id in component:
            row = records[object_id]
            row["certificate_status"] = "abstained"
            row["reason"] = "cyclic_support_component_unproven"
            row["support_cycle_object_ids"] = component
            row["support_cycle_edge_policy"] = (
                "authoritative_declared_parent_plus_measured_contact"
            )
    for object_id, row in records.items():
        if row.get("status") == "measured" and "certificate_status" not in row:
            row["certificate_status"] = "certified"

    bpy.context.view_layer.update()
    pose_after = {
        object_id: np.asarray(root.matrix_world, dtype=np.float64)
        for object_id, root in roots.items()
        if root is not None
    }
    maximum_pose_delta = max(
        (
            float(np.max(np.abs(pose_after[object_id] - before)))
            for object_id, before in pose_before.items()
        ),
        default=0.0,
    )
    measured = [row for row in records.values() if row["status"] == "measured"]
    certified = [
        row for row in measured if row.get("certificate_status") == "certified"
    ]
    return {
        "schema_version": "sceneproof_true_mesh_com_support_audit_v1",
        "policy": "audit_only_true_mesh_uniform_density_horizontal_support",
        "mutates_placement": False,
        "maximum_pose_delta": maximum_pose_delta,
        "thresholds": {
            "contact_tolerance_m": contact_tolerance_m,
            "surface_band_m": surface_band_m,
            "normal_cosine": normal_cosine,
            "stability_tolerance_m": stability_tolerance_m,
        },
        "objects": records,
        "summary": {
            "eligible_support_objects": len(records),
            "measured": len(measured),
            "abstained": len(records) - len(measured),
            "certified": len(certified),
            "cyclic_support_abstained": sum(len(row) for row in cyclic_components),
            "cyclic_support_components": cyclic_components,
            "grounded_cyclic_support_components": grounded_cyclic_components,
            "stable": sum(row.get("stability_class") == "stable" for row in certified),
            "marginal": sum(row.get("stability_class") == "marginal" for row in certified),
            "unstable": sum(row.get("stability_class") == "unstable" for row in certified),
        },
    }


def audit_sceneproof_sequential_vertical_first_contact(
    placement_document,
    object_ids,
    *,
    source_placement_path,
    scene_camera,
    visibility_tolerance=0.005,
    visibility_resolution=256,
    tangent_projection_margin_m=0.005,
    tangent_projection_limit_m=0.15,
):
    """Apply first contact, then the minimum certified support-plane repair."""
    ordered_ids = [
        object_id
        for object_id in placement_document.get("obj_info", {})
        if object_id != "scene_camera"
    ]
    roots = {
        object_id: bpy.data.objects.get(object_id)
        for object_id in ordered_ids
        if bpy.data.objects.get(object_id) is not None
    }
    floor_id = placement_document.get("reference_obj")
    sibling_audit = {
        "component_audits": [{"object_ids": list(object_ids)}]
    }

    def visibility():
        return audit_sceneproof_mesh_visibility(
            source_placement_path,
            ordered_ids,
            scene_camera,
            sibling_audit,
            resolution=int(visibility_resolution),
            minimum_pixels=1,
        )

    current_visibility = visibility()
    records = []
    accepted_ids = []
    for object_id in object_ids:
        root = roots.get(object_id)
        transaction = {
            "object_id": object_id,
            "accepted": False,
            "xy_frozen": True,
            "so3_frozen": True,
            "horizontal_policy": "frozen_unless_minimum_com_support_projection",
        }
        if root is None:
            transaction["reason"] = "missing_blender_root"
            records.append(transaction)
            continue
        before_matrix = np.asarray(root.matrix_world, dtype=np.float64)
        before_support = audit_sceneproof_true_mesh_com_support(
            placement_document, scoped_object_ids=[object_id]
        ).get("objects", {}).get(object_id, {})
        drop = before_support.get("vertical_first_contact_candidate")
        if not isinstance(drop, dict):
            transaction["reason"] = "no_vertical_first_contact_candidate"
            transaction["before_support"] = before_support
            records.append(transaction)
            continue
        drop_m = float(drop.get("drop_m", math.nan))
        supporter_id = drop.get("supporter_id")
        transaction.update(
            {
                "drop_m": drop_m,
                "supporter_id": supporter_id,
                "before_support": before_support,
            }
        )
        if not math.isfinite(drop_m) or not 0 < drop_m <= 0.5:
            transaction["reason"] = "drop_outside_trust_region"
            records.append(transaction)
            continue

        before_cache = {
            candidate_id: _sceneproof_evaluated_bvh(candidate_root)
            for candidate_id, candidate_root in roots.items()
            if candidate_id != object_id
        }
        before_counts, before_abstained = _sceneproof_overlap_triangle_pair_counts(
            object_id, roots, bvh_cache=before_cache
        )
        before_boundary = None
        if isinstance(floor_id, str) and floor_id in roots:
            before_boundary, _ = _sceneproof_true_mesh_boundary_error(
                root, roots[floor_id]
            )

        candidate_matrix = before_matrix.copy()
        candidate_matrix[2, 3] -= drop_m
        root.matrix_world = Matrix(candidate_matrix.tolist())
        bpy.context.view_layer.update()
        contact_support = audit_sceneproof_true_mesh_com_support(
            placement_document, scoped_object_ids=[object_id]
        ).get("objects", {}).get(object_id, {})
        tangent_delta = np.zeros(2, dtype=np.float64)
        tangent_reason = "not_required"
        if (
            contact_support.get("certificate_status") == "certified"
            and contact_support.get("stability_class") == "unstable"
        ):
            polygon = contact_support.get(
                "declared_parent_surface_polygon_world_xy_m"
            )
            center = contact_support.get("center_of_mass_world_m")
            try:
                tangent_delta = minimum_com_translation_into_support(
                    np.asarray(center, dtype=np.float64)[:2],
                    np.asarray(polygon, dtype=np.float64),
                    margin_m=float(tangent_projection_margin_m),
                )
                tangent_distance = float(np.linalg.norm(tangent_delta))
                if tangent_distance <= float(tangent_projection_limit_m) + 1e-12:
                    candidate_matrix[:2, 3] += tangent_delta
                    root.matrix_world = Matrix(candidate_matrix.tolist())
                    bpy.context.view_layer.update()
                    tangent_reason = "minimum_com_support_projection_applied"
                else:
                    tangent_delta = np.zeros(2, dtype=np.float64)
                    tangent_reason = "minimum_projection_outside_trust_region"
            except (TypeError, ValueError, IndexError):
                tangent_delta = np.zeros(2, dtype=np.float64)
                tangent_reason = "minimum_projection_unavailable"
        after_support = audit_sceneproof_true_mesh_com_support(
            placement_document, scoped_object_ids=[object_id]
        ).get("objects", {}).get(object_id, {})
        after_cache = {
            candidate_id: _sceneproof_evaluated_bvh(candidate_root)
            for candidate_id, candidate_root in roots.items()
            if candidate_id != object_id
        }
        after_counts, after_abstained = _sceneproof_overlap_triangle_pair_counts(
            object_id, roots, bvh_cache=after_cache
        )
        overlap_regressions = {
            candidate_id: {
                "before": int(before_counts.get(candidate_id, 0)),
                "after": int(after_count),
            }
            for candidate_id, after_count in after_counts.items()
            if int(after_count) > int(before_counts.get(candidate_id, 0))
        }
        after_boundary = None
        if isinstance(floor_id, str) and floor_id in roots:
            after_boundary, _ = _sceneproof_true_mesh_boundary_error(
                root, roots[floor_id]
            )
        after_visibility = visibility()
        before_visual = current_visibility.get("objects", {}).get(object_id, {})
        after_visual = after_visibility.get("objects", {}).get(object_id, {})
        visual_passed = bool(
            before_visual.get("status") == "measured"
            and after_visual.get("status") == "measured"
            and float(after_visual.get("iou", 0.0))
            >= float(before_visual.get("iou", 0.0)) - visibility_tolerance
            and float(after_visual.get("recall", 0.0))
            >= float(before_visual.get("recall", 0.0)) - visibility_tolerance
        )
        supporters = set(after_support.get("supporter_ids", []))
        support_passed = bool(
            after_support.get("certificate_status") == "certified"
            and after_support.get("stability_class") in {"stable", "marginal"}
            and supporter_id in supporters
        )
        boundary_passed = bool(
            before_boundary is None
            or after_boundary is None
            or after_boundary <= before_boundary + 1e-6
        )
        collision_passed = bool(
            not before_abstained
            and not after_abstained
            and not overlap_regressions
        )
        after_matrix = np.asarray(root.matrix_world, dtype=np.float64)
        xy_error = float(np.max(np.abs(after_matrix[:2, 3] - before_matrix[:2, 3])))
        expected_xy = before_matrix[:2, 3] + tangent_delta
        xy_policy_error = float(np.max(np.abs(after_matrix[:2, 3] - expected_xy)))
        xy_dtype_tolerance = float(
            max(
                1e-6,
                16.0
                * np.finfo(np.float32).eps
                * max(1.0, float(np.max(np.abs(expected_xy)))),
            )
        )
        rotation_error = float(np.max(np.abs(after_matrix[:3, :3] - before_matrix[:3, :3])))
        transaction.update(
            {
                "after_support": after_support,
                "first_contact_support": contact_support,
                "tangent_projection_delta_xy_m": tangent_delta.tolist(),
                "tangent_projection_norm_m": float(np.linalg.norm(tangent_delta)),
                "tangent_projection_reason": tangent_reason,
                "xy_policy_error_m": xy_policy_error,
                "xy_dtype_tolerance_m": xy_dtype_tolerance,
                "before_exact_overlap_triangle_pairs": before_counts,
                "after_exact_overlap_triangle_pairs": after_counts,
                "exact_overlap_regressions": overlap_regressions,
                "before_boundary_error_m": before_boundary,
                "after_boundary_error_m": after_boundary,
                "before_visibility": before_visual,
                "after_visibility": after_visual,
                "gates": {
                    "xy_frozen_or_certified_minimum_projection": bool(
                        xy_policy_error <= xy_dtype_tolerance
                        and xy_error
                        <= float(tangent_projection_limit_m) + xy_dtype_tolerance
                    ),
                    "so3_frozen": rotation_error <= 1e-7,
                    "support_certified_stable_or_marginal": support_passed,
                    "exact_overlap_nonincreasing": collision_passed,
                    "boundary_noninferior": boundary_passed,
                    "s1_mask_iou_recall_noninferior": visual_passed,
                },
            }
        )
        passed = all(transaction["gates"].values())
        transaction["accepted"] = bool(passed)
        transaction["reason"] = "committed" if passed else "gate_rejected_restored"
        if passed:
            accepted_ids.append(object_id)
            current_visibility = after_visibility
            placement_document["obj_info"][object_id][
                "pose_matrix_for_blender"
            ] = after_matrix.tolist()
        else:
            root.matrix_world = Matrix(before_matrix.tolist())
            bpy.context.view_layer.update()
        records.append(transaction)

    return {
        "schema_version": "sceneproof_sequential_vertical_first_contact_v1",
        "policy": "ordered_transactional_z_only_true_mesh_first_contact",
        "object_order": list(object_ids),
        "accepted_object_ids": accepted_ids,
        "transactions": records,
        "visibility_tolerance": float(visibility_tolerance),
        "mutates_placement_only_for_accepted_transactions": True,
    }


def _sceneproof_evaluated_bvh(root):
    """Build a world-space BVH from every evaluated mesh owned by one root."""
    from mathutils.bvhtree import BVHTree

    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices = []
    faces = []
    for mesh_object in _sceneproof_mesh_hierarchy(root):
        evaluated = mesh_object.evaluated_get(depsgraph)
        evaluated_mesh = evaluated.to_mesh()
        try:
            evaluated_mesh.calc_loop_triangles()
            offset = len(vertices)
            vertices.extend(
                evaluated.matrix_world @ vertex.co
                for vertex in evaluated_mesh.vertices
            )
            faces.extend(
                tuple(offset + index for index in triangle.vertices)
                for triangle in evaluated_mesh.loop_triangles
            )
        finally:
            evaluated.to_mesh_clear()
    if not vertices or not faces:
        raise ValueError(f"no evaluated mesh triangles for {root.name}")
    return BVHTree.FromPolygons(vertices, faces, all_triangles=True)


def _sceneproof_world_hierarchy_bound_points(root):
    """Return evaluated world-space bound corners for an object hierarchy."""
    points = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for mesh_object in _sceneproof_mesh_hierarchy(root):
        evaluated = mesh_object.evaluated_get(depsgraph)
        points.extend(
            np.asarray(evaluated.matrix_world @ Vector(corner), dtype=np.float64)
            for corner in evaluated.bound_box
        )
    if not points:
        raise ValueError(f"no evaluated world bounds for {root.name}")
    return np.asarray(points, dtype=np.float64)


def _sceneproof_world_hierarchy_bounds(root):
    """Return evaluated world AABB for one reconstructed object hierarchy."""
    points = _sceneproof_world_hierarchy_bound_points(root)
    return points.min(axis=0), points.max(axis=0)


def _sceneproof_dominant_upward_support_component(
    root, *, height_tolerance_m=0.03, normal_cosine=0.70
):
    """Select the broadest true-mesh upward support component.

    Faces are first grouped by mesh connectivity and near-equal height.  A
    height-layer fallback then merges disconnected coplanar pieces, which is
    important for assets whose exported top surface duplicates vertices.  The
    winner maximises projected XY area (then height), never merely elevation.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    connected_components = []
    all_faces = []
    for mesh_object in _sceneproof_mesh_hierarchy(root):
        evaluated = mesh_object.evaluated_get(depsgraph)
        evaluated_mesh = evaluated.to_mesh()
        try:
            evaluated_mesh.calc_loop_triangles()
            faces = []
            vertex_to_faces = defaultdict(list)
            for triangle in evaluated_mesh.loop_triangles:
                world = np.asarray(
                    [
                        evaluated.matrix_world @ evaluated_mesh.vertices[index].co
                        for index in triangle.vertices
                    ],
                    dtype=np.float64,
                )
                cross = np.cross(world[1] - world[0], world[2] - world[0])
                norm = float(np.linalg.norm(cross))
                if norm <= 1e-12 or float(cross[2]) / norm < normal_cosine:
                    continue
                area_xy = abs(float(cross[2])) * 0.5
                if area_xy <= 1e-10:
                    continue
                face_index = len(faces)
                face = {
                    "z": float(world[:, 2].mean()),
                    "area_xy": area_xy,
                    "vertices": tuple(int(value) for value in triangle.vertices),
                }
                faces.append(face)
                all_faces.append(face)
                for vertex_index in face["vertices"]:
                    vertex_to_faces[vertex_index].append(face_index)
            if not faces:
                continue
            parent = list(range(len(faces)))

            def find(index):
                while parent[index] != index:
                    parent[index] = parent[parent[index]]
                    index = parent[index]
                return index

            def union(first, second):
                first_root, second_root = find(first), find(second)
                if first_root != second_root:
                    parent[second_root] = first_root

            for indices in vertex_to_faces.values():
                for first in indices:
                    for second in indices:
                        if abs(faces[first]["z"] - faces[second]["z"]) <= height_tolerance_m:
                            union(first, second)
            groups = defaultdict(list)
            for index, face in enumerate(faces):
                groups[find(index)].append(face)
            for component_faces in groups.values():
                area = sum(face["area_xy"] for face in component_faces)
                height = sum(
                    face["z"] * face["area_xy"] for face in component_faces
                ) / max(area, 1e-12)
                connected_components.append({
                    "height_m": float(height),
                    "projected_area_m2": float(area),
                    "face_count": len(component_faces),
                    "source": "mesh_connected_component",
                })
        finally:
            evaluated.to_mesh_clear()
    if not all_faces:
        raise ValueError(f"no upward support faces for {root.name}")

    # Robust fallback for split/duplicated topology: merge faces into height
    # layers and allow the broadest layer to compete with connected components.
    height_layers = []
    for face in sorted(all_faces, key=lambda item: item["z"]):
        match = next(
            (
                layer for layer in height_layers
                if abs(layer["weighted_z"] / layer["area"] - face["z"])
                <= height_tolerance_m
            ),
            None,
        )
        if match is None:
            match = {"area": 0.0, "weighted_z": 0.0, "face_count": 0}
            height_layers.append(match)
        match["area"] += face["area_xy"]
        match["weighted_z"] += face["z"] * face["area_xy"]
        match["face_count"] += 1
    candidates = list(connected_components)
    for layer in height_layers:
        candidates.append({
            "height_m": float(layer["weighted_z"] / max(layer["area"], 1e-12)),
            "projected_area_m2": float(layer["area"]),
            "face_count": int(layer["face_count"]),
            "source": "coplanar_height_component",
        })
    winner = max(
        candidates,
        key=lambda item: (item["projected_area_m2"], item["height_m"]),
    )
    return {
        **winner,
        "candidate_component_count": len(candidates),
        "height_tolerance_m": float(height_tolerance_m),
        "normal_cosine": float(normal_cosine),
    }


def _sceneproof_sparse_aabb_overlaps(target_id, roots, bounds_cache, epsilon=0.002):
    """Cheap broad-phase witnesses using reconstructed Blender world bounds."""
    target_min, target_max = _sceneproof_world_hierarchy_bounds(roots[target_id])
    overlaps = []
    for object_id, root in roots.items():
        if object_id == target_id or root is None:
            continue
        try:
            other_min, other_max = bounds_cache.get(object_id, (None, None))
            if other_min is None:
                other_min, other_max = _sceneproof_world_hierarchy_bounds(root)
                bounds_cache[object_id] = (other_min, other_max)
        except (TypeError, ValueError, RuntimeError):
            continue
        extent = np.minimum(target_max, other_max) - np.maximum(target_min, other_min)
        if bool(np.all(extent > float(epsilon))):
            overlaps.append(object_id)
    return sorted(overlaps)


def _sceneproof_plane_attach_pairs(document):
    """Extract explicit object/structural-plane bindings from Relation Programs."""
    def canonical(value):
        if isinstance(value, str) and value.startswith("architecture:"):
            return value.split(":", 1)[1]
        return value

    pairs = set()
    bundle = document.get("sceneproof_relation_programs", {})
    for program in bundle.get("programs", []):
        if program.get("kind") != "PLANE_ATTACH":
            continue
        ids = [
            canonical(participant.get("object_id"))
            for participant in program.get("participants", [])
            if isinstance(participant, dict)
            and isinstance(participant.get("object_id"), str)
        ]
        planes = [value for value in ids if re.match(r"^(wall|ceiling)_\d+$", value)]
        objects = [value for value in ids if value not in planes]
        pairs.update((object_id, plane_id) for object_id in objects for plane_id in planes)
    return pairs


def apply_sceneproof_sparse_vertical_contact(
    placement_document,
    *,
    rollback_document=None,
    contact_tolerance_m=0.02,
    maximum_shift_m=0.5,
    minimum_hit_fraction=0.10,
    maximum_tangent_shift_m=0.15,
    maximum_program_tangent_shift_m=0.50,
):
    """Repair obvious support gaps/penetrations with sparse true-geometry rays.

    This deliberately is not a full collision or COM proof.  It is a cheap,
    version-agnostic final certificate shared by Fix61 and Fix114.  Every
    accepted vertical edit freezes XY/SO(3)/scale.  A failed declared-parent
    witness may first use a bounded tangent-only correction, then a Z-only
    first-contact drop onto another real upward surface.  Invalid vertical
    support chains are restored from the upstream incumbent.  Every edit is
    rolled back on a new reconstructed-world AABB overlap.  Wall and ceiling
    attachments are never gravity-dropped: they need
    an explicit PLANE_ATTACH program in addition to the legacy structural parent
    signal, otherwise they are reported unresolved.
    """
    obj_info = placement_document.get("obj_info", {})
    roots = {
        object_id: bpy.data.objects.get(object_id)
        for object_id in obj_info
        if object_id != "scene_camera"
    }
    plane_pairs = _sceneproof_plane_attach_pairs(placement_document)
    support_program_ids = _sceneproof_support_program_ids(placement_document)
    rollback_info = (
        rollback_document.get("obj_info", {})
        if isinstance(rollback_document, dict)
        else {}
    )
    bounds_cache = {}
    bvh_cache = {}
    support_component_cache = {}

    def depth(object_id, visiting=None):
        visiting = set() if visiting is None else set(visiting)
        if object_id in visiting:
            return 10_000
        visiting.add(object_id)
        row = obj_info.get(object_id, {})
        parent_id = row.get("supported") if isinstance(row, dict) else None
        if not isinstance(parent_id, str) or parent_id not in obj_info:
            return 0
        return 1 + depth(parent_id, visiting)

    candidates = []
    for object_id, row in obj_info.items():
        if object_id == "scene_camera" or not isinstance(row, dict):
            continue
        parent_id = row.get("supported")
        if not isinstance(parent_id, str) or row.get("SpatialRel") == "inside":
            continue
        try:
            candidate_root = roots.get(object_id)
            if candidate_root is None:
                raise ValueError("missing candidate root")
            bottom_z = float(_sceneproof_world_hierarchy_bounds(candidate_root)[0][2])
        except (TypeError, ValueError, RuntimeError):
            bottom_z = float("inf")
        candidates.append((depth(object_id), bottom_z, object_id, parent_id))

    records = []
    repaired = []
    unresolved = []
    held = []
    for _, _, object_id, parent_id in sorted(candidates):
        row = obj_info[object_id]
        root = roots.get(object_id)
        parent = roots.get(parent_id)
        record = {
            "object_id": object_id,
            "declared_parent_id": parent_id,
            "status": "unresolved",
            "mutates_xy": False,
            "mutates_so3": False,
            "mutates_scale": False,
        }
        structural = re.match(r"^(wall|ceiling)_\d+$", parent_id)
        if structural:
            against_wall = row.get("againstWall")
            if isinstance(against_wall, str):
                legacy_values = {against_wall}
            elif isinstance(against_wall, (list, tuple, set)):
                legacy_values = {
                    value for value in against_wall if isinstance(value, str)
                }
            else:
                legacy_values = set()
            legacy_signal = parent_id in legacy_values or row.get("supported") == parent_id
            program_signal = (object_id, parent_id) in plane_pairs
            record.update({
                "legacy_structural_signal": bool(legacy_signal),
                "plane_attach_program_signal": bool(program_signal),
            })
            if legacy_signal and program_signal:
                record.update(status="held", reason="double_witnessed_structural_attachment")
                held.append(object_id)
            else:
                record["reason"] = "structural_attachment_witness_incomplete"
                unresolved.append(object_id)
            records.append(record)
            continue
        if root is None or parent is None:
            record["reason"] = "missing_reconstructed_child_or_parent"
            unresolved.append(object_id)
            records.append(record)
            continue

        # A vertical, wall-attached slab is not a gravity support surface for
        # an ordinary child.  This catches category-free false chains such as
        # an image/window component declared as resting on a picture frame,
        # while preserving horizontal wall shelves and floor furniture.
        parent_row = obj_info.get(parent_id, {})
        parent_structural_ids = set()
        parent_supported = parent_row.get("supported") if isinstance(parent_row, dict) else None
        if isinstance(parent_supported, str) and re.match(
            r"^(wall|ceiling)_\d+$", parent_supported
        ):
            parent_structural_ids.add(parent_supported)
        parent_against = parent_row.get("againstWall") if isinstance(parent_row, dict) else None
        if isinstance(parent_against, str):
            parent_structural_ids.add(parent_against)
        elif isinstance(parent_against, (list, tuple, set)):
            parent_structural_ids.update(
                value for value in parent_against if isinstance(value, str)
            )
        witnessed_parent_planes = {
            plane_id
            for candidate_id, plane_id in plane_pairs
            if candidate_id == parent_id and plane_id in parent_structural_ids
        }
        try:
            parent_points = _sceneproof_world_hierarchy_bound_points(parent)
            parent_min, parent_max = parent_points.min(axis=0), parent_points.max(axis=0)
            parent_size = np.maximum(parent_max - parent_min, 1e-9)
            centered = parent_points - parent_points.mean(axis=0)
            covariance = centered.T @ centered / max(len(centered), 1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)
            thin_direction = eigenvectors[:, order[0]]
            major_direction = eigenvectors[:, order[-1]]
            thin_scale = math.sqrt(max(float(eigenvalues[order[0]]), 1e-12))
            major_scale = math.sqrt(max(float(eigenvalues[order[-1]]), 1e-12))
            vertical_slab = bool(
                thin_scale / max(major_scale, 1e-9) <= 0.15
                and abs(float(thin_direction[2])) <= 0.35
                and abs(float(major_direction[2])) >= 0.65
            )
        except (TypeError, ValueError, RuntimeError):
            vertical_slab = False
            parent_size = None
        child_plane_signal = any(
            candidate_id == object_id for candidate_id, _ in plane_pairs
        )
        if witnessed_parent_planes and vertical_slab and not child_plane_signal:
            rollback_row = rollback_info.get(object_id, {})
            rollback_pose = (
                rollback_row.get("pose_matrix_for_blender")
                if isinstance(rollback_row, dict)
                else None
            )
            restored = False
            if rollback_pose is not None:
                try:
                    root.matrix_world = Matrix(rollback_pose)
                    bpy.context.view_layer.update()
                    row["pose_matrix_for_blender"] = [
                        list(matrix_row) for matrix_row in root.matrix_world
                    ]
                    bounds_cache.clear()
                    bvh_cache.clear()
                    support_component_cache.clear()
                    restored = True
                except (TypeError, ValueError, RuntimeError):
                    restored = False
            record.update(
                reason="invalid_vertical_parent_support_chain",
                rollback_restored=restored,
                parent_structural_planes=sorted(witnessed_parent_planes),
                parent_world_size_m=(
                    [float(value) for value in parent_size]
                    if parent_size is not None else None
                ),
                parent_principal_thin_major_ratio=float(
                    thin_scale / max(major_scale, 1e-9)
                ),
                parent_thin_axis_world_z=float(thin_direction[2]),
            )
            unresolved.append(object_id)
            records.append(record)
            continue
        try:
            child_min, child_max = _sceneproof_world_hierarchy_bounds(root)
            bounds_cache[object_id] = (child_min, child_max)
            parent_min, parent_max = _sceneproof_world_hierarchy_bounds(parent)
            bounds_cache[parent_id] = (parent_min, parent_max)
            parent_tree = bvh_cache.get(parent_id)
            if parent_tree is None:
                parent_tree = _sceneproof_evaluated_bvh(parent)
                bvh_cache[parent_id] = parent_tree
            dominant_component = support_component_cache.get(parent_id)
            if dominant_component is None:
                dominant_component = _sceneproof_dominant_upward_support_component(
                    parent,
                    height_tolerance_m=max(contact_tolerance_m, 0.03),
                )
                support_component_cache[parent_id] = dominant_component
        except (TypeError, ValueError, RuntimeError) as error:
            record.update(reason="reconstructed_geometry_unavailable", error=str(error))
            unresolved.append(object_id)
            records.append(record)
            continue

        # Nine inset footprint witnesses plus the center.  Ray casting against
        # only the declared parent avoids the invalid global-top assumption that
        # caused Fix135's bed/desk/wardrobe failures.
        x0, y0 = child_min[:2]
        x1, y1 = child_max[:2]
        xs = np.linspace(x0, x1, 5)[1:-1]
        ys = np.linspace(y0, y1, 5)[1:-1]
        sample_xy = [(float(x), float(y)) for x in xs for y in ys]
        sample_xy.append((float((x0 + x1) / 2), float((y0 + y1) / 2)))
        support_band = max(float(contact_tolerance_m), 0.03)
        ray_z = float(dominant_component["height_m"] + support_band)
        ray_distance = float(2.0 * support_band)
        record["dominant_parent_support_component"] = dominant_component
        def ray_hits(tree, points, origin_z=ray_z, distance=ray_distance):
            values = []
            for x, y in points:
                location, normal, _, _ = tree.ray_cast(
                    Vector((x, y, origin_z)),
                    Vector((0.0, 0.0, -1.0)),
                    distance,
                )
                if location is None or normal is None or float(normal.z) < 0.5:
                    continue
                if float(location.z) <= origin_z + contact_tolerance_m:
                    values.append(float(location.z))
            return values

        hits = ray_hits(parent_tree, sample_xy)
        hit_fraction = len(hits) / len(sample_xy)
        record.update(
            sparse_ray_count=len(sample_xy),
            sparse_hit_count=len(hits),
            sparse_hit_fraction=float(hit_fraction),
        )
        if not hits or hit_fraction < minimum_hit_fraction:
            # First try the minimum bounded tangent translation that restores
            # real contact with the declared parent.  Height and SO(3) remain
            # frozen.  Deterministic radii and directions make the result
            # reproducible and cap the search at 15 cm by default.
            tangent_candidates = []
            parent_center = (parent_min[:2] + parent_max[:2]) / 2.0
            child_center = (child_min[:2] + child_max[:2]) / 2.0
            toward = parent_center - child_center
            toward_norm = float(np.linalg.norm(toward))
            angles = [2.0 * math.pi * index / 8.0 for index in range(8)]
            directions = [np.asarray((math.cos(a), math.sin(a))) for a in angles]
            if toward_norm > 1e-9:
                directions.append(toward / toward_norm)
            tangent_limit = float(maximum_tangent_shift_m)
            extended_program_search = bool(support_program_ids.get(object_id))
            if extended_program_search:
                tangent_limit = max(
                    tangent_limit, float(maximum_program_tangent_shift_m)
                )
            tangent_steps = max(5, int(math.ceil(tangent_limit / 0.05)))
            tangent_radii = np.linspace(
                tangent_limit / tangent_steps, tangent_limit, tangent_steps
            )
            for radius in tangent_radii:
                for direction in directions:
                    delta = np.asarray(direction, dtype=np.float64) * float(radius)
                    shifted_points = [
                        (x + float(delta[0]), y + float(delta[1]))
                        for x, y in sample_xy
                    ]
                    candidate_hits = ray_hits(parent_tree, shifted_points)
                    fraction = len(candidate_hits) / len(sample_xy)
                    if candidate_hits and fraction >= minimum_hit_fraction:
                        support = max(candidate_hits)
                        gap = float(child_min[2] - support)
                        if -contact_tolerance_m <= gap <= maximum_shift_m:
                            tangent_candidates.append((
                                float(np.linalg.norm(delta)), delta,
                                candidate_hits, fraction, gap, support,
                            ))
            tangent_candidates.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
            tangent_accepted = False
            if tangent_candidates:
                (
                    tangent_norm, tangent_delta, tangent_hits,
                    tangent_fraction, tangent_gap, tangent_support_z,
                ) = tangent_candidates[0]
                before_matrix = root.matrix_world.copy()
                before_overlaps = set(
                    _sceneproof_sparse_aabb_overlaps(object_id, roots, bounds_cache)
                ) - {parent_id}
                root.matrix_world[0][3] = float(root.matrix_world[0][3] + tangent_delta[0])
                root.matrix_world[1][3] = float(root.matrix_world[1][3] + tangent_delta[1])
                root.matrix_world[2][3] = float(root.matrix_world[2][3] - tangent_gap)
                bpy.context.view_layer.update()
                bounds_cache.pop(object_id, None)
                tangent_after_min, _ = _sceneproof_world_hierarchy_bounds(root)
                tangent_gap_after = float(tangent_after_min[2] - tangent_support_z)
                after_overlaps = set(
                    _sceneproof_sparse_aabb_overlaps(object_id, roots, bounds_cache)
                ) - {parent_id}
                new_overlaps = sorted(after_overlaps - before_overlaps)
                if (
                    not new_overlaps
                    and abs(tangent_gap_after) <= contact_tolerance_m + 1e-5
                ):
                    row["pose_matrix_for_blender"] = [
                        list(matrix_row) for matrix_row in root.matrix_world
                    ]
                    record.update(
                        status="repaired",
                        reason="certified_tangent_then_vertical_parent_contact",
                        tangent_shift_xy_m=[float(value) for value in tangent_delta],
                        tangent_projection_norm_m=float(tangent_norm),
                        tangent_vertical_shift_m=float(-tangent_gap),
                        contact_gap_after_m=float(tangent_gap_after),
                        tangent_projection_limit_m=float(tangent_limit),
                        extended_support_program_search=extended_program_search,
                        support_program_ids=support_program_ids.get(object_id, []),
                        sparse_hit_count_after=len(tangent_hits),
                        sparse_hit_fraction_after=float(tangent_fraction),
                        new_world_aabb_overlaps=[],
                    )
                    repaired.append(object_id)
                    tangent_accepted = True
                    bvh_cache.clear()
                    support_component_cache.clear()
                    bounds_cache.clear()
                else:
                    root.matrix_world = before_matrix
                    bpy.context.view_layer.update()
                    bounds_cache.pop(object_id, None)
                    record["tangent_projection_new_world_aabb_overlaps"] = new_overlaps
            if tangent_accepted:
                records.append(record)
                continue

            # Otherwise keep XY/SO(3) frozen and find the highest real upward
            # surface below the current footprint.  The declared parent, lower
            # siblings, and floor all compete as witnesses; walls and ceilings
            # cannot win because their normals are not upward.
            drop_origin_z = float(child_min[2] + contact_tolerance_m)
            drop_distance = float(maximum_shift_m + 2.0 * contact_tolerance_m)
            surface_candidates = []
            for supporter_id, supporter_root in roots.items():
                if (
                    supporter_id in {object_id, parent_id}
                    or supporter_root is None
                ):
                    continue
                supporter_tree = bvh_cache.get(supporter_id)
                if supporter_tree is None:
                    try:
                        supporter_tree = _sceneproof_evaluated_bvh(supporter_root)
                        bvh_cache[supporter_id] = supporter_tree
                    except (TypeError, ValueError, RuntimeError):
                        continue
                supporter_hits = ray_hits(
                    supporter_tree, sample_xy,
                    origin_z=drop_origin_z, distance=drop_distance,
                )
                if not supporter_hits:
                    continue
                fraction = len(supporter_hits) / len(sample_xy)
                if fraction < minimum_hit_fraction:
                    continue
                support = max(supporter_hits)
                gap = float(child_min[2] - support)
                # This branch is a repair, not a generic contact classifier.
                # A zero-motion hit on an unrelated sibling must never turn an
                # unresolved object into a repaired one.
                if contact_tolerance_m < gap <= maximum_shift_m:
                    surface_candidates.append((support, supporter_id, gap, fraction))
            surface_candidates.sort(key=lambda item: (-item[0], item[1]))
            if surface_candidates:
                support_z, supporter_id, drop_gap, supporter_fraction = surface_candidates[0]
                z_shift = -float(drop_gap)
                before_matrix = root.matrix_world.copy()
                before_overlaps = set(
                    _sceneproof_sparse_aabb_overlaps(object_id, roots, bounds_cache)
                ) - {supporter_id}
                root.matrix_world[2][3] = float(root.matrix_world[2][3] + z_shift)
                bpy.context.view_layer.update()
                bounds_cache.pop(object_id, None)
                after_min, _ = _sceneproof_world_hierarchy_bounds(root)
                after_overlaps = set(
                    _sceneproof_sparse_aabb_overlaps(object_id, roots, bounds_cache)
                ) - {supporter_id}
                new_overlaps = sorted(after_overlaps - before_overlaps)
                gap_after = float(after_min[2] - support_z)
                if not new_overlaps and abs(gap_after) <= contact_tolerance_m + 1e-5:
                    row["pose_matrix_for_blender"] = [
                        list(matrix_row) for matrix_row in root.matrix_world
                    ]
                    record.update(
                        status="repaired",
                        reason="certified_vertical_first_contact_drop",
                        actual_supporter_id=supporter_id,
                        z_shift_m=float(z_shift),
                        vertical_drop_m=max(float(drop_gap), 0.0),
                        sparse_hit_fraction_after=float(supporter_fraction),
                        contact_gap_after_m=gap_after,
                        new_world_aabb_overlaps=[],
                    )
                    repaired.append(object_id)
                    bvh_cache.clear()
                    support_component_cache.clear()
                    bounds_cache.clear()
                    records.append(record)
                    continue
                root.matrix_world = before_matrix
                bpy.context.view_layer.update()
                bounds_cache.pop(object_id, None)
                record.update(
                    vertical_drop_postcheck_failed=True,
                    vertical_drop_new_world_aabb_overlaps=new_overlaps,
                )
            rollback_row = rollback_info.get(object_id, {})
            rollback_pose = (
                rollback_row.get("pose_matrix_for_blender")
                if isinstance(rollback_row, dict)
                else None
            )
            restored = False
            if rollback_pose is not None:
                try:
                    root.matrix_world = Matrix(rollback_pose)
                    bpy.context.view_layer.update()
                    row["pose_matrix_for_blender"] = [
                        list(matrix_row) for matrix_row in root.matrix_world
                    ]
                    bounds_cache.clear()
                    bvh_cache.clear()
                    support_component_cache.clear()
                    restored = True
                except (TypeError, ValueError, RuntimeError):
                    restored = False
            record.update(
                reason="visible_support_unresolved",
                rollback_restored=restored,
            )
            unresolved.append(object_id)
            records.append(record)
            continue
        support_z = max(hits)
        gap_before = float(child_min[2] - support_z)
        record.update(
            support_surface_z_m=float(support_z),
            child_bottom_z_before_m=float(child_min[2]),
            contact_gap_before_m=gap_before,
        )
        if abs(gap_before) <= contact_tolerance_m:
            record.update(status="held", reason="sparse_contact_within_tolerance", z_shift_m=0.0)
            held.append(object_id)
            records.append(record)
            continue
        z_shift = -gap_before
        if abs(z_shift) > maximum_shift_m:
            record.update(reason="required_z_shift_exceeds_budget", z_shift_m=float(z_shift))
            unresolved.append(object_id)
            records.append(record)
            continue

        before_matrix = root.matrix_world.copy()
        before_overlaps = set(
            _sceneproof_sparse_aabb_overlaps(object_id, roots, bounds_cache)
        ) - {parent_id}
        root.matrix_world[2][3] = float(root.matrix_world[2][3] + z_shift)
        bpy.context.view_layer.update()
        bounds_cache.pop(object_id, None)
        after_min, after_max = _sceneproof_world_hierarchy_bounds(root)
        after_overlaps = set(
            _sceneproof_sparse_aabb_overlaps(object_id, roots, bounds_cache)
        ) - {parent_id}
        new_overlaps = sorted(after_overlaps - before_overlaps)
        gap_after = float(after_min[2] - support_z)
        if new_overlaps or abs(gap_after) > contact_tolerance_m + 1e-5:
            root.matrix_world = before_matrix
            bpy.context.view_layer.update()
            bounds_cache.pop(object_id, None)
            record.update(
                reason="postcheck_failed_restored",
                z_shift_m=float(z_shift),
                contact_gap_after_m=gap_after,
                new_world_aabb_overlaps=new_overlaps,
            )
            unresolved.append(object_id)
        else:
            row["pose_matrix_for_blender"] = [
                list(matrix_row) for matrix_row in root.matrix_world
            ]
            record.update(
                status="repaired",
                reason="sparse_declared_parent_z_contact_certified",
                z_shift_m=float(z_shift),
                child_bottom_z_after_m=float(after_min[2]),
                contact_gap_after_m=gap_after,
                new_world_aabb_overlaps=[],
            )
            repaired.append(object_id)
            # Descendants must query the moved incumbent, never stale geometry.
            bvh_cache.clear()
            support_component_cache.clear()
            bounds_cache.clear()
        records.append(record)

    return {
        "schema_version": "sceneproof_support_contact_routing_v3",
        "policy": "dominant_true_mesh_component_then_tangent_then_vertical_first_contact",
        "certificate_strength": "support_routed_sparse_geometry",
        "passed": not unresolved,
        "status": "sparse_geometry_certified" if not unresolved else "unresolved",
        "repaired_object_ids": repaired,
        "held_object_ids": held,
        "unresolved_object_ids": unresolved,
        "objects": records,
        "parameters": {
            "contact_tolerance_m": float(contact_tolerance_m),
            "maximum_shift_m": float(maximum_shift_m),
            "minimum_hit_fraction": float(minimum_hit_fraction),
            "maximum_tangent_shift_m": float(maximum_tangent_shift_m),
            "maximum_program_tangent_shift_m": float(
                maximum_program_tangent_shift_m
            ),
        },
    }


def apply_sceneproof_visual_safe_salvage(
    placement_document,
    sparse_audit,
    *,
    maximum_floor_shift_m=0.60,
    maximum_suppressed_objects=4,
    boundary_tolerance_m=0.01,
):
    """Presentation-only fallback for unresolved, non-structural leaves.

    Candidates first fall onto the true floor at frozen XY/SO(3).  If blocked,
    a deterministic bounded radial search is used.  A candidate is accepted
    only when its hierarchy AABB overlaps no non-floor object and remains
    inside the true floor boundary.  Objects with no safe candidate may be
    hidden from this render, subject to a small global cap.  This audit is not
    a physical certificate and must never be included in paper metrics.
    """
    obj_info = placement_document.get("obj_info", {})
    unresolved = list(sparse_audit.get("unresolved_object_ids", []))
    roots = {
        object_id: bpy.data.objects.get(object_id)
        for object_id in obj_info
        if object_id != "scene_camera"
    }
    floor_id = placement_document.get("reference_obj", "floor_0")
    floor_root = roots.get(floor_id) or bpy.data.objects.get(floor_id)
    if floor_root is None:
        return {
            "policy": "presentation_only_floor_fallback_v1",
            "eligible_for_paper_metrics": False,
            "floor_relocated_object_ids": [],
            "render_suppressed_object_ids": [],
            "unresolved_object_ids": unresolved,
            "reason": "missing_true_floor",
        }
    children = {object_id: [] for object_id in obj_info}
    for object_id, row in obj_info.items():
        if not isinstance(row, dict):
            continue
        parent_id = row.get("supported")
        if isinstance(parent_id, str) and parent_id in children:
            children[parent_id].append(object_id)
    plane_pairs = _sceneproof_plane_attach_pairs(placement_document)
    plane_attached = {object_id for object_id, _ in plane_pairs}
    unresolved_set = set(unresolved)

    def source_bbox_area(row):
        for key in ("bbox", "bbox_xyxy", "box"):
            value = row.get(key) if isinstance(row, dict) else None
            if isinstance(value, (list, tuple)) and len(value) == 4:
                try:
                    return max(0.0, float(value[2]) - float(value[0])) * max(
                        0.0, float(value[3]) - float(value[1])
                    )
                except (TypeError, ValueError):
                    pass
        return 0.0

    # Conservative multiplicity trim: only groups with more than two sibling
    # instances and at least one unresolved member are eligible.  Never remove
    # a resolved member merely to hit the cap; unresolved low-evidence tails go
    # first, with a small global suppression budget.
    multiplicity_groups = {}
    for object_id, row in obj_info.items():
        if not isinstance(row, dict) or object_id == "scene_camera":
            continue
        parent_id = row.get("supported")
        semantic_stem = re.sub(r"_\d+$", "", object_id)
        multiplicity_groups.setdefault((semantic_stem, parent_id), []).append(object_id)
    pre_suppressed = []
    for group_key, member_ids in sorted(multiplicity_groups.items()):
        if len(member_ids) <= 2 or not unresolved_set.intersection(member_ids):
            continue
        eligible = []
        for object_id in member_ids:
            row = obj_info[object_id]
            parent_id = row.get("supported")
            if (
                object_id not in unresolved_set
                or children.get(object_id)
                or object_id in plane_attached
                or re.match(r"^(floor|wall|ceiling)_\d+$", object_id)
                or (
                    isinstance(parent_id, str)
                    and re.match(r"^(wall|ceiling)_\d+$", parent_id)
                )
            ):
                continue
            eligible.append((source_bbox_area(row), object_id))
        excess = max(0, len(member_ids) - 2)
        for _, object_id in sorted(eligible)[:excess]:
            if len(pre_suppressed) >= int(maximum_suppressed_objects):
                break
            root = roots.get(object_id)
            if root is None:
                continue
            for candidate in [root, *list(root.children_recursive)]:
                candidate.hide_render = True
            pre_suppressed.append(object_id)
        if len(pre_suppressed) >= int(maximum_suppressed_objects):
            break
    unresolved = [value for value in unresolved if value not in pre_suppressed]
    try:
        floor_tree = _sceneproof_evaluated_bvh(floor_root)
    except (TypeError, ValueError, RuntimeError) as error:
        return {
            "policy": "presentation_only_floor_fallback_v1",
            "eligible_for_paper_metrics": False,
            "floor_relocated_object_ids": [],
            "render_suppressed_object_ids": [],
            "unresolved_object_ids": unresolved,
            "reason": f"floor_geometry_unavailable:{type(error).__name__}",
        }
    support_roots = {
        object_id: root for object_id, root in roots.items()
        if object_id not in pre_suppressed
    }
    support_roots[floor_id] = floor_root
    support_tree_cache = {floor_id: floor_tree}

    radii = [0.0]
    step = 0.15
    count = max(0, int(math.floor(maximum_floor_shift_m / step + 1e-9)))
    radii.extend(step * index for index in range(1, count + 1))
    directions = [
        np.asarray((math.cos(2.0 * math.pi * index / 16.0),
                    math.sin(2.0 * math.pi * index / 16.0)), dtype=np.float64)
        for index in range(16)
    ]
    relocated = []
    remaining = []
    protected_ids = set()
    records = []
    bounds_cache = {}
    for object_id in unresolved:
        row = obj_info.get(object_id, {})
        root = roots.get(object_id)
        record = {"object_id": object_id, "status": "unresolved"}
        parent_id = row.get("supported") if isinstance(row, dict) else None
        protected = bool(
            root is None
            or children.get(object_id)
            or object_id in plane_attached
            or re.match(r"^(floor|wall|ceiling)_\d+$", object_id)
            or (
                isinstance(parent_id, str)
                and re.match(r"^(wall|ceiling)_\d+$", parent_id)
            )
        )
        if protected:
            protected_ids.add(object_id)
            record["reason"] = "protected_nonleaf_or_structural_object"
            remaining.append(object_id)
            records.append(record)
            continue
        before = root.matrix_world.copy()
        accepted = None
        last_blockers = []
        floor_hit_trials = 0
        support_hit_trials = 0
        best_boundary_error = float("inf")
        for radius in radii:
            trial_directions = [np.zeros(2, dtype=np.float64)] if radius == 0 else directions
            for direction in trial_directions:
                root.matrix_world = before.copy()
                root.matrix_world[0][3] = float(root.matrix_world[0][3] + radius * direction[0])
                root.matrix_world[1][3] = float(root.matrix_world[1][3] + radius * direction[1])
                bpy.context.view_layer.update()
                trial_min, trial_max = _sceneproof_world_hierarchy_bounds(root)
                center_xy = (trial_min[:2] + trial_max[:2]) / 2.0
                ray_origin_z = float(trial_min[2] + 0.10)
                support_hits = []
                for supporter_id, supporter_root in support_roots.items():
                    if supporter_id == object_id or supporter_root is None:
                        continue
                    supporter_tree = support_tree_cache.get(supporter_id)
                    if supporter_tree is None:
                        try:
                            supporter_tree = _sceneproof_evaluated_bvh(supporter_root)
                            support_tree_cache[supporter_id] = supporter_tree
                        except (TypeError, ValueError, RuntimeError):
                            continue
                    location, normal, _, _ = supporter_tree.ray_cast(
                        Vector((float(center_xy[0]), float(center_xy[1]), ray_origin_z)),
                        Vector((0.0, 0.0, -1.0)),
                        20.0,
                    )
                    if (
                        location is None
                        or normal is None
                        or float(normal.z) < 0.5
                        or float(location.z) > float(trial_min[2]) + 0.02
                    ):
                        continue
                    support_hits.append((float(location.z), supporter_id))
                if not support_hits:
                    continue
                support_hit_trials += 1
                support_z, supporter_id = max(support_hits, key=lambda item: (item[0], item[1]))
                if supporter_id == floor_id:
                    floor_hit_trials += 1
                z_shift = float(support_z - trial_min[2])
                if z_shift > 0.02:
                    continue
                root.matrix_world[2][3] = float(root.matrix_world[2][3] + z_shift)
                bpy.context.view_layer.update()
                bounds_cache.clear()
                blockers = sorted(
                    candidate_id
                    for candidate_id in set(
                        _sceneproof_sparse_aabb_overlaps(
                            object_id, roots, bounds_cache
                        )
                    ) - {floor_id, supporter_id} - set(pre_suppressed)
                    if not re.match(r"^(wall|ceiling)_\d+$", candidate_id)
                )
                last_blockers = blockers
                try:
                    boundary_error, _ = _sceneproof_true_mesh_boundary_error(
                        root, floor_root
                    )
                except (TypeError, ValueError, RuntimeError):
                    boundary_error = float("inf")
                best_boundary_error = min(best_boundary_error, float(boundary_error))
                if not blockers and boundary_error <= boundary_tolerance_m:
                    accepted = {
                        "xy_shift_m": [float(radius * direction[0]), float(radius * direction[1])],
                        "z_shift_m": z_shift,
                        "actual_supporter_id": supporter_id,
                        "boundary_error_m": float(boundary_error),
                    }
                    break
            if accepted is not None:
                break
        if accepted is None:
            root.matrix_world = before
            bpy.context.view_layer.update()
            record["reason"] = "no_collision_free_support_or_floor_candidate"
            record["last_blockers"] = last_blockers
            record["support_hit_trials"] = support_hit_trials
            record["floor_hit_trials"] = floor_hit_trials
            record["best_boundary_error_m"] = (
                float(best_boundary_error)
                if math.isfinite(best_boundary_error) else None
            )
            remaining.append(object_id)
        else:
            row["pose_matrix_for_blender"] = [
                list(matrix_row) for matrix_row in root.matrix_world
            ]
            record.update(status="floor_relocated", reason="true_floor_fallback", **accepted)
            relocated.append(object_id)
            support_tree_cache.clear()
            support_tree_cache[floor_id] = floor_tree
        records.append(record)

    # Hide only a small number of unresolved leaf objects.  Repeated assets and
    # small hierarchy volumes are lower priority, but no category whitelist is
    # used and all suppressions remain explicit in the audit.
    suppression_candidates = []
    asset_counts = {}
    asset_keys = {}
    for object_id in remaining:
        row = obj_info.get(object_id, {})
        if (
            children.get(object_id)
            or object_id in plane_attached
            or object_id in protected_ids
        ):
            continue
        asset_key = next((row.get(key) for key in (
            "model_path", "retrieved_asset_path", "asset_path", "uid"
        ) if isinstance(row.get(key), str)), object_id)
        asset_keys[object_id] = asset_key
        asset_counts[asset_key] = asset_counts.get(asset_key, 0) + 1
    for object_id, asset_key in asset_keys.items():
        root = roots.get(object_id)
        if root is None:
            continue
        try:
            lower, upper = _sceneproof_world_hierarchy_bounds(root)
            volume = float(np.prod(np.maximum(upper - lower, 1e-6)))
        except (TypeError, ValueError, RuntimeError):
            volume = float("inf")
        suppression_candidates.append((
            0 if asset_counts.get(asset_key, 1) > 1 else 1,
            volume,
            object_id,
        ))
    suppressed = list(pre_suppressed)
    for object_id in pre_suppressed:
        records.append({
            "object_id": object_id,
            "status": "render_suppressed",
            "reason": "bounded_unresolved_multiplicity_trim",
        })
    for _, _, object_id in sorted(suppression_candidates):
        if len(suppressed) >= int(maximum_suppressed_objects):
            break
        root = roots[object_id]
        for candidate in [root, *list(root.children_recursive)]:
            candidate.hide_render = True
        suppressed.append(object_id)
        for record in records:
            if record["object_id"] == object_id:
                record.update(status="render_suppressed", reason="bounded_visual_salvage")
                break
    final_unresolved = [value for value in remaining if value not in suppressed]
    bpy.context.view_layer.update()
    return {
        "policy": "presentation_only_floor_fallback_v1",
        "eligible_for_paper_metrics": False,
        "maximum_suppressed_objects": int(maximum_suppressed_objects),
        "floor_relocated_object_ids": relocated,
        "render_suppressed_object_ids": suppressed,
        "unresolved_object_ids": final_unresolved,
        "objects": records,
    }


def _sceneproof_overlap_object_ids(target_id, roots, *, bvh_cache=None):
    """Return exact evaluated-mesh triangle-overlap witnesses for one root.

    If *bvh_cache* is provided it must be a dict mapping object_id to its
    precomputed BVH tree; every object that the target should be tested against
    must appear in the cache.  Objects missing from the cache are skipped with a
    logged abstention rather than rebuilt, because the whole point of the cache is
    to avoid N redundant BVH evaluations inside a tight loop.
    """
    target = roots[target_id]
    target_tree = _sceneproof_evaluated_bvh(target)
    overlaps = []
    for object_id, _ in sorted(roots.items()):
        if object_id == target_id:
            continue
        if bvh_cache is not None:
            cached = bvh_cache.get(object_id)
            if cached is None:
                overlaps.append(f"__MISSING_CACHE__:{object_id}")
                continue
            try:
                if target_tree.overlap(cached):
                    overlaps.append(object_id)
            except (TypeError, ValueError, RuntimeError):
                overlaps.append(f"__ABSTAIN__:{object_id}")
            continue
        root = roots[object_id]
        try:
            if target_tree.overlap(_sceneproof_evaluated_bvh(root)):
                overlaps.append(object_id)
        except (TypeError, ValueError, RuntimeError):
            overlaps.append(f"__ABSTAIN__:{object_id}")
    return overlaps


def _sceneproof_overlap_triangle_pair_counts(target_id, roots, *, bvh_cache):
    """Count exact evaluated-mesh BVH triangle-overlap pairs per object."""
    target = roots.get(target_id)
    if target is None:
        return {}, ["missing_target"]
    try:
        target_tree = _sceneproof_evaluated_bvh(target)
    except (TypeError, ValueError, RuntimeError) as error:
        return {}, [f"target:{type(error).__name__}"]
    counts = {}
    abstained = []
    for object_id, root in roots.items():
        if object_id == target_id:
            continue
        other_tree = bvh_cache.get(object_id)
        if other_tree is None:
            abstained.append(object_id)
            continue
        try:
            counts[object_id] = len(target_tree.overlap(other_tree))
        except (TypeError, ValueError, RuntimeError):
            abstained.append(object_id)
    return counts, sorted(abstained)


def _sceneproof_rotation_delta_radians(before, after):
    first = Matrix(np.asarray(before, dtype=np.float64).tolist()).to_quaternion()
    second = Matrix(np.asarray(after, dtype=np.float64).tolist()).to_quaternion()
    return shortest_rotation_angle(
        float(first.rotation_difference(second).angle)
    )


def _sceneproof_true_mesh_boundary_error(root, floor_root):
    """Return maximum XY half-space violation against the true floor hull."""
    floor_mesh = _sceneproof_world_trimesh(floor_root)
    object_mesh = _sceneproof_world_trimesh(root)
    polygon = convex_hull_2d(
        np.asarray(floor_mesh.vertices, dtype=np.float64)[:, :2]
    )
    if len(polygon) < 3:
        raise ValueError("floor mesh has no non-degenerate XY boundary")
    signed_area_twice = float(
        np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - polygon[:, 1] * np.roll(polygon[:, 0], -1)
        )
    )
    if signed_area_twice < 0:
        polygon = polygon[::-1].copy()
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1e-10):
        raise ValueError("floor boundary contains a zero-length edge")
    inward = np.stack((-edges[:, 1], edges[:, 0]), axis=1) / lengths[:, None]
    points = np.asarray(object_mesh.vertices, dtype=np.float64)[:, :2]
    signed = np.einsum(
        "pbd,bd->pb", points[:, None, :] - polygon[None, :, :], inward
    )
    return float(max(0.0, -float(np.min(signed)))), polygon.tolist()


def audit_sceneproof_local_gravity_settle(
    placement_document,
    object_id,
    *,
    duration_seconds=1.0,
):
    """Run one isolated full-SO(3) Bullet counterfactual and restore poses.

    This is deliberately an oracle, not a pose-writing operator.  The caller
    launches one Blender process per object, so Bullet caches and rigid-body
    ownership cannot leak between trials.  Exact-mesh COM/contact and all
    target collision candidates are remeasured after the settle.
    """
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("local settle requires a non-empty object id")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("local settle duration must be positive and finite")

    obj_info = placement_document.get("obj_info", {})
    roots = {
        candidate_id: bpy.data.objects.get(candidate_id)
        for candidate_id in obj_info
        if candidate_id != "scene_camera"
    }
    missing = sorted(key for key, root in roots.items() if root is None)
    roots = {key: root for key, root in roots.items() if root is not None}
    audit = {
        "schema_version": "sceneproof_local_gravity_settle_oracle_v1",
        "policy": "audit_only_process_isolated_full_so3",
        "object_id": object_id,
        "declared_parent_id": obj_info.get(object_id, {}).get("supported"),
        "duration_seconds": float(duration_seconds),
        "missing_scene_roots": missing,
        "mutates_placement": False,
        "status": "abstained",
    }
    root = roots.get(object_id)
    if root is None:
        audit["reason"] = "missing_target_root"
        return audit
    target_meshes = _sceneproof_mesh_hierarchy(root)
    if len(target_meshes) != 1 or target_meshes[0] != root:
        audit.update(
            {
                "reason": "multi_mesh_rigidbody_ownership_unproven",
                "owned_meshes": [mesh.name for mesh in target_meshes],
            }
        )
        return audit

    pose_before = {
        candidate_id: np.asarray(candidate.matrix_world, dtype=np.float64)
        for candidate_id, candidate in roots.items()
    }
    horizontal_mesh_radius_m = None
    horizontal_mesh_radius_error = None
    try:
        vertices_before = np.asarray(
            _sceneproof_world_trimesh(root).vertices, dtype=np.float64
        )
        origin_before = pose_before[object_id][:3, 3]
        horizontal_mesh_radius_m = float(
            np.max(
                np.linalg.norm(
                    vertices_before[:, :2] - origin_before[None, :2], axis=1
                )
            )
        )
    except (TypeError, ValueError, RuntimeError) as error:
        horizontal_mesh_radius_error = f"{type(error).__name__}: {error}"
    scene = bpy.context.scene
    scene_state = {
        "frame_start": int(scene.frame_start),
        "frame_end": int(scene.frame_end),
        "frame_current": int(scene.frame_current),
    }
    before_support = audit_sceneproof_true_mesh_com_support(
        placement_document, scoped_object_ids=[object_id]
    )
    before_row = before_support.get("objects", {}).get(object_id)
    # Pre-build BVH trees for all passive objects once.  Without this cache each
    # call to _sceneproof_overlap_object_ids rebuilds N-1 BVH trees from scratch,
    # and a local-settle probe calls it twice (before + after).  In casino with
    # 111 objects that is 220 builds of which 110 are identical — the passives do
    # not move during the simulation.  The cache turns this from O(#passives^2)
    # into O(#passives), which is the difference between usable per-object settle
    # and a per-scene timeout.
    passive_bvh_cache = {}
    for pid, p_root in roots.items():
        if pid == object_id:
            continue
        try:
            passive_bvh_cache[pid] = _sceneproof_evaluated_bvh(p_root)
        except (TypeError, ValueError, RuntimeError):
            passive_bvh_cache[pid] = None
    before_overlaps = _sceneproof_overlap_object_ids(
        object_id, roots, bvh_cache=passive_bvh_cache
    )
    before_overlap_pair_counts, before_overlap_count_abstained = (
        _sceneproof_overlap_triangle_pair_counts(
            object_id, roots, bvh_cache=passive_bvh_cache
        )
    )
    floor_id = placement_document.get("reference_obj")
    if not isinstance(floor_id, str) or floor_id not in roots:
        floor_id = next(
            (candidate_id for candidate_id in roots if re.match(r"^floor_\d+$", candidate_id)),
            None,
        )
    before_boundary_error = None
    boundary_polygon = None
    boundary_error = None
    if floor_id is not None:
        try:
            before_boundary_error, boundary_polygon = (
                _sceneproof_true_mesh_boundary_error(root, roots[floor_id])
            )
        except (TypeError, ValueError, RuntimeError) as error:
            boundary_error = f"{type(error).__name__}: {error}"
    colliders = []
    for candidate_id, candidate in roots.items():
        if candidate_id == object_id:
            continue
        colliders.extend(_sceneproof_mesh_hierarchy(candidate))

    try:
        active_override = {}
        for key, env_key in (
            ("linear_damping", "IMAGINARIUM_SETTLE_LINEAR_DAMPING"),
            ("angular_damping", "IMAGINARIUM_SETTLE_ANGULAR_DAMPING"),
            ("friction", "IMAGINARIUM_SETTLE_FRICTION"),
            ("margin", "IMAGINARIUM_SETTLE_MARGIN"),
        ):
            raw = os.environ.get(env_key)
            if raw is not None:
                try:
                    active_override[key] = float(raw)
                except (TypeError, ValueError):
                    pass
        active_settings = active_override or None
        passive_override = {}
        for key, env_key in (
            ("friction", "IMAGINARIUM_SETTLE_PASSIVE_FRICTION"),
            ("margin", "IMAGINARIUM_SETTLE_PASSIVE_MARGIN"),
        ):
            raw = os.environ.get(env_key)
            if raw is not None:
                try:
                    passive_override[key] = float(raw)
                except (TypeError, ValueError):
                    pass
        passive_settings = passive_override or None
        world_override = {}
        for key, env_key in (
            ("substeps", "IMAGINARIUM_SETTLE_SUBSTEPS"),
            ("solver_iterations", "IMAGINARIUM_SETTLE_SOLVER_ITERATIONS"),
        ):
            raw = os.environ.get(env_key)
            if raw is not None:
                try:
                    world_override[key] = int(raw)
                except (TypeError, ValueError):
                    pass
        world_settings = world_override or None
        simulated = run_drop_simulation(
            [root],
            colliders,
            duration=float(duration_seconds),
            scene=scene,
            active_settings=active_settings,
            passive_settings=passive_settings,
            world_settings=world_settings,
        )
        bpy.context.view_layer.update()
        pose_after = np.asarray(root.matrix_world, dtype=np.float64)
        after_support = audit_sceneproof_true_mesh_com_support(
            placement_document, scoped_object_ids=[object_id]
        )
        after_row = after_support.get("objects", {}).get(object_id)
        after_overlaps = _sceneproof_overlap_object_ids(
            object_id, roots, bvh_cache=passive_bvh_cache
        )
        after_overlap_pair_counts, after_overlap_count_abstained = (
            _sceneproof_overlap_triangle_pair_counts(
                object_id, roots, bvh_cache=passive_bvh_cache
            )
        )
        after_boundary_error = None
        if floor_id is not None and boundary_error is None:
            try:
                after_boundary_error, _ = _sceneproof_true_mesh_boundary_error(
                    root, roots[floor_id]
                )
            except (TypeError, ValueError, RuntimeError) as error:
                boundary_error = f"{type(error).__name__}: {error}"
        translation_delta = float(
            np.linalg.norm(pose_after[:3, 3] - pose_before[object_id][:3, 3])
        )
        rotation_delta = _sceneproof_rotation_delta_radians(
            pose_before[object_id], pose_after
        )
        translation_vector = (
            pose_after[:3, 3] - pose_before[object_id][:3, 3]
        )
        horizontal_translation = float(
            np.linalg.norm(translation_vector[:2])
        )
        vertical_translation = float(translation_vector[2])
        if horizontal_mesh_radius_m is None:
            horizontal_motion_passed = False
            horizontal_motion_bound = None
        else:
            horizontal_motion_passed, horizontal_motion_bound = (
                rotation_explained_horizontal_motion(
                    horizontal_translation,
                    rotation_delta,
                    horizontal_mesh_radius_m,
                    slip_tolerance_m=float(
                        os.environ.get(
                            "IMAGINARIUM_SETTLE_XY_SLIP_TOLERANCE_M", "0.005"
                        )
                    ),
                )
            )
        before_set = set(before_overlaps)
        after_set = set(after_overlaps)
        audit.update(
            {
                "status": "measured" if simulated else "abstained",
                "reason": (
                    "local_gravity_settle_measured"
                    if simulated
                    else "bullet_simulation_returned_false"
                ),
                "before_pose_matrix": pose_before[object_id].tolist(),
                "settled_pose_matrix": pose_after.tolist(),
                "translation_delta_m": translation_delta,
                "translation_delta_xyz_m": translation_vector.tolist(),
                "horizontal_translation_delta_m": horizontal_translation,
                "vertical_translation_delta_m": vertical_translation,
                "rotation_delta_rad": rotation_delta,
                "rotation_delta_deg": float(math.degrees(rotation_delta)),
                "before_support": before_row,
                "after_support": after_row,
                "before_collision_object_ids": sorted(before_overlaps),
                "after_collision_object_ids": sorted(after_overlaps),
                "new_collision_object_ids": sorted(after_set - before_set),
                "removed_collision_object_ids": sorted(before_set - after_set),
                "before_exact_overlap_triangle_pairs_by_object": before_overlap_pair_counts,
                "after_exact_overlap_triangle_pairs_by_object": after_overlap_pair_counts,
                "exact_overlap_count_abstained_object_ids": sorted(
                    set(before_overlap_count_abstained)
                    | set(after_overlap_count_abstained)
                ),
                "exact_overlap_triangle_pairs_nonincreasing": bool(
                    not before_overlap_count_abstained
                    and not after_overlap_count_abstained
                    and all(
                        after_overlap_pair_counts.get(name, 0)
                        <= before_overlap_pair_counts.get(name, 0)
                        for name in set(before_overlap_pair_counts)
                        | set(after_overlap_pair_counts)
                    )
                ),
                "horizontal_motion_certificate": {
                    "passed": horizontal_motion_passed,
                    "policy": "rotation_chord_plus_slip_tolerance",
                    "horizontal_mesh_radius_m": horizontal_mesh_radius_m,
                    "horizontal_mesh_radius_error": horizontal_mesh_radius_error,
                    "maximum_explained_horizontal_motion_m": horizontal_motion_bound,
                    "observed_horizontal_motion_m": horizontal_translation,
                    "slip_tolerance_m": float(
                        os.environ.get(
                            "IMAGINARIUM_SETTLE_XY_SLIP_TOLERANCE_M", "0.005"
                        )
                    ),
                },
                "collision_candidates_checked": max(len(roots) - 1, 0),
                "simulation_settings": {
                    "active_overrides": active_override,
                    "passive_overrides": passive_override,
                    "world_overrides": world_override,
                    "duration_seconds": float(duration_seconds),
                },
                "room_boundary_floor_id": floor_id,
                "room_boundary_polygon_xy_m": boundary_polygon,
                "before_boundary_error_m": before_boundary_error,
                "after_boundary_error_m": after_boundary_error,
                "boundary_measurement_error": boundary_error,
            }
        )
    except Exception as error:
        audit.update(
            {
                "status": "failed",
                "reason": "local_gravity_settle_exception",
                "error": f"{type(error).__name__}: {error}",
            }
        )
    finally:
        for candidate_id, matrix in pose_before.items():
            roots[candidate_id].matrix_world = Matrix(matrix.tolist())
        scene.frame_start = scene_state["frame_start"]
        scene.frame_end = scene_state["frame_end"]
        scene.frame_set(scene_state["frame_current"])
        bpy.context.view_layer.update()

    restored_delta = max(
        (
            float(
                np.max(
                    np.abs(
                        np.asarray(root_object.matrix_world, dtype=np.float64)
                        - pose_before[candidate_id]
                    )
                )
            )
            for candidate_id, root_object in roots.items()
        ),
        default=0.0,
    )
    restoration_scale = max(
        (
            float(np.max(np.abs(matrix)))
            for matrix in pose_before.values()
        ),
        default=1.0,
    )
    restoration_tolerance = float(
        16.0 * np.finfo(np.float32).eps * max(1.0, restoration_scale)
    )
    audit["incumbent_restored"] = bool(
        restored_delta <= restoration_tolerance
    )
    audit["maximum_restoration_error"] = restored_delta
    audit["restoration_tolerance"] = restoration_tolerance
    audit["restoration_storage_dtype"] = "float32"
    return audit


def audit_sceneproof_local_gravity_settle_bulk(
    placement_document: dict,
    object_ids: list[str],
    *,
    duration_seconds: float = 1.0,
) -> dict[str, dict]:
    """Drop *all* target objects simultaneously in one Bullet simulation,
    then audit each target independently.  The scene is restored to its
    pre-simulation state afterward.

    Returns  ``{object_id: audit_dict, ...}``, where each audit dict has the
    same schema as the single-object ``audit_sceneproof_local_gravity_settle``.
    """
    import bpy
    import numpy as np

    if not object_ids:
        return {}
    target_set = set(object_ids)
    obj_info = placement_document.get("obj_info", {})

    # 1. look up all roots and save poses
    pose_before: dict[str, np.ndarray] = {}
    roots: dict[str, bpy.types.Object] = {}
    for name in obj_info:
        if name == "scene_camera":
            continue
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        roots[name] = obj
        pose_before[name] = np.asarray(obj.matrix_world, dtype=np.float64)

    target_roots = [roots[tid] for tid in object_ids if tid in roots]
    if not target_roots:
        return {}

    # 2. build colliders: all OTHER roots (non-target) are static obstacles
    colliders = []
    for name, obj in roots.items():
        if name in target_set:
            continue
        colliders.extend(_sceneproof_mesh_hierarchy(obj))

    # 3.5 save pre-simulation overlaps and COM for collision detection
    before_com = audit_sceneproof_true_mesh_com_support(
        placement_document, scoped_object_ids=object_ids
    )
    before_overlaps: dict[str, set] = {}
    for target_id in object_ids:
        if target_id in roots:
            before_overlaps[target_id] = _sceneproof_overlap_object_ids(
                target_id, roots
            )

    # 4. one joint drop simulation
    scene = bpy.context.scene
    scene_state = {
        "frame_start": int(scene.frame_start),
        "frame_end": int(scene.frame_end),
        "frame_current": int(scene.frame_current),
    }
    try:
        simulated = run_drop_simulation(
            target_roots,
            colliders,
            duration=float(duration_seconds),
            scene=scene,
        )
        bpy.context.view_layer.update()
    except Exception as exc:
        simulated = False
        sim_error = f"{type(exc).__name__}: {exc}"
    else:
        sim_error = None

    # 4. COM audit after simulation for targets only
    after_com = audit_sceneproof_true_mesh_com_support(
        placement_document, scoped_object_ids=object_ids
    )

    # 5. audit each target
    results: dict[str, dict] = {}
    floor_id = placement_document.get("reference_obj")
    if not isinstance(floor_id, str) or floor_id not in roots:
        floor_id = next(
            (n for n in roots if re.match(r"^floor_\d+$", n)), None
        )

    for target_id in object_ids:
        target = roots.get(target_id)
        if target is None:
            results[target_id] = {
                "object_id": target_id,
                "status": "skipped",
                "reason": "object_not_found_in_scene",
            }
            continue

        pose_after = np.asarray(target.matrix_world, dtype=np.float64)
        translation = float(
            np.linalg.norm(
                pose_after[:3, 3] - pose_before[target_id][:3, 3]
            )
        )
        rotation = _sceneproof_rotation_delta_radians(
            pose_before[target_id], pose_after
        )
        rotation_deg = float(math.degrees(rotation))

        # collisions: new overlaps only
        after_overlaps = _sceneproof_overlap_object_ids(target_id, roots)
        new_collisions = sorted(
            set(after_overlaps)
            - before_overlaps.get(target_id, set())
        )

        before_row = before_com.get("objects", {}).get(target_id, {})
        after_row = after_com.get("objects", {}).get(target_id, {})

        before_boundary = None
        after_boundary = None
        boundary_meas_error = None
        if floor_id is not None:
            try:
                before_boundary, _ = _sceneproof_true_mesh_boundary_error(
                    target, roots[floor_id]
                )
                after_boundary, _ = _sceneproof_true_mesh_boundary_error(
                    target, roots[floor_id]
                )
            except Exception as exc:
                boundary_meas_error = f"{type(exc).__name__}: {exc}"

        results[target_id] = {
            "object_id": target_id,
            "status": "measured" if simulated else "abstained",
            "reason": (
                "bulk_gravity_settle_measured" if simulated
                else f"bullet_simulation_returned_false_{sim_error}"
            ),
            "before_pose_matrix": pose_before[target_id].tolist(),
            "settled_pose_matrix": pose_after.tolist(),
            "translation_delta_m": translation,
            "rotation_delta_rad": rotation,
            "rotation_delta_deg": rotation_deg,
            "before_support": before_row,
            "after_support": after_row,
            "new_collision_object_ids": new_collisions,
            "before_boundary_error_m": before_boundary,
            "after_boundary_error_m": after_boundary,
            "boundary_measurement_error": boundary_meas_error,
            "incumbent_restored": False,
        }

    # 6. restore ALL poses
    try:
        for name, obj in roots.items():
            obj.matrix_world = Matrix(pose_before[name].tolist())
        scene.frame_start = scene_state["frame_start"]
        scene.frame_end = scene_state["frame_end"]
        scene.frame_set(scene_state["frame_current"])
        bpy.context.view_layer.update()
    except Exception:
        pass

    restoration_scale = max(
        (float(np.max(np.abs(m))) for m in pose_before.values()), default=1.0
    )
    restoration_tol = float(
        16.0 * np.finfo(np.float32).eps * max(1.0, restoration_scale)
    )
    for name, audit in results.items():
        obj = roots.get(name)
        if obj is not None:
            after_mat = np.asarray(obj.matrix_world, dtype=np.float64)
            error = float(
                np.max(np.abs(after_mat - pose_before[name]))
            )
            audit["incumbent_restored"] = bool(error <= restoration_tol)
            audit["maximum_restoration_error"] = error
            audit["restoration_tolerance"] = restoration_tol
            audit["restoration_storage_dtype"] = "float32"

    return results


def audit_sceneproof_global_gravity_settle(
    placement_document: dict,
    *,
    duration_seconds: float = 2.0,
    output_root=None,
) -> dict[str, dict]:
    """Drop *all* non-structural objects simultaneously in one Bullet
    simulation, record every object that moved, and restore the scene.

    NOT PHYSICALLY WELL POSED — diagnostics only, never use for promotion.

    Every non-structural object becomes dynamic while only the architecture
    remains a static collider, so this routine also drops objects that are not
    free to fall:

    * a child whose declared support parent is a wall or a ceiling is attached,
      not resting, and slides off its wall onto the floor;
    * a child declared ``inside`` a container is held by containment semantics
      and falls out of its container;
    * a support parent is dynamic at the same time as its children, so a stack
      collapses instead of settling.

    Measured on Smoke5 this moves 95 to 98 per cent of all objects by more than
    5 cm and drives the plane family down by up to 0.52 and the semantic family
    down by up to 0.72.  For promotion use
    ``audit_sceneproof_local_gravity_settle_bulk`` with the target list produced
    by ``sceneproof_settle_eligibility_screen_fix86.py``: there only the screened
    targets are dynamic and every other object stays a static collider.

    Returns ``{object_id: audit_dict}``.
    """
    import bpy
    import numpy as np
    import math
    from pathlib import Path

    obj_info = placement_document.get("obj_info", {})

    # 1. classify: only non-STRUCTURAL objects become dynamic
    target_ids = []
    for name in obj_info:
        if name == "scene_camera":
            continue
        if re.match(r"^(floor|ground|wall|ceiling|carpet|rug|structural)_\d+$", name):
            continue
        target_ids.append(name)

    import traceback
    print(f"[SceneProof] Global settle: scanning {len(obj_info)} objects...", flush=True)

    # 2. save every pose
    pose_before: dict[str, np.ndarray] = {}
    roots: dict[str, bpy.types.Object] = {}
    for name in obj_info:
        if name == "scene_camera":
            continue
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        roots[name] = obj
        pose_before[name] = np.asarray(obj.matrix_world, dtype=np.float64)

    target_roots = [roots[tid] for tid in target_ids if tid in roots]
    if not target_roots:
        return {}

    # 3. colliders: architectural / non-target
    colliders = []
    for name, obj in roots.items():
        if name in set(target_ids):
            continue
        try:
            colliders.extend(_sceneproof_mesh_hierarchy(obj))
        except Exception:
            pass

    # 4. one global drop
    scene = bpy.context.scene
    simulated = False
    try:
        print(f"[SceneProof] Global settle: dropping {len(target_roots)} objects...", flush=True)
        simulated = run_drop_simulation(
            target_roots, colliders, duration=float(duration_seconds),
            scene=scene,
        )
        bpy.context.view_layer.update()
        print(f"[SceneProof] Global settle: sim returned {simulated}", flush=True)
    except Exception as exc:
        print(f"[SceneProof] Global settle ERROR: {exc}", flush=True)
        traceback.print_exc()
        simulated = False

    # 6. record per-object results (no COM measurement — too slow
    #    for 80+ objects; COM is audited post-hoc only for promoted
    #    candidates in the evaluation stage).
    results: dict[str, dict] = {}
    for target_id in target_ids:
        obj = roots.get(target_id)
        if obj is None:
            continue
        pose_after = np.asarray(obj.matrix_world, dtype=np.float64)
        trans = float(np.linalg.norm(pose_after[:3, 3] - pose_before[target_id][:3, 3]))
        rot = float(math.degrees(_sceneproof_rotation_delta_radians(
            pose_before[target_id], pose_after
        )))
        results[target_id] = {
            "object_id": target_id,
            "status": "measured" if simulated else "abstained",
            "translation_delta_m": trans,
            "rotation_delta_deg": rot,
            "before_pose_matrix": pose_before[target_id].tolist(),
            "settled_pose_matrix": pose_after.tolist(),
            "after_support": {},
        }

    # 7. restore all poses
    for name, obj in roots.items():
        obj.matrix_world = Matrix(pose_before.get(name, np.eye(4)).tolist())
    bpy.context.view_layer.update()

    # 8. write probes (optional)
    if output_root is not None:
        out_path = Path(output_root)
        out_path.mkdir(parents=True, exist_ok=True)
        import json as _json
        manifest: list[str] = []
        for tid, audit in results.items():
            probe = out_path / f"{tid}.json"
            probe.write_text(_json.dumps(audit, indent=2), encoding="utf-8")
            manifest.append(str(probe))
        manifest_path = out_path / "_manifest.json"
        manifest_path.write_text(_json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[SceneProof] Global settle: wrote {len(results)} probes to {output_root}", flush=True)
    print(f"[SceneProof] Global settle: return {len(results)} objects", flush=True)
    return results


def audit_sceneproof_mesh_visibility(
    obj_placement_info_json_path,
    ordered_ids,
    scene_camera,
    sibling_audit,
    *,
    plane_bindings=None,
    pose_matrices=None,
    local_corners=None,
    collision_pairs=None,
    footprint_hull_sizes=None,
    support_pairs=None,
    fixed_support_indices=None,
    fixed_support_heights=None,
    containment_pairs=None,
    boundary_object_indices=None,
    boundary_points=None,
    boundary_normals=None,
    visible_side_candidate_audit=False,
    tangent_candidate_audit=False,
    joint_tangent_candidate_audit=False,
    resolution=256,
    minimum_pixels=64,
    visible_side_clearance_m=0.002,
    visible_side_maximum_shift_m=0.15,
    visible_side_attachment_tolerance_m=0.005,
    tangent_maximum_shift_m=0.35,
    visibility_noharm_tolerance=0.005,
    minimum_recall_gain=0.05,
    physical_tolerance=1e-6,
    color_id_output_path=None,
    annotated_color_id_output_path=None,
):
    """Measure real visible mesh masks without changing the saved scene.

    The full-scene Workbench color-ID pass measures z-buffer-visible geometry. A
    second isolated pass is rendered only for low-visibility members of a
    plane-sibling component, separating occlusion from an off-screen or
    degenerate mesh.  The function restores every touched Blender property.
    """
    from PIL import Image, ImageDraw

    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    result_root = Path(obj_placement_info_json_path).resolve().parent.parent
    mask_root = result_root / "S1_scene_parsing_results" / "masks"
    if not mask_root.is_dir():
        raise FileNotFoundError(
            f"SceneProof mesh visibility requires {mask_root}"
        )
    if scene_camera is None:
        raise RuntimeError("SceneProof mesh visibility requires a camera")
    if resolution < 32 or minimum_pixels < 1:
        raise ValueError("invalid mesh visibility audit resolution/threshold")

    roots = {}
    mesh_groups = {}
    mesh_owners = {}
    for object_id in ordered_ids:
        root = bpy.data.objects.get(object_id)
        if root is None:
            continue
        meshes = _sceneproof_mesh_hierarchy(root)
        if not meshes:
            continue
        roots[object_id] = root
        mesh_groups[object_id] = meshes
        for mesh in meshes:
            owner = mesh_owners.setdefault(mesh.as_pointer(), object_id)
            if owner != object_id:
                raise RuntimeError(
                    "SceneProof visibility found shared mesh ownership: "
                    f"{mesh.name} -> {owner}, {object_id}"
                )

    # The object-level groups above intentionally follow the placement JSON,
    # but assigning every other renderable mesh to black made the occlusion
    # audit conflate walls, room shells, and importer-created auxiliaries into
    # UNKNOWN_FIXED_GEOMETRY.  Give each unowned mesh its own diagnostic ID.
    # These extra groups are attribution witnesses only: placement metrics and
    # candidate ownership continue to use ``mesh_groups`` exclusively.
    render_groups = dict(mesh_groups)
    render_display_names = {object_id: object_id for object_id in mesh_groups}
    for mesh in sorted(
        (obj for obj in scene.objects if obj.type == "MESH"),
        key=lambda obj: obj.name,
    ):
        if mesh.as_pointer() in mesh_owners:
            continue
        diagnostic_id = f"__scene_mesh__:{mesh.name}"
        suffix = 1
        while diagnostic_id in render_groups:
            suffix += 1
            diagnostic_id = f"__scene_mesh__:{mesh.name}:{suffix}"
        render_groups[diagnostic_id] = [mesh]
        render_display_names[diagnostic_id] = mesh.name

    def hierarchy_world_vertices(root):
        vertices = []
        for mesh_object in _sceneproof_mesh_hierarchy(root):
            matrix = mesh_object.matrix_world
            vertices.extend(
                tuple(matrix @ vertex.co)
                for vertex in mesh_object.data.vertices
            )
        if not vertices:
            vertices = [
                tuple(root.matrix_world @ Vector(corner))
                for corner in root.bound_box
            ]
        values = np.asarray(vertices, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                f"invalid mesh vertices for visibility audit: {root.name}"
            )
        return values

    candidate_physical_audit = bool(
        visible_side_candidate_audit
        or tangent_candidate_audit
        or joint_tangent_candidate_audit
    )
    if candidate_physical_audit:
        required_context = {
            "pose_matrices": pose_matrices,
            "local_corners": local_corners,
            "collision_pairs": collision_pairs,
            "support_pairs": support_pairs,
            "fixed_support_indices": fixed_support_indices,
            "fixed_support_heights": fixed_support_heights,
            "containment_pairs": containment_pairs,
            "boundary_object_indices": boundary_object_indices,
            "boundary_points": boundary_points,
            "boundary_normals": boundary_normals,
        }
        missing_context = [
            name for name, value in required_context.items() if value is None
        ]
        if missing_context:
            raise RuntimeError(
                "visible-side candidate audit is missing physical context: "
                + ", ".join(missing_context)
            )
        if not plane_bindings:
            raise RuntimeError(
                "visibility candidate audit has no PLANE_ATTACH bindings"
            )
        if pose_matrices.shape[0] != len(ordered_ids):
            raise ValueError(
                "visible-side pose batch does not match ordered_ids"
            )
        if (
            visible_side_clearance_m < 0
            or visible_side_maximum_shift_m <= 0
            or visible_side_attachment_tolerance_m < 0
            or tangent_maximum_shift_m <= 0
            or visibility_noharm_tolerance < 0
            or minimum_recall_gain <= 0
            or physical_tolerance < 0
        ):
            raise ValueError("invalid visible-side audit thresholds")

    component_ids = set()
    for component in (sibling_audit or {}).get("component_audits", []):
        component_ids.update(component.get("object_ids", []))

    camera_before = np.asarray(scene_camera.matrix_world, dtype=np.float64)
    pose_before = {
        object_id: np.asarray(root.matrix_world, dtype=np.float64)
        for object_id, root in roots.items()
    }
    render_state = {
        "camera": scene.camera,
        "engine": scene.render.engine,
        "resolution_x": scene.render.resolution_x,
        "resolution_y": scene.render.resolution_y,
        "resolution_percentage": scene.render.resolution_percentage,
        "use_border": scene.render.use_border,
        "filepath": scene.render.filepath,
        "use_file_extension": scene.render.use_file_extension,
        "file_format": scene.render.image_settings.file_format,
        "color_mode": scene.render.image_settings.color_mode,
        "color_depth": scene.render.image_settings.color_depth,
        "film_transparent": scene.render.film_transparent,
        "cycles_samples": scene.cycles.samples,
        "view_transform": scene.view_settings.view_transform,
        "exposure": scene.view_settings.exposure,
        "gamma": scene.view_settings.gamma,
    }
    object_state = {
        obj.as_pointer(): (
            obj,
            int(obj.pass_index),
            bool(obj.hide_render),
            tuple(obj.color),
        )
        for obj in scene.objects
    }
    # A well-separated RGB lattice is robust to small render/color roundoff
    # and leaves black available for the viewport background.
    levels = (0.1, 0.28, 0.46, 0.64, 0.82, 1.0)
    palette = [
        (red, green, blue)
        for red in levels
        for green in levels
        for blue in levels
    ]
    if len(render_groups) > len(palette):
        raise RuntimeError(
            "SceneProof visibility exceeded the color-ID palette"
        )
    render_colors = {
        group_id: palette[index]
        for index, group_id in enumerate(render_groups)
    }
    label_by_group = {
        group_id: index
        for index, group_id in enumerate(render_groups, start=1)
    }
    temporary_meshes = []
    temporary_materials = []
    original_mesh_data = {
        obj.as_pointer(): (obj, obj.data)
        for obj in scene.objects
        if obj.type == "MESH"
    }
    temporary_render_directory = tempfile.TemporaryDirectory(
        prefix="sceneproof_visibility_"
    )
    render_sequence = [0]

    def make_emission_material(name, color):
        material = bpy.data.materials.new(name=name)
        temporary_materials.append(material)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        output = nodes.new(type="ShaderNodeOutputMaterial")
        emission = nodes.new(type="ShaderNodeEmission")
        emission.inputs["Color"].default_value = (*color, 1.0)
        emission.inputs["Strength"].default_value = 1.0
        material.node_tree.links.new(
            emission.outputs["Emission"], output.inputs["Surface"]
        )
        return material

    def render_color_ids():
        bpy.context.view_layer.update()
        sequence_index = render_sequence[0]
        output_path = Path(temporary_render_directory.name) / (
            f"color_ids_{sequence_index:03d}.png"
        )
        render_sequence[0] += 1
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        if not output_path.is_file():
            raise RuntimeError(
                "Blender did not write the visibility color-ID image: "
                f"{output_path}"
            )
        with Image.open(output_path) as rendered_image:
            if color_id_output_path and sequence_index == 0:
                persistent_path = Path(color_id_output_path).resolve()
                persistent_path.parent.mkdir(parents=True, exist_ok=True)
                rendered_image.convert("RGB").save(persistent_path)
            values = np.asarray(
                rendered_image.convert("RGB"), dtype=np.float32
            ) / 255.0
        if values.shape != (resolution, resolution, 3):
            raise RuntimeError(
                "unexpected visibility image shape: "
                f"{values.shape}"
            )
        expected = np.asarray(
            [render_colors[group_id] for group_id in render_groups],
            dtype=np.float32,
        )
        return decode_color_id_image(values, expected, tolerance=0.07)

    def physical_components(candidate):
        if not candidate_physical_audit:
            return {}
        with torch.no_grad():
            _, collision_values = oriented_penetration_loss(
                candidate,
                local_corners,
                collision_pairs,
                footprint_hull_sizes,
            )
            _, contact_gaps = support_contact_loss(
                candidate,
                local_corners,
                support_pairs,
                fixed_support_indices,
                fixed_support_heights,
            )
            _, containment_squared = support_planar_containment_loss(
                candidate,
                local_corners,
                containment_pairs,
                footprint_hull_sizes,
            )
            _, boundary_errors = room_boundary_loss(
                candidate,
                local_corners,
                boundary_object_indices,
                boundary_points,
                boundary_normals,
            )
        return {
            "collision": collision_values.detach(),
            "support_contact": contact_gaps.abs().detach(),
            "support_containment": containment_squared.clamp_min(0).sqrt().detach(),
            "boundary": boundary_errors.detach(),
        }

    try:
        scene.camera = scene_camera
        scene.render.engine = "CYCLES"
        scene.render.resolution_x = int(resolution)
        scene.render.resolution_y = int(resolution)
        scene.render.resolution_percentage = 100
        scene.render.use_border = False
        scene.render.film_transparent = True
        scene.render.use_file_extension = True
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
        scene.cycles.samples = 1
        scene.view_settings.view_transform = "Raw"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        black_material = make_emission_material(
            "__sceneproof_visibility_black", (0.0, 0.0, 0.0)
        )
        color_materials = {
            group_id: make_emission_material(
                f"__sceneproof_visibility_{index}", color
            )
            for index, (group_id, color) in enumerate(
                render_colors.items(), start=1
            )
        }
        mesh_material = {
            obj.as_pointer(): black_material
            for obj in scene.objects
            if obj.type == "MESH"
        }
        for group_id, meshes in render_groups.items():
            for mesh in meshes:
                mesh_material[mesh.as_pointer()] = color_materials[group_id]
        # Work on temporary mesh datablocks so no original material slots or
        # polygon material indices are mutated, even when assets share data.
        for obj in scene.objects:
            if obj.type != "MESH":
                continue
            copied_data = obj.data.copy()
            temporary_meshes.append(copied_data)
            obj.data = copied_data
            copied_data.materials.clear()
            copied_data.materials.append(mesh_material[obj.as_pointer()])
            for polygon in copied_data.polygons:
                polygon.material_index = 0

        full_index = render_color_ids()
        label_to_object = {
            label_by_group[group_id]: render_display_names[group_id]
            for group_id in render_groups
        }
        records = {}
        observed_masks = {}
        isolated_masks = {}
        suspicious = []
        resampling = getattr(Image, "Resampling", Image).NEAREST
        for object_id in mesh_groups:
            label_index = label_by_group[object_id]
            mask_path = mask_root / f"{object_id}_mask.png"
            if not mask_path.is_file():
                records[object_id] = {
                    "color_id": list(render_colors[object_id]),
                    "status": "missing_s1_mask",
                }
                continue
            observed = np.asarray(
                Image.open(mask_path).convert("L").resize(
                    (resolution, resolution), resample=resampling
                )
            ) > 0
            observed_masks[object_id] = observed
            rendered = full_index == label_index
            metrics = binary_mask_metrics(rendered, observed)
            rendered_yx = np.argwhere(rendered)
            if len(rendered_yx):
                rendered_y0, rendered_x0 = rendered_yx.min(axis=0)
                rendered_y1, rendered_x1 = rendered_yx.max(axis=0) + 1
                rendered_cy, rendered_cx = rendered_yx.mean(axis=0)
                metrics["rendered_centroid_xy"] = [
                    float(rendered_cx), float(rendered_cy)
                ]
                metrics["rendered_bbox_xyxy"] = [
                    int(rendered_x0), int(rendered_y0),
                    int(rendered_x1), int(rendered_y1),
                ]
            else:
                metrics["rendered_centroid_xy"] = None
                metrics["rendered_bbox_xyxy"] = None
            metrics.update(
                {
                    "color_id": list(render_colors[object_id]),
                    "status": "measured",
                    "plane_sibling_member": object_id in component_ids,
                }
            )
            records[object_id] = metrics
            if (
                object_id in component_ids
                and metrics["rendered_visible_pixels"] < minimum_pixels
            ):
                suspicious.append(object_id)

        if annotated_color_id_output_path:
            if not color_id_output_path:
                raise ValueError(
                    "annotated color-ID output requires color_id_output_path"
                )
            annotated_path = Path(annotated_color_id_output_path).resolve()
            annotated_path.parent.mkdir(parents=True, exist_ok=True)
            annotated = Image.open(Path(color_id_output_path).resolve()).convert("RGB")
            draw = ImageDraw.Draw(annotated)
            for object_id in mesh_groups:
                pixels = np.argwhere(full_index == label_by_group[object_id])
                if not len(pixels):
                    continue
                center_y, center_x = np.median(pixels, axis=0)
                center_x, center_y = int(center_x), int(center_y)
                box = draw.textbbox((center_x, center_y), object_id)
                draw.rectangle(box, fill=(0, 0, 0))
                draw.text((center_x, center_y), object_id, fill=(255, 255, 255))
            annotated.save(annotated_path)

        scene_meshes = {
            obj.as_pointer(): obj
            for obj in scene.objects
            if obj.type == "MESH"
        }
        diagnostic_ids = set(suspicious)
        if joint_tangent_candidate_audit:
            for component in (sibling_audit or {}).get(
                "component_audits", []
            ):
                object_ids = component.get("object_ids", [])
                if any(object_id in diagnostic_ids for object_id in object_ids):
                    diagnostic_ids.update(
                        object_id
                        for object_id in object_ids
                        if object_id in mesh_groups
                        and object_id in observed_masks
                    )
        for object_id in sorted(diagnostic_ids):
            target_pointers = {
                mesh.as_pointer() for mesh in mesh_groups[object_id]
            }
            for pointer, mesh in scene_meshes.items():
                mesh.hide_render = pointer not in target_pointers
            isolated_index = render_color_ids()
            target_label = label_by_group[object_id]
            isolated_mask = isolated_index == target_label
            isolated_masks[object_id] = isolated_mask
            isolated_pixels = int(isolated_mask.sum())
            record = records[object_id]
            record["isolated_visible_pixels"] = isolated_pixels
            record["visibility_class"] = classify_visibility(
                record["rendered_visible_pixels"],
                isolated_pixels,
                minimum_pixels=minimum_pixels,
            )
            record["occlusion_attribution"] = attribute_occluders(
                full_index,
                isolated_mask,
                label_to_object,
                target_label=target_label,
            )

        # PLANE_ATTACH currently constrains distance/orientation to an
        # infinite plane.  Audit the missing finite-domain condition using
        # exact rendered mesh vertices: the child's projected footprint must
        # lie inside the host plane mesh's convex patch.  This is deliberately
        # witness-only; the returned minimum translation is never applied.
        finite_patch_audit = {
            "policy": "audit_only_exact_mesh_finite_plane_patch",
            "mutates_placement": False,
            "objects": [],
        }
        plane_ids = {
            binding.get("plane_id") for binding in (plane_bindings or [])
        }
        for binding in plane_bindings or []:
            child_id = binding.get("child_id")
            plane_id = binding.get("plane_id")
            record = {
                "child_id": child_id,
                "plane_id": plane_id,
                "status": "abstained",
            }
            child_root = roots.get(child_id)
            plane_root = bpy.data.objects.get(plane_id)
            if child_root is None or plane_root is None:
                record["reason"] = "missing_child_or_plane_mesh"
                finite_patch_audit["objects"].append(record)
                continue
            try:
                normal = np.asarray(binding["normal"], dtype=np.float64)
                normal /= max(float(np.linalg.norm(normal)), 1e-12)
                reference_axis = np.asarray((0.0, 0.0, 1.0))
                if abs(float(reference_axis @ normal)) > 0.95:
                    reference_axis = np.asarray((1.0, 0.0, 0.0))
                tangent = np.cross(reference_axis, normal)
                tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
                vertical = np.cross(normal, tangent)
                vertical /= max(float(np.linalg.norm(vertical)), 1e-12)
                host_vertices = hierarchy_world_vertices(plane_root)
                child_vertices = hierarchy_world_vertices(child_root)
                host_uv = np.column_stack(
                    (host_vertices @ tangent, host_vertices @ vertical)
                )
                child_uv = np.column_stack(
                    (child_vertices @ tangent, child_vertices @ vertical)
                )
                containment = minimum_translation_into_convex_polygon(
                    host_uv,
                    child_uv,
                    tolerance=1e-7,
                )
                translation_uv = containment.get("translation")
                translation_world = None
                if translation_uv is not None:
                    translation_world = (
                        tangent * float(translation_uv[0])
                        + vertical * float(translation_uv[1])
                    ).tolist()
                attribution = records.get(child_id, {}).get(
                    "occlusion_attribution", {}
                )
                dominant = attribution.get("dominant_occluder")
                record.update(
                    {
                        "status": "measured",
                        "reason": "finite_patch_measured",
                        "contained": containment["contained"],
                        "feasible": containment["feasible"],
                        "maximum_outside_distance_m": containment[
                            "maximum_outside_distance"
                        ],
                        "minimum_translation_uv_m": translation_uv,
                        "minimum_translation_world_m": translation_world,
                        "minimum_translation_norm_m": containment[
                            "translation_norm"
                        ],
                        "dominant_occluder": dominant,
                        "dominant_occluder_fraction": attribution.get(
                            "dominant_occluder_fraction"
                        ),
                        "cross_plane_occlusion": bool(
                            dominant in plane_ids and dominant != plane_id
                        ),
                    }
                )
            except (KeyError, ValueError, RuntimeError) as error:
                record["reason"] = "finite_patch_measurement_failed"
                record["error"] = str(error)
            finite_patch_audit["objects"].append(record)

        # Isolated diagnostics intentionally hide every non-target mesh.
        # Restore the full scene before evaluating any hypothetical repair.
        for obj, _, hide_render, _ in object_state.values():
            obj.hide_render = hide_render
        bpy.context.view_layer.update()

        tangent_audit = {
            "policy": "audit_only_mask_calibrated_per_object_plane_tangent",
            "enabled": bool(tangent_candidate_audit),
            "mutates_placement": False,
            "maximum_shift_m": float(tangent_maximum_shift_m),
            "visibility_noharm_tolerance": float(
                visibility_noharm_tolerance
            ),
            "minimum_recall_gain": float(minimum_recall_gain),
            "objects": [],
            "passing_candidates": 0,
        }
        if tangent_candidate_audit:
            physical_reference = physical_components(pose_matrices)
            id_to_index = {
                object_id: index
                for index, object_id in enumerate(ordered_ids)
            }
            binding_by_child = {
                binding.get("child_id"): binding
                for binding in (plane_bindings or [])
            }
            patch_by_child = {
                record.get("child_id"): record
                for record in finite_patch_audit["objects"]
            }
            for object_id in suspicious:
                baseline_record = records[object_id]
                attribution = baseline_record.get(
                    "occlusion_attribution", {}
                )
                binding = binding_by_child.get(object_id)
                patch_record = patch_by_child.get(object_id, {})
                object_audit = {
                    "object_id": object_id,
                    "accepted": False,
                    "reason": "not_cross_plane_occluded",
                    "candidates": [],
                }
                if (
                    binding is None
                    or not patch_record.get("cross_plane_occlusion", False)
                    or object_id not in isolated_masks
                ):
                    tangent_audit["objects"].append(object_audit)
                    continue

                normal = np.asarray(binding["normal"], dtype=np.float64)
                normal /= max(float(np.linalg.norm(normal)), 1e-12)
                tangent = np.cross(
                    np.asarray((0.0, 0.0, 1.0)), normal
                )
                if float(np.linalg.norm(tangent)) <= 1e-9:
                    object_audit["reason"] = "degenerate_wall_tangent"
                    tangent_audit["objects"].append(object_audit)
                    continue
                tangent /= float(np.linalg.norm(tangent))
                observed_yx = np.argwhere(observed_masks[object_id]).mean(
                    axis=0
                )
                isolated_yx = np.argwhere(isolated_masks[object_id]).mean(
                    axis=0
                )
                observed_xy = np.asarray(
                    (observed_yx[1], observed_yx[0]), dtype=np.float64
                )
                isolated_xy = np.asarray(
                    (isolated_yx[1], isolated_yx[0]), dtype=np.float64
                )
                vertices = hierarchy_world_vertices(roots[object_id])
                centre = vertices.mean(axis=0)

                def project_pixel(point):
                    coordinate = world_to_camera_view(
                        scene, scene_camera, Vector(tuple(point))
                    )
                    return np.asarray(
                        (
                            float(coordinate.x) * resolution,
                            (1.0 - float(coordinate.y)) * resolution,
                        ),
                        dtype=np.float64,
                    )

                probe_m = 0.01
                jacobian = (
                    project_pixel(centre + tangent * probe_m)
                    - project_pixel(centre)
                ) / probe_m
                denominator = float(jacobian @ jacobian)
                if denominator <= 1e-9:
                    object_audit["reason"] = "degenerate_image_jacobian"
                    tangent_audit["objects"].append(object_audit)
                    continue
                predicted = float(
                    (observed_xy - isolated_xy) @ jacobian / denominator
                )
                predicted = float(
                    np.clip(
                        predicted,
                        -tangent_maximum_shift_m,
                        tangent_maximum_shift_m,
                    )
                )
                direction = 1.0 if predicted >= 0 else -1.0
                candidate_offsets = sorted(
                    {
                        round(
                            float(
                                np.clip(
                                    value,
                                    -tangent_maximum_shift_m,
                                    tangent_maximum_shift_m,
                                )
                            ),
                            9,
                        )
                        for value in (
                            predicted,
                            predicted * 0.5,
                            direction * min(0.175, tangent_maximum_shift_m),
                            direction * tangent_maximum_shift_m,
                            -direction * min(0.175, tangent_maximum_shift_m),
                        )
                        if abs(value) > 1e-6
                    }
                )
                object_audit.update(
                    {
                        "reason": "no_passing_candidate",
                        "host_plane": binding.get("plane_id"),
                        "dominant_occluder": attribution.get(
                            "dominant_occluder"
                        ),
                        "observed_centroid_xy": observed_xy.tolist(),
                        "isolated_centroid_xy": isolated_xy.tolist(),
                        "tangent_world": tangent.tolist(),
                        "image_jacobian_px_per_m": jacobian.tolist(),
                        "predicted_offset_m": predicted,
                    }
                )
                base_matrix = roots[object_id].matrix_world.copy()
                target_index = id_to_index[object_id]
                target_label = label_by_group[object_id]
                plane_root = bpy.data.objects.get(binding.get("plane_id"))
                passing = []
                try:
                    for offset in candidate_offsets:
                        candidate_matrix = base_matrix.copy()
                        candidate_matrix.translation += Vector(
                            tuple(tangent * float(offset))
                        )
                        roots[object_id].matrix_world = candidate_matrix
                        bpy.context.view_layer.update()
                        candidate_labels = render_color_ids()
                        target_metrics = binary_mask_metrics(
                            candidate_labels == target_label,
                            observed_masks[object_id],
                        )
                        visibility_failures = []
                        for observed_id, observed_mask in observed_masks.items():
                            baseline = records.get(observed_id, {})
                            if baseline.get("status") != "measured":
                                continue
                            candidate_metrics = binary_mask_metrics(
                                candidate_labels
                                == label_by_group[observed_id],
                                observed_mask,
                            )
                            if (
                                candidate_metrics["iou"]
                                < float(baseline["iou"])
                                - float(visibility_noharm_tolerance)
                                or candidate_metrics["recall"]
                                < float(baseline["recall"])
                                - float(visibility_noharm_tolerance)
                            ):
                                visibility_failures.append(observed_id)

                        candidate_pose = pose_matrices.clone()
                        candidate_pose[target_index, :3, 3].add_(
                            torch.as_tensor(
                                tangent * float(offset),
                                dtype=candidate_pose.dtype,
                                device=candidate_pose.device,
                            )
                        )
                        candidate_physical = physical_components(
                            candidate_pose
                        )
                        physical_failures = {}
                        for family, reference_values in (
                            physical_reference.items()
                        ):
                            delta = (
                                candidate_physical[family] - reference_values
                            )
                            failed_rows = (
                                torch.nonzero(
                                    delta > float(physical_tolerance),
                                    as_tuple=False,
                                )
                                .flatten()
                                .detach()
                                .cpu()
                                .tolist()
                            )
                            if failed_rows:
                                physical_failures[family] = failed_rows

                        patch_contained = False
                        if plane_root is not None:
                            reference_axis = np.asarray((0.0, 0.0, 1.0))
                            vertical = np.cross(normal, tangent)
                            host_vertices = hierarchy_world_vertices(
                                plane_root
                            )
                            child_vertices = hierarchy_world_vertices(
                                roots[object_id]
                            )
                            host_uv = np.column_stack(
                                (
                                    host_vertices @ tangent,
                                    host_vertices @ vertical,
                                )
                            )
                            child_uv = np.column_stack(
                                (
                                    child_vertices @ tangent,
                                    child_vertices @ vertical,
                                )
                            )
                            patch_contained = bool(
                                minimum_translation_into_convex_polygon(
                                    host_uv,
                                    child_uv,
                                    tolerance=1e-7,
                                )["contained"]
                            )
                        recall_gain = float(
                            target_metrics["recall"]
                            - baseline_record["recall"]
                        )
                        candidate_record = {
                            "offset_m": float(offset),
                            "visible_pixels": int(
                                target_metrics["rendered_visible_pixels"]
                            ),
                            "iou": float(target_metrics["iou"]),
                            "recall": float(target_metrics["recall"]),
                            "recall_gain": recall_gain,
                            "finite_patch_contained": patch_contained,
                            "visibility_failures": visibility_failures,
                            "physical_failures": physical_failures,
                        }
                        candidate_record["passed"] = bool(
                            target_metrics["rendered_visible_pixels"]
                            >= minimum_pixels
                            and recall_gain >= float(minimum_recall_gain)
                            and patch_contained
                            and not visibility_failures
                            and not physical_failures
                        )
                        object_audit["candidates"].append(candidate_record)
                        if candidate_record["passed"]:
                            passing.append(candidate_record)
                finally:
                    roots[object_id].matrix_world = base_matrix
                    bpy.context.view_layer.update()
                if passing:
                    selected = max(
                        passing,
                        key=lambda candidate: (
                            candidate["recall"],
                            candidate["iou"],
                            -abs(candidate["offset_m"]),
                        ),
                    )
                    object_audit["accepted"] = True
                    object_audit["reason"] = "passing_candidate_found"
                    object_audit["would_select_offset_m"] = selected[
                        "offset_m"
                    ]
                    tangent_audit["passing_candidates"] += len(passing)
                tangent_audit["objects"].append(object_audit)

        joint_tangent_audit = {
            "policy": (
                "audit_only_component_assignment_joint_plane_tangent"
            ),
            "enabled": bool(joint_tangent_candidate_audit),
            "mutates_placement": False,
            "maximum_shift_m": float(tangent_maximum_shift_m),
            "components": [],
            "passing_candidates": 0,
        }
        if joint_tangent_candidate_audit:
            physical_reference = physical_components(pose_matrices)
            id_to_index = {
                object_id: index
                for index, object_id in enumerate(ordered_ids)
            }
            binding_by_child = {
                binding.get("child_id"): binding
                for binding in (plane_bindings or [])
            }
            for component in (sibling_audit or {}).get(
                "component_audits", []
            ):
                object_ids = [
                    object_id
                    for object_id in component.get("object_ids", [])
                    if object_id in observed_masks
                    and object_id in isolated_masks
                    and object_id in binding_by_child
                ]
                component_audit = {
                    "object_ids": object_ids,
                    "accepted": False,
                    "reason": "insufficient_cross_plane_component",
                    "candidates": [],
                }
                if len(object_ids) < 2:
                    continue
                bindings = [binding_by_child[name] for name in object_ids]
                host_ids = {binding.get("plane_id") for binding in bindings}
                cross_plane_ids = [
                    name
                    for name in object_ids
                    if records[name]
                    .get("occlusion_attribution", {})
                    .get("dominant_occluder")
                    in plane_ids - host_ids
                ]
                if len(host_ids) != 1 or not cross_plane_ids:
                    continue
                host_id = next(iter(host_ids))
                host_root = bpy.data.objects.get(host_id)
                if host_root is None:
                    component_audit["reason"] = "missing_host_plane"
                    joint_tangent_audit["components"].append(
                        component_audit
                    )
                    continue
                normal = np.asarray(bindings[0]["normal"], dtype=np.float64)
                normal /= max(float(np.linalg.norm(normal)), 1e-12)
                tangent = np.cross(
                    np.asarray((0.0, 0.0, 1.0)), normal
                )
                tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
                vertical = np.cross(normal, tangent)
                observed_centroids = []
                isolated_centroids = []
                observed_areas = []
                isolated_areas = []
                jacobians = []
                for object_id in object_ids:
                    observed_yx = np.argwhere(
                        observed_masks[object_id]
                    ).mean(axis=0)
                    isolated_yx = np.argwhere(
                        isolated_masks[object_id]
                    ).mean(axis=0)
                    observed_centroids.append(
                        np.asarray((observed_yx[1], observed_yx[0]))
                    )
                    isolated_centroids.append(
                        np.asarray((isolated_yx[1], isolated_yx[0]))
                    )
                    observed_areas.append(
                        int(observed_masks[object_id].sum())
                    )
                    isolated_areas.append(
                        int(isolated_masks[object_id].sum())
                    )
                    centre = hierarchy_world_vertices(
                        roots[object_id]
                    ).mean(axis=0)

                    def pixel(point):
                        coordinate = world_to_camera_view(
                            scene, scene_camera, Vector(tuple(point))
                        )
                        return np.asarray(
                            (
                                float(coordinate.x) * resolution,
                                (1.0 - float(coordinate.y)) * resolution,
                            )
                        )

                    jacobians.append(
                        (pixel(centre + tangent * 0.01) - pixel(centre))
                        / 0.01
                    )

                count = len(object_ids)
                costs = np.zeros((count, count), dtype=np.float64)
                offsets = np.zeros((count, count), dtype=np.float64)
                for mesh_index in range(count):
                    jacobian = jacobians[mesh_index]
                    denominator = max(float(jacobian @ jacobian), 1e-9)
                    for mask_index in range(count):
                        error = (
                            observed_centroids[mask_index]
                            - isolated_centroids[mesh_index]
                        )
                        raw_offset = float(
                            error @ jacobian / denominator
                        )
                        offset = float(
                            np.clip(
                                raw_offset,
                                -tangent_maximum_shift_m,
                                tangent_maximum_shift_m,
                            )
                        )
                        residual = error - jacobian * offset
                        area_error = math.log(
                            max(isolated_areas[mesh_index], 1)
                            / max(observed_areas[mask_index], 1)
                        )
                        overflow = max(
                            abs(raw_offset) - tangent_maximum_shift_m,
                            0.0,
                        )
                        costs[mesh_index, mask_index] = (
                            float(residual @ residual)
                            + 64.0 * area_error * area_error
                            + 10000.0 * overflow * overflow
                        )
                        offsets[mesh_index, mask_index] = offset
                mesh_rows, mask_rows = linear_sum_assignment(costs)
                mask_by_mesh = {
                    int(mesh_index): int(mask_index)
                    for mesh_index, mask_index in zip(mesh_rows, mask_rows)
                }
                assignment = {
                    object_ids[mesh_index]: object_ids[mask_index]
                    for mesh_index, mask_index in mask_by_mesh.items()
                }
                desired = np.asarray(
                    [
                        offsets[index, mask_by_mesh[index]]
                        for index in range(count)
                    ],
                    dtype=np.float64,
                )
                # Under hard occlusion the local camera-centroid model can
                # point toward a target mask while remaining behind the
                # occluding plane.  Prefer an empirically rendered one-sided
                # tangent witness when it has positive recall gain.  A witness
                # may have failed collision/no-harm in isolation: the joint
                # solve is precisely where its sibling separators move too,
                # and all gates are re-evaluated on the complete trial.
                witness_overrides = {}
                tangent_records = {
                    record.get("object_id"): record
                    for record in tangent_audit.get("objects", [])
                }
                for index, object_id in enumerate(object_ids):
                    witness_candidates = [
                        candidate
                        for candidate in tangent_records.get(
                            object_id, {}
                        ).get("candidates", [])
                        if candidate.get("finite_patch_contained", False)
                        and float(candidate.get("recall_gain", 0.0)) > 1e-9
                    ]
                    if not witness_candidates:
                        continue
                    witness = max(
                        witness_candidates,
                        key=lambda candidate: (
                            candidate["recall_gain"],
                            candidate["iou"],
                            -abs(candidate["offset_m"]),
                        ),
                    )
                    desired[index] = float(witness["offset_m"])
                    witness_overrides[object_id] = {
                        "offset_m": float(witness["offset_m"]),
                        "recall_gain": float(witness["recall_gain"]),
                        "isolated_physical_failures": witness.get(
                            "physical_failures", {}
                        ),
                        "isolated_visibility_failures": witness.get(
                            "visibility_failures", []
                        ),
                    }
                component_audit.update(
                    {
                        "reason": "no_passing_candidate",
                        "host_plane": host_id,
                        "cross_plane_object_ids": cross_plane_ids,
                        "assignment": assignment,
                        "rendered_witness_overrides": witness_overrides,
                        "desired_offsets_m": {
                            object_id: float(desired[index])
                            for index, object_id in enumerate(object_ids)
                        },
                    }
                )
                component_labels = [
                    label_by_group[object_id] for object_id in object_ids
                ]
                observed_union = np.logical_or.reduce(
                    [observed_masks[name] for name in object_ids]
                )
                baseline_union_metrics = binary_mask_metrics(
                    np.isin(full_index, component_labels), observed_union
                )
                baseline_assigned_metrics = {}
                for object_id in object_ids:
                    mask_id = assignment[object_id]
                    baseline_assigned_metrics[object_id] = (
                        binary_mask_metrics(
                            full_index == label_by_group[object_id],
                            observed_masks[mask_id],
                        )
                    )
                base_matrices = {
                    name: roots[name].matrix_world.copy()
                    for name in object_ids
                }
                grid_values_by_object = {}
                grid_seed_source_by_object = {}
                for component_index, object_id in enumerate(object_ids):
                    values_for_object = {0.0}
                    witness = witness_overrides.get(object_id)
                    if witness is not None:
                        seed = float(witness["offset_m"])
                        seed_source = "rendered_witness"
                    else:
                        seed = float(desired[component_index])
                        seed_source = "assignment_compensation"
                    magnitude = min(
                        abs(seed), tangent_maximum_shift_m
                    )
                    for fraction in (0.25, 0.5, 1.0):
                        value = min(
                            magnitude * fraction,
                            tangent_maximum_shift_m,
                        )
                        values_for_object.update((-value, value))
                    grid_values_by_object[object_id] = sorted(
                        round(value, 9) for value in values_for_object
                    )
                    grid_seed_source_by_object[object_id] = seed_source
                feasible_trials = []
                witness_weights = np.asarray(
                    [
                        max(
                            float(
                                witness_overrides.get(object_id, {}).get(
                                    "recall_gain", 0.0
                                )
                            ),
                            0.01,
                        )
                        for object_id in object_ids
                    ],
                    dtype=np.float64,
                )
                for values in product(
                    *[
                        grid_values_by_object[object_id]
                        for object_id in object_ids
                    ]
                ):
                    applied = np.asarray(values, dtype=np.float64)
                    if float(np.max(np.abs(applied))) <= 1e-9:
                        continue
                    candidate_pose = pose_matrices.clone()
                    for index, object_id in enumerate(object_ids):
                        candidate_pose[
                            id_to_index[object_id], :3, 3
                        ].add_(
                            torch.as_tensor(
                                tangent * float(applied[index]),
                                dtype=candidate_pose.dtype,
                                device=candidate_pose.device,
                            )
                        )
                    candidate_physical = physical_components(candidate_pose)
                    if any(
                        bool(
                            (
                                values_tensor
                                > physical_reference[family]
                                + float(physical_tolerance)
                            ).any().item()
                        )
                        for family, values_tensor in candidate_physical.items()
                    ):
                        continue
                    proxy_cost = float(
                        np.sum(witness_weights * (applied - desired) ** 2)
                        + 0.001 * np.sum(applied**2)
                    )
                    feasible_trials.append((proxy_cost, applied))
                feasible_trials.sort(
                    key=lambda item: (
                        item[0],
                        float(np.max(np.abs(item[1]))),
                        tuple(float(value) for value in item[1]),
                    )
                )
                component_audit[
                    "physics_feasible_grid_candidates_total"
                ] = len(feasible_trials)
                feasible_trials = feasible_trials[:8]
                component_audit["grid_values_m_by_object"] = (
                    grid_values_by_object
                )
                component_audit["grid_seed_source_by_object"] = (
                    grid_seed_source_by_object
                )
                component_audit["grid_candidates_total"] = int(
                    np.prod(
                        [
                            len(grid_values_by_object[object_id])
                            for object_id in object_ids
                        ]
                    )
                    - 1
                )
                component_audit["rendered_grid_candidates"] = len(
                    feasible_trials
                )
                passing = []
                try:
                    for trial_index, (_, applied) in enumerate(
                        feasible_trials
                    ):
                        candidate_pose = pose_matrices.clone()
                        for index, object_id in enumerate(object_ids):
                            matrix = base_matrices[object_id].copy()
                            matrix.translation += Vector(
                                tuple(tangent * float(applied[index]))
                            )
                            roots[object_id].matrix_world = matrix
                            candidate_pose[
                                id_to_index[object_id], :3, 3
                            ].add_(
                                torch.as_tensor(
                                    tangent * float(applied[index]),
                                    dtype=candidate_pose.dtype,
                                    device=candidate_pose.device,
                                )
                            )
                        bpy.context.view_layer.update()
                        candidate_labels = render_color_ids()
                        union_metrics = binary_mask_metrics(
                            np.isin(candidate_labels, component_labels),
                            observed_union,
                        )
                        assigned_metrics = {}
                        recovered_object_ids = []
                        visible_member_failures = []
                        for object_id in object_ids:
                            mask_id = assignment[object_id]
                            metrics = binary_mask_metrics(
                                candidate_labels
                                == label_by_group[object_id],
                                observed_masks[mask_id],
                            )
                            assigned_metrics[object_id] = {
                                "mask_id": mask_id,
                                "visible_pixels": metrics[
                                    "rendered_visible_pixels"
                                ],
                                "iou": metrics["iou"],
                                "recall": metrics["recall"],
                                "recall_gain": float(
                                    metrics["recall"]
                                    - baseline_assigned_metrics[object_id][
                                        "recall"
                                    ]
                                ),
                            }
                            baseline_member = baseline_assigned_metrics[
                                object_id
                            ]
                            if (
                                baseline_member["rendered_visible_pixels"]
                                >= minimum_pixels
                                and (
                                    metrics["iou"]
                                    < baseline_member["iou"]
                                    - float(visibility_noharm_tolerance)
                                    or metrics["recall"]
                                    < baseline_member["recall"]
                                    - float(visibility_noharm_tolerance)
                                )
                            ):
                                visible_member_failures.append(object_id)
                            if (
                                baseline_member["rendered_visible_pixels"]
                                < minimum_pixels
                                and metrics["rendered_visible_pixels"]
                                >= minimum_pixels
                                and assigned_metrics[object_id]["recall_gain"]
                                >= float(minimum_recall_gain)
                            ):
                                recovered_object_ids.append(object_id)
                        visibility_failures = []
                        for observed_id, observed_mask in observed_masks.items():
                            if observed_id in object_ids:
                                continue
                            baseline = records.get(observed_id, {})
                            if baseline.get("status") != "measured":
                                continue
                            metrics = binary_mask_metrics(
                                candidate_labels
                                == label_by_group[observed_id],
                                observed_mask,
                            )
                            if (
                                metrics["iou"]
                                < float(baseline["iou"])
                                - float(visibility_noharm_tolerance)
                                or metrics["recall"]
                                < float(baseline["recall"])
                                - float(visibility_noharm_tolerance)
                            ):
                                visibility_failures.append(observed_id)
                        candidate_physical = physical_components(
                            candidate_pose
                        )
                        physical_failures = {}
                        for family, reference_values in (
                            physical_reference.items()
                        ):
                            delta = (
                                candidate_physical[family] - reference_values
                            )
                            rows = (
                                torch.nonzero(
                                    delta > float(physical_tolerance),
                                    as_tuple=False,
                                )
                                .flatten()
                                .detach()
                                .cpu()
                                .tolist()
                            )
                            if rows:
                                physical_failures[family] = rows
                        host_vertices = hierarchy_world_vertices(host_root)
                        host_uv = np.column_stack(
                            (
                                host_vertices @ tangent,
                                host_vertices @ vertical,
                            )
                        )
                        patch_failures = []
                        for object_id in object_ids:
                            vertices = hierarchy_world_vertices(
                                roots[object_id]
                            )
                            child_uv = np.column_stack(
                                (
                                    vertices @ tangent,
                                    vertices @ vertical,
                                )
                            )
                            if not minimum_translation_into_convex_polygon(
                                host_uv, child_uv, tolerance=1e-7
                            )["contained"]:
                                patch_failures.append(object_id)
                        union_recall_gain = float(
                            union_metrics["recall"]
                            - baseline_union_metrics["recall"]
                        )
                        unresolved_object_ids = [
                            object_id
                            for object_id in object_ids
                            if baseline_assigned_metrics[object_id][
                                "rendered_visible_pixels"
                            ]
                            < minimum_pixels
                            and object_id not in recovered_object_ids
                        ]
                        candidate_record = {
                            "trial_index": int(trial_index),
                            "scale": None,
                            "offsets_m": {
                                object_id: float(applied[index])
                                for index, object_id in enumerate(object_ids)
                            },
                            "union_iou": float(union_metrics["iou"]),
                            "union_recall": float(union_metrics["recall"]),
                            "union_recall_gain": union_recall_gain,
                            "assigned_metrics": assigned_metrics,
                            "recovered_object_ids": recovered_object_ids,
                            "unresolved_object_ids": unresolved_object_ids,
                            "visible_member_failures": (
                                visible_member_failures
                            ),
                            "visibility_failures": visibility_failures,
                            "physical_failures": physical_failures,
                            "finite_patch_failures": patch_failures,
                        }
                        candidate_record["passed"] = bool(
                            union_recall_gain >= float(minimum_recall_gain)
                            and bool(recovered_object_ids)
                            and not visible_member_failures
                            and not visibility_failures
                            and not physical_failures
                            and not patch_failures
                        )
                        component_audit["candidates"].append(
                            candidate_record
                        )
                        if candidate_record["passed"]:
                            passing.append(candidate_record)
                finally:
                    for object_id, matrix in base_matrices.items():
                        roots[object_id].matrix_world = matrix
                    bpy.context.view_layer.update()
                if passing:
                    selected = max(
                        passing,
                        key=lambda candidate: (
                            candidate["union_recall"],
                            candidate["union_iou"],
                            -max(
                                abs(value)
                                for value in candidate["offsets_m"].values()
                            ),
                        ),
                    )
                    component_audit["accepted"] = True
                    component_audit["reason"] = "passing_candidate_found"
                    component_audit["would_select"] = selected
                    joint_tangent_audit["passing_candidates"] += len(
                        passing
                    )
                joint_tangent_audit["components"].append(component_audit)

        visible_side_audit = {
            "policy": "audit_only_exact_mesh_visible_side_normal_sweep",
            "enabled": bool(visible_side_candidate_audit),
            "mutates_placement": False,
            "clearance_m": float(visible_side_clearance_m),
            "attachment_tolerance_m": float(
                visible_side_attachment_tolerance_m
            ),
            "maximum_shift_m": float(visible_side_maximum_shift_m),
            "visibility_noharm_tolerance": float(
                visibility_noharm_tolerance
            ),
            "minimum_recall_gain": float(minimum_recall_gain),
            "physical_tolerance": float(physical_tolerance),
            "objects": [],
            "passing_candidates": 0,
        }
        if visible_side_candidate_audit:
            physical_reference = physical_components(pose_matrices)
            id_to_index = {
                object_id: index
                for index, object_id in enumerate(ordered_ids)
            }
            camera_position = np.asarray(
                scene_camera.matrix_world.translation, dtype=np.float64
            )
            binding_by_child = {}
            for binding in plane_bindings:
                binding_by_child.setdefault(binding["child_id"], binding)

            for object_id in sorted(component_ids):
                baseline_record = records.get(object_id, {})
                if (
                    baseline_record.get("status") != "measured"
                    or baseline_record.get("rendered_visible_pixels", 0)
                    >= minimum_pixels
                ):
                    continue
                binding = binding_by_child.get(object_id)
                root = roots.get(object_id)
                plane_root = (
                    bpy.data.objects.get(binding["plane_id"])
                    if binding is not None
                    else None
                )
                object_audit = {
                    "object_id": object_id,
                    "plane_id": (
                        binding.get("plane_id") if binding else None
                    ),
                    "baseline_visible_pixels": int(
                        baseline_record.get("rendered_visible_pixels", 0)
                    ),
                    "baseline_iou": float(
                        baseline_record.get("iou", 0.0)
                    ),
                    "baseline_recall": float(
                        baseline_record.get("recall", 0.0)
                    ),
                    "candidates": [],
                    "would_select_shift_m": None,
                    "reason": "missing_plane_binding_or_mesh",
                }
                if (
                    binding is None
                    or root is None
                    or plane_root is None
                    or object_id not in id_to_index
                    or object_id not in observed_masks
                ):
                    visible_side_audit["objects"].append(object_audit)
                    continue

                normal = np.asarray(binding["normal"], dtype=np.float64)
                normal_length = float(np.linalg.norm(normal))
                if normal_length <= 1e-9:
                    object_audit["reason"] = "degenerate_plane_normal"
                    visible_side_audit["objects"].append(object_audit)
                    continue
                normal /= normal_length
                plane_vertices = hierarchy_world_vertices(plane_root)
                plane_centre = plane_vertices.mean(axis=0)
                if float(np.dot(camera_position - plane_centre, normal)) < 0:
                    normal = -normal
                visible_surface_coordinate = float(
                    np.max(plane_vertices @ normal)
                )
                object_vertices = hierarchy_world_vertices(root)
                signed_distances = (
                    object_vertices @ normal - visible_surface_coordinate
                )
                first_percentile = float(
                    np.quantile(signed_distances, 0.01)
                )
                minimum_shift = max(
                    0.0,
                    float(visible_side_clearance_m) - first_percentile,
                )
                object_audit.update(
                    {
                        "reason": "no_passing_candidate",
                        "visible_normal": normal.tolist(),
                        "mesh_vertex_count": int(object_vertices.shape[0]),
                        "plane_vertex_count": int(plane_vertices.shape[0]),
                        "signed_distance_q01_m": first_percentile,
                        "minimum_visible_side_shift_m": minimum_shift,
                        "trust_region_exceeded": (
                            minimum_shift
                            > float(visible_side_maximum_shift_m) + 1e-12
                        ),
                    }
                )
                if object_audit["trust_region_exceeded"]:
                    object_audit["reason"] = "trust_region_exceeded"
                    visible_side_audit["objects"].append(object_audit)
                    continue

                candidate_shifts = sorted(
                    {
                        round(minimum_shift + addition, 9)
                        for addition in (0.0, 0.005, 0.01, 0.02)
                        if minimum_shift + addition > 1e-9
                        and minimum_shift + addition
                        <= float(visible_side_maximum_shift_m) + 1e-12
                    }
                )
                base_matrix = root.matrix_world.copy()
                target_index = id_to_index[object_id]
                target_label = label_by_group[object_id]
                passing = []
                try:
                    for shift in candidate_shifts:
                        candidate_matrix = base_matrix.copy()
                        candidate_matrix.translation += Vector(
                            tuple(normal * float(shift))
                        )
                        root.matrix_world = candidate_matrix
                        bpy.context.view_layer.update()
                        candidate_labels = render_color_ids()
                        target_metrics = binary_mask_metrics(
                            candidate_labels == target_label,
                            observed_masks[object_id],
                        )
                        visibility_failures = []
                        for observed_id, observed_mask in observed_masks.items():
                            baseline = records.get(observed_id, {})
                            if baseline.get("status") != "measured":
                                continue
                            observed_label = label_by_group[observed_id]
                            candidate_metrics = binary_mask_metrics(
                                candidate_labels == observed_label,
                                observed_mask,
                            )
                            if (
                                candidate_metrics["iou"]
                                < float(baseline["iou"])
                                - float(visibility_noharm_tolerance)
                                or candidate_metrics["recall"]
                                < float(baseline["recall"])
                                - float(visibility_noharm_tolerance)
                            ):
                                visibility_failures.append(observed_id)

                        candidate_pose = pose_matrices.clone()
                        candidate_pose[target_index, :3, 3].add_(
                            torch.as_tensor(
                                normal * float(shift),
                                dtype=candidate_pose.dtype,
                                device=candidate_pose.device,
                            )
                        )
                        candidate_physical = physical_components(
                            candidate_pose
                        )
                        physical_failures = {}
                        physical_max_delta = {}
                        for family, reference_values in (
                            physical_reference.items()
                        ):
                            candidate_values = candidate_physical[family]
                            delta = candidate_values - reference_values
                            physical_max_delta[family] = float(
                                delta.max().item() if delta.numel() else 0.0
                            )
                            failed_rows = (
                                torch.nonzero(
                                    delta > float(physical_tolerance),
                                    as_tuple=False,
                                )
                                .flatten()
                                .detach()
                                .cpu()
                                .tolist()
                            )
                            if failed_rows:
                                physical_failures[family] = failed_rows

                        exact_gap = abs(
                            first_percentile
                            + float(shift)
                            - float(visible_side_clearance_m)
                        )
                        recall_gain = (
                            float(target_metrics["recall"])
                            - float(baseline_record["recall"])
                        )
                        candidate_record = {
                            "shift_m": float(shift),
                            "visible_pixels": int(
                                target_metrics["rendered_visible_pixels"]
                            ),
                            "iou": float(target_metrics["iou"]),
                            "precision": float(target_metrics["precision"]),
                            "recall": float(target_metrics["recall"]),
                            "recall_gain": recall_gain,
                            "exact_attachment_gap_m": exact_gap,
                            "visibility_failures": visibility_failures,
                            "physical_failures": physical_failures,
                            "physical_max_delta": physical_max_delta,
                        }
                        candidate_record["passed"] = bool(
                            target_metrics["rendered_visible_pixels"]
                            >= minimum_pixels
                            and recall_gain >= float(minimum_recall_gain)
                            and not visibility_failures
                            and not physical_failures
                            and exact_gap
                            <= float(visible_side_attachment_tolerance_m)
                        )
                        object_audit["candidates"].append(candidate_record)
                        if candidate_record["passed"]:
                            passing.append(candidate_record)
                finally:
                    root.matrix_world = base_matrix
                    bpy.context.view_layer.update()

                if passing:
                    selected = max(
                        passing,
                        key=lambda candidate: (
                            candidate["recall"],
                            candidate["iou"],
                            -candidate["shift_m"],
                        ),
                    )
                    object_audit["would_select_shift_m"] = selected[
                        "shift_m"
                    ]
                    object_audit["reason"] = "passing_candidate_found"
                    visible_side_audit["passing_candidates"] += len(passing)
                visible_side_audit["objects"].append(object_audit)
    finally:
        for obj, pass_index, hide_render, color in object_state.values():
            obj.pass_index = pass_index
            obj.hide_render = hide_render
            obj.color = color
        for obj, original_data in original_mesh_data.values():
            obj.data = original_data
        for object_id, matrix in pose_before.items():
            roots[object_id].matrix_world = Matrix(matrix.tolist())
        scene.camera = render_state["camera"]
        scene.render.engine = render_state["engine"]
        scene.render.resolution_x = render_state["resolution_x"]
        scene.render.resolution_y = render_state["resolution_y"]
        scene.render.resolution_percentage = render_state[
            "resolution_percentage"
        ]
        scene.render.use_border = render_state["use_border"]
        scene.render.filepath = render_state["filepath"]
        scene.render.use_file_extension = render_state["use_file_extension"]
        scene.render.image_settings.file_format = render_state["file_format"]
        scene.render.image_settings.color_mode = render_state["color_mode"]
        scene.render.image_settings.color_depth = render_state["color_depth"]
        scene.render.film_transparent = render_state["film_transparent"]
        scene.cycles.samples = render_state["cycles_samples"]
        scene.view_settings.view_transform = render_state["view_transform"]
        scene.view_settings.exposure = render_state["exposure"]
        scene.view_settings.gamma = render_state["gamma"]
        bpy.context.view_layer.update()
        for mesh in temporary_meshes:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        for material in temporary_materials:
            if material.users == 0:
                bpy.data.materials.remove(material)
        temporary_render_directory.cleanup()

    camera_after = np.asarray(scene_camera.matrix_world, dtype=np.float64)
    camera_delta = float(np.max(np.abs(camera_after - camera_before)))
    pose_delta = max(
        (
            float(
                np.max(
                    np.abs(
                        np.asarray(roots[object_id].matrix_world, dtype=np.float64)
                        - matrix
                    )
                )
            )
            for object_id, matrix in pose_before.items()
        ),
        default=0.0,
    )
    # Blender stores evaluated transform components at float32-like precision
    # even when the audit snapshots are compared in float64.  Assigning the
    # exact incumbent matrix back through ``matrix_world`` may therefore incur
    # a sub-0.1 mm round-trip difference, especially for parented objects.
    # Treat that representation error as restored, while keeping the camera
    # guard strict.  The tolerance is configurable for reproducible audits and
    # remains far below any physical/contact or visibility acceptance margin.
    pose_restore_tolerance = float(
        os.environ.get(
            "IMAGINARIUM_SCENEPROOF_AUDIT_POSE_RESTORE_TOLERANCE",
            "1e-4",
        )
    )
    if pose_restore_tolerance <= 0.0:
        raise ValueError(
            "IMAGINARIUM_SCENEPROOF_AUDIT_POSE_RESTORE_TOLERANCE "
            "must be positive"
        )
    if camera_delta > 1e-6 or pose_delta > pose_restore_tolerance:
        raise RuntimeError(
            "SceneProof mesh visibility audit changed scene state: "
            f"camera_delta={camera_delta:.8g}, pose_delta={pose_delta:.8g}, "
            f"pose_tolerance={pose_restore_tolerance:.8g}"
        )
    zero_visible = [
        object_id
        for object_id, record in records.items()
        if record.get("status") == "measured"
        and record["rendered_visible_pixels"] < minimum_pixels
    ]
    audit = {
        "schema_version": "sceneproof_mesh_visibility_audit_v7",
        "policy": "audit_only_all_scene_mesh_cycles_emission_png_color_id",
        "resolution": [int(resolution), int(resolution)],
        "minimum_visible_pixels": int(minimum_pixels),
        "objects_with_meshes": len(mesh_groups),
        "render_mesh_groups": len(render_groups),
        "unowned_mesh_groups": len(render_groups) - len(mesh_groups),
        "objects_measured": sum(
            record.get("status") == "measured" for record in records.values()
        ),
        "plane_sibling_object_ids": sorted(component_ids),
        "isolated_diagnostics": len(diagnostic_ids),
        "zero_visible_object_ids": zero_visible,
        "camera_max_abs_delta": camera_delta,
        "pose_max_abs_delta": pose_delta,
        "pose_restore_tolerance": pose_restore_tolerance,
        "pose_restoration_certified": bool(
            pose_delta <= pose_restore_tolerance
        ),
        "mutates_placement": False,
        "objects": records,
    }
    audit["finite_plane_patch_audit"] = finite_patch_audit
    audit["tangent_candidate_audit"] = tangent_audit
    audit["joint_tangent_candidate_audit"] = joint_tangent_audit
    audit["visible_side_candidate_audit"] = visible_side_audit
    print(
        "[SceneProof] Mesh visibility audit: "
        f"measured={audit['objects_measured']}, "
        f"plane_siblings={len(component_ids)}, "
        f"isolated={len(diagnostic_ids)}, zero_visible={len(zero_visible)}, "
        f"camera_delta={camera_delta:.8g}, pose_delta={pose_delta:.8g}",
        flush=True,
    )
    measured_patches = [
        record
        for record in finite_patch_audit["objects"]
        if record.get("status") == "measured"
    ]
    print(
        "[SceneProof] Finite plane-patch audit: "
        f"measured={len(measured_patches)}, "
        "outside="
        f"{sum(not row['contained'] for row in measured_patches)}, "
        "cross_plane_occluded="
        f"{sum(row['cross_plane_occlusion'] for row in measured_patches)}, "
        "mutates_placement=False",
        flush=True,
    )
    if tangent_candidate_audit:
        selectable = sum(
            record.get("accepted", False)
            for record in tangent_audit["objects"]
        )
        print(
            "[SceneProof] Per-object tangent candidate audit: "
            f"objects={len(tangent_audit['objects'])}, "
            f"selectable={selectable}, passing_candidates="
            f"{tangent_audit['passing_candidates']}, "
            "mutates_placement=False",
            flush=True,
        )
    if joint_tangent_candidate_audit:
        selectable = sum(
            component.get("accepted", False)
            for component in joint_tangent_audit["components"]
        )
        print(
            "[SceneProof] Joint tangent candidate audit: "
            f"components={len(joint_tangent_audit['components'])}, "
            f"selectable={selectable}, passing_candidates="
            f"{joint_tangent_audit['passing_candidates']}, "
            "mutates_placement=False",
            flush=True,
        )
    if visible_side_candidate_audit:
        selectable = [
            record
            for record in visible_side_audit["objects"]
            if record.get("would_select_shift_m") is not None
        ]
        print(
            "[SceneProof] Visible-side normal candidate audit: "
            f"objects={len(visible_side_audit['objects'])}, "
            f"selectable={len(selectable)}, "
            "passing_candidates="
            f"{visible_side_audit['passing_candidates']}, "
            "mutates_placement=False",
            flush=True,
        )
    return audit


def build_depth_reprojection_observations(
    obj_placement_info_json_path,
    ordered_ids,
    object_info,
    scene_camera,
    base_matrices,
    local_corners,
    *,
    width,
    height,
    dtype,
    device,
    image_only=False,
):
    """Load S1 masks and, unless image-only, S0 depth observations."""
    from PIL import Image

    result_root = Path(obj_placement_info_json_path).resolve().parent.parent
    depth_path = result_root / "S0_geometry_pred_results" / "depth.png"
    mask_root = result_root / "S1_scene_parsing_results" / "masks"
    if not image_only and not depth_path.is_file():
        raise FileNotFoundError(
            f"Depth-aware LayoutVLM requires {depth_path}"
        )
    if not mask_root.is_dir():
        raise FileNotFoundError(
            f"Depth-aware LayoutVLM requires {mask_root}"
        )

    depth_mm = (
        None
        if image_only
        else np.asarray(Image.open(depth_path), dtype=np.float32)
    )
    if depth_mm is not None and depth_mm.shape != (height, width):
        raise ValueError(
            "Depth/image resolution mismatch: "
            f"depth={depth_mm.shape}, expected={(height, width)}"
        )
    minimum_pixels = int(
        os.environ.get(
            (
                "IMAGINARIUM_SCENEPROOF_IMAGE_GAUGE_MIN_MASK_PIXELS"
                if image_only
                else "IMAGINARIUM_LAYOUTVLM_DEPTH_MIN_PIXELS"
            ),
            "64" if image_only else "800",
        )
    )
    focal_x = float(scene_camera.data.lens) / float(
        scene_camera.data.sensor_width
    ) * float(width)
    focal_y = focal_x
    world_to_camera = np.asarray(
        [
            list(row)
            for row in scene_camera.matrix_world.inverted()
        ],
        dtype=np.float32,
    )
    with torch.no_grad():
        initial_corners = transform_points(
            base_matrices,
            local_corners,
        ).detach().cpu().numpy()

    indices = []
    boxes = []
    depths = []
    visible_surface_depths = []
    surface_to_center_offsets = []
    weights = []
    size_enabled = []
    skipped = []
    for index, object_id in enumerate(ordered_ids):
        info = object_info.get(object_id, {})
        box = info.get("boxes")
        mask_path = mask_root / f"{object_id}_mask.png"
        if (
            (
                not image_only
                and (
                    not isinstance(box, (list, tuple))
                    or len(box) != 4
                )
            )
            or not mask_path.is_file()
        ):
            skipped.append((object_id, "missing_bbox_or_mask"))
            continue
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        mask_count = int(mask.sum())
        valid = (
            mask
            if image_only
            else mask & np.isfinite(depth_mm) & (depth_mm > 0)
        )
        valid_count = int(valid.sum())
        if valid_count < minimum_pixels:
            skipped.append(
                (
                    object_id,
                    (
                        f"mask_pixels={valid_count}"
                        if image_only
                        else f"valid_depth_pixels={valid_count}"
                    ),
                )
            )
            continue
        try:
            if image_only:
                rows, columns = np.nonzero(mask)
                clean_box = [
                    float(columns.min()),
                    float(rows.min()),
                    float(columns.max() + 1),
                    float(rows.max() + 1),
                ]
            else:
                clean_box = [float(value) for value in box]
        except (TypeError, ValueError):
            skipped.append((object_id, "invalid_bbox"))
            continue

        camera_corners = (
            initial_corners[index] @ world_to_camera[:3, :3].T
            + world_to_camera[:3, 3]
        )
        initial_corner_depths = -camera_corners[:, 2]
        initial_center_depth = float(initial_corner_depths.mean())
        initial_front_depth = float(initial_corner_depths.min())
        surface_to_center = max(
            0.0,
            initial_center_depth - initial_front_depth,
        )
        if image_only:
            visible_surface_depth = initial_front_depth
            observed_center_depth = initial_center_depth
        else:
            values = depth_mm[valid] / 1000.0
            low, high = np.quantile(values, [0.1, 0.9])
            trimmed = values[(values >= low) & (values <= high)]
            visible_surface_depth = float(
                np.median(trimmed if trimmed.size else values)
            )
            if (
                not np.isfinite(visible_surface_depth)
                or visible_surface_depth <= 0
            ):
                skipped.append((object_id, "invalid_robust_depth"))
                continue
            observed_center_depth = visible_surface_depth + surface_to_center

        indices.append(index)
        boxes.append(clean_box)
        depths.append(observed_center_depth)
        visible_surface_depths.append(visible_surface_depth)
        surface_to_center_offsets.append(surface_to_center)
        weights.append(
            min(1.0, max(0.1, math.sqrt(mask_count / 8000.0)))
        )
        size_enabled.append(not bool(info.get("mask_is_truncated", False)))

    for object_id, reason in skipped:
        print(
            f"[LayoutVLM] Skipping {'image' if image_only else 'depth'} "
            f"observation {object_id}: {reason}",
            flush=True,
        )
    print(
        f"[LayoutVLM] {'Image-gauge' if image_only else 'Depth'} observations built: "
        f"accepted={len(indices)}, skipped={len(skipped)}, "
        f"source={'S1_masks' if image_only else depth_path}",
        flush=True,
    )
    observations = {
        "indices": torch.as_tensor(
            indices, dtype=torch.long, device=device
        ),
        "boxes": torch.as_tensor(
            boxes, dtype=dtype, device=device
        ).reshape(-1, 4),
        "depths": torch.as_tensor(
            depths, dtype=dtype, device=device
        ),
        "visible_surface_depths": torch.as_tensor(
            visible_surface_depths, dtype=dtype, device=device
        ),
        "surface_to_center_offsets": torch.as_tensor(
            surface_to_center_offsets, dtype=dtype, device=device
        ),
        "weights": torch.as_tensor(
            weights, dtype=dtype, device=device
        ),
        "size_enabled": torch.as_tensor(
            size_enabled, dtype=torch.bool, device=device
        ),
        "world_to_camera": torch.as_tensor(
            world_to_camera, dtype=dtype, device=device
        ),
        "image_size": torch.as_tensor(
            [width, height, focal_x, focal_y],
            dtype=dtype,
            device=device,
        ),
    }
    with torch.no_grad():
        (
            _,
            reference_centre_errors,
            reference_size_errors,
            reference_relative_errors,
        ) = depth_aware_reprojection_loss(
            base_matrices,
            local_corners,
            observations["indices"],
            observations["boxes"],
            observations["depths"],
            observations["weights"],
            observations["size_enabled"],
            observations["world_to_camera"],
            observations["image_size"],
        )
    observations.update(
        {
            "reference_centre_errors": reference_centre_errors.detach(),
            "reference_size_errors": reference_size_errors.detach(),
            "reference_relative_errors":
                reference_relative_errors.detach(),
        }
    )
    return observations

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log = open(log_file, 'w', buffering=1)  # 行缓冲

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()  # 立即刷新终端输出
        self.log.write(message)
        self.log.flush()  # 立即刷新文件输出

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ========== Texture Application Helpers ==========
def parse_texture_name(filename):
    """
    解析纹理文件名，提取基础名称和分辨率。
    支持格式: Tiles15_COL_VAR1_6K.jpg -> (Tiles15, 6K)
    """
    name_without_ext = os.path.splitext(filename)[0]
    
    # 尝试提取分辨率 (e.g., 1K, 2k, 4K, 8k)
    resolution = ""
    res_match = re.search(r'[_\-\s](\d+[Kk])', name_without_ext)
    if res_match:
        resolution = res_match.group(1)
        
    # 定义可能的纹理类型标识符
    indicators = [
        'COL_VAR1', 'COL', 'diff', 'diffuse', 'albedo', 
        'NRM16', 'NRM', 'nor', 'normal', 
        'GLOSS', 'rough', 'roughness', 
        'DISP16', 'DISP', 'disp', 'displacement', 'BUMP16', 'BUMP',
        'AO', 'ao'
    ]
    
    base_name = name_without_ext
    
    # 尝试通过移除标识符来找到 base_name
    for ind in indicators:
        pattern = re.compile(re.escape(ind), re.IGNORECASE)
        if pattern.search(name_without_ext):
            parts = pattern.split(name_without_ext)
            if len(parts) > 0:
                candidate = parts[0].rstrip('_- ')
                if candidate:
                    base_name = candidate
                    return base_name, resolution

    # 如果没有匹配到标准标识符，尝试使用简单的正则 (fallback)
    match = re.match(r'(.+?)_(diff|rough|nor|disp|ao|metal|COL|GLOSS|NRM|BUMP)', name_without_ext, re.IGNORECASE)
    if match:
        return match.group(1), resolution
        
    return None, None

def find_related_textures(folder, base_name, resolution):
    """
    在文件夹中查找相关的纹理文件。
    """
    textures = {}
    texture_types = {
        'diff': ['COL', 'diff', 'diffuse', 'color', 'albedo', 'base'],
        'rough': ['rough', 'roughness'],
        'gloss': ['GLOSS', 'gloss'],
        'nor': ['NRM', 'nor', 'normal', 'norm'],
        'disp': ['DISP', 'disp', 'displacement', 'height', 'BUMP', 'bump'],
        'ao': ['AO', 'ao', 'ambient', 'occlusion'],
        'metal': ['metal', 'metallic', 'metalness']
    }
    
    if not os.path.exists(folder):
        return textures

    try:
        files = os.listdir(folder)
    except Exception as e:
        print(f"Error listing directory {folder}: {e}")
        return textures

    for filename in files:
        # 检查文件名是否包含 base_name
        if not filename.startswith(base_name):
            continue

        # 如果指定了分辨率，检查分辨率匹配
        if resolution and resolution.lower() not in filename.lower():
            continue
            
        filename_lower = filename.lower()
        
        # 检查每种纹理类型
        for tex_type, keywords in texture_types.items():
            if tex_type in textures: # 已经找到该类型的纹理
                continue
                
            for kw in keywords:
                # 简单的包含检查，区分大小写通常不需要，因为 filename_lower 是小写
                # 增加一些边界检查以避免误匹配 (例如 'color' 匹配 'discolor' - 不太可能但在代码中要注意)
                if kw.lower() in filename_lower:
                    textures[tex_type] = os.path.join(folder, filename)
                    break
    
    return textures

def apply_textures_to_object(obj, textures, texture_size=1.0):
    if not obj or obj.type != 'MESH':
        print(f"❌ Object {obj.name if obj else 'None'} is not a mesh!")
        return
    
    # ⭐ 核心：读取物体尺寸，计算UV缩放
    dimensions = obj.dimensions
    # If dimensions are 0 (e.g. empty mesh), avoid div by zero
    if dimensions.x == 0 or dimensions.y == 0:
         uv_scale = (1.0, 1.0, 1.0)
    else:
        uv_scale = (
            dimensions.x / texture_size,
            dimensions.y / texture_size,
            dimensions.z / texture_size
        )
    
    print(f"📏 物体尺寸: {dimensions.x:.2f}m × {dimensions.y:.2f}m × {dimensions.z:.2f}m")
    print(f"📐 贴图尺寸: {texture_size}m × {texture_size}m")
    print(f"🔢 UV缩放: ({uv_scale[0]:.2f}, {uv_scale[1]:.2f}, {uv_scale[2]:.2f})")
    
    # 创建材质
    base_name = os.path.splitext(os.path.basename(textures.get('diff', 'Material')))[0]
    mat_name = base_name.replace('_diff', '').replace('_8k', '').replace('_4k', '')
    
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # 纹理坐标系统
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-900, 300)
    
    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.location = (-700, 300)
    mapping.inputs['Scale'].default_value = uv_scale  # ⭐ 应用计算的缩放
    
    links.new(tex_coord.outputs['UV'], mapping.inputs['Vector'])
    
    # Principled BSDF
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (300, 300)
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (600, 300)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    y_offset = 500
    
    # 漫反射
    if 'diff' in textures:
        try:
            diff_tex = nodes.new(type='ShaderNodeTexImage')
            diff_tex.image = bpy.data.images.load(textures['diff'])
            diff_tex.location = (-300, y_offset)
            links.new(mapping.outputs['Vector'], diff_tex.inputs['Vector'])
            links.new(diff_tex.outputs['Color'], bsdf.inputs['Base Color'])
            print(f"✓ 漫反射: {os.path.basename(textures['diff'])}")
            y_offset -= 300
        except Exception as e:
            print(f"Failed to load diff texture: {e}")
    
    # 粗糙度 / 光泽度
    # 优先使用 Roughness，如果没有则使用 Glossiness 并反转
    rough_path = textures.get('rough')
    gloss_path = textures.get('gloss')
    
    if rough_path:
        try:
            rough_tex = nodes.new(type='ShaderNodeTexImage')
            rough_tex.image = bpy.data.images.load(rough_path)
            rough_tex.image.colorspace_settings.name = 'Non-Color'
            rough_tex.location = (-300, y_offset)
            links.new(mapping.outputs['Vector'], rough_tex.inputs['Vector'])
            links.new(rough_tex.outputs['Color'], bsdf.inputs['Roughness'])
            print(f"✓ 粗糙度: {os.path.basename(rough_path)}")
            y_offset -= 300
        except Exception as e:
            print(f"Failed to load rough texture: {e}")
    elif gloss_path:
        try:
            gloss_tex = nodes.new(type='ShaderNodeTexImage')
            gloss_tex.image = bpy.data.images.load(gloss_path)
            gloss_tex.image.colorspace_settings.name = 'Non-Color'
            gloss_tex.location = (-600, y_offset)
            
            invert_node = nodes.new(type='ShaderNodeInvert')
            invert_node.location = (-300, y_offset)
            invert_node.inputs['Fac'].default_value = 1.0
            
            links.new(mapping.outputs['Vector'], gloss_tex.inputs['Vector'])
            links.new(gloss_tex.outputs['Color'], invert_node.inputs['Color'])
            links.new(invert_node.outputs['Color'], bsdf.inputs['Roughness'])
            print(f"✓ 光泽度 (Inverted to Roughness): {os.path.basename(gloss_path)}")
            y_offset -= 300
        except Exception as e:
            print(f"Failed to load gloss texture: {e}")

    # 法线
    if 'nor' in textures:
        try:
            nor_tex = nodes.new(type='ShaderNodeTexImage')
            nor_tex.image = bpy.data.images.load(textures['nor'])
            nor_tex.image.colorspace_settings.name = 'Non-Color'
            nor_tex.location = (-600, y_offset)
            
            normal_map = nodes.new(type='ShaderNodeNormalMap')
            normal_map.location = (-300, y_offset)
            
            links.new(mapping.outputs['Vector'], nor_tex.inputs['Vector'])
            links.new(nor_tex.outputs['Color'], normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
            print(f"✓ 法线: {os.path.basename(textures['nor'])}")
            y_offset -= 300
        except Exception as e:
            print(f"Failed to load nor texture: {e}")
    
    # 置换
    if 'disp' in textures:
        try:
            disp_tex = nodes.new(type='ShaderNodeTexImage')
            disp_tex.image = bpy.data.images.load(textures['disp'])
            disp_tex.image.colorspace_settings.name = 'Non-Color'
            disp_tex.location = (-600, y_offset)
            
            disp_node = nodes.new(type='ShaderNodeDisplacement')
            disp_node.location = (300, 0)
            disp_node.inputs['Scale'].default_value = 0.1
            
            links.new(mapping.outputs['Vector'], disp_tex.inputs['Vector'])
            links.new(disp_tex.outputs['Color'], disp_node.inputs['Height'])
            links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])
            print(f"✓ 置换: {os.path.basename(textures['disp'])}")
        except Exception as e:
            print(f"Failed to load disp texture: {e}")
    
    # 应用材质
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    print(f"\n✅ 材质已应用到 '{obj.name}'")

def apply_texture_from_path(obj, diff_texture_path):
    if not os.path.exists(diff_texture_path):
        print(f"Texture file not found: {diff_texture_path}")
        return
        
    folder = os.path.dirname(diff_texture_path)
    filename = os.path.basename(diff_texture_path)
    
    base_name, resolution = parse_texture_name(filename)
    
    if not base_name:
        print(f"Could not parse texture name: {filename}")
        textures = {'diff': diff_texture_path}
    else:
        textures = find_related_textures(folder, base_name, resolution)
        # Make sure we at least have the diff texture provided
        if 'diff' not in textures:
             textures['diff'] = diff_texture_path
        
    apply_textures_to_object(obj, textures, texture_size=1.0)
                        
class BlenderManager:
    """
    用于管理Blender场景中的物体操作:
      - 导入/导出FBX模型
      - 设置物体变换
      - 处理物体之间的空间关系
      - 更新和保存场景信息
    """
    def __init__(self, obj_list=None, obj_dimensions=None, tree_sons=None, processed_matrix=None, carpet=None):
        self.obj_list = obj_list if obj_list is not None else {}
        self.obj_dimensions = obj_dimensions if obj_dimensions is not None else {}
        self.tree_sons = tree_sons if tree_sons is not None else {}
        self.processed_matrix = processed_matrix if processed_matrix is not None else {}
        self.CARPET = carpet if carpet is not None else ["carpet_0", "rug_0"]
        self._loaded_assets = {}  # 缓存已加载的FBX对象
        
    def import_fbx(self, filepath):
        """导入FBX模型（优化版 + 缓存机制）"""
        
        # 检查缓存
        if filepath in self._loaded_assets:
            source_obj = self._loaded_assets[filepath]
            # 确保源对象仍然存在
            try:
                if source_obj.name in bpy.data.objects:
                    # 复制对象
                    new_obj = source_obj.copy()
                    new_obj.data = source_obj.data.copy()  # 深度复制Mesh，确保独立性
                    
                    # 链接到当前集合
                    bpy.context.collection.objects.link(new_obj)
                    
                    # 选中新对象
                    bpy.ops.object.select_all(action='DESELECT')
                    new_obj.select_set(True)
                    bpy.context.view_layer.objects.active = new_obj
                    
                    # 重置变换，确保状态干净
                    new_obj.location = (0, 0, 0)
                    new_obj.rotation_euler = (0, 0, 0)
                    new_obj.scale = (1, 1, 1)
                    
                    return new_obj
                else:
                    # 对象不存在，移除缓存
                    del self._loaded_assets[filepath]
            except Exception as e:
                print(f"Error reusing cached asset: {e}")
                pass

        # ⚡ 性能优化：优先尝试加载同名 .blend 文件
        blend_path = os.path.splitext(filepath)[0] + ".blend"
        if os.path.exists(blend_path):
            try:
                # 使用 Append 方式加载 .blend 中的所有 Object
                with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
                    data_to.objects = data_from.objects
                
                imported_objects = []
                for obj in data_to.objects:
                    if obj is not None:
                        bpy.context.collection.objects.link(obj)
                        imported_objects.append(obj)
                
                # 选中导入的物体
                bpy.ops.object.select_all(action='DESELECT')
                for obj in imported_objects:
                    obj.select_set(True)
                
                if imported_objects:
                    # 尝试找到根物体（没有父物体的物体），如果没有则返回第一个
                    root_obj = next((obj for obj in imported_objects if obj.parent is None), imported_objects[0])
                    bpy.context.view_layer.objects.active = root_obj
                    
                    # 存入缓存
                    self._loaded_assets[filepath] = root_obj
                    return root_obj
            except Exception as e:
                print(f"Error loading blend file {blend_path}: {e}, falling back to FBX.")

        # ⚡ 性能优化：跳过不必要的导入处理以加快加载速度
        bpy.ops.import_scene.fbx(
            filepath=filepath,
            use_anim=False,  # 跳过动画数据
            ignore_leaf_bones=True,  # 跳过叶子骨骼
            automatic_bone_orientation=False,  # 跳过骨骼方向计算
            use_custom_props=False,  # 跳过自定义属性
            use_custom_props_enum_as_string=False,  # 跳过枚举属性
        )
        
        obj = bpy.context.selected_objects[0]
        self._loaded_assets[filepath] = obj  # 存入缓存
        return obj
    
    def clear_scene(self,):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        # 删除所有集合
        for collection in bpy.data.collections:
            bpy.data.collections.remove(collection)
        # 删除所有孤立的数据块
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    
    def set_object_transform(self, obj, transform_matrix):
        """设置物体的变换矩阵"""
        blender_matrix = Matrix([
            transform_matrix[0], 
            transform_matrix[1], 
            transform_matrix[2], 
            transform_matrix[3]
        ])
        obj.matrix_world = blender_matrix

    def align_object_z_to_world_z(self, obj):
        """将物体的Z轴对齐到世界坐标系Z轴"""
        # 获取物体的世界变换矩阵
        world_matrix = obj.matrix_world

        # 提取物体的本地 z 轴在世界坐标系中的方向
        local_z_axis = world_matrix.to_3x3() @ Vector((0, 0, 1))
        local_z_axis.normalize()

        # 计算旋转轴和角度
        rotation_axis = local_z_axis.cross(Vector((0, 0, 1)))
        rotation_angle = local_z_axis.angle(Vector((0, 0, 1)))

        # 如果旋转轴非常小，说明已经对齐或需要 180 度旋转
        if rotation_axis.length < 1e-6:
            if local_z_axis.z > 0:
                return  # 已经对齐，无需操作
            else:
                #  180 度旋转，选择 x 轴
                rotation_axis = Vector((1, 0, 0))
                rotation_angle = 3.14159  # pi

        # 创建旋转四元数
        rotation_quat = Quaternion(rotation_axis, rotation_angle)

        # 创建新的旋转矩阵
        new_rotation = rotation_quat.to_matrix().to_4x4()

        # 保持原始位置
        new_matrix = Matrix.Translation(world_matrix.translation) @ new_rotation @ world_matrix.to_3x3().to_4x4()

        # 应用新的变换矩阵
        obj.matrix_world = new_matrix

    def extract_transform_components(self, matrix):
        """从4x4矩阵中提取变换分量"""
        # 从4x4矩阵中提取位置、旋转和缩放
        
        # 提取缩放
        # 使用矩阵的列向量长度来获取缩放值
        scale_x = matrix.col[0].xyz.length
        scale_y = matrix.col[1].xyz.length
        scale_z = matrix.col[2].xyz.length
        
        return Vector((scale_x, scale_y, scale_z))

    def get_matrix_world(self, obj):
        """获取物体的世界变换矩阵"""
        # 从当前的 matrix_world 中提取变换组件
        scale = self.extract_transform_components(obj.matrix_world)
        
        # Create translation matrix
        translation_matrix = Matrix.Translation(obj.location)
        
        # Create rotation matrix
        rotation_matrix = obj.rotation_euler.to_matrix().to_4x4()
        
        # Create scale matrix
        scale_matrix = Matrix.Scale(scale.x, 4, (1, 0, 0)) @ \
                      Matrix.Scale(scale.y, 4, (0, 1, 0)) @ \
                      Matrix.Scale(scale.z, 4, (0, 0, 1))
        
        # Combine translation, rotation and scale
        combined_matrix = translation_matrix @ rotation_matrix @ scale_matrix
        
        # Apply to object's matrix_world
        return combined_matrix

    def setup_camera(self, name):
        """设置场景相机"""
        bpy.ops.object.camera_add(location=(0, 0, 0))
        camera = bpy.context.object
        camera.name = name
        camera.data.lens = 30  # 焦距 (mm)  应该和深度估计时假设的相机参数一致
        camera.data.sensor_width = 36  # 传感器宽度 (mm)  应该和深度估计时假设的相机参数一致
        camera.data.clip_start = 0.1
        camera.data.clip_end = 100
        return camera

    def set_scene_world_render(self, enable=False):
        try:
            for window in bpy.context.window_manager.windows:
                screen = window.screen
                for area in screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                space.shading.use_scene_world_render = enable
                                return True
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False

    def ensure_object_visible(self, obj):
        if obj.type != 'MESH':
            print(f"Warning: {obj.name} is not a mesh object. Skipping material assignment.")
            return
        
        # 确保物体有材质
        if not obj.data.materials:
            mat = bpy.data.materials.new(name="Default_Material")
            obj.data.materials.append(mat)
        
        # 设置材质为不透明
        for mat in obj.data.materials:
            mat.use_nodes = True
            principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
            if principled_bsdf:
                principled_bsdf.inputs['Alpha'].default_value = 1.0
                
    #def render_scene(output_path, resolution_x=1920, resolution_y=1080):
    def render_scene(self, output_path, resolution_x=1024, resolution_y=1024, samples=32):
        """
        渲染场景
        
        Args:
            output_path: 输出路径
            resolution_x: 分辨率宽度
            resolution_y: 分辨率高度
            samples: 采样数（仅当 use_cycles=True 时有效）
            use_cycles: 是否使用 Cycles 渲染器（False 则使用 EEVEE，速度更快）
        """
        # 删除已有的太阳光源和相机补光
        for obj in list(bpy.data.objects):
            if obj.type == 'LIGHT':
                if obj.data.type == 'SUN' or obj.name == "Camera_Fill_Light":
                    bpy.data.objects.remove(obj, do_unlink=True)
            
        # 1. 设置世界环境光 (World Background) - 确保没有死黑
        if bpy.context.scene.world is None:
            bpy.context.scene.world = bpy.data.worlds.new("World")
        
        world = bpy.context.scene.world
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get('Background')
        if not bg_node:
            bg_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
            world.node_tree.links.new(bg_node.outputs['Background'], world.node_tree.nodes['World Output'].inputs['Surface'])
        
        # 设置环境光颜色和强度
        bg_node.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1.0) # 浅灰色，防止死黑
        bg_node.inputs['Strength'].default_value = 1.0 

        # 2. 添加主光源 (Sun) - 模拟方向光
        bpy.ops.object.light_add(type='SUN')
        sun = bpy.context.object
        sun.location = (0, 0, 10)
        sun.data.energy = 3.0  # 稍微降低太阳强度，让环境光发挥作用
        sun.data.angle = 0.2   # 增加一点角度，使阴影边缘柔和 (弧度)

        # 3. 添加相机方向的面光 (Area Light) - 模拟闪光灯/补光，提亮主体
        scene_camera = bpy.context.scene.camera
        if scene_camera:
            bpy.ops.object.light_add(type='AREA', location=scene_camera.location, rotation=scene_camera.rotation_euler)
            area_light = bpy.context.object
            area_light.name = "Camera_Fill_Light"
            # 如果场景很大，3W 可能太暗。通常 Area Light 在 Cycles 中需要较高的 Watt 值
            # 假设场景是室内尺度 (几米范围)，尝试几百瓦
            area_light.data.energy = 50.0  
            area_light.data.size = 2.0       # 柔和补光
            
            # 将补光稍微移到相机后上方，避免产生奇怪的高光
            # (这里简单起见还是保持在相机位置，或者稍微偏置)

        print(f"使用 Cycles 渲染引擎，采样数: {samples}")
        bpy.context.scene.render.engine = 'CYCLES'
        
        # 设置渲染采样数
        bpy.context.scene.cycles.samples = samples
        
        # 设置渲染滤波阈值
        bpy.context.scene.cycles.denoising_threshold = 0.1
        
        # 配置 Cycles 设置
        cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
        cycles_prefs.compute_device_type = 'CUDA'  # 或 'OPTIX' 如果使用 NVIDIA RTX 卡
        cycles_prefs.get_devices()  # 刷新设备列表
        
        # 设置场景使用 GPU 计算
        bpy.context.scene.cycles.device = 'GPU'
        
        # 启用所有可用的 GPU 设备
        for device in cycles_prefs.devices:
            if device.type == 'CUDA':  # 或 'OPTIX'
                device.use = True
        
        # 设置渲染分辨率
        bpy.context.scene.render.resolution_x = resolution_x
        bpy.context.scene.render.resolution_y = resolution_y
        bpy.context.scene.render.pixel_aspect_x = 1.0
        bpy.context.scene.render.pixel_aspect_y = 1.0
        bpy.context.scene.render.resolution_percentage = 100
        
        # 设置输出路径和格式
        bpy.context.scene.render.filepath = output_path
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        
        # 设置活动相机
        scene_camera = bpy.context.scene.camera
        if not scene_camera:
            raise Exception("No active camera in the scene!")
        
        # 取消选择所有物体
        bpy.ops.object.select_all(action='DESELECT')
        
        # #### 下面的代码会导致相机位姿发生变化, 由于scale有部分计算是基于相机渲染的, 相机内外参需要严格与深度估计一致，所以此处不能用下面的代码
        # # 选择不匹配特定模式的物体,放入视野
        # for obj in bpy.context.scene.objects:
        #     if not re.match(r'^(wall|floor)_\d+', obj.name):
        #         obj.select_set(True)
        # # 将相机对准选中的物体
        # bpy.ops.view3d.camera_to_view_selected()
        
        self.set_scene_world_render(False)
            
        # 渲染图片
        # 注意：这里的渲染仅用于可视化预览，不影响 S4 布局/位姿输出。
        # 若 GPU 显存不足（常见于共享卡被其他进程占用），回退到 CPU；
        # 若仍失败则跳过渲染，绝不因预览图而中断整个 S4 pipeline。
        try:
            bpy.ops.render.render(write_still=True)
        except RuntimeError as e:
            print(f"[render_scene] GPU 渲染失败({e})，回退到 CPU 渲染...", flush=True)
            try:
                bpy.context.scene.cycles.device = 'CPU'
                bpy.ops.render.render(write_still=True)
                print("[render_scene] CPU 渲染成功。", flush=True)
            except Exception as e2:
                print(f"[render_scene] CPU 渲染仍失败({e2})，跳过预览渲染，继续 pipeline。", flush=True)
    
    def get_world_bound_box(self, obj):
        """获取物体的世界坐标系包围盒"""
        world_matrix = self.get_matrix_world(obj)
        bbox = [world_matrix @ Vector(corner) for corner in obj.bound_box]
        return bbox

    def _resolve_settle_policy(self):
        """读取沉降策略，理由见 modules/_s4_settle.py 的模块说明。"""
        return resolve_settle_policy()

    def process_z(self, ground_name, obj_list, tree_sons, ground_height=0):
        """处理物体在Z轴方向上的位置关系，从地面开始"""
        self._settle_enabled, self._settle_max_gap = self._resolve_settle_policy()
        print(
            f"[SETTLE] policy enabled={self._settle_enabled} "
            f"max_gap={self._settle_max_gap:.3f}m"
        )
        ground_bbox = self.get_world_bound_box(obj_list[ground_name])
        ground_max_z = max(point.z for point in ground_bbox)

        # 首先处理直接放在地面上的物体及其后代
        for son in tree_sons.get(ground_name, []):
            if son not in obj_list:
                continue
            son_bbox = self.get_world_bound_box(obj_list[son])
            son_min_z = min(point.z for point in son_bbox)
            
            # 计算需要的位移
            delta_z = ground_max_z - son_min_z + ground_height
            
            # 调整子物体位置
            obj_list[son].location.z += delta_z
            
            # 递归调整该子物体的所有后代
            self.adjust_descendants(son, obj_list, tree_sons, delta_z)

        # 然后处理其他物体的位置关系
        self.process_other_objects(ground_name, obj_list, tree_sons, ground_max_z + ground_height)

    def adjust_descendants(self, obj_id, obj_list, tree_sons, delta_z):
        """递归调整物体及其所有后代的z位置（包括 inside 关系的物体）"""
        if obj_id in tree_sons:
            for son in tree_sons[obj_id]:
                # 优先从 obj_list 获取，如果不在则从 Blender 场景中直接获取
                # 这样可以处理 inside 关系的物体（它们不在 obj_list 中）
                obj = obj_list.get(son) or bpy.data.objects.get(son)
                if obj:
                    print(f"adjusting {son} location.z from {obj.location.z} to {obj.location.z + delta_z}")
                    obj.location.z += delta_z
                    self.adjust_descendants(son, obj_list, tree_sons, delta_z)

    def process_other_objects(self, parent_id, obj_list, tree_sons, parent_height):
        """处理非地面直接子物体的位置关系"""
        if parent_id in tree_sons:
            for son in tree_sons[parent_id]:
                if son not in obj_list:  # 是相机或需要内部摆放的物体
                    continue
                son_bbox = self.get_world_bound_box(obj_list[son])
                son_min_z = min(point.z for point in son_bbox)
                son_max_z = max(point.z for point in son_bbox)
                
                parent_bbox = self.get_world_bound_box(obj_list[parent_id])
                parent_max_z = max(point.z for point in parent_bbox)
                
                # 子物体底面对齐父物体顶面。判断本身在 modules/_s4_settle.py，
                # 那里没有 bpy 依赖因而可以单测；此处只负责把 Blender 量出来的
                # 两个 z 喂进去，并把位移施加到物体及其后代。
                delta_z, settle_reason = settle_delta_z(
                    son_min_z,
                    parent_max_z,
                    enabled=getattr(self, "_settle_enabled", True),
                    max_gap=getattr(self, "_settle_max_gap", DEFAULT_MAX_SETTLE_GAP_M),
                )
                if delta_z != 0.0:
                    print(
                        f"[SETTLE] {son} on {parent_id}: {settle_reason} "
                        f"gap={son_min_z - parent_max_z:+.4f} delta_z={delta_z:+.4f}"
                    )
                    obj_list[son].location.z += delta_z
                    # 同步更新 son 的所有后代物体
                    self.adjust_descendants(son, obj_list, tree_sons, delta_z)
                
                # 递归处理子物体
                self.process_other_objects(son, obj_list, tree_sons, son_max_z - son_min_z + parent_height)

    @staticmethod
    def process_rotation_against_wall(obj_name, obj_info, wall_name):
        """处理物体的旋转，使其 X 轴或 Y 轴的正负方向与墙壁对齐"""
        if obj_name not in bpy.data.objects:
            print(
                f"[S4] Missing object {obj_name} during wall rotation; skipping."
            )
            return False
        if wall_name not in bpy.data.objects:
            print(
                f"[S4] Missing structural wall {wall_name} for {obj_name} "
                "during rotation; preserving the current pose."
            )
            return False
        obj = bpy.data.objects[obj_name]
        wall = bpy.data.objects[wall_name]
        

        align_closest_axis_to_world_z(obj)
        
        wall_rotation = wall.rotation_euler.to_matrix()
        wall_normal = wall_rotation @ Vector((0, 0, 1))
        wall_normal.z = 0  # 投影到XY平面
        wall_normal.normalize()

        # 对于父物体是墙的物体，并且 alignToWallNormal==1，应该调整他们的位姿，让他们的正方向和墙的法向一致
        parent = obj_info.get('supported')
        should_align = obj_info.get('alignToWallNormal', 0) == 1
        if parent == wall_name and should_align:
            # 计算目标角度
            target_angle = math.atan2(wall_normal.y, wall_normal.x)
            
            # 设置物体旋转 (Y+ align with Wall Normal)
            obj.rotation_euler[2] = target_angle + math.pi / 2
            
            print(f"Force aligned {obj_name} positive direction (Y+) to wall {wall_name}")
            bpy.context.view_layer.objects.active = obj
            bpy.context.view_layer.update()
            return

        # 获取物体的局部 X 轴和 Y 轴在世界坐标系中的方向
        obj_x = obj.matrix_world.to_3x3() @ Vector((1, 0, 0))
        obj_y = obj.matrix_world.to_3x3() @ Vector((0, 1, 0))
        obj_x.z = obj_y.z = 0  # 投影到XY平面
        obj_x.normalize()
        obj_y.normalize()
        
        # 计算各轴与墙体法线的夹角
        angles = [
            (abs(obj_x.dot(wall_normal)), obj_x, "X+"),
            (abs((-obj_x).dot(wall_normal)), -obj_x, "X-"),
            (abs(obj_y.dot(wall_normal)), obj_y, "Y+"),
            (abs((-obj_y).dot(wall_normal)), -obj_y, "Y-")
        ]
        
        # 选择夹角最接近 1 (0°或180°) 的轴
        best_angle, best_axis, axis_name = max(angles, key=lambda x: x[0])
        
        # 计算需要旋转的角度
        dot_product = best_axis.dot(wall_normal)
        rotation_angle = math.acos(max(min(dot_product, 1), -1))
        
        # 确定旋转方向
        cross_product = best_axis.cross(wall_normal)
        rotation_direction = 1 if cross_product.z > 0 else -1
        
        # 如果夹角接近180度，选择较小的旋转角度
        if rotation_angle > math.pi/2:
            rotation_angle = math.pi - rotation_angle
            rotation_direction *= -1
        
        # 应用旋转
        obj.rotation_euler[2] += rotation_angle * rotation_direction
        
        print(f"Aligned {axis_name} axis. Rotation angle: {math.degrees(rotation_angle):.2f} degrees")
        
        bpy.context.view_layer.objects.active = obj
        bpy.context.view_layer.update()
        return True
    
    def process_rotation_against_wall_hierarchical(self, obj_info, obj_list, tree_sons):
        """按层级顺序处理靠墙物体的旋转（支持多级）"""
        # 生成层级处理顺序：从父到子
        hierarchy_levels = {}
        
        # def get_level(obj_name):
        #     if obj_name in hierarchy_levels:
        #         return hierarchy_levels[obj_name]
        #     parent = obj_info[obj_name].get('supported')
        #     if parent and parent in obj_list:
        #         hierarchy_levels[obj_name] = get_level(parent) + 1
        #     else:
        #         hierarchy_levels[obj_name] = 0  # 顶层物体层级为0
        #     return hierarchy_levels[obj_name]
        def get_level(obj_name, visited=None):
            if visited is None:
                visited = set()
            
            if obj_name in hierarchy_levels:
                return hierarchy_levels[obj_name]
            
            level = 0
            current = obj_name
            path = []
            
            while True:
                if current in visited:
                    # 检测到循环依赖，中断循环
                    print(f"警告：检测到对象的循环依赖: {current}")
                    break
                
                visited.add(current)
                path.append(current)
                
                parent = obj_info[current].get('supported')
                if parent and parent in obj_list:
                    if parent in hierarchy_levels:
                        level = hierarchy_levels[parent] + 1
                        break
                    current = parent
                    level += 1
                else:
                    break
            
            # 为路径中的所有对象分配层级
            for i, obj in enumerate(reversed(path)):
                hierarchy_levels[obj] = level - i
            
            return hierarchy_levels[obj_name]

        # 只处理需要靠墙的物体
        target_objects = [obj for obj in obj_info if obj_info[obj].get("againstWall")]
        for obj in target_objects:
            get_level(obj)

        # 按层级从浅到深排序（父级在前）
        sorted_objects = sorted(target_objects, key=lambda x: hierarchy_levels[x])

        # 按层级处理
        for obj_name in sorted_objects:
            if not obj_info[obj_name].get("againstWall"):
                continue

            # 创建针对当前物体子树的pose manager
            current_tree = {obj_name: tree_sons.get(obj_name, [])}
            pose_manager = RelativePoseManager(
                obj_list={k: v for k,v in obj_list.items() if k in [obj_name]+current_tree[obj_name]},  # 只包含当前子树
                tree_sons=current_tree,
                output_data_s2=obj_info
            )
            
            # 记录当前物体的子物体相对位姿
            pose_manager.record_relative_poses(obj_list[obj_name], current_tree[obj_name])
            
            # 处理当前物体旋转
            wall_name = obj_info[obj_name].get("most_like_wall")
            if not wall_name:
                continue
            self.process_rotation_against_wall(obj_name, obj_info[obj_name], wall_name)
            
            # 恢复子物体相对位姿
            pose_manager.restore_relative_poses(obj_list[obj_name], current_tree[obj_name])
            bpy.context.view_layer.update()
            
            print(f"Processed {obj_name} (level {hierarchy_levels[obj_name]}) with {len(current_tree[obj_name])} children")

    @staticmethod
    def process_translation_against_wall(obj_info, obj_list):
        """处理靠墙物体的位置"""
        for instance_id, obj in obj_list.items():
            info = obj_info[instance_id]
            if info["againstWall"]:
                # 将 againstWall 转换为列表
                wall_ids = [info["againstWall"]] if isinstance(info["againstWall"], str) else info["againstWall"]
                    
                for wall_id in wall_ids:
                    if wall_id not in bpy.data.objects:
                        print(
                            f"[S4] Missing structural wall {wall_id} for {instance_id}; "
                            "preserving the current pose."
                        )
                        continue
                    wall = obj_list[wall_id]
                    
                    # 获取墙的旋转矩阵和法向量
                    wall_rotation = wall.rotation_euler.to_matrix()
                    normal_vector = wall_rotation @ Vector((0, 0, 1))
                    normal_vector.z = 0  # 投影到XY平面
                    normal_vector.normalize()

                    # 计算墙到场景中心的向量
                    wall_to_center = Vector((0, 0, 0)) - wall.location

                    # 判断墙的法向是否指向场景中心
                    normal_points_to_center = normal_vector.dot(wall_to_center) > 0

                    # 计算物体的中心点
                    obj_center = obj.location

                    # 计算物体中心到墙的距离
                    center_distance = (obj_center - wall.location).dot(normal_vector)

                    # 判断物体是否在墙内（基于中心点）
                    is_inside = (center_distance > 0) if normal_points_to_center else (center_distance < 0)
                    
                    # 计算物体的包围盒
                    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

                    # 找到最靠近墙和最远离墙的点
                    closest_point = min(bbox_corners, key=lambda p: (p - wall.location).dot(normal_vector))
                    farthest_point = max(bbox_corners, key=lambda p: (p - wall.location).dot(normal_vector))

                    # 计算最近点和最远点到墙的距离
                    closest_distance = (closest_point - wall.location).dot(normal_vector)
                    farthest_distance = (farthest_point - wall.location).dot(normal_vector)
                    
                    # 计算墙的厚度
                    wall_thickness = wall.dimensions[2]

                    # 根据is_inside和点的位置计算移动距离
                    if is_inside:
                        # 物体中心在墙内，移动到完全进入墙内
                        move_direction_tag = farthest_distance + wall_thickness / 2
                    else:
                        # 物体中心在墙外，移动到完全进入墙内
                        move_direction_tag = closest_distance - wall_thickness / 2

                    # 计算移动方向（始终朝向墙内移动）
                    move_direction = -normal_vector if move_direction_tag > 0 else normal_vector

                    # 计算具体的移动距离
                    if is_inside:
                        if closest_distance*farthest_distance >0:
                            distance = min(abs(closest_distance), abs(farthest_distance)) - wall_thickness / 2
                            distance = distance * move_direction
                        elif closest_distance*farthest_distance <0:
                            distance = min(abs(closest_distance), abs(farthest_distance)) + wall_thickness / 2
                            distance = -distance * move_direction
                        else:
                            distance = wall_thickness / 2 * move_direction
                    else:
                        distance = max(abs(closest_distance), abs(farthest_distance)) + wall_thickness / 2
                        distance = distance * move_direction

                    # 计算新的位置
                    new_location = obj.location + distance

                    # 保持原来的z坐标
                    new_location.z = obj.location.z

                    # 更新物体位置
                    obj.location = new_location

                    # 更新场景
                    bpy.context.view_layer.update()
                        
    def process_wall(self, wall_id, obj_list, ground_name):
        """处理墙体位置"""
        min_penetration = float('0')
        wall = obj_list[wall_id] 
        ground = obj_list[ground_name]
        ground_bbox = self.get_world_bound_box(ground)
        ground_max_z = max(point.z for point in ground_bbox)
        wall_rotation = wall.rotation_euler.to_matrix()
        normal_vector = wall_rotation @ Vector((0, 0, 1))
        
        normal_vector.z = 0  # Project to XY plane
        normal_vector.normalize()
        wall_location = wall.location + ground_max_z * normal_vector
        # print("wall_id",wall.location,wall_location)
        
        for instance_id, obj in obj_list.items():
            if re.match(r"wall_\d+", instance_id):
                continue
            if instance_id == ground_name:
                continue
            
            obj_bbox = self.get_world_bound_box(obj)
            # Calculate penetration for each bbox point
            projections = []
            for point in obj_bbox:
                diff_vector = point - wall_location
                projection = diff_vector.dot(normal_vector)
                projections.append(projection)
            
            min_projection = min(projections)
            min_penetration = min(min_penetration, min_projection)
        
        print(wall_id, min_penetration)
        wall.location += normal_vector * min_penetration

    @staticmethod
    def process_directly_facing(all_obj_info, fbx_scaling_strategy):
        for obj_name, obj_info in all_obj_info.items():
            if obj_info.get('directlyFacing', None) is None:
                continue
            
            facing_item = obj_info['directlyFacing']
            retrieved_asset = all_obj_info[facing_item]["retrieved_asset"]
            scaling_strategy = fbx_scaling_strategy[retrieved_asset]
            
            obj = bpy.data.objects[obj_name]
            facing_obj = bpy.data.objects[facing_item]
            if scaling_strategy == 'RADIAL':
                # 旋转obj, 使其正方向指向facing_obj的中心
                direction = facing_obj.location - obj.location
                angle = math.atan2(direction.y, direction.x)
                obj.rotation_euler[2] = angle + math.pi / 2
            else:
                ''''
                计算obj中心在facing_obj的局部坐标系x轴上的矢量投影长度(dis_x)，假设facing_obj在x轴上的dimension投影是(-dimension_x/2, dimension_x/2)
                计算obj中心在facing_obj的局部坐标系y轴上的矢量投影长度(dis_y)，假设facing_obj在y轴上的dimension投影是(-dimension_y/2, dimension_y/2) 
                计算min(abs(dis_x-dimension_x/2）, abs(dis_x-(-dimension_x/2)))，如果更小的是前者，那x方向的候选对齐轴就是facing_obj的x负半轴（反一下） 
                计算min(abs(dis_y-dimension_y/2）, abs(dis_y-(-dimension_y/2)))，如果更小的是前者，那y方向的候选对齐轴就是facing_obj的y负半轴（反一下） 
                接下来就是确认obj的正方向应该与facing_obj的x负半轴旋转至一致还是与facing_obj的y负半轴旋转至一致。。。
                '''
                # 计算obj中心在facing_obj局部坐标系中的位置
                local_pos = facing_obj.matrix_world.inverted() @ obj.location
                # 还要考虑facing_obj的scale带来的影响
                local_pos = local_pos*facing_obj.scale
                
                # 获取facing_obj的尺寸
                dimension_x, dimension_y, _ = facing_obj.dimensions

                # 计算在x和y轴上的投影距离
                dis_x = local_pos.x
                dis_y = local_pos.y

                # 确定x方向的候选对齐轴
                x_align = -1 if abs(dis_x - dimension_x/2) < abs(dis_x + dimension_x/2) else 1

                # 确定y方向的候选对齐轴
                y_align = -1 if abs(dis_y - dimension_y/2) < abs(dis_y + dimension_y/2) else 1

                # 确定最终的对齐方向
                if abs(dis_x) > dimension_x/2 and abs(dis_y) > dimension_y/2:
                    direction = facing_obj.location - obj.location
                    direction.normalize()
                    align_axis = facing_obj.matrix_world.to_3x3().inverted() @ direction
                elif abs(dis_x) > dimension_x/2:
                    align_axis = mathutils.Vector((x_align, 0, 0))
                elif abs(dis_y) > dimension_y/2:
                    align_axis = mathutils.Vector((0, y_align, 0))
                else:
                    # 确定最终的对齐方向
                    if min(abs(dis_x + x_align*dimension_x/2), abs(dis_y + y_align*dimension_y/2)) == abs(dis_x + x_align*dimension_x/2):
                        align_axis = mathutils.Vector((x_align, 0, 0))
                    else:
                        align_axis = mathutils.Vector((0, y_align, 0))

                # 将align_axis转换到世界坐标系
                world_align_axis = facing_obj.matrix_world.to_3x3() @ align_axis

                # 计算旋转
                rot_quat = world_align_axis.to_track_quat('-Y', 'Z')
                obj.rotation_euler = rot_quat.to_euler()

            # 更新场景
            bpy.context.view_layer.update()

class Obj:
    """
    用于存储单个物体在优化过程中的关键信息:
      - instance_id: 物体在场景中的唯一标识
      - original_pos: 原始 (x, y) 位置
      - current_pos: 当前 (x, y) 位置 (随优化更新)
      - parent_id: 父物体 ID
      - bounding_box: 包含 min, max, length, theta 等信息的字典
      - is_against_wall: 是否靠墙 (如果有对应的墙ID)
      - relation: 物体与父物体的空间关系: "inside" / "on" / "None" 等
      - pose_3d: 原始或最新的 3D 位姿矩阵 (4x4)
    """
    def __init__(self, instance_id, info, base_fbx_path):
        self.instance_id = instance_id
        self.parent_id = info.get('supported', None)
        self.is_against_wall = info.get("againstWall", None)
        self.relation = info.get("SpatialRel", None)
        self.pose_3d = info.get("pose_matrix_for_blender", None)
        fbx_name = info['retrieved_asset']
        self.fbx_path = f"{base_fbx_path}/{fbx_name}.fbx"
        # 通过 bbox 中心来初始化原始位置和当前位置
        bbox_points = np.array(info["bbox"])
        min_corner = np.min(bbox_points, axis=0)
        max_corner = np.max(bbox_points, axis=0)
        center_x = (min_corner[0] + max_corner[0]) / 2.0
        center_y = (min_corner[1] + max_corner[1]) / 2.0

        self.original_pos = (center_x, center_y)
        self.current_pos = [center_x, center_y]

        # 物体的 bounding_box 信息: 包括 min, max, length, theta 等
        length = max_corner - min_corner
        theta = 0.0
        if self.pose_3d is not None:
            # 根据 4x4 矩阵, 提取 z 轴旋转角度
            theta = math.atan2(self.pose_3d[1][0], self.pose_3d[0][0])  
        self.bounding_box = {
            "length": [length[0], length[1]],  # 只使用 x, y 长度
            "min": [float(min_corner[0]), float(min_corner[1]), float(min_corner[2])],
            "max": [float(max_corner[0]), float(max_corner[1]), float(max_corner[2])],
            "theta": float(theta),
            # 记录初始中心位置 (x, y)，用于后续计算移动距离
            "x": float(center_x),
            "y": float(center_y),
        }


# ==================== VoxelManager类 ====================
class VoxelManager:
    """用于管理场景中体素化和碰撞检测"""
    
    def __init__(self, resolution=(128, 128, 128), precomputed_voxel_dir=None):
        self.resolution = resolution
        self.voxel_grids = {}
        self.mesh_cache = {}
        self.scene_bounds = {
            'min': [float('inf'), float('inf'), float('inf')],
            'max': [-float('inf'), -float('inf'), -float('inf')]
        }
        self.voxel_size = None
        self.scene_initialized = False
        
        # 预计算体素数据目录
        self.precomputed_voxel_dir = Path(precomputed_voxel_dir) if precomputed_voxel_dir else None
        self.precomputed_cache = {}  # 缓存加载的预计算数据
        self.voxel_load_stats = {'precomputed': 0, 'realtime': 0, 'failed': 0}
        
    def initialize_scene_bounds(self, obj_dict, wall_dict):
        """预先计算整个场景的边界"""
        if self.scene_initialized:
            return
        
        standard_directions = {
            'left': np.array([1, 0, 0]),
            'right': np.array([-1, 0, 0]),
            'front': np.array([0, -1, 0]),
            'back': np.array([0, 1, 0])
        }
        
        self.wall_constraints = {}
        
        for wall_id, wall_info in wall_dict.items():
            if not wall_id.startswith('wall'):
                continue
            
            wall_pose = np.array(wall_info['pose_matrix_for_blender'])
            wall_normal = wall_pose[:3, :3] @ np.array([0, 0, 1])
            wall_normal = wall_normal / np.linalg.norm(wall_normal)
            wall_point = wall_pose[:3, 3]
            
            wall_type = max(standard_directions.items(), 
                          key=lambda x: np.dot(wall_normal, x[1]))[0]
            
            if wall_type == 'left':
                self.wall_constraints['left'] = wall_point[0]
            elif wall_type == 'right':
                self.wall_constraints['right'] = wall_point[0]
            elif wall_type == 'front':
                self.wall_constraints['front'] = wall_point[1]
            elif wall_type == 'back':
                self.wall_constraints['back'] = wall_point[1]
            
            print(f"Wall {wall_id} classified as {wall_type} with constraint value {wall_point}")
        
        print("Final wall constraints:", self.wall_constraints)
        
        for inst_id, obj in obj_dict.items():
            mesh = self.load_mesh(obj.fbx_path)
            if obj.pose_3d is not None:
                mesh = mesh.copy()
                mesh = mesh.apply_transform(obj.pose_3d)
            
            bounds = mesh.bounds
            self.scene_bounds['min'] = np.minimum(self.scene_bounds['min'], bounds[0])
            self.scene_bounds['max'] = np.maximum(self.scene_bounds['max'], bounds[1])

        SCENE_MARGIN_FACTOR = 0.25
        original_scene_size = (np.array(self.scene_bounds['max']) - np.array(self.scene_bounds['min']))
        
        if 'left' in self.wall_constraints:
            self.scene_bounds['min'][0] = min(self.wall_constraints['left'], self.scene_bounds['min'][0])
        else:
            self.scene_bounds['min'][0] -= SCENE_MARGIN_FACTOR * original_scene_size[0]

        if 'right' in self.wall_constraints:
            self.scene_bounds['max'][0] = max(self.wall_constraints['right'], self.scene_bounds['max'][0])
        else:
            self.scene_bounds['max'][0] += SCENE_MARGIN_FACTOR * original_scene_size[0]

        if 'back' in self.wall_constraints:
            self.scene_bounds['min'][1] = min(self.wall_constraints['back'], self.scene_bounds['min'][1])
        else:
            self.scene_bounds['min'][1] -= SCENE_MARGIN_FACTOR * original_scene_size[1]

        if 'front' in self.wall_constraints:
            self.scene_bounds['max'][1] = max(self.wall_constraints['front'], self.scene_bounds['max'][1])
        else:
            self.scene_bounds['max'][1] += SCENE_MARGIN_FACTOR * original_scene_size[1]
        
        scene_size = (np.array(self.scene_bounds['max']) - np.array(self.scene_bounds['min']))
        self.voxel_size = scene_size / np.array(self.resolution)
        self.voxel_size = np.array([min(self.voxel_size)] * 3)
        self.resolution = (int(round(scene_size[0] / self.voxel_size[0])),
                           int(round(scene_size[1] / self.voxel_size[1])),
                           int(round(scene_size[2] / self.voxel_size[2])))
        self.scene_initialized = True
        
        print("Scene bounds:", self.scene_bounds)
        print("Voxel size:", self.voxel_size)

    def fbx2mesh(self, fbx_path):
        with pyassimp.load(str(fbx_path)) as scene:
            mesh = scene.meshes[0]
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    def load_mesh(self, fbx_path):
        """加载mesh"""
        mesh = self.fbx2mesh(fbx_path)
        return mesh

    def approximate_as_box_if_thin(self, mesh: trimesh.Trimesh, pitch: float) -> trimesh.Trimesh:
        """如果网格在某个维度极其薄，则其近似为一个长方体"""
        min_corner, max_corner = mesh.bounds
        size = max_corner - min_corner

        i_min = np.argmin(size)
        if size[i_min] < pitch:
            center = (max_corner + min_corner) / 2.0
            half_size = size / 2.0
            half_size[i_min] = pitch / 2.0

            box = trimesh.creation.box(extents=2.0 * half_size)
            box.apply_translation(center)
            return box
        else:
            return mesh
    
    def load_precomputed_voxels(self, mesh_path):
        """
        加载预计算的体素数据
        
        Returns:
            dict or None: 预计算的体素数据，如果不存在则返回None
        """
        if self.precomputed_voxel_dir is None:
            return None
        
        # 计算预计算文件路径
        mesh_path = Path(mesh_path)
        relative_path = mesh_path.relative_to(mesh_path.parents[0])
        voxel_file = self.precomputed_voxel_dir / relative_path.with_suffix('.voxel.pkl')
        
        # 检查缓存
        cache_key = str(mesh_path)
        if cache_key in self.precomputed_cache:
            return self.precomputed_cache[cache_key]
        
        # 尝试加载文件
        if not voxel_file.exists():
            return None
        
        try:
            with open(voxel_file, 'rb') as f:
                voxel_data = load_numpy_compatible_pickle(f)
            self.precomputed_cache[cache_key] = voxel_data
            return voxel_data
        except Exception as e:
            print(f"警告: 无法加载预计算体素 {voxel_file}: {e}")
            return None
    
    def voxelize_from_precomputed(self, voxel_data, instance_id, pose, scale=None):
        """
        从预计算的体素数据创建场景体素网格
        
        Args:
            voxel_data: 预计算的体素数据
            instance_id: 实例ID
            pose: 位姿矩阵
            scale: 缩放因子
        
        Returns:
            torch.Tensor: 体素网格
        """
        if not self.scene_initialized:
            raise RuntimeError("Scene bounds not initialized. Call initialize_scene_bounds first.")
        
        # 提取预计算数据
        voxel_indices = voxel_data['voxel_indices']  # (N, 3) int16
        origin = voxel_data['origin']  # (3,) float32
        pitch = voxel_data['pitch']  # float
        
        # 转换到世界坐标
        voxel_points_local = origin + voxel_indices * pitch
        
        # 应用缩放
        if scale is not None:
            scale_array = np.array(scale)
            voxel_points_local = voxel_points_local * scale_array
        
        # 应用位姿变换
        transform = np.array(pose)
        voxel_points_homogeneous = np.hstack([voxel_points_local, np.ones((len(voxel_points_local), 1))])
        voxel_points_world = (transform @ voxel_points_homogeneous.T).T[:, :3]
        
        # 转为Tensor放入GPU
        voxel_points_tensor = torch.from_numpy(voxel_points_world.astype(np.float32)).cuda()
        
        # 映射到整个场景的Grid系统中
        scene_min = torch.tensor(self.scene_bounds['min'], device='cuda', dtype=torch.float32)
        voxel_size_tensor = torch.tensor(self.voxel_size, device='cuda', dtype=torch.float32)
        
        relative_pos = voxel_points_tensor - scene_min
        voxel_coords = (relative_pos / voxel_size_tensor).long()
        
        # 创建全场景Grid
        grid = torch.zeros(self.resolution, dtype=torch.bool, device='cuda', requires_grad=False)
        
        # 过滤越界体素
        valid_mask = (
            (voxel_coords[:, 0] >= 0) & (voxel_coords[:, 0] < self.resolution[0]) &
            (voxel_coords[:, 1] >= 0) & (voxel_coords[:, 1] < self.resolution[1]) &
            (voxel_coords[:, 2] >= 0) & (voxel_coords[:, 2] < self.resolution[2])
        )
        voxel_coords = voxel_coords[valid_mask]
        
        if len(voxel_coords) == 0:
            print(f"警告: {instance_id} 的所有体素都超出场景边界")
            self.voxel_grids[instance_id] = grid
            return grid
        
        # 填充Grid
        grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = True
        
        self.voxel_grids[instance_id] = grid
        return grid

    def voxelize_object(self, mesh_path, instance_id, pose, scale=None):
        """
        将物体mesh转换为体素网格 (修复空心 + 防消失保护)
        优先尝试加载预计算的体素数据，如果不存在则实时计算
        """
        if not self.scene_initialized:
            raise RuntimeError("Scene bounds not initialized. Call initialize_scene_bounds first.")
        
        # 尝试加载预计算的体素数据
        precomputed_data = self.load_precomputed_voxels(mesh_path)
        if precomputed_data is not None:
            try:
                grid = self.voxelize_from_precomputed(precomputed_data, instance_id, pose, scale)
                self.voxel_load_stats['precomputed'] += 1
                return grid
            except Exception as e:
                print(f"警告: 使用预计算体素失败 ({instance_id}): {e}，回退到实时计算")
                self.voxel_load_stats['failed'] += 1
        
        # 回退到原始的实时体素化流程
        self.voxel_load_stats['realtime'] += 1
        
        mesh = self.load_mesh(mesh_path)
        if scale is not None:
            mesh.apply_scale(scale)

        transform = np.array(pose)
        mesh = mesh.apply_transform(transform)

        pitch = float(min(self.voxel_size))
        mesh = self.approximate_as_box_if_thin(mesh, pitch)

        # 1. 基础体素化
        voxels = mesh.voxelized(pitch=pitch, method='subdivide')
        
        if hasattr(voxels, 'matrix'):
            grid_np = voxels.matrix.copy()
        else:
            grid_np = voxels.encoding.dense.copy()

        # 获取 Origin (兼容性处理)
        if hasattr(voxels, 'origin'):
            grid_origin = voxels.origin
        elif hasattr(voxels, 'translation'):
            grid_origin = voxels.translation
        elif hasattr(voxels, 'transform'):
            grid_origin = voxels.transform[:3, 3]
        else:
            grid_origin = mesh.bounds[0]

        # ==================== 核心修改：带保护的实心化流程 ====================
        
        # A. 膨胀 (Dilation) - 封堵缝隙
        # 建议设置为 2 或 3，足以封住大部分家具底部的洞
        dilation_iter = 2
        grid_dilated = scipy.ndimage.binary_dilation(grid_np, iterations=dilation_iter)
        
        # B. 填充孔洞 (Fill Holes) - 实心化
        grid_filled = scipy.ndimage.binary_fill_holes(grid_dilated)
        
        # C. 安全腐蚀 (Safe Erosion) - 还原尺寸，但防止消失
        # 尝试腐蚀回去，次数通常比膨胀少 1 次，或者相等
        erosion_iter = 1 # 如果膨胀是2，腐蚀1比较安全；如果膨胀3，腐蚀2
        
        grid_eroded = scipy.ndimage.binary_erosion(grid_filled, iterations=erosion_iter)
        
        # --- 关键判断 ---
        if np.sum(grid_eroded) == 0:
            # 如果腐蚀把物体搞没了（针对画框等薄物体），就放弃腐蚀，使用填充后的版本
            # print(f"Notice: {instance_id} is too thin for erosion, keeping filled volume.")
            grid_final = grid_filled
        else:
            # 如果还有东西，就使用腐蚀后的版本（尺寸更准）
            grid_final = grid_eroded
            
        # ===================================================================

        # 计算体素在场景中的位置
        voxel_points_indices = np.argwhere(grid_final) # 使用 grid_final
        
        if len(voxel_points_indices) == 0:
            # 如果连膨胀后都是空的（极少见），做个兜底
            print(f"Warning: Object {instance_id} vanished completely!")
            grid = torch.zeros(self.resolution, dtype=torch.bool, device='cuda')
            self.voxel_grids[instance_id] = grid
            return grid

        # 将局部索引转为世界坐标
        voxel_points_world = grid_origin + voxel_points_indices * pitch
        
        # 转为 Tensor 放入 GPU
        voxel_points_tensor = torch.from_numpy(voxel_points_world).float().cuda()
        
        # 映射到整个场景的 Grid 系统中
        relative_pos = voxel_points_tensor - torch.tensor(self.scene_bounds['min'], device='cuda')
        voxel_coords = (relative_pos / torch.tensor(self.voxel_size, device='cuda')).long()
        
        # 创建全场景 Grid
        grid = torch.zeros(self.resolution, dtype=torch.bool, device='cuda', requires_grad=False)
        
        voxel_coords = torch.clamp(
            voxel_coords,
            torch.tensor(0, device='cuda'),
            torch.tensor(self.resolution, device='cuda') - 1
        )

        grid.index_put_(
            (voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]),
            torch.ones(len(voxel_coords), dtype=torch.bool, device='cuda'),
            accumulate=True
        )

        self.voxel_grids[instance_id] = grid
        return grid
        
    def move_grid(self, instance_id, offset):
        """移动体素网格"""
        dx, dy, dz = [int(round(o)) for o in offset]
        grid = self.voxel_grids[instance_id]
        
        if (abs(dx) >= grid.shape[0] or abs(dy) >= grid.shape[1] or abs(dz) >= grid.shape[2]):
            return False
        
        if dx > 0:
            if torch.any(grid[grid.shape[0]-dx:]):
                return False
        elif dx < 0:
            if torch.any(grid[:abs(dx)]):
                return False
            
        if dy > 0:
            if torch.any(grid[:, grid.shape[1]-dy:]):
                return False
        elif dy < 0:
            if torch.any(grid[:, :abs(dy)]):
                return False
            
        if dz > 0:
            if torch.any(grid[:, :, grid.shape[2]-dz:]):
                return False
        elif dz < 0:
            if torch.any(grid[:, :, :abs(dz)]):
                return False
        
        if dx > 0:
            grid = torch.cat([torch.zeros_like(grid[:dx]), grid[:-dx]], dim=0)
        elif dx < 0:
            grid = torch.cat([grid[-dx:], torch.zeros_like(grid[:-dx])], dim=0)
        
        if dy > 0:
            grid = torch.cat([torch.zeros_like(grid[:, :dy]), grid[:, :-dy]], dim=1)
        elif dy < 0:
            grid = torch.cat([grid[:, -dy:], torch.zeros_like(grid[:, :-dy])], dim=1)
        
        if dz > 0:
            grid = torch.cat([torch.zeros_like(grid[:, :, :dz]), grid[:, :, :-dz]], dim=2)
        elif dz < 0:
            grid = torch.cat([grid[:, :, -dz:], torch.zeros_like(grid[:, :, :-dz])], dim=2)
        
        self.voxel_grids[instance_id] = grid
        return True

    def world_to_voxel_offset(self, world_offset):
        """将世界坐标系的偏移转换为体素坐标系的偏移"""
        return world_offset / self.voxel_size

    def visualize_voxels(self, instance_ids=None, show_all=False):
        """
        可视化体素网格
        Args:
            instance_ids: 指定要可视化的物体ID列表，如果为None且show_all=True则显示所有物体
            show_all: 是否显示所有物体的体素
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if show_all:
            # 为不同物体随机分配颜色
            colors = plt.cm.rainbow(np.linspace(0, 1, len(self.voxel_grids)))
            for idx, (obj_id, grid) in enumerate(self.voxel_grids.items()):
                occupied = grid.cpu().numpy()
                x, y, z = np.where(occupied)
                ax.scatter(x, y, z, c=[colors[idx]], alpha=0.6, label=obj_id)
        elif instance_ids is not None:
            # 确保 instance_ids 是列表
            if isinstance(instance_ids, str):
                instance_ids = [instance_ids]
            
            # 指定的物体分配颜色
            colors = plt.cm.rainbow(np.linspace(0, 1, len(instance_ids)))
            for idx, obj_id in enumerate(instance_ids):
                if obj_id in self.voxel_grids:
                    occupied = self.voxel_grids[obj_id].cpu().numpy()
                    x, y, z = np.where(occupied)
                    ax.scatter(x, y, z, c=[colors[idx]], alpha=0.6, label=obj_id)
                else:
                    print(f"Warning: {obj_id} not found in voxel grids")
        else:
            print("No valid instance_ids provided and show_all=False")
            return
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if show_all or (instance_ids and len(instance_ids) > 1):
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title('Voxelized objects')
        plt.tight_layout()
        plt.savefig(f"voxel_visualization_{instance_ids}.png")


def save_voxel_debug_img_plt(obj_manager, output_path):
    """
    [加速版] 使用 matplotlib 渲染体素网格
    通过降采样 (Downsampling) 极大提高渲染速度
    """
    print("\n" + "="*60)
    print("生成体素调试图 (Matplotlib - Fast Mode)...")
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: 未找到 matplotlib")
        return

    vm = obj_manager.voxel_manager
    
    # --- 优化核心: 降采样步长 ---
    # step = 1: 原分辨率 (128^3 -> 200万点, 极慢)
    # step = 2: 1/8 数据量 (64^3, 较快)
    # step = 4: 1/64 数据量 (32^3, 极快)
    # 自动根据分辨率选择步长，保持渲染在 grid < 64 左右
    raw_res = vm.resolution
    max_dim = max(raw_res)
    
    if max_dim >= 64:
        step = 2 # 中等压缩
    else:
        step = 1
        
    print(f"  原始分辨率: {raw_res}, 降采样步长: {step} (加速渲染)")

    # 计算新的分辨率
    # 使用切片 [::step] 后的形状
    temp_slice = np.zeros(raw_res)[::step, ::step, ::step]
    res = temp_slice.shape
    
    # 准备绘图数据
    voxels = np.zeros(res, dtype=bool)
    colors = np.zeros(res + (4,), dtype=np.float32)
    
    palette = [
        (1, 0, 0, 0.6), (0, 1, 0, 0.6), (0, 0, 1, 0.6), (1, 1, 0, 0.6),
        (1, 0, 1, 0.6), (0, 1, 1, 0.6), (1, 0.5, 0, 0.6), (0.5, 0, 1, 0.6)
    ]
    
    idx_counter = 0
    has_data = False
    
    print("  合并并降采样体素数据...")
    for inst_id, grid_tensor in vm.voxel_grids.items():
        if torch.is_tensor(grid_tensor):
            # 这里不用 cpu().numpy() 整个数组，太慢。
            # 先在 GPU 上切片，再转 CPU，大幅减少数据传输
            # 注意：PyTorch 的切片是 view 操作，很快
            sliced_tensor = grid_tensor[::step, ::step, ::step]
            obj_grid = sliced_tensor.detach().cpu().numpy()
        else:
            obj_grid = grid_tensor[::step, ::step, ::step]
            
        if np.sum(obj_grid) == 0:
            continue
            
        voxels |= obj_grid
        color = palette[idx_counter % len(palette)]
        colors[obj_grid] = color # 这一步利用了 numpy 的 boolean indexing
        idx_counter += 1
        has_data = True

    if not has_data:
        print("  警告: 场景为空，未生成图片")
        return

    print("  开始渲染 Plot (加速中)...")
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 关键优化: edgecolor=None 去掉网格线，渲染速度快一倍
    ax.voxels(voxels, facecolors=colors, edgecolor=None, shade=True)
    
    # 设置 Box Aspect 保持比例
    ax.set_box_aspect(res)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"S4 Voxel Debug (Step={step}) - Objs: {idx_counter}")
    
    # 鸟瞰视角
    ax.view_init(elev=30, azim=-45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100) # dpi 稍微调低一点也有助于保存速度
    plt.close(fig)
    
    print(f"  体素图已保存: {output_path}")
    print("="*60 + "\n")


class ObjManager:
    """
    用于管理场景中所有物体以及执行优化程:
     - 维护 Obj 列表并提供碰撞检测、重叠面积计算、移动距离计算、模拟退火等功能
    """
    def __init__(self, precomputed_voxel_dir=None):
        self.obj_dict = {}          # instance_id -> Obj
        self.wall_dict = {}         # wall_id -> Obj
        self.overlap_list = []      # 用于预先存储物体间可能发生重叠的对
        self.initial_state = False  # 记录 overlap_list 是否初始化
        self.total_size = 0         # overlap_list 的大小
        self.carpet_list = ["carpet_0", "rug_0"]
        self.n = 0                  # 物体总数
        self.obj_info = {}          # 存储从 placement_info_new.json 中读取的 obj_info
        self.ground_name = None     # reference_obj
        self.voxel_manager = VoxelManager(precomputed_voxel_dir=precomputed_voxel_dir)
        
        # 修改：使用 defaultdict(float) 自动初始化为 0.0，彻底解决 KeyError
        self.per_obj_loss = defaultdict(float)
        self.temp_per_obj_loss = defaultdict(float)


    def load_data(self, base_dir):
        """
        从 placement_info_new.json 中加载数据，创建 Obj 实例并存储,
        ���初始化 voxel_manager
        """
        with open(f"{base_dir}/placement_info_new.json", 'r') as f:
            placement_info = json.load(f)

        self.obj_info = placement_info["obj_info"]
        self.ground_name = placement_info["reference_obj"]

        # 加正则表达式模式
        skip_pattern = re.compile(r'^(floor_\d+|wall_\d+|scene_camera)')
        
        for instance_id, info in self.obj_info.items():
            # 如果物体名称匹配模式则跳过
            if skip_pattern.match(instance_id):
                print(f"Skipping {instance_id}")
                self.wall_dict[instance_id] = {"pose_matrix_for_blender": info["pose_matrix_for_blender"]}
                continue
                
            obj = Obj(instance_id, info)
            self.obj_dict[instance_id] = obj

        self.voxel_manager.initialize_scene_bounds(self.obj_dict,self.wall_dict)
        for instance_id, obj in self.obj_dict.items():
            print(obj.fbx_path,instance_id)
            mesh_path = Path(obj.fbx_path)
            pose = obj.pose_3d
            self.voxel_manager.voxelize_object(mesh_path, instance_id,pose,scale=[1.1,1.1,1.0])

    def build_bbox_items(self):
        """
        构建 bbox_items 列表，用于后续初始化 overlap list
        [(instance_id, bounding_box), ...]
        """
        bbox_items = []
        for inst_id, obj in self.obj_dict.items():
            bbox_items.append((inst_id, obj.bounding_box))
        return bbox_items

    def init_overlap(self):
        """
        初始化 overlap_list，用于加速频繁的重叠检测:
          - 使用 bbox 快速预筛选能发生碰撞的物体对
          - 如果两个物体的 bbox  z 轴或 x,y 平面上不可能重叠，就不放入 overlap_list
        """
        if self.initial_state:
            return

        bbox_items = self.build_bbox_items()
        self.n = len(bbox_items)
        self.overlap_list = []
        self.total_size = 0

        for i in range(self.n):
            instance_id_i, bbox_i = bbox_items[i]
            overlap_i = []
            
            # 跳过地毯等特殊物体
            if instance_id_i in self.carpet_list:
                self.overlap_list.append([])
                continue

            for j in range(i + 1, self.n):
                instance_id_j, bbox_j = bbox_items[j]
                
                # 跳过地毯
                if instance_id_j in self.carpet_list:
                    continue

                # 跳过父子关系的体对
                if (self.obj_dict[instance_id_i].parent_id == instance_id_j or 
                    self.obj_dict[instance_id_j].parent_id == instance_id_i):
                    continue

                # 检查 z 轴方向是否重叠
                if bbox_i["min"][2] >= bbox_j["max"][2] - eps or bbox_j["min"][2] >= bbox_i["max"][2] - eps:
                    continue

                # 检查 x,y 平面上的距离
                # 考虑到物体可能动，在原始 bbox 基础上增加一定余量
                margin_x = (bbox_j["length"][0] + bbox_i["length"][0]) * 0.5
                margin_y = (bbox_j["length"][1] + bbox_i["length"][1]) * 0.5
                
                if (bbox_i["min"][0] >= bbox_j["max"][0] + margin_x or 
                    bbox_j["min"][0] >= bbox_i["max"][0] + margin_x):
                    continue
                    
                if (bbox_i["min"][1] >= bbox_j["max"][1] + margin_y or 
                    bbox_j["min"][1] >= bbox_i["max"][1] + margin_y):
                    continue

                # 将可能发生碰撞的物体对添加到列表
                overlap_i.append(j)

            self.overlap_list.append(overlap_i)
            self.total_size += len(overlap_i)
            
        # 修改：初始化完成后，立即重置一次 loss 字典，确保外部调用 calc 不报错
        self.reset_temp_loss()
        # 同步 per_obj_loss 以便第一次随机采样有数据
        self.per_obj_loss = self.temp_per_obj_loss.copy()

        self.initial_state = True

    def get_obj_index(self, inst_id, bbox_items):
        """
        返回在 bbox_items 列表中的索引, 用于在 overlap_list 中找到相目
        """
        for idx, (iid, _) in enumerate(bbox_items):
            if iid == inst_id:
                return idx
        return -1

    def reset_temp_loss(self):
        """重置临时 Loss 记录"""
        # 重新生成一个 defaultdict，确保之前的累积清零
        self.temp_per_obj_loss = defaultdict(float)
        # 显式把所有 key 置为 0，防止有些物体没有任何 loss 导致 key 不存在
        for k in self.obj_dict.keys():
            self.temp_per_obj_loss[k] = 0.0

    def calc_overlap_area(self, debug_mode=False, batch_size=8):
        """使用体素化方法批量计算重叠，只检查预筛选的物体对"""
        total_overlap = 0
        bbox_items = self.build_bbox_items()
        
        # 收集所有需要检查的物体对
        pairs_to_check = []
        for i, overlap_indices in enumerate(self.overlap_list):
            if not overlap_indices:
                continue
            id_i = bbox_items[i][0]
            for j in overlap_indices:
                id_j = bbox_items[j][0]
                pairs_to_check.append((id_i, id_j))
        
        # 批量处理物体对
        for start_idx in range(0, len(pairs_to_check), batch_size):
            batch_pairs = pairs_to_check[start_idx:start_idx + batch_size]
            
            # 准备这个批次的网格
            grids_1 = []
            grids_2 = []
            for id_1, id_2 in batch_pairs:
                grids_1.append(self.voxel_manager.voxel_grids[id_1])
                grids_2.append(self.voxel_manager.voxel_grids[id_2])
            
            # 将网格堆叠成批次
            batch_grids_1 = torch.stack(grids_1)  # [batch_size, *grid_shape]
            batch_grids_2 = torch.stack(grids_2)  # [batch_size, *grid_shape]
            
            # 批量计算重叠
            batch_overlap = torch.logical_and(batch_grids_1, batch_grids_2).sum(dim=(1,2,3))
            total_overlap += batch_overlap.sum().item()
            
            # --- 修改：安全的 Loss 累加 ---
            overlap_vals = batch_overlap.tolist()
            for idx, val in enumerate(overlap_vals):
                if val > 0:
                    id_1, id_2 = batch_pairs[idx]
                    # defaultdict 不需要检查 key 是否存在
                    self.temp_per_obj_loss[id_1] += val
                    self.temp_per_obj_loss[id_2] += val
            # ---------------------------
            
            if debug_mode:
                # 输出这个批次中有重叠的物体对
                for idx, (id_1, id_2) in enumerate(batch_pairs):
                    overlap = overlap_vals[idx]
                    if overlap > 0:
                        print(f"Overlap between {id_1} and {id_2}: {overlap}")
        
        return total_overlap

    def calc_movement(self):
        """
        计算所有物体的移动距离(平方和)
        """
        total_move = 0
        for inst_id, obj in self.obj_dict.items():
            ox, oy = obj.original_pos
            cx, cy = obj.current_pos
            total_move += (cx - ox)**2 + (cy - oy)**2
        return total_move

    def calc_constraints(self):
        """
        计算物体相对于其父物体的越界程度, 若出某个范围则产生罚分
        """
        k = 2
        cost = 0
        bbox_items = self.build_bbox_items()

        for inst_id, obj in self.obj_dict.items():
            parent_id = obj.parent_id
            if parent_id is None or parent_id not in self.obj_dict:
                continue
            if obj.relation == "inside":
                # 内部关系暂时忽略
                continue

            fa_obj = self.obj_dict[parent_id]
            fa_bbox = fa_obj.bounding_box  # 父物体的 bbox

            # 当前物体位置
            cx, cy = obj.current_pos
            length_x, length_y = obj.bounding_box["length"][0], obj.bounding_box["length"][1]

            # 检查是否在父物体 bbox 的某个范围内(这里是简单示例, ��据需要微调)
            if (cx - length_x/k >= fa_bbox["min"][0] and
                cx + length_x/k <= fa_bbox["max"][0] and
                cy - length_y/k >= fa_bbox["min"][1] and
                cy + length_y/k <= fa_bbox["max"][1]):
                continue

            # 计算与父物体 bbox 中心的距离并累计
            this_cost = (cx - fa_bbox["x"])**2 + (cy - fa_bbox["y"])**2
            cost += this_cost
            
            # --- 修改：安全累加 ---
            self.temp_per_obj_loss[inst_id] += this_cost * 100
            # --------------------

        return cost

    def try_perturb_random_obj(self, iteration, max_iterations):
        """
        随机扰动一个物体 (修改版: 加权采样 + 正态分布步长)
        """
        inst_ids = list(self.obj_dict.keys())
        
        # --- 策略1：物体选择 - 加权随机采样 (Weighted Sampling) ---
        # 优先选择 Loss 大的物体
        epsilon = 1.0 
        weights = []
        for inst_id in inst_ids:
            # defaultdict 保证了即使没有记录也是 0.0
            w = self.per_obj_loss[inst_id]
            weights.append(w + epsilon)
        
        weights = np.array(weights)
        obj_probs = weights / np.sum(weights)
        
        # 按概率选择物体
        chosen_id = np.random.choice(inst_ids, p=obj_probs)
        chosen_obj = self.obj_dict[chosen_id]

        old_x, old_y = chosen_obj.current_pos

        # --- 策略2：步长选择 - 动态正态分布衰减 (Gaussian Decay) ---
        # 随着 iteration 增加，最大步长从 20 线性衰减到 1
        progress = iteration / max_iterations
        
        # 计算当前允许的最大步长 (20 -> 1)
        current_max_scale = max(1, int(20 * (1.0 - progress)))
        
        if current_max_scale == 1:
            scale = 1
        else:
            # 生成候选步长 [1, 2, ..., current_max_scale]
            candidates = np.arange(1, current_max_scale + 1)
            
            # 使用"半正态分布"逻辑生成概率权重
            # 我们希望 1 的概率最大，current_max_scale 的概率最小
            # mu = 1 (分布中心在最左侧)
            # sigma 控制衰减速度。设为 current_max_scale / 2.5 可以保证平滑的拖尾
            sigma = max(1.0, current_max_scale / 2.5)
            
            # 高斯公式: exp(- (x - mu)^2 / (2 * sigma^2))
            # x 为步长，mu 为 1
            scale_weights = np.exp(-((candidates - 1)**2) / (2 * sigma**2))
            
            # 归一化，使其和为 1
            scale_probs = scale_weights / np.sum(scale_weights)
            
            # 按概率分布采样步长
            scale = np.random.choice(candidates, p=scale_probs)
        
        # 生成扰动方向
        move_x = np.random.choice([True, False])
        move_positive = np.random.choice([True, False])
        voxel_size = self.voxel_manager.voxel_size
        perturbation = np.zeros(2)

        if move_x:
            perturbation[0] = scale * voxel_size[0] if move_positive else -scale * voxel_size[0]
        else:
            perturbation[1] = scale * voxel_size[1] if move_positive else -scale * voxel_size[1]

        # 墙壁约束处理
        wall_id = chosen_obj.is_against_wall
        if wall_id is not None:
            # againstWall 可能是列表 ["wall_0"] 也可能是纯字符串 "wall_0"，
            # 直接 wall_id[0] 对字符串会取到首字符 'w' 导致 KeyError。统一归一化。
            if isinstance(wall_id, (list, tuple)):
                wall_key = wall_id[0] if len(wall_id) > 0 else None
            else:
                wall_key = wall_id
            if wall_key is not None and wall_key in self.wall_dict:
                wall_pose = self.wall_dict[wall_key]["pose_matrix_for_blender"]
                wall_np = np.array(wall_pose)
                normal_3d = wall_np[:3, :3] @ np.array([0, 0, 1])
                normal_2d = normal_3d[:2]
                normal_len = np.linalg.norm(normal_2d)
                if normal_len > 1e-9:
                    normal_2d = normal_2d / normal_len
                    dot_val = np.dot(perturbation, normal_2d)
                    perturbation = perturbation - dot_val * normal_2d
        
        # 特殊处理 floor_lamp
        if chosen_obj.instance_id == "floor_lamp_0":
            perturbation[1] *= 10

        voxel_perturbation = np.array([int(round(perturbation[0] / voxel_size[0])), 
                                       int(round(perturbation[1] / voxel_size[1])), 0])
        
        move_success = self.voxel_manager.move_grid(chosen_obj.instance_id, voxel_perturbation)
        if not move_success:
            return lambda: None
            
        chosen_obj.current_pos[0] += perturbation[0]
        chosen_obj.current_pos[1] += perturbation[1]

        def revert():
            chosen_obj.current_pos[0] = old_x
            chosen_obj.current_pos[1] = old_y
            self.voxel_manager.move_grid(chosen_obj.instance_id, -voxel_perturbation)

        return revert

    def simulated_annealing(self, initial_temp, alpha, max_iterations, penalty_factor):
        """模拟退火优化 (修改：管理 Loss 状态)"""
        M = 100
        log_time = 400
        
        # 初始化状态
        self.reset_temp_loss()
        # 计算初始能量的同时，会填充 self.temp_per_obj_loss
        current_energy = ( M*( self.calc_overlap_area() + self.calc_constraints() ) 
                           + self.calc_movement() )
        
        # 初始状态被接受，同步 Loss 记录
        self.per_obj_loss = self.temp_per_obj_loss.copy()
        
        temperature = initial_temp

        for iteration in range(max_iterations):
            # 传入 iteration 和 max_iterations 用于计算动态步长
            revert_callback = self.try_perturb_random_obj(iteration, max_iterations)

            # 准备计算新状态能量，重置临时 Loss
            self.reset_temp_loss()
            
            new_energy = ( M*( self.calc_overlap_area() + self.calc_constraints() ) 
                           + self.calc_movement() )

            if iteration % log_time == 0:
                print(f"Iteration {iteration}, Energy: {new_energy:.2f}, Temp: {temperature:.4f}")

            delta_energy = new_energy - current_energy
            
            # Metropolis 准则
            if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
                # 接受新状态
                current_energy = new_energy
                # 关键：更新用于下一次采样的权重分布
                self.per_obj_loss = self.temp_per_obj_loss.copy()
            else:
                # 拒绝新状态，回退
                revert_callback()
                # per_obj_loss 保持不变（还是上一次成功状态的 Loss）

            temperature *= alpha

            if temperature < 1e-3 and current_energy == 0:
                print("Converged early.")
                break

        return current_energy

    def save_to_json(self, file_path, data):
        """
        将数据存储到 JSON 文件
        """
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def main(self, base_dir):
        """
        执行流程:
          1. 读取 placement_info_new.json
          2. 预处理并初始化 overlap
          3. 运行模拟退火
          4. 将结果写回 JSON
        """
        self.load_data(base_dir)
        self.init_overlap()  # 准备 overlap_list1
        initial_temp = 100.0
        alpha = 0.99
        max_iterations = 10000
        penalty_factor = 1000.0
        # 添加可视化调用
        print("Visualizing initial voxel grids...")
        self.voxel_manager.visualize_voxels(show_all=True)

        final_energy = self.simulated_annealing(initial_temp, alpha, max_iterations, penalty_factor)

        # 将最终位置写进 final_pos.json
        final_position = {}
        for inst_id, obj in self.obj_dict.items():
            final_position[inst_id] = {
                "x": float(obj.current_pos[0]),
                "y": float(obj.current_pos[1])
            }
            moved_dist = math.sqrt(
                (obj.current_pos[0] - obj.original_pos[0])**2 + 
                (obj.current_pos[1] - obj.original_pos[1])**2
            )
            print(inst_id, "移动距离:", moved_dist)

        # 添加可视化调用
        print("Visualizing initial voxel grids...")
        self.voxel_manager.visualize_voxels(show_all=True)
        print("Final Overlap:", self.calc_overlap_area(debug_mode=True))
        print("Final Constraints:", self.calc_constraints())
        print("Final Energy:", final_energy)

        self.save_to_json(f"{base_dir}/final_pos.json", final_position)


class RelativePoseManager:
    def __init__(self, obj_list, tree_sons, output_data_s2):
        self.obj_list = obj_list
        self.tree_sons = tree_sons
        self.output_data_s2 = output_data_s2
        self.relative_poses = {}

    def record_relative_poses(self, parent_obj, sons_list):
        for son_name in sons_list:
            son_obj = bpy.data.objects[son_name]
            
            # 检查父物体的 scale 是否有 0 分量（会导致矩阵不可逆）
            parent_scale = list(parent_obj.scale)
            if 0 in parent_scale or min(abs(s) for s in parent_scale) < 1e-6:
                print(f"[Warning] Skipping {parent_obj.name} (scale has zero: {parent_scale})")
                continue
            
            # 计算相对变换矩阵
            relative_matrix = parent_obj.matrix_world.inverted() @ son_obj.matrix_world
            self.relative_poses[son_name] = relative_matrix
            
            # 递归处理当前子对象的子对象
            if son_name in self.tree_sons:
                self.record_relative_poses(son_obj, self.tree_sons[son_name])

    def record_all(self):
        for obj_name, obj in self.obj_list.items():
            if obj_name not in self.relative_poses and obj_name in self.tree_sons and self.output_data_s2["obj_info"][obj_name]["againstWall"] is not None:
                self.record_relative_poses(obj, self.tree_sons[obj_name])

    def restore_relative_poses(self, parent_obj, sons_list):
        for son_name in sons_list:
            son_obj = bpy.data.objects[son_name]
            
            if son_name in self.relative_poses:
                # 使用存储的相对矩阵恢复子对象的矩阵
                son_obj.matrix_world = parent_obj.matrix_world @ self.relative_poses[son_name]
            
            # 递归恢复子对象
            if son_name in self.tree_sons:
                self.restore_relative_poses(son_obj, self.tree_sons[son_name])

    def restore_all(self):
        stack = [(obj_name, obj) for obj_name, obj in self.obj_list.items() if obj_name in self.tree_sons]
        processed = set()  # 用于跟踪已处理的对象
        
        while stack:
            parent_name, parent_obj = stack.pop()
            if parent_name in processed:
                continue  # 如果这个对象已经处理过，跳过它
            processed.add(parent_name)
            
            sons_list = self.tree_sons[parent_name]
            
            for son_name in sons_list:
                son_obj = bpy.data.objects[son_name]
                
                if son_name in self.relative_poses:
                    son_obj.matrix_world = parent_obj.matrix_world @ self.relative_poses[son_name]
                
                if son_name in self.tree_sons and son_name not in processed:
                    stack.append((son_name, son_obj))



'''
下面开始是内部摆放的算法函数
'''

def get_closest_subspace(obj_name, parent_name, subspaces_info):
    obj = bpy.data.objects[obj_name]
    parent_obj = bpy.data.objects[parent_name]

    original_center = obj.matrix_world.translation.copy()
    parent_center = parent_obj.matrix_world.translation.copy()

    # 找到最靠近世界坐标系 z 轴的那个轴
    identity_axes = [Vector(axis) for axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]
    axes = [parent_obj.matrix_world.to_3x3() @ axis for axis in identity_axes]
    z_axis = Vector((0, 0, 1))
    closest_axis = max(axes, key=lambda axis: abs(axis.dot(z_axis)))

    # 计算物体中心到该轴的投影距离
    center_to_axis_projection = (original_center - parent_center).dot(closest_axis)

    # 计算 parent_obj 在该轴上的投影长度
    bounds = [parent_obj.matrix_world @ Vector(b) for b in parent_obj.bound_box]
    axis_projections = [(b - parent_center).dot(closest_axis) for b in bounds]
    parent_axis_length = max(axis_projections) - min(axis_projections)

    # 计算到表面的距离
    distance_to_surface = abs(abs(center_to_axis_projection) - parent_axis_length / 2)

    # 找到最近的子空间
    min_distance = float('inf')
    closest_subspace_info = None

    for subspace in subspaces_info:
        subspace_center = bpy.data.objects[subspace['name']].matrix_world.translation
        subspace_projection = (subspace_center - parent_center).dot(closest_axis)
        distance = abs(center_to_axis_projection - subspace_projection)
        if distance < min_distance:
            min_distance = distance
            closest_subspace_info = subspace

    # 判断是 on 还是 inside
    if distance_to_surface < min_distance:
        return "on", None
    else:
        return "inside", closest_subspace_info
    
def resolve_collisions_in_subspace(objects, subspace_obj, max_attempts=100):
    items_failed_and_del = []
    center = sum((obj.location for obj in objects), Vector()) / len(objects)
    directions = [obj.location - center for obj in objects]
    average_direction = sum(directions, Vector()).normalized()
    local_average_direction = subspace_obj.matrix_world.inverted() @ average_direction
    # Define local axes excluding the Z axis
    local_axes = [Vector((1, 0, 0)), Vector((0, 1, 0))]
    axis_scores = [local_average_direction.dot(axis) for axis in local_axes]
    main_axis = local_axes[np.argmax(axis_scores)]
    world_main_axis = subspace_obj.matrix_world.to_3x3() @ main_axis

    for _ in range(max_attempts):
        collision_found = False
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                if check_collision(obj1, obj2):
                    collision_found = True
                    # Try to resolve collision for obj2
                    if not resolve_single_collision(obj2, objects, subspace_obj, world_main_axis, main_axis):
                        print(f'failed, delete {obj2.name}')
                        items_failed_and_del.append(obj2.name)
                        # If unable to resolve, remove obj2
                        objects.remove(obj2)
                        bpy.data.objects.remove(obj2)
                    break  # Exit loop after handling the first collision
            if collision_found:
                break  # Exit outer loop if any collision was found and handled
        if not collision_found:
            break  # Exit if no collisions were found
    return items_failed_and_del

def resolve_single_collision(obj, objects, subspace_obj, world_main_axis, main_axis, max_adjustments=1000):
    """
    Attempt to adjust the position of a single object to resolve collisions
    along the main axis.
    """
    move_distance = 0.001  # Adjust move distance as needed

    # Transform main axis to world space
    initial_location = obj.location.copy()

    # Try moving in the positive direction
    for _ in range(max_adjustments):
        obj.location += world_main_axis * move_distance
        bpy.context.view_layer.update()

        # Check if no collisions and still within subspace
        if not any(check_collision(obj, other) for other in objects if other != obj) and check_within_subspace_direction(obj, subspace_obj, main_axis):
            return True  # Successfully resolved collision

        # Stop if out of subspace
        if not check_within_subspace_direction(obj, subspace_obj, main_axis):
            break

    # Reset to initial location
    obj.location = initial_location
    bpy.context.view_layer.update()

    # Try moving in the negative direction
    for _ in range(max_adjustments):
        obj.location -= world_main_axis * move_distance
        bpy.context.view_layer.update()

        # Check if no collisions and still within subspace
        if not any(check_collision(obj, other) for other in objects if other != obj) and check_within_subspace_direction(obj, subspace_obj, main_axis):
            return True  # Successfully resolved collision

        # Stop if out of subspace
        if not check_within_subspace_direction(obj, subspace_obj, main_axis):
            break

    # Reset to initial location if unsuccessful
    obj.location = initial_location
    bpy.context.view_layer.update()

    return False  # Could not resolve collision

def check_collision(obj1, obj2):
    """
    Check if two objects are colliding.
    """
    obj1_bounds = [obj1.matrix_world @ Vector(corner) for corner in obj1.bound_box]
    obj2_bounds = [obj2.matrix_world @ Vector(corner) for corner in obj2.bound_box]

    obj1_min = Vector((min(v[i] for v in obj1_bounds) for i in range(3)))
    obj1_max = Vector((max(v[i] for v in obj1_bounds) for i in range(3)))
    obj2_min = Vector((min(v[i] for v in obj2_bounds) for i in range(3)))
    obj2_max = Vector((max(v[i] for v in obj2_bounds) for i in range(3)))

    return all(obj1_max[i] > obj2_min[i] and obj1_min[i] < obj2_max[i] for i in range(3))

def check_within_subspace_direction(obj, subspace_obj, main_axis):
    """
    Check if the object is entirely within the subspace boundaries along a specific direction.
    """
    # Normalize the direction
    direction = main_axis.normalized()
    
    # Get local coordinates of the object's bounding box in the subspace's local space
    obj_bounds_local = [subspace_obj.matrix_world.inverted() @ (obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]
    
    # Get local coordinates of the subspace's bounding box
    subspace_bounds_local = [Vector(corner) for corner in subspace_obj.bound_box]
    
    # Project the bounding box corners onto the direction vector
    obj_projections = [corner.dot(direction) for corner in obj_bounds_local]
    subspace_projections = [corner.dot(direction) for corner in subspace_bounds_local]
    
    # Calculate min and max projections for the object and subspace
    obj_min_proj = min(obj_projections)
    obj_max_proj = max(obj_projections)
    subspace_min_proj = min(subspace_projections)
    subspace_max_proj = max(subspace_projections)
    
    # Check if the object is within the subspace boundaries along the direction
    tolerance = 1e-5
    if obj_min_proj < subspace_min_proj - tolerance or obj_max_proj > subspace_max_proj + tolerance:
        return False

    return True


def find_closest_subspace(vase_name, subspaces):
    vase = bpy.data.objects[vase_name]
    vase_center = vase.matrix_world.translation
    min_distance = float('inf')
    closest_subspace = None
    
    for subspace in subspaces:
        subspace_center = bpy.data.objects[subspace['name']].matrix_world.translation
        distance = (vase_center - subspace_center).length
        if distance < min_distance:
            min_distance = distance
            closest_subspace = subspace
    
    return closest_subspace

def align_obj_to_closest_subspace(obj_name, closest_subspace_info):
    obj = bpy.data.objects[obj_name]
    original_center = obj.matrix_world.translation.copy()

    subspace_obj = bpy.data.objects[closest_subspace_info['name']]
    subspace_matrix = subspace_obj.matrix_world

    # 获取子空间的z轴
    subspace_z = subspace_matrix.to_3x3().col[2]
    
    # 检查 subspace_z 是否为零向量
    if subspace_z.length < 1e-6:
        print(f"[Warning] align_obj_to_closest_subspace: subspace {closest_subspace_info['name']} has zero-length z-axis, skipping {obj_name}")
        return

    # 获取物体的三个轴
    obj_axes = [obj.matrix_world.to_3x3().col[i] for i in range(3)]
    
    # 过滤掉零长度的轴（可能是 scale 有零分量导致的）
    valid_axes = [axis for axis in obj_axes if axis.length > 1e-6]
    if not valid_axes:
        print(f"[Warning] align_obj_to_closest_subspace: {obj_name} has all zero-length axes (scale={list(obj.scale)}), skipping")
        return

    # 找到与子空间z轴最接近的物体轴（只在有效轴中选择）
    closest_axis = max(valid_axes, key=lambda axis: abs(axis.dot(subspace_z)))

    # 确保方向正确
    if closest_axis.dot(subspace_z) < 0:
        closest_axis = -closest_axis

    # 计算旋转矩阵
    rotation_axis = closest_axis.cross(subspace_z)
    
    # 检查 closest_axis 是否为零向量（理论上不会走到这里，但做个保险）
    if closest_axis.length < 1e-6:
        print(f"[Warning] align_obj_to_closest_subspace: {obj_name} closest_axis is zero-length, skipping")
        return
    
    angle = closest_axis.angle(subspace_z)

    if rotation_axis.length > 0.0001:
        # 创建旋转矩阵
        rotation_matrix = Matrix.Rotation(angle, 4, rotation_axis)

        # 保存当前的变换矩阵
        original_matrix = obj.matrix_world.copy()

        # 平移到原点，旋转，然后平移回去
        obj.matrix_world.translation -= original_center
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        obj.matrix_world.translation += original_center

        # 应用原始的平移
        obj.matrix_world.translation = original_matrix.translation

    # 恢复缩放
    bpy.context.view_layer.update()

    # 将物体的边界转换到子空间坐标系
    obj_bounds_local = [subspace_obj.matrix_world.inverted() @ (obj.matrix_world @ Vector(corner)) for corner in obj.bound_box]

    # 计算物体在子空间坐标系中的最小和最大坐标
    obj_min_local = Vector((min(v[i] for v in obj_bounds_local) for i in range(3)))
    obj_max_local = Vector((max(v[i] for v in obj_bounds_local) for i in range(3)))

    # 使用子空间的尺寸计算边界
    subspace_min_local = -0.5 * subspace_obj.dimensions
    subspace_max_local = 0.5 * subspace_obj.dimensions
    # 计算平移量，使物体进入子空间
    translation_offset_local_min = Vector((0, 0, 0))
    translation_offset_local_max = Vector((0, 0, 0))
    translation_offset_local = Vector((0, 0, 0))
    for i in range(3):
        if obj_min_local[i] < subspace_min_local[i] or obj_max_local[i] > subspace_max_local[i]:
            translation_offset_local_min[i] = subspace_min_local[i] - obj_min_local[i]
            translation_offset_local_max[i] = subspace_max_local[i] - obj_max_local[i]
            translation_offset_local[i] = (translation_offset_local_max[i] + translation_offset_local_min[i])/2

    # 将平移量从子空间坐标系转换回世界坐标系
    translation_offset_world = subspace_obj.matrix_world.to_3x3() @ translation_offset_local

    # 应用平移
    obj.matrix_world.translation += translation_offset_world
    bpy.context.view_layer.update()

    # 检查并缩放物体以适应子空间
    scale_obj_to_fit_subspace(obj, subspace_obj)
    bpy.context.view_layer.update()

    # [Fix] Apply scale so dimensions are baked into mesh, avoiding offset issues in alignment
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # 计算需要的平移量，使物体底部与子空间底部对齐
    # 下面这个代码有问题
    # 此处先把所有物体缩放应用到自己本身，这样他们的scale都是111  就不用操心move_obj_along_closest_axis_to_z中物体和subspace_obj的缩放问题了
    move_obj_along_closest_axis_to_z(obj, subspace_obj)
    bpy.context.view_layer.update()
    return

def scale_obj_to_fit_subspace(obj, subspace_obj):
    # 计算物体和子空间的边界
    obj_bounds = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    subspace_bounds = [subspace_obj.matrix_world @ Vector(corner) for corner in subspace_obj.bound_box]

    # 计算物体的最小和最大坐标
    obj_min = Vector((min(v[i] for v in obj_bounds) for i in range(3)))
    obj_max = Vector((max(v[i] for v in obj_bounds) for i in range(3)))

    # 计算子空间的最小和最大坐标
    subspace_min = Vector((min(v[i] for v in subspace_bounds) for i in range(3)))
    subspace_max = Vector((max(v[i] for v in subspace_bounds) for i in range(3)))

    # 检查物体是否在子空间内
    scale_factor = 1.0
    for i in range(3):
        obj_size = obj_max[i] - obj_min[i]
        subspace_size = subspace_max[i] - subspace_min[i]

        if obj_size > subspace_size:
            scale_factor = min(scale_factor, subspace_size / obj_size)

    # 如果需要缩放
    if scale_factor < 1.0:
        obj.scale *= scale_factor
        obj.scale *= 0.8 #不能太大
        bpy.context.view_layer.objects.active = obj
        bpy.context.view_layer.update()

    return scale_factor

def calculate_centered_translation(obj_min, obj_max, subspace_min, subspace_max, translation_offset):
    # 计算物体新的中心位置
    new_obj_center = (obj_min + obj_max) * 0.5 + translation_offset
    # 计算子空间中心位置
    subspace_center = (subspace_min + subspace_max) * 0.5

    # 计算中心偏移
    center_offset = subspace_center - new_obj_center

    # 只沿着进入方向平移
    for i in range(3):
        if translation_offset[i] != 0:
            translation_offset[i] += center_offset[i]

    return translation_offset

def find_closest_axis_to_world_z(obj):
    matrix_world = obj.matrix_world
    rotation_matrix = matrix_world.to_3x3().normalized()

    world_z = Vector((0, 0, 1))
    min_angle = float('inf')
    min_index = -1

    for i in range(3):
        axis = rotation_matrix.col[i]
        angle = world_z.angle(axis)

        if angle < min_angle:
            min_angle = angle
            min_index = i

    return rotation_matrix.col[min_index], min_index

def align_closest_axis_to_world_z(obj):
    closest_axis, axis_index = find_closest_axis_to_world_z(obj)

    # Calculate the rotation needed to align the closest axis to the world z-axis
    world_z = Vector((0, 0, 1))
    rotation_axis = closest_axis.cross(world_z)
    angle = closest_axis.angle(world_z)

    if rotation_axis.length > 0:
        rotation_axis.normalize()
        # Create a rotation matrix from the axis and angle
        rotation_matrix = Matrix.Rotation(angle, 4, rotation_axis)
        
        # Extract the translation part of the matrix
        translation = obj.matrix_world.to_translation()

        # Apply the rotation to the object's local rotation matrix
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        obj.matrix_world.translation = translation
        bpy.context.view_layer.update()

def align_carpet_z_to_world_z_positive(obj):
    """
    强制将地毯的局部 Z 轴正方向对齐到世界 Z 轴正方向（朝上）
    然后绕 Z 轴旋转最小角度，使地毯的 X 轴与最近墙的法向对齐
    这确保地毯平铺在地面上，正面朝上，且与墙壁平行/垂直
    """
    # Step 1: 将地毯的 Z 轴正方向对齐到世界 Z 轴正方向
    local_z_world = obj.matrix_world.to_3x3() @ Vector((0, 0, 1))
    local_z_world.normalize()
    
    world_z = Vector((0, 0, 1))
    
    # 计算旋转轴和角度
    rotation_axis = local_z_world.cross(world_z)
    angle = local_z_world.angle(world_z)
    
    if rotation_axis.length > 1e-6:
        rotation_axis.normalize()
        rotation_matrix = Matrix.Rotation(angle, 4, rotation_axis)
        
        translation = obj.matrix_world.to_translation()
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        obj.matrix_world.translation = translation
        bpy.context.view_layer.update()
    elif local_z_world.dot(world_z) < 0:
        # 如果 Z 轴完全相反（旋转轴长度接近 0 但点积为负），需要翻转 180 度
        rotation_matrix = Matrix.Rotation(math.pi, 4, Vector((1, 0, 0)))
        translation = obj.matrix_world.to_translation()
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        obj.matrix_world.translation = translation
        bpy.context.view_layer.update()
    
    # Step 2: 绕 Z 轴旋转最小角度，使地毯 X 轴与最近墙的法向对齐
    # 收集所有墙的法向（投影到 XY 平面）
    wall_normals = []
    for wall_obj in bpy.data.objects:
        if re.match(r'^wall_\d+$', wall_obj.name):
            # 墙的法向是其局部 Z 轴在世界坐标系中的方向
            wall_normal = wall_obj.matrix_world.to_3x3() @ Vector((0, 0, 1))
            wall_normal.z = 0  # 投影到 XY 平面
            if wall_normal.length > 1e-6:
                wall_normal.normalize()
                wall_normals.append(wall_normal)
    
    if not wall_normals:
        return  # 没有墙，无法对齐
    
    # 获取地毯当前的 X 轴方向（投影到 XY 平面）
    carpet_x = obj.matrix_world.to_3x3() @ Vector((1, 0, 0))
    carpet_x.z = 0
    if carpet_x.length < 1e-6:
        return  # X 轴几乎垂直，无法在 XY 平面对齐
    carpet_x.normalize()
    
    # 找到需要旋转的最小角度
    # 考虑墙法向的正负方向（因为对齐到 +normal 或 -normal 都可以）
    min_angle = float('inf')
    best_rotation = 0
    
    for wall_normal in wall_normals:
        # 检查正方向和负方向
        for sign in [1, -1]:
            target = wall_normal * sign
            # 计算从 carpet_x 到 target 的旋转角度（绕 Z 轴）
            dot = carpet_x.dot(target)
            dot = max(-1, min(1, dot))  # 限制在 [-1, 1] 范围内
            angle = math.acos(dot)
            
            # 确定旋转方向
            cross = carpet_x.cross(target)
            if cross.z < 0:
                angle = -angle
            
            if abs(angle) < abs(min_angle):
                min_angle = angle
                best_rotation = angle
    
    # 应用绕 Z 轴的旋转
    if abs(best_rotation) > 1e-6:
        rotation_matrix = Matrix.Rotation(best_rotation, 4, Vector((0, 0, 1)))
        translation = obj.matrix_world.to_translation()
        obj.matrix_world = rotation_matrix @ obj.matrix_world
        obj.matrix_world.translation = translation
        bpy.context.view_layer.update()
        
def move_obj_along_closest_axis_to_z(obj, target_obj):
    # 找到 target_obj 的最近轴
    closest_axis, axis_index = find_closest_axis_to_world_z(target_obj)

    # 计算物体和目标对象沿着该轴的最低点
    obj_low_point = min(
        [obj.matrix_world @ Vector(corner) for corner in obj.bound_box],
        key=lambda x: x.dot(closest_axis)
    )
    target_low_point = min(
        [target_obj.matrix_world @ Vector(corner) for corner in target_obj.bound_box],
        key=lambda x: x.dot(closest_axis)
    )

    # 计算沿该轴的移动距离
    move_distance = target_low_point.dot(closest_axis) - obj_low_point.dot(closest_axis)

    # 创建移动向量
    move_vector = closest_axis * move_distance

    # 应用移动
    obj.location += move_vector
    
    bpy.context.view_layer.objects.active = obj
    bpy.context.view_layer.update()
    

def create_subspace(name, parent_name, transform_matrix, scale_ratio):
    parent = bpy.data.objects[parent_name]
    parent_size = parent.dimensions
    parent_matrix = parent.matrix_world
    
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    cube = bpy.context.active_object
    cube.name = name
    
    # 计算缩放
    scale = [parent_size[i] * scale_ratio[i] for i in range(3)]
    
    # 应用变换矩阵
    local_matrix = Matrix(transform_matrix)
    
    # 先应用局部变换，再设置缩放
    cube.matrix_world = parent_matrix @ local_matrix
    cube.scale = scale
    
    bpy.context.view_layer.objects.active = cube
    bpy.context.view_layer.update()
    return

# 计算物体的几何中心
def calculate_geometric_center(obj):
    local_center = 0.125 * sum((Vector(b) for b in obj.bound_box), Vector())
    return obj.matrix_world @ local_center

'''
下面是对墙和地面的位姿的纠正
'''
def align_wall_to_axes(walls, ground):
    for wall in walls:
        geom_center = calculate_geometric_center(wall)
        
        # 将墙体的原点设置为几何中心
        bpy.context.view_layer.objects.active = wall
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        
        ## 离世界坐标系z轴最近的那个轴对齐z轴正或负轴
        # 获取墙体的局部坐标系轴
        local_axes = [wall.matrix_world.to_3x3() @ Vector(axis) for axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]
        
        # 找到最接近z轴的局部轴
        world_z_axis = Vector((0, 0, 1))
        closest_to_z = max(local_axes, key=lambda axis: abs(axis.dot(world_z_axis)))
        
        # 计算与z轴正方向和负方向的角度
        angle_to_positive_z = closest_to_z.angle(world_z_axis)
        angle_to_negative_z = closest_to_z.angle(-world_z_axis)
        
        # 选择最小的旋转角度
        if angle_to_positive_z < angle_to_negative_z:
            target_z_axis = world_z_axis
            z_angle = angle_to_positive_z
        else:
            target_z_axis = -world_z_axis
            z_angle = angle_to_negative_z
        
        # 计算旋转轴
        z_rotation_axis = closest_to_z.cross(target_z_axis)
        if z_rotation_axis.length > 0:
            z_rotation_axis.normalize()
            z_rotation_matrix = Matrix.Rotation(z_angle, 4, z_rotation_axis)
            wall.matrix_world = Matrix.Translation(geom_center) @ z_rotation_matrix @ Matrix.Translation(-geom_center) @ wall.matrix_world
            
        bpy.context.view_layer.objects.active = wall
        bpy.context.view_layer.update()
        
    wall_nums = len(walls)
    if wall_nums == 3:
        for wall in walls:
            ## 离世界坐标系x轴最近的那个轴对齐x轴正或负轴
            # 更新局部轴
            geom_center = calculate_geometric_center(wall)
            local_axes = [wall.matrix_world.to_3x3() @ Vector(axis) for axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]]
            
            # 找到最接近x轴的局部轴
            world_x_axis = Vector((1, 0, 0))
            closest_to_x = max(local_axes, key=lambda axis: abs(axis.dot(world_x_axis)))
            
            # 计算与x轴正方向和负方向的角度
            angle_to_positive_x = closest_to_x.angle(world_x_axis)
            angle_to_negative_x = closest_to_x.angle(-world_x_axis)
            
            # 选择最小的旋转角度
            if angle_to_positive_x < angle_to_negative_x:
                target_x_axis = world_x_axis
                x_angle = angle_to_positive_x
            else:
                target_x_axis = -world_x_axis
                x_angle = angle_to_negative_x
            
            # 计算旋转轴
            x_rotation_axis = closest_to_x.cross(target_x_axis)
            if x_rotation_axis.length > 0:
                x_rotation_axis.normalize()
                x_rotation_matrix = Matrix.Rotation(x_angle, 4, x_rotation_axis)
                wall.matrix_world = Matrix.Translation(geom_center) @ x_rotation_matrix @ Matrix.Translation(-geom_center) @ wall.matrix_world
            
            bpy.context.view_layer.objects.active = wall
            bpy.context.view_layer.update()
    bpy.context.view_layer.update()


def align_wall_to_world_axes(wall_x, wall_y, ground):
    # 获取 wall_x 的局部 Z 轴向量
    local_z_x = wall_x.matrix_world.to_3x3() @ mathutils.Vector((0, 0, 1))
    
    # 计算 wall_x 的 Z 轴与世界 Y 轴负方向的夹角
    target_vector = mathutils.Vector((0, -1, 0))
    angle_to_y_neg = local_z_x.angle(target_vector)
    
    # 确定旋转方向
    cross_prod_x = local_z_x.cross(target_vector)
    if cross_prod_x.z < 0:
        angle_to_y_neg = -angle_to_y_neg
    
    # 计算旋转矩阵
    rotation_matrix = mathutils.Matrix.Rotation(angle_to_y_neg, 4, 'Z')
    
    # 将所有对象绕 ground 的原点旋转
    for obj in bpy.context.scene.objects:
        obj_matrix = obj.matrix_world
        obj_location = obj.location - ground.location
        obj.matrix_world = rotation_matrix @ obj_matrix
        obj.location = rotation_matrix @ obj_location + ground.location
    
    # 使 wall_y 的 Z 轴与 wall_x 的 Z 轴垂直
    local_z_x = wall_x.matrix_world.to_3x3() @ Vector((0, 0, 1))
    local_z_y = wall_y.matrix_world.to_3x3() @ Vector((0, 0, 1))
    
    # 计算 wall_y 的 Z 轴与 wall_x 的 Z 轴的夹角
    angle_to_perpendicular = local_z_y.angle(local_z_x) - (math.pi / 2)
    
    # 确定旋转方向
    cross_prod_y = local_z_y.cross(local_z_x)
    if cross_prod_y.z > 0:
        angle_to_perpendicular = -angle_to_perpendicular
    
    # 将 wall_y 绕自身原点绕世界 Z 轴旋转
    bpy.context.view_layer.objects.active = wall_y
    bpy.ops.object.select_all(action='DESELECT')
    wall_y.select_set(True)
    bpy.ops.transform.rotate(value=angle_to_perpendicular, orient_axis='Z', orient_type='GLOBAL')
    bpy.context.view_layer.update()

'''
仿真功能函数
'''
def add_rigid_body(obj,dynamic=True):
    # 添加刚体物理
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add()
    # 设置刚体类型
    if dynamic:
        obj.rigid_body.type = 'ACTIVE'  # 静态物体

    else:
        obj.rigid_body.type = 'PASSIVE'  # 动态物体
    
    # 配置刚体属性
    obj.rigid_body.mass = 10  # 质量
    obj.rigid_body.friction = 10  # 摩擦力
    #弹性设置
    # 设置弹性为 0
    obj.rigid_body.restitution = 0.0
     # 设置线性和角度阻尼
    obj.rigid_body.linear_damping = 1  # 线性阻尼
    obj.rigid_body.angular_damping = 0.1  # 角度阻尼
        
    #设置碰撞
    num_faces = len(obj.data.polygons)
    print(f"num_faces: {num_faces}")
    print(obj.name)
    if num_faces>2000:

        # # 获取选中的物体
        # obj = bpy.context.object
        # # 添加 Solidify 修饰器
        # modifier = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
        # # 设置厚度，膨胀方向
        # modifier.thickness = 0.01
        # # 应用修饰器
        # bpy.ops.object.modifier_apply(modifier="Solidify")
        # # 获取选中的物体
        # obj = bpy.context.object

        # # 添加 Decimate 修饰器
        modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
        # 设置为 Planar 模式
        modifier.decimate_type = 'DISSOLVE'

        # 设置角度阈值（例如：15度）
        modifier.angle_limit = 15/180*3.1415926
        # # 设置简化比例（减少 50% 顶点）
        # modifier.ratio = 0.5
        # 应用修饰器
        bpy.ops.object.modifier_apply(modifier="Decimate")
        
        # 设置碰撞形状为 Mesh
        obj.rigid_body.collision_shape = 'MESH'

        # 可选：启用碰撞边距
        obj.rigid_body.use_margin = True
        obj.rigid_body.collision_margin = 0.001  # 碰撞边距值
        obj.rigid_body.use_deform = True  # 启用变形碰撞源
    else:
        obj.rigid_body.collision_shape = 'CONVEX_HULL'
        
        obj.rigid_body.use_margin = True
        obj.rigid_body.collision_margin = 0.001
                
# def process_rotation_against_wall(obj_name, obj_info, wall_name):
#     """处理物体的旋转以对齐墙壁, 这里是强迫模型的正朝向背靠墙面，这个是不合理的"""
#     # 获取墙壁的旋转矩阵并计算法向量
#     obj = bpy.data.objects[obj_name]
#     wall = bpy.data.objects[wall_name]
    
#     if not obj_info.get("natural_pose",False):
#         obj.rotation_euler[0] = 0
#         obj.rotation_euler[1] = 0
        
#     wall_rotation = wall.rotation_euler.to_matrix()
#     normal_vector = wall_rotation @ Vector((0, 0, 1))
#     normal_vector.z = 0  # 投影到XY平面
#     normal_vector.normalize()
    
#     # 计算物体的旋转角度
#     angle = math.atan2(normal_vector.y, normal_vector.x)
#     obj.rotation_euler[2] = angle + math.pi / 2


def cal_scale_refer_bbox(obj_name, scene_camera_name, bbox_size):
    """
    计算物体在 X 和 Y 方向上的缩放因子，使其在相机视图中的投影宽度和高度
    分别与指定的像素宽度和高度对齐。

    参数：
    - obj_name (str): 物体的名称。
    - scene_camera_name (str): 相机的名称。
    - bbox_width (float): 期望的物体投影的宽度（像素）。
    - bbox_height (float): 期望的物体投影的高度（像素）。

    返回值：
    - tuple: (scale_x, scale_y)，物体在 X 和 Y 方向上的缩放因子。
    """
    bbox_width, bbox_height = bbox_size
    # 获取物体和相机对象
    obj = bpy.data.objects[obj_name]
    camera = bpy.data.objects[scene_camera_name]

    scene = bpy.context.scene
    render = scene.render

    # 确保使用指定的相机作为活动相机
    scene.camera = camera

    # 获取渲染分辨率
    resolution_x = render.resolution_x * render.resolution_percentage / 100.0
    resolution_y = render.resolution_y * render.resolution_percentage / 100.0

    # 获取物体包围盒在世界坐标系中的顶点
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]

    # 将包围盒的每个顶点投影到相机视图（归一化设备坐标，NDC）
    coords_ndc = [world_to_camera_view(scene, camera, corner) for corner in bbox_corners]

    # 检查物体是否在相机视野内
    if not coords_ndc:
        print("警告：无法获取物体的投影坐标。")
        return 1, 1

    # 获取包围盒在相机视图中的最小和最大X、Y值
    min_x_ndc = min(c.x for c in coords_ndc)
    max_x_ndc = max(c.x for c in coords_ndc)
    min_y_ndc = min(c.y for c in coords_ndc)
    max_y_ndc = max(c.y for c in coords_ndc)

    # 将NDC坐标转换为像素坐标
    min_x_pixel = min_x_ndc * resolution_x
    max_x_pixel = max_x_ndc * resolution_x
    min_y_pixel = (1 - max_y_ndc) * resolution_y
    max_y_pixel = (1 - min_y_ndc) * resolution_y

    # 计算物体当前投影的宽度和高度（像素）
    current_width = abs(max_x_pixel - min_x_pixel)
    current_height = abs(max_y_pixel - min_y_pixel)

    # 检查当前宽度和高度是否为零，避免除以零
    if current_width == 0 or current_height == 0:
        print("错误：物体的当前投影宽度或高度为零，无法计算缩放因子。")
        return 1, 1

    # 计算在宽度和高度方向上需要的缩放因子
    scale_axis_x = bbox_width / current_width
    scale_axis_z = bbox_height / current_height

    return scale_axis_x, scale_axis_z
    
def estimate_scale_factors(obj_name, model_init_size, obb_size, scaling_strategy, mask_is_truncated, bbox_size=None, scene_camera_name=None):
    '''
    根据模型原始尺寸 model_init_size, obb_size 和缩放策略 scaling_strategy，计算模型最终摆放时在长、宽、高上的缩放参数。
    
    缩放策略 (Scaling Strategy) 命名体系：
    基于"约束维度"的命名，直接反映算法在计算 scale 时锁定了哪些轴，或者是否依赖 Pose。
    
    - ISOTROPIC (等比缩放)
        含义：各向同性。XYZ 三轴缩放比例必须完全一致。
        适用于：球体、艺术品、枪支等不能变形的物体。
        
    - RADIAL (径向约束)
        含义：圆柱状逻辑。X和Y轴（径向）必须保持比例一致（锁定），但Z轴（轴向）可以自由缩放。
        适用于：轮胎、桶、圆桌等圆柱形物体。
        
    - ALIGNED_ANISOTROPIC (对齐的各向异性)
        含义：XYZ 三轴完全独立缩放，且严格按照当前 Pose 的 X 对 X，Y 对 Y。
        适用于：箱子、墙画，或者你非常信任 Pose 估计准确性的情况。
        
    - SORTED_ANISOTROPIC (排序的各向异性)
        含义：XYZ 三轴独立缩放，但不信任 Pose 的方向，而是通过将长宽高排序（长对长、短对短）来计算缩放。
        适用于：长方体家具（墙画、书、显示器等），防止因为 Pose 预测偏了90度导致物体被压扁。
    
    缩放策略详细逻辑:
    
    [小物体情况: max(obb_size_products) <= 0.25]
    - ISOTROPIC: 
        计算当前model在相机视角中的像素height，然后求解scale_axis_z，返回[scale_axis_z, scale_axis_z, scale_axis_z]
    - SORTED_ANISOTROPIC 或 ALIGNED_ANISOTROPIC:
        计算当前model在相机视角中的像素height和width，然后求解scale_axis_x和scale_axis_z，返回[scale_axis_x, min(scale_axis_x, scale_axis_z), scale_axis_z]
    - RADIAL:
        计算当前model在相机视角中的像素height和width，然后求解scale_axis_x和scale_axis_z，返回[scale_axis_x, scale_axis_x, scale_axis_z]
        
    [大物体情况: max(obb_size_products) > 0.25]
    - ISOTROPIC: 
        scale_h = obb_size[2] / model_init_size[2]，返回(scale_h, scale_h, scale_h)
        
    - SORTED_ANISOTROPIC: 
        - 如果 model_init_size 的最长边/最短边比例 <= 5 (如桌子柜子)：
            把 model_init_size 和 obb_size 都 sort 一下，将它们作为对应，计算三条边的 scale。
        - 如果比例 > 5 (薄片，如画和屏幕)：
            把 model_init_size 和 obb_size 都 sort 一下，将它们作为对应，计算最长的两条边的 scale，第三条边按原先模型尺寸比例计算一个适中比例。
            
    - ALIGNED_ANISOTROPIC: 
        - 如果 model_init_size 的最长边/最短边比例 <= 5 (如桌子柜子)：
            model_init_size 和 obb_size 一一对应，直接计算相应的 xyz 的 scale，这种方式考虑到了 pose
        - 如果比例 > 5 (薄片，如画和屏幕)：
            model_init_size 和 obb_size 一一对应，只计算最长的两条边的 scale，第三条边按原先模型尺寸比例计算一个适中比例。防止画或者地毯太厚
            
    - RADIAL: 
        圆柱状的物体（如桶、轮胎和圆桌）；将模型的高按照 obb 的高进行缩放，长宽按照 obb 的尺寸里最大值（长、宽）进行统一缩放。
        
    另外，对于物体mask边缘被图片边缘截断的物体(mask_is_truncated=True)，需要进行更保守的scale计算，这部分处理待完善
    '''
    # Sanitize model_init_size to avoid division by zero
    model_init_size = [max(x, eps) for x in model_init_size]

    SCALE_THRESHOLD = [0.1, 5]
    scale_factors = [1,1,1]
    def apply_threshold(scale_factors, threshold):
        return [max(min(s, threshold[1]), threshold[0]) for s in scale_factors]

    # 计算 obb_size 的乘积
    obb_size_products = [
        obb_size[0] * obb_size[1],
        obb_size[0] * obb_size[2],
        obb_size[1] * obb_size[2]
    ]
    
    # 计算 model_init_size 的乘积
    model_init_size_products = [
        model_init_size[0] * model_init_size[1],
        model_init_size[0] * model_init_size[2],
        model_init_size[1] * model_init_size[2]
    ]

    # 处理小物体的情况
    if max(obb_size_products) <= 0.25:
        scale_axis_x, scale_axis_z = cal_scale_refer_bbox(obj_name, scene_camera_name, bbox_size)
        # 加个异常判断，如果scale_axis_z远大于scale_axis_x，可能说明物体的位姿估计错了，类似偏了90°
        if max(scale_axis_x,scale_axis_z) / min(scale_axis_x,scale_axis_z)> 5:
            return [1,1,1]
            
        if scaling_strategy == 'ISOTROPIC':
            scale_factors = [scale_axis_z, scale_axis_z, scale_axis_z]
        elif scaling_strategy in ['SORTED_ANISOTROPIC', 'ALIGNED_ANISOTROPIC']:
            # scale_factors = [scale_axis_x, (scale_axis_x + scale_axis_z)/2, scale_axis_z]
            scale_factors = [scale_axis_x, min(scale_axis_x, scale_axis_z), scale_axis_z]
        elif scaling_strategy == 'RADIAL':
            scale_factors = [scale_axis_x, scale_axis_x, scale_axis_z]
                
        if max(model_init_size_products) <= 0.15:
            return apply_threshold(scale_factors, [1,5])
        else:
            return apply_threshold(scale_factors, SCALE_THRESHOLD)

    # 处理大物体的情况
    else:
        if scaling_strategy == 'ISOTROPIC':
            scale_h = obb_size[2] / model_init_size[2]
            scale_factors = [scale_h, scale_h, scale_h]
        elif scaling_strategy == 'SORTED_ANISOTROPIC':
            model_ratio = max(model_init_size) / min(model_init_size)
            sorted_obb = sorted(obb_size, reverse=True)
            sorted_model = sorted(model_init_size, reverse=True)
            if model_ratio <= 5:
                scales = [o / m for o, m in zip(sorted_obb, sorted_model)]
            else:
                scales = [sorted_obb[i] / sorted_model[i] for i in range(2)]
                scales.append((scales[0] + scales[1]) / 2)
            
            # 创建从原始尺寸到排序后索引的映射
            size_to_index = {size: i for i, size in enumerate(sorted_model)}
            # 使用映射来创建scale_factors
            scale_factors = [scales[size_to_index[size]] for size in model_init_size]
                
        elif scaling_strategy == 'ALIGNED_ANISOTROPIC':
            model_ratio = max(model_init_size) / min(model_init_size)
            if model_ratio <= 5:
                # 将模型的 3 条边对应缩放到与 obb 尺寸一致
                scale_w = obb_size[0] / model_init_size[0]
                scale_h = obb_size[1] / model_init_size[1]
                scale_l = obb_size[2] / model_init_size[2]
                scale_factors = [scale_w, scale_h, scale_l]
            else:
                sorted_sizes = sorted(enumerate(model_init_size), key=lambda x: x[1], reverse=True)
                longest_two_indices = [idx for idx, _ in sorted_sizes[:2]]
                shortest_index = sorted_sizes[-1][0]
                # 计算最长两条边的缩放比例
                scale_factors = [1.0, 1.0, 1.0]
                for i in longest_two_indices:
                    scale_factors[i] = obb_size[i] / model_init_size[i]
                # 对最短边使用适中的缩放比例
                # scale_factors[shortest_index] = (scale_factors[longest_two_indices[0]] + scale_factors[longest_two_indices[1]]) / 2
                scale_factors[shortest_index] = min(scale_factors[longest_two_indices[0]] , scale_factors[longest_two_indices[1]])
                    
        elif scaling_strategy == 'RADIAL':
            scale_h = obb_size[2] / model_init_size[2]
            max_wl_target = max(obb_size[0], obb_size[1])
            max_wl_model = max(model_init_size[0], model_init_size[1])
            scale_wl = max_wl_target / max_wl_model
            scale_factors = [scale_wl, scale_wl, scale_h]
        return apply_threshold(scale_factors, SCALE_THRESHOLD)

def format_list(lst):
    return [f"{x:.3f}" for x in lst]

def estimate_scale_factors_for_object(obj_name, pcd_obb_size, pose, retrieved_asset_bbox_size, bbox_size, scene_camera_name, scaling_strategy, mask_is_truncated):
    """
    估计物体的缩放因子。

    参数：
    - obj_name: string，物体名称
    - pcd_obb_size: ndarray，形状为 (3,)，相机坐标系下观察到的包围盒尺寸 [dx, dy, dz]
    - pose: ndarray，形状为 (4, 4)，物体的姿态矩阵（从物体坐标系到世界坐标系的变换）
    - retrieved_asset_bbox_size: ndarray，形状为 (3,)，检索到的资产的包围盒尺寸 [dx, dy, dz]
    - bbox_size: tuple，物体在图像中的像素包围盒尺寸 (width, height)
    - scene_camera_name: string，场景相机名称
    - scaling_strategy: string，缩放策略，可选值：
        - 'ISOTROPIC': 等比缩放，XYZ 三轴缩放比例完全一致
        - 'RADIAL': 径向约束，XY 轴保持比例一致，Z 轴独立
        - 'ALIGNED_ANISOTROPIC': 对齐的各向异性，XYZ 按 Pose 一一对应缩放
        - 'SORTED_ANISOTROPIC': 排序的各向异性，通过长宽高排序匹配缩放
    - mask_is_truncated: Bool，该物体的 mask 是否被图片边界截断 False 或 True

    返回：
    - scale_factor: list，长度为 3，物体在 x, y, z 方向上的缩放因子
    """
    # 提取物体的旋转矩阵
    pose = np.array(pose)
    pcd_obb_size = np.array(pcd_obb_size)
    retrieved_asset_bbox_size = np.array(retrieved_asset_bbox_size)
    rotation_matrix = pose[:3, :3]
    
    # 定义物体局部坐标轴
    local_axes = np.eye(3)
    
    # 计算物体局部坐标轴在世界坐标系中的方向
    world_axes_vectors = rotation_matrix @ local_axes
    abs_vectors = np.abs(world_axes_vectors)
    
    # 找出物体在世界坐标系中主要对齐的轴（使用贪婪算法确保结果是排列）
    # 简单的 argmax 可能产生重复值（如[0,0,2]），导致某些轴的 scale 为 0
    alignment = np.zeros(3, dtype=int)
    used_cols = set()
    # 按每行最大值的大小排序，优先分配最确定的轴
    row_max_vals = np.max(abs_vectors, axis=1)
    row_order = np.argsort(row_max_vals)[::-1]  # 从最大到最小
    
    for row in row_order:
        # 获取该行的列排序（从大到小）
        col_order = np.argsort(abs_vectors[row])[::-1]
        # 找到第一个未被使用的列
        for col in col_order:
            if col not in used_cols:
                alignment[row] = col
                used_cols.add(col)
                break
    
    # ========== 轴对齐可靠性检测 ==========
    # 当旋转接近45度时，投影值接近，argmax结果不稳定
    # 阈值说明：cos(30°)≈0.866, cos(45°)≈0.707
    # 如果最大投影值与次大投影值的差小于阈值，说明该轴对齐不可靠
    ALIGNMENT_THRESHOLD = 0.15  # 约对应于主轴偏离超过约40度时触发
    
    xy_alignment_reliable = True
    for i in range(2):  # 只检查X和Y轴（行0和行1），Z轴通常可靠
        sorted_row = np.sort(abs_vectors[i])[::-1]
        if sorted_row[0] - sorted_row[1] < ALIGNMENT_THRESHOLD:
            xy_alignment_reliable = False
            break
    
    # 如果XY轴对齐不可靠，且使用的是 ALIGNED_ANISOTROPIC 策略，则回退到尺寸排序匹配
    # 其他策略（ISOTROPIC/RADIAL/SORTED_ANISOTROPIC）不依赖轴对齐，无需回退
    if not xy_alignment_reliable and scaling_strategy == 'ALIGNED_ANISOTROPIC':
        print(f'[AxisAlignment] {obj_name}: XY轴对齐不可靠，回退到尺寸排序匹配')
        
        # 对于XY轴：按尺寸大小匹配（长对长，短对短）
        # Z轴保持原有的pose对齐
        pcd_xy = pcd_obb_size[:2]
        asset_xy = retrieved_asset_bbox_size[:2]
        
        # 获取XY方向上从大到小的索引
        pcd_xy_order = np.argsort(pcd_xy)[::-1]      # pcd中 [大的索引, 小的索引]
        asset_xy_order = np.argsort(asset_xy)[::-1]  # asset中 [大的索引, 小的索引]
        
        # 创建XY的映射：pcd的第i大 对应 asset的第i大
        xy_mapping = np.zeros(2, dtype=int)
        for rank in range(2):
            xy_mapping[pcd_xy_order[rank]] = asset_xy_order[rank]
        
        # 组合映射：XY用排序匹配，Z用原有的pose对齐
        alignment = np.array([xy_mapping[0], xy_mapping[1], alignment[2]])
    
    # 根据对齐情况重新排列retrieved_asset_bbox_size
    reordered_retrieved_asset_bbox_size = retrieved_asset_bbox_size[alignment]
    
    # 估计缩放因子
    reordered_scale_factor = estimate_scale_factors(
        obj_name, 
        reordered_retrieved_asset_bbox_size,
        pcd_obb_size,
        scaling_strategy,
        mask_is_truncated,
        bbox_size,
        scene_camera_name
    )
    
    # 将缩放因子映射回原始坐标系
    original_scale_factor = np.zeros(3)
    for i, axis in enumerate(alignment):
        original_scale_factor[axis] = reordered_scale_factor[i]
    original_scale_factor=original_scale_factor.tolist()
    
    print('\nobj_name:', obj_name)
    print('retrieved_asset_bbox_size:', format_list(retrieved_asset_bbox_size), 
        'pcd_obb_size:', format_list(pcd_obb_size), 
        'reordered_retrieved_asset_bbox_size:', format_list(reordered_retrieved_asset_bbox_size), 
        'reordered_scale_factor:', format_list(reordered_scale_factor),
        'original_scale_factor:', format_list(original_scale_factor))
    print('scaling_strategy:', scaling_strategy, 'xy_alignment_reliable:', xy_alignment_reliable, '\n')

    return original_scale_factor

def simplify_placement(obj_placement_info):
    base_level_pattern = r'^(wall|floor|ceiling|carpet|rug)_\d+'
    first_level_obj_list = []
    second_level_obj_list = []

    # 获取一级摆放物体列表
    for obj_name, obj_info in obj_placement_info['obj_info'].items():
        if (
            not isinstance(obj_info, dict)
            or 'scene_camera' in obj_name.lower()
        ):
            continue
        parent = obj_info.get('supported')
        if parent and re.match(base_level_pattern, parent) and re.match(r'^(carpet|rug)_\d+', obj_name):
            obj_info['supported'] = 'floor_0'  # 地毯不作为一级摆放物体
        elif parent and not re.match(r'^(wall|floor|ceiling)_\d+', obj_name):
            first_level_obj_list.append(obj_name)

    # 获取二级摆放物体列表
    for obj_name, obj_info in obj_placement_info['obj_info'].items():
        if (
            not isinstance(obj_info, dict)
            or 'scene_camera' in obj_name.lower()
        ):
            continue
        parent = obj_info.get('supported')
        if parent in first_level_obj_list:
            second_level_obj_list.append(obj_name)

    # 合并三级及以上摆放物体的支持关系
    for obj_name, obj_info in obj_placement_info['obj_info'].items():
        if (
            not isinstance(obj_info, dict)
            or 'scene_camera' in obj_name.lower()
        ):
            continue
        parent = obj_info.get('supported')
        if parent in second_level_obj_list:
            obj_info['supported'] = (
                obj_placement_info['obj_info'][parent].get('supported')
            )

    return obj_placement_info

# ==================== 辅助函数 ====================
def create_cuboid(name, dimensions, color=(0.5, 0.5, 0.5, 1)):
    bpy.ops.mesh.primitive_cube_add(size=1)
    cuboid = bpy.context.active_object
    cuboid.name = name
    cuboid.scale = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    material = bpy.data.materials.new(name=f"{name}_material")
    material.use_nodes = True
    node_tree = material.node_tree

    node_tree.nodes.clear()

    principled_bsdf = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    material_output = node_tree.nodes.new(type='ShaderNodeOutputMaterial')

    principled_bsdf.inputs['Base Color'].default_value = color

    node_tree.links.new(principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])

    if cuboid.data.materials:
        cuboid.data.materials[0] = material
    else:
        cuboid.data.materials.append(material)

    return cuboid

def process_circular_dependencies(tree_sons, obj_info):
    def dfs(node, path):
        if node in path:
            # 检测到循环依赖
            cycle = path[path.index(node):]
            print(f"警告：检测到循环依赖: {' -> '.join(cycle + [node])}")
            return cycle
        
        path.append(node)
        for child in tree_sons.get(node, [])[:]:  # 创建一个副本以便我们可以在迭代时修改
            cycle = dfs(child, path)
            if cycle:
                if node in cycle:
                    # 当前节点在循环中，移除其父子关系
                    parent = obj_info[node].get('supported')
                    if parent:
                        obj_info[node]['supported'] = None
                        print(f"移除 {node} 的父对象 {parent}")
                    # 从tree_sons中移除这个子对象
                    tree_sons[node].remove(child)
                    print(f"从tree_sons中移除 {node} 的子对象 {child}")
                    return cycle[1:] if cycle[0] == node else cycle
                else:
                    # 当前节点不在循环中，继续向上传播
                    return cycle
        path.pop()
        return None

    for root in list(tree_sons.keys()):  # 使用keys的列表，因为我们可能会在迭代过程中修改tree_sons
        dfs(root, [])

    return tree_sons, obj_info

def get_bbox_info(obj):
    """获取物体的包围盒信息（世界坐标系）"""
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    min_corner = Vector((min(v.x for v in bbox_corners), 
                        min(v.y for v in bbox_corners), 
                        min(v.z for v in bbox_corners)))
    max_corner = Vector((max(v.x for v in bbox_corners), 
                        max(v.y for v in bbox_corners), 
                        max(v.z for v in bbox_corners)))
    
    length = max_corner - min_corner
    
    return {
        "min": min_corner,
        "max": max_corner,
        "length": length
    }

def check_bbox_overlap_fast(bbox1, bbox2):
    """快速检查两个bbox是否可能重叠（复用init_overlap的逻辑）"""
    # 检查 z 轴方向是否重叠
    if bbox1["min"][2] >= bbox2["max"][2] - eps or bbox2["min"][2] >= bbox1["max"][2] - eps:
        return False
    
    # 检查 x 轴方向
    if bbox1["min"][0] >= bbox2["max"][0] - eps or bbox2["min"][0] >= bbox1["max"][0] - eps:
        return False
    
    # 检查 y 轴方向
    if bbox1["min"][1] >= bbox2["max"][1] - eps or bbox2["min"][1] >= bbox1["max"][1] - eps:
        return False
    
    return True

def check_mesh_overlap_bvh(obj1, obj2):
    """使用BVH树精确检测两个物体的网格是否重叠"""
    import bmesh
    from mathutils.bvhtree import BVHTree
    
    # 确保物体有网格数据
    if obj1.type != 'MESH' or obj2.type != 'MESH':
        return False
    
    # 创建 bmesh 并应用世界变换
    bm1 = bmesh.new()
    bm1.from_mesh(obj1.data)
    bm1.transform(obj1.matrix_world)
    
    bm2 = bmesh.new()
    bm2.from_mesh(obj2.data)
    bm2.transform(obj2.matrix_world)
    
    try:
        # 创建 BVH 树
        tree1 = BVHTree.FromBMesh(bm1)
        tree2 = BVHTree.FromBMesh(bm2)
        
        # 检测重叠
        overlap = tree1.overlap(tree2)
        
        return len(overlap) > 0
    finally:
        # 清理
        bm1.free()
        bm2.free()

def layout(
    obj_placement_info_json_path,
    placeable_area_info_folder,
    base_fbx_path,
    fbx_csv_path,
    output_folder,
    precomputed_voxel_dir=None,
    debug=False,
    use_layoutvlm=False,
    layoutvlm_stage="reproject",
):
    # if os.path.exists(obj_placement_info_json_path.replace('.json','_s4.json')):
    #     print(f'{obj_placement_info_json_path} 的s4阶段已经完成, 跳过', flush=True)
    #     return
    
    # 设置双重输出
    os.makedirs(output_folder, exist_ok=True)
    scene_name = os.path.splitext(os.path.basename(obj_placement_info_json_path))[0]
    if scene_name.endswith('_placement_info'):
        scene_name = scene_name[:-len('_placement_info')]
    inference_log_path = os.path.join(output_folder, 'inference_log_s4.txt')
    sys.stdout = Logger(inference_log_path)
    
    blender_manager = BlenderManager()
    # 删除所有对象
    blender_manager.clear_scene()
    
    with open(obj_placement_info_json_path, 'r') as f:
        obj_placement_info = json.load(f)
    
    # 资产可摆放区域信息
    asset_placeable_area_json_path_dict = {}
    for file_name in os.listdir(placeable_area_info_folder):
        if file_name.endswith('json'):
            asset_name = file_name.split('.')[0]
            file_abs_path = os.path.join(placeable_area_info_folder, file_name)
            asset_placeable_area_json_path_dict[asset_name]=file_abs_path
    
    # 用于计算缩放 - 读取缩放策略 (Scaling Strategy)
    # 新命名体系：ISOTROPIC, RADIAL, ALIGNED_ANISOTROPIC, SORTED_ANISOTROPIC
    df = pd.read_csv(fbx_csv_path, skiprows=0)
    model_name_en_list = df['name_en'].tolist()
    scaling_strategy_list = df['scaling_strategy'].tolist()
    fbx_scaling_strategy = {
        str(model_name_en): str(scaling_strategy) if scaling_strategy and str(scaling_strategy) != 'nan' else None
        for model_name_en, scaling_strategy in zip(model_name_en_list, scaling_strategy_list)
    }

    # 读取 alignToWallNormal 属性
    align_to_wall_normal_list = df['alignToWallNormal'].tolist() if 'alignToWallNormal' in df.columns else [0] * len(df)
    fbx_align_to_wall_normal = {
        str(model_name_en): int(align_val) if align_val and str(align_val) != 'nan' else 0
        for model_name_en, align_val in zip(model_name_en_list, align_to_wall_normal_list)
    }
    
    # 设置相机
    scene_camera_name = "scene_camera"
    scene_camera = blender_manager.setup_camera(scene_camera_name)
    scene_camera.location = (0, 0, 0)

    resolution_x, resolution_y  = [1024, 1024]
    # resolution_x, resolution_y  = [1440, 1080]
    bpy.context.scene.render.resolution_x = resolution_x
    bpy.context.scene.render.resolution_y = resolution_y
        
    # 导入地面并获取其变换矩阵
    ground_name = obj_placement_info['reference_obj']
    ground = create_cuboid(ground_name, [10, 10, 0.04])
    
    # 获取地面相对于相机的变换矩阵
    ground_matrix = Matrix(obj_placement_info['obj_info'][ground_name]['pose_matrix_for_blender'])

    # 计算地面变换矩阵的逆矩阵
    ground_matrix_inv = ground_matrix.inverted()

    # 将地面设置为世界坐标系
    ground.matrix_world = Matrix.Identity(4)

    # 导入墙壁并应用变换
    wall_name_list = []
    for name in obj_placement_info['obj_info'].keys():
        if re.match(r'^(wall)_\d+$', name):
            wall_name_list.append(name)
    
    for wall_id in wall_name_list:
        wall = create_cuboid(wall_id, [10, 10, 0.04])  # 使用与地面相同的尺寸，您可以根据需要调整
        wall_matrix = Matrix(obj_placement_info['obj_info'][wall_id]['pose_matrix_for_blender'])
        # 将墙壁的变换矩阵转换为相对于地面的坐标系
        wall.matrix_world = ground_matrix_inv @ wall_matrix
        
    # 设置相机的变换
    camera_matrix = Matrix.Identity(4)  # 相机的初始变换矩阵
    scene_camera.matrix_world = ground_matrix_inv @ camera_matrix

    # 旋转相机180度
    rotation_angle_rad = math.radians(90)
    scene_camera.rotation_euler[0] += rotation_angle_rad

    # 更新场景
    bpy.context.view_layer.update()
    
    # 对每个墙体应用对齐
    missing_wall_ids = [
        wall_id for wall_id in wall_name_list if wall_id not in bpy.data.objects
    ]
    if missing_wall_ids:
        print(
            "[S4] Missing structural walls during axis alignment: "
            f"{missing_wall_ids}; preserving dependent object poses."
        )
    wall_objects = [
        bpy.data.objects[wall_id]
        for wall_id in wall_name_list
        if wall_id in bpy.data.objects
    ]
    align_wall_to_axes(wall_objects, ground)

    for obj_name, obj_info in obj_placement_info['obj_info'].items():
        # S4 output JSONs retain the camera record so they can be reused as a
        # refinement input.  Cameras are not layout objects and intentionally
        # have no support relation.
        if 'scene_camera' in obj_name.lower():
            continue
        # 初始化物体和support物体的摆放关系  默认是on  下面会有on和inside的判断
        obj_info['SpatialRel'] = 'on' if obj_info.get('supported') else None
        
        if re.match(r'^(wall|floor)_\d+$', obj_name):
            continue
        
        retrieved_asset = obj_info["retrieved_asset"]

        # 注入 alignToWallNormal 属性
        if retrieved_asset in fbx_align_to_wall_normal:
            obj_info['alignToWallNormal'] = fbx_align_to_wall_normal[retrieved_asset]

        fbx_path = os.path.normpath(os.path.join(base_fbx_path, f'{retrieved_asset}.fbx'))
        
        # 导入FBX文件
        obj = blender_manager.import_fbx(fbx_path)
        blender_manager.ensure_object_visible(obj)
        obj.name = obj_name

        # 设置初始位姿
        pose = Matrix(obj_info["pose_matrix_for_blender"])
        obj.matrix_world = ground_matrix_inv @ pose
        # if obj_info.get("againstWall", None):
        #     process_rotation_against_wall(obj_name, obj_info, obj_info["againstWall"])
        # else:
        #     obj_info["againstWall"]=None
        if obj_info.get("againstWall", None) is None:
            obj_info["againstWall"]=None
            obj_info["isAgainstWall"]=False
        
        # 天花板上的物体取消勾选阴影投射
        parent=obj_info.get('supported')
        if re.match(r'^ceiling_\d+', parent):
            obj.visible_shadow = False
            print(f"Disabled shadow casting for ceiling object: {obj_name}")
            
        bpy.context.view_layer.objects.active = obj
        # ⚡ 性能优化：移除循环内的视图更新，改为批量更新
        # bpy.context.view_layer.update()
    
    # ⚡ 性能优化：批量更新场景（替代循环内的多次更新）
    bpy.context.view_layer.update()

    # Apply retrieved textures to Wall, Floor, Ceiling
    print("\nApplying retrieved textures...")
    s3_dir = os.path.dirname(obj_placement_info_json_path)
    result_root = os.path.dirname(s3_dir)
    texture_json_path = os.path.join(result_root, 'S2_3d_retrieval_results', 'texture_retrieval_results.json')
    
    if os.path.exists(texture_json_path):
        print(f"Loading texture retrieval results from {texture_json_path}")
        try:
            with open(texture_json_path, 'r') as f:
                texture_results = json.load(f)
                
            for obj_name, texture_path in texture_results.items():
                obj = bpy.data.objects.get(obj_name)
                if obj:
                    print(f"Applying texture to {obj_name}: {texture_path}")
                    apply_texture_from_path(obj, texture_path)
                else:
                    print(f"Warning: Object {obj_name} not found for texture application.")
        except Exception as e:
             print(f"Error applying textures: {e}")
    else:
        print(f"Texture retrieval results not found at {texture_json_path}")
    
    bpy.context.view_layer.update()

    # 取消所有三级及以上的摆放关系, 先得到所有一级摆放物体list, 再将所有三级及以上的摆放物体向二级合并
    obj_placement_info = simplify_placement(obj_placement_info)
    
    # 创建输出字典
    output_data_s1 = {
        "reference_obj": obj_placement_info['reference_obj'],
        "scene_camrea_name": scene_camera_name,
        "obj_info": obj_placement_info['obj_info']
    }
    output_data_s1['obj_info'][scene_camera_name] = {}
    
    # 遍历场景中的所有网格对象, 更新dict中的位姿信息
    for obj in bpy.data.objects:
        world_matrix = obj.matrix_world
        matrix_list = [list(row) for row in world_matrix]
        output_data_s1["obj_info"][obj.name]['pose_matrix_for_blender'] = matrix_list
            
    # 将字典转换为 JSON 格式并保存
    output_path = os.path.join(output_folder, f'{scene_name}_placement_info_s1.json')
    with open(output_path, 'w') as f:
        json.dump(output_data_s1, f, indent=2)
    
    # Render-only reconstruction skips the disposable S1 preview. The final
    # certified render below uses the same source camera at 256 samples.
    if not os.environ.get("IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT"):
        bpy.context.scene.camera = bpy.data.objects[scene_camera_name]
        output_path = os.path.join(output_folder, f'{scene_name}_render_s1.png')
        blender_manager.render_scene(output_path, resolution_x, resolution_y)
        print(f"S3 render_s1 poses saved to: {output_path}", flush=True)
    
    # s2
    # 处理位姿, 对齐, scale
    tree_sons = {}
    processed_matrix = {}
    obj_list = {}
    output_data_s2 = copy.deepcopy(output_data_s1)

    # 初始化对象列表和尺寸，处理父物体不为墙的那些物体，立正
    for obj_name, obj_info in output_data_s2['obj_info'].items():
        obj = bpy.data.objects[obj_name]
        
        if obj_name == scene_camera_name or obj_info.get("SpatialRel") == "inside":
            continue
        
        obj_list[obj_name] = obj
        processed_matrix[obj_name] = obj.matrix_world
        
        ### 只处理父物体不为墙的那些物体
        parent = obj_info.get('supported')
        if parent and not re.match(r'^wall_\d+', parent) and obj_info.get("SpatialRel") == "on":
            tree_sons.setdefault(parent, []).append(obj_name)
        if not re.match(r'^wall_\d+', obj.name):  # 所有物体都立正
            if re.match(r'^(carpet|rug)_\d+', obj.name):
                # 现有位姿估计算法对大平面薄物体预测效果差，比如墙面、地毯等；
                # 地毯特殊处理：确保 Z 轴正方向与世界 Z 轴正方向（地面法向）一致
                align_carpet_z_to_world_z_positive(obj)
            else:
                align_closest_axis_to_world_z(obj)
            
    # 检查tree_sons, 处理循环依赖
    tree_sons, output_data_s2["obj_info"] = process_circular_dependencies(tree_sons, output_data_s2["obj_info"])
    
    # 确保父物体是墙的物体具有 againstWall 属性，以便 process_rotation_against_wall_hierarchical 能处理它们
    for obj_name, obj_info in output_data_s2["obj_info"].items():
        parent = obj_info.get('supported')
        if parent and re.match(r'^wall_\d+', parent):
            if not obj_info.get("againstWall"):
                obj_info["againstWall"] = parent

    # 处理靠墙物体的rotation，要依层级顺序旋转：
    # 将所有一级父物体旋转，然后保留其子物体的相对位姿
    # 然后将所有二级父物体旋转，然后保留其子物体的相对位姿
    # 。。。迭代至没有父物体；然后将剩余没有处理的所有物体都进行旋转
    blender_manager.process_rotation_against_wall_hierarchical(output_data_s2["obj_info"], obj_list, tree_sons)
    bpy.context.view_layer.update()

    # 遍历场景中的所有网格对象, 更新dict中的位姿信息
    for obj in bpy.data.objects:
        if obj.name in output_data_s2["obj_info"]:
            world_matrix = obj.matrix_world
            matrix_list = [list(row) for row in world_matrix]
            output_data_s2["obj_info"][obj.name]['pose_matrix_for_blender'] = matrix_list
    
    # 处理缩放
    for obj_name, obj_info in output_data_s2['obj_info'].items():
        if obj_name == scene_camera_name:
            continue
        
        obj = bpy.data.objects[obj_name]
        retrieved_asset = obj_info["retrieved_asset"]
        
        if retrieved_asset:
            mask_is_truncated = obj_info.get("mask_is_truncated", None)
            retrieved_asset_bbox_size = bpy.data.objects[obj_name].dimensions
            pcd_obb_size = obj_info['pcd_obb_size']
            scaling_strategy = fbx_scaling_strategy[retrieved_asset]

            boxes = obj_info['boxes']
            bbox_size = [max(abs(boxes[2]- boxes[0]), 1), max(abs(boxes[3]- boxes[1]), 1)]
            pose_matrix_list = [list(row) for row in obj.matrix_world]
            
            # ========== 相机坐标系夹角检测 & 自动回退到尺寸排序匹配 ==========
            # 作用是 当物体局部X轴在相机坐标系中与相机X/Y轴的夹角接近45度时，S1、S2得到的obb的参考尺寸的顺序可能会因旋转的微小差异导致X和Y轴颠倒
            original_cam_pose = np.array(obj_placement_info['obj_info'][obj_name]['pose_matrix_for_blender'])
            cam_rot = original_cam_pose[:3, :3]
            # 计算物体局部X轴在相机坐标系中与相机X/Y轴的夹角
            local_x_in_cam = cam_rot[:, 0]  # 第一列是局部X轴在相机系的方向
            # 与相机X轴(1,0,0)的夹角
            angle_x_to_camX = np.degrees(np.arccos(np.clip(np.abs(local_x_in_cam[0]), 0, 1)))
            angle_x_to_camY = np.degrees(np.arccos(np.clip(np.abs(local_x_in_cam[1]), 0, 1)))
            
            # 判断是否接近45度（40°~50°范围内认为不稳定）
            xy_angle_unstable = (40 <= angle_x_to_camX <= 50) or (40 <= angle_x_to_camY <= 50)
            
            # 如果相机坐标系下夹角接近45度，直接使用尺寸排序匹配（跳过pose对齐）
            if xy_angle_unstable and scaling_strategy == 'ALIGNED_ANISOTROPIC':
                print(f'[CamAngleFix] {obj_name}: 相机系夹角={angle_x_to_camX:.1f}°接近45°, 使用尺寸排序匹配')
                # 直接按尺寸排序匹配，不经过 estimate_scale_factors_for_object 的 pose 对齐
                pcd_obb_size_arr = np.array(pcd_obb_size)
                asset_bbox_size_arr = np.array(retrieved_asset_bbox_size)
                
                # 按尺寸从大到小排序
                pcd_order = np.argsort(pcd_obb_size_arr)[::-1]
                asset_order = np.argsort(asset_bbox_size_arr)[::-1]
                
                # 计算排序后的 scale（大对大，中对中，小对小）
                sorted_scales = pcd_obb_size_arr[pcd_order] / np.maximum(asset_bbox_size_arr[asset_order], eps)
                
                # 映射回原始坐标系
                scale_factors = np.zeros(3)
                for i in range(3):
                    scale_factors[asset_order[i]] = sorted_scales[i]
                
                # 应用阈值限制
                SCALE_THRESHOLD = [0.1, 5]
                scale_factors = [max(min(s, SCALE_THRESHOLD[1]), SCALE_THRESHOLD[0]) for s in scale_factors]
                print(f'  sorted_scales: {format_list(sorted_scales)}, final_scale: {format_list(scale_factors)}')
            else:
                scale_factors = estimate_scale_factors_for_object(obj_name, pcd_obb_size, pose_matrix_list, retrieved_asset_bbox_size, bbox_size,
                                scene_camera_name, scaling_strategy, mask_is_truncated)
            obj_info['scale'] = scale_factors
            obj.scale = scale_factors
        
        bpy.context.view_layer.objects.active = obj
        # ⚡ 性能优化：移除循环内的视图更新
        # bpy.context.view_layer.update()
    
    # ⚡ 性能优化：批量更新场景（替代缩放循环内的多次更新）
    bpy.context.view_layer.update()
    
    # 根据scene graph的group关系，让同组的物体的scale保持一致
    groups = defaultdict(list)
    for obj_name, obj_info in output_data_s2['obj_info'].items():
        if 'group' in obj_info:
            groups[obj_info['group']].append((obj_name, obj_info['scale']))
    # 对每个组进行处理
    for group, objects in groups.items():
        if len(objects) <= 1:
            continue  # 跳过只有一个物体的组

        # 收集所有scale
        all_scales = [np.array(obj[1]) for obj in objects]

        # 找出最频繁的scale（与其他scale平均距离最小的scale）
        min_avg_distance = float('inf')
        most_frequent_scale = None

        for scale in all_scales:
            avg_distance = np.mean([np.linalg.norm(scale - other_scale) for other_scale in all_scales])
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                most_frequent_scale = scale

        # 将最频繁的scale应用到组内所有物体
        for obj_name, _ in objects:
            output_data_s2['obj_info'][obj_name]['scale'] = most_frequent_scale.tolist()
            obj = bpy.data.objects[obj_name]
            obj.scale = most_frequent_scale.tolist()
            bpy.context.view_layer.objects.active = obj
            # ⚡ 性能优化：移除循环内的视图更新
            # bpy.context.view_layer.update()
    
    # ⚡ 性能优化：批量更新场景（替代组内循环的多次更新）
    bpy.context.view_layer.update()
    
    # ========== 安全检查：修复 scale 有零分量的物体 ==========
    # 某些 FBX 导入后可能有零 scale 分量，会导致后续矩阵计算失败
    MIN_SCALE = 0.001  # 最小 scale 值
    for obj_name, obj_info in output_data_s2['obj_info'].items():
        if obj_name == scene_camera_name:
            continue
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue
        scale = list(obj.scale)
        fixed = False
        for i in range(3):
            if abs(scale[i]) < MIN_SCALE:
                print(f'[Warning] {obj_name}: scale[{i}]={scale[i]} 接近零，修复为 {MIN_SCALE}')
                scale[i] = MIN_SCALE
                fixed = True
        if fixed:
            obj.scale = scale
            if 'scale' in obj_info:
                obj_info['scale'] = scale
    bpy.context.view_layer.update()
        
    # 处理直接面对关系（如果需要）
    blender_manager.process_directly_facing(output_data_s2['obj_info'], fbx_scaling_strategy)
    
    # # 处理wall, 调整位置，以减少与其他物体的重叠, 但是墙的移动可能会很夸张, 后面可能会导致场景与参考图片差异较大
    # for obj_name, obj_info in output_data_s2["obj_info"].items():
    #     if re.match(r"wall_\d+", obj_name):
    #         blender_manager.process_wall(obj_name, obj_list, ground_name)
    # bpy.context.view_layer.update()  
    
    # 遍历场景中的所有网格对象, 更新dict中的位姿信息
    for obj in bpy.data.objects:
        if obj.name in output_data_s2["obj_info"]:
            world_matrix = obj.matrix_world
            matrix_list = [list(row) for row in world_matrix]
            output_data_s2["obj_info"][obj.name]['pose_matrix_for_blender'] = matrix_list
    
    # 只考虑旋转纠正， 不考虑二级物体的靠墙位移纠正
    base_level_pattern = r'^(wall|floor|ceiling|carpet|rug)_\d+'
    for obj_name, obj_info in output_data_s2['obj_info'].items():
        parent = obj_info.get('supported', None)
        againstWall = obj_info.get("againstWall", None)
        if againstWall:
            if parent is None or not re.match(base_level_pattern, parent):
                obj_info["isAgainstWall"] = False
                obj_info["againstWall"] = None
        
    # 处理靠墙物体的translation
    relativePoseManager = RelativePoseManager(obj_list, tree_sons, output_data_s2)
    relativePoseManager.record_all()
    blender_manager.process_translation_against_wall(output_data_s2["obj_info"], obj_list)
    bpy.context.view_layer.update()
    relativePoseManager.restore_all()
    bpy.context.view_layer.update()

    # 遍历场景中的所有网格对象, 更新dict中的位姿信息
    for obj in bpy.data.objects:
        if obj.name in output_data_s2["obj_info"]:
            world_matrix = obj.matrix_world
            matrix_list = [list(row) for row in world_matrix]
            output_data_s2["obj_info"][obj.name]['pose_matrix_for_blender'] = matrix_list
        
    # 处理内部摆放
    items_failed_and_del = []
    for obj_name, obj_info in output_data_s2['obj_info'].items():
        retrieved_asset = obj_info.get('retrieved_asset')
        if retrieved_asset in asset_placeable_area_json_path_dict.keys():
            parent_name = obj_name
            if bpy.data.objects.get(parent_name) is None:
                print(
                    f"[Warning] Skipping placeable subspaces for {parent_name}: "
                    "parent object was not imported or was deleted",
                    flush=True,
                )
                continue
            placeable_area_info = json.load(open(asset_placeable_area_json_path_dict[obj_info['retrieved_asset']], 'r'))
            
            subspaces_info = []
            closest_subspace_mapping = {}
            for subspace in placeable_area_info:
                name = subspace['name']
                transform_matrix = subspace['transform_matrix']
                scale_ratio = subspace['scale_ratio']
                create_subspace(name, parent_name, transform_matrix, scale_ratio)
                subspaces_info.append(subspace)
                closest_subspace_mapping[name] = []
            
            sub_objs_info_list = []
            for name, value in output_data_s2['obj_info'].items():
                supported = value.get('supported')
                if supported == parent_name:
                    SpatialRel, closest_subspace_info = get_closest_subspace(name, parent_name, subspaces_info)
                    value['SpatialRel'] = SpatialRel
                    print(f'检测到 {name} 要 {SpatialRel} 于 {parent_name}')
                    if SpatialRel == "inside":
                        sub_objs_info_list.append((name, closest_subspace_info))
                        closest_subspace_mapping[closest_subspace_info['name']].append(name)

            for sub_objs_info in sub_objs_info_list:
                name, closest_subspace_info = sub_objs_info
                align_obj_to_closest_subspace(name, closest_subspace_info)
                bpy.context.view_layer.update()
            
            # 解决碰撞
            for subspace_name, obj_list in closest_subspace_mapping.items():
                if not obj_list: continue
                subspace_obj = bpy.data.objects[subspace_name]
                objects_in_subspace = [bpy.data.objects[name] for name in obj_list]
                items_failed_and_del.extend(resolve_collisions_in_subspace(objects_in_subspace, subspace_obj))
                bpy.context.view_layer.update()
            
            # ⚡ 性能优化：批量更新场景（替代碰撞循环内的多次更新）
            if closest_subspace_mapping:
                bpy.context.view_layer.update()
                
            # 删除所有子空间对象
            for subspace in subspaces_info:
                bpy.data.objects.remove(bpy.data.objects[subspace['name']], do_unlink=True)
            bpy.context.view_layer.update()

    # 从output_data_s2['obj_info']中删除items_failed_and_del
    for key in items_failed_and_del:
        output_data_s2['obj_info'].pop(key, None)
    print(f'内部摆放阶段删除了{items_failed_and_del}')
 
    obj_list={}
    for obj_name, obj_info in output_data_s2['obj_info'].items():
        obj = bpy.data.objects[obj_name]
        if obj_name == scene_camera_name or obj_info.get("SpatialRel") == "inside":
            continue
        obj_list[obj_name] = obj
    
    # 处理on的z轴空间关系
    blender_manager.process_z(ground_name, obj_list, tree_sons, 0)
    bpy.context.view_layer.update()
    
    # 更新并保存结果
    for instance_id, obj_info in output_data_s2['obj_info'].items():
        obj = bpy.data.objects.get(instance_id)
        if obj and instance_id != ground_name:
            obj_info["pose_matrix_for_blender"] = [list(row) for row in blender_manager.get_matrix_world(obj)]
            obj_info["bbox"] = [list(point) for point in blender_manager.get_world_bound_box(obj)]
            obj_info["length"] = list(obj.dimensions)

    bpy.context.view_layer.update()

    # 保存结果
    output_path = os.path.join(output_folder, f'{scene_name}_placement_info_s2.json')
    with open(output_path, 'w') as f:
        json.dump(output_data_s2, f, indent=2)

    # # 渲染场景
    # bpy.context.scene.camera = bpy.data.objects[scene_camera_name]
    # output_path = os.path.join(output_folder, f'{scene_name}_render_s2.png')
    # blender_manager.render_scene(output_path, resolution_x, resolution_y)
    # print(f"S3 render_s2 poses saved to: {output_path}", flush=True)

    # A certified SceneProof result can mix the incumbent and guarded trial on
    # an object-by-object basis, so neither branch render is the final scene.
    # This opt-in path rebuilds assets from the frozen S3 input, applies the
    # certified matrices, and renders without voxelization or another solve.
    # Its camera is restored from source S3, never from the certified result.
    render_only_placement = os.environ.get(
        "IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT"
    )
    if render_only_placement:
        placement_path = Path(render_only_placement).resolve()
        if not placement_path.is_file():
            raise FileNotFoundError(
                f"SceneProof render-only placement not found: {placement_path}"
            )
        with placement_path.open("r", encoding="utf-8") as handle:
            render_only_data = json.load(handle)
        render_only_info = render_only_data.get("obj_info")
        if not isinstance(render_only_info, dict):
            raise ValueError(
                "SceneProof render-only placement has no obj_info: "
                f"{placement_path}"
            )

        source_camera_info = obj_placement_info.get("obj_info", {}).get(
            scene_camera_name, {}
        )
        source_camera_pose = source_camera_info.get(
            "pose_matrix_for_blender"
        )
        source_camera_array = np.asarray(source_camera_pose, dtype=np.float64)
        if source_camera_array.shape != (4, 4) or not np.isfinite(
            source_camera_array
        ).all():
            raise RuntimeError(
                "SceneProof render-only requires a finite 4x4 source S3 "
                f"camera pose for {scene_camera_name}"
            )

        applied_objects = []
        missing_expected_objects = []
        ignored_nonrenderable_records = []
        reconstructed_ids = set(output_data_s2.get("obj_info", {}))
        for instance_id, info in render_only_info.items():
            if instance_id == scene_camera_name:
                continue
            pose = (
                info.get("pose_matrix_for_blender")
                if isinstance(info, dict)
                else None
            )
            pose_array = np.asarray(pose, dtype=np.float64)
            if pose_array.shape != (4, 4) or not np.isfinite(pose_array).all():
                if instance_id in reconstructed_ids:
                    raise RuntimeError(
                        "SceneProof render-only has an invalid certified pose "
                        f"for reconstructed object: {instance_id}"
                    )
                ignored_nonrenderable_records.append(instance_id)
                continue
            blender_obj = bpy.data.objects.get(instance_id)
            if blender_obj is None:
                if instance_id in reconstructed_ids:
                    missing_expected_objects.append(instance_id)
                else:
                    ignored_nonrenderable_records.append(instance_id)
                continue
            blender_obj.matrix_world = Matrix(pose_array.tolist())
            blender_obj.hide_render = False
            applied_objects.append(instance_id)

        if missing_expected_objects:
            raise RuntimeError(
                "SceneProof render-only lost objects owned by reconstructed S2: "
                + ", ".join(sorted(missing_expected_objects))
            )

        target_ids = set(render_only_info)
        for instance_id in output_data_s2.get("obj_info", {}):
            if instance_id in target_ids or instance_id == scene_camera_name:
                continue
            blender_obj = bpy.data.objects.get(instance_id)
            if blender_obj is not None:
                blender_obj.hide_render = True

        scene_camera = bpy.data.objects.get(scene_camera_name)
        if scene_camera is None:
            raise RuntimeError(
                f"SceneProof render-only camera is missing: {scene_camera_name}"
            )
        scene_camera.matrix_world = Matrix(source_camera_array.tolist())
        bpy.context.scene.camera = scene_camera
        bpy.context.view_layer.update()
        locked_camera_array = np.asarray(
            [list(row) for row in scene_camera.matrix_world],
            dtype=np.float64,
        )
        camera_assignment_delta = float(
            np.max(np.abs(locked_camera_array - source_camera_array))
        )
        camera_float32_tolerance = float(
            8.0
            * np.finfo(np.float32).eps
            * max(1.0, float(np.max(np.abs(source_camera_array))))
        )
        if camera_assignment_delta > camera_float32_tolerance:
            raise RuntimeError(
                "SceneProof render-only camera assignment exceeded float32 "
                "quantization tolerance: "
                f"max_abs_delta={camera_assignment_delta:.8g}, "
                f"tolerance={camera_float32_tolerance:.8g}"
            )

        com_audit_output = os.environ.get(
            "IMAGINARIUM_SCENEPROOF_TRUE_MESH_COM_AUDIT_OUTPUT"
        )
        local_settle_output = os.environ.get(
            "IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_AUDIT_OUTPUT"
        )
        vertical_transaction_output = os.environ.get(
            "IMAGINARIUM_SCENEPROOF_VERTICAL_TRANSACTION_AUDIT_OUTPUT"
        )
        sparse_vertical_output = os.environ.get(
            "IMAGINARIUM_SCENEPROOF_SPARSE_VERTICAL_CONTACT_AUDIT_OUTPUT"
        )
        render_identity_output = os.environ.get(
            "IMAGINARIUM_SCENEPROOF_RENDER_IDENTITY_AUDIT_OUTPUT"
        )
        structural_geometry_dump = os.environ.get(
            "IMAGINARIUM_SCENEPROOF_STRUCTURAL_GEOMETRY_DUMP"
        )
        if sparse_vertical_output:
            rollback_document = None
            rollback_placement = os.environ.get(
                "IMAGINARIUM_SCENEPROOF_SPARSE_ROLLBACK_PLACEMENT"
            )
            if rollback_placement:
                rollback_path = Path(rollback_placement).resolve()
                if rollback_path.is_file():
                    rollback_document = json.loads(
                        rollback_path.read_text(encoding="utf-8")
                    )
            sparse_audit = apply_sceneproof_sparse_vertical_contact(
                render_only_data,
                rollback_document=rollback_document,
                contact_tolerance_m=float(
                    os.environ.get(
                        "IMAGINARIUM_SCENEPROOF_SPARSE_CONTACT_TOLERANCE_M", "0.02"
                    )
                ),
                maximum_shift_m=float(
                    os.environ.get(
                        "IMAGINARIUM_SCENEPROOF_SPARSE_MAXIMUM_SHIFT_M", "0.5"
                    )
                ),
                minimum_hit_fraction=float(
                    os.environ.get(
                        "IMAGINARIUM_SCENEPROOF_SPARSE_MINIMUM_HIT_FRACTION", "0.10"
                    )
                ),
                maximum_tangent_shift_m=float(
                    os.environ.get(
                        "IMAGINARIUM_SCENEPROOF_SPARSE_MAXIMUM_TANGENT_SHIFT_M",
                        "0.15",
                    )
                ),
                maximum_program_tangent_shift_m=float(
                    os.environ.get(
                        "IMAGINARIUM_SCENEPROOF_SPARSE_MAXIMUM_PROGRAM_TANGENT_SHIFT_M",
                        "0.50",
                    )
                ),
            )
            if os.environ.get(
                "IMAGINARIUM_SCENEPROOF_VISUAL_SAFE_SALVAGE", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}:
                visual_safe = apply_sceneproof_visual_safe_salvage(
                    render_only_data,
                    sparse_audit,
                    maximum_floor_shift_m=float(
                        os.environ.get(
                            "IMAGINARIUM_SCENEPROOF_VISUAL_SAFE_MAX_FLOOR_SHIFT_M",
                            "0.60",
                        )
                    ),
                    maximum_suppressed_objects=int(
                        os.environ.get(
                            "IMAGINARIUM_SCENEPROOF_VISUAL_SAFE_MAX_SUPPRESSED",
                            "4",
                        )
                    ),
                )
                sparse_audit["visual_safe_salvage"] = visual_safe
                sparse_audit["status"] = "visual_salvaged"
                sparse_audit["passed"] = False
                sparse_audit["eligible_for_paper_metrics"] = False
                sparse_audit["unresolved_object_ids"] = visual_safe[
                    "unresolved_object_ids"
                ]
            sparse_path = Path(sparse_vertical_output).resolve()
            sparse_path.parent.mkdir(parents=True, exist_ok=True)
            sparse_audit["placement"] = str(placement_path)
            candidate_placement = os.environ.get(
                "IMAGINARIUM_SCENEPROOF_SPARSE_VERTICAL_CONTACT_PLACEMENT_OUTPUT"
            )
            if candidate_placement:
                candidate_path = Path(candidate_placement).resolve()
                candidate_path.parent.mkdir(parents=True, exist_ok=True)
                candidate_path.write_text(
                    json.dumps(render_only_data, indent=2), encoding="utf-8"
                )
                sparse_audit["candidate_placement"] = str(candidate_path)
            sparse_path.write_text(
                json.dumps(sparse_audit, indent=2), encoding="utf-8"
            )
            print(
                "[SceneProof] Support-contact routing: "
                f"status={sparse_audit['status']}, "
                f"repaired={len(sparse_audit['repaired_object_ids'])}, "
                f"held={len(sparse_audit['held_object_ids'])}, "
                f"unresolved={len(sparse_audit['unresolved_object_ids'])}, "
                f"output={sparse_path}",
                flush=True,
            )
        if structural_geometry_dump:
            # The ground slab is built procedurally and then excluded from
            # geometry serialization, so no artefact records its extent and every
            # downstream measurement that needs the floor polygon degrades to a
            # point.  Read the extent back from the constructed scene: this
            # measures the pipeline's own geometry rather than assuming a
            # construction constant, and it is the evidence the evaluator's
            # --structural-geometry-sidecar consumes.
            dumped = {}
            for instance_id in list(render_only_data.get("obj_info", {})):
                blender_object = bpy.data.objects.get(instance_id)
                if blender_object is None:
                    continue
                try:
                    world_corners = [
                        list(blender_object.matrix_world @ Vector(corner))
                        for corner in blender_object.bound_box
                    ]
                except Exception as error:
                    print(
                        "[SceneProof] Structural geometry dump: skipped "
                        f"{instance_id} ({type(error).__name__}: {error})",
                        flush=True,
                    )
                    continue
                dumped[instance_id] = {
                    "bbox": world_corners,
                    "length": list(blender_object.dimensions),
                    "measured_from": "constructed_blender_scene",
                }
            dump_path = Path(structural_geometry_dump).resolve()
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_path.write_text(
                json.dumps(
                    {
                        "schema_version": "sceneproof_structural_geometry_v1",
                        "placement": str(placement_path),
                        "obj_info": dumped,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(
                "[SceneProof] Structural geometry dump: "
                f"{len(dumped)} objects -> {dump_path}",
                flush=True,
            )
            if os.environ.get(
                "IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}:
                return
        if render_identity_output:
            identity_path = Path(render_identity_output).resolve()
            color_path = Path(
                os.environ.get(
                    "IMAGINARIUM_SCENEPROOF_RENDER_IDENTITY_COLOR_OUTPUT",
                    str(identity_path.with_name(identity_path.stem + "_color.png")),
                )
            ).resolve()
            annotated_path = Path(
                os.environ.get(
                    "IMAGINARIUM_SCENEPROOF_RENDER_IDENTITY_ANNOTATED_OUTPUT",
                    str(identity_path.with_name(identity_path.stem + "_annotated.png")),
                )
            ).resolve()
            identity_audit = audit_sceneproof_mesh_visibility(
                obj_placement_info_json_path,
                sorted(applied_objects),
                scene_camera,
                {"component_audits": []},
                resolution=int(
                    os.environ.get(
                        "IMAGINARIUM_SCENEPROOF_RENDER_IDENTITY_RESOLUTION",
                        "512",
                    )
                ),
                color_id_output_path=color_path,
                annotated_color_id_output_path=annotated_path,
            )
            identity_audit["placement"] = str(placement_path)
            identity_audit["color_id_image"] = str(color_path)
            identity_audit["annotated_color_id_image"] = str(annotated_path)
            identity_path.parent.mkdir(parents=True, exist_ok=True)
            identity_path.write_text(
                json.dumps(identity_audit, indent=2), encoding="utf-8"
            )
            print(
                "[SceneProof] Render identity audit complete: "
                f"objects={len(identity_audit.get('objects', {}))}, "
                f"output={identity_path}, annotated={annotated_path}",
                flush=True,
            )
        if local_settle_output:
            duration_seconds = float(
                os.environ.get(
                    "IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_DURATION_SECONDS",
                    "1.0",
                )
            )
            single_object_id = os.environ.get(
                "IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_OBJECT_ID", ""
            ).strip()
            batch_object_ids = os.environ.get(
                "IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_OBJECT_IDS", ""
            ).strip()

            # Build the list of objects to settle.  Batch mode reuses the
            # already-loaded scene (one Blender invocation per scene instead
            # of one per object) and the existing per-object auditor restores
            # pose_before in its ``finally`` block, so each object starts from
            # the incumbent layout.
            candidate_ids: list[str] = []
            if batch_object_ids:
                candidate_ids = [
                    candidate_id.strip()
                    for candidate_id in batch_object_ids.split(",")
                    if candidate_id.strip()
                ]
            elif single_object_id:
                candidate_ids = [single_object_id]

            if candidate_ids:
                output_root = Path(local_settle_output).resolve()
                is_single = (
                    len(candidate_ids) == 1
                    and not batch_object_ids
                )
                if is_single:
                    output_root.parent.mkdir(parents=True, exist_ok=True)
                else:
                    output_root.mkdir(parents=True, exist_ok=True)

                if not is_single and len(candidate_ids) > 1:
                    # Bulk mode: one simulation, all targets drop together.
                    bulk_results = audit_sceneproof_local_gravity_settle_bulk(
                        render_only_data,
                        candidate_ids,
                        duration_seconds=duration_seconds,
                    )
                    for candidate_id, audit in bulk_results.items():
                        audit["placement"] = str(placement_path)
                        path = output_root / f"{candidate_id}.json"
                        path.write_text(
                            json.dumps(audit, indent=2), encoding="utf-8"
                        )
                        st = audit.get("status", "?")
                        tr = audit.get("translation_delta_m", "?")
                        rdeg = audit.get("rotation_delta_deg", "?")
                        nc = len(audit.get("new_collision_object_ids", []))
                        rst = audit.get("incumbent_restored", "?")
                        print(
                            "[SceneProof] Local gravity settle (bulk): "
                            f"object={candidate_id}, status={st}, "
                            f"translation_m={tr}, rotation_deg={rdeg}, "
                            f"new_collisions={nc}, restored={rst}, "
                            f"output={path}",
                            flush=True,
                        )
                else:
                    # Single-object mode (backwards compatible).
                    for candidate_id in candidate_ids:
                        audit = audit_sceneproof_local_gravity_settle(
                            render_only_data,
                            candidate_id,
                            duration_seconds=duration_seconds,
                        )
                        audit["placement"] = str(placement_path)
                        path = output_root
                        if not is_single:
                            path = output_root / f"{candidate_id}.json"
                        path.write_text(
                            json.dumps(audit, indent=2), encoding="utf-8"
                        )
                        st = audit.get("status", "?")
                        tr = audit.get("translation_delta_m", "?")
                        rdeg = audit.get("rotation_delta_deg", "?")
                        nc = len(audit.get("new_collision_object_ids", []))
                        rst = audit.get("incumbent_restored", "?")
                        print(
                            "[SceneProof] Local gravity settle: "
                            f"object={candidate_id}, status={st}, "
                            f"translation_m={tr}, rotation_deg={rdeg}, "
                            f"new_collisions={nc}, restored={rst}, "
                            f"output={path}",
                            flush=True,
                        )
            if os.environ.get(
                "IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}:
                return
        if vertical_transaction_output:
            candidate_ids = [
                value.strip()
                for value in os.environ.get(
                    "IMAGINARIUM_SCENEPROOF_VERTICAL_TRANSACTION_OBJECT_IDS", ""
                ).split(",")
                if value.strip()
            ]
            if not candidate_ids and os.environ.get(
                "IMAGINARIUM_SCENEPROOF_VERTICAL_TRANSACTION_AUTO_DISCOVER", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}:
                discovery = audit_sceneproof_true_mesh_com_support(render_only_data)
                ranked = []
                for candidate_id, row in discovery.get("objects", {}).items():
                    drop = row.get("vertical_first_contact_candidate")
                    margin = row.get("declared_parent_surface_margin_m")
                    if not isinstance(drop, dict) or not isinstance(margin, (int, float)):
                        continue
                    if float(margin) >= -float(
                        os.environ.get(
                            "IMAGINARIUM_SCENEPROOF_VERTICAL_OVERHANG_MARGIN_M",
                            "0.005",
                        )
                    ):
                        continue
                    ranked.append((-float(margin), float(drop["drop_m"]), candidate_id))
                maximum_candidates = int(
                    os.environ.get(
                        "IMAGINARIUM_SCENEPROOF_VERTICAL_MAX_CANDIDATES_PER_SCENE",
                        "5",
                    )
                )
                selected = sorted(ranked, reverse=True)[:maximum_candidates]
                candidate_ids = [
                    candidate_id
                    for _, _, candidate_id in sorted(selected, key=lambda row: row[1])
                ]
            if not candidate_ids:
                transaction_path = Path(vertical_transaction_output).resolve()
                transaction_path.parent.mkdir(parents=True, exist_ok=True)
                empty_audit = {
                    "schema_version": "sceneproof_sequential_vertical_first_contact_v1",
                    "policy": "ordered_transactional_z_only_true_mesh_first_contact",
                    "object_order": [],
                    "accepted_object_ids": [],
                    "transactions": [],
                    "reason": "no_auto_discovered_candidates",
                }
                transaction_path.write_text(
                    json.dumps(empty_audit, indent=2), encoding="utf-8"
                )
                candidate_placement = os.environ.get(
                    "IMAGINARIUM_SCENEPROOF_VERTICAL_TRANSACTION_PLACEMENT_OUTPUT"
                )
                if candidate_placement:
                    candidate_path = Path(candidate_placement).resolve()
                    candidate_path.parent.mkdir(parents=True, exist_ok=True)
                    candidate_path.write_text(
                        json.dumps(render_only_data, indent=2), encoding="utf-8"
                    )
                print(
                    "[SceneProof] Sequential vertical first-contact: "
                    "accepted=0/0, objects=, reason=no_auto_discovered_candidates",
                    flush=True,
                )
                if os.environ.get(
                    "IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER", "0"
                ).strip().lower() in {"1", "true", "yes", "on"}:
                    return
            else:
                transaction_audit = audit_sceneproof_sequential_vertical_first_contact(
                    render_only_data,
                    candidate_ids,
                    source_placement_path=obj_placement_info_json_path,
                    scene_camera=scene_camera,
                    visibility_tolerance=float(
                        os.environ.get(
                            "IMAGINARIUM_SCENEPROOF_VERTICAL_VISIBILITY_TOLERANCE",
                            "0.005",
                        )
                    ),
                    visibility_resolution=int(
                        os.environ.get(
                            "IMAGINARIUM_SCENEPROOF_VERTICAL_VISIBILITY_RESOLUTION",
                            "256",
                        )
                    ),
                )
                transaction_path = Path(vertical_transaction_output).resolve()
                transaction_path.parent.mkdir(parents=True, exist_ok=True)
                transaction_audit["placement"] = str(placement_path)
                transaction_path.write_text(
                    json.dumps(transaction_audit, indent=2), encoding="utf-8"
                )
                candidate_placement = os.environ.get(
                    "IMAGINARIUM_SCENEPROOF_VERTICAL_TRANSACTION_PLACEMENT_OUTPUT"
                )
                if candidate_placement:
                    candidate_path = Path(candidate_placement).resolve()
                    candidate_path.parent.mkdir(parents=True, exist_ok=True)
                    candidate_path.write_text(
                        json.dumps(render_only_data, indent=2), encoding="utf-8"
                    )
                    transaction_audit["candidate_placement"] = str(candidate_path)
                    transaction_path.write_text(
                        json.dumps(transaction_audit, indent=2), encoding="utf-8"
                    )
                print(
                    "[SceneProof] Sequential vertical first-contact: "
                    f"accepted={len(transaction_audit['accepted_object_ids'])}/"
                    f"{len(candidate_ids)}, objects="
                    f"{','.join(transaction_audit['accepted_object_ids'])}, "
                    f"output={transaction_path}",
                    flush=True,
                )
                if os.environ.get(
                    "IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER", "0"
                ).strip().lower() in {"1", "true", "yes", "on"}:
                    return
        if com_audit_output:
            com_audit = audit_sceneproof_true_mesh_com_support(
                render_only_data,
                contact_tolerance_m=float(
                    os.environ.get(
                        "IMAGINARIUM_SCENEPROOF_COM_CONTACT_TOLERANCE_M",
                        "0.05",
                    )
                ),
                surface_band_m=float(
                    os.environ.get(
                        "IMAGINARIUM_SCENEPROOF_COM_SURFACE_BAND_M",
                        "0.01",
                    )
                ),
                normal_cosine=float(
                    os.environ.get(
                        "IMAGINARIUM_SCENEPROOF_COM_NORMAL_COSINE", "0.7"
                    )
                ),
                stability_tolerance_m=float(
                    os.environ.get(
                        "IMAGINARIUM_SCENEPROOF_COM_STABILITY_TOLERANCE_M",
                        "0.005",
                    )
                ),
            )
            com_audit["placement"] = str(placement_path)
            com_audit_path = Path(com_audit_output).resolve()
            com_audit_path.parent.mkdir(parents=True, exist_ok=True)
            com_audit_path.write_text(
                json.dumps(com_audit, indent=2), encoding="utf-8"
            )
            summary = com_audit["summary"]
            print(
                "[SceneProof] True-mesh COM support audit complete: "
                f"measured={summary['measured']}, "
                f"abstained={summary['abstained']}, "
                f"stable/marginal/unstable="
                f"{summary['stable']}/{summary['marginal']}/{summary['unstable']}, "
                f"pose_delta={com_audit['maximum_pose_delta']:.8g}, "
                f"output={com_audit_path}",
                flush=True,
            )
            skip_render = os.environ.get(
                "IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER", "0"
            ).strip().lower() in {"1", "true", "yes", "on"}
            if skip_render:
                return

        # ---- global gravity settle (all objects, one sim) ----
        global_settle_output = os.environ.get(
            "IMAGINARIUM_SCENEPROOF_GLOBAL_SETTLE_AUDIT_OUTPUT"
        )
        if global_settle_output:
            gs_duration = float(
                os.environ.get(
                    "IMAGINARIUM_SCENEPROOF_GLOBAL_SETTLE_DURATION_SECONDS",
                    "2.0",
                )
            )
            gs = audit_sceneproof_global_gravity_settle(
                render_only_data,
                duration_seconds=gs_duration,
                output_root=global_settle_output,
            )
            print(
                "[SceneProof] Global gravity settle complete: "
                f"objects={len(gs)}, "
                f"output={global_settle_output}",
                flush=True,
            )
        # -------------------------------------------------------

        if os.environ.get(
            "IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}:
            return

        if os.environ.get(
            "IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}:
            return

        render_only_output = Path(
            os.environ.get(
                "IMAGINARIUM_S4_RENDER_ONLY_OUTPUT",
                os.path.join(output_folder, f"{scene_name}_render_simu.png"),
            )
        ).resolve()
        render_only_output.parent.mkdir(parents=True, exist_ok=True)
        render_samples = int(
            os.environ.get("IMAGINARIUM_S4_RENDER_ONLY_SAMPLES", "256")
        )
        # 如果环境变量要求相机框定特定物体，在渲染前重新定位相机。
        cam_target = os.environ.get("IMAGINARIUM_S4_RENDER_ONLY_CAMERA_TARGET")
        if cam_target and cam_target in render_only_info:
            target_matrix = np.asarray(
                render_only_info[cam_target]["pose_matrix_for_blender"],
                dtype=np.float64,
            )
            target_pos = target_matrix[:3, 3]
            eye_dist = float(
                os.environ.get("IMAGINARIUM_S4_RENDER_ONLY_CAMERA_DISTANCE", "3.0")
            )
            eye_elev = math.radians(
                float(os.environ.get("IMAGINARIUM_S4_RENDER_ONLY_CAMERA_ELEVATION_DEG", "30"))
            )
            eye_az = math.radians(
                float(os.environ.get("IMAGINARIUM_S4_RENDER_ONLY_CAMERA_AZIMUTH_DEG", "45"))
            )
            eye = np.array([
                eye_dist * math.cos(eye_elev) * math.sin(eye_az),
                eye_dist * math.cos(eye_elev) * math.cos(eye_az),
                eye_dist * math.sin(eye_elev),
            ]) + target_pos
            forward = (target_pos - eye) / np.linalg.norm(target_pos - eye)
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, up)
            right /= np.linalg.norm(right)
            new_up = np.cross(right, forward)
            look_at = np.eye(4)
            look_at[:3, 0] = right
            look_at[:3, 1] = new_up
            look_at[:3, 2] = forward
            look_at[:3, 3] = eye
            scene_camera.matrix_world = Matrix(look_at.tolist())
            bpy.context.view_layer.update()
            # 更新锁定姿势以便后渲染漂移审计把它当意向性重定位而非错误
            locked_camera_array = np.asarray(scene_camera.matrix_world, dtype=np.float64)
            print(
                f"[render_only] camera framed on {cam_target} "
                f"eye={eye.tolist()}",
                flush=True,
            )
        blender_manager.render_scene(
            str(render_only_output),
            resolution_x,
            resolution_y,
            samples=render_samples,
        )
        camera_after = np.asarray(
            [list(row) for row in scene_camera.matrix_world],
            dtype=np.float64,
        )
        camera_render_drift = float(
            np.max(np.abs(camera_after - locked_camera_array))
        )
        camera_render_bitwise_stable = bool(
            np.array_equal(camera_after, locked_camera_array)
        )
        if not camera_render_bitwise_stable:
            raise RuntimeError(
                "SceneProof render-only camera changed during rendering: "
                f"max_abs_delta={camera_render_drift:.8g}"
            )
        if not render_only_output.is_file():
            raise RuntimeError(
                f"SceneProof render-only output was not created: {render_only_output}"
            )

        audit_path = Path(
            os.environ.get(
                "IMAGINARIUM_S4_RENDER_ONLY_AUDIT",
                str(render_only_output.with_suffix(".camera.json")),
            )
        ).resolve()
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit = {
            "schema_version": "sceneproof_locked_camera_render_v1",
            "placement": str(placement_path),
            "render": str(render_only_output),
            "camera_policy": "source_s3_scene_camera_locked",
            "camera_object": scene_camera_name,
            "requested_source_camera_matrix": source_camera_array.tolist(),
            "locked_blender_camera_matrix": locked_camera_array.tolist(),
            "render_camera_matrix": camera_after.tolist(),
            "camera_assignment_quantization_max_abs": camera_assignment_delta,
            "camera_render_drift_max_abs": camera_render_drift,
            "camera_render_bitwise_stable": camera_render_bitwise_stable,
            "camera_float32_tolerance": camera_float32_tolerance,
            "resolution": [int(resolution_x), int(resolution_y)],
            "samples": render_samples,
            "applied_object_count": len(applied_objects),
            "ignored_nonrenderable_record_count": len(
                ignored_nonrenderable_records
            ),
            "ignored_nonrenderable_record_ids": sorted(
                set(ignored_nonrenderable_records)
            ),
        }
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(
            "[SceneProof] Locked-camera certified render complete: "
            f"objects={len(applied_objects)}, "
            f"ignored_nonrenderable={len(ignored_nonrenderable_records)}, "
            f"camera_quantization={camera_assignment_delta:.3g}, "
            f"camera_render_drift={camera_render_drift:.3g}, "
            f"render={render_only_output}, audit={audit_path}",
            flush=True,
        )
        return

    # s3
    output_data_s3 = copy.deepcopy(output_data_s2)
    obj_manager = ObjManager(precomputed_voxel_dir=precomputed_voxel_dir)
    obj_manager.obj_info = output_data_s3["obj_info"].copy()
    obj_manager.ground_name = output_data_s3["reference_obj"]
    
    # 加正则表达式模式
    skip_pattern = r'^(wall|floor|ceiling|carpet|rug)_\d+'
    
    # 构建obj_dict和wall_dict
    for instance_id, info in obj_manager.obj_info.items():
        # 跳过墙体、地面等
        if re.match(skip_pattern, instance_id) or instance_id == scene_camera_name:
            print(f"Skipping {instance_id}")
            obj_manager.wall_dict[instance_id] = {"pose_matrix_for_blender": info["pose_matrix_for_blender"]}
            continue
        
        # 跳过内部摆放物体
        if info.get("SpatialRel") == "inside":
            print(f"Skipping inside object: {instance_id}")
            continue
        
        obj = Obj(instance_id, info, base_fbx_path)
        obj_manager.obj_dict[instance_id] = obj
    
    # 体素化
    print("\n初始化体素网格...")
    obj_manager.voxel_manager.initialize_scene_bounds(obj_manager.obj_dict, obj_manager.wall_dict)
    
    # 使用多线程加载体素数据（预计算的体素加载速度快，可以并行）
    print("\n开始体素化（多线程加载）...")
    import time
    start_time = time.time()
    
    def voxelize_single_object(args):
        """单个物体的体素化任务"""
        instance_id, obj, voxel_manager = args
        mesh_path = Path(obj.fbx_path)
        pose = obj.pose_3d
        try:
            voxel_manager.voxelize_object(mesh_path, instance_id, pose, scale=[1.1, 1.1, 1.0])
            return f"  体素化完成: {instance_id}"
        except Exception as e:
            return f"  体素化失败: {instance_id} - {str(e)}"
    
    # 准备任务列表
    tasks = [(instance_id, obj, obj_manager.voxel_manager) for instance_id, obj in obj_manager.obj_dict.items()]
    
    # 使用线程池并行处理（I/O密集型任务）
    max_workers = min(8, len(tasks))  # 最多8个线程
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(voxelize_single_object, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            print(result)
    
    voxel_time = time.time() - start_time
    
    # 打印统计信息
    stats = obj_manager.voxel_manager.voxel_load_stats
    print(f"\n体素化统计:")
    print(f"  预计算加载: {stats['precomputed']} 个")
    print(f"  实时计算: {stats['realtime']} 个")
    print(f"  加载失败: {stats['failed']} 个")
    print(f"  总耗时: {voxel_time:.2f}秒")
    if stats['precomputed'] > 0:
        avg_time = voxel_time / (stats['precomputed'] + stats['realtime'])
        print(f"  平均耗时: {avg_time:.3f}秒/物体")
    
    # 初始化重叠检测
    print("\n初始化重叠检测...")
    obj_manager.init_overlap()
    
    # 计算初始状态
    print("\n初始状态:")
    print(f"  Initial Overlap: {obj_manager.calc_overlap_area(debug_mode=True)}")
    print(f"  Initial Constraints: {obj_manager.calc_constraints()}")
    
    # Run either the legacy SA optimizer or the gated LayoutVLM optimizer.
    initial_temp = 100.0
    alpha = 0.99
    max_iterations = 5000
    penalty_factor = 1000.0

    layoutvlm_pose_matrices = None
    sceneproof_program_bundle = None
    sceneproof_bundle_object = None
    optimization_history = None
    sceneproof_live_factor_parity = None
    sceneproof_factor_binding_audit = None
    sceneproof_materialized_incumbent_audit = None
    if use_layoutvlm:
        if layoutvlm_stage not in LAYOUTVLM_STAGES:
            raise ValueError(
                "Unsupported LayoutVLM stage "
                f"{layoutvlm_stage!r}; expected "
                f"{'/'.join(LAYOUTVLM_STAGES)}."
            )
        print("\n[LayoutVLM] Running differentiable warm-start reprojection...")
        ordered_ids = list(obj_manager.obj_dict)
        reference_json_path = os.environ.get(
            "IMAGINARIUM_LAYOUTVLM_REFERENCE_JSON"
        )
        if layoutvlm_stage == "depth":
            if not reference_json_path:
                raise RuntimeError(
                    "Depth-aware LayoutVLM requires "
                    "IMAGINARIUM_LAYOUTVLM_REFERENCE_JSON pointing to the "
                    "frozen v4 full-400 S4 placement JSON."
                )
            reference_path = Path(reference_json_path)
            if not reference_path.is_file():
                raise FileNotFoundError(
                    f"LayoutVLM reference JSON not found: {reference_path}"
                )
            with reference_path.open("r", encoding="utf-8") as handle:
                reference_data = json.load(handle)
            reference_obj_info = reference_data.get("obj_info", {})
            missing_reference = [
                obj_id
                for obj_id in ordered_ids
                if not isinstance(reference_obj_info.get(obj_id), dict)
                or "pose_matrix_for_blender"
                not in reference_obj_info[obj_id]
            ]
            if missing_reference:
                raise RuntimeError(
                    "Frozen v4 S4 reference is missing optimized poses for: "
                    + ", ".join(missing_reference[:20])
                )
            for obj_id in ordered_ids:
                reference_pose = reference_obj_info[obj_id][
                    "pose_matrix_for_blender"
                ]
                obj_manager.obj_dict[obj_id].pose_3d = reference_pose
                output_data_s3["obj_info"][obj_id][
                    "pose_matrix_for_blender"
                ] = copy.deepcopy(reference_pose)
                blender_obj = bpy.data.objects.get(obj_id)
                if blender_obj is None:
                    raise RuntimeError(
                        "LayoutVLM reference object is missing from Blender: "
                        f"{obj_id}"
                    )
                blender_obj.matrix_world = Matrix(reference_pose)
            bpy.context.view_layer.update()
            print(
                "[LayoutVLM] Loaded frozen full-400 pose reference: "
                f"objects={len(ordered_ids)}, path={reference_path}",
                flush=True,
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        declared_base_matrices = stack_pose_matrices(
            [obj_manager.obj_dict[obj_id].pose_3d for obj_id in ordered_ids],
            device=device,
        )
        use_materialized_incumbent = (
            os.environ.get(
                "IMAGINARIUM_SCENEPROOF_MATERIALIZED_WARM_START",
                "0",
            ).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if use_materialized_incumbent:
            materialized_poses = [
                [list(row) for row in bpy.data.objects[obj_id].matrix_world]
                for obj_id in ordered_ids
            ]
            base_matrices = stack_pose_matrices(
                materialized_poses,
                device=device,
            )
            sceneproof_materialized_incumbent_audit = {
                "policy": "post_coordinate_conversion_blender_world_matrix",
                "objects": len(ordered_ids),
                "max_abs_delta_from_declared_pose": float(
                    torch.max(
                        torch.abs(base_matrices - declared_base_matrices)
                    ).detach().item()
                ),
            }
            print(
                "[SceneProof] Captured materialized warm-start incumbent: "
                f"objects={len(ordered_ids)}, "
                "source=post_coordinate_conversion_blender_world_matrix, "
                "max_abs_delta_from_declared_pose="
                f"{sceneproof_materialized_incumbent_audit['max_abs_delta_from_declared_pose']:.8g}",
                flush=True,
            )
        else:
            base_matrices = declared_base_matrices
        reprojection_error = identity_reprojection_error(base_matrices)
        yaw_delta, translation = initialize_pose_variables(base_matrices)
        reprojected = reproject_pose_matrices(
            base_matrices, yaw_delta, translation
        )
        if reprojection_error.item() > 1e-5:
            raise RuntimeError(
                "LayoutVLM warm-start reprojection changed the deterministic "
                f"S4-S2 pose (max_abs_error={reprojection_error.item():.8g})."
            )
        optimization_history = []
        active_set_router = False
        solver_name = "adam"
        if layoutvlm_stage in {
            "collision",
            "contact",
            "wall",
            "semantic",
            "boundary",
            "full",
            "depth",
        }:
            optimization_geometry_info = output_data_s3["obj_info"]
            optimization_geometry_path = None
            explicit_geometry_path = os.environ.get(
                "IMAGINARIUM_LAYOUTVLM_GEOMETRY_SNAPSHOT"
            )
            if explicit_geometry_path:
                geometry_candidates = [Path(explicit_geometry_path)]
            else:
                source_result_root = (
                    Path(obj_placement_info_json_path).resolve().parent.parent
                )
                geometry_candidates = sorted(
                    (
                        source_result_root / "S4_layout_refinement"
                    ).glob("*_placement_info_s3.json")
                )
            if len(geometry_candidates) > 1:
                raise RuntimeError(
                    "expected at most one frozen source geometry snapshot, "
                    f"found {len(geometry_candidates)}: "
                    + ", ".join(str(path) for path in geometry_candidates)
                )
            if geometry_candidates:
                geometry_path = geometry_candidates[0]
                if not geometry_path.is_file():
                    raise FileNotFoundError(
                        "frozen source geometry snapshot does not exist: "
                        f"{geometry_path}"
                    )
                with geometry_path.open("r", encoding="utf-8") as handle:
                    geometry_snapshot = json.load(handle)
                candidate_info = geometry_snapshot.get("obj_info")
                if not isinstance(candidate_info, dict):
                    raise ValueError(
                        "frozen source geometry snapshot has no obj_info: "
                        f"{geometry_path}"
                    )
                optimization_geometry_info = candidate_info
                optimization_geometry_path = geometry_path
            print(
                "[LayoutVLM] Frozen geometry snapshot: "
                + (
                    str(optimization_geometry_path)
                    if optimization_geometry_path is not None
                    else "current_import_fallback"
                ),
                flush=True,
            )
            local_corner_batches = []
            footprint_hull_counts = []
            frozen_s3_geometry_count = 0
            frozen_quad_footprint_count = 0
            frozen_nonquad_footprint_count = 0
            blender_geometry_fallback_count = 0
            for obj_id in ordered_ids:
                blender_obj = bpy.data.objects.get(obj_id)
                if blender_obj is None:
                    raise RuntimeError(
                        f"LayoutVLM object disappeared before optimization: {obj_id}"
                    )
                source_info = optimization_geometry_info.get(obj_id, {})
                try:
                    frozen_world_bbox = np.asarray(
                        source_info.get("bbox"), dtype=np.float64
                    )
                    frozen_pose = np.asarray(
                        source_info.get("pose_matrix_for_blender"),
                        dtype=np.float64,
                    )
                    if frozen_world_bbox.shape != (8, 3):
                        raise ValueError("bbox is not an 8-corner box")
                    if frozen_pose.shape != (4, 4):
                        raise ValueError("pose is not 4x4")
                    if not (
                        np.isfinite(frozen_world_bbox).all()
                        and np.isfinite(frozen_pose).all()
                    ):
                        raise ValueError("bbox or pose is non-finite")
                    homogeneous_bbox = np.concatenate(
                        (
                            frozen_world_bbox,
                            np.ones((8, 1), dtype=np.float64),
                        ),
                        axis=1,
                    )
                    local_bbox = (
                        homogeneous_bbox @ np.linalg.inv(frozen_pose).T
                    )[:, :3]
                    hull_indices = _convex_hull_indices_2d(
                        frozen_world_bbox[:, :2]
                    )
                    if not 3 <= len(hull_indices) <= 8:
                        raise ValueError(
                            "frozen bbox has a degenerate projected footprint: "
                            f"vertices={len(hull_indices)}"
                        )
                    footprint_hull_counts.append(len(hull_indices))
                    remaining_indices = [
                        index
                        for index in range(8)
                        if index not in hull_indices
                    ]
                    local_bbox = local_bbox[
                        hull_indices + remaining_indices
                    ]
                    if len(hull_indices) == 4:
                        frozen_quad_footprint_count += 1
                    else:
                        frozen_nonquad_footprint_count += 1
                    frozen_s3_geometry_count += 1
                except (KeyError, TypeError, ValueError, np.linalg.LinAlgError):
                    # Legacy or synthetic inputs may not persist an auditable
                    # S3 bbox. Keep those inputs runnable, but make the fallback
                    # explicit in the log instead of silently mixing geometry
                    # definitions with the physical evaluator.
                    local_bbox = np.asarray(
                        [tuple(corner) for corner in blender_obj.bound_box],
                        dtype=np.float64,
                    )
                    minimum = local_bbox.min(axis=0)
                    maximum = local_bbox.max(axis=0)
                    selector = np.asarray(
                        [
                            [0, 0, 0], [0, 0, 1],
                            [0, 1, 0], [0, 1, 1],
                            [1, 0, 0], [1, 0, 1],
                            [1, 1, 0], [1, 1, 1],
                        ],
                        dtype=np.float64,
                    )
                    local_bbox = (
                        minimum[None, :] * (1.0 - selector)
                        + maximum[None, :] * selector
                    )
                    fallback_world_bbox = np.asarray(
                        [
                            tuple(blender_obj.matrix_world @ Vector(corner))
                            for corner in local_bbox
                        ],
                        dtype=np.float64,
                    )
                    hull_indices = _convex_hull_indices_2d(
                        fallback_world_bbox[:, :2]
                    )
                    if not 3 <= len(hull_indices) <= 8:
                        raise RuntimeError(
                            "Blender fallback bbox has a degenerate projected "
                            f"footprint for {obj_id}: "
                            f"vertices={len(hull_indices)}"
                        )
                    remaining_indices = [
                        index
                        for index in range(8)
                        if index not in hull_indices
                    ]
                    local_bbox = local_bbox[
                        hull_indices + remaining_indices
                    ]
                    blender_geometry_fallback_count += 1
                    footprint_hull_counts.append(len(hull_indices))
                local_corner_batches.append(local_bbox)
            print(
                "[LayoutVLM] Optimization geometry source: "
                f"frozen_s3={frozen_s3_geometry_count}, "
                f"frozen_quad={frozen_quad_footprint_count}, "
                f"frozen_nonquad={frozen_nonquad_footprint_count}, "
                f"blender_fallback={blender_geometry_fallback_count}",
                flush=True,
            )
            local_corners = torch.as_tensor(
                np.asarray(local_corner_batches),
                dtype=base_matrices.dtype,
                device=device,
            )
            footprint_hull_sizes = torch.as_tensor(
                footprint_hull_counts,
                dtype=torch.long,
                device=device,
            )

            id_to_index = {
                obj_id: index for index, obj_id in enumerate(ordered_ids)
            }
            support_pair_keys = set()
            support_index_pairs = []
            fixed_support_indices = []
            fixed_support_heights = []
            fixed_support_pairs = []
            plane_object_indices = []
            plane_points = []
            plane_normals = []
            plane_orientation_mask = []
            plane_bindings = []
            max_initial_contact_gap = float(
                os.environ.get(
                    "IMAGINARIUM_LAYOUTVLM_MAX_CONTACT_GAP",
                    "0.5",
                )
            )

            world_z_bounds = {}
            for obj_id in ordered_ids:
                blender_obj = bpy.data.objects.get(obj_id)
                points = [
                    blender_obj.matrix_world @ Vector(corner)
                    for corner in blender_obj.bound_box
                ]
                world_z_bounds[obj_id] = (
                    min(point.z for point in points),
                    max(point.z for point in points),
                )

            for child_id, child in obj_manager.obj_dict.items():
                if child.parent_id in id_to_index:
                    child_index = id_to_index[child_id]
                    parent_index = id_to_index[child.parent_id]
                    pair_key = frozenset((child_index, parent_index))
                    if child.relation == "inside":
                        support_pair_keys.add(pair_key)
                        continue
                    if child.relation == "on":
                        initial_gap = (
                            world_z_bounds[child_id][0]
                            - world_z_bounds[child.parent_id][1]
                        )
                        if abs(initial_gap) > max_initial_contact_gap:
                            print(
                                "[LayoutVLM] Skipping anomalous support "
                                f"{child_id} -> {child.parent_id}: "
                                f"initial_gap={initial_gap:.6f}m exceeds "
                                f"{max_initial_contact_gap:.3f}m",
                                flush=True,
                            )
                            continue
                        support_pair_keys.add(pair_key)
                        support_index_pairs.append(
                            (child_index, parent_index)
                        )
                elif child.parent_id and child.relation == "on":
                    if re.match(r"^(wall|ceiling)_\d+$", child.parent_id):
                        print(
                            "[LayoutVLM] Deferring plane support "
                            f"{child_id} -> {child.parent_id} to the "
                            "wall/ceiling constraint stage.",
                            flush=True,
                        )
                        continue
                    if not re.match(
                        r"^(floor|ground)_\d+$",
                        child.parent_id,
                    ):
                        print(
                            "[LayoutVLM] Skipping unknown fixed support "
                            f"{child_id} -> {child.parent_id}",
                            flush=True,
                        )
                        continue
                    fixed_parent = bpy.data.objects.get(child.parent_id)
                    if fixed_parent is not None:
                        parent_world_bbox = [
                            fixed_parent.matrix_world @ Vector(corner)
                            for corner in fixed_parent.bound_box
                        ]
                        support_height = max(
                            point.z for point in parent_world_bbox
                        )
                        initial_gap = (
                            world_z_bounds[child_id][0] - support_height
                        )
                        if abs(initial_gap) > max_initial_contact_gap:
                            print(
                                "[LayoutVLM] Skipping anomalous fixed support "
                                f"{child_id} -> {child.parent_id}: "
                                f"initial_gap={initial_gap:.6f}m exceeds "
                                f"{max_initial_contact_gap:.3f}m",
                                flush=True,
                            )
                            continue
                        fixed_support_indices.append(id_to_index[child_id])
                        fixed_support_heights.append(support_height)
                        fixed_support_pairs.append((child_id, child.parent_id))

            # Fixed wall/ceiling planes are not optimization objects.  Build
            # their actual surface planes from the current Blender geometry,
            # with each normal pointing from the surface toward its child.
            seen_plane_constraints = set()
            for child_id, child in obj_manager.obj_dict.items():
                plane_ids = []
                against_wall = child.is_against_wall
                if isinstance(against_wall, (list, tuple)):
                    plane_ids.extend(against_wall)
                elif against_wall:
                    plane_ids.append(against_wall)
                if (
                    child.parent_id
                    and re.match(r"^(wall|ceiling)_\d+$", child.parent_id)
                ):
                    plane_ids.append(child.parent_id)

                for plane_id in plane_ids:
                    constraint_key = (child_id, plane_id)
                    if constraint_key in seen_plane_constraints:
                        continue
                    seen_plane_constraints.add(constraint_key)
                    plane_obj = bpy.data.objects.get(plane_id)
                    child_obj = bpy.data.objects.get(child_id)
                    if plane_obj is None or child_obj is None:
                        print(
                            "[LayoutVLM] Skipping missing fixed plane "
                            f"{child_id} -> {plane_id}",
                            flush=True,
                        )
                        continue

                    plane_bbox = [
                        plane_obj.matrix_world @ Vector(corner)
                        for corner in plane_obj.bound_box
                    ]
                    child_bbox = [
                        child_obj.matrix_world @ Vector(corner)
                        for corner in child_obj.bound_box
                    ]
                    plane_center = sum(plane_bbox, Vector()) / len(plane_bbox)
                    child_center = sum(child_bbox, Vector()) / len(child_bbox)
                    plane_normal = (
                        plane_obj.matrix_world.to_3x3() @ Vector((0, 0, 1))
                    )
                    is_wall = bool(re.match(r"^wall_\d+$", plane_id))
                    if is_wall:
                        plane_normal.z = 0.0
                    if plane_normal.length <= 1e-8:
                        print(
                            "[LayoutVLM] Skipping degenerate fixed plane "
                            f"{child_id} -> {plane_id}",
                            flush=True,
                        )
                        continue
                    plane_normal.normalize()
                    if plane_normal.dot(child_center - plane_center) < 0:
                        plane_normal.negate()
                    half_thickness = max(
                        abs((point - plane_center).dot(plane_normal))
                        for point in plane_bbox
                    )
                    surface_point = (
                        plane_center + plane_normal * half_thickness
                    )
                    initial_plane_gap = min(
                        (point - surface_point).dot(plane_normal)
                        for point in child_bbox
                    )
                    if abs(initial_plane_gap) > max_initial_contact_gap:
                        print(
                            "[LayoutVLM] Skipping anomalous fixed plane "
                            f"{child_id} -> {plane_id}: "
                            f"initial_gap={initial_plane_gap:.6f}m exceeds "
                            f"{max_initial_contact_gap:.3f}m",
                            flush=True,
                        )
                        continue
                    plane_object_indices.append(id_to_index[child_id])
                    plane_points.append(tuple(surface_point))
                    plane_normals.append(tuple(plane_normal))
                    plane_orientation_mask.append(is_wall)
                    plane_bindings.append(
                        {
                            "child_id": child_id,
                            "plane_id": plane_id,
                            "point": tuple(surface_point),
                            "normal": tuple(plane_normal),
                            "orientation_required": is_wall,
                        }
                    )

            collision_pairs = []
            for first_index, candidate_indices in enumerate(
                obj_manager.overlap_list
            ):
                for second_index in candidate_indices:
                    pair_key = frozenset((first_index, second_index))
                    if pair_key in support_pair_keys:
                        continue
                    collision_pairs.append((first_index, second_index))
            pair_indices = pair_index_tensor(collision_pairs, device=device)
            optimization_iterations = int(
                os.environ.get("IMAGINARIUM_LAYOUTVLM_ITERATIONS", "100")
            )
            support_pair_tensor = pair_index_tensor(
                support_index_pairs,
                device=device,
            )
            fixed_support_index_tensor = torch.as_tensor(
                fixed_support_indices,
                dtype=torch.long,
                device=device,
            )
            fixed_support_height_tensor = torch.as_tensor(
                fixed_support_heights,
                dtype=base_matrices.dtype,
                device=device,
            )
            if layoutvlm_stage == "collision":
                reprojected, optimization_history = optimize_collision_stage(
                    base_matrices,
                    local_corners,
                    pair_indices,
                    iterations=optimization_iterations,
                )
                max_z_drift = torch.max(
                    torch.abs(
                        reprojected[:, 2, 3] - base_matrices[:, 2, 3]
                    )
                ).item()
                stage_summary = f"max_z_drift={max_z_drift:.8g}"
            elif layoutvlm_stage == "contact":
                reprojected, optimization_history = optimize_contact_stage(
                    base_matrices,
                    local_corners,
                    pair_indices,
                    support_pair_tensor,
                    fixed_support_index_tensor,
                    fixed_support_height_tensor,
                    iterations=optimization_iterations,
                )
                stage_summary = (
                    f"support_pairs={len(support_index_pairs)}, "
                    f"fixed_supports={len(fixed_support_indices)}, "
                    f"projected_max_contact_gap="
                    f"{optimization_history[-1]['projected_max_contact_gap']:.8g}"
                )
            else:
                plane_object_index_tensor = torch.as_tensor(
                    plane_object_indices,
                    dtype=torch.long,
                    device=device,
                )
                plane_point_tensor = torch.as_tensor(
                    plane_points,
                    dtype=base_matrices.dtype,
                    device=device,
                ).reshape(-1, 3)
                plane_normal_tensor = torch.as_tensor(
                    plane_normals,
                    dtype=base_matrices.dtype,
                    device=device,
                ).reshape(-1, 3)
                plane_orientation_tensor = torch.as_tensor(
                    plane_orientation_mask,
                    dtype=torch.bool,
                    device=device,
                )
                if layoutvlm_stage == "wall":
                    reprojected, optimization_history = optimize_plane_stage(
                        base_matrices,
                        local_corners,
                        pair_indices,
                        support_pair_tensor,
                        fixed_support_index_tensor,
                        fixed_support_height_tensor,
                        plane_object_index_tensor,
                        plane_point_tensor,
                        plane_normal_tensor,
                        plane_orientation_tensor,
                        iterations=optimization_iterations,
                    )
                    stage_summary = (
                        f"support_pairs={len(support_index_pairs)}, "
                        f"fixed_supports={len(fixed_support_indices)}, "
                        f"wall_planes={sum(plane_orientation_mask)}, "
                        f"ceiling_planes="
                        f"{len(plane_orientation_mask) - sum(plane_orientation_mask)}, "
                        f"projected_max_contact_gap="
                        f"{optimization_history[-1]['projected_max_contact_gap']:.8g}, "
                        f"projected_max_plane_gap="
                        f"{optimization_history[-1]['projected_max_plane_gap']:.8g}"
                    )
                else:
                    with torch.no_grad():
                        initial_world_corners = transform_points(
                            base_matrices,
                            local_corners,
                        )
                        footprint_sizes = (
                            initial_world_corners[:, :, :2].amax(dim=1)
                            - initial_world_corners[:, :, :2].amin(dim=1)
                        ).detach().cpu().tolist()
                    semantic_specs = build_semantic_relation_specs(
                        output_data_s3["obj_info"],
                        ordered_ids,
                        base_matrices.detach().cpu().tolist(),
                        footprint_sizes,
                    )
                    if os.environ.get(
                        "IMAGINARIUM_SCENEPROOF_PROGRAM_IR", "0"
                    ).strip().lower() in {"1", "true", "yes", "on"}:
                        sceneproof_bundle = compile_legacy_relation_programs(
                            scene_id=Path(
                                obj_placement_info_json_path
                            ).stem,
                            obj_info=output_data_s3["obj_info"],
                            ordered_ids=ordered_ids,
                            support_pairs=support_index_pairs,
                            fixed_support_pairs=fixed_support_pairs,
                            plane_bindings=plane_bindings,
                            collision_pairs=collision_pairs,
                            semantic_specs=semantic_specs,
                            support_topology_authoritative=True,
                        )
                        sceneproof_bundle_object = sceneproof_bundle
                        sceneproof_program_bundle = sceneproof_bundle.to_dict()
                        sceneproof_program_bundle["content_hash"] = (
                            sceneproof_bundle.content_hash()
                        )
                        sceneproof_live_factor_parity = audit_live_factor_parity(
                            sceneproof_bundle,
                            ordered_ids=ordered_ids,
                            support_pairs=support_index_pairs,
                            fixed_support_pairs=fixed_support_pairs,
                            plane_bindings=plane_bindings,
                            collision_pairs=collision_pairs,
                            semantic_specs=semantic_specs,
                        )
                        print(
                            "[SceneProof] Live factor parity: "
                            f"passed={sceneproof_live_factor_parity['passed']}, "
                            "expected="
                            f"{sceneproof_live_factor_parity['expected_program_kinds']}, "
                            "compiled="
                            f"{sceneproof_live_factor_parity['compiled_program_kinds']}, "
                            "mismatches="
                            f"{sceneproof_live_factor_parity['mismatches']}",
                            flush=True,
                        )
                        if (
                            os.environ.get(
                                "IMAGINARIUM_SCENEPROOF_REQUIRE_FACTOR_PARITY",
                                "0",
                            ).strip().lower()
                            in {"1", "true", "yes", "on"}
                            and not sceneproof_live_factor_parity["passed"]
                        ):
                            raise RuntimeError(
                                "SceneProof live factor parity failed: "
                                f"{sceneproof_live_factor_parity['mismatches']}"
                            )
                        print(
                            "[SceneProof] Compiled Relation Program IR: "
                            f"programs={len(sceneproof_bundle.programs)}, "
                            "abstained="
                            f"{len(sceneproof_bundle.abstained_relations)}, "
                            "rejected="
                            f"{len(sceneproof_bundle.rejected_relations)}, "
                            "solver_binding=audit_only_until_block_parity",
                            flush=True,
                        )
                    for skipped_relation in semantic_specs["skipped"]:
                        print(
                            "[LayoutVLM] Skipping semantic relation "
                            f"{skipped_relation['relation']} "
                            f"{skipped_relation['source']} -> "
                            f"{skipped_relation['target']}: "
                            f"{skipped_relation['reason']}",
                            flush=True,
                        )
                    point_pair_tensor = pair_index_tensor(
                        semantic_specs["point_pairs"],
                        device=device,
                    )
                    point_offset_tensor = torch.as_tensor(
                        semantic_specs["point_offsets"],
                        dtype=base_matrices.dtype,
                        device=device,
                    )
                    distance_pair_tensor = pair_index_tensor(
                        semantic_specs["distance_pairs"],
                        device=device,
                    )
                    distance_minimum_tensor = torch.as_tensor(
                        semantic_specs["distance_minimum"],
                        dtype=base_matrices.dtype,
                        device=device,
                    )
                    distance_maximum_tensor = torch.as_tensor(
                        semantic_specs["distance_maximum"],
                        dtype=base_matrices.dtype,
                        device=device,
                    )
                    align_pair_tensor = pair_index_tensor(
                        semantic_specs["align_pairs"],
                        device=device,
                    )
                    align_offset_tensor = torch.as_tensor(
                        semantic_specs["align_offsets"],
                        dtype=base_matrices.dtype,
                        device=device,
                    )
                    max_initial_containment_error = float(
                        os.environ.get(
                            "IMAGINARIUM_LAYOUTVLM_MAX_CONTAINMENT_ERROR",
                            "0.5",
                        )
                    )
                    with torch.no_grad():
                        (
                            containment_pair_tensor,
                            rejected_containment_pair_tensor,
                            initial_containment_errors,
                        ) = gate_support_containment_pairs(
                            base_matrices,
                            local_corners,
                            support_pair_tensor,
                            max_initial_containment_error,
                            footprint_hull_sizes=footprint_hull_sizes,
                        )
                    containment_index_pairs = containment_pair_tensor.tolist()
                    skipped_containment_pairs = (
                        rejected_containment_pair_tensor.tolist()
                    )
                    for (
                        (child_index, parent_index),
                        initial_error,
                    ) in zip(
                        support_index_pairs,
                        initial_containment_errors.tolist(),
                    ):
                        if initial_error <= max_initial_containment_error:
                            continue
                        print(
                            "[LayoutVLM] Skipping anomalous containment "
                            f"{ordered_ids[child_index]} -> "
                            f"{ordered_ids[parent_index]}: "
                            f"initial_footprint_error="
                            f"{initial_error:.6f}m exceeds "
                            f"{max_initial_containment_error:.3f}m; "
                            "vertical contact is retained.",
                            flush=True,
                        )
                    if sceneproof_bundle_object is not None:
                        runtime_factor_rows = build_runtime_factor_rows(
                            ordered_ids=ordered_ids,
                            support_pairs=support_index_pairs,
                            fixed_support_pairs=fixed_support_pairs,
                            containment_pairs=containment_index_pairs,
                            plane_bindings=plane_bindings,
                            collision_pairs=collision_pairs,
                            semantic_specs=semantic_specs,
                        )
                        sceneproof_factor_binding_audit = (
                            audit_factor_semantics_and_ownership(
                                sceneproof_bundle_object,
                                runtime_factor_rows,
                            )
                        )
                        print(
                            "[SceneProof] Factor semantics + block ownership: "
                            f"passed={sceneproof_factor_binding_audit['passed']}, "
                            "runtime_rows="
                            f"{sceneproof_factor_binding_audit['runtime_rows']}, "
                            "bound="
                            f"{sceneproof_factor_binding_audit['bound_solver_factors']}, "
                            "abstained="
                            f"{sceneproof_factor_binding_audit['abstained_solver_factors']}, "
                            "safe_leaf_translations="
                            f"{len(sceneproof_factor_binding_audit['safe_leaf_translation_objects'])}, "
                            "mismatches="
                            f"{sceneproof_factor_binding_audit['mismatches']}",
                            flush=True,
                        )
                        if (
                            os.environ.get(
                                "IMAGINARIUM_SCENEPROOF_REQUIRE_BINDING_AUDIT",
                                "0",
                            ).strip().lower()
                            in {"1", "true", "yes", "on"}
                            and not sceneproof_factor_binding_audit["passed"]
                        ):
                            raise RuntimeError(
                                "SceneProof factor semantics/block ownership "
                                "audit failed: "
                                f"{sceneproof_factor_binding_audit['mismatches']}"
                            )
                    boundary_object_index_tensor = torch.empty(
                        (0,),
                        dtype=torch.long,
                        device=device,
                    )
                    boundary_point_tensor = base_matrices.new_zeros((0, 2))
                    boundary_normal_tensor = base_matrices.new_zeros((0, 2))
                    boundary_floor_id = None
                    if layoutvlm_stage in {"boundary", "full", "depth"}:
                        reference_id = output_data_s3.get("reference_obj")
                        floor_candidates = [
                            candidate
                            for candidate in (
                                reference_id,
                                "floor_0",
                                "ground_0",
                            )
                            if isinstance(candidate, str)
                        ]
                        floor_obj = None
                        for candidate in floor_candidates:
                            floor_obj = bpy.data.objects.get(candidate)
                            if floor_obj is not None:
                                boundary_floor_id = candidate
                                break
                        if floor_obj is None:
                            raise RuntimeError(
                                "LayoutVLM boundary stage could not find the "
                                f"reference floor; tried {floor_candidates}."
                            )
                        if (
                            floor_obj.type == "MESH"
                            and len(floor_obj.data.vertices) > 0
                        ):
                            floor_xy = np.asarray(
                                [
                                    tuple(
                                        (floor_obj.matrix_world @ vertex.co)[:2]
                                    )
                                    for vertex in floor_obj.data.vertices
                                ],
                                dtype=np.float64,
                            )
                        else:
                            floor_xy = np.asarray(
                                [
                                    tuple(
                                        (floor_obj.matrix_world @ Vector(corner))[:2]
                                    )
                                    for corner in floor_obj.bound_box
                                ],
                                dtype=np.float64,
                            )
                        floor_xy = np.unique(
                            np.round(floor_xy, decimals=6),
                            axis=0,
                        )
                        if floor_xy.shape[0] < 3:
                            raise RuntimeError(
                                "LayoutVLM boundary floor has fewer than "
                                f"three unique XY vertices: {boundary_floor_id}"
                            )
                        try:
                            hull = ConvexHull(floor_xy)
                        except Exception as error:
                            raise RuntimeError(
                                "LayoutVLM could not construct the convex "
                                f"room boundary from {boundary_floor_id}."
                            ) from error
                        boundary_polygon = torch.as_tensor(
                            floor_xy[hull.vertices],
                            dtype=base_matrices.dtype,
                            device=device,
                        )
                        (
                            boundary_point_tensor,
                            boundary_normal_tensor,
                        ) = convex_polygon_halfspaces(boundary_polygon)
                        boundary_object_index_tensor = torch.arange(
                            len(ordered_ids),
                            dtype=torch.long,
                            device=device,
                        )
                        print(
                            "[LayoutVLM] Room boundary built from "
                            f"{boundary_floor_id}: "
                            f"vertices={boundary_point_tensor.shape[0]}, "
                            f"objects={boundary_object_index_tensor.shape[0]}",
                            flush=True,
                        )
                    depth_observations = None
                    if layoutvlm_stage == "depth":
                        depth_observations = (
                            build_depth_reprojection_observations(
                                obj_placement_info_json_path,
                                ordered_ids,
                                output_data_s3["obj_info"],
                                scene_camera,
                                base_matrices,
                                local_corners,
                                width=resolution_x,
                                height=resolution_y,
                                dtype=base_matrices.dtype,
                                device=device,
                            )
                        )
                    image_gauge_observations = None
                    if os.environ.get(
                        "IMAGINARIUM_SCENEPROOF_PLANE_COMPONENT_IMAGE_GAUGE",
                        "0",
                    ).strip().lower() in {"1", "true", "yes", "on"}:
                        image_gauge_observations = (
                            build_depth_reprojection_observations(
                                obj_placement_info_json_path,
                                ordered_ids,
                                output_data_s3["obj_info"],
                                scene_camera,
                                base_matrices,
                                local_corners,
                                width=resolution_x,
                                height=resolution_y,
                                dtype=base_matrices.dtype,
                                device=device,
                                image_only=True,
                            )
                        )
                    sceneba_repair_reports = []
                    if (
                        layoutvlm_stage == "depth"
                        and depth_observations is not None
                        and os.environ.get(
                            "IMAGINARIUM_SCENEBA_DISCRETE_REPAIR", "0"
                        ).strip().lower()
                        in {"1", "true", "yes", "on"}
                    ):
                        support_parent_indices = torch.full(
                            (len(ordered_ids),),
                            -1,
                            dtype=torch.long,
                            device=device,
                        )
                        collision_exempt_mask = torch.eye(
                            len(ordered_ids),
                            dtype=torch.bool,
                            device=device,
                        )
                        lock_world_z = torch.zeros(
                            len(ordered_ids),
                            dtype=torch.bool,
                            device=device,
                        )
                        for child_id, child in obj_manager.obj_dict.items():
                            child_index = id_to_index.get(child_id)
                            if child_index is None:
                                continue
                            if child.parent_id:
                                lock_world_z[child_index] = True
                            parent_index = id_to_index.get(child.parent_id)
                            if parent_index is not None:
                                collision_exempt_mask[
                                    child_index, parent_index
                                ] = True
                                collision_exempt_mask[
                                    parent_index, child_index
                                ] = True
                                if child.relation in {"on", "inside"}:
                                    support_parent_indices[
                                        child_index
                                    ] = parent_index
                        yaw_offsets = tuple(
                            float(value)
                            for value in os.environ.get(
                                "IMAGINARIUM_SCENEBA_REPAIR_YAWS",
                                "0,90,180,270",
                            ).split(",")
                            if value.strip()
                        )
                        base_matrices, sceneba_repair_reports = (
                            select_confident_discrete_pose_repairs(
                                base_matrices,
                                local_corners,
                                depth_observations["indices"],
                                depth_observations["boxes"],
                                depth_observations["depths"],
                                depth_observations["size_enabled"],
                                depth_observations["world_to_camera"],
                                depth_observations["image_size"],
                                lock_world_z,
                                support_parent_indices=support_parent_indices,
                                collision_exempt_mask=collision_exempt_mask,
                                visible_surface_depths=depth_observations[
                                    "visible_surface_depths"
                                ],
                                surface_to_center_offsets=depth_observations[
                                    "surface_to_center_offsets"
                                ],
                                enable_asset_center_candidates=(
                                    os.environ.get(
                                        "IMAGINARIUM_SCENEBA_ASSET_CENTER_CANDIDATES",
                                        "0",
                                    )
                                    == "1"
                                ),
                                asset_center_offset_scales=tuple(
                                    float(value)
                                    for value in os.environ.get(
                                        "IMAGINARIUM_SCENEBA_ASSET_CENTER_SCALES",
                                        "0.5,1.0,1.5",
                                    ).split(",")
                                    if value.strip()
                                ),
                                enable_support_surface_candidates=(
                                    os.environ.get(
                                        "IMAGINARIUM_SCENEBA_SUPPORT_SURFACE_CANDIDATES",
                                        "0",
                                    )
                                    == "1"
                                ),
                                yaw_offsets_deg=yaw_offsets,
                                max_translation_m=float(
                                    os.environ.get(
                                        "IMAGINARIUM_SCENEBA_REPAIR_MAX_TRANSLATION",
                                        "0.5",
                                    )
                                ),
                                minimum_relative_improvement=float(
                                    os.environ.get(
                                        "IMAGINARIUM_SCENEBA_REPAIR_MIN_RELATIVE_GAIN",
                                        "0.08",
                                    )
                                ),
                                minimum_absolute_improvement=float(
                                    os.environ.get(
                                        "IMAGINARIUM_SCENEBA_REPAIR_MIN_ABSOLUTE_GAIN",
                                        "0.001",
                                    )
                                ),
                                minimum_runner_up_margin=float(
                                    os.environ.get(
                                        "IMAGINARIUM_SCENEBA_REPAIR_MIN_MARGIN",
                                        "0.0002",
                                    )
                                ),
                            )
                        )
                        accepted_repairs = [
                            report
                            for report in sceneba_repair_reports
                            if report["accepted"]
                        ]
                        for report in sceneba_repair_reports:
                            report["object_id"] = ordered_ids[
                                report["object_index"]
                            ]
                        output_data_s3["sceneba_discrete_repair"] = {
                            "schema_version":
                                "sceneba_discrete_repair_v2",
                            "observed_objects":
                                len(sceneba_repair_reports),
                            "accepted_objects": len(accepted_repairs),
                            "reports": sceneba_repair_reports,
                        }
                        print(
                            "[SceneBA] Discrete repair proposals complete: "
                            f"observed={len(sceneba_repair_reports)}, "
                            f"accepted={len(accepted_repairs)}, "
                            f"max_translation="
                            f"{os.environ.get('IMAGINARIUM_SCENEBA_REPAIR_MAX_TRANSLATION', '0.5')}m",
                            flush=True,
                        )
                        for report in accepted_repairs:
                            print(
                                "[SceneBA] ACCEPT "
                                f"{ordered_ids[report['object_index']]} "
                                f"yaw={report['selected_yaw_deg']:.0f} "
                                f"anchor={report['selected_anchor']} "
                                f"shift={report['selected_translation_shift_m']:.4f}m "
                                f"relative_gain={report['relative_improvement']:.4f} "
                                f"margin={report['runner_up_margin']:.6f}",
                                flush=True,
                            )
                    active_set_router = (
                        os.environ.get(
                            "IMAGINARIUM_LAYOUTVLM_ACTIVE_SET_ROUTER", "0"
                        ).strip().lower()
                        in {"1", "true", "yes", "on"}
                    )
                    if active_set_router and layoutvlm_stage != "full":
                        raise ValueError(
                            "the active-set compute router is currently "
                            "validated only for the full LayoutVLM stage"
                        )
                    checkpoint_values = tuple(
                        int(value.strip())
                        for value in os.environ.get(
                            "IMAGINARIUM_LAYOUTVLM_ROUTER_CHECKPOINTS",
                            "30,100",
                        ).split(",")
                        if value.strip()
                    )
                    router_threshold_env = {
                        "collision": "COLLISION",
                        "contact": "CONTACT",
                        "plane": "PLANE",
                        "orientation": "ORIENTATION",
                        "containment": "CONTAINMENT",
                        "semantic": "SEMANTIC",
                        "boundary": "BOUNDARY",
                        "depth_centre": "DEPTH_CENTER",
                        "depth_size": "DEPTH_SIZE",
                        "depth_relative": "DEPTH_RELATIVE",
                        "translation_update": "TRANSLATION_UPDATE",
                        "yaw_update": "YAW_UPDATE",
                    }
                    router_thresholds = {}
                    for threshold_name, environment_suffix in (
                        router_threshold_env.items()
                    ):
                        environment_name = (
                            "IMAGINARIUM_LAYOUTVLM_ROUTER_"
                            f"{environment_suffix}"
                        )
                        if environment_name in os.environ:
                            router_thresholds[threshold_name] = float(
                                os.environ[environment_name]
                            )
                    router_kwargs = {
                        "active_set_router": active_set_router,
                        "active_set_checkpoints": checkpoint_values,
                        "active_set_thresholds": router_thresholds or None,
                        "active_set_high_degree": int(
                            os.environ.get(
                                "IMAGINARIUM_LAYOUTVLM_ROUTER_HIGH_DEGREE",
                                "6",
                            )
                        ),
                        "active_set_wake_multiplier": float(
                            os.environ.get(
                                "IMAGINARIUM_LAYOUTVLM_ROUTER_WAKE_MULTIPLIER",
                                "1.5",
                            )
                        ),
                    }
                    solver_name = os.environ.get(
                        "IMAGINARIUM_LAYOUTVLM_SOLVER", "adam"
                    ).strip().lower()
                    solver_kwargs = {
                        "solver": solver_name,
                        "lm_initial_damping": float(
                            os.environ.get(
                                "IMAGINARIUM_SCENELM_INITIAL_DAMPING",
                                "0.01",
                            )
                        ),
                        "lm_pcg_iterations": int(
                            os.environ.get(
                                "IMAGINARIUM_SCENELM_PCG_ITERATIONS",
                                "12",
                            )
                        ),
                        "lm_pcg_tolerance": float(
                            os.environ.get(
                                "IMAGINARIUM_SCENELM_PCG_TOLERANCE",
                                "0.001",
                            )
                        ),
                        "lm_acceptance_threshold": float(
                            os.environ.get(
                                "IMAGINARIUM_SCENELM_ACCEPTANCE_THRESHOLD",
                                "0.1",
                            )
                        ),
                        "lm_gradient_tolerance": float(
                            os.environ.get(
                                "IMAGINARIUM_SCENELM_GRADIENT_TOLERANCE",
                                "0.00001",
                            )
                        ),
                        "lm_relative_energy_tolerance": float(
                            os.environ.get(
                                "IMAGINARIUM_SCENELM_RELATIVE_ENERGY_TOLERANCE",
                                "0.0001",
                            )
                        ),
                        "lm_patience": int(
                            os.environ.get(
                                "IMAGINARIUM_SCENELM_PATIENCE",
                                "3",
                            )
                        ),
                        "lm_max_translation_step": float(
                            os.environ.get(
                                "IMAGINARIUM_SCENELM_MAX_TRANSLATION_STEP",
                                "0.2",
                            )
                        ),
                        "lm_max_yaw_step_degrees": float(
                            os.environ.get(
                                "IMAGINARIUM_SCENELM_MAX_YAW_STEP_DEG",
                                "15",
                            )
                        ),
                        "lm_max_relation_releases": int(
                            os.environ.get(
                                "IMAGINARIUM_SCENELM_MAX_RELATION_RELEASES",
                                "1",
                            )
                        ),
                        "lm_collision_witness_weight": float(
                            os.environ.get(
                                "IMAGINARIUM_SCENELM_COLLISION_WITNESS_WEIGHT",
                                "25",
                            )
                        ),
                    }
                    depth_kwargs = {}
                    if depth_observations is not None:
                        depth_kwargs = {
                            "depth_observation_indices":
                                depth_observations["indices"],
                            "depth_observed_boxes":
                                depth_observations["boxes"],
                            "depth_observed_depths":
                                depth_observations["depths"],
                            "depth_observed_weights":
                                depth_observations["weights"],
                            "depth_bbox_size_enabled":
                                depth_observations["size_enabled"],
                            "depth_world_to_camera":
                                depth_observations["world_to_camera"],
                            "depth_image_size":
                                depth_observations["image_size"],
                            "depth_reference_centre_errors":
                                depth_observations[
                                    "reference_centre_errors"
                                ],
                            "depth_reference_size_errors":
                                depth_observations[
                                    "reference_size_errors"
                                ],
                            "depth_reference_relative_errors":
                                depth_observations[
                                    "reference_relative_errors"
                                ],
                            "depth_reprojection_weight": float(
                                os.environ.get(
                                    "IMAGINARIUM_LAYOUTVLM_DEPTH_WEIGHT",
                                    "1.0",
                                )
                            ),
                            "depth_centre_weight": float(
                                os.environ.get(
                                    "IMAGINARIUM_LAYOUTVLM_DEPTH_CENTER_WEIGHT",
                                    "1.0",
                                )
                            ),
                            "depth_size_weight": float(
                                os.environ.get(
                                    "IMAGINARIUM_LAYOUTVLM_DEPTH_SIZE_WEIGHT",
                                    "0.25",
                                )
                            ),
                            "depth_metric_weight": float(
                                os.environ.get(
                                    "IMAGINARIUM_LAYOUTVLM_DEPTH_METRIC_WEIGHT",
                                    "1.0",
                                )
                            ),
                            "optimize_yaw": (
                                os.environ.get(
                                    "IMAGINARIUM_LAYOUTVLM_DEPTH_FREEZE_YAW",
                                    "0",
                                ).strip().lower()
                                not in {"1", "true", "yes", "on"}
                            ),
                            "depth_trust_region_weight": float(
                                os.environ.get(
                                    "IMAGINARIUM_LAYOUTVLM_DEPTH_TRUST_WEIGHT",
                                    "1.0",
                                )
                            ),
                            "depth_centre_margin_pixels": float(
                                os.environ.get(
                                    "IMAGINARIUM_LAYOUTVLM_DEPTH_CENTER_MARGIN_PX",
                                    "2.0",
                                )
                            ),
                            "depth_size_margin_log": float(
                                os.environ.get(
                                    "IMAGINARIUM_LAYOUTVLM_DEPTH_SIZE_MARGIN_LOG",
                                    "0.02",
                                )
                            ),
                            "depth_relative_margin_log": float(
                                os.environ.get(
                                    "IMAGINARIUM_LAYOUTVLM_DEPTH_RELATIVE_MARGIN_LOG",
                                    "0.01",
                                )
                            ),
                        }
                    image_gauge_kwargs = {}
                    if image_gauge_observations is not None:
                        image_gauge_kwargs = {
                            "sceneproof_image_observation_indices":
                                image_gauge_observations["indices"],
                            "sceneproof_image_observed_boxes":
                                image_gauge_observations["boxes"],
                            "sceneproof_image_observed_depths":
                                image_gauge_observations["depths"],
                            "sceneproof_image_observed_weights":
                                image_gauge_observations["weights"],
                            "sceneproof_image_bbox_size_enabled":
                                image_gauge_observations["size_enabled"],
                            "sceneproof_image_world_to_camera":
                                image_gauge_observations["world_to_camera"],
                            "sceneproof_image_size":
                                image_gauge_observations["image_size"],
                        }
                    reprojected, optimization_history = optimize_semantic_stage(
                        base_matrices,
                        local_corners,
                        pair_indices,
                        support_pair_tensor,
                        containment_pair_tensor,
                        fixed_support_index_tensor,
                        fixed_support_height_tensor,
                        plane_object_index_tensor,
                        plane_point_tensor,
                        plane_normal_tensor,
                        plane_orientation_tensor,
                        point_pair_tensor,
                        point_offset_tensor,
                        distance_pair_tensor,
                        distance_minimum_tensor,
                        distance_maximum_tensor,
                        align_pair_tensor,
                        align_offset_tensor,
                        footprint_hull_sizes=footprint_hull_sizes,
                        boundary_object_indices=boundary_object_index_tensor,
                        boundary_points=boundary_point_tensor,
                        boundary_normals=boundary_normal_tensor,
                        iterations=optimization_iterations,
                        restore_best_state=(
                            layoutvlm_stage in {"full", "depth"}
                        ),
                        sceneproof_shadow_residual_parity=(
                            os.environ.get(
                                "IMAGINARIUM_SCENEPROOF_SHADOW_RESIDUAL_PARITY",
                                "0",
                            ).strip().lower()
                            in {"1", "true", "yes", "on"}
                        ),
                        sceneproof_use_program_residuals=(
                            os.environ.get(
                                "IMAGINARIUM_SCENEPROOF_USE_PROGRAM_RESIDUALS",
                                "0",
                            ).strip().lower()
                            in {"1", "true", "yes", "on"}
                        ),
                        sceneproof_residual_fallback=(
                            os.environ.get(
                                "IMAGINARIUM_SCENEPROOF_RESIDUAL_FALLBACK",
                                "1",
                            ).strip().lower()
                            in {"1", "true", "yes", "on"}
                        ),
                        sceneproof_factor_bindings=(
                            sceneproof_factor_binding_audit["bindings"]
                            if sceneproof_factor_binding_audit is not None
                            else None
                        ),
                        sceneproof_object_ids=ordered_ids,
                        sceneproof_shadow_jacobian_ownership=(
                            os.environ.get(
                                "IMAGINARIUM_SCENEPROOF_SHADOW_JACOBIAN_OWNERSHIP",
                                "0",
                            ).strip().lower()
                            in {"1", "true", "yes", "on"}
                        ),
                        sceneproof_required_stable_linearizations=int(
                            os.environ.get(
                                "IMAGINARIUM_SCENEPROOF_STABLE_LINEARIZATIONS",
                                "2",
                            )
                        ),
                        sceneproof_full_so3_guarded_schur=(
                            os.environ.get(
                                "IMAGINARIUM_SCENEPROOF_FULL_SO3_GUARDED_SCHUR",
                                "0",
                            ).strip().lower()
                            in {"1", "true", "yes", "on"}
                        ),
                        sceneproof_in_loop_guarded_schur=(
                            os.environ.get(
                                "IMAGINARIUM_SCENEPROOF_IN_LOOP_GUARDED_SCHUR",
                                "0",
                            ).strip().lower()
                            in {"1", "true", "yes", "on"}
                        ),
                        sceneproof_warm_start_anchored_plane_translation=(
                            os.environ.get(
                                "IMAGINARIUM_SCENEPROOF_WARM_START_ANCHORED_PLANE_TRANSLATION",
                                "0",
                            ).strip().lower()
                            in {"1", "true", "yes", "on"}
                        ),
                        sceneproof_plane_anchor_normal_limit_m=float(
                            os.environ.get(
                                "IMAGINARIUM_SCENEPROOF_PLANE_ANCHOR_NORMAL_LIMIT_M",
                                "0.02",
                            )
                        ),
                        sceneproof_plane_proxy_abstain_gap_m=float(
                            os.environ.get(
                                "IMAGINARIUM_SCENEPROOF_PLANE_PROXY_ABSTAIN_GAP_M",
                                "0",
                            )
                        ),
                        sceneproof_plane_attach_requires_witness=(
                            os.environ.get(
                                "IMAGINARIUM_SCENEPROOF_PLANE_ATTACH_REQUIRES_WITNESS",
                                "0",
                            ).strip().lower()
                            in {"1", "true", "yes", "on"}
                        ),
                        sceneproof_plane_sibling_tangent_projection=(
                            os.environ.get(
                                "IMAGINARIUM_SCENEPROOF_PLANE_SIBLING_TANGENT_PROJECTION",
                                "0",
                            ).strip().lower()
                            in {"1", "true", "yes", "on"}
                        ),
                        sceneproof_plane_sibling_max_shift_m=float(
                            os.environ.get(
                                "IMAGINARIUM_SCENEPROOF_PLANE_SIBLING_MAX_SHIFT_M",
                                "0.35",
                            )
                        ),
                        sceneproof_plane_component_image_gauge=(
                            os.environ.get(
                                "IMAGINARIUM_SCENEPROOF_PLANE_COMPONENT_IMAGE_GAUGE",
                                "0",
                            ).strip().lower()
                            in {"1", "true", "yes", "on"}
                        ),
                        **router_kwargs,
                        **solver_kwargs,
                        **depth_kwargs,
                        **image_gauge_kwargs,
                    )
                    stage_summary = (
                        f"support_pairs={len(support_index_pairs)}, "
                        f"fixed_supports={len(fixed_support_indices)}, "
                        f"wall_planes={sum(plane_orientation_mask)}, "
                        f"ceiling_planes="
                        f"{len(plane_orientation_mask) - sum(plane_orientation_mask)}, "
                        f"point_towards={len(semantic_specs['point_pairs'])}, "
                        f"distance={len(semantic_specs['distance_pairs'])}, "
                        f"align_with={len(semantic_specs['align_pairs'])}, "
                        f"containment_pairs={len(containment_index_pairs)}, "
                        f"containment_skipped="
                        f"{len(skipped_containment_pairs)}, "
                        f"semantic_skipped={len(semantic_specs['skipped'])}, "
                        f"projected_max_contact_gap="
                        f"{optimization_history[-1]['projected_max_contact_gap']:.8g}, "
                        f"projected_max_plane_gap="
                        f"{optimization_history[-1]['projected_max_plane_gap']:.8g}, "
                        f"projected_max_containment_error="
                        f"{optimization_history[-1]['projected_max_containment_error']:.8g}, "
                        f"projected_max_collision_penetration="
                        f"{optimization_history[-1]['projected_max_collision_penetration']:.8g}, "
                        f"projected_penetrating_pairs="
                        f"{int(optimization_history[-1]['projected_penetrating_pairs'])}"
                    )
                    if optimization_history[-1].get(
                        "sceneproof_shadow_residual_parity"
                    ):
                        stage_summary += (
                            ", sceneproof_shadow_checks="
                            f"{int(optimization_history[-1]['sceneproof_shadow_residual_checks'])}, "
                            "sceneproof_shadow_max_abs_error="
                            f"{optimization_history[-1]['sceneproof_shadow_residual_max_abs_error']:.8g}, "
                            "sceneproof_program_residual_selections="
                            f"{int(optimization_history[-1]['sceneproof_program_residual_selections'])}, "
                            "sceneproof_residual_fallbacks="
                            f"{int(optimization_history[-1]['sceneproof_residual_fallbacks'])}"
                        )
                    jacobian_ownership = optimization_history[-1].get(
                        "sceneproof_jacobian_ownership"
                    )
                    if jacobian_ownership:
                        final_jacobian = jacobian_ownership["final"]
                        stage_summary += (
                            ", sceneproof_jacobian_checks="
                            f"{jacobian_ownership['checks']}, "
                            "sceneproof_jacobian_max_leakage="
                            f"{jacobian_ownership['maximum_leakage']:.8g}, "
                            "sceneproof_stable_active="
                            f"{len(final_jacobian['stable_active_factor_ids'])}, "
                            "sceneproof_stable_inactive="
                            f"{len(final_jacobian['stable_inactive_factor_ids'])}, "
                            "sceneproof_eligible_leaf_translations="
                            f"{len(final_jacobian['eligible_leaf_translation_objects'])}"
                        )
                    guarded_schur = optimization_history[-1].get(
                        "sceneproof_full_so3_guarded_schur"
                    )
                    if guarded_schur:
                        stage_summary += (
                            ", sceneproof_full_so3_schur_accepted="
                            f"{guarded_schur['accepted']}, "
                            "sceneproof_schur_eliminated="
                            f"{len(guarded_schur.get('schur', {}).get('eliminated_leaf_objects', []))}, "
                            "sceneproof_collision_candidates_checked="
                            f"{guarded_schur.get('collision_guard', {}).get('collision_candidates_checked', 0)}, "
                            "sceneproof_incumbent_restored="
                            f"{guarded_schur['incumbent_restored']}"
                        )
                    containment_abstentions = optimization_history[-1].get(
                        "sceneproof_containment_projection_abstentions", []
                    )
                    if containment_abstentions:
                        stage_summary += (
                            ", sceneproof_containment_abstentions="
                            f"{len(containment_abstentions)}"
                        )
                    if layoutvlm_stage in {"boundary", "full", "depth"}:
                        stage_summary += (
                            f", boundary_floor={boundary_floor_id}, "
                            f"boundary_edges={boundary_point_tensor.shape[0]}, "
                            f"boundary_objects="
                            f"{boundary_object_index_tensor.shape[0]}, "
                            f"projected_max_boundary_error="
                            f"{optimization_history[-1]['projected_max_boundary_error']:.8g}"
                        )
                    if (
                        layoutvlm_stage in {"full", "depth"}
                        and not active_set_router
                        and solver_name == "adam"
                    ):
                        stage_summary += (
                            f", best_iteration="
                            f"{int(optimization_history[-1]['best_iteration'])}, "
                            f"best_total="
                            f"{optimization_history[-1]['best_total']:.8g}"
                        )
                    if solver_name in {"scenelm", "v5_scenelm"}:
                        stage_summary += (
                            f", solver={solver_name}, solver_iterations="
                            f"{int(optimization_history[-1]['solver_executed_iterations'])}, "
                            f"lm_accepted="
                            f"{int(optimization_history[-1]['lm_accepted_steps'])}, "
                            f"lm_rejected="
                            f"{int(optimization_history[-1]['lm_rejected_steps'])}, "
                            f"lm_final_damping="
                            f"{optimization_history[-1]['lm_final_damping']:.8g}, "
                            f"lm_converged="
                            f"{bool(optimization_history[-1]['lm_converged'])}"
                        )
                    if solver_name == "v5_scenelm":
                        relation_audit = optimization_history[-1][
                            "relation_coordinates"
                        ]
                        stage_summary += (
                            f", relation_parameters="
                            f"{relation_audit['parameters']}/"
                            f"{relation_audit['legacy_parameters']}, "
                            f"relation_parameter_reduction="
                            f"{relation_audit['parameter_reduction']:.6f}, "
                            f"relation_support_edges="
                            f"{relation_audit['support_edges']}, "
                            f"relation_schur_leaf_blocks="
                            f"{relation_audit['schur_leaf_blocks']}, "
                            f"relation_active_reduction="
                            f"{optimization_history[-1]['relation_active_reduction']:.6f}, "
                            f"relation_releases="
                            f"{int(optimization_history[-1]['relation_release_count'])}, "
                            f"relation_released_objects="
                            f"{optimization_history[-1]['relation_released_object_indices']}, "
                            f"collision_witnesses="
                            f"{optimization_history[-1]['collision_witness_count']}, "
                            f"certificate_stationarity_inf="
                            f"{optimization_history[-1]['certificate_stationarity_inf']:.8g}, "
                            f"certificate_primal_max="
                            f"{optimization_history[-1]['certificate_primal_max']:.8g}"
                        )
                    if active_set_router:
                        stage_summary += (
                            f", router_budget_counts="
                            f"(30={int(optimization_history[-1]['router_budget_30'])},"
                            f"100={int(optimization_history[-1]['router_budget_100'])},"
                            f"400={int(optimization_history[-1]['router_budget_full'])}), "
                            f"router_iteration_reduction="
                            f"{optimization_history[-1]['router_iteration_reduction']:.6f}, "
                            f"router_wakeups="
                            f"{int(optimization_history[-1]['router_wakeups'])}, "
                            f"router_protected="
                            f"{int(optimization_history[-1]['router_protected_objects'])}"
                        )
                    if layoutvlm_stage == "depth":
                        stage_summary += (
                            f", depth_observations="
                            f"{depth_observations['indices'].numel()}, "
                            f"depth_component_weights="
                            f"(center={depth_kwargs['depth_centre_weight']:.8g},"
                            f"size={depth_kwargs['depth_size_weight']:.8g},"
                            f"metric={depth_kwargs['depth_metric_weight']:.8g}), "
                            f"depth_yaw_optimized="
                            f"{bool(optimization_history[-1]['yaw_optimized'])}, "
                            f"depth_reprojection="
                            f"{optimization_history[-1]['final_depth_reprojection']:.8g}, "
                            f"depth_bbox_center_px="
                            f"{optimization_history[-1]['final_mean_depth_bbox_centre_error_px']:.8g}, "
                            f"depth_relative_error="
                            f"{optimization_history[-1]['final_mean_depth_relative_error']:.8g}, "
                            f"depth_trust_region="
                            f"{optimization_history[-1]['final_depth_trust_region']:.8g}, "
                            f"depth_max_center_excess="
                            f"{optimization_history[-1]['final_max_depth_bbox_centre_excess']:.8g}, "
                            f"depth_max_size_excess="
                            f"{optimization_history[-1]['final_max_depth_bbox_size_excess']:.8g}, "
                            f"depth_max_relative_excess="
                            f"{optimization_history[-1]['final_max_depth_relative_excess']:.8g}"
                        )
            print(
                f"[LayoutVLM] {layoutvlm_stage.title()} stage complete: "
                f"candidate_pairs={len(collision_pairs)}, "
                f"iterations={optimization_iterations}, "
                f"{stage_summary}",
                flush=True,
            )
            for record in optimization_history:
                contact_fields = ""
                if "contact" in record:
                    contact_fields = (
                        f"contact={record['contact']:.8f} "
                        f"max_contact_gap={record['max_contact_gap']:.8f} "
                    )
                plane_fields = ""
                if "plane" in record:
                    plane_fields = (
                        f"plane={record['plane']:.8f} "
                        f"max_plane_gap={record['max_plane_gap']:.8f} "
                        f"orientation={record['orientation']:.8f} "
                        f"max_orientation_error="
                        f"{record['max_orientation_error']:.8f} "
                    )
                semantic_fields = ""
                if "semantic" in record:
                    semantic_fields = (
                        f"containment={record['containment']:.8f} "
                        f"max_containment_error="
                        f"{record['max_containment_error']:.8f} "
                        f"semantic={record['semantic']:.8f} "
                        f"distance={record['distance']:.8f} "
                        f"align={record['align']:.8f} "
                        f"point={record['point']:.8f} "
                        f"boundary={record['boundary']:.8f} "
                        f"max_boundary_error="
                        f"{record['max_boundary_error']:.8f} "
                    )
                depth_fields = ""
                if "depth_reprojection" in record:
                    depth_fields = (
                        f"depth_reprojection="
                        f"{record['depth_reprojection']:.8f} "
                        f"depth_bbox_center_px="
                        f"{record['mean_depth_bbox_centre_error_px']:.4f} "
                        f"depth_bbox_size_log="
                        f"{record['mean_depth_bbox_size_log_error']:.6f} "
                        f"depth_relative_error="
                        f"{record['mean_depth_relative_error']:.6f} "
                        f"depth_trust_region="
                        f"{record['depth_trust_region']:.8f} "
                        f"depth_max_center_excess="
                        f"{record['max_depth_bbox_centre_excess']:.6f} "
                        f"depth_max_size_excess="
                        f"{record['max_depth_bbox_size_excess']:.6f} "
                        f"depth_max_relative_excess="
                        f"{record['max_depth_relative_excess']:.6f} "
                    )
                print(
                    "[LayoutVLM] "
                    f"iter={int(record['iteration'])} "
                    f"total={record['total']:.8f} "
                    f"collision={record['collision']:.8f} "
                    f"{contact_fields}"
                    f"{plane_fields}"
                    f"{semantic_fields}"
                    f"{depth_fields}"
                    f"warm_start={record['warm_start']:.8f} "
                    f"penetrating_pairs={int(record['penetrating_pairs'])}",
                    flush=True,
                )
        with torch.no_grad():
            reprojected_np = reprojected.detach().cpu().numpy()
        layoutvlm_pose_matrices = {
            obj_id: reprojected_np[index].tolist()
            for index, obj_id in enumerate(ordered_ids)
        }
        for index, obj_id in enumerate(ordered_ids):
            translation_delta = (
                reprojected_np[index, :2, 3]
                - base_matrices[index, :2, 3].detach().cpu().numpy()
            )
            original_xy = np.asarray(
                obj_manager.obj_dict[obj_id].original_pos,
                dtype=np.float32,
            )
            obj_manager.obj_dict[obj_id].current_pos = (
                original_xy + translation_delta
            ).tolist()
        final_energy = (
            optimization_history[-1]["total"] if optimization_history else 0.0
        )
        print(
            "[LayoutVLM] Warm-start reprojection passed: "
            f"objects={len(ordered_ids)}, "
            f"max_abs_error={reprojection_error.item():.8g}"
        )
    else:
        print("\n开始模拟退火优化...")
        final_energy = obj_manager.simulated_annealing(
            initial_temp, alpha, max_iterations, penalty_factor
        )
    
    # 输出优化结果
    print("\n" + "="*60)
    print("优化完成!")
    print("="*60)
    
    final_position = {}
    for inst_id, obj in obj_manager.obj_dict.items():
        final_position[inst_id] = {
            "x": float(obj.current_pos[0]),
            "y": float(obj.current_pos[1])
        }
        moved_dist = math.sqrt(
            (obj.current_pos[0] - obj.original_pos[0])**2 + 
            (obj.current_pos[1] - obj.original_pos[1])**2
        )
        print(f"{inst_id} 移动距离: {moved_dist:.4f}")
    
    print(f"\nFinal Overlap: {obj_manager.calc_overlap_area(debug_mode=True)}")
    print(f"Final Constraints: {obj_manager.calc_constraints()}")
    print(f"Final Energy: {final_energy}")
    
    # 保存体素可视化图像（仅在debug模式下）
    if debug:
        voxel_output_path = os.path.join(output_folder, f'{scene_name}_final_voxel_visualization.png')
        save_voxel_debug_img_plt(obj_manager, voxel_output_path)
    
    # 更新Blender场景中的物体位置
    print("\n更新Blender场景...")
    for instance_id, info in output_data_s3["obj_info"].items():
        if instance_id == scene_camera_name or instance_id == ground_name:
            continue
        if (
            use_layoutvlm
            and layoutvlm_pose_matrices is not None
            and instance_id in layoutvlm_pose_matrices
        ):
            obj = bpy.data.objects.get(instance_id)
            if obj:
                refined_pose = layoutvlm_pose_matrices[instance_id]
                info['pose_matrix_for_blender'] = refined_pose
                obj.matrix_world = Matrix(refined_pose)
        elif instance_id in final_position:
            obj = bpy.data.objects.get(instance_id)
            if obj:
                refined_pose = info['pose_matrix_for_blender']
                refined_pose[0][3] = final_position[instance_id]['x']  # 更新X位置
                refined_pose[1][3] = final_position[instance_id]['y']  # 更新Y位置
                
                info['pose_matrix_for_blender'] = refined_pose
                obj.matrix_world = Matrix(refined_pose)
    
    bpy.context.view_layer.update()
    
    # 更新内部摆放物体的位姿
    for instance_id, info in output_data_s3['obj_info'].items():
        if info.get("SpatialRel", None) == "inside":
            parent_name = info['supported']
            
            ori_obj_pose = Matrix(output_data_s2['obj_info'][instance_id]['pose_matrix_for_blender'])
            ori_parent_pose = Matrix(output_data_s2['obj_info'][parent_name]['pose_matrix_for_blender'])
            
            parent = bpy.data.objects.get(parent_name)
            if parent:
                relative_transform = ori_parent_pose.inverted() @ ori_obj_pose
                
                obj = bpy.data.objects.get(instance_id)
                if obj:
                    obj.matrix_world = parent.matrix_world @ relative_transform
    
    bpy.context.view_layer.update()
    
    # 保存最终的位姿信息
    for instance_id, info in output_data_s3['obj_info'].items():
        obj = bpy.data.objects.get(instance_id)
        if obj:
            output_data_s3['obj_info'][instance_id]['pose_matrix_for_blender'] = [
                list(row) for row in blender_manager.get_matrix_world(obj)
            ]

    output_path = os.path.join(output_folder, f'{scene_name}_placement_info_s3.json')
    with open(output_path, 'w') as f:
        json.dump(output_data_s3, f, indent=2)
        
    
    # 开始渲染
    bpy.context.scene.camera = bpy.data.objects[scene_camera_name]
    output_path = os.path.join(output_folder, f'{scene_name}_render_s3.png')
    blender_manager.render_scene(output_path, resolution_x, resolution_y)
    print(f"S3 render_s3 poses saved to: {output_path}", flush=True)

    # Compatible SceneLM support edges define kinematic child-to-parent
    # coordinates. Cache the certified relative transforms before the legacy
    # drop simulation so they can be recovered root-to-leaf afterwards.
    scenelm_kinematic_backsub = (
        use_layoutvlm
        and solver_name == "v5_scenelm"
        and bool(optimization_history)
        and os.environ.get(
            "IMAGINARIUM_SCENELM_KINEMATIC_BACKSUB", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
    )
    scenelm_support_relatives = {}
    scenelm_support_topological_order = []
    if scenelm_kinematic_backsub:
        relation_record = optimization_history[-1].get(
            "relation_coordinates", {}
        )
        certified_pairs = optimization_history[-1].get(
            "post_projection_certified_support_pairs", []
        )
        parent_indices = [-1] * len(ordered_ids)
        for child_index, parent_index in certified_pairs:
            child_index = int(child_index)
            parent_index = int(parent_index)
            if parent_indices[child_index] not in {-1, parent_index}:
                raise RuntimeError(
                    "a certified support child has multiple parents"
                )
            parent_indices[child_index] = parent_index
        remaining = set(range(len(ordered_ids)))
        scenelm_support_topological_order = []
        while remaining:
            ready = sorted(
                index
                for index in remaining
                if parent_indices[index] < 0
                or parent_indices[index] not in remaining
            )
            if not ready:
                raise RuntimeError(
                    "certified SceneLM support graph contains a cycle"
                )
            scenelm_support_topological_order.extend(ready)
            remaining.difference_update(ready)
        optimization_history[-1]["kinematic_parent_indices"] = (
            parent_indices
        )
        optimization_history[-1]["kinematic_promoted_edges"] = sum(
            1
            for child_index, parent_index in enumerate(parent_indices)
            if parent_index >= 0
            and relation_record.get("parent_indices", [])[child_index] < 0
        )
        for child_index, parent_index in enumerate(parent_indices):
            if parent_index < 0:
                continue
            child_obj = bpy.data.objects.get(ordered_ids[child_index])
            parent_obj = bpy.data.objects.get(ordered_ids[parent_index])
            if child_obj is None or parent_obj is None:
                raise RuntimeError(
                    "SceneLM kinematic support object disappeared before simulation"
                )
            scenelm_support_relatives[child_index] = (
                parent_obj.matrix_world.inverted() @ child_obj.matrix_world
            )
        print(
            "[SceneLM] Captured certified kinematic support graph: "
            f"edges={len(scenelm_support_relatives)}",
            flush=True,
        )
    

    # 使用 run_drop_simulation 进行物理仿真
    print("[PhysicsSimulation] 开始使用 run_drop_simulation 进行物理仿真...")
    
    # 收集所有有directlyFacing关系的物体ID
    directly_facing_objects = set()
    for instance_id, info in output_data_s3['obj_info'].items():
        if info.get("directlyFacing"):
            directly_facing_objects.add(instance_id)
            directly_facing_objects.add(info["directlyFacing"])
    
    # 1. 先找出所有inside_objects（内部摆放物体）
    inside_objects = []
    for instance_id, info in output_data_s3['obj_info'].items():
        if instance_id == scene_camera_name:
            continue
        if info.get("SpatialRel", None) == 'inside':
            inside_objects.append(instance_id)
    
    # 2. 找active物体（需要下落模拟的物体）
    active_objects = []
    for instance_id, info in output_data_s3['obj_info'].items():
        if instance_id == scene_camera_name:
            continue
        if instance_id in inside_objects:
            continue

        # 如果父物体是墙体或天花板，则作为passive
        parent_id = info['supported']
        if parent_id is None:
            continue
        if re.match(r'(wall|ceiling)_\d+', parent_id):
            continue
        if re.match(r'(carpet|rug)_\d+', instance_id):
            continue
        
        # 而对于地面的物体需要进一步判断
        is_first_level = re.match(r'(ground|floor|carpet|rug)_\d+', parent_id)
        
        if is_first_level:
            # 一级物体：如果有agentwall或directlyFacing关系，不是active
            if info.get("againstWall") or instance_id in directly_facing_objects:
                continue
            else:
                # 否则作为active候选
                obj = bpy.data.objects[instance_id]
                active_objects.append(obj)
        else:
            # 非一级物体（二级及以上）也作为active候选
            obj = bpy.data.objects[instance_id]
            active_objects.append(obj)
    
    # 3. 剩下的所有物体作为passive物体（排除inside_objects和carpet/rug）
    active_obj_names = set([obj.name for obj in active_objects])
    passive_objects = []
    for instance_id, info in output_data_s3['obj_info'].items():
        if instance_id == scene_camera_name:
            continue
        if instance_id in inside_objects:
            continue
        # 排除carpet/rug物体
        if re.match(r'(carpet|rug)_\d+', instance_id):
            continue
        if instance_id not in active_obj_names:
            obj = bpy.data.objects[instance_id]
            passive_objects.append(obj)
    
    # 4. 检测active物体是否真正浮空：向下移动0.5cm后检测干涉
    # 如果下移后与其他物体有干涉，说明物体接近支撑面，不是真正浮空，转为passive
    print("[PhysicsSimulation] 检测物体是否真正浮空（向下移动0.5cm检测干涉）...")
    
    # 收集所有非active物体（passive + inside）用于碰撞检测
    other_objects = passive_objects.copy()
    for instance_id in inside_objects:
        obj = bpy.data.objects[instance_id]
        other_objects.append(obj)
    
    # 预计算passive/inside物体的bbox信息
    print(f"[PhysicsSimulation] 预计算bbox信息...")
    other_bboxes = {obj: get_bbox_info(obj) for obj in other_objects}
    
    # 检查每个active物体下移0.5cm后是否与其他物体有干涉
    active_to_remove = []
    floating_count = 0
    drop_distance = 0.01  # 0.5cm = 0.01米
    
    for i, active_obj in enumerate(active_objects):
        # 临时将物体向下移动0.5cm
        original_matrix = active_obj.matrix_world.copy()
        temp_matrix = original_matrix.copy()
        temp_matrix[2][3] -= drop_distance
        active_obj.matrix_world = temp_matrix
        
        # 获取下移后的bbox
        active_bbox = get_bbox_info(active_obj)
        
        has_collision = False
        
        # 检测对象列表：passive/inside物体 + 其他active物体
        check_objects = other_objects.copy()
        for j, other_active in enumerate(active_objects):
            if j != i:
                check_objects.append(other_active)
        
        # 第一阶段：bbox快速预筛选
        candidates = []
        for other_obj in check_objects:
            # 获取bbox信息（对于其他active物体需要临时计算）
            if other_obj in other_bboxes:
                other_bbox = other_bboxes[other_obj]
            else:
                other_bbox = get_bbox_info(other_obj)
            
            if check_bbox_overlap_fast(active_bbox, other_bbox):
                candidates.append(other_obj)
        
        # 第二阶段：使用BVH树精确检测
        if candidates:
            for other_obj in candidates:
                if check_mesh_overlap_bvh(active_obj, other_obj):
                    has_collision = True
                    print(f"[PhysicsSimulation] {active_obj.name} 下移0.5cm后与 {other_obj.name} 干涉，判定为非浮空")
                    break
        
        # 恢复物体原始位置
        active_obj.matrix_world = original_matrix
        
        if has_collision:
            # 物体不是真正浮空，转为passive
            active_to_remove.append(active_obj)
        else:
            # 物体是真正浮空的
            floating_count += 1
            print(f"[PhysicsSimulation] {active_obj.name} 是真正浮空物体")
    
    # 将非浮空物体从active转移到passive
    if active_to_remove:
        print(f"[PhysicsSimulation] 将 {len(active_to_remove)} 个非浮空物体转为passive")
        for obj in active_to_remove:
            if obj in active_objects:
                active_objects.remove(obj)
                passive_objects.append(obj)
    
    print(f"[PhysicsSimulation] 共发现 {floating_count} 个真正浮空的物体需要仿真")
    
    duration = 1  # 默认1秒
    print(f"[PhysicsSimulation] 最终Active物体数: {len(active_objects)}, Passive物体数: {len(passive_objects)}, Inside物体数: {len(inside_objects)}")
    
    # 执行物理模拟（使用默认的world_settings）
    success = run_drop_simulation(
        objects=active_objects,
        colliders=passive_objects,
        duration=duration,
        scene=bpy.context.scene
    )
    
    if success:
        print(f"[PhysicsSimulation] 物理仿真完成！")
    else:
        print("[PhysicsSimulation] 物理仿真失败！")
    
    # 更新内部摆放物体的位姿
    print("[PhysicsSimulation] 更新内部摆放物体的位姿...")
    for instance_id in inside_objects:
        info = output_data_s3['obj_info'][instance_id]
        parent_name = info['supported']
        parent = bpy.data.objects.get(parent_name)
        obj = bpy.data.objects.get(instance_id)
        if (
            parent is None
            or obj is None
            or parent_name not in output_data_s2['obj_info']
            or instance_id not in output_data_s2['obj_info']
        ):
            print(
                f"[Warning] Skipping inside-pose restore for {instance_id}: "
                f"missing child or parent object {parent_name}",
                flush=True,
            )
            continue
        # 应该使用s2最终的内部摆放的相对pose
        ori_obj_pose = Matrix(output_data_s2['obj_info'][instance_id]['pose_matrix_for_blender'])
        ori_parent_pose = Matrix(output_data_s2['obj_info'][parent_name]['pose_matrix_for_blender'])
        relative_transform = ori_parent_pose.inverted() @ ori_obj_pose
        obj.matrix_world = parent.matrix_world @ relative_transform
        bpy.context.view_layer.update()
    print("[PhysicsSimulation] 内部摆放物体位姿更新完成。")

    # 仿真之后重做一次支撑面对齐。
    #
    # 为什么必须放在这里而不是 s2 段的 process_z：s2 段的对齐结果会被这次刚体仿真
    # 覆盖。实测证据（bedroom_01，同一输入同一开关跑两次）：位置不可重现的物体恰好
    # 等于被判为"真正浮空"并送进仿真的那批，其余 25 个物体逐位一致，运行间抖动最大
    # 0.38m。也就是说仿真既覆盖了对齐结果，又是唯一的非确定性来源。把对齐放到仿真
    # 之后，这些物体的 z 由父物体顶面这个几何量钉死，抖动随之消失。
    #
    # 为什么值得做：同一场景里 5 个枕头的 gap 约为 -0.45m，即深深扎进床里，pen_0 穿透
    # 桌面 0.77m。这个幅度高于 0.38m 的运行间噪声，是目前唯一信号大于噪声的几何缺陷；
    # 而 s2 段那些2 到 8cm 的小悬空比噪声小一个量级，改了也无法验证。
    if getattr(blender_manager, "_settle_enabled", None) is None:
        blender_manager._settle_enabled, blender_manager._settle_max_gap = (
            resolve_settle_policy()
        )
    if blender_manager._settle_enabled and settle_after_simulation_enabled():
        print("[SettleAfterSim] 仿真后重做支撑面对齐...")
        post_sim_obj_list = {}
        for instance_id in output_data_s3['obj_info']:
            if instance_id == scene_camera_name:
                continue
            if output_data_s3['obj_info'][instance_id].get("SpatialRel") == "inside":
                continue
            obj = bpy.data.objects.get(instance_id)
            if obj is not None:
                post_sim_obj_list[instance_id] = obj
        if ground_name in post_sim_obj_list:
            blender_manager.process_z(ground_name, post_sim_obj_list, tree_sons, 0)
            bpy.context.view_layer.update()
            print("[SettleAfterSim] 完成。")
        else:
            print(
                f"[SettleAfterSim] 跳过：地面 {ground_name} 不在物体表中",
                flush=True,
            )

    if scenelm_kinematic_backsub:
        parent_indices = optimization_history[-1]["kinematic_parent_indices"]
        correction_count = 0
        maximum_correction = 0.0
        expected_backsub_matrices = {}
        backsub_object_ids = []
        for child_index in scenelm_support_topological_order:
            child_index = int(child_index)
            if child_index not in scenelm_support_relatives:
                continue
            parent_index = int(parent_indices[child_index])
            child_obj = bpy.data.objects.get(ordered_ids[child_index])
            parent_obj = bpy.data.objects.get(ordered_ids[parent_index])
            before = child_obj.matrix_world.translation.copy()
            desired_matrix = (
                parent_obj.matrix_world
                @ scenelm_support_relatives[child_index]
            )
            # ACTIVE rigid bodies remain owned by the Bullet point-cache even
            # after matrix_world assignment. Transfer transform ownership back
            # to SceneLM before writing the certified kinematic pose.
            if child_obj.rigid_body is not None:
                child_obj.rigid_body.type = "PASSIVE"
            child_obj.matrix_world = desired_matrix
            expected_backsub_matrices[child_index] = np.asarray(
                desired_matrix, dtype=np.float64
            )
            backsub_object_ids.append(ordered_ids[child_index])
            correction = float(
                (child_obj.matrix_world.translation - before).length
            )
            maximum_correction = max(maximum_correction, correction)
            if correction > 1e-8:
                correction_count += 1
        bpy.context.view_layer.update()
        maximum_realization_error = 0.0
        for child_index, expected_matrix in expected_backsub_matrices.items():
            child_obj = bpy.data.objects.get(ordered_ids[child_index])
            realized_matrix = np.asarray(
                child_obj.matrix_world, dtype=np.float64
            )
            maximum_realization_error = max(
                maximum_realization_error,
                float(np.max(np.abs(realized_matrix - expected_matrix))),
            )
        if maximum_realization_error > 1e-6:
            raise RuntimeError(
                "SceneLM kinematic back-substitution did not acquire "
                "rigid-body transform ownership: "
                f"max_abs_error={maximum_realization_error:.8g}"
            )
        optimization_history[-1]["kinematic_backsub_edges"] = len(
            scenelm_support_relatives
        )
        optimization_history[-1]["kinematic_backsub_corrections"] = (
            correction_count
        )
        optimization_history[-1]["kinematic_backsub_max_correction_m"] = (
            maximum_correction
        )
        optimization_history[-1]["kinematic_backsub_realization_error"] = (
            maximum_realization_error
        )
        optimization_history[-1]["kinematic_backsub_object_ids"] = (
            backsub_object_ids
        )
        print(
            "[SceneLM] Kinematic back-substitution complete: "
            f"edges={len(scenelm_support_relatives)}, "
            f"corrected={correction_count}, "
            f"max_correction_m={maximum_correction:.8g}, "
            f"realization_error={maximum_realization_error:.8g}, "
            f"objects={','.join(backsub_object_ids)}",
            flush=True,
        )

        # A common 3D rigid-body rotation does not preserve containment of
        # vertically separated footprints after projection into world XY.
        # Retract only certified support children back onto the exact frozen
        # S3 support manifold after Bullet and kinematic back-substitution.
        # This is a local, GT-free correction: yaw, Z, scale, and all
        # non-support objects remain unchanged.
        certified_pairs_topological = [
            (child_index, int(parent_indices[child_index]))
            for child_index in scenelm_support_topological_order
            if child_index in scenelm_support_relatives
        ]
        final_pose_matrices = stack_pose_matrices(
            [
                [list(row) for row in bpy.data.objects[obj_id].matrix_world]
                for obj_id in ordered_ids
            ],
            device=device,
        )
        final_support_pairs = pair_index_tensor(
            certified_pairs_topological,
            device=device,
        )
        with torch.no_grad():
            _, pre_retraction_squared_errors = (
                support_planar_containment_loss(
                    final_pose_matrices,
                    local_corners,
                    final_support_pairs,
                    footprint_hull_sizes,
                )
            )
        retraction_tolerance = float(
            os.environ.get(
                "IMAGINARIUM_SCENELM_POSTSIM_RETRACTION_TOLERANCE_M",
                "0.0001",
            )
        )
        maximum_allowed_retraction = float(
            os.environ.get(
                "IMAGINARIUM_SCENELM_POSTSIM_MAX_SHIFT_M", "0.5"
            )
        )
        retraction_passes = int(
            os.environ.get(
                "IMAGINARIUM_SCENELM_POSTSIM_RETRACTION_PASSES", "16"
            )
        )
        working_pose_matrices = final_pose_matrices.clone()
        retracted_indices = []
        retraction_shift_by_object = {}
        # Root-to-leaf, one edge at a time: a repaired parent is finalized
        # before any child edge is considered. This prevents coupled
        # alternating projections from accumulating hierarchy-wide drift.
        for child_index, parent_index in certified_pairs_topological:
            single_pair = pair_index_tensor(
                [(child_index, parent_index)],
                device=device,
            )
            with torch.no_grad():
                _, current_squared_error = support_planar_containment_loss(
                    working_pose_matrices,
                    local_corners,
                    single_pair,
                    footprint_hull_sizes,
                )
                current_error = float(current_squared_error.sqrt().item())
            if current_error <= retraction_tolerance:
                continue
            edge_yaw, edge_translation = initialize_pose_variables(
                working_pose_matrices
            )
            project_support_footprints_(
                edge_yaw,
                edge_translation,
                working_pose_matrices,
                local_corners,
                single_pair,
                passes=retraction_passes,
                footprint_hull_sizes=footprint_hull_sizes,
            )
            candidate_pose_matrices = reproject_pose_matrices(
                working_pose_matrices,
                edge_yaw,
                edge_translation,
            )
            # ``initialize_pose_variables`` stores absolute world
            # translations, not zero-centred deltas.  Gate the realized
            # correction itself; using ``edge_translation`` directly would
            # incorrectly reject an object merely because it is far from the
            # world origin (pen_0 exposed this with a false 3.2587m shift).
            edge_shift = float(
                torch.linalg.vector_norm(
                    candidate_pose_matrices[child_index, :2, 3]
                    - working_pose_matrices[child_index, :2, 3]
                ).item()
            )
            with torch.no_grad():
                _, candidate_squared_error = (
                    support_planar_containment_loss(
                        candidate_pose_matrices,
                        local_corners,
                        single_pair,
                        footprint_hull_sizes,
                    )
                )
                candidate_error = float(
                    candidate_squared_error.sqrt().item()
                )
            if edge_shift > maximum_allowed_retraction:
                raise RuntimeError(
                    "SceneLM post-simulation support retraction exceeded "
                    "its trust region: "
                    f"object={ordered_ids[child_index]}, "
                    f"shift_m={edge_shift:.8g}, "
                    f"limit_m={maximum_allowed_retraction:.8g}"
                )
            if candidate_error > retraction_tolerance:
                raise RuntimeError(
                    "SceneLM post-simulation support retraction failed its "
                    "local certificate: "
                    f"object={ordered_ids[child_index]}, "
                    f"error_m={candidate_error:.8g}"
                )
            working_pose_matrices = candidate_pose_matrices
            retracted_indices.append(child_index)
            retraction_shift_by_object[child_index] = edge_shift
        retracted_pose_matrices = working_pose_matrices
        with torch.no_grad():
            _, post_retraction_squared_errors = (
                support_planar_containment_loss(
                    retracted_pose_matrices,
                    local_corners,
                    final_support_pairs,
                    footprint_hull_sizes,
                )
            )
            maximum_retraction = max(
                retraction_shift_by_object.values(), default=0.0
            )
            pre_retraction_max = float(
                pre_retraction_squared_errors.max().sqrt().item()
                if pre_retraction_squared_errors.numel()
                else 0.0
            )
            post_retraction_max = float(
                post_retraction_squared_errors.max().sqrt().item()
                if post_retraction_squared_errors.numel()
                else 0.0
            )
            retracted_numpy = retracted_pose_matrices.detach().cpu().numpy()
        for object_index in retracted_indices:
            child_obj = bpy.data.objects.get(ordered_ids[object_index])
            if child_obj.rigid_body is not None:
                child_obj.rigid_body.type = "PASSIVE"
            child_obj.matrix_world = Matrix(
                retracted_numpy[object_index].tolist()
            )
        bpy.context.view_layer.update()
        maximum_retraction_realization_error = 0.0
        for object_index in retracted_indices:
            realized_matrix = np.asarray(
                bpy.data.objects[ordered_ids[object_index]].matrix_world,
                dtype=np.float64,
            )
            maximum_retraction_realization_error = max(
                maximum_retraction_realization_error,
                float(
                    np.max(
                        np.abs(
                            realized_matrix
                            - retracted_numpy[object_index].astype(np.float64)
                        )
                    )
                ),
            )
        if maximum_retraction_realization_error > 1e-6:
            raise RuntimeError(
                "SceneLM post-simulation support retraction did not acquire "
                "transform ownership: "
                f"max_abs_error={maximum_retraction_realization_error:.8g}"
            )
        if post_retraction_max > retraction_tolerance:
            raise RuntimeError(
                "SceneLM post-simulation support retraction failed its "
                f"certificate: max_error_m={post_retraction_max:.8g}"
            )
        optimization_history[-1]["postsim_retraction_enabled"] = True
        optimization_history[-1]["postsim_retraction_edges"] = len(
            certified_pairs_topological
        )
        optimization_history[-1]["postsim_retraction_objects"] = len(
            retracted_indices
        )
        optimization_history[-1]["postsim_retraction_object_ids"] = [
            ordered_ids[index] for index in retracted_indices
        ]
        optimization_history[-1]["postsim_retraction_max_shift_m"] = (
            maximum_retraction
        )
        optimization_history[-1]["postsim_retraction_shift_limit_m"] = (
            maximum_allowed_retraction
        )
        optimization_history[-1]["postsim_retraction_tolerance_m"] = (
            retraction_tolerance
        )
        optimization_history[-1]["postsim_retraction_shift_by_object"] = {
            ordered_ids[index]: float(shift)
            for index, shift in retraction_shift_by_object.items()
        }
        optimization_history[-1]["postsim_pre_max_containment_error_m"] = (
            pre_retraction_max
        )
        optimization_history[-1]["postsim_max_containment_error_m"] = (
            post_retraction_max
        )
        optimization_history[-1]["postsim_retraction_realization_error"] = (
            maximum_retraction_realization_error
        )
        print(
            "[SceneLM] Post-simulation support retraction complete: "
            f"edges={len(certified_pairs_topological)}, "
            f"objects={len(retracted_indices)}, "
            f"max_shift_m={maximum_retraction:.8g}, "
            f"shift_limit_m={maximum_allowed_retraction:.8g}, "
            f"pre_max_error_m={pre_retraction_max:.8g}, "
            f"post_max_error_m={post_retraction_max:.8g}, "
            f"realization_error={maximum_retraction_realization_error:.8g}, "
            f"object_ids={','.join(ordered_ids[index] for index in retracted_indices)}",
            flush=True,
        )

    
    # 设置GPU渲染
    bpy.context.scene.cycles.device = 'GPU'

    output_data_s4 = output_data_s3.copy()
    if sceneproof_program_bundle is not None:
        output_data_s4["sceneproof_relation_programs"] = (
            sceneproof_program_bundle
        )
    if sceneproof_live_factor_parity is not None:
        output_data_s4["sceneproof_live_factor_parity"] = (
            sceneproof_live_factor_parity
        )
    if sceneproof_factor_binding_audit is not None:
        output_data_s4["sceneproof_factor_binding_audit"] = (
            sceneproof_factor_binding_audit
        )
    if sceneproof_materialized_incumbent_audit is not None:
        output_data_s4["sceneproof_materialized_incumbent"] = (
            sceneproof_materialized_incumbent_audit
        )
    if (
        optimization_history
        and optimization_history[-1].get(
            "sceneproof_shadow_residual_parity"
        )
    ):
        output_data_s4["sceneproof_shadow_residual_parity"] = {
            "schema_version": "sceneproof_shadow_residual_parity_v1",
            "passed": True,
            "checks": int(
                optimization_history[-1][
                    "sceneproof_shadow_residual_checks"
                ]
            ),
            "max_abs_error": float(
                optimization_history[-1][
                    "sceneproof_shadow_residual_max_abs_error"
                ]
            ),
            "legacy_path_retained": True,
            "program_residual_input": bool(
                optimization_history[-1].get(
                    "sceneproof_program_residual_input", 0
                )
            ),
            "program_residual_selections": int(
                optimization_history[-1].get(
                    "sceneproof_program_residual_selections", 0
                )
            ),
            "fallbacks": int(
                optimization_history[-1].get(
                    "sceneproof_residual_fallbacks", 0
                )
            ),
        }
    if (
        optimization_history
        and optimization_history[-1].get(
            "sceneproof_jacobian_ownership"
        )
    ):
        output_data_s4["sceneproof_jacobian_ownership"] = (
            optimization_history[-1]["sceneproof_jacobian_ownership"]
        )
    if (
        optimization_history
        and optimization_history[-1].get(
            "sceneproof_full_so3_guarded_schur"
        )
    ):
        output_data_s4["sceneproof_full_so3_guarded_schur"] = (
            optimization_history[-1]["sceneproof_full_so3_guarded_schur"]
        )
    for audit_key in (
        "sceneproof_plane_translation_anchor",
        "sceneproof_plane_proxy_abstention",
        "sceneproof_plane_sibling_tangent_projection",
        "sceneproof_plane_component_image_gauge",
        "sceneproof_containment_projection_abstentions",
    ):
        if optimization_history and optimization_history[-1].get(audit_key):
            output_data_s4[audit_key] = optimization_history[-1][audit_key]
    if (
        use_layoutvlm
        and solver_name in {"scenelm", "v5_scenelm"}
        and optimization_history
    ):
        solver_record = optimization_history[-1]
        output_data_s4["scenelm_solver"] = {
            "schema_version": (
                "scenelm_relation_manifold_v1"
                if solver_name == "v5_scenelm"
                else "scenelm_matrix_free_lm_v1"
            ),
            "solver": solver_name,
            "maximum_iterations": int(optimization_iterations),
            "executed_iterations": int(
                solver_record["solver_executed_iterations"]
            ),
            "accepted_steps": int(solver_record["lm_accepted_steps"]),
            "rejected_steps": int(solver_record["lm_rejected_steps"]),
            "final_damping": float(solver_record["lm_final_damping"]),
            "final_residual_energy": float(
                solver_record["lm_final_residual_energy"]
            ),
            "converged": bool(solver_record["lm_converged"]),
            "pcg_iterations": int(
                os.environ.get(
                    "IMAGINARIUM_SCENELM_PCG_ITERATIONS", "12"
                )
            ),
            "gradient_tolerance": float(
                os.environ.get(
                    "IMAGINARIUM_SCENELM_GRADIENT_TOLERANCE",
                    "0.00001",
                )
            ),
            "relative_energy_tolerance": float(
                os.environ.get(
                    "IMAGINARIUM_SCENELM_RELATIVE_ENERGY_TOLERANCE",
                    "0.0001",
                )
            ),
            "stationarity_inf": float(
                solver_record["certificate_stationarity_inf"]
            ),
            "primal_feasibility_max": float(
                solver_record["certificate_primal_max"]
            ),
        }
        if solver_name == "v5_scenelm":
            output_data_s4["scenelm_solver"].update(
                {
                    "relation_coordinates": solver_record[
                        "relation_coordinates"
                    ],
                    "active_step_total": int(
                        solver_record["relation_active_step_total"]
                    ),
                    "dense_step_total": int(
                        solver_record["relation_dense_step_total"]
                    ),
                    "active_reduction": float(
                        solver_record["relation_active_reduction"]
                    ),
                    "freezes": int(solver_record["relation_freezes"]),
                    "wakeups": int(solver_record["relation_wakeups"]),
                    "relation_release_count": int(
                        solver_record["relation_release_count"]
                    ),
                    "relation_released_object_indices": list(
                        solver_record["relation_released_object_indices"]
                    ),
                    "relation_release_iterations": list(
                        solver_record["relation_release_iterations"]
                    ),
                    "collision_witness_count": int(
                        solver_record["collision_witness_count"]
                    ),
                    "collision_witness_weight": float(
                        solver_record["collision_witness_weight"]
                    ),
                    "kinematic_backsub_enabled": bool(
                        scenelm_kinematic_backsub
                    ),
                    "kinematic_backsub_edges": int(
                        solver_record.get("kinematic_backsub_edges", 0)
                    ),
                    "kinematic_backsub_corrections": int(
                        solver_record.get(
                            "kinematic_backsub_corrections", 0
                        )
                    ),
                    "kinematic_backsub_max_correction_m": float(
                        solver_record.get(
                            "kinematic_backsub_max_correction_m", 0.0
                        )
                    ),
                    "kinematic_backsub_realization_error": float(
                        solver_record.get(
                            "kinematic_backsub_realization_error", 0.0
                        )
                    ),
                    "kinematic_backsub_object_ids": list(
                        solver_record.get(
                            "kinematic_backsub_object_ids", []
                        )
                    ),
                    "kinematic_promoted_edges": int(
                        solver_record.get("kinematic_promoted_edges", 0)
                    ),
                    "postsim_retraction_enabled": bool(
                        solver_record.get("postsim_retraction_enabled", False)
                    ),
                    "postsim_retraction_edges": int(
                        solver_record.get("postsim_retraction_edges", 0)
                    ),
                    "postsim_retraction_objects": int(
                        solver_record.get("postsim_retraction_objects", 0)
                    ),
                    "postsim_retraction_object_ids": list(
                        solver_record.get("postsim_retraction_object_ids", [])
                    ),
                    "postsim_retraction_max_shift_m": float(
                        solver_record.get("postsim_retraction_max_shift_m", 0.0)
                    ),
                    "postsim_retraction_shift_limit_m": float(
                        solver_record.get(
                            "postsim_retraction_shift_limit_m", 0.0
                        )
                    ),
                    "postsim_retraction_tolerance_m": float(
                        solver_record.get(
                            "postsim_retraction_tolerance_m", 0.0
                        )
                    ),
                    "postsim_retraction_shift_by_object": dict(
                        solver_record.get(
                            "postsim_retraction_shift_by_object", {}
                        )
                    ),
                    "postsim_pre_max_containment_error_m": float(
                        solver_record.get(
                            "postsim_pre_max_containment_error_m", 0.0
                        )
                    ),
                    "postsim_max_containment_error_m": float(
                        solver_record.get(
                            "postsim_max_containment_error_m", 0.0
                        )
                    ),
                    "postsim_retraction_realization_error": float(
                        solver_record.get(
                            "postsim_retraction_realization_error", 0.0
                        )
                    ),
                }
            )
    if (
        use_layoutvlm
        and active_set_router
        and optimization_history
        and "router_allocated_budgets" in optimization_history[-1]
    ):
        router_record = optimization_history[-1]
        output_data_s4["layoutvlm_active_set_router"] = {
            "enabled": True,
            "checkpoints": list(checkpoint_values),
            "iterations": int(optimization_iterations),
            "budget_by_object": {
                object_id: int(budget)
                for object_id, budget in zip(
                    ordered_ids,
                    router_record["router_allocated_budgets"],
                )
            },
            "constraint_degree_by_object": {
                object_id: int(degree)
                for object_id, degree in zip(
                    ordered_ids,
                    router_record["router_constraint_degree"],
                )
            },
            "active_step_total": int(
                router_record["router_active_step_total"]
            ),
            "dense_step_total": int(
                router_record["router_dense_step_total"]
            ),
            "iteration_reduction": float(
                router_record["router_iteration_reduction"]
            ),
            "wakeups": int(router_record["router_wakeups"]),
            "protected_objects": int(
                router_record["router_protected_objects"]
            ),
        }
    # Persist every placement-owned Blender root, not merely objects which
    # still have a rigid body. ``run_drop_simulation`` deliberately removes
    # ACTIVE rigid bodies after baking their visual transforms.  Filtering on
    # ``obj.rigid_body`` therefore kept the pre-simulation JSON pose while the
    # immediately following beauty render used the settled Blender pose.
    serialized_pose_matrices = {}
    missing_serialization_objects = []
    without_rigid_body = []
    for object_id, info in output_data_s4.get("obj_info", {}).items():
        obj = bpy.data.objects.get(object_id)
        if obj is None:
            missing_serialization_objects.append(object_id)
            continue
        pose_array = np.asarray(obj.matrix_world, dtype=np.float64)
        if pose_array.shape != (4, 4) or not np.isfinite(pose_array).all():
            raise RuntimeError(
                "SceneProof cannot serialize a non-finite Blender pose: "
                f"object={object_id}, shape={pose_array.shape}"
            )
        pose = pose_array.tolist()
        info["pose_matrix_for_blender"] = pose
        serialized_pose_matrices[object_id] = pose_array.copy()
        if obj.rigid_body is None:
            without_rigid_body.append(object_id)
        print(f"Object: {object_id}\n World Pose: {pose}")
    output_data_s4["sceneproof_pose_serialization"] = {
        "schema_version": "sceneproof_pose_serialization_v1",
        "policy": "all_placement_owned_blender_roots",
        "placement_records": len(output_data_s4.get("obj_info", {})),
        "serialized_objects": len(serialized_pose_matrices),
        "serialized_without_rigid_body": len(without_rigid_body),
        "serialized_without_rigid_body_ids": sorted(without_rigid_body),
        "missing_objects": len(missing_serialization_objects),
        "missing_object_ids": sorted(missing_serialization_objects),
    }
    print(
        "[SceneProof] Pose serialization ownership: "
        f"serialized={len(serialized_pose_matrices)}, "
        f"without_rigid_body={len(without_rigid_body)}, "
        f"missing={len(missing_serialization_objects)}, "
        "policy=all_placement_owned_blender_roots",
        flush=True,
    )

    if (
        os.environ.get(
            "IMAGINARIUM_SCENEPROOF_MESH_VISIBILITY_AUDIT", "0"
        ).strip().lower()
        in {"1", "true", "yes", "on"}
    ):
        visible_side_candidate_audit = (
            os.environ.get(
                "IMAGINARIUM_SCENEPROOF_VISIBLE_SIDE_CANDIDATE_AUDIT", "0"
            ).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        tangent_candidate_audit = (
            os.environ.get(
                "IMAGINARIUM_SCENEPROOF_TANGENT_CANDIDATE_AUDIT", "0"
            ).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        joint_tangent_candidate_audit = (
            os.environ.get(
                "IMAGINARIUM_SCENEPROOF_JOINT_TANGENT_CANDIDATE_AUDIT",
                "0",
            ).strip().lower()
            in {"1", "true", "yes", "on"}
        )
        visibility_pose_matrices = stack_pose_matrices(
            [
                [list(row) for row in bpy.data.objects[obj_id].matrix_world]
                for obj_id in ordered_ids
            ],
            device=device,
        )
        mesh_visibility_audit = audit_sceneproof_mesh_visibility(
            obj_placement_info_json_path,
            ordered_ids,
            bpy.data.objects.get(scene_camera_name),
            output_data_s4.get(
                "sceneproof_plane_sibling_tangent_projection", {}
            ),
            plane_bindings=plane_bindings,
            pose_matrices=visibility_pose_matrices,
            local_corners=local_corners,
            collision_pairs=pair_indices,
            footprint_hull_sizes=footprint_hull_sizes,
            support_pairs=support_pair_tensor,
            fixed_support_indices=fixed_support_index_tensor,
            fixed_support_heights=fixed_support_height_tensor,
            containment_pairs=containment_pair_tensor,
            boundary_object_indices=boundary_object_index_tensor,
            boundary_points=boundary_point_tensor,
            boundary_normals=boundary_normal_tensor,
            visible_side_candidate_audit=visible_side_candidate_audit,
            tangent_candidate_audit=tangent_candidate_audit,
            joint_tangent_candidate_audit=joint_tangent_candidate_audit,
            resolution=int(
                os.environ.get(
                    "IMAGINARIUM_SCENEPROOF_VISIBILITY_RESOLUTION", "256"
                )
            ),
            minimum_pixels=int(
                os.environ.get(
                    "IMAGINARIUM_SCENEPROOF_VISIBILITY_MIN_PIXELS", "64"
                )
            ),
            visible_side_clearance_m=float(
                os.environ.get(
                    "IMAGINARIUM_SCENEPROOF_VISIBLE_SIDE_CLEARANCE_M",
                    "0.002",
                )
            ),
            visible_side_maximum_shift_m=float(
                os.environ.get(
                    "IMAGINARIUM_SCENEPROOF_VISIBLE_SIDE_MAX_SHIFT_M",
                    "0.15",
                )
            ),
            visible_side_attachment_tolerance_m=float(
                os.environ.get(
                    "IMAGINARIUM_SCENEPROOF_VISIBLE_SIDE_ATTACHMENT_TOLERANCE_M",
                    "0.005",
                )
            ),
            tangent_maximum_shift_m=float(
                os.environ.get(
                    "IMAGINARIUM_SCENEPROOF_TANGENT_MAX_SHIFT_M",
                    "0.35",
                )
            ),
            visibility_noharm_tolerance=float(
                os.environ.get(
                    "IMAGINARIUM_SCENEPROOF_VISIBILITY_NOHARM_TOLERANCE",
                    "0.005",
                )
            ),
            minimum_recall_gain=float(
                os.environ.get(
                    "IMAGINARIUM_SCENEPROOF_VISIBLE_SIDE_MIN_RECALL_GAIN",
                    "0.05",
                )
            ),
            physical_tolerance=float(
                os.environ.get(
                    "IMAGINARIUM_SCENEPROOF_VISIBLE_SIDE_PHYSICAL_TOLERANCE",
                    "0.000001",
                )
            ),
        )
        output_data_s4["sceneproof_mesh_visibility_audit"] = (
            mesh_visibility_audit
        )

    # Determine output paths
    save_path = os.path.join(output_folder, f'{scene_name}_placement_info_s4.json')
    output_path = os.path.join(output_folder, f'{scene_name}_render_simu.png')
    
    # 将仿真数据写入 JSON 文件
    with open(save_path, "w") as json_file:
        json.dump(output_data_s4, json_file, indent=2)

    print(f"仿真数据已保存到: {save_path}")
    
    # 开始渲染（使用 Cycles 渲染器以获得高质量结果）
    bpy.context.scene.camera = bpy.data.objects[scene_camera_name]
    blender_manager.render_scene(output_path, resolution_x, resolution_y, samples=256)

    # Rendering must not own or mutate placement transforms.  This catches a
    # residual Bullet/depsgraph state leak immediately instead of publishing a
    # PNG which cannot be reconstructed from the adjacent placement JSON.
    post_render_max_pose_delta = 0.0
    post_render_changed_objects = []
    for object_id, serialized_pose in serialized_pose_matrices.items():
        obj = bpy.data.objects.get(object_id)
        if obj is None:
            post_render_changed_objects.append(object_id)
            post_render_max_pose_delta = float("inf")
            continue
        realized = np.asarray(obj.matrix_world, dtype=np.float64)
        delta = float(np.max(np.abs(realized - serialized_pose)))
        post_render_max_pose_delta = max(post_render_max_pose_delta, delta)
        if delta > 1e-6:
            post_render_changed_objects.append(object_id)
    if post_render_changed_objects:
        raise RuntimeError(
            "SceneProof rendering changed serialized object poses: "
            f"max_abs_delta={post_render_max_pose_delta:.8g}, "
            "objects=" + ",".join(post_render_changed_objects[:20])
        )
    print(
        "[SceneProof] Pose serialization/render parity: "
        f"objects={len(serialized_pose_matrices)}, "
        f"max_abs_delta={post_render_max_pose_delta:.8g}, passed=True",
        flush=True,
    )
    
    print(f"Final poses saved to: {output_path}", flush=True)

    
def main(args):
    # 从配置中读取基础路径
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    params = config["S4_blender_layout_and_corr"]

    placeable_area_info_folder = params['placeable_area_info_folder']
    base_fbx_path = params['base_fbx_path']
    fbx_csv_path = params['fbx_csv_path']
    precomputed_voxel_dir = params.get('precomputed_voxel_dir', None)
    
    obj_placement_info_json_path = args.obj_placement_info_json_path
    output_folder = args.output_folder
    debug = args.debug
    layout(
        obj_placement_info_json_path,
        placeable_area_info_folder,
        base_fbx_path,
        fbx_csv_path,
        output_folder,
        precomputed_voxel_dir,
        debug,
        args.use_layoutvlm,
        args.layoutvlm_stage,
    )

argv = sys.argv
if "--" not in argv:
    argv = []
else:
   argv = argv[argv.index("--") + 1:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = '3D Layout Processing Script',
        prog = "blender -b -python "+__file__+" --",
        )
    parser.add_argument('--obj_placement_info_json_path', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder for S4 results')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to save voxel visualization images')
    parser.add_argument(
        '--use_layoutvlm',
        action='store_true',
        help='Use the gated v4 LayoutVLM optimizer instead of legacy SA',
    )
    parser.add_argument(
        '--layoutvlm_stage',
        default='reproject',
        choices=LAYOUTVLM_STAGES,
        help='Incremental LayoutVLM implementation stage',
    )
    
    try:
        args = parser.parse_args(argv)
        main(args)
    except SystemExit as e:
        print(repr(e))

'''
blender --background --python S4_blender_layout_and_corr.py -- --obj_placement_info_json_path "saved_results/demo_result/S3_pose_inference/demo_placement_info.json"
'''
