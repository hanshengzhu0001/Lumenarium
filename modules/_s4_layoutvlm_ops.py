"""Differentiable pose operations for the v4 LayoutVLM optimizer.

This module deliberately has no Blender dependency.  The first v4 milestone
uses it to round-trip the deterministic S4-S2 warm start through a
differentiable representation before any optimization loss is enabled.

The optimized yaw is represented as a *delta* around the Blender world Z axis.
That preserves the full warm-start linear transform (including fixed scale and
any residual upright correction) instead of decomposing and rebuilding it.
"""

from __future__ import annotations

import math
import os
from typing import Any, Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F

from modules._s4_scenelm_relational import (
    compile_full_so3_relation_coordinates,
    compile_relation_coordinates,
)
from modules._sceneproof_residual_bridge import (
    assemble_program_shadow_residuals,
    build_residual_slice_bindings,
    residual_parity,
)
from modules._sceneproof_block_system import (
    LinearizedFactor,
    LinearizationStabilityTracker,
    ResidualSliceBinding,
    assemble_normal_system,
    audit_jacobian_block_ownership,
    certify_projected_stationarity,
    guarded_collision_trial,
    positive_spanning_poll_steps,
    restrict_normal_system_to_parameter_mask,
    rollback_object_parameter_blocks,
    solve_with_leaf_translation_schur,
)


def matrix_free_pcg(
    operator,
    right_hand_side: torch.Tensor,
    *,
    maximum_iterations: int = 12,
    relative_tolerance: float = 1e-3,
    diagonal_preconditioner: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """Solve a positive-definite linear system with matrix-free PCG.

    ``operator`` must evaluate ``A @ vector`` without materializing ``A``.
    SceneLM uses this for damped normal equations, whose dense matrix would
    destroy the sparsity and memory advantages of the scene factor graph.
    """
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    if right_hand_side.ndim != 1:
        raise ValueError("right_hand_side must be one-dimensional")
    if diagonal_preconditioner is not None:
        if diagonal_preconditioner.shape != right_hand_side.shape:
            raise ValueError(
                "diagonal_preconditioner must match right_hand_side"
            )
        if torch.any(diagonal_preconditioner <= 0):
            raise ValueError("diagonal_preconditioner must be positive")

    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    if diagonal_preconditioner is None:
        preconditioned = residual
    else:
        preconditioned = residual / diagonal_preconditioner
    direction = preconditioned.clone()
    residual_dot = torch.dot(residual, preconditioned)
    initial_norm = torch.linalg.vector_norm(residual)
    target = relative_tolerance * torch.clamp(initial_norm, min=1e-12)
    iterations = 0

    for iteration in range(maximum_iterations):
        applied = operator(direction)
        denominator = torch.dot(direction, applied).clamp_min(1e-20)
        step = residual_dot / denominator
        solution = solution + step * direction
        residual = residual - step * applied
        iterations = iteration + 1
        if torch.linalg.vector_norm(residual) <= target:
            break
        if diagonal_preconditioner is None:
            next_preconditioned = residual
        else:
            next_preconditioned = residual / diagonal_preconditioner
        next_residual_dot = torch.dot(residual, next_preconditioned)
        direction = (
            next_preconditioned
            + (next_residual_dot / residual_dot.clamp_min(1e-20))
            * direction
        )
        preconditioned = next_preconditioned
        residual_dot = next_residual_dot

    final_norm = torch.linalg.vector_norm(residual)
    return solution, {
        "iterations": float(iterations),
        "initial_residual_norm": float(initial_norm.detach().item()),
        "final_residual_norm": float(final_norm.detach().item()),
        "relative_residual": float(
            (final_norm / torch.clamp(initial_norm, min=1e-12))
            .detach()
            .item()
        ),
    }


def matrix_free_lm_step(
    residual_function,
    parameters: torch.Tensor,
    *,
    damping: float,
    pcg_iterations: int = 12,
    pcg_tolerance: float = 1e-3,
    acceptance_threshold: float = 0.1,
    diagonal_preconditioner: torch.Tensor | None = None,
    step_transform=None,
    parameter_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """Compute one matrix-free Gauss--Newton/Levenberg--Marquardt step.

    The normal-equation product is evaluated as ``J.T @ (J @ v) + lambda*v``
    through JVP/VJP transforms.  No dense Jacobian or Hessian is constructed.
    The returned candidate is accepted only when its measured reduction agrees
    with the local quadratic model.
    """
    if parameters.ndim != 1:
        raise ValueError("parameters must be one-dimensional")
    if damping <= 0:
        raise ValueError("damping must be positive")
    if not 0 <= acceptance_threshold < 1:
        raise ValueError("acceptance_threshold must lie in [0, 1)")
    if not hasattr(torch, "func"):
        raise RuntimeError("SceneLM requires torch.func.jvp/vjp")
    if parameter_mask is not None:
        if parameter_mask.shape != parameters.shape:
            raise ValueError("parameter_mask must match parameters")
        parameter_mask = parameter_mask.to(
            device=parameters.device,
            dtype=torch.bool,
        )

    current = parameters.detach()
    residuals, vjp_function = torch.func.vjp(residual_function, current)
    if residuals.ndim != 1:
        raise ValueError("residual_function must return a one-dimensional tensor")
    gradient = vjp_function(residuals)[0]
    if parameter_mask is not None:
        gradient = torch.where(parameter_mask, gradient, 0.0)

    def normal_operator(vector: torch.Tensor) -> torch.Tensor:
        if parameter_mask is not None:
            active_vector = torch.where(parameter_mask, vector, 0.0)
        else:
            active_vector = vector
        _, jacobian_vector = torch.func.jvp(
            residual_function,
            (current,),
            (active_vector,),
        )
        applied = vjp_function(jacobian_vector)[0] + damping * active_vector
        if parameter_mask is not None:
            # Frozen coordinates form an identity block with a zero RHS. This
            # keeps PCG positive definite while excluding them from all JVP /
            # VJP work and from the proposed update.
            return torch.where(parameter_mask, applied, vector)
        return applied

    direction, pcg = matrix_free_pcg(
        normal_operator,
        -gradient,
        maximum_iterations=pcg_iterations,
        relative_tolerance=pcg_tolerance,
        diagonal_preconditioner=diagonal_preconditioner,
    )
    if step_transform is not None:
        direction = step_transform(direction)
        if direction.shape != current.shape:
            raise ValueError("step_transform must preserve the parameter shape")
    if parameter_mask is not None:
        direction = torch.where(parameter_mask, direction, 0.0)
    applied_direction = normal_operator(direction)
    predicted_reduction = -(
        torch.dot(gradient, direction)
        + 0.5 * torch.dot(direction, applied_direction)
    )
    candidate = current + direction
    with torch.no_grad():
        candidate_residuals = residual_function(candidate)
        current_energy = 0.5 * residuals.square().sum()
        candidate_energy = 0.5 * candidate_residuals.square().sum()
        actual_reduction = current_energy - candidate_energy
        ratio = actual_reduction / torch.clamp(
            predicted_reduction, min=1e-20
        )
        accepted = bool(
            torch.isfinite(candidate_energy)
            and actual_reduction > 0
            and ratio >= acceptance_threshold
        )
    next_parameters = candidate if accepted else current
    diagnostics = {
        **pcg,
        "accepted": float(accepted),
        "damping": float(damping),
        "energy": float(current_energy.detach().item()),
        "candidate_energy": float(candidate_energy.detach().item()),
        "actual_reduction": float(actual_reduction.detach().item()),
        "predicted_reduction": float(predicted_reduction.detach().item()),
        "reduction_ratio": float(ratio.detach().item()),
        "gradient_inf": float(
            gradient.detach().abs().amax().item()
            if gradient.numel()
            else 0.0
        ),
        "step_norm": float(
            torch.linalg.vector_norm(direction).detach().item()
        ),
    }
    return next_parameters.detach(), diagnostics


def _validate_pose_batch(base_matrices: torch.Tensor) -> None:
    if base_matrices.ndim != 3 or base_matrices.shape[-2:] != (4, 4):
        raise ValueError(
            "base_matrices must have shape (N, 4, 4); "
            f"got {tuple(base_matrices.shape)}"
        )
    if not base_matrices.is_floating_point():
        raise TypeError("base_matrices must use a floating-point dtype")


def yaw_rotation_matrices(yaw_delta: torch.Tensor) -> torch.Tensor:
    """Return batched world-Z rotation matrices for ``yaw_delta`` radians."""
    if yaw_delta.ndim != 1:
        raise ValueError(
            f"yaw_delta must have shape (N,); got {tuple(yaw_delta.shape)}"
        )

    cos_yaw = torch.cos(yaw_delta)
    sin_yaw = torch.sin(yaw_delta)
    zeros = torch.zeros_like(yaw_delta)
    ones = torch.ones_like(yaw_delta)

    row0 = torch.stack((cos_yaw, -sin_yaw, zeros), dim=-1)
    row1 = torch.stack((sin_yaw, cos_yaw, zeros), dim=-1)
    row2 = torch.stack((zeros, zeros, ones), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def reproject_pose_matrices(
    base_matrices: torch.Tensor,
    yaw_delta: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Rebuild world matrices from fixed warm-start bases and pose variables.

    Args:
        base_matrices: ``(N, 4, 4)`` deterministic S4-S2 world matrices.
        yaw_delta: ``(N,)`` optimizable world-Z yaw offsets in radians.
        translation: ``(N, 3)`` optimizable absolute world translations.

    Returns:
        Differentiable ``(N, 4, 4)`` world matrices.
    """
    _validate_pose_batch(base_matrices)
    count = base_matrices.shape[0]
    if yaw_delta.shape != (count,):
        raise ValueError(
            f"yaw_delta must have shape ({count},); got {tuple(yaw_delta.shape)}"
        )
    if translation.shape != (count, 3):
        raise ValueError(
            "translation must have shape "
            f"({count}, 3); got {tuple(translation.shape)}"
        )
    if yaw_delta.device != base_matrices.device:
        raise ValueError("yaw_delta and base_matrices must be on the same device")
    if translation.device != base_matrices.device:
        raise ValueError("translation and base_matrices must be on the same device")

    yaw_rotation = yaw_rotation_matrices(yaw_delta)
    linear = torch.matmul(yaw_rotation, base_matrices[:, :3, :3])

    upper = torch.cat((linear, translation.unsqueeze(-1)), dim=-1)
    homogeneous_row = base_matrices[:, 3:4, :]
    return torch.cat((upper, homogeneous_row), dim=-2)


def local_box_corners(
    minimum: torch.Tensor,
    maximum: torch.Tensor,
) -> torch.Tensor:
    """Build the eight local AABB corners for a batch of objects."""
    if minimum.ndim != 2 or minimum.shape[-1] != 3:
        raise ValueError("minimum must have shape (N, 3)")
    if maximum.shape != minimum.shape:
        raise ValueError("maximum must have the same shape as minimum")

    selector = minimum.new_tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    return (
        minimum[:, None, :] * (1.0 - selector[None, :, :])
        + maximum[:, None, :] * selector[None, :, :]
    )


def transform_points(
    pose_matrices: torch.Tensor,
    local_points: torch.Tensor,
) -> torch.Tensor:
    """Transform batched local points into world space."""
    _validate_pose_batch(pose_matrices)
    if local_points.ndim != 3 or local_points.shape[0] != pose_matrices.shape[0]:
        raise ValueError("local_points must have shape (N, P, 3)")
    if local_points.shape[-1] != 3:
        raise ValueError("local_points must have shape (N, P, 3)")

    linear = pose_matrices[:, :3, :3]
    translation = pose_matrices[:, :3, 3]
    return torch.matmul(local_points, linear.transpose(1, 2)) + translation[:, None, :]


def _normalized_planar_axes(pose_matrices: torch.Tensor) -> torch.Tensor:
    """Return each upright box's local X/Y axes projected into world XY."""
    linear = pose_matrices[:, :2, :2]
    axes = linear.transpose(1, 2)
    lengths = torch.linalg.vector_norm(axes, dim=-1, keepdim=True).clamp_min(1e-8)
    return axes / lengths


def oriented_penetration_loss(
    pose_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    pair_indices: torch.Tensor,
    footprint_hull_sizes: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable OBB penetration surrogate based on the separating axes.

    The four planar axes of every pair are tested, while Z overlap is computed
    independently.  A separated pair contributes exactly zero.  An intersecting
    pair contributes its shallowest planar penetration multiplied by its Z
    penetration.  This is a stable pure-PyTorch fallback while the optional
    Rotated_IoU CUDA extension is validated on A10.

    Returns:
        ``(mean_loss, per_pair_penetration)``.
    """
    _validate_pose_batch(pose_matrices)
    if local_corners.ndim != 3 or local_corners.shape != (
        pose_matrices.shape[0],
        8,
        3,
    ):
        raise ValueError("local_corners must have shape (N, 8, 3)")
    if pair_indices.ndim != 2 or pair_indices.shape[-1] != 2:
        raise ValueError("pair_indices must have shape (P, 2)")
    if pair_indices.dtype != torch.long:
        raise TypeError("pair_indices must use torch.long")
    if pair_indices.device != pose_matrices.device:
        raise ValueError("pair_indices and pose_matrices must be on the same device")

    if pair_indices.shape[0] == 0:
        zero = pose_matrices.sum() * 0.0
        return zero, pose_matrices.new_zeros((0,))

    world_corners = transform_points(pose_matrices, local_corners)
    first = pair_indices[:, 0]
    second = pair_indices[:, 1]
    corners_first = world_corners[first]
    corners_second = world_corners[second]

    if footprint_hull_sizes is None:
        axes = _normalized_planar_axes(pose_matrices)
        pair_axes = torch.cat((axes[first], axes[second]), dim=1)
        pair_axis_mask = torch.ones(
            pair_axes.shape[:2],
            dtype=torch.bool,
            device=pose_matrices.device,
        )
    else:
        if footprint_hull_sizes.shape != (pose_matrices.shape[0],):
            raise ValueError("footprint_hull_sizes must have shape (N,)")
        if footprint_hull_sizes.dtype != torch.long:
            raise TypeError("footprint_hull_sizes must use torch.long")
        if footprint_hull_sizes.device != pose_matrices.device:
            raise ValueError(
                "footprint_hull_sizes and pose_matrices must share a device"
            )
        if bool(
            ((footprint_hull_sizes < 3) | (footprint_hull_sizes > 8))
            .any()
            .item()
        ):
            raise ValueError(
                "footprint_hull_sizes must contain values in [3, 8]"
            )

        polygons = world_corners[:, :, :2]
        slots = torch.arange(8, device=pose_matrices.device)[None, :]

        def polygon_axes(indices: torch.Tensor):
            polygon = polygons[indices]
            sizes = footprint_hull_sizes[indices]
            mask = slots < sizes[:, None]
            next_slots = torch.where(
                slots + 1 < sizes[:, None], slots + 1, 0
            ).expand(polygon.shape[0], -1)
            following = torch.gather(
                polygon,
                1,
                next_slots[:, :, None].expand(-1, -1, 2),
            )
            edges = following - polygon
            normals = torch.stack((-edges[..., 1], edges[..., 0]), dim=-1)
            normals = normals / torch.linalg.vector_norm(
                normals, dim=-1, keepdim=True
            ).clamp_min(1e-8)
            return normals, mask

        first_axes, first_mask = polygon_axes(first)
        second_axes, second_mask = polygon_axes(second)
        pair_axes = torch.cat((first_axes, second_axes), dim=1)
        pair_axis_mask = torch.cat((first_mask, second_mask), dim=1)
    projection_first = torch.einsum(
        "mpd,mad->mpa", corners_first[:, :, :2], pair_axes
    )
    projection_second = torch.einsum(
        "mpd,mad->mpa", corners_second[:, :, :2], pair_axes
    )
    planar_overlap = (
        torch.minimum(
            projection_first.amax(dim=1),
            projection_second.amax(dim=1),
        )
        - torch.maximum(
            projection_first.amin(dim=1),
            projection_second.amin(dim=1),
        )
    )
    planar_overlap = torch.where(
        pair_axis_mask,
        planar_overlap,
        torch.full_like(planar_overlap, torch.inf),
    )
    planar_penetration = torch.relu(planar_overlap).amin(dim=1)

    z_overlap = (
        torch.minimum(
            corners_first[:, :, 2].amax(dim=1),
            corners_second[:, :, 2].amax(dim=1),
        )
        - torch.maximum(
            corners_first[:, :, 2].amin(dim=1),
            corners_second[:, :, 2].amin(dim=1),
        )
    )
    z_penetration = torch.relu(z_overlap)
    per_pair = planar_penetration * z_penetration
    return per_pair.mean(), per_pair


def collision_separation_witness_axes(
    pose_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    pair_indices: torch.Tensor,
    footprint_hull_sizes: torch.Tensor,
) -> torch.Tensor:
    """Freeze the minimum-translation separating disjunction per pair.

    Collision non-overlap is a disjunction over polygon edge normals.  Once a
    pair is known to penetrate, choosing its cheapest separating direction
    converts that local disjunction into a stable scalar half-space witness.
    The direction is detached and may then be optimized without the
    min-axis switching that stalls a Gauss--Newton/LM linearization.
    """
    _validate_pose_batch(pose_matrices)
    if pair_indices.ndim != 2 or pair_indices.shape[-1] != 2:
        raise ValueError("pair_indices must have shape (P, 2)")
    if footprint_hull_sizes.shape != (pose_matrices.shape[0],):
        raise ValueError("footprint_hull_sizes must have shape (N,)")
    if pair_indices.shape[0] == 0:
        return pose_matrices.new_zeros((0, 2))
    world = transform_points(pose_matrices, local_corners)
    witnesses: list[torch.Tensor] = []
    for first_value, second_value in pair_indices.detach().cpu().tolist():
        first = int(first_value)
        second = int(second_value)
        first_size = int(footprint_hull_sizes[first].detach().cpu().item())
        second_size = int(footprint_hull_sizes[second].detach().cpu().item())
        first_polygon = world[first, :first_size, :2]
        second_polygon = world[second, :second_size, :2]
        axes: list[torch.Tensor] = []
        for polygon in (first_polygon, second_polygon):
            edges = torch.roll(polygon, shifts=-1, dims=0) - polygon
            normals = torch.stack((-edges[:, 1], edges[:, 0]), dim=1)
            normals = normals / torch.linalg.vector_norm(
                normals, dim=1, keepdim=True
            ).clamp_min(1.0e-8)
            axes.append(normals)
        candidate_axes = torch.cat(axes, dim=0)
        first_projection = first_polygon @ candidate_axes.T
        second_projection = second_polygon @ candidate_axes.T
        move_positive = (
            second_projection.amax(dim=0) - first_projection.amin(dim=0)
        )
        move_negative = (
            first_projection.amax(dim=0) - second_projection.amin(dim=0)
        )
        signed_axes = torch.cat((candidate_axes, -candidate_axes), dim=0)
        required = torch.cat((move_positive, move_negative), dim=0)
        witnesses.append(signed_axes[torch.argmin(required)])
    return torch.stack(witnesses).detach()


def collision_witness_residuals(
    pose_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    pair_indices: torch.Tensor,
    witness_axes: torch.Tensor,
    footprint_hull_sizes: torch.Tensor,
    *,
    margin: float = 1.0e-4,
) -> torch.Tensor:
    """Evaluate fixed separating half-space witnesses in metres."""
    _validate_pose_batch(pose_matrices)
    if pair_indices.shape != (witness_axes.shape[0], 2):
        raise ValueError("pair_indices and witness_axes must align")
    if witness_axes.ndim != 2 or witness_axes.shape[-1] != 2:
        raise ValueError("witness_axes must have shape (P, 2)")
    if footprint_hull_sizes.shape != (pose_matrices.shape[0],):
        raise ValueError("footprint_hull_sizes must have shape (N,)")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if pair_indices.shape[0] == 0:
        return pose_matrices.new_zeros((0,))
    world = transform_points(pose_matrices, local_corners)[..., :2]
    first = world[pair_indices[:, 0]]
    second = world[pair_indices[:, 1]]
    first_projection = torch.einsum("mpd,md->mp", first, witness_axes)
    second_projection = torch.einsum("mpd,md->mp", second, witness_axes)
    slots = torch.arange(8, device=pose_matrices.device)[None, :]
    first_mask = slots < footprint_hull_sizes[pair_indices[:, 0], None]
    second_mask = slots < footprint_hull_sizes[pair_indices[:, 1], None]
    first_minimum = torch.where(
        first_mask,
        first_projection,
        torch.full_like(first_projection, torch.inf),
    ).amin(dim=1)
    second_maximum = torch.where(
        second_mask,
        second_projection,
        torch.full_like(second_projection, -torch.inf),
    ).amax(dim=1)
    return torch.relu(
        second_maximum - first_minimum + margin
    )


def warm_start_regularization(
    yaw_delta: torch.Tensor,
    translation: torch.Tensor,
    base_matrices: torch.Tensor,
) -> torch.Tensor:
    """Small drift penalty around deterministic S4-S2 initialization."""
    _validate_pose_batch(base_matrices)
    translation_delta = translation - base_matrices[:, :3, 3]
    return yaw_delta.square().mean() + translation_delta.square().mean()


def depth_aware_reprojection_loss(
    pose_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    observation_indices: torch.Tensor,
    observed_boxes: torch.Tensor,
    observed_depths: torch.Tensor,
    observed_weights: torch.Tensor,
    bbox_size_enabled: torch.Tensor,
    world_to_camera: torch.Tensor,
    image_size: torch.Tensor,
    *,
    centre_weight: float = 1.0,
    size_weight: float = 0.25,
    metric_depth_weight: float = 1.0,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Anchor optimized OBBs to S1 masks and S0 metric depth.

    Blender cameras look along local ``-Z`` with ``+Y`` pointing up.  The
    supplied world-to-camera matrix therefore maps an OBB corner ``p`` to
    pixels as ``u = fx*x/-z + cx`` and ``v = cy - fy*y/-z``.

    The loss is deliberately robust and scale-free. Its three components
    have independent weights so image-plane and metric-depth supervision can
    be ablated without changing the observation set:

    * projected/observed bbox centers are compared in normalized pixels;
    * bbox sizes are compared in log space (disabled for truncated masks);
    * camera-forward depth is compared in log space.

    ``observed_depths`` are per-mask robust depths in metres.  The projected
    OBB centre is used instead of a front corner so that the target remains
    stable under yaw.  Upstream observation construction compensates the
    visible-surface-to-centre offset at the deterministic warm start.
    """
    _validate_pose_batch(pose_matrices)
    component_weights = {
        "centre_weight": centre_weight,
        "size_weight": size_weight,
        "metric_depth_weight": metric_depth_weight,
    }
    for name, value in component_weights.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    count = observation_indices.numel()
    if observation_indices.ndim != 1 or observation_indices.dtype != torch.long:
        raise ValueError("observation_indices must be a 1-D torch.long tensor")
    if observed_boxes.shape != (count, 4):
        raise ValueError("observed_boxes must have shape (D, 4)")
    for name, value in (
        ("observed_depths", observed_depths),
        ("observed_weights", observed_weights),
        ("bbox_size_enabled", bbox_size_enabled),
    ):
        if value.shape != (count,):
            raise ValueError(f"{name} must have shape (D,)")
    if world_to_camera.shape != (4, 4):
        raise ValueError("world_to_camera must have shape (4, 4)")
    if image_size.shape != (4,):
        raise ValueError("image_size must contain [width, height, fx, fy]")
    if count == 0:
        zero = pose_matrices.sum() * 0.0
        empty = pose_matrices.new_empty((0,))
        return zero, empty, empty, empty

    corners_world = transform_points(
        pose_matrices[observation_indices],
        local_corners[observation_indices],
    )
    rotation = world_to_camera[:3, :3]
    offset = world_to_camera[:3, 3]
    corners_camera = torch.einsum(
        "ij,dcj->dci",
        rotation,
        corners_world,
    ) + offset[None, None, :]

    width, height, focal_x, focal_y = image_size
    principal_x = width * 0.5
    principal_y = height * 0.5
    forward_depth = torch.clamp(-corners_camera[:, :, 2], min=1e-3)
    projected_x = focal_x * corners_camera[:, :, 0] / forward_depth + principal_x
    projected_y = principal_y - focal_y * corners_camera[:, :, 1] / forward_depth
    projected_boxes = torch.stack(
        (
            projected_x.amin(dim=1),
            projected_y.amin(dim=1),
            projected_x.amax(dim=1),
            projected_y.amax(dim=1),
        ),
        dim=1,
    )

    predicted_centres = 0.5 * (
        projected_boxes[:, :2] + projected_boxes[:, 2:]
    )
    observed_centres = 0.5 * (
        observed_boxes[:, :2] + observed_boxes[:, 2:]
    )
    pixel_scale = torch.stack((width, height))
    centre_delta = (predicted_centres - observed_centres) / pixel_scale
    centre_loss = F.smooth_l1_loss(
        centre_delta,
        torch.zeros_like(centre_delta),
        beta=0.02,
        reduction="none",
    ).mean(dim=1)

    predicted_sizes = torch.clamp(
        projected_boxes[:, 2:] - projected_boxes[:, :2],
        min=1.0,
    )
    observed_sizes = torch.clamp(
        observed_boxes[:, 2:] - observed_boxes[:, :2],
        min=1.0,
    )
    size_delta = torch.log(predicted_sizes / observed_sizes)
    size_loss = F.smooth_l1_loss(
        size_delta,
        torch.zeros_like(size_delta),
        beta=0.1,
        reduction="none",
    ).mean(dim=1)
    size_loss = size_loss * bbox_size_enabled.to(size_loss.dtype)

    predicted_depths = forward_depth.mean(dim=1)
    depth_delta = torch.log(
        torch.clamp(predicted_depths, min=1e-3)
        / torch.clamp(observed_depths, min=1e-3)
    )
    depth_loss = F.smooth_l1_loss(
        depth_delta,
        torch.zeros_like(depth_delta),
        beta=0.1,
        reduction="none",
    )

    weights = torch.clamp(observed_weights, min=0.0)
    denominator = torch.clamp(weights.sum(), min=1e-6)
    total = (
        weights
        * (
            centre_weight * centre_loss
            + size_weight * size_loss
            + metric_depth_weight * depth_loss
        )
    ).sum() / denominator
    centre_error_pixels = torch.linalg.vector_norm(
        predicted_centres - observed_centres,
        dim=1,
    )
    return (
        total,
        centre_error_pixels,
        size_delta.abs().mean(dim=1),
        depth_delta.abs(),
    )


def select_confident_discrete_pose_repairs(
    base_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    observation_indices: torch.Tensor,
    observed_boxes: torch.Tensor,
    observed_depths: torch.Tensor,
    bbox_size_enabled: torch.Tensor,
    world_to_camera: torch.Tensor,
    image_size: torch.Tensor,
    lock_world_z: torch.Tensor,
    support_parent_indices: torch.Tensor | None = None,
    collision_exempt_mask: torch.Tensor | None = None,
    *,
    visible_surface_depths: torch.Tensor | None = None,
    surface_to_center_offsets: torch.Tensor | None = None,
    enable_asset_center_candidates: bool = False,
    asset_center_offset_scales: Tuple[float, ...] = (0.5, 1.0, 1.5),
    enable_support_surface_candidates: bool = False,
    yaw_offsets_deg: Tuple[float, ...] = (0.0, 90.0, 180.0, 270.0),
    max_translation_m: float = 0.5,
    minimum_relative_improvement: float = 0.08,
    minimum_absolute_improvement: float = 1e-3,
    minimum_runner_up_margin: float = 2e-4,
    centre_weight: float = 1.0,
    size_weight: float = 0.25,
    metric_depth_weight: float = 1.0,
    yaw_prior_weight: float = 5e-4,
    translation_prior_weight: float = 2e-3,
    collision_increase_weight: float = 2.0,
    max_collision_increase_m3: float = 1e-3,
    max_support_degradation_m: float = 0.05,
) -> Tuple[torch.Tensor, list[dict]]:
    """Conservatively revise discrete yaw and metric translation anchors.

    This is a proposal selector, not another unconstrained optimizer.  Each
    observed object receives a small Cartesian product of:

    * world-Z yaw modes supplied by ``yaw_offsets_deg``;
    * current translation;
    * image-centre correction at the current depth;
    * metric-depth correction along the current camera ray;
    * joint observed-centre + metric-depth backprojection;
    * optional retrieved-asset surface-to-centre offset hypotheses;
    * optional support-aware joint hypotheses projected into the parent's
      oriented footprint.

    A non-current hypothesis is accepted only when it beats both the warm
    start and its runner-up by explicit margins.  Translation is bounded, and
    supported objects can lock world Z so image evidence cannot make them
    float.  The downstream contact/collision/plane optimizer remains the final
    safety layer.
    """
    _validate_pose_batch(base_matrices)
    count = observation_indices.numel()
    if lock_world_z.shape != (base_matrices.shape[0],):
        raise ValueError("lock_world_z must have shape (N,)")
    object_count = base_matrices.shape[0]
    if support_parent_indices is None:
        support_parent_indices = torch.full(
            (object_count,), -1, dtype=torch.long, device=base_matrices.device
        )
    if support_parent_indices.shape != (object_count,):
        raise ValueError("support_parent_indices must have shape (N,)")
    if collision_exempt_mask is None:
        collision_exempt_mask = torch.eye(
            object_count, dtype=torch.bool, device=base_matrices.device
        )
    if collision_exempt_mask.shape != (object_count, object_count):
        raise ValueError("collision_exempt_mask must have shape (N, N)")
    if count == 0:
        return base_matrices.clone(), []
    if max_translation_m <= 0:
        raise ValueError("max_translation_m must be positive")
    if not yaw_offsets_deg or 0.0 not in tuple(float(x) for x in yaw_offsets_deg):
        raise ValueError("yaw_offsets_deg must contain the zero/current mode")
    if visible_surface_depths is None:
        visible_surface_depths = observed_depths
    if surface_to_center_offsets is None:
        surface_to_center_offsets = torch.zeros_like(observed_depths)
    if visible_surface_depths.shape != (count,):
        raise ValueError("visible_surface_depths must have shape (M,)")
    if surface_to_center_offsets.shape != (count,):
        raise ValueError("surface_to_center_offsets must have shape (M,)")
    offset_scales = tuple(float(value) for value in asset_center_offset_scales)
    if enable_asset_center_candidates and (
        not offset_scales or any(value < 0.0 for value in offset_scales)
    ):
        raise ValueError(
            "asset_center_offset_scales must contain non-negative values"
        )

    width, height, focal_x, focal_y = image_size
    principal_x = width * 0.5
    principal_y = height * 0.5
    camera_rotation = world_to_camera[:3, :3]
    repaired = base_matrices.clone()
    reports: list[dict] = []
    all_world_corners = transform_points(base_matrices, local_corners)
    all_minimum = all_world_corners.amin(dim=1)
    all_maximum = all_world_corners.amax(dim=1)

    def aabb_overlap_sum(
        candidate_world: torch.Tensor, object_index: int
    ) -> torch.Tensor:
        candidate_minimum = candidate_world.amin(dim=0)
        candidate_maximum = candidate_world.amax(dim=0)
        overlap = torch.clamp(
            torch.minimum(candidate_maximum[None, :], all_maximum)
            - torch.maximum(candidate_minimum[None, :], all_minimum),
            min=0.0,
        )
        volume = overlap.prod(dim=1)
        volume = torch.where(
            collision_exempt_mask[object_index],
            torch.zeros_like(volume),
            volume,
        )
        return volume.sum()

    def footprint_error(candidate_world: torch.Tensor, object_index: int) -> torch.Tensor:
        parent_index = int(support_parent_indices[object_index].item())
        if parent_index < 0:
            return candidate_world.new_zeros(())
        child_minimum = candidate_world[:, :2].amin(dim=0)
        child_maximum = candidate_world[:, :2].amax(dim=0)
        parent_minimum = all_minimum[parent_index, :2]
        parent_maximum = all_maximum[parent_index, :2]
        lower = torch.relu(parent_minimum - child_minimum)
        upper = torch.relu(child_maximum - parent_maximum)
        return torch.maximum(lower, upper).amax()

    def project_candidate_into_support(
        pose: torch.Tensor,
        corners: torch.Tensor,
        object_index: int,
    ) -> Tuple[torch.Tensor | None, torch.Tensor]:
        """Project a candidate OBB footprint into its oriented parent OBB.

        Unlike the legacy support gate, this uses the complete child
        footprint rather than only its centre.  Candidates whose footprint is
        larger than the parent along either parent axis are infeasible and
        are omitted instead of being silently clamped.
        """
        parent_index = int(support_parent_indices[object_index].item())
        zero = pose.new_zeros(())
        if parent_index < 0:
            return None, zero
        parent_pose = base_matrices[parent_index : parent_index + 1]
        parent_center = parent_pose[0, :2, 3]
        parent_axes = _normalized_planar_axes(parent_pose)[0]
        parent_corners = all_world_corners[parent_index, :, :2]
        parent_projection = torch.einsum(
            "pd,ad->pa",
            parent_corners - parent_center,
            parent_axes,
        )
        parent_minimum = parent_projection.amin(dim=0)
        parent_maximum = parent_projection.amax(dim=0)

        candidate_world = (
            torch.matmul(corners, pose[:3, :3].transpose(0, 1))
            + pose[:3, 3]
        )
        child_projection = torch.einsum(
            "pd,ad->pa",
            candidate_world[:, :2] - parent_center,
            parent_axes,
        )
        child_minimum = child_projection.amin(dim=0)
        child_maximum = child_projection.amax(dim=0)
        lower_correction = parent_minimum - child_minimum
        upper_correction = parent_maximum - child_maximum
        if bool(torch.any(lower_correction > upper_correction + 1e-8).item()):
            return None, zero
        correction_axes = torch.maximum(
            lower_correction,
            torch.minimum(upper_correction, torch.zeros_like(lower_correction)),
        )
        correction_world = torch.einsum(
            "a,ad->d", correction_axes, parent_axes
        )
        if torch.linalg.vector_norm(correction_world).item() <= 1e-8:
            return None, zero
        projected = pose.clone()
        projected[:2, 3] = projected[:2, 3] + correction_world
        return projected, torch.linalg.vector_norm(correction_world)

    def candidate_errors(
        pose: torch.Tensor,
        corners: torch.Tensor,
        box: torch.Tensor,
        observed_depth: torch.Tensor,
        size_enabled: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        world = torch.matmul(corners, pose[:3, :3].transpose(0, 1)) + pose[:3, 3]
        camera = torch.matmul(world, camera_rotation.transpose(0, 1))
        camera = camera + world_to_camera[:3, 3]
        forward = torch.clamp(-camera[:, 2], min=1e-3)
        projected_x = focal_x * camera[:, 0] / forward + principal_x
        projected_y = principal_y - focal_y * camera[:, 1] / forward
        projected_box = torch.stack(
            (
                projected_x.amin(),
                projected_y.amin(),
                projected_x.amax(),
                projected_y.amax(),
            )
        )
        centre = 0.5 * (projected_box[:2] + projected_box[2:])
        observed_centre = 0.5 * (box[:2] + box[2:])
        centre_error = torch.linalg.vector_norm(centre - observed_centre)
        projected_size = torch.clamp(projected_box[2:] - projected_box[:2], min=1.0)
        observed_size = torch.clamp(box[2:] - box[:2], min=1.0)
        size_error = torch.log(projected_size / observed_size).abs().mean()
        size_error = size_error * size_enabled.to(size_error.dtype)
        depth_error = torch.log(
            torch.clamp(forward.mean(), min=1e-3)
            / torch.clamp(observed_depth, min=1e-3)
        ).abs()
        normalized = (
            centre_weight * centre_error / torch.maximum(width, height)
            + size_weight * size_error
            + metric_depth_weight * depth_error
        )
        return normalized, centre_error, size_error, depth_error

    for observation_offset, object_index_tensor in enumerate(observation_indices):
        object_index = int(object_index_tensor.item())
        base = base_matrices[object_index]
        corners = local_corners[object_index]
        observed_box = observed_boxes[observation_offset]
        observed_depth = observed_depths[observation_offset]
        visible_surface_depth = visible_surface_depths[observation_offset]
        surface_to_center_offset = surface_to_center_offsets[
            observation_offset
        ]
        size_enabled = bbox_size_enabled[observation_offset]
        current_world = all_world_corners[object_index]
        current_collision = aabb_overlap_sum(current_world, object_index)
        current_support_error = footprint_error(current_world, object_index)

        world_corners = (
            torch.matmul(corners, base[:3, :3].transpose(0, 1))
            + base[:3, 3]
        )
        world_center = world_corners.mean(dim=0)
        camera_center = (
            torch.matmul(camera_rotation, world_center)
            + world_to_camera[:3, 3]
        )
        current_depth = torch.clamp(-camera_center[2], min=1e-3)
        observed_centre = 0.5 * (observed_box[:2] + observed_box[2:])

        image_camera = camera_center.clone()
        image_camera[0] = (
            (observed_centre[0] - principal_x) * current_depth / focal_x
        )
        image_camera[1] = (
            (principal_y - observed_centre[1]) * current_depth / focal_y
        )
        ray_scale = observed_depth / current_depth
        depth_camera = camera_center.clone()
        depth_camera[:2] = depth_camera[:2] * ray_scale
        depth_camera[2] = -observed_depth
        joint_camera = camera_center.clone()
        joint_camera[0] = (
            (observed_centre[0] - principal_x) * observed_depth / focal_x
        )
        joint_camera[1] = (
            (principal_y - observed_centre[1]) * observed_depth / focal_y
        )
        joint_camera[2] = -observed_depth

        anchor_cameras = [
            ("current", camera_center, False),
            ("image_centre", image_camera, False),
            ("depth_ray", depth_camera, False),
            ("joint_backproject", joint_camera, False),
        ]
        asset_center_cameras = []
        if enable_asset_center_candidates:
            for offset_scale in offset_scales:
                if abs(offset_scale - 1.0) <= 1e-8:
                    continue
                target_depth = torch.clamp(
                    visible_surface_depth
                    + offset_scale * surface_to_center_offset,
                    min=1e-3,
                )
                target_camera = camera_center.clone()
                target_camera[0] = (
                    (observed_centre[0] - principal_x)
                    * target_depth
                    / focal_x
                )
                target_camera[1] = (
                    (principal_y - observed_centre[1])
                    * target_depth
                    / focal_y
                )
                target_camera[2] = -target_depth
                scale_tag = int(round(offset_scale * 100.0))
                asset_center_cameras.append(
                    (
                        f"asset_center_s{scale_tag:03d}",
                        target_camera,
                        False,
                    )
                )
        anchor_cameras.extend(asset_center_cameras)
        if (
            enable_support_surface_candidates
            and int(support_parent_indices[object_index].item()) >= 0
        ):
            anchor_cameras.append(
                ("support_surface_joint", joint_camera, True)
            )
            anchor_cameras.extend(
                (
                    f"support_surface_{name}",
                    target_camera,
                    True,
                )
                for name, target_camera, _ in asset_center_cameras
            )
        candidates = []
        for anchor_name, target_camera, project_support in anchor_cameras:
            delta_camera = target_camera - camera_center
            delta_world = torch.matmul(camera_rotation.transpose(0, 1), delta_camera)
            if bool(lock_world_z[object_index].item()):
                delta_world = delta_world.clone()
                delta_world[2] = 0.0
            translation = base[:3, 3] + delta_world
            for yaw_deg in yaw_offsets_deg:
                yaw = base.new_tensor([math.radians(float(yaw_deg))])
                pose = reproject_pose_matrices(
                    base.unsqueeze(0),
                    yaw,
                    translation.unsqueeze(0),
                )[0]
                support_projection_m = pose.new_zeros(())
                if project_support:
                    pose, support_projection_m = project_candidate_into_support(
                        pose,
                        corners,
                        object_index,
                    )
                    if pose is None:
                        continue
                shift = torch.linalg.vector_norm(
                    pose[:3, 3] - base[:3, 3]
                )
                if shift.item() > max_translation_m + 1e-8:
                    continue
                candidate_world = (
                    torch.matmul(corners, pose[:3, :3].transpose(0, 1))
                    + pose[:3, 3]
                )
                collision = aabb_overlap_sum(candidate_world, object_index)
                collision_increase = torch.relu(collision - current_collision)
                support_error = footprint_error(candidate_world, object_index)
                support_degradation = torch.relu(
                    support_error - current_support_error
                )
                if (
                    collision_increase.item() > max_collision_increase_m3
                    or support_degradation.item() > max_support_degradation_m
                ):
                    continue
                score, centre_error, size_error, depth_error = candidate_errors(
                    pose, corners, observed_box, observed_depth, size_enabled
                )
                yaw_turns = min(
                    abs(float(yaw_deg)) % 360.0,
                    360.0 - abs(float(yaw_deg)) % 360.0,
                ) / 90.0
                prior = (
                    yaw_prior_weight * yaw_turns
                    + translation_prior_weight * (shift / max_translation_m).square()
                    + collision_increase_weight * collision_increase
                )
                candidates.append(
                    {
                        "pose": pose,
                        "score": float((score + prior).item()),
                        "data_score": float(score.item()),
                        "centre_error_px": float(centre_error.item()),
                        "size_error_log": float(size_error.item()),
                        "depth_error_log": float(depth_error.item()),
                        "anchor": anchor_name,
                        "yaw_deg": float(yaw_deg),
                        "translation_shift_m": float(shift.item()),
                        "support_projection_m": float(
                            support_projection_m.item()
                        ),
                        "surface_to_center_offset_m": float(
                            surface_to_center_offset.item()
                        ),
                        "collision_increase_m3": float(
                            collision_increase.item()
                        ),
                        "support_degradation_m": float(
                            support_degradation.item()
                        ),
                    }
                )

        candidates.sort(key=lambda item: item["score"])
        current = next(
            item
            for item in candidates
            if item["anchor"] == "current" and item["yaw_deg"] == 0.0
        )
        best = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else current
        absolute_improvement = current["score"] - best["score"]
        relative_improvement = absolute_improvement / max(current["score"], 1e-8)
        runner_up_margin = runner_up["score"] - best["score"]
        changed = best is not current
        accepted = bool(
            changed
            and absolute_improvement >= minimum_absolute_improvement
            and relative_improvement >= minimum_relative_improvement
            and runner_up_margin >= minimum_runner_up_margin
        )
        if accepted:
            repaired[object_index] = best["pose"]
        serialized_candidates = []
        for rank, candidate in enumerate(candidates, start=1):
            serialized_candidates.append(
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "pose"
                }
                | {
                    "rank": rank,
                    "selected_by_verifier": bool(
                        accepted and candidate is best
                    ),
                    "pose_matrix_for_blender": candidate[
                        "pose"
                    ].detach().cpu().tolist(),
                }
            )
        reports.append(
            {
                "object_index": object_index,
                "accepted": accepted,
                "candidate_count": len(candidates),
                "current_score": current["score"],
                "best_score": best["score"],
                "absolute_improvement": absolute_improvement,
                "relative_improvement": relative_improvement,
                "runner_up_margin": runner_up_margin,
                "selected_anchor": best["anchor"] if accepted else "current",
                "selected_yaw_deg": best["yaw_deg"] if accepted else 0.0,
                "selected_translation_shift_m": (
                    best["translation_shift_m"] if accepted else 0.0
                ),
                "best_centre_error_px": best["centre_error_px"],
                "best_size_error_log": best["size_error_log"],
                "best_depth_error_log": best["depth_error_log"],
                "best_collision_increase_m3": best[
                    "collision_increase_m3"
                ],
                "best_support_degradation_m": best[
                    "support_degradation_m"
                ],
                "candidates": serialized_candidates,
            }
        )
    return repaired, reports


def no_harm_reprojection_penalty(
    centre_errors: torch.Tensor,
    size_errors: torch.Tensor,
    depth_errors: torch.Tensor,
    reference_centre_errors: torch.Tensor,
    reference_size_errors: torch.Tensor,
    reference_depth_errors: torch.Tensor,
    observation_weights: torch.Tensor,
    *,
    centre_margin_pixels: float = 2.0,
    size_margin_log: float = 0.02,
    depth_margin_log: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize per-object reprojection degradation beyond small margins.

    The absolute depth objective may trade a modest 2D error for a better
    metric-depth fit.  This trust region permits only explicitly bounded
    trades relative to the deterministic S4 warm start; violations are
    normalized by their margin so one badly drifting object cannot disappear
    inside a scene-wide mean.
    """
    expected_shape = centre_errors.shape
    for name, value in (
        ("size_errors", size_errors),
        ("depth_errors", depth_errors),
        ("reference_centre_errors", reference_centre_errors),
        ("reference_size_errors", reference_size_errors),
        ("reference_depth_errors", reference_depth_errors),
        ("observation_weights", observation_weights),
    ):
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {tuple(expected_shape)}"
            )
    if centre_margin_pixels <= 0 or size_margin_log <= 0 or depth_margin_log <= 0:
        raise ValueError("no-harm trust-region margins must be positive")
    if centre_errors.numel() == 0:
        zero = centre_errors.sum() * 0.0
        return zero, centre_errors, size_errors, depth_errors

    centre_excess = torch.relu(
        centre_errors - reference_centre_errors - centre_margin_pixels
    ) / centre_margin_pixels
    size_excess = torch.relu(
        size_errors - reference_size_errors - size_margin_log
    ) / size_margin_log
    depth_excess = torch.relu(
        depth_errors - reference_depth_errors - depth_margin_log
    ) / depth_margin_log
    per_object = (
        centre_excess.square()
        + 0.25 * size_excess.square()
        + depth_excess.square()
    )
    weights = torch.clamp(observation_weights, min=0.0)
    denominator = torch.clamp(weights.sum(), min=1e-6)
    penalty = (weights * per_object).sum() / denominator
    return penalty, centre_excess, size_excess, depth_excess


def support_contact_loss(
    pose_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    support_pairs: torch.Tensor,
    fixed_support_indices: torch.Tensor | None = None,
    fixed_support_heights: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Squared vertical gap for child-parent and child-fixed-surface contacts."""
    _validate_pose_batch(pose_matrices)
    if support_pairs.ndim != 2 or support_pairs.shape[-1] != 2:
        raise ValueError("support_pairs must have shape (S, 2)")
    if support_pairs.dtype != torch.long:
        raise TypeError("support_pairs must use torch.long")
    if support_pairs.device != pose_matrices.device:
        raise ValueError("support_pairs and pose_matrices must share a device")

    if fixed_support_indices is None:
        fixed_support_indices = torch.empty(
            (0,), dtype=torch.long, device=pose_matrices.device
        )
    if fixed_support_heights is None:
        fixed_support_heights = pose_matrices.new_empty((0,))
    if fixed_support_indices.ndim != 1:
        raise ValueError("fixed_support_indices must have shape (F,)")
    if fixed_support_heights.shape != fixed_support_indices.shape:
        raise ValueError(
            "fixed_support_heights must match fixed_support_indices"
        )

    world_corners = transform_points(pose_matrices, local_corners)
    gaps = []
    if support_pairs.shape[0]:
        child = support_pairs[:, 0]
        parent = support_pairs[:, 1]
        child_bottom = world_corners[child, :, 2].amin(dim=1)
        parent_top = world_corners[parent, :, 2].amax(dim=1)
        gaps.append(child_bottom - parent_top)
    if fixed_support_indices.shape[0]:
        fixed_bottom = world_corners[
            fixed_support_indices, :, 2
        ].amin(dim=1)
        gaps.append(fixed_bottom - fixed_support_heights)

    if not gaps:
        zero = pose_matrices.sum() * 0.0
        return zero, pose_matrices.new_zeros((0,))
    all_gaps = torch.cat(gaps)
    return all_gaps.square().mean(), all_gaps


def project_support_contacts_(
    yaw_delta: torch.Tensor,
    translation: torch.Tensor,
    base_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    support_pairs: torch.Tensor,
    fixed_support_indices: torch.Tensor | None = None,
    fixed_support_heights: torch.Tensor | None = None,
    *,
    passes: int = 3,
) -> None:
    """PGD projection that places every child bottom on its support surface."""
    if passes <= 0:
        raise ValueError("passes must be positive")
    if fixed_support_indices is None:
        fixed_support_indices = torch.empty(
            (0,), dtype=torch.long, device=translation.device
        )
    if fixed_support_heights is None:
        fixed_support_heights = translation.new_empty((0,))

    with torch.no_grad():
        for _ in range(passes):
            poses = reproject_pose_matrices(
                base_matrices,
                yaw_delta,
                translation,
            )
            world_corners = transform_points(poses, local_corners)
            if support_pairs.shape[0]:
                for child_index, parent_index in support_pairs.tolist():
                    child_bottom = world_corners[
                        child_index, :, 2
                    ].amin()
                    parent_top = world_corners[
                        parent_index, :, 2
                    ].amax()
                    translation[child_index, 2].add_(
                        parent_top - child_bottom
                    )
            if fixed_support_indices.shape[0]:
                for child_index, support_height in zip(
                    fixed_support_indices.tolist(),
                    fixed_support_heights.tolist(),
                ):
                    poses = reproject_pose_matrices(
                        base_matrices,
                        yaw_delta,
                        translation,
                    )
                    child_bottom = transform_points(
                        poses[child_index : child_index + 1],
                        local_corners[child_index : child_index + 1],
                    )[0, :, 2].amin()
                    translation[child_index, 2].add_(
                        translation.new_tensor(support_height) - child_bottom
                    )


def fixed_plane_loss(
    pose_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    object_indices: torch.Tensor,
    plane_points: torch.Tensor,
    plane_normals: torch.Tensor,
    orientation_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return contact and planar-orientation losses for fixed scene planes.

    Every normal must point from the plane surface toward the constrained
    object.  The signed contact gap is therefore the minimum signed distance
    of the object's world-space box corners to that plane.  A positive value
    means a gap, while a negative value means penetration.

    ``orientation_mask`` is normally true for walls and false for ceilings.
    SceneProof can opt into a frozen-geometry thin-axis convention with
    ``IMAGINARIUM_SCENEPROOF_THIN_AXIS_ATTACH_RATIO``.  When the smaller
    horizontal extent is below that fraction of the larger one, only the thin
    (attachment-facing) axis may align with the wall normal.  This prevents
    curtains, mirrors, and frames from satisfying the factor while edge-on.
    """
    _validate_pose_batch(pose_matrices)
    count = object_indices.shape[0]
    if object_indices.ndim != 1 or object_indices.dtype != torch.long:
        raise ValueError("object_indices must be a one-dimensional long tensor")
    if plane_points.shape != (count, 3):
        raise ValueError("plane_points must have shape (P, 3)")
    if plane_normals.shape != (count, 3):
        raise ValueError("plane_normals must have shape (P, 3)")
    if object_indices.device != pose_matrices.device:
        raise ValueError("fixed-plane tensors and pose_matrices must share a device")
    if orientation_mask is None:
        orientation_mask = torch.zeros(
            count, dtype=torch.bool, device=pose_matrices.device
        )
    if orientation_mask.shape != (count,) or orientation_mask.dtype != torch.bool:
        raise ValueError("orientation_mask must be a boolean tensor with shape (P,)")

    if count == 0:
        zero = pose_matrices.sum() * 0.0
        empty = pose_matrices.new_zeros((0,))
        return zero, zero, empty, empty

    normal_lengths = torch.linalg.vector_norm(
        plane_normals, dim=1, keepdim=True
    )
    if torch.any(normal_lengths <= 1e-8):
        raise ValueError("plane_normals must be non-zero")
    normals = plane_normals / normal_lengths

    world_corners = transform_points(pose_matrices, local_corners)[object_indices]
    signed_distances = torch.einsum(
        "pkd,pd->pk",
        world_corners - plane_points[:, None, :],
        normals,
    )
    gaps = signed_distances.amin(dim=1)
    contact = gaps.square().mean()

    planar_normals = normals[:, :2]
    planar_lengths = torch.linalg.vector_norm(planar_normals, dim=1)
    valid_orientation = orientation_mask & (planar_lengths > 1e-8)
    alignment_errors = pose_matrices.new_zeros((count,))
    if torch.any(valid_orientation):
        object_axes = _normalized_planar_axes(pose_matrices)[object_indices]
        unit_planar_normals = planar_normals / planar_lengths[:, None].clamp_min(1e-8)
        axis_alignment = torch.abs(
            torch.einsum("pad,pd->pa", object_axes, unit_planar_normals)
        )
        best_axis_alignment = axis_alignment.amax(dim=1)
        thin_axis_ratio = max(
            0.0,
            float(
                os.environ.get(
                    "IMAGINARIUM_SCENEPROOF_THIN_AXIS_ATTACH_RATIO", "0"
                )
            ),
        )
        if thin_axis_ratio > 0.0:
            selected_corners = local_corners[object_indices]
            local_extents = (
                selected_corners.amax(dim=1) - selected_corners.amin(dim=1)
            )[:, :2].clamp_min(1e-8)
            smallest_extent, thin_axis = local_extents.min(dim=1)
            largest_extent = local_extents.max(dim=1).values
            has_distinct_thin_axis = (
                smallest_extent / largest_extent.clamp_min(1e-8)
                <= thin_axis_ratio
            )
            attachment_alignment = axis_alignment.gather(
                1, thin_axis[:, None]
            ).squeeze(1)
            chosen_alignment = torch.where(
                has_distinct_thin_axis,
                attachment_alignment,
                best_axis_alignment,
            )
        else:
            chosen_alignment = best_axis_alignment
        alignment_errors = torch.where(
            valid_orientation,
            1.0 - chosen_alignment.clamp(max=1.0),
            alignment_errors,
        )
        orientation = alignment_errors[valid_orientation].square().mean()
    else:
        orientation = pose_matrices.sum() * 0.0
    return contact, orientation, gaps, alignment_errors


def project_fixed_planes_(
    yaw_delta: torch.Tensor,
    translation: torch.Tensor,
    base_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    object_indices: torch.Tensor,
    plane_points: torch.Tensor,
    plane_normals: torch.Tensor,
    *,
    passes: int = 2,
) -> None:
    """Project each constrained OBB face exactly onto its fixed plane."""
    if passes <= 0:
        raise ValueError("passes must be positive")
    if object_indices.shape[0] == 0:
        return

    normals = plane_normals / torch.linalg.vector_norm(
        plane_normals, dim=1, keepdim=True
    ).clamp_min(1e-8)
    with torch.no_grad():
        for _ in range(passes):
            for constraint_index, object_index in enumerate(
                object_indices.tolist()
            ):
                poses = reproject_pose_matrices(
                    base_matrices,
                    yaw_delta,
                    translation,
                )
                world_corners = transform_points(
                    poses[object_index : object_index + 1],
                    local_corners[object_index : object_index + 1],
                )[0]
                normal = normals[constraint_index]
                gap = torch.matmul(
                    world_corners - plane_points[constraint_index],
                    normal,
                ).amin()
                translation[object_index].sub_(gap * normal)


def enforce_warm_start_plane_translation_trust_(
    pose_matrices: torch.Tensor,
    base_matrices: torch.Tensor,
    object_indices: torch.Tensor,
    normals: torch.Tensor,
    *,
    normal_limit_m: float,
) -> dict[str, Any]:
    """Remove plane-tangent drift and bound normal motion around warm start."""
    if normal_limit_m < 0:
        raise ValueError("plane anchor normal trust limit must be non-negative")
    if object_indices.numel() == 0:
        return {
            "policy": "warm_start_plane_translation_trust",
            "objects": 0,
            "normal_limit_m": float(normal_limit_m),
            "pre_max_tangent_m": 0.0,
            "pre_max_normal_m": 0.0,
            "post_max_tangent_m": 0.0,
            "post_max_normal_m": 0.0,
        }
    unit_normals = normals / torch.clamp(
        torch.linalg.vector_norm(normals, dim=1, keepdim=True), min=1e-12
    )
    base_centres = base_matrices[object_indices, :3, 3]
    centres = pose_matrices[object_indices, :3, 3]
    delta = centres - base_centres
    normal_delta = torch.sum(delta * unit_normals, dim=1, keepdim=True)
    tangent_delta = delta - normal_delta * unit_normals
    clamped_normal = torch.clamp(
        normal_delta, min=-float(normal_limit_m), max=float(normal_limit_m)
    )
    with torch.no_grad():
        pose_matrices[object_indices, :3, 3] = (
            base_centres + clamped_normal * unit_normals
        )
    return {
        "policy": "warm_start_plane_translation_trust",
        "objects": int(object_indices.numel()),
        "normal_limit_m": float(normal_limit_m),
        "pre_max_tangent_m": float(
            torch.linalg.vector_norm(tangent_delta, dim=1).amax().detach().item()
        ),
        "pre_max_normal_m": float(normal_delta.abs().amax().detach().item()),
        "post_max_tangent_m": 0.0,
        "post_max_normal_m": float(clamped_normal.abs().amax().detach().item()),
    }


def _isotonic_non_decreasing(values: list[float]) -> list[float]:
    """Least-squares projection onto a non-decreasing scalar sequence."""
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([float(value), 1.0])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right_mean, right_weight = blocks.pop()
            left_mean, left_weight = blocks.pop()
            weight = left_weight + right_weight
            blocks.append(
                [
                    (left_mean * left_weight + right_mean * right_weight)
                    / weight,
                    weight,
                ]
            )
    projected: list[float] = []
    for mean, weight in blocks:
        projected.extend([mean] * int(weight))
    return projected


def project_plane_sibling_tangent_intervals_(
    pose_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    plane_object_indices: torch.Tensor,
    plane_points: torch.Tensor,
    plane_normals: torch.Tensor,
    collision_pairs: torch.Tensor,
    footprint_hull_sizes: torch.Tensor | None = None,
    *,
    object_ids: Sequence[str] | None = None,
    maximum_shift_m: float = 0.35,
    collision_tolerance_m: float = 1e-6,
    interval_margin_m: float = 0.0,
) -> dict[str, Any]:
    """Resolve overlapping siblings on one wall with a certified 1-D solve.

    The projection preserves every object's full rotation, height, and plane
    normal coordinate.  It changes only the shared wall-tangent coordinate,
    using the minimum-L2 ordered interval projection.  The complete collision
    candidate set is then re-evaluated element-wise.  Each active-collision
    component is certified independently, so an outlier abstains locally
    rather than suppressing a safe sibling repair elsewhere on the wall.
    """
    if maximum_shift_m < 0 or collision_tolerance_m < 0 or interval_margin_m < 0:
        raise ValueError("plane sibling projection limits must be non-negative")
    _validate_pose_batch(pose_matrices)
    if local_corners.shape != (pose_matrices.shape[0], 8, 3):
        raise ValueError("local_corners must have shape (N, 8, 3)")
    if object_ids is not None and len(object_ids) != pose_matrices.shape[0]:
        raise ValueError("object_ids must match the pose batch")
    count = int(plane_object_indices.numel())
    empty_audit: dict[str, Any] = {
        "policy": "minimum_l2_wall_tangent_interval_projection",
        "constraints": count,
        "groups_considered": 0,
        "components_considered": 0,
        "objects_moved": 0,
        "object_indices": [],
        "shift_by_object_m": {},
        "maximum_shift_m": 0.0,
        "trust_limit_m": float(maximum_shift_m),
        "collision_candidates_checked": int(collision_pairs.shape[0]),
        "collision_nonworsening": True,
        "accepted": False,
        "reason": "no_plane_siblings",
    }
    if count < 2:
        return empty_audit

    normals = plane_normals / torch.linalg.vector_norm(
        plane_normals, dim=1, keepdim=True
    ).clamp_min(1e-12)
    # Canonicalize the plane equation so opposite normal conventions group.
    groups: dict[tuple[int, int, int, int], dict[int, int]] = {}
    for row, object_index in enumerate(plane_object_indices.detach().cpu().tolist()):
        normal = normals[row].detach().cpu()
        point = plane_points[row].detach().cpu()
        if abs(float(normal[2])) >= 0.5:
            continue
        canonical = normal.clone()
        first_nonzero = next(
            (float(value) for value in canonical if abs(float(value)) > 1e-8),
            1.0,
        )
        if first_nonzero < 0:
            canonical = -canonical
        offset = float(torch.dot(canonical, point))
        key = tuple(
            int(round(float(value) * 10000.0)) for value in canonical
        ) + (int(round(offset * 10000.0)),)
        groups.setdefault(key, {}).setdefault(int(object_index), row)

    before_pose = pose_matrices.detach().clone()
    _, collision_before = oriented_penetration_loss(
        before_pose, local_corners, collision_pairs, footprint_hull_sizes
    )
    active_collision_pairs = {
        tuple(sorted((int(pair[0]), int(pair[1]))))
        for pair, penetration in zip(
            collision_pairs.detach().cpu().tolist(),
            collision_before.detach().cpu().tolist(),
        )
        if float(penetration) > float(collision_tolerance_m)
    }
    world = transform_points(before_pose, local_corners)
    proposed = before_pose.clone()
    collision_reference = collision_before
    accepted_shifts: dict[int, float] = {}
    component_audits: list[dict[str, Any]] = []
    components_considered = 0

    for object_rows in groups.values():
        if len(object_rows) < 2:
            continue
        first_row = next(iter(object_rows.values()))
        normal = normals[first_row]
        tangent = torch.stack((-normal[1], normal[0], normal.new_zeros(())))
        tangent = tangent / torch.linalg.vector_norm(tangent).clamp_min(1e-12)
        records: list[dict[str, float | int]] = []
        for object_index in object_rows:
            corners = world[object_index]
            tangent_values = torch.matmul(corners, tangent)
            normal_values = torch.matmul(corners, normal)
            records.append(
                {
                    "index": object_index,
                    "centre": float(torch.dot(before_pose[object_index, :3, 3], tangent)),
                    "half": float((tangent_values.amax() - tangent_values.amin()) * 0.5),
                    "tmin": float(tangent_values.amin()),
                    "tmax": float(tangent_values.amax()),
                    "nmin": float(normal_values.amin()),
                    "nmax": float(normal_values.amax()),
                    "zmin": float(corners[:, 2].amin()),
                    "zmax": float(corners[:, 2].amax()),
                }
            )

        # Only a numerically active collision factor may create a separator.
        # Frozen interval overlap is retained as a consistency check.
        adjacency: dict[int, set[int]] = {
            int(record["index"]): set() for record in records
        }
        for first_position, first in enumerate(records):
            for second in records[first_position + 1 :]:
                overlaps = (
                    tuple(
                        sorted((int(first["index"]), int(second["index"])))
                    ) in active_collision_pairs
                    and
                    min(float(first["tmax"]), float(second["tmax"]))
                    > max(float(first["tmin"]), float(second["tmin"]))
                    and min(float(first["nmax"]), float(second["nmax"]))
                    > max(float(first["nmin"]), float(second["nmin"]))
                    and min(float(first["zmax"]), float(second["zmax"]))
                    > max(float(first["zmin"]), float(second["zmin"]))
                )
                if overlaps:
                    first_index = int(first["index"])
                    second_index = int(second["index"])
                    adjacency[first_index].add(second_index)
                    adjacency[second_index].add(first_index)

        record_by_index = {int(record["index"]): record for record in records}
        unseen = set(adjacency)
        while unseen:
            seed = min(unseen)
            unseen.remove(seed)
            component = {seed}
            frontier = [seed]
            while frontier:
                current = frontier.pop()
                for neighbour in adjacency[current]:
                    if neighbour not in component:
                        component.add(neighbour)
                        unseen.discard(neighbour)
                        frontier.append(neighbour)
            if len(component) < 2:
                continue
            components_considered += 1
            ordered = sorted(
                (record_by_index[index] for index in component),
                key=lambda record: float(record["centre"]),
            )
            offsets = [0.0]
            for left, right in zip(ordered, ordered[1:]):
                offsets.append(
                    offsets[-1]
                    + float(left["half"])
                    + float(right["half"])
                    + float(interval_margin_m)
                )
            reduced = [
                float(record["centre"]) - offset
                for record, offset in zip(ordered, offsets)
            ]
            projected_reduced = _isotonic_non_decreasing(reduced)
            component_shifts: dict[int, float] = {}
            candidate = proposed.clone()
            for record, offset, value in zip(ordered, offsets, projected_reduced):
                shift = value + offset - float(record["centre"])
                object_index = int(record["index"])
                component_shifts[object_index] = shift
                candidate[object_index, :3, 3].add_(tangent * shift)

            component_maximum = max(
                abs(value) for value in component_shifts.values()
            )
            component_audit: dict[str, Any] = {
                "object_indices": sorted(component_shifts),
                "object_ids": (
                    [object_ids[index] for index in sorted(component_shifts)]
                    if object_ids is not None
                    else []
                ),
                "shift_by_object_m": {
                    str(index): float(component_shifts[index])
                    for index in sorted(component_shifts)
                },
                "maximum_shift_m": float(component_maximum),
                "accepted": False,
                "reason": "trust_region_exceeded",
                "worsened_collision_rows": [],
            }
            if component_maximum <= maximum_shift_m + 1e-12:
                _, collision_after = oriented_penetration_loss(
                    candidate,
                    local_corners,
                    collision_pairs,
                    footprint_hull_sizes,
                )
                worsened = (
                    collision_after
                    > collision_reference + float(collision_tolerance_m)
                )
                component_audit["worsened_collision_rows"] = (
                    torch.nonzero(worsened, as_tuple=False)
                    .flatten()
                    .detach()
                    .cpu()
                    .tolist()
                )
                if not bool(worsened.any().item()):
                    component_audit["accepted"] = True
                    component_audit["reason"] = "certified_component"
                    proposed = candidate
                    collision_reference = collision_after
                    accepted_shifts.update(component_shifts)
                else:
                    component_audit["reason"] = (
                        "collision_candidate_worsened"
                    )
            else:
                # The exact non-overlap projection may be far outside the
                # image-initialized trust region.  Search only along that
                # physically meaningful direction, from the largest bounded
                # step back toward the incumbent.  This is a deterministic
                # derivative-free descent, not a relaxed trust threshold.
                bounded_scale = maximum_shift_m / max(component_maximum, 1e-12)
                best_candidate = None
                best_collision = None
                best_shifts = None
                best_total = float(collision_reference.sum().detach().item())
                poll_records: list[dict[str, Any]] = []
                for fraction in (1.0, 0.75, 0.5, 0.25, 0.125):
                    scale = bounded_scale * fraction
                    poll_candidate = proposed.clone()
                    poll_shifts = {
                        index: shift * scale
                        for index, shift in component_shifts.items()
                    }
                    for object_index, shift in poll_shifts.items():
                        poll_candidate[object_index, :3, 3].add_(
                            tangent * shift
                        )
                    _, poll_collision = oriented_penetration_loss(
                        poll_candidate,
                        local_corners,
                        collision_pairs,
                        footprint_hull_sizes,
                    )
                    worsened = (
                        poll_collision
                        > collision_reference + float(collision_tolerance_m)
                    )
                    total = float(poll_collision.sum().detach().item())
                    improves = (
                        total
                        < float(collision_reference.sum().detach().item())
                        - float(collision_tolerance_m)
                    )
                    poll_records.append(
                        {
                            "fraction": fraction,
                            "maximum_shift_m": max(
                                abs(value) for value in poll_shifts.values()
                            ),
                            "collision_total": total,
                            "improves": improves,
                            "nonworsening": not bool(worsened.any().item()),
                        }
                    )
                    if not bool(worsened.any().item()) and improves and total < best_total:
                        best_candidate = poll_candidate
                        best_collision = poll_collision
                        best_shifts = poll_shifts
                        best_total = total
                component_audit["bounded_poll"] = poll_records
                if best_candidate is not None:
                    component_audit["accepted"] = True
                    component_audit["reason"] = "certified_bounded_descent"
                    component_audit["shift_by_object_m"] = {
                        str(index): float(best_shifts[index])
                        for index in sorted(best_shifts)
                    }
                    component_audit["maximum_shift_m"] = max(
                        abs(value) for value in best_shifts.values()
                    )
                    proposed = best_candidate
                    collision_reference = best_collision
                    accepted_shifts.update(best_shifts)
            component_audits.append(component_audit)

    audit = dict(empty_audit)
    audit["groups_considered"] = sum(len(rows) >= 2 for rows in groups.values())
    audit["components_considered"] = components_considered
    audit["component_audits"] = component_audits
    audit["components_accepted"] = sum(
        bool(component["accepted"]) for component in component_audits
    )
    audit["components_abstained"] = sum(
        not bool(component["accepted"]) for component in component_audits
    )
    if not accepted_shifts:
        if not component_audits:
            audit["reason"] = "no_active_plane_sibling_collisions"
        else:
            audit["reason"] = "all_components_abstained"
            audit["maximum_shift_m"] = max(
                float(component["maximum_shift_m"])
                for component in component_audits
            )
            audit["collision_nonworsening"] = all(
                component["reason"] != "collision_candidate_worsened"
                for component in component_audits
            )
        return audit

    maximum_shift = max(abs(value) for value in accepted_shifts.values())
    audit["maximum_shift_m"] = maximum_shift
    audit["object_indices"] = sorted(accepted_shifts)
    audit["object_ids"] = (
        [object_ids[index] for index in sorted(accepted_shifts)]
        if object_ids is not None
        else []
    )
    audit["objects_moved"] = sum(abs(value) > 1e-9 for value in accepted_shifts.values())
    audit["shift_by_object_m"] = {
        str(index): float(accepted_shifts[index]) for index in sorted(accepted_shifts)
    }
    with torch.no_grad():
        pose_matrices.copy_(proposed)
    audit["collision_nonworsening"] = True
    audit["accepted"] = True
    audit["reason"] = "certified_components_applied"
    return audit


def refine_plane_component_image_gauge_(
    pose_matrices: torch.Tensor,
    base_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    sibling_audit: dict[str, Any],
    plane_object_indices: torch.Tensor,
    plane_normals: torch.Tensor,
    collision_pairs: torch.Tensor,
    footprint_hull_sizes: torch.Tensor | None,
    observation_indices: torch.Tensor | None,
    observed_boxes: torch.Tensor | None,
    observed_depths: torch.Tensor | None,
    observed_weights: torch.Tensor | None,
    bbox_size_enabled: torch.Tensor | None,
    world_to_camera: torch.Tensor | None,
    image_size: torch.Tensor | None,
    *,
    support_pairs: torch.Tensor | None = None,
    fixed_support_indices: torch.Tensor | None = None,
    fixed_support_heights: torch.Tensor | None = None,
    containment_pairs: torch.Tensor | None = None,
    plane_points: torch.Tensor | None = None,
    plane_orientation_mask: torch.Tensor | None = None,
    boundary_object_indices: torch.Tensor | None = None,
    boundary_points: torch.Tensor | None = None,
    boundary_normals: torch.Tensor | None = None,
    maximum_total_shift_m: float = 0.35,
    centre_noharm_margin_px: float = 2.0,
    minimum_improvement_px: float = 0.1,
    collision_tolerance_m: float = 1e-6,
) -> dict[str, Any]:
    """Fix the unobservable common tangent gauge with mask-box evidence.

    Collision separation determines relative sibling coordinates but is
    invariant to a common translation along the wall.  This routine polls that
    one remaining scalar against S1 mask-derived boxes.  It never changes the
    relative layout, rotation, height, or wall-normal coordinate.
    """
    audit: dict[str, Any] = {
        "policy": "mask_bbox_component_tangent_gauge",
        "components_considered": 0,
        "components_accepted": 0,
        "components_abstained": 0,
        "objects_moved": 0,
        "component_audits": [],
    }
    required = (
        observation_indices,
        observed_boxes,
        observed_depths,
        observed_weights,
        bbox_size_enabled,
        world_to_camera,
        image_size,
    )
    if any(value is None for value in required) or not sibling_audit.get(
        "accepted", False
    ):
        audit["reason"] = "missing_observations_or_sibling_abstained"
        return audit

    support_pairs = (
        support_pairs
        if support_pairs is not None
        else torch.empty((0, 2), dtype=torch.long, device=pose_matrices.device)
    )
    fixed_support_indices = (
        fixed_support_indices
        if fixed_support_indices is not None
        else torch.empty((0,), dtype=torch.long, device=pose_matrices.device)
    )
    fixed_support_heights = (
        fixed_support_heights
        if fixed_support_heights is not None
        else pose_matrices.new_zeros((0,))
    )
    containment_pairs = (
        containment_pairs
        if containment_pairs is not None
        else torch.empty((0, 2), dtype=torch.long, device=pose_matrices.device)
    )
    plane_points = (
        plane_points
        if plane_points is not None
        else pose_matrices.new_zeros((plane_object_indices.shape[0], 3))
    )
    plane_orientation_mask = (
        plane_orientation_mask
        if plane_orientation_mask is not None
        else torch.zeros(
            plane_object_indices.shape[0],
            dtype=torch.bool,
            device=pose_matrices.device,
        )
    )
    boundary_object_indices = (
        boundary_object_indices
        if boundary_object_indices is not None
        else torch.empty((0,), dtype=torch.long, device=pose_matrices.device)
    )
    boundary_points = (
        boundary_points
        if boundary_points is not None
        else pose_matrices.new_zeros((0, 2))
    )
    boundary_normals = (
        boundary_normals
        if boundary_normals is not None
        else pose_matrices.new_zeros((0, 2))
    )

    def physical_residual_vector(candidate: torch.Tensor) -> torch.Tensor:
        _, collision_values = oriented_penetration_loss(
            candidate, local_corners, collision_pairs, footprint_hull_sizes
        )
        _, contact_gaps = support_contact_loss(
            candidate,
            local_corners,
            support_pairs,
            fixed_support_indices,
            fixed_support_heights,
        )
        _, _, plane_gaps, _ = fixed_plane_loss(
            candidate,
            local_corners,
            plane_object_indices,
            plane_points,
            plane_normals,
            plane_orientation_mask,
        )
        _, containment_errors = support_planar_containment_loss(
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
        return torch.cat(
            (
                collision_values,
                contact_gaps.abs(),
                plane_gaps.abs(),
                torch.sqrt(containment_errors.clamp_min(0.0)),
                boundary_errors,
            )
        )

    normals = plane_normals / torch.linalg.vector_norm(
        plane_normals, dim=1, keepdim=True
    ).clamp_min(1e-12)
    row_by_object: dict[int, int] = {}
    for row, index in enumerate(plane_object_indices.detach().cpu().tolist()):
        row_by_object.setdefault(int(index), row)
    physical_reference = physical_residual_vector(pose_matrices)
    _, incumbent_centre_errors, _, _ = depth_aware_reprojection_loss(
        base_matrices,
        local_corners,
        observation_indices,
        observed_boxes,
        observed_depths,
        observed_weights,
        bbox_size_enabled,
        world_to_camera,
        image_size,
    )
    observation_rows = {
        int(index): row
        for row, index in enumerate(observation_indices.detach().cpu().tolist())
    }

    for component in sibling_audit.get("component_audits", []):
        if not component.get("accepted", False):
            continue
        object_indices = [int(value) for value in component["object_indices"]]
        observed_objects = [
            index for index in object_indices if index in observation_rows
        ]
        component_audit: dict[str, Any] = {
            "object_indices": object_indices,
            "object_ids": component.get("object_ids", []),
            "observed_object_indices": observed_objects,
            "accepted": False,
            "offset_m": 0.0,
            "reason": "missing_component_observations",
            "poll": [],
        }
        audit["components_considered"] += 1
        if not observed_objects or object_indices[0] not in row_by_object:
            audit["components_abstained"] += 1
            audit["component_audits"].append(component_audit)
            continue

        normal = normals[row_by_object[object_indices[0]]]
        tangent = torch.stack((-normal[1], normal[0], normal.new_zeros(())))
        tangent = tangent / torch.linalg.vector_norm(tangent).clamp_min(1e-12)
        existing_shifts = torch.matmul(
            pose_matrices[object_indices, :3, 3]
            - base_matrices[object_indices, :3, 3],
            tangent,
        )
        lower = float((-maximum_total_shift_m - existing_shifts).amax().item())
        upper = float((maximum_total_shift_m - existing_shifts).amin().item())
        if lower > upper:
            component_audit["reason"] = "empty_trust_interval"
            audit["components_abstained"] += 1
            audit["component_audits"].append(component_audit)
            continue

        relevant_rows = torch.tensor(
            [observation_rows[index] for index in observed_objects],
            dtype=torch.long,
            device=pose_matrices.device,
        )
        _, current_errors, _, _ = depth_aware_reprojection_loss(
            pose_matrices,
            local_corners,
            observation_indices,
            observed_boxes,
            observed_depths,
            observed_weights,
            bbox_size_enabled,
            world_to_camera,
            image_size,
        )
        weights = observed_weights[relevant_rows].clamp_min(0.0)
        denominator = weights.sum().clamp_min(1e-6)
        current_score = float(
            (current_errors[relevant_rows] * weights).sum().item()
            / denominator.item()
        )
        best_score = current_score
        best_candidate = None
        best_collision = None
        best_offset = 0.0
        offsets = torch.linspace(
            lower,
            upper,
            steps=17,
            dtype=pose_matrices.dtype,
            device=pose_matrices.device,
        ).detach().cpu().tolist()
        offsets.append(0.0)
        for offset in sorted(set(float(value) for value in offsets)):
            if abs(offset) <= 1e-9:
                continue
            candidate = pose_matrices.clone()
            candidate[object_indices, :3, 3] = (
                candidate[object_indices, :3, 3] + tangent * offset
            )
            candidate_physical = physical_residual_vector(candidate)
            physical_nonworsening = not bool(
                (
                    candidate_physical
                    > physical_reference + float(collision_tolerance_m)
                ).any().item()
            )
            _, candidate_errors, _, _ = depth_aware_reprojection_loss(
                candidate,
                local_corners,
                observation_indices,
                observed_boxes,
                observed_depths,
                observed_weights,
                bbox_size_enabled,
                world_to_camera,
                image_size,
            )
            noharm = bool(
                (
                    candidate_errors[relevant_rows]
                    <= incumbent_centre_errors[relevant_rows]
                    + float(centre_noharm_margin_px)
                ).all().item()
            )
            score = float(
                (candidate_errors[relevant_rows] * weights).sum().item()
                / denominator.item()
            )
            component_audit["poll"].append(
                {
                    "offset_m": offset,
                    "mean_centre_error_px": score,
                    "collision_nonworsening": physical_nonworsening,
                    "physical_nonworsening": physical_nonworsening,
                    "image_noharm": noharm,
                }
            )
            if physical_nonworsening and noharm and score < best_score:
                best_score = score
                best_candidate = candidate
                best_collision = candidate_physical
                best_offset = offset

        component_audit["centre_error_before_px"] = current_score
        component_audit["centre_error_after_px"] = best_score
        if (
            best_candidate is not None
            and current_score - best_score >= minimum_improvement_px
        ):
            with torch.no_grad():
                pose_matrices.copy_(best_candidate)
            physical_reference = best_collision
            component_audit["accepted"] = True
            component_audit["offset_m"] = best_offset
            component_audit["reason"] = "certified_image_gauge"
            audit["components_accepted"] += 1
            audit["objects_moved"] += len(object_indices)
        else:
            component_audit["reason"] = "no_certified_image_improvement"
            audit["components_abstained"] += 1
        audit["component_audits"].append(component_audit)

    audit["reason"] = (
        "certified_components_applied"
        if audit["components_accepted"]
        else "all_components_abstained"
    )
    return audit


def planar_front_vectors(pose_matrices: torch.Tensor) -> torch.Tensor:
    """Return Imaginarium asset fronts (local -Y) projected into world XY."""
    _validate_pose_batch(pose_matrices)
    fronts = -pose_matrices[:, :2, 1]
    return fronts / torch.linalg.vector_norm(
        fronts, dim=1, keepdim=True
    ).clamp_min(1e-8)


def _rotate_planar_vectors(
    vectors: torch.Tensor,
    angles: torch.Tensor,
) -> torch.Tensor:
    if vectors.ndim != 2 or vectors.shape[-1] != 2:
        raise ValueError("vectors must have shape (P, 2)")
    if angles.shape != (vectors.shape[0],):
        raise ValueError("angles must have shape (P,)")
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    return torch.stack(
        (
            cosine * vectors[:, 0] - sine * vectors[:, 1],
            sine * vectors[:, 0] + cosine * vectors[:, 1],
        ),
        dim=1,
    )


def _validate_relation_offsets(
    pose_matrices: torch.Tensor,
    pair_indices: torch.Tensor,
    offsets: torch.Tensor,
) -> None:
    if pair_indices.ndim != 2 or pair_indices.shape[-1] != 2:
        raise ValueError("pair_indices must have shape (P, 2)")
    if pair_indices.dtype != torch.long:
        raise TypeError("pair_indices must use torch.long")
    if pair_indices.device != pose_matrices.device:
        raise ValueError("relation tensors and pose_matrices must share a device")
    if offsets.shape != (pair_indices.shape[0],):
        raise ValueError("offsets must have shape (P,)")
    if offsets.device != pose_matrices.device:
        raise ValueError("relation tensors and pose_matrices must share a device")


def distance_interval_loss(
    pose_matrices: torch.Tensor,
    pair_indices: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayoutVLM XY distance-band loss with a detached target object.

    The squared-distance hinge and per-constraint clamp match the original
    LayoutVLM implementation.  Relations are directed: gradients update only
    the first object in each pair.
    """
    _validate_pose_batch(pose_matrices)
    _validate_relation_offsets(pose_matrices, pair_indices, minimum)
    if maximum.shape != minimum.shape:
        raise ValueError("maximum must match minimum")
    if torch.any(minimum < 0) or torch.any(maximum < minimum):
        raise ValueError("distance bounds must satisfy 0 <= minimum <= maximum")
    if pair_indices.shape[0] == 0:
        zero = pose_matrices.sum() * 0.0
        empty = pose_matrices.new_zeros((0,))
        return zero, empty, empty

    source = pose_matrices[pair_indices[:, 0], :2, 3]
    target = pose_matrices[pair_indices[:, 1], :2, 3].detach()
    squared_distance = (source - target).square().sum(dim=1)
    penalties = (
        torch.relu(minimum.square() - squared_distance)
        + torch.relu(squared_distance - maximum.square())
    ).clamp(max=1.0)
    distances = torch.sqrt(squared_distance.clamp_min(0.0))
    return penalties.mean(), distances, penalties


def align_with_loss(
    pose_matrices: torch.Tensor,
    pair_indices: torch.Tensor,
    angle_offsets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LayoutVLM directed front-vector alignment loss."""
    _validate_pose_batch(pose_matrices)
    _validate_relation_offsets(pose_matrices, pair_indices, angle_offsets)
    if pair_indices.shape[0] == 0:
        zero = pose_matrices.sum() * 0.0
        return zero, pose_matrices.new_zeros((0,))

    fronts = planar_front_vectors(pose_matrices)
    source = fronts[pair_indices[:, 0]]
    source = _rotate_planar_vectors(source, -angle_offsets)
    target = fronts[pair_indices[:, 1]].detach()
    cosine = torch.einsum("pd,pd->p", source, target).clamp(-1.0, 1.0)
    errors = 1.0 - cosine
    return errors.mean(), errors


def _cross_2d(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def point_towards_loss(
    pose_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    pair_indices: torch.Tensor,
    angle_offsets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LayoutVLM directed facing loss with its ray/target-OBB zero case."""
    _validate_pose_batch(pose_matrices)
    _validate_relation_offsets(pose_matrices, pair_indices, angle_offsets)
    if local_corners.shape != (pose_matrices.shape[0], 8, 3):
        raise ValueError("local_corners must have shape (N, 8, 3)")
    if pair_indices.shape[0] == 0:
        zero = pose_matrices.sum() * 0.0
        return zero, pose_matrices.new_zeros((0,))

    source_indices = pair_indices[:, 0]
    target_indices = pair_indices[:, 1]
    fronts = planar_front_vectors(pose_matrices)
    rays = _rotate_planar_vectors(fronts[source_indices], -angle_offsets)
    origins = pose_matrices[source_indices, :2, 3]
    target_centers = pose_matrices[target_indices, :2, 3].detach()

    # local_box_corners order is min/min, min/max, max/min, max/max in XY.
    # [0, 4, 6, 2] traces the rectangular footprint cyclically.
    world_corners = transform_points(pose_matrices, local_corners)
    target_polygons = world_corners[
        target_indices
    ][:, (0, 4, 6, 2), :2].detach()
    edge_start = target_polygons
    edge_end = torch.roll(target_polygons, shifts=-1, dims=1)
    edges = edge_end - edge_start
    origin_to_edge = edge_start - origins[:, None, :]
    expanded_rays = rays[:, None, :].expand_as(edges)
    denominator = _cross_2d(expanded_rays, edges)
    safe_denominator = torch.where(
        denominator.abs() > 1e-8,
        denominator,
        torch.ones_like(denominator),
    )
    ray_parameter = _cross_2d(origin_to_edge, edges) / safe_denominator
    edge_parameter = _cross_2d(origin_to_edge, expanded_rays) / safe_denominator
    hits = (
        (denominator.abs() > 1e-8)
        & (ray_parameter >= 0.0)
        & (edge_parameter >= 0.0)
        & (edge_parameter <= 1.0)
    ).any(dim=1)

    target_directions = target_centers - origins
    target_directions = target_directions / torch.linalg.vector_norm(
        target_directions, dim=1, keepdim=True
    ).clamp_min(1e-8)
    cosine = torch.einsum(
        "pd,pd->p", rays, target_directions
    ).clamp(-1.0, 1.0)
    errors = torch.where(hits, torch.zeros_like(cosine), 1.0 - cosine)
    return errors.mean(), errors


def support_planar_containment_loss(
    pose_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    support_pairs: torch.Tensor,
    footprint_hull_sizes: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep each supported object's complete footprint inside its parent.

    This matches the physical-realizability evaluator. A center-only
    constraint can be zero even when most of a long child lies off its support.
    """
    _validate_pose_batch(pose_matrices)
    if support_pairs.ndim != 2 or support_pairs.shape[-1] != 2:
        raise ValueError("support_pairs must have shape (S, 2)")
    if support_pairs.dtype != torch.long:
        raise TypeError("support_pairs must use torch.long")
    if support_pairs.device != pose_matrices.device:
        raise ValueError("support_pairs and pose_matrices must share a device")
    if local_corners.shape != (pose_matrices.shape[0], 8, 3):
        raise ValueError("local_corners must have shape (N, 8, 3)")
    if support_pairs.shape[0] == 0:
        zero = pose_matrices.sum() * 0.0
        return zero, pose_matrices.new_zeros((0,))

    child = support_pairs[:, 0]
    parent = support_pairs[:, 1]
    world_corners = transform_points(pose_matrices, local_corners)
    if footprint_hull_sizes is None:
        parent_polygon = world_corners[parent][:, [0, 2, 6, 4], :2].detach()
        edge_mask = torch.ones(
            (support_pairs.shape[0], 4),
            dtype=torch.bool,
            device=pose_matrices.device,
        )
        next_polygon = torch.roll(parent_polygon, shifts=-1, dims=1)
    else:
        if footprint_hull_sizes.shape != (pose_matrices.shape[0],):
            raise ValueError("footprint_hull_sizes must have shape (N,)")
        if footprint_hull_sizes.dtype != torch.long:
            raise TypeError("footprint_hull_sizes must use torch.long")
        if footprint_hull_sizes.device != pose_matrices.device:
            raise ValueError(
                "footprint_hull_sizes and pose_matrices must share a device"
            )
        if bool(
            ((footprint_hull_sizes < 3) | (footprint_hull_sizes > 8))
            .any()
            .item()
        ):
            raise ValueError(
                "footprint_hull_sizes must contain values in [3, 8]"
            )
        parent_polygon = world_corners[parent, :, :2].detach()
        hull_sizes = footprint_hull_sizes[parent]
        slots = torch.arange(8, device=pose_matrices.device)[None, :]
        edge_mask = slots < hull_sizes[:, None]
        next_slots = torch.where(
            slots + 1 < hull_sizes[:, None], slots + 1, 0
        ).expand(parent_polygon.shape[0], -1)
        next_polygon = torch.gather(
            parent_polygon,
            1,
            next_slots[:, :, None].expand(-1, -1, 2),
        )
    edges = next_polygon - parent_polygon
    normals = torch.stack((-edges[..., 1], edges[..., 0]), dim=-1)
    normals = normals / torch.linalg.vector_norm(
        normals, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    centroid_weights = edge_mask.to(parent_polygon.dtype)
    centroids = (
        (parent_polygon * centroid_weights[:, :, None]).sum(dim=1)
        / centroid_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    )
    inward_score = torch.einsum(
        "sed,sed->se",
        centroids[:, None, :] - parent_polygon,
        normals,
    )
    normals = torch.where(
        (inward_score >= 0)[:, :, None], normals, -normals
    )
    child_corners = world_corners[child, :, :2]
    signed = torch.einsum(
        "sced,sed->sce",
        child_corners[:, :, None, :] - parent_polygon[:, None, :, :],
        normals,
    )
    violation = torch.relu(-signed)
    violation = torch.where(
        edge_mask[:, None, :], violation, torch.zeros_like(violation)
    )
    per_pair = violation.square().amax(dim=(1, 2))
    return per_pair.mean(), per_pair


def gate_support_containment_pairs(
    pose_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    support_pairs: torch.Tensor,
    maximum_initial_error: float = 0.5,
    footprint_hull_sizes: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split support edges by their warm-start footprint error.

    Vertical contact remains defined for every input support edge. Planar
    containment is safe only when the child is already geometrically
    consistent with the parent's footprint; otherwise hard PGD could move an
    image-initialized object by metres.

    Returns:
        ``(accepted_pairs, rejected_pairs, initial_errors_metres)``. The error
        vector follows the original ``support_pairs`` order.
    """
    if maximum_initial_error < 0:
        raise ValueError("maximum_initial_error must be non-negative")
    _, squared_errors = support_planar_containment_loss(
        pose_matrices,
        local_corners,
        support_pairs,
        footprint_hull_sizes,
    )
    errors = torch.sqrt(squared_errors)
    if support_pairs.shape[0]:
        child = support_pairs[:, 0]
        parent = support_pairs[:, 1]
        world_corners = transform_points(pose_matrices, local_corners)
        parent_centers = pose_matrices[parent, :2, 3]
        child_centers = pose_matrices[child, :2, 3]
        parent_axes = _normalized_planar_axes(pose_matrices)[parent]
        parent_projection = torch.einsum(
            "spd,sad->spa",
            world_corners[parent, :, :2] - parent_centers[:, None, :],
            parent_axes,
        )
        child_projection = torch.einsum(
            "spd,sad->spa",
            world_corners[child, :, :2] - child_centers[:, None, :],
            parent_axes,
        )
        parent_span = parent_projection.amax(dim=1) - parent_projection.amin(dim=1)
        child_span = child_projection.amax(dim=1) - child_projection.amin(dim=1)
        geometrically_feasible = torch.all(
            child_span <= parent_span + 1e-6, dim=1
        )
    else:
        geometrically_feasible = torch.empty(
            (0,), dtype=torch.bool, device=pose_matrices.device
        )
    accepted = (
        (errors <= maximum_initial_error) & geometrically_feasible
    )
    return support_pairs[accepted], support_pairs[~accepted], errors


def convex_polygon_halfspaces(
    polygon: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a convex room polygon into inward-facing unit half-spaces.

    ``polygon`` may be clockwise or counter-clockwise. Returned points and
    normals satisfy ``dot(x - point, normal) >= 0`` for interior points.
    """
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("polygon must have shape (B, 2)")
    if polygon.shape[0] < 3:
        raise ValueError("polygon must contain at least three vertices")
    if not polygon.is_floating_point():
        raise TypeError("polygon must use a floating-point dtype")

    signed_area_twice = torch.sum(
        polygon[:, 0] * torch.roll(polygon[:, 1], shifts=-1)
        - polygon[:, 1] * torch.roll(polygon[:, 0], shifts=-1)
    )
    if torch.abs(signed_area_twice) <= 1e-8:
        raise ValueError("polygon has zero area")
    vertices = torch.flip(polygon, dims=(0,)) if signed_area_twice < 0 else polygon
    edges = torch.roll(vertices, shifts=-1, dims=0) - vertices
    lengths = torch.linalg.vector_norm(edges, dim=1)
    if torch.any(lengths <= 1e-8):
        raise ValueError("polygon contains a zero-length edge")
    inward_normals = torch.stack((-edges[:, 1], edges[:, 0]), dim=1)
    inward_normals = inward_normals / lengths[:, None]

    # A convex polygon must place every vertex in every inward half-space.
    signed = torch.einsum(
        "vbd,bd->vb",
        vertices[:, None, :] - vertices[None, :, :],
        inward_normals,
    )
    if torch.any(signed < -1e-6):
        raise ValueError("polygon must be convex and ordered around its boundary")
    return vertices, inward_normals


def room_boundary_loss(
    pose_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    object_indices: torch.Tensor,
    boundary_points: torch.Tensor,
    boundary_normals: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Penalize OBB footprints that extend outside a convex room polygon."""
    _validate_pose_batch(pose_matrices)
    if object_indices.ndim != 1 or object_indices.dtype != torch.long:
        raise ValueError("object_indices must be a one-dimensional long tensor")
    if object_indices.device != pose_matrices.device:
        raise ValueError("object_indices and pose_matrices must share a device")
    if local_corners.shape != (pose_matrices.shape[0], 8, 3):
        raise ValueError("local_corners must have shape (N, 8, 3)")
    edge_count = boundary_points.shape[0]
    if boundary_points.shape != (edge_count, 2):
        raise ValueError("boundary_points must have shape (B, 2)")
    if boundary_normals.shape != (edge_count, 2):
        raise ValueError("boundary_normals must have shape (B, 2)")
    if boundary_points.device != pose_matrices.device:
        raise ValueError("boundary_points and pose_matrices must share a device")
    if boundary_normals.device != pose_matrices.device:
        raise ValueError("boundary_normals and pose_matrices must share a device")
    if object_indices.shape[0] == 0 or edge_count == 0:
        zero = pose_matrices.sum() * 0.0
        return zero, pose_matrices.new_zeros((object_indices.shape[0],))

    world_corners = transform_points(
        pose_matrices, local_corners
    )[object_indices, :, :2]
    signed = torch.einsum(
        "ocbd,bd->ocb",
        world_corners[:, :, None, :] - boundary_points[None, None, :, :],
        boundary_normals,
    )
    per_object = torch.relu(-signed).amax(dim=(1, 2))
    return per_object.square().mean(), per_object


def project_room_boundary_(
    yaw_delta: torch.Tensor,
    translation: torch.Tensor,
    base_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    object_indices: torch.Tensor,
    boundary_points: torch.Tensor,
    boundary_normals: torch.Tensor,
    *,
    passes: int = 3,
) -> None:
    """PGD-project every optimized OBB footprint into the convex room."""
    if passes <= 0:
        raise ValueError("passes must be positive")
    if object_indices.shape[0] == 0 or boundary_points.shape[0] == 0:
        return

    with torch.no_grad():
        for _ in range(passes):
            for object_index in object_indices.tolist():
                for edge_index in range(boundary_points.shape[0]):
                    poses = reproject_pose_matrices(
                        base_matrices,
                        yaw_delta,
                        translation,
                    )
                    corners = transform_points(
                        poses[object_index : object_index + 1],
                        local_corners[object_index : object_index + 1],
                    )[0, :, :2]
                    normal = boundary_normals[edge_index]
                    signed = torch.einsum(
                        "cd,d->c",
                        corners - boundary_points[edge_index],
                        normal,
                    )
                    minimum = signed.amin()
                    if minimum < 0:
                        translation[object_index, :2].add_(-minimum * normal)


def minimum_norm_halfspace_translation(
    normals: torch.Tensor,
    lower_bounds: torch.Tensor,
    *,
    tolerance: float = 1e-9,
) -> torch.Tensor:
    """Solve the exact two-dimensional minimum-translation half-space QP.

    The returned translation minimizes ``0.5 * ||delta||^2`` subject to
    ``normals @ delta >= lower_bounds``.  In two dimensions the optimum is
    either zero, the projection onto one active boundary, or the intersection
    of two active boundaries, so enumerating those active sets is exact and
    avoids the order-dependent drift of repeated sequential projections.
    """
    if normals.ndim != 2 or normals.shape[1] != 2:
        raise ValueError("normals must have shape (E, 2)")
    if lower_bounds.shape != (normals.shape[0],):
        raise ValueError("lower_bounds must have shape (E,)")
    if normals.device != lower_bounds.device:
        raise ValueError("normals and lower_bounds must share a device")
    if not normals.is_floating_point() or not lower_bounds.is_floating_point():
        raise ValueError("half-space inputs must be floating point")
    if normals.shape[0] == 0:
        return normals.new_zeros((2,))
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    zero = normals.new_zeros((2,))
    candidates: list[torch.Tensor] = []
    dtype_tolerance = 32.0 * torch.finfo(normals.dtype).eps
    numerical_tolerance = max(tolerance, dtype_tolerance)
    feasibility_tolerance = normals.new_tensor(numerical_tolerance)
    determinant_tolerance = normals.new_tensor(numerical_tolerance)

    def is_feasible(candidate: torch.Tensor) -> bool:
        residual = normals @ candidate - lower_bounds
        return bool(torch.all(residual >= -feasibility_tolerance).item())

    if is_feasible(zero):
        candidates.append(zero)

    for edge_index in range(normals.shape[0]):
        normal = normals[edge_index]
        denominator = torch.dot(normal, normal)
        if bool((denominator > determinant_tolerance).item()):
            candidate = lower_bounds[edge_index] * normal / denominator
            if is_feasible(candidate):
                candidates.append(candidate)

    for first_index in range(normals.shape[0]):
        for second_index in range(first_index + 1, normals.shape[0]):
            matrix = torch.stack(
                (normals[first_index], normals[second_index]), dim=0
            )
            determinant = torch.linalg.det(matrix)
            if bool((determinant.abs() <= determinant_tolerance).item()):
                continue
            rhs = torch.stack(
                (lower_bounds[first_index], lower_bounds[second_index])
            )
            candidate = torch.linalg.solve(matrix, rhs)
            if is_feasible(candidate):
                candidates.append(candidate)

    if not candidates:
        raise ValueError(
            "half-space constraints have no feasible two-dimensional "
            "translation"
        )
    squared_norms = torch.stack(
        [torch.dot(candidate, candidate) for candidate in candidates]
    )
    return candidates[int(torch.argmin(squared_norms).item())]


def project_support_footprints_(
    yaw_delta: torch.Tensor,
    translation: torch.Tensor,
    base_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    support_pairs: torch.Tensor,
    *,
    passes: int = 2,
    footprint_hull_sizes: torch.Tensor | None = None,
    infeasible_policy: str = "raise",
) -> list[dict[str, int | str]]:
    """Project complete child footprints into parent OBB footprints."""
    if passes <= 0:
        raise ValueError("passes must be positive")
    if support_pairs.shape[0] == 0:
        return []
    if infeasible_policy not in {"raise", "restore_warm_start_planar"}:
        raise ValueError(
            "infeasible_policy must be 'raise' or "
            "'restore_warm_start_planar'"
        )

    audit: list[dict[str, int | str]] = []
    abstained_pairs: set[tuple[int, int]] = set()

    with torch.no_grad():
        for _ in range(passes):
            for child_index, parent_index in support_pairs.tolist():
                pair = (int(child_index), int(parent_index))
                if pair in abstained_pairs:
                    continue
                poses = reproject_pose_matrices(
                    base_matrices,
                    yaw_delta,
                    translation,
                )
                parent_pose = poses[parent_index : parent_index + 1]
                parent_corners_all = transform_points(
                    parent_pose,
                    local_corners[parent_index : parent_index + 1],
                )[0]
                if footprint_hull_sizes is None:
                    parent_corners = parent_corners_all[[0, 2, 6, 4], :2]
                else:
                    hull_size = int(footprint_hull_sizes[parent_index].item())
                    if not 3 <= hull_size <= 8:
                        raise ValueError(
                            "footprint_hull_sizes must contain values in [3, 8]"
                        )
                    parent_corners = parent_corners_all[:hull_size, :2]
                edges = torch.roll(parent_corners, shifts=-1, dims=0) - parent_corners
                normals = torch.stack((-edges[:, 1], edges[:, 0]), dim=1)
                normals = normals / torch.linalg.vector_norm(
                    normals, dim=1, keepdim=True
                ).clamp_min(1e-8)
                centroid = parent_corners.mean(dim=0)
                inward_score = torch.einsum(
                    "ed,ed->e", centroid[None, :] - parent_corners, normals
                )
                normals = torch.where(
                    (inward_score >= 0)[:, None], normals, -normals
                )
                child_corners = transform_points(
                    poses[child_index : child_index + 1],
                    local_corners[child_index : child_index + 1],
                )[0, :, :2]
                signed = torch.einsum(
                    "ced,ed->ce",
                    child_corners[:, None, :] - parent_corners[None, :, :],
                    normals,
                )
                lower_bounds = -signed.amin(dim=0)
                try:
                    correction = minimum_norm_halfspace_translation(
                        normals,
                        lower_bounds,
                    )
                except ValueError as error:
                    if infeasible_policy == "raise":
                        raise ValueError(
                            "support footprint cannot be made feasible by "
                            "translation: "
                            f"child_index={child_index}, "
                            f"parent_index={parent_index}"
                        ) from error
                    # A relation that becomes infeasible after continuous
                    # rotation cannot be repaired by inventing a proxy
                    # footprint.  Fail closed locally: restore the child's
                    # warm-start planar gauge, keep its already-projected Z
                    # contact, and abstain from this containment projection.
                    yaw_delta[child_index].zero_()
                    translation[child_index, :2].copy_(
                        base_matrices[child_index, :2, 3]
                    )
                    audit.append(
                        {
                            "child_index": int(child_index),
                            "parent_index": int(parent_index),
                            "status": "abstained_restore_warm_start_planar",
                        }
                    )
                    abstained_pairs.add(pair)
                    continue
                translation[child_index, :2].add_(correction)
    return audit


def optimize_collision_stage(
    base_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    pair_indices: torch.Tensor,
    *,
    iterations: int = 100,
    learning_rate: float = 0.01,
    collision_weight: float = 1.0,
    warm_start_weight: float = 0.01,
) -> Tuple[torch.Tensor, list[dict[str, float]]]:
    """Optimize only collision penetration as the first loss-bearing stage.

    Z is held at the deterministic warm start until support/contact loss is
    introduced.  This prevents collision-only optimization from making objects
    float above one another.
    """
    _validate_pose_batch(base_matrices)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    yaw_delta, translation = initialize_pose_variables(base_matrices)
    optimizer = torch.optim.Adam([yaw_delta, translation], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, iterations // 4),
        gamma=0.5,
    )
    history: list[dict[str, float]] = []

    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        pose_matrices = reproject_pose_matrices(
            base_matrices,
            yaw_delta,
            translation,
        )
        collision, per_pair = oriented_penetration_loss(
            pose_matrices,
            local_corners,
            pair_indices,
        )
        warm_start = warm_start_regularization(
            yaw_delta,
            translation,
            base_matrices,
        )
        total = collision_weight * collision + warm_start_weight * warm_start
        total.backward()
        if translation.grad is not None:
            translation.grad[:, 2].zero_()
        torch.nn.utils.clip_grad_norm_([yaw_delta, translation], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if iteration == 0 or iteration == iterations - 1 or (iteration + 1) % 25 == 0:
            history.append(
                {
                    "iteration": float(iteration + 1),
                    "total": float(total.detach().item()),
                    "collision": float(collision.detach().item()),
                    "warm_start": float(warm_start.detach().item()),
                    "penetrating_pairs": float(
                        torch.count_nonzero(per_pair.detach() > 0).item()
                    ),
                }
            )

    final_pose_matrices = reproject_pose_matrices(
        base_matrices,
        yaw_delta,
        translation,
    )
    return final_pose_matrices.detach(), history


def optimize_contact_stage(
    base_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    collision_pairs: torch.Tensor,
    support_pairs: torch.Tensor,
    fixed_support_indices: torch.Tensor | None = None,
    fixed_support_heights: torch.Tensor | None = None,
    *,
    iterations: int = 150,
    learning_rate: float = 0.01,
    collision_weight: float = 1.0,
    contact_weight: float = 2.0,
    warm_start_weight: float = 0.01,
) -> Tuple[torch.Tensor, list[dict[str, float]]]:
    """Joint collision/contact Adam stage with support-plane PGD."""
    _validate_pose_batch(base_matrices)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    yaw_delta, translation = initialize_pose_variables(base_matrices)
    optimizer = torch.optim.Adam([yaw_delta, translation], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, iterations // 4),
        gamma=0.5,
    )
    history: list[dict[str, float]] = []

    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        pose_matrices = reproject_pose_matrices(
            base_matrices,
            yaw_delta,
            translation,
        )
        collision, per_pair = oriented_penetration_loss(
            pose_matrices,
            local_corners,
            collision_pairs,
        )
        contact, gaps = support_contact_loss(
            pose_matrices,
            local_corners,
            support_pairs,
            fixed_support_indices,
            fixed_support_heights,
        )
        warm_start = warm_start_regularization(
            yaw_delta,
            translation,
            base_matrices,
        )
        total = (
            collision_weight * collision
            + contact_weight * contact
            + warm_start_weight * warm_start
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_([yaw_delta, translation], max_norm=1.0)
        optimizer.step()
        scheduler.step()
        project_support_contacts_(
            yaw_delta,
            translation,
            base_matrices,
            local_corners,
            support_pairs,
            fixed_support_indices,
            fixed_support_heights,
        )

        if iteration == 0 or iteration == iterations - 1 or (iteration + 1) % 25 == 0:
            history.append(
                {
                    "iteration": float(iteration + 1),
                    "total": float(total.detach().item()),
                    "collision": float(collision.detach().item()),
                    "contact": float(contact.detach().item()),
                    "max_contact_gap": float(
                        gaps.detach().abs().amax().item() if gaps.numel() else 0.0
                    ),
                    "warm_start": float(warm_start.detach().item()),
                    "penetrating_pairs": float(
                        torch.count_nonzero(per_pair.detach() > 0).item()
                    ),
                }
            )

    project_support_contacts_(
        yaw_delta,
        translation,
        base_matrices,
        local_corners,
        support_pairs,
        fixed_support_indices,
        fixed_support_heights,
    )
    final_pose_matrices = reproject_pose_matrices(
        base_matrices,
        yaw_delta,
        translation,
    )
    _, final_gaps = support_contact_loss(
        final_pose_matrices,
        local_corners,
        support_pairs,
        fixed_support_indices,
        fixed_support_heights,
    )
    if history:
        if solver == "scenelm":
            with torch.no_grad():
                final_solver_residuals = dense_factor_residuals(
                    pack_pose_parameters(yaw_delta, translation)
                )
                final_solver_energy = float(
                    final_solver_residuals.square().sum().item()
                )
            history[-1]["total"] = final_solver_energy
            history[-1]["lm_final_residual_energy"] = final_solver_energy
        history[-1]["projected_max_contact_gap"] = float(
            final_gaps.detach().abs().amax().item()
            if final_gaps.numel()
            else 0.0
        )
    return final_pose_matrices.detach(), history


def optimize_plane_stage(
    base_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    collision_pairs: torch.Tensor,
    support_pairs: torch.Tensor,
    fixed_support_indices: torch.Tensor,
    fixed_support_heights: torch.Tensor,
    plane_object_indices: torch.Tensor,
    plane_points: torch.Tensor,
    plane_normals: torch.Tensor,
    plane_orientation_mask: torch.Tensor,
    *,
    iterations: int = 200,
    learning_rate: float = 0.01,
    collision_weight: float = 1.0,
    contact_weight: float = 2.0,
    plane_weight: float = 2.0,
    orientation_weight: float = 0.25,
    warm_start_weight: float = 0.01,
) -> Tuple[torch.Tensor, list[dict[str, float]]]:
    """Joint collision/support/fixed-plane stage with contact PGD."""
    _validate_pose_batch(base_matrices)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    yaw_delta, translation = initialize_pose_variables(base_matrices)
    optimizer = torch.optim.Adam([yaw_delta, translation], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, iterations // 4),
        gamma=0.5,
    )
    history: list[dict[str, float]] = []

    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        pose_matrices = reproject_pose_matrices(
            base_matrices,
            yaw_delta,
            translation,
        )
        collision, per_pair = oriented_penetration_loss(
            pose_matrices,
            local_corners,
            collision_pairs,
        )
        contact, contact_gaps = support_contact_loss(
            pose_matrices,
            local_corners,
            support_pairs,
            fixed_support_indices,
            fixed_support_heights,
        )
        plane, orientation, plane_gaps, alignment_errors = fixed_plane_loss(
            pose_matrices,
            local_corners,
            plane_object_indices,
            plane_points,
            plane_normals,
            plane_orientation_mask,
        )
        warm_start = warm_start_regularization(
            yaw_delta,
            translation,
            base_matrices,
        )
        total = (
            collision_weight * collision
            + contact_weight * contact
            + plane_weight * plane
            + orientation_weight * orientation
            + warm_start_weight * warm_start
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_([yaw_delta, translation], max_norm=1.0)
        optimizer.step()
        scheduler.step()
        project_support_contacts_(
            yaw_delta,
            translation,
            base_matrices,
            local_corners,
            support_pairs,
            fixed_support_indices,
            fixed_support_heights,
        )
        project_fixed_planes_(
            yaw_delta,
            translation,
            base_matrices,
            local_corners,
            plane_object_indices,
            plane_points,
            plane_normals,
        )

        if iteration == 0 or iteration == iterations - 1 or (iteration + 1) % 25 == 0:
            history.append(
                {
                    "iteration": float(iteration + 1),
                    "total": float(total.detach().item()),
                    "collision": float(collision.detach().item()),
                    "contact": float(contact.detach().item()),
                    "max_contact_gap": float(
                        contact_gaps.detach().abs().amax().item()
                        if contact_gaps.numel()
                        else 0.0
                    ),
                    "plane": float(plane.detach().item()),
                    "max_plane_gap": float(
                        plane_gaps.detach().abs().amax().item()
                        if plane_gaps.numel()
                        else 0.0
                    ),
                    "orientation": float(orientation.detach().item()),
                    "max_orientation_error": float(
                        alignment_errors.detach().amax().item()
                        if alignment_errors.numel()
                        else 0.0
                    ),
                    "warm_start": float(warm_start.detach().item()),
                    "penetrating_pairs": float(
                        torch.count_nonzero(per_pair.detach() > 0).item()
                    ),
                }
            )

    project_support_contacts_(
        yaw_delta,
        translation,
        base_matrices,
        local_corners,
        support_pairs,
        fixed_support_indices,
        fixed_support_heights,
    )
    project_fixed_planes_(
        yaw_delta,
        translation,
        base_matrices,
        local_corners,
        plane_object_indices,
        plane_points,
        plane_normals,
    )
    final_pose_matrices = reproject_pose_matrices(
        base_matrices,
        yaw_delta,
        translation,
    )
    _, final_contact_gaps = support_contact_loss(
        final_pose_matrices,
        local_corners,
        support_pairs,
        fixed_support_indices,
        fixed_support_heights,
    )
    _, _, final_plane_gaps, final_alignment_errors = fixed_plane_loss(
        final_pose_matrices,
        local_corners,
        plane_object_indices,
        plane_points,
        plane_normals,
        plane_orientation_mask,
    )
    if history:
        history[-1]["projected_max_contact_gap"] = float(
            final_contact_gaps.detach().abs().amax().item()
            if final_contact_gaps.numel()
            else 0.0
        )
        history[-1]["projected_max_plane_gap"] = float(
            final_plane_gaps.detach().abs().amax().item()
            if final_plane_gaps.numel()
            else 0.0
        )
        history[-1]["final_max_orientation_error"] = float(
            final_alignment_errors.detach().amax().item()
            if final_alignment_errors.numel()
            else 0.0
        )
    return final_pose_matrices.detach(), history


def _scatter_object_max_(
    destination: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Scatter maximum factor residuals onto their incident objects."""
    if indices.numel() == 0 or values.numel() == 0:
        return
    flat_indices = indices.reshape(-1)
    if indices.ndim == 2:
        flat_values = values.reshape(-1, 1).expand(-1, indices.shape[1]).reshape(-1)
    else:
        flat_values = values.reshape(-1)
    for index, value in zip(flat_indices, flat_values):
        destination[index] = torch.maximum(destination[index], value)


def active_set_object_residuals(
    object_count: int,
    *,
    collision_pairs: torch.Tensor,
    collision_values: torch.Tensor,
    support_pairs: torch.Tensor,
    contact_gaps: torch.Tensor,
    fixed_support_indices: torch.Tensor,
    plane_object_indices: torch.Tensor,
    plane_gaps: torch.Tensor,
    plane_alignment_errors: torch.Tensor,
    containment_pairs: torch.Tensor,
    containment_errors: torch.Tensor,
    distance_pairs: torch.Tensor,
    distance_penalties: torch.Tensor,
    align_pairs: torch.Tensor,
    align_errors: torch.Tensor,
    point_pairs: torch.Tensor,
    point_errors: torch.Tensor,
    boundary_object_indices: torch.Tensor,
    boundary_errors: torch.Tensor,
    depth_observation_indices: torch.Tensor,
    depth_centre_errors: torch.Tensor,
    depth_size_errors: torch.Tensor,
    depth_relative_errors: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Convert sparse factor residuals into auditable per-object maxima.

    Pairwise factors are assigned to both endpoints.  This is deliberately
    conservative: if an active object makes a frozen neighbour's constraint
    unsafe, the neighbour is eligible for immediate wake-up.
    """
    if object_count <= 0:
        raise ValueError("object_count must be positive")
    if collision_pairs.device != collision_values.device:
        raise ValueError("factor indices and values must share a device")
    device = collision_values.device
    dtype = collision_values.dtype

    def zeros() -> torch.Tensor:
        return torch.zeros(object_count, dtype=dtype, device=device)

    result = {
        "collision": zeros(),
        "contact": zeros(),
        "plane": zeros(),
        "orientation": zeros(),
        "containment": zeros(),
        "semantic": zeros(),
        "boundary": zeros(),
        "depth_centre": zeros(),
        "depth_size": zeros(),
        "depth_relative": zeros(),
    }
    _scatter_object_max_(
        result["collision"], collision_pairs, collision_values.detach().abs()
    )
    pair_contact_count = support_pairs.shape[0]
    _scatter_object_max_(
        result["contact"],
        support_pairs,
        contact_gaps[:pair_contact_count].detach().abs(),
    )
    _scatter_object_max_(
        result["contact"],
        fixed_support_indices,
        contact_gaps[pair_contact_count:].detach().abs(),
    )
    _scatter_object_max_(
        result["plane"], plane_object_indices, plane_gaps.detach().abs()
    )
    _scatter_object_max_(
        result["orientation"],
        plane_object_indices,
        plane_alignment_errors.detach().abs(),
    )
    _scatter_object_max_(
        result["containment"],
        containment_pairs,
        torch.sqrt(torch.clamp_min(containment_errors.detach(), 0.0)),
    )
    _scatter_object_max_(
        result["semantic"], distance_pairs, distance_penalties.detach().abs()
    )
    _scatter_object_max_(
        result["semantic"], align_pairs, align_errors.detach().abs()
    )
    _scatter_object_max_(
        result["semantic"], point_pairs, point_errors.detach().abs()
    )
    _scatter_object_max_(
        result["boundary"],
        boundary_object_indices,
        boundary_errors.detach().abs(),
    )
    _scatter_object_max_(
        result["depth_centre"],
        depth_observation_indices,
        depth_centre_errors.detach().abs(),
    )
    _scatter_object_max_(
        result["depth_size"],
        depth_observation_indices,
        depth_size_errors.detach().abs(),
    )
    _scatter_object_max_(
        result["depth_relative"],
        depth_observation_indices,
        depth_relative_errors.detach().abs(),
    )
    return result


def active_set_safe_mask(
    residuals: dict[str, torch.Tensor],
    translation_update_ema: torch.Tensor,
    yaw_update_ema: torch.Tensor,
    *,
    thresholds: dict[str, float] | None = None,
    threshold_multiplier: float = 1.0,
) -> torch.Tensor:
    """Return objects that are converged and below every safety threshold."""
    defaults = {
        "collision": 1.0e-4,
        "contact": 0.02,
        "plane": 0.02,
        "orientation": 1.0 - math.cos(math.radians(15.0)),
        "containment": 0.02,
        "semantic": 0.10,
        "boundary": 0.02,
        "depth_centre": 4.0,
        "depth_size": 0.05,
        "depth_relative": 0.03,
        "translation_update": 0.01,
        "yaw_update": math.radians(1.0),
    }
    if thresholds:
        unknown = set(thresholds) - set(defaults)
        if unknown:
            raise ValueError(f"unknown active-set thresholds: {sorted(unknown)}")
        defaults.update({key: float(value) for key, value in thresholds.items()})
    if threshold_multiplier <= 0:
        raise ValueError("threshold_multiplier must be positive")
    defaults = {
        key: value * threshold_multiplier for key, value in defaults.items()
    }
    safe = torch.ones_like(translation_update_ema, dtype=torch.bool)
    for key, values in residuals.items():
        safe &= values <= defaults[key]
    safe &= translation_update_ema <= defaults["translation_update"]
    safe &= yaw_update_ema <= defaults["yaw_update"]
    return safe


def certified_lm_convergence(
    numerical_converged: bool,
    object_safe_mask: torch.Tensor,
) -> bool:
    """Require both numerical stagnation and a residual-safe scene."""
    if object_safe_mask.ndim != 1 or object_safe_mask.dtype != torch.bool:
        raise ValueError(
            "object_safe_mask must be a one-dimensional bool tensor"
        )
    return bool(
        numerical_converged
        and object_safe_mask.numel()
        and torch.all(object_safe_mask).item()
    )


def collision_connected_release_mask(
    collision_residuals: torch.Tensor,
    *,
    threshold: float = 1.0e-4,
) -> torch.Tensor:
    """Select only objects incident to an unsafe collision factor.

    ``active_set_object_residuals`` has already scattered every pairwise
    penetration residual onto both endpoints.  Thresholding that object-wise
    vector therefore gives the smallest collision-connected block set that
    can be released without falling back to a dense scene update.
    """
    if collision_residuals.ndim != 1:
        raise ValueError("collision_residuals must be one-dimensional")
    if not collision_residuals.is_floating_point():
        raise ValueError("collision_residuals must be floating point")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    return collision_residuals > threshold


def _clear_adam_rows_(
    optimizer: torch.optim.Optimizer,
    parameters: Sequence[torch.Tensor],
    row_mask: torch.Tensor,
) -> None:
    """Clear Adam momentum for frozen rows so they cannot drift on resume."""
    if not torch.any(row_mask):
        return
    with torch.no_grad():
        for parameter in parameters:
            state = optimizer.state.get(parameter, {})
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = state.get(key)
                if isinstance(value, torch.Tensor) and value.ndim:
                    value[row_mask] = 0


def _rescale_active_mean(
    active_mean: torch.Tensor,
    active_mass: int | float | torch.Tensor,
    full_mass: int | float | torch.Tensor,
) -> torch.Tensor:
    """Preserve the dense objective's denominator after factor pruning.

    Frozen--frozen factors have zero gradient, so their forward graph can be
    omitted.  The retained factor sum must still be divided by the *original*
    dense factor mass; otherwise every freeze event silently changes the
    relative loss weights and therefore the active objects' trajectory.
    """
    active = torch.as_tensor(
        active_mass,
        dtype=active_mean.dtype,
        device=active_mean.device,
    )
    full = torch.as_tensor(
        full_mass,
        dtype=active_mean.dtype,
        device=active_mean.device,
    )
    # Do not inspect CUDA scalars here: this helper runs for every factor on
    # every iteration, and a Python truth test would force a device
    # synchronization that can erase the router's computational savings.
    for name, value in (("active_mass", active_mass), ("full_mass", full_mass)):
        if not isinstance(value, torch.Tensor) and float(value) < 0:
            raise ValueError(f"{name} must be non-negative")
    return active_mean * active / torch.clamp(full, min=1.0)


def _active_child_pair_mask(
    pair_indices: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """Select hard-projection pairs whose mutated child is still active."""
    if pair_indices.numel() == 0:
        return torch.empty(
            (0,), dtype=torch.bool, device=pair_indices.device
        )
    return active_mask[pair_indices[:, 0]]


def optimize_semantic_stage(
    base_matrices: torch.Tensor,
    local_corners: torch.Tensor,
    collision_pairs: torch.Tensor,
    support_pairs: torch.Tensor,
    containment_pairs: torch.Tensor,
    fixed_support_indices: torch.Tensor,
    fixed_support_heights: torch.Tensor,
    plane_object_indices: torch.Tensor,
    plane_points: torch.Tensor,
    plane_normals: torch.Tensor,
    plane_orientation_mask: torch.Tensor,
    point_pairs: torch.Tensor,
    point_offsets: torch.Tensor,
    distance_pairs: torch.Tensor,
    distance_minimum: torch.Tensor,
    distance_maximum: torch.Tensor,
    align_pairs: torch.Tensor,
    align_offsets: torch.Tensor,
    *,
    footprint_hull_sizes: torch.Tensor | None = None,
    boundary_object_indices: torch.Tensor | None = None,
    boundary_points: torch.Tensor | None = None,
    boundary_normals: torch.Tensor | None = None,
    depth_observation_indices: torch.Tensor | None = None,
    depth_observed_boxes: torch.Tensor | None = None,
    depth_observed_depths: torch.Tensor | None = None,
    depth_observed_weights: torch.Tensor | None = None,
    depth_bbox_size_enabled: torch.Tensor | None = None,
    depth_world_to_camera: torch.Tensor | None = None,
    depth_image_size: torch.Tensor | None = None,
    depth_reference_centre_errors: torch.Tensor | None = None,
    depth_reference_size_errors: torch.Tensor | None = None,
    depth_reference_relative_errors: torch.Tensor | None = None,
    iterations: int = 250,
    learning_rate: float = 0.01,
    collision_weight: float = 1.0,
    contact_weight: float = 2.0,
    plane_weight: float = 2.0,
    orientation_weight: float = 0.25,
    containment_weight: float = 1.0,
    semantic_weight: float = 0.5,
    boundary_weight: float = 1.0,
    depth_reprojection_weight: float = 1.0,
    depth_centre_weight: float = 1.0,
    depth_size_weight: float = 0.25,
    depth_metric_weight: float = 1.0,
    depth_trust_region_weight: float = 1.0,
    depth_centre_margin_pixels: float = 2.0,
    depth_size_margin_log: float = 0.02,
    depth_relative_margin_log: float = 0.01,
    warm_start_weight: float = 0.01,
    restore_best_state: bool = False,
    optimize_yaw: bool = True,
    active_set_router: bool = False,
    active_set_checkpoints: Sequence[int] = (30, 100),
    active_set_thresholds: dict[str, float] | None = None,
    active_set_high_degree: int = 6,
    active_set_wake_multiplier: float = 1.5,
    solver: str = "adam",
    lm_initial_damping: float = 1e-2,
    lm_pcg_iterations: int = 12,
    lm_pcg_tolerance: float = 1e-3,
    lm_acceptance_threshold: float = 0.1,
    lm_gradient_tolerance: float = 1e-5,
    lm_relative_energy_tolerance: float = 1e-4,
    lm_patience: int = 3,
    lm_max_translation_step: float = 0.20,
    lm_max_yaw_step_degrees: float = 15.0,
    lm_max_relation_releases: int = 1,
    lm_collision_witness_weight: float = 25.0,
    sceneproof_shadow_residual_parity: bool = False,
    sceneproof_use_program_residuals: bool = False,
    sceneproof_residual_fallback: bool = True,
    sceneproof_factor_bindings: Sequence[dict[str, Any]] | None = None,
    sceneproof_object_ids: Sequence[str] | None = None,
    sceneproof_shadow_jacobian_ownership: bool = False,
    sceneproof_required_stable_linearizations: int = 2,
    sceneproof_full_so3_guarded_schur: bool = False,
    sceneproof_in_loop_guarded_schur: bool = False,
    sceneproof_warm_start_anchored_plane_translation: bool = False,
    sceneproof_plane_anchor_normal_limit_m: float = 0.02,
    sceneproof_plane_proxy_abstain_gap_m: float = 0.0,
    sceneproof_plane_attach_requires_witness: bool = False,
    sceneproof_plane_sibling_tangent_projection: bool = False,
    sceneproof_plane_sibling_max_shift_m: float = 0.35,
    sceneproof_plane_component_image_gauge: bool = False,
    sceneproof_image_observation_indices: torch.Tensor | None = None,
    sceneproof_image_observed_boxes: torch.Tensor | None = None,
    sceneproof_image_observed_depths: torch.Tensor | None = None,
    sceneproof_image_observed_weights: torch.Tensor | None = None,
    sceneproof_image_bbox_size_enabled: torch.Tensor | None = None,
    sceneproof_image_world_to_camera: torch.Tensor | None = None,
    sceneproof_image_size: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, list[dict[str, Any]]]:
    """Joint physics/plane/semantic stage before room-boundary projection.

    Primitive semantic penalties are averaged over all retained constraints,
    matching LayoutVLM's ``L_semantic`` construction. Vertical contact and
    support-footprint containment remain separate physical terms.
    """
    _validate_pose_batch(base_matrices)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    solver = solver.strip().lower()
    if solver not in {"adam", "scenelm", "v5_scenelm"}:
        raise ValueError(
            "solver must be 'adam', 'scenelm', or 'v5_scenelm'"
        )
    is_lm_solver = solver in {"scenelm", "v5_scenelm"}
    if sceneproof_shadow_jacobian_ownership:
        if solver != "v5_scenelm":
            raise ValueError(
                "SceneProof Jacobian ownership audit requires v5_scenelm"
            )
        if not sceneproof_factor_bindings or not sceneproof_object_ids:
            raise ValueError(
                "SceneProof Jacobian ownership audit requires factor bindings "
                "and ordered object IDs"
            )
        if len(sceneproof_object_ids) != int(base_matrices.shape[0]):
            raise ValueError("SceneProof object IDs must match the pose batch")
    if sceneproof_full_so3_guarded_schur:
        if not sceneproof_shadow_jacobian_ownership:
            raise ValueError(
                "full-SO(3) guarded Schur requires the Jacobian ownership gate"
            )
        if iterations < sceneproof_required_stable_linearizations:
            raise ValueError(
                "guarded Schur requires enough iterations to stabilize factors"
            )
    if sceneproof_plane_anchor_normal_limit_m < 0:
        raise ValueError("plane anchor normal trust limit must be non-negative")
    if sceneproof_plane_proxy_abstain_gap_m < 0:
        raise ValueError("plane proxy abstention gap must be non-negative")
    if sceneproof_plane_sibling_max_shift_m < 0:
        raise ValueError("plane sibling trust limit must be non-negative")
    if (
        sceneproof_in_loop_guarded_schur
        and not sceneproof_full_so3_guarded_schur
    ):
        raise ValueError(
            "in-loop guarded Schur requires the full-SO(3) guarded Schur gate"
        )
    if is_lm_solver:
        if active_set_router:
            raise ValueError(
                "SceneLM and the rejected active-set router are mutually exclusive"
            )
        if lm_initial_damping <= 0:
            raise ValueError("lm_initial_damping must be positive")
        if lm_pcg_iterations <= 0:
            raise ValueError("lm_pcg_iterations must be positive")
        if lm_pcg_tolerance <= 0:
            raise ValueError("lm_pcg_tolerance must be positive")
        if not 0 <= lm_acceptance_threshold < 1:
            raise ValueError("lm_acceptance_threshold must lie in [0, 1)")
        if lm_gradient_tolerance <= 0:
            raise ValueError("lm_gradient_tolerance must be positive")
        if lm_relative_energy_tolerance <= 0:
            raise ValueError("lm_relative_energy_tolerance must be positive")
        if lm_patience <= 0:
            raise ValueError("lm_patience must be positive")
        if lm_max_translation_step <= 0:
            raise ValueError("lm_max_translation_step must be positive")
        if lm_max_yaw_step_degrees <= 0:
            raise ValueError("lm_max_yaw_step_degrees must be positive")
        if lm_max_relation_releases < 0:
            raise ValueError("lm_max_relation_releases must be non-negative")
        if lm_collision_witness_weight <= 0:
            raise ValueError("lm_collision_witness_weight must be positive")
        # Accepted LM steps are monotonic under the factor residual objective.
        # Restoring an Adam-style sampled "best" state would discard the final
        # accepted/projected state and makes the trust-region audit ambiguous.
        restore_best_state = False
    checkpoints = tuple(int(value) for value in active_set_checkpoints)
    if active_set_router:
        if len(checkpoints) != 2 or checkpoints != tuple(sorted(set(checkpoints))):
            raise ValueError("active_set_checkpoints must contain two increasing values")
        if checkpoints[0] <= 0 or checkpoints[1] >= iterations:
            raise ValueError("active-set checkpoints must lie inside iterations")
        if active_set_high_degree < 0:
            raise ValueError("active_set_high_degree must be non-negative")
        if active_set_wake_multiplier < 1.0:
            raise ValueError("active_set_wake_multiplier must be at least 1")
        # The active factor set changes at each checkpoint, so scalar totals
        # are not comparable across the entire run.  A global best-state
        # restore would therefore be mathematically invalid for routed runs.
        restore_best_state = False
    boundary_arguments = (
        boundary_object_indices,
        boundary_points,
        boundary_normals,
    )
    if any(value is None for value in boundary_arguments) and not all(
        value is None for value in boundary_arguments
    ):
        raise ValueError(
            "boundary_object_indices, boundary_points, and boundary_normals "
            "must either all be provided or all be omitted"
        )
    if boundary_object_indices is None:
        boundary_object_indices = torch.empty(
            (0,), dtype=torch.long, device=base_matrices.device
        )
        boundary_points = base_matrices.new_zeros((0, 2))
        boundary_normals = base_matrices.new_zeros((0, 2))
    depth_arguments = (
        depth_observation_indices,
        depth_observed_boxes,
        depth_observed_depths,
        depth_observed_weights,
        depth_bbox_size_enabled,
        depth_world_to_camera,
        depth_image_size,
        depth_reference_centre_errors,
        depth_reference_size_errors,
        depth_reference_relative_errors,
    )
    if any(value is None for value in depth_arguments) and not all(
        value is None for value in depth_arguments
    ):
        raise ValueError(
            "all depth-aware reprojection tensors must be provided together"
        )
    if depth_observation_indices is None:
        depth_observation_indices = torch.empty(
            (0,), dtype=torch.long, device=base_matrices.device
        )
        depth_observed_boxes = base_matrices.new_zeros((0, 4))
        depth_observed_depths = base_matrices.new_zeros((0,))
        depth_observed_weights = base_matrices.new_zeros((0,))
        depth_bbox_size_enabled = torch.empty(
            (0,), dtype=torch.bool, device=base_matrices.device
        )
        depth_world_to_camera = torch.eye(
            4, dtype=base_matrices.dtype, device=base_matrices.device
        )
        depth_image_size = base_matrices.new_tensor(
            [1.0, 1.0, 1.0, 1.0]
        )
        depth_reference_centre_errors = base_matrices.new_zeros((0,))
        depth_reference_size_errors = base_matrices.new_zeros((0,))
        depth_reference_relative_errors = base_matrices.new_zeros((0,))

    yaw_delta, translation = initialize_pose_variables(base_matrices)
    relation_coordinates = None
    relation_parameters = None
    relation_active_objects = None
    relation_stable_steps = None
    relation_freeze_count = 0
    relation_wakeup_count = 0
    relation_active_step_total = 0
    relation_release_count = 0
    relation_released_objects: set[int] = set()
    relation_release_iterations: list[int] = []
    collision_witness_pairs = torch.empty(
        (0, 2), dtype=torch.long, device=base_matrices.device
    )
    collision_witness_axes = base_matrices.new_zeros((0, 2))
    sceneproof_shadow_checks = 0
    sceneproof_shadow_max_abs_error = 0.0
    sceneproof_program_residual_selections = 0
    sceneproof_residual_fallbacks = 0
    sceneproof_jacobian_audits: list[dict[str, Any]] = []
    sceneproof_guarded_schur_audit: dict[str, Any] | None = None
    sceneproof_guarded_pose_override: torch.Tensor | None = None
    containment_projection_abstentions: list[dict[str, int | str]] = []
    plane_proxy_abstain_indices = torch.empty(
        (0,), dtype=torch.long, device=base_matrices.device
    )
    plane_proxy_abstain_initial_gaps = base_matrices.new_zeros((0,))
    sceneproof_stability_tracker = (
        LinearizationStabilityTracker(
            required_consecutive=sceneproof_required_stable_linearizations
        )
        if sceneproof_shadow_jacobian_ownership
        else None
    )
    if solver == "v5_scenelm":
        # The chart is compiled after the initial factor audit below. Objects
        # whose predicted relation is incompatible with the warm start remain
        # expressive world-space blocks rather than being irreversibly locked
        # to an incorrect support or architectural plane.
        pass
    optimizer_parameters = [translation]
    if optimize_yaw:
        optimizer_parameters.insert(0, yaw_delta)
    else:
        # Depth refinement may be used strictly as a translation correction
        # on top of a frozen, already-evaluated S4 rotation solution.
        yaw_delta = yaw_delta.detach()
    optimizer = None
    scheduler = None
    if solver == "adam":
        optimizer = torch.optim.Adam(optimizer_parameters, lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, iterations // 4),
            gamma=0.5,
        )
    history: list[dict[str, Any]] = []
    best_total = float("inf")
    best_iteration = 0
    best_yaw_delta: torch.Tensor | None = None
    best_translation: torch.Tensor | None = None
    object_count = base_matrices.shape[0]
    allocated_budget = torch.full(
        (object_count,),
        iterations,
        dtype=torch.long,
        device=base_matrices.device,
    )
    translation_update_ema = base_matrices.new_zeros((object_count,))
    yaw_update_ema = base_matrices.new_zeros((object_count,))
    active_step_total = 0
    executed_iterations = 0
    wakeup_count = 0
    freeze_count_30 = 0
    freeze_count_100 = 0
    constraint_degree = torch.zeros(
        object_count, dtype=torch.long, device=base_matrices.device
    )
    # Protect hubs in the *structural* scene graph.  Broad-phase collision
    # candidate degree is intentionally excluded: dense rooms give ordinary
    # objects many transient collision edges, and counting those edges would
    # permanently disable routing for most of the scene.  Collision safety is
    # instead enforced by residual gating and two-endpoint wake-up.
    for factor_indices in (
        support_pairs,
        containment_pairs,
        distance_pairs,
        align_pairs,
        point_pairs,
    ):
        if factor_indices.numel():
            constraint_degree.index_add_(
                0,
                factor_indices.reshape(-1),
                torch.ones(
                    factor_indices.numel(),
                    dtype=torch.long,
                    device=base_matrices.device,
                ),
            )
    protected_objects = constraint_degree >= active_set_high_degree

    def collect_residuals(
        pose_matrices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _, collision_values = oriented_penetration_loss(
            pose_matrices,
            local_corners,
            collision_pairs,
            footprint_hull_sizes,
        )
        _, check_contact_gaps = support_contact_loss(
            pose_matrices,
            local_corners,
            support_pairs,
            fixed_support_indices,
            fixed_support_heights,
        )
        (
            _,
            _,
            check_plane_gaps,
            check_alignment_errors,
        ) = fixed_plane_loss(
            pose_matrices,
            local_corners,
            plane_object_indices,
            plane_points,
            plane_normals,
            plane_orientation_mask,
        )
        _, check_containment_errors = support_planar_containment_loss(
            pose_matrices,
            local_corners,
            containment_pairs,
            footprint_hull_sizes,
        )
        _, _, check_distance_penalties = distance_interval_loss(
            pose_matrices,
            distance_pairs,
            distance_minimum,
            distance_maximum,
        )
        _, check_align_errors = align_with_loss(
            pose_matrices, align_pairs, align_offsets
        )
        _, check_point_errors = point_towards_loss(
            pose_matrices, local_corners, point_pairs, point_offsets
        )
        _, check_boundary_errors = room_boundary_loss(
            pose_matrices,
            local_corners,
            boundary_object_indices,
            boundary_points,
            boundary_normals,
        )
        (
            _,
            check_depth_centre_errors,
            check_depth_size_errors,
            check_depth_relative_errors,
        ) = depth_aware_reprojection_loss(
            pose_matrices,
            local_corners,
            depth_observation_indices,
            depth_observed_boxes,
            depth_observed_depths,
            depth_observed_weights,
            depth_bbox_size_enabled,
            depth_world_to_camera,
            depth_image_size,
            centre_weight=depth_centre_weight,
            size_weight=depth_size_weight,
            metric_depth_weight=depth_metric_weight,
        )
        return active_set_object_residuals(
            object_count,
            collision_pairs=collision_pairs,
            collision_values=collision_values,
            support_pairs=support_pairs,
            contact_gaps=check_contact_gaps,
            fixed_support_indices=fixed_support_indices,
            plane_object_indices=plane_object_indices,
            plane_gaps=check_plane_gaps,
            plane_alignment_errors=check_alignment_errors,
            containment_pairs=containment_pairs,
            containment_errors=check_containment_errors,
            distance_pairs=distance_pairs,
            distance_penalties=check_distance_penalties,
            align_pairs=align_pairs,
            align_errors=check_align_errors,
            point_pairs=point_pairs,
            point_errors=check_point_errors,
            boundary_object_indices=boundary_object_indices,
            boundary_errors=check_boundary_errors,
            depth_observation_indices=depth_observation_indices,
            depth_centre_errors=check_depth_centre_errors,
            depth_size_errors=check_depth_size_errors,
            depth_relative_errors=check_depth_relative_errors,
        )

    if solver == "v5_scenelm":
        relation_compatible = torch.ones(
            object_count, dtype=torch.bool, device=base_matrices.device
        )
        _, initial_contact_gaps = support_contact_loss(
            base_matrices,
            local_corners,
            support_pairs,
            fixed_support_indices,
            fixed_support_heights,
        )
        pair_contact_count = int(support_pairs.shape[0])
        if pair_contact_count:
            relation_compatible[support_pairs[:, 0]] &= (
                initial_contact_gaps[:pair_contact_count].abs() <= 0.10
            )
        if fixed_support_indices.numel():
            relation_compatible[fixed_support_indices] &= (
                initial_contact_gaps[pair_contact_count:].abs() <= 0.10
            )
        (
            _,
            _,
            initial_plane_gaps,
            initial_orientation_errors,
        ) = fixed_plane_loss(
            base_matrices,
            local_corners,
            plane_object_indices,
            plane_points,
            plane_normals,
            plane_orientation_mask,
        )
        if plane_object_indices.numel():
            if sceneproof_plane_attach_requires_witness:
                plane_proxy_abstain_indices = torch.unique(
                    plane_object_indices
                )
                plane_proxy_abstain_initial_gaps = (
                    initial_plane_gaps.detach()
                )
            elif sceneproof_plane_proxy_abstain_gap_m > 0:
                proxy_abstain_mask = (
                    initial_plane_gaps.abs()
                    > float(sceneproof_plane_proxy_abstain_gap_m)
                )
                plane_proxy_abstain_indices = torch.unique(
                    plane_object_indices[proxy_abstain_mask]
                )
                plane_proxy_abstain_initial_gaps = initial_plane_gaps[
                    proxy_abstain_mask
                ].detach()
            relation_compatible[plane_object_indices] &= (
                (initial_plane_gaps.abs() <= 0.10)
                & (
                    initial_orientation_errors.abs()
                    <= 1.0 - math.cos(math.radians(30.0))
                )
            )
        _, initial_containment_errors = support_planar_containment_loss(
            base_matrices,
            local_corners,
            containment_pairs,
            footprint_hull_sizes,
        )
        if containment_pairs.shape[0]:
            relation_compatible[containment_pairs[:, 0]] &= (
                torch.sqrt(
                    torch.clamp_min(initial_containment_errors, 0.0)
                )
                <= 0.10
            )
        relaxed_objects = torch.nonzero(
            ~relation_compatible, as_tuple=False
        ).reshape(-1)
        relation_coordinates = compile_relation_coordinates(
            base_matrices,
            support_pairs,
            fixed_support_indices,
            plane_object_indices,
            plane_normals,
            optimise_yaw=optimize_yaw,
            free_object_indices=relaxed_objects.detach().cpu().tolist(),
            warm_start_anchored_plane_translation=(
                sceneproof_warm_start_anchored_plane_translation
            ),
        )
        relation_parameters = relation_coordinates.zero_parameters()
        relation_active_objects = torch.ones(
            base_matrices.shape[0],
            dtype=torch.bool,
            device=base_matrices.device,
        )
        relation_active_objects[plane_proxy_abstain_indices] = False
        relation_stable_steps = torch.zeros(
            base_matrices.shape[0],
            dtype=torch.long,
            device=base_matrices.device,
        )

    active_set_revision = 0
    factor_cache_revision = -1
    factor_cache: dict[str, torch.Tensor] = {}
    active_horizon = iterations
    router_has_frozen = False
    lm_damping = float(lm_initial_damping)
    lm_small_reduction_count = 0
    lm_accepted_steps = 0
    lm_rejected_steps = 0
    lm_last_diagnostics: dict[str, float] = {}
    lm_should_stop = False

    def pack_pose_parameters(
        current_yaw: torch.Tensor,
        current_translation: torch.Tensor,
    ) -> torch.Tensor:
        if solver == "v5_scenelm":
            if relation_parameters is None:
                raise RuntimeError("v5 SceneLM relation parameters are missing")
            return relation_parameters
        flattened_translation = current_translation.reshape(-1)
        if optimize_yaw:
            return torch.cat((current_yaw, flattened_translation))
        return flattened_translation

    def unpack_pose_parameters(
        flattened: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if flattened.ndim != 1:
            raise ValueError("flattened SceneLM pose must be one-dimensional")
        if solver == "v5_scenelm":
            if relation_coordinates is None:
                raise RuntimeError("v5 SceneLM relation chart is missing")
            return relation_coordinates.decode(flattened)
        if optimize_yaw:
            expected = object_count * 4
            if flattened.numel() != expected:
                raise ValueError(
                    f"SceneLM expected {expected} parameters; "
                    f"got {flattened.numel()}"
                )
            unpacked_yaw = flattened[:object_count]
            unpacked_translation = flattened[object_count:].reshape(
                object_count, 3
            )
        else:
            expected = object_count * 3
            if flattened.numel() != expected:
                raise ValueError(
                    f"SceneLM expected {expected} parameters; "
                    f"got {flattened.numel()}"
                )
            unpacked_yaw = yaw_delta
            unpacked_translation = flattened.reshape(object_count, 3)
        return unpacked_yaw, unpacked_translation

    def dense_factor_residuals(flattened: torch.Tensor) -> torch.Tensor:
        """Return residuals whose squared norm is the dense LayoutVLM loss.

        Linear hinge/overlap terms are represented by their square roots.
        The tiny additive constant is independent of pose, avoids an infinite
        derivative at an exact zero residual, and cancels in LM reduction
        ratios because factor cardinality is fixed.
        """
        current_yaw, current_translation = unpack_pose_parameters(flattened)
        if solver == "v5_scenelm":
            if relation_coordinates is None:
                raise RuntimeError("v5 SceneLM relation chart is missing")
            current_pose = relation_coordinates.pose_matrices(flattened)
        else:
            current_pose = reproject_pose_matrices(
                base_matrices,
                current_yaw,
                current_translation,
            )
        _, collision_values = oriented_penetration_loss(
            current_pose,
            local_corners,
            collision_pairs,
            footprint_hull_sizes,
        )
        witness_values = (
            collision_witness_residuals(
                current_pose,
                local_corners,
                collision_witness_pairs,
                collision_witness_axes,
                footprint_hull_sizes,
            )
            if collision_witness_pairs.shape[0]
            else flattened[:0]
        )
        _, contact_values = support_contact_loss(
            current_pose,
            local_corners,
            support_pairs,
            fixed_support_indices,
            fixed_support_heights,
        )
        (
            _,
            _,
            plane_values,
            orientation_values,
        ) = fixed_plane_loss(
            current_pose,
            local_corners,
            plane_object_indices,
            plane_points,
            plane_normals,
            plane_orientation_mask,
        )
        _, containment_values = support_planar_containment_loss(
            current_pose,
            local_corners,
            containment_pairs,
            footprint_hull_sizes,
        )
        _, _, distance_values = distance_interval_loss(
            current_pose,
            distance_pairs,
            distance_minimum,
            distance_maximum,
        )
        _, align_values = align_with_loss(
            current_pose, align_pairs, align_offsets
        )
        _, point_values = point_towards_loss(
            current_pose, local_corners, point_pairs, point_offsets
        )
        _, boundary_values = room_boundary_loss(
            current_pose,
            local_corners,
            boundary_object_indices,
            boundary_points,
            boundary_normals,
        )
        (
            current_depth,
            current_depth_centre,
            current_depth_size,
            current_depth_relative,
        ) = depth_aware_reprojection_loss(
            current_pose,
            local_corners,
            depth_observation_indices,
            depth_observed_boxes,
            depth_observed_depths,
            depth_observed_weights,
            depth_bbox_size_enabled,
            depth_world_to_camera,
            depth_image_size,
            centre_weight=depth_centre_weight,
            size_weight=depth_size_weight,
            metric_depth_weight=depth_metric_weight,
        )
        current_depth_trust, _, _, _ = no_harm_reprojection_penalty(
            current_depth_centre,
            current_depth_size,
            current_depth_relative,
            depth_reference_centre_errors,
            depth_reference_size_errors,
            depth_reference_relative_errors,
            depth_observed_weights,
            centre_margin_pixels=depth_centre_margin_pixels,
            size_margin_log=depth_size_margin_log,
            depth_margin_log=depth_relative_margin_log,
        )

        # Python scalar broadcasting is FuncTorch-safe. Constructing a tensor
        # through ``flattened.new_tensor`` inside a vjp/jvp transform fails in
        # Blender's bundled PyTorch with a FuncTorchGradWrapper dispatch-key
        # error.
        epsilon = 1e-12
        pieces: list[torch.Tensor] = []

        def append_squared(
            values: torch.Tensor,
            weight: float,
            mass: int,
        ) -> None:
            if values.numel() and weight > 0:
                pieces.append(
                    values.reshape(-1)
                    * math.sqrt(float(weight) / max(int(mass), 1))
                )

        def append_linear(
            values: torch.Tensor,
            weight: float,
            mass: int,
        ) -> None:
            if values.numel() and weight > 0:
                scaled = (
                    torch.clamp(values.reshape(-1), min=0.0)
                    * (float(weight) / max(int(mass), 1))
                )
                pieces.append(torch.sqrt(scaled + epsilon))

        append_linear(
            collision_values,
            collision_weight,
            collision_pairs.shape[0],
        )
        append_squared(
            witness_values,
            lm_collision_witness_weight,
            collision_witness_pairs.shape[0],
        )
        append_squared(
            contact_values,
            contact_weight,
            support_pairs.shape[0] + fixed_support_indices.shape[0],
        )
        append_squared(
            plane_values,
            plane_weight,
            plane_object_indices.shape[0],
        )
        valid_orientation = orientation_values[plane_orientation_mask]
        append_squared(
            valid_orientation,
            orientation_weight,
            valid_orientation.numel(),
        )
        append_linear(
            containment_values,
            containment_weight,
            containment_pairs.shape[0],
        )
        semantic_mass = (
            distance_pairs.shape[0]
            + align_pairs.shape[0]
            + point_pairs.shape[0]
        )
        append_linear(distance_values, semantic_weight, semantic_mass)
        append_linear(align_values, semantic_weight, semantic_mass)
        append_linear(point_values, semantic_weight, semantic_mass)
        append_squared(
            boundary_values,
            boundary_weight,
            boundary_object_indices.shape[0],
        )
        if depth_observation_indices.numel():
            if depth_reprojection_weight > 0:
                pieces.append(
                    torch.sqrt(
                        torch.clamp(current_depth, min=0.0)
                        * depth_reprojection_weight
                        + epsilon
                    ).reshape(1)
                )
            if depth_trust_region_weight > 0:
                pieces.append(
                    torch.sqrt(
                        torch.clamp(current_depth_trust, min=0.0)
                        * depth_trust_region_weight
                        + epsilon
                    ).reshape(1)
                )
        if warm_start_weight > 0:
            warm_scale = math.sqrt(
                float(warm_start_weight) / max(object_count, 1)
            )
            if optimize_yaw:
                pieces.append(current_yaw * warm_scale)
            pieces.append(
                (
                    current_translation - base_matrices[:, :3, 3]
                ).reshape(-1)
                * (warm_scale / math.sqrt(3.0))
            )
        if not pieces:
            # Preserve a differentiable zero residual for degenerate tests.
            legacy_residuals = flattened[:1] * 0.0
        else:
            legacy_residuals = torch.cat(pieces)
        if (
            sceneproof_shadow_residual_parity
            or sceneproof_use_program_residuals
        ):
            shadow_residuals = assemble_program_shadow_residuals(
                flattened=flattened,
                collision_values=collision_values,
                collision_weight=collision_weight,
                collision_mass=collision_pairs.shape[0],
                witness_values=witness_values,
                witness_weight=lm_collision_witness_weight,
                witness_mass=collision_witness_pairs.shape[0],
                contact_values=contact_values,
                contact_weight=contact_weight,
                contact_mass=(
                    support_pairs.shape[0] + fixed_support_indices.shape[0]
                ),
                plane_values=plane_values,
                plane_weight=plane_weight,
                plane_mass=plane_object_indices.shape[0],
                orientation_values=valid_orientation,
                orientation_weight=orientation_weight,
                containment_values=containment_values,
                containment_weight=containment_weight,
                containment_mass=containment_pairs.shape[0],
                distance_values=distance_values,
                align_values=align_values,
                point_values=point_values,
                semantic_weight=semantic_weight,
                semantic_mass=semantic_mass,
                boundary_values=boundary_values,
                boundary_weight=boundary_weight,
                boundary_mass=boundary_object_indices.shape[0],
                current_depth=current_depth,
                current_depth_trust=current_depth_trust,
                depth_observation_count=depth_observation_indices.numel(),
                depth_reprojection_weight=depth_reprojection_weight,
                depth_trust_region_weight=depth_trust_region_weight,
                current_yaw=current_yaw,
                current_translation=current_translation,
                base_matrices=base_matrices,
                optimize_yaw=optimize_yaw,
                warm_start_weight=warm_start_weight,
            )
            parity = residual_parity(legacy_residuals, shadow_residuals)
            nonlocal sceneproof_shadow_checks
            nonlocal sceneproof_shadow_max_abs_error
            nonlocal sceneproof_program_residual_selections
            nonlocal sceneproof_residual_fallbacks
            sceneproof_shadow_checks += 1
            sceneproof_shadow_max_abs_error = max(
                sceneproof_shadow_max_abs_error,
                float(parity["max_abs_error"]),
            )
            if not parity["passed"]:
                if sceneproof_use_program_residuals and sceneproof_residual_fallback:
                    sceneproof_residual_fallbacks += 1
                    return legacy_residuals
                raise RuntimeError(
                    "SceneProof shadow residual parity failed: " f"{parity}"
                )
            if sceneproof_use_program_residuals:
                sceneproof_program_residual_selections += 1
                return shadow_residuals
        return legacy_residuals

    def run_sceneproof_jacobian_ownership_audit(
        parameters: torch.Tensor,
    ) -> dict[str, Any]:
        if (
            relation_coordinates is None
            or sceneproof_stability_tracker is None
            or sceneproof_factor_bindings is None
            or sceneproof_object_ids is None
        ):
            raise RuntimeError("SceneProof Jacobian audit state is missing")
        orientation_object_indices = plane_object_indices[
            plane_orientation_mask
        ]
        bindings = build_residual_slice_bindings(
            factor_bindings=list(sceneproof_factor_bindings),
            object_ids=tuple(sceneproof_object_ids),
            collision_pairs=collision_pairs,
            collision_witness_pairs=collision_witness_pairs,
            support_count=(
                int(support_pairs.shape[0])
                + int(fixed_support_indices.shape[0])
            ),
            plane_count=int(plane_object_indices.shape[0]),
            orientation_object_indices=orientation_object_indices,
            containment_count=int(containment_pairs.shape[0]),
            distance_count=int(distance_pairs.shape[0]),
            align_count=int(align_pairs.shape[0]),
            point_count=int(point_pairs.shape[0]),
            boundary_object_indices=boundary_object_indices,
            depth_observation_indices=depth_observation_indices,
            collision_weight=collision_weight,
            witness_weight=lm_collision_witness_weight,
            contact_weight=contact_weight,
            plane_weight=plane_weight,
            orientation_weight=orientation_weight,
            containment_weight=containment_weight,
            semantic_weight=semantic_weight,
            boundary_weight=boundary_weight,
            depth_reprojection_weight=depth_reprojection_weight,
            depth_trust_region_weight=depth_trust_region_weight,
            optimize_yaw=optimize_yaw,
            warm_start_weight=warm_start_weight,
        )
        audit_parameters = parameters.detach()
        residuals = dense_factor_residuals(audit_parameters)
        jacobian = torch.func.jacrev(dense_factor_residuals)(audit_parameters)
        object_slices = {
            block.object_index: (block.parameter_slice,)
            for block in relation_coordinates.blocks
        }
        dependencies: dict[int, tuple[int, ...]] = {}
        for block in relation_coordinates.blocks:
            ancestors: list[int] = []
            parent = int(block.parent_index)
            while parent >= 0:
                ancestors.append(parent)
                parent = int(relation_coordinates.blocks[parent].parent_index)
            dependencies[block.object_index] = tuple(ancestors)
        report = audit_jacobian_block_ownership(
            residuals=residuals,
            jacobian=jacobian,
            bindings=bindings,
            object_parameter_slices=object_slices,
            object_dependencies=dependencies,
        )
        block_slices = {
            f"object:{block.object_index}": block.parameter_slice
            for block in relation_coordinates.blocks
        }
        linearized_factors: list[LinearizedFactor] = []
        for binding in bindings:
            owners: set[int] = set()
            for object_index in binding.declared_object_indices:
                owners.add(int(object_index))
                owners.update(
                    int(value)
                    for value in dependencies.get(object_index, ())
                )
            factor_rows = jacobian[binding.residual_slice]
            linearized_factors.append(
                LinearizedFactor(
                    factor_id=binding.factor_id,
                    residual=residuals[binding.residual_slice],
                    jacobians={
                        f"object:{object_index}": factor_rows[
                            :, relation_coordinates.blocks[
                                object_index
                            ].parameter_slice
                        ]
                        for object_index in sorted(owners)
                    },
                    object_indices=tuple(sorted(owners)),
                )
            )
        block_normal, block_gradient, assembly_diagnostics = (
            assemble_normal_system(
                parameter_count=int(jacobian.shape[1]),
                slices=block_slices,
                factors=linearized_factors,
                damping=0.0,
                dtype=jacobian.dtype,
                device=jacobian.device,
            )
        )
        dense_normal = jacobian.transpose(0, 1) @ jacobian
        dense_gradient = jacobian.transpose(0, 1) @ residuals
        normal_error = float(
            (block_normal - dense_normal).detach().abs().amax().item()
        )
        gradient_error = float(
            (block_gradient - dense_gradient).detach().abs().amax().item()
        )
        if jacobian.dtype in {torch.float16, torch.bfloat16, torch.float32}:
            absolute_tolerance = 5e-5
            relative_tolerance = 5e-5
        else:
            absolute_tolerance = 1e-9
            relative_tolerance = 1e-9
        normal_limit = absolute_tolerance + relative_tolerance * float(
            dense_normal.detach().abs().amax().item()
        )
        gradient_limit = absolute_tolerance + relative_tolerance * float(
            dense_gradient.detach().abs().amax().item()
        )
        normal_parity_passed = bool(
            normal_error <= normal_limit and gradient_error <= gradient_limit
        )
        report["normal_system_parity"] = {
            "passed": normal_parity_passed,
            "normal_max_abs_error": normal_error,
            "gradient_max_abs_error": gradient_error,
            "normal_tolerance": normal_limit,
            "gradient_tolerance": gradient_limit,
            "assembly": assembly_diagnostics,
        }
        report["passed"] = bool(report["passed"] and normal_parity_passed)
        all_factor_ids = [binding.factor_id for binding in bindings]
        stable_active = sceneproof_stability_tracker.update(
            report["active_factor_ids"], all_factor_ids
        )
        stable_inactive = (
            sceneproof_stability_tracker.stable_inactive_factor_ids()
        )
        unstable = set(sceneproof_stability_tracker.unstable_factor_ids())
        stable_active_set = set(stable_active)
        stable_inactive_set = set(stable_inactive)
        eligible: list[int] = []
        rejected: dict[int, str] = {}
        for object_index in relation_coordinates.leaf_object_indices:
            parent = int(
                relation_coordinates.blocks[object_index].parent_index
            )
            incident = [
                binding
                for binding in bindings
                if object_index in binding.declared_object_indices
            ]
            unresolved = [
                binding.factor_id
                for binding in incident
                if binding.factor_id in unstable
                or (
                    binding.factor_id not in stable_active_set
                    and binding.factor_id not in stable_inactive_set
                )
            ]
            if unresolved:
                rejected[object_index] = (
                    "unstable_factors:" + ",".join(sorted(unresolved))
                )
                continue
            cross_edges = [
                binding.factor_id
                for binding in incident
                if binding.factor_id in stable_active_set
                and not set(binding.declared_object_indices).issubset(
                    {object_index, parent}
                )
            ]
            if cross_edges:
                rejected[object_index] = (
                    "active_cross_factors:" + ",".join(sorted(cross_edges))
                )
                continue
            eligible.append(object_index)
        report["stable_active_factor_ids"] = list(stable_active)
        report["stable_inactive_factor_ids"] = list(stable_inactive)
        report["unstable_factor_ids"] = sorted(unstable)
        report["eligible_leaf_translation_objects"] = eligible
        report["rejected_leaf_translation_objects"] = rejected
        report["rotation_parameters_eliminated"] = 0
        report["rotation_policy"] = "all_rotations_retained_in_root_system"
        report["residual_slices"] = [
            {
                "factor_id": binding.factor_id,
                "channel": binding.channel,
                "start": binding.start,
                "stop": binding.stop,
                "declared_object_indices": list(
                    binding.declared_object_indices
                ),
                "collision_pair": (
                    list(binding.collision_pair)
                    if binding.collision_pair is not None
                    else None
                ),
            }
            for binding in bindings
        ]
        # Per-factor dense values make the placement JSON unnecessarily large;
        # the ownership/leakage summary and stable factor IDs are sufficient
        # for the fail-closed gate.
        report.pop("per_factor", None)
        return report

    def run_full_so3_guarded_schur_trial(
        incumbent_pose: torch.Tensor,
        stable_audit: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run one fail-closed SO(3) Schur trial around an incumbent pose."""
        if sceneproof_factor_bindings is None or sceneproof_object_ids is None:
            raise RuntimeError("guarded Schur requires factor bindings")
        if relation_coordinates is None:
            raise RuntimeError("guarded Schur requires the audited relation chart")
        chart = compile_full_so3_relation_coordinates(
            incumbent_pose.detach(),
            support_pairs,
            fixed_support_indices,
            plane_object_indices,
            plane_normals,
            free_object_indices=relation_coordinates.relaxed_object_indices,
            warm_start_anchored_plane_translation=(
                sceneproof_warm_start_anchored_plane_translation
            ),
        )

        def residual_function(parameters: torch.Tensor) -> torch.Tensor:
            pose = chart.pose_matrices(parameters)
            _, collision_values = oriented_penetration_loss(
                pose, local_corners, collision_pairs, footprint_hull_sizes
            )
            witness_values = (
                collision_witness_residuals(
                    pose,
                    local_corners,
                    collision_witness_pairs,
                    collision_witness_axes,
                    footprint_hull_sizes,
                )
                if collision_witness_pairs.shape[0]
                else parameters[:0]
            )
            _, contact_values = support_contact_loss(
                pose,
                local_corners,
                support_pairs,
                fixed_support_indices,
                fixed_support_heights,
            )
            _, _, plane_values, orientation_values = fixed_plane_loss(
                pose,
                local_corners,
                plane_object_indices,
                plane_points,
                plane_normals,
                plane_orientation_mask,
            )
            _, containment_values = support_planar_containment_loss(
                pose,
                local_corners,
                containment_pairs,
                footprint_hull_sizes,
            )
            _, _, distance_values = distance_interval_loss(
                pose, distance_pairs, distance_minimum, distance_maximum
            )
            _, align_values = align_with_loss(pose, align_pairs, align_offsets)
            _, point_values = point_towards_loss(
                pose, local_corners, point_pairs, point_offsets
            )
            _, boundary_values = room_boundary_loss(
                pose,
                local_corners,
                boundary_object_indices,
                boundary_points,
                boundary_normals,
            )
            (
                current_depth,
                depth_centre,
                depth_size,
                depth_relative,
            ) = depth_aware_reprojection_loss(
                pose,
                local_corners,
                depth_observation_indices,
                depth_observed_boxes,
                depth_observed_depths,
                depth_observed_weights,
                depth_bbox_size_enabled,
                depth_world_to_camera,
                depth_image_size,
                centre_weight=depth_centre_weight,
                size_weight=depth_size_weight,
                metric_depth_weight=depth_metric_weight,
            )
            current_depth_trust, _, _, _ = no_harm_reprojection_penalty(
                depth_centre,
                depth_size,
                depth_relative,
                depth_reference_centre_errors,
                depth_reference_size_errors,
                depth_reference_relative_errors,
                depth_observed_weights,
                centre_margin_pixels=depth_centre_margin_pixels,
                size_margin_log=depth_size_margin_log,
                depth_margin_log=depth_relative_margin_log,
            )
            factor_residuals = assemble_program_shadow_residuals(
                flattened=parameters,
                collision_values=collision_values,
                collision_weight=collision_weight,
                collision_mass=collision_pairs.shape[0],
                witness_values=witness_values,
                witness_weight=lm_collision_witness_weight,
                witness_mass=collision_witness_pairs.shape[0],
                contact_values=contact_values,
                contact_weight=contact_weight,
                contact_mass=(
                    support_pairs.shape[0] + fixed_support_indices.shape[0]
                ),
                plane_values=plane_values,
                plane_weight=plane_weight,
                plane_mass=plane_object_indices.shape[0],
                orientation_values=orientation_values[plane_orientation_mask],
                orientation_weight=orientation_weight,
                containment_values=containment_values,
                containment_weight=containment_weight,
                containment_mass=containment_pairs.shape[0],
                distance_values=distance_values,
                align_values=align_values,
                point_values=point_values,
                semantic_weight=semantic_weight,
                semantic_mass=(
                    distance_pairs.shape[0]
                    + align_pairs.shape[0]
                    + point_pairs.shape[0]
                ),
                boundary_values=boundary_values,
                boundary_weight=boundary_weight,
                boundary_mass=boundary_object_indices.shape[0],
                current_depth=current_depth,
                current_depth_trust=current_depth_trust,
                depth_observation_count=depth_observation_indices.numel(),
                depth_reprojection_weight=depth_reprojection_weight,
                depth_trust_region_weight=depth_trust_region_weight,
                current_yaw=parameters[:0],
                current_translation=incumbent_pose[:, :3, 3],
                base_matrices=incumbent_pose,
                optimize_yaw=False,
                warm_start_weight=0.0,
            )
            if warm_start_weight <= 0:
                return factor_residuals
            parameter_scale = math.sqrt(
                float(warm_start_weight) / max(chart.parameter_count, 1)
            )
            return torch.cat((factor_residuals, parameters * parameter_scale))

        def component_values(parameters: torch.Tensor) -> dict[str, float]:
            pose = chart.pose_matrices(parameters)
            collision_value, collision_rows = oriented_penetration_loss(
                pose, local_corners, collision_pairs, footprint_hull_sizes
            )
            contact_value, _ = support_contact_loss(
                pose,
                local_corners,
                support_pairs,
                fixed_support_indices,
                fixed_support_heights,
            )
            plane_value, orientation_value, _, _ = fixed_plane_loss(
                pose,
                local_corners,
                plane_object_indices,
                plane_points,
                plane_normals,
                plane_orientation_mask,
            )
            containment_value, _ = support_planar_containment_loss(
                pose,
                local_corners,
                containment_pairs,
                footprint_hull_sizes,
            )
            distance_value, _, _ = distance_interval_loss(
                pose, distance_pairs, distance_minimum, distance_maximum
            )
            align_value, _ = align_with_loss(pose, align_pairs, align_offsets)
            point_value, _ = point_towards_loss(
                pose, local_corners, point_pairs, point_offsets
            )
            boundary_value, _ = room_boundary_loss(
                pose,
                local_corners,
                boundary_object_indices,
                boundary_points,
                boundary_normals,
            )
            depth_value, depth_centre, depth_size, depth_relative = (
                depth_aware_reprojection_loss(
                    pose,
                    local_corners,
                    depth_observation_indices,
                    depth_observed_boxes,
                    depth_observed_depths,
                    depth_observed_weights,
                    depth_bbox_size_enabled,
                    depth_world_to_camera,
                    depth_image_size,
                    centre_weight=depth_centre_weight,
                    size_weight=depth_size_weight,
                    metric_depth_weight=depth_metric_weight,
                )
            )
            depth_trust, _, _, _ = no_harm_reprojection_penalty(
                depth_centre,
                depth_size,
                depth_relative,
                depth_reference_centre_errors,
                depth_reference_size_errors,
                depth_reference_relative_errors,
                depth_observed_weights,
                centre_margin_pixels=depth_centre_margin_pixels,
                size_margin_log=depth_size_margin_log,
                depth_margin_log=depth_relative_margin_log,
            )
            return {
                "collision": float(collision_value.detach().item()),
                "collision_max": float(
                    collision_rows.detach().amax().item()
                    if collision_rows.numel()
                    else 0.0
                ),
                "support_contact": float(contact_value.detach().item()),
                "support_containment": float(
                    containment_value.detach().item()
                ),
                "plane": float(plane_value.detach().item()),
                "orientation": float(orientation_value.detach().item()),
                "semantic_distance": float(distance_value.detach().item()),
                "semantic_align": float(align_value.detach().item()),
                "semantic_point": float(point_value.detach().item()),
                "boundary": float(boundary_value.detach().item()),
                "depth": float(depth_value.detach().item()),
                "depth_trust": float(depth_trust.detach().item()),
            }

        bindings = build_residual_slice_bindings(
            factor_bindings=list(sceneproof_factor_bindings),
            object_ids=tuple(sceneproof_object_ids),
            collision_pairs=collision_pairs,
            collision_witness_pairs=collision_witness_pairs,
            support_count=(
                int(support_pairs.shape[0])
                + int(fixed_support_indices.shape[0])
            ),
            plane_count=int(plane_object_indices.shape[0]),
            orientation_object_indices=plane_object_indices[
                plane_orientation_mask
            ],
            containment_count=int(containment_pairs.shape[0]),
            distance_count=int(distance_pairs.shape[0]),
            align_count=int(align_pairs.shape[0]),
            point_count=int(point_pairs.shape[0]),
            boundary_object_indices=boundary_object_indices,
            depth_observation_indices=depth_observation_indices,
            collision_weight=collision_weight,
            witness_weight=lm_collision_witness_weight,
            contact_weight=contact_weight,
            plane_weight=plane_weight,
            orientation_weight=orientation_weight,
            containment_weight=containment_weight,
            semantic_weight=semantic_weight,
            boundary_weight=boundary_weight,
            depth_reprojection_weight=depth_reprojection_weight,
            depth_trust_region_weight=depth_trust_region_weight,
            optimize_yaw=False,
            warm_start_weight=0.0,
        )
        if warm_start_weight > 0:
            cursor = bindings[-1].stop if bindings else 0
            warm_bindings: list[ResidualSliceBinding] = []
            for block in chart.blocks:
                width = block.parameter_stop - block.parameter_start
                warm_bindings.append(
                    ResidualSliceBinding(
                        factor_id=f"system:full_so3_warm:{block.object_index}",
                        channel="full_so3_warm",
                        start=cursor,
                        stop=cursor + width,
                        declared_object_indices=(block.object_index,),
                    )
                )
                cursor += width
            bindings = tuple(bindings) + tuple(warm_bindings)
        incumbent = chart.zero_parameters()
        residuals = residual_function(incumbent)
        jacobian = torch.func.jacrev(residual_function)(incumbent)
        object_slices = {
            block.object_index: (block.parameter_slice,)
            for block in chart.blocks
        }
        dependencies: dict[int, tuple[int, ...]] = {}
        for block in chart.blocks:
            ancestors: list[int] = []
            parent = int(block.parent_index)
            while parent >= 0:
                ancestors.append(parent)
                parent = int(chart.blocks[parent].parent_index)
            dependencies[block.object_index] = tuple(ancestors)
        ownership = audit_jacobian_block_ownership(
            residuals=residuals,
            jacobian=jacobian,
            bindings=bindings,
            object_parameter_slices=object_slices,
            object_dependencies=dependencies,
        )
        if not ownership["passed"]:
            return incumbent_pose, {
                "schema_version": "sceneproof_full_so3_guarded_schur_v1",
                "accepted": False,
                "reason": "full_so3_jacobian_ownership_failed",
                "chart": chart.metadata(),
                "audited_allowed_leaf_objects": list(
                    stable_audit.get(
                        "eligible_leaf_translation_objects", []
                    )
                ),
                "final_stable_leaf_objects": [],
                "final_linearization_rejections": {},
                "schur": {
                    "eliminated_leaf_objects": [],
                    "rotation_parameters_eliminated": 0,
                },
                "collision_guard": {
                    "accepted": False,
                    "collision_candidates_checked": 0,
                    "failed_factor_ids": [],
                },
                "selected_collision_guard": {
                    "accepted": False,
                    "collision_candidates_checked": 0,
                },
                "component_failures": {
                    "jacobian_ownership": "failed"
                },
                "selected_component_failures": {
                    "jacobian_ownership": "failed"
                },
                "rotation_parameters_eliminated": 0,
                "incumbent_restored": True,
                "ownership": ownership,
            }
        block_slices = {
            f"object:{block.object_index}": block.parameter_slice
            for block in chart.blocks
        }
        factors: list[LinearizedFactor] = []
        for binding in bindings:
            owners: set[int] = set()
            for object_index in binding.declared_object_indices:
                owners.add(int(object_index))
                owners.update(dependencies.get(object_index, ()))
            rows = jacobian[binding.residual_slice]
            factors.append(
                LinearizedFactor(
                    binding.factor_id,
                    residuals[binding.residual_slice],
                    {
                        f"object:{index}": rows[:, chart.blocks[index].parameter_slice]
                        for index in sorted(owners)
                    },
                    tuple(sorted(owners)),
                )
            )
        normal, gradient, assembly = assemble_normal_system(
            parameter_count=chart.parameter_count,
            slices=block_slices,
            factors=factors,
            damping=lm_damping,
            dtype=jacobian.dtype,
            device=jacobian.device,
        )
        dense_gradient = jacobian.transpose(0, 1) @ residuals
        dense_normal = jacobian.transpose(0, 1) @ jacobian
        if lm_damping:
            dense_diagonal = torch.diagonal(dense_normal).abs().clamp_min(1.0)
            dense_normal = dense_normal + float(lm_damping) * torch.diag(
                dense_diagonal
            )
        gradient_consistency_error = float(
            (gradient - dense_gradient).abs().amax().detach().item()
        )
        normal_consistency_error = float(
            (normal - dense_normal).abs().amax().detach().item()
        )
        # Certify the two assembled tensors independently.  A shared scale is
        # unsound here: the normal matrix can be many orders of magnitude
        # larger than the gradient and would then give the gradient check an
        # effectively unbounded absolute tolerance.  The mixed error test is
        # the standard backward-error form, with a small dtype-derived floor
        # for entries whose reference magnitude is close to zero.
        consistency_epsilon = float(torch.finfo(jacobian.dtype).eps)
        consistency_rtol = 64.0 * consistency_epsilon
        consistency_atol = 64.0 * consistency_epsilon
        gradient_reference_inf = float(
            dense_gradient.abs().amax().detach().item()
        )
        normal_reference_inf = float(
            dense_normal.abs().amax().detach().item()
        )
        gradient_consistency_tolerance = (
            consistency_atol
            + consistency_rtol * gradient_reference_inf
        )
        normal_consistency_tolerance = (
            consistency_atol
            + consistency_rtol * normal_reference_inf
        )
        gradient_relative_error = gradient_consistency_error / max(
            gradient_reference_inf,
            consistency_atol,
        )
        normal_relative_error = normal_consistency_error / max(
            normal_reference_inf,
            consistency_atol,
        )
        linearization_consistency = {
            "passed": bool(
                math.isfinite(gradient_consistency_error)
                and math.isfinite(normal_consistency_error)
                and math.isfinite(gradient_relative_error)
                and math.isfinite(normal_relative_error)
                and gradient_consistency_error
                <= gradient_consistency_tolerance
                and normal_consistency_error
                <= normal_consistency_tolerance
            ),
            "gradient_max_abs_error": gradient_consistency_error,
            "normal_max_abs_error": normal_consistency_error,
            "gradient_reference_inf": gradient_reference_inf,
            "normal_reference_inf": normal_reference_inf,
            "gradient_relative_error": gradient_relative_error,
            "normal_relative_error": normal_relative_error,
            "gradient_tolerance": gradient_consistency_tolerance,
            "normal_tolerance": normal_consistency_tolerance,
            "relative_tolerance": consistency_rtol,
            "absolute_tolerance": consistency_atol,
            "dtype_epsilon": consistency_epsilon,
        }
        stable_active = set(stable_audit["stable_active_factor_ids"])
        stable_inactive = set(stable_audit["stable_inactive_factor_ids"])
        full_active = set(ownership["active_factor_ids"])
        incidence = tuple(
            binding.declared_object_indices
            for binding in bindings
            if binding.factor_id in stable_active
        )
        audited_allowed_leaves = tuple(
            int(value)
            for value in stable_audit["eligible_leaf_translation_objects"]
        )
        allowed_leaves: list[int] = []
        final_linearization_rejections: dict[int, list[str]] = {}
        for object_index in audited_allowed_leaves:
            changed: list[str] = []
            for binding in bindings:
                if object_index not in binding.declared_object_indices:
                    continue
                # This unary tangent prior is introduced only by the full-SO(3)
                # trial.  It is structurally stable and cannot enlarge a leaf
                # separator, so it must not invalidate the preceding live
                # linearization audit.
                if binding.channel == "full_so3_warm":
                    continue
                factor_id = binding.factor_id
                current_active = factor_id in full_active
                stable_state_matches = (
                    current_active and factor_id in stable_active
                ) or (
                    not current_active and factor_id in stable_inactive
                )
                if not stable_state_matches:
                    changed.append(factor_id)
            if changed:
                final_linearization_rejections[object_index] = sorted(changed)
            else:
                allowed_leaves.append(object_index)
        responsibility_objects: set[int] = set(allowed_leaves)
        for object_index in tuple(responsibility_objects):
            parent = int(chart.blocks[object_index].parent_index)
            if parent >= 0:
                responsibility_objects.add(parent)
        for binding in bindings:
            if binding.factor_id not in stable_active:
                continue
            participants = set(binding.declared_object_indices)
            if participants & set(allowed_leaves):
                responsibility_objects.update(participants)
        for object_index in tuple(responsibility_objects):
            responsibility_objects.update(dependencies.get(object_index, ()))
        active_parameter_mask = torch.zeros(
            chart.parameter_count,
            dtype=torch.bool,
            device=normal.device,
        )
        for object_index in responsibility_objects:
            active_parameter_mask[chart.blocks[object_index].parameter_slice] = True
        restricted_normal, restricted_gradient, restriction = (
            restrict_normal_system_to_parameter_mask(
                normal,
                gradient,
                active_parameter_mask,
            )
        )
        raw_step, schur = solve_with_leaf_translation_schur(
            restricted_normal,
            -restricted_gradient,
            chart,
            incidence,
            allowed_leaf_objects=tuple(allowed_leaves),
        )
        raw_directional_derivative = float(
            torch.dot(restricted_gradient, raw_step).detach().item()
        )
        step, global_trust_scale = chart.cap_step_globally(
            raw_step,
            max_translation=lm_max_translation_step,
            max_rotation_radians=math.radians(lm_max_yaw_step_degrees),
        )
        capped_directional_derivative = float(
            torch.dot(restricted_gradient, step).detach().item()
        )
        direction_source = "schur"
        active_gradient_inf = float(
            restricted_gradient[active_parameter_mask].abs().amax()
            .detach().item()
            if bool(active_parameter_mask.any().item())
            else 0.0
        )
        direction_numeric_floor = max(
            1e-9,
            16.0 * float(torch.finfo(step.dtype).eps),
        )
        if (
            active_gradient_inf > direction_numeric_floor
            and (
                not math.isfinite(capped_directional_derivative)
                or capped_directional_derivative >= 0.0
            )
        ):
            cauchy = -restricted_gradient
            step, global_trust_scale = chart.cap_step_globally(
                cauchy,
                max_translation=lm_max_translation_step,
                max_rotation_radians=math.radians(
                    lm_max_yaw_step_degrees
                ),
            )
            capped_directional_derivative = float(
                torch.dot(restricted_gradient, step).detach().item()
            )
            direction_source = "projected_gradient_cauchy"
        stationarity = certify_projected_stationarity(
            restricted_gradient,
            step,
            active_parameter_mask,
            tolerance=1e-9,
        )
        collision_bindings = tuple(
            binding
            for binding in bindings
            if binding.channel == "collision_oriented_penetration"
        )

        def collision_rows(parameters: torch.Tensor) -> torch.Tensor:
            _, values = oriented_penetration_loss(
                chart.pose_matrices(parameters),
                local_corners,
                collision_pairs,
                footprint_hull_sizes,
            )
            return values

        before_components = component_values(incumbent)
        component_tolerance = 1e-5
        before_energy = float(torch.dot(residuals, residuals).detach().item())
        directional_probe_records: list[dict[str, float]] = []
        for probe_scale in (0.1, 0.01, 0.001, 0.0001):
            forward_residuals = residual_function(
                incumbent + step * probe_scale
            )
            reverse_residuals = residual_function(
                incumbent - step * probe_scale
            )
            forward_energy = float(
                torch.dot(forward_residuals, forward_residuals)
                .detach()
                .item()
            )
            reverse_energy = float(
                torch.dot(reverse_residuals, reverse_residuals)
                .detach()
                .item()
            )
            directional_probe_records.append(
                {
                    "scale": probe_scale,
                    "forward_energy": forward_energy,
                    "reverse_energy": reverse_energy,
                    "forward_delta": forward_energy - before_energy,
                    "reverse_delta": reverse_energy - before_energy,
                    "forward_one_sided_slope": (
                        forward_energy - before_energy
                    ) / probe_scale,
                    "reverse_one_sided_slope": (
                        reverse_energy - before_energy
                    ) / probe_scale,
                    "central_slope": (
                        forward_energy - reverse_energy
                    ) / (2.0 * probe_scale),
                }
            )
        directional_model_audit = {
            "analytic_energy_directional_derivative": (
                2.0 * capped_directional_derivative
            ),
            "probes": directional_probe_records,
            "forward_descent_observed": any(
                row["forward_delta"] < 0.0
                for row in directional_probe_records
            ),
            "reverse_descent_observed": any(
                row["reverse_delta"] < 0.0
                for row in directional_probe_records
            ),
        }
        parent_by_object = {
            block.object_index: block.parent_index for block in chart.blocks
        }
        trial_records: list[dict[str, Any]] = []
        collision_blocked_descent_trials: list[dict[str, Any]] = []
        accepted = False
        selected = incumbent
        selected_components = before_components
        selected_component_failures: dict[str, Any] = {}
        selected_energy = before_energy
        selected_collision_guard: dict[str, Any] = {
            "accepted": True,
            "collision_candidates_checked": len(collision_bindings),
            "failed_factor_ids": [],
        }
        guarded_candidate = incumbent
        collision_guard: dict[str, Any] = selected_collision_guard
        after_components = before_components
        component_failures: dict[str, Any] = {}
        after_energy = before_energy
        objective_passed = False
        accepted_scale = 0.0
        armijo_c1 = 1e-4
        backtracking_scales = tuple(0.5 ** index for index in range(16))
        for scale in backtracking_scales:
            candidate = incumbent + step * scale
            guarded, guard = guarded_collision_trial(
                incumbent_parameters=incumbent,
                candidate_parameters=candidate,
                collision_bindings=collision_bindings,
                evaluate_collision_residuals=collision_rows,
                parent_by_object=parent_by_object,
                primary_child_objects=tuple(sorted(responsibility_objects)),
            )
            components = component_values(candidate)
            failures = {
                key: {
                    "incumbent": before_components[key],
                    "candidate": components[key],
                    "increase": components[key] - before_components[key],
                }
                for key in before_components
                if components[key]
                > before_components[key] + component_tolerance
            }
            candidate_residuals = residual_function(candidate)
            energy = float(
                torch.dot(candidate_residuals, candidate_residuals)
                .detach()
                .item()
            )
            armijo_bound = (
                before_energy
                + 2.0 * armijo_c1 * scale * capped_directional_derivative
            )
            meaningful_step = bool(
                float((step * scale).abs().amax().detach().item())
                > float(stationarity["effective_tolerance"])
            )
            energy_passed = bool(
                math.isfinite(energy)
                and energy < before_energy
                and energy <= armijo_bound
            )
            trial_records.append(
                {
                    "scale": scale,
                    "collision_passed": bool(guard["accepted"]),
                    "failed_collision_factors": list(
                        guard["failed_factor_ids"]
                    ),
                    "component_failures": failures,
                    "objective": energy,
                    "objective_delta": energy - before_energy,
                    "armijo_bound": armijo_bound,
                    "objective_passed": energy_passed,
                    "meaningful_step": meaningful_step,
                }
            )
            if (
                not guard["accepted"]
                and not failures
                and energy_passed
                and meaningful_step
            ):
                collision_blocked_descent_trials.append(
                    {
                        "scale": scale,
                        "candidate": candidate.detach().clone(),
                        "guard": guard,
                        "energy": energy,
                    }
                )
            if scale == 1.0:
                collision_guard = guard
                after_components = components
                component_failures = failures
                after_energy = energy
                objective_passed = energy_passed
            if (
                guard["accepted"]
                and not failures
                and energy_passed
                and meaningful_step
                and linearization_consistency["passed"]
                and schur["rotation_parameters_eliminated"] == 0
                and set(schur["eliminated_leaf_objects"]).issubset(
                    set(allowed_leaves)
                )
            ):
                accepted = True
                accepted_scale = scale
                selected = guarded
                guarded_candidate = guarded
                selected_components = components
                selected_component_failures = failures
                selected_energy = energy
                selected_collision_guard = guard
                break
        local_resolve: dict[str, Any] = {"attempted": False}
        if not accepted and not collision_guard["accepted"]:
            released = [
                int(value)
                for value in collision_guard["released_object_indices"]
            ]
            local_indices = sorted(
                index
                for object_index in released
                for index in range(
                    chart.blocks[object_index].rotation_start,
                    chart.blocks[object_index].translation_stop,
                )
            )
            local_resolve = {
                "attempted": bool(local_indices),
                "released_object_indices": released,
                "released_factor_ids": collision_guard["failed_factor_ids"],
                "accepted": False,
            }
            if local_indices:
                index_tensor = torch.as_tensor(
                    local_indices, dtype=torch.long, device=normal.device
                )
                local_normal = normal[index_tensor][:, index_tensor]
                local_gradient = gradient[index_tensor]
                try:
                    local_step = torch.linalg.solve(
                        local_normal,
                        -local_gradient,
                    )
                    local_full_step = torch.zeros_like(incumbent)
                    local_full_step[index_tensor] = local_step
                    local_full_step, local_trust_scale = chart.cap_step_globally(
                        local_full_step,
                        max_translation=lm_max_translation_step,
                        max_rotation_radians=math.radians(
                            lm_max_yaw_step_degrees
                        ),
                    )
                    local_directional_derivative = float(
                        torch.dot(gradient, local_full_step).detach().item()
                    )
                    local_trials: list[dict[str, Any]] = []
                    for local_scale in backtracking_scales:
                        local_candidate = (
                            incumbent + local_full_step * local_scale
                        )
                        local_guarded, local_collision_guard = (
                            guarded_collision_trial(
                                incumbent_parameters=incumbent,
                                candidate_parameters=local_candidate,
                                collision_bindings=collision_bindings,
                                evaluate_collision_residuals=collision_rows,
                                parent_by_object=parent_by_object,
                                # The local solve may move every released
                                # separator.  Classify exactly that scope when
                                # assigning responsibility for a new collision.
                                primary_child_objects=tuple(released),
                            )
                        )
                        local_after = component_values(local_candidate)
                        local_failures = {
                            key: {
                                "incumbent": before_components[key],
                                "candidate": local_after[key],
                                "increase": (
                                    local_after[key] - before_components[key]
                                ),
                            }
                            for key in before_components
                            if local_after[key]
                            > before_components[key] + component_tolerance
                        }
                        local_residuals = residual_function(local_candidate)
                        local_energy = float(
                            torch.dot(local_residuals, local_residuals)
                            .detach()
                            .item()
                        )
                        local_armijo_bound = (
                            before_energy
                            + 2.0
                            * armijo_c1
                            * local_scale
                            * local_directional_derivative
                        )
                        local_meaningful = bool(
                            float(
                                (local_full_step * local_scale)
                                .abs().amax().detach().item()
                            ) > float(stationarity["effective_tolerance"])
                        )
                        local_objective_passed = bool(
                            math.isfinite(local_energy)
                            and local_energy < before_energy
                            and local_energy <= local_armijo_bound
                        )
                        local_accepted = bool(
                            local_collision_guard["accepted"]
                            and not local_failures
                            and local_objective_passed
                            and local_meaningful
                            and linearization_consistency["passed"]
                        )
                        local_trials.append(
                            {
                                "scale": local_scale,
                                "collision_passed": bool(
                                    local_collision_guard["accepted"]
                                ),
                                "failed_collision_factors": list(
                                    local_collision_guard["failed_factor_ids"]
                                ),
                                "component_failures": local_failures,
                                "objective": local_energy,
                                "objective_delta": local_energy - before_energy,
                                "armijo_bound": local_armijo_bound,
                                "objective_passed": local_objective_passed,
                                "meaningful_step": local_meaningful,
                            }
                        )
                        local_resolve.update(
                            {
                                "accepted": local_accepted,
                                "collision_guard": local_collision_guard,
                                "component_failures": local_failures,
                                "energy": local_energy,
                                "trials": local_trials,
                                "directional_derivative": (
                                    local_directional_derivative
                                ),
                                "global_trust_scale": local_trust_scale,
                            }
                        )
                        if local_accepted:
                            guarded_candidate = local_guarded
                            accepted = True
                            accepted_scale = local_scale
                            selected_components = local_after
                            selected_component_failures = local_failures
                            selected_energy = local_energy
                            selected_collision_guard = local_collision_guard
                            break
                except RuntimeError as error:
                    local_resolve["error"] = str(error)
        # If a globally descending trial is blocked only by localized
        # collision witnesses, restore the complete SO(3)+translation blocks
        # in that responsibility scope and re-certify the complement.  This
        # is a partial commit, never a relaxed collision threshold.
        partial_commit: dict[str, Any] = {
            "attempted": False,
            "accepted": False,
            "trials": [],
        }
        if not accepted and collision_blocked_descent_trials:
            for source in sorted(
                collision_blocked_descent_trials,
                key=lambda row: float(row["energy"]),
            ):
                released = [
                    int(value)
                    for value in source["guard"]["released_object_indices"]
                ]
                partial_candidate, rollback_audit = (
                    rollback_object_parameter_blocks(
                        incumbent_parameters=incumbent,
                        candidate_parameters=source["candidate"],
                        coordinates=chart,
                        object_indices=released,
                    )
                )
                partial_guarded, partial_collision_guard = (
                    guarded_collision_trial(
                        incumbent_parameters=incumbent,
                        candidate_parameters=partial_candidate,
                        collision_bindings=collision_bindings,
                        evaluate_collision_residuals=collision_rows,
                        parent_by_object=parent_by_object,
                        primary_child_objects=tuple(
                            sorted(responsibility_objects)
                        ),
                    )
                )
                partial_components = component_values(partial_candidate)
                partial_failures = {
                    key: {
                        "incumbent": before_components[key],
                        "candidate": partial_components[key],
                        "increase": (
                            partial_components[key] - before_components[key]
                        ),
                    }
                    for key in before_components
                    if partial_components[key]
                    > before_components[key] + component_tolerance
                }
                partial_residuals = residual_function(partial_candidate)
                partial_energy = float(
                    torch.dot(partial_residuals, partial_residuals)
                    .detach()
                    .item()
                )
                partial_delta = partial_candidate - incumbent
                partial_directional_derivative = float(
                    torch.dot(gradient, partial_delta).detach().item()
                )
                partial_armijo_bound = (
                    before_energy
                    + 2.0 * armijo_c1 * partial_directional_derivative
                )
                partial_meaningful = bool(
                    float(partial_delta.abs().amax().detach().item())
                    > float(stationarity["effective_tolerance"])
                )
                partial_objective_passed = bool(
                    math.isfinite(partial_energy)
                    and partial_directional_derivative < 0.0
                    and partial_energy < before_energy
                    and partial_energy <= partial_armijo_bound
                )
                partial_accepted = bool(
                    partial_collision_guard["accepted"]
                    and not partial_failures
                    and partial_objective_passed
                    and partial_meaningful
                    and linearization_consistency["passed"]
                    and schur["rotation_parameters_eliminated"] == 0
                )
                record = {
                    "source_scale": source["scale"],
                    "rollback": rollback_audit,
                    "collision_guard": partial_collision_guard,
                    "component_failures": partial_failures,
                    "objective": partial_energy,
                    "objective_delta": partial_energy - before_energy,
                    "directional_derivative": partial_directional_derivative,
                    "armijo_bound": partial_armijo_bound,
                    "objective_passed": partial_objective_passed,
                    "meaningful_step": partial_meaningful,
                    "accepted": partial_accepted,
                }
                partial_commit["attempted"] = True
                partial_commit["trials"].append(record)
                if partial_accepted:
                    accepted = True
                    accepted_scale = float(source["scale"])
                    guarded_candidate = partial_guarded
                    selected_components = partial_components
                    selected_component_failures = partial_failures
                    selected_energy = partial_energy
                    selected_collision_guard = partial_collision_guard
                    partial_commit["accepted"] = True
                    partial_commit["selected"] = record
                    break
        # At a one-sided contact switch the generalized Jacobian can select
        # the wrong sign even though the exact two-sided probe identifies a
        # descending reverse ray.  Recover that sign explicitly, but still
        # require the complete collision and component-wise no-harm gates.
        # This is cheaper than a coordinate poll and therefore precedes it.
        reverse_direction_recovery: dict[str, Any] = {
            "attempted": False,
            "accepted": False,
        }
        reverse_only_descent = bool(
            not directional_model_audit["forward_descent_observed"]
            and directional_model_audit["reverse_descent_observed"]
        )
        if not accepted and reverse_only_descent:
            reverse_trials: list[dict[str, Any]] = []
            for reverse_scale in backtracking_scales:
                reverse_candidate = incumbent - step * reverse_scale
                reverse_guarded, reverse_guard = guarded_collision_trial(
                    incumbent_parameters=incumbent,
                    candidate_parameters=reverse_candidate,
                    collision_bindings=collision_bindings,
                    evaluate_collision_residuals=collision_rows,
                    parent_by_object=parent_by_object,
                    primary_child_objects=tuple(
                        sorted(responsibility_objects)
                    ),
                )
                reverse_components = component_values(reverse_candidate)
                reverse_failures = {
                    key: {
                        "incumbent": before_components[key],
                        "candidate": reverse_components[key],
                        "increase": (
                            reverse_components[key]
                            - before_components[key]
                        ),
                    }
                    for key in before_components
                    if reverse_components[key]
                    > before_components[key] + component_tolerance
                }
                reverse_residuals = residual_function(reverse_candidate)
                reverse_energy = float(
                    torch.dot(reverse_residuals, reverse_residuals)
                    .detach()
                    .item()
                )
                reverse_meaningful = bool(
                    float(
                        (step * reverse_scale).abs().amax().detach().item()
                    ) > float(stationarity["effective_tolerance"])
                )
                reverse_objective_passed = bool(
                    math.isfinite(reverse_energy)
                    and reverse_energy < before_energy
                )
                reverse_accepted = bool(
                    reverse_guard["accepted"]
                    and not reverse_failures
                    and reverse_objective_passed
                    and reverse_meaningful
                    and linearization_consistency["passed"]
                    and schur["rotation_parameters_eliminated"] == 0
                )
                reverse_record = {
                    "scale": reverse_scale,
                    "collision_passed": bool(reverse_guard["accepted"]),
                    "failed_collision_factors": list(
                        reverse_guard["failed_factor_ids"]
                    ),
                    "component_failures": reverse_failures,
                    "objective": reverse_energy,
                    "objective_delta": reverse_energy - before_energy,
                    "objective_passed": reverse_objective_passed,
                    "meaningful_step": reverse_meaningful,
                    "accepted": reverse_accepted,
                }
                reverse_trials.append(reverse_record)
                reverse_direction_recovery.update(
                    {
                        "attempted": True,
                        "trials": reverse_trials,
                        "accepted": reverse_accepted,
                    }
                )
                if reverse_accepted:
                    accepted = True
                    accepted_scale = reverse_scale
                    guarded_candidate = reverse_guarded
                    selected_components = reverse_components
                    selected_component_failures = reverse_failures
                    selected_energy = reverse_energy
                    selected_collision_guard = reverse_guard
                    reverse_direction_recovery["selected"] = reverse_record
                    break
        # A smooth Gauss--Newton model is not reliable at a contact switch.
        # When the exact objective rises in both directions of the Schur step,
        # use a positive-spanning poll in the already audited responsibility
        # subspace.  The poll is derivative-free, unit-aware, and every point
        # must pass the same exact collision and component-wise no-harm gates.
        # This is intentionally an exceptional fallback; smooth regions keep
        # the topology-Schur fast path and all SO(3) variables stay in root.
        two_sided_kink = bool(
            not directional_model_audit["forward_descent_observed"]
            and not directional_model_audit["reverse_descent_observed"]
        )
        nonsmooth_poll: dict[str, Any] = {
            "attempted": False,
            "accepted": False,
            "classification": (
                "two_sided_nonsmooth_kink" if two_sided_kink else None
            ),
            "method": "responsibility_positive_spanning_poll",
        }
        if not accepted and (two_sided_kink or reverse_only_descent):
            rotation_parameter_mask = torch.zeros_like(
                active_parameter_mask
            )
            for block in chart.blocks:
                rotation_parameter_mask[block.rotation_slice] = True
            poll_radii = (
                (math.radians(0.5), 0.005),
                (math.radians(0.25), 0.0025),
                (math.radians(0.125), 0.00125),
            )
            poll_records: list[dict[str, Any]] = []
            best_poll: dict[str, Any] | None = None
            for radius_index, (rotation_radius, translation_radius) in enumerate(
                poll_radii
            ):
                basis = positive_spanning_poll_steps(
                    active_parameter_mask,
                    rotation_parameter_mask,
                    rotation_radius=rotation_radius,
                    translation_radius=translation_radius,
                )
                for parameter_index, sign, raw_poll_step in basis:
                    poll_step = raw_poll_step.to(
                        dtype=incumbent.dtype,
                        device=incumbent.device,
                    )
                    poll_candidate = incumbent + poll_step
                    poll_guarded, poll_guard = guarded_collision_trial(
                        incumbent_parameters=incumbent,
                        candidate_parameters=poll_candidate,
                        collision_bindings=collision_bindings,
                        evaluate_collision_residuals=collision_rows,
                        parent_by_object=parent_by_object,
                        primary_child_objects=tuple(
                            sorted(responsibility_objects)
                        ),
                    )
                    poll_components = component_values(poll_candidate)
                    poll_failures = {
                        key: {
                            "incumbent": before_components[key],
                            "candidate": poll_components[key],
                            "increase": (
                                poll_components[key]
                                - before_components[key]
                            ),
                        }
                        for key in before_components
                        if poll_components[key]
                        > before_components[key] + component_tolerance
                    }
                    poll_residuals = residual_function(poll_candidate)
                    poll_energy = float(
                        torch.dot(poll_residuals, poll_residuals)
                        .detach()
                        .item()
                    )
                    poll_meaningful = bool(
                        float(poll_step.abs().amax().detach().item())
                        > float(stationarity["effective_tolerance"])
                    )
                    poll_objective_passed = bool(
                        math.isfinite(poll_energy)
                        and poll_energy < before_energy
                    )
                    poll_accepted = bool(
                        poll_guard["accepted"]
                        and not poll_failures
                        and poll_objective_passed
                        and poll_meaningful
                        and linearization_consistency["passed"]
                    )
                    record = {
                        "radius_index": radius_index,
                        "rotation_radius_radians": rotation_radius,
                        "translation_radius_m": translation_radius,
                        "parameter_index": parameter_index,
                        "parameter_kind": (
                            "rotation"
                            if bool(
                                rotation_parameter_mask[
                                    parameter_index
                                ].item()
                            )
                            else "translation"
                        ),
                        "sign": sign,
                        "collision_passed": bool(poll_guard["accepted"]),
                        "failed_collision_factors": list(
                            poll_guard["failed_factor_ids"]
                        ),
                        "component_failures": poll_failures,
                        "objective": poll_energy,
                        "objective_delta": poll_energy - before_energy,
                        "objective_passed": poll_objective_passed,
                        "meaningful_step": poll_meaningful,
                        "accepted": poll_accepted,
                    }
                    poll_records.append(record)
                    if poll_accepted and (
                        best_poll is None
                        or poll_energy < best_poll["energy"]
                    ):
                        best_poll = {
                            "energy": poll_energy,
                            "parameters": poll_guarded,
                            "components": poll_components,
                            "component_failures": poll_failures,
                            "collision_guard": poll_guard,
                            "record": record,
                        }
                # A complete +/- coordinate basis is a positive-spanning set.
                # Once one radius has a safe descent point, selecting its best
                # point avoids spending more evaluations on smaller meshes.
                if best_poll is not None:
                    break
            nonsmooth_poll.update(
                {
                    "attempted": True,
                    "active_parameters": int(
                        active_parameter_mask.sum().item()
                    ),
                    "rotation_parameters_polled": int(
                        (
                            active_parameter_mask
                            & rotation_parameter_mask
                        ).sum().item()
                    ),
                    "translation_parameters_polled": int(
                        (
                            active_parameter_mask
                            & ~rotation_parameter_mask
                        ).sum().item()
                    ),
                    "radii": [
                        {
                            "rotation_radius_radians": rotation_radius,
                            "translation_radius_m": translation_radius,
                        }
                        for rotation_radius, translation_radius in poll_radii
                    ],
                    "records": poll_records,
                    "accepted": best_poll is not None,
                    "poll_stationary": best_poll is None,
                }
            )
            if best_poll is not None:
                accepted = True
                accepted_scale = 1.0
                guarded_candidate = best_poll["parameters"]
                selected_components = best_poll["components"]
                selected_component_failures = best_poll[
                    "component_failures"
                ]
                selected_energy = best_poll["energy"]
                selected_collision_guard = best_poll["collision_guard"]
                nonsmooth_poll["selected"] = best_poll["record"]
        selected = guarded_candidate if accepted else incumbent
        rotation_step_max = max(
            (
                float(
                    torch.linalg.vector_norm(step[block.rotation_slice])
                    .detach()
                    .item()
                )
                for block in chart.blocks
            ),
            default=0.0,
        )
        translation_step_max = max(
            (
                float(
                    torch.linalg.vector_norm(step[block.translation_slice])
                    .detach()
                    .item()
                )
                for block in chart.blocks
            ),
            default=0.0,
        )
        selected_delta = selected - incumbent
        selected_rotation_step_max = max(
            (
                float(
                    torch.linalg.vector_norm(
                        selected_delta[block.rotation_slice]
                    ).detach().item()
                )
                for block in chart.blocks
            ),
            default=0.0,
        )
        selected_translation_step_max = max(
            (
                float(
                    torch.linalg.vector_norm(
                        selected_delta[block.translation_slice]
                    ).detach().item()
                )
                for block in chart.blocks
            ),
            default=0.0,
        )
        return chart.pose_matrices(selected).detach(), {
            "schema_version": "sceneproof_full_so3_guarded_schur_v1",
            "accepted": accepted,
            "certified_stationary": bool(stationarity["certified"]),
            "reason": (
                "accepted"
                if accepted
                else "certified_stationary_noop"
                if stationarity["certified"]
                else "incumbent_restored_after_guard_failure"
            ),
            "chart": chart.metadata(),
            "audited_allowed_leaf_objects": list(audited_allowed_leaves),
            "final_stable_leaf_objects": list(allowed_leaves),
            "final_linearization_rejections": final_linearization_rejections,
            "assembly": assembly,
            "linearization_consistency": linearization_consistency,
            "directional_model_audit": directional_model_audit,
            "reverse_direction_recovery": reverse_direction_recovery,
            "nonsmooth_active_set_poll": nonsmooth_poll,
            "armijo_c1": armijo_c1,
            "responsibility_object_indices": sorted(responsibility_objects),
            "responsibility_active_parameters": restriction[
                "active_parameters"
            ],
            "frozen_root_parameters": restriction["frozen_parameters"],
            "schur": schur,
            "direction_source": direction_source,
            "raw_directional_derivative": raw_directional_derivative,
            "capped_directional_derivative": capped_directional_derivative,
            "global_trust_scale": global_trust_scale,
            "direction_numeric_floor": direction_numeric_floor,
            "stationarity_certificate": stationarity,
            "trial_records": trial_records,
            "accepted_scale": accepted_scale,
            "selected_trial_kind": (
                "collision_responsibility_partial_commit"
                if partial_commit.get("accepted")
                else
                "reverse_direction_recovery"
                if reverse_direction_recovery.get("accepted")
                else "nonsmooth_positive_spanning_poll"
                if nonsmooth_poll.get("accepted")
                else "local_backtracking"
                if local_resolve.get("accepted")
                else "responsibility_schur_backtracking"
                if accepted
                else "stationarity_certificate"
                if stationarity["certified"]
                else "rollback"
            ),
            "collision_guard": collision_guard,
            "local_resolve": local_resolve,
            "collision_responsibility_partial_commit": partial_commit,
            "component_incumbent": before_components,
            "component_candidate": after_components,
            "component_failures": component_failures,
            "selected_components": selected_components,
            "selected_component_failures": selected_component_failures,
            "objective_before": before_energy,
            "objective_after": after_energy,
            "selected_objective": selected_energy,
            "objective_passed": objective_passed,
            "selected_collision_guard": selected_collision_guard,
            "rotation_step_max_radians": rotation_step_max,
            "translation_step_max_m": translation_step_max,
            "selected_rotation_step_max_radians": (
                selected_rotation_step_max
            ),
            "selected_translation_step_max_m": (
                selected_translation_step_max
            ),
            "rotation_parameters_eliminated": 0,
            "incumbent_restored": not accepted,
        }

    if sceneproof_shadow_residual_parity or sceneproof_use_program_residuals:
        shadow_parameters = pack_pose_parameters(
            yaw_delta.detach(), translation.detach()
        )
        dense_factor_residuals(shadow_parameters)

    for iteration in range(iterations):
        if active_set_router and iteration >= active_horizon:
            break
        if solver == "v5_scenelm":
            if (
                relation_coordinates is None
                or relation_parameters is None
                or relation_active_objects is None
            ):
                raise RuntimeError("v5 SceneLM state was not initialised")
            yaw_delta, translation = relation_coordinates.decode(
                relation_parameters
            )
            active_mask = relation_active_objects
        else:
            active_mask = iteration < allocated_budget
        executed_iterations = iteration + 1
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        if factor_cache_revision != active_set_revision:
            def active_pair_mask(
                pair_indices: torch.Tensor,
            ) -> torch.Tensor:
                if not active_set_router:
                    return torch.ones(
                        pair_indices.shape[0],
                        dtype=torch.bool,
                        device=pair_indices.device,
                    )
                if pair_indices.numel() == 0:
                    return torch.empty(
                        (0,),
                        dtype=torch.bool,
                        device=pair_indices.device,
                    )
                return active_mask[pair_indices].any(dim=1)

            collision_factor_mask = active_pair_mask(collision_pairs)
            support_factor_mask = active_pair_mask(support_pairs)
            containment_factor_mask = active_pair_mask(containment_pairs)
            # Pairwise losses stay live when either endpoint is active.
            # The hard support projections, however, mutate only the child
            # (column 0), so they must never receive a frozen child merely
            # because its parent is still active.
            support_projection_mask = _active_child_pair_mask(
                support_pairs,
                active_mask,
            )
            containment_projection_mask = _active_child_pair_mask(
                containment_pairs,
                active_mask,
            )
            distance_factor_mask = active_pair_mask(distance_pairs)
            align_factor_mask = active_pair_mask(align_pairs)
            point_factor_mask = active_pair_mask(point_pairs)
            fixed_factor_mask = active_mask[fixed_support_indices]
            plane_factor_mask = active_mask[plane_object_indices]
            boundary_factor_mask = active_mask[boundary_object_indices]
            depth_factor_mask = active_mask[depth_observation_indices]
            factor_cache = {
                "collision_factor_mask": collision_factor_mask,
                "support_factor_mask": support_factor_mask,
                "containment_factor_mask": containment_factor_mask,
                "distance_factor_mask": distance_factor_mask,
                "align_factor_mask": align_factor_mask,
                "point_factor_mask": point_factor_mask,
                "fixed_factor_mask": fixed_factor_mask,
                "plane_factor_mask": plane_factor_mask,
                "boundary_factor_mask": boundary_factor_mask,
                "depth_factor_mask": depth_factor_mask,
                "active_collision_pairs":
                    collision_pairs[collision_factor_mask],
                "active_support_pairs":
                    support_pairs[support_factor_mask],
                "active_support_projection_pairs":
                    support_pairs[support_projection_mask],
                "active_containment_pairs":
                    containment_pairs[containment_factor_mask],
                "active_containment_projection_pairs":
                    containment_pairs[containment_projection_mask],
                "active_distance_pairs":
                    distance_pairs[distance_factor_mask],
                "active_align_pairs": align_pairs[align_factor_mask],
                "active_point_pairs": point_pairs[point_factor_mask],
                "active_fixed_support_indices":
                    fixed_support_indices[fixed_factor_mask],
                "active_fixed_support_heights":
                    fixed_support_heights[fixed_factor_mask],
                "active_plane_object_indices":
                    plane_object_indices[plane_factor_mask],
                "active_plane_points": plane_points[plane_factor_mask],
                "active_plane_normals": plane_normals[plane_factor_mask],
                "active_plane_orientation_mask":
                    plane_orientation_mask[plane_factor_mask],
                "active_boundary_object_indices":
                    boundary_object_indices[boundary_factor_mask],
                "active_depth_observation_indices":
                    depth_observation_indices[depth_factor_mask],
            }
            factor_cache_revision = active_set_revision

        collision_factor_mask = factor_cache["collision_factor_mask"]
        support_factor_mask = factor_cache["support_factor_mask"]
        containment_factor_mask = factor_cache["containment_factor_mask"]
        distance_factor_mask = factor_cache["distance_factor_mask"]
        align_factor_mask = factor_cache["align_factor_mask"]
        point_factor_mask = factor_cache["point_factor_mask"]
        fixed_factor_mask = factor_cache["fixed_factor_mask"]
        plane_factor_mask = factor_cache["plane_factor_mask"]
        boundary_factor_mask = factor_cache["boundary_factor_mask"]
        depth_factor_mask = factor_cache["depth_factor_mask"]
        active_collision_pairs = factor_cache["active_collision_pairs"]
        active_support_pairs = factor_cache["active_support_pairs"]
        active_support_projection_pairs = factor_cache[
            "active_support_projection_pairs"
        ]
        active_containment_pairs = factor_cache[
            "active_containment_pairs"
        ]
        active_containment_projection_pairs = factor_cache[
            "active_containment_projection_pairs"
        ]
        active_distance_pairs = factor_cache["active_distance_pairs"]
        active_align_pairs = factor_cache["active_align_pairs"]
        active_point_pairs = factor_cache["active_point_pairs"]
        active_fixed_support_indices = factor_cache[
            "active_fixed_support_indices"
        ]
        active_fixed_support_heights = factor_cache[
            "active_fixed_support_heights"
        ]
        active_plane_object_indices = factor_cache[
            "active_plane_object_indices"
        ]
        active_plane_points = factor_cache["active_plane_points"]
        active_plane_normals = factor_cache["active_plane_normals"]
        active_plane_orientation_mask = factor_cache[
            "active_plane_orientation_mask"
        ]
        active_boundary_object_indices = factor_cache[
            "active_boundary_object_indices"
        ]
        active_depth_observation_indices = factor_cache[
            "active_depth_observation_indices"
        ]
        routed_yaw_delta = torch.where(
            active_mask,
            yaw_delta,
            yaw_delta.detach(),
        )
        routed_translation = torch.where(
            active_mask[:, None],
            translation,
            translation.detach(),
        )
        pose_matrices = reproject_pose_matrices(
            base_matrices,
            routed_yaw_delta,
            routed_translation,
        )
        collision, per_pair = oriented_penetration_loss(
            pose_matrices,
            local_corners,
            active_collision_pairs,
            footprint_hull_sizes,
        )
        contact, contact_gaps = support_contact_loss(
            pose_matrices,
            local_corners,
            active_support_pairs,
            active_fixed_support_indices,
            active_fixed_support_heights,
        )
        plane, orientation, plane_gaps, alignment_errors = fixed_plane_loss(
            pose_matrices,
            local_corners,
            active_plane_object_indices,
            active_plane_points,
            active_plane_normals,
            active_plane_orientation_mask,
        )
        containment, containment_errors = support_planar_containment_loss(
            pose_matrices,
            local_corners,
            active_containment_pairs,
            footprint_hull_sizes,
        )
        distance, distances, distance_penalties = distance_interval_loss(
            pose_matrices,
            active_distance_pairs,
            distance_minimum[distance_factor_mask],
            distance_maximum[distance_factor_mask],
        )
        align, align_errors = align_with_loss(
            pose_matrices,
            active_align_pairs,
            align_offsets[align_factor_mask],
        )
        point, point_errors = point_towards_loss(
            pose_matrices,
            local_corners,
            active_point_pairs,
            point_offsets[point_factor_mask],
        )
        boundary, boundary_errors = room_boundary_loss(
            pose_matrices,
            local_corners,
            active_boundary_object_indices,
            boundary_points,
            boundary_normals,
        )
        (
            depth_reprojection,
            depth_centre_errors,
            depth_size_errors,
            depth_relative_errors,
        ) = depth_aware_reprojection_loss(
            pose_matrices,
            local_corners,
            active_depth_observation_indices,
            depth_observed_boxes[depth_factor_mask],
            depth_observed_depths[depth_factor_mask],
            depth_observed_weights[depth_factor_mask],
            depth_bbox_size_enabled[depth_factor_mask],
            depth_world_to_camera,
            depth_image_size,
            centre_weight=depth_centre_weight,
            size_weight=depth_size_weight,
            metric_depth_weight=depth_metric_weight,
        )
        (
            depth_trust_region,
            depth_centre_excess,
            depth_size_excess,
            depth_relative_excess,
        ) = no_harm_reprojection_penalty(
            depth_centre_errors,
            depth_size_errors,
            depth_relative_errors,
            depth_reference_centre_errors[depth_factor_mask],
            depth_reference_size_errors[depth_factor_mask],
            depth_reference_relative_errors[depth_factor_mask],
            depth_observed_weights[depth_factor_mask],
            centre_margin_pixels=depth_centre_margin_pixels,
            size_margin_log=depth_size_margin_log,
            depth_margin_log=depth_relative_margin_log,
        )
        if active_set_router:
            collision = _rescale_active_mean(
                collision,
                active_collision_pairs.shape[0],
                collision_pairs.shape[0],
            )
            contact = _rescale_active_mean(
                contact,
                active_support_pairs.shape[0]
                + active_fixed_support_indices.shape[0],
                support_pairs.shape[0] + fixed_support_indices.shape[0],
            )
            plane = _rescale_active_mean(
                plane,
                active_plane_object_indices.shape[0],
                plane_object_indices.shape[0],
            )
            orientation = _rescale_active_mean(
                orientation,
                torch.count_nonzero(active_plane_orientation_mask),
                torch.count_nonzero(plane_orientation_mask),
            )
            containment = _rescale_active_mean(
                containment,
                active_containment_pairs.shape[0],
                containment_pairs.shape[0],
            )
            boundary = _rescale_active_mean(
                boundary,
                active_boundary_object_indices.shape[0],
                boundary_object_indices.shape[0],
            )
            active_depth_mass = depth_observed_weights[
                depth_factor_mask
            ].clamp_min(0.0).sum()
            full_depth_mass = depth_observed_weights.clamp_min(0.0).sum()
            depth_reprojection = _rescale_active_mean(
                depth_reprojection,
                active_depth_mass,
                full_depth_mass,
            )
            depth_trust_region = _rescale_active_mean(
                depth_trust_region,
                active_depth_mass,
                full_depth_mass,
            )
        semantic_terms = [
            terms
            for terms in (
                distance_penalties,
                align_errors,
                point_errors,
            )
            if terms.numel()
        ]
        full_semantic_count = (
            distance_pairs.shape[0]
            + align_pairs.shape[0]
            + point_pairs.shape[0]
        )
        if semantic_terms and active_set_router:
            semantic = (
                sum(terms.sum() for terms in semantic_terms)
                / max(full_semantic_count, 1)
            )
        elif semantic_terms:
            semantic = torch.cat(semantic_terms).mean()
        else:
            semantic = pose_matrices.sum() * 0.0
        warm_start = warm_start_regularization(
            routed_yaw_delta[active_mask],
            routed_translation[active_mask],
            base_matrices[active_mask],
        )
        if active_set_router:
            warm_start = _rescale_active_mean(
                warm_start,
                torch.count_nonzero(active_mask),
                object_count,
            )
        total = (
            collision_weight * collision
            + contact_weight * contact
            + plane_weight * plane
            + orientation_weight * orientation
            + containment_weight * containment
            + semantic_weight * semantic
            + boundary_weight * boundary
            + depth_reprojection_weight * depth_reprojection
            + depth_trust_region_weight * depth_trust_region
            + warm_start_weight * warm_start
        )
        if active_set_router and router_has_frozen:
            current_residuals = active_set_object_residuals(
                object_count,
                collision_pairs=active_collision_pairs,
                collision_values=per_pair,
                support_pairs=active_support_pairs,
                contact_gaps=contact_gaps,
                fixed_support_indices=active_fixed_support_indices,
                plane_object_indices=active_plane_object_indices,
                plane_gaps=plane_gaps,
                plane_alignment_errors=alignment_errors,
                containment_pairs=active_containment_pairs,
                containment_errors=containment_errors,
                distance_pairs=active_distance_pairs,
                distance_penalties=distance_penalties,
                align_pairs=active_align_pairs,
                align_errors=align_errors,
                point_pairs=active_point_pairs,
                point_errors=point_errors,
                boundary_object_indices=active_boundary_object_indices,
                boundary_errors=boundary_errors,
                depth_observation_indices=active_depth_observation_indices,
                depth_centre_errors=depth_centre_errors,
                depth_size_errors=depth_size_errors,
                depth_relative_errors=depth_relative_errors,
            )
            wake_safe = active_set_safe_mask(
                current_residuals,
                torch.zeros_like(translation_update_ema),
                torch.zeros_like(yaw_update_ema),
                thresholds=active_set_thresholds,
                threshold_multiplier=active_set_wake_multiplier,
            )
            wake_mask = (~active_mask) & (~wake_safe)
            if torch.any(wake_mask):
                first_wake = (
                    wake_mask
                    & (allocated_budget <= checkpoints[0])
                    & (iteration < checkpoints[1])
                )
                later_wake = wake_mask & (~first_wake)
                allocated_budget[first_wake] = checkpoints[1]
                allocated_budget[later_wake] = iterations
                wakeup_count += int(torch.count_nonzero(wake_mask).item())
                active_set_revision += 1
                active_horizon = int(allocated_budget.max().item())
                router_has_frozen = bool(
                    torch.any(allocated_budget <= iteration + 1).item()
                )
        detached_total = float(total.detach().item())
        # In the depth stage iteration 1 is the raw warm start and has not
        # passed the hard contact/plane/boundary projections. Saving it and
        # projecting only after restoration would return a pose that was
        # never actually scored. Later iterations begin from a projected
        # state. Preserve the legacy behaviour for non-depth stages.
        best_candidate_is_projected = (
            iteration > 0 or depth_observation_indices.numel() == 0
        )
        if (
            restore_best_state
            and best_candidate_is_projected
            and detached_total < best_total
        ):
            best_total = detached_total
            best_iteration = iteration + 1
            best_yaw_delta = yaw_delta.detach().clone()
            best_translation = translation.detach().clone()
        translation_before_step = translation.detach().clone()
        yaw_before_step = yaw_delta.detach().clone()
        frozen_mask = ~active_mask
        if solver == "adam":
            if optimizer is None or scheduler is None:
                raise RuntimeError("Adam optimizer was not initialized")
            total.backward()
            if translation.grad is not None:
                translation.grad.mul_(active_mask[:, None])
            if optimize_yaw and yaw_delta.grad is not None:
                yaw_delta.grad.mul_(active_mask)
            _clear_adam_rows_(optimizer, optimizer_parameters, frozen_mask)
            torch.nn.utils.clip_grad_norm_(optimizer_parameters, max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                translation[frozen_mask] = translation_before_step[frozen_mask]
                if optimize_yaw:
                    yaw_delta[frozen_mask] = yaw_before_step[frozen_mask]
            scheduler.step()
        else:
            current_parameters = pack_pose_parameters(
                yaw_delta.detach(),
                translation.detach(),
            )
            if sceneproof_shadow_jacobian_ownership:
                jacobian_audit = run_sceneproof_jacobian_ownership_audit(
                    current_parameters
                )
                sceneproof_jacobian_audits.append(jacobian_audit)
                if not jacobian_audit["passed"]:
                    raise RuntimeError(
                        "SceneProof Jacobian block ownership audit failed: "
                        f"leakage={jacobian_audit['leakage']}, "
                        "uncovered="
                        f"{jacobian_audit['uncovered_residual_rows']}, "
                        "duplicated="
                        f"{jacobian_audit['duplicated_residual_rows']}"
                    )

            in_loop_guarded_accepted = False
            if (
                sceneproof_in_loop_guarded_schur
                and len(sceneproof_jacobian_audits)
                >= sceneproof_required_stable_linearizations
            ):
                if relation_coordinates is None:
                    raise RuntimeError(
                        "in-loop guarded Schur requires relation coordinates"
                    )
                in_loop_incumbent_pose = relation_coordinates.pose_matrices(
                    current_parameters
                ).detach()
                (
                    in_loop_pose,
                    in_loop_audit,
                ) = run_full_so3_guarded_schur_trial(
                    in_loop_incumbent_pose,
                    sceneproof_jacobian_audits[-1],
                )
                in_loop_audit["execution_phase"] = "lm_iteration"
                in_loop_audit["execution_iteration"] = int(iteration + 1)
                sceneproof_guarded_schur_audit = in_loop_audit
                in_loop_guarded_accepted = bool(in_loop_audit["accepted"])
                if in_loop_guarded_accepted:
                    sceneproof_guarded_pose_override = in_loop_pose.detach()
                    print(
                        "[SceneProof] In-loop full-SO(3) guarded Schur "
                        f"accepted: iteration={iteration + 1}, "
                        f"scale={in_loop_audit['accepted_scale']:.8g}, "
                        "objective_delta="
                        f"{in_loop_audit['selected_objective'] - in_loop_audit['objective_before']:.8g}",
                        flush=True,
                    )

            def cap_scene_step(direction: torch.Tensor) -> torch.Tensor:
                if solver == "v5_scenelm":
                    if (
                        relation_coordinates is None
                        or relation_active_objects is None
                    ):
                        raise RuntimeError("v5 SceneLM relation chart is missing")
                    return relation_coordinates.cap_step(
                        direction,
                        max_translation=lm_max_translation_step,
                        max_yaw_radians=math.radians(
                            lm_max_yaw_step_degrees
                        ),
                        active_objects=relation_active_objects,
                    )
                if optimize_yaw:
                    yaw_step = direction[:object_count]
                    translation_step = direction[object_count:].reshape(
                        object_count, 3
                    )
                else:
                    yaw_step = direction.new_zeros((object_count,))
                    translation_step = direction.reshape(object_count, 3)
                yaw_limit = math.radians(lm_max_yaw_step_degrees)
                yaw_scale = yaw_limit / torch.clamp(
                    yaw_step.abs(), min=yaw_limit
                )
                translation_norm = torch.linalg.vector_norm(
                    translation_step, dim=1
                )
                translation_scale = lm_max_translation_step / torch.clamp(
                    translation_norm, min=lm_max_translation_step
                )
                object_scale = torch.minimum(yaw_scale, translation_scale)
                if optimize_yaw:
                    return torch.cat(
                        (
                            yaw_step * object_scale,
                            (
                                translation_step * object_scale[:, None]
                            ).reshape(-1),
                        )
                    )
                return (translation_step * object_scale[:, None]).reshape(-1)

            if in_loop_guarded_accepted:
                # The accepted pose lives in the independent full-SO(3)
                # chart and is the final candidate for this guarded Smoke
                # path.  Do not round-trip it through the yaw-only legacy
                # relation chart.  Keep the incumbent parameters untouched
                # solely so existing history/certificate plumbing remains
                # well-defined; the pose override owns the emitted result.
                next_parameters = current_parameters
                lm_last_diagnostics = {
                    "accepted": 0.0,
                    "energy": float(in_loop_audit["objective_before"]),
                    "actual_reduction": 0.0,
                    "predicted_reduction": 0.0,
                    "reduction_ratio": 0.0,
                    "gradient_inf": float(
                        in_loop_audit["stationarity_certificate"][
                            "projected_gradient_inf"
                        ]
                    ),
                    "step_norm": float(
                        max(
                            in_loop_audit[
                                "selected_rotation_step_max_radians"
                            ],
                            in_loop_audit[
                                "selected_translation_step_max_m"
                            ],
                        )
                    ),
                    "iterations": 0.0,
                    "relative_residual": 0.0,
                }
                accepted = False
            else:
                next_parameters, lm_last_diagnostics = matrix_free_lm_step(
                    dense_factor_residuals,
                    current_parameters,
                    damping=lm_damping,
                    pcg_iterations=lm_pcg_iterations,
                    pcg_tolerance=lm_pcg_tolerance,
                    acceptance_threshold=lm_acceptance_threshold,
                    step_transform=cap_scene_step,
                    parameter_mask=(
                        relation_coordinates.parameter_mask_from_objects(
                            relation_active_objects
                        )
                        if solver == "v5_scenelm"
                        and relation_coordinates is not None
                        and relation_active_objects is not None
                        else None
                    ),
                )
                accepted = bool(lm_last_diagnostics["accepted"])
                if accepted:
                    lm_accepted_steps += 1
                    if lm_last_diagnostics["reduction_ratio"] > 0.75:
                        lm_damping = max(lm_damping * 0.5, 1e-8)
                    elif lm_last_diagnostics["reduction_ratio"] < 0.25:
                        lm_damping = min(lm_damping * 2.0, 1e8)
                else:
                    lm_rejected_steps += 1
                    lm_damping = min(lm_damping * 4.0, 1e8)
            next_yaw, next_translation = unpack_pose_parameters(
                next_parameters
            )
            if solver == "v5_scenelm":
                relation_parameters = next_parameters.detach()
                yaw_delta = next_yaw
                translation = next_translation
            else:
                with torch.no_grad():
                    translation.copy_(next_translation)
                    if optimize_yaw:
                        yaw_delta.copy_(next_yaw)
            relative_reduction = (
                lm_last_diagnostics["actual_reduction"]
                / max(lm_last_diagnostics["energy"], 1e-12)
                if accepted
                else 0.0
            )
            if accepted and relative_reduction < lm_relative_energy_tolerance:
                lm_small_reduction_count += 1
            elif accepted:
                lm_small_reduction_count = 0
            lm_should_stop = bool(
                in_loop_guarded_accepted
                or lm_last_diagnostics["gradient_inf"] < lm_gradient_tolerance
                or lm_small_reduction_count >= lm_patience
            )
        active_step_total += int(torch.count_nonzero(active_mask).item())

        # The already-validated contact/plane projections remain hard
        # constraints.  Original LayoutVLM also projects on-top centers into
        # the parent's planar footprint.
        if solver != "v5_scenelm":
            project_support_contacts_(
                yaw_delta,
                translation,
                base_matrices,
                local_corners,
                active_support_projection_pairs,
                active_fixed_support_indices,
                active_fixed_support_heights,
            )
            project_support_footprints_(
                yaw_delta,
                translation,
                base_matrices,
                local_corners,
                active_containment_projection_pairs,
                footprint_hull_sizes=footprint_hull_sizes,
            )
            project_fixed_planes_(
                yaw_delta,
                translation,
                base_matrices,
                local_corners,
                active_plane_object_indices,
                active_plane_points,
                active_plane_normals,
            )
            project_room_boundary_(
                yaw_delta,
                translation,
                base_matrices,
                local_corners,
                active_boundary_object_indices,
                boundary_points,
                boundary_normals,
            )
        with torch.no_grad():
            translation_step = torch.linalg.vector_norm(
                translation - translation_before_step,
                dim=1,
            )
            yaw_step = torch.abs(yaw_delta - yaw_before_step)
            translation_update_ema.mul_(0.8).add_(0.2 * translation_step)
            yaw_update_ema.mul_(0.8).add_(0.2 * yaw_step)

        if solver == "v5_scenelm":
            if (
                relation_coordinates is None
                or relation_parameters is None
                or relation_active_objects is None
                or relation_stable_steps is None
            ):
                raise RuntimeError("v5 SceneLM active-set state is missing")
            relation_active_step_total += int(
                torch.count_nonzero(relation_active_objects).item()
            )
            numerical_lm_should_stop = bool(lm_should_stop)
            with torch.no_grad():
                certified_pose = relation_coordinates.pose_matrices(
                    relation_parameters
                )
                certified_residuals = collect_residuals(certified_pose)
                certified_safe = active_set_safe_mask(
                    certified_residuals,
                    translation_step,
                    yaw_step,
                    thresholds=active_set_thresholds,
                )
                accepted_step = bool(
                    lm_last_diagnostics.get("accepted", 0.0)
                )
                stable_now = (
                    relation_active_objects
                    & certified_safe
                    & accepted_step
                )
                relation_stable_steps[stable_now] += 1
                relation_stable_steps[
                    relation_active_objects & ~stable_now
                ] = 0
                freeze_mask = (
                    relation_active_objects
                    & (relation_stable_steps >= lm_patience)
                )
                if torch.any(freeze_mask):
                    relation_active_objects[freeze_mask] = False
                    relation_freeze_count += int(
                        torch.count_nonzero(freeze_mask).item()
                    )
                wake_mask = (~relation_active_objects) & (~certified_safe)
                if relation_release_count:
                    release_scope = torch.zeros_like(wake_mask)
                    if relation_released_objects:
                        release_scope[
                            torch.as_tensor(
                                sorted(relation_released_objects),
                                dtype=torch.long,
                                device=wake_mask.device,
                            )
                        ] = True
                    wake_mask &= release_scope
                if torch.any(wake_mask):
                    relation_active_objects[wake_mask] = True
                    relation_stable_steps[wake_mask] = 0
                    relation_wakeup_count += int(
                        torch.count_nonzero(wake_mask).item()
                    )
                collision_threshold = float(
                    (active_set_thresholds or {}).get("collision", 1.0e-4)
                )
                release_mask = collision_connected_release_mask(
                    certified_residuals["collision"],
                    threshold=collision_threshold,
                )
                should_release = bool(
                    numerical_lm_should_stop
                    and not torch.all(certified_safe).item()
                    and torch.any(release_mask).item()
                    and relation_release_count < lm_max_relation_releases
                )
                if should_release:
                    _, release_pair_residuals = oriented_penetration_loss(
                        certified_pose,
                        local_corners,
                        collision_pairs,
                        footprint_hull_sizes,
                    )
                    unsafe_pair_mask = (
                        release_pair_residuals > collision_threshold
                    )
                    collision_witness_pairs = collision_pairs[
                        unsafe_pair_mask
                    ]
                    collision_witness_axes = (
                        collision_separation_witness_axes(
                            certified_pose,
                            local_corners,
                            collision_witness_pairs,
                            footprint_hull_sizes,
                        )
                    )
                    release_indices = torch.nonzero(
                        release_mask, as_tuple=False
                    ).reshape(-1)
                    relaxed_indices = sorted(
                        set(relation_coordinates.relaxed_object_indices)
                        | set(release_indices.detach().cpu().tolist())
                    )
                    # Rebase at the last certified pose so zero parameters in
                    # the expanded chart preserve the incumbent exactly.
                    relation_coordinates = compile_relation_coordinates(
                        certified_pose.detach(),
                        support_pairs,
                        fixed_support_indices,
                        plane_object_indices,
                        plane_normals,
                        optimise_yaw=optimize_yaw,
                        free_object_indices=relaxed_indices,
                        warm_start_anchored_plane_translation=(
                            sceneproof_warm_start_anchored_plane_translation
                        ),
                    )
                    relation_parameters = relation_coordinates.zero_parameters()
                    relation_active_objects = release_mask.clone()
                    relation_stable_steps.zero_()
                    relation_release_count += 1
                    relation_released_objects.update(
                        int(value)
                        for value in release_indices.detach().cpu().tolist()
                    )
                    relation_release_iterations.append(iteration + 1)
                    lm_damping = float(lm_initial_damping)
                    lm_small_reduction_count = 0
                    lm_should_stop = False
                    print(
                        "[SceneLM] Collision-connected relation release: "
                        f"iteration={iteration + 1}, "
                        f"release={relation_release_count}/"
                        f"{lm_max_relation_releases}, "
                        f"objects={release_indices.detach().cpu().tolist()}, "
                        f"witnesses={collision_witness_pairs.shape[0]}, "
                        f"damping_reset={lm_damping:.8g}",
                        flush=True,
                    )
                else:
                    lm_should_stop = certified_lm_convergence(
                        numerical_lm_should_stop
                        or not torch.any(relation_active_objects),
                        certified_safe,
                    )

        if active_set_router and (iteration + 1) in checkpoints:
            with torch.no_grad():
                checkpoint_pose = reproject_pose_matrices(
                    base_matrices, yaw_delta, translation
                )
                checkpoint_residuals = collect_residuals(checkpoint_pose)
                checkpoint_safe = active_set_safe_mask(
                    checkpoint_residuals,
                    translation_update_ema,
                    yaw_update_ema,
                    thresholds=active_set_thresholds,
                )
                checkpoint_safe &= ~protected_objects
                if iteration + 1 == checkpoints[0]:
                    freeze_mask = active_mask & checkpoint_safe
                    allocated_budget[freeze_mask] = checkpoints[0]
                    freeze_count_30 += int(
                        torch.count_nonzero(freeze_mask).item()
                    )
                else:
                    freeze_mask = (
                        active_mask
                        & checkpoint_safe
                        & (allocated_budget > checkpoints[0])
                    )
                    allocated_budget[freeze_mask] = checkpoints[1]
                    freeze_count_100 += int(
                        torch.count_nonzero(freeze_mask).item()
                    )
                _clear_adam_rows_(
                    optimizer,
                    optimizer_parameters,
                    freeze_mask,
                )
                if torch.any(freeze_mask):
                    active_set_revision += 1
                    active_horizon = int(allocated_budget.max().item())
                    router_has_frozen = bool(
                        torch.any(
                            allocated_budget <= iteration + 1
                        ).item()
                    )

        if (
            iteration == 0
            or iteration == iterations - 1
            or (iteration + 1) % 25 == 0
            or (active_set_router and (iteration + 1) in checkpoints)
            or (is_lm_solver and lm_should_stop)
        ):
            history.append(
                {
                    "iteration": float(iteration + 1),
                    "total": float(total.detach().item()),
                    "collision": float(collision.detach().item()),
                    "contact": float(contact.detach().item()),
                    "max_contact_gap": float(
                        contact_gaps.detach().abs().amax().item()
                        if contact_gaps.numel()
                        else 0.0
                    ),
                    "plane": float(plane.detach().item()),
                    "max_plane_gap": float(
                        plane_gaps.detach().abs().amax().item()
                        if plane_gaps.numel()
                        else 0.0
                    ),
                    "orientation": float(orientation.detach().item()),
                    "max_orientation_error": float(
                        alignment_errors.detach().amax().item()
                        if alignment_errors.numel()
                        else 0.0
                    ),
                    "containment": float(containment.detach().item()),
                    "max_containment_error": float(
                        torch.sqrt(containment_errors.detach()).amax().item()
                        if containment_errors.numel()
                        else 0.0
                    ),
                    "semantic": float(semantic.detach().item()),
                    "distance": float(distance.detach().item()),
                    "max_distance": float(
                        distances.detach().amax().item()
                        if distances.numel()
                        else 0.0
                    ),
                    "align": float(align.detach().item()),
                    "point": float(point.detach().item()),
                    "boundary": float(boundary.detach().item()),
                    "max_boundary_error": float(
                        boundary_errors.detach().amax().item()
                        if boundary_errors.numel()
                        else 0.0
                    ),
                    "depth_reprojection": float(
                        depth_reprojection.detach().item()
                    ),
                    "mean_depth_bbox_centre_error_px": float(
                        depth_centre_errors.detach().mean().item()
                        if depth_centre_errors.numel()
                        else 0.0
                    ),
                    "mean_depth_bbox_size_log_error": float(
                        depth_size_errors.detach().mean().item()
                        if depth_size_errors.numel()
                        else 0.0
                    ),
                    "mean_depth_relative_error": float(
                        depth_relative_errors.detach().mean().item()
                        if depth_relative_errors.numel()
                        else 0.0
                    ),
                    "depth_trust_region": float(
                        depth_trust_region.detach().item()
                    ),
                    "max_depth_bbox_centre_excess": float(
                        depth_centre_excess.detach().amax().item()
                        if depth_centre_excess.numel()
                        else 0.0
                    ),
                    "max_depth_bbox_size_excess": float(
                        depth_size_excess.detach().amax().item()
                        if depth_size_excess.numel()
                        else 0.0
                    ),
                    "max_depth_relative_excess": float(
                        depth_relative_excess.detach().amax().item()
                        if depth_relative_excess.numel()
                        else 0.0
                    ),
                    "warm_start": float(warm_start.detach().item()),
                    "penetrating_pairs": float(
                        torch.count_nonzero(per_pair.detach() > 0).item()
                    ),
                }
            )
            history[-1]["solver"] = solver
            if is_lm_solver:
                history[-1].update(
                    {
                        "lm_accepted": lm_last_diagnostics.get(
                            "accepted", 0.0
                        ),
                        "lm_damping": float(lm_damping),
                        "lm_gradient_inf": lm_last_diagnostics.get(
                            "gradient_inf", 0.0
                        ),
                        "lm_step_norm": lm_last_diagnostics.get(
                            "step_norm", 0.0
                        ),
                        "lm_reduction_ratio": lm_last_diagnostics.get(
                            "reduction_ratio", 0.0
                        ),
                        "lm_pcg_iterations": lm_last_diagnostics.get(
                            "iterations", 0.0
                        ),
                        "lm_pcg_relative_residual": lm_last_diagnostics.get(
                            "relative_residual", 0.0
                        ),
                        "lm_accepted_steps": float(lm_accepted_steps),
                        "lm_rejected_steps": float(lm_rejected_steps),
                        "lm_converged": float(lm_should_stop),
                    }
                )
        if is_lm_solver and lm_should_stop:
            break

    if restore_best_state:
        if best_yaw_delta is None or best_translation is None:
            raise RuntimeError("best-state restoration had no evaluated state")
        with torch.no_grad():
            yaw_delta.copy_(best_yaw_delta)
            translation.copy_(best_translation)

    if solver == "v5_scenelm":
        if relation_coordinates is None or relation_parameters is None:
            raise RuntimeError("v5 SceneLM final state is missing")
        final_yaw, final_translation = relation_coordinates.decode(
            relation_parameters
        )
        yaw_delta = final_yaw.detach().clone()
        translation = final_translation.detach().clone()

    if (
        sceneproof_full_so3_guarded_schur
        and not sceneproof_in_loop_guarded_schur
    ):
        if (
            solver != "v5_scenelm"
            or relation_coordinates is None
            or relation_parameters is None
            or not sceneproof_jacobian_audits
        ):
            raise RuntimeError("guarded Schur final audit state is missing")
        # Match the established SceneLM incumbent exactly before rebasing the
        # full-SO(3) chart.  A rejected trial therefore returns the same hard-
        # projected pose that the legacy v5 path would have emitted.
        project_support_contacts_(
            yaw_delta,
            translation,
            base_matrices,
            local_corners,
            support_pairs,
            fixed_support_indices,
            fixed_support_heights,
        )
        containment_projection_abstentions.extend(project_support_footprints_(
            yaw_delta,
            translation,
            base_matrices,
            local_corners,
            containment_pairs,
            footprint_hull_sizes=footprint_hull_sizes,
            infeasible_policy="restore_warm_start_planar",
        ))
        project_fixed_planes_(
            yaw_delta,
            translation,
            base_matrices,
            local_corners,
            plane_object_indices,
            plane_points,
            plane_normals,
        )
        project_room_boundary_(
            yaw_delta,
            translation,
            base_matrices,
            local_corners,
            boundary_object_indices,
            boundary_points,
            boundary_normals,
        )
        incumbent_pose = reproject_pose_matrices(
            base_matrices,
            yaw_delta,
            translation,
        ).detach()
        (
            sceneproof_guarded_pose_override,
            sceneproof_guarded_schur_audit,
        ) = run_full_so3_guarded_schur_trial(
            incumbent_pose,
            sceneproof_jacobian_audits[-1],
        )
        print(
            "[SceneProof] Full-SO(3) guarded Schur trial: "
            f"accepted={sceneproof_guarded_schur_audit['accepted']}, "
            "eligible="
            f"{sceneproof_guarded_schur_audit.get('audited_allowed_leaf_objects', [])}, "
            "eliminated="
            f"{sceneproof_guarded_schur_audit.get('schur', {}).get('eliminated_leaf_objects', [])}, "
            "collision_failures="
            f"{sceneproof_guarded_schur_audit.get('collision_guard', {}).get('failed_factor_ids', [])}, "
            "component_failures="
            f"{sorted(sceneproof_guarded_schur_audit.get('selected_component_failures', {}))}, "
            "incumbent_restored="
            f"{sceneproof_guarded_schur_audit['incumbent_restored']}",
            flush=True,
        )

    if sceneproof_guarded_pose_override is None:
        project_support_contacts_(
            yaw_delta,
            translation,
            base_matrices,
            local_corners,
            support_pairs,
            fixed_support_indices,
            fixed_support_heights,
        )
        containment_projection_abstentions.extend(project_support_footprints_(
            yaw_delta,
            translation,
            base_matrices,
            local_corners,
            containment_pairs,
            footprint_hull_sizes=footprint_hull_sizes,
            infeasible_policy=(
                "restore_warm_start_planar"
                if solver == "v5_scenelm"
                else "raise"
            ),
        ))
        project_fixed_planes_(
            yaw_delta,
            translation,
            base_matrices,
            local_corners,
            plane_object_indices,
            plane_points,
            plane_normals,
        )
        project_room_boundary_(
            yaw_delta,
            translation,
            base_matrices,
            local_corners,
            boundary_object_indices,
            boundary_points,
            boundary_normals,
        )
        final_pose_matrices = reproject_pose_matrices(
            base_matrices,
            yaw_delta,
            translation,
        )
    else:
        final_pose_matrices = sceneproof_guarded_pose_override
    plane_proxy_abstain_audit = None
    if plane_proxy_abstain_indices.numel():
        with torch.no_grad():
            final_pose_matrices[plane_proxy_abstain_indices] = base_matrices[
                plane_proxy_abstain_indices
            ]
        plane_proxy_abstain_audit = {
            "policy": (
                "missing_orientation_witness_preserve_full_warm_start_pose"
                if sceneproof_plane_attach_requires_witness
                else "proxy_incompatible_preserve_full_warm_start_pose"
            ),
            "gap_threshold_m": float(sceneproof_plane_proxy_abstain_gap_m),
            "object_indices": plane_proxy_abstain_indices.detach().cpu().tolist(),
            "initial_signed_gaps_m": (
                plane_proxy_abstain_initial_gaps.detach().cpu().tolist()
            ),
            "objects": int(plane_proxy_abstain_indices.numel()),
            "rotation_frozen": True,
            "translation_frozen": True,
            "orientation_witness_required": bool(
                sceneproof_plane_attach_requires_witness
            ),
        }
        print(
            "[SceneProof] Plane proxy abstention: "
            f"objects={plane_proxy_abstain_audit['objects']}, "
            f"gap_threshold_m={sceneproof_plane_proxy_abstain_gap_m:.8g}, "
            "pose_policy=preserve_full_warm_start",
            flush=True,
        )
    plane_anchor_audit = None
    if sceneproof_warm_start_anchored_plane_translation:
        plane_anchor_audit = enforce_warm_start_plane_translation_trust_(
            final_pose_matrices,
            base_matrices,
            plane_object_indices,
            plane_normals,
            normal_limit_m=sceneproof_plane_anchor_normal_limit_m,
        )
        print(
            "[SceneProof] Warm-start plane translation trust: "
            f"objects={plane_anchor_audit['objects']}, "
            f"normal_limit_m={plane_anchor_audit['normal_limit_m']:.8g}, "
            f"pre_max_tangent_m={plane_anchor_audit['pre_max_tangent_m']:.8g}, "
            f"pre_max_normal_m={plane_anchor_audit['pre_max_normal_m']:.8g}, "
            f"post_max_tangent_m={plane_anchor_audit['post_max_tangent_m']:.8g}, "
            f"post_max_normal_m={plane_anchor_audit['post_max_normal_m']:.8g}",
            flush=True,
        )
    plane_sibling_audit = None
    if sceneproof_plane_sibling_tangent_projection:
        plane_sibling_audit = project_plane_sibling_tangent_intervals_(
            final_pose_matrices,
            local_corners,
            plane_object_indices,
            plane_points,
            plane_normals,
            collision_pairs,
            footprint_hull_sizes,
            object_ids=sceneproof_object_ids,
            maximum_shift_m=sceneproof_plane_sibling_max_shift_m,
        )
        print(
            "[SceneProof] Plane sibling tangent projection: "
            f"accepted={plane_sibling_audit['accepted']}, "
            f"groups={plane_sibling_audit['groups_considered']}, "
            f"components={plane_sibling_audit['components_considered']}, "
            f"accepted_components={plane_sibling_audit.get('components_accepted', 0)}, "
            f"abstained_components={plane_sibling_audit.get('components_abstained', 0)}, "
            f"objects_moved={plane_sibling_audit['objects_moved']}, "
            f"max_shift_m={plane_sibling_audit['maximum_shift_m']:.8g}, "
            f"collision_nonworsening={plane_sibling_audit['collision_nonworsening']}, "
            f"reason={plane_sibling_audit['reason']}",
            flush=True,
        )
    plane_image_gauge_audit = None
    if (
        sceneproof_plane_component_image_gauge
        and plane_sibling_audit is not None
    ):
        plane_image_gauge_audit = refine_plane_component_image_gauge_(
            final_pose_matrices,
            base_matrices,
            local_corners,
            plane_sibling_audit,
            plane_object_indices,
            plane_normals,
            collision_pairs,
            footprint_hull_sizes,
            sceneproof_image_observation_indices,
            sceneproof_image_observed_boxes,
            sceneproof_image_observed_depths,
            sceneproof_image_observed_weights,
            sceneproof_image_bbox_size_enabled,
            sceneproof_image_world_to_camera,
            sceneproof_image_size,
            support_pairs=support_pairs,
            fixed_support_indices=fixed_support_indices,
            fixed_support_heights=fixed_support_heights,
            containment_pairs=containment_pairs,
            plane_points=plane_points,
            plane_orientation_mask=plane_orientation_mask,
            boundary_object_indices=boundary_object_indices,
            boundary_points=boundary_points,
            boundary_normals=boundary_normals,
            maximum_total_shift_m=sceneproof_plane_sibling_max_shift_m,
            centre_noharm_margin_px=depth_centre_margin_pixels,
        )
        print(
            "[SceneProof] Plane component image gauge: "
            f"accepted_components={plane_image_gauge_audit['components_accepted']}, "
            f"abstained_components={plane_image_gauge_audit['components_abstained']}, "
            f"objects_moved={plane_image_gauge_audit['objects_moved']}, "
            f"reason={plane_image_gauge_audit['reason']}",
            flush=True,
        )
    _, final_collision_values = oriented_penetration_loss(
        final_pose_matrices,
        local_corners,
        collision_pairs,
        footprint_hull_sizes,
    )
    _, final_contact_gaps = support_contact_loss(
        final_pose_matrices,
        local_corners,
        support_pairs,
        fixed_support_indices,
        fixed_support_heights,
    )
    _, _, final_plane_gaps, _ = fixed_plane_loss(
        final_pose_matrices,
        local_corners,
        plane_object_indices,
        plane_points,
        plane_normals,
        plane_orientation_mask,
    )
    _, final_containment_errors = support_planar_containment_loss(
        final_pose_matrices,
        local_corners,
        containment_pairs,
        footprint_hull_sizes,
    )
    _, final_boundary_errors = room_boundary_loss(
        final_pose_matrices,
        local_corners,
        boundary_object_indices,
        boundary_points,
        boundary_normals,
    )
    (
        final_depth_reprojection,
        final_depth_centre_errors,
        final_depth_size_errors,
        final_depth_relative_errors,
    ) = depth_aware_reprojection_loss(
        final_pose_matrices,
        local_corners,
        depth_observation_indices,
        depth_observed_boxes,
        depth_observed_depths,
        depth_observed_weights,
        depth_bbox_size_enabled,
        depth_world_to_camera,
        depth_image_size,
        centre_weight=depth_centre_weight,
        size_weight=depth_size_weight,
        metric_depth_weight=depth_metric_weight,
    )
    (
        final_depth_trust_region,
        final_depth_centre_excess,
        final_depth_size_excess,
        final_depth_relative_excess,
    ) = no_harm_reprojection_penalty(
        final_depth_centre_errors,
        final_depth_size_errors,
        final_depth_relative_errors,
        depth_reference_centre_errors,
        depth_reference_size_errors,
        depth_reference_relative_errors,
        depth_observed_weights,
        centre_margin_pixels=depth_centre_margin_pixels,
        size_margin_log=depth_size_margin_log,
        depth_margin_log=depth_relative_margin_log,
    )
    if history:
        history[-1]["projected_max_collision_penetration"] = float(
            final_collision_values.detach().amax().item()
            if final_collision_values.numel()
            else 0.0
        )
        history[-1]["projected_penetrating_pairs"] = float(
            torch.count_nonzero(final_collision_values.detach() > 0).item()
        )
        history[-1]["projected_max_contact_gap"] = float(
            final_contact_gaps.detach().abs().amax().item()
            if final_contact_gaps.numel()
            else 0.0
        )
        history[-1]["projected_max_plane_gap"] = float(
            final_plane_gaps.detach().abs().amax().item()
            if final_plane_gaps.numel()
            else 0.0
        )
        history[-1]["projected_max_containment_error"] = float(
            torch.sqrt(final_containment_errors.detach()).amax().item()
            if final_containment_errors.numel()
            else 0.0
        )
        history[-1]["projected_max_boundary_error"] = float(
            final_boundary_errors.detach().amax().item()
            if final_boundary_errors.numel()
            else 0.0
        )
        certified_support_pairs: list[list[int]] = []
        if containment_pairs.shape[0]:
            pair_matches = (
                containment_pairs[:, None, :]
                == support_pairs[None, :, :]
            ).all(dim=-1)
            if not bool(pair_matches.any(dim=1).all().item()):
                raise RuntimeError(
                    "a containment pair is missing from support pairs"
                )
            support_positions = pair_matches.to(torch.int64).argmax(dim=1)
            containment_metres = torch.sqrt(
                torch.clamp_min(final_containment_errors.detach(), 0.0)
            )
            contact_metres = final_contact_gaps.detach()[support_positions].abs()
            certified_mask = (
                (containment_metres <= 0.05)
                & (contact_metres <= 0.05)
            )
            certified_support_pairs = containment_pairs[
                certified_mask
            ].detach().cpu().tolist()
        history[-1]["post_projection_certified_support_pairs"] = (
            certified_support_pairs
        )
        history[-1]["final_depth_reprojection"] = float(
            final_depth_reprojection.detach().item()
        )
        history[-1]["final_mean_depth_bbox_centre_error_px"] = float(
            final_depth_centre_errors.detach().mean().item()
            if final_depth_centre_errors.numel()
            else 0.0
        )
        history[-1]["final_mean_depth_bbox_size_log_error"] = float(
            final_depth_size_errors.detach().mean().item()
            if final_depth_size_errors.numel()
            else 0.0
        )
        history[-1]["final_mean_depth_relative_error"] = float(
            final_depth_relative_errors.detach().mean().item()
            if final_depth_relative_errors.numel()
            else 0.0
        )
        history[-1]["final_depth_trust_region"] = float(
            final_depth_trust_region.detach().item()
        )
        history[-1]["final_max_depth_bbox_centre_excess"] = float(
            final_depth_centre_excess.detach().amax().item()
            if final_depth_centre_excess.numel()
            else 0.0
        )
        history[-1]["final_max_depth_bbox_size_excess"] = float(
            final_depth_size_excess.detach().amax().item()
            if final_depth_size_excess.numel()
            else 0.0
        )
        history[-1]["final_max_depth_relative_excess"] = float(
            final_depth_relative_excess.detach().amax().item()
            if final_depth_relative_excess.numel()
            else 0.0
        )
        if restore_best_state:
            history[-1]["best_iteration"] = float(best_iteration)
            history[-1]["best_total"] = best_total
        history[-1]["yaw_optimized"] = float(optimize_yaw)
        history[-1]["solver"] = solver
        history[-1]["solver_executed_iterations"] = float(
            executed_iterations
        )
        history[-1]["lm_accepted_steps"] = float(lm_accepted_steps)
        history[-1]["lm_rejected_steps"] = float(lm_rejected_steps)
        history[-1]["lm_final_damping"] = float(lm_damping)
        history[-1]["lm_converged"] = float(
            is_lm_solver and lm_should_stop
        )
        history[-1]["active_set_router"] = float(active_set_router)
        history[-1]["router_executed_iterations"] = float(executed_iterations)
        history[-1]["router_active_step_total"] = float(active_step_total)
        history[-1]["router_dense_step_total"] = float(
            object_count * iterations
        )
        history[-1]["router_iteration_reduction"] = float(
            1.0 - active_step_total / max(object_count * iterations, 1)
            if active_set_router
            else 0.0
        )
        history[-1]["router_budget_30"] = float(
            torch.count_nonzero(allocated_budget == checkpoints[0]).item()
            if active_set_router
            else 0
        )
        history[-1]["router_budget_100"] = float(
            torch.count_nonzero(allocated_budget == checkpoints[1]).item()
            if active_set_router
            else 0
        )
        history[-1]["router_budget_full"] = float(
            torch.count_nonzero(allocated_budget == iterations).item()
            if active_set_router
            else object_count
        )
        history[-1]["router_freeze_30"] = float(freeze_count_30)
        history[-1]["router_freeze_100"] = float(freeze_count_100)
        history[-1]["router_wakeups"] = float(wakeup_count)
        history[-1]["router_protected_objects"] = float(
            torch.count_nonzero(protected_objects).item()
        )
        history[-1]["router_allocated_budgets"] = [
            int(value) for value in allocated_budget.detach().cpu().tolist()
        ]
        history[-1]["router_constraint_degree"] = [
            int(value) for value in constraint_degree.detach().cpu().tolist()
        ]
        history[-1]["sceneproof_shadow_residual_parity"] = float(
            sceneproof_shadow_residual_parity
            or sceneproof_use_program_residuals
        )
        history[-1]["sceneproof_shadow_residual_checks"] = float(
            sceneproof_shadow_checks
        )
        history[-1]["sceneproof_shadow_residual_max_abs_error"] = float(
            sceneproof_shadow_max_abs_error
        )
        history[-1]["sceneproof_program_residual_input"] = float(
            sceneproof_use_program_residuals
        )
        history[-1]["sceneproof_program_residual_selections"] = float(
            sceneproof_program_residual_selections
        )
        history[-1]["sceneproof_residual_fallbacks"] = float(
            sceneproof_residual_fallbacks
        )
        if sceneproof_jacobian_audits:
            final_jacobian_audit = sceneproof_jacobian_audits[-1]
            history[-1]["sceneproof_jacobian_ownership"] = {
                "schema_version": "sceneproof_jacobian_ownership_audit_v1",
                "passed": all(
                    bool(record["passed"])
                    for record in sceneproof_jacobian_audits
                ),
                "checks": len(sceneproof_jacobian_audits),
                "maximum_leakage": max(
                    (
                        max(
                            (
                                float(value["max_abs_leakage"])
                                for value in record["leakage"]
                            ),
                            default=0.0,
                        )
                        for record in sceneproof_jacobian_audits
                    ),
                    default=0.0,
                ),
                "final": final_jacobian_audit,
                "trial_collision_guard": (
                    "enabled"
                    if sceneproof_full_so3_guarded_schur
                    else "implemented_not_enabled"
                ),
            }
        if sceneproof_guarded_schur_audit is not None:
            history[-1]["sceneproof_full_so3_guarded_schur"] = (
                sceneproof_guarded_schur_audit
            )
        if plane_anchor_audit is not None:
            history[-1]["sceneproof_plane_translation_anchor"] = (
                plane_anchor_audit
            )
        if plane_proxy_abstain_audit is not None:
            history[-1]["sceneproof_plane_proxy_abstention"] = (
                plane_proxy_abstain_audit
            )
        if plane_sibling_audit is not None:
            history[-1]["sceneproof_plane_sibling_tangent_projection"] = (
                plane_sibling_audit
            )
        if plane_image_gauge_audit is not None:
            history[-1]["sceneproof_plane_component_image_gauge"] = (
                plane_image_gauge_audit
            )
        history[-1]["sceneproof_containment_projection_abstentions"] = list(
            containment_projection_abstentions
        )
        history[-1]["sceneproof_containment_projection_abstention_count"] = int(
            len(containment_projection_abstentions)
        )
        if is_lm_solver:
            if solver == "v5_scenelm":
                if relation_parameters is None:
                    raise RuntimeError("v5 SceneLM certificate state is missing")
                certificate_parameters = relation_parameters.detach()
            else:
                certificate_parameters = pack_pose_parameters(
                    yaw_delta.detach(), translation.detach()
                )
            with torch.enable_grad():
                certificate_parameters = certificate_parameters.requires_grad_(
                    True
                )
                certificate_residuals = dense_factor_residuals(
                    certificate_parameters
                )
                certificate_energy = 0.5 * certificate_residuals.square().sum()
                certificate_gradient = torch.autograd.grad(
                    certificate_energy,
                    certificate_parameters,
                    allow_unused=False,
                )[0]
            history[-1]["lm_final_residual_energy"] = float(
                certificate_energy.detach().item()
            )
            history[-1]["certificate_stationarity_inf"] = float(
                certificate_gradient.detach().abs().amax().item()
                if certificate_gradient.numel()
                else 0.0
            )
            history[-1]["certificate_primal_max"] = max(
                history[-1]["projected_max_collision_penetration"],
                history[-1]["projected_max_contact_gap"],
                history[-1]["projected_max_plane_gap"],
                history[-1]["projected_max_containment_error"],
                history[-1]["projected_max_boundary_error"],
            )
        if solver == "v5_scenelm":
            if (
                relation_coordinates is None
                or relation_active_objects is None
            ):
                raise RuntimeError("v5 SceneLM audit state is missing")
            relation_metadata = relation_coordinates.metadata()
            history[-1]["relation_coordinates"] = relation_metadata
            history[-1]["relation_active_step_total"] = float(
                relation_active_step_total
            )
            history[-1]["relation_dense_step_total"] = float(
                object_count * executed_iterations
            )
            history[-1]["relation_active_reduction"] = float(
                1.0
                - relation_active_step_total
                / max(object_count * executed_iterations, 1)
            )
            history[-1]["relation_freezes"] = float(
                relation_freeze_count
            )
            history[-1]["relation_wakeups"] = float(
                relation_wakeup_count
            )
            history[-1]["relation_active_objects_final"] = float(
                torch.count_nonzero(relation_active_objects).item()
            )
            history[-1]["relation_release_count"] = float(
                relation_release_count
            )
            history[-1]["relation_released_object_indices"] = sorted(
                relation_released_objects
            )
            history[-1]["relation_release_iterations"] = list(
                relation_release_iterations
            )
            history[-1]["collision_witness_count"] = int(
                collision_witness_pairs.shape[0]
            )
            history[-1]["collision_witness_weight"] = float(
                lm_collision_witness_weight
            )
    return final_pose_matrices.detach(), history


def pair_index_tensor(
    pairs: Sequence[Tuple[int, int]],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a validated pair-index tensor, including the empty-pair case."""
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long, device=device)
    result = torch.as_tensor(pairs, dtype=torch.long, device=device)
    if result.ndim != 2 or result.shape[-1] != 2:
        raise ValueError("pairs must be a sequence of (first, second) indices")
    if torch.any(result < 0):
        raise ValueError("pair indices must be non-negative")
    return result


def initialize_pose_variables(
    base_matrices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create zero-yaw and warm-start translation optimization variables."""
    _validate_pose_batch(base_matrices)
    yaw_delta = torch.zeros(
        base_matrices.shape[0],
        dtype=base_matrices.dtype,
        device=base_matrices.device,
        requires_grad=True,
    )
    translation = (
        base_matrices[:, :3, 3]
        .clone()
        .detach()
        .requires_grad_(True)
    )
    return yaw_delta, translation


def identity_reprojection_error(base_matrices: torch.Tensor) -> torch.Tensor:
    """Maximum absolute error of a zero-delta pose round trip."""
    yaw_delta, translation = initialize_pose_variables(base_matrices)
    reprojected = reproject_pose_matrices(base_matrices, yaw_delta, translation)
    return torch.max(torch.abs(reprojected - base_matrices))


def stack_pose_matrices(
    matrices: Iterable[Iterable[Iterable[float]]],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert JSON/Blender-style nested matrices into a validated tensor."""
    result = torch.as_tensor(list(matrices), dtype=dtype, device=device)
    _validate_pose_batch(result)
    if not torch.isfinite(result).all():
        raise ValueError("pose matrices contain NaN or infinity")
    return result
