"""Independent SceneProof assembly of the legacy LayoutVLM residual vector."""

from __future__ import annotations

import math
from typing import Any

import torch

from modules._sceneproof_block_system import ResidualSliceBinding


def build_residual_slice_bindings(
    *,
    factor_bindings: list[dict[str, Any]],
    object_ids: list[str] | tuple[str, ...],
    collision_pairs: torch.Tensor,
    collision_witness_pairs: torch.Tensor,
    support_count: int,
    plane_count: int,
    orientation_object_indices: torch.Tensor,
    containment_count: int,
    distance_count: int,
    align_count: int,
    point_count: int,
    boundary_object_indices: torch.Tensor,
    depth_observation_indices: torch.Tensor,
    collision_weight: float,
    witness_weight: float,
    contact_weight: float,
    plane_weight: float,
    orientation_weight: float,
    containment_weight: float,
    semantic_weight: float,
    boundary_weight: float,
    depth_reprojection_weight: float,
    depth_trust_region_weight: float,
    optimize_yaw: bool,
    warm_start_weight: float,
) -> tuple[ResidualSliceBinding, ...]:
    """Mirror residual ordering while attaching stable factor/block owners."""
    id_to_index = {str(value): index for index, value in enumerate(object_ids)}
    by_channel: dict[str, list[dict[str, Any]]] = {}
    for binding in factor_bindings:
        by_channel.setdefault(str(binding["channel"]), []).append(binding)
    result: list[ResidualSliceBinding] = []
    cursor = 0

    def append_program_channel(channel: str, count: int, enabled: bool) -> None:
        nonlocal cursor
        records = by_channel.get(channel, [])
        if len(records) != count:
            raise ValueError(
                f"factor-binding count mismatch for {channel}: "
                f"{len(records)} != {count}"
            )
        if not enabled:
            return
        for record in records:
            variables = tuple(id_to_index[value] for value in record["variable_objects"])
            collision_pair = (
                tuple(id_to_index[value] for value in record["relation_key"][:2])
                if channel == "collision_oriented_penetration"
                else None
            )
            result.append(
                ResidualSliceBinding(
                    factor_id=str(record["factor_id"]),
                    channel=channel,
                    start=cursor,
                    stop=cursor + 1,
                    declared_object_indices=variables,
                    collision_pair=collision_pair,
                )
            )
            cursor += 1

    append_program_channel(
        "collision_oriented_penetration",
        int(collision_pairs.shape[0]),
        collision_weight > 0,
    )
    if collision_witness_pairs.shape[0] and witness_weight > 0:
        for first, second in collision_witness_pairs.detach().cpu().tolist():
            result.append(
                ResidualSliceBinding(
                    factor_id=f"system:collision_witness:{first}:{second}",
                    channel="collision_separating_witness",
                    start=cursor,
                    stop=cursor + 1,
                    declared_object_indices=(int(first), int(second)),
                    collision_pair=(int(first), int(second)),
                )
            )
            cursor += 1
    append_program_channel("support_contact_gap", support_count, contact_weight > 0)
    append_program_channel("plane_gap", plane_count, plane_weight > 0)
    append_program_channel(
        "plane_orientation_alignment",
        int(orientation_object_indices.numel()),
        orientation_weight > 0,
    )
    append_program_channel(
        "support_containment_error", containment_count, containment_weight > 0
    )
    append_program_channel(
        "semantic_distance_interval", distance_count, semantic_weight > 0
    )
    append_program_channel("semantic_align", align_count, semantic_weight > 0)
    append_program_channel(
        "semantic_point_towards", point_count, semantic_weight > 0
    )
    if boundary_object_indices.numel() and boundary_weight > 0:
        for object_index in boundary_object_indices.detach().cpu().tolist():
            result.append(
                ResidualSliceBinding(
                    factor_id=f"system:boundary:{object_index}",
                    channel="room_boundary",
                    start=cursor,
                    stop=cursor + 1,
                    declared_object_indices=(int(object_index),),
                )
            )
            cursor += 1
    depth_owners = tuple(
        sorted(set(int(value) for value in depth_observation_indices.detach().cpu().tolist()))
    )
    if depth_owners and depth_reprojection_weight > 0:
        result.append(
            ResidualSliceBinding(
                "system:depth_reprojection",
                "depth_reprojection",
                cursor,
                cursor + 1,
                depth_owners,
            )
        )
        cursor += 1
    if depth_owners and depth_trust_region_weight > 0:
        result.append(
            ResidualSliceBinding(
                "system:depth_trust_region",
                "depth_trust_region",
                cursor,
                cursor + 1,
                depth_owners,
            )
        )
        cursor += 1
    if warm_start_weight > 0:
        if optimize_yaw:
            for object_index in range(len(object_ids)):
                result.append(
                    ResidualSliceBinding(
                        f"system:warm_rotation:{object_index}",
                        "warm_rotation",
                        cursor,
                        cursor + 1,
                        (object_index,),
                    )
                )
                cursor += 1
        for object_index in range(len(object_ids)):
            result.append(
                ResidualSliceBinding(
                    f"system:warm_translation:{object_index}",
                    "warm_translation",
                    cursor,
                    cursor + 3,
                    (object_index,),
                )
            )
            cursor += 3
    return tuple(result)


def assemble_program_shadow_residuals(
    *,
    flattened: torch.Tensor,
    collision_values: torch.Tensor,
    collision_weight: float,
    collision_mass: int,
    witness_values: torch.Tensor,
    witness_weight: float,
    witness_mass: int,
    contact_values: torch.Tensor,
    contact_weight: float,
    contact_mass: int,
    plane_values: torch.Tensor,
    plane_weight: float,
    plane_mass: int,
    orientation_values: torch.Tensor,
    orientation_weight: float,
    containment_values: torch.Tensor,
    containment_weight: float,
    containment_mass: int,
    distance_values: torch.Tensor,
    align_values: torch.Tensor,
    point_values: torch.Tensor,
    semantic_weight: float,
    semantic_mass: int,
    boundary_values: torch.Tensor,
    boundary_weight: float,
    boundary_mass: int,
    current_depth: torch.Tensor,
    current_depth_trust: torch.Tensor,
    depth_observation_count: int,
    depth_reprojection_weight: float,
    depth_trust_region_weight: float,
    current_yaw: torch.Tensor,
    current_translation: torch.Tensor,
    base_matrices: torch.Tensor,
    optimize_yaw: bool,
    warm_start_weight: float,
) -> torch.Tensor:
    """Assemble the same vector independently from named program channels."""
    epsilon = 1e-12
    pieces: list[torch.Tensor] = []

    def squared(values: torch.Tensor, weight: float, mass: int) -> None:
        if values.numel() and weight > 0:
            pieces.append(
                values.reshape(-1)
                * math.sqrt(float(weight) / max(int(mass), 1))
            )

    def linear(values: torch.Tensor, weight: float, mass: int) -> None:
        if values.numel() and weight > 0:
            scaled = (
                torch.clamp(values.reshape(-1), min=0.0)
                * (float(weight) / max(int(mass), 1))
            )
            pieces.append(torch.sqrt(scaled + epsilon))

    linear(collision_values, collision_weight, collision_mass)
    squared(witness_values, witness_weight, witness_mass)
    squared(contact_values, contact_weight, contact_mass)
    squared(plane_values, plane_weight, plane_mass)
    squared(
        orientation_values,
        orientation_weight,
        int(orientation_values.numel()),
    )
    linear(containment_values, containment_weight, containment_mass)
    linear(distance_values, semantic_weight, semantic_mass)
    linear(align_values, semantic_weight, semantic_mass)
    linear(point_values, semantic_weight, semantic_mass)
    squared(boundary_values, boundary_weight, boundary_mass)
    if depth_observation_count:
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
        object_count = int(base_matrices.shape[0])
        scale = math.sqrt(float(warm_start_weight) / max(object_count, 1))
        if optimize_yaw:
            pieces.append(current_yaw * scale)
        pieces.append(
            (current_translation - base_matrices[:, :3, 3]).reshape(-1)
            * (scale / math.sqrt(3.0))
        )
    if not pieces:
        return flattened[:1] * 0.0
    return torch.cat(pieces)


def residual_parity(
    legacy: torch.Tensor,
    shadow: torch.Tensor,
    *,
    absolute_tolerance: float = 1e-10,
    relative_tolerance: float = 1e-8,
) -> dict[str, Any]:
    shape_match = tuple(legacy.shape) == tuple(shadow.shape)
    max_error = (
        float((legacy - shadow).detach().abs().amax().item())
        if shape_match and legacy.numel()
        else (0.0 if shape_match else float("inf"))
    )
    passed = shape_match and bool(
        torch.allclose(
            legacy.detach(),
            shadow.detach(),
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    )
    return {
        "passed": passed,
        "shape_match": shape_match,
        "legacy_size": int(legacy.numel()),
        "shadow_size": int(shadow.numel()),
        "max_abs_error": max_error,
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
    }
