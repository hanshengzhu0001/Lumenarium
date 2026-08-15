"""Explicit block assembly and safe leaf-translation Schur elimination."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

from modules._s4_scenelm_relational import (
    SO3RelationCoordinateSystem,
    block_schur_complement_solve,
)


@dataclass(frozen=True)
class LinearizedFactor:
    factor_id: str
    residual: torch.Tensor
    jacobians: Mapping[str, torch.Tensor]
    object_indices: tuple[int, ...]


@dataclass(frozen=True)
class ResidualSliceBinding:
    """Immutable ownership of one contiguous residual-vector slice."""

    factor_id: str
    channel: str
    start: int
    stop: int
    declared_object_indices: tuple[int, ...]
    collision_pair: tuple[int, int] | None = None

    @property
    def residual_slice(self) -> slice:
        return slice(self.start, self.stop)

    def validate(self, residual_count: int, object_count: int) -> None:
        if not self.factor_id or not self.channel:
            raise ValueError("factor_id and channel must be non-empty")
        if not 0 <= self.start < self.stop <= residual_count:
            raise ValueError("invalid residual slice")
        if not self.declared_object_indices:
            raise ValueError("a residual slice must declare variable owners")
        if any(not 0 <= value < object_count for value in self.declared_object_indices):
            raise ValueError("declared object index is out of range")
        if self.collision_pair is not None and len(self.collision_pair) != 2:
            raise ValueError("collision_pair must contain two object indices")


class LinearizationStabilityTracker:
    """Require factors to remain active across consecutive linearizations."""

    def __init__(self, required_consecutive: int = 2):
        if required_consecutive <= 0:
            raise ValueError("required_consecutive must be positive")
        self.required_consecutive = int(required_consecutive)
        self._counts: dict[str, int] = {}
        self._states: dict[str, bool] = {}
        self._state_streaks: dict[str, int] = {}

    def update(
        self,
        active_factor_ids: Sequence[str],
        known_factor_ids: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        active = set(active_factor_ids)
        known = set(self._counts) | active | set(known_factor_ids or ())
        for factor_id in known:
            state = factor_id in active
            self._counts[factor_id] = (
                self._counts.get(factor_id, 0) + 1
                if state
                else 0
            )
            if self._states.get(factor_id) == state:
                self._state_streaks[factor_id] = (
                    self._state_streaks.get(factor_id, 0) + 1
                )
            else:
                self._states[factor_id] = state
                self._state_streaks[factor_id] = 1
        return tuple(
            sorted(
                factor_id
                for factor_id, count in self._counts.items()
                if count >= self.required_consecutive
            )
        )

    def counts(self) -> dict[str, int]:
        return dict(sorted(self._counts.items()))

    def stable_inactive_factor_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                factor_id
                for factor_id, state in self._states.items()
                if not state
                and self._state_streaks.get(factor_id, 0)
                >= self.required_consecutive
            )
        )

    def unstable_factor_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                factor_id
                for factor_id in self._states
                if self._state_streaks.get(factor_id, 0)
                < self.required_consecutive
            )
        )


def audit_jacobian_block_ownership(
    *,
    residuals: torch.Tensor,
    jacobian: torch.Tensor,
    bindings: Sequence[ResidualSliceBinding],
    object_parameter_slices: Mapping[int, Sequence[slice]],
    object_dependencies: Mapping[int, Sequence[int]] | None = None,
    leakage_tolerance: float = 1e-8,
    activity_jacobian_tolerance: float = 1e-8,
    activity_residual_tolerance: float = 1e-5,
) -> dict[str, object]:
    """Audit slice coverage, block leakage, and numerical factor activity.

    The dense Jacobian is a reference oracle used only by the audit gate.  The
    production block assembler consumes the returned declared blocks and does
    not call this routine.
    """
    if residuals.ndim != 1 or jacobian.ndim != 2:
        raise ValueError("residuals/Jacobian must have shapes (R,) and (R,P)")
    if jacobian.shape[0] != residuals.numel():
        raise ValueError("Jacobian row count must match residuals")
    parameter_count = int(jacobian.shape[1])
    object_count = len(object_parameter_slices)
    coverage = torch.zeros(residuals.numel(), dtype=torch.long, device=residuals.device)
    active_factor_ids: list[str] = []
    active_collision_factor_ids: list[str] = []
    inactive_collision_factor_ids: list[str] = []
    leakage: list[dict[str, object]] = []
    per_factor: list[dict[str, object]] = []
    dependencies = object_dependencies or {}
    for binding in bindings:
        binding.validate(int(residuals.numel()), object_count)
        coverage[binding.residual_slice] += 1
        owners: set[int] = set()
        for object_index in binding.declared_object_indices:
            owners.add(int(object_index))
            owners.update(int(value) for value in dependencies.get(object_index, ()))
        allowed = torch.zeros(parameter_count, dtype=torch.bool, device=jacobian.device)
        for object_index in owners:
            for parameter_slice in object_parameter_slices[object_index]:
                allowed[parameter_slice] = True
        rows = jacobian[binding.residual_slice]
        declared_max = float(
            rows[:, allowed].detach().abs().amax().item()
            if bool(torch.any(allowed).item()) and rows.numel()
            else 0.0
        )
        leaked_max = float(
            rows[:, ~allowed].detach().abs().amax().item()
            if bool(torch.any(~allowed).item()) and rows.numel()
            else 0.0
        )
        residual_max = float(
            residuals[binding.residual_slice].detach().abs().amax().item()
        )
        active = bool(
            declared_max > activity_jacobian_tolerance
            or residual_max > activity_residual_tolerance
        )
        if active:
            active_factor_ids.append(binding.factor_id)
            if binding.collision_pair is not None:
                active_collision_factor_ids.append(binding.factor_id)
        elif binding.collision_pair is not None:
            inactive_collision_factor_ids.append(binding.factor_id)
        if leaked_max > leakage_tolerance:
            leakage.append({
                "factor_id": binding.factor_id,
                "channel": binding.channel,
                "max_abs_leakage": leaked_max,
            })
        per_factor.append({
            "factor_id": binding.factor_id,
            "channel": binding.channel,
            "declared_objects": sorted(owners),
            "residual_max_abs": residual_max,
            "declared_jacobian_max_abs": declared_max,
            "leaked_jacobian_max_abs": leaked_max,
            "active": active,
        })
    uncovered = torch.nonzero(coverage == 0, as_tuple=False).reshape(-1).tolist()
    duplicated = torch.nonzero(coverage > 1, as_tuple=False).reshape(-1).tolist()
    return {
        "passed": not leakage and not uncovered and not duplicated,
        "residual_count": int(residuals.numel()),
        "parameter_count": parameter_count,
        "bindings": len(bindings),
        "active_factor_ids": sorted(active_factor_ids),
        "active_collision_factor_ids": sorted(active_collision_factor_ids),
        "inactive_collision_factor_ids": sorted(inactive_collision_factor_ids),
        "leakage": leakage,
        "uncovered_residual_rows": uncovered,
        "duplicated_residual_rows": duplicated,
        "per_factor": per_factor,
    }


def stable_leaf_translation_objects(
    *,
    coordinates: SO3RelationCoordinateSystem,
    bindings: Sequence[ResidualSliceBinding],
    stable_active_factor_ids: Sequence[str],
) -> tuple[int, ...]:
    """Return safe leaf translations under the stable numerical factor graph."""
    active = set(stable_active_factor_ids)
    incidence = [
        binding.declared_object_indices
        for binding in bindings
        if binding.factor_id in active
    ]
    return eligible_leaf_translation_objects(coordinates, incidence)


def guarded_collision_trial(
    *,
    incumbent_parameters: torch.Tensor,
    candidate_parameters: torch.Tensor,
    collision_bindings: Sequence[ResidualSliceBinding],
    evaluate_collision_residuals: Callable[[torch.Tensor], torch.Tensor],
    parent_by_object: Mapping[int, int],
    primary_child_objects: Sequence[int] | None = None,
    activation_tolerance: float = 1e-5,
    worsening_tolerance: float = 1e-7,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Reject a Schur trial that activates or worsens any collision candidate."""
    before = evaluate_collision_residuals(incumbent_parameters).detach()
    after = evaluate_collision_residuals(candidate_parameters).detach()
    if before.shape != after.shape or before.ndim != 1:
        raise ValueError("collision evaluator must return a stable vector")
    if len(collision_bindings) != before.numel():
        raise ValueError("collision bindings must align with evaluator rows")
    failed: list[ResidualSliceBinding] = []
    for index, binding in enumerate(collision_bindings):
        newly_active = bool(
            before[index].abs() <= activation_tolerance
            and after[index].abs() > activation_tolerance
        )
        worsened = bool(
            after[index].abs() > before[index].abs() + worsening_tolerance
            and after[index].abs() > activation_tolerance
        )
        if newly_active or worsened:
            failed.append(binding)
    if not failed:
        return candidate_parameters, {
            "accepted": True,
            "collision_candidates_checked": len(collision_bindings),
            "failed_factor_ids": [],
            "released_object_indices": [],
            "reason": "collision_guard_passed",
        }
    released: set[int] = set()
    released_children: set[int] = set()
    external_separators: set[int] = set()
    parent_separators: set[int] = set()
    primary = (
        {int(value) for value in primary_child_objects}
        if primary_child_objects is not None
        else None
    )
    for binding in failed:
        pair = binding.collision_pair or tuple(binding.declared_object_indices[:2])
        pair_set = {int(value) for value in pair}
        children = pair_set & primary if primary is not None else pair_set
        if not children:
            children = pair_set
        external = pair_set - children
        released_children.update(children)
        external_separators.update(external)
        for object_index in children:
            parent = int(parent_by_object.get(object_index, -1))
            if parent >= 0:
                parent_separators.add(parent)
    released.update(released_children)
    released.update(external_separators)
    released.update(parent_separators)
    return incumbent_parameters, {
        "accepted": False,
        "collision_candidates_checked": len(collision_bindings),
        "failed_factor_ids": sorted(binding.factor_id for binding in failed),
        "released_object_indices": sorted(released),
        "released_child_indices": sorted(released_children),
        "external_separator_indices": sorted(external_separators),
        "parent_separator_indices": sorted(parent_separators),
        "reason": "collision_activated_or_worsened",
    }


def rollback_object_parameter_blocks(
    *,
    incumbent_parameters: torch.Tensor,
    candidate_parameters: torch.Tensor,
    coordinates: SO3RelationCoordinateSystem,
    object_indices: Sequence[int],
) -> tuple[torch.Tensor, dict[str, object]]:
    """Restore complete SO(3)+translation blocks for a failed witness scope."""
    if incumbent_parameters.shape != candidate_parameters.shape:
        raise ValueError("incumbent and candidate parameters must align")
    if incumbent_parameters.ndim != 1:
        raise ValueError("parameter vectors must be one-dimensional")
    restored = candidate_parameters.clone()
    unique = sorted({int(value) for value in object_indices})
    for object_index in unique:
        if not 0 <= object_index < len(coordinates.blocks):
            raise IndexError(f"object index out of range: {object_index}")
        block = coordinates.blocks[object_index]
        restored[block.parameter_slice] = incumbent_parameters[
            block.parameter_slice
        ]
    return restored, {
        "restored_object_indices": unique,
        "restored_parameter_count": sum(
            coordinates.blocks[index].parameter_stop
            - coordinates.blocks[index].parameter_start
            for index in unique
        ),
        "rotation_parameters_restored": sum(
            coordinates.blocks[index].rotation_stop
            - coordinates.blocks[index].rotation_start
            for index in unique
        ),
    }


def variable_slices(
    coordinates: SO3RelationCoordinateSystem,
) -> dict[str, slice]:
    result: dict[str, slice] = {}
    for block in coordinates.blocks:
        result[f"object:{block.object_index}:rotation"] = block.rotation_slice
        result[f"object:{block.object_index}:translation"] = block.translation_slice
    return result


def assemble_normal_system(
    *,
    parameter_count: int,
    slices: Mapping[str, slice],
    factors: Sequence[LinearizedFactor],
    damping: float,
    dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    if damping < 0:
        raise ValueError("damping must be non-negative")
    normal = torch.zeros((parameter_count, parameter_count), dtype=dtype, device=device)
    gradient = torch.zeros(parameter_count, dtype=dtype, device=device)
    block_products = 0
    residual_count = 0
    for factor in factors:
        if factor.residual.ndim != 1:
            raise ValueError("factor residuals must be one-dimensional")
        residual_count += int(factor.residual.numel())
        items = list(factor.jacobians.items())
        for name, jacobian in items:
            if name not in slices:
                raise KeyError(f"unknown variable block {name!r}")
            target = slices[name]
            width = target.stop - target.start
            if jacobian.shape != (factor.residual.numel(), width):
                raise ValueError(f"jacobian shape mismatch for {name!r}")
            gradient[target] += jacobian.transpose(0, 1) @ factor.residual
        for first_name, first_jacobian in items:
            first_slice = slices[first_name]
            for second_name, second_jacobian in items:
                second_slice = slices[second_name]
                normal[first_slice, second_slice] += (
                    first_jacobian.transpose(0, 1) @ second_jacobian
                )
                block_products += 1
    if damping:
        diagonal = torch.diagonal(normal).abs().clamp_min(1.0)
        normal += float(damping) * torch.diag(diagonal)
    return normal, gradient, {
        "factors": len(factors),
        "residuals": residual_count,
        "block_products": block_products,
    }


def restrict_normal_system_to_parameter_mask(
    normal_matrix: torch.Tensor,
    gradient: torch.Tensor,
    active_parameter_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Freeze parameters outside a responsibility subgraph exactly.

    Frozen rows and columns become an identity block with zero right-hand
    side, so neither the direct solve nor Schur back-substitution can move an
    undeclared root variable.
    """
    if normal_matrix.ndim != 2 or normal_matrix.shape[0] != normal_matrix.shape[1]:
        raise ValueError("normal_matrix must be square")
    count = int(normal_matrix.shape[0])
    if gradient.shape != (count,):
        raise ValueError("gradient shape does not match normal_matrix")
    if active_parameter_mask.shape != (count,):
        raise ValueError("active_parameter_mask has the wrong shape")
    if active_parameter_mask.dtype != torch.bool:
        raise TypeError("active_parameter_mask must be boolean")
    restricted_normal = normal_matrix.clone()
    restricted_gradient = gradient.clone()
    frozen = torch.nonzero(~active_parameter_mask, as_tuple=False).reshape(-1)
    if frozen.numel():
        restricted_normal[frozen, :] = 0.0
        restricted_normal[:, frozen] = 0.0
        restricted_normal[frozen, frozen] = 1.0
        restricted_gradient[frozen] = 0.0
    return restricted_normal, restricted_gradient, {
        "active_parameters": int(active_parameter_mask.sum().item()),
        "frozen_parameters": int(frozen.numel()),
    }


def certify_projected_stationarity(
    gradient: torch.Tensor,
    step: torch.Tensor,
    active_parameter_mask: torch.Tensor,
    *,
    tolerance: float = 1e-9,
    ulp_multiplier: float = 16.0,
) -> dict[str, float | bool]:
    """Certify that the responsibility subproblem has no first-order step."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if ulp_multiplier <= 0:
        raise ValueError("ulp_multiplier must be positive")
    if gradient.ndim != 1 or step.shape != gradient.shape:
        raise ValueError("gradient and step must be matching vectors")
    if active_parameter_mask.shape != gradient.shape:
        raise ValueError("active_parameter_mask has the wrong shape")
    if active_parameter_mask.dtype != torch.bool:
        raise TypeError("active_parameter_mask must be boolean")
    active_gradient = gradient[active_parameter_mask]
    active_step = step[active_parameter_mask]
    gradient_inf = float(
        active_gradient.abs().amax().detach().item()
        if active_gradient.numel()
        else 0.0
    )
    step_inf = float(
        active_step.abs().amax().detach().item()
        if active_step.numel()
        else 0.0
    )
    finite = math.isfinite(gradient_inf) and math.isfinite(step_inf)
    dtype_floor = float(torch.finfo(step.dtype).eps) * float(ulp_multiplier)
    effective_tolerance = max(float(tolerance), dtype_floor)
    return {
        "certified": bool(
            active_gradient.numel()
            and finite
            and gradient_inf <= effective_tolerance
            and step_inf <= effective_tolerance
        ),
        "projected_gradient_inf": gradient_inf,
        "step_inf": step_inf,
        "requested_tolerance": float(tolerance),
        "dtype_epsilon": float(torch.finfo(step.dtype).eps),
        "ulp_multiplier": float(ulp_multiplier),
        "effective_tolerance": effective_tolerance,
    }


def positive_spanning_poll_steps(
    active_parameter_mask: torch.Tensor,
    rotation_parameter_mask: torch.Tensor,
    *,
    rotation_radius: float,
    translation_radius: float,
) -> tuple[tuple[int, int, torch.Tensor], ...]:
    """Build a deterministic positive-spanning basis for a kink poll.

    The poll is deliberately restricted to the audited responsibility
    subspace.  Each active coordinate is tested in both signs, while SO(3)
    tangent and metric-translation coordinates receive unit-aware radii.
    This is the fail-closed fallback used when a smooth Schur model predicts
    descent but the exact objective increases in both directions.
    """
    if active_parameter_mask.ndim != 1:
        raise ValueError("active_parameter_mask must be one-dimensional")
    if rotation_parameter_mask.shape != active_parameter_mask.shape:
        raise ValueError("rotation_parameter_mask has the wrong shape")
    if active_parameter_mask.dtype != torch.bool:
        raise TypeError("active_parameter_mask must be boolean")
    if rotation_parameter_mask.dtype != torch.bool:
        raise TypeError("rotation_parameter_mask must be boolean")
    if rotation_radius <= 0.0 or translation_radius <= 0.0:
        raise ValueError("poll radii must be positive")
    records: list[tuple[int, int, torch.Tensor]] = []
    for index in torch.nonzero(
        active_parameter_mask, as_tuple=False
    ).reshape(-1).tolist():
        radius = (
            float(rotation_radius)
            if bool(rotation_parameter_mask[index].item())
            else float(translation_radius)
        )
        for sign in (-1, 1):
            step = torch.zeros(
                active_parameter_mask.shape,
                dtype=torch.get_default_dtype(),
                device=active_parameter_mask.device,
            )
            step[index] = float(sign) * radius
            records.append((int(index), int(sign), step))
    return tuple(records)


def eligible_leaf_translation_objects(
    coordinates: SO3RelationCoordinateSystem,
    factor_object_incidence: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """Eliminate a leaf translation only when its separator is its parent."""
    eligible: list[int] = []
    for block in coordinates.blocks:
        if not block.eliminable_translation or block.parent_index < 0:
            continue
        allowed = {block.object_index, block.parent_index}
        safe = True
        for incidence in factor_object_incidence:
            participants = {int(value) for value in incidence}
            if block.object_index in participants and not participants.issubset(allowed):
                safe = False
                break
        if safe:
            eligible.append(block.object_index)
    return tuple(eligible)


def solve_with_leaf_translation_schur(
    normal_matrix: torch.Tensor,
    right_hand_side: torch.Tensor,
    coordinates: SO3RelationCoordinateSystem,
    factor_object_incidence: Sequence[Sequence[int]],
    *,
    jitter: float = 1e-9,
    maximum_local_condition: float = 1e8,
    allowed_leaf_objects: Sequence[int] | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    eligible = eligible_leaf_translation_objects(
        coordinates, factor_object_incidence
    )
    if allowed_leaf_objects is not None:
        allowed = {int(value) for value in allowed_leaf_objects}
        eligible = tuple(value for value in eligible if value in allowed)
    eliminated: list[int] = []
    retained_for_condition: list[int] = []
    rejected: dict[int, str] = {}
    for object_index in eligible:
        block = coordinates.blocks[object_index]
        indices = list(range(block.translation_start, block.translation_stop))
        local = normal_matrix[indices][:, indices]
        if not bool(torch.isfinite(local).all().item()):
            rejected[object_index] = "non_finite_local_block"
            continue
        eigenvalues = torch.linalg.eigvalsh(local)
        minimum = float(eigenvalues.amin().detach().cpu().item())
        maximum = float(eigenvalues.amax().detach().cpu().item())
        if minimum <= 0.0:
            rejected[object_index] = "local_block_not_spd"
            continue
        condition = maximum / max(minimum, 1e-30)
        if condition > maximum_local_condition:
            rejected[object_index] = "local_block_ill_conditioned"
            continue
        eliminated.extend(indices)
        retained_for_condition.append(object_index)
    eliminated_tensor = torch.as_tensor(
        eliminated,
        dtype=torch.long,
        device=normal_matrix.device,
    )
    solution, diagnostics = block_schur_complement_solve(
        normal_matrix,
        right_hand_side,
        eliminated_tensor,
        jitter=jitter,
    )
    result: dict[str, object] = dict(diagnostics)
    result.update(
        {
            "eligible_leaf_objects": list(eligible),
            "eliminated_leaf_objects": retained_for_condition,
            "rejected_leaf_objects": rejected,
            "eliminated_translation_parameters": len(eliminated),
            "rotation_parameters_eliminated": 0,
        }
    )
    return solution, result
