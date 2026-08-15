"""Relation-conditioned coordinates and sparse linear algebra for SceneLM v5.

The legacy S4 optimizers expose an independent world-space yaw and XYZ
translation for every object.  Indoor layouts are substantially lower
dimensional: an object supported by another object moves in the tangent plane
of that support and inherits its parent's rigid motion; an object attached to
an architectural plane moves only in that plane.  This module compiles those
relations into a differentiable coordinate chart without depending on
Blender.

The chart is deliberately conservative.  It changes neither asset identity,
scale, nor the warm-start pose.  Unsupported or ambiguous relations fall back
to the legacy four-dimensional object block instead of guessing a constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import torch


def _normalise(vector: torch.Tensor, *, epsilon: float = 1e-12) -> torch.Tensor:
    return vector / torch.clamp(torch.linalg.vector_norm(vector), min=epsilon)


def _plane_basis(normal: torch.Tensor) -> torch.Tensor:
    """Return a deterministic orthonormal 3x2 tangent basis."""
    normal = _normalise(normal)
    abs_normal = normal.abs()
    axis_index = int(torch.argmin(abs_normal).detach().cpu().item())
    reference = torch.zeros_like(normal)
    reference[axis_index] = 1.0
    first = _normalise(torch.linalg.cross(normal, reference))
    second = _normalise(torch.linalg.cross(normal, first))
    return torch.stack((first, second), dim=1)


def _world_xy_basis(reference: torch.Tensor) -> torch.Tensor:
    basis = reference.new_zeros((3, 2))
    basis[0, 0] = 1.0
    basis[1, 1] = 1.0
    return basis


def _rotation_z(angle: torch.Tensor) -> torch.Tensor:
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    zero = torch.zeros_like(angle)
    one = torch.ones_like(angle)
    return torch.stack(
        (
            torch.stack((cosine, -sine, zero)),
            torch.stack((sine, cosine, zero)),
            torch.stack((zero, zero, one)),
        )
    )


def _topological_order(parent_indices: Sequence[int]) -> tuple[int, ...]:
    count = len(parent_indices)
    state = [0] * count
    order: list[int] = []

    def visit(index: int) -> None:
        if state[index] == 2:
            return
        if state[index] == 1:
            raise ValueError("support graph contains a directed cycle")
        state[index] = 1
        parent = int(parent_indices[index])
        if parent >= 0:
            if parent >= count:
                raise ValueError("support parent index is out of range")
            if parent == index:
                raise ValueError("an object cannot support itself")
            visit(parent)
        state[index] = 2
        order.append(index)

    for object_index in range(count):
        visit(object_index)
    return tuple(order)


@dataclass(frozen=True)
class RelationBlock:
    object_index: int
    parameter_start: int
    parameter_stop: int
    translation_dofs: int
    mode: str
    parent_index: int
    depth: int

    @property
    def parameter_slice(self) -> slice:
        return slice(self.parameter_start, self.parameter_stop)


@dataclass
class RelationCoordinateSystem:
    """Compiled differentiable chart for one frozen scene graph."""

    base_matrices: torch.Tensor
    parent_indices: torch.Tensor
    translation_bases: tuple[torch.Tensor, ...]
    blocks: tuple[RelationBlock, ...]
    topological_order: tuple[int, ...]
    optimise_yaw: bool
    relaxed_object_indices: tuple[int, ...] = ()

    @property
    def object_count(self) -> int:
        return int(self.base_matrices.shape[0])

    @property
    def parameter_count(self) -> int:
        return self.blocks[-1].parameter_stop if self.blocks else 0

    @property
    def legacy_parameter_count(self) -> int:
        return self.object_count * (4 if self.optimise_yaw else 3)

    @property
    def constrained_object_count(self) -> int:
        return sum(block.mode != "free" for block in self.blocks)

    @property
    def support_edge_count(self) -> int:
        return sum(block.parent_index >= 0 for block in self.blocks)

    @property
    def leaf_object_indices(self) -> tuple[int, ...]:
        parents = {
            block.parent_index for block in self.blocks if block.parent_index >= 0
        }
        return tuple(
            block.object_index
            for block in self.blocks
            if block.object_index not in parents and block.parent_index >= 0
        )

    def zero_parameters(self, *, requires_grad: bool = False) -> torch.Tensor:
        return torch.zeros(
            self.parameter_count,
            dtype=self.base_matrices.dtype,
            device=self.base_matrices.device,
            requires_grad=requires_grad,
        )

    def parameter_mask_from_objects(self, active_objects: torch.Tensor) -> torch.Tensor:
        if active_objects.shape != (self.object_count,):
            raise ValueError("active_objects must have shape (N,)")
        result = torch.zeros(
            self.parameter_count,
            dtype=torch.bool,
            device=self.base_matrices.device,
        )
        for block in self.blocks:
            result[block.parameter_slice] = active_objects[block.object_index]
        return result

    def object_step_norms(self, parameter_step: torch.Tensor) -> torch.Tensor:
        if parameter_step.shape != (self.parameter_count,):
            raise ValueError("parameter_step has the wrong shape")
        norms = self.base_matrices.new_zeros((self.object_count,))
        for block in self.blocks:
            values = parameter_step[block.parameter_slice]
            if values.numel():
                norms[block.object_index] = torch.linalg.vector_norm(values)
        return norms

    def cap_step(
        self,
        parameter_step: torch.Tensor,
        *,
        max_translation: float,
        max_yaw_radians: float,
        active_objects: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if parameter_step.shape != (self.parameter_count,):
            raise ValueError("parameter_step has the wrong shape")
        capped: list[torch.Tensor] = []
        for block in self.blocks:
            values = parameter_step[block.parameter_slice]
            cursor = 0
            scales: list[torch.Tensor] = []
            if self.optimise_yaw:
                yaw = values[0]
                scales.append(
                    yaw.new_tensor(max_yaw_radians)
                    / torch.clamp(yaw.abs(), min=max_yaw_radians)
                )
                cursor = 1
            translation = values[cursor:]
            translation_norm = torch.linalg.vector_norm(translation)
            scales.append(
                translation_norm.new_tensor(max_translation)
                / torch.clamp(translation_norm, min=max_translation)
            )
            scale = torch.stack(scales).amin()
            if active_objects is not None:
                scale = scale * active_objects[block.object_index].to(scale.dtype)
            capped.append(values * scale)
        return torch.cat(capped) if capped else parameter_step

    def decode(self, parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode minimal coordinates into world yaw deltas and translations."""
        if parameters.shape != (self.parameter_count,):
            raise ValueError(
                f"expected {self.parameter_count} relation parameters; "
                f"got {tuple(parameters.shape)}"
            )
        local_yaw: list[torch.Tensor] = []
        local_translation: list[torch.Tensor] = []
        for block, basis in zip(self.blocks, self.translation_bases):
            values = parameters[block.parameter_slice]
            cursor = 0
            if self.optimise_yaw:
                local_yaw.append(values[0])
                cursor = 1
            else:
                local_yaw.append(parameters.new_zeros(()))
            local_translation.append(basis @ values[cursor:])

        world_yaw: list[torch.Tensor | None] = [None] * self.object_count
        world_translation: list[torch.Tensor | None] = [None] * self.object_count
        base_centres = self.base_matrices[:, :3, 3]
        for index in self.topological_order:
            parent = int(self.parent_indices[index].detach().cpu().item())
            if parent < 0:
                world_yaw[index] = local_yaw[index]
                world_translation[index] = (
                    base_centres[index] + local_translation[index]
                )
                continue

            parent_yaw = world_yaw[parent]
            parent_translation = world_translation[parent]
            if parent_yaw is None or parent_translation is None:
                raise RuntimeError("support graph was not decoded topologically")
            rotation = _rotation_z(parent_yaw)
            base_offset = base_centres[index] - base_centres[parent]
            inherited_centre = (
                parent_translation + rotation @ base_offset
            )
            # The child tangent coordinates live in its parent's rotating
            # support frame, so a parent yaw carries both the child centre and
            # its local correction.
            world_translation[index] = (
                inherited_centre + rotation @ local_translation[index]
            )
            world_yaw[index] = parent_yaw + local_yaw[index]

        return (
            torch.stack([value for value in world_yaw if value is not None]),
            torch.stack(
                [value for value in world_translation if value is not None]
            ),
        )

    def pose_matrices(self, parameters: torch.Tensor) -> torch.Tensor:
        yaw_delta, translation = self.decode(parameters)
        cosine = torch.cos(yaw_delta)
        sine = torch.sin(yaw_delta)
        zeros = torch.zeros_like(yaw_delta)
        ones = torch.ones_like(yaw_delta)
        rotation = torch.stack(
            (
                torch.stack((cosine, -sine, zeros), dim=-1),
                torch.stack((sine, cosine, zeros), dim=-1),
                torch.stack((zeros, zeros, ones), dim=-1),
            ),
            dim=-2,
        )
        linear = rotation @ self.base_matrices[:, :3, :3]
        upper = torch.cat((linear, translation.unsqueeze(-1)), dim=-1)
        return torch.cat((upper, self.base_matrices[:, 3:4, :]), dim=-2)

    def metadata(self) -> dict[str, Any]:
        mode_counts: dict[str, int] = {}
        for block in self.blocks:
            mode_counts[block.mode] = mode_counts.get(block.mode, 0) + 1
        return {
            "schema_version": "scenelm_relation_coordinates_v1",
            "objects": self.object_count,
            "parameters": self.parameter_count,
            "legacy_parameters": self.legacy_parameter_count,
            "parameter_reduction": (
                1.0
                - self.parameter_count / max(self.legacy_parameter_count, 1)
            ),
            "constrained_objects": self.constrained_object_count,
            "relaxed_objects": len(self.relaxed_object_indices),
            "relaxed_object_indices": list(self.relaxed_object_indices),
            "support_edges": self.support_edge_count,
            "schur_leaf_blocks": len(self.leaf_object_indices),
            "parent_indices": self.parent_indices.detach().cpu().tolist(),
            "topological_order": list(self.topological_order),
            "mode_counts": mode_counts,
        }


def compile_relation_coordinates(
    base_matrices: torch.Tensor,
    support_pairs: torch.Tensor,
    fixed_support_indices: torch.Tensor,
    plane_object_indices: torch.Tensor,
    plane_normals: torch.Tensor,
    *,
    optimise_yaw: bool = True,
    free_object_indices: Iterable[int] = (),
    warm_start_anchored_plane_translation: bool = False,
) -> RelationCoordinateSystem:
    """Compile support and architectural relations into minimal blocks."""
    if base_matrices.ndim != 3 or base_matrices.shape[1:] != (4, 4):
        raise ValueError("base_matrices must have shape (N,4,4)")
    count = int(base_matrices.shape[0])
    relaxed = {int(value) for value in free_object_indices}
    if any(value < 0 or value >= count for value in relaxed):
        raise ValueError("free object index is out of range")
    parent = [-1] * count
    for child_value, parent_value in support_pairs.detach().cpu().tolist():
        child = int(child_value)
        parent_index = int(parent_value)
        if not 0 <= child < count or not 0 <= parent_index < count:
            raise ValueError("support pair index is out of range")
        if parent[child] not in (-1, parent_index):
            raise ValueError("an object has multiple optimised support parents")
        if child not in relaxed:
            parent[child] = parent_index
    order = _topological_order(parent)

    plane_by_object: dict[int, torch.Tensor] = {}
    if plane_object_indices.shape[0] != plane_normals.shape[0]:
        raise ValueError("plane object indices and normals must align")
    for object_value, normal in zip(
        plane_object_indices.detach().cpu().tolist(), plane_normals
    ):
        object_index = int(object_value)
        if not 0 <= object_index < count:
            raise ValueError("plane object index is out of range")
        plane_by_object.setdefault(object_index, normal)
    fixed = {
        int(value) for value in fixed_support_indices.detach().cpu().tolist()
    }

    depth = [0] * count
    for index in order:
        if parent[index] >= 0:
            depth[index] = depth[parent[index]] + 1

    blocks: list[RelationBlock] = []
    bases: list[torch.Tensor] = []
    cursor = 0
    for index in range(count):
        if index in relaxed:
            basis = torch.eye(
                3,
                dtype=base_matrices.dtype,
                device=base_matrices.device,
            )
            mode = "free"
        elif index in plane_by_object:
            if warm_start_anchored_plane_translation:
                basis = _normalise(plane_by_object[index])[:, None]
                mode = "plane_n_anchored"
            else:
                basis = _plane_basis(plane_by_object[index])
                mode = "plane"
        elif parent[index] >= 0:
            basis = _world_xy_basis(base_matrices)
            mode = "support"
        elif index in fixed:
            basis = _world_xy_basis(base_matrices)
            mode = "fixed_support"
        else:
            basis = torch.eye(
                3,
                dtype=base_matrices.dtype,
                device=base_matrices.device,
            )
            mode = "free"
        translation_dofs = int(basis.shape[1])
        block_size = translation_dofs + int(optimise_yaw)
        blocks.append(
            RelationBlock(
                object_index=index,
                parameter_start=cursor,
                parameter_stop=cursor + block_size,
                translation_dofs=translation_dofs,
                mode=mode,
                parent_index=parent[index],
                depth=depth[index],
            )
        )
        bases.append(basis)
        cursor += block_size

    return RelationCoordinateSystem(
        base_matrices=base_matrices,
        parent_indices=torch.as_tensor(
            parent, dtype=torch.long, device=base_matrices.device
        ),
        translation_bases=tuple(bases),
        blocks=tuple(blocks),
        topological_order=order,
        optimise_yaw=optimise_yaw,
        relaxed_object_indices=tuple(sorted(relaxed)),
    )


def block_schur_complement_solve(
    normal_matrix: torch.Tensor,
    right_hand_side: torch.Tensor,
    eliminated_indices: torch.Tensor,
    *,
    jitter: float = 1e-9,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Solve a damped normal system by exact Schur elimination.

    This routine is the correctness reference for the production matrix-free
    tree preconditioner.  It is also used for small root systems where an
    explicit block matrix is cheaper than PCG.
    """
    if normal_matrix.ndim != 2 or normal_matrix.shape[0] != normal_matrix.shape[1]:
        raise ValueError("normal_matrix must be square")
    count = int(normal_matrix.shape[0])
    if right_hand_side.shape != (count,):
        raise ValueError("right_hand_side has the wrong shape")
    eliminated_indices = eliminated_indices.to(
        dtype=torch.long, device=normal_matrix.device
    )
    if eliminated_indices.ndim != 1:
        raise ValueError("eliminated_indices must be one-dimensional")
    if eliminated_indices.numel() and (
        torch.any(eliminated_indices < 0)
        or torch.any(eliminated_indices >= count)
    ):
        raise ValueError("eliminated index is out of range")
    keep_mask = torch.ones(count, dtype=torch.bool, device=normal_matrix.device)
    keep_mask[eliminated_indices] = False
    retained_indices = torch.nonzero(keep_mask, as_tuple=False).reshape(-1)
    identity = torch.eye(
        count, dtype=normal_matrix.dtype, device=normal_matrix.device
    )
    matrix = normal_matrix + float(jitter) * identity

    if eliminated_indices.numel() == 0 or retained_indices.numel() == 0:
        solution = torch.linalg.solve(matrix, right_hand_side)
        return solution, {
            "eliminated": float(eliminated_indices.numel()),
            "retained": float(retained_indices.numel()),
        }

    h_rr = matrix[retained_indices][:, retained_indices]
    h_re = matrix[retained_indices][:, eliminated_indices]
    h_er = matrix[eliminated_indices][:, retained_indices]
    h_ee = matrix[eliminated_indices][:, eliminated_indices]
    b_r = right_hand_side[retained_indices]
    b_e = right_hand_side[eliminated_indices]
    solved_er = torch.linalg.solve(h_ee, h_er)
    solved_e_rhs = torch.linalg.solve(h_ee, b_e)
    schur = h_rr - h_re @ solved_er
    reduced_rhs = b_r - h_re @ solved_e_rhs
    retained_solution = torch.linalg.solve(schur, reduced_rhs)
    eliminated_solution = torch.linalg.solve(
        h_ee, b_e - h_er @ retained_solution
    )
    solution = torch.zeros_like(right_hand_side)
    solution[retained_indices] = retained_solution
    solution[eliminated_indices] = eliminated_solution
    return solution, {
        "eliminated": float(eliminated_indices.numel()),
        "retained": float(retained_indices.numel()),
        "schur_dimension": float(retained_indices.numel()),
    }


def parameter_indices_for_objects(
    coordinate_system: RelationCoordinateSystem,
    object_indices: Iterable[int],
) -> torch.Tensor:
    values: list[int] = []
    for object_index in object_indices:
        block = coordinate_system.blocks[int(object_index)]
        values.extend(range(block.parameter_start, block.parameter_stop))
    return torch.as_tensor(
        values,
        dtype=torch.long,
        device=coordinate_system.base_matrices.device,
    )


# ---------------------------------------------------------------------------
# SceneProof v2 full-SO(3) relation chart.
#
# The v1 chart above remains available for frozen ablations.  The v2 chart is
# intentionally separate: support edges carry child centres in the parent's
# moving translation frame, while every non-rigid object keeps an independent
# world SO(3) tangent block.


def _skew(vector: torch.Tensor) -> torch.Tensor:
    if vector.shape[-1] != 3:
        raise ValueError("SO(3) tangent vectors must end in dimension 3")
    x_value, y_value, z_value = vector.unbind(dim=-1)
    zero = torch.zeros_like(x_value)
    return torch.stack(
        (
            torch.stack((zero, -z_value, y_value), dim=-1),
            torch.stack((z_value, zero, -x_value), dim=-1),
            torch.stack((-y_value, x_value, zero), dim=-1),
        ),
        dim=-2,
    )


def so3_exp(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Differentiable exponential map with stable small-angle series."""
    if rotation_vector.shape[-1] != 3:
        raise ValueError("rotation_vector must have shape (..., 3)")
    theta_squared = rotation_vector.square().sum(dim=-1)
    theta = torch.sqrt(torch.clamp_min(theta_squared, 1e-24))
    safe_theta_squared = torch.clamp_min(theta_squared, 1e-24)
    sine_over_theta = torch.sin(theta) / theta
    one_minus_cosine_over_theta_squared = (
        1.0 - torch.cos(theta)
    ) / safe_theta_squared
    small = theta_squared < 1e-8
    a_value = torch.where(
        small,
        1.0 - theta_squared / 6.0 + theta_squared.square() / 120.0,
        sine_over_theta,
    )
    b_value = torch.where(
        small,
        0.5 - theta_squared / 24.0 + theta_squared.square() / 720.0,
        one_minus_cosine_over_theta_squared,
    )
    skew = _skew(rotation_vector)
    identity = torch.eye(
        3,
        dtype=rotation_vector.dtype,
        device=rotation_vector.device,
    ).expand(rotation_vector.shape[:-1] + (3, 3))
    return (
        identity
        + a_value[..., None, None] * skew
        + b_value[..., None, None] * (skew @ skew)
    )


@dataclass(frozen=True)
class SO3RelationBlock:
    object_index: int
    rotation_start: int
    rotation_stop: int
    translation_start: int
    translation_stop: int
    mode: str
    parent_index: int
    depth: int
    eliminable_translation: bool

    @property
    def parameter_slice(self) -> slice:
        return slice(self.rotation_start, self.translation_stop)

    @property
    def parameter_start(self) -> int:
        """Compatibility alias for generic block-iteration utilities."""
        return self.rotation_start

    @property
    def parameter_stop(self) -> int:
        """Compatibility alias for generic block-iteration utilities."""
        return self.translation_stop

    @property
    def rotation_slice(self) -> slice:
        return slice(self.rotation_start, self.rotation_stop)

    @property
    def translation_slice(self) -> slice:
        return slice(self.translation_start, self.translation_stop)

    @property
    def translation_dofs(self) -> int:
        return self.translation_stop - self.translation_start


@dataclass
class SO3RelationCoordinateSystem:
    """Full-SO(3), relation-conditioned SceneProof coordinate chart."""

    base_matrices: torch.Tensor
    parent_indices: torch.Tensor
    translation_bases: tuple[torch.Tensor, ...]
    blocks: tuple[SO3RelationBlock, ...]
    topological_order: tuple[int, ...]
    relaxed_object_indices: tuple[int, ...] = ()
    plane_translation_policy: str = "tangent_free"

    @property
    def object_count(self) -> int:
        return int(self.base_matrices.shape[0])

    @property
    def parameter_count(self) -> int:
        return self.blocks[-1].translation_stop if self.blocks else 0

    @property
    def rotation_parameter_count(self) -> int:
        return 3 * self.object_count

    @property
    def leaf_object_indices(self) -> tuple[int, ...]:
        return tuple(
            block.object_index
            for block in self.blocks
            if block.eliminable_translation
        )

    def zero_parameters(self, *, requires_grad: bool = False) -> torch.Tensor:
        return torch.zeros(
            self.parameter_count,
            dtype=self.base_matrices.dtype,
            device=self.base_matrices.device,
            requires_grad=requires_grad,
        )

    def parameter_mask_from_objects(self, active_objects: torch.Tensor) -> torch.Tensor:
        if active_objects.shape != (self.object_count,):
            raise ValueError("active_objects must have shape (N,)")
        mask = torch.zeros(
            self.parameter_count,
            dtype=torch.bool,
            device=self.base_matrices.device,
        )
        for block in self.blocks:
            mask[block.parameter_slice] = active_objects[block.object_index]
        return mask

    def cap_step(
        self,
        step: torch.Tensor,
        *,
        max_translation: float,
        max_rotation_radians: float,
        active_objects: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if step.shape != (self.parameter_count,):
            raise ValueError("step has the wrong shape")
        result = step.clone()
        for block in self.blocks:
            rotation = result[block.rotation_slice]
            translation = result[block.translation_slice]
            rotation_norm = torch.linalg.vector_norm(rotation)
            translation_norm = torch.linalg.vector_norm(translation)
            rotation_scale = rotation.new_tensor(max_rotation_radians) / torch.clamp(
                rotation_norm, min=max_rotation_radians
            )
            translation_scale = translation.new_tensor(max_translation) / torch.clamp(
                translation_norm, min=max_translation
            )
            scale = torch.minimum(rotation_scale, translation_scale)
            if active_objects is not None:
                scale = scale * active_objects[block.object_index].to(scale.dtype)
            result[block.parameter_slice] *= scale
        return result

    def cap_step_globally(
        self,
        step: torch.Tensor,
        *,
        max_translation: float,
        max_rotation_radians: float,
        active_objects: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Apply one trust scale to every block, preserving descent direction."""
        if step.shape != (self.parameter_count,):
            raise ValueError("step has the wrong shape")
        if max_translation <= 0 or max_rotation_radians <= 0:
            raise ValueError("trust limits must be positive")
        scale = step.new_tensor(1.0)
        for block in self.blocks:
            if (
                active_objects is not None
                and not bool(active_objects[block.object_index].item())
            ):
                continue
            rotation_norm = torch.linalg.vector_norm(step[block.rotation_slice])
            translation_norm = torch.linalg.vector_norm(
                step[block.translation_slice]
            )
            rotation_scale = step.new_tensor(max_rotation_radians) / torch.clamp(
                rotation_norm, min=max_rotation_radians
            )
            translation_scale = step.new_tensor(max_translation) / torch.clamp(
                translation_norm, min=max_translation
            )
            scale = torch.minimum(
                scale, torch.minimum(rotation_scale, translation_scale)
            )
        result = step * scale
        if active_objects is not None:
            for block in self.blocks:
                if not bool(active_objects[block.object_index].item()):
                    result[block.parameter_slice] = 0.0
        return result, float(scale.detach().item())

    def decode(self, parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if parameters.shape != (self.parameter_count,):
            raise ValueError("parameters have the wrong shape")
        rotation_vectors = torch.stack(
            [parameters[block.rotation_slice] for block in self.blocks]
        )
        rotation_deltas = so3_exp(rotation_vectors)
        local_translations = [
            basis @ parameters[block.translation_slice]
            for block, basis in zip(self.blocks, self.translation_bases)
        ]
        base_centres = self.base_matrices[:, :3, 3]
        world_centres: list[torch.Tensor | None] = [None] * self.object_count
        for index in self.topological_order:
            parent = int(self.parent_indices[index].detach().cpu().item())
            if parent < 0:
                world_centres[index] = base_centres[index] + local_translations[index]
                continue
            parent_centre = world_centres[parent]
            if parent_centre is None:
                raise RuntimeError("support graph was not decoded topologically")
            base_offset = base_centres[index] - base_centres[parent]
            # Parent motion transports the child's centre and its tangent
            # correction.  It does not alter the child's independent SO(3)
            # orientation block.
            world_centres[index] = (
                parent_centre
                + rotation_deltas[parent] @ base_offset
                + rotation_deltas[parent] @ local_translations[index]
            )
        return rotation_deltas, torch.stack(
            [value for value in world_centres if value is not None]
        )

    def pose_matrices(self, parameters: torch.Tensor) -> torch.Tensor:
        rotation_deltas, translation = self.decode(parameters)
        linear = rotation_deltas @ self.base_matrices[:, :3, :3]
        upper = torch.cat((linear, translation.unsqueeze(-1)), dim=-1)
        return torch.cat((upper, self.base_matrices[:, 3:4, :]), dim=-2)

    def leaf_translation_parameter_indices(self) -> torch.Tensor:
        values: list[int] = []
        for block in self.blocks:
            if block.eliminable_translation:
                values.extend(range(block.translation_start, block.translation_stop))
        return torch.as_tensor(
            values,
            dtype=torch.long,
            device=self.base_matrices.device,
        )

    def metadata(self) -> dict[str, Any]:
        mode_counts: dict[str, int] = {}
        for block in self.blocks:
            mode_counts[block.mode] = mode_counts.get(block.mode, 0) + 1
        return {
            "schema_version": "sceneproof_full_so3_relation_coordinates_v2",
            "objects": self.object_count,
            "parameters": self.parameter_count,
            "rotation_parameters": self.rotation_parameter_count,
            "rotation_mode": "independent_world_so3",
            "scale_mode": "frozen",
            "plane_translation_policy": self.plane_translation_policy,
            "parent_indices": self.parent_indices.detach().cpu().tolist(),
            "topological_order": list(self.topological_order),
            "leaf_translation_objects": list(self.leaf_object_indices),
            "leaf_translation_parameters": int(
                self.leaf_translation_parameter_indices().numel()
            ),
            "mode_counts": mode_counts,
        }


def compile_full_so3_relation_coordinates(
    base_matrices: torch.Tensor,
    support_pairs: torch.Tensor,
    fixed_support_indices: torch.Tensor,
    plane_object_indices: torch.Tensor,
    plane_normals: torch.Tensor,
    *,
    free_object_indices: Iterable[int] = (),
    support_normal_dof: bool = False,
    plane_normal_dof: bool = False,
    warm_start_anchored_plane_translation: bool = False,
) -> SO3RelationCoordinateSystem:
    """Compile a safe full-SO(3) chart without parent-orientation inheritance."""
    if base_matrices.ndim != 3 or base_matrices.shape[1:] != (4, 4):
        raise ValueError("base_matrices must have shape (N,4,4)")
    count = int(base_matrices.shape[0])
    relaxed = {int(value) for value in free_object_indices}
    if any(value < 0 or value >= count for value in relaxed):
        raise ValueError("free object index is out of range")
    parent = [-1] * count
    for child_value, parent_value in support_pairs.detach().cpu().tolist():
        child, parent_index = int(child_value), int(parent_value)
        if not 0 <= child < count or not 0 <= parent_index < count:
            raise ValueError("support pair index is out of range")
        if parent[child] not in (-1, parent_index):
            raise ValueError("an object has multiple support parents")
        if child not in relaxed:
            parent[child] = parent_index
    order = _topological_order(parent)
    children = {parent_index for parent_index in parent if parent_index >= 0}

    plane_by_object: dict[int, torch.Tensor] = {}
    if plane_object_indices.shape[0] != plane_normals.shape[0]:
        raise ValueError("plane object indices and normals must align")
    for object_value, normal in zip(
        plane_object_indices.detach().cpu().tolist(), plane_normals
    ):
        object_index = int(object_value)
        if not 0 <= object_index < count:
            raise ValueError("plane object index is out of range")
        plane_by_object.setdefault(object_index, normal)
    fixed = {int(value) for value in fixed_support_indices.detach().cpu().tolist()}
    depth = [0] * count
    for index in order:
        if parent[index] >= 0:
            depth[index] = depth[parent[index]] + 1

    blocks: list[SO3RelationBlock] = []
    bases: list[torch.Tensor] = []
    cursor = 0
    for index in range(count):
        if index in relaxed:
            basis = torch.eye(3, dtype=base_matrices.dtype, device=base_matrices.device)
            mode = "free"
        elif parent[index] >= 0:
            basis = _world_xy_basis(base_matrices)
            if support_normal_dof:
                basis = torch.eye(3, dtype=base_matrices.dtype, device=base_matrices.device)
            mode = "support_uvh" if support_normal_dof else "support_uv"
        elif index in plane_by_object:
            normal = _normalise(plane_by_object[index])[:, None]
            if warm_start_anchored_plane_translation:
                # The S4 warm start is the chart origin.  Plane contact supplies
                # evidence along the normal, but no factor observes arbitrary
                # motion inside the plane reliably.  Remove those two gauge
                # directions from the chart instead of trying to regularise
                # them with a category-specific loss.
                basis = normal
                mode = "plane_n_anchored"
            else:
                tangent = _plane_basis(plane_by_object[index])
                basis = (
                    torch.cat((tangent, normal), dim=1)
                    if plane_normal_dof
                    else tangent
                )
                mode = "plane_uvn" if plane_normal_dof else "plane_uv"
        elif index in fixed:
            basis = _world_xy_basis(base_matrices)
            mode = "fixed_support_uv"
        else:
            basis = torch.eye(3, dtype=base_matrices.dtype, device=base_matrices.device)
            mode = "free"
        rotation_start = cursor
        rotation_stop = cursor + 3
        translation_start = rotation_stop
        translation_stop = translation_start + int(basis.shape[1])
        blocks.append(
            SO3RelationBlock(
                object_index=index,
                rotation_start=rotation_start,
                rotation_stop=rotation_stop,
                translation_start=translation_start,
                translation_stop=translation_stop,
                mode=mode,
                parent_index=parent[index],
                depth=depth[index],
                eliminable_translation=(
                    parent[index] >= 0 and index not in children
                ),
            )
        )
        bases.append(basis)
        cursor = translation_stop
    return SO3RelationCoordinateSystem(
        base_matrices=base_matrices,
        parent_indices=torch.as_tensor(parent, dtype=torch.long, device=base_matrices.device),
        translation_bases=tuple(bases),
        blocks=tuple(blocks),
        topological_order=order,
        relaxed_object_indices=tuple(sorted(relaxed)),
        plane_translation_policy=(
            "warm_start_anchored_normal_only"
            if warm_start_anchored_plane_translation
            else "tangent_plus_optional_normal"
        ),
    )
