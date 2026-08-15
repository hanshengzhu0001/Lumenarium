"""Pure factor-semantics and variable-ownership audit for SceneProof.

This module intentionally imports neither PyTorch nor Blender and never calls
the residual kernels or the certificate executor.  It reconciles their
immutable manifests, preventing either implementation from certifying itself.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from modules._sceneproof_program_ir import FactorSpec, ProgramBundle, RelationProgram


@dataclass(frozen=True)
class RuntimeFactorRow:
    channel: str
    relation_key: tuple[str, ...]
    variable_objects: tuple[str, ...]
    unit: str

    def identity(self) -> tuple[str, tuple[str, ...]]:
        return self.channel, self.relation_key


def _normalise_pairs(
    values: Iterable[Sequence[Any]], object_ids: Sequence[str]
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for first, second in values:
        first_id = object_ids[int(first)] if isinstance(first, int) else str(first)
        second_id = object_ids[int(second)] if isinstance(second, int) else str(second)
        result.append((first_id, second_id))
    return result


def _program_relation_key(program: RelationProgram) -> tuple[str, ...]:
    source = program.source_relation
    if program.kind in {"SUPPORT", "STACK", "PLANE_ATTACH"}:
        return str(source["child"]), str(source["parent"])
    if program.kind == "COLLISION_EXCLUSION":
        return tuple(sorted((str(source["first"]), str(source["second"]))))
    if program.kind in {"POINT_TOWARDS", "ALIGN", "DISTANCE"}:
        return str(source["source"]), str(source["target"])
    return (program.program_id,)


def build_runtime_factor_rows(
    *,
    ordered_ids: Sequence[str],
    support_pairs: Iterable[Sequence[Any]] = (),
    fixed_support_pairs: Iterable[Sequence[Any]] = (),
    containment_pairs: Iterable[Sequence[Any]] = (),
    plane_bindings: Iterable[Mapping[str, Any]] = (),
    collision_pairs: Iterable[Sequence[Any]] = (),
    semantic_specs: Mapping[str, Any] | None = None,
) -> tuple[RuntimeFactorRow, ...]:
    """Describe live differentiable rows without evaluating their kernels."""
    object_ids = tuple(str(value) for value in ordered_ids)
    known = set(object_ids)
    rows: list[RuntimeFactorRow] = []

    for child, parent in _normalise_pairs(support_pairs, object_ids):
        variables = (child, parent) if parent in known else (child,)
        rows.append(RuntimeFactorRow("support_contact_gap", (child, parent), variables, "m"))
    for child, parent in fixed_support_pairs:
        child_id, parent_id = str(child), str(parent)
        variables = (child_id, parent_id) if parent_id in known else (child_id,)
        rows.append(RuntimeFactorRow("support_contact_gap", (child_id, parent_id), variables, "m"))
    for child, parent in _normalise_pairs(containment_pairs, object_ids):
        variables = (child, parent) if parent in known else (child,)
        rows.append(RuntimeFactorRow("support_containment_error", (child, parent), variables, "m"))

    for binding in plane_bindings:
        child, parent = str(binding["child_id"]), str(binding["plane_id"])
        rows.append(RuntimeFactorRow("plane_gap", (child, parent), (child,), "m"))
        if bool(binding.get("orientation_required", True)):
            rows.append(RuntimeFactorRow("plane_orientation_alignment", (child, parent), (child,), "dimensionless"))

    for first, second in _normalise_pairs(collision_pairs, object_ids):
        relation_key = tuple(sorted((first, second)))
        rows.append(RuntimeFactorRow("collision_oriented_penetration", relation_key, (first, second), "m"))

    semantics = dict(semantic_specs or {})
    semantic_channels = (
        ("point_pairs", "semantic_point_towards", "dimensionless"),
        ("align_pairs", "semantic_align", "dimensionless"),
        ("distance_pairs", "semantic_distance_interval", "m2"),
    )
    for pair_key, channel, unit in semantic_channels:
        for source, target in _normalise_pairs(semantics.get(pair_key, ()), object_ids):
            # Legacy semantic kernels detach the target.  The immutable
            # ownership manifest records the actual differentiable owner.
            rows.append(RuntimeFactorRow(channel, (source, target), (source,), unit))
    return tuple(rows)


def audit_factor_semantics_and_ownership(
    bundle: ProgramBundle,
    runtime_rows: Sequence[RuntimeFactorRow],
) -> dict[str, Any]:
    """Bind solver factors to live rows and audit roles, units, and ownership."""
    bundle.validate()
    program_by_factor: dict[str, RelationProgram] = {}
    factor_by_identity: dict[tuple[str, tuple[str, ...]], tuple[RelationProgram, FactorSpec]] = {}
    duplicate_factor_bindings: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()
    certificate_units: Counter[str] = Counter()
    for program in bundle.programs:
        relation_key = _program_relation_key(program)
        for factor in program.factors:
            program_by_factor[factor.factor_id] = program
            role_counts[factor.role] += 1
            if factor.certificate_unit is not None:
                certificate_units[factor.certificate_unit] += 1
            if factor.role not in {"solver", "both"}:
                continue
            identity = (str(factor.solver_channel), relation_key)
            if identity in factor_by_identity:
                duplicate_factor_bindings.append({"channel": identity[0], "relation_key": list(identity[1])})
            factor_by_identity[identity] = (program, factor)

    runtime_counter = Counter(row.identity() for row in runtime_rows)
    duplicate_runtime_rows = [
        {"channel": channel, "relation_key": list(key), "count": count}
        for (channel, key), count in runtime_counter.items()
        if count != 1
    ]
    bound: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    matched_factor_ids: set[str] = set()
    for row in runtime_rows:
        match = factor_by_identity.get(row.identity())
        if match is None:
            mismatches.append({
                "kind": "runtime_row_without_ir_factor",
                "channel": row.channel,
                "relation_key": list(row.relation_key),
            })
            continue
        program, factor = match
        matched_factor_ids.add(factor.factor_id)
        unit_passed = factor.solver_unit == row.unit
        ownership_passed = tuple(factor.solver_variable_objects) == tuple(row.variable_objects)
        if not unit_passed or not ownership_passed:
            mismatches.append({
                "kind": "factor_contract_mismatch",
                "factor_id": factor.factor_id,
                "unit": {"ir": factor.solver_unit, "runtime": row.unit},
                "variable_objects": {"ir": list(factor.solver_variable_objects), "runtime": list(row.variable_objects)},
            })
        bound.append({
            "program_id": program.program_id,
            "factor_id": factor.factor_id,
            "channel": row.channel,
            "unit": row.unit,
            "relation_key": list(row.relation_key),
            "variable_objects": list(row.variable_objects),
            "unit_passed": unit_passed,
            "ownership_passed": ownership_passed,
        })

    abstained: list[dict[str, Any]] = []
    for _, factor in factor_by_identity.values():
        if factor.factor_id in matched_factor_ids:
            continue
        program = program_by_factor[factor.factor_id]
        if factor.solver_channel == "support_containment_error":
            abstained.append({
                "program_id": program.program_id,
                "factor_id": factor.factor_id,
                "reason": "live_geometric_containment_gate",
            })
        else:
            mismatches.append({
                "kind": "ir_solver_factor_without_runtime_row",
                "program_id": program.program_id,
                "factor_id": factor.factor_id,
                "channel": factor.solver_channel,
            })

    incidence: dict[str, set[str]] = defaultdict(set)
    for record in bound:
        for object_id in record["variable_objects"]:
            incidence[object_id].add(record["factor_id"])
    ownership_by_object: dict[str, Any] = {}
    parent_by_child: dict[str, str] = {}
    eliminable_claims: set[str] = set()
    for program in bundle.programs:
        for block in program.variable_blocks:
            ownership_by_object.setdefault(block.object_id, block)
            if block.parent_frame in bundle.object_ids:
                parent_by_child[block.object_id] = str(block.parent_frame)
            if block.eliminable_translation:
                eliminable_claims.add(block.object_id)

    safe_leaf_translations: list[str] = []
    rejected_leaf_translations: dict[str, str] = {}
    for child in sorted(eliminable_claims):
        parent = parent_by_child.get(child)
        if parent is None:
            rejected_leaf_translations[child] = "missing_parent_separator"
            continue
        unsafe = []
        for record in bound:
            variables = set(record["variable_objects"])
            if child in variables and not variables.issubset({child, parent}):
                unsafe.append(record["factor_id"])
        if unsafe:
            rejected_leaf_translations[child] = "cross_factor:" + ",".join(sorted(unsafe))
        else:
            safe_leaf_translations.append(child)

    passed = not (
        mismatches or duplicate_factor_bindings or duplicate_runtime_rows
    )
    return {
        "schema_version": "sceneproof_factor_binding_audit_v1",
        "passed": passed,
        "program_bundle_hash": bundle.content_hash(),
        "runtime_rows": len(runtime_rows),
        "bound_solver_factors": len(bound),
        "abstained_solver_factors": len(abstained),
        "factor_roles": dict(sorted(role_counts.items())),
        "solver_channels": dict(sorted(Counter(row.channel for row in runtime_rows).items())),
        "certificate_units": dict(sorted(certificate_units.items())),
        "bindings": bound,
        "abstentions": abstained,
        "mismatches": mismatches,
        "duplicate_factor_bindings": duplicate_factor_bindings,
        "duplicate_runtime_rows": duplicate_runtime_rows,
        "object_factor_degree": {key: len(value) for key, value in sorted(incidence.items())},
        "safe_leaf_translation_objects": safe_leaf_translations,
        "rejected_leaf_translation_objects": rejected_leaf_translations,
        "rotation_policy": "all_world_so3_retained_in_root_system",
        "scale_policy": "frozen",
        "solver_executor_intertwined": False,
    }

