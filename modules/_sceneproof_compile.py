"""Compile legacy Lumenarium relations into the SceneProof program IR."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from modules._sceneproof_program_ir import (
    CertificatePolicy,
    FactorSpec,
    PartRef,
    ProbeSpec,
    ProgramBundle,
    RelationProgram,
    VariableBlockSpec,
    stable_json,
)


DEFAULT_THRESHOLDS = {
    "collision_fraction": 0.05,
    "contact_gap_m": 0.05,
    "containment_error_m": 0.05,
    "support_footprint_overlap_ratio": 0.90,
    "plane_gap_m": 0.05,
    "plane_orientation_deg": 15.0,
}


def _geometry_ref(object_id: str, obj_info: Mapping[str, Mapping[str, Any]]) -> str:
    info = obj_info.get(object_id, {})
    for key in ("asset_path", "asset_id", "asset", "model_path"):
        value = info.get(key)
        if isinstance(value, str) and value:
            return f"asset:{value}"
    return f"frozen_s3:{object_id}"


def _part(
    object_id: str,
    part_id: str,
    obj_info: Mapping[str, Mapping[str, Any]],
    *,
    provenance: str = "frozen_s3",
) -> PartRef:
    return PartRef(
        object_id=object_id,
        part_id=part_id,
        geometry_ref=_geometry_ref(object_id, obj_info),
        provenance=provenance,
    )


def _normalise_pairs(
    pairs: Iterable[Sequence[Any]], ordered_ids: Sequence[str]
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for child, parent in pairs:
        child_id = ordered_ids[int(child)] if isinstance(child, int) else str(child)
        parent_id = ordered_ids[int(parent)] if isinstance(parent, int) else str(parent)
        result.append((child_id, parent_id))
    return result


def _factor(
    program_id: str,
    suffix: str,
    kind: str,
    participants: tuple[PartRef, ...],
    *,
    operator: str,
    threshold: float,
    measurement: str,
    severity: str = "hard",
    role: str = "both",
    solver_channel: str | None = None,
    solver_unit: str | None = None,
    solver_variable_objects: tuple[str, ...] = (),
    certificate_unit: str | None = None,
) -> FactorSpec:
    return FactorSpec(
        factor_id=f"{program_id}:{suffix}",
        kind=kind,
        participants=participants,
        parameters={
            "measurement": measurement,
            "operator": operator,
            "threshold": float(threshold),
        },
        severity=severity,
        role=role,
        solver_channel=solver_channel,
        solver_unit=solver_unit,
        solver_variable_objects=(
            solver_variable_objects if role in {"solver", "both"} else ()
        ),
        certificate_measurement=(
            measurement if role in {"certificate", "both"} else None
        ),
        certificate_unit=(
            certificate_unit if role in {"certificate", "both"} else None
        ),
    )


def _program_policy() -> CertificatePolicy:
    return CertificatePolicy(thresholds=dict(DEFAULT_THRESHOLDS))


def compile_legacy_relation_programs(
    *,
    scene_id: str,
    obj_info: Mapping[str, Mapping[str, Any]],
    ordered_ids: Sequence[str],
    support_pairs: Iterable[Sequence[Any]] = (),
    fixed_support_pairs: Iterable[Sequence[Any]] = (),
    plane_bindings: Iterable[Mapping[str, Any]] = (),
    collision_pairs: Iterable[Sequence[Any]] = (),
    semantic_specs: Mapping[str, Any] | None = None,
    affordance_metadata: Mapping[str, Any] | None = None,
    support_topology_authoritative: bool = False,
) -> ProgramBundle:
    """Compile relations without guessing unavailable functional geometry.

    `support_pairs` and `collision_pairs` may contain object IDs or indices into
    `ordered_ids`.  `INSIDE` is discovered from legacy `supported`/`SpatialRel`
    fields and compiles only when explicit cavity and opening metadata exists.
    """
    object_ids = tuple(str(value) for value in ordered_ids)
    object_set = set(object_ids)
    support = _normalise_pairs(support_pairs, object_ids)
    fixed_support = [(str(child), str(parent)) for child, parent in fixed_support_pairs]
    collisions = _normalise_pairs(collision_pairs, object_ids)
    planes = [dict(value) for value in plane_bindings]
    semantics = dict(semantic_specs or {})
    affordances = dict(affordance_metadata or {})

    relation_records: list[dict[str, Any]] = []
    for child, parent in support:
        relation_records.append({"kind": "SUPPORT", "child": child, "parent": parent})
    for child, parent in fixed_support:
        relation_records.append({"kind": "SUPPORT", "child": child, "parent": parent})
    for binding in planes:
        relation_records.append(
            {
                "kind": "PLANE_ATTACH",
                "child": str(binding["child_id"]),
                "parent": str(binding["plane_id"]),
                "normal": list(binding.get("normal", ())),
                "orientation_required": bool(
                    binding.get("orientation_required", True)
                ),
            }
        )
    for first, second in collisions:
        relation_records.append(
            {"kind": "COLLISION_EXCLUSION", "first": first, "second": second}
        )

    explicit_support = {(child, parent) for child, parent in support + fixed_support}
    explicit_planes = {
        (str(binding["child_id"]), str(binding["plane_id"]))
        for binding in planes
    }
    for child in object_ids:
        info = obj_info.get(child, {})
        parent = info.get("supported")
        relation = info.get("SpatialRel", info.get("relation"))
        # Mirror live S4's provisional normalization.  A later Blender
        # subspace pass may refine `on` to `inside`, but missing relation text
        # must not silently erase a support edge from compiler accounting.
        if relation is None and isinstance(parent, str):
            relation = "on"
        relation_values = (
            {str(value).lower() for value in relation}
            if isinstance(relation, (list, tuple, set))
            else {str(relation).lower()}
        )
        if isinstance(parent, str) and "inside" in relation_values:
            relation_records.append(
                {"kind": "INSIDE", "child": child, "parent": parent}
            )
        elif (
            isinstance(parent, str)
            and "on" in relation_values
            and (child, parent) not in explicit_support
        ):
            # Wall/ceiling attachment is owned exclusively by PLANE_ATTACH.
            # Compiling the same edge as SUPPORT creates conflicting variable
            # ownership and double-counts the constraint.
            if (child, parent) in explicit_planes:
                continue
            relation_records.append({
                "kind": (
                    "UNBOUND_SUPPORT"
                    if support_topology_authoritative
                    else "SUPPORT"
                ),
                "child": child,
                "parent": parent,
                "reason": (
                    "not accepted by authoritative live support topology"
                    if support_topology_authoritative
                    else "legacy relation discovery"
                ),
            })

    semantic_kinds = (
        ("point_pairs", "point_offsets", "POINT_TOWARDS"),
        ("align_pairs", "align_offsets", "ALIGN"),
        ("distance_pairs", None, "DISTANCE"),
    )
    for pair_key, offset_key, kind in semantic_kinds:
        for index, pair in enumerate(semantics.get(pair_key, ())):
            source, target = _normalise_pairs((pair,), object_ids)[0]
            record = {"kind": kind, "source": source, "target": target}
            if offset_key is not None:
                values = semantics.get(offset_key, ())
                record["offset"] = float(values[index]) if index < len(values) else 0.0
            elif kind == "DISTANCE":
                minimums = semantics.get("distance_minimum", ())
                maximums = semantics.get("distance_maximum", ())
                record["minimum"] = float(minimums[index])
                record["maximum"] = float(maximums[index])
            relation_records.append(record)

    # Preserve semantic relations rejected by the warm-start consistency
    # adapter in the bundle accounting.  They must not silently disappear:
    # they are explicit REJECTED inputs, not compiled proof programs.
    for skipped in semantics.get("skipped", ()):
        relation_records.append(
            {
                "kind": "SEMANTIC_SKIPPED",
                **dict(skipped),
            }
        )

    # S1 fields and already-materialized S4 tensors can describe the same
    # relation.  Deduplicate before accounting so no relation silently
    # disappears and repeated serialization stays deterministic.
    unique_records: list[dict[str, Any]] = []
    seen_records: set[str] = set()
    for record in relation_records:
        key = stable_json(record)
        if key in seen_records:
            continue
        seen_records.add(key)
        unique_records.append(record)
    relation_records = unique_records

    parent_by_child: dict[str, str] = {}
    for record in relation_records:
        if record["kind"] != "SUPPORT":
            continue
        child, parent = record["child"], record["parent"]
        if child in parent_by_child and parent_by_child[child] != parent:
            raise ValueError(f"multiple support parents for {child!r}")
        if parent in object_set:
            parent_by_child[child] = parent
    plane_children = {
        record["child"]
        for record in relation_records
        if record["kind"] == "PLANE_ATTACH"
    }
    children_by_parent: dict[str, set[str]] = {}
    for child, parent in parent_by_child.items():
        children_by_parent.setdefault(parent, set()).add(child)

    def block_for(object_id: str) -> VariableBlockSpec:
        parent = parent_by_child.get(object_id)
        if parent is not None:
            chart = "support_uvh"
        elif object_id in plane_children:
            chart = "plane_uvn"
        else:
            chart = "world_xyz"
        return VariableBlockSpec(
            object_id=object_id,
            rotation_mode="world_so3",
            translation_chart=chart,
            parent_frame=parent,
            eliminable_translation=(
                parent is not None and object_id not in children_by_parent
            ),
        )

    blocks = {object_id: block_for(object_id) for object_id in object_ids}
    programs: list[RelationProgram] = []
    abstained: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_program_ids: set[str] = set()

    def add_program(program: RelationProgram) -> None:
        if program.program_id in seen_program_ids:
            return
        program.validate()
        seen_program_ids.add(program.program_id)
        programs.append(program)

    for record in relation_records:
        kind = record["kind"]
        endpoints = [
            value
            for key, value in record.items()
            if key in {"child", "parent", "first", "second", "source", "target"}
        ]
        unknown = [
            value
            for value in endpoints
            if value not in object_set and not str(value).startswith(
                ("floor_", "ground_", "wall_", "ceiling_", "architecture:")
            )
        ]
        if unknown:
            rejected.append({**record, "reason": f"unknown objects: {unknown}"})
            continue

        if kind in {"SUPPORT", "STACK"}:
            child, parent = record["child"], record["parent"]
            parent_ref_id = (
                parent if parent in object_set else f"architecture:{parent}"
            )
            child_part = _part(child, "support_patch_bottom", obj_info)
            parent_part = _part(
                parent_ref_id,
                "support_patch_top",
                obj_info,
                provenance="frozen_s3_or_architecture",
            )
            participants = (child_part, parent_part)
            program_id = f"support:{child}->{parent_ref_id}"
            solver_support_objects = (
                (child, parent) if parent in object_set else (child,)
            )
            program_blocks = [blocks[child]]
            if parent in blocks:
                program_blocks.append(blocks[parent])
            factors = (
                _factor(
                    program_id,
                    "contact",
                    "support_contact",
                    participants,
                    operator="abs_le",
                    threshold=DEFAULT_THRESHOLDS["contact_gap_m"],
                    measurement="contact_gap_m",
                    solver_channel="support_contact_gap",
                    solver_unit="m",
                    solver_variable_objects=solver_support_objects,
                    certificate_unit="m",
                ),
                _factor(
                    program_id,
                    "containment",
                    "support_containment",
                    participants,
                    operator="le",
                    threshold=DEFAULT_THRESHOLDS["containment_error_m"],
                    measurement="containment_error_m",
                    solver_channel="support_containment_error",
                    solver_unit="m",
                    solver_variable_objects=solver_support_objects,
                    certificate_unit="m",
                ),
                _factor(
                    program_id,
                    "footprint",
                    "support_footprint",
                    participants,
                    operator="ge",
                    threshold=DEFAULT_THRESHOLDS[
                        "support_footprint_overlap_ratio"
                    ],
                    measurement="support_footprint_overlap_ratio",
                    role="certificate",
                    certificate_unit="ratio",
                ),
            )
            probes = (
                ProbeSpec(
                    probe_id=f"{program_id}:perturbation",
                    kind="support_perturbation_survival",
                    parameters={"translation_ratio": 0.05, "max_m": 0.02, "degrees": 5.0},
                    required_evidence=("perturbation_survival",),
                ),
            )
            add_program(
                RelationProgram(
                    program_id=program_id,
                    kind=kind,
                    participants=participants,
                    variable_blocks=tuple(program_blocks),
                    factors=factors,
                    probes=probes,
                    certificate_policy=_program_policy(),
                    source_relation=record,
                )
            )
        elif kind == "PLANE_ATTACH":
            child, parent = record["child"], record["parent"]
            parent_ref_id = (
                parent if parent in object_set else f"architecture:{parent}"
            )
            participants = (
                _part(child, "attachment_patch", obj_info),
                _part(
                    parent_ref_id,
                    "plane_surface",
                    obj_info,
                    provenance="frozen_architecture",
                ),
            )
            program_id = f"plane:{child}->{parent_ref_id}"
            factors = [
                _factor(
                    program_id,
                    "gap",
                    "plane_gap",
                    participants,
                    operator="abs_le",
                    threshold=DEFAULT_THRESHOLDS["plane_gap_m"],
                    measurement="plane_gap_m",
                    solver_channel="plane_gap",
                    solver_unit="m",
                    solver_variable_objects=(child,),
                    certificate_unit="m",
                )
            ]
            if record.get("orientation_required", True):
                factors.append(
                    _factor(
                        program_id,
                        "orientation",
                        "plane_orientation",
                        participants,
                        operator="abs_le",
                        threshold=DEFAULT_THRESHOLDS["plane_orientation_deg"],
                        measurement="plane_orientation_deg",
                        solver_channel="plane_orientation_alignment",
                        solver_unit="dimensionless",
                        solver_variable_objects=(child,),
                        certificate_unit="deg",
                    )
                )
            add_program(
                RelationProgram(
                    program_id=program_id,
                    kind=kind,
                    participants=participants,
                    variable_blocks=(blocks[child],),
                    factors=tuple(factors),
                    probes=(
                        ProbeSpec(
                            probe_id=f"{program_id}:detachment",
                            kind="plane_detachment",
                            required_evidence=("detachment_survival",),
                        ),
                    ),
                    certificate_policy=_program_policy(),
                    source_relation=record,
                )
            )
        elif kind == "INSIDE":
            child, parent = record["child"], record["parent"]
            parent_meta = affordances.get(parent, {})
            cavities = parent_meta.get("cavities") or []
            openings = parent_meta.get("openings") or []
            if not cavities or not openings:
                abstained.append(
                    {
                        **record,
                        "status": "ABSTAIN",
                        "reason": "missing explicit cavity or opening geometry",
                    }
                )
                continue
            participants = (
                _part(child, "whole", obj_info),
                _part(parent, str(cavities[0]["part_id"]), obj_info, provenance="affordance_cache"),
                _part(parent, str(openings[0]["part_id"]), obj_info, provenance="affordance_cache"),
            )
            program_id = f"inside:{child}->{parent}"
            add_program(
                RelationProgram(
                    program_id=program_id,
                    kind=kind,
                    participants=participants,
                    variable_blocks=(blocks[child], blocks[parent]),
                    factors=(
                        _factor(
                            program_id,
                            "containment",
                            "inside_containment",
                            participants[:2],
                            operator="le",
                            threshold=DEFAULT_THRESHOLDS["containment_error_m"],
                            measurement="inside_containment_error_m",
                            role="certificate",
                            certificate_unit="m",
                        ),
                        _factor(
                            program_id,
                            "opening",
                            "opening_passage",
                            participants,
                            operator="true",
                            threshold=1.0,
                            measurement="opening_passage",
                            role="certificate",
                            certificate_unit="bool",
                        ),
                    ),
                    probes=(
                        ProbeSpec(
                            probe_id=f"{program_id}:passage",
                            kind="opening_passage",
                            required_evidence=("opening_passage",),
                        ),
                    ),
                    certificate_policy=_program_policy(),
                    source_relation=record,
                )
            )
        elif kind == "COLLISION_EXCLUSION":
            first, second = record["first"], record["second"]
            participants = (
                _part(first, "whole", obj_info),
                _part(second, "whole", obj_info),
            )
            program_id = f"collision:{min(first, second)}<->{max(first, second)}"
            add_program(
                RelationProgram(
                    program_id=program_id,
                    kind=kind,
                    participants=participants,
                    variable_blocks=(blocks[first], blocks[second]),
                    factors=(
                        _factor(
                            program_id,
                            "fraction",
                            "collision_fraction",
                            participants,
                            operator="le",
                            threshold=DEFAULT_THRESHOLDS["collision_fraction"],
                            measurement="collision_fraction",
                            solver_channel="collision_oriented_penetration",
                            solver_unit="m",
                            solver_variable_objects=(first, second),
                            certificate_unit="ratio",
                        ),
                    ),
                    probes=(),
                    certificate_policy=_program_policy(),
                    source_relation=record,
                )
            )
        elif kind in {"POINT_TOWARDS", "ALIGN", "DISTANCE"}:
            source, target = record["source"], record["target"]
            participants = (
                _part(source, "whole", obj_info),
                _part(target, "whole", obj_info),
            )
            program_id = f"{kind.lower()}:{source}->{target}"
            parameters = {key: value for key, value in record.items() if key not in {"kind", "source", "target"}}
            add_program(
                RelationProgram(
                    program_id=program_id,
                    kind=kind,
                    participants=participants,
                    variable_blocks=(blocks[source], blocks[target]),
                    factors=(
                        FactorSpec(
                            factor_id=f"{program_id}:semantic",
                            kind=kind.lower(),
                            participants=participants,
                            parameters=parameters,
                            severity="soft",
                            role="solver",
                            solver_channel={
                                "POINT_TOWARDS": "semantic_point_towards",
                                "ALIGN": "semantic_align",
                                "DISTANCE": "semantic_distance_interval",
                            }[kind],
                            solver_unit=(
                                "m2" if kind == "DISTANCE" else "dimensionless"
                            ),
                            solver_variable_objects=(source,),
                        ),
                    ),
                    probes=(),
                    certificate_policy=_program_policy(),
                    source_relation=record,
                )
            )
        else:
            rejected.append({**record, "reason": "unsupported relation kind"})

    geometry_payload = {
        object_id: _geometry_ref(object_id, obj_info) for object_id in object_ids
    }
    geometry_hash = hashlib.sha256(
        stable_json(geometry_payload).encode("utf-8")
    ).hexdigest()
    bundle = ProgramBundle(
        scene_id=scene_id,
        geometry_manifest_hash=geometry_hash,
        object_ids=object_ids,
        programs=tuple(programs),
        abstained_relations=tuple(abstained),
        rejected_relations=tuple(rejected),
        compiler_audit={
            "input_relations": len(relation_records),
            "compiled": len(programs),
            "abstained": len(abstained),
            "rejected": len(rejected),
            "rotation_mode": "world_so3",
            "scale_mode": "frozen",
            "solver_binding": "audit_only_until_block_parity",
            "support_topology_authoritative": bool(
                support_topology_authoritative
            ),
        },
    )
    bundle.validate()
    return bundle


def audit_live_factor_parity(
    bundle: ProgramBundle,
    *,
    ordered_ids: Sequence[str],
    support_pairs: Iterable[Sequence[Any]] = (),
    fixed_support_pairs: Iterable[Sequence[Any]] = (),
    plane_bindings: Iterable[Mapping[str, Any]] = (),
    collision_pairs: Iterable[Sequence[Any]] = (),
    semantic_specs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare live S4 factor arrays with their compiled program owners."""
    object_ids = tuple(str(value) for value in ordered_ids)
    support = set(_normalise_pairs(support_pairs, object_ids))
    support.update((str(child), str(parent)) for child, parent in fixed_support_pairs)
    planes = {
        (str(value["child_id"]), str(value["plane_id"]))
        for value in plane_bindings
    }
    collisions = {
        tuple(sorted(pair))
        for pair in _normalise_pairs(collision_pairs, object_ids)
        if pair[0] != pair[1]
    }
    semantics = dict(semantic_specs or {})
    expected = {
        "SUPPORT": len(support),
        "PLANE_ATTACH": len(planes),
        "COLLISION_EXCLUSION": len(collisions),
        "POINT_TOWARDS": len(set(map(tuple, semantics.get("point_pairs", ())))),
        "ALIGN": len(set(map(tuple, semantics.get("align_pairs", ())))),
        "DISTANCE": len(set(map(tuple, semantics.get("distance_pairs", ())))),
    }
    compiled_counter = Counter(program.kind for program in bundle.programs)
    compiled = {kind: int(compiled_counter.get(kind, 0)) for kind in expected}
    mismatches = {
        kind: {"expected": expected[kind], "compiled": compiled[kind]}
        for kind in expected
        if expected[kind] != compiled[kind]
    }
    rejected_counter = Counter(
        str(relation.get("kind", "UNKNOWN"))
        for relation in bundle.rejected_relations
    )
    expected_semantic_skips = len(semantics.get("skipped", ()))
    actual_semantic_skips = int(rejected_counter.get("SEMANTIC_SKIPPED", 0))
    if expected_semantic_skips != actual_semantic_skips:
        mismatches["SEMANTIC_SKIPPED"] = {
            "expected": expected_semantic_skips,
            "compiled": actual_semantic_skips,
        }
    accounted = (
        int(bundle.compiler_audit["input_relations"])
        == len(bundle.programs)
        + len(bundle.abstained_relations)
        + len(bundle.rejected_relations)
    )
    if not accounted:
        mismatches["RELATION_ACCOUNTING"] = {
            "expected": int(bundle.compiler_audit["input_relations"]),
            "compiled": (
                len(bundle.programs)
                + len(bundle.abstained_relations)
                + len(bundle.rejected_relations)
            ),
        }
    return {
        "schema_version": "sceneproof_live_factor_parity_v1",
        "passed": not mismatches,
        "expected_program_kinds": expected,
        "compiled_program_kinds": compiled,
        "mismatches": mismatches,
        "abstained": len(bundle.abstained_relations),
        "rejected": len(bundle.rejected_relations),
        "rejected_by_kind": dict(sorted(rejected_counter.items())),
        "relation_accounting_passed": accounted,
        "program_bundle_hash": bundle.content_hash(),
    }
