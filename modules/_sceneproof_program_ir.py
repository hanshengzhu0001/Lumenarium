"""Typed, deterministic Relation Program IR for SceneProof.

This module deliberately has no Blender, NumPy, or PyTorch dependency.  It is
the shared contract between relation compilation, numeric optimization, and
independent execution/certification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "sceneproof_relation_program_v2"
VALID_PROGRAM_KINDS = {
    "SUPPORT",
    "STACK",
    "PLANE_ATTACH",
    "CEILING_ATTACH",
    "HANG",
    "INSIDE",
    "COLLISION_EXCLUSION",
    "POINT_TOWARDS",
    "ALIGN",
    "DISTANCE",
}
VALID_ROTATION_MODES = {"world_so3", "rigid_relative_so3"}
VALID_TRANSLATION_CHARTS = {
    "world_xyz",
    "support_uvh",
    "plane_uvn",
}
VALID_SEVERITIES = {"hard", "soft"}
VALID_STATUSES = {"PASS", "FAIL", "ABSTAIN"}
VALID_FACTOR_ROLES = {"solver", "certificate", "both"}


def _require_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def stable_json(value: Any) -> str:
    """Return canonical JSON suitable for hashes and byte-level rollback."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class PartRef:
    object_id: str
    part_id: str
    geometry_ref: str
    confidence: float = 1.0
    provenance: str = "frozen_s3"

    def validate(self) -> None:
        _require_nonempty(self.object_id, "object_id")
        _require_nonempty(self.part_id, "part_id")
        _require_nonempty(self.geometry_ref, "geometry_ref")
        _require_nonempty(self.provenance, "provenance")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("part confidence must lie in [0, 1]")


@dataclass(frozen=True)
class VariableBlockSpec:
    object_id: str
    rotation_mode: str
    translation_chart: str
    scale_mode: str = "frozen"
    parent_frame: str | None = None
    eliminable_translation: bool = False

    def validate(self) -> None:
        _require_nonempty(self.object_id, "object_id")
        if self.rotation_mode not in VALID_ROTATION_MODES:
            raise ValueError(f"unsupported rotation mode {self.rotation_mode!r}")
        if self.translation_chart not in VALID_TRANSLATION_CHARTS:
            raise ValueError(
                f"unsupported translation chart {self.translation_chart!r}"
            )
        if self.scale_mode != "frozen":
            raise ValueError("Relation Program v1 requires frozen scale")
        if self.translation_chart == "support_uvh" and not self.parent_frame:
            raise ValueError("support_uvh requires a parent_frame")

    def ownership_key(self) -> tuple[Any, ...]:
        return (
            self.rotation_mode,
            self.translation_chart,
            self.scale_mode,
            self.parent_frame,
            bool(self.eliminable_translation),
        )


@dataclass(frozen=True)
class FactorSpec:
    factor_id: str
    kind: str
    participants: tuple[PartRef, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    robust_kernel: Mapping[str, Any] = field(
        default_factory=lambda: {"kind": "huber", "delta": 1.0}
    )
    severity: str = "hard"
    role: str = "certificate"
    solver_channel: str | None = None
    solver_unit: str | None = None
    solver_variable_objects: tuple[str, ...] = ()
    certificate_measurement: str | None = None
    certificate_unit: str | None = None

    def validate(self) -> None:
        _require_nonempty(self.factor_id, "factor_id")
        _require_nonempty(self.kind, "factor kind")
        if not self.participants:
            raise ValueError("a factor must have at least one participant")
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(f"unsupported factor severity {self.severity!r}")
        if self.role not in VALID_FACTOR_ROLES:
            raise ValueError(f"unsupported factor role {self.role!r}")
        if self.role in {"solver", "both"}:
            _require_nonempty(str(self.solver_channel or ""), "solver_channel")
            _require_nonempty(str(self.solver_unit or ""), "solver_unit")
            if not self.solver_variable_objects:
                raise ValueError("solver factors require variable ownership")
            participant_objects = {
                participant.object_id for participant in self.participants
            }
            if not set(self.solver_variable_objects).issubset(participant_objects):
                raise ValueError("solver variable owner is outside factor participants")
        elif (
            self.solver_channel is not None
            or self.solver_unit is not None
            or self.solver_variable_objects
        ):
            raise ValueError("certificate-only factors cannot own solver channels")
        if self.role in {"certificate", "both"}:
            _require_nonempty(
                str(self.certificate_measurement or ""),
                "certificate_measurement",
            )
            _require_nonempty(
                str(self.certificate_unit or ""), "certificate_unit"
            )
            legacy_measurement = self.parameters.get("measurement")
            if (
                legacy_measurement is not None
                and str(legacy_measurement) != self.certificate_measurement
            ):
                raise ValueError(
                    "certificate measurement disagrees with factor parameters"
                )
        elif (
            self.certificate_measurement is not None
            or self.certificate_unit is not None
        ):
            raise ValueError("solver-only factors cannot own certificate evidence")
        for participant in self.participants:
            participant.validate()


@dataclass(frozen=True)
class ProbeSpec:
    probe_id: str
    kind: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 1.0
    required_evidence: tuple[str, ...] = ()

    def validate(self) -> None:
        _require_nonempty(self.probe_id, "probe_id")
        _require_nonempty(self.kind, "probe kind")
        if float(self.timeout_seconds) <= 0.0:
            raise ValueError("probe timeout must be positive")


@dataclass(frozen=True)
class CertificatePolicy:
    thresholds: Mapping[str, float]
    aggregation: str = "component_wise"
    missing_evidence: str = "abstain"

    def validate(self) -> None:
        if self.aggregation != "component_wise":
            raise ValueError("Relation Program v1 requires component-wise policy")
        if self.missing_evidence != "abstain":
            raise ValueError("missing evidence must cause abstention")
        for name, value in self.thresholds.items():
            _require_nonempty(str(name), "threshold name")
            if not isinstance(value, (int, float)):
                raise TypeError(f"threshold {name!r} must be numeric")


@dataclass(frozen=True)
class RelationProgram:
    program_id: str
    kind: str
    participants: tuple[PartRef, ...]
    variable_blocks: tuple[VariableBlockSpec, ...]
    factors: tuple[FactorSpec, ...]
    probes: tuple[ProbeSpec, ...]
    certificate_policy: CertificatePolicy
    source_relation: Mapping[str, Any]
    compiler_confidence: float = 1.0

    def validate(self) -> None:
        _require_nonempty(self.program_id, "program_id")
        if self.kind not in VALID_PROGRAM_KINDS:
            raise ValueError(f"unsupported program kind {self.kind!r}")
        if not self.participants:
            raise ValueError("a relation program must have participants")
        if not 0.0 <= float(self.compiler_confidence) <= 1.0:
            raise ValueError("compiler confidence must lie in [0, 1]")
        for participant in self.participants:
            participant.validate()
        participant_objects = {
            participant.object_id for participant in self.participants
        }
        factor_ids: set[str] = set()
        for factor in self.factors:
            factor.validate()
            if factor.factor_id in factor_ids:
                raise ValueError(f"duplicate factor id {factor.factor_id!r}")
            factor_ids.add(factor.factor_id)
            if not {
                participant.object_id for participant in factor.participants
            }.issubset(participant_objects):
                raise ValueError("factor participant is outside its program")
        probe_ids: set[str] = set()
        for probe in self.probes:
            probe.validate()
            if probe.probe_id in probe_ids:
                raise ValueError(f"duplicate probe id {probe.probe_id!r}")
            probe_ids.add(probe.probe_id)
        block_objects: set[str] = set()
        for block in self.variable_blocks:
            block.validate()
            if block.object_id in block_objects:
                raise ValueError("a program repeats a variable block")
            block_objects.add(block.object_id)
        self.certificate_policy.validate()
        if self.kind == "INSIDE":
            part_ids = {part.part_id for part in self.participants}
            if not any("cavity" in value for value in part_ids) or not any(
                "opening" in value for value in part_ids
            ):
                raise ValueError(
                    "compiled INSIDE programs require cavity and opening parts"
                )


@dataclass(frozen=True)
class ProgramBundle:
    scene_id: str
    geometry_manifest_hash: str
    object_ids: tuple[str, ...]
    programs: tuple[RelationProgram, ...]
    abstained_relations: tuple[Mapping[str, Any], ...] = ()
    rejected_relations: tuple[Mapping[str, Any], ...] = ()
    compiler_audit: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version {self.schema_version!r}")
        _require_nonempty(self.scene_id, "scene_id")
        _require_nonempty(self.geometry_manifest_hash, "geometry_manifest_hash")
        if len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("object_ids must be unique")
        known_objects = set(self.object_ids)
        program_ids: set[str] = set()
        ownership: dict[str, tuple[Any, ...]] = {}
        parent_by_child: dict[str, str] = {}
        for program in self.programs:
            program.validate()
            if program.program_id in program_ids:
                raise ValueError(f"duplicate program id {program.program_id!r}")
            program_ids.add(program.program_id)
            for participant in program.participants:
                if (
                    participant.object_id not in known_objects
                    and not participant.object_id.startswith("architecture:")
                ):
                    raise ValueError(
                        f"unknown program object {participant.object_id!r}"
                    )
            for block in program.variable_blocks:
                if block.object_id not in known_objects:
                    raise ValueError(f"unknown variable owner {block.object_id!r}")
                key = block.ownership_key()
                if block.object_id in ownership and ownership[block.object_id] != key:
                    raise ValueError(
                        f"conflicting variable ownership for {block.object_id!r}"
                    )
                ownership[block.object_id] = key
                parent = block.parent_frame
                if parent in known_objects:
                    previous = parent_by_child.get(block.object_id)
                    if previous not in (None, parent):
                        raise ValueError(
                            f"multiple support parents for {block.object_id!r}"
                        )
                    parent_by_child[block.object_id] = parent

        state: dict[str, int] = {}

        def visit(object_id: str) -> None:
            marker = state.get(object_id, 0)
            if marker == 2:
                return
            if marker == 1:
                raise ValueError("relation-program ownership graph contains a cycle")
            state[object_id] = 1
            parent = parent_by_child.get(object_id)
            if parent is not None:
                visit(parent)
            state[object_id] = 2

        for object_id in self.object_ids:
            visit(object_id)

        expected = self.compiler_audit.get("input_relations")
        if expected is not None:
            accounted = (
                len(self.programs)
                + len(self.abstained_relations)
                + len(self.rejected_relations)
            )
            if int(expected) != accounted:
                raise ValueError(
                    f"relation accounting mismatch: {expected} != {accounted}"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def canonical_json(self) -> str:
        return stable_json(self.to_dict())

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def program_index(bundle: ProgramBundle) -> dict[str, RelationProgram]:
    bundle.validate()
    return {program.program_id: program for program in bundle.programs}


def variable_ownership(
    programs: Iterable[RelationProgram],
) -> dict[str, VariableBlockSpec]:
    result: dict[str, VariableBlockSpec] = {}
    for program in programs:
        for block in program.variable_blocks:
            previous = result.get(block.object_id)
            if previous is not None and previous.ownership_key() != block.ownership_key():
                raise ValueError(
                    f"conflicting variable ownership for {block.object_id!r}"
                )
            result[block.object_id] = block
    return result
