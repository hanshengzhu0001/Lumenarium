"""Component-wise no-harm arbitration and witness-local rollback."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from modules._sceneproof_execute import ProgramCertificate
from modules._sceneproof_program_ir import ProgramBundle, program_index


DEFAULT_NONINFERIORITY_MARGINS = {
    "collision": -0.005,
    "support": -0.005,
    "plane": -0.005,
    "semantic": -0.005,
    "rotation": -0.01,
    "translation": -0.005,
}


@dataclass(frozen=True)
class ArbitrationDecision:
    accepted: bool
    component_deltas: Mapping[str, float]
    failed_components: tuple[str, ...]
    regressed_programs: tuple[str, ...]
    release_object_ids: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def witness_local_release_objects(
    bundle: ProgramBundle,
    program_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return only failed-program objects and their declared separators."""
    index = program_index(bundle)
    released: set[str] = set()
    known = set(bundle.object_ids)
    for program_id in program_ids:
        program = index.get(program_id)
        if program is None:
            raise KeyError(f"unknown failed program {program_id!r}")
        for block in program.variable_blocks:
            released.add(block.object_id)
            if block.parent_frame in known:
                released.add(str(block.parent_frame))
    return tuple(sorted(released))


def arbitrate_candidate(
    *,
    bundle: ProgramBundle,
    incumbent_metrics: Mapping[str, float],
    candidate_metrics: Mapping[str, float],
    incumbent_certificates: Sequence[ProgramCertificate],
    candidate_certificates: Sequence[ProgramCertificate],
    margins: Mapping[str, float] | None = None,
) -> ArbitrationDecision:
    """Accept only when every predeclared component and hard program is safe."""
    bundle.validate()
    effective_margins = dict(DEFAULT_NONINFERIORITY_MARGINS)
    if margins is not None:
        effective_margins.update({key: float(value) for key, value in margins.items()})
    incumbent_by_id = {
        certificate.program_id: certificate
        for certificate in incumbent_certificates
    }
    candidate_by_id = {
        certificate.program_id: certificate
        for certificate in candidate_certificates
    }
    regressed: list[str] = []
    for program in bundle.programs:
        incumbent = incumbent_by_id.get(program.program_id)
        candidate = candidate_by_id.get(program.program_id)
        if incumbent is None or candidate is None:
            regressed.append(program.program_id)
            continue
        if incumbent.status == "PASS" and candidate.status != "PASS":
            regressed.append(program.program_id)
        elif candidate.status == "FAIL" and incumbent.status != "FAIL":
            regressed.append(program.program_id)

    deltas: dict[str, float] = {}
    failed_components: list[str] = []
    for component, margin in effective_margins.items():
        if component not in incumbent_metrics or component not in candidate_metrics:
            failed_components.append(component)
            continue
        delta = float(candidate_metrics[component]) - float(
            incumbent_metrics[component]
        )
        deltas[component] = delta
        if delta < margin:
            failed_components.append(component)

    failed_programs = tuple(sorted(set(regressed)))
    release_objects = witness_local_release_objects(bundle, failed_programs)
    accepted = not failed_components and not failed_programs
    if failed_programs:
        reason = "hard relation program regressed"
    elif failed_components:
        reason = "component-wise non-inferiority failed"
    else:
        reason = "all component and program gates passed"
    return ArbitrationDecision(
        accepted=accepted,
        component_deltas=deltas,
        failed_components=tuple(sorted(set(failed_components))),
        regressed_programs=failed_programs,
        release_object_ids=release_objects,
        reason=reason,
    )


def select_or_rollback(
    incumbent: Any,
    candidate: Any,
    decision: ArbitrationDecision,
) -> Any:
    """Return an isolated copy; a rejection can never mutate the incumbent."""
    return copy.deepcopy(candidate if decision.accepted else incumbent)
