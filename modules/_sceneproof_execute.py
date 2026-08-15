"""Independent execution and certification of SceneProof relation programs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from modules._sceneproof_program_ir import RelationProgram, VALID_STATUSES


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: str
    measurement: str
    value: Any
    operator: str
    threshold: float | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid check status {self.status!r}")


@dataclass(frozen=True)
class ProgramCertificate:
    program_id: str
    kind: str
    status: str
    static_status: str
    checks: tuple[CheckResult, ...]
    probes: tuple[CheckResult, ...]
    witness: Mapping[str, Any] | None
    missing_evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"invalid program status {self.status!r}")
        if self.static_status not in VALID_STATUSES:
            raise ValueError(f"invalid static status {self.static_status!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _aggregate_status(results: tuple[CheckResult, ...]) -> str:
    if any(result.status == "FAIL" for result in results):
        return "FAIL"
    if any(result.status == "ABSTAIN" for result in results):
        return "ABSTAIN"
    return "PASS"


def _evaluate(
    *,
    check_id: str,
    measurement: str,
    operator: str,
    threshold: float | None,
    measurements: Mapping[str, Any],
) -> CheckResult:
    if measurement not in measurements or measurements[measurement] is None:
        return CheckResult(
            check_id=check_id,
            status="ABSTAIN",
            measurement=measurement,
            value=None,
            operator=operator,
            threshold=threshold,
            reason="missing required evidence",
        )
    value = measurements[measurement]
    if operator == "true":
        passed = bool(value)
    else:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return CheckResult(
                check_id=check_id,
                status="ABSTAIN",
                measurement=measurement,
                value=value,
                operator=operator,
                threshold=threshold,
                reason="non-numeric evidence",
            )
        if not math.isfinite(numeric) or threshold is None:
            return CheckResult(
                check_id=check_id,
                status="ABSTAIN",
                measurement=measurement,
                value=value,
                operator=operator,
                threshold=threshold,
                reason="non-finite evidence or missing threshold",
            )
        if operator == "le":
            passed = numeric <= threshold
        elif operator == "ge":
            passed = numeric >= threshold
        elif operator == "abs_le":
            passed = abs(numeric) <= threshold
        else:
            raise ValueError(f"unsupported certificate operator {operator!r}")
    return CheckResult(
        check_id=check_id,
        status="PASS" if passed else "FAIL",
        measurement=measurement,
        value=value,
        operator=operator,
        threshold=threshold,
        reason=None if passed else "threshold violation",
    )


def execute_relation_program(
    program: RelationProgram,
    measurements: Mapping[str, Any],
) -> ProgramCertificate:
    """Execute a program from measurements produced independently of its loss."""
    program.validate()
    checks: list[CheckResult] = []
    for factor in program.factors:
        # Solver-only factors are deliberately invisible to the independent
        # proof executor.  This prevents a differentiable residual from
        # certifying itself and keeps solver/certificate implementations
        # coupled only through immutable FactorSpec semantics.
        if factor.role == "solver":
            continue
        if factor.severity != "hard":
            continue
        parameters = factor.parameters
        checks.append(
            _evaluate(
                check_id=factor.factor_id,
                measurement=str(
                    factor.certificate_measurement
                    or parameters.get("measurement", factor.kind)
                ),
                operator=str(parameters.get("operator", "le")),
                threshold=(
                    float(parameters["threshold"])
                    if "threshold" in parameters
                    else None
                ),
                measurements=measurements,
            )
        )
    probes: list[CheckResult] = []
    for probe in program.probes:
        for evidence in probe.required_evidence:
            probes.append(
                _evaluate(
                    check_id=probe.probe_id,
                    measurement=evidence,
                    operator="true",
                    threshold=1.0,
                    measurements=measurements,
                )
            )
    static_results = tuple(checks)
    probe_results = tuple(probes)
    static_status = _aggregate_status(static_results)
    status = _aggregate_status(static_results + probe_results)
    failed = next(
        (
            result
            for result in static_results + probe_results
            if result.status == "FAIL"
        ),
        None,
    )
    witness = None
    if failed is not None:
        witness = {
            "program_id": program.program_id,
            "factor_or_probe_id": failed.check_id,
            "object_ids": sorted(
                {part.object_id for part in program.participants}
            ),
            "measurement": failed.measurement,
            "value": failed.value,
            "operator": failed.operator,
            "threshold": failed.threshold,
        }
    missing = tuple(
        sorted(
            {
                result.measurement
                for result in static_results + probe_results
                if result.status == "ABSTAIN"
            }
        )
    )
    return ProgramCertificate(
        program_id=program.program_id,
        kind=program.kind,
        status=status,
        static_status=static_status,
        checks=static_results,
        probes=probe_results,
        witness=witness,
        missing_evidence=missing,
    )
