#!/usr/bin/env python3
"""Aggregate process-isolated local gravity-settle counterfactual probes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def restoration_certificate(row: dict) -> tuple[bool, float]:
    before = row.get("before_pose_matrix")
    try:
        values = [abs(float(value)) for matrix_row in before for value in matrix_row]
        scale = max(values) if len(values) == 16 else 1.0
        if not math.isfinite(scale):
            scale = 1.0
    except (TypeError, ValueError):
        scale = 1.0
    tolerance = float(
        row.get(
            "restoration_tolerance",
            16.0 * 1.1920928955078125e-7 * max(1.0, scale),
        )
    )
    error = row.get("maximum_restoration_error")
    if error is None:
        return bool(row.get("incumbent_restored")), tolerance
    restored = (
        math.isfinite(float(error))
        and float(error) <= tolerance
    )
    return bool(restored), tolerance


def classify_probe(row: dict) -> str:
    if row.get("status") != "measured":
        return "abstained_or_failed"
    restored, _ = restoration_certificate(row)
    if not restored:
        return "unsafe_restoration_failure"
    if row.get("new_collision_object_ids"):
        return "rejected_new_collision"
    after = row.get("after_support") or {}
    if (
        after.get("certificate_status") == "certified"
        and after.get("stability_class") in {"stable", "marginal"}
        and after.get("declared_parent_contact_present")
    ):
        return "locally_promising_requires_full_component_gates"
    return "visibility_or_support_unresolved"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-audit", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    action_audit = json.loads(args.action_audit.read_text(encoding="utf-8"))
    expected = [
        row["object_id"]
        for row in action_audit.get("actionable_objects", [])
        if row.get("action") == "local_gravity_settle_probe_candidate"
    ]
    probes = {}
    missing = []
    for object_id in expected:
        path = args.probe_root / f"{object_id}.json"
        if not path.is_file():
            missing.append(object_id)
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        restored, restoration_tolerance = restoration_certificate(row)
        row["incumbent_restored"] = restored
        row["restoration_tolerance"] = restoration_tolerance
        row["restoration_storage_dtype"] = "float32"
        row["outcome"] = classify_probe(row)
        probes[object_id] = row

    outcomes = {}
    for row in probes.values():
        outcome = row["outcome"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    restored = all(row.get("incumbent_restored") for row in probes.values())
    failures = [
        object_id
        for object_id, row in probes.items()
        if row.get("status") == "failed"
        or row.get("outcome") == "unsafe_restoration_failure"
    ]
    promising = [
        object_id
        for object_id, row in probes.items()
        if row.get("outcome")
        == "locally_promising_requires_full_component_gates"
    ]
    complete = len(probes) == len(expected) and not missing
    if not complete or failures or not restored:
        decision = "unsafe_stop_and_diagnose_local_settle"
    elif promising:
        decision = "measure_full_component_gates_before_any_pose_commit"
    else:
        decision = "hold_fix76_local_settle_unresolved"

    result = {
        "schema_version": "sceneproof_local_settle_oracle_aggregate_v1",
        "policy": "audit_only_no_pose_commit_full_so3_process_isolated",
        "action_audit": str(args.action_audit.resolve()),
        "expected_object_ids": expected,
        "missing_object_ids": missing,
        "probes": probes,
        "summary": {
            "expected": len(expected),
            "measured": len(probes),
            "failures": len(failures),
            "all_incumbents_restored": restored,
            "outcomes": outcomes,
            "locally_promising_object_ids": promising,
        },
        "decision": decision,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "SCENEPROOF LOCAL FULL-SO(3) GRAVITY-SETTLE ORACLE",
        "=" * 72,
        f"EXPECTED/MEASURED={len(expected)}/{len(probes)} FAILURES={len(failures)}",
        f"ALL_INCUMBENTS_RESTORED={restored}",
        f"OUTCOMES={json.dumps(outcomes, sort_keys=True)}",
        f"DECISION={decision}",
    ]
    for object_id in expected:
        row = probes.get(object_id)
        if row is None:
            lines.append(f"{object_id} outcome=missing")
            continue
        after = row.get("after_support") or {}
        lines.append(
            f"{object_id} outcome={row['outcome']} "
            f"translation_m={row.get('translation_delta_m')} "
            f"rotation_deg={row.get('rotation_delta_deg')} "
            f"support={after.get('certificate_status')}/"
            f"{after.get('stability_class')} "
            f"new_collisions={row.get('new_collision_object_ids', [])}"
        )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out.resolve()}")
    print(f"Wrote {args.report.resolve()}")
    print("\n".join(lines[2:]))


if __name__ == "__main__":
    main()
