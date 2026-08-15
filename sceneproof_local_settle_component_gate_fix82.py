#!/usr/bin/env python3
"""Fail-closed component gates for one materialized local-settle candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gt_value(document: dict, version: str, primary: str, legacy: str) -> float:
    row = document["versions"][version]
    return float(row.get(primary, row.get(legacy)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--incumbent-version", required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--physical-margin", type=float, default=1e-6)
    parser.add_argument("--pose-margin", type=float, default=0.005)
    parser.add_argument("--boundary-margin", type=float, default=1e-6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    probe, physical, gt = load(args.probe), load(args.physical), load(args.gt)
    base = physical["versions"][args.incumbent_version]["aggregate"]
    candidate = physical["versions"][args.candidate_version]["aggregate"]
    family_deltas = {}
    gates = {}
    for family in ("collision", "support", "plane", "semantic"):
        first = base["families"].get(family, {}).get("score")
        second = candidate["families"].get(family, {}).get("score")
        if first is None and second is None:
            gates[f"{family}_unchanged_or_nonevaluable"] = True
            continue
        if first is None or second is None:
            gates[f"{family}_noninferior"] = False
            continue
        delta = float(second) - float(first)
        family_deltas[family] = delta
        gates[f"{family}_noninferior"] = delta >= -args.physical_margin

    before_boundary = probe.get("before_boundary_error_m")
    after_boundary = probe.get("after_boundary_error_m")
    boundary_evaluable = (
        before_boundary is not None
        and after_boundary is not None
        and math.isfinite(float(before_boundary))
        and math.isfinite(float(after_boundary))
    )
    gates["true_mesh_boundary_evaluable"] = boundary_evaluable
    gates["true_mesh_boundary_noninferior"] = bool(
        boundary_evaluable
        and float(after_boundary)
        <= float(before_boundary) + args.boundary_margin
    )
    gates["no_new_exact_mesh_collision"] = not probe.get(
        "new_collision_object_ids"
    )
    gates["incumbent_restoration_certified"] = bool(
        probe.get("incumbent_restored")
    )
    after_support = probe.get("after_support") or {}
    gates["true_mesh_support_stable"] = bool(
        after_support.get("certificate_status") == "certified"
        and after_support.get("stability_class") in {"stable", "marginal"}
        and after_support.get("declared_parent_contact_present")
    )

    rotation_delta = gt_value(
        gt, args.candidate_version, "rotation_auc60_aligned", "rotation_auc60"
    ) - gt_value(
        gt, args.incumbent_version, "rotation_auc60_aligned", "rotation_auc60"
    )
    translation_delta = gt_value(
        gt,
        args.candidate_version,
        "translation_auc05_aligned",
        "translation_auc05",
    ) - gt_value(
        gt,
        args.incumbent_version,
        "translation_auc05_aligned",
        "translation_auc05",
    )
    gates["rotation_noninferior"] = rotation_delta >= -args.pose_margin
    gates["translation_noninferior"] = translation_delta >= -args.pose_margin
    gates["no_evaluator_failures"] = not physical.get("failures") and not gt.get(
        "failures"
    )
    passed = all(gates.values())
    result = {
        "schema_version": "sceneproof_local_settle_component_gate_v1",
        "passed": passed,
        "promoted": False,
        "object_id": probe.get("object_id"),
        "physical_family_deltas": family_deltas,
        "rotation_delta": rotation_delta,
        "translation_delta": translation_delta,
        "boundary_before_m": before_boundary,
        "boundary_after_m": after_boundary,
        "gates": gates,
        "decision": (
            "render_candidate_before_scoped_commit"
            if passed
            else "rollback_object_to_fix76"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out.resolve()}")
    print(
        f"PASSED={passed} OBJECT={probe.get('object_id')} "
        f"ROT_DELTA={rotation_delta:+.9f} TRANS_DELTA={translation_delta:+.9f}"
    )
    print(f"FAMILY_DELTAS={json.dumps(family_deltas, sort_keys=True)}")
    print(f"GATES={json.dumps(gates, sort_keys=True)}")
    print(f"DECISION={result['decision']}")


if __name__ == "__main__":
    main()
