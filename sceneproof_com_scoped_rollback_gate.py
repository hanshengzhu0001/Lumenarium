#!/usr/bin/env python3
"""Final metric gate for the materialized COM-scoped Paper30 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def gt_value(document, version, key):
    value = document["versions"][version].get(key)
    return float(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--fix61-final-gates", type=Path, required=True)
    parser.add_argument("--incumbent-version", required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--target-version", required=True)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    materialization = load(args.materialization)
    protocol = load(args.protocol)
    physical = load(args.physical)
    gt = load(args.gt)
    fix61_gate = load(args.fix61_final_gates)
    versions = physical["versions"]
    incumbent = versions[args.incumbent_version]["aggregate"]
    candidate = versions[args.candidate_version]["aggregate"]
    target = versions[args.target_version]["aggregate"]
    gates = {
        "protocol_passed": bool(protocol.get("passed")),
        "materialization_complete": not materialization.get("failures")
        and materialization.get("completed") == 30,
        "nonzero_scene_local_rollbacks": materialization.get(
            "rolled_back_objects", 0
        )
        > 0,
    }
    family_rows = {}
    for family in ("collision", "support", "plane", "semantic"):
        base = incumbent["families"][family]["score"]
        old = candidate["families"][family]["score"]
        new = target["families"][family]["score"]
        if base is None or old is None or new is None:
            continue
        row = {
            "delta_vs_incumbent": float(new) - float(base),
            "delta_vs_fix61": float(new) - float(old),
        }
        row["noninferior_to_incumbent"] = (
            row["delta_vs_incumbent"] >= -args.margin
        )
        row["nonworse_than_fix61"] = row["delta_vs_fix61"] >= -1e-8
        family_rows[family] = row
        gates[f"{family}_noninferior_to_incumbent"] = row[
            "noninferior_to_incumbent"
        ]
        gates[f"{family}_nonworse_than_fix61"] = row[
            "nonworse_than_fix61"
        ]
    macro_base = float(incumbent["headline_macro_realizability"])
    macro_old = float(candidate["headline_macro_realizability"])
    macro_new = float(target["headline_macro_realizability"])
    macro_vs_incumbent = macro_new - macro_base
    macro_vs_fix61 = macro_new - macro_old
    gates["physical_macro_noninferior_to_incumbent"] = (
        macro_vs_incumbent >= -args.margin
    )
    gates["physical_macro_improves_fix61"] = macro_vs_fix61 > 0.0

    gt_rows = {}
    for key, margin in (
        ("object_recovery", 0.0),
        ("scene_graph_parent_accuracy_gt", 0.0),
        ("rotation_auc60_aligned", 0.01),
        ("translation_auc05_aligned", 0.005),
    ):
        base = gt_value(gt, args.incumbent_version, key)
        old = gt_value(gt, args.candidate_version, key)
        new = gt_value(gt, args.target_version, key)
        if base is None or old is None or new is None:
            continue
        gt_rows[key] = {
            "delta_vs_incumbent": new - base,
            "delta_vs_fix61": new - old,
        }
        gates[f"{key}_noninferior_to_incumbent"] = new - base >= -margin

    old_runtime = fix61_gate["runtime"]
    runtime_rows = [
        json.loads(row)
        for row in args.materialization.with_name(
            "materialization_runtime.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    extra_mean = sum(row["elapsed_seconds"] for row in runtime_rows) / max(
        len(runtime_rows), 1
    )
    end_to_end = float(old_runtime["certified_end_to_end_seconds"]) + extra_mean
    speedup = float(old_runtime["legacy_seconds"]) / end_to_end
    runtime = {
        "legacy_seconds": float(old_runtime["legacy_seconds"]),
        "fix61_end_to_end_seconds": float(
            old_runtime["certified_end_to_end_seconds"]
        ),
        "materialization_mean_seconds": extra_mean,
        "fix68_end_to_end_seconds": end_to_end,
        "speedup": speedup,
    }
    gates["sa5000_speedup_at_least_1_5x"] = speedup >= 1.5
    gates["no_failures"] = (
        not materialization.get("failures")
        and not physical.get("failures")
        and not gt.get("failures")
    )
    passed = all(gates.values())
    result = {
        "schema_version": "sceneproof_com_scoped_rollback_gate_v1",
        "passed": passed,
        "physical_macro_delta_vs_incumbent": macro_vs_incumbent,
        "physical_macro_delta_vs_fix61": macro_vs_fix61,
        "physical_families": family_rows,
        "gt": gt_rows,
        "runtime": runtime,
        "gates": gates,
        "decision": (
            "render_and_visually_audit_fix68" if passed else "retain_fix61"
        ),
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.out.resolve()}")
    print(
        f"PASSED={passed} MACRO_VS_INCUMBENT={macro_vs_incumbent:+.9f} "
        f"MACRO_VS_FIX61={macro_vs_fix61:+.9f} SPEEDUP={speedup:.3f}x "
        f"DECISION={result['decision']}"
    )
    print("FAMILY_ROWS=", json.dumps(family_rows, sort_keys=True))
    print("GT_ROWS=", json.dumps(gt_rows, sort_keys=True))
    print("GATES=", json.dumps(gates, sort_keys=True))


if __name__ == "__main__":
    main()
