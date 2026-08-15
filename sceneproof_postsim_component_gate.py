#!/usr/bin/env python3
"""Final gate for the cached post-simulation component certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def gt_metric(document: dict[str, Any], version: str, primary: str, legacy: str) -> float:
    row = document["versions"][version]
    value = row.get(primary, row.get(legacy))
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--legacy-version")
    parser.add_argument("--incumbent-version", required=True)
    parser.add_argument("--target-version", required=True)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--rotation-margin", type=float, default=0.01)
    parser.add_argument("--translation-margin", type=float, default=0.005)
    parser.add_argument("--minimum-speedup", type=float, default=1.5)
    parser.add_argument("--allow-safe-abstain", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    certificate, physical, gt = (
        load(args.certificate), load(args.physical), load(args.gt)
    )
    base = physical["versions"][args.incumbent_version]["aggregate"]
    target = physical["versions"][args.target_version]["aggregate"]
    deltas: dict[str, float] = {}
    gates: dict[str, bool] = {}
    for family in ("collision", "support", "plane", "semantic"):
        first = base["families"][family]["score"]
        second = target["families"][family]["score"]
        if first is None or second is None:
            continue
        deltas[family] = float(second) - float(first)
        gates[f"{family}_noninferior"] = deltas[family] >= -args.margin
    macro_delta = float(target["headline_macro_realizability"]) - float(
        base["headline_macro_realizability"]
    )
    rotation_delta = gt_metric(
        gt, args.target_version, "rotation_auc60_aligned", "rotation_auc60"
    ) - gt_metric(
        gt, args.incumbent_version, "rotation_auc60_aligned", "rotation_auc60"
    )
    translation_delta = gt_metric(
        gt, args.target_version, "translation_auc05_aligned", "translation_auc05"
    ) - gt_metric(
        gt, args.incumbent_version, "translation_auc05_aligned", "translation_auc05"
    )
    retained = sum(
        len(row.get("retained_changed_objects", []))
        for row in certificate.get("scenes", {}).values()
    )
    gates.update(
        {
            "physical_macro_noninferior": macro_delta >= -args.margin,
            "rotation_noninferior": rotation_delta >= -args.rotation_margin,
            "translation_noninferior": translation_delta >= -args.translation_margin,
            "nonzero_candidate_change_retained": retained > 0,
            "no_failures": not certificate.get("failures")
            and not physical.get("failures")
            and not gt.get("failures"),
        }
    )
    runtime = None
    if args.legacy_version:
        legacy_runtime = physical["versions"][args.legacy_version]["runtime"].get(
            "mean_seconds"
        )
        target_runtime = physical["versions"][args.target_version]["runtime"].get(
            "mean_seconds"
        )
        if legacy_runtime is not None and target_runtime is not None:
            speedup = float(legacy_runtime) / float(target_runtime)
            runtime = {
                "legacy_seconds": float(legacy_runtime),
                "certified_gpu_seconds": float(target_runtime),
                "certified_end_to_end_seconds": float(target_runtime),
                "speedup": speedup,
            }
            gates["sa5000_gpu_time_speedup"] = speedup >= args.minimum_speedup
        else:
            gates["sa5000_gpu_time_speedup"] = False
    safety_passed = all(
        value
        for name, value in gates.items()
        if name != "nonzero_candidate_change_retained"
    )
    accepted_nonzero = bool(safety_passed and retained > 0)
    safe_abstain = bool(
        args.allow_safe_abstain and safety_passed and retained == 0
    )
    passed = bool(accepted_nonzero or safe_abstain)
    outcome = (
        "accepted_nonzero"
        if accepted_nonzero
        else "safe_abstain_incumbent"
        if safe_abstain
        else "unsafe_or_no_effect"
    )
    result = {
        "schema_version": "sceneproof_postsim_component_gate_v2",
        "passed": passed,
        "safety_passed": safety_passed,
        "outcome": outcome,
        "safe_abstain_allowed": bool(args.allow_safe_abstain),
        "retained_changed_objects": retained,
        "physical_macro_delta": macro_delta,
        "physical_family_deltas": deltas,
        "rotation_delta": rotation_delta,
        "translation_delta": translation_delta,
        "runtime": runtime,
        "gates": gates,
        "decision": (
            "integrate_postsim_component_certificate"
            if accepted_nonzero
            else "expand_to_smoke5_safe_abstain"
            if safe_abstain
            else "stop_or_revise_component_scope"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(
        f"PASSED={passed} OUTCOME={outcome} RETAINED={retained} "
        f"PHYSICAL_DELTA={macro_delta:+.6f} "
        f"ROT_DELTA={rotation_delta:+.6f} TRANS_DELTA={translation_delta:+.6f}"
    )
    if runtime is not None:
        print(
            "RUNTIME_SA5000/CERTIFIED/SPEEDUP=",
            f"{runtime['legacy_seconds']:.3f}/"
            f"{runtime['certified_gpu_seconds']:.3f}/"
            f"{runtime['speedup']:.3f}x",
        )
    print("FAMILY_DELTAS=", json.dumps(deltas, sort_keys=True))
    print("GATES=", json.dumps(gates, sort_keys=True))
    print(f"DECISION={result['decision']}")


if __name__ == "__main__":
    main()
