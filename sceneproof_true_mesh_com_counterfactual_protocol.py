#!/usr/bin/env python3
"""Aggregate per-scene COM counterfactual oracles without materializing poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    scenes = [
        row.strip()
        for row in args.manifest.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    rows = {}
    failures = []
    total_gain = 0.0
    for scene in scenes:
        path = args.oracle_root / f"{scene}.json"
        if not path.is_file():
            failures.append({"scene": scene, "reason": "missing_oracle"})
            continue
        try:
            audit = json.loads(path.read_text(encoding="utf-8"))
            selected = audit.get("selected_oracle")
            authorized = bool(
                audit.get("implementation_authorized") and selected
            )
            gain = (
                float(selected["delta_vs_candidate"]["physical_macro"])
                if authorized
                else 0.0
            )
            if authorized and (
                not selected.get("safe_against_incumbent") or gain <= 0.0
            ):
                raise ValueError("authorized oracle is not safe positive gain")
            rows[scene] = {
                "implementation_authorized": authorized,
                "rollback_object_ids": (
                    selected.get("rollback_object_ids", [])
                    if authorized
                    else []
                ),
                "physical_macro_gain": gain,
                "oracle": str(path.resolve()),
            }
            total_gain += gain
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            failures.append({"scene": scene, "reason": repr(error)})

    completed = len(rows)
    authorized_scenes = sum(
        row["implementation_authorized"] for row in rows.values()
    )
    passed = not failures and completed == len(scenes)
    result = {
        "schema_version": "sceneproof_true_mesh_com_counterfactual_protocol_v1",
        "scenes_expected": len(scenes),
        "scenes_completed": completed,
        "failures": failures,
        "authorized_scenes": authorized_scenes,
        "safe_abstain_scenes": completed - authorized_scenes,
        "summed_positive_scene_macro_gain": total_gain,
        "passed": passed,
        "materialization_authorized": bool(passed and authorized_scenes > 0),
        "decision": (
            "materialize_scene_local_scoped_rollbacks"
            if passed and authorized_scenes > 0
            else "retain_fix61"
        ),
        "scenes": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.out.resolve()}")
    print(
        f"SCENES={completed}/{len(scenes)} FAILURES={len(failures)} "
        f"AUTHORIZED={authorized_scenes} SAFE_ABSTAIN={completed-authorized_scenes} "
        f"SUM_GAIN={total_gain:.9f} PASSED={passed} "
        f"DECISION={result['decision']}"
    )
    for scene, row in rows.items():
        if row["implementation_authorized"]:
            print(
                f"{scene} gain={row['physical_macro_gain']:.9f} "
                f"rollback={','.join(row['rollback_object_ids'])}"
            )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
