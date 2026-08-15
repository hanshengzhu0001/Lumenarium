#!/usr/bin/env python3
"""Materialize only Paper30 rollback sets authorized by the COM oracle."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

from sceneproof_postsim_component_certifier import (
    changed_objects,
    rollback_poses,
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def placement(root: Path, scene: str, version: str) -> Path:
    return (
        root
        / f"{scene}_{version}_result"
        / "S4_layout_refinement"
        / f"{scene}_{version}_placement_info_s4.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--incumbent-version", required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--target-version", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime-jsonl", type=Path, required=True)
    args = parser.parse_args()

    protocol = load(args.protocol)
    if not protocol.get("passed") or not protocol.get(
        "materialization_authorized"
    ):
        raise SystemExit("counterfactual protocol did not authorize materialization")
    scenes = [
        row.strip()
        for row in args.manifest.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    report = {
        "schema_version": "sceneproof_com_scoped_rollback_materialization_v1",
        "incumbent_version": args.incumbent_version,
        "candidate_version": args.candidate_version,
        "target_version": args.target_version,
        "protocol": str(args.protocol.resolve()),
        "scenes": {},
        "failures": [],
    }
    runtime_rows = []
    for scene in scenes:
        started = time.perf_counter()
        try:
            incumbent = load(
                placement(args.saved_results, scene, args.incumbent_version)
            )
            candidate = load(
                placement(args.saved_results, scene, args.candidate_version)
            )
            selected = copy.deepcopy(candidate)
            initial_changed = changed_objects(incumbent, candidate)
            protocol_row = protocol["scenes"][scene]
            rollback = set(protocol_row.get("rollback_object_ids", []))
            if not rollback.issubset(initial_changed):
                raise ValueError("oracle rollback contains an unchanged object")
            rollback_poses(selected, incumbent, rollback)
            remaining = changed_objects(incumbent, selected)
            expected_remaining = initial_changed - rollback
            if remaining != expected_remaining:
                raise ValueError("materialized pose ownership mismatch")
            certificate = {
                "accepted": True,
                "full_incumbent_fallback": False,
                "oracle_authorized": bool(
                    protocol_row.get("implementation_authorized")
                ),
                "initial_changed_objects": sorted(initial_changed),
                "rolled_back_objects": sorted(rollback),
                "retained_changed_objects": sorted(remaining),
                "physical_macro_oracle_gain": protocol_row.get(
                    "physical_macro_gain", 0.0
                ),
            }
            selected["sceneproof_com_scoped_rollback_certificate"] = certificate
            output = placement(args.saved_results, scene, args.target_version)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
            elapsed = time.perf_counter() - started
            report["scenes"][scene] = {**certificate, "output": str(output)}
            runtime_rows.append(
                {
                    "scene": scene,
                    "version": f"{args.target_version}_materialization",
                    "engine": "sceneproof_com_certificate",
                    "stage": "cached_scoped_rollback",
                    "gpu": None,
                    "elapsed_seconds": elapsed,
                    "status": "ok",
                    "return_code": 0,
                }
            )
            print(
                f"{scene} rollback={len(rollback)} retained={len(remaining)} "
                f"authorized={certificate['oracle_authorized']}"
            )
        except Exception as error:
            report["failures"].append({"scene": scene, "error": repr(error)})
            print(f"FAIL scene={scene} error={error!r}")

    report["completed"] = len(report["scenes"])
    report["authorized_scenes"] = sum(
        row["oracle_authorized"] for row in report["scenes"].values()
    )
    report["rolled_back_objects"] = sum(
        len(row["rolled_back_objects"]) for row in report["scenes"].values()
    )
    report["retained_changed_objects"] = sum(
        len(row["retained_changed_objects"])
        for row in report["scenes"].values()
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.runtime_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in runtime_rows),
        encoding="utf-8",
    )
    print(f"Wrote {args.out.resolve()}")
    print(
        f"SCENES={report['completed']}/{len(scenes)} "
        f"FAILURES={len(report['failures'])} "
        f"AUTHORIZED={report['authorized_scenes']} "
        f"ROLLED_BACK={report['rolled_back_objects']} "
        f"RETAINED={report['retained_changed_objects']}"
    )
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
