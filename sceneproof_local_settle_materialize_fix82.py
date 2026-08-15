#!/usr/bin/env python3
"""Materialize one locally promising settle pose as a temporary candidate."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from sceneproof_local_settle_oracle_fix80 import classify_probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--allow-measured-candidate",
        action="store_true",
        help="materialize a measured audit candidate; full gates remain mandatory",
    )
    args = parser.parse_args()

    incumbent = json.loads(args.incumbent.read_text(encoding="utf-8"))
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    outcome = classify_probe(probe)
    measured_candidate = bool(
        args.allow_measured_candidate
        and probe.get("status") == "measured"
        and probe.get("incumbent_restored") is True
        and not probe.get("new_collision_object_ids")
    )
    if outcome != "locally_promising_requires_full_component_gates" and not measured_candidate:
        raise RuntimeError(f"probe is not locally promising: {outcome}")
    object_id = probe["object_id"]
    info = incumbent.get("obj_info", {}).get(object_id)
    if not isinstance(info, dict):
        raise RuntimeError(f"incumbent is missing object {object_id}")
    settled = probe.get("settled_pose_matrix")
    if not isinstance(settled, list) or len(settled) != 4:
        raise RuntimeError("probe has no valid settled pose")

    candidate = copy.deepcopy(incumbent)
    candidate["obj_info"][object_id]["pose_matrix_for_blender"] = settled
    candidate["sceneproof_local_settle_candidate"] = {
        "schema_version": "sceneproof_local_settle_candidate_v1",
        "policy": (
            "measured_probe_temporary_candidate_requires_full_gates"
            if measured_candidate
            else "single_object_temporary_candidate_requires_full_gates"
        ),
        "incumbent": str(args.incumbent.resolve()),
        "probe": str(args.probe.resolve()),
        "object_id": object_id,
        "full_so3": True,
        "pose_changes": 1,
        "promoted": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(candidate, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out.resolve()}")
    print(f"OBJECT={object_id} OUTCOME={outcome} PROMOTED=False")


if __name__ == "__main__":
    main()
