#!/usr/bin/env python3
"""Route witnessed Fix76 support states to COM projection or settle probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sceneproof_pillow_pose_lineage_fix75 import matrix_metrics


def classify_support_action(row: dict) -> str:
    certificate = row.get("certificate_status", "certified")
    stability = row.get("stability_class")
    parent_contact = bool(row.get("declared_parent_contact_present"))
    reason = row.get("reason")
    if certificate == "certified" and stability == "unstable" and parent_contact:
        tolerance = float(row.get("stability_tolerance_m", 0.005))
        intrinsic = row.get("intrinsic_child_contact_margin_m")
        parent_margin = row.get("declared_parent_surface_margin_m")
        if intrinsic is not None and float(intrinsic) < -tolerance:
            return "local_gravity_settle_probe_candidate"
        if (
            intrinsic is not None
            and float(intrinsic) >= -tolerance
            and parent_margin is not None
            and float(parent_margin) < -tolerance
        ):
            return "com_projection_candidate"
        return "local_gravity_settle_probe_candidate"
    if certificate == "certified" and stability in {"stable", "marginal"}:
        return "hold_certified_support"
    if reason == "no_mesh_or_voxel_horizontal_contact_patch":
        return "local_gravity_settle_probe_candidate"
    if reason == "cyclic_support_component_unproven":
        return "abstain_cyclic_support"
    return "abstain_unproven_support"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline_objects = baseline.get("objects", {})
    candidate_objects = candidate.get("objects", {})
    rows = []
    for object_id, row in sorted(candidate_objects.items()):
        action = classify_support_action(row)
        first = baseline_objects.get(object_id, {})
        pose_delta = None
        if first.get("pose_matrix_for_blender") and row.get("pose_matrix_for_blender"):
            pose_delta = matrix_metrics(
                first["pose_matrix_for_blender"], row["pose_matrix_for_blender"]
            )
        rows.append(
            {
                "object_id": object_id,
                "declared_parent_id": row.get("declared_parent_id"),
                "certificate_status": row.get("certificate_status"),
                "stability_class": row.get("stability_class"),
                "com_signed_margin_m": row.get("com_signed_margin_m"),
                "intrinsic_child_contact_margin_m": row.get(
                    "intrinsic_child_contact_margin_m"
                ),
                "declared_parent_surface_margin_m": row.get(
                    "declared_parent_surface_margin_m"
                ),
                "declared_parent_contact_present": row.get(
                    "declared_parent_contact_present"
                ),
                "reason": row.get("reason"),
                "action": action,
                "pose_delta_from_original_fix43": pose_delta,
            }
        )
    action_counts = {}
    for row in rows:
        action_counts[row["action"]] = action_counts.get(row["action"], 0) + 1
    actionable = [
        row
        for row in rows
        if row["action"]
        in {"com_projection_candidate", "local_gravity_settle_probe_candidate"}
    ]
    report = {
        "schema_version": "sceneproof_com_action_audit_v1",
        "policy": "audit_only_no_pose_mutation",
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "action_counts": action_counts,
        "actionable_objects": actionable,
        "all_support_objects": rows,
        "decision": (
            "implement_scoped_com_projection_and_settle_smoke1"
            if actionable
            else "hold_fix76_no_actionable_support_witness"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out.resolve()}")
    print(
        f"ACTIONS={json.dumps(action_counts, sort_keys=True)} "
        f"ACTIONABLE={len(actionable)} DECISION={report['decision']}"
    )
    for row in actionable:
        delta = row["pose_delta_from_original_fix43"] or {}
        print(
            f"OBJECT={row['object_id']} PARENT={row['declared_parent_id']} "
            f"ACTION={row['action']} STABILITY={row['stability_class']} "
            f"MARGIN={row['com_signed_margin_m']} "
            f"INTRINSIC={row['intrinsic_child_contact_margin_m']} "
            f"PARENT_MARGIN={row['declared_parent_surface_margin_m']} "
            f"POSE_DELTA={delta.get('translation_norm_m')}"
        )
    print(f"FIX78_ACTION_AUDIT={args.out.resolve()}")


if __name__ == "__main__":
    main()
