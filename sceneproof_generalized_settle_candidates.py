#!/usr/bin/env python3
"""Discover generalized SceneProof settle candidates from cached true-mesh audits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


STRUCTURAL = re.compile(r"^(floor|ground|wall|ceiling|carpet|rug)_\d+$")
ATTACHED_PARENT = re.compile(r"^(wall|ceiling)_\d+$")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def support_id(value: Any) -> str | None:
    if isinstance(value, (list, tuple)):
        value = next((item for item in value if item), None)
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def valid_pose(value: Any) -> bool:
    try:
        return (
            isinstance(value, list)
            and len(value) == 4
            and all(isinstance(row, list) and len(row) == 4 for row in value)
            and all(float(item) == float(item) for row in value for item in row)
        )
    except (TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-version", required=True)
    parser.add_argument("--true-mesh-audit-root", type=Path, required=True)
    parser.add_argument("--physical-objects", type=Path, required=True)
    parser.add_argument("--minimum-gap-m", type=float, default=0.005)
    parser.add_argument("--maximum-gap-m", type=float, default=0.5)
    parser.add_argument("--max-representatives", type=int, default=30)
    parser.add_argument("--all-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    scenes = [
        line.strip()
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    physical_rows: dict[tuple[str, str], dict[str, str]] = {}
    with args.physical_objects.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("version") != args.baseline_version:
                continue
            scene = row.get("scene")
            object_id = row.get("object_id")
            if scene and object_id:
                physical_rows[(scene, object_id)] = row

    for scene in scenes:
        placement_path = (
            args.saved_results
            / f"{scene}_{args.baseline_version}_result"
            / "S4_layout_refinement"
            / f"{scene}_{args.baseline_version}_placement_info_s4.json"
        )
        audit_path = (
            args.true_mesh_audit_root
            / f"{scene}__{args.baseline_version}.json"
        )
        if not placement_path.is_file() or not audit_path.is_file():
            failures.append(
                {
                    "scene": scene,
                    "reason": "missing_input",
                    "placement": str(placement_path),
                    "audit": str(audit_path),
                }
            )
            continue

        placement = load(placement_path)
        audit = load(audit_path)
        info = placement.get("obj_info", {})
        records = audit.get("objects", {})
        declared_children: dict[str, list[str]] = {}
        for object_id, object_info in info.items():
            if not isinstance(object_info, dict):
                continue
            parent = support_id(object_info.get("supported"))
            if parent:
                declared_children.setdefault(parent, []).append(object_id)

        for object_id, object_info in sorted(info.items()):
            reasons: list[str] = []
            if not isinstance(object_info, dict):
                continue
            parent = support_id(object_info.get("supported"))
            relation = str(object_info.get("SpatialRel", "")).strip().lower()
            record = records.get(object_id, {})
            if not isinstance(record, dict) or not record:
                record = {}
                reasons.append("missing_true_mesh_audit_record")
            margin = record.get("com_signed_margin_m")
            try:
                margin = float(margin) if margin is not None else None
            except (TypeError, ValueError):
                margin = None
            no_contact = bool(
                record.get("reason")
                in {
                    "no_exact_horizontal_contact_patch",
                    "no_mesh_or_voxel_horizontal_contact_patch",
                }
                or not record.get("declared_parent_contact_present", False)
            )
            com_unstable = bool(margin is not None and margin < 0.0)
            physical = physical_rows.get((scene, object_id))
            gap = None
            if physical is not None:
                try:
                    gap = float(physical.get("support_contact_gap_m", ""))
                    if not math.isfinite(gap):
                        gap = None
                except (TypeError, ValueError):
                    gap = None
            actionable_gap = bool(
                no_contact
                and gap is not None
                and args.minimum_gap_m < gap <= args.maximum_gap_m
            )

            if STRUCTURAL.match(object_id):
                reasons.append("structural")
            if not parent or parent not in info:
                reasons.append("missing_declared_parent")
            if relation == "inside":
                reasons.append("held_by_containment")
            if parent and ATTACHED_PARENT.match(parent):
                reasons.append("wall_or_ceiling_attached")
            if declared_children.get(object_id):
                reasons.append("is_support_parent")
            cycle = record.get("cyclic_support_component")
            if isinstance(cycle, list) and cycle:
                reasons.append("cyclic_support_component")
            if not valid_pose(object_info.get("pose_matrix_for_blender")):
                reasons.append("invalid_or_missing_pose")
            if no_contact and not com_unstable and not actionable_gap:
                reasons.append("no_actionable_finite_downward_gap")
            if not (com_unstable or actionable_gap):
                reasons.append("no_actionable_com_or_contact_witness")

            row = {
                "scene": scene,
                "object_id": object_id,
                "declared_parent_id": parent,
                "com_signed_margin_m": margin,
                "com_unstable": com_unstable,
                "no_declared_parent_contact": no_contact,
                "support_contact_gap_m": gap,
                "actionable_downward_gap": actionable_gap,
                "audit_status": record.get("status"),
                "audit_reason": record.get("reason"),
                "asset_key": str(
                    object_info.get("retrieved_asset")
                    or object_info.get("fbx_name")
                    or object_info.get("best_match_vid")
                    or f"{scene}:{object_id}"
                ),
            }
            if reasons:
                row["exclusion_reasons"] = sorted(set(reasons))
                excluded.append(row)
            else:
                row["witness"] = (
                    "negative_com_margin"
                    if com_unstable
                    else "missing_exact_contact_with_finite_downward_gap"
                )
                eligible.append(row)

    def severity(row: dict[str, Any]) -> tuple[float, float]:
        margin = row.get("com_signed_margin_m")
        negative = max(0.0, -float(margin)) if margin is not None else 0.0
        gap = row.get("support_contact_gap_m")
        return negative, float(gap) if gap is not None else 0.0

    representative_by_asset: dict[str, dict[str, Any]] = {}
    for row in eligible:
        key = row["asset_key"]
        incumbent = representative_by_asset.get(key)
        if incumbent is None or severity(row) > severity(incumbent):
            representative_by_asset[key] = row
    representatives = sorted(
        representative_by_asset.values(), key=severity, reverse=True
    )[: max(0, args.max_representatives)]

    result = {
        "schema_version": "sceneproof_generalized_settle_candidates_v1",
        "baseline_version": args.baseline_version,
        "scenes_expected": len(scenes),
        "scenes_completed": len(scenes) - len(failures),
        "failures": failures,
        "eligible": eligible,
        "representatives": representatives,
        "excluded": excluded,
        "summary": {
            "eligible": len(eligible),
            "unique_assets": len(representative_by_asset),
            "representatives": len(representatives),
            "excluded": len(excluded),
            "failures": len(failures),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.all_report is not None:
        args.all_report.parent.mkdir(parents=True, exist_ok=True)
        all_lines = ["scene\tobject_id\tdeclared_parent_id\twitness\tasset_key"]
        all_lines.extend(
            f"{row['scene']}\t{row['object_id']}\t{row['declared_parent_id']}\t{row['witness']}\t{row['asset_key']}"
            for row in eligible
        )
        args.all_report.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    lines = ["scene\tobject_id\tdeclared_parent_id\twitness\tasset_key"]
    lines.extend(
        f"{row['scene']}\t{row['object_id']}\t{row['declared_parent_id']}\t{row['witness']}\t{row['asset_key']}"
        for row in representatives
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out.resolve()}")
    print(f"Wrote {args.report.resolve()}")
    print(
        f"SCENES={result['scenes_completed']}/{len(scenes)} "
        f"ELIGIBLE={len(eligible)} UNIQUE_ASSETS={len(representative_by_asset)} "
        f"REPRESENTATIVES={len(representatives)} EXCLUDED={len(excluded)} "
        f"FAILURES={len(failures)}"
    )


if __name__ == "__main__":
    main()
