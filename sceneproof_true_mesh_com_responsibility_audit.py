#!/usr/bin/env python3
"""Compare smooth and certified true-mesh COM support witnesses on Paper30."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from modules._sceneproof_support_stability import physical_support_score


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def matrix_delta(first: Any, second: Any) -> float | None:
    try:
        values = [
            (float(a) - float(b)) ** 2
            for row_a, row_b in zip(first, second)
            for a, b in zip(row_a, row_b)
        ]
    except (TypeError, ValueError):
        return None
    return math.sqrt(sum(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--smooth-version", required=True)
    parser.add_argument("--final-version", required=True)
    parser.add_argument("--physical-objects", type=Path, required=True)
    parser.add_argument("--margin-regression-m", type=float, default=0.005)
    parser.add_argument("--localization-threshold", type=float, default=0.70)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    scenes = [
        line.strip()
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    physical_rows = {}
    with args.physical_objects.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            physical_rows[(row["scene"], row["object_id"], row["version"])] = row

    failures = []
    scene_records = {}
    comparisons = []
    total_negative_support = 0.0
    localized_negative_support = 0.0
    for scene in scenes:
        smooth_path = args.audit_root / f"{scene}__{args.smooth_version}.json"
        final_path = args.audit_root / f"{scene}__{args.final_version}.json"
        if not smooth_path.is_file() or not final_path.is_file():
            failures.append(
                {
                    "scene": scene,
                    "error": "missing_audit",
                    "smooth": str(smooth_path),
                    "final": str(final_path),
                }
            )
            continue
        smooth = load(smooth_path)
        final = load(final_path)
        if smooth.get("maximum_pose_delta") != 0.0 or final.get("maximum_pose_delta") != 0.0:
            failures.append({"scene": scene, "error": "audit_mutated_pose"})
            continue
        smooth_objects = smooth.get("objects", {})
        final_objects = final.get("objects", {})
        scene_rows = []
        for object_id in sorted(set(smooth_objects) & set(final_objects)):
            first = smooth_objects[object_id]
            second = final_objects[object_id]
            first_measured = (
                first.get("status") == "measured"
                and first.get("certificate_status", "certified")
                == "certified"
            )
            second_measured = (
                second.get("status") == "measured"
                and second.get("certificate_status", "certified")
                == "certified"
            )
            first_margin = first.get("com_signed_margin_m") if first_measured else None
            second_margin = second.get("com_signed_margin_m") if second_measured else None
            margin_delta = (
                float(second_margin) - float(first_margin)
                if first_margin is not None and second_margin is not None
                else None
            )
            pose_change = matrix_delta(
                first.get("pose_matrix_for_blender"),
                second.get("pose_matrix_for_blender"),
            )
            smooth_physical = physical_rows.get(
                (scene, object_id, args.smooth_version)
            )
            final_physical = physical_rows.get(
                (scene, object_id, args.final_version)
            )
            smooth_score = (
                physical_support_score(smooth_physical)
                if smooth_physical is not None
                else None
            )
            final_score = (
                physical_support_score(final_physical)
                if final_physical is not None
                else None
            )
            score_delta = (
                final_score - smooth_score
                if smooth_score is not None and final_score is not None
                else None
            )
            final_unstable = second.get("stability_class") == "unstable"
            introduced_unstable = (
                final_unstable
                and first.get("stability_class") != "unstable"
            )
            margin_regressed = bool(
                margin_delta is not None
                and margin_delta < -args.margin_regression_m
            )
            localized = bool(final_unstable or margin_regressed)
            if score_delta is not None and score_delta < 0:
                loss = -score_delta
                total_negative_support += loss
                if localized:
                    localized_negative_support += loss
            row = {
                "scene": scene,
                "object_id": object_id,
                "declared_parent_id": second.get("declared_parent_id"),
                "support_program_ids": second.get("support_program_ids", []),
                "pose_matrix_delta": pose_change,
                "smooth_status": first.get("status"),
                "final_status": second.get("status"),
                "smooth_certificate_status": first.get(
                    "certificate_status"
                ),
                "final_certificate_status": second.get(
                    "certificate_status"
                ),
                "paired_certified": first_measured and second_measured,
                "smooth_stability": first.get("stability_class"),
                "final_stability": second.get("stability_class"),
                "smooth_margin_m": first_margin,
                "final_margin_m": second_margin,
                "margin_delta_m": margin_delta,
                "support_score_delta": score_delta,
                "introduced_unstable": introduced_unstable,
                "margin_regressed": margin_regressed,
                "responsibility_localized": localized,
                "final_supporter_ids": second.get("supporter_ids", []),
                "final_reason": second.get("reason"),
            }
            scene_rows.append(row)
            comparisons.append(row)
        scene_records[scene] = {
            "smooth_summary": smooth.get("summary", {}),
            "final_summary": final.get("summary", {}),
            "introduced_unstable": sum(row["introduced_unstable"] for row in scene_rows),
            "margin_regressions": sum(row["margin_regressed"] for row in scene_rows),
        }

    comparisons.sort(
        key=lambda row: (
            row["final_margin_m"] is None,
            row["final_margin_m"] if row["final_margin_m"] is not None else 0.0,
        )
    )
    measured_pairs = sum(
        row["paired_certified"]
        for row in comparisons
    )
    introduced = [row for row in comparisons if row["introduced_unstable"]]
    regressed = [row for row in comparisons if row["margin_regressed"]]
    localization_fraction = (
        localized_negative_support / total_negative_support
        if total_negative_support > 0
        else None
    )
    result = {
        "schema_version": "sceneproof_true_mesh_com_responsibility_audit_v1",
        "smooth_version": args.smooth_version,
        "final_version": args.final_version,
        "scenes_expected": len(scenes),
        "scenes_completed": len(scene_records),
        "failures": failures,
        "measurement": {
            "paired_support_objects": len(comparisons),
            "paired_measured": measured_pairs,
            "introduced_unstable": len(introduced),
            "margin_regressions": len(regressed),
        },
        "responsibility": {
            "total_negative_support_score": total_negative_support,
            "localized_negative_support_score": localized_negative_support,
            "localized_fraction": localization_fraction,
            "required_fraction": args.localization_threshold,
            "localization_gate": bool(
                localization_fraction is not None
                and localization_fraction >= args.localization_threshold
            ),
        },
        "implementation_authorized": False,
        "implementation_blocker": (
            "local counterfactual rollback/COM-projection oracle has not yet "
            "proved positive macro gain"
        ),
        "decision": (
            "measure_local_counterfactual_oracle"
            if not failures and measured_pairs
            else "audit_incomplete"
        ),
        "scenes": scene_records,
        "introduced_unstable_objects": introduced,
        "worst_margin_regressions": regressed[:50],
        "all_comparisons": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "SCENEPROOF TRUE-MESH COM RESPONSIBILITY AUDIT",
        "=" * 72,
        f"SCENES={len(scene_records)}/{len(scenes)} FAILURES={len(failures)}",
        f"PAIRED/MEASURED={len(comparisons)}/{measured_pairs}",
        f"INTRODUCED_UNSTABLE={len(introduced)} MARGIN_REGRESSIONS={len(regressed)}",
        "LOCALIZED_NEGATIVE_SUPPORT="
        + (
            "n/a"
            if localization_fraction is None
            else f"{localization_fraction:.3f}"
        ),
        f"DECISION={result['decision']}",
    ]
    for row in comparisons[:30]:
        if not (row["introduced_unstable"] or row["margin_regressed"]):
            continue
        lines.append(
            f"{row['scene']} {row['object_id']} parent={row['declared_parent_id']} "
            f"margin={row['smooth_margin_m']}->{row['final_margin_m']} "
            f"support_delta={row['support_score_delta']} "
            f"introduced={row['introduced_unstable']}"
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.report}")
    print(" ".join(lines[2:7]))


if __name__ == "__main__":
    main()
