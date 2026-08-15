#!/usr/bin/env python3
"""Routing, composition, and reporting for the rigid-only Fix84 re-evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def probe_route(probe: dict[str, Any], stage: str) -> str:
    if probe.get("status") != "measured" or not probe.get("incumbent_restored"):
        return "reject"
    if probe.get("new_collision_object_ids"):
        return "reject"
    after = probe.get("after_support") or {}
    promising = (
        after.get("certificate_status") == "certified"
        and after.get("stability_class") in {"stable", "marginal"}
        and after.get("declared_parent_contact_present")
    )
    if promising:
        return "select"
    if after.get("stability_class") != "unstable":
        return "reject"
    if stage == "primary":
        return "retry_damping"
    if stage == "damping":
        return "retry_friction"
    return "reject"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compose(args: argparse.Namespace) -> None:
    selected_rows = read_rows(args.selected)
    selected: dict[str, list[dict[str, str]]] = {}
    for row in selected_rows:
        if row.get("accepted") == "true":
            selected.setdefault(row["scene"], []).append(row)
    scenes = [
        line.strip()
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    certificate: dict[str, Any] = {
        "schema_version": "sceneproof_rigid_only_adaptive_certificate_v1",
        "policy": "process_isolated_full_so3_adaptive_then_fix84_scoped_commit",
        "baseline_version": args.baseline_version,
        "target_version": args.target_version,
        "target_manifest": str(args.target_manifest.resolve()),
        "scenes": {},
        "failures": [],
    }
    args.target_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.target_manifest.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    for scene in scenes:
        source = (
            args.saved_results
            / f"{scene}_{args.baseline_version}_result"
            / "S4_layout_refinement"
            / f"{scene}_{args.baseline_version}_placement_info_s4.json"
        )
        if not source.is_file():
            certificate["failures"].append(
                {"scene": scene, "reason": "missing_baseline", "path": str(source)}
            )
            continue
        document = load(source)
        retained: list[str] = []
        for row in selected.get(scene, []):
            object_id = row["object_id"]
            probe = load(Path(row["probe"]))
            settled = probe.get("settled_pose_matrix")
            info = document.get("obj_info", {}).get(object_id)
            if not isinstance(info, dict) or not isinstance(settled, list):
                certificate["failures"].append(
                    {"scene": scene, "object_id": object_id, "reason": "invalid_commit"}
                )
            else:
                info["pose_matrix_for_blender"] = settled
                retained.append(object_id)
        document["sceneproof_rigid_only_adaptive_commit"] = {
            "schema_version": "sceneproof_rigid_only_adaptive_commit_v1",
            "baseline_version": args.baseline_version,
            "target_version": args.target_version,
            "retained_object_ids": retained,
            "unchanged_baseline_copy": not retained,
        }
        output = (
            args.saved_results
            / f"{scene}_{args.target_version}_result"
            / "S4_layout_refinement"
            / f"{scene}_{args.target_version}_placement_info_s4.json"
        )
        write_json(output, document)
        certificate["scenes"][scene] = {
            "accepted": True,
            "fallback": False,
            "retained_changed_objects": retained,
            "placement": str(output.resolve()),
        }
    write_json(args.certificate, certificate)
    print(f"Wrote {args.certificate.resolve()}")
    print(
        f"SCENES={len(certificate['scenes'])}/{len(scenes)} "
        f"FAILURES={len(certificate['failures'])} "
        f"RETAINED={sum(len(v['retained_changed_objects']) for v in certificate['scenes'].values())}"
    )
    if certificate["failures"]:
        raise SystemExit(2)


def metric(document: dict[str, Any], version: str, key: str) -> float | None:
    value = document.get("versions", {}).get(version, {}).get(key)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def summarize(args: argparse.Namespace) -> None:
    rows = read_rows(args.selected)
    detailed_rows: list[dict[str, Any]] = []
    for row in rows:
        detail: dict[str, Any] = dict(row)
        probe_path = Path(row["probe"])
        if row.get("probe") != "-" and probe_path.is_file():
            probe = load(probe_path)
            before = probe.get("before_support") or {}
            after = probe.get("after_support") or {}
            parent_id = probe.get("declared_parent_id")
            before_gaps = before.get("contact_gap_by_supporter_m") or {}
            after_gaps = after.get("contact_gap_by_supporter_m") or {}
            detail.update(
                {
                    "translation_delta_m": probe.get("translation_delta_m"),
                    "rotation_delta_deg": probe.get("rotation_delta_deg"),
                    "com_margin_before_m": before.get("com_signed_margin_m"),
                    "com_margin_after_m": after.get("com_signed_margin_m"),
                    "stability_before": before.get("stability_class"),
                    "stability_after": after.get("stability_class"),
                    "contact_gap_before_m": before_gaps.get(parent_id),
                    "contact_gap_after_m": after_gaps.get(parent_id),
                    "new_collision_object_ids": probe.get("new_collision_object_ids", []),
                    "boundary_error_before_m": probe.get("before_boundary_error_m"),
                    "boundary_error_after_m": probe.get("after_boundary_error_m"),
                    "simulation_settings": probe.get("simulation_settings"),
                }
            )
        detailed_rows.append(detail)
    physical, gt, certificate = load(args.physical), load(args.gt), load(args.certificate)
    base = physical["versions"][args.baseline_version]["aggregate"]
    target = physical["versions"][args.target_version]["aggregate"]
    family_deltas: dict[str, float | None] = {}
    for family in ("collision", "support", "plane", "semantic"):
        before = base.get("families", {}).get(family, {}).get("score")
        after = target.get("families", {}).get(family, {}).get("score")
        family_deltas[family] = (
            float(after) - float(before)
            if isinstance(before, (int, float)) and isinstance(after, (int, float))
            else None
        )
    macro_delta = float(target["headline_macro_realizability"]) - float(
        base["headline_macro_realizability"]
    )
    r0 = metric(gt, args.baseline_version, "rotation_auc60_aligned")
    r1 = metric(gt, args.target_version, "rotation_auc60_aligned")
    t0 = metric(gt, args.baseline_version, "translation_auc05_aligned")
    t1 = metric(gt, args.target_version, "translation_auc05_aligned")
    report = {
        "schema_version": "sceneproof_rigid_only_adaptive_eval_v1",
        "baseline_version": args.baseline_version,
        "target_version": args.target_version,
        "retained_changed_objects": sum(
            len(row.get("retained_changed_objects", []))
            for row in certificate.get("scenes", {}).values()
        ),
        "per_object_trials": detailed_rows,
        "physical_family_deltas": family_deltas,
        "physical_macro_delta": macro_delta,
        "rotation_delta": None if r0 is None or r1 is None else r1 - r0,
        "translation_delta": None if t0 is None or t1 is None else t1 - t0,
        "failures": certificate.get("failures", []) + physical.get("failures", []) + gt.get("failures", []),
    }
    write_json(args.out, report)
    lines = [
        "SCENEPROOF RIGID-ONLY ADAPTIVE SETTLE EVALUATION",
        "=" * 72,
        f"BASELINE={args.baseline_version}",
        f"TARGET={args.target_version}",
        f"RETAINED_CHANGED_OBJECTS={report['retained_changed_objects']}",
        f"PHYSICAL_MACRO_DELTA={macro_delta:+.9f}",
        "FAMILY_DELTAS=" + json.dumps(family_deltas, sort_keys=True),
        f"ROTATION_DELTA={report['rotation_delta']}",
        f"TRANSLATION_DELTA={report['translation_delta']}",
        f"FAILURES={len(report['failures'])}",
    ]
    for row in detailed_rows:
        lines.append(
            f"{row['scene']} {row['object_id']} profile={row['profile']} "
            f"strict={row['strict_passed']} relaxed={row['relaxed_passed']} "
            f"accepted={row['accepted']} elapsed_s={row['elapsed_seconds']} "
            f"translation_m={row.get('translation_delta_m')} "
            f"rotation_deg={row.get('rotation_delta_deg')} "
            f"com_margin={row.get('com_margin_before_m')}->{row.get('com_margin_after_m')} "
            f"contact_gap={row.get('contact_gap_before_m')}->{row.get('contact_gap_after_m')} "
            f"new_collisions={row.get('new_collision_object_ids')} "
            f"boundary={row.get('boundary_error_before_m')}->{row.get('boundary_error_after_m')}"
        )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:]))
    print(f"RIGID_SETTLE_EVAL={args.out.resolve()}")
    print(f"RIGID_SETTLE_REPORT={args.report.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    route = sub.add_parser("route")
    route.add_argument("--probe", type=Path, required=True)
    route.add_argument("--stage", choices=("primary", "damping", "friction"), required=True)
    compose_parser = sub.add_parser("compose")
    compose_parser.add_argument("--saved-results", type=Path, required=True)
    compose_parser.add_argument("--manifest", type=Path, required=True)
    compose_parser.add_argument("--selected", type=Path, required=True)
    compose_parser.add_argument("--baseline-version", required=True)
    compose_parser.add_argument("--target-version", required=True)
    compose_parser.add_argument("--target-manifest", type=Path, required=True)
    compose_parser.add_argument("--certificate", type=Path, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--selected", type=Path, required=True)
    summary.add_argument("--physical", type=Path, required=True)
    summary.add_argument("--gt", type=Path, required=True)
    summary.add_argument("--certificate", type=Path, required=True)
    summary.add_argument("--baseline-version", required=True)
    summary.add_argument("--target-version", required=True)
    summary.add_argument("--out", type=Path, required=True)
    summary.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "route":
        print(probe_route(load(args.probe), args.stage))
    elif args.command == "compose":
        compose(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
