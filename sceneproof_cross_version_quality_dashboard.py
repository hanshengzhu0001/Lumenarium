#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--physical-dir", type=Path, required=True)
    parser.add_argument("--versions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-txt", type=Path, required=True)
    parser.add_argument("--runtime-json", type=Path, action="append", default=[])
    args = parser.parse_args()
    versions = args.versions.split(",")
    labels = args.labels.split(",")
    if len(versions) != len(labels):
        raise SystemExit("versions/labels length mismatch")
    gt = load(args.gt)["versions"]
    runtimes = {}
    for path in args.runtime_json:
        record = load(path)
        runtimes[record["version"]] = record
    rows = []
    for version, label in zip(versions, labels):
        physical_doc = load(args.physical_dir / f"{version}.json")
        physical = physical_doc["versions"][version]["aggregate"]
        pose = gt[version]
        primary = pose.get("primary_metrics", {})
        families = physical.get("families", {})
        row = {
            "label": label, "version": version,
            "matched_objects": pose.get("matched_object_count"),
            "gt_objects": pose.get("gt_object_count"),
            "recovery": pose.get("object_recovery"),
            "parent_accuracy": pose.get("scene_graph_parent_accuracy_gt"),
            "rotation_auc60": pose.get("rotation_auc60_aligned"),
            "translation_auc05": pose.get("translation_auc05_aligned"),
            "primary_recovery": primary.get("object_recovery"),
            "primary_parent_accuracy": primary.get("parent_accuracy"),
            "primary_rotation_auc60": primary.get("rotation_auc60_aligned"),
            "primary_translation_auc05": primary.get("translation_auc05_aligned"),
            "physical_macro": physical.get("headline_macro_realizability"),
            "physical_critical": physical.get("headline_critical_realizability"),
            "collision": families.get("collision", {}).get("score"),
            "support": families.get("support", {}).get("score"),
            "plane": families.get("plane", {}).get("score"),
            "semantic": families.get("semantic", {}).get("score"),
            "collision_pairs": physical.get("unintended_collision_pairs"),
            "end_to_end_mean_s": runtimes.get(version, {}).get("mean_scene_seconds"),
            "two_a10_wall_s": runtimes.get(version, {}).get("wall_clock_seconds"),
        }
        rows.append(row)
    report = {
        "schema_version": "sceneproof_cross_version_quality_v1",
        "pose_protocol": "Paper30, visible mask >=8000px, aligned pose",
        "physical_protocol": "version-native frozen geometry; common legacy collision policy",
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with args.out_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    # The Paper30 headline protocol is the >=8000px Primary subset. Keep the
    # overall metrics in JSON/CSV for diagnostics, but print Primary in the
    # human-facing table to avoid silently mixing evaluation populations.
    columns = ("label", "primary_recovery", "primary_parent_accuracy",
               "primary_rotation_auc60", "primary_translation_auc05",
               "physical_macro", "collision", "support",
               "plane", "semantic", "collision_pairs", "end_to_end_mean_s",
               "two_a10_wall_s")
    lines = ["SCENEPROOF PAPER30 CROSS-VERSION QUALITY", "=" * 100,
             " ".join(f"{name:>16}" for name in columns)]
    for row in rows:
        lines.append(" ".join(
            f"{(row[name] if row[name] is not None else 'n/a'):>16.6f}"
            if isinstance(row[name], float) else f"{str(row[name]):>16}"
            for name in columns
        ))
    args.out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_txt.read_text(encoding="utf-8"))
    print(f"CROSS_VERSION_JSON={args.out_json.resolve()}")


if __name__ == "__main__":
    main()
