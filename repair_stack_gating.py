#!/usr/bin/env python3
"""Repair stale v3 S3 stack annotations with the current conservative gate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from modules._s3_legacy_functions import detect_stacking_pairs
from modules.pose import _should_apply_stacking_pair


GRAPH_FIELDS = (
    "supported",
    "isOnFloor",
    "isAgainstWall",
    "isHangingFromCeiling",
    "isHangingOnWall",
    "againstWall",
    "directlyFacing",
    "group",
)


def read_scenes(value: str) -> list[str]:
    path = Path(value)
    if path.exists():
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def copy_cached_stages(src_scene_dir: Path, dst_scene_dir: Path, copy_mode: str) -> None:
    if dst_scene_dir.exists():
        shutil.rmtree(dst_scene_dir)
    dst_scene_dir.mkdir(parents=True, exist_ok=True)
    copy_function = os.link if copy_mode == "hardlink" else shutil.copy2
    for stage in (
        "S0_geometry_pred_results",
        "S1_scene_parsing_results",
        "S2_3d_retrieval_results",
        "S3_pose_inference",
        "stage_logs",
    ):
        src = src_scene_dir / stage
        if src.exists():
            shutil.copytree(src, dst_scene_dir / stage, copy_function=copy_function)
    pipeline_log = src_scene_dir / "pipeline.log"
    if pipeline_log.exists():
        copy_function(pipeline_log, dst_scene_dir / "pipeline.log")
    (dst_scene_dir / "S4_layout_refinement").mkdir(exist_ok=True)


def repair_scene(dst_root: Path, scene: str, variant: str) -> dict:
    scene_dir = dst_root / f"{scene}_{variant}_result"
    s1_dir = scene_dir / "S1_scene_parsing_results"
    s3_path = scene_dir / "S3_pose_inference" / f"{scene}_{variant}_placement_info.json"
    graph_path = s1_dir / "scene_graph_result_final.json"

    scene_graph = load_json(graph_path)
    placement = load_json(s3_path)
    obj_info = placement.get("obj_info", {})

    previous_pairs = placement.get("stacking_pairs", [])

    for obj_name, info in obj_info.items():
        if not isinstance(info, dict):
            continue
        info.pop("stacked_on", None)
        graph_props = scene_graph.get(obj_name)
        if isinstance(graph_props, dict):
            for field in GRAPH_FIELDS:
                if field in graph_props:
                    info[field] = graph_props[field]

    raw_pairs = detect_stacking_pairs(str(s1_dir), scene_graph)
    accepted_pairs = []
    rejected_pairs = []
    for lower, upper in raw_pairs:
        if upper not in obj_info:
            rejected_pairs.append([lower, upper, "missing_upper_placement"])
            continue
        should_apply, reason = _should_apply_stacking_pair(lower, upper, scene_graph, placement)
        if not should_apply:
            rejected_pairs.append([lower, upper, reason])
            continue
        accepted_pairs.append([lower, upper])
        obj_info[upper]["stacked_on"] = lower
        obj_info[upper]["supported"] = lower

    placement["stacking_pairs_raw"] = [[lower, upper] for lower, upper in raw_pairs]
    placement["stacking_pairs"] = accepted_pairs
    placement["stacking_pairs_rejected"] = rejected_pairs
    placement.setdefault("_repair_metadata", {})
    placement["_repair_metadata"]["stack_gate_repaired"] = True
    placement["_repair_metadata"]["previous_stacking_pair_count"] = len(previous_pairs)
    placement["_repair_metadata"]["current_stacking_pair_count"] = len(accepted_pairs)
    placement["_repair_metadata"]["rejected_stacking_pair_count"] = len(rejected_pairs)

    backup = s3_path.with_suffix(".pre_stackgate_repair.json")
    if not backup.exists():
        shutil.copy2(s3_path, backup)
    write_json(s3_path, placement)

    return {
        "scene": scene,
        "previous": len(previous_pairs),
        "raw": len(raw_pairs),
        "accepted": len(accepted_pairs),
        "rejected": len(rejected_pairs),
        "accepted_pairs": accepted_pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--dest-root", required=True)
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--variant", default="v3")
    parser.add_argument("--report-out", default="")
    parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="hardlink")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    dest_root = Path(args.dest_root)
    scenes = read_scenes(args.scenes)
    report = []

    for scene in scenes:
        src_scene_dir = source_root / f"{scene}_{args.variant}_result"
        dst_scene_dir = dest_root / f"{scene}_{args.variant}_result"
        copy_cached_stages(src_scene_dir, dst_scene_dir, args.copy_mode)
        report.append(repair_scene(dest_root, scene, args.variant))

    if args.report_out:
        write_json(Path(args.report_out), report)
    for row in report:
        print(
            row["scene"],
            f"previous={row['previous']}",
            f"raw={row['raw']}",
            f"accepted={row['accepted']}",
            f"rejected={row['rejected']}",
            f"pairs={row['accepted_pairs']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
