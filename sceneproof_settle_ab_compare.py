#!/usr/bin/env python3
"""比较沉降开关两侧的 placement，独立于 S4 自己打印的日志。

为什么不只读日志
----------------
S4 会为每次位移打印一行 [SETTLE]，但那是决策时刻的意图。位移之后还要经过靠墙平移、
组内scale 统一、刚体落地仿真与位姿序列化，任何一步都可能把 z 再改一次。所以这里从
两份写盘的 placement 反算实际 z 差：日志说做了什么，这个脚本说最终留下了什么。两者
不一致本身就是要报告的结论。

z 取自 pose_matrix_for_blender 的平移分量，与 S4 写盘时的定义一致。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


CAMERA = re.compile(r"camera", re.IGNORECASE)
STRUCTURAL = re.compile(r"^(floor|ground|wall|ceiling)_\d+$", re.IGNORECASE)


def translation_z(info: dict) -> float | None:
    matrix = info.get("pose_matrix_for_blender")
    if not isinstance(matrix, list) or len(matrix) < 3:
        return None
    row = matrix[2]
    if not isinstance(row, list) or len(row) < 4:
        return None
    try:
        return float(row[3])
    except (TypeError, ValueError):
        return None


def load_z(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    heights: dict[str, float] = {}
    for object_id, info in (document.get("obj_info") or {}).items():
        if not isinstance(info, dict) or CAMERA.search(object_id):
            continue
        if STRUCTURAL.match(object_id):
            continue
        value = translation_z(info)
        if value is not None:
            heights[object_id] = value
    return heights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--settle-off", type=Path, required=True)
    parser.add_argument("--settle-on", type=Path, required=True)
    parser.add_argument("--settle-log", type=Path)
    parser.add_argument("--moved-threshold-m", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()

    before = load_z(args.settle_off)
    after = load_z(args.settle_on)
    shared = sorted(set(before) & set(after))

    moved = []
    for object_id in shared:
        delta = after[object_id] - before[object_id]
        if abs(delta) >= args.moved_threshold_m:
            moved.append(
                {
                    "object_id": object_id,
                    "z_with_settle_off_m": before[object_id],
                    "z_with_settle_on_m": after[object_id],
                    "delta_z_m": delta,
                }
            )
    moved.sort(key=lambda item: -abs(item["delta_z_m"]))

    reasons: dict[str, int] = {}
    if args.settle_log and args.settle_log.exists():
        pattern = re.compile(r"^\[SETTLE\] (\S+) on (\S+): (\w+) ")
        for line in args.settle_log.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            match = pattern.match(line.strip())
            if match:
                reasons[match.group(3)] = reasons.get(match.group(3), 0) + 1

    lowered = [item for item in moved if item["delta_z_m"] < 0]
    raised = [item for item in moved if item["delta_z_m"] > 0]
    report = {
        "schema_version": "sceneproof_settle_ab_v1",
        "scene": args.scene,
        "object_count": len(shared),
        "objects_only_in_one_side": sorted(set(before) ^ set(after)),
        "moved_count": len(moved),
        "moved_share_of_scene": len(moved) / len(shared) if shared else 0.0,
        "lowered_count": len(lowered),
        "raised_count": len(raised),
        "largest_drop_m": min((item["delta_z_m"] for item in moved), default=0.0),
        "largest_lift_m": max((item["delta_z_m"] for item in moved), default=0.0),
        "settle_log_reason_counts": dict(sorted(reasons.items())),
        "moved": moved[: max(args.top_k, 0)],
        "policy": {
            "z_is_read_from_the_written_placement_not_from_the_decision_log": True,
            "a_mismatch_between_log_and_placement_is_itself_a_finding": True,
        },
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"SETTLE A/B objects={len(shared)} moved={len(moved)}"
          f" ({report['moved_share_of_scene']:.0%})")
    print(f"  lowered={len(lowered)} raised={len(raised)}"
          f" largest_drop={report['largest_drop_m']:+.4f}m"
          f" largest_lift={report['largest_lift_m']:+.4f}m")
    if reasons:
        print("  decisions S4 logged:")
        for name, count in report["settle_log_reason_counts"].items():
            print(f"    {name}={count}")
    if report["objects_only_in_one_side"]:
        print("  WARNING objects present on only one side:"
              f" {report['objects_only_in_one_side']}")
    for item in report["moved"]:
        print(
            "    {}: {:+.4f}m ({:.4f} -> {:.4f})".format(
                item["object_id"],
                item["delta_z_m"],
                item["z_with_settle_off_m"],
                item["z_with_settle_on_m"],
            )
        )
    print(f"Wrote {args.out_report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
