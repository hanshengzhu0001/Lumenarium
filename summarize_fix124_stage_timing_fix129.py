#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime
from pathlib import Path


STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+-")
STAGES = {
    "s0": "S0_geometry.log",
    "s1": "S1_parsing.log",
    "s2": "S2_retrieval.log",
    "s3": "S3_pose.log",
}


def log_span(path):
    stamps = []
    for line in path.read_text(errors="replace").splitlines():
        match = STAMP.match(line)
        if match:
            stamps.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f"))
    if len(stamps) < 2:
        raise ValueError(f"need at least two timestamped rows: {path}")
    return (max(stamps) - min(stamps)).total_seconds()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--runtime-log-root", required=True, type=Path)
    parser.add_argument("--source-version", default="v4_deepsearch")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fast-s4-mean", type=float, default=192.930)
    parser.add_argument("--medium-s4-mean", type=float, default=359.263)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.runtime_log_root.glob("runtime_gpu*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                rows.append(row)
    by_scene = {row["scene"]: row for row in rows}
    if len(by_scene) != 30:
        raise SystemExit(f"refusing incomplete runtime rows: {len(by_scene)}/30")

    scene_rows = []
    for scene in sorted(by_scene):
        result = args.results_root / f"{scene}_{args.source_version}_result"
        stage_root = result / "stage_logs"
        stage_seconds = {}
        for stage, filename in STAGES.items():
            path = stage_root / filename
            if not path.is_file():
                raise SystemExit(f"missing stage log: {path}")
            stage_seconds[stage] = log_span(path)
        total = float(by_scene[scene]["elapsed_seconds"])
        overhead = total - sum(stage_seconds.values())
        if overhead < -1.0:
            raise SystemExit(f"negative overhead for {scene}: {overhead}")
        scene_rows.append({
            "scene": scene,
            "gpu": by_scene[scene]["gpu"],
            "total_s0_s3_seconds": total,
            "stage_seconds": stage_seconds,
            "startup_transition_cleanup_seconds": max(0.0, overhead),
        })

    sums = {stage: sum(r["stage_seconds"][stage] for r in scene_rows) for stage in STAGES}
    overhead_sum = sum(r["startup_transition_cleanup_seconds"] for r in scene_rows)
    s03_sum = sum(r["total_s0_s3_seconds"] for r in scene_rows)
    closure_error = s03_sum - sum(sums.values()) - overhead_sum
    fast_total = s03_sum + args.fast_s4_mean * 30
    medium_total = s03_sum + args.medium_s4_mean * 30
    summary = {
        "schema_version": "sceneproof_fix124_stage_timing_v1",
        "scenes": 30,
        "stage_mean_seconds": {stage: seconds / 30 for stage, seconds in sums.items()},
        "stage_gpu_hours": {stage: seconds / 3600 for stage, seconds in sums.items()},
        "startup_transition_cleanup_mean_seconds": overhead_sum / 30,
        "startup_transition_cleanup_gpu_hours": overhead_sum / 3600,
        "s0_s3_mean_seconds": s03_sum / 30,
        "s0_s3_gpu_hours": s03_sum / 3600,
        "accounting_closure_error_seconds": closure_error,
        "v5_fast": {
            "definition": "S0-S3 + Fix61",
            "s4_mean_seconds": args.fast_s4_mean,
            "s0_s4_mean_seconds": fast_total / 30,
            "s0_s4_gpu_hours": fast_total / 3600,
        },
        "v5_medium": {
            "definition": "S0-S3 + Fix61 + Fix114",
            "s4_mean_seconds": args.medium_s4_mean,
            "s0_s4_mean_seconds": medium_total / 30,
            "s0_s4_gpu_hours": medium_total / 3600,
        },
        "per_scene": scene_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    for stage in STAGES:
        print(f"{stage.upper()}_MEAN_SECONDS={sums[stage] / 30:.3f}")
        print(f"{stage.upper()}_GPU_HOURS={sums[stage] / 3600:.3f}")
    print(f"OVERHEAD_MEAN_SECONDS={overhead_sum / 30:.3f}")
    print(f"S03_MEAN_SECONDS={s03_sum / 30:.3f}")
    print(f"ACCOUNTING_CLOSURE_ERROR_SECONDS={closure_error:.6f}")
    print(f"V5_FAST_S0_S4_MEAN_SECONDS={fast_total / 30:.3f}")
    print(f"V5_MEDIUM_S0_S4_MEAN_SECONDS={medium_total / 30:.3f}")
    print(f"FIX129_TIMING={args.out.resolve()}")


if __name__ == "__main__":
    main()
