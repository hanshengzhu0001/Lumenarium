#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime
from pathlib import Path


START_RE = re.compile(
    r"START_S03 scene=(\S+) gpu=(\d+).*?"
    r"((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) .*? \d{4})$"
)
DONE_RE = re.compile(
    r"DONE_S03 scene=(\S+) gpu=(\d+) elapsed=([0-9.]+) "
    r"status=(\S+) attempts=(\d+) batch=(\d+)"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fast-s4-mean", type=float, default=192.930)
    ap.add_argument("--medium-s4-mean", type=float, default=359.263)
    ap.add_argument("--render-wall", type=float, default=415.570)
    args = ap.parse_args()

    runtime_rows = []
    for path in sorted(args.log_root.glob("runtime_gpu*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            if line.strip():
                runtime_rows.append(json.loads(line))

    successful = {}
    for row in runtime_rows:
        if row.get("status") == "ok":
            successful[row["scene"]] = row
    if len(successful) != 30:
        raise SystemExit(
            f"refusing incomplete S0-S3 timing: successful={len(successful)}/30"
        )

    starts = {}
    done = {}
    for path in sorted(args.log_root.glob("gpu*.log")):
        for line in path.read_text(errors="replace").splitlines():
            m = START_RE.search(line)
            if m:
                stamp = re.sub(r"\s+[A-Z]{2,5}\s+(\d{4})$", r" \1", m.group(3))
                starts[m.group(1)] = datetime.strptime(
                    stamp, "%a %b %d %H:%M:%S %Y"
                )
            m = DONE_RE.search(line)
            if m and m.group(4) == "ok":
                done[m.group(1)] = float(m.group(3))

    missing_spans = sorted(set(successful) - (set(starts) & set(done)))
    if missing_spans:
        raise SystemExit(f"missing START/DONE span rows: {missing_spans}")

    first_start = min(starts[s] for s in successful)
    last_finish = max(
        starts[s].timestamp() + done[s] for s in successful
    )
    s03_wall = last_finish - first_start.timestamp()
    s03_gpu = sum(float(r["elapsed_seconds"]) for r in successful.values())
    fast_s4_gpu = args.fast_s4_mean * 30
    medium_s4_gpu = args.medium_s4_mean * 30
    fast_total_gpu = s03_gpu + fast_s4_gpu
    medium_total_gpu = s03_gpu + medium_s4_gpu
    result = {
        "schema_version": "sceneproof_fix124_s03_recovered_timing_v1",
        "scenes": 30,
        "a10_devices": 2,
        "s0_s3": {
            "useful_gpu_seconds": s03_gpu,
            "mean_seconds_per_scene": s03_gpu / 30,
            "two_a10_wall_seconds": s03_wall,
            "two_a10_wall_hours": s03_wall / 3600,
            "first_start": first_start.isoformat(),
            "last_finish": datetime.fromtimestamp(last_finish).isoformat(),
        },
        "version_definition": {
            "v5_fast": "DeepSearch S0-S3 + SceneLM/Fix61",
            "v5_medium": "V5-fast + SceneProof/Fix114 true-mesh repair",
        },
        "v5_fast_s0_s4_useful_compute": {
            "s4_mean_seconds_per_scene": args.fast_s4_mean,
            "gpu_seconds": fast_total_gpu,
            "gpu_hours": fast_total_gpu / 3600,
            "mean_seconds_per_scene": fast_total_gpu / 30,
            "ideal_balanced_two_a10_hours": fast_total_gpu / 7200,
        },
        "v5_medium_s0_s4_useful_compute": {
            "s4_mean_seconds_per_scene": args.medium_s4_mean,
            "gpu_seconds": medium_total_gpu,
            "gpu_hours": medium_total_gpu / 3600,
            "mean_seconds_per_scene": medium_total_gpu / 30,
            "ideal_balanced_two_a10_hours": medium_total_gpu / 7200,
        },
        "final_render": {
            "two_a10_wall_seconds": args.render_wall,
            "reported_separately": True,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"S03_SCENES=30/30")
    print(f"S03_MEAN_SECONDS={s03_gpu / 30:.3f}")
    print(f"S03_USEFUL_GPU_HOURS={s03_gpu / 3600:.3f}")
    print(f"S03_TWO_A10_WALL_SECONDS={s03_wall:.3f}")
    print(f"S03_TWO_A10_WALL_HOURS={s03_wall / 3600:.3f}")
    print(f"V5_FAST_S0_S4_MEAN_SECONDS={fast_total_gpu / 30:.3f}")
    print(f"V5_FAST_S0_S4_USEFUL_GPU_HOURS={fast_total_gpu / 3600:.3f}")
    print(f"V5_FAST_IDEAL_TWO_A10_HOURS={fast_total_gpu / 7200:.3f}")
    print(f"V5_MEDIUM_S0_S4_MEAN_SECONDS={medium_total_gpu / 30:.3f}")
    print(f"V5_MEDIUM_S0_S4_USEFUL_GPU_HOURS={medium_total_gpu / 3600:.3f}")
    print(f"V5_MEDIUM_IDEAL_TWO_A10_HOURS={medium_total_gpu / 7200:.3f}")
    print(f"FIX128_TIMING={args.out.resolve()}")


if __name__ == "__main__":
    main()
