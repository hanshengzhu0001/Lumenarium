#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime
from pathlib import Path


STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+-")
TIME = re.compile(r"Time:\s*([0-9.]+)s")


def lines(path):
    return path.read_text(errors="replace").splitlines()


def span(rows):
    stamps = []
    for row in rows:
        match = STAMP.match(row)
        if match:
            stamps.append(datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f"))
    if len(stamps) < 2:
        raise ValueError("need at least two timestamped rows")
    return (max(stamps) - min(stamps)).total_seconds()


def explicit_time(rows, marker):
    matches = []
    for row in rows:
        if marker in row:
            match = TIME.search(row)
            if match:
                matches.append(float(match.group(1)))
    if len(matches) != 1:
        raise ValueError(f"expected one timing for {marker!r}, found {len(matches)}")
    return matches[0]


def timestamp(row):
    match = STAMP.match(row)
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f") if match else None


def interval(rows, start_marker, end_marker):
    start = None
    for row in rows:
        if start is None and start_marker in row:
            start = timestamp(row)
        elif start is not None and end_marker in row:
            end = timestamp(row)
            if start is not None and end is not None:
                return (end - start).total_seconds()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True, type=Path)
    ap.add_argument("--scene", default="bedroom_01")
    ap.add_argument("--source-version", default="v4_deepsearch")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    stage_root = args.results_root / f"{args.scene}_{args.source_version}_result" / "stage_logs"
    s0_path = stage_root / "S0_geometry.log"
    s1_path = stage_root / "S1_parsing.log"
    if not s0_path.is_file() or not s1_path.is_file():
        raise SystemExit(f"missing stage logs: s0={s0_path.is_file()} s1={s1_path.is_file()}")

    s0_rows, s1_rows = lines(s0_path), lines(s1_path)
    s0_total, s1_total = span(s0_rows), span(s1_rows)
    detection = explicit_time(s1_rows, "Pipeline Step 1 finished")
    segmentation = explicit_time(s1_rows, "Pipeline Step 2 finished")
    step3 = explicit_time(s1_rows, "Pipeline Step 3 finished")
    clustering = explicit_time(s1_rows, "Part 3.1: 3D clustering finished")
    crop_metadata = explicit_time(s1_rows, "Part 3.2: obj_bbox_crop_and_save finished")
    initial_scene_graph = explicit_time(s1_rows, "Part 3.3: Scene graph generation finished")
    floor_parent_verification = explicit_time(
        s1_rows, "Part 3.4: Floor parent verification finished"
    )
    semantic = explicit_time(s1_rows, "Part 3.5: Semantic analysis finished")
    step3_known = (
        clustering
        + crop_metadata
        + initial_scene_graph
        + floor_parent_verification
        + semantic
    )
    step3_unattributed = step3 - step3_known
    if step3_unattributed < -1.0:
        raise SystemExit(f"invalid Step 3 closure: gap={step3_unattributed:.3f}s")
    step3_unattributed = max(0.0, step3_unattributed)
    scene_graph = step3 - semantic
    if scene_graph < 0:
        raise SystemExit("invalid S1 timing: semantic exceeds Pipeline Step 3")
    s1_accounted = detection + segmentation + scene_graph + semantic
    s1_overhead = s1_total - s1_accounted
    if s1_overhead < -1.0:
        raise SystemExit(f"invalid S1 closure: overhead={s1_overhead:.3f}s")
    s1_overhead = max(0.0, s1_overhead)

    depth = interval(s0_rows, "Running depth estimation", "Saved depth map")
    depth_status = "timestamp_measured" if depth is not None else "unavailable_in_existing_log"
    s0_other = None if depth is None else max(0.0, s0_total - depth)

    s1_parts = {
        "detection": detection,
        "segmentation": segmentation,
        "scene_graph_and_preprocessing": scene_graph,
        "semantic_api": semantic,
        "overhead": s1_overhead,
    }
    report = {
        "schema_version": "sceneproof_s1_substage_timing_v1",
        "scene": args.scene,
        "source_version": args.source_version,
        "evidence": {"s0_log": str(s0_path.resolve()), "s1_log": str(s1_path.resolve())},
        "scope_note": "Depth belongs to S0 and is not included in the S1 total.",
        "s1": {
            "total_seconds": s1_total,
            "components_seconds": s1_parts,
            "components_percent": {k: 100.0 * v / s1_total for k, v in s1_parts.items()},
            "closure_error_seconds": s1_total - sum(s1_parts.values()),
            "step3": {
                "total_seconds": step3,
                "components_seconds": {
                    "3d_clustering": clustering,
                    "object_crop_and_metadata": crop_metadata,
                    "initial_scene_graph_generation": initial_scene_graph,
                    "floor_parent_verification": floor_parent_verification,
                    "semantic_api": semantic,
                    "unattributed_gap": step3_unattributed,
                },
                "closure_error_seconds": step3 - (
                    clustering
                    + crop_metadata
                    + initial_scene_graph
                    + floor_parent_verification
                    + semantic
                    + step3_unattributed
                ),
                "initial_scene_graph_note": (
                    "This bucket may include API latency; inspect nested request logs before "
                    "classifying it as purely local compute."
                ),
            },
        },
        "s0": {
            "total_seconds": s0_total,
            "depth_seconds": depth,
            "depth_status": depth_status,
            "other_geometry_loading_overhead_seconds": s0_other,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"SCENE={args.scene}")
    print(f"S1_TOTAL_SECONDS={s1_total:.3f}")
    for key, value in s1_parts.items():
        print(f"S1_{key.upper()}_SECONDS={value:.3f} SHARE={100.0 * value / s1_total:.2f}%")
    print(f"S1_CLOSURE_ERROR_SECONDS={report['s1']['closure_error_seconds']:.6f}")
    print(f"STEP3_TOTAL_SECONDS={step3:.3f}")
    print(f"STEP3_3D_CLUSTERING_SECONDS={clustering:.3f}")
    print(f"STEP3_OBJECT_CROP_METADATA_SECONDS={crop_metadata:.3f}")
    print(f"STEP3_INITIAL_SCENE_GRAPH_SECONDS={initial_scene_graph:.3f}")
    print(f"STEP3_FLOOR_PARENT_VERIFICATION_SECONDS={floor_parent_verification:.3f}")
    print(f"STEP3_SEMANTIC_API_SECONDS={semantic:.3f}")
    print(f"STEP3_UNATTRIBUTED_GAP_SECONDS={step3_unattributed:.3f}")
    print(f"STEP3_CLOSURE_ERROR_SECONDS={report['s1']['step3']['closure_error_seconds']:.6f}")
    if depth is None:
        print("S0_DEPTH_SECONDS=unavailable_in_existing_log")
    else:
        print(f"S0_DEPTH_SECONDS={depth:.3f}")
        print(f"S0_OTHER_SECONDS={s0_other:.3f}")
    print(f"FIX130_AUDIT={args.out.resolve()}")


if __name__ == "__main__":
    main()
