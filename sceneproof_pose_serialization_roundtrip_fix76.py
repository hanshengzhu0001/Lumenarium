#!/usr/bin/env python3
"""Certify that an in-process S4 render round-trips through its saved JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sceneproof_pillow_pose_lineage_fix75 import matrix_metrics
from sceneproof_render_roundtrip_compare import compare, load_rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-placement", type=Path, required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument("--inprocess-render", type=Path, required=True)
    parser.add_argument("--roundtrip-render", type=Path, required=True)
    parser.add_argument("--pipeline-log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference_placement.read_text(encoding="utf-8"))
    current = json.loads(args.placement.read_text(encoding="utf-8"))
    serialization = current.get("sceneproof_pose_serialization", {})
    image_metrics = compare(
        load_rgb(args.inprocess_render), load_rgb(args.roundtrip_render)
    )
    pillow_changes = {}
    reference_info = reference.get("obj_info", {})
    current_info = current.get("obj_info", {})
    for object_id in sorted(set(reference_info) & set(current_info)):
        if not object_id.startswith("pillow_"):
            continue
        first = reference_info[object_id].get("pose_matrix_for_blender")
        second = current_info[object_id].get("pose_matrix_for_blender")
        if first is None or second is None:
            continue
        metrics = matrix_metrics(first, second)
        if metrics["max_abs"] > 1e-7:
            pillow_changes[object_id] = metrics

    log_text = ""
    if args.pipeline_log is not None and args.pipeline_log.is_file():
        log_text = args.pipeline_log.read_text(encoding="utf-8", errors="replace")
    parity_match = re.search(
        r"Pose serialization/render parity:.*max_abs_delta=([^, ]+).*passed=True",
        log_text,
    )
    pose_drift_guard_passed = bool(parity_match)
    pose_drift_max_abs = (
        float(parity_match.group(1)) if parity_match is not None else None
    )
    gates = {
        "all_placement_roots_serialized": bool(
            serialization.get("policy") == "all_placement_owned_blender_roots"
            and serialization.get("serialized_objects")
            == serialization.get("placement_records")
        ),
        "no_missing_blender_roots": serialization.get("missing_objects") == 0,
        "inprocess_render_pose_drift_guard": pose_drift_guard_passed,
        # Independent Cycles renders are not bitwise deterministic.  PSNR and
        # a bounded changed-pixel fraction guard structural parity while
        # permitting stochastic sampling/denoising differences.
        "beauty_roundtrip_psnr": image_metrics["psnr_db"] >= 55.0,
        "beauty_roundtrip_changed_pixels": (
            image_metrics["changed_pixel_fraction_gt_2_levels"] <= 0.002
        ),
    }
    passed = all(gates.values())
    report = {
        "schema_version": "sceneproof_pose_serialization_roundtrip_v1",
        "passed": passed,
        "placement": str(args.placement.resolve()),
        "reference_placement": str(args.reference_placement.resolve()),
        "inprocess_render": str(args.inprocess_render.resolve()),
        "roundtrip_render": str(args.roundtrip_render.resolve()),
        "serialization": serialization,
        "pipeline_log": (
            str(args.pipeline_log.resolve()) if args.pipeline_log else None
        ),
        "post_render_pose_drift_max_abs": pose_drift_max_abs,
        "image_roundtrip": image_metrics,
        "gates": gates,
        "pillow_pose_changes_from_original_fix43_json": pillow_changes,
        "decision": (
            "pose_serialization_fix_validated"
            if passed
            else "stop_and_diagnose_pose_serialization"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out.resolve()}")
    print(
        f"PASSED={passed} PSNR={image_metrics['psnr_db']:.4f}dB "
        "CHANGED_PIXELS="
        f"{image_metrics['changed_pixel_fraction_gt_2_levels']:.6f} "
        f"SERIALIZED={serialization.get('serialized_objects')} "
        "WITHOUT_RIGID_BODY="
        f"{serialization.get('serialized_without_rigid_body')} "
        f"PILLOWS_CHANGED={len(pillow_changes)} "
        f"DECISION={report['decision']}"
    )
    print(f"GATES={json.dumps(gates, sort_keys=True)}")
    for object_id, metrics in pillow_changes.items():
        print(
            f"PILLOW={object_id} "
            f"translation_m={metrics['translation_norm_m']:.9g} "
            f"linear={metrics['linear_frobenius']:.9g}"
        )
    print(f"FIX76_AUDIT={args.out.resolve()}")


if __name__ == "__main__":
    main()
