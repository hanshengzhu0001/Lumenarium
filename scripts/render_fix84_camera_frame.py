#!/usr/bin/env python3
"""Reposition the source S3 placement's scene camera so single_sofa_chair_1 is framed.

Usage (run before the S4 render-only path):
    python scripts/render_fix84_camera_frame.py \
        --source-placement <path-to-v4_deepsearch-placement> \
        --target-placement <path-to-fix84-commit-placement> \
        --target-object single_sofa_chair_1 \
        --distance 2.5 \
        --elevation-deg 25 \
        --azimuth-deg 35 \
        --output-suffix _chair_frame

It writes a copy of the source placement with the camera repositioned; the S4
render-only path reads the camera from this modified source.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def find_object_world_translation(placement: dict, object_id: str) -> np.ndarray:
    info = placement.get("obj_info", {}).get(object_id)
    if not isinstance(info, dict):
        raise SystemExit(f"object {object_id} not in placement")
    matrix = np.asarray(info["pose_matrix_for_blender"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise SystemExit(f"pose matrix for {object_id} is not 4x4")
    return matrix[:3, 3].copy()


def find_scene_camera_matrix(placement: dict) -> tuple[str, np.ndarray]:
    for key in ("scene_camera", "SceneCamera", "sceneCam"):
        info = placement.get("obj_info", {}).get(key)
        if isinstance(info, dict) and "pose_matrix_for_blender" in info:
            return key, np.asarray(info["pose_matrix_for_blender"], dtype=np.float64)
    raise SystemExit("scene_camera not found in placement")


def look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)
    matrix = np.eye(4)
    matrix[:3, 0] = right
    matrix[:3, 1] = new_up
    matrix[:3, 2] = forward
    matrix[:3, 3] = eye
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-placement", type=Path, required=True)
    parser.add_argument("--target-placement", type=Path, required=True)
    parser.add_argument("--target-object", default="single_sofa_chair_1")
    parser.add_argument("--distance", type=float, default=2.5)
    parser.add_argument("--elevation-deg", type=float, default=25.0)
    parser.add_argument("--azimuth-deg", type=float, default=35.0)
    parser.add_argument("--output-suffix", default="_chair_frame")
    args = parser.parse_args()

    source = json.loads(args.source_placement.read_text(encoding="utf-8"))
    target = json.loads(args.target_placement.read_text(encoding="utf-8"))

    chair_pos = find_object_world_translation(target, args.target_object)
    cam_id, _cam_matrix = find_scene_camera_matrix(source)

    elevation = math.radians(args.elevation_deg)
    azimuth = math.radians(args.azimuth_deg)
    offset = np.array(
        [
            args.distance * math.cos(elevation) * math.sin(azimuth),
            args.distance * math.cos(elevation) * math.cos(azimuth),
            args.distance * math.sin(elevation),
        ]
    )
    eye = chair_pos + offset
    up_vec = np.array([0.0, 0.0, 1.0])
    new_cam_matrix = look_at_matrix(eye, chair_pos, up_vec)

    source["obj_info"][cam_id]["pose_matrix_for_blender"] = new_cam_matrix.tolist()
    out_path = args.source_placement.with_name(
        args.source_placement.stem + args.output_suffix + args.source_placement.suffix
    )
    out_path.write_text(json.dumps(source, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Chair world: {chair_pos.tolist()}")
    print(f"Camera eye : {eye.tolist()}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()