#!/usr/bin/env python3
"""GT-based evaluation for the frozen v1/v3 Imaginarium run.

This script intentionally does not run generation. It reads:
  - demo/*_v1.png to define the 151 scene ids
  - saved_results/*_result/S4_layout_refinement/*_placement_info_s4.json
  - asset_data/imaginarium_3d_scene_layout_dataset/*/*/*_meta.json

It writes:
  - eval_freeze_manifest.json
  - eval_gt_metrics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


VERSIONS = ("v1", "v3")
IGNORED_PREFIXES = ("floor", "wall", "ceiling", "scene_camera")
IMAGE_AREA = 1024 * 1024


SYNONYM_GROUPS = {
    "storage_shelf": {
        "desk_bookshelf",
        "storage_rack",
        "storage_shelf",
        "storage_locker",
        "storage_cabinet",
        "file_cabinet",
        "shelving",
        "bookshelf",
        "shelf",
        "rack",
    },
    "potted_plant": {
        "flower_pot",
        "potted_plant",
        "small_potted_plant",
        "large_potted_plant",
        "plant_pot",
        "planter",
    },
    "sofa_chair": {
        "single_sofa_chair",
        "sofa_chair",
        "backrest_chair",
        "armchair",
        "lounge_chair",
        "leather_chair",
    },
    "table": {
        "coffee_table",
        "dining_table",
        "office_desk",
        "desk",
        "side_table",
        "long_table",
        "workbench",
        "table",
    },
    "cabinet": {
        "kitchen_cabinet",
        "display_cabinet",
        "cabinet",
        "wardrobe",
        "locker",
    },
    "box": {
        "cardboard_box",
        "empty_cardboard_box",
        "shipping_box",
        "storage_box",
        "small_storage_box",
        "gift_box",
        "wooden_crate",
        "large_wooden_crate",
        "military_box",
        "crate",
    },
    "speaker": {
        "speaker",
        "small_speaker",
        "audio_speaker",
        "loudspeaker",
    },
    "sign_board": {
        "billboard",
        "billboard_stand",
        "freestanding_sign",
        "display_board",
        "sign",
        "sign_board",
    },
    "railing": {
        "street_railing",
        "railing",
        "fence",
        "guardrail",
    },
    "tank": {
        "fuel_tank",
        "large_water_storage_tank",
        "water_tank",
        "industrial_oil_drum",
        "industrial_plastic_drum",
        "barrel",
    },
    "tv_monitor": {
        "lcd_tv",
        "wall_mounted_lcd_tv",
        "vintage_television_set",
        "television",
        "tv",
        "computer_monitor",
        "vintage_computer_monitor",
        "vintage_surveillance_monitor",
        "monitor",
    },
    "radio": {
        "portable_radio",
        "desktop_radio",
        "radio",
    },
    "industrial_component": {
        "discarded_industrial_component",
        "mechanical_component",
        "small_mechanical_component",
        "vent_component",
        "air_vent_duct",
        "ventilation",
        "water_pipe",
        "cable",
        "portable_generator",
    },
    "lamp": {
        "ceiling_lamp",
        "chandelier",
        "wall_mounted_lamp_holder",
        "desktop_table_lamp",
        "table_lamp",
        "floor_lamp",
        "lamp",
    },
    "vase": {"vase", "large_vase", "ceramic_vase"},
    "bowl": {"bowl", "small_bowl"},
    "plate": {"plate", "plates"},
    "carpet": {"carpet", "rug"},
    "fruit": {"fruit", "apple", "lemon", "pomegranate", "watermelon", "tomato"},
}

ALIAS_TO_CANONICAL = {
    alias: canonical for canonical, aliases in SYNONYM_GROUPS.items() for alias in aliases
}
GENERIC_CANONICALS = {"table", "cabinet", "box", "lamp", "fruit"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def scene_ids(demo_dir: Path) -> list[str]:
    ids = []
    for path in demo_dir.glob("*_v1.png"):
        ids.append(path.name[: -len("_v1.png")])
    return sorted(ids)


def read_scene_filter(value: str | None) -> list[str] | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def meta_path(dataset_dir: Path, scene: str) -> Path:
    matches = list(dataset_dir.glob(f"*/{scene}/{scene}_meta.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one meta for {scene}, found {len(matches)}")
    return matches[0]


def detect_items_path(meta_json_path: Path, scene: str) -> Path:
    return meta_json_path.with_name(f"{scene}_detect_items.pkl")


def s4_path(saved_results: Path, scene: str, version: str) -> Path:
    folder = saved_results / f"{scene}_{version}_result" / "S4_layout_refinement"
    matches = list(folder.glob(f"{scene}_{version}_placement_info_s4.json"))
    if not matches:
        matches = list(folder.glob("*_placement_info_s4.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one S4 json for {scene} {version}, found {len(matches)}")
    return matches[0]


def norm_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower().replace(".fbx", ""))


def norm_label(value: Any) -> str:
    raw = str(value or "").lower().replace(".fbx", "")
    raw = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", raw)
    parts = [p for p in re.split(r"[^a-z0-9]+", raw) if p]
    while parts and parts[-1].isdigit():
        parts.pop()
    return "_".join(parts)


def tokens(value: Any) -> set[str]:
    raw = str(value or "").lower().replace(".fbx", "")
    parts = re.split(r"[^a-z0-9]+", raw)
    stop = {"a", "an", "the", "sm", "nn", "packed", "2k", "01", "02", "03"}
    return {p for p in parts if p and p not in stop and not p.isdigit()}


def canonical_labels(*values: Any) -> set[str]:
    labels = set()
    for value in values:
        norm = norm_label(value)
        if not norm:
            continue
        labels.add(norm)
        if norm in ALIAS_TO_CANONICAL:
            labels.add(ALIAS_TO_CANONICAL[norm])

        parts = norm.split("_")
        for i in range(len(parts)):
            for j in range(i + 1, min(len(parts), i + 4) + 1):
                phrase = "_".join(parts[i:j])
                canonical = ALIAS_TO_CANONICAL.get(phrase)
                if canonical:
                    labels.add(canonical)
    return labels


def token_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    inter = len(left & right)
    if inter == 0:
        return 0.0
    return inter / max(len(left), len(right))


def matrix(info: dict[str, Any], key: str) -> np.ndarray | None:
    value = info.get(key)
    if not value:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.shape != (4, 4):
        return None
    return arr


def translation(mat: np.ndarray) -> np.ndarray:
    return mat[:3, 3].astype(float)


def rotation(mat: np.ndarray) -> np.ndarray:
    r = mat[:3, :3].astype(float)
    # Project away scale/shear to the closest rotation matrix.
    try:
        u, _, vt = np.linalg.svd(r)
        rr = u @ vt
        if np.linalg.det(rr) < 0:
            u[:, -1] *= -1
            rr = u @ vt
        return rr
    except np.linalg.LinAlgError:
        return np.eye(3)


def rotation_error_deg(pred_r: np.ndarray, gt_r: np.ndarray) -> float:
    rel = pred_r.T @ gt_r
    value = (np.trace(rel) - 1.0) / 2.0
    value = float(np.clip(value, -1.0, 1.0))
    return math.degrees(math.acos(value))


def auc_at(errors: list[float], threshold: float) -> float | None:
    if not errors:
        return None
    clipped = np.clip(np.asarray(errors, dtype=float), 0.0, threshold)
    xs = np.concatenate([[0.0], np.sort(clipped), [threshold]])
    recalls = np.concatenate([[0.0], np.arange(1, len(clipped) + 1) / len(clipped), [1.0]])
    area = np.trapezoid(recalls, xs) / threshold
    return float(area)


def summarize_values(values: list[float], threshold: float | None = None) -> dict[str, Any]:
    if not values:
        out: dict[str, Any] = {"n": 0, "mean": None, "median": None}
        if threshold is not None:
            out["recall_at_threshold"] = None
            out["auc_at_threshold"] = None
        return out
    arr = np.asarray(values, dtype=float)
    out = {
        "n": int(len(values)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }
    if threshold is not None:
        out["recall_at_threshold"] = float(np.mean(arr <= threshold))
        out["auc_at_threshold"] = auc_at(values, threshold)
    return out


def parent_category(parent: Any) -> str:
    p = str(parent or "").lower()
    if p in {"", "none", "null"}:
        return "none"
    if p.startswith("floor") or p == "ground":
        return "floor"
    if p.startswith("wall"):
        return "wall"
    if p.startswith("ceiling"):
        return "ceiling"
    return "object"


def pred_parent_id(info: dict[str, Any]) -> str | None:
    supported = info.get("supported")
    if supported:
        return str(supported)
    if info.get("isOnFloor") is True:
        return "floor"
    if info.get("isHangingOnWall") is True or info.get("isAgainstWall") is True:
        return "wall"
    if info.get("isHangingFromCeiling") is True:
        return "ceiling"
    return None


def predicted_objects(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for obj_id, info in data.get("obj_info", {}).items():
        low = obj_id.lower()
        if low.startswith(IGNORED_PREFIXES):
            continue
        if not isinstance(info, dict):
            continue
        out[obj_id] = info
    return out


def gt_objects(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for obj_id, info in data.get("objects", {}).items():
        low = obj_id.lower()
        if low.startswith(IGNORED_PREFIXES):
            continue
        if not isinstance(info, dict):
            continue
        if info.get("type") and info.get("type") != "MESH":
            continue
        class_low = str(info.get("class_en") or "").lower()
        if class_low.startswith(("floor", "wall", "ceiling")):
            continue
        out[obj_id] = info
    return out


def load_detect_items(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def find_detect_item(detect_items: dict[str, Any], gt_id: str, gt: dict[str, Any]) -> dict[str, Any] | None:
    if gt_id in detect_items:
        return detect_items[gt_id]
    stem = str(gt.get("fbx_name") or "").replace(".fbx", "")
    if stem and stem in detect_items:
        return detect_items[stem]
    normalized_gt = norm_text(gt_id)
    normalized_stem = norm_text(stem)
    for key, value in detect_items.items():
        nk = norm_text(key)
        if nk and nk in {normalized_gt, normalized_stem}:
            return value
    return None


def visible_gt_objects(
    gt: dict[str, dict[str, Any]],
    detect_items: dict[str, Any],
    min_mask_area: int,
    min_bbox_size: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if min_mask_area <= 0 and min_bbox_size <= 0:
        return gt, {
            "enabled": False,
            "all_gt_object_count": len(gt),
            "visible_gt_object_count": len(gt),
            "filtered_gt_object_count": 0,
            "filters": {},
            "filtered_reasons": {},
        }

    visible = {}
    filtered_reasons: dict[str, int] = defaultdict(int)
    details = {}
    for gt_id, info in gt.items():
        item = find_detect_item(detect_items, gt_id, info)
        if not item:
            filtered_reasons["missing_detect_item"] += 1
            details[gt_id] = {"visible": False, "reason": "missing_detect_item"}
            continue
        bbox = item.get("bbox_xyxy") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bbox_w = max(0, x2 - x1)
        bbox_h = max(0, y2 - y1)
        mask = item.get("mask")
        mask_area = int(np.asarray(mask).sum()) if mask is not None else 0
        if mask_area < min_mask_area:
            filtered_reasons["small_mask_area"] += 1
            details[gt_id] = {"visible": False, "reason": "small_mask_area", "mask_area": mask_area}
            continue
        if min(bbox_w, bbox_h) < min_bbox_size:
            filtered_reasons["small_bbox"] += 1
            details[gt_id] = {
                "visible": False,
                "reason": "small_bbox",
                "mask_area": mask_area,
                "bbox_xyxy": bbox,
            }
            continue
        visible[gt_id] = info
        details[gt_id] = {
            "visible": True,
            "mask_area": mask_area,
            "mask_area_ratio": mask_area / IMAGE_AREA,
            "bbox_xyxy": bbox,
        }

    return visible, {
        "enabled": True,
        "all_gt_object_count": len(gt),
        "visible_gt_object_count": len(visible),
        "filtered_gt_object_count": len(gt) - len(visible),
        "filters": {
            "min_mask_area": min_mask_area,
            "min_bbox_size": min_bbox_size,
        },
        "filtered_reasons": dict(filtered_reasons),
        "details": details,
    }


def object_similarity(pred_id: str, pred: dict[str, Any], gt_id: str, gt: dict[str, Any]) -> tuple[float, str]:
    pred_asset = pred.get("retrieved_asset")
    gt_fbx = gt.get("fbx_name")
    if pred_asset:
        npred = norm_text(pred_asset)
        if npred and npred in {norm_text(gt_id), norm_text(gt_fbx)}:
            return 1.0, "asset_exact"

    pred_canonical = canonical_labels(pred_id, pred_asset)
    gt_canonical = canonical_labels(gt_id, gt_fbx, gt.get("class_en"), gt.get("caption_en"))
    shared = pred_canonical & gt_canonical
    useful_shared = shared - GENERIC_CANONICALS
    if useful_shared:
        return 0.82, "canonical_synonym"
    if shared:
        return 0.68, "canonical_generic"

    pred_tokens = tokens(pred_id) | tokens(pred_asset)
    gt_tokens = tokens(gt_id) | tokens(gt_fbx) | tokens(gt.get("class_en")) | tokens(gt.get("caption_en"))
    score = token_score(pred_tokens, gt_tokens)
    if score > 0:
        return score, "token"
    return 0.0, "none"


def match_objects(
    pred_objs: dict[str, dict[str, Any]],
    gt_objs: dict[str, dict[str, Any]],
    min_match_score: float,
) -> list[dict[str, Any]]:
    candidates = []
    for pred_id, pred in pred_objs.items():
        pm = matrix(pred, "pose_matrix_for_blender")
        if pm is None:
            continue
        pp = translation(pm)
        for gt_id, gt in gt_objs.items():
            gm = matrix(gt, "matrix_world_4x4")
            if gm is None:
                continue
            score, method = object_similarity(pred_id, pred, gt_id, gt)
            if score < min_match_score:
                continue
            dist = float(np.linalg.norm(pp - translation(gm)))
            candidates.append((score, -dist, pred_id, gt_id, method, dist))

    # Hungarian algorithm for max-weight bipartite matching (replaces greedy)
    if not candidates:
        return []
    from scipy.optimize import linear_sum_assignment
    pred_ids = sorted(set(c[2] for c in candidates))
    gt_ids   = sorted(set(c[3] for c in candidates))
    pid2idx  = {p:i for i,p in enumerate(pred_ids)}
    gid2idx  = {g:i for i,g in enumerate(gt_ids)}
    cost = np.full((len(pred_ids), len(gt_ids)), 1e9)
    for score, _, pred_id, gt_id, method, dist in candidates:
        cost[pid2idx[pred_id], gid2idx[gt_id]] = -score
    row_ind, col_ind = linear_sum_assignment(cost)
    # Build match list from Hungarian output
    # Recover best (score, method, dist) for each matched pair
    pair_map = {}
    for score, _, pred_id, gt_id, method, dist in candidates:
        key = (pid2idx[pred_id], gid2idx[gt_id])
        if key not in pair_map or score > pair_map[key][0]:
            pair_map[key] = (score, method, dist)
    matches = []
    for ri, ci in zip(row_ind, col_ind):
        if cost[ri, ci] >= 1e9 - 1:
            continue
        score, method, dist = pair_map.get((ri, ci), (0.0, "none", 0.0))
        pred_id = pred_ids[ri]
        gt_id = gt_ids[ci]
        matches.append(
            {
                "pred_id": pred_id,
                "gt_id": gt_id,
                "score": float(score),
                "method": method,
                "raw_translation_distance": float(dist),
            }
        )
    return matches


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray] | None:
    if src.shape[0] < 3 or dst.shape[0] < 3:
        return None
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    var_src = float(np.sum(src_c * src_c) / src.shape[0])
    if var_src <= 1e-12:
        return None
    cov = (dst_c.T @ src_c) / src.shape[0]
    try:
        u, s, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None
    d = np.ones(3)
    if np.linalg.det(u @ vt) < 0:
        d[-1] = -1
    r = u @ np.diag(d) @ vt
    scale = float(np.sum(s * d) / var_src)
    t = dst_mean - scale * (r @ src_mean)
    return scale, r, t


def semantic_parent_match(
    gt: dict[str, dict[str, Any]],
    gt_parent_id: str,
    matched_gt_parent_id: str,
) -> bool:
    """Semantic parent comparison: compare class_en canonical labels instead of asset IDs."""
    gt_parent_info = gt.get(gt_parent_id, {})
    matched_info = gt.get(matched_gt_parent_id, {})
    if not gt_parent_info or not matched_info:
        return False

    gt_class = str(gt_parent_info.get("class_en") or "").lower()
    matched_class = str(matched_info.get("class_en") or "").lower()
    if not gt_class or not matched_class:
        return False

    # Exact class match
    if gt_class == matched_class:
        return True

    # Canonical label overlap via synonym groups
    gt_canonical = canonical_labels(gt_class)
    matched_canonical = canonical_labels(matched_class)
    if gt_canonical & matched_canonical:
        return True

    # Shared canonical (excluding generics)
    shared = (gt_canonical & matched_canonical) - GENERIC_CANONICALS
    if shared:
        return True

    # Token overlap as last resort
    gt_tokens = tokens(gt_class)
    matched_tokens = tokens(matched_class)
    return len(gt_tokens & matched_tokens) >= max(1, min(len(gt_tokens), len(matched_tokens)) * 0.5)


def eval_scene(
    meta: dict[str, Any],
    pred: dict[str, Any],
    min_match_score: float,
    detect_items: dict[str, Any] | None = None,
    min_visible_mask_area: int = 0,
    min_visible_bbox_size: int = 0,
    semantic_parent: bool = True,
) -> dict[str, Any]:
    all_gt = gt_objects(meta)
    gt, visibility = visible_gt_objects(
        all_gt,
        detect_items or {},
        min_mask_area=min_visible_mask_area,
        min_bbox_size=min_visible_bbox_size,
    )
    pred_objs = predicted_objects(pred)
    matches = match_objects(pred_objs, gt, min_match_score=min_match_score)
    pred_to_gt = {m["pred_id"]: m["gt_id"] for m in matches}

    raw_t_errors = []
    aligned_t_errors = []
    raw_r_errors = []
    aligned_r_errors = []
    pose_pairs = []

    for m in matches:
        pred_info = pred_objs[m["pred_id"]]
        gt_info = gt[m["gt_id"]]
        pm = matrix(pred_info, "pose_matrix_for_blender")
        gm = matrix(gt_info, "matrix_world_4x4")
        if pm is None or gm is None:
            continue
        pose_pairs.append((m, pm, gm))
        raw_t_errors.append(float(np.linalg.norm(translation(pm) - translation(gm))))
        raw_r_errors.append(rotation_error_deg(rotation(pm), rotation(gm)))

    align = None
    if len(pose_pairs) >= 3:
        src = np.stack([translation(pm) for _, pm, _ in pose_pairs])
        dst = np.stack([translation(gm) for _, _, gm in pose_pairs])
        align = umeyama(src, dst)

    alignment_info: dict[str, Any] = {"available": align is not None, "matched_points": len(pose_pairs)}
    if align is not None:
        scale, r_align, t_align = align
        alignment_info.update({"scale": scale, "rotation": r_align.tolist(), "translation": t_align.tolist()})
        for _, pm, gm in pose_pairs:
            ap = scale * (r_align @ translation(pm)) + t_align
            aligned_t_errors.append(float(np.linalg.norm(ap - translation(gm))))
            aligned_r_errors.append(rotation_error_deg(r_align @ rotation(pm), rotation(gm)))
    else:
        aligned_t_errors = list(raw_t_errors)
        aligned_r_errors = list(raw_r_errors)

    parent_total = 0
    parent_correct = 0
    parent_details = []
    for m in matches:
        pred_info = pred_objs[m["pred_id"]]
        gt_info = gt[m["gt_id"]]
        gt_parent = gt_info.get("parent")
        gt_cat = parent_category(gt_parent)
        pp = pred_parent_id(pred_info)
        pred_cat = parent_category(pp)
        correct = False
        match_method = "none"
        if gt_cat in {"floor", "wall", "ceiling", "none"}:
            correct = pred_cat == gt_cat
            match_method = "category" if correct else "mismatch"
        elif pp in pred_to_gt:
            if semantic_parent:
                correct = semantic_parent_match(gt, gt_parent, pred_to_gt[pp])
                match_method = "semantic_class" if correct else "semantic_mismatch"
            else:
                correct = pred_to_gt[pp] == gt_parent
                match_method = "exact_id" if correct else "id_mismatch"
        parent_total += 1
        parent_correct += int(correct)
        parent_details.append(
            {
                "pred_id": m["pred_id"],
                "gt_id": m["gt_id"],
                "pred_parent": pp,
                "gt_parent": gt_parent,
                "correct": correct,
                "method": match_method,
            }
        )

    # Paper-compatible: split metrics by primary (floor/wall/ceiling-supported) vs secondary
    def _gt_is_primary(gt_id: str, gt_info: dict[str, Any]) -> bool:
        parent = str(gt_info.get("parent", "")).lower()
        return parent.startswith("floor") or parent.startswith("wall") or parent.startswith("ceiling")

    primary_gt = {k: v for k, v in gt.items() if _gt_is_primary(k, v)}
    secondary_gt = {k: v for k, v in gt.items() if not _gt_is_primary(k, v)}

    primary_matches = [m for m in matches if _gt_is_primary(m["gt_id"], gt[m["gt_id"]])]
    secondary_matches = [m for m in matches if not _gt_is_primary(m["gt_id"], gt[m["gt_id"]])]

    primary_parent = [(p["correct"],) for p in parent_details if _gt_is_primary(p["gt_id"], gt[p["gt_id"]])]
    secondary_parent = [(p["correct"],) for p in parent_details if not _gt_is_primary(p["gt_id"], gt[p["gt_id"]])]

    primary_pose = [(m, pm, gm) for m, pm, gm in pose_pairs if _gt_is_primary(m["gt_id"], gt[m["gt_id"]])]
    secondary_pose = [(m, pm, gm) for m, pm, gm in pose_pairs if not _gt_is_primary(m["gt_id"], gt[m["gt_id"]])]

    def _compute_subset(sub_gt, sub_matches, sub_parent, sub_pose):
        n_gt = len(sub_gt)
        n_match = len(sub_matches)
        recovery = n_match / n_gt if n_gt else None
        parent_correct_count = sum(1 for (c,) in sub_parent if c)
        parent_total = len(sub_parent)
        parent_acc = parent_correct_count / parent_total if parent_total else None

        raw_t = [float(np.linalg.norm(translation(pm)-translation(gm))) for _,pm,gm in sub_pose]
        raw_r = [rotation_error_deg(rotation(pm), rotation(gm)) for _,pm,gm in sub_pose]
        aligned_t = raw_t
        aligned_r = raw_r
        if align is not None:
            aligned_t = [float(np.linalg.norm((scale*(r_align@translation(pm))+t_align)-translation(gm))) for _,pm,gm in sub_pose]
            aligned_r = [rotation_error_deg(r_align@rotation(pm), rotation(gm)) for _,pm,gm in sub_pose]

        return {
            "object_count": n_gt,
            "matched_object_count": n_match,
            "object_recovery": recovery,
            "parent_accuracy": parent_acc,
            "parent_eval_count": parent_total,
            "rotation_auc60_aligned": summarize_values(aligned_r, threshold=60.0).get("auc_at_threshold"),
            "translation_auc05_aligned": summarize_values(aligned_t, threshold=0.5).get("auc_at_threshold"),
        }

    primary_metrics = _compute_subset(primary_gt, primary_matches, primary_parent, primary_pose)
    secondary_metrics = _compute_subset(secondary_gt, secondary_matches, secondary_parent, secondary_pose)

    return {
        "all_gt_object_count": len(all_gt),
        "visible_gt_object_count": len(gt),
        "filtered_gt_object_count": len(all_gt) - len(gt),
        "gt_object_count": len(gt),
        "pred_object_count": len(pred_objs),
        "matched_object_count": len(matches),
        "object_recovery": len(matches) / len(gt) if gt else None,
        "prediction_match_rate": len(matches) / len(pred_objs) if pred_objs else None,
        "scene_graph_parent_accuracy_gt": parent_correct / parent_total if parent_total else None,
        "parent_eval_count": parent_total,
        "pose_eval_count": len(pose_pairs),
        "translation_raw": summarize_values(raw_t_errors, threshold=0.5),
        "translation_aligned": summarize_values(aligned_t_errors, threshold=0.5),
        "rotation_raw": summarize_values(raw_r_errors, threshold=60.0),
        "rotation_aligned": summarize_values(aligned_r_errors, threshold=60.0),
        "alignment": alignment_info,
        "visibility": visibility,
        "matches": matches,
        "parent_details": parent_details,
        "primary_metrics": primary_metrics,
        "secondary_metrics": secondary_metrics,
    }


def weighted_mean(items: list[dict[str, Any]], key: str, weight_key: str | None = None) -> float | None:
    vals = []
    weights = []
    for item in items:
        value = item.get(key)
        if value is None:
            continue
        weight = item.get(weight_key, 1) if weight_key else 1
        if weight is None or weight <= 0:
            continue
        vals.append(float(value) * float(weight))
        weights.append(float(weight))
    if not weights:
        return None
    return sum(vals) / sum(weights)


def aggregate(version_scenes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scenes = list(version_scenes.values())
    out = {
        "scene_count": len(scenes),
        "all_gt_object_count": sum(s.get("all_gt_object_count", s["gt_object_count"]) for s in scenes),
        "visible_gt_object_count": sum(s.get("visible_gt_object_count", s["gt_object_count"]) for s in scenes),
        "filtered_gt_object_count": sum(s.get("filtered_gt_object_count", 0) for s in scenes),
        "gt_object_count": sum(s["gt_object_count"] for s in scenes),
        "pred_object_count": sum(s["pred_object_count"] for s in scenes),
        "matched_object_count": sum(s["matched_object_count"] for s in scenes),
        "object_recovery": None,
        "prediction_match_rate": None,
        "scene_graph_parent_accuracy_gt": None,
        "translation_auc05_aligned": None,
        "rotation_auc60_aligned": None,
        "translation_auc05_raw": None,
        "rotation_auc60_raw": None,
    }
    if out["gt_object_count"]:
        out["object_recovery"] = out["matched_object_count"] / out["gt_object_count"]
    if out["pred_object_count"]:
        out["prediction_match_rate"] = out["matched_object_count"] / out["pred_object_count"]
    parent_total = sum(s["parent_eval_count"] for s in scenes)
    if parent_total:
        out["scene_graph_parent_accuracy_gt"] = weighted_mean(
            scenes, "scene_graph_parent_accuracy_gt", "parent_eval_count"
        )

    for prefix, threshold in [("translation_raw", 0.5), ("translation_aligned", 0.5), ("rotation_raw", 60.0), ("rotation_aligned", 60.0)]:
        errors = []
        # Reconstruct aggregate from summary is lossy, so average scene AUC weighted by pose count.
        auc_key = "auc_at_threshold"
        weight_items = []
        for s in scenes:
            summary = s[prefix]
            if summary["n"]:
                weight_items.append({"auc": summary[auc_key], "n": summary["n"]})
        auc = weighted_mean(weight_items, "auc", "n")
        if prefix == "translation_aligned":
            out["translation_auc05_aligned"] = auc
        elif prefix == "translation_raw":
            out["translation_auc05_raw"] = auc
        elif prefix == "rotation_aligned":
            out["rotation_auc60_aligned"] = auc
        elif prefix == "rotation_raw":
            out["rotation_auc60_raw"] = auc

    # Paper-compatible: primary/secondary split
    for subset_key in ["primary_metrics", "secondary_metrics"]:
        sub_scenes = [s.get(subset_key, {}) for s in scenes if s.get(subset_key)]
        total_obj = sum(m.get("object_count", 0) for m in sub_scenes)
        total_match = sum(m.get("matched_object_count", 0) for m in sub_scenes)
        p_items = [{"auc": m.get("parent_accuracy"), "n": m.get("parent_eval_count", 0)} 
                   for m in sub_scenes if m.get("parent_accuracy") is not None and m.get("parent_eval_count")]
        r_auc_items = [{"auc": m.get("rotation_auc60_aligned"), "n": m.get("object_count", 0)} 
                       for m in sub_scenes if m.get("rotation_auc60_aligned") is not None]
        t_auc_items = [{"auc": m.get("translation_auc05_aligned"), "n": m.get("object_count", 0)} 
                       for m in sub_scenes if m.get("translation_auc05_aligned") is not None]
        out[subset_key] = {
            "object_count": total_obj,
            "matched_object_count": total_match,
            "object_recovery": total_match / total_obj if total_obj else None,
            "parent_accuracy": weighted_mean(p_items, "auc", "n"),
            "rotation_auc60_aligned": weighted_mean(r_auc_items, "auc", "n"),
            "translation_auc05_aligned": weighted_mean(t_auc_items, "auc", "n"),
        }

    return out


def aggregate_by_category(scenes: dict[str, dict[str, Any]], scene_to_category: dict[str, str]) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for scene, result in scenes.items():
        grouped[scene_to_category[scene]][scene] = result
    return {cat: aggregate(vals) for cat, vals in sorted(grouped.items())}


def build_freeze_manifest(
    scenes: list[str],
    saved_results: Path,
    batch_logs: Path,
    dataset_dir: Path,
    versions: tuple[str, ...],
) -> dict[str, Any]:
    s4_entries = []
    for scene in scenes:
        for version in versions:
            path = s4_path(saved_results, scene, version)
            stat = path.stat()
            s4_entries.append(
                {
                    "scene": scene,
                    "version": version,
                    "path": str(path),
                    "real_path": str(path.resolve()),
                    "size_bytes": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "sha256": sha256_file(path),
                }
            )

    logs = []
    if batch_logs.exists():
        for path in sorted(batch_logs.glob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            logs.append(
                {
                    "path": str(path),
                    "real_path": str(path.resolve()),
                    "size_bytes": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "sha256": sha256_file(path),
                }
            )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scene_count": len(scenes),
        "expected_task_count": len(scenes) * len(versions),
        "s4_json_count": len(s4_entries),
        "dataset_dir": str(dataset_dir),
        "saved_results": {"path": str(saved_results), "real_path": str(saved_results.resolve())},
        "batch_logs": {"path": str(batch_logs), "real_path": str(batch_logs.resolve()) if batch_logs.exists() else None},
        "git": {
            "head": run_git(["rev-parse", "HEAD"]),
            "branch": run_git(["branch", "--show-current"]),
            "status_short": run_git(["status", "--short", "--untracked-files=no"]).splitlines(),
            "diff_stat": run_git(["diff", "--stat"]).splitlines(),
        },
        "s4_json": s4_entries,
        "logs": logs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="asset_data/imaginarium_3d_scene_layout_dataset")
    parser.add_argument("--saved-results", default="saved_results")
    parser.add_argument("--demo-dir", default="demo")
    parser.add_argument("--batch-logs", default="batch_logs")
    parser.add_argument("--metrics-out", default="eval_gt_metrics.json")
    parser.add_argument("--manifest-out", default="eval_freeze_manifest.json")
    parser.add_argument("--min-match-score", type=float, default=0.34)
    parser.add_argument("--min-visible-mask-area", type=int, default=0)
    parser.add_argument("--min-visible-bbox-size", type=int, default=0)
    parser.add_argument("--semantic-parent", action="store_true", default=True, help="Use semantic parent matching (class name) instead of exact asset ID")
    parser.add_argument("--no-semantic-parent", dest="semantic_parent", action="store_false", help="Use exact asset ID for parent matching (stricter)")
    parser.add_argument("--scenes", default="", help="Comma-separated scene ids or a text file with one scene id per line")
    parser.add_argument("--versions", default=",".join(VERSIONS), help="Comma-separated versions to evaluate, e.g. v1,v3 or v3")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    saved_results = Path(args.saved_results)
    demo_dir = Path(args.demo_dir)
    batch_logs = Path(args.batch_logs)
    scenes = scene_ids(demo_dir)
    selected_scenes = read_scene_filter(args.scenes)
    if selected_scenes is not None:
        selected = set(selected_scenes)
        scenes = [scene for scene in scenes if scene in selected]
    versions = tuple(v.strip() for v in args.versions.split(",") if v.strip())
    if not versions:
        raise SystemExit("--versions must include at least one version")

    manifest = build_freeze_manifest(scenes, saved_results, batch_logs, dataset_dir, versions)
    write_json(Path(args.manifest_out), manifest)

    scene_to_category = {}
    per_version: dict[str, dict[str, dict[str, Any]]] = {v: {} for v in versions}
    failures = []
    for scene in scenes:
        try:
            mp = meta_path(dataset_dir, scene)
            scene_to_category[scene] = mp.parent.parent.name
            meta = load_json(mp)
            detect_items = load_detect_items(detect_items_path(mp, scene))
            for version in versions:
                pred = load_json(s4_path(saved_results, scene, version))
                per_version[version][scene] = eval_scene(
                    meta,
                    pred,
                    min_match_score=args.min_match_score,
                    detect_items=detect_items,
                    min_visible_mask_area=args.min_visible_mask_area,
                    min_visible_bbox_size=args.min_visible_bbox_size,
                    semantic_parent=args.semantic_parent,
                )
        except Exception as exc:
            failures.append({"scene": scene, "error": str(exc)})

    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "gt_source": str(dataset_dir),
            "prediction_source": str(saved_results),
            "scene_ids_source": str(demo_dir / "*_v1.png"),
            "object_matching": {
                "min_match_score": args.min_match_score,
                "primary": "exact retrieved_asset/fbx_name match",
                "fallback": "canonical synonym match, then token overlap among pred id/retrieved asset and GT id/fbx/class/caption",
                "canonical_synonym_groups": {
                    key: sorted(values) for key, values in SYNONYM_GROUPS.items()
                },
                "note": "Prediction ids and GT asset ids differ, so match coverage is reported explicitly.",
            },
            "parent_matching": {
                "method": "semantic_class" if args.semantic_parent else "exact_asset_id",
                "semantic_parent": args.semantic_parent,
                "description": "Semantic parent: compare class_en canonical labels (via synonym groups + token overlap). Exact parent: require identical asset ID." if args.semantic_parent else "Exact parent: require pred_to_gt[parent] == gt_parent (strict asset ID match).",
            },
            "visible_gt_filtering": {
                "min_visible_mask_area": args.min_visible_mask_area,
                "min_visible_bbox_size": args.min_visible_bbox_size,
                "source": "*_detect_items.pkl masks/bboxes",
                "denominator": "gt_object_count after structural and visible-GT filtering",
            },
            "pose": {
                "raw": "direct matrix_world comparison",
                "aligned": "scene-level Umeyama similarity transform from matched object centers when >=3 pose pairs",
                "dashboard_default": "aligned",
            },
        },
        "failures": failures,
        "versions": {},
        "by_category": {},
        "scenes": per_version,
    }
    for version in versions:
        metrics["versions"][version] = aggregate(per_version[version])
        metrics["by_category"][version] = aggregate_by_category(per_version[version], scene_to_category)

    write_json(Path(args.metrics_out), metrics)
    print(f"Wrote {args.manifest_out}")
    print(f"Wrote {args.metrics_out}")
    for version in versions:
        agg = metrics["versions"][version]
        print(
            version,
            f"matched={agg['matched_object_count']}/{agg['gt_object_count']}",
            f"parent_acc={agg['scene_graph_parent_accuracy_gt']}",
            f"rot_auc60={agg['rotation_auc60_aligned']}",
            f"trans_auc05={agg['translation_auc05_aligned']}",
        )
        # Paper-compatible Table 3
        for subset_key, label in [("primary_metrics", "Primary"), ("secondary_metrics", "Secondary")]:
            sub = agg.get(subset_key, {})
            if sub:
                print(f"  {label}: recovery={sub.get('object_recovery')}, "
                      f"parent_acc={sub.get('parent_accuracy')}, "
                      f"rot_auc={sub.get('rotation_auc60_aligned')}, "
                      f"trans_auc={sub.get('translation_auc05_aligned')}")
    if failures:
        print(f"Failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
