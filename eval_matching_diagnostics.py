#!/usr/bin/env python3
"""Diagnose where objects disappear before GT matching.

This is a companion to eval_gt_metrics.py. It focuses on pipeline attrition:
S1 detections -> S1 scene graph -> S2 retrieval -> S4 placement -> GT matches.
"""

from __future__ import annotations

import json
import re
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSIONS = ("v1", "v3")
STRUCTURAL = ("floor", "wall", "ceiling", "ground", "scene_camera")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def is_structural(name: str) -> bool:
    low = name.lower()
    return low.startswith(STRUCTURAL)


def is_anonymous(name: str) -> bool:
    return re.match(r"^object(?:_\d+)+$", name) is not None


def normalize_category(value: Any, fallback: str = "unknown") -> str:
    raw = str(value or fallback or "unknown").lower().replace(".fbx", "")
    raw = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", raw)
    parts = [p for p in re.split(r"[^a-z0-9]+", raw) if p]
    while parts and parts[-1].isdigit():
        parts.pop()
    return "_".join(parts) if parts else "unknown"


def top_counts(counter: Counter[str], limit: int = 20) -> dict[str, int]:
    return {key: count for key, count in counter.most_common(limit)}


def sum_count_dicts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.get(key, {}))
    return top_counts(counts)


def visible_gt_ids(metric_scene: dict[str, Any], gt_objects: dict[str, Any]) -> list[str]:
    details = metric_scene.get("visibility", {}).get("details", {})
    if not isinstance(details, dict):
        return list(gt_objects)
    visible = [gt_id for gt_id, info in details.items() if isinstance(info, dict) and info.get("visible")]
    return visible or list(gt_objects)


def count_keys(path: Path, obj_info: bool = False) -> tuple[list[str], dict[str, Any] | None]:
    if not path.exists():
        return [], None
    data = load_json(path)
    if obj_info:
        data = data.get("obj_info", {})
    if not isinstance(data, dict):
        return [], None
    return [k for k in data if not is_structural(k)], data


def scene_ids(demo_dir: Path) -> list[str]:
    return sorted(path.name[: -len("_v1.png")] for path in demo_dir.glob("*_v1.png"))


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


def safe_ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "scene_count": len(rows),
        "all_gt_object_count": sum(r.get("all_gt_object_count", r["gt_object_count"]) for r in rows),
        "visible_gt_object_count": sum(r.get("visible_gt_object_count", r["gt_object_count"]) for r in rows),
        "filtered_gt_object_count": sum(r.get("filtered_gt_object_count", 0) for r in rows),
        "gt_object_count": sum(r["gt_object_count"] for r in rows),
        "s1_bbox_count": sum(r["s1_bbox_count"] for r in rows),
        "s1_scene_graph_count": sum(r["s1_scene_graph_count"] for r in rows),
        "s2_retrieval_count": sum(r["s2_retrieval_count"] for r in rows),
        "s4_object_count": sum(r["s4_object_count"] for r in rows),
        "matched_object_count": sum(r["matched_object_count"] for r in rows),
        "anonymous_s1_count": sum(r["anonymous_s1_count"] for r in rows),
        "anonymous_scene_graph_count": sum(r["anonymous_scene_graph_count"] for r in rows),
        "scene_graph_missing_from_retrieval": sum(r["scene_graph_missing_from_retrieval"] for r in rows),
        "anonymous_scene_graph_missing_from_retrieval": sum(
            r["anonymous_scene_graph_missing_from_retrieval"] for r in rows
        ),
    }
    out["s4_per_gt"] = safe_ratio(out["s4_object_count"], out["gt_object_count"])
    out["matched_per_gt"] = safe_ratio(out["matched_object_count"], out["gt_object_count"])
    out["s2_per_s1"] = safe_ratio(out["s2_retrieval_count"], out["s1_bbox_count"])
    out["s4_per_s1"] = safe_ratio(out["s4_object_count"], out["s1_bbox_count"])
    out["anonymous_s1_ratio"] = safe_ratio(out["anonymous_s1_count"], out["s1_bbox_count"])
    out["anonymous_graph_ratio"] = safe_ratio(out["anonymous_scene_graph_count"], out["s1_scene_graph_count"])
    out["missing_graph_anonymous_share"] = safe_ratio(
        out["anonymous_scene_graph_missing_from_retrieval"], out["scene_graph_missing_from_retrieval"]
    )
    out["unmatched_visible_gt_categories"] = sum_count_dicts(rows, "unmatched_visible_gt_categories")
    out["unmatched_s4_pred_categories"] = sum_count_dicts(rows, "unmatched_s4_pred_categories")
    out["matched_gt_categories"] = sum_count_dicts(rows, "matched_gt_categories")
    return out


def diagnosis_summary(overall: dict[str, Any]) -> str:
    if not overall:
        return "No versions were evaluated."
    parts = []
    for version, row in overall.items():
        anon_ratio = row.get("anonymous_s1_ratio") or 0.0
        s2_per_s1 = row.get("s2_per_s1") or 0.0
        matched_per_gt = row.get("matched_per_gt") or 0.0
        if anon_ratio >= 0.10 and s2_per_s1 < 0.90:
            issue = "S1 anonymous labels are still causing S2 attrition"
        elif s2_per_s1 >= 0.90 and matched_per_gt < 0.30:
            issue = "S1/S2 retention is healthy; remaining loss is GT/category/matcher coverage"
        else:
            issue = "attrition is mixed across S1/S2/S4 and GT matching"
        parts.append(f"{version}: {issue}")
    return "; ".join(parts) + "."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="asset_data/imaginarium_3d_scene_layout_dataset")
    parser.add_argument("--saved-results", default="saved_results")
    parser.add_argument("--demo-dir", default="demo")
    parser.add_argument("--metrics", default="eval_gt_metrics.json")
    parser.add_argument("--out", default="eval_matching_diagnostics.json")
    parser.add_argument("--scenes", default="", help="Comma-separated scene ids or a text file with one scene id per line")
    parser.add_argument("--versions", default=",".join(VERSIONS), help="Comma-separated versions to evaluate, e.g. v1,v3 or v3")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    saved_results = Path(args.saved_results)
    metrics_path = Path(args.metrics)
    versions = tuple(v.strip() for v in args.versions.split(",") if v.strip())
    if not versions:
        raise SystemExit("--versions must include at least one version")
    metrics = load_json(metrics_path) if metrics_path.exists() else {"scenes": {v: {} for v in versions}}

    per_version_rows: dict[str, list[dict[str, Any]]] = {v: [] for v in versions}
    by_scene: dict[str, dict[str, Any]] = {}

    scenes = scene_ids(Path(args.demo_dir))
    selected_scenes = read_scene_filter(args.scenes)
    if selected_scenes is not None:
        selected = set(selected_scenes)
        scenes = [scene for scene in scenes if scene in selected]

    for scene in scenes:
        mp = meta_path(dataset_dir, scene)
        category = mp.parent.parent.name
        gt_objects = {
            k: v
            for k, v in load_json(mp).get("objects", {}).items()
            if isinstance(v, dict) and (not v.get("type") or v.get("type") == "MESH")
        }
        by_scene[scene] = {"category": category, "versions": {}}

        for version in versions:
            base = saved_results / f"{scene}_{version}_result"
            s1_keys, _ = count_keys(base / "S1_scene_parsing_results/object_bboxes_json.json")
            graph_keys, graph = count_keys(base / "S1_scene_parsing_results/scene_graph_result_final.json")
            retrieval_keys, retrieval = count_keys(base / "S2_3d_retrieval_results/retrieval_results_final.json")
            s4_keys, _ = count_keys(
                base / "S4_layout_refinement" / f"{scene}_{version}_placement_info_s4.json",
                obj_info=True,
            )

            retrieval_set = set(retrieval_keys)
            missing_from_retrieval = [k for k in graph_keys if k not in retrieval_set]
            empty_retrieval = []
            if isinstance(retrieval, dict):
                empty_retrieval = [k for k in retrieval_keys if not retrieval.get(k)]

            metric_scene = metrics.get("scenes", {}).get(version, {}).get(scene, {})
            metric_gt_count = metric_scene.get("gt_object_count", len(gt_objects))
            matches = metric_scene.get("matches", [])
            matched_gt = {m.get("gt_id") for m in matches if isinstance(m, dict)}
            matched_pred = {m.get("pred_id") for m in matches if isinstance(m, dict)}
            visible_ids = visible_gt_ids(metric_scene, gt_objects)
            unmatched_gt = [gt_id for gt_id in visible_ids if gt_id not in matched_gt and gt_id in gt_objects]
            unmatched_pred = [pred_id for pred_id in s4_keys if pred_id not in matched_pred]

            unmatched_gt_categories = Counter(
                normalize_category(
                    gt_objects[gt_id].get("class_en")
                    or gt_objects[gt_id].get("caption_en")
                    or gt_objects[gt_id].get("fbx_name")
                    or gt_id
                )
                for gt_id in unmatched_gt
            )
            unmatched_pred_categories = Counter(normalize_category(pred_id) for pred_id in unmatched_pred)
            matched_gt_categories = Counter(
                normalize_category(
                    gt_objects[gt_id].get("class_en")
                    or gt_objects[gt_id].get("caption_en")
                    or gt_objects[gt_id].get("fbx_name")
                    or gt_id
                )
                for gt_id in matched_gt
                if gt_id in gt_objects
            )
            row = {
                "scene": scene,
                "category": category,
                "version": version,
                "all_gt_object_count": metric_scene.get("all_gt_object_count", len(gt_objects)),
                "visible_gt_object_count": metric_scene.get("visible_gt_object_count", metric_gt_count),
                "filtered_gt_object_count": metric_scene.get("filtered_gt_object_count", 0),
                "gt_object_count": metric_gt_count,
                "s1_bbox_count": len(s1_keys),
                "s1_scene_graph_count": len(graph_keys),
                "s2_retrieval_count": len(retrieval_keys),
                "s4_object_count": len(s4_keys),
                "matched_object_count": metric_scene.get("matched_object_count", 0),
                "anonymous_s1_count": sum(is_anonymous(k) for k in s1_keys),
                "anonymous_scene_graph_count": sum(is_anonymous(k) for k in graph_keys),
                "scene_graph_missing_from_retrieval": len(missing_from_retrieval),
                "anonymous_scene_graph_missing_from_retrieval": sum(is_anonymous(k) for k in missing_from_retrieval),
                "empty_retrieval_count": len(empty_retrieval),
                "s2_per_s1": safe_ratio(len(retrieval_keys), len(s1_keys)),
                "s4_per_s1": safe_ratio(len(s4_keys), len(s1_keys)),
                "matched_per_gt": safe_ratio(metric_scene.get("matched_object_count", 0), metric_gt_count),
                "anonymous_s1_ratio": safe_ratio(sum(is_anonymous(k) for k in s1_keys), len(s1_keys)),
                "missing_from_retrieval_sample": missing_from_retrieval[:20],
                "empty_retrieval_sample": empty_retrieval[:20],
                "unmatched_visible_gt_count": len(unmatched_gt),
                "unmatched_s4_pred_count": len(unmatched_pred),
                "unmatched_visible_gt_categories": top_counts(unmatched_gt_categories),
                "unmatched_s4_pred_categories": top_counts(unmatched_pred_categories),
                "matched_gt_categories": top_counts(matched_gt_categories),
                "unmatched_visible_gt_sample": unmatched_gt[:20],
                "unmatched_s4_pred_sample": unmatched_pred[:20],
            }
            per_version_rows[version].append(row)
            by_scene[scene]["versions"][version] = row

    by_category: dict[str, dict[str, Any]] = {}
    overall: dict[str, Any] = {}
    for version, rows in per_version_rows.items():
        overall[version] = summarize(rows)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["category"]].append(row)
        by_category[version] = {cat: summarize(cat_rows) for cat, cat_rows in sorted(grouped.items())}

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": diagnosis_summary(overall),
        "overall": overall,
        "by_category": by_category,
        "scenes": by_scene,
        "notes": {
            "anonymous_object_rule": "object_N labels normalize to class 'object', which has no retrieval class, so S2 skips them.",
            "gt_denominator": "Uses metrics gt_object_count when available, so calibrated runs can report visible-GT denominator.",
            "unmatched_categories": "Visible GT categories are based on class_en/caption/fbx fallback; prediction categories are normalized S4 object ids.",
        },
    }
    write_json(Path(args.out), report)

    print(f"Wrote {args.out}")
    for version in versions:
        o = overall[version]
        print(
            version,
            f"S1={o['s1_bbox_count']}",
            f"S2={o['s2_retrieval_count']} ({pct(o['s2_per_s1'])})",
            f"S4={o['s4_object_count']} ({pct(o['s4_per_s1'])})",
            f"matched={o['matched_object_count']}/{o['gt_object_count']} ({pct(o['matched_per_gt'])})",
            f"anon_s1={pct(o['anonymous_s1_ratio'])}",
            f"missing_is_anon={pct(o['missing_graph_anonymous_share'])}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
