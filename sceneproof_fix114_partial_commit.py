#!/usr/bin/env python3
"""Greedily retain the largest useful family-wise noninferior Fix114 subset."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from eval_physical_realizability import find_geometry_snapshot, find_s4

FAMILIES = ("collision", "support", "plane", "semantic")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def materialize(base, candidate, retained):
    result = copy.deepcopy(base)
    for object_id in retained:
        if object_id in candidate.get("obj_info", {}):
            result["obj_info"][object_id] = copy.deepcopy(
                candidate["obj_info"][object_id]
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-results", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--geometry-version", required=True)
    parser.add_argument("--baseline-version", required=True)
    parser.add_argument("--visual-version", required=True)
    parser.add_argument("--target-version", required=True)
    parser.add_argument("--transactions", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    scenes = [x.strip() for x in args.scenes.read_text().splitlines() if x.strip()]
    documents = {}
    active = []
    initially_rejected = {}
    for scene in scenes:
        base = load(find_s4(args.saved_results, scene, args.baseline_version))
        visual = load(find_s4(args.saved_results, scene, args.visual_version))
        tx = load(args.transactions / f"{scene}.json")
        accepted = [x for x in tx.get("accepted_object_ids", []) if x in visual.get("obj_info", {})]
        rejected = [r.get("object_id") for r in tx.get("transactions", []) if not r.get("accepted")]
        initially_rejected[scene] = [x for x in rejected if isinstance(x, str)]
        documents[scene] = (base, visual)
        active.extend((scene, object_id) for object_id in accepted)

    cache = {}
    def evaluate(retained_pairs):
        key = tuple(sorted(retained_pairs))
        if key in cache:
            return cache[key]
        retained_by_scene = {scene: set() for scene in scenes}
        for scene, object_id in retained_pairs:
            retained_by_scene[scene].add(object_id)
        with tempfile.TemporaryDirectory(prefix="sceneproof_fix114_subset_") as directory:
            root = Path(directory)
            manifest = root / "manifest.txt"
            manifest.write_text("\n".join(scenes) + "\n")
            for scene in scenes:
                geometry = find_geometry_snapshot(args.saved_results, scene, args.geometry_version)
                geometry_dir = root / f"{scene}_{args.geometry_version}_result" / "S4_layout_refinement"
                geometry_dir.mkdir(parents=True)
                shutil.copy2(geometry, geometry_dir / geometry.name)
                for version, doc in (
                    (args.baseline_version, documents[scene][0]),
                    (args.target_version, materialize(*documents[scene], retained_by_scene[scene])),
                ):
                    folder = root / f"{scene}_{version}_result" / "S4_layout_refinement"
                    folder.mkdir(parents=True)
                    (folder / f"{scene}_{version}_placement_info_s4.json").write_text(
                        json.dumps(doc), encoding="utf-8"
                    )
            metrics = root / "metrics.json"
            process = subprocess.run(
                [sys.executable, "eval_physical_realizability.py", "--saved-results", str(root),
                 "--scenes", str(manifest), "--versions", f"{args.baseline_version},{args.target_version}",
                 "--geometry-version", args.geometry_version, "--baseline-version", args.baseline_version,
                 "--metrics-out", str(metrics),
                 "--scene-csv", str(root / "scenes.csv"),
                 "--object-csv", str(root / "objects.csv"),
                 "--report-out", str(root / "report.txt")],
                capture_output=True, text=True,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    "subset physical evaluator failed "
                    f"(rc={process.returncode}):\n{process.stdout}\n{process.stderr}"
                )
            data = load(metrics)["versions"]
            before = data[args.baseline_version]["aggregate"]
            after = data[args.target_version]["aggregate"]
            deltas = {}
            for family in FAMILIES:
                first = before.get("families", {}).get(family, {}).get("score")
                second = after.get("families", {}).get(family, {}).get("score")
                deltas[family] = None if first is None or second is None else second - first
            deltas["physical_macro"] = after["headline_macro_realizability"] - before["headline_macro_realizability"]
        cache[key] = deltas
        return deltas

    def passed(deltas):
        return all(value is None or value >= -args.tolerance for value in deltas.values())

    def rank(deltas):
        values = [value for value in deltas.values() if value is not None]
        return (min(values, default=0.0), sum(values), len(values))

    retained = set(active)
    trace = []
    while retained:
        current = evaluate(retained)
        trace.append({"retained": sorted(retained), "deltas": current})
        if passed(current):
            break
        trials = []
        for pair in sorted(retained):
            subset = retained - {pair}
            deltas = evaluate(subset)
            trials.append((rank(deltas), pair, deltas))
        _, removed, removal_deltas = max(trials, key=lambda row: row[0])
        retained.remove(removed)
        trace[-1]["removed"] = list(removed)
        trace[-1]["removal_deltas"] = removal_deltas
    final_deltas = evaluate(retained)
    if not passed(final_deltas):
        retained.clear()
        final_deltas = evaluate(retained)

    retained_by_scene = {scene: set() for scene in scenes}
    for scene, object_id in retained:
        retained_by_scene[scene].add(object_id)
    rolled_back = sorted(set(active) - retained)
    unresolved = {scene: sorted(set(initially_rejected[scene]) | {o for s, o in rolled_back if s == scene}) for scene in scenes}
    unresolved = {scene: rows for scene, rows in unresolved.items() if rows}
    for scene in scenes:
        output = materialize(*documents[scene], retained_by_scene[scene])
        output["sceneproof_fix114_partial_commit"] = {
            "visual_version": args.visual_version,
            "retained_object_ids": sorted(retained_by_scene[scene]),
            "unresolved_object_ids": unresolved.get(scene, []),
        }
        folder = args.saved_results / f"{scene}_{args.target_version}_result" / "S4_layout_refinement"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{scene}_{args.target_version}_placement_info_s4.json").write_text(
            json.dumps(output, indent=2), encoding="utf-8"
        )
    report = {
        "schema_version": "sceneproof_fix114_maximal_noninferior_subset_v1",
        "passed": passed(final_deltas), "baseline_version": args.baseline_version,
        "visual_version": args.visual_version, "target_version": args.target_version,
        "initial_candidates": [list(x) for x in sorted(active)],
        "retained": [list(x) for x in sorted(retained)],
        "rolled_back": [list(x) for x in rolled_back], "unresolved": unresolved,
        "family_deltas": final_deltas, "trace": trace,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASSED={report['passed']} INITIAL={len(active)} RETAINED={len(retained)} ROLLED_BACK={len(rolled_back)} UNRESOLVED={sum(map(len,unresolved.values()))}")
    print(f"FAMILY_DELTAS={json.dumps(final_deltas, sort_keys=True)}")
    print(f"FIX114_PARTIAL_COMMIT={args.out.resolve()}")


if __name__ == "__main__":
    main()
