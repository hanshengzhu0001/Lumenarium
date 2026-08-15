"""Attribute final physical regressions to accepted SceneProof trials."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any


FAMILIES = ("collision", "support", "plane", "semantic")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def placement(root: Path, scene: str, version: str) -> Path:
    directory = root / f"{scene}_{version}_result" / "S4_layout_refinement"
    matches = sorted(directory.glob("*_placement_info_s4.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one placement in {directory}, found {len(matches)}"
        )
    return matches[0]


def number(value: str | None) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def matrix_delta(first: Any, second: Any) -> float | None:
    if not isinstance(first, list) or not isinstance(second, list):
        return None
    try:
        values = [
            (float(a) - float(b)) ** 2
            for row_a, row_b in zip(first, second)
            for a, b in zip(row_a, row_b)
        ]
    except (TypeError, ValueError):
        return None
    return math.sqrt(sum(values))


def aggregate_counterfactual(
    physical: dict[str, Any],
    scenes: list[str],
    control: str,
    candidate: str,
    rolled_back: set[str],
) -> dict[str, Any]:
    chosen = {
        scene: control if scene in rolled_back else candidate
        for scene in scenes
    }
    macros: list[float] = []
    families: dict[str, dict[str, float]] = {}
    for scene in scenes:
        data = physical["versions"][chosen[scene]]["scenes"][scene]
        value = data.get("headline_macro_realizability")
        if value is not None:
            macros.append(float(value))
    for family in FAMILIES:
        total = 0
        weighted = 0.0
        for scene in scenes:
            entry = physical["versions"][chosen[scene]]["scenes"][scene][
                "families"
            ][family]
            count = int(entry.get("n", 0))
            score = entry.get("score")
            if count and score is not None:
                total += count
                weighted += count * float(score)
        families[family] = {
            "n": total,
            "score": weighted / total if total else None,
        }
    return {
        "rolled_back_scenes": sorted(rolled_back),
        "physical_macro": sum(macros) / len(macros) if macros else None,
        "families": families,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-results", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--object-csv", type=Path, required=True)
    parser.add_argument("--control-version", required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    physical = load(args.physical)
    aggregate = load(args.aggregate)
    scenes = sorted(aggregate.get("scenes", {}))
    accepted_scenes = sorted(
        scene
        for scene, row in aggregate.get("scenes", {}).items()
        if row.get("accepted_nonzero")
    )

    scene_reports: dict[str, Any] = {}
    changed_objects: set[tuple[str, str]] = set()
    for scene in scenes:
        base_scene = physical["versions"][args.control_version]["scenes"][scene]
        cand_scene = physical["versions"][args.candidate_version]["scenes"][scene]
        base_doc = load(placement(args.saved_results, scene, args.control_version))
        cand_doc = load(placement(args.saved_results, scene, args.candidate_version))
        pose_changes: list[dict[str, Any]] = []
        for object_id in sorted(
            set(base_doc.get("obj_info", {}))
            & set(cand_doc.get("obj_info", {}))
        ):
            delta = matrix_delta(
                base_doc["obj_info"][object_id].get("pose_matrix_for_blender"),
                cand_doc["obj_info"][object_id].get("pose_matrix_for_blender"),
            )
            if delta is not None and delta > 1e-7:
                changed_objects.add((scene, object_id))
                pose_changes.append(
                    {"object_id": object_id, "matrix_frobenius_delta": delta}
                )
        scene_reports[scene] = {
            "accepted": scene in accepted_scenes,
            "trial": aggregate["scenes"][scene].get("selected_trial_kind"),
            "physical_macro_delta": float(
                cand_scene["headline_macro_realizability"]
                - base_scene["headline_macro_realizability"]
            ),
            "family_deltas": {
                family: float(
                    cand_scene["families"][family]["score"]
                    - base_scene["families"][family]["score"]
                )
                if cand_scene["families"][family]["score"] is not None
                and base_scene["families"][family]["score"] is not None
                else None
                for family in FAMILIES
            },
            "changed_object_count": len(pose_changes),
            "changed_objects": sorted(
                pose_changes,
                key=lambda row: row["matrix_frobenius_delta"],
                reverse=True,
            ),
        }

    object_rows: dict[tuple[str, str, str], dict[str, str]] = {}
    with args.object_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            object_rows[(row["scene"], row["object_id"], row["version"])] = row
    object_deltas: list[dict[str, Any]] = []
    fields = (
        "local_realizability",
        "collision_overlap_fraction",
        "support_contact_gap_m",
        "support_containment_error_m",
        "support_footprint_overlap_ratio",
        "plane_contact_gap_m",
        "plane_orientation_error_deg",
        "semantic_error",
    )
    identities = sorted({(scene, object_id) for scene, object_id, _ in object_rows})
    for scene, object_id in identities:
        base = object_rows.get((scene, object_id, args.control_version))
        cand = object_rows.get((scene, object_id, args.candidate_version))
        if base is None or cand is None:
            continue
        deltas = {}
        for field in fields:
            first, second = number(base.get(field)), number(cand.get(field))
            deltas[field] = (
                second - first if first is not None and second is not None else None
            )
        local_delta = deltas["local_realizability"]
        if (scene, object_id) in changed_objects or (
            local_delta is not None and abs(local_delta) > 1e-9
        ):
            object_deltas.append(
                {
                    "scene": scene,
                    "object_id": object_id,
                    "pose_changed": (scene, object_id) in changed_objects,
                    "deltas": deltas,
                }
            )
    object_deltas.sort(
        key=lambda row: (
            row["deltas"]["local_realizability"] is None,
            row["deltas"]["local_realizability"]
            if row["deltas"]["local_realizability"] is not None
            else 0.0,
        )
    )

    reference = aggregate_counterfactual(
        physical, scenes, args.control_version, args.candidate_version, set(scenes)
    )
    counterfactuals: list[dict[str, Any]] = []
    for size in range(len(accepted_scenes) + 1):
        for subset in itertools.combinations(accepted_scenes, size):
            row = aggregate_counterfactual(
                physical,
                scenes,
                args.control_version,
                args.candidate_version,
                set(subset),
            )
            gates = {
                "physical_macro": row["physical_macro"]
                >= reference["physical_macro"] - args.margin,
                **{
                    family: row["families"][family]["score"]
                    >= reference["families"][family]["score"] - args.margin
                    for family in FAMILIES
                    if row["families"][family]["score"] is not None
                    and reference["families"][family]["score"] is not None
                },
            }
            row["gates"] = gates
            row["passed"] = all(gates.values())
            counterfactuals.append(row)
    passing = [row for row in counterfactuals if row["passed"]]
    minimum_rollback = min(
        (len(row["rolled_back_scenes"]) for row in passing),
        default=None,
    )
    minimal = [
        row
        for row in passing
        if len(row["rolled_back_scenes"]) == minimum_rollback
    ]

    result = {
        "schema_version": "sceneproof_postsim_responsibility_audit_v1",
        "control_version": args.control_version,
        "candidate_version": args.candidate_version,
        "accepted_scenes": accepted_scenes,
        "reference": reference,
        "scenes": scene_reports,
        "worst_object_deltas": object_deltas[:30],
        "minimal_passing_scene_rollbacks": minimal,
        "all_counterfactuals": counterfactuals,
        "decision": (
            "implement_postsim_component_certificate_with_scoped_rollback"
            if minimal
            else "candidate_family_not_recoverable_by_accepted_scene_rollback"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "SCENEPROOF POST-SIMULATION RESPONSIBILITY AUDIT",
        "=" * 72,
        f"accepted_scenes={accepted_scenes}",
        "",
    ]
    for scene, row in scene_reports.items():
        lines.append(
            f"{scene}: accepted={row['accepted']} trial={row['trial']} "
            f"macro_delta={row['physical_macro_delta']:+.6f} "
            f"families={row['family_deltas']} changed={row['changed_object_count']}"
        )
    lines.extend(["", "Minimal passing whole-scene rollback oracle:"])
    for row in minimal:
        family_text = ", ".join(
            f"{name}: {float(values['score']):.6f}"
            if values.get("score") is not None
            else f"{name}: n/a"
            for name, values in row["families"].items()
        )
        lines.append(
            f"  rollback={row['rolled_back_scenes']} "
            f"macro={row['physical_macro']:.6f} "
            f"families={{{family_text}}}"
        )
    lines.extend(["", "Worst object deltas:"])
    for row in object_deltas[:20]:
        lines.append(
            f"  {row['scene']}/{row['object_id']} changed={row['pose_changed']} "
            f"local_delta={row['deltas']['local_realizability']} "
            f"support_overlap_delta={row['deltas']['support_footprint_overlap_ratio']} "
            f"semantic_delta={row['deltas']['semantic_error']}"
        )
    lines.extend(["", f"DECISION={result['decision']}"])
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.report}")
    print(f"ACCEPTED_SCENES={accepted_scenes}")
    print(f"MINIMAL_PASSING_ROLLBACKS={[row['rolled_back_scenes'] for row in minimal]}")
    for scene, row in scene_reports.items():
        print(
            f"{scene} accepted={row['accepted']} "
            f"macro_delta={row['physical_macro_delta']:+.6f} "
            f"family_deltas={row['family_deltas']}"
        )
    print(f"DECISION={result['decision']}")


if __name__ == "__main__":
    main()
