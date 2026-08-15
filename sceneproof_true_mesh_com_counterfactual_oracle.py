#!/usr/bin/env python3
"""Audit whether COM witnesses justify local rollback to the smooth incumbent."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
from typing import Any

from sceneproof_postsim_component_certifier import (
    changed_objects,
    evaluator_args,
    family_gates,
    relation_neighborhoods,
    rollback_poses,
    scoped_component,
)
from modules._sceneproof_support_stability import (
    ungrounded_cyclic_components,
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_vector(metrics: dict[str, Any]) -> dict[str, float | None]:
    result = {
        family: metrics.get("families", {}).get(family, {}).get("score")
        for family in ("collision", "support", "plane", "semantic")
    }
    result["physical_macro"] = metrics.get("headline_macro_realizability")
    return result


def metric_deltas(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, float | None]:
    a, b = metric_vector(first), metric_vector(second)
    return {
        key: (
            float(b[key]) - float(a[key])
            if a[key] is not None and b[key] is not None
            else None
        )
        for key in a
    }


def unique_components(
    witnesses: set[str], adjacency: dict[str, set[str]], changed: set[str]
) -> list[set[str]]:
    result: list[set[str]] = []
    seen: set[tuple[str, ...]] = set()
    for witness in sorted(witnesses & changed):
        component = scoped_component(witness, adjacency, changed) & changed
        key = tuple(sorted(component))
        if key and key not in seen:
            seen.add(key)
            result.append(set(key))
    return result


def main() -> None:
    from eval_physical_realizability import (
        evaluate_scene,
        find_geometry_snapshot,
        find_s4,
        load_json,
        validate_geometry_snapshot,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-results", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--geometry-version", default="v4_deepsearch")
    parser.add_argument("--incumbent-version", required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--com-audit-root", type=Path, required=True)
    parser.add_argument("--responsibility", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.005)
    parser.add_argument("--meaningful-gain-tolerance", type=float, default=1e-6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source_path = find_geometry_snapshot(
        args.saved_results, args.scene, args.geometry_version
    )
    source = load_json(source_path)
    validate_geometry_snapshot(source, source_path)
    incumbent = load_json(
        find_s4(args.saved_results, args.scene, args.incumbent_version)
    )
    candidate = load_json(
        find_s4(args.saved_results, args.scene, args.candidate_version)
    )
    final_audit = load(
        args.com_audit_root
        / f"{args.scene}__{args.candidate_version}.json"
    )
    responsibility = load(args.responsibility)

    changed = changed_objects(incumbent, candidate)
    adjacency = relation_neighborhoods(candidate.get("obj_info", {}))
    audit_objects = final_audit.get("objects", {})
    support_graph = {}
    for object_id, row in audit_objects.items():
        if row.get("status") != "measured":
            continue
        edges = set(row.get("supporter_ids", []))
        parent = row.get("declared_parent_id")
        if isinstance(parent, str):
            edges.add(parent)
        support_graph[object_id] = edges
    grounded_nodes = {
        object_id
        for object_id in candidate.get("obj_info", {})
        if re.match(r"^(floor|ground)_\d+$", object_id)
    }
    _, grounded_cycles = ungrounded_cyclic_components(
        support_graph, grounded_nodes
    )
    grounded_cycle_nodes = {
        object_id for component in grounded_cycles for object_id in component
    }
    abstained = {
        object_id
        for object_id, row in audit_objects.items()
        if row.get("certificate_status") != "certified"
        and object_id not in grounded_cycle_nodes
    }
    unstable = {
        object_id
        for object_id, row in audit_objects.items()
        if row.get("certificate_status") == "certified"
        and row.get("stability_class") == "unstable"
    }
    regressed = {
        row["object_id"]
        for row in responsibility.get("worst_margin_regressions", [])
        if row.get("scene") == args.scene
    }
    witness_groups = {
        "abstained": abstained,
        "unstable": unstable,
        "margin_regressed": regressed,
    }
    components = {
        name: unique_components(witnesses, adjacency, changed)
        for name, witnesses in witness_groups.items()
    }

    proposals: dict[tuple[str, ...], set[str]] = {}
    proposal_sources: dict[tuple[str, ...], set[str]] = {}
    for name, rows in components.items():
        for component in rows:
            key = tuple(sorted(component))
            proposals[key] = component
            proposal_sources.setdefault(key, set()).add(name)
        union = set().union(*rows) if rows else set()
        if union:
            key = tuple(sorted(union))
            proposals[key] = union
            proposal_sources.setdefault(key, set()).add(f"all_{name}")
    fail_closed_components = (
        components.get("abstained", []) + components.get("unstable", [])
    )
    required_fail_closed = (
        set().union(*fail_closed_components) if fail_closed_components else set()
    )
    if required_fail_closed:
        key = tuple(sorted(required_fail_closed))
        proposals[key] = required_fail_closed
        proposal_sources.setdefault(key, set()).add("all_fail_closed")
    regression_components = components.get("margin_regressed", [])
    all_regressions = (
        set().union(*regression_components) if regression_components else set()
    )
    if required_fail_closed and all_regressions:
        combined = required_fail_closed | all_regressions
        key = tuple(sorted(combined))
        proposals[key] = combined
        proposal_sources.setdefault(key, set()).add(
            "fail_closed_plus_all_regressions"
        )
    all_components = set().union(*proposals.values()) if proposals else set()
    if all_components:
        key = tuple(sorted(all_components))
        proposals[key] = all_components
        proposal_sources.setdefault(key, set()).add("all_witnesses")

    eval_args = evaluator_args()
    incumbent_metrics, _ = evaluate_scene(source, incumbent, eval_args)
    candidate_metrics, _ = evaluate_scene(source, candidate, eval_args)
    trials = []
    for key in sorted(proposals, key=lambda row: (len(row), row)):
        selected = copy.deepcopy(candidate)
        rollback_poses(selected, incumbent, proposals[key])
        metrics, _ = evaluate_scene(source, selected, eval_args)
        safe, gates = family_gates(
            incumbent_metrics, metrics, args.margin
        )
        gain = metric_deltas(candidate_metrics, metrics)
        trials.append(
            {
                "rollback_object_ids": list(key),
                "sources": sorted(proposal_sources[key]),
                "safe_against_incumbent": safe,
                "gates": gates,
                "delta_vs_candidate": gain,
                "positive_macro_oracle_gain": bool(
                    gain["physical_macro"] is not None
                    and gain["physical_macro"] > 1e-12
                ),
            }
        )

    passing = [
        row
        for row in trials
        if row["safe_against_incumbent"]
        and row["positive_macro_oracle_gain"]
    ]
    passing.sort(
        key=lambda row: (
            float(row["delta_vs_candidate"]["physical_macro"]),
            -len(row["rollback_object_ids"]),
        ),
        reverse=True,
    )
    eligible = [
        row
        for row in passing
        if required_fail_closed.issubset(row["rollback_object_ids"])
    ]
    selected = None
    if eligible:
        best_gain = max(
            float(row["delta_vs_candidate"]["physical_macro"])
            for row in eligible
        )
        near_best = [
            row
            for row in eligible
            if float(row["delta_vs_candidate"]["physical_macro"])
            >= best_gain - args.meaningful_gain_tolerance
        ]
        selected = min(
            near_best,
            key=lambda row: (
                len(row["rollback_object_ids"]),
                -float(row["delta_vs_candidate"]["physical_macro"]),
            ),
        )
    result = {
        "schema_version": "sceneproof_true_mesh_com_counterfactual_oracle_v1",
        "scene": args.scene,
        "incumbent_version": args.incumbent_version,
        "candidate_version": args.candidate_version,
        "changed_objects": sorted(changed),
        "witnesses": {
            key: sorted(value & changed) for key, value in witness_groups.items()
        },
        "grounded_cycle_object_ids": sorted(grounded_cycle_nodes),
        "required_fail_closed_rollback_object_ids": sorted(
            required_fail_closed
        ),
        "meaningful_gain_tolerance": args.meaningful_gain_tolerance,
        "incumbent_metrics": metric_vector(incumbent_metrics),
        "candidate_metrics": metric_vector(candidate_metrics),
        "trials": trials,
        "selected_oracle": selected,
        "implementation_authorized": selected is not None,
        "decision": (
            "implement_scoped_com_fail_closed_rollback"
            if selected is not None
            else "retain_fix61_and_abstain"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.out.resolve()}")
    print(
        f"SCENE={args.scene} CHANGED={len(changed)} "
        f"ABSTAINED_WITNESSES={len(abstained & changed)} "
        f"REGRESSED_WITNESSES={len(regressed & changed)} "
        f"TRIALS={len(trials)} PASSING={len(passing)} "
        f"DECISION={result['decision']}"
    )
    if selected is not None:
        print(
            "SELECTED_ROLLBACK="
            + ",".join(selected["rollback_object_ids"])
            + " MACRO_GAIN="
            + f"{selected['delta_vs_candidate']['physical_macro']:.9f}"
        )


if __name__ == "__main__":
    main()
