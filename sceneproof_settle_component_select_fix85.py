#!/usr/bin/env python3
"""SceneProof Fix85: exact per-component attribution of a bulk gravity settle.

Problem
-------
A bulk Bullet settle proposes one new pose per object that moved.  Deciding
which of those poses to keep is a subset-selection problem over ``N`` proposed
pose changes.  Two obvious strategies both fail:

* *Per-object promotion* runs one full scene evaluation per proposal.Cost is
  ``N`` evaluations per scene (``N`` up to 109 here), which is prohibitive.
* *Batch promotion* runs a single evaluation but returns a single scene-level
  verdict, so a handful of bad poses veto every good pose in the scene.

This module removes the trade-off using a structural property of the evaluator
rather than an approximation.

The structural property
-----------------------
``eval_physical_realizability.py`` computes every family score as an unweighted
mean of per-object terms,

    S_f = (1 / N_f) * sum over i in O_f of s_f(i),

where ``O_f`` is the set of objects contributing a term to family ``f``.  ``O_f``
is fixed by the frozen source geometry and the declared support graph, never by
pose, so ``N_f`` is a constant.  The evaluator now writes each ``s_f(i)``
verbatim as a ``{family}_term`` column, so no term has to be re-derived from raw
errors and tolerances.

Each term has a bounded dependency set:``s_f(i)`` reads the pose of ``i``, of
``i``'s declared support parent, of every object whose swept bounding box can
produce a non-negligible overlap with ``i``, and of every object appearing as
the target of one of ``i``'s semantic relations.  Call that set ``dep(i)``.

Let ``M`` be the moved set and, for each object ``i``, let
``D(i) = dep(i) ∩ M``.  Treating every non-empty ``D(i)`` as a hyperedge and
taking connected components partitions ``M`` into components ``C_1..C_K`` such
that each object's term set is owned by exactly one component.  Then for any
union of whole components ``S``,

    ΔS_f(S) = sum over C ⊆ S of ΔS_f(C),
    ΔS_f(C) = (1 / N_f) * sum over i owned by C of (s_f^cand(i) - s_f^base(i)).

The right-hand side is read off the single batch evaluation.  So one batch
evaluation yields the exact delta vector of every component, and any union of
componentwise non-inferior components is non-inferior by construction.

Decision discipline
-------------------
The decomposition is used to *propose* a subset, never to *decide*.  The promoted
subset is always re-evaluated once and gated on the measured result.  The
predicted-versus-measured agreement is reported as evidence, so an incomplete
dependency model degrades efficiency but can never turn into a false promotion.

Cost: two scene evaluations per scene, independent of ``N``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

import eval_physical_realizability as evaluator
from modules._s4_layoutvlm_relations import build_semantic_relation_specs

FAMILIES: tuple[str, ...] = ("collision", "support", "plane", "boundary", "semantic")
DEFAULT_GATED_FAMILIES: tuple[str, ...] = ("collision", "support", "plane")

SCHEMA = "sceneproof_settle_component_selection_v1"


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------
class DisjointSet:
    """Union-find over a fixed item set."""

    def __init__(self, items: Iterable[str]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, first: str, second: str) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self._parent[second_root] = first_root


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if np.isfinite(value) else None
    text = str(raw).strip()
    if text == "" or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def valid_pose(value: Any) -> list[list[float]] | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != (4, 4) or not np.isfinite(array).all():
        return None
    return array.tolist()


# --------------------------------------------------------------------------
# probe ingestion
# --------------------------------------------------------------------------
def load_probes(
    probe_dir: Path,
    incumbent_info: dict[str, Any],
    *,
    min_translation_m: float,
    require_measured: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Return accepted probes keyed by object id, plus a rejection reason map."""
    probes: dict[str, dict[str, Any]] = {}
    rejected: dict[str, str] = {}
    for path in sorted(Path(probe_dir).glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            probe = load_json(path)
        except (OSError, json.JSONDecodeError) as error:
            rejected[path.stem] = f"unreadable_probe:{type(error).__name__}"
            continue
        object_id = str(probe.get("object_id") or path.stem)
        if object_id not in incumbent_info:
            rejected[object_id] = "absent_from_incumbent"
            continue
        if evaluator.STRUCTURAL.match(object_id):
            rejected[object_id] = "structural_object"
            continue
        if require_measured and probe.get("status") != "measured":
            rejected[object_id] = f"status:{probe.get('status')}"
            continue
        pose = valid_pose(probe.get("settled_pose_matrix"))
        if pose is None:
            rejected[object_id] = "invalid_settled_pose_matrix"
            continue
        translation = optional_float(probe.get("translation_delta_m"))
        if translation is None:
            rejected[object_id] = "missing_translation_delta"
            continue
        if translation < min_translation_m:
            rejected[object_id] = "below_min_translation"
            continue
        probes[object_id] = {
            "object_id": object_id,
            "settled_pose_matrix": pose,
            "translation_delta_m": translation,
            "rotation_delta_deg": optional_float(probe.get("rotation_delta_deg")),
            "probe_path": str(path),
        }
    return probes, rejected


def materialize_placement(
    incumbent: dict[str, Any],
    probes: dict[str, dict[str, Any]],
    object_ids: Sequence[str],
    *,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Write the settled poses of ``object_ids`` into a copy of ``incumbent``.

    The pose key is ``pose_matrix_for_blender`` because that is the only pose the
    evaluators and the renderer read.  Writing any other key produces a
    candidate that is byte-different but semantically identical to the
    incumbent, which silently invalidates every downstream comparison.
    """
    candidate = copy.deepcopy(incumbent)
    info = candidate.setdefault("obj_info", {})
    committed = []
    for object_id in object_ids:
        probe = probes.get(object_id)
        if probe is None or object_id not in info:
            continue
        info[object_id]["pose_matrix_for_blender"] = probe["settled_pose_matrix"]
        committed.append(object_id)
    candidate["sceneproof_settle_commit"] = {
        "schema_version": SCHEMA,
        "pose_key": "pose_matrix_for_blender",
        "committed_object_ids": sorted(committed),
        "pose_changes": len(committed),
        **provenance,
    }
    return candidate


# --------------------------------------------------------------------------
# dependency model
# --------------------------------------------------------------------------
def axis_bounds(geometry: Any) -> tuple[np.ndarray, np.ndarray]:
    corners = geometry.world_corners
    return corners.min(axis=0), corners.max(axis=0)


def union_bounds(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    return np.minimum(first[0], second[0]), np.maximum(first[1], second[1])


def bounds_overlap(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    epsilon: float = 1e-9,
) -> bool:
    return bool(
        np.all(first[0] <= second[1] + epsilon)
        and np.all(second[0] <= first[1] + epsilon)
    )


def overlap_volume(first: Any, second: Any) -> float:
    """Replicate the evaluator's unintended-overlap volume for one pair."""
    z_overlap = min(first.z_max, second.z_max) - max(first.z_min, second.z_min)
    if z_overlap <= 1e-6:
        return 0.0
    intersection = evaluator.convex_intersection(first.polygon, second.polygon)
    return float(evaluator.polygon_area(intersection) * z_overlap)


def semantic_targets(
    source_info: dict[str, Any],
    geometries: dict[str, Any],
) -> dict[str, set[str]]:
    """Return, per source object, the objects its semantic terms read."""
    ordered = list(geometries)
    warm_matrices = [
        source_info.get(name, {}).get(
            "pose_matrix_for_blender", geometries[name].matrix.tolist()
        )
        for name in ordered
    ]
    footprints = [
        [
            float(np.ptp(geometries[name].polygon[:, 0])),
            float(np.ptp(geometries[name].polygon[:, 1])),
        ]
        for name in ordered
    ]
    specs = build_semantic_relation_specs(
        source_info, ordered, warm_matrices, footprints
    )
    targets: dict[str, set[str]] = {}
    for key in ("align_pairs", "point_pairs", "distance_pairs"):
        for pair in specs.get(key, []) or []:
            source_index, target_index = int(pair[0]), int(pair[1])
            if not (
                0 <= source_index < len(ordered) and 0 <= target_index < len(ordered)
            ):
                continue
            targets.setdefault(ordered[source_index], set()).add(ordered[target_index])
    return targets


def build_dependency_sets(
    source_info: dict[str, Any],
    incumbent_info: dict[str, Any],
    batch_info: dict[str, Any],
    *,
    collision_volume_tolerance: float,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Return ``dep(i)`` for every scored object, plus a diagnostics block.

    ``dep(i)`` is a sound over-approximation: it contains every object whose pose
    can change any term of ``i`` under any mixture of incumbent and settled
    poses.  Over-approximating only coarsens the component partition; it can
    never make an attributed delta wrong.
    """
    base_geometries = evaluator.build_geometries(source_info, incumbent_info)
    candidate_geometries = evaluator.build_geometries(source_info, batch_info)
    shared = [name for name in base_geometries if name in candidate_geometries]
    object_ids = [name for name in shared if not evaluator.STRUCTURAL.match(name)]

    dependencies: dict[str, set[str]] = {name: {name} for name in object_ids}

    # Support and plane terms read the declared parent's pose.
    for child_id in object_ids:
        parent_id = evaluator.support_id(
            candidate_geometries[child_id].info.get("supported")
        )
        if parent_id and parent_id in candidate_geometries:
            dependencies[child_id].add(parent_id)

    # Collision terms read every object that can overlap non-negligibly.  The
    # swept bound prefilter is cheap; the exact four-configuration overlap test
    # keeps the graph as sparse as soundness allows.
    swept = {
        name: union_bounds(
            axis_bounds(base_geometries[name]), axis_bounds(candidate_geometries[name])
        )
        for name in object_ids
    }
    prefiltered_pairs = 0
    collision_edges = 0
    for index, first_id in enumerate(object_ids):
        for second_id in object_ids[index + 1 :]:
            if not bounds_overlap(swept[first_id], swept[second_id]):
                continue
            prefiltered_pairs += 1
            coupled = False
            for first in (base_geometries[first_id], candidate_geometries[first_id]):
                for second in (
                    base_geometries[second_id],
                    candidate_geometries[second_id],
                ):
                    if overlap_volume(first, second) > collision_volume_tolerance:
                        coupled = True
                        break
                if coupled:
                    break
            if coupled:
                collision_edges += 1
                dependencies[first_id].add(second_id)
                dependencies[second_id].add(first_id)

    # Semantic terms read their relation targets.  The relation set itself is
    # derived from footprints, which depend on pose, so both configurations are
    # unioned.
    semantic_edges = 0
    for geometries in (base_geometries, candidate_geometries):
        for source_id, targets in semantic_targets(source_info, geometries).items():
            if source_id not in dependencies:
                continue
            for target_id in targets:
                if target_id not in dependencies[source_id]:
                    dependencies[source_id].add(target_id)
                    semantic_edges += 1

    diagnostics = {
        "scored_object_count": len(object_ids),
        "collision_prefiltered_pairs": prefiltered_pairs,
        "collision_dependency_edges": collision_edges,
        "semantic_dependency_edges": semantic_edges,
        "mean_dependency_degree": (
            float(np.mean([len(value) for value in dependencies.values()]))
            if dependencies
            else 0.0
        ),
    }
    return dependencies, diagnostics


# --------------------------------------------------------------------------
# term tables and attribution
# --------------------------------------------------------------------------
def load_term_table(
    object_csv: Path,
    scene: str,
    versions: Sequence[str],
) -> dict[str, dict[str, dict[str, float | None]]]:
    table: dict[str, dict[str, dict[str, float | None]]] = {
        version: {} for version in versions
    }
    with Path(object_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "collision_term" not in reader.fieldnames:
            raise SystemExit(
                f"{object_csv} has no '*_term' columns; re-run "
                "eval_physical_realizability.py with the Fix85 term export"
            )
        for row in reader:
            if row.get("scene") != scene:
                continue
            version = row.get("version")
            if version not in table:
                continue
            object_id = row.get("object_id")
            if not object_id:
                continue
            table[version][object_id] = {
                family: optional_float(row.get(f"{family}_term"))
                for family in FAMILIES
            }
    return table


def family_denominators(
    physical: dict[str, Any],
    scene: str,
    version: str,
) -> dict[str, int]:
    scenes = physical.get("versions", {}).get(version, {}).get("scenes", {})
    families = scenes.get(scene, {}).get("families", {})
    return {
        family: int(families.get(family, {}).get("n") or 0) for family in FAMILIES
    }


def family_scores(
    physical: dict[str, Any],
    scene: str,
    version: str,
) -> dict[str, float | None]:
    scenes = physical.get("versions", {}).get(version, {}).get("scenes", {})
    families = scenes.get(scene, {}).get("families", {})
    return {
        family: optional_float(families.get(family, {}).get("score"))
        for family in FAMILIES
    }


def verify_term_export(
    table: dict[str, dict[str, dict[str, float | None]]],
    physical: dict[str, Any],
    scene: str,
    baseline_version: str,
    candidate_version: str,
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Check that the exported terms reproduce the evaluator's own family means.

    This is the falsifiable precondition of the whole method.  If it fails, the
    terms are not a faithful decomposition and no attribution is attempted.
    """
    checks: dict[str, Any] = {"per_family": {}, "ok": True, "violations": []}
    baseline_terms = table[baseline_version]
    candidate_terms = table[candidate_version]
    denominators = family_denominators(physical, scene, baseline_version)
    candidate_denominators = family_denominators(physical, scene, candidate_version)

    for family in FAMILIES:
        base_contributors = {
            object_id
            for object_id, terms in baseline_terms.items()
            if terms[family] is not None
        }
        candidate_contributors = {
            object_id
            for object_id, terms in candidate_terms.items()
            if terms[family] is not None
        }
        entry: dict[str, Any] = {
            "denominator_baseline": denominators.get(family, 0),
            "denominator_candidate": candidate_denominators.get(family, 0),
            "contributors_baseline": len(base_contributors),
            "contributors_candidate": len(candidate_contributors),
            "contributor_sets_equal": base_contributors == candidate_contributors,
            "denominators_equal": denominators.get(family)
            == candidate_denominators.get(family),
        }
        entry["denominator_matches_contributors"] = (
            denominators.get(family, 0) == len(base_contributors)
            and candidate_denominators.get(family, 0) == len(candidate_contributors)
        )

        reconstructed = {}
        for version, terms, contributors in (
            (baseline_version, baseline_terms, base_contributors),
            (candidate_version, candidate_terms, candidate_contributors),
        ):
            count = len(contributors)
            reconstructed[version] = (
                float(sum(terms[object_id][family] for object_id in contributors))
                / count
                if count
                else None
            )
        official = {
            version: family_scores(physical, scene, version)[family]
            for version in (baseline_version, candidate_version)
        }
        entry["reconstructed_score"] = reconstructed
        entry["official_score"] = official
        residuals = []
        for version in (baseline_version, candidate_version):
            left, right = reconstructed[version], official[version]
            if left is None and right is None:
                residuals.append(0.0)
            elif left is None or right is None:
                residuals.append(float("inf"))
            else:
                residuals.append(abs(left - right))
        entry["max_residual"] = max(residuals) if residuals else 0.0
        entry["ok"] = bool(
            entry["contributor_sets_equal"]
            and entry["denominators_equal"]
            and entry["denominator_matches_contributors"]
            and entry["max_residual"] <= tolerance
        )
        if not entry["ok"]:
            checks["ok"] = False
            checks["violations"].append(family)
        checks["per_family"][family] = entry
    return checks


def partition_components(
    dependencies: dict[str, set[str]],
    moved: set[str],
) -> list[dict[str, Any]]:
    """Partition ``moved`` so that every term is owned by exactly one component."""
    disjoint = DisjointSet(sorted(moved))
    touched_by_object: dict[str, list[str]] = {}
    for object_id, dependency in dependencies.items():
        touched = sorted(dependency & moved)
        if not touched:
            continue
        touched_by_object[object_id] = touched
        anchor = touched[0]
        for other in touched[1:]:
            disjoint.union(anchor, other)

    members: dict[str, list[str]] = {}
    for object_id in sorted(moved):
        members.setdefault(disjoint.find(object_id), []).append(object_id)
    owned: dict[str, list[str]] = {}
    for object_id, touched in touched_by_object.items():
        owned.setdefault(disjoint.find(touched[0]), []).append(object_id)

    components = []
    for index, (root, member_ids) in enumerate(sorted(members.items())):
        components.append(
            {
                "component_index": index,
                "component_root": root,
                "member_object_ids": sorted(member_ids),
                "owned_term_object_ids": sorted(owned.get(root, [])),
                "member_count": len(member_ids),
                "owned_term_count": len(owned.get(root, [])),
            }
        )
    return components


def attribute_components(
    components: list[dict[str, Any]],
    table: dict[str, dict[str, dict[str, float | None]]],
    denominators: dict[str, int],
    baseline_version: str,
    candidate_version: str,
) -> None:
    """Attach the exact per-family delta vector to each component, in place."""
    baseline_terms = table[baseline_version]
    candidate_terms = table[candidate_version]
    for component in components:
        delta: dict[str, float] = {}
        contributions: dict[str, int] = {}
        for family in FAMILIES:
            denominator = denominators.get(family, 0)
            if not denominator:
                delta[family] = 0.0
                contributions[family] = 0
                continue
            total = 0.0
            count = 0
            for object_id in component["owned_term_object_ids"]:
                base = baseline_terms.get(object_id, {}).get(family)
                cand = candidate_terms.get(object_id, {}).get(family)
                if base is None or cand is None:
                    continue
                total += cand - base
                count += 1
            delta[family] = total / denominator
            contributions[family] = count
        component["delta"] = delta
        component["term_contributions"] = contributions
        component["total_delta"] = float(sum(delta.values()))


def verify_additivity(
    components: list[dict[str, Any]],
    table: dict[str, dict[str, dict[str, float | None]]],
    denominators: dict[str, int],
    physical: dict[str, Any],
    scene: str,
    baseline_version: str,
    candidate_version: str,
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Check that component deltas sum to the measured batch delta.

    Also check that every object owned by no component has an identical term in
    both versions.  A non-zero delta there would mean the dependency model
    missed a coupling.
    """
    baseline_terms = table[baseline_version]
    candidate_terms = table[candidate_version]
    owned = {
        object_id
        for component in components
        for object_id in component["owned_term_object_ids"]
    }
    unowned_drift: list[dict[str, Any]] = []
    for object_id in sorted(set(baseline_terms) | set(candidate_terms)):
        if object_id in owned:
            continue
        for family in FAMILIES:
            base = baseline_terms.get(object_id, {}).get(family)
            cand = candidate_terms.get(object_id, {}).get(family)
            if base is None or cand is None:
                continue
            if abs(cand - base) > tolerance:
                unowned_drift.append(
                    {
                        "object_id": object_id,
                        "family": family,
                        "baseline_term": base,
                        "candidate_term": cand,
                        "delta": cand - base,
                    }
                )

    base_scores = family_scores(physical, scene, baseline_version)
    candidate_scores = family_scores(physical, scene, candidate_version)
    per_family: dict[str, Any] = {}
    ok = not unowned_drift
    for family in FAMILIES:
        predicted = float(
            sum(component["delta"].get(family, 0.0) for component in components)
        )
        left, right = base_scores.get(family), candidate_scores.get(family)
        measured = None if left is None or right is None else right - left
        residual = (
            abs(predicted - measured) if measured is not None else abs(predicted)
        )
        entry = {
            "predicted_delta": predicted,
            "measured_delta": measured,
            "residual": residual,
            "denominator": denominators.get(family, 0),
            "ok": residual <= tolerance,
        }
        if not entry["ok"]:
            ok = False
        per_family[family] = entry
    return {
        "ok": bool(ok),
        "per_family": per_family,
        "unowned_term_drift": unowned_drift[:50],
        "unowned_term_drift_count": len(unowned_drift),
        "tolerance": tolerance,
    }


def select_components(
    components: list[dict[str, Any]],
    *,
    gated_families: Sequence[str],
    epsilon: float,
    allow_surplus_trading: bool,
) -> dict[str, Any]:
    """Choose the components to promote.

    Pass one accepts every component that is non-inferior in each gated family on
    its own.  Because component deltas are additive and disjoint, accepting all
    of them is safe regardless of order, and the promoted union is non-inferior
    in every gated family by construction.

    Pass two is optional.  It admits a component that regresses one gated family
    only while the already accepted surplus in that family covers the regression.
    It sweeps repeatedly until a full sweep admits nothing, so the outcome does
    not depend on where a component happened to sit in a single ordering.
    """
    ranked = sorted(
        components, key=lambda item: (-item["total_delta"], item["component_index"])
    )
    accepted: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    running = {family: 0.0 for family in gated_families}

    for component in ranked:
        if all(
            component["delta"].get(family, 0.0) >= -epsilon
            for family in gated_families
        ):
            component["selection_reason"] = "componentwise_non_inferior"
            accepted.append(component)
            for family in gated_families:
                running[family] += component["delta"].get(family, 0.0)
        else:
            component["selection_reason"] = "would_regress_gated_family"
            pending.append(component)

    if allow_surplus_trading:
        progressed = True
        while progressed:
            progressed = False
            for component in list(pending):
                trial = {
                    family: running[family] + component["delta"].get(family, 0.0)
                    for family in gated_families
                }
                if all(value >= -epsilon for value in trial.values()):
                    component["selection_reason"] = (
                        "accepted_against_accumulated_surplus"
                    )
                    accepted.append(component)
                    pending.remove(component)
                    running = trial
                    progressed = True

    accepted.sort(key=lambda item: item["component_index"])
    rejected = sorted(pending, key=lambda item: item["component_index"])
    accepted_objects = sorted(
        object_id
        for component in accepted
        for object_id in component["member_object_ids"]
    )
    predicted = {
        family: float(
            sum(component["delta"].get(family, 0.0) for component in accepted)
        )
        for family in FAMILIES
    }
    return {
        "policy": (
            "surplus_trading" if allow_surplus_trading else "componentwise_strict"
        ),
        "gated_families": list(gated_families),
        "epsilon": epsilon,
        "accepted_component_indices": [
            component["component_index"] for component in accepted
        ],
        "rejected_component_indices": [
            component["component_index"] for component in rejected
        ],
        "accepted_component_count": len(accepted),
        "rejected_component_count": len(rejected),
        "accepted_object_ids": accepted_objects,
        "accepted_object_count": len(accepted_objects),
        "predicted_scene_delta": predicted,
        "predicted_gated_delta_non_negative": all(
            predicted[family] >= -epsilon for family in gated_families
        ),
    }


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--min-translation-m", type=float, default=0.05)
    parser.add_argument(
        "--allow-unmeasured-probes",
        action="store_true",
        help="accept probes whose status is not 'measured' (diagnostics only)",
    )


def command_materialize_batch(args: argparse.Namespace) -> int:
    incumbent = load_json(args.incumbent)
    probes, rejected = load_probes(
        args.probe_dir,
        incumbent.get("obj_info", {}),
        min_translation_m=args.min_translation_m,
        require_measured=not args.allow_unmeasured_probes,
    )
    candidate = materialize_placement(
        incumbent,
        probes,
        sorted(probes),
        provenance={
            "stage": "batch_all_moved",
            "incumbent": str(args.incumbent.resolve()),
            "probe_dir": str(args.probe_dir.resolve()),
            "min_translation_m": args.min_translation_m,
            "rejected_probe_count": len(rejected),
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out.resolve()}")
    print(f"MOVED={len(probes)} REJECTED={len(rejected)}")
    if not probes:
        print("ERROR: no probe passed the moved-object filter", flush=True)
        return 1
    return 0


def command_select(args: argparse.Namespace) -> int:
    incumbent = load_json(args.incumbent)
    # Resolve the frozen geometry snapshot through the evaluator's own locator so
    # the dependency model can never read a different scene than the evaluator
    # scored.
    geometry_path = evaluator.find_geometry_snapshot(
        Path(args.saved_results), args.scene, args.geometry_version
    )
    geometry = load_json(geometry_path)
    evaluator.validate_geometry_snapshot(geometry, geometry_path)
    batch_candidate = load_json(args.batch_candidate)
    physical = load_json(args.batch_physical)

    probes, rejected = load_probes(
        args.probe_dir,
        incumbent.get("obj_info", {}),
        min_translation_m=args.min_translation_m,
        require_measured=not args.allow_unmeasured_probes,
    )
    if not probes:
        raise SystemExit("no probe passed the moved-object filter")

    table = load_term_table(
        args.batch_objects_csv,
        args.scene,
        (args.baseline_version, args.candidate_version),
    )
    for version in (args.baseline_version, args.candidate_version):
        if not table[version]:
            raise SystemExit(
                f"{args.batch_objects_csv} has no rows for scene={args.scene} "
                f"version={version}"
            )

    export_check = verify_term_export(
        table,
        physical,
        args.scene,
        args.baseline_version,
        args.candidate_version,
        tolerance=args.residual_tolerance,
    )

    dependencies, dependency_diagnostics = build_dependency_sets(
        geometry.get("obj_info", {}),
        incumbent.get("obj_info", {}),
        batch_candidate.get("obj_info", {}),
        collision_volume_tolerance=args.collision_volume_tolerance,
    )
    moved = {object_id for object_id in probes if object_id in dependencies}
    components = partition_components(dependencies, moved)
    denominators = family_denominators(physical, args.scene, args.baseline_version)
    attribute_components(
        components, table, denominators, args.baseline_version, args.candidate_version
    )
    additivity_check = verify_additivity(
        components,
        table,
        denominators,
        physical,
        args.scene,
        args.baseline_version,
        args.candidate_version,
        tolerance=args.residual_tolerance,
    )

    gated = tuple(
        family.strip()
        for family in args.gated_families.split(",")
        if family.strip()
    )
    unknown = [family for family in gated if family not in FAMILIES]
    if unknown:
        raise SystemExit(f"unknown gated families: {unknown}")

    selection = select_components(
        components,
        gated_families=gated,
        epsilon=args.epsilon,
        allow_surplus_trading=args.allow_surplus_trading,
    )
    preconditions_ok = bool(export_check["ok"] and additivity_check["ok"])
    if not preconditions_ok and not args.allow_unverified_attribution:
        selection["accepted_object_ids"] = []
        selection["accepted_object_count"] = 0
        selection["accepted_component_indices"] = []
        selection["abstained"] = True
        selection["abstain_reason"] = (
            "term_export_or_additivity_verification_failed"
        )
    else:
        selection["abstained"] = False

    sizes = [component["member_count"] for component in components]
    report = {
        "schema_version": SCHEMA,
        "scene": args.scene,
        "baseline_version": args.baseline_version,
        "candidate_version": args.candidate_version,
        "incumbent": str(args.incumbent.resolve()),
        "batch_candidate": str(args.batch_candidate.resolve()),
        "geometry_snapshot": str(geometry_path.resolve()),
        "probe_dir": str(args.probe_dir.resolve()),
        "min_translation_m": args.min_translation_m,
        "moved_object_count": len(moved),
        "rejected_probe_count": len(rejected),
        "family_denominators": denominators,
        "component_count": len(components),
        "component_size_max": max(sizes) if sizes else 0,
        "component_size_mean": float(np.mean(sizes)) if sizes else 0.0,
        "dependency_diagnostics": dependency_diagnostics,
        "term_export_check": export_check,
        "additivity_check": additivity_check,
        "preconditions_ok": preconditions_ok,
        "selection": selection,
        "components": components,
    }
    args.out_selection.parent.mkdir(parents=True, exist_ok=True)
    args.out_selection.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out_selection.resolve()}")

    if args.out_candidate is not None:
        candidate = materialize_placement(
            incumbent,
            probes,
            selection["accepted_object_ids"],
            provenance={
                "stage": "component_selected_subset",
                "incumbent": str(args.incumbent.resolve()),
                "selection": str(args.out_selection.resolve()),
                "policy": selection["policy"],
                "gated_families": selection["gated_families"],
                "predicted_scene_delta": selection["predicted_scene_delta"],
                "requires_confirmation_evaluation": True,
            },
        )
        args.out_candidate.parent.mkdir(parents=True, exist_ok=True)
        args.out_candidate.write_text(
            json.dumps(candidate, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.out_candidate.resolve()}")

    print(
        f"MOVED={len(moved)} COMPONENTS={len(components)} "
        f"MAX_COMPONENT={max(sizes) if sizes else 0} "
        f"ACCEPTED_COMPONENTS={selection['accepted_component_count']} "
        f"ACCEPTED_OBJECTS={selection['accepted_object_count']}"
    )
    print(
        "TERM_EXPORT_OK={} ADDITIVITY_OK={} ABSTAINED={}".format(
            export_check["ok"], additivity_check["ok"], selection["abstained"]
        )
    )
    for family in FAMILIES:
        entry = additivity_check["per_family"][family]
        print(
            f"  {family}: batch_measured={entry['measured_delta']} "
            f"batch_predicted={entry['predicted_delta']} "
            f"residual={entry['residual']:.3e} "
            f"selected_predicted={selection['predicted_scene_delta'][family]:+.6f}"
        )
    return 0 if preconditions_ok else 2


def command_confirm(args: argparse.Namespace) -> int:
    """Gate the promoted subset on the measured confirmation evaluation."""
    selection = load_json(args.selection)
    physical = load_json(args.confirm_physical)
    predicted = selection["selection"]["predicted_scene_delta"]
    gated = selection["selection"]["gated_families"]

    base_scores = family_scores(physical, args.scene, args.baseline_version)
    candidate_scores = family_scores(physical, args.scene, args.candidate_version)
    per_family = {}
    measured_ok = True
    agreement_ok = True
    for family in FAMILIES:
        left, right = base_scores.get(family), candidate_scores.get(family)
        measured = None if left is None or right is None else right - left
        residual = (
            None if measured is None else abs(measured - predicted.get(family, 0.0))
        )
        gated_here = family in gated
        family_ok = (
            True
            if measured is None or not gated_here
            else measured >= -args.epsilon
        )
        if not family_ok:
            measured_ok = False
        if residual is not None and residual > args.residual_tolerance:
            agreement_ok = False
        per_family[family] = {
            "baseline_score": left,
            "candidate_score": right,
            "measured_delta": measured,
            "predicted_delta": predicted.get(family),
            "prediction_residual": residual,
            "gated": gated_here,
            "non_inferior": family_ok,
        }

    gt_ok = True
    gt_detail: dict[str, Any] = {}
    if args.confirm_gt is not None:
        gt = load_json(args.confirm_gt)
        baseline_gt = gt.get("versions", {}).get(args.baseline_version, {})
        candidate_gt = gt.get("versions", {}).get(args.candidate_version, {})
        budgets = {
            "rotation_auc60_aligned": args.gt_rotation_budget,
            "rotation_auc60_raw": args.gt_rotation_budget,
            "translation_auc05_aligned": args.gt_translation_budget,
            "translation_auc05_raw": args.gt_translation_budget,
        }
        for key, budget in budgets.items():
            left = optional_float(baseline_gt.get(key))
            right = optional_float(candidate_gt.get(key))
            if left is None or right is None:
                continue
            ok = right >= left - budget
            gt_detail[key] = {
                "baseline": left,
                "candidate": right,
                "delta": right - left,
                "budget": budget,
                "non_inferior": ok,
            }
            if not ok:
                gt_ok = False

    promoted = bool(
        measured_ok
        and gt_ok
        and not selection["selection"].get("abstained", False)
        and selection.get("preconditions_ok", False)
        and selection["selection"]["accepted_object_count"] > 0
    )
    report = {
        "schema_version": "sceneproof_settle_confirmation_v1",
        "scene": args.scene,
        "baseline_version": args.baseline_version,
        "candidate_version": args.candidate_version,
        "selection": str(args.selection.resolve()),
        "promoted_object_ids": selection["selection"]["accepted_object_ids"],
        "promoted_object_count": selection["selection"]["accepted_object_count"],
        "physical": per_family,
        "gt": gt_detail,
        "physical_non_inferior": measured_ok,
        "gt_non_inferior": gt_ok,
        "prediction_agreement_ok": agreement_ok,
        "promoted": promoted,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out.resolve()}")
    for family, entry in per_family.items():
        print(
            f"  {family}: measured={entry['measured_delta']} "
            f"predicted={entry['predicted_delta']} "
            f"residual={entry['prediction_residual']} gated={entry['gated']} "
            f"ok={entry['non_inferior']}"
        )
    for key, entry in gt_detail.items():
        print(
            f"  gt/{key}: {entry['baseline']} -> {entry['candidate']} "
            f"delta={entry['delta']:+.6f} ok={entry['non_inferior']}"
        )
    print(
        f"PHYSICAL_OK={measured_ok} GT_OK={gt_ok} "
        f"PREDICTION_AGREEMENT={agreement_ok} "
        f"PROMOTED_OBJECTS={report['promoted_object_count']} PROMOTED={promoted}"
    )
    return 0 if promoted else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    batch = subparsers.add_parser(
        "materialize-batch",
        help="commit every moved probe into one candidate placement",
    )
    add_common_arguments(batch)
    batch.add_argument("--out", type=Path, required=True)
    batch.set_defaults(handler=command_materialize_batch)

    select = subparsers.add_parser(
        "select", help="attribute the batch evaluation and choose a safe subset"
    )
    add_common_arguments(select)
    select.add_argument("--scene", required=True)
    select.add_argument("--saved-results", type=Path, default=Path("a10_reusable_results/paper30"))
    select.add_argument("--geometry-version", required=True)
    select.add_argument("--batch-candidate", type=Path, required=True)
    select.add_argument("--batch-physical", type=Path, required=True)
    select.add_argument("--batch-objects-csv", type=Path, required=True)
    select.add_argument("--baseline-version", required=True)
    select.add_argument("--candidate-version", required=True)
    select.add_argument("--out-selection", type=Path, required=True)
    select.add_argument("--out-candidate", type=Path, default=None)
    select.add_argument(
        "--gated-families", default=",".join(DEFAULT_GATED_FAMILIES)
    )
    select.add_argument("--epsilon", type=float, default=1e-9)
    select.add_argument("--residual-tolerance", type=float, default=1e-7)
    select.add_argument("--collision-volume-tolerance", type=float, default=1e-6)
    select.add_argument("--allow-surplus-trading", action="store_true")
    select.add_argument(
        "--allow-unverified-attribution",
        action="store_true",
        help="keep the selection even if the verification preconditions fail",
    )
    select.set_defaults(handler=command_select)

    confirm = subparsers.add_parser(
        "confirm", help="gate the promoted subset on the measured confirmation run"
    )
    confirm.add_argument("--scene", required=True)
    confirm.add_argument("--selection", type=Path, required=True)
    confirm.add_argument("--confirm-physical", type=Path, required=True)
    confirm.add_argument("--confirm-gt", type=Path, default=None)
    confirm.add_argument("--baseline-version", required=True)
    confirm.add_argument("--candidate-version", required=True)
    confirm.add_argument("--out", type=Path, required=True)
    confirm.add_argument("--epsilon", type=float, default=1e-6)
    confirm.add_argument("--residual-tolerance", type=float, default=1e-6)
    confirm.add_argument("--gt-rotation-budget", type=float, default=0.005)
    confirm.add_argument("--gt-translation-budget", type=float, default=0.001)
    confirm.set_defaults(handler=command_confirm)

    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
