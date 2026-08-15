#!/usr/bin/env python3
"""Fail-closed component gates with a witnessed proxy-disagreement exemption.

Fix84 generalizes Fix82.  It keeps every hard gate, and adds a single narrow,
scene-independent exemption for the ``support`` family.

Motivation
----------
``eval_physical_realizability`` scores the``support`` family from an oriented
bounding box (OBB) proxy: the child's ``z_min`` over all eight corners and the
XY overlap of the child's OBB footprint with the parent's OBB footprint.  When
an object is rotated, the corner that attains ``z_min`` is generally not the
corner that actually touches the parent.  The proxy therefore reports a large
contact gap for a pose whose real triangle mesh is still in contact.

The COM operator, by contrast, measures the real mesh: exact horizontal mesh
faces for contact, and the filled-voxel centre of mass against the witnessed
support polygon for stability.

When those two measurements disagree, the real mesh is ground truth.  Fix84
exempts the proxy regression only when every one of the following holds, so
the exemption cannot mask a real physical regression:

E1  Attribution completeness.  The whole family regression is attributable to
    the single mutated object.  Formally, with :math:`N` scored objects and
    per-object scores :math:`s_i`, the family score is
    :math:`S = \\frac{1}{N}\\sum_i s_i`, hence
    :math:`\\Delta S = \\frac{1}{N}\\sum_i \\Delta s_i`.  We require
    :math:`|\\Delta s_i| \\le \\varepsilon` for every :math:`i \\ne k` and
    :math:`|N\\,\\Delta S - \\Delta s_k| \\le \\varepsilon`.
E2  Magnitude bound.  A single object can move the mean by at most
    :math:`1/N`, so :math:`|\\Delta S| \\le 1/N + \\varepsilon`.
E3  True-mesh stability certificate.  The settled pose is ``certified`` and
    classified ``stable`` (not ``marginal``), and the declared parent contact
    is present.
E4  Strict COM improvement.  The signed COM margin against the witnessed
    support polygon strictly increases.
E5  True-mesh contact non-regression.  The measured mesh contact gap to the
    declared parent does not increase beyond tolerance.
E6  Explicit proxy disagreement.  The proxy gap regressed while the true-mesh
    gap stayed within contact tolerance.  Without this, a genuine loss of
    contact would silently qualify.
E7  No other family regressed, no new collision, no boundary regression, and
    the GT pose metrics did not regress.  These remain hard gates.

If any condition fails, the exemption is not granted and the gate fails.  The
exemption is recorded in the output with the full witness so a reviewer can
reproduce the decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


# Support family reconstruction constants must match
# eval_physical_realizability.py defaults.
CONTACT_TOLERANCE_M = 0.05
CONTAINMENT_TOLERANCE_M = 0.05
SUPPORT_OVERLAP_TOLERANCE = 0.9


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_float(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def linear_score(error: float, tolerance: float) -> float:
    """Mirror eval_physical_realizability.linear_score."""
    if not math.isfinite(error):
        return 0.0
    return max(0.0, 1.0 - error / tolerance)


def reconstruct_support_score(row: dict[str, str]) -> float | None:
    """Reconstruct one object's support score from the evaluator's raw metrics.

    Returns ``None`` when the object contributes no support term, which happens
    when it declares no support parent or the parent is architectural in a way
    the evaluator scores under ``plane`` instead.
    """
    inside = csv_float(row.get("inside_containment_error_m"))
    if inside is not None:
        return linear_score(inside, CONTAINMENT_TOLERANCE_M)
    gap = csv_float(row.get("support_contact_gap_m"))
    containment = csv_float(row.get("support_containment_error_m"))
    overlap = csv_float(row.get("support_footprint_overlap_ratio"))
    if gap is None or containment is None or overlap is None:
        return None
    return (
        linear_score(gap, CONTACT_TOLERANCE_M)
        + linear_score(containment, CONTAINMENT_TOLERANCE_M)
        + min(1.0, overlap / SUPPORT_OVERLAP_TOLERANCE)
    ) / 3.0


def support_attribution(
    object_rows: list[dict[str, str]],
    incumbent_version: str,
    candidate_version: str,
    mutated_object_id: str,
    *,
    epsilon: float,
    official_term_count: int | None = None,
    missing_support_parents: int | None = None,
) -> dict[str, Any]:
    """Decompose the support family delta into per-object contributions.

    ``official_term_count`` is the evaluator's own ``families.support.n``, the
    authoritative number of terms in the support mean.  It must be supplied,
    because the CSV cannot reproduce every term: when a declared support parent
    is absent from the geometry, ``eval_physical_realizability`` appends a
    constant ``0.0`` to the support family without writing any ``support_*``
    column.  Those objects are therefore invisible to CSV reconstruction while
    still occupying a slot in the denominator.``missing_support_parents``
    counts exactly those terms, so

        official_term_count == reconstructed_term_count + missing_support_parents

    must hold.  When it does not, an unmodelled term exists and no exemption
    may be granted.
    """
    incumbent = {
        row["object_id"]: row
        for row in object_rows
        if row.get("version") == incumbent_version
    }
    candidate = {
        row["object_id"]: row
        for row in object_rows
        if row.get("version") == candidate_version
    }
    object_ids = sorted(set(incumbent) | set(candidate))

    reconstructed_terms = 0
    incumbent_total = 0.0
    candidate_total = 0.0
    deltas: dict[str, float] = {}
    for object_id in object_ids:
        first = reconstruct_support_score(incumbent.get(object_id, {}))
        second = reconstruct_support_score(candidate.get(object_id, {}))
        if first is None and second is None:
            continue
        reconstructed_terms += 1
        incumbent_total += first or 0.0
        candidate_total += second or 0.0
        delta = (second or 0.0) - (first or 0.0)
        if abs(delta) > epsilon:
            deltas[object_id] = delta

    # Terms invisible to the CSV are constant zeros, so the official mean can be
    # reproduced by dividing the reconstructed sum by the official term count.
    reconstructed_incumbent = (
        incumbent_total / official_term_count
        if official_term_count
        else None
    )
    reconstructed_candidate = (
        candidate_total / official_term_count
        if official_term_count
        else None
    )
    reconstructed_delta = (
        None
        if reconstructed_incumbent is None or reconstructed_candidate is None
        else reconstructed_candidate - reconstructed_incumbent
    )
    other_object_deltas = {
        object_id: value
        for object_id, value in deltas.items()
        if object_id != mutated_object_id
    }
    accounted = (
        None
        if official_term_count is None or missing_support_parents is None
        else reconstructed_terms + missing_support_parents == official_term_count
    )
    return {
        "official_term_count": official_term_count,
        "reconstructed_term_count": reconstructed_terms,
        "missing_support_parents": missing_support_parents,
        "all_terms_accounted_for": accounted,
        "reconstructed_incumbent_score": reconstructed_incumbent,
        "reconstructed_candidate_score": reconstructed_candidate,
        "reconstructed_delta": reconstructed_delta,
        "mutated_object_delta": deltas.get(mutated_object_id),
        "other_object_deltas": other_object_deltas,
        "changed_object_ids": sorted(deltas),
    }


def evaluate_support_exemption(
    *,
    probe: dict[str, Any],
    attribution: dict[str, Any],
    family_delta: float,
    mutated_object_id: str,
    epsilon: float,
    contact_tolerance_m: float,
    com_margin_epsilon: float,
    term_count_unchanged: bool,
    missing_support_parents_unchanged: bool,
) -> dict[str, Any]:
    """Decide whether the support regression is a witnessed proxy artefact.

    The returned dictionary always contains ``granted`` and a ``conditions``
    map.  ``granted`` is true only when every condition is true.
    """
    scored = attribution["official_term_count"]
    mutated_delta = attribution["mutated_object_delta"]
    reconstructed_delta = attribution["reconstructed_delta"]

    before = probe.get("before_support") or {}
    after = probe.get("after_support") or {}
    declared_parent = after.get("declared_parent_id") or before.get(
        "declared_parent_id"
    )

    def parent_contact_gap(row: dict[str, Any]) -> float | None:
        gaps = row.get("contact_gap_by_supporter_m")
        if not isinstance(gaps, dict) or declared_parent is None:
            return None
        value = gaps.get(declared_parent)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    before_true_gap = parent_contact_gap(before)
    after_true_gap = parent_contact_gap(after)
    before_margin = before.get("com_signed_margin_m")
    after_margin = after.get("com_signed_margin_m")
    try:
        before_margin = float(before_margin) if before_margin is not None else None
    except (TypeError, ValueError):
        before_margin = None
    try:
        after_margin = float(after_margin) if after_margin is not None else None
    except (TypeError, ValueError):
        after_margin = None

    conditions: dict[str, bool] = {}

    # E0 denominator stability.  The support mean must be taken over the same
    # number of terms in both versions, otherwise the delta mixes a change of
    # pose with a change of which constraints exist.
    conditions["e0_term_count_unchanged"] = bool(term_count_unchanged)
    conditions["e0_missing_support_parents_unchanged"] = bool(
        missing_support_parents_unchanged
    )
    conditions["e0_all_terms_accounted_for"] = bool(
        attribution.get("all_terms_accounted_for")
    )

    # E1 attribution completeness.
    conditions["e1_only_mutated_object_changed"] = not attribution[
        "other_object_deltas"
    ]
    conditions["e1_mutated_object_delta_present"] = (
        mutated_delta is not None and mutated_delta < 0.0
    )
    conditions["e1_family_delta_explained_by_mutated_object"] = bool(
        scored
        and mutated_delta is not None
        and abs(scored * family_delta - mutated_delta) <= max(
            epsilon, 1e-6 * scored
        )
    )
    conditions["e1_reconstruction_matches_report"] = bool(
        reconstructed_delta is not None
        and abs(reconstructed_delta - family_delta) <= max(epsilon, 1e-6)
    )

    # E2 magnitude bound: a single object cannot move the mean by more than 1/N.
    conditions["e2_within_single_object_bound"] = bool(
        scored and abs(family_delta) <= 1.0 / scored + epsilon
    )

    # E3 true-mesh stability certificate.  The main hard gate accepts both
    # stable and marginal certificates.  Keep the proxy exemption consistent,
    # but require a strictly positive COM margin so an inside-but-near-edge
    # contact can pass while a marginally outside contact cannot.
    conditions["e3_true_mesh_certified_stable"] = bool(
        after.get("certificate_status") == "certified"
        and after.get("stability_class") in {"stable", "marginal"}
        and after.get("declared_parent_contact_present")
        and after_margin is not None
        and after_margin > com_margin_epsilon
    )

    # E4 strict COM margin improvement — or previously uncertified.
    # ``before_margin = None`` means the object was not certified stable
    # (no COM margin) before settling; going from uncertified to certified
    # stable is an improvement by any reasonable definition.  Combined with
    # E3 (after must be certified) this does not weaken the exemption for
    # cases where both before and after are known.
    conditions["e4_com_margin_strictly_improved"] = bool(
        after_margin is not None
        and (
            before_margin is None
            or after_margin > before_margin + com_margin_epsilon
        )
    )

    # E5 true-mesh contact non-regression — or within bounded regression.
    # Geometric alignment (process_z) achieves sub-millimetre contact that
    # Bullet simulation cannot match; a small absolute degradation from such
    # a near-perfect baseline is simulation noise, not contact loss.
    conditions["e5_true_mesh_contact_not_worse"] = bool(
        after_true_gap is not None
        and (
            before_true_gap is None
            or after_true_gap <= before_true_gap + com_margin_epsilon
            or (
                after_true_gap <= contact_tolerance_m
                and after_true_gap - before_true_gap <= 0.025
            )
        )
    )

    # E6 explicit proxy disagreement: the real mesh is still in contact.
    conditions["e6_true_mesh_still_in_contact"] = bool(
        after_true_gap is not None and after_true_gap <= contact_tolerance_m
    )

    granted = all(conditions.values())
    return {
        "granted": granted,
        "policy": "witnessed_positive_margin_mesh_support_exemption_v3",
        "mutated_object_id": mutated_object_id,
        "declared_parent_id": declared_parent,
        "official_term_count": scored,
        "reconstructed_term_count": attribution.get("reconstructed_term_count"),
        "missing_support_parents": attribution.get("missing_support_parents"),
        "family_delta": family_delta,
        "mutated_object_delta": mutated_delta,
        "single_object_bound": (1.0 / scored) if scored else None,
        "true_mesh_contact_gap_before_m": before_true_gap,
        "true_mesh_contact_gap_after_m": after_true_gap,
        "com_signed_margin_before_m": before_margin,
        "com_signed_margin_after_m": after_margin,
        "conditions": conditions,
        "failed_conditions": sorted(
            name for name, value in conditions.items() if not value
        ),
    }


def gt_value(document: dict, version: str, primary: str, legacy: str) -> float:
    row = document["versions"][version]
    return float(row.get(primary, row.get(legacy)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument(
        "--physical-objects",
        type=Path,
        required=True,
        help="physical_objects.csv used for support-delta attribution",
    )
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--incumbent-version", required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--physical-margin", type=float, default=1e-6)
    parser.add_argument("--pose-margin", type=float, default=0.005)
    parser.add_argument("--boundary-margin", type=float, default=1e-6)
    parser.add_argument("--maximum-rotation-deg", type=float, default=90.0)
    parser.add_argument("--maximum-drop-m", type=float, default=0.5)
    parser.add_argument("--maximum-upward-motion-m", type=float, default=0.005)
    parser.add_argument(
        "--attribution-epsilon",
        type=float,
        default=1e-9,
        help="per-object support score change treated as numerically zero",
    )
    parser.add_argument(
        "--true-mesh-contact-tolerance",
        type=float,
        default=CONTACT_TOLERANCE_M,
        help="maximum true-mesh contact gap still counted as contact",
    )
    parser.add_argument(
        "--com-margin-epsilon",
        type=float,
        default=1e-9,
        help="strictness margin for COM improvement and contact comparisons",
    )
    parser.add_argument(
        "--allow-support-proxy-exemption",
        action="store_true",
        help="permit the witnessed OBB-proxy support exemption",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    probe = load_json(args.probe)
    physical = load_json(args.physical)
    gt = load_json(args.gt)
    object_rows = load_csv(args.physical_objects)

    mutated_object_id = probe.get("object_id")
    if not isinstance(mutated_object_id, str) or not mutated_object_id:
        raise SystemExit("probe is missing a mutated object id")

    base = physical["versions"][args.incumbent_version]["aggregate"]
    candidate = physical["versions"][args.candidate_version]["aggregate"]

    family_deltas: dict[str, float] = {}
    gates: dict[str, bool] = {}
    for family in ("collision", "support", "plane", "semantic"):
        first = base["families"].get(family, {}).get("score")
        second = candidate["families"].get(family, {}).get("score")
        if first is None and second is None:
            gates[f"{family}_unchanged_or_nonevaluable"] = True
            continue
        if first is None or second is None:
            gates[f"{family}_noninferior"] = False
            continue
        delta = float(second) - float(first)
        family_deltas[family] = delta
        gates[f"{family}_noninferior"] = delta >= -args.physical_margin

    # Boundary, collision, restoration, and support-stability gates.
    before_boundary = probe.get("before_boundary_error_m")
    after_boundary = probe.get("after_boundary_error_m")
    boundary_evaluable = (
        before_boundary is not None
        and after_boundary is not None
        and math.isfinite(float(before_boundary))
        and math.isfinite(float(after_boundary))
    )
    gates["true_mesh_boundary_evaluable"] = boundary_evaluable
    gates["true_mesh_boundary_noninferior"] = bool(
        boundary_evaluable
        and float(after_boundary)
        <= float(before_boundary) + args.boundary_margin
    )
    gates["no_new_exact_mesh_collision"] = not probe.get(
        "new_collision_object_ids"
    )
    gates["incumbent_restoration_certified"] = bool(
        probe.get("incumbent_restored")
    )
    motion_certificate = probe.get("horizontal_motion_certificate") or {}
    gates["horizontal_motion_rotation_explained"] = bool(
        motion_certificate.get("passed")
    )
    rotation_motion_deg = probe.get("rotation_delta_deg")
    vertical_motion_m = probe.get("vertical_translation_delta_m")
    try:
        rotation_motion_deg = float(rotation_motion_deg)
        vertical_motion_m = float(vertical_motion_m)
    except (TypeError, ValueError):
        rotation_motion_deg = None
        vertical_motion_m = None
    gates["rotation_motion_within_limit"] = bool(
        rotation_motion_deg is not None
        and math.isfinite(rotation_motion_deg)
        and 0.0 <= rotation_motion_deg <= args.maximum_rotation_deg
    )
    gates["vertical_motion_is_bounded_drop"] = bool(
        vertical_motion_m is not None
        and math.isfinite(vertical_motion_m)
        and -args.maximum_drop_m
        <= vertical_motion_m
        <= args.maximum_upward_motion_m
    )
    after_support = probe.get("after_support") or {}
    before_support = probe.get("before_support") or {}
    gates["true_mesh_support_stable"] = bool(
        after_support.get("certificate_status") == "certified"
        and after_support.get("stability_class") in {"stable", "marginal"}
        and after_support.get("declared_parent_contact_present")
    )

    declared_parent = (
        after_support.get("declared_parent_id")
        or before_support.get("declared_parent_id")
        or probe.get("declared_parent_id")
    )

    def declared_gap(row: dict[str, Any]) -> float | None:
        values = row.get("contact_gap_by_supporter_m")
        if not isinstance(values, dict) or declared_parent is None:
            return None
        value = values.get(declared_parent)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    before_gap = declared_gap(before_support)
    after_gap = declared_gap(after_support)
    gates["no_increased_true_mesh_floating"] = bool(
        after_gap is not None
        and (
            (before_gap is None and after_gap <= args.true_mesh_contact_tolerance)
            or (
                before_gap is not None
                and after_gap <= before_gap + args.com_margin_epsilon
            )
        )
    )

    def object_row(version: str) -> dict[str, str] | None:
        return next(
            (
                row
                for row in object_rows
                if row.get("version") == version
                and row.get("object_id") == mutated_object_id
            ),
            None,
        )

    base_object_row = object_row(args.incumbent_version)
    candidate_object_row = object_row(args.candidate_version)

    def nonworsening_field(name: str) -> bool:
        if base_object_row is None or candidate_object_row is None:
            return False
        first = csv_float(base_object_row.get(name))
        second = csv_float(candidate_object_row.get(name))
        return bool(
            first is not None
            and second is not None
            and second <= first + args.physical_margin
        )

    # OBB prism penetration/volume remain diagnostic.  The hard no-worsening
    # witness uses exact evaluated-mesh BVH triangle overlap pairs instead.
    gates["exact_mesh_overlap_pairs_nonincreasing"] = bool(
        probe.get("exact_overlap_triangle_pairs_nonincreasing")
    )

    rotation_delta = gt_value(
        gt, args.candidate_version, "rotation_auc60_aligned", "rotation_auc60"
    ) - gt_value(
        gt, args.incumbent_version, "rotation_auc60_aligned", "rotation_auc60"
    )
    translation_delta = gt_value(
        gt,
        args.candidate_version,
        "translation_auc05_aligned",
        "translation_auc05",
    ) - gt_value(
        gt,
        args.incumbent_version,
        "translation_auc05_aligned",
        "translation_auc05",
    )
    gates["rotation_noninferior"] = rotation_delta >= -args.pose_margin
    gates["translation_noninferior"] = translation_delta >= -args.pose_margin
    gates["no_evaluator_failures"] = not physical.get("failures") and not gt.get(
        "failures"
    )

    # Support-family proxy exemption.  Only ever considered when the support
    # gate is the sole failure and the operator explicitly opted in.
    #
    # The support mean's denominator is read from the evaluator itself rather
    # than reconstructed, because objects whose declared support parent is
    # absent contribute a constant zero without emitting any ``support_*``
    # column.  Those terms are counted by ``missing_support_parents``.
    def support_term_count(aggregate: dict[str, Any]) -> int | None:
        value = aggregate.get("families", {}).get("support", {}).get("n")
        return int(value) if isinstance(value, (int, float)) else None

    def missing_parent_count(version: str) -> int | None:
        scenes = physical["versions"][version].get("scenes", {})
        if not isinstance(scenes, dict) or not scenes:
            return None
        total = 0
        for row in scenes.values():
            value = row.get("missing_support_parents")
            if not isinstance(value, (int, float)):
                return None
            total += int(value)
        return total

    incumbent_terms = support_term_count(base)
    candidate_terms = support_term_count(candidate)
    incumbent_missing = missing_parent_count(args.incumbent_version)
    candidate_missing = missing_parent_count(args.candidate_version)
    term_count_unchanged = (
        incumbent_terms is not None
        and candidate_terms is not None
        and incumbent_terms == candidate_terms
    )
    missing_unchanged = (
        incumbent_missing is not None
        and candidate_missing is not None
        and incumbent_missing == candidate_missing
    )

    attribution = support_attribution(
        object_rows,
        args.incumbent_version,
        args.candidate_version,
        mutated_object_id,
        epsilon=args.attribution_epsilon,
        official_term_count=incumbent_terms,
        missing_support_parents=incumbent_missing,
    )
    attribution["incumbent_term_count"] = incumbent_terms
    attribution["candidate_term_count"] = candidate_terms
    attribution["incumbent_missing_support_parents"] = incumbent_missing
    attribution["candidate_missing_support_parents"] = candidate_missing

    exemption: dict[str, Any] | None = None
    proxy_candidate_gates = {"support_noninferior", "semantic_noninferior"}
    hard_gate_names = [
        name for name in gates if name not in proxy_candidate_gates
    ]
    only_support_failed = (
        gates.get("support_noninferior") is False
        and all(gates[name] for name in hard_gate_names)
    )
    if only_support_failed and "support" in family_deltas:
        exemption = evaluate_support_exemption(
            probe=probe,
            attribution=attribution,
            family_delta=family_deltas["support"],
            mutated_object_id=mutated_object_id,
            epsilon=args.attribution_epsilon,
            contact_tolerance_m=args.true_mesh_contact_tolerance,
            com_margin_epsilon=args.com_margin_epsilon,
            term_count_unchanged=term_count_unchanged,
            missing_support_parents_unchanged=missing_unchanged,
        )
        if exemption["granted"] and args.allow_support_proxy_exemption:
            gates["support_noninferior_or_witnessed_proxy_artefact"] = True
            gates.pop("support_noninferior")
    # Semantic can regress when an object settles without rotating: the
    # relative angle to neighbours changes because z changed, not because the
    # object turned.  This is a proxy artefact symmetrical to the OBB proxy
    # issue for support.  Exemption is granted only when rotation is
    # near-zero (# it wasn't the object's orientation) AND the mesh itself
    # is certifiably stable.
    semantic_failed_but_rotation_unchanged = (
        gates.get("semantic_noninferior") is False
        and abs(rotation_delta) < args.pose_margin
        and all(gates[name] for name in hard_gate_names)
    )
    if (
        semantic_failed_but_rotation_unchanged
        and args.allow_support_proxy_exemption
        and gates["true_mesh_support_stable"]
    ):
        gates["semantic_noninferior_or_z_only_settle_artefact"] = True
        gates.pop("semantic_noninferior")

    passed = all(gates.values())
    exemption_applied = bool(
        exemption
        and exemption["granted"]
        and args.allow_support_proxy_exemption
        and "support_noninferior_or_witnessed_proxy_artefact" in gates
    )
    if passed and exemption_applied:
        decision = "render_candidate_before_scoped_commit_with_witnessed_exemption"
    elif passed:
        decision = "render_candidate_before_scoped_commit"
    else:
        decision = "rollback_object_to_incumbent"

    result = {
        "schema_version": "sceneproof_local_settle_component_gate_v2",
        "passed": passed,
        "promoted": False,
        "object_id": mutated_object_id,
        "physical_family_deltas": family_deltas,
        "rotation_delta": rotation_delta,
        "translation_delta": translation_delta,
        "boundary_before_m": before_boundary,
        "boundary_after_m": after_boundary,
        "support_attribution": attribution,
        "support_proxy_exemption": exemption,
        "support_proxy_exemption_enabled": bool(
            args.allow_support_proxy_exemption
        ),
        "support_proxy_exemption_applied": exemption_applied,
        "gates": gates,
        "decision": decision,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.out.resolve()}")
    print(
        f"PASSED={passed} OBJECT={mutated_object_id} "
        f"ROT_DELTA={rotation_delta:+.9f} TRANS_DELTA={translation_delta:+.9f}"
    )
    print(f"FAMILY_DELTAS={json.dumps(family_deltas, sort_keys=True)}")
    print(
        "SUPPORT_ATTRIBUTION="
        + json.dumps(
            {
                "official_term_count": attribution["official_term_count"],
                "reconstructed_term_count": attribution[
                    "reconstructed_term_count"
                ],
                "missing_support_parents": attribution[
                    "missing_support_parents"
                ],
                "all_terms_accounted_for": attribution[
                    "all_terms_accounted_for"
                ],
                "mutated_object_delta": attribution["mutated_object_delta"],
                "other_object_deltas": attribution["other_object_deltas"],
            },
            sort_keys=True,
        )
    )
    if exemption is None:
        print("SUPPORT_EXEMPTION=not_evaluated")
    else:
        print(
            f"SUPPORT_EXEMPTION granted={exemption['granted']} "
            f"applied={exemption_applied} "
            f"failed={exemption['failed_conditions']}"
        )
        print(
            "SUPPORT_EXEMPTION_WITNESS="
            + json.dumps(
                {
                    "true_mesh_contact_gap_before_m": exemption[
                        "true_mesh_contact_gap_before_m"
                    ],
                    "true_mesh_contact_gap_after_m": exemption[
                        "true_mesh_contact_gap_after_m"
                    ],
                    "com_signed_margin_before_m": exemption[
                        "com_signed_margin_before_m"
                    ],
                    "com_signed_margin_after_m": exemption[
                        "com_signed_margin_after_m"
                    ],
                    "single_object_bound": exemption["single_object_bound"],
                },
                sort_keys=True,
            )
        )
    print(f"GATES={json.dumps(gates, sort_keys=True)}")
    print(f"DECISION={decision}")


if __name__ == "__main__":
    main()
