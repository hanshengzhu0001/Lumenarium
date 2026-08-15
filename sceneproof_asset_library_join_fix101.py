#!/usr/bin/env python3
"""SceneProof Fix101: join the placement documents to the asset library metadata.

Why this supersedes every name heuristic in Fix97to Fix100
----------------------------------------------------------
``asset_data/imaginarium_asset_info.csv`` holds 2043 assets with, per asset:

    name_en              the identifier used as ``retrieved_asset``
    bbx                  the authored real-world dimensions, in metres
    class_en             a curated fine category, for example Multi_person_sofa
    retrieval_class_en   a curated coarse category, for example Stool_chair_or_sofa
    scaling_strategy     the same value``fbx_scaling_strategy`` supplies

So an external reference has been inside the pipeline the whole time and was never
used for validation.  Three things follow, and this file measures all three rather
than asserting them.

1. The identity everything rests on becomes checkable
-----------------------------------------------------
Five rounds of attribution rest on ``asset native size = length / scale``, which follows
from ``S4_blender_layout_and_corr.py`` line 7277 together with the definition of a
Blender object's ``dimensions``.  That much is by construction and needs no measurement:
``length / scale`` *is* the mesh's own bounding box.  What is empirical, and what this
file measures, is whether that mesh box equals the size the library authors in ``bbx``.

Fix102 measured 342 of 374 objects agreeing, 91.4 per cent overall and 78.8, 87.3, 94.6,
92.5 and 100 per cent by scene.  The 32 failures are neither noise nor one mechanism, so
each is now classified by the *shape* of its disagreement.  Four in ``official_01`` are
one scalar on all three axes (``bookshelf_26`` 2.785, ``sculpture_1`` 2.310,
``file_folder_3`` 1.702, ``small_potted_plant_1`` 1.487), which cannot come from a
curator authoring one dimension differently and instead means a scale was applied where
the ``scale`` field does not record it.  Others differ on exactly one axis while the
remaining two match to three decimals (``44_sk75_CasinoTable02`` 1.336 in height, a
1.24 m casino table against an authored and plausible 0.924 m; ``tv_cabinet_0`` 1.227;
``desktop_table_lamp`` 1.100; ``desk_0`` 1.031), which is the opposite situation and
points at a single authored number rather than at a transform.  The paper may therefore
state the mesh identity unqualified and must state library-to-mesh agreement as a rate.

2. Retrieval correctness needs two witnesses, not one
-----------------------------------------------------
Scene categories and ``class_en`` are one aligned vocabulary of 498 classes: zero of
374 objects abstained, so the comparison is exact rather than a substring search.  But
``class_en`` is a curated *bucket*, not the asset's identity, and Fix102 proved it on a
case that admits no argument.  ``wardrobe_0`` retrieved ``a_SM_Wardrobe_01`` and was
reported as a substitution because the bucket reads ``Storage_locker``.  No asset can be
a better answer for a wardrobe than one named Wardrobe.  The same shape recurs:
``a_Signs22`` bucketed ``Billboard`` for a ``sign``, ``0_SM_Shelf_2`` bucketed
``Display_cabinet`` for a ``bookshelf``, ``44_sk82_KidCycle01`` bucketed ``Toy_car`` for
a ``children_tricycle``, ``a_SM_papers_pages_04`` bucketed ``Map`` for ``paper``,
``a_SM_KitchenFruit_Tomato01`` bucketed ``Vegetable`` for ``fruit``.

So the asset identifier is read as a second, independent witness.  A disagreement counts
as a substitution only when the bucket disagrees *and* the identifier fails to
corroborate the category.  The asymmetry is deliberate: a matching identifier is
positive evidence and may excuse a disagreement, a non-matching identifier is only
absence of evidence and may never create one.  Where the identifier carries no words at
all (``b_33``, ``d_1000003614815``, ``21_SM_PC_01ae``) the object stays a defect and is
counted apart, so a reader can see how much of the rate rests on no evidence.

Token comparison is suffix-aware because English closed compounds defeated Fix102:
``teacup`` against ``Water_cup`` shares no underscore-separated token yet names the same
object.  A shorter token counts as the head of a longer one when it is a suffix and the
remaining prefix is itself at least three characters, which admits tea|cup, book|shelf
and tri|cycle while rejecting o|pen.  Regular plurals are folded; irregular ones are
not, so ``0_steel_frame_shelves_03`` still fails to corroborate ``bookshelf``.

3. The library size is a reference the corpus could not be
----------------------------------------------------------
The cross-scene corpus of Fix99 and Fix100 is a majority reference and is uniformly
wrong exactly where a fix is needed: every ``pen`` retrieving ``a_SM_Point_Lamp_4``
renders at 1.81 m over three instances, and twelve instances of a ``bookshelf``
retrieving ``a_SM_locker_locker_main`` all render at 1.80 m.  It detects scene-level
inconsistency and cannot serve as a size prior.  ``bbx`` is authored per asset and is
independent of how the pipeline used it, so ``rendered / bbx`` is the effective scale
as the library sees it.

This tool changes nothing.  It measures a join and reports rates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_physical_realizability import STRUCTURAL, CAMERA  # noqa: E402
from sceneproof_scaling_chain_attribution_fix97 import (  # noqa: E402
    as_vector,
    category_of,
    equivalent_linear_factor,
    implied_native_size,
    longest_edge_ratio,
    production_scale_branch,
    size_disagreement,
    sorted_ratio,
    symmetric_factor,
)


def parse_bbx(raw: str | None) -> np.ndarray | None:
    """Parse the library's ``bbx`` field, three comma-separated metre values."""
    if not raw:
        return None
    parts = [piece.strip() for piece in str(raw).split(",")]
    if len(parts) != 3:
        return None
    try:
        values = np.asarray([float(piece) for piece in parts], dtype=np.float64)
    except ValueError:
        return None
    if not np.isfinite(values).all() or (values <= 0).any():
        return None
    return values


def load_asset_library(path: Path) -> dict[str, dict[str, Any]]:
    """Index the library by ``name_en``.

    The file is read as utf-8-sig because it is exported from a spreadsheet and
    carries a byte order mark; decoding it as plain utf-8 fails on the first field
    name and would silently lose every row.
    """
    library: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("name_en") or "").strip()
            if not name:
                continue
            library[name] = {
                "bbx_m": parse_bbx(row.get("bbx")),
                "class_en": (row.get("class_en") or "").strip(),
                "retrieval_class_en": (row.get("retrieval_class_en") or "").strip(),
                "scaling_strategy": (row.get("scaling_strategy") or "").strip(),
            }
    return library


def library_vocabulary(library: dict[str, dict[str, Any]]) -> set[str]:
    """Every ``class_en`` value, lowercased, as the vocabulary to compare against."""
    return {
        entry["class_en"].lower()
        for entry in library.values()
        if entry["class_en"]
    }


def singular(token: str) -> str:
    """Fold a regular English plural so``papers`` and ``paper`` are one word.

    Irregular plurals are left alone rather than handled by a table, which is why
    ``shelves`` does not reach ``shelf``.  That is a declared blind spot, not an
    oversight: a hand-written irregular list is the kind of curation this whole line of
    work exists to avoid.
    """
    lowered = token.lower()
    if len(lowered) >= 5 and lowered.endswith("ies"):
        return lowered[:-3] + "y"
    if len(lowered) >= 4 and lowered.endswith("s") and not lowered.endswith("ss"):
        return lowered[:-1]
    return lowered


def tokens_relate(first: str, second: str) -> bool:
    """Whether two tokens name the same thing, allowing English closed compounds.

    ``teacup`` and ``cup`` are one word and zero shared tokens, which is what made
    Fix102 call a teacup retrieving a water cup a substitution.  A shorter token counts
    as the head of a longer one when it is a suffix and the remaining prefix is itself at
    least three characters.  That threshold is what rejects the coincidence ``pen``
    inside ``open`` while admitting tea|cup, book|shelf and tri|cycle.
    """
    if first == second:
        return True
    longer, shorter = (first, second) if len(first) >= len(second) else (second, first)
    return (
        len(shorter) >= 3
        and len(longer) - len(shorter) >= 3
        and longer.endswith(shorter)
    )


def label_tokens(label: str) -> list[str]:
    """Tokens of a curated label, short connectives dropped and plurals folded."""
    return [
        singular(token)
        for token in label.lower().split("_")
        if len(token) >= 3 and token != "the"
    ]


CAMEL_OR_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+")


def asset_name_tokens(name: str) -> list[str]:
    """Words carried by the asset identifier itself.

    Identifiers are a source prefix, an optional batch code and then words in snake or
    camel case, so both are split; anything under three letters or purely numeric goes,
    which removes the ``SM``, ``sk``, ``uf`` and ``NN`` codes without naming them.
    ``b_33`` and ``21_SM_PC_01ae`` reduce to nothing, and that emptiness is reported as
    absence of evidence rather than as disagreement.
    """
    words: list[str] = []
    for chunk in str(name).split("_"):
        for piece in CAMEL_OR_WORD.findall(chunk):
            if len(piece) >= 3:
                words.append(singular(piece))
    return words


def asset_name_relation(category: str, asset: str | None) -> str:
    """Whether the asset's own identifier corroborates the object category."""
    if not asset:
        return "asset_name_is_opaque"
    words = asset_name_tokens(asset)
    own = label_tokens(category)
    if not words:
        return "asset_name_is_opaque"
    if not own:
        return "asset_name_shares_nothing"
    if any(tokens_relate(own[-1], word) for word in words):
        return "asset_name_carries_the_head_noun"
    if any(tokens_relate(token, word) for token in own for word in words):
        return "asset_name_carries_a_token"
    return "asset_name_shares_nothing"


CORROBORATING = {
    "asset_name_carries_the_head_noun",
    "asset_name_carries_a_token",
}


def label_relation(category: str, asset_class: str) -> str:
    """How two curated labels relate.

    A contradiction rate of 30 to 70 per cent conflates two different things, and the
    library's own labels show it: ``pen_holder`` against ``Desktop_pen_holder`` is one
    object at a finer granularity, while ``pen`` against ``Chandelier`` is a
    substitution.  Only ``shares_nothing`` is a candidate defect, and even that is then
    put to the asset identifier before being counted.

    Known imperfection, stated rather than patched with a hand list: ``stack_of_chips``
    against ``Stack_of_poker_cards`` shares ``stack`` and is called one family although
    it is a substitution.
    """
    own = label_tokens(category)
    other = label_tokens(asset_class)
    if category == asset_class.lower():
        return "identical_label"
    if not own or not other:
        return "shares_nothing"
    if tokens_relate(own[-1], other[-1]):
        return "shares_the_head_noun"
    if any(tokens_relate(mine, theirs) for mine in own for theirs in other):
        return "shares_a_token"
    return "shares_nothing"


def edges_agree_within_quantisation(
    computed: np.ndarray,
    authored: np.ndarray,
    *,
    quantum: float,
    tolerance: float,
) -> tuple[bool, float]:
    """Compare two boxes accounting for the library's rounding of ``bbx``.

    ``bbx`` is recorded to three decimal places, so an authored 0.004 stands for a
    true value anywhere in [0.0035, 0.0045], which is plus or minus 12.5 per cent.  A
    flat relative tolerance is therefore not defensible near the quantisation
    granularity: ``mouse_1`` printed identical computed and authored boxes and was
    still failed at 1.127, entirely inside the rounding noise.

    Returns whether every sorted edge lies inside its rounding interval widened by
    ``tolerance``, and the worst plain edge ratio for reporting.
    """
    order_computed = np.sort(computed)[::-1]
    order_authored = np.sort(authored)[::-1]
    low = np.maximum(order_authored - quantum, 1e-9) * (1.0 - tolerance)
    high = (order_authored + quantum) * (1.0 + tolerance)
    inside = bool(np.all((order_computed >= low) & (order_computed <= high)))
    ratio = order_computed / np.maximum(order_authored, 1e-9)
    worst = float(np.max(np.maximum(ratio, 1.0 / np.maximum(ratio, 1e-9))))
    return inside, worst


def identity_mismatch_shape(
    computed: np.ndarray,
    authored: np.ndarray,
    *,
    quantum: float,
    tolerance: float,
) -> tuple[str, list[float], float]:
    """Name the shape of an identity failure, because the shape names the cause.

    One scalar on all three axes cannot come from a curator authoring one dimension
    differently; it means a scale was applied where the ``scale`` field does not record
    it.  A disagreement confined to one axis, with the other two matching to three
    decimals, is the opposite: the mesh is consistent and one authored number is not.
    Collapsing the two into a single pass rate hides that they need different fixes.

    Uniformity is tested by rescaling the authored box by the geometric mean ratio and
    re-running the rounding-interval comparison, not by a fixed spread threshold: at
    ``file_folder_3`` the thinnest edge is 0.013 m, where three-decimal rounding alone is
    four per cent, so any fixed threshold would be arbitrary at exactly the sizes that
    matter.
    """
    order_computed = np.sort(computed)[::-1]
    order_authored = np.sort(authored)[::-1]
    ratio = order_computed / np.maximum(order_authored, 1e-9)
    geometric = float(np.exp(np.mean(np.log(np.maximum(ratio, 1e-9)))))

    rescaled_inside, _ = edges_agree_within_quantisation(
        order_computed,
        order_authored * geometric,
        quantum=quantum,
        tolerance=tolerance,
    )
    if rescaled_inside:
        return "uniform_scalar_offset", ratio.tolist(), geometric

    low = np.maximum(order_authored - quantum, 1e-9) * (1.0 - tolerance)
    high = (order_authored + quantum) * (1.0 + tolerance)
    outside = int(np.sum(~((order_computed >= low) & (order_computed <= high))))
    if outside == 1:
        return "one_axis_only", ratio.tolist(), geometric
    return "general", ratio.tolist(), geometric


def classify_retrieval(
    category: str, entry: dict[str, Any] | None, vocabulary: set[str]
) -> str:
    """Compare a scene category against the library's curated class for its asset.

    ``object_category_absent_from_library_vocabulary`` measured zero across all 364
    Smoke5 objects, so the two sides are one vocabulary and this comparison covers
    every object; the branch is kept because that is a measurement, not a guarantee.
    """
    if entry is None:
        return "asset_absent_from_library"
    asset_class = entry["class_en"].lower()
    if not asset_class:
        return "asset_absent_from_library"
    if category == asset_class:
        return "asset_class_matches_the_object_category"
    if category not in vocabulary:
        return "object_category_absent_from_library_vocabulary"
    return "asset_class_contradicts_the_object_category"


def fallback_scale_signature(
    scale: np.ndarray | None, observed: np.ndarray | None
) -> str:
    """Whether this object bears the signature of the unscaled small-object fallback.

    Line 6414 returns ``[1, 1, 1]`` when the pixel-bbox estimator reports an anisotropy
    above five, and line 6411's estimator returns ``1, 1`` on its own failures.  Neither
    the anisotropy nor the estimator's return value is written to the placement
    document, so the branch cannot be isolated from what survives to disk.  What can be
    isolated is its signature: a scale of exactly one on all three axes, on an object
    whose observed footprint puts it on the small-object path.

    This is therefore an upper bound and is named as one.  A scale of exactly one is
    also what a correct estimate of one produces, and what the clamps at lines 6425 and
    6427 produce when the estimate leaves the permitted range and lands on a bound.  The
    audit reports the bound and the subset within it that does visible damage, because
    the second number is what decides whether the branch is worth changing.
    """
    if scale is None:
        return "no_scale_recorded"
    if not bool(np.all(np.abs(scale - 1.0) < 1e-6)):
        return "scale_was_estimated"
    branch = production_scale_branch(observed)
    if branch is None:
        return "unscaled_with_no_observed_box"
    if branch == "small_object_pixel_bbox_path_ignores_the_observed_box":
        return "unscaled_on_the_small_object_path"
    return "unscaled_on_the_large_object_path"


FALLBACK_SIGNATURES = {
    "unscaled_on_the_small_object_path",
    "unscaled_with_no_observed_box",
}


def audit_scene(
    placement: dict[str, Any],
    library: dict[str, dict[str, Any]],
    *,
    vocabulary: set[str],
    identity_tolerance: float,
    size_mismatch_factor: float,
    top_k: int,
    fallback_evidence_factor: float = 3.0,
    library_quantum_m: float = 0.0005,
) -> dict[str, Any]:
    obj_info = placement.get("obj_info", {})
    findings: list[dict[str, Any]] = []

    for object_id, info in sorted(obj_info.items()):
        if not isinstance(info, dict) or CAMERA.search(object_id):
            continue
        if STRUCTURAL.match(object_id):
            continue
        asset = (info.get("retrieved_asset") or "").strip()
        entry = library.get(asset) if asset else None
        category = category_of(object_id)
        length = as_vector(info.get("length"))
        scale = as_vector(info.get("scale"))
        observed = as_vector(info.get("pcd_obb_size"))
        native = implied_native_size(length, scale)
        bbx = entry["bbx_m"] if entry else None

        # Verification of the identity, not a defect test.  Compared on sorted edges
        # so that an axis permutation between the library's convention and Blender's
        # cannot be mistaken for a disagreement.
        identity = {
            "computed_native_size_m": None if native is None else native.tolist(),
            "library_bbx_m": None if bbx is None else bbx.tolist(),
            "identity_worst_edge_ratio": None,
            "identity_holds": None,
            "identity_mismatch_shape": None,
            "identity_sorted_edge_ratios": None,
            "identity_geometric_mean_ratio": None,
        }
        if native is not None and bbx is not None:
            inside, worst = edges_agree_within_quantisation(
                native, bbx, quantum=library_quantum_m, tolerance=identity_tolerance
            )
            identity["identity_worst_edge_ratio"] = worst
            identity["identity_holds"] = inside
            if not inside:
                shape, ratios, geometric = identity_mismatch_shape(
                    native,
                    bbx,
                    quantum=library_quantum_m,
                    tolerance=identity_tolerance,
                )
                identity["identity_mismatch_shape"] = shape
                identity["identity_sorted_edge_ratios"] = ratios
                identity["identity_geometric_mean_ratio"] = geometric

        rendered_over_library = None
        rendered_over_library_longest = None
        if length is not None and bbx is not None:
            rendered_over_library = equivalent_linear_factor(length, bbx)
            rendered_over_library_longest = longest_edge_ratio(length, bbx)

        verdict = classify_retrieval(category, entry, vocabulary)
        relation = (
            label_relation(category, entry["class_en"])
            if entry and entry["class_en"]
            else None
        )
        name_relation = asset_name_relation(category, asset)

        signature = fallback_scale_signature(scale, observed)
        # With the scale at exactly one the rendered box is the asset's native box, so
        # this is the distance between the asset the fallback accepted and the depth
        # evidence it declined to use.  The library entry is the independent witness
        # that the rendered extent is the asset's authored size and not an artefact.
        against_evidence = size_disagreement(length, observed)

        reasons: list[str] = []
        # A disagreement between the curated bucket and the category is a defect only
        # when the asset's own identifier also fails to corroborate the category.  The
        # identifier may excuse, never accuse: a match is positive evidence, a non-match
        # is only the absence of it.
        if (
            verdict == "asset_class_contradicts_the_object_category"
            and relation == "shares_nothing"
            and name_relation not in CORROBORATING
        ):
            reasons.append("retrieved_asset_is_a_different_kind_of_object")
        signed = [
            value
            for value in (rendered_over_library, rendered_over_library_longest)
            if value is not None
        ]
        if signed:
            if max(signed) > size_mismatch_factor:
                reasons.append("rendered_size_far_larger_than_the_authored_asset")
            if min(signed) < 1.0 / size_mismatch_factor:
                reasons.append("rendered_size_far_smaller_than_the_authored_asset")
        if (
            signature in FALLBACK_SIGNATURES
            and (against_evidence["worst"] or 1.0) > fallback_evidence_factor
        ):
            reasons.append("unscaled_asset_is_far_larger_than_the_depth_evidence")

        findings.append(
            {
                "object_id": object_id,
                "category": category,
                "retrieved_asset": asset or None,
                "asset_class_en": entry["class_en"] if entry else None,
                "asset_retrieval_class_en": (
                    entry["retrieval_class_en"] if entry else None
                ),
                "library_scaling_strategy": (
                    entry["scaling_strategy"] if entry else None
                ),
                "length_m": None if length is None else length.tolist(),
                "scale": None if scale is None else scale.tolist(),
                "pcd_obb_size_m": None if observed is None else observed.tolist(),
                "rendered_over_library_volume_factor": rendered_over_library,
                "rendered_over_library_longest_edge_ratio": (
                    rendered_over_library_longest
                ),
                "retrieval_verdict": verdict,
                "label_relation": relation,
                "asset_name_relation": name_relation,
                "fallback_scale_signature": signature,
                "rendered_over_observed_volume_factor": (
                    against_evidence["volume_factor"]
                ),
                "rendered_over_observed_longest_edge_ratio": (
                    against_evidence["longest_edge_ratio"]
                ),
                "defect_reasons": reasons,
                **identity,
            }
        )

    joined = [item for item in findings if item["library_bbx_m"] is not None]
    identity_checked = [
        item for item in findings if item["identity_holds"] is not None
    ]
    identity_failed = [item for item in identity_checked if not item["identity_holds"]]
    identity_failed.sort(key=lambda item: -(item["identity_worst_edge_ratio"] or 0.0))

    shape_counts: dict[str, int] = {}
    for item in identity_failed:
        key = item["identity_mismatch_shape"] or "unknown"
        shape_counts[key] = shape_counts.get(key, 0) + 1

    # Denominator as well as numerator, so a strategy with many objects and few
    # failures cannot be mistaken for the cause.
    by_strategy: dict[str, dict[str, int]] = {}
    for item in identity_checked:
        key = item["library_scaling_strategy"] or "unrecorded"
        bucket = by_strategy.setdefault(key, {"checked": 0, "failed": 0})
        bucket["checked"] += 1
        if not item["identity_holds"]:
            bucket["failed"] += 1

    verdict_counts: dict[str, int] = {}
    for item in findings:
        verdict_counts[item["retrieval_verdict"]] = (
            verdict_counts.get(item["retrieval_verdict"], 0) + 1
        )

    contradictions = [
        item
        for item in findings
        if item["retrieval_verdict"] == "asset_class_contradicts_the_object_category"
    ]
    contradictions.sort(key=lambda item: item["object_id"])
    relation_counts: dict[str, int] = {}
    for item in contradictions:
        key = item["label_relation"] or "unknown"
        relation_counts[key] = relation_counts.get(key, 0) + 1
    substitutions = [
        item
        for item in contradictions
        if item["label_relation"] == "shares_nothing"
        and item["asset_name_relation"] not in CORROBORATING
    ]
    excused_by_the_asset_name = [
        item
        for item in contradictions
        if item["label_relation"] == "shares_nothing"
        and item["asset_name_relation"] in CORROBORATING
    ]
    substitutions_on_no_evidence = [
        item
        for item in substitutions
        if item["asset_name_relation"] == "asset_name_is_opaque"
    ]

    signature_counts: dict[str, int] = {}
    for item in findings:
        key = item["fallback_scale_signature"]
        signature_counts[key] = signature_counts.get(key, 0) + 1
    fallback_candidates = [
        item for item in findings if item["fallback_scale_signature"] in FALLBACK_SIGNATURES
    ]
    fallback_damaging = [
        item
        for item in fallback_candidates
        if "unscaled_asset_is_far_larger_than_the_depth_evidence"
        in item["defect_reasons"]
    ]
    fallback_damaging.sort(
        key=lambda item: -(item["rendered_over_observed_longest_edge_ratio"] or 0.0)
    )

    flagged = [item for item in findings if item["defect_reasons"]]
    reason_counts: dict[str, int] = {}
    for item in flagged:
        for reason in item["defect_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    object_count = len(findings)
    return {
        "object_count": object_count,
        "joined_to_library_count": len(joined),
        "join_rate": len(joined) / object_count if object_count else 0.0,
        "identity_checked_count": len(identity_checked),
        "identity_failed_count": len(identity_failed),
        "identity_pass_rate": (
            (len(identity_checked) - len(identity_failed)) / len(identity_checked)
            if identity_checked
            else 0.0
        ),
        "worst_identity_failures": identity_failed[: max(top_k, 0)],
        "identity_mismatch_shapes": dict(sorted(shape_counts.items())),
        "identity_by_scaling_strategy": dict(sorted(by_strategy.items())),
        "retrieval_verdict_counts": dict(sorted(verdict_counts.items())),
        "contradiction_label_relations": dict(sorted(relation_counts.items())),
        "substitution_count": len(substitutions),
        "substitution_share_of_scene": (
            len(substitutions) / object_count if object_count else 0.0
        ),
        "substitutions_resting_on_an_opaque_asset_name_count": len(
            substitutions_on_no_evidence
        ),
        "excused_by_the_asset_name_count": len(excused_by_the_asset_name),
        "excused_by_the_asset_name": [
            {
                "object_id": item["object_id"],
                "category": item["category"],
                "retrieved_asset": item["retrieved_asset"],
                "asset_class_en": item["asset_class_en"],
                "asset_name_relation": item["asset_name_relation"],
            }
            for item in excused_by_the_asset_name
        ],
        "fallback_signature_counts": dict(sorted(signature_counts.items())),
        "fallback_candidate_count": len(fallback_candidates),
        "fallback_candidate_share_of_scene": (
            len(fallback_candidates) / object_count if object_count else 0.0
        ),
        "fallback_damaging_count": len(fallback_damaging),
        "fallback_damaging_share_of_scene": (
            len(fallback_damaging) / object_count if object_count else 0.0
        ),
        "fallback_damaging": [
            {
                "object_id": item["object_id"],
                "category": item["category"],
                "retrieved_asset": item["retrieved_asset"],
                "rendered_size_m": item["length_m"],
                "observed_size_m": item["pcd_obb_size_m"],
                "library_bbx_m": item["library_bbx_m"],
                "identity_holds": item["identity_holds"],
                "longest_edge_over_the_evidence": (
                    item["rendered_over_observed_longest_edge_ratio"]
                ),
                "volume_factor_over_the_evidence": (
                    item["rendered_over_observed_volume_factor"]
                ),
            }
            for item in fallback_damaging[: max(top_k, 0)]
        ],
        "retrieval_substitutions": [
            {
                "object_id": item["object_id"],
                "category": item["category"],
                "retrieved_asset": item["retrieved_asset"],
                "asset_class_en": item["asset_class_en"],
                "asset_retrieval_class_en": item["asset_retrieval_class_en"],
                "asset_name_relation": item["asset_name_relation"],
            }
            for item in substitutions
        ],
        "retrieval_contradictions": [
            {
                "object_id": item["object_id"],
                "category": item["category"],
                "retrieved_asset": item["retrieved_asset"],
                "asset_class_en": item["asset_class_en"],
                "asset_retrieval_class_en": item["asset_retrieval_class_en"],
                "label_relation": item["label_relation"],
            }
            for item in contradictions
        ],
        "reason_counts": dict(sorted(reason_counts.items())),
        "reason_flag_rates": {
            name: count / object_count if object_count else 0.0
            for name, count in sorted(reason_counts.items())
        },
        "all_findings": findings,
        "policy": {
            "the_library_is_an_external_reference_already_inside_the_pipeline": True,
            "identity_is_verified_against_authored_dimensions_not_assumed": True,
            "identity_tolerance_accounts_for_the_three_decimal_rounding_of_bbx": True,
            "identity_failures_are_classified_by_shape_not_pooled_into_one_rate": True,
            "the_mesh_identity_is_by_construction_the_library_agreement_is_a_rate": True,
            "a_substitution_needs_both_the_curated_class_and_the_asset_name_to_fail": True,
            "the_asset_name_may_excuse_a_disagreement_and_may_never_create_one": True,
            "finer_or_coarser_labels_for_one_object_are_not_a_retrieval_error": True,
            "label_relation_is_a_heuristic_over_curated_labels_not_an_exact_test": True,
            "the_fallback_candidate_count_is_an_upper_bound_not_a_branch_hit_count": True,
            "nothing_is_modified_by_this_tool": True,
        },
    }


def format_box(values: Any, digits: int = 3) -> str:
    if values is None:
        return "n/a"
    return "[" + ",".join(f"{float(v):.{digits}f}" for v in values) + "]"


def format_factor(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--placement", type=Path, required=True)
    parser.add_argument(
        "--asset-info-csv",
        type=Path,
        default=Path("asset_data/imaginarium_asset_info.csv"),
    )
    parser.add_argument("--identity-tolerance", type=float, default=0.02)
    parser.add_argument("--library-quantum-m", type=float, default=0.0005)
    parser.add_argument("--size-mismatch-factor", type=float, default=3.0)
    parser.add_argument("--fallback-evidence-factor", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args()

    library = load_asset_library(args.asset_info_csv)
    vocabulary = library_vocabulary(library)
    with args.placement.open("r", encoding="utf-8") as handle:
        placement = json.load(handle)

    result = audit_scene(
        placement,
        library,
        vocabulary=vocabulary,
        identity_tolerance=args.identity_tolerance,
        library_quantum_m=args.library_quantum_m,
        size_mismatch_factor=args.size_mismatch_factor,
        fallback_evidence_factor=args.fallback_evidence_factor,
        top_k=args.top_k,
    )
    report = {
        "schema_version": "sceneproof_asset_library_join_v3",
        "scene": args.scene,
        "library_asset_count": len(library),
        "library_vocabulary_size": len(vocabulary),
        "thresholds": {
            "identity_tolerance": args.identity_tolerance,
            "library_quantum_m": args.library_quantum_m,
            "size_mismatch_factor": args.size_mismatch_factor,
            "fallback_evidence_factor": args.fallback_evidence_factor,
        },
        **result,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out_report.resolve()}")

    print(
        "JOIN objects={} joined={} ({:.0%}) library_assets={} vocabulary={}".format(
            result["object_count"],
            result["joined_to_library_count"],
            result["join_rate"],
            len(library),
            len(vocabulary),
        )
    )
    print(
        "IDENTITY length/scale vs authored bbx: {}/{} hold ({:.1%})".format(
            result["identity_checked_count"] - result["identity_failed_count"],
            result["identity_checked_count"],
            result["identity_pass_rate"],
        )
    )
    for item in result["worst_identity_failures"]:
        print(
            "    {}: {} computed={} authored={} worst_edge={}x".format(
                item["object_id"],
                item["identity_mismatch_shape"],
                format_box(item["computed_native_size_m"]),
                format_box(item["library_bbx_m"]),
                format_factor(item["identity_worst_edge_ratio"]),
            )
        )
        print(
            "        ratios={} gmean={} asset={} strategy={}".format(
                format_box(item["identity_sorted_edge_ratios"]),
                format_factor(item["identity_geometric_mean_ratio"]),
                item["retrieved_asset"],
                item["library_scaling_strategy"] or "unrecorded",
            )
        )
    if result["identity_mismatch_shapes"]:
        print("  shape of the disagreement:")
        for name, count in result["identity_mismatch_shapes"].items():
            print(f"    {name}={count}")
    if result["identity_by_scaling_strategy"]:
        print("  by scaling_strategy (failed/checked):")
        for name, bucket in result["identity_by_scaling_strategy"].items():
            rate = bucket["failed"] / bucket["checked"] if bucket["checked"] else 0.0
            print(f"    {name}={bucket['failed']}/{bucket['checked']} ({rate:.0%})")
    print("RETRIEVAL verdicts:")
    for name, count in result["retrieval_verdict_counts"].items():
        share = count / result["object_count"] if result["object_count"] else 0.0
        print(f"    {name}={count} ({share:.0%})")
    if result["contradiction_label_relations"]:
        print("  of the contradictions, how the two curated labels relate:")
        for name, count in result["contradiction_label_relations"].items():
            print(f"    {name}={count}")
    print(
        "SUBSTITUTIONS a different kind of object entirely: {} ({:.0%} of scene)".format(
            result["substitution_count"], result["substitution_share_of_scene"]
        )
    )
    print(
        "  of these, resting on an opaque asset name and so on no evidence: {}".format(
            result["substitutions_resting_on_an_opaque_asset_name_count"]
        )
    )
    for item in result["retrieval_substitutions"]:
        print(
            "    {}: {} -> {} ({})".format(
                item["object_id"],
                item["category"],
                item["asset_class_en"],
                item["retrieved_asset"],
            )
        )
    print(
        "EXCUSED curated class disagrees but the asset name agrees: {}".format(
            result["excused_by_the_asset_name_count"]
        )
    )
    for item in result["excused_by_the_asset_name"]:
        print(
            "    {}: {} -> class {} but asset {} ({})".format(
                item["object_id"],
                item["category"],
                item["asset_class_en"],
                item["retrieved_asset"],
                item["asset_name_relation"],
            )
        )
    print(
        "FALLBACK unscaled on the small-object path: {} ({:.0%} of scene, UPPER BOUND)".format(
            result["fallback_candidate_count"],
            result["fallback_candidate_share_of_scene"],
        )
    )
    for name, count in result["fallback_signature_counts"].items():
        print(f"    {name}={count}")
    print(
        "  of these, the unscaled asset is far larger than the depth evidence:"
        " {} ({:.0%} of scene)".format(
            result["fallback_damaging_count"],
            result["fallback_damaging_share_of_scene"],
        )
    )
    for item in result["fallback_damaging"]:
        print(
            "    {}: rendered={} observed={} longest={}x vol={}x".format(
                item["object_id"],
                format_box(item["rendered_size_m"]),
                format_box(item["observed_size_m"]),
                format_factor(item["longest_edge_over_the_evidence"]),
                format_factor(item["volume_factor_over_the_evidence"]),
            )
        )
        print(
            "        authored={} identity_holds={} asset={}".format(
                format_box(item["library_bbx_m"]),
                item["identity_holds"],
                item["retrieved_asset"],
            )
        )
    if result["reason_counts"]:
        print("  defect reasons (count, share of scene):")
        for name, count in result["reason_counts"].items():
            print(f"    {name}={count} ({result['reason_flag_rates'][name]:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
