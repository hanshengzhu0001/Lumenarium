#!/usr/bin/env python3
"""SceneProof Fix97/98: attribute each visible size defect by arithmetic, not by heuristic.

The one exact identity everything rests on
------------------------------------------
``S4_blender_layout_and_corr.py`` line 7277 records
``obj_info["length"] = list(obj.dimensions)``, and a Blender object's ``dimensions``
is its local bounding box multiplied by its scale, per local axis.  Therefore

    retrieved asset's native size  =  length / scale

exactly.  Each object then carries three comparable boxes:

    pcd_obb_size        what depth estimation observed
    length / scale      what the retrieved asset natively is
    length              what was rendered

Confirmed on real data: ``pillow_16`` has native 2.76 x 1.04 x 0.86, a real sofa,
against an observed box of 1.07 x 0.25 x 0.25, a correct pillow.

What the first run of this file got wrong, and what replaced it
--------------------------------------------------------------
Four defects were found by reading its own output, and all four are fixed here.

1.``gaming_table_2`` occupies 55 per cent of the casino frame and was dropped
    from the report entirely, because every test was one of *internal consistency*
    and its chain is self-consistent: rendered 5.80 x 4.19 x 0.92 reproduces
    observed 5.79 x 4.19 x 0.92.  It is consistently wrong.Removing Fix96's
    peer and volume outlier tests removed the only external reference, which was
    an over-correction.  The replacement is sharper than what was removed:
    objects sharing one ``retrieved_asset`` have the same native size *by
    construction*, so their rendered sizes are directly comparable, and a member
    disagreeing with its same-asset peers is evidence rather than noise.  Fix96
    compared against the scene median volume, which mixes bottles with sofas.

2.  The claim that a large ``native/observed`` ratio means retrieval returned an
    asset too big was wrong.  ``ceiling_fan_0`` observed 0.27 x 0.45 x 0.39, asset
    1.46 x 1.46 x 0.52, rendered 1.27 x 1.27 x 0.29: the1.27 m fan is right and
    the 0.45 m observation is wrong.  The ratio names a *disagreement between two
    references*, and says nothing about which one is correct.  It is now reported
    as a number, never as a reason on its own.

3.  Reasons fired on objects whose problem the scale had already corrected.
    ``trash_bin_0`` retrieved a 1.72 m bin asset, and the scale brought it to
    0.47 m against an observed 0.48 m: the pipeline behaved correctly, yet it was
    flagged.  Every size reason is now stated about the *rendered* size, which is
    what reaches the image.

4.  The category-versus-asset-name test matched in one direction only, so
    ``bookshelf_0`` retrieving ``0_SM_Shelf_2`` was flagged although``Shelf`` is a
    correct asset for a bookshelf.  Matching is now bidirectional.

Why the stage taxonomy asks a different question now
----------------------------------------------------
The first version named the mechanism that last touched the number, which put82
per cent of the casino into ``scale_overwritten_by_group_consistency``.  That is
true and useless: the group pass explains where a number came from, not which
layer is responsible.  Because ``rendered = native x scale`` is exact, there are
only three possibilities, and they are decidable:

    rendered_size_followed_the_observation   the render agrees with depth
    rendered_size_followed_the_asset         the scale left the asset roughly as-is
    rendered_size_followed_neither           the scale invented a third size

This is symmetric and does not pretend to know which reference deserved to win.
The mechanisms - abstention, clamp bound, group overwrite, production branch - are
kept as separate annotations, since each object can carry several.

The single mechanism behind the worst offenders
-----------------------------------------------
Counted explicitly as ``root_cause_rollup``: an observed box that is tiny or a
degenerate sliver, an asset that disagrees with it by a large factor, and a render
that followed the asset.  In ``livingroom_10`` this is four spurious 1.9 m
bookshelves whose observed boxes are 2 to 17 cm fragments, together about a fifth
of the frame, plus ``paper_cup_1`` at 46.8 per cent of the street frame as a 2 m
duct.  For41 per cent of ``livingroom_10`` the scale is exactly one, meaning no
scaling decision was made at all.

What this file still cannot see
-------------------------------
Wrong shape at plausible size, such as a chair retrieved as a curved sheet.  That
needs the mesh or the render, and is declared rather than left implicit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_physical_realizability import STRUCTURAL, CAMERA  # noqa: E402
from sceneproof_scene_defect_screen_fix96 import (  # noqa: E402
    DEFAULT_LENS_MM,
    DEFAULT_RESOLUTION,
    DEFAULT_SENSOR_MM,
    as_matrix,
    as_vector,
    category_of,
    framed_pixel_area,
    project_to_pixels,
    world_corners,
)

# Mirrors SCALE_THRESHOLD in estimate_scale_factors_for_object.  A factor sitting on
# a bound is a capped runaway estimate, which is a different failure from a factor
# that merely happens to be large.
CLAMP_BOUNDS = (0.1, 5.0)

# Mirrors the small-object branch predicate at line 6410.
SMALL_OBJECT_FOOTPRINT_LIMIT = 0.25

# Tokens that carry no semantic content in an asset name.
GENERIC_ASSET_TOKENS = frozenset(
    {"sm", "sk", "uf", "lod", "compiled", "mesh", "var", "obj", "fbx", "group", "packed"}
)

ALPHABETIC_RUN = re.compile(r"[a-z]+")


def scale_is_exact_abstention(scale: np.ndarray) -> bool:
    """True when the scale is exactly the identity.

    ``estimate_scale_factors_for_object`` initialises ``scale_factors`` to
    ``[1, 1, 1]`` and returns it unchanged at line 6414 when the pixel-derived
    factors disagree by more than fivefold, which it reads as a broken pose.
    ``cal_scale_refer_bbox`` likewise returns ``1, 1`` when projection fails.  An
    exactly unit scale is therefore an abstention, and the rendered size is the
    retrieved asset's own.  A genuinely computed factor of 1.0 to nine decimals has
    negligible probability.
    """
    return bool(np.all(np.abs(scale - 1.0) < 1e-9))


def scale_components_at_clamp_bound(scale: np.ndarray) -> list[float]:
    return [
        float(component)
        for component in scale
        if any(abs(float(component) - bound) < 1e-6 for bound in CLAMP_BOUNDS)
    ]


def observed_footprint_products(observed: np.ndarray) -> float:
    """max pairwise product of the observed box edges, as line 6396 computes it."""
    return float(
        max(
            observed[0] * observed[1],
            observed[0] * observed[2],
            observed[1] * observed[2],
        )
    )


def production_scale_branch(observed: np.ndarray | None) -> str | None:
    """Which branch of estimate_scale_factors_for_object this object took."""
    if observed is None or (observed <= 0).any():
        return None
    if observed_footprint_products(observed) <= SMALL_OBJECT_FOOTPRINT_LIMIT:
        return "small_object_pixel_bbox_path_ignores_the_observed_box"
    return "large_object_observed_box_ratio_path"


def implied_native_size(
    length: np.ndarray | None, scale: np.ndarray | None
) -> np.ndarray | None:
    """The retrieved asset's own dimensions, from length = native * scale."""
    if length is None or scale is None:
        return None
    if np.any(np.abs(scale) < 1e-9):
        return None
    return length / scale


def sorted_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Edge-wise ratio after sorting both boxes, so axis order cannot confound it."""
    return np.sort(numerator)[::-1] / np.maximum(np.sort(denominator)[::-1], 1e-9)


def worst_axis_factor(ratio: np.ndarray) -> float:
    """The largest per-edge disagreement, as a factor that is always >= 1.

    Kept as a diagnostic only.  It was used to decide reasons in an earlier version
    and that was wrong: taking the maximum over edges hands the decision to the
    thinnest axis, which is where depth estimation is least reliable and where the
    visual consequence is smallest.  ``curtain_0`` rendered 1.62 x 0.09 x 2.22
    against an observed 1.62 x 0.62 x 2.22 scored 6.62 by this measure, although its
    two long edges agree to three decimal places and the render is essentially
    correct; the whole disagreement was9 cm of curtain thickness against 62 cm.
    """
    positive = np.maximum(np.abs(ratio), 1e-9)
    return float(np.max(np.maximum(positive, 1.0 / positive)))


def equivalent_linear_factor(
    first: np.ndarray | None, second: np.ndarray | None
) -> float | None:
    """The uniform linear scale factor between two boxes.

    Defined as the cube root of the volume ratio, which is dimensionally what a
    scale factor is, so it cannot be dominated by any single axis.  For the scaling
    chain it is also exactly consistent: since ``rendered = native * scale`` per
    axis, the factor from the asset to the render is the cube root of the product of
    the scale components.

    Values below one mean the first box is the smaller one, so the sign of the
    disagreement is preserved and a caller can distinguish too large from too small.
    """
    if first is None or second is None:
        return None
    if (first <= 0).any() or (second <= 0).any():
        return None
    return float(np.cbrt(np.prod(first) / np.prod(second)))


def symmetric_factor(value: float | None) -> float | None:
    """Fold a ratio onto [1, inf), so half the size and twice the size both give 2."""
    if value is None or value <= 0:
        return None
    return max(value, 1.0 / value)


def longest_edge_ratio(
    first: np.ndarray | None, second: np.ndarray | None
) -> float | None:
    """Ratio of the two boxes' longest edges.

    Complements the volume factor, and is needed because that factor has a blind
    spot: a rod and a blob of similar volume score close to one.  ``pen_0`` renders
    0.05 x 0.05 x 1.81, a 1.81 m rod visible through the whole bedroom image, against
    an observed 0.06 x 0.44 x 0.08.  Their volumes differ by only 2.14, so the volume
    factor is 1.29 and says nothing, while the longest edges differ by 4.11.

    The longest edge cannot be dominated by the thinnest axis, which is whatruled out
    using an aspect ratio here: ``curtain_0`` has a rendered aspect of 24.7 against an
    observed 3.58, so an aspect test brings the thin-axis artefact straight back,
    whereas its longest edges are equal to three decimal places.
    """
    if first is None or second is None:
        return None
    if (first <= 0).any() or (second <= 0).any():
        return None
    return float(np.max(first) / np.max(second))


def size_disagreement(
    first: np.ndarray | None, second: np.ndarray | None
) -> dict[str, float | None]:
    """Two independent views of how differently sized two boxes are.

    ``volume_factor`` catches being uniformly the wrong size; ``longest_edge_ratio``
    catches the wrong extent at a similar volume.  Both are reported signed, so a
    caller can tell too large from too small, and ``worst`` is the symmetric folding
    of whichever is further from one.
    """
    volume = equivalent_linear_factor(first, second)
    longest = longest_edge_ratio(first, second)
    candidates = [value for value in (volume, longest) if value is not None]
    worst = None
    if candidates:
        worst = max(symmetric_factor(value) or 1.0 for value in candidates)
    return {
        "volume_factor": volume,
        "longest_edge_ratio": longest,
        "worst": worst,
    }


def boxes_agree(first: np.ndarray, second: np.ndarray, *, tolerance: float) -> bool:
    ratio = sorted_ratio(first, second)
    return bool(np.all(np.abs(ratio - 1.0) <= tolerance))


def normalise_asset_name(asset: str) -> str:
    """Lowercase alphabetic content of an asset name, generic tokens removed."""
    return "".join(informative_asset_runs(asset, minimum=1))


def informative_asset_runs(asset: str, *, minimum: int = 4) -> list[str]:
    return [
        run
        for run in ALPHABETIC_RUN.findall(asset.lower())
        if run not in GENERIC_ASSET_TOKENS and len(run) >= minimum
    ]


def category_tokens(category: str) -> list[str]:
    """Tokens of a category name long enough to be searched for in an asset name."""
    return [token for token in category.lower().split("_") if len(token) >= 4]


def retrieval_name_check(
    category: str, asset: str | None, scene_categories: set[str]
) -> tuple[list[str], dict[str, Any]]:
    """Compare the object's category against its retrieved asset's name.

    Matching is bidirectional.  A one-directional test flagged ``bookshelf_0`` for
    retrieving ``0_SM_Shelf_2``, because ``bookshelf`` does not appear inside
    ``shelf``; the reverse containment is what makes that a correct retrieval.

    Deliberately conservative in two further ways.  Asset names in this library are
    often opaque ids such as ``b_114``, which say nothing either way, so those
    abstain rather than count as agreement and the reported rate covers only the
    testable subset.  Category names shorter than four characters, such as ``pen``,
    also abstain, since a three-letter substring search is not evidence.
    """
    report: dict[str, Any] = {
        "asset_name_normalised": None,
        "asset_name_is_testable": False,
        "asset_name_names_these_other_categories": [],
        "asset_name_does_not_name_its_category": False,
    }
    if not asset:
        return [], report
    normalised = normalise_asset_name(asset)
    report["asset_name_normalised"] = normalised
    own = category_tokens(category)
    if len(normalised) < 4 or not own:
        return [], report
    report["asset_name_is_testable"] = True

    category_joined = category.lower().replace("_", "")
    asset_runs = informative_asset_runs(asset)
    forward = any(token in normalised for token in own)
    backward = any(run in category_joined for run in asset_runs)
    if forward or backward:
        return [], report

    others = sorted(
        other
        for other in scene_categories
        if other != category
        and any(token in normalised for token in category_tokens(other))
    )
    report["asset_name_names_these_other_categories"] = others
    if others:
        return ["retrieved_asset_names_a_different_category_in_this_scene"], report
    report["asset_name_does_not_name_its_category"] = True
    return ["retrieved_asset_name_does_not_name_its_category"], report


def corpus_key(asset: str, category: str) -> str:
    return f"{asset}||{category}"


def collect_asset_sizes(placement: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    """Longest rendered edge per object, grouped by asset and by asset-with-category.

    The same asset has the same native size everywhere, so its rendered sizes across
    scenes are comparable and give an external reference needing no hand-written size
    table.Grouping by asset alone is not enough, because one asset is reused for
    unrelated categories whose correct sizes differ: ``a_SM_CartonGarbage05`` serves
    both ``stack_of_chips`` in the casino and ``discarded_wooden_board`` in the street
    scene, and pooling them let the casino's oversized chips dominate the median and
    accuse the street scene's correctly sized boards.  Conditioning on the category
    separates the two populations.
    """
    by_asset: dict[str, list[float]] = {}
    by_asset_and_category: dict[str, list[float]] = {}
    for object_id, info in (placement.get("obj_info") or {}).items():
        if not isinstance(info, dict) or CAMERA.search(object_id):
            continue
        if STRUCTURAL.match(object_id):
            continue
        asset = info.get("retrieved_asset")
        length = as_vector(info.get("length"))
        if asset and length is not None and (length > 0).all():
            longest = float(np.max(length))
            by_asset.setdefault(asset, []).append(longest)
            by_asset_and_category.setdefault(
                corpus_key(asset, category_of(object_id)), []
            ).append(longest)
    return {"by_asset": by_asset, "by_asset_and_category": by_asset_and_category}


def summarise_asset_corpus(
    per_scene: list[dict[str, dict[str, list[float]]]]
) -> dict[str, dict[str, dict[str, Any]]]:
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for grouping in ("by_asset", "by_asset_and_category"):
        merged: dict[str, list[float]] = {}
        for scene in per_scene:
            for key, values in (scene.get(grouping) or {}).items():
                merged.setdefault(key, []).extend(values)
        summary[grouping] = {
            key: {
                "count": len(values),
                "median_max_edge_m": float(np.median(values)),
                "min_max_edge_m": float(np.min(values)),
                "max_max_edge_m": float(np.max(values)),
            }
            for key, values in merged.items()
        }
    return summary


def which_reference_the_render_followed(
    *,
    observed: np.ndarray | None,
    scale: np.ndarray | None,
    length: np.ndarray | None,
    follow_factor: float,
) -> tuple[str, dict[str, float | None]]:
    """Decide whether the rendered size followed depth, the asset, or neither.

    Both distances are equivalent linear factors, so neither can be decided by a
    single thin axis.  The distance from the render to the asset is exactly the cube
    root of the product of the scale components, because ``rendered = native * scale``.
    """
    distances: dict[str, float | None] = {
        "rendered_over_observed_factor": None,
        "rendered_over_asset_factor": None,
        "rendered_over_observed_worst_axis": None,
    }
    if observed is None or scale is None or length is None:
        return "undetermined_scaling_chain_incomplete", distances

    to_observation = size_disagreement(length, observed)["worst"]
    to_asset = symmetric_factor(equivalent_linear_factor(length, length / scale))
    distances["rendered_over_observed_factor"] = to_observation
    distances["rendered_over_asset_factor"] = to_asset
    distances["rendered_over_observed_worst_axis"] = worst_axis_factor(
        sorted_ratio(length, observed)
    )
    if to_observation is None or to_asset is None:
        return "undetermined_scaling_chain_incomplete", distances

    observation_close = to_observation <= follow_factor
    asset_close = to_asset <= follow_factor
    if observation_close and to_observation <= to_asset:
        return "rendered_size_followed_the_observation", distances
    if asset_close:
        return "rendered_size_followed_the_asset", distances
    if observation_close:
        return "rendered_size_followed_the_observation", distances
    return "rendered_size_followed_neither", distances


def screen_scene(
    placement: dict[str, Any],
    *,
    lens_mm: float,
    sensor_mm: float,
    resolution: int,
    agreement_tolerance: float,
    follow_factor: float,
    degenerate_aspect: float,
    size_mismatch_factor: float,
    peer_mismatch_factor: float,
    minimum_peer_count: int,
    tiny_observed_edge_m: float,
    top_k: int,
    asset_corpus: dict[str, dict[str, Any]] | None = None,
    corpus_mismatch_factor: float = 2.0,
    minimum_corpus_count: int = 3,
) -> dict[str, Any]:
    obj_info = placement.get("obj_info", {})
    focal_px = lens_mm / sensor_mm * resolution

    camera_matrix = None
    for name, info in obj_info.items():
        if CAMERA.search(name) and isinstance(info, dict):
            camera_matrix = as_matrix(info.get("pose_matrix_for_blender"))
            break

    entries: list[dict[str, Any]] = []
    for object_id, info in sorted(obj_info.items()):
        if not isinstance(info, dict) or CAMERA.search(object_id):
            continue
        if STRUCTURAL.match(object_id):
            continue
        entries.append(
            {
                "object_id": object_id,
                "category": category_of(object_id),
                "retrieved_asset": info.get("retrieved_asset"),
                "group": info.get("group"),
                "length": as_vector(info.get("length")),
                "scale": as_vector(info.get("scale")),
                "observed": as_vector(info.get("pcd_obb_size")),
                "corners": world_corners(info),
            }
        )

    scene_categories = {item["category"] for item in entries}

    # Objects sharing one retrieved asset have the same native size by construction,
    # so their rendered sizes are directly comparable and a disagreement isolates
    # the scale factor.  This is the external reference the first version lacked,
    # and it is stricter than comparing against a scene median volume, which mixes
    # bottles with sofas.
    peers_by_asset: dict[str, list[float]] = {}
    for item in entries:
        if item["retrieved_asset"] and item["length"] is not None:
            peers_by_asset.setdefault(item["retrieved_asset"], []).append(
                float(np.max(item["length"]))
            )
    peer_median = {
        asset: float(np.median(values))
        for asset, values in peers_by_asset.items()
        if len(values) >= minimum_peer_count
    }

    # Group membership is read from the document rather than inferred from repeated
    # scale values, because the group pass at lines 7099 to 7126 makes repeated
    # values its intended outcome.
    group_members: dict[Any, list[tuple[str, tuple[float, ...]]]] = {}
    for item in entries:
        if item["group"] is not None and item["scale"] is not None:
            group_members.setdefault(item["group"], []).append(
                (item["object_id"], tuple(np.round(item["scale"], 9)))
            )
    group_shared_ids: set[str] = set()
    for members in group_members.values():
        if len(members) < 2:
            continue
        for object_id, vector in members:
            if sum(1 for _, other in members if other == vector) >= 2:
                group_shared_ids.add(object_id)

    findings: list[dict[str, Any]] = []
    for item in entries:
        observed, scale, length = item["observed"], item["scale"], item["length"]
        native = implied_native_size(length, scale)
        reasons: list[str] = []

        observed_aspect = None
        observed_is_tiny = None
        if observed is not None and (observed > 0).all():
            observed_aspect = float(observed.max() / max(observed.min(), 1e-9))
            observed_is_tiny = bool(observed.max() < tiny_observed_edge_m)
            if observed_aspect > degenerate_aspect:
                reasons.append("observed_box_is_a_degenerate_sliver")

        # Reported as numbers, never as a reason on their own.  A large value means
        # the two references disagree; it does not say which of them is wrong.
        # ceiling_fan_0 is the counterexample: asset 1.46 m, observed 0.45 m, and the
        # asset is right.
        asset_versus_observed = size_disagreement(native, observed)
        asset_over_observed = asset_versus_observed["worst"]

        # Stated about the rendered size, because that is what reaches the image, and
        # measured two independent ways so that neither a thin axis nor a coincidence
        # of volume can hide a defect.  trash_bin_0 retrieved a 1.72 m bin and the
        # scale brought it to 0.47 m against an observed 0.48 m, so nothing survived
        # into the render and both views agree it is fine.
        rendered_versus_observed = size_disagreement(length, observed)
        signed = [
            value
            for value in (
                rendered_versus_observed["volume_factor"],
                rendered_versus_observed["longest_edge_ratio"],
            )
            if value is not None
        ]
        if signed:
            if max(signed) > size_mismatch_factor:
                reasons.append("rendered_size_far_larger_than_what_depth_observed")
            if min(signed) < 1.0 / size_mismatch_factor:
                reasons.append("rendered_size_far_smaller_than_what_depth_observed")

        rendered_over_peer_median = None
        median = peer_median.get(item["retrieved_asset"] or "")
        if median and length is not None:
            rendered_over_peer_median = float(np.max(length)) / max(median, 1e-9)
            if (
                rendered_over_peer_median > peer_mismatch_factor
                or rendered_over_peer_median < 1.0 / peer_mismatch_factor
            ):
                reasons.append("rendered_size_disagrees_with_its_same_asset_peers")

        # The only reference that can see a family which is uniformly wrong inside one
        # scene.  All three bedroom picture frames render at 3.51 m with an in-scene
        # peer median of exactly 1.00, so nothing local can flag a 3.5 m picture
        # frame; across the corpus the same asset renders near 1.0 m in 37 instances.
        #
        # Both this and the in-scene peer test are MAJORITY references: they detect
        # that an object differs from the others, not that it is wrong.  Where the
        # majority is wrong they accuse the minority that is right, which is why
        # wall_mounted_picture_frame_3, the only correctly sized frame in its scene,
        # has an in-scene peer median of 0.42.  Conditioning the corpus on the
        # category as well as the asset removes the worst source of that inversion.
        rendered_over_corpus_median = None
        rendered_over_asset_only_corpus_median = None
        corpus_sample_count = 0
        asset = item["retrieved_asset"] or ""
        if asset and length is not None and asset_corpus:
            longest = float(np.max(length))
            by_pair = (asset_corpus.get("by_asset_and_category") or {}).get(
                corpus_key(asset, item["category"])
            )
            by_asset_only = (asset_corpus.get("by_asset") or {}).get(asset)
            if by_asset_only and by_asset_only.get("median_max_edge_m"):
                rendered_over_asset_only_corpus_median = longest / max(
                    float(by_asset_only["median_max_edge_m"]), 1e-9
                )
            if by_pair:
                corpus_sample_count = int(by_pair.get("count", 0))
                value = by_pair.get("median_max_edge_m")
                if value and corpus_sample_count >= minimum_corpus_count:
                    rendered_over_corpus_median = longest / max(float(value), 1e-9)
                    if (
                        rendered_over_corpus_median > corpus_mismatch_factor
                        or rendered_over_corpus_median < 1.0 / corpus_mismatch_factor
                    ):
                        reasons.append(
                            "rendered_size_disagrees_with_the_same_asset_across_scenes"
                        )

        # Only the variant that names another category present in this scene stays a
        # reason.  The weaker variant sits at 15 to 48 per cent of the objects, above
        # the rate at which a signal can be triaged, and it has known false positives
        # from plurals such as 0_steel_frame_shelves for a bookshelf and from
        # uninformative asset names such as a_SM_Decor_6.  Its precision is not
        # established, so it is reported as a field and not counted as a defect.
        name_reasons, name_report = retrieval_name_check(
            item["category"], item["retrieved_asset"], scene_categories
        )
        reasons.extend(
            reason
            for reason in name_reasons
            if reason == "retrieved_asset_names_a_different_category_in_this_scene"
        )

        abstained = scale is not None and scale_is_exact_abstention(scale)
        clamped = [] if scale is None else scale_components_at_clamp_bound(scale)

        followed, distances = which_reference_the_render_followed(
            observed=observed,
            scale=scale,
            length=length,
            follow_factor=follow_factor,
        )

        # The mechanism behind the worst offenders in four of five scenes: the render
        # followed the asset while the asset grossly disagreed with the observation.
        #
        # This counts a *mechanism*, not an error.Deciding whether the asset
        # deserved to win needs a reference this file does not have, and
        # ceiling_fan_0 is the case where the same mechanism produced the right
        # answer: observed 0.27 x 0.45 x 0.39 is wrong and the 1.27 m fan is right.
        #
        # An earlier version also required the observed box to be tiny or degenerate.
        # That conjunct only caused misses: bookshelf_0 observed 0.80 x 0.42 x 0.07
        # and rendered a 1.91 m shelf, which is the mechanism exactly, yet its
        # longest observed edge is 0.80 m and its aspect 11.4, so both gates let it
        # through.  Dropped.
        asset_won_against_a_disagreeing_observation = bool(
            asset_over_observed is not None
            and asset_over_observed > size_mismatch_factor
            and followed == "rendered_size_followed_the_asset"
        )
        if asset_won_against_a_disagreeing_observation:
            reasons.append("size_determined_by_the_asset_against_a_disagreeing_observation")

        projection = {
            "in_front_of_camera": None,
            "pixel_area_fraction": None,
            "fully_outside_frame": None,
            "pixel_bbox": None,
        }
        if camera_matrix is not None and item["corners"] is not None:
            pixels, in_front = project_to_pixels(
                item["corners"], camera_matrix, focal_px=focal_px, resolution=resolution
            )
            projection = framed_pixel_area(pixels, in_front, resolution)

        findings.append(
            {
                "object_id": item["object_id"],
                "category": item["category"],
                "retrieved_asset": item["retrieved_asset"],
                "group": item["group"],
                "length_m": None if length is None else length.tolist(),
                "scale": None if scale is None else scale.tolist(),
                "pcd_obb_size_m": None if observed is None else observed.tolist(),
                "implied_asset_native_size_m": (
                    None if native is None else native.tolist()
                ),
                "observed_box_aspect": observed_aspect,
                "observed_box_is_tiny": observed_is_tiny,
                "asset_over_observed_factor": asset_over_observed,
                "asset_over_observed_volume_factor": asset_versus_observed[
                    "volume_factor"
                ],
                "asset_over_observed_longest_edge_ratio": asset_versus_observed[
                    "longest_edge_ratio"
                ],
                "rendered_over_observed_volume_factor": rendered_versus_observed[
                    "volume_factor"
                ],
                "rendered_over_observed_longest_edge_ratio": rendered_versus_observed[
                    "longest_edge_ratio"
                ],
                "rendered_over_peer_median": rendered_over_peer_median,
                "rendered_over_corpus_median": rendered_over_corpus_median,
                "rendered_over_asset_only_corpus_median": (
                    rendered_over_asset_only_corpus_median
                ),
                "same_asset_corpus_count": corpus_sample_count,
                "same_asset_peer_count": len(
                    peers_by_asset.get(item["retrieved_asset"] or "", [])
                ),
                "world_size_reproduces_observed_box": (
                    None
                    if (length is None or observed is None)
                    else boxes_agree(length, observed, tolerance=agreement_tolerance)
                ),
                "production_scale_branch": production_scale_branch(observed),
                "scale_is_exactly_one_an_abstention": abstained,
                "scale_components_on_clamp_bound": clamped,
                "scale_shared_with_its_group": item["object_id"] in group_shared_ids,
                "size_determined_by_the_asset_against_a_disagreeing_observation": (
                    asset_won_against_a_disagreeing_observation
                ),
                **distances,
                **name_report,
                "defect_reasons": reasons,
                "size_was_set_by": followed,
                "screen_area_fraction": projection["pixel_area_fraction"],
                "in_front_of_camera": projection["in_front_of_camera"],
                "fully_outside_frame": projection["fully_outside_frame"],
                "screen_salience": (
                    (projection["pixel_area_fraction"] or 0.0) if reasons else 0.0
                ),
            }
        )

    flagged = [item for item in findings if item["defect_reasons"]]
    flagged.sort(key=lambda item: -item["screen_salience"])

    object_count = len(findings)
    reason_counts: dict[str, int] = {}
    for item in flagged:
        for reason in item["defect_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    # Printed so that a detector which fires on most of the scene is visibly useless
    # instead of quietly trusted.  That is how the first version's clamp detector,
    # at 80 to 90 per cent, was caught.
    reason_rates = {
        name: count / object_count if object_count else 0.0
        for name, count in reason_counts.items()
    }

    stage_counts: dict[str, int] = {}
    mechanism_counts = {
        "scale_is_exactly_one_an_abstention": 0,
        "scale_components_on_clamp_bound": 0,
        "scale_shared_with_its_group": 0,
        "small_object_pixel_bbox_path_ignores_the_observed_box": 0,
    }
    for item in findings:
        stage_counts[item["size_was_set_by"]] = (
            stage_counts.get(item["size_was_set_by"], 0) + 1
        )
        if item["scale_is_exactly_one_an_abstention"]:
            mechanism_counts["scale_is_exactly_one_an_abstention"] += 1
        if item["scale_components_on_clamp_bound"]:
            mechanism_counts["scale_components_on_clamp_bound"] += 1
        if item["scale_shared_with_its_group"]:
            mechanism_counts["scale_shared_with_its_group"] += 1
        if (
            item["production_scale_branch"]
            == "small_object_pixel_bbox_path_ignores_the_observed_box"
        ):
            mechanism_counts[
                "small_object_pixel_bbox_path_ignores_the_observed_box"
            ] += 1

    inherited = [
        item
        for item in findings
        if item["size_determined_by_the_asset_against_a_disagreeing_observation"]
    ]
    testable = [item for item in findings if item["asset_name_is_testable"]]
    return {
        "object_count": object_count,
        "flagged_count": len(flagged),
        "camera_available": camera_matrix is not None,
        "focal_px": focal_px,
        "reason_counts": dict(sorted(reason_counts.items())),
        "reason_flag_rates": {name: reason_rates[name] for name in sorted(reason_rates)},
        "stage_counts": dict(sorted(stage_counts.items())),
        "mechanism_counts": mechanism_counts,
        "root_cause_rollup": {
            "objects_whose_size_the_asset_determined": len(inherited),
            "share_of_scene": len(inherited) / object_count if object_count else 0.0,
            "screen_area_they_cover": sum(
                item["screen_area_fraction"] or 0.0 for item in inherited
            ),
            "this_counts_a_mechanism_not_an_error": True,
            "object_ids": [item["object_id"] for item in inherited],
        },
        "asset_name_testable_count": len(testable),
        "asset_name_opaque_count": object_count - len(testable),
        "asset_name_weak_verdict_count_reported_not_counted": sum(
            1 for item in findings if item["asset_name_does_not_name_its_category"]
        ),
        "worst_by_screen_area": flagged[: max(top_k, 0)],
        "all_findings": findings,
        "policy": {
            "native_asset_size_is_exact_arithmetic_from_length_over_scale": True,
            "production_small_object_branch_predicate_is_reproduced_exactly": True,
            "group_membership_is_read_not_inferred_from_repeated_scale_values": True,
            "size_reasons_are_stated_about_the_rendered_size_not_the_asset": True,
            "asset_over_observed_is_a_disagreement_not_an_attribution_of_blame": True,
            "same_asset_peers_share_a_native_size_by_construction": True,
            "a_family_uniformly_wrong_in_one_scene_needs_the_cross_scene_corpus": True,
            "peer_and_corpus_are_majority_references_they_detect_difference_not_error": True,
            "corpus_is_conditioned_on_asset_and_category_to_separate_reused_assets": True,
            "size_is_measured_by_volume_and_by_longest_edge_because_either_alone_misses": True,
            "weak_asset_name_verdict_is_reported_but_not_counted_as_a_defect": True,
            "wrong_shape_at_plausible_size_is_not_detectable_here": True,
            "no_pose_asset_or_scale_is_modified_by_this_tool": True,
        },
    }


def format_box(values: Any, digits: int = 2) -> str:
    if values is None:
        return "n/a"
    return "[" + ",".join(f"{float(v):.{digits}f}" for v in values) + "]"


def format_factor(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def build_corpus(paths: list[Path]) -> dict[str, dict[str, Any]]:
    per_scene = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                per_scene.append(collect_asset_sizes(json.load(handle)))
        except (OSError, ValueError) as error:
            print(f"  corpus skip {path}: {error}")
    return summarise_asset_corpus(per_scene)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene")
    parser.add_argument("--placement", type=Path)
    parser.add_argument("--lens-mm", type=float, default=DEFAULT_LENS_MM)
    parser.add_argument("--sensor-mm", type=float, default=DEFAULT_SENSOR_MM)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--agreement-tolerance", type=float, default=0.10)
    parser.add_argument("--follow-factor", type=float, default=2.0)
    parser.add_argument("--degenerate-aspect", type=float, default=20.0)
    parser.add_argument("--size-mismatch-factor", type=float, default=3.0)
    parser.add_argument("--peer-mismatch-factor", type=float, default=2.0)
    parser.add_argument("--minimum-peer-count", type=int, default=3)
    parser.add_argument("--tiny-observed-edge-m", type=float, default=0.30)
    parser.add_argument("--corpus-mismatch-factor", type=float, default=2.0)
    parser.add_argument("--minimum-corpus-count", type=int, default=3)
    parser.add_argument(
        "--asset-corpus",
        type=Path,
        help="cross-scene same-asset size corpus written by --emit-asset-corpus",
    )
    parser.add_argument(
        "--emit-asset-corpus",
        type=Path,
        help="build the corpus from --corpus-placement and exit",
    )
    parser.add_argument("--corpus-placement", type=Path, nargs="*", default=[])
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--out-report", type=Path)
    args = parser.parse_args()

    if args.emit_asset_corpus:
        corpus = build_corpus(list(args.corpus_placement))
        args.emit_asset_corpus.parent.mkdir(parents=True, exist_ok=True)
        args.emit_asset_corpus.write_text(
            json.dumps(corpus, indent=2,ensure_ascii=False) + "\n", encoding="utf-8"
        )
        multi = sum(
            1
            for entry in (corpus.get("by_asset_and_category") or {}).values()
            if entry["count"] >= 3
        )
        print(
            "Wrote corpus {}: {} assets, {} asset-and-category pairs, {} pairs with "
            "at least 3 instances".format(
                args.emit_asset_corpus.resolve(),
                len(corpus.get("by_asset") or {}),
                len(corpus.get("by_asset_and_category") or {}),
                multi,
            )
        )
        return 0

    if not args.scene or not args.placement or not args.out_report:
        parser.error("--scene, --placement and --out-report are required")

    asset_corpus = None
    if args.asset_corpus and args.asset_corpus.is_file():
        with args.asset_corpus.open("r", encoding="utf-8") as handle:
            asset_corpus = json.load(handle)

    with args.placement.open("r", encoding="utf-8") as handle:
        placement = json.load(handle)

    result = screen_scene(
        placement,
        lens_mm=args.lens_mm,
        sensor_mm=args.sensor_mm,
        resolution=args.resolution,
        agreement_tolerance=args.agreement_tolerance,
        follow_factor=args.follow_factor,
        degenerate_aspect=args.degenerate_aspect,
        size_mismatch_factor=args.size_mismatch_factor,
        peer_mismatch_factor=args.peer_mismatch_factor,
        minimum_peer_count=args.minimum_peer_count,
        tiny_observed_edge_m=args.tiny_observed_edge_m,
        top_k=args.top_k,
        asset_corpus=asset_corpus,
        corpus_mismatch_factor=args.corpus_mismatch_factor,
        minimum_corpus_count=args.minimum_corpus_count,
    )
    report = {
        "schema_version": "sceneproof_scaling_chain_attribution_v2",
        "scene": args.scene,
        "thresholds": {
            "agreement_tolerance": args.agreement_tolerance,
            "follow_factor": args.follow_factor,
            "degenerate_aspect": args.degenerate_aspect,
            "size_mismatch_factor": args.size_mismatch_factor,
            "peer_mismatch_factor": args.peer_mismatch_factor,
            "minimum_peer_count": args.minimum_peer_count,
            "corpus_mismatch_factor": args.corpus_mismatch_factor,
            "minimum_corpus_count": args.minimum_corpus_count,
            "tiny_observed_edge_m": args.tiny_observed_edge_m,
        },
        "camera": {
            "lens_mm": args.lens_mm,
            "sensor_mm": args.sensor_mm,
            "resolution": args.resolution,
        },
        **result,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.out_report.resolve()}")

    print(
        "CHAIN objects={} flagged={} camera={} asset_names_testable={}/{}".format(
            result["object_count"],
            result["flagged_count"],
            result["camera_available"],
            result["asset_name_testable_count"],
            result["object_count"],
        )
    )
    for item in result["worst_by_screen_area"]:
        print(
            "  {}: screen={:.2%} asset={} peers={}".format(
                item["object_id"],
                item["screen_area_fraction"] or 0.0,
                item["retrieved_asset"],
                item["same_asset_peer_count"],
            )
        )
        print(
            "      observed={} native={} rendered={} scale={}".format(
                format_box(item["pcd_obb_size_m"]),
                format_box(item["implied_asset_native_size_m"]),
                format_box(item["length_m"]),
                format_box(item["scale"]),
            )
        )
        print(
            "      set_by={} to_observation={}x(vol) {}x(longest) to_asset={}x".format(
                item["size_was_set_by"],
                format_factor(item["rendered_over_observed_volume_factor"]),
                format_factor(item["rendered_over_observed_longest_edge_ratio"]),
                format_factor(item["rendered_over_asset_factor"]),
            )
        )
        print(
            "      asset/observed={}x peer_median={}x corpus_median={}x(n={}) "
            "branch={}".format(
                format_factor(item["asset_over_observed_factor"]),
                format_factor(item["rendered_over_peer_median"]),
                format_factor(item["rendered_over_corpus_median"]),
                item["same_asset_corpus_count"],
                item["production_scale_branch"],
            )
        )
        print(f"      reasons={','.join(item['defect_reasons'])}")
    if result["reason_counts"]:
        print("  reasons (count, share of scene):")
        for name, count in result["reason_counts"].items():
            print(f"      {name}={count} ({result['reason_flag_rates'][name]:.0%})")
    weak = result["asset_name_weak_verdict_count_reported_not_counted"]
    print(
        "asset name does not name its category: {} objects ({:.0%}) - reported, "
        "NOT counted as a defect, precision not established".format(
            weak, weak / result["object_count"] if result["object_count"] else 0.0
        )
    )
    print("  size_was_set_by:")
    for name, count in result["stage_counts"].items():
        print(f"      {name}={count}")
    print("  mechanisms present (an object may carry several):")
    for name, count in result["mechanism_counts"].items():
        share = count / result["object_count"] if result["object_count"] else 0.0
        print(f"      {name}={count} ({share:.0%})")
    rollup = result["root_cause_rollup"]
    print(
        "  MECHANISM the asset determined the size against a disagreeing "
        "observation: {} objects ({:.0%}) covering {:.1%} of the frame".format(
            rollup["objects_whose_size_the_asset_determined"],
            rollup["share_of_scene"],
            rollup["screen_area_they_cover"],
        )
    )
    print(
        "      this is a mechanism count, not an error count: whether the asset "
        "deserved to win needs a reference this tool does not have"
    )
    if rollup["object_ids"]:
        print("      " + ", ".join(rollup["object_ids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
