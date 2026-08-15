"""Tests for the Fix97/98 scaling-chain attribution.

Every real-case number here is taken verbatim from the Smoke5 runs, so these tests
pin the four defects found by reading the first run's own output and would fail if
any of them returned.
"""

from __future__ import annotations

import unittest

import numpy as np

from sceneproof_scaling_chain_attribution_fix97 import (
    boxes_agree,
    category_tokens,
    collect_asset_sizes,
    corpus_key,
    equivalent_linear_factor,
    implied_native_size,
    informative_asset_runs,
    longest_edge_ratio,
    normalise_asset_name,
    observed_footprint_products,
    production_scale_branch,
    retrieval_name_check,
    scale_components_at_clamp_bound,
    scale_is_exact_abstention,
    screen_scene,
    size_disagreement,
    sorted_ratio,
    summarise_asset_corpus,
    symmetric_factor,
    which_reference_the_render_followed,
    worst_axis_factor,
)


def vector(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


class ImpliedNativeSizeTests(unittest.TestCase):
    def test_it_is_exact_division_because_dimensions_equal_native_times_scale(self):
        native = implied_native_size(vector(2.0, 1.0, 0.5), vector(2.0, 4.0, 0.5))
        self.assertIsNotNone(native)
        np.testing.assert_allclose(native, vector(1.0, 0.25, 1.0))

    def test_pillow_16_native_size_is_a_real_sofa(self):
        # Real per-axis values from the run: rendered 6.54, 2.47, 3.08 with scale
        # 2.37, 2.37, 3.60.  The first version of this test paired sorted edges by
        # hand and asserted 2.759, 1.300, 0.686; it passed while encoding a wrong
        # axis pairing, because it was fed the sorted vector rather than the real
        # one.  The conclusion, a 2.76 m sofa retrieved for a pillow, is unchanged.
        native = implied_native_size(
            vector(6.54, 2.47, 3.08), vector(2.37, 2.37, 3.60)
        )
        np.testing.assert_allclose(native, vector(2.759, 1.042, 0.856), atol=2e-3)
        self.assertGreater(float(np.max(native)), 2.5)

    def test_a_zero_scale_component_abstains_rather_than_dividing_by_zero(self):
        self.assertIsNone(
            implied_native_size(vector(1.0, 1.0, 1.0), vector(1.0, 0.0, 1.0))
        )

    def test_missing_input_abstains(self):
        self.assertIsNone(implied_native_size(None, vector(1.0, 1.0, 1.0)))
        self.assertIsNone(implied_native_size(vector(1.0, 1.0, 1.0), None))


class SizeFactorTests(unittest.TestCase):
    """The aggregator matters more than the threshold, and this pins which one."""

    def test_symmetric_factor_folds_both_directions_onto_one_scale(self):
        self.assertAlmostEqual(symmetric_factor(2.0), 2.0)
        self.assertAlmostEqual(symmetric_factor(0.5), 2.0)
        self.assertAlmostEqual(symmetric_factor(1.0), 1.0)
        self.assertIsNone(symmetric_factor(None))
        self.assertIsNone(symmetric_factor(0.0))

    def test_the_equivalent_linear_factor_is_the_cube_root_of_the_volume_ratio(self):
        factor = equivalent_linear_factor(
            vector(2.0, 2.0, 2.0), vector(1.0, 1.0, 1.0)
        )
        self.assertAlmostEqual(factor, 2.0)

    def test_a_thin_axis_alone_no_longer_decides_the_verdict(self):
        # curtain_0 in livingroom_10, at 7.86per cent of the frame.  Its two long
        # edges agree with depth to three decimal places; the entire disagreement is
        # 9 cm of curtain thickness against an observed 62 cm.  The worst-axis
        # measure called this 6.9; the equivalent linear factor calls it 1.90.
        rendered = vector(1.62, 0.09, 2.22)
        observed = vector(1.62, 0.62, 2.22)
        self.assertGreater(worst_axis_factor(sorted_ratio(rendered, observed)), 6.0)
        self.assertAlmostEqual(
            symmetric_factor(equivalent_linear_factor(rendered, observed)),
            1.902,
            places=2,
        )

    def test_the_genuine_bookshelf_defect_survives_the_new_aggregator(self):
        # bookshelf_0: a 1.91 m shelf rendered from a 7 cm thick observation.  A
        # measure restricted to the two longest edges would score this 2.86 and miss
        # it; the volume-equivalent factor scores it 3.39.
        rendered = vector(1.20, 0.40, 1.91)
        observed = vector(0.80, 0.42, 0.07)
        self.assertAlmostEqual(
            symmetric_factor(equivalent_linear_factor(rendered, observed)),
            3.390,
            places=2,
        )

    def test_the_worst_cases_stay_far_above_any_threshold(self):
        for rendered, observed, expected in (
            (vector(0.92, 2.00, 0.92), vector(0.11, 0.04, 0.03), 23.4),
            (vector(1.34, 0.83, 0.36), vector(0.07, 0.01, 0.01), 38.5),
        ):
            with self.subTest(expected=expected):
                self.assertAlmostEqual(
                    symmetric_factor(equivalent_linear_factor(rendered, observed)),
                    expected,
                    delta=0.2,
                )

    def test_a_corrected_case_stays_below_any_threshold(self):
        # trash_bin_0: a 1.72 m bin asset correctly brought to 0.47 m.
        self.assertAlmostEqual(
            symmetric_factor(
                equivalent_linear_factor(
                    vector(0.19, 0.24, 0.47), vector(0.32, 0.32, 0.48)
                )
            ),
            1.319,
            places=2,
        )

    def test_a_degenerate_box_abstains(self):
        self.assertIsNone(
            equivalent_linear_factor(vector(1.0, 1.0, 1.0), vector(1.0, 0.0, 1.0))
        )
        self.assertIsNone(equivalent_linear_factor(None, vector(1.0, 1.0, 1.0)))
        self.assertIsNone(
            longest_edge_ratio(vector(1.0, 1.0, 1.0), vector(1.0, 0.0, 1.0))
        )


class RodBlindSpotTests(unittest.TestCase):
    """The volume factor alone loses pen_0, a rod visible through the whole image."""

    PEN_RENDERED = (0.05, 0.05, 1.81)
    PEN_OBSERVED = (0.06, 0.44, 0.08)

    def test_the_volume_factor_alone_says_a_rod_agrees_with_a_blob(self):
        factor = symmetric_factor(
            equivalent_linear_factor(
                vector(*self.PEN_RENDERED), vector(*self.PEN_OBSERVED)
            )
        )
        self.assertAlmostEqual(factor, 1.289, places=2)
        self.assertLess(factor, 3.0)

    def test_the_longest_edge_recovers_it(self):
        self.assertAlmostEqual(
            longest_edge_ratio(
                vector(*self.PEN_RENDERED), vector(*self.PEN_OBSERVED)
            ),
            1.81 / 0.44,
            places=4,
        )

    def test_an_aspect_ratio_would_have_brought_the_thin_axis_artefact_back(self):
        # Why the longest edge and not an aspect ratio: curtain_0's rendered aspect is
        # 24.7 against an observed 3.58, a factor of 6.9, while its longest edges are
        # equal.  An aspect test would flag the curtain again.
        rendered = np.sort(vector(1.62, 0.09, 2.22))[::-1]
        observed = np.sort(vector(1.62, 0.62, 2.22))[::-1]
        rendered_aspect = rendered[0] / rendered[2]
        observed_aspect = observed[0] / observed[2]
        self.assertGreater(rendered_aspect / observed_aspect, 6.0)
        self.assertAlmostEqual(
            longest_edge_ratio(rendered, observed), 1.0, places=6
        )

    def test_the_union_of_both_views_is_correct_on_every_known_case(self):
        cases = (
            ("curtain_0", (1.62, 0.09, 2.22), (1.62, 0.62, 2.22), False),
            ("sign_2", (1.71, 0.31, 0.55), (1.71, 0.08, 0.55), False),
            ("ceiling_fan_0", (1.27, 1.27, 0.29), (0.27, 0.45, 0.39), False),
            ("trash_bin_0", (0.19, 0.24, 0.47), (0.32, 0.32, 0.48), False),
            ("gaming_table_2", (5.80, 4.19, 0.92), (5.79, 4.19, 0.92), False),
            ("picture_frame_2", (3.51, 0.14, 0.97), (4.38, 0.08, 0.97), False),
            ("pen_0", (0.05, 0.05, 1.81), (0.06, 0.44, 0.08), True),
            ("bookshelf_0", (1.20, 0.40, 1.91), (0.80, 0.42, 0.07), True),
            ("bookshelf_9", (0.93, 0.29, 1.95), (0.07, 0.13, 0.02), True),
            ("paper_cup_1", (0.92, 2.00, 0.92), (0.11, 0.04, 0.03), True),
            ("stack_of_chips_9", (1.34, 0.83, 0.36), (0.07, 0.01, 0.01), True),
            ("single_sofa_chair_2", (0.13, 0.13, 0.09), (4.27, 0.02, 0.13), True),
        )
        for name, rendered, observed, should_flag in cases:
            with self.subTest(name=name):
                views = size_disagreement(vector(*rendered), vector(*observed))
                flagged = views["worst"] > 3.0
                self.assertEqual(flagged, should_flag)


class ScaleAbstentionTests(unittest.TestCase):
    def test_exact_identity_is_an_abstention(self):
        self.assertTrue(scale_is_exact_abstention(vector(1.0, 1.0, 1.0)))

    def test_a_computed_factor_near_one_is_not_an_abstention(self):
        self.assertFalse(scale_is_exact_abstention(vector(1.0000001, 1.0, 1.0)))
        self.assertFalse(scale_is_exact_abstention(vector(0.99, 1.0, 1.0)))


class ClampBoundTests(unittest.TestCase):
    def test_the_upper_bound_is_detected_on_the_real_picture_frame_scale(self):
        self.assertEqual(
            scale_components_at_clamp_bound(vector(5.00, 4.03, 1.81)), [5.0]
        )

    def test_the_sculpture_hit_the_bound_on_all_three_axes(self):
        # sculpture_0: observed 3.00 m tall, asset natively 0.42 m, so the estimate
        # ran away and was capped at fivefold on every axis.
        self.assertEqual(
            scale_components_at_clamp_bound(vector(5.0, 5.0, 5.0)), [5.0, 5.0, 5.0]
        )

    def test_a_merely_large_factor_is_not_on_the_bound(self):
        self.assertEqual(scale_components_at_clamp_bound(vector(1.37, 1.37, 1.37)), [])
        self.assertEqual(scale_components_at_clamp_bound(vector(4.99, 0.11, 2.0)), [])


class ProductionBranchTests(unittest.TestCase):
    def test_the_predicate_matches_line_6396_pairwise_products(self):
        self.assertAlmostEqual(
            observed_footprint_products(vector(1.07, 0.25, 0.25)), 0.2675
        )

    def test_pillow_16_took_the_large_object_path_by_a_narrow_margin(self):
        self.assertEqual(
            production_scale_branch(vector(1.07, 0.25, 0.25)),
            "large_object_observed_box_ratio_path",
        )

    def test_paper_cup_1_took_the_pixel_path_that_ignores_the_observed_box(self):
        self.assertEqual(
            production_scale_branch(vector(0.11, 0.04, 0.03)),
            "small_object_pixel_bbox_path_ignores_the_observed_box",
        )

    def test_bookshelf_9_observed_a_seven_centimetre_fragment(self):
        self.assertEqual(
            production_scale_branch(vector(0.07, 0.13, 0.02)),
            "small_object_pixel_bbox_path_ignores_the_observed_box",
        )

    def test_a_degenerate_or_missing_box_abstains(self):
        self.assertIsNone(production_scale_branch(None))
        self.assertIsNone(production_scale_branch(vector(1.0, 0.0, 1.0)))


class BoxAgreementTests(unittest.TestCase):
    def test_axis_order_cannot_confound_the_comparison(self):
        self.assertTrue(
            boxes_agree(
                vector(1.65, 0.79, 0.73), vector(1.65, 0.73, 0.79), tolerance=0.10
            )
        )

    def test_gaming_table_2_rendered_size_reproduces_its_observed_box(self):
        self.assertTrue(
            boxes_agree(
                vector(5.80, 4.19, 0.92), vector(5.79, 4.19, 0.92), tolerance=0.10
            )
        )

    def test_sorted_ratio_is_descending_edgewise(self):
        np.testing.assert_allclose(
            sorted_ratio(vector(1.0, 4.0, 2.0), vector(2.0, 1.0, 0.5)),
            vector(2.0, 2.0, 2.0),
        )


class WhichReferenceTests(unittest.TestCase):
    """Real objects.  The taxonomy asks which reference the render followed."""

    def follow(self, observed, scale, length):
        stage, _ = which_reference_the_render_followed(
            observed=observed, scale=scale, length=length, follow_factor=2.0
        )
        return stage

    def test_gaming_table_2_followed_depth(self):
        self.assertEqual(
            self.follow(
                vector(5.79, 4.19, 0.92),
                vector(1.98, 2.56, 0.74),
                vector(5.80, 4.19, 0.92),
            ),
            "rendered_size_followed_the_observation",
        )

    def test_paper_cup_1_followed_the_asset(self):
        self.assertEqual(
            self.follow(
                vector(0.11, 0.04, 0.03),
                vector(1.0, 1.0, 1.0),
                vector(0.92, 2.00, 0.92),
            ),
            "rendered_size_followed_the_asset",
        )

    def test_ceiling_fan_0_followed_the_asset_and_the_asset_was_right(self):
        # observed 0.27 x 0.45 x 0.39 is the wrong reference here; the 1.27 m fan is
        # correct.  The taxonomy must name which reference won without claiming the
        # winner was wrong.
        self.assertEqual(
            self.follow(
                vector(0.27, 0.45, 0.39),
                vector(0.87, 0.87, 0.57),
                vector(1.27, 1.27, 0.29),
            ),
            "rendered_size_followed_the_asset",
        )

    def test_trash_bin_0_followed_depth_because_the_scale_corrected_the_asset(self):
        self.assertEqual(
            self.follow(
                vector(0.32, 0.32, 0.48),
                vector(0.31, 0.27, 0.27),
                vector(0.19, 0.24, 0.47),
            ),
            "rendered_size_followed_the_observation",
        )

    def test_pillow_16_followed_neither_because_the_scale_invented_a_third_size(self):
        self.assertEqual(
            self.follow(
                vector(1.07, 0.25, 0.25),
                vector(2.37, 2.37, 3.60),
                vector(6.54, 2.47, 3.08),
            ),
            "rendered_size_followed_neither",
        )

    def test_stack_of_chips_3_followed_the_garbage_carton_asset(self):
        self.assertEqual(
            self.follow(
                vector(0.02, 0.08, 0.01),
                vector(1.37, 1.37, 1.37),
                vector(1.34, 0.83, 0.36),
            ),
            "rendered_size_followed_the_asset",
        )

    def test_an_incomplete_chain_is_reported_as_undetermined(self):
        self.assertEqual(
            self.follow(None, vector(1.0, 1.0, 1.0), vector(1.0, 1.0, 1.0)),
            "undetermined_scaling_chain_incomplete",
        )

    def test_both_distances_are_reported_as_numbers(self):
        _, distances = which_reference_the_render_followed(
            observed=vector(1.0, 1.0, 1.0),
            scale=vector(2.0, 2.0, 2.0),
            length=vector(2.0, 2.0, 2.0),
            follow_factor=2.0,
        )
        self.assertAlmostEqual(distances["rendered_over_observed_factor"], 2.0)
        self.assertAlmostEqual(distances["rendered_over_asset_factor"], 2.0)


class AssetNameTests(unittest.TestCase):
    def test_generic_tokens_are_stripped(self):
        self.assertEqual(normalise_asset_name("0_SM_Sofa_2"), "sofa")
        self.assertEqual(normalise_asset_name("44_sk75_CasinoTable02"), "casinotable")

    def test_opaque_ids_normalise_to_almost_nothing(self):
        self.assertEqual(normalise_asset_name("b_114"), "b")
        self.assertEqual(normalise_asset_name("d_1000003759813"), "d")

    def test_short_category_tokens_are_not_searched_for(self):
        self.assertEqual(category_tokens("pen"), [])
        self.assertEqual(category_tokens("paper_cup"), ["paper"])
        self.assertEqual(
            category_tokens("multi_person_sofa"), ["multi", "person", "sofa"]
        )

    def test_informative_runs_drop_short_and_generic_pieces(self):
        self.assertEqual(informative_asset_runs("0_SM_Shelf_2"), ["shelf"])
        self.assertEqual(
            informative_asset_runs("0_ceiling_fan_2k_packed"), ["ceiling"]
        )


class RetrievalNameCheckTests(unittest.TestCase):
    LIVINGROOM = {"pillow", "multi_person_sofa", "single_sofa_chair", "floor_lamp"}
    STREET = {"paper_cup", "trash_bin", "trash_bag", "sign", "beverage_bottle"}

    def test_pillow_16_is_caught_naming_the_sofa_category(self):
        reasons, report = retrieval_name_check("pillow", "0_SM_Sofa_2", self.LIVINGROOM)
        self.assertEqual(
            reasons, ["retrieved_asset_names_a_different_category_in_this_scene"]
        )
        self.assertIn(
            "multi_person_sofa", report["asset_name_names_these_other_categories"]
        )

    def test_paper_cup_1_is_caught_as_naming_no_category_here(self):
        reasons, report = retrieval_name_check(
            "paper_cup", "4_SM_Ventilation1_7", self.STREET
        )
        self.assertEqual(reasons, ["retrieved_asset_name_does_not_name_its_category"])
        self.assertTrue(report["asset_name_is_testable"])

    def test_a_shelf_asset_for_a_bookshelf_is_no_longer_a_false_positive(self):
        # The first version matched in one direction only and flagged this, although
        # 'Shelf' is a correct asset for a bookshelf.
        reasons, report = retrieval_name_check(
            "bookshelf", "0_SM_Shelf_2", {"bookshelf"}
        )
        self.assertEqual(reasons, [])
        self.assertTrue(report["asset_name_is_testable"])

    def test_the_true_positives_survive_the_bidirectional_relaxation(self):
        for category, asset, scene in (
            ("stack_of_chips", "a_SM_CartonGarbage05", {"stack_of_chips"}),
            ("beverage_bottle", "a_SM_Vase_6", {"beverage_bottle"}),
            ("paper_cup", "4_SM_Ventilation1_7", {"paper_cup"}),
        ):
            with self.subTest(category=category):
                reasons, _ = retrieval_name_check(category, asset, scene)
                self.assertEqual(
                    reasons, ["retrieved_asset_name_does_not_name_its_category"]
                )

    def test_an_opaque_asset_id_abstains_instead_of_counting_as_agreement(self):
        reasons, report = retrieval_name_check("casino_chair", "b_114", {"casino_chair"})
        self.assertEqual(reasons, [])
        self.assertFalse(report["asset_name_is_testable"])

    def test_a_correct_retrieval_is_not_flagged(self):
        for category, asset in (
            ("office_desk", "a_SM_desk_compiled"),
            ("bookshelf", "a_SM_BookShelf_01a"),
            ("trash_bag", "a_SM_Trash_Separated_g2"),
            ("multi_person_sofa", "0_SM_Sofa_2"),
            ("wall_mounted_picture_frame", "a_SM_Frame04"),
            ("ceiling_fan", "0_ceiling_fan_2k_packed"),
        ):
            with self.subTest(category=category):
                reasons, _ = retrieval_name_check(category, asset, {category})
                self.assertEqual(reasons, [])

    def test_a_category_with_no_long_token_abstains_rather_than_guessing(self):
        reasons, report = retrieval_name_check("pen", "a_SM_Point_Lamp_4", {"pen"})
        self.assertEqual(reasons, [])
        self.assertFalse(report["asset_name_is_testable"])

    def test_a_missing_asset_abstains(self):
        reasons, report = retrieval_name_check("pillow", None, {"pillow"})
        self.assertEqual(reasons, [])
        self.assertIsNone(report["asset_name_normalised"])


def placement_with(objects: dict[str, dict]) -> dict:
    # Camera at (0, -4, 1) looking towards +Y with world +Z as up, so objects near
    # the origin are in front of it.  Columns of the rotation are the camera's local
    # x, y and z axes in world coordinates, and Blender looks along local -Z.
    camera = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -4.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    obj_info = {"scene_camera": {"pose_matrix_for_blender": camera}}
    obj_info.update(objects)
    return {"obj_info": obj_info}


def box_corners(centre, size):
    cx, cy, cz = centre
    sx, sy, sz = (value / 2.0 for value in size)
    return [
        [cx + dx * sx, cy + dy * sy, cz + dz * sz]
        for dx in (-1, 1)
        for dy in (-1, 1)
        for dz in (-1, 1)
    ]


def run(placement, **overrides):
    kwargs = dict(
        lens_mm=30.0,
        sensor_mm=36.0,
        resolution=1024,
        agreement_tolerance=0.10,
        follow_factor=2.0,
        degenerate_aspect=20.0,
        size_mismatch_factor=3.0,
        peer_mismatch_factor=2.0,
        minimum_peer_count=3,
        tiny_observed_edge_m=0.30,
        top_k=8,
    )
    kwargs.update(overrides)
    return screen_scene(placement, **kwargs)


class SameAssetPeerTests(unittest.TestCase):
    """The external reference the first version lacked, restored precisely."""

    def gaming_tables(self):
        sizes = {
            "gaming_table_0": ([2.51, 1.15, 0.88], [0.86, 0.71, 0.71]),
            "gaming_table_1": ([2.61, 1.31, 0.90], [0.89, 0.80, 0.73]),
            "gaming_table_2": ([5.80, 4.19, 0.92], [1.98, 2.56, 0.74]),
        }
        objects = {}
        for index, (name, (length, scale)) in enumerate(sizes.items()):
            objects[name] = {
                "retrieved_asset": "44_sk75_CasinoTable02",
                "length": length,
                "scale": scale,
                # Each table's observed box equals its rendered size, which is why
                # no internal-consistency test can see the defect.
                "pcd_obb_size": length,
                "bbox": box_corners((index * 3.0, 0.0,0.5), length),
            }
        return placement_with(objects)

    def test_the_oversized_table_is_flagged_against_its_same_asset_peers(self):
        result = run(self.gaming_tables())
        by_id = {item["object_id"]: item for item in result["all_findings"]}
        self.assertIn(
            "rendered_size_disagrees_with_its_same_asset_peers",
            by_id["gaming_table_2"]["defect_reasons"],
        )
        self.assertAlmostEqual(
            by_id["gaming_table_2"]["rendered_over_peer_median"], 5.80 / 2.61, places=3
        )

    def test_its_correctly_sized_peers_are_not_flagged(self):
        result = run(self.gaming_tables())
        by_id = {item["object_id"]: item for item in result["all_findings"]}
        self.assertEqual(by_id["gaming_table_0"]["defect_reasons"], [])
        self.assertEqual(by_id["gaming_table_1"]["defect_reasons"], [])

    def test_the_oversized_table_is_no_longer_absent_from_the_report(self):
        # The regression this test exists for: the largest object in the casino
        # frame carried no reason at all and dropped out of the ranked list.
        result = run(self.gaming_tables())
        self.assertIn(
            "gaming_table_2",
            {item["object_id"] for item in result["worst_by_screen_area"]},
        )

    def test_a_family_that_is_uniformly_wrong_cannot_be_caught_this_way(self):
        # All four livingroom bookshelves render at 1.91 m, so the peer test is
        # silent by construction.  Stated as a test so the limit is explicit.
        objects = {
            f"bookshelf_{index}": {
                "retrieved_asset": "b_33",
                "length": [1.20, 0.40, 1.91],
                "scale": [1.0, 1.0, 1.0],
                "pcd_obb_size": [0.16, 0.11, 0.04],
                "bbox": box_corners((index * 1.5, 0.0, 1.0), (1.20, 0.40, 1.91)),
            }
            for index in range(4)
        }
        result = run(placement_with(objects))
        for item in result["all_findings"]:
            self.assertNotIn(
                "rendered_size_disagrees_with_its_same_asset_peers",
                item["defect_reasons"],
            )

    def test_fewer_peers_than_the_minimum_abstains(self):
        objects = {
            f"chair_{index}": {
                "retrieved_asset": "b_9",
                "length": [1.0, 1.0, 1.0 + 5.0 * index],
                "scale": [1.0, 1.0, 1.0],
                "pcd_obb_size": [1.0, 1.0, 1.0 + 5.0 * index],
                "bbox": box_corners((index * 2.0, 0.0, 0.5), (1.0, 1.0, 1.0)),
            }
            for index in range(2)
        }
        result = run(placement_with(objects))
        for item in result["all_findings"]:
            self.assertIsNone(item["rendered_over_peer_median"])


class SizeReasonSurvivalTests(unittest.TestCase):
    """Size reasons must be about the rendered size, not about the asset."""

    def test_trash_bin_0_is_not_flagged_because_the_scale_corrected_it(self):
        placement = placement_with(
            {
                "trash_bin_0": {
                    "retrieved_asset": "14_SM_Trash_Bin_Medium_HalfOpen_02",
                    "length": [0.19, 0.24, 0.47],
                    "scale": [0.31, 0.27, 0.27],
                    "pcd_obb_size": [0.32, 0.32, 0.48],
                    "bbox": box_corners((0.0, 0.0, 0.24), (0.19, 0.24, 0.47)),
                }
            }
        )
        item = run(placement)["all_findings"][0]
        self.assertNotIn(
            "rendered_size_far_larger_than_what_depth_observed", item["defect_reasons"]
        )
        # The longest edges differ by 1.7407/ 0.48 = 3.63, which is the larger of the
        # two views, so this number exceeds the mechanism threshold.  The mechanism
        # still does not fire, because the render followed the observation rather than
        # the asset: the scale did its job.  That separation is the point.
        self.assertAlmostEqual(item["asset_over_observed_factor"], 3.62654, places=4)
        self.assertEqual(
            item["size_was_set_by"], "rendered_size_followed_the_observation"
        )
        self.assertFalse(
            item["size_determined_by_the_asset_against_a_disagreeing_observation"]
        )

    def test_paper_cup_1_is_still_flagged_because_the_error_reached_the_image(self):
        placement = placement_with(
            {
                "paper_cup_1": {
                    "retrieved_asset": "4_SM_Ventilation1_7",
                    "length": [0.92, 2.00, 0.92],
                    "scale": [1.0, 1.0, 1.0],
                    "pcd_obb_size": [0.11, 0.04, 0.03],
                    "bbox": box_corners((0.0, 0.0, 1.0), (0.92, 2.00, 0.92)),
                }
            }
        )
        item = run(placement)["all_findings"][0]
        self.assertIn(
            "rendered_size_far_larger_than_what_depth_observed", item["defect_reasons"]
        )
        self.assertIn(
            "size_determined_by_the_asset_against_a_disagreeing_observation",
            item["defect_reasons"],
        )


class RootCauseRollupTests(unittest.TestCase):
    def bookshelves(self):
        observed = {
            "bookshelf_0": [0.80, 0.42, 0.07],
            "bookshelf_6": [0.16, 0.11, 0.04],
            "bookshelf_9": [0.07, 0.13, 0.02],
            "bookshelf_12": [0.38, 0.37, 0.17],
        }
        objects = {}
        for index, (name, box) in enumerate(observed.items()):
            objects[name] = {
                "retrieved_asset": "b_33",
                "length": [1.20, 0.40, 1.91],
                "scale": [1.0, 1.0, 1.0],
                "pcd_obb_size": box,
                "bbox": box_corners((index * 1.5 - 2.0, 0.0, 1.0), (1.20, 0.40, 1.91)),
            }
        return placement_with(objects)

    def test_all_four_spurious_bookshelves_are_rolled_up_as_one_mechanism(self):
        # bookshelf_0 has a0.80 m longest observed edge and an aspect of 11.4, so
        # neither a tiny-box gate nor a degenerate-sliver gate reaches it, yet it
        # renders a 1.91 m shelf from a 7 cm-thick observation.  That is why the
        # rollup is defined on the asset winning against the observation alone.
        result = run(self.bookshelves())
        rollup = result["root_cause_rollup"]
        self.assertEqual(rollup["objects_whose_size_the_asset_determined"], 4)
        self.assertAlmostEqual(rollup["share_of_scene"], 1.0)
        self.assertGreater(rollup["screen_area_they_cover"], 0.0)
        self.assertTrue(rollup["this_counts_a_mechanism_not_an_error"])

    def test_the_mechanism_counts_record_the_abstention_and_the_pixel_branch(self):
        result = run(self.bookshelves())
        self.assertEqual(
            result["mechanism_counts"]["scale_is_exactly_one_an_abstention"], 4
        )
        # Only three of the four are on the pixel path: bookshelf_0's observed
        # footprint is 0.80 x 0.42 = 0.336, above the 0.25 production threshold.
        self.assertEqual(
            result["mechanism_counts"][
                "small_object_pixel_bbox_path_ignores_the_observed_box"
            ],
            3,
        )

    def test_an_object_whose_render_followed_depth_is_not_in_the_rollup(self):
        placement = placement_with(
            {
                "gaming_table_2": {
                    "retrieved_asset": "44_sk75_CasinoTable02",
                    "length": [5.80, 4.19, 0.92],
                    "scale": [1.98, 2.56, 0.74],
                    "pcd_obb_size": [5.79, 4.19, 0.92],
                    "bbox": box_corners((0.0, 0.0, 0.5), (5.80, 4.19, 0.92)),
                }
            }
        )
        result = run(placement)
        self.assertEqual(
            result["root_cause_rollup"]["objects_whose_size_the_asset_determined"], 0
        )

    def test_the_mechanism_is_counted_even_when_it_produced_the_right_answer(self):
        # ceiling_fan_0: the 1.27 m fan is correct and the 0.45 m observation is
        # wrong, and the asset is what determined the size, so the mechanism is
        # present.  That is precisely why the rollup is a mechanism count.
        #
        # An earlier version asserted the opposite, that the fan drops out.  It did,
        # but only because that version measured size by volume alone, and the same
        # measure silently lost pen_0, a 1.81 m rod standing through the whole
        # bedroom image.  Their longest edges differ by 1.46/ 0.45 = 3.24.
        placement = placement_with(
            {
                "ceiling_fan_0": {
                    "retrieved_asset": "0_ceiling_fan_2k_packed",
                    "length": [1.27, 1.27, 0.29],
                    "scale": [0.87, 0.87, 0.57],
                    "pcd_obb_size": [0.27, 0.45, 0.39],
                    "bbox": box_corners((0.0, 0.0, 2.4), (1.27, 1.27, 0.29)),
                }
            }
        )
        result = run(placement)
        item = result["all_findings"][0]
        self.assertAlmostEqual(item["asset_over_observed_factor"], 3.24393, places=4)
        self.assertEqual(item["size_was_set_by"], "rendered_size_followed_the_asset")
        self.assertEqual(
            result["root_cause_rollup"]["objects_whose_size_the_asset_determined"], 1
        )

    def test_pen_0_is_back_in_the_mechanism_list(self):
        # The regression this test exists for: a 1.81 m lamp asset retrieved for a
        # pen, which is the vertical rod in the bedroom render.  Volume alone scores
        # it 1.29 and drops it without a trace.
        placement = placement_with(
            {
                "pen_0": {
                    "retrieved_asset": "a_SM_Point_Lamp_4",
                    "length": [0.05, 0.05, 1.81],
                    "scale": [1.0, 1.0, 1.0],
                    "pcd_obb_size": [0.06, 0.44, 0.08],
                    "bbox": box_corners((0.0, 0.0, 0.9), (0.05, 0.05, 1.81)),
                }
            }
        )
        result = run(placement)
        item = result["all_findings"][0]
        self.assertIn(
            "rendered_size_far_larger_than_what_depth_observed", item["defect_reasons"]
        )
        self.assertIn(
            "size_determined_by_the_asset_against_a_disagreeing_observation",
            item["defect_reasons"],
        )
        self.assertEqual(item["size_was_set_by"], "rendered_size_followed_the_asset")
        self.assertEqual(
            result["root_cause_rollup"]["objects_whose_size_the_asset_determined"], 1
        )

    def test_the_rollup_remains_a_mechanism_count_not_an_error_count(self):
        # The disclaimer is not conditional on any particular object dropping out:
        # nothing in this tool can decide whether the asset deserved to win.
        result = run(self.bookshelves())
        self.assertTrue(
            result["root_cause_rollup"]["this_counts_a_mechanism_not_an_error"]
        )


class CrossSceneCorpusTests(unittest.TestCase):
    """The only reference that can see a family which is uniformly wrong in a scene."""

    def frames_at(self, rendered_edge: float) -> dict:
        return placement_with(
            {
                f"wall_mounted_picture_frame_{index}": {
                    "retrieved_asset": "a_SM_Frame04",
                    "length": [rendered_edge, 0.14, 0.97],
                    "scale": [rendered_edge / 0.70, 4.03, 1.81],
                    "pcd_obb_size": [4.38, 0.08, 0.97],
                    "bbox": box_corners(
                        (index * 1.2 - 1.2, 0.0, 1.5), (rendered_edge, 0.14, 0.97)
                    ),
                }
                for index in range(3)
            }
        )

    def test_collect_asset_sizes_skips_structure_and_the_camera(self):
        sizes = collect_asset_sizes(
            placement_with(
                {
                    "floor_0": {"retrieved_asset": "a_floor", "length": [8.0, 8.0, 0.1]},
                    "chair_0": {"retrieved_asset": "b_9", "length": [0.5, 0.5, 1.0]},
                }
            )
        )
        self.assertEqual(set(sizes["by_asset"]), {"b_9"})
        self.assertEqual(sizes["by_asset"]["b_9"], [1.0])
        self.assertEqual(
            sizes["by_asset_and_category"][corpus_key("b_9", "chair")], [1.0]
        )

    def test_the_corpus_summarises_each_asset_across_scenes(self):
        corpus = summarise_asset_corpus(
            [
                {
                    "by_asset": {"a_SM_Frame04": [0.70, 0.72]},
                    "by_asset_and_category": {
                        corpus_key("a_SM_Frame04", "wall_mounted_picture_frame"): [
                            0.70,
                            0.72,
                        ]
                    },
                },
                {
                    "by_asset": {"a_SM_Frame04": [0.68, 3.51]},
                    "by_asset_and_category": {
                        corpus_key("a_SM_Frame04", "wall_mounted_picture_frame"): [
                            0.68,
                            3.51,
                        ]
                    },
                },
            ]
        )
        entry = corpus["by_asset"]["a_SM_Frame04"]
        self.assertEqual(entry["count"], 4)
        self.assertAlmostEqual(entry["median_max_edge_m"], 0.71)
        self.assertAlmostEqual(entry["max_max_edge_m"], 3.51)
        pair = corpus["by_asset_and_category"][
            corpus_key("a_SM_Frame04", "wall_mounted_picture_frame")
        ]
        self.assertEqual(pair["count"], 4)

    def frame_corpus(self, sizes):
        return summarise_asset_corpus(
            [
                {
                    "by_asset": {"a_SM_Frame04": list(sizes)},
                    "by_asset_and_category": {
                        corpus_key("a_SM_Frame04", "wall_mounted_picture_frame"): list(
                            sizes
                        )
                    },
                }
            ]
        )

    def test_a_uniformly_wrong_family_is_caught_against_the_other_scenes(self):
        # All three bedroom frames render at 3.51 m with an in-scene peer median of
        # exactly 1.00, so no local test can see a 3.5 m picture frame.Across the
        # corpus the same asset renders near 0.70 m.
        corpus = self.frame_corpus([0.70, 0.71, 0.69, 0.72])
        result = run(self.frames_at(3.51), asset_corpus=corpus)
        for item in result["all_findings"]:
            self.assertIn(
                "rendered_size_disagrees_with_the_same_asset_across_scenes",
                item["defect_reasons"],
            )
            self.assertAlmostEqual(
                item["rendered_over_corpus_median"], 3.51 / 0.705, places=3
            )
            self.assertAlmostEqual(item["rendered_over_peer_median"], 1.0)

    def test_a_family_consistent_with_the_corpus_is_not_flagged_by_it(self):
        corpus = self.frame_corpus([0.70, 0.71, 0.69, 0.72])
        result = run(self.frames_at(0.70), asset_corpus=corpus)
        for item in result["all_findings"]:
            self.assertNotIn(
                "rendered_size_disagrees_with_the_same_asset_across_scenes",
                item["defect_reasons"],
            )

    def test_too_few_corpus_samples_abstains(self):
        corpus = self.frame_corpus([0.70, 0.71])
        result = run(self.frames_at(3.51), asset_corpus=corpus)
        for item in result["all_findings"]:
            self.assertIsNone(item["rendered_over_corpus_median"])
            self.assertNotIn(
                "rendered_size_disagrees_with_the_same_asset_across_scenes",
                item["defect_reasons"],
            )

    def test_one_asset_reused_for_two_categories_is_not_pooled(self):
        # a_SM_CartonGarbage05 serves stack_of_chips in the casino at 1.34 m and
        # discarded_wooden_board in the street scene at 0.36 m.  Pooled by asset the
        # median is dominated by the casino's oversized chips and accuses the street
        # scene's boards; conditioned on the category each population stands alone.
        corpus = summarise_asset_corpus(
            [
                {
                    "by_asset": {"a_SM_CartonGarbage05": [1.34] * 13 + [0.36] * 4},
                    "by_asset_and_category": {
                        corpus_key("a_SM_CartonGarbage05", "stack_of_chips"): [1.34]
                        * 13,
                        corpus_key(
                            "a_SM_CartonGarbage05", "discarded_wooden_board"
                        ): [0.36]
                        * 4,
                    },
                }
            ]
        )
        placement = placement_with(
            {
                "discarded_wooden_board_6": {
                    "retrieved_asset": "a_SM_CartonGarbage05",
                    "length": [0.36, 0.08, 0.03],
                    "scale": [0.37, 0.12, 0.12],
                    "pcd_obb_size": [0.44, 0.15, 0.03],
                    "bbox": box_corners((0.0, 0.0, 0.02), (0.36, 0.08, 0.03)),
                }
            }
        )
        item = run(placement, asset_corpus=corpus)["all_findings"][0]
        self.assertAlmostEqual(item["rendered_over_corpus_median"], 1.0, places=6)
        self.assertNotIn(
            "rendered_size_disagrees_with_the_same_asset_across_scenes",
            item["defect_reasons"],
        )
        # The pooled figure is still reported, so the contamination stays visible.
        self.assertAlmostEqual(
            item["rendered_over_asset_only_corpus_median"], 0.36 / 1.34, places=3
        )

    def test_without_a_corpus_the_tool_behaves_as_before(self):
        result = run(self.frames_at(3.51))
        for item in result["all_findings"]:
            self.assertIsNone(item["rendered_over_corpus_median"])
            self.assertEqual(item["same_asset_corpus_count"], 0)


class WeakNameVerdictTests(unittest.TestCase):
    def test_it_is_reported_but_never_counted_as_a_defect(self):
        # At 15 to 48 per cent of objects, with known false positives from plurals
        # and from uninformative names, its precision is not established.
        placement = placement_with(
            {
                "beverage_bottle_0": {
                    "retrieved_asset": "a_SM_Vase_6",
                    "length": [0.12, 0.12, 0.34],
                    "scale": [1.0, 1.0, 1.0],
                    "pcd_obb_size": [0.12, 0.12, 0.34],
                    "bbox": box_corners((0.0, 0.0, 0.17), (0.12, 0.12, 0.34)),
                }
            }
        )
        result = run(placement)
        item = result["all_findings"][0]
        self.assertTrue(item["asset_name_does_not_name_its_category"])
        self.assertNotIn(
            "retrieved_asset_name_does_not_name_its_category", item["defect_reasons"]
        )
        self.assertEqual(
            result["asset_name_weak_verdict_count_reported_not_counted"], 1
        )

    def test_the_different_category_variant_is_still_counted(self):
        result = run(
            placement_with(
                {
                    "pillow_16": {
                        "retrieved_asset": "0_SM_Sofa_2",
                        "length": [6.54, 2.47, 3.08],
                        "scale": [2.37, 2.37, 3.60],
                        "pcd_obb_size": [1.07, 0.25, 0.25],
                        "bbox": box_corners((0.0, 0.0, 1.2), (6.54, 2.47, 3.08)),
                    },
                    "multi_person_sofa_1": {
                        "retrieved_asset": "0_SM_Sofa_2",
                        "length": [3.04, 1.11, 0.85],
                        "scale": [1.10, 1.07, 0.99],
                        "pcd_obb_size": [1.11, 3.04, 0.85],
                        "bbox": box_corners((1.0, 0.0, 0.4), (3.04, 1.11, 0.85)),
                    },
                }
            )
        )
        pillow = next(
            item for item in result["all_findings"] if item["object_id"] == "pillow_16"
        )
        self.assertIn(
            "retrieved_asset_names_a_different_category_in_this_scene",
            pillow["defect_reasons"],
        )


class ScreenSceneTests(unittest.TestCase):
    def build(self):
        return placement_with(
            {
                "floor_0": {"length": [8.0, 8.0, 0.1], "scale": [1.0, 1.0, 1.0]},
                "pillow_16": {
                    "retrieved_asset": "0_SM_Sofa_2",
                    "group": "sofa_group",
                    "length": [6.54, 2.47, 3.08],
                    "scale": [2.37, 2.37, 3.60],
                    "pcd_obb_size": [1.07, 0.25, 0.25],
                    "bbox": box_corners((0.0, 0.0, 1.2), (6.54, 2.47, 3.08)),
                },
                "multi_person_sofa_1": {
                    "retrieved_asset": "0_SM_Sofa_2",
                    "length": [3.04, 1.11, 0.85],
                    "scale": [1.10, 1.07, 0.99],
                    "pcd_obb_size": [1.11, 3.04, 0.85],
                    "bbox": box_corners((1.0, 0.0, 0.4), (3.04, 1.11, 0.85)),
                },
            }
        )

    def test_the_floor_is_excluded_as_a_structural_element(self):
        result = run(self.build())
        self.assertEqual(result["object_count"], 2)
        self.assertNotIn(
            "floor_0", {item["object_id"] for item in result["all_findings"]}
        )

    def test_the_camera_is_found_and_the_worst_offender_ranks_first(self):
        result = run(self.build())
        self.assertTrue(result["camera_available"])
        self.assertEqual(result["worst_by_screen_area"][0]["object_id"], "pillow_16")

    def test_the_correctly_retrieved_sofa_is_not_flagged(self):
        result = run(self.build())
        sofa = next(
            item
            for item in result["all_findings"]
            if item["object_id"] == "multi_person_sofa_1"
        )
        self.assertEqual(sofa["defect_reasons"], [])
        self.assertEqual(
            sofa["size_was_set_by"], "rendered_size_followed_the_observation"
        )

    def test_group_membership_is_an_annotation_not_a_stage(self):
        # The first version made this a stage, which put 82 per cent of the casino
        # into it and hid the actionable statement.
        result = run(self.build())
        pillow = next(
            item for item in result["all_findings"] if item["object_id"] == "pillow_16"
        )
        self.assertFalse(pillow["scale_shared_with_its_group"])
        self.assertEqual(pillow["size_was_set_by"], "rendered_size_followed_neither")
        self.assertNotIn(
            "scale_overwritten_by_group_consistency", set(result["stage_counts"])
        )

    def test_flag_rates_are_reported_so_a_useless_detector_is_visible(self):
        result = run(self.build())
        self.assertEqual(set(result["reason_counts"]), set(result["reason_flag_rates"]))
        for name, count in result["reason_counts"].items():
            self.assertAlmostEqual(
                result["reason_flag_rates"][name], count / result["object_count"]
            )

    def test_opaque_asset_names_are_counted_separately_from_testable_ones(self):
        placement = placement_with(
            {
                "casino_chair_4": {
                    "retrieved_asset": "b_114",
                    "length": [1.01, 0.68, 0.64],
                    "scale": [1.65, 1.65, 1.65],
                    "pcd_obb_size": [0.60, 0.60, 1.01],
                    "bbox": box_corners((0.0, 0.0, 0.3), (1.01, 0.68, 0.64)),
                }
            }
        )
        result = run(placement)
        self.assertEqual(result["asset_name_testable_count"], 0)
        self.assertEqual(result["asset_name_opaque_count"], 1)

    def test_a_degenerate_observed_box_is_flagged(self):
        placement = placement_with(
            {
                "wall_mounted_picture_frame_2": {
                    "retrieved_asset": "a_SM_Frame04",
                    "length": [3.51, 0.14, 0.97],
                    "scale": [5.00, 4.03, 1.81],
                    "pcd_obb_size": [4.38, 0.08, 0.97],
                    "bbox": box_corners((0.0, 0.0, 1.5), (3.51, 0.14, 0.97)),
                }
            }
        )
        item = run(placement)["all_findings"][0]
        self.assertIn("observed_box_is_a_degenerate_sliver", item["defect_reasons"])
        self.assertEqual(item["scale_components_on_clamp_bound"], [5.0])

    def test_the_withdrawn_detectors_are_declared_and_absent(self):
        result = run(self.build())
        self.assertTrue(
            result["policy"]["size_reasons_are_stated_about_the_rendered_size_not_the_asset"]
        )
        self.assertTrue(
            result["policy"][
                "asset_over_observed_is_a_disagreement_not_an_attribution_of_blame"
            ]
        )
        for withdrawn in (
            "scale_component_looks_clamped",
            "retrieved_asset_is_far_larger_than_what_depth_observed",
        ):
            self.assertNotIn(withdrawn, set(result["reason_counts"]))


if __name__ == "__main__":
    unittest.main()
