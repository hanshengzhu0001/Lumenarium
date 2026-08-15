"""Tests for the Fix101 asset-library join.

The library values here are the real ones read from
``asset_data/imaginarium_asset_info.csv``, so these tests pin both the verification of
the ``length / scale`` identity and the two false verdicts that the library exposed in
the name heuristics of Fix97 to Fix100.
"""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sceneproof_asset_library_join_fix101 import (
    asset_name_relation,
    asset_name_tokens,
    audit_scene,
    classify_retrieval,
    edges_agree_within_quantisation,
    fallback_scale_signature,
    identity_mismatch_shape,
    label_relation,
    library_vocabulary,
    load_asset_library,
    parse_bbx,
    singular,
    tokens_relate,
)


class QuantisationTests(unittest.TestCase):
    """bbx is recorded to 3 dp, so a flat relative tolerance is not defensible."""

    def test_the_mouse_false_failure_is_inside_the_rounding_noise(self):
        # mouse_1 printed identical computed and authored boxes and was failed at
        # 1.127: authored 0.004 stands for [0.0035, 0.0045], plus or minus 12.5%.
        inside, worst = edges_agree_within_quantisation(
            np.array([0.013, 0.041, 0.00355]),
            np.array([0.013, 0.041, 0.004]),
            quantum=0.0005,
            tolerance=0.02,
        )
        self.assertTrue(inside)
        self.assertGreater(worst, 1.10)

    def test_the_casino_table_still_fails(self):
        inside, worst = edges_agree_within_quantisation(
            np.array([2.922, 1.634, 1.234]),
            np.array([2.922, 1.634, 0.924]),
            quantum=0.0005,
            tolerance=0.02,
        )
        self.assertFalse(inside)
        self.assertAlmostEqual(worst, 1.335, delta=0.01)

    def test_a_three_percent_disagreement_on_a_metre_scale_edge_still_fails(self):
        # desk_0: 0.819 against an authored 0.844, where rounding is only 0.06%.
        inside, _ = edges_agree_within_quantisation(
            np.array([1.449, 0.749, 0.819]),
            np.array([1.446, 0.741, 0.844]),
            quantum=0.0005,
            tolerance=0.02,
        )
        self.assertFalse(inside)

    def test_the_curtain_and_sculpture_failures_are_genuine(self):
        for computed, authored in (
            ([1.447, 0.082, 1.919], [1.891, 0.273, 2.540]),
            ([0.215, 0.334, 0.419], [0.385, 0.032, 0.502]),
        ):
            with self.subTest(computed=computed):
                inside, _ = edges_agree_within_quantisation(
                    np.array(computed),
                    np.array(authored),
                    quantum=0.0005,
                    tolerance=0.02,
                )
                self.assertFalse(inside)


class LabelRelationTests(unittest.TestCase):
    """A finer or coarser label for one object is not a retrieval error."""

    def test_granularity_differences_are_not_substitutions(self):
        for category, asset_class in (
            ("pen_holder", "Desktop_pen_holder"),
            ("computer_keyboard", "Keyboard"),
            ("office_desk", "Desk"),
            ("bookshelf", "Tall_bookshelf"),
            ("bowl", "Small_bowl"),
            ("cup", "Water_cup"),
            ("file_folder", "Folder"),
            ("radio", "Portable_radio"),
            ("casino_chair", "Backrest_chair"),
        ):
            with self.subTest(category=category):
                self.assertNotEqual(
                    label_relation(category, asset_class), "shares_nothing"
                )

    def test_real_substitutions_share_nothing(self):
        for category, asset_class in (
            ("pen", "Chandelier"),
            ("pillow", "Snack"),
            ("pillow", "Multi_person_sofa"),
            ("fruit", "Speaker"),
            ("stack_of_chips", "Cigarette_box"),
            ("mouse", "Clip"),
            ("desktop_mouse_pad", "Sign"),
            ("paper_cup", "Discarded_industrial_component"),
            ("clock", "Speaker"),
        ):
            with self.subTest(category=category):
                self.assertEqual(
                    label_relation(category, asset_class), "shares_nothing"
                )

    def test_an_identical_label_is_named_as_such(self):
        self.assertEqual(label_relation("pillow", "Pillow"), "identical_label")

    def test_the_known_imperfection_is_pinned_not_hidden(self):
        # Declared in the docstring: a shared container word counts as one family.
        self.assertEqual(
            label_relation("stack_of_chips", "Stack_of_poker_cards"),
            "shares_a_token",
        )

    def test_a_closed_compound_is_not_a_substitution(self):
        # Fix102 called a teacup retrieving a water cup a substitution, because English
        # writes teacup as one word and the two labels share no underscored token.
        self.assertEqual(label_relation("teacup", "Water_cup"), "shares_the_head_noun")
        self.assertEqual(
            label_relation("bookshelf", "Corner_shelf"), "shares_the_head_noun"
        )

    def test_the_curated_bucket_alone_still_fails_on_a_wardrobe(self):
        # Unchanged and deliberately so: at the label level a wardrobe and a storage
        # locker share nothing.  What resolves this case is the asset name, not the
        # label, and that is asserted in AssetNameWitnessTests.
        self.assertEqual(label_relation("wardrobe", "Storage_locker"), "shares_nothing")


class TokenTests(unittest.TestCase):
    def test_regular_plurals_fold_and_irregular_ones_are_left_alone(self):
        self.assertEqual(singular("papers"), "paper")
        self.assertEqual(singular("signs"), "sign")
        self.assertEqual(singular("glass"), "glass")
        self.assertEqual(singular("shelves"), "shelve")

    def test_a_suffix_counts_as_a_head_only_with_a_real_prefix_left_over(self):
        for shorter, longer in (
            ("cup", "teacup"),
            ("shelf", "bookshelf"),
            ("cycle", "tricycle"),
            ("board", "billboard"),
        ):
            with self.subTest(longer=longer):
                self.assertTrue(tokens_relate(shorter, longer))

    def test_the_open_pen_coincidence_is_rejected(self):
        # The whole reason the prefix must be three characters:'open' ends with 'pen'
        # and a pen retrieving an open shelf must not be excused by that.
        self.assertFalse(tokens_relate("pen", "open"))
        self.assertFalse(tokens_relate("pad", "sign"))

    def test_identifiers_are_split_on_both_snake_and_camel_case(self):
        self.assertEqual(asset_name_tokens("a_SM_Wardrobe_01"), ["wardrobe"])
        self.assertEqual(asset_name_tokens("44_sk82_KidCycle01"), ["kid", "cycle"])
        self.assertEqual(asset_name_tokens("a_Signs22"), ["sign"])
        self.assertEqual(asset_name_tokens("a_SM_papers_pages_04"), ["paper", "page"])
        self.assertEqual(asset_name_tokens("44_sk25_PCCase03"), ["case"])

    def test_an_opaque_identifier_carries_no_words(self):
        for name in ("b_33", "d_1000003614815", "21_SM_PC_01ae"):
            with self.subTest(name=name):
                self.assertEqual(asset_name_tokens(name), [])
                self.assertEqual(
                    asset_name_relation("bookshelf", name), "asset_name_is_opaque"
                )

    def test_the_identifier_corroborates_where_the_bucket_did_not(self):
        for category, asset in (
            ("wardrobe", "a_SM_Wardrobe_01"),
            ("sign", "30_Sign_2_2"),
            ("bookshelf", "0_SM_Shelf_2"),
            ("children_tricycle", "44_sk82_KidCycle01"),
            ("paper", "a_SM_papers_pages_04"),
            ("fruit", "a_SM_KitchenFruit_Tomato01"),
            ("display_cabinet", "44_sk42_WineCabinet01"),
        ):
            with self.subTest(category=category):
                self.assertIn(
                    asset_name_relation(category, asset),
                    {"asset_name_carries_the_head_noun", "asset_name_carries_a_token"},
                )

    def test_the_identifier_does_not_excuse_the_catastrophic_substitutions(self):
        for category, asset in (
            ("pillow", "0_SM_Sofa_2"),
            ("pen", "a_SM_Point_Lamp_4"),
            ("stack_of_chips", "11_SM_Cigarette_05"),
            ("mouse", "21_SM_Stationery_NN_01ae"),
            ("bookshelf", "a_SM_locker_locker_main"),
            ("bookshelf", "a_SM_FilingCabinet01f"),
            ("trash_bin", "a_SM_kitchen_Canisters01_C"),
            ("bookshelf", "0_steel_frame_shelves_03"),
        ):
            with self.subTest(category=category):
                self.assertEqual(
                    asset_name_relation(category, asset), "asset_name_shares_nothing"
                )


class IdentityShapeTests(unittest.TestCase):
    """One scalar on three axes and one wrong axis need different fixes."""

    def shape(self, computed, authored):
        return identity_mismatch_shape(
            np.array(computed), np.array(authored), quantum=0.0005, tolerance=0.02
        )[0]

    def test_one_scalar_on_all_three_axes_is_named_as_such(self):
        # A curator cannot author all three axes wrong by one factor; a transform can.
        self.assertEqual(
            self.shape([0.216, 0.115, 0.270], [0.600, 0.320, 0.751]),
            "uniform_scalar_offset",
        )
        self.assertEqual(
            self.shape([0.094, 0.094, 0.115], [0.217, 0.217, 0.265]),
            "uniform_scalar_offset",
        )

    def test_uniformity_survives_the_rounding_of_a_thin_edge(self):
        # file_folder_3 is 0.008 m thick against an authored 0.013, where three-decimal
        # rounding alone is four per cent.  A fixed spread threshold calls this general.
        self.assertEqual(
            self.shape([0.108, 0.153, 0.008], [0.184, 0.260, 0.013]),
            "uniform_scalar_offset",
        )

    def test_one_wrong_axis_is_separated_from_a_transform(self):
        for computed, authored in (
            ([2.922, 1.634, 1.234], [2.922, 1.634, 0.924]),
            ([2.444, 0.564, 0.664], [2.444, 0.571, 0.814]),
            ([0.333, 0.333, 0.646], [0.333, 0.333, 0.710]),
            ([1.449, 0.749, 0.819], [1.446, 0.741, 0.844]),
        ):
            with self.subTest(computed=computed):
                self.assertEqual(self.shape(computed, authored), "one_axis_only")

    def test_a_disagreement_on_every_axis_by_different_factors_is_general(self):
        self.assertEqual(
            self.shape([1.447, 0.082, 1.919], [1.891, 0.273, 2.540]), "general"
        )


class FallbackSignatureTests(unittest.TestCase):
    """The unscaled small-object path, measured as an upper bound and named as one."""

    def test_an_exactly_unit_scale_on_a_small_footprint_is_the_signature(self):
        self.assertEqual(
            fallback_scale_signature(
                np.array([1.0, 1.0, 1.0]), np.array([0.17, 0.05, 0.12])
            ),
            "unscaled_on_the_small_object_path",
        )

    def test_an_estimated_scale_is_not_the_signature(self):
        self.assertEqual(
            fallback_scale_signature(
                np.array([1.31, 1.31, 2.28]), np.array([0.17, 0.05, 0.12])
            ),
            "scale_was_estimated",
        )

    def test_a_unit_scale_on_a_large_footprint_took_the_other_branch(self):
        self.assertEqual(
            fallback_scale_signature(
                np.array([1.0, 1.0, 1.0]), np.array([1.20, 0.40, 1.91])
            ),
            "unscaled_on_the_large_object_path",
        )

    def test_a_missing_observed_box_is_reported_not_guessed(self):
        self.assertEqual(
            fallback_scale_signature(np.array([1.0, 1.0, 1.0]), None),
            "unscaled_with_no_observed_box",
        )
        self.assertEqual(
            fallback_scale_signature(None, np.array([0.1, 0.1, 0.1])),
            "no_scale_recorded",
        )

# Verbatim from the library.
LIBRARY = {
    "0_SM_Sofa_2":("2.762,1.042,0.856", "Multi_person_sofa", "Stool_chair_or_sofa"),
    "a_SM_Point_Lamp_4": ("0.050,0.050,1.814", "Chandelier", "Lighting_equipment"),
    "4_SM_Ventilation1_7": (
        "0.922,1.999,0.922",
        "Discarded_industrial_component",
        "Industrial_component",
    ),
    "a_SM_CartonGarbage05": (
        "0.975,0.607,0.259",
        "Stacked_torn_cardboard_piece",
        "Cardboard",
    ),
    "a_SM_Decor_6": ("0.364,0.286,0.360", "Small_potted_plant", "Plant"),
    "d_1000003759813": ("0.633,0.210,0.338", "Pillow", "Bedding"),
    "44_sk75_CasinoTable02": ("2.922,1.634,0.924", "Gaming_table", "Gaming_table"),
    "b_33": ("1.200,0.403,1.910", "Storage_locker", "Storage_locker_rack"),
    "a_SM_Wardrobe_01": ("0.998,0.544,2.000", "Storage_locker", "Storage_locker_rack"),
}


def write_library(rows) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".csv", delete=False, encoding="utf-8-sig", newline=""
    )
    writer = csv.writer(handle)
    writer.writerow(
        ["id", "name_en", "bbx", "class_en", "retrieval_class_en", "scaling_strategy"]
    )
    for index, (name, (bbx, class_en, retrieval_class)) in enumerate(rows.items()):
        writer.writerow([index, name, bbx, class_en, retrieval_class, "ALIGNED_ANISOTROPIC"])
    handle.close()
    return Path(handle.name)


class ParseBbxTests(unittest.TestCase):
    def test_three_metre_values_are_parsed(self):
        np.testing.assert_allclose(
            parse_bbx("2.762,1.042,0.856"), np.array([2.762, 1.042, 0.856])
        )

    def test_whitespace_is_tolerated(self):
        np.testing.assert_allclose(
            parse_bbx(" 1.0 , 2.0 , 3.0 "), np.array([1.0, 2.0, 3.0])
        )

    def test_malformed_or_degenerate_values_abstain(self):
        for raw in ("", None, "1.0,2.0", "1.0,2.0,3.0,4.0", "a,b,c", "1.0,0.0,3.0"):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_bbx(raw))


class LoadLibraryTests(unittest.TestCase):
    def test_a_byte_order_mark_does_not_lose_every_row(self):
        # The real file is a spreadsheet export and begins with a BOM.  Reading it as
        # plain utf-8 makes the first field name unmatchable and silently yields an
        # empty index, so this is pinned rather than left to chance.
        path = write_library(LIBRARY)
        try:
            library = load_asset_library(path)
        finally:
            path.unlink()
        self.assertEqual(len(library), len(LIBRARY))
        self.assertIn("0_SM_Sofa_2", library)

    def test_the_vocabulary_is_the_lowercased_class_column(self):
        path = write_library(LIBRARY)
        try:
            vocabulary = library_vocabulary(load_asset_library(path))
        finally:
            path.unlink()
        self.assertIn("multi_person_sofa", vocabulary)
        self.assertIn("pillow", vocabulary)
        self.assertNotIn("Pillow", vocabulary)


class ClassifyRetrievalTests(unittest.TestCase):
    VOCABULARY = {
        "multi_person_sofa",
        "pillow",
        "small_potted_plant",
        "gaming_table",
        "chandelier",
        "storage_locker",
        "tall_bookshelf",
    }

    def entry(self, name):
        bbx, class_en, retrieval_class = LIBRARY[name]
        return {
            "bbx_m": parse_bbx(bbx),
            "class_en": class_en,
            "retrieval_class_en": retrieval_class,
            "scaling_strategy": "",
        }

    def test_a_pillow_retrieving_a_sofa_is_a_contradiction_the_library_testifies_to(self):
        self.assertEqual(
            classify_retrieval("pillow", self.entry("0_SM_Sofa_2"), self.VOCABULARY),
            "asset_class_contradicts_the_object_category",
        )

    def test_an_opaque_identifier_is_no_longer_an_abstention(self):
        # Fix97 to Fix100 abstained on d_1000003759813because the name says nothing.
        # The library records its class as exactly Pillow.
        self.assertEqual(
            classify_retrieval(
                "pillow", self.entry("d_1000003759813"), self.VOCABULARY
            ),
            "asset_class_matches_the_object_category",
        )

    def test_the_decor_false_positive_is_resolved(self):
        # a_SM_Decor_6 was flagged by the name heuristic; its class is exactly
        # Small_potted_plant.
        self.assertEqual(
            classify_retrieval(
                "small_potted_plant", self.entry("a_SM_Decor_6"), self.VOCABULARY
            ),
            "asset_class_matches_the_object_category",
        )

    def test_a_category_outside_the_vocabulary_abstains(self):
        # 'pen' is not a library class, so a pen retrieving a chandelier cannot be
        # judged by this comparison, however wrong it looks.
        self.assertEqual(
            classify_retrieval(
                "pen", self.entry("a_SM_Point_Lamp_4"), self.VOCABULARY
            ),
            "object_category_absent_from_library_vocabulary",
        )

    def test_a_failed_join_is_reported_as_such(self):
        self.assertEqual(
            classify_retrieval("pillow", None, self.VOCABULARY),
            "asset_absent_from_library",
        )


def placement_with(objects):
    return {"obj_info": {"scene_camera": {"pose_matrix_for_blender": None}, **objects}}


class IdentityVerificationTests(unittest.TestCase):
    """The claim five rounds of attribution rest on, checked against the library."""

    def run_audit(self, objects):
        path = write_library(LIBRARY)
        try:
            library = load_asset_library(path)
            return audit_scene(
                placement_with(objects),
                library,
                vocabulary=library_vocabulary(library),
                identity_tolerance=0.02,
                size_mismatch_factor=3.0,
                top_k=6,
            )
        finally:
            path.unlink()

    def test_pillow_16_computed_native_size_matches_the_authored_sofa(self):
        result = self.run_audit(
            {
                "pillow_16": {
                    "retrieved_asset": "0_SM_Sofa_2",
                    "length": [6.54, 2.47, 3.08],
                    "scale": [2.37, 2.37, 3.60],
                }
            }
        )
        item = result["all_findings"][0]
        self.assertTrue(item["identity_holds"])
        self.assertLess(item["identity_worst_edge_ratio"], 1.02)
        self.assertEqual(result["identity_pass_rate"], 1.0)

    def test_the_identity_holds_across_the_hand_checked_assets(self):
        objects = {
            "pen_0": ("a_SM_Point_Lamp_4", [0.05, 0.05, 1.81], [1.0, 1.0, 1.0]),
            "paper_cup_1": ("4_SM_Ventilation1_7", [0.92, 2.00, 0.92], [1.0, 1.0, 1.0]),
            "bookshelf_0": ("b_33", [1.20, 0.40, 1.91], [1.0, 1.0, 1.0]),
            "small_potted_plant_0": (
                "a_SM_Decor_6",
                [0.48, 0.37, 0.82],
                [1.31, 1.31, 2.28],
            ),
        }
        result = self.run_audit(
            {
                name: {"retrieved_asset": asset, "length": length, "scale": scale}
                for name, (asset, length, scale) in objects.items()
            }
        )
        self.assertEqual(result["identity_failed_count"], 0)

    def test_the_casino_table_disagreement_is_reported_not_hidden(self):
        # Two independent objects compute a native height of 1.24 m while the library
        # authors 0.924 m, a consistent factor of 1.345.  A 1.24 m casino table is
        # implausible, so the library is right and the tool must surface this.
        result = self.run_audit(
            {
                "gaming_table_0": {
                    "retrieved_asset": "44_sk75_CasinoTable02",
                    "length": [2.51, 1.15, 0.88],
                    "scale": [0.86, 0.71, 0.71],
                },
                "gaming_table_2": {
                    "retrieved_asset": "44_sk75_CasinoTable02",
                    "length": [5.80, 4.19, 0.92],
                    "scale": [1.98, 2.56, 0.74],
                },
            }
        )
        self.assertEqual(result["identity_failed_count"], 2)
        for item in result["worst_identity_failures"]:
            self.assertAlmostEqual(item["identity_worst_edge_ratio"], 1.345, delta=0.02)

    def test_an_axis_permutation_is_not_mistaken_for_a_disagreement(self):
        # The library records2.762, 1.042, 0.856 while this object's local axes put
        # the long edge last.  Sorting both sides before comparing is what makes the
        # verification about size rather than about axis conventions.
        result = self.run_audit(
            {
                "multi_person_sofa_0": {
                    "retrieved_asset": "0_SM_Sofa_2",
                    "length": [1.042, 0.856, 2.762],
                    "scale": [1.0, 1.0, 1.0],
                }
            }
        )
        self.assertTrue(result["all_findings"][0]["identity_holds"])


class LibrarySizeTests(unittest.TestCase):
    def run_audit(self, objects, **overrides):
        path = write_library(LIBRARY)
        kwargs = dict(
            identity_tolerance=0.02, size_mismatch_factor=3.0, top_k=6
        )
        kwargs.update(overrides)
        try:
            library = load_asset_library(path)
            return audit_scene(
                placement_with(objects),
                library,
                vocabulary=library_vocabulary(library),
                **kwargs,
            )
        finally:
            path.unlink()

    def test_pillow_16_is_a_retrieval_error_not_an_extreme_scale(self):
        # Clarifying, and the opposite of what this test first asserted.  Relative to
        # its own asset the 6.54 m block is inflated only 2.72 times by volume and
        # 2.37 times on its longest edge, both under the gate.  It occupies 81.6 per
        # cent of the frame because retrieval returned a 2.76 m sofa for a pillow, not
        # because the scale was extreme.  The two checks are genuinely orthogonal.
        result = self.run_audit(
            {
                "pillow_16": {
                    "retrieved_asset": "0_SM_Sofa_2",
                    "length": [6.54, 2.47, 3.08],
                    "scale": [2.37, 2.37, 3.60],
                }
            }
        )
        item = result["all_findings"][0]
        self.assertEqual(
            item["defect_reasons"], ["retrieved_asset_is_a_different_kind_of_object"]
        )
        self.assertAlmostEqual(
            item["rendered_over_library_volume_factor"], 2.724, places=2
        )
        self.assertAlmostEqual(
            item["rendered_over_library_longest_edge_ratio"], 6.54 / 2.762, places=3
        )

    def test_the_three_and_a_half_metre_picture_frame_is_caught_by_the_authored_size(
        self,
    ):
        # This is the case the cross-scene corpus was built for in Fix99, where the
        # in-scene peer median was exactly 1.00 because the whole family was wrong.
        # The authored dimensions settle it with no majority reference at all:
        # 3.51 / 0.701 = 5.01 on the longest edge.
        path = write_library({"a_SM_Frame04": ("0.701,0.035,0.537", "Wall_mounted_picture_frame", "Picture_frame")})
        try:
            library = load_asset_library(path)
            result = audit_scene(
                placement_with(
                    {
                        "wall_mounted_picture_frame_2": {
                            "retrieved_asset": "a_SM_Frame04",
                            "length": [3.51, 0.14, 0.97],
                            "scale": [5.00, 4.03, 1.81],
                        }
                    }
                ),
                library,
                vocabulary=library_vocabulary(library),
                identity_tolerance=0.02,
                size_mismatch_factor=3.0,
                top_k=6,
            )
        finally:
            path.unlink()
        item = result["all_findings"][0]
        self.assertIn(
            "rendered_size_far_larger_than_the_authored_asset", item["defect_reasons"]
        )
        self.assertAlmostEqual(
            item["rendered_over_library_longest_edge_ratio"], 3.51 / 0.701, places=3
        )
        self.assertEqual(
            item["retrieval_verdict"], "asset_class_matches_the_object_category"
        )

    def test_an_asset_used_at_its_authored_size_is_not_flagged(self):
        result = self.run_audit(
            {
                "pillow_0": {
                    "retrieved_asset": "d_1000003759813",
                    "length": [0.633, 0.210, 0.338],
                    "scale": [1.0, 1.0, 1.0],
                }
            }
        )
        self.assertEqual(result["all_findings"][0]["defect_reasons"], [])
        self.assertEqual(
            result["all_findings"][0]["retrieval_verdict"],
            "asset_class_matches_the_object_category",
        )

    def test_the_join_rate_is_reported_and_structure_is_excluded(self):
        result = self.run_audit(
            {
                "floor_0": {"retrieved_asset": "b_33", "length": [8.0, 8.0, 0.1]},
                "pillow_0": {
                    "retrieved_asset": "d_1000003759813",
                    "length": [0.633, 0.210, 0.338],
                    "scale": [1.0, 1.0, 1.0],
                },
                "mystery_0": {
                    "retrieved_asset": "not_in_the_library",
                    "length": [1.0, 1.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                },
            }
        )
        self.assertEqual(result["object_count"], 2)
        self.assertEqual(result["joined_to_library_count"], 1)
        self.assertAlmostEqual(result["join_rate"], 0.5)
        self.assertEqual(
            result["retrieval_verdict_counts"]["asset_absent_from_library"], 1
        )

    def test_a_shrunken_asset_is_reported_as_far_smaller(self):
        result = self.run_audit(
            {
                "sign_0": {
                    "retrieved_asset": "a_SM_CartonGarbage05",
                    "length": [0.098, 0.061, 0.026],
                    "scale": [0.1, 0.1, 0.1],
                }
            }
        )
        self.assertIn(
            "rendered_size_far_smaller_than_the_authored_asset",
            result["all_findings"][0]["defect_reasons"],
        )


class AssetNameWitnessTests(unittest.TestCase):
    """The curated bucket is not the asset's identity, so it cannot convict alone."""

    def run_audit(self, objects):
        path = write_library(LIBRARY)
        try:
            library = load_asset_library(path)
            return audit_scene(
                placement_with(objects),
                library,
                # The real vocabulary is 498 classes and abstained on zero of 374
                # objects, so a nine-entry fixture would abstain for a reason that does
                # not exist in the corpus.  The two categories under test are added
                # explicitly rather than by inventing library rows for them.
                vocabulary=library_vocabulary(library) | {"wardrobe", "bookshelf"},
                identity_tolerance=0.02,
                size_mismatch_factor=3.0,
                top_k=12,
            )
        finally:
            path.unlink()

    def test_a_wardrobe_retrieving_an_asset_named_wardrobe_is_not_a_substitution(self):
        # Fix102 reported this as a substitution because the bucket reads Storage_locker.
        # No asset can be a better answer for a wardrobe than one named Wardrobe.
        result = self.run_audit(
            {
                "wardrobe_0": {
                    "retrieved_asset": "a_SM_Wardrobe_01",
                    "length": [0.998, 0.544, 2.000],
                    "scale": [1.0, 1.0, 1.0],
                    "pcd_obb_size": [1.02, 0.56, 2.03],
                }
            }
        )
        item = result["all_findings"][0]
        self.assertEqual(
            item["retrieval_verdict"], "asset_class_contradicts_the_object_category"
        )
        self.assertEqual(item["label_relation"], "shares_nothing")
        self.assertEqual(
            item["asset_name_relation"], "asset_name_carries_the_head_noun"
        )
        self.assertEqual(result["substitution_count"], 0)
        self.assertEqual(result["excused_by_the_asset_name_count"], 1)

    def test_an_opaque_identifier_leaves_the_object_a_defect_on_no_evidence(self):
        # b_33 carries no words, so there is nothing to corroborate or contradict.  The
        # object stays a defect and is counted apart, because a rate that rests on an
        # absence of evidence has to be visible as such.
        result = self.run_audit(
            {
                "bookshelf_0": {
                    "retrieved_asset": "b_33",
                    "length": [1.200, 0.403, 1.910],
                    "scale": [1.0, 1.0, 1.0],
                    "pcd_obb_size": [1.22, 0.41, 1.93],
                }
            }
        )
        self.assertEqual(result["substitution_count"], 1)
        self.assertEqual(
            result["substitutions_resting_on_an_opaque_asset_name_count"], 1
        )

    def test_the_identifier_cannot_create_a_defect_only_excuse_one(self):
        # d_1000003759813 is opaque and its bucket is exactly Pillow.  An opaque name
        # must not turn an agreeing bucket into a defect.
        result = self.run_audit(
            {
                "pillow_0": {
                    "retrieved_asset": "d_1000003759813",
                    "length": [0.633, 0.210, 0.338],
                    "scale": [1.0, 1.0, 1.0],
                    "pcd_obb_size": [0.64, 0.22, 0.35],
                }
            }
        )
        item = result["all_findings"][0]
        self.assertEqual(item["asset_name_relation"], "asset_name_is_opaque")
        self.assertEqual(item["defect_reasons"], [])


class FallbackDamageTests(unittest.TestCase):
    """Does changing line 6414 buy anything?  This is the number that decides."""

    def run_audit(self, objects, **overrides):
        path = write_library(LIBRARY)
        kwargs = dict(
            identity_tolerance=0.02,
            size_mismatch_factor=3.0,
            fallback_evidence_factor=3.0,
            top_k=12,
        )
        kwargs.update(overrides)
        try:
            library = load_asset_library(path)
            return audit_scene(
                placement_with(objects),
                library,
                vocabulary=library_vocabulary(library),
                **kwargs,
            )
        finally:
            path.unlink()

    def test_an_unscaled_asset_over_a_fragment_of_depth_evidence_is_flagged(self):
        # The livingroom bookshelves: the observed box is a fragment of the order of 2
        # to 17 cm, the fallback declines to scale, and the authored 1.91 m locker lands
        # whole.  scale = [1,1,1] is therefore not a neutral default but the assertion
        # 'this object is exactly as large as its asset', and the library is the witness
        # that the assertion resolves to 1.91 m.
        result = self.run_audit(
            {
                "bookshelf_0": {
                    "retrieved_asset": "b_33",
                    "length": [1.200, 0.403, 1.910],
                    "scale": [1.0, 1.0, 1.0],
                    "pcd_obb_size": [0.17, 0.05, 0.12],
                }
            }
        )
        item = result["all_findings"][0]
        self.assertEqual(
            item["fallback_scale_signature"], "unscaled_on_the_small_object_path"
        )
        self.assertIn(
            "unscaled_asset_is_far_larger_than_the_depth_evidence",
            item["defect_reasons"],
        )
        self.assertTrue(item["identity_holds"])
        self.assertAlmostEqual(
            item["rendered_over_observed_longest_edge_ratio"], 1.910 / 0.17, places=3
        )
        self.assertEqual(result["fallback_damaging_count"], 1)

    def test_an_unscaled_asset_that_matches_its_evidence_is_not_flagged(self):
        # The bound must not convict every unit scale: where the depth box agrees with
        # the asset, declining to scale was the right answer.
        result = self.run_audit(
            {
                "pillow_0": {
                    "retrieved_asset": "d_1000003759813",
                    "length": [0.633, 0.210, 0.338],
                    "scale": [1.0, 1.0, 1.0],
                    "pcd_obb_size": [0.64, 0.22, 0.35],
                }
            }
        )
        item = result["all_findings"][0]
        self.assertEqual(
            item["fallback_scale_signature"], "unscaled_on_the_small_object_path"
        )
        self.assertEqual(result["fallback_candidate_count"], 1)
        self.assertEqual(result["fallback_damaging_count"], 0)

    def test_the_candidate_count_is_a_superset_and_the_damage_count_the_subset(self):
        result = self.run_audit(
            {
                "bookshelf_0": {
                    "retrieved_asset": "b_33",
                    "length": [1.200, 0.403, 1.910],
                    "scale": [1.0, 1.0, 1.0],
                    "pcd_obb_size": [0.17, 0.05, 0.12],
                },
                "pillow_0": {
                    "retrieved_asset": "d_1000003759813",
                    "length": [0.633, 0.210, 0.338],
                    "scale": [1.0, 1.0, 1.0],
                    "pcd_obb_size": [0.64, 0.22, 0.35],
                },
                "small_potted_plant_0": {
                    "retrieved_asset": "a_SM_Decor_6",
                    "length": [0.48, 0.37, 0.82],
                    "scale": [1.31, 1.31, 2.28],
                    "pcd_obb_size": [0.30, 0.24, 0.35],
                },
            }
        )
        self.assertEqual(result["fallback_candidate_count"], 2)
        self.assertEqual(result["fallback_damaging_count"], 1)
        self.assertEqual(
            result["fallback_signature_counts"]["scale_was_estimated"], 1
        )
        self.assertAlmostEqual(result["fallback_damaging_share_of_scene"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
