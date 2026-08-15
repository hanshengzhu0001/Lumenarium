#!/usr/bin/env python3
"""Tests for Fix92: auditing the collision instrument before its score.

The collision family is measured on oriented bounding boxes treated as solid
prisms.  Two properties hold by construction and are pinned here:

* the prism contains the true mesh, so the reported pairs are a complete superset
  of the truly interpenetrating pairs and the score is a lower bound;
* the cavity under a table top is solid, so a correctly tucked chair is reported
  as a collision.

The second property is the artefact the audit has to expose, and the first is
what makes auditing only the reported pairs sufficient.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_physical_realizability as evaluator
import sceneproof_collision_measurement_audit_fix92 as fix92


def pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> list[list[float]]:
    matrix = np.eye(4)
    matrix[:3, 3] = (x, y, z)
    return matrix.tolist()


def box_bbox(centre, half) -> list[list[float]]:
    return [
        [
            centre[0] + sx * half[0],
            centre[1] + sy * half[1],
            centre[2] + sz * half[2],
        ]
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    ]


def solid(centre, half, **extra) -> dict:
    return {
        "pose_matrix_for_blender": pose(*centre),
        "bbox": box_bbox(centre, half),
        "length": [2 * half[0], 2 * half[1], 2 * half[2]],
        **extra,
    }


def evaluator_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        collision_volume_tolerance=1e-6,
        collision_fraction_tolerance=0.05,
        contact_tolerance=0.05,
        containment_tolerance=0.05,
        support_overlap_tolerance=0.9,
        plane_tolerance=0.05,
        plane_orientation_tolerance=15.0,
        boundary_tolerance=0.05,
        semantic_tolerance=0.25,
        critical_threshold=0.5,
        collision_pairs_csv=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def tucked_chair_scene() -> dict:
    """A table with a chair correctly tucked under its top.

    The table top sits at z = 0.75.  The chair's seat and back occupy the space
    beneath and beside it, entirely within the table's vertical span, which is
    exactly the legal arrangement that a solid prism reports as a collision.
    """
    return {
        "obj_info": {
            "floor_0": solid((0.0, 0.0, 0.0), (5.0, 5.0, 0.02)),
            "table_0": solid(
                (0.0, 0.0, 0.375), (0.8, 0.4, 0.375), supported="floor_0"
            ),
            "chair_0": solid(
                (0.0, 0.25, 0.35), (0.25, 0.25, 0.35), supported="floor_0"
            ),
        },
        "reference_obj": "floor_0",
    }


class TuckedChairIsReportedAsCollisionTest(unittest.TestCase):
    """The artefact itself, reproduced at unit level."""

    def setUp(self) -> None:
        self.document = tucked_chair_scene()
        self.metrics, self.rows = evaluator.evaluate_scene(
            self.document,
            self.document,
            evaluator_args(collision_pairs_csv="unused_but_enables_collection"),
        )

    def test_a_legal_tuck_costs_the_whole_collision_term(self) -> None:
        terms = {row["object_id"]: row["collision_term"] for row in self.rows}
        self.assertEqual(terms["chair_0"], 0.0)
        self.assertEqual(terms["table_0"], 0.0)
        self.assertEqual(self.metrics["families"]["collision"]["score"], 0.0)

    def test_the_pair_is_exported_with_the_geometry_that_explains_it(self) -> None:
        details = self.metrics["collision_pair_details"]
        self.assertEqual(len(details), 1)
        detail = details[0]
        self.assertEqual(detail["smaller_object_id"], "chair_0")
        self.assertEqual(detail["larger_object_id"], "table_0")
        # The chair's whole vertical span lies inside the table's span.
        self.assertGreaterEqual(detail["smaller_z_min"], detail["larger_z_min"] - 1e-9)
        self.assertLessEqual(detail["smaller_z_max"], detail["larger_z_max"] + 1e-9)
        self.assertGreater(detail["intersection_over_smaller_footprint"], 0.25)

    def test_penetration_depth_is_exported_in_metres(self) -> None:
        detail = self.metrics["collision_pair_details"][0]
        # The chair spans y in [0, 0.5] and the table y in [-0.4, 0.4], so the
        # lateral push-out distance is 0.4 m; vertically the overlap is the chair's
        # whole 0.7 m height.  The minimum translation is therefore the lateral one.
        self.assertAlmostEqual(detail["penetration_depth_m"], 0.4, places=9)
        self.assertEqual(detail["penetration_depth_axis"], "lateral")

    def test_per_object_absolute_quantities_are_reported_unconditionally(self) -> None:
        metrics, rows = evaluator.evaluate_scene(
            self.document, self.document, evaluator_args()
        )
        by_id = {row["object_id"]: row for row in rows}
        self.assertAlmostEqual(
            by_id["chair_0"]["collision_max_penetration_depth_m"], 0.4, places=9
        )
        self.assertGreater(
            by_id["chair_0"]["collision_max_intersection_volume_m3"], 0.0
        )
        # Structural objects are outside the collision denominator entirely.
        self.assertNotIn("floor_0", by_id)

    def test_details_are_absent_unless_requested(self) -> None:
        metrics, _ = evaluator.evaluate_scene(
            self.document, self.document, evaluator_args()
        )
        self.assertNotIn("collision_pair_details", metrics)
        self.assertEqual(
            metrics["families"]["collision"]["score"],
            self.metrics["families"]["collision"]["score"],
        )

    def test_the_measurement_model_is_declared_unconditionally(self) -> None:
        metrics, _ = evaluator.evaluate_scene(
            self.document, self.document, evaluator_args()
        )
        self.assertEqual(
            metrics["collision_geometry_model"], "oriented_bounding_box_solid_prism"
        )
        self.assertTrue(
            metrics["collision_score_is_lower_bound_under_finer_geometry"]
        )


class DeclaredSupportIsNeverACollisionTest(unittest.TestCase):
    def test_a_cup_on_a_table_is_not_reported(self) -> None:
        document = {
            "obj_info": {
                "floor_0": solid((0.0, 0.0, 0.0), (5.0, 5.0, 0.02)),
                "table_0": solid(
                    (0.0, 0.0, 0.375), (0.8, 0.4, 0.375), supported="floor_0"
                ),
                "cup_0": solid(
                    (0.0, 0.0, 0.79), (0.04, 0.04, 0.04), supported="table_0"
                ),
            },
            "reference_obj": "floor_0",
        }
        metrics, _ = evaluator.evaluate_scene(
            document, document, evaluator_args(collision_pairs_csv="enabled")
        )
        self.assertEqual(metrics["collision_pair_details"], [])
        self.assertEqual(metrics["unintended_collision_pairs"], 0)


class ClassificationTest(unittest.TestCase):
    @staticmethod
    def row(**overrides) -> dict:
        base = {
            "first_id": "a",
            "second_id": "b",
            "overlap_fraction": 0.5,
            "intersection_volume_m3": 0.01,
            "z_overlap_m": 0.5,
            "smaller_z_min": 0.1,
            "smaller_z_max": 0.6,
            "larger_z_min": 0.0,
            "larger_z_max": 0.75,
            "intersection_over_smaller_footprint": 0.6,
        }
        base.update(overrides)
        return base

    def classify(self, **overrides) -> str:
        return fix92.classify(
            self.row(**overrides),
            contact_band_m=0.02,
            cavity_area_fraction=0.25,
            span_tolerance_m=0.01,
        )

    def test_a_span_inside_a_span_with_real_footprint_overlap_is_enclosure(self):
        self.assertEqual(self.classify(), "enclosure_shaped")

    def test_a_chair_back_rising_above_a_table_is_still_the_cavity_artefact(self):
        # The casino case: same floor, chair back above the table top, footprint
        # still inside the table's hull.  The first classifier called this real.
        self.assertEqual(
            self.classify(smaller_z_min=0.0, larger_z_min=0.0, smaller_z_max=1.03),
            "partial_vertical_enclosure",
        )

    def test_a_thin_vertical_overlap_is_a_missing_support_edge(self) -> None:
        self.assertEqual(
            self.classify(z_overlap_m=0.004), "contact_band_only"
        )

    def test_a_floating_object_above_a_lower_one_is_lateral(self) -> None:
        # Bases differ, so neither enclosure pattern applies: this is a genuine
        # side-on or straddling overlap.
        self.assertEqual(
            self.classify(smaller_z_min=0.4, larger_z_min=0.0, smaller_z_max=1.2),
            "lateral_interpenetration",
        )

    def test_a_grazing_footprint_is_lateral_even_when_vertically_enclosed(self):
        self.assertEqual(
            self.classify(intersection_over_smaller_footprint=0.05),
            "lateral_interpenetration",
        )

    def test_a_grazing_footprint_is_lateral_even_when_sharing_a_base(self) -> None:
        self.assertEqual(
            self.classify(
                smaller_z_min=0.0,
                larger_z_min=0.0,
                smaller_z_max=1.03,
                intersection_over_smaller_footprint=0.05,
            ),
            "lateral_interpenetration",
        )

    def test_an_unmeasurable_span_is_never_called_an_artefact(self) -> None:
        self.assertEqual(
            self.classify(smaller_z_min=None), "lateral_interpenetration"
        )
        self.assertEqual(
            self.classify(smaller_z_min=""), "lateral_interpenetration"
        )


class RescoreTest(unittest.TestCase):
    """Exempting a class must change only the numerator."""

    def setUp(self) -> None:
        self.object_ids = ["chair_0", "table_0", "lamp_0"]
        self.rows = [
            {
                "first_id": "chair_0",
                "second_id": "table_0",
                "overlap_fraction": "0.4",
                "_class": "enclosure_shaped",
            },
            {
                "first_id": "lamp_0",
                "second_id": "table_0",
                "overlap_fraction": "0.3",
                "_class": "lateral_interpenetration",
            },
        ]

    def test_as_is_matches_the_evaluator_formula(self) -> None:
        # chair and table and lamp all exceed the tolerance, so all three score 0.
        self.assertAlmostEqual(
            fix92.rescore(
                self.rows, self.object_ids, set(), fraction_tolerance=0.05
            ),
            0.0,
        )

    def test_exempting_a_class_only_clears_the_objects_it_explains(self) -> None:
        score = fix92.rescore(
            self.rows,
            self.object_ids,
            {"enclosure_shaped"},
            fraction_tolerance=0.05,
        )
        # chair_0 is cleared; lamp_0 and table_0 still collide laterally.
        self.assertAlmostEqual(score, 1.0 / 3.0)

    def test_the_denominator_never_changes(self) -> None:
        for exempt in ((), ("enclosure_shaped",), fix92.CLASSES):
            with self.subTest(exempt=exempt):
                score = fix92.rescore(
                    self.rows,
                    self.object_ids,
                    set(exempt),
                    fraction_tolerance=0.05,
                )
                self.assertIn(score, [n / 3.0 for n in range(4)])

    def test_exempting_everything_recovers_a_perfect_score(self) -> None:
        self.assertAlmostEqual(
            fix92.rescore(
                self.rows,
                self.object_ids,
                set(fix92.CLASSES),
                fraction_tolerance=0.05,
            ),
            1.0,
        )

    def test_an_object_with_no_pair_already_scores_one(self) -> None:
        score = fix92.rescore(
            [],["a", "b"], set(), fraction_tolerance=0.05
        )
        self.assertAlmostEqual(score, 1.0)


class AuditSceneTest(unittest.TestCase):
    def setUp(self) -> None:
        self.object_ids = ["chair_0", "chair_1", "table_0", "shelf_0", "box_0"]
        self.rows = [
            # Two chairs tucked under the same table.
            {
                "first_id": "chair_0",
                "second_id": "table_0",
                "overlap_fraction": "0.4",
                "intersection_volume_m3": "0.02",
                "z_overlap_m": "0.5",
                "smaller_z_min": "0.1",
                "smaller_z_max": "0.6",
                "larger_z_min": "0.0",
                "larger_z_max": "0.75",
                "intersection_over_smaller_footprint": "0.6",
            },
            {
                "first_id": "chair_1",
                "second_id": "table_0",
                "overlap_fraction": "0.35",
                "intersection_volume_m3": "0.018",
                "z_overlap_m": "0.5",
                "smaller_z_min": "0.1",
                "smaller_z_max": "0.6",
                "larger_z_min": "0.0",
                "larger_z_max": "0.75",
                "intersection_over_smaller_footprint": "0.55",
            },
            # A box driven sideways into a shelf: a real interpenetration.
            {
                "first_id": "box_0",
                "second_id": "shelf_0",
                "overlap_fraction": "0.6",
                "intersection_volume_m3": "0.05",
                "z_overlap_m": "0.3",
                "smaller_z_min": "0.0",
                "smaller_z_max": "0.4",
                "larger_z_min": "0.1",
                "larger_z_max": "1.8",
                "intersection_over_smaller_footprint": "0.5",
            },
        ]
        self.result = fix92.audit_scene(
            self.rows,
            self.object_ids,
            fraction_tolerance=0.05,
            contact_band_m=0.02,
            cavity_area_fraction=0.25,
            span_tolerance_m=0.01,
        )

    def test_classes_partition_the_reported_pairs(self) -> None:
        total = sum(
            self.result["classes"][name]["pair_count"] for name in fix92.CLASSES
        )
        self.assertEqual(total, self.result["reported_pair_count"])

    def test_the_tucked_chairs_land_in_the_artefact_class(self) -> None:
        block = self.result["classes"]["enclosure_shaped"]
        self.assertEqual(block["pair_count"], 2)
        self.assertEqual(
            block["object_ids_explained_solely_by_this_class"],
            ["chair_0", "chair_1", "table_0"],
        )

    def test_the_real_pair_is_not_exempted_by_the_artefact_class(self) -> None:
        # Exempting the tucks clears three of five objects; the box and the shelf
        # keep their zero.
        self.assertAlmostEqual(
            self.result["classes"]["enclosure_shaped"]["score_if_exempt"], 3 / 5
        )
        self.assertAlmostEqual(
            self.result["score_if_only_lateral_interpenetration_counts"], 3 / 5
        )

    def test_worst_lateral_pairs_are_reported_for_inspection(self) -> None:
        worst = self.result["worst_lateral_pairs"]
        self.assertEqual(len(worst), 1)
        self.assertEqual(
            {worst[0]["first_id"], worst[0]["second_id"]}, {"box_0", "shelf_0"}
        )

    def test_the_lower_bound_property_is_recorded(self) -> None:
        model = self.result["measurement_model"]
        self.assertTrue(model["prism_contains_true_mesh"])
        self.assertTrue(
            model["reported_pairs_are_complete_superset_of_true_pairs"]
        )
        self.assertTrue(model["as_is_score_is_a_lower_bound"])
        self.assertTrue(model["exempt_scores_are_attribution_not_measurement"])
        self.assertTrue(model["certification_requires_true_mesh_intersection"])

    def test_as_is_score_is_reported_alongside_the_attribution(self) -> None:
        self.assertAlmostEqual(self.result["collision_score_as_is"], 0.0)


class ObjectIdLoadingTest(unittest.TestCase):
    """The denominator must come from the evaluator, not be re-derived."""

    def setUp(self) -> None:
        self.path = Path(__file__).resolve().parent / "_fix92_objects_tmp.csv"
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["version", "scene", "object_id", "collision_term"]
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "version": "v",
                        "scene": "s",
                        "object_id": "chair_0",
                        "collision_term": "0.0",
                    },
                    {
                        "version": "v",
                        "scene": "s",
                        "object_id": "table_0",
                        "collision_term": "1.0",
                    },
                    # A camera or structural row carries no collision term.
                    {
                        "version": "v",
                        "scene": "s",
                        "object_id": "floor_0",
                        "collision_term": "",
                    },
                    # Another scene must not leak in.
                    {
                        "version": "v",
                        "scene": "other",
                        "object_id": "sofa_0",
                        "collision_term": "1.0",
                    },
                ]
            )

    def tearDown(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def test_only_scoring_objects_of_the_requested_scene_are_counted(self) -> None:
        self.assertEqual(
            fix92.load_object_ids(self.path, "s", "v"), ["chair_0", "table_0"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
