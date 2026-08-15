#!/usr/bin/env python3
"""Tests for Fix91: separating a legitimate correction from a vacuous one.

Backfilling the measured ground slab raised Smoke5 support scores steeply, but
the slab is a 10 m by 10 m by 0.04 m procedural placeholder.  Its top face is
real and recovering the contact gap is a genuine correction; its lateral extent
carries no room information, so recovering containment and footprint overlap only
makes those summands trivially satisfied.

These tests pin the split, and pin the evaluator behaviour that keeps the
distinction visible.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_physical_realizability as evaluator
import sceneproof_support_gain_decomposition_fix91 as fix91


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
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]


def solid(centre, half, **extra) -> dict:
    info = {
        "pose_matrix_for_blender": pose(*centre),
        "bbox": box_bbox(centre, half),
        "length": [2 * half[0], 2 * half[1], 2 * half[2]],
        "retrieved_asset": "asset",
    }
    info.update(extra)
    return info


def evaluator_args(**overrides) -> argparse.Namespace:
    values = {
        "collision_volume_tolerance": 1e-6,
        "collision_fraction_tolerance": 0.05,
        "contact_tolerance": 0.05,
        "containment_tolerance": 0.05,
        "support_overlap_tolerance": 0.9,
        "plane_tolerance": 0.05,
        "plane_orientation_tolerance": 15.0,
        "boundary_tolerance": 0.05,
        "semantic_angle_tolerance": 20.0,
        "distance_tolerance": 0.1,
        "repair_degenerate_footprints": False,
        "abstain_on_unmeasurable_footprints": False,
        "placeholder_structural_lateral_extent": False,
        "_structural_geometry_sidecar": {},
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class PlaceholderSlabScene:
    """A 4 m room sitting on the pipeline's 10 m placeholder slab."""

    def document(self) -> dict:
        # The slab spans z in [-0.02, +0.02], so its top face is at +0.02 and every
        # resting child must start exactly there.
        info = {
            # Measured geometry: exactly 10 x 10 x 0.04, top face at +0.02.
            "floor_0": solid((0.0, 0.0, 0.0), (5.0, 5.0, 0.02)),
            # Rests on the slab: z from 0.02 to 0.72.
            "table_0": solid((1.0, 1.0, 0.37), (0.4, 0.4, 0.35), supported="floor_0"),
            # Rests on the slab as well, placed clear of the table.
            "chair_0": solid(
                (-1.0, -1.0, 0.27), (0.25, 0.25, 0.25), supported="floor_0"
            ),
            # Rests on the table's real asset bound: z from 0.72 to 0.82.
            "cup_0": solid(
                (1.0, 1.0, 0.77), (0.05, 0.05, 0.05), supported="table_0"
            ),
        }
        return {"obj_info": info, "reference_obj": "floor_0"}

    def score(self, **flags):
        document = self.document()
        metrics, rows = evaluator.evaluate_scene(
            document, document, evaluator_args(**flags)
        )
        return metrics, {row["object_id"]: row for row in rows}


class VacuityDemonstrationTest(unittest.TestCase):
    """With the placeholder slab present, lateral checks stop discriminating."""

    def setUp(self) -> None:
        self.scene = PlaceholderSlabScene()
        self.metrics, self.rows = self.scene.score()

    def test_every_slab_child_is_trivially_contained(self) -> None:
        for child in ("chair_0", "table_0"):
            self.assertEqual(self.rows[child]["support_containment_error_m"], 0.0)
            self.assertAlmostEqual(
                self.rows[child]["support_footprint_overlap_ratio"], 1.0
            )
            self.assertAlmostEqual(self.rows[child]["support_term"], 1.0, places=9)

    def test_boundary_is_trivially_satisfied(self) -> None:
        self.assertAlmostEqual(self.metrics["families"]["boundary"]["score"], 1.0)


class PlaceholderLateralExtentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = PlaceholderSlabScene()
        self.metrics, self.rows = self.scene.score(
            placeholder_structural_lateral_extent=True
        )

    def test_slab_children_keep_only_the_contact_gap_summand(self) -> None:
        for child in ("chair_0", "table_0"):
            self.assertEqual(self.rows[child]["support_summand_count"], 1)
            self.assertIsNone(self.rows[child]["support_containment_error_m"])
            self.assertIsNone(self.rows[child]["support_footprint_overlap_ratio"])
            self.assertEqual(
                self.rows[child]["support_lateral_extent_unmeasurable_parent_id"],
                "floor_0",
            )

    def test_real_asset_parents_keep_all_three_summands(self) -> None:
        self.assertEqual(self.rows["cup_0"]["support_summand_count"], 3)
        self.assertEqual(self.rows["cup_0"]["support_containment_error_m"], 0.0)

    def test_denominator_is_preserved(self) -> None:
        plain, _ = self.scene.score()
        self.assertEqual(
            plain["families"]["support"]["n"],
            self.metrics["families"]["support"]["n"],
        )

    def test_boundary_becomes_unanswerable(self) -> None:
        self.assertEqual(self.metrics["families"]["boundary"]["n"], 0)
        self.assertIsNone(self.metrics["families"]["boundary"]["score"])
        self.assertEqual(self.metrics["abstained_counts"]["boundary"], 3)

    def test_estimand_change_is_flagged_even_though_n_is_intact(self) -> None:
        self.assertTrue(self.metrics["estimand_changed_by_partial_summands"])
        self.assertFalse(self.metrics["scores_comparable_to_non_abstained_runs"])
        self.assertEqual(
            self.metrics["partial_summand_support_terms"], ["chair_0", "table_0"]
        )

    def test_contact_gap_is_still_measured_against_the_real_top_face(self) -> None:
        for child in ("chair_0", "table_0"):
            self.assertAlmostEqual(
                self.rows[child]["support_contact_gap_m"], 0.0, places=9
            )
            self.assertAlmostEqual(self.rows[child]["support_term"], 1.0, places=9)


class GainDecompositionTest(unittest.TestCase):
    """The closed-form split of a slab child's support gain."""

    def setUp(self) -> None:
        self.directory = Path(__file__).resolve().parent
        self.baseline_csv = self.directory / "_fix91_asis_tmp.csv"
        self.backfilled_csv = self.directory / "_fix91_sidecar_tmp.csv"
        self.placement = {
            "obj_info": {
                "floor_0": {},
                "table_0": {},
                "chair_0": {"supported": "floor_0"},
                "cup_0": {"supported": "table_0"},
            }
        }
        self.write(
            self.baseline_csv,
            [
                self.row(
                    "chair_0",
                    support_term=0.2,
                    gap=0.02,
                    containment=float("inf"),
                    overlap=0.0,
                ),
                self.row(
                    "cup_0", support_term=1.0, gap=0.0, containment=0.0, overlap=1.0
                ),
            ],
        )
        self.write(
            self.backfilled_csv,
            [
                self.row(
                    "chair_0", support_term=1.0, gap=0.0, containment=0.0, overlap=1.0
                ),
                self.row(
                    "cup_0", support_term=1.0, gap=0.0, containment=0.0, overlap=1.0
                ),
            ],
        )

    def tearDown(self) -> None:
        for path in (self.baseline_csv, self.backfilled_csv):
            if path.exists():
                path.unlink()

    @staticmethod
    def row(object_id, *, support_term, gap, containment, overlap) -> dict:
        return {
            "version": "v",
            "scene": "s",
            "object_id": object_id,
            "support_term": support_term,
            "support_contact_gap_m": gap,
            "support_containment_error_m": containment,
            "support_footprint_overlap_ratio": overlap,
        }

    @staticmethod
    def write(path: Path, rows: list[dict]) -> None:
        fieldnames = [
            "version",
            "scene",
            "object_id",
            "support_term",
            "support_contact_gap_m",
            "support_containment_error_m",
            "support_footprint_overlap_ratio",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def decompose(self):
        return fix91.decompose(
            fix91.load_rows(self.baseline_csv, "s", "v"),
            fix91.load_rows(self.backfilled_csv, "s", "v"),
            self.placement,
            contact_tolerance=0.05,
            containment_tolerance=0.05,
            overlap_tolerance=0.9,
        )

    def test_total_matches_the_observed_term_change(self) -> None:
        result = self.decompose()
        # One object moved 0.2 -> 1.0 out of two support terms.
        self.assertAlmostEqual(result["total_support_delta"], 0.8 / 2)

    def test_only_the_contact_gap_part_is_legitimate(self) -> None:
        result = self.decompose()
        # Closed form: (1 - (1 - 0.02/0.05)) / 3 = 0.4/3 per object.
        self.assertAlmostEqual(
            result["legitimate_contact_gap_delta"], (0.4 / 3) / 2
        )
        self.assertAlmostEqual(result["vacuous_lateral_delta"], (2 / 3) / 2)
        self.assertAlmostEqual(result["legitimate_fraction"], (0.4 / 3) / 0.8)
        self.assertLess(result["legitimate_fraction"], 0.17)

    def test_structural_parents_carry_the_vacuous_part(self) -> None:
        result = self.decompose()
        self.assertAlmostEqual(
            result["structural_parent_lateral_delta"], (2 / 3) / 2
        )
        self.assertAlmostEqual(
            result["structural_parent_contact_gap_delta"], (0.4 / 3) / 2
        )

    def test_unchanged_objects_are_not_listed(self) -> None:
        result = self.decompose()
        self.assertEqual(
            [entry["object_id"] for entry in result["per_object"]], ["chair_0"]
        )
        entry = result["per_object"][0]
        self.assertTrue(entry["parent_is_structural_placeholder"])
        self.assertAlmostEqual(entry["legitimate_gap_gain"], 0.4 / 3)
        self.assertAlmostEqual(entry["vacuous_lateral_gain"], 2 / 3)

    def test_no_change_yields_no_gain(self) -> None:
        self.write(
            self.backfilled_csv,
            [
                self.row(
                    "chair_0",
                    support_term=0.2,
                    gap=0.02,
                    containment=float("inf"),
                    overlap=0.0,
                ),
                self.row(
                    "cup_0", support_term=1.0, gap=0.0, containment=0.0, overlap=1.0
                ),
            ],
        )
        result = self.decompose()
        self.assertAlmostEqual(result["total_support_delta"], 0.0)
        self.assertIsNone(result["legitimate_fraction"])
        self.assertEqual(result["per_object"], [])


class SidecarExcludesCameraTest(unittest.TestCase):
    def test_camera_is_never_backfilled(self) -> None:
        document = {
            "obj_info": {
                "floor_0": {"pose_matrix_for_blender": pose()},
                "scene_camera": {"pose_matrix_for_blender": pose(0, 0, 1.5)},
                "chair_0": solid(
                    (0.0, 0.0, 0.27), (0.25, 0.25, 0.25), supported="floor_0"
                ),
            },
            "reference_obj": "floor_0",
        }
        sidecar = {
            "floor_0": {
                "bbox": box_bbox((0.0, 0.0, 0.0), (5.0, 5.0, 0.02)),
                "length": [10.0, 10.0, 0.04],
            },
            "scene_camera": {
                "bbox": box_bbox((0.0, 0.0, 1.5), (0.1, 0.1, 0.1)),
                "length": [0.2, 0.2, 0.2],
            },
        }
        metrics, _ = evaluator.evaluate_scene(
            document,
            document,
            evaluator_args(_structural_geometry_sidecar=sidecar),
        )
        self.assertEqual(
            metrics["structural_geometry_backfilled_object_ids"], ["floor_0"]
        )


class PerObjectLineTest(unittest.TestCase):
    """The per-object log line must bind every value to its own label.

    The first Smoke5 run of Fix91 printed nine arguments into eight fields, so
    every value was shifted one slot left: the boolean placeholder flag appeared
    as ``term 1.0000``, the as-is term appeared as the sidecar term, and the
    genuine gap gain appeared under the ``vacuous`` label.  The aggregates were
    correct, but the line invited exactly the wrong reading.
    """

    ENTRY = {
        "object_id": "bedside_table_0",
        "support_parent_id": "floor_0",
        "parent_is_structural_placeholder": True,
        "support_term_before": 0.2,
        "support_term_after": 0.9999999,
        "contact_gap_before_m": 0.02,
        "contact_gap_after_m": 0.0,
        "legitimate_gap_gain": 0.4 / 3,
        "vacuous_lateral_gain": 2 / 3,
    }

    def test_labels_carry_the_values_they_name(self) -> None:
        line = fix91.format_per_object_line(self.ENTRY)
        self.assertIn("bedside_table_0: parent=floor_0", line)
        self.assertIn("placeholder=True", line)
        self.assertIn("term 0.2000->1.0000", line)
        self.assertIn("gap 0.020000m->0.000000m", line)
        self.assertIn("legit=+0.1333", line)
        self.assertIn("vacuous=+0.6667", line)

    def test_the_shifted_rendering_can_no_longer_occur(self) -> None:
        line = fix91.format_per_object_line(self.ENTRY)
        self.assertNotIn("term 1.0000->0.2000", line)
        self.assertNotIn("parent=bedside_table_0", line)
        self.assertNotIn("placeholder=floor_0", line)
        self.assertNotIn("vacuous=+0.1333", line)

    def test_an_unmeasured_gap_is_not_rendered_as_zero(self) -> None:
        entry = dict(self.ENTRY, contact_gap_before_m=None)
        self.assertIn("gap unmeasured->", fix91.format_per_object_line(entry))
        entry = dict(self.ENTRY, contact_gap_before_m=float("inf"))
        self.assertIn("gap inf->", fix91.format_per_object_line(entry))


if __name__ == "__main__":
    unittest.main(verbosity=2)
