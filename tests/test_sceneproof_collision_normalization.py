#!/usr/bin/env python3
"""Tests for Fix93: is the collision normalization well posed?

Two things are pinned here.

First, the penetration depth is *exact* for the geometry the evaluator uses.  A
prism is a convex polygon crossed with a vertical interval, so its Minkowski
difference with another prism is a product set, and the distance from the origin
to the boundary of a product set is the smaller of the factors' distances.  The
tests check that composition against hand-computed values, including a rotated
polygon where the minimum translation is along neither world axis.

Second, the disagreement between normalizations is reproduced from the actual
Smoke5 numbers: nine cubic millimetres between two books is fully penalised while
sixty-one litres between a bin and a cup is not penalised at all.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_physical_realizability as evaluator
import sceneproof_collision_normalization_audit_fix93 as fix93


def rectangle(x_min, x_max, y_min, y_max) -> np.ndarray:
    return evaluator.convex_hull(
        np.asarray(
            [
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            ],
            dtype=np.float64,
        )
    )


class PenetrationDepthExactnessTest(unittest.TestCase):
    def prism(self, polygon, z_min, z_max):
        return evaluator.Geometry(
            name="probe",
            info={},
            matrix=np.eye(4),
            local_corners=np.zeros((8, 3)),
            world_corners=np.zeros((8, 3)),
            polygon=polygon,
            z_min=z_min,
            z_max=z_max,
            volume=1.0,
        )

    def test_disjoint_polygons_have_zero_depth(self) -> None:
        left = rectangle(0.0, 1.0, 0.0, 1.0)
        right = rectangle(2.0, 3.0, 0.0, 1.0)
        self.assertEqual(evaluator.polygon_penetration_depth_2d(left, right), 0.0)

    def test_touching_polygons_have_zero_depth(self) -> None:
        left = rectangle(0.0, 1.0, 0.0, 1.0)
        right = rectangle(1.0, 2.0, 0.0, 1.0)
        self.assertEqual(evaluator.polygon_penetration_depth_2d(left, right), 0.0)

    def test_the_minimum_axis_is_chosen_not_the_first(self) -> None:
        # Overlap is 0.9 along x and 0.1 along y.
        left = rectangle(0.0, 1.0, 0.0, 1.0)
        right = rectangle(0.1, 1.1, 0.9, 1.9)
        self.assertAlmostEqual(
            evaluator.polygon_penetration_depth_2d(left, right), 0.1, places=12
        )

    def test_a_fully_contained_polygon_needs_the_shortest_way_out(self) -> None:
        outer = rectangle(0.0, 10.0, 0.0, 10.0)
        inner = rectangle(4.0, 5.0, 1.0, 2.0)
        # The overlap *length* along y is 1.0, the inner rectangle's own height,
        # but translating by 1.0 leaves it still inside.  Escaping downwards costs
        # its height plus the 1.0 gap to y = 0, which is the shortest way out.
        self.assertAlmostEqual(
            evaluator.polygon_penetration_depth_2d(outer, inner), 2.0, places=12
        )

    def test_a_rotated_overlap_resolves_along_the_edge_normal(self) -> None:
        # A diamond whose left corner pokes into a wall occupying x <= 0.
        diamond = evaluator.convex_hull(
            np.asarray(
                [(-0.1, 0.0), (0.4, 0.5), (0.9, 0.0), (0.4, -0.5)], dtype=np.float64
            )
        )
        wall = rectangle(-5.0, 0.0, -5.0, 5.0)
        # Pushing along +x by 0.1 clears it; every other axis costs more.
        self.assertAlmostEqual(
            evaluator.polygon_penetration_depth_2d(diamond, wall), 0.1, places=12
        )

    def test_degenerate_polygons_yield_zero(self) -> None:
        point = np.asarray([(0.0, 0.0)], dtype=np.float64)
        square = rectangle(-1.0, 1.0, -1.0, 1.0)
        self.assertEqual(
            evaluator.polygon_penetration_depth_2d(point, square), 0.0
        )

    def test_interval_separation_is_not_the_overlap_length(self) -> None:
        # Nested intervals: overlap length 1.0, separation distance 2.0.
        self.assertAlmostEqual(
            evaluator.interval_separation_distance(0.0, 10.0, 1.0, 2.0), 2.0
        )
        # Partial overlap: the two agree.
        self.assertAlmostEqual(
            evaluator.interval_separation_distance(0.0, 1.0, 0.6, 1.6), 0.4
        )
        # Disjoint and merely touching both yield zero.
        self.assertEqual(
            evaluator.interval_separation_distance(0.0, 1.0, 2.0, 3.0), 0.0
        )
        self.assertEqual(
            evaluator.interval_separation_distance(0.0, 1.0, 1.0, 2.0), 0.0
        )

    def test_the_prism_depth_is_the_smaller_of_the_two_factors(self) -> None:
        left = rectangle(0.0, 1.0, 0.0, 1.0)
        right = rectangle(0.6, 1.6, 0.0, 1.0)
        self.assertAlmostEqual(
            evaluator.polygon_penetration_depth_2d(left, right), 0.4, places=12
        )
        # Vertical escape of 0.05 beats the lateral 0.4.
        depth, axis = evaluator.prism_penetration_depth(
            self.prism(left, 0.0, 1.0), self.prism(right, 0.95, 2.0)
        )
        self.assertAlmostEqual(depth, 0.05, places=12)
        self.assertEqual(axis, "vertical")
        # Fully overlapping in z, so the lateral escape wins.
        depth, axis = evaluator.prism_penetration_depth(
            self.prism(left, 0.0, 1.0), self.prism(right, 0.0, 1.0)
        )
        self.assertAlmostEqual(depth, 0.4, places=12)
        self.assertEqual(axis, "lateral")

    def test_no_vertical_overlap_means_no_penetration(self) -> None:
        square = rectangle(0.0, 1.0, 0.0, 1.0)
        depth, axis = evaluator.prism_penetration_depth(
            self.prism(square, 0.0, 1.0), self.prism(square, 1.0, 2.0)
        )
        self.assertEqual((depth, axis), (0.0, "disjoint"))


class SpearmanTest(unittest.TestCase):
    def test_identical_rankings_correlate_perfectly(self) -> None:
        self.assertAlmostEqual(
            fix93.spearman([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]), 1.0, places=12
        )

    def test_reversed_rankings_anticorrelate_perfectly(self) -> None:
        self.assertAlmostEqual(
            fix93.spearman([1.0, 2.0, 3.0], [0.3, 0.2, 0.1]), -1.0, places=12
        )

    def test_ties_use_midranks(self) -> None:
        self.assertAlmostEqual(
            fix93.spearman([1.0, 1.0, 2.0], [5.0, 5.0, 9.0]), 1.0, places=12
        )

    def test_a_constant_series_has_no_ranking_to_compare(self) -> None:
        self.assertIsNone(fix93.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))

    def test_too_few_points_returns_none(self) -> None:
        self.assertIsNone(fix93.spearman([1.0], [2.0]))


class NormalizationDisagreementTest(unittest.TestCase):
    """The two Smoke5 pairs that forced this audit, reproduced at unit level."""

    def setUp(self) -> None:
        self.object_ids = ["book_1", "book_2", "trash_bin_1", "paper_cup_1"]
        self.rows = [
            {
                # Nine cubic millimetres between two thin books.
                "first_id": "book_1",
                "second_id": "book_2",
                "intersection_volume_m3": "0.000009",
                "overlap_fraction": "0.5067",
                "penetration_depth_m": "0.0006",
            },
            {
                # Sixty-one litres between a bin and a cup.
                "first_id": "trash_bin_1",
                "second_id": "paper_cup_1",
                "intersection_volume_m3": "0.061128",
                "overlap_fraction": "0.0359",
                "penetration_depth_m": "0.31",
            },
        ]
        self.result = fix93.audit_normalizations(
            self.rows,
            self.object_ids,
            fraction_tolerance=0.05,
            depth_tolerance=0.01,
            volume_tolerance=0.001,
        )

    def test_fraction_punishes_the_books_and_spares_the_bin(self) -> None:
        # Under today's definition the two books score exactly zero, while the bin
        # and the cup keep most of their credit despite the far larger overlap.
        # Every family score is a linear ramp, so 0.0359 against a 0.05 tolerance
        # leaves 0.282 rather than a clean pass.
        self.assertAlmostEqual(self.result["scores"]["fraction"], 0.141, places=6)

    def test_depth_and_volume_reverse_that_verdict(self) -> None:
        # 0.6 mm sits well inside a 10 mm tolerance, so the books keep 0.94 each;
        # 0.31 m and 61 litres are far outside any tolerance, so the bin and the
        # cup lose everything.  Theordering is the opposite of the fraction's.
        self.assertAlmostEqual(self.result["scores"]["depth"], 0.47, places=6)
        self.assertAlmostEqual(self.result["scores"]["volume"], 0.4955, places=6)
        self.assertEqual(
            self.result["agreement"]["fraction_vs_depth"][
                "objects_with_flipped_verdict"
            ],
            4,
        )
        self.assertAlmostEqual(
            self.result["agreement"]["fraction_vs_depth"][
                "spearman_rank_correlation"
            ],
            -1.0,
            places=9,
        )

    def test_the_inversion_is_reported_with_its_ratio(self) -> None:
        inversion = self.result["absolute_scale_inversion"]
        self.assertTrue(inversion["ordering_is_inverted"])
        self.assertEqual(
            {
                inversion["smallest_fully_penalised_pair"]["first_id"],
                inversion["smallest_fully_penalised_pair"]["second_id"],
            },
            {"book_1", "book_2"},
        )
        self.assertEqual(
            {
                inversion["largest_unpenalised_pair"]["first_id"],
                inversion["largest_unpenalised_pair"]["second_id"],
            },
            {"trash_bin_1", "paper_cup_1"},
        )
        self.assertAlmostEqual(
            inversion["unpenalised_over_penalised_volume_ratio"],
            0.061128 / 0.000009,
            places=3,
        )

    def test_an_object_with_no_reported_pair_scores_one_under_every_rule(self) -> None:
        result = fix93.audit_normalizations(
            [],
            ["lonely_0", "lonely_1"],
            fraction_tolerance=0.05,
            depth_tolerance=0.01,
            volume_tolerance=0.001,
        )
        for name in fix93.NORMALIZATIONS:
            self.assertAlmostEqual(result["scores"][name], 1.0)
        self.assertIsNone(result["absolute_scale_inversion"])

    def test_the_denominator_is_the_object_count_under_every_rule(self) -> None:
        for name in fix93.NORMALIZATIONS:
            self.assertEqual(self.result["object_count"], len(self.object_ids))
            self.assertIsNotNone(self.result["scores"][name])

    def test_the_tool_states_that_it_replaces_no_score(self) -> None:
        interpretation = self.result["interpretation"]
        self.assertTrue(interpretation["fraction_is_not_scale_invariant"])
        self.assertTrue(
            interpretation["depth_is_exact_minimum_translation_for_prisms"]
        )
        self.assertTrue(interpretation["no_score_here_replaces_the_evaluator_score"])


class FractionDefinitionTest(unittest.TestCase):
    """Pin the cause: the denominator is the smaller object's own volume."""

    @staticmethod
    def evaluate(smaller_half, larger_half, offset):
        def bbox(centre, half):
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

        def entry(centre, half, **extra):
            matrix = np.eye(4)
            matrix[:3, 3] = centre
            return {
                "pose_matrix_for_blender": matrix.tolist(),
                "bbox": bbox(centre, half),
                "length": [2 * half[0], 2 * half[1], 2 * half[2]],
                **extra,
            }

        document = {
            "obj_info": {
                "floor_0": entry((0.0, 0.0, 0.0), (5.0, 5.0, 0.02)),
                "small_0": entry(
                    (offset, 0.0, smaller_half[2]), smaller_half, supported="floor_0"
                ),
                "large_0": entry(
                    (0.0, 0.0, larger_half[2]), larger_half, supported="floor_0"
                ),
            },
            "reference_obj": "floor_0",
        }
        args = argparse.Namespace(
            collision_volume_tolerance=1e-9,
            collision_fraction_tolerance=0.05,
            contact_tolerance=0.05,
            containment_tolerance=0.05,
            support_overlap_tolerance=0.9,
            plane_tolerance=0.05,
            plane_orientation_tolerance=15.0,
            boundary_tolerance=0.05,
            semantic_tolerance=0.25,
            critical_threshold=0.5,
            collision_pairs_csv="enabled",
        )
        metrics, _ = evaluator.evaluate_scene(document, document, args)
        return metrics["collision_pair_details"][0]

    def test_the_same_absolute_overlap_scores_differently_by_object_size(self) -> None:
        # Both probes push 1 cm into the same larger object across the same 0.1 m
        # by 0.1 m cross-section, so the absolute overlap is identical: 1 cm of
        # depth and 1e-4 m3 of volume.  Only the probe's own length differs, from
        # 0.1 m to 2.0 m, which changes its volume twentyfold and nothing else.
        short = self.evaluate(
            smaller_half=(0.05, 0.05, 0.05),
            larger_half=(0.5, 0.5, 0.4),
            offset=0.54,
        )
        long = self.evaluate(
            smaller_half=(1.0, 0.05, 0.05),
            larger_half=(0.5, 0.5, 0.4),
            offset=1.49,
        )
        for probe in (short, long):
            self.assertAlmostEqual(probe["penetration_depth_m"], 0.01, places=9)
            self.assertAlmostEqual(
                probe["intersection_volume_m3"], 1e-4, places=12
            )
        # Identical geometry of contact, twentyfold difference in the score input.
        self.assertAlmostEqual(short["overlap_fraction"], 0.1, places=9)
        self.assertAlmostEqual(long["overlap_fraction"], 0.005, places=9)
        self.assertAlmostEqual(
            short["overlap_fraction"] / long["overlap_fraction"], 20.0, places=6
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
