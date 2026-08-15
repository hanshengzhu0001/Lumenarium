#!/usr/bin/env python3
"""Tests for the gravity-settle eligibility screen (Fix86 rules, Fix87 correction).

Every case below is anchored on a real Smoke5 observation, so a regression in the
screen shows up as a failing assertion rather than as another wasted simulation
run.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sceneproof_settle_eligibility_screen_fix86 as screen86

CONTACT_TOL = 0.05
CONTAINMENT_TOL = 0.05
OVERLAP_TOL = 0.9


def row(
    object_id: str,
    *,
    support_term: object = "",
    plane_term: object = "",
    inside_containment_error_m: object = "",
    support_contact_gap_m: object = "",
    support_containment_error_m: object = "",
    support_footprint_overlap_ratio: object = "",
) -> dict[str, object]:
    return {
        "object_id": object_id,
        "support_term": support_term,
        "plane_term": plane_term,
        "inside_containment_error_m": inside_containment_error_m,
        "support_contact_gap_m": support_contact_gap_m,
        "support_containment_error_m": support_containment_error_m,
        "support_footprint_overlap_ratio": support_footprint_overlap_ratio,
    }


def footprint(vertices: int, area: float = 1.0) -> dict[str, object]:
    return {
        "footprint_vertex_count": vertices,
        "footprint_area_m2": area,
        "footprint_degenerate": vertices < 3,
        "z_min_m": 0.0,
        "z_max_m": 1.0,
    }


class ObservedTermDecodeTest(unittest.TestCase):
    """The screen must decode the evaluator's term exactly, not approximately."""

    def test_streelitter_litter_term_is_reproduced(self) -> None:
        # Observed on streelitter_01: gap 0.02, containment inf, overlap 0.0,
        # support_term 0.2000.
        gap_part = screen86.linear_score(0.02, CONTACT_TOL)
        containment_part = screen86.linear_score(float("inf"), CONTAINMENT_TOL)
        overlap_part = min(1.0, 0.0 / OVERLAP_TOL)
        self.assertAlmostEqual(gap_part, 0.6)
        self.assertEqual(containment_part, 0.0)
        self.assertEqual(overlap_part, 0.0)
        self.assertAlmostEqual((gap_part + containment_part + overlap_part) / 3, 0.2)

    def test_wooden_board_12_term_is_reproduced(self) -> None:
        # Observed: gap 3.7e-9, containment 0.7012, overlap 0.0, term 0.3333.
        # The gap summand is 1 - 3.7e-9/0.05 = 1 - 7.45e-8, i.e. saturated for
        # every practical purpose: gravity has nothing left to recover here.
        gap_part = screen86.linear_score(3.725289784983765e-09, CONTACT_TOL)
        containment_part = screen86.linear_score(0.7011597634463567, CONTAINMENT_TOL)
        overlap_part = min(1.0, 0.0 / OVERLAP_TOL)
        self.assertAlmostEqual(gap_part, 1.0, places=6)
        self.assertLess(1.0 - gap_part, 1e-7)
        self.assertEqual(containment_part, 0.0)
        self.assertAlmostEqual(
            (gap_part + containment_part + overlap_part) / 3, 1 / 3, places=7
        )


class AttainableGainTest(unittest.TestCase):
    def test_gain_is_the_closed_form(self) -> None:
        for gap in (0.0, 0.001, 0.02, 0.05, 0.4):
            expected = min(1.0, gap / CONTACT_TOL) / 3.0
            self.assertAlmostEqual(
                screen86.attainable_settle_gain(gap, CONTACT_TOL), expected
            )

    def test_saturates_beyond_the_tolerance(self) -> None:
        self.assertAlmostEqual(
            screen86.attainable_settle_gain(10.0, CONTACT_TOL), 1 / 3
        )

    def test_contacting_object_has_no_gain(self) -> None:
        self.assertLess(
            screen86.attainable_settle_gain(3.7e-09, CONTACT_TOL), 1e-7
        )

    def test_non_finite_gap_has_no_gain(self) -> None:
        self.assertEqual(
            screen86.attainable_settle_gain(float("inf"), CONTACT_TOL), 0.0
        )
        self.assertEqual(screen86.attainable_settle_gain(None, CONTACT_TOL), 0.0)


class ScreenRuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.placement = {
            "obj_info": {
                "ground_1": {},
                "wall_1": {},
                "thin_rail_1": {"supported": "ground_1"},
                "table_1": {"supported": "ground_1"},
                "trash_bag_1": {"supported": "thin_rail_1"},
                "board_12": {"supported": "thin_rail_1"},
                "cup_1": {"supported": "table_1"},
                "picture_1": {"supported": "wall_1", "SpatialRel": "against"},
                "book_1": {"supported": "shelf_1", "SpatialRel": "inside"},
                "shelf_1": {"supported": "ground_1"},
                "orphan_1": {"supported": "ghost_9"},
            }
        }
        self.footprints = {
            "ground_1": footprint(4, 100.0),
            "wall_1": footprint(4, 2.0),
            # A rail whose XY projection collapsed to a line: the only condition
            # under which the evaluator's containment error is infinite.
            "thin_rail_1": footprint(2, 0.0),
            "table_1": footprint(4, 1.2),
            "shelf_1": footprint(4, 0.8),
        }
        self.rows = {
            # Floating 2 cm above a degenerate-footprint parent: gravity can
            # recover the gap summand only.
            "trash_bag_1": row(
                "trash_bag_1",
                support_term=0.2,
                support_contact_gap_m=0.02,
                support_containment_error_m=float("inf"),
                support_footprint_overlap_ratio=0.0,
            ),
            # Already in exact contact: nothing for gravity to do.
            "board_12": row(
                "board_12",
                support_term=1 / 3,
                support_contact_gap_m=3.725289784983765e-09,
                support_containment_error_m=0.7011597634463567,
                support_footprint_overlap_ratio=0.0,
            ),
            # Genuinely floating above a sound parent.
            "cup_1": row(
                "cup_1",
                support_term=0.6,
                support_contact_gap_m=0.08,
                support_containment_error_m=0.0,
                support_footprint_overlap_ratio=1.0,
            ),
            "picture_1": row("picture_1", plane_term=0.9),
            "book_1": row(
                "book_1", support_term=0.5, inside_containment_error_m=0.02
            ),
            "orphan_1": row("orphan_1", support_term=0.0),
            "table_1": row(
                "table_1",
                support_term=0.9,
                support_contact_gap_m=0.03,
                support_containment_error_m=0.0,
                support_footprint_overlap_ratio=1.0,
            ),
        }

    def run_screen(self, **overrides):
        kwargs = {
            "contact_tolerance": CONTACT_TOL,
            "containment_tolerance": CONTAINMENT_TOL,
            "overlap_tolerance": OVERLAP_TOL,
            "minimum_gap_m": 0.002,
            "minimum_attainable_gain": 0.01,
        }
        kwargs.update(overrides)
        return screen86.screen(
            self.placement, self.rows, self.footprints, **kwargs
        )

    def test_already_contacting_object_is_excluded(self) -> None:
        result = self.run_screen()
        self.assertIn("board_12", result["excluded"]["already_in_contact"])
        self.assertNotIn(
            "board_12", [entry["object_id"] for entry in result["eligible"]]
        )

    def test_floating_objects_are_eligible_largest_gain_first(self) -> None:
        result = self.run_screen()
        self.assertEqual(
            [entry["object_id"] for entry in result["eligible"]],
            ["cup_1", "trash_bag_1"],
        )
        gains = [entry["attainable_settle_gain"] for entry in result["eligible"]]
        self.assertEqual(gains, sorted(gains, reverse=True))

    def test_degenerate_parent_is_reported_with_its_ceiling(self) -> None:
        result = self.run_screen()
        by_id = {entry["object_id"]: entry for entry in result["eligible"]}
        bag = by_id["trash_bag_1"]
        self.assertTrue(bag["parent_footprint_degenerate"])
        self.assertAlmostEqual(bag["support_term_ceiling"], 1 / 3)
        self.assertEqual(bag["parent_footprint_vertex_count"], 2)
        self.assertAlmostEqual(bag["attainable_settle_gain"], 0.4 / 3)
        self.assertGreaterEqual(result["parent_footprint_degenerate_count"], 1)

    def test_sound_parent_has_no_ceiling(self) -> None:
        result = self.run_screen()
        by_id = {entry["object_id"]: entry for entry in result["eligible"]}
        self.assertAlmostEqual(by_id["cup_1"]["support_term_ceiling"], 1.0)
        self.assertFalse(by_id["cup_1"]["parent_footprint_degenerate"])

    def test_semantic_exclusions_still_hold(self) -> None:
        result = self.run_screen()
        self.assertIn("picture_1", result["excluded"]["no_support_term"])
        self.assertIn("book_1", result["excluded"]["held_by_containment"])
        self.assertIn("orphan_1", result["excluded"]["missing_support_parent"])
        self.assertIn("table_1", result["excluded"]["is_support_parent"])

    def test_gain_floor_filters_marginal_targets(self) -> None:
        # cup_1 floats 8 cm above its parent, past the 5 cm contact tolerance, so
        # its attainable gain saturates at 1/3 and survives a0.2 floor.  The
        # 2 cm litter gap yields only 0.133 and is filtered out.
        result = self.run_screen(minimum_attainable_gain=0.2)
        self.assertEqual(
            [entry["object_id"] for entry in result["eligible"]], ["cup_1"]
        )
        self.assertIn(
            "trash_bag_1", result["excluded"]["attainable_gain_below_floor"]
        )

    def test_floor_above_saturation_filters_everything(self) -> None:
        result = self.run_screen(minimum_attainable_gain=0.4)
        self.assertEqual(result["eligible"], [])
        for object_id in ("cup_1", "trash_bag_1"):
            self.assertIn(
                object_id, result["excluded"]["attainable_gain_below_floor"]
            )

    def test_minimum_gap_filters_numerically_zero_gaps(self) -> None:
        result = self.run_screen(minimum_gap_m=0.05)
        self.assertEqual(
            [entry["object_id"] for entry in result["eligible"]], ["cup_1"]
        )
        self.assertIn("trash_bag_1", result["excluded"]["already_in_contact"])


class DeficitAttributionTest(unittest.TestCase):
    def test_components_sum_to_the_total_for_resting_support(self) -> None:
        placement = {
            "obj_info": {
                "ground_1": {},
                "a_1": {"supported": "ground_1"},
                "b_1": {"supported": "ground_1"},
            }
        }
        footprints = {"ground_1": footprint(4, 50.0)}
        rows = {
            "a_1": row(
                "a_1",
                support_term=0.2,
                support_contact_gap_m=0.02,
                support_containment_error_m=float("inf"),
                support_footprint_overlap_ratio=0.0,
            ),
            "b_1": row(
                "b_1",
                support_term=1.0,
                support_contact_gap_m=0.0,
                support_containment_error_m=0.0,
                support_footprint_overlap_ratio=1.0,
            ),
        }
        result = screen86.screen(
            placement,
            rows,
            footprints,
            contact_tolerance=CONTACT_TOL,
            containment_tolerance=CONTAINMENT_TOL,
            overlap_tolerance=OVERLAP_TOL,
            minimum_gap_m=0.002,
            minimum_attainable_gain=0.01,
        )
        deficit = result["support_deficit"]
        self.assertAlmostEqual(deficit["total"], (1 - 0.2) / 2)
        reassembled = (
            deficit["contact_gap_component"]
            + deficit["containment_component"]
            + deficit["footprint_overlap_component"]
        )
        self.assertAlmostEqual(reassembled, deficit["total"])

    def test_recoverable_never_exceeds_the_gap_component(self) -> None:
        placement = {
            "obj_info": {"ground_1": {}, "a_1": {"supported": "ground_1"}}
        }
        footprints = {"ground_1": footprint(4, 50.0)}
        rows = {
            "a_1": row(
                "a_1",
                support_term=0.2,
                support_contact_gap_m=0.02,
                support_containment_error_m=float("inf"),
                support_footprint_overlap_ratio=0.0,
            )
        }
        result = screen86.screen(
            placement,
            rows,
            footprints,
            contact_tolerance=CONTACT_TOL,
            containment_tolerance=CONTAINMENT_TOL,
            overlap_tolerance=OVERLAP_TOL,
            minimum_gap_m=0.002,
            minimum_attainable_gain=0.01,
        )
        deficit = result["support_deficit"]
        self.assertLessEqual(
            deficit["recoverable_by_settling"] - 1e-12,
            deficit["contact_gap_component"],
        )


class HelperTest(unittest.TestCase):
    def test_infinity_survives_csv_parsing(self) -> None:
        self.assertTrue(math.isinf(screen86.optional_float("inf")))
        self.assertTrue(math.isinf(screen86.optional_float("Infinity")))
        self.assertIsNone(screen86.optional_float(""))
        self.assertIsNone(screen86.optional_float("none"))

    def test_list_valued_parent_resolves_to_first_entry(self) -> None:
        self.assertEqual(screen86.support_id(["", "table_1"]), "table_1")
        self.assertIsNone(screen86.support_id("   "))

    def test_structural_names(self) -> None:
        for name in ("floor_1", "ground_2", "wall_12", "ceiling_0", "rug_2"):
            self.assertTrue(screen86.structural(name), name)
        for name in ("table_1", "wall_lamp_1", "floor_lamp_0"):
            self.assertFalse(screen86.structural(name), name)


class TermCsvTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).resolve().parent / "_fix86_objects_tmp.csv"

    def tearDown(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def write(self, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_only_the_requested_scene_and_version_are_loaded(self) -> None:
        self.write(
            [
                {"version": "base", "scene": "s", "object_id": "a", "support_term": 0.1},
                {"version": "cand", "scene": "s", "object_id": "b", "support_term": 0.2},
                {"version": "base", "scene": "t", "object_id": "c", "support_term": 0.3},
            ],
            ["version", "scene", "object_id", "support_term"],
        )
        rows = screen86.load_baseline_terms(self.path, "s", "base")
        self.assertEqual(sorted(rows), ["a"])

    def test_missing_term_column_is_rejected(self) -> None:
        self.write(
            [{"version": "base", "scene": "s", "object_id": "a"}],
            ["version", "scene", "object_id"],
        )
        with self.assertRaises(SystemExit):
            screen86.load_baseline_terms(self.path, "s", "base")


if __name__ == "__main__":
    unittest.main(verbosity=2)
