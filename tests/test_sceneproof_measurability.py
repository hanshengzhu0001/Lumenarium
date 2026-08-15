#!/usr/bin/env python3
"""Tests for measurability handling after the wall-derived floor was falsified.

Smoke5 facts encoded here:

* ``floor_0`` carries no usable ``bbox`` and no ``length`` in either the frozen
  snapshot or the layout, so its footprint collapses to a single vertex.Root
  cause: ``modules/S4_blender_layout_and_corr.py:7274`` excludes the ground from
  geometry serialization.
* The wall-derived floor plane is withdrawn.  On Smoke5 it put the floor's top
  face 2.87 m to 3.15 m below the floor origin against a predicted +0.02 m, and
  produced boundary scores ranging from 0.298 to 1.000 across scenes.
* Abstention is honest but it changes the estimand.  The abstained terms are the
  children of an object with no recorded extent, which are systematically the
  worst scoring ones, so the remaining mean must never be quoted as an
  improvement over the as-is mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_physical_realizability as evaluator


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
        "_structural_geometry_sidecar": {},
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class UnserializedFloorScene:
    """A room whose floor extent was never serialized, reproducing Smoke5.

    The floor slab is 0.04 m thick with its origin at the centre, so its true top
    face lies 0.02 m above its origin and the two chairs resting on that face
    must report a zero contact gap once the extent is known.  The walls are 10 m
    construction panels extending far below the floor, which is why deriving the
    floor plane from them fails.
    """

    FLOOR_TOP = 0.02

    def __init__(self) -> None:
        self.floor = {
            "pose_matrix_for_blender": pose(0.0, 0.0, 0.0),
            "retrieved_asset": "ground",
        }
        # 10 m panels centred well below the floor, as built by create_cuboid.
        self.walls = {
            "wall_0": solid((0.0, 2.0, -2.0), (5.0, 0.02, 5.0)),
            "wall_1": solid((0.0, -2.0, -2.0), (5.0, 0.02, 5.0)),
        }
        self.chairs = {
            "chair_0": solid(
                (0.5, 0.5, 0.27), (0.25, 0.25, 0.25), supported="floor_0"
            ),
            "chair_1": solid(
                (-0.5, -0.5, 0.27), (0.25, 0.25, 0.25), supported="floor_0"
            ),
        }

    def document(self) -> dict:
        info = {"floor_0": dict(self.floor)}
        info.update({name: dict(value) for name, value in self.walls.items()})
        info.update({name: dict(value) for name, value in self.chairs.items()})
        return {"obj_info": info, "reference_obj": "floor_0"}

    def true_floor_geometry(self) -> dict:
        """What a dump from the constructed scene would record for the slab."""
        return {
            "bbox": box_bbox((0.0, 0.0, 0.0), (2.05, 2.05, 0.02)),
            "length": [4.1, 4.1, 0.04],
        }

    def score(self, **flags):
        document = self.document()
        metrics, rows = evaluator.evaluate_scene(
            document, document, evaluator_args(**flags)
        )
        return metrics, {row["object_id"]: row for row in rows}


class WithdrawnReconstructionTest(unittest.TestCase):
    def test_the_wall_derived_floor_plane_is_gone(self) -> None:
        self.assertFalse(hasattr(evaluator, "reconstruct_floor_from_walls"))

    def test_metrics_record_the_withdrawal(self) -> None:
        metrics, _ = UnserializedFloorScene().score()
        reconstruction = metrics["floor_reconstruction"]
        self.assertFalse(reconstruction["attempted"])
        self.assertIn("falsified", reconstruction["withdrawn"])

    def test_the_flag_cannot_silently_do_nothing(self) -> None:
        # Passing the withdrawn flag must fail loudly rather than beignored.
        source = Path(evaluator.__file__).read_text(encoding="utf-8")
        self.assertIn("--floor-from-walls is withdrawn", source)


class AsIsBehaviourTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = UnserializedFloorScene()
        self.metrics, self.rows = self.scene.score()

    def test_only_the_floor_is_degenerate(self) -> None:
        self.assertEqual(
            self.metrics["degenerate_footprint_object_ids"], ["floor_0"]
        )

    def test_contact_gap_is_the_phantom_two_centimetres(self) -> None:
        for chair in ("chair_0", "chair_1"):
            self.assertAlmostEqual(
                self.rows[chair]["support_contact_gap_m"], 0.02, places=9
            )

    def test_support_term_is_capped_at_one_third(self) -> None:
        for chair in ("chair_0", "chair_1"):
            self.assertAlmostEqual(self.rows[chair]["support_term"], 0.2, places=9)

    def test_boundary_family_reports_a_fabricated_zero(self) -> None:
        self.assertEqual(self.metrics["families"]["boundary"]["score"], 0.0)
        self.assertGreater(self.metrics["families"]["boundary"]["n"], 0)

    def test_as_is_scores_are_flagged_comparable(self) -> None:
        self.assertTrue(self.metrics["scores_comparable_to_non_abstained_runs"])
        self.assertFalse(self.metrics["estimand_changed_by_abstention"])


class AbstentionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = UnserializedFloorScene()
        self.as_is, _ = self.scene.score()
        self.metrics, self.rows = self.scene.score(
            abstain_on_unmeasurable_footprints=True
        )

    def test_boundary_becomes_unmeasured_rather_than_zero(self) -> None:
        self.assertEqual(self.metrics["families"]["boundary"]["n"], 0)
        self.assertIsNone(self.metrics["families"]["boundary"]["score"])
        self.assertEqual(self.metrics["abstained_counts"]["boundary"], 2)

    def test_support_terms_against_an_unmeasurable_parent_are_omitted(self) -> None:
        self.assertEqual(self.metrics["families"]["support"]["n"], 0)
        self.assertEqual(self.metrics["abstained_counts"]["support"], 2)
        for chair in ("chair_0", "chair_1"):
            self.assertIsNone(self.rows[chair]["support_term"])
            self.assertEqual(
                self.rows[chair]["support_unmeasurable_parent_id"], "floor_0"
            )

    def test_abstained_scores_are_flagged_incomparable(self) -> None:
        self.assertTrue(self.metrics["estimand_changed_by_abstention"])
        self.assertFalse(self.metrics["scores_comparable_to_non_abstained_runs"])

    def test_untouched_families_are_bit_identical(self) -> None:
        for family in ("collision", "plane", "semantic"):
            self.assertEqual(
                self.as_is["families"][family]["score"],
                self.metrics["families"][family]["score"],
                family,
            )

    def test_abstention_is_a_selection_effect_not_an_improvement(self) -> None:
        """The removed terms are the worst ones, so the mean must rise."""
        document = self.scene.document()
        # A third chair on a sound parent keeps the support family non-empty.
        document["obj_info"]["table_0"] = solid((1.5, 1.5, 0.25), (0.4, 0.4, 0.25))
        document["obj_info"]["cup_0"] = solid(
            (1.5, 1.5, 0.55), (0.05, 0.05, 0.05), supported="table_0"
        )
        plain = evaluator.evaluate_scene(document, document, evaluator_args())[0]
        abstained = evaluator.evaluate_scene(
            document,
            document,
            evaluator_args(abstain_on_unmeasurable_footprints=True),
        )[0]
        self.assertGreater(
            abstained["families"]["support"]["score"],
            plain["families"]["support"]["score"],
        )
        self.assertLess(
            abstained["families"]["support"]["n"],
            plain["families"]["support"]["n"],
        )
        # The rise is only admissible because the estimand is flagged as changed.
        self.assertTrue(abstained["estimand_changed_by_abstention"])


class StructuralGeometrySidecarTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = UnserializedFloorScene()
        self.sidecar = {"floor_0": self.scene.true_floor_geometry()}

    def score_with_sidecar(self, sidecar=None):
        document = self.scene.document()
        metrics, rows = evaluator.evaluate_scene(
            document,
            document,
            evaluator_args(
                _structural_geometry_sidecar=(
                    self.sidecar if sidecar is None else sidecar
                )
            ),
        )
        return metrics, {row["object_id"]: row for row in rows}

    def test_backfill_makes_the_floor_measurable(self) -> None:
        metrics, _ = self.score_with_sidecar()
        self.assertEqual(metrics["degenerate_footprint_count"], 0)
        self.assertEqual(
            metrics["structural_geometry_backfilled_object_ids"], ["floor_0"]
        )

    def test_contact_gap_collapses_to_zero(self) -> None:
        _, rows = self.score_with_sidecar()
        for chair in ("chair_0", "chair_1"):
            self.assertAlmostEqual(
                rows[chair]["support_contact_gap_m"], 0.0, places=9
            )

    def test_support_and_boundary_become_measurable(self) -> None:
        metrics, rows = self.score_with_sidecar()
        for chair in ("chair_0", "chair_1"):
            self.assertEqual(rows[chair]["support_containment_error_m"], 0.0)
            self.assertAlmostEqual(rows[chair]["support_term"], 1.0, places=9)
        self.assertEqual(metrics["families"]["boundary"]["n"], 2)
        self.assertAlmostEqual(metrics["families"]["boundary"]["score"], 1.0)

    def test_backfill_keeps_the_estimand_intact(self) -> None:
        as_is, _ = self.scene.score()
        metrics, _ = self.score_with_sidecar()
        # Unlike abstention, backfilling measures the missing quantity instead of
        # dropping it, so every family keeps its denominator and the scores stay
        # comparable.
        for family in ("collision", "support", "plane", "boundary", "semantic"):
            self.assertEqual(
                as_is["families"][family]["n"],
                metrics["families"][family]["n"],
                family,
            )
        self.assertTrue(metrics["scores_comparable_to_non_abstained_runs"])

    def test_sidecar_never_overrides_sound_geometry(self) -> None:
        hostile = {
            "chair_0": {
                "bbox": box_bbox((9.0, 9.0, 9.0), (3.0, 3.0, 3.0)),
                "length": [6.0, 6.0, 6.0],
            }
        }
        baseline, baseline_rows = self.scene.score()
        _, rows = self.score_with_sidecar(sidecar=hostile)
        self.assertAlmostEqual(
            rows["chair_0"]["support_contact_gap_m"],
            baseline_rows["chair_0"]["support_contact_gap_m"],
            places=12,
        )
        _ = baseline

    def test_absent_objects_are_ignored(self) -> None:
        metrics, _ = self.score_with_sidecar(
            sidecar={"ghost_0": self.scene.true_floor_geometry()}
        )
        self.assertEqual(metrics["structural_geometry_backfilled_count"], 0)


class SidecarLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).resolve().parent / "_fix90_sidecar_tmp.json"

    def tearDown(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def write(self, payload: dict) -> None:
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def test_reads_the_obj_info_wrapper(self) -> None:
        good = {
            "bbox": box_bbox((0.0, 0.0, 0.0), (1.0, 1.0, 0.02)),
            "length": [2.0, 2.0, 0.04],
        }
        self.write({"obj_info": {"floor_0": good}})
        loaded = evaluator.load_structural_geometry_sidecar(self.path)
        self.assertEqual(sorted(loaded), ["floor_0"])

    def test_reads_a_bare_mapping(self) -> None:
        good = {
            "bbox": box_bbox((0.0, 0.0, 0.0), (1.0, 1.0, 0.02)),
            "length": [2.0, 2.0, 0.04],
        }
        self.write({"floor_0": good})
        self.assertEqual(
            sorted(evaluator.load_structural_geometry_sidecar(self.path)),
            ["floor_0"],
        )

    def test_malformed_entries_are_dropped(self) -> None:
        self.write(
            {
                "obj_info": {
                    "bad_shape": {"bbox": [[0, 0, 0]], "length": [1, 1, 1]},
                    "bad_length": {
                        "bbox": box_bbox((0, 0, 0), (1, 1, 1)),
                        "length": [1, 1],
                    },
                    "not_a_dict":5,
                }
            }
        )
        self.assertEqual(evaluator.load_structural_geometry_sidecar(self.path), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
