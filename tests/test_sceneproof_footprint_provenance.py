#!/usr/bin/env python3
"""Tests for the Fix88 footprint provenance audit and the repair path.

The Smoke5 observation these tests pin down: the floor's footprint convex hull had
a single vertex, which by the evaluator's own construction forces an infinite
containment error, a zero footprint overlap, a constant-zero boundary family, and
a contact gap inflated by the slab's half thickness.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval_physical_realizability as evaluator
import sceneproof_footprint_provenance_audit_fix88 as audit88


def pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> list[list[float]]:
    matrix = np.eye(4)
    matrix[:3, 3] = (x, y, z)
    return matrix.tolist()


def box_bbox(
    x: float, y: float, z: float, half_x: float, half_y: float, half_z: float
) -> list[list[float]]:
    return [
        [x + sx * half_x, y + sy * half_y, z + sz * half_z]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]


def collapsed_bbox(x: float, y: float, z: float) -> list[list[float]]:
    """Eight coincident corners, which is what a missing bbox degrades into."""
    return [[x, y, z] for _ in range(8)]


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
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class DegenerateFloorScene:
    """A floor whose snapshot geometry collapsed, with one chair resting on it.

    The floor slab is 0.04 m thick and the chair rests exactly on its top face, so
    a correct measurement reports a zero contact gap and full containment, while
    the collapsed snapshot must report a 0.02 m gap, infinite containment and zero
    overlap.
    """

    def __init__(self) -> None:
        self.floor_pose = pose(0.0, 0.0, 0.0)
        self.chair_pose = pose(0.0, 0.0, 0.22)
        # Slab centred on the origin, 0.04 m thick, so its top face is at +0.02.
        self.floor_layout = {
            "pose_matrix_for_blender": self.floor_pose,
            "bbox": box_bbox(0.0, 0.0, 0.0, 5.0, 5.0, 0.02),
            "length": [10.0, 10.0, 0.04],
            "retrieved_asset": "floor",
        }
        self.floor_snapshot = {
            "pose_matrix_for_blender": self.floor_pose,
            "bbox": collapsed_bbox(0.0, 0.0, 0.0),
            "length": [0.0, 0.0, 0.0],
            "retrieved_asset": "floor",
        }
        chair_bbox = box_bbox(0.0, 0.0, 0.22, 0.25, 0.25, 0.20)
        self.chair = {
            "pose_matrix_for_blender": self.chair_pose,
            "bbox": chair_bbox,
            "length": [0.5, 0.5, 0.4],
            "retrieved_asset": "chair",
            "supported": "floor_0",
        }

    def geometry(self) -> dict:
        return {
            "obj_info": {
                "floor_0": dict(self.floor_snapshot),
                "chair_0": dict(self.chair),
            },
            "reference_obj": "floor_0",
        }

    def placement(self) -> dict:
        return {
            "obj_info": {
                "floor_0": dict(self.floor_layout),
                "chair_0": dict(self.chair),
            },
            "reference_obj": "floor_0",
        }


class BuildGeometriesRepairTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = DegenerateFloorScene()

    def test_collapsed_snapshot_yields_a_single_vertex(self) -> None:
        geometries = evaluator.build_geometries(
            self.scene.geometry()["obj_info"], self.scene.placement()["obj_info"]
        )
        self.assertEqual(len(geometries["floor_0"].polygon), 1)
        self.assertEqual(
            evaluator.degenerate_footprint_object_ids(geometries), ["floor_0"]
        )

    def test_repair_recovers_the_footprint_from_the_layout(self) -> None:
        geometries = evaluator.build_geometries(
            self.scene.geometry()["obj_info"],
            self.scene.placement()["obj_info"],
            repair_degenerate_footprints=True,
        )
        self.assertGreaterEqual(len(geometries["floor_0"].polygon), 3)
        self.assertEqual(evaluator.degenerate_footprint_object_ids(geometries), [])
        self.assertAlmostEqual(geometries["floor_0"].z_max, 0.02)

    def test_repair_leaves_sound_objects_untouched(self) -> None:
        without = evaluator.build_geometries(
            self.scene.geometry()["obj_info"], self.scene.placement()["obj_info"]
        )
        with_repair = evaluator.build_geometries(
            self.scene.geometry()["obj_info"],
            self.scene.placement()["obj_info"],
            repair_degenerate_footprints=True,
        )
        np.testing.assert_allclose(
            without["chair_0"].world_corners, with_repair["chair_0"].world_corners
        )


class ArtefactMagnitudeTest(unittest.TestCase):
    """The four consequences claimed for a collapsed floor, each measured."""

    def setUp(self) -> None:
        self.scene = DegenerateFloorScene()
        self.source = self.scene.geometry()
        self.target = self.scene.placement()

    def score(self, repair: bool):
        metrics, rows = evaluator.evaluate_scene(
            self.source, self.target, evaluator_args(repair_degenerate_footprints=repair)
        )
        return metrics, {row["object_id"]: row for row in rows}

    def test_phantom_gap_equals_half_the_slab_thickness(self) -> None:
        _, rows = self.score(repair=False)
        self.assertAlmostEqual(rows["chair_0"]["support_contact_gap_m"], 0.02, places=9)
        _, repaired_rows = self.score(repair=True)
        self.assertAlmostEqual(
            repaired_rows["chair_0"]["support_contact_gap_m"], 0.0, places=9
        )

    def test_containment_is_infinite_and_overlap_is_zero(self) -> None:
        _, rows = self.score(repair=False)
        self.assertFalse(np.isfinite(rows["chair_0"]["support_containment_error_m"]))
        self.assertEqual(rows["chair_0"]["support_footprint_overlap_ratio"], 0.0)

    def test_support_term_is_capped_at_one_third_and_recovers_to_one(self) -> None:
        _, rows = self.score(repair=False)
        # gap summand 1 - 0.02/0.05 = 0.6, other two summands zero.
        self.assertAlmostEqual(rows["chair_0"]["support_term"], 0.2, places=9)
        _, repaired_rows = self.score(repair=True)
        self.assertAlmostEqual(repaired_rows["chair_0"]["support_term"], 1.0, places=9)

    def test_boundary_family_is_a_constant_zero_until_repaired(self) -> None:
        metrics, rows = self.score(repair=False)
        self.assertEqual(metrics["families"]["boundary"]["score"], 0.0)
        self.assertFalse(np.isfinite(rows["chair_0"]["boundary_error_m"]))
        repaired_metrics, repaired_rows = self.score(repair=True)
        self.assertEqual(repaired_metrics["families"]["boundary"]["score"], 1.0)
        self.assertAlmostEqual(repaired_rows["chair_0"]["boundary_error_m"], 0.0)

    def test_metrics_report_the_provenance_counts(self) -> None:
        metrics, _ = self.score(repair=False)
        self.assertEqual(metrics["degenerate_footprint_count"], 1)
        self.assertEqual(metrics["repaired_footprint_count"], 0)
        self.assertFalse(metrics["footprint_repair_enabled"])
        repaired_metrics, _ = self.score(repair=True)
        self.assertEqual(repaired_metrics["degenerate_footprint_count"], 0)
        self.assertEqual(repaired_metrics["repaired_footprint_count"], 1)
        self.assertEqual(
            repaired_metrics["repaired_footprint_object_ids"], ["floor_0"]
        )
        self.assertTrue(repaired_metrics["footprint_repair_enabled"])

    def test_family_denominators_are_unchanged_by_repair(self) -> None:
        metrics, _ = self.score(repair=False)
        repaired_metrics, _ = self.score(repair=True)
        for family in ("collision", "support", "plane", "boundary", "semantic"):
            self.assertEqual(
                metrics["families"][family]["n"],
                repaired_metrics["families"][family]["n"],
                family,
            )


class ProvenanceAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = DegenerateFloorScene()
        self.result = audit88.audit_scene(
            self.scene.geometry(), self.scene.placement()
        )

    def test_degenerate_object_is_classified_repairable(self) -> None:
        self.assertEqual(
            self.result["degenerate_footprint_object_ids"], ["floor_0"]
        )
        self.assertEqual(self.result["repairable_from_layout"], ["floor_0"])
        self.assertEqual(self.result["unrepairable_count"], 0)
        self.assertEqual(
            self.result["objects"]["floor_0"]["classification"],
            "repairable_from_layout",
        )

    def test_contaminated_children_are_listed(self) -> None:
        self.assertEqual(
            self.result["children_contaminated_by_degenerate_parent"], ["chair_0"]
        )
        self.assertEqual(self.result["objects"]["floor_0"]["declared_child_count"], 1)

    def test_boundary_family_is_flagged_dead(self) -> None:
        self.assertTrue(self.result["boundary_family_dead"])
        self.assertEqual(self.result["reference_object"], "floor_0")

    def test_phantom_gap_prediction_matches_half_the_thickness(self) -> None:
        hypothesis = {
            entry["parent_id"]: entry
            for entry in self.result["phantom_gap_hypothesis"]
        }
        self.assertAlmostEqual(
            hypothesis["floor_0"]["predicted_phantom_gap_m"], 0.02, places=9
        )
        self.assertAlmostEqual(
            hypothesis["floor_0"]["layout_z_extent_m"], 0.04, places=9
        )

    def test_unrepairable_when_the_layout_is_also_collapsed(self) -> None:
        placement = self.scene.placement()
        placement["obj_info"]["floor_0"] = dict(self.scene.floor_snapshot)
        result = audit88.audit_scene(self.scene.geometry(), placement)
        self.assertEqual(
            result["objects"]["floor_0"]["classification"],
            "unrepairable_needs_upstream_snapshot",
        )
        self.assertEqual(result["repairable_from_layout_count"], 0)

    def test_sound_scene_reports_nothing_degenerate(self) -> None:
        geometry = self.scene.geometry()
        geometry["obj_info"]["floor_0"] = dict(self.scene.floor_layout)
        result = audit88.audit_scene(geometry, self.scene.placement())
        self.assertEqual(result["degenerate_footprint_count"], 0)
        self.assertFalse(result["boundary_family_dead"])
        self.assertEqual(result["objects"]["floor_0"]["classification"], "sound")


class GeometryFieldsTest(unittest.TestCase):
    def test_collapsed_bbox_is_reported_as_one_unique_point(self) -> None:
        fields = audit88.geometry_fields(
            {
                "pose_matrix_for_blender": pose(),
                "bbox": collapsed_bbox(0.0, 0.0, 0.0),
                "length": [0.0, 0.0, 0.0],
            }
        )
        self.assertTrue(fields["bbox_usable"])
        self.assertEqual(fields["bbox_unique_points"], 1)
        self.assertFalse(fields["length_usable"])

    def test_sound_bbox_is_reported_as_eight_unique_points(self) -> None:
        fields = audit88.geometry_fields(
            {
                "pose_matrix_for_blender": pose(),
                "bbox": box_bbox(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                "length": [2.0, 2.0, 2.0],
            }
        )
        self.assertEqual(fields["bbox_unique_points"], 8)
        self.assertTrue(fields["length_usable"])

    def test_missing_info_is_reported_absent(self) -> None:
        self.assertFalse(audit88.geometry_fields(None)["present"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
