#!/usr/bin/env python3
"""Tests for Fix94: selecting the objects that visibly hang off their support.

The selection reads measurements that are already on disk, so these tests pin the
decision logic rather than any physics: which objects are flagged, which are
correctly left alone, and whether a horizontal nudge is feasible or the object has
to be tipped.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sceneproof_overhang_screen_fix94 as fix94


def bbox(centre, half) -> list[list[float]]:
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


def entry(centre, half, **extra) -> dict:
    return {
        "bbox": bbox(centre, half),
        "length": [2 * half[0], 2 * half[1], 2 * half[2]],
        **extra,
    }


def record(
    *,
    margin,
    com,
    polygon,
    status="measured",
    certificate="certified",
    stability="unstable",
    area=0.5,
) -> dict:
    return {
        "status": status,
        "certificate_status": certificate,
        "stability_class": stability,
        "com_signed_margin_m": margin,
        "center_of_mass_world_m": list(com),
        "support_polygon_world_xy_m": [list(point) for point in polygon],
        "support_polygon_area_m2": area,
    }


def square(x_min, x_max, y_min, y_max) -> list[tuple[float, float]]:
    return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]


class GeometryHelperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.polygon = np.asarray(square(0.0, 1.0, 0.0, 1.0), dtype=np.float64)

    def test_an_interior_point_is_its_own_closest_point(self) -> None:
        point = np.asarray([0.5, 0.5])
        result = fix94.closest_point_in_convex_polygon(point, self.polygon)
        np.testing.assert_allclose(result, point)

    def test_an_exterior_point_projects_onto_the_nearest_edge(self) -> None:
        result = fix94.closest_point_in_convex_polygon(
            np.asarray([1.3, 0.5]), self.polygon
        )
        np.testing.assert_allclose(result, [1.0, 0.5], atol=1e-12)

    def test_a_point_beyond_a_corner_projects_onto_the_corner(self) -> None:
        result = fix94.closest_point_in_convex_polygon(
            np.asarray([2.0, 2.0]), self.polygon
        )
        np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-12)

    def test_a_degenerate_polygon_has_no_projection(self) -> None:
        self.assertIsNone(
            fix94.closest_point_in_convex_polygon(
                np.asarray([1.0, 1.0]), np.asarray([[0.0, 0.0]])
            )
        )


class PillowOnABedEdgeTest(unittest.TestCase):
    """A pillow whose COM sits 6 cm past the mattress edge, 0.6 m off the floor."""

    def setUp(self) -> None:
        self.placement = {
            "obj_info": {
                "floor_0": entry((0.0, 0.0, 0.0), (5.0, 5.0, 0.02)),
                "double_bed_0": entry(
                    (0.0, 0.0, 0.3), (1.0, 1.0, 0.3), supported="floor_0"
                ),
                "pillow_0": entry(
                    (1.06, 0.0, 0.65), (0.2, 0.15, 0.05), supported="double_bed_0"
                ),
            },
            "reference_obj": "floor_0",
        }
        self.com_audit = {
            "objects": {
                "pillow_0": record(
                    margin=-0.06,
                    com=(1.06, 0.0, 0.65),
                    polygon=square(0.86, 1.0, -0.15, 0.15),
                )
            }
        }

    def screen(self, **overrides):
        settings = dict(
            margin_threshold_m=0.005,
            translate_budget_m=0.15,
            new_overlap_tolerance_m3=1e-6,
            elevation_saturation_m=0.5,
            top_k=3,
        )
        settings.update(overrides)
        return fix94.screen_scene(self.com_audit, self.placement, **settings)

    def test_the_pillow_is_flagged(self) -> None:
        result = self.screen()
        self.assertEqual(result["candidate_count"], 1)
        entry_ = result["selected"][0]
        self.assertEqual(entry_["object_id"], "pillow_0")
        self.assertAlmostEqual(entry_["overhang_distance_m"], 0.06)
        self.assertAlmostEqual(entry_["elevation_m"], 0.6)

    def test_a_6_cm_overhang_is_repaired_by_a_6_cm_nudge(self) -> None:
        entry_ = self.screen()["selected"][0]
        self.assertEqual(entry_["recommended_action"], "translate")
        self.assertTrue(entry_["translate_feasible"])
        np.testing.assert_allclose(
            entry_["proposed_translation_xy_m"], [-0.06, 0.0], atol=1e-12
        )

    def test_salience_saturates_with_height_so_it_stays_a_ranking_only(self) -> None:
        # 0.6 m is above the 0.5 m saturation point, so salience is the raw
        # overhang; a pillow at 0.25 m would rank at half that.
        self.assertAlmostEqual(self.screen()["selected"][0]["visual_salience"], 0.06)
        low = {
            "objects": {
                "pillow_0": record(
                    margin=-0.06,
                    com=(1.06, 0.0, 0.2),
                    polygon=square(0.86, 1.0, -0.15, 0.15),
                )
            }
        }
        placement = {
            "obj_info": {
                **self.placement["obj_info"],
                "pillow_0": entry(
                    (1.06, 0.0, 0.25), (0.2, 0.15, 0.05), supported="double_bed_0"
                ),
            },
            "reference_obj": "floor_0",
        }
        result = fix94.screen_scene(
            low,
            placement,
            margin_threshold_m=0.005,
            translate_budget_m=0.15,
            new_overlap_tolerance_m3=1e-6,
            elevation_saturation_m=0.5,
            top_k=3,
        )
        self.assertAlmostEqual(result["selected"][0]["visual_salience"], 0.024)

    def test_a_travel_beyond_budget_is_routed_to_tipping(self) -> None:
        entry_ = self.screen(translate_budget_m=0.05)["selected"][0]
        self.assertEqual(entry_["recommended_action"], "tip")
        self.assertIn("travel_exceeds_budget", entry_["translate_blockers"])

    def test_a_nudge_that_would_hit_something_is_routed_to_tipping(self) -> None:
        # A suitcase parked just inboard of the pillow: clear of it now, but the
        # 6 cm nudge would drive the pillow into it.
        self.placement["obj_info"]["suitcase_0"] = entry(
            (0.81, 0.0, 0.65), (0.03, 0.15, 0.05), supported="double_bed_0"
        )
        entry_ = self.screen()["selected"][0]
        self.assertEqual(entry_["recommended_action"], "tip")
        self.assertIn("new_overlap", entry_["translate_blockers"])
        self.assertEqual(entry_["new_overlap_object_ids"], ["suitcase_0"])

    def test_an_overlap_that_already_exists_does_not_veto_the_nudge(self) -> None:
        # The suitcase sits inside the pillow's current footprint span, so the
        # nudge leaves the overlap unchanged.  That is the collision family's
        # problem, not a reason to refuse a repair that makes nothing worse.
        self.placement["obj_info"]["suitcase_0"] = entry(
            (1.0, 0.0, 0.65), (0.05, 0.15, 0.05), supported="double_bed_0"
        )
        entry_ = self.screen()["selected"][0]
        self.assertEqual(entry_["recommended_action"], "translate")
        self.assertEqual(entry_["new_overlap_object_ids"], [])

    def test_an_overhang_below_the_visibility_threshold_is_ignored(self) -> None:
        self.com_audit["objects"]["pillow_0"]["com_signed_margin_m"] = -0.001
        result = self.screen()
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["excluded_counts"]["com_inside_support_polygon"], 1)

    def test_no_pose_is_touched(self) -> None:
        before = str(self.placement)
        self.screen()
        self.assertEqual(str(self.placement), before)
        self.assertTrue(self.screen()["policy"]["no_pose_is_modified_by_this_tool"])

    def test_missing_contact_uses_true_parent_surface_as_candidate_witness(self):
        self.com_audit["objects"]["pillow_0"] = {
            "status": "abstained",
            "reason": "no_mesh_or_voxel_horizontal_contact_patch",
            "center_of_mass_world_m": [1.06, 0.0, 0.65],
            "mesh_volume_m3": 0.01,
            "declared_parent_contact_present": False,
            "declared_parent_surface_margin_m": -0.06,
            "declared_parent_surface_polygon_world_xy_m": square(
                -1.0, 1.0, -1.0, 1.0
            ),
        }
        result = self.screen()
        candidate = result["selected"][0]
        self.assertEqual(
            candidate["support_witness_mode"],
            "declared_parent_surface_without_current_contact",
        )
        self.assertTrue(candidate["post_projection_contact_must_be_certified"])


class ExclusionTest(unittest.TestCase):
    def base(self, **info_overrides):
        info = {
            "floor_0": entry((0.0, 0.0, 0.0), (5.0, 5.0, 0.02)),
            "shelf_0": entry((0.0, 0.0, 0.5), (0.5, 0.3, 0.5), supported="floor_0"),
            "thing_0": entry(
                (0.6, 0.0, 1.05), (0.1, 0.1, 0.05), supported="shelf_0"
            ),
        }
        info["thing_0"].update(info_overrides)
        return {"obj_info": info, "reference_obj": "floor_0"}

    def audit(self, **record_overrides):
        settings = dict(
            margin=-0.08,
            com=(0.6, 0.0, 1.05),
            polygon=square(0.3, 0.5, -0.1, 0.1),
        )
        settings.update(record_overrides)
        return {"objects": {"thing_0": record(**settings)}}

    def screen(self, placement, audit):
        return fix94.screen_scene(
            audit,
            placement,
            margin_threshold_m=0.005,
            translate_budget_m=0.15,
            new_overlap_tolerance_m3=1e-6,
            elevation_saturation_m=0.5,
            top_k=3,
        )

    def test_an_unmeasured_com_is_not_treated_as_a_defect(self) -> None:
        result = self.screen(self.base(), self.audit(status="abstained"))
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["excluded_counts"]["com_abstained"], 1)

    def test_an_uncertified_com_is_left_alone(self) -> None:
        result = self.screen(self.base(), self.audit(certificate="abstain"))
        self.assertEqual(result["excluded_counts"]["com_uncertified"], 1)

    def test_a_wall_child_is_a_plane_attachment_not_a_falling_object(self) -> None:
        placement = self.base(supported="wall_0")
        placement["obj_info"]["wall_0"] = entry((0.0, 2.0, 1.5), (2.0, 0.02, 1.5))
        result = self.screen(placement, self.audit())
        self.assertEqual(result["excluded_counts"]["plane_attachment"], 1)

    def test_a_contained_object_is_held_not_resting(self) -> None:
        result = self.screen(self.base(SpatialRel="inside"), self.audit())
        self.assertEqual(result["excluded_counts"]["containment_held"], 1)

    def test_a_support_parent_is_never_moved(self) -> None:
        placement = self.base()
        placement["obj_info"]["cup_0"] = entry(
            (0.6, 0.0, 1.15), (0.03, 0.03, 0.04), supported="thing_0"
        )
        result = self.screen(placement, self.audit())
        self.assertEqual(
            result["excluded_counts"]["is_a_declared_support_parent"], 1
        )


class SimulationGroupingTest(unittest.TestCase):
    def build(self, positions):
        info = {"floor_0": entry((0.0, 0.0, 0.0), (5.0, 5.0, 0.02))}
        objects = {}
        for index, (x, y) in enumerate(positions):
            name = f"pillow_{index}"
            info[name] = entry((x, y, 0.65), (0.2, 0.15, 0.05), supported="bed_0")
            objects[name] = record(
                margin=-0.05 - index * 0.001,
                com=(x, y, 0.65),
                polygon=square(x - 0.3, x - 0.1, y - 0.15, y + 0.15),
            )
        info["bed_0"] = entry((0.0, 0.0, 0.3), (1.0, 1.0, 0.3), supported="floor_0")
        return {"objects": objects}, {
            "obj_info": info,
            "reference_obj": "floor_0",
        }

    def screen(self, positions, top_k=3):
        audit, placement = self.build(positions)
        return fix94.screen_scene(
            audit,
            placement,
            margin_threshold_m=0.005,
            translate_budget_m=0.15,
            new_overlap_tolerance_m3=1e-6,
            elevation_saturation_m=0.5,
            top_k=top_k,
        )

    def test_far_apart_targets_share_one_simulation(self) -> None:
        result = self.screen([(0.0, 0.0), (3.0, 0.0), (-3.0, 0.0)])
        self.assertEqual(len(result["simulation_groups"]), 3)
        self.assertTrue(
            all(len(group) == 1 for group in result["simulation_groups"])
        )

    def test_touching_targets_are_separated_into_one_group(self) -> None:
        # Two pillows whose footprints overlap: one would become the other's
        # moving ground, so they must be simulated together, not independently.
        result = self.screen([(0.0, 0.0), (0.1, 0.0), (3.0, 0.0)])
        sizes = sorted(len(group) for group in result["simulation_groups"])
        self.assertEqual(sizes, [1, 2])

    def test_top_k_caps_the_work(self) -> None:
        result = self.screen([(0.0, 0.0), (3.0, 0.0), (-3.0, 0.0)], top_k=2)
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(result["selected_count"], 2)


class ComMeasurementConsistencyTest(unittest.TestCase):
    """The only automatic filter is an impossibility test on the data itself.

    Uniform density is assumed throughout and checked by rendering, not by
    geometric proxies.  Two earlier proxies were withdrawn after Smoke5 showed a
    fill-ratio floor fires on every legged furniture item, including a chair whose
    settle Fix84 had verified, and that a COM-height ceiling split three identical
    pillows apart at an arbitrary line.  What remains are tests no real object can
    fail: a mesh cannot exceed its own bounding box, and a centre of mass cannot
    lie outside the object.
    """

    def scene(self, *, half, mesh_volume, com_z):
        placement = {
            "obj_info": {
                "floor_0": entry((0.0, 0.0, 0.0), (5.0, 5.0, 0.02)),
                "probe_0": entry((0.0, 0.0, half[2]), half, supported="floor_0"),
            },
            "reference_obj": "floor_0",
        }
        audit = {
            "objects": {
                "probe_0": {
                    **record(
                        margin=-0.29,
                        com=(0.0, 0.0, com_z),
                        polygon=square(-0.05, 0.05, -0.05, 0.05),
                    ),
                    "mesh_volume_m3": mesh_volume,
                }
            }
        }
        return audit, placement

    def screen(self, audit, placement, **overrides):
        settings = dict(
            margin_threshold_m=0.005,
            translate_budget_m=0.15,
            new_overlap_tolerance_m3=1e-6,
            elevation_saturation_m=0.5,
            top_k=3,
            fill_ratio_ceiling=1.05,
            com_height_band=0.02,
        )
        settings.update(overrides)
        return fix94.screen_scene(audit, placement, **settings)

    def test_a_thin_object_thickened_by_voxels_is_caught(self) -> None:
        # A mouse pad:0.3 by 0.25 by 0.003 m box, reported mesh volume 0.00045 m3,
        # a fill ratio near 2.No object can be larger than its own box.
        audit, placement = self.scene(
            half=(0.15, 0.125, 0.0015), mesh_volume=0.00045, com_z=0.003
        )
        entry_ = self.screen(audit, placement)["selected"][0]
        self.assertGreater(entry_["fill_ratio"], 1.05)
        self.assertFalse(entry_["com_measurement_consistent"])
        self.assertIn(
            "mesh_volume_exceeds_its_bounding_box",
            entry_["com_inconsistency_reasons"],
        )
        self.assertFalse(entry_["actionable"])

    def test_a_com_below_the_object_is_caught(self) -> None:
        # A file folder reported with a relative COM height of about -0.12.
        audit, placement = self.scene(
            half=(0.15, 0.1, 0.05), mesh_volume=0.002, com_z=-0.012
        )
        entry_ = self.screen(audit, placement)["selected"][0]
        self.assertLess(entry_["com_height_ratio"], 0.0)
        self.assertIn(
            "com_outside_its_bounding_box", entry_["com_inconsistency_reasons"]
        )

    def test_a_com_above_the_object_is_caught(self) -> None:
        audit, placement = self.scene(
            half=(0.15, 0.1, 0.05), mesh_volume=0.002, com_z=0.2
        )
        entry_ = self.screen(audit, placement)["selected"][0]
        self.assertGreater(entry_["com_height_ratio"], 1.02)
        self.assertIn(
            "com_outside_its_bounding_box", entry_["com_inconsistency_reasons"]
        )

    def test_a_legged_object_is_no_longer_filtered(self) -> None:
        # A chair fills about a tenth of its box because the space between its
        # legs is empty.  That is a correct low fill ratio, not a defect, and
        # Fix84 verified this very settle.
        audit, placement = self.scene(
            half=(0.3, 0.3, 0.45), mesh_volume=0.0284, com_z=0.45
        )
        entry_ = self.screen(audit, placement)["selected"][0]
        self.assertLess(entry_["fill_ratio"], 0.15)
        self.assertTrue(entry_["com_measurement_consistent"])
        self.assertTrue(entry_["actionable"])

    def test_a_com_above_mid_height_is_no_longer_filtered(self) -> None:
        # Uniform density calling an object top-heavy is not a reason to abstain;
        # that verdict is checked in the render.
        audit, placement = self.scene(
            half=(0.2, 0.2, 0.5), mesh_volume=0.16, com_z=0.8
        )
        entry_ = self.screen(audit, placement)["selected"][0]
        self.assertAlmostEqual(entry_["com_height_ratio"], 0.8)
        self.assertTrue(entry_["actionable"])

    def test_inconsistent_candidates_are_still_reported_by_default(self) -> None:
        audit, placement = self.scene(
            half=(0.15, 0.125, 0.0015), mesh_volume=0.00045, com_z=0.003
        )
        result = self.screen(audit, placement)
        self.assertEqual(result["selected_count"], 1)
        self.assertFalse(result["require_consistent_com"])

    def test_requiring_consistency_drops_them_from_the_repair_list(self) -> None:
        audit, placement = self.scene(
            half=(0.15, 0.125, 0.0015), mesh_volume=0.00045, com_z=0.003
        )
        result = self.screen(audit, placement, require_consistent_com=True)
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["excluded_counts"]["com_measurement_inconsistent"], 1)

    def test_a_missing_mesh_volume_is_not_an_inconsistency(self) -> None:
        audit, placement = self.scene(
            half=(0.2, 0.15, 0.05), mesh_volume=None, com_z=0.05
        )
        entry_ = self.screen(audit, placement)["selected"][0]
        self.assertIsNone(entry_["fill_ratio"])
        self.assertTrue(entry_["com_measurement_consistent"])

    def test_the_policy_states_what_is_assumed_and_what_is_tested(self) -> None:
        audit, placement = self.scene(
            half=(0.2, 0.15, 0.05), mesh_volume=0.005, com_z=0.05
        )
        policy = self.screen(audit, placement)["policy"]
        self.assertTrue(policy["uniform_density_is_assumed_and_checked_by_rendering"])
        self.assertTrue(
            policy["consistency_filter_is_a_data_validity_test_not_a_world_model"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
