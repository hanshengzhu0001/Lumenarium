#!/usr/bin/env python3
"""Tests for Fix96: locating visible defects and attributing them to a stage.

The camera projection is checked against hand-computed pixel coordinates, because
the whole point of this screen is that it ranks by what the camera actually sees:
the previous overhang ranking assumed a side view and consequently put four
invisible candidates at the top.

The attribution is checked to follow the scaling chain rather than the symptom, so
a runaway depth estimate is not reported as a retrieval fault.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sceneproof_scene_defect_screen_fix96 as fix96


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
    matrix = np.eye(4)
    matrix[:3, 3] = centre
    return {
        "pose_matrix_for_blender": matrix.tolist(),
        "bbox": bbox(centre, half),
        "length": [2 * half[0], 2 * half[1], 2 * half[2]],
        "retrieved_asset": "asset",
        **extra,
    }


def camera_at(position, *, looking_down_negative_z=True) -> dict:
    """A camera at ``position``.

    A Blender camera looks along its local -Z.  With identity rotation that is
    straight down, which is the top-down framing several Smoke5 scenes use.  With
    ``looking_down_negative_z=False`` the camera is rotated so its local -Z points
    along world +Y, that is horizontally forward.
    """
    matrix = np.eye(4)
    matrix[:3, 3] = position
    if not looking_down_negative_z:
        # Columns are the local x, y, z axes expressed in world coordinates, so
        # local -Z maps to world +Y and local +Y maps to world +Z.
        matrix[:3, :3] = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    return {"pose_matrix_for_blender": matrix.tolist()}


class ProjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.resolution = 1024
        self.focal = 30.0 / 36.0 * self.resolution  # 853.33 px

    def test_a_point_on_the_axis_lands_at_the_image_centre(self) -> None:
        camera = np.eye(4)
        camera[:3, 3] = (0.0, 0.0, 5.0)
        pixels, in_front = fix96.project_to_pixels(
            np.asarray([[0.0, 0.0, 0.0]]),
            camera,
            focal_px=self.focal,
            resolution=self.resolution,
        )
        self.assertTrue(in_front.all())
        np.testing.assert_allclose(pixels[0], [512.0, 512.0], atol=1e-9)

    def test_an_offset_point_scales_with_focal_length_over_depth(self) -> None:
        camera = np.eye(4)
        camera[:3, 3] = (0.0, 0.0, 5.0)
        # 1 m to the right, 5 m away: offset is focal / 5 pixels.
        pixels, _ = fix96.project_to_pixels(
            np.asarray([[1.0, 0.0, 0.0]]),
            camera,
            focal_px=self.focal,
            resolution=self.resolution,
        )
        self.assertAlmostEqual(pixels[0][0], 512.0 + self.focal / 5.0, places=6)
        self.assertAlmostEqual(pixels[0][1], 512.0, places=9)

    def test_up_in_the_world_is_up_in_the_image(self) -> None:
        camera = np.eye(4)
        camera[:3, 3] = (0.0, 0.0, 5.0)
        pixels, _ = fix96.project_to_pixels(
            np.asarray([[0.0, 1.0, 0.0]]),
            camera,
            focal_px=self.focal,
            resolution=self.resolution,
        )
        self.assertLess(pixels[0][1], 512.0)

    def test_a_point_behind_the_camera_is_not_in_front(self) -> None:
        camera = np.eye(4)
        camera[:3, 3] = (0.0, 0.0, 5.0)
        _, in_front = fix96.project_to_pixels(
            np.asarray([[0.0, 0.0, 10.0]]),
            camera,
            focal_px=self.focal,
            resolution=self.resolution,
        )
        self.assertFalse(in_front.any())

    def test_an_object_off_frame_reports_zero_area(self) -> None:
        pixels = np.asarray([[-500.0, -500.0], [-100.0, -100.0]])
        report = fix96.framed_pixel_area(
            pixels, np.asarray([True, True]), self.resolution
        )
        self.assertTrue(report["fully_outside_frame"])
        self.assertEqual(report["pixel_area_fraction"], 0.0)

    def test_a_half_frame_object_reports_half_the_area(self) -> None:
        pixels = np.asarray([[0.0, 0.0], [1024.0, 512.0]])
        report = fix96.framed_pixel_area(
            pixels, np.asarray([True, True]), self.resolution
        )
        self.assertAlmostEqual(report["pixel_area_fraction"], 0.5)


class ShapeSignatureTest(unittest.TestCase):
    def signatures(self, length):
        return fix96.shape_signatures(
            np.asarray(length, dtype=np.float64), rod_aspect=20.0, sheet_aspect=0.02
        )

    def test_a_thin_rod_is_flagged(self) -> None:
        # The vertical pole seen in two scenes: 3 m tall, 5 cm across.
        reasons, shape = self.signatures([0.05, 0.05, 3.0])
        self.assertIn("rod_like_extreme_aspect", reasons)
        self.assertAlmostEqual(shape["rod_aspect"], 60.0)

    def test_a_paper_thin_sheet_is_flagged(self) -> None:
        reasons, _ = self.signatures([2.0, 1.5, 0.01])
        self.assertIn("sheet_like_extreme_aspect", reasons)

    def test_ordinary_furniture_is_not_flagged(self) -> None:
        reasons, _ = self.signatures([2.0, 0.9, 0.8])
        self.assertEqual(reasons, [])

    def test_a_missing_length_yields_no_signature(self) -> None:
        reasons, shape = fix96.shape_signatures(
            None, rod_aspect=20.0, sheet_aspect=0.02
        )
        self.assertEqual(reasons, [])
        self.assertIsNone(shape["sorted_edges_m"])


class AttributionTest(unittest.TestCase):
    def screen(self, obj_info, **overrides):
        settings = dict(
            lens_mm=30.0,
            sensor_mm=36.0,
            resolution=1024,
            rod_aspect=20.0,
            sheet_aspect=0.02,
            scale_high=5.0,
            scale_low=0.2,
            scale_anisotropy=10.0,
            volume_outlier_factor=50.0,
            category_outlier_factor=5.0,
            top_k=8,
        )
        settings.update(overrides)
        return fix96.screen_scene({"obj_info": obj_info}, **settings)

    def base_room(self):
        info = {
            "floor_0": entry((0.0, 0.0, 0.0), (5.0, 5.0, 0.02)),
            "scene_camera": camera_at((0.0, -6.0, 1.5)),
        }
        for index in range(4):
            info[f"chair_{index}"] = entry(
                (index * 0.6- 1.0, 0.0, 0.45), (0.25, 0.25, 0.45)
            )
        return info

    def test_a_runaway_scale_is_attributed_to_depth_driven_scaling(self) -> None:
        info = self.base_room()
        info["multi_person_sofa_0"] = entry(
            (0.0, 0.0, 3.0),
            (4.0, 3.0, 3.0),
            scale=[12.0, 11.0, 12.5],
            pcd_obb_size=[8.0, 6.0, 6.0],
        )
        result = self.screen(info)
        flagged = {item["object_id"]: item for item in result["all_flagged"]}
        self.assertIn("multi_person_sofa_0", flagged)
        sofa = flagged["multi_person_sofa_0"]
        self.assertIn("scale_factor_far_above_one", sofa["defect_reasons"])
        self.assertEqual(sofa["likely_stage"], "s3_depth_driven_scaling")
        self.assertEqual(result["stage_counts"]["s3_depth_driven_scaling"], 1)

    def test_a_wrong_size_with_a_sane_scale_points_at_the_asset(self) -> None:
        info = self.base_room()
        info["duct_0"] = entry(
            (0.0, 0.0, 1.0), (3.0, 1.0, 1.0), scale=[1.0, 1.0, 1.0]
        )
        flagged = {
            item["object_id"]: item for item in self.screen(info)["all_flagged"]
        }
        self.assertIn("duct_0", flagged)
        self.assertEqual(
            flagged["duct_0"]["likely_stage"], "asset_dimensions_or_retrieval"
        )
        self.assertIn(
            "volume_far_above_scene_median", flagged["duct_0"]["defect_reasons"]
        )

    def test_a_peer_disagreement_is_caught_even_without_a_scale_field(self) -> None:
        info = self.base_room()
        # One chair ten times the volume of its three peers.
        info["chair_4"] = entry((3.0, 0.0, 1.0), (0.6, 0.6, 1.0))
        flagged = {
            item["object_id"]: item for item in self.screen(info)["all_flagged"]
        }
        self.assertIn("chair_4", flagged)
        self.assertIn(
            "volume_disagrees_with_same_category_peers",
            flagged["chair_4"]["defect_reasons"],
        )

    def test_a_clamped_scale_shared_by_several_objects_is_flagged(self) -> None:
        info = self.base_room()
        for index in range(3):
            info[f"litter_{index}"] = entry(
                (index * 0.5, 1.0, 0.1), (0.1, 0.1, 0.1), scale=[3.0, 3.0, 3.0]
            )
        flagged = {
            item["object_id"]: item for item in self.screen(info)["all_flagged"]
        }
        self.assertIn("litter_0", flagged)
        self.assertIn(
            "scale_component_looks_clamped", flagged["litter_0"]["defect_reasons"]
        )

    def test_ranking_is_by_screen_area_not_by_world_size(self) -> None:
        info = self.base_room()
        # A horizontally aimed camera, so "far" and "near" mean what they look
        # like: identity rotation would point straight down instead.
        info["scene_camera"] = camera_at(
            (0.0, -6.0, 1.5), looking_down_negative_z=False
        )
        # A 6 m block 86 m away, and a 3 m rod 3 m away.  The block is a thousand
        # times the volume and a quarter of the screen area.
        info["far_block_0"] = entry(
            (0.0, 80.0, 3.0), (3.0, 3.0, 3.0), scale=[9.0, 8.0, 9.5]
        )
        info["near_rod_0"] = entry(
            (0.0, -3.0, 1.5), (0.03, 0.03, 1.5), scale=[1.0, 1.1, 0.9]
        )
        result = self.screen(info)
        order = [item["object_id"] for item in result["worst_by_screen_area"]]
        self.assertEqual(order[0], "near_rod_0")
        by_id = {item["object_id"]: item for item in result["all_flagged"]}
        self.assertGreater(
            by_id["near_rod_0"]["pixel_area_fraction"],
            by_id["far_block_0"]["pixel_area_fraction"],
        )
        self.assertGreater(
            by_id["far_block_0"]["volume_m3"], by_id["near_rod_0"]["volume_m3"]
        )

    def test_structural_objects_and_the_camera_are_excluded(self) -> None:
        info = self.base_room()
        info["wall_0"] = entry((0.0, 5.0, 1.5), (5.0, 0.02, 1.5))
        result = self.screen(info)
        names = {item["object_id"] for item in result["all_flagged"]}
        self.assertNotIn("floor_0", names)
        self.assertNotIn("wall_0", names)
        self.assertNotIn("scene_camera", names)

    def test_a_clean_scene_flags_nothing(self) -> None:
        result = self.screen(self.base_room())
        self.assertEqual(result["flagged_count"], 0)
        self.assertTrue(result["camera_available"])

    def test_a_missing_camera_is_reported_not_fatal(self) -> None:
        info = self.base_room()
        del info["scene_camera"]
        info["duct_0"] = entry((0.0, 0.0, 1.0), (3.0, 1.0, 1.0))
        result = self.screen(info)
        self.assertFalse(result["camera_available"])
        self.assertEqual(result["flagged_count"], 1)
        self.assertIsNone(result["all_flagged"][0]["pixel_area_fraction"])

    def test_the_policy_records_what_this_screen_cannot_see(self) -> None:
        policy = self.screen(self.base_room())["policy"]
        self.assertTrue(policy["ranking_uses_the_actual_scene_camera"])
        self.assertTrue(policy["attribution_reads_the_scaling_chain_not_a_guess"])
        # A chair retrieved as a curved sheet has the right size and the wrong
        # shape, which no dimension test can detect.
        self.assertTrue(policy["wrong_shape_is_not_detectable_here_only_wrong_size"])
        self.assertTrue(policy["no_pose_or_asset_is_modified_by_this_tool"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
