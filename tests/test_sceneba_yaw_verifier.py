import unittest

try:
    import numpy as np
    import sceneba_yaw_verifier as yaw
    import sceneba_yaw_oracle as oracle
except ImportError:
    np = None
    yaw = None
    oracle = None


@unittest.skipIf(np is None or yaw is None, "NumPy/OpenCV dependencies required")
class SceneBAYawVerifierTest(unittest.TestCase):
    def test_world_yaw_preserves_translation_and_column_scales(self):
        base = np.eye(4)
        base[:3, :3] = np.diag([2.0, 3.0, 4.0])
        base[:3, 3] = [1.0, 2.0, 3.0]
        result = yaw.world_yaw_pose(base, 90.0)
        self.assertTrue(np.allclose(result[:3, 3], base[:3, 3]))
        self.assertTrue(
            np.allclose(
                np.linalg.svd(result[:3, :3], compute_uv=False),
                np.linalg.svd(base[:3, :3], compute_uv=False),
            )
        )

    def test_world_yaw_premultiplies_tilted_asset_basis(self):
        base = np.eye(4)
        base[:3, :3] = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -2.0], [0.0, 3.0, 0.0]]
        )
        result = yaw.world_yaw_pose(base, 90.0)
        expected = yaw.yaw_matrix(90.0) @ base[:3, :3]
        self.assertTrue(np.allclose(result[:3, :3], expected))
        self.assertTrue(np.allclose(result[2], base[2]))

    def test_mask_metrics_reward_exact_silhouette(self):
        observed = np.zeros((32, 32), dtype=bool)
        observed[8:24, 10:22] = True
        exact = yaw.mask_metrics(observed, observed)
        shifted = np.roll(observed, 5, axis=1)
        wrong = yaw.mask_metrics(shifted, observed)
        self.assertAlmostEqual(exact["silhouette_iou"], 1.0)
        self.assertAlmostEqual(exact["boundary_chamfer"], 0.0)
        self.assertLess(exact["silhouette_loss"], wrong["silhouette_loss"])
        self.assertLess(exact["boundary_chamfer"], wrong["boundary_chamfer"])

    def test_occlusion_metric_does_not_penalize_hidden_candidate_surface(self):
        observed_mask = np.zeros((8, 8), dtype=bool)
        scene_depth = np.full((8, 8), 2.0, dtype=float)
        candidate_depth = np.full((8, 8), np.inf, dtype=float)
        candidate_depth[2, 2] = 3.0
        hidden = yaw.depth_metrics(candidate_depth, observed_mask, scene_depth)
        candidate_depth[2, 2] = 1.0
        visible = yaw.depth_metrics(candidate_depth, observed_mask, scene_depth)
        self.assertEqual(hidden["occlusion_violation"], 0.0)
        self.assertEqual(visible["occlusion_violation"], 1.0)

    def test_dino_reprojection_prefers_correct_yaw(self):
        pose = np.eye(4)
        pose[2, 3] = -4.0
        world_to_camera = np.eye(4)
        intrinsics = np.asarray(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
        )
        local = {
            (0, 0): np.asarray([-1.0, -0.5, 0.0]),
            (1, 0): np.asarray([1.0, -0.5, 0.0]),
            (0, 1): np.asarray([-1.0, 0.5, 0.0]),
            (1, 1): np.asarray([1.0, 0.5, 0.0]),
        }
        points = np.stack(list(local.values()))
        target, _ = yaw.project_points(
            points, pose, world_to_camera, intrinsics
        )
        record = {
            "view_id": 0,
            "query_points_xy": (
                (target - np.asarray([0.0, 0.0])) / 14.0 - 0.5
            ).tolist(),
            "template_points_xy": [[0, 0], [1, 0], [0, 1], [1, 1]],
            "scores": [1.0, 1.0, 1.0, 1.0],
        }
        # A full-frame 224x224 query makes patch_to_original_pixels the
        # inverse of the construction above.
        correct = yaw.dino_reprojection_score(
            pose,
            world_to_camera,
            intrinsics,
            [record],
            {0: local},
            [0.0, 0.0, 224.0, 224.0],
        )
        wrong = yaw.dino_reprojection_score(
            yaw.world_yaw_pose(pose, 90.0),
            world_to_camera,
            intrinsics,
            [record],
            {0: local},
            [0.0, 0.0, 224.0, 224.0],
        )
        self.assertLess(
            correct["dino_reprojection_loss"],
            wrong["dino_reprojection_loss"],
        )

    def test_verifier_requires_consensus_and_physical_safety(self):
        current = {
            "yaw_deg": 0.0,
            "composite_loss": 0.50,
            "silhouette_loss": 0.50,
            "boundary_chamfer": 0.10,
            "depth_loss": 0.30,
            "dino_orientation_loss": 0.40,
            "collision_increase": 0.0,
            "support_degradation_m": 0.0,
            "symmetry_equivalent_to_current": True,
            "semantic_orientation_affected": False,
            "plane_supported": False,
        }
        candidate = {
            "yaw_deg": 90.0,
            "composite_loss": 0.35,
            "silhouette_loss": 0.45,
            "boundary_chamfer": 0.08,
            "depth_loss": 0.25,
            "dino_orientation_loss": 0.35,
            "collision_increase": 0.0,
            "support_degradation_m": 0.0,
            "symmetry_equivalent_to_current": False,
            "semantic_orientation_affected": False,
            "plane_supported": False,
        }
        _, decision = yaw.verifier_decision([current, candidate])
        self.assertTrue(decision["accepted"])
        unsafe = dict(candidate, collision_increase=0.1)
        _, decision = yaw.verifier_decision([current, unsafe])
        self.assertFalse(decision["accepted"])

        semantic_unsafe = dict(
            candidate, semantic_orientation_affected=True
        )
        _, decision = yaw.verifier_decision([current, semantic_unsafe])
        self.assertFalse(decision["accepted"])

        plane_unsafe = dict(candidate, plane_supported=True)
        _, decision = yaw.verifier_decision([current, plane_unsafe])
        self.assertFalse(decision["accepted"])


@unittest.skipIf(np is None or oracle is None, "NumPy dependencies required")
class SceneBAYawOracleTest(unittest.TestCase):
    def test_policy_never_selects_symmetric_equivalent_mode(self):
        current = {
            "yaw_deg": 0.0,
            "composite_loss": 0.5,
            "silhouette_loss": 0.5,
            "boundary_chamfer": 0.1,
            "depth_loss": 0.3,
            "dino_orientation_loss": 0.4,
        }
        equivalent = {
            "yaw_deg": 180.0,
            "composite_loss": 0.2,
            "silhouette_loss": 0.4,
            "boundary_chamfer": 0.05,
            "depth_loss": 0.2,
            "dino_orientation_loss": 0.3,
            "collision_increase": 0.0,
            "support_degradation_m": 0.0,
            "symmetry_equivalent_to_current": True,
        }
        row = {"candidates": [current, equivalent]}
        self.assertIsNone(oracle.policy_candidate(row, (0.01, 0.0, 2)))


if __name__ == "__main__":
    unittest.main()
