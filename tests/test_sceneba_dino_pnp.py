import unittest

try:
    import numpy as np
    import sceneba_dino_pnp_candidates as pnp
except ImportError:
    np = None
    pnp = None


@unittest.skipIf(np is None or pnp is None, "NumPy is required")
class SceneBADinoPnPTest(unittest.TestCase):
    def test_render_unit_scale_recovers_centimetre_fbx_conversion(self):
        mesh_points = np.asarray(
            [
                [-50.0, -50.0, -50.0],
                [50.0, 50.0, 50.0],
            ],
            dtype=float,
        )
        raw_radius = np.linalg.norm([50.0, 50.0, 50.0])
        rendered_radius = raw_radius * 0.01
        camera_distance = rendered_radius / 0.36 * 1.1
        camera_to_world = np.eye(4)
        camera_to_world[2, 3] = camera_distance
        self.assertAlmostEqual(
            pnp.infer_render_unit_scale(mesh_points, camera_to_world),
            0.01,
            places=8,
        )

    def test_patch_to_original_pixels_inverts_square_crop(self):
        points = np.asarray([[0.0, 0.0], [15.0, 15.0]])
        pixels = pnp.patch_to_original_pixels(
            points, [100.0, 200.0, 324.0, 424.0]
        )
        self.assertTrue(np.allclose(pixels[0], [107.0, 207.0]))
        self.assertTrue(np.allclose(pixels[1], [317.0, 417.0]))

    def test_patch_to_original_pixels_removes_letterbox_padding(self):
        points = np.asarray([[0.0, 4.0], [15.0, 11.0]])
        pixels = pnp.patch_to_original_pixels(
            points, [10.0, 20.0, 234.0, 132.0]
        )
        self.assertAlmostEqual(pixels[0, 0], 17.0)
        self.assertAlmostEqual(pixels[1, 0], 227.0)
        self.assertGreaterEqual(pixels[0, 1], 20.0)
        self.assertLessEqual(pixels[1, 1], 132.0)

    def test_opencv_camera_pose_converts_to_blender_world(self):
        rotation, translation = pnp.rigid_camera_to_world(
            np.eye(3), np.asarray([1.0, 2.0, 3.0]), np.eye(4)
        )
        self.assertTrue(
            np.allclose(rotation, np.diag([1.0, -1.0, -1.0]))
        )
        self.assertTrue(np.allclose(translation, [1.0, -2.0, -3.0]))

    def test_ransac_pnp_recovers_synthetic_metric_translation(self):
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV is required")
        query_patch = np.asarray(
            [
                [3, 3],
                [12, 3],
                [3, 12],
                [12, 12],
                [5, 7],
                [10, 7],
                [7, 5],
                [7, 10],
            ],
            dtype=float,
        )
        query_pixels = (query_patch + 0.5) * 14.0
        depth_offsets = np.asarray(
            [-0.5, 0.4, 0.2, -0.3, 0.6, -0.6, 0.8, -0.8]
        )
        total_depth = 5.0 + depth_offsets
        focal = 200.0
        object_points = np.column_stack(
            [
                (query_pixels[:, 0] - 112.0) * total_depth / focal,
                (query_pixels[:, 1] - 112.0) * total_depth / focal,
                depth_offsets,
            ]
        )
        template_patch = np.column_stack(
            [np.arange(8) % 4 + 5, np.arange(8) // 4 + 5]
        ).astype(float)
        pointmap = {
            (int(template[0]), int(template[1])): point
            for template, point in zip(template_patch, object_points)
        }
        candidate = pnp.solve_pnp_candidate(
            record={
                "view_id": 7,
                "query_points_xy": query_patch.tolist(),
                "template_points_xy": template_patch.tolist(),
                "scores": [0.9] * 8,
            },
            pointmap=pointmap,
            query_box=[0.0, 0.0, 224.0, 224.0],
            query_intrinsics=np.asarray(
                [[focal, 0.0, 112.0], [0.0, focal, 112.0], [0.0, 0.0, 1.0]]
            ),
            camera_to_world=np.eye(4),
            reference_matrix=np.eye(4),
            minimum_matches=8,
            reprojection_threshold_px=2.0,
        )
        self.assertIsNotNone(candidate)
        pose = np.asarray(candidate["pose_matrix_for_blender"])
        self.assertTrue(
            np.allclose(pose[:3, 3], [0.0, 0.0, -5.0], atol=0.1)
        )
        self.assertGreaterEqual(candidate["pnp_inliers"], 6)


if __name__ == "__main__":
    unittest.main()
