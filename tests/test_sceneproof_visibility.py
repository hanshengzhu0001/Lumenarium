import unittest

import numpy as np

from modules._sceneproof_visibility import (
    attribute_occluders,
    binary_mask_metrics,
    classify_visibility,
    decode_color_id_image,
    minimum_translation_into_convex_polygon,
)


class SceneProofVisibilityTest(unittest.TestCase):
    def test_finite_plane_patch_returns_minimum_tangent_translation(self):
        result = minimum_translation_into_convex_polygon(
            [[-1, -1], [1, -1], [1, 1], [-1, 1]],
            [[1.3, -0.2], [1.7, -0.2], [1.7, 0.2], [1.3, 0.2]],
        )
        self.assertTrue(result["feasible"])
        self.assertFalse(result["contained"])
        np.testing.assert_allclose(result["translation"], [-0.7, 0.0])
        self.assertAlmostEqual(result["maximum_outside_distance"], 0.7)

    def test_finite_plane_patch_keeps_contained_child_fixed(self):
        result = minimum_translation_into_convex_polygon(
            [[-1, -1], [1, -1], [1, 1], [-1, 1]],
            [[-0.2, -0.2], [0.2, 0.2]],
        )
        self.assertTrue(result["contained"])
        np.testing.assert_allclose(result["translation"], [0.0, 0.0])

    def test_occluder_attribution_uses_frontmost_full_scene_labels(self):
        full = np.array([[1, 2], [2, 0]], dtype=np.int32)
        isolated = np.ones((2, 2), dtype=bool)
        result = attribute_occluders(
            full,
            isolated,
            {1: "target", 2: "wall"},
            target_label=1,
        )
        self.assertEqual(result["target_visible_pixels"], 1)
        self.assertEqual(result["unknown_or_background_pixels"], 1)
        self.assertEqual(result["dominant_occluder"], "wall")
        self.assertEqual(result["dominant_occluder_pixels"], 2)
        self.assertAlmostEqual(result["dominant_occluder_fraction"], 0.5)

    def test_color_id_decoder_rejects_background_and_recovers_ids(self):
        image = np.array(
            [[[0.0, 0.0, 0.0], [0.21, 0.49, 0.81]]],
            dtype=np.float32,
        )
        labels = decode_color_id_image(
            image,
            [[0.2, 0.5, 0.8], [0.8, 0.5, 0.2]],
        )
        np.testing.assert_array_equal(labels, [[0, 1]])

    def test_binary_mask_metrics_are_per_object(self):
        observed = np.array([[1, 1], [0, 0]], dtype=bool)
        rendered = np.array([[0, 1], [0, 1]], dtype=bool)
        result = binary_mask_metrics(rendered, observed)
        self.assertEqual(result["rendered_visible_pixels"], 2)
        self.assertEqual(result["observed_mask_pixels"], 2)
        self.assertEqual(result["intersection_pixels"], 1)
        self.assertEqual(result["union_pixels"], 3)
        self.assertAlmostEqual(result["iou"], 1.0 / 3.0)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)

    def test_visibility_class_separates_occlusion_from_absence(self):
        self.assertEqual(
            classify_visibility(0, 100, minimum_pixels=10),
            "fully_occluded",
        )
        self.assertEqual(
            classify_visibility(0, 0, minimum_pixels=10),
            "outside_view_or_degenerate",
        )
        self.assertEqual(
            classify_visibility(20, 100, minimum_pixels=10),
            "partially_occluded",
        )
        self.assertEqual(
            classify_visibility(100, 100, minimum_pixels=10),
            "visible",
        )


if __name__ == "__main__":
    unittest.main()
