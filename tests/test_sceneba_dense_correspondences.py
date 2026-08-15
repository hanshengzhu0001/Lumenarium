import unittest

try:
    import numpy as np
    from modules._s3_legacy_functions import (
        build_sceneba_dense_correspondences,
    )
except ImportError:
    np = None
    build_sceneba_dense_correspondences = None


@unittest.skipIf(
    np is None or build_sceneba_dense_correspondences is None,
    "A10 inference dependencies are required",
)
class SceneBADenseCorrespondenceTest(unittest.TestCase):
    def test_capture_preserves_view_alignment_and_sorts_scores(self):
        view_ids = np.asarray([[7, 9], [11, 13]])
        query = np.asarray(
            [
                [
                    [[1, 2], [3, 4], [-1, -1]],
                    [[5, 6], [7, 8], [-1, -1]],
                ],
                [
                    [[1, 2], [3, 4], [-1, -1]],
                    [[5, 6], [7, 8], [-1, -1]],
                ],
            ]
        )
        template = query + np.asarray([1, 1])
        scores = np.asarray(
            [
                [[0.6, 0.9, 0.0], [0.8, 0.7, 0.0]],
                [[0.6, 0.9, 0.0], [0.8, 0.7, 0.0]],
            ]
        )
        result = build_sceneba_dense_correspondences(
            view_ids=view_ids,
            query_points=query,
            template_points=template,
            point_scores=scores,
            rotation_angles=[0, 90],
            max_views=2,
        )
        self.assertEqual(
            result["schema_version"],
            "sceneba_dense_correspondence_v1",
        )
        self.assertEqual([row["view_id"] for row in result["views"]], [7, 9])
        self.assertEqual(result["views"][0]["match_count"], 2)
        self.assertEqual(result["views"][0]["scores"], [0.9, 0.6])


if __name__ == "__main__":
    unittest.main()
