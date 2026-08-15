import unittest

from sceneproof_pillow_pose_lineage_fix75 import matrix_metrics


class SceneProofPillowPoseLineageTest(unittest.TestCase):
    def test_matrix_metrics_separates_translation_and_linear_change(self):
        first = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        second = [row[:] for row in first]
        second[0][3] = 0.3
        second[1][1] = 0.8
        metrics = matrix_metrics(first, second)
        self.assertAlmostEqual(metrics["translation_norm_m"], 0.3)
        self.assertAlmostEqual(metrics["linear_frobenius"], 0.2)
        self.assertAlmostEqual(metrics["max_abs"], 0.3)


if __name__ == "__main__":
    unittest.main()
