import unittest

from sceneproof_support_visual_identity_audit import pose_delta


def info(tx=0.0):
    return {
        "pose_matrix_for_blender": [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }


class SupportVisualIdentityAuditTest(unittest.TestCase):
    def test_pose_delta_separates_translation_and_linear_change(self):
        delta = pose_delta(info(0.0), info(0.25))
        self.assertAlmostEqual(delta["translation_l2_m"], 0.25)
        self.assertAlmostEqual(delta["linear_l2"], 0.0)
        self.assertAlmostEqual(delta["matrix_l2"], 0.25)

    def test_missing_pose_is_explicit(self):
        delta = pose_delta({}, info())
        self.assertIsNone(delta["matrix_l2"])


if __name__ == "__main__":
    unittest.main()
