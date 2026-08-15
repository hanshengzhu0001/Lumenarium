import math
import unittest

import numpy as np

import sceneproof_attachment_axis_audit as audit


class SceneProofAttachmentAxisAuditTest(unittest.TestCase):
    def test_wall_normal_uses_smallest_scaled_axis_and_faces_child(self):
        wall = np.eye(4, dtype=np.float64)
        wall[:3, :3] = np.diag([0.05, 4.0, 2.5])
        child = np.eye(4, dtype=np.float64)
        child[0, 3] = -1.0
        normal, axis, distinctness = audit.infer_wall_normal(child, wall)
        self.assertEqual(axis, 0)
        np.testing.assert_allclose(normal, [-1.0, 0.0, 0.0])
        self.assertGreater(distinctness, 1.0)

    def test_rotation_angle_detects_quarter_turn_axis_swap(self):
        first = np.eye(4, dtype=np.float64)
        second = np.eye(4, dtype=np.float64)
        second[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        self.assertAlmostEqual(audit.angle_deg(first, second), 90.0)

    def test_nonuniform_scale_does_not_change_rotation_angle(self):
        first = np.eye(4, dtype=np.float64)
        second = np.eye(4, dtype=np.float64)
        first[:3, :3] = np.diag([0.2, 2.0, 3.0])
        second[:3, :3] = np.diag([0.4, 4.0, 6.0])
        self.assertTrue(math.isclose(audit.angle_deg(first, second), 0.0))


if __name__ == "__main__":
    unittest.main()
