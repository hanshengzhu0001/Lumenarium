import unittest

try:
    import numpy as np
    import sceneba_yaw_symmetry_audit as symmetry
except ImportError:
    np = None
    symmetry = None


@unittest.skipIf(np is None or symmetry is None, "NumPy/SciPy required")
class SceneBAYawSymmetryAuditTest(unittest.TestCase):
    def test_square_cloud_is_quarter_turn_symmetric(self):
        points = np.asarray(
            [
                [-1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        first = np.eye(4)
        second = np.eye(4)
        second[:3, :3] = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        self.assertAlmostEqual(
            symmetry.normalized_symmetric_chamfer(points, first, second),
            0.0,
        )

    def test_asymmetric_cloud_is_not_quarter_turn_symmetric(self):
        points = np.asarray(
            [
                [-1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [2.0, 0.25, 0.0],
            ]
        )
        first = np.eye(4)
        second = np.eye(4)
        second[:3, :3] = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        self.assertGreater(
            symmetry.normalized_symmetric_chamfer(points, first, second),
            0.02,
        )


if __name__ == "__main__":
    unittest.main()
