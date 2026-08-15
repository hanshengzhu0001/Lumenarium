import unittest

from sceneproof_render_identity_triplet_compare import distance


class RenderIdentityCompareTest(unittest.TestCase):
    def test_centroid_distance(self):
        self.assertAlmostEqual(distance([1, 2], [4, 6]), 5.0)
        self.assertIsNone(distance(None, [4, 6]))


if __name__ == "__main__":
    unittest.main()
