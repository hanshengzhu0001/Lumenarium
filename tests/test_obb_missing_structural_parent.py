import unittest
import numpy as np

from utils.obb import refine_all_obbs_with_scene_graph


class MissingStructuralParentTest(unittest.TestCase):
    def test_missing_wall_parent_preserves_original_obb(self):
        obb = np.arange(24, dtype=float).reshape(8, 3)
        actual = refine_all_obbs_with_scene_graph(
            [obb], ["shelf_0"],
            {"floor_0": {"matrix": np.eye(4).tolist()}},
            {"shelf_0": {"supported": "wall_0"}},
        )
        np.testing.assert_array_equal(actual[0], obb)


if __name__ == "__main__":
    unittest.main()
