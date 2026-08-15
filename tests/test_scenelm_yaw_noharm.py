import unittest

from scenelm_yaw_noharm_materialize import (
    has_explicit_orientation_evidence,
    materialize,
    replace_linear_pose,
)


class SceneLMYawNoHarmTest(unittest.TestCase):
    def test_orientation_evidence_is_structural_not_category_guessing(self):
        self.assertTrue(has_explicit_orientation_evidence({"supported": "wall_1"}))
        self.assertTrue(has_explicit_orientation_evidence({"alignWith": "desk_0"}))
        self.assertFalse(has_explicit_orientation_evidence({"supported": "desk_0"}))

    def test_replace_linear_pose_keeps_candidate_translation(self):
        candidate = [[1, 0, 0, 4], [0, 1, 0, 5], [0, 0, 1, 6], [0, 0, 0, 1]]
        source = [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]
        result = replace_linear_pose(candidate, source)
        self.assertEqual([result[i][3] for i in range(3)], [4, 5, 6])
        self.assertEqual([row[:3] for row in result[:3]], [row[:3] for row in source[:3]])

    def test_structural_policy_rolls_back_only_unobservable_yaw(self):
        identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        rotated = [[0, -1, 0, 2], [1, 0, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]]
        source = {"obj_info": {"free": {"pose_matrix_for_blender": identity}, "wall": {"pose_matrix_for_blender": identity}}}
        candidate = {"obj_info": {"free": {"supported": "table_0", "pose_matrix_for_blender": rotated}, "wall": {"supported": "wall_0", "pose_matrix_for_blender": rotated}}}
        output, rolled_back = materialize(source, candidate, mode="freeze_unobservable")
        self.assertEqual(rolled_back, ["free"])
        self.assertEqual(output["obj_info"]["free"]["pose_matrix_for_blender"][0][:3], identity[0][:3])
        self.assertEqual(output["obj_info"]["wall"]["pose_matrix_for_blender"][0][:3], rotated[0][:3])


if __name__ == "__main__":
    unittest.main()
