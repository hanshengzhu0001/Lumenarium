import unittest

from scenelm_rotation_responsibility_audit import (
    materialize_policy,
    optimizable_object_ids,
    relation_class,
    released_object_ids,
    replace_linear_pose,
    yaw_observability,
)
from scenelm_rotation_responsibility_finalize import evaluate_policy


IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
ROTATED = [
    [0.0, -1.0, 0.0, 4.0],
    [1.0, 0.0, 0.0, 5.0],
    [0.0, 0.0, 1.0, 6.0],
    [0.0, 0.0, 0.0, 1.0],
]


class SceneLMRotationResponsibilityTest(unittest.TestCase):
    def test_rotation_swap_preserves_candidate_translation(self):
        output = replace_linear_pose(ROTATED, IDENTITY)
        self.assertEqual([output[i][3] for i in range(3)], [4.0, 5.0, 6.0])
        self.assertEqual([row[:3] for row in output[:3]], [row[:3] for row in IDENTITY[:3]])

    def test_rotation_swap_does_not_import_reference_scale(self):
        candidate = [row[:] for row in ROTATED]
        candidate[0][0], candidate[0][1] = 0.0, -2.0
        reference = [row[:] for row in IDENTITY]
        reference[0][0], reference[1][1], reference[2][2] = 9.0, 8.0, 7.0
        output = replace_linear_pose(candidate, reference)
        column_norms = [sum(output[r][c] ** 2 for r in range(3)) ** 0.5 for c in range(3)]
        self.assertEqual([round(value, 6) for value in column_norms], [1.0, 2.0, 1.0])

    def test_observability_requires_evidence_or_anisotropy(self):
        ambiguous = yaw_observability({"bbox": [[-1, -1], [1, 1]]}, 1.15)
        anisotropic = yaw_observability({"bbox": [[-2, -1], [2, 1]]}, 1.15)
        semantic = yaw_observability({"alignWith": "desk_0"}, 1.15)
        self.assertFalse(ambiguous["observed"])
        self.assertTrue(anisotropic["observed"])
        self.assertTrue(semantic["observed"])

    def test_relation_and_released_index_mapping_are_auditable(self):
        document = {
            "obj_info": {
                "scene_camera": {"pose_matrix_for_blender": IDENTITY},
                "floor_0": {"pose_matrix_for_blender": IDENTITY},
                "desk_0": {"pose_matrix_for_blender": IDENTITY},
                "pen_0": {"pose_matrix_for_blender": IDENTITY, "supported": "desk_0"},
            },
            "scenelm_solver": {
                "relation_coordinates": {"objects": 2},
                "relation_released_object_indices": [1],
            },
        }
        self.assertEqual(optimizable_object_ids(document), ["desk_0", "pen_0"])
        self.assertEqual(released_object_ids(document), {"pen_0"})
        self.assertEqual(relation_class(document["obj_info"]["pen_0"]), "support_child")

    def test_policy_is_oracle_bounded_and_never_changes_translation(self):
        candidate = {"obj_info": {"a": {"pose_matrix_for_blender": ROTATED}, "b": {"pose_matrix_for_blender": ROTATED}}}
        reference = {"obj_info": {"a": {"pose_matrix_for_blender": IDENTITY}, "b": {"pose_matrix_for_blender": IDENTITY}}}
        rows = [
            {"object_id": "a", "reference_rotation_better": True, "observed": True, "released": False, "collision_offender": False, "relation_class": "free_root"},
            {"object_id": "b", "reference_rotation_better": True, "observed": False, "released": False, "collision_offender": False, "relation_class": "free_root"},
        ]
        output, selected = materialize_policy(candidate, reference, rows, policy="oracle_observed", output_version="audit")
        self.assertEqual(selected, ["a"])
        self.assertEqual([output["obj_info"]["a"]["pose_matrix_for_blender"][i][3] for i in range(3)], [4.0, 5.0, 6.0])
        self.assertEqual(output["obj_info"]["b"]["pose_matrix_for_blender"], ROTATED)

    def test_final_gate_requires_rotation_and_every_physical_component(self):
        gt = {"versions": {
            "candidate": {"rotation_auc60_aligned": 0.50, "translation_auc05_aligned": 0.20},
            "oracle": {"rotation_auc60_aligned": 0.53, "translation_auc05_aligned": 0.20},
        }}
        def version(macro, collision, support, plane):
            return {"aggregate": {"headline_macro_realizability": macro, "families": {
                "collision": {"score": collision}, "support": {"score": support}, "plane": {"score": plane}
            }}}
        physical = {"versions": {
            "candidate": version(0.60, 0.60, 0.60, 0.60),
            "oracle": version(0.60, 0.60, 0.58, 0.60),
        }}
        result = evaluate_policy(gt, physical, candidate="candidate", version="oracle", minimum_rotation_gain=0.02, translation_margin=0.005, physical_margin=0.005, component_margin=0.005)
        self.assertFalse(result["all_gates_pass"])
        self.assertFalse(result["gates"]["physical_components_noninferior"])


if __name__ == "__main__":
    unittest.main()
