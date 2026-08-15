import unittest

from sceneproof_visible_support_proxy import apply_visible_support_proxy


def obj(z, bbox=((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)), **extra):
    return {
        "bbox": [list(bbox[0]), list(bbox[1])],
        "pose_matrix_for_blender": [
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, z], [0, 0, 0, 1]
        ],
        **extra,
    }


def scene(objects, programs=()):
    return {
        "obj_info": objects,
        "sceneproof_relation_programs": {"programs": list(programs)},
        "sceneproof_mesh_visibility_audit": {
            "objects": {
                key: {"status": "measured", "rendered_visible_pixels": 100}
                for key in objects if key != "floor_0"
            }
        },
    }


class VisibleSupportProxyTest(unittest.TestCase):
    def test_z_only_projection_keeps_xy_and_rotation(self):
        data = scene({
            "floor_0": obj(-0.5),
            "chair_0": obj(0.7, supported="floor_0"),
        })
        output, certificate = apply_visible_support_proxy(data)
        pose = output["obj_info"]["chair_0"]["pose_matrix_for_blender"]
        self.assertAlmostEqual(pose[2][3], 0.5)
        self.assertEqual(pose[0][3], 0)
        self.assertEqual(pose[1][3], 0)
        self.assertEqual(certificate["status"], "proxy_certified")
        self.assertEqual(certificate["repaired_object_ids"], ["chair_0"])

    def test_wall_attachment_requires_two_independent_signals(self):
        one_signal = scene({
            "wall_0": obj(1.0),
            "picture_0": obj(1.0, againstWall="wall_0"),
        })
        _, certificate = apply_visible_support_proxy(one_signal)
        self.assertEqual(certificate["status"], "unresolved")
        self.assertEqual(
            certificate["objects"]["picture_0"]["reason"],
            "semantic_attachment_ambiguous",
        )
        two_signals = scene(
            one_signal["obj_info"],
            [{"kind": "PLANE_ATTACH", "participants": [
                {"object_id": "picture_0"}, {"object_id": "wall_0"}
            ]}],
        )
        _, certificate = apply_visible_support_proxy(two_signals)
        self.assertEqual(certificate["status"], "proxy_certified")

    def test_fast_audit_does_not_mutate_available_repair(self):
        data = scene({
            "floor_0": obj(-0.5),
            "chair_0": obj(0.7, supported="floor_0"),
        })
        output, certificate = apply_visible_support_proxy(
            data, repair_enabled=False
        )
        self.assertEqual(
            output["obj_info"]["chair_0"]["pose_matrix_for_blender"][2][3], 0.7
        )
        self.assertEqual(certificate["status"], "unresolved")
        self.assertEqual(
            certificate["objects"]["chair_0"]["reason"],
            "obb_contact_repair_available",
        )

    def test_new_obb_overlap_rolls_back(self):
        data = scene({
            "floor_0": obj(-0.5),
            "box_0": obj(0.8, supported="floor_0"),
            "blocker_0": obj(0.3),
        })
        output, certificate = apply_visible_support_proxy(data)
        self.assertEqual(
            output["obj_info"]["box_0"]["pose_matrix_for_blender"][2][3], 0.8
        )
        self.assertEqual(certificate["objects"]["box_0"]["reason"], "new_obb_overlap")


if __name__ == "__main__":
    unittest.main()
