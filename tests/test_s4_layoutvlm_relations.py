import math
import unittest

from modules._s4_layoutvlm_relations import build_semantic_relation_specs


def pose(x=0.0, y=0.0, yaw=0.0):
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return [
        [cosine, -sine, 0.0, x],
        [sine, cosine, 0.0, y],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


class LayoutVLMRelationBuilderTest(unittest.TestCase):
    def test_directly_facing_builds_point_and_warm_distance_band(self):
        ids = ["chair_0", "table_0"]
        matrices = [pose(yaw=math.pi / 2.0), pose(x=2.0)]
        specs = build_semantic_relation_specs(
            {
                "chair_0": {"directlyFacing": "table_0"},
                "table_0": {},
            },
            ids,
            matrices,
            [(0.5, 0.5), (1.0, 1.0)],
        )
        self.assertEqual(specs["point_pairs"], [(0, 1)])
        self.assertEqual(specs["distance_pairs"], [(0, 1)])
        self.assertLessEqual(specs["distance_minimum"][0], 2.0)
        self.assertGreaterEqual(specs["distance_maximum"][0], 2.0)

    def test_inconsistent_facing_relation_is_filtered(self):
        specs = build_semantic_relation_specs(
            {
                "chair_0": {"directlyFacing": "table_0"},
                "table_0": {},
            },
            ["chair_0", "table_0"],
            [pose(), pose(x=2.0)],
            [(0.5, 0.5), (1.0, 1.0)],
        )
        self.assertEqual(specs["point_pairs"], [])
        self.assertTrue(
            any(
                entry["reason"] == "not self-consistent with warm start"
                for entry in specs["skipped"]
            )
        )

    def test_group_alignment_excludes_table_facing_members(self):
        ids = ["cabinet_0", "cabinet_1", "chair_0", "chair_1", "table_0"]
        matrices = [
            pose(),
            pose(x=1.0),
            pose(y=2.0, yaw=0.0),
            pose(y=-2.0, yaw=math.pi),
            pose(),
        ]
        info = {
            "cabinet_0": {"group": "cabinet"},
            "cabinet_1": {"group": "cabinet"},
            "chair_0": {"group": "chairs", "directlyFacing": "table_0"},
            "chair_1": {"group": "chairs", "directlyFacing": "table_0"},
            "table_0": {},
        }
        specs = build_semantic_relation_specs(
            info,
            ids,
            matrices,
            [(1.0, 1.0)] * len(ids),
            point_min_cosine=-1.0,
        )
        self.assertEqual(specs["align_pairs"], [(1, 0)])
        self.assertNotIn((2, 3), specs["align_pairs"])
        self.assertNotIn((3, 2), specs["align_pairs"])

    def test_missing_targets_are_auditable_and_stacked_on_is_not_routed(self):
        specs = build_semantic_relation_specs(
            {
                "box_0": {
                    "stacked_on": "box_1",
                    "alignWith": "missing_0",
                },
                "box_1": {},
            },
            ["box_0", "box_1"],
            [pose(), pose()],
            [(1.0, 1.0), (1.0, 1.0)],
        )
        self.assertNotIn("stack_pairs", specs)
        self.assertEqual(specs["align_pairs"], [])
        self.assertTrue(
            any(entry["target"] == "missing_0" for entry in specs["skipped"])
        )


if __name__ == "__main__":
    unittest.main()
