import unittest

from sceneproof_cold_start_selector import fast_proxy, rank, relation_coverage


class ColdStartSelectorTest(unittest.TestCase):
    def test_fast_proxy_penalizes_extreme_geometry_and_missing_parent(self):
        doc = {"obj_info": {
            "small_0": {"pcd_obb_size": [1, 1, 1], "supported": "floor_0"},
            "floor_0": {"pcd_obb_size": [10, 10, 0.1]},
            "bad_0": {"pcd_obb_size": [8, 8, 8], "supported": "missing_0"},
        }}
        proxy = fast_proxy(doc)
        self.assertIn("bad_0", proxy["extreme_geometry_object_ids"])
        self.assertEqual(proxy["missing_support_parent_count"], 1)

    def test_high_rank_prefers_fewer_severe_pairs_before_macro(self):
        good = {"high": {"metrics": {
            "headline_critical_realizability": 0.2,
            "headline_macro_realizability": 0.5,
            "unintended_collision_pairs": 2,
        }, "severe_collision_pair_count": 0, "relation_program_count": 10,
            "relation_program_coverage": {"coverage": 0.8}}}
        bad = {"high": {"metrics": {
            "headline_critical_realizability": 0.9,
            "headline_macro_realizability": 0.8,
            "unintended_collision_pairs": 1,
        }, "severe_collision_pair_count": 1, "relation_program_count": 10,
            "relation_program_coverage": {"coverage": 0.9}}}
        self.assertGreater(rank(good, "high"), rank(bad, "high"))

    def test_relation_coverage_counts_unique_scene_objects(self):
        coverage = relation_coverage({
            "obj_info": {"chair": {}, "table": {}, "scene_camera": {}},
            "sceneproof_relation_programs": {"programs": [{
                "kind": "SUPPORT",
                "participants": [
                    {"object_id": "chair"},
                    {"object_id": "architecture:floor_0"},
                ],
            }]},
        })
        self.assertEqual(2, coverage["evaluable_object_count"])
        self.assertEqual(["chair"], coverage["covered_object_ids"])
        self.assertEqual(0.5, coverage["coverage"])


if __name__ == "__main__":
    unittest.main()
