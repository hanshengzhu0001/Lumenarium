import unittest

from sceneproof_cold_start_selector import fast_proxy, rank


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

    def test_high_rank_prefers_positive_critical_and_fewer_severe_pairs(self):
        good = {"high": {"metrics": {
            "headline_critical_realizability": 0.2,
            "headline_macro_realizability": 0.5,
            "unintended_collision_pairs": 2,
        }, "severe_collision_pair_count": 1, "relation_program_count": 10}}
        bad = {"high": {"metrics": {
            "headline_critical_realizability": 0.0,
            "headline_macro_realizability": 0.8,
            "unintended_collision_pairs": 1,
        }, "severe_collision_pair_count": 0, "relation_program_count": 10}}
        self.assertGreater(rank(good, "high"), rank(bad, "high"))


if __name__ == "__main__":
    unittest.main()
