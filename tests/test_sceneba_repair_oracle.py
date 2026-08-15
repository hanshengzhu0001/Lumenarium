import unittest

import sceneba_repair_oracle as audit


class SceneBARepairOracleTest(unittest.TestCase):
    def test_shift_buckets_match_predeclared_edges(self):
        self.assertEqual(audit.shift_bucket(0.10), "[0,0.10]")
        self.assertEqual(audit.shift_bucket(0.20), "(0.10,0.20]")
        self.assertEqual(audit.shift_bucket(0.35), "(0.20,0.35]")
        self.assertEqual(audit.shift_bucket(0.36), "(0.35,0.50]")

    def test_repair_precision_counts_improvement_and_harm(self):
        summary = audit.summarize_deltas([0.2, 0.0, -0.1, 0.05])
        self.assertEqual(summary["n"], 4)
        self.assertEqual(summary["improved"], 2)
        self.assertEqual(summary["harmed"], 1)
        self.assertEqual(summary["unchanged"], 1)
        self.assertAlmostEqual(summary["precision"], 0.5)
        self.assertAlmostEqual(summary["harm_rate"], 0.25)

    def test_policy_requires_gain_margin_and_shift(self):
        report = {
            "absolute_improvement": 0.02,
            "relative_improvement": 0.20,
            "runner_up_margin": 0.001,
            "candidates": [
                {
                    "anchor": "depth_ray",
                    "yaw_deg": 0,
                    "translation_shift_m": 0.15,
                }
            ],
        }
        self.assertTrue(audit.policy_accept(report, (0.10, 0.0005, 0.20)))
        self.assertFalse(audit.policy_accept(report, (0.30, 0.0005, 0.20)))
        self.assertFalse(audit.policy_accept(report, (0.10, 0.0020, 0.20)))
        self.assertFalse(audit.policy_accept(report, (0.10, 0.0005, 0.10)))

    def test_policy_never_accepts_current_pose(self):
        report = {
            "absolute_improvement": 1.0,
            "relative_improvement": 1.0,
            "runner_up_margin": 1.0,
            "candidates": [
                {
                    "anchor": "current",
                    "yaw_deg": 0,
                    "translation_shift_m": 0.0,
                }
            ],
        }
        self.assertFalse(audit.policy_accept(report, (0.0, 0.0, 0.5)))

    def test_official_summary_weights_per_scene_auc_like_main_evaluator(self):
        result = audit.official_summary(
            [
                ("scene_a", 0.0),
                ("scene_b", 0.1),
                ("scene_b", 0.6),
            ],
            threshold=0.5,
        )
        scene_a = result["per_scene"]["scene_a"]["auc_at_threshold"]
        scene_b = result["per_scene"]["scene_b"]["auc_at_threshold"]
        expected = (scene_a + 2 * scene_b) / 3
        self.assertAlmostEqual(result["auc_at_threshold"], expected)
        self.assertEqual(
            result["aggregation"], "scene_auc_weighted_by_pose_count"
        )


if __name__ == "__main__":
    unittest.main()
