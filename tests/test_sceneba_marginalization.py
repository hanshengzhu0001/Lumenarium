import unittest

from sceneba_marginal_factor_audit import evaluate, logmeanexp
from sceneba_marginal_factor_table import select_hypotheses


class SceneBAMarginalizationTest(unittest.TestCase):
    def test_hypothesis_cap_balances_two_views_and_two_yaws(self):
        rows = [
            {
                "view_id": view,
                "view_rank": rank,
                "yaw_deg": yaw,
                "source": "s3_winner" if rank == 0 and yaw == 0 else "proposal",
                "pose_matrix_for_blender": [[1, 0, 0, 0]] * 4,
            }
            for rank, view in enumerate((10, 20, 30))
            for yaw in (0, 90, 180, 270)
        ]
        selected = select_hypotheses(rows)
        self.assertEqual(len(selected), 4)
        self.assertEqual({row["view_id"] for row in selected}, {10, 20})
        self.assertEqual({row["yaw_deg"] for row in selected}, {0, 180})

    def test_logmeanexp_does_not_reward_duplicate_hypotheses(self):
        single = logmeanexp([0.25], temperature=1.0)
        duplicated = logmeanexp([0.25, 0.25], temperature=1.0)
        self.assertAlmostEqual(single, duplicated)

    def test_evaluation_applies_predeclared_gate(self):
        rows = []
        answers = []
        for index in range(8):
            sample_id = str(index)
            answers.append(
                {
                    "sample_id": sample_id,
                    "scene": f"s{index % 2}",
                    "category": f"c{index % 2}",
                    "correct_candidate_index": 0,
                }
            )
            for candidate, loss in ((0, 0.0), (1, 1.0), (2, 2.0)):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "candidate_index": candidate,
                        "asset_rank": candidate + 1,
                        "factors": {"silhouette": loss},
                    }
                )
        config = {
            "temperature": 1.0,
            "rules": {"primary": {"silhouette": 1.0}},
            "retrieval_prior_rules": [],
            "primary_rule": "primary",
            "go_no_go": {
                "minimum_correct_of_8": 5,
                "minimum_mrr": 0.75,
                "minimum_scene_coverage": 2,
                "minimum_category_coverage": 2,
            },
        }
        report = evaluate(
            {"rows": rows, "failures": []}, config, {"samples": answers}
        )
        self.assertEqual(report["decision"], "continue_stage1")


if __name__ == "__main__":
    unittest.main()
