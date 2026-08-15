import unittest

from sceneba_asset_vlm_build_recoverable import (
    recoverable_samples,
    stratified_cap,
)
from sceneba_asset_vlm_eval_recoverable import evaluate


def candidate(asset, rank):
    return {
        "asset": asset,
        "asset_rank": rank,
        "available": True,
        "s3_winner": {"best_match_vid": rank + 10},
    }


class SceneBAAssetVLMRecoverableTest(unittest.TestCase):
    def test_builder_keeps_only_top1_wrong_gt_in_top3(self):
        oracle = {
            "scenes": {
                "scene_a": {
                    "details": [
                        {
                            "pred_id": "chair_0",
                            "gt_asset": "correct.fbx",
                            "asset_rank": 2,
                        },
                        {
                            "pred_id": "table_0",
                            "gt_asset": "first.fbx",
                            "asset_rank": 1,
                        },
                    ]
                }
            }
        }
        banks = {
            "scene_a": {
                "objects": {
                    "chair_0": {
                        "candidates": [
                            candidate("wrong", 1),
                            candidate("correct", 2),
                            candidate("other", 3),
                        ]
                    },
                    "table_0": {
                        "candidates": [
                            candidate("first", 1),
                            candidate("second", 2),
                            candidate("third", 3),
                        ]
                    },
                }
            }
        }
        samples = recoverable_samples(
            oracle=oracle,
            pose_banks=banks,
            scenes=["scene_a"],
            top_k=3,
        )
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["object_id"], "chair_0")
        self.assertEqual(samples[0]["_answer"]["correct_candidate_index"], 1)

    def test_stratified_cap_round_robins_scenes(self):
        samples = [
            {
                "sample_id": str(index),
                "scene": scene,
                "category": "chair",
                "object_id": f"chair_{index}",
            }
            for index, scene in enumerate(
                ["scene_a", "scene_a", "scene_a", "scene_b", "scene_b"]
            )
        ]
        selected = stratified_cap(samples, 3)
        self.assertEqual({row["scene"] for row in selected}, {"scene_a", "scene_b"})

    def test_evaluation_counts_abstention_as_strict_error(self):
        results = {
            "reports": [
                {
                    "sample_id": "a",
                    "selected_candidate_index": 1,
                    "consensus_confidence": 0.9,
                    "requests": [
                        {"selected_candidate_index": 1},
                        {"selected_candidate_index": 1},
                    ],
                },
                {
                    "sample_id": "b",
                    "selected_candidate_index": None,
                    "consensus_confidence": 0.0,
                    "requests": [
                        {"selected_candidate_index": None},
                        {"selected_candidate_index": None},
                    ],
                },
            ]
        }
        answer = {
            "samples": [
                {
                    "sample_id": "a",
                    "scene": "scene_a",
                    "object_id": "chair_0",
                    "category": "chair",
                    "correct_candidate_index": 1,
                },
                {
                    "sample_id": "b",
                    "scene": "scene_b",
                    "object_id": "table_0",
                    "category": "table",
                    "correct_candidate_index": 2,
                },
            ]
        }
        report = evaluate(results, answer)
        self.assertEqual(report["strict_top3_accuracy"], 0.5)
        self.assertEqual(report["accepted_precision"], 1.0)
        self.assertEqual(
            report["permutation_stability_including_abstain"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
