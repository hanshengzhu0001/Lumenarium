import math
import unittest

try:
    import numpy as np
    from sceneba_moge_noc_witness import (
        apply_similarity,
        robust_similarity_witness,
        weighted_similarity,
    )
    from sceneba_moge_noc_witness_eval import evaluate
except (ImportError, OSError):  # pragma: no cover - minimal local host
    np = None


@unittest.skipIf(np is None, "NumPy and the geometry stack are required")
class SceneBAMoGeNOCWitnessTest(unittest.TestCase):
    def test_weighted_similarity_recovers_known_transform(self):
        source = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        angle = math.radians(35.0)
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        target = 2.5 * (source @ rotation.T) + np.asarray(
            [0.3, -0.8, 4.0]
        )
        weights = np.ones(len(source))
        scale, fitted_rotation, translation = weighted_similarity(
            source, target, weights
        )
        predicted = apply_similarity(
            source, scale, fitted_rotation, translation
        )
        self.assertTrue(np.allclose(predicted, target, atol=1e-9))

    def test_robust_witness_tolerates_one_outlier(self):
        rng = np.random.default_rng(7)
        source = rng.uniform(0.0, 1.0, size=(40, 3))
        target = 1.2 * source + np.asarray([0.2, -0.1, 2.0])
        target[-1] += 10.0
        witness = robust_similarity_witness(
            source,
            target,
            np.ones(len(source)) / len(source),
            seed=3,
        )
        self.assertGreater(witness["inlier_ratio"], 0.9)
        self.assertLess(witness["normalized_weighted_median"], 1e-6)

    def test_predeclared_gate_requires_five_of_eight(self):
        rows = []
        answers = []
        for index in range(8):
            correct_index = 1
            winner = 1 if index < 5 else 0
            losses = [0.4, 0.4, 0.5]
            losses[winner] = 0.1
            rows.append(
                {
                    "sample_id": str(index),
                    "scene": f"scene_{index % 3}",
                    "object_id": f"object_{index}",
                    "category": "object",
                    "candidates": [
                        {
                            "candidate_index": candidate_index,
                            "asset_rank": candidate_index + 1,
                            "witness_loss": loss,
                        }
                        for candidate_index, loss in enumerate(losses)
                    ],
                }
            )
            answers.append(
                {
                    "sample_id": str(index),
                    "correct_candidate_index": correct_index,
                }
            )
        report = evaluate(
            {"rows": rows, "failures": []},
            {"samples": answers},
        )
        self.assertEqual(report["correct_top1"], 5)
        self.assertTrue(report["gates"]["top1_at_least_5_of_8"])


if __name__ == "__main__":
    unittest.main()
