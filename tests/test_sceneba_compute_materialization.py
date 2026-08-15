import unittest

from sceneba_finalize_compute_frontier import metric


class SceneBAComputeMaterializationTest(unittest.TestCase):
    def test_metric_accepts_fallback_paths(self):
        document = {"versions": {"v": {"aggregate": {"macro": 0.75}}}}
        self.assertEqual(
            metric(
                document,
                "v",
                ("aggregate", "missing"),
                ("aggregate", "macro"),
            ),
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
