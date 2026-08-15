import unittest

from sceneproof_local_settle_oracle_fix80 import (
    classify_probe,
    restoration_certificate,
)


class SceneProofLocalSettleOracleTest(unittest.TestCase):
    def test_promising_requires_restore_collision_and_support_certificates(self):
        row = {
            "status": "measured",
            "incumbent_restored": True,
            "new_collision_object_ids": [],
            "after_support": {
                "certificate_status": "certified",
                "stability_class": "stable",
                "declared_parent_contact_present": True,
            },
        }
        self.assertEqual(
            classify_probe(row),
            "locally_promising_requires_full_component_gates",
        )

    def test_new_collision_fails_closed(self):
        row = {
            "status": "measured",
            "incumbent_restored": True,
            "new_collision_object_ids": ["desk_0"],
            "after_support": {
                "certificate_status": "certified",
                "stability_class": "stable",
                "declared_parent_contact_present": True,
            },
        }
        self.assertEqual(classify_probe(row), "rejected_new_collision")

    def test_restore_failure_is_never_promoted(self):
        self.assertEqual(
            classify_probe(
                {
                    "status": "measured",
                    "incumbent_restored": False,
                    "new_collision_object_ids": [],
                }
            ),
            "unsafe_restoration_failure",
        )

    def test_float32_two_ulp_restore_error_is_certified(self):
        restored, tolerance = restoration_certificate(
            {
                "before_pose_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "maximum_restoration_error": 2.384185791015625e-7,
            }
        )
        self.assertTrue(restored)
        self.assertGreaterEqual(tolerance, 2.384185791015625e-7)


if __name__ == "__main__":
    unittest.main()
