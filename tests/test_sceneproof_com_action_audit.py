import unittest

from sceneproof_com_action_audit_fix78 import classify_support_action


class SceneProofCOMActionAuditTest(unittest.TestCase):
    def test_routes_only_witnessed_instability_to_projection(self):
        self.assertEqual(
            classify_support_action(
                {
                    "certificate_status": "certified",
                    "stability_class": "unstable",
                    "declared_parent_contact_present": True,
                    "intrinsic_child_contact_margin_m": 0.02,
                    "declared_parent_surface_margin_m": -0.03,
                }
            ),
            "com_projection_candidate",
        )
        self.assertEqual(
            classify_support_action(
                {
                    "certificate_status": "certified",
                    "stability_class": "unstable",
                    "declared_parent_contact_present": True,
                    "intrinsic_child_contact_margin_m": -0.03,
                    "declared_parent_surface_margin_m": 1.0,
                }
            ),
            "local_gravity_settle_probe_candidate",
        )
        self.assertEqual(
            classify_support_action(
                {
                    "certificate_status": "certified",
                    "stability_class": "unstable",
                    "declared_parent_contact_present": False,
                }
            ),
            "abstain_unproven_support",
        )

    def test_missing_contact_routes_to_settle_probe_not_projection(self):
        self.assertEqual(
            classify_support_action(
                {"reason": "no_mesh_or_voxel_horizontal_contact_patch"}
            ),
            "local_gravity_settle_probe_candidate",
        )


if __name__ == "__main__":
    unittest.main()
