import unittest

try:
    import numpy as np
    import sceneba_yaw_feature_audit as audit
except ImportError:
    np = None
    audit = None


@unittest.skipIf(np is None or audit is None, "NumPy required")
class SceneBAYawFeatureAuditTest(unittest.TestCase):
    def test_structural_gate_rejects_semantic_and_plane_changes(self):
        base = {
            "yaw_deg": 90,
            "symmetry_equivalent_to_current": False,
            "collision_increase": 0.0,
            "support_degradation_m": 0.0,
            "plane_supported": False,
            "semantic_orientation_affected": False,
        }
        self.assertTrue(audit.constraint_safe(base, "structural"))
        self.assertFalse(
            audit.constraint_safe(
                dict(base, semantic_orientation_affected=True),
                "structural",
            )
        )
        self.assertFalse(
            audit.constraint_safe(
                dict(base, plane_supported=True),
                "structural",
            )
        )

    def test_feature_vector_uses_candidate_improvements(self):
        current = {
            "silhouette_loss": 0.5,
            "boundary_chamfer": 0.4,
            "depth_loss": 0.3,
            "dino_orientation_loss": 0.2,
            "dino_reprojection_loss": 0.2,
            "composite_loss": 0.4,
        }
        candidate = {
            "yaw_deg": 180,
            "silhouette_loss": 0.4,
            "boundary_chamfer": 0.3,
            "depth_loss": 0.2,
            "dino_orientation_loss": 0.1,
            "dino_reprojection_loss": 0.1,
            "composite_loss": 0.3,
            "collision_increase": 0,
            "support_degradation_m": 0,
            "mode_iou_with_current": 0.5,
        }
        features = audit.feature_vector(candidate, current)
        self.assertTrue(np.all(features[:6] > 0))
        self.assertEqual(features[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
