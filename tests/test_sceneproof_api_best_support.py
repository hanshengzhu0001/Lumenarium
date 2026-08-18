import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class V5BestExhaustiveSupportWiringTest(unittest.TestCase):
    def test_best_is_single_fix61_plus_exhaustive_support_job(self):
        app = (ROOT / "sceneproof_api" / "app.py").read_text(encoding="utf-8")
        runner = (
            ROOT / "scripts" / "run_sceneproof_frozen_single_job_fix115.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("for trial_index in range(3)", app)
        self.assertIn('initial_state="queued"', app)
        self.assertIn("s4_fix61_then_exhaustive_support", runner)
        self.assertIn("IMAGINARIUM_SCENEPROOF_SPARSE_AUDIT_ALL_OBJECTS=1", runner)
        self.assertIn("SCENEPROOF_API_BEST_MAXIMUM_DROP_M:-3.0", runner)

    def test_s4_exposes_transactional_all_object_audit(self):
        source = (
            ROOT / "modules" / "S4_blender_layout_and_corr.py"
        ).read_text(encoding="utf-8")
        self.assertIn("audit_all_objects=False", source)
        self.assertIn("no_true_surface_below_current_footprint", source)
        self.assertIn("certified_exhaustive_vertical_first_contact_drop", source)
        self.assertIn("rollback_restored=True", source)
        self.assertIn("exhaustive_support_routed_geometry", source)


if __name__ == "__main__":
    unittest.main()
