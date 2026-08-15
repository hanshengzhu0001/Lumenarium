import unittest

from sceneba_compute_frontier import no_harm


class SceneBAComputeFrontierTest(unittest.TestCase):
    def test_noninferior_candidate_passes(self):
        reference = {
            "local_realizability": 0.8,
            "collision_overlap_fraction": 0.02,
            "support_contact_gap_m": 0.01,
            "support_containment_error_m": 0.01,
            "support_footprint_overlap_ratio": 0.95,
            "inside_containment_error_m": None,
            "plane_contact_gap_m": None,
            "semantic_error": 0.1,
        }
        candidate = dict(reference, local_realizability=0.79)
        passed, reasons = no_harm(
            candidate,
            reference,
            local_margin=0.02,
            collision_margin=0.01,
            contact_margin_m=0.02,
            plane_margin_m=0.02,
            semantic_margin=0.1,
        )
        self.assertTrue(passed)
        self.assertEqual(reasons, [])

    def test_collision_harm_rejects_candidate(self):
        reference = {
            "local_realizability": 0.8,
            "collision_overlap_fraction": 0.0,
        }
        candidate = {
            "local_realizability": 0.9,
            "collision_overlap_fraction": 0.2,
        }
        passed, reasons = no_harm(
            candidate,
            reference,
            local_margin=0.02,
            collision_margin=0.01,
            contact_margin_m=0.02,
            plane_margin_m=0.02,
            semantic_margin=0.1,
        )
        self.assertFalse(passed)
        self.assertIn("collision_overlap_fraction", reasons)


if __name__ == "__main__":
    unittest.main()
