import unittest
import os
import tempfile
from pathlib import Path

try:
    import numpy as np
    import sceneba_yaw_vlm_verifier as verifier
    import sceneba_yaw_vlm_audit as audit
except ImportError:
    np = None
    verifier = None
    audit = None


@unittest.skipIf(np is None or verifier is None, "NumPy/PIL required")
class SceneBAYawVLMVerifierTest(unittest.TestCase):
    def test_dotenv_loader_sets_missing_values_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "SCENEBA_TEST_SECRET='from-file'\n"
                "SCENEBA_TEST_KEEP=from-file\n",
                encoding="utf-8",
            )
            os.environ["SCENEBA_TEST_KEEP"] = "existing"
            os.environ.pop("SCENEBA_TEST_SECRET", None)
            try:
                verifier.load_dotenv_file(path)
                self.assertEqual(
                    os.environ["SCENEBA_TEST_SECRET"], "from-file"
                )
                self.assertEqual(
                    os.environ["SCENEBA_TEST_KEEP"], "existing"
                )
            finally:
                os.environ.pop("SCENEBA_TEST_SECRET", None)
                os.environ.pop("SCENEBA_TEST_KEEP", None)

    def test_response_parser_enforces_labels_and_confidence_range(self):
        parsed = verifier.parse_response(
            '{"choice":"B","confidence":1.7,"observable":true}',
            {"A", "B", "ABSTAIN"},
        )
        self.assertEqual(parsed["choice"], "B")
        self.assertEqual(parsed["confidence"], 1.0)
        invalid = verifier.parse_response(
            '{"choice":"Z","confidence":0.9,"observable":true}',
            {"A", "B", "ABSTAIN"},
        )
        self.assertEqual(invalid["choice"], "ABSTAIN")

    def test_nearest_template_view_uses_camera_relative_orientation(self):
        pose = np.eye(4)
        world_to_camera = np.eye(4)
        cameras = np.stack([np.eye(4), np.eye(4)])
        cameras[1, :3, :3] = np.asarray(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        view, angle = verifier.nearest_template_view(
            pose, world_to_camera, cameras
        )
        self.assertEqual(view, 0)
        self.assertAlmostEqual(angle, 0.0)

    def test_contact_sheet_protocol_keeps_fixed_tile_yaw_order(self):
        source = Path("sceneba_yaw_vlm_verifier.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '"permutation_protocol": "fixed_tiles_randomized_labels"',
            source,
        )
        self.assertIn('"tile_yaw_order": tile_yaw_order', source)
        self.assertNotIn("random.Random(seed).shuffle(shuffled)", source)


@unittest.skipIf(audit is None, "Audit dependencies required")
class SceneBAYawVLMAuditTest(unittest.TestCase):
    def test_quotient_error_mods_out_equivalent_half_turn(self):
        row = {
            "current_rotation_error_deg": 170.0,
            "candidates": [
                {
                    "yaw_deg": 0.0,
                    "rotation_error_deg": 170.0,
                    "shape_chamfer_with_current": 0.0,
                },
                {
                    "yaw_deg": 180.0,
                    "rotation_error_deg": 10.0,
                    "shape_chamfer_with_current": 0.005,
                },
            ],
        }
        self.assertAlmostEqual(audit.quotient_error(row), 10.0)

    def test_quotient_error_keeps_observable_mode_distinct(self):
        row = {
            "current_rotation_error_deg": 170.0,
            "candidates": [
                {
                    "yaw_deg": 0.0,
                    "rotation_error_deg": 170.0,
                    "shape_chamfer_with_current": 0.0,
                },
                {
                    "yaw_deg": 180.0,
                    "rotation_error_deg": 10.0,
                    "shape_chamfer_with_current": 0.2,
                },
            ],
        }
        self.assertAlmostEqual(audit.quotient_error(row), 170.0)


if __name__ == "__main__":
    unittest.main()
