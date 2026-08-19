import importlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from sceneproof_api import demo_pin
from sceneproof_api.worker import PIPELINE_PROFILE, resolve_seed, trial_seed


ROOT = Path(__file__).resolve().parents[1]


class DemoPinLoaderTest(unittest.TestCase):
    def setUp(self):
        os.environ.pop(demo_pin.SEED_ENVIRONMENT_VARIABLE, None)
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "DEMO_PIN.json"
        self.addCleanup(self.temp.cleanup)

    def _write(self, document):
        self.path.write_text(json.dumps(document), encoding="utf-8")

    def test_unset_and_malformed_pins_are_reported_as_absent(self):
        self.assertEqual({}, demo_pin.load_demo_pin(self.path))  # missing file
        for document in ({"seed": None}, {"seed": "not-a-number"},
                         {"seed": -1}, {"seed": demo_pin.SEED_MAXIMUM + 1},
                         {"seed": True}, ["seed"]):
            self._write(document)
            self.assertEqual({}, demo_pin.load_demo_pin(self.path), document)

    def test_valid_pin_carries_its_provenance(self):
        self._write({"seed": 123456, "scene_id": "a" * 32, "job_id": "b" * 32})
        pin = demo_pin.load_demo_pin(self.path)
        self.assertEqual(123456, pin["seed"])
        self.assertEqual("a" * 32, pin["scene_id"])
        self.assertEqual(str(self.path), pin["source"])

    def test_environment_overrides_the_file(self):
        self._write({"seed": 111})
        with mock.patch.dict(
            os.environ, {demo_pin.SEED_ENVIRONMENT_VARIABLE: "222"}
        ):
            self.assertEqual(222, demo_pin.demo_seed(self.path))
        self.assertEqual(111, demo_pin.demo_seed(self.path))

    def test_pin_survives_a_byte_order_mark(self):
        # An editor on the host can add a BOM; treating that as "unset" would
        # disable the profile silently.
        self.path.write_text(json.dumps({"seed": 555}), encoding="utf-8-sig")
        self.assertEqual(555, demo_pin.demo_seed(self.path))

    def test_shipped_pin_defaults_to_disabled(self):
        # The repository must not carry a seed: a pin is host state, and an
        # inherited one would silently make demo runs reproduce someone else.
        document = json.loads(
            (ROOT / "sceneproof_api" / "DEMO_PIN.json").read_text(encoding="utf-8")
        )
        self.assertIsNone(document["seed"])


class DemoSeedRoutingTest(unittest.TestCase):
    def setUp(self):
        os.environ.pop(demo_pin.SEED_ENVIRONMENT_VARIABLE, None)

    def test_demo_uses_the_pin_and_other_profiles_do_not(self):
        job = {"job_id": "a" * 32, "profile": "demo"}
        with mock.patch("sceneproof_api.worker.demo_seed", return_value=4242):
            self.assertEqual(4242, resolve_seed(job))
            self.assertEqual(
                trial_seed(job), resolve_seed({**job, "profile": "best"})
            )

    def test_demo_without_a_pin_fails_instead_of_falling_back(self):
        with mock.patch("sceneproof_api.worker.demo_seed", return_value=None):
            with self.assertRaises(RuntimeError) as raised:
                resolve_seed({"job_id": "a" * 32, "profile": "demo"})
        self.assertIn("pin_sceneproof_demo_seed", str(raised.exception))

    def test_demo_is_translated_before_reaching_the_pipeline(self):
        self.assertEqual("best", PIPELINE_PROFILE["demo"])
        runner = (
            ROOT / "scripts" / "run_sceneproof_frozen_single_job_fix115.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('case "$profile" in fast|medium|best)', runner)
        # The runner does contain the word demo (the demo/ image directory), so
        # the leak to guard against is specifically a fourth accepted profile.
        self.assertNotIn("|demo)", runner)

    def test_user_interface_exposes_the_profile_and_its_pin(self):
        markup = (ROOT / "sceneproof_api" / "ui.html").read_text(encoding="utf-8")
        self.assertIn('<option value="demo">', markup)
        self.assertIn("demo-pin", markup)
        self.assertIn("pinned trial seed", markup)


class DemoProfileHTTPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.pop(demo_pin.SEED_ENVIRONMENT_VARIABLE, None)
        cls.temp = tempfile.TemporaryDirectory()
        os.environ["SCENEPROOF_API_STATE_ROOT"] = cls.temp.name
        os.environ["SCENEPROOF_WORKER_TOKEN"] = "test-worker-token"
        cls.module = importlib.reload(
            importlib.import_module("sceneproof_api.app")
        )
        from fastapi.testclient import TestClient
        cls.client = TestClient(cls.module.app)
        buffer = io.BytesIO()
        Image.new("RGB", (1024, 1024), (30, 60, 90)).save(buffer, "PNG")
        cls.payload = buffer.getvalue()
        cls.pin_path = Path(cls.temp.name) / "DEMO_PIN.json"

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.temp.cleanup()

    def _submit(self, key):
        return self.client.post(
            "/v1/jobs",
            files={"image": ("scene.png", self.payload, "image/png")},
            data={"profile": "demo"},
            headers={"Idempotency-Key": key},
        )

    def test_demo_is_refused_while_no_seed_is_pinned(self):
        with mock.patch.object(demo_pin, "PIN_PATH", self.pin_path):
            response = self._submit("demo-without-pin")
        self.assertEqual(422, response.status_code)
        self.assertIn("pin_sceneproof_demo_seed", response.json()["detail"])

    def test_pinned_demo_submissions_are_accepted_and_never_replayed(self):
        self.pin_path.write_text(json.dumps({"seed": 987654}), encoding="utf-8")
        with mock.patch.object(demo_pin, "PIN_PATH", self.pin_path):
            first = self._submit("demo-repeat")
            second = self._submit("demo-repeat")
            release = self.client.get("/v1/releases/current").json()
        self.assertEqual(202, first.status_code)
        self.assertEqual(987654, first.json()["pinned_seed"])
        self.assertEqual("demo", first.json()["job"]["profile"])
        self.assertTrue(first.json()["created"] and second.json()["created"])
        self.assertNotEqual(
            first.json()["job"]["job_id"], second.json()["job"]["job_id"]
        )
        self.assertEqual(987654, release["demo_pin"]["seed"])
        self.assertIn("demo", release["profiles"])

    def test_unknown_profiles_are_still_rejected(self):
        response = self.client.post(
            "/v1/jobs",
            files={"image": ("scene.png", self.payload, "image/png")},
            data={"profile": "showcase"},
        )
        self.assertEqual(422, response.status_code)


if __name__ == "__main__":
    unittest.main()
