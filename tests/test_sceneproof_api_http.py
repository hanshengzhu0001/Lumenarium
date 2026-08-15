import importlib
import io
import os
import tempfile
import unittest

from PIL import Image


class SceneProofAPIHTTPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        os.environ["SCENEPROOF_API_STATE_ROOT"] = cls.temp.name
        os.environ["SCENEPROOF_WORKER_TOKEN"] = "test-worker-token"
        module = importlib.import_module("sceneproof_api.app")
        cls.module = importlib.reload(module)
        from fastapi.testclient import TestClient
        cls.client = TestClient(cls.module.app)
        image = Image.new("RGB", (1024, 1024), (20, 40, 60))
        cls.image = io.BytesIO()
        image.save(cls.image, "PNG")
        cls.payload = cls.image.getvalue()

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.temp.cleanup()

    def test_full_coordinator_protocol_and_idempotency(self):
        submitted = self.client.post(
            "/v1/jobs",
            files={"image": ("scene.png", self.payload, "image/png")},
            headers={"Idempotency-Key": "http-integration-one"},
        )
        self.assertEqual(202, submitted.status_code)
        body = submitted.json()
        self.assertTrue(body["created"])
        job_id = body["job"]["job_id"]
        self.assertNotIn("input_path", body["job"])
        duplicate = self.client.post(
            "/v1/jobs",
            files={"image": ("scene.png", self.payload, "image/png")},
            headers={"Idempotency-Key": "http-integration-one"},
        )
        self.assertFalse(duplicate.json()["created"])
        self.assertEqual(job_id, duplicate.json()["job"]["job_id"])

        token = {"X-Worker-Token": "test-worker-token"}
        claimed = self.client.post(
            "/internal/claim",
            json={"worker_id": "host-a:gpu0", "lease_seconds": 60},
            headers=token,
        )
        self.assertEqual(job_id, claimed.json()["job"]["job_id"])
        wrong = token | {"X-Worker-ID": "host-b:gpu0"}
        owner = token | {"X-Worker-ID": "host-a:gpu0"}
        self.assertEqual(
            409,
            self.client.get(f"/internal/jobs/{job_id}/input", headers=wrong).status_code,
        )
        self.assertEqual(
            200,
            self.client.get(f"/internal/jobs/{job_id}/input", headers=owner).status_code,
        )
        for name in ("placement.json", "render.png", "evaluation.json", "result.json", "sceneproof-result.zip"):
            uploaded = self.client.post(
                f"/internal/jobs/{job_id}/artifacts/{name}",
                files={"artifact": (name, b"artifact")},
                headers=owner,
            )
            self.assertEqual(200, uploaded.status_code)
        finished = self.client.post(
            "/internal/finish",
            json={
                "job_id": job_id,
                "worker_id": "host-a:gpu0",
                "succeeded": True,
                "result": {"ok": True},
            },
            headers=token,
        )
        self.assertEqual(200, finished.status_code)
        self.assertEqual(
            200, self.client.get(f"/v1/jobs/{job_id}/artifacts/render.png").status_code
        )
        self.assertEqual(
            409, self.client.post(f"/v1/jobs/{job_id}/cancel").status_code
        )

    def test_rejects_wrong_image_size(self):
        image = Image.new("RGB", (512, 512))
        payload = io.BytesIO()
        image.save(payload, "PNG")
        response = self.client.post(
            "/v1/jobs",
            files={"image": ("small.png", payload.getvalue(), "image/png")},
        )
        self.assertEqual(422, response.status_code)


if __name__ == "__main__":
    unittest.main()
