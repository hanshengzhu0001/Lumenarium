import tempfile
import unittest
from pathlib import Path

from sceneproof_api.store import JobStore
from sceneproof_api.worker import STAGE_PATTERN


class SceneProofJobStoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = JobStore(Path(self.temp.name) / "jobs.sqlite3")

    def tearDown(self):
        self.temp.cleanup()

    def test_idempotent_create_returns_original_job(self):
        first, created = self.store.create(
            release_id="r1", input_path="in", artifact_dir="out",
            idempotency_key="same",
        )
        second, created_again = self.store.create(
            release_id="r1", input_path="other", artifact_dir="other",
            idempotency_key="same",
        )
        self.assertTrue(created)
        self.assertFalse(created_again)
        self.assertEqual(first["job_id"], second["job_id"])

    def test_claim_heartbeat_and_finish_require_same_worker(self):
        job, _ = self.store.create(
            release_id="r1", input_path="in", artifact_dir="out",
            idempotency_key=None,
        )
        claimed = self.store.claim(worker_id="host-a:gpu0", lease_seconds=60)
        self.assertEqual(job["job_id"], claimed["job_id"])
        self.assertFalse(self.store.heartbeat(
            job_id=job["job_id"], worker_id="host-b:gpu0", stage="S1",
            progress=0.2, lease_seconds=60,
        ))
        self.assertTrue(self.store.heartbeat(
            job_id=job["job_id"], worker_id="host-a:gpu0", stage="S1",
            progress=0.2, lease_seconds=60,
        ))
        self.assertTrue(self.store.finish(
            job_id=job["job_id"], worker_id="host-a:gpu0",
            succeeded=True, result={"render": "final.png"},
        ))
        self.assertEqual("succeeded", self.store.get(job["job_id"])["state"])

    def test_cancel_revokes_worker_lease_and_is_terminal(self):
        job, _ = self.store.create(
            release_id="r1", input_path="in", artifact_dir="out",
            idempotency_key=None,
        )
        self.store.claim(worker_id="host-a:gpu0", lease_seconds=60)
        self.assertTrue(self.store.owns_active_lease(
            job_id=job["job_id"], worker_id="host-a:gpu0"
        ))
        self.assertTrue(self.store.cancel(job["job_id"]))
        self.assertFalse(self.store.owns_active_lease(
            job_id=job["job_id"], worker_id="host-a:gpu0"
        ))
        self.assertFalse(self.store.finish(
            job_id=job["job_id"], worker_id="host-a:gpu0", succeeded=True
        ))

    def test_worker_stage_markers_are_machine_parseable(self):
        match = STAGE_PATTERN.search(
            "SCENEPROOF_API_STAGE=sceneproof_fix114 PROGRESS=0.75\n"
        )
        self.assertEqual("sceneproof_fix114", match.group(1))
        self.assertEqual(0.75, float(match.group(2)))


if __name__ == "__main__":
    unittest.main()
