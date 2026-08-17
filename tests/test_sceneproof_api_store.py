import tempfile
import unittest
from pathlib import Path

from sceneproof_api.store import JobStore
from sceneproof_api.worker import STAGE_PATTERN, requires_cold_rerun
from sceneproof_cold_start_selector import pose_reprojection_proxy


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

    def test_worker_distinguishes_profile_and_cold_reruns(self):
        self.assertFalse(requires_cold_rerun({
            "idempotency_key": "sha256:x:rerun:profile:abc",
        }))
        self.assertTrue(requires_cold_rerun({
            "idempotency_key": "sha256:x:rerun:cold:abc",
        }))
        self.assertFalse(requires_cold_rerun({"idempotency_key": None}))

    def test_best_parent_waits_while_children_are_claimable(self):
        parent, _ = self.store.create(
            release_id="r1", input_path="in", artifact_dir="best",
            idempotency_key="best", profile="best", initial_state="waiting",
        )
        for trial_index in range(3):
            self.store.create(
                release_id="r1", input_path="in",
                artifact_dir=f"trial-{trial_index}",
                idempotency_key=f"trial-{trial_index}", profile="fast",
                parent_job_id=parent["job_id"], trial_index=trial_index,
            )
        children = self.store.children(parent["job_id"])
        self.assertEqual([0, 1, 2], [row["trial_index"] for row in children])
        claimed = self.store.claim(worker_id="host-a:gpu0", lease_seconds=60)
        self.assertEqual(0, claimed["trial_index"])
        self.assertNotEqual(parent["job_id"], claimed["job_id"])

    def test_pose_proxy_uses_only_source_8000px_equivalent_objects(self):
        proxy = pose_reprojection_proxy({
            "sceneproof_mesh_visibility_audit": {
                "resolution": [256, 256],
                "objects": {
                    "large": {
                        "status": "measured", "observed_mask_pixels": 500,
                        "iou": 0.5, "precision": 0.75, "recall": 0.6,
                    },
                    "small": {
                        "status": "measured", "observed_mask_pixels": 499,
                        "iou": 1.0, "precision": 1.0, "recall": 1.0,
                    },
                },
            },
        })
        self.assertEqual(["large"], list(proxy["objects"]))
        self.assertAlmostEqual(2 * 0.75 * 0.6 / 1.35,
                               proxy["objects"]["large"]["f1"])

    def test_pose_proxy_reads_guarded_candidate_envelope(self):
        proxy = pose_reprojection_proxy({
            "sceneproof_cold_start_pose_proxy_audit": {
                "source": "guarded_candidate_pre_certificate",
                "final_pose_exact": False,
                "mesh_visibility_audit": {
                    "resolution": [256, 256],
                    "objects": {
                        "chair": {
                            "status": "measured",
                            "observed_mask_pixels": 500,
                            "iou": 0.5,
                            "precision": 0.5,
                            "recall": 0.5,
                        }
                    },
                },
            }
        })
        self.assertEqual(["chair"], list(proxy["objects"]))
        self.assertEqual("guarded_candidate_pre_certificate",
                         proxy["audit_source"])
        self.assertFalse(proxy["final_pose_exact"])


if __name__ == "__main__":
    unittest.main()
