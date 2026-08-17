import unittest

from sceneproof_api.worker import trial_seed


class SceneProofWorkerSeedTest(unittest.TestCase):
    def test_seed_is_stable_and_bounded(self):
        job = {"job_id": "a" * 32, "trial_index": 0}
        self.assertEqual(trial_seed(job), trial_seed(job))
        self.assertGreaterEqual(trial_seed(job), 0)
        self.assertLessEqual(trial_seed(job), 0x7FFFFFFF)

    def test_trials_and_jobs_receive_distinct_seeds(self):
        seeds = {
            trial_seed({"job_id": "a" * 32, "trial_index": 0}),
            trial_seed({"job_id": "a" * 32, "trial_index": 1}),
            trial_seed({"job_id": "b" * 32, "trial_index": 0}),
        }
        self.assertEqual(3, len(seeds))


if __name__ == "__main__":
    unittest.main()
