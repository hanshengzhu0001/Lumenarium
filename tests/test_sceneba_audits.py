import unittest

import numpy as np

import sceneba_paired_audit as paired
import sceneba_build_pose_bank as pose_bank
import sceneba_topk_oracle as oracle


class PairedBootstrapTest(unittest.TestCase):
    def test_version_scenes_supports_gt_and_physical_schemas(self):
        gt = {"scenes": {"v": {"room": {"x": 1}}}, "versions": {"v": {}}}
        physical = {"versions": {"v": {"scenes": {"room": {"x": 2}}}}}
        self.assertEqual(paired.version_scenes(gt, "v")["room"]["x"], 1)
        self.assertEqual(paired.version_scenes(physical, "v")["room"]["x"], 2)
        self.assertEqual(oracle.version_scenes(gt, "v")["room"]["x"], 1)

    def test_scene_pairing_and_noninferiority(self):
        baseline = {
            "a": {"translation_aligned": {"auc_at_threshold": 0.4, "n": 2}},
            "b": {"translation_aligned": {"auc_at_threshold": 0.6, "n": 2}},
        }
        candidate = {
            "a": {"translation_aligned": {"auc_at_threshold": 0.5, "n": 2}},
            "b": {"translation_aligned": {"auc_at_threshold": 0.7, "n": 2}},
        }
        result = paired.paired_bootstrap(
            ["a", "b"],
            baseline,
            candidate,
            lambda scene: paired.gt_value(scene, "translation_auc05"),
            samples=1000,
            confidence=0.95,
            seed=3,
            noninferiority_margin=-0.005,
        )
        self.assertAlmostEqual(result["delta_candidate_minus_baseline"], 0.1)
        self.assertTrue(result["noninferior"])
        self.assertEqual(result["paired_scene_count"], 2)

    def test_physical_macro_excludes_boundary(self):
        scene = {
            "families": {
                "collision": {"score": 0.8, "n": 1},
                "support": {"score": 0.6, "n": 1},
                "plane": {"score": 0.4, "n": 1},
                "boundary": {"score": 0.0, "n": 1},
                "semantic": {"score": 1.0, "n": 1},
            }
        }
        value, weight = paired.physical_value(scene, "physical_macro4")
        self.assertAlmostEqual(value, 0.7)
        self.assertEqual(weight, 1.0)


class TopKOracleTest(unittest.TestCase):
    def test_asset_candidate_parsing_and_rank(self):
        retrieval = {
            "chair_0": [
                ["wrong.lsph", 2.0],
                {"asset_name": "SM_Chair_01.fbx"},
                "third",
            ]
        }
        candidates = oracle.asset_candidates(retrieval, "chair_0")
        self.assertEqual(candidates, ["wrong.lsph", "SM_Chair_01.fbx", "third"])
        self.assertEqual(oracle.first_rank(candidates, "SM_Chair_01.fbx"), 2)

    def test_parent_candidates_keep_current_first(self):
        info = {"supported": "floor_0", "parent_candidates": ["table_0"]}
        graph = {"supported_candidates": [{"parent": "desk_0"}]}
        self.assertEqual(
            oracle.parent_candidates(info, graph),
            ["floor_0", "table_0", "desk_0"],
        )

    def test_yaw_pose_mode_oracle_can_fix_180_degree_mode(self):
        pred = np.eye(4)
        gt = np.eye(4)
        gt[:3, :3] = oracle.yaw_matrix(180)
        current, best, translation = oracle.pose_mode_errors(
            pred,
            gt,
            {"available": False},
            [0, 90, 180, 270],
        )
        self.assertAlmostEqual(current, 180.0)
        self.assertAlmostEqual(best, 0.0, places=6)
        self.assertAlmostEqual(translation, 0.0)

    def test_scene_oracle_reestimates_alignment_from_audited_pose(self):
        gt_objects = {}
        pred_objects = {}
        matches = []
        for index, point in enumerate(
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        ):
            pred_id = f"chair_{index}"
            gt_id = f"gt_{index}"
            pm = np.eye(4)
            pm[:3, 3] = point
            gm = np.eye(4)
            gm[:3, 3] = 2.0 * np.asarray(point) + np.asarray([5.0, 7.0, 0.0])
            pred_objects[pred_id] = {
                "pose_matrix_for_blender": pm.tolist(),
                "supported": "floor_0",
            }
            gt_objects[gt_id] = {
                "type": "MESH",
                "matrix_world_4x4": gm.tolist(),
                "parent": "floor_0",
                "fbx_name": f"asset_{index}.fbx",
            }
            matches.append({"pred_id": pred_id, "gt_id": gt_id})
        result = oracle.evaluate_scene(
            scene="synthetic",
            match_scene={
                "matches": matches,
                # Deliberately wrong transform: it must not be reused.
                "alignment": {
                    "available": True,
                    "scale": 99,
                    "rotation": np.eye(3).tolist(),
                    "translation": [99, 99, 99],
                },
            },
            meta={"objects": gt_objects},
            placement={"obj_info": pred_objects},
            retrieval={
                f"chair_{i}": [[f"asset_{i}.fbx", 1.0]] for i in range(3)
            },
            graph={},
            geometry_placement=None,
            pose_bank=None,
            yaw_offsets=[0],
            semantic_parent=True,
        )
        self.assertTrue(result["alignment"]["available"])
        self.assertAlmostEqual(result["alignment"]["scale"], 2.0)
        self.assertTrue(all(error < 1e-8 for error in result["translation_current"]))

    def test_pose_bank_joint_oracle_uses_full_denominator(self):
        pred_objects = {}
        gt_objects = {}
        matches = []
        for index, x in enumerate((0.0, 1.0, 2.0)):
            pred_id, gt_id = f"chair_{index}", f"gt_{index}"
            pm = np.eye(4)
            pm[0, 3] = x + (0.4 if index == 0 else 0.1)
            gm = np.eye(4)
            gm[0, 3] = x
            pred_objects[pred_id] = {
                "pose_matrix_for_blender": pm.tolist(),
                "supported": "floor_0",
            }
            gt_objects[gt_id] = {
                "type": "MESH",
                "matrix_world_4x4": gm.tolist(),
                "parent": "floor_0",
                "fbx_name": f"asset_{index}.fbx",
            }
            matches.append({"pred_id": pred_id, "gt_id": gt_id})
        # Only object 0 has its exact GT asset in the bank.  The other two
        # must remain in the oracle denominator with their current errors.
        corrected = np.eye(4)
        bank = {
            "objects": {
                "chair_0": {
                    "candidates": [
                        {
                            "asset": "asset_0.fbx",
                            "hypotheses": [
                                {"pose_matrix_for_blender": corrected.tolist()}
                            ],
                        }
                    ]
                }
            }
        }
        result = oracle.evaluate_scene(
            scene="synthetic",
            match_scene={"matches": matches},
            meta={"objects": gt_objects},
            placement={"obj_info": pred_objects},
            retrieval={
                f"chair_{i}": [[f"asset_{i}.fbx", 1.0]] for i in range(3)
            },
            graph={},
            geometry_placement=None,
            pose_bank=bank,
            yaw_offsets=[0],
            semantic_parent=True,
        )
        self.assertEqual(result["asset_pose_bank_available_count"], 1)
        self.assertEqual(len(result["asset_pose_bank_translation_oracle"]), 3)
        self.assertEqual(
            result["asset_pose_bank_translation_oracle"][1:],
            result["translation_current"][1:],
        )

    def test_pose_bank_schema_and_rank_selection(self):
        retrieval = {
            "chair_0": [["asset_a", 0.9], ["asset_b", 0.8], ["asset_a", 0.7]]
        }
        candidates = pose_bank.ranked_candidates(retrieval, 3)
        self.assertEqual([x["asset"] for x in candidates["chair_0"]], ["asset_a", "asset_b"])
        rank_results = {
            0: {
                "chair_0": {
                    "pose_matrix_for_blender": np.eye(4).tolist(),
                    "sceneba_pose_hypotheses": [
                        {
                            "view_id": 1,
                            "yaw_deg": 0,
                            "pose_matrix_for_blender": np.eye(4).tolist(),
                        }
                    ],
                }
            }
        }
        bank = pose_bank.assemble_bank(
            scene="room",
            source_version="v4_deepsearch",
            candidates=candidates,
            rank_results=rank_results,
            rank_runtime={0: 1.0},
            rank_peak_mib={0: 123.0},
            skipped={},
            top_k_assets=3,
            top_k_views=3,
            yaw_offsets=[0, 90, 180, 270],
        )
        pose_bank.validate_bank(bank)
        self.assertEqual(bank["hypothesis_count"], 1)


if __name__ == "__main__":
    unittest.main()
