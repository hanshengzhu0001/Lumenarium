import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

import eval_physical_realizability as physical


def pose(x=0.0, y=0.0, z=0.0):
    result = np.eye(4)
    result[:3, 3] = [x, y, z]
    return result


def bbox(center, length):
    center = np.asarray(center, dtype=float)
    half = 0.5 * np.asarray(length, dtype=float)
    return [
        (center + np.asarray([x, y, z]) * half).tolist()
        for x in (-1, 1)
        for y in (-1, 1)
        for z in (-1, 1)
    ]


def info(matrix, length, *, supported=None):
    center = np.asarray(matrix)[:3, 3]
    return {
        "pose_matrix_for_blender": np.asarray(matrix).tolist(),
        "bbox": bbox(center, length),
        "length": list(length),
        "supported": supported,
        "SpatialRel": "on" if supported else None,
    }


def args():
    return SimpleNamespace(
        collision_volume_tolerance=1e-6,
        collision_fraction_tolerance=0.05,
        contact_tolerance=0.05,
        containment_tolerance=0.05,
        support_overlap_tolerance=0.9,
        plane_tolerance=0.05,
        plane_orientation_tolerance=15.0,
        boundary_tolerance=0.05,
        semantic_angle_tolerance=20.0,
        distance_tolerance=0.1,
        collision_policy="auto",
    )


class PhysicalRealizabilityTest(unittest.TestCase):
    def setUp(self):
        floor = info(pose(0, 0, -0.05), [10, 10, 0.1])
        first = info(pose(-2, 0, 0.5), [1, 1, 1], supported="floor_0")
        second = info(pose(2, 0, 0.5), [1, 1, 1], supported="floor_0")
        self.source = {
            "reference_obj": "floor_0",
            "obj_info": {
                "floor_0": floor,
                "box_0": first,
                "box_1": second,
            },
        }

    def test_perfect_floor_supported_scene_scores_one(self):
        metrics, objects = physical.evaluate_scene(
            self.source,
            self.source,
            args(),
        )
        self.assertEqual(metrics["unintended_collision_pairs"], 0)
        self.assertAlmostEqual(metrics["families"]["collision"]["score"], 1.0)
        self.assertAlmostEqual(metrics["families"]["support"]["score"], 1.0)
        self.assertAlmostEqual(metrics["families"]["boundary"]["score"], 1.0)
        self.assertAlmostEqual(metrics["macro_realizability"], 1.0)
        self.assertEqual(
            metrics["headline_families"],
            ["collision", "support", "plane", "semantic"],
        )
        self.assertAlmostEqual(metrics["headline_macro_realizability"], 1.0)
        self.assertAlmostEqual(metrics["headline_critical_realizability"], 1.0)
        self.assertTrue(all(row["local_realizability"] == 1.0 for row in objects))

    def test_list_valued_support_parent_is_normalized(self):
        source = json.loads(json.dumps(self.source))
        source["obj_info"]["box_0"]["supported"] = ["floor_0"]
        metrics, _ = physical.evaluate_scene(source, source, args())
        self.assertEqual(metrics["missing_support_parents"], 0)
        self.assertAlmostEqual(metrics["families"]["support"]["score"], 1.0)

    def test_geometry_snapshot_prefers_s4_preoptimization_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "room_v_result" / "S4_layout_refinement"
            folder.mkdir(parents=True)
            snapshot = folder / "room_v_placement_info_s3.json"
            snapshot.write_text("{}", encoding="utf-8")
            self.assertEqual(
                physical.find_geometry_snapshot(root, "room", "v"),
                snapshot,
            )

    def test_incomplete_geometry_snapshot_is_rejected(self):
        data = {
            "obj_info": {
                "box_0": {
                    "pose_matrix_for_blender": pose().tolist(),
                    "pcd_obb_size": [1, 1, 1],
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "usable_bbox_and_length=0/1"):
            physical.validate_geometry_snapshot(data, Path("bad.json"))

    def test_overlap_is_localized_and_reduces_collision_score(self):
        target = json.loads(json.dumps(self.source))
        target["obj_info"]["box_0"]["pose_matrix_for_blender"] = pose(0, 0, 0.5).tolist()
        target["obj_info"]["box_1"]["pose_matrix_for_blender"] = pose(0.25, 0, 0.5).tolist()
        metrics, objects = physical.evaluate_scene(self.source, target, args())
        self.assertEqual(metrics["unintended_collision_pairs"], 1)
        self.assertLess(metrics["families"]["collision"]["score"], 1.0)
        by_id = {row["object_id"]: row for row in objects}
        self.assertGreater(by_id["box_0"]["collision_overlap_fraction"], 0.0)
        self.assertGreater(by_id["box_1"]["collision_overlap_fraction"], 0.0)

    def test_relation_program_attachment_is_not_scored_as_collision(self):
        target = json.loads(json.dumps(self.source))
        target["obj_info"]["box_0"]["pose_matrix_for_blender"] = pose(0, 0, 0.5).tolist()
        target["obj_info"]["box_1"]["pose_matrix_for_blender"] = pose(0.25, 0, 0.5).tolist()
        target["sceneproof_relation_programs"] = {
            "programs": [
                {
                    "kind": "PLANE_ATTACH",
                    "participants": [
                        {"object_id": "box_0"},
                        {"object_id": "box_1"},
                    ],
                }
            ]
        }
        metrics, _ = physical.evaluate_scene(self.source, target, args())
        self.assertEqual(metrics["unintended_collision_pairs"], 0)
        self.assertEqual(
            metrics["collision_relation_policy"],
            "relation_program_conditioned_candidates_and_contact_exemptions",
        )

    def test_relation_program_collision_candidate_remains_scored(self):
        target = json.loads(json.dumps(self.source))
        target["obj_info"]["box_0"]["pose_matrix_for_blender"] = pose(0, 0, 0.5).tolist()
        target["obj_info"]["box_1"]["pose_matrix_for_blender"] = pose(0.25, 0, 0.5).tolist()
        target["sceneproof_relation_programs"] = {
            "programs": [
                {
                    "kind": "COLLISION_EXCLUSION",
                    "participants": [
                        {"object_id": "box_0"},
                        {"object_id": "box_1"},
                    ],
                }
            ]
        }
        metrics, _ = physical.evaluate_scene(self.source, target, args())
        self.assertEqual(metrics["unintended_collision_pairs"], 1)

    def test_legacy_collision_policy_is_common_across_program_availability(self):
        target = json.loads(json.dumps(self.source))
        target["obj_info"]["box_0"]["pose_matrix_for_blender"] = pose(0, 0, 0.5).tolist()
        target["obj_info"]["box_1"]["pose_matrix_for_blender"] = pose(0.25, 0, 0.5).tolist()
        target["sceneproof_relation_programs"] = {"programs": [{
            "kind": "PLANE_ATTACH", "participants": [
                {"object_id": "box_0"}, {"object_id": "box_1"},
            ]
        }]}
        common = args()
        common.collision_policy = "legacy"
        metrics, _ = physical.evaluate_scene(self.source, target, common)
        self.assertEqual(metrics["unintended_collision_pairs"], 1)
        self.assertEqual(
            metrics["collision_relation_policy"],
            "legacy_all_pairs_with_direct_support_exemptions",
        )

    def test_runtime_jsonl_is_machine_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime_gpu0.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "scene": "a",
                                "elapsed_seconds": 12.5,
                                "status": "ok",
                            }
                        ),
                        json.dumps(
                            {
                                "scene": "b",
                                "elapsed_seconds": 99,
                                "status": "fail",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(physical.parse_runtime_jsonl(path), {"a": 12.5})

    def test_runtime_is_restricted_to_evaluated_manifest_scenes(self):
        runtime = {"holdout_a": 10.0, "holdout_b": 20.0, "other": 999.0}
        self.assertEqual(
            physical.restrict_runtime_to_scenes(
                runtime, ["holdout_a", "holdout_b"]
            ),
            {"holdout_a": 10.0, "holdout_b": 20.0},
        )

    def test_ascii_report_includes_pass_rates_and_composite_runtime(self):
        aggregate = {
            "macro_realizability": 0.9,
            "critical_realizability": 0.8,
            "mean_local_realizability": 0.85,
            "mean_scene_p10_local_realizability": 0.7,
            "families": {
                name: {"score": 0.9, "pass_rate": 0.8, "n": 2}
                for name in ("collision", "support", "plane", "boundary", "semantic")
            },
        }
        report = {
            "thresholds": {"contact_gap_m": 0.05},
            "versions": {
                "legacy": {
                    "aggregate": aggregate,
                    "runtime": physical.runtime_summary({"a": 100.0}),
                },
                "candidate": {
                    "aggregate": aggregate,
                    "runtime": physical.runtime_summary(
                        {"a": 40.0},
                        mode="composite",
                        components=["full", "depth"],
                        stage_only={"a": 10.0},
                    ),
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.ascii"
            physical.write_report(path, report, "legacy")
            text = path.read_text(encoding="utf-8")
        self.assertIn("Threshold pass rates:", text)
        self.assertIn("2.50x", text)
        self.assertIn("components=full+depth", text)


if __name__ == "__main__":
    unittest.main()
