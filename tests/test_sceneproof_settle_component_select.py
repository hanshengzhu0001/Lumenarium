#!/usr/bin/env python3
"""Tests for the Fix85 component-attributed settle promotion.

The suite covers the three properties the method relies on:

* the promoted candidate is written with the only pose key the evaluators read;
* the moved set is partitioned so that every scored term is owned by exactly one
  component, and the per-family deltas are additive across components;
* the verification preconditions actually fire when the decomposition is not
  faithful, so a broken decomposition abstains instead of promoting.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sceneproof_settle_component_select_fix85 as fix85


IDENTITY = np.eye(4).tolist()


def pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> list[list[float]]:
    matrix = np.eye(4)
    matrix[:3, 3] = (x, y, z)
    return matrix.tolist()


def unit_box_bbox(x: float, y: float, z: float, half: float = 0.5) -> list[list[float]]:
    return [
        [x + sx * half, y + sy * half, z + sz * half]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]


def object_info(
    x: float,
    y: float,
    z: float,
    *,
    half: float = 0.5,
    supported: str | None = None,
) -> dict[str, object]:
    info: dict[str, object] = {
        "pose_matrix_for_blender": pose(x, y, z),
        "bbox": unit_box_bbox(x, y, z, half),
        "length": [2 * half, 2 * half, 2 * half],
        "retrieved_asset": "asset",
    }
    if supported is not None:
        info["supported"] = supported
    return info


def probe_payload(
    object_id: str,
    *,
    translation: float = 0.20,
    status: str = "measured",
    settled: object | None = None,
) -> dict[str, object]:
    return {
        "object_id": object_id,
        "status": status,
        "translation_delta_m": translation,
        "rotation_delta_deg": 1.0,
        "before_pose_matrix": IDENTITY,
        "settled_pose_matrix": IDENTITY if settled is None else settled,
    }


class DisjointSetTest(unittest.TestCase):
    def test_transitive_union(self) -> None:
        disjoint = fix85.DisjointSet(["a", "b", "c", "d"])
        disjoint.union("a", "b")
        disjoint.union("b", "c")
        self.assertEqual(disjoint.find("a"), disjoint.find("c"))
        self.assertNotEqual(disjoint.find("a"), disjoint.find("d"))


class ProbeLoadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(__file__).resolve().parent / "_fix85_probe_tmp"
        self.temporary.mkdir(exist_ok=True)
        self.incumbent_info = {
            "chair_1": object_info(0, 0, 0),
            "cup_1": object_info(3, 0, 0),
            "jitter_1": object_info(6, 0, 0),
            "broken_1": object_info(9, 0, 0),
            "floor_1": object_info(0, 0, -1, half=10.0),
        }

    def tearDown(self) -> None:
        for path in self.temporary.glob("*"):
            path.unlink()
        self.temporary.rmdir()

    def write(self, name: str, payload: dict[str, object]) -> None:
        (self.temporary / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_filters(self) -> None:
        self.write("chair_1.json", probe_payload("chair_1", translation=0.30))
        self.write("cup_1.json", probe_payload("cup_1", status="abstained"))
        self.write("jitter_1.json", probe_payload("jitter_1", translation=0.004))
        self.write("broken_1.json", probe_payload("broken_1", settled=[[1, 2], [3, 4]]))
        self.write("floor_1.json", probe_payload("floor_1", translation=0.9))
        self.write("absent_1.json", probe_payload("absent_1", translation=0.9))
        self.write("_manifest.json", {"ignored": True})

        probes, rejected = fix85.load_probes(
            self.temporary,
            self.incumbent_info,
            min_translation_m=0.05,
            require_measured=True,
        )
        self.assertEqual(sorted(probes), ["chair_1"])
        self.assertEqual(rejected["cup_1"], "status:abstained")
        self.assertEqual(rejected["jitter_1"], "below_min_translation")
        self.assertEqual(rejected["broken_1"], "invalid_settled_pose_matrix")
        self.assertEqual(rejected["floor_1"], "structural_object")
        self.assertEqual(rejected["absent_1"], "absent_from_incumbent")

    def test_unmeasured_probes_can_be_admitted(self) -> None:
        self.write("cup_1.json", probe_payload("cup_1", status="abstained"))
        probes, _ = fix85.load_probes(
            self.temporary,
            self.incumbent_info,
            min_translation_m=0.05,
            require_measured=False,
        )
        self.assertEqual(sorted(probes), ["cup_1"])


class MaterializationTest(unittest.TestCase):
    """Regression test for the defect that invalidated the global settle run."""

    def test_writes_the_pose_key_the_evaluators_read(self) -> None:
        incumbent = {"obj_info": {"chair_1": object_info(0, 0, 0)}}
        settled = pose(0.0, 0.0, -0.4)
        probes = {
            "chair_1": {
                "object_id": "chair_1",
                "settled_pose_matrix": settled,
                "translation_delta_m": 0.4,
            }
        }
        candidate = fix85.materialize_placement(
            incumbent, probes, ["chair_1"], provenance={"stage": "test"}
        )
        info = candidate["obj_info"]["chair_1"]
        self.assertEqual(info["pose_matrix_for_blender"], settled)
        self.assertNotIn("matrix_after_settle", info)
        self.assertEqual(
            candidate["sceneproof_settle_commit"]["pose_key"],
            "pose_matrix_for_blender",
        )
        self.assertEqual(
            candidate["sceneproof_settle_commit"]["committed_object_ids"], ["chair_1"]
        )
        # The incumbent must not be mutated in place.
        self.assertEqual(
            incumbent["obj_info"]["chair_1"]["pose_matrix_for_blender"], pose(0, 0, 0)
        )

    def test_unknown_objects_are_not_committed(self) -> None:
        incumbent = {"obj_info": {"chair_1": object_info(0, 0, 0)}}
        probes = {
            "ghost_1": {
                "object_id": "ghost_1",
                "settled_pose_matrix": IDENTITY,
                "translation_delta_m": 1.0,
            }
        }
        candidate = fix85.materialize_placement(
            incumbent, probes, ["ghost_1"], provenance={}
        )
        self.assertEqual(
            candidate["sceneproof_settle_commit"]["committed_object_ids"], []
        )


class DependencyModelTest(unittest.TestCase):
    def test_support_parent_and_distance_separate_components(self) -> None:
        source_info = {
            "floor_1": object_info(0, 0, -0.5, half=20.0),
            "table_1": object_info(0, 0, 0),
            "cup_1": object_info(0, 0, 1.0, half=0.1, supported="table_1"),
            "lamp_1": object_info(30, 0, 0),
        }
        incumbent_info = {name: dict(info) for name, info in source_info.items()}
        batch_info = {name: dict(info) for name, info in source_info.items()}
        # The cup settles 30 cm down onto the table; the lamp settles in place.
        batch_info["cup_1"] = object_info(0, 0, 0.7, half=0.1, supported="table_1")

        dependencies, diagnostics = fix85.build_dependency_sets(
            source_info,
            incumbent_info,
            batch_info,
            collision_volume_tolerance=1e-6,
        )
        self.assertIn("table_1", dependencies["cup_1"])
        self.assertIn("cup_1", dependencies["cup_1"])
        self.assertNotIn("lamp_1", dependencies["cup_1"])
        self.assertNotIn("table_1", dependencies["lamp_1"])
        self.assertEqual(diagnostics["scored_object_count"], 3)

    def test_overlapping_objects_are_coupled(self) -> None:
        source_info = {
            "floor_1": object_info(0, 0, -0.5, half=20.0),
            "box_a": object_info(0, 0, 0),
            "box_b": object_info(5, 0, 0),
        }
        incumbent_info = {name: dict(info) for name, info in source_info.items()}
        batch_info = {name: dict(info) for name, info in source_info.items()}
        # box_b slides onto box_a, producing a real overlap volume.
        batch_info["box_b"] = object_info(0.2, 0, 0)

        dependencies, _ = fix85.build_dependency_sets(
            source_info,
            incumbent_info,
            batch_info,
            collision_volume_tolerance=1e-6,
        )
        self.assertIn("box_b", dependencies["box_a"])
        self.assertIn("box_a", dependencies["box_b"])


class PartitionAndAttributionTest(unittest.TestCase):
    def setUp(self) -> None:
        # Two independent groups plus one object that depends on nothing moved.
        self.dependencies = {
            "chair_1": {"chair_1", "table_1"},
            "table_1": {"table_1", "chair_1"},
            "cushion_1": {"cushion_1", "sofa_1"},
            "sofa_1": {"sofa_1", "cushion_1"},
            "picture_1": {"picture_1"},
        }
        self.moved = {"chair_1", "cushion_1"}

    def test_components_own_every_touched_term_exactly_once(self) -> None:
        components = fix85.partition_components(self.dependencies, self.moved)
        self.assertEqual(len(components), 2)
        owned = [object_id for c in components for object_id in c["owned_term_object_ids"]]
        self.assertEqual(len(owned), len(set(owned)))
        by_member = {c["member_object_ids"][0]: c for c in components}
        self.assertEqual(
            by_member["chair_1"]["owned_term_object_ids"], ["chair_1", "table_1"]
        )
        self.assertEqual(
            by_member["cushion_1"]["owned_term_object_ids"], ["cushion_1", "sofa_1"]
        )
        self.assertNotIn("picture_1", owned)

    def test_shared_static_neighbour_merges_components(self) -> None:
        dependencies = dict(self.dependencies)
        # A single static object now reads both moved objects, so no finer
        # decomposition is exact and the two components must merge.
        dependencies["shelf_1"] = {"shelf_1", "chair_1", "cushion_1"}
        components = fix85.partition_components(dependencies, self.moved)
        self.assertEqual(len(components), 1)
        self.assertEqual(
            components[0]["member_object_ids"], ["chair_1", "cushion_1"]
        )

    def test_delta_is_additive_across_components(self) -> None:
        components = fix85.partition_components(self.dependencies, self.moved)
        denominators = {family: 5 for family in fix85.FAMILIES}
        baseline = {
            "chair_1": {family: 0.40 for family in fix85.FAMILIES},
            "table_1": {family: 0.50 for family in fix85.FAMILIES},
            "cushion_1": {family: 0.60 for family in fix85.FAMILIES},
            "sofa_1": {family: 0.70 for family in fix85.FAMILIES},
            "picture_1": {family: 0.80 for family in fix85.FAMILIES},
        }
        candidate = {
            "chair_1": {family: 0.90 for family in fix85.FAMILIES},
            "table_1": {family: 0.50 for family in fix85.FAMILIES},
            "cushion_1": {family: 0.10 for family in fix85.FAMILIES},
            "sofa_1": {family: 0.70 for family in fix85.FAMILIES},
            "picture_1": {family: 0.80 for family in fix85.FAMILIES},
        }
        table = {"base": baseline, "cand": candidate}
        fix85.attribute_components(components, table, denominators, "base", "cand")
        by_member = {c["member_object_ids"][0]: c for c in components}
        self.assertAlmostEqual(by_member["chair_1"]["delta"]["support"], 0.5 / 5)
        self.assertAlmostEqual(by_member["cushion_1"]["delta"]["support"], -0.5 / 5)

        measured = (
            sum(candidate[o]["support"] for o in candidate) / 5
            - sum(baseline[o]["support"] for o in baseline) / 5
        )
        physical = {
            "versions": {
                "base": {
                    "scenes": {
                        "s": {
                            "families": {
                                family: {
                                    "n": 5,
                                    "score": sum(
                                        baseline[o][family] for o in baseline
                                    )
                                    / 5,
                                }
                                for family in fix85.FAMILIES
                            }
                        }
                    }
                },
                "cand": {
                    "scenes": {
                        "s": {
                            "families": {
                                family: {
                                    "n": 5,
                                    "score": sum(
                                        candidate[o][family] for o in candidate
                                    )
                                    / 5,
                                }
                                for family in fix85.FAMILIES
                            }
                        }
                    }
                },
            }
        }
        check = fix85.verify_additivity(
            components,
            table,
            denominators,
            physical,
            "s",
            "base",
            "cand",
            tolerance=1e-12,
        )
        self.assertTrue(check["ok"])
        self.assertAlmostEqual(
            check["per_family"]["support"]["predicted_delta"], measured
        )
        self.assertEqual(check["unowned_term_drift_count"], 0)

    def test_unowned_drift_is_detected(self) -> None:
        components = fix85.partition_components(self.dependencies, self.moved)
        denominators = {family: 5 for family in fix85.FAMILIES}
        baseline = {"picture_1": {family: 0.80 for family in fix85.FAMILIES}}
        candidate = {"picture_1": {family: 0.10 for family in fix85.FAMILIES}}
        table = {"base": baseline, "cand": candidate}
        fix85.attribute_components(components, table, denominators, "base", "cand")
        physical = {"versions": {"base": {}, "cand": {}}}
        check = fix85.verify_additivity(
            components,
            table,
            denominators,
            physical,
            "s",
            "base",
            "cand",
            tolerance=1e-12,
        )
        self.assertFalse(check["ok"])
        self.assertGreater(check["unowned_term_drift_count"], 0)


class TermExportVerificationTest(unittest.TestCase):
    def physical(self, baseline_score: float, candidate_score: float, n: int = 2):
        return {
            "versions": {
                "base": {
                    "scenes": {
                        "s": {"families": {"support": {"n": n, "score": baseline_score}}}
                    }
                },
                "cand": {
                    "scenes": {
                        "s": {
                            "families": {"support": {"n": n, "score": candidate_score}}
                        }
                    }
                },
            }
        }

    def table(self, baseline: dict, candidate: dict):
        empty = {family: None for family in fix85.FAMILIES}
        return {
            "base": {k: {**empty, "support": v} for k, v in baseline.items()},
            "cand": {k: {**empty, "support": v} for k, v in candidate.items()},
        }

    def test_faithful_export_passes(self) -> None:
        table = self.table({"a": 0.2, "b": 0.4}, {"a": 0.6, "b": 0.4})
        check = fix85.verify_term_export(
            table, self.physical(0.3, 0.5), "s", "base", "cand", tolerance=1e-12
        )
        self.assertTrue(check["per_family"]["support"]["ok"])

    def test_denominator_mismatch_fails(self) -> None:
        table = self.table({"a": 0.2, "b": 0.4}, {"a": 0.6, "b": 0.4})
        check = fix85.verify_term_export(
            table, self.physical(0.3, 0.5, n=3), "s", "base", "cand", tolerance=1e-12
        )
        self.assertFalse(check["per_family"]["support"]["ok"])
        self.assertIn("support", check["violations"])

    def test_contributor_set_change_fails(self) -> None:
        table = self.table({"a": 0.2, "b": 0.4}, {"a": 0.6})
        check = fix85.verify_term_export(
            table, self.physical(0.3, 0.6, n=2), "s", "base", "cand", tolerance=1e-12
        )
        self.assertFalse(check["per_family"]["support"]["contributor_sets_equal"])
        self.assertFalse(check["per_family"]["support"]["ok"])

    def test_score_residual_fails(self) -> None:
        table = self.table({"a": 0.2, "b": 0.4}, {"a": 0.6, "b": 0.4})
        check = fix85.verify_term_export(
            table, self.physical(0.3, 0.9), "s", "base", "cand", tolerance=1e-12
        )
        self.assertFalse(check["per_family"]["support"]["ok"])
        self.assertGreater(check["per_family"]["support"]["max_residual"], 0.3)


class SelectionTest(unittest.TestCase):
    def components(self):
        def make(index: int, member: str, collision: float, support: float):
            return {
                "component_index": index,
                "component_root": member,
                "member_object_ids": [member],
                "owned_term_object_ids": [member],
                "member_count": 1,
                "owned_term_count": 1,
                "delta": {
                    "collision": collision,
                    "support": support,
                    "plane": 0.0,
                    "boundary": 0.0,
                    "semantic": 0.0,
                },
                "total_delta": collision + support,
            }

        return [
            make(0, "good_1", 0.0, 0.05),
            make(1, "bad_1", -0.02, 0.30),
            make(2, "neutral_1", 0.0, 0.0),
        ]

    def test_strict_policy_rejects_any_gated_regression(self) -> None:
        selection = fix85.select_components(
            self.components(),
            gated_families=("collision", "support", "plane"),
            epsilon=1e-9,
            allow_surplus_trading=False,
        )
        self.assertEqual(selection["accepted_object_ids"], ["good_1", "neutral_1"])
        self.assertEqual(selection["rejected_component_indices"], [1])
        self.assertGreaterEqual(selection["predicted_scene_delta"]["support"], 0.0)
        self.assertTrue(selection["predicted_gated_delta_non_negative"])

    def test_surplus_trading_needs_prior_surplus(self) -> None:
        components = self.components()
        components.append(
            {
                "component_index": 3,
                "component_root": "rich_1",
                "member_object_ids": ["rich_1"],
                "owned_term_object_ids": ["rich_1"],
                "member_count": 1,
                "owned_term_count": 1,
                "delta": {
                    "collision": 0.10,
                    "support": 0.10,
                    "plane": 0.0,
                    "boundary": 0.0,
                    "semantic": 0.0,
                },
                "total_delta": 0.20,
            }
        )
        selection = fix85.select_components(
            components,
            gated_families=("collision", "support", "plane"),
            epsilon=1e-9,
            allow_surplus_trading=True,
        )
        # rich_1 is ranked first and its collision surplus absorbs bad_1.
        self.assertIn("bad_1", selection["accepted_object_ids"])
        self.assertGreaterEqual(selection["predicted_scene_delta"]["collision"], 0.0)

    def test_selected_prediction_is_the_sum_of_accepted_components(self) -> None:
        components = self.components()
        selection = fix85.select_components(
            components,
            gated_families=("collision", "support"),
            epsilon=1e-9,
            allow_surplus_trading=False,
        )
        accepted = [
            c
            for c in components
            if c["component_index"] in selection["accepted_component_indices"]
        ]
        for family in fix85.FAMILIES:
            self.assertAlmostEqual(
                selection["predicted_scene_delta"][family],
                sum(c["delta"][family] for c in accepted),
            )


class TermTableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path(__file__).resolve().parent / "_fix85_objects_tmp.csv"

    def tearDown(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def write(self, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_empty_terms_become_none(self) -> None:
        fieldnames = ["version", "scene", "object_id"] + [
            f"{family}_term" for family in fix85.FAMILIES
        ]
        self.write(
            [
                {
                    "version": "base",
                    "scene": "s",
                    "object_id": "chair_1",
                    "collision_term": 1.0,
                    "support_term": None,
                    "plane_term": "",
                    "boundary_term": 0.5,
                    "semantic_term": "nan",
                },
                {
                    "version": "base",
                    "scene": "other",
                    "object_id": "chair_1",
                    "collision_term": 0.0,
                },
            ],
            fieldnames,
        )
        table = fix85.load_term_table(self.path, "s", ("base",))
        terms = table["base"]["chair_1"]
        self.assertEqual(terms["collision"], 1.0)
        self.assertIsNone(terms["support"])
        self.assertIsNone(terms["plane"])
        self.assertEqual(terms["boundary"], 0.5)
        self.assertIsNone(terms["semantic"])

    def test_missing_term_columns_are_rejected(self) -> None:
        self.write(
            [{"version": "base", "scene": "s", "object_id": "chair_1"}],
            ["version", "scene", "object_id"],
        )
        with self.assertRaises(SystemExit):
            fix85.load_term_table(self.path, "s", ("base",))


class EvaluatorTermExportTest(unittest.TestCase):
    """The evaluator must emit terms that reproduce its own family means."""

    def test_family_means_equal_the_mean_of_exported_terms(self) -> None:
        import argparse

        import eval_physical_realizability as evaluator

        source = {
            "obj_info": {
                "floor_1": object_info(0, 0, -0.5, half=20.0),
                "table_1": object_info(0, 0, 0),
                "cup_1": object_info(0, 0, 1.0, half=0.1, supported="table_1"),
                "orphan_1": object_info(4, 0, 0, supported="missing_9"),
            },
            "reference_obj": "floor_1",
        }
        target = {
            "obj_info": {
                name: dict(info) for name, info in source["obj_info"].items()
            },
            "reference_obj": "floor_1",
        }
        args = argparse.Namespace(
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
        )
        metrics, rows = evaluator.evaluate_scene(source, target, args)
        by_id = {row["object_id"]: row for row in rows}
        # A declared parent that does not exist contributes a constant zero term.
        self.assertEqual(by_id["orphan_1"]["support_term"], 0.0)
        for family, aggregate in metrics["families"].items():
            terms = [
                row[f"{family}_term"]
                for row in rows
                if row.get(f"{family}_term") is not None
            ]
            self.assertEqual(len(terms), aggregate["n"])
            if aggregate["n"]:
                self.assertAlmostEqual(
                    float(np.mean(terms)), aggregate["score"], places=12
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
