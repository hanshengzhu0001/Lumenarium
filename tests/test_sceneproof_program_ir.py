import json
import unittest

from modules._sceneproof_certificate import (
    arbitrate_candidate,
    select_or_rollback,
)
from modules._sceneproof_compile import (
    audit_live_factor_parity,
    compile_legacy_relation_programs,
)
from modules._sceneproof_execute import execute_relation_program
from sceneproof_compile_audit import (
    _footprint_size,
    _legacy_inputs,
    _optimization_ids,
    _source_relation_counts,
    _spatial_relations,
)


class SceneProofProgramIRTest(unittest.TestCase):
    def test_coverage_adapter_reads_live_against_wall_field(self):
        obj_info = {
            "chair_0": {
                "pose_matrix_for_blender": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "againstWall": "wall_2",
            }
        }
        _, _, planes = _legacy_inputs(obj_info, ["chair_0"])
        self.assertEqual(planes[0]["plane_id"], "wall_2")

    def test_coverage_adapter_mirrors_live_support_default(self):
        info = {"supported": "table_0", "SpatialRel": None}
        relations, defaulted = _spatial_relations(info)
        self.assertEqual(relations, {"on"})
        self.assertTrue(defaulted)
        obj_info = {
            "table_0": {"pose_matrix_for_blender": []},
            "mug_0": {"pose_matrix_for_blender": [], **info},
        }
        support, fixed, _ = _legacy_inputs(
            obj_info, ["table_0", "mug_0"]
        )
        self.assertEqual(support, [("mug_0", "table_0")])
        self.assertEqual(fixed, [])
        counts = _source_relation_counts(
            obj_info, ["table_0", "mug_0"]
        )
        self.assertEqual(counts["support_internal"], 1)
        self.assertEqual(counts["spatialrel_defaulted_to_on"], 1)

    def test_coverage_object_set_excludes_live_architecture(self):
        pose = [[1, 0, 0, 0]] * 4
        obj_info = {
            "wall_0": {"pose_matrix_for_blender": pose},
            "floor_0": {"pose_matrix_for_blender": pose},
            "chair_0": {"pose_matrix_for_blender": pose},
            "coin_0": {
                "pose_matrix_for_blender": pose,
                "supported": "box_0",
                "SpatialRel": "inside",
            },
        }
        self.assertEqual(_optimization_ids(obj_info), ["chair_0"])

    def test_coverage_footprint_uses_bbox_extent(self):
        self.assertEqual(
            _footprint_size({"bbox": [[-2, -1, 0], [3, 2, 4]]}),
            (5.0, 3.0),
        )

    def test_semantic_skips_are_explicitly_rejected(self):
        obj_info = {"chair_0": {}, "table_0": {}}
        bundle = compile_legacy_relation_programs(
            scene_id="semantic_skip",
            obj_info=obj_info,
            ordered_ids=("chair_0", "table_0"),
            semantic_specs={
                "skipped": [{
                    "relation": "directlyFacing",
                    "source": "chair_0",
                    "target": "missing_0",
                    "reason": "missing optimization object",
                }]
            },
        )
        self.assertEqual(len(bundle.rejected_relations), 1)
        self.assertEqual(bundle.rejected_relations[0]["kind"], "SEMANTIC_SKIPPED")
        self.assertEqual(bundle.compiler_audit["input_relations"], 1)

    def test_plane_attachment_exclusively_owns_wall_parent(self):
        obj_info = {
            "picture_0": {
                "supported": "wall_0",
                "SpatialRel": "on",
                "againstWall": "wall_0",
            }
        }
        bundle = compile_legacy_relation_programs(
            scene_id="wall_ownership",
            obj_info=obj_info,
            ordered_ids=("picture_0",),
            plane_bindings=({
                "child_id": "picture_0",
                "plane_id": "wall_0",
                "orientation_required": True,
            },),
        )
        self.assertEqual(
            [program.kind for program in bundle.programs],
            ["PLANE_ATTACH"],
        )

    def test_missing_relation_unknown_parent_is_explicitly_rejected(self):
        obj_info = {
            "mug_0": {"supported": "missing_table_0", "SpatialRel": None}
        }
        bundle = compile_legacy_relation_programs(
            scene_id="unknown_support",
            obj_info=obj_info,
            ordered_ids=("mug_0",),
        )
        self.assertEqual(len(bundle.programs), 0)
        self.assertEqual(len(bundle.rejected_relations), 1)
        self.assertEqual(bundle.rejected_relations[0]["kind"], "SUPPORT")
        self.assertIn("missing_table_0", bundle.rejected_relations[0]["reason"])

    def test_authoritative_topology_never_reintroduces_unbound_support(self):
        obj_info = {
            "table_0": {},
            "mug_0": {"supported": "table_0", "SpatialRel": "on"},
        }
        bundle = compile_legacy_relation_programs(
            scene_id="authoritative_support",
            obj_info=obj_info,
            ordered_ids=("table_0", "mug_0"),
            support_topology_authoritative=True,
        )
        self.assertFalse(any(p.kind == "SUPPORT" for p in bundle.programs))
        self.assertEqual(len(bundle.rejected_relations), 1)
        self.assertEqual(
            bundle.rejected_relations[0]["kind"], "UNBOUND_SUPPORT"
        )

    def test_authoritative_topology_compiles_only_owned_support(self):
        obj_info = {
            "table_0": {},
            "mug_0": {"supported": "table_0", "SpatialRel": "on"},
        }
        bundle = compile_legacy_relation_programs(
            scene_id="owned_support",
            obj_info=obj_info,
            ordered_ids=("table_0", "mug_0"),
            support_pairs=(("mug_0", "table_0"),),
            support_topology_authoritative=True,
        )
        self.assertEqual(
            [p.kind for p in bundle.programs],
            ["SUPPORT"],
        )
        self.assertEqual(len(bundle.rejected_relations), 0)

    def test_live_factor_parity_matches_all_owned_families(self):
        obj_info = {
            "table_0": {},
            "mug_0": {"supported": "table_0", "SpatialRel": "on"},
            "chair_0": {},
        }
        semantics = {
            "point_pairs": [(2, 0)],
            "point_offsets": [0.0],
            "distance_pairs": [(2, 0)],
            "distance_minimum": [0.5],
            "distance_maximum": [2.0],
            "align_pairs": [],
            "align_offsets": [],
            "skipped": [],
        }
        bundle = compile_legacy_relation_programs(
            scene_id="live_parity",
            obj_info=obj_info,
            ordered_ids=("table_0", "mug_0", "chair_0"),
            support_pairs=(("mug_0", "table_0"),),
            collision_pairs=((1, 2),),
            semantic_specs=semantics,
            support_topology_authoritative=True,
        )
        parity = audit_live_factor_parity(
            bundle,
            ordered_ids=("table_0", "mug_0", "chair_0"),
            support_pairs=((1, 0),),
            collision_pairs=((1, 2),),
            semantic_specs=semantics,
        )
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["mismatches"], {})

    def test_live_factor_parity_detects_missing_program_owner(self):
        bundle = compile_legacy_relation_programs(
            scene_id="live_parity_failure",
            obj_info={"table_0": {}, "mug_0": {}},
            ordered_ids=("table_0", "mug_0"),
            support_topology_authoritative=True,
        )
        parity = audit_live_factor_parity(
            bundle,
            ordered_ids=("table_0", "mug_0"),
            support_pairs=((1, 0),),
        )
        self.assertFalse(parity["passed"])
        self.assertEqual(parity["mismatches"]["SUPPORT"]["expected"], 1)

    def _bundle(self):
        obj_info = {
            "table_0": {"asset_id": "table"},
            "mug_0": {
                "asset_id": "mug",
                "supported": "table_0",
                "SpatialRel": "on",
            },
            "box_0": {"asset_id": "box"},
            "coin_0": {
                "asset_id": "coin",
                "supported": "box_0",
                "SpatialRel": "inside",
            },
        }
        return compile_legacy_relation_programs(
            scene_id="fixture",
            obj_info=obj_info,
            ordered_ids=("table_0", "mug_0", "box_0", "coin_0"),
            support_pairs=(("mug_0", "table_0"),),
            collision_pairs=(("mug_0", "box_0"),),
        )

    def test_compiler_is_deterministic_and_accounts_for_every_relation(self):
        first = self._bundle()
        second = self._bundle()
        self.assertEqual(first.canonical_json(), second.canonical_json())
        self.assertEqual(first.content_hash(), second.content_hash())
        audit = first.compiler_audit
        self.assertEqual(
            audit["input_relations"],
            audit["compiled"] + audit["abstained"] + audit["rejected"],
        )

    def test_duplicate_legacy_and_explicit_support_is_compiled_once(self):
        bundle = self._bundle()
        support = [program for program in bundle.programs if program.kind == "SUPPORT"]
        self.assertEqual(len(support), 1)

    def test_all_compiled_objects_keep_world_so3_and_frozen_scale(self):
        bundle = self._bundle()
        blocks = {
            block.object_id: block
            for program in bundle.programs
            for block in program.variable_blocks
        }
        self.assertTrue(blocks)
        self.assertTrue(
            all(block.rotation_mode == "world_so3" for block in blocks.values())
        )
        self.assertTrue(
            all(block.scale_mode == "frozen" for block in blocks.values())
        )
        self.assertEqual(blocks["mug_0"].parent_frame, "table_0")
        self.assertTrue(blocks["mug_0"].eliminable_translation)

    def test_inside_without_cavity_and_opening_abstains(self):
        bundle = self._bundle()
        self.assertFalse(any(program.kind == "INSIDE" for program in bundle.programs))
        inside = [
            relation
            for relation in bundle.abstained_relations
            if relation["kind"] == "INSIDE"
        ]
        self.assertEqual(len(inside), 1)
        self.assertEqual(inside[0]["status"], "ABSTAIN")
        self.assertIn("cavity", inside[0]["reason"])
        self.assertIn("opening", inside[0]["reason"])

    def test_inside_compiles_only_with_explicit_functional_geometry(self):
        obj_info = {
            "box_0": {},
            "coin_0": {"supported": "box_0", "SpatialRel": "inside"},
        }
        bundle = compile_legacy_relation_programs(
            scene_id="inside_fixture",
            obj_info=obj_info,
            ordered_ids=("box_0", "coin_0"),
            affordance_metadata={
                "box_0": {
                    "cavities": [{"part_id": "cavity_0"}],
                    "openings": [{"part_id": "opening_0"}],
                }
            },
        )
        program = next(program for program in bundle.programs if program.kind == "INSIDE")
        part_ids = {part.part_id for part in program.participants}
        self.assertIn("cavity_0", part_ids)
        self.assertIn("opening_0", part_ids)

    def test_executor_abstains_on_missing_probe_and_fails_with_witness(self):
        bundle = self._bundle()
        support = next(program for program in bundle.programs if program.kind == "SUPPORT")
        abstained = execute_relation_program(
            support,
            {
                "contact_gap_m": 0.0,
                "containment_error_m": 0.0,
                "support_footprint_overlap_ratio": 1.0,
            },
        )
        self.assertEqual(abstained.static_status, "PASS")
        self.assertEqual(abstained.status, "ABSTAIN")
        self.assertIn("perturbation_survival", abstained.missing_evidence)

        failed = execute_relation_program(
            support,
            {
                "contact_gap_m": 0.0,
                "containment_error_m": 0.03,
                "support_footprint_overlap_ratio": 0.75,
                "perturbation_survival": True,
            },
        )
        self.assertEqual(failed.status, "FAIL")
        self.assertEqual(failed.witness["program_id"], support.program_id)
        self.assertEqual(
            failed.witness["measurement"],
            "support_footprint_overlap_ratio",
        )

    def test_component_failure_cannot_be_hidden_by_macro_and_rolls_back(self):
        bundle = self._bundle()
        certificates = []
        for program in bundle.programs:
            measurements = {}
            for factor in program.factors:
                measurement = factor.parameters.get("measurement")
                operator = factor.parameters.get("operator")
                threshold = factor.parameters.get("threshold")
                if measurement is None:
                    continue
                measurements[measurement] = (
                    True
                    if operator == "true"
                    else threshold
                )
            for probe in program.probes:
                for evidence in probe.required_evidence:
                    measurements[evidence] = True
            certificates.append(execute_relation_program(program, measurements))
        incumbent = {
            "collision": 0.5,
            "support": 0.5,
            "plane": 0.5,
            "semantic": 0.5,
            "rotation": 0.5,
            "translation": 0.5,
        }
        candidate = dict(incumbent)
        candidate.update({"collision": 0.7, "support": 0.49})
        decision = arbitrate_candidate(
            bundle=bundle,
            incumbent_metrics=incumbent,
            candidate_metrics=candidate,
            incumbent_certificates=certificates,
            candidate_certificates=certificates,
        )
        self.assertFalse(decision.accepted)
        self.assertIn("support", decision.failed_components)
        original = {"poses": [[1, 2, 3]], "tag": "incumbent"}
        selected = select_or_rollback(
            original,
            {"poses": [[9, 9, 9]], "tag": "candidate"},
            decision,
        )
        self.assertEqual(
            json.dumps(selected, sort_keys=True),
            json.dumps(original, sort_keys=True),
        )
        selected["poses"][0][0] = -1
        self.assertEqual(original["poses"][0][0], 1)

    def test_hard_program_regression_releases_only_witness_neighborhood(self):
        bundle = self._bundle()
        support = next(program for program in bundle.programs if program.kind == "SUPPORT")
        passed = execute_relation_program(
            support,
            {
                "contact_gap_m": 0.0,
                "containment_error_m": 0.0,
                "support_footprint_overlap_ratio": 1.0,
                "perturbation_survival": True,
            },
        )
        failed = execute_relation_program(
            support,
            {
                "contact_gap_m": 0.0,
                "containment_error_m": 0.2,
                "support_footprint_overlap_ratio": 1.0,
                "perturbation_survival": True,
            },
        )
        other_programs = [
            program for program in bundle.programs if program.program_id != support.program_id
        ]
        other_certificates = []
        for program in other_programs:
            # Soft-only programs pass statically; a collision program gets its
            # explicit passing measurement.
            measurements = {"collision_fraction": 0.0}
            other_certificates.append(execute_relation_program(program, measurements))
        metrics = {name: 1.0 for name in ("collision", "support", "plane", "semantic", "rotation", "translation")}
        decision = arbitrate_candidate(
            bundle=bundle,
            incumbent_metrics=metrics,
            candidate_metrics=metrics,
            incumbent_certificates=[passed, *other_certificates],
            candidate_certificates=[failed, *other_certificates],
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.regressed_programs, (support.program_id,))
        self.assertEqual(
            set(decision.release_object_ids),
            {"mug_0", "table_0"},
        )


if __name__ == "__main__":
    unittest.main()
