import unittest

from modules._sceneproof_compile import compile_legacy_relation_programs
from modules._sceneproof_factor_binding import (
    RuntimeFactorRow,
    audit_factor_semantics_and_ownership,
    build_runtime_factor_rows,
)


class SceneProofFactorBindingTest(unittest.TestCase):
    def _bundle(self):
        obj_info = {
            "table_0": {},
            "mug_0": {"supported": "table_0", "SpatialRel": "on"},
            "chair_0": {},
        }
        semantics = {
            "point_pairs": [(2, 0)],
            "point_offsets": [0.0],
            "align_pairs": [],
            "align_offsets": [],
            "distance_pairs": [],
            "skipped": [],
        }
        return compile_legacy_relation_programs(
            scene_id="binding",
            obj_info=obj_info,
            ordered_ids=("table_0", "mug_0", "chair_0"),
            support_pairs=((1, 0),),
            collision_pairs=((1, 2),),
            semantic_specs=semantics,
            support_topology_authoritative=True,
        ), semantics

    def test_factor_semantics_bind_units_and_detached_target_ownership(self):
        bundle, semantics = self._bundle()
        rows = build_runtime_factor_rows(
            ordered_ids=bundle.object_ids,
            support_pairs=((1, 0),),
            containment_pairs=((1, 0),),
            collision_pairs=((1, 2),),
            semantic_specs=semantics,
        )
        audit = audit_factor_semantics_and_ownership(bundle, rows)
        self.assertTrue(audit["passed"], audit["mismatches"])
        point = next(
            row for row in audit["bindings"]
            if row["channel"] == "semantic_point_towards"
        )
        self.assertEqual(point["variable_objects"], ["chair_0"])
        self.assertFalse(audit["solver_executor_intertwined"])

    def test_missing_containment_row_is_explicit_abstention(self):
        bundle, semantics = self._bundle()
        rows = build_runtime_factor_rows(
            ordered_ids=bundle.object_ids,
            support_pairs=((1, 0),),
            containment_pairs=(),
            collision_pairs=((1, 2),),
            semantic_specs=semantics,
        )
        audit = audit_factor_semantics_and_ownership(bundle, rows)
        self.assertTrue(audit["passed"], audit["mismatches"])
        self.assertEqual(audit["abstained_solver_factors"], 1)
        self.assertEqual(
            audit["abstentions"][0]["reason"],
            "live_geometric_containment_gate",
        )

    def test_unit_or_owner_mismatch_fails_closed(self):
        bundle, semantics = self._bundle()
        rows = list(build_runtime_factor_rows(
            ordered_ids=bundle.object_ids,
            support_pairs=((1, 0),),
            containment_pairs=((1, 0),),
            collision_pairs=((1, 2),),
            semantic_specs=semantics,
        ))
        first = rows[0]
        rows[0] = RuntimeFactorRow(
            first.channel,
            first.relation_key,
            ("mug_0",),
            "dimensionless",
        )
        audit = audit_factor_semantics_and_ownership(bundle, rows)
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["mismatches"][0]["kind"], "factor_contract_mismatch")

    def test_cross_factor_prevents_leaf_translation_elimination(self):
        bundle, semantics = self._bundle()
        rows = build_runtime_factor_rows(
            ordered_ids=bundle.object_ids,
            support_pairs=((1, 0),),
            containment_pairs=((1, 0),),
            collision_pairs=((1, 2),),
            semantic_specs=semantics,
        )
        audit = audit_factor_semantics_and_ownership(bundle, rows)
        self.assertNotIn("mug_0", audit["safe_leaf_translation_objects"])
        self.assertIn("cross_factor", audit["rejected_leaf_translation_objects"]["mug_0"])


if __name__ == "__main__":
    unittest.main()
