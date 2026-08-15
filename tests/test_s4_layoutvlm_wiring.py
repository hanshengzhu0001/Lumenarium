import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LayoutVLMWiringTest(unittest.TestCase):
    def test_sceneproof_paper30_render_locks_source_camera_and_is_resumable(self):
        blender_source = (
            PROJECT_ROOT / "modules" / "S4_blender_layout_and_corr.py"
        ).read_text(encoding="utf-8")
        render_runner = (
            PROJECT_ROOT
            / "scripts"
            / "render_sceneproof_certified_paper30.sh"
        ).read_text(encoding="utf-8")
        paper30_runner = (
            PROJECT_ROOT
            / "scripts"
            / "run_sceneproof_certified_paper30_fix25.sh"
        ).read_text(encoding="utf-8")
        s4_runner = (
            PROJECT_ROOT / "scripts" / "run_paper30_v4_s4_only_dual_gpu.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT", blender_source)
        self.assertIn("source_s3_scene_camera_locked", blender_source)
        self.assertIn("camera_assignment_delta > camera_float32_tolerance", blender_source)
        self.assertIn("np.array_equal(camera_after, locked_camera_array)", blender_source)
        self.assertIn("camera_render_bitwise_stable", blender_source)
        self.assertIn("np.finfo(np.float32).eps", blender_source)
        self.assertIn("ignored_nonrenderable_record_ids", blender_source)
        self.assertIn("missing_expected_objects", blender_source)
        self.assertIn("CACHED_RENDER", render_runner)
        self.assertIn("SCENEPROOF_RENDER_SAMPLES", render_runner)
        self.assertIn("--runtime-jsonl", paper30_runner)
        self.assertIn("paired_bootstrap_10000.json", paper30_runner)
        self.assertIn('touch "$runtime_log"', s4_runner)
        self.assertNotIn(': > "$runtime_log"', s4_runner)

    def test_numpy_pickle_compat_remaps_only_numpy_private_core(self):
        from modules._numpy_pickle_compat import remap_numpy_pickle_module

        self.assertEqual(
            remap_numpy_pickle_module("numpy._core.numeric"),
            "numpy.core.numeric",
        )
        self.assertEqual(
            remap_numpy_pickle_module("numpy._core.multiarray"),
            "numpy.core.multiarray",
        )
        self.assertEqual(remap_numpy_pickle_module("numpy.linalg"), "numpy.linalg")
        blender_source = (
            PROJECT_ROOT / "modules" / "S4_blender_layout_and_corr.py"
        ).read_text(encoding="utf-8")
        self.assertIn("class _NumPyCompatUnpickler", blender_source)
        self.assertNotIn(
            "from modules._numpy_pickle_compat", blender_source
        )

    def test_v4_launcher_enables_deepsearch_and_layoutvlm(self):
        source = (PROJECT_ROOT / "run_imaginarium_I2Layout_v4.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"IMAGINARIUM_USE_DEEPSEARCH"] = "1"', source)
        self.assertIn('"IMAGINARIUM_USE_LAYOUTVLM"] = "1"', source)
        self.assertIn('"IMAGINARIUM_LAYOUTVLM_STAGE", "full"', source)
        self.assertIn('"IMAGINARIUM_LAYOUTVLM_ITERATIONS", "400"', source)

    def test_blender_layout_keeps_legacy_fallback(self):
        source = (
            PROJECT_ROOT / "modules" / "S4_blender_layout_and_corr.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        layout_functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "layout"
        ]
        self.assertEqual(len(layout_functions), 1)
        args = [arg.arg for arg in layout_functions[0].args.args]
        self.assertIn("use_layoutvlm", args)
        self.assertIn("layoutvlm_stage", args)
        self.assertIn("obj_manager.simulated_annealing(", source)
        self.assertIn("layoutvlm_pose_matrices[instance_id]", source)
        self.assertIn("optimize_collision_stage(", source)
        self.assertIn("optimize_plane_stage(", source)
        self.assertIn("optimize_semantic_stage(", source)
        self.assertIn("build_semantic_relation_specs(", source)
        self.assertIn(
            '"IMAGINARIUM_LAYOUTVLM_MAX_CONTAINMENT_ERROR"',
            source,
        )
        self.assertIn("vertical contact is retained.", source)
        self.assertIn("containment_pair_tensor,", source)
        self.assertNotIn("stack_pair_tensor", source)
        self.assertNotIn("Enabling strong stacked_on", source)
        self.assertIn('"boundary",', source)
        self.assertIn(
            "if layoutvlm_stage not in LAYOUTVLM_STAGES:",
            source,
        )
        self.assertIn("choices=LAYOUTVLM_STAGES", source)
        self.assertIn("Room boundary built from", source)
        self.assertIn("boundary_object_indices=", source)
        self.assertIn(
            'layoutvlm_stage in {"full", "depth"}',
            source,
        )
        self.assertIn('"full",', source)
        self.assertIn('"depth",', source)
        self.assertIn("build_depth_reprojection_observations(", source)
        self.assertIn("depth_observation_indices", source)
        self.assertIn("reference_centre_errors", source)
        self.assertIn(
            '"IMAGINARIUM_LAYOUTVLM_DEPTH_TRUST_WEIGHT"',
            source,
        )
        self.assertIn(
            '"IMAGINARIUM_LAYOUTVLM_DEPTH_CENTER_WEIGHT"',
            source,
        )
        self.assertIn(
            '"IMAGINARIUM_LAYOUTVLM_DEPTH_SIZE_WEIGHT"',
            source,
        )
        self.assertIn(
            '"IMAGINARIUM_LAYOUTVLM_DEPTH_METRIC_WEIGHT"',
            source,
        )
        self.assertIn(
            '"IMAGINARIUM_LAYOUTVLM_DEPTH_FREEZE_YAW"',
            source,
        )
        self.assertIn(
            '"IMAGINARIUM_LAYOUTVLM_DEPTH_CENTER_MARGIN_PX"',
            source,
        )
        self.assertIn(
            '"IMAGINARIUM_SCENEBA_DISCRETE_REPAIR"',
            source,
        )
        self.assertIn(
            '"IMAGINARIUM_LAYOUTVLM_ACTIVE_SET_ROUTER"',
            source,
        )
        self.assertIn(
            '"IMAGINARIUM_LAYOUTVLM_ROUTER_CHECKPOINTS"',
            source,
        )
        self.assertIn('"active_set_router": active_set_router', source)
        self.assertIn(
            "select_confident_discrete_pose_repairs(",
            source,
        )

    def test_dual_gpu_runner_forwards_depth_trust_region(self):
        source = (
            PROJECT_ROOT / "scripts" /
            "run_paper30_v4_s4_only_dual_gpu.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'IMAGINARIUM_LAYOUTVLM_DEPTH_TRUST_WEIGHT="$depth_trust_weight"',
            source,
        )
        self.assertIn(
            'IMAGINARIUM_LAYOUTVLM_DEPTH_CENTER_WEIGHT="$depth_center_weight"',
            source,
        )
        self.assertIn(
            'IMAGINARIUM_LAYOUTVLM_DEPTH_SIZE_WEIGHT="$depth_size_weight"',
            source,
        )
        self.assertIn(
            'IMAGINARIUM_LAYOUTVLM_DEPTH_METRIC_WEIGHT="$depth_metric_weight"',
            source,
        )
        self.assertIn(
            'IMAGINARIUM_LAYOUTVLM_DEPTH_FREEZE_YAW="$depth_freeze_yaw"',
            source,
        )
        self.assertIn(
            'IMAGINARIUM_LAYOUTVLM_DEPTH_CENTER_MARGIN_PX="$depth_center_margin_px"',
            source,
        )
        self.assertIn(
            'IMAGINARIUM_SCENEBA_DISCRETE_REPAIR="$sceneba_discrete_repair"',
            source,
        )
        self.assertIn(
            'IMAGINARIUM_LAYOUTVLM_ACTIVE_SET_ROUTER="$active_set_router"',
            source,
        )
        self.assertIn(
            'IMAGINARIUM_LAYOUTVLM_ROUTER_CHECKPOINTS="$router_checkpoints"',
            source,
        )
        self.assertIn(
            'IMAGINARIUM_SCENEBA_ASSET_CENTER_CANDIDATES="$sceneba_asset_center_candidates"',
            source,
        )
        self.assertIn(
            'IMAGINARIUM_SCENEBA_ASSET_CENTER_SCALES="$sceneba_asset_center_scales"',
            source,
        )
        self.assertIn(
            'IMAGINARIUM_SCENEBA_SUPPORT_SURFACE_CANDIDATES="$sceneba_support_surface_candidates"',
            source,
        )
        self.assertIn(
            'if test "$s4_engine" = "layoutvlm" && test "$layoutvlm_stage" = "depth"; then',
            source,
        )
        self.assertIn(
            'source_version="${IMAGINARIUM_S4_SOURCE_VERSION:-v4_deepsearch}"',
            source,
        )
        self.assertIn(
            'source_stage="${IMAGINARIUM_S4_SOURCE_STAGE:-S3_pose_inference}"',
            source,
        )
        self.assertIn(
            'reference_version="${IMAGINARIUM_S4_REFERENCE_VERSION:-v4}"',
            source,
        )
        self.assertIn(
            'reference_stage="${IMAGINARIUM_S4_REFERENCE_STAGE:-S4_layout_refinement}"',
            source,
        )
        self.assertIn(
            'reference_pattern="${IMAGINARIUM_S4_REFERENCE_PATTERN:-*_placement_info_s4.json}"',
            source,
        )
        self.assertIn(
            'IMAGINARIUM_LAYOUTVLM_REFERENCE_JSON="$reference_s4"',
            source,
        )
        self.assertIn(
            's4_engine="${IMAGINARIUM_S4_ENGINE:-layoutvlm}"',
            source,
        )
        self.assertIn(
            'runtime_gpu${gpu}.jsonl',
            source,
        )
        self.assertIn(
            'elapsed_seconds',
            source,
        )
        self.assertIn("local free rc worker_status=0", source)
        self.assertIn('if test "$completed" -ne "$expected"; then', source)
        self.assertIn(
            'test "$source_ready" -ne "$preflight_expected"',
            source,
        )
        self.assertIn(
            'IMAGINARIUM_LAYOUTVLM_SOLVER="$layoutvlm_solver"',
            source,
        )
        self.assertIn(
            'IMAGINARIUM_SCENELM_PCG_ITERATIONS="$scenelm_pcg_iterations"',
            source,
        )
        self.assertIn(
            'IMAGINARIUM_SCENELM_MAX_RELATION_RELEASES='
            '"$scenelm_max_relation_releases"',
            source,
        )
        self.assertIn(
            'IMAGINARIUM_SCENELM_COLLISION_WITNESS_WEIGHT='
            '"$scenelm_collision_witness_weight"',
            source,
        )

    def test_scenelm_is_opt_in_and_persists_solver_audit(self):
        source = (
            PROJECT_ROOT / "modules" / "S4_blender_layout_and_corr.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"IMAGINARIUM_LAYOUTVLM_SOLVER", "adam"',
            source,
        )
        self.assertIn('"solver": solver_name', source)
        self.assertIn('"scenelm_solver"', source)
        self.assertIn(
            '"scenelm_matrix_free_lm_v1"',
            source,
        )

    def test_v5_scenelm_persists_relation_chart_and_certificates(self):
        source = (
            PROJECT_ROOT / "modules" / "S4_blender_layout_and_corr.py"
        ).read_text(encoding="utf-8")
        runner = (
            PROJECT_ROOT / "scripts" / "run_scenelm_paired_smoke.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('solver_name in {"scenelm", "v5_scenelm"}', source)
        self.assertIn('"scenelm_relation_manifold_v1"', source)
        self.assertIn('"relation_coordinates"', source)
        self.assertIn('"stationarity_inf"', source)
        self.assertIn('"primal_feasibility_max"', source)
        self.assertIn("IMAGINARIUM_SCENELM_KINEMATIC_BACKSUB", source)
        self.assertIn('"kinematic_backsub_edges"', source)
        self.assertIn('child_obj.rigid_body.type = "PASSIVE"', source)
        self.assertIn('"kinematic_backsub_realization_error"', source)
        self.assertIn('"post_projection_certified_support_pairs"', source)
        self.assertIn('"kinematic_promoted_edges"', source)
        self.assertIn('"relation_release_count"', source)
        self.assertIn('"relation_released_object_indices"', source)
        self.assertIn('"collision_witness_count"', source)
        self.assertIn('scenelm_solver="${SCENELM_SOLVER:-scenelm}"', runner)
        self.assertIn(
            'IMAGINARIUM_LAYOUTVLM_SOLVER="$scenelm_solver"', runner
        )

    def test_sceneproof_program_ir_is_opt_in_and_persisted(self):
        source = (
            PROJECT_ROOT / "modules" / "S4_blender_layout_and_corr.py"
        ).read_text(encoding="utf-8")
        self.assertIn("IMAGINARIUM_SCENEPROOF_PROGRAM_IR", source)
        self.assertIn("compile_legacy_relation_programs", source)
        self.assertIn('"sceneproof_relation_programs"', source)
        self.assertIn('"sceneproof_live_factor_parity"', source)
        self.assertIn("IMAGINARIUM_SCENEPROOF_REQUIRE_FACTOR_PARITY", source)
        self.assertIn("audit_live_factor_parity", source)
        self.assertIn("IMAGINARIUM_SCENEPROOF_SHADOW_RESIDUAL_PARITY", source)
        self.assertIn("IMAGINARIUM_SCENEPROOF_USE_PROGRAM_RESIDUALS", source)
        self.assertIn("IMAGINARIUM_SCENEPROOF_RESIDUAL_FALLBACK", source)
        self.assertIn("IMAGINARIUM_SCENEPROOF_REQUIRE_BINDING_AUDIT", source)
        self.assertIn('"sceneproof_factor_binding_audit"', source)
        self.assertIn('"sceneproof_shadow_residual_parity"', source)
        self.assertIn(
            "IMAGINARIUM_SCENEPROOF_SHADOW_JACOBIAN_OWNERSHIP",
            source,
        )
        self.assertIn('"sceneproof_jacobian_ownership"', source)
        self.assertIn(
            "IMAGINARIUM_SCENEPROOF_FULL_SO3_GUARDED_SCHUR",
            source,
        )
        self.assertIn(
            "IMAGINARIUM_SCENEPROOF_IN_LOOP_GUARDED_SCHUR",
            source,
        )
        self.assertIn('"sceneproof_full_so3_guarded_schur"', source)
        self.assertIn("audit_only_until_block_parity", source)

    def test_layoutvlm_uses_frozen_s3_geometry_for_physical_factors(self):
        source = (
            PROJECT_ROOT / "modules" / "S4_blender_layout_and_corr.py"
        ).read_text(encoding="utf-8")
        self.assertIn('source_info.get("bbox")', source)
        self.assertIn(
            'source_info.get("pose_matrix_for_blender")', source
        )
        self.assertIn("np.linalg.inv(frozen_pose)", source)
        self.assertIn("_convex_hull_indices_2d", source)
        self.assertIn("local_corner_batches", source)
        self.assertIn("footprint_hull_sizes", source)
        self.assertIn("Post-simulation support retraction complete", source)
        self.assertIn("postsim_max_containment_error_m", source)
        self.assertIn("IMAGINARIUM_LAYOUTVLM_GEOMETRY_SNAPSHOT", source)
        self.assertIn("IMAGINARIUM_SCENELM_POSTSIM_MAX_SHIFT_M", source)
        self.assertIn("projected_max_collision_penetration", source)
        ops_source = (
            PROJECT_ROOT / "modules" / "_s4_layoutvlm_ops.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'history[-1]["projected_max_collision_penetration"],',
            ops_source,
        )
        self.assertIn(
            "candidate_pose_matrices[child_index, :2, 3]", source
        )
        self.assertIn(
            "working_pose_matrices[child_index, :2, 3]", source
        )
        self.assertNotIn(
            "edge_translation[child_index, :2]\n                ).item()",
            source,
        )
        self.assertNotIn(
            "local_corners = local_box_corners(local_minimum, local_maximum)",
            source,
        )
        self.assertIn(
            "[LayoutVLM] Optimization geometry source:", source
        )
        self.assertIn("blender_fallback=", source)

    def test_true_mesh_com_audit_is_read_only_and_separate_from_solver(self):
        source = (
            PROJECT_ROOT / "modules" / "S4_blender_layout_and_corr.py"
        ).read_text(encoding="utf-8")
        runner = (
            PROJECT_ROOT
            / "scripts"
            / "run_sceneproof_true_mesh_com_paper30_fix62.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("audit_sceneproof_true_mesh_com_support", source)
        self.assertIn("_sceneproof_world_trimesh", source)
        self.assertIn("true_mesh_filled_voxel_uniform_density", source)
        self.assertIn("filled_voxel_mass_properties_unproven", source)
        self.assertIn("binary_fill_holes", source)
        self.assertIn("voxel_heightfield_contact_points", source)
        self.assertIn("cyclic_support_component_unproven", source)
        self.assertIn("supporter's vertical", source)
        self.assertIn("authoritative_declared_parent_plus_measured_contact", source)
        self.assertIn("IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER", source)
        self.assertIn("mutates_placement", source)
        self.assertIn("IMAGINARIUM_SCENEPROOF_TRUE_MESH_COM_AUDIT_OUTPUT", runner)
        self.assertIn("sceneproof_true_mesh_com_responsibility_audit.py", runner)

    def test_true_mesh_com_counterfactual_is_audit_only(self):
        source = (
            PROJECT_ROOT / "sceneproof_true_mesh_com_counterfactual_oracle.py"
        ).read_text(encoding="utf-8")
        self.assertIn("positive_macro_oracle_gain", source)
        self.assertIn("safe_against_incumbent", source)
        self.assertIn("rollback_poses", source)
        self.assertIn("required_fail_closed_rollback_object_ids", source)
        self.assertIn("meaningful_gain_tolerance", source)
        self.assertIn("grounded_cycle_object_ids", source)
        self.assertNotIn("output_path.write_text", source)
        protocol = (
            PROJECT_ROOT / "sceneproof_true_mesh_com_counterfactual_protocol.py"
        ).read_text(encoding="utf-8")
        self.assertIn("materialize_scene_local_scoped_rollbacks", protocol)
        self.assertIn("safe_abstain_scenes", protocol)

    def test_com_scoped_materialization_is_exact_and_regated(self):
        materializer = (
            PROJECT_ROOT / "sceneproof_com_scoped_rollback_materialize.py"
        ).read_text(encoding="utf-8")
        gate = (
            PROJECT_ROOT / "sceneproof_com_scoped_rollback_gate.py"
        ).read_text(encoding="utf-8")
        self.assertIn("expected_remaining", materializer)
        self.assertIn("oracle rollback contains an unchanged object", materializer)
        self.assertIn("physical_macro_improves_fix61", gate)
        self.assertIn("object_recovery", gate)
        self.assertIn("scene_graph_parent_accuracy_gt", gate)
        render = (
            PROJECT_ROOT
            / "scripts"
            / "render_sceneproof_com_scoped_rollback_fix69.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("expected 6 authorized scenes", render)
        self.assertIn("source_s3_scene_camera_locked", (
            PROJECT_ROOT / "scripts" / "render_sceneproof_certified_paper30.sh"
        ).read_text(encoding="utf-8"))

    def test_support_visual_identity_audit_is_read_only(self):
        source = (
            PROJECT_ROOT / "sceneproof_support_visual_identity_audit.py"
        ).read_text(encoding="utf-8")
        runner = (
            PROJECT_ROOT
            / "scripts"
            / "run_sceneproof_support_visual_identity_smoke1_fix70.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("audit_only_s1_identity_pose_ownership_com_certificate", source)
        self.assertIn('"mutates_placement": False', source)
        self.assertIn("rollback_not_materialized_exactly", source)
        self.assertIn("uncertified_change_retained", source)
        self.assertIn("unstable_change_retained", source)
        self.assertNotIn("rollback_poses(", source)
        self.assertIn("support_visual_identity_smoke1_fix70", runner)

    def test_render_identity_audit_labels_final_materialized_meshes(self):
        source = (
            PROJECT_ROOT / "modules" / "S4_blender_layout_and_corr.py"
        ).read_text(encoding="utf-8")
        runner = (
            PROJECT_ROOT
            / "scripts"
            / "run_sceneproof_render_identity_smoke1_fix71.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("IMAGINARIUM_SCENEPROOF_RENDER_IDENTITY_AUDIT_OUTPUT", source)
        self.assertIn("annotated_color_id_output_path", source)
        self.assertIn("IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER=1", runner)
        self.assertIn("v5_sceneproof_com_scoped_rollback_paper30_fix68", runner)

    def test_render_identity_triplet_compares_visual_pose_ownership(self):
        compare = (
            PROJECT_ROOT / "sceneproof_render_identity_triplet_compare.py"
        ).read_text(encoding="utf-8")
        runner = (
            PROJECT_ROOT
            / "scripts"
            / "run_sceneproof_render_identity_triplet_smoke1_fix72.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("pose_render_ownership_mismatch", compare)
        self.assertIn(
            "audit_only_locked_camera_rendered_centroid_ownership", compare
        )
        self.assertIn("SCENEPROOF_IDENTITY_VERSION", runner)
        self.assertIn("render_identity_triplet_compare.py", runner)

    def test_locked_render_cache_is_pose_fresh_and_forceable(self):
        runner = (
            PROJECT_ROOT / "scripts" / "render_sceneproof_certified_paper30.sh"
        ).read_text(encoding="utf-8")
        smoke = (
            PROJECT_ROOT
            / "scripts"
            / "rerender_sceneproof_fix68_bedroom_smoke1_fix73.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('test "$render" -nt "$placement"', runner)
        self.assertIn('test "$audit" -nt "$placement"', runner)
        self.assertIn("SCENEPROOF_RENDER_FORCE", runner)
        self.assertIn("SCENEPROOF_RENDER_FORCE=1", smoke)

    def test_s4_serializes_settled_poses_without_rigidbody_ownership(self):
        source = (
            PROJECT_ROOT / "modules" / "S4_blender_layout_and_corr.py"
        ).read_text(encoding="utf-8")
        self.assertIn("all_placement_owned_blender_roots", source)
        self.assertIn("serialized_without_rigid_body_ids", source)
        self.assertIn("Pose serialization/render parity", source)
        self.assertNotIn("if obj.type == 'MESH' and obj.rigid_body:", source)
        runner = (
            PROJECT_ROOT
            / "scripts"
            / "run_sceneproof_pose_serialization_smoke1_fix76.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("fix76_inprocess_bedroom.png", runner)
        self.assertIn("fix76_roundtrip_bedroom.png", runner)
        self.assertIn("SCENEPROOF_RENDER_FORCE=1", runner)

    def test_com_audit_separates_intrinsic_tipping_from_parent_overhang(self):
        source = (
            PROJECT_ROOT / "modules" / "S4_blender_layout_and_corr.py"
        ).read_text(encoding="utf-8")
        router = (
            PROJECT_ROOT / "sceneproof_com_action_audit_fix78.py"
        ).read_text(encoding="utf-8")
        self.assertIn("intrinsic_child_contact_margin_m", source)
        self.assertIn("declared_parent_surface_margin_m", source)
        self.assertIn("local_gravity_settle_probe_candidate", router)
        self.assertIn("com_projection_candidate", router)

    def test_benchmark_suite_runs_fair_legacy_layout_and_composite_timing(self):
        source = (
            PROJECT_ROOT / "scripts" / "run_paper30_s4_benchmark_suite.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("IMAGINARIUM_S4_ENGINE=legacy", source)
        self.assertIn("IMAGINARIUM_LAYOUTVLM_ITERATIONS=5000", source)
        self.assertIn("IMAGINARIUM_S4_ENGINE=layoutvlm", source)
        self.assertIn("IMAGINARIUM_LAYOUTVLM_ITERATIONS=400", source)
        self.assertIn("IMAGINARIUM_LAYOUTVLM_DEPTH_FREEZE_YAW=1", source)
        self.assertIn('--runtime-composite "$depth_version=$layout_version+$depth_version"', source)
        self.assertIn("--min-visible-mask-area 8000", source)

    def test_active_router_eval_can_preserve_versioned_audits(self):
        source = (
            PROJECT_ROOT
            / "scripts"
            / "eval_sceneba_active_router_smoke5.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("SCENEBA_ACTIVE_ROUTER_AUDIT_DIR", source)

    def test_s4_refinement_skips_camera_before_support_lookup(self):
        source = (
            PROJECT_ROOT / "modules" / "S4_blender_layout_and_corr.py"
        ).read_text(encoding="utf-8")
        camera_guard = source.index(
            "if 'scene_camera' in obj_name.lower():",
            source.index(
                "for obj_name, obj_info in "
                "obj_placement_info['obj_info'].items():"
            ),
        )
        support_lookup = source.index(
            "obj_info['SpatialRel'] = "
            "'on' if obj_info.get('supported') else None",
            camera_guard,
        )
        self.assertLess(camera_guard, support_lookup)
        simplify_start = source.index("def simplify_placement(")
        simplify_end = source.index(
            "# ==================== 辅助函数",
            simplify_start,
        )
        simplify_source = source[simplify_start:simplify_end]
        self.assertNotIn(
            "parent = obj_info['supported']",
            simplify_source,
        )
        self.assertIn("obj_info.get('supported')", simplify_source)
        self.assertIn(
            "'scene_camera' in obj_name.lower()",
            simplify_source,
        )
        self.assertIn(
            '"IMAGINARIUM_LAYOUTVLM_REFERENCE_JSON"',
            source,
        )
        self.assertIn(
            "[LayoutVLM] Loaded frozen full-400 pose reference:",
            source,
        )

    def test_layout_module_forwards_gate_to_blender(self):
        source = (PROJECT_ROOT / "modules" / "layout.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('os.environ.get("IMAGINARIUM_USE_LAYOUTVLM", "0")', source)
        self.assertIn('"--use_layoutvlm"', source)
        self.assertIn('"--layoutvlm_stage"', source)


if __name__ == "__main__":
    unittest.main()
