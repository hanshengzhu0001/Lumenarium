import math
import unittest

try:
    import torch
except (ImportError, OSError):  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required")
class SceneProofResidualBridgeTest(unittest.TestCase):
    def test_independent_shadow_assembly_and_parity(self):
        from modules._sceneproof_residual_bridge import (
            assemble_program_shadow_residuals,
            residual_parity,
        )

        dtype = torch.float64
        zero = torch.empty((0,), dtype=dtype)
        base = torch.eye(4, dtype=dtype).reshape(1, 4, 4)
        translation = base[:, :3, 3].clone()
        shadow = assemble_program_shadow_residuals(
            flattened=torch.zeros(4, dtype=dtype),
            collision_values=torch.tensor([0.25], dtype=dtype),
            collision_weight=2.0,
            collision_mass=1,
            witness_values=zero,
            witness_weight=1.0,
            witness_mass=0,
            contact_values=torch.tensor([0.5], dtype=dtype),
            contact_weight=4.0,
            contact_mass=1,
            plane_values=zero,
            plane_weight=1.0,
            plane_mass=0,
            orientation_values=zero,
            orientation_weight=1.0,
            containment_values=zero,
            containment_weight=1.0,
            containment_mass=0,
            distance_values=zero,
            align_values=zero,
            point_values=zero,
            semantic_weight=1.0,
            semantic_mass=0,
            boundary_values=zero,
            boundary_weight=1.0,
            boundary_mass=0,
            current_depth=torch.tensor(0.0, dtype=dtype),
            current_depth_trust=torch.tensor(0.0, dtype=dtype),
            depth_observation_count=0,
            depth_reprojection_weight=0.0,
            depth_trust_region_weight=0.0,
            current_yaw=torch.zeros(1, dtype=dtype),
            current_translation=translation,
            base_matrices=base,
            optimize_yaw=True,
            warm_start_weight=0.0,
        )
        expected = torch.tensor(
            [math.sqrt(0.5 + 1e-12), 1.0], dtype=dtype
        )
        report = residual_parity(expected, shadow)
        self.assertTrue(report["passed"])
        self.assertLessEqual(report["max_abs_error"], 1e-12)

    def test_residual_parity_rejects_shape_or_value_mismatch(self):
        from modules._sceneproof_residual_bridge import residual_parity

        self.assertFalse(
            residual_parity(torch.zeros(1), torch.zeros(2))["passed"]
        )
        self.assertFalse(
            residual_parity(torch.zeros(1), torch.ones(1))["passed"]
        )


@unittest.skipIf(torch is None, "PyTorch is required")
class SceneProofSO3ChartTest(unittest.TestCase):
    def _chart(self):
        from modules._s4_scenelm_relational import (
            compile_full_so3_relation_coordinates,
        )

        base = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)
        base[1, :3, 3] = torch.tensor([2.0, 0.0, 1.0], dtype=torch.float64)
        base[2, :3, 3] = torch.tensor([2.5, 0.0, 1.5], dtype=torch.float64)
        chart = compile_full_so3_relation_coordinates(
            base,
            torch.tensor([[1, 0], [2, 1]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 3), dtype=torch.float64),
        )
        return base, chart

    def test_zero_delta_preserves_full_matrices(self):
        base, chart = self._chart()
        actual = chart.pose_matrices(chart.zero_parameters())
        self.assertTrue(torch.allclose(actual, base, atol=1e-12))
        self.assertEqual(chart.metadata()["rotation_mode"], "independent_world_so3")

    def test_parent_rotation_carries_child_center_but_not_child_orientation(self):
        base, chart = self._chart()
        parameters = chart.zero_parameters()
        root = chart.blocks[0]
        parameters[root.rotation_slice] = torch.tensor(
            [0.0, 0.0, math.pi / 2], dtype=torch.float64
        )
        pose = chart.pose_matrices(parameters)
        self.assertTrue(
            torch.allclose(
                pose[1, :3, 3],
                torch.tensor([0.0, 2.0, 1.0], dtype=torch.float64),
                atol=1e-9,
            )
        )
        # The child has a zero independent SO(3) tangent, so its orientation
        # remains its base orientation even when the parent rotates.
        self.assertTrue(torch.allclose(pose[1, :3, :3], base[1, :3, :3], atol=1e-9))

    def test_child_has_three_independent_rotation_parameters(self):
        base, chart = self._chart()
        parameters = chart.zero_parameters()
        child = chart.blocks[1]
        self.assertEqual(child.rotation_stop - child.rotation_start, 3)
        parameters[child.rotation_slice] = torch.tensor(
            [0.15, -0.10, 0.05], dtype=torch.float64
        )
        pose = chart.pose_matrices(parameters)
        self.assertFalse(torch.allclose(pose[1, :3, :3], base[1, :3, :3]))
        self.assertTrue(torch.allclose(pose[0], base[0], atol=1e-12))

    def test_so3_exp_is_orthogonal_and_differentiable(self):
        from modules._s4_scenelm_relational import so3_exp

        vector = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64, requires_grad=True)
        rotation = so3_exp(vector)
        identity = rotation.transpose(0, 1) @ rotation
        self.assertTrue(torch.allclose(identity, torch.eye(3, dtype=torch.float64), atol=1e-10))
        loss = rotation[0, 1] + 2.0 * rotation[2, 0]
        loss.backward()
        self.assertTrue(torch.isfinite(vector.grad).all())

    def test_step_cap_limits_full_rotation_norm(self):
        _, chart = self._chart()
        step = torch.ones(chart.parameter_count, dtype=torch.float64)
        capped = chart.cap_step(
            step,
            max_translation=0.25,
            max_rotation_radians=math.radians(15.0),
        )
        for block in chart.blocks:
            self.assertLessEqual(
                torch.linalg.vector_norm(capped[block.rotation_slice]).item(),
                math.radians(15.0) + 1e-12,
            )

    def test_only_leaf_translation_is_marked_eliminable(self):
        _, chart = self._chart()
        self.assertFalse(chart.blocks[0].eliminable_translation)
        self.assertFalse(chart.blocks[1].eliminable_translation)
        self.assertTrue(chart.blocks[2].eliminable_translation)
        indices = set(chart.leaf_translation_parameter_indices().tolist())
        self.assertEqual(
            indices,
            set(range(chart.blocks[2].translation_start, chart.blocks[2].translation_stop)),
        )
        for block in chart.blocks:
            self.assertEqual(block.parameter_start, block.rotation_start)
            self.assertEqual(block.parameter_stop, block.translation_stop)
            self.assertEqual(
                block.parameter_slice,
                slice(block.rotation_start, block.translation_stop),
            )
            self.assertTrue(indices.isdisjoint(range(block.rotation_start, block.rotation_stop)))

    def test_warm_start_anchored_plane_chart_allows_only_normal_translation(self):
        from modules._s4_scenelm_relational import (
            compile_full_so3_relation_coordinates,
        )

        base = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        base[0, :3, 3] = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
        normal = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        chart = compile_full_so3_relation_coordinates(
            base,
            torch.empty((0, 2), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            normal,
            warm_start_anchored_plane_translation=True,
        )
        block = chart.blocks[0]
        self.assertEqual(block.mode, "plane_n_anchored")
        self.assertEqual(block.translation_dofs, 1)
        parameters = chart.zero_parameters()
        parameters[block.translation_slice] = 0.25
        parameters[block.rotation_slice] = torch.tensor(
            [0.1, -0.2, 0.3], dtype=torch.float64
        )
        pose = chart.pose_matrices(parameters)
        delta = pose[0, :3, 3] - base[0, :3, 3]
        self.assertTrue(
            torch.allclose(delta, torch.tensor([0.25, 0.0, 0.0], dtype=torch.float64))
        )
        self.assertFalse(torch.allclose(pose[0, :3, :3], base[0, :3, :3]))
        self.assertEqual(
            chart.metadata()["plane_translation_policy"],
            "warm_start_anchored_normal_only",
        )


@unittest.skipIf(torch is None, "PyTorch is required")
class SceneProofBlockSystemTest(unittest.TestCase):
    def test_block_normal_assembly_matches_dense_reference(self):
        from modules._sceneproof_block_system import (
            LinearizedFactor,
            assemble_normal_system,
        )

        residual = torch.tensor([0.2, -0.4], dtype=torch.float64)
        first = torch.tensor([[1.0, 2.0], [0.5, -1.0]], dtype=torch.float64)
        second = torch.tensor([[0.0], [3.0]], dtype=torch.float64)
        factor = LinearizedFactor(
            factor_id="support:0:1",
            residual=residual,
            jacobians={"object:0": first, "object:1": second},
            object_indices=(0, 1),
        )
        normal, gradient, diagnostics = assemble_normal_system(
            parameter_count=3,
            slices={"object:0": slice(0, 2), "object:1": slice(2, 3)},
            factors=(factor,),
            damping=0.0,
            dtype=torch.float64,
            device="cpu",
        )
        dense = torch.cat((first, second), dim=1)
        self.assertTrue(torch.allclose(normal, dense.T @ dense, atol=1e-12))
        self.assertTrue(
            torch.allclose(gradient, dense.T @ residual, atol=1e-12)
        )
        self.assertEqual(diagnostics["factors"], 1)

    def test_jacobian_ownership_detects_leakage_and_inactive_collision(self):
        from modules._sceneproof_block_system import (
            ResidualSliceBinding,
            audit_jacobian_block_ownership,
        )

        residuals = torch.tensor([0.2, 1e-6], dtype=torch.float64)
        jacobian = torch.tensor(
            [[1.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        bindings = (
            ResidualSliceBinding("support:0:contact", "support", 0, 1, (0, 1)),
            ResidualSliceBinding("collision:0:1", "collision", 1, 2, (0, 1), (0, 1)),
        )
        report = audit_jacobian_block_ownership(
            residuals=residuals,
            jacobian=jacobian,
            bindings=bindings,
            object_parameter_slices={0: (slice(0, 2),), 1: (slice(2, 4),)},
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["active_collision_factor_ids"], [])
        self.assertEqual(report["inactive_collision_factor_ids"], ["collision:0:1"])

        leaked = audit_jacobian_block_ownership(
            residuals=residuals[:1],
            jacobian=torch.tensor([[1.0, 0.0, 0.0, 0.4]], dtype=torch.float64),
            bindings=(ResidualSliceBinding("unary:0", "unary", 0, 1, (0,)),),
            object_parameter_slices={0: (slice(0, 2),), 1: (slice(2, 4),)},
        )
        self.assertFalse(leaked["passed"])
        self.assertGreater(leaked["leakage"][0]["max_abs_leakage"], 0.0)

    def test_stability_requires_consecutive_active_linearizations(self):
        from modules._sceneproof_block_system import LinearizationStabilityTracker

        tracker = LinearizationStabilityTracker(required_consecutive=2)
        self.assertEqual(tracker.update(["support"]), ())
        self.assertEqual(tracker.update(["support"]), ("support",))
        self.assertEqual(tracker.update([]), ())
        self.assertEqual(tracker.counts()["support"], 0)

    def test_collision_trial_rolls_back_and_releases_local_separator(self):
        from modules._sceneproof_block_system import (
            ResidualSliceBinding,
            guarded_collision_trial,
        )

        binding = ResidualSliceBinding(
            "collision:1:2", "collision", 0, 1, (1, 2), (1, 2)
        )

        def collision(parameters):
            return torch.relu(parameters[:1] - 0.1)

        incumbent = torch.tensor([0.0], dtype=torch.float64)
        candidate = torch.tensor([0.3], dtype=torch.float64)
        result, audit = guarded_collision_trial(
            incumbent_parameters=incumbent,
            candidate_parameters=candidate,
            collision_bindings=(binding,),
            evaluate_collision_residuals=collision,
            parent_by_object={1: 0, 2: -1},
        )
        self.assertTrue(torch.equal(result, incumbent))
        self.assertFalse(audit["accepted"])
        self.assertEqual(audit["failed_factor_ids"], ["collision:1:2"])
        self.assertEqual(audit["released_object_indices"], [0, 1, 2])

    def test_collision_release_is_scoped_to_child_external_and_parent(self):
        from modules._sceneproof_block_system import (
            ResidualSliceBinding,
            guarded_collision_trial,
        )

        binding = ResidualSliceBinding(
            "collision:1:2", "collision", 0, 1, (1, 2), (1, 2)
        )

        def collision(parameters):
            return torch.relu(parameters[:1] - 0.1)

        _, audit = guarded_collision_trial(
            incumbent_parameters=torch.tensor([0.0], dtype=torch.float64),
            candidate_parameters=torch.tensor([0.3], dtype=torch.float64),
            collision_bindings=(binding,),
            evaluate_collision_residuals=collision,
            parent_by_object={1: 0, 2: 3},
            primary_child_objects=(1,),
        )
        self.assertEqual(audit["released_child_indices"], [1])
        self.assertEqual(audit["external_separator_indices"], [2])
        self.assertEqual(audit["parent_separator_indices"], [0])
        self.assertEqual(audit["released_object_indices"], [0, 1, 2])

    def test_partial_commit_restores_complete_object_blocks(self):
        from modules._s4_scenelm_relational import (
            compile_full_so3_relation_coordinates,
        )
        from modules._sceneproof_block_system import (
            rollback_object_parameter_blocks,
        )

        base = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)
        chart = compile_full_so3_relation_coordinates(
            base,
            torch.tensor([[1, 0]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 3), dtype=torch.float64),
        )
        incumbent = chart.zero_parameters()
        candidate = torch.arange(
            chart.parameter_count, dtype=torch.float64
        ) + 1.0
        restored, audit = rollback_object_parameter_blocks(
            incumbent_parameters=incumbent,
            candidate_parameters=candidate,
            coordinates=chart,
            object_indices=(1,),
        )
        block = chart.blocks[1]
        torch.testing.assert_close(
            restored[block.parameter_slice],
            incumbent[block.parameter_slice],
        )
        for index, other in enumerate(chart.blocks):
            if index != 1:
                torch.testing.assert_close(
                    restored[other.parameter_slice],
                    candidate[other.parameter_slice],
                )
        self.assertEqual(audit["restored_object_indices"], [1])
        self.assertEqual(audit["rotation_parameters_restored"], 3)

    def test_stable_leaf_elimination_never_removes_rotation_blocks(self):
        from modules._s4_scenelm_relational import compile_full_so3_relation_coordinates
        from modules._sceneproof_block_system import (
            ResidualSliceBinding,
            stable_leaf_translation_objects,
        )

        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        chart = compile_full_so3_relation_coordinates(
            base,
            torch.tensor([[1, 0]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 3), dtype=torch.float64),
        )
        binding = ResidualSliceBinding("support", "support", 0, 1, (0, 1))
        self.assertEqual(
            stable_leaf_translation_objects(
                coordinates=chart,
                bindings=(binding,),
                stable_active_factor_ids=("support",),
            ),
            (1,),
        )
        rotation_indices = {
            index
            for block in chart.blocks
            for index in range(block.rotation_start, block.rotation_stop)
        }
        self.assertTrue(
            rotation_indices.isdisjoint(chart.leaf_translation_parameter_indices().tolist())
        )

    def test_leaf_translation_schur_matches_full_solve_and_retains_rotations(self):
        from modules._s4_scenelm_relational import (
            compile_full_so3_relation_coordinates,
        )
        from modules._sceneproof_block_system import (
            solve_with_leaf_translation_schur,
        )

        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[1, 2, 3] = 1.0
        chart = compile_full_so3_relation_coordinates(
            base,
            torch.tensor([[1, 0]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 3), dtype=torch.float64),
        )
        count = chart.parameter_count
        generator = torch.Generator().manual_seed(7)
        matrix = torch.randn((count, count), dtype=torch.float64, generator=generator)
        normal = matrix.transpose(0, 1) @ matrix + 0.5 * torch.eye(count, dtype=torch.float64)
        rhs = torch.randn(count, dtype=torch.float64, generator=generator)
        actual, diagnostics = solve_with_leaf_translation_schur(
            normal,
            rhs,
            chart,
            factor_object_incidence=((0,), (1,), (0, 1)),
            jitter=0.0,
        )
        expected = torch.linalg.solve(normal, rhs)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-9))
        self.assertEqual(diagnostics["eliminated_leaf_objects"], [1])
        self.assertEqual(diagnostics["rotation_parameters_eliminated"], 0)

    def test_schur_respects_audited_leaf_allowlist(self):
        from modules._s4_scenelm_relational import (
            compile_full_so3_relation_coordinates,
        )
        from modules._sceneproof_block_system import (
            solve_with_leaf_translation_schur,
        )

        base = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)
        chart = compile_full_so3_relation_coordinates(
            base,
            torch.tensor([[1, 0], [2, 0]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 3), dtype=torch.float64),
        )
        normal = torch.eye(chart.parameter_count, dtype=torch.float64)
        rhs = torch.ones(chart.parameter_count, dtype=torch.float64)
        _, diagnostics = solve_with_leaf_translation_schur(
            normal,
            rhs,
            chart,
            factor_object_incidence=((0, 1), (0, 2)),
            allowed_leaf_objects=(1,),
            jitter=0.0,
        )
        self.assertEqual(diagnostics["eliminated_leaf_objects"], [1])
        self.assertEqual(diagnostics["rotation_parameters_eliminated"], 0)

    def test_cross_edge_prevents_leaf_elimination(self):
        from modules._s4_scenelm_relational import (
            compile_full_so3_relation_coordinates,
        )
        from modules._sceneproof_block_system import (
            eligible_leaf_translation_objects,
        )

        base = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)
        chart = compile_full_so3_relation_coordinates(
            base,
            torch.tensor([[1, 0]], dtype=torch.long),
            torch.tensor([0, 2], dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 3), dtype=torch.float64),
        )
        self.assertEqual(
            eligible_leaf_translation_objects(chart, ((0, 1),)),
            (1,),
        )
        self.assertEqual(
            eligible_leaf_translation_objects(chart, ((0, 1), (1, 2))),
            (),
        )

    def test_responsibility_mask_freezes_unrelated_root_parameters(self):
        from modules._sceneproof_block_system import (
            restrict_normal_system_to_parameter_mask,
        )

        normal = torch.tensor(
            [[4.0, 1.0, 2.0], [1.0, 3.0, 1.0], [2.0, 1.0, 5.0]],
            dtype=torch.float64,
        )
        gradient = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        restricted, rhs, audit = restrict_normal_system_to_parameter_mask(
            normal,
            gradient,
            torch.tensor([True, False, True]),
        )
        step = torch.linalg.solve(restricted, -rhs)
        self.assertEqual(step[1].item(), 0.0)
        self.assertEqual(restricted[1, 1].item(), 1.0)
        self.assertEqual(restricted[0, 1].item(), 0.0)
        self.assertEqual(restricted[1, 2].item(), 0.0)
        self.assertEqual(audit, {"active_parameters": 2, "frozen_parameters": 1})

    def test_stationarity_requires_both_small_gradient_and_small_step(self):
        from modules._sceneproof_block_system import (
            certify_projected_stationarity,
        )

        mask = torch.tensor([True, True, False])
        certified = certify_projected_stationarity(
            torch.tensor([1e-12, 0.0, 9.0], dtype=torch.float64),
            torch.tensor([0.0, 1e-12, 9.0], dtype=torch.float64),
            mask,
            tolerance=1e-9,
        )
        self.assertTrue(certified["certified"])
        self.assertGreaterEqual(certified["effective_tolerance"], 1e-9)
        rejected = certify_projected_stationarity(
            torch.tensor([1e-4, 0.0, 0.0], dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
            mask,
            tolerance=1e-9,
        )
        self.assertFalse(rejected["certified"])

    def test_positive_spanning_poll_is_unit_aware_and_responsibility_scoped(self):
        from modules._sceneproof_block_system import (
            positive_spanning_poll_steps,
        )

        active = torch.tensor([True, False, True, True])
        rotations = torch.tensor([True, True, False, False])
        steps = positive_spanning_poll_steps(
            active,
            rotations,
            rotation_radius=0.1,
            translation_radius=0.02,
        )
        self.assertEqual(len(steps), 6)
        self.assertEqual(
            [(index, sign) for index, sign, _ in steps],
            [(0, -1), (0, 1), (2, -1), (2, 1), (3, -1), (3, 1)],
        )
        for index, sign, step in steps:
            expected = 0.1 if index == 0 else 0.02
            self.assertAlmostEqual(step[index].item(), sign * expected)
            self.assertEqual(torch.count_nonzero(step).item(), 1)

    def test_global_so3_cap_preserves_direction_and_all_block_limits(self):
        from modules._s4_scenelm_relational import (
            compile_full_so3_relation_coordinates,
        )

        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        chart = compile_full_so3_relation_coordinates(
            base,
            torch.tensor([[1, 0]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 3), dtype=torch.float64),
        )
        step = torch.arange(
            1, chart.parameter_count + 1, dtype=torch.float64
        )
        capped, scale = chart.cap_step_globally(
            step,
            max_translation=0.2,
            max_rotation_radians=0.1,
        )
        self.assertGreater(scale, 0.0)
        self.assertLessEqual(scale, 1.0)
        self.assertTrue(torch.allclose(capped, step * scale))
        for block in chart.blocks:
            self.assertLessEqual(
                torch.linalg.vector_norm(capped[block.rotation_slice]).item(),
                0.1 + 1e-12,
            )
            self.assertLessEqual(
                torch.linalg.vector_norm(capped[block.translation_slice]).item(),
                0.2 + 1e-12,
            )


if __name__ == "__main__":
    unittest.main()
