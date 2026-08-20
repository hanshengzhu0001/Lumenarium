import math
import os
import unittest
from unittest import mock

try:
    import torch
except (ImportError, OSError):  # pragma: no cover - torch may be unavailable/blocked
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required")
class LayoutVLMPoseOpsTest(unittest.TestCase):
    def setUp(self):
        from modules import _s4_layoutvlm_ops as ops

        self.ops = ops

    def test_zero_delta_round_trip_preserves_full_matrix(self):
        base = torch.tensor(
            [
                [
                    [0.0, -2.0, 0.0, 1.25],
                    [3.0, 0.0, 0.0, -2.5],
                    [0.0, 0.0, 4.0, 0.75],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float64,
        )
        error = self.ops.identity_reprojection_error(base)
        self.assertLessEqual(error.item(), 1e-12)

    def test_matrix_free_pcg_solves_positive_definite_system(self):
        matrix = torch.tensor(
            [[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64
        )
        rhs = torch.tensor([1.0, 2.0], dtype=torch.float64)
        expected = torch.linalg.solve(matrix, rhs)
        actual, diagnostics = self.ops.matrix_free_pcg(
            lambda vector: matrix @ vector,
            rhs,
            maximum_iterations=8,
            relative_tolerance=1e-10,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-9))
        self.assertLess(diagnostics["relative_residual"], 1e-8)

    def test_matrix_free_lm_accepts_reducing_nonlinear_step(self):
        parameters = torch.tensor([3.0, -2.0], dtype=torch.float64)

        def residual_function(value):
            return torch.stack(
                (
                    value[0] - 1.0,
                    2.0 * (value[1] + 1.0),
                )
            )

        candidate, diagnostics = self.ops.matrix_free_lm_step(
            residual_function,
            parameters,
            damping=1e-3,
            pcg_iterations=8,
            pcg_tolerance=1e-10,
        )
        self.assertEqual(diagnostics["accepted"], 1.0)
        self.assertLess(
            residual_function(candidate).square().sum().item(),
            residual_function(parameters).square().sum().item(),
        )
        self.assertTrue(
            torch.allclose(
                candidate,
                torch.tensor([1.0, -1.0], dtype=torch.float64),
                atol=2e-3,
            )
        )

    def test_matrix_free_lm_parameter_mask_freezes_coordinates(self):
        parameters = torch.tensor([3.0, -2.0], dtype=torch.float64)

        def residual_function(value):
            return torch.stack((value[0] - 1.0, value[1] + 1.0))

        candidate, diagnostics = self.ops.matrix_free_lm_step(
            residual_function,
            parameters,
            damping=1e-3,
            pcg_iterations=8,
            pcg_tolerance=1e-10,
            parameter_mask=torch.tensor([True, False]),
        )
        self.assertEqual(diagnostics["accepted"], 1.0)
        self.assertAlmostEqual(candidate[0].item(), 1.0, places=2)
        self.assertEqual(candidate[1].item(), parameters[1].item())

    def test_relation_coordinates_preserve_warm_start_and_reduce_dofs(self):
        from modules._s4_scenelm_relational import (
            compile_relation_coordinates,
        )

        base = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)
        base[0, :3, 3] = torch.tensor([0.0, 0.0, 0.0])
        base[1, :3, 3] = torch.tensor([1.0, 0.0, 1.0])
        base[2, :3, 3] = torch.tensor([1.2, 0.1, 1.5])
        coordinates = compile_relation_coordinates(
            base,
            torch.tensor([[2, 1]], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 3), dtype=torch.float64),
        )
        parameters = coordinates.zero_parameters()
        self.assertTrue(
            torch.allclose(coordinates.pose_matrices(parameters), base)
        )
        metadata = coordinates.metadata()
        self.assertEqual(metadata["parent_indices"], [-1, -1, 1])
        self.assertEqual(metadata["topological_order"], [0, 1, 2])

    def test_relation_coordinates_relax_incompatible_object_to_free(self):
        from modules._s4_scenelm_relational import (
            compile_relation_coordinates,
        )

        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        coordinates = compile_relation_coordinates(
            base,
            torch.tensor([[1, 0]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 3), dtype=torch.float64),
            free_object_indices=[1],
        )
        self.assertEqual(coordinates.blocks[1].mode, "free")
        self.assertEqual(coordinates.blocks[1].parent_index, -1)
        self.assertEqual(coordinates.relaxed_object_indices, (1,))
        self.assertLess(
            coordinates.parameter_count,
            coordinates.legacy_parameter_count,
        )
        self.assertEqual(coordinates.support_edge_count, 0)
        self.assertEqual(coordinates.leaf_object_indices, ())

    def test_relation_child_inherits_parent_yaw_and_translation(self):
        from modules._s4_scenelm_relational import (
            compile_relation_coordinates,
        )

        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[1, 0, 3] = 2.0
        coordinates = compile_relation_coordinates(
            base,
            torch.tensor([[1, 0]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, 3), dtype=torch.float64),
        )
        parameters = coordinates.zero_parameters()
        root = coordinates.blocks[0]
        parameters[root.parameter_start] = math.pi / 2
        parameters[root.parameter_start + 1] = 1.0
        yaw, translation = coordinates.decode(parameters)
        self.assertAlmostEqual(yaw[1].item(), math.pi / 2)
        self.assertTrue(
            torch.allclose(
                translation[1],
                torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64),
                atol=1e-9,
            )
        )

    def test_relation_plane_coordinates_remove_normal_motion(self):
        from modules._s4_scenelm_relational import (
            compile_relation_coordinates,
        )

        base = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        normal = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        coordinates = compile_relation_coordinates(
            base,
            torch.empty((0, 2), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            normal,
        )
        parameters = coordinates.zero_parameters()
        block = coordinates.blocks[0]
        parameters[block.parameter_start + 1 :] = torch.tensor([2.0, -3.0])
        _, translation = coordinates.decode(parameters)
        self.assertAlmostEqual(translation[0, 0].item(), 0.0, places=9)

    def test_relation_plane_coordinates_can_anchor_tangent_motion(self):
        from modules._s4_scenelm_relational import compile_relation_coordinates

        base = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        base[0, :3, 3] = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        coordinates = compile_relation_coordinates(
            base,
            torch.empty((0, 2), dtype=torch.long),
            torch.empty((0,), dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
            warm_start_anchored_plane_translation=True,
        )
        block = coordinates.blocks[0]
        self.assertEqual(block.mode, "plane_n_anchored")
        self.assertEqual(block.translation_dofs, 1)
        parameters = coordinates.zero_parameters()
        parameters[block.parameter_start + 1] = 0.4
        _, translation = coordinates.decode(parameters)
        self.assertTrue(
            torch.allclose(
                translation[0],
                torch.tensor([1.4, 2.0, 3.0], dtype=torch.float64),
                atol=1e-12,
            )
        )

    def test_plane_anchor_trust_removes_tangent_and_clamps_normal(self):
        base = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        pose = base.clone()
        pose[0, :3, 3] = torch.tensor([0.30, 0.40, -0.20], dtype=torch.float64)
        audit = self.ops.enforce_warm_start_plane_translation_trust_(
            pose,
            base,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
            normal_limit_m=0.02,
        )
        self.assertTrue(
            torch.allclose(
                pose[0, :3, 3],
                torch.tensor([0.02, 0.0, 0.0], dtype=torch.float64),
                atol=1e-12,
            )
        )
        self.assertAlmostEqual(audit["pre_max_tangent_m"], math.sqrt(0.20), places=12)
        self.assertAlmostEqual(audit["pre_max_normal_m"], 0.30, places=12)
        self.assertAlmostEqual(audit["post_max_normal_m"], 0.02, places=12)

    def test_relation_coordinates_reject_support_cycle(self):
        from modules._s4_scenelm_relational import (
            compile_relation_coordinates,
        )

        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        with self.assertRaisesRegex(ValueError, "cycle"):
            compile_relation_coordinates(
                base,
                torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
                torch.empty((0, 3), dtype=torch.float64),
            )

    def test_block_schur_matches_dense_solve(self):
        from modules._s4_scenelm_relational import (
            block_schur_complement_solve,
        )

        matrix = torch.tensor(
            [
                [5.0, 1.0, 0.5, 0.0],
                [1.0, 4.0, 0.0, 0.25],
                [0.5, 0.0, 3.0, 0.2],
                [0.0, 0.25, 0.2, 2.0],
            ],
            dtype=torch.float64,
        )
        rhs = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)
        actual, diagnostics = block_schur_complement_solve(
            matrix,
            rhs,
            torch.tensor([2, 3]),
            jitter=0.0,
        )
        expected = torch.linalg.solve(matrix, rhs)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-10))
        self.assertEqual(diagnostics["eliminated"], 2.0)

    def test_world_yaw_delta_rotates_linear_basis(self):
        base = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        yaw = torch.tensor([math.pi / 2], dtype=torch.float64)
        translation = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        result = self.ops.reproject_pose_matrices(base, yaw, translation)
        expected = torch.tensor(
            [
                [
                    [0.0, -1.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float64,
        )
        self.assertTrue(torch.allclose(result, expected, atol=1e-12))

    def test_gradients_reach_yaw_and_translation(self):
        base = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        base[0, 0, 0] = 2.0
        yaw, translation = self.ops.initialize_pose_variables(base)
        pose = self.ops.reproject_pose_matrices(base, yaw, translation)
        local_point = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float64)
        world_point = self.ops.transform_points(pose, local_point)
        loss = world_point[..., 1].sum() + world_point[..., 2].sum()
        loss.backward()
        self.assertIsNotNone(yaw.grad)
        self.assertIsNotNone(translation.grad)
        self.assertGreater(abs(yaw.grad.item()), 0.0)
        self.assertGreater(abs(translation.grad[0, 2].item()), 0.0)

    def test_local_box_corners_cover_minimum_and_maximum(self):
        minimum = torch.tensor([[-1.0, -2.0, -3.0]])
        maximum = torch.tensor([[1.0, 2.0, 3.0]])
        corners = self.ops.local_box_corners(minimum, maximum)
        self.assertEqual(tuple(corners.shape), (1, 8, 3))
        self.assertTrue(torch.equal(corners.amin(dim=1), minimum))
        self.assertTrue(torch.equal(corners.amax(dim=1), maximum))

    def test_oriented_penetration_is_zero_for_separated_boxes(self):
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[1, 0, 3] = 3.0
        bounds_min = torch.full((2, 3), -0.5, dtype=torch.float64)
        bounds_max = torch.full((2, 3), 0.5, dtype=torch.float64)
        corners = self.ops.local_box_corners(bounds_min, bounds_max)
        pairs = self.ops.pair_index_tensor([(0, 1)])
        loss, per_pair = self.ops.oriented_penetration_loss(base, corners, pairs)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(per_pair.item(), 0.0)

    def test_oriented_penetration_has_translation_gradient(self):
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[1, 0, 3] = 0.25
        bounds_min = torch.full((2, 3), -0.5, dtype=torch.float64)
        bounds_max = torch.full((2, 3), 0.5, dtype=torch.float64)
        corners = self.ops.local_box_corners(bounds_min, bounds_max)
        yaw, translation = self.ops.initialize_pose_variables(base)
        pose = self.ops.reproject_pose_matrices(base, yaw, translation)
        pairs = self.ops.pair_index_tensor([(0, 1)])
        loss, _ = self.ops.oriented_penetration_loss(pose, corners, pairs)
        loss.backward()
        self.assertGreater(loss.item(), 0.0)
        self.assertIsNotNone(translation.grad)
        self.assertGreater(torch.linalg.vector_norm(translation.grad).item(), 0.0)

    def test_variable_edge_collision_axes_reject_aabb_false_positive(self):
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[1, 0, 3] = 1.5
        base[1, 1, 3] = 1.5
        footprint = torch.tensor(
            [
                [-1.0, 0.0],
                [0.0, -1.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        lower = torch.cat(
            (footprint, torch.full((4, 1), -0.5, dtype=torch.float64)),
            dim=1,
        )
        upper = torch.cat(
            (footprint, torch.full((4, 1), 0.5, dtype=torch.float64)),
            dim=1,
        )
        corners = torch.stack(
            (torch.cat((lower, upper)), torch.cat((lower, upper)))
        )
        pairs = self.ops.pair_index_tensor([(0, 1)])
        legacy, _ = self.ops.oriented_penetration_loss(
            base, corners, pairs
        )
        exact, exact_pairs = self.ops.oriented_penetration_loss(
            base,
            corners,
            pairs,
            torch.tensor([4, 4], dtype=torch.long),
        )
        self.assertGreater(legacy.item(), 0.0)
        self.assertEqual(exact.item(), 0.0)
        self.assertEqual(exact_pairs.item(), 0.0)

    def test_collision_optimizer_reduces_penetration_without_moving_z(self):
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[1, 0, 3] = 0.25
        bounds_min = torch.full((2, 3), -0.5, dtype=torch.float64)
        bounds_max = torch.full((2, 3), 0.5, dtype=torch.float64)
        corners = self.ops.local_box_corners(bounds_min, bounds_max)
        pairs = self.ops.pair_index_tensor([(0, 1)])
        initial, _ = self.ops.oriented_penetration_loss(base, corners, pairs)
        optimized, history = self.ops.optimize_collision_stage(
            base,
            corners,
            pairs,
            iterations=40,
            learning_rate=0.05,
            warm_start_weight=0.0,
        )
        final, _ = self.ops.oriented_penetration_loss(optimized, corners, pairs)
        self.assertLess(final.item(), initial.item())
        self.assertTrue(torch.equal(optimized[:, 2, 3], base[:, 2, 3]))
        self.assertGreaterEqual(len(history), 2)

    def test_support_contact_loss_and_projection(self):
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        # Both boxes have half-height 0.5.  A child center at z=0.9 puts its
        # bottom 0.1 below the parent's top, exercising penetration recovery.
        base[0, 2, 3] = 0.9
        bounds_min = torch.full((2, 3), -0.5, dtype=torch.float64)
        bounds_max = torch.full((2, 3), 0.5, dtype=torch.float64)
        corners = self.ops.local_box_corners(bounds_min, bounds_max)
        support_pairs = self.ops.pair_index_tensor([(0, 1)])
        loss, gaps = self.ops.support_contact_loss(
            base, corners, support_pairs
        )
        self.assertAlmostEqual(gaps.item(), -0.1)
        self.assertGreater(loss.item(), 0.0)

        yaw, translation = self.ops.initialize_pose_variables(base)
        self.ops.project_support_contacts_(
            yaw,
            translation,
            base,
            corners,
            support_pairs,
            passes=2,
        )
        projected = self.ops.reproject_pose_matrices(base, yaw, translation)
        projected_loss, projected_gaps = self.ops.support_contact_loss(
            projected, corners, support_pairs
        )
        self.assertLessEqual(projected_loss.item(), 1e-12)
        self.assertLessEqual(abs(projected_gaps.item()), 1e-12)

    def test_fixed_floor_contact_projection(self):
        base = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        base[0, 2, 3] = 0.75
        bounds_min = torch.full((1, 3), -0.5, dtype=torch.float64)
        bounds_max = torch.full((1, 3), 0.5, dtype=torch.float64)
        corners = self.ops.local_box_corners(bounds_min, bounds_max)
        no_pairs = self.ops.pair_index_tensor([])
        fixed_indices = torch.tensor([0], dtype=torch.long)
        fixed_heights = torch.tensor([0.0], dtype=torch.float64)
        yaw, translation = self.ops.initialize_pose_variables(base)
        self.ops.project_support_contacts_(
            yaw,
            translation,
            base,
            corners,
            no_pairs,
            fixed_indices,
            fixed_heights,
        )
        projected = self.ops.reproject_pose_matrices(base, yaw, translation)
        _, gaps = self.ops.support_contact_loss(
            projected,
            corners,
            no_pairs,
            fixed_indices,
            fixed_heights,
        )
        self.assertLessEqual(abs(gaps.item()), 1e-12)

    def test_wall_plane_contact_projection(self):
        base = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        base[0, 0, 3] = 1.5
        bounds_min = torch.full((1, 3), -0.5, dtype=torch.float64)
        bounds_max = torch.full((1, 3), 0.5, dtype=torch.float64)
        corners = self.ops.local_box_corners(bounds_min, bounds_max)
        object_indices = torch.tensor([0], dtype=torch.long)
        plane_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        plane_normals = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        orientation_mask = torch.tensor([True])

        _, _, gaps, _ = self.ops.fixed_plane_loss(
            base,
            corners,
            object_indices,
            plane_points,
            plane_normals,
            orientation_mask,
        )
        self.assertAlmostEqual(gaps.item(), 1.0)

        yaw, translation = self.ops.initialize_pose_variables(base)
        self.ops.project_fixed_planes_(
            yaw,
            translation,
            base,
            corners,
            object_indices,
            plane_points,
            plane_normals,
        )
        projected = self.ops.reproject_pose_matrices(base, yaw, translation)
        _, _, projected_gaps, _ = self.ops.fixed_plane_loss(
            projected,
            corners,
            object_indices,
            plane_points,
            plane_normals,
            orientation_mask,
        )
        self.assertLessEqual(abs(projected_gaps.item()), 1e-12)
        self.assertAlmostEqual(projected[0, 0, 3].item(), 0.5)

    def test_ceiling_plane_projects_object_top(self):
        base = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        base[0, 2, 3] = 1.5
        bounds_min = torch.full((1, 3), -0.5, dtype=torch.float64)
        bounds_max = torch.full((1, 3), 0.5, dtype=torch.float64)
        corners = self.ops.local_box_corners(bounds_min, bounds_max)
        object_indices = torch.tensor([0], dtype=torch.long)
        plane_points = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float64)
        plane_normals = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float64)
        yaw, translation = self.ops.initialize_pose_variables(base)
        self.ops.project_fixed_planes_(
            yaw,
            translation,
            base,
            corners,
            object_indices,
            plane_points,
            plane_normals,
        )
        projected = self.ops.reproject_pose_matrices(base, yaw, translation)
        _, _, gaps, _ = self.ops.fixed_plane_loss(
            projected,
            corners,
            object_indices,
            plane_points,
            plane_normals,
        )
        self.assertLessEqual(abs(gaps.item()), 1e-12)
        self.assertAlmostEqual(projected[0, 2, 3].item(), 2.5)

    def test_wall_orientation_loss_prefers_parallel_box_axis(self):
        base = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        object_indices = torch.tensor([0], dtype=torch.long)
        plane_points = torch.zeros((1, 3), dtype=torch.float64)
        plane_normals = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        orientation_mask = torch.tensor([True])
        bounds_min = torch.full((1, 3), -0.5, dtype=torch.float64)
        bounds_max = torch.full((1, 3), 0.5, dtype=torch.float64)
        corners = self.ops.local_box_corners(bounds_min, bounds_max)

        aligned = self.ops.reproject_pose_matrices(
            base,
            torch.tensor([0.0], dtype=torch.float64),
            base[:, :3, 3],
        )
        rotated = self.ops.reproject_pose_matrices(
            base,
            torch.tensor([math.pi / 6], dtype=torch.float64),
            base[:, :3, 3],
        )
        _, aligned_orientation, _, _ = self.ops.fixed_plane_loss(
            aligned,
            corners,
            object_indices,
            plane_points,
            plane_normals,
            orientation_mask,
        )
        _, rotated_orientation, _, _ = self.ops.fixed_plane_loss(
            rotated,
            corners,
            object_indices,
            plane_points,
            plane_normals,
            orientation_mask,
        )
        self.assertEqual(aligned_orientation.item(), 0.0)
        self.assertGreater(rotated_orientation.item(), 0.0)

    def test_sceneproof_thin_wall_asset_uses_attachment_facing_axis(self):
        base = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        object_indices = torch.tensor([0], dtype=torch.long)
        plane_points = torch.zeros((1, 3), dtype=torch.float64)
        plane_normals = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        orientation_mask = torch.tensor([True])
        bounds_min = torch.tensor([[-0.05, -1.0, -1.5]], dtype=torch.float64)
        bounds_max = torch.tensor([[0.05, 1.0, 1.5]], dtype=torch.float64)
        corners = self.ops.local_box_corners(bounds_min, bounds_max)

        with mock.patch.dict(
            os.environ,
            {"IMAGINARIUM_SCENEPROOF_THIN_AXIS_ATTACH_RATIO": "0.25"},
        ):
            _, correct, _, _ = self.ops.fixed_plane_loss(
                base,
                corners,
                object_indices,
                plane_points,
                plane_normals,
                orientation_mask,
            )
            edge_on_pose = base.clone()
            edge_on_pose[0, :3, :3] = torch.tensor(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            )
            _, edge_on, _, _ = self.ops.fixed_plane_loss(
                edge_on_pose,
                corners,
                object_indices,
                plane_points,
                plane_normals,
                orientation_mask,
            )

        self.assertAlmostEqual(correct.item(), 0.0)
        self.assertGreater(edge_on.item(), 0.9)

    def test_plane_optimizer_preserves_floor_and_wall_contacts(self):
        base = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        base[0, 0, 3] = 1.0
        base[0, 2, 3] = 0.5
        bounds_min = torch.full((1, 3), -0.5, dtype=torch.float64)
        bounds_max = torch.full((1, 3), 0.5, dtype=torch.float64)
        corners = self.ops.local_box_corners(bounds_min, bounds_max)
        empty_pairs = self.ops.pair_index_tensor([])
        fixed_indices = torch.tensor([0], dtype=torch.long)
        fixed_heights = torch.tensor([0.0], dtype=torch.float64)
        plane_indices = torch.tensor([0], dtype=torch.long)
        plane_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        plane_normals = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        orientation_mask = torch.tensor([True])
        optimized, history = self.ops.optimize_plane_stage(
            base,
            corners,
            empty_pairs,
            empty_pairs,
            fixed_indices,
            fixed_heights,
            plane_indices,
            plane_points,
            plane_normals,
            orientation_mask,
            iterations=5,
        )
        _, support_gaps = self.ops.support_contact_loss(
            optimized,
            corners,
            empty_pairs,
            fixed_indices,
            fixed_heights,
        )
        _, _, plane_gaps, _ = self.ops.fixed_plane_loss(
            optimized,
            corners,
            plane_indices,
            plane_points,
            plane_normals,
            orientation_mask,
        )
        self.assertLessEqual(abs(support_gaps.item()), 1e-12)
        self.assertLessEqual(abs(plane_gaps.item()), 1e-12)
        self.assertLessEqual(history[-1]["projected_max_contact_gap"], 1e-12)
        self.assertLessEqual(history[-1]["projected_max_plane_gap"], 1e-12)

    def test_distance_interval_is_directed_and_matches_layoutvlm_hinge(self):
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[1, 0, 3] = 2.0
        pairs = self.ops.pair_index_tensor([(0, 1)])
        minimum = torch.tensor([1.0], dtype=torch.float64)
        maximum = torch.tensor([3.0], dtype=torch.float64)
        loss, distances, penalties = self.ops.distance_interval_loss(
            base, pairs, minimum, maximum
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertAlmostEqual(distances.item(), 2.0)
        self.assertEqual(penalties.item(), 0.0)

        yaw, translation = self.ops.initialize_pose_variables(base)
        pose = self.ops.reproject_pose_matrices(base, yaw, translation)
        # Keep the violation below LayoutVLM's per-constraint clamp(max=1).
        # At distance 2.0 and max 1.8, the squared hinge is 0.76 and must
        # retain a non-zero directed gradient.  A max of 1.0 would saturate
        # the official clamp and correctly produce zero gradient.
        tight_maximum = torch.tensor([1.8], dtype=torch.float64)
        loss, _, _ = self.ops.distance_interval_loss(
            pose, pairs, torch.zeros_like(minimum), tight_maximum
        )
        loss.backward()
        self.assertGreater(abs(translation.grad[0, 0].item()), 0.0)
        self.assertEqual(translation.grad[1, 0].item(), 0.0)

    def test_align_with_uses_imaginarium_minus_y_front_and_offset(self):
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        yaw = torch.tensor([math.pi / 2.0, 0.0], dtype=torch.float64)
        pose = self.ops.reproject_pose_matrices(
            base, yaw, base[:, :3, 3]
        )
        pairs = self.ops.pair_index_tensor([(0, 1)])
        loss, errors = self.ops.align_with_loss(
            pose,
            pairs,
            torch.tensor([math.pi / 2.0], dtype=torch.float64),
        )
        self.assertLessEqual(loss.item(), 1e-12)
        self.assertLessEqual(errors.item(), 1e-12)

    def test_point_towards_is_zero_when_front_ray_hits_target_obb(self):
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[1, 0, 3] = 2.0
        yaw = torch.tensor([math.pi / 2.0, 0.0], dtype=torch.float64)
        pose = self.ops.reproject_pose_matrices(
            base, yaw, base[:, :3, 3]
        )
        bounds_min = torch.full((2, 3), -0.5, dtype=torch.float64)
        bounds_max = torch.full((2, 3), 0.5, dtype=torch.float64)
        corners = self.ops.local_box_corners(bounds_min, bounds_max)
        pairs = self.ops.pair_index_tensor([(0, 1)])
        loss, errors = self.ops.point_towards_loss(
            pose,
            corners,
            pairs,
            torch.zeros(1, dtype=torch.float64),
        )
        self.assertLessEqual(loss.item(), 1e-12)
        self.assertLessEqual(errors.item(), 1e-12)

    def test_support_footprint_projection_moves_child_center_inside_parent(self):
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[0, 0, 3] = 2.0
        base[0, 2, 3] = 1.0
        bounds_min = torch.full((2, 3), -0.5, dtype=torch.float64)
        bounds_max = torch.full((2, 3), 0.5, dtype=torch.float64)
        corners = self.ops.local_box_corners(bounds_min, bounds_max)
        pairs = self.ops.pair_index_tensor([(0, 1)])
        initial, _ = self.ops.support_planar_containment_loss(
            base, corners, pairs
        )
        self.assertGreater(initial.item(), 0.0)

        yaw, translation = self.ops.initialize_pose_variables(base)
        self.ops.project_support_footprints_(
            yaw, translation, base, corners, pairs
        )
        projected = self.ops.reproject_pose_matrices(
            base, yaw, translation
        )
        final, errors = self.ops.support_planar_containment_loss(
            projected, corners, pairs
        )
        self.assertLessEqual(final.item(), 1e-12)
        self.assertLessEqual(errors.item(), 1e-12)
        # Equal-size child and parent require coincident planar centres for
        # complete-footprint containment.
        self.assertAlmostEqual(projected[0, 0, 3].item(), 0.0)

    def test_containment_gate_rejects_large_warm_start_error(self):
        base = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)
        base[0, 0, 3] = 0.25
        base[1, 0, 3] = 1.0
        bounds_min = torch.full((3, 3), -0.5, dtype=torch.float64)
        bounds_max = torch.full((3, 3), 0.5, dtype=torch.float64)
        corners = self.ops.local_box_corners(bounds_min, bounds_max)
        pairs = self.ops.pair_index_tensor([(0, 2), (1, 2)])

        accepted, rejected, errors = (
            self.ops.gate_support_containment_pairs(
                base,
                corners,
                pairs,
                maximum_initial_error=0.5,
            )
        )

        self.assertEqual(accepted.tolist(), [[0, 2]])
        self.assertEqual(rejected.tolist(), [[1, 2]])
        self.assertAlmostEqual(errors[0].item(), 0.25)
        self.assertAlmostEqual(errors[1].item(), 1.0)

    def test_containment_gate_rejects_geometrically_impossible_pair(self):
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        minimum = torch.tensor(
            [[-1.0, -1.0, -0.1], [-0.5, -0.5, -0.1]],
            dtype=torch.float64,
        )
        maximum = torch.tensor(
            [[1.0, 1.0, 0.1], [0.5, 0.5, 0.1]],
            dtype=torch.float64,
        )
        corners = self.ops.local_box_corners(minimum, maximum)
        # Object 0 is the child and is twice as wide as parent object 1.
        pairs = self.ops.pair_index_tensor([(0, 1)])
        accepted, rejected, errors = (
            self.ops.gate_support_containment_pairs(
                base, corners, pairs, maximum_initial_error=10.0
            )
        )
        self.assertEqual(accepted.shape[0], 0)
        self.assertEqual(rejected.tolist(), [[0, 1]])
        self.assertGreater(errors.item(), 0.0)

    def test_containment_gate_does_not_change_vertical_contact_pairs(self):
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[0, 0, 3] = 2.0
        corners = self.ops.local_box_corners(
            torch.full((2, 3), -0.5, dtype=torch.float64),
            torch.full((2, 3), 0.5, dtype=torch.float64),
        )
        support_pairs = self.ops.pair_index_tensor([(0, 1)])

        _, rejected, _ = self.ops.gate_support_containment_pairs(
            base,
            corners,
            support_pairs,
            maximum_initial_error=0.5,
        )

        self.assertEqual(rejected.tolist(), [[0, 1]])
        self.assertEqual(support_pairs.tolist(), [[0, 1]])

    def test_convex_room_halfspaces_accept_clockwise_polygon(self):
        clockwise = torch.tensor(
            [
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, 1.0],
                [1.0, -1.0],
            ],
            dtype=torch.float64,
        )
        points, normals = self.ops.convex_polygon_halfspaces(clockwise)
        center_signed = torch.einsum(
            "bd,bd->b",
            -points,
            normals,
        )
        self.assertTrue(torch.all(center_signed > 0))

    def test_room_boundary_loss_and_projection_use_full_obb_footprint(self):
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[0, 0, 3] = 0.75
        corners = self.ops.local_box_corners(
            torch.full((2, 3), -0.5, dtype=torch.float64),
            torch.full((2, 3), 0.5, dtype=torch.float64),
        )
        polygon = torch.tensor(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            dtype=torch.float64,
        )
        boundary_points, boundary_normals = (
            self.ops.convex_polygon_halfspaces(polygon)
        )
        object_indices = torch.tensor([0, 1], dtype=torch.long)
        initial, errors = self.ops.room_boundary_loss(
            base,
            corners,
            object_indices,
            boundary_points,
            boundary_normals,
        )
        self.assertGreater(initial.item(), 0.0)
        self.assertAlmostEqual(errors[0].item(), 0.25)
        self.assertAlmostEqual(errors[1].item(), 0.0)

        yaw, translation = self.ops.initialize_pose_variables(base)
        self.ops.project_room_boundary_(
            yaw,
            translation,
            base,
            corners,
            object_indices,
            boundary_points,
            boundary_normals,
        )
        projected = self.ops.reproject_pose_matrices(
            base,
            yaw,
            translation,
        )
        final, errors = self.ops.room_boundary_loss(
            projected,
            corners,
            object_indices,
            boundary_points,
            boundary_normals,
        )
        self.assertLessEqual(final.item(), 1e-12)
        self.assertLessEqual(errors.amax().item(), 1e-12)
        self.assertAlmostEqual(projected[0, 0, 3].item(), 0.5)

    def test_variable_edge_support_polygon_projects_complete_child(self):
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[1, 0, 3] = 1.2
        parent = torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [-0.5, -0.8, 0.0],
                [0.5, -0.8, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.8, 0.0],
                [-0.5, 0.8, 0.0],
                [0.0, 0.0, -0.1],
                [0.0, 0.0, 0.1],
            ],
            dtype=torch.float64,
        )
        child = self.ops.local_box_corners(
            torch.tensor([[-0.2, -0.2, -0.2]], dtype=torch.float64),
            torch.tensor([[0.2, 0.2, 0.2]], dtype=torch.float64),
        )[0]
        corners = torch.stack((parent, child))
        pairs = self.ops.pair_index_tensor([(1, 0)])
        hull_sizes = torch.tensor([6, 4], dtype=torch.long)
        yaw, translation = self.ops.initialize_pose_variables(base)

        initial, _ = self.ops.support_planar_containment_loss(
            base, corners, pairs, hull_sizes
        )
        self.assertGreater(initial.item(), 0.0)
        self.ops.project_support_footprints_(
            yaw,
            translation,
            base,
            corners,
            pairs,
            passes=4,
            footprint_hull_sizes=hull_sizes,
        )
        projected = self.ops.reproject_pose_matrices(
            base, yaw, translation
        )
        final, errors = self.ops.support_planar_containment_loss(
            projected, corners, pairs, hull_sizes
        )
        self.assertLess(final.item(), 1e-12)
        self.assertLess(errors.max().item(), 1e-12)
        # Projection finds a feasible translated footprint; it is not a
        # centre-seeking operator.  For this hexagon the closest feasible
        # solution remains off-centre, which is both valid and desirable.
        projected_x = projected[1, 0, 3]
        self.assertTrue(torch.isfinite(projected_x).item())
        self.assertGreater(abs(projected_x.item() - 1.2), 1e-6)
        self.assertLess(
            torch.linalg.vector_norm(translation[1, :2]).item(), 1.0
        )

    def test_minimum_norm_halfspace_translation_uses_two_active_edges(self):
        normals = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
        )
        lower_bounds = torch.tensor([1.0, 1.0], dtype=torch.float64)
        correction = self.ops.minimum_norm_halfspace_translation(
            normals, lower_bounds
        )
        torch.testing.assert_close(
            correction,
            torch.tensor([1.0, 1.0], dtype=torch.float64),
        )

    def test_minimum_norm_halfspace_translation_rejects_infeasible_set(self):
        normals = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64
        )
        lower_bounds = torch.tensor([1.0, 0.0], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "no feasible"):
            self.ops.minimum_norm_halfspace_translation(
                normals, lower_bounds
            )

    def test_support_projection_can_abstain_and_restore_warm_start_planar(self):
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[0, 0, 3] = 0.25
        minimum = torch.tensor(
            [[-1.0, -1.0, -0.1], [-0.5, -0.5, -0.1]],
            dtype=torch.float64,
        )
        maximum = torch.tensor(
            [[1.0, 1.0, 0.1], [0.5, 0.5, 0.1]],
            dtype=torch.float64,
        )
        corners = self.ops.local_box_corners(minimum, maximum)
        pairs = self.ops.pair_index_tensor([(0, 1)])
        yaw, translation = self.ops.initialize_pose_variables(base)
        with torch.no_grad():
            yaw[0] = 0.4
            translation[0, 0] = 2.0
            translation[0, 1] = -1.0
            translation[0, 2] = 0.75

        audit = self.ops.project_support_footprints_(
            yaw,
            translation,
            base,
            corners,
            pairs,
            infeasible_policy="restore_warm_start_planar",
        )

        self.assertEqual(len(audit), 1)
        self.assertEqual(
            audit[0]["status"], "abstained_restore_warm_start_planar"
        )
        self.assertAlmostEqual(yaw[0].item(), 0.0)
        torch.testing.assert_close(translation[0, :2], base[0, :2, 3])
        self.assertAlmostEqual(translation[0, 2].item(), 0.75)

    def test_scenelm_convergence_requires_residual_safe_scene(self):
        unsafe = torch.tensor([True, False, True], dtype=torch.bool)
        safe = torch.ones(3, dtype=torch.bool)
        self.assertFalse(
            self.ops.certified_lm_convergence(True, unsafe)
        )
        self.assertFalse(
            self.ops.certified_lm_convergence(False, safe)
        )
        self.assertTrue(
            self.ops.certified_lm_convergence(True, safe)
        )

    def test_collision_release_selects_only_unsafe_pair_endpoints(self):
        residuals = torch.tensor(
            [0.0, 1.0e-5, 0.2, 1.0e-4], dtype=torch.float64
        )
        mask = self.ops.collision_connected_release_mask(
            residuals, threshold=1.0e-4
        )
        self.assertTrue(
            torch.equal(
                mask,
                torch.tensor([False, False, True, False]),
            )
        )

    def test_collision_release_rebase_preserves_incumbent_pose(self):
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[1, 2, 3] = 1.0
        support = self.ops.pair_index_tensor([(1, 0)])
        empty_indices = torch.empty((0,), dtype=torch.long)
        empty_normals = torch.empty((0, 3), dtype=torch.float64)
        chart = self.ops.compile_relation_coordinates(
            base,
            support,
            empty_indices,
            empty_indices,
            empty_normals,
        )
        parameters = chart.zero_parameters()
        parameters[chart.blocks[0].parameter_slice.start] = 0.25
        incumbent = chart.pose_matrices(parameters).detach()
        released = self.ops.compile_relation_coordinates(
            incumbent,
            support,
            empty_indices,
            empty_indices,
            empty_normals,
            free_object_indices=[1],
        )
        torch.testing.assert_close(
            released.pose_matrices(released.zero_parameters()),
            incumbent,
        )

    def test_collision_witness_freezes_a_valid_separating_disjunction(self):
        pose = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        pose[1, 0, 3] = 0.5
        ordered_box = torch.tensor(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
            dtype=torch.float64,
        )
        corners = ordered_box.unsqueeze(0).repeat(2, 1, 1)
        pairs = self.ops.pair_index_tensor([(0, 1)])
        hull_sizes = torch.full((2,), 4, dtype=torch.long)
        axes = self.ops.collision_separation_witness_axes(
            pose, corners, pairs, hull_sizes
        )
        before = self.ops.collision_witness_residuals(
            pose, corners, pairs, axes, hull_sizes
        )
        separated = pose.clone()
        separated[0, :2, 3] += axes[0] * 0.6
        after = self.ops.collision_witness_residuals(
            separated, corners, pairs, axes, hull_sizes
        )
        self.assertGreater(before.item(), 0.49)
        self.assertLessEqual(after.item(), 1.0e-12)

    def test_full_optimizer_restores_an_evaluated_best_state(self):
        base = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        base[0, 0, 3] = 0.75
        corners = self.ops.local_box_corners(
            torch.full((1, 3), -0.5, dtype=torch.float64),
            torch.full((1, 3), 0.5, dtype=torch.float64),
        )
        empty_pairs = self.ops.pair_index_tensor([])
        empty_indices = torch.empty((0,), dtype=torch.long)
        empty_values = torch.empty((0,), dtype=torch.float64)
        empty_points3 = torch.empty((0, 3), dtype=torch.float64)
        empty_mask = torch.empty((0,), dtype=torch.bool)
        boundary_points, boundary_normals = (
            self.ops.convex_polygon_halfspaces(
                torch.tensor(
                    [
                        [-1.0, -1.0],
                        [1.0, -1.0],
                        [1.0, 1.0],
                        [-1.0, 1.0],
                    ],
                    dtype=torch.float64,
                )
            )
        )

        final, history = self.ops.optimize_semantic_stage(
            base,
            corners,
            empty_pairs,
            empty_pairs,
            empty_pairs,
            empty_indices,
            empty_values,
            empty_indices,
            empty_points3,
            empty_points3,
            empty_mask,
            empty_pairs,
            empty_values,
            empty_pairs,
            empty_values,
            empty_values,
            empty_pairs,
            empty_values,
            boundary_object_indices=torch.tensor([0], dtype=torch.long),
            boundary_points=boundary_points,
            boundary_normals=boundary_normals,
            iterations=4,
            restore_best_state=True,
        )

        self.assertIn("best_iteration", history[-1])
        self.assertIn("best_total", history[-1])
        self.assertGreaterEqual(history[-1]["best_iteration"], 2.0)
        _, final_errors = self.ops.room_boundary_loss(
            final,
            corners,
            torch.tensor([0], dtype=torch.long),
            boundary_points,
            boundary_normals,
        )
        self.assertLessEqual(final_errors.amax().item(), 1e-12)

    def test_full_optimizer_can_freeze_yaw(self):
        angle = torch.tensor(0.4, dtype=torch.float64)
        base = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        base[0, 0, 0] = torch.cos(angle)
        base[0, 0, 1] = -torch.sin(angle)
        base[0, 1, 0] = torch.sin(angle)
        base[0, 1, 1] = torch.cos(angle)
        corners = self.ops.local_box_corners(
            torch.full((1, 3), -0.5, dtype=torch.float64),
            torch.full((1, 3), 0.5, dtype=torch.float64),
        )
        empty_pairs = self.ops.pair_index_tensor([])
        empty_indices = torch.empty((0,), dtype=torch.long)
        empty_values = torch.empty((0,), dtype=torch.float64)
        empty_points3 = torch.empty((0, 3), dtype=torch.float64)
        empty_mask = torch.empty((0,), dtype=torch.bool)

        final, history = self.ops.optimize_semantic_stage(
            base,
            corners,
            empty_pairs,
            empty_pairs,
            empty_pairs,
            empty_indices,
            empty_values,
            empty_indices,
            empty_points3,
            empty_points3,
            empty_mask,
            empty_pairs,
            empty_values,
            empty_pairs,
            empty_values,
            empty_values,
            empty_pairs,
            empty_values,
            iterations=2,
            optimize_yaw=False,
        )

        self.assertTrue(torch.equal(final[:, :3, :3], base[:, :3, :3]))
        self.assertEqual(history[-1]["yaw_optimized"], 0.0)

    def test_semantic_weight_scales_total_energy(self):
        # Two objects separated by 2.0 along x while a distance constraint caps
        # them at 1.5.  The semantic term is therefore a hinge violation, and
        # its contribution to the total energy must scale with semantic_weight.
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[1, 0, 3] = 2.0
        corners = self.ops.local_box_corners(
            torch.full((2, 3), -0.5, dtype=torch.float64),
            torch.full((2, 3), 0.5, dtype=torch.float64),
        )
        empty_pairs = self.ops.pair_index_tensor([])
        empty_indices = torch.empty((0,), dtype=torch.long)
        empty_values = torch.empty((0,), dtype=torch.float64)
        empty_points3 = torch.empty((0, 3), dtype=torch.float64)
        empty_mask = torch.empty((0,), dtype=torch.bool)
        distance_pairs = self.ops.pair_index_tensor([(0, 1)])
        distance_min = torch.tensor([0.0], dtype=torch.float64)
        distance_max = torch.tensor([1.5], dtype=torch.float64)

        def total_at(weight):
            _, history = self.ops.optimize_semantic_stage(
                base,
                corners,
                empty_pairs,
                empty_pairs,
                empty_pairs,
                empty_indices,
                empty_values,
                empty_indices,
                empty_points3,
                empty_points3,
                empty_mask,
                empty_pairs,
                empty_values,
                distance_pairs,
                distance_min,
                distance_max,
                empty_pairs,
                empty_values,
                iterations=2,
                solver="adam",
                semantic_weight=weight,
            )
            return history[-1]["total"]

        self.assertGreater(total_at(1.0), total_at(0.0))

    def test_scenelm_full_optimizer_persists_solver_certificate(self):
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[0, 2, 3] = 1.5
        corners = self.ops.local_box_corners(
            torch.full((2, 3), -0.5, dtype=torch.float64),
            torch.full((2, 3), 0.5, dtype=torch.float64),
        )
        empty_pairs = self.ops.pair_index_tensor([])
        support_pairs = self.ops.pair_index_tensor([(0, 1)])
        empty_indices = torch.empty((0,), dtype=torch.long)
        empty_values = torch.empty((0,), dtype=torch.float64)
        empty_points3 = torch.empty((0, 3), dtype=torch.float64)
        empty_mask = torch.empty((0,), dtype=torch.bool)

        final, history = self.ops.optimize_semantic_stage(
            base,
            corners,
            empty_pairs,
            support_pairs,
            empty_pairs,
            empty_indices,
            empty_values,
            empty_indices,
            empty_points3,
            empty_points3,
            empty_mask,
            empty_pairs,
            empty_values,
            empty_pairs,
            empty_values,
            empty_values,
            empty_pairs,
            empty_values,
            iterations=3,
            solver="scenelm",
            lm_pcg_iterations=4,
            warm_start_weight=0.0,
        )

        _, gaps = self.ops.support_contact_loss(
            final, corners, support_pairs
        )
        self.assertLess(gaps.abs().max().item(), 1e-9)
        self.assertEqual(history[-1]["solver"], "scenelm")
        self.assertIn("lm_final_residual_energy", history[-1])
        self.assertGreaterEqual(history[-1]["lm_accepted_steps"], 0.0)

    def test_v5_scenelm_compiles_relations_and_certifies_solution(self):
        base = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)
        base[0, 0, 3] = -1.0
        base[1, 0, 3] = 1.0
        base[2, :3, 3] = torch.tensor([1.0, 0.0, 1.0])
        corners = self.ops.local_box_corners(
            torch.full((3, 3), -0.5, dtype=torch.float64),
            torch.full((3, 3), 0.5, dtype=torch.float64),
        )
        empty_pairs = self.ops.pair_index_tensor([])
        support_pairs = self.ops.pair_index_tensor([(2, 1)])
        empty_indices = torch.empty((0,), dtype=torch.long)
        empty_values = torch.empty((0,), dtype=torch.float64)
        empty_points3 = torch.empty((0, 3), dtype=torch.float64)
        empty_mask = torch.empty((0,), dtype=torch.bool)

        final, history = self.ops.optimize_semantic_stage(
            base,
            corners,
            self.ops.pair_index_tensor([(0, 1)]),
            support_pairs,
            empty_pairs,
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([-0.5, -0.5], dtype=torch.float64),
            empty_indices,
            empty_points3,
            empty_points3,
            empty_mask,
            empty_pairs,
            empty_values,
            empty_pairs,
            empty_values,
            empty_values,
            empty_pairs,
            empty_values,
            iterations=4,
            solver="v5_scenelm",
            lm_pcg_iterations=4,
            lm_patience=2,
            warm_start_weight=0.01,
        )

        self.assertEqual(final.shape, base.shape)
        record = history[-1]
        self.assertEqual(record["solver"], "v5_scenelm")
        self.assertIn("relation_coordinates", record)
        self.assertLess(
            record["relation_coordinates"]["parameters"],
            record["relation_coordinates"]["legacy_parameters"],
        )
        self.assertIn("certificate_stationarity_inf", record)
        self.assertIn("certificate_primal_max", record)
        self.assertIn("post_projection_certified_support_pairs", record)
        self.assertGreaterEqual(record["relation_active_reduction"], 0.0)
        self.assertIn("relation_release_count", record)
        self.assertIn("relation_released_object_indices", record)
        self.assertIn("relation_release_iterations", record)
        self.assertIn("collision_witness_count", record)
        self.assertIn("collision_witness_weight", record)

    def test_active_set_residuals_wake_both_pair_endpoints(self):
        pairs = self.ops.pair_index_tensor([(0, 1)])
        empty_pairs = self.ops.pair_index_tensor([])
        empty_indices = torch.empty((0,), dtype=torch.long)
        empty_values = torch.empty((0,), dtype=torch.float64)
        residuals = self.ops.active_set_object_residuals(
            3,
            collision_pairs=pairs,
            collision_values=torch.tensor([0.2], dtype=torch.float64),
            support_pairs=pairs,
            contact_gaps=torch.tensor([0.03, 0.01], dtype=torch.float64),
            fixed_support_indices=torch.tensor([2], dtype=torch.long),
            plane_object_indices=empty_indices,
            plane_gaps=empty_values,
            plane_alignment_errors=empty_values,
            containment_pairs=empty_pairs,
            containment_errors=empty_values,
            distance_pairs=empty_pairs,
            distance_penalties=empty_values,
            align_pairs=empty_pairs,
            align_errors=empty_values,
            point_pairs=empty_pairs,
            point_errors=empty_values,
            boundary_object_indices=empty_indices,
            boundary_errors=empty_values,
            depth_observation_indices=empty_indices,
            depth_centre_errors=empty_values,
            depth_size_errors=empty_values,
            depth_relative_errors=empty_values,
        )
        self.assertTrue(
            torch.equal(
                residuals["collision"],
                torch.tensor([0.2, 0.2, 0.0], dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.equal(
                residuals["contact"],
                torch.tensor([0.03, 0.03, 0.01], dtype=torch.float64),
            )
        )

    def test_active_set_clears_frozen_adam_momentum(self):
        parameter = torch.ones((2, 3), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([parameter], lr=0.1)
        parameter.sum().backward()
        optimizer.step()
        state = optimizer.state[parameter]
        self.assertGreater(state["exp_avg"].abs().sum().item(), 0.0)
        self.ops._clear_adam_rows_(
            optimizer,
            [parameter],
            torch.tensor([True, False]),
        )
        self.assertEqual(state["exp_avg"][0].abs().sum().item(), 0.0)
        self.assertEqual(state["exp_avg_sq"][0].abs().sum().item(), 0.0)
        self.assertGreater(state["exp_avg"][1].abs().sum().item(), 0.0)

    def test_active_factor_mean_keeps_dense_denominator(self):
        residuals = torch.tensor(
            [1.0, 3.0], dtype=torch.float64, requires_grad=True
        )
        active_mean = residuals.square().mean()
        routed = self.ops._rescale_active_mean(
            active_mean,
            active_mass=2,
            full_mass=4,
        )
        expected = residuals.square().sum() / 4.0
        self.assertTrue(torch.equal(routed, expected))
        routed.backward()
        self.assertTrue(
            torch.equal(
                residuals.grad,
                torch.tensor([0.5, 1.5], dtype=torch.float64),
            )
        )

    def test_active_projection_pairs_require_active_child(self):
        pairs = self.ops.pair_index_tensor([(0, 1), (1, 2), (2, 0)])
        active = torch.tensor([False, True, False])
        mask = self.ops._active_child_pair_mask(pairs, active)
        self.assertTrue(
            torch.equal(mask, torch.tensor([False, True, False]))
        )

    def test_active_set_freezes_converged_object_at_first_checkpoint(self):
        base = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        corners = self.ops.local_box_corners(
            torch.full((1, 3), -0.5, dtype=torch.float64),
            torch.full((1, 3), 0.5, dtype=torch.float64),
        )
        empty_pairs = self.ops.pair_index_tensor([])
        empty_indices = torch.empty((0,), dtype=torch.long)
        empty_values = torch.empty((0,), dtype=torch.float64)
        empty_points3 = torch.empty((0, 3), dtype=torch.float64)
        empty_mask = torch.empty((0,), dtype=torch.bool)
        final, history = self.ops.optimize_semantic_stage(
            base,
            corners,
            empty_pairs,
            empty_pairs,
            empty_pairs,
            empty_indices,
            empty_values,
            empty_indices,
            empty_points3,
            empty_points3,
            empty_mask,
            empty_pairs,
            empty_values,
            empty_pairs,
            empty_values,
            empty_values,
            empty_pairs,
            empty_values,
            iterations=5,
            active_set_router=True,
            active_set_checkpoints=(2, 4),
        )
        self.assertTrue(torch.equal(final, base))
        self.assertEqual(history[-1]["router_budget_30"], 1.0)
        self.assertEqual(history[-1]["router_budget_100"], 0.0)
        self.assertEqual(history[-1]["router_budget_full"], 0.0)
        self.assertEqual(history[-1]["router_active_step_total"], 2.0)
        self.assertAlmostEqual(
            history[-1]["router_iteration_reduction"], 0.6
        )

    def test_collision_candidate_degree_does_not_permanently_protect(self):
        base = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
        base[1, 0, 3] = 3.0
        corners = self.ops.local_box_corners(
            torch.full((2, 3), -0.5, dtype=torch.float64),
            torch.full((2, 3), 0.5, dtype=torch.float64),
        )
        collision_pairs = self.ops.pair_index_tensor([(0, 1)])
        empty_pairs = self.ops.pair_index_tensor([])
        empty_indices = torch.empty((0,), dtype=torch.long)
        empty_values = torch.empty((0,), dtype=torch.float64)
        empty_points3 = torch.empty((0, 3), dtype=torch.float64)
        empty_mask = torch.empty((0,), dtype=torch.bool)
        _, history = self.ops.optimize_semantic_stage(
            base,
            corners,
            collision_pairs,
            empty_pairs,
            empty_pairs,
            empty_indices,
            empty_values,
            empty_indices,
            empty_points3,
            empty_points3,
            empty_mask,
            empty_pairs,
            empty_values,
            empty_pairs,
            empty_values,
            empty_values,
            empty_pairs,
            empty_values,
            iterations=5,
            active_set_router=True,
            active_set_checkpoints=(2, 4),
            active_set_high_degree=1,
        )
        self.assertEqual(history[-1]["router_protected_objects"], 0.0)
        self.assertEqual(history[-1]["router_budget_30"], 2.0)

    def test_depth_reprojection_is_zero_at_matching_bbox_and_depth(self):
        pose = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        pose[0, 2, 3] = -5.0
        corners = self.ops.local_box_corners(
            torch.full((1, 3), -0.5, dtype=torch.float64),
            torch.full((1, 3), 0.5, dtype=torch.float64),
        )
        # With fx=fy=100 and a box spanning z=[-5.5,-4.5], the extrema
        # project to 100 +/- 50/4.5.
        radius = 50.0 / 4.5
        observed_box = torch.tensor(
            [[100.0 - radius, 100.0 - radius, 100.0 + radius, 100.0 + radius]],
            dtype=torch.float64,
        )
        loss, centre, size, depth = self.ops.depth_aware_reprojection_loss(
            pose,
            corners,
            torch.tensor([0], dtype=torch.long),
            observed_box,
            torch.tensor([5.0], dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.bool),
            torch.eye(4, dtype=torch.float64),
            torch.tensor([200.0, 200.0, 100.0, 100.0], dtype=torch.float64),
        )
        self.assertLessEqual(loss.item(), 1e-12)
        self.assertLessEqual(centre.item(), 1e-12)
        self.assertLessEqual(size.item(), 1e-12)
        self.assertLessEqual(depth.item(), 1e-12)

    def test_depth_reprojection_provides_translation_gradient(self):
        base = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        base[0, 2, 3] = -5.0
        corners = self.ops.local_box_corners(
            torch.full((1, 3), -0.5, dtype=torch.float64),
            torch.full((1, 3), 0.5, dtype=torch.float64),
        )
        yaw, translation = self.ops.initialize_pose_variables(base)
        translation.data[0, 0] = 0.5
        translation.data[0, 2] = -6.0
        pose = self.ops.reproject_pose_matrices(base, yaw, translation)
        radius = 50.0 / 4.5
        observed_box = torch.tensor(
            [[100.0 - radius, 100.0 - radius, 100.0 + radius, 100.0 + radius]],
            dtype=torch.float64,
        )
        loss, _, _, _ = self.ops.depth_aware_reprojection_loss(
            pose,
            corners,
            torch.tensor([0], dtype=torch.long),
            observed_box,
            torch.tensor([5.0], dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.bool),
            torch.eye(4, dtype=torch.float64),
            torch.tensor([200.0, 200.0, 100.0, 100.0], dtype=torch.float64),
        )
        loss.backward()
        self.assertGreater(abs(translation.grad[0, 0].item()), 0.0)
        self.assertGreater(abs(translation.grad[0, 2].item()), 0.0)

    def test_depth_reprojection_component_weights_are_independent(self):
        pose = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        pose[0, 0, 3] = 0.5
        pose[0, 2, 3] = -6.0
        corners = self.ops.local_box_corners(
            torch.full((1, 3), -0.5, dtype=torch.float64),
            torch.full((1, 3), 0.5, dtype=torch.float64),
        )
        observed_box = torch.tensor(
            [[90.0, 90.0, 110.0, 110.0]], dtype=torch.float64
        )
        common = (
            pose,
            corners,
            torch.tensor([0], dtype=torch.long),
            observed_box,
            torch.tensor([5.0], dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.bool),
            torch.eye(4, dtype=torch.float64),
            torch.tensor(
                [200.0, 200.0, 100.0, 100.0], dtype=torch.float64
            ),
        )
        zero, _, _, _ = self.ops.depth_aware_reprojection_loss(
            *common,
            centre_weight=0.0,
            size_weight=0.0,
            metric_depth_weight=0.0,
        )
        centre, _, _, _ = self.ops.depth_aware_reprojection_loss(
            *common,
            centre_weight=1.0,
            size_weight=0.0,
            metric_depth_weight=0.0,
        )
        size, _, _, _ = self.ops.depth_aware_reprojection_loss(
            *common,
            centre_weight=0.0,
            size_weight=1.0,
            metric_depth_weight=0.0,
        )
        metric, _, _, _ = self.ops.depth_aware_reprojection_loss(
            *common,
            centre_weight=0.0,
            size_weight=0.0,
            metric_depth_weight=1.0,
        )
        combined, _, _, _ = self.ops.depth_aware_reprojection_loss(
            *common,
            centre_weight=1.0,
            size_weight=1.0,
            metric_depth_weight=1.0,
        )
        self.assertEqual(zero.item(), 0.0)
        self.assertGreater(centre.item(), 0.0)
        self.assertGreater(size.item(), 0.0)
        self.assertGreater(metric.item(), 0.0)
        self.assertAlmostEqual(
            combined.item(),
            centre.item() + size.item() + metric.item(),
        )

    def test_no_harm_reprojection_is_zero_inside_margins(self):
        reference_centre = torch.tensor([10.0, 20.0], dtype=torch.float64)
        reference_size = torch.tensor([0.1, 0.2], dtype=torch.float64)
        reference_depth = torch.tensor([0.05, 0.1], dtype=torch.float64)
        penalty, centre_excess, size_excess, depth_excess = (
            self.ops.no_harm_reprojection_penalty(
                reference_centre + torch.tensor(
                    [1.9, -3.0], dtype=torch.float64
                ),
                reference_size + torch.tensor(
                    [0.019, -0.05], dtype=torch.float64
                ),
                reference_depth + torch.tensor(
                    [0.009, -0.02], dtype=torch.float64
                ),
                reference_centre,
                reference_size,
                reference_depth,
                torch.ones(2, dtype=torch.float64),
            )
        )
        self.assertLessEqual(penalty.item(), 1e-12)
        self.assertLessEqual(centre_excess.amax().item(), 1e-12)
        self.assertLessEqual(size_excess.amax().item(), 1e-12)
        self.assertLessEqual(depth_excess.amax().item(), 1e-12)

    def test_no_harm_reprojection_penalizes_and_backpropagates_drift(self):
        centre = torch.tensor(
            [13.0], dtype=torch.float64, requires_grad=True
        )
        size = torch.tensor(
            [0.13], dtype=torch.float64, requires_grad=True
        )
        depth = torch.tensor(
            [0.08], dtype=torch.float64, requires_grad=True
        )
        penalty, centre_excess, size_excess, depth_excess = (
            self.ops.no_harm_reprojection_penalty(
                centre,
                size,
                depth,
                torch.tensor([10.0], dtype=torch.float64),
                torch.tensor([0.1], dtype=torch.float64),
                torch.tensor([0.05], dtype=torch.float64),
                torch.ones(1, dtype=torch.float64),
            )
        )
        self.assertGreater(penalty.item(), 0.0)
        self.assertGreater(centre_excess.item(), 0.0)
        self.assertGreater(size_excess.item(), 0.0)
        self.assertGreater(depth_excess.item(), 0.0)
        penalty.backward()
        self.assertGreater(abs(centre.grad.item()), 0.0)
        self.assertGreater(abs(size.grad.item()), 0.0)
        self.assertGreater(abs(depth.grad.item()), 0.0)

    def test_discrete_repair_accepts_bounded_image_centre_anchor(self):
        dtype = torch.float64
        base = torch.eye(4, dtype=dtype).unsqueeze(0)
        base[0, 2, 3] = -5.0
        minimum = torch.tensor([[-0.5, -0.5, -0.5]], dtype=dtype)
        maximum = torch.tensor([[0.5, 0.5, 0.5]], dtype=dtype)
        corners = self.ops.local_box_corners(minimum, maximum)
        repaired, report = self.ops.select_confident_discrete_pose_repairs(
            base,
            corners,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[475.0, 444.0, 575.0, 556.0]], dtype=dtype),
            torch.tensor([5.0], dtype=dtype),
            torch.tensor([True]),
            torch.eye(4, dtype=dtype),
            torch.tensor([1000.0, 1000.0, 500.0, 500.0], dtype=dtype),
            torch.tensor([False]),
            yaw_offsets_deg=(0.0,),
            max_translation_m=0.5,
            minimum_relative_improvement=0.01,
            minimum_runner_up_margin=0.0,
        )
        self.assertTrue(report[0]["accepted"])
        self.assertEqual(report[0]["selected_anchor"], "image_centre")
        self.assertGreater(len(report[0]["candidates"]), 1)
        self.assertTrue(
            any(
                candidate["selected_by_verifier"]
                for candidate in report[0]["candidates"]
            )
        )
        self.assertEqual(
            len(report[0]["candidates"][0]["pose_matrix_for_blender"]),
            4,
        )
        self.assertGreater(repaired[0, 0, 3].item(), 0.0)
        self.assertLessEqual(
            report[0]["selected_translation_shift_m"], 0.5
        )

    def test_discrete_repair_keeps_supported_height_and_rejects_large_anchor(self):
        dtype = torch.float64
        base = torch.eye(4, dtype=dtype).unsqueeze(0)
        base[0, 2, 3] = -5.0
        corners = self.ops.local_box_corners(
            torch.tensor([[-0.5, -0.5, -0.5]], dtype=dtype),
            torch.tensor([[0.5, 0.5, 0.5]], dtype=dtype),
        )
        repaired, report = self.ops.select_confident_discrete_pose_repairs(
            base,
            corners,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[800.0, 450.0, 900.0, 550.0]], dtype=dtype),
            torch.tensor([1.0], dtype=dtype),
            torch.tensor([True]),
            torch.eye(4, dtype=dtype),
            torch.tensor([1000.0, 1000.0, 500.0, 500.0], dtype=dtype),
            torch.tensor([True]),
            yaw_offsets_deg=(0.0,),
            max_translation_m=0.1,
        )
        self.assertFalse(report[0]["accepted"])
        self.assertTrue(torch.allclose(repaired, base))
        self.assertEqual(repaired[0, 2, 3].item(), base[0, 2, 3].item())

    def test_discrete_repair_persists_asset_center_offset_hypotheses(self):
        dtype = torch.float64
        base = torch.eye(4, dtype=dtype).unsqueeze(0)
        base[0, 2, 3] = -5.0
        corners = self.ops.local_box_corners(
            torch.tensor([[-0.5, -0.5, -0.5]], dtype=dtype),
            torch.tensor([[0.5, 0.5, 0.5]], dtype=dtype),
        )
        _, report = self.ops.select_confident_discrete_pose_repairs(
            base,
            corners,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[450.0, 450.0, 550.0, 550.0]], dtype=dtype),
            torch.tensor([5.0], dtype=dtype),
            torch.tensor([True]),
            torch.eye(4, dtype=dtype),
            torch.tensor([1000.0, 1000.0, 500.0, 500.0], dtype=dtype),
            torch.tensor([False]),
            visible_surface_depths=torch.tensor([4.0], dtype=dtype),
            surface_to_center_offsets=torch.tensor([1.0], dtype=dtype),
            enable_asset_center_candidates=True,
            asset_center_offset_scales=(0.5, 1.0, 1.5),
            yaw_offsets_deg=(0.0,),
            max_translation_m=1.0,
        )
        candidates = {
            candidate["anchor"]: candidate
            for candidate in report[0]["candidates"]
        }
        self.assertIn("asset_center_s050", candidates)
        self.assertIn("asset_center_s150", candidates)
        self.assertAlmostEqual(
            candidates["asset_center_s050"]["surface_to_center_offset_m"],
            1.0,
        )

    def test_support_surface_candidate_projects_complete_child_footprint(self):
        dtype = torch.float64
        base = torch.eye(4, dtype=dtype).repeat(2, 1, 1)
        base[:, 2, 3] = -5.0
        minimum = torch.tensor(
            [[-0.2, -0.2, -0.2], [-0.6, -0.6, -0.2]],
            dtype=dtype,
        )
        maximum = -minimum
        corners = self.ops.local_box_corners(minimum, maximum)
        repaired, report = self.ops.select_confident_discrete_pose_repairs(
            base,
            corners,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[560.0, 480.0, 600.0, 520.0]], dtype=dtype),
            torch.tensor([5.0], dtype=dtype),
            torch.tensor([True]),
            torch.eye(4, dtype=dtype),
            torch.tensor([1000.0, 1000.0, 500.0, 500.0], dtype=dtype),
            torch.tensor([True, False]),
            support_parent_indices=torch.tensor([1, -1], dtype=torch.long),
            enable_support_surface_candidates=True,
            yaw_offsets_deg=(0.0,),
            max_translation_m=1.0,
            minimum_relative_improvement=0.01,
            minimum_absolute_improvement=0.0,
            minimum_runner_up_margin=0.0,
        )
        support_candidates = [
            candidate
            for candidate in report[0]["candidates"]
            if candidate["anchor"] == "support_surface_joint"
        ]
        self.assertEqual(len(support_candidates), 1)
        candidate_pose = torch.tensor(
            support_candidates[0]["pose_matrix_for_blender"],
            dtype=dtype,
        )
        candidate_world = (
            corners[0] @ candidate_pose[:3, :3].transpose(0, 1)
            + candidate_pose[:3, 3]
        )
        self.assertLessEqual(candidate_world[:, 0].amax().item(), 0.6 + 1e-8)
        self.assertGreater(
            support_candidates[0]["support_projection_m"], 0.0
        )
        self.assertAlmostEqual(repaired[0, 2, 3].item(), -5.0)

    def test_plane_sibling_projection_is_minimum_tangent_only_and_certified(self):
        dtype = torch.float64
        poses = torch.eye(4, dtype=dtype).repeat(2, 1, 1)
        poses[1, 0, 3] = 0.5
        minimum = torch.tensor(
            [[-0.5, -0.05, -0.5], [-0.5, -0.05, -0.5]],
            dtype=dtype,
        )
        corners = self.ops.local_box_corners(minimum, -minimum)
        original_linear = poses[:, :3, :3].clone()
        audit = self.ops.project_plane_sibling_tangent_intervals_(
            poses,
            corners,
            torch.tensor([0, 1], dtype=torch.long),
            torch.zeros((2, 3), dtype=dtype),
            torch.tensor([[0.0, 1.0, 0.0]] * 2, dtype=dtype),
            torch.tensor([[0, 1]], dtype=torch.long),
            maximum_shift_m=0.35,
        )
        self.assertTrue(audit["accepted"])
        self.assertAlmostEqual(audit["maximum_shift_m"], 0.25)
        self.assertTrue(torch.equal(poses[:, :3, :3], original_linear))
        self.assertTrue(torch.equal(poses[:, 1:3, 3], torch.zeros((2, 2), dtype=dtype)))
        self.assertAlmostEqual(poses[0, 0, 3].item(), -0.25)
        self.assertAlmostEqual(poses[1, 0, 3].item(), 0.75)

    def test_plane_sibling_projection_abstains_with_zero_trust_region(self):
        dtype = torch.float64
        poses = torch.eye(4, dtype=dtype).repeat(2, 1, 1)
        poses[1, 0, 3] = 0.5
        incumbent = poses.clone()
        minimum = torch.tensor(
            [[-0.5, -0.05, -0.5], [-0.5, -0.05, -0.5]],
            dtype=dtype,
        )
        audit = self.ops.project_plane_sibling_tangent_intervals_(
            poses,
            self.ops.local_box_corners(minimum, -minimum),
            torch.tensor([0, 1], dtype=torch.long),
            torch.zeros((2, 3), dtype=dtype),
            torch.tensor([[0.0, 1.0, 0.0]] * 2, dtype=dtype),
            torch.tensor([[0, 1]], dtype=torch.long),
            maximum_shift_m=0.0,
        )
        self.assertFalse(audit["accepted"])
        self.assertEqual(audit["reason"], "all_components_abstained")
        self.assertEqual(
            audit["component_audits"][0]["reason"],
            "trust_region_exceeded",
        )
        self.assertTrue(torch.equal(poses, incumbent))

    def test_plane_sibling_projection_bounds_outlier_without_blocking_safe_component(self):
        dtype = torch.float64
        poses = torch.eye(4, dtype=dtype).repeat(4, 1, 1)
        poses[1, 0, 3] = 0.5
        poses[2, 0, 3] = 5.0
        poses[3, 0, 3] = 5.0
        minimum = torch.tensor(
            [
                [-0.5, -0.05, -0.5],
                [-0.5, -0.05, -0.5],
                [-1.0, -0.05, -0.5],
                [-1.0, -0.05, -0.5],
            ],
            dtype=dtype,
        )
        audit = self.ops.project_plane_sibling_tangent_intervals_(
            poses,
            self.ops.local_box_corners(minimum, -minimum),
            torch.arange(4, dtype=torch.long),
            torch.zeros((4, 3), dtype=dtype),
            torch.tensor([[0.0, 1.0, 0.0]] * 4, dtype=dtype),
            torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
            maximum_shift_m=0.35,
        )
        self.assertTrue(audit["accepted"])
        self.assertEqual(audit["components_accepted"], 1)
        self.assertEqual(audit["components_abstained"], 1)
        self.assertAlmostEqual(poses[0, 0, 3].item(), -0.25)
        self.assertAlmostEqual(poses[1, 0, 3].item(), 0.75)
        self.assertAlmostEqual(poses[2, 0, 3].item(), 5.0)
        self.assertAlmostEqual(poses[3, 0, 3].item(), 5.0)
        self.assertIn(
            "trust_region_exceeded",
            [component["reason"] for component in audit["component_audits"]],
        )

    def test_plane_component_image_gauge_uses_common_tangent_only(self):
        dtype = torch.float64
        base = torch.eye(4, dtype=dtype).repeat(2, 1, 1)
        base[0, 0, 3] = -0.5
        base[1, 0, 3] = 0.5
        base[:, 2, 3] = -5.0
        poses = base.clone()
        corners = torch.zeros((2, 8, 3), dtype=dtype)
        sibling_audit = {
            "accepted": True,
            "component_audits": [
                {
                    "accepted": True,
                    "object_indices": [0, 1],
                    "object_ids": ["panel_0", "panel_1"],
                }
            ],
        }
        audit = self.ops.refine_plane_component_image_gauge_(
            poses,
            base,
            corners,
            sibling_audit,
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([[0.0, 1.0, 0.0]] * 2, dtype=dtype),
            torch.empty((0, 2), dtype=torch.long),
            None,
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor(
                [[440.0, 500.0, 440.0, 500.0],
                 [640.0, 500.0, 640.0, 500.0]],
                dtype=dtype,
            ),
            torch.tensor([5.0, 5.0], dtype=dtype),
            torch.ones(2, dtype=dtype),
            torch.zeros(2, dtype=torch.bool),
            torch.eye(4, dtype=dtype),
            torch.tensor([1000.0, 1000.0, 1000.0, 1000.0], dtype=dtype),
        )
        self.assertEqual(audit["components_accepted"], 1)
        self.assertAlmostEqual(
            poses[1, 0, 3].item() - poses[0, 0, 3].item(), 1.0
        )
        self.assertAlmostEqual(poses[0, 1, 3].item(), 0.0)
        self.assertAlmostEqual(poses[0, 2, 3].item(), -5.0)
        self.assertGreater(poses[0, 0, 3].item(), base[0, 0, 3].item())

if __name__ == "__main__":
    unittest.main()
