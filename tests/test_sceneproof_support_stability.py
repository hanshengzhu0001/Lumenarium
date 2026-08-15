import unittest

import numpy as np

from modules._sceneproof_support_stability import (
    erode_convex_polygon,
    minimum_translation_into_convex_polygon,
    convex_polygon_intersection,
    physical_support_score,
    signed_margin_to_convex_polygon,
    stability_class,
    strongly_connected_components,
    ungrounded_cyclic_components,
    voxel_heightfield_contact_points,
    voxel_top_surface_component,
    voxel_vertical_first_contact,
)


class SceneProofSupportStabilityTest(unittest.TestCase):
    def test_minimum_translation_projects_into_eroded_support(self):
        polygon = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        )
        eroded = erode_convex_polygon(polygon, 0.1)
        self.assertEqual(
            {tuple(np.round(row, 12)) for row in eroded},
            {(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)},
        )
        delta = minimum_translation_into_convex_polygon(
            (1.2, 0.5), polygon, margin_m=0.1
        )
        np.testing.assert_allclose(delta, [-0.3, 0.0], atol=1e-12)

    def test_minimum_translation_is_zero_for_feasible_com(self):
        polygon = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(
            minimum_translation_into_convex_polygon(
                (0.5, 0.5), polygon, margin_m=0.1
            ),
            [0.0, 0.0],
        )

    def test_voxel_top_surface_selects_nearest_connected_component(self):
        first = [[x, y, 1.0] for x in (0.0, 0.1) for y in (0.0, 0.1)]
        second = [[x, y, 1.0] for x in (2.0, 2.1) for y in (0.0, 0.1)]
        hull, audit = voxel_top_surface_component(
            np.asarray(first + second, dtype=np.float64),
            query_xy=(2.05, 0.05),
            reference_z_m=1.0,
            grid_pitch_m=0.1,
            height_tolerance_m=0.01,
        )
        self.assertGreater(float(hull[:, 0].min()), 1.9)
        self.assertEqual(audit["connected_components"], 2)
        self.assertEqual(audit["selected_component_cells"], 4)

    def test_vertical_first_contact_keeps_xy_and_finds_highest_surface(self):
        child = np.asarray(
            [[x, y, 1.0] for x in (0.0, 0.1) for y in (0.0, 0.1)],
            dtype=np.float64,
        )
        supporter = np.asarray(
            [[x, y, 0.7] for x in (0.0, 0.1) for y in (0.0, 0.1)],
            dtype=np.float64,
        )
        drop, hull, audit = voxel_vertical_first_contact(
            child,
            supporter,
            grid_pitch_m=0.1,
            maximum_drop_m=0.5,
            penetration_tolerance_m=0.01,
            contact_band_m=0.01,
        )
        self.assertAlmostEqual(drop, 0.3)
        self.assertEqual(len(hull), 4)
        self.assertEqual(audit["first_contact_cells"], 4)

    def test_vertical_first_contact_rejects_existing_penetration(self):
        child = np.asarray(
            [[x, y, 1.0] for x in (0.0, 0.1) for y in (0.0, 0.1)],
            dtype=np.float64,
        )
        supporter = child.copy()
        supporter[:, 2] = 1.1
        with self.assertRaisesRegex(ValueError, "penetrates"):
            voxel_vertical_first_contact(
                child,
                supporter,
                grid_pitch_m=0.1,
                maximum_drop_m=0.5,
                penetration_tolerance_m=0.01,
                contact_band_m=0.01,
            )

    def test_signed_margin_distinguishes_stable_and_unstable_com(self):
        polygon = np.asarray(
            [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
        )
        self.assertAlmostEqual(
            signed_margin_to_convex_polygon((0.0, 0.0), polygon), 1.0
        )
        self.assertAlmostEqual(
            signed_margin_to_convex_polygon((1.25, 0.0), polygon), -0.25
        )
        self.assertEqual(stability_class(-0.25), "unstable")
        self.assertEqual(stability_class(0.001), "marginal")
        self.assertEqual(stability_class(0.02), "stable")

    def test_contact_patch_uses_mesh_polygon_intersection(self):
        child = np.asarray(
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
        )
        parent = np.asarray(
            [[0.0, -1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 1.0]]
        )
        contact = convex_polygon_intersection(child, parent)
        self.assertAlmostEqual(contact[:, 0].min(), 0.0)
        self.assertAlmostEqual(contact[:, 0].max(), 0.5)
        self.assertAlmostEqual(
            signed_margin_to_convex_polygon((0.4, 0.0), contact), 0.1
        )
        self.assertLess(
            signed_margin_to_convex_polygon((-0.1, 0.0), contact), 0.0
        )

    def test_support_score_matches_frozen_physical_evaluator(self):
        row = {
            "support_contact_gap_m": "0.0",
            "support_containment_error_m": "0.025",
            "support_footprint_overlap_ratio": "0.9",
        }
        self.assertAlmostEqual(physical_support_score(row), 5.0 / 6.0)
        self.assertIsNone(physical_support_score({}))

    def test_voxel_heightfield_recovers_curved_contact_patch(self):
        child = np.asarray(
            [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0],
             [0.1, 0.1, 1.0], [0.0, 0.1, 1.0],
             [0.05, 0.05, 1.2]]
        )
        supporter = np.asarray(
            [[0.0, 0.0, 0.99], [0.1, 0.0, 0.99],
             [0.1, 0.1, 0.99], [0.0, 0.1, 0.99]]
        )
        points, audit = voxel_heightfield_contact_points(
            child, supporter, grid_pitch_m=0.1, contact_tolerance_m=0.02
        )
        self.assertEqual(len(points), 4)
        self.assertEqual(audit["accepted_contact_cells"], 4.0)

    def test_support_scc_detects_mutual_cycle(self):
        components = strongly_connected_components(
            {"laptop": {"holder"}, "holder": {"laptop"}, "table": set()}
        )
        self.assertIn(["holder", "laptop"], components)

    def test_grounded_cycle_reaches_floor_through_table(self):
        ungrounded, grounded = ungrounded_cyclic_components(
            {
                "laptop": {"holder", "desk"},
                "holder": {"laptop", "desk"},
                "desk": {"floor_0"},
            },
            {"floor_0"},
        )
        self.assertEqual(ungrounded, [])
        self.assertIn(["holder", "laptop"], grounded)

    def test_floating_cycle_remains_unproven(self):
        ungrounded, grounded = ungrounded_cyclic_components(
            {"a": {"b"}, "b": {"a"}}, {"floor_0"}
        )
        self.assertEqual(grounded, [])
        self.assertIn(["a", "b"], ungrounded)


if __name__ == "__main__":
    unittest.main()
