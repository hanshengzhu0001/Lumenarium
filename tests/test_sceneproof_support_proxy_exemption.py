import math
import unittest

from sceneproof_local_settle_component_gate_fix84 import (
    evaluate_support_exemption,
    linear_score,
    reconstruct_support_score,
    support_attribution,
)


def probe(
    *,
    before_gap=0.004280238101879757,
    after_gap=0.002885860080520313,
    before_margin=-0.06059356999141394,
    after_margin=0.24414690862473928,
    stability="stable",
    certificate="certified",
    parent_contact=True,
    parent_id="floor_0",
):
    return {
        "object_id": "single_sofa_chair_1",
        "before_support": {
            "declared_parent_id": parent_id,
            "com_signed_margin_m": before_margin,
            "contact_gap_by_supporter_m": {parent_id: before_gap},
            "stability_class": "unstable",
            "certificate_status": "certified",
            "declared_parent_contact_present": True,
        },
        "after_support": {
            "declared_parent_id": parent_id,
            "com_signed_margin_m": after_margin,
            "contact_gap_by_supporter_m": {parent_id: after_gap},
            "stability_class": stability,
            "certificate_status": certificate,
            "declared_parent_contact_present": parent_contact,
        },
    }


def attribution(
    *,
    official=25,
    reconstructed=24,
    missing=1,
    mutated_delta=-0.2,
    others=None,
    accounted=None,
):
    if accounted is None:
        accounted = reconstructed + missing == official
    return {
        "official_term_count": official,
        "reconstructed_term_count": reconstructed,
        "missing_support_parents": missing,
        "all_terms_accounted_for": accounted,
        "reconstructed_incumbent_score": 0.599936,
        "reconstructed_candidate_score": 0.591936,
        "reconstructed_delta": mutated_delta / official if official else None,
        "mutated_object_delta": mutated_delta,
        "other_object_deltas": others or {},
        "changed_object_ids": ["single_sofa_chair_1"],
    }


class SupportScoreReconstructionTest(unittest.TestCase):
    def test_infinite_containment_scores_zero_not_none(self):
        score = reconstruct_support_score(
            {
                "support_contact_gap_m": "0.02",
                "support_containment_error_m": "inf",
                "support_footprint_overlap_ratio": "0.0",
            }
        )
        self.assertIsNotNone(score)
        self.assertAlmostEqual(score, linear_score(0.02, 0.05) / 3.0)

    def test_inside_containment_path_takes_precedence(self):
        score = reconstruct_support_score(
            {
                "inside_containment_error_m": "0.01",
                "support_contact_gap_m": "0.02",
                "support_containment_error_m": "0.0",
                "support_footprint_overlap_ratio": "1.0",
            }
        )
        self.assertAlmostEqual(score, linear_score(0.01, 0.05))

    def test_object_without_support_parent_contributes_nothing(self):
        self.assertIsNone(reconstruct_support_score({"object_id": "curtain_0"}))


class SupportAttributionTest(unittest.TestCase):
    def rows(self, *, candidate_gap="0.0839911488570445"):
        template = {
            "support_containment_error_m": "inf",
            "support_footprint_overlap_ratio": "0.0",
        }
        return [
            {
                "version": "inc",
                "object_id": "single_sofa_chair_1",
                "support_contact_gap_m": "0.02",
                **template,
            },
            {
                "version": "cand",
                "object_id": "single_sofa_chair_1",
                "support_contact_gap_m": candidate_gap,
                **template,
            },
            {
                "version": "inc",
                "object_id": "desk_0",
                "support_contact_gap_m": "0.02",
                **template,
            },
            {
                "version": "cand",
                "object_id": "desk_0",
                "support_contact_gap_m": "0.02",
                **template,
            },
        ]

    def test_attribution_isolates_the_mutated_object(self):
        result = support_attribution(
            self.rows(),
            "inc",
            "cand",
            "single_sofa_chair_1",
            epsilon=1e-9,
            official_term_count=2,
            missing_support_parents=0,
        )
        self.assertEqual(result["reconstructed_term_count"], 2)
        self.assertEqual(result["official_term_count"], 2)
        self.assertTrue(result["all_terms_accounted_for"])
        self.assertEqual(result["other_object_deltas"], {})
        self.assertLess(result["mutated_object_delta"], 0.0)

    def test_attribution_reports_collateral_changes(self):
        rows = self.rows()
        for row in rows:
            if row["version"] == "cand" and row["object_id"] == "desk_0":
                row["support_contact_gap_m"] = "0.04"
        result = support_attribution(
            rows,
            "inc",
            "cand",
            "single_sofa_chair_1",
            epsilon=1e-9,
            official_term_count=2,
            missing_support_parents=0,
        )
        self.assertIn("desk_0", result["other_object_deltas"])

    def test_missing_parent_terms_are_accounted_for(self):
        # A declared support parent absent from the geometry contributes a
        # constant zero to the support mean without emitting any support_*
        # column, so the reconstruction is short by exactly that count.
        result = support_attribution(
            self.rows(),
            "inc",
            "cand",
            "single_sofa_chair_1",
            epsilon=1e-9,
            official_term_count=3,
            missing_support_parents=1,
        )
        self.assertEqual(result["reconstructed_term_count"], 2)
        self.assertTrue(result["all_terms_accounted_for"])

    def test_unexplained_term_count_is_flagged(self):
        result = support_attribution(
            self.rows(),
            "inc",
            "cand",
            "single_sofa_chair_1",
            epsilon=1e-9,
            official_term_count=5,
            missing_support_parents=1,
        )
        self.assertFalse(result["all_terms_accounted_for"])

    def test_reconstruction_uses_the_official_denominator(self):
        result = support_attribution(
            self.rows(),
            "inc",
            "cand",
            "single_sofa_chair_1",
            epsilon=1e-9,
            official_term_count=3,
            missing_support_parents=1,
        )
        # Two reconstructed terms divided by the official three-term mean.
        expected_incumbent = (
            2 * (linear_score(0.02, 0.05) / 3.0)
        ) / 3
        self.assertAlmostEqual(
            result["reconstructed_incumbent_score"], expected_incumbent
        )


class SupportExemptionTest(unittest.TestCase):
    def evaluate(self, **kwargs):
        defaults = dict(
            probe=probe(),
            attribution=attribution(),
            family_delta=-0.2 / 25,
            mutated_object_id="single_sofa_chair_1",
            epsilon=1e-9,
            contact_tolerance_m=0.05,
            com_margin_epsilon=1e-9,
            term_count_unchanged=True,
            missing_support_parents_unchanged=True,
        )
        defaults.update(kwargs)
        return evaluate_support_exemption(**defaults)

    def test_witnessed_proxy_artefact_is_exempted(self):
        result = self.evaluate()
        self.assertTrue(result["granted"], result["failed_conditions"])

    def test_changed_term_count_blocks_the_exemption(self):
        # An object gained or lost a support constraint, so the two means are
        # taken over different constraint sets and are not comparable.
        result = self.evaluate(term_count_unchanged=False)
        self.assertFalse(result["granted"])
        self.assertIn("e0_term_count_unchanged", result["failed_conditions"])

    def test_changed_missing_parent_count_blocks_the_exemption(self):
        result = self.evaluate(missing_support_parents_unchanged=False)
        self.assertFalse(result["granted"])
        self.assertIn(
            "e0_missing_support_parents_unchanged", result["failed_conditions"]
        )

    def test_unexplained_support_terms_block_the_exemption(self):
        result = self.evaluate(
            attribution=attribution(official=30, reconstructed=24, missing=1)
        )
        self.assertFalse(result["granted"])
        self.assertIn("e0_all_terms_accounted_for", result["failed_conditions"])

    def test_collateral_regression_blocks_the_exemption(self):
        result = self.evaluate(
            attribution=attribution(others={"desk_0": -0.05})
        )
        self.assertFalse(result["granted"])
        self.assertIn("e1_only_mutated_object_changed", result["failed_conditions"])

    def test_marginal_stability_is_not_sufficient(self):
        result = self.evaluate(probe=probe(stability="marginal"))
        self.assertFalse(result["granted"])
        self.assertIn(
            "e3_true_mesh_certified_stable", result["failed_conditions"]
        )

    def test_uncertified_support_is_not_sufficient(self):
        result = self.evaluate(probe=probe(certificate="abstain"))
        self.assertFalse(result["granted"])
        self.assertIn(
            "e3_true_mesh_certified_stable", result["failed_conditions"]
        )

    def test_com_margin_must_strictly_improve(self):
        result = self.evaluate(
            probe=probe(before_margin=0.2, after_margin=0.2)
        )
        self.assertFalse(result["granted"])
        self.assertIn(
            "e4_com_margin_strictly_improved", result["failed_conditions"]
        )

    def test_true_mesh_contact_regression_blocks_the_exemption(self):
        result = self.evaluate(probe=probe(after_gap=0.03))
        self.assertFalse(result["granted"])
        self.assertIn(
            "e5_true_mesh_contact_not_worse", result["failed_conditions"]
        )

    def test_real_loss_of_contact_is_never_exempted(self):
        # The real mesh left the parent surface entirely.  Even with a better
        # COM margin this must fail: it is a genuine physical regression.
        result = self.evaluate(
            probe=probe(before_gap=0.004, after_gap=0.30)
        )
        self.assertFalse(result["granted"])
        self.assertIn(
            "e6_true_mesh_still_in_contact", result["failed_conditions"]
        )

    def test_regression_larger_than_one_object_is_rejected(self):
        result = self.evaluate(
            attribution=attribution(
                official=4, reconstructed=4, missing=0, mutated_delta=-0.2
            ),
            family_delta=-0.5,
        )
        self.assertFalse(result["granted"])
        self.assertIn(
            "e2_within_single_object_bound", result["failed_conditions"]
        )

    def test_family_delta_must_match_the_attributed_object(self):
        result = self.evaluate(family_delta=-0.05)
        self.assertFalse(result["granted"])
        self.assertIn(
            "e1_family_delta_explained_by_mutated_object",
            result["failed_conditions"],
        )

    def test_positive_mutated_delta_is_not_an_exemption_case(self):
        result = self.evaluate(
            attribution=attribution(mutated_delta=0.2),
            family_delta=0.2 / 25,
        )
        self.assertFalse(result["granted"])
        self.assertIn(
            "e1_mutated_object_delta_present", result["failed_conditions"]
        )

    def test_missing_parent_contact_blocks_the_exemption(self):
        result = self.evaluate(probe=probe(parent_contact=False))
        self.assertFalse(result["granted"])
        self.assertIn(
            "e3_true_mesh_certified_stable", result["failed_conditions"]
        )

    def test_exemption_is_scene_independent(self):
        # A different scene, object, parent, term count, and missing-parent
        # count: the same witness structure still qualifies.  Nothing in the
        # rule refers to a specific scene or object.
        other = probe(
            parent_id="side_table_3",
            before_gap=0.001,
            after_gap=0.0005,
            before_margin=-0.01,
            after_margin=0.02,
        )
        other["object_id"] = "vase_7"
        result = self.evaluate(
            probe=other,
            attribution=attribution(
                official=11, reconstructed=9, missing=2, mutated_delta=-0.33
            ),
            family_delta=-0.33 / 11,
            mutated_object_id="vase_7",
        )
        self.assertTrue(result["granted"], result["failed_conditions"])

    def test_fix82_measured_values_are_exempted(self):
        # The exact numbers measured on the A10 for single_sofa_chair_1.
        result = self.evaluate(
            attribution=attribution(
                official=25,
                reconstructed=24,
                missing=1,
                mutated_delta=-0.20000025710056737,
            ),
            family_delta=-0.008000010284022752,
        )
        self.assertTrue(result["granted"], result["failed_conditions"])
        self.assertAlmostEqual(result["single_object_bound"], 1.0 / 25)


if __name__ == "__main__":
    unittest.main()
