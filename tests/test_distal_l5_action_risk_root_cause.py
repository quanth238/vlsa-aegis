import unittest

from main.multilink_ellipsoid.l5_action_risk_root_cause import (
    feature_support, root_cause_decision,
)


class L5RootCauseAuditTests(unittest.TestCase):
    def test_feature_support_separates_context_and_action(self):
        train = [[0.0] * 86, [0.0] * 86]
        train[1][51] = 2.0
        query = [[0.0] * 86]
        query[0][51] = 1.0
        report = feature_support(
            train, query, ["same", "same"], ["same"], [1.0] * 86
        )
        self.assertEqual(report["same_state_in_fit_count"], 1)
        self.assertEqual(
            report["groups"]["state_context_0_50"]["maximum_nearest_rms_z"], 0.0
        )
        self.assertGreater(
            report["groups"]["candidate_residual_51_85"]["minimum_nearest_rms_z"], 0.0
        )

    def test_coverage_is_primary_when_interpolation_passes(self):
        report = root_cause_decision(
            train_rmse_m=0.0002, action_holdout_rmse_m=0.0003,
            grouped_validation_rmse_m=0.03,
            train_boundary_states=[2, 2, 1],
            validation_boundary_states=[2, 0, 0],
            train_active_witnesses=[25, 0, 0],
            train_per_row_false_safe=[0, 1, 0],
        )
        self.assertEqual(
            report["primary_root_cause"],
            "insufficient_grouped_state_and_active_boundary_coverage",
        )
        self.assertTrue(report["symmetric_fit_retains_per_row_false_safe"])
        self.assertFalse(report["input_representation_causality_proven"])


if __name__ == "__main__":
    unittest.main()
