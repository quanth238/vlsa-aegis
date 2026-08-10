import json
from pathlib import Path
import unittest

import numpy as np

from main.multilink_ellipsoid.factorized_unsupported_state_audit import (
    interpret_state, load_unsupported_state_config, standardized_state_support,
    unsupported_state_indexes,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_factorized_unsupported_state_audit_moka10.v1.json"


class UnsupportedStateAuditTests(unittest.TestCase):
    def test_config_freezes_no_training_audit(self):
        config = load_unsupported_state_config(CONFIG)
        self.assertEqual(config["population"]["expected_unsupported_state_count"], 3)
        self.assertTrue(config["forbidden_actions"]["training"])
        self.assertTrue(config["forbidden_actions"]["verdict_change"])

    def test_unsupported_is_any_state_without_accepted_exact_safe_action(self):
        exact = np.asarray([[0.1], [0.2], [0.1], [-0.1]])
        predicted = np.asarray([[-0.1], [-0.2], [0.1], [0.1]])
        output = unsupported_state_indexes(
            exact_margin=np.repeat(exact, 7, axis=1),
            predicted_margin=np.repeat(predicted, 7, axis=1),
            state_index=np.asarray([0, 0, 1, 1]),
            selected=np.ones(4, dtype=bool), expected_count=1,
        )
        self.assertEqual(output, [0])

    def test_unsupported_includes_region_without_an_exact_safe_action(self):
        exact = np.repeat(np.asarray([[-0.1], [-0.2]]), 7, axis=1)
        predicted = np.repeat(np.asarray([[0.1], [0.2]]), 7, axis=1)
        output = unsupported_state_indexes(
            exact_margin=exact, predicted_margin=predicted,
            state_index=np.asarray([2, 2]), selected=np.ones(2, dtype=bool),
            expected_count=1,
        )
        self.assertEqual(output, [2])

    def test_support_uses_train_only_normalization(self):
        output = standardized_state_support(
            state_features=np.asarray([[0.0], [1.0], [0.1], [0.2], [0.15]]),
            state_indexes=[0, 1, 2, 3, 4],
            splits=["train", "train", "train", "train", "test"],
            case_ids=["a", "b", "c", "d", "e"],
            exact_safe_support=[True] * 5, query_indexes=[4], maximum_abs_z=5.0,
        )
        self.assertTrue(output["queries"]["4"]["support_pass"])

    def test_interpretation_prioritizes_coverage_then_decoder(self):
        config = json.loads(CONFIG.read_text())
        coverage = interpret_state(
            closest_predicted_margin_m=-0.001, support_pass=False,
            member_accepted_counts=[1, 0, 0, 0, 0],
            candidate_exact_safe_count=5, large_trajectory_error=True,
            config=config,
        )
        self.assertEqual(
            coverage["primary_explanation"], "unsupported_robot_or_controller_state"
        )
        decoder = interpret_state(
            closest_predicted_margin_m=-0.010, support_pass=True,
            member_accepted_counts=[0] * 5, candidate_exact_safe_count=5,
            large_trajectory_error=True, config=config,
        )
        self.assertEqual(decoder["primary_explanation"], "large_execution_trajectory_error")

    def test_interpretation_identifies_missing_local_safe_action_first(self):
        config = json.loads(CONFIG.read_text())
        output = interpret_state(
            closest_predicted_margin_m=None, support_pass=False,
            member_accepted_counts=[0] * 5, candidate_exact_safe_count=0,
            large_trajectory_error=True, config=config,
        )
        self.assertEqual(
            output["primary_explanation"],
            "no_exact_safe_action_in_registered_candidate_region",
        )


if __name__ == "__main__":
    unittest.main()
