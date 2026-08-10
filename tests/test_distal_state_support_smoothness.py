import itertools
import unittest
from pathlib import Path

import numpy as np

from main.multilink_ellipsoid.multi_region_affine_oracle import fixed_regions
from main.multilink_ellipsoid.state_support_smoothness import (
    boolean_jaccard, compare_oracles, context_feature_indexes,
    linear_quantile, load_config, oracle_signature,
)
from main.multilink_ellipsoid.two_step_margin import PAIR_FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]


class StateSupportSmoothnessTests(unittest.TestCase):
    def test_registered_config(self):
        config = load_config(
            ROOT / "configs/vlsa_distal_state_support_smoothness_moka10.v1.json"
        )
        self.assertEqual(config["immutable_source"]["expected_state_count"], 50)
        self.assertFalse(config["decision"]["closed_loop_E05_authorized"])
        self.assertFalse(config["decision"]["training_in_this_gate"])

    def test_linear_quantile_is_version_independent(self):
        self.assertEqual(linear_quantile([0.0, 10.0], 0.95), 9.5)
        self.assertEqual(linear_quantile([2.0], 0.25), 2.0)
        with self.assertRaises(ValueError):
            linear_quantile([], 0.5)

    def test_boolean_jaccard(self):
        self.assertEqual(boolean_jaccard([False], [False]), 1.0)
        self.assertAlmostEqual(
            boolean_jaccard([True, True, False], [True, False, True]), 1.0 / 3.0
        )

    def test_context_removes_candidate_and_constraint_identity(self):
        indexes = context_feature_indexes()
        names = [PAIR_FEATURE_NAMES[index] for index in indexes]
        self.assertEqual(len(names), 43)
        self.assertFalse(any(name.startswith("constraint_one_hot_") for name in names))
        self.assertFalse(any(name.startswith("candidate_first_xyz_") for name in names))
        self.assertIn("nominal_first_xyz_0", names)

    @staticmethod
    def _oracle(constant):
        axes = [np.linspace(-1.0, 1.0, 5)] * 3
        candidates = np.asarray(list(itertools.product(*axes)), dtype=np.float64)
        partition = {
            "axis_intervals": [[-1.0, 0.0], [-0.5, 0.5], [0.0, 1.0]],
            "axis_centers": [-0.5, 0.0, 0.5],
            "inclusive_membership_tolerance": 1.0e-12,
            "fit_actions_per_region": 27,
            "region_count": 27,
        }
        regions = fixed_regions(candidates, [-1.0] * 3, [1.0] * 3, partition)
        targets = []
        for region in regions:
            targets.append({
                "region_index": region["region_index"],
                "anchor_xyz": region["anchor_xyz"],
                "anchor_margin_m": [constant] * 7,
                "one_sided_error_m": [0.0] * 7,
                "gradient_m_per_action": [[0.0, 0.0, 0.0] for _ in range(7)],
            })
        state = {
            "candidate_first_xyz": candidates.tolist(),
            "action_lower": [-1.0] * 3,
            "action_upper": [1.0] * 3,
        }
        return oracle_signature(
            state, {"regions": regions, "regional_targets": targets}
        )

    def test_oracle_comparison_uses_induced_values_and_decisions(self):
        first = self._oracle(0.01)
        second = self._oracle(0.02)
        comparison = compare_oracles(first, second)
        self.assertAlmostEqual(comparison["union_margin_RMSE_m"], 0.01)
        self.assertAlmostEqual(comparison["regional_row_value_RMSE_m"], 0.01)
        self.assertEqual(comparison["accepted_set_Jaccard"], 1.0)
        self.assertEqual(comparison["left_accepted_count"], 125)
        self.assertEqual(comparison["right_accepted_count"], 125)


if __name__ == "__main__":
    unittest.main()
