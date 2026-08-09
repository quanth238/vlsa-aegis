import json
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from main.multilink_ellipsoid.affine_oracle_comparison import (
    affine_values, fit_direct_halfspaces, fit_finite_difference_lower_bounds,
    load_config, summarize_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "vlsa_distal_affine_oracle_comparison_moka10.v1.json"


class AffineOracleComparisonTests(unittest.TestCase):
    def test_config_is_frozen(self):
        config = load_config(CONFIG)
        self.assertEqual(config["fresh_actions"]["count_per_state"], 96)
        self.assertFalse(config["decision_gate"]["closed_loop_authorized"])

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy is unavailable")
    def test_direct_halfspace_has_no_fit_false_safe(self):
        xyz = np.asarray([
            [-1.0, 0.0, 0.0], [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0], [1.0, 0.0, 0.0],
        ])
        margin = np.tile(xyz[:, :1], (1, 7))
        config = load_config(CONFIG)
        fitted = fit_direct_halfspaces(
            xyz, margin, [0.0, 0.0, 0.0], [-1.0] * 3, [1.0] * 3,
            config["direct_halfspace"],
        )
        values = np.asarray([
            affine_values(
                fitted["intercept_at_nominal"], fitted["gradients_per_action"],
                item, [0.0, 0.0, 0.0],
            ) for item in xyz
        ])
        self.assertFalse(np.any(np.logical_and(values >= 0.0, margin < 0.0)))
        self.assertEqual(fitted["all_fit_false_safe_count"], 0)

    def test_finite_difference_is_tightened_on_fit_grid(self):
        xyz = np.asarray([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        margin = np.tile((0.1 + 0.2 * xyz[:, :1]), (1, 7))
        fitted = fit_finite_difference_lower_bounds(
            [0.1] * 7, [[0.12] * 7, [0.1] * 7, [0.1] * 7],
            [[0.08] * 7, [0.1] * 7, [0.1] * 7], 0.1,
            xyz, margin, [0.0, 0.0, 0.0], 1.0e-6,
        )
        values = np.asarray([
            affine_values(
                fitted["intercept_at_nominal_m"],
                fitted["gradients_m_per_action"], item, [0.0, 0.0, 0.0],
            ) for item in xyz
        ])
        self.assertLessEqual(float(np.max(values - margin)), 1.0e-10)

    def test_summary_counts_false_safe_and_recall(self):
        summary = summarize_predictions([
            {"true_safe": True, "predicted_safe": True},
            {"true_safe": True, "predicted_safe": False},
            {"true_safe": False, "predicted_safe": True},
        ])
        self.assertEqual(summary["false_safe_action_count"], 1)
        self.assertEqual(summary["safe_action_recall"], 0.5)

    def test_unknown_config_key_is_rejected(self):
        config = json.loads(CONFIG.read_text())
        config["unknown"] = True
        temporary = ROOT / "tests" / ".affine-oracle-comparison-invalid.json"
        try:
            temporary.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(temporary)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
