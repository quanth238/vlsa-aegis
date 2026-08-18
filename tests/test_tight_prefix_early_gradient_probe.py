"""Tests for the immediately-earlier-query frozen-critic probe."""

from __future__ import annotations

import unittest
from pathlib import Path

from main.multilink_ellipsoid.tight_prefix_early_gradient_probe import (
    hypothesis_verdict, load_config, summarize_cases,
    symmetric_multi_probe_actions,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_tight_prefix_early_gradient_probe.v1.json"


class TightPrefixEarlyGradientProbeTests(unittest.TestCase):
    def test_config_freezes_one_query_earlier_validation_only(self) -> None:
        value = load_config(CONFIG)
        self.assertEqual([row["early_state_step"] for row in value["cases"]], [180, 20])
        self.assertTrue(all(row["split"] == "validation" for row in value["cases"]))
        self.assertIn("model_retraining_or_checkpoint_change", value["forbidden"])
        self.assertIn("trust_region_enlargement", value["forbidden"])

    def test_multi_probe_is_equal_norm_and_symmetric(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is unavailable")
        anchor = np.zeros((5, 7), dtype=np.float64)
        gradient = np.arange(15, dtype=np.float64).reshape(5, 3) + 1.0
        random = np.stack([
            np.roll(gradient, shift, axis=0) for shift in range(1, 5)
        ])
        result = symmetric_multi_probe_actions(
            anchor, gradient, random, requested_radius=0.25,
            minimum_radius=0.01, bound_headroom=0.8,
        )
        self.assertEqual(len(result["matched_random"]), 4)
        self.assertLessEqual(result["post_clipping_symmetry_error"], 1e-15)
        self.assertLessEqual(result["equal_norm_maximum_error"], 1e-15)

    def test_hypothesis_requires_safe_supported_descent_and_random_win(self) -> None:
        records = [{
            "eligible": True, "nominal_exact_safe": True, "triggered": True,
            "gradient_down_exact_safe": True,
            "gradient_down_descends": True,
            "gradient_down_beats_up": True,
            "gradient_down_beats_random": [True, True, True, False],
            "down_risk_change": -0.01,
        }]
        summary = summarize_cases(records)
        verdict = hypothesis_verdict(summary, {
            "minimum_eligible_case_count": 1,
            "minimum_random_win_rate": 0.75,
        })
        self.assertTrue(verdict["early_bounded_gradient_hypothesis_confirmed"])

    def test_no_eligible_state_cannot_confirm(self) -> None:
        record = {
            "eligible": False, "nominal_exact_safe": True, "triggered": False,
            "gradient_down_exact_safe": False,
            "gradient_down_descends": False,
            "gradient_down_beats_up": False,
            "gradient_down_beats_random": [], "down_risk_change": 0.0,
        }
        verdict = hypothesis_verdict(summarize_cases([record]), {
            "minimum_eligible_case_count": 1, "minimum_random_win_rate": 0.75,
        })
        self.assertFalse(verdict["early_bounded_gradient_hypothesis_confirmed"])

    def test_nominal_prep_retains_legacy_zero_alpha_alias(self) -> None:
        import numpy as np
        from scripts.prepare_tight_prefix_early_gradient_probe import _nominal_only

        record = _nominal_only(np.zeros((5, 7)), {}, {})[0]
        self.assertEqual(record["requested_alpha"], 0.0)
        self.assertEqual(record["requested_correction_l2_action"], 0.0)

    def test_video_comparisons_bind_one_gradient_win_and_loss(self) -> None:
        from scripts.render_tight_prefix_early_gradient_comparisons import (
            comparison_specs,
        )

        result = comparison_specs({
            "gradient_down": -0.114290,
            "matched_random_1": -0.115462,
            "matched_random_3": -0.110182,
        })
        self.assertEqual(result[0]["winner"], "gradient_down")
        self.assertEqual(result[1]["winner"], "matched_random_1")


if __name__ == "__main__":
    unittest.main()
