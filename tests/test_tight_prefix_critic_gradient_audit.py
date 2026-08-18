from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path

from main.multilink_ellipsoid.tight_prefix_critic_gradient_audit import (
    aggregate_pair_metrics, feasibility_verdict, load_config,
    pair_direction_record, smoothmax_weights, unsafe_recall,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_tight_prefix_critic_gradient_audit.v1.json"


def _pair(
    *, state="s0", symmetry=0.0, minus_score=-1.0, plus_score=1.0,
    nominal=0.2, minus=-0.1, plus=0.3,
):
    return pair_direction_record(
        state_id=state, pair_names=["minus", "plus"],
        symmetry_error=symmetry,
        predicted_minus_score=minus_score,
        predicted_plus_score=plus_score,
        nominal_true_risk=nominal,
        minus_true_risk=minus,
        plus_true_risk=plus,
        symmetry_tolerance=1.0e-8,
        score_tolerance=1.0e-12,
        risk_tolerance=1.0e-10,
    )


class TightPrefixCriticGradientAuditTest(unittest.TestCase):
    def test_config_freezes_zero_simulation_directional_gate(self):
        config = load_config(CONFIG)
        self.assertEqual(config["audit"]["decision_split"], "validation")
        self.assertEqual(len(config["audit"]["candidate_pairs"]), 6)
        self.assertIn("new_simulator_rollout", config["forbidden"])
        self.assertIn("denoising_or_flow_guidance", config["forbidden"])
        self.assertEqual(
            config["structural_audit"]["per_row_unrepresented_tangent_dimensions"],
            10,
        )

    def test_smoothmax_weights_are_stable_and_normalized(self):
        weights = smoothmax_weights([-1000.0, 0.0, 1.0], 20.0)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertGreater(weights[2], weights[1])
        self.assertEqual(weights[0], 0.0)

    def test_pair_classifies_correct_descent_and_safe_conversion(self):
        value = _pair()
        self.assertTrue(value["eligible_symmetric_pair"])
        self.assertEqual(value["predicted_selected_candidate"], "minus")
        self.assertTrue(value["direction_correct"])
        self.assertTrue(value["true_risk_descends_from_nominal"])
        self.assertTrue(value["unsafe_nominal_converted_safe"])
        self.assertFalse(value["safe_nominal_regressed_unsafe"])

    def test_asymmetric_pair_is_excluded(self):
        value = _pair(symmetry=1.0e-4)
        self.assertFalse(value["eligible_symmetric_pair"])
        self.assertIsNone(value["predicted_selected_candidate"])
        metrics = aggregate_pair_metrics([value])
        self.assertEqual(metrics["eligible_symmetric_pair_count"], 0)
        self.assertIsNone(metrics["direction_accuracy"])

    def test_gate_requires_trigger_direction_descent_and_support(self):
        config = load_config(CONFIG)
        records = []
        for index in range(8):
            records.append(_pair(state="s%d" % (index % 2)))
        pairs = aggregate_pair_metrics(records)
        trigger = unsafe_recall([
            {"actual": 0.1, "predicted": 0.2},
            {"actual": 0.2, "predicted": 0.3},
            {"actual": -0.1, "predicted": -0.2},
        ])
        verdict = feasibility_verdict(
            trigger=trigger, pairs=pairs, numerical_gradient_pass=True,
            gate=config["feasibility_gate"],
        )
        self.assertTrue(verdict["all_checks_pass"])
        self.assertTrue(verdict["action_space_exact_gradient_probe_authorized"])
        self.assertFalse(verdict["flow_guidance_authorized"])

    def test_false_safe_trigger_blocks_probe(self):
        config = load_config(CONFIG)
        records = [_pair(state="s%d" % (index % 2)) for index in range(8)]
        trigger = unsafe_recall([
            {"actual": 0.1, "predicted": -0.2},
            {"actual": 0.2, "predicted": 0.3},
        ])
        verdict = feasibility_verdict(
            trigger=trigger, pairs=aggregate_pair_metrics(records),
            numerical_gradient_pass=True, gate=config["feasibility_gate"],
        )
        self.assertFalse(verdict["checks"]["unsafe_recall"])
        self.assertFalse(verdict["all_checks_pass"])

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") and importlib.util.find_spec("torch"),
        "NumPy and PyTorch unavailable",
    )
    def test_autograd_helper_matches_centered_finite_difference(self):
        import numpy as np
        import torch
        from scripts.audit_tight_prefix_critic_gradient import (
            _value_and_gradient,
        )

        model = torch.nn.Sequential(torch.nn.Linear(7, 1, bias=False)).double()
        with torch.no_grad():
            model[0].weight.copy_(
                torch.as_tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.float64)
            )
        bundle = {
            "model": model,
            "feature_mean": np.zeros(7),
            "feature_scale": np.ones(7),
            "target_mean": np.zeros(1),
            "target_scale": np.ones(1),
        }
        value, gradient, maximum_error = _value_and_gradient(
            bundle, [1.0] * 7, epsilon=1.0e-5,
        )
        self.assertAlmostEqual(value, 28.0)
        self.assertEqual(gradient, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertLess(maximum_error, 1.0e-8)


if __name__ == "__main__":
    unittest.main()
