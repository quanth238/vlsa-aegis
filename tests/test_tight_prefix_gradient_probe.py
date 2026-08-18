"""Tests for the exact compact-critic gradient feasibility probe."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from main.multilink_ellipsoid.tight_prefix_gradient_probe import (
    aggregate_anchor_records, feasibility_verdict, load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_tight_prefix_gradient_probe.v1.json"


class TightPrefixGradientProbeTests(unittest.TestCase):
    def test_config_freezes_six_opened_validation_anchors(self) -> None:
        value = load_config(CONFIG)
        self.assertEqual(len(value["cases"]), 2)
        self.assertEqual(sum(
            len(item["unsafe_anchor_candidates"]) for item in value["cases"]
        ), 6)
        self.assertEqual(value["probe"]["probe_names"], [
            "gradient_down", "gradient_up", "matched_random",
        ])
        self.assertIn("test_split_use", value["forbidden"])

    def test_arbitrary_action_replay_is_additive_and_opt_in(self) -> None:
        source = (
            ROOT / "scripts/audit_distal_compiled_box_risk_target.py"
        ).read_text()
        self.assertIn(
            'action_overrides = audit_config.get("candidate_action_overrides")',
            source,
        )
        self.assertIn(
            "actions, phases = candidate_action_sequence(", source,
        )
        self.assertIn(
            '"compiled-box action override requires prefix-only rollout"',
            source,
        )

    def test_symmetric_probe_has_equal_norm_and_no_bound_crossing(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is not installed in the local system Python")
        from main.multilink_ellipsoid.tight_prefix_gradient_probe import (
            symmetric_probe_actions,
        )

        anchor = np.zeros((5, 7), dtype=np.float64)
        anchor[0, 0] = 1.0
        gradient = np.arange(15, dtype=np.float64).reshape(5, 3) + 1.0
        random = np.flip(gradient, axis=0).copy()
        result = symmetric_probe_actions(
            anchor, gradient, random, requested_radius=0.25,
            minimum_radius=0.01, bound_headroom=0.8,
        )
        down = np.asarray(result["gradient_down"])
        up = np.asarray(result["gradient_up"])
        matched = np.asarray(result["matched_random"])
        self.assertEqual(down[0, 0], 1.0)
        self.assertEqual(up[0, 0], 1.0)
        self.assertLessEqual(float(np.max(np.abs(down[:, :3]))), 1.0)
        self.assertLessEqual(result["post_clipping_symmetry_error"], 1e-15)
        self.assertLessEqual(result["equal_norm_maximum_error"], 1e-15)
        self.assertAlmostEqual(
            float(np.linalg.norm(matched[:, :3] - anchor[:, :3])),
            result["radius_l2_action"], places=14,
        )

    def test_metrics_and_gate(self) -> None:
        config = json.loads(CONFIG.read_text())
        records = []
        for index in range(6):
            records.append({
                "case_id": "case-%d" % (index % 2),
                "direction_correct": index < 4,
                "gradient_down_descends": index < 4,
                "gradient_down_beats_random": index < 3,
                "gradient_down_converts_safe": index == 0,
                "down_risk_change": -0.1 if index < 4 else 0.01,
                "anchor_prediction_absolute_error": 1e-6,
                "post_clipping_symmetry_error": 0.0,
            })
        metrics = aggregate_anchor_records(records)
        verdict = feasibility_verdict(metrics, config["gate"])
        self.assertEqual(metrics["probe_rollout_count"], 18)
        self.assertTrue(verdict["all_checks_pass"])
        self.assertFalse(verdict["flow_guidance_or_QP_authorized"])

    def test_gate_rejects_non_descent(self) -> None:
        config = json.loads(CONFIG.read_text())
        records = [{
            "case_id": "case",
            "direction_correct": False,
            "gradient_down_descends": False,
            "gradient_down_beats_random": False,
            "gradient_down_converts_safe": False,
            "down_risk_change": 0.1,
            "anchor_prediction_absolute_error": 0.0,
            "post_clipping_symmetry_error": 0.0,
        } for _ in range(6)]
        verdict = feasibility_verdict(
            aggregate_anchor_records(records), config["gate"],
        )
        self.assertFalse(verdict["all_checks_pass"])


if __name__ == "__main__":
    unittest.main()
