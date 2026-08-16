from __future__ import annotations

import unittest
from pathlib import Path

from main.multilink_ellipsoid.whole_body_q_only_prediction import (
    evaluate_prediction_gate, load_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/vlsa_distal_whole_body_q_only_prediction_protocol.v1.json"


def _metrics(*, safe=True, rank=0.8, boundary=0.04):
    return {
        "global": {
            "near_boundary_RMSE": boundary,
            "rank_spearman": rank,
            "states": [{
                "state_id": "x",
                "actual_safe_candidate_count": 1,
                "predicted_safe_support": True,
                "selected_actual_safe": safe,
            }],
        }
    }


class WholeBodyQOnlyPredictionTest(unittest.TestCase):
    def test_protocol_freezes_diagnostic_training_and_test_gate(self):
        protocol = load_protocol(PROTOCOL)
        self.assertTrue(protocol["dataset"]["train_even_if_coverage_fails_as_diagnostic"])
        self.assertEqual(protocol["model"]["weight_decay"], 0.0001)
        self.assertEqual(protocol["prediction_gate"]["maximum_selected_false_safe_count"], 0)

    def test_prediction_gate_requires_every_heldout_split(self):
        protocol = load_protocol(PROTOCOL)
        passed = evaluate_prediction_gate(
            {"validation": _metrics(), "test": _metrics()},
            protocol, source_replay_exact=True,
        )
        self.assertTrue(passed["passes"])
        failed = evaluate_prediction_gate(
            {"validation": _metrics(), "test": _metrics(safe=False)},
            protocol, source_replay_exact=True,
        )
        self.assertFalse(failed["passes"])


if __name__ == "__main__":
    unittest.main()
