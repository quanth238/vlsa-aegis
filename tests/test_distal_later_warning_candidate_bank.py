import unittest
from pathlib import Path

from main.multilink_ellipsoid.later_warning_candidate_bank import (
    derive_replay_archive, load_config, minimum_actual_intervention,
)


ROOT = Path(__file__).resolve().parents[1]


class LaterWarningCandidateBankTest(unittest.TestCase):
    def test_config(self):
        config = load_config(
            ROOT / "configs/vlsa_distal_later_warning_candidate_bank.v1.json"
        )
        self.assertEqual(config["state_step"], 105)
        self.assertEqual(config["query_index"], 21)

    def test_minimum_actual_intervention_ignores_unknown_and_alpha_order(self):
        rows = [
            {"name": "nominal", "order": 0, "requested_alpha": 0.0,
             "effective_post_AEGIS_correction_l2_action": 0.0,
             "combined_risk": [0.1], "terminal_status": "UNSAFE_CONTACT_OR_CAR",
             "exact_safe": False},
            {"name": "small_alpha_large_actual", "order": 1, "requested_alpha": 0.5,
             "effective_post_AEGIS_correction_l2_action": 0.8,
             "combined_risk": [-0.1], "terminal_status": "SAFE_TERMINAL",
             "exact_safe": True},
            {"name": "large_alpha_small_actual", "order": 2, "requested_alpha": 1.0,
             "effective_post_AEGIS_correction_l2_action": 0.6,
             "combined_risk": [-0.05], "terminal_status": "SAFE_TERMINAL",
             "exact_safe": True},
            {"name": "timeout", "order": 3, "requested_alpha": 1.5,
             "effective_post_AEGIS_correction_l2_action": 0.1,
             "combined_risk": [-0.2], "terminal_status": "UNKNOWN_TIMEOUT",
             "exact_safe": False},
        ]
        selected = minimum_actual_intervention(rows)
        self.assertEqual(selected["name"], "large_alpha_small_actual")

    def test_derive_replay_archive(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is unavailable")
        config = load_config(
            ROOT / "configs/vlsa_distal_later_warning_candidate_bank.v1.json"
        )
        qps = []
        for offset in range(5):
            qps.append({
                "context": {
                    "nominal_translational": [float(offset)] * 7,
                    "executed_action": [float(offset + 10)] * 7,
                },
                "z_before": [1.0, 2.0, 3.0],
                "z_after": [1.0, 2.0, 3.0],
            })
        source = {
            "schema_version": config["source_result_schema"],
            "case_id": config["case_id"],
            "source": {"commit": config["source_result_commit"]},
            "result_payload_sha256": config["source_result_payload_sha256"],
            "actions": [
                {"step": step, "action": [float(step)] * 7}
                for step in range(105)
            ],
            "policy_queries": [{
                "query_index": 21, "rng_seed": 42,
                "returned_actions_sha256": "query", "nominal_qp_records": qps,
            }],
        }
        sealed = {
            "case_id": config["case_id"], "task_success": True,
            "perception": {"mvee_center": [0.0, 0.0, 0.0]},
        }
        value = derive_replay_archive(source, sealed, config)
        self.assertEqual(len(value["actions"]), 110)
        self.assertEqual(value["actions"][104]["qp"]["z_after"], [1.0, 2.0, 3.0])
        self.assertEqual(value["actions"][105]["nominal_translational"], [0.0] * 7)
        self.assertEqual(value["actions"][105]["executed"], [10.0] * 7)


if __name__ == "__main__":
    unittest.main()
