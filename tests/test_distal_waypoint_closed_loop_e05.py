from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from main.multilink_ellipsoid.waypoint_candidate_closed_loop import (
    WAYPOINT_CONFIG_SCHEMA,
    load_waypoint_config,
    select_waypoint_candidate,
    waypoint_chunks,
)


ROOT = Path(__file__).resolve().parents[1]
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class DistalWaypointClosedLoopE05Tests(unittest.TestCase):
    def test_checked_in_preregistration_loads(self) -> None:
        config = load_waypoint_config(
            ROOT / "configs/vlsa_distal_waypoint_closed_loop_e05.v1.json"
        )
        self.assertEqual(config["schema_version"], WAYPOINT_CONFIG_SCHEMA)
        self.assertEqual(len(config["constraint_order"]), 8)
        self.assertEqual(config["waypoint_search"]["horizon_actions"], 4)
        self.assertFalse(config["candidate_selector"]["QP_used"])
        self.assertFalse(config["candidate_selector"]["learned_model_used"])

    @unittest.skipUnless(HAS_NUMPY, "NumPy is an allocation dependency")
    def test_waypoint_library_counts_and_preserves_nontranslation(self) -> None:
        import numpy as np

        config = load_waypoint_config(
            ROOT / "configs/vlsa_distal_waypoint_closed_loop_e05.v1.json"
        )
        for horizon, expected in ((1, 28), (2, 97), (3, 124), (4, 124)):
            nominal = np.zeros((horizon, 7), dtype=np.float64)
            nominal[:, :3] = [0.123, 0.234, -0.345]
            nominal[:, 3:6] = [0.01, -0.02, 0.03]
            nominal[:, 6] = np.linspace(0.8, 1.0, horizon)
            candidates = waypoint_chunks(nominal, config)
            self.assertEqual(len(candidates), expected)
            self.assertTrue(all(
                np.array_equal(item["chunk"][:, 3:], nominal[:, 3:])
                for item in candidates
            ))
            self.assertTrue(all(
                np.max(np.abs(item["chunk"][:, :3])) <= 1.0
                for item in candidates
            ))

    def test_selector_prioritizes_task_then_reference_then_correction(self) -> None:
        records = [
            {
                "source": "close",
                "chunk": [[0.0] * 7],
                "chunk_correction_l2": 0.1,
                "minimum_all_eight_margin_m": [0.01] * 8,
                "raw_safe": True,
                "task_success_in_chunk": False,
                "terminal_eef_reference_error_m": 0.001,
            },
            {
                "source": "task",
                "chunk": [[0.1] * 7],
                "chunk_correction_l2": 1.0,
                "minimum_all_eight_margin_m": [0.01] * 8,
                "raw_safe": True,
                "task_success_in_chunk": True,
                "terminal_eef_reference_error_m": 0.1,
            },
        ]
        selected = select_waypoint_candidate(records, minimum_margin_m=1.0e-6)
        self.assertTrue(selected["valid"])
        self.assertEqual(selected["selected_source"], "task")
        self.assertEqual(selected["eligible_candidate_count"], 2)

    def test_selector_rejects_proxy_or_raw_unsafe(self) -> None:
        records = [
            {
                "source": "proxy_unsafe",
                "chunk": [[0.0] * 7],
                "chunk_correction_l2": 0.0,
                "minimum_all_eight_margin_m": [0.0] * 8,
                "raw_safe": True,
                "task_success_in_chunk": False,
                "terminal_eef_reference_error_m": 0.0,
            },
            {
                "source": "raw_unsafe",
                "chunk": [[0.0] * 7],
                "chunk_correction_l2": 0.0,
                "minimum_all_eight_margin_m": [0.01] * 8,
                "raw_safe": False,
                "task_success_in_chunk": False,
                "terminal_eef_reference_error_m": 0.0,
            },
        ]
        selected = select_waypoint_candidate(records, minimum_margin_m=1.0e-6)
        self.assertFalse(selected["valid"])
        self.assertEqual(selected["eligible_candidate_count"], 0)

    def test_harness_is_receding_exact_and_not_qp_or_learned(self) -> None:
        source = (
            ROOT / "scripts/evaluate_distal_waypoint_closed_loop_e05.py"
        ).read_text(encoding="utf-8")
        self.assertIn("for step in range(len(actions))", source)
        self.assertIn("waypoint_chunks(nominal_chunk, config)", source)
        self.assertIn("select_waypoint_candidate", source)
        self.assertIn("fresh_exact_waypoint_verification", source)
        self.assertIn("executed_next_state_matches_exact_clone", source)
        self.assertNotIn("solve_affine_certificate_qp", source)
        self.assertNotIn("WebsocketClientPolicy", source)


if __name__ == "__main__":
    unittest.main()
