from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from main.multilink_ellipsoid.exact_candidate_closed_loop import (
    EXACT_CANDIDATE_CONFIG_SCHEMA,
    load_exact_candidate_config,
    select_exact_safe_candidate,
)


class DistalExactCandidateClosedLoopE05Tests(unittest.TestCase):
    def test_checked_in_preregistration_loads(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_exact_candidate_config(
            root / "configs" / "vlsa_distal_exact_candidate_closed_loop_e05.v1.json"
        )
        self.assertEqual(config["schema_version"], EXACT_CANDIDATE_CONFIG_SCHEMA)
        self.assertEqual(len(config["constraint_order"]), 8)
        self.assertEqual(config["sampling"]["expected_grid_action_count"], 512)
        self.assertFalse(config["candidate_selector"]["affine_QP_used"])

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") is not None, "NumPy runtime dependency"
    )
    def test_selector_chooses_closest_safe_then_grid_index(self) -> None:
        xyz = [[0.3, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]
        margins = [[0.01] * 8, [0.02] * 8, [0.02] * 8]
        selected = select_exact_safe_candidate(
            xyz, margins, [True, True, True], [7, 9, 3], [0.0, 0.0, 0.0],
            minimum_margin_m=1.0e-6,
        )
        self.assertTrue(selected["valid"])
        self.assertEqual(selected["selected_grid_index"], 3)
        self.assertEqual(selected["eligible_candidate_count"], 3)

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") is not None, "NumPy runtime dependency"
    )
    def test_selector_rejects_margin_and_raw_unsafe_candidates(self) -> None:
        selected = select_exact_safe_candidate(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [[0.0] * 8, [0.01] * 8],
            [True, False],
            [0, 1],
            [0.0, 0.0, 0.0],
            minimum_margin_m=1.0e-6,
        )
        self.assertFalse(selected["valid"])
        self.assertEqual(selected["eligible_candidate_count"], 0)

    def test_harness_is_receding_exact_and_has_no_qp(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (
            root / "scripts" / "evaluate_distal_exact_candidate_closed_loop_e05.py"
        ).read_text(encoding="utf-8")
        self.assertIn("for step in range(len(actions))", source)
        self.assertIn("grid_actions(first[:3], source_config)", source)
        self.assertIn("select_exact_safe_candidate", source)
        self.assertIn("fresh_exact_two_step_verification", source)
        self.assertIn("executed_next_state_matches_exact_clone", source)
        self.assertNotIn("solve_affine_certificate_qp", source)
        self.assertNotIn("fit_candidate_conditioned_affine_certificate", source)


if __name__ == "__main__":
    unittest.main()
