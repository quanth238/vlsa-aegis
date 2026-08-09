from __future__ import annotations

from pathlib import Path
import unittest

from main.multilink_ellipsoid.exact_candidate_continue import (
    CONTINUE_CONFIG_SCHEMA,
    load_continue_config,
)


class DistalExactCandidateContinueE05Tests(unittest.TestCase):
    def test_checked_in_config_loads(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_continue_config(
            root / "configs" / "vlsa_distal_exact_candidate_continue_e05.v1.json"
        )
        self.assertEqual(config["schema_version"], CONTINUE_CONFIG_SCHEMA)
        self.assertEqual(
            config["primary_case"]["unsafe_continuation_start_step"], 187
        )
        self.assertFalse(config["continuation"]["stop_on_raw_contact"])
        self.assertFalse(config["continuation"]["candidate_search"])

    def test_harness_replays_prefix_then_bypasses_filter(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (
            root / "scripts" / "evaluate_distal_exact_candidate_continue_e05.py"
        ).read_text(encoding="utf-8")
        self.assertIn("for record in accepted_prefix", source)
        self.assertIn("for step in range(prefix_count, expected_count)", source)
        self.assertIn('"safety_filter_bypassed": True', source)
        self.assertIn('"unsafe_continuation": True', source)
        self.assertIn('"candidate_search_used": False', source)
        self.assertNotIn("grid_actions(", source)
        self.assertNotIn("solve_affine_certificate_qp", source)


if __name__ == "__main__":
    unittest.main()
