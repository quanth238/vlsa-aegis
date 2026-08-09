from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from main.multilink_ellipsoid.oracle_affine_closed_loop import (
    ORACLE_AFFINE_CLOSED_LOOP_CONFIG_SCHEMA,
    ORACLE_AFFINE_CLOSED_LOOP_CONFIG_SCHEMA_V2,
    load_oracle_affine_closed_loop_config,
    summarize_exact_chunk_all_eight,
)


class DistalAffineOracleClosedLoopE05Tests(unittest.TestCase):
    def test_checked_in_preregistration_loads(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_oracle_affine_closed_loop_config(
            root / "configs" / "vlsa_distal_affine_oracle_closed_loop_e05.v1.json"
        )
        self.assertEqual(
            config["schema_version"], ORACLE_AFFINE_CLOSED_LOOP_CONFIG_SCHEMA
        )
        self.assertEqual(config["constraint_order"][0], "L5_part_0")
        self.assertEqual(len(config["constraint_order"]), 7)
        self.assertTrue(
            config["execution"]["require_released_AEGIS_EE_proxy_nonnegative"]
        )
        self.assertFalse(config["decision_gate"]["neural_training_authorized"])

    def test_checked_in_eight_constraint_repair_loads(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_oracle_affine_closed_loop_config(
            root / "configs" / "vlsa_distal_affine_oracle_closed_loop_e05.v2.json"
        )
        self.assertEqual(
            config["schema_version"], ORACLE_AFFINE_CLOSED_LOOP_CONFIG_SCHEMA_V2
        )
        self.assertEqual(len(config["constraint_order"]), 8)
        self.assertEqual(config["constraint_order"][-1], "released_AEGIS_EE_proxy")
        self.assertEqual(config["affine_certificate"]["qp_constraint_count"], 8)

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") is not None, "NumPy runtime dependency"
    )
    def test_exact_chunk_summary_keeps_eighth_EE_row(self) -> None:
        transition_a = {
            "minimum_substep_clearance_m": [0.01] * 7 + [0.02],
            "raw_protected_contact_count": 0,
            "maximum_within_step_obstacle_l1_displacement_m": 0.0,
            "next_state_sha256": "a" * 64,
            "env_step_wall_seconds": 0.1,
        }
        transition_b = {
            "minimum_substep_clearance_m": [0.005] * 7 + [-0.001],
            "raw_protected_contact_count": 0,
            "maximum_within_step_obstacle_l1_displacement_m": 0.0,
            "next_state_sha256": "b" * 64,
            "env_step_wall_seconds": 0.2,
        }
        summary = summarize_exact_chunk_all_eight(
            {"transitions": [transition_a, transition_b]},
            maximum_obstacle_displacement_m=1.0e-4,
        )
        self.assertTrue(summary["D_opt_seven_distal_safe"])
        self.assertFalse(summary["released_AEGIS_EE_proxy_safe"])
        self.assertFalse(summary["safe_for_execution"])
        self.assertEqual(summary["first_transition_next_state_sha256"], "a" * 64)
        self.assertAlmostEqual(summary["env_step_wall_seconds"], 0.3)

    def test_harness_is_receding_and_video_backed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (
            root / "scripts" / "evaluate_distal_affine_oracle_closed_loop_e05.py"
        ).read_text(encoding="utf-8")
        self.assertIn("for step in range(len(actions))", source)
        self.assertIn("grid_actions(first[:3], source_config)", source)
        self.assertIn("fit_candidate_conditioned_affine_certificate", source)
        self.assertIn("solve_affine_certificate_qp", source)
        self.assertIn("executed_next_state_matches_exact_clone", source)
        self.assertIn("video_writer.append_data", source)


if __name__ == "__main__":
    unittest.main()
