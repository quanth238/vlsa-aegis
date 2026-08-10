"""Structural tests for the omitted-variable insufficiency audit."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from main.multilink_ellipsoid.omitted_variable_audit import (
    CONFIG_SCHEMA, analyze_records, axis_rotation, complete_model_gate,
    final_decision, intervention_specs, load_config, select_candidate_indexes,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "vlsa_distal_omitted_variable_audit_moka10.v1.json"


class OmittedVariableAuditTest(unittest.TestCase):
    def test_registered_protocol_has_four_independent_families(self) -> None:
        config = load_config(CONFIG)
        self.assertEqual(config["schema_version"], CONFIG_SCHEMA)
        specs = intervention_specs(config)
        self.assertEqual(len(specs), 19)
        self.assertEqual(
            {item["family"] for item in specs},
            {
                "baseline", "goal_orientation", "rotation_action",
                "gripper_command", "controller_memory",
            },
        )
        self.assertTrue(config["forbidden_actions"]["QP"])

    def test_candidate_selection_retains_nearest_safe_and_unsafe(self) -> None:
        values = np.linspace(-0.0124, 0.0124, 125)
        state = {"candidates": [
            {"rollout_minimum_ellipsoid_margin_m": [float(value)] * 7}
            for value in values
        ]}
        selected = select_candidate_indexes(state)
        self.assertEqual(len(selected), 2)
        self.assertEqual(
            {item["selection_source"] for item in selected},
            {"closest_safe", "closest_unsafe"},
        )

    def test_identical_old_hash_with_safe_and_unsafe_is_insufficient(self) -> None:
        base = {
            "state_index": 4, "candidate_index": 7, "old56_sha256": "a" * 64,
            "baseline_old56_sha256": "a" * 64,
            "old56_hash_equal_to_baseline": True,
            "effective_intervention": True, "rollout_executed": True,
            "variant_family": "baseline", "variant_name": "baseline",
            "proxy_safe": True, "raw_contact_free": True,
            "rollout_minimum_ellipsoid_margin_m": [0.001] * 7,
            "next_state_sha256_per_action": ["b" * 64, "c" * 64],
        }
        changed = dict(base)
        changed.update({
            "variant_family": "rotation_action",
            "variant_name": "first_rotation_0_+1", "proxy_safe": False,
            "rollout_minimum_ellipsoid_margin_m": [-0.001] + [0.001] * 6,
            "next_state_sha256_per_action": ["d" * 64, "e" * 64],
        })
        audit = analyze_records([base, changed])
        self.assertTrue(audit["old56_provably_insufficient"])
        self.assertEqual(audit["identical_old56_proxy_class_conflict_count"], 1)
        complete = complete_model_gate({
            "arms": {"completeOSC": {"metrics": {"test": {
                "proxy_false_safe_action_count": 2,
                "state_safe_support_count": 15,
            }}}}
        })
        decision = final_decision(audit, complete)
        self.assertEqual(
            decision["conclusion"],
            "old56_insufficient_but_complete_plain_MLP_still_fails",
        )

    def test_axis_rotations_are_proper(self) -> None:
        for axis in range(3):
            matrix = axis_rotation(axis, 15.0)
            np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-15)
            self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0)


if __name__ == "__main__":
    unittest.main()
