"""Structural tests for the paired input-only ablation."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from main.multilink_ellipsoid.complete_osc_margin import training_arrays
from main.multilink_ellipsoid.matched_input_ablation import (
    CONFIG_SCHEMA, load_config, matched_decision, old56_training_arrays,
    paired_row_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "vlsa_distal_matched_input_ablation_moka10.v1.json"


def _transform(center, size):
    return {
        "center_m": list(center),
        "rotation_world_from_local": np.eye(3).tolist(),
        "semiaxes_or_half_size_m": list(size),
    }


def _dataset() -> dict:
    robot = [_transform([0.0, 0.1 * row, 0.0], [0.1, 0.1, 0.1]) for row in range(7)]
    obstacle = [_transform([1.0, 0.0, 0.0], [0.2, 0.2, 0.2])]
    current = []
    for row in range(7):
        center = np.asarray([0.0, 0.1 * row, 0.0])
        displacement = np.asarray([1.0, 0.0, 0.0]) - center
        direction = displacement / np.linalg.norm(displacement)
        current.append(float(
            np.linalg.norm(displacement) - 0.1
            - np.sum(0.2 * np.abs(direction))
        ))
    candidates = []
    for index, x in enumerate((-0.5, 0.5)):
        actions = np.zeros((2, 7), dtype=np.float64)
        actions[0, :3] = [x, -x, x]
        candidates.append({
            "candidate_index": index,
            "full_two_action_commands": actions.tolist(),
            "rollout_minimum_ellipsoid_margin_m": (
                np.asarray(current) - 0.001 * index
            ).tolist(),
        })
    state = {
        "state_index": 3, "split": "train",
        "complete_input_vector": [1.0, 2.0, 3.0],
        "semantic_state": {
            "q_rad": [0.0] * 7, "qdot_rad_s": [0.0] * 7,
            "goal_EE_position_m": [0.3, 0.2, 0.1],
            "seven_robot_ellipsoid_transforms": robot,
            "exact_obstacle_primitive_transforms": obstacle,
        },
        "current_ellipsoid_margin_m": current,
        "nominal_first_action": [0.0] * 7,
        "nominal_second_action": [0.0] * 7,
        "candidates": candidates,
    }
    return {
        "state_records": [state],
        "summary": {"pair_count": 2},
    }


class MatchedInputAblationTest(unittest.TestCase):
    def test_registered_config(self) -> None:
        config = load_config(CONFIG)
        self.assertEqual(config["schema_version"], CONFIG_SCHEMA)
        self.assertEqual(config["paired_arms"]["old56"]["dimension"], 56)
        self.assertTrue(config["forbidden_actions"]["additional_simulation"])

    def test_old56_projection_pairs_exactly_with_complete_rows(self) -> None:
        dataset = _dataset()
        old, reconstruction = old56_training_arrays(dataset)
        complete = training_arrays(dataset)
        self.assertEqual(old["features"].shape, (14, 56))
        self.assertEqual(complete["features"].shape, (14, 24))
        self.assertLessEqual(
            reconstruction["maximum_fresh_clearance_reconstruction_error_m"],
            1.0e-12,
        )
        receipt = paired_row_receipt(old, complete)
        self.assertTrue(receipt["exact_row_identity_and_label_match"])
        self.assertEqual(receipt["row_count"], 14)

    def test_interpretation_is_frozen(self) -> None:
        def arm(train_fit, test_pass):
            return {
                "train": {"fit_gate_pass": train_fit},
                "test": {"full_prediction_gate_pass": test_pass},
            }

        decision = matched_decision({
            "old56": arm(True, False), "completeOSC": arm(True, False),
        })
        self.assertEqual(
            decision["conclusion"], "insufficient_state_or_setup_coverage"
        )
        decision = matched_decision({
            "old56": arm(False, False), "completeOSC": arm(False, False),
        })
        self.assertEqual(
            decision["conclusion"],
            "output_target_or_plain_MLP_representation_problem",
        )


if __name__ == "__main__":
    unittest.main()
