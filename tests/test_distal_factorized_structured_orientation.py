"""Structural tests for the frozen structured-orientation experiment."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest

from main.multilink_ellipsoid.factorized_one_sided_geometry import (
    train_one_sided_geometry_ensemble,
)
from main.multilink_ellipsoid.factorized_structured_orientation import (
    _root_rotation_indexes, load_structured_config, structured_decision,
)
from main.multilink_ellipsoid.factorized_time_conditioned_decoder import (
    train_time_conditioned_ensemble,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_factorized_structured_orientation_moka10.v1.json"


class StructuredOrientationTest(unittest.TestCase):
    def test_registered_config_is_frozen(self) -> None:
        config = load_structured_config(CONFIG)
        self.assertEqual(
            config["structured_representation"]["structured_input_dimension"],
            1842,
        )
        self.assertTrue(config["forbidden_actions"]["closed_loop"])

    def test_invalid_config_is_rejected(self) -> None:
        payload = json.loads(CONFIG.read_text())
        payload["structured_representation"]["appended_dimension"] = 9
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                load_structured_config(path)

    def test_root_rotation_indexes_are_column_major_6d(self) -> None:
        names = ["unused"]
        for row in range(3):
            for column in range(3):
                names.append(
                    "semantic_state.obstacle_root_orientation_world_from_body"
                    f"[{row}][{column}].value"
                )
        selected = _root_rotation_indexes(names)
        self.assertEqual(
            [names[index] for index in selected],
            [
                "semantic_state.obstacle_root_orientation_world_from_body[0][0].value",
                "semantic_state.obstacle_root_orientation_world_from_body[1][0].value",
                "semantic_state.obstacle_root_orientation_world_from_body[2][0].value",
                "semantic_state.obstacle_root_orientation_world_from_body[0][1].value",
                "semantic_state.obstacle_root_orientation_world_from_body[1][1].value",
                "semantic_state.obstacle_root_orientation_world_from_body[2][1].value",
            ],
        )

    def test_training_overrides_are_opt_in(self) -> None:
        flat = inspect.signature(train_one_sided_geometry_ensemble)
        timed = inspect.signature(train_time_conditioned_ensemble)
        self.assertIsNone(flat.parameters["normalization_override"].default)
        self.assertIsNone(timed.parameters["normalization_override"].default)

    def test_decision_requires_matched_flat_gate(self) -> None:
        config = load_structured_config(CONFIG)
        representation = {
            "maximum_absolute_normalized_value_by_split": {
                "train": 8.0, "validation": 10.0, "test": 11.0,
            }
        }
        safety = {
            "false_safe_action_count": 0,
            "exact_safe_action_recall": 0.94,
            "near_boundary_RMSE_m": 0.002,
        }
        metrics = {"test_safety": safety}
        temporal = {
            "validation": {"terminal_joint_RMSE_rad": 0.02},
            "test": {"terminal_joint_RMSE_rad": 0.018},
        }
        support = {"supported_state_count": 12}
        decision = structured_decision(
            representation=representation, flat_metrics=metrics,
            time_metrics=metrics, flat_temporal=temporal,
            time_temporal=temporal, flat_support=support,
            time_support={"supported_state_count": 7},
            source_time_metrics={
                "test_safety": {"near_boundary_RMSE_m": 0.0157}
            },
            source_time_temporal={
                "test": {"terminal_joint_RMSE_rad": 0.022889}
            }, source_time_support_count=6, config=config,
        )
        self.assertTrue(decision["flat_model_candidate_GO"])
        self.assertTrue(decision["time_decoder_candidate_GO"])
        self.assertFalse(decision["QP_or_closed_loop_authorized"])


if __name__ == "__main__":
    unittest.main()
