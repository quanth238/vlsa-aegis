from __future__ import annotations

import unittest
from pathlib import Path

from main.multilink_ellipsoid.whole_body_q_only_diagnostic import (
    SHARED_INPUT_DIMENSION, load_config, shared_constraint_feature,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_whole_body_q_only_diagnostic.v1.json"


class WholeBodyQOnlyDiagnosticTest(unittest.TestCase):
    def test_contract_keeps_test_sealed_and_weight_decay_fixed(self):
        config = load_config(CONFIG)
        self.assertFalse(config["dataset"]["test_access"])
        self.assertEqual(config["model"]["weight_decay"], 0.0001)
        self.assertEqual(
            config["arms"]["shared_constraint_135D"]["input_dimension"],
            SHARED_INPUT_DIMENSION,
        )

    def test_shared_feature_is_constraint_conditioned_and_causal(self):
        rows = []
        for index in range(9):
            rows.append({
                "center_m": [0.1 * index, 0.0, 0.0],
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "semiaxes_m": [0.1, 0.1, 0.1],
            })
        exact = {
            "initial_exact_robot_rows": rows,
            "initial_compiled_obstacle_boxes": [{
                "center_m": [0.0, 0.0, 0.0],
                "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "half_extents_m": [0.1, 0.1, 0.1],
            }],
            "exact_group_target": {
                "initial_row_normalized_radial_slack": [0.2] * 9,
            },
            "source_nominal_five_action_chunk": [[0.0] * 7 for _ in range(5)],
            "physical_context": {
                "arm_joint_position_rad": [0.0] * 7,
                "arm_joint_velocity_rad_s": [0.0] * 7,
                "eef_position_m": [0.0, 0.0, 0.0],
                "controller_snapshot": {"goal_pos": [0.0, 0.0, 0.0]},
            },
        }
        candidate = {"source_executed_actions": [[0.1] * 7 for _ in range(5)]}
        feature = shared_constraint_feature(exact, candidate, 2)
        self.assertEqual(len(feature), SHARED_INPUT_DIMENSION)
        self.assertEqual(feature[-6:], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
