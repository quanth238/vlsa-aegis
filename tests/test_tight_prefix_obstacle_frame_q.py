from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from main.multilink_ellipsoid.compact_safety_coordinate_q import (
    safety_coordinate_feature,
)
from main.multilink_ellipsoid.tight_prefix_obstacle_frame_q import (
    COMPACT_INPUT_DIMENSION, MODEL_ROWS, OBSTACLE_FRAME_INPUT_DIMENSION,
    constraint_action_frame, load_config, obstacle_frame_feature,
    same_state_pairwise_rank_accuracy,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_tight_prefix_obstacle_frame_q.v1.json"


def _exact():
    rows = []
    for index in range(10):
        rows.append({
            "center_m": [0.3 + 0.01 * index, 0.0, 0.0],
            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]],
            "semiaxes_m": [0.1, 0.1, 0.1],
            "body_name": "row-%d" % index,
            "geom_name": "geom-%d" % index,
            "bound_source": "test",
        })
    return {
        "initial_exact_robot_rows": rows,
        "initial_compiled_obstacle_boxes": [{
            "geom_id": 1, "geom_name": "obstacle", "body_id": 1,
            "body_name": "obstacle", "center_m": [0.0, 0.0, 0.0],
            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]],
            "half_extents_m": [0.1, 0.1, 0.1],
        }],
        "exact_group_target": {
            "initial_row_normalized_radial_slack": [
                1.0 + 0.1 * index for index in range(10)
            ],
        },
        "source_nominal_five_action_chunk": [[0.0] * 7 for _ in range(5)],
    }


class TightPrefixObstacleFrameQTests(unittest.TestCase):
    def test_config_is_matched_and_test_sealed(self):
        config = load_config(CONFIG)
        self.assertEqual(
            config["representations"]["compact_7D"]["input_dimension"],
            COMPACT_INPUT_DIMENSION,
        )
        self.assertEqual(
            config["representations"]["obstacle_frame_17D"][
                "input_dimension"
            ], OBSTACLE_FRAME_INPUT_DIMENSION,
        )
        self.assertFalse(config["dataset"]["test_access"])
        self.assertIn("directional_loss", config["forbidden"])
        self.assertIn("full_episode", config["forbidden"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy unavailable")
    def test_17D_preserves_every_7D_normal_coordinate(self):
        candidate = {"source_executed_actions": [
            [0.2, 0.3, -0.4, 0.1, 0.2, 0.3, -0.7] for _ in range(5)
        ]}
        compact = safety_coordinate_feature(
            _exact(), candidate, 0, translation_scale=0.05,
            model_rows=MODEL_ROWS,
        )
        obstacle = obstacle_frame_feature(
            _exact(), candidate, 0, translation_scale=0.05,
            model_rows=MODEL_ROWS,
        )
        self.assertEqual(len(compact), COMPACT_INPUT_DIMENSION)
        self.assertEqual(len(obstacle), OBSTACLE_FRAME_INPUT_DIMENSION)
        self.assertAlmostEqual(compact[0], obstacle[0])
        self.assertAlmostEqual(compact[1], obstacle[1])
        for step in range(5):
            self.assertAlmostEqual(compact[2 + step], obstacle[2 + 3 * step])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy unavailable")
    def test_tangent_action_is_visible_only_to_17D(self):
        zero = {"source_executed_actions": [[0.0] * 7 for _ in range(5)]}
        tangent = {"source_executed_actions": [
            [0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(5)
        ]}
        compact_zero = safety_coordinate_feature(
            _exact(), zero, 0, translation_scale=0.05,
            model_rows=MODEL_ROWS,
        )
        compact_tangent = safety_coordinate_feature(
            _exact(), tangent, 0, translation_scale=0.05,
            model_rows=MODEL_ROWS,
        )
        obstacle_zero = obstacle_frame_feature(
            _exact(), zero, 0, translation_scale=0.05,
            model_rows=MODEL_ROWS,
        )
        obstacle_tangent = obstacle_frame_feature(
            _exact(), tangent, 0, translation_scale=0.05,
            model_rows=MODEL_ROWS,
        )
        self.assertEqual(compact_zero, compact_tangent)
        self.assertNotEqual(obstacle_zero, obstacle_tangent)
        frame = constraint_action_frame(_exact(), 0)
        self.assertEqual(frame["tangent_box_axis_index"], 1)

    def test_pairwise_rank_is_grouped_by_state(self):
        samples = []
        predictions = []
        for order, (target, estimate) in enumerate(((0.3, 0.1), (0.1, 0.2))):
            for row in (0, 1):
                samples.append({
                    "state_id": "s0", "candidate_name": "c%d" % order,
                    "row_index": row, "risk": target - 0.01 * row,
                })
                predictions.append([estimate - 0.01 * row])
        report = same_state_pairwise_rank_accuracy(
            samples, predictions, rows=(0, 1),
        )
        self.assertEqual(report["pair_count"], 1)
        self.assertEqual(report["correct_pair_count"], 0)
        self.assertEqual(report["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
