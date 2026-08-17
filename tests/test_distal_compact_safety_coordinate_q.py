from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path

from main.multilink_ellipsoid.compact_safety_coordinate_q import (
    INPUT_DIMENSION, MODEL_ROWS, candidate_records, evaluate_selector,
    load_config, safety_coordinate_feature,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_compact_safety_coordinate_q.v1.json"


def _exact():
    rows = []
    for index in range(9):
        rows.append({
            "center_m": [0.3 + 0.01 * index, 0.0, 0.0],
            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]],
            "semiaxes_m": [0.1, 0.1, 0.1],
            "body_name": f"row-{index}",
            "geom_name": f"geom-{index}",
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
                1.0 + 0.1 * index for index in range(9)
            ],
        },
        "source_nominal_five_action_chunk": [[0.0] * 7 for _ in range(5)],
    }


def _row_sample(candidate, order, row, risk, correction):
    return {
        "state_id": "state", "split": "test",
        "candidate_name": candidate, "candidate_order": order,
        "applied_correction_l2_action": correction,
        "known_outcome": True, "physical_veto": False,
        "row_index": row, "risk": risk,
    }


class CompactSafetyCoordinateQTest(unittest.TestCase):
    def test_contract_is_inference_only_and_seven_dimensional(self):
        config = load_config(CONFIG)
        self.assertFalse(config["simulation_rollouts"])
        self.assertEqual(config["feature"]["input_dimension"], INPUT_DIMENSION)
        self.assertIn("exact_verifier_at_inference", config["forbidden"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy unavailable")
    def test_feature_uses_current_slack_and_effective_five_step_projection(self):
        candidate = {
            "source_executed_actions": [[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                                        for _ in range(5)],
        }
        feature = safety_coordinate_feature(
            _exact(), candidate, 0, translation_scale=0.05,
        )
        self.assertEqual(len(feature), 7)
        self.assertAlmostEqual(feature[0], 1.0)
        self.assertAlmostEqual(feature[1], 0.0)
        for value in feature[2:]:
            self.assertAlmostEqual(value, 0.1)

    def test_selector_scores_rows_once_and_keeps_l6_as_audit(self):
        samples = []
        predictions = []
        for candidate, order, risk, estimate, correction in (
            ("nominal", 0, 0.2, 0.1, 0.0),
            ("corrected", 1, -0.1, -0.2, 1.0),
        ):
            for row in MODEL_ROWS:
                row_risk = risk
                if candidate == "corrected" and row == 5:
                    row_risk = 0.3
                samples.append(_row_sample(
                    candidate, order, row, row_risk, correction,
                ))
                predictions.append([estimate])
        records = candidate_records(
            samples, predictions, primary_rows=(0, 1, 2, 3, 4),
            physical_rows=(1, 2, 3, 4, 5, 6),
        )
        result = evaluate_selector(
            records, rule="minimum_predicted_primary_risk",
        )
        self.assertEqual(result["known_safe_primary_selection_count"], 1)
        self.assertEqual(result["known_all_physical_unsafe_selection_count"], 1)
        self.assertEqual(result["ignored_L6_failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
