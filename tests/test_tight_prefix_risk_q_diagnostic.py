from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from main.multilink_ellipsoid.compact_safety_coordinate_q import (
    safety_coordinate_feature,
)
from main.multilink_ellipsoid.tight_prefix_risk_q_diagnostic import (
    DIAGNOSTIC_ROWS, INPUT_DIMENSION, MODEL_ROWS, PRIMARY_ROWS, load_config,
    row_future_risks, scientific_view,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_tight_prefix_risk_q_diagnostic.v1.json"


def _exact(row_count=10):
    rows = []
    for index in range(row_count):
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
                1.0 + 0.1 * index for index in range(row_count)
            ],
        },
        "source_nominal_five_action_chunk": [[0.0] * 7 for _ in range(5)],
    }


class TightPrefixRiskQDiagnosticTest(unittest.TestCase):
    def test_protocol_keeps_model_small_and_claims_separate(self):
        config = load_config(CONFIG)
        self.assertEqual(config["feature"]["input_dimension"], INPUT_DIMENSION)
        self.assertEqual(tuple(config["feature"]["model_rows"]), MODEL_ROWS)
        self.assertEqual(tuple(config["primary_rows"]), PRIMARY_ROWS)
        self.assertEqual(tuple(config["diagnostic_rows"]), DIAGNOSTIC_ROWS)
        self.assertEqual(config["model"]["weight_decay"], 0.0001)
        self.assertEqual(config["dataset"]["test_status"],
                         "already_opened_diagnostic")
        self.assertIn("action_correction", config["forbidden"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy unavailable")
    def test_existing_seven_dimensional_feature_accepts_ten_tight_rows(self):
        candidate = {
            "source_executed_actions": [
                [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(5)
            ],
        }
        feature = safety_coordinate_feature(
            _exact(), candidate, 9, translation_scale=0.05,
            model_rows=MODEL_ROWS,
        )
        self.assertEqual(len(feature), INPUT_DIMENSION)
        self.assertAlmostEqual(feature[0], 1.9)

    def test_scientific_view_excludes_only_run_specific_fields(self):
        result = {
            "source": {"commit": "a"}, "allocation": {"job": "1"},
            "result_payload_sha256": "hash", "model": {"value": 1},
        }
        self.assertEqual(scientific_view(result), {"model": {"value": 1}})

    def test_row_target_is_worst_negative_slack_over_prefix(self):
        candidate = {"exact_group_target": {"trace": [
            {"row_normalized_radial_slack": [0.2, -0.1]},
            {"row_normalized_radial_slack": [-0.3, 0.4]},
        ]}}
        self.assertEqual(row_future_risks(candidate, (0, 1)), [0.3, 0.1])


if __name__ == "__main__":
    unittest.main()
