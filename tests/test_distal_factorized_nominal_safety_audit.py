from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from main.multilink_ellipsoid.factorized_nominal_safety_audit import (
    audit_formula, load_formula_audit_config,
)


ROOT = Path(__file__).resolve().parents[1]


class NominalSafetyFormulaAuditTest(unittest.TestCase):
    def test_registered_config_loads(self) -> None:
        config = load_formula_audit_config(
            ROOT / "configs/vlsa_distal_factorized_nominal_safety_formula_audit_moka10.v1.json"
        )
        self.assertEqual(config["population"]["substep_count"], 51)
        self.assertEqual(config["population"]["constraint_count"], 7)

    def test_invalid_config_is_rejected(self) -> None:
        source = ROOT / "configs/vlsa_distal_factorized_nominal_safety_formula_audit_moka10.v1.json"
        value = json.loads(source.read_text())
        value["population"]["substep_count"] = 50
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_formula_audit_config(path)

    def test_decomposition_identifies_joint_prediction_error(self) -> None:
        shape = (3, 51, 7)
        dynamic = np.full(shape, 0.01, dtype=np.float64)
        dynamic[2, -1, 0] = -0.001
        exact_static = dynamic.copy()
        predicted_static = exact_static.copy()
        predicted_static[2, -1, 0] = 0.002
        exact_q = np.zeros(shape, dtype=np.float64)
        predicted_q = exact_q.copy()
        predicted_q[2, -1, 0] = 0.01
        arrays = {
            "ellipsoid_clearance_m": dynamic,
            "joint_position_rad": exact_q,
            "split": np.asarray(["train", "validation", "test"], dtype=object),
            "source_code": np.asarray([2, 2, 2], dtype=np.int8),
        }
        records = {
            "row_index": np.arange(3, dtype=np.int64),
            "split_code": np.asarray([0, 1, 2], dtype=np.int8),
            "exact_joint_position_rad": exact_q,
            "predicted_joint_position_rad": predicted_q,
            "exact_static_clearance_m": exact_static,
            "predicted_static_clearance_m": predicted_static,
        }
        predictions = {
            "exact_q_static_minimum_margin_m": np.min(exact_static, axis=1),
            "flat_minimum_margin_m": np.min(predicted_static, axis=1),
            "time_minimum_margin_m": np.min(predicted_static, axis=1),
            "flat_joint_position_rad": predicted_q,
            "time_joint_position_rad": predicted_q,
        }
        config = {
            "population": {"random_action_count": 3},
            "decision_gate": {
                "maximum_k0_exact_q_recomposition_absolute_error_m": 1e-6,
                "maximum_test_exact_q_minimum_boundary_RMSE_m": 5e-4,
                "maximum_test_exact_q_false_safe_action_count": 0,
                "minimum_test_exact_q_safe_recall": 0.0,
                "minimum_joint_prediction_to_geometry_RMSE_ratio": 2.0,
                "interpretation_if_pass": "joint",
                "interpretation_if_fail": "geometry",
            },
        }
        result = audit_formula(
            arrays=arrays, records=records,
            structured_predictions=predictions,
            flat_output_shape=[51, 7], time_output_shape=[51, 7],
            config=config,
        )
        self.assertTrue(result["decision"]["joint_prediction_is_dominant"])
        self.assertEqual(
            result["split_metrics"]["test"]["predicted_q_safety_error"]
            ["minimum"]["false_safe_action_count"], 1,
        )
        self.assertEqual(
            result["decomposition_identity_maximum_absolute_error_m"], 0.0
        )
        self.assertFalse(result["formula_receipt"]["QP_or_h_linearization_used"])


if __name__ == "__main__":
    unittest.main()
