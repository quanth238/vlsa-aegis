"""Structural tests for the complete-OSC paired margin diagnostic."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
import unittest

import numpy as np

from main.multilink_ellipsoid.complete_osc_margin import (
    CONFIG_SCHEMA, action_row_features, flatten_numeric_tree, load_config,
    prediction_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "vlsa_distal_complete_osc_margin_moka10.v1.json"


class CompleteOscMarginTest(unittest.TestCase):
    def test_registered_config(self) -> None:
        config = load_config(CONFIG)
        self.assertEqual(config["schema_version"], CONFIG_SCHEMA)
        self.assertEqual(config["immutable_source"]["expected_pair_count"], 10625)
        self.assertTrue(config["forbidden_actions"]["QP"])
        self.assertEqual(config["prediction_gate"]["test_proxy_false_safe_action_count"], 0)

    def test_rejects_changed_training_contract(self) -> None:
        config = load_config(CONFIG)
        config.pop("config_file_sha256")
        config.pop("config_payload_sha256")
        config["training"]["epochs"] = 501
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            import json
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training differs"):
                load_config(path)

    def test_numeric_snapshot_flattening_is_named_and_stable(self) -> None:
        value = {
            "controller": {"goal": np.asarray([1.0, 2.0]), "update": True},
            "sim": [3, None],
        }
        names, vector = flatten_numeric_tree(value)
        names_repeat, vector_repeat = flatten_numeric_tree(copy.deepcopy(value))
        self.assertEqual(names, names_repeat)
        np.testing.assert_array_equal(vector, vector_repeat)
        self.assertIn("controller.goal[0]", names)
        self.assertIn("sim[1].present", names)

    def test_action_row_preserves_full_two_action_command(self) -> None:
        state = np.asarray([1.0, 2.0, 3.0])
        actions = np.arange(14, dtype=np.float64).reshape(2, 7)
        feature = action_row_features(state, actions, 5)
        np.testing.assert_array_equal(feature[:3], state)
        np.testing.assert_array_equal(feature[3:17], actions.reshape(-1))
        np.testing.assert_array_equal(feature[-7:], [0, 0, 0, 0, 0, 1, 0])

    def test_prediction_gate_counts_action_level_false_safe(self) -> None:
        config = load_config(CONFIG)
        rows = 15 * 125 * 7
        arrays = {
            "margin_m": np.full(rows, 0.01),
            "split": np.asarray(["test"] * rows, dtype=object),
            "state_index": np.repeat(np.arange(15), 125 * 7),
            "action_index": np.tile(np.repeat(np.arange(125), 7), 15),
            "constraint_index": np.tile(np.arange(7), 15 * 125),
        }
        prediction = np.full(rows, 0.01)
        arrays["margin_m"][0] = -0.001
        metrics = prediction_metrics(arrays, prediction, config)
        self.assertEqual(metrics["test_proxy_false_safe_action_count"], 1)
        self.assertEqual(metrics["test_state_safe_support_count"], 15)
        self.assertFalse(metrics["prediction_gate_pass"])


if __name__ == "__main__":
    unittest.main()
