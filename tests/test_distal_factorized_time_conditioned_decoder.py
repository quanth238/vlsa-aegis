import json
from pathlib import Path
import unittest

import numpy as np

from main.multilink_ellipsoid.factorized_time_conditioned_decoder import (
    _build_time_model, eligible_state_support, load_time_decoder_config,
    time_decoder_decision, time_encoding,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_factorized_time_conditioned_decoder_moka10.v1.json"


class TimeConditionedDecoderTests(unittest.TestCase):
    def test_config_freezes_prediction_only_gate(self):
        config = load_time_decoder_config(CONFIG)
        self.assertTrue(config["architecture"]["shared_weights_across_all_51_substeps"])
        self.assertTrue(config["forbidden_actions"]["QP"])
        self.assertTrue(config["forbidden_actions"]["candidate_region_expansion"])

    def test_time_encoding_is_registered_and_finite(self):
        encoded = time_encoding(np.arange(51))
        self.assertEqual(encoded.shape, (51, 9))
        self.assertTrue(np.all(np.isfinite(encoded)))
        self.assertEqual(encoded[0, 0], 0.0)
        self.assertEqual(encoded[-1, 0], 1.0)

    def test_model_enforces_exact_initial_residual(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable in local control-plane runtime")
        config = json.loads(CONFIG.read_text())
        model = _build_time_model(12, config["architecture"])
        output = model(torch.zeros((3, 12), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (3, 51, 7))
        self.assertTrue(torch.equal(output[:, 0], torch.zeros_like(output[:, 0])))
        sampled = model(
            torch.ones((3, 12), dtype=torch.float32),
            np.asarray([1, 17, 49, 50], dtype=np.int64),
        )
        self.assertEqual(tuple(sampled.shape), (3, 4, 7))
        sampled.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_support_excludes_states_without_any_exact_safe_candidate(self):
        arrays = {
            "split": np.asarray(["test"] * 4, dtype=object),
            "source_code": np.asarray([2] * 4, dtype=np.int8),
            "state_index": np.asarray([0, 0, 1, 1], dtype=np.int64),
        }
        exact = np.repeat(np.asarray([[0.1], [0.2], [-0.1], [-0.2]]), 7, axis=1)
        predicted = np.repeat(np.asarray([[0.1], [-0.1], [0.1], [0.1]]), 7, axis=1)
        support = eligible_state_support(
            exact_margin=exact, predicted_margin=predicted,
            arrays=arrays, split_name="test",
        )
        self.assertEqual(support["eligible_state_indexes"], [0])
        self.assertEqual(support["supported_state_indexes"], [0])
        self.assertTrue(support["all_eligible_states_supported"])

    def test_decision_never_authorizes_qp(self):
        config = json.loads(CONFIG.read_text())
        safety = {
            "false_safe_action_count": 0, "state_safe_support_count": 13,
            "exact_safe_action_recall": 1.0, "near_boundary_RMSE_m": 0.001,
        }
        metrics = {
            "validation_safety": dict(safety), "test_safety": dict(safety),
            "joint_sensitivity": {"mean_cosine": 0.9},
            "margin_sensitivity": {"mean_cosine": 0.9},
        }
        source_metrics = {
            "validation_safety": dict(safety),
            "test_safety": {**safety, "state_safe_support_count": 12},
        }
        source_temporal = {
            "validation": {"terminal_joint_RMSE_rad": 0.02},
            "test": {"terminal_joint_RMSE_rad": 0.02},
        }
        temporal = {
            "validation": {"terminal_joint_RMSE_rad": 0.01},
            "test": {"terminal_joint_RMSE_rad": 0.01},
        }
        support = {"all_eligible_states_supported": True, "supported_state_count": 13}
        decision = time_decoder_decision(
            source_metrics=source_metrics, source_temporal=source_temporal,
            experimental_metrics=metrics, experimental_temporal=temporal,
            validation_support=support, test_support=support, config=config,
        )
        self.assertTrue(decision["time_conditioned_mechanism_GO"])
        self.assertFalse(decision["QP_or_closed_loop_authorized"])


if __name__ == "__main__":
    unittest.main()
