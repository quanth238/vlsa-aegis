from pathlib import Path
import unittest

import numpy as np

from main.multilink_ellipsoid.factorized_explicit_execution_jacobian import (
    _build_explicit_model, action_phase_encoding,
    causal_execution_jacobian_mask, load_explicit_jacobian_config,
    nominal_row_mapping,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_factorized_explicit_execution_jacobian_moka10.v1.json"


class ExplicitExecutionJacobianTests(unittest.TestCase):
    def test_config_freezes_prediction_only_gate(self):
        config = load_explicit_jacobian_config(CONFIG)
        self.assertEqual(config["architecture"]["action_dimension"], 14)
        self.assertEqual(
            config["causal_contract"]["second_action_starts_at_substep"], 26,
        )
        self.assertTrue(
            config["forbidden_before_prediction_pass"]
            ["new_unseen_episode_evaluation"]
        )
        self.assertTrue(config["forbidden_before_prediction_pass"]["QP"])

    def test_action_phase_and_causal_mask_match_two_action_rollout(self):
        phase = action_phase_encoding()
        mask = causal_execution_jacobian_mask()
        self.assertEqual(phase.shape, (51, 2))
        self.assertEqual(mask.shape, (51, 7, 14))
        self.assertTrue(np.all(phase[0] == 0.0))
        self.assertTrue(np.all(phase[1:26, 0] == 1.0))
        self.assertTrue(np.all(phase[1:26, 1] == 0.0))
        self.assertTrue(np.all(phase[26:, 0] == 0.0))
        self.assertTrue(np.all(phase[26:, 1] == 1.0))
        self.assertTrue(np.all(mask[0] == 0.0))
        self.assertTrue(np.all(mask[1:26, :, 7:14] == 0.0))
        self.assertTrue(np.all(mask[26:, :, 7:14] == 1.0))

    def test_decoder_enforces_q0_and_causal_jacobian_zeros(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable in local control-plane runtime")
        config = load_explicit_jacobian_config(CONFIG)
        model = _build_explicit_model(12, config["architecture"])
        displacement, jacobian = model(
            torch.zeros((3, 12), dtype=torch.float32)
        )
        self.assertEqual(tuple(displacement.shape), (3, 51, 7))
        self.assertEqual(tuple(jacobian.shape), (3, 51, 7, 14))
        self.assertTrue(torch.equal(
            displacement[:, 0], torch.zeros_like(displacement[:, 0])
        ))
        self.assertTrue(torch.equal(
            jacobian[:, 0], torch.zeros_like(jacobian[:, 0])
        ))
        self.assertTrue(torch.equal(
            jacobian[:, 1:26, :, 7:14],
            torch.zeros_like(jacobian[:, 1:26, :, 7:14]),
        ))
        (displacement.square().mean() + jacobian.square().mean()).backward()
        self.assertTrue(any(
            parameter.grad is not None for parameter in model.parameters()
        ))

    def test_nominal_anchor_is_unique_per_state(self):
        arrays = {
            "state_index": np.asarray([4, 4, 4, 9, 9, 9]),
            "candidate_index": np.asarray([0, 1, 2, 0, 1, 2]),
        }
        mapping = nominal_row_mapping(arrays)
        np.testing.assert_array_equal(mapping["state_index"], [4, 9])
        np.testing.assert_array_equal(mapping["anchor_row_index"], [0, 3])
        np.testing.assert_array_equal(
            mapping["row_to_anchor_ordinal"], [0, 0, 0, 1, 1, 1],
        )

    def test_source_has_no_recursive_joint_integration(self):
        source = (
            ROOT / "main/multilink_ellipsoid/factorized_explicit_execution_jacobian.py"
        ).read_text()
        self.assertNotIn("cumsum", source)
        self.assertIn("nominal_displacement + torch.einsum", source)
        self.assertIn("jacobian * self.causal_jacobian_mask", source)


if __name__ == "__main__":
    unittest.main()
