from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MokaResponseFieldTests(unittest.TestCase):
    def test_audit_forbids_training_qp_and_execution(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs" / "vlsa_distal_moka_response_field_e05_audit.v1.json").read_text()
        )
        self.assertEqual(
            config["forbidden"],
            [
                "model_training",
                "model_parameter_change",
                "larger_correction",
                "qp",
                "corrected_execution",
                "closed_loop",
            ],
        )
        source = (root / "scripts" / "audit_distal_moka_response_field_e05.py").read_text()
        self.assertNotIn("train_response_model(", source)
        self.assertNotIn("solve_qp", source)

    def test_frozen_model_loader_reproduces_prediction(self):
        try:
            import numpy as np
            import torch
        except ImportError:
            self.skipTest("NumPy and PyTorch are available in the H100 environment")
        from main.multilink_ellipsoid.moka_response_field import (
            build_response_model,
            load_frozen_response_model,
            predict_response,
            witness_features,
        )

        torch.manual_seed(7)
        context = np.asarray([1.0, 2.0, 3.0])
        input_dimension = witness_features(context).shape[1]
        original = build_response_model(torch, input_dimension, 8)
        payload = {
            "feature_mean": np.zeros(input_dimension).tolist(),
            "feature_scale": np.ones(input_dimension).tolist(),
            "value_scale": 0.2,
            "response_scale": 0.3,
            "state_dict": {
                name: value.detach().numpy().tolist()
                for name, value in original.state_dict().items()
            },
        }
        frozen = load_frozen_response_model(torch, payload)
        expected = {
            "model": original,
            "device": torch.device("cpu"),
            "feature_mean": np.zeros(input_dimension),
            "feature_scale": np.ones(input_dimension),
            "value_scale": 0.2,
            "response_scale": 0.3,
        }
        expected_value, expected_gradient = predict_response(expected, context)
        actual_value, actual_gradient = predict_response(frozen, context)
        self.assertTrue(np.array_equal(expected_value, actual_value))
        self.assertTrue(np.array_equal(expected_gradient, actual_gradient))

    def _numpy(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is available in the H100 evaluation environment")
        return np

    def test_preregistered_config_is_frozen(self) -> None:
        from main.multilink_ellipsoid.moka_response_field import load_moka_response_config

        config = load_moka_response_config(
            ROOT / "configs/vlsa_distal_moka_response_field_e05.v1.json"
        )
        self.assertEqual(config["state_split"]["train_steps"], [178, 179, 180, 181, 182])
        self.assertEqual(config["state_split"]["validation_steps"], [183, 184])
        self.assertEqual(config["state_split"]["test_steps"], [185, 186])
        self.assertEqual(config["geometry"]["expected_compiled_moka_box_count"], 15)
        self.assertEqual(config["sampling"]["paired_direction_count_per_state"], 32)
        self.assertIn("not_closed_loop", config["claim_scope"])

    def test_witness_features_keep_time_and_row_identity(self) -> None:
        np = self._numpy()
        from main.multilink_ellipsoid.moka_response_field import witness_features

        features = witness_features(np.asarray([1.0, 2.0]))
        self.assertEqual(features.shape, (140, 12))
        np.testing.assert_allclose(np.sum(features[:, -7:], axis=1), 1.0)
        self.assertNotEqual(float(features[0, 2]), float(features[-1, 2]))

    def test_multi_primitive_margin_keeps_robot_rows(self) -> None:
        np = self._numpy()
        from main.multilink_ellipsoid.geometry import Ellipsoid
        from main.multilink_ellipsoid.moka_response_field import multi_primitive_link_margins

        links = [
            Ellipsoid(
                center=np.asarray([float(index), 0.0, 0.0]),
                rotation=np.eye(3),
                semiaxes_m=np.ones(3) * 0.1,
            )
            for index in range(7)
        ]
        obstacles = [
            Ellipsoid(center=np.asarray([0.5, 0.0, 0.0]), rotation=np.eye(3), semiaxes_m=np.ones(3) * 0.1),
            Ellipsoid(center=np.asarray([10.0, 0.0, 0.0]), rotation=np.eye(3), semiaxes_m=np.ones(3) * 0.1),
        ]
        values = multi_primitive_link_margins(links, obstacles)
        self.assertEqual(values.shape, (7,))
        self.assertLess(float(values[0]), float(values[6]))

    def test_runner_has_no_qp_or_closed_loop_execution(self) -> None:
        source = (ROOT / "scripts/evaluate_distal_moka_response_field_e05.py").read_text()
        self.assertIn('"corrected_execution_attempted": False', source)
        self.assertIn("train_response_model", source)
        self.assertIn("matched_random_p_value", source)
        self.assertNotIn("MultiConstraintQp", source)

    def test_validator_recomputes_all_gates(self) -> None:
        source = (ROOT / "scripts/validate_distal_moka_response_field_e05.py").read_text()
        self.assertIn("Moka radius gate differs", source)
        self.assertIn("Moka state gate differs", source)
        self.assertIn("Moka aggregate gate differs", source)


if __name__ == "__main__":
    unittest.main()
