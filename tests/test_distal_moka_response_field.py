from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MokaResponseFieldTests(unittest.TestCase):
    def test_frozen_ranker_audit_changes_only_test_directions(self):
        config = json.loads(
            (ROOT / "configs/vlsa_distal_moka_frozen_ranker_audit_e05.v1.json").read_text()
        )
        self.assertEqual(config["state"], {"step": 182})
        self.assertEqual(config["sampling"]["radius_action"], 0.0125)
        self.assertEqual(config["sampling"]["paired_direction_count"], 64)
        self.assertEqual(config["sampling"]["best_of_n_prefixes"], [4, 8, 16, 32, 64])
        self.assertEqual(config["frozen_model"]["arm"], "direction_conditioned_scalar")
        self.assertIn("model_training", config["forbidden"])
        self.assertIn("qp", config["forbidden"])
        source = (ROOT / "scripts/audit_distal_moka_frozen_ranker_e05.py").read_text()
        self.assertIn("load_frozen_scalar_model", source)
        self.assertNotIn("train_scalar_model", source)
        self.assertNotIn("solve_qp", source)
        self.assertIn('"training_or_control_attempted": False', source)

    def test_frozen_ranker_validator_recomputes_primary_gate(self):
        source = (ROOT / "scripts/validate_distal_moka_frozen_ranker_audit_e05.py").read_text()
        self.assertIn("_binomial(successes, count)", source)
        self.assertIn('metrics["gate"]["checks"] == checks', source)
        self.assertIn('set(reports) == {"4", "8", "16", "32", "64"}', source)

    def test_frozen_scalar_loader_reconstructs_exact_architecture(self):
        source = (ROOT / "main/multilink_ellipsoid/moka_local_action_value.py").read_text()
        self.assertIn("def load_frozen_scalar_model", source)
        self.assertIn("model = build_scalar_model", source)
        self.assertIn('"0.weight"', source)

    def test_local_action_value_ablation_is_matched_and_prediction_only(self):
        config = json.loads(
            (ROOT / "configs/vlsa_distal_moka_local_action_value_e05.v1.json").read_text()
        )
        self.assertEqual(config["sampling"]["radius_action"], 0.0125)
        self.assertEqual(config["sampling"]["train_direction_indexes"], [0, 20])
        self.assertEqual(config["sampling"]["validation_direction_indexes"], [20, 24])
        self.assertEqual(config["sampling"]["test_direction_indexes"], [24, 32])
        self.assertEqual(config["input"]["model_input_dimension"], 40)
        self.assertEqual(
            set(config["matched_models"]),
            {"direction_conditioned_scalar", "nonlinear_local_action_value"},
        )
        self.assertIn("qp", config["forbidden"])
        source = (ROOT / "scripts/evaluate_distal_moka_local_action_value_e05.py").read_text()
        self.assertIn("response_model = train_scalar_model", source)
        self.assertIn("value_model = train_scalar_model", source)
        self.assertIn('"correction_or_qp_attempted": False', source)
        self.assertNotIn("solve_qp", source)

    def test_local_action_value_validator_recomputes_model_gates(self):
        source = (ROOT / "scripts/validate_distal_moka_local_action_value_e05.py").read_text()
        self.assertIn('set(reports) == {"direction_conditioned_scalar", "nonlinear_local_action_value"}', source)
        self.assertIn('report["gate"]["checks"] == checks', source)
        self.assertIn('result.get("passing_models") == passing', source)
        self.assertIn('len(parameter_counts) == 1', source)

    def test_scalar_model_architecture_is_shared(self):
        source = (ROOT / "main/multilink_ellipsoid/moka_local_action_value.py").read_text()
        self.assertEqual(source.count("def build_scalar_model"), 1)
        self.assertIn("model = build_scalar_model", source)
        self.assertNotIn("MultiConstraintQp", source)

    def test_secant_radius_ablation_changes_only_physical_radius(self):
        config = json.loads(
            (ROOT / "configs/vlsa_distal_moka_secant_radius_e05.v1.json").read_text()
        )
        compact = json.loads(
            (ROOT / "configs/vlsa_distal_moka_compact_input_ablation_e05.v1.json").read_text()
        )
        self.assertEqual(config["model"], compact["model"])
        self.assertEqual(config["input"]["context_groups"], ["nominal_first_five_xyz"])
        self.assertEqual(config["sampling"]["test_radii_action"], [0.025, 0.0125])
        self.assertEqual(config["direction_protocol"]["generation_radius_action"], 0.05)
        self.assertTrue(config["direction_protocol"]["reuse_exact_directions_across_radii"])
        self.assertIn("input_change", config["forbidden"])
        self.assertIn("direction_change", config["forbidden"])
        self.assertIn("qp", config["forbidden"])
        source = (ROOT / "scripts/evaluate_distal_moka_secant_radius_e05.py").read_text()
        self.assertIn("for radius in sampling[\"test_radii_action\"]", source)
        self.assertIn("float(radius) * direction", source)
        self.assertNotIn("solve_qp", source)

    def test_secant_radius_validator_recomputes_each_gate(self):
        source = (ROOT / "scripts/validate_distal_moka_secant_radius_e05.py").read_text()
        self.assertIn('set(reports) == {"0.025", "0.0125"}', source)
        self.assertIn('report["local_ridge"]["design_rank"] == 15', source)
        self.assertIn('report["gate"]["checks"] == checks', source)
        self.assertIn('result.get("passing_radii") == passing', source)

    def test_compact_input_ablation_changes_only_the_input(self):
        config = json.loads(
            (
                ROOT
                / "configs/vlsa_distal_moka_compact_input_ablation_e05.v1.json"
            ).read_text()
        )
        baseline = json.loads(
            (ROOT / "configs/vlsa_distal_moka_response_field_e05.v1.json").read_text()
        )
        self.assertEqual(config["model"], baseline["model"])
        self.assertEqual(config["state"]["step"], 182)
        self.assertEqual(config["input"]["context_dimension"], 15)
        self.assertEqual(config["input"]["model_input_dimension"], 25)
        self.assertEqual(config["input"]["context_groups"], ["nominal_first_five_xyz"])
        self.assertIn("qp", config["forbidden"])
        self.assertIn("additional_input_group", config["forbidden"])
        source = (
            ROOT / "scripts/evaluate_distal_moka_compact_input_ablation_e05.py"
        ).read_text()
        self.assertIn("train_response_model([train_state], [train_state]", source)
        self.assertNotIn("solve_qp", source)

    def test_compact_input_validator_recomputes_the_gate(self):
        source = (
            ROOT / "scripts/validate_distal_moka_compact_input_ablation_e05.py"
        ).read_text()
        self.assertIn('ridge.get("design_rank") == 15', source)
        self.assertIn('result["gate"]["checks"] == checks', source)
        self.assertIn('result["gate"]["pass"] is bool(all(checks.values()))', source)

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
