import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # The local structural gate intentionally stays light.
    np = None

from main.multilink_ellipsoid.local_residual_bound import (
    build_pool, load_config, query_upper_bound,
)
from main.multilink_ellipsoid.action_conditioned_margin import (
    build_model, load_model, save_model,
)
from scripts.evaluate_distal_local_residual_bound_moka10 import (
    _recorded_safety_flags,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_local_residual_bound_moka10.v1.json"


class LocalResidualBoundTest(unittest.TestCase):
    def test_frozen_grouped_protocol(self):
        config = load_config(CONFIG)
        self.assertEqual(config["split"]["expected_state_counts"], {
            "train": 60, "validation": 10, "test": 15,
        })
        self.assertTrue(config["split"][
            "test_never_used_for_fold_training_residual_estimation_normalization_or_thresholds"
        ])
        self.assertEqual(config["leave_one_episode_out"]["fold_count"], 12)
        self.assertEqual(config["leave_one_episode_out"][
            "expected_residual_count"
        ], 52500)
        self.assertFalse(config["decision"]["closed_loop_in_this_gate"])

    @unittest.skipUnless(np is not None, "numpy is validated in the H100 preflight")
    def test_neighbor_maximum_is_one_sided_and_constraint_local(self):
        config = load_config(CONFIG)
        settings = dict(config["local_bound"])
        settings["neighbor_count"] = 2
        features = []
        residuals = []
        constraints = []
        for constraint in range(7):
            for location, residual in ((0.0, -0.002), (1.0, 0.003)):
                row = np.zeros(46)
                row[0] = location
                features.append(row)
                residuals.append(residual + constraint * 0.001)
                constraints.append(constraint)
        pool = build_pool(features, residuals, constraints)
        query = np.zeros((1, 7, 46))
        result = query_upper_bound(pool, query, settings)
        expected = np.asarray([
            max(0.0, 0.003 + constraint * 0.001) + 0.001
            for constraint in range(7)
        ])
        np.testing.assert_allclose(result["upper_error_m"][0], expected)

    def test_config_rejects_adaptive_test_calibration(self):
        payload = json.loads(CONFIG.read_text())
        payload["local_bound"]["test_labels_used_for_adaptation"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "bound differs"):
                load_config(path)

    def test_both_immutable_off_grid_schemas_keep_the_same_true_label(self):
        self.assertEqual(_recorded_safety_flags({
            "true_safe": True,
        }), (None, True))
        self.assertEqual(_recorded_safety_flags({
            "arms": {"0mm": {
                "multi_region_predicted_safe": False, "true_safe": True,
            }},
        }), (False, True))

    @unittest.skipUnless(np is not None, "numerical runtime is required")
    def test_immutable_action_model_round_trip(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is validated in the H100 preflight")
        model = build_model([4]).to(dtype=torch.float64)
        state = {
            "state_dicts": [{
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }],
            "feature_mean": np.zeros(56),
            "feature_standard_deviation": np.ones(56),
            "output_mean_mm": 1.5, "output_scale_mm": 2.0,
            "calibration_m": np.arange(7, dtype=np.float64) * 0.001,
            "hidden_widths": [4], "ensemble_seeds": [11],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_model(path, state)
            loaded, loaded_state = load_model(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(list(loaded[0].state_dict()), list(model.state_dict()))
        for name, value in model.state_dict().items():
            np.testing.assert_array_equal(
                loaded[0].state_dict()[name].detach().numpy(), value.numpy()
            )
        np.testing.assert_array_equal(
            loaded_state["calibration_m"], state["calibration_m"]
        )


if __name__ == "__main__":
    unittest.main()
