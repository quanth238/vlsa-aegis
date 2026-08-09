import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from main.multilink_ellipsoid.ridge_huber_oracle import (
    affine_values, boundary_weights, fit_ridge_huber_gradient, load_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "vlsa_distal_ridge_huber_oracle_moka10.v1.json"


class RidgeHuberOracleTests(unittest.TestCase):
    def test_config_is_frozen(self):
        config = load_config(CONFIG)
        self.assertEqual(config["immutable_source"]["expected_state_count"], 50)
        self.assertEqual(config["resampling"]["replicate_count"], 16)

    def test_boundary_weights_emphasize_boundary(self):
        settings = load_config(CONFIG)["ridge_huber"]
        weights = boundary_weights([0.0, 0.005, 0.05], settings)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "scipy is unavailable")
    def test_linear_gradient_and_lower_bound(self):
        settings = load_config(CONFIG)["ridge_huber"]
        xyz = np.asarray([
            [-1.0, 0.0, 0.0], [-0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0],
        ])
        margin = 0.001 + 0.01 * xyz[:, 0]
        fitted = fit_ridge_huber_gradient(
            xyz, margin, [0.0, 0.0, 0.0], 0.001, settings
        )
        self.assertAlmostEqual(fitted["gradient_m_per_action"][0], 0.01, places=4)
        lower = np.asarray([
            0.001 - fitted["one_sided_error_m"]
            + np.dot(fitted["gradient_m_per_action"], item)
            for item in xyz
        ])
        self.assertLessEqual(float(np.max(lower - margin)), 1.0e-10)

    def test_affine_value_uses_b_minus_e(self):
        values = affine_values(
            [0.002] * 7, [[0.01, 0.0, 0.0]] * 7, [0.001] * 7,
            [0.1, 0.0, 0.0], [0.0, 0.0, 0.0],
        )
        self.assertTrue(np.allclose(values, 0.002))

    def test_unknown_config_key_is_rejected(self):
        config = json.loads(CONFIG.read_text())
        config["unknown"] = True
        temporary = ROOT / "tests" / ".ridge-huber-invalid.json"
        try:
            temporary.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(temporary)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
