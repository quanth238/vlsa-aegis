import json
from pathlib import Path
import unittest

from main.multilink_ellipsoid.affine_coefficient_model import (
    affine_values,
    coefficient_grid_actions,
    coefficient_targets,
    load_affine_coefficient_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "vlsa_distal_affine_coefficient_moka10.v1.json"


class AffineCoefficientModelTests(unittest.TestCase):
    def test_config_and_grid_are_frozen(self):
        config = load_affine_coefficient_config(CONFIG)
        lower, upper, candidates = coefficient_grid_actions([0.0, 0.5, -0.5], config)
        self.assertEqual(len(candidates), 125)
        self.assertEqual(
            config["action_sampling"]["expected_certificate_action_count_per_state"],
            126,
        )
        self.assertTrue(
            config["action_sampling"]["include_exact_nominal_certificate_anchor"]
        )
        self.assertEqual(lower.tolist(), [-0.5, 0.0, -1.0])
        self.assertEqual(upper.tolist(), [0.5, 1.0, 0.0])

    def test_error_recovers_lower_intercept(self):
        certificate = {
            "intercept_at_nominal_m": [0.001] * 7,
            "gradients_m_per_action": [[0.01, 0.0, 0.0]] * 7,
        }
        target = coefficient_targets([0.004] * 7, certificate)
        self.assertEqual(target["state_conditioned_error_m"], [0.003] * 7)
        values = affine_values(
            target["nominal_margin_m"], target["gradient_m_per_action"],
            target["state_conditioned_error_m"], [0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        )
        self.assertTrue(all(abs(value - 0.002) < 1.0e-12 for value in values))

    def test_unknown_config_key_is_rejected(self):
        config = json.loads(CONFIG.read_text())
        config["unknown"] = True
        temporary = ROOT / "tests" / ".affine-coefficient-invalid.json"
        try:
            temporary.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_affine_coefficient_config(temporary)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
