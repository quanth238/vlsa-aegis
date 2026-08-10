import unittest
from pathlib import Path

from main.multilink_ellipsoid.adaptive_native_distance import load_config


ROOT = Path(__file__).resolve().parents[1]


class AdaptiveNativeDistanceConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(
            ROOT / "configs/vlsa_distal_adaptive_native_distance_moka10.v1.json"
        )

    def test_overlap_query_is_zero_cutoff(self):
        measurement = self.config["measurement"]
        self.assertEqual(measurement["overlap_distmax_m"], 0.0)
        self.assertEqual(
            measurement["positive_distmax_sequence_m"],
            [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064],
        )

    def test_right_censoring_is_not_an_exact_label(self):
        self.assertEqual(
            self.config["measurement"]["right_censored_value_semantics"],
            "safe_lower_bound_not_exact_target",
        )

    def test_no_learning_or_control(self):
        gate = self.config["gate"]
        self.assertFalse(gate["training_in_this_gate"])
        self.assertFalse(gate["QP_in_this_gate"])
        self.assertFalse(gate["closed_loop_in_this_gate"])


if __name__ == "__main__":
    unittest.main()
