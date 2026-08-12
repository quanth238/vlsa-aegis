import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from main.multilink_ellipsoid.counterfactual_field import (
    fit_paired_field,
    load_counterfactual_field_config,
    matched_random_p_value,
    paired_chunk_directions,
    project_direction,
)


ROOT = Path(__file__).resolve().parents[1]


class CounterfactualFieldTest(unittest.TestCase):
    def test_registered_config_loads(self):
        value = load_counterfactual_field_config(
            ROOT / "configs/vlsa_distal_counterfactual_field_e05.v1.json"
        )
        self.assertEqual(value["sampling"]["paired_direction_count"], 64)

    def test_config_rejects_protocol_drift(self):
        path = ROOT / "configs/vlsa_distal_counterfactual_field_e05.v1.json"
        value = json.loads(path.read_text())
        value["sampling"]["paired_direction_count"] = 63
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary) / "changed.json"
            changed.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "pair count"):
                load_counterfactual_field_config(changed)

    def test_directions_are_paired_feasible_and_endpoint_preserving(self):
        nominal = np.asarray(
            [[0.9, 0.1, -0.2], [0.2, -0.1, 0.3], [0.0, 0.2, -0.1], [-0.2, 0.0, 0.1], [-0.1, -0.2, -0.1]],
            dtype=np.float64,
        )
        directions = paired_chunk_directions(64, 5, nominal, 0.1, 1.0)
        self.assertEqual(directions.shape, (64, 15))
        self.assertTrue(np.allclose(np.linalg.norm(directions, axis=1), 1.0))
        self.assertTrue(np.allclose(np.sum(directions.reshape(64, 5, 3), axis=1), 0.0, atol=1e-12))
        self.assertLessEqual(np.max(np.abs(nominal.reshape(1, 15) + 0.1 * directions)), 1.0)
        self.assertLessEqual(np.max(np.abs(nominal.reshape(1, 15) - 0.1 * directions)), 1.0)

    def test_fit_recovers_linear_field_and_holds_out(self):
        rng = np.random.RandomState(8)
        nominal = np.zeros((5, 3), dtype=np.float64)
        directions = paired_chunk_directions(64, 9, nominal, 0.1, 1.0)
        truth = rng.normal(size=15)
        truth -= truth.reshape(5, 3).mean(axis=0).reshape(1, 3).repeat(5, axis=0).reshape(15)
        epsilon = 0.1
        derivative = directions.dot(truth)
        result = fit_paired_field(
            directions,
            epsilon * derivative,
            -epsilon * derivative,
            epsilon,
            48,
            1e-9,
        )
        self.assertGreater(result["heldout_pearson"], 0.999)
        self.assertGreater(result["heldout_sign_accuracy"], 0.99)
        self.assertLess(result["heldout_rmse"], 1e-6)

    def test_comparator_projection_and_random_p_value(self):
        nominal = np.zeros((5, 3), dtype=np.float64)
        basis = paired_chunk_directions(32, 7, nominal, 0.1, 1.0)
        projected = project_direction(np.arange(15, dtype=np.float64), basis)
        self.assertAlmostEqual(float(np.linalg.norm(projected)), 1.0)
        self.assertTrue(np.allclose(np.sum(projected.reshape(5, 3), axis=0), 0.0, atol=1e-10))
        self.assertAlmostEqual(matched_random_p_value(2.0, [0.0, 1.0, 3.0]), 0.5)


if __name__ == "__main__":
    unittest.main()
