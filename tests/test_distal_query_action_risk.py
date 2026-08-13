import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from main.multilink_ellipsoid.pure_backup import orthonormal_local_frame
from main.multilink_ellipsoid.query_action_risk import (
    candidate_definitions,
    combine_row_minima,
    exact_safe,
    load_config,
    risk_from_row_minimum,
    temporal_profile,
)


ROOT = Path(__file__).resolve().parents[1]


class QueryActionRiskTest(unittest.TestCase):
    def test_config_and_candidate_bank(self):
        config = load_config(ROOT / "configs/vlsa_distal_query_action_risk_e05.v1.json")
        nominal = np.zeros((5, 7), dtype=np.float64)
        nominal[:, 6] = np.linspace(-1.0, 1.0, 5)
        frame = orthonormal_local_frame([1.0, 0.0, 0.0])
        candidates = candidate_definitions(nominal, frame, config)
        self.assertEqual(len(candidates), 37)
        self.assertEqual(candidates[0]["name"], "nominal")
        for candidate in candidates:
            actions = np.asarray(candidate["actions"])
            np.testing.assert_array_equal(actions[:, 3:], nominal[:, 3:])
            self.assertLessEqual(np.max(np.abs(actions[:, :3])), 1.0)

    def test_profiles_have_unit_l2_norm(self):
        for name in ("constant", "front_loaded"):
            self.assertAlmostEqual(np.linalg.norm(temporal_profile(name)), 1.0)

    def test_risk_composition_and_terminal(self):
        combined = combine_row_minima([0.01] * 7, [0.002] * 7)
        risk = risk_from_row_minimum(combined, 0.001)
        self.assertEqual(combined, [0.002] * 7)
        self.assertTrue(all(item < 0.0 for item in risk))
        self.assertTrue(exact_safe({
            "terminal_status": "SAFE_TERMINAL",
            "physical_veto": False,
            "combined_risk": risk,
        }))
        self.assertFalse(exact_safe({
            "terminal_status": "UNKNOWN_TIMEOUT",
            "physical_veto": False,
            "combined_risk": risk,
        }))

    def test_loader_rejects_protocol_change(self):
        source = ROOT / "configs/vlsa_distal_query_action_risk_e05.v1.json"
        value = json.loads(source.read_text())
        value["state"]["query_boundary_step"] = 184
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
