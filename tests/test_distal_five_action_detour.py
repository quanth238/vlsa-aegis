import json
from pathlib import Path
import unittest

import numpy as np

from main.multilink_ellipsoid.five_action_detour import (
    load_five_action_detour_config,
    solve_detour_qp,
)


ROOT = Path(__file__).resolve().parents[1]


class FiveActionDetourTest(unittest.TestCase):
    def test_config_and_qp_preserve_endpoint(self):
        import cvxpy as cp

        config = load_five_action_detour_config(
            ROOT / "configs/vlsa_distal_five_action_detour_e05.v1.json"
        )
        nominal = np.zeros((5, 7), dtype=np.float64)
        h = np.full((5, 7), 0.002, dtype=np.float64)
        rows = np.zeros((35, 15), dtype=np.float64)
        terminal_rows = np.zeros((3, 15), dtype=np.float64)
        result = solve_detour_qp(
            cp=cp,
            nominal_actions=nominal,
            center_correction=np.zeros(15),
            clearance_trace_m=h,
            clearance_jacobian_m_per_action=rows,
            terminal_eef_error_m=np.zeros(3),
            terminal_eef_jacobian_m_per_action=terminal_rows,
            config=config,
        )
        self.assertTrue(result["valid"])
        correction = np.asarray(result["total_correction"]).reshape(5, 3)
        np.testing.assert_allclose(np.sum(correction, axis=0), 0.0, atol=1e-8)

    def test_config_rejects_changed_activation(self):
        source = ROOT / "configs/vlsa_distal_five_action_detour_e05.v1.json"
        value = json.loads(source.read_text())
        value["state_protocol"]["activation_step"] = 181
        temporary = ROOT / "tests" / ".temporary-five-action-detour.json"
        try:
            temporary.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_five_action_detour_config(temporary)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
