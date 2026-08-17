import json
from pathlib import Path
import tempfile
import unittest

from main.multilink_ellipsoid.learned_risk_sqp import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_learned_risk_sqp_feasibility.v1.json"


class LearnedRiskSqpTest(unittest.TestCase):
    def test_config_freezes_ee_and_multi_constraint_arms(self):
        config = load_config(CONFIG)
        self.assertEqual(config["SQP"]["arms"]["EE_only"], [0])
        self.assertEqual(config["SQP"]["arms"]["EE_palm_L5"], [0, 1, 2, 3, 4])
        self.assertFalse(config["forbidden"]["QP_action_execution"])
        self.assertFalse(config["forbidden"]["candidate_future_rollout"])

    def test_config_rejects_qp_action_execution(self):
        value = json.loads(CONFIG.read_text())
        value["forbidden"]["QP_action_execution"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "forbidden"):
                load_config(path)

    def test_evaluator_never_executes_qp_action(self):
        source = (
            ROOT / "scripts/evaluate_learned_risk_sqp_feasibility.py"
        ).read_text()
        self.assertIn("solve_sqp", source)
        self.assertIn('"QP_action_executed": False', source)
        self.assertIn('"candidate_future_rollout_count": 0', source)
        self.assertNotIn("monitor.execute(arm", source)

    def test_synthetic_linear_risk_qp(self):
        try:
            import numpy as np
            import osqp  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("desktop lacks allocation numerical dependencies")
        from main.multilink_ellipsoid.learned_risk_sqp import solve_sqp

        nominal = np.zeros((5, 7), dtype=np.float64)

        def risk(actions):
            value = float(actions[0, 0])
            return [0.1 + value, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6]

        result = solve_sqp(
            nominal, risk, constraint_rows=[0], maximum_iterations=6,
            finite_difference_action=0.005, trust_region_action=0.25,
            maximum_total_correction_action=0.75,
            line_search_fractions=[1.0, 0.5, 0.25, 0.125], risk_margin=0.0,
        )
        self.assertTrue(result["nonlinear_constraints_satisfied"])
        self.assertLessEqual(result["final_predicted_risk_by_row"][0], 1e-7)
        self.assertFalse(result["QP_action_executed"])


if __name__ == "__main__":
    unittest.main()
