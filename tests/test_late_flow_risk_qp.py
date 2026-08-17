import json
from pathlib import Path
import tempfile
import unittest

from main.multilink_ellipsoid.late_flow_risk_qp import (
    central_difference_batches, load_config, pullback_jacobian,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_late_flow_risk_qp_diagnostic.v1.json"


class LateFlowRiskQpTest(unittest.TestCase):
    def test_config_freezes_one_multi_constraint_inference_only_check(self):
        config = load_config(CONFIG)
        self.assertEqual(config["QP"]["constraint_rows"], [0, 1, 2, 3, 4])
        self.assertEqual(
            config["QP"]["constraint_groups"], ["end_effector", "palm", "L5"],
        )
        self.assertTrue(config["forbidden"]["new_data_collection"])
        self.assertTrue(config["forbidden"]["QP_action_execution"])
        self.assertTrue(config["forbidden"]["fixed_13_candidate_selection"])

    def test_config_rejects_action_execution(self):
        value = json.loads(CONFIG.read_text())
        value["forbidden"]["QP_action_execution"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "forbidden"):
                load_config(path)

    def test_central_batches_cover_each_dimension_once_per_sign(self):
        batches = central_difference_batches(epsilon=0.005)
        records = [record for batch in batches for record in batch["records"]]
        self.assertEqual(len(batches), 3)
        self.assertTrue(all(
            len(batch["residuals"]) == 13
            and all(len(branch) == 10 for branch in batch["residuals"])
            and all(len(row) == 3 for branch in batch["residuals"] for row in branch)
            for batch in batches
        ))
        self.assertEqual(
            sorted((record["dimension"], record["sign"]) for record in records),
            sorted((dimension, sign) for dimension in range(15) for sign in (-1, 1)),
        )

    def test_pullback_recovers_linear_jacobian(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("desktop lacks NumPy")
        epsilon = 0.005
        target = np.arange(7 * 15, dtype=np.float64).reshape(7, 15) / 100.0
        nominal = np.linspace(-0.3, 0.3, 7)
        scored = []
        for batch in central_difference_batches(epsilon=epsilon):
            risks = np.repeat(nominal[None, :], 13, axis=0)
            for record in batch["records"]:
                risks[record["branch_index"]] += (
                    record["sign"] * epsilon * target[:, record["dimension"]]
                )
            scored.append({"risk_by_branch_and_row": risks.tolist()})
        recovered_nominal, recovered = pullback_jacobian(
            scored_batches=scored, epsilon=epsilon,
        )
        self.assertTrue(np.array_equal(np.asarray(recovered_nominal), nominal))
        self.assertTrue(np.allclose(np.asarray(recovered), target, atol=1.0e-12))

    def test_evaluator_never_executes_qp_output_or_candidate_future(self):
        source = (
            ROOT / "scripts/evaluate_late_flow_risk_qp_diagnostic.py"
        ).read_text()
        self.assertIn('"QP_action_executed": False', source)
        self.assertIn('"simulator_candidate_rollout_count": 0', source)
        self.assertIn('"fixed_candidate_selection_count": 0', source)
        self.assertNotIn("monitor.execute(qp", source.lower())

    def test_synthetic_pullback_qp_has_conservative_linearized_solution(self):
        try:
            import numpy as np
            import osqp  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("desktop lacks allocation numerical dependencies")
        from main.multilink_ellipsoid.late_flow_risk_qp import solve_pullback_qp

        risk = np.full(7, -0.2, dtype=np.float64)
        risk[0] = 0.1
        jacobian = np.zeros((7, 15), dtype=np.float64)
        jacobian[0, 0] = 1.0
        result = solve_pullback_qp(
            nominal_risk=risk, jacobian=jacobian,
            risk_margin=0.0, linearized_tightening=0.0001,
            trust_region_action=0.25,
        )
        self.assertTrue(result["QP_valid"])
        self.assertTrue(result["linearized_constraints_satisfied"])
        self.assertLessEqual(result["linearized_risk_by_row"][0], 0.0)
        self.assertLessEqual(result["correction_linf_action"], 0.25 + 1.0e-8)


if __name__ == "__main__":
    unittest.main()
