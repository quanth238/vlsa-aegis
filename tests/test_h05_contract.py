from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class H05ContractTest(unittest.TestCase):
    def test_projection_uses_frozen_model_not_simulator_rollouts(self) -> None:
        source = (ROOT / "main/crfs_oracle/runner.py").read_text(encoding="utf-8")
        self.assertIn("solve_kinematic_projection", source)
        self.assertNotIn("solve_simulator_projection(\n                nominal_actions", source)
        projection = (ROOT / "main/crfs_oracle/projection.py").read_text(encoding="utf-8")
        self.assertIn("static branch geometry", projection)
        self.assertIn("response_matrix", projection)
        self.assertIn("bump = np.sin", projection)
        self.assertIn("feasible_attempts", projection)
        self.assertIn("horizon = nominal.shape[0]", projection)
        self.assertIn("delta[-1]", projection)
        self.assertIn('"type": "fixed_endpoint_clearance"', projection)
        self.assertLess(projection.index("fixed_endpoint ="), projection.index("from scipy.optimize import minimize"))
        self.assertIn("candidate[:, :3] >= low[:, :3]", projection)

    def test_colliding_population_manifest_is_bounded_and_unique(self) -> None:
        cases = [
            json.loads(line)
            for line in (ROOT / "manifests/oracle_h05_colliding.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(cases), 20)
        self.assertEqual(len({case["case_id"] for case in cases}), 20)

    def test_terminal_evidence_records_both_registered_horizons(self) -> None:
        summary = json.loads((ROOT / "evidence/h05/h05-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "failed_stop_rule")
        self.assertEqual(summary["h5"]["certified_infeasible"], 20)
        self.assertEqual(summary["h10_refinement"]["certified_infeasible"], 20)
        self.assertEqual(summary["h5"]["feasible_fraction"], 0.0)
        self.assertEqual(summary["h10_refinement"]["feasible_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
