from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
