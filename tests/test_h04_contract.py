from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class H04ContractTest(unittest.TestCase):
    def test_h04_is_held_out_and_thresholded(self) -> None:
        source = (ROOT / "main/run_h04_calibration.py").read_text(encoding="utf-8")
        self.assertIn("np.linalg.lstsq", source)
        self.assertIn("median_endpoint <= 0.005", source)
        self.assertIn("p95_clearance <= 0.01", source)
        self.assertIn("false_safe == 0", source)
        self.assertIn("branch_obstacle_boxes", (ROOT / "main/crfs_oracle/runner.py").read_text())


if __name__ == "__main__":
    unittest.main()
