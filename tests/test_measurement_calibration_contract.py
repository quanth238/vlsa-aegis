from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MeasurementCalibrationContractTest(unittest.TestCase):
    def test_calibration_is_bounded_and_requires_diverse_states(self) -> None:
        source = (ROOT / "main/audit_mujoco_sphere_box.py").read_text(encoding="utf-8")
        self.assertIn('if samples < 50:', source)
        self.assertIn('"missed_contact_count": 0', source)
        self.assertIn('max_boundary_error <= 1e-8', source)
        self.assertIn('sign_disagreements == 0', source)

    def test_calibration_runs_only_in_slurm(self) -> None:
        source = (ROOT / "slurm/measurement_calibration_mig.sbatch").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --time=00:10:00", source)
        self.assertIn("${SLURM_JOB_ID}", source)


if __name__ == "__main__":
    unittest.main()
