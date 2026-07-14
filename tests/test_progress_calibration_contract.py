from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProgressCalibrationContractTest(unittest.TestCase):
    def test_config_freezes_reach_phase_and_executed_prefix(self) -> None:
        value = json.loads(
            (ROOT / "configs/experiments/reach_progress_calibration.json").read_text(encoding="utf-8")
        )
        self.assertEqual(value["action_horizon"], 10)
        self.assertEqual(value["executed_prefix"], 5)
        self.assertEqual(value["progress_calibration"]["target_object"], "akita_black_bowl_1")
        self.assertEqual(value["progress_calibration"]["minimum_positive_examples"], 50)
        self.assertEqual(value["progress_calibration"]["simulator_safety_margin_m"], 0.005)
        self.assertEqual(value["progress_calibration"]["quantile_method"], "inverted_cdf")

    def test_runner_is_nominal_only_and_writes_atomically(self) -> None:
        source = (ROOT / "main/crfs_oracle/progress_calibration.py").read_text(encoding="utf-8")
        self.assertIn('intervention_mode="none"', source)
        self.assertIn("atomic_write_json(output, result)", source)
        self.assertIn("simulator_safety_margin_m", source)
        self.assertIn("eligible_for_p_min", source)
        self.assertNotIn("solve_kinematic_projection", source)
        self.assertNotIn("torch", source.lower())
        self.assertNotIn("train", source.lower())

    def test_summary_refuses_group_leakage_and_incomplete_results(self) -> None:
        source = (ROOT / "main/summarize_reach_progress.py").read_text(encoding="utf-8")
        self.assertIn("calibration/evaluation group leakage", source)
        self.assertIn("results are not validation-ready", source)
        self.assertIn("minimum_positive_examples", source)
        self.assertIn("ordered_result_set_digest", source)


if __name__ == "__main__":
    unittest.main()
