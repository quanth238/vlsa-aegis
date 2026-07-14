from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MeasurementBatchContractTest(unittest.TestCase):
    def test_batch_is_measurement_only_and_restartable(self) -> None:
        source = (ROOT / "main/run_crfs_measurement_batch.py").read_text(encoding="utf-8")
        self.assertIn("if not config.stop_after_measurement", source)
        self.assertIn('"batch-failure.json"', source)
        self.assertIn("skipped_valid_measurement", (ROOT / "main/crfs_oracle/runner.py").read_text(encoding="utf-8"))

    def test_one_policy_server_serves_a_declared_case_range(self) -> None:
        source = (ROOT / "scripts/hpc/run_oracle_case.sh").read_text(encoding="utf-8")
        self.assertIn("main/run_crfs_measurement_batch.py", source)
        self.assertIn('--case-start "$CASE_START" --case-end "$CASE_END"', source)
        self.assertLess(source.index("scripts/serve_policy.py"), source.index("main/run_crfs_measurement_batch.py"))


if __name__ == "__main__":
    unittest.main()
