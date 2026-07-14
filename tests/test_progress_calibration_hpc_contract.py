from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ProgressCalibrationHpcContractTest(unittest.TestCase):
    def test_shared_runner_defaults_to_baseline_and_whitelists_modes(self) -> None:
        source = (ROOT / "scripts/hpc/run_oracle_case.sh").read_text(encoding="utf-8")
        self.assertIn('CRFS_RUNNER_MODE:=oracle', source)
        self.assertIn('reach_calibration)', source)
        self.assertIn('endpoint_free)', source)
        self.assertIn('main/run_crfs_oracle.py', source)
        self.assertIn('main/run_reach_progress_batch.py', source)
        self.assertIn('main/run_crfs_endpoint_free.py', source)
        self.assertIn('unknown CRFS_RUNNER_MODE', source)

    def test_r00_jobs_remain_allocation_backed_and_bounded(self) -> None:
        smoke = (ROOT / "slurm/reach_progress_calibration_mig.sbatch").read_text(encoding="utf-8")
        full = (ROOT / "slurm/reach_progress_calibration_main.sbatch").read_text(encoding="utf-8")
        summary = (ROOT / "slurm/reach_progress_summary.sbatch").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:1", smoke)
        self.assertIn("#SBATCH --gres=gpu:1", full)
        self.assertIn("CRFS_RUNNER_MODE=reach_calibration", smoke)
        self.assertIn("CRFS_RUNNER_MODE=reach_calibration", full)
        self.assertNotIn("#SBATCH --gres=gpu", summary)
        self.assertIn("main/summarize_reach_progress.py", summary)

    def test_full_submitter_caps_gpu_equivalents_and_runs_preflight(self) -> None:
        source = (ROOT / "scripts/hpc/submit_reach_progress_calibration.sh").read_text(encoding="utf-8")
        self.assertIn("scripts/hpc/preflight.sh", source)
        self.assertIn('test "$TASKS" -le 2', source)
        self.assertIn("--array='0-$LAST%2'", source)

    def test_smoke_indexes_the_frozen_full_manifest(self) -> None:
        source = (ROOT / "scripts/hpc/submit_reach_progress_smoke.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("frozen 120-case calibration manifest", source)
        self.assertIn("scripts/hpc/preflight.sh", source)


if __name__ == "__main__":
    unittest.main()
