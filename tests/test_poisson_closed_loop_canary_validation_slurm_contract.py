from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
BATCH_PATH = ROOT / "slurm/poisson_closed_loop_canary_validate.sbatch"


class PoissonClosedLoopValidationSlurmContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = BATCH_PATH.read_text(encoding="utf-8")

    def test_shell_is_valid_bounded_and_cpu_only(self):
        subprocess.run(["bash", "-n", str(BATCH_PATH)], check=True)
        self.assertIn("#SBATCH --cpus-per-task=2", self.source)
        self.assertIn("#SBATCH --mem=8G", self.source)
        self.assertIn("#SBATCH --time=00:30:00", self.source)
        self.assertNotIn("#SBATCH --gres", self.source)
        self.assertNotIn("#SBATCH --partition", self.source)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', self.source)
        self.assertIn("SLURM_GPUS_ON_NODE", self.source)

    def test_consumer_binds_exact_result_source_and_producer(self):
        for variable in (
            "REMOTE_REPO",
            "CLOSED_LOOP_RESULT",
            "HISTORICAL_RESULT_ROOT",
            "EXPECTED_PRODUCER_GIT_COMMIT",
            "EXPECTED_CONSUMER_GIT_COMMIT",
            "EXPECTED_PRODUCER_JOB_ID",
            "EXPECTED_PRODUCER_HOST",
            "EXPECTED_PRODUCER_DEVICE",
            "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST",
            "SLURM_CPUS_PER_TASK",
        ):
            with self.subTest(variable=variable):
                self.assertIn("${%s:?" % variable, self.source)
        self.assertIn("require_canonical_real_file", self.source)
        self.assertIn('"result.json"', self.source)
        self.assertIn(
            "producer and independent consumer must use distinct Slurm jobs",
            self.source,
        )
        self.assertIn('if [[ -n "$(git status --short)" ]]', self.source)
        self.assertIn(
            'git rev-parse HEAD)" != "${EXPECTED_CONSUMER_GIT_COMMIT}"',
            self.source,
        )

    def test_validator_success_is_atomically_published_without_overwrite(self):
        self.assertIn(
            "scripts/validate_poisson_closed_loop_canary_artifact.py",
            self.source,
        )
        for argument in (
            '--repo-root "${REMOTE_REPO}"',
            '--result "${CLOSED_LOOP_RESULT}"',
            '--historical-result-root "${HISTORICAL_RESULT_ROOT}"',
            '--expected-producer-commit "${EXPECTED_PRODUCER_GIT_COMMIT}"',
            '--consumer-commit "${EXPECTED_CONSUMER_GIT_COMMIT}"',
            '--expected-producer-job-id "${EXPECTED_PRODUCER_JOB_ID}"',
            '--expected-producer-host "${EXPECTED_PRODUCER_HOST}"',
            '--expected-producer-device "${EXPECTED_PRODUCER_DEVICE}"',
        ):
            with self.subTest(argument=argument):
                self.assertIn(argument, self.source)
        self.assertIn('>"${PARTIAL_PATH}"', self.source)
        self.assertIn('mv -- "${PARTIAL_PATH}" "${RECEIPT_PATH}"', self.source)
        self.assertIn("validation receipt target already exists", self.source)
        self.assertIn('value.get("artifact_valid") is not True', self.source)
        self.assertLess(
            self.source.index('>"${PARTIAL_PATH}"'),
            self.source.index('mv -- "${PARTIAL_PATH}" "${RECEIPT_PATH}"'),
        )


if __name__ == "__main__":
    unittest.main()
