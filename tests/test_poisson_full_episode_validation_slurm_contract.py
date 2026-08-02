from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
BATCH_PATH = ROOT / "slurm/poisson_full_episode_feasibility_validate.sbatch"


class PoissonFullEpisodeValidationSlurmContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = BATCH_PATH.read_text(encoding="utf-8")

    def test_shell_is_valid_and_cpu_only(self):
        subprocess.run(["bash", "-n", str(BATCH_PATH)], check=True)
        self.assertIn("#SBATCH --cpus-per-task=2", self.source)
        self.assertIn("#SBATCH --mem=8G", self.source)
        self.assertNotIn("#SBATCH --gres", self.source)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', self.source)

    def test_consumer_binds_exact_terminal_artifact_and_producer(self):
        for variable in (
            "REMOTE_REPO",
            "FULL_EPISODE_RESULT",
            "EXPECTED_GIT_COMMIT",
            "EXPECTED_PRODUCER_JOB_ID",
            "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST",
            "SLURM_CPUS_PER_TASK",
        ):
            with self.subTest(variable=variable):
                self.assertIn('${%s:?' % variable, self.source)
        self.assertIn('"${REGISTERED_RESULT_ROOT}"/*/result.json', self.source)
        self.assertIn("require_canonical_real_file", self.source)
        self.assertIn("producer and independent consumer must use distinct jobs", self.source)

    def test_consumer_uses_full_episode_validator_and_protocol(self):
        self.assertIn(
            "scripts/validate_poisson_full_episode_feasibility_artifact.py",
            self.source,
        )
        self.assertIn(
            "configs/vlsa_poisson_full_episode_feasibility.v1.json",
            self.source,
        )
        self.assertIn('--expected-job-id "${EXPECTED_PRODUCER_JOB_ID}"', self.source)
        self.assertIn('if [[ -n "$(git status --short)" ]]', self.source)


if __name__ == "__main__":
    unittest.main()
