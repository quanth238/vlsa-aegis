from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
BATCH_PATH = ROOT / "slurm/poisson_full_episode_feasibility.sbatch"


class PoissonFullEpisodeSlurmContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = BATCH_PATH.read_text(encoding="utf-8")

    def test_launcher_has_valid_shell_syntax(self):
        subprocess.run(["bash", "-n", str(BATCH_PATH)], check=True)

    def test_launcher_uses_one_bounded_h100_allocation(self):
        self.assertIn("#SBATCH --partition=mig", self.source)
        self.assertIn("#SBATCH --account=normal", self.source)
        self.assertIn("#SBATCH --qos=normal", self.source)
        self.assertEqual(self.source.count("#SBATCH --gres=gpu:1"), 1)
        self.assertIn("#SBATCH --cpus-per-task=8", self.source)
        self.assertIn("#SBATCH --mem=64G", self.source)
        self.assertNotIn("#SBATCH --array", self.source)
        for variable in (
            "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST",
            "SLURM_CPUS_PER_TASK",
            "CUDA_VISIBLE_DEVICES",
        ):
            with self.subTest(variable=variable):
                self.assertIn('${%s:?' % variable, self.source)

    def test_launcher_binds_clean_source_and_immutable_result(self):
        self.assertIn(
            "/home/quanth/working_space/vlsa-aegis-poisson-fast-feasibility",
            self.source,
        )
        self.assertIn(
            "/mnt/data/quanth/experiments/vlsa-aegis-poisson-full-episode-feasibility",
            self.source,
        )
        for variable in (
            "REMOTE_REPO",
            "RESULT_ROOT",
            "RUN_ID",
            "EXPECTED_GIT_COMMIT",
            "HISTORICAL_RESULT_ROOT",
        ):
            with self.subTest(variable=variable):
                self.assertIn('${%s:?' % variable, self.source)
        self.assertIn('if [[ -n "$(git status --short)" ]]', self.source)
        self.assertIn('"$(git rev-parse HEAD)"', self.source)
        self.assertIn('"${RESULT_ROOT}/${RUN_ID}"', self.source)

    def test_launcher_uses_full_episode_protocol_without_policy_server(self):
        self.assertIn("run_poisson_fast_feasibility.py", self.source)
        self.assertIn(
            "configs/vlsa_poisson_full_episode_feasibility.v1.json", self.source
        )
        self.assertIn("vlsa-t1-goal-ii-t0-e05", self.source)
        self.assertNotIn("POLICY_PYTHON", self.source)
        self.assertNotIn("policy_server", self.source.lower())
        self.assertNotIn("PARITY_RESULT", self.source)
        self.assertNotIn("IDENTIFICATION_RESULT", self.source)

    def test_launcher_uses_registered_runtime(self):
        self.assertIn(
            "/mnt/data/quanth/venvs/safety_vla/main/bin/python", self.source
        )
        self.assertIn("export MUJOCO_GL=osmesa", self.source)
        self.assertIn("export PYOPENGL_PLATFORM=osmesa", self.source)
        self.assertIn("export LIBERO_CONFIG_PATH=", self.source)


if __name__ == "__main__":
    unittest.main()
