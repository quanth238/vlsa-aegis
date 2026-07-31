from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
BATCH_PATH = ROOT / "slurm/poisson_active_canary.sbatch"


class PoissonActiveSlurmContractTest(unittest.TestCase):
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
        self.assertNotIn("#SBATCH --time", self.source)
        self.assertNotIn("#SBATCH --array", self.source)
        self.assertIn("SLURM_JOB_ID", self.source)
        self.assertIn("SLURM_JOB_NODELIST", self.source)
        self.assertIn("SLURM_CPUS_PER_TASK", self.source)
        self.assertIn("CUDA_VISIBLE_DEVICES", self.source)

    def test_launcher_requires_registered_remote_paths(self):
        self.assertIn(
            "/home/quanth/working_space/vlsa-aegis-poisson-feasibility", self.source
        )
        self.assertIn(
            "/mnt/data/quanth/venvs/safety_vla/main/bin/python", self.source
        )
        self.assertIn(
            "/mnt/data/quanth/experiments/vlsa-aegis-poisson-feasibility", self.source
        )
        self.assertNotIn("/home/quanth/working_space/vlsa-aegis-table-repro", self.source)
        self.assertNotIn(
            "/mnt/data/quanth/experiments/vlsa-poisson-active-canary", self.source
        )
        self.assertIn('${REMOTE_REPO:?REMOTE_REPO is required}', self.source)
        self.assertIn('${RESULT_ROOT:?RESULT_ROOT is required}', self.source)
        self.assertIn('"${REMOTE_REPO}" != "${REGISTERED_REMOTE_REPO}"', self.source)
        self.assertIn('"${RESULT_ROOT}" != "${REGISTERED_RESULT_ROOT}"', self.source)
        self.assertIn("vlsa-t1-goal-ii-t0-e05", self.source)

    def test_launcher_requires_all_bound_inputs(self):
        required = (
            "RUN_ID",
            "EXPECTED_GIT_COMMIT",
            "HISTORICAL_RESULT_ROOT",
            "CHECKPOINT_RECEIPT",
            "NUMERIC_VALIDATION_RESULT",
            "PARITY_RESULT",
            "SHADOW_IDENTIFICATION_RESULT",
        )
        for variable in required:
            with self.subTest(variable=variable):
                self.assertIn('${%s:?' % variable, self.source)
        self.assertIn('if [[ -n "$(git status --short)" ]]', self.source)
        self.assertIn('"$(git rev-parse HEAD)"', self.source)
        self.assertIn('"${EXPECTED_GIT_COMMIT}"', self.source)
        self.assertIn("require_regular_file", self.source)

    def test_launcher_uses_osmesa_and_registered_import_roots(self):
        self.assertIn("export MUJOCO_GL=osmesa", self.source)
        self.assertIn("export PYOPENGL_PLATFORM=osmesa", self.source)
        self.assertIn(
            'export PYTHONPATH="${REMOTE_REPO}/openpi/src:${REMOTE_REPO}/openpi/packages/openpi-client/src:${REMOTE_REPO}:${REMOTE_REPO}/main:${REMOTE_REPO}/safelibero',
            self.source,
        )
        self.assertIn("export LIBERO_CONFIG_PATH=", self.source)

    def test_launcher_executes_active_canary_with_every_bound_artifact(self):
        self.assertIn(
            'exec "${EVALUATION_PYTHON}" scripts/run_poisson_active_canary.py',
            self.source,
        )
        expected_arguments = (
            '--repo-root "${REMOTE_REPO}"',
            '--historical-result-root "${HISTORICAL_RESULT_ROOT}"',
            '--checkpoint-receipt "${CHECKPOINT_RECEIPT}"',
            '--numeric-validation-result "${NUMERIC_VALIDATION_RESULT}"',
            '--parity-result "${PARITY_RESULT}"',
            '--shadow-identification-result "${SHADOW_IDENTIFICATION_RESULT}"',
            '--output-root "${RESULT_ROOT}"',
            '--run-id "${RUN_ID}"',
            '--case-id "${CASE_ID}"',
        )
        for argument in expected_arguments:
            with self.subTest(argument=argument):
                self.assertIn(argument, self.source)


if __name__ == "__main__":
    unittest.main()
