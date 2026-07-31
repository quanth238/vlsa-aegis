from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PoissonSlurmContractTest(unittest.TestCase):
    def test_numeric_validation_uses_one_bounded_allocation(self):
        source = (ROOT / "slurm/poisson_numeric_validation.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --partition=mig", source)
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("#SBATCH --cpus-per-task=4", source)
        self.assertIn("#SBATCH --mem=32G", source)
        self.assertNotIn("#SBATCH --time", source)
        self.assertIn('if [[ -n "$(git status --short)" ]]', source)
        self.assertIn("EXPECTED_GIT_COMMIT", source)
        self.assertIn('git rev-parse HEAD', source)
        self.assertIn('"${EVALUATION_PYTHON}" scripts/run_poisson_numeric_validation.py', source)

    def test_validation_requires_zero_skips_and_clean_source(self):
        source = (ROOT / "scripts/run_poisson_numeric_validation.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("and not skipped", source)
        self.assertIn('and not source["status_short"]', source)
        self.assertIn("and bool(slurm_job_id)", source)
        self.assertIn('and gpu_inventory["available"]', source)
        self.assertIn('all("H100" in device["name"]', source)
        self.assertIn('and h100_only', source)
        self.assertIn('"--query-gpu=name,uuid,memory.total"', source)
        self.assertIn('"scientific_result": False', source)
        self.assertIn("synthetic and numerical tests do not establish SafeLIBERO safety efficacy", source)


if __name__ == "__main__":
    unittest.main()
