from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT / "slurm/poisson_triggered_rescue.sbatch"
CONSUMER = ROOT / "slurm/poisson_triggered_rescue_validate.sbatch"


class TriggeredRescueSlurmContractTests(unittest.TestCase):
    def test_producer_is_one_h100_without_policy_server(self):
        source = PRODUCER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("#SBATCH --cpus-per-task=8", source)
        self.assertNotIn("serve_policy.py", source)
        self.assertNotIn("OPENPI_PYTHON", source)
        self.assertIn("run_poisson_triggered_rescue.py", source)

    def test_producer_rejects_existing_run_and_unclean_source(self):
        source = PRODUCER.read_text(encoding="utf-8")
        self.assertIn('git status --short', source)
        self.assertIn('git rev-parse HEAD', source)
        self.assertIn('run root already exists', source)

    def test_heldout_case_requires_a_validated_positive_e05(self):
        source = PRODUCER.read_text(encoding="utf-8")
        self.assertIn("E05_VALIDATION_RECEIPT", source)
        self.assertIn("SAFE_TASK_SUCCESS_USEFUL_CORRECTION", source)
        self.assertIn('value.get("feasible") is True', source)

    def test_consumer_is_distinct_cpu_only_and_writes_one_receipt(self):
        source = CONSUMER.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=2", source)
        self.assertNotIn("#SBATCH --gres", source)
        self.assertIn("EXPECTED_PRODUCER_JOB_ID", source)
        self.assertIn("validation_receipt.json", source)
        self.assertIn("validate_poisson_triggered_rescue_artifact.py", source)


if __name__ == "__main__":
    unittest.main()
