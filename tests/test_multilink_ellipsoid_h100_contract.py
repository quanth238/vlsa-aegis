from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MultilinkEllipsoidH100ContractTests(unittest.TestCase):
    def test_sbatch_is_single_h100_bounded(self) -> None:
        source = (ROOT / "slurm/multilink_ellipsoid_shadow_e05.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("#SBATCH --cpus-per-task=8", source)
        self.assertIn("#SBATCH --mem=64G", source)
        self.assertIn("#SBATCH --exclude=worker-3", source)
        self.assertNotIn("#SBATCH --array", source)

    def test_runner_requires_h100_and_uses_registered_interpreters(self) -> None:
        source = (ROOT / "slurm/run_multilink_ellipsoid_shadow_e05.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("simulation requires exactly one visible H100", source)
        self.assertIn("/mnt/data/quanth/venvs/openpi/bin/python", source)
        self.assertIn("/mnt/data/quanth/venvs/safety_vla/main/bin/python", source)
        self.assertIn("trap cleanup EXIT", source)
        self.assertIn("tests.test_multilink_ellipsoid", source)
        self.assertIn("vlsa_distal_three_ellipsoid_shadow_e05.v2.json", source)

    def test_table1_result_is_read_only_validator_input(self) -> None:
        source = (ROOT / "slurm/run_multilink_ellipsoid_shadow_e05.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--archived \"$ARCHIVED_TABLE1_RESULT\"", source)
        self.assertNotIn(">\"$ARCHIVED_TABLE1_RESULT\"", source)
        self.assertNotIn("mv \"$ARCHIVED_TABLE1_RESULT\"", source)
        self.assertIn("vlsa-aegis-distal-three-ellipsoid", source)

    def test_exact_action_replay_is_single_h100_and_read_only(self) -> None:
        source = (ROOT / "slurm/distal_three_ellipsoid_replay_e05.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("#SBATCH --cpus-per-task=8", source)
        self.assertIn("#SBATCH --mem=64G", source)
        self.assertIn("#SBATCH --exclude=worker-3", source)
        self.assertIn("replay_distal_three_ellipsoid_shadow.py", source)
        self.assertIn("--archived \"$ARCHIVED_TABLE1_RESULT\"", source)
        self.assertNotIn(">\"$ARCHIVED_TABLE1_RESULT\"", source)

    def test_replay_verifier_is_allocation_backed(self) -> None:
        source = (ROOT / "slurm/validate_distal_three_ellipsoid_replay.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("#SBATCH --exclude=worker-3", source)
        self.assertIn("validate_distal_three_ellipsoid_replay.py", source)
        self.assertIn("--producer-commit \"$PRODUCER_COMMIT\"", source)
        self.assertIn("--validator-commit \"$VALIDATOR_COMMIT\"", source)

    def test_active_multicbf_replay_is_h100_allocation_backed(self) -> None:
        source = (
            ROOT / "slurm/distal_three_ellipsoid_multicbf_e05.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("#SBATCH --exclude=worker-3", source)
        self.assertIn("tests.test_multilink_ellipsoid", source)
        self.assertIn("replay_distal_three_ellipsoid_multicbf.py", source)
        self.assertIn("vlsa_distal_three_ellipsoid_multicbf_e05.v1.json", source)
        self.assertIn("$ARCHIVED_TABLE1_RESULT", source)
        self.assertNotIn(">\"$ARCHIVED_TABLE1_RESULT\"", source)


if __name__ == "__main__":
    unittest.main()
