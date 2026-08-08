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

        runner = (
            ROOT / "scripts/replay_distal_three_ellipsoid_rollout_multicbf.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"terminal_filter_record"', runner)
        self.assertIn("None if not filter_records else filter_records[-1]", runner)

    def test_active_multicbf_video_replays_accepted_ledger_on_h100(self) -> None:
        source = (
            ROOT / "slurm/render_distal_three_ellipsoid_multicbf_video.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("#SBATCH --exclude=worker-3", source)
        self.assertIn("render_distal_three_ellipsoid_multicbf_video.py", source)
        self.assertIn("$ACCEPTED_RESULT", source)
        self.assertIn("$EXPECTED_RESULT_SHA256", source)
        self.assertIn("IMAGEIO_FFMPEG_EXE", source)
        self.assertNotIn(">\"$ACCEPTED_RESULT\"", source)

    def test_rollout_multicbf_is_single_h100_and_uses_cloned_steps(self) -> None:
        source = (
            ROOT / "slurm/distal_three_ellipsoid_rollout_multicbf_e05.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("#SBATCH --cpus-per-task=8", source)
        self.assertIn("#SBATCH --mem=64G", source)
        self.assertIn("#SBATCH --exclude=worker-3", source)
        self.assertIn("tests.test_multilink_ellipsoid", source)
        self.assertIn("replay_distal_three_ellipsoid_rollout_multicbf.py", source)
        self.assertIn(
            "vlsa_distal_three_ellipsoid_rollout_multicbf_e05.v1.json", source
        )
        self.assertIn("$ARCHIVED_TABLE1_RESULT", source)
        self.assertNotIn(">\"$ARCHIVED_TABLE1_RESULT\"", source)

    def test_predictive_flow_is_single_h100_and_runs_inside_sampler(self) -> None:
        source = (
            ROOT / "slurm/predictive_flow_guidance_e05.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("#SBATCH --cpus-per-task=8", source)
        self.assertIn("#SBATCH --mem=64G", source)
        self.assertIn("#SBATCH --exclude=worker-0,worker-3", source)
        self.assertNotIn("#SBATCH --array", source)
        self.assertIn("pi05_libero", source)
        self.assertIn("preflight_predictive_flow_projection.py", source)
        self.assertIn("evaluate_predictive_flow_guidance_e05.py", source)
        self.assertIn("$ARCHIVED_TABLE1_RESULT", source)
        self.assertNotIn(">\"$ARCHIVED_TABLE1_RESULT\"", source)

        sampler = (ROOT / "openpi/src/openpi/models/pi0.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("project_action_xyz_halfspaces", sampler)
        self.assertIn("x_next = project_action_xyz_halfspaces", sampler)
        self.assertIn("flow_guidance_rows is not None", sampler)

        runner = (
            ROOT / "scripts/evaluate_predictive_flow_guidance_e05.py"
        ).read_text(encoding="utf-8")
        self.assertIn("identify_osc_clearance_model", runner)
        self.assertIn("exact_trajectory_verification", runner)
        self.assertIn("initial_policy_action_chunk_sha256", runner)


if __name__ == "__main__":
    unittest.main()
