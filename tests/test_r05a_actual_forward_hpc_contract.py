"""Static fail-closed checks for the AF-00A held Slurm transaction."""

from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "scripts/hpc/submit_r05a_actual_forward_canary.sh"
GPU_WRAPPER = ROOT / "scripts/hpc/run_r05a_actual_forward_canary.sh"
CPU_WRAPPER = ROOT / "scripts/hpc/validate_r05a_actual_forward_canary.sh"
GPU_SBATCH = ROOT / "slurm/r05a_actual_forward_canary_h100.sbatch"
CPU_SBATCH = ROOT / "slurm/r05a_actual_forward_canary_validate_cpu.sbatch"
ARRAY_IDENTITY_HELPER = (
    ROOT / "scripts/hpc/lib/slurm_array_task_record_identity.sh"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class ActualForwardHpcContractTests(unittest.TestCase):
    def test_shell_entrypoints_parse(self) -> None:
        for path in (
            SUBMITTER,
            GPU_WRAPPER,
            CPU_WRAPPER,
            GPU_SBATCH,
            CPU_SBATCH,
            ARRAY_IDENTITY_HELPER,
        ):
            result = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=f"{path}: {result.stderr}")

    def test_release_only_commit_is_exactly_two_paths(self) -> None:
        value = _text(SUBMITTER)
        expected_adr = "docs/decisions/0055-release-behavior-tested-actual-forward-canary.md"
        self.assertIn(f"RELEASE_ADR={expected_adr}", value)
        self.assertIn('EXPECTED_RELEASE_DIFF=$(printf \'%s\\n\' "$CONFIG" "$RELEASE_ADR"', value)
        self.assertIn("remote_release_diff=", value)
        self.assertIn(expected_adr, value)
        self.assertNotIn("0053-release-corrected-actual-forward-cem-canary.md", value)
        release_contract = value.split(".execution_release.allowed_release_diff_paths == [", 1)[1].split("]", 1)[0]
        self.assertIn("configs/experiments/r05a_actual_forward_canary.json", release_contract)
        self.assertIn(expected_adr, release_contract)
        self.assertNotIn("DECISIONS.md", release_contract)

    def test_both_sbatch_calls_export_exact_transaction_environment(self) -> None:
        value = _text(SUBMITTER)
        self.assertEqual(value.count("sbatch --parsable"), 2)
        self.assertEqual(value.count("--export=ALL"), 2)
        for token in (
            'RUN_ID="$run_id"',
            'EXPECTED_GIT_COMMIT="$expected_commit"',
            'REMOTE_REPO="$remote_repo"',
            'SOURCE_CONTRACT="$source_contract"',
            'SUBMISSION="$submission_receipt"',
            'RELEASE_FINGERPRINT="$release_fingerprint"',
            'RELEASE_FINGERPRINT_SHA256_FILE="$release_fingerprint_sha_file"',
        ):
            self.assertIn(token, value)
        self.assertIn('SOURCE_JOB_ID="$gpu_job_id"', value)
        self.assertIn('EXPECTED_SOURCE_CONTRACT_SHA256="$source_contract_sha"', value)
        self.assertIn('EXPECTED_HELD_GPU_SUBMISSION_SHA256="$held_gpu_submission_sha"', value)

    def test_numeric_ids_are_receipted_before_any_post_sbatch_inspection(self) -> None:
        value = _text(SUBMITTER)
        gpu_receipt = value.index('mv "$provisional_gpu_tmp" "$provisional_gpu_receipt"')
        gpu_task_query = value.index('gpu_task_record=$(scontrol show job "${gpu_job_id}_0" -o)')
        cpu_receipt = value.index('mv "$provisional_cpu_tmp" "$provisional_cpu_receipt"')
        cpu_query = value.index('cpu_record=$(scontrol show job "$cpu_job_id" -o)')
        self.assertLess(gpu_receipt, gpu_task_query)
        self.assertLess(cpu_receipt, cpu_query)
        self.assertIn("automatic_cancellation_allowed:false", value)
        self.assertNotIn("scancel", value)

    def test_login_control_plane_does_not_hash_the_multi_gib_checkpoint(self) -> None:
        value = _text(SUBMITTER)
        self.assertNotIn('sha256sum "$checkpoint"', value)
        self.assertIn('test -f "$checkpoint"', value)
        self.assertIn("988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed", value)

    def test_exact_task_parent_resources_and_afterany_are_validated(self) -> None:
        value = _text(SUBMITTER)
        for token in (
            'scontrol show job "${gpu_job_id}_0" -o',
            'scontrol show job "$gpu_job_id" -o',
            '"ArrayTaskThrottle=1"',
            'case "$array_task_id" in 0|0%1)',
            'require_single_tres "$task_record" cpu 8',
            'require_single_memory_tres "$task_record" 65536',
            'require_single_tres "$task_record" node 1',
            'require_single_tres "$task_record" gres/gpu 1',
            'require_single_tres "$record" cpu 2',
            'require_single_memory_tres "$record" 8192',
            'require_single_tres "$record" node 1',
            'test "$observed_dependency" = "afterany:$gpu_id"',
        ):
            self.assertIn(token, value)

    def test_vinuni_normalized_array_task_job_id_is_behaviorally_validated(self) -> None:
        real_parent_form = (
            "JobId=28279 ArrayJobId=28279 ArrayTaskId=0 ArrayTaskThrottle=1 "
            "JobState=PENDING Reason=JobHeldUser ReqNodeList=worker-1 "
            "ReqTRES=cpu=8,mem=64G,node=1,billing=8,gres/gpu=1"
        )
        command = (
            f'. "{ARRAY_IDENTITY_HELPER}"; '
            'crfs_validate_exact_array_task_identity "$1" 28279 0'
        )
        accepted = (
            real_parent_form,
            real_parent_form.replace("JobId=28279 ", "JobId=28279_0 ", 1),
        )
        rejected = (
            real_parent_form.replace("JobId=28279 ", "JobId=99999 ", 1),
            real_parent_form.replace("ArrayJobId=28279 ", "ArrayJobId=99999 ", 1),
            real_parent_form.replace("ArrayTaskId=0 ", "ArrayTaskId=1 ", 1),
            real_parent_form.replace("JobId=28279 ", "", 1),
            real_parent_form + " JobId=28279",
        )
        for record in accepted:
            self.assertEqual(
                subprocess.run(["bash", "-c", command, "_", record]).returncode,
                0,
            )
        for record in rejected:
            self.assertNotEqual(
                subprocess.run(["bash", "-c", command, "_", record]).returncode,
                0,
            )
        for path in (SUBMITTER, GPU_WRAPPER):
            value = _text(path)
            self.assertIn("slurm_array_task_record_identity.sh", value)
            self.assertIn("crfs_validate_exact_array_task_identity", value)

    def test_transaction_is_fingerprinted_then_released_exactly_once(self) -> None:
        value = _text(SUBMITTER)
        ordered = (
            "sbatch --parsable --hold",
            'mv "$provisional_gpu_tmp" "$provisional_gpu_receipt"',
            'mv "$held_tmp" "$held_gpu_submission"',
            'mv "$source_tmp" "$source_contract"',
            'mv "$provisional_cpu_tmp" "$provisional_cpu_receipt"',
            'mv "$submission_tmp" "$submission_receipt"',
            "# Final pre-release fingerprint.",
            'mv "$fingerprint_tmp" "$release_fingerprint"',
            'scontrol release "$gpu_job_id"',
        )
        offsets = [value.index(token) for token in ordered]
        self.assertEqual(offsets, sorted(offsets))
        self.assertEqual(value.count("scontrol release"), 1)
        for binding in (
            "provisional_gpu_job_id",
            "held_gpu_submission",
            "source_contract",
            "provisional_cpu_job_id",
            "atomic_submission",
            "held_gpu_task_job_record",
            "held_gpu_parent_job_record",
            "cpu_afterany_job_record",
        ):
            self.assertIn(binding, value)
        self.assertIn("released_at_fingerprint_time:false", value)
        self.assertIn("final-pre-release-fingerprint.sha256", value)

    def test_both_wrappers_validate_submission_and_release_fingerprint(self) -> None:
        for path in (GPU_WRAPPER, CPU_WRAPPER):
            value = _text(path)
            for token in (
                "SUBMISSION",
                "RELEASE_FINGERPRINT",
                "RELEASE_FINGERPRINT_SHA256_FILE",
                "EXPECTED_SUBMISSION_SHA256",
                "r05a_actual_forward_canary_final_pre_release_fingerprint",
                "held_gpu_task_job_record",
                "held_gpu_parent_job_record",
                "cpu_afterany_job_record",
            ):
                self.assertIn(token, value)
        cpu = _text(CPU_WRAPPER)
        for option in (
            "--held-gpu-submission",
            "--expected-held-gpu-submission-sha256",
            "--submission",
            "--expected-submission-sha256",
            "--release-fingerprint",
            "--expected-release-fingerprint-sha256",
        ):
            self.assertIn(option, cpu)

    def test_sbatch_files_remain_exact_and_cpu_only_publisher_has_no_gpu(self) -> None:
        gpu = _text(GPU_SBATCH)
        cpu = _text(CPU_SBATCH)
        for token in (
            "#SBATCH --partition=main",
            "#SBATCH --account=normal",
            "#SBATCH --qos=normal",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=64G",
            "#SBATCH --time=02:00:00",
            "#SBATCH --array=0-0%1",
            "#SBATCH --nodelist=worker-1",
            "#SBATCH --no-requeue",
        ):
            self.assertIn(token, gpu)
        self.assertNotIn("--gres", cpu)
        self.assertIn("#SBATCH --cpus-per-task=2", cpu)
        self.assertIn("#SBATCH --mem=8G", cpu)
        self.assertIn("#SBATCH --time=00:15:00", cpu)
        self.assertIn("#SBATCH --no-requeue", cpu)


if __name__ == "__main__":
    unittest.main()
