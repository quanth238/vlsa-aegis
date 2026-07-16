"""Fail-closed checks for the TRL-00A held Slurm publication transaction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest

from publish_crfs_r05a_reference_trajectory_lift_canary import (
    _failure_receipt,
    _validate_allocation_tests,
)


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "scripts/hpc/submit_r05a_reference_trajectory_lift_canary.sh"
WORKLOAD = ROOT / "scripts/hpc/run_r05a_reference_trajectory_lift_workload.sh"
GPU_WRAPPER = ROOT / "scripts/hpc/run_r05a_reference_trajectory_lift_canary.sh"
CPU_WRAPPER = ROOT / "scripts/hpc/validate_r05a_reference_trajectory_lift_canary.sh"
GPU_SBATCH = ROOT / "slurm/r05a_reference_trajectory_lift_canary_h100.sbatch"
CPU_SBATCH = ROOT / "slurm/r05a_reference_trajectory_lift_canary_validate_cpu.sbatch"
PUBLISHER = ROOT / "main/publish_crfs_r05a_reference_trajectory_lift_canary.py"
RESULT_SCHEMA = (
    ROOT / "schemas/r05a-reference-trajectory-lift-canary-envelope.schema.json"
)
RECEIPT_SCHEMA = (
    ROOT
    / "schemas/r05a-reference-trajectory-lift-canary-publication-receipt.schema.json"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class ReferenceTrajectoryLiftHpcContractTests(unittest.TestCase):
    def test_shell_entrypoints_parse_and_required_scripts_are_executable(self) -> None:
        for path in (
            SUBMITTER,
            WORKLOAD,
            GPU_WRAPPER,
            CPU_WRAPPER,
            GPU_SBATCH,
            CPU_SBATCH,
        ):
            result = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=f"{path}: {result.stderr}")
        for path in (SUBMITTER, WORKLOAD, GPU_WRAPPER, CPU_WRAPPER):
            self.assertTrue(path.stat().st_mode & stat.S_IXUSR, path)

    def test_exact_gpu_and_cpu_resources_are_frozen(self) -> None:
        gpu = _text(GPU_SBATCH)
        cpu = _text(CPU_SBATCH)
        for token in (
            "#SBATCH --job-name=crfs-r05a-trl00a",
            "#SBATCH --partition=main",
            "#SBATCH --account=normal",
            "#SBATCH --qos=normal",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=64G",
            "#SBATCH --time=00:30:00",
            "#SBATCH --array=0-0%1",
            "#SBATCH --nodelist=worker-1",
            "#SBATCH --no-requeue",
        ):
            self.assertIn(token, gpu)
        self.assertNotIn("--gres", cpu)
        for token in (
            "#SBATCH --job-name=crfs-r05a-trl00a-val",
            "#SBATCH --cpus-per-task=2",
            "#SBATCH --mem=8G",
            "#SBATCH --time=00:15:00",
            "#SBATCH --no-requeue",
        ):
            self.assertIn(token, cpu)

    def test_only_complete_finite_18_request_branch_can_reach_publisher(self) -> None:
        workload = _text(WORKLOAD)
        publisher = _text(PUBLISHER)
        for token in (
            ".request_ledger.complete_finite_exact_policy_request_count == 18",
            '.payload_type == "r05a_reference_trajectory_lift_raw_payload"',
            '.branch == "finite"',
            "length==18",
            "trl00a-raw-payload.json",
            "trl00a-tensors.npz",
            "request-ledger.json",
        ):
            self.assertIn(token, workload)
        for token in (
            "validate_reference_trajectory_lift_npz",
            '"raw payload is not the complete finite branch"',
            '"raw payload contains a terminal or nonfinite branch"',
            '"request_count") == 18',
            '"apparatus_inconclusive"',
        ):
            self.assertIn(token, publisher)
        combined = workload + publisher
        self.assertNotIn("typed_terminal", combined)
        self.assertNotIn("16-call", combined)
        self.assertNotIn("request_count == 16", combined)

    def test_exact_13_phase_ledger_and_split_interpreter_tests(self) -> None:
        workload = _text(WORKLOAD)
        self.assertIn(".request_ledger.complete_ordered_phases == [", workload)
        phases = (
            "compiled_frozen_pre",
            "eager_source_trace_pre",
            "eager_normalized_final_pre",
            "zero_schedule_pre",
            "arm_a_equal_split",
            "budgeted_lift_generation",
            "budgeted_lift_replay",
            "raw_lift_generation",
            "raw_lift_replay",
            "zero_schedule_post",
            "eager_normalized_final_post",
            "eager_source_trace_post",
            "compiled_frozen_post",
        )
        for phase in phases:
            self.assertIn(f'"name":"{phase}"', workload)
        self.assertIn('"$OPENPI_PYTHON" -m unittest discover', workload)
        self.assertIn("'test_reference_trajectory_lift.py'", workload)
        self.assertIn('"$LIBERO_PYTHON" -m unittest', workload)
        self.assertIn("tests.test_r05a_reference_trajectory_lift_validation", workload)
        self.assertIn("[openpi-python]", workload)
        self.assertIn("[libero-python]", workload)

    def test_combined_allocation_log_requires_both_zero_skip_suites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allocation-focused-tests.log"
            path.write_text(
                "[openpi-python]\nRan 7 tests in 0.100s\n\nOK\n"
                "[libero-python]\nRan 19 tests in 0.200s\n\nOK\n",
                encoding="utf-8",
            )
            summary = _validate_allocation_tests(path)
            self.assertEqual(summary["tests_run"], 26)
            self.assertEqual(summary["skipped"], 0)
            path.write_text(
                "[openpi-python]\nRan 7 tests in 0.100s\n\nOK\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "interpreter groups"):
                _validate_allocation_tests(path)

    def test_gpu_never_publishes_and_cpu_is_the_only_atomic_publisher(self) -> None:
        workload = _text(WORKLOAD)
        publisher = _text(PUBLISHER)
        self.assertIn('test ! -e "$RESULT"', workload)
        self.assertNotIn('mv "$', workload.split('test ! -e "$RESULT"', 1)[1])
        self.assertIn("os.replace(candidate, output)", publisher)
        self.assertIn("_atomic_json(receipt, receipt_value)", publisher)
        self.assertIn("output.unlink(missing_ok=True)", publisher)
        self.assertIn("result_sha256", publisher)

    def test_failure_receipt_cannot_create_a_scientific_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "trl-run"
            case_dir = run_root / "crfs-1069f29a8d76463a"
            case_dir.mkdir(parents=True)
            output = case_dir / "results.json"
            receipt = run_root / "cpu-afterany-validation.json"
            args = argparse.Namespace(
                run_root=str(run_root),
                output=str(output),
                receipt=str(receipt),
                source_job_id="123",
                publisher_job_id="124",
            )
            _failure_receipt(args, ValueError("nonfinite raw leaf"))
            value = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "apparatus_inconclusive")
            self.assertEqual(value["outcome"], "apparatus_inconclusive")
            self.assertFalse(value["published"])
            self.assertFalse(value["passed"])
            self.assertIsNone(value["result_sha256"])
            self.assertFalse(output.exists())

    def test_shell_fallback_receipt_carries_required_schema_fields(self) -> None:
        value = _text(CPU_WRAPPER)
        for token in (
            "status:\"apparatus_inconclusive\"",
            "outcome:\"apparatus_inconclusive\"",
            "run_id:$run",
            'case_id:"crfs-1069f29a8d76463a"',
            "git_commit:$commit",
            'source_node:"worker-1"',
            'dependency:("afterany:"+$source)',
            "result_sha256:null",
            "published:false",
            "passed:false",
            "collision_or_progress_claim_allowed:false",
            "raw_authority_is_minimum_required_claim_allowed:false",
        ):
            self.assertIn(token, value)

    def test_source_current_pairing_and_external_sources_are_independent(self) -> None:
        publisher = _text(PUBLISHER)
        for token in (
            '"policy_observation_source": r02_pairing.get("policy_observation")',
            '"branch_snapshot_source": r02_pairing.get("branch_snapshot")',
            '"eager_actions_source": r02_pairing.get("eager_actions")',
            '"eager_trace_source": r02_pairing.get("eager_trace")',
            'records["eager_actions_before_current"]',
            'records["eager_actions_after_current"]',
            'records["eager_trace_before_current"]',
            'records["eager_trace_after_current"]',
            '"checkpoint_model_sha256"',
            '"normalization_asset_sha256"',
            '"af00a_tensor_sha256"',
            '"af00a_results_sha256"',
            "_sha256(path) == observed[key]",
        ):
            self.assertIn(token, publisher)
        self.assertNotIn("all(source_payload", publisher)

    def test_held_transaction_is_fingerprinted_before_single_release(self) -> None:
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
        self.assertEqual(value.count("sbatch --parsable"), 2)
        self.assertEqual(value.count("scontrol release"), 1)
        self.assertIn("dependency=afterany:$gpu_job_id", value)
        self.assertNotIn("scancel", value)

    def test_release_is_fail_closed_direct_child_and_exact_two_path_diff(self) -> None:
        for path in (SUBMITTER, GPU_WRAPPER, CPU_WRAPPER):
            value = _text(path)
            self.assertIn("accepted_implementation_commit", value)
            self.assertIn("rev-list --parents -n 1", value)
            self.assertIn(
                "configs/experiments/r05a_reference_trajectory_lift_canary.json",
                value,
            )
            self.assertIn(
                "release-reference-trajectory-lift-canary.md",
                value,
            )
        submitter = _text(SUBMITTER)
        self.assertIn("PARENT_CONFIG=", submitter)
        self.assertIn(".execution_release == null", submitter)
        self.assertIn("RELEASE_DIFF=", submitter)

    def test_source_contract_binds_new_sampler_publisher_and_schemas(self) -> None:
        for path in (SUBMITTER, GPU_WRAPPER):
            value = _text(path)
            for token in (
                "openpi/src/openpi/models_pytorch/crfs_reference_trajectory.py",
                "main/publish_crfs_r05a_reference_trajectory_lift_canary.py",
                "r05a-reference-trajectory-lift-canary-envelope.schema.json",
                "r05a-reference-trajectory-lift-canary-publication-receipt.schema.json",
            ):
                self.assertIn(token, value)

    def test_login_control_plane_does_not_hash_multi_gib_checkpoint(self) -> None:
        value = _text(SUBMITTER)
        self.assertNotIn('sha256sum "$checkpoint"', value)
        self.assertIn('test -f "$checkpoint"', value)
        self.assertIn(
            "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",
            value,
        )

    def test_schemas_are_explicit_valid_json_and_have_stable_digest_shape(self) -> None:
        for path in (RESULT_SCHEMA, RECEIPT_SCHEMA):
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                value["$schema"],
                "https://json-schema.org/draft/2020-12/schema",
            )
            self.assertFalse(value["additionalProperties"])
            self.assertIsInstance(value["required"], list)
            self.assertGreater(len(value["required"]), 10)
            self.assertRegex(hashlib.sha256(path.read_bytes()).hexdigest(), r"^[0-9a-f]{64}$")
        result = json.loads(RESULT_SCHEMA.read_text(encoding="utf-8"))
        receipt = json.loads(RECEIPT_SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(result["properties"]["branch"]["const"], "finite")
        self.assertEqual(
            result["$defs"]["validation"]["properties"]["request_count"]["const"],
            18,
        )
        self.assertIn("apparatus_inconclusive", receipt["properties"]["outcome"]["enum"])


if __name__ == "__main__":
    unittest.main()
