from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts" / "submit_vlsa_postpublication_analysis_v3.sh"
RUNNER = ROOT / "slurm" / "run_vlsa_postpublication_analysis_v3.sh"
SBATCH = ROOT / "slurm" / "vlsa_postpublication_analysis_v3.sbatch"
BUILDER = ROOT / "analysis" / "build_safelibero_postpublication_v3.py"


class PostpublicationV3LaunchContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.submit = SUBMIT.read_text(encoding="utf-8")
        cls.runner = RUNNER.read_text(encoding="utf-8")
        cls.sbatch = SBATCH.read_text(encoding="utf-8")
        cls.builder = BUILDER.read_text(encoding="utf-8")

    def test_exact_population_and_runtime_are_frozen(self) -> None:
        for text in (self.submit, self.runner):
            self.assertIn("POPULATION_ARRAY_JOB_ID=28609", text)
            self.assertIn("PUBLISHER_JOB_ID=28610", text)
            self.assertIn(
                "POPULATION_RUN_ID="
                "vlsa-table1-contact-authority-population-20260718a",
                text,
            )
            self.assertIn(
                "POPULATION_SOURCE_GIT_COMMIT="
                "1592aa59361f431ba96c6ddcbebcb596f6c20853",
                text,
            )
            self.assertIn(
                "ACCEPTED_V3_IMPLEMENTATION_COMMIT="
                "1c92370cb5b1278a4fcdda81332ec8eed9a49f8b",
                text,
            )

    def test_v2_must_be_terminal_and_dependency_is_exact(self) -> None:
        for text in (self.submit, self.runner):
            self.assertIn(
                'require_exact_terminal_job "$V2_ANALYSIS_JOB_ID" '
                '"$EXPECTED_V2_JOB_NAME"',
                text,
            )
            self.assertIn("afterok:$V2_ANALYSIS_JOB_ID", text)
        self.assertIn("--dependency=\"afterok:$V2_ANALYSIS_JOB_ID\"", self.submit)
        self.assertNotIn("afterany:$V2_ANALYSIS_JOB_ID", self.submit)

    def test_submit_is_shell_only_and_held_before_release(self) -> None:
        self.assertNotRegex(
            self.submit,
            re.compile(r"(^|[ /])python(?:3)?([ \"']|$)", re.MULTILINE),
        )
        self.assertIn("sbatch --parsable --hold", self.submit)
        self.assertIn('inspect_held_job "$job_id"', self.submit)
        self.assertIn('scontrol release "$job_id"', self.submit)
        self.assertLess(
            self.submit.index('inspect_held_job "$job_id"'),
            self.submit.index('scontrol release "$job_id"'),
        )

    def test_resources_are_exactly_cpu_only(self) -> None:
        expected = {
            "#SBATCH --partition=main",
            "#SBATCH --account=normal",
            "#SBATCH --qos=normal",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=4",
            "#SBATCH --mem=32G",
            "#SBATCH --time=04:00:00",
            "#SBATCH --exclude=worker-3",
            "#SBATCH --no-requeue",
        }
        for line in expected:
            self.assertIn(line, self.sbatch)
        self.assertNotRegex(self.sbatch, re.compile(r"^#SBATCH .*gpu", re.M))
        self.assertIn("CPU-only analysis must not receive CUDA", self.runner)
        self.assertIn("SLURM_JOB_GPUS", self.runner)
        self.assertIn("ReqTRES AllocTRES TresPerNode TresPerTask Gres", self.submit)

    def test_runner_rejects_login_worker3_dirty_or_wrong_commit(self) -> None:
        self.assertIn("worker-3|login*|login-restricted*", self.runner)
        self.assertIn("status --short --untracked-files=all", self.runner)
        self.assertIn("repository is dirty", self.runner)
        self.assertIn("rev-parse HEAD", self.runner)
        self.assertIn("merge-base --is-ancestor", self.runner)

    def test_every_scientific_input_is_content_bound(self) -> None:
        required = (
            "EXPECTED_V1_PUBLICATION_RECEIPT_SHA256",
            "EXPECTED_PREPUBLISH_RECEIPT_SHA256",
            "EXPECTED_SUMMARY_V2_SHA256",
            "EXPECTED_ANALYSIS_V2_RECEIPT_SHA256",
            "EXPECTED_V2_SUBMISSION_RECEIPT_SHA256",
            "EXPECTED_V2_RELEASE_RECEIPT_SHA256",
            "EXPECTED_CONFIG_SHA256",
            "EXPECTED_MANIFEST_SHA256",
            "EXPECTED_MANIFEST_RECEIPT_SHA256",
            "EXPECTED_V3_MODULE_SHA256",
            "EXPECTED_V3_BUILDER_SHA256",
            "EXPECTED_V3_RUNNER_SHA256",
            "EXPECTED_V3_SBATCH_SHA256",
        )
        for name in required:
            self.assertIn(f"${{{name}:?", self.runner)
            self.assertIn(f"${{{name}:?", self.submit)
        self.assertIn("EXPECTED_V3_SUBMIT_HELPER_SHA256", self.submit)

    def test_output_is_unused_and_terminal_receipt_is_builder_only(self) -> None:
        self.assertIn(
            "analysis-v3 output directory already exists", self.runner
        )
        self.assertIn(
            "analysis-v3 output directory already exists", self.submit
        )
        self.assertNotIn("analysis-v3-receipt.json", self.submit)
        self.assertNotIn("analysis-v3-receipt.json", self.runner)
        self.assertIn(
            'receipt_path = output_root / "analysis-v3-receipt.json"',
            self.builder,
        )
        self.assertGreater(
            self.builder.index("_write_atomic_json(receipt_path, receipt)"),
            self.builder.index("build_population_failure_artifacts_v3"),
        )

    def test_no_broad_or_destructive_cluster_commands(self) -> None:
        for text in (self.submit, self.runner):
            for forbidden in (
                "scancel",
                "pkill",
                "rm -rf",
                "rsync --delete",
            ):
                self.assertNotIn(forbidden, text)

    def test_control_receipts_bind_job_dependency_and_sources(self) -> None:
        for token in (
            "vlsa_postpublication_analysis_v3_submission_intent.v1",
            "vlsa_postpublication_analysis_v3_submission_receipt.v1",
            "vlsa_postpublication_analysis_v3_release_receipt.v1",
            "analysis_v2_job_id",
            "summary_v2_sha256",
            "analysis_v2_receipt_sha256",
            "v3_module_sha256",
            "held_before_release",
            "cpu_only",
            "source_git_commit",
            "output_dir",
            "cpus_per_task",
            "time_limit",
            "gpus",
        ):
            self.assertIn(token, self.submit)

    def test_result_receipt_binds_own_launch_receipts(self) -> None:
        for token in (
            "V3_SUBMISSION_RECEIPT",
            "V3_RELEASE_RECEIPT",
            "--analysis-v3-submission-receipt",
            "--analysis-v3-release-receipt",
            "--analysis-v3-job-id",
            "--expected-analysis-v3-submission-receipt-sha256",
            "--expected-analysis-v3-release-receipt-sha256",
        ):
            self.assertIn(token, self.runner)
        for token in (
            "_validate_v3_control_receipts",
            '"launch": launch_binding',
            '"submission_receipt": _artifact(submission_path)',
            '"release_receipt": _artifact(release_path)',
        ):
            self.assertIn(token, self.builder)

    def test_recovery_never_requires_unused_output_after_release(self) -> None:
        self.assertIn("initial_output_policy=allow_existing", self.submit)
        self.assertIn(
            'validate_frozen_inputs "$initial_output_policy"', self.submit
        )
        held = self.submit.index(
            'if [[ "$state" == PENDING && "$reason" == JobHeldUser ]]'
        )
        release = self.submit.index('scontrol release "$job_id"', held)
        require_unused = self.submit.index(
            "validate_frozen_inputs unused", held
        )
        self.assertLess(require_unused, release)
        self.assertIn(
            "validate_frozen_inputs allow_existing",
            self.submit[release:],
        )


if __name__ == "__main__":
    unittest.main()
