from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts import build_aegis_publisher_timeout_recovery_authority as recovery
from scripts import validate_aegis_run_artifacts as artifacts
from scripts.aegis_receipt_utils import (
    ReceiptError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_path,
    write_json_exclusive,
)


class PublisherTimeoutRecoveryAuthorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.experiment_root = self.root / "experiments"
        self.run_root = self.experiment_root / recovery.RUN_ID
        self.timed_out_attempt_root = (
            self.run_root
            / "publication-attempts"
            / f"job-{recovery.TIMED_OUT_PUBLISHER_JOB_ID}"
        )
        self.current_attempt_root = (
            self.run_root / "publication-attempts/job-30000"
        )
        self.timed_out_attempt_root.mkdir(parents=True)
        self.current_attempt_root.mkdir(parents=True)

        self.population_repo = self.root / "population"
        self._git("init", str(self.population_repo), cwd=self.root)
        self._configure_git(self.population_repo)
        (self.population_repo / "baseline.txt").write_text(
            "population\n", encoding="utf-8"
        )
        self._git("add", ".", cwd=self.population_repo)
        self._git("commit", "-m", "population", cwd=self.population_repo)
        self.population_commit = self._git(
            "rev-parse", "HEAD", cwd=self.population_repo
        )

        self.timed_out_repo = self.root / "timed-out-publisher"
        self._git(
            "clone", str(self.population_repo), str(self.timed_out_repo), cwd=self.root
        )
        self._configure_git(self.timed_out_repo)
        (self.timed_out_repo / "publisher.txt").write_text(
            "job-28940\n", encoding="utf-8"
        )
        self._git("add", ".", cwd=self.timed_out_repo)
        self._git("commit", "-m", "timed-out publisher", cwd=self.timed_out_repo)
        self.timed_out_commit = self._git(
            "rev-parse", "HEAD", cwd=self.timed_out_repo
        )

        self.recovery_repo = self.root / "recovery"
        self._git(
            "clone", str(self.timed_out_repo), str(self.recovery_repo), cwd=self.root
        )
        self._configure_git(self.recovery_repo)
        for relative in recovery.EXPECTED_RECOVERY_CHANGED_PATHS:
            path = self.recovery_repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"reviewed recovery: {relative}\n", encoding="utf-8")
        self._git("add", ".", cwd=self.recovery_repo)
        self._git("commit", "-m", "finalize-only recovery", cwd=self.recovery_repo)
        self.recovery_commit = self._git(
            "rev-parse", "HEAD", cwd=self.recovery_repo
        )

        prior_authority = {
            "schema_version": "vlsa_table1_publisher_retry_authority.v3",
            "status": "validated",
            "scientific_result": False,
            "run_id": recovery.RUN_ID,
            "population_array_job_id": recovery.POPULATION_ARRAY_JOB_ID,
            "publisher_slurm": {
                "job_id": recovery.TIMED_OUT_PUBLISHER_JOB_ID,
                "host": "worker-1",
                "dependency": f"afterany:{recovery.POPULATION_ARRAY_JOB_ID}",
            },
        }
        prior_authority["receipt_payload_sha256"] = sha256_bytes(
            canonical_json_bytes(prior_authority)
        )
        write_json_exclusive(
            self.timed_out_attempt_root / "publisher-retry-authority.json",
            prior_authority,
        )
        for relative in (
            "population-array-sacct.txt",
            "prepublish-validation.json",
            "population-summary.json",
            "failure-analysis/cases.jsonl",
            "failure-analysis/report.json",
            "failure-analysis/report.md",
            "gallery/index.html",
        ):
            path = self.timed_out_attempt_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"immutable {relative}\n", encoding="utf-8")
        self.expected_artifacts = {
            str(path.relative_to(self.timed_out_attempt_root)): (
                path.stat().st_size,
                sha256_path(path),
            )
            for path in self.timed_out_attempt_root.rglob("*")
            if path.is_file()
        }

        self.timeout_log = self.root / "publisher-28940.out"
        self.timeout_suffix = "job 28940 reached the exact time limit\n"
        self.timeout_log.write_text(
            "completed immutable publication artifacts\n" + self.timeout_suffix,
            encoding="utf-8",
        )
        self.accounting = self.current_attempt_root / "timed-out-sacct.txt"
        row = recovery.EXPECTED_TIMEOUT_ACCOUNTING
        self.accounting.write_text(
            "|".join(
                row[field]
                for field in (
                    "job_id",
                    "job_id_raw",
                    "state",
                    "exit_code",
                    "node",
                    "elapsed",
                    "start",
                    "end",
                )
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _git(*arguments: str, cwd: Path) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip()

    def _configure_git(self, repo: Path) -> None:
        self._git("config", "user.email", "test@example.com", cwd=repo)
        self._git("config", "user.name", "Test", cwd=repo)

    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            run_root=self.run_root,
            population_source_repo=self.population_repo,
            timed_out_publisher_source_repo=self.timed_out_repo,
            recovery_source_repo=self.recovery_repo,
            recovery_source_commit=self.recovery_commit,
            timed_out_attempt_root=self.timed_out_attempt_root,
            timed_out_publisher_log=self.timeout_log,
            timed_out_publisher_accounting=self.accounting,
            validator=self.recovery_repo
            / "scripts/validate_aegis_run_artifacts.py",
            recovery_authority_builder=self.recovery_repo
            / "scripts/build_aegis_publisher_timeout_recovery_authority.py",
            recovery_runner=self.recovery_repo
            / "slurm/run_aegis_population_publisher_timeout_finalize.sh",
            recovery_sbatch=self.recovery_repo
            / "slurm/aegis_population_publisher_timeout_finalize.sbatch",
            output=self.current_attempt_root
            / "publisher-timeout-recovery-authority.json",
        )

    def _patch_exact_fixture(self):
        return mock.patch.multiple(
            recovery,
            EXPERIMENT_ROOT=self.experiment_root,
            POPULATION_SOURCE_COMMIT=self.population_commit,
            TIMED_OUT_PUBLISHER_SOURCE_COMMIT=self.timed_out_commit,
            TIMED_OUT_PUBLISHER_LOG_SHA256=sha256_path(self.timeout_log),
            TIMED_OUT_PUBLISHER_LOG_BYTES=self.timeout_log.stat().st_size,
            EXPECTED_TIMEOUT_LOG_SUFFIX=self.timeout_suffix,
            EXPECTED_ATTEMPT_ARTIFACTS=self.expected_artifacts,
        )

    def test_binds_exact_timeout_to_finalize_only_recovery(self) -> None:
        with self._patch_exact_fixture(), mock.patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": "30000",
                "SLURM_JOB_DEPENDENCY": (
                    f"afterany:{recovery.TIMED_OUT_PUBLISHER_JOB_ID}"
                ),
                "SLURMD_NODENAME": "worker-2",
            },
            clear=False,
        ):
            receipt = recovery.build_receipt(self._args())

        self.assertEqual(receipt["status"], "validated")
        self.assertEqual(receipt["recovery_scope"], "population_finalize_only")
        self.assertTrue(receipt["preserves_immutable_result_tree"])
        self.assertFalse(receipt["permits_inference_or_simulation"])
        self.assertEqual(
            receipt["timed_out_publisher"]["publisher_slurm"]["state"],
            "TIMEOUT",
        )
        self.assertEqual(
            receipt["recovery_publisher_slurm"]["dependency"],
            f"afterany:{recovery.TIMED_OUT_PUBLISHER_JOB_ID}",
        )
        self.assertEqual(
            receipt["recovery_source"]["changed_paths"],
            list(recovery.EXPECTED_RECOVERY_CHANGED_PATHS),
        )

    def test_rejects_any_mutation_of_timed_out_attempt(self) -> None:
        target = self.timed_out_attempt_root / "population-summary.json"
        target.write_text("mutated\n", encoding="utf-8")
        with self._patch_exact_fixture(), mock.patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": "30000",
                "SLURM_JOB_DEPENDENCY": (
                    f"afterany:{recovery.TIMED_OUT_PUBLISHER_JOB_ID}"
                ),
                "SLURMD_NODENAME": "worker-2",
            },
            clear=False,
        ), self.assertRaisesRegex(ReceiptError, "size differs|SHA-256 differs"):
            recovery.build_receipt(self._args())


class PublisherTimeoutRecoveryValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run_root = self.root / "run"
        self.authority_path = (
            self.run_root
            / "publication-attempts/job-30000"
            / "publisher-timeout-recovery-authority.json"
        )
        self.authority_path.parent.mkdir(parents=True)
        authority = {
            "schema_version": (
                "vlsa_table1_publisher_timeout_recovery_authority.v1"
            ),
            "status": "validated",
            "scientific_result": False,
            "run_id": recovery.RUN_ID,
            "population_array_job_id": recovery.POPULATION_ARRAY_JOB_ID,
            "preserves_immutable_result_tree": True,
            "permits_inference_or_simulation": False,
            "recovery_scope": "population_finalize_only",
            "timed_out_publisher": {
                "publisher_slurm": {
                    "job_id": recovery.TIMED_OUT_PUBLISHER_JOB_ID,
                    "host": "worker-1",
                    "dependency": (
                        f"afterany:{recovery.POPULATION_ARRAY_JOB_ID}"
                    ),
                    "state": "TIMEOUT",
                    "exit_code": "0:0",
                }
            },
            "recovery_publisher_slurm": {
                "job_id": "30000",
                "host": "worker-2",
                "dependency": (
                    f"afterany:{recovery.TIMED_OUT_PUBLISHER_JOB_ID}"
                ),
            },
        }
        authority["receipt_payload_sha256"] = sha256_bytes(
            canonical_json_bytes(authority)
        )
        write_json_exclusive(self.authority_path, authority)
        self.timed_out_retry_authority = {
            "publisher_slurm": {
                "job_id": recovery.TIMED_OUT_PUBLISHER_JOB_ID,
                "host": "worker-1",
                "dependency": f"afterany:{recovery.POPULATION_ARRAY_JOB_ID}",
            }
        }
        self.environment = {
            "SLURM_JOB_ID": "30000",
            "SLURM_JOB_DEPENDENCY": (
                f"afterany:{recovery.TIMED_OUT_PUBLISHER_JOB_ID}"
            ),
            "SLURMD_NODENAME": "worker-2",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_accepts_only_exact_recovery_and_prior_retry_bindings(self) -> None:
        publisher_slurm, record = (
            artifacts.validate_publisher_timeout_recovery_authority(
                self.authority_path,
                run_root=self.run_root,
                expected_run_id=recovery.RUN_ID,
                expected_population_array_job_id=(
                    recovery.POPULATION_ARRAY_JOB_ID
                ),
                timed_out_retry_authority=self.timed_out_retry_authority,
                environment=self.environment,
            )
        )
        self.assertEqual(
            publisher_slurm,
            {
                "job_id": "30000",
                "host": "worker-2",
                "dependency": (
                    f"afterany:{recovery.TIMED_OUT_PUBLISHER_JOB_ID}"
                ),
            },
        )
        self.assertEqual(record["sha256"], sha256_path(self.authority_path))

    def test_rejects_wrong_current_dependency(self) -> None:
        with self.assertRaisesRegex(ReceiptError, "dependency"):
            artifacts.validate_publisher_timeout_recovery_authority(
                self.authority_path,
                run_root=self.run_root,
                expected_run_id=recovery.RUN_ID,
                expected_population_array_job_id=(
                    recovery.POPULATION_ARRAY_JOB_ID
                ),
                timed_out_retry_authority=self.timed_out_retry_authority,
                environment={
                    **self.environment,
                    "SLURM_JOB_DEPENDENCY": (
                        f"afterok:{recovery.TIMED_OUT_PUBLISHER_JOB_ID}"
                    ),
                },
            )

    def test_rejects_wrong_timed_out_retry_binding(self) -> None:
        with self.assertRaisesRegex(ReceiptError, "timed-out retry"):
            artifacts.validate_publisher_timeout_recovery_authority(
                self.authority_path,
                run_root=self.run_root,
                expected_run_id=recovery.RUN_ID,
                expected_population_array_job_id=(
                    recovery.POPULATION_ARRAY_JOB_ID
                ),
                timed_out_retry_authority={
                    "publisher_slurm": {
                        **self.timed_out_retry_authority["publisher_slurm"],
                        "job_id": "28921",
                    }
                },
                environment=self.environment,
            )


if __name__ == "__main__":
    unittest.main()
