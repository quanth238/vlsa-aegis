from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from scripts import build_aegis_publisher_retry_authority as retry
from scripts import validate_aegis_run_artifacts as artifacts
from scripts.aegis_receipt_utils import (
    ReceiptError,
    sha256_path,
    write_json_exclusive,
)


RUN_ID = "vlsa-table1-contact-authority-population-20260718a"
POPULATION_JOB_ID = "28609"
PREVIOUS_PUBLISHER_JOB_ID = "28610"
CURRENT_PUBLISHER_JOB_ID = "30000"


class PublisherRetryAuthorityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.population_repo = root / "population-source"
        self.publisher_repo = root / "publisher-source"
        self.run_root = root / "run"
        self.attempt_root = (
            self.run_root
            / "publication-attempts"
            / f"job-{CURRENT_PUBLISHER_JOB_ID}"
        )
        self.attempt_root.mkdir(parents=True)
        self._git("init", str(self.population_repo), cwd=root)
        self._git(
            "config",
            "user.email",
            "test@example.com",
            cwd=self.population_repo,
        )
        self._git(
            "config", "user.name", "Test", cwd=self.population_repo
        )
        for relative in (
            "scripts/validate_aegis_run_artifacts.py",
            "slurm/run_aegis_population_publisher.sh",
            "tests/test_aegis_population_streaming_validation.py",
            "tests/test_aegis_slurm_contract.py",
        ):
            path = self.population_repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("population\n", encoding="utf-8")
        self._git("add", ".", cwd=self.population_repo)
        self._git("commit", "-m", "population", cwd=self.population_repo)
        self.population_commit = self._git(
            "rev-parse", "HEAD", cwd=self.population_repo
        )
        self._git(
            "clone",
            str(self.population_repo),
            str(self.publisher_repo),
            cwd=root,
        )
        self._git(
            "config",
            "user.email",
            "test@example.com",
            cwd=self.publisher_repo,
        )
        self._git(
            "config", "user.name", "Test", cwd=self.publisher_repo
        )
        for relative in retry.EXPECTED_CHANGED_PATHS:
            path = self.publisher_repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative == "scripts/validate_aegis_run_artifacts.py":
                path.write_bytes(Path(artifacts.__file__).read_bytes())
            else:
                path.write_text("publisher\n", encoding="utf-8")
        self._git("add", ".", cwd=self.publisher_repo)
        self._git("commit", "-m", "publisher", cwd=self.publisher_repo)
        self.publisher_commit = self._git(
            "rev-parse", "HEAD", cwd=self.publisher_repo
        )

        self.previous_log = root / "publisher.log"
        self.previous_log.write_text(
            retry.EXPECTED_FAILURE_LOG,
            encoding="utf-8",
        )
        self.previous_failure = root / "publisher-failure.json"
        self.previous_failure.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "vlsa_table1_population_publisher_failure.v1"
                    ),
                    "status": "apparatus_failure",
                    "scientific_result": False,
                    "run_id": RUN_ID,
                    "population_array_job_id": POPULATION_JOB_ID,
                    "publisher_job_id": PREVIOUS_PUBLISHER_JOB_ID,
                    "host": "worker-1",
                    "failure_stage": "population_prepublish_validation",
                    "exit_code": 2,
                },
                sort_keys=True,
                indent=2,
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

    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            run_root=self.run_root,
            run_id=RUN_ID,
            population_array_job_id=POPULATION_JOB_ID,
            population_source_repo=self.population_repo,
            population_source_commit=self.population_commit,
            publisher_source_repo=self.publisher_repo,
            publisher_source_commit=self.publisher_commit,
            previous_publisher_job_id=PREVIOUS_PUBLISHER_JOB_ID,
            previous_publisher_log=self.previous_log,
            expected_previous_publisher_log_sha256=sha256_path(
                self.previous_log
            ),
            previous_publisher_failure=self.previous_failure,
            expected_previous_publisher_failure_sha256=sha256_path(
                self.previous_failure
            ),
            validator=(
                self.publisher_repo
                / "scripts/validate_aegis_run_artifacts.py"
            ),
            publisher_authority_builder=(
                self.publisher_repo
                / "scripts/build_aegis_publisher_retry_authority.py"
            ),
            publisher_runner=(
                self.publisher_repo
                / "slurm/run_aegis_population_publisher.sh"
            ),
            publisher_sbatch=(
                self.publisher_repo
                / "slurm/aegis_population_publisher_retry.sbatch"
            ),
            output=self.attempt_root / "publisher-retry-authority.json",
        )

    def test_binds_clean_publication_only_retry(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": CURRENT_PUBLISHER_JOB_ID,
                "SLURM_JOB_DEPENDENCY": f"afterany:{POPULATION_JOB_ID}",
                "SLURMD_NODENAME": "worker-2",
            },
            clear=False,
        ):
            receipt = retry.build_receipt(self._args())

        self.assertEqual(receipt["status"], "validated")
        self.assertTrue(receipt["preserves_immutable_result_tree"])
        self.assertFalse(receipt["permits_inference_or_simulation"])
        self.assertEqual(
            receipt["publisher_source"]["changed_paths"],
            list(retry.EXPECTED_CHANGED_PATHS),
        )
        self.assertEqual(
            receipt["recovery_from"]["publisher_job_id"],
            PREVIOUS_PUBLISHER_JOB_ID,
        )
        write_json_exclusive(self._args().output, receipt)
        with mock.patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": CURRENT_PUBLISHER_JOB_ID,
                "SLURM_JOB_DEPENDENCY": f"afterany:{POPULATION_JOB_ID}",
                "SLURMD_NODENAME": "worker-2",
            },
            clear=False,
        ):
            validated = artifacts.validate_publisher_retry_authority(
                self._args().output,
                run_root=self.run_root,
                expected_population_commit=self.population_commit,
                expected_run_id=RUN_ID,
                expected_population_array_job_id=POPULATION_JOB_ID,
            )
        self.assertEqual(
            validated["publisher_source_git_commit"],
            self.publisher_commit,
        )

    def test_rejects_any_unreviewed_changed_path(self) -> None:
        extra = self.publisher_repo / "analysis/unreviewed.py"
        extra.parent.mkdir(parents=True)
        extra.write_text("changed\n", encoding="utf-8")
        self._git("add", ".", cwd=self.publisher_repo)
        self._git("commit", "-m", "unreviewed", cwd=self.publisher_repo)
        args = self._args()
        args.publisher_source_commit = self._git(
            "rev-parse", "HEAD", cwd=self.publisher_repo
        )
        with mock.patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": CURRENT_PUBLISHER_JOB_ID,
                "SLURM_JOB_DEPENDENCY": f"afterany:{POPULATION_JOB_ID}",
                "SLURMD_NODENAME": "worker-2",
            },
            clear=False,
        ), self.assertRaisesRegex(
            ReceiptError,
            "changed-path allowlist differs",
        ):
            retry.build_receipt(args)

    def test_rejects_changed_previous_failure_log(self) -> None:
        self.previous_log.write_text(
            retry.EXPECTED_FAILURE_LOG + "changed\n",
            encoding="utf-8",
        )
        args = self._args()
        args.expected_previous_publisher_log_sha256 = sha256_path(
            self.previous_log
        )
        with mock.patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": CURRENT_PUBLISHER_JOB_ID,
                "SLURM_JOB_DEPENDENCY": f"afterany:{POPULATION_JOB_ID}",
                "SLURMD_NODENAME": "worker-2",
            },
            clear=False,
        ), self.assertRaisesRegex(
            ReceiptError,
            "failure message differs",
        ):
            retry.build_receipt(args)


if __name__ == "__main__":
    unittest.main()
