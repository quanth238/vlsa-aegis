#!/usr/bin/env python3
"""Bind the exact job-28940 timeout to one finalize-only publisher recovery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aegis_receipt_utils import (  # noqa: E402
    ReceiptError,
    canonical_json_bytes,
    load_json_object,
    sha256_bytes,
    sha256_path,
    verify_payload_sha256,
    write_json_exclusive,
)


SCHEMA_VERSION = "vlsa_table1_publisher_timeout_recovery_authority.v1"
RUN_ID = "vlsa-table1-contact-authority-population-20260718a"
EXPERIMENT_ROOT = Path("/mnt/data/quanth/experiments/vlsa-aegis-table1")
POPULATION_ARRAY_JOB_ID = "28609"
TIMED_OUT_PUBLISHER_JOB_ID = "28940"
POPULATION_SOURCE_COMMIT = "1592aa59361f431ba96c6ddcbebcb596f6c20853"
TIMED_OUT_PUBLISHER_SOURCE_COMMIT = (
    "5fcb15015a6d24b5e86c685d5cdfaa748778fb99"
)
TIMED_OUT_PUBLISHER_LOG_SHA256 = (
    "583d4db1555d924c46edec4dda4c623a95fa00c83c7a4d05c492b5c948f58b87"
)
TIMED_OUT_PUBLISHER_LOG_BYTES = 1342
EXPECTED_TIMEOUT_LOG_SUFFIX = (
    "slurmstepd: error: *** JOB 28940 ON worker-1 CANCELLED AT "
    "2026-07-21T21:09:08 DUE TO TIME LIMIT ***\n"
)
EXPECTED_TIMEOUT_ACCOUNTING = {
    "job_id": TIMED_OUT_PUBLISHER_JOB_ID,
    "job_id_raw": TIMED_OUT_PUBLISHER_JOB_ID,
    "state": "TIMEOUT",
    "exit_code": "0:0",
    "node": "worker-1",
    "elapsed": "04:00:09",
    "start": "2026-07-21T17:08:59",
    "end": "2026-07-21T21:09:08",
}
EXPECTED_ATTEMPT_ARTIFACTS = {
    "population-array-sacct.txt": (
        1964,
        "6b798a245196810df0a2359fced2d4930cfa81dc42c01b22bfdf7196eb62ba4f",
    ),
    "publisher-retry-authority.json": (
        4140,
        "8d437e04a496d28b92cce4f75c13a109688f774943cefb21f638102fa1055d95",
    ),
    "prepublish-validation.json": (
        172280,
        "05df4c759a478069f1df3fe318c6f6237e65aeb667212f208ec94e2ab2a13eaf",
    ),
    "population-summary.json": (
        26251,
        "c2702d40d53436b89e48a5e68d743a76403d076a5c5539fa6c74a850af698330",
    ),
    "failure-analysis/cases.jsonl": (
        9743044,
        "2280c3f1dc7e25755f650e7af0deb140263edec0679e3395db13a539b5a6a780",
    ),
    "failure-analysis/report.json": (
        3615,
        "6d1d74040c55512ad9035cdd4a2976c6eeae15c6d32e17aff6164164cdc44de4",
    ),
    "failure-analysis/report.md": (
        1876,
        "22fb277b6779165bf84d54faf7a8234535d40c0c6bc3a1e6dca93be2c00b1883",
    ),
    "gallery/index.html": (
        7176729,
        "40781fa0a817931ad23bb12b2b7be2b858e16033f97ed18b7f76b86d291b2bb2",
    ),
}
EXPECTED_RECOVERY_CHANGED_PATHS = (
    "scripts/build_aegis_publisher_timeout_recovery_authority.py",
    "scripts/validate_aegis_run_artifacts.py",
    "slurm/aegis_population_publisher_timeout_finalize.sbatch",
    "slurm/run_aegis_population_publisher_timeout_finalize.sh",
    "tests/test_aegis_publisher_timeout_recovery.py",
    "tests/test_aegis_slurm_contract.py",
)


def _require_git_commit(value: str, *, label: str) -> str:
    if (
        len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReceiptError(f"{label} must be a lowercase 40-character commit")
    return value


def _git(repo: Path, *arguments: str) -> str:
    if repo.is_symlink() or not repo.is_dir():
        raise ReceiptError(f"source repository is missing or symlinked: {repo}")
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReceiptError(
            f"git source verification failed for {repo}: {' '.join(arguments)}"
        ) from error
    return completed.stdout.strip()


def _require_clean_commit(repo: Path, commit: str, *, label: str) -> None:
    if _git(repo, "rev-parse", "HEAD") != commit:
        raise ReceiptError(f"{label} source commit differs")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ReceiptError(f"{label} source tree is dirty")


def _parse_timeout_accounting(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ReceiptError("timed-out publisher accounting is missing or symlinked")
    try:
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, UnicodeDecodeError) as error:
        raise ReceiptError("timed-out publisher accounting is unreadable") from error
    if len(lines) != 1:
        raise ReceiptError("timed-out publisher accounting must contain one row")
    fields = lines[0].split("|")
    if len(fields) != 8:
        raise ReceiptError("timed-out publisher accounting field count differs")
    row = dict(
        zip(
            (
                "job_id",
                "job_id_raw",
                "state",
                "exit_code",
                "node",
                "elapsed",
                "start",
                "end",
            ),
            fields,
            strict=True,
        )
    )
    if row != EXPECTED_TIMEOUT_ACCOUNTING:
        raise ReceiptError(
            "timed-out publisher accounting differs: "
            f"observed={row!r}, expected={EXPECTED_TIMEOUT_ACCOUNTING!r}"
        )
    return row


def build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    expected_run_root = (EXPERIMENT_ROOT / RUN_ID).resolve()
    if run_root != expected_run_root:
        raise ReceiptError("timeout recovery run root differs")

    population_repo = args.population_source_repo.resolve()
    timed_out_repo = args.timed_out_publisher_source_repo.resolve()
    recovery_repo = args.recovery_source_repo.resolve()
    recovery_commit = _require_git_commit(
        args.recovery_source_commit,
        label="recovery source commit",
    )
    _require_clean_commit(
        population_repo,
        POPULATION_SOURCE_COMMIT,
        label="population",
    )
    _require_clean_commit(
        timed_out_repo,
        TIMED_OUT_PUBLISHER_SOURCE_COMMIT,
        label="timed-out publisher",
    )
    _require_clean_commit(
        recovery_repo,
        recovery_commit,
        label="recovery publisher",
    )
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(recovery_repo),
                "merge-base",
                "--is-ancestor",
                TIMED_OUT_PUBLISHER_SOURCE_COMMIT,
                recovery_commit,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReceiptError(
            "timeout recovery release is not descended from job-28940 source"
        ) from error
    changed_paths = tuple(
        sorted(
            line
            for line in _git(
                recovery_repo,
                "diff",
                "--name-only",
                "--diff-filter=ACDMRTUXB",
                f"{TIMED_OUT_PUBLISHER_SOURCE_COMMIT}..{recovery_commit}",
            ).splitlines()
            if line
        )
    )
    if changed_paths != EXPECTED_RECOVERY_CHANGED_PATHS:
        raise ReceiptError(
            "timeout recovery changed-path allowlist differs: "
            f"observed={changed_paths!r}, "
            f"expected={EXPECTED_RECOVERY_CHANGED_PATHS!r}"
        )

    timed_out_attempt_root = args.timed_out_attempt_root.resolve()
    if timed_out_attempt_root != (
        run_root
        / "publication-attempts"
        / f"job-{TIMED_OUT_PUBLISHER_JOB_ID}"
    ):
        raise ReceiptError("timed-out publisher attempt root differs")
    observed_relative_files = tuple(
        sorted(
            str(path.relative_to(timed_out_attempt_root))
            for path in timed_out_attempt_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
    )
    expected_relative_files = tuple(sorted(EXPECTED_ATTEMPT_ARTIFACTS))
    if observed_relative_files != expected_relative_files:
        raise ReceiptError(
            "timed-out publisher attempt inventory differs: "
            f"observed={observed_relative_files!r}, "
            f"expected={expected_relative_files!r}"
        )
    attempt_inventory: list[dict[str, Any]] = []
    for relative_path in expected_relative_files:
        path = timed_out_attempt_root / relative_path
        expected_size, expected_sha256 = EXPECTED_ATTEMPT_ARTIFACTS[
            relative_path
        ]
        if path.is_symlink() or not path.is_file():
            raise ReceiptError(
                f"timed-out publisher artifact is missing: {relative_path}"
            )
        if path.stat().st_size != expected_size:
            raise ReceiptError(
                f"timed-out publisher artifact size differs: {relative_path}"
            )
        if sha256_path(path) != expected_sha256:
            raise ReceiptError(
                f"timed-out publisher artifact SHA-256 differs: {relative_path}"
            )
        attempt_inventory.append(
            {
                "relative_path": relative_path,
                "bytes": expected_size,
                "sha256": expected_sha256,
            }
        )

    timed_out_log = args.timed_out_publisher_log.resolve()
    if timed_out_log.is_symlink() or not timed_out_log.is_file():
        raise ReceiptError("timed-out publisher log is missing or symlinked")
    if timed_out_log.stat().st_size != TIMED_OUT_PUBLISHER_LOG_BYTES:
        raise ReceiptError("timed-out publisher log byte count differs")
    if sha256_path(timed_out_log) != TIMED_OUT_PUBLISHER_LOG_SHA256:
        raise ReceiptError("timed-out publisher log SHA-256 differs")
    try:
        timed_out_log_text = timed_out_log.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ReceiptError("timed-out publisher log is unreadable") from error
    if not timed_out_log_text.endswith(EXPECTED_TIMEOUT_LOG_SUFFIX):
        raise ReceiptError("timed-out publisher terminal message differs")
    accounting = _parse_timeout_accounting(
        args.timed_out_publisher_accounting.resolve()
    )

    prior_authority_path = (
        timed_out_attempt_root / "publisher-retry-authority.json"
    )
    prior_authority = load_json_object(
        prior_authority_path,
        label="timed-out publisher retry authority",
    )
    if prior_authority.get("schema_version") != (
        "vlsa_table1_publisher_retry_authority.v3"
    ):
        raise ReceiptError("timed-out publisher authority schema differs")
    if prior_authority.get("status") != "validated":
        raise ReceiptError("timed-out publisher authority status differs")
    if prior_authority.get("scientific_result") is not False:
        raise ReceiptError("timed-out publisher authority is scientific")
    if prior_authority.get("run_id") != RUN_ID:
        raise ReceiptError("timed-out publisher authority run differs")
    if prior_authority.get("population_array_job_id") != (
        POPULATION_ARRAY_JOB_ID
    ):
        raise ReceiptError("timed-out publisher authority array differs")
    prior_slurm = prior_authority.get("publisher_slurm")
    if not isinstance(prior_slurm, dict) or prior_slurm != {
        "job_id": TIMED_OUT_PUBLISHER_JOB_ID,
        "host": "worker-1",
        "dependency": f"afterany:{POPULATION_ARRAY_JOB_ID}",
    }:
        raise ReceiptError("timed-out publisher authority Slurm binding differs")
    prior_authority_payload_sha256 = verify_payload_sha256(
        prior_authority,
        field="receipt_payload_sha256",
        label="timed-out publisher retry authority",
    )

    if (run_root / "population-publication-receipt.json").exists():
        raise ReceiptError("run-level publication receipt already exists")
    if (timed_out_attempt_root / "publisher-failure.json").exists():
        raise ReceiptError("unexpected timed-out publisher failure receipt exists")

    environment = os.environ
    current_job_id = environment.get("SLURM_JOB_ID", "")
    current_dependency = environment.get("SLURM_JOB_DEPENDENCY", "")
    current_host = environment.get("SLURMD_NODENAME", "")
    if not current_job_id or current_job_id == TIMED_OUT_PUBLISHER_JOB_ID:
        raise ReceiptError("timeout recovery requires one new exact Slurm job")
    if current_dependency != f"afterany:{TIMED_OUT_PUBLISHER_JOB_ID}":
        raise ReceiptError("timeout recovery dependency differs")
    if current_host == "worker-3" or current_host.startswith(
        ("login", "login-restricted")
    ):
        raise ReceiptError(f"timeout recovery cannot execute on {current_host}")
    expected_output_parent = (
        run_root / "publication-attempts" / f"job-{current_job_id}"
    )
    output = args.output.resolve()
    if output.parent != expected_output_parent:
        raise ReceiptError("timeout recovery authority escaped its attempt root")

    source_files = {
        "validator": args.validator.resolve(),
        "authority_builder": args.recovery_authority_builder.resolve(),
        "runner": args.recovery_runner.resolve(),
        "sbatch": args.recovery_sbatch.resolve(),
    }
    source_hashes: dict[str, str] = {}
    for label, path in source_files.items():
        try:
            path.relative_to(recovery_repo)
        except ValueError as error:
            raise ReceiptError(
                f"timeout recovery {label} escaped reviewed source"
            ) from error
        if path.is_symlink() or not path.is_file():
            raise ReceiptError(
                f"timeout recovery {label} is missing or symlinked"
            )
        source_hashes[f"{label}_sha256"] = sha256_path(path)

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "validated",
        "scientific_result": False,
        "run_id": RUN_ID,
        "population_array_job_id": POPULATION_ARRAY_JOB_ID,
        "preserves_immutable_result_tree": True,
        "permits_inference_or_simulation": False,
        "recovery_scope": "population_finalize_only",
        "population_source": {
            "repo": str(population_repo),
            "git_commit": POPULATION_SOURCE_COMMIT,
            "clean": True,
        },
        "timed_out_publisher_source": {
            "repo": str(timed_out_repo),
            "git_commit": TIMED_OUT_PUBLISHER_SOURCE_COMMIT,
            "clean": True,
        },
        "recovery_source": {
            "repo": str(recovery_repo),
            "git_commit": recovery_commit,
            "clean": True,
            "changed_paths": list(changed_paths),
            **source_hashes,
        },
        "timed_out_publisher": {
            "publisher_slurm": {
                "job_id": accounting["job_id"],
                "host": accounting["node"],
                "dependency": f"afterany:{POPULATION_ARRAY_JOB_ID}",
                "state": accounting["state"],
                "exit_code": accounting["exit_code"],
                "elapsed": accounting["elapsed"],
                "start": accounting["start"],
                "end": accounting["end"],
            },
            "log": {
                "path": str(timed_out_log),
                "bytes": TIMED_OUT_PUBLISHER_LOG_BYTES,
                "sha256": TIMED_OUT_PUBLISHER_LOG_SHA256,
            },
            "retry_authority": {
                "path": str(prior_authority_path),
                "sha256": EXPECTED_ATTEMPT_ARTIFACTS[
                    "publisher-retry-authority.json"
                ][1],
                "receipt_payload_sha256": prior_authority_payload_sha256,
            },
            "attempt_root": str(timed_out_attempt_root),
            "attempt_inventory": attempt_inventory,
            "final_receipt_missing": True,
            "failure_receipt_missing": True,
        },
        "recovery_publisher_slurm": {
            "job_id": current_job_id,
            "host": current_host,
            "dependency": current_dependency,
        },
    }
    receipt["receipt_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(receipt)
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate exact job-28940 finalize-only recovery authority"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--population-source-repo", type=Path, required=True)
    parser.add_argument(
        "--timed-out-publisher-source-repo", type=Path, required=True
    )
    parser.add_argument("--recovery-source-repo", type=Path, required=True)
    parser.add_argument("--recovery-source-commit", required=True)
    parser.add_argument("--timed-out-attempt-root", type=Path, required=True)
    parser.add_argument("--timed-out-publisher-log", type=Path, required=True)
    parser.add_argument(
        "--timed-out-publisher-accounting", type=Path, required=True
    )
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--recovery-authority-builder", type=Path, required=True)
    parser.add_argument("--recovery-runner", type=Path, required=True)
    parser.add_argument("--recovery-sbatch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        receipt = build_receipt(args)
        write_json_exclusive(args.output.resolve(), receipt)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "output": str(args.output.resolve()),
                    "receipt_payload_sha256": receipt[
                        "receipt_payload_sha256"
                    ],
                    "receipt_file_sha256": sha256_path(args.output.resolve()),
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ReceiptError, ValueError) as error:
        print(
            f"AEGIS timeout recovery authority failed: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
