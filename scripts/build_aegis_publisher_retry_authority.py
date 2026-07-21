#!/usr/bin/env python3
"""Bind two reviewed publication-only fixes to one immutable retry."""

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

from scripts.aegis_receipt_utils import (
    ReceiptError,
    canonical_json_bytes,
    load_json_object,
    require_sha256,
    sha256_bytes,
    sha256_path,
    write_json_exclusive,
)


SCHEMA_VERSION = "vlsa_table1_publisher_retry_authority.v2"
EXPECTED_FAILURE_LOG = (
    "AEGIS artifact validation failed: "
    "vlsa-t1-spatial-i-t0-e00/pi05: contact event_counts_by_role "
    "keys/order differs: "
    "observed=('dynamic_other', 'dynamic_task_object', 'robot', "
    "'static_support', 'unknown'), "
    "expected=('robot', 'static_support', 'dynamic_task_object', "
    "'dynamic_other', 'unknown')\n"
)
PRIOR_RETRY_LOG_PREFIX = (
    '{"output": '
    '"/mnt/data/quanth/experiments/vlsa-aegis-table1/'
    'vlsa-table1-contact-authority-population-20260718a/'
    'publication-attempts/job-28906/publisher-retry-authority.json", '
    '"receipt_file_sha256": '
    '"0ab7c8fded8357a545ccf85fd0325905485427b9810b2022f51b6bd30e3dd798", '
    '"receipt_payload_sha256": '
    '"68ad8e700b458780a65543b017e9917d126828fbefcb9f0dfc06218dcc6e1220", '
    '"status": "validated"}\n'
    "AEGIS artifact validation failed: paired-canary action-invariant "
    "evidence differs: observed="
)
PRIOR_RETRY_OBSERVED_REFERENCE = (
    "'path': '/home/quanth/working_space/vlsa-aegis-table-repro/"
    "fixtures/vlsa_table1_canary_action_reference.json'"
)
PRIOR_RETRY_EXPECTED_REFERENCE = (
    "'path': '/home/quanth/working_space/vlsa-aegis-publication-retry/"
    "fixtures/vlsa_table1_canary_action_reference.json'"
)
EXPECTED_CHANGED_PATHS = (
    "scripts/build_aegis_publisher_retry_authority.py",
    "scripts/validate_aegis_run_artifacts.py",
    "slurm/aegis_population_publisher_retry.sbatch",
    "slurm/run_aegis_population_publisher.sh",
    "tests/test_aegis_population_streaming_validation.py",
    "tests/test_aegis_publisher_retry_authority.py",
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
    observed = _git(repo, "rev-parse", "HEAD")
    if observed != commit:
        raise ReceiptError(
            f"{label} commit differs: observed={observed}, expected={commit}"
        )
    dirty = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ReceiptError(f"{label} source tree is dirty")


def build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    population_repo = args.population_source_repo.resolve()
    publisher_repo = args.publisher_source_repo.resolve()
    population_commit = _require_git_commit(
        args.population_source_commit,
        label="population source commit",
    )
    publisher_commit = _require_git_commit(
        args.publisher_source_commit,
        label="publisher source commit",
    )
    if publisher_commit == population_commit:
        raise ReceiptError("publisher retry must use a distinct reviewed commit")
    _require_clean_commit(
        population_repo,
        population_commit,
        label="population",
    )
    _require_clean_commit(
        publisher_repo,
        publisher_commit,
        label="publisher",
    )
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(publisher_repo),
                "merge-base",
                "--is-ancestor",
                population_commit,
                publisher_commit,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReceiptError(
            "publisher release is not descended from the population source"
        ) from error
    changed_paths = tuple(
        sorted(
            line
            for line in _git(
                publisher_repo,
                "diff",
                "--name-only",
                "--diff-filter=ACDMRTUXB",
                f"{population_commit}..{publisher_commit}",
            ).splitlines()
            if line
        )
    )
    if changed_paths != EXPECTED_CHANGED_PATHS:
        raise ReceiptError(
            "publisher retry changed-path allowlist differs: "
            f"observed={changed_paths!r}, expected={EXPECTED_CHANGED_PATHS!r}"
        )

    previous_log = args.previous_publisher_log.resolve()
    previous_failure_path = args.previous_publisher_failure.resolve()
    expected_log_sha256 = require_sha256(
        args.expected_previous_publisher_log_sha256,
        label="previous publisher log SHA-256",
    )
    expected_failure_sha256 = require_sha256(
        args.expected_previous_publisher_failure_sha256,
        label="previous publisher failure SHA-256",
    )
    if previous_log.is_symlink() or not previous_log.is_file():
        raise ReceiptError("previous publisher log is missing or symlinked")
    if sha256_path(previous_log) != expected_log_sha256:
        raise ReceiptError("previous publisher log SHA-256 differs")
    try:
        previous_log_text = previous_log.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ReceiptError("previous publisher log is unreadable") from error
    if previous_log_text != EXPECTED_FAILURE_LOG:
        raise ReceiptError("previous publisher failure message differs")
    previous_failure = load_json_object(
        previous_failure_path,
        label="previous publisher failure receipt",
    )
    if sha256_path(previous_failure_path) != expected_failure_sha256:
        raise ReceiptError("previous publisher failure receipt SHA-256 differs")
    expected_failure = {
        "schema_version": "vlsa_table1_population_publisher_failure.v1",
        "status": "apparatus_failure",
        "scientific_result": False,
        "run_id": args.run_id,
        "population_array_job_id": args.population_array_job_id,
        "publisher_job_id": args.previous_publisher_job_id,
        "host": "worker-1",
        "failure_stage": "population_prepublish_validation",
        "exit_code": 2,
    }
    if previous_failure != expected_failure:
        raise ReceiptError("previous publisher failure receipt payload differs")

    prior_retry_log = args.prior_retry_publisher_log.resolve()
    prior_retry_failure_path = args.prior_retry_publisher_failure.resolve()
    prior_retry_authority_path = args.prior_retry_publisher_authority.resolve()
    expected_prior_log_sha256 = require_sha256(
        args.expected_prior_retry_publisher_log_sha256,
        label="prior retry publisher log SHA-256",
    )
    expected_prior_failure_sha256 = require_sha256(
        args.expected_prior_retry_publisher_failure_sha256,
        label="prior retry publisher failure SHA-256",
    )
    expected_prior_authority_sha256 = require_sha256(
        args.expected_prior_retry_publisher_authority_sha256,
        label="prior retry publisher authority SHA-256",
    )
    if prior_retry_log.is_symlink() or not prior_retry_log.is_file():
        raise ReceiptError("prior retry publisher log is missing or symlinked")
    if sha256_path(prior_retry_log) != expected_prior_log_sha256:
        raise ReceiptError("prior retry publisher log SHA-256 differs")
    try:
        prior_retry_log_text = prior_retry_log.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ReceiptError("prior retry publisher log is unreadable") from error
    if (
        not prior_retry_log_text.startswith(PRIOR_RETRY_LOG_PREFIX)
        or PRIOR_RETRY_OBSERVED_REFERENCE not in prior_retry_log_text
        or PRIOR_RETRY_EXPECTED_REFERENCE not in prior_retry_log_text
    ):
        raise ReceiptError("prior retry publisher failure message differs")
    prior_retry_failure = load_json_object(
        prior_retry_failure_path,
        label="prior retry publisher failure receipt",
    )
    if sha256_path(prior_retry_failure_path) != expected_prior_failure_sha256:
        raise ReceiptError(
            "prior retry publisher failure receipt SHA-256 differs"
        )
    expected_prior_failure = {
        "schema_version": "vlsa_table1_population_publisher_failure.v1",
        "status": "apparatus_failure",
        "scientific_result": False,
        "run_id": args.run_id,
        "population_array_job_id": args.population_array_job_id,
        "publisher_job_id": args.prior_retry_publisher_job_id,
        "host": "worker-1",
        "failure_stage": "population_prepublish_validation",
        "exit_code": 2,
    }
    if prior_retry_failure != expected_prior_failure:
        raise ReceiptError(
            "prior retry publisher failure receipt payload differs"
        )
    prior_retry_authority = load_json_object(
        prior_retry_authority_path,
        label="prior retry publisher authority receipt",
    )
    if sha256_path(prior_retry_authority_path) != expected_prior_authority_sha256:
        raise ReceiptError(
            "prior retry publisher authority receipt SHA-256 differs"
        )
    for field, expected in (
        ("schema_version", "vlsa_table1_publisher_retry_authority.v1"),
        ("status", "validated"),
        ("scientific_result", False),
        ("run_id", args.run_id),
        ("population_array_job_id", args.population_array_job_id),
        ("preserves_immutable_result_tree", True),
        ("permits_inference_or_simulation", False),
        ("failure_class", "publisher_validator_json_key_order"),
    ):
        if prior_retry_authority.get(field) != expected:
            raise ReceiptError(
                f"prior retry publisher authority {field} differs"
            )
    prior_retry_slurm = prior_retry_authority.get("publisher_slurm")
    prior_retry_recovery = prior_retry_authority.get("recovery_from")
    if not isinstance(prior_retry_slurm, dict) or not isinstance(
        prior_retry_recovery, dict
    ):
        raise ReceiptError("prior retry publisher authority is incomplete")
    if (
        prior_retry_slurm.get("job_id") != args.prior_retry_publisher_job_id
        or prior_retry_recovery.get("publisher_job_id")
        != args.previous_publisher_job_id
    ):
        raise ReceiptError("prior retry publisher authority chain differs")

    environment = os.environ
    current_job_id = environment.get("SLURM_JOB_ID", "")
    current_dependency = environment.get("SLURM_JOB_DEPENDENCY", "")
    current_host = environment.get("SLURMD_NODENAME", "")
    if not current_job_id or current_job_id in {
        args.previous_publisher_job_id,
        args.prior_retry_publisher_job_id,
    }:
        raise ReceiptError("publisher retry requires one new exact Slurm job")
    if current_dependency != f"afterany:{args.population_array_job_id}":
        raise ReceiptError("publisher retry dependency differs")
    if current_host == "worker-3" or current_host.startswith(
        ("login", "login-restricted")
    ):
        raise ReceiptError(f"publisher retry cannot execute on {current_host}")

    expected_attempt_root = (
        run_root / "publication-attempts" / f"job-{current_job_id}"
    )
    output = args.output.resolve()
    if output.parent != expected_attempt_root:
        raise ReceiptError("publisher retry authority escaped its attempt root")

    source_files = {
        "validator": args.validator.resolve(),
        "authority_builder": args.publisher_authority_builder.resolve(),
        "publisher_runner": args.publisher_runner.resolve(),
        "publisher_sbatch": args.publisher_sbatch.resolve(),
    }
    source_hashes: dict[str, str] = {}
    for label, path in source_files.items():
        try:
            path.relative_to(publisher_repo)
        except ValueError as error:
            raise ReceiptError(
                f"publisher {label} escaped the reviewed source repository"
            ) from error
        if path.is_symlink() or not path.is_file():
            raise ReceiptError(f"publisher {label} is missing or symlinked")
        source_hashes[f"{label}_sha256"] = sha256_path(path)

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "validated",
        "scientific_result": False,
        "run_id": args.run_id,
        "population_array_job_id": args.population_array_job_id,
        "preserves_immutable_result_tree": True,
        "permits_inference_or_simulation": False,
        "failure_class": "publisher_validator_source_root_rebinding",
        "population_source": {
            "repo": str(population_repo),
            "git_commit": population_commit,
            "clean": True,
        },
        "publisher_source": {
            "repo": str(publisher_repo),
            "git_commit": publisher_commit,
            "clean": True,
            "changed_paths": list(changed_paths),
            **source_hashes,
        },
        "recovery_from": {
            "publisher_job_id": args.previous_publisher_job_id,
            "failure_stage": "population_prepublish_validation",
            "exit_code": 2,
            "log": {
                "path": str(previous_log),
                "sha256": expected_log_sha256,
            },
            "failure_receipt": {
                "path": str(previous_failure_path),
                "sha256": expected_failure_sha256,
            },
        },
        "prior_retry": {
            "publisher_job_id": args.prior_retry_publisher_job_id,
            "failure_stage": "population_prepublish_validation",
            "exit_code": 2,
            "log": {
                "path": str(prior_retry_log),
                "sha256": expected_prior_log_sha256,
            },
            "failure_receipt": {
                "path": str(prior_retry_failure_path),
                "sha256": expected_prior_failure_sha256,
            },
            "authority_receipt": {
                "path": str(prior_retry_authority_path),
                "sha256": expected_prior_authority_sha256,
            },
        },
        "publisher_slurm": {
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
        description="Validate one immutable publication-only retry authority"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--population-array-job-id", required=True)
    parser.add_argument("--population-source-repo", type=Path, required=True)
    parser.add_argument("--population-source-commit", required=True)
    parser.add_argument("--publisher-source-repo", type=Path, required=True)
    parser.add_argument("--publisher-source-commit", required=True)
    parser.add_argument("--previous-publisher-job-id", required=True)
    parser.add_argument("--previous-publisher-log", type=Path, required=True)
    parser.add_argument(
        "--expected-previous-publisher-log-sha256", required=True
    )
    parser.add_argument(
        "--previous-publisher-failure", type=Path, required=True
    )
    parser.add_argument(
        "--expected-previous-publisher-failure-sha256", required=True
    )
    parser.add_argument("--prior-retry-publisher-job-id", required=True)
    parser.add_argument(
        "--prior-retry-publisher-log", type=Path, required=True
    )
    parser.add_argument(
        "--expected-prior-retry-publisher-log-sha256", required=True
    )
    parser.add_argument(
        "--prior-retry-publisher-failure", type=Path, required=True
    )
    parser.add_argument(
        "--expected-prior-retry-publisher-failure-sha256", required=True
    )
    parser.add_argument(
        "--prior-retry-publisher-authority", type=Path, required=True
    )
    parser.add_argument(
        "--expected-prior-retry-publisher-authority-sha256", required=True
    )
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument(
        "--publisher-authority-builder", type=Path, required=True
    )
    parser.add_argument("--publisher-runner", type=Path, required=True)
    parser.add_argument("--publisher-sbatch", type=Path, required=True)
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
                    "receipt_file_sha256": sha256_path(
                        args.output.resolve()
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ReceiptError, ValueError) as error:
        print(f"AEGIS publisher retry authority failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
