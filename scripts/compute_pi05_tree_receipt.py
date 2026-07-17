#!/usr/bin/env python3
"""Compute the exact pi0.5 Orbax tree hash inside one Slurm allocation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aegis_receipt_utils import (  # noqa: E402
    ReceiptError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_path,
    write_json_exclusive,
)
from scripts import validate_aegis_assets as assets  # noqa: E402


RECEIPT_SCHEMA = "vlsa_table1_pi05_hash_receipt.v1"
UPSTREAM_COMMIT = "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b"
EXPERIMENT_ROOT = Path("/mnt/data/quanth/experiments")


def allocation_identity(environment: dict[str, str]) -> dict[str, str]:
    required = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURMD_NODENAME",
    )
    missing = [key for key in required if not environment.get(key)]
    if missing:
        raise ReceiptError(
            f"checkpoint hashing requires a Slurm allocation: missing {missing}"
        )
    host = environment["SLURMD_NODENAME"]
    if host == "worker-3" or host.startswith(("login", "login-restricted")):
        raise ReceiptError(f"checkpoint hashing cannot execute on {host}")
    if environment["SLURM_ARRAY_TASK_ID"] != "0":
        raise ReceiptError("checkpoint hash receipt is task zero only")
    return {
        "job_id": environment["SLURM_JOB_ID"],
        "array_job_id": environment["SLURM_ARRAY_JOB_ID"],
        "array_task_id": environment["SLURM_ARRAY_TASK_ID"],
        "host": host,
    }


def build_receipt(
    *,
    repo_root: Path,
    expected_commit: str,
    checkpoint: Path,
    slurm: dict[str, str],
) -> dict[str, Any]:
    source = assets.validate_source(
        repo_root,
        expected_commit=expected_commit,
        upstream_commit=UPSTREAM_COMMIT,
    )
    expected_paths = set(assets.PI05_METADATA_FILES) | set(
        assets.PI05_DATA_FILES
    )
    identity_before = assets.pi05_checkpoint_filesystem_identity(
        checkpoint, expected_paths
    )
    checkpoint_record = assets.validate_pi05_checkpoint(
        checkpoint,
        expected_tree_sha256=None,
    )
    started = time.time()
    tree_sha256 = assets._tree_content_sha256(checkpoint, expected_paths)
    finished = time.time()
    checkpoint_record_after = assets.validate_pi05_checkpoint(
        checkpoint,
        expected_tree_sha256=None,
    )
    identity_after = assets.pi05_checkpoint_filesystem_identity(
        checkpoint, expected_paths
    )
    if (
        identity_after != identity_before
        or checkpoint_record_after != checkpoint_record
    ):
        raise ReceiptError(
            "pi0.5 checkpoint changed while the full content hash was running"
        )
    checkpoint_record.update(
        {
            "full_content_tree_sha256": tree_sha256,
            "full_content_hash_verified": True,
            "filesystem_identity": identity_after,
        }
    )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "passed",
        "scientific_result": False,
        "source": source,
        "checkpoint": checkpoint_record,
        "hash_algorithm": (
            "sha256(path_nul_size_nul_file_sha256_newline), "
            "paths sorted lexicographically"
        ),
        "slurm": slurm,
        "host": socket.gethostname(),
        "timing": {
            "hash_started_unix": started,
            "hash_finished_unix": finished,
            "hash_wall_seconds": finished - started,
            "created_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        },
        "execution": {
            "policy_model_executed": False,
            "simulator_executed": False,
            "groundingdino_executed": False,
            "qp_executed": False,
            "training_executed": False,
        },
    }
    receipt["receipt_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(receipt)
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create one immutable allocation-backed pi0.5 tree hash"
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=assets.DEFAULT_PI05_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        slurm = allocation_identity(dict(os.environ))
        output = args.output.resolve()
        try:
            output.relative_to(EXPERIMENT_ROOT)
        except ValueError as error:
            raise ReceiptError(
                "checkpoint hash receipt must remain under "
                "/mnt/data/quanth/experiments"
            ) from error
        receipt = build_receipt(
            repo_root=args.repo_root.resolve(),
            expected_commit=args.expected_commit,
            checkpoint=args.checkpoint.resolve(),
            slurm=slurm,
        )
        write_json_exclusive(output, receipt)
        print(
            f"pi05_tree_sha256="
            f"{receipt['checkpoint']['full_content_tree_sha256']}"
        )
        print(f"receipt={output}")
        print(f"receipt_sha256={sha256_path(output)}")
        return 0
    except (KeyError, OSError, ReceiptError, assets.PreflightError) as error:
        print(f"pi0.5 checkpoint hash failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
