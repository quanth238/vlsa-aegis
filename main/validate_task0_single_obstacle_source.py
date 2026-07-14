#!/usr/bin/env python3
"""Independently validate one generated task-0 source bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
from typing import Any, Mapping

from crfs_harness.artifacts import file_sha256, load_json
from crfs_oracle.generated_source import (
    ACCEPTED_STATUS,
    validate_generated_source_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-source-slurm-job-id", required=True)
    parser.add_argument("--expected-source-slurm-array-job-id", required=True)
    parser.add_argument("--expected-source-slurm-array-task-id", required=True)
    parser.add_argument("--expected-generation-request-id")
    return parser


def main() -> int:
    args = _parser().parse_args()
    artifact = Path(args.artifact).expanduser().resolve()
    errors: list[str] = []
    artifact_sha256: str | None = None
    try:
        value = load_json(artifact)
        artifact_sha256 = file_sha256(artifact)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        value = None
        errors.append(f"cannot read artifact: {error}")
    if not isinstance(value, Mapping):
        if value is not None:
            errors.append("artifact must contain an object")
    else:
        try:
            errors.extend(validate_generated_source_artifact(value))
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError) as error:
            errors.append(f"semantic validator raised on malformed artifact: {error}")
        if value.get("status") != ACCEPTED_STATUS:
            errors.append("independent source validator requires an accepted bundle")
        config = value.get("config") if isinstance(value.get("config"), Mapping) else {}
        if config.get("file_sha256") != args.expected_config_sha256:
            errors.append("config.file_sha256 differs from independent expectation")
        source = value.get("source_state") if isinstance(value.get("source_state"), Mapping) else {}
        if args.expected_generation_request_id is not None and source.get("generation_request_id") != (
            args.expected_generation_request_id
        ):
            errors.append("source_state.generation_request_id differs from expectation")
        provenance = value.get("provenance") if isinstance(value.get("provenance"), Mapping) else {}
        expected_slurm = {
            "slurm_job_id": args.expected_source_slurm_job_id,
            "slurm_array_job_id": args.expected_source_slurm_array_job_id,
            "slurm_array_task_id": args.expected_source_slurm_array_task_id,
        }
        for key, expected in expected_slurm.items():
            if provenance.get(key) != expected:
                errors.append(f"provenance.{key} differs from expected source allocation")
    report: dict[str, Any] = {
        "artifact": str(artifact),
        "artifact_sha256": artifact_sha256,
        "expected_source": {
            "slurm_job_id": args.expected_source_slurm_job_id,
            "slurm_array_job_id": args.expected_source_slurm_array_job_id,
            "slurm_array_task_id": args.expected_source_slurm_array_task_id,
            "generation_request_id": args.expected_generation_request_id,
            "config_sha256": args.expected_config_sha256,
        },
        "validator_provenance": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "host": socket.gethostname(),
        },
        "validation_error_count": len(errors),
        "validation_errors": errors,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
