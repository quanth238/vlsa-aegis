#!/usr/bin/env python3
"""Independently validate one finalized R05A allocation canary artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crfs_harness.artifacts import atomic_write_json, file_sha256, load_json
from crfs_oracle.r05a_canary import (
    validate_r05a_canary_result,
    validate_r05a_canary_schema,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--source-job-id", required=True)
    args = parser.parse_args()
    result_path = Path(args.result)
    value: dict = {}
    try:
        value = load_json(result_path)
        if not isinstance(value, dict):
            raise ValueError("result is not an object")
        errors = validate_r05a_canary_result(value) + validate_r05a_canary_schema(
            value, require_jsonschema=True
        )
        provenance = value.get("provenance")
        if not isinstance(provenance, dict):
            errors.append("result has no provenance object")
        else:
            if str(provenance.get("slurm_array_job_id")) != str(args.source_job_id):
                errors.append("result array job id differs from exact afterany source job")
            if str(provenance.get("slurm_array_task_id")) != "0":
                errors.append("result is not bound to exact source array task zero")
        expected_run_id = result_path.parents[1].name
        if value.get("run_id") != expected_run_id:
            errors.append("result run id differs from its immutable run directory")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        errors = [f"cannot load result: {error}"]
    receipt = {
        "schema_version": "1.0",
        "artifact_role": "r05a_inverse_flow_canary_cpu_afterany_validation",
        "source_job_id": str(args.source_job_id),
        "run_id": value.get("run_id") if isinstance(value, dict) else None,
        "result_path": str(result_path),
        "result_sha256": file_sha256(result_path) if result_path.is_file() else None,
        "passed": not errors,
        "errors": errors,
        "scientific_claim_allowed": False,
        "probe_training_authorized": False,
    }
    atomic_write_json(args.receipt, receipt)
    if errors:
        print("R05A canary validation failed: " + "; ".join(errors))
        return 1
    print(f"validated {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
