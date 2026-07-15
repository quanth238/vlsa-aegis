#!/usr/bin/env python3
"""CPU-afterany sole publisher for one R05A sampled-current canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import os

from crfs_oracle.r05a_sampled_current_canary import publish_sampled_current_envelope


def _atomic_failure_receipt(path: Path, args: argparse.Namespace, error: Exception) -> None:
    value = {
        "schema_version": "2.0",
        "artifact_role": "r05a_sampled_current_canary_cpu_publication",
        "source_job_id": args.source_job_id,
        "source_task_id": f"{args.source_job_id}_0",
        "source_job_state": args.source_job_state,
        "source_exit_code": args.source_exit_code,
        "source_query_status": 0,
        "failure_stage": "python_publication",
        "publisher_job_id": args.publisher_job_id,
        "source_contract_path": args.source_contract,
        "source_contract_sha256_external": args.expected_source_contract_sha256,
        "result_path": args.result,
        "result_sha256": None,
        "passed": False,
        "published": False,
        "errors": [str(error)],
        "scientific_claim_allowed": False,
        "probe_training_authorized": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True)
    parser.add_argument("--host-telemetry", required=True)
    parser.add_argument("--gpu-samples", required=True)
    parser.add_argument("--allocation-tests-log", required=True)
    parser.add_argument("--source-contract", required=True)
    parser.add_argument("--expected-source-contract-sha256", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--source-job-id", required=True)
    parser.add_argument("--source-job-state", required=True)
    parser.add_argument("--source-exit-code", required=True)
    parser.add_argument("--publisher-job-id", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    receipt = Path(args.receipt)
    try:
        result, receipt = publish_sampled_current_envelope(
            payload_path=args.payload,
            host_telemetry_path=args.host_telemetry,
            gpu_samples_path=args.gpu_samples,
            allocation_tests_log=args.allocation_tests_log,
            source_contract_path=args.source_contract,
            expected_source_contract_sha256=args.expected_source_contract_sha256,
            submission_path=args.submission,
            source_job_id=args.source_job_id,
            source_job_state=args.source_job_state,
            source_exit_code=args.source_exit_code,
            publisher_job_id=args.publisher_job_id,
            result_path=args.result,
            receipt_path=args.receipt,
            repo_root=Path(__file__).resolve().parents[1],
        )
    except Exception as error:  # Fail closed and preserve one small exact receipt.
        candidate = Path(args.result).parent / ".results.candidate.json"
        candidate.unlink(missing_ok=True)
        if not receipt.exists():
            _atomic_failure_receipt(receipt, args, error)
        print(f"sampled-current publication failed: {error}")
        return 1
    print(f"published {result}")
    print(f"receipt {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
