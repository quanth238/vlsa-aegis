#!/usr/bin/env python3
"""CPU-afterany sole publisher for one CFS-00A constrained-flow canary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from crfs_oracle.r05a_constrained_flow_publication import (
    publish_constrained_flow_envelope,
)


def _atomic_failure_receipt(
    path: Path, args: argparse.Namespace, error: Exception
) -> None:
    value = {
        "schema_version": "1.0",
        "artifact_role": "r05a_constrained_flow_canary_cpu_publication",
        "source_job_id": args.source_job_id,
        "source_task_id": f"{args.source_job_id}_0",
        "source_job_state": args.source_job_state,
        "source_exit_code": args.source_exit_code,
        "failure_stage": "python_publication",
        "publisher_job_id": args.publisher_job_id,
        "expected_git_commit": args.expected_git_commit,
        "source_contract_path": args.source_contract,
        "source_contract_sha256_external": args.expected_source_contract_sha256,
        "result_path": args.output,
        "result_sha256": None,
        "passed": False,
        "published": False,
        "errors": [str(error)],
        "one_case_transport_mechanism_claim_allowed": False,
        "simulator_efficacy_claim_allowed": False,
        "infeasibility_claim_allowed": False,
        "probe_training_authorized": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-payload", required=True)
    parser.add_argument("--constrained-flow-payload", required=True)
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
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-receipt", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    receipt = Path(args.validation_receipt)
    try:
        result, receipt = publish_constrained_flow_envelope(
            legacy_payload_path=args.legacy_payload,
            constrained_flow_payload_path=args.constrained_flow_payload,
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
            expected_git_commit=args.expected_git_commit,
            result_path=args.output,
            receipt_path=args.validation_receipt,
            repo_root=Path(__file__).resolve().parents[1],
        )
    except Exception as error:  # Fail closed and leave a small exact receipt.
        candidate = Path(args.output).parent / ".results.candidate.json"
        candidate.unlink(missing_ok=True)
        if not receipt.exists():
            _atomic_failure_receipt(receipt, args, error)
        print(f"constrained-flow publication failed: {error}")
        return 1
    print(f"published {result}")
    print(f"receipt {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
