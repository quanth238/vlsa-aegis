#!/usr/bin/env python3
"""Run the paired CFS-00A transport canary inside its reviewed allocation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from crfs_harness.artifacts import (
    atomic_write_json,
    file_sha256,
    validate_jsonl_unique,
)
from crfs_harness.manifest import validate_case
from crfs_oracle.r05a_canary import (
    CASE_ID,
    GROUP_ID,
    r05a_canary_config_from_mapping,
)
from crfs_oracle.r05a_constrained_flow_canary import (
    PAYLOAD_TYPE,
    constrained_flow_scientific_config_hash,
    run_r05a_constrained_flow_canary,
    validate_constrained_flow_config,
)
from crfs_oracle.runner import oracle_config_from_mapping


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True, help="frozen constrained-flow config")
    parser.add_argument("--legacy-config", required=True, help="frozen R05A config")
    parser.add_argument("--r02-raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    return parser


def _object(path: str | Path, *, name: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{name} must be a JSON object")
    return value


def main() -> int:
    args = _parser().parse_args()
    cases, errors = validate_jsonl_unique(args.manifest, "case_id")
    for case in cases:
        errors.extend(
            f"{case.get('case_id', '<unknown>')}: {error}"
            for error in validate_case(case)
        )
    if errors:
        raise SystemExit("invalid R05A manifest: " + "; ".join(errors))
    if len(cases) != 3 or args.case_index != 0:
        raise SystemExit("CFS-00A is fixed to row zero of the three-case manifest")
    case = cases[0]
    if case.get("case_id") != CASE_ID or case.get("group_id") != GROUP_ID:
        raise SystemExit("CFS-00A case/group identity changed")

    constrained_value = _object(args.config, name="constrained-flow config")
    config_errors = validate_constrained_flow_config(constrained_value)
    if config_errors:
        raise SystemExit("invalid constrained-flow config: " + "; ".join(config_errors))
    if constrained_value.get("ready_to_run") is not True:
        raise SystemExit("constrained-flow config is not released for allocation execution")
    legacy_value = _object(args.legacy_config, name="legacy R05A config")
    if legacy_value.get("ready_to_run") is not True:
        raise SystemExit("legacy R05A config is not released")

    root = Path(__file__).resolve().parents[1]
    oracle = oracle_config_from_mapping(
        legacy_value,
        host=args.host,
        port=args.port,
        checkpoint_id=args.checkpoint_id,
        checkpoint_sha256=args.checkpoint_sha256,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    legacy_config = r05a_canary_config_from_mapping(
        legacy_value,
        oracle,
        repo_root=root,
        source_r02_results_root=args.r02_raw_root,
        config_file_sha256=file_sha256(args.legacy_config),
    )

    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    try:
        output, status = run_r05a_constrained_flow_canary(
            case,
            constrained_value,
            legacy_config,
            repo_root=root,
            input_manifest_sha256=file_sha256(args.manifest),
            constrained_config_path=args.config,
            legacy_config_path=args.legacy_config,
            client=client,
        )
    except Exception as error:
        # Registered Jacobian/numeric/determinism failures are scientific
        # apparatus outcomes, not missing evidence.  Preserve one small raw
        # terminal payload so the independent afterany publisher can classify
        # it without inventing arm values.  Preflight/config errors above still
        # fail the allocation normally.
        case_dir = Path(args.output_root) / args.run_id / CASE_ID
        output = case_dir / "constrained-flow-payload.json"
        legacy_payload = case_dir / "canary-payload.json"
        failure = {
            "schema_version": "1.0",
            "payload_type": PAYLOAD_TYPE,
            "payload_variant": "terminal_apparatus_failure",
            "status": "apparatus_inconclusive",
            "case_id": CASE_ID,
            "run_id": args.run_id,
            "config": {
                "constrained_flow_path": str(Path(args.config)),
                "constrained_flow_sha256": file_sha256(args.config),
                "constrained_flow_scientific_hash": constrained_flow_scientific_config_hash(
                    constrained_value
                ),
                "legacy_path": str(Path(args.legacy_config)),
                "legacy_sha256": file_sha256(args.legacy_config),
            },
            "failure": {
                "stage": "paired_transport_execution",
                "error_type": type(error).__name__,
                "message": str(error),
                "finite_failure_is_infeasibility": False,
                "numeric_failure_is_method_negative": False,
            },
            "legacy_payload": {
                "path": str(legacy_payload),
                "exists": legacy_payload.is_file(),
                "sha256": (
                    file_sha256(legacy_payload)
                    if legacy_payload.is_file()
                    else None
                ),
            },
            "provenance": {
                "git_commit": os.environ.get("EXPECTED_GIT_COMMIT"),
                "source_node": os.uname().nodename.split(".", 1)[0],
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            "outcome": {
                "status": "apparatus_inconclusive",
                "scientific_claim_allowed": False,
                "infeasibility_claim_allowed": False,
                "collision_or_progress_claim_allowed": False,
                "probe_training_authorized": False,
                "automatic_next_gate_authorized": False,
            },
            "simulator_use": {
                "policy_generated_action_steps_executed": 0,
                "teacher_generated_action_steps_executed": 0,
                "efficacy_rollouts_executed": 0,
                "simulator_efficacy_evaluated": False,
            },
        }
        atomic_write_json(output, failure)
        status = "apparatus_inconclusive"
    print(f"{status} {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
