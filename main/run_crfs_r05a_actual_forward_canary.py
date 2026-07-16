#!/usr/bin/env python3
"""Run the preregistered AF-00A actual-forward teacher canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crfs_harness.artifacts import file_sha256, validate_jsonl_unique
from crfs_harness.manifest import validate_case
from crfs_oracle.r05a_canary import (
    CASE_ID,
    GROUP_ID,
    r05a_canary_config_from_mapping,
)
from crfs_oracle.r05a_actual_forward_canary import run_actual_forward_canary
from crfs_oracle.runner import oracle_config_from_mapping


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True, help="released AF-00A config")
    parser.add_argument(
        "--legacy-config",
        required=True,
        help="frozen R05A config used only for the common source/replay apparatus",
    )
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
        raise SystemExit("invalid AF-00A manifest: " + "; ".join(errors))
    if len(cases) != 3 or args.case_index != 0:
        raise SystemExit("AF-00A is fixed to row zero of the three-case manifest")
    case = cases[0]
    if case.get("case_id") != CASE_ID or case.get("group_id") != GROUP_ID:
        raise SystemExit("AF-00A case/group identity changed")

    actual = _object(args.config, name="AF-00A config")
    if actual.get("ready_to_run") is not True or actual.get("blocked_on") != []:
        raise SystemExit("AF-00A config has not been released for allocation execution")
    release = actual.get("execution_release")
    if not isinstance(release, dict) or release.get("run_id") != args.run_id:
        raise SystemExit("AF-00A run ID differs from the exact execution release")
    if actual.get("flow_contract", {}).get("ordinary_residual_schedule_path_only") is not True:
        raise SystemExit("AF-00A ordinary residual-schedule contract changed")
    if actual.get("flow_contract", {}).get("inverse_flow_teacher_allowed") is not False:
        raise SystemExit("AF-00A inverse-flow teacher must remain disabled")
    if actual.get("flow_contract", {}).get("autograd_allowed") is not False:
        raise SystemExit("AF-00A autograd must remain disabled")

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
    output, status = run_actual_forward_canary(
        case,
        actual,
        legacy_config,
        repo_root=root,
        input_manifest_sha256=file_sha256(args.manifest),
        actual_config_path=args.config,
        legacy_config_path=args.legacy_config,
        client=client,
    )
    print(f"{status} {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
