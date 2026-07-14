#!/usr/bin/env python3
"""Run a restartable range of R00 cases against one allocated policy server."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from crfs_harness.artifacts import atomic_write_json, validate_jsonl_unique
from crfs_harness.manifest import validate_case
from crfs_oracle.progress_calibration import (
    reach_calibration_config_from_mapping,
    run_reach_calibration_case,
)
from crfs_oracle.runner import SafeLiberoCase, oracle_config_from_mapping


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case-start", type=int, required=True)
    parser.add_argument("--case-end", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args()

    cases, errors = validate_jsonl_unique(args.manifest, "case_id")
    if not 0 <= args.case_start <= args.case_end < len(cases):
        raise SystemExit(f"case range [{args.case_start}, {args.case_end}] outside manifest length {len(cases)}")
    selected = cases[args.case_start : args.case_end + 1]
    for case in selected:
        errors.extend(f"{case.get('case_id', '<unknown>')}: {error}" for error in validate_case(case))
    if errors:
        raise SystemExit("invalid manifest: " + "; ".join(errors))
    value = json.loads(Path(args.config).read_text(encoding="utf-8"))
    oracle = oracle_config_from_mapping(
        value,
        host=args.host,
        port=args.port,
        checkpoint_id=args.checkpoint_id,
        checkpoint_sha256=args.checkpoint_sha256,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    config = reach_calibration_config_from_mapping(value, oracle)

    from openpi_client import websocket_client_policy

    counts: dict[str, int] = {}
    failures = 0
    root = Path(__file__).resolve().parents[1]
    client = websocket_client_policy.WebsocketClientPolicy(oracle.host, oracle.port)
    environment = SafeLiberoCase(selected[0], oracle)
    try:
        for case_index, case in enumerate(selected, start=args.case_start):
            try:
                output, status = run_reach_calibration_case(
                    case,
                    config,
                    repo_root=root,
                    client=client,
                    environment=environment,
                )
                counts[status] = counts.get(status, 0) + 1
                print(json.dumps({
                    "event": "reach_progress_case_complete",
                    "case_index": case_index,
                    "case_id": case["case_id"],
                    "status": status,
                    "output": str(output),
                }, sort_keys=True), flush=True)
            except Exception as error:
                failures += 1
                failure_path = Path(args.output_root) / args.run_id / str(case["case_id"]) / "batch-failure.json"
                atomic_write_json(failure_path, {
                    "status": "failed",
                    "case_index": case_index,
                    "case_id": case["case_id"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                })
                traceback.print_exc()
    finally:
        environment.close()
    print(json.dumps({"event": "reach_progress_batch_complete", "counts": counts, "failures": failures}, sort_keys=True))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
