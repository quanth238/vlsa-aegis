#!/usr/bin/env python3
"""Run one manifest-indexed CRFS oracle case in the baseline SafeLIBERO stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crfs_harness.artifacts import validate_jsonl_unique
from crfs_harness.manifest import validate_case
from crfs_oracle.runner import oracle_config_from_mapping, run_case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args()

    cases, errors = validate_jsonl_unique(args.manifest, "case_id")
    if not 0 <= args.case_index < len(cases):
        raise SystemExit(f"case index {args.case_index} outside manifest length {len(cases)}")
    errors.extend(validate_case(cases[args.case_index]))
    if errors:
        raise SystemExit("invalid manifest: " + "; ".join(errors))
    with Path(args.config).open(encoding="utf-8") as handle:
        value = json.load(handle)
    config = oracle_config_from_mapping(
        value,
        host=args.host,
        port=args.port,
        checkpoint_id=args.checkpoint_id,
        checkpoint_sha256=args.checkpoint_sha256,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    output, status = run_case(cases[args.case_index], config, repo_root=Path(__file__).resolve().parents[1])
    print(f"{status} {output}")
    return 0 if status in {
        "completed",
        "infeasible",
        "measurement_only_completed",
        "skipped_valid_completion",
        "skipped_valid_measurement",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
