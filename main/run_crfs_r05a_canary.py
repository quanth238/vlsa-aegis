#!/usr/bin/env python3
"""Run the fixed R05A inverse-flow allocation canary payload stage."""

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
    run_r05a_canary,
)
from crfs_oracle.runner import oracle_config_from_mapping


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--r02-raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    return parser


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
        raise SystemExit("R05A allocation canary is fixed to row zero of the three-case manifest")
    case = cases[0]
    if case.get("case_id") != CASE_ID or case.get("group_id") != GROUP_ID:
        raise SystemExit("R05A allocation canary case/group identity changed")
    root = Path(__file__).resolve().parents[1]
    value = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("R05A canary config must be an object")
    oracle = oracle_config_from_mapping(
        value,
        host=args.host,
        port=args.port,
        checkpoint_id=args.checkpoint_id,
        checkpoint_sha256=args.checkpoint_sha256,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    config = r05a_canary_config_from_mapping(
        value,
        oracle,
        repo_root=root,
        source_r02_results_root=args.r02_raw_root,
        config_file_sha256=file_sha256(args.config),
    )
    output, status = run_r05a_canary(
        case,
        config,
        repo_root=root,
        input_manifest_sha256=file_sha256(args.manifest),
    )
    print(f"{status} {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
