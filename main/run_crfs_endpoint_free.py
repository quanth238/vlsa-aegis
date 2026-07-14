#!/usr/bin/env python3
"""Run one immutable R01 endpoint-free feasibility case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crfs_harness.artifacts import file_sha256, validate_jsonl_unique
from crfs_harness.manifest import validate_case
from crfs_oracle.endpoint_free_runner import (
    endpoint_free_config_from_mapping,
    run_endpoint_free_case,
)
from crfs_oracle.runner import oracle_config_from_mapping


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
    for case in cases:
        errors.extend(f"{case.get('case_id', '<unknown>')}: {error}" for error in validate_case(case))
    if errors:
        raise SystemExit("invalid manifest: " + "; ".join(errors))

    config_path = Path(args.config)
    value = json.loads(config_path.read_text(encoding="utf-8"))
    declared_manifest = value.get("manifest")
    if not isinstance(declared_manifest, str) or Path(declared_manifest).name != Path(args.manifest).name:
        raise SystemExit("R01 command manifest does not match the manifest frozen in its config")
    runtime_value = dict(value)
    runtime_value.setdefault("intervention_step", int(runtime_value["sampler_steps"]) // 2)
    planner = runtime_value.get("planner", {})
    runtime_value.setdefault(
        "optimizer_max_iterations",
        int(planner.get("optimizer_max_iterations", 1)) if isinstance(planner, dict) else 1,
    )
    oracle = oracle_config_from_mapping(
        runtime_value,
        host=args.host,
        port=args.port,
        checkpoint_id=args.checkpoint_id,
        checkpoint_sha256=args.checkpoint_sha256,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    manifest_sha256 = file_sha256(args.manifest)
    config = endpoint_free_config_from_mapping(
        value,
        oracle,
        repo_root=Path(__file__).resolve().parents[1],
        evaluation_manifest_sha256=manifest_sha256,
    )
    output, status = run_endpoint_free_case(
        cases[args.case_index],
        config,
        repo_root=Path(__file__).resolve().parents[1],
        input_manifest_sha256=manifest_sha256,
    )
    print(f"{status} {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
