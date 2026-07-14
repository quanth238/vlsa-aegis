#!/usr/bin/env python3
"""Run one immutable R02 paired oracle-flow case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crfs_harness.artifacts import file_sha256, validate_jsonl_unique
from crfs_harness.manifest import validate_case
from crfs_oracle.r02_runner import (
    r02_config_from_mapping,
    run_r02_case,
)
from crfs_oracle.runner import oracle_config_from_mapping


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--r01-raw-root", required=True)
    parser.add_argument("--r01-summary", required=True)
    parser.add_argument("--r01-summary-sha256", required=True)
    parser.add_argument("--parity-artifact", required=True)
    parser.add_argument("--parity-artifact-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    cases, errors = validate_jsonl_unique(args.manifest, "case_id")
    if not 0 <= args.case_index < len(cases):
        raise SystemExit(
            f"case index {args.case_index} outside manifest length {len(cases)}"
        )
    for case in cases:
        errors.extend(
            f"{case.get('case_id', '<unknown>')}: {error}"
            for error in validate_case(case)
        )
    if errors:
        raise SystemExit("invalid manifest: " + "; ".join(errors))
    if len(cases) != 20 or len({case.get("group_id") for case in cases}) != 20:
        raise SystemExit("R02 requires the complete frozen 20-case/20-group manifest")

    root = Path(__file__).resolve().parents[1]
    value = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("R02 config must be a JSON object")
    declared_manifest = value.get("manifest")
    if not isinstance(declared_manifest, str) or Path(declared_manifest).name != Path(
        args.manifest
    ).name:
        raise SystemExit("R02 command manifest differs from the frozen config manifest")
    settings = value.get("r02")
    if not isinstance(settings, dict):
        raise SystemExit("R02 config has no r02 object")
    settings = dict(settings)
    settings.update(
        {
            "r01_results_root": args.r01_raw_root,
            "r01_summary_artifact": args.r01_summary,
            "r01_summary_sha256": args.r01_summary_sha256,
            "sampler_parity_artifact": args.parity_artifact,
            "sampler_parity_sha256": args.parity_artifact_sha256,
        }
    )
    runtime_value = dict(value)
    runtime_value["r02"] = settings
    runtime_value.setdefault("intervention_step", 5)
    runtime_value.setdefault("optimizer_max_iterations", 1)
    runtime_value.setdefault("measurement_repeats", int(settings.get("simulator_repeats", 2)))
    runtime_value.setdefault("stop_after_measurement", False)
    oracle = oracle_config_from_mapping(
        runtime_value,
        host=args.host,
        port=args.port,
        checkpoint_id=args.checkpoint_id,
        checkpoint_sha256=args.checkpoint_sha256,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    config = r02_config_from_mapping(runtime_value, oracle, repo_root=root)
    manifest_sha256 = file_sha256(args.manifest)
    if set(config.r01_result_hashes) != {str(case["case_id"]) for case in cases}:
        raise SystemExit("R02 manifest identities differ from the bound R01 result set")
    output, status = run_r02_case(
        cases[args.case_index],
        config,
        repo_root=root,
        input_manifest_sha256=manifest_sha256,
    )
    print(f"{status} {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
