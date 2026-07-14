#!/usr/bin/env python3
"""Run one immutable R03A strong-analytic necessity-test case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crfs_harness.artifacts import file_sha256, validate_jsonl_unique
from crfs_harness.manifest import validate_case
from crfs_oracle.r03a_runner import r03a_config_from_mapping, run_r03a_case
from crfs_oracle.r03a_validation import r03a_config_hash
from crfs_oracle.runner import oracle_config_from_mapping


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--r02-raw-root", required=True)
    parser.add_argument("--r03-summary", required=True)
    parser.add_argument("--r03-summary-sha256", required=True)
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
    if len(cases) != 17 or len({case.get("group_id") for case in cases}) != 17:
        raise SystemExit("R03A requires the complete frozen 17-case/17-group subset")

    root = Path(__file__).resolve().parents[1]
    config_file_sha256 = file_sha256(args.config)
    value = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("R03A config must be a JSON object")
    scientific_config_hash = r03a_config_hash(value)
    declared_manifest = value.get("manifest")
    if not isinstance(declared_manifest, str) or Path(declared_manifest).name != Path(
        args.manifest
    ).name:
        raise SystemExit("R03A command manifest differs from the frozen config manifest")
    settings = value.get("r03a")
    if not isinstance(settings, dict):
        raise SystemExit("R03A config has no r03a object")
    runtime_settings = dict(settings)
    runtime_settings.update(
        {
            "source_r02_results_root": args.r02_raw_root,
            "source_r03_summary_artifact": args.r03_summary,
            "source_r03_summary_sha256": args.r03_summary_sha256,
        }
    )
    runtime_value = dict(value)
    runtime_value["r03a"] = runtime_settings
    runtime_value.setdefault("intervention_step", 5)
    runtime_value.setdefault("optimizer_max_iterations", 1)
    runtime_value.setdefault(
        "measurement_repeats", int(runtime_settings.get("simulator_repeats", 2))
    )
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
    config = r03a_config_from_mapping(
        runtime_value,
        oracle,
        repo_root=root,
        config_file_sha256=config_file_sha256,
        scientific_config_hash=scientific_config_hash,
    )
    manifest_sha256 = file_sha256(args.manifest)
    case_ids = tuple(str(case["case_id"]) for case in cases)
    if case_ids != config.eligible_case_ids:
        raise SystemExit("R03A manifest order/identities differ from the bound R03 subset")
    output, status = run_r03a_case(
        cases[args.case_index],
        config,
        repo_root=root,
        input_manifest_sha256=manifest_sha256,
    )
    print(f"{status} {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
