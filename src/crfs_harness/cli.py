"""Command-line entry point for local and HPC harness operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .aggregate import aggregate_results
from .artifacts import atomic_write_json, load_json, validate_case_result, validate_jsonl_unique
from .manifest import build_cases, validate_case
from .synthetic import run_synthetic_case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crfs-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("build-manifest", help="Build a deterministic JSONL case manifest")
    manifest_parser.add_argument("--output", required=True)
    manifest_parser.add_argument("--task-suite", default="safelibero_spatial")
    manifest_parser.add_argument("--safety-level", choices=("I", "II"), default="II")
    manifest_parser.add_argument("--task-index", type=int, default=0)
    manifest_parser.add_argument("--episodes", type=int, nargs="+", required=True)
    manifest_parser.add_argument("--seeds-per-episode", type=int, default=1)
    manifest_parser.add_argument("--seed-namespace", default="crfs-oracle-v1")

    synthetic_parser = subparsers.add_parser("synthetic", help="Run the non-research end-to-end fixture")
    synthetic_parser.add_argument("--manifest", required=True)
    synthetic_parser.add_argument("--output-root", required=True)
    synthetic_parser.add_argument("--run-id", required=True)
    synthetic_parser.add_argument("--case-index", type=int)
    synthetic_parser.add_argument("--overwrite", action="store_true")

    validate_parser = subparsers.add_parser("validate-result", help="Validate one results.json")
    validate_parser.add_argument("path")

    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate validated case results")
    aggregate_parser.add_argument("--results-root", required=True)
    aggregate_parser.add_argument("--output", required=True)
    aggregate_parser.add_argument("--bootstrap-seed", type=int, default=0)

    arguments = parser.parse_args(argv)
    if arguments.command == "build-manifest":
        return _build_manifest(arguments)
    if arguments.command == "synthetic":
        return _synthetic(arguments)
    if arguments.command == "validate-result":
        return _validate_result(arguments)
    if arguments.command == "aggregate":
        return _aggregate(arguments)
    parser.error(f"Unknown command: {arguments.command}")
    return 2


def _build_manifest(arguments: argparse.Namespace) -> int:
    cases = build_cases(
        arguments.task_suite,
        arguments.safety_level,
        arguments.task_index,
        arguments.episodes,
        arguments.seeds_per_episode,
        arguments.seed_namespace,
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True) + "\n")
    temporary.replace(output)
    print(f"wrote {len(cases)} immutable cases to {output}")
    return 0


def _synthetic(arguments: argparse.Namespace) -> int:
    cases, errors = validate_jsonl_unique(arguments.manifest, "case_id")
    for case in cases:
        errors.extend(f"{case.get('case_id', '<unknown>')}: {error}" for error in validate_case(case))
    if errors:
        print("WHAT: manifest validation failed", file=sys.stderr)
        print("WHY: paired oracle arms require unique immutable case identities", file=sys.stderr)
        print("FIX: rebuild with the build-manifest command", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    selected = cases if arguments.case_index is None else [cases[arguments.case_index]]
    statuses: dict[str, int] = {}
    for case in selected:
        path, status = run_synthetic_case(case, arguments.output_root, arguments.run_id, overwrite=arguments.overwrite)
        statuses[status] = statuses.get(status, 0) + 1
        print(f"{case['case_id']} {status} {path}")
    print(json.dumps(statuses, sort_keys=True))
    return 0


def _validate_result(arguments: argparse.Namespace) -> int:
    value = load_json(arguments.path)
    errors = validate_case_result(value)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"valid: {arguments.path}")
    return 0


def _aggregate(arguments: argparse.Namespace) -> int:
    paths = sorted(Path(arguments.results_root).glob("*/results.json"))
    results = []
    errors = []
    for path in paths:
        value = load_json(path)
        result_errors = validate_case_result(value)
        if result_errors:
            errors.extend(f"{path}: {error}" for error in result_errors)
        else:
            results.append(value)
    if errors:
        print("WHAT: invalid case results prevent aggregation", file=sys.stderr)
        print("WHY: partial files must never count as research completion", file=sys.stderr)
        print("FIX: rerun or repair only the listed cases", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    aggregate = aggregate_results(results, arguments.bootstrap_seed)
    atomic_write_json(arguments.output, aggregate)
    print(f"aggregated {len(results)} validated results into {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
