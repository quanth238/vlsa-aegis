"""Command-line entry point for local and HPC harness operations."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

from .aggregate import aggregate_results
from .artifacts import atomic_write_json, load_json, validate_case_result, validate_jsonl_unique
from .manifest import build_cases, build_official_cases, validate_case
from .official_state import init_state_row_identity
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

    official_manifest_parser = subparsers.add_parser(
        "build-official-manifest",
        help="Build a manifest content-bound to released SafeLIBERO saved states",
    )
    official_manifest_parser.add_argument("--output", required=True)
    official_manifest_parser.add_argument("--task-suite", required=True)
    official_manifest_parser.add_argument("--safety-level", choices=("I", "II"), required=True)
    official_manifest_parser.add_argument("--task-index", type=int, required=True)
    official_manifest_parser.add_argument("--task-name", required=True)
    official_manifest_parser.add_argument("--init-states-root", required=True)
    official_manifest_parser.add_argument("--episodes", type=int, nargs="+", required=True)
    official_manifest_parser.add_argument("--seeds-per-episode", type=int, default=1)
    official_manifest_parser.add_argument("--seed-namespace", required=True)

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
    if arguments.command == "build-official-manifest":
        return _build_official_manifest(arguments)
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
    _write_manifest(arguments.output, cases)
    print(f"wrote {len(cases)} immutable cases to {Path(arguments.output)}")
    return 0


def _write_manifest(output_value: str | Path, cases: list[dict]) -> None:
    output = Path(output_value)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, sort_keys=True) + "\n")
    temporary.replace(output)


def _build_official_manifest(arguments: argparse.Namespace) -> int:
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise SystemExit(
            "build-official-manifest requires the SafeLIBERO PyTorch/NumPy environment"
        ) from error

    episodes = list(arguments.episodes)
    if len(set(episodes)) != len(episodes):
        raise SystemExit("official manifest episodes must be unique")
    relative_path = (
        f"{arguments.task_suite}/{arguments.task_name}_level_"
        f"{arguments.safety_level}.pruned_init"
    )
    init_states_root = Path(arguments.init_states_root).resolve()
    init_state_path = init_states_root.joinpath(*relative_path.split("/"))
    try:
        init_state_path.resolve().relative_to(init_states_root)
    except ValueError as error:
        raise SystemExit("official init-state path escapes --init-states-root") from error
    if not init_state_path.is_file():
        raise SystemExit(f"official init-state file does not exist: {init_state_path}")

    init_state_file_bytes = init_state_path.read_bytes()
    init_state_file_sha256 = hashlib.sha256(init_state_file_bytes).hexdigest()
    states = torch.load(io.BytesIO(init_state_file_bytes), map_location="cpu")
    row_identities = {}
    for episode_index in episodes:
        if episode_index < 0 or episode_index >= len(states):
            raise SystemExit(
                f"episode {episode_index} outside official state count {len(states)}"
            )
        row = states[episode_index]
        if hasattr(row, "detach"):
            row = row.detach()
        if hasattr(row, "cpu"):
            row = row.cpu()
        row_identities[episode_index] = init_state_row_identity(
            np.ascontiguousarray(np.asarray(row))
        )
    cases = build_official_cases(
        task_suite=arguments.task_suite,
        safety_level=arguments.safety_level,
        task_index=arguments.task_index,
        task_name=arguments.task_name,
        init_state_relative_path=relative_path,
        init_state_file_sha256=init_state_file_sha256,
        row_identities=row_identities,
        seeds_per_episode=arguments.seeds_per_episode,
        seed_namespace=arguments.seed_namespace,
    )
    _write_manifest(arguments.output, cases)
    print(
        f"wrote {len(cases)} official-state-bound immutable cases to "
        f"{Path(arguments.output)}"
    )
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
