#!/usr/bin/env python3
"""Run one opt-in R04A frozen-continuation label-contract case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crfs_harness.artifacts import file_sha256, validate_jsonl_unique
from crfs_harness.manifest import validate_case
from crfs_oracle.r04_labels import (
    r04_label_config_from_mapping,
    run_r04_label_case,
)
from crfs_oracle.runner import oracle_config_from_mapping


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-index", required=True, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    config_file_sha256 = file_sha256(config_path)
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("R04A config must contain a JSON object")

    manifest_path = Path(args.manifest).resolve()
    cases, errors = validate_jsonl_unique(manifest_path, "case_id")
    for case in cases:
        errors.extend(
            f"{case.get('case_id', '<unknown>')}: {error}"
            for error in validate_case(case)
        )
    if errors:
        raise SystemExit("invalid R04A manifest: " + "; ".join(errors))
    if not 0 <= args.case_index < len(cases):
        raise SystemExit(
            f"case index {args.case_index} outside manifest length {len(cases)}"
        )
    declared_manifest = value.get("manifest")
    if not isinstance(declared_manifest, str) or not declared_manifest:
        raise SystemExit("R04A config must declare its immutable manifest")
    declared_path = Path(declared_manifest).expanduser()
    if not declared_path.is_absolute():
        declared_path = (root / declared_path).resolve()
    else:
        declared_path = declared_path.resolve()
    if declared_path != manifest_path:
        raise SystemExit("R04A command manifest differs from the frozen config path")
    manifest_sha256 = file_sha256(manifest_path)
    if value.get("manifest_sha256") != manifest_sha256:
        raise SystemExit("R04A command manifest differs from the frozen config hash")
    if value.get("checkpoint_sha256") != args.checkpoint_sha256:
        raise SystemExit("R04A CLI checkpoint differs from the frozen config hash")

    runtime_value = dict(value)
    runtime_value.setdefault("response_matrix_m_per_action", None)
    oracle = oracle_config_from_mapping(
        runtime_value,
        host=args.host,
        port=args.port,
        checkpoint_id=args.checkpoint_id,
        checkpoint_sha256=args.checkpoint_sha256,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    config = r04_label_config_from_mapping(
        runtime_value,
        oracle,
        repo_root=root,
        config_file_sha256=config_file_sha256,
    )
    if args.case_index != config.smoke_case_index:
        raise SystemExit(
            "R04A apparatus runner is restricted to the frozen smoke_case_index"
        )
    case = cases[args.case_index]
    if not (
        case.get("task_suite") == value.get("task_suite")
        and case.get("safety_level") == value.get("safety_level")
        and int(case.get("task_index", -1)) == int(value.get("task_index", -2))
    ):
        raise SystemExit("R04A smoke case task identity differs from the frozen config")

    output, status = run_r04_label_case(
        case,
        config,
        repo_root=root,
        input_manifest_sha256=manifest_sha256,
    )
    print(f"{status} {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
