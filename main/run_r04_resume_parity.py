#!/usr/bin/env python3
"""Run or independently validate one opt-in R04B resume/edit parity case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from crfs_harness.artifacts import file_sha256, load_json, validate_jsonl_unique
from crfs_harness.manifest import validate_case
from crfs_oracle.r04_resume import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CONFIG_FILE_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_R04A_VALIDATION_SUMMARY_SHA256,
    r04_resume_config_from_mapping,
    run_r04_resume_case,
    validate_r04_resume_result,
)
from crfs_oracle.runner import oracle_config_from_mapping


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-artifact")
    parser.add_argument(
        "--expected-config-file-sha256", default=EXPECTED_CONFIG_FILE_SHA256
    )
    parser.add_argument(
        "--expected-manifest-sha256", default=EXPECTED_MANIFEST_SHA256
    )
    parser.add_argument(
        "--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256
    )
    parser.add_argument(
        "--expected-r04a-validation-summary-sha256",
        default=EXPECTED_R04A_VALIDATION_SUMMARY_SHA256,
    )
    parser.add_argument("--expected-source-slurm-job-id")
    parser.add_argument("--expected-source-slurm-array-job-id")
    parser.add_argument("--expected-source-slurm-array-task-id")
    parser.add_argument("--manifest")
    parser.add_argument("--config")
    parser.add_argument("--output-root")
    parser.add_argument("--run-id")
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--checkpoint-sha256")
    return parser


def _validate_artifact(args: argparse.Namespace) -> int:
    artifact = Path(args.validate_artifact).expanduser().resolve()
    errors: list[str] = []
    artifact_sha256: str | None = None
    try:
        value = load_json(artifact)
        artifact_sha256 = file_sha256(artifact)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        value = None
        errors.append(f"cannot read artifact: {error}")
    if not isinstance(value, Mapping):
        if value is not None:
            errors.append("artifact must contain a JSON object")
    else:
        try:
            errors.extend(validate_r04_resume_result(value))
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError) as error:
            errors.append(f"semantic validator raised on malformed artifact: {error}")
        source = value.get("source_evidence")
        if not isinstance(source, Mapping):
            source = {}
        expected_source = {
            "config_file_sha256": args.expected_config_file_sha256,
            "input_manifest_sha256": args.expected_manifest_sha256,
            "checkpoint_sha256": args.expected_checkpoint_sha256,
        }
        for key, expected in expected_source.items():
            if source.get(key) != expected:
                errors.append(f"source_evidence.{key} differs from independent expectation")
        summary = source.get("r04a_validation_summary")
        if not isinstance(summary, Mapping) or summary.get("sha256") != (
            args.expected_r04a_validation_summary_sha256
        ):
            errors.append(
                "source_evidence.r04a_validation_summary differs from independent expectation"
            )
        provenance = value.get("provenance")
        if not isinstance(provenance, Mapping):
            provenance = {}
        expected_slurm = {
            "slurm_job_id": args.expected_source_slurm_job_id,
            "slurm_array_job_id": args.expected_source_slurm_array_job_id,
            "slurm_array_task_id": args.expected_source_slurm_array_task_id,
        }
        for key, expected in expected_slurm.items():
            if expected is None:
                errors.append(f"independent validation requires expected {key}")
            elif provenance.get(key) != expected:
                errors.append(f"provenance.{key} differs from expected source allocation")
    report: dict[str, Any] = {
        "artifact": str(artifact),
        "artifact_sha256": artifact_sha256,
        "validation_error_count": len(errors),
        "validation_errors": errors,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if not errors else 1


def _require_run_arguments(args: argparse.Namespace) -> None:
    required = (
        "manifest",
        "config",
        "output_root",
        "run_id",
        "case_index",
        "checkpoint_id",
        "checkpoint_sha256",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(
            "R04B run mode missing required arguments: " + ", ".join(missing)
        )


def main() -> int:
    args = _parser().parse_args()
    if args.validate_artifact:
        missing_source_identity = [
            option
            for option, value in (
                ("--expected-source-slurm-job-id", args.expected_source_slurm_job_id),
                (
                    "--expected-source-slurm-array-job-id",
                    args.expected_source_slurm_array_job_id,
                ),
                (
                    "--expected-source-slurm-array-task-id",
                    args.expected_source_slurm_array_task_id,
                ),
            )
            if value is None
        ]
        if missing_source_identity:
            raise SystemExit(
                "R04B validation mode requires all source Slurm identity arguments: "
                + ", ".join(missing_source_identity)
            )
        return _validate_artifact(args)
    _require_run_arguments(args)

    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    config_file_sha256 = file_sha256(config_path)
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("R04B config must contain a JSON object")

    manifest_path = Path(args.manifest).resolve()
    cases, errors = validate_jsonl_unique(manifest_path, "case_id")
    for case in cases:
        errors.extend(
            f"{case.get('case_id', '<unknown>')}: {error}"
            for error in validate_case(case)
        )
    if errors:
        raise SystemExit("invalid R04B manifest: " + "; ".join(errors))
    if not 0 <= args.case_index < len(cases):
        raise SystemExit(
            f"case index {args.case_index} outside manifest length {len(cases)}"
        )
    declared_manifest = value.get("manifest")
    if not isinstance(declared_manifest, str) or not declared_manifest:
        raise SystemExit("R04B config must declare its immutable manifest")
    declared_path = Path(declared_manifest).expanduser()
    if not declared_path.is_absolute():
        declared_path = (root / declared_path).resolve()
    else:
        declared_path = declared_path.resolve()
    if declared_path != manifest_path:
        raise SystemExit("R04B command manifest differs from the frozen config path")
    manifest_sha256 = file_sha256(manifest_path)
    if value.get("manifest_sha256") != manifest_sha256:
        raise SystemExit("R04B command manifest differs from the frozen config hash")
    if value.get("checkpoint_sha256") != args.checkpoint_sha256:
        raise SystemExit("R04B CLI checkpoint differs from the frozen config hash")

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
    config = r04_resume_config_from_mapping(
        runtime_value,
        oracle,
        repo_root=root,
        config_file_sha256=config_file_sha256,
    )
    if args.case_index != config.smoke_case_index:
        raise SystemExit("R04B runner is restricted to the frozen smoke_case_index")
    case = cases[args.case_index]
    if not (
        case.get("task_suite") == value.get("task_suite")
        and case.get("safety_level") == value.get("safety_level")
        and int(case.get("task_index", -1)) == int(value.get("task_index", -2))
    ):
        raise SystemExit("R04B smoke case task identity differs from the frozen config")

    output, status = run_r04_resume_case(
        case,
        config,
        repo_root=root,
        input_manifest_sha256=manifest_sha256,
    )
    print(f"{status} {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
