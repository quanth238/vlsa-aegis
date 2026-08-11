#!/usr/bin/env python3
"""Decompose nominal-safety error into execution and geometry components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from main.multilink_ellipsoid.factorized_execution_pilot import (
    dataset_arrays, payload_sha256,
)
from main.multilink_ellipsoid.factorized_nominal_safety_audit import (
    RESULT_SCHEMA, audit_formula, load_formula_audit_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trajectory-metadata", type=Path, required=True)
    parser.add_argument("--trajectory-array", type=Path, required=True)
    parser.add_argument("--terminal-records", type=Path, required=True)
    parser.add_argument("--terminal-result", type=Path, required=True)
    parser.add_argument("--terminal-validation", type=Path, required=True)
    parser.add_argument("--structured-flat-model", type=Path, required=True)
    parser.add_argument("--structured-time-model", type=Path, required=True)
    parser.add_argument("--structured-predictions", type=Path, required=True)
    parser.add_argument("--structured-result", type=Path, required=True)
    parser.add_argument("--structured-validation", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)


def resolved_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        key: value.resolve() for key, value in {
            "repo": args.repo_root, "config": args.config,
            "metadata": args.trajectory_metadata, "array": args.trajectory_array,
            "terminal_records": args.terminal_records,
            "terminal_result": args.terminal_result,
            "terminal_validation": args.terminal_validation,
            "structured_flat_model": args.structured_flat_model,
            "structured_time_model": args.structured_time_model,
            "structured_predictions": args.structured_predictions,
            "structured_result": args.structured_result,
            "structured_validation": args.structured_validation,
        }.items()
    }


def load_and_audit(
    args: argparse.Namespace,
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    import numpy as np

    paths = resolved_paths(args)
    config = load_formula_audit_config(paths["config"])
    source = config["immutable_source"]
    for key, path_key in (
        ("trajectory_array_file_sha256", "array"),
        ("trajectory_metadata_file_sha256", "metadata"),
        ("terminal_bias_records_file_sha256", "terminal_records"),
        ("terminal_bias_result_file_sha256", "terminal_result"),
        ("terminal_bias_validation_file_sha256", "terminal_validation"),
        ("structured_flat_model_file_sha256", "structured_flat_model"),
        ("structured_time_model_file_sha256", "structured_time_model"),
        ("structured_predictions_file_sha256", "structured_predictions"),
        ("structured_result_file_sha256", "structured_result"),
        ("structured_validation_file_sha256", "structured_validation"),
    ):
        _require(
            _file_sha256(paths[path_key]) == source[key],
            "nominal-safety immutable source differs",
        )
    terminal_result = _load(paths["terminal_result"])
    terminal_validation = _load(paths["terminal_validation"])
    structured_result = _load(paths["structured_result"])
    structured_validation = _load(paths["structured_validation"])
    _require(
        terminal_result.get("result_payload_sha256")
        == source["terminal_bias_result_payload_sha256"]
        and terminal_validation.get("validation_payload_sha256")
        == source["terminal_bias_validation_payload_sha256"]
        and terminal_validation.get("valid") is True
        and structured_result.get("result_payload_sha256")
        == source["structured_result_payload_sha256"]
        and structured_validation.get("validation_payload_sha256")
        == source["structured_validation_payload_sha256"]
        and structured_validation.get("valid") is True,
        "nominal-safety validated source differs",
    )
    metadata = _load(paths["metadata"])
    _require(
        metadata.get("array_dataset", {}).get("file_sha256")
        == _file_sha256(paths["array"]),
        "nominal-safety trajectory pair differs",
    )
    trajectory_archive = np.load(paths["array"], allow_pickle=False)
    arrays = dataset_arrays(metadata, trajectory_archive)
    terminal_archive = np.load(paths["terminal_records"], allow_pickle=False)
    records = {key: terminal_archive[key] for key in terminal_archive.files}
    prediction_archive = np.load(
        paths["structured_predictions"], allow_pickle=False
    )
    predictions = {key: prediction_archive[key] for key in prediction_archive.files}
    flat_archive = np.load(paths["structured_flat_model"], allow_pickle=False)
    time_archive = np.load(paths["structured_time_model"], allow_pickle=False)
    audit = audit_formula(
        arrays=arrays, records=records, structured_predictions=predictions,
        flat_output_shape=flat_archive["output_shape"],
        time_output_shape=time_archive["output_shape"], config=config,
    )
    return paths, config, audit


def main() -> int:
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    paths, config, audit = load_and_audit(args)
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        **audit,
        "immutable_artifact_receipt": {
            key: _file_sha256(path) for key, path in paths.items()
            if key not in {"repo", "config"}
        },
        "forbidden_action_receipt": {
            key: False for key in config["forbidden_actions"]
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "formula_receipt": result["formula_receipt"],
        "test": result["split_metrics"]["test"],
        "current_structured_models": result["current_structured_models"],
        "decision": result["decision"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
