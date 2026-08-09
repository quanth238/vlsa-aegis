#!/usr/bin/env python3
"""Exactly audit the paired action-188 boundary-capacity MLP projections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.boundary_capacity import (
    BOUNDARY_RESULT_SCHEMA,
    load_boundary_capacity_config,
)
from main.multilink_ellipsoid.execution_margin_nn import load_model_artifact
from scripts.evaluate_distal_execution_margin_nn_e05 import (
    _projection_audit,
    _verify_sources,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    CASE_ID,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)
from scripts.train_distal_boundary_capacity_e05 import TRAINING_SCHEMA


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    false_safe_result_path: Path,
    obstacle_discovery_result_path: Path,
    baseline_nn_result_path: Path,
    geometry_config_path: Path,
    exact_box_config_path: Path,
    config_path: Path,
    expected_commit: str,
    dataset_path: Path,
    dataset_result_path: Path,
    dataset_validation_path: Path,
    training_path: Path,
) -> dict[str, Any]:
    from main.evaluate_safelibero_aegis import _runtime_imports, read_jsonl, validate_case_row
    from main.multilink_ellipsoid.obstacle_primitives import load_obstacle_primitive_config
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    started = time.perf_counter_ns()
    config = load_boundary_capacity_config(config_path)
    geometry_config = load_shadow_config(geometry_config_path)
    exact_box_config = load_obstacle_primitive_config(exact_box_config_path)
    archived, false_safe, discovery = _verify_sources(
        archived_path=archived_path,
        false_safe_result_path=false_safe_result_path,
        obstacle_discovery_result_path=obstacle_discovery_result_path,
        geometry_config_path=geometry_config_path,
        exact_box_config_path=exact_box_config_path,
        config=config,
    )
    baseline = _load(baseline_nn_result_path)
    _require(
        _file_sha256(baseline_nn_result_path)
        == config["immutable_sources"]["baseline_nn_result_file_sha256"]
        and baseline.get("result_payload_sha256")
        == config["immutable_sources"]["baseline_nn_result_payload_sha256"]
        and baseline.get("config", {}).get("config_file_sha256")
        == config["immutable_sources"]["previous_nn_config_file_sha256"]
        and baseline.get("research_direction_go") is False,
        "boundary-capacity immutable nominal-data baseline differs",
    )
    dataset = _load(dataset_path)
    dataset_result = _load(dataset_result_path)
    dataset_validation = _load(dataset_validation_path)
    _require(
        dataset.get("dataset_payload_sha256")
        == _hash_without(dataset, "dataset_payload_sha256")
        and dataset.get("source_commit") == expected_commit
        and dataset_result.get("source", {}).get("commit") == expected_commit
        and dataset_result.get("dataset", {}).get("file_sha256")
        == _file_sha256(dataset_path)
        and dataset_validation.get("status") == "valid"
        and dataset_validation.get("source_commit") == expected_commit
        and dataset_validation.get("result_file_sha256")
        == _file_sha256(dataset_result_path)
        and dataset_validation.get("dataset_file_sha256")
        == _file_sha256(dataset_path)
        and dataset_validation.get("neural_training_authorized") is True,
        "boundary-capacity dataset gate differs",
    )
    training = _load(training_path)
    _require(
        training.get("schema_version") == TRAINING_SCHEMA
        and training.get("status") == "complete"
        and training.get("source", {}).get("commit") == expected_commit
        and training.get("training_payload_sha256")
        == _hash_without(training, "training_payload_sha256")
        and training.get("dataset", {}).get("file_sha256")
        == _file_sha256(dataset_path),
        "boundary-capacity training artifact differs",
    )
    matches = [
        row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID
    ]
    _require(len(matches) == 1, "boundary-capacity manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    arm_results = {}
    for arm in config["training"]["arms"]:
        arm_training = training.get("arms", {}).get(arm, {})
        model_identity = arm_training.get("model_artifact", {})
        model_path = Path(model_identity.get("path", "")).resolve()
        _require(
            model_path.is_file()
            and not model_path.is_symlink()
            and model_identity.get("file_sha256") == _file_sha256(model_path)
            and model_path.parent == training_path.parent.resolve(),
            "boundary-capacity model artifact differs: %s" % arm,
        )
        model, model_state = load_model_artifact(model_path, device="cpu")
        projection = _projection_audit(
            runtime=runtime,
            case=case,
            archived=archived,
            actions=false_safe["actions"],
            geometry_config=geometry_config,
            exact_box_config=exact_box_config,
            config=config,
            model=model,
            model_state=model_state,
            step_override=int(config["state"]["step"]),
        )
        model_gate = bool(
            arm_training.get("training", {}).get("model_capacity_gate_pass")
        )
        projection_gate = bool(projection["projection_gate_pass"])
        arm_results[arm] = {
            "model_artifact": model_identity,
            "training": arm_training.get("training"),
            "primary_projection": projection,
            "decision": {
                "model_capacity_gate_pass": model_gate,
                "projection_gate_pass": projection_gate,
                "arm_capacity_gate_pass": bool(model_gate and projection_gate),
            },
        }
    gradient_go = bool(
        arm_results["boundary_margin_gradient"]["decision"][
            "arm_capacity_gate_pass"
        ]
    )
    if not arm_results["boundary_margin_gradient"]["decision"][
        "model_capacity_gate_pass"
    ]:
        stop_reason = "gradient_supervised_model_failed_same_state_capacity_gate"
    elif not arm_results["boundary_margin_gradient"]["decision"][
        "projection_gate_pass"
    ]:
        stop_reason = "gradient_supervised_projection_failed_exact_cloned_OSC_gate"
    else:
        stop_reason = None
    baseline_training = baseline.get("training", {})
    baseline_test = baseline_training.get("split_metrics", {}).get("test", {})
    result = {
        "schema_version": BOUNDARY_RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "case_id": CASE_ID,
        "claim_scope": config["claim_scope"],
        "source": source,
        "allocation": allocation,
        "config": config,
        "immutable_sources": {
            "archived_table1": str(archived_path),
            "false_safe_action_ledger": str(false_safe_result_path),
            "exact_box_discovery": str(obstacle_discovery_result_path),
            "baseline_nominal_data_nn_result": str(baseline_nn_result_path),
            "discovery_schema_version": discovery.get("schema_version"),
            "read_only": True,
        },
        "dataset_gate": {
            "dataset": {
                "path": str(dataset_path),
                "file_sha256": _file_sha256(dataset_path),
                "payload_sha256": dataset["dataset_payload_sha256"],
                "record_count": len(dataset["records"]),
            },
            "result_path": str(dataset_result_path),
            "result_file_sha256": _file_sha256(dataset_result_path),
            "validation_path": str(dataset_validation_path),
            "validation_file_sha256": _file_sha256(dataset_validation_path),
            "neural_training_authorized": True,
        },
        "training_artifact": {
            "path": str(training_path),
            "file_sha256": _file_sha256(training_path),
            "payload_sha256": training["training_payload_sha256"],
        },
        "immutable_nominal_data_baseline": {
            "result_file_sha256": _file_sha256(baseline_nn_result_path),
            "model_gate_pass": baseline_training.get(
                "model_learnability_gate_pass"
            ),
            "test_rmse_m": baseline_test.get("rmse_m"),
            "test_current_clearance_baseline_rmse_m": baseline_test.get(
                "current_clearance_baseline_rmse_m"
            ),
            "research_direction_go": baseline.get("research_direction_go"),
            "stop_reason": baseline.get("stop_reason"),
        },
        "arms": arm_results,
        "decision": {
            "research_direction_go": gradient_go,
            "stop_reason": stop_reason,
            "interpretation": (
                "same_state_local_capacity_only_not_generalization_or_closed_loop"
            ),
        },
        "research_direction_go": gradient_go,
        "stop_reason": stop_reason,
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        "failure": None,
    }
    result["result_payload_sha256"] = _hash_without(
        result, "result_payload_sha256"
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--false-safe-result", type=Path, required=True)
    parser.add_argument("--obstacle-discovery-result", type=Path, required=True)
    parser.add_argument("--baseline-nn-result", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-result", type=Path, required=True)
    parser.add_argument("--dataset-validation", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        false_safe_result_path=args.false_safe_result.resolve(),
        obstacle_discovery_result_path=args.obstacle_discovery_result.resolve(),
        baseline_nn_result_path=args.baseline_nn_result.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        exact_box_config_path=args.exact_box_config.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
        dataset_path=args.dataset.resolve(),
        dataset_result_path=args.dataset_result.resolve(),
        dataset_validation_path=args.dataset_validation.resolve(),
        training_path=args.training.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result["decision"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
