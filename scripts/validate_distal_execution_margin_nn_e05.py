#!/usr/bin/env python3
"""Validate one completed E05 execution-margin neural mechanism result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.execution_margin_nn import (
    EXECUTION_MARGIN_RESULT_SCHEMA,
    load_execution_margin_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    CASE_ID,
    _atomic_write,
    _file_sha256,
    _load,
    _require,
)


VALIDATION_SCHEMA = "vlsa_distal_execution_margin_nn_e05_validation.v1"


def _payload_hash(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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


def _safe_artifact(result_path: Path, recorded_path: str) -> Path:
    candidate = Path(recorded_path)
    _require(candidate.is_absolute(), "artifact path is not absolute")
    candidate = candidate.resolve()
    _require(
        candidate.parent == result_path.parent.resolve()
        and candidate.is_file()
        and not candidate.is_symlink(),
        "artifact is not a regular file in the immutable run root",
    )
    return candidate


def validate(
    result_path: Path,
    *,
    expected_commit: str,
    expected_config: Mapping[str, Any],
) -> dict[str, Any]:
    result = _load(result_path)
    _require(
        result.get("schema_version") == EXECUTION_MARGIN_RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and result.get("case_id") == CASE_ID
        and result.get("failure") is None,
        "execution-margin result identity differs",
    )
    recorded_payload = result.get("result_payload_sha256")
    _require(
        isinstance(recorded_payload, str)
        and recorded_payload == _payload_hash(result),
        "execution-margin result payload hash differs",
    )
    _require(result.get("config") == expected_config, "execution-margin config differs")
    source = result.get("source", {})
    allocation = result.get("allocation", {})
    _require(
        source.get("commit") == expected_commit
        and source.get("dirty") is False
        and str(allocation.get("slurm_job_id", "")).isdigit()
        and "H100" in str(allocation.get("device", {}).get("name", "")),
        "execution-margin source or H100 allocation differs",
    )
    geometry = result.get("geometry", {})
    robot = geometry.get("accepted_robot_and_released_ee", {})
    boxes = geometry.get("exact_obstacle_boxes", {})
    _require(
        robot.get("distal_ellipsoid_count") == 7
        and robot.get("total_constraint_geometry_count") == 8
        and boxes.get("exact_box_count") == 15
        and boxes.get("geometric_inflation") == 0.0,
        "execution-margin geometry authority differs",
    )
    dataset_identity = result.get("dataset", {})
    dataset_path = _safe_artifact(result_path, dataset_identity.get("path", ""))
    dataset = _load(dataset_path)
    dataset_payload = dataset.get("dataset_payload_sha256")
    dataset_without_hash = dict(dataset)
    dataset_without_hash.pop("dataset_payload_sha256", None)
    computed_dataset_payload = hashlib.sha256(
        json.dumps(
            dataset_without_hash,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _require(
        dataset_identity.get("file_sha256") == _file_sha256(dataset_path)
        and dataset_identity.get("payload_sha256") == dataset_payload
        == computed_dataset_payload
        and dataset_identity.get("sample_count") == 1131
        and dataset.get("summary", {}).get("state_count") == 13
        and dataset.get("summary", {}).get("candidate_count_per_state") == 87
        and len(dataset.get("records", [])) == 1131,
        "execution-margin dataset identity or dimensions differ",
    )
    model_identity = result.get("model_artifact", {})
    model_path = _safe_artifact(result_path, model_identity.get("path", ""))
    _require(
        model_identity.get("file_sha256") == _file_sha256(model_path)
        and int(model_identity.get("size_bytes", -1)) == model_path.stat().st_size,
        "execution-margin model artifact differs",
    )
    training_identity = result.get("training_artifact", {})
    training_path = _safe_artifact(result_path, training_identity.get("path", ""))
    training_record = _load(training_path)
    _require(
        training_identity.get("file_sha256") == _file_sha256(training_path)
        and training_identity.get("payload_sha256")
        == training_record.get("training_payload_sha256")
        == _hash_without(training_record, "training_payload_sha256")
        and training_record.get("model_artifact") == model_identity
        and training_record.get("dataset") == dataset_identity,
        "execution-margin training artifact differs",
    )
    training = result.get("training", {})
    metrics = training.get("split_metrics", {})
    _require(
        training.get("device") == "cuda"
        and "H100" in str(training.get("cuda_device_name"))
        and training.get("parameter_count") == 21767
        and metrics.get("train", {}).get("sample_count") == 696
        and metrics.get("validation", {}).get("sample_count") == 174
        and metrics.get("test", {}).get("sample_count") == 261,
        "execution-margin training dimensions differ",
    )
    expected_model_gate = bool(
        metrics["test"]["conservative_false_safe_candidate_count"] == 0
        and metrics["test"]["rmse_m"]
        < metrics["test"]["current_clearance_baseline_rmse_m"]
    )
    _require(
        training.get("model_learnability_gate_pass") is expected_model_gate,
        "execution-margin model gate is inconsistent",
    )
    projection = result.get("primary_projection", {})
    exact = projection.get("projected_exact_transition")
    if projection.get("neural_projection", {}).get("valid"):
        _require(
            isinstance(exact, dict)
            and exact.get("captured_state_count")
            == exact.get("expected_internal_mujoco_step_count") + 1,
            "valid neural projection lacks exact substep verification",
        )
    expected_projection_gate = bool(
        projection.get("activated")
        and projection.get("neural_projection", {}).get("valid")
        and projection.get("projected_exact_raw_safe")
        and projection.get("projected_exact_proxy_safe")
    )
    _require(
        projection.get("projection_gate_pass") is expected_projection_gate,
        "execution-margin projection gate is inconsistent",
    )
    representation_gate = bool(
        result.get("dataset_summary", {}).get(
            "primary_local_jointly_safe_candidate_exists"
        )
    )
    decision = result.get("decision", {})
    expected_go = bool(expected_model_gate and representation_gate and expected_projection_gate)
    _require(
        decision.get("model_learnability_gate_pass") is expected_model_gate
        and decision.get("representation_gate_pass") is representation_gate
        and decision.get("projection_gate_pass") is expected_projection_gate
        and decision.get("research_direction_go") is expected_go
        and result.get("research_direction_go") is expected_go
        and result.get("stop_reason") == decision.get("stop_reason")
        and ((decision.get("stop_reason") is None) is expected_go),
        "execution-margin ordered decision differs",
    )
    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": "valid",
        "scientific_result": False,
        "case_id": CASE_ID,
        "source_commit": expected_commit,
        "slurm_job_id": allocation["slurm_job_id"],
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": recorded_payload,
        "dataset_file_sha256": dataset_identity["file_sha256"],
        "model_file_sha256": model_identity["file_sha256"],
        "research_direction_go": expected_go,
        "stop_reason": decision.get("stop_reason"),
        "checks": {
            "clean_H100_source": True,
            "immutable_D_opt_D_sim_dataset": True,
            "state_grouped_training_and_calibration": True,
            "ordered_model_representation_projection_gate": True,
            "exact_projected_substep_verification_when_valid": True,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = validate(
        args.result.resolve(),
        expected_commit=args.expected_commit,
        expected_config=load_execution_margin_config(args.config.resolve()),
    )
    _atomic_write(args.output.resolve(), record)
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
