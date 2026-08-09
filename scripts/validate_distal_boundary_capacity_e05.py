#!/usr/bin/env python3
"""Validate the paired action-188 boundary-capacity result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.boundary_capacity import (
    BOUNDARY_RESULT_SCHEMA,
    load_boundary_capacity_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    CASE_ID,
    _atomic_write,
    _file_sha256,
    _load,
    _require,
)
from scripts.train_distal_boundary_capacity_e05 import TRAINING_SCHEMA


VALIDATION_SCHEMA = "vlsa_distal_boundary_capacity_e05_validation.v1"


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


def validate(
    result_path: Path,
    *,
    expected_commit: str,
    expected_config: Mapping[str, Any],
) -> dict[str, Any]:
    result = _load(result_path)
    _require(
        result.get("schema_version") == BOUNDARY_RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and result.get("case_id") == CASE_ID
        and result.get("failure") is None
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256")
        and result.get("config") == expected_config,
        "boundary-capacity result identity differs",
    )
    source = result.get("source", {})
    allocation = result.get("allocation", {})
    _require(
        source.get("commit") == expected_commit
        and source.get("dirty") is False
        and str(allocation.get("slurm_job_id", "")).isdigit()
        and "H100" in str(allocation.get("device", {}).get("name", "")),
        "boundary-capacity source or H100 allocation differs",
    )
    dataset_gate = result.get("dataset_gate", {})
    dataset_identity = dataset_gate.get("dataset", {})
    dataset_path = Path(dataset_identity.get("path", "")).resolve()
    dataset_result_path = Path(dataset_gate.get("result_path", "")).resolve()
    dataset_validation_path = Path(
        dataset_gate.get("validation_path", "")
    ).resolve()
    _require(
        dataset_path.is_file()
        and dataset_result_path.is_file()
        and dataset_validation_path.is_file()
        and dataset_identity.get("file_sha256") == _file_sha256(dataset_path)
        and dataset_gate.get("result_file_sha256")
        == _file_sha256(dataset_result_path)
        and dataset_gate.get("validation_file_sha256")
        == _file_sha256(dataset_validation_path)
        and dataset_gate.get("neural_training_authorized") is True,
        "boundary-capacity external dataset gate differs",
    )
    training_identity = result.get("training_artifact", {})
    training_path = Path(training_identity.get("path", "")).resolve()
    _require(
        training_path.is_file()
        and not training_path.is_symlink()
        and training_path.parent == result_path.parent.resolve()
        and training_identity.get("file_sha256") == _file_sha256(training_path),
        "boundary-capacity training artifact identity differs",
    )
    training = _load(training_path)
    _require(
        training.get("schema_version") == TRAINING_SCHEMA
        and training.get("training_payload_sha256")
        == _hash_without(training, "training_payload_sha256")
        == training_identity.get("payload_sha256"),
        "boundary-capacity training payload differs",
    )
    arm_passes = {}
    for arm in expected_config["training"]["arms"]:
        value = result.get("arms", {}).get(arm, {})
        trained = training.get("arms", {}).get(arm, {})
        _require(
            value.get("model_artifact") == trained.get("model_artifact")
            and value.get("training") == trained.get("training"),
            "boundary-capacity paired arm binding differs: %s" % arm,
        )
        model_identity = value.get("model_artifact", {})
        model_path = Path(model_identity.get("path", "")).resolve()
        _require(
            model_path.is_file()
            and not model_path.is_symlink()
            and model_path.parent == result_path.parent.resolve()
            and model_identity.get("file_sha256") == _file_sha256(model_path),
            "boundary-capacity model artifact differs: %s" % arm,
        )
        audit = value.get("training", {})
        test = audit.get("split_metrics", {}).get("test", {})
        expected_model_gate = bool(
            test.get("critical_boundary_rmse_m", float("inf"))
            < test.get(
                "critical_boundary_current_clearance_baseline_rmse_m", 0.0
            )
            and test.get("conservative_false_safe_candidate_count") == 0
            and test.get("critical_gradient_cosine_count", 0) > 0
            and test.get("critical_gradient_cosine_mean", -1.0)
            >= expected_config["decision_gate"][
                "critical_gradient_cosine_similarity_minimum"
            ]
        )
        projection = value.get("primary_projection", {})
        exact = projection.get("projected_exact_transition")
        if projection.get("neural_projection", {}).get("valid"):
            _require(
                isinstance(exact, dict)
                and exact.get("captured_state_count")
                == exact.get("expected_internal_mujoco_step_count") + 1,
                "valid boundary projection lacks exact substep verification",
            )
        expected_projection_gate = bool(
            projection.get("activated")
            and projection.get("neural_projection", {}).get("valid")
            and projection.get("projected_exact_raw_safe")
            and projection.get("projected_exact_proxy_safe")
        )
        decision = value.get("decision", {})
        expected_arm_pass = bool(expected_model_gate and expected_projection_gate)
        _require(
            audit.get("device") == "cuda"
            and "H100" in str(audit.get("cuda_device_name"))
            and audit.get("parameter_count") == 21767
            and audit.get("model_capacity_gate_pass") is expected_model_gate
            and decision.get("model_capacity_gate_pass") is expected_model_gate
            and decision.get("projection_gate_pass") is expected_projection_gate
            and decision.get("arm_capacity_gate_pass") is expected_arm_pass,
            "boundary-capacity arm decision differs: %s" % arm,
        )
        arm_passes[arm] = expected_arm_pass
    expected_go = arm_passes["boundary_margin_gradient"]
    decision = result.get("decision", {})
    _require(
        decision.get("research_direction_go") is expected_go
        and result.get("research_direction_go") is expected_go
        and result.get("stop_reason") == decision.get("stop_reason")
        and ((decision.get("stop_reason") is None) is expected_go),
        "boundary-capacity research decision differs",
    )
    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": "valid",
        "scientific_result": False,
        "case_id": CASE_ID,
        "source_commit": expected_commit,
        "slurm_job_id": allocation["slurm_job_id"],
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "research_direction_go": expected_go,
        "stop_reason": decision.get("stop_reason"),
        "arm_capacity_gate_pass": arm_passes,
        "checks": {
            "clean_H100_source": True,
            "validated_pretraining_dataset_gate": True,
            "paired_fixed_architecture_arms": True,
            "critical_boundary_and_gradient_gates": True,
            "valid_seven_row_QP_and_exact_substep_verification": True,
            "same_state_claim_scope_only": True,
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
        expected_config=load_boundary_capacity_config(args.config.resolve()),
    )
    _atomic_write(args.output.resolve(), record)
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
