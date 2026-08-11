#!/usr/bin/env python3
"""Audit frozen explicit-J ensemble members without training or simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Tuple

from main.multilink_ellipsoid.factorized_execution_pilot import payload_sha256
from main.multilink_ellipsoid.factorized_explicit_execution_jacobian import (
    load_explicit_jacobian_config, load_explicit_jacobian_weights,
    nominal_row_mapping,
)
from main.multilink_ellipsoid.factorized_explicit_jacobian_audit import (
    audit_decision, cancellation_audit, load_explicit_jacobian_audit_config,
    member_predictions, member_sensitivity_audit,
    selected_member_sensitivities, trajectory_audit,
)
from scripts.evaluate_distal_direct_horizon_displacement_moka10 import (
    _hash_array, add_direct_horizon_arguments, direct_paths,
    validate_direct_sources,
)
from scripts.evaluate_distal_direct_horizon_normalized_secant_moka10 import (
    validate_matched_sources,
)
from scripts.evaluate_distal_explicit_execution_jacobian_moka10 import (
    validate_normalized_source,
)
from scripts.evaluate_distal_factorized_cumulative_delta_q_moka10 import prepare
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


RESULT_SCHEMA = "vlsa_distal_factorized_explicit_jacobian_audit_result.v1"
VALIDATION_SCHEMA = (
    "vlsa_distal_factorized_explicit_jacobian_audit_validation.v1"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_direct_horizon_arguments(parser)
    parser.add_argument("--matched-direct-run", type=Path, required=True)
    parser.add_argument("--matched-direct-validation", type=Path, required=True)
    parser.add_argument("--root-cause-run", type=Path, required=True)
    parser.add_argument("--matched-normalized-config", type=Path, required=True)
    parser.add_argument("--matched-normalized-run", type=Path, required=True)
    parser.add_argument(
        "--matched-normalized-validation", type=Path, required=True,
    )
    parser.add_argument("--audit-config", type=Path, required=True)
    parser.add_argument("--explicit-run", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def audit_paths(args: argparse.Namespace) -> Dict[str, Path]:
    return {
        "matched_direct_run": args.matched_direct_run.resolve(),
        "matched_direct_validation": args.matched_direct_validation.resolve(),
        "root_cause_run": args.root_cause_run.resolve(),
        "matched_normalized_config": args.matched_normalized_config.resolve(),
        "matched_normalized_run": args.matched_normalized_run.resolve(),
        "matched_normalized_validation": (
            args.matched_normalized_validation.resolve()
        ),
        "audit_config": args.audit_config.resolve(),
        "explicit_run": args.explicit_run.resolve(),
        "experimental_model": (args.explicit_run / "model.npz").resolve(),
        "predictions": (args.explicit_run / "predictions.npz").resolve(),
        "records": args.records.resolve(),
        "output": args.output.resolve(),
    }


def validate_explicit_source(
    paths: Mapping[str, Path], audit_config: Mapping[str, Any],
) -> Dict[str, Any]:
    source = audit_config["immutable_source"]
    run = paths["explicit_run"]
    _require(run.is_dir() and not run.is_symlink(),
             "explicit-J audit source directory differs")
    for path, key in (
        (paths["direct_config"], "explicit_config_file_sha256"),
        (run / "model.npz", "explicit_model_file_sha256"),
        (run / "predictions.npz", "explicit_predictions_file_sha256"),
        (run / "result.json", "explicit_result_file_sha256"),
        (run / "validation.json", "explicit_validation_file_sha256"),
    ):
        _require(_file_sha256(path) == source[key],
                 "explicit-J audit source hash differs")
    result = _load(run / "result.json")
    validation = _load(run / "validation.json")
    _require(
        result.get("result_payload_sha256")
        == source["explicit_result_payload_sha256"]
        == payload_sha256(result, "result_payload_sha256")
        and validation.get("validation_payload_sha256")
        == source["explicit_validation_payload_sha256"]
        == payload_sha256(validation, "validation_payload_sha256")
        and validation.get("valid") is True
        and result.get("decision", {}).get(
            "explicit_execution_jacobian_prediction_gate_pass"
        ) is False,
        "explicit-J audit source receipt differs",
    )
    return result


def perform_audit(
    *, models: Any, state: Mapping[str, Any], arrays: Mapping[str, Any],
    sensitivities: Mapping[str, Any], config: Mapping[str, Any],
    source_result: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    np = __import__("numpy")
    q, nominal, jacobian = member_predictions(models, state, arrays)
    mapping = nominal_row_mapping(arrays)
    selected = selected_member_sensitivities(
        arrays=arrays, sensitivities=sensitivities,
        state_indexes=mapping["state_index"], member_jacobian=jacobian,
    )
    sensitivity = member_sensitivity_audit(
        arrays=arrays, sensitivities=sensitivities,
        selected=selected, config=config,
    )
    trajectory = trajectory_audit(arrays, q)
    cancellation = cancellation_audit(selected, sensitivities, arrays)
    decision = audit_decision(
        sensitivity=sensitivity, cancellation=cancellation,
        training_audits=source_result["training"]["member_audits"],
        config=config,
    )
    records = {
        "member_joint_sensitivity_rad_per_action": selected,
        "exact_joint_sensitivity_rad_per_action": np.asarray(
            sensitivities["joint_sensitivity_rad_per_action"],
            dtype=np.float64,
        ),
        "state_index": np.asarray(
            sensitivities["state_index"], dtype=np.int64,
        ),
        "dimension_index": np.asarray(
            sensitivities["dimension_index"], dtype=np.int64,
        ),
    }
    audit = {
        "sensitivity": sensitivity,
        "trajectory": trajectory,
        "cancellation": cancellation,
        "decision": decision,
        "member_joint_prediction_sha256": [
            _hash_array(q[index]) for index in range(len(q))
        ],
        "member_nominal_displacement_sha256": [
            _hash_array(nominal[index]) for index in range(len(nominal))
        ],
        "member_execution_J_sha256": [
            _hash_array(jacobian[index]) for index in range(len(jacobian))
        ],
        "selected_member_sensitivity_sha256": _hash_array(selected),
    }
    return audit, records


def main() -> int:
    import numpy as np
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    (
        paths, _, _, _, _, _, arrays, sensitivities, _, representation, _,
    ) = prepare(args)
    paths.update(direct_paths(args))
    paths.update(audit_paths(args))
    explicit_config = load_explicit_jacobian_config(paths["direct_config"])
    audit_config = load_explicit_jacobian_audit_config(paths["audit_config"])
    validate_direct_sources(paths, explicit_config)
    validate_matched_sources(paths, explicit_config)
    validate_normalized_source(paths, explicit_config)
    source_result = validate_explicit_source(paths, audit_config)
    _require(
        int(representation["structured_input_dimension"])
        == int(arrays["features"].shape[1])
        and len(arrays["features"])
        == audit_config["population"]["existing_rollout_count"],
        "explicit-J audit population differs",
    )
    models, state = load_explicit_jacobian_weights(
        paths["experimental_model"]
    )
    audit, records = perform_audit(
        models=models, state=state, arrays=arrays,
        sensitivities=sensitivities, config=audit_config,
        source_result=source_result,
    )
    paths["records"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths["records"], **records)
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True,
        "claim_scope": audit_config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": audit_config,
        "explicit_source": {
            "result_file_sha256": _file_sha256(
                paths["explicit_run"] / "result.json"
            ),
            "result_payload_sha256": source_result["result_payload_sha256"],
            "member_audits": source_result["training"]["member_audits"],
        },
        "representation": representation,
        "audit": audit,
        "records": {
            "path": str(paths["records"]),
            "file_sha256": _file_sha256(paths["records"]),
            "selected_member_sensitivity_sha256": audit[
                "selected_member_sensitivity_sha256"
            ],
        },
        "forbidden_action_receipt": {
            key + "_executed": False for key in audit_config["forbidden_actions"]
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "decision": audit["decision"],
        "cancellation": audit["cancellation"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
