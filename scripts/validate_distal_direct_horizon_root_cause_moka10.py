#!/usr/bin/env python3
"""Independently validate the frozen direct-horizon root-cause audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from main.multilink_ellipsoid.factorized_direct_horizon_displacement import (
    apply_structured_representation, load_direct_horizon_config,
    load_direct_horizon_weights, predict_direct_horizon,
)
from main.multilink_ellipsoid.factorized_direct_horizon_root_cause_audit import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, load_audit_config, member_predictions,
    state_input_distances,
)
from main.multilink_ellipsoid.factorized_execution_pilot import payload_sha256
from main.multilink_ellipsoid.factorized_structured_orientation import (
    load_structured_config,
)
from scripts.audit_distal_direct_horizon_root_cause_moka10 import (
    _state_episode, analyze, validate_immutable_audit_sources,
)
from scripts.evaluate_distal_direct_horizon_displacement_moka10 import (
    add_direct_horizon_arguments, direct_paths, load_reserved,
    validate_direct_sources,
)
from scripts.evaluate_distal_factorized_cumulative_delta_q_moka10 import prepare
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def main() -> int:
    import numpy as np
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_direct_horizon_arguments(parser)
    parser.add_argument("--audit-config", type=Path, required=True)
    parser.add_argument("--frozen-model", type=Path, required=True)
    parser.add_argument("--frozen-predictions", type=Path, required=True)
    parser.add_argument("--frozen-result", type=Path, required=True)
    parser.add_argument("--frozen-validation", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-result-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    (
        paths, _, _, _, complete_dataset, _, training_arrays, _, normalization,
        representation, _,
    ) = prepare(args)
    paths.update(direct_paths(args))
    paths.update({
        "audit_config": args.audit_config.resolve(),
        "frozen_model": args.frozen_model.resolve(),
        "frozen_predictions": args.frozen_predictions.resolve(),
        "frozen_result": args.frozen_result.resolve(),
        "frozen_validation": args.frozen_validation.resolve(),
        "records": args.records.resolve(), "result": args.result.resolve(),
        "output": args.output.resolve(),
    })
    audit_config = load_audit_config(paths["audit_config"])
    direct_config = load_direct_horizon_config(paths["direct_config"])
    structured_config = load_structured_config(paths["structured_config"])
    validate_direct_sources(paths, direct_config)
    validate_immutable_audit_sources(paths=paths, audit_config=audit_config)
    result = _load(paths["result"])
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("result_payload_sha256")
        == payload_sha256(result, "result_payload_sha256")
        and result.get("source", {}).get("commit") == args.expected_result_commit
        and result.get("records", {}).get("file_sha256")
        == _file_sha256(paths["records"]),
        "direct-horizon root-cause result differs",
    )
    _, reserved_metadata, reserved_raw = load_reserved(paths, direct_config)
    reserved_arrays, reserved_representation = apply_structured_representation(
        complete_dataset, reserved_raw, structured_config,
    )
    _require(
        reserved_representation["removed_indexes_sha256"]
        == representation["removed_feature_indexes_sha256"],
        "direct-horizon root-cause validation representation differs",
    )
    records = np.load(paths["records"], allow_pickle=False)
    models, model_state = load_direct_horizon_weights(paths["frozen_model"])
    source_q = predict_direct_horizon(models, model_state, training_arrays)
    reserved_q = predict_direct_horizon(models, model_state, reserved_arrays)
    source_member_q = member_predictions(models, model_state, training_arrays)
    reserved_member_q = member_predictions(models, model_state, reserved_arrays)
    q_equal = bool(
        np.array_equal(source_q, records["source_predicted_joint_position_rad"])
        and np.array_equal(reserved_q, records["reserved_predicted_joint_position_rad"])
        and np.array_equal(source_member_q, records["source_member_joint_position_rad"])
        and np.array_equal(reserved_member_q, records["reserved_member_joint_position_rad"])
    )
    stored = np.load(paths["frozen_predictions"], allow_pickle=False)
    source_predicted_h = records["source_predicted_clearance_m"]
    source_exact_static_h = records["source_exact_q_static_clearance_m"]
    reserved_predicted_h = records["reserved_predicted_clearance_m"]
    reserved_exact_static_h = records["reserved_exact_q_static_clearance_m"]
    source_validation = np.asarray(training_arrays["split"], dtype=object) == "validation"
    frozen_minimum_equal = bool(
        np.array_equal(
            np.min(reserved_predicted_h, axis=1),
            stored["direct_horizon_minimum_margin_m"],
        )
        and np.array_equal(
            np.min(reserved_exact_static_h, axis=1),
            stored["exact_q_static_minimum_margin_m"],
        )
        and np.array_equal(
            np.min(source_predicted_h[source_validation], axis=1),
            stored["validation_direct_horizon_minimum_margin_m"][source_validation],
        )
        and np.array_equal(
            np.min(source_exact_static_h[source_validation], axis=1),
            stored["validation_exact_q_static_minimum_margin_m"][source_validation],
        )
    )
    source_state_episode = _state_episode(complete_dataset["state_records"])
    reserved_state_episode = _state_episode(reserved_metadata["state_records"])
    source_input_distance = state_input_distances(
        training_arrays=training_arrays, target_arrays=training_arrays,
        feature_mean=normalization["feature_mean"],
        feature_std=normalization["feature_std"],
        retained_feature_count=representation["retained_feature_count"],
        training_state_to_episode=source_state_episode,
    )
    reserved_input_distance = state_input_distances(
        training_arrays=training_arrays, target_arrays=reserved_arrays,
        feature_mean=normalization["feature_mean"],
        feature_std=normalization["feature_std"],
        retained_feature_count=representation["retained_feature_count"],
        training_state_to_episode=source_state_episode,
    )
    recomputed = analyze(
        audit_config=audit_config, training_arrays=training_arrays,
        reserved_arrays=reserved_arrays, source_q=source_q,
        reserved_q=reserved_q, source_member_q=source_member_q,
        reserved_member_q=reserved_member_q,
        source_predicted_h=source_predicted_h,
        source_exact_static_h=source_exact_static_h,
        reserved_predicted_h=reserved_predicted_h,
        reserved_exact_static_h=reserved_exact_static_h,
        source_state_episode=source_state_episode,
        reserved_state_episode=reserved_state_episode,
        source_input_distance=source_input_distance,
        reserved_input_distance=reserved_input_distance,
    )
    audit_equal = bool(recomputed == result["audit"])
    valid = bool(q_equal and frozen_minimum_equal and audit_equal)
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "complete",
        "valid": valid, "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(),
        "result_file_sha256": _file_sha256(paths["result"]),
        "result_payload_sha256": result["result_payload_sha256"],
        "records_file_sha256": _file_sha256(paths["records"]),
        "audit": {
            "frozen_model_predictions_exactly_reproduced": q_equal,
            "clearance_trace_minima_match_independently_validated_artifact": frozen_minimum_equal,
            "all_root_cause_metrics_and_decision_reproduced": audit_equal,
            "decision": recomputed["decision"],
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    output["validation_payload_sha256"] = payload_sha256(
        output, "validation_payload_sha256"
    )
    _atomic_write(paths["output"], output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
