#!/usr/bin/env python3
"""Audit the frozen direct-horizon model without training or new rollouts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.factorized_direct_horizon_displacement import (
    apply_structured_representation, load_direct_horizon_config,
    load_direct_horizon_weights, predict_direct_horizon,
)
from main.multilink_ellipsoid.factorized_direct_horizon_root_cause_audit import (
    RESULT_SCHEMA, _population_metrics, load_audit_config, member_predictions,
    root_cause_decision, sensitivity_audit, state_input_distances,
    strip_private_arrays,
)
from main.multilink_ellipsoid.factorized_execution_pilot import (
    payload_sha256, sensitivity_arrays,
)
from main.multilink_ellipsoid.factorized_structured_orientation import (
    load_structured_config,
)
from scripts.evaluate_distal_direct_horizon_displacement_moka10 import (
    _hash_array, add_direct_horizon_arguments, direct_paths,
    evaluate_reserved_geometry, load_reserved, validate_direct_sources,
)
from scripts.evaluate_distal_factorized_cumulative_delta_q_moka10 import prepare
from scripts.evaluate_distal_factorized_one_sided_geometry_moka10 import (
    geometry_kwargs,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _subset(arrays: Mapping[str, Any], selected: Any) -> dict[str, Any]:
    import numpy as np

    mask = np.asarray(selected, dtype=bool)
    output = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        output[key] = array[mask] if array.ndim and len(array) == len(mask) else value
    return output


def _save_records(path: Path, records: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name, dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **records)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return {"path": str(path), "file_sha256": _file_sha256(path)}


def _state_episode(records: Any) -> dict[int, str]:
    return {
        int(item["state_index"]): str(item["case_id"])
        for item in records
    }


def validate_immutable_audit_sources(
    *, paths: Mapping[str, Path], audit_config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    source = audit_config["immutable_source"]
    for source_key, path_key in (
        ("direct_config_file_sha256", "direct_config"),
        ("model_file_sha256", "frozen_model"),
        ("predictions_file_sha256", "frozen_predictions"),
        ("result_file_sha256", "frozen_result"),
        ("validation_file_sha256", "frozen_validation"),
        ("source_metadata_file_sha256", "metadata"),
        ("source_array_file_sha256", "array"),
        ("reserved_metadata_file_sha256", "reserved_metadata"),
        ("reserved_array_file_sha256", "reserved_array"),
        ("reserved_collection_validation_file_sha256", "reserved_validation"),
    ):
        _require(
            _file_sha256(paths[path_key]) == source[source_key],
            "direct-horizon root-cause immutable source differs",
        )
    result = _load(paths["frozen_result"])
    validation = _load(paths["frozen_validation"])
    _require(
        result.get("result_payload_sha256") == source["result_payload_sha256"]
        and result.get("decision", {}).get("prediction_gate_GO") is False
        and validation.get("validation_payload_sha256")
        == source["validation_payload_sha256"]
        and validation.get("valid") is True,
        "direct-horizon root-cause validated NO-GO differs",
    )
    return result, validation


def analyze(
    *, audit_config: Mapping[str, Any], training_arrays: Mapping[str, Any],
    reserved_arrays: Mapping[str, Any], source_q: Any, reserved_q: Any,
    source_member_q: Any, reserved_member_q: Any, source_predicted_h: Any,
    source_exact_static_h: Any, reserved_predicted_h: Any,
    reserved_exact_static_h: Any, source_state_episode: Mapping[int, str],
    reserved_state_episode: Mapping[int, str], source_input_distance: Mapping[int, Any],
    reserved_input_distance: Mapping[int, Any],
) -> dict[str, Any]:
    import numpy as np

    populations = {}
    source_split = np.asarray(training_arrays["split"], dtype=object)
    for split_name, public_name in (
        ("train", "train"), ("validation", "validation"),
        ("test", "diagnostic"),
    ):
        selected = source_split == split_name
        subarrays = _subset(training_arrays, selected)
        states = set(np.asarray(subarrays["state_index"], dtype=np.int64).tolist())
        populations[public_name] = _population_metrics(
            name=public_name, arrays=subarrays,
            predicted_q=np.asarray(source_q)[selected],
            member_q=np.asarray(source_member_q)[:, selected],
            predicted_h=np.asarray(source_predicted_h)[selected],
            exact_static_h=np.asarray(source_exact_static_h)[selected],
            state_to_episode={key: source_state_episode[key] for key in states},
            input_distance={key: source_input_distance[key] for key in states},
            config=audit_config,
        )
    populations["reserved"] = _population_metrics(
        name="reserved", arrays=reserved_arrays, predicted_q=reserved_q,
        member_q=reserved_member_q, predicted_h=reserved_predicted_h,
        exact_static_h=reserved_exact_static_h,
        state_to_episode=reserved_state_episode,
        input_distance=reserved_input_distance, config=audit_config,
    )
    source_state_split = {
        int(state): str(split)
        for state, split in zip(
            np.asarray(training_arrays["state_index"], dtype=np.int64),
            source_split,
        )
    }
    reserved_state_split = {
        int(state): "reserved"
        for state in np.asarray(reserved_arrays["state_index"], dtype=np.int64)
    }
    sensitivities = {
        "source": sensitivity_audit(
            arrays=training_arrays, predicted_q=source_q,
            predicted_h=source_predicted_h,
            sensitivities=sensitivity_arrays(training_arrays),
            state_to_split=source_state_split,
        ),
        "reserved": sensitivity_audit(
            arrays=reserved_arrays, predicted_q=reserved_q,
            predicted_h=reserved_predicted_h,
            sensitivities=sensitivity_arrays(reserved_arrays),
            state_to_split=reserved_state_split,
        ),
    }
    decision = root_cause_decision(
        populations=populations, sensitivities=sensitivities,
        config=audit_config,
    )
    return {
        "populations": {
            key: strip_private_arrays(value) for key, value in populations.items()
        },
        "sensitivity": sensitivities,
        "decision": decision,
    }


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    (
        paths, _, _, factorized_config, complete_dataset,
        complete_collection, training_arrays, _, normalization,
        representation, _,
    ) = prepare(args)
    paths.update(direct_paths(args))
    paths.update({
        "audit_config": args.audit_config.resolve(),
        "frozen_model": args.frozen_model.resolve(),
        "frozen_predictions": args.frozen_predictions.resolve(),
        "frozen_result": args.frozen_result.resolve(),
        "frozen_validation": args.frozen_validation.resolve(),
        "records": args.records.resolve(), "output": args.output.resolve(),
    })
    audit_config = load_audit_config(paths["audit_config"])
    direct_config = load_direct_horizon_config(paths["direct_config"])
    structured_config = load_structured_config(paths["structured_config"])
    validate_direct_sources(paths, direct_config)
    frozen_result, frozen_validation = validate_immutable_audit_sources(
        paths=paths, audit_config=audit_config,
    )
    _, reserved_metadata, reserved_raw = load_reserved(paths, direct_config)
    reserved_arrays, reserved_representation = apply_structured_representation(
        complete_dataset, reserved_raw, structured_config,
    )
    _require(
        reserved_representation["removed_indexes_sha256"]
        == representation["removed_feature_indexes_sha256"]
        and reserved_representation["retained_indexes_sha256"]
        == representation["retained_feature_indexes_sha256"],
        "direct-horizon root-cause representation differs",
    )
    models, model_state = load_direct_horizon_weights(paths["frozen_model"])
    source_q = predict_direct_horizon(models, model_state, training_arrays)
    reserved_q = predict_direct_horizon(models, model_state, reserved_arrays)
    stored = np.load(paths["frozen_predictions"], allow_pickle=False)
    _require(
        np.array_equal(
            reserved_q, stored["direct_horizon_joint_position_rad"]
        ) and _hash_array(reserved_q)
        == frozen_result["predictions"]["direct_horizon_joint_sha256"],
        "direct-horizon root-cause frozen prediction differs",
    )
    source_member_q = member_predictions(models, model_state, training_arrays)
    reserved_member_q = member_predictions(models, model_state, reserved_arrays)
    source_geometry = geometry_kwargs(
        paths, factorized_config, complete_dataset, complete_collection,
        training_arrays, source_q, ("train", "validation", "test"),
        return_trace=True,
    )
    source_predicted_h = source_geometry.pop("predicted_clearance_trace_m")
    source_exact_static_h = source_geometry.pop(
        "exact_q_static_clearance_trace_m"
    )
    source_geometry.pop("predicted_minimum_margin_m")
    source_geometry.pop("exact_q_static_minimum_margin_m")
    reserved_geometry = evaluate_reserved_geometry(
        paths=paths, factorized_config=factorized_config,
        complete_collection=complete_collection, metadata=reserved_metadata,
        arrays=reserved_raw,
        predictions={"direct_horizon_displacement": reserved_q},
        return_trace=True,
    )
    reserved_traces = reserved_geometry.pop("clearance_trace_m")
    reserved_predicted_h = reserved_traces["direct_horizon_displacement"]
    reserved_exact_static_h = reserved_traces["exact_q_static"]
    reserved_geometry.pop("minimum_margin_m")
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
    audit = analyze(
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
    records = _save_records(paths["records"], {
        "source_predicted_joint_position_rad": source_q,
        "source_member_joint_position_rad": source_member_q,
        "source_predicted_clearance_m": source_predicted_h,
        "source_exact_q_static_clearance_m": source_exact_static_h,
        "reserved_predicted_joint_position_rad": reserved_q,
        "reserved_member_joint_position_rad": reserved_member_q,
        "reserved_predicted_clearance_m": reserved_predicted_h,
        "reserved_exact_q_static_clearance_m": reserved_exact_static_h,
    })
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": audit_config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": audit_config,
        "independently_validated_frozen_NO_GO": {
            "result_file_sha256": _file_sha256(paths["frozen_result"]),
            "validation_file_sha256": _file_sha256(paths["frozen_validation"]),
            "validation_valid": bool(frozen_validation["valid"]),
            "decision": frozen_result["decision"],
        },
        "geometry_receipt": {
            "source": source_geometry, "reserved": reserved_geometry,
        },
        "records": records, "audit": audit,
        "forbidden_action_receipt": {
            key + "_executed": False for key in audit_config["forbidden_actions"]
        },
        "future_task3_manifest_opened_or_evaluated": False,
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "decision": audit["decision"],
        "train": audit["populations"]["train"]["safety"],
        "validation": audit["populations"]["validation"]["safety"],
        "reserved": audit["populations"]["reserved"]["safety"],
        "unsupported": audit["populations"]["reserved"][
            "unsupported_recoverable_states"
        ],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
