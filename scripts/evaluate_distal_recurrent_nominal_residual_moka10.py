#!/usr/bin/env python3
"""Run the fitted recurrent nominal-plus-residual execution-model gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.factorized_execution_pilot import payload_sha256
from main.multilink_ellipsoid.factorized_recurrent_nominal_residual import (
    fitted_prediction_gate, load_recurrent_config, predict_recurrent,
    save_recurrent_weights, train_recurrent_ensemble,
)
from main.multilink_ellipsoid.factorized_time_conditioned_decoder import (
    eligible_state_support,
)
from scripts.evaluate_distal_direct_horizon_displacement_moka10 import (
    _hash_array, add_direct_horizon_arguments, direct_paths,
)
from scripts.evaluate_distal_direct_horizon_normalized_secant_moka10 import (
    compute_metrics,
)
from scripts.evaluate_distal_factorized_cumulative_delta_q_moka10 import prepare
from scripts.evaluate_distal_factorized_one_sided_geometry_moka10 import (
    geometry_kwargs,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


RESULT_SCHEMA = "vlsa_distal_factorized_recurrent_nominal_residual_result.v1"
VALIDATION_SCHEMA = (
    "vlsa_distal_factorized_recurrent_nominal_residual_validation.v1"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    add_direct_horizon_arguments(parser)
    parser.add_argument("--recurrent-config", type=Path, required=True)
    parser.add_argument("--normalized-direct-run", type=Path, required=True)
    parser.add_argument("--normalized-direct-validation", type=Path, required=True)
    parser.add_argument("--explicit-member-audit-run", type=Path, required=True)
    parser.add_argument("--future-untouched-manifest", type=Path, required=True)
    parser.add_argument("--experimental-model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def recurrent_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = direct_paths(args)
    paths.update({
        "recurrent_config": args.recurrent_config.resolve(),
        "normalized_direct_run": args.normalized_direct_run.resolve(),
        "normalized_direct_validation": args.normalized_direct_validation.resolve(),
        "explicit_member_audit_run": args.explicit_member_audit_run.resolve(),
        "future_untouched_manifest": args.future_untouched_manifest.resolve(),
        "experimental_model": args.experimental_model.resolve(),
        "predictions": args.predictions.resolve(),
        "output": args.output.resolve(),
    })
    return paths


def validate_recurrent_sources(
    paths: Mapping[str, Path], config: Mapping[str, Any],
) -> dict[str, Any]:
    source = config["immutable_source"]
    direct = paths["normalized_direct_run"]
    audit = paths["explicit_member_audit_run"]
    for directory in (direct, audit):
        _require(directory.is_dir() and not directory.is_symlink(),
                 "recurrent immutable source directory differs")
    checks = (
        (paths["complete_dataset"], "complete_dataset_file_sha256"),
        (paths["complete_collection"], "complete_collection_file_sha256"),
        (paths["array"], "trajectory_array_file_sha256"),
        (paths["metadata"], "trajectory_metadata_file_sha256"),
        (paths["collection"], "trajectory_collection_file_sha256"),
        (paths["direct_config"], "normalized_direct_config_file_sha256"),
        (direct / "model.npz", "normalized_direct_model_file_sha256"),
        (direct / "predictions.npz", "normalized_direct_predictions_file_sha256"),
        (direct / "result.json", "normalized_direct_result_file_sha256"),
        (paths["normalized_direct_validation"],
         "normalized_direct_validation_file_sha256"),
        (audit / "result.json", "explicit_member_audit_result_file_sha256"),
        (audit / "validation.json",
         "explicit_member_audit_validation_file_sha256"),
        (paths["future_untouched_manifest"],
         "future_untouched_manifest_file_sha256"),
    )
    for path, key in checks:
        _require(_file_sha256(path) == source[key],
                 "recurrent immutable source hash differs: %s" % key)
    direct_result = _load(direct / "result.json")
    direct_validation = _load(paths["normalized_direct_validation"])
    audit_result = _load(audit / "result.json")
    audit_validation = _load(audit / "validation.json")
    _require(
        direct_result.get("result_payload_sha256")
        == payload_sha256(direct_result, "result_payload_sha256")
        and direct_validation.get("valid") is True
        and direct_validation.get("result_payload_sha256")
        == direct_result["result_payload_sha256"]
        and audit_result.get("result_payload_sha256")
        == payload_sha256(audit_result, "result_payload_sha256")
        and audit_validation.get("valid") is True
        and audit_result.get("audit", {}).get("decision", {}).get(
            "classification"
        ) == "member_level_optimization_or_checkpoint_failure",
        "recurrent immutable source receipt differs",
    )
    return direct_result


def support_metrics(
    arrays: Mapping[str, Any], predicted_minimum: Any,
) -> dict[str, Any]:
    return {
        split: eligible_state_support(
            exact_margin=arrays["minimum_margin_m"],
            predicted_margin=predicted_minimum,
            arrays=arrays, split_name=split,
        ) for split in ("train", "validation", "test")
    }


def fresh_direct_baseline(
    *, paths: Mapping[str, Path], arrays: Mapping[str, Any],
    sensitivities: Mapping[str, Any], factorized_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    stored = np.load(
        paths["normalized_direct_run"] / "predictions.npz", allow_pickle=False,
    )
    q = stored["joint_position_rad"]
    trace = stored["predicted_clearance_trace_m"]
    metrics = compute_metrics(
        arrays=arrays, predicted_q=q, predicted_trace=trace,
        sensitivities=sensitivities, factorized_config=factorized_config,
    )
    support = support_metrics(arrays, np.min(trace, axis=1))
    return metrics, support


def main() -> int:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.shadow import allocation_record

    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    (
        paths, _, _, factorized_config, complete_dataset,
        complete_collection, arrays, sensitivities, normalization,
        representation, local_geometry,
    ) = prepare(args)
    paths.update(recurrent_paths(args))
    config = load_recurrent_config(paths["recurrent_config"])
    direct_result = validate_recurrent_sources(paths, config)
    torch.set_num_threads(8)
    direct_metrics, direct_support = fresh_direct_baseline(
        paths=paths, arrays=arrays, sensitivities=sensitivities,
        factorized_config=factorized_config,
    )
    _require(
        direct_metrics == direct_result["metrics"],
        "recurrent fresh direct-baseline metrics differ",
    )
    models, state, training = train_recurrent_ensemble(
        arrays, sensitivities, local_geometry, config, normalization,
    )
    predicted_q, member_prediction = predict_recurrent(models, state, arrays)
    geometry = geometry_kwargs(
        paths, factorized_config, complete_dataset, complete_collection,
        arrays, predicted_q, ("train", "validation", "test"),
        return_trace=True,
    )
    predicted_trace = geometry.pop("predicted_clearance_trace_m")
    exact_static_trace = geometry.pop("exact_q_static_clearance_trace_m")
    predicted_minimum = geometry.pop("predicted_minimum_margin_m")
    exact_static_minimum = geometry.pop("exact_q_static_minimum_margin_m")
    metrics = compute_metrics(
        arrays=arrays, predicted_q=predicted_q,
        predicted_trace=predicted_trace, sensitivities=sensitivities,
        factorized_config=factorized_config,
    )
    support = support_metrics(arrays, predicted_minimum)
    decision = fitted_prediction_gate(
        metrics=metrics, support=support, config=config,
    )
    model_receipt = save_recurrent_weights(paths["experimental_model"], state)
    paths["predictions"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["predictions"], joint_position_rad=predicted_q,
        predicted_clearance_trace_m=predicted_trace,
        predicted_minimum_margin_m=predicted_minimum,
        exact_q_static_clearance_trace_m=exact_static_trace,
        exact_q_static_minimum_margin_m=exact_static_minimum,
    )
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(paths["repo"], args.expected_commit),
        "allocation": allocation_record(), "config": config,
        "matched_comparison": {
            "same_existing_dataset": True,
            "same_grouped_splits": True,
            "same_ensemble_seeds": True,
            "same_normalized_paired_secant_loss": True,
            "frozen_direct_result_file_sha256": _file_sha256(
                paths["normalized_direct_run"] / "result.json"
            ),
            "direct_metrics_freshly_reproduced": True,
            "direct_metrics": direct_metrics,
            "direct_support": direct_support,
        },
        "representation": representation,
        "local_geometry_jacobian_audit": local_geometry["audit"],
        "training": training, "model": model_receipt,
        "predictions": {
            "path": str(paths["predictions"]),
            "file_sha256": _file_sha256(paths["predictions"]),
            "joint_sha256": _hash_array(predicted_q),
            "clearance_trace_sha256": _hash_array(predicted_trace),
            "member_joint_sha256": hashlib.sha256(
                np.asarray(member_prediction["member_joint_position_rad"])
                .tobytes()
            ).hexdigest(),
        },
        "geometry": geometry, "metrics": metrics, "support": support,
        "decision": decision,
        "future_untouched_episode_receipt": {
            "manifest_file_sha256": _file_sha256(
                paths["future_untouched_manifest"]
            ),
            "opened_or_evaluated": False,
            "opening_authorized": bool(
                decision["untouched_episode_evaluation_authorized"]
            ),
        },
        "forbidden_action_receipt": {
            "untouched_episode_opened": False,
            "matched_random_rollout_executed": False,
            "flow_guidance_executed": False,
            "calibration_executed": False,
            "QP_executed": False, "closed_loop_executed": False,
            "poisson_or_SDF_executed": False,
            "binary_classifier_executed": False,
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256"
    )
    _atomic_write(paths["output"], result)
    print(json.dumps({
        "decision": decision,
        "recurrent_validation_sensitivity": metrics["sensitivity"]
        ["validation"]["all_horizon_trace"],
        "recurrent_validation_safety": metrics["safety"]["validation"],
        "recurrent_validation_temporal": metrics["temporal_joint"]["validation"],
        "direct_validation_sensitivity": direct_metrics["sensitivity"]
        ["validation"]["all_horizon_trace"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
