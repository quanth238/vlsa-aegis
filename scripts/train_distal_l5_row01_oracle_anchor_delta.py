#!/usr/bin/env python3
"""Train the frozen-rollout exact-anchor L5 action-response diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def load_anchored_samples(config: Mapping[str, Any], repo_root: Path) -> tuple[Any, Any]:
    import numpy as np

    from main.multilink_ellipsoid.l5_row01_component_audit import (
        load_config as load_component_config,
        payload_sha256 as component_payload_sha256,
    )
    from scripts.train_distal_l5_row01_component_audit import load_component_samples

    source = config["source"]
    component_path = repo_root / source["component_audit_config"]
    _require(_file_sha256(component_path) == source["component_audit_config_file_sha256"],
             "oracle-anchor component config differs")
    component_config = load_component_config(component_path)
    samples, _ = load_component_samples(component_config, repo_root)
    frozen_path = Path(source["frozen_component_result"])
    _require(_file_sha256(frozen_path) == source["frozen_component_result_file_sha256"],
             "oracle-anchor frozen component file differs")
    frozen = _load(frozen_path)
    _require(
        frozen["result_payload_sha256"]
        == source["frozen_component_result_payload_sha256"]
        and frozen["result_payload_sha256"] == component_payload_sha256(frozen),
        "oracle-anchor frozen component payload differs",
    )
    output = {"train": [], "validation": []}
    excluded = {"train": [], "validation": []}
    minimum = int(config["dataset"]["minimum_non_nominal_responses_per_state"])
    for split in ("train", "validation"):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in samples[split]:
            grouped.setdefault(str(item["state_id"]), []).append(item)
        for state_id, items in sorted(grouped.items()):
            anchors = [item for item in items if (
                item["candidate_name"] == "nominal"
                and int(item["candidate_order"]) == 0
                and float(item["applied_correction_l2_action"]) <= 1.0e-12
            )]
            responses = [item for item in items if float(
                item["applied_correction_l2_action"]
            ) > 1.0e-12]
            if len(anchors) != 1 or len(responses) < minimum:
                excluded[split].append({
                    "state_id": state_id, "anchor_count": len(anchors),
                    "response_count": len(responses),
                })
                continue
            anchor = np.asarray(anchors[0]["combined_target"], dtype=np.float64)
            for item in items:
                feature = np.asarray(item["relative_feature_vector"], dtype=np.float64)
                _require(feature.shape == (134,), "oracle-anchor relative feature differs")
                context = np.concatenate((feature[:55], feature[90:]))
                residual = feature[55:90] - feature[20:55]
                target = np.asarray(item["combined_target"], dtype=np.float64)
                augmented = dict(item)
                augmented["context_vector"] = context.tolist()
                augmented["residual_vector"] = residual.tolist()
                augmented["anchor_target"] = anchor.tolist()
                augmented["delta_target"] = (target - anchor).tolist()
                output[split].append(augmented)
    observed_states = {
        split: len({item["state_id"] for item in output[split]})
        for split in ("train", "validation")
    }
    observed_candidates = {
        split: len(output[split]) for split in ("train", "validation")
    }
    observed_responses = {
        split: sum(float(item["applied_correction_l2_action"]) > 1.0e-12
                   for item in output[split])
        for split in ("train", "validation")
    }
    _require(observed_states == config["dataset"]["expected_state_counts"],
             "oracle-anchor state counts differ")
    _require(observed_candidates == config["dataset"]["expected_candidate_counts_including_anchor"],
             "oracle-anchor candidate counts differ")
    _require(observed_responses == config["dataset"]["expected_response_counts"],
             "oracle-anchor response counts differ")
    return {**output, "excluded": excluded}, frozen


def arrays(samples: Sequence[Mapping[str, Any]]) -> tuple[Any, Any, Any, Any]:
    import numpy as np
    return (
        np.asarray([item["context_vector"] for item in samples], dtype=np.float64),
        np.asarray([item["residual_vector"] for item in samples], dtype=np.float64),
        np.asarray([item["delta_target"] for item in samples], dtype=np.float64),
        np.asarray([item["anchor_target"] for item in samples], dtype=np.float64),
    )


def _model_record(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in bundle.items() if key not in {
        "model", "device", "context_mean", "context_scale", "residual_scale",
        "target_scale",
    }}


def run(repo_root: Path, config_path: Path, expected_commit: str) -> dict[str, Any]:
    from main.multilink_ellipsoid.l5_row01_oracle_anchor_delta import (
        RESULT_SCHEMA, load_config, payload_sha256, predict, response_metrics,
        train_model,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    source = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    samples, frozen = load_anchored_samples(config, repo_root)
    train_arrays = arrays(samples["train"])
    validation_arrays = arrays(samples["validation"])
    bundle = train_model(*train_arrays, *validation_arrays, config["model"])
    predictions = {}
    reports = {}
    for split in ("train", "validation"):
        context, residual, _, _ = arrays(samples[split])
        values = predict(bundle, context, residual)
        predictions[split] = values.tolist()
        reports[split] = response_metrics(
            values, samples[split],
            sign_deadband_m=float(config["evaluation"]["improvement_sign_deadband_m"]),
            random_seed=int(config["evaluation"]["random_seed"]),
            random_draws=int(config["evaluation"]["random_draws_per_state"]),
        )
    validation = reports["validation"]
    gate_config = config["gates"]
    gates = {
        "validation_delta_RMSE_below_zero_change_baseline":
        validation["delta_rmse_m"] < validation["zero_change_delta_rmse_m"],
        "validation_improvement_sign_accuracy_minimum":
        validation["improvement_sign_accuracy"]
        >= float(gate_config["validation_improvement_sign_accuracy_minimum"]),
        "validation_safe_unsafe_pair_order_accuracy_minimum":
        validation["safe_unsafe_pair_order_accuracy"]
        >= float(gate_config["validation_safe_unsafe_pair_order_accuracy_minimum"]),
        "predicted_safe_support_every_recoverable_validation_state":
        validation["supported_recoverable_state_count"]
        == validation["recoverable_state_count"],
        "selected_exact_row01_safe_every_recoverable_validation_state":
        validation["selected_exact_row01_safe_state_count"]
        == validation["recoverable_state_count"],
        "selected_safe_rate_above_seeded_random":
        validation["selected_exact_row01_safe_rate"]
        > validation["seeded_random_row01_safe_rate"],
        "task_preserving_win_or_tie_vs_strongest_registered_path_every_recoverable_state":
        validation["task_preserving_win_or_tie_vs_strongest_state_count"]
        == validation["recoverable_state_count"],
        "architectural_zero_response_maximum_error_m":
        bundle["architectural_zero_response_maximum_error_m"]
        <= float(gate_config["architectural_zero_response_maximum_error_m"]),
        "reserved_test_episodes_unopened": True,
    }
    passed = all(gates.values())
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "source": source,
        "allocation": allocation_record(), "claim_scope": config["claim_scope"],
        "config": config,
        "dataset": {
            "state_counts": {
                split: len({item["state_id"] for item in samples[split]})
                for split in ("train", "validation")
            },
            "candidate_counts_including_anchor": {
                split: len(samples[split]) for split in ("train", "validation")
            },
            "response_counts": {
                split: sum(float(item["applied_correction_l2_action"]) > 1.0e-12
                           for item in samples[split])
                for split in ("train", "validation")
            },
            "excluded_states": samples["excluded"],
            "labels_and_episode_splits_unchanged": True,
            "exact_nominal_anchor_supplied_at_evaluation": True,
            "independent_direction_labels_available": False,
            "reserved_test_episodes_unopened": True,
        },
        "frozen_component_result_payload_sha256": frozen["result_payload_sha256"],
        "model": _model_record(bundle), "predictions": predictions,
        "reports": reports, "gates": gates,
        "response_gate_pass": passed,
        "small_active_boundary_pilot_authorized": passed,
        "nominal_risk_model_authorized": False,
        "QP_authorized": False, "closed_loop_authorized": False,
        "interpretation": (
            "exact_anchor_delta_registered_path_response_pass"
            if passed else "exact_anchor_delta_registered_path_response_no_go"
        ),
    }
    result["result_payload_sha256"] = payload_sha256(result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.repo_root.resolve(), args.config.resolve(), args.expected_commit)
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "dataset": result["dataset"],
        "train": result["reports"]["train"],
        "validation": result["reports"]["validation"],
        "gates": result["gates"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
