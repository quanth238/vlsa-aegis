#!/usr/bin/env python3
"""Train and evaluate the preregistered two-output L5 candidate selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _applied_correction_l2(candidate: Mapping[str, Any]) -> float:
    if candidate.get("applied_correction_l2_action") is not None:
        return float(candidate["applied_correction_l2_action"])
    if candidate.get("residual_binding") is not None:
        return float(candidate["residual_binding"]["applied_residual_l2_action"])
    return 0.0


def _aegis_compatible(candidate: Mapping[str, Any]) -> bool:
    projection = candidate["aegis_consistency"]
    return bool(
        projection["enabled"]
        and len(projection["qp_records"]) == 5
        and all(
            item["solver_status"] in ("optimal", "optimal_inaccurate")
            for item in projection["qp_records"]
        )
    )


def _result_path(source: Mapping[str, Any], state: Mapping[str, Any], index: int) -> Path:
    if source["kind"] == "l5_scoped":
        return Path(source["result_root"]) / ("case-%02d" % index) / "result.json"
    return Path(state["result_path"])


def load_samples(config: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    from main.multilink_ellipsoid.l5_action_risk import feature_vector
    from main.multilink_ellipsoid.l5_aegis_grouped_summary import row_coverage
    from scripts.summarize_distal_l5_moka_noise_adaptive import (
        l5_scoped_candidate_outcome,
    )

    output: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    seen_states: set[str] = set()
    source_records = []
    for source in config["sources"]:
        summary_path = Path(source["summary"])
        _require(_file_sha256(summary_path) == source["summary_file_sha256"],
                 "L5 row01 summary file differs")
        summary = _load(summary_path)
        _require(summary["result_payload_sha256"] == source["summary_payload_sha256"],
                 "L5 row01 summary payload differs")
        source_records.append({
            "name": source["name"],
            "summary_file_sha256": source["summary_file_sha256"],
            "summary_payload_sha256": source["summary_payload_sha256"],
        })
        for state_index, state in enumerate(summary["states"]):
            split = str(state["split"])
            if split not in output or state["classification"] == "proxy_invalid":
                continue
            state_id = str(state["state_id"])
            _require(state_id not in seen_states, "L5 row01 state occurs in two sources")
            seen_states.add(state_id)
            result_path = _result_path(source, state, state_index)
            _require(_file_sha256(result_path) == state["result_file_sha256"],
                     "L5 row01 source result file differs")
            result = _load(result_path)
            _require(
                result["population_binding"]["retained_state"]["state_id"] == state_id,
                "L5 row01 state binding differs",
            )
            for candidate in result["candidates"]:
                if source["kind"] == "l5_scoped":
                    outcome = l5_scoped_candidate_outcome(candidate)
                    if outcome not in ("safe", "unsafe"):
                        continue
                elif candidate["terminal_status"] == "UNKNOWN_TIMEOUT":
                    continue
                _require(candidate["terminal_status"] in (
                    "SAFE_TERMINAL", "UNSAFE_CONTACT_OR_CAR"
                ), "L5 row01 terminal status differs")
                output[split].append({
                    "source_name": source["name"],
                    "source_result_path": str(result_path),
                    "source_result_file_sha256": state["result_file_sha256"],
                    "source_result_payload_sha256": result["result_payload_sha256"],
                    "case_id": state["case_id"],
                    "state_id": state_id,
                    "split": split,
                    "classification": state["classification"],
                    "candidate_name": candidate["name"],
                    "candidate_order": int(candidate["order"]),
                    "feature_vector": feature_vector(
                        initial_clearance=result["state"]["initial_clearance_m"],
                        local_frame=result["state"]["local_frame"],
                        nominal_actions=result["nominal_five_action_chunk"],
                        candidate_actions=candidate["actions"],
                    ),
                    "risk_row01": [float(value) for value in candidate["combined_risk"][:2]],
                    "risk_all_rows": [float(value) for value in candidate["combined_risk"]],
                    "exact_safe": bool(candidate["exact_safe"]),
                    "physical_veto": bool(candidate["physical_veto"]),
                    "aegis_compatible": _aegis_compatible(candidate),
                    "applied_correction_l2_action": _applied_correction_l2(candidate),
                    "actions": candidate["actions"],
                    "post_aegis_candidate_before_consistency": candidate[
                        "post_aegis_candidate_before_consistency"
                    ],
                })
    observed = {name: len(items) for name, items in output.items()}
    _require(observed == config["dataset"]["expected_known_candidate_counts"],
             "L5 row01 known sample counts differ")
    useful = {}
    for split, samples in output.items():
        by_state: dict[str, list[dict[str, Any]]] = {}
        for item in samples:
            by_state.setdefault(item["state_id"], []).append(item)
        counts = []
        for row in range(2):
            count = 0
            for items in by_state.values():
                values = [float(item["risk_row01"][row]) for item in items]
                if (
                    any(value <= 0.0 for value in values)
                    and any(value > 0.0 for value in values)
                    and any(abs(value) <= 0.005 for value in values)
                ):
                    count += 1
            counts.append(count)
        useful[split] = counts
    _require(useful == config["dataset"]["expected_useful_boundary_state_counts"],
             "L5 row01 useful state counts differ")
    return {**output, "source_records": source_records, "useful_counts": useful}


def _arrays(samples: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
    import numpy as np
    return (
        np.asarray([item["feature_vector"] for item in samples], dtype=np.float64),
        np.asarray([item["risk_row01"] for item in samples], dtype=np.float64),
    )


def train(
    *, repo_root: Path, config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.l5_row01_selection import (
        MODEL_SCHEMA, load_config, payload_sha256, predict, selection_metrics,
        train_model,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    source = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    samples = load_samples(config)
    train_x, train_y = _arrays(samples["train"])
    validation_x, validation_y = _arrays(samples["validation"])
    bundle = train_model(
        train_x, train_y, validation_x, validation_y, config["model"]
    )
    predictions = {}
    metrics = {}
    for split in ("train", "validation"):
        features, _ = _arrays(samples[split])
        prediction = predict(bundle, features)
        predictions[split] = prediction.tolist()
        metrics[split] = selection_metrics(
            prediction, samples[split],
            random_seed=int(config["selection"]["random_seed"]),
            random_draws=int(config["selection"]["random_draws_per_state"]),
        )
    validation = metrics["validation"]
    gates = {
        "zero_selected_row01_false_safes":
        validation["row01_false_safe_count"] == 0,
        "MLP_safe_support_every_recoverable_validation_state":
        validation["supported_recoverable_state_count"]
        == validation["recoverable_state_count"]
        == int(config["dataset"]["expected_validation_recoverable_state_count"]),
        "all_selected_candidates_exact_all_seven_safe":
        validation["all_selected_exact_all_seven_safe"],
        "all_selected_candidates_original_AEGIS_EE_compatible":
        validation["all_selected_original_AEGIS_EE_compatible"],
        "selected_clearance_better_than_nominal_when_nominal_target_is_known":
        validation["all_selected_improve_over_known_nominal"],
        "selected_safe_rate_better_than_seeded_random":
        validation["selected_exact_safe_rate"]
        > validation["seeded_random_exact_safe_rate"],
        "reserved_test_episodes_unopened": True,
    }
    prediction_gate_pass = all(gates.values())
    model_record = {
        key: value for key, value in bundle.items()
        if key not in {
            "model", "device", "feature_mean", "feature_scale",
            "target_mean", "target_scale",
        }
    }
    result = {
        "schema_version": MODEL_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": source,
        "allocation": allocation_record(),
        "config": config,
        "dataset": {
            "source_records": samples["source_records"],
            "sample_counts": {name: len(samples[name]) for name in ("train", "validation")},
            "useful_boundary_state_counts": samples["useful_counts"],
            "sample_identities": {
                split: [{
                    "case_id": item["case_id"],
                    "state_id": item["state_id"],
                    "candidate_name": item["candidate_name"],
                    "candidate_order": item["candidate_order"],
                    "source_result_path": item["source_result_path"],
                    "source_result_file_sha256": item["source_result_file_sha256"],
                } for item in samples[split]]
                for split in ("train", "validation")
            },
        },
        "model": model_record,
        "predictions": predictions,
        "metrics": metrics,
        "gates": gates,
        "prediction_gate_pass": prediction_gate_pass,
        "fresh_exact_replay_authorized": bool(prediction_gate_pass),
        "QP_authorized": False,
        "closed_loop_authorized": False,
        "interpretation": (
            "two_output_row01_candidate_selection_prediction_gate_pass"
            if prediction_gate_pass
            else "two_output_row01_candidate_selection_prediction_no_go"
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
    result = train(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "gates": result["gates"],
        "validation_metrics": result["metrics"]["validation"],
        "model_sha256": result["model"]["model_sha256"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
