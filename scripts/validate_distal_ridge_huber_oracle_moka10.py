#!/usr/bin/env python3
"""Independently validate the ridge-Huber oracle decision artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.ridge_huber_oracle import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, affine_values, load_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


METHODS = ("ridge_huber", "minimum_L1")


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--off-grid-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.result.resolve()
    dataset_path = args.dataset.resolve()
    off_grid_path = args.off_grid_result.resolve()
    config = load_config(args.config.resolve())
    result = _load(result_path)
    dataset = _load(dataset_path)
    off_grid = _load(off_grid_path)
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256")
        and result.get("config", {}).get("config_file_sha256")
        == config["config_file_sha256"]
        and _file_sha256(dataset_path)
        == config["immutable_source"]["dataset_file_sha256"]
        and _file_sha256(off_grid_path)
        == config["immutable_source"]["off_grid_result_file_sha256"],
        "ridge-Huber result identity differs",
    )
    source_states = {item["state_index"]: item for item in dataset["state_records"]}
    fresh_states = {item["state_index"]: item for item in off_grid["state_results"]}
    states = result.get("state_results", [])
    expected_count = int(config["immutable_source"]["expected_state_count"])
    _require(
        len(states) == expected_count
        and [item["state_index"] for item in states] == list(range(expected_count))
        and set(source_states) == set(fresh_states) == set(range(expected_count)),
        "ridge-Huber state set differs",
    )
    pairs = {method: [] for method in METHODS}
    state_accept_counts = {method: 0 for method in METHODS}
    qp_counts = {method: 0 for method in METHODS}
    stability_gate = True
    minimum_l1_stability_gate = True
    maximum_motion = float(config["exact_verification"][
        "maximum_per_step_obstacle_l1_displacement_m"
    ])
    for state in states:
        index = int(state["state_index"])
        source = source_states[index]
        fresh_source = fresh_states[index]
        _require(
            state["case_id"] == source["case_id"] == fresh_source["case_id"]
            and state["state_step"] == source["state_step"] == fresh_source["state_step"]
            and state["split"] == source["split"] == fresh_source["split"]
            and state["feature_max_abs_error"] <= 1.0e-8
            and state["clearance_max_abs_error_m"] <= 1.0e-8,
            "ridge-Huber state identity/receipt differs",
        )
        fit_xyz = np.asarray(source["candidate_first_xyz"], dtype=np.float64)
        fit_margins = np.asarray(
            source["candidate_minimum_distal_margin_m"], dtype=np.float64
        )
        nominal = np.asarray(source["nominal_first_action"][:3], dtype=np.float64)
        ridge = state["ridge_huber_target"]
        ridge_fit = np.asarray([
            affine_values(
                ridge["nominal_margin_m"], ridge["gradient_m_per_action"],
                ridge["one_sided_error_m"], xyz, nominal,
            ) for xyz in fit_xyz
        ])
        _require(
            float(np.max(ridge_fit - fit_margins)) <= 1.0e-10
            and not np.any(np.logical_and(ridge_fit >= 0.0, fit_margins < 0.0)),
            "ridge-Huber fit lower bound differs",
        )
        audits = state["resampling_stability"]["row_audits"]
        stable = bool(all((not item["active"]) or item["stable"] for item in audits))
        _require(
            len(audits) == 7
            and state["resampling_stability"]["all_active_rows_stable"] is stable,
            "ridge-Huber stability decision differs",
        )
        stability_gate = bool(stability_gate and stable)
        l1_audits = state["minimum_L1_resampling_stability"]["row_audits"]
        l1_stable = bool(all(
            (not item["active"]) or item["stable"] for item in l1_audits
        ))
        _require(
            len(l1_audits) == 7
            and state["minimum_L1_resampling_stability"][
                "all_active_rows_stable"
            ] is l1_stable,
            "minimum-L1 stability decision differs",
        )
        minimum_l1_stability_gate = bool(
            minimum_l1_stability_gate and l1_stable
        )
        local_pairs = {method: [] for method in METHODS}
        _require(
            len(state["fresh_actions"])
            == int(config["immutable_source"]["off_grid_actions_per_state"]),
            "ridge-Huber off-grid action count differs",
        )
        for record, immutable in zip(
            state["fresh_actions"], fresh_source["fresh_actions"]
        ):
            _require(
                record["fresh_index"] == immutable["fresh_index"]
                and record["candidate_xyz"] == immutable["candidate_xyz"]
                and record["minimum_distal_margin_m"]
                == immutable["minimum_distal_margin_m"]
                and record["true_safe"] is immutable["true_safe"],
                "ridge-Huber immutable off-grid label differs",
            )
            true_safe = bool(record["true_safe"])
            flags = {
                "ridge_huber": bool(np.all(np.asarray(
                    record["ridge_huber_values_m"], dtype=np.float64
                ) >= 0.0)),
                "minimum_L1": bool(np.all(np.asarray(
                    record["minimum_L1_values_m"], dtype=np.float64
                ) >= 0.0)),
            }
            for method in METHODS:
                _require(
                    record[method + "_predicted_safe"] is flags[method],
                    "ridge-Huber off-grid prediction flag differs",
                )
                pair = (true_safe, flags[method])
                pairs[method].append(pair)
                local_pairs[method].append(pair)
        for method in METHODS:
            accepted = sum(true and predicted for true, predicted in local_pairs[method])
            false_safe = sum(
                (not true) and predicted for true, predicted in local_pairs[method]
            )
            summary = state["off_grid_summaries"][method]
            _require(
                summary["accepted_true_safe_action_count"] == accepted
                and summary["false_safe_action_count"] == false_safe,
                "ridge-Huber per-state off-grid summary differs",
            )
            state_accept_counts[method] += int(accepted >= 1)
            qp = state["QP"][method]
            exact = qp["fresh_exact_two_step"]
            exact_safe = bool(
                exact is not None and exact["D_opt_seven_distal_safe"]
                and exact["D_sim_raw_safe"]
                and exact["raw_protected_contact_count"] == 0
                and exact["maximum_within_step_obstacle_l1_displacement_m"]
                <= maximum_motion
            )
            gate = bool(
                qp["solution"]["valid"]
                and qp["solution"]["diagnostics"]["input_constraint_count"] == 7
                and exact_safe
            )
            _require(qp["gate_pass"] is gate, "ridge-Huber QP gate differs")
            qp_counts[method] += int(gate)
    recomputed = {}
    for method in METHODS:
        true_count = sum(true for true, _ in pairs[method])
        accepted = sum(true and predicted for true, predicted in pairs[method])
        false_safe = sum((not true) and predicted for true, predicted in pairs[method])
        predicted = sum(predicted for _, predicted in pairs[method])
        recomputed[method] = {
            "action_count": len(pairs[method]),
            "true_safe_action_count": true_count,
            "predicted_safe_action_count": predicted,
            "accepted_true_safe_action_count": accepted,
            "false_safe_action_count": false_safe,
            "safe_action_recall": accepted / true_count if true_count else None,
            "states_with_accepted_safe_action": state_accept_counts[method],
            "states_with_valid_fresh_safe_QP": qp_counts[method],
        }
        _require(
            result["method_aggregates"][method] == recomputed[method],
            "ridge-Huber aggregate differs",
        )
    ridge_gate = bool(
        recomputed["ridge_huber"]["false_safe_action_count"] == 0
        and recomputed["ridge_huber"]["states_with_accepted_safe_action"]
        == expected_count
        and stability_gate
        and recomputed["ridge_huber"]["states_with_valid_fresh_safe_QP"]
        == expected_count
    )
    l1_gate = bool(
        recomputed["minimum_L1"]["false_safe_action_count"] == 0
        and recomputed["minimum_L1"]["states_with_accepted_safe_action"]
        == expected_count
        and recomputed["minimum_L1"]["states_with_valid_fresh_safe_QP"]
        == expected_count
    )
    decision = result["decision"]
    _require(
        decision["ridge_huber_oracle_gate_pass"] is ridge_gate
        and decision["minimum_L1_comparator_gate_pass"] is l1_gate
        and decision["all_active_rows_resampling_stable"] is stability_gate
        and decision["minimum_L1_all_active_rows_resampling_stable"]
        is minimum_l1_stability_gate
        and decision["learned_training_authorized"] is ridge_gate
        and decision["learned_training_executed"] is False
        and decision["closed_loop_e05_authorized"] is False
        and result["new_cloned_OSC_rollout_count"] == expected_count * 2,
        "ridge-Huber final decision differs",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "scientific_result": False, "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "dataset_file_sha256": _file_sha256(dataset_path),
        "off_grid_result_file_sha256": _file_sha256(off_grid_path),
        "method_aggregates": recomputed,
        "all_active_rows_resampling_stable": stability_gate,
        "minimum_L1_all_active_rows_resampling_stable": (
            minimum_l1_stability_gate
        ),
        "ridge_huber_oracle_gate_pass": ridge_gate,
        "minimum_L1_comparator_gate_pass": l1_gate,
        "learned_training_authorized": ridge_gate,
        "closed_loop_e05_authorized": False,
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
