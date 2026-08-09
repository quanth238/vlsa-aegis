#!/usr/bin/env python3
"""Independently validate the no-training affine oracle comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.affine_oracle_comparison import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, load_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.result.resolve()
    dataset_path = args.dataset.resolve()
    config = load_config(args.config.resolve())
    result = _load(result_path)
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256")
        and result.get("config", {}).get("config_file_sha256")
        == config["config_file_sha256"]
        and _file_sha256(dataset_path)
        == config["immutable_source"]["dataset_file_sha256"]
        == result.get("immutable_dataset", {}).get("file_sha256"),
        "affine oracle-comparison identity differs",
    )
    states = result.get("state_results", [])
    expected_states = int(config["immutable_source"]["expected_state_count"])
    fresh_count = int(config["fresh_actions"]["count_per_state"])
    maximum_motion = float(config["exact_verification"][
        "maximum_per_step_obstacle_l1_displacement_m"
    ])
    _require(
        len(states) == expected_states
        and [item["state_index"] for item in states] == list(range(expected_states)),
        "affine oracle-comparison state count/index differs",
    )
    flattened: dict[str, list[tuple[bool, bool]]] = {
        "direct_halfspace": [], "finite_difference": [],
    }
    supported_states = 0
    accepted_states = {name: 0 for name in flattened}
    for state in states:
        _require(
            state["feature_max_abs_error"] <= 1.0e-8
            and state["clearance_max_abs_error_m"] <= 1.0e-8
            and state["fit_action_count"]
            == int(config["immutable_source"]["expected_fit_actions_per_state"])
            and len(state["finite_difference_probe_records"]) == 6
            and len(state["fresh_actions"]) == fresh_count
            and state["direct_halfspace"]["all_fit_false_safe_count"] == 0
            and state["finite_difference"]["maximum_fit_overbound_m"] <= 1.0e-10
            and state["finite_difference"]["fit_false_safe_entry_count"] == 0,
            "affine oracle-comparison state apparatus gate differs",
        )
        true_safe_count = 0
        state_pairs = {name: [] for name in flattened}
        for candidate in state["fresh_actions"]:
            margins = np.asarray(
                candidate["minimum_distal_margin_m"], dtype=np.float64
            )
            proxy_safe = bool(np.all(margins >= 0.0))
            raw_safe = bool(
                candidate["D_sim_raw_safe"]
                and candidate["raw_protected_contact_count"] == 0
                and candidate["maximum_within_step_obstacle_l1_displacement_m"]
                <= maximum_motion
            )
            true_safe = bool(proxy_safe and raw_safe)
            _require(
                candidate["D_opt_seven_distal_safe"] is proxy_safe
                and candidate["true_safe"] is true_safe,
                "affine oracle-comparison exact fresh label differs",
            )
            true_safe_count += int(true_safe)
            for method in flattened:
                key = "row_values" if method == "direct_halfspace" else "row_values_m"
                values = np.asarray(
                    candidate["predictions"][method][key], dtype=np.float64
                )
                predicted = bool(np.all(values >= 0.0))
                _require(
                    candidate["predictions"][method]["all_rows_safe"] is predicted,
                    "affine oracle-comparison prediction flag differs",
                )
                pair = (true_safe, predicted)
                flattened[method].append(pair)
                state_pairs[method].append(pair)
        _require(
            state["true_safe_fresh_action_count"] == true_safe_count,
            "affine oracle-comparison state safe support differs",
        )
        if true_safe_count > 0:
            supported_states += 1
            for method, pairs in state_pairs.items():
                if any(true and predicted for true, predicted in pairs):
                    accepted_states[method] += 1
        for method, pairs in state_pairs.items():
            true_count = sum(true for true, _ in pairs)
            accepted = sum(true and predicted for true, predicted in pairs)
            false_safe = sum((not true) and predicted for true, predicted in pairs)
            summary = state["method_summaries"][method]
            _require(
                summary["fresh_action_count"] == fresh_count
                and summary["true_safe_action_count"] == true_count
                and summary["accepted_true_safe_action_count"] == accepted
                and summary["false_safe_action_count"] == false_safe,
                "affine oracle-comparison state summary differs",
            )
    recomputed = {}
    gates = config["decision_gate"]
    for method, pairs in flattened.items():
        true_count = sum(true for true, _ in pairs)
        predicted_count = sum(predicted for _, predicted in pairs)
        accepted = sum(true and predicted for true, predicted in pairs)
        false_safe = sum((not true) and predicted for true, predicted in pairs)
        recall = accepted / true_count if true_count else None
        coverage = accepted_states[method] / supported_states if supported_states else None
        passed = bool(
            false_safe == int(gates["false_safe_action_count"])
            and recall is not None
            and recall >= float(gates["minimum_aggregate_safe_action_recall"])
            and coverage is not None
            and coverage >= float(gates["minimum_safe_support_state_coverage"])
        )
        recomputed[method] = {
            "fresh_action_count": len(pairs),
            "true_safe_action_count": true_count,
            "predicted_safe_action_count": predicted_count,
            "accepted_true_safe_action_count": accepted,
            "false_safe_action_count": false_safe,
            "safe_action_recall": recall,
            "fresh_safe_support_state_count": supported_states,
            "accepted_safe_support_state_count": accepted_states[method],
            "safe_support_state_coverage": coverage,
            "method_gate_pass": passed,
        }
        recorded = result["method_aggregates"][method]
        for key, value in recomputed[method].items():
            _require(recorded[key] == value, "affine oracle-comparison aggregate differs")
    supported = bool(any(item["method_gate_pass"] for item in recomputed.values()))
    decision = result["decision"]
    _require(
        decision["direct_halfspace_gate_pass"]
        is recomputed["direct_halfspace"]["method_gate_pass"]
        and decision["finite_difference_gate_pass"]
        is recomputed["finite_difference"]["method_gate_pass"]
        and decision["oracle_affine_assumption_supported"] is supported
        and decision["future_direct_halfspace_training_authorized"] is supported
        and decision["closed_loop_e05_authorized"] is False
        and result["rollout_counts"] == {
            "finite_difference": expected_states * 6,
            "fresh_evaluation": expected_states * fresh_count,
            "total_new_cloned_OSC": expected_states * (6 + fresh_count),
        },
        "affine oracle-comparison decision differs",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "scientific_result": False, "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "dataset_file_sha256": _file_sha256(dataset_path),
        "method_aggregates": recomputed,
        "oracle_affine_assumption_supported": supported,
        "closed_loop_e05_authorized": False,
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
