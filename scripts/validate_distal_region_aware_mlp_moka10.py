#!/usr/bin/env python3
"""Independently validate the region-aware unseen-state result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.region_aware_mlp import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, jaccard, load_config,
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = _load(args.result.resolve())
    config = load_config(args.config.resolve())
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("source", {}).get("dirty") is False
        and result.get("config", {}).get("config_file_sha256")
        == config["config_file_sha256"]
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256")
        and result.get("model_artifact", {}).get("file_sha256")
        == _file_sha256(args.model.resolve()),
        "region-aware result identity differs",
    )
    learned = []
    oracle = []
    false_safe = 0
    support = 0
    state_jaccards = []
    valid_qp = 0
    exact_safe = 0
    ee_compatible = 0
    shifts = []
    for state in result["test_state_results"]:
        learned_flags = [bool(value) for value in state["learned_accepted_flags"]]
        oracle_flags = [bool(value) for value in state["oracle_accepted_flags"]]
        true_flags = [bool(value) for value in state["true_safe_flags"]]
        _require(
            len(learned_flags) == len(oracle_flags) == len(true_flags) == 96,
            "region-aware state decision length differs",
        )
        false_indexes = [
            index for index, (predicted, true) in enumerate(zip(learned_flags, true_flags))
            if predicted and not true
        ]
        accepted_true = sum(
            predicted and true for predicted, true in zip(learned_flags, true_flags)
        )
        state_jaccard = jaccard(oracle_flags, learned_flags)
        _require(
            state["false_safe_fresh_indexes"] == false_indexes
            and int(state["accepted_true_safe_action_count"]) == accepted_true
            and abs(float(state["accepted_set_jaccard_to_oracle"]) - state_jaccard)
            <= 1.0e-15,
            "region-aware state decisions differ",
        )
        learned.extend(learned_flags)
        oracle.extend(oracle_flags)
        false_safe += len(false_indexes)
        support += int(accepted_true > 0)
        state_jaccards.append(state_jaccard)
        qp = state["QP"]
        exact = state["fresh_exact_two_step"]
        selected = None if qp is None else qp.get("selected")
        if selected is not None:
            valid_qp += 1
        if exact is not None:
            exact_safe += int(bool(exact["true_safe"]))
            ee_compatible += int(
                float(exact["minimum_released_AEGIS_EE_margin_m"]) >= 0.0
            )
        shift = state["selected_action_shift_from_oracle_l2"]
        if shift is not None:
            shifts.append(float(shift))
    aggregate = {
        "state_count": len(result["test_state_results"]),
        "off_grid_action_count": len(learned),
        "off_grid_false_safe_action_count": false_safe,
        "global_accepted_set_jaccard_to_oracle": jaccard(oracle, learned),
        "minimum_state_accepted_set_jaccard_to_oracle": min(state_jaccards),
        "safe_support_state_count": support,
        "valid_selected_QP_count": valid_qp,
        "fresh_exact_safe_selected_QP_count": exact_safe,
        "released_AEGIS_EE_compatible_selected_QP_count": ee_compatible,
        "selected_action_shift_from_oracle_l2_p95": (
            None if not shifts else float(np.quantile(shifts, 0.95))
        ),
        "selected_action_shift_from_oracle_l2_maximum": (
            None if not shifts else float(np.max(shifts))
        ),
    }
    recorded = result["test_aggregates"]
    for key, value in aggregate.items():
        if isinstance(value, float):
            _require(abs(value - float(recorded[key])) <= 1.0e-15,
                     "region-aware aggregate differs: %s" % key)
        else:
            _require(value == recorded[key],
                     "region-aware aggregate differs: %s" % key)
    gates = config["learned_gate"]
    preliminary = bool(
        false_safe == int(gates["test_off_grid_false_safe_action_count"])
        and aggregate["global_accepted_set_jaccard_to_oracle"]
        >= float(gates["minimum_global_accepted_set_jaccard_to_oracle"])
        and aggregate["minimum_state_accepted_set_jaccard_to_oracle"]
        >= float(gates["minimum_state_accepted_set_jaccard_to_oracle"])
        and support == int(gates["required_test_state_safe_support_count"])
    )
    p95 = aggregate["selected_action_shift_from_oracle_l2_p95"]
    maximum = aggregate["selected_action_shift_from_oracle_l2_maximum"]
    learned_pass = bool(
        preliminary
        and valid_qp == int(gates["required_valid_selected_QP_count"])
        and exact_safe == int(gates["required_fresh_exact_safe_selected_QP_count"])
        and ee_compatible
        == int(gates["required_released_AEGIS_EE_compatible_selected_QP_count"])
        and p95 is not None
        and p95 <= float(gates["selected_action_shift_from_oracle_l2_p95_maximum"])
        and maximum is not None
        and maximum <= float(gates["selected_action_shift_from_oracle_l2_maximum"])
    )
    _require(
        result["decision"]["off_grid_preliminary_gate_pass"] is preliminary
        and result["decision"]["learned_gate_pass"] is learned_pass
        and result["decision"]["closed_loop_e05_authorized"] is learned_pass
        and result["decision"]["closed_loop_e05_executed"] is False,
        "region-aware decision differs",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "scientific_result": False, "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "model_file_sha256": _file_sha256(args.model.resolve()),
        "aggregates": aggregate, "learned_gate_pass": learned_pass,
        "closed_loop_e05_authorized": learned_pass,
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
