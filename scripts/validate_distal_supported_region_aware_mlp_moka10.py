#!/usr/bin/env python3
"""Independently validate the supported-state regional MLP result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.region_aware_mlp import jaccard
from main.multilink_ellipsoid.supported_region_aware_mlp import (
    RESULT_SCHEMA, VALIDATION_FRESH_SCHEMA, VALIDATION_SCHEMA, fresh_payload,
    load_config,
)
from scripts.evaluate_distal_supported_region_aware_mlp_moka10 import (
    _test_aggregates,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation-fresh", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = _load(args.result.resolve())
    fresh = _load(args.validation_fresh.resolve())
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
        == _file_sha256(args.model.resolve())
        and fresh.get("schema_version") == VALIDATION_FRESH_SCHEMA
        and fresh.get("source", {}).get("commit") == args.expected_commit
        and fresh.get("source", {}).get("dirty") is False
        and fresh.get("validation_fresh_payload_sha256")
        == fresh_payload(fresh)
        and result.get("validation_fresh", {}).get("file_sha256")
        == _file_sha256(args.validation_fresh.resolve()),
        "supported region-aware result identity differs",
    )
    _require(
        fresh.get("summary", {}).get("state_count") == 5
        and fresh.get("summary", {}).get("fresh_action_count") == 480
        and [int(item["state_index"]) for item in fresh["state_results"]]
        == config["split"]["new_validation_state_indexes"]
        and all(
            len(item["fresh_actions"]) == 96 for item in fresh["state_results"]
        ),
        "supported region-aware validation calibration population differs",
    )
    for state in result["test_state_results"]:
        learned = [bool(value) for value in state["learned_accepted_flags"]]
        oracle = [bool(value) for value in state["oracle_accepted_flags"]]
        true = [bool(value) for value in state["true_safe_flags"]]
        false_indexes = [
            index for index, (predicted, safe) in enumerate(zip(learned, true))
            if predicted and not safe
        ]
        accepted_true = sum(
            predicted and safe for predicted, safe in zip(learned, true)
        )
        _require(
            len(learned) == len(oracle) == len(true) == 96
            and state["false_safe_fresh_indexes"] == false_indexes
            and int(state["accepted_true_safe_action_count"]) == accepted_true
            and abs(
                float(state["accepted_set_jaccard_to_oracle"])
                - jaccard(oracle, learned)
            ) <= 1.0e-15,
            "supported region-aware state decision differs",
        )
    aggregates = _test_aggregates(result["test_state_results"])
    for key, value in aggregates.items():
        recorded = result["test_aggregates"].get(key)
        if isinstance(value, float):
            _require(
                abs(value - float(recorded)) <= 1.0e-15,
                "supported region-aware aggregate differs: %s" % key,
            )
        else:
            _require(
                value == recorded,
                "supported region-aware aggregate differs: %s" % key,
            )
    gates = config["learned_gate"]
    preliminary = bool(
        aggregates["off_grid_false_safe_action_count"]
        == int(gates["test_off_grid_false_safe_action_count"])
        and aggregates["global_accepted_set_jaccard_to_oracle"]
        >= float(gates["minimum_global_accepted_set_jaccard_to_oracle"])
        and aggregates["minimum_state_accepted_set_jaccard_to_oracle"]
        >= float(gates["minimum_state_accepted_set_jaccard_to_oracle"])
        and aggregates["safe_support_state_count"]
        == int(gates["required_test_state_safe_support_count"])
    )
    p95 = aggregates["selected_action_shift_from_oracle_l2_p95"]
    maximum = aggregates["selected_action_shift_from_oracle_l2_maximum"]
    learned_pass = bool(
        preliminary
        and aggregates["valid_selected_QP_count"]
        == int(gates["required_valid_selected_QP_count"])
        and aggregates["fresh_exact_safe_selected_QP_count"]
        == int(gates["required_fresh_exact_safe_selected_QP_count"])
        and aggregates["released_AEGIS_EE_compatible_selected_QP_count"]
        == int(gates[
            "required_released_AEGIS_EE_compatible_selected_QP_count"
        ])
        and p95 is not None
        and p95 <= float(
            gates["selected_action_shift_from_oracle_l2_p95_maximum"]
        )
        and maximum is not None
        and maximum <= float(
            gates["selected_action_shift_from_oracle_l2_maximum"]
        )
    )
    _require(
        result["decision"]["off_grid_preliminary_gate_pass"] is preliminary
        and result["decision"]["learned_gate_pass"] is learned_pass
        and result["decision"][
            "receding_closed_loop_E05_preregistration_authorized"
        ] is learned_pass
        and result["decision"][
            "action_conditioned_model_preregistration_authorized"
        ] is (not learned_pass)
        and result["decision"]["closed_loop_e05_executed"] is False,
        "supported region-aware decision differs",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "scientific_result": True, "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "model_file_sha256": _file_sha256(args.model.resolve()),
        "validation_fresh_file_sha256": _file_sha256(
            args.validation_fresh.resolve()
        ),
        "aggregates": aggregates, "learned_gate_pass": learned_pass,
        "receding_closed_loop_E05_preregistration_authorized": learned_pass,
        "action_conditioned_model_preregistration_authorized": not learned_pass,
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
