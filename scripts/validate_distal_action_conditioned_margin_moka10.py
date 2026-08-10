#!/usr/bin/env python3
"""Independently validate the action-conditioned margin experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.action_conditioned_margin import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, jaccard, load_config, target_values,
)
from scripts.evaluate_distal_action_conditioned_margin_moka10 import _aggregates
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
    parser.add_argument("--expanded-oracle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = _load(args.result.resolve())
    oracle = _load(args.expanded_oracle.resolve())
    config = load_config(args.config.resolve())
    source = config["immutable_source"]
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
        and _file_sha256(args.expanded_oracle.resolve())
        == source["expanded_oracle_file_sha256"],
        "action-conditioned result identity differs",
    )
    oracle_by_index = {
        int(item["state_index"]): item for item in oracle["state_results"]
    }
    for state in result["test_state_results"]:
        source_state = oracle_by_index[int(state["state_index"])]
        targets = state["predicted_targets"]
        learned = []
        oracle_flags = []
        true_flags = []
        for action in source_state["fresh_actions"]:
            learned.append(bool(any(
                all(value >= 0.0 for value in target_values(
                    targets[int(region_index)], action["candidate_xyz"]
                ))
                for region_index in action["containing_region_indexes"]
            )))
            oracle_flags.append(bool(
                action["arms"]["0mm"]["multi_region_predicted_safe"]
            ))
            true_flags.append(bool(action["arms"]["0mm"]["true_safe"]))
        false_indexes = [
            index for index, (accepted, safe)
            in enumerate(zip(learned, true_flags))
            if accepted and not safe
        ]
        accepted_true = sum(
            accepted and safe for accepted, safe in zip(learned, true_flags)
        )
        _require(
            learned == state["learned_accepted_flags"]
            and oracle_flags == state["oracle_accepted_flags"]
            and true_flags == state["true_safe_flags"]
            and false_indexes == state["false_safe_fresh_indexes"]
            and accepted_true == int(state["accepted_true_safe_action_count"])
            and abs(
                jaccard(oracle_flags, learned)
                - float(state["accepted_set_jaccard_to_oracle"])
            ) <= 1.0e-15,
            "action-conditioned state decision differs",
        )
    aggregates = _aggregates(result["test_state_results"])
    for key, value in aggregates.items():
        recorded = result["test_aggregates"].get(key)
        if isinstance(value, float):
            _require(
                abs(value - float(recorded)) <= 1.0e-15,
                "action-conditioned aggregate differs: %s" % key,
            )
        else:
            _require(
                value == recorded,
                "action-conditioned aggregate differs: %s" % key,
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
        == int(gates["required_released_AEGIS_EE_compatible_selected_QP_count"])
        and p95 is not None
        and p95 <= float(gates["selected_action_shift_from_oracle_l2_p95_maximum"])
        and maximum is not None
        and maximum <= float(gates["selected_action_shift_from_oracle_l2_maximum"])
    )
    _require(
        result["decision"]["off_grid_preliminary_gate_pass"] is preliminary
        and result["decision"]["learned_gate_pass"] is learned_pass
        and result["decision"][
            "receding_closed_loop_E05_preregistration_authorized"
        ] is learned_pass
        and result["decision"]["plain_action_conditioned_MLP_rejected"]
        is (not learned_pass)
        and result["decision"]["closed_loop_e05_executed"] is False,
        "action-conditioned decision differs",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "scientific_result": True, "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "model_file_sha256": _file_sha256(args.model.resolve()),
        "aggregates": aggregates, "learned_gate_pass": learned_pass,
        "receding_closed_loop_E05_preregistration_authorized": learned_pass,
        "plain_action_conditioned_MLP_rejected": not learned_pass,
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
