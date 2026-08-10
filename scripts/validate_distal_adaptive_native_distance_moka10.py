#!/usr/bin/env python3
"""Validate adaptive native-distance H100 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.adaptive_native_distance import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, load_config, result_payload,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def _payload(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("validation_payload_sha256", None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
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
        and result.get("result_payload_sha256") == result_payload(result)
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("source", {}).get("dirty") is False
        and result.get("config", {}).get("config_file_sha256")
        == config["config_file_sha256"],
        "adaptive native-distance result identity differs",
    )
    states = result["state_results"]
    episodes = result["episode_results"]
    witnesses = result["primary_contact_witnesses"]
    _require(
        len(states) == config["population"]["expected_state_count"]
        and len(episodes) == len(config["population"]["case_ids"]),
        "adaptive native-distance population differs",
    )
    measurements = [item["current_measurement"] for item in states]
    rollouts = [item["nominal_two_action_rollout"] for item in states]
    records = measurements + rollouts + witnesses
    aggregates = {
        "state_count": len(states), "episode_count": len(episodes),
        "semantic_protected_group_inventory_hash_count": len({
            item["semantic_protected_group_sha256"] for item in episodes
        }),
        "protected_group_count": episodes[0]["protected_group_count"],
        "initially_safe_test_state_count": sum(
            item["initially_safe"] for item in states
        ),
        "registered_raw_contact_count": sum(
            int(item["raw_protected_contact_count"]) for item in witnesses
        ),
        "negative_without_raw_pair_contact_count": sum(
            int(item["negative_without_raw_pair_contact_count"])
            for item in records
        ),
        "raw_pair_contact_without_negative_count": sum(
            int(item["raw_pair_contact_without_negative_count"])
            for item in records
        ),
        "cutoff_induced_negative_count": sum(
            int(item["cutoff_induced_negative_count"]) for item in records
        ),
        "repeated_uncensored_inconsistency_count": sum(
            int(item["repeated_uncensored_inconsistency_count"])
            for item in records
        ),
        "unregistered_raw_pair_count": sum(
            int(item["unregistered_raw_pair_count"]) for item in records
        ),
        "all_queries_finite": all(item["all_queries_finite"] for item in records),
    }
    _require(aggregates == result["aggregates"],
             "adaptive native-distance aggregates differ")
    gate = config["gate"]
    passed = bool(
        aggregates["semantic_protected_group_inventory_hash_count"]
        == gate["semantic_protected_group_inventory_hash_count"]
        and aggregates["negative_without_raw_pair_contact_count"]
        <= gate["maximum_negative_without_raw_pair_contact_count"]
        and aggregates["raw_pair_contact_without_negative_count"]
        <= gate["maximum_raw_pair_contact_without_negative_count"]
        and aggregates["cutoff_induced_negative_count"]
        <= gate["maximum_cutoff_induced_negative_count"]
        and aggregates["repeated_uncensored_inconsistency_count"]
        <= gate["maximum_repeated_uncensored_inconsistency_count"]
        and aggregates["unregistered_raw_pair_count"] == 0
        and aggregates["all_queries_finite"]
        and aggregates["registered_raw_contact_count"]
        >= gate["required_registered_raw_contact_count"]
        and aggregates["initially_safe_test_state_count"]
        == gate["required_initially_safe_test_state_count"]
    )
    _require(
        result["decision"] == {
            "adaptive_native_distance_gate_pass": passed,
            "full_physical_target_collection_preregistration_authorized": passed,
            "training_authorized": False, "QP_authorized": False,
            "closed_loop_E05_authorized": False,
            "stop_reason": None if passed else "adaptive_native_distance_gate_failed",
        },
        "adaptive native-distance decision differs",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "expected_commit": args.expected_commit, "aggregates": aggregates,
        "adaptive_native_distance_gate_pass": passed,
        "full_physical_target_collection_preregistration_authorized": passed,
        "training_authorized": False, "closed_loop_E05_authorized": False,
    }
    output["validation_payload_sha256"] = _payload(output)
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
