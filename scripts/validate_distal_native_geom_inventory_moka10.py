#!/usr/bin/env python3
"""Independently validate the native MuJoCo geometry inventory result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.native_geom_margin import (
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
        "native inventory result identity differs",
    )
    states = result["state_results"]
    episodes = result["episode_results"]
    witnesses = result["primary_contact_witnesses"]
    _require(
        len(states) == config["population"]["expected_state_count"]
        and len(episodes) == config["population"]["expected_complete_episode_count"]
        and [int(item["state_index"]) for item in states]
        == list(range(len(states))),
        "native inventory validated population differs",
    )
    semantic_hashes = {
        str(item["semantic_protected_group_sha256"]) for item in episodes
    }
    rollouts = [item["nominal_two_action_rollout"] for item in states]
    aggregates = {
        "state_count": len(states), "episode_count": len(episodes),
        "protected_group_count": len(states[0]["semantic_protected_groups"]),
        "semantic_inventory_hash_count": len(semantic_hashes),
        "prevention_state_count": sum(item["cohort"] == "prevention" for item in states),
        "recovery_state_count": sum(item["cohort"] == "recovery" for item in states),
        "test_initially_native_safe_state_count": sum(
            item["split"] == "test" and item["cohort"] == "prevention"
            for item in states
        ),
        "primary_contact_witness_count": sum(
            int(item["raw_protected_contact_count"]) for item in witnesses
        ),
        "all_queries_finite_and_uncensored": all(
            item["all_queries_finite_and_uncensored"] for item in rollouts + witnesses
        ),
        "every_raw_contact_pair_registered": all(
            item["every_raw_contact_pair_registered"] for item in rollouts + witnesses
        ),
        "every_raw_contact_distance_consistent": all(
            item["every_raw_contact_distance_consistent"] for item in rollouts + witnesses
        ),
    }
    _require(aggregates == result["aggregates"],
             "native inventory aggregate recomputation differs")
    gate = config["gate"]
    passed = bool(
        aggregates["semantic_inventory_hash_count"] == 1
        and aggregates["all_queries_finite_and_uncensored"]
        and aggregates["every_raw_contact_pair_registered"]
        and aggregates["every_raw_contact_distance_consistent"]
        and aggregates["primary_contact_witness_count"]
        >= int(gate["required_primary_contact_witness_count"])
        and aggregates["test_initially_native_safe_state_count"]
        == int(gate["required_test_initially_native_safe_state_count"])
    )
    _require(
        result["decision"] == {
            "native_inventory_gate_pass": passed,
            "physical_target_collection_authorized": passed,
            "training_authorized": False, "QP_authorized": False,
            "closed_loop_E05_authorized": False,
            "stop_reason": None if passed else "native_geometry_inventory_gate_failed",
        },
        "native inventory decision differs",
    )
    validation = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "expected_commit": args.expected_commit,
        "aggregates": aggregates,
        "native_inventory_gate_pass": passed,
        "physical_target_collection_authorized": passed,
        "training_authorized": False, "closed_loop_E05_authorized": False,
    }
    validation["validation_payload_sha256"] = _payload(validation)
    _atomic_write(args.output.resolve(), validation)
    print(json.dumps(validation, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
