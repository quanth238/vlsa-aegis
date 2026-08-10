#!/usr/bin/env python3
"""Independently validate the distal margin-target authority audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.margin_target_authority import (
    RESULT_SCHEMA, VALIDATION_SCHEMA, load_config,
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = _load(args.result.resolve())
    config = load_config(args.config.resolve())
    source = config["immutable_source"]
    population = result["population"]
    aggregate = result["aggregates"]
    case_dangerous = sum(
        int(item["ellipsoid_safe_raw_distal_unsafe_count"])
        for item in result["case_results"]
    )
    gate_pass = case_dangerous <= int(
        config["gate"]["maximum_ellipsoid_safe_raw_distal_unsafe_count"]
    )
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
        and int(population["state_count"]) == int(source["expected_state_count"])
        and int(population["fit_action_count"])
        == int(source["expected_fit_action_count"])
        and int(population["fresh_state_count"])
        == int(source["expected_fresh_state_count"])
        and int(population["fresh_action_count"])
        == int(source["expected_fresh_action_count"])
        and int(aggregate["ellipsoid_safe_raw_distal_unsafe_count"])
        == case_dangerous
        and aggregate["target_authority_gate_pass"] is gate_pass
        and aggregate["local_residual_calibration_preregistration_authorized"]
        is gate_pass
        and aggregate["closed_loop_E05_authorized"] is False
        and result["decision"]["training_executed"] is False
        and result["decision"]["QP_executed"] is False
        and result["decision"]["new_simulation_executed"] is False,
        "margin-target result differs",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA, "status": "valid",
        "scientific_result": True, "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "population": population, "aggregates": aggregate,
        "target_authority_gate_pass": gate_pass,
        "local_residual_calibration_preregistration_authorized": gate_pass,
        "closed_loop_E05_authorized": False,
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
