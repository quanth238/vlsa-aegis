#!/usr/bin/env python3
"""Independent verifier for the paired contact-gradient implementation audit."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_distal_contact_gradient_sign_e05 import (
    RESULT_SCHEMA,
    _hash_without,
    load_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write, _load, _require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    result = _load(args.result.resolve())
    _require(result.get("schema_version") == RESULT_SCHEMA, "gradient-sign result schema differs")
    _require(result.get("config") == config, "gradient-sign result config differs")
    _require(result.get("source", {}).get("commit") == args.expected_commit, "gradient-sign source commit differs")
    _require(result.get("source", {}).get("dirty") is False, "gradient-sign source is dirty")
    _require(result.get("result_payload_sha256") == _hash_without(result, "result_payload_sha256"), "gradient-sign payload differs")
    epsilons = [float(value) for value in config["epsilon_action"]]
    tolerance = float(config["model_consistency_gate"]["strict_risk_ordering_tolerance"])
    relative_limit = float(config["model_consistency_gate"]["smallest_epsilon_central_difference_relative_error_maximum"])
    all_state_pass = []
    minus_count = plus_count = tie_count = 0
    for step in config["test_steps"]:
        records = [item for item in result["records"] if item["state_step"] == step]
        _require(len(records) == len(epsilons), "gradient-sign record count differs")
        _require([item["epsilon_action"] for item in records] == epsilons, "gradient-sign epsilon order differs")
        for item in records:
            epsilon = float(item["epsilon_action"])
            _require(item["model_ordering_pass"] is (item["minus"]["risk"] + tolerance < item["plus"]["risk"]), "gradient-sign model ordering differs")
            derivative = (item["plus"]["risk"] - item["minus"]["risk"]) / (2.0 * epsilon)
            _require(abs(derivative - item["finite_difference_directional_derivative"]) <= 1e-15, "gradient-sign derivative differs")
            _require(item["indexing_pass"] is True, "gradient-sign indexing receipt failed")
            _require(item["simulator_preference"] in {"minus", "plus", "tie"}, "gradient-sign simulator preference differs")
            minus_count += item["simulator_preference"] == "minus"
            plus_count += item["simulator_preference"] == "plus"
            tie_count += item["simulator_preference"] == "tie"
        state_pass = bool(
            all(item["model_ordering_pass"] and item["indexing_pass"] for item in records)
            and records[0]["finite_difference_relative_error"] <= relative_limit
        )
        stored = next(item for item in result["state_results"] if item["state_step"] == step)
        _require(stored["all_model_orderings_pass"] is all(item["model_ordering_pass"] for item in records), "gradient-sign state model gate differs")
        _require(stored["all_indexing_checks_pass"] is True, "gradient-sign state indexing gate differs")
        _require(stored["smallest_epsilon_finite_difference_pass"] is (records[0]["finite_difference_relative_error"] <= relative_limit), "gradient-sign finite-difference gate differs")
        all_state_pass.append(state_pass)
    implementation_pass = bool(all(all_state_pass))
    misalignment = bool(implementation_pass and plus_count > 0)
    decision = result["decision"]
    _require(decision["implementation_consistency_pass"] is implementation_pass, "gradient-sign implementation decision differs")
    _require(decision["simulator_misalignment_observed"] is misalignment, "gradient-sign alignment decision differs")
    _require(decision["simulator_minus_preference_count"] == minus_count, "gradient-sign minus count differs")
    _require(decision["simulator_plus_preference_count"] == plus_count, "gradient-sign plus count differs")
    _require(decision["simulator_tie_count"] == tie_count, "gradient-sign tie count differs")
    _require(decision["closed_loop_authorized"] is False, "gradient-sign closed-loop must remain unauthorized")
    _require(result["total_fresh_rollout_count"] == 2 * len(config["test_steps"]) * len(epsilons), "gradient-sign rollout count differs")
    validation = {
        "schema_version": "vlsa_distal_contact_gradient_sign_audit_e05_validation.v1",
        "status": "passed",
        "scientific_result": True,
        "expected_commit": args.expected_commit,
        "result_payload_sha256": result["result_payload_sha256"],
        "implementation_consistency_pass": implementation_pass,
        "simulator_misalignment_observed": misalignment,
        "closed_loop_authorized": False,
    }
    validation["validation_payload_sha256"] = hashlib.sha256(
        json.dumps(validation, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    _atomic_write(args.output.resolve(), validation)
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

