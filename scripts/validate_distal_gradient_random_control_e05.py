#!/usr/bin/env python3
"""Independent verifier for the matched-random learned-gradient control."""

import argparse
import hashlib
import json
from pathlib import Path

from main.multilink_ellipsoid.gradient_random_control import (
    empirical_equal_or_earlier_p, first_safe_radius,
)
from scripts.evaluate_distal_gradient_random_control_e05 import (
    RESULT_SCHEMA, _hash_without, load_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write, _load, _require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve()); result = _load(args.result.resolve())
    _require(result.get("schema_version") == RESULT_SCHEMA and result.get("config") == config, "random-control result contract differs")
    _require(result.get("source", {}).get("commit") == args.expected_commit and result.get("source", {}).get("dirty") is False, "random-control source differs")
    _require(result.get("result_payload_sha256") == _hash_without(result, "result_payload_sha256"), "random-control payload differs")
    radii = [float(value) for value in config["correction_radii_action"]]
    count = int(config["random_control"]["direction_count_per_state"])
    threshold = float(config["randomization_gate"]["empirical_equal_or_earlier_safe_p_maximum"])
    recomputed = []
    for step in config["test_steps"]:
        learned = [item for item in result["learned_records"] if item["state_step"] == step]
        random = [item for item in result["random_records"] if item["state_step"] == step]
        _require(len(learned) == len(radii) and len(random) == count * len(radii), "random-control record population differs")
        _require([item["radius_action"] for item in learned] == radii, "learned radii differ")
        learned_first = first_safe_radius(radii, [item["D_sim_raw_safe"] for item in learned])
        random_first = []
        for index in range(count):
            selected = [item for item in random if item["direction_index"] == index]
            _require(len(selected) == len(radii) and [item["radius_action"] for item in selected] == radii, "random direction radii differ")
            random_first.append(first_safe_radius(radii, [item["D_sim_raw_safe"] for item in selected]))
        p_value = empirical_equal_or_earlier_p(learned_first, random_first)
        stored = next(item for item in result["state_results"] if item["state_step"] == step)
        passed = bool(learned_first is not None and p_value <= threshold)
        _require(stored["learned_first_safe_radius_action"] == learned_first, "learned first-safe radius differs")
        _require(abs(stored["empirical_equal_or_earlier_safe_p"] - p_value) <= 1e-15, "empirical p differs")
        _require(stored["directional_advantage_pass"] is passed, "state decision differs")
        recomputed.append(passed)
    advantage = bool(all(recomputed))
    _require(result["decision"]["learned_gradient_beats_matched_random"] is advantage, "overall random-control decision differs")
    _require(result["decision"]["closed_loop_authorized"] is False, "closed-loop must remain unauthorized")
    validation = {
        "schema_version": "vlsa_distal_gradient_random_control_e05_validation.v1",
        "status": "passed", "scientific_result": True,
        "expected_commit": args.expected_commit,
        "result_payload_sha256": result["result_payload_sha256"],
        "learned_gradient_beats_matched_random": advantage,
        "closed_loop_authorized": False,
    }
    validation["validation_payload_sha256"] = hashlib.sha256(json.dumps(
        validation, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    _atomic_write(args.output.resolve(), validation)
    print(json.dumps(validation, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
