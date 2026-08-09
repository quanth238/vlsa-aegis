#!/usr/bin/env python3
"""Independent verifier for paired directional contact-risk training."""

import argparse
import hashlib
import json
from pathlib import Path

from main.multilink_ellipsoid.contact_directional_ranker import (
    burden_key,
    empirical_burden_p,
)
from scripts.evaluate_distal_contact_directional_ranker_e05 import (
    RESULT_SCHEMA,
    _hash_without,
    load_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _load,
    _require,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    result = _load(args.result.resolve())
    _require(result.get("schema_version") == RESULT_SCHEMA, "directional result schema differs")
    _require(result.get("config") == config, "directional result config differs")
    _require(result.get("source", {}).get("commit") == args.expected_commit, "directional source commit differs")
    _require(result.get("source", {}).get("dirty") is False, "directional source is dirty")
    _require(result.get("result_payload_sha256") == _hash_without(result, "result_payload_sha256"), "directional result payload differs")
    _require(result.get("model", {}).get("file_sha256") == _file_sha256(args.model.resolve()), "directional model file differs")
    groups = config["state_groups"]
    _require(all(item["split"] != "test" for item in result["pair_records"] if item["split"] in {"train", "validation"}), "directional pair split audit failed")
    validation_metrics = result["training"]["pair_metrics"]["validation"]
    validation_accuracy = validation_metrics["informative_pair_accuracy"]
    validation_pass = bool(
        validation_metrics["informative_pair_count"] > 0
        and validation_accuracy >= float(config["decision_gate"]["minimum_validation_informative_pair_accuracy"])
    )
    random_count = int(config["matched_random_test"]["random_direction_count_per_state"])
    state_passes = []
    for step in groups["test_steps"]:
        learned = [item for item in result["learned_records"] if item["state_step"] == step]
        random = [item for item in result["random_records"] if item["state_step"] == step]
        _require(len(learned) == 2 and len(random) == random_count, "directional test population differs")
        negative = next(item for item in learned if item["arm"] == "revised_negative_gradient")
        positive = next(item for item in learned if item["arm"] == "revised_positive_gradient")
        _require(negative["radius_action"] == config["matched_random_test"]["radius_action"], "directional learned radius differs")
        empirical = empirical_burden_p(
            negative["contact_burden"], [item["contact_burden"] for item in random]
        )
        stored = next(item for item in result["state_results"] if item["state_step"] == step)
        _require(abs(stored["empirical_burden_p"] - empirical) <= 1e-15, "directional empirical value differs")
        _require(stored["random_at_least_as_good_count"] == sum(
            burden_key(item["contact_burden"]) <= burden_key(negative["contact_burden"])
            for item in random
        ), "directional random rank differs")
        _require(stored["negative_contact_burden"] == negative["contact_burden"], "directional negative burden differs")
        _require(stored["positive_contact_burden"] == positive["contact_burden"], "directional positive burden differs")
        _require(stored["negative_risk_at_radius"] < stored["positive_risk_at_radius"], "directional internal sign ordering differs")
        passed = empirical <= float(config["matched_random_test"]["empirical_p_maximum"])
        _require(stored["matched_random_gate_pass"] is passed, "directional state gate differs")
        state_passes.append(passed)
    matched_pass = bool(all(state_passes))
    mechanism_go = bool(validation_pass and matched_pass)
    decision = result["decision"]
    _require(decision["validation_directional_gate_pass"] is validation_pass, "directional validation decision differs")
    _require(decision["revised_gradient_beats_matched_random_on_both_unseen_states"] is matched_pass, "directional matched decision differs")
    _require(decision["paired_directional_supervision_mechanism_go"] is mechanism_go, "directional mechanism decision differs")
    _require(decision["closed_loop_authorized"] is False, "directional closed-loop must remain unauthorized")
    _require(result["total_fresh_rollout_count"] == len(groups["test_steps"]) * (random_count + 2), "directional rollout count differs")
    validation = {
        "schema_version": "vlsa_distal_contact_directional_ranker_e05_validation.v1",
        "status": "passed",
        "scientific_result": True,
        "expected_commit": args.expected_commit,
        "result_payload_sha256": result["result_payload_sha256"],
        "model_file_sha256": result["model"]["file_sha256"],
        "paired_directional_supervision_mechanism_go": mechanism_go,
        "closed_loop_authorized": False,
    }
    validation["validation_payload_sha256"] = hashlib.sha256(json.dumps(
        validation, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    _atomic_write(args.output.resolve(), validation)
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

