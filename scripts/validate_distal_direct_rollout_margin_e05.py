#!/usr/bin/env python3
"""Independent structural validator for the direct rollout-margin result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.direct_rollout_margin import (
    DIRECT_MARGIN_RESULT_SCHEMA,
    DIRECT_MARGIN_VALIDATION_SCHEMA,
    load_direct_margin_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _load,
    _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_direct_margin_config(args.config.resolve())
    result = _load(args.result.resolve())
    _require(
        result.get("schema_version") == DIRECT_MARGIN_RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256"),
        "direct-margin result identity differs",
    )
    _require(
        result.get("config", {}).get("config_payload_sha256")
        == config["config_payload_sha256"],
        "direct-margin result config differs",
    )
    allocation = result.get("allocation", {})
    _require(
        "H100" in str(allocation.get("device", {}).get("name"))
        and str(allocation.get("slurm_job_id", "")).isdigit(),
        "direct-margin result is not H100 allocation-backed",
    )
    state = result.get("state", {})
    nominal = state.get("nominal_exact_two_step", {})
    _require(
        state.get("case_id") == config["case_id"]
        and state.get("state_step") == config["state_step"]
        and float(nominal.get("rho_roll_m")) < 0.0
        and nominal.get("D_sim_raw_safe") is True,
        "direct-margin registered E05 state differs",
    )
    matched = result.get("matched_random", {})
    random_records = matched.get("random_records", [])
    learned = matched.get("learned", {})
    _require(
        len(random_records) == int(config["matched_random"]["direction_count"])
        and all(
            item.get("radius_action") == config["matched_random"]["radius_action"]
            for item in random_records
        )
        and learned.get("radius_action") == config["matched_random"]["radius_action"],
        "direct-margin matched-random records differ",
    )
    equal_or_better = sum(
        float(item["rho_improvement_m"])
        >= float(learned["rho_improvement_m"]) - 1.0e-15
        for item in random_records
    )
    empirical = (1.0 + equal_or_better) / (1.0 + len(random_records))
    random_pass = bool(
        float(learned["rho_improvement_m"]) > 0.0
        and empirical
        <= float(config["matched_random"][
            "empirical_equal_or_better_p_maximum"
        ])
    )
    _require(
        equal_or_better == matched.get("random_equal_or_better_count")
        and abs(empirical - float(matched.get("empirical_equal_or_better_p"))) <= 1.0e-15
        and matched.get("matched_random_gate_pass") is random_pass,
        "direct-margin matched-random decision differs",
    )
    projections = result.get("projections", {})
    _require(set(projections) == {"uncalibrated", "calibrated"}, "direct-margin projection arms differ")
    projection_passes = {}
    for name, record in projections.items():
        projection = record.get("projection", {})
        qps = [
            item.get("qp") for item in projection.get("iterations", [])
            if item.get("qp") is not None
        ]
        seven_row = bool(qps and all(
            item.get("valid") is True and len(item.get("lower", [])) == 7
            for item in qps
        ))
        exact = record.get("fresh_exact_two_step")
        passed = bool(
            projection.get("valid") is True and seven_row and exact is not None
            and exact.get("D_opt_proxy_safe") is True
            and exact.get("D_sim_raw_safe") is True
            and float(exact.get("rho_roll_m")) >= 0.0
            and int(exact.get("raw_protected_contact_count")) == 0
            and float(exact.get("maximum_within_step_obstacle_l1_displacement_m")) <= 1.0e-4
        )
        _require(
            record.get("valid_seven_row_qp") is seven_row
            and record.get("projection_gate_pass") is passed,
            "direct-margin projection decision differs: %s" % name,
        )
        projection_passes[name] = passed
    training = result.get("training", {}).get("audit", {})
    model_gate = bool(training.get("e05_model_gate_pass"))
    false_safe_gate = bool(training.get("test_false_safe_gate_pass"))
    research_go = bool(
        model_gate and false_safe_gate and random_pass
        and projection_passes["calibrated"]
    )
    decision = result.get("decision", {})
    _require(
        decision.get("e05_model_gate_pass") is model_gate
        and decision.get("all_test_false_safe_gate_pass") is false_safe_gate
        and decision.get("learned_gradient_beats_matched_random") is random_pass
        and decision.get("calibrated_seven_row_qp_exact_gate_pass")
        is projection_passes["calibrated"]
        and decision.get("research_direction_go") is research_go
        and decision.get("closed_loop_authorized") is False,
        "direct-margin final decision differs",
    )
    output = {
        "schema_version": DIRECT_MARGIN_VALIDATION_SCHEMA,
        "status": "validated", "scientific_result": False,
        "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "e05_model_gate_pass": model_gate,
        "all_test_false_safe_gate_pass": false_safe_gate,
        "learned_gradient_beats_matched_random": random_pass,
        "calibrated_seven_row_qp_exact_gate_pass": projection_passes["calibrated"],
        "research_direction_go": research_go,
        "closed_loop_authorized": False,
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
