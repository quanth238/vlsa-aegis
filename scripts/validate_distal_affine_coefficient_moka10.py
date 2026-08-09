#!/usr/bin/env python3
"""Independently validate the learned affine-coefficient decision artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.affine_coefficient_model import (
    AFFINE_COEFFICIENT_RESULT_SCHEMA,
    AFFINE_COEFFICIENT_VALIDATION_SCHEMA,
    load_affine_coefficient_config,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_affine_coefficient_config(args.config.resolve())
    result = _load(args.result.resolve())
    _require(
        result.get("schema_version") == AFFINE_COEFFICIENT_RESULT_SCHEMA
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256")
        and result.get("config", {}).get("config_file_sha256")
        == config["config_file_sha256"],
        "affine-coefficient result identity differs",
    )
    decision = result["decision"]
    metrics = result["training"]["test_metrics"]
    authorized = bool(
        metrics["gradient_gate_pass"] and metrics["false_safe_gate_pass"]
    )
    _require(
        decision["learned_evaluation_authorized"] is authorized,
        "affine-coefficient evaluation authorization differs",
    )
    if authorized:
        states = result["state_results"]
        _require(len(states) == 15, "affine-coefficient test-state count differs")
        for state in states:
            exact = state["exact_two_step_verification"]
            expected = bool(
                state["qp"]["valid"]
                and state["qp"]["diagnostics"]["input_constraint_count"] == 7
                and exact is not None and exact["D_opt_seven_distal_safe"]
                and exact["released_AEGIS_EE_proxy_safe"]
                and exact["D_sim_raw_safe"]
                and exact["maximum_within_step_obstacle_l1_displacement_m"] <= 1.0e-4
            )
            _require(
                state["state_gate_pass"] is expected,
                "affine-coefficient state decision differs",
            )
        projection = bool(
            len(result["case_results"]) == 3
            and all(item["case_gate_pass"] for item in result["case_results"].values())
        )
    else:
        _require(
            result["state_results"] == [] and result["case_results"] == {},
            "unauthorized affine-coefficient simulation was executed",
        )
        projection = False
    _require(
        decision["test_projection_gate_pass"] is projection
        and decision["research_direction_go"] is bool(authorized and projection)
        and decision["closed_loop_e05_authorized"] is False,
        "affine-coefficient final decision differs",
    )
    output = {
        "schema_version": AFFINE_COEFFICIENT_VALIDATION_SCHEMA,
        "status": "valid", "scientific_result": False,
        "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "research_direction_go": decision["research_direction_go"],
        "closed_loop_e05_authorized": False,
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
