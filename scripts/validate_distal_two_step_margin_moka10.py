#!/usr/bin/env python3
"""Validate the paired two-step generalization and conditional E05 result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.two_step_margin import (
    TWO_STEP_RESULT_SCHEMA,
    TWO_STEP_VALIDATION_SCHEMA,
    load_two_step_config,
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
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_two_step_config(args.config.resolve())
    result = _load(args.result.resolve())
    _require(
        result.get("schema_version") == TWO_STEP_RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True,
        "two-step result schema or status differs",
    )
    _require(
        result.get("source", {}).get("commit") == args.expected_commit
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256"),
        "two-step result source or payload differs",
    )
    allocation = result.get("allocation", {})
    _require(
        "H100" in str(allocation.get("device", {}).get("name"))
        and str(allocation.get("slurm_job_id", "")).isdigit(),
        "two-step result is not H100 allocation-backed",
    )
    _require(
        result.get("config", {}).get("config_payload_sha256")
        == config["config_payload_sha256"],
        "two-step result config differs",
    )
    arms = result.get("arms", {})
    _require(set(arms) == {"global", "factorized"}, "two-step result arms differ")
    for name, arm in arms.items():
        projections = arm.get("held_out_exact_projections", {})
        _require(
            set(projections) == set(result.get("test_case_ids", [])),
            "two-step held-out projection cases differ",
        )
        projection_gate = all(
            item.get("projection_gate_pass") is True for item in projections.values()
        )
        model_gate = bool(arm.get("training", {}).get("held_out_model_gate_pass"))
        decision = arm.get("decision", {})
        _require(
            decision.get("held_out_model_gate_pass") is model_gate
            and decision.get("every_test_episode_projection_gate_pass")
            is projection_gate
            and decision.get("arm_generalization_gate_pass")
            is bool(model_gate and projection_gate),
            "two-step arm decision is inconsistent: %s" % name,
        )
        for projection in projections.values():
            if projection.get("projection_gate_pass"):
                exact = projection.get("projected_exact_summary", {})
                _require(
                    projection.get("valid_seven_row_qp") is True
                    and exact.get("D_opt_proxy_safe") is True
                    and exact.get("D_sim_raw_safe") is True,
                    "two-step passing projection lacks exact authority",
                )
    factorized_go = bool(
        arms["factorized"]["decision"]["arm_generalization_gate_pass"]
    )
    closed_loop = result.get("closed_loop_e05")
    _require(
        (closed_loop is not None) is factorized_go,
        "two-step closed-loop ordering differs",
    )
    validated = False
    if closed_loop is not None:
        _require(
            closed_loop.get("online_cloned_simulator_oracle_used") is False,
            "two-step closed loop used an online simulator oracle",
        )
        validated = bool(
            closed_loop.get("primary_problem_solved") is True
            and closed_loop.get("native_task_success") is True
            and closed_loop.get("first_protected_contact_step") is None
            and closed_loop.get("first_paper_car_step") is None
            and float(closed_loop.get("minimum_exact_substep_clearance_m")) >= 0.0
        )
    decision = result.get("decision", {})
    _require(
        decision.get("factorized_grouped_generalization_go") is factorized_go
        and decision.get("closed_loop_e05_executed") is factorized_go
        and decision.get("research_direction_validated") is validated,
        "two-step final decision is inconsistent",
    )
    output = {
        "schema_version": TWO_STEP_VALIDATION_SCHEMA,
        "status": "validated", "scientific_result": False,
        "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "factorized_grouped_generalization_go": factorized_go,
        "closed_loop_e05_executed": factorized_go,
        "research_direction_validated": validated,
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
