#!/usr/bin/env python3
"""Validate the exact-box 8 mm first-crossing heuristic result."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.inflated_trigger import (
    INFLATED_TRIGGER_RESULT_SCHEMA,
    intervention_pass,
    is_first_crossing_candidate,
    load_inflated_trigger_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write, _load


VALIDATION_SCHEMA = "vlsa_distal_exact_box_inflated_trigger_e05_validation.v1"
CASE_ID = "vlsa-t1-goal-ii-t0-e05"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(
    result: Mapping[str, Any],
    expected_commit: str,
    expected_config: Mapping[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(result))
    recorded_payload_sha256 = payload.pop("result_payload_sha256", None)
    _require(
        recorded_payload_sha256 == hashlib.sha256(_canonical(payload)).hexdigest(),
        "result payload SHA-256 differs",
    )
    _require(
        result.get("schema_version") == INFLATED_TRIGGER_RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and result.get("case_id") == CASE_ID
        and result.get("failure") is None,
        "completed inflated-trigger result identity differs",
    )
    _require(
        result.get("source", {}).get("commit") == expected_commit
        and result.get("source", {}).get("dirty") is False,
        "result source identity differs",
    )
    allocation = result.get("allocation", {})
    _require(
        str(allocation.get("slurm_job_id", "")).isdigit()
        and "H100" in str(allocation.get("device", {}).get("name", "")),
        "result lacks an H100 Slurm allocation",
    )
    _require(result.get("trigger_config") == expected_config, "trigger config differs")
    for name, expected in expected_config["prior_results"].items():
        source = result.get("prior_results", {}).get(name, {})
        _require(
            all(source.get(key) == value for key, value in expected.items())
            and source.get("read_only") is True,
            "prior result binding differs: %s" % name,
        )
    shell = float(expected_config["rounded_box_shell"]["shell_m"])
    geometry = result.get("geometry", {})
    robot = geometry.get("accepted_robot_and_released_ee", {})
    obstacle = geometry.get("exact_obstacle_union", {})
    _require(
        shell == 0.008
        and geometry.get("rounded_box_shell_m") == shell
        and robot.get("distal_ellipsoid_count") == 7
        and robot.get("total_constraint_geometry_count") == 8
        and obstacle.get("primitive_count") == 15
        and obstacle.get("exact_box_count") == 15
        and obstacle.get("geometric_inflation") == 0.0,
        "inflated exact-box geometry differs",
    )
    scan = result.get("trigger_scan", {})
    records = scan.get("records")
    selected_step = scan.get("selected_step")
    _require(
        isinstance(records, list)
        and records
        and scan.get("scanned_step_count") == len(records)
        and scan.get("immutable_prefix_action_count") == selected_step
        and len(scan.get("immutable_prefix_action_sha256", [])) == selected_step
        and [item.get("step") for item in records] == list(range(len(records)))
        and records[-1].get("step") == selected_step,
        "first-crossing scan ledger differs",
    )
    _require(
        not any(item.get("selected_first_crossing") for item in records[:-1])
        and records[-1].get("selected_first_crossing") is True
        and is_first_crossing_candidate(
            records[-1]["current_exact_clearance_m"],
            records[-1]["nominal_minimum_exact_clearance_m"],
            shell,
        ),
        "selected record is not the first registered crossing",
    )
    for record in records:
        current = record.get("current_exact_clearance_m", [])
        inflated_current = record.get("current_inflated_clearance_m", [])
        nominal = record.get("nominal_minimum_exact_clearance_m", [])
        inflated_nominal = record.get("nominal_minimum_inflated_clearance_m", [])
        _require(
            len(current) == len(inflated_current) == len(nominal) == len(inflated_nominal) == 8
            and all(abs((float(a) - shell) - float(b)) <= 1.0e-15 for a, b in zip(current, inflated_current))
            and all(abs((float(a) - shell) - float(b)) <= 1.0e-15 for a, b in zip(nominal, inflated_nominal)),
            "rounded-shell clearance arithmetic differs",
        )
    effective = result.get("effective_audit_settings", {})
    _require(
        effective.get("affine_model", {}).get("clearance_target_m") == shell,
        "effective QP target differs from rounded shell",
    )
    audit = result.get("oracle_affine_audit", {})
    candidates = audit.get("candidate_records")
    affine = audit.get("affine_model", {})
    qp = audit.get("oracle_affine_qp", {})
    _require(
        isinstance(candidates, list)
        and audit.get("candidate_count") == len(candidates) == 87
        and affine.get("fit_candidate_count") == 60
        and len(affine.get("gradients_m_per_action", [])) == 8
        and qp.get("valid") is audit.get("decision", {}).get("qp_valid"),
        "first-crossing affine/QP ledger differs",
    )
    expected_pass = intervention_pass(audit)
    _require(
        result.get("inflated_first_crossing_test_pass") is expected_pass,
        "inflated first-crossing decision differs",
    )
    decision = audit["decision"]
    if not decision["local_jointly_raw_and_proxy_safe_candidate_exists"]:
        expected_stop = "no_jointly_raw_and_inflated_proxy_safe_candidate_at_first_crossing"
    elif not decision["affine_candidate_gate_pass"]:
        expected_stop = "inflated_clearance_map_not_conservatively_affine_on_candidates"
    elif not decision["qp_valid"]:
        expected_stop = "inflated_first_crossing_qp_infeasible"
    elif not (decision["qp_exact_raw_safe"] and decision["qp_exact_proxy_safe"]):
        expected_stop = "inflated_first_crossing_qp_failed_exact_verification"
    else:
        expected_stop = None
    _require(result.get("stop_reason") == expected_stop, "stop reason differs")
    if qp.get("valid"):
        exact = qp.get("exact_substep_transition", {})
        _require(
            exact.get("captured_state_count")
            == exact.get("expected_internal_mujoco_step_count") + 1,
            "valid QP lacks exact substep verification",
        )
    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": "valid",
        "scientific_result": False,
        "case_id": CASE_ID,
        "source_commit": expected_commit,
        "slurm_job_id": allocation["slurm_job_id"],
        "result_payload_sha256": recorded_payload_sha256,
        "selected_step": selected_step,
        "inflated_first_crossing_test_pass": expected_pass,
        "stop_reason": expected_stop,
        "checks": {
            "payload_identity": True,
            "clean_h100_source": True,
            "immutable_prior_and_action_bindings": True,
            "exact_zero_inflation_boxes_plus_8mm_rounded_shell": True,
            "deterministic_first_crossing_selection": True,
            "complete_affine_qp_and_exact_transition_ledger": True,
            "decision_consistency": True,
        },
    }


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_inflated_trigger_config(args.config.resolve())
    record = validate(_load(args.result.resolve()), args.expected_commit, config)
    _atomic_write(args.output.resolve(), record)
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
