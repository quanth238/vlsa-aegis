#!/usr/bin/env python3
"""Validate the privileged exact-candidate closed-loop E05 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.exact_candidate_closed_loop import (
    EXACT_CANDIDATE_RESULT_SCHEMA,
    EXACT_CANDIDATE_VALIDATION_SCHEMA,
    load_exact_candidate_config,
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
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_exact_candidate_config(args.config.resolve())
    result = _load(args.result.resolve())
    _require(
        result.get("schema_version") == EXACT_CANDIDATE_RESULT_SCHEMA
        and result.get("status") in ("complete", "method_failure")
        and result.get("scientific_result") is True
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("source", {}).get("dirty") is False
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256"),
        "exact-candidate result identity differs",
    )
    _require(
        result.get("config", {}).get("config_payload_sha256")
        == config["config_payload_sha256"],
        "exact-candidate result config differs",
    )
    allocation = result.get("allocation", {})
    _require(
        "H100" in str(allocation.get("device", {}).get("name"))
        and str(allocation.get("slurm_job_id", "")).isdigit(),
        "exact-candidate result is not H100 allocation-backed",
    )
    prerequisite = result.get("prerequisite", {})
    _require(
        prerequisite.get("observed_safe_grid_candidate_at_step_186") is True
        and prerequisite.get("affine_certificate_failed_at_step_186") is True
        and prerequisite.get("result_file_sha256")
        == config["prerequisite"]["eight_row_result_file_sha256"]
        and prerequisite.get("result_payload_sha256")
        == config["prerequisite"]["eight_row_result_payload_sha256"]
        and prerequisite.get("validation_file_sha256")
        == config["prerequisite"]["eight_row_validation_file_sha256"],
        "exact-candidate prerequisite differs",
    )
    records = result.get("actions", [])
    _require(isinstance(records, list), "exact-candidate records are invalid")
    executed = [item for item in records if item.get("executed") is True]
    _require(
        [int(item["step"]) for item in executed] == list(range(len(executed))),
        "exact-candidate executed indexes differ",
    )
    maximum_step_motion = float(
        config["activation"]["maximum_per_step_obstacle_l1_displacement_m"]
    )
    minimum_candidate_margin = float(
        config["candidate_selector"]["minimum_all_eight_margin_m"]
    )
    executed_safe = True
    clone_match = True
    interventions_valid = True
    no_qp = True
    for item in records:
        nominal = item.get("nominal_exact_summary", {})
        intervention = bool(item.get("intervention"))
        _require(
            intervention is (not bool(nominal.get("safe_for_execution"))),
            "exact-candidate activation differs at step %s" % item.get("step"),
        )
        no_qp = bool(
            no_qp
            and item.get("affine_QP_used") is False
            and "qp" not in item
            and "certificate" not in item
        )
        if intervention:
            grid = item.get("grid", {})
            selector = item.get("selector", {})
            valid = bool(
                grid.get("candidate_count")
                == config["sampling"]["expected_grid_action_count"]
                and grid.get("affine_QP_used") is False
                and grid.get("eligible_candidate_count")
                == selector.get("eligible_candidate_count")
            )
            if item.get("executed") is True:
                selected = item.get("selected_exact_summary", {})
                selected_margins = selected.get(
                    "minimum_all_eight_substep_clearance_m"
                )
                valid = bool(
                    valid
                    and selector.get("valid") is True
                    and selector.get("reason") == "exact_safe_candidate_selected"
                    and isinstance(selected_margins, list)
                    and len(selected_margins) == 8
                    and min(float(value) for value in selected_margins)
                    >= minimum_candidate_margin
                    and selected.get("safe_for_execution") is True
                )
            else:
                valid = bool(
                    valid
                    and selector.get("valid") is False
                    and selector.get("eligible_candidate_count") == 0
                )
            interventions_valid = bool(interventions_valid and valid)
        if item.get("executed") is not True:
            continue
        measurement = item.get("executed_measurement", {})
        margins = measurement.get("minimum_all_eight_substep_clearance_m")
        safe = bool(
            isinstance(margins, list)
            and len(margins) == 8
            and min(float(value) for value in margins) >= 0.0
            and measurement.get("D_opt_seven_distal_safe") is True
            and measurement.get("released_AEGIS_EE_proxy_safe") is True
            and measurement.get("raw_protected_contact_count") == 0
            and float(measurement.get(
                "maximum_within_step_obstacle_l1_displacement_m", 1.0
            )) <= maximum_step_motion
        )
        executed_safe = bool(executed_safe and safe)
        clone_match = bool(
            clone_match
            and item.get("executed_next_state_matches_exact_clone") is True
            and measurement.get("next_state_sha256")
            == item.get("selected_exact_summary", {}).get(
                "first_transition_next_state_sha256"
            )
        )
    closed = result.get("closed_loop", {})
    intervention_count = sum(bool(item.get("intervention")) for item in records)
    _require(
        closed.get("action_count") == len(executed)
        and closed.get("intervention_count") == intervention_count
        and closed.get("modified_action_count")
        == sum(bool(item.get("modified")) for item in executed)
        and closed.get("affine_QP_used") is False,
        "exact-candidate summary differs",
    )
    task_success = bool(closed.get("native_task_success"))
    no_contact = closed.get("first_protected_contact_step") is None
    no_car = closed.get("first_paper_CAR_step") is None
    expected_go = bool(
        closed.get("failure") is None
        and task_success
        and no_contact
        and no_car
        and executed_safe
        and clone_match
        and interventions_valid
        and no_qp
    )
    decision = result.get("decision", {})
    _require(
        closed.get("primary_problem_solved") is expected_go
        and decision.get("privileged_exact_candidate_closed_loop_e05_go")
        is expected_go
        and decision.get("neural_training_authorized") is False
        and decision.get("deployable_method_demonstrated") is False
        and decision.get("QP_success_demonstrated") is False,
        "exact-candidate final decision differs",
    )
    output = {
        "schema_version": EXACT_CANDIDATE_VALIDATION_SCHEMA,
        "status": "validated",
        "scientific_result": False,
        "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "executed_action_count": len(executed),
        "intervention_count": intervention_count,
        "all_executed_substeps_all_eight_safe": executed_safe,
        "all_executed_next_states_match_clone": clone_match,
        "all_interventions_follow_exact_selector_contract": interventions_valid,
        "affine_QP_used": not no_qp,
        "native_task_success": task_success,
        "zero_protected_contact": no_contact,
        "zero_paper_CAR": no_car,
        "privileged_exact_candidate_closed_loop_e05_go": expected_go,
        "neural_training_authorized": False,
        "deployable_method_demonstrated": False,
        "QP_success_demonstrated": False,
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
