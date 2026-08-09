#!/usr/bin/env python3
"""Validate the H100 object-aware refined suffix artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.object_refined_suffix import (
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config.resolve())
    result = _load(args.result.resolve())
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("status") in ("complete", "method_failure")
        and result.get("scientific_result") is True
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("source", {}).get("dirty") is False
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256")
        and result.get("config", {}).get("config_payload_sha256")
        == config["config_payload_sha256"],
        "object-refined result identity differs",
    )
    allocation = result.get("allocation", {})
    _require(
        "H100" in str(allocation.get("device", {}).get("name"))
        and str(allocation.get("slurm_job_id", "")).isdigit(),
        "object-refined result is not H100 allocation-backed",
    )
    prefix = result.get("prefix_replay", {})
    _require(
        prefix.get("executed_action_count")
        == config["primary_case"]["suffix_start_step"]
        and prefix.get("all_next_state_hashes_match_validated_artifact") is True
        and isinstance(prefix.get("terminal_next_state_sha256"), str)
        and len(prefix["terminal_next_state_sha256"]) == 64,
        "exact prefix replay differs",
    )
    threshold = float(
        config["activation"]["intervene_if_any_eight_margin_below_m"]
    )
    maximum_motion = float(
        config["activation"]["maximum_per_step_obstacle_l1_displacement_m"]
    )
    records = result.get("actions", [])
    _require(isinstance(records, list), "suffix records differ")
    executed = [record for record in records if record.get("executed") is True]
    _require(
        [int(record["step"]) for record in executed]
        == list(range(config["primary_case"]["suffix_start_step"],
                      config["primary_case"]["suffix_start_step"] + len(executed))),
        "suffix executed steps differ",
    )
    all_safe = True
    all_clone_match = True
    selector_valid = True
    no_qp = True
    no_model = True
    for record in records:
        nominal = record.get("nominal_exact_summary", {})
        nominal_safe = bool(
            nominal.get("D_sim_raw_safe") is True
            and min(float(value) for value in nominal.get(
                "minimum_all_eight_substep_clearance_m", [-1.0]
            )) >= threshold
        )
        _require(record.get("intervention") is (not nominal_safe),
                 "suffix activation differs")
        no_qp = bool(no_qp and record.get("QP_used", False) is False)
        no_model = bool(no_model and record.get("learned_model_used", False) is False)
        if record.get("intervention"):
            search = record.get("search", {})
            selector = record.get("selector", {})
            selector_valid = bool(
                selector_valid
                and search.get("candidate_count", 0) >= 28
                and len(search.get("refinement_stages", [])) == 2
                and search.get("eligible_candidate_count")
                == selector.get("eligible_candidate_count")
                and search.get("QP_used") is False
                and search.get("learned_model_used") is False
            )
            if record.get("executed"):
                selector_valid = bool(
                    selector_valid and selector.get("valid") is True
                    and selector.get("reason")
                    == "exact_safe_object_refined_chunk_selected"
                    and record.get("selected_exact_summary", {}).get(
                        "safe_for_execution"
                    ) is True
                )
        if not record.get("executed"):
            continue
        measurement = record.get("executed_measurement", {})
        margins = measurement.get("minimum_all_eight_substep_clearance_m", [])
        safe = bool(
            len(margins) == 8 and min(float(value) for value in margins) >= 0.0
            and measurement.get("D_opt_seven_distal_safe") is True
            and measurement.get("released_AEGIS_EE_proxy_safe") is True
            and measurement.get("raw_protected_contact_count") == 0
            and float(measurement.get(
                "maximum_within_step_obstacle_l1_displacement_m", 1.0
            )) <= maximum_motion
        )
        all_safe = bool(all_safe and safe)
        all_clone_match = bool(
            all_clone_match
            and record.get("executed_next_state_matches_exact_clone") is True
            and measurement.get("next_state_sha256")
            == record.get("selected_exact_summary", {}).get(
                "first_transition_next_state_sha256"
            )
        )
    suffix = result.get("suffix", {})
    task_success = suffix.get("native_task_success") is True
    no_contact = suffix.get("first_protected_contact_step") is None
    no_car = suffix.get("first_paper_CAR_step") is None
    expected_go = bool(
        suffix.get("failure") is None and task_success and no_contact and no_car
        and all_safe and all_clone_match and selector_valid and no_qp and no_model
    )
    _require(
        result.get("combined", {}).get("primary_problem_solved") is expected_go
        and result.get("decision", {}).get(
            "privileged_object_refined_suffix_e05_go"
        ) is expected_go
        and result.get("decision", {}).get("neural_training_authorized") is False,
        "object-refined decision differs",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated", "scientific_result": False,
        "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "exact_prefix_state_hash_replay": True,
        "suffix_executed_action_count": len(executed),
        "all_suffix_substeps_all_eight_safe": all_safe,
        "all_suffix_next_states_match_clone": all_clone_match,
        "all_interventions_follow_object_refinement_contract": selector_valid,
        "native_task_success": task_success,
        "zero_protected_contact": no_contact, "zero_paper_CAR": no_car,
        "QP_used": not no_qp, "learned_model_used": not no_model,
        "privileged_object_refined_suffix_e05_go": expected_go,
        "neural_training_authorized": False,
        "deployable_method_demonstrated": False,
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
