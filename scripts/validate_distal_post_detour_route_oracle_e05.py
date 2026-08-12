#!/usr/bin/env python3
"""Independent validator for the action-187 persistent-route oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _record_gate(candidate: Mapping[str, Any], gate: Mapping[str, Any]) -> bool:
    record = candidate["record"]
    _require(int(record["sample_count"]) == 126, "internal sample count differs")
    _require(record["substep_counts"] == [25] * 5, "substep counts differ")
    _require(
        float(record["maximum_boundary_equivalence_error_m"]) <= 1.0e-12,
        "boundary equivalence differs",
    )
    _require(len(record["row_minimum_clearance_m"]) == 7, "row count differs")
    return bool(
        float(record["minimum_clearance_m"])
        >= float(gate["internal_substep_clearance_buffer_m"])
        and len(record["protected_contacts"]) == int(gate["protected_raw_contact_count"])
        and float(record["maximum_active_obstacle_l1_displacement_m"])
        <= float(gate["paper_car_threshold_m"])
    )


def validate(
    *, result_path: Path, expected_commit: str, validator_commit: str
) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    _require(
        result["schema_version"] == "vlsa_distal_post_detour_route_oracle_e05_result.v1",
        "result schema differs",
    )
    payload = dict(result)
    claimed = payload.pop("result_payload_sha256")
    computed = _sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    _require(claimed == computed, "result self-hash differs")
    _require(result["source"]["commit"] == expected_commit, "producer commit differs")
    _require(result["status"] == "complete" and result["scientific_result"] is True, "result incomplete")
    _require(result["case_id"] == "vlsa-t1-goal-ii-t0-e05", "case differs")
    _require("H100" in result["allocation"]["device"]["name"], "producer was not H100")
    _require(result["post_prefix_state"]["step"] == 187, "post-prefix step differs")
    gate_config = result["config"]["gate"]
    arms = result["arms"]
    nominal = arms["fixed_live_nominal"]
    positive = arms["registered_positive_control"]
    nominal_safe = _record_gate(nominal, gate_config)
    positive_safe = _record_gate(positive, gate_config)
    _require(bool(nominal["verification_gate"]) == nominal_safe, "nominal gate differs")
    _require(bool(positive["verification_gate"]) == positive_safe, "positive gate differs")
    modes = result["config"]["route_search"]["modes"]
    norms = result["config"]["route_search"]["coarse_norms_action"]
    persistent_safe = 0
    margins: dict[str, list[float]] = {}
    for mode in modes:
        records = arms["persistent_routes"][mode]
        _require(len(records) == len(norms), "route scale count differs")
        margins[mode] = []
        for item in records:
            safe = _record_gate(item, gate_config)
            _require(bool(item["verification_gate"]) == safe, "persistent route gate differs")
            persistent_safe += int(safe)
            margins[mode].append(1000.0 * float(item["record"]["minimum_clearance_m"]))
    refined = arms["best_route_plus_smooth_refinement"]["final"]
    refined_exact_safe = _record_gate(refined, gate_config)
    _require(bool(refined["verification_gate"]) == refined_exact_safe, "refined exact gate differs")
    refined_safe = bool(
        refined_exact_safe
        and refined["heldout_direction_gate"]
        and refined["within_total_radius"]
    )
    derivative = arms["full_action_derivative_free"]
    expected_candidates = (
        int(result["config"]["derivative_free"]["candidate_count_per_generation"])
        * int(result["config"]["derivative_free"]["generations"])
    )
    _require(int(derivative["candidate_count"]) == expected_candidates, "derivative-free budget differs")
    derivative_safe = _record_gate(derivative["best"], gate_config)
    _require(bool(derivative["best"]["verification_gate"]) == derivative_safe, "derivative-free gate differs")
    recomputed = {
        "nominal_unsafe": not nominal_safe,
        "positive_control_verified_safe": positive_safe,
        "persistent_route_safe_count": persistent_safe,
        "route_plus_smooth_verified_safe": refined_safe,
        "derivative_free_verified_safe": derivative_safe,
        "passed": bool(persistent_safe or refined_safe),
    }
    _require(result["gate"] == recomputed, "aggregate gate differs")
    if recomputed["passed"]:
        interpretation = "generic_persistent_route_oracle_rediscovered_safe_action187_continuation"
    elif derivative_safe:
        interpretation = "full_action_search_succeeds_but_generic_persistent_route_family_fails"
    else:
        interpretation = "action187_route_oracle_no_go_within_registered_family_and_budget"
    _require(result["interpretation"] == interpretation, "interpretation differs")
    validation = {
        "schema_version": "vlsa_distal_post_detour_route_oracle_e05_validation.v1",
        "status": "validated",
        "scientific_result": True,
        "producer_commit": expected_commit,
        "validator_commit": validator_commit,
        "result_file_sha256": _sha256(result_path.read_bytes()),
        "result_payload_sha256": claimed,
        "gate": recomputed,
        "interpretation": interpretation,
        "margins_mm": {
            "nominal": 1000.0 * float(nominal["record"]["minimum_clearance_m"]),
            "positive_control": 1000.0 * float(positive["record"]["minimum_clearance_m"]),
            "persistent_routes": margins,
            "refined": 1000.0 * float(refined["record"]["minimum_clearance_m"]),
            "derivative_free": 1000.0 * float(derivative["best"]["record"]["minimum_clearance_m"]),
        },
    }
    validation["validation_payload_sha256"] = _sha256(
        json.dumps(validation, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    return validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(
        result_path=args.result.resolve(),
        expected_commit=args.expected_commit,
        validator_commit=args.validator_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
