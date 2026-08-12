#!/usr/bin/env python3
"""Independent artifact validator for the post-detour live-continuation gate."""

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


def _record_gate(record: Mapping[str, Any], gate: Mapping[str, Any]) -> bool:
    return bool(
        float(record["minimum_clearance_m"])
        >= float(gate["internal_substep_clearance_buffer_m"])
        and len(record["protected_contacts"]) == int(gate["protected_raw_contact_count"])
        and float(record["maximum_active_obstacle_l1_displacement_m"])
        <= float(gate["paper_car_threshold_m"])
    )


def _check_record(record: Mapping[str, Any], *, expected_actions: int) -> None:
    _require(int(record["sample_count"]) == 1 + 25 * expected_actions, "sample count differs")
    _require(record["substep_counts"] == [25] * expected_actions, "substep counts differ")
    _require(
        float(record["maximum_boundary_equivalence_error_m"]) <= 1.0e-12,
        "boundary equivalence differs",
    )
    _require(len(record["row_minimum_clearance_m"]) == 7, "row count differs")


def validate(
    *, result_path: Path, expected_commit: str, validator_commit: str
) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    _require(
        result["schema_version"] == "vlsa_distal_post_detour_live_gate_e05_result.v1",
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
    _require(result["policy_query"]["query_index"] == 37, "policy query differs")
    _require(len(result["released_aegis_chunk"]["actions"]) == 5, "live chunk differs")
    arms = result["arms"]
    gate_config = result["config"]["gate"]
    prefix = arms["registered_compound_prefix"]
    registered = arms["registered_compound_continuation"]
    nominal = arms["fresh_pi05_released_aegis"]
    for arm in (prefix, registered, nominal):
        _check_record(arm["record"], expected_actions=5)
        _require(
            bool(arm["verification_gate"]) == _record_gate(arm["record"], gate_config),
            "arm gate differs",
        )
    field = arms["fresh_pi05_released_aegis_plus_smooth_field_if_needed"]
    field_safe = False
    if field["status"] == "evaluated":
        _require(field["optimization"] is not None and field["final"] is not None, "field evidence missing")
        _check_record(field["final"]["record"], expected_actions=5)
        _require(
            bool(field["final"]["verification_gate"])
            == _record_gate(field["final"]["record"], gate_config),
            "field final gate differs",
        )
        iterations = field["optimization"]["iterations"]
        heldout = bool(
            iterations
            and all(
                item["heldout"]["cosine"] is not None
                and float(item["heldout"]["cosine"])
                >= float(gate_config["minimum_heldout_direction_cosine"])
                and float(item["heldout"]["sign_accuracy"])
                >= float(gate_config["minimum_heldout_sign_accuracy"])
                for item in iterations
            )
        )
        _require(heldout == bool(field["heldout_direction_gate"]), "heldout gate differs")
        field_safe = bool(field["final"]["verification_gate"] and heldout)
    else:
        _require(field["status"] == "not_needed_nominal_safe", "field status differs")
        _require(bool(nominal["verification_gate"]), "unsafe nominal skipped field")
    recomputed_gate = {
        "registered_compound_prefix_verified_safe": bool(prefix["verification_gate"]),
        "registered_compound_continuation_verified_safe": bool(registered["verification_gate"]),
        "fresh_pi05_continuation_verified_safe": bool(nominal["verification_gate"]),
        "smooth_field_continuation_verified_safe": field_safe,
    }
    recomputed_gate["passed"] = bool(
        recomputed_gate["registered_compound_prefix_verified_safe"]
        and recomputed_gate["registered_compound_continuation_verified_safe"]
        and (
            recomputed_gate["fresh_pi05_continuation_verified_safe"]
            or recomputed_gate["smooth_field_continuation_verified_safe"]
        )
    )
    _require(result["gate"] == recomputed_gate, "aggregate gate differs")
    if not recomputed_gate["registered_compound_prefix_verified_safe"] or not recomputed_gate[
        "registered_compound_continuation_verified_safe"
    ]:
        interpretation = "registered_safe_compound_binding_failed"
    elif recomputed_gate["fresh_pi05_continuation_verified_safe"]:
        interpretation = "fresh_pi05_continuation_locally_safe_without_additional_field"
    elif recomputed_gate["smooth_field_continuation_verified_safe"]:
        interpretation = "smooth_field_recovers_fresh_pi05_local_continuation"
    else:
        interpretation = "post_detour_live_local_continuation_gate_no_go"
    _require(result["interpretation"] == interpretation, "interpretation differs")
    validation = {
        "schema_version": "vlsa_distal_post_detour_live_gate_e05_validation.v1",
        "status": "validated",
        "scientific_result": True,
        "producer_commit": expected_commit,
        "validator_commit": validator_commit,
        "result_file_sha256": _sha256(result_path.read_bytes()),
        "result_payload_sha256": claimed,
        "gate": recomputed_gate,
        "interpretation": interpretation,
        "margins_mm": {
            "registered_continuation": 1000.0 * float(registered["record"]["minimum_clearance_m"]),
            "fresh_pi05": 1000.0 * float(nominal["record"]["minimum_clearance_m"]),
            "smooth_field": (
                None
                if field["final"] is None
                else 1000.0 * float(field["final"]["record"]["minimum_clearance_m"])
            ),
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
