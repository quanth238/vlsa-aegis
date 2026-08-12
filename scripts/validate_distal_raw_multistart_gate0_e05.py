#!/usr/bin/env python3
"""Independent validator for the raw-AEGIS multi-start Gate 0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _verify_record(record: dict[str, Any], result: dict[str, Any]) -> bool:
    config = result["config"]
    gate = config["gate"]
    expected = int(config["internal_verification"]["expected_mujoco_substeps_per_action"])
    _require(record["sample_count"] == 1 + 20 * expected, "internal sample count differs")
    _require(all(int(item) == expected for item in record["substep_counts"]), "substep count differs")
    _require(
        float(record["maximum_boundary_equivalence_error_m"])
        <= float(config["internal_verification"]["ordinary_env_step_boundary_equivalence_tolerance"]),
        "boundary equivalence differs",
    )
    return bool(
        float(record["minimum_clearance_m"])
        >= float(gate["internal_substep_clearance_buffer_m"])
        and len(record["protected_contacts"]) == int(gate["protected_raw_contact_count"])
        and float(record["maximum_active_obstacle_l1_displacement_m"])
        <= float(gate["paper_car_threshold_m"])
    )


def validate(path: Path, expected_commit: str, validator_commit: str) -> dict[str, Any]:
    result = json.loads(path.read_text())
    claimed = result["result_payload_sha256"]
    payload = dict(result)
    payload.pop("result_payload_sha256")
    _require(hashlib.sha256(_canonical(payload)).hexdigest() == claimed, "result hash differs")
    _require(result["schema_version"] == "vlsa_distal_raw_multistart_gate0_e05_result.v1", "schema differs")
    _require(result["source"]["commit"] == expected_commit, "source commit differs")
    _require(not result["source"]["dirty"], "source tree was dirty")
    _require(result["status"] == "complete" and result["scientific_result"], "result incomplete")
    arms = result["arms"]
    _require(
        set(arms)
        == {"raw_aegis", "full_compound_trajectory", "transplanted_compound_prefix_raw_suffix"},
        "arms differ",
    )
    raw_safe = _verify_record(arms["raw_aegis"]["record"], result)
    full_safe = _verify_record(arms["full_compound_trajectory"]["record"], result)
    transplant = arms["transplanted_compound_prefix_raw_suffix"]
    transplant_safe = _verify_record(transplant["record"], result)
    _require(bool(arms["raw_aegis"]["verification_gate"]) == raw_safe, "raw gate differs")
    _require(bool(arms["full_compound_trajectory"]["verification_gate"]) == full_safe, "full gate differs")
    _require(bool(transplant["verification_gate"]) == transplant_safe, "transplant gate differs")
    identity = transplant["continuation_identity"]
    raw_suffix_same = identity["raw_suffix_sha256"] == identity["transplanted_suffix_sha256"]
    compound_prefix_same = identity["compound_prefix_sha256"] == identity["transplanted_prefix_sha256"]
    no_clipping = int(transplant["clipped_coordinate_count"]) == 0
    expected_gate = {
        "raw_aegis_unsafe": not raw_safe,
        "full_compound_verified_safe": full_safe,
        "transplanted_prefix_verified_safe": transplant_safe,
        "no_action_bound_clipping": no_clipping,
        "raw_suffix_bitwise_identical": raw_suffix_same,
        "compound_prefix_bitwise_identical": compound_prefix_same,
    }
    expected_gate["search_authorized"] = bool(
        expected_gate["raw_aegis_unsafe"]
        and full_safe
        and transplant_safe
        and no_clipping
        and raw_suffix_same
        and compound_prefix_same
    )
    for key, value in expected_gate.items():
        _require(bool(result["gate"][key]) == value, "gate differs: %s" % key)
    expected_interpretation = (
        "raw_fixed_suffix_contains_verified_safe_positive_control_search_authorized"
        if full_safe and transplant_safe
        else "five_action_prefix_insufficient_without_adaptive_or_modified_continuation_search_blocked"
        if full_safe
        else "full_compound_failed_internal_substep_authority_binding_or_transient_audit_required"
    )
    _require(result["interpretation"] == expected_interpretation, "interpretation differs")
    return {
        "schema_version": "vlsa_distal_raw_multistart_gate0_e05_validation.v1",
        "status": "valid",
        "scientific_result": True,
        "source_result_path": str(path),
        "source_result_payload_sha256": claimed,
        "source_commit": expected_commit,
        "validator_commit": validator_commit,
        "raw_internal_margin_mm": 1000.0 * float(arms["raw_aegis"]["record"]["minimum_clearance_m"]),
        "full_compound_internal_margin_mm": 1000.0 * float(arms["full_compound_trajectory"]["record"]["minimum_clearance_m"]),
        "transplanted_prefix_internal_margin_mm": 1000.0 * float(transplant["record"]["minimum_clearance_m"]),
        "transplanted_correction_l2_action": float(transplant["applied_l2_action"]),
        "search_authorized": expected_gate["search_authorized"],
        "interpretation": result["interpretation"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = validate(args.result.resolve(), args.expected_commit, args.validator_commit)
    receipt["validation_payload_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    _write(args.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
