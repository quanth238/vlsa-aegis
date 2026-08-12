#!/usr/bin/env python3
"""Independent validator for the direct E05 smooth-field attribution gate."""

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


def _verify_internal(record: dict[str, Any], result: dict[str, Any]) -> bool:
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
        and not record["protected_contacts"]
        and float(record["maximum_active_obstacle_l1_displacement_m"])
        <= float(gate["paper_car_threshold_m"])
    )


def validate(path: Path, expected_commit: str, validator_commit: str) -> dict[str, Any]:
    result = json.loads(path.read_text())
    claimed = result["result_payload_sha256"]
    payload = dict(result)
    payload.pop("result_payload_sha256")
    _require(hashlib.sha256(_canonical(payload)).hexdigest() == claimed, "result hash differs")
    _require(result["source"]["commit"] == expected_commit, "source commit differs")
    _require(not result["source"]["dirty"], "source tree was dirty")
    _require(result["status"] == "complete" and result["scientific_result"], "result incomplete")
    arms = result["arms"]
    _require(
        set(arms)
        == {
            "raw_aegis",
            "earlier_five_action_detour",
            "detour_plus_smooth_compound",
            "direct_raw_aegis_smooth_field",
        },
        "attribution arms differ",
    )
    raw_safe = _verify_internal(arms["raw_aegis"]["record"], result)
    detour_safe = _verify_internal(arms["earlier_five_action_detour"]["record"], result)
    compound_safe = _verify_internal(arms["detour_plus_smooth_compound"]["record"], result)
    _require(raw_safe == bool(arms["raw_aegis"]["verification_gate"]), "raw gate differs")
    _require(
        detour_safe == bool(arms["earlier_five_action_detour"]["verification_gate"]),
        "detour gate differs",
    )
    _require(
        compound_safe == bool(arms["detour_plus_smooth_compound"]["verification_gate"]),
        "compound gate differs",
    )
    direct = arms["direct_raw_aegis_smooth_field"]
    smallest = direct["scale_search_on_discovered_ray"]["smallest_verified"]
    direct_safe = smallest is not None
    if smallest is not None:
        _require(_verify_internal(smallest["record"], result), "selected direct scale differs")
        smaller = [
            item
            for item in direct["scale_search_on_discovered_ray"]["grid"]
            if float(item["scale"]) < float(smallest["scale"])
        ]
        _require(not any(item["verification_gate"] for item in smaller), "selected scale is not smallest")
    heldout = bool(
        direct["optimization"]["iterations"]
        and all(
            item["heldout"]["cosine"] is not None
            and float(item["heldout"]["cosine"])
            >= float(result["config"]["gate"]["minimum_heldout_direction_cosine"])
            and float(item["heldout"]["sign_accuracy"])
            >= float(result["config"]["gate"]["minimum_heldout_sign_accuracy"])
            for item in direct["optimization"]["iterations"]
        )
    )
    gate = result["gate"]
    expected_gate = {
        "raw_aegis_unsafe": not raw_safe,
        "direct_smooth_verified_safe": direct_safe,
        "compound_comparator_verified_safe_diagnostic": compound_safe,
        "heldout_direction_gate": heldout,
        "internal_substep_count_exact": True,
    }
    for key, value in expected_gate.items():
        _require(bool(gate[key]) == value, "gate differs: %s" % key)
    passed = bool(not raw_safe and direct_safe and heldout)
    _require(bool(gate["passed"]) == passed, "aggregate gate differs")
    receipt = {
        "schema_version": "vlsa_distal_smooth_field_attribution_e05_validation.v1",
        "status": "valid",
        "scientific_result": True,
        "source_result_path": str(path),
        "source_result_payload_sha256": claimed,
        "source_commit": expected_commit,
        "validator_commit": validator_commit,
        "raw_aegis_internal_margin_mm": 1000.0
        * float(arms["raw_aegis"]["record"]["minimum_clearance_m"]),
        "detour_internal_margin_mm": 1000.0
        * float(arms["earlier_five_action_detour"]["record"]["minimum_clearance_m"]),
        "compound_internal_margin_mm": 1000.0
        * float(arms["detour_plus_smooth_compound"]["record"]["minimum_clearance_m"]),
        "direct_internal_margin_mm": None
        if smallest is None
        else 1000.0 * float(smallest["record"]["minimum_clearance_m"]),
        "direct_smallest_registered_scale": None if smallest is None else float(smallest["scale"]),
        "gate_passed": passed,
        "interpretation": result["interpretation"],
    }
    return receipt


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

