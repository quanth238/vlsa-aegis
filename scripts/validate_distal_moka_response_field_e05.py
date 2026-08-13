#!/usr/bin/env python3
"""Independently validate the immutable E05 Moka response-field result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import _load, _require, _sha256


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate(result_path: Path, expected_commit: str) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.counterfactual_field import matched_random_p_value

    result = _load(result_path)
    _require(result.get("schema_version") == "vlsa_distal_moka_response_field_e05_result.v1", "Moka result schema differs")
    _require(result.get("status") == "complete" and result.get("scientific_result") is True, "Moka result is incomplete")
    _require(result.get("source", {}).get("commit") == expected_commit, "Moka producer commit differs")
    payload = dict(result)
    claimed = payload.pop("result_payload_sha256")
    recomputed = _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    _require(claimed == recomputed, "Moka result payload hash differs")
    _require(result["archived_table1"]["read_only"] is True, "Table-1 was not read-only")
    _require(result["compiled_moka_box_count"] == 15, "Moka compiled primitive count differs")
    _require(result["model"]["frozen_before_test"] is True, "Moka model was not frozen before test")
    _require(result["corrected_execution_attempted"] is False, "Moka gate improperly executed correction")
    config = result["config"]
    passed = 0
    recomputed_states = []
    for state in result["test"]:
        directional = state["heldout_directional_metrics"]
        radii = []
        for radius in state["radii"]:
            random_p = float(radius["matched_random_p_value"])
            _require(0.0 < random_p <= 1.0, "Moka random p-value differs")
            learned_gain = float(radius["learned_gain_m"])
            fixed_gain = float(radius["fixed_gain_m"])
            radius_pass = bool(
                learned_gain >= float(config["gate"]["minimum_exact_gain_m"])
                and learned_gain - fixed_gain >= float(config["gate"]["minimum_learned_fixed_gain_m"])
                and random_p <= float(config["gate"]["maximum_matched_random_p_value"])
                and (
                    not config["gate"]["require_zero_new_raw_protected_contacts"]
                    or int(radius["learned"]["protected_contact_count"])
                    <= next(
                        int(record["base"]["protected_contact_count"])
                        for record in result["state_records"]
                        if int(record["step"]) == int(state["step"])
                    )
                )
            )
            _require(radius_pass == bool(radius["gate_pass"]), "Moka radius gate differs")
            radii.append(radius_pass)
        state_pass = bool(
            float(state["learned_exact_direction_cosine"])
            >= float(config["gate"]["minimum_learned_exact_oracle_cosine"])
            and float(directional["sign_accuracy"])
            >= float(config["gate"]["minimum_directional_sign_accuracy"])
            and any(radii)
        )
        _require(state_pass == bool(state["state_gate_pass"]), "Moka state gate differs")
        passed += int(state_pass)
        recomputed_states.append({"step": int(state["step"]), "passed": state_pass})
    gate = bool(passed >= int(config["gate"]["minimum_test_state_pass_count"]))
    _require(gate == bool(result["learning_gate_pass"]), "Moka aggregate gate differs")
    expected = (
        "one_task_moka_controller_conditioned_response_supported"
        if gate
        else "one_task_moka_controller_conditioned_response_strict_no_go"
    )
    _require(result["interpretation"] == expected, "Moka interpretation differs")
    return {
        "schema_version": "vlsa_distal_moka_response_field_e05_validation.v1",
        "status": "validated",
        "result_payload_sha256": claimed,
        "expected_producer_commit": expected_commit,
        "test_states": recomputed_states,
        "test_state_pass_count": passed,
        "learning_gate_pass": gate,
        "interpretation": expected,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = validate(args.result.resolve(), args.expected_producer_commit)
    receipt["validation_payload_sha256"] = _sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    _atomic_write(args.output.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
