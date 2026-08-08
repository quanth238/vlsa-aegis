#!/usr/bin/env python3
"""Validate the preregistered one-state 8 mm clearance-margin test."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.oracle_affine import (
    MARGIN8_AFFINE_RESULT_SCHEMA,
    load_oracle_affine_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write, _load


VALIDATION_SCHEMA = "vlsa_distal_oracle_affine_margin8mm_e05_validation.v1"
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
        result.get("schema_version") == MARGIN8_AFFINE_RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and result.get("case_id") == CASE_ID
        and result.get("failure") is None,
        "completed 8 mm margin result identity differs",
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
    _require(result.get("config") == expected_config, "margin config differs")
    _require(
        result.get("obstacle_primitive_config") is None,
        "margin test must retain the frozen released AEGIS obstacle proxy",
    )
    geometry = result.get("geometry", {})
    _require(
        geometry.get("distal_ellipsoid_count") == 7
        and geometry.get("total_constraint_geometry_count") == 8,
        "registered robot/EE ellipsoids differ",
    )
    state = result.get("audit_state", {})
    _require(
        state.get("step") == 192
        and state.get("immutable_prefix_action_count") == 192
        and len(state.get("immutable_prefix_action_sha256", [])) == 192,
        "audit state binding differs",
    )
    audit = result.get("oracle_affine_audit", {})
    candidates = audit.get("candidate_records")
    nominal = audit.get("nominal_substep_transition", {})
    _require(
        isinstance(candidates, list)
        and audit.get("candidate_count") == len(candidates) == 87
        and nominal.get("captured_state_count")
        == nominal.get("expected_internal_mujoco_step_count") + 1
        and nominal.get("raw_protected_contact_count", 0) > 0,
        "candidate or nominal substep ledger differs",
    )
    nominal_clearance = nominal.get("minimum_substep_clearance_m")
    _require(
        isinstance(nominal_clearance, list)
        and len(nominal_clearance) == 8
        and any(float(value) < 0.008 for value in nominal_clearance),
        "8 mm margin did not declare the nominal transition unsafe",
    )
    affine = audit.get("affine_model", {})
    _require(
        affine.get("fit_candidate_count") == 60
        and len(affine.get("gradients_m_per_action", [])) == 8
        and all(len(row) == 3 for row in affine["gradients_m_per_action"]),
        "oracle affine dimensions differ",
    )
    decision = audit.get("decision", {})
    intervention = decision.get("margin_intervention", {})
    qp = audit.get("oracle_affine_qp", {})
    exact = qp.get("exact_substep_transition")
    qp_raw_safe = bool(
        exact is not None
        and exact.get("raw_protected_contact_count") == 0
        and float(exact.get("maximum_within_step_obstacle_l1_displacement_m", 1.0))
        <= 0.0001
    )
    qp_margin_safe = bool(
        exact is not None
        and len(exact.get("minimum_substep_clearance_m", [])) == 8
        and all(
            float(value) >= 0.008
            for value in exact["minimum_substep_clearance_m"]
        )
    )
    expected_pass = bool(qp.get("valid") and qp_raw_safe and qp_margin_safe)
    _require(
        intervention.get("formula")
        == "h_corrected_equals_h_aegis_minus_0.008_m"
        and intervention.get("clearance_margin_m") == 0.008
        and intervention.get("nominal_declared_unsafe") is True
        and intervention.get("qp_action_raw_safe") is qp_raw_safe
        and intervention.get("qp_action_margin_safe") is qp_margin_safe
        and intervention.get("test_pass") is expected_pass
        and result.get("margin8mm_test_pass") is expected_pass,
        "8 mm intervention decision is inconsistent",
    )
    if qp.get("valid"):
        _require(
            exact.get("captured_state_count")
            == exact.get("expected_internal_mujoco_step_count") + 1,
            "valid margin QP lacks exact internal-substep verification",
        )
    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": "valid",
        "scientific_result": False,
        "case_id": CASE_ID,
        "source_commit": expected_commit,
        "slurm_job_id": allocation["slurm_job_id"],
        "result_payload_sha256": recorded_payload_sha256,
        "margin8mm_test_pass": expected_pass,
        "qp_valid": bool(qp.get("valid")),
        "qp_exact_raw_safe": qp_raw_safe,
        "qp_exact_margin_safe": qp_margin_safe,
        "checks": {
            "payload_identity": True,
            "clean_h100_source": True,
            "immutable_state_and_action_binding": True,
            "current_ellipsoids_and_frozen_aegis_obstacle_retained": True,
            "fixed_8mm_formula": True,
            "complete_exact_substep_ledger": True,
            "outcome_consistency": True,
        },
    }


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    expected_config = load_oracle_affine_config(args.config.resolve())
    record = validate(
        _load(args.result.resolve()), args.expected_commit, expected_config
    )
    _atomic_write(args.output.resolve(), record)
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
