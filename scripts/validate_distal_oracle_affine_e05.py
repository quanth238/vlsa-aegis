#!/usr/bin/env python3
"""Structurally validate one completed E05 oracle-affine result."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write, _load


RESULT_SCHEMA = "vlsa_distal_oracle_affine_e05_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_oracle_affine_e05_validation.v1"
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


def validate(result: Mapping[str, Any], expected_commit: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(result))
    recorded_payload_sha256 = payload.pop("result_payload_sha256", None)
    computed_payload_sha256 = hashlib.sha256(_canonical(payload)).hexdigest()
    _require(
        recorded_payload_sha256 == computed_payload_sha256,
        "result payload SHA-256 differs",
    )
    _require(
        result.get("schema_version") == RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and result.get("case_id") == CASE_ID
        and result.get("failure") is None,
        "completed result identity differs",
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
    source = result.get("false_safe_source", {})
    _require(
        source.get("file_sha256")
        == "d79a28585e74cece727fbc3d3e6a72eb1647a782cae4962ec3af9045f448e603"
        and source.get("payload_sha256")
        == "2a4ecd5ff79a363141e0927d1dfbcfb26248ec9bb5e2bc123cc8f879c3395a53"
        and source.get("slurm_job_id") == "37109"
        and source.get("read_only") is True,
        "false-safe source identity differs",
    )
    source_clearance = source.get("audit_step_endpoint_clearance_m")
    _require(
        isinstance(source_clearance, list)
        and len(source_clearance) == 8
        and all(float(item) > 0.0 for item in source_clearance),
        "false-safe source does not have eight positive endpoint clearances",
    )
    source_contacts = source.get("audit_step_protected_contact_events")
    _require(
        isinstance(source_contacts, list)
        and any(
            event.get("other", {}).get("geom_name")
            == "robot0_link6_collision"
            and float(event.get("distance", 1.0)) <= 0.0
            for event in source_contacts
        ),
        "false-safe source does not contain direct L6 contact",
    )
    audit_state = result.get("audit_state", {})
    _require(
        audit_state.get("step") == 192
        and audit_state.get("immutable_prefix_action_count") == 192
        and len(audit_state.get("immutable_prefix_action_sha256", [])) == 192
        and audit_state.get("immutable_prefix_source")
        == "job_37109_executed_sitl_action_ledger",
        "audit state binding differs",
    )
    audit = result.get("oracle_affine_audit", {})
    candidates = audit.get("candidate_records")
    _require(
        isinstance(candidates, list)
        and audit.get("candidate_count") == len(candidates)
        and len(candidates) >= 60,
        "candidate ledger is incomplete",
    )
    nominal = audit.get("nominal_substep_transition", {})
    _require(
        nominal.get("captured_state_count")
        == nominal.get("expected_internal_mujoco_step_count") + 1
        and len(nominal.get("minimum_substep_clearance_m", [])) == 8
        and len(nominal.get("endpoint_clearance_m", [])) == 8,
        "nominal internal-substep trace is incomplete",
    )
    affine = audit.get("affine_model", {})
    _require(
        len(affine.get("gradients_m_per_action", [])) == 8
        and all(len(row) == 3 for row in affine["gradients_m_per_action"])
        and len(affine.get("one_sided_overprediction_bound_m", [])) == 8,
        "affine model dimensions differ",
    )
    decision = audit.get("decision", {})
    expected_go = bool(
        decision.get("geometry_contact_witness_pass")
        and decision.get("local_jointly_raw_and_proxy_safe_candidate_exists")
        and decision.get("affine_candidate_gate_pass")
        and decision.get("qp_valid")
        and decision.get("qp_exact_raw_safe")
        and decision.get("qp_exact_proxy_safe")
    )
    _require(
        decision.get("research_direction_go") is expected_go
        and result.get("research_direction_go") is expected_go
        and result.get("stop_reason") == decision.get("stop_reason")
        and ((decision.get("stop_reason") is None) is expected_go),
        "decision ledger is internally inconsistent",
    )
    qp = audit.get("oracle_affine_qp", {})
    _require(
        qp.get("valid") is decision.get("qp_valid")
        and len(qp.get("lower", [])) == 8
        and len(qp.get("action_lower", [])) == 3
        and len(qp.get("action_upper", [])) == 3,
        "QP ledger is internally inconsistent",
    )
    if qp.get("valid"):
        exact = qp.get("exact_substep_transition", {})
        _require(
            exact.get("captured_state_count")
            == exact.get("expected_internal_mujoco_step_count") + 1,
            "valid QP lacks an exact internal-substep trace",
        )
    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": "valid",
        "scientific_result": False,
        "case_id": CASE_ID,
        "source_commit": expected_commit,
        "slurm_job_id": allocation["slurm_job_id"],
        "result_payload_sha256": recorded_payload_sha256,
        "research_direction_go": expected_go,
        "stop_reason": decision.get("stop_reason"),
        "checks": {
            "payload_identity": True,
            "clean_h100_source": True,
            "immutable_false_safe_binding": True,
            "complete_substep_trace": True,
            "affine_dimensions": True,
            "decision_consistency": True,
        },
    }


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = validate(_load(args.result.resolve()), args.expected_commit)
    _atomic_write(args.output.resolve(), record)
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
