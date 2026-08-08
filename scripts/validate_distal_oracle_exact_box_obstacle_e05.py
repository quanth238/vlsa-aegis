#!/usr/bin/env python3
"""Validate one completed exact MuJoCo obstacle-box E05 oracle result."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.obstacle_primitives import (
    EXACT_BOX_OBSTACLE_RESULT_SCHEMA,
    load_obstacle_primitive_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write, _load


VALIDATION_SCHEMA = "vlsa_distal_oracle_exact_box_obstacle_e05_validation.v1"
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
        result.get("schema_version") == EXACT_BOX_OBSTACLE_RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and result.get("case_id") == CASE_ID
        and result.get("failure") is None,
        "completed exact-box result identity differs",
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
    _require(
        result.get("obstacle_primitive_config") == expected_config,
        "exact-box obstacle config differs",
    )
    discovery = result.get("obstacle_discovery_source", {})
    expected_discovery = expected_config["discovery_source"]
    _require(
        discovery.get("file_sha256") == expected_discovery["file_sha256"]
        and discovery.get("result_payload_sha256")
        == expected_discovery["result_payload_sha256"]
        and discovery.get("slurm_job_id") == expected_discovery["slurm_job_id"]
        and discovery.get("obstacle_geom_count") == 15
        and discovery.get("obstacle_geom_names")
        == expected_discovery["obstacle_geom_names"]
        and discovery.get("obstacle_geom_type") == "box"
        and discovery.get("read_only") is True,
        "exact-box discovery binding differs",
    )
    geometry = result.get("geometry", {})
    robot = geometry.get("accepted_robot_and_released_ee", {})
    union = geometry.get("conservative_obstacle_union", {})
    primitives = union.get("primitives")
    _require(
        robot.get("distal_ellipsoid_count") == 7
        and robot.get("total_constraint_geometry_count") == 8,
        "accepted robot/EE geometry differs",
    )
    _require(
        isinstance(primitives, list)
        and len(primitives) == union.get("primitive_count") == 15
        and union.get("exact_box_count") == 15
        and union.get("geometric_inflation") == 0.0
        and union.get("all_collision_geoms_represented_exactly_once") is True
        and union.get("all_enclosure_certificates_verified") is True
        and [item.get("geom_name") for item in primitives]
        == expected_discovery["obstacle_geom_names"],
        "exact obstacle-box union summary differs",
    )
    _require(
        all(
            item.get("source_geom_kind") == "box"
            and item.get("bound_source") == "exact_compiled_mujoco_oriented_box"
            and len(item.get("half_size_m", [])) == 3
            and item.get("enclosure_certificate", {}).get("verified") is True
            and item.get("enclosure_certificate", {}).get("proof")
            == "exact_compiled_mujoco_box_identity"
            and item.get("enclosure_certificate", {}).get("geometric_inflation")
            == 0.0
            for item in primitives
        ),
        "exact obstacle-box certificates differ",
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
        and nominal.get("obstacle_proxy", {}).get("mode")
        == "live_exact_oriented_box_union"
        and nominal.get("obstacle_proxy", {}).get("primitive_count") == 15
        and nominal.get("captured_state_count")
        == nominal.get("expected_internal_mujoco_step_count") + 1,
        "exact-box candidate or substep ledger differs",
    )
    contacts = nominal.get("raw_protected_contact_events")
    _require(
        isinstance(contacts, list)
        and contacts
        and all(
            "minimum_obstacle_containment_value" in item
            and "source_obstacle_containment_value" in item
            and "source_obstacle_contact_point_covered" in item
            for item in contacts
        ),
        "exact-box contact witness ledger is incomplete",
    )
    affine = audit.get("affine_model", {})
    _require(
        affine.get("fit_candidate_count") == 60
        and len(affine.get("gradients_m_per_action", [])) == 8
        and all(len(row) == 3 for row in affine["gradients_m_per_action"]),
        "oracle affine dimensions differ",
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
        "exact-box decision ledger is inconsistent",
    )
    qp = audit.get("oracle_affine_qp", {})
    _require(
        qp.get("valid") is decision.get("qp_valid")
        and len(qp.get("lower", [])) == 8
        and len(qp.get("action_lower", [])) == 3
        and len(qp.get("action_upper", [])) == 3,
        "exact-box QP ledger differs",
    )
    if qp.get("valid"):
        exact = qp.get("exact_substep_transition", {})
        _require(
            exact.get("captured_state_count")
            == exact.get("expected_internal_mujoco_step_count") + 1,
            "valid exact-box QP lacks exact substep verification",
        )
    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": "valid",
        "scientific_result": False,
        "case_id": CASE_ID,
        "source_commit": expected_commit,
        "slurm_job_id": allocation["slurm_job_id"],
        "result_payload_sha256": recorded_payload_sha256,
        "exact_box_count": 15,
        "research_direction_go": expected_go,
        "stop_reason": decision.get("stop_reason"),
        "checks": {
            "payload_identity": True,
            "clean_h100_source": True,
            "immutable_discovery_and_state_binding": True,
            "exact_zero_inflation_box_union": True,
            "complete_contact_and_substep_ledger": True,
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
    expected_config = load_obstacle_primitive_config(args.config.resolve())
    record = validate(
        _load(args.result.resolve()), args.expected_commit, expected_config
    )
    _atomic_write(args.output.resolve(), record)
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
