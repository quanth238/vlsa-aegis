#!/usr/bin/env python3
"""Validate one completed conservative-obstacle E05 oracle result."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.obstacle_primitives import (
    OBSTACLE_PRIMITIVE_RESULT_SCHEMA,
    load_obstacle_primitive_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import _atomic_write, _load


VALIDATION_SCHEMA = "vlsa_distal_oracle_mesh_obstacle_e05_validation.v1"
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
    computed_payload_sha256 = hashlib.sha256(_canonical(payload)).hexdigest()
    _require(
        recorded_payload_sha256 == computed_payload_sha256,
        "result payload SHA-256 differs",
    )
    _require(
        result.get("schema_version") == OBSTACLE_PRIMITIVE_RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and result.get("case_id") == CASE_ID
        and result.get("failure") is None,
        "completed conservative-obstacle result identity differs",
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
        "result obstacle primitive config differs",
    )
    base = result.get("config", {})
    base_identity = expected_config["base_audit_config"]
    _require(
        base.get("config_file_sha256") == base_identity["config_file_sha256"]
        and base.get("config_payload_sha256")
        == base_identity["config_payload_sha256"],
        "result base oracle config differs",
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
        and 1 <= len(primitives) <= 64
        and union.get("primitive_count") == len(primitives)
        and union.get("mesh_primitive_count", 0)
        + union.get("closed_form_primitive_count", 0)
        == len(primitives)
        and union.get("all_collision_geoms_represented_exactly_once") is True
        and union.get("all_enclosure_certificates_verified") is True,
        "obstacle primitive union certificate summary differs",
    )
    geom_ids = [int(item.get("geom_id", -1)) for item in primitives]
    _require(
        len(set(geom_ids)) == len(geom_ids)
        and all(
            item.get("enclosure_certificate", {}).get("verified") is True
            and item.get("enclosure_certificate", {}).get(
                "rigid_source_geom_pose_update_preserves_enclosure"
            )
            is True
            for item in primitives
        ),
        "obstacle primitive certificates are incomplete",
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
        and audit.get("candidate_count") == len(candidates)
        and len(candidates) == 87
        and nominal.get("obstacle_proxy", {}).get("mode")
        == "live_certified_collision_primitive_union"
        and nominal.get("obstacle_proxy", {}).get("primitive_count")
        == len(primitives)
        and nominal.get("captured_state_count")
        == nominal.get("expected_internal_mujoco_step_count") + 1,
        "oracle candidate or substep ledger differs",
    )
    contacts = nominal.get("raw_protected_contact_events")
    _require(
        isinstance(contacts, list)
        and contacts
        and all(
            "source_obstacle_proxy_quadratic" in item
            and "source_obstacle_contact_point_covered" in item
            for item in contacts
        ),
        "nominal contact witness ledger is incomplete",
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
            "valid QP lacks exact internal-substep verification",
        )
    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": "valid",
        "scientific_result": False,
        "case_id": CASE_ID,
        "source_commit": expected_commit,
        "slurm_job_id": allocation["slurm_job_id"],
        "result_payload_sha256": recorded_payload_sha256,
        "obstacle_primitive_count": len(primitives),
        "research_direction_go": expected_go,
        "stop_reason": decision.get("stop_reason"),
        "checks": {
            "payload_identity": True,
            "clean_h100_source": True,
            "immutable_config_and_state_binding": True,
            "obstacle_union_certificates": True,
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
