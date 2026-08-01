#!/usr/bin/env python3
"""Independently validate one terminal compact Poisson feasibility artifact.

This consumer intentionally does not import the producer classifier.  It
reconstructs pairing, contact, QP, activation, clearance, and active-interval
motion from the two arm traces in the immutable result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main.poisson_fullbody.contracts import (  # noqa: E402
    ArtifactContractError,
    load_hashed_json,
    sha256_file,
)


SUMMARY_SCHEMA = "vlsa_poisson_fast_feasibility_independent_validation.v1"
RESULT_SCHEMA = "vlsa_poisson_fast_feasibility_result.v1"
EXPECTED_CASE = "vlsa-t1-goal-ii-t0-e05"
EXPECTED_COMMIT = "b1854d6b82836757fdfed50ad0788791a62e651b"
EXPECTED_PROTOCOL_FILE_SHA256 = (
    "f7d208ab65167a3c19275c7150ecaae19abd5760571b657c4fe598d2e49f4c20"
)
EXPECTED_HISTORICAL_FILE_SHA256 = (
    "273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b"
)
EXPECTED_HISTORICAL_PAYLOAD_SHA256 = (
    "ec58c8581297751de33756e76efbf5b18e24a5f836aa6a8f147d0caebdd92d1c"
)
EXPECTED_ACTION_SEQUENCE_SHA256 = (
    "e5e2df4efee667fee0b0b7596dbc203a3cce84e67cb3507cb839132d2ee67481"
)
EXPECTED_BOUNDARY_B_SHA256 = (
    "2b4d25f12da2e1c110883b9f2194c9790270f36911f449d8c4f058a039c90335"
)
EXPECTED_BOUNDARY_B_RAW_SHA256 = (
    "ff1a9a80d03178c9ac85e5dc6750aac98f920a050462df0b17d7ca1cdbb069c7"
)
EXPECTED_FIELD_BUNDLE_SHA256 = (
    "979ebc41198293cec1a4d95e1c9df67d121767824e5b0d4a6c089bdb030d7f77"
)
EXPECTED_FIELD_PROTOCOL_SHA256 = (
    "2125989269a2ffeeb8d3408d56e4aaa74e1dc5816256d8210bf9ee686c5f5a30"
)
EXPECTED_FIELD_PARAMETER_BLOCK_SHA256 = (
    "11de833ab8b1586948e7eb039c449a66a0c1a4107fa3232b236ad1ecb9ceeabe"
)
EXPECTED_FULL_ROBOT_SAMPLING_SHA256 = (
    "063a56c76c4d554f35523e7e171f6076c950d147f702d5a6eff1b59acdd78ba1"
)
EXPECTED_BRANCH = "codex/poisson-fast-feasibility"
EXPECTED_CLAIM_SCOPE = (
    "one_post_hoc_0.4_second_window_offline_empirical_contact_feasibility_"
    "not_tracking_certified_not_task_success_not_full_episode_not_population_safety"
)

START_BOUNDARY = 4500
END_BOUNDARY = 4700
EXPECTED_ACTIONS = tuple(range(180, 188))
EXPECTED_FILTER_UPDATES = 40
EXPECTED_PHYSICS_SUBSTEPS = 200
SUBSTEPS_PER_FILTER = 5
INNER_DT_S = 0.01
PHYSICS_DT_S = 0.002

SAFE_RESIDUAL_MINIMUM = -5.0e-7
ACTIVATION_RESIDUAL_MAXIMUM = -5.0e-7
CORRECTION_MINIMUM = 1.0e-4
SAFE_COMMAND_MINIMUM = 5.0e-2
MOTION_RATIO_MINIMUM = 0.25
MEASURED_MOTION_MINIMUM = 1.0e-4
CARTESIAN_PATH_MINIMUM = 1.0e-4
TRACKING_LINF_MAXIMUM = 0.05
TRACKING_RMSE_MAXIMUM = 0.02
BOUND_TOLERANCE = 5.0e-8
FLOAT_TOLERANCE = 1.0e-10


class _Audit:
    def __init__(self) -> None:
        self.discrepancies: List[str] = []

    def check(self, condition: bool, message: str) -> None:
        if not condition and message not in self.discrepancies:
            self.discrepancies.append(message)

    def mapping(self, value: Any, label: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            self.check(False, "%s_not_object" % label)
            return {}
        return value

    def sequence(self, value: Any, label: str) -> Sequence[Any]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            self.check(False, "%s_not_array" % label)
            return ()
        return value

    def integer(self, value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            self.check(False, "%s_not_integer" % label)
            return 0
        return int(value)

    def number(self, value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.check(False, "%s_not_number" % label)
            return 0.0
        output = float(value)
        if not math.isfinite(output):
            self.check(False, "%s_not_finite" % label)
            return 0.0
        return output

    def boolean(self, value: Any, label: str) -> bool:
        if not isinstance(value, bool):
            self.check(False, "%s_not_boolean" % label)
            return False
        return bool(value)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _close(left: float, right: float, tolerance: float = FLOAT_TOLERANCE) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-9, abs_tol=tolerance)


def _sha256(audit: _Audit, value: Any, label: str) -> str:
    valid = (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
    audit.check(valid, "%s_invalid" % label)
    return value if valid else ""


def _nonempty_mapping(
    audit: _Audit,
    value: Any,
    label: str,
) -> Mapping[str, Any]:
    result = audit.mapping(value, label)
    audit.check(bool(result), "%s_empty" % label)
    return result


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return _norm(tuple(float(a) - float(b) for a, b in zip(left, right)))


def _vector(
    audit: _Audit,
    value: Any,
    length: int,
    label: str,
) -> Tuple[float, ...]:
    rows = audit.sequence(value, label)
    audit.check(len(rows) == length, "%s_length_differs" % label)
    output = tuple(audit.number(item, "%s_item" % label) for item in rows[:length])
    if len(output) < length:
        output += (0.0,) * (length - len(output))
    return output


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _contact_boundary(audit: _Audit, record: Mapping[str, Any], label: str) -> int:
    observation = audit.integer(record.get("observation_index"), "%s_observation" % label)
    phase = record.get("source_phase")
    if phase == "live_solver_phase_preintegration_geometry":
        return START_BOUNDARY + observation
    if phase == "post_integration_recomputed":
        return START_BOUNDARY + observation + 1
    audit.check(False, "%s_source_phase_unknown" % label)
    return END_BOUNDARY + 1


def _validate_settled_contacts(
    audit: _Audit,
    settled: Mapping[str, Any],
    link_geom_ids: Iterable[int],
    robot_geom_ids: Iterable[int],
    obstacle_geom_ids: Iterable[int],
    label: str,
) -> None:
    """Reconstruct settled contact from the raw candidate ledger.

    The physical ledger and its summary fields are producer conveniences.  A
    negative signed distance in the candidate ledger remains authoritative even
    if those derived fields are stale or omitted.
    """

    robot_set = set(int(value) for value in robot_geom_ids)
    obstacle_set = set(int(value) for value in obstacle_geom_ids)
    link_set = set(int(value) for value in link_geom_ids)
    candidates: List[Mapping[str, Any]] = []
    for index, value in enumerate(
        audit.sequence(
            settled.get("candidate_contact_point_records"),
            "%s_settled_candidate_records" % label,
        )
    ):
        record_label = "%s_settled_candidate_%d" % (label, index)
        record = audit.mapping(value, record_label)
        distance = audit.number(
            record.get("contact_distance_m"), "%s_distance" % record_label
        )
        physical = distance <= 0.0
        audit.check(
            record.get("is_physical_nonpositive_distance_contact") is physical,
            "%s_physical_flag_differs" % record_label,
        )
        audit.check(
            record.get("source_phase")
            == "settled_post_integration_recomputed",
            "%s_phase_differs" % record_label,
        )
        for field in (
            "observation_index",
            "high_level_index",
            "inner_control_index",
            "physics_substep_index",
        ):
            audit.check(
                record.get(field) is None,
                "%s_%s_not_none" % (record_label, field),
            )
        robot_geom_id = audit.integer(
            record.get("robot_geom_id"), "%s_robot_geom" % record_label
        )
        obstacle_geom_id = audit.integer(
            record.get("obstacle_geom_id"), "%s_obstacle_geom" % record_label
        )
        audit.check(
            robot_geom_id in robot_set,
            "%s_robot_geom_outside_authority" % record_label,
        )
        audit.check(
            obstacle_geom_id in obstacle_set,
            "%s_obstacle_geom_outside_authority" % record_label,
        )
        candidates.append(record)

    physical = [
        record
        for record in candidates
        if record.get("is_physical_nonpositive_distance_contact") is True
    ]
    physical_ledger = [
        audit.mapping(value, "%s_settled_physical_%d" % (label, index))
        for index, value in enumerate(
            audit.sequence(
                settled.get("physical_contact_point_records"),
                "%s_settled_physical_records" % label,
            )
        )
    ]
    audit.check(
        _canonical(physical_ledger) == _canonical(physical),
        "%s_settled_physical_ledger_differs" % label,
    )
    audit.check(
        settled.get("candidate_contact_point_record_count") == len(candidates),
        "%s_settled_candidate_count_differs" % label,
    )
    audit.check(
        settled.get("physical_contact_point_record_count") == len(physical),
        "%s_settled_physical_count_differs" % label,
    )
    expected_first_candidate = candidates[0] if candidates else None
    expected_first_physical = physical[0] if physical else None
    audit.check(
        _canonical(settled.get("first_candidate_contact_point_record"))
        == _canonical(expected_first_candidate),
        "%s_settled_first_candidate_differs" % label,
    )
    audit.check(
        _canonical(settled.get("first_physical_contact_point_record"))
        == _canonical(expected_first_physical),
        "%s_settled_first_physical_differs" % label,
    )
    any_present = bool(physical)
    link_present = any(
        audit.integer(
            record.get("robot_geom_id"), "%s_settled_link_robot_geom" % label
        )
        in link_set
        for record in physical
    )
    audit.check(
        settled.get("any_robot_obstacle_contact") is any_present,
        "%s_settled_any_contact_differs" % label,
    )
    audit.check(
        settled.get("link56_obstacle_contact") is link_present,
        "%s_settled_link_contact_differs" % label,
    )
    audit.check(not any_present, "%s_settled_contact_present" % label)


def _reconstruct_contacts(
    audit: _Audit,
    arm: Mapping[str, Any],
    link_geom_ids: Iterable[int],
    robot_geom_ids: Iterable[int],
    obstacle_geom_ids: Iterable[int],
    label: str,
) -> Dict[str, Any]:
    measurement = audit.mapping(arm.get("measurement"), "%s_measurement" % label)
    settled = audit.mapping(measurement.get("settled_state"), "%s_settled" % label)
    _validate_settled_contacts(
        audit,
        settled,
        link_geom_ids,
        robot_geom_ids,
        obstacle_geom_ids,
        label,
    )
    records: List[Mapping[str, Any]] = []
    robot_set = set(int(value) for value in robot_geom_ids)
    obstacle_set = set(int(value) for value in obstacle_geom_ids)
    post_candidates: List[Mapping[str, Any]] = []
    post_candidate_physical: List[Mapping[str, Any]] = []
    for index, value in enumerate(
        audit.sequence(
            measurement.get("post_state_candidate_contact_point_records"),
            "%s_post_state_candidate_records" % label,
        )
    ):
        record_label = "%s_post_state_candidate_%d" % (label, index)
        record = audit.mapping(value, record_label)
        distance = audit.number(
            record.get("contact_distance_m"), "%s_distance" % record_label
        )
        physical = distance <= 0.0
        audit.check(
            record.get("is_physical_nonpositive_distance_contact") is physical,
            "%s_physical_flag_differs" % record_label,
        )
        audit.check(
            record.get("source_phase") == "post_integration_recomputed",
            "%s_phase_differs" % record_label,
        )
        robot_geom_id = audit.integer(
            record.get("robot_geom_id"), "%s_robot_geom" % record_label
        )
        obstacle_geom_id = audit.integer(
            record.get("obstacle_geom_id"), "%s_obstacle_geom" % record_label
        )
        audit.check(
            robot_geom_id in robot_set,
            "%s_robot_geom_outside_authority" % record_label,
        )
        audit.check(
            obstacle_geom_id in obstacle_set,
            "%s_obstacle_geom_outside_authority" % record_label,
        )
        post_candidates.append(record)
        if physical:
            post_candidate_physical.append(record)
    stored_post_physical = [
        audit.mapping(value, "%s_post_state_physical_ledger_%d" % (label, index))
        for index, value in enumerate(
            audit.sequence(
                measurement.get("post_state_physical_contact_point_records"),
                "%s_post_state_physical_ledger" % label,
            )
        )
    ]
    audit.check(
        _canonical(stored_post_physical) == _canonical(post_candidate_physical),
        "%s_post_state_physical_ledger_differs" % label,
    )
    audit.check(
        measurement.get("post_state_candidate_contact_point_record_count")
        == len(post_candidates),
        "%s_post_state_candidate_count_differs" % label,
    )
    phase_counts: Dict[str, int] = {}
    for field, expected_phase in (
        (
            "live_solver_phase_contact_point_records",
            "live_solver_phase_preintegration_geometry",
        ),
        ("post_state_physical_contact_point_records", "post_integration_recomputed"),
    ):
        phase_count = 0
        for index, value in enumerate(
            audit.sequence(measurement.get(field), "%s_%s" % (label, field))
        ):
            record = audit.mapping(value, "%s_%s_%d" % (label, field, index))
            distance = audit.number(
                record.get("contact_distance_m"),
                "%s_%s_%d_distance" % (label, field, index),
            )
            physical = distance <= 0.0
            audit.check(
                record.get("is_physical_nonpositive_distance_contact") is physical,
                "%s_%s_%d_physical_flag_differs" % (label, field, index),
            )
            audit.check(
                record.get("source_phase") == expected_phase,
                "%s_%s_%d_phase_differs" % (label, field, index),
            )
            observation = audit.integer(
                record.get("observation_index"),
                "%s_%s_%d_observation" % (label, field, index),
            )
            audit.check(
                0 <= observation < EXPECTED_PHYSICS_SUBSTEPS,
                "%s_%s_%d_observation_out_of_range" % (label, field, index),
            )
            expected_index = [
                observation // 25,
                (observation // 5) % 5,
                observation % 5,
            ]
            observed_index = [
                record.get("high_level_index"),
                record.get("inner_control_index"),
                record.get("physics_substep_index"),
            ]
            audit.check(
                observed_index == expected_index,
                "%s_%s_%d_index_differs" % (label, field, index),
            )
            robot_geom_id = audit.integer(
                record.get("robot_geom_id"),
                "%s_%s_%d_robot_geom" % (label, field, index),
            )
            obstacle_geom_id = audit.integer(
                record.get("obstacle_geom_id"),
                "%s_%s_%d_obstacle_geom" % (label, field, index),
            )
            audit.check(
                robot_geom_id in robot_set,
                "%s_%s_%d_robot_geom_outside_authority" % (label, field, index),
            )
            audit.check(
                obstacle_geom_id in obstacle_set,
                "%s_%s_%d_obstacle_geom_outside_authority" % (label, field, index),
            )
            if physical:
                records.append(record)
                phase_count += 1
        phase_counts[field] = phase_count
    live_count = phase_counts.get("live_solver_phase_contact_point_records", 0)
    post_count = phase_counts.get("post_state_physical_contact_point_records", 0)
    audit.check(
        measurement.get("live_solver_nonpositive_contact_point_record_count")
        == live_count,
        "%s_live_contact_count_differs" % label,
    )
    audit.check(
        measurement.get("post_state_physical_contact_point_record_count")
        == post_count,
        "%s_post_contact_count_differs" % label,
    )
    audit.check(
        measurement.get("rollout_phase_physical_contact_point_record_count")
        == live_count + post_count,
        "%s_rollout_contact_count_differs" % label,
    )
    audit.check(
        measurement.get("total_physical_contact_point_record_count")
        == live_count + post_count,
        "%s_total_contact_count_differs" % label,
    )
    live_candidates = audit.integer(
        measurement.get("live_solver_candidate_contact_point_record_count"),
        "%s_live_candidate_count" % label,
    )
    audit.check(
        live_candidates
        == len(
            audit.sequence(
                measurement.get("live_solver_phase_contact_point_records"),
                "%s_live_solver_candidate_records" % label,
            )
        ),
        "%s_live_candidate_count_differs" % label,
    )
    audit.check(
        measurement.get("total_candidate_contact_point_record_count")
        == live_candidates + len(post_candidates),
        "%s_total_candidate_count_differs" % label,
    )
    link_set = set(int(value) for value in link_geom_ids)
    any_boundaries = [
        _contact_boundary(audit, row, "%s_contact" % label) for row in records
    ]
    link_boundaries = [
        _contact_boundary(audit, row, "%s_link_contact" % label)
        for row in records
        if audit.integer(row.get("robot_geom_id"), "%s_robot_geom" % label)
        in link_set
    ]
    any_present = bool(any_boundaries)
    link_present = bool(link_boundaries)
    first_any = min(any_boundaries) if any_boundaries else None
    first_link = min(link_boundaries) if link_boundaries else None
    audit.check(
        measurement.get("any_robot_obstacle_contact") is any_present,
        "%s_measurement_any_contact_differs" % label,
    )
    audit.check(
        measurement.get("rollout_any_robot_obstacle_contact") is any_present,
        "%s_rollout_any_contact_differs" % label,
    )
    audit.check(
        measurement.get("live_solver_any_robot_obstacle_contact") is bool(live_count),
        "%s_live_any_contact_differs" % label,
    )
    audit.check(
        measurement.get("post_state_any_robot_obstacle_contact") is bool(post_count),
        "%s_post_any_contact_differs" % label,
    )
    audit.check(
        measurement.get("link56_obstacle_contact") is link_present,
        "%s_measurement_link_contact_differs" % label,
    )
    literal = audit.mapping(arm.get("literal_contact"), "%s_literal_contact" % label)
    audit.check(
        literal.get("any_robot_selected_obstacle_present") is any_present,
        "%s_literal_any_contact_differs" % label,
    )
    audit.check(
        literal.get("link56_present") is link_present,
        "%s_literal_link_contact_differs" % label,
    )
    audit.check(
        literal.get("first_any_robot_physical_boundary") == first_any,
        "%s_first_any_boundary_differs" % label,
    )
    audit.check(
        literal.get("first_link56_physical_boundary") == first_link,
        "%s_first_link_boundary_differs" % label,
    )
    audit.check(
        measurement.get("physical_contact_distance_semantics")
        == "mujoco_contact_dist_le_0",
        "%s_contact_semantics_differs" % label,
    )
    physics_rows = audit.sequence(arm.get("physics_trace"), "%s_physics_trace" % label)
    flagged_observations: List[int] = []
    for index, value in enumerate(physics_rows):
        row = audit.mapping(value, "%s_contact_physics_%d" % (label, index))
        flag = audit.boolean(
            row.get("literal_contact_observed"),
            "%s_contact_physics_%d_literal_flag" % (label, index),
        )
        if flag:
            flagged_observations.append(index)
    if arm.get("arm_name") == "joint_velocity_adapter_plus_link56_psf":
        expected_flags = (
            [
                min(
                    audit.integer(record.get("observation_index"), "%s_flag_observation" % label)
                    for record in records
                )
            ]
            if records
            else []
        )
        audit.check(
            flagged_observations == expected_flags,
            "%s_physics_literal_contact_flags_differ" % label,
        )
    else:
        audit.check(
            not flagged_observations,
            "%s_adapter_physics_literal_contact_flag_unexpected" % label,
        )
    return {
        "any_present": any_present,
        "link_present": link_present,
        "first_any_boundary": first_any,
        "first_link_boundary": first_link,
        "record_count": len(records),
    }


def _validate_exogenous_pairing(
    audit: _Audit,
    adapter_commands: Sequence[Mapping[str, Any]],
    psf_commands: Sequence[Mapping[str, Any]],
) -> None:
    """Reconstruct the frozen high-level action pairing visible in both arms."""

    audit.check(
        len(adapter_commands) == len(psf_commands),
        "pair_command_population_differs",
    )
    for index, (adapter_row, psf_row) in enumerate(
        zip(adapter_commands, psf_commands)
    ):
        label = "pair_command_%d" % index
        for field in ("source_action_index", "inner_control_index"):
            audit.check(
                adapter_row.get(field) == psf_row.get(field),
                "%s_%s_differs" % (label, field),
            )
        adapter_step = audit.mapping(adapter_row.get("adapter"), "%s_adapter" % label)
        psf_step = audit.mapping(psf_row.get("adapter"), "%s_psf" % label)
        adapter_diag = audit.mapping(
            adapter_step.get("diagnostics"), "%s_adapter_diagnostics" % label
        )
        psf_diag = audit.mapping(
            psf_step.get("diagnostics"), "%s_psf_diagnostics" % label
        )
        adapter_gripper = audit.number(
            adapter_diag.get("gripper_command_unchanged"),
            "%s_adapter_gripper" % label,
        )
        psf_gripper = audit.number(
            psf_diag.get("gripper_command_unchanged"),
            "%s_psf_gripper" % label,
        )
        audit.check(
            _close(adapter_gripper, psf_gripper),
            "%s_gripper_differs" % label,
        )
        if adapter_row.get("inner_control_index") == 0:
            for field in ("position_error_m", "orientation_error_rotvec_rad"):
                left = _vector(
                    audit,
                    adapter_diag.get(field),
                    3,
                    "%s_adapter_%s" % (label, field),
                )
                right = _vector(
                    audit,
                    psf_diag.get(field),
                    3,
                    "%s_psf_%s" % (label, field),
                )
                audit.check(
                    all(_close(a, b) for a, b in zip(left, right)),
                    "%s_%s_differs" % (label, field),
                )
            left_twist = _vector(
                audit,
                adapter_step.get("desired_twist"),
                6,
                "%s_adapter_desired_twist" % label,
            )
            right_twist = _vector(
                audit,
                psf_step.get("desired_twist"),
                6,
                "%s_psf_desired_twist" % label,
            )
            audit.check(
                all(_close(a, b) for a, b in zip(left_twist, right_twist)),
                "%s_high_level_action_differs" % label,
            )
    if adapter_commands and psf_commands:
        adapter_initial = _vector(
            audit,
            adapter_commands[0].get("nominal_qdot_rad_s"),
            7,
            "pair_initial_adapter_nominal",
        )
        psf_initial = _vector(
            audit,
            psf_commands[0].get("nominal_qdot_rad_s"),
            7,
            "pair_initial_psf_nominal",
        )
        audit.check(
            all(_close(a, b) for a, b in zip(adapter_initial, psf_initial)),
            "pair_initial_nominal_command_differs",
        )


def _validate_adapter_execution(
    audit: _Audit,
    adapter: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
) -> None:
    for index, row in enumerate(commands):
        label = "adapter_command_%d" % index
        nominal = _vector(
            audit, row.get("nominal_qdot_rad_s"), 7, "%s_nominal" % label
        )
        executed = _vector(
            audit, row.get("executed_qdot_rad_s"), 7, "%s_executed" % label
        )
        lower = _vector(
            audit,
            row.get("nominal_dynamic_lower_bound_rad_s"),
            7,
            "%s_lower" % label,
        )
        upper = _vector(
            audit,
            row.get("nominal_dynamic_upper_bound_rad_s"),
            7,
            "%s_upper" % label,
        )
        audit.check(
            all(_close(a, b) for a, b in zip(nominal, executed)),
            "%s_nominal_executed_differs" % label,
        )
        audit.check(
            _close(
                audit.number(row.get("correction_l2_rad_s"), "%s_correction" % label),
                0.0,
            ),
            "%s_correction_nonzero" % label,
        )
        within = all(
            low - BOUND_TOLERANCE <= value <= high + BOUND_TOLERANCE
            and abs(value) <= 0.5 + BOUND_TOLERANCE
            for value, low, high in zip(executed, lower, upper)
        )
        audit.check(within, "%s_bounds_failed" % label)
        audit.check(
            row.get("nominal_within_dynamic_joint_bounds") is True,
            "%s_nominal_bound_flag_false" % label,
        )
        execution = audit.mapping(row.get("execution"), "%s_execution" % label)
        issued = _vector(
            audit,
            execution.get("executed_qdot_physical_rad_s"),
            7,
            "%s_execution_issued" % label,
        )
        audit.check(
            all(_close(a, b) for a, b in zip(issued, executed)),
            "%s_execution_issued_differs" % label,
        )
    expected = len(commands)
    for field in (
        "issued_command_bound_check_count",
        "nominal_dynamic_bound_check_count",
    ):
        audit.check(adapter.get(field) == expected, "adapter_%s_differs" % field)


def _expected_command_keys(count: int) -> List[Tuple[int, int]]:
    return [
        (180 + index // 5, index % 5)
        for index in range(count)
    ]


def _expected_physics_keys(count: int) -> List[Tuple[int, int, int]]:
    return [
        (180 + index // 25, (index % 25) // 5, index % 5)
        for index in range(count)
    ]


def _validate_arm_cadence(
    audit: _Audit,
    arm: Mapping[str, Any],
    label: str,
    *,
    contact_present: bool,
    must_complete: bool,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    cadence = audit.mapping(arm.get("execution_cadence"), "%s_cadence" % label)
    for field, expected in (
        ("control_timestep_s", INNER_DT_S),
        ("wrapper_model_timestep_s", PHYSICS_DT_S),
        ("mujoco_model_timestep_s", PHYSICS_DT_S),
    ):
        observed = audit.number(cadence.get(field), "%s_%s" % (label, field))
        audit.check(_close(observed, expected), "%s_%s_differs" % (label, field))
    audit.check(
        arm.get("qp_max_iterations_effective") == 50000,
        "%s_qp_iteration_budget_differs" % label,
    )
    commands = [
        audit.mapping(value, "%s_command_%d" % (label, index))
        for index, value in enumerate(
            audit.sequence(arm.get("command_trace"), "%s_command_trace" % label)
        )
    ]
    physics = [
        audit.mapping(value, "%s_physics_%d" % (label, index))
        for index, value in enumerate(
            audit.sequence(arm.get("physics_trace"), "%s_physics_trace" % label)
        )
    ]
    command_count = len(commands)
    physics_count = len(physics)
    for field in ("filter_update_count",):
        audit.check(arm.get(field) == command_count, "%s_%s_differs" % (label, field))
    for field in ("physics_substep_count", "physics_trace_row_count"):
        audit.check(arm.get(field) == physics_count, "%s_%s_differs" % (label, field))
    measurement = audit.mapping(arm.get("measurement"), "%s_measurement" % label)
    monitor_count = audit.integer(
        measurement.get("observed_physics_substeps"), "%s_monitor_count" % label
    )
    audit.check(
        arm.get("monitor_observed_physics_substep_count") == monitor_count,
        "%s_stored_monitor_count_differs" % label,
    )
    audit.check(monitor_count == physics_count, "%s_monitor_trace_count_differs" % label)
    audit.check(
        arm.get("physics_monitor_trace_counts_match") is True,
        "%s_monitor_trace_match_false" % label,
    )
    if must_complete:
        audit.check(command_count == EXPECTED_FILTER_UPDATES, "%s_update_count_differs" % label)
        audit.check(physics_count == EXPECTED_PHYSICS_SUBSTEPS, "%s_substep_count_differs" % label)
        audit.check(arm.get("exposure_complete") is True, "%s_exposure_incomplete" % label)
    elif contact_present:
        expected_updates = (
            (physics_count + SUBSTEPS_PER_FILTER - 1) // SUBSTEPS_PER_FILTER
            if physics_count > 0
            else 0
        )
        audit.check(physics_count > 0, "%s_contact_without_substep" % label)
        audit.check(command_count == expected_updates, "%s_prefix_cadence_differs" % label)
        audit.check(command_count <= EXPECTED_FILTER_UPDATES, "%s_prefix_too_long" % label)
        audit.check(physics_count <= EXPECTED_PHYSICS_SUBSTEPS, "%s_prefix_too_long" % label)
        audit.check(arm.get("contact_terminated_early") is True, "%s_contact_not_terminal" % label)
    observed_command_keys = [
        (
            audit.integer(row.get("source_action_index"), "%s_command_source" % label),
            audit.integer(row.get("inner_control_index"), "%s_command_inner" % label),
        )
        for row in commands
    ]
    audit.check(
        observed_command_keys == _expected_command_keys(command_count),
        "%s_command_index_population_differs" % label,
    )
    observed_physics_keys = []
    for index, row in enumerate(physics):
        observed_physics_keys.append(
            (
                audit.integer(row.get("source_action_index"), "%s_physics_source" % label),
                audit.integer(row.get("inner_control_index"), "%s_physics_inner" % label),
                audit.integer(row.get("physics_substep_index"), "%s_physics_substep" % label),
            )
        )
        audit.check(
            row.get("observation_index") == index,
            "%s_physics_observation_index_differs" % label,
        )
        audit.check(
            row.get("post_state_physical_boundary") == START_BOUNDARY + index + 1,
            "%s_physics_boundary_differs" % label,
        )
        audit.check(
            row.get("local_action_index") == index // 25,
            "%s_physics_local_action_index_differs" % label,
        )
    audit.check(
        observed_physics_keys == _expected_physics_keys(physics_count),
        "%s_physics_index_population_differs" % label,
    )
    audit.check(
        list(arm.get("source_action_indexes", ())) == list(EXPECTED_ACTIONS),
        "%s_source_actions_differ" % label,
    )
    if physics:
        audit.check(measurement.get("first_index") == [0, 0, 0], "%s_first_index_differs" % label)
        final_local_index = physics_count - 1
        expected_last_index = [
            final_local_index // (5 * SUBSTEPS_PER_FILTER),
            (final_local_index // SUBSTEPS_PER_FILTER) % 5,
            final_local_index % SUBSTEPS_PER_FILTER,
        ]
        audit.check(
            measurement.get("last_index") == expected_last_index,
            "%s_last_index_differs" % label,
        )
    for index, row in enumerate(commands):
        audit.check(
            row.get("local_action_index") == index // 5,
            "%s_command_local_action_index_differs" % label,
        )
        audit.check(
            row.get("physical_boundary") == START_BOUNDARY + index * SUBSTEPS_PER_FILTER,
            "%s_command_boundary_differs" % label,
        )
    for index, row in enumerate(physics):
        command_index = index // SUBSTEPS_PER_FILTER
        if command_index >= len(commands):
            audit.check(False, "%s_physics_without_command" % label)
            continue
        issued = _vector(
            audit,
            row.get("issued_qvel_rad_s"),
            7,
            "%s_physics_%d_issued" % (label, index),
        )
        executed = _vector(
            audit,
            commands[command_index].get("executed_qdot_rad_s"),
            7,
            "%s_command_%d_executed_binding" % (label, command_index),
        )
        audit.check(
            all(_close(left, right) for left, right in zip(issued, executed)),
            "%s_physics_%d_issued_command_differs" % (label, index),
        )
    return commands, physics


def _validate_pairing(
    audit: _Audit,
    adapter: Mapping[str, Any],
    psf: Mapping[str, Any],
    boundary_hash: Any,
) -> None:
    fields = (
        "model_topology_sha256",
        "physical_model_sha256",
        "compiled_mjb_sha256",
        "official_integration_state_sha256",
        "target_official_integration_state_sha256",
        "settled_state_sha256",
        "target_state_sha256",
        "controller",
        "controller_software_state",
        "pid_memory_reset",
    )
    adapter_restore = audit.mapping(adapter.get("restore"), "adapter_restore")
    psf_restore = audit.mapping(psf.get("restore"), "psf_restore")
    hash_fields = fields[:7]
    mapping_fields = fields[7:]
    for label, restore in (("adapter", adapter_restore), ("psf", psf_restore)):
        audit.check(
            restore.get("schema_version")
            == "vlsa_poisson_controller_state_restore.v1",
            "%s_restore_schema_differs" % label,
        )
        for field in hash_fields:
            _sha256(
                audit,
                restore.get(field),
                "%s_restore_%s" % (label, field),
            )
        for field in mapping_fields:
            _nonempty_mapping(
                audit,
                restore.get(field),
                "%s_restore_%s" % (label, field),
            )
        audit.check(
            restore.get("official_integration_state_available") is True,
            "%s_official_restore_unavailable" % label,
        )
        audit.check(
            restore.get("exact_flattened_state") is True,
            "%s_flattened_restore_not_exact" % label,
        )
    for field in fields:
        audit.check(
            _canonical(adapter_restore.get(field)) == _canonical(psf_restore.get(field)),
            "pair_restore_%s_differs" % field,
        )
    for label, arm, restore in (
        ("adapter", adapter, adapter_restore),
        ("psf", psf, psf_restore),
    ):
        audit.check(arm.get("fresh_adapter") is True, "%s_adapter_not_fresh" % label)
        audit.check(
            restore.get("official_integration_state_sha256")
            == restore.get("target_official_integration_state_sha256"),
            "%s_official_restore_differs" % label,
        )
        audit.check(
            restore.get("settled_state_sha256") == restore.get("target_state_sha256"),
            "%s_flattened_restore_differs" % label,
        )
        start_hash = _sha256(
            audit,
            arm.get("start_official_raw_bytes_sha256"),
            "%s_start_boundary_hash" % label,
        )
        audit.check(
            start_hash == boundary_hash,
            "%s_start_boundary_hash_differs" % label,
        )
    audit.check(
        adapter.get("start_official_raw_bytes_sha256")
        == psf.get("start_official_raw_bytes_sha256"),
        "pair_start_raw_hash_differs",
    )


def _validate_qp(
    audit: _Audit,
    adapter: Mapping[str, Any],
    psf: Mapping[str, Any],
    commands: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    sample_count = audit.integer(psf.get("protected_sample_count"), "psf_sample_count")
    audit.check(sample_count > 0, "psf_sample_population_empty")
    normalized: List[Dict[str, Any]] = []
    solve_times: List[float] = []
    iterations: List[int] = []
    for index, row in enumerate(commands):
        label = "psf_command_%d" % index
        nominal = _vector(audit, row.get("nominal_qdot_rad_s"), 7, "%s_nominal" % label)
        executed = _vector(audit, row.get("executed_qdot_rad_s"), 7, "%s_executed" % label)
        lower = _vector(audit, row.get("nominal_dynamic_lower_bound_rad_s"), 7, "%s_lower" % label)
        upper = _vector(audit, row.get("nominal_dynamic_upper_bound_rad_s"), 7, "%s_upper" % label)
        correction = _norm(tuple(safe - raw for safe, raw in zip(executed, nominal)))
        stored_correction = audit.number(row.get("correction_l2_rad_s"), "%s_correction" % label)
        audit.check(_close(correction, stored_correction), "%s_correction_differs" % label)
        nominal_within_bounds = all(
            low - BOUND_TOLERANCE <= value <= high + BOUND_TOLERANCE
            for value, low, high in zip(nominal, lower, upper)
        )
        audit.check(
            row.get("nominal_within_dynamic_joint_bounds") is nominal_within_bounds,
            "%s_nominal_bound_flag_differs" % label,
        )
        audit.check(
            all(low - BOUND_TOLERANCE <= value <= high + BOUND_TOLERANCE for value, low, high in zip(executed, lower, upper)),
            "%s_dynamic_bound_failed" % label,
        )
        audit.check(all(abs(value) <= 0.5 + BOUND_TOLERANCE for value in executed), "%s_physical_bound_failed" % label)
        audit.check(nominal_within_bounds, "%s_nominal_bounds_confounded" % label)
        safe_residual = audit.number(
            row.get("safe_cbf_residual_minimum_m2_per_s"), "%s_safe_residual" % label
        )
        audit.check(safe_residual >= SAFE_RESIDUAL_MINIMUM, "%s_safe_residual_failed" % label)
        nominal_residual = audit.number(
            row.get("nominal_cbf_residual_minimum_m2_per_s"), "%s_nominal_residual" % label
        )
        qp = audit.mapping(row.get("qp"), "%s_qp" % label)
        status = str(qp.get("status", "")).lower()
        audit.check(qp.get("solver") == "osqp", "%s_qp_solver_differs" % label)
        audit.check(status in ("solved", "solved inaccurate"), "%s_qp_not_solved" % label)
        iteration = audit.integer(qp.get("iterations"), "%s_iterations" % label)
        audit.check(0 <= iteration <= 50000, "%s_iterations_out_of_range" % label)
        solve_time = audit.number(qp.get("solve_time_seconds"), "%s_solve_time" % label)
        audit.check(solve_time >= 0.0, "%s_solve_time_negative" % label)
        audit.check(qp.get("input_constraint_count") == sample_count, "%s_constraint_count_differs" % label)
        audit.check(qp.get("solved_constraint_count") == sample_count, "%s_solved_constraint_count_differs" % label)
        qp_correction = audit.number(qp.get("correction_l2_rad_s"), "%s_qp_correction" % label)
        audit.check(_close(qp_correction, correction), "%s_qp_correction_differs" % label)
        qp_safe_residual = audit.number(
            qp.get("minimum_raw_cbf_residual_m2_per_s"),
            "%s_qp_safe_residual" % label,
        )
        qp_nominal_residual = audit.number(
            qp.get("nominal_minimum_raw_cbf_residual_m2_per_s"),
            "%s_qp_nominal_residual" % label,
        )
        audit.check(
            _close(qp_safe_residual, safe_residual),
            "%s_qp_safe_residual_differs" % label,
        )
        audit.check(
            _close(qp_nominal_residual, nominal_residual),
            "%s_qp_nominal_residual_differs" % label,
        )
        bound_violation = audit.number(
            qp.get("maximum_velocity_bound_violation_rad_s"),
            "%s_qp_bound_violation" % label,
        )
        audit.check(
            bound_violation <= BOUND_TOLERANCE,
            "%s_qp_bound_violation" % label,
        )
        execution = audit.mapping(row.get("execution"), "%s_execution" % label)
        audit.check(
            _vector(
                audit,
                execution.get("executed_qdot_physical_rad_s"),
                7,
                "%s_execution_executed" % label,
            )
            == executed,
            "%s_execution_executed_differs" % label,
        )
        audit.check(
            _vector(
                audit,
                execution.get("nominal_qdot_physical_rad_s"),
                7,
                "%s_execution_nominal" % label,
            )
            == nominal,
            "%s_execution_nominal_differs" % label,
        )
        audit.check(
            _close(
                audit.number(
                    execution.get("filter_correction_l2_rad_s"),
                    "%s_execution_correction" % label,
                ),
                correction,
            ),
            "%s_execution_correction_differs" % label,
        )
        solve_times.append(solve_time)
        iterations.append(iteration)
        normalized.append(
            {
                "raw": row,
                "source": row.get("source_action_index"),
                "inner": row.get("inner_control_index"),
                "boundary": row.get("physical_boundary"),
                "nominal": nominal,
                "executed": executed,
                "correction": correction,
                "safe_residual": safe_residual,
                "nominal_residual": nominal_residual,
                "nominal_bounds": row.get("nominal_within_dynamic_joint_bounds") is True,
                "solve_time": solve_time,
                "iterations": iteration,
                "status": status,
            }
        )
    expected = len(commands)
    for field in (
        "qp_solve_count",
        "qp_postcheck_count",
        "joint_limit_postcheck_count",
        "issued_command_bound_check_count",
        "nominal_dynamic_bound_check_count",
    ):
        audit.check(psf.get(field) == expected, "psf_%s_differs" % field)
    audit.check(psf.get("nominal_dynamic_bound_violation_count") == 0, "psf_nominal_bound_violation")
    audit.check(psf.get("all_issued_commands_within_physical_bounds") is True, "psf_issued_bounds_false")
    audit.check(psf.get("all_nominal_commands_within_dynamic_joint_bounds") is True, "psf_nominal_bounds_false")
    audit.check(adapter.get("nominal_dynamic_bound_violation_count") == 0, "adapter_nominal_bound_violation")
    audit.check(adapter.get("all_issued_commands_within_physical_bounds") is True, "adapter_issued_bounds_false")
    audit.check(adapter.get("all_nominal_commands_within_dynamic_joint_bounds") is True, "adapter_nominal_bounds_false")
    if normalized:
        observed_minimum = min(row["safe_residual"] for row in normalized)
        stored_minimum = audit.number(psf.get("minimum_safe_cbf_residual_m2_per_s"), "psf_minimum_safe_residual")
        audit.check(_close(observed_minimum, stored_minimum), "psf_minimum_safe_residual_differs")
    pre_observations = audit.integer(psf.get("pre_filter_field_observation_count"), "psf_pre_field_observations")
    pre_queries = audit.integer(psf.get("pre_filter_field_query_count"), "psf_pre_field_queries")
    audit.check(pre_observations == expected, "psf_pre_field_observation_count_differs")
    audit.check(pre_queries == expected * sample_count, "psf_pre_field_query_count_differs")
    audit.check(psf.get("invalid_field_query_count") == 0, "psf_invalid_field_query")
    precontact_field_valid = bool(
        pre_observations == expected
        and pre_queries == expected * sample_count
        and psf.get("invalid_field_query_count") == 0
        and psf.get("nonpositive_post_state_field_query_count") == 0
    )
    qp_postchecks_passed = psf.get("qp_postcheck_count") == expected
    joint_limit_postchecks_passed = psf.get("joint_limit_postcheck_count") == expected
    qp_solved = bool(
        normalized
        and len(normalized) == expected
        and all(row["status"] in ("solved", "solved inaccurate") for row in normalized)
    )
    precontact_execution_valid = bool(
        qp_solved
        and qp_postchecks_passed
        and joint_limit_postchecks_passed
        and psf.get("issued_command_bound_check_count") == expected
        and psf.get("static_precontact_admissible") is True
        and precontact_field_valid
        and normalized
        and min(row["safe_residual"] for row in normalized) >= SAFE_RESIDUAL_MINIMUM
    )
    audit.check(
        psf.get("precontact_field_queries_valid_and_positive")
        is precontact_field_valid,
        "psf_precontact_field_valid_flag_differs",
    )
    audit.check(
        psf.get("precontact_execution_valid") is precontact_execution_valid,
        "psf_precontact_execution_valid_flag_differs",
    )
    if psf.get("exposure_complete") is True:
        audit.check(psf.get("post_state_field_observation_count") == 1, "psf_endpoint_field_observation_differs")
        audit.check(psf.get("post_state_field_query_count") == sample_count, "psf_endpoint_field_query_count_differs")
        audit.check(psf.get("nonpositive_post_state_field_query_count") == 0, "psf_endpoint_field_nonpositive")
        full_field_valid = bool(
            psf.get("post_state_field_observation_count") == 1
            and psf.get("post_state_field_query_count") == sample_count
            and psf.get("nonpositive_post_state_field_query_count") == 0
            and psf.get("invalid_field_query_count") == 0
        )
        audit.check(
            psf.get("all_post_state_field_queries_valid_and_positive")
            is full_field_valid,
            "psf_full_field_valid_flag_differs",
        )
    return normalized


def _material_activation(
    audit: _Audit,
    commands: Sequence[Mapping[str, Any]],
    psf: Mapping[str, Any],
    adapter_contact_boundary: Optional[int],
) -> List[Mapping[str, Any]]:
    if adapter_contact_boundary is None:
        return []
    material = [
        row
        for row in commands
        if isinstance(row.get("boundary"), int)
        and int(row["boundary"]) < int(adapter_contact_boundary)
        and float(row["nominal_residual"]) <= ACTIVATION_RESIDUAL_MAXIMUM
        and float(row["correction"]) >= CORRECTION_MINIMUM
        and row.get("nominal_bounds") is True
    ]
    trace = [
        audit.mapping(row, "activation_trace_row")
        for row in audit.sequence(psf.get("activation_trace"), "psf_activation_trace")
    ]
    trace_material = [
        row
        for row in trace
        if isinstance(row.get("physical_boundary"), int)
        and int(row["physical_boundary"]) < int(adapter_contact_boundary)
        and audit.number(row.get("filter_correction_l2_rad_s"), "activation_correction")
        >= CORRECTION_MINIMUM
        and row.get("nominal_within_dynamic_joint_bounds") is True
    ]
    expected_keys = [(row["source"], row["inner"]) for row in material]
    observed_keys = [
        (row.get("source_action_index"), row.get("inner_control_index"))
        for row in trace_material
    ]
    audit.check(observed_keys == expected_keys, "material_activation_trace_differs")
    for row in trace_material:
        sample = audit.mapping(row.get("argmin_protected_sample"), "activation_argmin_sample")
        audit.check(
            sample.get("body_name") in ("robot0_link5", "robot0_link6"),
            "activation_argmin_outside_link56",
        )
    return material


def _active_motion(
    audit: _Audit,
    material: Sequence[Mapping[str, Any]],
    physics_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    keys = {(row["source"], row["inner"]) for row in material}
    physics_by_key: Dict[Tuple[Any, Any], List[Mapping[str, Any]]] = {}
    for row in physics_rows:
        key = (row.get("source_action_index"), row.get("inner_control_index"))
        if key in keys:
            physics_by_key.setdefault(key, []).append(row)
    nominal_integral = sum(_norm(row["nominal"]) * INNER_DT_S for row in material)
    safe_integral = sum(_norm(row["executed"]) * INNER_DT_S for row in material)
    measured_integral = 0.0
    cartesian_path = 0.0
    target_progress = 0.0
    complete = 0
    for command in material:
        key = (command["source"], command["inner"])
        rows = sorted(
            physics_by_key.get(key, []),
            key=lambda row: int(row.get("physics_substep_index", -1)),
        )
        if len(rows) != SUBSTEPS_PER_FILTER or [row.get("physics_substep_index") for row in rows] != list(range(5)):
            continue
        previous = _vector(
            audit,
            command["raw"].get("eef_position_before_update_world_m"),
            3,
            "active_eef_start",
        )
        for row in rows:
            measured = _vector(audit, row.get("measured_qvel_rad_s"), 7, "active_measured_qvel")
            measured_integral += _norm(measured) * PHYSICS_DT_S
            current = _vector(audit, row.get("eef_position_world_m"), 3, "active_eef_position")
            cartesian_path += _distance(previous, current)
            previous = current
        target_start = audit.number(command["raw"].get("target_error_before_update_m"), "active_target_start")
        target_end = audit.number(rows[-1].get("target_error_after_substep_m"), "active_target_end")
        target_progress += target_start - target_end
        complete += 1
    ratio = 0.0 if nominal_integral == 0.0 else safe_integral / nominal_integral
    return {
        "active_update_count": len(material),
        "complete_active_interval_count": complete,
        "maximum_filter_correction_norm_rad_s": max(
            (float(row["correction"]) for row in material), default=0.0
        ),
        "maximum_executed_command_norm_rad_s": max(
            (_norm(row["executed"]) for row in material), default=0.0
        ),
        "safe_to_nominal_command_motion_ratio": ratio,
        "measured_joint_motion_integral_rad": measured_integral,
        "cartesian_path_length_m": cartesian_path,
        "cartesian_target_progress_m_diagnostic_only": target_progress,
    }


def _tracking(
    audit: _Audit,
    rows: Sequence[Mapping[str, Any]],
    label: str,
) -> Dict[str, Any]:
    squared: List[float] = []
    absolute: List[float] = []
    for index, row in enumerate(rows):
        measured = _vector(audit, row.get("measured_qvel_rad_s"), 7, "%s_measured_%d" % (label, index))
        issued = _vector(audit, row.get("issued_qvel_rad_s"), 7, "%s_issued_%d" % (label, index))
        errors = tuple(a - b for a, b in zip(measured, issued))
        stored_errors = _vector(
            audit,
            row.get("tracking_error_rad_s"),
            7,
            "%s_stored_error_%d" % (label, index),
        )
        audit.check(
            all(_close(left, right) for left, right in zip(errors, stored_errors)),
            "%s_stored_error_%d_differs" % (label, index),
        )
        absolute.extend(abs(value) for value in errors)
        squared.extend(value * value for value in errors)
    linf = max(absolute, default=0.0)
    rmse = math.sqrt(sum(squared) / len(squared)) if squared else 0.0
    return {
        "observation_count": len(rows),
        "linf_rad_s": linf,
        "rmse_rad_s": rmse,
        "within_registered_thresholds": bool(
            rows and linf <= TRACKING_LINF_MAXIMUM and rmse <= TRACKING_RMSE_MAXIMUM
        ),
        "diagnostic_only": True,
    }


def _compare_record(
    audit: _Audit,
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    fields: Iterable[str],
    label: str,
) -> None:
    for field in fields:
        left = observed.get(field)
        right = expected.get(field)
        if isinstance(right, float):
            left_number = audit.number(left, "%s_%s" % (label, field))
            audit.check(_close(left_number, right), "%s_%s_differs" % (label, field))
        else:
            audit.check(left == right, "%s_%s_differs" % (label, field))


def validate_payload(
    payload_value: Mapping[str, Any],
    *,
    expected_commit: str = EXPECTED_COMMIT,
    expected_job_id: Optional[str] = None,
    result_path: Optional[Path] = None,
) -> Dict[str, Any]:
    audit = _Audit()
    payload = audit.mapping(payload_value, "result")
    audit.check(payload.get("schema_version") == RESULT_SCHEMA, "result_schema_differs")
    audit.check(payload.get("status") == "complete", "result_status_not_complete")
    audit.check(payload.get("partial_output_interpreted") is False, "partial_output_interpreted")
    audit.check(payload.get("case_id") == EXPECTED_CASE, "case_id_differs")
    audit.check(payload.get("claim_scope") == EXPECTED_CLAIM_SCOPE, "claim_scope_differs")
    if result_path is not None:
        audit.check(payload.get("run_id") == result_path.parent.name, "run_id_path_differs")
    provenance = audit.mapping(payload.get("provenance"), "provenance")
    source = audit.mapping(provenance.get("source"), "provenance_source")
    allocation = audit.mapping(provenance.get("allocation"), "provenance_allocation")
    audit.check(source.get("commit") == expected_commit, "source_commit_differs")
    audit.check(source.get("branch") == EXPECTED_BRANCH, "source_branch_differs")
    audit.check(provenance.get("case_id") == payload.get("case_id"), "provenance_case_differs")
    audit.check(provenance.get("run_id") == payload.get("run_id"), "provenance_run_differs")
    audit.check(provenance.get("protocol_file_sha256") == EXPECTED_PROTOCOL_FILE_SHA256, "protocol_file_hash_differs")
    audit.check(provenance.get("historical_result_file_sha256") == EXPECTED_HISTORICAL_FILE_SHA256, "historical_file_hash_differs")
    audit.check(provenance.get("historical_result_payload_sha256") == EXPECTED_HISTORICAL_PAYLOAD_SHA256, "historical_payload_hash_differs")
    audit.check(provenance.get("historical_executed_action_sequence_sha256") == EXPECTED_ACTION_SEQUENCE_SHA256, "historical_action_hash_differs")
    audit.check(provenance.get("online_policy_query_count") == 0, "online_policy_queries_nonzero")
    audit.check(
        allocation.get("gpu_name") == "NVIDIA H100 80GB HBM3",
        "allocation_gpu_not_exact_h100",
    )
    audit.check(bool(allocation.get("host")), "allocation_host_missing")
    audit.check(str(allocation.get("slurm_job_id", "")).isdigit(), "slurm_job_id_invalid")
    if expected_job_id is not None:
        audit.check(str(allocation.get("slurm_job_id")) == str(expected_job_id), "slurm_job_id_differs")
    exploratory = audit.mapping(provenance.get("exploratory_execution"), "exploratory_execution")
    audit.check(exploratory.get("solver_claim") == "offline_contact_feasibility_not_realtime_100hz", "solver_claim_differs")

    window = audit.mapping(payload.get("window"), "window")
    audit.check(window.get("start_boundary") == START_BOUNDARY, "window_start_differs")
    audit.check(window.get("end_boundary") == END_BOUNDARY, "window_end_differs")
    audit.check(list(window.get("source_action_indexes", ())) == list(EXPECTED_ACTIONS), "window_actions_differ")
    audit.check(window.get("post_action_179_flattened_state_sha256") == EXPECTED_BOUNDARY_B_SHA256, "boundary_B_flattened_hash_differs")
    boundary_raw_hash = _sha256(
        audit,
        window.get("official_boundary_B_raw_bytes_sha256"),
        "boundary_B_raw_hash",
    )
    audit.check(
        boundary_raw_hash == EXPECTED_BOUNDARY_B_RAW_SHA256,
        "boundary_B_raw_hash_differs",
    )
    static = audit.mapping(window.get("settled_to_B_static_field_assumption"), "static_assumption")
    audit.check(static.get("admissible") is True, "static_field_assumption_invalid")

    field = audit.mapping(payload.get("field"), "field")
    resolved = audit.mapping(field.get("resolved_geometry"), "resolved_geometry")
    link_geom_ids = [
        audit.integer(value, "link56_geom_id")
        for value in audit.sequence(resolved.get("link56_geom_ids"), "link56_geom_ids")
    ]
    robot_geom_ids = [
        audit.integer(value, "robot_geom_id")
        for value in audit.sequence(resolved.get("robot_geom_ids"), "robot_geom_ids")
    ]
    obstacle_geom_ids = [
        audit.integer(value, "obstacle_geom_id")
        for value in audit.sequence(
            resolved.get("obstacle_geom_ids"), "obstacle_geom_ids"
        )
    ]
    audit.check(bool(link_geom_ids), "link56_geom_population_empty")
    audit.check(bool(robot_geom_ids), "robot_geom_population_empty")
    audit.check(bool(obstacle_geom_ids), "obstacle_geom_population_empty")
    bundle_hashes = audit.mapping(field.get("bundle_hashes"), "field_bundle_hashes")
    for hash_name, expected_hash in (
        ("bundle_sha256", EXPECTED_FIELD_BUNDLE_SHA256),
        ("protocol_sha256", EXPECTED_FIELD_PROTOCOL_SHA256),
        ("parameter_block_sha256", EXPECTED_FIELD_PARAMETER_BLOCK_SHA256),
    ):
        observed_hash = _sha256(
            audit,
            bundle_hashes.get(hash_name),
            "field_%s" % hash_name,
        )
        audit.check(
            observed_hash == expected_hash,
            "field_%s_differs" % hash_name,
        )
    full_robot_sampling = audit.mapping(
        field.get("full_robot_sampling"), "full_robot_sampling"
    )
    observed_sampling_hash = hashlib.sha256(
        _canonical(full_robot_sampling)
    ).hexdigest()
    audit.check(
        observed_sampling_hash == EXPECTED_FULL_ROBOT_SAMPLING_SHA256,
        "full_robot_sampling_hash_differs",
    )
    sampling_epsilon = audit.number(
        full_robot_sampling.get("epsilon_m"), "full_robot_sampling_epsilon"
    )
    audit.check(_close(sampling_epsilon, 0.05), "full_robot_sampling_epsilon_differs")
    sampling_records = [
        audit.mapping(value, "full_robot_sampling_record_%d" % index)
        for index, value in enumerate(
            audit.sequence(
                full_robot_sampling.get("geom_records"),
                "full_robot_sampling_records",
            )
        )
    ]
    sampling_geom_ids: List[int] = []
    sampling_radii: List[float] = []
    for index, record in enumerate(sampling_records):
        sampling_geom_ids.append(
            audit.integer(record.get("geom_id"), "sampling_geom_id_%d" % index)
        )
        radius = audit.number(
            record.get("certified_surface_cover_radius_m"),
            "sampling_radius_%d" % index,
        )
        sampling_radii.append(radius)
        audit.check(
            0.0 < radius <= sampling_epsilon,
            "sampling_radius_%d_out_of_range" % index,
        )
        audit.check(
            record.get("selection_authority")
            == "authoritative_resolved_geom_ids",
            "sampling_authority_%d_differs" % index,
        )
        audit.check(
            record.get("mask_collision_enabled") is True,
            "sampling_collision_mask_%d_disabled" % index,
        )
    audit.check(
        sorted(sampling_geom_ids) == sorted(robot_geom_ids)
        and len(sampling_geom_ids) == len(set(sampling_geom_ids)),
        "full_robot_sampling_geom_population_differs",
    )
    certified_full_robot_radius = max(sampling_radii, default=0.0)
    arms = audit.mapping(payload.get("arms"), "arms")
    adapter = audit.mapping(arms.get("adapter_only"), "adapter_arm")
    psf = audit.mapping(arms.get("adapter_plus_psf"), "psf_arm")
    audit.check(adapter.get("arm_name") == "joint_velocity_adapter_only", "adapter_arm_name_differs")
    audit.check(psf.get("arm_name") == "joint_velocity_adapter_plus_link56_psf", "psf_arm_name_differs")
    _validate_pairing(audit, adapter, psf, boundary_raw_hash)
    adapter_contact = _reconstruct_contacts(
        audit,
        adapter,
        link_geom_ids,
        robot_geom_ids,
        obstacle_geom_ids,
        "adapter",
    )
    psf_contact = _reconstruct_contacts(
        audit,
        psf,
        link_geom_ids,
        robot_geom_ids,
        obstacle_geom_ids,
        "psf",
    )
    adapter_commands, adapter_physics = _validate_arm_cadence(
        audit, adapter, "adapter", contact_present=adapter_contact["any_present"], must_complete=True
    )
    psf_commands_raw, psf_physics = _validate_arm_cadence(
        audit,
        psf,
        "psf",
        contact_present=psf_contact["any_present"],
        must_complete=not psf_contact["any_present"],
    )
    _validate_exogenous_pairing(audit, adapter_commands, psf_commands_raw)
    _validate_adapter_execution(audit, adapter, adapter_commands)
    audit.check(adapter_contact["link_present"], "adapter_link56_contact_missing")
    audit.check(
        adapter_contact["first_link_boundary"] == adapter_contact["first_any_boundary"],
        "adapter_first_contact_not_link56",
    )
    audit.check(adapter.get("static_precontact_admissible") is True, "adapter_precontact_static_invalid")
    audit.check(psf.get("static_precontact_admissible") is True, "psf_precontact_static_invalid")
    if not psf_contact["any_present"]:
        audit.check(psf.get("static_full_window_admissible") is True, "psf_full_static_invalid")

    psf_commands = _validate_qp(audit, adapter, psf, psf_commands_raw)
    material = _material_activation(
        audit, psf_commands, psf, adapter_contact["first_link_boundary"]
    )
    first_activation = material[0]["boundary"] if material else None
    if material:
        audit.check(
            first_activation < adapter_contact["first_link_boundary"],
            "activation_not_before_adapter_contact",
        )
    activation = audit.mapping(payload.get("activation_evidence"), "activation_evidence")
    audit.check(activation.get("adapter_contact_physical_boundary") == adapter_contact["first_link_boundary"], "activation_adapter_contact_boundary_differs")
    audit.check(activation.get("material_activation_count") == len(material), "activation_count_differs")
    if material:
        first_record = audit.mapping(activation.get("first_material_activation"), "first_material_activation")
        audit.check(first_record.get("physical_boundary") == first_activation, "first_activation_boundary_differs")

    clearance = audit.mapping(psf.get("conservative_full_robot_clearance"), "psf_clearance")
    clearance_values = [
        audit.number(row.get("cumulative_full_robot_surface_clearance_lower_bound_m"), "psf_clearance_trace")
        for row in psf_physics
    ]
    for previous, current in zip(clearance_values, clearance_values[1:]):
        audit.check(current <= previous + FLOAT_TOLERANCE, "clearance_trace_not_cumulative_minimum")
    clearance_minimum = min(clearance_values) if clearance_values else 0.0
    stored_clearance = audit.number(clearance.get("minimum_full_surface_lower_bound_m"), "stored_clearance")
    measurement_clearance = audit.mapping(
        audit.mapping(psf.get("measurement"), "psf_measurement").get("sample_clearance"),
        "psf_sample_clearance",
    )
    measurement_minimum = audit.number(
        measurement_clearance.get("full_surface_clearance_lower_bound_m"),
        "measurement_clearance_minimum",
    )
    audit.check(_close(clearance_minimum, stored_clearance), "stored_clearance_differs")
    audit.check(_close(clearance_minimum, measurement_minimum), "measurement_clearance_differs")
    audit.check(
        _close(
            clearance_minimum,
            audit.number(psf.get("minimum_D_sim_m_diagnostic_only"), "psf_minimum_D_sim"),
        ),
        "psf_minimum_D_sim_differs",
    )
    audit.check(
        _close(
            clearance_minimum,
            audit.number(
                audit.mapping(psf.get("measurement"), "psf_measurement").get(
                    "D_sim_min_m"
                ),
                "psf_measurement_D_sim",
            ),
        ),
        "psf_measurement_D_sim_differs",
    )
    audit.check(clearance.get("available") is True, "clearance_unavailable")
    audit.check(measurement_clearance.get("available") is True, "measurement_clearance_unavailable")
    audit.check(
        measurement_clearance.get("authority")
        == "exact_sample_to_obb_plus_certified_coverage_lower_bound",
        "measurement_clearance_authority_differs",
    )
    exact_distance = audit.number(
        measurement_clearance.get("minimum_exact_sample_to_obb_distance_m"),
        "measurement_exact_sample_distance",
    )
    coverage_radius = audit.number(
        measurement_clearance.get("certified_coverage_radius_m"),
        "measurement_coverage_radius",
    )
    audit.check(
        _close(coverage_radius, certified_full_robot_radius),
        "measurement_coverage_radius_certificate_differs",
    )
    audit.check(
        _close(exact_distance - coverage_radius, measurement_minimum),
        "measurement_clearance_formula_differs",
    )

    active_motion = _active_motion(audit, material, psf_physics)
    stored_motion = audit.mapping(activation.get("active_interval_motion"), "stored_active_motion")
    _compare_record(
        audit,
        stored_motion,
        active_motion,
        (
            "active_update_count",
            "complete_active_interval_count",
            "maximum_filter_correction_norm_rad_s",
            "maximum_executed_command_norm_rad_s",
            "safe_to_nominal_command_motion_ratio",
            "measured_joint_motion_integral_rad",
            "cartesian_path_length_m",
            "cartesian_target_progress_m_diagnostic_only",
        ),
        "active_motion",
    )

    adapter_precontact_rows = [
        row
        for row in adapter_physics
        if isinstance(row.get("post_state_physical_boundary"), int)
        and adapter_contact["first_link_boundary"] is not None
        and int(row["post_state_physical_boundary"])
        < int(adapter_contact["first_link_boundary"])
    ]
    adapter_tracking = _tracking(audit, adapter_precontact_rows, "adapter_tracking")
    psf_tracking = _tracking(audit, psf_physics, "psf_tracking")
    _compare_record(
        audit,
        audit.mapping(adapter.get("tracking_precontact"), "stored_adapter_tracking"),
        adapter_tracking,
        ("observation_count", "linf_rad_s", "rmse_rad_s"),
        "adapter_tracking",
    )
    _compare_record(
        audit,
        audit.mapping(psf.get("tracking_full_window"), "stored_psf_tracking"),
        psf_tracking,
        ("observation_count", "linf_rad_s", "rmse_rad_s"),
        "psf_tracking",
    )

    timing = audit.mapping(payload.get("timing"), "timing")
    started = audit.number(timing.get("started_unix"), "timing_started")
    finished = audit.number(timing.get("finished_unix"), "timing_finished")
    elapsed = audit.number(timing.get("elapsed_seconds"), "timing_elapsed")
    audit.check(finished >= started and elapsed >= 0.0, "timing_order_invalid")
    audit.check(abs((finished - started) - elapsed) <= 1.0, "timing_elapsed_differs")
    solve_times = [float(row["solve_time"]) for row in psf_commands]
    iteration_values = [int(row["iterations"]) for row in psf_commands]
    status_counts = Counter(str(row["status"]) for row in psf_commands)
    timing_diagnostic = {
        "total_run_elapsed_seconds": elapsed,
        "qp_count": len(solve_times),
        "qp_solver_status_counts": dict(sorted(status_counts.items())),
        "qp_solve_time_sum_seconds": sum(solve_times),
        "qp_solve_time_mean_seconds": (
            sum(solve_times) / len(solve_times) if solve_times else 0.0
        ),
        "qp_solve_time_max_seconds": max(solve_times, default=0.0),
        "qp_solve_time_p95_seconds": _percentile(solve_times, 0.95),
        "qp_iterations_max": max(iteration_values, default=0),
        "qp_iterations_p95": _percentile(iteration_values, 0.95),
        "qp_solver_time_over_10ms_count": sum(value > INNER_DT_S for value in solve_times),
        "realtime_100hz_claim_supported": False,
        "realtime_limitation": (
            "osqp_solve_time_excludes_field_jacobian_controller_and_simulator_work"
        ),
    }

    baseline_reproduced = bool(
        adapter_contact["link_present"]
        and adapter_contact["first_link_boundary"] == adapter_contact["first_any_boundary"]
    )
    psf_complete_no_contact = bool(
        not psf_contact["any_present"]
        and len(psf_commands_raw) == EXPECTED_FILTER_UPDATES
        and len(psf_physics) == EXPECTED_PHYSICS_SUBSTEPS
    )
    empirical_candidate = bool(
        baseline_reproduced and psf_complete_no_contact and material
    )
    clearance_candidate = bool(
        empirical_candidate and clearance.get("available") is True and clearance_minimum > 0.0
    )
    motion_checks = {
        "complete_active_intervals": bool(
            active_motion["active_update_count"] > 0
            and active_motion["active_update_count"]
            == active_motion["complete_active_interval_count"]
        ),
        "material_filter_correction": active_motion[
            "maximum_filter_correction_norm_rad_s"
        ]
        >= CORRECTION_MINIMUM,
        "material_safe_command": active_motion[
            "maximum_executed_command_norm_rad_s"
        ]
        >= SAFE_COMMAND_MINIMUM,
        "material_command_retention": active_motion[
            "safe_to_nominal_command_motion_ratio"
        ]
        >= MOTION_RATIO_MINIMUM,
        "material_realized_joint_motion": active_motion[
            "measured_joint_motion_integral_rad"
        ]
        >= MEASURED_MOTION_MINIMUM,
        "material_cartesian_motion": active_motion["cartesian_path_length_m"]
        >= CARTESIAN_PATH_MINIMUM,
    }
    useful_candidate = bool(empirical_candidate and all(motion_checks.values()))
    stop_candidate = bool(empirical_candidate and not useful_candidate)

    metrics = audit.mapping(payload.get("metrics"), "stored_metrics")
    numeric_metrics = {
        "adapter_first_link56_contact_physical_boundary": adapter_contact[
            "first_link_boundary"
        ],
        "psf_first_activation_physical_boundary": first_activation,
        "psf_activation_update_count_before_adapter_contact": len(material),
        "psf_conservative_full_robot_surface_clearance_lower_bound_m": clearance_minimum,
        "psf_minimum_D_sim_m": clearance_minimum,
        "active_cartesian_path_length_m": active_motion["cartesian_path_length_m"],
        "active_safe_measured_joint_motion_integral_rad": active_motion[
            "measured_joint_motion_integral_rad"
        ],
        "active_safe_to_nominal_command_motion_ratio": active_motion[
            "safe_to_nominal_command_motion_ratio"
        ],
        "maximum_active_filter_correction_norm_rad_s": active_motion[
            "maximum_filter_correction_norm_rad_s"
        ],
        "maximum_active_safe_command_norm_rad_s": active_motion[
            "maximum_executed_command_norm_rad_s"
        ],
        "adapter_precontact_tracking_linf_rad_s": adapter_tracking["linf_rad_s"],
        "adapter_precontact_tracking_rmse_rad_s": adapter_tracking["rmse_rad_s"],
        "psf_tracking_linf_rad_s": psf_tracking["linf_rad_s"],
        "psf_tracking_rmse_rad_s": psf_tracking["rmse_rad_s"],
    }
    for field_name, expected_value in numeric_metrics.items():
        observed_value = metrics.get(field_name)
        if expected_value is None:
            audit.check(observed_value is None, "metric_%s_differs" % field_name)
        elif isinstance(expected_value, int):
            audit.check(observed_value == expected_value, "metric_%s_differs" % field_name)
        else:
            audit.check(
                _close(
                    audit.number(observed_value, "metric_%s" % field_name),
                    float(expected_value),
                ),
                "metric_%s_differs" % field_name,
            )
    boolean_metrics = {
        "adapter_link56_contact_present": adapter_contact["link_present"],
        "adapter_first_selected_obstacle_contact_is_link56": bool(
            adapter_contact["link_present"]
            and adapter_contact["first_link_boundary"]
            == adapter_contact["first_any_boundary"]
        ),
        "psf_any_robot_selected_obstacle_contact_present": psf_contact["any_present"],
        "psf_link56_contact_present": psf_contact["link_present"],
        "psf_nominal_cbf_activation_with_material_correction_before_adapter_contact": bool(
            material
        ),
        "psf_active_interval_motion_complete": motion_checks[
            "complete_active_intervals"
        ],
        "psf_conservative_full_robot_clearance_lower_bound_available": bool(
            clearance.get("available") is True
        ),
        "psf_all_qp_solved": bool(
            psf_commands
            and all(
                row["status"] in ("solved", "solved inaccurate")
                for row in psf_commands
            )
        ),
        "psf_all_qp_postchecks_passed": bool(
            psf.get("qp_postcheck_count") == len(psf_commands)
        ),
        "psf_all_joint_limit_postchecks_passed": bool(
            psf.get("joint_limit_postcheck_count") == len(psf_commands)
        ),
        "psf_precontact_field_queries_valid_and_positive": bool(
            psf.get("pre_filter_field_observation_count") == len(psf_commands)
            and psf.get("pre_filter_field_query_count")
            == len(psf_commands) * psf.get("protected_sample_count", 0)
            and psf.get("invalid_field_query_count") == 0
            and psf.get("nonpositive_post_state_field_query_count") == 0
        ),
        "psf_precontact_execution_valid": bool(
            psf.get("precontact_execution_valid") is True
        ),
        "psf_all_post_state_field_queries_valid_and_positive": bool(
            psf.get("post_state_field_observation_count") == 1
            and psf.get("post_state_field_query_count")
            == psf.get("protected_sample_count")
            and psf.get("nonpositive_post_state_field_query_count") == 0
            and psf.get("invalid_field_query_count") == 0
        ),
    }
    for field_name, expected_value in boolean_metrics.items():
        audit.check(
            metrics.get(field_name) is expected_value,
            "metric_%s_differs" % field_name,
        )

    producer = audit.mapping(payload.get("classification"), "producer_classification")
    if not baseline_reproduced:
        expected_outcome = "INCONCLUSIVE"
        expected_mechanism = "NOT_APPLICABLE"
    elif psf_contact["any_present"]:
        expected_outcome = "CONTACT_PREVENTION_FAILED"
        expected_mechanism = "NOT_APPLICABLE"
    elif not empirical_candidate:
        expected_outcome = "INCONCLUSIVE"
        expected_mechanism = "NOT_APPLICABLE"
    elif not clearance_candidate:
        expected_outcome = "UNCERTIFIED_CLEARANCE"
        expected_mechanism = "NOT_APPLICABLE"
    else:
        expected_outcome = "CONTACT_PREVENTION_FEASIBLE"
        expected_mechanism = (
            "MOTION_PRESERVING_CORRECTION" if useful_candidate else "STOP_ONLY"
        )
    audit.check(producer.get("primary_outcome") == expected_outcome, "producer_primary_outcome_differs")
    audit.check(producer.get("safety_mechanism") == expected_mechanism, "producer_safety_mechanism_differs")
    expected_feasible = expected_outcome == "CONTACT_PREVENTION_FEASIBLE"
    expected_failed = expected_outcome == "CONTACT_PREVENTION_FAILED"
    expected_uncertified = expected_outcome == "UNCERTIFIED_CLEARANCE"
    expected_motion_preserving = expected_mechanism == "MOTION_PRESERVING_CORRECTION"
    expected_stop_only = expected_mechanism == "STOP_ONLY"
    for field_name, expected_value in (
        ("contact_prevention_feasible", expected_feasible),
        ("contact_prevention_failed", expected_failed),
        ("uncertified_clearance", expected_uncertified),
        ("motion_preserving_correction", expected_motion_preserving),
        ("stop_only", expected_stop_only),
        ("tracking_diagnostic_only", True),
    ):
        audit.check(
            producer.get(field_name) is expected_value,
            "producer_%s_differs" % field_name,
        )
    producer_motion_checks = audit.mapping(
        producer.get("motion_preservation_checks"),
        "producer_motion_checks",
    )
    for field_name, expected_value in motion_checks.items():
        if field_name == "complete_active_intervals":
            continue
        audit.check(
            producer_motion_checks.get(field_name) is expected_value,
            "producer_motion_check_%s_differs" % field_name,
        )

    valid = not audit.discrepancies
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "validated" if valid else "rejected",
        "artifact_valid": valid,
        "run_id": payload.get("run_id"),
        "case_id": payload.get("case_id"),
        "result_payload_sha256": payload.get("result_payload_sha256"),
        "empirical_contact_prevention": empirical_candidate if valid else None,
        "contact_prevention_failed": (
            bool(baseline_reproduced and psf_contact["any_present"])
            if valid
            else None
        ),
        "clearance_certified": clearance_candidate if valid else None,
        "useful_motion": useful_candidate if valid else None,
        "stop_only": stop_candidate if valid else None,
        "adapter_first_contact_physical_boundary": adapter_contact[
            "first_link_boundary"
        ],
        "first_material_activation_physical_boundary": first_activation,
        "minimum_conservative_full_robot_clearance_m": clearance_minimum,
        "motion_checks": motion_checks,
        "active_interval_motion": active_motion,
        "tracking_diagnostic": {
            "adapter_precontact": adapter_tracking,
            "psf_full_window": psf_tracking,
            "tracking_certified": bool(
                adapter_tracking["within_registered_thresholds"]
                and psf_tracking["within_registered_thresholds"]
            ),
            "outcome_gate": False,
        },
        "timing_diagnostic": timing_diagnostic,
        "producer_classification": {
            "primary_outcome": producer.get("primary_outcome"),
            "safety_mechanism": producer.get("safety_mechanism"),
        },
        "discrepancies": audit.discrepancies,
    }


def validate_artifact(
    result_path: Path,
    *,
    expected_commit: str = EXPECTED_COMMIT,
    expected_job_id: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(result_path)
    payload = load_hashed_json(path)
    summary = validate_payload(
        payload,
        expected_commit=expected_commit,
        expected_job_id=expected_job_id,
        result_path=path,
    )
    summary["result_file_sha256"] = sha256_file(path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--expected-commit", default=EXPECTED_COMMIT)
    parser.add_argument("--expected-job-id")
    arguments = parser.parse_args()
    try:
        summary = validate_artifact(
            arguments.result,
            expected_commit=arguments.expected_commit,
            expected_job_id=arguments.expected_job_id,
        )
    except (ArtifactContractError, OSError, ValueError, TypeError) as error:
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "status": "rejected",
            "artifact_valid": False,
            "discrepancies": ["artifact_load_failed:%s" % error],
        }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if summary.get("artifact_valid") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
