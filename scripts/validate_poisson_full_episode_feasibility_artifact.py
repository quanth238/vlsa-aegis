#!/usr/bin/env python3
"""Independent consumer for the fixed-suffix full-episode Poisson experiment.

The producer classifier is deliberately not imported.  This consumer rebuilds
the paired exposure, literal-contact outcome, hard-QP outcome, native BDDL task
outcome, and useful-motion outcome from the terminal traces.  It accepts only
the explicitly versioned full-episode result and protocol contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import stat
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
from scripts import validate_poisson_fast_feasibility_artifact as narrow  # noqa: E402


SUMMARY_SCHEMA = "vlsa_poisson_full_episode_independent_validation.v1"
RESULT_SCHEMA = "vlsa_poisson_full_episode_feasibility_result.v1"
PROTOCOL_SCHEMA = "vlsa_poisson_full_episode_feasibility_protocol.v1"
METRICS_SCHEMA = "vlsa_poisson_full_episode_feasibility_metrics.v1"
CLASSIFICATION_SCHEMA = "vlsa_poisson_full_episode_feasibility_classification.v1"
PROTOCOL_ID = "vlsa-poisson-link56-full-episode-hybrid-v1"
EXPECTED_CASE = "vlsa-t1-goal-ii-t0-e05"
EXPECTED_BRANCH = "codex/poisson-fast-feasibility"
EXPECTED_SOURCE_ARM = "pi05_plus_aegis_translational"

EXPECTED_HISTORICAL_FILE_SHA256 = (
    "273d77cba3fd5b1e457817ad20f8572b5628aeaa8b5b9f768b32e4f095fbad8b"
)
EXPECTED_HISTORICAL_PAYLOAD_SHA256 = (
    "ec58c8581297751de33756e76efbf5b18e24a5f836aa6a8f147d0caebdd92d1c"
)
EXPECTED_ACTION_SEQUENCE_SHA256 = (
    "e5e2df4efee667fee0b0b7596dbc203a3cce84e67cb3507cb839132d2ee67481"
)
EXPECTED_BOUNDARY_STATE_SHA256 = (
    "2b4d25f12da2e1c110883b9f2194c9790270f36911f449d8c4f058a039c90335"
)
EXPECTED_BOUNDARY_RAW_SHA256 = (
    "ff1a9a80d03178c9ac85e5dc6750aac98f920a050462df0b17d7ca1cdbb069c7"
)
EXPECTED_SUFFIX_RECORD_SHA256 = (
    "fdbddb5e98c0d4872773a9c980c7efd5bef0acc14b961ee785a5fb1323ed0dc7"
)
EXPECTED_SUFFIX_ACTION_ARRAY_SHA256 = (
    "30f914fb87dbf6313afa0a8998179ecc24cf289556e95ace7a3132298da6f7ac"
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
EXPECTED_CASE_ROW_SHA256 = (
    "ee42bf1e8587acf2fd4fe73cb05d7974de836a37f53acc8035e0995b1d1fca9e"
)

ALLOWED_CLASSIFICATIONS = (
    "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
    "CONTACT_PREVENTED_TASK_FAILED",
    "STOP_ONLY",
    "CONTACT_REMAINS_OR_SHIFTED",
    "INCONCLUSIVE_APPARATUS",
)
METRIC_KEYS = (
    "schema_version",
    "exact_paired_start",
    "shared_prefix_complete",
    "full_recorded_episode_complete",
    "adapter_exposure_complete",
    "psf_exposure_complete",
    "adapter_physics_monitor_trace_counts_match",
    "psf_physics_monitor_trace_counts_match",
    "adapter_filter_update_count",
    "psf_filter_update_count",
    "adapter_physics_substep_count",
    "psf_physics_substep_count",
    "adapter_completed_suffix_action_count",
    "psf_completed_suffix_action_count",
    "psf_qp_count_complete",
    "psf_qp_postchecks_complete",
    "psf_joint_limit_postchecks_complete",
    "all_issued_commands_within_physical_bounds",
    "both_nominal_commands_within_dynamic_joint_bounds",
    "psf_invalid_field_query_count",
    "psf_all_post_state_field_queries_valid_and_positive",
    "static_selected_obstacle_admissible",
    "boundary_goal_unsatisfied",
    "adapter_link56_contact_present",
    "adapter_first_selected_obstacle_contact_is_link56",
    "psf_link56_contact_present",
    "psf_any_robot_selected_obstacle_contact_present",
    "psf_clearance_certified",
    "material_correction_before_adapter_contact",
    "first_material_correction_physical_boundary",
    "adapter_first_link56_contact_physical_boundary",
    "material_correction_update_count",
    "maximum_correction_norm_rad_s",
    "filter_correction_integral_rad",
    "post_correction_measured_joint_motion_integral_rad",
    "post_correction_cartesian_path_length_m",
    "post_correction_executed_command_integral_rad",
    "post_correction_zero_command_fraction",
    "psf_task_success_ever",
    "psf_terminal_task_success",
    "psf_first_task_success_source_action_index",
    "psf_task_success_after_material_correction",
    "adapter_task_success_ever",
    "adapter_terminal_task_success",
)
EXPECTED_CLAIM_SCOPE = (
    "one_offline_single_case_hybrid_full_recorded_episode_controller_"
    "feasibility_not_closed_loop_policy_not_population_safety_not_realtime"
)

# These are frozen protocol facts.  `_protocol_expectations` independently
# derives every exposure count and boundary from the primitive protocol fields;
# the validator never trusts compact producer counts as its source of truth.
FROZEN_PREFIX_START_ACTION = 0
FROZEN_PREFIX_END_ACTION = 179
FROZEN_SUFFIX_START_ACTION = 180
FROZEN_SUFFIX_END_ACTION = 236
FROZEN_CONTROLS_PER_ACTION = 5
FROZEN_SUBSTEPS_PER_CONTROL = 5
FROZEN_CONTROL_DT_S = 0.01
FROZEN_PHYSICS_DT_S = 0.002

SAFE_RESIDUAL_MINIMUM = -5.0e-7
ACTIVATION_RESIDUAL_MAXIMUM = -5.0e-7
CORRECTION_MINIMUM = 1.0e-4
BOUND_TOLERANCE = 5.0e-8
FLOAT_TOLERANCE = 1.0e-10


_Audit = narrow._Audit
_canonical = narrow._canonical
_close = narrow._close
_sha256 = narrow._sha256
_norm = narrow._norm
_distance = narrow._distance
_vector = narrow._vector


def _load_plain_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactContractError("protocol is missing, nonregular, or symlinked")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ArtifactContractError("protocol must contain one object")
    return value


def _protocol_expectations(protocol_value: Mapping[str, Any]) -> Dict[str, Any]:
    """Strictly decode primitive protocol facts and derive exposure counts."""

    protocol = dict(protocol_value)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ArtifactContractError("full-episode protocol schema differs")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ArtifactContractError("full-episode protocol id differs")
    case = protocol.get("case")
    if not isinstance(case, Mapping) or case.get("case_id") != EXPECTED_CASE:
        raise ArtifactContractError("full-episode protocol case differs")
    if case.get("selected_obstacle_name") != "moka_pot_obstacle_1":
        raise ArtifactContractError("full-episode selected obstacle differs")
    if list(case.get("protected_robot_body_names", ())) != [
        "robot0_link5",
        "robot0_link6",
    ]:
        raise ArtifactContractError("full-episode protected links differ")
    source = protocol.get("source")
    episode = protocol.get("episode")
    cadence = protocol.get("cadence")
    acceptance = protocol.get("acceptance")
    if not isinstance(source, Mapping):
        raise ArtifactContractError("full-episode source block is absent")
    if not isinstance(episode, Mapping):
        raise ArtifactContractError("full-episode episode block is absent")
    if not isinstance(cadence, Mapping):
        raise ArtifactContractError("full-episode cadence block is absent")
    if not isinstance(acceptance, Mapping):
        raise ArtifactContractError("full-episode acceptance block is absent")

    def integer(mapping: Mapping[str, Any], field: str) -> int:
        value = mapping.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ArtifactContractError("protocol %s must be an integer" % field)
        return int(value)

    prefix_start = integer(episode, "shared_prefix_start_action_index")
    prefix_end = integer(episode, "shared_prefix_end_action_index_inclusive")
    suffix_start = integer(episode, "suffix_start_action_index")
    suffix_end = integer(episode, "suffix_end_action_index_inclusive")
    controls = integer(cadence, "inner_updates_per_high_level_action")
    substeps = integer(cadence, "physics_substeps_per_inner_update")
    if (
        prefix_start != FROZEN_PREFIX_START_ACTION
        or prefix_end != FROZEN_PREFIX_END_ACTION
        or suffix_start != FROZEN_SUFFIX_START_ACTION
        or suffix_end != FROZEN_SUFFIX_END_ACTION
        or controls != FROZEN_CONTROLS_PER_ACTION
        or substeps != FROZEN_SUBSTEPS_PER_CONTROL
    ):
        raise ArtifactContractError("full-episode protocol primitive differs")
    if prefix_end + 1 != suffix_start:
        raise ArtifactContractError("prefix and suffix are not contiguous")

    control_hz = integer(cadence, "filter_frequency_hz")
    physics_hz = integer(cadence, "physics_frequency_hz")
    if (
        cadence.get("high_level_frequency_hz") != 20
        or control_hz != 100
        or physics_hz != 500
    ):
        raise ArtifactContractError("full-episode frequencies differ")
    control_dt = 1.0 / control_hz if control_hz else math.inf
    physics_dt = cadence.get("physics_timestep_s")
    if (
        isinstance(control_dt, bool)
        or not isinstance(control_dt, (int, float))
        or not math.isclose(float(control_dt), FROZEN_CONTROL_DT_S)
        or isinstance(physics_dt, bool)
        or not isinstance(physics_dt, (int, float))
        or not math.isclose(float(physics_dt), FROZEN_PHYSICS_DT_S)
    ):
        raise ArtifactContractError("full-episode protocol cadence differs")

    action_count = suffix_end - suffix_start + 1
    filter_updates = action_count * controls
    physics_substeps = filter_updates * substeps
    start_boundary = suffix_start * controls * substeps
    end_boundary = (suffix_end + 1) * controls * substeps
    prefix_action_count = prefix_end - prefix_start + 1
    if (prefix_action_count, start_boundary, action_count, filter_updates, physics_substeps, end_boundary) != (
        180,
        4500,
        57,
        285,
        1425,
        5925,
    ):
        raise ArtifactContractError("derived full-episode exposure differs")
    if cadence.get("expected_filter_updates_per_arm") != filter_updates:
        raise ArtifactContractError("stored expected filter-update count differs")
    if cadence.get("expected_physics_substeps_per_arm") != physics_substeps:
        raise ArtifactContractError("stored expected physics-substep count differs")
    if episode.get("branch_physical_boundary") != start_boundary:
        raise ArtifactContractError("stored branch boundary differs")
    if episode.get("terminal_physical_boundary") != end_boundary:
        raise ArtifactContractError("stored terminal boundary differs")
    if episode.get("no_event_physical_boundary_sentinel") != end_boundary + 1:
        raise ArtifactContractError("stored no-event boundary sentinel differs")
    if episode.get("shared_prefix_action_count") != prefix_action_count:
        raise ArtifactContractError("stored prefix count differs")
    if episode.get("suffix_action_count") != action_count:
        raise ArtifactContractError("stored suffix count differs")
    if episode.get("recorded_full_episode_action_count") != 237:
        raise ArtifactContractError("stored full-episode count differs")
    if episode.get("psf_contact_policy") != (
        "terminate_as_valid_scientific_negative_without_unsafe_h_nonpositive_fallback"
    ):
        raise ArtifactContractError("PSF contact policy differs")
    expected_boundary_goal_values = episode.get("expected_boundary_goal_values")
    if expected_boundary_goal_values != [False]:
        raise ArtifactContractError("boundary native-goal vector differs")
    expected_source = {
        "historical_result_file_sha256": EXPECTED_HISTORICAL_FILE_SHA256,
        "historical_result_payload_sha256": EXPECTED_HISTORICAL_PAYLOAD_SHA256,
        "historical_executed_action_sequence_sha256": EXPECTED_ACTION_SEQUENCE_SHA256,
        "suffix_action_record_sha256": EXPECTED_SUFFIX_RECORD_SHA256,
        "suffix_action_array_sha256": EXPECTED_SUFFIX_ACTION_ARRAY_SHA256,
        "post_action_179_flattened_state_sha256": EXPECTED_BOUNDARY_STATE_SHA256,
    }
    for field, expected in expected_source.items():
        if source.get(field) != expected:
            raise ArtifactContractError("protocol source %s differs" % field)
    thresholds = {
        "minimum_safe_cbf_residual_m2_per_s": float(
            acceptance.get("minimum_safe_cbf_residual_m2_per_s")
        ),
        "maximum_nominal_cbf_residual_for_activation_m2_per_s": float(
            acceptance.get(
                "maximum_nominal_cbf_residual_for_activation_m2_per_s"
            )
        ),
        "maximum_invalid_field_queries": integer(
            acceptance, "maximum_invalid_field_queries"
        ),
        "minimum_filter_correction_norm_rad_s": float(
            acceptance.get("minimum_filter_correction_norm_rad_s")
        ),
        "minimum_filter_correction_integral_rad": float(
            acceptance.get("minimum_filter_correction_integral_rad")
        ),
        "minimum_post_correction_measured_joint_motion_integral_rad": float(
            acceptance.get(
                "minimum_post_correction_measured_joint_motion_integral_rad"
            )
        ),
        "minimum_post_correction_cartesian_path_length_m": float(
            acceptance.get("minimum_post_correction_cartesian_path_length_m")
        ),
        "minimum_post_correction_executed_command_integral_rad": float(
            acceptance.get(
                "minimum_post_correction_executed_command_integral_rad"
            )
        ),
        "maximum_post_correction_zero_command_fraction": float(
            acceptance.get("maximum_post_correction_zero_command_fraction")
        ),
    }
    if not all(math.isfinite(value) for value in thresholds.values()):
        raise ArtifactContractError("acceptance threshold is nonfinite")
    return {
        "prefix_start": prefix_start,
        "prefix_end": prefix_end,
        "prefix_action_count": prefix_action_count,
        "suffix_start": suffix_start,
        "suffix_end": suffix_end,
        "source_actions": tuple(range(suffix_start, suffix_end + 1)),
        "action_count": action_count,
        "controls_per_action": controls,
        "substeps_per_control": substeps,
        "filter_updates": filter_updates,
        "physics_substeps": physics_substeps,
        "start_boundary": start_boundary,
        "end_boundary": end_boundary,
        "no_event_boundary": end_boundary + 1,
        "control_dt_s": float(control_dt),
        "physics_dt_s": float(physics_dt),
        "expected_boundary_goal_values": tuple(expected_boundary_goal_values),
        "thresholds": thresholds,
    }


def _expected_command_keys(expectation: Mapping[str, Any]) -> List[Tuple[int, int]]:
    return [
        (
            int(expectation["suffix_start"]) + index // int(expectation["controls_per_action"]),
            index % int(expectation["controls_per_action"]),
        )
        for index in range(int(expectation["filter_updates"]))
    ]


def _expected_physics_keys(
    expectation: Mapping[str, Any],
) -> List[Tuple[int, int, int]]:
    controls = int(expectation["controls_per_action"])
    substeps = int(expectation["substeps_per_control"])
    per_action = controls * substeps
    return [
        (
            int(expectation["suffix_start"]) + index // per_action,
            (index % per_action) // substeps,
            index % substeps,
        )
        for index in range(int(expectation["physics_substeps"]))
    ]


def _contact_boundary(
    audit: _Audit,
    record: Mapping[str, Any],
    label: str,
    expectation: Mapping[str, Any],
) -> int:
    observation = audit.integer(record.get("observation_index"), "%s_observation" % label)
    phase = record.get("source_phase")
    if phase == "live_solver_phase_preintegration_geometry":
        return int(expectation["start_boundary"]) + observation
    if phase == "post_integration_recomputed":
        return int(expectation["start_boundary"]) + observation + 1
    audit.check(False, "%s_source_phase_unknown" % label)
    return int(expectation["end_boundary"]) + 1


def _reconstruct_contacts(
    audit: _Audit,
    arm: Mapping[str, Any],
    link_geom_ids: Iterable[int],
    robot_geom_ids: Iterable[int],
    obstacle_geom_ids: Iterable[int],
    label: str,
    expectation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Rebuild any-robot selected-obstacle contact from signed raw records."""

    measurement = audit.mapping(arm.get("measurement"), "%s_measurement" % label)
    settled = audit.mapping(measurement.get("settled_state"), "%s_settled" % label)
    narrow._validate_settled_contacts(
        audit,
        settled,
        link_geom_ids,
        robot_geom_ids,
        obstacle_geom_ids,
        label,
    )
    robot_set = {int(value) for value in robot_geom_ids}
    obstacle_set = {int(value) for value in obstacle_geom_ids}
    link_set = {int(value) for value in link_geom_ids}
    physics_count = int(expectation["physics_substeps"])
    controls = int(expectation["controls_per_action"])
    substeps = int(expectation["substeps_per_control"])
    per_action = controls * substeps

    post_candidates: List[Mapping[str, Any]] = []
    post_physical: List[Mapping[str, Any]] = []
    for index, value in enumerate(
        audit.sequence(
            measurement.get("post_state_candidate_contact_point_records"),
            "%s_post_candidates" % label,
        )
    ):
        row_label = "%s_post_candidate_%d" % (label, index)
        row = audit.mapping(value, row_label)
        distance = audit.number(row.get("contact_distance_m"), "%s_distance" % row_label)
        physical = distance <= 0.0
        audit.check(
            row.get("is_physical_nonpositive_distance_contact") is physical,
            "%s_physical_flag_differs" % row_label,
        )
        audit.check(
            row.get("source_phase") == "post_integration_recomputed",
            "%s_phase_differs" % row_label,
        )
        audit.check(
            audit.integer(row.get("robot_geom_id"), "%s_robot_geom" % row_label)
            in robot_set,
            "%s_robot_geom_outside_authority" % row_label,
        )
        audit.check(
            audit.integer(row.get("obstacle_geom_id"), "%s_obstacle_geom" % row_label)
            in obstacle_set,
            "%s_obstacle_geom_outside_authority" % row_label,
        )
        post_candidates.append(row)
        if physical:
            post_physical.append(row)
    stored_post_physical = [
        audit.mapping(value, "%s_post_physical_%d" % (label, index))
        for index, value in enumerate(
            audit.sequence(
                measurement.get("post_state_physical_contact_point_records"),
                "%s_post_physical" % label,
            )
        )
    ]
    audit.check(
        _canonical(stored_post_physical) == _canonical(post_physical),
        "%s_post_physical_ledger_differs" % label,
    )
    audit.check(
        measurement.get("post_state_candidate_contact_point_record_count")
        == len(post_candidates),
        "%s_post_candidate_count_differs" % label,
    )

    records: List[Mapping[str, Any]] = []
    live_records = audit.sequence(
        measurement.get("live_solver_phase_contact_point_records"),
        "%s_live_records" % label,
    )
    phase_rows = (
        (live_records, "live_solver_phase_preintegration_geometry", "live"),
        (stored_post_physical, "post_integration_recomputed", "post"),
    )
    phase_counts: Dict[str, int] = {}
    for values, expected_phase, phase_label in phase_rows:
        physical_count = 0
        for index, value in enumerate(values):
            row_label = "%s_%s_%d" % (label, phase_label, index)
            row = audit.mapping(value, row_label)
            distance = audit.number(row.get("contact_distance_m"), "%s_distance" % row_label)
            physical = distance <= 0.0
            audit.check(
                row.get("is_physical_nonpositive_distance_contact") is physical,
                "%s_physical_flag_differs" % row_label,
            )
            audit.check(row.get("source_phase") == expected_phase, "%s_phase_differs" % row_label)
            observation = audit.integer(row.get("observation_index"), "%s_observation" % row_label)
            audit.check(0 <= observation < physics_count, "%s_observation_out_of_range" % row_label)
            expected_index = [
                observation // per_action,
                (observation // substeps) % controls,
                observation % substeps,
            ]
            audit.check(
                [
                    row.get("high_level_index"),
                    row.get("inner_control_index"),
                    row.get("physics_substep_index"),
                ]
                == expected_index,
                "%s_index_differs" % row_label,
            )
            audit.check(
                audit.integer(row.get("robot_geom_id"), "%s_robot_geom" % row_label)
                in robot_set,
                "%s_robot_geom_outside_authority" % row_label,
            )
            audit.check(
                audit.integer(row.get("obstacle_geom_id"), "%s_obstacle_geom" % row_label)
                in obstacle_set,
                "%s_obstacle_geom_outside_authority" % row_label,
            )
            if physical:
                records.append(row)
                physical_count += 1
        phase_counts[phase_label] = physical_count

    live_count = phase_counts.get("live", 0)
    post_count = phase_counts.get("post", 0)
    for field, expected in (
        ("live_solver_candidate_contact_point_record_count", len(live_records)),
        ("live_solver_nonpositive_contact_point_record_count", live_count),
        ("post_state_physical_contact_point_record_count", post_count),
        ("rollout_phase_physical_contact_point_record_count", live_count + post_count),
        ("total_physical_contact_point_record_count", live_count + post_count),
        ("total_candidate_contact_point_record_count", len(live_records) + len(post_candidates)),
    ):
        audit.check(measurement.get(field) == expected, "%s_%s_differs" % (label, field))

    boundaries = [
        _contact_boundary(audit, row, "%s_contact" % label, expectation)
        for row in records
    ]
    link_boundaries = [
        _contact_boundary(audit, row, "%s_link_contact" % label, expectation)
        for row in records
        if audit.integer(row.get("robot_geom_id"), "%s_contact_robot_geom" % label)
        in link_set
    ]
    any_present = bool(boundaries)
    link_present = bool(link_boundaries)
    first_any = min(boundaries) if boundaries else None
    first_link = min(link_boundaries) if link_boundaries else None
    for field, expected in (
        ("any_robot_obstacle_contact", any_present),
        ("rollout_any_robot_obstacle_contact", any_present),
        ("live_solver_any_robot_obstacle_contact", bool(live_count)),
        ("post_state_any_robot_obstacle_contact", bool(post_count)),
        ("link56_obstacle_contact", link_present),
    ):
        audit.check(measurement.get(field) is expected, "%s_%s_differs" % (label, field))
    audit.check(
        measurement.get("physical_contact_distance_semantics")
        == "mujoco_contact_dist_le_0",
        "%s_contact_semantics_differs" % label,
    )
    literal = audit.mapping(arm.get("literal_contact"), "%s_literal_contact" % label)
    for field, expected in (
        ("any_robot_selected_obstacle_present", any_present),
        ("link56_present", link_present),
        ("first_any_robot_physical_boundary", first_any),
        ("first_link56_physical_boundary", first_link),
    ):
        audit.check(literal.get(field) == expected, "%s_literal_%s_differs" % (label, field))
    physics_rows = audit.sequence(arm.get("physics_trace"), "%s_contact_physics" % label)
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
                    audit.integer(
                        row.get("observation_index"),
                        "%s_flag_observation" % label,
                    )
                    for row in records
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
        "minimum_contact_distance_m": (
            min(
                audit.number(
                    row.get("contact_distance_m"),
                    "%s_minimum_contact_distance" % label,
                )
                for row in records
            )
            if records
            else None
        ),
        "robot_geom_ids_contacted": sorted(
            {
                audit.integer(row.get("robot_geom_id"), "%s_contact_robot_geom" % label)
                for row in records
            }
        ),
    }


def _expected_d_sim(
    full_surface_clearance_lower_bound_m: float,
    minimum_physical_contact_distance_m: Optional[float],
) -> float:
    """Apply the monitor's signed-contact authority to sampled clearance."""

    if minimum_physical_contact_distance_m is None:
        return float(full_surface_clearance_lower_bound_m)
    return min(
        float(full_surface_clearance_lower_bound_m),
        float(minimum_physical_contact_distance_m),
        0.0,
    )


def _validate_cadence(
    audit: _Audit,
    arm: Mapping[str, Any],
    label: str,
    expectation: Mapping[str, Any],
    *,
    contact_present: bool,
    must_complete: bool,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    cadence = audit.mapping(arm.get("execution_cadence"), "%s_cadence" % label)
    for field, expected in (
        ("control_timestep_s", expectation["control_dt_s"]),
        ("wrapper_model_timestep_s", expectation["physics_dt_s"]),
        ("mujoco_model_timestep_s", expectation["physics_dt_s"]),
    ):
        audit.check(
            _close(audit.number(cadence.get(field), "%s_%s" % (label, field)), float(expected)),
            "%s_%s_differs" % (label, field),
        )
    commands = [
        audit.mapping(value, "%s_command_%d" % (label, index))
        for index, value in enumerate(
            audit.sequence(arm.get("command_trace"), "%s_commands" % label)
        )
    ]
    physics = [
        audit.mapping(value, "%s_physics_%d" % (label, index))
        for index, value in enumerate(
            audit.sequence(arm.get("physics_trace"), "%s_physics" % label)
        )
    ]
    expected_commands = int(expectation["filter_updates"])
    expected_physics = int(expectation["physics_substeps"])
    if must_complete:
        audit.check(len(commands) == expected_commands, "%s_update_count_differs" % label)
        audit.check(len(physics) == expected_physics, "%s_substep_count_differs" % label)
        audit.check(arm.get("exposure_complete") is True, "%s_exposure_incomplete" % label)
        audit.check(arm.get("contact_terminated_early") is False, "%s_contact_terminated_early" % label)
    elif contact_present:
        expected_prefix_commands = (
            (len(physics) + int(expectation["substeps_per_control"]) - 1)
            // int(expectation["substeps_per_control"])
            if physics
            else 0
        )
        audit.check(0 < len(commands) <= expected_commands, "%s_contact_prefix_update_count_invalid" % label)
        audit.check(0 < len(physics) <= expected_physics, "%s_contact_prefix_substep_count_invalid" % label)
        audit.check(len(commands) == expected_prefix_commands, "%s_contact_prefix_cadence_differs" % label)
        audit.check(arm.get("exposure_complete") is False, "%s_contact_prefix_marked_complete" % label)
        audit.check(arm.get("contact_terminated_early") is True, "%s_contact_not_terminal" % label)
    else:
        audit.check(False, "%s_incomplete_without_contact" % label)
    for field, expected in (
        ("filter_update_count", len(commands)),
        ("physics_substep_count", len(physics)),
        ("physics_trace_row_count", len(physics)),
        ("monitor_observed_physics_substep_count", len(physics)),
    ):
        audit.check(arm.get(field) == expected, "%s_%s_differs" % (label, field))
    measurement = audit.mapping(arm.get("measurement"), "%s_measurement" % label)
    audit.check(
        measurement.get("observed_physics_substeps") == len(physics),
        "%s_measurement_substep_count_differs" % label,
    )
    audit.check(arm.get("physics_monitor_trace_counts_match") is True, "%s_trace_count_match_false" % label)
    audit.check(
        list(arm.get("source_action_indexes", ())) == list(expectation["source_actions"]),
        "%s_source_actions_differ" % label,
    )

    observed_command_keys: List[Tuple[int, int]] = []
    for index, row in enumerate(commands):
        observed_command_keys.append(
            (
                audit.integer(row.get("source_action_index"), "%s_command_source" % label),
                audit.integer(row.get("inner_control_index"), "%s_command_inner" % label),
            )
        )
        audit.check(
            row.get("local_action_index")
            == index // int(expectation["controls_per_action"]),
            "%s_command_local_index_differs" % label,
        )
        audit.check(
            row.get("physical_boundary")
            == int(expectation["start_boundary"])
            + index * int(expectation["substeps_per_control"]),
            "%s_command_boundary_differs" % label,
        )
    audit.check(
        observed_command_keys == _expected_command_keys(expectation)[: len(commands)],
        "%s_command_population_differs" % label,
    )

    observed_physics_keys: List[Tuple[int, int, int]] = []
    substeps = int(expectation["substeps_per_control"])
    per_action = int(expectation["controls_per_action"]) * substeps
    for index, row in enumerate(physics):
        observed_physics_keys.append(
            (
                audit.integer(row.get("source_action_index"), "%s_physics_source" % label),
                audit.integer(row.get("inner_control_index"), "%s_physics_inner" % label),
                audit.integer(row.get("physics_substep_index"), "%s_physics_substep" % label),
            )
        )
        audit.check(row.get("observation_index") == index, "%s_physics_observation_differs" % label)
        audit.check(
            row.get("post_state_physical_boundary")
            == int(expectation["start_boundary"]) + index + 1,
            "%s_physics_boundary_differs" % label,
        )
        audit.check(row.get("local_action_index") == index // per_action, "%s_physics_local_index_differs" % label)
        command_index = index // substeps
        if command_index >= len(commands):
            audit.check(False, "%s_physics_without_command" % label)
        else:
            issued = _vector(audit, row.get("issued_qvel_rad_s"), 7, "%s_physics_issued" % label)
            executed = _vector(
                audit,
                commands[command_index].get("executed_qdot_rad_s"),
                7,
                "%s_command_executed_binding" % label,
            )
            audit.check(
                all(_close(left, right) for left, right in zip(issued, executed)),
                "%s_command_to_physics_binding_differs" % label,
            )
    audit.check(
        observed_physics_keys == _expected_physics_keys(expectation)[: len(physics)],
        "%s_physics_population_differs" % label,
    )
    if physics:
        audit.check(measurement.get("first_index") == [0, 0, 0], "%s_first_index_differs" % label)
        final_index = len(physics) - 1
        expected_last = [
            final_index
            // (
                int(expectation["controls_per_action"])
                * int(expectation["substeps_per_control"])
            ),
            (final_index // int(expectation["substeps_per_control"]))
            % int(expectation["controls_per_action"]),
            final_index % int(expectation["substeps_per_control"]),
        ]
        audit.check(
            measurement.get("last_index") == expected_last,
            "%s_last_index_differs" % label,
        )
    return commands, physics


def _validate_goal_definition(
    audit: _Audit, value: Any, label: str
) -> Tuple[Mapping[str, Any], Sequence[Mapping[str, Any]]]:
    definition = audit.mapping(value, "%s_definition" % label)
    audit.check(definition.get("schema_version") == "safelibero_goal_progress.v1", "%s_schema_differs" % label)
    audit.check(definition.get("source") == "native_bddl_goal_predicates", "%s_source_differs" % label)
    audit.check(definition.get("logic") == "conjunction", "%s_logic_differs" % label)
    atoms = [
        audit.mapping(value, "%s_atom_%d" % (label, index))
        for index, value in enumerate(audit.sequence(definition.get("goal_atoms"), "%s_atoms" % label))
    ]
    audit.check(bool(atoms), "%s_atoms_empty" % label)
    for index, atom in enumerate(atoms):
        audit.check(atom.get("index") == index, "%s_atom_index_differs" % label)
        audit.check(atom.get("predicate") in ("in", "on"), "%s_atom_predicate_invalid" % label)
        arguments = audit.sequence(atom.get("arguments"), "%s_atom_arguments" % label)
        audit.check(
            len(arguments) == 2 and all(isinstance(item, str) and item for item in arguments),
            "%s_atom_arguments_invalid" % label,
        )
    stored_hash = _sha256(audit, definition.get("goal_definition_sha256"), "%s_hash" % label)
    hash_payload = {key: val for key, val in definition.items() if key != "goal_definition_sha256"}
    audit.check(hashlib.sha256(_canonical(hash_payload)).hexdigest() == stored_hash, "%s_hash_differs" % label)
    return definition, atoms


def _validate_goal_snapshot(
    audit: _Audit,
    row: Mapping[str, Any],
    atoms: Sequence[Mapping[str, Any]],
    previous_values: Optional[Sequence[bool]],
    label: str,
    *,
    transition_metadata_available: bool,
) -> Tuple[bool, ...]:
    values_raw = audit.sequence(row.get("values"), "%s_values" % label)
    audit.check(
        len(values_raw) == len(atoms) and all(isinstance(value, bool) for value in values_raw),
        "%s_values_invalid" % label,
    )
    values = tuple(bool(value) for value in values_raw[: len(atoms)])
    satisfied = sum(values)
    audit.check(row.get("satisfied_count") == satisfied, "%s_satisfied_count_differs" % label)
    expected_fraction = satisfied / len(atoms) if atoms else 0.0
    audit.check(
        _close(audit.number(row.get("fraction"), "%s_fraction" % label), expected_fraction),
        "%s_fraction_differs" % label,
    )
    audit.check(row.get("all_satisfied") is all(values), "%s_all_satisfied_differs" % label)
    before = _sha256(audit, row.get("simulator_state_sha256_before"), "%s_state_before" % label)
    after = _sha256(audit, row.get("simulator_state_sha256_after"), "%s_state_after" % label)
    audit.check(before == after, "%s_state_not_inert" % label)
    audit.check(row.get("inert") is True, "%s_inert_flag_false" % label)
    poses = audit.sequence(row.get("argument_poses"), "%s_argument_poses" % label)
    audit.check(len(poses) == len(atoms), "%s_argument_pose_count_differs" % label)
    for index, (pose_value, atom) in enumerate(zip(poses, atoms)):
        pose = audit.mapping(pose_value, "%s_pose_%d" % (label, index))
        audit.check(pose.get("atom_index") == index, "%s_pose_index_differs" % label)
        arguments = audit.sequence(pose.get("arguments"), "%s_pose_arguments" % label)
        expected_names = list(atom.get("arguments", ()))
        audit.check(len(arguments) == 2, "%s_pose_argument_count_differs" % label)
        for argument_index, argument_value in enumerate(arguments):
            argument = audit.mapping(argument_value, "%s_pose_argument_%d" % (label, argument_index))
            if argument_index < len(expected_names):
                audit.check(argument.get("name") == expected_names[argument_index], "%s_pose_argument_name_differs" % label)
            audit.check(argument.get("object_state_type") in ("object", "site"), "%s_pose_argument_type_invalid" % label)
            _vector(audit, argument.get("position"), 3, "%s_pose_position" % label)
            _vector(audit, argument.get("quaternion"), 4, "%s_pose_quaternion" % label)
    if transition_metadata_available and previous_values is not None:
        newly = [index for index, (old, new) in enumerate(zip(previous_values, values)) if not old and new]
        regressed = [index for index, (old, new) in enumerate(zip(previous_values, values)) if old and not new]
        audit.check(list(row.get("newly_satisfied_indices", ())) == newly, "%s_newly_satisfied_differs" % label)
        audit.check(list(row.get("regressed_indices", ())) == regressed, "%s_regressed_differs" % label)
    else:
        audit.check(row.get("transition_metadata_available") is False, "%s_transition_availability_differs" % label)
        audit.check(list(row.get("newly_satisfied_indices", ())) == [], "%s_placeholder_newly_nonempty" % label)
        audit.check(list(row.get("regressed_indices", ())) == [], "%s_placeholder_regressed_nonempty" % label)
    return values


def _validate_task(
    audit: _Audit,
    arm: Mapping[str, Any],
    label: str,
    expectation: Mapping[str, Any],
    *,
    allow_contact_terminal: bool,
) -> Dict[str, Any]:
    task = audit.mapping(arm.get("task"), "%s_task" % label)
    audit.check(task.get("source") == "native_bddl_goal_predicates", "%s_task_source_differs" % label)
    definition, atoms = _validate_goal_definition(audit, task.get("goal_definition"), "%s_goal" % label)
    expected_boundary = tuple(expectation["expected_boundary_goal_values"])
    audit.check(tuple(task.get("expected_boundary_goal_values", ())) == expected_boundary, "%s_expected_boundary_goal_differs" % label)
    audit.check(task.get("boundary_goal_values_exact") is True, "%s_boundary_goal_not_exact" % label)
    boundary = audit.mapping(task.get("boundary_goal_snapshot"), "%s_boundary_goal" % label)
    audit.check(boundary.get("snapshot_kind") in (None, "branch_boundary_pre_action"), "%s_boundary_snapshot_kind_differs" % label)
    audit.check(boundary.get("step") == int(expectation["suffix_start"]) - 1, "%s_boundary_step_differs" % label)
    boundary_values = _validate_goal_snapshot(
        audit,
        boundary,
        atoms,
        None,
        "%s_boundary_goal" % label,
        transition_metadata_available=False,
    )
    audit.check(boundary_values == expected_boundary, "%s_boundary_values_differ" % label)
    audit.check(boundary.get("all_satisfied") is False, "%s_boundary_already_success" % label)

    ledger = [
        audit.mapping(value, "%s_goal_ledger_%d" % (label, index))
        for index, value in enumerate(
            audit.sequence(task.get("goal_progress_ledger"), "%s_goal_ledger" % label)
        )
    ]
    partial_terminal: Optional[Mapping[str, Any]] = None
    if ledger and ledger[-1].get("snapshot_kind") == "partial_action_terminal_state_diagnostic":
        partial_terminal = ledger[-1]
        audit.check(allow_contact_terminal, "%s_unexpected_partial_goal_terminal" % label)
        completed = ledger[1:-1]
    else:
        completed = ledger[1:]
    if partial_terminal is None:
        audit.check(
            len(ledger) == int(expectation["action_count"]) + 1,
            "%s_goal_ledger_count_differs" % label,
        )
    else:
        audit.check(
            0 <= len(completed) < int(expectation["action_count"]),
            "%s_partial_goal_ledger_count_invalid" % label,
        )
    if ledger:
        for field_name, expected_value in boundary.items():
            audit.check(
                _canonical(ledger[0].get(field_name)) == _canonical(expected_value),
                "%s_boundary_ledger_%s_differs" % (label, field_name),
            )
        for field_name, expected_value in (
            ("snapshot_kind", "branch_boundary_pre_action"),
            ("local_action_index", None),
            ("source_action_index", int(expectation["suffix_start"]) - 1),
            ("reward", None),
            ("returned_done", None),
            ("returned_info", None),
            ("returned_observation_sha256", None),
        ):
            audit.check(
                ledger[0].get(field_name) == expected_value,
                "%s_boundary_ledger_%s_differs" % (label, field_name),
            )
    previous = boundary_values
    rewards: List[float] = []
    observations: List[str] = []
    successful_steps: List[int] = []
    for local_index, row in enumerate(completed):
        row_label = "%s_goal_step_%d" % (label, local_index)
        source_index = int(expectation["suffix_start"]) + local_index
        audit.check(row.get("snapshot_kind") == "completed_high_level_post_step", "%s_snapshot_kind_differs" % row_label)
        audit.check(row.get("local_action_index") == local_index, "%s_local_index_differs" % row_label)
        audit.check(row.get("source_action_index") == source_index, "%s_source_index_differs" % row_label)
        audit.check(row.get("step") == source_index, "%s_step_differs" % row_label)
        values = _validate_goal_snapshot(
            audit,
            row,
            atoms,
            previous,
            row_label,
            transition_metadata_available=True,
        )
        previous = values
        reward = audit.number(row.get("reward"), "%s_reward" % row_label)
        rewards.append(reward)
        done = audit.boolean(row.get("returned_done"), "%s_done" % row_label)
        audit.check(done is all(values), "%s_done_goal_differs" % row_label)
        audit.mapping(row.get("returned_info"), "%s_info" % row_label)
        observation = _sha256(audit, row.get("returned_observation_sha256"), "%s_observation_hash" % row_label)
        observations.append(observation)
        if all(values):
            successful_steps.append(source_index)

    terminal_snapshot: Mapping[str, Any] = completed[-1] if completed else boundary
    if partial_terminal is not None:
        partial_source = int(expectation["suffix_start"]) + len(completed)
        audit.check(
            partial_terminal.get("local_action_index") == len(completed),
            "%s_partial_local_index_differs" % label,
        )
        audit.check(
            partial_terminal.get("source_action_index") == partial_source,
            "%s_partial_source_index_differs" % label,
        )
        audit.check(
            partial_terminal.get("step") == partial_source,
            "%s_partial_step_differs" % label,
        )
        partial_values = _validate_goal_snapshot(
            audit,
            partial_terminal,
            atoms,
            previous,
            "%s_partial_goal" % label,
            transition_metadata_available=True,
        )
        audit.check(partial_terminal.get("reward") is None, "%s_partial_reward_present" % label)
        audit.check(partial_terminal.get("returned_done") is None, "%s_partial_done_present" % label)
        audit.check(partial_terminal.get("returned_info") is None, "%s_partial_info_present" % label)
        _sha256(
            audit,
            partial_terminal.get("returned_observation_sha256"),
            "%s_partial_observation_hash" % label,
        )
        if all(partial_values):
            successful_steps.append(partial_source)
        terminal_snapshot = partial_terminal

    audit.check(task.get("registered_source_action_count") == int(expectation["action_count"]), "%s_registered_action_count_differs" % label)
    audit.check(task.get("completed_source_action_count") == len(completed), "%s_completed_action_count_differs" % label)
    fixed_exposure_complete = bool(
        partial_terminal is None
        and len(completed) == int(expectation["action_count"])
    )
    audit.check(
        task.get("fixed_exposure_complete") is fixed_exposure_complete,
        "%s_fixed_exposure_flag_differs" % label,
    )
    audit.check(task.get("continue_fixed_exposure_after_success") is True, "%s_success_changed_exposure" % label)
    audit.check(task.get("initial_task_success_at_branch") is False, "%s_initial_success_differs" % label)
    first_success = successful_steps[0] if successful_steps else None
    terminal_success = bool(terminal_snapshot.get("all_satisfied") is True)
    ever_success = bool(successful_steps)
    audit.check(task.get("ever_task_success_at_or_after_branch") is ever_success, "%s_ever_success_differs" % label)
    audit.check(task.get("first_task_success_source_action_index") == first_success, "%s_first_success_differs" % label)
    audit.check(task.get("terminal_task_success") is terminal_success, "%s_terminal_success_differs" % label)
    terminal_fraction = audit.number(
        terminal_snapshot.get("fraction"), "%s_terminal_fraction" % label
    )
    audit.check(_close(audit.number(task.get("terminal_goal_fraction"), "%s_stored_terminal_fraction" % label), terminal_fraction), "%s_terminal_fraction_differs" % label)
    audit.check(_close(audit.number(task.get("reward_sum"), "%s_reward_sum" % label), sum(rewards)), "%s_reward_sum_differs" % label)
    expected_terminal_reward = rewards[-1] if rewards else None
    if expected_terminal_reward is None:
        audit.check(task.get("terminal_reward") is None, "%s_terminal_reward_differs" % label)
    else:
        audit.check(_close(audit.number(task.get("terminal_reward"), "%s_terminal_reward" % label), expected_terminal_reward), "%s_terminal_reward_differs" % label)
    audit.check(list(task.get("returned_observation_sha256_ledger", ())) == observations, "%s_observation_ledger_differs" % label)
    terminal_state = _sha256(audit, task.get("terminal_simulator_state_sha256"), "%s_terminal_state" % label)
    terminal_official = _sha256(audit, task.get("terminal_official_integration_state_raw_bytes_sha256"), "%s_terminal_official_state" % label)
    terminal_observation = _sha256(audit, task.get("terminal_observation_sha256"), "%s_terminal_observation" % label)
    audit.check(
        terminal_state == terminal_snapshot.get("simulator_state_sha256_after"),
        "%s_terminal_state_goal_differs" % label,
    )
    if partial_terminal is not None:
        audit.check(
            terminal_observation
            == partial_terminal.get("returned_observation_sha256"),
            "%s_partial_terminal_observation_differs" % label,
        )
    return {
        "definition": definition,
        "terminal_success": terminal_success,
        "ever_success": ever_success,
        "first_success_source_action_index": first_success,
        "reward_sum": sum(rewards),
        "terminal_reward": expected_terminal_reward,
        "terminal_fraction": terminal_fraction,
        "terminal_state_sha256": terminal_state,
        "terminal_official_state_sha256": terminal_official,
        "terminal_observation_sha256": terminal_observation,
        "completed_action_count": len(completed),
        "fixed_exposure_complete": fixed_exposure_complete,
        "partial_contact_terminal": partial_terminal is not None,
    }


def _post_correction_motion(
    audit: _Audit,
    commands: Sequence[Mapping[str, Any]],
    physics: Sequence[Mapping[str, Any]],
    first_correction_boundary: Optional[int],
    expectation: Mapping[str, Any],
) -> Dict[str, Any]:
    if first_correction_boundary is None:
        return {
            "first_correction_physical_boundary": int(
                expectation["no_event_boundary"]
            ),
            "command_update_count": 0,
            "physics_substep_count": 0,
            "complete_command_interval_count": 0,
            "maximum_correction_norm_rad_s": 0.0,
            "filter_correction_integral_rad": 0.0,
            "executed_command_integral_rad": 0.0,
            "measured_joint_motion_integral_rad": 0.0,
            "cartesian_path_length_m": 0.0,
            "zero_command_fraction": 1.0,
        }
    selected_commands = [
        row
        for row in commands
        if audit.integer(row.get("physical_boundary"), "post_correction_boundary")
        >= int(first_correction_boundary)
    ]
    selected_keys = {
        (
            audit.integer(row.get("source_action_index"), "post_correction_source"),
            audit.integer(row.get("inner_control_index"), "post_correction_inner"),
        )
        for row in selected_commands
    }
    selected_physics = [
        row
        for row in physics
        if (
            audit.integer(row.get("source_action_index"), "post_physics_source"),
            audit.integer(row.get("inner_control_index"), "post_physics_inner"),
        )
        in selected_keys
    ]
    corrections = [
        audit.number(row.get("correction_l2_rad_s"), "post_correction_norm")
        for row in selected_commands
    ]
    executed_vectors = [
        _vector(audit, row.get("executed_qdot_rad_s"), 7, "post_correction_executed")
        for row in selected_commands
    ]
    correction_integral = sum(
        value * float(expectation["control_dt_s"]) for value in corrections
    )
    command_integral = sum(
        _norm(value) * float(expectation["control_dt_s"])
        for value in executed_vectors
    )
    measured_integral = sum(
        _norm(_vector(audit, row.get("measured_qvel_rad_s"), 7, "post_correction_measured"))
        * float(expectation["physics_dt_s"])
        for row in selected_physics
    )
    cartesian_path = 0.0
    complete_intervals = 0
    for command in selected_commands:
        key = (
            audit.integer(command.get("source_action_index"), "post_path_source"),
            audit.integer(command.get("inner_control_index"), "post_path_inner"),
        )
        rows = sorted(
            (
                row
                for row in selected_physics
                if (row.get("source_action_index"), row.get("inner_control_index"))
                == key
            ),
            key=lambda row: audit.integer(
                row.get("physics_substep_index"), "post_path_substep"
            ),
        )
        if len(rows) != int(expectation["substeps_per_control"]):
            continue
        previous = _vector(
            audit,
            command.get("eef_position_before_update_world_m"),
            3,
            "post_path_start",
        )
        for row in rows:
            current = _vector(
                audit, row.get("eef_position_world_m"), 3, "post_path_eef"
            )
            cartesian_path += _distance(previous, current)
            previous = current
        complete_intervals += 1
    zero_count = sum(_norm(value) <= 1.0e-8 for value in executed_vectors)
    return {
        "first_correction_physical_boundary": int(first_correction_boundary),
        "command_update_count": len(selected_commands),
        "physics_substep_count": len(selected_physics),
        "complete_command_interval_count": complete_intervals,
        "maximum_correction_norm_rad_s": max(corrections, default=0.0),
        "filter_correction_integral_rad": correction_integral,
        "executed_command_integral_rad": command_integral,
        "measured_joint_motion_integral_rad": measured_integral,
        "cartesian_path_length_m": cartesian_path,
        "zero_command_fraction": (
            1.0 if not selected_commands else float(zero_count) / len(selected_commands)
        ),
    }


def _validate_frozen_source_actions(
    audit: _Audit,
    adapter_commands: Sequence[Mapping[str, Any]],
    psf_commands: Sequence[Mapping[str, Any]],
    expectation: Mapping[str, Any],
) -> None:
    """Bind every entered controller update to the frozen suffix actions."""

    reconstructed: List[List[float]] = []
    for source_index in expectation["source_actions"]:
        adapter_rows = [
            row
            for row in adapter_commands
            if row.get("source_action_index") == source_index
        ]
        psf_rows = [
            row
            for row in psf_commands
            if row.get("source_action_index") == source_index
        ]
        # A contact-negative PSF prefix may not enter later source actions;
        # the adapter arm is complete and remains the array authority.
        audit.check(
            len(adapter_rows) == int(expectation["controls_per_action"]),
            "adapter_source_action_update_count_differs",
        )
        adapter_values = [
            _vector(audit, row.get("source_action"), 7, "adapter_source_action")
            for row in adapter_rows
        ]
        if adapter_values:
            first = adapter_values[0]
            audit.check(
                all(_canonical(value) == _canonical(first) for value in adapter_values),
                "adapter_source_action_changed_within_high_level_action",
            )
            audit.check(
                all(value == 0.0 for value in first[3:6]),
                "frozen_source_action_contains_rotation",
            )
            reconstructed.append(list(first))
        if psf_rows:
            psf_values = [
                _vector(audit, row.get("source_action"), 7, "psf_source_action")
                for row in psf_rows
            ]
            audit.check(
                all(
                    _canonical(value) == _canonical(adapter_values[0])
                    for value in psf_values
                ),
                "paired_source_action_differs",
            )
    audit.check(
        hashlib.sha256(_canonical(reconstructed)).hexdigest()
        == EXPECTED_SUFFIX_ACTION_ARRAY_SHA256,
        "reconstructed_suffix_action_array_hash_differs",
    )


def _validate_field_and_clearance(
    audit: _Audit,
    field: Mapping[str, Any],
    robot_geom_ids: Sequence[int],
    psf: Mapping[str, Any],
    psf_physics: Sequence[Mapping[str, Any]],
    *,
    contact_present: bool,
    minimum_contact_distance_m: Optional[float],
) -> bool:
    bundle_hashes = audit.mapping(field.get("bundle_hashes"), "field_bundle_hashes")
    for name, expected in (
        ("bundle_sha256", EXPECTED_FIELD_BUNDLE_SHA256),
        ("protocol_sha256", EXPECTED_FIELD_PROTOCOL_SHA256),
        ("parameter_block_sha256", EXPECTED_FIELD_PARAMETER_BLOCK_SHA256),
    ):
        observed = _sha256(audit, bundle_hashes.get(name), "field_%s" % name)
        audit.check(observed == expected, "field_%s_differs" % name)
    sampling = audit.mapping(field.get("full_robot_sampling"), "full_robot_sampling")
    audit.check(
        hashlib.sha256(_canonical(sampling)).hexdigest()
        == EXPECTED_FULL_ROBOT_SAMPLING_SHA256,
        "full_robot_sampling_hash_differs",
    )
    epsilon = audit.number(sampling.get("epsilon_m"), "sampling_epsilon")
    audit.check(_close(epsilon, 0.05), "sampling_epsilon_differs")
    records = [
        audit.mapping(value, "sampling_record_%d" % index)
        for index, value in enumerate(
            audit.sequence(sampling.get("geom_records"), "sampling_records")
        )
    ]
    ids: List[int] = []
    radii: List[float] = []
    for index, row in enumerate(records):
        ids.append(audit.integer(row.get("geom_id"), "sampling_geom_%d" % index))
        radius = audit.number(
            row.get("certified_surface_cover_radius_m"),
            "sampling_radius_%d" % index,
        )
        radii.append(radius)
        audit.check(0.0 < radius <= epsilon, "sampling_radius_%d_invalid" % index)
        audit.check(
            row.get("selection_authority") == "authoritative_resolved_geom_ids",
            "sampling_authority_%d_differs" % index,
        )
        audit.check(
            row.get("mask_collision_enabled") is True,
            "sampling_mask_%d_disabled" % index,
        )
    audit.check(
        sorted(ids) == sorted(robot_geom_ids) and len(ids) == len(set(ids)),
        "sampling_robot_geom_population_differs",
    )
    certified_radius = max(radii, default=0.0)
    clearance = audit.mapping(
        psf.get("conservative_full_robot_clearance"), "psf_clearance"
    )
    trace_values = [
        audit.number(
            row.get("cumulative_full_robot_surface_clearance_lower_bound_m"),
            "psf_clearance_trace_%d" % index,
        )
        for index, row in enumerate(psf_physics)
    ]
    for previous, current in zip(trace_values, trace_values[1:]):
        audit.check(
            current <= previous + FLOAT_TOLERANCE,
            "psf_clearance_trace_not_cumulative_minimum",
        )
    minimum = min(trace_values) if trace_values else 0.0
    stored = audit.number(
        clearance.get("minimum_full_surface_lower_bound_m"),
        "psf_stored_clearance",
    )
    measurement = audit.mapping(psf.get("measurement"), "psf_measurement")
    sample_clearance = audit.mapping(
        measurement.get("sample_clearance"), "psf_sample_clearance"
    )
    measured = audit.number(
        sample_clearance.get("full_surface_clearance_lower_bound_m"),
        "psf_measured_clearance",
    )
    audit.check(_close(minimum, stored), "psf_stored_clearance_differs")
    audit.check(_close(minimum, measured), "psf_measured_clearance_differs")
    arm_d_sim = audit.number(psf.get("minimum_D_sim_m_diagnostic_only"), "psf_D_sim")
    measured_d_sim = audit.number(
        measurement.get("D_sim_min_m"), "psf_measurement_D_sim"
    )
    audit.check(
        (minimum_contact_distance_m is not None) is contact_present,
        "psf_contact_distance_presence_differs",
    )
    expected_d_sim = _expected_d_sim(minimum, minimum_contact_distance_m)
    audit.check(_close(arm_d_sim, expected_d_sim), "psf_D_sim_differs")
    audit.check(
        _close(measured_d_sim, expected_d_sim),
        "psf_measurement_D_sim_differs",
    )
    audit.check(
        measurement.get("contact_authority_clamped_D_sim")
        is bool(contact_present and expected_d_sim < minimum),
        "psf_D_sim_contact_clamp_flag_differs",
    )
    audit.check(clearance.get("available") is True, "psf_clearance_unavailable")
    audit.check(sample_clearance.get("available") is True, "psf_sample_clearance_unavailable")
    audit.check(
        sample_clearance.get("authority")
        == "exact_sample_to_obb_plus_certified_coverage_lower_bound",
        "psf_clearance_authority_differs",
    )
    exact = audit.number(
        sample_clearance.get("minimum_exact_sample_to_obb_distance_m"),
        "psf_exact_sample_distance",
    )
    observed_radius = audit.number(
        sample_clearance.get("certified_coverage_radius_m"),
        "psf_coverage_radius",
    )
    audit.check(
        _close(observed_radius, certified_radius),
        "psf_coverage_radius_certificate_differs",
    )
    audit.check(_close(exact - observed_radius, measured), "psf_clearance_formula_differs")
    return bool(clearance.get("available") is True and stored > 0.0)


def _independent_classification(
    metrics: Mapping[str, Any], expectation: Mapping[str, Any]
) -> Dict[str, Any]:
    """Reproduce classifier-v1 without importing producer outcome code."""

    expected_updates = int(expectation["filter_updates"])
    expected_substeps = int(expectation["physics_substeps"])
    expected_actions = int(expectation["action_count"])
    thresholds = expectation["thresholds"]
    psf_contact_present = bool(
        metrics["psf_link56_contact_present"]
        or metrics["psf_any_robot_selected_obstacle_contact_present"]
    )
    apparatus_reasons: List[str] = []
    for field, reason in (
        ("exact_paired_start", "paired_start_not_exact"),
        ("shared_prefix_complete", "shared_prefix_incomplete"),
        ("adapter_exposure_complete", "adapter_exposure_incomplete"),
        (
            "adapter_physics_monitor_trace_counts_match",
            "adapter_physics_monitor_trace_count_mismatch",
        ),
        (
            "psf_physics_monitor_trace_counts_match",
            "psf_physics_monitor_trace_count_mismatch",
        ),
        (
            "all_issued_commands_within_physical_bounds",
            "issued_command_physical_bounds_failed",
        ),
        (
            "both_nominal_commands_within_dynamic_joint_bounds",
            "nominal_dynamic_joint_bounds_confounded",
        ),
        ("static_selected_obstacle_admissible", "static_obstacle_inadmissible"),
        ("boundary_goal_unsatisfied", "task_was_already_complete_at_branch"),
    ):
        if not metrics[field]:
            apparatus_reasons.append(reason)
    for field, expected in (
        ("adapter_filter_update_count", expected_updates),
        ("adapter_physics_substep_count", expected_substeps),
        ("adapter_completed_suffix_action_count", expected_actions),
    ):
        if metrics[field] != expected:
            apparatus_reasons.append("%s_differs" % field)
    if metrics["psf_invalid_field_query_count"] != thresholds[
        "maximum_invalid_field_queries"
    ]:
        apparatus_reasons.append("psf_invalid_field_query_count_differs")
    if psf_contact_present:
        if not 0 < metrics["psf_filter_update_count"] <= expected_updates:
            apparatus_reasons.append("psf_contact_prefix_filter_count_invalid")
        if not 0 < metrics["psf_physics_substep_count"] <= expected_substeps:
            apparatus_reasons.append("psf_contact_prefix_physics_count_invalid")
        if not 0 <= metrics["psf_completed_suffix_action_count"] <= expected_actions:
            apparatus_reasons.append("psf_contact_prefix_action_count_invalid")
    else:
        for field, reason in (
            ("full_recorded_episode_complete", "full_recorded_episode_incomplete"),
            ("psf_exposure_complete", "psf_exposure_incomplete"),
            ("psf_qp_count_complete", "psf_qp_count_incomplete"),
            ("psf_qp_postchecks_complete", "psf_qp_postchecks_incomplete"),
            (
                "psf_joint_limit_postchecks_complete",
                "psf_joint_limit_postchecks_incomplete",
            ),
            (
                "psf_all_post_state_field_queries_valid_and_positive",
                "psf_post_state_field_queries_invalid",
            ),
        ):
            if not metrics[field]:
                apparatus_reasons.append(reason)
        for field, expected in (
            ("psf_filter_update_count", expected_updates),
            ("psf_physics_substep_count", expected_substeps),
            ("psf_completed_suffix_action_count", expected_actions),
        ):
            if metrics[field] != expected:
                apparatus_reasons.append("%s_differs" % field)

    baseline_reproduced = bool(
        metrics["adapter_link56_contact_present"]
        and metrics["adapter_first_selected_obstacle_contact_is_link56"]
    )
    contact_prevented = bool(
        baseline_reproduced
        and not psf_contact_present
        and metrics["psf_clearance_certified"]
    )
    correction_attributed = bool(
        metrics["material_correction_before_adapter_contact"]
        and metrics["material_correction_update_count"] > 0
        and metrics["first_material_correction_physical_boundary"]
        < metrics["adapter_first_link56_contact_physical_boundary"]
        and metrics["maximum_correction_norm_rad_s"]
        >= thresholds["minimum_filter_correction_norm_rad_s"]
        and metrics["filter_correction_integral_rad"]
        >= thresholds["minimum_filter_correction_integral_rad"]
    )
    motion_checks = {
        "measured_joint_motion": metrics[
            "post_correction_measured_joint_motion_integral_rad"
        ]
        >= thresholds[
            "minimum_post_correction_measured_joint_motion_integral_rad"
        ],
        "cartesian_motion": metrics["post_correction_cartesian_path_length_m"]
        >= thresholds["minimum_post_correction_cartesian_path_length_m"],
        "issued_motion": metrics[
            "post_correction_executed_command_integral_rad"
        ]
        >= thresholds[
            "minimum_post_correction_executed_command_integral_rad"
        ],
        "not_nearly_all_zero": metrics["post_correction_zero_command_fraction"]
        <= thresholds["maximum_post_correction_zero_command_fraction"],
    }
    useful_correction = bool(correction_attributed and all(motion_checks.values()))
    first_success = metrics["psf_first_task_success_source_action_index"]
    task_successful = bool(
        metrics["psf_task_success_ever"]
        and metrics["psf_terminal_task_success"]
        and metrics["psf_task_success_after_material_correction"]
        and first_success is not None
        and int(expectation["suffix_start"])
        <= first_success
        <= int(expectation["suffix_end"])
        and (first_success + 1) * 25
        > metrics["first_material_correction_physical_boundary"]
    )
    pair_complete = bool(
        not apparatus_reasons
        and metrics["adapter_exposure_complete"]
        and (metrics["psf_exposure_complete"] or psf_contact_present)
    )
    apparatus_valid = not apparatus_reasons
    stop_only = bool(
        apparatus_valid
        and baseline_reproduced
        and not psf_contact_present
        and correction_attributed
        and not useful_correction
    )
    if not apparatus_valid:
        classification = "INCONCLUSIVE_APPARATUS"
        reasons = apparatus_reasons
    elif not baseline_reproduced:
        classification = "INCONCLUSIVE_APPARATUS"
        reasons = ["adapter_did_not_reproduce_first_link56_contact"]
    elif psf_contact_present:
        classification = "CONTACT_REMAINS_OR_SHIFTED"
        reasons = [
            "psf_link56_contact_present"
            if metrics["psf_link56_contact_present"]
            else "psf_shifted_contact_to_other_robot_geometry"
        ]
    elif not metrics["psf_clearance_certified"]:
        classification = "INCONCLUSIVE_APPARATUS"
        reasons = ["psf_no_contact_but_full_robot_clearance_uncertified"]
    elif not correction_attributed:
        classification = "INCONCLUSIVE_APPARATUS"
        reasons = ["contact_prevention_not_attributable_to_material_psf_correction"]
    elif not useful_correction:
        classification = "STOP_ONLY"
        reasons = [
            "post_correction_motion_gate_failed:%s" % field
            for field, passed in motion_checks.items()
            if not passed
        ]
    elif not task_successful:
        classification = "CONTACT_PREVENTED_TASK_FAILED"
        reasons = ["native_safelibero_goal_not_satisfied_after_useful_correction"]
    else:
        classification = "SAFE_TASK_SUCCESS_USEFUL_CORRECTION"
        reasons = []
    return {
        "schema_version": CLASSIFICATION_SCHEMA,
        "classification": classification,
        "goal_achieved": classification == "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
        "apparatus_valid": apparatus_valid,
        "pair_complete": pair_complete,
        "baseline_reproduced": baseline_reproduced,
        "contact_prevented": contact_prevented,
        "task_successful": task_successful,
        "useful_correction": useful_correction,
        "stop_only": stop_only,
        "apparatus_failure_reasons": apparatus_reasons,
        "classification_reasons": reasons,
        "gates": {
            "correction_attributed": correction_attributed,
            "motion": motion_checks,
            "native_task_success_ever": metrics["psf_task_success_ever"],
            "native_task_success_terminal": metrics["psf_terminal_task_success"],
            "native_task_success_after_correction": metrics[
                "psf_task_success_after_material_correction"
            ],
            "no_psf_robot_obstacle_contact": not psf_contact_present,
            "full_robot_clearance_certified": metrics["psf_clearance_certified"],
        },
        "diagnostics": {
            "tracking_is_feasibility_gate": False,
            "timing_is_realtime_claim_gate": False,
            "adapter_task_success_ever": metrics["adapter_task_success_ever"],
            "adapter_terminal_task_success": metrics[
                "adapter_terminal_task_success"
            ],
        },
        "claim_scope": EXPECTED_CLAIM_SCOPE,
    }


def _validate_producer_classification(
    audit: _Audit,
    producer_value: Any,
    expected: Mapping[str, Any],
) -> None:
    producer = audit.mapping(producer_value, "classification")
    audit.check(
        producer.get("schema_version") == CLASSIFICATION_SCHEMA,
        "classification_schema_differs",
    )
    audit.check(
        producer.get("classification") in ALLOWED_CLASSIFICATIONS,
        "classification_label_invalid",
    )
    audit.check(
        _canonical(producer) == _canonical(expected),
        "producer_classification_differs",
    )


def validate_payload(
    payload_value: Mapping[str, Any],
    protocol_value: Mapping[str, Any],
    *,
    protocol_file_sha256: str,
    expected_commit: str,
    expected_job_id: Optional[str] = None,
    result_path: Optional[Path] = None,
) -> Dict[str, Any]:
    expectation = _protocol_expectations(protocol_value)
    audit = _Audit()
    payload = audit.mapping(payload_value, "result")
    audit.check(payload.get("schema_version") == RESULT_SCHEMA, "result_schema_differs")
    audit.check(payload.get("status") == "complete", "result_status_not_complete")
    audit.check(payload.get("partial_output_interpreted") is False, "partial_output_interpreted")
    audit.check(payload.get("case_id") == EXPECTED_CASE, "case_id_differs")
    audit.check(payload.get("protocol_id") == PROTOCOL_ID, "protocol_id_differs")
    if result_path is not None:
        audit.check(payload.get("run_id") == result_path.parent.name, "run_id_path_differs")

    provenance = audit.mapping(payload.get("provenance"), "provenance")
    source = audit.mapping(provenance.get("source"), "provenance_source")
    allocation = audit.mapping(provenance.get("allocation"), "provenance_allocation")
    audit.check(source.get("commit") == expected_commit, "source_commit_differs")
    audit.check(source.get("branch") == EXPECTED_BRANCH, "source_branch_differs")
    audit.check(provenance.get("case_id") == EXPECTED_CASE, "provenance_case_differs")
    audit.check(
        provenance.get("case_row_sha256") == EXPECTED_CASE_ROW_SHA256,
        "case_row_hash_differs",
    )
    audit.check(provenance.get("run_id") == payload.get("run_id"), "provenance_run_differs")
    audit.check(provenance.get("protocol_file_sha256") == protocol_file_sha256, "protocol_file_hash_differs")
    for field, expected in (
        ("historical_result_file_sha256", EXPECTED_HISTORICAL_FILE_SHA256),
        ("historical_result_payload_sha256", EXPECTED_HISTORICAL_PAYLOAD_SHA256),
        ("historical_executed_action_sequence_sha256", EXPECTED_ACTION_SEQUENCE_SHA256),
        ("suffix_action_record_sha256", EXPECTED_SUFFIX_RECORD_SHA256),
        ("suffix_action_array_sha256", EXPECTED_SUFFIX_ACTION_ARRAY_SHA256),
    ):
        audit.check(provenance.get(field) == expected, "%s_differs" % field)
    audit.check(provenance.get("online_policy_query_count") == 0, "online_policy_queries_nonzero")
    audit.check(allocation.get("gpu_name") == "NVIDIA H100 80GB HBM3", "allocation_gpu_not_exact_h100")
    audit.check(bool(allocation.get("host")), "allocation_host_missing")
    audit.check(str(allocation.get("slurm_job_id", "")).isdigit(), "slurm_job_id_invalid")
    if expected_job_id is not None:
        audit.check(str(allocation.get("slurm_job_id")) == str(expected_job_id), "slurm_job_id_differs")

    episode = audit.mapping(payload.get("episode"), "episode")
    for field, expected in (
        ("prefix_start_action_index", expectation["prefix_start"]),
        ("prefix_end_action_index_inclusive", expectation["prefix_end"]),
        ("prefix_action_count", expectation["prefix_action_count"]),
        ("suffix_start_action_index", expectation["suffix_start"]),
        ("suffix_end_action_index_inclusive", expectation["suffix_end"]),
        ("suffix_action_count", expectation["action_count"]),
        ("start_boundary", expectation["start_boundary"]),
        ("end_boundary", expectation["end_boundary"]),
    ):
        audit.check(episode.get(field) == expected, "episode_%s_differs" % field)
    audit.check(episode.get("post_action_179_flattened_state_sha256") == EXPECTED_BOUNDARY_STATE_SHA256, "episode_boundary_state_hash_differs")
    boundary_raw_hash = _sha256(audit, episode.get("official_boundary_raw_bytes_sha256"), "episode_boundary_raw_hash")
    audit.check(
        boundary_raw_hash == EXPECTED_BOUNDARY_RAW_SHA256,
        "episode_boundary_raw_hash_differs",
    )
    audit.check(list(episode.get("source_action_indexes", ())) == list(expectation["source_actions"]), "episode_source_actions_differ")

    field = audit.mapping(payload.get("field"), "field")
    resolved = audit.mapping(field.get("resolved_geometry"), "resolved_geometry")
    link_geom_ids = [audit.integer(value, "link_geom_id") for value in audit.sequence(resolved.get("link56_geom_ids"), "link_geom_ids")]
    robot_geom_ids = [audit.integer(value, "robot_geom_id") for value in audit.sequence(resolved.get("robot_geom_ids"), "robot_geom_ids")]
    obstacle_geom_ids = [audit.integer(value, "obstacle_geom_id") for value in audit.sequence(resolved.get("obstacle_geom_ids"), "obstacle_geom_ids")]
    audit.check(bool(link_geom_ids), "link_geom_population_empty")
    audit.check(bool(robot_geom_ids), "robot_geom_population_empty")
    audit.check(bool(obstacle_geom_ids), "obstacle_geom_population_empty")

    arms = audit.mapping(payload.get("arms"), "arms")
    adapter = audit.mapping(arms.get("adapter_only"), "adapter")
    psf = audit.mapping(arms.get("adapter_plus_psf"), "psf")
    audit.check(adapter.get("arm_name") == "joint_velocity_adapter_only", "adapter_arm_name_differs")
    audit.check(psf.get("arm_name") == "joint_velocity_adapter_plus_link56_psf", "psf_arm_name_differs")
    narrow._validate_pairing(audit, adapter, psf, boundary_raw_hash)

    adapter_contact = _reconstruct_contacts(audit, adapter, link_geom_ids, robot_geom_ids, obstacle_geom_ids, "adapter", expectation)
    psf_contact = _reconstruct_contacts(audit, psf, link_geom_ids, robot_geom_ids, obstacle_geom_ids, "psf", expectation)
    adapter_commands, adapter_physics = _validate_cadence(
        audit,
        adapter,
        "adapter",
        expectation,
        contact_present=adapter_contact["any_present"],
        must_complete=True,
    )
    psf_commands_raw, psf_physics = _validate_cadence(
        audit,
        psf,
        "psf",
        expectation,
        contact_present=psf_contact["any_present"],
        must_complete=not psf_contact["any_present"],
    )
    narrow._validate_exogenous_pairing(
        audit, adapter_commands[: len(psf_commands_raw)], psf_commands_raw
    )
    _validate_frozen_source_actions(
        audit,
        adapter_commands,
        psf_commands_raw,
        expectation,
    )
    narrow._validate_adapter_execution(audit, adapter, adapter_commands)

    psf_commands = narrow._validate_qp(audit, adapter, psf, psf_commands_raw)
    material_cutoff = (
        adapter_contact["first_link_boundary"]
        if adapter_contact["first_link_boundary"] is not None
        else expectation["no_event_boundary"]
    )
    material = narrow._material_activation(
        audit, psf_commands, psf, int(material_cutoff)
    )
    first_activation = material[0]["boundary"] if material else None
    post_motion = _post_correction_motion(
        audit,
        psf_commands_raw,
        psf_physics,
        first_activation,
        expectation,
    )
    adapter_task = _validate_task(
        audit,
        adapter,
        "adapter",
        expectation,
        allow_contact_terminal=False,
    )
    psf_task = _validate_task(
        audit,
        psf,
        "psf",
        expectation,
        allow_contact_terminal=psf_contact["any_present"],
    )
    audit.check(_canonical(adapter_task["definition"]) == _canonical(psf_task["definition"]), "pair_goal_definition_differs")

    pair_fields = (
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
    adapter_restore = audit.mapping(adapter.get("restore"), "adapter_restore_metrics")
    psf_restore = audit.mapping(psf.get("restore"), "psf_restore_metrics")
    exact_pair = bool(
        all(
            _canonical(adapter_restore.get(name))
            == _canonical(psf_restore.get(name))
            for name in pair_fields
        )
        and adapter.get("start_official_raw_bytes_sha256")
        == psf.get("start_official_raw_bytes_sha256")
        and adapter.get("fresh_adapter") is True
        and psf.get("fresh_adapter") is True
    )

    shared_prefix = audit.mapping(episode.get("shared_osc_prefix"), "shared_prefix")
    prefix_static = audit.mapping(
        shared_prefix.get("settled_to_boundary_static_field_assumption"),
        "prefix_static_assumption",
    )
    shared_prefix_complete = bool(
        shared_prefix.get("completed_action_count") == expectation["prefix_action_count"]
        and shared_prefix.get("terminal_state_sha256")
        == EXPECTED_BOUNDARY_STATE_SHA256
        and list(shared_prefix.get("terminal_goal_values", ()))
        == list(expectation["expected_boundary_goal_values"])
        and prefix_static.get("admissible") is True
    )
    audit.check(
        shared_prefix.get("execution")
        == "exact_historical_actions_0_through_179_under_unchanged_OSC",
        "shared_prefix_execution_differs",
    )
    suffix = audit.mapping(
        episode.get("paired_joint_velocity_suffix"), "paired_suffix"
    )
    audit.check(
        suffix.get("action_record_sha256") == EXPECTED_SUFFIX_RECORD_SHA256,
        "paired_suffix_record_hash_differs",
    )
    audit.check(
        suffix.get("action_array_sha256") == EXPECTED_SUFFIX_ACTION_ARRAY_SHA256,
        "paired_suffix_array_hash_differs",
    )
    audit.check(
        suffix.get("controller_updates_per_arm") == expectation["filter_updates"],
        "paired_suffix_registered_updates_differ",
    )
    audit.check(
        suffix.get("physics_substeps_per_arm") == expectation["physics_substeps"],
        "paired_suffix_registered_substeps_differ",
    )
    audit.check(
        episode.get("complete_recorded_episode_action_count") == 237,
        "complete_recorded_episode_count_differs",
    )
    historical = audit.mapping(
        episode.get("historical_aegis_reference"), "historical_aegis_reference"
    )
    expected_historical = {
        "authority": "frozen_manifest_selection_evidence_and_historical_replay",
        "diagnostic_only_not_a_new_acceptance_gate": True,
        "aegis_collision": True,
        "aegis_task_success": True,
        "literal_robot_contact_bodies": ["robot0_link5"],
        "first_relevant_contact_step": 187,
        "car_collision_step": 188,
        "released_aegis_barrier_h_at_first_relevant_contact": 0.009312042732980995,
        "released_aegis_barrier_h_at_car_crossing": 0.005099169140377209,
        "released_aegis_barrier_h_values_positive": True,
        "terminal_recorded_action": {
            "source_action_index": 236,
            "done": True,
            "reward": 1.0,
            "native_goal_values": [True],
        },
    }
    audit.check(
        _canonical(historical) == _canonical(expected_historical),
        "historical_aegis_reference_differs",
    )
    audit.check(episode.get("field_construction_boundary") == 0, "field_boundary_differs")

    psf_clearance_certified = _validate_field_and_clearance(
        audit,
        field,
        robot_geom_ids,
        psf,
        psf_physics,
        contact_present=psf_contact["any_present"],
        minimum_contact_distance_m=psf_contact["minimum_contact_distance_m"],
    )
    sample_count = audit.integer(psf.get("protected_sample_count"), "psf_sample_count_metrics")
    invalid_queries = audit.integer(psf.get("invalid_field_query_count"), "psf_invalid_queries_metrics")
    post_observations = audit.integer(
        psf.get("post_state_field_observation_count"), "psf_post_observations"
    )
    post_queries = audit.integer(
        psf.get("post_state_field_query_count"), "psf_post_queries"
    )
    nonpositive_queries = audit.integer(
        psf.get("nonpositive_post_state_field_query_count"),
        "psf_nonpositive_post_queries",
    )
    post_field_valid = bool(
        post_observations == 1
        and post_queries == sample_count
        and nonpositive_queries == 0
        and invalid_queries == 0
    )
    audit.check(
        psf.get("all_post_state_field_queries_valid_and_positive")
        is post_field_valid,
        "psf_post_field_valid_flag_differs",
    )
    static_admissible = bool(
        prefix_static.get("admissible") is True
        and adapter.get("static_precontact_admissible") is True
        and psf.get("static_full_window_admissible") is True
    )
    first_activation_metric = (
        int(first_activation)
        if first_activation is not None
        else int(expectation["no_event_boundary"])
    )
    adapter_contact_metric = (
        int(adapter_contact["first_link_boundary"])
        if adapter_contact["first_link_boundary"] is not None
        else int(expectation["no_event_boundary"])
    )
    correction_before_contact = bool(
        first_activation is not None
        and adapter_contact["first_link_boundary"] is not None
        and int(first_activation) < int(adapter_contact["first_link_boundary"])
    )
    psf_first_success = psf_task["first_success_source_action_index"]
    success_after_correction = bool(
        first_activation is not None
        and psf_first_success is not None
        and (int(psf_first_success) + 1) * 25 > int(first_activation)
    )
    full_episode_complete = bool(
        shared_prefix_complete
        and episode.get("complete_recorded_episode_action_count") == 237
        and adapter.get("exposure_complete") is True
        and psf.get("exposure_complete") is True
        and adapter_task["completed_action_count"] == expectation["action_count"]
        and psf_task["completed_action_count"] == expectation["action_count"]
    )
    reconstructed_metrics = {
        "schema_version": METRICS_SCHEMA,
        "exact_paired_start": exact_pair,
        "shared_prefix_complete": shared_prefix_complete,
        "full_recorded_episode_complete": full_episode_complete,
        "adapter_exposure_complete": adapter.get("exposure_complete") is True,
        "psf_exposure_complete": psf.get("exposure_complete") is True,
        "adapter_physics_monitor_trace_counts_match": adapter.get("physics_monitor_trace_counts_match") is True,
        "psf_physics_monitor_trace_counts_match": psf.get("physics_monitor_trace_counts_match") is True,
        "adapter_filter_update_count": len(adapter_commands),
        "psf_filter_update_count": len(psf_commands_raw),
        "adapter_physics_substep_count": len(adapter_physics),
        "psf_physics_substep_count": len(psf_physics),
        "adapter_completed_suffix_action_count": adapter_task["completed_action_count"],
        "psf_completed_suffix_action_count": psf_task["completed_action_count"],
        "psf_qp_count_complete": psf.get("qp_solve_count") == expectation["filter_updates"],
        "psf_qp_postchecks_complete": psf.get("qp_postcheck_count") == expectation["filter_updates"],
        "psf_joint_limit_postchecks_complete": psf.get("joint_limit_postcheck_count") == expectation["filter_updates"],
        "all_issued_commands_within_physical_bounds": bool(
            adapter.get("all_issued_commands_within_physical_bounds") is True
            and psf.get("all_issued_commands_within_physical_bounds") is True
        ),
        "both_nominal_commands_within_dynamic_joint_bounds": bool(
            adapter.get("all_nominal_commands_within_dynamic_joint_bounds") is True
            and psf.get("all_nominal_commands_within_dynamic_joint_bounds") is True
        ),
        "psf_invalid_field_query_count": invalid_queries,
        "psf_all_post_state_field_queries_valid_and_positive": post_field_valid,
        "static_selected_obstacle_admissible": static_admissible,
        "boundary_goal_unsatisfied": bool(
            adapter.get("task", {}).get("boundary_goal_values_exact") is True
            and psf.get("task", {}).get("boundary_goal_values_exact") is True
            and not adapter.get("task", {}).get("initial_task_success_at_branch")
            and not psf.get("task", {}).get("initial_task_success_at_branch")
        ),
        "adapter_link56_contact_present": adapter_contact["link_present"],
        "adapter_first_selected_obstacle_contact_is_link56": bool(
            adapter_contact["first_link_boundary"] is not None
            and adapter_contact["first_link_boundary"]
            == adapter_contact["first_any_boundary"]
        ),
        "psf_link56_contact_present": psf_contact["link_present"],
        "psf_any_robot_selected_obstacle_contact_present": psf_contact["any_present"],
        "psf_clearance_certified": psf_clearance_certified,
        "material_correction_before_adapter_contact": correction_before_contact,
        "first_material_correction_physical_boundary": first_activation_metric,
        "adapter_first_link56_contact_physical_boundary": adapter_contact_metric,
        "material_correction_update_count": len(material),
        "maximum_correction_norm_rad_s": post_motion["maximum_correction_norm_rad_s"],
        "filter_correction_integral_rad": post_motion["filter_correction_integral_rad"],
        "post_correction_measured_joint_motion_integral_rad": post_motion["measured_joint_motion_integral_rad"],
        "post_correction_cartesian_path_length_m": post_motion["cartesian_path_length_m"],
        "post_correction_executed_command_integral_rad": post_motion["executed_command_integral_rad"],
        "post_correction_zero_command_fraction": post_motion["zero_command_fraction"],
        "psf_task_success_ever": psf_task["ever_success"],
        "psf_terminal_task_success": psf_task["terminal_success"],
        "psf_first_task_success_source_action_index": psf_first_success,
        "psf_task_success_after_material_correction": success_after_correction,
        "adapter_task_success_ever": adapter_task["ever_success"],
        "adapter_terminal_task_success": adapter_task["terminal_success"],
    }
    metrics = audit.mapping(payload.get("metrics"), "metrics")
    audit.check(set(metrics) == set(METRIC_KEYS), "metric_key_set_differs")
    for field_name in METRIC_KEYS:
        expected = reconstructed_metrics[field_name]
        observed = metrics.get(field_name)
        if isinstance(expected, float):
            audit.check(
                _close(audit.number(observed, "metric_%s" % field_name), expected),
                "metric_%s_differs" % field_name,
            )
        else:
            audit.check(observed == expected, "metric_%s_differs" % field_name)

    activation = audit.mapping(payload.get("activation_evidence"), "activation_evidence")
    audit.check(
        _close(
            audit.number(activation.get("threshold_m2_per_s"), "activation_threshold"),
            expectation["thresholds"][
                "maximum_nominal_cbf_residual_for_activation_m2_per_s"
            ],
        ),
        "activation_threshold_differs",
    )
    audit.check(
        activation.get("adapter_contact_physical_boundary")
        == adapter_contact["first_link_boundary"],
        "activation_adapter_contact_boundary_differs",
    )
    audit.check(
        _canonical(activation.get("first_material_correction"))
        == _canonical(material[0]["raw"] if material else None),
        "activation_first_material_row_differs",
    )
    audit.check(
        activation.get("material_correction_count") == len(material),
        "activation_material_count_differs",
    )
    stored_motion = audit.mapping(
        activation.get("post_correction_motion"), "stored_post_correction_motion"
    )
    for field_name, expected in post_motion.items():
        observed = stored_motion.get(field_name)
        if isinstance(expected, float):
            audit.check(
                _close(
                    audit.number(observed, "stored_motion_%s" % field_name),
                    expected,
                ),
                "stored_motion_%s_differs" % field_name,
            )
        else:
            audit.check(
                observed == expected, "stored_motion_%s_differs" % field_name
            )

    expected_classification = _independent_classification(
        reconstructed_metrics, expectation
    )
    _validate_producer_classification(
        audit, payload.get("classification"), expected_classification
    )
    audit.check(payload.get("claim_scope") == EXPECTED_CLAIM_SCOPE, "claim_scope_differs")

    valid = not audit.discrepancies
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "validated" if valid else "rejected",
        "artifact_valid": valid,
        "run_id": payload.get("run_id"),
        "case_id": payload.get("case_id"),
        "result_payload_sha256": payload.get("result_payload_sha256"),
        "classification": expected_classification["classification"] if valid else None,
        "pair_complete": expected_classification["pair_complete"] if valid else None,
        "baseline_reproduced": expected_classification["baseline_reproduced"] if valid else None,
        "contact_prevented": expected_classification["contact_prevented"] if valid else None,
        "task_successful": expected_classification["task_successful"] if valid else None,
        "useful_correction": expected_classification["useful_correction"] if valid else None,
        "stop_only": expected_classification["stop_only"] if valid else None,
        "adapter_terminal_task_success": adapter_task["terminal_success"],
        "psf_terminal_task_success": psf_task["terminal_success"],
        "adapter_first_contact_physical_boundary": adapter_contact["first_any_boundary"],
        "first_material_correction_physical_boundary": first_activation,
        "psf_contacted_robot_geom_ids": psf_contact["robot_geom_ids_contacted"],
        "post_correction_motion": post_motion,
        "discrepancies": audit.discrepancies,
    }


def validate_artifact(
    result_path: Path,
    protocol_path: Path,
    *,
    expected_commit: str,
    expected_job_id: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(result_path)
    protocol = Path(protocol_path)
    if path.name != "result.json":
        raise ArtifactContractError("only terminal result.json is accepted")
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ArtifactContractError("terminal result is unavailable") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ArtifactContractError("terminal result must be a nonsymlink regular file")
    payload = load_hashed_json(path)
    protocol_payload = _load_plain_json(protocol)
    summary = validate_payload(
        payload,
        protocol_payload,
        protocol_file_sha256=sha256_file(protocol),
        expected_commit=expected_commit,
        expected_job_id=expected_job_id,
        result_path=path,
    )
    summary["result_file_sha256"] = sha256_file(path)
    summary["protocol_file_sha256"] = sha256_file(protocol)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-job-id")
    arguments = parser.parse_args()
    try:
        summary = validate_artifact(
            arguments.result,
            arguments.protocol,
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
