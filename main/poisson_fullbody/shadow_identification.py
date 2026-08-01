"""Read-only static-Poisson identification over an unchanged OSC rollout.

The observer in this module never writes simulator state and never constructs
or applies a control command.  It evaluates the registered link-5/6 surface
samples on a clone-forwarded MuJoCo state, records typed field-query failures,
and computes an *observed* CBF directional diagnostic from the simulator's
instantaneous arm velocity.  The diagnostic is not a hypothetical QP result.

The selected-obstacle field is static.  Translation, rotation, and collision-
surface drift are checked before every query.  Once any registered threshold
is crossed, the observer permanently stops querying the field while still
emitting one trace row for every physics callback.  This lets the surrounding
runner complete exact historical replay and contact measurement after the
static model has become inadmissible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

from main.poisson_fullbody.geometry import OrientedBox
from main.poisson_fullbody.jacobians import point_translational_jacobian
from main.poisson_fullbody.measurement import clone_forwarded_state
from main.poisson_fullbody.poisson_field import QueryInvalidReason
from main.poisson_fullbody.robot_samples import BodySample


class ShadowIdentificationError(RuntimeError):
    """Raised when a read-only shadow trace is incomplete or inconsistent."""


REGISTERED_QUERY_INVALID_REASONS = frozenset(
    reason.value for reason in QueryInvalidReason
)


def rollout_contact_boundary_index(
    observation_index: int, source_phase: str
) -> int:
    """Map a contact record to the field-trace boundary at the same time."""

    if (
        isinstance(observation_index, bool)
        or not isinstance(observation_index, int)
        or observation_index < 0
    ):
        raise ShadowIdentificationError(
            "rollout contact observation index must be a nonnegative integer"
        )
    if source_phase == "live_solver_phase_preintegration_geometry":
        return int(observation_index) - 1
    if source_phase == "post_integration_recomputed":
        return int(observation_index)
    raise ShadowIdentificationError("rollout contact source phase is invalid")


def registered_filter_update_available_before_contact(
    *,
    warning_observation_index: int,
    contact_observation_index: int,
    contact_source_phase: str,
    physics_substeps_per_filter_update: int,
) -> bool:
    """Return whether a scheduled filter solve can use the warning in time."""

    if (
        isinstance(warning_observation_index, bool)
        or not isinstance(warning_observation_index, int)
        or warning_observation_index < 0
        or isinstance(physics_substeps_per_filter_update, bool)
        or not isinstance(physics_substeps_per_filter_update, int)
        or physics_substeps_per_filter_update <= 0
    ):
        raise ShadowIdentificationError("filter-update timing inputs are invalid")
    contact_trace_boundary = rollout_contact_boundary_index(
        contact_observation_index, contact_source_phase
    )
    warning_physics_boundary = int(warning_observation_index) + 1
    update_period = int(physics_substeps_per_filter_update)
    next_filter_boundary = (
        (warning_physics_boundary + update_period - 1) // update_period
    ) * update_period
    contact_physics_boundary = contact_trace_boundary + 1
    return next_filter_boundary < contact_physics_boundary


def _modules() -> Tuple[Any, Any]:
    try:
        import mujoco
        import numpy as np
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("MuJoCo and NumPy are required for shadow identification") from error
    return mujoco, np


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _artifact_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShadowIdentificationError("%s must be a mapping" % label)
    return value


def _artifact_integer(
    value: Any, label: str, *, minimum: Optional[int] = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ShadowIdentificationError("%s must be an integer" % label)
    observed = int(value)
    if minimum is not None and observed < minimum:
        raise ShadowIdentificationError(
            "%s must be at least %d" % (label, int(minimum))
        )
    return observed


def _artifact_cadence_index(value: Any, label: str) -> Tuple[int, int, int]:
    # Producer-side records retain tuples; JSON round trips them as lists.
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ShadowIdentificationError(
            "%s must be a three-element cadence sequence" % label
        )
    result = tuple(
        _artifact_integer(item, "%s[%d]" % (label, index), minimum=0)
        for index, item in enumerate(value)
    )
    return result  # type: ignore[return-value]


def _artifact_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, (tuple, list)):
        raise ShadowIdentificationError("%s must be a sequence" % label)
    return value


def _artifact_boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ShadowIdentificationError("%s must be boolean" % label)
    return value


def _artifact_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ShadowIdentificationError(
            "%s must be a lowercase SHA-256 digest" % label
        )
    return value


def _artifact_physical_contact(record: Mapping[str, Any], label: str) -> bool:
    raw_distance = record.get("contact_distance_m")
    if (
        isinstance(raw_distance, bool)
        or not isinstance(raw_distance, (int, float))
        or not math.isfinite(float(raw_distance))
    ):
        raise ShadowIdentificationError("%s distance must be finite" % label)
    physical = _artifact_boolean(
        record.get("is_physical_nonpositive_distance_contact"),
        "%s physical-contact flag" % label,
    )
    if physical is not (float(raw_distance) <= 0.0):
        raise ShadowIdentificationError(
            "%s physical flag differs from MuJoCo nonpositive distance" % label
        )
    return physical


def _artifact_nonnegative_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ShadowIdentificationError("%s must be finite and nonnegative" % label)
    return float(value)


def _same_artifact_value(left: Any, right: Any) -> bool:
    """Compare typed producer values and their equivalent JSON representation."""

    try:
        return _canonical(left) == _canonical(right)
    except (TypeError, ValueError):
        return False


def _artifact_finite_vector(value: Any, length: int, label: str) -> Tuple[float, ...]:
    sequence = _artifact_sequence(value, label)
    if len(sequence) != length:
        raise ShadowIdentificationError("%s has the wrong length" % label)
    result = []
    for index, item in enumerate(sequence):
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise ShadowIdentificationError(
                "%s[%d] must be finite" % (label, index)
            )
        result.append(float(item))
    return tuple(result)


def _artifact_finite_matrix(
    value: Any, rows: int, columns: int, label: str
) -> Tuple[Tuple[float, ...], ...]:
    sequence = _artifact_sequence(value, label)
    if len(sequence) != rows:
        raise ShadowIdentificationError("%s has the wrong row count" % label)
    return tuple(
        _artifact_finite_vector(
            row,
            columns,
            "%s[%d]" % (label, row_index),
        )
        for row_index, row in enumerate(sequence)
    )


def _validate_sample_provenance(
    raw: Any,
    *,
    group: Mapping[str, Any],
    registered_sample_id_range: Tuple[int, int],
    label: str,
) -> Mapping[str, Any]:
    evidence = _artifact_mapping(raw, label)
    sample_id = _artifact_integer(
        evidence.get("sample_id"), "%s sample_id" % label, minimum=0
    )
    sample_id_start, sample_id_stop = registered_sample_id_range
    if not sample_id_start <= sample_id < sample_id_stop:
        raise ShadowIdentificationError(
            "%s sample ID is not owned by its protected geom" % label
        )
    for field in ("body_id", "body_name", "geom_id", "geom_name"):
        if evidence.get(field) != group.get(field):
            raise ShadowIdentificationError(
                "%s provenance differs at %s" % (label, field)
            )
    return evidence


def _validate_invalid_sample_evidence(
    raw: Any,
    *,
    group: Mapping[str, Any],
    registered_sample_id_range: Tuple[int, int],
    expected_reason: Optional[str],
    label: str,
) -> Mapping[str, Any]:
    evidence = _validate_sample_provenance(
        raw,
        group=group,
        registered_sample_id_range=registered_sample_id_range,
        label=label,
    )
    reason = evidence.get("reason")
    if reason not in REGISTERED_QUERY_INVALID_REASONS:
        raise ShadowIdentificationError("%s reason is invalid" % label)
    if expected_reason is not None and reason != expected_reason:
        raise ShadowIdentificationError("%s reason differs" % label)
    _artifact_finite_vector(
        evidence.get("point_world_m"), 3, "%s point_world_m" % label
    )
    return evidence


def _validate_cbf_diagnostic(
    raw: Any,
    *,
    group: Mapping[str, Any],
    registered_sample_id_range: Tuple[int, int],
    alpha_gain_per_s: float,
    label: str,
) -> Mapping[str, Any]:
    diagnostic = _validate_sample_provenance(
        raw,
        group=group,
        registered_sample_id_range=registered_sample_id_range,
        label=label,
    )
    _artifact_finite_vector(
        diagnostic.get("point_world_m"), 3, "%s point_world_m" % label
    )
    gradient = _artifact_finite_vector(
        diagnostic.get("gradient_world_m"), 3, "%s gradient_world_m" % label
    )
    arm_qvel = _artifact_finite_vector(
        diagnostic.get("observed_arm_qvel_rad_s"),
        7,
        "%s observed_arm_qvel_rad_s" % label,
    )
    point_jacobian = _artifact_finite_matrix(
        diagnostic.get("point_translational_jacobian_arm_3x7"),
        3,
        7,
        "%s point_translational_jacobian_arm_3x7" % label,
    )
    numeric = {}
    for field in (
        "h_m2",
        "gradient_norm_m",
        "observed_grad_h_J_qdot_m2_per_s",
        "alpha_h_m2_per_s",
        "observed_cbf_lhs_m2_per_s",
    ):
        raw_value = diagnostic.get(field)
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or not math.isfinite(float(raw_value))
        ):
            raise ShadowIdentificationError("%s %s is invalid" % (label, field))
        numeric[field] = float(raw_value)
    expected_gradient_norm = math.sqrt(sum(value * value for value in gradient))
    if not math.isclose(
        numeric["gradient_norm_m"],
        expected_gradient_norm,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ShadowIdentificationError("%s gradient norm differs" % label)
    point_velocity = tuple(
        sum(point_jacobian[row][column] * arm_qvel[column] for column in range(7))
        for row in range(3)
    )
    expected_directional = sum(
        gradient[row] * point_velocity[row] for row in range(3)
    )
    if not math.isclose(
        numeric["observed_grad_h_J_qdot_m2_per_s"],
        expected_directional,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ShadowIdentificationError(
            "%s directional derivative differs from gradient, Jacobian, and qvel"
            % label
        )
    expected_alpha_h = alpha_gain_per_s * numeric["h_m2"]
    if not math.isclose(
        numeric["alpha_h_m2_per_s"],
        expected_alpha_h,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ShadowIdentificationError("%s alpha*h arithmetic differs" % label)
    expected_lhs = (
        numeric["observed_grad_h_J_qdot_m2_per_s"]
        + numeric["alpha_h_m2_per_s"]
    )
    if not math.isclose(
        numeric["observed_cbf_lhs_m2_per_s"],
        expected_lhs,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ShadowIdentificationError("%s CBF LHS arithmetic differs" % label)
    if not isinstance(diagnostic.get("observed_cbf_diagnostic_semantics"), str):
        raise ShadowIdentificationError("%s semantics are absent" % label)
    return diagnostic


def _resolved_id_name_map(
    resolved: Mapping[str, Any], *, id_field: str, name_field: str, label: str
) -> Dict[int, str]:
    raw_ids = _artifact_sequence(resolved.get(id_field), "%s IDs" % label)
    raw_names = _artifact_sequence(resolved.get(name_field), "%s names" % label)
    if not raw_ids or len(raw_ids) != len(raw_names):
        raise ShadowIdentificationError("%s ID/name coverage differs" % label)
    result: Dict[int, str] = {}
    for index, (raw_id, raw_name) in enumerate(zip(raw_ids, raw_names)):
        identifier = _artifact_integer(
            raw_id, "%s ID[%d]" % (label, index), minimum=0
        )
        if identifier in result or not isinstance(raw_name, str) or not raw_name:
            raise ShadowIdentificationError("%s ID/name provenance is invalid" % label)
        result[identifier] = raw_name
    return result


def _validate_contact_record_provenance(
    record: Mapping[str, Any],
    *,
    robot_geom_names: Mapping[int, str],
    obstacle_geom_names: Mapping[int, str],
    robot_body_names: Mapping[int, str],
    obstacle_body_names: Mapping[int, str],
    collision_pairs: Set[Tuple[int, int]],
    component_by_geom: Mapping[int, Mapping[str, Any]],
    label: str,
) -> None:
    """Bind a serialized MuJoCo contact to the selected geometry authority."""

    robot_geom = _artifact_integer(
        record.get("robot_geom_id"), "%s robot_geom_id" % label, minimum=0
    )
    obstacle_geom = _artifact_integer(
        record.get("obstacle_geom_id"), "%s obstacle_geom_id" % label, minimum=0
    )
    if (
        robot_geom not in robot_geom_names
        or obstacle_geom not in obstacle_geom_names
        or (robot_geom, obstacle_geom) not in collision_pairs
        or record.get("robot_geom_name") != robot_geom_names[robot_geom]
        or record.get("obstacle_geom_name") != obstacle_geom_names[obstacle_geom]
    ):
        raise ShadowIdentificationError(
            "%s is outside the resolved robot/selected-obstacle contact authority"
            % label
        )

    robot_body = _artifact_integer(
        record.get("robot_body_id"), "%s robot_body_id" % label, minimum=0
    )
    obstacle_body = _artifact_integer(
        record.get("obstacle_body_id"), "%s obstacle_body_id" % label, minimum=0
    )
    if (
        robot_body not in robot_body_names
        or obstacle_body not in obstacle_body_names
        or record.get("robot_body_name") != robot_body_names[robot_body]
        or record.get("obstacle_body_name") != obstacle_body_names[obstacle_body]
    ):
        raise ShadowIdentificationError(
            "%s body provenance is outside the resolved geometry trees" % label
        )
    component = component_by_geom.get(robot_geom)
    if component is not None and (
        robot_body != component.get("body_id")
        or record.get("robot_body_name") != component.get("body_name")
    ):
        raise ShadowIdentificationError(
            "%s protected-link body provenance differs" % label
        )

    geom1 = _artifact_integer(
        record.get("mujoco_geom1_id"), "%s mujoco_geom1_id" % label, minimum=0
    )
    geom2 = _artifact_integer(
        record.get("mujoco_geom2_id"), "%s mujoco_geom2_id" % label, minimum=0
    )
    if (geom1, geom2) == (robot_geom, obstacle_geom):
        expected_name1 = robot_geom_names[robot_geom]
        expected_name2 = obstacle_geom_names[obstacle_geom]
    elif (geom1, geom2) == (obstacle_geom, robot_geom):
        expected_name1 = obstacle_geom_names[obstacle_geom]
        expected_name2 = robot_geom_names[robot_geom]
    else:
        raise ShadowIdentificationError(
            "%s MuJoCo geom orientation differs from its resolved pair" % label
        )
    if (
        record.get("mujoco_geom1_name") != expected_name1
        or record.get("mujoco_geom2_name") != expected_name2
    ):
        raise ShadowIdentificationError(
            "%s MuJoCo geom-name provenance differs" % label
        )


def _first_measurement_link_contact(
    measurement: Mapping[str, Any],
    *,
    link_geoms: Set[int],
    robot_geom_names: Mapping[int, str],
    obstacle_geom_names: Mapping[int, str],
    robot_body_names: Mapping[int, str],
    obstacle_body_names: Mapping[int, str],
    collision_pairs: Set[Tuple[int, int]],
    component_by_geom: Mapping[int, Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Reconstruct the producer's authoritative first link-5/6 contact."""

    settled = _artifact_mapping(
        measurement.get("settled_state"), "measurement.settled_state"
    )
    settled_records = _artifact_sequence(
        settled.get("physical_contact_point_records"),
        "measurement settled physical-contact records",
    )
    settled_candidate_records = _artifact_sequence(
        settled.get("candidate_contact_point_records"),
        "measurement settled candidate-contact records",
    )
    live_records = _artifact_sequence(
        measurement.get("live_solver_phase_contact_point_records"),
        "measurement live-solver contact records",
    )
    post_candidate_records = _artifact_sequence(
        measurement.get("post_state_candidate_contact_point_records"),
        "measurement post-state candidate-contact records",
    )
    post_records = _artifact_sequence(
        measurement.get("post_state_physical_contact_point_records"),
        "measurement post-state physical-contact records",
    )

    settled_candidates = []
    for index, raw in enumerate(settled_candidate_records):
        record = _artifact_mapping(
            raw, "settled candidate-contact record[%d]" % index
        )
        _validate_contact_record_provenance(
            record,
            robot_geom_names=robot_geom_names,
            obstacle_geom_names=obstacle_geom_names,
            robot_body_names=robot_body_names,
            obstacle_body_names=obstacle_body_names,
            collision_pairs=collision_pairs,
            component_by_geom=component_by_geom,
            label="settled candidate-contact record[%d]" % index,
        )
        if (
            record.get("source_phase")
            != "settled_post_integration_recomputed"
            or record.get("observation_index") is not None
        ):
            raise ShadowIdentificationError(
                "settled candidate-contact ledger has the wrong phase"
            )
        _artifact_physical_contact(record, "settled candidate contact")
        settled_candidates.append(record)

    settled_physical = []
    for index, raw in enumerate(settled_records):
        record = _artifact_mapping(
            raw, "settled physical-contact record[%d]" % index
        )
        _validate_contact_record_provenance(
            record,
            robot_geom_names=robot_geom_names,
            obstacle_geom_names=obstacle_geom_names,
            robot_body_names=robot_body_names,
            obstacle_body_names=obstacle_body_names,
            collision_pairs=collision_pairs,
            component_by_geom=component_by_geom,
            label="settled physical-contact record[%d]" % index,
        )
        if (
            record.get("source_phase")
            != "settled_post_integration_recomputed"
            or record.get("observation_index") is not None
            or _artifact_physical_contact(record, "settled contact") is not True
        ):
            raise ShadowIdentificationError(
                "settled physical-contact ledger contains a nonphysical record"
        )
        settled_physical.append(record)
    expected_settled_physical = [
        record
        for record in settled_candidates
        if record.get("is_physical_nonpositive_distance_contact") is True
    ]
    if not _same_artifact_value(settled_physical, expected_settled_physical):
        raise ShadowIdentificationError(
            "settled physical-contact ledger is not the candidate-ledger subset"
        )

    live_physical = []
    for index, raw in enumerate(live_records):
        record = _artifact_mapping(raw, "live-solver contact record[%d]" % index)
        _validate_contact_record_provenance(
            record,
            robot_geom_names=robot_geom_names,
            obstacle_geom_names=obstacle_geom_names,
            robot_body_names=robot_body_names,
            obstacle_body_names=obstacle_body_names,
            collision_pairs=collision_pairs,
            component_by_geom=component_by_geom,
            label="live-solver contact record[%d]" % index,
        )
        physical = _artifact_physical_contact(record, "live-solver contact")
        if record.get("source_phase") != "live_solver_phase_preintegration_geometry":
            raise ShadowIdentificationError(
                "live-solver contact ledger has the wrong source phase"
            )
        if physical:
            live_physical.append(record)

    post_candidates = []
    for index, raw in enumerate(post_candidate_records):
        record = _artifact_mapping(
            raw, "post-state candidate-contact record[%d]" % index
        )
        _validate_contact_record_provenance(
            record,
            robot_geom_names=robot_geom_names,
            obstacle_geom_names=obstacle_geom_names,
            robot_body_names=robot_body_names,
            obstacle_body_names=obstacle_body_names,
            collision_pairs=collision_pairs,
            component_by_geom=component_by_geom,
            label="post-state candidate-contact record[%d]" % index,
        )
        if record.get("source_phase") != "post_integration_recomputed":
            raise ShadowIdentificationError(
                "post-state candidate-contact ledger has the wrong source phase"
            )
        _artifact_physical_contact(record, "post-state candidate contact")
        post_candidates.append(record)

    post_physical = []
    for index, raw in enumerate(post_records):
        record = _artifact_mapping(raw, "post-state physical-contact record[%d]" % index)
        _validate_contact_record_provenance(
            record,
            robot_geom_names=robot_geom_names,
            obstacle_geom_names=obstacle_geom_names,
            robot_body_names=robot_body_names,
            obstacle_body_names=obstacle_body_names,
            collision_pairs=collision_pairs,
            component_by_geom=component_by_geom,
            label="post-state physical-contact record[%d]" % index,
        )
        if (
            record.get("source_phase") != "post_integration_recomputed"
            or _artifact_physical_contact(record, "post-state contact") is not True
        ):
            raise ShadowIdentificationError(
                "post-state physical-contact ledger contains a nonphysical record"
        )
        post_physical.append(record)
    expected_post_physical = [
        record
        for record in post_candidates
        if record.get("is_physical_nonpositive_distance_contact") is True
    ]
    if not _same_artifact_value(post_physical, expected_post_physical):
        raise ShadowIdentificationError(
            "post-state physical-contact ledger is not the candidate-ledger subset"
        )

    rollout_physical = live_physical + post_physical
    all_physical = settled_physical + rollout_physical
    expected_counts = (
        (
            settled,
            "candidate_contact_point_record_count",
            len(settled_candidates),
        ),
        (
            settled,
            "physical_contact_point_record_count",
            len(settled_physical),
        ),
        (
            measurement,
            "live_solver_candidate_contact_point_record_count",
            len(live_records),
        ),
        (
            measurement,
            "live_solver_nonpositive_contact_point_record_count",
            len(live_physical),
        ),
        (
            measurement,
            "post_state_candidate_contact_point_record_count",
            len(post_candidates),
        ),
        (
            measurement,
            "post_state_physical_contact_point_record_count",
            len(post_physical),
        ),
        (
            measurement,
            "rollout_phase_physical_contact_point_record_count",
            len(rollout_physical),
        ),
        (
            measurement,
            "total_candidate_contact_point_record_count",
            len(settled_candidates) + len(live_records) + len(post_candidates),
        ),
        (
            measurement,
            "total_physical_contact_point_record_count",
            len(all_physical),
        ),
    )
    for container, field, expected in expected_counts:
        if (
            _artifact_integer(
                container.get(field), "measurement.%s" % field, minimum=0
            )
            != expected
        ):
            raise ShadowIdentificationError(
                "measurement physical-contact count differs at %s" % field
            )

    expected_flags = (
        ("any_robot_obstacle_contact", bool(all_physical)),
        ("rollout_any_robot_obstacle_contact", bool(rollout_physical)),
        ("post_state_any_robot_obstacle_contact", bool(post_physical)),
        ("live_solver_any_robot_obstacle_contact", bool(live_physical)),
        (
            "link56_obstacle_contact",
            any(
                _artifact_integer(
                    record.get("robot_geom_id"), "physical contact robot geom", minimum=0
                )
                in link_geoms
                for record in all_physical
            ),
        ),
        (
            "rollout_link56_obstacle_contact",
            any(
                _artifact_integer(
                    record.get("robot_geom_id"), "rollout contact robot geom", minimum=0
                )
                in link_geoms
                for record in rollout_physical
            ),
        ),
        (
            "post_state_link56_obstacle_contact",
            any(
                _artifact_integer(
                    record.get("robot_geom_id"), "post-state contact robot geom", minimum=0
                )
                in link_geoms
                for record in post_physical
            ),
        ),
        (
            "live_solver_link56_obstacle_contact",
            any(
                _artifact_integer(
                    record.get("robot_geom_id"), "live-solver contact robot geom", minimum=0
                )
                in link_geoms
                for record in live_physical
            ),
        ),
    )
    for field, expected in expected_flags:
        if _artifact_boolean(measurement.get(field), "measurement.%s" % field) is not expected:
            raise ShadowIdentificationError(
                "measurement physical-contact flag differs at %s" % field
            )

    candidates = []
    for record in all_physical:
        geom_id = _artifact_integer(
            record.get("robot_geom_id"), "physical contact robot geom", minimum=0
        )
        if geom_id in link_geoms:
            candidates.append(record)
    if not candidates:
        return None

    def order(record: Mapping[str, Any]) -> Tuple[int, int, int]:
        raw_observation = record.get("observation_index")
        observation = (
            -1
            if raw_observation is None
            else _artifact_integer(
                raw_observation, "physical contact observation_index", minimum=0
            )
        )
        phase = {
            "settled_post_integration_recomputed": 0,
            "live_solver_phase_preintegration_geometry": 1,
            "post_integration_recomputed": 2,
        }.get(record.get("source_phase"), 3)
        contact_index = _artifact_integer(
            record.get("mujoco_contact_index"),
            "physical contact mujoco_contact_index",
            minimum=0,
        )
        return observation, phase, contact_index

    return min(candidates, key=order)


def _first_trace_signal(
    trace: Sequence[Any], *, geom_id: Optional[int], kind: str
) -> Optional[Dict[str, Any]]:
    """Reconstruct the observer signal used to authorize the active canary."""

    for row_index, raw_row in enumerate(trace):
        row = _artifact_mapping(raw_row, "trace[%d]" % row_index)
        if row.get("field_query_attempted") is not True:
            continue
        groups = _artifact_sequence(row.get("per_geom"), "trace per-geom records")
        for group_index, raw_group in enumerate(groups):
            group = _artifact_mapping(
                raw_group, "trace[%d].per_geom[%d]" % (row_index, group_index)
            )
            observed_geom = _artifact_integer(
                group.get("geom_id"), "trace per-geom geom_id", minimum=0
            )
            if geom_id is not None and observed_geom != geom_id:
                continue
            evidence: Optional[Mapping[str, Any]] = None
            if kind == "any_invalid":
                if (
                    _artifact_integer(
                        group.get("invalid_query_count"),
                        "trace per-geom invalid_query_count",
                        minimum=0,
                    )
                    > 0
                ):
                    evidence = _artifact_mapping(
                        group.get("first_invalid_sample"),
                        "trace first invalid sample",
                    )
            elif kind == "cbf_lhs_negative":
                minimum_lhs = group.get("minimum_observed_cbf_lhs_sample")
                if minimum_lhs is not None:
                    candidate = _artifact_mapping(
                        minimum_lhs, "trace minimum observed CBF sample"
                    )
                    raw_lhs = candidate.get("observed_cbf_lhs_m2_per_s")
                    if (
                        isinstance(raw_lhs, bool)
                        or not isinstance(raw_lhs, (int, float))
                        or not math.isfinite(float(raw_lhs))
                    ):
                        raise ShadowIdentificationError(
                            "trace minimum observed CBF value is invalid"
                        )
                    if float(raw_lhs) < 0.0:
                        evidence = candidate
            else:  # pragma: no cover - private caller is closed over two kinds
                raise ShadowIdentificationError("unknown trace signal kind")
            if evidence is not None:
                geom_name = group.get("geom_name")
                body_name = group.get("body_name")
                if not isinstance(geom_name, str) or not isinstance(body_name, str):
                    raise ShadowIdentificationError(
                        "trace signal geometry names are invalid"
                    )
                return {
                    "observation_index": _artifact_integer(
                        row.get("observation_index"),
                        "trace signal observation_index",
                        minimum=0,
                    ),
                    "high_level_index": _artifact_integer(
                        row.get("high_level_index"),
                        "trace signal high_level_index",
                        minimum=0,
                    ),
                    "physics_substep_index": _artifact_integer(
                        row.get("physics_substep_index"),
                        "trace signal physics_substep_index",
                        minimum=0,
                    ),
                    "signal_kind": kind,
                    "geom_id": observed_geom,
                    "geom_name": geom_name,
                    "body_id": _artifact_integer(
                        group.get("body_id"), "trace signal body_id", minimum=0
                    ),
                    "body_name": body_name,
                    "evidence": dict(evidence),
                }
    return None


def _drift_crossing_reasons(
    raw_drift: Any,
    *,
    thresholds: Mapping[str, float],
    expected_geom_ids: Set[int],
    label: str,
) -> Tuple[str, ...]:
    """Validate one raw drift record and return its registered crossings."""

    drift = _artifact_mapping(raw_drift, label)
    raw_geoms = _artifact_sequence(drift.get("geoms"), "%s.geoms" % label)
    if not raw_geoms:
        raise ShadowIdentificationError("%s geoms are empty" % label)
    observed_geom_ids = set()
    translations = []
    rotations = []
    surfaces = []
    all_reasons = set()
    for index, raw_geom in enumerate(raw_geoms):
        geom = _artifact_mapping(raw_geom, "%s.geoms[%d]" % (label, index))
        geom_id = _artifact_integer(
            geom.get("geom_id"), "%s geom_id" % label, minimum=0
        )
        if geom_id in observed_geom_ids:
            raise ShadowIdentificationError("%s has duplicate geom IDs" % label)
        observed_geom_ids.add(geom_id)
        translation = _artifact_nonnegative_number(
            geom.get("translation_m"), "%s translation" % label
        )
        rotation = _artifact_nonnegative_number(
            geom.get("rotation_rad"), "%s rotation" % label
        )
        surface = _artifact_nonnegative_number(
            geom.get("maximum_surface_point_displacement_m"),
            "%s surface displacement" % label,
        )
        translations.append(translation)
        rotations.append(rotation)
        surfaces.append(surface)
        expected_reasons = []
        if translation > thresholds["translation_m"]:
            expected_reasons.append("translation_threshold_crossed")
        if rotation > thresholds["rotation_rad"]:
            expected_reasons.append("rotation_threshold_crossed")
        if surface > thresholds["surface_m"]:
            expected_reasons.append("surface_threshold_crossed")
        raw_reasons = _artifact_sequence(
            geom.get("threshold_crossing_reasons"),
            "%s threshold-crossing reasons" % label,
        )
        if list(raw_reasons) != expected_reasons:
            raise ShadowIdentificationError(
                "%s threshold-crossing reasons differ" % label
            )
        all_reasons.update(expected_reasons)
    if observed_geom_ids != expected_geom_ids:
        raise ShadowIdentificationError(
            "%s geom coverage differs from the selected obstacle" % label
        )
    aggregates = (
        ("maximum_translation_m", max(translations)),
        ("maximum_rotation_rad", max(rotations)),
        ("maximum_surface_point_displacement_m", max(surfaces)),
    )
    for field, expected in aggregates:
        observed = _artifact_nonnegative_number(
            drift.get(field), "%s.%s" % (label, field)
        )
        if observed != expected:
            raise ShadowIdentificationError(
                "%s aggregate %s differs" % (label, field)
            )
    return tuple(sorted(all_reasons))


def validate_shadow_replay_record(
    shadow: Mapping[str, Any],
    *,
    action_count: int,
    inner_updates_per_high_level_action: int,
    physics_substeps_per_inner_update: int,
    physics_timestep_s: float,
    contact_definition: str,
    alpha_gain_per_s: float,
    static_drift_thresholds: Mapping[str, float],
    differential_audit_config: Mapping[str, Any],
) -> None:
    """Independently validate the complete serialized shadow replay.

    This validator is deliberately pure: the producer invokes it before
    publishing a passed artifact, and the active consumer invokes it again
    after loading JSON.  It therefore accepts typed runtime cadence tuples and
    their JSON list representation, but recomputes counts, coordinates, trace
    hashing, field invalidation stopping, and warning/contact arithmetic.
    """

    record = _artifact_mapping(shadow, "shadow replay")
    actions = _artifact_integer(action_count, "action_count", minimum=1)
    inner_count = _artifact_integer(
        inner_updates_per_high_level_action,
        "inner_updates_per_high_level_action",
        minimum=1,
    )
    physics_count = _artifact_integer(
        physics_substeps_per_inner_update,
        "physics_substeps_per_inner_update",
        minimum=1,
    )
    timestep = float(physics_timestep_s)
    if not math.isfinite(timestep) or timestep <= 0.0:
        raise ShadowIdentificationError(
            "physics_timestep_s must be finite and positive"
        )
    if not isinstance(contact_definition, str) or not contact_definition:
        raise ShadowIdentificationError("contact_definition must be nonempty")
    if (
        isinstance(alpha_gain_per_s, bool)
        or not isinstance(alpha_gain_per_s, (int, float))
        or not math.isfinite(float(alpha_gain_per_s))
        or float(alpha_gain_per_s) <= 0.0
    ):
        raise ShadowIdentificationError("expected alpha_gain_per_s is invalid")
    expected_alpha = float(alpha_gain_per_s)
    expected_drift_thresholds = _artifact_mapping(
        static_drift_thresholds, "expected static-field drift thresholds"
    )
    registered_drift_thresholds: Dict[str, float] = {}
    for field in ("translation_m", "rotation_rad", "surface_m"):
        raw_threshold = expected_drift_thresholds.get(field)
        if (
            isinstance(raw_threshold, bool)
            or not isinstance(raw_threshold, (int, float))
            or not math.isfinite(float(raw_threshold))
            or float(raw_threshold) < 0.0
        ):
            raise ShadowIdentificationError(
                "expected static-field drift threshold %s is invalid" % field
            )
        registered_drift_thresholds[field] = float(raw_threshold)
    per_high = inner_count * physics_count
    expected = actions * per_high
    for field, value in (
        ("executed_action_count", actions),
        ("callback_count", expected),
        ("expected_callback_count", expected),
    ):
        if _artifact_integer(record.get(field), "shadow.%s" % field, minimum=0) != value:
            raise ShadowIdentificationError("shadow %s differs" % field)

    measurement = _artifact_mapping(record.get("measurement"), "shadow measurement")
    if (
        _artifact_integer(
            measurement.get("observed_physics_substeps"),
            "measurement.observed_physics_substeps",
            minimum=0,
        )
        != expected
    ):
        raise ShadowIdentificationError("measurement callback count differs")
    first_index = _artifact_cadence_index(
        measurement.get("first_index"), "measurement.first_index"
    )
    last_index = _artifact_cadence_index(
        measurement.get("last_index"), "measurement.last_index"
    )
    if first_index != (0, 0, 0) or last_index != (
        actions - 1,
        inner_count - 1,
        physics_count - 1,
    ):
        raise ShadowIdentificationError("measurement cadence endpoints differ")
    if measurement.get("physical_contact_distance_semantics") != contact_definition:
        raise ShadowIdentificationError("measurement contact semantics differ")

    raw_action_state_hashes = _artifact_sequence(
        record.get("action_boundary_state_sha256_ledger"),
        "action-boundary state SHA-256 ledger",
    )
    if len(raw_action_state_hashes) != actions:
        raise ShadowIdentificationError(
            "action-boundary state SHA-256 ledger length differs"
        )
    action_state_hashes = [
        _artifact_sha256(
            value, "action-boundary state SHA-256 ledger[%d]" % index
        )
        for index, value in enumerate(raw_action_state_hashes)
    ]
    state_sequence_sha256 = _artifact_sha256(
        record.get("state_sequence_sha256"), "state sequence SHA-256"
    )
    if state_sequence_sha256 != hashlib.sha256(
        _canonical(action_state_hashes)
    ).hexdigest():
        raise ShadowIdentificationError(
            "state sequence SHA-256 differs from its action-boundary ledger"
        )
    terminal_state_sha256 = _artifact_sha256(
        record.get("terminal_simulator_state_sha256"),
        "terminal simulator state SHA-256",
    )
    if terminal_state_sha256 != action_state_hashes[-1]:
        raise ShadowIdentificationError(
            "terminal simulator state differs from the action-boundary ledger"
        )
    _artifact_sha256(
        record.get("observation_sequence_sha256"),
        "observation sequence SHA-256",
    )

    raw_callback_state_ledger = _artifact_sequence(
        record.get("callback_state_read_only_ledger"),
        "callback state read-only ledger",
    )
    if len(raw_callback_state_ledger) != expected:
        raise ShadowIdentificationError(
            "callback state read-only ledger length differs"
        )
    callback_after_hashes = []
    for index, raw_callback_record in enumerate(raw_callback_state_ledger):
        callback_record = _artifact_mapping(
            raw_callback_record, "callback state read-only ledger[%d]" % index
        )
        expected_high = index // per_high
        expected_inner = (index % per_high) // physics_count
        expected_physics = index % physics_count
        for field, expected_value in (
            ("observation_index", index),
            ("high_level_index", expected_high),
            ("inner_control_index", expected_inner),
            ("physics_substep_index", expected_physics),
        ):
            if _artifact_integer(
                callback_record.get(field),
                "callback state read-only ledger[%d].%s" % (index, field),
                minimum=0,
            ) != expected_value:
                raise ShadowIdentificationError(
                    "callback state read-only cadence differs at %d" % index
                )
        before_sha256 = _artifact_sha256(
            callback_record.get("before_sha256"),
            "callback state read-only ledger[%d].before_sha256" % index,
        )
        after_sha256 = _artifact_sha256(
            callback_record.get("after_sha256"),
            "callback state read-only ledger[%d].after_sha256" % index,
        )
        if (
            _artifact_boolean(
                callback_record.get("exact_array_equal"),
                "callback state read-only ledger[%d].exact_array_equal"
                % index,
            )
            is not True
            or before_sha256 != after_sha256
        ):
            raise ShadowIdentificationError(
                "callback changed the complete MuJoCo integration state at %d"
                % index
            )
        callback_after_hashes.append(after_sha256)
    if _artifact_sha256(
        record.get("callback_state_read_only_ledger_sha256"),
        "callback state read-only ledger SHA-256",
    ) != hashlib.sha256(_canonical(list(raw_callback_state_ledger))).hexdigest():
        raise ShadowIdentificationError(
            "callback state read-only ledger SHA-256 differs"
        )
    if _artifact_sha256(
        record.get("callback_state_sequence_sha256"),
        "callback state sequence SHA-256",
    ) != hashlib.sha256(_canonical(callback_after_hashes)).hexdigest():
        raise ShadowIdentificationError(
            "callback state sequence SHA-256 differs from its ledger"
        )

    construction = _artifact_mapping(
        record.get("construction"), "shadow construction"
    )
    resolved = _artifact_mapping(
        construction.get("resolved_geometry"), "resolved geometry"
    )
    raw_link_geoms = resolved.get("link56_geom_ids")
    if not isinstance(raw_link_geoms, (tuple, list)) or not raw_link_geoms:
        raise ShadowIdentificationError("resolved link56 geom IDs are absent")
    link_geom_sequence = tuple(
        _artifact_integer(value, "resolved link56 geom ID", minimum=0)
        for value in raw_link_geoms
    )
    link_geoms = set(link_geom_sequence)
    if len(link_geoms) != len(raw_link_geoms):
        raise ShadowIdentificationError("resolved link56 geom IDs are not unique")
    raw_obstacle_geoms = _artifact_sequence(
        resolved.get("obstacle_geom_ids"), "resolved obstacle geom IDs"
    )
    obstacle_geoms = {
        _artifact_integer(value, "resolved obstacle geom ID", minimum=0)
        for value in raw_obstacle_geoms
    }
    if not obstacle_geoms or len(obstacle_geoms) != len(raw_obstacle_geoms):
        raise ShadowIdentificationError(
            "resolved obstacle geom IDs must be nonempty and unique"
        )
    field_bundle = _artifact_mapping(
        construction.get("field_bundle"), "construction field bundle"
    )
    registered_sample_count = _artifact_integer(
        field_bundle.get("protected_sample_count"),
        "field bundle protected_sample_count",
        minimum=1,
    )
    raw_components = _artifact_sequence(
        field_bundle.get("surface_components"), "field bundle surface components"
    )
    expected_component_fields = {
        "geom_id",
        "geom_name",
        "body_id",
        "body_name",
        "geometry_kind",
        "certificate_kind",
        "surface_element_count",
        "sample_count",
        "certified_surface_cover_radius_m",
    }
    component_by_geom: Dict[int, Mapping[str, Any]] = {}
    component_sample_ranges: Dict[int, Tuple[int, int]] = {}
    component_cover_radii = []
    component_sample_total = 0
    for index, raw_component in enumerate(raw_components):
        component = _artifact_mapping(
            raw_component, "field bundle surface component[%d]" % index
        )
        if set(component) != expected_component_fields:
            raise ShadowIdentificationError(
                "field bundle surface component fields differ at %d" % index
            )
        geom_id = _artifact_integer(
            component.get("geom_id"), "surface component geom_id", minimum=0
        )
        if geom_id in component_by_geom:
            raise ShadowIdentificationError(
                "field bundle has duplicate surface-component geom IDs"
            )
        sample_count = _artifact_integer(
            component.get("sample_count"),
            "surface component sample_count",
            minimum=1,
        )
        _artifact_integer(
            component.get("surface_element_count"),
            "surface component surface_element_count",
            minimum=1,
        )
        for field in (
            "geom_name",
            "body_name",
            "geometry_kind",
            "certificate_kind",
        ):
            value = component.get(field)
            if not isinstance(value, str) or not value:
                raise ShadowIdentificationError(
                    "surface component %s is invalid" % field
                )
        component_cover_radii.append(
            _artifact_nonnegative_number(
                component.get("certified_surface_cover_radius_m"),
                "surface component certified cover radius",
            )
        )
        component_by_geom[geom_id] = component
        sample_start = component_sample_total
        component_sample_total += sample_count
        component_sample_ranges[geom_id] = (
            sample_start,
            component_sample_total,
        )
    if (
        tuple(component_by_geom) != link_geom_sequence
        or component_sample_total != registered_sample_count
    ):
        raise ShadowIdentificationError(
            "field bundle ordered sample/component coverage differs from resolved "
            "link56 geoms"
        )
    sampling_epsilon_m = _artifact_nonnegative_number(
        field_bundle.get("protected_sampling_epsilon_m"),
        "field bundle protected sampling epsilon",
    )
    maximum_cover_radius_m = _artifact_nonnegative_number(
        field_bundle.get("protected_sampling_maximum_surface_cover_radius_m"),
        "field bundle protected sampling maximum cover radius",
    )
    coverage_semantics = field_bundle.get(
        "protected_sampling_coverage_semantics"
    )
    if (
        sampling_epsilon_m <= 0.0
        or maximum_cover_radius_m >= sampling_epsilon_m
        or not component_cover_radii
        or any(radius >= sampling_epsilon_m for radius in component_cover_radii)
        or maximum_cover_radius_m != max(component_cover_radii)
        or not isinstance(coverage_semantics, str)
        or not coverage_semantics
    ):
        raise ShadowIdentificationError(
            "field bundle protected sampling certificate is invalid"
        )
    raw_protected_samples = _artifact_sequence(
        field_bundle.get("protected_samples"),
        "field bundle protected samples",
    )
    if len(raw_protected_samples) != registered_sample_count:
        raise ShadowIdentificationError(
            "field bundle protected sample ledger length differs"
        )
    protected_samples = []
    expected_sample_fields = {
        "sample_id",
        "body_id",
        "body_name",
        "geom_id",
        "geom_name",
        "point_body_local_m",
        "source",
    }
    for sample_index, raw_sample in enumerate(raw_protected_samples):
        sample_record = _artifact_mapping(
            raw_sample,
            "field bundle protected sample[%d]" % sample_index,
        )
        if set(sample_record) != expected_sample_fields:
            raise ShadowIdentificationError(
                "field bundle protected sample fields differ at %d" % sample_index
            )
        sample_id = _artifact_integer(
            sample_record.get("sample_id"),
            "field bundle protected sample sample_id",
            minimum=0,
        )
        body_id = _artifact_integer(
            sample_record.get("body_id"),
            "field bundle protected sample body_id",
            minimum=0,
        )
        geom_id = _artifact_integer(
            sample_record.get("geom_id"),
            "field bundle protected sample geom_id",
            minimum=0,
        )
        body_name = sample_record.get("body_name")
        geom_name = sample_record.get("geom_name")
        source = sample_record.get("source")
        if (
            sample_id != sample_index
            or not isinstance(body_name, str)
            or not body_name
            or not isinstance(geom_name, str)
            or not geom_name
            or source != "collision_geom_surface"
        ):
            raise ShadowIdentificationError(
                "field bundle protected sample identity differs at %d" % sample_index
            )
        local_point = _artifact_finite_vector(
            sample_record.get("point_body_local_m"),
            3,
            "field bundle protected sample local point",
        )
        protected_samples.append(
            {
                "sample_id": sample_id,
                "body_id": body_id,
                "body_name": body_name,
                "geom_id": geom_id,
                "geom_name": geom_name,
                "point_body_local_m": list(local_point),
                "source": source,
            }
        )
    field_bundle_hashes = _artifact_mapping(
        field_bundle.get("hashes"), "construction field bundle hashes"
    )
    protected_sample_payload = {
        "epsilon_m": sampling_epsilon_m,
        "maximum_surface_cover_radius_m": maximum_cover_radius_m,
        "coverage_semantics": coverage_semantics,
        "components": list(raw_components),
        "samples": protected_samples,
    }
    if _artifact_sha256(
        field_bundle_hashes.get("protected_samples_sha256"),
        "field bundle protected_samples_sha256",
    ) != hashlib.sha256(_canonical(protected_sample_payload)).hexdigest():
        raise ShadowIdentificationError(
            "field bundle protected sample SHA-256 differs from its ledger"
        )
    robot_geom_names = _resolved_id_name_map(
        resolved,
        id_field="robot_geom_ids",
        name_field="robot_geom_names",
        label="resolved robot geoms",
    )
    obstacle_geom_names = _resolved_id_name_map(
        resolved,
        id_field="obstacle_geom_ids",
        name_field="obstacle_geom_names",
        label="resolved obstacle geoms",
    )
    robot_body_names = _resolved_id_name_map(
        resolved,
        id_field="robot_body_ids",
        name_field="robot_body_names",
        label="resolved robot bodies",
    )
    obstacle_body_names = _resolved_id_name_map(
        resolved,
        id_field="obstacle_body_ids",
        name_field="obstacle_body_names",
        label="resolved obstacle bodies",
    )
    if (
        link_geoms - set(robot_geom_names)
        or obstacle_geoms != set(obstacle_geom_names)
        or set(robot_geom_names) & set(obstacle_geom_names)
    ):
        raise ShadowIdentificationError(
            "resolved protected/obstacle geom authority is inconsistent"
        )
    raw_link_names = _artifact_sequence(
        resolved.get("link56_geom_names"), "resolved link56 geom names"
    )
    if list(raw_link_names) != [
        robot_geom_names[geom_id] for geom_id in link_geom_sequence
    ]:
        raise ShadowIdentificationError(
            "resolved link56 geom ID/name authority differs"
        )
    for geom_id, component in component_by_geom.items():
        body_id = _artifact_integer(
            component.get("body_id"), "surface component body_id", minimum=0
        )
        if (
            component.get("geom_name") != robot_geom_names[geom_id]
            or body_id not in robot_body_names
            or component.get("body_name") != robot_body_names[body_id]
        ):
            raise ShadowIdentificationError(
                "surface component provenance differs from resolved geometry"
            )
        sample_start, sample_stop = component_sample_ranges[geom_id]
        for sample in protected_samples[sample_start:sample_stop]:
            if (
                sample["geom_id"] != geom_id
                or sample["geom_name"] != robot_geom_names[geom_id]
                or sample["body_id"] != body_id
                or sample["body_name"] != robot_body_names[body_id]
            ):
                raise ShadowIdentificationError(
                    "protected sample ledger differs from its ordered component range"
                )
    raw_pairs = _artifact_sequence(
        resolved.get("collision_enabled_pairs"),
        "resolved collision-enabled pairs",
    )
    collision_pairs: Set[Tuple[int, int]] = set()
    for index, raw_pair in enumerate(raw_pairs):
        pair_values = _artifact_sequence(
            raw_pair, "resolved collision-enabled pair[%d]" % index
        )
        if len(pair_values) != 2:
            raise ShadowIdentificationError(
                "resolved collision-enabled pair has the wrong length"
            )
        pair = (
            _artifact_integer(
                pair_values[0], "resolved pair robot geom", minimum=0
            ),
            _artifact_integer(
                pair_values[1], "resolved pair obstacle geom", minimum=0
            ),
        )
        if (
            pair in collision_pairs
            or pair[0] not in robot_geom_names
            or pair[1] not in obstacle_geom_names
        ):
            raise ShadowIdentificationError(
                "resolved collision-enabled pair authority is invalid"
            )
        collision_pairs.add(pair)
    if not collision_pairs:
        raise ShadowIdentificationError(
            "resolved collision-enabled pair authority is empty"
        )
    drift_thresholds = _artifact_mapping(
        construction.get("static_field_drift_thresholds"),
        "construction static-field drift thresholds",
    )
    for field in ("translation_m", "rotation_rad", "surface_m"):
        raw_threshold = drift_thresholds.get(field)
        if (
            isinstance(raw_threshold, bool)
            or not isinstance(raw_threshold, (int, float))
            or not math.isfinite(float(raw_threshold))
            or float(raw_threshold) < 0.0
        ):
            raise ShadowIdentificationError(
                "static-field drift threshold %s is invalid" % field
            )
        if float(raw_threshold) != registered_drift_thresholds[field]:
            raise ShadowIdentificationError(
                "construction static-field drift threshold %s differs" % field
            )
    raw_construction_alpha = construction.get("cbf_alpha_gain_per_s")
    if (
        isinstance(raw_construction_alpha, bool)
        or not isinstance(raw_construction_alpha, (int, float))
        or not math.isfinite(float(raw_construction_alpha))
        or float(raw_construction_alpha) != expected_alpha
    ):
        raise ShadowIdentificationError(
            "construction CBF alpha differs from the bound protocol"
        )
    if not _same_artifact_value(
        construction.get("settled_measurement"), measurement.get("settled_state")
    ):
        raise ShadowIdentificationError(
            "construction and rollout settled measurements differ"
        )
    authoritative_contact = _first_measurement_link_contact(
        measurement,
        link_geoms=link_geoms,
        robot_geom_names=robot_geom_names,
        obstacle_geom_names=obstacle_geom_names,
        robot_body_names=robot_body_names,
        obstacle_body_names=obstacle_body_names,
        collision_pairs=collision_pairs,
        component_by_geom=component_by_geom,
    )
    read_only = _artifact_mapping(
        construction.get("complete_integration_state_read_only_audit"),
        "construction read-only audit",
    )
    if (
        read_only.get("exact_array_equal") is not True
        or not isinstance(read_only.get("before_sha256"), str)
        or read_only.get("before_sha256") != read_only.get("after_sha256")
    ):
        raise ShadowIdentificationError("construction read-only audit differs")
    raw_arm_dofs = _artifact_sequence(
        construction.get("arm_dof_indices"),
        "construction arm DOF indices",
    )
    arm_dof_indices = tuple(
        _artifact_integer(
            value,
            "construction arm DOF index",
            minimum=0,
        )
        for value in raw_arm_dofs
    )
    if len(arm_dof_indices) != 7 or len(set(arm_dof_indices)) != 7:
        raise ShadowIdentificationError(
            "construction arm DOF indices are incomplete or duplicated"
        )
    try:
        from main.poisson_fullbody.jacobians import (
            DifferentialAuditError,
            validate_protected_sample_differential_audit,
        )

        differential_validation = validate_protected_sample_differential_audit(
            _artifact_mapping(
                construction.get("settled_link56_differential_audit"),
                "settled link56 differential audit",
            ),
            expected_samples=protected_samples,
            expected_arm_dof_indices=arm_dof_indices,
            expected_integration_state_sha256=_artifact_sha256(
                read_only.get("before_sha256"),
                "construction read-only before_sha256",
            ),
            expected_differential_audit_config=differential_audit_config,
        )
    except DifferentialAuditError as error:
        raise ShadowIdentificationError(
            "settled link56 differential audit is invalid: %s" % error
        ) from error
    if not _same_artifact_value(
        construction.get("settled_link56_differential_audit_validation"),
        differential_validation,
    ):
        raise ShadowIdentificationError(
            "serialized differential-audit validation summary differs"
        )

    identification = _artifact_mapping(
        record.get("poisson_identification"), "Poisson identification"
    )
    for field in ("observed_callback_count", "expected_callback_count"):
        if (
            _artifact_integer(
                identification.get(field),
                "identification.%s" % field,
                minimum=0,
            )
            != expected
        ):
            raise ShadowIdentificationError("identification %s differs" % field)
    trace = identification.get("trace")
    if not isinstance(trace, list) or len(trace) != expected:
        raise ShadowIdentificationError("identification trace length differs")
    first_invalidation = identification.get("first_static_field_invalidation")
    invalidation_index: Optional[int] = None
    invalidation_record: Optional[Mapping[str, Any]] = None
    if first_invalidation is not None:
        invalidation = _artifact_mapping(
            first_invalidation, "first static field invalidation"
        )
        invalidation_record = invalidation
        invalidation_index = _artifact_integer(
            invalidation.get("observation_index"),
            "first invalidation observation_index",
            minimum=0,
        )
        if invalidation_index >= expected:
            raise ShadowIdentificationError(
                "first invalidation observation is outside the replay"
            )
        if (
            _artifact_integer(
                invalidation.get("high_level_index"),
                "first invalidation high_level_index",
                minimum=0,
            )
            != invalidation_index // per_high
            or _artifact_integer(
                invalidation.get("physics_substep_index"),
                "first invalidation physics_substep_index",
                minimum=0,
            )
            != invalidation_index % per_high
            or invalidation.get("reason") != "static_selected_obstacle_drift"
        ):
            raise ShadowIdentificationError(
                "first static field invalidation coordinates differ"
            )

    attempted_count = 0
    skipped_count = 0
    valid_query_total = 0
    lhs_total = 0
    observed_first_drift_crossing: Optional[int] = None
    for index, raw_row in enumerate(trace):
        row = _artifact_mapping(raw_row, "trace[%d]" % index)
        if (
            _artifact_integer(
                row.get("observation_index"),
                "trace[%d].observation_index" % index,
                minimum=0,
            )
            != index
            or _artifact_integer(
                row.get("high_level_index"),
                "trace[%d].high_level_index" % index,
                minimum=0,
            )
            != index // per_high
            or _artifact_integer(
                row.get("physics_substep_index"),
                "trace[%d].physics_substep_index" % index,
                minimum=0,
            )
            != index % per_high
        ):
            raise ShadowIdentificationError("trace cadence differs at %d" % index)
        for field, expected_time in (
            ("time_from_first_callback_s", index * timestep),
            ("time_from_settled_state_s", (index + 1) * timestep),
        ):
            raw_time = row.get(field)
            if (
                isinstance(raw_time, bool)
                or not isinstance(raw_time, (int, float))
                or not math.isfinite(float(raw_time))
                or not math.isclose(
                    float(raw_time), expected_time, rel_tol=0.0, abs_tol=1e-12
                )
            ):
                raise ShadowIdentificationError(
                    "trace timing differs at %d for %s" % (index, field)
                )
        drift_reasons = _drift_crossing_reasons(
            row.get("drift"),
            thresholds=registered_drift_thresholds,
            expected_geom_ids=obstacle_geoms,
            label="trace[%d].drift" % index,
        )
        if drift_reasons and observed_first_drift_crossing is None:
            observed_first_drift_crossing = index
        attempted = row.get("field_query_attempted")
        static_valid = row.get("static_field_admissible_for_this_query")
        if not isinstance(attempted, bool) or not isinstance(static_valid, bool):
            raise ShadowIdentificationError(
                "trace query/static flags are not boolean at %d" % index
            )
        expected_attempted = (
            invalidation_index is None or index < invalidation_index
        )
        if attempted is not expected_attempted or static_valid is not expected_attempted:
            raise ShadowIdentificationError(
                "field queries did not stop exactly at invalidation index %d"
                % index
            )
        sample_count = _artifact_integer(
            row.get("sample_count"), "trace[%d].sample_count" % index, minimum=1
        )
        if sample_count != registered_sample_count:
            raise ShadowIdentificationError(
                "trace sample count differs from the registered field bundle"
            )
        valid_count = _artifact_integer(
            row.get("valid_query_count"),
            "trace[%d].valid_query_count" % index,
            minimum=0,
        )
        invalid_count = _artifact_integer(
            row.get("invalid_query_count"),
            "trace[%d].invalid_query_count" % index,
            minimum=0,
        )
        lhs_count = _artifact_integer(
            row.get("observed_cbf_lhs_evaluation_count"),
            "trace[%d].observed_cbf_lhs_evaluation_count" % index,
            minimum=0,
        )
        raw_groups = _artifact_sequence(
            row.get("per_geom"), "trace[%d].per_geom" % index
        )
        if attempted:
            attempted_count += 1
            if valid_count + invalid_count != sample_count or lhs_count != valid_count:
                raise ShadowIdentificationError(
                    "trace query coverage differs at %d" % index
                )
            observed_groups = {}
            group_sample_total = 0
            group_valid_total = 0
            group_invalid_total = 0
            group_lhs_total = 0
            aggregate_invalid_reasons = Counter()
            minimum_h_candidates = []
            minimum_lhs_candidates = []
            row_arm_qvel: Optional[Tuple[float, ...]] = None
            for group_index, raw_group in enumerate(raw_groups):
                group = _artifact_mapping(
                    raw_group,
                    "trace[%d].per_geom[%d]" % (index, group_index),
                )
                geom_id = _artifact_integer(
                    group.get("geom_id"), "trace per-geom geom_id", minimum=0
                )
                if geom_id in observed_groups:
                    raise ShadowIdentificationError(
                        "trace per-geom coverage has duplicate geom IDs"
                    )
                component = component_by_geom.get(geom_id)
                if component is None:
                    raise ShadowIdentificationError(
                        "trace per-geom coverage includes an unregistered geom"
                    )
                sample_id_range = component_sample_ranges[geom_id]
                observed_groups[geom_id] = group
                for field in ("geom_name", "body_id", "body_name"):
                    if group.get(field) != component.get(field):
                        raise ShadowIdentificationError(
                            "trace per-geom provenance differs at %s" % field
                        )
                group_sample = _artifact_integer(
                    group.get("sample_count"),
                    "trace per-geom sample_count",
                    minimum=1,
                )
                group_valid = _artifact_integer(
                    group.get("valid_query_count"),
                    "trace per-geom valid_query_count",
                    minimum=0,
                )
                group_invalid = _artifact_integer(
                    group.get("invalid_query_count"),
                    "trace per-geom invalid_query_count",
                    minimum=0,
                )
                group_lhs = _artifact_integer(
                    group.get("observed_cbf_lhs_evaluation_count"),
                    "trace per-geom CBF evaluation count",
                    minimum=0,
                )
                if (
                    group_sample != component.get("sample_count")
                    or group_valid + group_invalid != group_sample
                    or group_lhs != group_valid
                ):
                    raise ShadowIdentificationError(
                        "trace per-geom sample/query coverage differs"
                    )
                raw_reason_counts = _artifact_mapping(
                    group.get("invalid_reason_counts"),
                    "trace per-geom invalid reason counts",
                )
                reason_counts: Dict[str, int] = {}
                for raw_reason, raw_count in raw_reason_counts.items():
                    if not isinstance(raw_reason, str) or not raw_reason:
                        raise ShadowIdentificationError(
                            "trace invalid-query reason is invalid"
                        )
                    reason_counts[raw_reason] = _artifact_integer(
                        raw_count,
                        "trace invalid-query reason count",
                        minimum=1,
                    )
                if sum(reason_counts.values()) != group_invalid:
                    raise ShadowIdentificationError(
                        "trace invalid-query reason counts differ"
                    )
                aggregate_invalid_reasons.update(reason_counts)
                first_invalid = group.get("first_invalid_sample")
                raw_by_reason = _artifact_mapping(
                    group.get("first_invalid_sample_by_reason"),
                    "trace first invalid samples by reason",
                )
                if group_invalid:
                    first_invalid_evidence = _validate_invalid_sample_evidence(
                        first_invalid,
                        group=group,
                        registered_sample_id_range=sample_id_range,
                        expected_reason=None,
                        label="trace first invalid sample",
                    )
                    if first_invalid_evidence.get("reason") not in reason_counts:
                        raise ShadowIdentificationError(
                            "trace first invalid sample reason was not counted"
                        )
                    if set(raw_by_reason) != set(reason_counts):
                        raise ShadowIdentificationError(
                            "trace invalid-sample reason coverage differs"
                        )
                    for reason, raw_evidence in raw_by_reason.items():
                        _validate_invalid_sample_evidence(
                            raw_evidence,
                            group=group,
                            registered_sample_id_range=sample_id_range,
                            expected_reason=reason,
                            label="trace first invalid sample for %s" % reason,
                        )
                elif first_invalid is not None or raw_by_reason:
                    raise ShadowIdentificationError(
                        "trace zero-invalid group retains invalid evidence"
                    )

                minimum_h = group.get("minimum_h_sample")
                minimum_lhs = group.get("minimum_observed_cbf_lhs_sample")
                if group_valid:
                    minimum_h_record = _validate_cbf_diagnostic(
                        minimum_h,
                        group=group,
                        registered_sample_id_range=sample_id_range,
                        alpha_gain_per_s=expected_alpha,
                        label="trace per-geom minimum-h diagnostic",
                    )
                    minimum_lhs_record = _validate_cbf_diagnostic(
                        minimum_lhs,
                        group=group,
                        registered_sample_id_range=sample_id_range,
                        alpha_gain_per_s=expected_alpha,
                        label="trace per-geom minimum-LHS diagnostic",
                    )
                    if (
                        minimum_h_record.get("sample_id")
                        == minimum_lhs_record.get("sample_id")
                        and not _same_artifact_value(
                            minimum_h_record, minimum_lhs_record
                        )
                    ):
                        raise ShadowIdentificationError(
                            "one protected sample has contradictory CBF diagnostics"
                        )
                    for diagnostic in (minimum_h_record, minimum_lhs_record):
                        diagnostic_qvel = _artifact_finite_vector(
                            diagnostic.get("observed_arm_qvel_rad_s"),
                            7,
                            "trace callback arm qvel",
                        )
                        if row_arm_qvel is None:
                            row_arm_qvel = diagnostic_qvel
                        elif diagnostic_qvel != row_arm_qvel:
                            raise ShadowIdentificationError(
                                "one trace callback has contradictory arm qvel"
                            )
                    minimum_h_candidates.append(minimum_h_record)
                    minimum_lhs_candidates.append(minimum_lhs_record)
                elif minimum_h is not None or minimum_lhs is not None:
                    raise ShadowIdentificationError(
                        "trace zero-valid group retains CBF diagnostics"
                    )
                group_sample_total += group_sample
                group_valid_total += group_valid
                group_invalid_total += group_invalid
                group_lhs_total += group_lhs
            if (
                set(observed_groups) != set(component_by_geom)
                or group_sample_total != sample_count
                or group_valid_total != valid_count
                or group_invalid_total != invalid_count
                or group_lhs_total != lhs_count
            ):
                raise ShadowIdentificationError(
                    "trace per-geom aggregates differ at %d" % index
                )
            expected_reason_counts = dict(sorted(aggregate_invalid_reasons.items()))
            if not _same_artifact_value(
                row.get("invalid_reason_counts"), expected_reason_counts
            ):
                raise ShadowIdentificationError(
                    "trace row invalid-reason aggregates differ"
                )
            expected_minimum_h = (
                min(minimum_h_candidates, key=lambda value: float(value["h_m2"]))
                if minimum_h_candidates
                else None
            )
            expected_minimum_lhs = (
                min(
                    minimum_lhs_candidates,
                    key=lambda value: float(
                        value["observed_cbf_lhs_m2_per_s"]
                    ),
                )
                if minimum_lhs_candidates
                else None
            )
            for sample_field, value_field, expected_sample in (
                ("minimum_h_sample", "minimum_h_m2", expected_minimum_h),
                (
                    "minimum_observed_cbf_lhs_sample",
                    "minimum_observed_cbf_lhs_m2_per_s",
                    expected_minimum_lhs,
                ),
            ):
                if not _same_artifact_value(row.get(sample_field), expected_sample):
                    raise ShadowIdentificationError(
                        "trace row %s differs from per-geom diagnostics"
                        % sample_field
                    )
                expected_value = (
                    float(expected_sample["h_m2"])
                    if sample_field == "minimum_h_sample"
                    and expected_sample is not None
                    else (
                        float(expected_sample["observed_cbf_lhs_m2_per_s"])
                        if expected_sample is not None
                        else None
                    )
                )
                raw_value = row.get(value_field)
                if expected_value is None:
                    if raw_value is not None:
                        raise ShadowIdentificationError(
                            "trace row %s must be absent" % value_field
                        )
                elif (
                    isinstance(raw_value, bool)
                    or not isinstance(raw_value, (int, float))
                    or not math.isfinite(float(raw_value))
                    or float(raw_value) != expected_value
                ):
                    raise ShadowIdentificationError(
                        "trace row %s differs from its diagnostic" % value_field
                    )
        else:
            skipped_count += 1
            if (
                row.get("skip_reason")
                != "static_selected_obstacle_drift_invalidated_field"
                or valid_count != 0
                or invalid_count != 0
                or lhs_count != 0
                or len(raw_groups) != 0
                or row.get("invalid_reason_counts") != {}
                or row.get("minimum_h_m2") is not None
                or row.get("minimum_h_sample") is not None
                or row.get("minimum_observed_cbf_lhs_m2_per_s") is not None
                or row.get("minimum_observed_cbf_lhs_sample") is not None
            ):
                raise ShadowIdentificationError(
                    "invalidated trace row differs at %d" % index
                )
        valid_query_total += valid_count
        lhs_total += lhs_count

    if observed_first_drift_crossing != invalidation_index:
        raise ShadowIdentificationError(
            "first static-field drift crossing differs from invalidation"
        )
    if invalidation_record is not None:
        invalidation_row = _artifact_mapping(
            trace[invalidation_index], "trace row at first invalidation"
        )
        drift_reasons = _drift_crossing_reasons(
            invalidation_row.get("drift"),
            thresholds=registered_drift_thresholds,
            expected_geom_ids=obstacle_geoms,
            label="trace drift at first invalidation",
        )
        if (
            not _same_artifact_value(
                invalidation_record.get("drift"), invalidation_row.get("drift")
            )
            or list(
                _artifact_sequence(
                    invalidation_record.get("threshold_crossing_reasons"),
                    "first invalidation threshold-crossing reasons",
                )
            )
            != list(drift_reasons)
        ):
            raise ShadowIdentificationError(
                "first invalidation evidence differs from the trace crossing"
            )

    if identification.get("every_valid_query_has_observed_cbf_lhs") is not True:
        raise ShadowIdentificationError("CBF residual coverage flag differs")
    for field, value in (
        ("field_query_callback_count", attempted_count),
        (
            "field_query_skipped_after_static_invalidation_count",
            skipped_count,
        ),
        ("valid_field_query_count", valid_query_total),
        ("observed_cbf_lhs_evaluation_count", lhs_total),
    ):
        if (
            _artifact_integer(
                identification.get(field),
                "identification.%s" % field,
                minimum=0,
            )
            != value
        ):
            raise ShadowIdentificationError(
                "identification aggregate %s differs" % field
            )
    trace_hash = hashlib.sha256(_canonical(trace)).hexdigest()
    if identification.get("trace_sha256") != trace_hash:
        raise ShadowIdentificationError("identification trace SHA-256 differs")

    exception_count = _artifact_integer(
        record.get("monitor_static_drift_exception_count"),
        "monitor drift exception count",
        minimum=0,
    )
    exception_rows = record.get("monitor_static_drift_exceptions")
    if (
        not isinstance(exception_rows, list)
        or exception_count != 0
        or exception_rows
    ):
        raise ShadowIdentificationError("monitor drift exception ledger differs")

    assessment = _artifact_mapping(
        identification.get("contact_prediction_assessment"),
        "contact prediction assessment",
    )
    assessment_name = assessment.get("assessment")
    allowed = {
        "no_link56_contact_outcome",
        "link56_contact_already_present_after_settling",
        "no_registered_precontact_warning_on_contact_geom",
        "registered_warning_preceded_link56_contact",
        "registered_warning_coincident_with_link56_contact",
        "registered_warning_followed_link56_contact",
    }
    if assessment_name not in allowed:
        raise ShadowIdentificationError("contact prediction assessment is unknown")
    contact = assessment.get("first_link56_contact")
    primary = assessment.get("primary_registered_warning")
    lead = assessment.get("lead_physics_substeps")
    lead_time = assessment.get("lead_time_s")
    if not _same_artifact_value(contact, authoritative_contact):
        raise ShadowIdentificationError(
            "contact assessment differs from the MuJoCo measurement ledger"
        )
    signals = _artifact_mapping(identification.get("signals"), "identification signals")
    contact_geom_for_signal = (
        _artifact_integer(
            authoritative_contact.get("robot_geom_id"),
            "authoritative contact robot_geom_id",
            minimum=0,
        )
        if authoritative_contact is not None
        else None
    )
    expected_any_warning = (
        _first_trace_signal(trace, geom_id=contact_geom_for_signal, kind="any_invalid")
        if contact_geom_for_signal is not None
        else None
    )
    expected_lhs_warning = (
        _first_trace_signal(
            trace, geom_id=contact_geom_for_signal, kind="cbf_lhs_negative"
        )
        if contact_geom_for_signal is not None
        else None
    )
    if not _same_artifact_value(
        signals.get("first_any_fail_closed_query_on_contact_geom"),
        expected_any_warning,
    ) or not _same_artifact_value(
        signals.get("first_observed_minimum_cbf_lhs_negative_on_contact_geom"),
        expected_lhs_warning,
    ):
        raise ShadowIdentificationError(
            "contact-geom warning signals differ from the raw trace"
        )
    warning_candidates = [
        value
        for value in (expected_any_warning, expected_lhs_warning)
        if value is not None
    ]
    expected_primary = (
        min(
            warning_candidates,
            key=lambda value: int(value["observation_index"]),
        )
        if warning_candidates
        else None
    )
    if not _same_artifact_value(primary, expected_primary):
        raise ShadowIdentificationError(
            "primary registered warning differs from the raw trace"
        )
    if contact is None:
        if (
            assessment_name != "no_link56_contact_outcome"
            or primary is not None
            or lead is not None
            or lead_time is not None
        ):
            raise ShadowIdentificationError("no-contact assessment differs")
        return
    contact_record = _artifact_mapping(contact, "first link56 contact")
    contact_geom = _artifact_integer(
        contact_record.get("robot_geom_id"), "contact robot_geom_id", minimum=0
    )
    if (
        contact_geom not in link_geoms
        or contact_record.get("is_physical_nonpositive_distance_contact") is not True
    ):
        raise ShadowIdentificationError(
            "first link56 contact is not a physical resolved-link contact"
        )
    raw_contact_observation = contact_record.get("observation_index")
    if raw_contact_observation is None:
        if (
            assessment_name != "link56_contact_already_present_after_settling"
            or lead is not None
            or lead_time is not None
            or contact_record.get("source_phase")
            != "settled_post_integration_recomputed"
        ):
            raise ShadowIdentificationError("settled-contact assessment differs")
        if primary is not None and not isinstance(primary, Mapping):
            raise ShadowIdentificationError(
                "settled-contact primary warning is malformed"
            )
        return
    contact_observation = _artifact_integer(
        raw_contact_observation, "contact observation_index", minimum=0
    )
    if contact_observation >= expected or contact_record.get("source_phase") not in {
        "live_solver_phase_preintegration_geometry",
        "post_integration_recomputed",
    }:
        raise ShadowIdentificationError("rollout contact coordinate differs")
    contact_high = _artifact_integer(
        contact_record.get("high_level_index"), "contact high_level_index", minimum=0
    )
    contact_inner = _artifact_integer(
        contact_record.get("inner_control_index"),
        "contact inner_control_index",
        minimum=0,
    )
    contact_physics = _artifact_integer(
        contact_record.get("physics_substep_index"),
        "contact physics_substep_index",
        minimum=0,
    )
    if (
        contact_inner >= inner_count
        or contact_physics >= physics_count
        or (contact_high * inner_count + contact_inner) * physics_count
        + contact_physics
        != contact_observation
    ):
        raise ShadowIdentificationError("rollout contact cadence differs")
    if primary is None:
        if (
            assessment_name != "no_registered_precontact_warning_on_contact_geom"
            or lead is not None
            or lead_time is not None
        ):
            raise ShadowIdentificationError("no-warning assessment differs")
        return
    warning = _artifact_mapping(primary, "primary registered warning")
    warning_observation = _artifact_integer(
        warning.get("observation_index"), "warning observation_index", minimum=0
    )
    warning_geom = _artifact_integer(
        warning.get("geom_id"), "warning geom_id", minimum=0
    )
    lead_substeps = _artifact_integer(lead, "warning lead_physics_substeps")
    if warning_observation >= expected or warning_geom != contact_geom:
        raise ShadowIdentificationError("warning/contact identity differs")
    contact_boundary_index = rollout_contact_boundary_index(
        contact_observation, str(contact_record.get("source_phase"))
    )
    if warning_observation + lead_substeps != contact_boundary_index:
        raise ShadowIdentificationError("warning lead arithmetic differs")
    if (
        isinstance(lead_time, bool)
        or not isinstance(lead_time, (int, float))
        or not math.isfinite(float(lead_time))
        or not math.isclose(
            float(lead_time), lead_substeps * timestep, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ShadowIdentificationError("warning lead time differs")
    expected_assessment = (
        "registered_warning_preceded_link56_contact"
        if lead_substeps > 0
        else (
            "registered_warning_coincident_with_link56_contact"
            if lead_substeps == 0
            else "registered_warning_followed_link56_contact"
        )
    )
    if assessment_name != expected_assessment:
        raise ShadowIdentificationError("warning/contact assessment differs")


def _raw_model_data(sim: Any) -> Tuple[Any, Any]:
    if not hasattr(sim, "model") or not hasattr(sim, "data"):
        raise TypeError("sim must expose model and data")
    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    return model, data


def _rotation_angle(reference: Any, current: Any, np: Any) -> float:
    relative = reference.T @ current
    cosine = (float(np.trace(relative)) - 1.0) * 0.5
    return float(math.acos(max(-1.0, min(1.0, cosine))))


def _reason_text(query: Any) -> str:
    reason = getattr(query, "reason", None)
    value = getattr(reason, "value", reason)
    return "unknown_invalid_query" if value is None else str(value)


def _sample_record(sample: BodySample) -> Dict[str, Any]:
    return {
        "sample_id": int(sample.sample_id),
        "body_id": int(sample.body_id),
        "body_name": str(sample.body_name),
        "geom_id": int(sample.geom_id),
        "geom_name": str(sample.geom_name),
    }


@dataclass(frozen=True)
class StaticDriftThresholds:
    translation_m: float
    rotation_rad: float
    surface_m: float

    def __post_init__(self) -> None:
        for value, label in (
            (self.translation_m, "translation_m"),
            (self.rotation_rad, "rotation_rad"),
            (self.surface_m, "surface_m"),
        ):
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError("%s must be finite and nonnegative" % label)


def robot_root_body_ids(
    robot_body_ids: Iterable[int], body_parent_ids: Sequence[int]
) -> Tuple[int, ...]:
    """Return the minimal authoritative roots of a registered robot-body set."""

    bodies = tuple(sorted(set(int(value) for value in robot_body_ids)))
    if not bodies or any(value <= 0 or value >= len(body_parent_ids) for value in bodies):
        raise ValueError("robot body IDs must be nonempty, unique, non-world model IDs")
    body_set = set(bodies)
    roots = tuple(value for value in bodies if int(body_parent_ids[value]) not in body_set)
    if not roots:
        raise ValueError("robot body set has no root outside itself")
    return roots


class StaticPoissonShadowObserver:
    """Evaluate a certified field on every callback without mutating physics."""

    def __init__(
        self,
        *,
        field: Any,
        samples: Iterable[BodySample],
        settled_obstacle_boxes: Iterable[OrientedBox],
        arm_dof_indices: Sequence[int],
        alpha_gain_per_s: float,
        physics_timestep_s: float,
        drift_thresholds: StaticDriftThresholds,
        physics_substeps_per_high_level_action: int = 25,
        snapshot_provider: Optional[Callable[[Any], Tuple[Any, Any]]] = None,
        jacobian_provider: Optional[Callable[..., Tuple[Any, Any]]] = None,
    ) -> None:
        _, np = _modules()
        records = tuple(samples)
        boxes = tuple(settled_obstacle_boxes)
        dofs = tuple(int(value) for value in arm_dof_indices)
        if not records or any(not isinstance(value, BodySample) for value in records):
            raise ValueError("samples must contain at least one BodySample")
        if tuple(sample.sample_id for sample in records) != tuple(range(len(records))):
            raise ValueError("shadow sample IDs must be contiguous and deterministic")
        if not boxes or any(not isinstance(value, OrientedBox) for value in boxes):
            raise ValueError("settled_obstacle_boxes must contain OrientedBox records")
        geom_ids = tuple(int(box.geom_id) for box in boxes)
        if len(geom_ids) != len(set(geom_ids)):
            raise ValueError("settled obstacle geom IDs must be unique")
        if len(dofs) != 7 or len(set(dofs)) != 7 or any(value < 0 for value in dofs):
            raise ValueError("arm_dof_indices must contain seven unique nonnegative IDs")
        alpha = float(alpha_gain_per_s)
        timestep = float(physics_timestep_s)
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha_gain_per_s must be finite and positive")
        if not math.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("physics_timestep_s must be finite and positive")
        if (
            isinstance(physics_substeps_per_high_level_action, bool)
            or int(physics_substeps_per_high_level_action)
            != physics_substeps_per_high_level_action
            or int(physics_substeps_per_high_level_action) <= 0
        ):
            raise ValueError("physics_substeps_per_high_level_action must be positive")
        self._field = field
        self._samples = records
        self._boxes = boxes
        self._settled_vertices = {
            int(box.geom_id): np.asarray(box.vertices(), dtype=np.float64)
            for box in boxes
        }
        self._dofs = dofs
        self._alpha = alpha
        self._timestep = timestep
        self._drift_thresholds = drift_thresholds
        self._substeps_per_high = int(physics_substeps_per_high_level_action)
        self._snapshot_provider = snapshot_provider
        self._forwarded_clone = None
        self._snapshot_model = None
        self._jacobian_provider = jacobian_provider or point_translational_jacobian
        self._trace = []
        self._static_valid = True
        self._first_invalidation: Optional[Dict[str, Any]] = None

    @property
    def trace(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(self._trace)

    @property
    def static_field_valid(self) -> bool:
        return self._static_valid

    def _snapshot(self, sim: Any) -> Tuple[Any, Any]:
        if self._snapshot_provider is not None:
            return self._snapshot_provider(sim)
        model, live_data = _raw_model_data(sim)
        if self._snapshot_model is None:
            self._snapshot_model = model
        elif model is not self._snapshot_model:
            raise ShadowIdentificationError(
                "shadow callback received a different MuJoCo model"
            )
        self._forwarded_clone = clone_forwarded_state(
            model, live_data, reusable_clone=self._forwarded_clone
        )
        return model, self._forwarded_clone

    def _drift(self, data: Any) -> Dict[str, Any]:
        _, np = _modules()
        geoms = []
        for settled in self._boxes:
            geom_id = int(settled.geom_id)
            position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
            rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
                raise ShadowIdentificationError("selected-obstacle pose is non-finite")
            translation = float(np.linalg.norm(position - settled.center))
            rotation_angle = _rotation_angle(settled.R, rotation, np)
            current = OrientedBox(
                center=position,
                R=rotation,
                half_extents=settled.half_extents,
                geom_id=geom_id,
            )
            surface = float(
                np.max(
                    np.linalg.norm(
                        np.asarray(current.vertices(), dtype=np.float64)
                        - self._settled_vertices[geom_id],
                        axis=1,
                    )
                )
            )
            reasons = []
            if translation > float(self._drift_thresholds.translation_m):
                reasons.append("translation_threshold_crossed")
            if rotation_angle > float(self._drift_thresholds.rotation_rad):
                reasons.append("rotation_threshold_crossed")
            if surface > float(self._drift_thresholds.surface_m):
                reasons.append("surface_threshold_crossed")
            geoms.append(
                {
                    "geom_id": geom_id,
                    "translation_m": translation,
                    "rotation_rad": rotation_angle,
                    "maximum_surface_point_displacement_m": surface,
                    "threshold_crossing_reasons": reasons,
                }
            )
        return {
            "maximum_translation_m": max(row["translation_m"] for row in geoms),
            "maximum_rotation_rad": max(row["rotation_rad"] for row in geoms),
            "maximum_surface_point_displacement_m": max(
                row["maximum_surface_point_displacement_m"] for row in geoms
            ),
            "geoms": geoms,
        }

    def _invalidate(
        self,
        *,
        observation_index: int,
        high_level_index: int,
        physics_substep_index: int,
        drift: Mapping[str, Any],
    ) -> None:
        if not self._static_valid:
            return
        reasons = sorted(
            {
                reason
                for geom in drift["geoms"]
                for reason in geom["threshold_crossing_reasons"]
            }
        )
        if not reasons:
            return
        self._static_valid = False
        self._first_invalidation = {
            "observation_index": int(observation_index),
            "high_level_index": int(high_level_index),
            "physics_substep_index": int(physics_substep_index),
            "reason": "static_selected_obstacle_drift",
            "threshold_crossing_reasons": reasons,
            "drift": dict(drift),
            "semantics": (
                "field queries stop before evaluating the first post-state that "
                "crosses any registered static-obstacle drift threshold"
            ),
        }

    def observe(
        self,
        sim: Any,
        *,
        high_level_index: int,
        physics_substep_index: int,
    ) -> Mapping[str, Any]:
        expected = len(self._trace)
        observed = (
            int(high_level_index) * self._substeps_per_high
            + int(physics_substep_index)
        )
        if observed != expected:
            raise ShadowIdentificationError(
                "shadow callback gap/duplicate: expected %d, got %d"
                % (expected, observed)
            )
        model, data = self._snapshot(sim)
        _, np = _modules()
        drift = self._drift(data)
        self._invalidate(
            observation_index=observed,
            high_level_index=int(high_level_index),
            physics_substep_index=int(physics_substep_index),
            drift=drift,
        )
        base = {
            "observation_index": observed,
            "high_level_index": int(high_level_index),
            "physics_substep_index": int(physics_substep_index),
            "time_from_first_callback_s": float(observed * self._timestep),
            "time_from_settled_state_s": float((observed + 1) * self._timestep),
            "static_field_admissible_for_this_query": bool(self._static_valid),
            "drift": drift,
        }
        if not self._static_valid:
            row = dict(base)
            row.update(
                {
                    "field_query_attempted": False,
                    "skip_reason": "static_selected_obstacle_drift_invalidated_field",
                    "sample_count": len(self._samples),
                    "valid_query_count": 0,
                    "invalid_query_count": 0,
                    "observed_cbf_lhs_evaluation_count": 0,
                    "invalid_reason_counts": {},
                    "minimum_h_m2": None,
                    "minimum_h_sample": None,
                    "minimum_observed_cbf_lhs_m2_per_s": None,
                    "minimum_observed_cbf_lhs_sample": None,
                    "per_geom": [],
                }
            )
            self._trace.append(row)
            return row

        qvel = np.asarray(data.qvel, dtype=np.float64)
        if any(value >= qvel.shape[0] for value in self._dofs):
            raise ShadowIdentificationError("arm DOF index is outside data.qvel")
        arm_velocity = qvel[np.asarray(self._dofs, dtype=np.int64)]
        if arm_velocity.shape != (7,) or not np.all(np.isfinite(arm_velocity)):
            raise ShadowIdentificationError("instantaneous arm velocity is invalid")

        by_geom: Dict[int, Dict[str, Any]] = {}
        all_invalid = Counter()
        for sample in self._samples:
            point = sample.world_point(data)
            query = self._field.query(point)
            group = by_geom.setdefault(
                int(sample.geom_id),
                {
                    "geom_id": int(sample.geom_id),
                    "geom_name": str(sample.geom_name),
                    "body_id": int(sample.body_id),
                    "body_name": str(sample.body_name),
                    "sample_count": 0,
                    "valid_query_count": 0,
                    "invalid_query_count": 0,
                    "observed_cbf_lhs_evaluation_count": 0,
                    "invalid_reason_counts": Counter(),
                    "first_invalid_sample": None,
                    "first_invalid_sample_by_reason": {},
                    "minimum_h_sample": None,
                    "minimum_observed_cbf_lhs_sample": None,
                },
            )
            group["sample_count"] += 1
            if not bool(getattr(query, "valid", False)):
                reason = _reason_text(query)
                group["invalid_query_count"] += 1
                group["invalid_reason_counts"][reason] += 1
                all_invalid[reason] += 1
                if group["first_invalid_sample"] is None:
                    group["first_invalid_sample"] = {
                        **_sample_record(sample),
                        "reason": reason,
                        "point_world_m": [float(value) for value in point],
                    }
                if reason not in group["first_invalid_sample_by_reason"]:
                    group["first_invalid_sample_by_reason"][reason] = {
                        **_sample_record(sample),
                        "reason": reason,
                        "point_world_m": [float(value) for value in point],
                    }
                continue
            value = getattr(query, "value", None)
            gradient = getattr(query, "gradient", None)
            if value is None or gradient is None:
                raise ShadowIdentificationError("valid field query omitted h or gradient")
            value = float(value)
            gradient_array = np.asarray(gradient, dtype=np.float64)
            if (
                not math.isfinite(value)
                or gradient_array.shape != (3,)
                or not np.all(np.isfinite(gradient_array))
            ):
                raise ShadowIdentificationError("valid field query is non-finite")
            group["valid_query_count"] += 1
            jacobian_point, jacobian = self._jacobian_provider(
                model, data, sample, self._dofs
            )
            jacobian_point = np.asarray(jacobian_point, dtype=np.float64)
            jacobian = np.asarray(jacobian, dtype=np.float64)
            if jacobian_point.shape != (3,) or jacobian.shape != (3, 7):
                raise ShadowIdentificationError("protected-sample Jacobian shape is invalid")
            if not np.all(np.isfinite(jacobian)) or not np.all(
                np.isfinite(jacobian_point)
            ):
                raise ShadowIdentificationError("protected-sample Jacobian is non-finite")
            if float(np.max(np.abs(jacobian_point - point))) > 1.0e-12:
                raise ShadowIdentificationError(
                    "field query and Jacobian use different world points"
                )
            point_velocity = jacobian @ arm_velocity
            directional = float(gradient_array @ point_velocity)
            lhs = float(directional + self._alpha * value)
            if not math.isfinite(directional) or not math.isfinite(lhs):
                raise ShadowIdentificationError("observed CBF diagnostic is non-finite")
            group["observed_cbf_lhs_evaluation_count"] += 1
            diagnostic = {
                **_sample_record(sample),
                "point_world_m": [float(item) for item in point],
                "h_m2": value,
                "gradient_world_m": [float(item) for item in gradient_array],
                "gradient_norm_m": float(np.linalg.norm(gradient_array)),
                "observed_arm_qvel_rad_s": [float(item) for item in arm_velocity],
                "point_translational_jacobian_arm_3x7": [
                    [float(item) for item in row] for row in jacobian
                ],
                "observed_grad_h_J_qdot_m2_per_s": directional,
                "alpha_h_m2_per_s": float(self._alpha * value),
                "observed_cbf_lhs_m2_per_s": lhs,
                "observed_cbf_diagnostic_semantics": (
                    "post-integration instantaneous MuJoCo point Jacobian and qvel "
                    "at this protected surface sample; not a QP nominal"
                ),
            }
            previous_h = group["minimum_h_sample"]
            if previous_h is None or value < float(previous_h["h_m2"]):
                group["minimum_h_sample"] = diagnostic
            previous_lhs = group["minimum_observed_cbf_lhs_sample"]
            if previous_lhs is None or lhs < float(
                previous_lhs["observed_cbf_lhs_m2_per_s"]
            ):
                group["minimum_observed_cbf_lhs_sample"] = diagnostic

        per_geom = []
        for geom_id in sorted(by_geom):
            group = by_geom[geom_id]
            if int(group["observed_cbf_lhs_evaluation_count"]) != int(
                group["valid_query_count"]
            ):
                raise ShadowIdentificationError(
                    "not every valid field query received a CBF residual evaluation"
                )
            group["invalid_reason_counts"] = dict(
                sorted(group["invalid_reason_counts"].items())
            )
            group["first_invalid_sample_by_reason"] = dict(
                sorted(group["first_invalid_sample_by_reason"].items())
            )
            per_geom.append(group)

        minimum_h_samples = [
            row["minimum_h_sample"]
            for row in per_geom
            if row["minimum_h_sample"] is not None
        ]
        minimum_lhs_samples = [
            row["minimum_observed_cbf_lhs_sample"]
            for row in per_geom
            if row["minimum_observed_cbf_lhs_sample"] is not None
        ]
        overall_h = (
            min(minimum_h_samples, key=lambda row: float(row["h_m2"]))
            if minimum_h_samples
            else None
        )
        overall_lhs = (
            min(
                minimum_lhs_samples,
                key=lambda row: float(row["observed_cbf_lhs_m2_per_s"]),
            )
            if minimum_lhs_samples
            else None
        )
        row = dict(base)
        row.update(
            {
                "field_query_attempted": True,
                "sample_count": len(self._samples),
                "valid_query_count": sum(
                    int(group["valid_query_count"]) for group in per_geom
                ),
                "invalid_query_count": sum(
                    int(group["invalid_query_count"]) for group in per_geom
                ),
                "observed_cbf_lhs_evaluation_count": sum(
                    int(group["observed_cbf_lhs_evaluation_count"])
                    for group in per_geom
                ),
                "invalid_reason_counts": dict(sorted(all_invalid.items())),
                "minimum_h_m2": (
                    float(overall_h["h_m2"]) if overall_h is not None else None
                ),
                "minimum_h_sample": (
                    dict(overall_h) if overall_h is not None else None
                ),
                "minimum_observed_cbf_lhs_m2_per_s": (
                    float(overall_lhs["observed_cbf_lhs_m2_per_s"])
                    if overall_lhs is not None
                    else None
                ),
                "minimum_observed_cbf_lhs_sample": (
                    dict(overall_lhs) if overall_lhs is not None else None
                ),
                "per_geom": per_geom,
            }
        )
        self._trace.append(row)
        return row

    @staticmethod
    def _first_signal(
        trace: Sequence[Mapping[str, Any]],
        *,
        geom_id: Optional[int],
        kind: str,
    ) -> Optional[Dict[str, Any]]:
        for row in trace:
            if not row.get("field_query_attempted"):
                continue
            groups = row.get("per_geom", [])
            if geom_id is not None:
                groups = [
                    group
                    for group in groups
                    if int(group["geom_id"]) == int(geom_id)
                ]
            for group in groups:
                evidence: Optional[Mapping[str, Any]] = None
                if kind == "invalid_cell" and int(
                    group["invalid_reason_counts"].get("invalid_cell", 0)
                ) > 0:
                    evidence = group.get("first_invalid_sample_by_reason", {}).get(
                        "invalid_cell"
                    )
                elif kind == "any_invalid" and int(group["invalid_query_count"]) > 0:
                    evidence = group.get("first_invalid_sample")
                elif kind == "cbf_lhs_negative":
                    minimum_lhs = group.get("minimum_observed_cbf_lhs_sample")
                    if (
                        minimum_lhs is not None
                        and float(minimum_lhs["observed_cbf_lhs_m2_per_s"]) < 0.0
                    ):
                        evidence = minimum_lhs
                if evidence is not None:
                    return {
                        "observation_index": int(row["observation_index"]),
                        "high_level_index": int(row["high_level_index"]),
                        "physics_substep_index": int(row["physics_substep_index"]),
                        "signal_kind": kind,
                        "geom_id": int(group["geom_id"]),
                        "geom_name": str(group["geom_name"]),
                        "body_id": int(group["body_id"]),
                        "body_name": str(group["body_name"]),
                        "evidence": dict(evidence),
                    }
        return None

    def result(
        self,
        *,
        expected_callback_count: int,
        first_link56_contact: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if len(self._trace) != int(expected_callback_count):
            raise ShadowIdentificationError(
                "shadow trace is incomplete: expected %d callbacks, observed %d"
                % (int(expected_callback_count), len(self._trace))
            )
        contact_observation: Optional[int] = None
        contact_geom: Optional[int] = None
        contact_phase: Optional[str] = None
        if first_link56_contact is not None:
            raw_observation = first_link56_contact.get("observation_index")
            if raw_observation is not None:
                contact_observation = int(raw_observation)
                contact_phase = str(first_link56_contact.get("source_phase", ""))
                if contact_phase not in {
                    "live_solver_phase_preintegration_geometry",
                    "post_integration_recomputed",
                }:
                    raise ShadowIdentificationError(
                        "rollout contact lacks an authoritative temporal phase"
                    )
            contact_geom = int(first_link56_contact["robot_geom_id"])

        signals = {
            "first_any_fail_closed_query": self._first_signal(
                self._trace, geom_id=None, kind="any_invalid"
            ),
            "first_invalid_cell_query": self._first_signal(
                self._trace, geom_id=None, kind="invalid_cell"
            ),
            "first_observed_minimum_cbf_lhs_negative": self._first_signal(
                self._trace, geom_id=None, kind="cbf_lhs_negative"
            ),
            "first_any_fail_closed_query_on_contact_geom": (
                self._first_signal(
                    self._trace, geom_id=contact_geom, kind="any_invalid"
                )
                if contact_geom is not None
                else None
            ),
            "first_invalid_cell_query_on_contact_geom": (
                self._first_signal(
                    self._trace, geom_id=contact_geom, kind="invalid_cell"
                )
                if contact_geom is not None
                else None
            ),
            "first_observed_minimum_cbf_lhs_negative_on_contact_geom": (
                self._first_signal(
                    self._trace,
                    geom_id=contact_geom,
                    kind="cbf_lhs_negative",
                )
                if contact_geom is not None
                else None
            ),
        }
        candidates = [
            signals["first_any_fail_closed_query_on_contact_geom"],
            signals[
                "first_observed_minimum_cbf_lhs_negative_on_contact_geom"
            ],
        ]
        candidates = [value for value in candidates if value is not None]
        primary = (
            min(candidates, key=lambda value: int(value["observation_index"]))
            if candidates
            else None
        )
        if first_link56_contact is None:
            assessment = "no_link56_contact_outcome"
            lead_substeps = None
        elif contact_observation is None:
            assessment = "link56_contact_already_present_after_settling"
            lead_substeps = None
        elif primary is None:
            assessment = "no_registered_precontact_warning_on_contact_geom"
            lead_substeps = None
        else:
            # Row N is a post-integration field query at (N + 1) * dt.  A
            # live-solver record at N describes the pre-integration boundary
            # N * dt, whereas a forwarded post-state contact at N describes
            # (N + 1) * dt.  Compare physical boundaries, not raw row labels.
            contact_boundary_index = rollout_contact_boundary_index(
                contact_observation, str(contact_phase)
            )
            lead_substeps = contact_boundary_index - int(
                primary["observation_index"]
            )
            if lead_substeps > 0:
                assessment = "registered_warning_preceded_link56_contact"
            elif lead_substeps == 0:
                assessment = "registered_warning_coincident_with_link56_contact"
            else:
                assessment = "registered_warning_followed_link56_contact"
        valid_h = [
            float(row["minimum_h_m2"])
            for row in self._trace
            if row.get("minimum_h_m2") is not None
        ]
        valid_lhs = [
            float(row["minimum_observed_cbf_lhs_m2_per_s"])
            for row in self._trace
            if row.get("minimum_observed_cbf_lhs_m2_per_s") is not None
        ]
        valid_query_count = sum(
            int(row["valid_query_count"])
            for row in self._trace
            if row.get("field_query_attempted")
        )
        lhs_evaluation_count = sum(
            int(row["observed_cbf_lhs_evaluation_count"])
            for row in self._trace
            if row.get("field_query_attempted")
        )
        if lhs_evaluation_count != valid_query_count:
            raise ShadowIdentificationError(
                "aggregate CBF residual coverage differs from valid-query coverage"
            )
        trace_hash = hashlib.sha256(_canonical(self._trace)).hexdigest()
        return {
            "observed_callback_count": len(self._trace),
            "expected_callback_count": int(expected_callback_count),
            "field_query_callback_count": sum(
                bool(row["field_query_attempted"]) for row in self._trace
            ),
            "field_query_skipped_after_static_invalidation_count": sum(
                not bool(row["field_query_attempted"]) for row in self._trace
            ),
            "first_static_field_invalidation": self._first_invalidation,
            "minimum_valid_h_m2": min(valid_h) if valid_h else None,
            "minimum_observed_cbf_lhs_m2_per_s": (
                min(valid_lhs) if valid_lhs else None
            ),
            "valid_field_query_count": valid_query_count,
            "observed_cbf_lhs_evaluation_count": lhs_evaluation_count,
            "every_valid_query_has_observed_cbf_lhs": True,
            "signal_definitions": {
                "any_fail_closed_query": (
                    "any typed query invalidity; the registered active policy must "
                    "fail closed for outside-grid, invalid-cell, nondifferentiable-"
                    "face, nonregular-zero-cell, and nonfinite queries, although "
                    "not all reasons indicate obstacle proximity"
                ),
                "invalid_cell_query": (
                    "sample left the certified connected free-cell domain; this "
                    "alone does not distinguish buffered-obstacle, outer-boundary, "
                    "or disconnected-component causes"
                ),
                "observed_minimum_cbf_lhs_negative": (
                    "minimum over every valid protected sample of grad(h)^T J "
                    "qdot + alpha*h < 0, using post-integration MuJoCo qvel; "
                    "this is exhaustive for the registered sampled constraints "
                    "but is not a hypothetical QP solve"
                ),
            },
            "signals": signals,
            "contact_prediction_assessment": {
                "assessment": assessment,
                "first_link56_contact": (
                    dict(first_link56_contact)
                    if first_link56_contact is not None
                    else None
                ),
                "primary_registered_warning": primary,
                "lead_physics_substeps": lead_substeps,
                "lead_time_s": (
                    float(lead_substeps * self._timestep)
                    if lead_substeps is not None
                    else None
                ),
                "contact_source_phase": contact_phase,
                "same_callback_phase_semantics": (
                    "field query N is at the post-integration boundary (N+1)*dt; "
                    "a live_solver_phase_preintegration_geometry contact at N is "
                    "at N*dt, while a post_integration_recomputed contact at N is "
                    "at (N+1)*dt"
                ),
                "interpretation_limit": (
                    "one outcome-conditioned canary tests temporal identification "
                    "only; it does not establish specificity or active safety efficacy"
                ),
            },
            "hypothetical_qp": {
                "computed": False,
                "reason": (
                    "the unchanged OSC replay does not expose the registered "
                    "joint-velocity adapter nominal; fabricating a QP nominal would "
                    "not be semantically paired"
                ),
            },
            "trace_sha256": trace_hash,
            "trace": list(self._trace),
        }


__all__ = [
    "ShadowIdentificationError",
    "StaticDriftThresholds",
    "StaticPoissonShadowObserver",
    "robot_root_body_ids",
    "registered_filter_update_available_before_contact",
    "rollout_contact_boundary_index",
    "validate_shadow_replay_record",
]
