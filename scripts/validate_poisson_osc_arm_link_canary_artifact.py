#!/usr/bin/env python3
"""Independent complete-only consumer for the post-OSC arm-link canary."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from scripts import run_poisson_fast_feasibility as fast


class OscCanaryValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OscCanaryValidationError(message)


def _canonical_sha256(value: Any) -> str:
    return fast._sha256(fast._canonical(value))


def _float64_sha256(value: Any, np: Any) -> str:
    array = np.asarray(value, dtype=np.float64)
    _require(np.all(np.isfinite(array)), "array hash input is non-finite")
    contiguous = np.ascontiguousarray(array)
    header = fast._canonical(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}
    )
    digest = hashlib.sha256()
    digest.update(b"vlsa-table1-array-v1\0")
    digest.update(header)
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _float64_vector7_bytes_and_sha256(value: Any) -> Tuple[bytes, str]:
    """Validate and hash one 7D vector without an allocation-only NumPy import."""

    _require(
        isinstance(value, (list, tuple)) and len(value) == 7,
        "7D float64 vector is malformed",
    )
    numbers = []
    for item in value:
        _require(
            not isinstance(item, bool)
            and isinstance(item, (int, float))
            and math.isfinite(float(item)),
            "7D float64 vector is non-finite",
        )
        numbers.append(float(item))
    endian = "<" if sys.byteorder == "little" else ">"
    payload = struct.pack(endian + "7d", *numbers)
    header = fast._canonical({"dtype": endian + "f8", "shape": [7]})
    digest = hashlib.sha256()
    digest.update(b"vlsa-table1-array-v1\0")
    digest.update(header)
    digest.update(b"\0")
    digest.update(payload)
    return payload, digest.hexdigest()


def _validated_sample_rows(
    evidence: Mapping[str, Any],
    *,
    expected_geom_ids: Sequence[int],
    expected_body_ids: Sequence[int],
    expected_hash: str,
    label: str,
    require_declared_count: bool = True,
) -> Tuple[Dict[int, int], int]:
    """Reconstruct a serialized local-frame sample ledger and its identity."""

    rows = evidence.get("samples")
    _require(isinstance(rows, list) and rows, "%s raw samples are absent" % label)
    _require(_canonical_sha256(rows) == expected_hash, "%s sample hash differs" % label)
    geom_ids = {int(value) for value in expected_geom_ids}
    body_ids = {int(value) for value in expected_body_ids}
    counts: Dict[int, int] = {}
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), "%s sample row is invalid" % label)
        point = row.get("point_body_local_m")
        _require(
            row.get("sample_id") == index
            and int(row.get("geom_id", -1)) in geom_ids
            and int(row.get("body_id", -1)) in body_ids
            and isinstance(row.get("geom_name"), str)
            and bool(row.get("geom_name"))
            and isinstance(row.get("body_name"), str)
            and bool(row.get("body_name"))
            and row.get("source") == "collision_geom_surface"
            and isinstance(point, list)
            and len(point) == 3
            and all(math.isfinite(float(value)) for value in point),
            "%s sample provenance differs at row %d" % (label, index),
        )
        geom_id = int(row["geom_id"])
        counts[geom_id] = counts.get(geom_id, 0) + 1
    if require_declared_count:
        _require(
            int(evidence.get("sample_count", -1)) == len(rows),
            "%s sample count differs" % label,
        )
    return counts, len(rows)


def _validate_field_sample_static_evidence(
    *,
    apparatus: Mapping[str, Any],
    treatment: Mapping[str, Any],
    metrics: Mapping[str, Any],
    runtime_protocol: Mapping[str, Any],
    runtime_protocol_sha256: str,
    runtime_parameter_block_sha256: str,
) -> Dict[str, Any]:
    """Reconstruct field identities and admissibility from serialized rows."""

    resolved = apparatus.get("resolved_geometry")
    _require(isinstance(resolved, Mapping), "resolved geometry is absent")
    robot_geom_ids = [int(value) for value in resolved.get("robot_geom_ids", ())]
    robot_body_ids = [int(value) for value in resolved.get("robot_body_ids", ())]
    link_geom_ids = [int(value) for value in resolved.get("link56_geom_ids", ())]
    link_body_ids = [int(value) for value in resolved.get("link56_body_ids", ())]
    obstacle_geom_ids = [int(value) for value in resolved.get("obstacle_geom_ids", ())]
    _require(
        robot_geom_ids and robot_body_ids and link_geom_ids and link_body_ids
        and obstacle_geom_ids,
        "resolved field/sample identities are empty",
    )

    full = apparatus.get("full_robot_sampling")
    _require(isinstance(full, Mapping), "full-robot sampling evidence is absent")
    _require(
        apparatus.get("full_robot_sampling_sha256") == _canonical_sha256(full),
        "full-robot sampling block hash differs",
    )
    full_counts, full_count = _validated_sample_rows(
        full,
        expected_geom_ids=robot_geom_ids,
        expected_body_ids=robot_body_ids,
        expected_hash=str(full.get("sample_ledger_sha256", "")),
        label="full-robot",
    )
    from main.poisson_fullbody.surface_sampling import validate_robot_sample_evidence

    validate_robot_sample_evidence(
        full,
        resolved_geom_ids=robot_geom_ids,
        resolved_geom_names=[str(value) for value in resolved.get("robot_geom_names", ())],
        resolved_body_ids=robot_body_ids,
        roundtrip_field="roundtrip",
    )
    records = full.get("geom_records")
    _require(isinstance(records, list), "full-robot component ledger is absent")
    _require(
        {int(row.get("geom_id", -1)): int(row.get("sample_count", -1)) for row in records}
        == full_counts,
        "full-robot component counts differ from raw samples",
    )

    protected = apparatus.get("protected_sampling")
    hashes = apparatus.get("field_bundle_hashes")
    diagnostics = apparatus.get("field_diagnostics")
    boxes = apparatus.get("field_obstacle_boxes")
    _require(isinstance(protected, Mapping), "protected sampling evidence is absent")
    _require(isinstance(hashes, Mapping), "field bundle hashes are absent")
    _require(isinstance(diagnostics, Mapping), "field diagnostics are absent")
    _require(isinstance(boxes, list) and boxes, "field obstacle boxes are absent")
    for name, value in hashes.items():
        _require(
            isinstance(name, str)
            and isinstance(value, str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", value)),
            "field component hash is malformed",
        )
    _require(
        hashes.get("protocol_sha256") == runtime_protocol_sha256
        and hashes.get("parameter_block_sha256") == runtime_parameter_block_sha256,
        "field bundle/runtime identity differs",
    )
    _require(
        _canonical_sha256(boxes) == hashes.get("obstacle_geometry_sha256"),
        "field obstacle geometry hash differs",
    )
    protected_counts, protected_count = _validated_sample_rows(
        protected,
        expected_geom_ids=link_geom_ids,
        expected_body_ids=link_body_ids,
        expected_hash=_canonical_sha256(protected.get("samples")),
        label="protected",
        require_declared_count=False,
    )
    components = protected.get("components")
    _require(isinstance(components, list), "protected component ledger is absent")
    _require(
        {int(row.get("geom_id", -1)): int(row.get("sample_count", -1)) for row in components}
        == protected_counts,
        "protected component counts differ from raw samples",
    )
    epsilon = float(protected.get("epsilon_m"))
    maximum_cover = float(protected.get("maximum_surface_cover_radius_m"))
    _require(
        _canonical_sha256(protected) == hashes.get("protected_samples_sha256"),
        "protected sample payload hash differs",
    )
    _require(
        math.isfinite(epsilon)
        and epsilon > 0.0
        and math.isfinite(maximum_cover)
        and 0.0 <= maximum_cover < epsilon
        and all(
            0.0 <= float(row.get("certified_surface_cover_radius_m")) < epsilon
            for row in components
        ),
        "protected surface cover certificate differs",
    )

    expected_bundle_identity = {
        "protocol_sha256": hashes["protocol_sha256"],
        "parameter_block_sha256": hashes["parameter_block_sha256"],
        "obstacle_geometry_sha256": hashes["obstacle_geometry_sha256"],
        "occupancy_sha256": hashes["occupancy_sha256"],
        "domain_sha256": hashes["domain_sha256"],
        "system_sha256": hashes["system_sha256"],
        "field_sha256": hashes["field_sha256"],
        "protected_samples_sha256": hashes["protected_samples_sha256"],
        "diagnostics": diagnostics,
    }
    _require(
        _canonical_sha256(expected_bundle_identity) == hashes.get("bundle_sha256"),
        "field bundle aggregate hash differs",
    )
    field_certificate = {
        "field_bundle_hashes": dict(hashes),
        "field_diagnostics": dict(diagnostics),
        "field_obstacle_boxes": list(boxes),
        "protected_sampling": dict(protected),
    }
    _require(
        apparatus.get("serialized_field_certificate_sha256")
        == _canonical_sha256(field_certificate),
        "serialized field certificate hash differs",
    )

    poisson = diagnostics.get("poisson")
    _require(isinstance(poisson, Mapping), "Poisson diagnostics are absent")
    tolerance = float(runtime_protocol["poisson"]["normalized_backward_error_tolerance"])
    _require(
        diagnostics.get("obstacle_geom_count") == len(boxes) == len(obstacle_geom_ids)
        and diagnostics.get("protected_surface_component_count") == len(components)
        and diagnostics.get("protected_sample_count") == protected_count
        and int(diagnostics.get("active_vertex_count", -1))
        == int(diagnostics.get("boundary_vertex_count", -2))
        + int(diagnostics.get("interior_vertex_count", -2))
        and int(poisson.get("unknown_count", -1))
        == int(diagnostics.get("interior_vertex_count", -2))
        and diagnostics.get("poisson_method") == "red_black_sor"
        and 0 <= int(diagnostics.get("poisson_iterations", -1))
        <= int(runtime_protocol["poisson"]["max_iterations"])
        and poisson.get("finite") is True
        and poisson.get("passed") is True
        and poisson.get("expected_sign") == "nonnegative"
        and int(poisson.get("sign_violation_count", -1)) == 0
        and float(poisson.get("backward_error_linf")) <= tolerance
        and float(poisson.get("boundary_linf")) <= 1e-12
        and float(poisson.get("interior_min")) > 0.0
        and float(diagnostics.get("minimum_initial_h_m2")) > 0.0
        and math.isclose(
            float(diagnostics.get("required_outer_boundary_clearance_m")),
            float(runtime_protocol["occupancy"]["outer_boundary_clearance_m"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and float(diagnostics.get("minimum_outer_boundary_clearance_m"))
        >= float(diagnostics.get("required_outer_boundary_clearance_m")),
        "field diagnostics fail independent consistency checks",
    )

    static_rows = treatment.get("static_obstacle_trace")
    physics_rows = treatment.get("physics_trace")
    _require(isinstance(static_rows, list) and static_rows, "static trace is absent")
    _require(isinstance(physics_rows, list), "physics trace is absent")
    _require(
        treatment.get("static_obstacle_trace_sha256") == _canonical_sha256(static_rows),
        "static trace hash differs",
    )
    static_fields = (
        "translation_drift_m",
        "rotation_drift_rad",
        "surface_drift_m",
        "maximum_body_linear_speed_m_s",
        "maximum_body_angular_speed_rad_s",
    )
    for index, row in enumerate(static_rows):
        _require(
            isinstance(row, Mapping)
            and set(row) == set(static_fields)
            and all(math.isfinite(float(row[field])) and float(row[field]) >= 0.0 for field in static_fields),
            "static trace row %d is invalid" % index,
        )
    _require(
        len(static_rows) == len(physics_rows) + 1
        and fast._canonical(static_rows[0]) == fast._canonical(apparatus.get("settled_obstacle"))
        and all(
            fast._canonical(static_rows[index + 1])
            == fast._canonical(row.get("selected_obstacle"))
            for index, row in enumerate(physics_rows)
        ),
        "static trace is not bound to every physics row",
    )
    limits = runtime_protocol["admissibility"]
    admissible = bool(
        max(float(row["translation_drift_m"]) for row in static_rows)
        <= float(limits["max_selected_geom_translation_drift_m"])
        and max(float(row["rotation_drift_rad"]) for row in static_rows)
        <= float(limits["max_selected_geom_rotation_drift_rad"])
        and max(float(row["surface_drift_m"]) for row in static_rows)
        <= float(limits["max_selected_geom_surface_drift_m"])
        and max(float(row["maximum_body_linear_speed_m_s"]) for row in static_rows)
        <= float(limits["max_selected_body_linear_speed_m_s"])
        and max(float(row["maximum_body_angular_speed_rad_s"]) for row in static_rows)
        <= float(limits["max_selected_body_angular_speed_rad_s"])
    )
    _require(
        metrics.get("static_field_admissible") is admissible,
        "static admissibility flag differs from raw trace",
    )
    return {
        "full_robot_sample_count": full_count,
        "protected_sample_count": protected_count,
        "static_row_count": len(static_rows),
        "static_admissible": admissible,
        "bundle_sha256": hashes["bundle_sha256"],
    }


def _validate_solved_qp_certificate(
    row: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    controller: Mapping[str, Any],
    torque_actuators: Sequence[Mapping[str, Any]],
    np: Any,
) -> Dict[str, float]:
    """Independently reconstruct the solved torque QP and its KKT optimum."""

    try:
        from scipy.optimize import nnls
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise OscCanaryValidationError(
            "SciPy is required for independent QP optimality validation"
        ) from error

    sensitivity = row.get("sensitivity")
    diagnostics = row.get("shield_diagnostics")
    _require(isinstance(sensitivity, Mapping), "solved row sensitivity is absent")
    _require(isinstance(diagnostics, Mapping), "solved row diagnostics are absent")
    full = np.asarray(sensitivity.get("full_epsilon_sensitivity"), dtype=np.float64)
    half = np.asarray(sensitivity.get("half_epsilon_sensitivity"), dtype=np.float64)
    selected = np.asarray(
        sensitivity.get("torque_to_next_arm_qvel_sensitivity"), dtype=np.float64
    )
    epsilon = np.asarray(sensitivity.get("torque_epsilon_nm"), dtype=np.float64)
    _require(
        full.shape == half.shape == selected.shape == (7, 7)
        and epsilon.shape == (7,)
        and np.all(np.isfinite(full))
        and np.all(np.isfinite(half))
        and np.all(np.isfinite(selected))
        and np.all(np.isfinite(epsilon))
        and np.array_equal(selected, half)
        and np.all(epsilon == float(protocol["shield"]["torque_epsilon_nm"])),
        "two-resolution sensitivity shape or selected estimate differs",
    )
    atol = float(sensitivity.get("agreement_atol"))
    rtol = float(sensitivity.get("agreement_rtol"))
    _require(
        atol == float(protocol["shield"]["sensitivity_agreement_atol"])
        and rtol == float(protocol["shield"]["sensitivity_agreement_rtol"]),
        "sensitivity agreement tolerances differ",
    )
    difference = np.abs(full - half)
    agreement_scale = atol + rtol * np.maximum(np.abs(full), np.abs(half))
    max_absolute = float(np.max(difference))
    max_scaled = float(np.max(difference / agreement_scale))
    _require(
        np.all(difference <= agreement_scale)
        and math.isclose(
            float(sensitivity.get("maximum_epsilon_agreement_absolute_error")),
            max_absolute,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and math.isclose(
            float(sensitivity.get("maximum_epsilon_agreement_scaled_error")),
            max_scaled,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "two-resolution sensitivity agreement differs",
    )

    snapshot = sensitivity.get("snapshot")
    _require(isinstance(snapshot, Mapping), "sensitivity snapshot is absent")
    lower_torque = np.asarray(snapshot.get("arm_torque_lower_nm"), dtype=np.float64)
    upper_torque = np.asarray(snapshot.get("arm_torque_upper_nm"), dtype=np.float64)
    nominal_torque = np.asarray(row.get("nominal_torque_nm"), dtype=np.float64)
    command_torque = np.asarray(row.get("command_torque_nm"), dtype=np.float64)
    delta = command_torque - nominal_torque
    nominal_next = np.asarray(
        row.get("nominal_predicted_next_qvel_rad_s"), dtype=np.float64
    )
    range_by_actuator = {
        int(actuator["actuator_id"]): actuator["control_range"]
        for actuator in torque_actuators
    }
    expected_control_ranges = np.asarray(
        [
            range_by_actuator[int(actuator_id)]
            for actuator_id in controller.get("arm_actuator_indexes", ())
        ],
        dtype=np.float64,
    )
    _require(
        lower_torque.shape == upper_torque.shape == nominal_torque.shape
        == command_torque.shape == nominal_next.shape == (7,)
        and np.all(np.isfinite(lower_torque))
        and np.all(np.isfinite(upper_torque))
        and expected_control_ranges.shape == (7, 2)
        and np.array_equal(lower_torque, expected_control_ranges[:, 0])
        and np.array_equal(upper_torque, expected_control_ranges[:, 1])
        and np.array_equal(
            np.asarray(snapshot.get("nominal_arm_torque_nm"), dtype=np.float64),
            nominal_torque,
        )
        and np.array_equal(
            np.asarray(sensitivity.get("nominal_next_arm_qvel_rad_s"), dtype=np.float64),
            nominal_next,
        )
        and tuple(snapshot.get("arm_actuator_ids", ()))
        == tuple(controller.get("arm_actuator_indexes", ()))
        and tuple(snapshot.get("arm_qpos_indices", ()))
        == tuple(controller.get("arm_qpos_indexes", ()))
        and tuple(snapshot.get("arm_qvel_indices", ()))
        == tuple(controller.get("arm_qvel_indexes", ()))
        and float(snapshot.get("timestep_seconds")) == 0.002
        and sensitivity.get("non_arm_ctrl_preserved") is True
        and math.isfinite(float(sensitivity.get("nominal_next_time_seconds")))
        and bool(re.fullmatch(r"[0-9a-f]{64}", str(snapshot.get("integration_state_sha256", ""))))
        and bool(re.fullmatch(r"[0-9a-f]{64}", str(snapshot.get("all_ctrl_sha256", ""))))
        and bool(
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(sensitivity.get("nominal_next_integration_state_sha256", "")),
            )
        ),
        "sensitivity snapshot, perturbation bounds, or controller indexes differ",
    )
    stencil_rows = sensitivity.get("finite_difference_column_stencils")
    _require(
        isinstance(stencil_rows, list) and len(stencil_rows) == 7,
        "finite-difference stencil ledger is absent",
    )
    for column, plan in enumerate(stencil_rows):
        _require(isinstance(plan, Mapping), "finite-difference plan is invalid")
        base = float(nominal_torque[column])
        low = float(lower_torque[column])
        high = float(upper_torque[column])
        requested = float(epsilon[column])
        negative_room = base - low
        positive_room = high - base
        expected_stencil = (
            "centered"
            if requested <= negative_room and requested <= positive_room
            else ("forward" if min(requested, positive_room) >= min(requested, negative_room) else "backward")
        )
        _require(
            int(plan.get("column_index", -1)) == column
            and float(plan.get("nominal_torque_nm")) == base
            and float(plan.get("lower_bound_nm")) == low
            and float(plan.get("upper_bound_nm")) == high
            and float(plan.get("requested_full_epsilon_nm")) == requested
            and math.isclose(float(plan.get("available_negative_delta_nm")), negative_room, rel_tol=0.0, abs_tol=1e-15)
            and math.isclose(float(plan.get("available_positive_delta_nm")), positive_room, rel_tol=0.0, abs_tol=1e-15)
            and plan.get("difference_formula")
            == "(v_next(delta_1)-v_next(delta_0))/(delta_1-delta_0)",
            "finite-difference plan authority differs at column %d" % column,
        )
        for resolution_name, scale in (("full_resolution", 1.0), ("half_resolution", 0.5)):
            resolution = plan.get(resolution_name)
            _require(isinstance(resolution, Mapping), "stencil resolution is absent")
            deltas = resolution.get("sample_deltas_nm")
            _require(
                resolution.get("stencil") == expected_stencil
                and isinstance(deltas, (list, tuple))
                and len(deltas) == 2,
                "stencil type or samples differ at column %d" % column,
            )
            left, right = (float(deltas[0]), float(deltas[1]))
            denominator = right - left
            _require(
                math.isfinite(left)
                and math.isfinite(right)
                and denominator > 0.0
                and math.isclose(
                    float(resolution.get("denominator_nm")),
                    denominator,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                and low <= base + left <= high
                and low <= base + right <= high
                and max(abs(left), abs(right)) <= requested * scale + 1e-15,
                "stencil perturbation violates bounds at column %d" % column,
            )
            if expected_stencil == "centered":
                _require(left < 0.0 < right, "centered stencil is not two-sided")
            elif expected_stencil == "forward":
                _require(left == 0.0 < right, "forward stencil differs")
            else:
                _require(left < 0.0 == right, "backward stencil differs")
        full_deltas = plan["full_resolution"]["sample_deltas_nm"]
        expected_bound_adapted = bool(
            expected_stencil != "centered"
            or max(abs(float(value)) for value in full_deltas) < requested
        )
        _require(
            plan.get("bound_adapted") is expected_bound_adapted,
            "bound-adapted stencil flag differs at column %d" % column,
        )
    for field, observed in (
        ("nominal_torque_nm", nominal_torque),
        ("torque_lower_nm", lower_torque),
        ("torque_upper_nm", upper_torque),
        ("nominal_next_qvel_rad_s", nominal_next),
    ):
        _require(
            np.array_equal(np.asarray(diagnostics.get(field), dtype=np.float64), observed),
            "shield diagnostic %s differs" % field,
        )
    _require(
        diagnostics.get("schema") == "vlsa_poisson_post_osc_torque_shield.v1"
        and diagnostics.get("constraint_equation")
        == "a@(v_nom_next+S@delta_tau)+alpha*h>=margin"
        and diagnostics.get("sensitivity_already_includes_dt_and_contact_effects") is True
        and float(diagnostics.get("dt_seconds")) == 0.002
        and float(diagnostics.get("alpha")) == float(protocol["shield"]["alpha_gain_per_s"])
        and float(diagnostics.get("margin")) == float(protocol["shield"]["margin_m2_per_s"])
        and math.isclose(
            float(diagnostics.get("sensitivity_max_absolute_error")),
            max_absolute,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "shield diagnostics are not bound to registered inputs",
    )

    h = np.asarray(row.get("poisson_h_m2"), dtype=np.float64)
    gradient_rows = np.asarray(
        row.get("joint_gradient_rows_m2_per_rad"), dtype=np.float64
    )
    _require(
        h.ndim == 1
        and h.size > 0
        and gradient_rows.shape == (h.size, 7),
        "solved CBF rows are invalid",
    )
    gain_rows = gradient_rows @ selected
    required_gain = (
        float(protocol["shield"]["margin_m2_per_s"])
        - float(protocol["shield"]["alpha_gain_per_s"]) * h
        - gradient_rows @ nominal_next
    )
    row_scales = np.max(np.abs(gain_rows), axis=1)
    uncertainty = np.sum(np.abs(gradient_rows), axis=1) * max_absolute
    controllable = row_scales - uncertainty > 0.0
    _require(
        not np.any((~controllable) & (required_gain > 0.0)),
        "serialized solved QP contains an uncontrollable violated row",
    )
    normalized_rows = gain_rows[controllable] / row_scales[controllable, None]
    normalized_lower = required_gain[controllable] / row_scales[controllable]
    delta_lower = lower_torque - nominal_torque
    delta_upper = upper_torque - nominal_torque
    feasibility_tolerance = max(
        5e-7,
        20.0 * float(protocol["shield"]["solver_eps_abs"]),
        20.0 * float(protocol["shield"]["solver_eps_rel"]),
    )
    normalized_slack = normalized_rows @ delta - normalized_lower
    _require(
        np.all(normalized_slack >= -feasibility_tolerance)
        and np.all(delta >= delta_lower - feasibility_tolerance)
        and np.all(delta <= delta_upper + feasibility_tolerance),
        "independently reconstructed solved QP is infeasible",
    )

    # For the strictly convex minimum-norm QP, primal feasibility plus a
    # nonnegative KKT multiplier certificate is sufficient and necessary for
    # the unique global optimum.  Reconstruct the certificate without OSQP.
    active_tolerance = max(2e-6, 50.0 * feasibility_tolerance)
    active_normals: List[Any] = [
        normalized_rows[index]
        for index, slack in enumerate(normalized_slack)
        if float(slack) <= active_tolerance
    ]
    identity = np.eye(7, dtype=np.float64)
    active_normals.extend(
        identity[index]
        for index in range(7)
        if float(delta[index] - delta_lower[index]) <= active_tolerance
    )
    active_normals.extend(
        -identity[index]
        for index in range(7)
        if float(delta_upper[index] - delta[index]) <= active_tolerance
    )
    _require(active_normals, "nonzero solved correction has no active QP constraint")
    active = np.asarray(active_normals, dtype=np.float64)
    multipliers, kkt_residual = nnls(active.T, delta)
    kkt_tolerance = max(2e-5, 200.0 * feasibility_tolerance) * max(
        1.0, float(np.linalg.norm(delta))
    )
    _require(
        np.all(np.isfinite(multipliers))
        and math.isfinite(float(kkt_residual))
        and float(kkt_residual) <= kkt_tolerance,
        "candidate is feasible but lacks an independent minimum-norm KKT certificate",
    )
    return {
        "maximum_sensitivity_absolute_error": max_absolute,
        "maximum_sensitivity_scaled_error": max_scaled,
        "minimum_normalized_qp_slack": (
            float(np.min(normalized_slack)) if normalized_slack.size else math.inf
        ),
        "kkt_stationarity_residual": float(kkt_residual),
        "correction_l2_nm": float(np.linalg.norm(delta)),
    }


def _manifest_case(path: Path, case_id: str) -> tuple[Dict[str, Any], str]:
    matches = []
    with path.open("rb") as stream:
        for raw in stream:
            if raw.strip():
                value = json.loads(raw)
                if value.get("case_id") == case_id:
                    matches.append(
                        (value, hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest())
                    )
    _require(len(matches) == 1, "contact manifest case is absent or duplicated")
    return matches[0]


def _bound_file(root: Path, relative: str, expected_sha256: str) -> Path:
    _require(
        isinstance(relative, str)
        and bool(relative)
        and not Path(relative).is_absolute()
        and all(part not in ("", ".", "..") for part in Path(relative).parts),
        "bound relative path is invalid",
    )
    current = root
    _require(current.is_dir() and not current.is_symlink(), "bound root is invalid")
    for part in Path(relative).parts:
        current = current / part
        _require(not current.is_symlink(), "bound path traverses a symlink")
    _require(current.is_file(), "bound file is absent")
    _require(fast._file_sha256(current) == expected_sha256, "bound file hash differs")
    return current


def _motion(rows: Sequence[Mapping[str, Any]], first: Any) -> Dict[str, float]:
    import numpy as np

    if first is None:
        return {"joint": 0.0, "eef": 0.0, "zero": 1.0}
    selected = [row for row in rows if int(row["physical_boundary"]) >= int(first)]
    if not selected:
        return {"joint": 0.0, "eef": 0.0, "zero": 1.0}
    joint = sum(float(row["measured_qvel_l2_rad_s"]) * 0.002 for row in selected)
    positions = [np.asarray(row["eef_position_world_m"], dtype=np.float64) for row in selected]
    eef = sum(float(np.linalg.norm(right - left)) for left, right in zip(positions, positions[1:]))
    zero = sum(float(row["torque_correction_l2_nm"]) < 1e-12 for row in selected) / len(selected)
    return {"joint": float(joint), "eef": float(eef), "zero": float(zero)}


def _close(left: Any, right: Any, label: str, tolerance: float = 1e-12) -> None:
    _require(
        math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance),
        "%s differs" % label,
    )


def _exact_float(left: Any, right: Any, label: str) -> None:
    _require(
        not isinstance(left, bool)
        and isinstance(left, (int, float))
        and not isinstance(right, bool)
        and isinstance(right, (int, float))
        and math.isfinite(float(left))
        and math.isfinite(float(right))
        and struct.pack(">d", float(left)) == struct.pack(">d", float(right)),
        "%s differs" % label,
    )


def _strict_integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        "%s must be an integer" % label,
    )
    return int(value)


def _validate_paper_car_cadence_and_historical_binding(
    car_ledger: Sequence[Mapping[str, Any]],
    *,
    action_count: int,
    historical_settled_position_sha256: str,
) -> None:
    """Require exact settled/20 Hz CAR rows and immutable settled pairing."""

    _require(
        isinstance(action_count, int)
        and not isinstance(action_count, bool)
        and action_count >= 0
        and isinstance(car_ledger, Sequence)
        and not isinstance(car_ledger, (str, bytes))
        and len(car_ledger) == action_count + 1,
        "paper CAR endpoint count differs",
    )
    _require(bool(car_ledger), "paper CAR settled row is absent")
    settled = car_ledger[0]
    _require(isinstance(settled, Mapping), "paper CAR settled row is invalid")
    settled_index = settled.get("source_action_index")
    _require(
        isinstance(settled_index, int)
        and not isinstance(settled_index, bool)
        and settled_index == -1
        and settled.get("snapshot_kind") == "settled_pre_action"
        and settled.get("active_obstacle_position_observation_array_sha256")
        == historical_settled_position_sha256,
        "paper CAR settled row or historical binding differs",
    )
    _exact_float(
        settled.get("l1_displacement_from_settled_m"),
        0.0,
        "paper CAR settled displacement",
    )
    for expected_index, row in enumerate(car_ledger[1:]):
        _require(isinstance(row, Mapping), "paper CAR endpoint row is invalid")
        observed_index = row.get("source_action_index")
        _require(
            isinstance(observed_index, int)
            and not isinstance(observed_index, bool)
            and observed_index == expected_index
            and row.get("snapshot_kind") == "completed_high_level_endpoint",
            "paper CAR endpoint cadence differs at row %d" % (expected_index + 1),
        )


def _require_resolved_body_name(
    resolved: Mapping[str, Any],
    *,
    ids_field: str,
    names_field: str,
    expected_body_id: int,
    expected_body_name: str,
    label: str,
) -> Dict[int, str]:
    """Validate one parallel resolved ID/name ledger and its selected row."""

    ids = resolved.get(ids_field)
    names = resolved.get(names_field)
    _require(
        isinstance(ids, Sequence)
        and not isinstance(ids, (str, bytes))
        and isinstance(names, Sequence)
        and not isinstance(names, (str, bytes))
        and len(ids) > 0
        and len(ids) == len(names),
        "%s resolved body ID/name ledger differs" % label,
    )
    parsed_ids = []
    parsed_names = []
    for body_id, body_name in zip(ids, names):
        _require(
            not isinstance(body_id, bool)
            and isinstance(body_id, int)
            and isinstance(body_name, str)
            and bool(body_name),
            "%s resolved body ID/name row is malformed" % label,
        )
        parsed_ids.append(int(body_id))
        parsed_names.append(str(body_name))
    _require(
        len(set(parsed_ids)) == len(parsed_ids),
        "%s resolved body IDs are duplicated" % label,
    )
    name_by_id = dict(zip(parsed_ids, parsed_names))
    _require(
        name_by_id.get(int(expected_body_id)) == str(expected_body_name),
        "%s selected root body name differs" % label,
    )
    return name_by_id


def _validate_paper_car_endpoint_row(
    row: Mapping[str, Any],
    *,
    index: int,
    car_key: str,
    obstacle_root_body_id: int,
    settled_car_position: Any,
    np: Any,
) -> float:
    """Independently reconstruct one exact paper-CAR observation endpoint."""

    observed_position = np.asarray(
        row.get("active_obstacle_position_observation_world_m"), dtype=np.float64
    )
    live_body_position = np.asarray(
        row.get("active_obstacle_root_position_world_m"), dtype=np.float64
    )
    forwarded_body_position = np.asarray(
        row.get(
            "active_obstacle_root_position_post_integration_forwarded_world_m"
        ),
        dtype=np.float64,
    )
    forwarded_component_delta = np.asarray(
        row.get("observation_post_integration_forwarded_component_delta_m"),
        dtype=np.float64,
    )
    settled = np.asarray(settled_car_position, dtype=np.float64)
    observed_hash = _float64_sha256(observed_position, np)
    live_hash = _float64_sha256(live_body_position, np)
    forwarded_hash = _float64_sha256(forwarded_body_position, np)
    reconstructed_forwarded_delta = forwarded_body_position - observed_position
    forwarded_delta_hash = _float64_sha256(forwarded_component_delta, np)
    reconstructed_forwarded_delta_hash = _float64_sha256(
        reconstructed_forwarded_delta, np
    )
    _require(
        observed_position.shape == (3,)
        and live_body_position.shape == (3,)
        and forwarded_body_position.shape == (3,)
        and forwarded_component_delta.shape == (3,)
        and settled.shape == (3,)
        and np.all(np.isfinite(observed_position))
        and np.all(np.isfinite(live_body_position))
        and np.all(np.isfinite(forwarded_body_position))
        and np.all(np.isfinite(forwarded_component_delta))
        and np.all(np.isfinite(settled))
        and np.array_equal(observed_position, live_body_position)
        and observed_hash == live_hash
        and np.array_equal(forwarded_component_delta, reconstructed_forwarded_delta)
        and forwarded_delta_hash == reconstructed_forwarded_delta_hash
        and row.get("observation_body_xpos_bitwise_equal") is True
        and row.get("paper_car_observation_key") == car_key,
        "paper CAR same-phase observation/body authority differs at row %d" % index,
    )
    _require(
        row.get("paper_car_position_source")
        == "selected_obstacle_pos_native_observation"
        and _strict_integer(
            row.get("active_obstacle_root_body_id"),
            "paper CAR row root body ID",
        )
        == int(obstacle_root_body_id)
        and row.get("active_obstacle_root_position_phase")
        == "live_solver_phase_preintegration_geometry"
        and row.get("post_integration_forwarded_pose_role")
        == "phase_diagnostic_only_not_paper_car_metric"
        and row.get("active_obstacle_position_observation_array_sha256")
        == observed_hash
        and row.get("active_obstacle_root_position_array_sha256") == live_hash
        and row.get(
            "active_obstacle_root_position_post_integration_array_sha256"
        )
        == forwarded_hash,
        "paper CAR phase or array identity differs at row %d" % index,
    )
    _require(
        row.get(
            "observation_post_integration_forwarded_component_delta_array_sha256"
        )
        == forwarded_delta_hash,
        "paper CAR forwarded component-delta identity differs at row %d" % index,
    )
    _exact_float(
        row.get("observation_live_solver_phase_l1_delta_m"),
        float(np.sum(np.abs(observed_position - live_body_position))),
        "paper CAR live-phase delta row %d" % index,
    )
    _exact_float(
        row.get("observation_post_integration_forwarded_l1_delta_m"),
        float(np.sum(np.abs(reconstructed_forwarded_delta))),
        "paper CAR forwarded L1 diagnostic row %d" % index,
    )
    _exact_float(
        row.get("observation_post_integration_forwarded_linf_delta_m"),
        float(np.max(np.abs(reconstructed_forwarded_delta))),
        "paper CAR forwarded Linf diagnostic row %d" % index,
    )
    displacement = float(np.sum(np.abs(observed_position - settled)))
    _exact_float(
        row.get("l1_displacement_from_settled_m"),
        displacement,
        "paper CAR row %d" % index,
    )
    return displacement


def _validate_compact_constraint_trace(
    row: Mapping[str, Any], *, expected_sample_count: int, np: Any
) -> Mapping[str, Mapping[str, Any]]:
    """Validate the bounded per-substep ledger without inventing raw arrays."""

    trace = row.get("constraint_trace")
    _require(isinstance(trace, Mapping), "compact constraint trace is absent")
    _require(
        trace.get("schema_version")
        == "vlsa_poisson_compact_constraint_trace.v1"
        and trace.get("array_hash_format")
        == "sha256_vlsa-table1-array-v1_header_and_c_order_float64_bytes"
        and int(trace.get("sample_count", -1)) == int(expected_sample_count),
        "compact constraint trace identity differs",
    )
    arrays = trace.get("arrays")
    expected_shapes = {
        "poisson_h_m2": [int(expected_sample_count)],
        "joint_gradient_rows_m2_per_rad": [int(expected_sample_count), 7],
        "actual_hdot_m2_per_s": [int(expected_sample_count)],
        "candidate_exact_clone_hdot_m2_per_s": [int(expected_sample_count)],
        "nominal_exact_clone_hdot_m2_per_s": [int(expected_sample_count)],
    }
    _require(
        isinstance(arrays, Mapping) and set(arrays) == set(expected_shapes),
        "compact constraint array ledger differs",
    )
    expected_dtype = np.dtype(np.float64).str
    for name, shape in expected_shapes.items():
        evidence = arrays.get(name)
        _require(
            isinstance(evidence, Mapping)
            and set(evidence) == {"dtype", "shape", "sha256", "minimum"}
            and evidence.get("dtype") == expected_dtype
            and evidence.get("shape") == shape
            and bool(re.fullmatch(r"[0-9a-f]{64}", str(evidence.get("sha256", ""))))
            and not isinstance(evidence.get("minimum"), bool)
            and math.isfinite(float(evidence.get("minimum"))),
            "compact %s evidence is malformed" % name,
        )
    _require(
        float(arrays["poisson_h_m2"]["minimum"]) > 0.0,
        "compact Poisson minimum is nonpositive",
    )
    return arrays


def _validate_compact_shield_diagnostics(
    row: Mapping[str, Any], *, expected_sample_count: int
) -> Mapping[str, Any]:
    evidence = row.get("compact_shield_diagnostics")
    status = row.get("shield_status")
    _require(
        isinstance(evidence, Mapping)
        and evidence.get("schema_version")
        == "vlsa_poisson_compact_shield_diagnostics.v1"
        and evidence.get("shield_status") == status
        and int(evidence.get("sample_count", -1)) == int(expected_sample_count)
        and bool(
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(evidence.get("full_diagnostics_sha256", "")),
            )
        ),
        "compact shield diagnostics identity differs",
    )
    scalars = evidence.get("scalars")
    _require(isinstance(scalars, Mapping), "compact shield scalar ledger is absent")
    for name, value in scalars.items():
        _require(
            isinstance(name, str)
            and (
                value is None
                or (
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and math.isfinite(float(value))
                )
            ),
            "compact shield scalar %s is invalid" % name,
        )
    if status == "nominal_safe_exact_clone":
        _require(
            evidence.get("solver_attempted") is False
            and evidence.get("solver") is None
            and evidence.get("solver_status") is None
            and evidence.get("sensitivity_certificate_sha256") is None,
            "compact nominal-pass diagnostics differ",
        )
    else:
        _require(
            status == "solved"
            and evidence.get("solver_attempted") is True
            and evidence.get("solver") == "osqp"
            and str(evidence.get("solver_status", "")).lower()
            in ("solved", "solved inaccurate")
            and int(scalars.get("input_constraint_count", -1))
            == int(expected_sample_count)
            and bool(
                re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(evidence.get("sensitivity_certificate_sha256", "")),
                )
            ),
            "compact solved-QP diagnostics differ",
        )
    return evidence


def _independent_constraint_attribution(
    nominal_residuals: Any,
    protected_samples: Sequence[Mapping[str, Any]],
    *,
    np: Any,
) -> Dict[str, Any]:
    """Reconstruct which protected-link samples made the nominal step unsafe."""

    residuals = np.asarray(nominal_residuals, dtype=np.float64)
    _require(
        residuals.ndim == 1
        and residuals.size > 0
        and residuals.size == len(protected_samples)
        and np.all(np.isfinite(residuals)),
        "constraint attribution inputs differ",
    )
    minimum = float(np.min(residuals))
    minimum_index = int(np.argmin(residuals))
    tolerance = 1e-12
    near_indices = np.flatnonzero(np.abs(residuals - minimum) <= tolerance)
    negative_indices = np.flatnonzero(residuals < 0.0)
    return {
        "schema_version": "vlsa_poisson_constraint_attribution.v1",
        "selection_rule": "first_np_argmin_of_nominal_exact_clone_cbf_residual",
        "minimum_nominal_residual_m2_per_s": minimum,
        "minimum_sample_index": minimum_index,
        "minimum_sample": dict(protected_samples[minimum_index]),
        "near_minimum_absolute_tolerance_m2_per_s": tolerance,
        "near_minimum_sample_indices": [int(value) for value in near_indices],
        "near_minimum_body_names": sorted(
            {
                str(protected_samples[int(value)]["body_name"])
                for value in near_indices
            }
        ),
        "negative_nominal_residual_sample_count": int(negative_indices.size),
        "negative_nominal_residual_body_names": sorted(
            {
                str(protected_samples[int(value)]["body_name"])
                for value in negative_indices
            }
        ),
    }


def _validate_first_divergence_full_certificate(
    row: Mapping[str, Any],
    *,
    compact_arrays: Mapping[str, Mapping[str, Any]],
    compact_diagnostics: Mapping[str, Any],
    protocol: Mapping[str, Any],
    controller: Mapping[str, Any],
    torque_actuators: Sequence[Mapping[str, Any]],
    protected_samples: Sequence[Mapping[str, Any]],
    np: Any,
) -> Dict[str, Any]:
    """Reconstruct raw residuals and the QP only for the causal divergence."""

    certificate = row.get("full_constraint_qp_certificate")
    _require(isinstance(certificate, Mapping), "first divergence certificate is absent")
    _require(
        certificate.get("schema_version")
        == "vlsa_poisson_first_divergence_full_qp_certificate.v1"
        and int(certificate.get("physical_boundary", -1))
        == int(row.get("physical_boundary", -2))
        and int(certificate.get("sample_count", -1))
        == int(row["constraint_trace"]["sample_count"])
        and row.get("full_constraint_qp_certificate_sha256")
        == _canonical_sha256(certificate),
        "first divergence certificate identity differs",
    )
    sample_count = int(certificate["sample_count"])
    arrays = {
        name: np.asarray(certificate.get(name), dtype=np.float64)
        for name in (
            "poisson_h_m2",
            "joint_gradient_rows_m2_per_rad",
            "actual_hdot_m2_per_s",
            "candidate_exact_clone_hdot_m2_per_s",
            "nominal_exact_clone_hdot_m2_per_s",
        )
    }
    expected_shapes = {
        "poisson_h_m2": (sample_count,),
        "joint_gradient_rows_m2_per_rad": (sample_count, 7),
        "actual_hdot_m2_per_s": (sample_count,),
        "candidate_exact_clone_hdot_m2_per_s": (sample_count,),
        "nominal_exact_clone_hdot_m2_per_s": (sample_count,),
    }
    for name, array in arrays.items():
        _require(
            array.shape == expected_shapes[name]
            and np.all(np.isfinite(array))
            and compact_arrays[name].get("sha256") == _float64_sha256(array, np),
            "full certificate %s differs from its compact identity" % name,
        )
        _close(
            compact_arrays[name].get("minimum"),
            float(np.min(array)),
            "full certificate %s minimum" % name,
        )
    sensitivity = certificate.get("sensitivity")
    diagnostics = certificate.get("shield_diagnostics")
    _require(
        isinstance(sensitivity, Mapping)
        and isinstance(diagnostics, Mapping)
        and compact_diagnostics.get("sensitivity_certificate_sha256")
        == _canonical_sha256(sensitivity)
        and compact_diagnostics.get("full_diagnostics_sha256")
        == _canonical_sha256(diagnostics),
        "full sensitivity/diagnostic certificate hashes differ",
    )
    h = arrays["poisson_h_m2"]
    gradients = arrays["joint_gradient_rows_m2_per_rad"]
    measured = np.asarray(row.get("measured_qvel_rad_s"), dtype=np.float64)
    candidate_qvel = np.asarray(
        row.get("predicted_next_qvel_rad_s"), dtype=np.float64
    )
    nominal_qvel = np.asarray(
        row.get("nominal_predicted_next_qvel_rad_s"), dtype=np.float64
    )
    _require(
        np.allclose(
            arrays["actual_hdot_m2_per_s"],
            gradients @ measured,
            rtol=0.0,
            atol=1e-12,
        )
        and np.allclose(
            arrays["candidate_exact_clone_hdot_m2_per_s"],
            gradients @ candidate_qvel,
            rtol=0.0,
            atol=1e-12,
        )
        and np.allclose(
            arrays["nominal_exact_clone_hdot_m2_per_s"],
            gradients @ nominal_qvel,
            rtol=0.0,
            atol=1e-12,
        ),
        "full divergence hdot reconstruction differs",
    )
    alpha_h_minus_margin = (
        float(protocol["shield"]["alpha_gain_per_s"]) * h
        - float(protocol["shield"]["margin_m2_per_s"])
    )
    residuals = {
        "minimum_actual_cbf_residual_m2_per_s": (
            arrays["actual_hdot_m2_per_s"] + alpha_h_minus_margin
        ),
        "candidate_exact_clone_minimum_cbf_residual_m2_per_s": (
            arrays["candidate_exact_clone_hdot_m2_per_s"]
            + alpha_h_minus_margin
        ),
        "nominal_exact_clone_minimum_cbf_residual_m2_per_s": (
            arrays["nominal_exact_clone_hdot_m2_per_s"]
            + alpha_h_minus_margin
        ),
    }
    for field, values in residuals.items():
        _close(row.get(field), float(np.min(values)), "full divergence %s" % field)
    independent_attribution = _independent_constraint_attribution(
        residuals["nominal_exact_clone_minimum_cbf_residual_m2_per_s"],
        protected_samples,
        np=np,
    )
    _require(
        fast._canonical(certificate.get("constraint_attribution"))
        == fast._canonical(independent_attribution),
        "first divergence protected-link constraint attribution differs",
    )
    tolerance = float(
        protocol["shield"]["actual_cbf_residual_tolerance_m2_per_s"]
    )
    _require(
        float(np.min(residuals["minimum_actual_cbf_residual_m2_per_s"]))
        >= -tolerance
        and float(
            np.min(
                residuals[
                    "candidate_exact_clone_minimum_cbf_residual_m2_per_s"
                ]
            )
        )
        >= -tolerance,
        "full divergence residual reconstruction failed",
    )
    certificate_row = dict(row)
    certificate_row.update(certificate)
    qp_audit = _validate_solved_qp_certificate(
        certificate_row,
        protocol=protocol,
        controller=controller,
        torque_actuators=torque_actuators,
        np=np,
    )
    return {
        **qp_audit,
        "constraint_attribution": independent_attribution,
    }


def _validate_registered_contact_evidence(
    *,
    scope: Any,
    measurement: Any,
    physics: Sequence[Mapping[str, Any]],
    resolved_geometry: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reconstruct every forbidden category from serialized authoritative IDs."""

    _require(isinstance(scope, Mapping), "registered contact scope is absent")
    scope_without_hash = dict(scope)
    identity_sha256 = scope_without_hash.pop("identity_sha256", None)
    _require(
        scope.get("schema_version") == "vlsa_poisson_registered_contact_scope.v1"
        and identity_sha256 == _canonical_sha256(scope_without_hash),
        "registered contact scope identity differs",
    )
    geom_count = int(scope.get("model_geom_count", -1))

    def id_set(field: str) -> set:
        values = scope.get(field)
        _require(
            isinstance(values, Sequence)
            and not isinstance(values, (str, bytes))
            and list(values) == sorted({int(value) for value in values}),
            "registered contact %s is malformed" % field,
        )
        return {int(value) for value in values}

    robot = id_set("robot_geom_ids")
    robot_owned = id_set("robot_owned_geom_ids")
    selected = id_set("selected_obstacle_geom_ids")
    link56 = id_set("link56_geom_ids")
    external = id_set("external_nonrobot_geom_ids")
    _require(
        geom_count > 0
        and robot
        and selected
        and link56
        and robot_owned | external == set(range(geom_count))
        and not robot_owned & external
        and robot <= robot_owned
        and selected <= external
        and link56 <= robot
        and robot
        == {int(value) for value in resolved_geometry.get("robot_geom_ids", ())}
        and selected
        == {int(value) for value in resolved_geometry.get("obstacle_geom_ids", ())}
        and link56
        == {int(value) for value in resolved_geometry.get("link56_geom_ids", ())},
        "registered contact ID partition differs",
    )
    identities = scope.get("geom_identities")
    _require(
        isinstance(identities, Sequence)
        and len(identities) == geom_count
        and [int(row.get("geom_id", -1)) for row in identities]
        == list(range(geom_count)),
        "registered geom identity ledger differs",
    )
    identity = {int(row["geom_id"]): row for row in identities}
    robot_body_ids = {
        int(value) for value in resolved_geometry.get("robot_body_ids", ())
    }
    _require(
        robot_body_ids
        and robot_owned
        == {
            geom_id
            for geom_id, row in identity.items()
            if int(row.get("body_id", -1)) in robot_body_ids
        },
        "registered robot-owned geom set differs from the robot body tree",
    )
    for geom_id, row in identity.items():
        _require(
            isinstance(row, Mapping)
            and isinstance(row.get("geom_name"), str)
            and isinstance(row.get("body_name"), str)
            and isinstance(row.get("body_id"), int)
            and row.get("is_robot_geom") is (geom_id in robot)
            and row.get("is_robot_owned_geom") is (geom_id in robot_owned)
            and row.get("is_selected_obstacle_geom") is (geom_id in selected)
            and row.get("is_link56_geom") is (geom_id in link56)
            and row.get("is_external_nonrobot_geom") is (geom_id in external),
            "registered geom identity role differs",
        )

    def validate_record(record: Any, *, settled: bool) -> Dict[str, Any]:
        _require(isinstance(record, Mapping), "registered contact row is invalid")
        unhashed = dict(record)
        record_sha256 = unhashed.pop("record_sha256", None)
        geom1 = int(record.get("geom1_id", -1))
        geom2 = int(record.get("geom2_id", -1))
        selected_contact = bool(
            (geom1 in robot and geom2 in selected)
            or (geom2 in robot and geom1 in selected)
        )
        link_external_contact = bool(
            (geom1 in link56 and geom2 in external)
            or (geom2 in link56 and geom1 in external)
        )
        _require(
            record.get("schema_version")
            == "vlsa_poisson_registered_forbidden_contact.v1"
            and geom1 in identity
            and geom2 in identity
            and geom1 != geom2
            and math.isfinite(float(record.get("contact_distance_m")))
            and float(record.get("contact_distance_m")) <= 0.0
            and (selected_contact or link_external_contact)
            and record_sha256 == _canonical_sha256(unhashed),
            "registered forbidden contact row identity differs",
        )
        robot_geom_id = geom1 if geom1 in robot else geom2
        external_geom_id = geom2 if robot_geom_id == geom1 else geom1
        expected_categories = []
        if selected_contact:
            expected_categories.append("any_robot_vs_selected_obstacle")
        if link_external_contact:
            expected_categories.append("link56_vs_external_nonrobot")
        _require(
            record.get("robot_geom_id") == robot_geom_id
            and record.get("external_geom_id") == external_geom_id
            and record.get("robot_geom_name")
            == identity[robot_geom_id]["geom_name"]
            and record.get("robot_body_id")
            == identity[robot_geom_id]["body_id"]
            and record.get("robot_body_name")
            == identity[robot_geom_id]["body_name"]
            and record.get("external_geom_name")
            == identity[external_geom_id]["geom_name"]
            and record.get("external_body_id")
            == identity[external_geom_id]["body_id"]
            and record.get("external_body_name")
            == identity[external_geom_id]["body_name"]
            and record.get("contact_categories") == expected_categories
            and record.get("any_robot_selected_obstacle_contact")
            is selected_contact
            and record.get("link56_external_nonrobot_contact")
            is link_external_contact
            and record.get("link56_nonselected_external_contact")
            is bool(link_external_contact and external_geom_id not in selected),
            "registered forbidden contact categories differ",
        )
        if settled:
            _require(
                record.get("source_phase") == "settled_forwarded"
                and record.get("physical_boundary") is None
                and record.get("source_action_index") is None
                and record.get("physics_substep_index") is None,
                "settled registered contact cadence differs",
            )
        else:
            boundary = int(record.get("physical_boundary", -1))
            _require(
                record.get("source_phase")
                in (
                    "live_solver_phase_preintegration_geometry",
                    "post_integration_recomputed",
                )
                and 0 <= boundary < len(physics)
                and int(record.get("source_action_index", -1)) == boundary // 25
                and int(record.get("physics_substep_index", -1)) == boundary % 25,
                "rollout registered contact cadence differs",
            )
        return dict(record)

    _require(
        isinstance(measurement, Mapping)
        and measurement.get("schema_version")
        == "vlsa_poisson_registered_contact_measurement.v1"
        and measurement.get("scope_identity_sha256") == identity_sha256
        and int(measurement.get("observed_physics_substeps", -1)) == len(physics),
        "registered contact measurement header differs",
    )
    settled_records = measurement.get("settled_contact_records")
    rollout_records = measurement.get("rollout_contact_records")
    _require(
        isinstance(settled_records, Sequence)
        and isinstance(rollout_records, Sequence),
        "registered contact record ledgers are absent",
    )
    settled_validated = [
        validate_record(record, settled=True) for record in settled_records
    ]
    rollout_validated = [
        validate_record(record, settled=False) for record in rollout_records
    ]
    flattened_trace = []
    for boundary, row in enumerate(physics):
        records = row.get("registered_forbidden_contacts")
        _require(
            row.get("registered_contact_scope_checked") is True
            and isinstance(records, Sequence)
            and int(row.get("physical_boundary", -1)) == boundary,
            "physics row lacks registered contact monitoring",
        )
        validated = [validate_record(record, settled=False) for record in records]
        flattened_trace.extend(validated)
        categories = sorted(
            {
                category
                for record in validated
                for category in record["contact_categories"]
            }
        )
        _require(
            row.get("registered_forbidden_contact_seen") is bool(validated)
            and row.get("registered_contact_categories") == categories,
            "physics registered contact flags differ",
        )
        candidate = row.get("candidate_exact_clone_contact")
        _require(
            isinstance(candidate, Mapping)
            and candidate.get("literal_contact") is False
            and candidate.get("literal_contact_count") == 0
            and candidate.get("contacts") == []
            and candidate.get("contact_categories") == [],
            "executed candidate clone was not forbidden-contact free",
        )
    _require(
        _canonical_sha256(flattened_trace) == _canonical_sha256(rollout_validated),
        "registered contact trace and aggregate ledger differ",
    )
    all_records = settled_validated + rollout_validated
    categories = sorted(
        {
            category
            for record in rollout_validated
            for category in record["contact_categories"]
        }
    )
    selected_any = any(
        record["any_robot_selected_obstacle_contact"] for record in all_records
    )
    link_external_any = any(
        record["link56_external_nonrobot_contact"] for record in all_records
    )
    shifted_any = any(
        record["link56_nonselected_external_contact"] for record in all_records
    )
    _require(
        measurement.get("settled_forbidden_contact") is bool(settled_validated)
        and measurement.get("rollout_forbidden_contact")
        is bool(rollout_validated)
        and measurement.get("rollout_contact_categories") == categories
        and measurement.get("any_registered_forbidden_contact")
        is bool(all_records)
        and measurement.get("any_robot_selected_obstacle_contact")
        is selected_any
        and measurement.get("any_link56_external_nonrobot_contact")
        is link_external_any
        and measurement.get("any_link56_nonselected_external_contact")
        is shifted_any,
        "registered contact aggregate flags differ",
    )
    return {
        "settled_contact": bool(settled_validated),
        "rollout_contact": bool(rollout_validated),
        "rollout_contact_categories": categories,
        "rollout_contact_records_sha256": _canonical_sha256(rollout_validated),
        "any_robot_selected_obstacle_contact": selected_any,
        "any_link56_external_nonrobot_contact": link_external_any,
        "any_link56_nonselected_external_contact": shifted_any,
    }


def _validate_partial_action_ledger(
    *,
    terminal_kind: str,
    actions: Sequence[Mapping[str, Any]],
    planner_actions: Sequence[Mapping[str, Any]],
    partial_action_record: Any,
    physics_substep_count: int,
) -> Dict[str, Any]:
    """Validate the sole planner row allowed beyond completed actions."""

    partial_terminal = terminal_kind in (
        "literal_registered_forbidden_contact",
        "safety_method_stop_before_physics",
    )
    if not partial_terminal:
        _require(
            partial_action_record is None and len(planner_actions) == len(actions),
            "complete terminal contains a partial action",
        )
        return {"present": False, "completed_physics_substeps": 0}
    _require(
        isinstance(partial_action_record, Mapping)
        and len(planner_actions) == len(actions) + 1,
        "partial terminal does not contain exactly one extra planner action",
    )
    required_fields = {
        "schema_version",
        "terminal_kind",
        "source_action_index",
        "planner_action_trace_index",
        "executed_high_level_action",
        "executed_high_level_action_sha256",
        "planner_executed_action_sha256",
        "planner_native_observation_sha256",
        "pre_action_observation_sha256",
        "completed_physics_substeps",
        "physics_boundary_start",
        "physics_boundary_end_exclusive",
        "terminal_observation_kind",
        "terminal_observation_sha256",
        "terminal_forbidden_contact_categories",
        "terminal_forbidden_contact_records_sha256",
    }
    _require(
        set(partial_action_record) == required_fields
        and partial_action_record.get("schema_version")
        == "vlsa_poisson_partial_action_record.v1"
        and partial_action_record.get("terminal_kind") == terminal_kind,
        "partial action schema or terminal differs",
    )
    source_index = len(actions)
    planner_row = planner_actions[-1]
    completed = int(partial_action_record.get("completed_physics_substeps", -1))
    expected_completed = int(physics_substep_count) - source_index * 25
    valid_count = (
        1 <= completed <= 25
        if terminal_kind == "literal_registered_forbidden_contact"
        else 0 <= completed < 25
    )
    action_bytes, action_hash = _float64_vector7_bytes_and_sha256(
        partial_action_record.get("executed_high_level_action")
    )
    planner_action_bytes, planner_action_hash = _float64_vector7_bytes_and_sha256(
        planner_row.get("executed")
    )
    for field in (
        "executed_high_level_action_sha256",
        "planner_executed_action_sha256",
        "planner_native_observation_sha256",
        "pre_action_observation_sha256",
        "terminal_observation_sha256",
        "terminal_forbidden_contact_records_sha256",
    ):
        _require(
            bool(
                re.fullmatch(
                    r"[0-9a-f]{64}", str(partial_action_record.get(field, ""))
                )
            ),
            "partial action %s is malformed" % field,
        )
    _require(
        int(partial_action_record.get("source_action_index", -1)) == source_index
        and int(partial_action_record.get("planner_action_trace_index", -1))
        == source_index
        and int(planner_row.get("source_action_index", -1)) == source_index
        and partial_action_record.get("executed_high_level_action_sha256")
        == action_hash
        and partial_action_record.get("planner_executed_action_sha256")
        == action_hash
        and planner_row.get("executed_action_sha256") == planner_action_hash
        and planner_action_hash == action_hash
        and planner_action_bytes == action_bytes
        and partial_action_record.get("planner_native_observation_sha256")
        == planner_row.get("native_observation_sha256")
        == partial_action_record.get("pre_action_observation_sha256")
        and completed == expected_completed
        and valid_count
        and int(partial_action_record.get("physics_boundary_start", -1))
        == source_index * 25
        and int(partial_action_record.get("physics_boundary_end_exclusive", -1))
        == int(physics_substep_count),
        "partial action planner/physics binding differs",
    )
    if actions:
        _require(
            partial_action_record.get("pre_action_observation_sha256")
            == actions[-1].get("observation_sha256"),
            "partial action does not consume the last completed observation",
        )
    expected_kind = (
        "post_contact_observable_refresh"
        if terminal_kind == "literal_registered_forbidden_contact"
        else "post_method_stop_observable_refresh"
    )
    _require(
        partial_action_record.get("terminal_observation_kind") == expected_kind,
        "partial terminal observation kind differs",
    )
    categories = partial_action_record.get(
        "terminal_forbidden_contact_categories"
    )
    allowed_categories = {
        "any_robot_vs_selected_obstacle",
        "link56_vs_external_nonrobot",
    }
    _require(
        isinstance(categories, Sequence)
        and not isinstance(categories, (str, bytes))
        and list(categories) == sorted(set(categories))
        and set(categories) <= allowed_categories
        and (
            bool(categories)
            if terminal_kind == "literal_registered_forbidden_contact"
            else not categories
        ),
        "partial terminal forbidden-contact categories differ",
    )
    return {
        "present": True,
        "completed_physics_substeps": completed,
        "terminal_forbidden_contact_categories": list(categories),
        "terminal_forbidden_contact_records_sha256": partial_action_record[
            "terminal_forbidden_contact_records_sha256"
        ],
    }


def validate(
    result_path: Path,
    protocol_path: Path,
    historical_result_root: Path,
    numeric_validation_result: Path,
    *,
    expected_numeric_job_id: str,
    expected_producer_job_id: str,
    expected_producer_commit: str,
    expected_consumer_job_id: str,
    expected_consumer_commit: str,
) -> Dict[str, Any]:
    from main.poisson_fullbody.contracts import load_hashed_json
    from main.poisson_fullbody.osc_arm_link_canary import (
        RESULT_SCHEMA,
        classify_osc_arm_link_canary,
        validate_osc_arm_link_canary_protocol,
    )
    from main.poisson_fullbody.osc_numeric_prerequisite import (
        validate_numeric_prerequisite_artifact,
    )

    protocol = fast._json(protocol_path, "post-OSC canary protocol")
    derived = validate_osc_arm_link_canary_protocol(protocol)
    for value, label in (
        (expected_numeric_job_id, "numeric job ID"),
        (expected_producer_job_id, "producer job ID"),
        (expected_consumer_job_id, "consumer job ID"),
    ):
        _require(bool(re.fullmatch(r"[0-9]+", value)), "%s is malformed" % label)
    for value, label in (
        (expected_producer_commit, "producer commit"),
        (expected_consumer_commit, "consumer commit"),
    ):
        _require(
            bool(re.fullmatch(r"[0-9a-f]{40}", value)),
            "%s is malformed" % label,
        )
    _require(
        len(
            {
                expected_numeric_job_id,
                expected_producer_job_id,
                expected_consumer_job_id,
            }
        )
        == 3,
        "numeric, producer, and consumer must use distinct Slurm allocations",
    )
    consumer_slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    _require(
        consumer_slurm_job_id == expected_consumer_job_id,
        "consumer Slurm job ID differs",
    )
    repo_root = protocol_path.resolve().parents[1]
    consumer_source = fast._git(repo_root)
    _require(
        consumer_source.get("commit") == expected_consumer_commit,
        "consumer source commit differs",
    )
    registered_result_root = result_path.parent.parent
    _require(
        numeric_validation_result.is_absolute()
        and numeric_validation_result.name == "result.json"
        and numeric_validation_result.parent.parent == registered_result_root
        and numeric_validation_result.parent.name.startswith(
            "vlsa-poisson-post-osc-numeric-"
        ),
        "numeric prerequisite is outside the registered result namespace",
    )
    numeric_prerequisite = validate_numeric_prerequisite_artifact(
        numeric_validation_result,
        contract=derived["numeric_prerequisite"],
        expected_job_id=expected_numeric_job_id,
        expected_commit=expected_producer_commit,
        expected_python_executable=(
            "/mnt/data/quanth/venvs/safety_vla/main/bin/python"
        ),
        expected_parent=numeric_validation_result.parent,
        forbidden_job_id=expected_consumer_job_id,
    )
    selection = protocol["selection"]
    from main.poisson_fullbody.feasibility_protocol import load_feasibility_protocol

    runtime_path = repo_root / protocol["runtime"]["relative_path"]
    _require(
        fast._file_sha256(runtime_path) == protocol["runtime"]["raw_file_sha256"],
        "runtime protocol raw hash differs",
    )
    runtime_protocol, runtime_hashes = load_feasibility_protocol(
        runtime_path,
        expected_protocol_sha256=protocol["runtime"]["semantic_sha256"],
    )
    _require(
        runtime_hashes.parameter_block_sha256
        == protocol["runtime"]["parameter_block_sha256"],
        "runtime protocol parameter block differs",
    )
    study_protocol_path = repo_root / selection["study_protocol_relative_path"]
    manifest_path = repo_root / selection["manifest_relative_path"]
    manifest_receipt_path = repo_root / selection["manifest_receipt_relative_path"]
    _require(
        fast._file_sha256(study_protocol_path)
        == selection["study_protocol_file_sha256"],
        "study protocol hash differs",
    )
    _require(
        fast._file_sha256(manifest_path) == selection["manifest_file_sha256"],
        "contact manifest hash differs",
    )
    _require(
        fast._file_sha256(manifest_receipt_path)
        == selection["manifest_receipt_file_sha256"],
        "contact manifest receipt hash differs",
    )
    remote_receipt_path = repo_root / selection["remote_artifact_receipt_relative_path"]
    _require(
        fast._file_sha256(remote_receipt_path)
        == selection["remote_artifact_receipt_file_sha256"],
        "remote artifact receipt hash differs",
    )
    case, case_row_sha256 = _manifest_case(manifest_path, protocol["case"]["case_id"])
    _require(case_row_sha256 == selection["case_row_sha256"], "case row hash differs")
    historical_binding = case.get("historical_result")
    _require(isinstance(historical_binding, Mapping), "historical binding is absent")
    historical_path = _bound_file(
        historical_result_root,
        str(historical_binding.get("relative_path")),
        str(historical_binding.get("file_sha256")),
    )
    historical_value = fast._json(historical_path, "historical AEGIS result")
    from main.poisson_fullbody.shadow_replay import load_historical_action_replay

    replay = load_historical_action_replay(
        historical_path,
        expected_case_id=protocol["case"]["case_id"],
        expected_arm=protocol["historical_control"]["source_arm"],
    )
    _require(
        replay.result_file_sha256
        == protocol["historical_control"]["result_file_sha256"]
        and replay.result_payload_sha256
        == protocol["historical_control"]["result_payload_sha256"]
        and replay.executed_sequence_sha256
        == protocol["historical_control"]["executed_action_sequence_sha256"],
        "historical replay binding differs",
    )
    remote_receipt = fast._json(remote_receipt_path, "remote artifact receipt")
    _require(
        remote_receipt.get("case_id") == case["case_id"]
        and Path(str(remote_receipt.get("historical_root", ""))).resolve()
        == historical_result_root.resolve(),
        "remote receipt authority differs",
    )
    remote_rows = remote_receipt.get("artifacts")
    _require(isinstance(remote_rows, Sequence) and len(remote_rows) == 3, "remote receipt rows differ")
    auxiliary = case["artifact_bindings"]
    expected_remote = {
        "result.json": historical_binding["file_sha256"],
        "episode.mp4": auxiliary["video_sha256"],
        "active_obstacle_contacts.json.gz": auxiliary[
            "detailed_contact_gzip_sha256"
        ],
    }
    for remote_row in remote_rows:
        _require(isinstance(remote_row, Mapping), "remote receipt row is invalid")
        basename = Path(str(remote_row.get("relative_path", ""))).name
        _require(basename in expected_remote, "remote receipt filename differs")
        target = _bound_file(
            historical_result_root,
            str(remote_row["relative_path"]),
            str(expected_remote[basename]),
        )
        _require(
            remote_row.get("regular_file") is True
            and remote_row.get("symlink") is False
            and int(remote_row.get("byte_count", -1)) == int(target.stat().st_size),
            "remote artifact regular-file receipt differs",
        )
    result = load_hashed_json(result_path)
    _require(result.get("schema_version") == RESULT_SCHEMA, "result schema differs")
    _require(result.get("status") == "complete", "partial or failed result rejected")
    _require(result.get("scientific_result") is True, "result is not scientific")
    _require(result.get("partial_output_interpreted") is False, "partial-output flag differs")
    _require(result.get("protocol_id") == protocol["protocol_id"], "protocol ID differs")
    _require(result.get("case_id") == protocol["case"]["case_id"], "case differs")
    _require(result.get("claim_scope") == protocol["claim_scope"], "claim scope differs")
    provenance = result.get("provenance")
    _require(isinstance(provenance, Mapping), "provenance is absent")
    _require(
        fast._canonical(provenance.get("numeric_prerequisite"))
        == fast._canonical(numeric_prerequisite),
        "producer numeric prerequisite binding differs",
    )
    source = provenance.get("source")
    _require(isinstance(source, Mapping) and not source.get("status_short"), "source is dirty")
    _require(
        source.get("commit") == expected_producer_commit,
        "producer source commit differs",
    )
    allocation = provenance.get("allocation")
    _require(isinstance(allocation, Mapping), "Slurm allocation is absent")
    _require(
        str(allocation.get("slurm_job_id")) == expected_producer_job_id,
        "producer Slurm job ID differs",
    )
    _require(
        isinstance(allocation.get("gpu_name"), str)
        and "H100" in allocation["gpu_name"],
        "producer H100 allocation is not verified",
    )
    expected_evaluation_python = Path(
        "/mnt/data/quanth/venvs/safety_vla/main/bin/python"
    ).resolve()
    _require(
        Path(str(provenance.get("python_executable", ""))).resolve()
        == expected_evaluation_python
        and Path(sys.executable).resolve() == expected_evaluation_python,
        "producer or consumer evaluation Python differs",
    )
    producer_versions = provenance.get("evaluation_package_versions")
    _require(isinstance(producer_versions, Mapping), "producer package versions are absent")
    consumer_versions = {
        package: importlib.metadata.version(package)
        for package in ("mujoco", "numpy", "scipy", "osqp", "robosuite")
    }
    _require(
        dict(producer_versions) == consumer_versions,
        "producer and consumer evaluation package versions differ",
    )
    _require(provenance.get("historical_control_rerun") is False, "baseline was rerun")
    _require(
        provenance.get("protocol_file_sha256") == fast._file_sha256(protocol_path),
        "protocol raw hash differs",
    )
    for field, expected in (
        ("study_protocol_file_sha256", selection["study_protocol_file_sha256"]),
        ("manifest_file_sha256", selection["manifest_file_sha256"]),
        ("manifest_receipt_file_sha256", selection["manifest_receipt_file_sha256"]),
        (
            "remote_artifact_receipt_file_sha256",
            selection["remote_artifact_receipt_file_sha256"],
        ),
        ("manifest_case_row_sha256", selection["case_row_sha256"]),
    ):
        _require(provenance.get(field) == expected, "%s differs" % field)
    checkpoint_identity = provenance.get("checkpoint_identity")
    _require(isinstance(checkpoint_identity, Mapping), "checkpoint identity is absent")
    from scripts.run_poisson_closed_loop_canary import _checkpoint_identity

    independently_observed_checkpoint = _checkpoint_identity(protocol["online_policy"])
    _require(
        fast._canonical(checkpoint_identity)
        == fast._canonical(independently_observed_checkpoint)
        and checkpoint_identity.get("full_content_tree_sha256")
        == protocol["online_policy"]["checkpoint_tree_sha256"]
        and checkpoint_identity.get("receipt_file_sha256")
        == protocol["online_policy"]["checkpoint_receipt_sha256"]
        and checkpoint_identity.get("current_filesystem_identity_matches_receipt")
        is True,
        "producer checkpoint identity differs from independent filesystem receipt",
    )

    treatment = result.get("treatment")
    metrics = result.get("metrics")
    planner = result.get("planner")
    video = result.get("video")
    historical = result.get("historical_control")
    apparatus = result.get("apparatus")
    for value, label in (
        (treatment, "treatment"),
        (metrics, "metrics"),
        (planner, "planner"),
        (video, "video"),
        (historical, "historical_control"),
        (apparatus, "apparatus"),
    ):
        _require(isinstance(value, Mapping), "%s is absent" % label)
    field_audit = _validate_field_sample_static_evidence(
        apparatus=apparatus,
        treatment=treatment,
        metrics=metrics,
        runtime_protocol=runtime_protocol,
        runtime_protocol_sha256=runtime_hashes.protocol_sha256,
        runtime_parameter_block_sha256=runtime_hashes.parameter_block_sha256,
    )
    _require(
        metrics.get("contact_scope") == protocol["execution"]["contact_scope"],
        "registered union contact scope differs",
    )
    _require(historical.get("rerun") is False, "historical baseline rerun differs")
    direct_pairs = case.get("direct_link_active_obstacle_pairs")
    historical_telemetry = historical_value.get("contact_telemetry")
    historical_unique_pairs = (
        historical_telemetry.get("unique_contact_pairs")
        if isinstance(historical_telemetry, Mapping)
        else None
    )
    historical_geom_pairs = {
        (str(row.get("geom1")), str(row.get("geom2")))
        for row in historical_unique_pairs or ()
        if isinstance(row, Mapping)
    }
    direct_historical_contact = bool(
        isinstance(direct_pairs, Sequence)
        and direct_pairs
        and any(
            isinstance(row, Mapping)
            and row.get("actual_link_body_name") == "robot0_link5"
            and (
                (
                    str(row.get("active_obstacle_geom_name")),
                    str(row.get("link_geom_name")),
                )
                in historical_geom_pairs
                or (
                    str(row.get("link_geom_name")),
                    str(row.get("active_obstacle_geom_name")),
                )
                in historical_geom_pairs
            )
            for row in direct_pairs
        )
        and historical_telemetry.get("first_contact_step")
        == int(derived["historical_contact_action"])
    )
    _require(
        direct_historical_contact
        and historical.get("direct_link56_selected_obstacle_contact_verified")
        is True
        and metrics.get("historical_direct_link56_contact_verified") is True,
        "historical direct contact is absent",
    )
    _require(
        replay.historical_task_success is True
        and historical.get("historical_task_success") is True
        and metrics.get("historical_control_task_success") is True,
        "historical task failed",
    )
    controller = apparatus.get("controller")
    _require(
        isinstance(controller, Mapping)
        and controller.get("original_osc_verified") is True
        and controller.get("controller_name") == "OSC_POSE"
        and controller.get("controller_class_qualname")
        == "OperationalSpaceController"
        and str(controller.get("controller_class_module", "")).endswith(".osc")
        and bool(
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(controller.get("controller_implementation_file_sha256", "")),
            )
        )
        and controller.get("controller_control_dim") == 6
        and controller.get("environment_action_dim") == 7
        and controller.get("control_frequency_hz") == 20
        and controller.get("physics_frequency_hz") == 500,
        "original OSC cadence is not verified",
    )
    controller_module = importlib.import_module(
        str(controller["controller_class_module"])
    )
    controller_class = getattr(
        controller_module, str(controller["controller_class_qualname"]), None
    )
    controller_source_path = inspect.getsourcefile(controller_class)
    _require(
        controller_source_path is not None
        and fast._file_sha256(Path(controller_source_path))
        == controller["controller_implementation_file_sha256"],
        "consumer-installed OSC implementation differs from producer",
    )
    torque_units = apparatus.get("arm_actuator_torque_units")
    _require(
        apparatus.get("field_frame")
        == "world_frame_frozen_at_post_settling_selected_obstacle_pose",
        "Poisson obstacle-field frame differs",
    )
    _require(
        isinstance(torque_units, Mapping)
        and torque_units.get("schema_version")
        == "vlsa_poisson_direct_hinge_torque_units.v1"
        and torque_units.get("control_unit") == "newton_metre",
        "arm actuator torque-unit contract is absent",
    )
    torque_actuators = torque_units.get("actuators")
    _require(
        isinstance(torque_actuators, Sequence) and len(torque_actuators) == 7,
        "torque-unit actuator ledger differs",
    )
    actuator_ids = set()
    joint_ids = set()
    torque_units_verified = True
    required_torque_checks = {
        "joint_transmission",
        "transmission_joint_matches",
        "hinge_joint",
        "fixed_gain",
        "unit_gain",
        "no_bias",
        "no_activation_dynamics",
        "control_limit_enabled",
        "unit_joint_gear",
        "force_limit_cannot_clip_control_range",
    }
    for actuator in torque_actuators:
        _require(isinstance(actuator, Mapping), "torque actuator row is invalid")
        actuator_ids.add(int(actuator.get("actuator_id", -1)))
        joint_ids.add(int(actuator.get("joint_id", -1)))
        checks = actuator.get("checks")
        control_range = actuator.get("control_range")
        force_range = actuator.get("force_range")
        gain_parameters = actuator.get("gain_parameters")
        bias_parameters = actuator.get("bias_parameters")
        gear = actuator.get("gear")
        raw_limit_valid = bool(
            actuator.get("control_limited") is True
            and isinstance(control_range, list)
            and len(control_range) == 2
            and all(math.isfinite(float(value)) for value in control_range)
            and float(control_range[0]) < float(control_range[1])
            and isinstance(force_range, list)
            and len(force_range) == 2
            and all(math.isfinite(float(value)) for value in force_range)
            and (
                actuator.get("force_limited") is False
                or (
                    actuator.get("force_limited") is True
                    and float(force_range[0]) <= float(control_range[0])
                    and float(force_range[1]) >= float(control_range[1])
                )
            )
            and isinstance(gain_parameters, list)
            and len(gain_parameters) == 10
            and float(gain_parameters[0]) == 1.0
            and all(float(value) == 0.0 for value in gain_parameters[1:])
            and isinstance(bias_parameters, list)
            and len(bias_parameters) == 10
            and all(float(value) == 0.0 for value in bias_parameters)
            and isinstance(gear, list)
            and len(gear) == 6
            and float(gear[0]) == 1.0
            and all(float(value) == 0.0 for value in gear[1:])
        )
        row_verified = bool(
            isinstance(checks, Mapping)
            and set(checks) == required_torque_checks
            and all(checks.get(field) is True for field in required_torque_checks)
            and actuator.get("verified") is True
            and raw_limit_valid
        )
        torque_units_verified = torque_units_verified and row_verified
    torque_units_verified = bool(
        torque_units_verified
        and len(actuator_ids) == 7
        and len(joint_ids) == 7
        and torque_units.get("verified") is True
    )
    _require(
        torque_units_verified
        and metrics.get("direct_unit_gain_hinge_torque_actuators_verified") is True,
        "direct unit-gain hinge torque authority differs",
    )

    physics = treatment.get("physics_trace")
    actions = treatment.get("action_trace")
    parity = treatment.get("parity_trace")
    goals = treatment.get("goal_progress_ledger")
    car_ledger = treatment.get("paper_car_endpoint_ledger")
    _require(isinstance(physics, Sequence), "physics trace is absent")
    _require(isinstance(actions, Sequence), "action trace is absent")
    _require(isinstance(parity, Sequence), "parity trace is absent")
    _require(isinstance(goals, Sequence), "goal trace is absent")
    _require(isinstance(car_ledger, Sequence) and car_ledger, "paper CAR ledger is absent")
    _require(
        treatment.get("paper_car_endpoint_ledger_schema_version")
        == "vlsa_poisson_paper_car_endpoint_ledger.v2",
        "paper CAR ledger schema differs",
    )
    car_authority = apparatus.get("paper_car_authority")
    resolved_car_geometry = apparatus.get("resolved_geometry")
    resolved_obstacle_roots = (
        list(resolved_car_geometry.get("obstacle_root_body_ids", ()))
        if isinstance(resolved_car_geometry, Mapping)
        else []
    )
    _require(isinstance(car_authority, Mapping), "paper CAR authority is absent")
    observable_car_root_id = _strict_integer(
        car_authority.get("observable_root_body_id"),
        "paper CAR observable root body ID",
    )
    contact_car_root_id = _strict_integer(
        car_authority.get("contact_authority_root_body_id"),
        "paper CAR contact root body ID",
    )
    _require(
        len(resolved_obstacle_roots) == 1,
        "paper CAR resolved obstacle root count differs",
    )
    resolved_car_root_id = _strict_integer(
        resolved_obstacle_roots[0], "paper CAR resolved obstacle root body ID"
    )
    _require(
        car_authority.get("schema_version")
        == "vlsa_poisson_paper_car_authority.v1"
        and car_authority.get("metric")
        == protocol["paper_car_measurement"]["metric"]
        and car_authority.get("paper_car_observation_key")
        == "%s_pos" % protocol["case"]["selected_obstacle_name"]
        and car_authority.get("selected_obstacle_name")
        == protocol["case"]["selected_obstacle_name"]
        and observable_car_root_id == contact_car_root_id == resolved_car_root_id
        and car_authority.get("observable_root_body_name")
        == protocol["case"]["selected_obstacle_root_body_name"]
        and car_authority.get("contact_authority_root_body_name")
        == protocol["case"]["selected_obstacle_root_body_name"]
        and car_authority.get("observable_and_contact_root_body_ids_equal") is True
        and car_authority.get("position_source")
        == protocol["paper_car_measurement"]["position_source"]
        and car_authority.get("root_body_binding")
        == protocol["paper_car_measurement"]["root_body_binding"]
        and car_authority.get("post_integration_forwarded_pose_role")
        == protocol["paper_car_measurement"][
            "post_integration_forwarded_pose_role"
        ]
        and car_authority.get(
            "historical_settled_active_obstacle_position_sha256"
        )
        == replay.settled_active_obstacle_position_sha256,
        "paper CAR observation/root-body authority differs",
    )
    _require_resolved_body_name(
        resolved_car_geometry,
        ids_field="obstacle_body_ids",
        names_field="obstacle_body_names",
        expected_body_id=observable_car_root_id,
        expected_body_name=protocol["case"]["selected_obstacle_root_body_name"],
        label="paper CAR obstacle",
    )
    protected_samples = apparatus["protected_sampling"]["samples"]
    _require(
        isinstance(protected_samples, Sequence)
        and len(protected_samples) == int(field_audit["protected_sample_count"]),
        "protected sample attribution ledger differs",
    )
    _require(
        treatment.get("physics_trace_schema_version")
        == "vlsa_poisson_osc_arm_link_compact_physics_trace.v1"
        and treatment.get("full_qp_certificate_scope")
        == "first_byte_different_torque_row_only",
        "compact physics trace contract differs",
    )
    _require(
        [int(row["physical_boundary"]) for row in physics] == list(range(len(physics))),
        "physics boundaries are incomplete or duplicated",
    )
    _require(
        all(
            int(row["source_action_index"]) == int(row["physical_boundary"]) // 25
            and int(row["physics_substep_index"]) == int(row["physical_boundary"]) % 25
            for row in physics
        ),
        "physics/action cadence differs",
    )
    _require(
        all(
            row.get("shield_status")
            in ("nominal_safe_exact_clone", "solved")
            for row in physics
        ),
        "invalid shield decision is present",
    )
    _require(
        all(
            isinstance(row.get("candidate_exact_clone_contact"), Mapping)
            and row["candidate_exact_clone_contact"].get("literal_contact") is False
            and float(row["candidate_exact_clone_minimum_cbf_residual_m2_per_s"])
            >= -float(protocol["shield"]["actual_cbf_residual_tolerance_m2_per_s"])
            for row in physics
        ),
        "exact nonlinear candidate pre-physics postcheck failed",
    )
    import numpy as np

    solved_qp_count = 0
    independently_validated_qp_audits: List[Dict[str, Any]] = []
    first_byte_divergence_seen = False
    for index, row in enumerate(physics):
        compact_arrays = _validate_compact_constraint_trace(
            row,
            expected_sample_count=int(field_audit["protected_sample_count"]),
            np=np,
        )
        compact_diagnostics = _validate_compact_shield_diagnostics(
            row,
            expected_sample_count=int(field_audit["protected_sample_count"]),
        )
        nominal_torque = np.asarray(row.get("nominal_torque_nm"), dtype=np.float64)
        command_torque = np.asarray(row.get("command_torque_nm"), dtype=np.float64)
        torque_delta = np.asarray(row.get("torque_delta_nm"), dtype=np.float64)
        measured_qvel = np.asarray(row.get("measured_qvel_rad_s"), dtype=np.float64)
        candidate_qvel = np.asarray(
            row.get("predicted_next_qvel_rad_s"), dtype=np.float64
        )
        nominal_qvel = np.asarray(
            row.get("nominal_predicted_next_qvel_rad_s"), dtype=np.float64
        )
        _require(
            nominal_torque.shape == (7,)
            and command_torque.shape == (7,)
            and torque_delta.shape == (7,)
            and measured_qvel.shape == (7,)
            and candidate_qvel.shape == (7,)
            and nominal_qvel.shape == (7,),
            "compact physics vectors are invalid at physics row %d" % index,
        )
        for array in (
            nominal_torque,
            command_torque,
            torque_delta,
            measured_qvel,
            candidate_qvel,
            nominal_qvel,
        ):
            _require(np.all(np.isfinite(array)), "non-finite physics array at row %d" % index)
        reconstructed_delta = command_torque - nominal_torque
        _require(
            np.allclose(reconstructed_delta, torque_delta, rtol=0.0, atol=1e-12),
            "torque delta differs at row %d" % index,
        )
        _close(
            row.get("torque_correction_l2_nm"),
            float(np.linalg.norm(torque_delta)),
            "torque correction row %d" % index,
        )
        _close(
            row.get("measured_qvel_l2_rad_s"),
            float(np.linalg.norm(measured_qvel)),
            "qvel norm row %d" % index,
        )
        _close(
            row.get("prediction_error_l2_rad_s"),
            float(np.linalg.norm(measured_qvel - candidate_qvel)),
            "prediction error row %d" % index,
        )
        byte_equal = bool(
            command_torque.tobytes(order="C")
            == nominal_torque.tobytes(order="C")
        )
        _require(
            row.get("command_byte_identical_to_nominal") is byte_equal
            and row.get("nominal_torque_sha256")
            == hashlib.sha256(nominal_torque.tobytes(order="C")).hexdigest()
            and row.get("command_torque_sha256")
            == hashlib.sha256(command_torque.tobytes(order="C")).hexdigest(),
            "torque byte authority differs at row %d" % index,
        )
        _require(
            bool(
                np.allclose(
                    measured_qvel, candidate_qvel, rtol=0.0, atol=1e-10
                )
            )
            is bool(row.get("candidate_exact_clone_matches_live_qvel")),
            "candidate/live equality flag differs at row %d" % index,
        )
        candidate_contact = row.get("candidate_exact_clone_contact")
        _require(
            isinstance(candidate_contact, Mapping)
            and candidate_contact.get("literal_contact") is False
            and int(candidate_contact.get("literal_contact_count", -1)) == 0
            and candidate_contact.get("contacts") == []
            and candidate_contact.get("phases_checked")
            == [
                "live_solver_phase_preintegration_geometry",
                "post_integration_recomputed",
            ],
            "candidate clone contact union differs at row %d" % index,
        )
        if row.get("shield_status") == "nominal_safe_exact_clone":
            _require(
                byte_equal
                and np.array_equal(torque_delta, np.zeros(7))
                and compact_diagnostics.get("solver_attempted") is False,
                "nominal exact pass-through evidence differs",
            )
        else:
            _require(
                row.get("shield_status") == "solved"
                and compact_diagnostics.get("solver_attempted") is True,
                "solved QP evidence differs",
            )
            solved_qp_count += 1
        is_first_byte_divergence = bool(
            not byte_equal and not first_byte_divergence_seen
        )
        if is_first_byte_divergence:
            _require(
                row.get("shield_status") == "solved",
                "first byte-different torque is not a solved QP",
            )
            independently_validated_qp_audits.append(
                _validate_first_divergence_full_certificate(
                    row,
                    compact_arrays=compact_arrays,
                    compact_diagnostics=compact_diagnostics,
                    protocol=protocol,
                    controller=controller,
                    torque_actuators=torque_actuators,
                    protected_samples=protected_samples,
                    np=np,
                )
            )
            first_byte_divergence_seen = True
        else:
            _require(
                row.get("full_constraint_qp_certificate") is None
                and row.get("full_constraint_qp_certificate_sha256") is None,
                "full QP certificate appears outside the first divergence",
            )
        residual_scalars = (
            "minimum_actual_cbf_residual_m2_per_s",
            "candidate_exact_clone_minimum_cbf_residual_m2_per_s",
            "nominal_exact_clone_minimum_cbf_residual_m2_per_s",
        )
        _require(
            all(
                not isinstance(row.get(field), bool)
                and math.isfinite(float(row.get(field)))
                for field in residual_scalars
            ),
            "compact residual minima are invalid at row %d" % index,
        )
        _require(
            float(row["candidate_exact_clone_minimum_cbf_residual_m2_per_s"])
            >= -float(protocol["shield"]["actual_cbf_residual_tolerance_m2_per_s"])
            and float(row["minimum_actual_cbf_residual_m2_per_s"])
            >= -float(protocol["shield"]["actual_cbf_residual_tolerance_m2_per_s"]),
            "compact CBF residual minimum failed at physics row %d" % index,
        )
    _require(
        int(metrics.get("solved_qp_count", -1)) == solved_qp_count
        and int(metrics.get("first_divergence_full_qp_certificate_count", -1))
        == len(independently_validated_qp_audits),
        "producer QP/certificate counts differ",
    )
    _require(
        all(
            row.get("actual_cbf_residual_postcheck_pass") is True
            and row.get("candidate_exact_clone_matches_live_qvel") is True
            for row in physics
        ),
        "actual CBF residual postcheck failed",
    )
    first_material_rows = [
        row
        for row in physics
        if float(row["torque_correction_l2_nm"])
        >= derived["acceptance"]["material_torque_correction_l2_nm"]
    ]
    first_material = (
        int(first_material_rows[0]["physical_boundary"])
        if first_material_rows
        else None
    )
    first_action = None if first_material is None else first_material // 25
    divergence_rows = [
        row
        for row in physics
        if row.get("command_byte_identical_to_nominal") is False
    ]
    first_divergence = (
        int(divergence_rows[0]["physical_boundary"])
        if divergence_rows
        else None
    )
    first_divergence_action = (
        None if first_divergence is None else first_divergence // 25
    )
    first_divergence_correction = (
        None
        if not divergence_rows
        else float(divergence_rows[0]["torque_correction_l2_nm"])
    )
    first_divergence_is_material = bool(
        first_divergence is not None
        and first_divergence == first_material
        and first_divergence_correction
        >= derived["acceptance"]["material_torque_correction_l2_nm"]
    )
    _require(
        metrics.get("first_torque_divergence_physical_boundary")
        == first_divergence
        and metrics.get("first_torque_divergence_source_action_index")
        == first_divergence_action
        and metrics.get("first_torque_divergence_is_material")
        is first_divergence_is_material,
        "first byte-different torque divergence differs",
    )
    divergence_attribution = (
        None
        if not independently_validated_qp_audits
        else independently_validated_qp_audits[0]["constraint_attribution"]
    )
    _require(
        metrics.get("first_divergence_minimum_constraint_body_name")
        == (
            None
            if divergence_attribution is None
            else divergence_attribution["minimum_sample"]["body_name"]
        )
        and metrics.get("first_divergence_near_minimum_constraint_body_names")
        == (
            []
            if divergence_attribution is None
            else divergence_attribution["near_minimum_body_names"]
        )
        and metrics.get("first_divergence_negative_constraint_body_names")
        == (
            []
            if divergence_attribution is None
            else divergence_attribution["negative_nominal_residual_body_names"]
        ),
        "first divergence protected-link body attribution differs",
    )
    if first_divergence_correction is None:
        _require(
            metrics.get("first_torque_divergence_correction_l2_nm") is None,
            "absent first divergence correction differs",
        )
    else:
        _close(
            metrics.get("first_torque_divergence_correction_l2_nm"),
            first_divergence_correction,
            "first divergence correction",
        )
    first_material_residual_improvement = (
        None
        if not first_material_rows
        else float(
            first_material_rows[0][
                "candidate_exact_clone_minimum_cbf_residual_m2_per_s"
            ]
            - first_material_rows[0][
                "nominal_exact_clone_minimum_cbf_residual_m2_per_s"
            ]
        )
    )
    first_material_next_qvel_change = (
        None
        if not first_material_rows
        else float(
            np.linalg.norm(
                np.asarray(
                    first_material_rows[0]["predicted_next_qvel_rad_s"],
                    dtype=np.float64,
                )
                - np.asarray(
                    first_material_rows[0]["nominal_predicted_next_qvel_rad_s"],
                    dtype=np.float64,
                )
            )
        )
    )
    _require(
        metrics.get("first_material_correction_physical_boundary") == first_material,
        "first material correction boundary differs",
    )
    _require(
        metrics.get("first_material_correction_source_action_index") == first_action,
        "first material correction action differs",
    )
    _require(
        metrics.get("material_correction_present") is (first_material is not None),
        "material-correction flag differs",
    )
    if first_material_residual_improvement is None:
        _require(
            metrics.get("first_material_exact_cbf_residual_improvement_m2_per_s")
            is None
            and metrics.get("first_material_exact_next_qvel_change_l2_rad_s")
            is None,
            "absent directed safety improvement differs",
        )
    else:
        _close(
            metrics.get(
                "first_material_exact_cbf_residual_improvement_m2_per_s"
            ),
            first_material_residual_improvement,
            "first material exact CBF residual improvement",
        )
        _close(
            metrics.get("first_material_exact_next_qvel_change_l2_rad_s"),
            first_material_next_qvel_change,
            "first material exact next-qvel change",
        )
    nominal_negative = bool(
        divergence_rows
        and float(
            divergence_rows[0][
                "nominal_exact_clone_minimum_cbf_residual_m2_per_s"
            ]
        )
        < 0.0
    )
    _require(
        metrics.get("first_divergence_nominal_exact_cbf_residual_negative")
        is nominal_negative,
        "first divergence nominal residual sign differs",
    )
    correction_goal = treatment.get("first_material_correction_goal_snapshot")
    expected_task_incomplete = None
    if first_material is not None:
        _require(isinstance(correction_goal, Mapping), "correction-time goal snapshot is absent")
        _require(
            int(correction_goal.get("step", -1)) == first_action
            and correction_goal.get("inert") is True
            and correction_goal.get("simulator_state_sha256_before")
            == correction_goal.get("simulator_state_sha256_after"),
            "correction-time goal snapshot is not state-inert",
        )
        expected_task_incomplete = not bool(correction_goal.get("all_satisfied"))
    _require(
        metrics.get("task_incomplete_at_first_material_correction")
        is expected_task_incomplete,
        "task state at first correction differs",
    )
    prior_live_contact_free = bool(
        first_material is not None
        and all(
            row.get("registered_forbidden_contact_seen") is False
            for row in physics
            if int(row["physical_boundary"]) < int(first_material)
        )
    )
    predivergence_exact = all(
        row.get("command_byte_identical_to_nominal") is True
        for row in physics
        if first_divergence is None
        or int(row["physical_boundary"]) < first_divergence
    )
    prefix_contact_free = bool(
        first_divergence is not None
        and all(
            row.get("registered_forbidden_contact_seen") is False
            for row in physics
            if int(row["physical_boundary"]) < first_divergence
        )
    )
    action62_exception = bool(
        first_divergence_action is not None
        and (
            first_divergence_action < int(derived["historical_contact_action"])
            or (
                first_divergence_action == int(derived["historical_contact_action"])
                and len(parity) == int(derived["historical_contact_action"])
            )
        )
    )
    expected_before = bool(
        first_action is not None
        and first_divergence_is_material
        and (
            first_action < int(derived["historical_contact_action"])
            or (
                first_action == int(derived["historical_contact_action"])
                and prior_live_contact_free
                and prefix_contact_free
                and predivergence_exact
                and action62_exception
            )
        )
    )
    _require(
        metrics.get("material_correction_before_historical_contact") is expected_before,
        "correction/contact ordering differs",
    )
    _require(
        metrics.get("live_substeps_before_material_correction_contact_free")
        is prior_live_contact_free,
        "live pre-correction contact-free flag differs",
    )
    expected_parity_count = (
        len(actions) if first_divergence_action is None else first_divergence_action
    )
    _require(len(parity) == expected_parity_count, "pre-correction parity count differs")
    _require(
        [int(row.get("source_action_index", -1)) for row in parity]
        == list(range(expected_parity_count)),
        "historical parity indexes differ",
    )
    _require(
        [int(row.get("source_action_index", -1)) for row in actions]
        == list(range(len(actions))),
        "completed action indexes differ",
    )
    goal_by_index = {
        int(row.get("source_action_index", -1)): row for row in goals
    }
    _require(len(goal_by_index) == len(goals), "goal indexes are duplicated")
    for index in range(expected_parity_count):
        parity_row = parity[index]
        action_row = actions[index]
        goal_row = goal_by_index.get(index)
        historical_step = replay.steps[index]
        _require(isinstance(goal_row, Mapping), "historical goal row is absent")
        _require(
            parity_row.get("historical_match") is True
            and parity_row.get("simulator_state_sha256")
            == historical_step.simulator_state_sha256
            and action_row.get("state_sha256")
            == historical_step.simulator_state_sha256
            and parity_row.get("observation_sha256")
            == action_row.get("observation_sha256")
            == goal_row.get("observation_sha256")
            and np.array_equal(
                np.asarray(action_row.get("executed_high_level_action"), dtype=np.float64),
                np.asarray(historical_step.action, dtype=np.float64),
            )
            and float(goal_row.get("reward")) == historical_step.reward
            and bool(goal_row.get("done")) is historical_step.done
            and tuple(goal_row.get("values", ())) == historical_step.goal_values,
            "historical action/state/reward/goal parity differs at action %d" % index,
        )
    _require(
        metrics.get("all_predivergence_torque_commands_byte_identical_nominal")
        is predivergence_exact
        and metrics.get("all_live_substeps_before_first_divergence_contact_free")
        is prefix_contact_free
        and metrics.get(
            "action62_same_action_exception_has_action61_endpoint_parity_exact"
        )
        is action62_exception,
        "predivergence authority differs",
    )

    trace_contact = any(
        row.get("literal_robot_selected_obstacle_contact_seen") is True
        for row in physics
    )
    measurement = treatment.get("measurement")
    _require(isinstance(measurement, Mapping), "authoritative measurement is absent")
    _require(
        int(measurement.get("observed_physics_substeps")) == len(physics),
        "measurement contact/cadence differs from physics trace",
    )
    physical_count = int(
        measurement.get("total_physical_contact_point_record_count")
    )
    rollout_count = int(
        measurement.get("rollout_phase_physical_contact_point_record_count")
    )
    post_records = measurement.get("post_state_physical_contact_point_records")
    live_records = measurement.get("live_solver_phase_contact_point_records")
    _require(
        isinstance(post_records, Sequence) and isinstance(live_records, Sequence),
        "measurement physical-contact record ledgers are absent",
    )
    settled_measurement = measurement.get("settled_state")
    _require(isinstance(settled_measurement, Mapping), "settled contact evidence is absent")
    settled_records = settled_measurement.get("physical_contact_point_records")
    _require(isinstance(settled_records, Sequence), "settled physical records are absent")
    for record in live_records:
        _require(isinstance(record, Mapping), "live contact row is invalid")
        distance = float(record.get("contact_distance_m"))
        _require(
            math.isfinite(distance)
            and record.get("is_physical_nonpositive_distance_contact")
            is (distance <= 0.0),
            "live contact distance/flag differs",
        )
    live_physical = [
        row for row in live_records if float(row["contact_distance_m"]) <= 0.0
    ]
    _require(
        all(
            isinstance(row, Mapping)
            and math.isfinite(float(row.get("contact_distance_m")))
            and float(row.get("contact_distance_m")) <= 0.0
            and row.get("is_physical_nonpositive_distance_contact") is True
            for row in list(settled_records) + list(post_records)
        ),
        "physical record ledger contains a nonphysical row",
    )
    rollout_union = live_physical + list(post_records)
    all_physical_records = list(settled_records) + rollout_union
    measured_contact = bool(all_physical_records)
    _require(
        trace_contact is bool(rollout_union)
        and metrics.get("any_robot_selected_obstacle_contact") is measured_contact
        and measurement.get("any_robot_obstacle_contact") is measured_contact
        and measurement.get("rollout_any_robot_obstacle_contact")
        is bool(rollout_union),
        "contact union differs from trace or metrics",
    )
    resolved_geometry = apparatus.get("resolved_geometry")
    _require(isinstance(resolved_geometry, Mapping), "resolved geometry is absent")
    robot_geom_ids = {int(value) for value in resolved_geometry.get("robot_geom_ids", ())}
    obstacle_geom_ids = {int(value) for value in resolved_geometry.get("obstacle_geom_ids", ())}
    link_geom_ids = {int(value) for value in resolved_geometry.get("link56_geom_ids", ())}
    _require(robot_geom_ids and obstacle_geom_ids and link_geom_ids, "resolved contact IDs are absent")
    for record in all_physical_records:
        _require(
            int(record.get("robot_geom_id", -1)) in robot_geom_ids
            and int(record.get("obstacle_geom_id", -1)) in obstacle_geom_ids,
            "physical contact record lies outside registered scope",
        )
    reconstructed_link_contact = any(
        int(record.get("robot_geom_id", -1)) in link_geom_ids
        for record in all_physical_records
    )
    _require(
        measurement.get("link56_obstacle_contact") is reconstructed_link_contact
        and metrics.get("link56_selected_obstacle_contact")
        is reconstructed_link_contact,
        "link5/6 contact union differs",
    )
    _require(
        physical_count == len(all_physical_records)
        and rollout_count == len(rollout_union)
        and int(measurement.get("post_state_physical_contact_point_record_count"))
        == len(post_records)
        and int(measurement.get("live_solver_nonpositive_contact_point_record_count"))
        == len(live_physical),
        "physical contact counts differ from record union",
    )
    if measured_contact:
        _require(
            physical_count > 0
            and rollout_count > 0
            and measurement.get("first_physical_contact_point_record") is not None,
            "contact flag lacks authoritative physical records",
        )
    else:
        _require(
            physical_count == 0
            and rollout_count == 0
            and measurement.get("first_physical_contact_point_record") is None
            and not post_records
            and not any(
                row.get("is_physical_nonpositive_distance_contact") is True
                for row in live_records
                if isinstance(row, Mapping)
            ),
            "contact-free flag conflicts with physical-contact records",
        )
    registered_contact_audit = _validate_registered_contact_evidence(
        scope=apparatus.get("registered_contact_scope"),
        measurement=treatment.get("registered_contact_measurement"),
        physics=physics,
        resolved_geometry=resolved_geometry,
    )
    registered_contact = bool(registered_contact_audit["rollout_contact"])
    _require(
        registered_contact_audit["settled_contact"] is False
        and metrics.get("any_registered_forbidden_contact")
        is registered_contact
        and metrics.get("any_robot_selected_obstacle_contact")
        is registered_contact_audit["any_robot_selected_obstacle_contact"]
        and metrics.get("any_link56_external_nonrobot_contact")
        is registered_contact_audit["any_link56_external_nonrobot_contact"]
        and metrics.get("any_link56_nonselected_external_contact")
        is registered_contact_audit["any_link56_nonselected_external_contact"]
        and measured_contact
        is registered_contact_audit["any_robot_selected_obstacle_contact"]
        and metrics.get("every_executed_substep_registered_contact_scope_monitored")
        is True
        and metrics.get("every_executed_substep_contact_monitored") is True,
        "registered union contact metrics differ",
    )
    _require(
        int(metrics.get("executed_physics_substep_count")) == len(physics)
        and int(metrics.get("shield_decision_count")) == len(physics),
        "shield exposure count differs",
    )
    _require(
        metrics.get("no_policy_query_before_divergence") is True
        and planner.get("no_policy_query_before_divergence") is True,
        "policy was queried before divergence",
    )
    divergence = planner.get("divergence_source_action_index")
    _require(
        divergence == first_divergence_action
        and planner.get("divergence_physical_boundary") == first_divergence,
        "planner divergence differs from first byte-different torque",
    )
    planner_actions = planner.get("action_trace")
    policy_queries = planner.get("policy_queries")
    _require(
        isinstance(planner_actions, Sequence)
        and isinstance(policy_queries, Sequence)
        and [int(row.get("source_action_index", -1)) for row in planner_actions]
        == list(range(len(planner_actions))),
        "planner action cadence differs",
    )
    _require(
        planner.get("action_trace_sha256") == _canonical_sha256(planner_actions)
        and planner.get("policy_query_trace_sha256")
        == _canonical_sha256(policy_queries),
        "planner trace aggregate hash differs",
    )
    terminal_kind = treatment.get("terminal_kind")
    partial_action_record = treatment.get("partial_action_record")
    partial_action_audit = _validate_partial_action_ledger(
        terminal_kind=str(terminal_kind),
        actions=actions,
        planner_actions=planner_actions,
        partial_action_record=partial_action_record,
        physics_substep_count=len(physics),
    )
    if terminal_kind == "literal_registered_forbidden_contact":
        _require(
            partial_action_audit.get("terminal_forbidden_contact_categories")
            == registered_contact_audit["rollout_contact_categories"]
            and partial_action_audit.get(
                "terminal_forbidden_contact_records_sha256"
            )
            == registered_contact_audit["rollout_contact_records_sha256"],
            "partial contact terminal is not bound to registered contact evidence",
        )
    policy_server = planner.get("policy_server")
    endpoint = planner.get("policy_server_endpoint")
    if policy_queries:
        _require(
            isinstance(policy_server, Mapping)
            and policy_server.get("status") == "available"
            and isinstance(endpoint, Mapping)
            and endpoint.get("host") == "127.0.0.1"
            and isinstance(endpoint.get("port"), int)
            and 20000 <= int(endpoint["port"]) < 50000
            and planner.get("policy_server_identity_sha256")
            == _canonical_sha256(policy_server),
            "fresh policy server identity is absent or unbound",
        )
    else:
        _require(
            policy_server is None
            and planner.get("policy_server_identity_sha256") is None,
            "unused policy server identity differs",
        )
    query_by_index: Dict[int, Mapping[str, Any]] = {}
    for query in policy_queries:
        query_index = int(query.get("query_index", -1))
        _require(query_index not in query_by_index, "policy query index is duplicated")
        chunk = np.asarray(query.get("returned_actions"), dtype=np.float64)
        _require(
            chunk.shape == (int(protocol["execution"]["model_action_horizon"]), 7)
            and np.all(np.isfinite(chunk))
            and query.get("returned_actions_sha256") == _float64_sha256(chunk, np)
            and query.get("returned_action_shape") == list(chunk.shape)
            and query.get("source") == "fresh_pi05_from_treatment_observation",
            "fresh policy response bytes or shape differ",
        )
        query_by_index[query_index] = query
    cached_end = planner.get("cached_current_chunk_end_source_action_index")
    expected_cached = 0
    expected_fresh = 0
    expected_aegis_z = (
        None
        if divergence is None
        else np.asarray(
            historical_value["actions"][int(divergence)]["qp"]["z_after"],
            dtype=np.float64,
        )
    )
    for planner_row in planner_actions:
        source_index = int(planner_row["source_action_index"])
        source_label = planner_row.get("source")
        executed = np.asarray(planner_row.get("executed"), dtype=np.float64)
        action_binding = (
            actions[source_index]
            if source_index < len(actions)
            else partial_action_record
        )
        _require(
            isinstance(action_binding, Mapping)
            and executed.shape == (7,)
            and np.all(np.isfinite(executed))
            and planner_row.get("executed_action_sha256")
            == _float64_sha256(executed, np)
            and np.array_equal(
                executed,
                np.asarray(
                    action_binding.get("executed_high_level_action"),
                    dtype=np.float64,
                ),
            )
            and action_binding.get("executed_high_level_action_sha256")
            == _float64_sha256(executed, np),
            "planner/treatment executed action chain differs",
        )
        expected_observation = (
            None
            if source_index == 0
            else actions[source_index - 1].get("observation_sha256")
        )
        if expected_observation is not None:
            _require(
                planner_row.get("native_observation_sha256") == expected_observation,
                "planner action does not consume the preceding treatment observation",
            )
        if divergence is None or source_index <= int(divergence):
            _require(
                source_label == "archived_aegis_executed_pre_divergence"
                and source_index < len(replay.actions)
                and np.array_equal(
                    np.asarray(planner_row.get("executed"), dtype=np.float64),
                    np.asarray(replay.actions[source_index], dtype=np.float64),
                ),
                "archived planner prefix differs",
            )
        elif source_index <= int(cached_end):
            expected_cached += 1
            archived_raw = np.asarray(
                historical_value["actions"][source_index].get("nominal_raw"),
                dtype=np.float64,
            )
            planner_raw = np.asarray(planner_row.get("nominal_raw"), dtype=np.float64)
            _require(
                source_label
                == "archived_current_pi05_chunk_reprocessed_live_aegis",
                "cached treatment chunk source differs",
            )
            _require(
                archived_raw.shape == planner_raw.shape == (7,)
                and np.array_equal(planner_raw, archived_raw)
                and planner_row.get("nominal_raw_sha256")
                == _float64_sha256(planner_raw, np),
                "cached historical raw chunk differs from the bound result",
            )
        else:
            expected_fresh += 1
            query_index = source_index // 5
            query = query_by_index.get(query_index)
            planner_raw = np.asarray(planner_row.get("nominal_raw"), dtype=np.float64)
            returned_chunk = (
                np.asarray(query.get("returned_actions"), dtype=np.float64)
                if isinstance(query, Mapping)
                else np.asarray([])
            )
            _require(
                source_label == "fresh_pi05_reprocessed_live_aegis"
                and int(planner_row.get("query_index", -1)) == query_index
                and int(planner_row.get("query_chunk_offset", -1))
                == source_index % 5,
                "fresh treatment action provenance differs",
            )
            _require(
                isinstance(query, Mapping)
                and returned_chunk.shape[0] > source_index % 5
                and planner_raw.shape == (7,)
                and np.array_equal(
                    planner_raw, returned_chunk[source_index % 5]
                )
                and planner_row.get("nominal_raw_sha256")
                == _float64_sha256(planner_raw, np),
                "fresh policy response is not bound to the executed action chain",
            )
        if divergence is not None and source_index > int(divergence):
            raw = np.asarray(planner_row.get("nominal_raw"), dtype=np.float64)
            nominal_translational = np.asarray(
                planner_row.get("nominal_translational"), dtype=np.float64
            )
            expected_nominal = np.zeros(7, dtype=np.float64)
            expected_nominal[:3] = raw[:3]
            expected_nominal[6] = raw[6]
            z_before = np.asarray(planner_row.get("aegis_z_before"), dtype=np.float64)
            z_after = np.asarray(planner_row.get("aegis_z_after"), dtype=np.float64)
            aegis_qp = planner_row.get("aegis_qp")
            context = aegis_qp.get("context") if isinstance(aegis_qp, Mapping) else None
            _require(
                nominal_translational.shape == (7,)
                and np.array_equal(nominal_translational, expected_nominal)
                and executed[3] == executed[4] == executed[5] == 0.0
                and executed[6] == raw[6]
                and expected_aegis_z is not None
                and np.array_equal(z_before, expected_aegis_z)
                and z_after.shape == (3,)
                and np.all(np.isfinite(z_after))
                and math.isclose(float(np.linalg.norm(z_after)), 1.0, rel_tol=0.0, abs_tol=1e-10)
                and isinstance(aegis_qp, Mapping)
                and aegis_qp.get("status") == "solved"
                and str(aegis_qp.get("solver_status")) in ("optimal", "optimal_inaccurate")
                and np.array_equal(
                    np.asarray(aegis_qp.get("z_before"), dtype=np.float64), z_before
                )
                and np.array_equal(
                    np.asarray(aegis_qp.get("z_after"), dtype=np.float64), z_after
                )
                and isinstance(context, Mapping)
                and np.array_equal(
                    np.asarray(context.get("executed_action"), dtype=np.float64),
                    executed,
                )
                and context.get("executed_action_array_sha256")
                == _float64_sha256(executed, np),
                "post-divergence AEGIS action or virtual-state chain differs",
            )
            expected_aegis_z = z_after
    _require(
        int(planner.get("cached_current_chunk_action_count", -1)) == expected_cached
        and int(planner.get("fresh_own_observation_action_count", -1))
        == expected_fresh,
        "planner cached/fresh action counts differ",
    )
    expected_query_sources = sorted(
        {
            int(row["source_action_index"])
            for row in planner_actions
            if row.get("source") == "fresh_pi05_reprocessed_live_aegis"
            and int(row["source_action_index"]) % 5 == 0
        }
    )
    _require(
        [int(row.get("source_action_index", -1)) for row in policy_queries]
        == expected_query_sources,
        "fresh policy query schedule differs",
    )
    base_noise_seed = int(case["source_case"]["policy_noise_seed"])
    for query in policy_queries:
        _require(
            divergence is not None
            and int(query["source_action_index"]) > int(divergence),
            "policy query does not use a post-divergence observation",
        )
        _require(
            int(query["source_action_index"]) % 5 == 0
            and int(query["query_index"])
            == int(query["source_action_index"]) // 5
            and int(query["rng_seed"])
            == base_noise_seed + int(query["query_index"]),
            "policy query seed or cadence differs",
        )
        _require(
            query.get("native_observation_sha256")
            == planner_actions[int(query["source_action_index"])].get(
                "native_observation_sha256"
            ),
            "policy query does not bind the planner observation",
        )
    motion = _motion(physics, first_material)
    _close(metrics.get("post_correction_joint_motion_integral_rad"), motion["joint"], "joint motion")
    _close(metrics.get("post_correction_eef_path_length_m"), motion["eef"], "EEF path")
    _close(metrics.get("post_correction_zero_torque_delta_fraction"), motion["zero"], "zero fraction")
    _require(
        len(goals) == len(actions)
        and [int(row.get("source_action_index", -1)) for row in goals]
        == list(range(len(goals))),
        "goal/action terminal cadence differs",
    )
    method_stop = treatment.get("safety_method_stop")
    method_stopped = bool(
        terminal_kind == "safety_method_stop_before_physics"
        and isinstance(method_stop, Mapping)
        and int(method_stop.get("source_action_index", -1)) == len(actions)
        and int(method_stop.get("physics_boundary_before_unexecuted_step", -1))
        == len(physics)
        and str(method_stop.get("reason", "")).endswith("stop_before_physics")
    )
    _require(
        partial_action_audit.get("present")
        is bool(
            terminal_kind == "literal_registered_forbidden_contact"
            or method_stopped
        ),
        "partial action record does not match the terminal condition",
    )
    _require(
        metrics.get("safety_method_stop_before_physics") is method_stopped
        and metrics.get("all_shield_decisions_valid") is (not method_stopped),
        "safety method-stop evidence differs",
    )
    task_success = bool(
        goals
        and goals[-1].get("all_satisfied") is True
        and goals[-1].get("done") is True
        and actions[-1].get("task_success") is True
        and terminal_kind == "native_task_success"
    )
    complete_terminal = bool(
        (
            terminal_kind == "native_task_success"
            and task_success
            and not registered_contact
            and len(physics) == len(actions) * 25
        )
        or (
            terminal_kind == "literal_registered_forbidden_contact"
            and registered_contact
            and 1 <= len(physics) - len(actions) * 25 <= 25
        )
        or (
            terminal_kind == "maximum_action_count"
            and not registered_contact
            and not task_success
            and len(actions) == int(derived["maximum_action_count"])
            and len(physics) == len(actions) * 25
        )
        or (
            method_stopped
            and not registered_contact
            and len(actions) * 25 <= len(physics) < (len(actions) + 1) * 25
        )
    )
    _require(
        metrics.get("complete_terminal_condition_reached") is complete_terminal,
        "terminal condition differs",
    )
    _require(metrics.get("native_task_success") is task_success, "native task flag differs")
    _validate_paper_car_cadence_and_historical_binding(
        car_ledger,
        action_count=len(actions),
        historical_settled_position_sha256=(
            replay.settled_active_obstacle_position_sha256
        ),
    )
    car_key = "%s_pos" % protocol["case"]["selected_obstacle_name"]
    settled_car_position = np.asarray(
        car_ledger[0].get("active_obstacle_position_observation_world_m"),
        dtype=np.float64,
    )
    reconstructed_car = [
        _validate_paper_car_endpoint_row(
            row,
            index=index,
            car_key=car_key,
            obstacle_root_body_id=observable_car_root_id,
            settled_car_position=settled_car_position,
            np=np,
        )
        for index, row in enumerate(car_ledger)
    ]
    maximum_car = max(reconstructed_car)
    _exact_float(
        metrics.get(
            "maximum_active_obstacle_l1_displacement_at_completed_action_endpoints_m"
        ),
        maximum_car,
        "maximum paper CAR displacement",
    )
    expected_car_safe = bool(
        maximum_car
        <= float(protocol["acceptance"]["paper_car_l1_displacement_threshold_m"])
    )
    _require(
        metrics.get("treatment_paper_car_avoided") is expected_car_safe,
        "paper CAR outcome differs",
    )

    video_path = result_path.parent / str(video.get("path"))
    _require(
        not video_path.is_symlink()
        and video_path.is_file()
        and fast._file_sha256(video_path) == video.get("sha256"),
        "video file or hash differs",
    )
    _require(
        video.get("decoded_successfully") is True
        and video.get("real_simulation_frames") is True
        and video.get("two_dimensional_safety_overlay") is False,
        "video contract differs",
    )
    independent = classify_osc_arm_link_canary(dict(metrics), protocol)
    _require(
        not independent.get("feasible")
        or (
            first_divergence_is_material
            and len(independently_validated_qp_audits) == 1
        ),
        "positive feasibility lacks a material independently validated first divergence",
    )
    _require(
        metrics.get("allocation_numeric_prerequisite_verified") is True,
        "producer did not record the independently verified numeric prerequisite",
    )
    _require(
        fast._canonical(independent) == fast._canonical(result.get("classification")),
        "producer classification differs from independent reconstruction",
    )
    return {
        "status": "validated",
        "producer_status": "complete",
        "case_id": result["case_id"],
        "run_id": result["run_id"],
        "result_file_sha256": fast._file_sha256(result_path),
        "classification": independent,
        "action_count": len(actions),
        "physics_substep_count": len(physics),
        "solved_qp_count": solved_qp_count,
        "independently_kkt_validated_solved_qp_count": len(
            independently_validated_qp_audits
        ),
        "independent_kkt_validation_scope": (
            "first_byte_different_torque_row_only"
        ),
        "first_divergence_constraint_attribution": divergence_attribution,
        "partial_action_audit": partial_action_audit,
        "field_sample_static_audit": field_audit,
        "first_material_correction_physical_boundary": first_material,
        "literal_contact_present": measured_contact,
        "native_task_success": task_success,
        "producer_source_commit": expected_producer_commit,
        "producer_slurm_job_id": expected_producer_job_id,
        "numeric_prerequisite": numeric_prerequisite,
        "numeric_slurm_job_id": expected_numeric_job_id,
        "consumer_source_commit": expected_consumer_commit,
        "consumer_slurm_job_id": expected_consumer_job_id,
        "partial_output_interpreted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--historical-result-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/vlsa_poisson_osc_arm_link_canary.v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numeric-validation-result", type=Path, required=True)
    parser.add_argument("--expected-numeric-job-id", required=True)
    parser.add_argument("--expected-producer-job-id", required=True)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--expected-consumer-job-id", required=True)
    parser.add_argument("--expected-consumer-commit", required=True)
    arguments = parser.parse_args()
    from main.poisson_fullbody.contracts import publish_hashed_json
    from main.poisson_fullbody.osc_arm_link_canary import VALIDATION_SCHEMA

    candidate: Dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "rejected",
        "partial_output_interpreted": False,
    }
    try:
        candidate.update(
            validate(
                arguments.result.resolve(),
                arguments.protocol.resolve(),
                arguments.historical_result_root.resolve(),
                arguments.numeric_validation_result,
                expected_numeric_job_id=arguments.expected_numeric_job_id,
                expected_producer_job_id=arguments.expected_producer_job_id,
                expected_producer_commit=arguments.expected_producer_commit,
                expected_consumer_job_id=arguments.expected_consumer_job_id,
                expected_consumer_commit=arguments.expected_consumer_commit,
            )
        )
    except Exception as error:
        candidate["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    publish_hashed_json(arguments.output.resolve(), candidate)
    print(json.dumps({"status": candidate["status"], "output": str(arguments.output)}, sort_keys=True))
    return 0 if candidate["status"] == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
