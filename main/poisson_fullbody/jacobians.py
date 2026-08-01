"""Arbitrary-point Jacobians and source-bound differential audits.

The claim-bearing audit in this module never perturbs live simulator data.  It
copies MuJoCo's complete ``mjSTATE_INTEGRATION`` into a fresh ``MjData`` and
uses tangent-space ``mj_integratePos`` perturbations on that clone.  Its
validator is deliberately pure: it needs only JSON plus independently supplied
sample, DOF, state, and protocol identities.
"""

import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from main.poisson_fullbody.contracts import canonical_json_bytes, sha256_bytes
from main.poisson_fullbody.poisson_field import QueryInvalidReason, TrilinearPoissonField
from main.poisson_fullbody.robot_samples import BodySample


PROTECTED_SAMPLE_DIFFERENTIAL_SCHEMA = (
    "vlsa_poisson_protected_sample_differential_audit.v1"
)
DIFFERENTIAL_AUDIT_HASH_FIELD = "audit_payload_sha256"
_DIFFERENTIAL_CONFIG_KEYS = (
    "state_source",
    "perturbation_integrator",
    "point_jacobian_delta_rad",
    "point_jacobian_absolute_tolerance_m_per_rad",
    "point_jacobian_relative_tolerance",
    "point_jacobian_near_zero_frobenius_m_per_rad",
    "arm_tangent_reconstruction_tolerance_rad_s",
    "nonarm_tangent_leakage_tolerance_rad_s",
    "joint_velocity_directions_rad_s",
    "coupled_eta_ladder_s",
    "coupled_absolute_tolerance_m2_per_s",
    "coupled_relative_tolerance",
    "coupled_near_zero_m2_per_s",
    "same_trilinear_cell_required",
    "required_direction_count_per_sample",
    "stencil_selection_policy",
    "finite_difference_resolution_policy",
    "failure_policy",
)


class DifferentialAuditError(ValueError):
    """Raised when differential evidence is incomplete or inconsistent."""


def _modules() -> Tuple[Any, Any]:
    try:
        import mujoco
        import numpy as np
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("MuJoCo and NumPy are required for point Jacobians") from error
    return mujoco, np


def _raw_model_data(model: Any, data: Any) -> Tuple[Any, Any]:
    return getattr(model, "_model", model), getattr(data, "_data", data)


def point_translational_jacobian(
    model: Any,
    data: Any,
    sample: BodySample,
    arm_dof_indices: Sequence[int],
) -> Tuple[Any, Any]:
    """Return live world point and its 3x7 Panda velocity Jacobian."""

    mujoco, np = _modules()
    indexes = np.asarray(arm_dof_indices, dtype=np.int64)
    if indexes.shape != (7,) or len(set(int(value) for value in indexes)) != 7:
        raise ValueError("arm_dof_indices must contain seven unique DOFs")
    raw_model, raw_data = _raw_model_data(model, data)
    nv = int(raw_model.nv)
    if np.any(indexes < 0) or np.any(indexes >= nv):
        raise ValueError("arm DOF index is outside model.nv")
    point = sample.world_point(data)
    jacobian_position = np.zeros((3, nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, nv), dtype=np.float64)
    mujoco.mj_jac(
        raw_model,
        raw_data,
        jacobian_position,
        jacobian_rotation,
        point,
        int(sample.body_id),
    )
    arm_jacobian = jacobian_position[:, indexes]
    if not np.all(np.isfinite(arm_jacobian)):
        raise ValueError("MuJoCo returned a non-finite point Jacobian")
    return point, arm_jacobian


def evaluate_point_jacobians(
    model: Any,
    data: Any,
    samples: Iterable[BodySample],
    arm_dof_indices: Sequence[int],
) -> Tuple[Any, Any]:
    np = _modules()[1]
    points: List[Any] = []
    jacobians: List[Any] = []
    for sample in samples:
        point, jacobian = point_translational_jacobian(
            model, data, sample, arm_dof_indices
        )
        points.append(point)
        jacobians.append(jacobian)
    if not points:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3, 7), dtype=np.float64),
        )
    return np.stack(points, axis=0), np.stack(jacobians, axis=0)


def finite_difference_point_jacobian(
    model: Any,
    data: Any,
    sample: BodySample,
    arm_qpos_indices: Sequence[int],
    arm_dof_indices: Sequence[int],
    *,
    delta_rad: float = 1e-6,
    absolute_tolerance_m_per_rad: float = 2e-6,
    relative_tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """Central-difference one sample without leaving simulator state changed."""

    mujoco, np = _modules()
    qpos_indexes = np.asarray(arm_qpos_indices, dtype=np.int64)
    if qpos_indexes.shape != (7,) or len(set(int(value) for value in qpos_indexes)) != 7:
        raise ValueError("arm_qpos_indices must contain seven unique indexes")
    if delta_rad <= 0.0 or not np.isfinite(delta_rad):
        raise ValueError("delta_rad must be finite and positive")
    raw_model, raw_data = _raw_model_data(model, data)
    if np.any(qpos_indexes < 0) or np.any(qpos_indexes >= int(raw_model.nq)):
        raise ValueError("arm qpos index is outside model.nq")
    _, analytic = point_translational_jacobian(
        model, data, sample, arm_dof_indices
    )
    original_qpos = np.asarray(raw_data.qpos, dtype=np.float64).copy()
    original_qvel = np.asarray(raw_data.qvel, dtype=np.float64).copy()
    numerical = np.zeros((3, 7), dtype=np.float64)
    try:
        for column, qpos_index in enumerate(qpos_indexes):
            raw_data.qpos[:] = original_qpos
            raw_data.qpos[int(qpos_index)] += float(delta_rad)
            mujoco.mj_forward(raw_model, raw_data)
            plus = sample.world_point(data).copy()

            raw_data.qpos[:] = original_qpos
            raw_data.qpos[int(qpos_index)] -= float(delta_rad)
            mujoco.mj_forward(raw_model, raw_data)
            minus = sample.world_point(data).copy()
            numerical[:, column] = (plus - minus) / (2.0 * float(delta_rad))
    finally:
        raw_data.qpos[:] = original_qpos
        raw_data.qvel[:] = original_qvel
        mujoco.mj_forward(raw_model, raw_data)

    difference = analytic - numerical
    maximum_absolute = float(np.max(np.abs(difference)))
    denominator = max(float(np.linalg.norm(numerical)), 1e-12)
    relative_frobenius = float(np.linalg.norm(difference) / denominator)
    passed = bool(
        maximum_absolute <= float(absolute_tolerance_m_per_rad)
        or relative_frobenius <= float(relative_tolerance)
    )
    return {
        "sample_id": int(sample.sample_id),
        "body_name": sample.body_name,
        "delta_rad": float(delta_rad),
        "analytic": analytic,
        "numerical": numerical,
        "maximum_absolute_error_m_per_rad": maximum_absolute,
        "relative_frobenius_error": relative_frobenius,
        "absolute_tolerance_m_per_rad": float(absolute_tolerance_m_per_rad),
        "relative_tolerance": float(relative_tolerance),
        "passed": passed,
    }


def directional_field_derivative(
    gradient_world: Sequence[float],
    point_jacobian: Any,
    joint_velocity: Sequence[float],
) -> float:
    """Compute grad(h)^T J qdot in physical units."""

    np = _modules()[1]
    gradient = np.asarray(gradient_world, dtype=np.float64)
    jacobian = np.asarray(point_jacobian, dtype=np.float64)
    velocity = np.asarray(joint_velocity, dtype=np.float64)
    if gradient.shape != (3,) or jacobian.shape != (3, 7) or velocity.shape != (7,):
        raise ValueError("gradient, Jacobian, and velocity shapes must be (3,), (3,7), (7,)")
    if not (
        np.all(np.isfinite(gradient))
        and np.all(np.isfinite(jacobian))
        and np.all(np.isfinite(velocity))
    ):
        raise ValueError("directional derivative inputs must be finite")
    return float(gradient @ jacobian @ velocity)


def _json_copy(value: Any) -> Any:
    """Return a strict JSON-native copy (and reject NaN/Inf)."""

    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DifferentialAuditError("%s must be a finite number" % label)
    result = float(value)
    if not math.isfinite(result):
        raise DifferentialAuditError("%s must be a finite number" % label)
    return result


def _positive_number(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result <= 0.0:
        raise DifferentialAuditError("%s must be strictly positive" % label)
    return result


def _parse_differential_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, Mapping) or set(config) != set(_DIFFERENTIAL_CONFIG_KEYS):
        raise DifferentialAuditError(
            "differential_audit_config must contain exactly the registered keys"
        )
    output = _json_copy(dict(config))
    literals = {
        "state_source": "settled_mujoco_mjstate_integration_clone",
        "perturbation_integrator": "mujoco_mj_integratePos_full_nv_tangent",
        "stencil_selection_policy": (
            "largest_eta_with_valid_base_plus_minus_in_same_exact_cell"
        ),
        "finite_difference_resolution_policy": (
            "fail_on_no_certified_stencil_or_detected_cancellation"
        ),
        "failure_policy": "fail_before_active_physics_retain_artifact",
    }
    for key, expected in literals.items():
        if output[key] != expected:
            raise DifferentialAuditError("differential_audit_config.%s is invalid" % key)
    for key in (
        "point_jacobian_delta_rad",
        "point_jacobian_absolute_tolerance_m_per_rad",
        "point_jacobian_relative_tolerance",
        "point_jacobian_near_zero_frobenius_m_per_rad",
        "arm_tangent_reconstruction_tolerance_rad_s",
        "nonarm_tangent_leakage_tolerance_rad_s",
        "coupled_absolute_tolerance_m2_per_s",
        "coupled_relative_tolerance",
        "coupled_near_zero_m2_per_s",
    ):
        output[key] = _positive_number(output[key], "differential_audit_config.%s" % key)
    if output["same_trilinear_cell_required"] is not True:
        raise DifferentialAuditError("same_trilinear_cell_required must be true")
    directions = output["joint_velocity_directions_rad_s"]
    if not isinstance(directions, list) or len(directions) != 9:
        raise DifferentialAuditError("exactly nine joint-velocity directions are required")
    parsed_directions = []
    for index, direction in enumerate(directions):
        if not isinstance(direction, list) or len(direction) != 7:
            raise DifferentialAuditError("direction %d must have seven entries" % index)
        parsed_directions.append(
            [_finite_number(value, "direction[%d]" % index) for value in direction]
        )
    output["joint_velocity_directions_rad_s"] = parsed_directions
    eta = output["coupled_eta_ladder_s"]
    if not isinstance(eta, list) or not eta:
        raise DifferentialAuditError("coupled_eta_ladder_s must be non-empty")
    output["coupled_eta_ladder_s"] = [
        _positive_number(value, "coupled_eta_ladder_s") for value in eta
    ]
    if any(
        output["coupled_eta_ladder_s"][index]
        <= output["coupled_eta_ladder_s"][index + 1]
        for index in range(len(output["coupled_eta_ladder_s"]) - 1)
    ):
        raise DifferentialAuditError("coupled eta ladder must be strictly decreasing")
    count = output["required_direction_count_per_sample"]
    if isinstance(count, bool) or not isinstance(count, int) or count != len(directions):
        raise DifferentialAuditError("all configured directions must be required")
    return output


def _sample_identity(value: Any) -> Dict[str, Any]:
    record = value.to_dict() if isinstance(value, BodySample) else value
    expected_keys = {
        "sample_id",
        "body_id",
        "body_name",
        "geom_id",
        "geom_name",
        "point_body_local_m",
        "source",
    }
    if not isinstance(record, Mapping) or set(record) != expected_keys:
        raise DifferentialAuditError("sample identity has invalid keys")
    output = _json_copy(dict(record))
    for key in ("sample_id", "body_id", "geom_id"):
        if isinstance(output[key], bool) or not isinstance(output[key], int) or output[key] < 0:
            raise DifferentialAuditError("sample identity %s is invalid" % key)
    for key in ("body_name", "geom_name", "source"):
        if not isinstance(output[key], str) or not output[key]:
            raise DifferentialAuditError("sample identity %s is invalid" % key)
    point = output["point_body_local_m"]
    if not isinstance(point, list) or len(point) != 3:
        raise DifferentialAuditError("sample local point must have length three")
    output["point_body_local_m"] = [
        _finite_number(item, "sample local point") for item in point
    ]
    return output


def _sample_identities(samples: Iterable[Any]) -> List[Dict[str, Any]]:
    result = [_sample_identity(sample) for sample in samples]
    if not result:
        raise DifferentialAuditError("at least one protected sample is required")
    identifiers = [sample["sample_id"] for sample in result]
    if len(identifiers) != len(set(identifiers)):
        raise DifferentialAuditError("protected sample IDs must be unique")
    return result


def _arm_dofs(value: Sequence[int], nv: Optional[int] = None) -> List[int]:
    if isinstance(value, (str, bytes)):
        raise DifferentialAuditError("arm_dof_indices must be a sequence")
    result = list(value)
    if len(result) != 7 or any(
        isinstance(item, bool) or not isinstance(item, int) for item in result
    ):
        raise DifferentialAuditError("arm_dof_indices must contain seven integers")
    if len(set(result)) != 7 or any(item < 0 for item in result):
        raise DifferentialAuditError("arm_dof_indices must be unique and nonnegative")
    if nv is not None and any(item >= nv for item in result):
        raise DifferentialAuditError("arm DOF index is outside model.nv")
    return result


def _state_vector(mujoco: Any, np: Any, model: Any, data: Any) -> Any:
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(int(mujoco.mj_stateSize(model, specification)), dtype=np.float64)
    mujoco.mj_getState(model, data, state, specification)
    if not np.all(np.isfinite(state)):
        raise DifferentialAuditError("MuJoCo integration state is non-finite")
    return state


def _array_sha256(np: Any, value: Any) -> str:
    array = np.ascontiguousarray(value)
    header = canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    digest = hashlib.sha256()
    digest.update(b"vlsa-table1-array-v1\0")
    digest.update(header)
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def mujoco_integration_state_sha256(model: Any, data: Any) -> str:
    """Hash MuJoCo's complete official integration state as little-endian f64."""

    mujoco, np = _modules()
    raw_model, raw_data = _raw_model_data(model, data)
    return _array_sha256(np, _state_vector(mujoco, np, raw_model, raw_data))


def _matrix(value: Any, rows: int, columns: int, label: str) -> List[List[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise DifferentialAuditError("%s must have %d rows" % (label, rows))
    result = []
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise DifferentialAuditError("%s has invalid columns" % label)
        result.append([_finite_number(item, label) for item in row])
    return result


def _vector(value: Any, length: int, label: str) -> List[float]:
    if not isinstance(value, list) or len(value) != length:
        raise DifferentialAuditError("%s must have length %d" % (label, length))
    return [_finite_number(item, label) for item in value]


def _frobenius(matrix: Sequence[Sequence[float]]) -> float:
    return math.sqrt(sum(value * value for row in matrix for value in row))


def _point_metrics(
    analytic: Sequence[Sequence[float]],
    numerical: Sequence[Sequence[float]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    error = [
        [float(analytic[row][column] - numerical[row][column]) for column in range(7)]
        for row in range(3)
    ]
    maximum = max(abs(value) for row in error for value in row)
    error_norm = _frobenius(error)
    numerical_norm = _frobenius(numerical)
    near_zero = numerical_norm <= config["point_jacobian_near_zero_frobenius_m_per_rad"]
    relative = None if near_zero else error_norm / numerical_norm
    absolute_passed = maximum <= config[
        "point_jacobian_absolute_tolerance_m_per_rad"
    ]
    relative_passed = bool(
        near_zero or relative <= config["point_jacobian_relative_tolerance"]
    )
    return {
        "error_m_per_rad_3x7": error,
        "maximum_absolute_error_m_per_rad": maximum,
        "error_frobenius_m_per_rad": error_norm,
        "numerical_frobenius_m_per_rad": numerical_norm,
        "relative_error": relative,
        "relative_criterion_waived_for_near_zero_numeric_frobenius": near_zero,
        "absolute_tolerance_m_per_rad": config[
            "point_jacobian_absolute_tolerance_m_per_rad"
        ],
        "relative_tolerance": config["point_jacobian_relative_tolerance"],
        "near_zero_frobenius_m_per_rad": config[
            "point_jacobian_near_zero_frobenius_m_per_rad"
        ],
        "absolute_passed": absolute_passed,
        "relative_passed": relative_passed,
        "passed": bool(absolute_passed and relative_passed),
    }


def _tangent_reconstruction(
    requested: Sequence[float],
    plus_reconstructed: Sequence[float],
    minus_reconstructed: Sequence[float],
    arm_dofs: Sequence[int],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    plus_error = [
        float(observed - expected)
        for observed, expected in zip(plus_reconstructed, requested)
    ]
    minus_error = [
        float(observed + expected)
        for observed, expected in zip(minus_reconstructed, requested)
    ]
    arm = set(arm_dofs)
    arm_max = max(
        [abs(plus_error[index]) for index in arm_dofs]
        + [abs(minus_error[index]) for index in arm_dofs]
    )
    nonarm_indices = [index for index in range(len(requested)) if index not in arm]
    nonarm_max = max(
        [0.0]
        + [abs(plus_reconstructed[index]) for index in nonarm_indices]
        + [abs(minus_reconstructed[index]) for index in nonarm_indices]
    )
    arm_tolerance = config["arm_tangent_reconstruction_tolerance_rad_s"]
    nonarm_tolerance = config["nonarm_tangent_leakage_tolerance_rad_s"]
    return {
        "requested_tangent_nv_rad_s": [float(value) for value in requested],
        "plus_reconstructed_tangent_nv_rad_s": [
            float(value) for value in plus_reconstructed
        ],
        "minus_reconstructed_tangent_nv_rad_s": [
            float(value) for value in minus_reconstructed
        ],
        "plus_error_nv_rad_s": plus_error,
        "minus_error_nv_rad_s": minus_error,
        "maximum_arm_reconstruction_error_rad_s": arm_max,
        "maximum_nonarm_leakage_rad_s": nonarm_max,
        "arm_tolerance_rad_s": arm_tolerance,
        "nonarm_tolerance_rad_s": nonarm_tolerance,
        "arm_passed": bool(arm_max <= arm_tolerance),
        "nonarm_passed": bool(nonarm_max <= nonarm_tolerance),
        "passed": bool(arm_max <= arm_tolerance and nonarm_max <= nonarm_tolerance),
    }


def _query_record(query: Any, point: Sequence[float]) -> Dict[str, Any]:
    reason = query.reason.value if isinstance(query.reason, QueryInvalidReason) else query.reason
    return {
        "point_world_m": [float(value) for value in point],
        "valid": bool(query.valid),
        "value_m2": None if query.value is None else float(query.value),
        "gradient_m": (
            None if query.gradient is None else [float(value) for value in query.gradient]
        ),
        "reason": reason,
        "cell_index": (
            None if query.cell_index is None else [int(value) for value in query.cell_index]
        ),
        "local_coordinates": (
            None
            if query.local_coordinates is None
            else [float(value) for value in query.local_coordinates]
        ),
        "outer_boundary_clearance_m": (
            None
            if query.outer_boundary_clearance_m is None
            else float(query.outer_boundary_clearance_m)
        ),
    }


def _noneligible_reasons(
    base: Mapping[str, Any], plus: Mapping[str, Any], minus: Mapping[str, Any]
) -> List[str]:
    reasons = []
    for label, query in (("base", base), ("plus", plus), ("minus", minus)):
        if not query["valid"]:
            reasons.append("%s_query_invalid:%s" % (label, query["reason"]))
    if base["valid"] and plus["valid"] and base["cell_index"] != plus["cell_index"]:
        reasons.append("plus_cell_differs_from_base")
    if base["valid"] and minus["valid"] and base["cell_index"] != minus["cell_index"]:
        reasons.append("minus_cell_differs_from_base")
    return reasons


def _batched_perturb(
    mujoco: Any,
    np: Any,
    model: Any,
    clone: Any,
    base_state: Any,
    base_qpos: Any,
    tangent: Any,
    step: float,
    samples: Sequence[BodySample],
    arm_dofs: Sequence[int],
    config: Mapping[str, Any],
) -> Tuple[List[List[float]], List[List[float]], Dict[str, Any]]:
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    points = []
    reconstructed = []
    for sign in (1.0, -1.0):
        mujoco.mj_setState(model, clone, base_state, specification)
        mujoco.mj_integratePos(model, clone.qpos, tangent, sign * float(step))
        perturbed_qpos = np.asarray(clone.qpos, dtype=np.float64).copy()
        velocity = np.zeros(int(model.nv), dtype=np.float64)
        mujoco.mj_differentiatePos(
            model, velocity, float(step), base_qpos, perturbed_qpos
        )
        reconstructed.append(velocity)
        mujoco.mj_forward(model, clone)
        points.append(
            [
                [float(value) for value in sample.world_point(clone)]
                for sample in samples
            ]
        )
    reconstruction = _tangent_reconstruction(
        tangent.tolist(), reconstructed[0].tolist(), reconstructed[1].tolist(), arm_dofs, config
    )
    return points[0], points[1], reconstruction


def _classification_ledger(sample_records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    ledger = []
    for sample_index, sample in enumerate(sample_records):
        directions = []
        for direction in sample["coupled_directions"]:
            selected_index = direction["selected_attempt_index"]
            selected = (
                None if selected_index is None else direction["attempts"][selected_index]
            )
            comparison = direction["derived_comparison"]
            directions.append(
                {
                    "direction_index": direction["direction_index"],
                    "selected_attempt_index": selected_index,
                    "selected_eta_s": None if selected is None else selected["eta_s"],
                    "selected_cell_index": (
                        None
                        if selected is None
                        else selected["plus_query"]["cell_index"]
                    ),
                    "selected_base_cell_index": (
                        None
                        if selected is None
                        else sample["base_field_query"]["cell_index"]
                    ),
                    "selected_plus_cell_index": (
                        None if selected is None else selected["plus_query"]["cell_index"]
                    ),
                    "selected_minus_cell_index": (
                        None if selected is None else selected["minus_query"]["cell_index"]
                    ),
                    "attempt_classification": [
                        {
                            "eta_s": attempt["eta_s"],
                            "base_valid": sample["base_field_query"]["valid"],
                            "base_reason": sample["base_field_query"]["reason"],
                            "base_cell_index": sample["base_field_query"]["cell_index"],
                            "plus_valid": attempt["plus_query"]["valid"],
                            "plus_reason": attempt["plus_query"]["reason"],
                            "plus_cell_index": attempt["plus_query"]["cell_index"],
                            "minus_valid": attempt["minus_query"]["valid"],
                            "minus_reason": attempt["minus_query"]["reason"],
                            "minus_cell_index": attempt["minus_query"]["cell_index"],
                            "eligible_same_cell_stencil": attempt[
                                "eligible_same_cell_stencil"
                            ],
                            "noneligible_reasons": attempt["noneligible_reasons"],
                            "tangent_reconstruction_passed": attempt[
                                "tangent_reconstruction"
                            ]["passed"],
                            "selected": attempt["selected"],
                        }
                        for attempt in direction["attempts"]
                    ],
                    "selected_tangent_reconstruction_passed": (
                        None
                        if selected is None
                        else selected["tangent_reconstruction"]["passed"]
                    ),
                    "central_numerator_cancellation_detected": (
                        None
                        if comparison is None
                        else comparison["central_numerator_cancellation_detected"]
                    ),
                    "passed": direction["passed"],
                }
            )
        ledger.append(
            {
                "sample_index": sample_index,
                "sample_id": sample["identity"]["sample_id"],
                "point_jacobian_passed": sample["point_jacobian_passed"],
                "directions": directions,
                "passed": sample["passed"],
            }
        )
    return ledger


def _stable_audit_hashes(
    state_sha256: str,
    arm_dofs: Sequence[int],
    identities: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    sample_records: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    specification_sha256 = sha256_bytes(canonical_json_bytes(config))
    identity_sha256 = sha256_bytes(canonical_json_bytes(identities))
    binding = {
        "schema_version": PROTECTED_SAMPLE_DIFFERENTIAL_SCHEMA,
        "integration_state_sha256": state_sha256,
        "arm_dof_indices": list(arm_dofs),
        "ordered_sample_identity_sha256": identity_sha256,
        "specification_sha256": specification_sha256,
    }
    return {
        "specification_sha256": specification_sha256,
        "binding_sha256": sha256_bytes(canonical_json_bytes(binding)),
        "classification_ledger_sha256": sha256_bytes(
            canonical_json_bytes(_classification_ledger(sample_records))
        ),
    }


def audit_protected_sample_differentials(
    model: Any,
    data: Any,
    samples: Iterable[BodySample],
    arm_dof_indices: Sequence[int],
    field: TrilinearPoissonField,
    differential_audit_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build exhaustive point-Jacobian and coupled field evidence.

    All samples are evaluated from each shared perturbation, so cost is a
    function of directions/eta attempts rather than sample count.
    """

    mujoco, np = _modules()
    raw_model, raw_data = _raw_model_data(model, data)
    config = _parse_differential_config(differential_audit_config)
    records = list(samples)
    identities = _sample_identities(records)
    if any(not isinstance(sample, BodySample) for sample in records):
        raise DifferentialAuditError("samples must contain BodySample records")
    dofs = _arm_dofs(arm_dof_indices, int(raw_model.nv))
    if not isinstance(field, TrilinearPoissonField):
        raise DifferentialAuditError("field must be a TrilinearPoissonField")

    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    source_before = _state_vector(mujoco, np, raw_model, raw_data)
    source_hash = _array_sha256(np, source_before)
    clone = mujoco.MjData(raw_model)
    mujoco.mj_setState(raw_model, clone, source_before, specification)
    clone_base_hash = _array_sha256(
        np, _state_vector(mujoco, np, raw_model, clone)
    )
    mujoco.mj_forward(raw_model, clone)
    base_qpos = np.asarray(clone.qpos, dtype=np.float64).copy()
    base_points = [
        [float(value) for value in sample.world_point(clone)] for sample in records
    ]
    analytic = []
    for sample in records:
        _, jacobian = point_translational_jacobian(raw_model, clone, sample, dofs)
        analytic.append(
            [[float(value) for value in row] for row in np.asarray(jacobian)]
        )
    base_queries = [
        _query_record(field.query(point), point) for point in base_points
    ]

    point_plus = [[None for _ in range(7)] for _ in records]
    point_minus = [[None for _ in range(7)] for _ in records]
    point_reconstruction = []
    delta = config["point_jacobian_delta_rad"]
    for column, dof in enumerate(dofs):
        tangent = np.zeros(int(raw_model.nv), dtype=np.float64)
        tangent[dof] = 1.0
        plus, minus, reconstruction = _batched_perturb(
            mujoco,
            np,
            raw_model,
            clone,
            source_before,
            base_qpos,
            tangent,
            delta,
            records,
            dofs,
            config,
        )
        point_reconstruction.append(reconstruction)
        for sample_index in range(len(records)):
            point_plus[sample_index][column] = plus[sample_index]
            point_minus[sample_index][column] = minus[sample_index]

    numerical = []
    point_metrics = []
    for sample_index in range(len(records)):
        matrix = [
            [
                float(
                    (point_plus[sample_index][column][axis]
                    - point_minus[sample_index][column][axis])
                    / (2.0 * delta)
                )
                for column in range(7)
            ]
            for axis in range(3)
        ]
        numerical.append(matrix)
        point_metrics.append(_point_metrics(analytic[sample_index], matrix, config))

    # Attempts are appended sample-by-sample, direction-by-direction while
    # perturbations themselves remain batched across all samples.
    attempts = [
        [[] for _ in config["joint_velocity_directions_rad_s"]] for _ in records
    ]
    selected = [[False for _ in config["joint_velocity_directions_rad_s"]] for _ in records]
    for direction_index, arm_direction in enumerate(
        config["joint_velocity_directions_rad_s"]
    ):
        tangent = np.zeros(int(raw_model.nv), dtype=np.float64)
        tangent[np.asarray(dofs, dtype=np.int64)] = np.asarray(
            arm_direction, dtype=np.float64
        )
        for eta in config["coupled_eta_ladder_s"]:
            if all(
                selected[sample_index][direction_index]
                for sample_index in range(len(records))
            ):
                break
            plus, minus, reconstruction = _batched_perturb(
                mujoco,
                np,
                raw_model,
                clone,
                source_before,
                base_qpos,
                tangent,
                eta,
                records,
                dofs,
                config,
            )
            for sample_index in range(len(records)):
                if selected[sample_index][direction_index]:
                    continue
                plus_query = _query_record(field.query(plus[sample_index]), plus[sample_index])
                minus_query = _query_record(
                    field.query(minus[sample_index]), minus[sample_index]
                )
                reasons = _noneligible_reasons(
                    base_queries[sample_index], plus_query, minus_query
                )
                eligible = not reasons
                attempt = {
                    "eta_s": float(eta),
                    "plus_point_world_m": plus[sample_index],
                    "minus_point_world_m": minus[sample_index],
                    "plus_query": plus_query,
                    "minus_query": minus_query,
                    "tangent_reconstruction": reconstruction,
                    "eligible_same_cell_stencil": bool(eligible),
                    "noneligible_reasons": reasons,
                    "selected": bool(eligible),
                }
                attempts[sample_index][direction_index].append(attempt)
                if eligible:
                    selected[sample_index][direction_index] = True

    sample_records = []
    total_attempts = 0
    failed_attempts = 0
    selected_directions = 0
    passed_directions = 0
    for sample_index, identity in enumerate(identities):
        directions = []
        gradient = base_queries[sample_index]["gradient_m"]
        for direction_index, arm_direction in enumerate(
            config["joint_velocity_directions_rad_s"]
        ):
            direction_attempts = attempts[sample_index][direction_index]
            total_attempts += len(direction_attempts)
            failed_attempts += sum(
                not attempt["eligible_same_cell_stencil"]
                for attempt in direction_attempts
            )
            chosen = next(
                (attempt for attempt in direction_attempts if attempt["selected"]), None
            )
            derived = None
            passed = False
            if chosen is not None:
                selected_directions += 1
                eta = chosen["eta_s"]
                point_fd = [
                    float((plus - minus) / (2.0 * eta))
                    for plus, minus in zip(
                        chosen["plus_point_world_m"], chosen["minus_point_world_m"]
                    )
                ]
                jv = [
                    float(
                        sum(
                            analytic[sample_index][axis][column]
                            * arm_direction[column]
                            for column in range(7)
                        )
                    )
                    for axis in range(3)
                ]
                point_error = [float(left - right) for left, right in zip(point_fd, jv)]
                point_error_max = max(abs(value) for value in point_error)
                point_fd_norm = math.sqrt(sum(value * value for value in point_fd))
                point_direction_near_zero = bool(
                    point_fd_norm
                    <= config["point_jacobian_near_zero_frobenius_m_per_rad"]
                )
                point_direction_relative = (
                    None
                    if point_direction_near_zero
                    else math.sqrt(sum(value * value for value in point_error))
                    / point_fd_norm
                )
                point_direction_absolute_pass = bool(
                    point_error_max
                    <= config["point_jacobian_absolute_tolerance_m_per_rad"]
                )
                point_direction_relative_pass = bool(
                    point_direction_near_zero
                    or point_direction_relative
                    <= config["point_jacobian_relative_tolerance"]
                )
                h_fd = float(
                    (chosen["plus_query"]["value_m2"] - chosen["minus_query"]["value_m2"])
                    / (2.0 * eta)
                )
                grad_point_fd = float(sum(g * v for g, v in zip(gradient, point_fd)))
                grad_jv = float(sum(g * v for g, v in zip(gradient, jv)))
                error = abs(h_fd - grad_point_fd)
                scale = abs(h_fd)
                near_zero = scale <= config["coupled_near_zero_m2_per_s"]
                relative = None if near_zero else error / scale
                absolute_pass = error <= config["coupled_absolute_tolerance_m2_per_s"]
                relative_pass = bool(
                    near_zero or relative <= config["coupled_relative_tolerance"]
                )
                point_error_norm = math.sqrt(sum(value * value for value in point_error))
                gradient_norm = math.sqrt(sum(value * value for value in gradient))
                direct_error = abs(h_fd - grad_jv)
                direct_allowance = float(
                    config["coupled_absolute_tolerance_m2_per_s"]
                    + gradient_norm * point_error_norm
                )
                cancellation = bool(
                    chosen["plus_query"]["value_m2"]
                    == chosen["minus_query"]["value_m2"]
                    and abs(grad_point_fd) > config["coupled_near_zero_m2_per_s"]
                )
                passed = bool(
                    chosen["tangent_reconstruction"]["passed"]
                    and point_direction_absolute_pass
                    and point_direction_relative_pass
                    and absolute_pass
                    and relative_pass
                    and direct_error <= direct_allowance
                    and not cancellation
                )
                derived = {
                    "selected_eta_s": eta,
                    "point_fd_velocity_m_s": point_fd,
                    "analytic_jacobian_velocity_m_s": jv,
                    "point_velocity_error_m_s": point_error,
                    "point_velocity_error_norm_m_s": point_error_norm,
                    "point_velocity_maximum_absolute_error_m_s": point_error_max,
                    "point_velocity_numeric_norm_m_s": point_fd_norm,
                    "point_velocity_relative_error": point_direction_relative,
                    "point_velocity_relative_criterion_waived_for_near_zero_numeric_norm": point_direction_near_zero,
                    "point_velocity_absolute_tolerance_m_s": config[
                        "point_jacobian_absolute_tolerance_m_per_rad"
                    ],
                    "point_velocity_relative_tolerance": config[
                        "point_jacobian_relative_tolerance"
                    ],
                    "point_velocity_near_zero_norm_m_s": config[
                        "point_jacobian_near_zero_frobenius_m_per_rad"
                    ],
                    "point_velocity_absolute_passed": point_direction_absolute_pass,
                    "point_velocity_relative_passed": point_direction_relative_pass,
                    "field_central_derivative_m2_per_s": h_fd,
                    "gradient_dot_point_fd_m2_per_s": grad_point_fd,
                    "gradient_dot_analytic_jacobian_m2_per_s": grad_jv,
                    "field_vs_point_fd_absolute_error_m2_per_s": error,
                    "field_vs_point_fd_relative_error": relative,
                    "relative_criterion_waived_for_near_zero_numeric_derivative": near_zero,
                    "absolute_tolerance_m2_per_s": config[
                        "coupled_absolute_tolerance_m2_per_s"
                    ],
                    "relative_tolerance": config["coupled_relative_tolerance"],
                    "near_zero_m2_per_s": config["coupled_near_zero_m2_per_s"],
                    "absolute_passed": absolute_pass,
                    "relative_passed": relative_pass,
                    "direct_analytic_absolute_error_m2_per_s": direct_error,
                    "direct_analytic_allowance_m2_per_s": direct_allowance,
                    "direct_analytic_passed": bool(direct_error <= direct_allowance),
                    "central_numerator_cancellation_detected": cancellation,
                }
            if passed:
                passed_directions += 1
            directions.append(
                {
                    "direction_index": direction_index,
                    "arm_joint_velocity_rad_s": list(arm_direction),
                    "attempts": direction_attempts,
                    "selected_attempt_index": (
                        None if chosen is None else len(direction_attempts) - 1
                    ),
                    "derived_comparison": derived,
                    "passed": passed,
                }
            )
        point_pass = bool(
            point_metrics[sample_index]["passed"]
            and all(item["passed"] for item in point_reconstruction)
        )
        coupled_pass = bool(
            len(directions) == config["required_direction_count_per_sample"]
            and all(direction["passed"] for direction in directions)
        )
        sample_records.append(
            {
                "identity": identity,
                "base_point_world_m": base_points[sample_index],
                "base_field_query": base_queries[sample_index],
                "analytic_point_jacobian_m_per_rad_3x7": analytic[sample_index],
                "plus_points_world_m_by_arm_dof": point_plus[sample_index],
                "minus_points_world_m_by_arm_dof": point_minus[sample_index],
                "numerical_point_jacobian_m_per_rad_3x7": numerical[sample_index],
                "point_jacobian_metrics": point_metrics[sample_index],
                "point_perturbation_tangent_reconstruction_by_arm_dof": point_reconstruction,
                "point_jacobian_passed": point_pass,
                "coupled_directions": directions,
                "eligible_coupled_direction_count": sum(
                    direction["selected_attempt_index"] is not None
                    for direction in directions
                ),
                "coupled_directions_passed": coupled_pass,
                "passed": bool(point_pass and coupled_pass),
            }
        )

    mujoco.mj_setState(raw_model, clone, source_before, specification)
    clone_final = _state_vector(mujoco, np, raw_model, clone)
    source_after = _state_vector(mujoco, np, raw_model, raw_data)
    source_final_hash = _array_sha256(np, source_after)
    clone_final_hash = _array_sha256(np, clone_final)
    state_ok = bool(
        source_hash == clone_base_hash == clone_final_hash == source_final_hash
    )
    counts = {
        "sample_count": len(records),
        "point_jacobian_passed_sample_count": sum(
            record["point_jacobian_passed"] for record in sample_records
        ),
        "required_coupled_direction_count": len(records)
        * config["required_direction_count_per_sample"],
        "selected_coupled_direction_count": selected_directions,
        "passed_coupled_direction_count": passed_directions,
        "coupled_attempt_count": total_attempts,
        "noneligible_coupled_attempt_count": failed_attempts,
        "passed_sample_count": sum(record["passed"] for record in sample_records),
    }
    stable_hashes = _stable_audit_hashes(
        source_hash, dofs, identities, config, sample_records
    )
    payload = {
        "schema_version": PROTECTED_SAMPLE_DIFFERENTIAL_SCHEMA,
        "integration_state": {
            "state_specification": "mjSTATE_INTEGRATION",
            "element_count": int(source_before.size),
            "source_initial_sha256": source_hash,
            "clone_base_sha256": clone_base_hash,
            "clone_final_sha256": clone_final_hash,
            "source_final_sha256": source_final_hash,
            "source_state_unchanged": bool(source_hash == source_final_hash),
            "clone_state_restored": bool(source_hash == clone_base_hash == clone_final_hash),
        },
        "arm_dof_indices": dofs,
        "ordered_samples": identities,
        "ordered_sample_identity_sha256": sha256_bytes(canonical_json_bytes(identities)),
        "specification_sha256": stable_hashes["specification_sha256"],
        "binding_sha256": stable_hashes["binding_sha256"],
        "classification_ledger_sha256": stable_hashes[
            "classification_ledger_sha256"
        ],
        "differential_audit_config": config,
        "sample_records": sample_records,
        "counts": counts,
        "passed": bool(state_ok and all(record["passed"] for record in sample_records)),
    }
    payload[DIFFERENTIAL_AUDIT_HASH_FIELD] = sha256_bytes(canonical_json_bytes(payload))
    return _json_copy(payload)


def _exact_keys(value: Any, keys: Sequence[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise DifferentialAuditError("%s has invalid keys" % label)
    return value


def _validate_query_record(value: Any, point: Sequence[float], label: str) -> Dict[str, Any]:
    query = _exact_keys(
        value,
        (
            "point_world_m",
            "valid",
            "value_m2",
            "gradient_m",
            "reason",
            "cell_index",
            "local_coordinates",
            "outer_boundary_clearance_m",
        ),
        label,
    )
    observed_point = _vector(query["point_world_m"], 3, label + ".point")
    if observed_point != list(point):
        raise DifferentialAuditError("%s is bound to a different point" % label)
    if not isinstance(query["valid"], bool):
        raise DifferentialAuditError("%s.valid must be boolean" % label)
    if query["valid"]:
        value_m2 = _finite_number(query["value_m2"], label + ".value_m2")
        gradient = _vector(query["gradient_m"], 3, label + ".gradient")
        cell = query["cell_index"]
        if (
            not isinstance(cell, list)
            or len(cell) != 3
            or any(isinstance(item, bool) or not isinstance(item, int) for item in cell)
        ):
            raise DifferentialAuditError("%s.cell_index is invalid" % label)
        local = _vector(query["local_coordinates"], 3, label + ".local")
        clearance = _finite_number(
            query["outer_boundary_clearance_m"], label + ".clearance"
        )
        if query["reason"] is not None:
            raise DifferentialAuditError("valid query cannot have an invalid reason")
        return {
            "point_world_m": observed_point,
            "valid": True,
            "value_m2": value_m2,
            "gradient_m": gradient,
            "reason": None,
            "cell_index": list(cell),
            "local_coordinates": local,
            "outer_boundary_clearance_m": clearance,
        }
    allowed = {reason.value for reason in QueryInvalidReason}
    if query["reason"] not in allowed:
        raise DifferentialAuditError("invalid query has an unknown typed reason")
    if any(
        query[key] is not None
        for key in (
            "value_m2",
            "gradient_m",
            "cell_index",
            "local_coordinates",
            "outer_boundary_clearance_m",
        )
    ):
        raise DifferentialAuditError("invalid query must not contain field values")
    return {
        "point_world_m": observed_point,
        "valid": False,
        "value_m2": None,
        "gradient_m": None,
        "reason": query["reason"],
        "cell_index": None,
        "local_coordinates": None,
        "outer_boundary_clearance_m": None,
    }


def _validate_tangent_record(
    value: Any,
    expected_tangent: Sequence[float],
    arm_dofs: Sequence[int],
    config: Mapping[str, Any],
    label: str,
) -> Dict[str, Any]:
    keys = (
        "requested_tangent_nv_rad_s",
        "plus_reconstructed_tangent_nv_rad_s",
        "minus_reconstructed_tangent_nv_rad_s",
        "plus_error_nv_rad_s",
        "minus_error_nv_rad_s",
        "maximum_arm_reconstruction_error_rad_s",
        "maximum_nonarm_leakage_rad_s",
        "arm_tolerance_rad_s",
        "nonarm_tolerance_rad_s",
        "arm_passed",
        "nonarm_passed",
        "passed",
    )
    record = _exact_keys(value, keys, label)
    nv = len(expected_tangent)
    requested = _vector(record["requested_tangent_nv_rad_s"], nv, label)
    plus = _vector(record["plus_reconstructed_tangent_nv_rad_s"], nv, label)
    minus = _vector(record["minus_reconstructed_tangent_nv_rad_s"], nv, label)
    if requested != list(expected_tangent):
        raise DifferentialAuditError("%s requested a different tangent" % label)
    reconstructed = _tangent_reconstruction(requested, plus, minus, arm_dofs, config)
    if _json_copy(record) != reconstructed:
        raise DifferentialAuditError("%s tangent diagnostics do not reconstruct" % label)
    return reconstructed


def _coupled_comparison(
    chosen: Mapping[str, Any],
    analytic: Sequence[Sequence[float]],
    gradient: Sequence[float],
    arm_direction: Sequence[float],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    eta = float(chosen["eta_s"])
    point_fd = [
        float((plus - minus) / (2.0 * eta))
        for plus, minus in zip(
            chosen["plus_point_world_m"], chosen["minus_point_world_m"]
        )
    ]
    jv = [
        float(
            sum(
                analytic[axis][column] * arm_direction[column]
                for column in range(7)
            )
        )
        for axis in range(3)
    ]
    point_error = [float(left - right) for left, right in zip(point_fd, jv)]
    point_error_max = max(abs(value) for value in point_error)
    point_fd_norm = math.sqrt(sum(value * value for value in point_fd))
    point_direction_near_zero = bool(
        point_fd_norm
        <= config["point_jacobian_near_zero_frobenius_m_per_rad"]
    )
    point_direction_relative = (
        None
        if point_direction_near_zero
        else math.sqrt(sum(value * value for value in point_error)) / point_fd_norm
    )
    point_direction_absolute_pass = bool(
        point_error_max <= config["point_jacobian_absolute_tolerance_m_per_rad"]
    )
    point_direction_relative_pass = bool(
        point_direction_near_zero
        or point_direction_relative <= config["point_jacobian_relative_tolerance"]
    )
    h_fd = float(
        (chosen["plus_query"]["value_m2"] - chosen["minus_query"]["value_m2"])
        / (2.0 * eta)
    )
    grad_point_fd = float(sum(g * v for g, v in zip(gradient, point_fd)))
    grad_jv = float(sum(g * v for g, v in zip(gradient, jv)))
    error = abs(h_fd - grad_point_fd)
    scale = abs(h_fd)
    near_zero = scale <= config["coupled_near_zero_m2_per_s"]
    relative = None if near_zero else error / scale
    absolute_pass = error <= config["coupled_absolute_tolerance_m2_per_s"]
    relative_pass = bool(near_zero or relative <= config["coupled_relative_tolerance"])
    point_error_norm = math.sqrt(sum(value * value for value in point_error))
    gradient_norm = math.sqrt(sum(value * value for value in gradient))
    direct_error = abs(h_fd - grad_jv)
    direct_allowance = float(
        config["coupled_absolute_tolerance_m2_per_s"]
        + gradient_norm * point_error_norm
    )
    cancellation = bool(
        chosen["plus_query"]["value_m2"] == chosen["minus_query"]["value_m2"]
        and abs(grad_point_fd) > config["coupled_near_zero_m2_per_s"]
    )
    return {
        "selected_eta_s": eta,
        "point_fd_velocity_m_s": point_fd,
        "analytic_jacobian_velocity_m_s": jv,
        "point_velocity_error_m_s": point_error,
        "point_velocity_error_norm_m_s": point_error_norm,
        "point_velocity_maximum_absolute_error_m_s": point_error_max,
        "point_velocity_numeric_norm_m_s": point_fd_norm,
        "point_velocity_relative_error": point_direction_relative,
        "point_velocity_relative_criterion_waived_for_near_zero_numeric_norm": point_direction_near_zero,
        "point_velocity_absolute_tolerance_m_s": config[
            "point_jacobian_absolute_tolerance_m_per_rad"
        ],
        "point_velocity_relative_tolerance": config[
            "point_jacobian_relative_tolerance"
        ],
        "point_velocity_near_zero_norm_m_s": config[
            "point_jacobian_near_zero_frobenius_m_per_rad"
        ],
        "point_velocity_absolute_passed": point_direction_absolute_pass,
        "point_velocity_relative_passed": point_direction_relative_pass,
        "field_central_derivative_m2_per_s": h_fd,
        "gradient_dot_point_fd_m2_per_s": grad_point_fd,
        "gradient_dot_analytic_jacobian_m2_per_s": grad_jv,
        "field_vs_point_fd_absolute_error_m2_per_s": error,
        "field_vs_point_fd_relative_error": relative,
        "relative_criterion_waived_for_near_zero_numeric_derivative": near_zero,
        "absolute_tolerance_m2_per_s": config["coupled_absolute_tolerance_m2_per_s"],
        "relative_tolerance": config["coupled_relative_tolerance"],
        "near_zero_m2_per_s": config["coupled_near_zero_m2_per_s"],
        "absolute_passed": absolute_pass,
        "relative_passed": relative_pass,
        "direct_analytic_absolute_error_m2_per_s": direct_error,
        "direct_analytic_allowance_m2_per_s": direct_allowance,
        "direct_analytic_passed": bool(direct_error <= direct_allowance),
        "central_numerator_cancellation_detected": cancellation,
    }


def validate_protected_sample_differential_audit(
    payload: Mapping[str, Any],
    *,
    expected_samples: Iterable[Any],
    expected_arm_dof_indices: Sequence[int],
    expected_integration_state_sha256: str,
    expected_differential_audit_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Purely reconstruct and require one complete passing audit artifact."""

    top_keys = (
        "schema_version",
        "integration_state",
        "arm_dof_indices",
        "ordered_samples",
        "ordered_sample_identity_sha256",
        "specification_sha256",
        "binding_sha256",
        "classification_ledger_sha256",
        "differential_audit_config",
        "sample_records",
        "counts",
        "passed",
        DIFFERENTIAL_AUDIT_HASH_FIELD,
    )
    top = _exact_keys(payload, top_keys, "differential audit")
    if top["schema_version"] != PROTECTED_SAMPLE_DIFFERENTIAL_SCHEMA:
        raise DifferentialAuditError("differential audit schema is unsupported")
    observed_hash = top[DIFFERENTIAL_AUDIT_HASH_FIELD]
    if not isinstance(observed_hash, str) or len(observed_hash) != 64:
        raise DifferentialAuditError("audit_payload_sha256 is invalid")
    unhashed = {key: value for key, value in top.items() if key != DIFFERENTIAL_AUDIT_HASH_FIELD}
    if sha256_bytes(canonical_json_bytes(unhashed)) != observed_hash:
        raise DifferentialAuditError("audit_payload_sha256 does not match payload")

    expected_config = _parse_differential_config(expected_differential_audit_config)
    if _json_copy(top["differential_audit_config"]) != expected_config:
        raise DifferentialAuditError("differential audit config differs from authority")
    dofs = _arm_dofs(expected_arm_dof_indices)
    if top["arm_dof_indices"] != dofs:
        raise DifferentialAuditError("arm DOF order differs from authority")
    identities = _sample_identities(expected_samples)
    if top["ordered_samples"] != identities:
        raise DifferentialAuditError("ordered protected samples differ from authority")
    identity_hash = sha256_bytes(canonical_json_bytes(identities))
    if top["ordered_sample_identity_sha256"] != identity_hash:
        raise DifferentialAuditError("ordered sample identity hash does not match")
    if (
        not isinstance(expected_integration_state_sha256, str)
        or len(expected_integration_state_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_integration_state_sha256)
    ):
        raise DifferentialAuditError("expected integration state hash is invalid")
    state = _exact_keys(
        top["integration_state"],
        (
            "state_specification",
            "element_count",
            "source_initial_sha256",
            "clone_base_sha256",
            "clone_final_sha256",
            "source_final_sha256",
            "source_state_unchanged",
            "clone_state_restored",
        ),
        "integration_state",
    )
    if state["state_specification"] != "mjSTATE_INTEGRATION":
        raise DifferentialAuditError("wrong MuJoCo state specification")
    if isinstance(state["element_count"], bool) or not isinstance(state["element_count"], int) or state["element_count"] <= 0:
        raise DifferentialAuditError("integration state element count is invalid")
    hashes = [
        state[key]
        for key in (
            "source_initial_sha256",
            "clone_base_sha256",
            "clone_final_sha256",
            "source_final_sha256",
        )
    ]
    if hashes != [expected_integration_state_sha256] * 4:
        raise DifferentialAuditError("source/clone integration state binding failed")
    if state["source_state_unchanged"] is not True or state["clone_state_restored"] is not True:
        raise DifferentialAuditError("source or clone state was not preserved")

    rows = top["sample_records"]
    if not isinstance(rows, list) or len(rows) != len(identities):
        raise DifferentialAuditError("sample record population is incomplete")
    nv = None
    total_attempts = 0
    noneligible_attempts = 0
    selected_count = 0
    direction_pass_count = 0
    point_pass_count = 0
    sample_pass_count = 0
    for sample_index, (row, identity) in enumerate(zip(rows, identities)):
        sample = _exact_keys(
            row,
            (
                "identity",
                "base_point_world_m",
                "base_field_query",
                "analytic_point_jacobian_m_per_rad_3x7",
                "plus_points_world_m_by_arm_dof",
                "minus_points_world_m_by_arm_dof",
                "numerical_point_jacobian_m_per_rad_3x7",
                "point_jacobian_metrics",
                "point_perturbation_tangent_reconstruction_by_arm_dof",
                "point_jacobian_passed",
                "coupled_directions",
                "eligible_coupled_direction_count",
                "coupled_directions_passed",
                "passed",
            ),
            "sample_records[%d]" % sample_index,
        )
        if sample["identity"] != identity:
            raise DifferentialAuditError("sample record identity/order mismatch")
        base_point = _vector(sample["base_point_world_m"], 3, "base point")
        base_query = _validate_query_record(sample["base_field_query"], base_point, "base query")
        analytic = _matrix(sample["analytic_point_jacobian_m_per_rad_3x7"], 3, 7, "analytic Jacobian")
        plus_points = _matrix(sample["plus_points_world_m_by_arm_dof"], 7, 3, "plus points")
        minus_points = _matrix(sample["minus_points_world_m_by_arm_dof"], 7, 3, "minus points")
        numerical = _matrix(sample["numerical_point_jacobian_m_per_rad_3x7"], 3, 7, "numeric Jacobian")
        expected_numerical = [
            [
                float((plus_points[column][axis] - minus_points[column][axis]) / (2.0 * expected_config["point_jacobian_delta_rad"]))
                for column in range(7)
            ]
            for axis in range(3)
        ]
        if numerical != expected_numerical:
            raise DifferentialAuditError("numerical point Jacobian does not reconstruct")
        metrics = _point_metrics(analytic, numerical, expected_config)
        if _json_copy(sample["point_jacobian_metrics"]) != metrics:
            raise DifferentialAuditError("point Jacobian metrics do not reconstruct")
        tangent_rows = sample["point_perturbation_tangent_reconstruction_by_arm_dof"]
        if not isinstance(tangent_rows, list) or len(tangent_rows) != 7:
            raise DifferentialAuditError("point tangent reconstruction is incomplete")
        point_tangent_pass = True
        for column, tangent_row in enumerate(tangent_rows):
            if nv is None:
                requested = tangent_row.get("requested_tangent_nv_rad_s") if isinstance(tangent_row, Mapping) else None
                if not isinstance(requested, list):
                    raise DifferentialAuditError("tangent reconstruction lacks full-nv array")
                nv = len(requested)
                _arm_dofs(dofs, nv)
            expected_tangent = [0.0] * nv
            expected_tangent[dofs[column]] = 1.0
            reconstructed = _validate_tangent_record(
                tangent_row, expected_tangent, dofs, expected_config, "point tangent"
            )
            point_tangent_pass = point_tangent_pass and reconstructed["passed"]
        point_passed = bool(metrics["passed"] and point_tangent_pass)
        if sample["point_jacobian_passed"] is not point_passed:
            raise DifferentialAuditError("point_jacobian_passed does not reconstruct")
        point_pass_count += int(point_passed)

        directions = sample["coupled_directions"]
        expected_directions = expected_config["joint_velocity_directions_rad_s"]
        if not isinstance(directions, list) or len(directions) != len(expected_directions):
            raise DifferentialAuditError("coupled direction population is incomplete")
        sample_selected = 0
        sample_coupled_pass = True
        for direction_index, (direction, arm_direction) in enumerate(zip(directions, expected_directions)):
            direction = _exact_keys(
                direction,
                (
                    "direction_index",
                    "arm_joint_velocity_rad_s",
                    "attempts",
                    "selected_attempt_index",
                    "derived_comparison",
                    "passed",
                ),
                "coupled direction",
            )
            if direction["direction_index"] != direction_index or direction["arm_joint_velocity_rad_s"] != arm_direction:
                raise DifferentialAuditError("coupled direction identity is wrong")
            attempt_rows = direction["attempts"]
            if not isinstance(attempt_rows, list) or not attempt_rows:
                raise DifferentialAuditError("coupled direction has no eta attempts")
            chosen = None
            for attempt_index, attempt in enumerate(attempt_rows):
                attempt = _exact_keys(
                    attempt,
                    (
                        "eta_s",
                        "plus_point_world_m",
                        "minus_point_world_m",
                        "plus_query",
                        "minus_query",
                        "tangent_reconstruction",
                        "eligible_same_cell_stencil",
                        "noneligible_reasons",
                        "selected",
                    ),
                    "coupled attempt",
                )
                if attempt_index >= len(expected_config["coupled_eta_ladder_s"]) or attempt["eta_s"] != expected_config["coupled_eta_ladder_s"][attempt_index]:
                    raise DifferentialAuditError("eta attempts are not the registered prefix")
                plus_point = _vector(attempt["plus_point_world_m"], 3, "coupled plus point")
                minus_point = _vector(attempt["minus_point_world_m"], 3, "coupled minus point")
                plus_query = _validate_query_record(attempt["plus_query"], plus_point, "coupled plus query")
                minus_query = _validate_query_record(attempt["minus_query"], minus_point, "coupled minus query")
                reasons = _noneligible_reasons(base_query, plus_query, minus_query)
                eligible = not reasons
                if attempt["noneligible_reasons"] != reasons or attempt["eligible_same_cell_stencil"] is not eligible or attempt["selected"] is not eligible:
                    raise DifferentialAuditError("coupled stencil eligibility does not reconstruct")
                expected_tangent = [0.0] * nv
                for arm_column, dof in enumerate(dofs):
                    expected_tangent[dof] = arm_direction[arm_column]
                _validate_tangent_record(
                    attempt["tangent_reconstruction"], expected_tangent, dofs, expected_config, "coupled tangent"
                )
                total_attempts += 1
                noneligible_attempts += int(not eligible)
                if eligible:
                    if chosen is not None or attempt_index != len(attempt_rows) - 1:
                        raise DifferentialAuditError("eta selection did not stop at largest eligible stencil")
                    chosen = attempt
            if chosen is None and len(attempt_rows) != len(expected_config["coupled_eta_ladder_s"]):
                raise DifferentialAuditError("failed eta search did not retain every attempt")
            expected_selected_index = None if chosen is None else len(attempt_rows) - 1
            if direction["selected_attempt_index"] != expected_selected_index:
                raise DifferentialAuditError("selected eta index does not reconstruct")
            comparison = None
            direction_passed = False
            if chosen is not None:
                sample_selected += 1
                selected_count += 1
                comparison = _coupled_comparison(
                    chosen, analytic, base_query["gradient_m"], arm_direction, expected_config
                )
                tangent_pass = chosen["tangent_reconstruction"]["passed"]
                direction_passed = bool(
                    tangent_pass
                    and comparison["point_velocity_absolute_passed"]
                    and comparison["point_velocity_relative_passed"]
                    and comparison["absolute_passed"]
                    and comparison["relative_passed"]
                    and comparison["direct_analytic_passed"]
                    and not comparison["central_numerator_cancellation_detected"]
                )
            if direction["derived_comparison"] != comparison or direction["passed"] is not direction_passed:
                raise DifferentialAuditError("coupled comparison/pass flag does not reconstruct")
            direction_pass_count += int(direction_passed)
            sample_coupled_pass = sample_coupled_pass and direction_passed
        if sample["eligible_coupled_direction_count"] != sample_selected:
            raise DifferentialAuditError("eligible coupled count does not reconstruct")
        expected_coupled = bool(
            len(directions) == expected_config["required_direction_count_per_sample"]
            and sample_coupled_pass
        )
        expected_sample_pass = bool(point_passed and expected_coupled)
        if sample["coupled_directions_passed"] is not expected_coupled or sample["passed"] is not expected_sample_pass:
            raise DifferentialAuditError("sample pass flags do not reconstruct")
        sample_pass_count += int(expected_sample_pass)

    expected_counts = {
        "sample_count": len(identities),
        "point_jacobian_passed_sample_count": point_pass_count,
        "required_coupled_direction_count": len(identities)
        * expected_config["required_direction_count_per_sample"],
        "selected_coupled_direction_count": selected_count,
        "passed_coupled_direction_count": direction_pass_count,
        "coupled_attempt_count": total_attempts,
        "noneligible_coupled_attempt_count": noneligible_attempts,
        "passed_sample_count": sample_pass_count,
    }
    if top["counts"] != expected_counts:
        raise DifferentialAuditError("audit counts do not reconstruct")
    stable_hashes = _stable_audit_hashes(
        expected_integration_state_sha256, dofs, identities, expected_config, rows
    )
    for key, expected in stable_hashes.items():
        if top[key] != expected:
            raise DifferentialAuditError("%s does not reconstruct" % key)
    expected_pass = bool(sample_pass_count == len(identities))
    if top["passed"] is not expected_pass:
        raise DifferentialAuditError("top-level audit pass flag does not reconstruct")
    if not expected_pass:
        raise DifferentialAuditError("protected-sample differential audit did not pass")
    return {
        "schema_version": PROTECTED_SAMPLE_DIFFERENTIAL_SCHEMA,
        "audit_payload_sha256": observed_hash,
        "ordered_sample_identity_sha256": identity_hash,
        "integration_state_sha256": expected_integration_state_sha256,
        "specification_sha256": stable_hashes["specification_sha256"],
        "binding_sha256": stable_hashes["binding_sha256"],
        "classification_ledger_sha256": stable_hashes[
            "classification_ledger_sha256"
        ],
        "counts": expected_counts,
        "passed": True,
    }
