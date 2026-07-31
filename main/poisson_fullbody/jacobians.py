"""Arbitrary body-point Jacobians and finite-difference validation."""

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from main.poisson_fullbody.robot_samples import BodySample


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
