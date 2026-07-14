"""Nearest safe, endpoint-preserving action projection using simulator rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ProjectionResult:
    feasible: bool
    actions: list[list[float]] | None
    correction: list[list[float]] | None
    objective: float | None
    verified_clearance_m: float | None
    endpoint_error_m: float | None
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    evaluations: int
    infeasibility_certificate: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


RolloutFn = Callable[[np.ndarray], dict]


def _kinematic_clearance(
    actions: np.ndarray,
    start_eef_center_m: np.ndarray,
    response_matrix: np.ndarray,
    obstacle_boxes: list[dict],
    eef_radius_m: float,
    samples_per_segment: int = 26,
) -> float:
    from .measurement import signed_distance_point_to_oriented_box

    position = np.asarray(start_eef_center_m, dtype=np.float64).copy()
    minimum = float("inf")
    for action in actions:
        next_position = position + response_matrix @ action[:3]
        for alpha in np.linspace(0.0, 1.0, samples_per_segment):
            point = (1.0 - alpha) * position + alpha * next_position
            for box in obstacle_boxes:
                distance = signed_distance_point_to_oriented_box(
                    point,
                    box["center_m"],
                    np.asarray(box["rotation_world"], dtype=np.float64).reshape(3, 3),
                    box["half_size_m"],
                ) - eef_radius_m
                minimum = min(minimum, float(distance))
        position = next_position
    return minimum


def solve_kinematic_projection(
    nominal_prefix: np.ndarray,
    *,
    start_eef_center_m: np.ndarray,
    response_matrix: np.ndarray,
    obstacle_boxes: list[dict],
    eef_radius_m: float,
    safety_margin_m: float,
    action_low: float | np.ndarray = -1.0,
    action_high: float | np.ndarray = 1.0,
    max_iterations: int = 120,
    ftol: float = 1e-7,
) -> ProjectionResult:
    """Solve Eq. (8) with frozen H04 kinematics and static branch geometry."""
    nominal = np.asarray(nominal_prefix, dtype=np.float64)
    if nominal.ndim != 2 or nominal.shape[0] < 2 or nominal.shape[1] < 3:
        raise ValueError(f"Expected nominal prefix shape (H>=2, >=3), got {nominal.shape}")
    low = np.broadcast_to(np.asarray(action_low, dtype=np.float64), nominal.shape)
    high = np.broadcast_to(np.asarray(action_high, dtype=np.float64), nominal.shape)
    if np.any(nominal < low - 1e-8) or np.any(nominal > high + 1e-8):
        raise ValueError("Nominal prefix lies outside declared action bounds")
    response = np.asarray(response_matrix, dtype=np.float64).reshape(3, 3)

    # Every admissible correction has zero translational sum, so all candidates
    # share this endpoint in the frozen H04 model.  If the endpoint itself
    # violates the requested margin, the swept-path constraint is impossible;
    # this is a certificate, not an optimizer failure.
    fixed_endpoint = np.asarray(start_eef_center_m, dtype=np.float64) + response @ nominal[:, :3].sum(axis=0)
    endpoint_clearance = _kinematic_clearance(
        np.zeros((1, nominal.shape[1]), dtype=np.float64),
        fixed_endpoint,
        response,
        obstacle_boxes,
        eef_radius_m,
        samples_per_segment=1,
    )
    if endpoint_clearance < safety_margin_m:
        certificate = {
            "type": "fixed_endpoint_clearance",
            "endpoint_m": fixed_endpoint.tolist(),
            "clearance_upper_bound_m": float(endpoint_clearance),
            "safety_margin_m": float(safety_margin_m),
        }
        return ProjectionResult(
            feasible=False,
            actions=None,
            correction=None,
            objective=None,
            verified_clearance_m=float(endpoint_clearance),
            endpoint_error_m=0.0,
            optimizer_success=False,
            optimizer_status=2,
            optimizer_message="fixed endpoint violates the safety margin",
            evaluations=1,
            infeasibility_certificate=certificate,
        )

    from scipy.optimize import minimize

    evaluations = 0
    cache: dict[bytes, tuple[np.ndarray, float]] = {}

    def evaluate(free_delta: np.ndarray) -> tuple[np.ndarray, float]:
        nonlocal evaluations
        key = np.asarray(free_delta, dtype=np.float64).tobytes()
        if key not in cache:
            actions = _decode_translation(free_delta, nominal, low, high)
            clearance = _kinematic_clearance(
                actions, start_eef_center_m, response, obstacle_boxes, eef_radius_m
            )
            cache[key] = actions, clearance
            evaluations += 1
        return cache[key]

    def objective(free_delta: np.ndarray) -> float:
        actions, _ = evaluate(free_delta)
        difference = actions[:, :3] - nominal[:, :3]
        return 0.5 * float(np.sum(difference * difference))

    def clearance_constraint(free_delta: np.ndarray) -> float:
        return evaluate(free_delta)[1] - safety_margin_m

    def final_lower_bound(free_delta: np.ndarray) -> np.ndarray:
        return _decode_translation(free_delta, nominal, low, high)[-1, :3] - low[-1, :3]

    def final_upper_bound(free_delta: np.ndarray) -> np.ndarray:
        return high[-1, :3] - _decode_translation(free_delta, nominal, low, high)[-1, :3]

    horizon = nominal.shape[0]
    bounds = [
        (low[row, column] - nominal[row, column], high[row, column] - nominal[row, column])
        for row in range(horizon - 1)
        for column in range(3)
    ]
    constraints = [
        {"type": "ineq", "fun": clearance_constraint},
        {"type": "ineq", "fun": final_lower_bound},
        {"type": "ineq", "fun": final_upper_bound},
    ]
    starts = [np.zeros((horizon - 1) * 3, dtype=np.float64)]
    # The pointwise minimum over obstacle boxes is nonsmooth. Generic temporal
    # bump starts prevent the colliding zero correction from being the only
    # basin considered; all starts still satisfy exact zero-sum correction.
    bump = np.sin(2.0 * np.pi * np.arange(horizon, dtype=np.float64) / horizon)[:-1]
    for axis in range(3):
        for sign in (-1.0, 1.0):
            for amplitude in (0.25, 0.5):
                start = np.zeros((horizon - 1, 3), dtype=np.float64)
                start[:, axis] = sign * amplitude * bump
                starts.append(start.reshape(-1))

    attempts = []
    for start in starts:
        clipped = np.asarray(
            [np.clip(value, lower, upper) for value, (lower, upper) in zip(start, bounds)],
            dtype=np.float64,
        )
        attempt = minimize(
            objective,
            clipped,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": int(max_iterations), "ftol": float(ftol), "disp": False},
        )
        actions, attempt_clearance = evaluate(np.asarray(attempt.x, dtype=np.float64))
        correction = actions[:, :3] - nominal[:, :3]
        attempt_objective = 0.5 * float(np.sum(correction * correction))
        attempts.append((attempt, actions, attempt_clearance, attempt_objective))

    feasible_attempts = [item for item in attempts if item[2] >= safety_margin_m]
    if feasible_attempts:
        result, candidate, clearance, selected_objective = min(feasible_attempts, key=lambda item: item[3])
    else:
        result, candidate, clearance, selected_objective = max(attempts, key=lambda item: item[2])
    correction = candidate[:, :3] - nominal[:, :3]
    endpoint_error = float(np.linalg.norm(response @ correction.sum(axis=0)))
    within_bounds = bool(np.all(candidate >= low - 1e-7) and np.all(candidate <= high + 1e-7))
    feasible = bool(clearance >= safety_margin_m and endpoint_error <= 1e-6 and within_bounds)
    return ProjectionResult(
        feasible=feasible,
        actions=candidate.tolist() if feasible else None,
        correction=correction.tolist() if feasible else None,
        objective=selected_objective if feasible else None,
        verified_clearance_m=clearance,
        endpoint_error_m=endpoint_error,
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        evaluations=evaluations,
    )


def _decode_translation(
    free_delta: np.ndarray,
    nominal: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    """Use the final correction to enforce exact zero-sum translation."""
    prefix = nominal.copy()
    horizon = nominal.shape[0]
    delta = np.zeros((horizon, 3), dtype=np.float64)
    delta[:-1] = np.asarray(free_delta, dtype=np.float64).reshape(horizon - 1, 3)
    delta[-1] = -delta[:-1].sum(axis=0)
    prefix[:, :3] += delta
    return prefix


def solve_simulator_projection(
    nominal_prefix: np.ndarray,
    rollout: RolloutFn,
    *,
    safety_margin_m: float,
    action_low: float | np.ndarray = -1.0,
    action_high: float | np.ndarray = 1.0,
    max_iterations: int = 120,
    ftol: float = 1e-7,
) -> ProjectionResult:
    """Solve Eq. (8) from ``main.tex`` against real simulator measurements.

    Only the first five translational actions vary. Orientation and gripper
    commands stay nominal. Endpoint preservation is structural: the fifth
    translational correction is the negative sum of the first four.
    """
    from scipy.optimize import minimize

    nominal = np.asarray(nominal_prefix, dtype=np.float64)
    if nominal.ndim != 2 or nominal.shape[0] != 5 or nominal.shape[1] < 3:
        raise ValueError(f"Expected nominal prefix shape (5, >=3), got {nominal.shape}")
    low = np.broadcast_to(np.asarray(action_low, dtype=np.float64), nominal.shape)
    high = np.broadcast_to(np.asarray(action_high, dtype=np.float64), nominal.shape)
    if np.any(nominal < low - 1e-8) or np.any(nominal > high + 1e-8):
        raise ValueError("Nominal prefix lies outside declared action bounds")

    cache: dict[bytes, tuple[np.ndarray, dict]] = {}
    evaluations = 0

    def evaluate(free_delta: np.ndarray) -> tuple[np.ndarray, dict]:
        nonlocal evaluations
        key = np.asarray(free_delta, dtype=np.float64).tobytes()
        if key not in cache:
            actions = _decode_translation(free_delta, nominal, low, high)
            cache[key] = actions, rollout(actions)
            evaluations += 1
        return cache[key]

    def objective(free_delta: np.ndarray) -> float:
        actions, _ = evaluate(free_delta)
        difference = actions[:, :3] - nominal[:, :3]
        return 0.5 * float(np.sum(difference * difference))

    def clearance_constraint(free_delta: np.ndarray) -> float:
        _, result = evaluate(free_delta)
        return float(result["clearance_m"]) - safety_margin_m

    def fifth_lower_bound(free_delta: np.ndarray) -> np.ndarray:
        actions = _decode_translation(free_delta, nominal, low, high)
        return actions[4, :3] - low[4, :3]

    def fifth_upper_bound(free_delta: np.ndarray) -> np.ndarray:
        actions = _decode_translation(free_delta, nominal, low, high)
        return high[4, :3] - actions[4, :3]

    bounds: list[tuple[float, float]] = []
    for row in range(4):
        for column in range(3):
            bounds.append((low[row, column] - nominal[row, column], high[row, column] - nominal[row, column]))

    result = minimize(
        objective,
        np.zeros(12, dtype=np.float64),
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {"type": "ineq", "fun": clearance_constraint},
            {"type": "ineq", "fun": fifth_lower_bound},
            {"type": "ineq", "fun": fifth_upper_bound},
        ],
        options={"maxiter": int(max_iterations), "ftol": float(ftol), "disp": False},
    )
    candidate, verification = evaluate(np.asarray(result.x, dtype=np.float64))
    correction = candidate[:, :3] - nominal[:, :3]
    endpoint_error = float(np.linalg.norm(correction.sum(axis=0)))
    clearance = float(verification["clearance_m"])
    within_bounds = bool(np.all(candidate >= low - 1e-7) and np.all(candidate <= high + 1e-7))
    feasible = bool(clearance >= safety_margin_m and endpoint_error <= 1e-6 and within_bounds)
    return ProjectionResult(
        feasible=feasible,
        actions=candidate.tolist() if feasible else None,
        correction=correction.tolist() if feasible else None,
        objective=objective(np.asarray(result.x, dtype=np.float64)) if feasible else None,
        verified_clearance_m=clearance,
        endpoint_error_m=endpoint_error,
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        evaluations=evaluations,
    )
