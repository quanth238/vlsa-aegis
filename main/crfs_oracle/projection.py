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

    def to_dict(self) -> dict:
        return asdict(self)


RolloutFn = Callable[[np.ndarray], dict]


def _decode_translation(
    free_delta: np.ndarray,
    nominal: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    """Use the fifth correction to enforce exact zero-sum translation."""
    prefix = nominal.copy()
    delta = np.zeros((5, 3), dtype=np.float64)
    delta[:4] = np.asarray(free_delta, dtype=np.float64).reshape(4, 3)
    delta[4] = -delta[:4].sum(axis=0)
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
