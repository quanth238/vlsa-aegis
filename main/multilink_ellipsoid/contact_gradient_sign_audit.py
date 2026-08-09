"""Pure helpers for the paired contact-risk gradient sign audit."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def paired_gradient_candidates(
    nominal_xyz: Sequence[float],
    gradient: Sequence[float],
    epsilon: float,
    action_limit: float,
) -> tuple[Any, Any, Any]:
    """Return ``u-epsilon*g_hat`` and ``u+epsilon*g_hat`` without clipping."""

    import numpy as np

    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    vector = np.asarray(gradient, dtype=np.float64)
    if nominal.shape != (3,) or vector.shape != (3,):
        raise ValueError("gradient-sign audit expects three XYZ dimensions")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("gradient-sign audit requires a nonzero finite gradient")
    amount = float(epsilon)
    if not np.isfinite(amount) or amount <= 0.0:
        raise ValueError("gradient-sign audit epsilon must be positive")
    direction = vector / norm
    minus = nominal - amount * direction
    plus = nominal + amount * direction
    limit = float(action_limit)
    if np.any(np.abs(minus) > limit) or np.any(np.abs(plus) > limit):
        raise ValueError("gradient-sign audit pair exceeds action bounds")
    return minus, plus, direction


def contact_burden(transition: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize raw substep contacts without using a mesh-distance oracle."""

    events = list(transition.get("raw_protected_contact_events", []))
    penetrations = [max(0.0, -float(item["distance_m"])) for item in events]
    return {
        "raw_safe": not events,
        "contact_count": len(events),
        "summed_penetration_m": float(sum(penetrations)),
        "maximum_penetration_m": float(max(penetrations) if penetrations else 0.0),
        "first_contact_substep": (
            None if not events else min(int(item["substep_index"]) for item in events)
        ),
    }


def compare_contact_burden(
    minus: Mapping[str, Any], plus: Mapping[str, Any], tolerance_m: float
) -> str:
    """Return the simulator-preferred sign using raw contact evidence."""

    minus_safe = bool(minus["raw_safe"])
    plus_safe = bool(plus["raw_safe"])
    if minus_safe != plus_safe:
        return "minus" if minus_safe else "plus"
    minus_count = int(minus["contact_count"])
    plus_count = int(plus["contact_count"])
    if minus_count != plus_count:
        return "minus" if minus_count < plus_count else "plus"
    difference = float(minus["summed_penetration_m"]) - float(
        plus["summed_penetration_m"]
    )
    if abs(difference) <= float(tolerance_m):
        return "tie"
    return "minus" if difference < 0.0 else "plus"

