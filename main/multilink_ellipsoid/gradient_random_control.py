"""Pure helpers for the learned-gradient matched-random control."""

from __future__ import annotations

import math
import random
from typing import Sequence


def sample_feasible_unit_directions(
    nominal_xyz: Sequence[float], *, radius: float, count: int, seed: int,
    action_limit: float, maximum_attempts: int,
) -> list[list[float]]:
    if len(nominal_xyz) != 3 or radius <= 0 or count < 1 or action_limit <= 0:
        raise ValueError("random-control sampling arguments are invalid")
    generator = random.Random(int(seed))
    output = []
    attempts = 0
    while len(output) < count and attempts < maximum_attempts:
        attempts += 1
        values = [generator.gauss(0.0, 1.0) for _ in range(3)]
        norm = math.sqrt(sum(value * value for value in values))
        if norm <= 1e-15:
            continue
        direction = [value / norm for value in values]
        candidate = [
            float(base) + float(radius) * value
            for base, value in zip(nominal_xyz, direction)
        ]
        if all(-action_limit <= value <= action_limit for value in candidate):
            output.append(direction)
    if len(output) != count:
        raise ValueError("random-control feasible direction sampling exhausted")
    return output


def first_safe_radius(radii: Sequence[float], safe: Sequence[bool]) -> float | None:
    if len(radii) != len(safe) or not radii:
        raise ValueError("random-control radius arrays differ or are empty")
    values = [float(radius) for radius, passed in zip(radii, safe) if bool(passed)]
    return None if not values else min(values)


def empirical_equal_or_earlier_p(
    learned_first_safe: float | None,
    random_first_safe: Sequence[float | None],
) -> float:
    if learned_first_safe is None or not random_first_safe:
        return 1.0
    count = sum(
        value is not None and float(value) <= float(learned_first_safe) + 1e-15
        for value in random_first_safe
    )
    return (1.0 + float(count)) / (1.0 + float(len(random_first_safe)))
