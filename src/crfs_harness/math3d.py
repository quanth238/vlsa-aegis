"""Small dependency-free vector helpers used by harness-level tests."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

Vector = tuple[float, ...]
Matrix = tuple[Vector, ...]


def vector(values: Iterable[float]) -> Vector:
    return tuple(float(value) for value in values)


def add(left: Sequence[float], right: Sequence[float]) -> Vector:
    _same_size(left, right)
    return tuple(a + b for a, b in zip(left, right, strict=True))


def subtract(left: Sequence[float], right: Sequence[float]) -> Vector:
    _same_size(left, right)
    return tuple(a - b for a, b in zip(left, right, strict=True))


def scale(value: Sequence[float], factor: float) -> Vector:
    return tuple(factor * item for item in value)


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    _same_size(left, right)
    return sum(a * b for a, b in zip(left, right, strict=True))


def norm(value: Sequence[float]) -> float:
    return math.sqrt(dot(value, value))


def normalize(value: Sequence[float]) -> Vector:
    magnitude = norm(value)
    if magnitude <= 1e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return scale(value, 1.0 / magnitude)


def frobenius(value: Sequence[Sequence[float]]) -> float:
    return math.sqrt(sum(item * item for row in value for item in row))


def matrix_add(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    _same_size(left, right)
    return tuple(add(a, b) for a, b in zip(left, right, strict=True))


def matrix_subtract(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    _same_size(left, right)
    return tuple(subtract(a, b) for a, b in zip(left, right, strict=True))


def matrix_scale(value: Sequence[Sequence[float]], factor: float) -> Matrix:
    return tuple(scale(row, factor) for row in value)


def zeros(rows: int, columns: int) -> Matrix:
    if rows <= 0 or columns <= 0:
        raise ValueError("Matrix dimensions must be positive")
    return tuple(tuple(0.0 for _ in range(columns)) for _ in range(rows))


def _same_size(left: Sequence[object], right: Sequence[object]) -> None:
    if len(left) != len(right):
        raise ValueError(f"Mismatched dimensions: {len(left)} != {len(right)}")
