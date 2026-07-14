"""CRFS oracle experiment extensions for the VLSA-Aegis baseline."""

from .measurement import ClearanceMeasurement, GeomClearanceMonitor
from .projection import ProjectionResult, solve_simulator_projection

__all__ = [
    "ClearanceMeasurement",
    "GeomClearanceMonitor",
    "ProjectionResult",
    "solve_simulator_projection",
]
