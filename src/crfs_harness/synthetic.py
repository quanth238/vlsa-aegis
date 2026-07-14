"""Dependency-free end-to-end fixture for the oracle harness.

This module validates orchestration, pairing, intervention signs, endpoint
projection, atomic completion, and aggregation. It is explicitly marked
`evidence_tier=synthetic`; its output cannot satisfy a SafeLIBERO research gate.
"""

from __future__ import annotations

import datetime as dt
import math
import random
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, content_hash, valid_completion
from .geometry import SphereObstacle, rollout_positions, swept_sphere_clearance
from .intervention import endpoint_project, endpoint_sum, equal_norm_random, integrate_remaining_flow
from .math3d import Matrix, frobenius, matrix_add, matrix_scale, matrix_subtract, norm, subtract
from .projection import sine_bump_repair


def run_synthetic_case(
    case: dict[str, Any],
    output_root: str | Path,
    run_id: str,
    safety_margin_m: float = 0.02,
    intervention_time: float = 0.5,
    overwrite: bool = False,
) -> tuple[Path, str]:
    destination = Path(output_root) / case["case_id"] / "results.json"
    if destination.exists() and valid_completion(destination) and not overwrite:
        return destination, "skipped_valid_completion"

    horizon = 5
    p0 = (-1.0, 0.0, 0.0)
    nominal: Matrix = tuple((0.4, 0.0, 0.0) for _ in range(horizon))
    rng = random.Random(case["policy_seed"])
    obstacle = SphereObstacle(
        center=(rng.uniform(-0.08, 0.08), rng.uniform(0.025, 0.075), 0.0),
        radius_m=0.24,
    )
    eef_radius_m = 0.05
    nominal_positions = rollout_positions(p0, nominal, kappa=1.0)
    nominal_clearance = swept_sphere_clearance(nominal_positions, obstacle, eef_radius_m)
    repair = sine_bump_repair(
        nominal_actions=nominal,
        p0=p0,
        obstacle=obstacle,
        eef_radius_m=eef_radius_m,
        safety_margin_m=safety_margin_m,
        action_bound=1.0,
    )
    if repair is None:
        result = {
            "schema_version": "1.0",
            "case_id": case["case_id"],
            "run_id": run_id,
            "status": "infeasible",
            "config_hash": content_hash({"safety_margin_m": safety_margin_m, "intervention_time": intervention_time}),
            "provenance": _provenance(case),
            "repair": {"feasible": False, "solver": "synthetic_sine_bump"},
            "trials": {},
        }
        atomic_write_json(destination, result)
        return destination, "infeasible"

    noise = tuple(tuple(rng.gauss(0.0, 1.0) for _ in range(3)) for _ in range(horizon))
    x_t = matrix_add(matrix_scale(noise, intervention_time), matrix_scale(nominal, 1.0 - intervention_time))

    def base_velocity(_state: Matrix, _time: float) -> Matrix:
        return matrix_subtract(noise, nominal)

    random_correction = equal_norm_random(repair.correction, case["random_control_seed"])
    nominal_final = integrate_remaining_flow(x_t, base_velocity, intervention_time, 5)
    oracle_final = integrate_remaining_flow(
        x_t, base_velocity, intervention_time, 5, correction=repair.correction
    )
    random_final = integrate_remaining_flow(
        x_t, base_velocity, intervention_time, 5, correction=random_correction
    )
    bridge_final = integrate_remaining_flow(
        x_t, base_velocity, intervention_time, 5, correction=repair.correction, one_shot_edit=True
    )
    trials = {
        "nominal": _trial(nominal_final, nominal, p0, obstacle, eef_radius_m, safety_margin_m),
        "direct_repair": _trial(repair.actions, nominal, p0, obstacle, eef_radius_m, safety_margin_m),
        "random_residual": _trial(random_final, nominal, p0, obstacle, eef_radius_m, safety_margin_m),
        "oracle_residual": _trial(oracle_final, nominal, p0, obstacle, eef_radius_m, safety_margin_m),
        "bridge_edit": _trial(bridge_final, nominal, p0, obstacle, eef_radius_m, safety_margin_m),
    }
    result = {
        "schema_version": "1.0",
        "case_id": case["case_id"],
        "run_id": run_id,
        "status": "completed",
        "config_hash": content_hash(
            {
                "safety_margin_m": safety_margin_m,
                "intervention_time": intervention_time,
                "horizon": horizon,
                "sampler_steps": 10,
            }
        ),
        "provenance": _provenance(case),
        "measurement": {
            "kind": "analytic_swept_sphere",
            "evidence_tier": "synthetic",
            "warning": "Not MuJoCo evidence and cannot pass a CRFS research gate.",
            "nominal_clearance_m": nominal_clearance,
            "safety_margin_m": safety_margin_m,
        },
        "repair": {
            "feasible": True,
            "solver": "synthetic_sine_bump",
            "verified_clearance_m": repair.clearance_m,
            "endpoint_error_m": repair.endpoint_error_m,
            "objective": repair.objective,
            "correction_norm": frobenius(repair.correction),
            "endpoint_correction_sum": list(endpoint_sum(repair.correction)),
        },
        "trials": trials,
    }
    atomic_write_json(destination, result)
    return destination, "completed"


def _trial(
    actions: Matrix,
    nominal: Matrix,
    p0: tuple[float, float, float],
    obstacle: SphereObstacle,
    eef_radius_m: float,
    safety_margin_m: float,
) -> dict[str, Any]:
    positions = rollout_positions(p0, actions, kappa=1.0)
    nominal_positions = rollout_positions(p0, nominal, kappa=1.0)
    clearance = swept_sphere_clearance(positions, obstacle, eef_radius_m)
    return {
        "clearance_m": clearance,
        "safe": clearance >= safety_margin_m - 1e-10,
        "endpoint_error_m": norm(subtract(positions[-1], nominal_positions[-1])),
        "action_change_norm": frobenius(matrix_subtract(actions, nominal)),
    }


def _provenance(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_tier": "synthetic",
        "group_id": case["group_id"],
        "policy_seed": case["policy_seed"],
        "random_control_seed": case["random_control_seed"],
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline_repository": "THU-RCSCT/vlsa-aegis",
        "baseline_commit": "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b",
    }
