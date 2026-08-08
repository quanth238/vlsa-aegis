#!/usr/bin/env python3
"""Allocation-side numeric check for task-metric multi-row projection."""

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.pi0 import project_action_xyz_metric_halfspaces


def main() -> int:
    actions = jnp.zeros((1, 10, 32), dtype=jnp.float32)
    rows = np.zeros((4, 30), dtype=np.float64)
    rows[0, 0] = 1.0
    rows[1, 1] = 1.0
    rows[2, 0] = -1.0
    rows[3, 2] = 1.0
    lower = np.asarray([0.5, 0.2, -0.8, -0.1], dtype=np.float64)
    metric = np.diag(np.linspace(0.5, 2.0, 30))
    directions = np.linalg.solve(metric, rows.T).T
    projected = jax.jit(project_action_xyz_metric_halfspaces)(
        actions,
        jnp.asarray(rows, dtype=jnp.float32),
        jnp.asarray(lower, dtype=jnp.float32),
        jnp.asarray(directions, dtype=jnp.float32),
    )
    xyz = np.asarray(projected[0, :, :3]).reshape(-1)
    residual = rows @ xyz - lower
    if float(np.min(residual)) < -1.0e-5:
        raise RuntimeError("task-metric Dykstra projection violates a halfspace")
    if not np.array_equal(np.asarray(projected[..., 3:]), np.zeros((1, 10, 29))):
        raise RuntimeError("task-metric projection changed a non-XYZ dimension")
    print(
        {
            "device": str(jax.devices()[0]),
            "minimum_residual": float(np.min(residual)),
            "projection": "task_metric_multi_halfspace",
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
