#!/usr/bin/env python3
"""Allocation-side numeric check for the JAX flow-guidance projection."""

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.pi0 import project_action_xyz_halfspaces


def main() -> int:
    actions = jnp.zeros((1, 10, 32), dtype=jnp.float32)
    rows = np.zeros((4, 30), dtype=np.float32)
    rows[0, 0] = 1.0
    rows[1, 1] = 1.0
    rows[2, 0] = -1.0
    rows[3, 2] = 1.0
    lower = jnp.asarray([0.5, 0.2, -0.8, -0.1], dtype=jnp.float32)
    projected = jax.jit(project_action_xyz_halfspaces)(
        actions, jnp.asarray(rows), lower
    )
    xyz = np.asarray(projected[0, :, :3]).reshape(-1)
    residual = rows @ xyz - np.asarray(lower)
    if float(np.min(residual)) < -1.0e-5:
        raise RuntimeError("JAX Dykstra projection violates a preflight halfspace")
    if not np.array_equal(np.asarray(projected[..., 3:]), np.zeros((1, 10, 29))):
        raise RuntimeError("JAX flow projection changed a non-XYZ action dimension")
    print(
        {
            "device": str(jax.devices()[0]),
            "minimum_residual": float(np.min(residual)),
            "projected_xyz_prefix": xyz[:3].tolist(),
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
