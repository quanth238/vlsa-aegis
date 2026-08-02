"""Allocation-only cross-solver check on the active registered production grid.

This test intentionally is not part of ordinary local numerical bring-up.  It
allocates the exact v5 116 x 101 x 111 vertex grid and compares the registered
red-black SOR solve with SciPy conjugate gradient.  The Slurm numerical gate
sets the explicit opt-in environment variable and rejects every skip.  The
immutable v4 workspace and legacy v3 101-cubed constructor paths have separate
regression tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUN_PRODUCTION_GRID = (
    os.environ.get("VLSA_POISSON_RUN_PRODUCTION_GRID_VALIDATION") == "1"
)

# These are comparison tolerances, not PDE stopping criteria.  The SOR and CG
# stopping criteria remain the exact values registered below in the runtime
# protocol.  Units follow the implementation contract: h [m^2], grad(h) [m].
MAX_GRID_VALUE_ABS_DIFFERENCE_M2 = 1.0e-5
MAX_QUERY_VALUE_ABS_DIFFERENCE_M2 = 5.0e-6
MAX_QUERY_GRADIENT_COMPONENT_ABS_DIFFERENCE_M = 2.0e-5
MAX_REFERENCE_VALUE_NEAR_OUTER_BOUNDARY_M2 = 1.0e-3
MAX_NEAR_OUTER_BOUNDARY_CLEARANCE_M = 2.5e-4


@unittest.skipUnless(
    RUN_PRODUCTION_GRID,
    "set VLSA_POISSON_RUN_PRODUCTION_GRID_VALIDATION=1 inside the allocation",
)
class ProductionGridCrossSolverTest(unittest.TestCase):
    def test_registered_v5_sor_matches_scipy_cg_off_grid(self) -> None:
        import numpy as np

        from main.poisson_fullbody.feasibility_protocol import (
            load_feasibility_protocol,
        )
        from main.poisson_fullbody.poisson_field import (
            TrilinearPoissonField,
            build_poisson_system,
            solve_poisson_reference,
            solve_poisson_sor,
        )
        from main.poisson_fullbody.voxel_grid import GridSpec, PoissonDomain

        protocol_path = (
            ROOT / "configs/vlsa_poisson_runtime_protocol.canary.v5.json"
        )
        protocol, protocol_hashes = load_feasibility_protocol(protocol_path)
        workspace = protocol["workspace"]
        poisson = protocol["poisson"]

        vertex_shape = tuple(workspace["grid_shape_vertices"])
        spacing = np.asarray(workspace["grid_spacing_m"], dtype=np.float64)
        lower = np.asarray(workspace["minimum_m"], dtype=np.float64)
        upper = np.asarray(workspace["maximum_m"], dtype=np.float64)
        self.assertEqual(vertex_shape, (116, 101, 111))
        np.testing.assert_array_equal(spacing, np.full(3, 0.02))
        np.testing.assert_array_equal(lower, np.asarray([-1.3, -1.0, -0.2]))
        np.testing.assert_array_equal(upper, np.asarray([1.0, 1.0, 2.0]))
        self.assertEqual(poisson["solver"], "red_black_sor")
        self.assertEqual(poisson["equation"], "minus_laplacian_h_equals_forcing")
        self.assertEqual(workspace["boundary_condition"], "homogeneous_dirichlet_h_zero")

        grid = GridSpec.from_bounds(lower, upper, vertex_shape)
        # ``GridSpec.from_bounds`` reconstructs spacing from binary64 bounds.
        # For the 2.3 m x-span, division by 115 differs from the literal 0.02
        # by one rounding unit.  Use the same frozen construction tolerance as
        # the production field-bundle gate; this is not a PDE comparison
        # tolerance or a change to the registered physical grid.
        np.testing.assert_allclose(
            grid.spacing, spacing, rtol=0.0, atol=1.0e-15
        )

        # This synthetic allocation check isolates the production-size
        # operator and solver.  Every cell is in one free component, with the
        # six outer faces as the zero Dirichlet boundary.  Geometry/domain
        # construction has separate exact tests in the same numeric harness.
        cell_shape = tuple(value - 1 for value in vertex_shape)
        component_cells = np.ones(cell_shape, dtype=bool)
        active_vertices = np.ones(vertex_shape, dtype=bool)
        boundary_vertices = np.zeros(vertex_shape, dtype=bool)
        boundary_vertices[0, :, :] = True
        boundary_vertices[-1, :, :] = True
        boundary_vertices[:, 0, :] = True
        boundary_vertices[:, -1, :] = True
        boundary_vertices[:, :, 0] = True
        boundary_vertices[:, :, -1] = True
        domain = PoissonDomain(
            component_cells=component_cells,
            active_vertices=active_vertices,
            boundary_vertices=boundary_vertices,
            interior_vertices=active_vertices & ~boundary_vertices,
        )

        timings = {}
        started = time.perf_counter()
        system = build_poisson_system(
            grid,
            domain,
            forcing=float(poisson["forcing_value"]),
            boundary_values=float(poisson["boundary_value"]),
        )
        timings["system_build_seconds"] = time.perf_counter() - started
        self.assertEqual(system.shape, vertex_shape)
        self.assertEqual(system.unknown_count, 114 * 99 * 109)

        started = time.perf_counter()
        sor = solve_poisson_sor(
            system,
            omega=float(poisson["relaxation_omega"]),
            tolerance=float(poisson["normalized_backward_error_tolerance"]),
            max_iterations=int(poisson["max_iterations"]),
            check_every=int(poisson["residual_check_interval"]),
        )
        timings["sor_seconds"] = time.perf_counter() - started
        self.assertTrue(sor.converged, sor.message)
        self.assertTrue(sor.diagnostics.passed, sor.diagnostics)

        # CG is an independent iterative solver and avoids the prohibitive
        # fill-in/memory risk of a sparse direct solve at ~1M unknowns.
        started = time.perf_counter()
        reference = solve_poisson_reference(
            system,
            method="cg",
            tolerance=1.0e-11,
            max_iterations=10000,
        )
        timings["scipy_cg_seconds"] = time.perf_counter() - started
        self.assertTrue(reference.converged, reference.message)
        self.assertTrue(reference.diagnostics.passed, reference.diagnostics)

        grid_value_error = float(
            np.max(np.abs(sor.values - reference.values))
        )
        self.assertLessEqual(
            grid_value_error,
            MAX_GRID_VALUE_ABS_DIFFERENCE_M2,
            "production-grid SOR/CG value difference exceeded tolerance",
        )

        sor_field = TrilinearPoissonField(grid, sor.values, domain=domain)
        reference_field = TrilinearPoissonField(
            grid, reference.values, domain=domain
        )
        # Coordinates are expressed in grid-index units and deliberately have
        # no integer component.  The first eight points lie within 0.25 mm of
        # an outer face/corner, where h approaches the zero boundary and a
        # relative-only comparison would be ill-conditioned.
        maximum_grid_coordinate = np.asarray(vertex_shape, dtype=np.float64) - 1.0
        query_grid_coordinates = np.asarray(
            [
                [0.001, 37.375, 62.625],
                [maximum_grid_coordinate[0] - 0.001, 51.250, 17.625],
                [41.125, 0.002, 70.875],
                [58.625, maximum_grid_coordinate[1] - 0.002, 24.125],
                [27.375, 63.625, 0.003],
                [74.125, 35.375, maximum_grid_coordinate[2] - 0.003],
                [0.005, 0.007, 0.011],
                [
                    maximum_grid_coordinate[0] - 0.009,
                    maximum_grid_coordinate[1] - 0.013,
                    maximum_grid_coordinate[2] - 0.017,
                ],
                [55.125, 52.375, 50.625],
                [12.125, 83.375, 44.625],
            ],
            dtype=np.float64,
        )
        query_points = lower[None, :] + query_grid_coordinates * spacing[None, :]
        value_errors = []
        gradient_errors = []
        reference_values = []
        boundary_clearances = []
        for point in query_points:
            sor_query = sor_field.query(point)
            reference_query = reference_field.query(point)
            self.assertTrue(sor_query.valid, sor_query)
            self.assertTrue(reference_query.valid, reference_query)
            self.assertEqual(sor_query.cell_index, reference_query.cell_index)
            value_errors.append(abs(sor_query.value - reference_query.value))
            gradient_errors.append(
                float(
                    np.max(
                        np.abs(
                            np.asarray(sor_query.gradient)
                            - np.asarray(reference_query.gradient)
                        )
                    )
                )
            )
            reference_values.append(float(reference_query.value))
            boundary_clearances.append(
                float(reference_query.outer_boundary_clearance_m)
            )

        maximum_query_value_error = max(value_errors)
        maximum_query_gradient_error = max(gradient_errors)
        self.assertLessEqual(
            maximum_query_value_error,
            MAX_QUERY_VALUE_ABS_DIFFERENCE_M2,
        )
        self.assertLessEqual(
            maximum_query_gradient_error,
            MAX_QUERY_GRADIENT_COMPONENT_ABS_DIFFERENCE_M,
        )
        self.assertTrue(all(value > 0.0 for value in reference_values))
        self.assertLessEqual(
            max(reference_values[:8]),
            MAX_REFERENCE_VALUE_NEAR_OUTER_BOUNDARY_M2,
        )
        self.assertLessEqual(
            max(boundary_clearances[:8]),
            MAX_NEAR_OUTER_BOUNDARY_CLEARANCE_M,
        )

        metrics = {
            "protocol_semantic_sha256": protocol_hashes.protocol_sha256,
            "parameter_block_sha256": protocol_hashes.parameter_block_sha256,
            "vertex_shape": list(vertex_shape),
            "spacing_m": spacing.tolist(),
            "unknown_count": system.unknown_count,
            "sor_iterations": sor.iterations,
            "scipy_cg_iterations": reference.iterations,
            "sor_backward_error_linf": sor.diagnostics.backward_error_linf,
            "scipy_cg_backward_error_linf": (
                reference.diagnostics.backward_error_linf
            ),
            "max_grid_value_abs_difference_m2": grid_value_error,
            "max_query_value_abs_difference_m2": maximum_query_value_error,
            "max_query_gradient_component_abs_difference_m": (
                maximum_query_gradient_error
            ),
            "max_reference_value_near_outer_boundary_m2": max(
                reference_values[:8]
            ),
            "max_near_outer_boundary_clearance_m": max(
                boundary_clearances[:8]
            ),
            "registered_tolerances": {
                "sor_backward_error": float(
                    poisson["normalized_backward_error_tolerance"]
                ),
                "cg_relative_residual": 1.0e-11,
                "grid_value_abs_difference_m2": (
                    MAX_GRID_VALUE_ABS_DIFFERENCE_M2
                ),
                "query_value_abs_difference_m2": (
                    MAX_QUERY_VALUE_ABS_DIFFERENCE_M2
                ),
                "query_gradient_component_abs_difference_m": (
                    MAX_QUERY_GRADIENT_COMPONENT_ABS_DIFFERENCE_M
                ),
            },
            "timings_seconds": timings,
        }
        print("POISSON_PRODUCTION_GRID_METRICS=" + json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
