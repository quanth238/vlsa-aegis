"""Independent synthetic checks for physical-unit Poisson numerics."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "main/poisson_fullbody/poisson_field.py"

try:
    import numpy as np
    import scipy  # noqa: F401

    NUMERICS_AVAILABLE = True
except ImportError:
    np = None
    NUMERICS_AVAILABLE = False


def load_module():
    spec = importlib.util.spec_from_file_location("poisson_field_under_test", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(MODULE_PATH))
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


poisson = load_module()


@dataclass(frozen=True)
class TinyGrid:
    lower: object
    upper: object
    vertex_shape: tuple

    @property
    def spacing(self):
        return (self.upper - self.lower) / (
            np.asarray(self.vertex_shape, dtype=float) - 1.0
        )


@dataclass(frozen=True)
class TinyDomain:
    interior_mask: object
    boundary_vertex_mask: object
    valid_vertex_mask: object

    @property
    def solve_mask(self):
        return self.interior_mask

    @property
    def interior_vertices(self):
        return self.interior_mask

    @property
    def boundary_vertices(self):
        return self.boundary_vertex_mask

    @property
    def active_vertices(self):
        return self.valid_vertex_mask

    @property
    def component_cells(self):
        return np.ones(
            tuple(item - 1 for item in self.interior_mask.shape), dtype=bool
        )


def rectangular_domain(shape):
    interior = np.zeros(shape, dtype=bool)
    interior[1:-1, 1:-1, 1:-1] = True
    boundary = np.zeros(shape, dtype=bool)
    boundary[0, :, :] = True
    boundary[-1, :, :] = True
    boundary[:, 0, :] = True
    boundary[:, -1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True
    return TinyDomain(interior, boundary, np.ones(shape, dtype=bool))


def grid_vertices(grid):
    axes = [
        np.linspace(grid.lower[axis], grid.upper[axis], grid.vertex_shape[axis])
        for axis in range(3)
    ]
    return np.meshgrid(*axes, indexing="ij")


@unittest.skipUnless(NUMERICS_AVAILABLE, "NumPy and SciPy are optional locally")
class PoissonSystemTest(unittest.TestCase):
    def test_system_certificate_arrays_are_independent_contiguous_and_read_only(self):
        grid = TinyGrid(np.zeros(3), np.ones(3), (5, 5, 5))
        domain = rectangular_domain(grid.vertex_shape)
        forcing = np.ones(grid.vertex_shape, order="F")
        system = poisson.build_poisson_system(
            grid, domain, forcing=forcing
        )

        forcing[:] = 9.0
        domain.interior_mask[:] = False
        self.assertTrue(np.all(system.forcing == 1.0))
        self.assertTrue(np.any(system.interior_mask))
        certified_matrix = system.matrix.toarray()
        detached_matrix = system.matrix
        detached_matrix.data = np.zeros_like(detached_matrix.data)
        self.assertFalse(np.array_equal(detached_matrix.toarray(), certified_matrix))
        np.testing.assert_array_equal(system.matrix.toarray(), certified_matrix)
        for array in (
            system.rhs,
            system.lower,
            system.upper,
            system.spacing,
            system.interior_mask,
            system.boundary_mask,
            system.valid_vertex_mask,
            system.forcing,
            system.boundary_values,
            system.unknown_flat_indices,
            system.matrix.data,
            system.matrix.indices,
            system.matrix.indptr,
        ):
            self.assertTrue(array.flags.c_contiguous)
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flat[0] = array.flat[0]
            with self.assertRaises(ValueError):
                array.setflags(write=True)

    def test_manufactured_anisotropic_polynomial_is_exact(self):
        grid = TinyGrid(
            lower=np.asarray([-0.4, 0.2, 1.1]),
            upper=np.asarray([0.8, 1.7, 2.0]),
            vertex_shape=(9, 8, 7),
        )
        domain = rectangular_domain(grid.vertex_shape)
        x, y, z = grid_vertices(grid)
        xr = x - grid.lower[0]
        yr = y - grid.lower[1]
        zr = z - grid.lower[2]
        lx, ly, lz = grid.upper - grid.lower
        px = xr * (lx - xr)
        py = yr * (ly - yr)
        pz = zr * (lz - zr)
        exact = px * py * pz
        # Quadratic second differences are exact, including on an anisotropic
        # grid, so this manufactured RHS should reproduce the polynomial.
        forcing = 2.0 * (py * pz + px * pz + px * py)
        system = poisson.build_poisson_system(
            grid, domain, forcing=forcing, boundary_values=exact
        )
        result = poisson.solve_poisson_reference(system, tolerance=1.0e-12)
        self.assertTrue(result.converged, result.message)
        self.assertLess(np.max(np.abs(result.values - exact)), 2.0e-14)
        self.assertTrue(result.diagnostics.passed)
        self.assertLess(result.diagnostics.backward_error_linf, 2.0e-15)
        self.assertEqual(result.diagnostics.expected_sign, "nonnegative")
        self.assertEqual(result.diagnostics.sign_violation_count, 0)

    def test_constant_forcing_respects_discrete_maximum_principle(self):
        grid = TinyGrid(
            lower=np.zeros(3),
            upper=np.asarray([1.0, 0.7, 1.4]),
            vertex_shape=(10, 8, 9),
        )
        domain = rectangular_domain(grid.vertex_shape)
        system = poisson.build_poisson_system(grid, domain, forcing=1.0)
        result = poisson.solve_poisson_reference(system)
        self.assertTrue(result.converged, result.message)
        self.assertGreater(result.diagnostics.interior_min, 0.0)
        self.assertEqual(result.diagnostics.sign_violation_count, 0)
        self.assertEqual(result.diagnostics.boundary_linf, 0.0)

    def test_canonical_voxel_grid_and_domain_contract(self):
        import sys

        main_path = str(ROOT / "main")
        if main_path not in sys.path:
            sys.path.insert(0, main_path)
        from poisson_fullbody.voxel_grid import GridSpec, build_connected_domain

        grid = GridSpec(
            lower=[0.0, 0.0, 0.0],
            spacing=[0.1, 0.12, 0.14],
            cell_shape=(8, 7, 6),
        )
        blocked = np.zeros(grid.cell_shape, dtype=bool)
        blocked[3:5, 3:5, 2:4] = True
        domain = build_connected_domain(
            grid, blocked, seed_cells=[(0, 0, 0)]
        )
        system = poisson.build_poisson_system(grid, domain)
        solution = poisson.solve_poisson_reference(system, tolerance=1.0e-12)
        field = poisson.TrilinearPoissonField(
            grid, solution.values, domain=domain
        )
        free = field.query([0.15, 0.18, 0.21])
        occupied = field.query([0.35, 0.42, 0.35])
        self.assertTrue(solution.converged, solution.message)
        self.assertTrue(free.valid)
        self.assertGreater(free.value, 0.0)
        self.assertFalse(occupied.valid)
        self.assertEqual(
            occupied.reason, poisson.QueryInvalidReason.INVALID_CELL
        )

    def test_red_black_sor_agrees_with_reference(self):
        grid = TinyGrid(
            lower=np.asarray([-0.2, 0.1, 0.4]),
            upper=np.asarray([0.9, 1.6, 1.0]),
            vertex_shape=(12, 10, 9),
        )
        domain = rectangular_domain(grid.vertex_shape)
        x, y, z = grid_vertices(grid)
        forcing = 0.8 + 0.2 * x - 0.1 * y + 0.05 * z
        system = poisson.build_poisson_system(grid, domain, forcing=forcing)
        reference = poisson.solve_poisson_reference(system, tolerance=1.0e-12)
        conjugate_gradient = poisson.solve_poisson_reference(
            system,
            method="cg",
            tolerance=1.0e-11,
            max_iterations=5000,
        )
        sor = poisson.solve_poisson_sor(
            system,
            omega=1.65,
            tolerance=1.0e-10,
            max_iterations=10000,
            check_every=2,
        )
        self.assertTrue(reference.converged, reference.message)
        self.assertTrue(conjugate_gradient.converged, conjugate_gradient.message)
        self.assertGreater(conjugate_gradient.iterations, 0)
        self.assertLess(
            np.max(np.abs(conjugate_gradient.values - reference.values)),
            2.0e-11,
        )
        self.assertTrue(sor.converged, sor.message)
        self.assertLess(
            np.max(np.abs(sor.values - reference.values)),
            2.0e-10,
        )
        # The registered stopping test is backward error.  RHS-relative
        # residual is retained as a debug quantity and need not equal it.
        self.assertLess(sor.diagnostics.relative_residual_linf, 5.0e-9)
        self.assertLess(sor.diagnostics.backward_error_linf, 1.1e-10)
        self.assertEqual(sor.diagnostics.boundary_linf, 0.0)
        with self.assertRaisesRegex(ValueError, "1 < omega < 2"):
            poisson.solve_poisson_sor(system, omega=1.0, max_iterations=1)

    def test_sor_does_not_call_sign_invalid_initial_field_converged(self):
        grid = TinyGrid(np.zeros(3), np.ones(3), (5, 5, 5))
        domain = rectangular_domain(grid.vertex_shape)
        system = poisson.build_poisson_system(grid, domain, forcing=1.0)
        result = poisson.solve_poisson_sor(
            system,
            initial=-1.0,
            omega=1.7,
            tolerance=1.0,
            max_iterations=0,
        )
        self.assertFalse(result.converged)
        self.assertFalse(result.diagnostics.passed)
        self.assertGreater(result.diagnostics.sign_violation_count, 0)

    def test_boundary_mask_is_checked_fail_closed(self):
        grid = TinyGrid(np.zeros(3), np.ones(3), (5, 5, 5))
        domain = rectangular_domain(grid.vertex_shape)
        bad_boundary = domain.boundary_vertex_mask.copy()
        bad_boundary[0, 2, 2] = False
        bad = TinyDomain(
            domain.interior_mask,
            bad_boundary,
            domain.valid_vertex_mask,
        )
        with self.assertRaisesRegex(ValueError, "omits required stencil vertex"):
            poisson.build_poisson_system(grid, bad)
        with self.assertRaisesRegex(TypeError, "boolean dtype"):
            poisson.build_poisson_system(
                grid, domain.interior_mask.astype(np.int8)
            )
        with self.assertRaisesRegex(ValueError, "finite values"):
            poisson.build_poisson_system(grid, domain, forcing=np.nan)

    def test_diagnostics_detect_boundary_and_sign_violations(self):
        grid = TinyGrid(np.zeros(3), np.ones(3), (6, 6, 6))
        domain = rectangular_domain(grid.vertex_shape)
        system = poisson.build_poisson_system(grid, domain, forcing=1.0)
        solution = poisson.solve_poisson_reference(system).values
        corrupted = solution.copy()
        corrupted[0, 2, 2] = 0.03
        corrupted[2, 2, 2] = -0.01
        diagnostics = poisson.poisson_diagnostics(system, corrupted)
        unknowns = system.unknowns_from_values(corrupted)
        residual = system.matrix.dot(unknowns) - system.rhs
        matrix_linf = np.max(
            np.asarray(abs(system.matrix).sum(axis=1)).reshape(-1)
        )
        expected_backward_error = np.max(np.abs(residual)) / (
            matrix_linf * np.max(np.abs(unknowns))
            + np.max(np.abs(system.rhs))
        )
        self.assertFalse(diagnostics.passed)
        self.assertAlmostEqual(diagnostics.boundary_linf, 0.03)
        self.assertGreater(diagnostics.sign_violation_count, 0)
        self.assertGreater(diagnostics.residual_linf, 0.0)
        self.assertGreater(diagnostics.backward_error_linf, 0.0)
        self.assertAlmostEqual(
            diagnostics.backward_error_linf,
            expected_backward_error,
            places=15,
        )
        for field, invalid in (
            ("residual_tolerance", float("inf")),
            ("boundary_tolerance", -1.0),
            ("sign_tolerance", float("nan")),
        ):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                poisson.poisson_diagnostics(
                    system, solution, **{field: invalid}
                )

    def test_physical_scale_invariance_and_dimensionless_diagnostics(self):
        shape = (8, 9, 7)
        base_grid = TinyGrid(np.zeros(3), np.asarray([1.0, 1.4, 0.8]), shape)
        scale = 3.7
        scaled_grid = TinyGrid(
            base_grid.lower * scale,
            base_grid.upper * scale,
            shape,
        )
        domain = rectangular_domain(shape)
        base_system = poisson.build_poisson_system(base_grid, domain, forcing=1.0)
        scaled_system = poisson.build_poisson_system(
            scaled_grid, domain, forcing=1.0
        )
        base = poisson.solve_poisson_reference(base_system, tolerance=1.0e-12)
        scaled = poisson.solve_poisson_reference(
            scaled_system, tolerance=1.0e-12
        )
        self.assertTrue(base.converged and scaled.converged)
        self.assertLess(
            np.max(np.abs(scaled.values - scale * scale * base.values)),
            5.0e-13,
        )
        self.assertLess(base.diagnostics.relative_residual_linf, 2.0e-13)
        self.assertLess(scaled.diagnostics.relative_residual_linf, 2.0e-13)
        self.assertLess(base.diagnostics.backward_error_linf, 2.0e-15)
        self.assertLess(scaled.diagnostics.backward_error_linf, 2.0e-15)
        self.assertEqual(base.diagnostics.sign_violation_count, 0)
        self.assertEqual(scaled.diagnostics.sign_violation_count, 0)
        base_field = poisson.TrilinearPoissonField(base_grid, base.values)
        scaled_field = poisson.TrilinearPoissonField(scaled_grid, scaled.values)
        point = np.asarray([0.43, 0.72, 0.31])
        base_query = base_field.query(point)
        scaled_query = scaled_field.query(point * scale)
        self.assertTrue(base_query.valid and scaled_query.valid)
        self.assertAlmostEqual(
            scaled_query.value,
            scale * scale * base_query.value,
            places=12,
        )
        np.testing.assert_allclose(
            scaled_query.gradient,
            scale * np.asarray(base_query.gradient),
            rtol=2.0e-12,
            atol=2.0e-12,
        )


@unittest.skipUnless(NUMERICS_AVAILABLE, "NumPy and SciPy are optional locally")
class TrilinearPoissonFieldTest(unittest.TestCase):
    @staticmethod
    def polynomial(point):
        x, y, z = point
        return (
            0.4
            + 1.2 * x
            - 0.7 * y
            + 0.3 * z
            + 0.8 * x * y
            - 0.4 * x * z
            + 0.6 * y * z
            + 0.25 * x * y * z
        )

    @staticmethod
    def gradient(point):
        x, y, z = point
        return np.asarray(
            [
                1.2 + 0.8 * y - 0.4 * z + 0.25 * y * z,
                -0.7 + 0.8 * x + 0.6 * z + 0.25 * x * z,
                0.3 - 0.4 * x + 0.6 * y + 0.25 * x * y,
            ]
        )

    def make_field(self):
        grid = TinyGrid(
            lower=np.asarray([-1.1, 0.3, 2.0]),
            upper=np.asarray([0.7, 1.6, 3.4]),
            vertex_shape=(6, 7, 5),
        )
        x, y, z = grid_vertices(grid)
        values = self.polynomial((x, y, z))
        return grid, poisson.TrilinearPoissonField(grid, values)

    def test_field_certificate_arrays_are_independent_contiguous_and_read_only(self):
        lower = np.asarray([-1.1, 0.3, 2.0])
        upper = np.asarray([0.7, 1.6, 3.4])
        grid = TinyGrid(lower=lower, upper=upper, vertex_shape=(6, 7, 5))
        x, y, z = grid_vertices(grid)
        values = np.asfortranarray(self.polynomial((x, y, z)))
        valid_vertices = np.ones(grid.vertex_shape, dtype=bool, order="F")
        cell_shape = tuple(item - 1 for item in grid.vertex_shape)
        valid_cells = np.ones(cell_shape, dtype=bool, order="F")
        field = poisson.TrilinearPoissonField(
            grid,
            values,
            valid_mask=valid_vertices,
            valid_cell_mask=valid_cells,
        )
        expected_value = float(field.values[1, 1, 1])

        values[:] = -100.0
        valid_vertices[:] = False
        valid_cells[:] = False
        lower[:] = -100.0
        upper[:] = 100.0
        self.assertEqual(float(field.values[1, 1, 1]), expected_value)
        self.assertTrue(np.all(field.valid_mask))
        self.assertTrue(np.all(field.valid_cell_mask))
        for array in (
            field.lower,
            field.upper,
            field.spacing,
            field.values,
            field.valid_mask,
            field.valid_cell_mask,
        ):
            self.assertTrue(array.flags.c_contiguous)
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flat[0] = array.flat[0]
            with self.assertRaises(ValueError):
                array.setflags(write=True)

    def test_trilinear_polynomial_and_physical_gradient_are_exact(self):
        _, field = self.make_field()
        point = np.asarray([-0.37, 0.91, 2.73])
        query = field.query(point)
        self.assertTrue(query.valid)
        self.assertIsNone(query.reason)
        self.assertAlmostEqual(query.value, self.polynomial(point), places=13)
        np.testing.assert_allclose(
            query.gradient,
            self.gradient(point),
            rtol=2.0e-13,
            atol=2.0e-13,
        )

    def test_analytic_gradient_matches_finite_difference(self):
        _, field = self.make_field()
        point = np.asarray([0.11, 1.03, 2.38])
        query = field.query(point)
        self.assertTrue(query.valid)
        finite_difference = []
        delta = 1.0e-6
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = delta
            plus = field.query(point + offset)
            minus = field.query(point - offset)
            self.assertTrue(plus.valid and minus.valid)
            finite_difference.append((plus.value - minus.value) / (2.0 * delta))
        np.testing.assert_allclose(
            query.gradient,
            finite_difference,
            rtol=2.0e-9,
            atol=2.0e-9,
        )

    def test_queries_report_typed_invalid_reasons(self):
        grid, field = self.make_field()
        malformed = field.query([1.0, 2.0])
        self.assertEqual(
            malformed.reason, poisson.QueryInvalidReason.INVALID_POINT_SHAPE
        )
        nonfinite = field.query([0.0, np.nan, 2.5])
        self.assertEqual(
            nonfinite.reason, poisson.QueryInvalidReason.NONFINITE_POINT
        )
        outside = field.query([grid.upper[0] + 0.01, 0.8, 2.5])
        self.assertEqual(outside.reason, poisson.QueryInvalidReason.OUTSIDE_GRID)

        valid = np.ones(grid.vertex_shape, dtype=bool)
        valid[2, 2, 2] = False
        masked = poisson.TrilinearPoissonField(grid, field.values, valid_mask=valid)
        masked_point = grid.lower + grid.spacing * np.asarray([1.7, 1.7, 1.7])
        invalid_cell = masked.query(masked_point)
        self.assertEqual(
            invalid_cell.reason, poisson.QueryInvalidReason.INVALID_CELL
        )

        valid_cells = np.ones(tuple(item - 1 for item in grid.vertex_shape), dtype=bool)
        valid_cells[1, 1, 1] = False
        cell_masked = poisson.TrilinearPoissonField(
            grid,
            field.values,
            valid_cell_mask=valid_cells,
        )
        invalid_occupied_cell = cell_masked.query(masked_point)
        self.assertEqual(
            invalid_occupied_cell.reason,
            poisson.QueryInvalidReason.INVALID_CELL,
        )
        face_point = grid.lower + grid.spacing * np.asarray([2.0, 1.5, 1.5])
        invalid_closed_face = cell_masked.query(face_point)
        self.assertEqual(
            invalid_closed_face.reason,
            poisson.QueryInvalidReason.INVALID_CELL,
        )

        smooth_mask = poisson.TrilinearPoissonField(grid, field.values)
        nondifferentiable_face = smooth_mask.query(face_point)
        self.assertEqual(
            nondifferentiable_face.reason,
            poisson.QueryInvalidReason.NONDIFFERENTIABLE_INTERNAL_FACE,
        )

        corrupt_values = field.values.copy()
        corrupt_values[2, 2, 2] = np.inf
        corrupt = poisson.TrilinearPoissonField(grid, corrupt_values)
        nonfinite_cell = corrupt.query(masked_point)
        self.assertEqual(
            nonfinite_cell.reason,
            poisson.QueryInvalidReason.NONFINITE_CELL_VALUES,
        )

        zero_values = field.values.copy()
        zero_values[1:3, 1:3, 1:3] = 0.0
        zero_cell = poisson.TrilinearPoissonField(grid, zero_values).query(
            masked_point
        )
        self.assertEqual(
            zero_cell.reason,
            poisson.QueryInvalidReason.NONREGULAR_ZERO_CELL,
        )

    def test_upper_boundary_uses_last_cell_without_extrapolation(self):
        grid, field = self.make_field()
        query = field.query(grid.upper)
        self.assertTrue(query.valid)
        self.assertEqual(query.cell_index, tuple(item - 2 for item in grid.vertex_shape))
        self.assertEqual(query.local_coordinates, (1.0, 1.0, 1.0))
        self.assertEqual(query.outer_boundary_clearance_m, 0.0)
        self.assertAlmostEqual(query.value, self.polynomial(grid.upper), places=12)

    def test_query_reports_physical_outer_boundary_clearance(self):
        grid, field = self.make_field()
        point = grid.lower + np.asarray([0.31, 0.42, 0.53])
        query = field.query(point)
        self.assertTrue(query.valid)
        expected = float(
            np.min(np.concatenate((point - grid.lower, grid.upper - point)))
        )
        self.assertAlmostEqual(query.outer_boundary_clearance_m, expected)
        outside = field.query(grid.upper + np.asarray([0.1, 0.0, 0.0]))
        self.assertIsNone(outside.outer_boundary_clearance_m)


if __name__ == "__main__":
    unittest.main()
