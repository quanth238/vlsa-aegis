"""Dependency-lazy finite-difference Poisson fields for static CBFs.

The module deliberately has no import-time NumPy or SciPy dependency.  The
public functions import them only when numerical work is requested, which
keeps the released VLSA / AEGIS paths unchanged when this opt-in experiment is
not used.

The discrete convention is

    - Laplacian(u) = forcing

on ``domain.interior_vertices`` (or an accepted mask alias), with Dirichlet
values on every axis-adjacent non-interior vertex.  Grid arrays use
``(x, y, z)``
axis order.  Grid specifications are duck typed and are expected to expose
``lower``, ``vertex_shape``, and either ``spacing`` or ``upper``.

With physical coordinates measured in metres and unit forcing, ``u`` has
units of square metres and its world-coordinate gradient has units of metres.
The field is a safety function, not a metric distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import itertools
import math
from typing import Any, Optional, Sequence, Tuple


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised without extras
        raise RuntimeError("Poisson numerics require NumPy") from exc
    return np


def _scipy_sparse() -> Tuple[Any, Any]:
    try:
        import scipy.sparse as sparse
        import scipy.sparse.linalg as sparse_linalg
    except ImportError as exc:  # pragma: no cover - exercised without extras
        raise RuntimeError("Poisson reference solves require SciPy") from exc
    return sparse, sparse_linalg


def _first_attribute(value: Any, names: Sequence[str]) -> Any:
    for name in names:
        if hasattr(value, name):
            result = getattr(value, name)
            if result is not None:
                return result
    return None


@dataclass(frozen=True)
class _GridData:
    lower: Any
    upper: Any
    spacing: Any
    shape: Tuple[int, int, int]


def _read_grid(grid: Any) -> _GridData:
    """Read a minimal 3-D vertex grid without depending on its concrete type."""

    np = _numpy()
    raw_shape = _first_attribute(grid, ("vertex_shape", "shape"))
    if raw_shape is None:
        raw_cell_shape = _first_attribute(grid, ("cell_shape",))
        if raw_cell_shape is not None:
            raw_shape = tuple(int(item) + 1 for item in raw_cell_shape)
    raw_lower = _first_attribute(grid, ("lower", "origin", "minimum"))
    raw_spacing = _first_attribute(grid, ("spacing", "vertex_spacing"))
    raw_upper = _first_attribute(grid, ("upper", "maximum"))
    if raw_shape is None or raw_lower is None:
        raise ValueError(
            "grid must expose vertex_shape (or shape) and lower (or origin)"
        )
    shape_values = tuple(int(item) for item in raw_shape)
    if len(shape_values) != 3 or any(item < 2 for item in shape_values):
        raise ValueError("grid vertex_shape must contain three values >= 2")
    lower = np.asarray(raw_lower, dtype=float)
    if lower.shape != (3,) or not np.all(np.isfinite(lower)):
        raise ValueError("grid lower bound must be a finite length-three vector")
    if raw_spacing is None:
        if raw_upper is None:
            raise ValueError("grid must expose spacing or upper")
        upper = np.asarray(raw_upper, dtype=float)
        if upper.shape != (3,) or not np.all(np.isfinite(upper)):
            raise ValueError("grid upper bound must be a finite length-three vector")
        spacing = (upper - lower) / (
            np.asarray(shape_values, dtype=float) - 1.0
        )
    else:
        spacing = np.asarray(raw_spacing, dtype=float)
        if spacing.shape == ():
            spacing = np.repeat(spacing, 3)
        if spacing.shape != (3,):
            raise ValueError("grid spacing must be a scalar or length-three vector")
        upper = lower + spacing * (
            np.asarray(shape_values, dtype=float) - 1.0
        )
        if raw_upper is not None:
            declared_upper = np.asarray(raw_upper, dtype=float)
            if declared_upper.shape != (3,) or not np.allclose(
                declared_upper,
                upper,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise ValueError("grid upper, spacing, and vertex_shape disagree")
            upper = declared_upper
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0.0):
        raise ValueError("grid spacing must be finite and strictly positive")
    return _GridData(
        lower=lower.copy(),
        upper=upper.copy(),
        spacing=spacing.copy(),
        shape=shape_values,
    )


def _as_full_array(value: Any, shape: Tuple[int, int, int], name: str) -> Any:
    np = _numpy()
    array = np.asarray(value, dtype=float)
    if array.shape == ():
        scalar = float(array)
        if not math.isfinite(scalar):
            raise ValueError("{} must contain only finite values".format(name))
        return np.full(shape, scalar, dtype=float)
    if array.shape != shape:
        raise ValueError("{} must be scalar or have grid shape {}".format(name, shape))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} must contain only finite values".format(name))
    return array.copy()


def _read_mask(value: Any, shape: Tuple[int, int, int], name: str) -> Any:
    np = _numpy()
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError("{} must have grid shape {}".format(name, shape))
    if array.dtype != np.bool_:
        raise TypeError("{} must have boolean dtype".format(name))
    return array.copy()


def _readonly_c_copy(value: Any, *, dtype: Any = None) -> Any:
    """Return an independent C-order array backed by immutable bytes.

    Merely clearing NumPy's ``WRITEABLE`` flag is reversible for an owning
    array.  A bytes-backed view makes attempts to re-enable writes fail too,
    which is required for a durable numerical certificate.
    """

    np = _numpy()
    contiguous = np.array(value, dtype=dtype, order="C", copy=True)
    immutable_storage = contiguous.tobytes(order="C")
    return np.frombuffer(
        immutable_storage,
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape, order="C")


def _domain_mask(domain: Any, names: Sequence[str]) -> Any:
    if domain is None:
        return None
    return _first_attribute(domain, names)


@dataclass(frozen=True, init=False)
class PoissonSystem:
    """Sparse ``-Laplacian`` system and the arrays needed to audit it."""

    _matrix: Any
    rhs: Any
    lower: Any
    upper: Any
    spacing: Any
    shape: Tuple[int, int, int]
    interior_mask: Any
    boundary_mask: Any
    valid_vertex_mask: Any
    forcing: Any
    boundary_values: Any
    unknown_flat_indices: Any

    def __init__(
        self,
        *,
        matrix: Any,
        rhs: Any,
        lower: Any,
        upper: Any,
        spacing: Any,
        shape: Tuple[int, int, int],
        interior_mask: Any,
        boundary_mask: Any,
        valid_vertex_mask: Any,
        forcing: Any,
        boundary_values: Any,
        unknown_flat_indices: Any,
    ) -> None:
        # Keep the constructor's public ``matrix=`` API while storing the
        # certificate privately.  The public property below returns defensive
        # copies because SciPy permits replacing ``csr.data`` even when the
        # original storage arrays are marked read-only.
        object.__setattr__(self, "_matrix", matrix)
        object.__setattr__(self, "rhs", rhs)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "interior_mask", interior_mask)
        object.__setattr__(self, "boundary_mask", boundary_mask)
        object.__setattr__(self, "valid_vertex_mask", valid_vertex_mask)
        object.__setattr__(self, "forcing", forcing)
        object.__setattr__(self, "boundary_values", boundary_values)
        object.__setattr__(self, "unknown_flat_indices", unknown_flat_indices)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Detach and freeze every numerical array in the field certificate."""

        np = _numpy()
        matrix = self._matrix.copy().tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.sort_indices()
        # A frozen dataclass does not make a SciPy sparse matrix immutable.
        # Freeze all three CSR storage arrays explicitly after taking a copy.
        matrix.data = _readonly_c_copy(matrix.data)
        matrix.indices = _readonly_c_copy(matrix.indices)
        matrix.indptr = _readonly_c_copy(matrix.indptr)
        object.__setattr__(self, "_matrix", matrix)

        for name, dtype in (
            ("rhs", np.float64),
            ("lower", np.float64),
            ("upper", np.float64),
            ("spacing", np.float64),
            ("interior_mask", np.bool_),
            ("boundary_mask", np.bool_),
            ("valid_vertex_mask", np.bool_),
            ("forcing", np.float64),
            ("boundary_values", np.float64),
            ("unknown_flat_indices", np.int64),
        ):
            object.__setattr__(
                self,
                name,
                _readonly_c_copy(getattr(self, name), dtype=dtype),
            )

    @property
    def matrix(self) -> Any:
        """Return a detached read-only CSR view of the certified operator.

        The returned SciPy object itself remains structurally mutable (SciPy
        has no immutable sparse-matrix type), but it owns a copy.  Replacing
        its ``data``, ``indices``, or ``indptr`` therefore cannot alter this
        ``PoissonSystem`` or later diagnostics/solves.
        """

        matrix = self._matrix.copy().tocsr(copy=True)
        matrix.data = _readonly_c_copy(matrix.data)
        matrix.indices = _readonly_c_copy(matrix.indices)
        matrix.indptr = _readonly_c_copy(matrix.indptr)
        return matrix

    @property
    def unknown_count(self) -> int:
        return int(self.unknown_flat_indices.size)

    def values_from_unknowns(self, unknown_values: Any) -> Any:
        """Reconstruct the full vertex field from an unknown-vector solution."""

        np = _numpy()
        vector = np.asarray(unknown_values, dtype=float)
        if vector.shape != (self.unknown_count,):
            raise ValueError(
                "unknown solution must have shape ({},)".format(self.unknown_count)
            )
        values = self.boundary_values.copy()
        values.ravel(order="C")[self.unknown_flat_indices] = vector
        return values

    def unknowns_from_values(self, values: Any) -> Any:
        np = _numpy()
        array = np.asarray(values, dtype=float)
        if array.shape != self.shape:
            raise ValueError("field values must have grid shape {}".format(self.shape))
        return array.ravel(order="C")[self.unknown_flat_indices].copy()


def build_poisson_system(
    grid: Any,
    domain: Any,
    forcing: Any = 1.0,
    boundary_values: Any = 0.0,
) -> PoissonSystem:
    """Build a sparse anisotropic finite-difference ``-Laplacian`` system.

    ``domain`` may itself be a boolean interior mask, or it may expose
    ``interior_vertices`` / ``interior_mask`` / ``solve_mask``.  The sibling
    voxel module's canonical names are ``active_vertices``,
    ``boundary_vertices``, and ``interior_vertices``; descriptive mask aliases
    are accepted for lightweight callers.  A supplied boundary mask is
    checked fail-closed: it must contain every non-interior vertex used by the
    six-point stencil.
    """

    np = _numpy()
    sparse, _ = _scipy_sparse()
    grid_data = _read_grid(grid)
    shape = grid_data.shape

    if isinstance(domain, np.ndarray) or (
        hasattr(domain, "shape")
        and not hasattr(domain, "interior_vertices")
        and not hasattr(domain, "interior_mask")
        and not hasattr(domain, "solve_mask")
    ):
        interior_source = domain
    else:
        interior_source = _domain_mask(
            domain, ("interior_vertices", "interior_mask", "solve_mask")
        )
    if interior_source is None:
        raise ValueError(
            "domain must be an interior mask or expose interior_vertices"
        )
    interior = _read_mask(interior_source, shape, "interior_mask")

    # Every interior point needs all six stencil neighbors in the grid.  This
    # makes the outer grid faces explicit Dirichlet boundary vertices instead
    # of silently changing the operator near an edge.
    if (
        np.any(interior[0, :, :])
        or np.any(interior[-1, :, :])
        or np.any(interior[:, 0, :])
        or np.any(interior[:, -1, :])
        or np.any(interior[:, :, 0])
        or np.any(interior[:, :, -1])
    ):
        raise ValueError("interior_mask may not include an outer grid face")

    required_boundary = np.zeros(shape, dtype=bool)
    for axis in range(3):
        for offset in (-1, 1):
            shifted = np.roll(interior, shift=offset, axis=axis)
            required_boundary |= shifted & ~interior
    # np.roll wraps, but an interior-free outer face guarantees that wrapped
    # locations cannot be selected as a required neighbor.  Clear them anyway
    # so the invariant remains visible if this implementation is changed.
    required_boundary &= ~interior

    supplied_boundary = _domain_mask(
        domain, ("boundary_vertices", "boundary_vertex_mask", "boundary_mask")
    )
    if supplied_boundary is None:
        boundary = required_boundary
    else:
        boundary = _read_mask(
            supplied_boundary, shape, "boundary_vertex_mask"
        )
        missing = required_boundary & ~boundary
        if np.any(missing):
            first = tuple(int(item) for item in np.argwhere(missing)[0])
            raise ValueError(
                "boundary_vertex_mask omits required stencil vertex {}".format(
                    first
                )
            )
        boundary |= required_boundary
    if np.any(boundary & interior):
        raise ValueError("interior and boundary masks must be disjoint")

    valid_source = _domain_mask(
        domain,
        (
            "active_vertices",
            "valid_vertex_mask",
            "free_vertex_mask",
            "valid_mask",
        ),
    )
    valid = (
        np.ones(shape, dtype=bool)
        if valid_source is None
        else _read_mask(valid_source, shape, "valid_vertex_mask")
    )
    if np.any(interior & ~valid) or np.any(boundary & ~valid):
        raise ValueError("interior and boundary vertices must be valid vertices")

    forcing_array = _as_full_array(forcing, shape, "forcing")
    boundary_array = _as_full_array(
        boundary_values, shape, "boundary_values"
    )
    unknown_flat = np.flatnonzero(interior.ravel(order="C"))
    unknown_lookup = np.full(int(np.prod(shape)), -1, dtype=np.int64)
    unknown_lookup[unknown_flat] = np.arange(unknown_flat.size, dtype=np.int64)
    coefficients = 1.0 / np.square(grid_data.spacing)
    diagonal = 2.0 * float(np.sum(coefficients))

    rows = []
    columns = []
    data = []
    rhs = forcing_array.ravel(order="C")[unknown_flat].astype(float, copy=True)
    for row, flat_index in enumerate(unknown_flat):
        index = list(np.unravel_index(int(flat_index), shape, order="C"))
        rows.append(row)
        columns.append(row)
        data.append(diagonal)
        for axis in range(3):
            coefficient = float(coefficients[axis])
            for offset in (-1, 1):
                neighbor = list(index)
                neighbor[axis] += offset
                neighbor_flat = int(
                    np.ravel_multi_index(tuple(neighbor), shape, order="C")
                )
                neighbor_unknown = int(unknown_lookup[neighbor_flat])
                if neighbor_unknown >= 0:
                    rows.append(row)
                    columns.append(neighbor_unknown)
                    data.append(-coefficient)
                else:
                    rhs[row] += coefficient * float(
                        boundary_array[tuple(neighbor)]
                    )

    matrix = sparse.csr_matrix(
        (np.asarray(data, dtype=float), (rows, columns)),
        shape=(unknown_flat.size, unknown_flat.size),
        dtype=float,
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    return PoissonSystem(
        matrix=matrix,
        rhs=rhs,
        lower=grid_data.lower,
        upper=grid_data.upper,
        spacing=grid_data.spacing,
        shape=shape,
        interior_mask=interior,
        boundary_mask=boundary,
        valid_vertex_mask=valid,
        forcing=forcing_array,
        boundary_values=boundary_array,
        unknown_flat_indices=unknown_flat,
    )


@dataclass(frozen=True)
class PoissonDiagnostics:
    finite: bool
    unknown_count: int
    residual_linf: float
    residual_l2: float
    relative_residual_linf: float
    relative_residual_l2: float
    backward_error_linf: float
    boundary_linf: float
    interior_min: Optional[float]
    interior_max: Optional[float]
    expected_sign: Optional[str]
    sign_violation_count: int
    sign_violation_linf: float
    passed: bool


def poisson_diagnostics(
    system: PoissonSystem,
    values: Any,
    expected_sign: Optional[str] = "auto",
    residual_tolerance: float = 1.0e-8,
    boundary_tolerance: float = 1.0e-12,
    sign_tolerance: float = 1.0e-12,
) -> PoissonDiagnostics:
    """Audit backward error, Dirichlet values, and maximum-principle sign.

    The registered scale-safe residual is the normwise backward error

    ``rho_inf = ||A x - b||inf / (||A||inf ||x||inf + ||b||inf)``.

    Raw and RHS-relative residuals remain available for debugging, but
    ``passed`` uses ``rho_inf``.
    """

    np = _numpy()
    for value, name in (
        (residual_tolerance, "residual_tolerance"),
        (boundary_tolerance, "boundary_tolerance"),
        (sign_tolerance, "sign_tolerance"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("{} must be finite and nonnegative".format(name))
    array = np.asarray(values, dtype=float)
    if array.shape != system.shape:
        raise ValueError("field values must have grid shape {}".format(system.shape))
    if expected_sign not in ("auto", "nonnegative", "nonpositive", None):
        raise ValueError(
            "expected_sign must be auto, nonnegative, nonpositive, or None"
        )
    finite = bool(np.all(np.isfinite(array)))
    if system.unknown_count and finite:
        unknowns = system.unknowns_from_values(array)
        residual = system._matrix.dot(unknowns) - system.rhs
        residual_linf = float(np.max(np.abs(residual)))
        residual_l2 = float(np.linalg.norm(residual))
        rhs_linf = float(np.max(np.abs(system.rhs)))
        rhs_l2 = float(np.linalg.norm(system.rhs))
        relative_linf = residual_linf / max(rhs_linf, 1.0e-300)
        relative_l2 = residual_l2 / max(rhs_l2, 1.0e-300)
        matrix_row_sums = np.asarray(
            abs(system._matrix).sum(axis=1), dtype=float
        ).reshape(-1)
        matrix_linf = float(np.max(matrix_row_sums))
        unknown_linf = float(np.max(np.abs(unknowns)))
        backward_denominator = matrix_linf * unknown_linf + rhs_linf
        if backward_denominator == 0.0:
            backward_error = 0.0 if residual_linf == 0.0 else float("inf")
        else:
            backward_error = residual_linf / backward_denominator
        interior_values = array[system.interior_mask]
        interior_min = float(np.min(interior_values))
        interior_max = float(np.max(interior_values))
    elif system.unknown_count:
        residual_linf = float("inf")
        residual_l2 = float("inf")
        relative_linf = float("inf")
        relative_l2 = float("inf")
        backward_error = float("inf")
        interior_values = array[system.interior_mask]
        interior_min = None
        interior_max = None
    else:
        residual_linf = 0.0
        residual_l2 = 0.0
        relative_linf = 0.0
        relative_l2 = 0.0
        backward_error = 0.0
        interior_values = np.asarray([], dtype=float)
        interior_min = None
        interior_max = None

    if np.any(system.boundary_mask):
        boundary_error = np.abs(
            array[system.boundary_mask]
            - system.boundary_values[system.boundary_mask]
        )
        boundary_linf = float(np.max(boundary_error))
        boundary_targets = system.boundary_values[system.boundary_mask]
    else:
        boundary_linf = 0.0
        boundary_targets = np.asarray([], dtype=float)

    resolved_sign = expected_sign
    if expected_sign == "auto":
        forcing_values = system.forcing[system.interior_mask]
        if (
            np.all(forcing_values >= -sign_tolerance)
            and np.all(boundary_targets >= -sign_tolerance)
        ):
            resolved_sign = "nonnegative"
        elif (
            np.all(forcing_values <= sign_tolerance)
            and np.all(boundary_targets <= sign_tolerance)
        ):
            resolved_sign = "nonpositive"
        else:
            resolved_sign = None

    if resolved_sign == "nonnegative" and interior_values.size:
        violations = np.maximum(-interior_values - sign_tolerance, 0.0)
    elif resolved_sign == "nonpositive" and interior_values.size:
        violations = np.maximum(interior_values - sign_tolerance, 0.0)
    else:
        violations = np.asarray([], dtype=float)
    sign_count = int(np.count_nonzero(violations > 0.0))
    sign_linf = float(np.max(violations)) if violations.size else 0.0

    passed = bool(
        finite
        and backward_error <= residual_tolerance
        and boundary_linf <= boundary_tolerance
        and sign_count == 0
    )
    return PoissonDiagnostics(
        finite=finite,
        unknown_count=system.unknown_count,
        residual_linf=residual_linf,
        residual_l2=residual_l2,
        relative_residual_linf=relative_linf,
        relative_residual_l2=relative_l2,
        backward_error_linf=backward_error,
        boundary_linf=boundary_linf,
        interior_min=interior_min,
        interior_max=interior_max,
        expected_sign=resolved_sign,
        sign_violation_count=sign_count,
        sign_violation_linf=sign_linf,
        passed=passed,
    )


@dataclass(frozen=True)
class PoissonSolveResult:
    values: Any
    converged: bool
    method: str
    iterations: int
    backward_error_target: float
    # Compatibility alias; this is dimensionless and equals the registered
    # backward-error target, not a raw-residual threshold.
    residual_target: float
    diagnostics: PoissonDiagnostics
    message: str


def solve_poisson_reference(
    system: PoissonSystem,
    method: str = "spsolve",
    tolerance: float = 1.0e-11,
    max_iterations: Optional[int] = None,
) -> PoissonSolveResult:
    """Solve with SciPy's independent sparse direct or conjugate-gradient path."""

    np = _numpy()
    _, sparse_linalg = _scipy_sparse()
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and strictly positive")
    normalized_method = str(method).lower()
    if normalized_method not in ("spsolve", "cg"):
        raise ValueError("reference method must be 'spsolve' or 'cg'")
    target = tolerance
    if system.unknown_count == 0:
        values = system.boundary_values.copy()
        diagnostics = poisson_diagnostics(
            system, values, residual_tolerance=tolerance
        )
        return PoissonSolveResult(
            values=values,
            converged=diagnostics.passed,
            method=normalized_method,
            iterations=0,
            backward_error_target=target,
            residual_target=target,
            diagnostics=diagnostics,
            message=(
                "empty interior"
                if diagnostics.passed
                else "empty interior failed field diagnostics"
            ),
        )

    iterations = 1 if normalized_method == "spsolve" else 0
    info = 0
    if normalized_method == "spsolve":
        unknowns = sparse_linalg.spsolve(system._matrix, system.rhs)
    else:
        counter = [0]

        def callback(_: Any) -> None:
            counter[0] += 1

        kwargs = {
            "maxiter": max_iterations,
            "callback": callback,
        }
        # SciPy 1.12 renamed ``tol`` to ``rtol``.  Supporting both keeps the
        # source usable in the H100 environment (1.10) and current dev hosts.
        try:
            unknowns, info = sparse_linalg.cg(
                system._matrix,
                system.rhs,
                rtol=tolerance,
                atol=0.0,
                **kwargs
            )
        except TypeError:  # pragma: no cover - depends on installed SciPy
            unknowns, info = sparse_linalg.cg(
                system._matrix,
                system.rhs,
                tol=tolerance,
                atol=0.0,
                **kwargs
            )
        iterations = counter[0]
    values = system.values_from_unknowns(np.asarray(unknowns, dtype=float))
    diagnostics = poisson_diagnostics(
        system,
        values,
        residual_tolerance=tolerance,
    )
    finite = bool(np.all(np.isfinite(unknowns)))
    converged = bool(
        finite
        and info == 0
        and diagnostics.passed
    )
    if not finite:
        message = "solver returned non-finite values"
    elif info < 0:
        message = "CG failed with illegal input or breakdown ({})".format(info)
    elif info > 0:
        message = "CG reached its iteration limit ({})".format(info)
    elif converged:
        message = "converged"
    else:
        message = "solver backward error exceeded the registered tolerance"
    return PoissonSolveResult(
        values=values,
        converged=converged,
        method=normalized_method,
        iterations=iterations,
        backward_error_target=target,
        residual_target=target,
        diagnostics=diagnostics,
        message=message,
    )


def solve_poisson_sor(
    system: PoissonSystem,
    omega: float = 1.7,
    tolerance: float = 1.0e-9,
    max_iterations: int = 20000,
    initial: Any = 0.0,
    check_every: int = 1,
) -> PoissonSolveResult:
    """Solve by vectorized red-black SOR using the physical grid spacings."""

    np = _numpy()
    if not math.isfinite(omega) or not (1.0 < omega < 2.0):
        raise ValueError("production SOR omega must satisfy 1 < omega < 2")
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and strictly positive")
    if int(max_iterations) != max_iterations or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer")
    if int(check_every) != check_every or check_every <= 0:
        raise ValueError("check_every must be a positive integer")
    max_iterations = int(max_iterations)
    check_every = int(check_every)

    values = system.boundary_values.copy()
    initial_array = np.asarray(initial, dtype=float)
    if initial_array.shape == ():
        initial_scalar = float(initial_array)
        if not math.isfinite(initial_scalar):
            raise ValueError("initial field contains a non-finite value")
        values[system.interior_mask] = initial_scalar
    elif initial_array.shape == system.shape:
        if not np.all(np.isfinite(initial_array)):
            raise ValueError("initial field contains non-finite values")
        values[system.interior_mask] = initial_array[system.interior_mask]
    else:
        raise ValueError(
            "initial must be scalar or have grid shape {}".format(system.shape)
        )

    target = tolerance
    if system.unknown_count:
        matrix_row_sums = np.asarray(
            abs(system._matrix).sum(axis=1), dtype=float
        ).reshape(-1)
        matrix_linf = float(np.max(matrix_row_sums))
        rhs_linf = float(np.max(np.abs(system.rhs)))
    else:
        matrix_linf = 0.0
        rhs_linf = 0.0

    def residual_and_backward_error() -> Tuple[float, float]:
        if system.unknown_count == 0:
            return 0.0, 0.0
        unknowns = values.ravel(order="C")[system.unknown_flat_indices]
        residual = system._matrix.dot(unknowns) - system.rhs
        residual_value = float(np.max(np.abs(residual)))
        denominator = (
            matrix_linf * float(np.max(np.abs(unknowns))) + rhs_linf
        )
        if denominator == 0.0:
            backward = 0.0 if residual_value == 0.0 else float("inf")
        else:
            backward = residual_value / denominator
        return residual_value, backward

    residual, backward_error = residual_and_backward_error()
    if backward_error <= target:
        diagnostics = poisson_diagnostics(
            system, values, residual_tolerance=tolerance
        )
        return PoissonSolveResult(
            values=values,
            converged=diagnostics.passed,
            method="red_black_sor",
            iterations=0,
            backward_error_target=target,
            residual_target=target,
            diagnostics=diagnostics,
            message=(
                "initial field satisfies all diagnostics"
                if diagnostics.passed
                else "initial field satisfies backward error but fails field diagnostics"
            ),
        )

    inverse_spacing_sq = 1.0 / np.square(system.spacing)
    diagonal = 2.0 * float(np.sum(inverse_spacing_sq))
    core_mask = system.interior_mask[1:-1, 1:-1, 1:-1]
    x_index, y_index, z_index = np.ogrid[
        1 : system.shape[0] - 1,
        1 : system.shape[1] - 1,
        1 : system.shape[2] - 1,
    ]
    parity = (x_index + y_index + z_index) & 1
    center = values[1:-1, 1:-1, 1:-1]
    forcing_core = system.forcing[1:-1, 1:-1, 1:-1]
    iterations = 0
    converged = False
    for sweep in range(1, max_iterations + 1):
        for color in (0, 1):
            candidate = (
                forcing_core
                + inverse_spacing_sq[0]
                * (values[:-2, 1:-1, 1:-1] + values[2:, 1:-1, 1:-1])
                + inverse_spacing_sq[1]
                * (values[1:-1, :-2, 1:-1] + values[1:-1, 2:, 1:-1])
                + inverse_spacing_sq[2]
                * (values[1:-1, 1:-1, :-2] + values[1:-1, 1:-1, 2:])
            ) / diagonal
            update_mask = core_mask & (parity == color)
            center[update_mask] = (
                (1.0 - omega) * center[update_mask]
                + omega * candidate[update_mask]
            )
        iterations = sweep
        if sweep % check_every == 0 or sweep == max_iterations:
            residual, backward_error = residual_and_backward_error()
            if not math.isfinite(backward_error):
                break
            if backward_error <= target:
                # Sign and boundary diagnostics are part of convergence, not
                # optional report-only metadata.
                candidate_diagnostics = poisson_diagnostics(
                    system, values, residual_tolerance=tolerance
                )
                if candidate_diagnostics.passed:
                    converged = True
                    break

    diagnostics = poisson_diagnostics(
        system,
        values,
        residual_tolerance=tolerance,
    )
    # The loop flag and final independently recomputed diagnostics must agree.
    converged = bool(converged and diagnostics.passed)
    if converged:
        message = "converged"
    elif not math.isfinite(backward_error):
        message = "SOR produced a non-finite backward error"
    else:
        message = (
            "SOR reached max_iterations with backward error {:.6g} "
            "(raw residual {:.6g})"
        ).format(
            backward_error, residual
        )
    return PoissonSolveResult(
        values=values,
        converged=converged,
        method="red_black_sor",
        iterations=iterations,
        backward_error_target=target,
        residual_target=target,
        diagnostics=diagnostics,
        message=message,
    )


class QueryInvalidReason(Enum):
    """Typed fail-closed reasons for a field query that cannot be certified."""

    INVALID_POINT_SHAPE = "invalid_point_shape"
    NONFINITE_POINT = "nonfinite_point"
    OUTSIDE_GRID = "outside_grid"
    INVALID_CELL = "invalid_cell"
    NONDIFFERENTIABLE_INTERNAL_FACE = "nondifferentiable_internal_face"
    NONREGULAR_ZERO_CELL = "nonregular_zero_cell"
    NONFINITE_CELL_VALUES = "nonfinite_cell_values"


@dataclass(frozen=True)
class FieldQuery:
    valid: bool
    value: Optional[float]
    gradient: Optional[Tuple[float, float, float]]
    reason: Optional[QueryInvalidReason]
    cell_index: Optional[Tuple[int, int, int]]
    local_coordinates: Optional[Tuple[float, float, float]]
    outer_boundary_clearance_m: Optional[float]


class TrilinearPoissonField:
    """Trilinear ``h [m^2]`` and analytic world gradient ``grad h [m]``."""

    def __init__(
        self,
        grid: Any,
        values: Any,
        valid_mask: Any = None,
        domain: Any = None,
        valid_cell_mask: Any = None,
    ) -> None:
        np = _numpy()
        grid_data = _read_grid(grid)
        array = np.asarray(values, dtype=float)
        if array.shape != grid_data.shape:
            raise ValueError(
                "field values must have grid shape {}".format(grid_data.shape)
            )
        if domain is not None and (
            valid_mask is not None or valid_cell_mask is not None
        ):
            raise ValueError("provide explicit validity masks or domain, not both")
        if domain is not None:
            valid_mask = _domain_mask(
                domain,
                (
                    "active_vertices",
                    "valid_vertex_mask",
                    "free_vertex_mask",
                    "valid_mask",
                ),
            )
            valid_cell_mask = _domain_mask(
                domain, ("component_cells", "component_cell_mask")
            )
        self.shape = grid_data.shape
        self.lower = _readonly_c_copy(grid_data.lower, dtype=np.float64)
        self.upper = _readonly_c_copy(grid_data.upper, dtype=np.float64)
        self.spacing = _readonly_c_copy(grid_data.spacing, dtype=np.float64)
        self.values = _readonly_c_copy(array, dtype=np.float64)
        valid_vertices = (
            np.ones(self.shape, dtype=bool)
            if valid_mask is None
            else _read_mask(valid_mask, self.shape, "valid_mask")
        )
        cell_shape = tuple(item - 1 for item in self.shape)
        valid_cells = (
            np.ones(cell_shape, dtype=bool)
            if valid_cell_mask is None
            else _read_mask(valid_cell_mask, cell_shape, "valid_cell_mask")
        )
        self.valid_mask = _readonly_c_copy(valid_vertices, dtype=np.bool_)
        self.valid_cell_mask = _readonly_c_copy(valid_cells, dtype=np.bool_)

    def _invalid(
        self, reason: QueryInvalidReason
    ) -> FieldQuery:
        return FieldQuery(
            valid=False,
            value=None,
            gradient=None,
            reason=reason,
            cell_index=None,
            local_coordinates=None,
            outer_boundary_clearance_m=None,
        )

    def query(self, point: Any) -> FieldQuery:
        """Return ``(h, grad h)`` or a typed invalid reason, never extrapolate."""

        np = _numpy()
        try:
            position = np.asarray(point, dtype=float)
        except (TypeError, ValueError):
            return self._invalid(QueryInvalidReason.INVALID_POINT_SHAPE)
        if position.shape != (3,):
            return self._invalid(QueryInvalidReason.INVALID_POINT_SHAPE)
        if not np.all(np.isfinite(position)):
            return self._invalid(QueryInvalidReason.NONFINITE_POINT)

        magnitude = max(
            1.0,
            float(np.max(np.abs(self.lower))),
            float(np.max(np.abs(self.upper))),
        )
        bound_tolerance = 16.0 * float(np.finfo(float).eps) * magnitude
        if np.any(position < self.lower - bound_tolerance) or np.any(
            position > self.upper + bound_tolerance
        ):
            return self._invalid(QueryInvalidReason.OUTSIDE_GRID)
        position = np.minimum(np.maximum(position, self.lower), self.upper)
        grid_coordinate = (position - self.lower) / self.spacing
        maximum_index = np.asarray(self.shape, dtype=int) - 1
        # Closed-cell occupancy means a point exactly on an internal face,
        # edge, or vertex belongs to every incident cell.  Reject the query if
        # any such cell is outside the certified free component; choosing only
        # floor(point) could otherwise certify a point touching an obstacle.
        incident_per_axis = []
        for axis in range(3):
            coordinate = float(grid_coordinate[axis])
            nearest = int(round(coordinate))
            scaled_tolerance = bound_tolerance / float(self.spacing[axis])
            if abs(coordinate - nearest) <= scaled_tolerance:
                candidates = tuple(
                    value
                    for value in (nearest - 1, nearest)
                    if 0 <= value < int(maximum_index[axis])
                )
            else:
                candidates = (int(math.floor(coordinate)),)
            incident_per_axis.append(candidates)
        for incident in itertools.product(*incident_per_axis):
            if not bool(self.valid_cell_mask[incident]):
                return self._invalid(QueryInvalidReason.INVALID_CELL)
        # Trilinear interpolation is C0, not C1. At an internal cell face its
        # one-sided gradients can disagree, so selecting floor/high-side would
        # silently certify the wrong directional derivative. The initial
        # feasibility runtime fails closed at exact faces; it remains an
        # empirical sampled-data approximation rather than the paper's smooth
        # continuous PSF theorem.
        if any(len(candidates) > 1 for candidates in incident_per_axis):
            return self._invalid(
                QueryInvalidReason.NONDIFFERENTIABLE_INTERNAL_FACE
            )
        cell = np.floor(grid_coordinate).astype(int)
        cell = np.minimum(cell, maximum_index - 1)
        cell = np.maximum(cell, 0)
        local = grid_coordinate - cell
        local = np.minimum(np.maximum(local, 0.0), 1.0)
        i, j, k = (int(cell[0]), int(cell[1]), int(cell[2]))
        # Vertex masks cannot identify the interior of an occupied cell: an
        # obstacle cell may share all eight of its vertices with neighboring
        # free cells.  The containing cell therefore has an independent,
        # mandatory validity check whenever the domain provides cell labels.
        if not bool(self.valid_cell_mask[i, j, k]):
            return self._invalid(QueryInvalidReason.INVALID_CELL)
        mask_cell = self.valid_mask[i : i + 2, j : j + 2, k : k + 2]
        if mask_cell.shape != (2, 2, 2) or not np.all(mask_cell):
            return self._invalid(QueryInvalidReason.INVALID_CELL)
        cell_values = self.values[i : i + 2, j : j + 2, k : k + 2]
        if not np.all(np.isfinite(cell_values)):
            return self._invalid(QueryInvalidReason.NONFINITE_CELL_VALUES)
        if np.all(cell_values == 0.0):
            return self._invalid(QueryInvalidReason.NONREGULAR_ZERO_CELL)

        tx, ty, tz = (float(local[0]), float(local[1]), float(local[2]))
        wx = (1.0 - tx, tx)
        wy = (1.0 - ty, ty)
        wz = (1.0 - tz, tz)
        value = 0.0
        derivative_x = 0.0
        derivative_y = 0.0
        derivative_z = 0.0
        for ix in (0, 1):
            for iy in (0, 1):
                for iz in (0, 1):
                    vertex_value = float(cell_values[ix, iy, iz])
                    value += wx[ix] * wy[iy] * wz[iz] * vertex_value
                    derivative_x += (
                        (-1.0 if ix == 0 else 1.0)
                        * wy[iy]
                        * wz[iz]
                        * vertex_value
                        / float(self.spacing[0])
                    )
                    derivative_y += (
                        wx[ix]
                        * (-1.0 if iy == 0 else 1.0)
                        * wz[iz]
                        * vertex_value
                        / float(self.spacing[1])
                    )
                    derivative_z += (
                        wx[ix]
                        * wy[iy]
                        * (-1.0 if iz == 0 else 1.0)
                        * vertex_value
                        / float(self.spacing[2])
                    )
        return FieldQuery(
            valid=True,
            value=float(value),
            gradient=(
                float(derivative_x),
                float(derivative_y),
                float(derivative_z),
            ),
            reason=None,
            cell_index=(i, j, k),
            local_coordinates=(tx, ty, tz),
            outer_boundary_clearance_m=float(
                np.min(
                    np.concatenate(
                        (position - self.lower, self.upper - position)
                    )
                )
            ),
        )


# A concise alias for callers that do not need to name the interpolation rule.
PoissonField = TrilinearPoissonField
