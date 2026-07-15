"""Same-budget linearized inverse control for the CFS-00A diagnostic.

This module is deliberately opt-in.  It does not alter the historical R05A
direct-shooting solver or the default pi0.5 sampler.  The registered problem is
small: 75 active integrated increments (five flow steps by first-five XYZ) and
35 physically scaled terminal outputs (first-five seven-channel actions).

The linear solve produces a deterministic candidate and diagnostics.  Its
status is never an infeasibility certificate.  Only replay through the full
nonlinear frozen recurrence can satisfy the transport fidelity gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

from . import crfs_inverse_control as _legacy


class LinearizedControlStatus(IntEnum):
    """Stable statuses for the registered finite linearized computation."""

    SOLVED = 0
    ZERO_JACOBIAN = 1


@dataclass(frozen=True)
class LinearizedControlConfig:
    """Frozen CFS-00A projected-FISTA settings.

    The 35-by-75 convex algebra runs on CPU in float64.  This is both cheap and
    deterministic; the Jacobian and nonlinear replays remain in the model's
    native floating dtype and device.
    """

    fista_iterations: int = 4096
    finite_difference_epsilon_fractions: tuple[float, float] = (1.0 / 256.0, 1.0 / 512.0)
    finite_difference_relative_l2_tolerance: float = 0.10
    finite_difference_absolute_l2_tolerance: float = 1.0e-3

    def validate(self) -> None:
        if self.fista_iterations != 4096:
            raise ValueError("CFS-00A requires exactly 4096 projected-FISTA updates")
        if self.finite_difference_epsilon_fractions != (1.0 / 256.0, 1.0 / 512.0):
            raise ValueError("CFS-00A finite-difference epsilon fractions changed")
        if self.finite_difference_relative_l2_tolerance != 0.10:
            raise ValueError("CFS-00A finite-difference relative tolerance changed")
        if self.finite_difference_absolute_l2_tolerance != 1.0e-3:
            raise ValueError("CFS-00A finite-difference absolute tolerance changed")


@dataclass(frozen=True)
class FiniteDifferenceDiagnostics:
    directions: Tensor
    epsilon_values: Tensor
    plus_target_physical: Tensor
    minus_target_physical: Tensor
    autograd_directional_derivatives: Tensor
    central_directional_derivatives: Tensor
    absolute_l2_errors: Tensor
    relative_l2_errors: Tensor
    checks_passed: Tensor
    directions_passed: Tensor
    passed: bool
    relative_l2_tolerance: float
    absolute_l2_tolerance: float


@dataclass(frozen=True)
class FistaDiagnostics:
    updates: int
    selected_iteration: int
    spectral_norm: Tensor
    step_size: Tensor
    objective_half_squared_l2: Tensor
    historical_weighted_mse: Tensor
    executed_historical_weighted_mse: Tensor
    projected_gradient_mapping_norm: Tensor
    effective_rank: int
    convex_dtype: str = "torch.float64"
    convex_device: str = "cpu"
    early_stopping_used: bool = False
    adaptive_restart_used: bool = False
    optimality_certificate: bool = False
    infeasibility_certificate: bool = False


@dataclass(frozen=True)
class LinearizedControlResult:
    """Detached audit record for the linear candidate and nonlinear replay."""

    status: LinearizedControlStatus
    selected_candidate_float64: Tensor
    model_candidate_pre_projection: Tensor
    model_candidate_post_projection: Tensor
    active_increments: Tensor
    schedule: Tensor
    increments: Tensor
    rollout: _legacy.FlowRollout
    baseline_rollout: _legacy.FlowRollout
    baseline_final: Tensor
    target: Tensor
    budget: Tensor
    realized_target_delta_norm: Tensor
    jacobian: Tensor
    jacobian_singular_values: Tensor
    weighted_matrix_singular_values: Tensor
    jacobian_effective_rank: int
    weights: Tensor
    baseline_target_error: Tensor
    linear_predicted_target_error: Tensor
    nonlinear_target_error: Tensor
    linearization_error: Tensor
    linear_fidelity_error: Tensor
    nonlinear_fidelity_error: Tensor
    linear_metrics: _legacy.FidelityMetrics
    nonlinear_metrics: _legacy.FidelityMetrics
    linear_objective: Tensor
    nonlinear_objective: Tensor
    per_step_norms: Tensor
    path_length: Tensor
    energy: Tensor
    fista: FistaDiagnostics
    finite_difference: FiniteDifferenceDiagnostics
    target_pairing_exact: bool
    schedule_valid: bool
    jacobian_valid: bool
    terminal_overwrite_used: bool = False
    optimality_certificate: bool = False
    infeasibility_certificate: bool = False


def _detach_rollout(value: _legacy.FlowRollout) -> _legacy.FlowRollout:
    return _legacy.FlowRollout(
        states=value.states.detach().clone(),
        base_velocities=value.base_velocities.detach().clone(),
        control_velocities=value.control_velocities.detach().clone(),
        total_velocities=value.total_velocities.detach().clone(),
        increments=value.increments.detach().clone(),
        times=value.times.detach().clone(),
        final=value.final.detach().clone(),
    )


def _detach_metrics(value: _legacy.FidelityMetrics) -> _legacy.FidelityMetrics:
    return _legacy.FidelityMetrics(
        xyz_max_abs=value.xyz_max_abs.detach().clone(),
        xyz_rms=value.xyz_rms.detach().clone(),
        full_max_abs=value.full_max_abs.detach().clone(),
        full_rms=value.full_rms.detach().clone(),
        feasible=value.feasible,
    )


def _validate_inputs(
    initial_state: Tensor,
    target: Tensor,
    control_mask: Tensor,
    target_mask: Tensor,
    model_to_physical_scale: Tensor,
    control_budget: Tensor | float,
    legacy_config: _legacy.InverseControlConfig,
) -> tuple[Tensor, Tensor]:
    legacy_config.validate()
    if not initial_state.is_floating_point() or target.dtype != initial_state.dtype:
        raise ValueError("initial_state and target must share a floating dtype")
    if target.shape != initial_state.shape or target.device != initial_state.device:
        raise ValueError("target shape and device must match initial_state")
    if not torch.equal(control_mask, _legacy.first_five_xyz_mask_like(initial_state)):
        raise ValueError("control_mask must be exactly first-five XYZ")
    if not torch.equal(
        target_mask,
        _legacy.first_five_channels_mask_like(initial_state, channels=7),
    ):
        raise ValueError("target_mask must be exactly first-five seven-channel")
    for name, value in (("initial_state", initial_state), ("target", target)):
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} contains a nonfinite value")
    try:
        scale = torch.broadcast_to(model_to_physical_scale, initial_state.shape).detach()
    except RuntimeError as error:
        raise ValueError("model_to_physical_scale is not broadcastable") from error
    if scale.dtype != initial_state.dtype or scale.device != initial_state.device:
        raise ValueError("model_to_physical_scale dtype/device must match initial_state")
    if not bool(torch.isfinite(scale).all().item()) or not bool((scale > 0).all().item()):
        raise ValueError("model_to_physical_scale must be finite and strictly positive")
    budget = torch.as_tensor(
        control_budget,
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    if budget.ndim != 0 or not bool(torch.isfinite(budget).item()):
        raise ValueError("control_budget must be a finite scalar")
    if float(budget.item()) < 0.0:
        raise ValueError("control_budget must be nonnegative")
    return scale, budget


def _project_product_balls(value: Tensor, radius: Tensor) -> Tensor:
    """Project five compact 15-vectors onto independent L2 balls."""

    if value.shape != (5, 15):
        raise ValueError("compact controls must have shape (5, 15)")
    if radius.ndim != 0 or float(radius.item()) < 0.0:
        raise ValueError("product-ball radius must be a nonnegative scalar")
    norms = torch.linalg.vector_norm(value, dim=1)
    tiny = torch.finfo(value.dtype).tiny
    scale = torch.clamp(radius / torch.clamp(norms, min=tiny), max=1.0)
    return value * scale[:, None]


def _weighted_rows(
    control_mask: Tensor,
    target_mask: Tensor,
    legacy_config: _legacy.InverseControlConfig,
) -> Tensor:
    xyz_rows = control_mask[target_mask]
    if xyz_rows.shape != (35,) or int(torch.count_nonzero(xyz_rows).item()) != 15:
        raise ValueError("registered target/control masks do not define 15 of 35 XYZ rows")
    xyz_weight = 1.0 / legacy_config.xyz_rms_tolerance
    remaining_weight = 1.0 / legacy_config.full_rms_tolerance
    xyz_rows_cpu = xyz_rows.detach().to(device="cpu")
    return torch.where(
        xyz_rows_cpu,
        torch.full_like(xyz_rows_cpu, xyz_weight, dtype=torch.float64),
        torch.full_like(xyz_rows_cpu, remaining_weight, dtype=torch.float64),
    )


def _registered_directions() -> Tensor:
    """Return three fixed unit directions without using a random generator."""

    indices = torch.arange(75, dtype=torch.int64, device="cpu")
    raw = torch.stack(
        [
            torch.ones(75, dtype=torch.float64),
            torch.where(indices.remainder(2) == 0, 1.0, -1.0).to(torch.float64),
            torch.where(((indices * 17 + 3).remainder(31)) < 15, 1.0, -1.0).to(
                torch.float64
            ),
        ],
        dim=0,
    )
    return raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)


def _finite_difference_diagnostics(
    jacobian: Tensor,
    active_terminal,
    zero_active: Tensor,
    control_mask: Tensor,
    target_mask: Tensor,
    scale: Tensor,
    budget: Tensor,
    config: LinearizedControlConfig,
) -> FiniteDifferenceDiagnostics:
    """Check registered Jacobian-vector products against central differences."""

    directions64 = _registered_directions()
    directions = directions64.to(dtype=zero_active.dtype, device=zero_active.device)
    radius = budget / 5.0
    epsilon_values = torch.stack(
        [
            radius * fraction
            for fraction in config.finite_difference_epsilon_fractions
        ]
    )
    if not bool((epsilon_values > 0).all().item()):
        raise ValueError("CFS-00A finite-difference checks require a positive budget")

    autograd_products: list[Tensor] = []
    central_by_direction: list[Tensor] = []
    plus_by_direction: list[Tensor] = []
    minus_by_direction: list[Tensor] = []
    absolute_by_direction: list[Tensor] = []
    relative_by_direction: list[Tensor] = []
    passed_by_direction: list[Tensor] = []
    for direction in directions:
        autograd_product = jacobian @ direction
        autograd_products.append(autograd_product.detach())
        central_by_epsilon: list[Tensor] = []
        plus_by_epsilon: list[Tensor] = []
        minus_by_epsilon: list[Tensor] = []
        absolute_by_epsilon: list[Tensor] = []
        relative_by_epsilon: list[Tensor] = []
        passed_by_epsilon: list[Tensor] = []
        for epsilon in epsilon_values:
            compact = epsilon * direction.reshape(5, 15)
            plus = torch.zeros_like(zero_active)
            minus = torch.zeros_like(zero_active)
            plus[:, control_mask] = compact
            minus[:, control_mask] = -compact
            with torch.no_grad():
                plus_terminal = active_terminal(plus)
                minus_terminal = active_terminal(minus)
            plus_target = (plus_terminal * scale)[target_mask]
            minus_target = (minus_terminal * scale)[target_mask]
            central = (
                (plus_target - minus_target)
                / (2.0 * epsilon)
            )
            difference = central - autograd_product
            absolute = torch.linalg.vector_norm(difference)
            denominator = torch.maximum(
                torch.maximum(
                    torch.linalg.vector_norm(central),
                    torch.linalg.vector_norm(autograd_product),
                ),
                torch.tensor(1.0e-6, dtype=absolute.dtype, device=absolute.device),
            )
            relative = absolute / denominator
            passed = (relative <= config.finite_difference_relative_l2_tolerance) | (
                absolute <= config.finite_difference_absolute_l2_tolerance
            )
            central_by_epsilon.append(central.detach())
            plus_by_epsilon.append(plus_target.detach())
            minus_by_epsilon.append(minus_target.detach())
            absolute_by_epsilon.append(absolute.detach())
            relative_by_epsilon.append(relative.detach())
            passed_by_epsilon.append(passed.detach())
        central_by_direction.append(torch.stack(central_by_epsilon, dim=0))
        plus_by_direction.append(torch.stack(plus_by_epsilon, dim=0))
        minus_by_direction.append(torch.stack(minus_by_epsilon, dim=0))
        absolute_by_direction.append(torch.stack(absolute_by_epsilon, dim=0))
        relative_by_direction.append(torch.stack(relative_by_epsilon, dim=0))
        passed_by_direction.append(torch.stack(passed_by_epsilon, dim=0))

    checks_passed = torch.stack(passed_by_direction, dim=0)
    directions_passed = torch.any(checks_passed, dim=1)
    return FiniteDifferenceDiagnostics(
        directions=directions.detach().clone(),
        epsilon_values=epsilon_values.detach().clone(),
        plus_target_physical=torch.stack(plus_by_direction, dim=0),
        minus_target_physical=torch.stack(minus_by_direction, dim=0),
        autograd_directional_derivatives=torch.stack(autograd_products, dim=0),
        central_directional_derivatives=torch.stack(central_by_direction, dim=0),
        absolute_l2_errors=torch.stack(absolute_by_direction, dim=0),
        relative_l2_errors=torch.stack(relative_by_direction, dim=0),
        checks_passed=checks_passed.detach().clone(),
        directions_passed=directions_passed.detach().clone(),
        passed=bool(torch.all(directions_passed).item()),
        relative_l2_tolerance=config.finite_difference_relative_l2_tolerance,
        absolute_l2_tolerance=config.finite_difference_absolute_l2_tolerance,
    )


def _fista_product_balls(
    jacobian: Tensor,
    baseline_error: Tensor,
    weights: Tensor,
    budget: Tensor,
    config: LinearizedControlConfig,
) -> tuple[Tensor, Tensor, FistaDiagnostics]:
    """Solve the fixed weighted linear least-squares diagnostic."""

    def require_finite(name: str, value: Tensor) -> None:
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"projected FISTA produced nonfinite {name}")

    # Convex algebra is intentionally small, device-fixed, and performed after
    # a single audit-visible cast from the native Jacobian.
    matrix = (jacobian.detach().to(device="cpu", dtype=torch.float64) * weights[:, None]).contiguous()
    offset = (baseline_error.detach().to(device="cpu", dtype=torch.float64) * weights).contiguous()
    radius = budget.detach().to(device="cpu", dtype=torch.float64) / 5.0
    require_finite("weighted matrix", matrix)
    require_finite("weighted offset", offset)
    require_finite("constraint radius", radius)
    singular_values = torch.linalg.svdvals(matrix)
    require_finite("weighted singular values", singular_values)
    exact_zero = bool(torch.count_nonzero(matrix).item() == 0)

    zero = torch.zeros((5, 15), dtype=torch.float64, device="cpu")

    def half_squared_l2(compact: Tensor) -> Tensor:
        residual = matrix @ compact.reshape(-1) + offset
        return 0.5 * torch.sum(torch.square(residual))

    def historical_mse(compact: Tensor) -> Tensor:
        residual = matrix @ compact.reshape(-1) + offset
        return torch.mean(torch.square(residual))

    if exact_zero:
        objective = half_squared_l2(zero)
        mse = historical_mse(zero)
        require_finite("zero-Jacobian objective", objective)
        require_finite("zero-Jacobian historical MSE", mse)
        diagnostic = FistaDiagnostics(
            updates=0,
            selected_iteration=0,
            spectral_norm=torch.zeros((), dtype=torch.float64),
            step_size=torch.zeros((), dtype=torch.float64),
            objective_half_squared_l2=objective.detach().clone(),
            historical_weighted_mse=mse.detach().clone(),
            executed_historical_weighted_mse=mse.detach().clone(),
            projected_gradient_mapping_norm=torch.zeros((), dtype=torch.float64),
            effective_rank=0,
        )
        return zero, singular_values, diagnostic

    sigma_max = singular_values[0]
    if not bool(torch.isfinite(sigma_max).item()) or not float(sigma_max.item()) > 0.0:
        raise ValueError("weighted Jacobian has an invalid spectral norm")
    step_size = 1.0 / torch.square(sigma_max)
    require_finite("step size", step_size)
    x_value = zero.clone()
    y_value = zero.clone()
    momentum = torch.ones((), dtype=torch.float64)
    best = x_value.clone()
    best_iteration = 0
    best_mse = historical_mse(best)
    require_finite("initial historical MSE", best_mse)

    for update in range(1, config.fista_iterations + 1):
        residual = matrix @ y_value.reshape(-1) + offset
        require_finite(f"residual at update {update}", residual)
        gradient = (matrix.T @ residual).reshape(5, 15)
        require_finite(f"gradient at update {update}", gradient)
        next_x = _project_product_balls(y_value - step_size * gradient, radius)
        require_finite(f"projected iterate at update {update}", next_x)
        next_mse = historical_mse(next_x)
        require_finite(f"historical MSE at update {update}", next_mse)
        # Strict comparison preserves the earliest iterate on an exact tie.
        if bool((next_mse < best_mse).item()):
            best = next_x.detach().clone()
            best_mse = next_mse.detach().clone()
            best_iteration = update
        next_momentum = (1.0 + torch.sqrt(1.0 + 4.0 * torch.square(momentum))) / 2.0
        require_finite(f"momentum at update {update}", next_momentum)
        y_value = next_x + ((momentum - 1.0) / next_momentum) * (next_x - x_value)
        require_finite(f"extrapolated iterate at update {update}", y_value)
        x_value = next_x
        momentum = next_momentum

    best_residual = matrix @ best.reshape(-1) + offset
    require_finite("selected residual", best_residual)
    best_gradient = (matrix.T @ best_residual).reshape(5, 15)
    require_finite("selected gradient", best_gradient)
    mapped = _project_product_balls(best - step_size * best_gradient, radius)
    require_finite("selected projected-gradient point", mapped)
    gradient_mapping_norm = torch.linalg.vector_norm(
        ((best - mapped) / step_size).reshape(-1)
    )
    require_finite("projected-gradient mapping norm", gradient_mapping_norm)
    threshold = (
        torch.finfo(torch.float64).eps
        * max(matrix.shape)
        * sigma_max
    )
    effective_rank = int(torch.count_nonzero(singular_values > threshold).item())
    diagnostic = FistaDiagnostics(
        updates=config.fista_iterations,
        selected_iteration=best_iteration,
        spectral_norm=sigma_max.detach().clone(),
        step_size=step_size.detach().clone(),
        objective_half_squared_l2=(0.5 * torch.sum(torch.square(best_residual))).detach().clone(),
        historical_weighted_mse=best_mse.detach().clone(),
        executed_historical_weighted_mse=torch.full((), float("nan"), dtype=torch.float64),
        projected_gradient_mapping_norm=gradient_mapping_norm.detach().clone(),
        effective_rank=effective_rank,
    )
    return best, singular_values, diagnostic


def solve_linearized_control(
    initial_state: Tensor,
    target: Tensor,
    velocity_fn: _legacy.VelocityFn,
    *,
    control_mask: Tensor,
    target_mask: Tensor,
    model_to_physical_scale: Tensor,
    control_budget: Tensor | float,
    legacy_config: _legacy.InverseControlConfig,
    config: LinearizedControlConfig | None = None,
) -> LinearizedControlResult:
    """Compute and nonlinearly replay the registered same-budget candidate."""

    config = config or LinearizedControlConfig()
    config.validate()
    scale, budget = _validate_inputs(
        initial_state,
        target,
        control_mask,
        target_mask,
        model_to_physical_scale,
        control_budget,
        legacy_config,
    )
    initial_state = initial_state.detach()
    target = target.detach()
    dt = torch.tensor(
        legacy_config.dt,
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    active_count = legacy_config.num_steps - legacy_config.intervention_step
    if active_count != 5:
        raise ValueError("CFS-00A requires exactly five active flow steps")
    times = _legacy._sampler_times(initial_state, legacy_config)

    # The inactive prefix is common to every row-wise VJP and is detached once.
    x_prefix = initial_state
    with torch.no_grad():
        for step in range(legacy_config.intervention_step):
            velocity = _legacy._call_velocity(velocity_fn, x_prefix, times[step], step)
            x_prefix = x_prefix + dt * velocity
            if not bool(torch.isfinite(x_prefix).all().item()):
                raise ValueError("nonfinite state in the inactive prefix")
    x_prefix = x_prefix.detach()

    def active_terminal(candidate: Tensor) -> Tensor:
        if candidate.shape != (active_count, *initial_state.shape):
            raise ValueError("active increment candidate shape changed")
        x_value = x_prefix
        for local_index, step in enumerate(
            range(legacy_config.intervention_step, legacy_config.num_steps)
        ):
            velocity = _legacy._call_velocity(velocity_fn, x_value, times[step], step)
            control = candidate[local_index] / dt
            total = _legacy._controlled_velocity_for_solve(velocity, control)
            x_value = x_value + dt * total
            if not bool(torch.isfinite(x_value).all().item()):
                raise ValueError("nonfinite state in an active rollout")
        return x_value

    zero_active = torch.zeros(
        (active_count, *initial_state.shape),
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    zero_schedule = torch.zeros(
        (legacy_config.num_steps, *initial_state.shape),
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    with torch.no_grad():
        baseline_rollout = _legacy.replay_velocity_schedule(
            initial_state,
            velocity_fn,
            zero_schedule,
            config=legacy_config,
        )
        baseline_terminal = active_terminal(zero_active)
    if not (
        torch.equal(baseline_terminal, baseline_rollout.final)
        and torch.equal(torch.signbit(baseline_terminal), torch.signbit(baseline_rollout.final))
    ):
        raise ValueError("linearization zero-control baseline differs from canonical replay")
    baseline_final = baseline_rollout.final.detach()
    if not _legacy._finite_values_bitwise_equal(
        target,
        baseline_final,
        ~control_mask,
    ):
        raise ValueError("target pairing outside first-five XYZ is not bitwise exact")
    baseline_fidelity_error = (baseline_final - target) * scale
    baseline_target_error = baseline_fidelity_error[target_mask].detach().clone()
    delta = torch.where(
        control_mask,
        target - baseline_final,
        torch.zeros_like(initial_state),
    )
    realized_target_delta_norm = torch.linalg.vector_norm(delta.reshape(-1))

    # Row-wise reverse mode bounds memory for the checkpointed frozen model.
    jacobian_rows: list[Tensor] = []
    for row_index in range(35):
        with torch.enable_grad():
            candidate = zero_active.detach().requires_grad_(requires_grad=True)
            terminal = active_terminal(candidate)
            output = ((terminal - target) * scale)[target_mask]
            scalar = output[row_index]
            gradient = torch.autograd.grad(scalar, candidate, retain_graph=False)[0]
        compact_gradient = gradient[:, control_mask].reshape(-1)
        if compact_gradient.shape != (75,):
            raise RuntimeError("linearized control Jacobian row is not length 75")
        jacobian_rows.append(compact_gradient.detach())
    jacobian = torch.stack(jacobian_rows, dim=0)
    if jacobian.shape != (35, 75) or not bool(torch.isfinite(jacobian).all().item()):
        raise ValueError("linearized control Jacobian is invalid")
    finite_difference = _finite_difference_diagnostics(
        jacobian,
        active_terminal,
        zero_active,
        control_mask,
        target_mask,
        scale,
        budget,
        config,
    )
    if not finite_difference.passed:
        raise ValueError(
            "registered finite-difference validation rejected the autograd Jacobian"
        )
    weights = _weighted_rows(control_mask, target_mask, legacy_config)
    selected64, weighted_singular_values64, fista = _fista_product_balls(
        jacobian,
        baseline_target_error,
        weights,
        budget,
        config,
    )

    jacobian64 = jacobian.detach().to(device="cpu", dtype=torch.float64)
    jacobian_singular_values64 = torch.linalg.svdvals(jacobian64)
    if not bool(torch.isfinite(jacobian_singular_values64).all().item()):
        raise ValueError("raw Jacobian singular values are nonfinite")
    jacobian_sigma_max = jacobian_singular_values64[0]
    jacobian_rank_threshold = (
        torch.finfo(torch.float64).eps
        * max(jacobian64.shape)
        * jacobian_sigma_max
    )
    jacobian_effective_rank = int(
        torch.count_nonzero(
            jacobian_singular_values64 > jacobian_rank_threshold
        ).item()
    )

    model_candidate_pre_projection = torch.zeros_like(zero_active)
    model_candidate_pre_projection[:, control_mask] = selected64.reshape(5, 15).to(
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    model_candidate_post_projection = _legacy._project_increments(
        model_candidate_pre_projection,
        control_mask,
        budget,
        legacy_config,
    ).detach()
    active_increments = model_candidate_post_projection
    schedule = zero_schedule.clone()
    schedule[legacy_config.intervention_step :] = active_increments / dt
    increments, per_step_norms, path_length = _legacy.validate_schedule_constraints(
        schedule,
        control_mask,
        budget,
        config=legacy_config,
    )
    with torch.no_grad():
        rollout = _legacy.replay_velocity_schedule(
            initial_state,
            velocity_fn,
            schedule,
            config=legacy_config,
        )
        nonlinear_fidelity_error = (rollout.final - target) * scale
        nonlinear_metrics = _legacy._fidelity_metrics(
            nonlinear_fidelity_error,
            control_mask,
            target_mask,
            legacy_config,
        )
        nonlinear_objective = _legacy._objective(
            nonlinear_fidelity_error,
            control_mask,
            target_mask,
            legacy_config,
        )

    # The residual-schedule transport executes ``dt * (c / dt)`` in float32.
    # That round trip is not byte-identical to ``c`` for every float32 value,
    # so the scientific linear prediction must use the exact increments owned
    # by the canonical replay rather than the pre-transport candidate.
    executed_compact64 = increments[
        legacy_config.intervention_step :
    ][:, control_mask].reshape(-1).to(
        device="cpu", dtype=torch.float64
    )
    linear_target_error64 = (
        baseline_target_error.to(device="cpu", dtype=torch.float64)
        + jacobian.to(device="cpu", dtype=torch.float64) @ executed_compact64
    )
    linear_target_error = linear_target_error64.to(
        dtype=initial_state.dtype,
        device=initial_state.device,
    )
    linear_fidelity_error = baseline_fidelity_error.detach().clone()
    linear_fidelity_error[target_mask] = linear_target_error
    linear_metrics = _legacy._fidelity_metrics(
        linear_fidelity_error,
        control_mask,
        target_mask,
        legacy_config,
    )
    linear_objective = _legacy._objective(
        linear_fidelity_error,
        control_mask,
        target_mask,
        legacy_config,
    )
    nonlinear_target_error = nonlinear_fidelity_error[target_mask]
    executed_weighted = linear_target_error64 * weights
    executed_mse = torch.mean(torch.square(executed_weighted))
    fista = FistaDiagnostics(
        updates=fista.updates,
        selected_iteration=fista.selected_iteration,
        spectral_norm=fista.spectral_norm,
        step_size=fista.step_size,
        objective_half_squared_l2=fista.objective_half_squared_l2,
        historical_weighted_mse=fista.historical_weighted_mse,
        executed_historical_weighted_mse=executed_mse.detach().clone(),
        projected_gradient_mapping_norm=fista.projected_gradient_mapping_norm,
        effective_rank=fista.effective_rank,
    )
    status = (
        LinearizedControlStatus.ZERO_JACOBIAN
        if bool(torch.count_nonzero(jacobian).item() == 0)
        else LinearizedControlStatus.SOLVED
    )
    return LinearizedControlResult(
        status=status,
        selected_candidate_float64=selected64.detach().clone(),
        model_candidate_pre_projection=model_candidate_pre_projection.detach().clone(),
        model_candidate_post_projection=model_candidate_post_projection.detach().clone(),
        active_increments=active_increments.detach().clone(),
        schedule=schedule.detach().clone(),
        increments=increments.detach().clone(),
        rollout=_detach_rollout(rollout),
        baseline_rollout=_detach_rollout(baseline_rollout),
        baseline_final=baseline_final.detach().clone(),
        target=target.detach().clone(),
        budget=budget.detach().clone(),
        realized_target_delta_norm=realized_target_delta_norm.detach().clone(),
        jacobian=jacobian.detach().clone(),
        jacobian_singular_values=jacobian_singular_values64.detach().clone(),
        weighted_matrix_singular_values=weighted_singular_values64.detach().clone(),
        jacobian_effective_rank=jacobian_effective_rank,
        weights=weights.detach().clone(),
        baseline_target_error=baseline_target_error.detach().clone(),
        linear_predicted_target_error=linear_target_error.detach().clone(),
        nonlinear_target_error=nonlinear_target_error.detach().clone(),
        linearization_error=(nonlinear_target_error - linear_target_error).detach().clone(),
        linear_fidelity_error=linear_fidelity_error.detach().clone(),
        nonlinear_fidelity_error=nonlinear_fidelity_error.detach().clone(),
        linear_metrics=_detach_metrics(linear_metrics),
        nonlinear_metrics=_detach_metrics(nonlinear_metrics),
        linear_objective=linear_objective.detach().clone(),
        nonlinear_objective=nonlinear_objective.detach().clone(),
        per_step_norms=per_step_norms.detach().clone(),
        path_length=path_length.detach().clone(),
        energy=torch.sum(torch.square(increments)).detach().clone(),
        fista=fista,
        finite_difference=finite_difference,
        target_pairing_exact=True,
        schedule_valid=True,
        jacobian_valid=finite_difference.passed,
    )


def linearize_active_increments(*args, **kwargs) -> LinearizedControlResult:
    """Public descriptive alias used by diagnostic tooling."""

    return solve_linearized_control(*args, **kwargs)


def replay_linearized_candidate(
    initial_state: Tensor,
    velocity_fn: _legacy.VelocityFn,
    result: LinearizedControlResult,
    *,
    legacy_config: _legacy.InverseControlConfig,
) -> _legacy.FlowRollout:
    """Independently replay a recorded candidate through the canonical recurrence."""

    return _legacy.replay_velocity_schedule(
        initial_state,
        velocity_fn,
        result.schedule,
        config=legacy_config,
    )
