"""Budgeted inverse control for the R05A frozen-flow teacher.

This module contains no robot geometry and no simulator feedback.  It solves a
small deterministic direct-shooting problem around a caller-supplied frozen
velocity field.  The controlled recurrence is always

    x_(k+1) = x_k + dt * (v(x_k, t_k) + u_k)

with ten reverse-time Euler steps, ``dt=-0.1``, and controls restricted to
steps 5--9.  The optimizer parameter is the integrated increment
``c_k = dt * u_k`` so that the registered path budget is explicit.

The finite search is a feasibility procedure, not an optimality or
infeasibility certificate.  It reports nonconvergence when no admissible
candidate reaches the fixed fidelity gates.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
import math

import torch
from torch import Tensor

VelocityFn = Callable[[Tensor, Tensor, int], Tensor]


class InverseControlStatus(IntEnum):
    """Stable numeric statuses suitable for tensor-only sampler traces."""

    CONVERGED = 0
    ZERO_BUDGET_CONVERGED = 1
    TARGET_PAIRING_MISMATCH = 2
    MAX_ITERATIONS = 3
    NONFINITE = 4


@dataclass(frozen=True)
class InverseControlConfig:
    """Fixed direct-shooting settings.

    Structural sampler settings are validated against the registered R05A
    protocol.  Optimization settings remain explicit so the real experiment
    can content-bind them after IFT-00.
    """

    num_steps: int = 10
    intervention_step: int = 5
    dt: float = -0.1
    max_iterations: int = 128
    learning_rate: float = 0.02
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    xyz_max_abs_tolerance: float = 0.010
    xyz_rms_tolerance: float = 0.005
    full_max_abs_tolerance: float = 0.050
    full_rms_tolerance: float = 0.015
    constraint_slack_ulps: int = 8
    stop_on_first_feasible: bool = False

    def validate(self) -> None:
        if self.num_steps != 10:
            raise ValueError("R05A inverse control requires exactly ten Euler steps")
        if self.intervention_step != 5:
            raise ValueError("R05A inverse control requires intervention_step=5")
        if self.dt != -0.1:
            raise ValueError("R05A inverse control requires dt=-0.1")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not math.isfinite(self.learning_rate) or not self.learning_rate > 0.0:
            raise ValueError("learning_rate must be positive")
        if (
            not math.isfinite(self.adam_beta1)
            or not math.isfinite(self.adam_beta2)
            or not 0.0 <= self.adam_beta1 < 1.0
            or not 0.0 <= self.adam_beta2 < 1.0
        ):
            raise ValueError("Adam beta values must lie in [0, 1)")
        if not math.isfinite(self.adam_epsilon) or not self.adam_epsilon > 0.0:
            raise ValueError("adam_epsilon must be positive")
        for name, value in (
            ("xyz_max_abs_tolerance", self.xyz_max_abs_tolerance),
            ("xyz_rms_tolerance", self.xyz_rms_tolerance),
            ("full_max_abs_tolerance", self.full_max_abs_tolerance),
            ("full_rms_tolerance", self.full_rms_tolerance),
        ):
            if not math.isfinite(value) or not value > 0.0:
                raise ValueError(f"{name} must be positive")
        if self.constraint_slack_ulps < 0:
            raise ValueError("constraint_slack_ulps must be nonnegative")
        if self.stop_on_first_feasible:
            raise ValueError("R05A requires the complete fixed iteration count")


@dataclass(frozen=True)
class FlowRollout:
    """Complete audit trace for one canonical ten-step schedule replay."""

    states: Tensor
    base_velocities: Tensor
    control_velocities: Tensor
    total_velocities: Tensor
    increments: Tensor
    times: Tensor
    final: Tensor


@dataclass(frozen=True)
class FidelityMetrics:
    xyz_max_abs: Tensor
    xyz_rms: Tensor
    full_max_abs: Tensor
    full_rms: Tensor
    feasible: bool


@dataclass(frozen=True)
class InverseControlResult:
    """Detached solver result; no autograd graph is retained."""

    status: InverseControlStatus
    converged: bool
    iterations: int
    schedule: Tensor | None
    increments: Tensor | None
    rollout: FlowRollout | None
    baseline_final: Tensor | None
    target: Tensor
    budget: Tensor | None
    realized_target_delta_norm: Tensor | None
    path_length: Tensor | None
    per_step_norms: Tensor | None
    energy: Tensor | None
    model_error: Tensor | None
    fidelity_error: Tensor | None
    metrics: FidelityMetrics | None
    objective: Tensor | None
    fidelity_scale: Tensor
    target_pairing_checked: bool
    target_pairing_exact: bool
    target_pairing_max_abs: Tensor | None
    schedule_valid: bool
    baseline_valid: bool
    nonfinite_detected: bool
    terminal_overwrite_used: bool = False
    optimality_certificate: bool = False
    infeasibility_certificate: bool = False


class _NonFiniteFlowError(RuntimeError):
    pass


class _ExactControlAdd(torch.autograd.Function):
    """Exact no-op forward with an identity control derivative at zero.

    The canonical replay must preserve baseline signed-zero bytes, so its
    forward uses ``where``.  Direct shooting must nevertheless be able to move
    a control initialized at zero; the custom backward supplies the derivative
    of ordinary addition for both arguments.
    """

    @staticmethod
    def forward(base_velocity: Tensor, control: Tensor) -> Tensor:
        return torch.where(
            control == 0,
            base_velocity,
            base_velocity + control,
        )

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        del ctx, inputs, output

    @staticmethod
    def backward(ctx, gradient: Tensor) -> tuple[Tensor, Tensor]:
        del ctx
        return gradient, gradient


def _controlled_velocity_for_solve(base_velocity: Tensor, control: Tensor) -> Tensor:
    return _ExactControlAdd.apply(base_velocity, control)


def first_five_xyz_mask_like(value: Tensor) -> Tensor:
    """Return a bool mask for the first five action rows and XYZ channels."""

    if value.ndim < 2 or value.shape[-2] < 5 or value.shape[-1] < 3:
        raise ValueError("value must expose at least five action rows and three channels")
    mask = torch.zeros_like(value, dtype=torch.bool)
    mask[..., :5, :3] = True
    return mask


def first_five_channels_mask_like(value: Tensor, *, channels: int = 7) -> Tensor:
    """Return a bool mask for the first five rows and requested channels."""

    if channels < 1:
        raise ValueError("channels must be positive")
    if value.ndim < 2 or value.shape[-2] < 5 or value.shape[-1] < channels:
        raise ValueError(f"value must expose at least five action rows and {channels} channels")
    mask = torch.zeros_like(value, dtype=torch.bool)
    mask[..., :5, :channels] = True
    return mask


def reverse_active_schedule(schedule: Tensor, *, intervention_step: int = 5) -> Tensor:
    """Reverse only active rows, preserving the exact control multiset."""

    if schedule.ndim < 1 or schedule.shape[0] != 10:
        raise ValueError("schedule must have ten flow-step rows")
    if intervention_step != 5:
        raise ValueError("R05A reverse diagnostic requires intervention_step=5")
    reversed_schedule = schedule.detach().clone()
    reversed_schedule[intervention_step:] = torch.flip(
        schedule[intervention_step:], dims=(0,)
    ).contiguous()
    return reversed_schedule


def _validate_finite(name: str, value: Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains a nonfinite value")


def _finite_values_bitwise_equal(left: Tensor, right: Tensor, mask: Tensor) -> bool:
    """Compare finite selected values exactly, including the sign of zero."""

    left_values = left[mask]
    right_values = right[mask]
    return bool(
        torch.equal(left_values, right_values)
        and torch.equal(torch.signbit(left_values), torch.signbit(right_values))
    )


def _sampler_times(initial_state: Tensor, config: InverseControlConfig) -> tuple[Tensor, ...]:
    """Reproduce the sampler's iterative float time updates exactly."""

    dt_tensor = torch.tensor(config.dt, dtype=initial_state.dtype, device=initial_state.device)
    time = torch.tensor(1.0, dtype=initial_state.dtype, device=initial_state.device)
    times = []
    for _ in range(config.num_steps):
        times.append(time.detach().clone())
        # This deliberately mirrors ``time += dt`` in PI0Pytorch.sample_actions.
        time += dt_tensor
    return tuple(times)


def _call_velocity(velocity_fn: VelocityFn, x_t: Tensor, time: Tensor, step: int) -> Tensor:
    velocity = velocity_fn(x_t, time, step)
    if not isinstance(velocity, Tensor) or velocity.shape != x_t.shape:
        shape = None if not isinstance(velocity, Tensor) else tuple(velocity.shape)
        raise ValueError(
            f"velocity_fn step {step} returned shape {shape}, expected {tuple(x_t.shape)}"
        )
    if not bool(torch.isfinite(velocity).all().item()):
        raise _NonFiniteFlowError(f"velocity_fn step {step} returned a nonfinite value")
    return velocity


def replay_velocity_schedule(
    initial_state: Tensor,
    velocity_fn: VelocityFn,
    schedule: Tensor,
    *,
    config: InverseControlConfig | None = None,
) -> FlowRollout:
    """Replay a velocity schedule through the exact registered recurrence."""

    config = config or InverseControlConfig()
    config.validate()
    if not initial_state.is_floating_point():
        raise ValueError("initial_state must be floating point")
    expected_shape = (config.num_steps, *initial_state.shape)
    if tuple(schedule.shape) != expected_shape:
        raise ValueError(f"schedule shape {tuple(schedule.shape)} != {expected_shape}")
    if schedule.dtype != initial_state.dtype or schedule.device != initial_state.device:
        raise ValueError("schedule dtype and device must match initial_state")
    _validate_finite("initial_state", initial_state)
    _validate_finite("schedule", schedule)

    dt_tensor = torch.tensor(config.dt, dtype=initial_state.dtype, device=initial_state.device)
    step_times = _sampler_times(initial_state, config)
    x_t = initial_state
    states = [x_t]
    base_velocities = []
    total_velocities = []
    times = []
    for step in range(config.num_steps):
        time = step_times[step]
        base_velocity = _call_velocity(velocity_fn, x_t, time, step)
        control = schedule[step]
        # Preserve the exact baseline value, including signed zero, wherever
        # the intervention is inactive.
        total_velocity = torch.where(control == 0, base_velocity, base_velocity + control)
        # Keep this expression identical to the sampler.  Replacing it by
        # x + dt*v + c changes floating-point rounding.
        x_next = x_t + dt_tensor * total_velocity
        if not bool(torch.isfinite(x_next).all().item()):
            raise _NonFiniteFlowError(f"Euler state after step {step} is nonfinite")
        base_velocities.append(base_velocity)
        total_velocities.append(total_velocity)
        times.append(time)
        states.append(x_next)
        x_t = x_next

    return FlowRollout(
        states=torch.stack(states, dim=0),
        base_velocities=torch.stack(base_velocities, dim=0),
        control_velocities=schedule,
        total_velocities=torch.stack(total_velocities, dim=0),
        increments=dt_tensor * schedule,
        times=torch.stack(times, dim=0),
        final=x_t,
    )


def validate_schedule_constraints(
    schedule: Tensor,
    control_mask: Tensor,
    budget: Tensor | float,
    *,
    config: InverseControlConfig | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Validate timing, coordinate mask, path budget, and per-step cap."""

    config = config or InverseControlConfig()
    config.validate()
    if schedule.shape[0] != config.num_steps or tuple(schedule.shape[1:]) != tuple(control_mask.shape):
        raise ValueError("schedule/control_mask shape mismatch")
    if control_mask.dtype != torch.bool or control_mask.device != schedule.device:
        raise ValueError("control_mask must be bool on the schedule device")
    if not torch.equal(control_mask, first_five_xyz_mask_like(schedule[0])):
        raise ValueError("R05A control_mask must be exactly first-five XYZ")
    _validate_finite("schedule", schedule)
    if bool(torch.count_nonzero(schedule[: config.intervention_step]).item()):
        raise ValueError("controls must be exactly zero at steps 0--4")
    expanded_mask = control_mask.unsqueeze(0).expand_as(schedule)
    if bool(torch.count_nonzero(torch.where(expanded_mask, 0.0, schedule)).item()):
        raise ValueError("controls must be exactly zero outside the registered mask")

    dt_tensor = torch.tensor(config.dt, dtype=schedule.dtype, device=schedule.device)
    increments = dt_tensor * schedule
    active = increments[config.intervention_step :]
    per_step = torch.linalg.vector_norm(active.reshape(active.shape[0], -1), dim=1)
    path = torch.sum(per_step)
    budget_tensor = torch.as_tensor(budget, dtype=schedule.dtype, device=schedule.device)
    if budget_tensor.ndim != 0 or not bool(torch.isfinite(budget_tensor).item()):
        raise ValueError("budget must be a finite scalar")
    if float(budget_tensor.item()) < 0.0:
        raise ValueError("budget must be nonnegative")
    if float(budget_tensor.item()) == 0.0 and bool(torch.count_nonzero(active).item()):
        raise ValueError("zero budget requires exactly zero active increments")
    active_count = config.num_steps - config.intervention_step
    cap = budget_tensor / active_count
    positive_inf = torch.full_like(budget_tensor, float("inf"))
    allowed_budget = budget_tensor
    allowed_cap = cap
    for _ in range(config.constraint_slack_ulps):
        allowed_budget = torch.nextafter(allowed_budget, positive_inf)
        allowed_cap = torch.nextafter(allowed_cap, positive_inf)
    if bool((per_step > allowed_cap).any().item()):
        raise ValueError("a control increment exceeds B/5")
    if bool((path > allowed_budget).item()):
        raise ValueError("control path exceeds B")
    return increments, per_step, path


def _project_increments(
    increments: Tensor,
    control_mask: Tensor,
    budget: Tensor,
    config: InverseControlConfig,
) -> Tensor:
    expanded_mask = control_mask.unsqueeze(0).expand_as(increments)
    masked = torch.where(expanded_mask, increments, torch.zeros_like(increments))
    active_count = config.num_steps - config.intervention_step
    flat = masked.reshape(active_count, -1)
    norms = torch.linalg.vector_norm(flat, dim=1)
    cap = budget / active_count
    safe_norms = torch.clamp(norms, min=torch.finfo(masked.dtype).tiny)
    scales = torch.clamp(cap / safe_norms, max=1.0)
    projected = masked * scales.reshape((active_count,) + (1,) * control_mask.ndim)
    return torch.where(expanded_mask, projected, torch.zeros_like(projected))


def _fidelity_metrics(
    fidelity_error: Tensor,
    control_mask: Tensor,
    target_mask: Tensor,
    config: InverseControlConfig,
) -> FidelityMetrics:
    xyz_error = fidelity_error[control_mask]
    full_error = fidelity_error[target_mask]
    xyz_max = torch.max(torch.abs(xyz_error))
    xyz_rms = torch.sqrt(torch.mean(torch.square(xyz_error)))
    full_max = torch.max(torch.abs(full_error))
    full_rms = torch.sqrt(torch.mean(torch.square(full_error)))
    feasible = bool(
        (xyz_max <= config.xyz_max_abs_tolerance).item()
        and (xyz_rms <= config.xyz_rms_tolerance).item()
        and (full_max <= config.full_max_abs_tolerance).item()
        and (full_rms <= config.full_rms_tolerance).item()
    )
    return FidelityMetrics(
        xyz_max_abs=xyz_max,
        xyz_rms=xyz_rms,
        full_max_abs=full_max,
        full_rms=full_rms,
        feasible=feasible,
    )


def _objective(
    fidelity_error: Tensor,
    control_mask: Tensor,
    target_mask: Tensor,
    config: InverseControlConfig,
) -> Tensor:
    # Tight XYZ coordinates and the remaining target coordinates are scaled by
    # their registered RMS gates so neither group silently dominates by units.
    xyz_values = fidelity_error[control_mask] / config.xyz_rms_tolerance
    remaining_mask = target_mask & ~control_mask
    terms = [torch.square(xyz_values)]
    if bool(remaining_mask.any().item()):
        full_values = fidelity_error[remaining_mask] / config.full_rms_tolerance
        terms.append(torch.square(full_values))
    return torch.mean(torch.cat([term.reshape(-1) for term in terms]))


def _detach_rollout(rollout: FlowRollout) -> FlowRollout:
    return FlowRollout(
        states=rollout.states.detach().clone(),
        base_velocities=rollout.base_velocities.detach().clone(),
        control_velocities=rollout.control_velocities.detach().clone(),
        total_velocities=rollout.total_velocities.detach().clone(),
        increments=rollout.increments.detach().clone(),
        times=rollout.times.detach().clone(),
        final=rollout.final.detach().clone(),
    )


def _detach_metrics(metrics: FidelityMetrics) -> FidelityMetrics:
    return FidelityMetrics(
        xyz_max_abs=metrics.xyz_max_abs.detach().clone(),
        xyz_rms=metrics.xyz_rms.detach().clone(),
        full_max_abs=metrics.full_max_abs.detach().clone(),
        full_rms=metrics.full_rms.detach().clone(),
        feasible=metrics.feasible,
    )


def _failure_result(
    status: InverseControlStatus,
    target: Tensor,
    fidelity_scale: Tensor,
    *,
    iterations: int,
    nonfinite: bool,
) -> InverseControlResult:
    # Do not fabricate a zero-velocity trajectory when the supplied frozen
    # field is nonfinite.  Unavailable scientific quantities remain absent and
    # the caller must branch on the explicit validity flags/status.
    return InverseControlResult(
        status=status,
        converged=False,
        iterations=iterations,
        schedule=None,
        increments=None,
        rollout=None,
        baseline_final=None,
        target=target.detach().clone(),
        budget=None,
        realized_target_delta_norm=None,
        path_length=None,
        per_step_norms=None,
        energy=None,
        model_error=None,
        fidelity_error=None,
        metrics=None,
        objective=None,
        fidelity_scale=fidelity_scale.detach().clone(),
        target_pairing_checked=False,
        target_pairing_exact=False,
        target_pairing_max_abs=None,
        schedule_valid=False,
        baseline_valid=False,
        nonfinite_detected=nonfinite,
    )


def solve_inverse_control(
    initial_state: Tensor,
    target: Tensor,
    velocity_fn: VelocityFn,
    *,
    control_mask: Tensor,
    target_mask: Tensor,
    model_to_physical_scale: Tensor,
    control_budget: Tensor | float | None = None,
    config: InverseControlConfig | None = None,
) -> InverseControlResult:
    """Find a budgeted schedule that transports ``initial_state`` to ``target``.

    The prefix through step 4 is evaluated once and detached.  Every optimizer
    trial therefore differentiates only through active steps 5--9.  Gradients
    are requested only for the integrated increments; caller model parameters
    never receive ``.grad`` values.
    """

    config = config or InverseControlConfig()
    config.validate()
    if not initial_state.is_floating_point() or target.dtype != initial_state.dtype:
        raise ValueError("initial_state and target must share a floating dtype")
    if target.shape != initial_state.shape or target.device != initial_state.device:
        raise ValueError("target shape and device must match initial_state")
    for name, mask in (("control_mask", control_mask), ("target_mask", target_mask)):
        if mask.shape != initial_state.shape or mask.dtype != torch.bool or mask.device != initial_state.device:
            raise ValueError(f"{name} must be a bool tensor matching initial_state")
        if not bool(mask.any().item()):
            raise ValueError(f"{name} must select at least one coordinate")
    if bool((control_mask & ~target_mask).any().item()):
        raise ValueError("control_mask must be a subset of target_mask")
    if not torch.equal(control_mask, first_five_xyz_mask_like(initial_state)):
        raise ValueError("R05A control_mask must be exactly first-five XYZ")
    if not torch.equal(
        target_mask, first_five_channels_mask_like(initial_state, channels=7)
    ):
        raise ValueError("R05A target_mask must be exactly first-five seven-channel")
    _validate_finite("initial_state", initial_state)
    _validate_finite("target", target)
    initial_state = initial_state.detach()
    target = target.detach()

    try:
        scale = torch.broadcast_to(model_to_physical_scale, initial_state.shape).detach()
    except RuntimeError as exc:
        raise ValueError("model_to_physical_scale is not broadcastable to initial_state") from exc
    if scale.dtype != initial_state.dtype or scale.device != initial_state.device:
        raise ValueError("model_to_physical_scale dtype/device must match initial_state")
    _validate_finite("model_to_physical_scale", scale)
    if not bool((scale > 0).all().item()):
        raise ValueError("model_to_physical_scale must be strictly positive")

    explicit_budget = None
    if control_budget is not None:
        explicit_budget = torch.as_tensor(
            control_budget,
            dtype=initial_state.dtype,
            device=initial_state.device,
        )
        if explicit_budget.ndim != 0 or not bool(torch.isfinite(explicit_budget).item()):
            raise ValueError("control_budget must be a finite scalar")
        if float(explicit_budget.item()) < 0.0:
            raise ValueError("control_budget must be nonnegative")

    dt_tensor = torch.tensor(config.dt, dtype=initial_state.dtype, device=initial_state.device)
    step_times = _sampler_times(initial_state, config)
    zero_state = torch.zeros_like(initial_state)
    zero_schedule = torch.zeros(
        (config.num_steps, *initial_state.shape),
        dtype=initial_state.dtype,
        device=initial_state.device,
    )

    # The inactive prefix is immutable for every candidate.  Evaluate it once
    # rather than retaining five unnecessary transformer graphs per iteration.
    prefix_states = [initial_state.detach()]
    prefix_velocities = []
    x_prefix = initial_state.detach()
    try:
        with torch.no_grad():
            for step in range(config.intervention_step):
                time = step_times[step]
                velocity = _call_velocity(velocity_fn, x_prefix, time, step)
                x_prefix = x_prefix + dt_tensor * velocity
                if not bool(torch.isfinite(x_prefix).all().item()):
                    raise _NonFiniteFlowError(f"Euler state after prefix step {step} is nonfinite")
                prefix_velocities.append(velocity.detach())
                prefix_states.append(x_prefix.detach())

            x_baseline = x_prefix
            baseline_active_states = [x_baseline.detach()]
            baseline_active_velocities = []
            for step in range(config.intervention_step, config.num_steps):
                time = step_times[step]
                velocity = _call_velocity(velocity_fn, x_baseline, time, step)
                x_baseline = x_baseline + dt_tensor * velocity
                if not bool(torch.isfinite(x_baseline).all().item()):
                    raise _NonFiniteFlowError(f"Euler state after baseline step {step} is nonfinite")
                baseline_active_velocities.append(velocity.detach())
                baseline_active_states.append(x_baseline.detach())
            baseline_final = x_baseline.detach()
    except _NonFiniteFlowError:
        return _failure_result(
            InverseControlStatus.NONFINITE,
            target,
            scale,
            iterations=0,
            nonfinite=True,
        )

    baseline_error = (baseline_final - target) * scale
    baseline_metrics = _fidelity_metrics(baseline_error, control_mask, target_mask, config)

    def assemble_rollout(
        schedule: Tensor,
        active_states: list[Tensor],
        active_base_velocities: list[Tensor],
        active_total_velocities: list[Tensor],
    ) -> FlowRollout:
        states = torch.cat(
            [
                torch.stack(prefix_states, dim=0),
                torch.stack(active_states[1:], dim=0),
            ],
            dim=0,
        )
        base_velocities = torch.cat(
            [
                torch.stack(prefix_velocities, dim=0),
                torch.stack(active_base_velocities, dim=0),
            ],
            dim=0,
        )
        total_velocities = torch.cat(
            [
                torch.stack(prefix_velocities, dim=0),
                torch.stack(active_total_velocities, dim=0),
            ],
            dim=0,
        )
        return FlowRollout(
            states=states,
            base_velocities=base_velocities,
            control_velocities=schedule,
            total_velocities=total_velocities,
            increments=dt_tensor * schedule,
            times=torch.stack(step_times, dim=0),
            final=states[-1],
        )

    baseline_rollout = assemble_rollout(
        zero_schedule,
        baseline_active_states,
        baseline_active_velocities,
        baseline_active_velocities,
    )
    outside_control = ~control_mask
    target_pairing_exact = _finite_values_bitwise_equal(
        target, baseline_final, outside_control
    )
    target_pairing_max_abs = torch.max(torch.abs(baseline_error[outside_control]))
    if not target_pairing_exact:
        with torch.no_grad():
            baseline_objective = _objective(
                baseline_error, control_mask, target_mask, config
            )
        return InverseControlResult(
            status=InverseControlStatus.TARGET_PAIRING_MISMATCH,
            converged=False,
            iterations=0,
            schedule=None,
            increments=None,
            rollout=_detach_rollout(baseline_rollout),
            baseline_final=baseline_final.detach().clone(),
            target=target.detach().clone(),
            budget=None,
            realized_target_delta_norm=None,
            path_length=None,
            per_step_norms=None,
            energy=None,
            model_error=(baseline_final - target).detach().clone(),
            fidelity_error=baseline_error.detach().clone(),
            metrics=_detach_metrics(baseline_metrics),
            objective=baseline_objective.detach().clone(),
            fidelity_scale=scale.detach().clone(),
            target_pairing_checked=True,
            target_pairing_exact=False,
            target_pairing_max_abs=target_pairing_max_abs.detach().clone(),
            schedule_valid=False,
            baseline_valid=True,
            nonfinite_detected=False,
        )

    delta = torch.where(control_mask, target - baseline_final, zero_state)
    realized_target_delta_norm = torch.linalg.vector_norm(delta.reshape(-1))
    # Real R05A runs pass the immutable source-R02 Delta* norm explicitly.
    # Do not silently replace that registered budget with the norm of
    # ``float32(target - baseline)``: target addition can change the latter by
    # one or more ULPs. Synthetic callers may omit the override and retain the
    # natural target-derived budget.
    budget = (
        realized_target_delta_norm
        if explicit_budget is None
        else explicit_budget
    )

    if float(budget.item()) == 0.0 and baseline_metrics.feasible:
        with torch.no_grad():
            baseline_objective = _objective(
                baseline_error, control_mask, target_mask, config
            )
        return InverseControlResult(
            status=InverseControlStatus.ZERO_BUDGET_CONVERGED,
            converged=True,
            iterations=0,
            schedule=zero_schedule,
            increments=zero_schedule,
            rollout=_detach_rollout(baseline_rollout),
            baseline_final=baseline_final,
            target=target.detach().clone(),
            budget=budget,
            realized_target_delta_norm=realized_target_delta_norm.detach().clone(),
            path_length=torch.zeros_like(budget),
            per_step_norms=torch.zeros(
                config.num_steps - config.intervention_step,
                dtype=initial_state.dtype,
                device=initial_state.device,
            ),
            energy=torch.zeros_like(budget),
            model_error=(baseline_final - target).detach().clone(),
            fidelity_error=baseline_error.detach().clone(),
            metrics=_detach_metrics(baseline_metrics),
            objective=baseline_objective.detach().clone(),
            fidelity_scale=scale.detach().clone(),
            target_pairing_checked=True,
            target_pairing_exact=True,
            target_pairing_max_abs=target_pairing_max_abs.detach().clone(),
            schedule_valid=True,
            baseline_valid=True,
            nonfinite_detected=False,
        )

    active_count = config.num_steps - config.intervention_step
    initial_increments = delta.unsqueeze(0).expand(active_count, *delta.shape) / active_count
    increments = _project_increments(
        initial_increments, control_mask, budget, config
    ).detach()
    first_moment = torch.zeros_like(increments)
    second_moment = torch.zeros_like(increments)

    best_increments = increments.detach().clone()
    best_objective: Tensor | None = None
    best_energy: Tensor | None = None
    best_feasible = False
    iterations_run = 0
    nonfinite_detected = False

    def active_rollout(
        candidate: Tensor,
        *,
        capture: bool = False,
    ) -> tuple[Tensor, Tensor, tuple[list[Tensor], list[Tensor], list[Tensor]] | None]:
        x_t = x_prefix.detach()
        captured_states = [x_t] if capture else None
        captured_base = [] if capture else None
        captured_total = [] if capture else None
        for local_index, step in enumerate(
            range(config.intervention_step, config.num_steps)
        ):
            time = step_times[step]
            velocity = _call_velocity(velocity_fn, x_t, time, step)
            control = candidate[local_index] / dt_tensor
            total_velocity = _controlled_velocity_for_solve(velocity, control)
            x_t = x_t + dt_tensor * total_velocity
            if not bool(torch.isfinite(x_t).all().item()):
                raise _NonFiniteFlowError(f"Euler state after trial step {step} is nonfinite")
            if capture:
                assert captured_states is not None
                assert captured_base is not None
                assert captured_total is not None
                captured_states.append(x_t)
                captured_base.append(velocity)
                captured_total.append(total_velocity)
        error = (x_t - target) * scale
        captured = None
        if capture:
            assert captured_states is not None
            assert captured_base is not None
            assert captured_total is not None
            captured = (captured_states, captured_base, captured_total)
        return x_t, error, captured

    for iteration in range(config.max_iterations + 1):
        with torch.enable_grad():
            candidate = increments.detach().requires_grad_(requires_grad=True)
            try:
                _terminal, fidelity_error, _captured = active_rollout(candidate)
            except _NonFiniteFlowError:
                nonfinite_detected = True
                break
            objective = _objective(fidelity_error, control_mask, target_mask, config)
            if not bool(torch.isfinite(objective).item()):
                nonfinite_detected = True
                break
            metrics = _fidelity_metrics(
                fidelity_error.detach(), control_mask, target_mask, config
            )
            energy = torch.sum(torch.square(candidate))
            candidate_objective = objective.detach()
            candidate_energy = energy.detach()

            better_fit = best_objective is None or bool(
                (candidate_objective < best_objective).item()
            )
            better_feasible = metrics.feasible and (
                not best_feasible
                or best_energy is None
                or bool((candidate_energy < best_energy).item())
                or (
                    bool((candidate_energy == best_energy).item())
                    and better_fit
                )
            )
            if better_feasible or (not best_feasible and better_fit):
                best_increments = candidate.detach().clone()
                best_objective = candidate_objective.clone()
                best_energy = candidate_energy.clone()
                best_feasible = metrics.feasible

            iterations_run = iteration
            if iteration == config.max_iterations:
                break

            gradient = torch.autograd.grad(objective, candidate, retain_graph=False)[0]
            if not bool(torch.isfinite(gradient).all().item()):
                nonfinite_detected = True
                break
        with torch.no_grad():
            first_moment = (
                config.adam_beta1 * first_moment
                + (1.0 - config.adam_beta1) * gradient
            )
            second_moment = (
                config.adam_beta2 * second_moment
                + (1.0 - config.adam_beta2) * torch.square(gradient)
            )
            update_index = iteration + 1
            corrected_first = first_moment / (1.0 - config.adam_beta1**update_index)
            corrected_second = second_moment / (1.0 - config.adam_beta2**update_index)
            updated = candidate - config.learning_rate * corrected_first / (
                torch.sqrt(corrected_second) + config.adam_epsilon
            )
            increments = _project_increments(
                updated, control_mask, budget, config
            ).detach()

    schedule = zero_schedule.clone()
    schedule[config.intervention_step :] = best_increments / dt_tensor
    try:
        with torch.no_grad():
            _terminal, _error, captured = active_rollout(
                best_increments, capture=True
            )
            assert captured is not None
            chosen_rollout = assemble_rollout(schedule, *captured)
    except _NonFiniteFlowError:
        return _failure_result(
            InverseControlStatus.NONFINITE,
            target,
            scale,
            iterations=iterations_run,
            nonfinite=True,
        )
    with torch.no_grad():
        chosen_increments, per_step, path = validate_schedule_constraints(
            schedule, control_mask, budget, config=config
        )
        model_error = chosen_rollout.final - target
        fidelity_error = model_error * scale
        metrics = _fidelity_metrics(
            fidelity_error, control_mask, target_mask, config
        )
        objective = _objective(
            fidelity_error, control_mask, target_mask, config
        )
    status = (
        InverseControlStatus.NONFINITE
        if nonfinite_detected
        else (
            InverseControlStatus.CONVERGED
            if metrics.feasible
            else InverseControlStatus.MAX_ITERATIONS
        )
    )
    return InverseControlResult(
        status=status,
        converged=status is InverseControlStatus.CONVERGED,
        iterations=iterations_run,
        schedule=schedule.detach().clone(),
        increments=chosen_increments.detach().clone(),
        rollout=_detach_rollout(chosen_rollout),
        baseline_final=baseline_final.detach().clone(),
        target=target.detach().clone(),
        budget=budget.detach().clone(),
        realized_target_delta_norm=realized_target_delta_norm.detach().clone(),
        path_length=path.detach().clone(),
        per_step_norms=per_step.detach().clone(),
        energy=torch.sum(torch.square(chosen_increments)).detach().clone(),
        model_error=model_error.detach().clone(),
        fidelity_error=fidelity_error.detach().clone(),
        metrics=_detach_metrics(metrics),
        objective=objective.detach().clone(),
        fidelity_scale=scale.detach().clone(),
        target_pairing_checked=True,
        target_pairing_exact=True,
        target_pairing_max_abs=target_pairing_max_abs.detach().clone(),
        schedule_valid=status is not InverseControlStatus.NONFINITE,
        baseline_valid=True,
        nonfinite_detected=nonfinite_detected,
    )
