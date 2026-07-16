"""Optimizer-free online lift of a terminal action delta into flow controls.

This module is deliberately independent of the historical inverse-control and
constrained-flow implementations.  It implements only the reference-trajectory
arithmetic used by the opt-in ``reference_trajectory_lift`` sampler mode.

For active Euler step ``k = 5 + j``, ``reference_states[j + 1]`` is already
the fully constructed registered reference ``xbar + alpha * delta``.  The raw
integrated control increment is the first-five-XYZ part of the difference
between that desired state and the actual uncontrolled next state.
The optional product-ball mode projects each of the five 15-coordinate
increments independently onto radius ``budget / 5``.  Its norm and projection
scale are computed by an explicit fixed-order float64 reduction and are exposed
to the caller for independent reconstruction.  The requested increment is cast
once to float32, transported to a requested velocity, and transported back to
the authoritative executed float32 increment before it enters budget audits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

RAW_PROJECTION = "raw"
PRODUCT_BALL_PROJECTION = "product_ball"
PROJECTION_MODES = frozenset({RAW_PROJECTION, PRODUCT_BALL_PROJECTION})


@dataclass(frozen=True)
class ReferenceStep:
    """Detached-free tensors needed to apply and audit one online step."""

    desired_next: Tensor
    uncontrolled_next: Tensor
    raw_increment: Tensor
    requested_increment: Tensor
    requested_velocity: Tensor
    executed_increment: Tensor
    raw_norm_f64: Tensor
    requested_norm_f64: Tensor
    executed_norm_f64: Tensor
    projection_scale_f64: Tensor
    projected: Tensor


class RawFloat32Unrepresentable(RuntimeError):  # noqa: N818 - explicit apparatus error
    """Abort a raw lift before applying a nonfinite prospective float32 leaf."""

    error_type = "RAW_FLOAT32_UNREPRESENTABLE"

    def __init__(self, step: int, leaf: str):
        self.step = int(step)
        self.leaf = str(leaf)
        super().__init__(f"{self.error_type}({self.step},{self.leaf})")


def require_finite_reference_leaf(
    value: Tensor,
    *,
    projection_mode: str,
    step: int,
    leaf: str,
) -> None:
    """Fail before application; callers must treat this as apparatus-inconclusive."""

    if bool(torch.isfinite(value).all().item()):
        return
    if projection_mode == RAW_PROJECTION:
        raise RawFloat32Unrepresentable(step, leaf)
    raise RuntimeError(f"reference_trajectory_lift step {step} produced nonfinite {leaf}")


def reference_alpha_like(value: Tensor) -> Tensor:
    """Return the frozen six-knot alpha schedule on ``value``'s device."""

    if value.dtype != torch.float32:
        raise ValueError("reference alpha requires a float32 tensor")
    alpha = torch.zeros((6,), dtype=torch.float32, device=value.device)
    numerators = torch.arange(1, 6, dtype=torch.float32, device=value.device)
    denominator = torch.tensor(5, dtype=torch.float32, device=value.device)
    alpha[1:] = numerators / denominator
    return alpha


def first_five_xyz_mask_like(value: Tensor) -> Tensor:
    """Return the exact first-five-action, XYZ-channel mask."""

    if value.ndim < 2 or value.shape[-2] < 5 or value.shape[-1] < 3:
        raise ValueError("reference value must expose at least five action rows and XYZ")
    mask = torch.zeros_like(value, dtype=torch.bool)
    mask[..., :5, :3] = True
    return mask


def finite_bitwise_equal(left: Tensor, right: Tensor) -> bool:
    """Compare finite tensors exactly, including the sign bit of zero."""

    return bool(
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and bool(torch.isfinite(left).all().item())
        and bool(torch.isfinite(right).all().item())
        and torch.equal(left, right)
        and torch.equal(torch.signbit(left), torch.signbit(right))
    )


def fixed_order_first_five_xyz_norm_f64(value: Tensor) -> Tensor:
    """Compute a 15-coordinate L2 norm with an explicit float64 order.

    The returned shape is ``value.shape[:-2]``.  No fused/vector reduction is
    used, so an independent validator can reproduce the exact accumulation
    order from the persisted float32 coordinates.
    """

    if value.dtype != torch.float32:
        raise ValueError("reference norm input must preserve float32")
    selected = value[..., :5, :3].reshape(*value.shape[:-2], 15).to(torch.float64)
    squared_sum = torch.zeros(selected.shape[:-1], dtype=torch.float64, device=value.device)
    for coordinate in range(15):
        component = selected[..., coordinate]
        squared_sum = squared_sum + component * component
    return torch.sqrt(squared_sum)


def validate_reference_inputs(
    initial_state: Tensor,
    reference_states: Tensor,
    delta: Tensor,
    budget: Tensor,
    projection_mode: str,
) -> None:
    """Validate the complete strict online-lift tensor contract."""

    if projection_mode not in PROJECTION_MODES:
        raise ValueError(
            "reference projection_mode must be 'raw' or 'product_ball'"
        )
    if initial_state.dtype != torch.float32:
        raise ValueError("reference initial state must preserve float32")
    expected_reference_shape = (initial_state.shape[0], 6, *initial_state.shape[1:])
    if tuple(reference_states.shape) != expected_reference_shape:
        raise ValueError(
            f"reference_states shape {tuple(reference_states.shape)} does not match "
            f"{expected_reference_shape}"
        )
    if tuple(delta.shape) != tuple(initial_state.shape):
        raise ValueError(
            f"reference delta shape {tuple(delta.shape)} does not match {tuple(initial_state.shape)}"
        )
    for name, value in (("reference_states", reference_states), ("reference delta", delta)):
        if value.dtype != torch.float32 or value.device != initial_state.device:
            raise ValueError(f"{name} must match initial-state float32 dtype and device")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} contains a nonfinite value")
    if budget.ndim != 0 or budget.dtype != torch.float32 or budget.device != initial_state.device:
        raise ValueError("reference budget must be a scalar float32 tensor on the sampler device")
    if not bool(torch.isfinite(budget).item()) or float(budget.item()) <= 0.0:
        raise ValueError("reference budget must be finite and positive")

    mask = first_five_xyz_mask_like(delta)
    outside = delta[~mask]
    if bool(torch.count_nonzero(outside).item()) or bool(torch.signbit(outside).any().item()):
        raise ValueError("reference delta must be exact positive zero outside first-five XYZ")


def reference_step(
    x_t: Tensor,
    v_base: Tensor,
    reference_next: Tensor,
    budget: Tensor,
    dt: Tensor,
    projection_mode: str,
    *,
    step_index: int,
) -> ReferenceStep:
    """Compute one raw or product-ball projected online control."""

    if x_t.dtype != torch.float32 or v_base.dtype != torch.float32:
        raise ValueError("reference online state and base velocity must preserve float32")
    if x_t.shape != v_base.shape or x_t.shape != reference_next.shape:
        raise ValueError("reference online tensors must have identical shapes")
    if dt.ndim != 0 or dt.dtype != torch.float32 or dt.device != x_t.device:
        raise ValueError("reference dt must be scalar float32 on the sampler device")
    if not bool(torch.isfinite(dt).item()) or float(dt.item()) == 0.0:
        raise ValueError("reference dt must be finite and nonzero")
    if projection_mode not in PROJECTION_MODES:
        raise ValueError("unsupported reference projection mode")

    for leaf, value in (
        ("state", x_t),
        ("base_velocity", v_base),
        ("reference_next", reference_next),
    ):
        require_finite_reference_leaf(
            value,
            projection_mode=projection_mode,
            step=step_index,
            leaf=leaf,
        )

    mask = first_five_xyz_mask_like(x_t)
    desired_next = reference_next
    uncontrolled_next = x_t + dt * v_base
    require_finite_reference_leaf(
        uncontrolled_next,
        projection_mode=projection_mode,
        step=step_index,
        leaf="uncontrolled_next",
    )
    raw_increment = torch.where(
        mask,
        desired_next - uncontrolled_next,
        torch.zeros_like(x_t),
    )
    require_finite_reference_leaf(
        raw_increment,
        projection_mode=projection_mode,
        step=step_index,
        leaf="raw_increment",
    )

    raw_norm_f64 = fixed_order_first_five_xyz_norm_f64(raw_increment)
    projection_scale_f64 = torch.ones_like(raw_norm_f64)
    projected = torch.zeros_like(raw_norm_f64, dtype=torch.bool)
    requested_increment = raw_increment
    if projection_mode == PRODUCT_BALL_PROJECTION:
        cap_f32 = budget / torch.tensor(5.0, dtype=torch.float32, device=x_t.device)
        cap_f64 = cap_f32.to(torch.float64)
        projected = raw_norm_f64 > cap_f64
        safe_norm = torch.where(projected, raw_norm_f64, torch.ones_like(raw_norm_f64))
        candidate_scale = cap_f64 / safe_norm
        projection_scale_f64 = torch.where(
            projected,
            candidate_scale,
            projection_scale_f64,
        )
        scale_shape = (*projection_scale_f64.shape, 1, 1)
        scaled = (raw_increment.to(torch.float64) * projection_scale_f64.reshape(scale_shape)).to(
            torch.float32
        )
        projected_shape = (*projected.shape, 1, 1)
        requested_increment = torch.where(
            projected.reshape(projected_shape),
            scaled,
            raw_increment,
        )
        requested_increment = torch.where(
            mask,
            requested_increment,
            torch.zeros_like(requested_increment),
        )

    require_finite_reference_leaf(
        requested_increment,
        projection_mode=projection_mode,
        step=step_index,
        leaf="requested_increment",
    )
    requested_norm_f64 = fixed_order_first_five_xyz_norm_f64(requested_increment)
    requested_velocity = torch.where(
        mask,
        requested_increment / dt,
        torch.zeros_like(requested_increment),
    )
    require_finite_reference_leaf(
        requested_velocity,
        projection_mode=projection_mode,
        step=step_index,
        leaf="requested_velocity",
    )
    # This float32 c -> u -> c seam is authoritative for both the trace and
    # every registered budget check.  The projected/requested increment is
    # retained separately because division and multiplication need not round
    # back to identical bytes.
    executed_increment = torch.where(
        mask,
        dt * requested_velocity,
        torch.zeros_like(requested_velocity),
    )
    require_finite_reference_leaf(
        executed_increment,
        projection_mode=projection_mode,
        step=step_index,
        leaf="executed_increment",
    )
    executed_norm_f64 = fixed_order_first_five_xyz_norm_f64(executed_increment)
    return ReferenceStep(
        desired_next=desired_next,
        uncontrolled_next=uncontrolled_next,
        raw_increment=raw_increment,
        requested_increment=requested_increment,
        requested_velocity=requested_velocity,
        executed_increment=executed_increment,
        raw_norm_f64=raw_norm_f64,
        requested_norm_f64=requested_norm_f64,
        executed_norm_f64=executed_norm_f64,
        projection_scale_f64=projection_scale_f64,
        projected=projected,
    )


__all__ = [
    "PRODUCT_BALL_PROJECTION",
    "PROJECTION_MODES",
    "RAW_PROJECTION",
    "RawFloat32Unrepresentable",
    "ReferenceStep",
    "finite_bitwise_equal",
    "first_five_xyz_mask_like",
    "fixed_order_first_five_xyz_norm_f64",
    "reference_alpha_like",
    "reference_step",
    "require_finite_reference_leaf",
    "validate_reference_inputs",
]
