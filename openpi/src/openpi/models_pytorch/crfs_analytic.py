"""Differentiable analytic trajectory field for the opt-in CRFS kill test.

This module contains no VLA/model calls.  It treats an approximate-clean action
as an independent leaf, applies the frozen action affine and H04 response, and
differentiates only the analytic sphere/OBB margin energy.  The identity leaf is
what prevents the field from backpropagating through the frozen policy.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F  # noqa: N812


EXECUTED_ACTIONS = 5
TRANSLATION_DIMS = 3
SAMPLES_PER_SEGMENT = 26
SAFETY_MARGIN_M = 0.005
SOFTPLUS_TAU_M = 0.005


def _require_tensor(
    value: Tensor,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    device: torch.device | None = None,
) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must be float32, got {value.dtype}")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} device {value.device} does not match analytic field device {device}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains a nonfinite value")


def validate_analytic_controls(
    *,
    action_horizon: int,
    action_dim: int,
    device: torch.device,
    action_offset_xyz: Tensor,
    action_scale_xyz: Tensor,
    branch_eef_center_m: Tensor,
    response_matrix_m_per_action: Tensor,
    obstacle_centers_m: Tensor,
    obstacle_rotations_world: Tensor,
    obstacle_half_sizes_m: Tensor,
    eef_radius_m: Tensor,
    model_l2_path_budget: Tensor,
) -> None:
    """Fail closed on the complete tensor contract before any model work."""

    if action_horizon < EXECUTED_ACTIONS or action_dim < TRANSLATION_DIMS:
        raise ValueError("analytic trajectory field requires at least five actions and three action dimensions")
    _require_tensor(action_offset_xyz, name="crfs_action_offset_xyz", shape=(3,), device=device)
    _require_tensor(action_scale_xyz, name="crfs_action_scale_xyz", shape=(3,), device=device)
    _require_tensor(branch_eef_center_m, name="crfs_branch_eef_center_m", shape=(3,), device=device)
    _require_tensor(
        response_matrix_m_per_action,
        name="crfs_response_matrix_m_per_action",
        shape=(3, 3),
        device=device,
    )
    _require_tensor(eef_radius_m, name="crfs_eef_radius_m", shape=(), device=device)
    _require_tensor(model_l2_path_budget, name="crfs_model_l2_path_budget", shape=(), device=device)

    if obstacle_centers_m.ndim != 2 or obstacle_centers_m.shape[1] != 3 or obstacle_centers_m.shape[0] < 1:
        raise ValueError(
            "crfs_obstacle_centers_m must have shape (num_obbs>=1, 3), "
            f"got {tuple(obstacle_centers_m.shape)}"
        )
    num_obbs = int(obstacle_centers_m.shape[0])
    _require_tensor(
        obstacle_centers_m,
        name="crfs_obstacle_centers_m",
        shape=(num_obbs, 3),
        device=device,
    )
    _require_tensor(
        obstacle_rotations_world,
        name="crfs_obstacle_rotations_world",
        shape=(num_obbs, 3, 3),
        device=device,
    )
    _require_tensor(
        obstacle_half_sizes_m,
        name="crfs_obstacle_half_sizes_m",
        shape=(num_obbs, 3),
        device=device,
    )
    if not bool((action_scale_xyz > 0).all().item()):
        raise ValueError("crfs_action_scale_xyz must be strictly positive")
    if not bool((obstacle_half_sizes_m > 0).all().item()):
        raise ValueError("crfs_obstacle_half_sizes_m must be strictly positive")
    if not bool((eef_radius_m > 0).item()):
        raise ValueError("crfs_eef_radius_m must be strictly positive")
    if not bool((model_l2_path_budget > 0).item()):
        raise ValueError("crfs_model_l2_path_budget must be strictly positive")

    identity = torch.eye(3, dtype=torch.float32, device=device).expand(num_obbs, -1, -1)
    gram = obstacle_rotations_world.transpose(1, 2) @ obstacle_rotations_world
    determinant = torch.linalg.det(obstacle_rotations_world)
    if not bool(torch.allclose(gram, identity, rtol=1.0e-5, atol=1.0e-5)) or not bool(
        torch.allclose(determinant, torch.ones_like(determinant), rtol=1.0e-5, atol=1.0e-5)
    ):
        raise ValueError("crfs_obstacle_rotations_world must contain proper orthonormal rotations")


def predict_eef_trajectory(
    approximate_clean: Tensor,
    *,
    action_offset_xyz: Tensor,
    action_scale_xyz: Tensor,
    branch_eef_center_m: Tensor,
    response_matrix_m_per_action: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return physical XYZ commands, five EEF centers, and 26 samples/segment."""

    physical_xyz = (
        approximate_clean[:, :EXECUTED_ACTIONS, :TRANSLATION_DIMS] * action_scale_xyz
        + action_offset_xyz
    )
    displacement_m = physical_xyz @ response_matrix_m_per_action.transpose(0, 1)
    eef_centers_m = branch_eef_center_m.reshape(1, 1, 3) + torch.cumsum(displacement_m, dim=1)
    segment_starts_m = torch.cat(
        [
            branch_eef_center_m.reshape(1, 1, 3).expand(approximate_clean.shape[0], -1, -1),
            eef_centers_m[:, :-1, :],
        ],
        dim=1,
    )
    alpha = torch.linspace(
        0.0,
        1.0,
        SAMPLES_PER_SEGMENT,
        dtype=approximate_clean.dtype,
        device=approximate_clean.device,
    ).reshape(1, 1, SAMPLES_PER_SEGMENT, 1)
    sampled_centers_m = segment_starts_m[:, :, None, :] + alpha * (
        eef_centers_m[:, :, None, :] - segment_starts_m[:, :, None, :]
    )
    return physical_xyz, eef_centers_m, sampled_centers_m


def _sphere_obb_clearances(
    sampled_centers_m: Tensor,
    *,
    obstacle_centers_m: Tensor,
    obstacle_rotations_world: Tensor,
    obstacle_half_sizes_m: Tensor,
    eef_radius_m: Tensor,
) -> Tensor:
    world_delta = sampled_centers_m[:, :, :, None, :] - obstacle_centers_m.reshape(1, 1, 1, -1, 3)
    # Simulator rotations store local axes as world-frame columns, hence
    # local = R^T (world - center).
    local = torch.einsum("bskni,nij->bsknj", world_delta, obstacle_rotations_world)
    extent_delta = torch.abs(local) - obstacle_half_sizes_m.reshape(1, 1, 1, -1, 3)
    outside = torch.linalg.vector_norm(torch.clamp_min(extent_delta, 0.0), dim=-1)
    inside = torch.clamp_max(torch.max(extent_delta, dim=-1).values, 0.0)
    return outside + inside - eef_radius_m


def analytic_trajectory_field(
    approximate_clean: Tensor,
    *,
    action_offset_xyz: Tensor,
    action_scale_xyz: Tensor,
    branch_eef_center_m: Tensor,
    response_matrix_m_per_action: Tensor,
    obstacle_centers_m: Tensor,
    obstacle_rotations_world: Tensor,
    obstacle_half_sizes_m: Tensor,
    eef_radius_m: Tensor,
) -> dict[str, Tensor]:
    """Differentiate the frozen full-trajectory margin energy.

    The input is detached into a new leaf.  Thus autograd sees the requested
    identity approximate-clean Jacobian and has no graph path into the VLA.
    """

    if approximate_clean.ndim != 3:
        raise ValueError(f"approximate_clean must have rank 3, got {tuple(approximate_clean.shape)}")
    if approximate_clean.shape[1] < EXECUTED_ACTIONS or approximate_clean.shape[2] < TRANSLATION_DIMS:
        raise ValueError("approximate_clean must contain the first five translation commands")
    if approximate_clean.dtype != torch.float32:
        raise ValueError(f"approximate_clean must be float32, got {approximate_clean.dtype}")
    if not bool(torch.isfinite(approximate_clean).all().item()):
        raise ValueError("approximate_clean contains a nonfinite value")

    with torch.enable_grad():
        clean_leaf = approximate_clean.detach().requires_grad_(True)
        physical_xyz, eef_centers_m, sampled_centers_m = predict_eef_trajectory(
            clean_leaf,
            action_offset_xyz=action_offset_xyz,
            action_scale_xyz=action_scale_xyz,
            branch_eef_center_m=branch_eef_center_m,
            response_matrix_m_per_action=response_matrix_m_per_action,
        )
        clearances_m = _sphere_obb_clearances(
            sampled_centers_m,
            obstacle_centers_m=obstacle_centers_m,
            obstacle_rotations_world=obstacle_rotations_world,
            obstacle_half_sizes_m=obstacle_half_sizes_m,
            eef_radius_m=eef_radius_m,
        )
        hard_min_clearance_m = torch.amin(clearances_m, dim=(1, 2, 3))
        margin_argument = (SAFETY_MARGIN_M - clearances_m) / SOFTPLUS_TAU_M
        energy = torch.sum(F.softplus(margin_argument).square(), dim=(1, 2, 3))
        margin_satisfied = hard_min_clearance_m >= SAFETY_MARGIN_M
        # Safe batch members retain diagnostics but contribute no gradient.
        unsafe_energy = torch.sum(energy * (~margin_satisfied).to(energy.dtype))
        raw_gradient = torch.autograd.grad(unsafe_energy, clean_leaf, create_graph=False)[0]

    energy_gradient = torch.zeros_like(raw_gradient)
    energy_gradient[:, :EXECUTED_ACTIONS, :TRANSLATION_DIMS] = raw_gradient[
        :, :EXECUTED_ACTIONS, :TRANSLATION_DIMS
    ]
    gradient_l2 = torch.linalg.vector_norm(
        energy_gradient[:, :EXECUTED_ACTIONS, :TRANSLATION_DIMS].reshape(energy_gradient.shape[0], -1),
        dim=1,
    )
    finite = (
        torch.isfinite(energy)
        & torch.isfinite(hard_min_clearance_m)
        & torch.isfinite(gradient_l2)
        & torch.isfinite(energy_gradient).reshape(energy_gradient.shape[0], -1).all(dim=1)
    )
    gradient_valid = finite & (gradient_l2 > 0.0)
    applied = (~margin_satisfied) & gradient_valid
    denominator = torch.clamp_min(gradient_l2, torch.finfo(gradient_l2.dtype).tiny).reshape(-1, 1, 1)
    normalized_gradient = torch.where(
        applied.reshape(-1, 1, 1),
        energy_gradient / denominator,
        torch.zeros_like(energy_gradient),
    )
    return {
        "physical_predicted_clean_xyz": physical_xyz.detach(),
        "predicted_eef_centers_m": eef_centers_m.detach(),
        "hard_min_clearance_m": hard_min_clearance_m.detach(),
        "energy": energy.detach(),
        "energy_gradient": energy_gradient.detach(),
        "normalized_energy_gradient": normalized_gradient.detach(),
        "gradient_l2": gradient_l2.detach(),
        "margin_satisfied": margin_satisfied.detach(),
        "gradient_finite": finite.detach(),
        "gradient_valid": gradient_valid.detach(),
        "applied": applied.detach(),
    }


def scale_field_to_velocity(
    normalized_energy_gradient: Tensor,
    applied: Tensor,
    *,
    model_l2_path_budget: Tensor,
    active_horizon: float,
) -> Tensor:
    """Additive reverse-time velocity with integrated path budget ``B``."""

    if active_horizon <= 0.0:
        raise ValueError("analytic trajectory field has no remaining integration horizon")
    gain = model_l2_path_budget / active_horizon
    return torch.where(
        applied.reshape(-1, 1, 1),
        normalized_energy_gradient * gain,
        torch.zeros_like(normalized_energy_gradient),
    )
