"""Opt-in adapter for the CFS linearized-warm-start comparison.

The historical inverse-flow implementation is an immutable experiment input.
This module therefore installs process-local wrappers around its two public
call sites instead of editing the solver or sampler.  With no request context,
both wrappers immediately call the saved historical functions with the exact
arguments they received.

Only the separate :mod:`scripts.serve_cfs_policy` executable installs these
hooks.  A request selects the comparison through the reserved
``__crfs__.experiment_arm`` field.  The field is removed before the ordinary
Policy sees the request, and a :class:`ContextVar` keeps the injected candidate
and diagnostics isolated between concurrent callers.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass
from enum import Enum
import threading
from typing import Any

import numpy as np
import torch

from openpi.models_pytorch import crfs_inverse_control as _inverse_control
from openpi.models_pytorch import crfs_linearized_control as _linearized_control


_EXPERIMENT_ARM = "linearized_warm_start"
_TRACE_KEY = "linearized_warm_start"
_MISSING = object()


@dataclass
class _ComparisonContext:
    arm: str
    phase: str = "waiting_for_solver"
    solve_calls: int = 0
    historical_projection_calls: int = 0
    injection_count: int = 0
    projection_idempotence_checks: int = 0
    projection_idempotence_exact: bool = False
    candidate: torch.Tensor | None = None
    diagnostic: dict[str, Any] | None = None


_REQUEST_CONTEXT: ContextVar[_ComparisonContext | None] = ContextVar(
    "crfs_constrained_flow_request",
    default=None,
)

_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
_ORIGINAL_SOLVE_INVERSE_CONTROL = _inverse_control.solve_inverse_control
_ORIGINAL_PROJECT_INCREMENTS = _inverse_control._project_increments


def _wire_value(value: Any) -> Any:
    """Detach a linear-solver diagnostic into a msgpack-compatible tree."""

    if isinstance(value, torch.Tensor):
        return np.asarray(value.detach().cpu().numpy())
    if isinstance(value, Enum):
        return _wire_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _wire_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise RuntimeError("CFS diagnostic mappings must have string keys")
        return {key: _wire_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_wire_value(item) for item in value)
    if isinstance(value, list):
        return [_wire_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.asarray(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise RuntimeError(f"unsupported CFS diagnostic value type: {type(value).__name__}")


def _linearized_diagnostic(result: Any) -> dict[str, Any]:
    if result is None or not is_dataclass(result):
        raise RuntimeError("CFS linearized solver did not return its required dataclass diagnostic")
    diagnostic = _wire_value(result)
    if not isinstance(diagnostic, dict) or not diagnostic:
        raise RuntimeError("CFS linearized solver returned an empty diagnostic")
    return diagnostic


def _same_tensor_contract(candidate: torch.Tensor, historical_input: torch.Tensor) -> bool:
    return bool(
        isinstance(candidate, torch.Tensor)
        and candidate.shape == historical_input.shape
        and candidate.dtype == historical_input.dtype
        and candidate.device == historical_input.device
        and bool(torch.isfinite(candidate).all().item())
    )


def _patched_project_increments(
    increments: torch.Tensor,
    control_mask: torch.Tensor,
    budget: torch.Tensor,
    config: _inverse_control.InverseControlConfig,
) -> torch.Tensor:
    context = _REQUEST_CONTEXT.get()
    if context is None or context.phase == "computing_linearized_candidate":
        return _ORIGINAL_PROJECT_INCREMENTS(increments, control_mask, budget, config)
    if context.phase != "running_historical_refinement":
        raise RuntimeError(f"CFS projection called in invalid phase {context.phase!r}")

    context.historical_projection_calls += 1
    if context.injection_count == 0:
        if context.candidate is None:
            raise RuntimeError("CFS historical initialization has no linearized candidate")
        if not _same_tensor_contract(context.candidate, increments):
            raise RuntimeError("CFS linearized candidate violates the historical initializer tensor contract")
        context.injection_count += 1
        # The linearized solver has already run the production projection once.
        # Re-evaluate it here only as a bitwise idempotence check, then return
        # the recorded candidate so the comparison cannot incur a second
        # rounding step.  Every later Adam projection runs normally.
        checked = _ORIGINAL_PROJECT_INCREMENTS(context.candidate, control_mask, budget, config)
        context.projection_idempotence_checks += 1
        context.projection_idempotence_exact = bool(
            torch.equal(checked, context.candidate)
            and torch.equal(torch.signbit(checked), torch.signbit(context.candidate))
        )
        if not context.projection_idempotence_exact:
            raise RuntimeError("CFS production projection is not bitwise idempotent on the injected candidate")
        return context.candidate
    return _ORIGINAL_PROJECT_INCREMENTS(increments, control_mask, budget, config)


def _patched_solve_inverse_control(
    initial_state: torch.Tensor,
    target: torch.Tensor,
    velocity_fn: _inverse_control.VelocityFn,
    *,
    control_mask: torch.Tensor,
    target_mask: torch.Tensor,
    model_to_physical_scale: torch.Tensor,
    control_budget: torch.Tensor | float | None = None,
    config: _inverse_control.InverseControlConfig | None = None,
) -> _inverse_control.InverseControlResult:
    context = _REQUEST_CONTEXT.get()
    if context is None:
        return _ORIGINAL_SOLVE_INVERSE_CONTROL(
            initial_state,
            target,
            velocity_fn,
            control_mask=control_mask,
            target_mask=target_mask,
            model_to_physical_scale=model_to_physical_scale,
            control_budget=control_budget,
            config=config,
        )
    if context.arm != _EXPERIMENT_ARM:
        raise RuntimeError(f"CFS solver received unsupported experiment arm {context.arm!r}")
    if context.phase != "waiting_for_solver" or context.solve_calls != 0:
        raise RuntimeError("CFS comparison request must invoke the inverse solver exactly once")
    if not isinstance(config, _inverse_control.InverseControlConfig):
        raise RuntimeError("CFS comparison requires an explicit InverseControlConfig")
    if control_budget is None:
        raise RuntimeError("CFS comparison requires an explicit control budget")

    context.solve_calls += 1
    context.phase = "computing_linearized_candidate"
    linearized_result = _linearized_control.solve_linearized_control(
        initial_state,
        target,
        velocity_fn,
        control_mask=control_mask,
        target_mask=target_mask,
        model_to_physical_scale=model_to_physical_scale,
        control_budget=control_budget,
        legacy_config=config,
    )
    finite_difference = getattr(linearized_result, "finite_difference", None)
    if (
        getattr(linearized_result, "jacobian_valid", None) is not True
        or finite_difference is None
        or getattr(finite_difference, "passed", None) is not True
    ):
        raise RuntimeError(
            "CFS refuses to inject a candidate whose registered Jacobian validation did not pass"
        )
    candidate = getattr(linearized_result, "active_increments", None)
    if candidate is None or not isinstance(candidate, torch.Tensor):
        raise RuntimeError("CFS linearized solver did not return active_increments")
    context.candidate = candidate.detach().clone()
    context.diagnostic = _linearized_diagnostic(linearized_result)

    context.phase = "running_historical_refinement"
    result = _ORIGINAL_SOLVE_INVERSE_CONTROL(
        initial_state,
        target,
        velocity_fn,
        control_mask=control_mask,
        target_mask=target_mask,
        model_to_physical_scale=model_to_physical_scale,
        control_budget=control_budget,
        config=config,
    )
    context.phase = "historical_refinement_complete"
    if context.injection_count != 1:
        raise RuntimeError(
            "CFS historical refinement must inject exactly once; "
            f"observed {context.injection_count}"
        )
    return result


def install_constrained_flow_hooks() -> None:
    """Install the process-local comparison hooks exactly once.

    Reinstallation or installation over any pre-existing monkeypatch is an
    apparatus error.  The ordinary server never imports or calls this function.
    """

    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            raise RuntimeError("CFS constrained-flow hooks are already installed")
        if _inverse_control.solve_inverse_control is not _ORIGINAL_SOLVE_INVERSE_CONTROL:
            raise RuntimeError("CFS refuses to replace an already modified inverse solver")
        if _inverse_control._project_increments is not _ORIGINAL_PROJECT_INCREMENTS:
            raise RuntimeError("CFS refuses to replace an already modified production projector")
        _inverse_control.solve_inverse_control = _patched_solve_inverse_control
        _inverse_control._project_increments = _patched_project_increments
        _INSTALLED = True


class ConstrainedFlowPolicyAdapter:
    """Request-local wrapper around the ordinary OpenPI Policy."""

    def __init__(self, policy: Any):
        if not _INSTALLED:
            raise RuntimeError("CFS hooks must be installed before constructing the policy adapter")
        self._policy = policy

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata

    def _delegate(self, obs: dict, noise: np.ndarray | None) -> dict:
        if noise is None:
            return self._policy.infer(obs)
        return self._policy.infer(obs, noise=noise)

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        controls = obs.get("__crfs__")
        if not isinstance(controls, Mapping) or "experiment_arm" not in controls:
            # Preserve the exact object and invocation surface for every
            # request that does not opt into this separate experiment server.
            return self._delegate(obs, noise)

        experiment_arm = controls.get("experiment_arm", _MISSING)
        if experiment_arm != _EXPERIMENT_ARM:
            raise ValueError(
                "Unsupported __crfs__.experiment_arm; expected "
                f"{_EXPERIMENT_ARM!r}, got {experiment_arm!r}"
            )
        if _REQUEST_CONTEXT.get() is not None:
            raise RuntimeError("nested CFS comparison requests are forbidden")

        ordinary_controls = dict(controls)
        removed_arm = ordinary_controls.pop("experiment_arm", _MISSING)
        if removed_arm != _EXPERIMENT_ARM:
            raise RuntimeError("CFS failed to remove the registered experiment arm")
        ordinary_obs = dict(obs)
        ordinary_obs["__crfs__"] = ordinary_controls

        context = _ComparisonContext(arm=_EXPERIMENT_ARM)
        token = _REQUEST_CONTEXT.set(context)
        try:
            outputs = self._delegate(ordinary_obs, noise)
            if context.solve_calls != 1:
                raise RuntimeError(
                    "CFS comparison request must invoke the inverse solver exactly once; "
                    f"observed {context.solve_calls}"
                )
            if context.injection_count != 1:
                raise RuntimeError(
                    "CFS comparison request must inject exactly once; "
                    f"observed {context.injection_count}"
                )
            if (
                context.projection_idempotence_checks != 1
                or not context.projection_idempotence_exact
            ):
                raise RuntimeError(
                    "CFS comparison request is missing its exact production-projection check"
                )
            if context.diagnostic is None:
                raise RuntimeError("CFS comparison request is missing its linearized diagnostic")
            if not isinstance(outputs, Mapping):
                raise RuntimeError("CFS ordinary Policy returned a non-mapping result")
            trace = outputs.get("crfs_trace")
            if not isinstance(trace, Mapping):
                raise RuntimeError("CFS ordinary Policy result is missing crfs_trace")
            if _TRACE_KEY in trace:
                raise RuntimeError(f"CFS trace key {_TRACE_KEY!r} already exists")

            diagnostic = dict(context.diagnostic)
            if context.candidate is None:
                raise RuntimeError("CFS comparison request lost its injected candidate")
            diagnostic.update(
                experiment_arm=_EXPERIMENT_ARM,
                adapter_solve_calls=context.solve_calls,
                adapter_historical_projection_calls=context.historical_projection_calls,
                adapter_injection_count=context.injection_count,
                adapter_projection_idempotence_checks=context.projection_idempotence_checks,
                adapter_projection_idempotence_exact=context.projection_idempotence_exact,
                adapter_injected_initial_increments=_wire_value(context.candidate),
                adapter_only_historical_initialization_changed=True,
                optimality_certificate=False,
                infeasibility_certificate=False,
                nonlinear_feasibility_certificate=False,
            )
            outputs_with_trace = dict(outputs)
            trace_with_diagnostic = dict(trace)
            trace_with_diagnostic[_TRACE_KEY] = diagnostic
            outputs_with_trace["crfs_trace"] = trace_with_diagnostic
            return outputs_with_trace
        finally:
            _REQUEST_CONTEXT.reset(token)


__all__ = [
    "ConstrainedFlowPolicyAdapter",
    "install_constrained_flow_hooks",
]
