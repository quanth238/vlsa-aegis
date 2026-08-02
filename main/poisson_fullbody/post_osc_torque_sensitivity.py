"""Exact local one-substep torque sensitivity after an OSC command.

The intended call site is the pre-physics callback immediately after the
ordinary OSC controller has written ``data.ctrl``.  The live ``MjData`` is
never stepped, forwarded, edited, or restored by this module.  Instead, the
complete official ``mjSTATE_INTEGRATION`` vector is captured once and every
nominal or perturbed transition is evaluated in a fresh ``MjData`` clone.

For arm actuator ``j`` the preferred local column is a centered difference.
If the requested centered stencil would cross a compiled actuator bound, the
column instead uses a bound-respecting forward or backward difference.  No
perturbed torque is clipped.  The selected stencil is evaluated at two
positive resolutions, ``h`` and ``h / 2``, and fails closed unless the two
estimates agree within caller-configured absolute and relative tolerances.
The smaller-resolution estimate is returned.  Exact sampled torque deltas are
recorded per column so an artifact consumer can reconstruct every quotient.

Each clone calls ``mj_step`` exactly once, so ``S`` already includes one
MuJoCo timestep and any local constraint or contact response.  It must not be
multiplied by ``dt`` again.

NumPy and the official MuJoCo bindings are imported lazily.  This keeps the
released evaluator importable when the opt-in feasibility dependencies are
not allocated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Integral
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple


ARM_DOF = 7


class TorqueSensitivityError(RuntimeError):
    """Fail-closed error for an invalid cloned transition or derivative."""


@dataclass(frozen=True)
class PostOscIntegrationSnapshot:
    """Immutable pre-physics authority captured after OSC wrote ``ctrl``."""

    integration_state: Any
    integration_state_sha256: str
    all_ctrl: Any
    arm_actuator_ids: Tuple[int, ...]
    arm_qpos_indices: Tuple[int, ...]
    arm_qvel_indices: Tuple[int, ...]
    arm_torque_lower_nm: Any
    arm_torque_upper_nm: Any
    nominal_arm_torque_nm: Any
    model_nq: int
    model_nv: int
    model_nu: int
    model_object_identity: int
    state_size: int
    state_specification: int
    timestep_seconds: float
    pre_step_contact_count: int


@dataclass(frozen=True)
class OneSubstepTransition:
    """Exact result of one cloned ``mj_step`` from a registered snapshot."""

    arm_torque_nm: Any
    next_qpos: Any
    next_qvel: Any
    next_arm_qpos: Any
    next_arm_qvel: Any
    next_integration_state_sha256: str
    next_time_seconds: float
    post_step_contact_count: int
    selected_contact_data: Any
    non_arm_ctrl_preserved: bool


@dataclass(frozen=True)
class PostOscTorqueSensitivity:
    """Validated local map from seven arm torques to ordered output velocity.

    The two legacy ``*_arm_*`` fields remain byte-compatible for the default
    arm-qvel output order.  They are ``None`` when an explicit, different
    output subspace is requested; callers must then consume the generic
    ``*_output_*`` fields together with ``output_qvel_indices``.
    """

    nominal_next_arm_qvel_rad_s: Any
    torque_to_next_arm_qvel_sensitivity: Any
    full_epsilon_sensitivity: Any
    half_epsilon_sensitivity: Any
    torque_epsilon_nm: Any
    maximum_epsilon_agreement_absolute_error: float
    maximum_epsilon_agreement_scaled_error: float
    agreement_atol: float
    agreement_rtol: float
    nominal_transition: OneSubstepTransition
    analytic_free_dynamics_diagnostic: Mapping[str, Any]
    finite_difference_column_stencils: Tuple[Mapping[str, Any], ...]
    output_qvel_indices: Tuple[int, ...] = ()
    nominal_next_output_qvel_rad_s: Optional[Any] = None
    torque_to_next_output_qvel_sensitivity: Optional[Any] = None


def _modules(
    mujoco_module: Optional[Any] = None, numpy_module: Optional[Any] = None
) -> Tuple[Any, Any]:
    if numpy_module is None:
        try:
            import numpy as numpy_module  # type: ignore[no-redef]
        except ImportError as error:  # pragma: no cover - allocation dependency
            raise RuntimeError("NumPy is required for torque sensitivity") from error
    if mujoco_module is None:
        try:
            import mujoco as mujoco_module  # type: ignore[no-redef]
        except ImportError as error:  # pragma: no cover - allocation dependency
            raise RuntimeError(
                "official MuJoCo is required for torque sensitivity"
            ) from error
    return mujoco_module, numpy_module


def _raw_model_data(model_or_sim: Any, live_data: Optional[Any]) -> Tuple[Any, Any]:
    if live_data is not None:
        model_candidate = model_or_sim
        data_candidate = live_data
    else:
        candidate = getattr(model_or_sim, "sim", model_or_sim)
        if not hasattr(candidate, "model") or not hasattr(candidate, "data"):
            raise TypeError(
                "model_or_sim must expose model/data when live_data is omitted"
            )
        model_candidate = candidate.model
        data_candidate = candidate.data
    return (
        getattr(model_candidate, "_model", model_candidate),
        getattr(data_candidate, "_data", data_candidate),
    )


def _raw_model(model_or_sim: Any) -> Any:
    candidate = getattr(model_or_sim, "sim", model_or_sim)
    model_candidate = getattr(candidate, "model", candidate)
    return getattr(model_candidate, "_model", model_candidate)


def _require_official_objects(
    model: Any, data: Optional[Any], mujoco: Any
) -> None:
    model_type = getattr(mujoco, "MjModel", None)
    data_type = getattr(mujoco, "MjData", None)
    if model_type is None or not isinstance(model, model_type):
        raise TypeError("target must expose an official MuJoCo MjModel")
    if data is not None and (data_type is None or not isinstance(data, data_type)):
        raise TypeError("target must expose an official MuJoCo MjData")


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError("%s must be finite and nonnegative" % label)
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("%s must be finite and nonnegative" % label)
    return result


def _validated_ids(
    values: Sequence[int], *, count: int, label: str
) -> Tuple[int, ...]:
    output = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("%s must contain integer indexes" % label)
        if value < 0 or value >= count:
            raise ValueError("%s contains an out-of-range index: %r" % (label, value))
        output.append(int(value))
    if len(output) != ARM_DOF:
        raise ValueError("%s must contain exactly seven indexes" % label)
    if len(set(output)) != ARM_DOF:
        raise ValueError("%s must not contain duplicate indexes" % label)
    return tuple(output)


def _validated_output_qvel_indices(
    values: Sequence[int], *, count: int
) -> Tuple[int, ...]:
    output = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("output_qvel_indices must contain integer indexes")
        if value < 0 or value >= count:
            raise ValueError(
                "output_qvel_indices contains an out-of-range index: %r" % value
            )
        output.append(int(value))
    if not output:
        raise ValueError("output_qvel_indices must be nonempty")
    if len(set(output)) != len(output):
        raise ValueError("output_qvel_indices must not contain duplicate indexes")
    return tuple(output)


def _readonly_copy(value: Any, np: Any) -> Any:
    output = np.asarray(value, dtype=np.float64).copy()
    output.setflags(write=False)
    return output


def _state_vector(model: Any, data: Any, specification: int, mujoco: Any, np: Any) -> Any:
    state = np.empty(int(mujoco.mj_stateSize(model, specification)), dtype=np.float64)
    mujoco.mj_getState(model, data, state, specification)
    if state.ndim != 1 or not np.all(np.isfinite(state)):
        raise TorqueSensitivityError("MuJoCo integration state is non-finite")
    return state


def _state_digest(state: Any) -> str:
    return hashlib.sha256(state.tobytes(order="C")).hexdigest()


def capture_post_osc_integration_state(
    model_or_sim: Any,
    *,
    arm_actuator_ids: Sequence[int],
    arm_qpos_indices: Sequence[int],
    arm_qvel_indices: Sequence[int],
    live_data: Optional[Any] = None,
    mujoco_module: Optional[Any] = None,
    numpy_module: Optional[Any] = None,
) -> PostOscIntegrationSnapshot:
    """Capture the exact callback state without altering the live ``MjData``.

    The seven actuator bounds are read from the compiled model.  Every arm
    actuator must have a finite, enabled control range and the already-written
    nominal OSC torque must lie inside it.
    """

    mujoco, np = _modules(mujoco_module, numpy_module)
    model, data = _raw_model_data(model_or_sim, live_data)
    _require_official_objects(model, data, mujoco)
    nq = int(model.nq)
    nv = int(model.nv)
    nu = int(model.nu)
    if nq <= 0 or nv <= 0 or nu < ARM_DOF:
        raise ValueError("compiled MuJoCo dimensions cannot contain a seven-DOF arm")
    actuator_ids = _validated_ids(
        arm_actuator_ids, count=nu, label="arm_actuator_ids"
    )
    qpos_indices = _validated_ids(
        arm_qpos_indices, count=nq, label="arm_qpos_indices"
    )
    qvel_indices = _validated_ids(
        arm_qvel_indices, count=nv, label="arm_qvel_indices"
    )
    specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    before = _state_vector(model, data, specification, mujoco, np)
    ctrl = np.asarray(data.ctrl, dtype=np.float64)
    if ctrl.shape != (nu,) or not np.all(np.isfinite(ctrl)):
        raise ValueError("live data.ctrl must be one finite value per actuator")

    ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    if ctrlrange.shape != (nu, 2) or not np.all(np.isfinite(ctrlrange)):
        raise ValueError("compiled actuator_ctrlrange must be finite with shape (nu, 2)")
    selected_range = ctrlrange[np.asarray(actuator_ids, dtype=np.int64)]
    lower = selected_range[:, 0]
    upper = selected_range[:, 1]
    if np.any(lower >= upper):
        raise ValueError("selected arm actuator control ranges are contradictory")
    if hasattr(model, "actuator_ctrllimited"):
        limited = np.asarray(model.actuator_ctrllimited)[
            np.asarray(actuator_ids, dtype=np.int64)
        ]
        if limited.shape != (ARM_DOF,) or not np.all(limited != 0):
            raise ValueError("all selected arm actuators must have enabled ctrl limits")
    nominal = ctrl[np.asarray(actuator_ids, dtype=np.int64)]
    if np.any(nominal < lower) or np.any(nominal > upper):
        raise ValueError("nominal OSC arm torque violates compiled actuator bounds")
    timestep = float(model.opt.timestep)
    if not math.isfinite(timestep) or timestep <= 0.0:
        raise ValueError("compiled MuJoCo timestep must be finite and positive")

    # Reading state and controls must itself be observational.  This guard also
    # catches a nonstandard binding whose state getter has side effects.
    after = _state_vector(model, data, specification, mujoco, np)
    if not np.array_equal(before, after):
        raise TorqueSensitivityError("capturing state mutated the live MuJoCo data")

    return PostOscIntegrationSnapshot(
        integration_state=_readonly_copy(before, np),
        integration_state_sha256=_state_digest(before),
        all_ctrl=_readonly_copy(ctrl, np),
        arm_actuator_ids=actuator_ids,
        arm_qpos_indices=qpos_indices,
        arm_qvel_indices=qvel_indices,
        arm_torque_lower_nm=_readonly_copy(lower, np),
        arm_torque_upper_nm=_readonly_copy(upper, np),
        nominal_arm_torque_nm=_readonly_copy(nominal, np),
        model_nq=nq,
        model_nv=nv,
        model_nu=nu,
        model_object_identity=id(model),
        state_size=int(before.size),
        state_specification=specification,
        timestep_seconds=timestep,
        pre_step_contact_count=int(getattr(data, "ncon", 0)),
    )


def _validate_snapshot(model: Any, snapshot: PostOscIntegrationSnapshot, mujoco: Any) -> None:
    if not isinstance(snapshot, PostOscIntegrationSnapshot):
        raise TypeError("snapshot must be a PostOscIntegrationSnapshot")
    dimensions = (int(model.nq), int(model.nv), int(model.nu))
    registered = (snapshot.model_nq, snapshot.model_nv, snapshot.model_nu)
    if dimensions != registered:
        raise ValueError("snapshot and target MuJoCo model dimensions differ")
    if id(model) != snapshot.model_object_identity:
        raise ValueError("snapshot belongs to a different MuJoCo model object")
    expected_specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    expected_size = int(mujoco.mj_stateSize(model, expected_specification))
    if (
        snapshot.state_specification != expected_specification
        or snapshot.state_size != expected_size
        or snapshot.integration_state.shape != (expected_size,)
        or _state_digest(snapshot.integration_state)
        != snapshot.integration_state_sha256
    ):
        raise ValueError("snapshot integration-state contract is invalid")


def _candidate_torque(
    snapshot: PostOscIntegrationSnapshot, value: Sequence[float], np: Any
) -> Any:
    torque = np.asarray(value, dtype=np.float64)
    if torque.shape != (ARM_DOF,) or not np.all(np.isfinite(torque)):
        raise ValueError("arm_torque must be a finite seven-vector")
    if np.any(torque < snapshot.arm_torque_lower_nm) or np.any(
        torque > snapshot.arm_torque_upper_nm
    ):
        raise ValueError("arm_torque violates compiled actuator bounds")
    return torque


def _prepared_clone(model: Any, snapshot: PostOscIntegrationSnapshot, mujoco: Any, np: Any) -> Any:
    """Create current derived fields while preserving the exact state vector."""

    clone = mujoco.MjData(model)
    mujoco.mj_setState(
        model,
        clone,
        np.asarray(snapshot.integration_state, dtype=np.float64).copy(),
        snapshot.state_specification,
    )
    # A fresh MjData has no current derived dynamics.  Forward it once, then
    # restore the complete integration vector because mj_forward may rewrite
    # qacc_warmstart.  The qpos/qvel-derived arrays remain current.
    mujoco.mj_forward(model, clone)
    mujoco.mj_setState(
        model,
        clone,
        np.asarray(snapshot.integration_state, dtype=np.float64).copy(),
        snapshot.state_specification,
    )
    restored = _state_vector(
        model, clone, snapshot.state_specification, mujoco, np
    )
    if not np.array_equal(restored, snapshot.integration_state):
        raise TorqueSensitivityError("fresh clone did not restore the exact snapshot")
    return clone


def clone_one_substep_transition(
    model_or_sim: Any,
    *,
    snapshot: PostOscIntegrationSnapshot,
    arm_torque: Sequence[float],
    selected_contact_hook: Optional[Callable[[Any, Any], Any]] = None,
    mujoco_module: Optional[Any] = None,
    numpy_module: Optional[Any] = None,
) -> OneSubstepTransition:
    """Step one fresh clone with only the seven arm controls replaced.

    ``selected_contact_hook(model, clone)`` is invoked after the one substep
    and can extract the same selected-contact data used by a simulator
    postcheck.  It receives only the clone, never the live data.
    """

    mujoco, np = _modules(mujoco_module, numpy_module)
    model = _raw_model(model_or_sim)
    _require_official_objects(model, None, mujoco)
    _validate_snapshot(model, snapshot, mujoco)
    torque = _candidate_torque(snapshot, arm_torque, np)
    if selected_contact_hook is not None and not callable(selected_contact_hook):
        raise TypeError("selected_contact_hook must be callable or None")

    live_model = live_data = live_before = None
    candidate = getattr(model_or_sim, "sim", model_or_sim)
    if hasattr(candidate, "model") and hasattr(candidate, "data"):
        live_model, live_data = _raw_model_data(candidate, None)
        _require_official_objects(live_model, live_data, mujoco)
        if live_model is not model:
            raise ValueError("live simulator and snapshot model objects differ")
        live_before = _state_vector(
            live_model, live_data, snapshot.state_specification, mujoco, np
        )
        if not np.array_equal(live_before, snapshot.integration_state):
            raise ValueError("live simulator no longer matches the captured snapshot")

    clone = _prepared_clone(model, snapshot, mujoco, np)
    actuator_ids = np.asarray(snapshot.arm_actuator_ids, dtype=np.int64)
    all_ids = np.arange(snapshot.model_nu, dtype=np.int64)
    non_arm_ids = all_ids[~np.isin(all_ids, actuator_ids)]
    if not np.array_equal(clone.ctrl, snapshot.all_ctrl):
        raise TorqueSensitivityError("clone controls differ before arm perturbation")
    clone.ctrl[actuator_ids] = torque
    if not np.array_equal(clone.ctrl[non_arm_ids], snapshot.all_ctrl[non_arm_ids]):
        raise TorqueSensitivityError("arm perturbation changed a non-arm control")

    start_time = float(clone.time)
    mujoco.mj_step(model, clone)
    expected_time = start_time + snapshot.timestep_seconds
    if not math.isclose(float(clone.time), expected_time, rel_tol=0.0, abs_tol=1e-12):
        raise TorqueSensitivityError("clone did not advance by exactly one timestep")
    qpos = np.asarray(clone.qpos, dtype=np.float64).copy()
    qvel = np.asarray(clone.qvel, dtype=np.float64).copy()
    next_integration_state = _state_vector(
        model, clone, snapshot.state_specification, mujoco, np
    )
    if (
        qpos.shape != (snapshot.model_nq,)
        or qvel.shape != (snapshot.model_nv,)
        or not np.all(np.isfinite(qpos))
        or not np.all(np.isfinite(qvel))
    ):
        raise TorqueSensitivityError("one-substep clone produced a non-finite state")
    if not np.array_equal(clone.ctrl[non_arm_ids], snapshot.all_ctrl[non_arm_ids]):
        raise TorqueSensitivityError("one-substep dynamics changed a non-arm control")

    selected_contact_data = (
        selected_contact_hook(model, clone)
        if selected_contact_hook is not None
        else None
    )
    if live_before is not None:
        live_after = _state_vector(
            live_model, live_data, snapshot.state_specification, mujoco, np
        )
        if not np.array_equal(live_before, live_after):
            raise TorqueSensitivityError(
                "cloned candidate transition mutated the live MuJoCo data"
            )
    return OneSubstepTransition(
        arm_torque_nm=_readonly_copy(torque, np),
        next_qpos=_readonly_copy(qpos, np),
        next_qvel=_readonly_copy(qvel, np),
        next_arm_qpos=_readonly_copy(
            qpos[np.asarray(snapshot.arm_qpos_indices, dtype=np.int64)], np
        ),
        next_arm_qvel=_readonly_copy(
            qvel[np.asarray(snapshot.arm_qvel_indices, dtype=np.int64)], np
        ),
        next_integration_state_sha256=_state_digest(next_integration_state),
        next_time_seconds=float(clone.time),
        post_step_contact_count=int(getattr(clone, "ncon", 0)),
        selected_contact_data=selected_contact_data,
        non_arm_ctrl_preserved=True,
    )


def _epsilon_vector(value: Any, np: Any) -> Any:
    if isinstance(value, bool):
        raise ValueError("torque_epsilon_nm must be finite and positive")
    epsilon = np.asarray(value, dtype=np.float64)
    if epsilon.shape == ():
        epsilon = np.full(ARM_DOF, float(epsilon), dtype=np.float64)
    if epsilon.shape != (ARM_DOF,) or not np.all(np.isfinite(epsilon)) or np.any(
        epsilon <= 0.0
    ):
        raise ValueError("torque_epsilon_nm must be a finite positive scalar or seven-vector")
    return epsilon


def _sampled_delta(
    *,
    nominal: float,
    requested_delta: float,
    lower: float,
    upper: float,
    column: int,
    resolution: str,
) -> float:
    """Return the exact representable delta of one bound-valid sample.

    This intentionally does not clip.  A requested perturbation that rounds
    back to the nominal torque has no numerical resolution and is rejected.
    """

    candidate = nominal + requested_delta
    actual_delta = candidate - nominal
    if (
        not math.isfinite(candidate)
        or not math.isfinite(actual_delta)
        or actual_delta == 0.0
        or math.copysign(1.0, actual_delta)
        != math.copysign(1.0, requested_delta)
    ):
        raise TorqueSensitivityError(
            "column %d has no positive perturbation resolution at %s resolution"
            % (column, resolution)
        )
    if candidate < lower or candidate > upper:
        raise TorqueSensitivityError(
            "column %d %s perturbation violates actuator bounds without clipping"
            % (column, resolution)
        )
    return float(actual_delta)


def _finite_difference_plans(
    snapshot: PostOscIntegrationSnapshot,
    epsilon: Any,
    np: Any,
) -> Tuple[Mapping[str, Any], ...]:
    """Choose one fixed feasible stencil per column at two resolutions."""

    nominal = np.asarray(snapshot.nominal_arm_torque_nm, dtype=np.float64)
    lower = np.asarray(snapshot.arm_torque_lower_nm, dtype=np.float64)
    upper = np.asarray(snapshot.arm_torque_upper_nm, dtype=np.float64)
    plans = []
    for column in range(ARM_DOF):
        base = float(nominal[column])
        requested = float(epsilon[column])
        low = float(lower[column])
        high = float(upper[column])
        negative_room = float(base - low)
        positive_room = float(high - base)
        if negative_room < 0.0 or positive_room < 0.0:
            raise ValueError("nominal torque lies outside registered actuator bounds")

        # A centered stencil is preferred only when the requested full
        # resolution fits on both sides.  Otherwise use one feasible side at
        # both resolutions, which makes the convergence comparison coherent.
        if requested <= negative_room and requested <= positive_room:
            stencil = "centered"
            signed_full = (-requested, requested)
        else:
            forward_step = min(requested, positive_room)
            backward_step = min(requested, negative_room)
            if forward_step <= 0.0 and backward_step <= 0.0:
                raise TorqueSensitivityError(
                    "column %d has no positive perturbation resolution" % column
                )
            if forward_step >= backward_step:
                stencil = "forward"
                signed_full = (0.0, forward_step)
            else:
                stencil = "backward"
                signed_full = (-backward_step, 0.0)

        resolutions: Dict[str, Mapping[str, Any]] = {}
        for name, scale in (("full", 1.0), ("half", 0.5)):
            exact_deltas = []
            for requested_delta in signed_full:
                scaled_delta = float(requested_delta * scale)
                if scaled_delta == 0.0:
                    exact_deltas.append(0.0)
                    continue
                exact_deltas.append(
                    _sampled_delta(
                        nominal=base,
                        requested_delta=scaled_delta,
                        lower=low,
                        upper=high,
                        column=column,
                        resolution=name,
                    )
                )
            denominator = float(exact_deltas[1] - exact_deltas[0])
            if not math.isfinite(denominator) or denominator <= 0.0:
                raise TorqueSensitivityError(
                    "column %d has no positive perturbation resolution at %s resolution"
                    % (column, name)
                )
            resolutions[name] = {
                "stencil": stencil,
                "sample_deltas_nm": tuple(float(value) for value in exact_deltas),
                "denominator_nm": denominator,
            }
        plans.append(
            {
                "column_index": int(column),
                "nominal_torque_nm": base,
                "lower_bound_nm": low,
                "upper_bound_nm": high,
                "requested_full_epsilon_nm": requested,
                "available_negative_delta_nm": negative_room,
                "available_positive_delta_nm": positive_room,
                "bound_adapted": bool(
                    stencil != "centered"
                    or max(abs(value) for value in signed_full) < requested
                ),
                "full_resolution": resolutions["full"],
                "half_resolution": resolutions["half"],
                "difference_formula": (
                    "(v_next(delta_1)-v_next(delta_0))/(delta_1-delta_0)"
                ),
            }
        )
    return tuple(plans)


def _planned_sensitivity(
    model: Any,
    snapshot: PostOscIntegrationSnapshot,
    plans: Sequence[Mapping[str, Any]],
    resolution: str,
    nominal_transition: OneSubstepTransition,
    output_qvel_indices: Sequence[int],
    mujoco: Any,
    np: Any,
) -> Any:
    output_ids = np.asarray(output_qvel_indices, dtype=np.int64)
    sensitivity = np.empty((output_ids.size, ARM_DOF), dtype=np.float64)
    nominal = np.asarray(snapshot.nominal_arm_torque_nm, dtype=np.float64)
    resolution_key = "%s_resolution" % resolution
    for column, plan in enumerate(plans):
        if int(plan["column_index"]) != column:
            raise TorqueSensitivityError("finite-difference column plan is unordered")
        resolution_plan = plan[resolution_key]
        deltas = tuple(float(value) for value in resolution_plan["sample_deltas_nm"])
        if len(deltas) != 2:
            raise TorqueSensitivityError("finite-difference stencil must have two samples")
        transitions = []
        for delta in deltas:
            if delta == 0.0:
                transitions.append(nominal_transition)
                continue
            candidate = nominal.copy()
            candidate[column] += delta
            transitions.append(
                clone_one_substep_transition(
                    model,
                    snapshot=snapshot,
                    arm_torque=candidate,
                    mujoco_module=mujoco,
                    numpy_module=np,
                )
            )
        denominator = float(resolution_plan["denominator_nm"])
        sensitivity[:, column] = (
            transitions[1].next_qvel[output_ids]
            - transitions[0].next_qvel[output_ids]
        ) / denominator
    if not np.all(np.isfinite(sensitivity)):
        raise TorqueSensitivityError(
            "%s-resolution finite-difference sensitivity is non-finite" % resolution
        )
    return sensitivity


def _analytic_diagnostic(
    model: Any,
    snapshot: PostOscIntegrationSnapshot,
    nominal_transition: OneSubstepTransition,
    measured_sensitivity: Any,
    output_qvel_indices: Sequence[int],
    mujoco: Any,
    np: Any,
) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "formula": "dt * E * M^-1 * B",
        "acceptance_authority": False,
        "available": False,
        "reason": None,
    }
    if snapshot.pre_step_contact_count != 0 or nominal_transition.post_step_contact_count != 0:
        base["reason"] = "contact_present"
        return base
    try:
        clone = _prepared_clone(model, snapshot, mujoco, np)
        mass = np.empty((snapshot.model_nv, snapshot.model_nv), dtype=np.float64)
        mujoco.mj_fullM(model, mass, clone.qM)
        moments = np.asarray(clone.actuator_moment, dtype=np.float64)
        if moments.shape != (snapshot.model_nu, snapshot.model_nv):
            raise ValueError("actuator_moment has an unexpected shape")
        actuator_ids = np.asarray(snapshot.arm_actuator_ids, dtype=np.int64)
        qvel_ids = np.asarray(output_qvel_indices, dtype=np.int64)
        force_map = moments[actuator_ids, :].T
        analytic_full = snapshot.timestep_seconds * np.linalg.solve(mass, force_map)
        analytic = analytic_full[qvel_ids, :]
        if not np.all(np.isfinite(analytic)):
            raise ValueError("analytic diagnostic is non-finite")
        difference = measured_sensitivity - analytic
        base.update(
            {
                "available": True,
                "reason": None,
                "matrix": analytic.tolist(),
                "maximum_absolute_error": float(np.max(np.abs(difference))),
                "frobenius_error": float(np.linalg.norm(difference)),
                "note": (
                    "Diagnostic only: constraints, integrator choice, damping, "
                    "and nonlinear dynamics can make the exact cloned derivative differ."
                ),
            }
        )
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        base["reason"] = "unavailable: %s" % error
    return base


def estimate_post_osc_torque_sensitivity(
    model_or_sim: Any,
    *,
    snapshot: PostOscIntegrationSnapshot,
    torque_epsilon_nm: Any,
    agreement_atol: float = 1e-7,
    agreement_rtol: float = 1e-3,
    output_qvel_indices: Optional[Sequence[int]] = None,
    mujoco_module: Optional[Any] = None,
    numpy_module: Optional[Any] = None,
) -> PostOscTorqueSensitivity:
    """Return an exact local map from seven torques to ``M`` velocities.

    With no explicit output indexes, ``M=7`` and the historical arm-qvel
    result is unchanged.  Otherwise the output rows follow the caller's exact
    ordered, unique MuJoCo qvel indexes and the sensitivity has shape
    ``(M, 7)``.
    """

    mujoco, np = _modules(mujoco_module, numpy_module)
    model = _raw_model(model_or_sim)
    _require_official_objects(model, None, mujoco)
    _validate_snapshot(model, snapshot, mujoco)
    selected_output_indices = _validated_output_qvel_indices(
        (
            snapshot.arm_qvel_indices
            if output_qvel_indices is None
            else output_qvel_indices
        ),
        count=snapshot.model_nv,
    )
    epsilon = _epsilon_vector(torque_epsilon_nm, np)
    atol = _finite_nonnegative(agreement_atol, "agreement_atol")
    rtol = _finite_nonnegative(agreement_rtol, "agreement_rtol")
    if atol == 0.0 and rtol == 0.0:
        raise ValueError("at least one agreement tolerance must be positive")
    nominal = np.asarray(snapshot.nominal_arm_torque_nm, dtype=np.float64)
    plans = _finite_difference_plans(snapshot, epsilon, np)

    # Guard the live authority when a sim rather than a bare model is supplied.
    live_model = live_data = live_before = None
    candidate = getattr(model_or_sim, "sim", model_or_sim)
    if hasattr(candidate, "model") and hasattr(candidate, "data"):
        live_model, live_data = _raw_model_data(candidate, None)
        live_before = _state_vector(
            live_model, live_data, snapshot.state_specification, mujoco, np
        )
        if not np.array_equal(live_before, snapshot.integration_state):
            raise ValueError("live simulator no longer matches the captured snapshot")

    nominal_transition = clone_one_substep_transition(
        model,
        snapshot=snapshot,
        arm_torque=nominal,
        mujoco_module=mujoco,
        numpy_module=np,
    )
    full = _planned_sensitivity(
        model,
        snapshot,
        plans,
        "full",
        nominal_transition,
        selected_output_indices,
        mujoco,
        np,
    )
    half = _planned_sensitivity(
        model,
        snapshot,
        plans,
        "half",
        nominal_transition,
        selected_output_indices,
        mujoco,
        np,
    )
    difference = np.abs(full - half)
    scale = atol + rtol * np.maximum(np.abs(full), np.abs(half))
    passed = bool(np.all(difference <= scale))
    scaled_error = float(np.max(difference / scale))
    if not passed:
        raise TorqueSensitivityError(
            "finite-difference sensitivities at epsilon and half-epsilon disagree "
            "(max scaled error %.9g)" % scaled_error
        )

    if live_before is not None:
        live_after = _state_vector(
            live_model, live_data, snapshot.state_specification, mujoco, np
        )
        if not np.array_equal(live_before, live_after):
            raise TorqueSensitivityError(
                "cloned sensitivity calculation mutated the live MuJoCo data"
            )

    diagnostic = _analytic_diagnostic(
        model,
        snapshot,
        nominal_transition,
        half,
        selected_output_indices,
        mujoco,
        np,
    )
    output_ids = np.asarray(selected_output_indices, dtype=np.int64)
    nominal_output = _readonly_copy(
        nominal_transition.next_qvel[output_ids], np
    )
    sensitivity_output = _readonly_copy(half, np)
    default_arm_output = selected_output_indices == snapshot.arm_qvel_indices
    return PostOscTorqueSensitivity(
        nominal_next_arm_qvel_rad_s=(
            _readonly_copy(nominal_transition.next_arm_qvel, np)
            if default_arm_output
            else None
        ),
        torque_to_next_arm_qvel_sensitivity=(
            _readonly_copy(half, np) if default_arm_output else None
        ),
        full_epsilon_sensitivity=_readonly_copy(full, np),
        half_epsilon_sensitivity=_readonly_copy(half, np),
        torque_epsilon_nm=_readonly_copy(epsilon, np),
        maximum_epsilon_agreement_absolute_error=float(np.max(difference)),
        maximum_epsilon_agreement_scaled_error=scaled_error,
        agreement_atol=atol,
        agreement_rtol=rtol,
        nominal_transition=nominal_transition,
        analytic_free_dynamics_diagnostic=diagnostic,
        finite_difference_column_stencils=plans,
        output_qvel_indices=selected_output_indices,
        nominal_next_output_qvel_rad_s=nominal_output,
        torque_to_next_output_qvel_sensitivity=sensitivity_output,
    )


__all__ = [
    "OneSubstepTransition",
    "PostOscIntegrationSnapshot",
    "PostOscTorqueSensitivity",
    "TorqueSensitivityError",
    "capture_post_osc_integration_state",
    "clone_one_substep_transition",
    "estimate_post_osc_torque_sensitivity",
]
