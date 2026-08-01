"""Exact OSC-settled-state transfer into the opt-in joint-velocity arm."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from collections import deque
from collections.abc import Mapping
from typing import Any, Dict, Sequence, Tuple


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("NumPy is required for controller state transfer") from error
    return np


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _array_sha256(value: Any) -> str:
    np = _numpy()
    array = np.ascontiguousarray(value)
    header = _canonical({"dtype": array.dtype.str, "shape": list(array.shape)})
    digest = hashlib.sha256()
    digest.update(b"vlsa-poisson-array-v1\0")
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _model_names(model: Any, kind: str, count: int) -> Tuple[Any, ...]:
    method = getattr(model, "%s_id2name" % kind, None)
    if method is None:
        names = getattr(model, "%s_names" % kind, None)
        if names is None or len(names) != count:
            try:
                import mujoco

                raw_model = getattr(model, "_model", model)
                object_type = {
                    "body": mujoco.mjtObj.mjOBJ_BODY,
                    "geom": mujoco.mjtObj.mjOBJ_GEOM,
                    "joint": mujoco.mjtObj.mjOBJ_JOINT,
                    "actuator": mujoco.mjtObj.mjOBJ_ACTUATOR,
                }[kind]
                return tuple(
                    mujoco.mj_id2name(raw_model, object_type, index)
                    for index in range(count)
                )
            except (ImportError, KeyError, TypeError, ValueError):
                raise ValueError(
                    "model does not expose authoritative %s names" % kind
                )
        return tuple(None if value is None else str(value) for value in names)
    return tuple(
        None if method(index) is None else str(method(index))
        for index in range(count)
    )


def model_topology_contract(model_or_sim: Any) -> Dict[str, Any]:
    """Hash identities that must not change when switching controllers."""

    np = _numpy()
    candidate = getattr(model_or_sim, "model", model_or_sim)
    model = getattr(candidate, "_model", candidate)
    record = {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "njnt": int(model.njnt),
        "body_names": _model_names(candidate, "body", int(model.nbody)),
        "geom_names": _model_names(candidate, "geom", int(model.ngeom)),
        "joint_names": _model_names(candidate, "joint", int(model.njnt)),
        "body_parentid": np.asarray(model.body_parentid, dtype=np.int64).tolist(),
        "geom_bodyid": np.asarray(model.geom_bodyid, dtype=np.int64).tolist(),
        "geom_type": np.asarray(model.geom_type, dtype=np.int64).tolist(),
        "geom_size_sha256": _array_sha256(
            np.asarray(model.geom_size, dtype=np.float64)
        ),
    }
    return {
        "schema_version": "vlsa_poisson_model_topology.v1",
        "sha256": hashlib.sha256(_canonical(record)).hexdigest(),
        "record": record,
    }


def model_physics_contract(model_or_sim: Any) -> Dict[str, Any]:
    """Hash every readable compiled MuJoCo model array and scalar.

    Controller switching is allowed to change software controller state, but
    not the compiled physical model.  Hashing the complete exposed model
    payload is intentionally stricter than the topology-only compatibility
    record used by older structural tests.
    """

    np = _numpy()
    candidate = getattr(model_or_sim, "model", model_or_sim)
    model = getattr(candidate, "_model", candidate)
    records = []
    for name in sorted(item for item in dir(model) if not item.startswith("_")):
        try:
            value = getattr(model, name)
        except Exception:
            continue
        if isinstance(value, np.ndarray):
            records.append(
                {
                    "name": name,
                    "kind": "array",
                    "dtype": value.dtype.str,
                    "shape": list(value.shape),
                    "sha256": _array_sha256(value),
                }
            )
        elif isinstance(value, (bool, int, float, str)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("compiled MuJoCo model contains a non-finite scalar")
            records.append({"name": name, "kind": "scalar", "value": value})
        elif isinstance(value, bytes):
            records.append(
                {
                    "name": name,
                    "kind": "bytes",
                    "bytes": len(value),
                    "sha256": hashlib.sha256(value).hexdigest(),
                }
            )
    if not any(record["kind"] == "array" for record in records):
        raise ValueError("model does not expose compiled MuJoCo arrays")
    option_records = []
    option = getattr(model, "opt", None)
    if option is not None:
        for name in sorted(item for item in dir(option) if not item.startswith("_")):
            try:
                value = getattr(option, name)
            except Exception:
                continue
            if isinstance(value, np.ndarray):
                option_records.append(
                    {
                        "name": name,
                        "kind": "array",
                        "dtype": value.dtype.str,
                        "shape": list(value.shape),
                        "sha256": _array_sha256(value),
                    }
                )
            elif isinstance(value, (bool, int, float, str)):
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(
                        "compiled MuJoCo option contains a non-finite scalar"
                    )
                option_records.append(
                    {"name": name, "kind": "scalar", "value": value}
                )

    # The official MJB is the strongest identity: it serializes the complete
    # compiled model, including nested mjOption fields such as timestep,
    # gravity, integrator, and solver.  The explicit field ledgers remain in
    # the payload as an interpretable fallback for structural fakes.
    compiled_mjb_sha256 = None
    compiled_mjb_bytes = None
    try:
        import mujoco

        if isinstance(model, mujoco.MjModel):
            size = int(mujoco.mj_sizeModel(model))
            if size <= 0:
                raise ValueError("official MuJoCo returned an invalid MJB size")
            buffer = np.empty(size, dtype=np.uint8)
            mujoco.mj_saveModel(model, buffer=buffer)
            compiled_mjb_bytes = int(buffer.nbytes)
            compiled_mjb_sha256 = hashlib.sha256(buffer.tobytes()).hexdigest()
    except ImportError:
        pass
    nq = int(model.nq)
    nv = int(model.nv)
    na = int(model.na)
    if nq <= 0 or nv <= 0 or na < 0:
        raise ValueError("compiled MuJoCo state dimensions are invalid")
    integration_state_size = None
    try:
        import mujoco

        if isinstance(model, mujoco.MjModel):
            integration_state_size = int(
                mujoco.mj_stateSize(
                    model, int(mujoco.mjtState.mjSTATE_INTEGRATION)
                )
            )
    except ImportError:
        pass
    if integration_state_size is None:
        # Structural unit-test doubles do not expose the official MuJoCo
        # state API.  Their integration state has only the common prefix.
        integration_state_size = 1 + nq + nv + na
    if integration_state_size < 1 + nq + nv + na:
        raise ValueError("official MuJoCo integration-state layout is truncated")
    flattened_state_size = 1 + nq + nv + na
    payload = {
        "schema_version": "vlsa_poisson_physical_model.v3",
        "compiled_mjb_sha256": compiled_mjb_sha256,
        "compiled_mjb_bytes": compiled_mjb_bytes,
        "nq": nq,
        "nv": nv,
        "na": na,
        "mjstate_integration_size": integration_state_size,
        "robosuite_flattened_state_size": flattened_state_size,
        "robosuite_flattened_state_layout": "time_qpos_qvel_act_no_udd_tail",
        "fields": records,
        "options": option_records,
    }
    return {
        "schema_version": payload["schema_version"],
        "sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
        "field_count": len(records),
        "option_field_count": len(option_records),
        "compiled_mjb_sha256": compiled_mjb_sha256,
        "compiled_mjb_bytes": compiled_mjb_bytes,
        "nq": nq,
        "nv": nv,
        "na": na,
        "mjstate_integration_size": integration_state_size,
        "robosuite_flattened_state_size": flattened_state_size,
        "robosuite_flattened_state_layout": "time_qpos_qvel_act_no_udd_tail",
    }


def _official_integration_state(model_or_sim: Any) -> Tuple[Any, Any, Any]:
    """Return raw model/data and an official complete integration state."""

    np = _numpy()
    try:
        import mujoco
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("official MuJoCo is required for complete state pairing") from error
    candidate = getattr(model_or_sim, "sim", model_or_sim)
    model_candidate = getattr(candidate, "model", None)
    data_candidate = getattr(candidate, "data", None)
    model = getattr(model_candidate, "_model", model_candidate)
    data = getattr(data_candidate, "_data", data_candidate)
    if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
        raise TypeError("environment does not expose official MuJoCo model/data")
    specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(int(mujoco.mj_stateSize(model, specification)), dtype=np.float64)
    mujoco.mj_getState(model, data, state, specification)
    if not np.all(np.isfinite(state)):
        raise ValueError("official MuJoCo integration state is non-finite")
    return model, data, state


def joint_velocity_controller_contract(env: Any) -> Dict[str, Any]:
    """Validate the exact 100 Hz, seven-arm-plus-gripper execution interface."""

    np = _numpy()
    if len(env.robots) != 1:
        raise ValueError("joint-velocity pilot requires exactly one robot")
    robot = env.robots[0]
    controller = robot.controller
    if str(controller.name) != "JOINT_VELOCITY":
        raise ValueError("target controller must be JOINT_VELOCITY")
    if int(controller.control_dim) != 7 or int(env.env.action_dim) != 8:
        raise ValueError("joint-velocity action must be seven arm values plus gripper")
    if int(env.env.control_freq) != 100:
        raise ValueError("joint-velocity environment must run at exactly 100 Hz")
    if not math.isclose(float(env.env.control_timestep), 0.01, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("joint-velocity control timestep must be 0.01 seconds")
    if not math.isclose(float(env.env.model_timestep), 0.002, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("MuJoCo timestep must be 0.002 seconds")
    output_min = np.asarray(controller.output_min, dtype=np.float64)
    output_max = np.asarray(controller.output_max, dtype=np.float64)
    input_min = np.asarray(controller.input_min, dtype=np.float64)
    input_max = np.asarray(controller.input_max, dtype=np.float64)
    if input_min.shape not in ((), (7,)) or input_max.shape not in ((), (7,)):
        raise ValueError("joint-velocity controller input bounds have wrong shape")
    if not (
        np.array_equal(np.broadcast_to(input_min, (7,)), np.full(7, -1.0))
        and np.array_equal(np.broadcast_to(input_max, (7,)), np.full(7, 1.0))
    ):
        raise ValueError("joint-velocity controller input bounds must be exactly +/-1")
    if output_min.shape != (7,) or output_max.shape != (7,):
        raise ValueError("joint-velocity controller output bounds have wrong shape")
    if not (
        np.array_equal(output_min, np.full(7, -0.5))
        and np.array_equal(output_max, np.full(7, 0.5))
    ):
        raise ValueError("normalized joint action must map exactly to +/-0.5 rad/s")
    if getattr(controller, "interpolator", None) is not None:
        raise ValueError("the registered joint-velocity controller has no interpolator")
    controller_class = type(controller)
    module_name = str(controller_class.__module__)
    qualified_name = str(controller_class.__qualname__)
    try:
        module_path = inspect.getsourcefile(controller_class)
        if not module_path:
            raise ValueError("controller class has no source file")
        with open(module_path, "rb") as stream:
            module_sha256 = hashlib.sha256(stream.read()).hexdigest()
    except (OSError, TypeError) as error:
        raise ValueError("controller implementation source is unavailable") from error
    config_path = os.path.join(
        os.path.dirname(str(module_path)), "config", "joint_velocity.json"
    )
    try:
        with open(config_path, "rb") as stream:
            config_sha256 = hashlib.sha256(stream.read()).hexdigest()
    except OSError:
        # Structural test doubles intentionally have no installed Robosuite
        # configuration file.  Production Stage-13 validation rejects this
        # null value against its H100-frozen protocol authority.
        config_sha256 = None
    panda_xml_path = os.path.join(
        os.path.dirname(os.path.dirname(str(module_path))),
        "models",
        "assets",
        "robots",
        "panda",
        "robot.xml",
    )
    try:
        with open(panda_xml_path, "rb") as stream:
            panda_xml_sha256 = hashlib.sha256(stream.read()).hexdigest()
    except OSError:
        panda_xml_sha256 = None

    gains = {}
    for gain_name in ("kp", "ki", "kd"):
        gain = np.asarray(getattr(controller, gain_name, None), dtype=np.float64)
        if gain.shape != (7,) or not np.all(np.isfinite(gain)):
            raise ValueError(
                "joint-velocity controller %s is missing or invalid" % gain_name
            )
        gains[gain_name] = gain
    if not (
        np.array_equal(gains["ki"], gains["kp"] * 0.005)
        and np.array_equal(gains["kd"], gains["kp"] * 0.001)
    ):
        raise ValueError("joint-velocity controller PID gain relationship differs")
    if int(getattr(controller, "control_freq", -1)) != 100:
        raise ValueError("joint-velocity controller policy frequency must be 100 Hz")
    velocity_limits_value = getattr(controller, "velocity_limits", None)
    if velocity_limits_value is not None:
        velocity_limits = np.asarray(velocity_limits_value, dtype=np.float64)
        if not np.all(np.isfinite(velocity_limits)):
            raise ValueError("joint-velocity controller velocity_limits are invalid")
        velocity_limits_record = velocity_limits.tolist()
    else:
        velocity_limits_record = None
    # Robosuite's production Controller stores the constructor's
    # ``joint_indexes`` mapping as three separate arrays.  Keep the artifact
    # schema normalized to the original mapping while reading the real runtime
    # attributes directly.
    joint_index_record = {}
    for field, attribute in (
        ("joints", "joint_index"),
        ("qpos", "qpos_index"),
        ("qvel", "qvel_index"),
    ):
        indexes = np.asarray(getattr(controller, attribute, None), dtype=np.int64)
        if indexes.shape != (7,):
            raise ValueError(
                "joint-velocity controller %s lacks seven indexes" % attribute
            )
        joint_index_record[field] = indexes.tolist()
    arm_qpos_indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
    arm_qvel_indexes = np.asarray(robot._ref_joint_vel_indexes, dtype=np.int64)
    joint_ids = np.asarray(
        getattr(robot, "_ref_joint_indexes", None), dtype=np.int64
    )
    if joint_ids.shape != (7,):
        raise ValueError("robot does not expose seven ordered arm joint IDs")
    if not (
        np.array_equal(np.asarray(joint_index_record["joints"]), joint_ids)
        and np.array_equal(
            np.asarray(joint_index_record["qpos"]), arm_qpos_indexes
        )
        and np.array_equal(np.asarray(joint_index_record["qvel"]), arm_qvel_indexes)
    ):
        raise ValueError("controller and robot arm joint index order differ")
    actuator_indexes = np.asarray(
        getattr(robot, "_ref_joint_actuator_indexes", None), dtype=np.int64
    )
    if actuator_indexes.shape != (7,):
        raise ValueError("robot does not expose seven ordered arm actuator indexes")
    actuator_limits = getattr(controller, "actuator_limits", None)
    if not isinstance(actuator_limits, (tuple, list)) or len(actuator_limits) != 2:
        raise ValueError("joint-velocity controller actuator_limits are missing")
    actuator_min = np.asarray(actuator_limits[0], dtype=np.float64)
    actuator_max = np.asarray(actuator_limits[1], dtype=np.float64)
    if (
        actuator_min.shape != (7,)
        or actuator_max.shape != (7,)
        or not np.all(np.isfinite(actuator_min))
        or not np.all(np.isfinite(actuator_max))
        or np.any(actuator_min >= actuator_max)
    ):
        raise ValueError("joint-velocity controller actuator ranges are invalid")
    model = env.sim.model
    compiled_ctrlrange = np.asarray(
        model.actuator_ctrlrange[actuator_indexes], dtype=np.float64
    )
    if not np.array_equal(
        compiled_ctrlrange,
        np.column_stack((actuator_min, actuator_max)),
    ):
        raise ValueError("controller actuator ranges differ from compiled MuJoCo ranges")
    actuator_names = _model_names(
        model, "actuator", int(getattr(model, "nu", len(actuator_indexes)))
    )
    ordered_actuator_names = [actuator_names[index] for index in actuator_indexes]
    joint_names = _model_names(model, "joint", int(model.njnt))
    ordered_joint_names = [joint_names[index] for index in joint_ids]
    return {
        "controller_class_module": module_name,
        "controller_class_qualname": qualified_name,
        "controller_implementation_file_sha256": module_sha256,
        "controller_configuration_file_sha256": config_sha256,
        "panda_robot_xml_file_sha256": panda_xml_sha256,
        "controller_name": str(controller.name),
        "environment_action_dim": int(env.env.action_dim),
        "arm_control_dim": int(controller.control_dim),
        "control_frequency_hz": int(env.env.control_freq),
        "control_timestep_s": float(env.env.control_timestep),
        "physics_timestep_s": float(env.env.model_timestep),
        "physics_substeps_per_control": int(
            env.env.control_timestep / env.env.model_timestep
        ),
        "normalized_input_lower": np.broadcast_to(input_min, (7,)).tolist(),
        "normalized_input_upper": np.broadcast_to(input_max, (7,)).tolist(),
        "physical_output_lower_rad_s": output_min.tolist(),
        "physical_output_upper_rad_s": output_max.tolist(),
        "velocity_gain_kp": gains["kp"].tolist(),
        "velocity_gain_ki": gains["ki"].tolist(),
        "velocity_gain_kd": gains["kd"].tolist(),
        "controller_policy_frequency_hz": int(controller.control_freq),
        "velocity_limits_rad_s": velocity_limits_record,
        "controller_joint_index": joint_index_record,
        "arm_qpos_indexes": arm_qpos_indexes.tolist(),
        "arm_qvel_indexes": arm_qvel_indexes.tolist(),
        "arm_joint_ids": joint_ids.tolist(),
        "arm_joint_names": ordered_joint_names,
        "arm_actuator_ids": actuator_indexes.tolist(),
        "arm_actuator_names": ordered_actuator_names,
        "arm_actuator_lower": actuator_min.tolist(),
        "arm_actuator_upper": actuator_max.tolist(),
        "compiled_arm_actuator_ctrlrange": compiled_ctrlrange.tolist(),
        "interpolator": None,
    }


def _finite_tolerance(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError("%s must be finite and positive" % label)
    output = float(value)
    if not math.isfinite(output) or output <= 0.0:
        raise ValueError("%s must be finite and positive" % label)
    return output


def _software_value(value: Any) -> Any:
    """Return a finite JSON snapshot of controller software memory."""

    np = _numpy()
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise ValueError("controller software state contains a non-finite array")
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, np.generic):
        return _software_value(value.item())
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("controller software state contains a non-finite scalar")
        return value
    if isinstance(value, (list, tuple)):
        return [_software_value(item) for item in value]
    if isinstance(value, deque):
        return [_software_value(item) for item in list(value)]
    if isinstance(value, Mapping):
        return {
            str(key): _software_value(item)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    raise ValueError(
        "controller software state contains unsupported type %s"
        % type(value).__name__
    )


def _controller_software_state(controller: Any) -> Dict[str, Any]:
    fields = (
        "initial_joint",
        "goal_vel",
        "current_vel",
        "last_err",
        "summed_err",
        "last_joint_vel",
        "torques",
        "new_update",
        "saturated",
        "joint_pos",
        "joint_vel",
        "torque_compensation",
    )
    state = {
        field: {
            "available": hasattr(controller, field),
            "value": _software_value(getattr(controller, field, None)),
        }
        for field in fields
    }
    derivative = getattr(controller, "derr_buf", None)
    derivative_fields = {}
    if derivative is not None:
        for field in ("_size", "size", "_buf", "buf", "_data", "data"):
            if hasattr(derivative, field):
                derivative_fields[field] = _software_value(
                    getattr(derivative, field)
                )
        derivative_dict = getattr(derivative, "__dict__", None)
        if isinstance(derivative_dict, dict):
            for field, value in sorted(derivative_dict.items()):
                if field not in derivative_fields:
                    derivative_fields[field] = _software_value(value)
    state["derivative_ring_buffer"] = {
        "available": derivative is not None,
        "fields": derivative_fields,
    }
    payload = {
        "schema_version": "vlsa_poisson_joint_velocity_controller_software_state.v1",
        "fields": state,
    }
    payload["sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def restore_osc_settled_state_into_joint_velocity_env(
    source_osc_env: Any,
    target_joint_velocity_env: Any,
    settled_state: Sequence[float],
    *,
    max_arm_qpos_error_rad: float,
    max_arm_qvel_error_rad_s: float,
    require_official_integration_state: bool = True,
) -> Dict[str, Any]:
    """Restore one OSC-settled physical state and reset all JV PID memory.

    This function changes only the explicitly supplied target environment.
    The source environment is read-only.  It also copies episode bookkeeping
    so five 100 Hz controls can later be grouped as one 20 Hz task step.
    """

    np = _numpy()
    qpos_tolerance = _finite_tolerance(
        max_arm_qpos_error_rad, "max_arm_qpos_error_rad"
    )
    qvel_tolerance = _finite_tolerance(
        max_arm_qvel_error_rad_s, "max_arm_qvel_error_rad_s"
    )
    if not isinstance(require_official_integration_state, bool):
        raise TypeError("require_official_integration_state must be Boolean")
    source_topology = model_topology_contract(source_osc_env.sim)
    target_topology = model_topology_contract(target_joint_velocity_env.sim)
    if source_topology["sha256"] != target_topology["sha256"]:
        raise ValueError("OSC and joint-velocity MuJoCo model topologies differ")
    source_physics = None
    target_physics = None
    source_official_state = None
    target_official_state = None
    try:
        source_physics = model_physics_contract(source_osc_env.sim)
        target_physics = model_physics_contract(target_joint_velocity_env.sim)
        if source_physics["sha256"] != target_physics["sha256"]:
            raise ValueError("OSC and joint-velocity compiled physical models differ")
        source_raw_model, _, source_official_state = _official_integration_state(
            source_osc_env
        )
        target_raw_model, target_raw_data, _ = _official_integration_state(
            target_joint_velocity_env
        )
        if int(source_raw_model.nq) != int(target_raw_model.nq):
            raise ValueError("OSC and joint-velocity official state layouts differ")
        import mujoco

        specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_setState(
            target_raw_model,
            target_raw_data,
            source_official_state.copy(),
            specification,
        )
        # Refresh position-dependent derived arrays once, then restore the
        # complete integration vector a second time. ``mj_forward`` is allowed
        # to update warm-start state; the second set preserves exact pairing
        # while the qpos-derived geometry remains current.
        target_joint_velocity_env.sim.forward()
        mujoco.mj_setState(
            target_raw_model,
            target_raw_data,
            source_official_state.copy(),
            specification,
        )
        _, _, target_official_state = _official_integration_state(
            target_joint_velocity_env
        )
        if not np.array_equal(source_official_state, target_official_state):
            raise RuntimeError("complete MuJoCo integration-state restore was not exact")
    except (ImportError, RuntimeError, TypeError, ValueError):
        if require_official_integration_state:
            raise
        source_physics = None
        target_physics = None
        source_official_state = None
        target_official_state = None
    controller_record = joint_velocity_controller_contract(target_joint_velocity_env)

    source_state = np.asarray(
        source_osc_env.sim.get_state().flatten(), dtype=np.float64
    )
    registered_state = np.asarray(settled_state, dtype=np.float64)
    if (
        registered_state.ndim != 1
        or registered_state.shape != source_state.shape
        or not np.all(np.isfinite(registered_state))
    ):
        raise ValueError("settled_state must be one finite flattened simulator state")
    if not np.array_equal(source_state, registered_state):
        raise ValueError("settled_state is not the source OSC environment's exact state")

    source_robot = source_osc_env.robots[0]
    target_robot = target_joint_velocity_env.robots[0]
    source_qpos_indexes = np.asarray(
        source_robot._ref_joint_pos_indexes, dtype=np.int64
    )
    source_qvel_indexes = np.asarray(
        source_robot._ref_joint_vel_indexes, dtype=np.int64
    )
    target_qpos_indexes = np.asarray(
        target_robot._ref_joint_pos_indexes, dtype=np.int64
    )
    target_qvel_indexes = np.asarray(
        target_robot._ref_joint_vel_indexes, dtype=np.int64
    )
    if not (
        np.array_equal(source_qpos_indexes, target_qpos_indexes)
        and np.array_equal(source_qvel_indexes, target_qvel_indexes)
        and source_qpos_indexes.shape == (7,)
        and source_qvel_indexes.shape == (7,)
    ):
        raise ValueError("OSC and joint-velocity Panda joint indexes differ")

    if source_official_state is None:
        target_joint_velocity_env.sim.set_state_from_flattened(
            registered_state.copy()
        )
        target_joint_velocity_env.sim.forward()
    target_joint_velocity_env.env.timestep = int(source_osc_env.env.timestep)
    target_joint_velocity_env.env.cur_time = float(source_osc_env.env.cur_time)
    target_joint_velocity_env.env.done = bool(source_osc_env.env.done)

    controller = target_robot.controller
    controller.update_initial_joints(
        np.asarray(
            target_joint_velocity_env.sim.data.qpos[target_qpos_indexes],
            dtype=np.float64,
        )
    )
    controller.reset_goal()
    controller.current_vel = np.zeros(7, dtype=np.float64)
    controller.last_err = np.zeros(7, dtype=np.float64)
    controller.summed_err = np.zeros(7, dtype=np.float64)
    controller.derr_buf.clear()
    controller.saturated = False
    controller.last_joint_vel = np.asarray(
        target_joint_velocity_env.sim.data.qvel[target_qvel_indexes],
        dtype=np.float64,
    ).copy()
    controller.torques = None
    controller.new_update = True
    controller.update(force=True)

    state_before_observables = np.asarray(
        target_joint_velocity_env.sim.get_state().flatten(), dtype=np.float64
    ).copy()
    target_joint_velocity_env._post_process()
    target_joint_velocity_env._update_observables(force=True)
    target_joint_velocity_env.check_success()
    controller_software_state = _controller_software_state(controller)
    state_after_observables = np.asarray(
        target_joint_velocity_env.sim.get_state().flatten(), dtype=np.float64
    ).copy()
    if not np.array_equal(state_before_observables, state_after_observables):
        raise RuntimeError("target observation/controller synchronization changed physics")
    if source_official_state is not None:
        # ``controller.update(force=True)`` intentionally forwards MuJoCo to
        # synchronize Jacobians and controller memory.  Forward dynamics may
        # rewrite hidden integration fields such as qacc_warmstart even though
        # qpos/qvel are unchanged.  Reapply the registered full state once,
        # without another forward, so both paired arms enter their first
        # scheduled 100 Hz forward from the exact same historical state.
        mujoco.mj_setState(
            target_raw_model,
            target_raw_data,
            source_official_state.copy(),
            int(mujoco.mjtState.mjSTATE_INTEGRATION),
        )
        _, _, target_official_state = _official_integration_state(
            target_joint_velocity_env
        )
        if not np.array_equal(source_official_state, target_official_state):
            raise RuntimeError(
                "controller synchronization changed the complete MuJoCo integration state"
            )

    source_arm_qpos = np.asarray(
        source_osc_env.sim.data.qpos[source_qpos_indexes], dtype=np.float64
    )
    target_arm_qpos = np.asarray(
        target_joint_velocity_env.sim.data.qpos[target_qpos_indexes], dtype=np.float64
    )
    source_arm_qvel = np.asarray(
        source_osc_env.sim.data.qvel[source_qvel_indexes], dtype=np.float64
    )
    target_arm_qvel = np.asarray(
        target_joint_velocity_env.sim.data.qvel[target_qvel_indexes], dtype=np.float64
    )
    qpos_error = float(np.max(np.abs(target_arm_qpos - source_arm_qpos)))
    qvel_error = float(np.max(np.abs(target_arm_qvel - source_arm_qvel)))
    exact_state = bool(np.array_equal(state_after_observables, registered_state))
    if (
        not exact_state
        or qpos_error > qpos_tolerance
        or qvel_error > qvel_tolerance
    ):
        raise RuntimeError("joint-velocity state restore failed its exact/tolerance checks")
    return {
        "schema_version": "vlsa_poisson_controller_state_restore.v1",
        "source_controller": "OSC_POSE",
        "target_controller": "JOINT_VELOCITY",
        "model_topology_sha256": source_topology["sha256"],
        "physical_model_sha256": (
            None if source_physics is None else source_physics["sha256"]
        ),
        "compiled_mjb_sha256": (
            None if source_physics is None else source_physics["compiled_mjb_sha256"]
        ),
        "official_integration_state_available": source_official_state is not None,
        "official_integration_state_sha256": (
            None
            if source_official_state is None
            else _array_sha256(source_official_state)
        ),
        "target_official_integration_state_sha256": (
            None
            if target_official_state is None
            else _array_sha256(target_official_state)
        ),
        "settled_state_sha256": _array_sha256(registered_state),
        "target_state_sha256": _array_sha256(state_after_observables),
        "exact_flattened_state": exact_state,
        "maximum_arm_qpos_error_rad": qpos_error,
        "maximum_arm_qvel_error_rad_s": qvel_error,
        "max_arm_qpos_error_tolerance_rad": qpos_tolerance,
        "max_arm_qvel_error_tolerance_rad_s": qvel_tolerance,
        "copied_timestep": int(target_joint_velocity_env.env.timestep),
        "copied_cur_time_s": float(target_joint_velocity_env.env.cur_time),
        "copied_done": bool(target_joint_velocity_env.env.done),
        "controller": controller_record,
        "controller_software_state": controller_software_state,
        "pid_memory_reset": {
            "goal_velocity_zero": bool(np.array_equal(controller.goal_vel, np.zeros(7))),
            "current_velocity_zero": bool(np.array_equal(controller.current_vel, np.zeros(7))),
            "last_error_zero": bool(np.array_equal(controller.last_err, np.zeros(7))),
            "summed_error_zero": bool(np.array_equal(controller.summed_err, np.zeros(7))),
            "derivative_buffer_size": int(controller.derr_buf._size),
            "saturated": bool(controller.saturated),
        },
    }


__all__ = [
    "joint_velocity_controller_contract",
    "model_topology_contract",
    "model_physics_contract",
    "restore_osc_settled_state_into_joint_velocity_env",
]
