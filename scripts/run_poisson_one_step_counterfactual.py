#!/usr/bin/env python3
"""Allocation-only one-filter-interval Poisson counterfactual.

This Stage-13 runner replays the immutable historical OSC prefix once, finds
the first registered 100 Hz boundary at or after the independently identified
warning and strictly before the first selected link-5/6 MuJoCo contact, and
restores that exact state into two fresh JOINT_VELOCITY environments.  The two
arms hold either the measured OSC five-substep average or one Poisson-QP
correction for exactly five 2 ms physics substeps.  No policy process is
started or queried.

``result.json`` is never written from partial physics.  A complete candidate
must first pass the pure core validator, is then atomically published, loaded
back and validated again, and only then receives its final validation receipt.
Every existing run directory is immutable and rejected.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import sys
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_CASE_ID = "vlsa-t1-goal-ii-t0-e05"
EXPECTED_NUMERIC_SCHEMA = "vlsa_poisson_numeric_validation.v1"
EXPECTED_PARITY_SCHEMA = "vlsa_poisson_shadow_parity.v3"
EXPECTED_IDENTIFICATION_SCHEMA = "vlsa_poisson_shadow_identification.v4"
EXPECTED_RESULT_SCHEMA = "vlsa_poisson_one_step_counterfactual_result.v1"
ARM_NAMES = ("nominal", "psf")
PHYSICS_SUBSTEPS_PER_OSC_ACTION = 25
PHYSICS_SUBSTEPS_PER_FILTER_UPDATE = 5
PHYSICS_TIMESTEP_S = 0.002
COUNTERFACTUAL_HORIZON_S = 0.01
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PHYSICAL_MODEL_CONTRACT_KEYS = (
    "schema_version",
    "sha256",
    "field_count",
    "option_field_count",
    "compiled_mjb_sha256",
    "compiled_mjb_bytes",
    "nq",
    "nv",
    "na",
    "mjstate_integration_size",
    "robosuite_flattened_state_size",
    "robosuite_flattened_state_layout",
)


class OneStepRunnerError(RuntimeError):
    """Typed apparatus/protocol refusal before a scientific result exists."""


class _PrefixComplete(RuntimeError):
    """Private sentinel used to stop OSC immediately at the captured boundary."""


class _NominalVelocityPreflightInadmissible(RuntimeError):
    """Private sentinel for one complete, non-executed scientific outcome."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"vlsa-table1-array-v1\0")
    digest.update(header)
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise OneStepRunnerError("%s must be an existing real file" % label)
    return path.resolve()


def _load_json_object(path: Path, label: str) -> Dict[str, Any]:
    _require_regular_file(path, label)

    def reject_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("duplicate JSON key %r" % key)
            output[key] = value
        return output

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError("non-finite JSON constant %r" % item)
            ),
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise OneStepRunnerError("%s is invalid JSON" % label) from error
    if not isinstance(value, dict):
        raise OneStepRunnerError("%s must contain one JSON object" % label)
    return value


def _require_exact_physical_model_contract(
    value: Any, label: str
) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(
        PHYSICAL_MODEL_CONTRACT_KEYS
    ):
        raise OneStepRunnerError("%s physical-model fields differ" % label)
    if value.get("schema_version") != "vlsa_poisson_physical_model.v3":
        raise OneStepRunnerError("%s physical-model schema differs" % label)
    for field in ("sha256", "compiled_mjb_sha256"):
        digest = value.get(field)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise OneStepRunnerError(
                "%s physical-model %s is invalid" % (label, field)
            )
    for field in ("field_count", "option_field_count", "compiled_mjb_bytes"):
        item = value.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise OneStepRunnerError(
                "%s physical-model %s is invalid" % (label, field)
            )
    dimensions = {}
    for field, minimum in (("nq", 1), ("nv", 1), ("na", 0)):
        item = value.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
            raise OneStepRunnerError(
                "%s physical-model %s is invalid" % (label, field)
            )
        dimensions[field] = int(item)
    common_size = 1 + dimensions["nq"] + dimensions["nv"] + dimensions["na"]
    integration_size = value.get("mjstate_integration_size")
    flattened_size = value.get("robosuite_flattened_state_size")
    if (
        isinstance(integration_size, bool)
        or not isinstance(integration_size, int)
        or integration_size < common_size
        or isinstance(flattened_size, bool)
        or not isinstance(flattened_size, int)
        or flattened_size != common_size
        or value.get("robosuite_flattened_state_layout")
        != "time_qpos_qvel_act_no_udd_tail"
    ):
        raise OneStepRunnerError("%s physical-model state layout differs" % label)
    return dict(value)


def _validated_new_output(output_root: Path, run_id: str) -> Path:
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise OneStepRunnerError(
            "run ID must be 1-128 portable alphanumeric/dot/underscore/hyphen characters"
        )
    root = Path(output_root)
    if root.is_symlink() or not root.is_dir():
        raise OneStepRunnerError("output root must be an existing real directory")
    resolved = root.resolve()
    output = resolved / run_id
    if output.parent != resolved or output.is_symlink():
        raise OneStepRunnerError("run output must remain inside the registered root")
    if output.exists():
        raise OneStepRunnerError(
            "run output already exists; Stage 13 requires a new immutable run ID"
        )
    output.mkdir(mode=0o755)
    return output


def _protocol_paths(root: Path, protocol_path: Path) -> Tuple[Path, Dict[str, Any], str]:
    path = protocol_path if protocol_path.is_absolute() else root / protocol_path
    path = _require_regular_file(path, "one-step protocol")
    protocol = _load_json_object(path, "one-step protocol")
    if (
        protocol.get("schema_version")
        != "vlsa_poisson_one_step_counterfactual_protocol.v2"
        or protocol.get("result_contract", {}).get("schema_version")
        != EXPECTED_RESULT_SCHEMA
    ):
        raise OneStepRunnerError("one-step protocol schema/result binding differs")
    if protocol.get("case", {}).get("case_id") != DEFAULT_CASE_ID:
        raise OneStepRunnerError("one-step protocol is not bound to the registered canary")
    return path, protocol, _file_sha256(path)


def _raw_model_data(sim: Any) -> Tuple[Any, Any]:
    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    return model, data


def _official_integration_state(sim: Any) -> Any:
    import mujoco
    import numpy as np

    model, data = _raw_model_data(sim)
    specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(int(mujoco.mj_stateSize(model, specification)), dtype=np.float64)
    mujoco.mj_getState(model, data, state, specification)
    if state.ndim != 1 or state.size <= 0 or not np.all(np.isfinite(state)):
        raise OneStepRunnerError("official MuJoCo integration state is invalid")
    return state


def _require_integration_state_prefix_layout(
    official: Any,
    *,
    time_value: float,
    qpos: Any,
    qvel: Any,
    act: Any,
) -> None:
    """Require MuJoCo's exact common [time, qpos, qvel, act] prefix."""

    import numpy as np

    official_array = np.asarray(official, dtype=np.float64)
    qpos_array = np.asarray(qpos, dtype=np.float64)
    qvel_array = np.asarray(qvel, dtype=np.float64)
    act_array = np.asarray(act, dtype=np.float64)
    expected_prefix = np.concatenate(
        (
            np.asarray([float(time_value)], dtype=np.float64),
            qpos_array,
            qvel_array,
            act_array,
        )
    )
    if (
        official_array.ndim != 1
        or qpos_array.ndim != 1
        or qvel_array.ndim != 1
        or act_array.ndim != 1
        or official_array.size < expected_prefix.size
        or not np.array_equal(
            official_array[: expected_prefix.size], expected_prefix
        )
    ):
        raise OneStepRunnerError(
            "mjSTATE_INTEGRATION prefix is not exact [time, qpos, qvel, act]"
        )


def _capture_state(env: Any, physical_boundary: int) -> Dict[str, Any]:
    import numpy as np

    official = _official_integration_state(env.sim)
    flattened = np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()
    model, data = _raw_model_data(env.sim)
    qpos = np.asarray(data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(data.qvel, dtype=np.float64).copy()
    act = np.asarray(data.act, dtype=np.float64).copy()
    _require_integration_state_prefix_layout(
        official,
        time_value=float(data.time),
        qpos=qpos,
        qvel=qvel,
        act=act,
    )
    expected_flattened = np.concatenate(
        (
            np.asarray([float(data.time)], dtype=np.float64),
            qpos,
            qvel,
            act,
        )
    )
    if not (
        flattened.ndim == 1
        and qpos.shape == (int(model.nq),)
        and qvel.shape == (int(model.nv),)
        and act.shape == (int(model.na),)
        and flattened.shape == expected_flattened.shape
        and np.array_equal(flattened, expected_flattened)
        and np.all(np.isfinite(flattened))
        and np.all(np.isfinite(qpos))
        and np.all(np.isfinite(qvel))
        and np.all(np.isfinite(act))
    ):
        raise OneStepRunnerError("captured MuJoCo state is incomplete or non-finite")
    return {
        "physical_boundary": int(physical_boundary),
        "mujoco_state_specification": "mjSTATE_INTEGRATION",
        "integration_state": official.tolist(),
        "integration_state_sha256": _array_sha256(official),
        "flattened_simulator_state": flattened.tolist(),
        "flattened_simulator_state_sha256": _array_sha256(flattened),
        "qpos": qpos.tolist(),
        "qpos_sha256": _array_sha256(qpos),
        "qvel": qvel.tolist(),
        "qvel_sha256": _array_sha256(qvel),
        "wrapper_bookkeeping": {
            "timestep": int(env.env.timestep),
            "cur_time_s": float(env.env.cur_time),
            "done": bool(env.env.done),
        },
    }


def _contact_from_identification(
    identification: Mapping[str, Any], protocol: Mapping[str, Any]
) -> Tuple[Dict[str, Any], int, int]:
    from main.poisson_fullbody.one_step_counterfactual import (
        contact_physical_boundary_index,
    )

    shadow = identification.get("shadow_replay")
    poisson = shadow.get("poisson_identification") if isinstance(shadow, Mapping) else None
    assessment = (
        poisson.get("contact_prediction_assessment")
        if isinstance(poisson, Mapping)
        else None
    )
    contact = assessment.get("first_link56_contact") if isinstance(assessment, Mapping) else None
    warning = (
        assessment.get("primary_registered_warning")
        if isinstance(assessment, Mapping)
        else None
    )
    if (
        not isinstance(contact, Mapping)
        or not isinstance(warning, Mapping)
        or assessment.get("assessment")
        != "registered_warning_preceded_link56_contact"
        or contact.get("is_physical_nonpositive_distance_contact") is not True
    ):
        raise OneStepRunnerError(
            "validated identification lacks the actionable physical link-5/6 contact"
        )
    try:
        contact_boundary = contact_physical_boundary_index(
            int(contact["observation_index"]), str(contact["source_phase"])
        )
        warning_boundary = int(warning["observation_index"]) + 1
        period = int(
            protocol["counterfactual_boundary"][
                "physics_substeps_per_filter_update"
            ]
        )
        boundary = ((warning_boundary + period - 1) // period) * period
    except (KeyError, TypeError, ValueError) as error:
        raise OneStepRunnerError(
            "warning/contact/filter boundary derivation failed"
        ) from error
    if (
        boundary < 0
        or boundary < warning_boundary
        or boundary - warning_boundary >= period
        or boundary >= contact_boundary
    ):
        raise OneStepRunnerError(
            "first scheduled filter boundary after warning is not before contact"
        )
    target = dict(contact)
    required = (
        "robot_geom_id",
        "robot_geom_name",
        "robot_body_id",
        "robot_body_name",
        "obstacle_geom_id",
        "obstacle_geom_name",
        "obstacle_body_id",
        "obstacle_body_name",
    )
    if any(field not in target for field in required):
        raise OneStepRunnerError("identification target contact identity is incomplete")
    return target, int(contact_boundary), int(boundary)


def _identification_callback_authority(
    identification: Mapping[str, Any],
) -> Dict[str, Any]:
    """Project the identification's independently hashed callback state ledger."""

    shadow = identification.get("shadow_replay")
    if not isinstance(shadow, Mapping):
        raise OneStepRunnerError("identification shadow replay is missing")
    rows = shadow.get("callback_state_read_only_ledger")
    count = shadow.get("callback_count")
    if (
        not isinstance(rows, list)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or len(rows) != count
    ):
        raise OneStepRunnerError("identification callback state population differs")
    after_ledger: List[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise OneStepRunnerError(
                "identification callback state row %d is invalid" % index
            )
        digest = row.get("after_sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise OneStepRunnerError(
                "identification callback state row %d lacks an exact after hash"
                % index
            )
        after_ledger.append(digest)
    if shadow.get("callback_state_read_only_ledger_sha256") != _sha256_bytes(
        _canonical(rows)
    ):
        raise OneStepRunnerError(
            "identification callback read-only ledger hash differs"
        )
    sequence_sha256 = _sha256_bytes(_canonical(after_ledger))
    if shadow.get("callback_state_sequence_sha256") != sequence_sha256:
        raise OneStepRunnerError(
            "identification callback after-state sequence hash differs"
        )
    return {
        "callback_state_read_only_count": count,
        "callback_state_read_only_after_sha256_ledger": after_ledger,
        "callback_state_sequence_sha256": sequence_sha256,
    }


def _capture_exact_prefix(
    *,
    env: Any,
    evaluator: Any,
    runtime: Mapping[str, Any],
    replay: Any,
    parity: Mapping[str, Any],
    identification: Mapping[str, Any],
    boundary: int,
) -> Dict[str, Any]:
    """Execute the exact OSC prefix once and capture B and B+5."""

    from scripts.run_poisson_shadow_parity import _check_step

    np = runtime["np"]
    end_boundary = int(boundary) + PHYSICS_SUBSTEPS_PER_FILTER_UPDATE
    parity_ledger = parity["callback_replay"][
        "official_integration_state_sha256_ledger"
    ]
    id_rows = identification["shadow_replay"]["callback_state_read_only_ledger"]
    identification_ledger = [row["after_sha256"] for row in id_rows]
    if parity_ledger != identification_ledger:
        raise OneStepRunnerError("parity and identification callback ledgers differ")
    if end_boundary > len(parity_ledger):
        raise OneStepRunnerError("counterfactual endpoint exceeds the exact replay ledger")

    captures: Dict[int, Dict[str, Any]] = {}
    if boundary == 0:
        captures[0] = _capture_state(env, 0)
    observed_hashes: List[str] = []
    previous_goal = evaluator._goal_progress_snapshot(
        env,
        evaluator._goal_progress_definition(env)[1],
        step=-1,
        previous_values=None,
    )["values"]
    goal_atoms = evaluator._goal_progress_definition(env)[1]
    stopped = False
    for expected_step in replay.steps:
        high_index = int(expected_step.step)

        def callback(sim: Any, substep_index: int) -> None:
            nonlocal stopped
            physical = high_index * PHYSICS_SUBSTEPS_PER_OSC_ACTION + int(substep_index) + 1
            state_hash = _array_sha256(_official_integration_state(sim))
            if state_hash != parity_ledger[physical - 1]:
                raise OneStepRunnerError(
                    "OSC callback differs from exact parity at physical boundary %d"
                    % physical
                )
            if state_hash != identification_ledger[physical - 1]:
                raise OneStepRunnerError(
                    "OSC callback differs from identification at physical boundary %d"
                    % physical
                )
            observed_hashes.append(state_hash)
            if physical in (boundary, end_boundary):
                captures[physical] = _capture_state(env, physical)
            if physical == end_boundary:
                stopped = True
                raise _PrefixComplete()

        try:
            observation, reward, done, _ = env.step_with_substep_callback(
                expected_step.action,
                callback,
                expected_substeps=PHYSICS_SUBSTEPS_PER_OSC_ACTION,
            )
        except _PrefixComplete:
            break
        _, _, previous_goal = _check_step(
            evaluator=evaluator,
            env=env,
            observation=observation,
            reward=reward,
            done=done,
            expected_step=expected_step,
            goal_atoms=goal_atoms,
            previous_goal_values=previous_goal,
            np=np,
        )
        proxy = evaluator._eef_proxy(runtime, observation)
        evaluator._update_eef_marker(env, proxy)
    if not stopped or set(captures) != {boundary, end_boundary}:
        raise OneStepRunnerError("OSC prefix did not capture exactly B and B+5")
    expected_prefix = parity_ledger[:end_boundary]
    if observed_hashes != expected_prefix:
        raise OneStepRunnerError("observed OSC callback prefix is incomplete")
    return {
        "execution": "one_exact_historical_OSC_prefix_no_policy_query",
        "observed_callback_count": len(observed_hashes),
        "expected_callback_count": end_boundary,
        "observed_callback_sha256_ledger": observed_hashes,
        "observed_callback_prefix_sha256": _sha256_bytes(_canonical(observed_hashes)),
        "parity_callback_prefix_sha256": _sha256_bytes(_canonical(expected_prefix)),
        "identification_callback_prefix_sha256": _sha256_bytes(
            _canonical(identification_ledger[:end_boundary])
        ),
        "historical_executed_action_sequence_sha256": replay.executed_sequence_sha256,
        "state_at_B": captures[boundary],
        "state_at_B_plus_5": captures[end_boundary],
    }


def _restore_source_to_boundary(source_env: Any, state: Mapping[str, Any]) -> Dict[str, Any]:
    import mujoco
    import numpy as np

    model, data = _raw_model_data(source_env.sim)
    official = np.asarray(state["integration_state"], dtype=np.float64)
    specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_setState(model, data, official.copy(), specification)
    source_env.sim.forward()
    mujoco.mj_setState(model, data, official.copy(), specification)
    bookkeeping = state["wrapper_bookkeeping"]
    source_env.env.timestep = int(bookkeeping["timestep"])
    source_env.env.cur_time = float(bookkeeping["cur_time_s"])
    source_env.env.done = bool(bookkeeping["done"])
    observed_official = _official_integration_state(source_env.sim)
    observed_flattened = np.asarray(
        source_env.sim.get_state().flatten(), dtype=np.float64
    )
    expected_flattened = np.asarray(
        state["flattened_simulator_state"], dtype=np.float64
    )
    if not np.array_equal(observed_official, official):
        raise OneStepRunnerError("source official state restore to B is not exact")
    if not np.array_equal(observed_flattened, expected_flattened):
        raise OneStepRunnerError("source flattened state restore to B is not exact")
    return {
        "method": "mj_setState_forward_mj_setState_plus_wrapper_bookkeeping",
        "integration_state_sha256": _array_sha256(observed_official),
        "flattened_simulator_state_sha256": _array_sha256(observed_flattened),
        "exact_official_integration_state": True,
        "exact_flattened_simulator_state": True,
        "wrapper_bookkeeping": dict(bookkeeping),
    }


def _derive_nominal_velocity(
    source_env: Any,
    state_at_B: Mapping[str, Any],
    state_at_B_plus_5: Mapping[str, Any],
    arm_dof_indices: Sequence[int],
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    import mujoco
    import numpy as np

    model, _ = _raw_model_data(source_env.sim)
    qpos_B = np.asarray(state_at_B["qpos"], dtype=np.float64)
    qpos_B5 = np.asarray(state_at_B_plus_5["qpos"], dtype=np.float64)
    qdot = np.empty(int(model.nv), dtype=np.float64)
    mujoco.mj_differentiatePos(
        model, qdot, COUNTERFACTUAL_HORIZON_S, qpos_B, qpos_B5
    )
    indexes = np.asarray(arm_dof_indices, dtype=np.int64)
    if indexes.shape != (7,) or len(set(int(value) for value in indexes)) != 7:
        raise OneStepRunnerError("registered arm DOF slice is invalid")
    if np.any(indexes < 0) or np.any(indexes >= int(model.nv)):
        raise OneStepRunnerError("registered arm DOF slice exceeds compiled model")
    joint_ids = np.asarray(model.dof_jntid, dtype=np.int64)[indexes]
    if len(set(int(value) for value in joint_ids)) != 7:
        raise OneStepRunnerError("registered arm DOFs do not map to seven joints")
    arm_qpos_indices: List[int] = []
    arm_joint_types: List[str] = []
    arm_joint_qpos_widths: List[int] = []
    for dof_index, joint_id in zip(indexes, joint_ids):
        if int(model.jnt_dofadr[int(joint_id)]) != int(dof_index):
            raise OneStepRunnerError(
                "registered arm DOF is not the first DOF of its compiled joint"
            )
        if int(model.jnt_type[int(joint_id)]) != int(
            mujoco.mjtJoint.mjJNT_HINGE
        ):
            raise OneStepRunnerError(
                "registered arm nominal reconstruction requires scalar hinge joints"
            )
        qpos_index = int(model.jnt_qposadr[int(joint_id)])
        next_qpos_index = (
            int(model.jnt_qposadr[int(joint_id) + 1])
            if int(joint_id) + 1 < int(model.njnt)
            else int(model.nq)
        )
        qpos_width = next_qpos_index - qpos_index
        if qpos_width != 1:
            raise OneStepRunnerError(
                "registered arm hinge joint does not have one qpos coordinate"
            )
        arm_qpos_indices.append(qpos_index)
        arm_joint_types.append("hinge")
        arm_joint_qpos_widths.append(qpos_width)
    arm = qdot[indexes]
    instantaneous_full = np.asarray(state_at_B["qvel"], dtype=np.float64)
    instantaneous_arm = instantaneous_full[indexes]
    estimator = protocol["nominal_velocity_estimator"]
    expected_qpos_indices = estimator.get("expected_arm_qpos_indices")
    if (
        not isinstance(expected_qpos_indices, list)
        or arm_qpos_indices != expected_qpos_indices
    ):
        raise OneStepRunnerError(
            "compiled arm hinge qpos mapping differs from frozen protocol"
        )
    lower = np.asarray(estimator["arm_velocity_lower_rad_s"], dtype=np.float64)
    upper = np.asarray(estimator["arm_velocity_upper_rad_s"], dtype=np.float64)
    finite = bool(np.all(np.isfinite(qdot)) and np.all(np.isfinite(arm)))
    in_bounds = bool(finite and np.all(arm >= lower) and np.all(arm <= upper))
    lower_excess = np.maximum(lower - arm, 0.0)
    upper_excess = np.maximum(arm - upper, 0.0)
    violation_indices = [
        int(index)
        for index in range(7)
        if float(lower_excess[index]) > 0.0
        or float(upper_excess[index]) > 0.0
    ]
    maximum_excess = float(
        max(
            float(np.max(lower_excess)),
            float(np.max(upper_excess)),
        )
    )
    record = {
        "method": "mujoco_mj_differentiatePos_full_nv",
        "interval_s": COUNTERFACTUAL_HORIZON_S,
        "qpos_B_sha256": state_at_B["qpos_sha256"],
        "qpos_B_plus_5_sha256": state_at_B_plus_5["qpos_sha256"],
        "full_nv_count": int(model.nv),
        "arm_dof_indices": [int(value) for value in indexes],
        "arm_qpos_indices": arm_qpos_indices,
        "arm_joint_types": arm_joint_types,
        "arm_joint_qpos_widths": arm_joint_qpos_widths,
        "qdot_nom_full_nv": qdot.tolist(),
        "qdot_nom_arm_slice": arm.tolist(),
        "instantaneous_source_qvel_full_nv_at_B": instantaneous_full.tolist(),
        "instantaneous_source_arm_qvel_at_B": instantaneous_arm.tolist(),
        "estimated_minus_instantaneous_arm_qvel": (
            arm - instantaneous_arm
        ).tolist(),
        "registered_lower_rad_s": lower.tolist(),
        "registered_upper_rad_s": upper.tolist(),
        "finite": finite,
        "arm_velocity_within_registered_bounds": in_bounds,
        "bound_violation_indices": violation_indices,
        "lower_bound_excess_rad_s": lower_excess.tolist(),
        "upper_bound_excess_rad_s": upper_excess.tolist(),
        "maximum_bound_excess_rad_s": maximum_excess,
        "out_of_bounds_policy": "inadmissible_no_hidden_clipping",
        "hidden_clipping_applied": False,
    }
    record["nominal_derivation_sha256"] = _sha256_bytes(_canonical(record))
    if not finite:
        raise OneStepRunnerError("nominal mj_differentiatePos velocity is non-finite")
    return record


def _mujoco_name(model: Any, object_type: Any, object_id: int, prefix: str) -> str:
    import mujoco

    value = mujoco.mj_id2name(model, object_type, int(object_id))
    return str(value) if value else "%s_%d" % (prefix, int(object_id))


def _all_contact_records(
    model: Any,
    data: Any,
    source_phase: str,
    physical_boundary: int,
) -> List[Dict[str, Any]]:
    import mujoco
    import numpy as np

    records = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        distance = float(contact.dist)
        position = np.asarray(contact.pos, dtype=np.float64)
        if not math.isfinite(distance) or not np.all(np.isfinite(position)):
            raise OneStepRunnerError("MuJoCo contact record is non-finite")
        records.append(
            {
                "source_phase": str(source_phase),
                "physical_boundary": int(physical_boundary),
                "mujoco_contact_index": int(contact_index),
                "mujoco_geom1_id": geom1,
                "mujoco_geom1_name": _mujoco_name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom1, "geom"
                ),
                "mujoco_body1_id": body1,
                "mujoco_body1_name": _mujoco_name(
                    model, mujoco.mjtObj.mjOBJ_BODY, body1, "body"
                ),
                "mujoco_geom2_id": geom2,
                "mujoco_geom2_name": _mujoco_name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom2, "geom"
                ),
                "mujoco_body2_id": body2,
                "mujoco_body2_name": _mujoco_name(
                    model, mujoco.mjtObj.mjOBJ_BODY, body2, "body"
                ),
                "contact_distance_m": distance,
                "is_physical_nonpositive_distance_contact": bool(distance <= 0.0),
                "solver_constraint_active": bool(int(contact.efc_address) >= 0),
                "efc_address": int(contact.efc_address),
                "contact_includemargin_m": float(contact.includemargin),
                "position_world_m": position.tolist(),
            }
        )
    return records


def _current_obstacle_boxes(model: Any, data: Any, settled_boxes: Sequence[Any]) -> Tuple[Any, ...]:
    import numpy as np
    from main.poisson_fullbody.geometry import OrientedBox

    output = []
    for settled in settled_boxes:
        geom_id = int(settled.geom_id)
        output.append(
            OrientedBox(
                center=np.asarray(data.geom_xpos[geom_id], dtype=np.float64),
                R=np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3),
                half_extents=np.asarray(model.geom_size[geom_id][:3], dtype=np.float64),
                geom_id=geom_id,
            )
        )
    return tuple(output)


def _rotation_angle(reference: Any, current: Any) -> float:
    import numpy as np

    relative = np.matmul(reference.T, current)
    cosine = (float(np.trace(relative)) - 1.0) * 0.5
    return float(math.acos(max(-1.0, min(1.0, cosine))))


def _selected_obstacle_drift(
    current_boxes: Sequence[Any], settled_boxes: Sequence[Any]
) -> Dict[str, float]:
    import numpy as np

    current_by_id = {int(box.geom_id): box for box in current_boxes}
    translation = 0.0
    rotation = 0.0
    surface = 0.0
    if set(current_by_id) != {int(box.geom_id) for box in settled_boxes}:
        raise OneStepRunnerError("selected-obstacle geom identity changed")
    for reference in settled_boxes:
        current = current_by_id[int(reference.geom_id)]
        translation = max(
            translation,
            float(np.linalg.norm(current.center - reference.center)),
        )
        rotation = max(rotation, _rotation_angle(reference.R, current.R))
        surface = max(
            surface,
            float(
                np.max(
                    np.linalg.norm(current.vertices() - reference.vertices(), axis=1)
                )
            ),
        )
    return {
        "selected_obstacle_translation_drift_m": float(translation),
        "selected_obstacle_rotation_drift_rad": float(rotation),
        "selected_obstacle_surface_drift_m": float(surface),
    }


def _selected_obstacle_body_motion(
    model: Any, data: Any, obstacle_body_ids: Sequence[int]
) -> Dict[str, Any]:
    import mujoco
    import numpy as np

    records = []
    for body_id in sorted(set(int(value) for value in obstacle_body_ids)):
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            int(body_id),
            velocity,
            0,
        )
        if not np.all(np.isfinite(velocity)):
            raise OneStepRunnerError("selected-obstacle body velocity is non-finite")
        angular = velocity[:3]
        linear = velocity[3:]
        records.append(
            {
                "body_id": int(body_id),
                "body_name": _mujoco_name(
                    model, mujoco.mjtObj.mjOBJ_BODY, body_id, "body"
                ),
                "angular_velocity_world_rad_per_s": angular.tolist(),
                "linear_velocity_world_m_per_s": linear.tolist(),
                "angular_speed_rad_per_s": float(np.linalg.norm(angular)),
                "linear_speed_m_per_s": float(np.linalg.norm(linear)),
            }
        )
    if not records:
        raise OneStepRunnerError("selected-obstacle body set is empty")
    return {
        "selected_obstacle_body_velocity_records": records,
        "selected_obstacle_max_body_linear_speed_m_s": max(
            row["linear_speed_m_per_s"] for row in records
        ),
        "selected_obstacle_max_body_angular_speed_rad_s": max(
            row["angular_speed_rad_per_s"] for row in records
        ),
        "velocity_reference": "mj_objectVelocity_world_orientation_rot_then_lin",
    }


def _classify_selected_contacts(
    records: Sequence[Mapping[str, Any]],
    resolved: Any,
    target_contact: Mapping[str, Any],
) -> Dict[str, Any]:
    pair_set = {
        (int(robot_geom), int(obstacle_geom))
        for robot_geom, obstacle_geom in resolved.collision_enabled_pairs
    }
    target_pair = (
        int(target_contact["robot_geom_id"]),
        int(target_contact["obstacle_geom_id"]),
    )
    selected_physical = []
    target_records = []
    shifted_records = []
    for record in records:
        if record.get("is_physical_nonpositive_distance_contact") is not True:
            continue
        geom1 = int(record["mujoco_geom1_id"])
        geom2 = int(record["mujoco_geom2_id"])
        pair = None
        if (geom1, geom2) in pair_set:
            pair = (geom1, geom2)
        elif (geom2, geom1) in pair_set:
            pair = (geom2, geom1)
        if pair is None:
            continue
        selected_physical.append(dict(record))
        if pair == target_pair:
            target_records.append(dict(record))
        else:
            shifted_records.append(dict(record))
    return {
        "target_contact_present": bool(target_records),
        "any_robot_to_selected_obstacle_contact_present": bool(selected_physical),
        "any_shifted_robot_to_selected_obstacle_contact_present": bool(
            shifted_records
        ),
        "target_physical_contact_records": target_records,
        "robot_to_selected_obstacle_physical_contact_records": selected_physical,
        "shifted_robot_to_selected_obstacle_physical_contact_records": shifted_records,
    }


def _state_measurement(
    *,
    sim: Any,
    bundle: Any,
    resolved: Any,
    full_samples: Any,
    arm_dof_indices: Sequence[int],
    target_contact: Mapping[str, Any],
    physical_boundary: int,
    commanded_arm_qdot: Sequence[float],
    normalized_action: Sequence[float],
    include_live_solver: bool = True,
) -> Dict[str, Any]:
    import numpy as np
    from main.poisson_fullbody.geometry import (
        minimum_point_to_oriented_boxes_distance,
    )
    from main.poisson_fullbody.jacobians import evaluate_point_jacobians
    from main.poisson_fullbody.measurement import clone_forwarded_state

    model, data = _raw_model_data(sim)
    official = _official_integration_state(sim)
    _require_integration_state_prefix_layout(
        official,
        time_value=float(data.time),
        qpos=data.qpos,
        qvel=data.qvel,
        act=data.act,
    )
    forwarded = clone_forwarded_state(model, data)
    qpos = np.asarray(forwarded.qpos, dtype=np.float64)
    qvel = np.asarray(forwarded.qvel, dtype=np.float64)
    if not (
        np.array_equal(qpos, np.asarray(data.qpos, dtype=np.float64))
        and np.array_equal(qvel, np.asarray(data.qvel, dtype=np.float64))
    ):
        raise OneStepRunnerError(
            "forwarded measurement qpos/qvel differ from integration-state source"
        )
    arm_indexes = np.asarray(arm_dof_indices, dtype=np.int64)
    points, jacobians = evaluate_point_jacobians(
        model, forwarded, bundle.protected_samples.samples, arm_indexes
    )
    queries = [bundle.field.query(point) for point in points]
    validity = []
    h_values: List[Optional[float]] = []
    gradient_values: List[Optional[List[float]]] = []
    for sample_index, (sample, query) in enumerate(
        zip(bundle.protected_samples.samples, queries)
    ):
        valid = bool(query.valid and query.value is not None and query.gradient is not None)
        reason = None
        if not valid:
            raw_reason = getattr(query, "reason", None)
            reason = str(getattr(raw_reason, "value", raw_reason or "invalid_query"))
        validity.append(
            {
                "sample_index": int(sample_index),
                "sample_id": int(sample.sample_id),
                "valid": valid,
                "reason": reason,
            }
        )
        h_values.append(float(query.value) if valid else None)
        gradient_values.append(
            np.asarray(query.gradient, dtype=np.float64).tolist() if valid else None
        )
    d_opt = np.asarray(
        [
            minimum_point_to_oriented_boxes_distance(point, bundle.obstacle_boxes)
            for point in points
        ],
        dtype=np.float64,
    )
    current_boxes = _current_obstacle_boxes(model, forwarded, bundle.obstacle_boxes)
    full_points = [sample.world_point(forwarded) for sample in full_samples.samples]
    full_sample_distances = [
        minimum_point_to_oriented_boxes_distance(point, current_boxes)
        for point in full_points
    ]
    if (
        not full_sample_distances
        or len(full_sample_distances) != len(full_samples.samples)
        or any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in full_sample_distances
        )
    ):
        raise OneStepRunnerError(
            "full-robot sample-distance ledger is incomplete or invalid"
        )
    minimum_sample_distance = min(float(value) for value in full_sample_distances)
    lower_bound = float(
        minimum_sample_distance - full_samples.maximum_surface_cover_radius_m
    )
    if not isinstance(include_live_solver, bool):
        raise OneStepRunnerError("include_live_solver must be Boolean")
    live_records = (
        _all_contact_records(
            model,
            data,
            "live_solver_phase_preintegration_geometry",
            int(physical_boundary) - 1,
        )
        if include_live_solver
        else []
    )
    post_records = _all_contact_records(
        model,
        forwarded,
        "post_integration_recomputed",
        int(physical_boundary),
    )
    contacts = live_records + post_records
    selected = _classify_selected_contacts(contacts, resolved, target_contact)
    post_selected = _classify_selected_contacts(
        post_records, resolved, target_contact
    )
    selected_distances = [
        float(record["contact_distance_m"])
        for record in selected["robot_to_selected_obstacle_physical_contact_records"]
    ]
    interval_d_sim = (
        min([lower_bound, 0.0] + selected_distances)
        if selected_distances
        else lower_bound
    )
    post_selected_distances = [
        float(record["contact_distance_m"])
        for record in post_selected[
            "robot_to_selected_obstacle_physical_contact_records"
        ]
    ]
    post_d_sim = (
        min([lower_bound, 0.0] + post_selected_distances)
        if post_selected_distances
        else lower_bound
    )
    output: Dict[str, Any] = {
        "physical_boundary": int(physical_boundary),
        "integration_state_sha256": _array_sha256(official),
        "integration_state": official.tolist(),
        "qpos": qpos.tolist(),
        "qvel": qvel.tolist(),
        "commanded_arm_qdot": [float(value) for value in commanded_arm_qdot],
        "normalized_joint_velocity_action_8d": [
            float(value) for value in normalized_action
        ],
        "measured_arm_qdot": qvel[arm_indexes].tolist(),
        "ordered_sample_h_m2": h_values,
        "ordered_sample_grad_h_m": gradient_values,
        "ordered_field_query_validity_and_reason": validity,
        "ordered_sample_D_opt_m": d_opt.tolist(),
        "minimum_D_opt_m": float(np.min(d_opt)),
        "minimum_D_sim_m": float(interval_d_sim),
        "interval_union_minimum_D_sim_m": float(interval_d_sim),
        "post_state_minimum_D_sim_m": float(post_d_sim),
        "minimum_exact_full_robot_sample_to_current_obstacle_m": float(
            minimum_sample_distance
        ),
        "ordered_full_robot_sample_distance_m": [
            float(value) for value in full_sample_distances
        ],
        "registered_full_robot_sample_count": len(full_samples.samples),
        "registered_full_robot_sample_ledger_sha256": (
            full_samples.sample_ledger_sha256
        ),
        "certified_full_robot_coverage_radius_m": float(
            full_samples.maximum_surface_cover_radius_m
        ),
        "all_mujoco_contact_pairs_with_distances": contacts,
        "exact_ordered_nonpositive_physical_contact_subset": [
            dict(record)
            for record in contacts
            if record["is_physical_nonpositive_distance_contact"] is True
        ],
        "invalid_field_query_count": sum(not row["valid"] for row in validity),
    }
    output.update(_selected_obstacle_drift(current_boxes, bundle.obstacle_boxes))
    output.update(
        _selected_obstacle_body_motion(
            model, forwarded, resolved.obstacle_body_ids
        )
    )
    output.update(selected)
    return output


def _boundary_filter_evidence(
    *,
    source_env: Any,
    bundle: Any,
    arm_dof_indices: Sequence[int],
    arm_qpos_indices: Sequence[int],
    qdot_nominal: Sequence[float],
    target_contact: Mapping[str, Any],
    runtime_protocol: Mapping[str, Any],
    stage_protocol: Mapping[str, Any],
    warning_authorization: Mapping[str, Any],
) -> Dict[str, Any]:
    import numpy as np
    from main.poisson_fullbody.cbf_qp import HardCbfQp, joint_velocity_bounds
    from main.poisson_fullbody.geometry import minimum_point_to_oriented_boxes_distance
    from main.poisson_fullbody.jacobians import evaluate_point_jacobians
    from main.poisson_fullbody.measurement import clone_forwarded_state
    from scripts.run_poisson_active_canary import _joint_limits

    model, data = _raw_model_data(source_env.sim)
    forwarded = clone_forwarded_state(model, data)
    arm_dofs = np.asarray(arm_dof_indices, dtype=np.int64)
    arm_qpos = np.asarray(arm_qpos_indices, dtype=np.int64)
    nominal = np.asarray(qdot_nominal, dtype=np.float64)
    points, jacobians = evaluate_point_jacobians(
        model, forwarded, bundle.protected_samples.samples, arm_dofs
    )
    queries = [bundle.field.query(point) for point in points]
    if not queries or any(not query.valid or query.value is None for query in queries):
        raise OneStepRunnerError("boundary-B protected Poisson field query is invalid")
    h = np.asarray([float(query.value) for query in queries], dtype=np.float64)
    gradients = np.asarray([query.gradient for query in queries], dtype=np.float64)
    if np.any(h <= 0.0):
        raise OneStepRunnerError(
            "boundary B is not a strict positive-h safe start for every protected sample"
        )
    d_opt = np.asarray(
        [
            minimum_point_to_oriented_boxes_distance(point, bundle.obstacle_boxes)
            for point in points
        ],
        dtype=np.float64,
    )
    qp_config = runtime_protocol["qp"]
    q_min, q_max = _joint_limits(source_env)
    _require_frozen_joint_limits(
        stage_protocol=stage_protocol,
        observed_lower_rad=q_min,
        observed_upper_rad=q_max,
    )
    q_arm = np.asarray(forwarded.qpos, dtype=np.float64)[arm_qpos]
    lower, upper = joint_velocity_bounds(
        q_arm,
        q_min,
        q_max,
        qp_config["velocity_lower_rad_s"],
        qp_config["velocity_upper_rad_s"],
        alpha_joint=float(qp_config["joint_limit_alpha_per_s"]),
        control_dt_seconds=COUNTERFACTUAL_HORIZON_S,
        position_margin_rad=float(qp_config["joint_position_margin_rad"]),
    )
    alpha = float(runtime_protocol["cbf"]["alpha_gain_per_s"])
    rows = np.einsum("ni,nij->nj", gradients, jacobians)
    row_lower = -alpha * h
    nominal_residuals = rows @ nominal + alpha * h
    qp = HardCbfQp(
        eps_abs=float(qp_config["eps_abs"]),
        eps_rel=float(qp_config["eps_rel"]),
        max_iter=int(qp_config["max_iterations"]),
        postcheck_cbf_tolerance=float(qp_config["postcheck_cbf_tolerance"]),
        postcheck_bound_tolerance=float(
            qp_config["postcheck_bound_tolerance_rad_s"]
        ),
    )
    result = qp.solve_from_field(
        nominal,
        h,
        gradients,
        jacobians,
        lower,
        upper,
        alpha=alpha,
        weight_diagonal=qp_config["weight_diagonal"],
        require_safe_start=True,
    )
    if not result.valid or result.qdot_safe is None:
        raise OneStepRunnerError("boundary-B hard Poisson QP failed: %s" % result.reason)
    safe = np.asarray(result.qdot_safe, dtype=np.float64)
    safe_residuals = rows @ safe + alpha * h
    samples = bundle.protected_samples.samples
    warning = warning_authorization.get("primary_registered_warning")
    warning_evidence = (
        warning.get("evidence") if isinstance(warning, Mapping) else None
    )
    if not isinstance(warning_evidence, Mapping):
        raise OneStepRunnerError("primary warning sample evidence is absent")
    try:
        registered_warning_sample = int(warning_evidence["sample_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise OneStepRunnerError("primary warning sample ID is invalid") from error
    if not 0 <= registered_warning_sample < len(samples):
        raise OneStepRunnerError("primary warning sample ID exceeds bound samples")
    warning_sample = samples[registered_warning_sample]
    identity_fields = {
        "sample_id": int(warning_sample.sample_id),
        "geom_id": int(warning_sample.geom_id),
        "geom_name": str(warning_sample.geom_name),
        "body_id": int(warning_sample.body_id),
        "body_name": str(warning_sample.body_name),
    }
    if any(warning_evidence.get(key) != value for key, value in identity_fields.items()):
        raise OneStepRunnerError(
            "primary warning sample identity differs from bound sample ledger"
        )
    if int(warning_sample.geom_id) != int(target_contact["robot_geom_id"]):
        raise OneStepRunnerError(
            "primary warning sample geom differs from historical target contact"
        )
    trend_sample = int(np.argmin(nominal_residuals))
    if trend_sample != registered_warning_sample:
        raise OneStepRunnerError(
            "primary warning sample is not the minimum nominal CBF residual at B"
        )
    diagnostics = dict(result.diagnostics)
    physical_lower = np.asarray(qp_config["velocity_lower_rad_s"], dtype=np.float64)
    physical_upper = np.asarray(qp_config["velocity_upper_rad_s"], dtype=np.float64)
    q_min_array = np.asarray(q_min, dtype=np.float64)
    q_max_array = np.asarray(q_max, dtype=np.float64)
    alpha_joint = float(qp_config["joint_limit_alpha_per_s"])
    position_margin = float(qp_config["joint_position_margin_rad"])
    allowed_lower = q_min_array + position_margin
    allowed_upper = q_max_array - position_margin
    continuous_lower = -alpha_joint * (q_arm - q_min_array)
    continuous_upper = alpha_joint * (q_max_array - q_arm)
    one_step_lower = (allowed_lower - q_arm) / COUNTERFACTUAL_HORIZON_S
    one_step_upper = (allowed_upper - q_arm) / COUNTERFACTUAL_HORIZON_S
    reconstructed_lower = np.maximum.reduce(
        (physical_lower, continuous_lower, one_step_lower)
    )
    reconstructed_upper = np.minimum.reduce(
        (physical_upper, continuous_upper, one_step_upper)
    )
    if not np.array_equal(reconstructed_lower, lower) or not np.array_equal(
        reconstructed_upper, upper
    ):
        raise OneStepRunnerError("joint-limit bound arithmetic differs from QP inputs")
    return {
        "arm_dof_indices": [int(value) for value in arm_dofs],
        "ordered_sample_identity": [sample.to_dict() for sample in samples],
        "ordered_sample_count": len(samples),
        "ordered_sample_ledger_sha256": bundle.hashes.protected_samples_sha256,
        "ordered_sample_points_world": points.tolist(),
        "ordered_sample_point_jacobians": jacobians.tolist(),
        "ordered_sample_h_m2": h.tolist(),
        "ordered_sample_grad_h_m": gradients.tolist(),
        "ordered_sample_D_opt_m": d_opt.tolist(),
        "velocity_lower_rad_s": lower.tolist(),
        "velocity_upper_rad_s": upper.tolist(),
        "one_CBF_row_per_exact_bound_sample": {
            "row_count": int(rows.shape[0]),
            "sample_count": len(samples),
            "rows_m_per_rad": rows.tolist(),
            "lower_bounds_m2_per_s": row_lower.tolist(),
            "alpha_gain_per_s": alpha,
            "no_slack": True,
        },
        "joint_velocity_bound_rows": {
            "physical_lower_rad_s": physical_lower.tolist(),
            "physical_upper_rad_s": physical_upper.tolist(),
            "final_lower_rad_s": lower.tolist(),
            "final_upper_rad_s": upper.tolist(),
        },
        "joint_position_constraint_rows": {
            "q_arm_rad": q_arm.tolist(),
            "q_min_rad": q_min_array.tolist(),
            "q_max_rad": q_max_array.tolist(),
            "position_margin_rad": position_margin,
            "allowed_lower_rad": allowed_lower.tolist(),
            "allowed_upper_rad": allowed_upper.tolist(),
            "joint_limit_alpha_per_s": alpha_joint,
            "continuous_lower_rad_s": continuous_lower.tolist(),
            "continuous_upper_rad_s": continuous_upper.tolist(),
            "one_step_dt_s": COUNTERFACTUAL_HORIZON_S,
            "one_step_lower_rad_s": one_step_lower.tolist(),
            "one_step_upper_rad_s": one_step_upper.tolist(),
        },
        "nominal_CBF_residuals_m2_per_s": nominal_residuals.tolist(),
        "safe_CBF_residuals_m2_per_s": safe_residuals.tolist(),
        "qdot_safe": safe.tolist(),
        "correction_norm": float(np.linalg.norm(safe - nominal)),
        "QP_status": diagnostics.get("status"),
        "QP_iterations": diagnostics.get("iterations"),
        "QP_postcheck": diagnostics,
        "QP_reason": str(result.reason),
        "trend_sample_index": int(trend_sample),
        "trend_sample_id": int(samples[trend_sample].sample_id),
        "trend_sample_robot_geom_id": int(samples[trend_sample].geom_id),
        "trend_sample_nominal_residual_m2_per_s": float(
            nominal_residuals[trend_sample]
        ),
    }


def _require_frozen_joint_limits(
    *,
    stage_protocol: Mapping[str, Any],
    observed_lower_rad: Sequence[float],
    observed_upper_rad: Sequence[float],
) -> bool:
    """Bind the compiled Panda joint ranges to the external Stage-13 protocol."""

    execution = stage_protocol.get("qp_execution")
    if not isinstance(execution, Mapping):
        raise OneStepRunnerError("Stage-13 QP execution authority is missing")
    expected_lower = execution.get("expected_arm_q_min_rad")
    expected_upper = execution.get("expected_arm_q_max_rad")
    tolerance_raw = execution.get("joint_limit_match_absolute_tolerance_rad")
    if (
        not isinstance(expected_lower, list)
        or not isinstance(expected_upper, list)
        or len(expected_lower) != 7
        or len(expected_upper) != 7
        or isinstance(tolerance_raw, bool)
        or not isinstance(tolerance_raw, (int, float))
    ):
        raise OneStepRunnerError("frozen Panda joint-limit authority is incomplete")
    observed_lower = [float(value) for value in observed_lower_rad]
    observed_upper = [float(value) for value in observed_upper_rad]
    frozen_lower = [float(value) for value in expected_lower]
    frozen_upper = [float(value) for value in expected_upper]
    tolerance = float(tolerance_raw)
    if (
        len(observed_lower) != 7
        or len(observed_upper) != 7
        or not math.isfinite(tolerance)
        or tolerance < 0.0
        or any(
            not math.isfinite(value)
            for value in observed_lower
            + observed_upper
            + frozen_lower
            + frozen_upper
        )
        or any(
            frozen_lower[index] >= frozen_upper[index] for index in range(7)
        )
    ):
        raise OneStepRunnerError("Panda joint-limit authority is invalid")
    differences = [
        abs(observed_lower[index] - frozen_lower[index]) for index in range(7)
    ] + [
        abs(observed_upper[index] - frozen_upper[index]) for index in range(7)
    ]
    if any(value > tolerance for value in differences):
        raise OneStepRunnerError(
            "compiled Panda joint limits differ from frozen Stage-13 authority"
        )
    return True


def _capture_joint_velocity_controller_authority(
    *,
    evaluator: Any,
    runtime: Mapping[str, Any],
    case: Mapping[str, Any],
    expected_physical_model: Mapping[str, Any],
) -> Dict[str, Any]:
    """Inspect the exact live JV interface even when physics is inadmissible."""

    from main.poisson_fullbody.controller_bridge import (
        joint_velocity_controller_contract,
        model_physics_contract,
    )

    env = None
    try:
        env, _, _, _ = evaluator._build_environment(
            runtime,
            case,
            render_resolution=evaluator.TABLE_RENDER_RESOLUTION,
            controller="JOINT_VELOCITY",
            control_frequency_hz=100,
        )
        observed_physical_model = _require_exact_physical_model_contract(
            model_physics_contract(env.sim), "controller-authority compiled model"
        )
        if _canonical(observed_physical_model) != _canonical(
            expected_physical_model
        ):
            raise OneStepRunnerError(
                "controller-authority environment physical model differs"
            )
        contract = joint_velocity_controller_contract(env)
        if not isinstance(contract, Mapping):
            raise OneStepRunnerError("joint-velocity controller contract is absent")
        return dict(contract)
    finally:
        if env is not None:
            env.close()


def _run_counterfactual_arm(
    *,
    arm_name: str,
    evaluator: Any,
    runtime: Mapping[str, Any],
    case: Mapping[str, Any],
    replay: Any,
    source_env: Any,
    state_at_B: Mapping[str, Any],
    boundary: int,
    command_arm_qdot: Sequence[float],
    gripper_command: float,
    bundle: Any,
    resolved: Any,
    full_samples: Any,
    target_contact: Mapping[str, Any],
    runtime_protocol: Mapping[str, Any],
    boundary_filter: Mapping[str, Any],
    expected_arm_qpos_indices: Sequence[int],
) -> Dict[str, Any]:
    import numpy as np
    from main.poisson_fullbody.controller_bridge import (
        restore_osc_settled_state_into_joint_velocity_env,
    )
    from main.poisson_fullbody.joint_velocity_adapter import (
        normalized_joint_velocity_action,
    )

    if arm_name not in ARM_NAMES:
        raise OneStepRunnerError("unknown counterfactual arm")
    command = np.asarray(command_arm_qdot, dtype=np.float64)
    if command.shape != (7,) or not np.all(np.isfinite(command)):
        raise OneStepRunnerError("counterfactual arm command must be one finite 7-vector")
    action = normalized_joint_velocity_action(command, float(gripper_command))
    if action.shape != (8,) or not np.all(np.isfinite(action)):
        raise OneStepRunnerError("normalized JV controller action is invalid")
    if not np.array_equal(action[:7], command / 0.5) or float(action[7]) != float(
        gripper_command
    ):
        raise OneStepRunnerError("JV action encoding changed command or gripper")

    env = None
    try:
        env, _, initial_observation, _ = evaluator._build_environment(
            runtime,
            case,
            render_resolution=evaluator.TABLE_RENDER_RESOLUTION,
            controller="JOINT_VELOCITY",
            control_frequency_hz=100,
        )
        initial_proxy = evaluator._eef_proxy(runtime, initial_observation)
        evaluator._update_eef_marker(env, initial_proxy)
        admissibility = runtime_protocol["admissibility"]
        restore = restore_osc_settled_state_into_joint_velocity_env(
            source_env,
            env,
            np.asarray(state_at_B["flattened_simulator_state"], dtype=np.float64),
            max_arm_qpos_error_rad=float(
                admissibility["max_state_restore_qpos_error_rad"]
            ),
            max_arm_qvel_error_rad_s=float(
                admissibility["max_state_restore_qvel_error_rad_s"]
            ),
        )
        expected_state_hash = state_at_B["integration_state_sha256"]
        expected_flattened_hash = state_at_B[
            "flattened_simulator_state_sha256"
        ]
        pid_memory_reset = restore.get("pid_memory_reset")
        if (
            restore.get("official_integration_state_available") is not True
            or restore.get("official_integration_state_sha256") != expected_state_hash
            or restore.get("target_official_integration_state_sha256")
            != expected_state_hash
            or restore.get("exact_flattened_state") is not True
            or restore.get("settled_state_sha256") != expected_flattened_hash
            or restore.get("target_state_sha256") != expected_flattened_hash
            or not isinstance(pid_memory_reset, Mapping)
            or pid_memory_reset.get("goal_velocity_zero") is not True
            or pid_memory_reset.get("current_velocity_zero") is not True
            or pid_memory_reset.get("last_error_zero") is not True
            or pid_memory_reset.get("summed_error_zero") is not True
            or pid_memory_reset.get("derivative_buffer_size") != 0
            or pid_memory_reset.get("saturated") is not False
        ):
            raise OneStepRunnerError("joint-velocity arm did not restore exact B state")
        arm_dofs = tuple(int(value) for value in env.robots[0]._ref_joint_vel_indexes)
        arm_qpos = tuple(int(value) for value in env.robots[0]._ref_joint_pos_indexes)
        expected_dofs = tuple(int(value) for value in boundary_filter["arm_dof_indices"])
        expected_qpos = tuple(int(value) for value in expected_arm_qpos_indices)
        if arm_dofs != expected_dofs or arm_qpos != expected_qpos:
            raise OneStepRunnerError(
                "fresh JV arm DOF/qpos registration differs from source"
            )

        start = _state_measurement(
            sim=env.sim,
            bundle=bundle,
            resolved=resolved,
            full_samples=full_samples,
            arm_dof_indices=arm_dofs,
            target_contact=target_contact,
            physical_boundary=boundary,
            commanded_arm_qdot=command,
            normalized_action=action,
            include_live_solver=False,
        )
        if start["integration_state_sha256"] != expected_state_hash:
            raise OneStepRunnerError("arm start measurement is not exact boundary B")
        ledger: List[Dict[str, Any]] = []

        def callback(sim: Any, substep_index: int) -> None:
            before = _array_sha256(_official_integration_state(sim))
            row = _state_measurement(
                sim=sim,
                bundle=bundle,
                resolved=resolved,
                full_samples=full_samples,
                arm_dof_indices=arm_dofs,
                target_contact=target_contact,
                physical_boundary=boundary + int(substep_index) + 1,
                commanded_arm_qdot=command,
                normalized_action=action,
            )
            after = _array_sha256(_official_integration_state(sim))
            if before != after or row["integration_state_sha256"] != before:
                raise OneStepRunnerError("counterfactual measurement mutated physics")
            elapsed = (int(substep_index) + 1) * PHYSICS_TIMESTEP_S
            h_B = np.asarray(boundary_filter["ordered_sample_h_m2"], dtype=np.float64)
            gradients_B = np.asarray(
                boundary_filter["ordered_sample_grad_h_m"], dtype=np.float64
            )
            jacobians_B = np.asarray(
                boundary_filter["ordered_sample_point_jacobians"], dtype=np.float64
            )
            derivative = np.einsum(
                "ni,nij,j->n", gradients_B, jacobians_B, command
            )
            predicted = h_B + elapsed * derivative
            row["first_order_predicted_h_m2"] = predicted.tolist()
            row["prediction_error_m2"] = [
                (
                    None
                    if actual is None
                    else float(actual) - float(predicted[index])
                )
                for index, actual in enumerate(row["ordered_sample_h_m2"])
            ]
            ledger.append(row)

        env.step_with_substep_callback(
            action,
            callback,
            expected_substeps=PHYSICS_SUBSTEPS_PER_FILTER_UPDATE,
            update_observables=False,
            collect_observations=False,
        )
        if (
            len(ledger) != PHYSICS_SUBSTEPS_PER_FILTER_UPDATE
            or [row["physical_boundary"] for row in ledger]
            != list(range(boundary + 1, boundary + 6))
        ):
            raise OneStepRunnerError("counterfactual arm lacks complete five-substep exposure")

        measured = np.asarray(
            [row["measured_arm_qdot"] for row in ledger], dtype=np.float64
        )
        errors = measured - command[None, :]
        tracking_linf = float(np.max(np.abs(errors)))
        tracking_rmse = float(np.sqrt(np.mean(np.square(errors))))
        model, _ = _raw_model_data(env.sim)
        start_qpos = np.asarray(start["qpos"], dtype=np.float64)
        end_qpos = np.asarray(ledger[-1]["qpos"], dtype=np.float64)
        tangent_displacement = np.empty(int(model.nv), dtype=np.float64)
        import mujoco

        mujoco.mj_differentiatePos(
            model, tangent_displacement, 1.0, start_qpos, end_qpos
        )
        measured_motion_l2 = float(
            np.linalg.norm(tangent_displacement[np.asarray(arm_dofs, dtype=np.int64)])
        )
        command_norm = float(np.linalg.norm(command))
        command_integral = command_norm * COUNTERFACTUAL_HORIZON_S
        measured_motion_ratio = (
            None
            if command_integral == 0.0
            else measured_motion_l2 / command_integral
        )
        integrated_speed_l2 = float(
            sum(np.linalg.norm(row) for row in measured) * PHYSICS_TIMESTEP_S
        )
        threshold = runtime_protocol["admissibility"]
        maximum_drift = {
            "selected_obstacle_translation_drift_m": max(
                float(row["selected_obstacle_translation_drift_m"])
                for row in [start] + ledger
            ),
            "selected_obstacle_rotation_drift_rad": max(
                float(row["selected_obstacle_rotation_drift_rad"])
                for row in [start] + ledger
            ),
            "selected_obstacle_surface_drift_m": max(
                float(row["selected_obstacle_surface_drift_m"])
                for row in [start] + ledger
            ),
            "selected_obstacle_max_body_linear_speed_m_s": max(
                float(row["selected_obstacle_max_body_linear_speed_m_s"])
                for row in [start] + ledger
            ),
            "selected_obstacle_max_body_angular_speed_rad_s": max(
                float(row["selected_obstacle_max_body_angular_speed_rad_s"])
                for row in [start] + ledger
            ),
        }
        static_admissible = bool(
            maximum_drift["selected_obstacle_translation_drift_m"]
            <= float(threshold["max_selected_geom_translation_drift_m"])
            and maximum_drift["selected_obstacle_rotation_drift_rad"]
            <= float(threshold["max_selected_geom_rotation_drift_rad"])
            and maximum_drift["selected_obstacle_surface_drift_m"]
            <= float(threshold["max_selected_geom_surface_drift_m"])
            and maximum_drift["selected_obstacle_max_body_linear_speed_m_s"]
            <= float(threshold["max_selected_body_linear_speed_m_s"])
            and maximum_drift["selected_obstacle_max_body_angular_speed_rad_s"]
            <= float(threshold["max_selected_body_angular_speed_rad_s"])
        )
        valid_runtime_h = [
            float(value)
            for row in ledger
            for value in row["ordered_sample_h_m2"]
            if value is not None
        ]
        return {
            "arm_name": arm_name,
            "controller": "JOINT_VELOCITY_100Hz",
            "filter_solve_count": 0 if arm_name == "nominal" else 1,
            "held_command_physics_substeps": len(ledger),
            "commanded_arm_qdot": command.tolist(),
            "normalized_joint_velocity_action_8d": action.tolist(),
            "source_gripper_command": float(gripper_command),
            "hidden_clipping_applied": False,
            "restore": restore,
            "start_boundary": start,
            "physics_substep_ledger": ledger,
            "tracking": {
                "error_ledger_rad_s": errors.tolist(),
                "linf_rad_s": tracking_linf,
                "rmse_rad_s": tracking_rmse,
                "registered_linf_max_rad_s": float(
                    threshold["max_joint_velocity_tracking_linf_rad_s"]
                ),
                "registered_rmse_max_rad_s": float(
                    threshold["max_joint_velocity_tracking_rmse_rad_s"]
                ),
                "within_registered_thresholds": bool(
                    tracking_linf
                    <= float(threshold["max_joint_velocity_tracking_linf_rad_s"])
                    and tracking_rmse
                    <= float(threshold["max_joint_velocity_tracking_rmse_rad_s"])
                ),
            },
            "motion": {
                "measured_arm_tangent_displacement": tangent_displacement[
                    np.asarray(arm_dofs, dtype=np.int64)
                ].tolist(),
                "measured_arm_motion_l2_rad": measured_motion_l2,
                "integrated_measured_joint_speed_l2_rad": integrated_speed_l2,
                "command_norm_rad_s": command_norm,
                "command_integral_over_horizon_rad": command_integral,
                "measured_motion_to_command_integral_ratio": measured_motion_ratio,
                "nonzero_command": bool(command_norm > 0.0),
                "nonzero_measured_joint_motion": bool(measured_motion_l2 > 0.0),
            },
            "maximum_selected_obstacle_drift": maximum_drift,
            "static_field_admissible_for_complete_interval": static_admissible,
            "minimum_h_m2": min(valid_runtime_h) if valid_runtime_h else None,
            "invalid_field_query_count": sum(
                int(row["invalid_field_query_count"]) for row in ledger
            ),
            "minimum_D_sim_m": min(
                float(row["minimum_D_sim_m"]) for row in ledger
            ),
            "any_target_contact": any(
                row["target_contact_present"] for row in ledger
            ),
            "any_robot_to_selected_obstacle_contact": any(
                row["any_robot_to_selected_obstacle_contact_present"]
                for row in ledger
            ),
            "any_shifted_robot_to_selected_obstacle_contact": any(
                row["any_shifted_robot_to_selected_obstacle_contact_present"]
                for row in ledger
            ),
        }
    finally:
        if env is not None:
            env.close()


def _require_exact_paired_arm_start(
    *,
    nominal: Mapping[str, Any],
    psf: Mapping[str, Any],
    boundary_measurement: Mapping[str, Any],
) -> bool:
    """Reject a pair unless controller software and physical starts are exact."""

    restore_fields = (
        "model_topology_sha256",
        "physical_model_sha256",
        "compiled_mjb_sha256",
        "controller",
        "controller_software_state",
        "pid_memory_reset",
    )
    nominal_restore = nominal.get("restore")
    psf_restore = psf.get("restore")
    if not isinstance(nominal_restore, Mapping) or not isinstance(
        psf_restore, Mapping
    ):
        raise OneStepRunnerError("paired arm restore evidence is missing")
    for field in restore_fields:
        if field not in nominal_restore or field not in psf_restore:
            raise OneStepRunnerError(
                "paired arm restore evidence lacks %s" % field
            )
        if _canonical(nominal_restore[field]) != _canonical(psf_restore[field]):
            raise OneStepRunnerError(
                "paired arm restore %s differs" % field
            )

    for arm_name, restore in (("nominal", nominal_restore), ("psf", psf_restore)):
        software = restore["controller_software_state"]
        if not isinstance(software, Mapping):
            raise OneStepRunnerError(
                "%s controller software state is missing" % arm_name
            )
        observed_hash = software.get("sha256")
        if (
            not isinstance(observed_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", observed_hash) is None
        ):
            raise OneStepRunnerError(
                "%s controller software state hash is invalid" % arm_name
            )
        unhashed = {key: value for key, value in software.items() if key != "sha256"}
        if _sha256_bytes(_canonical(unhashed)) != observed_hash:
            raise OneStepRunnerError(
                "%s controller software state hash does not reconstruct" % arm_name
            )
    if (
        nominal_restore["controller_software_state"]["sha256"]
        != psf_restore["controller_software_state"]["sha256"]
    ):
        raise OneStepRunnerError("paired controller software state hashes differ")

    start_drop = {"commanded_arm_qdot", "normalized_joint_velocity_action_8d"}
    starts = (
        ("nominal", nominal.get("start_boundary")),
        ("psf", psf.get("start_boundary")),
        ("source boundary B", boundary_measurement),
    )
    projections: List[bytes] = []
    for label, start in starts:
        if not isinstance(start, Mapping):
            raise OneStepRunnerError("%s start measurement is missing" % label)
        projection = {
            key: value for key, value in start.items() if key not in start_drop
        }
        projections.append(_canonical(projection))
    if not (projections[0] == projections[1] == projections[2]):
        raise OneStepRunnerError(
            "paired arm physical start differs from exact source boundary B"
        )
    return True


def _static_field_admissibility_windows(
    *,
    protocol: Mapping[str, Any],
    boundary: int,
    nominal_any_contact_boundary: Optional[int],
    nominal: Mapping[str, Any],
    psf: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build phase-correct static-field evidence without censoring collisions."""

    scope_protocol = protocol["acceptance"].get(
        "static_field_admissibility_scopes"
    )
    expected_scope_keys = {
        "nominal_with_selected_obstacle_contact",
        "nominal_without_selected_obstacle_contact",
        "poisson_filtered_velocity",
    }
    if (
        not isinstance(scope_protocol, Mapping)
        or set(scope_protocol) != expected_scope_keys
        or any(
            not isinstance(scope_protocol[key], str) or not scope_protocol[key]
            for key in expected_scope_keys
        )
    ):
        raise OneStepRunnerError(
            "static-field admissibility scopes are incomplete or invalid"
        )
    threshold_keys = {
        "translation_m",
        "rotation_rad",
        "surface_m",
        "linear_speed_m_s",
        "angular_speed_rad_s",
    }
    if not isinstance(thresholds, Mapping) or set(thresholds) != threshold_keys:
        raise OneStepRunnerError("static-field thresholds are incomplete")
    numeric_thresholds = {key: float(thresholds[key]) for key in threshold_keys}
    if any(
        not math.isfinite(value) or value < 0.0
        for value in numeric_thresholds.values()
    ):
        raise OneStepRunnerError("static-field thresholds are invalid")

    metric_fields = {
        "translation_m": "selected_obstacle_translation_drift_m",
        "rotation_rad": "selected_obstacle_rotation_drift_rad",
        "surface_m": "selected_obstacle_surface_drift_m",
        "linear_speed_m_s": "selected_obstacle_max_body_linear_speed_m_s",
        "angular_speed_rad_s": "selected_obstacle_max_body_angular_speed_rad_s",
    }

    def ordered_rows(arm: Mapping[str, Any], label: str) -> List[Mapping[str, Any]]:
        start = arm.get("start_boundary")
        ledger = arm.get("physics_substep_ledger")
        if not isinstance(start, Mapping) or not isinstance(ledger, list):
            raise OneStepRunnerError("%s static-field rows are incomplete" % label)
        rows = [start] + ledger
        observed_boundaries = [int(row["physical_boundary"]) for row in rows]
        expected_boundaries = list(range(int(boundary), int(boundary) + 6))
        if observed_boundaries != expected_boundaries:
            raise OneStepRunnerError(
                "%s static-field rows do not cover B through B+5" % label
            )
        return rows

    nominal_rows = ordered_rows(nominal, "nominal")
    psf_rows = ordered_rows(psf, "psf")
    if bool(nominal.get("any_robot_to_selected_obstacle_contact")) != (
        nominal_any_contact_boundary is not None
    ):
        raise OneStepRunnerError(
            "nominal selected-obstacle contact flag differs from phase-correct C_any_nom"
        )
    if nominal_any_contact_boundary is not None and not (
        int(boundary) <= int(nominal_any_contact_boundary) <= int(boundary) + 5
    ):
        raise OneStepRunnerError("phase-correct C_nom is outside the measured interval")

    def window(
        *,
        rows: Sequence[Mapping[str, Any]],
        scope: str,
        cutoff: Optional[int],
    ) -> Dict[str, Any]:
        included = [
            row
            for row in rows
            if cutoff is None or int(row["physical_boundary"]) < int(cutoff)
        ]
        maxima: Dict[str, Optional[float]] = {}
        for key, field in metric_fields.items():
            values = [float(row[field]) for row in included]
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise OneStepRunnerError(
                    "static-field window contains an invalid %s" % key
                )
            maxima[key] = None if not values else max(values)
        admissible = bool(
            included
            and all(
                maxima[key] is not None
                and float(maxima[key]) <= numeric_thresholds[key]
                for key in threshold_keys
            )
        )
        return {
            "scope": scope,
            "cutoff_physical_boundary_exclusive": cutoff,
            "included_physical_boundaries": [
                int(row["physical_boundary"]) for row in included
            ],
            "observed_maxima": maxima,
            "thresholds": dict(numeric_thresholds),
            "admissible": admissible,
        }

    nominal_scope_key = (
        "nominal_with_selected_obstacle_contact"
        if nominal_any_contact_boundary is not None
        else "nominal_without_selected_obstacle_contact"
    )
    nominal_window = window(
        rows=nominal_rows,
        scope=str(scope_protocol[nominal_scope_key]),
        cutoff=nominal_any_contact_boundary,
    )
    psf_window = window(
        rows=psf_rows,
        scope=str(scope_protocol["poisson_filtered_velocity"]),
        cutoff=None,
    )
    return {
        "scope_protocol": dict(scope_protocol),
        "nominal": nominal_window,
        "psf": psf_window,
    }


def _tracking_admissibility_windows(
    *,
    protocol: Mapping[str, Any],
    boundary: int,
    nominal_any_contact_boundary: Optional[int],
    nominal: Mapping[str, Any],
    psf: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate nominal tracking before first contact and PSF tracking in full."""

    acceptance = protocol["acceptance"]
    scope_protocol = acceptance.get("tracking_admissibility_scopes")
    expected_scope_keys = {
        "nominal_with_selected_obstacle_contact",
        "nominal_without_selected_obstacle_contact",
        "poisson_filtered_velocity",
    }
    if (
        not isinstance(scope_protocol, Mapping)
        or set(scope_protocol) != expected_scope_keys
        or any(
            not isinstance(scope_protocol[key], str) or not scope_protocol[key]
            for key in expected_scope_keys
        )
    ):
        raise OneStepRunnerError(
            "joint-velocity tracking scopes are incomplete or invalid"
        )
    thresholds = {
        "linf_rad_s": float(acceptance["joint_velocity_tracking_linf_max_rad_s"]),
        "rmse_rad_s": float(acceptance["joint_velocity_tracking_rmse_max_rad_s"]),
    }
    if any(
        not math.isfinite(value) or value < 0.0 for value in thresholds.values()
    ):
        raise OneStepRunnerError("joint-velocity tracking thresholds are invalid")

    def rows(arm: Mapping[str, Any], label: str) -> List[Dict[str, Any]]:
        ledger = arm.get("physics_substep_ledger")
        tracking = arm.get("tracking")
        error_ledger = (
            tracking.get("error_ledger_rad_s")
            if isinstance(tracking, Mapping)
            else None
        )
        if (
            not isinstance(ledger, list)
            or not isinstance(error_ledger, list)
            or len(ledger) != 5
            or len(error_ledger) != 5
        ):
            raise OneStepRunnerError("%s tracking evidence is incomplete" % label)
        if (
            float(tracking.get("registered_linf_max_rad_s", float("nan")))
            != thresholds["linf_rad_s"]
            or float(tracking.get("registered_rmse_max_rad_s", float("nan")))
            != thresholds["rmse_rad_s"]
        ):
            raise OneStepRunnerError(
                "%s tracking thresholds differ from Stage-13 protocol" % label
            )
        output: List[Dict[str, Any]] = []
        expected_boundaries = list(range(int(boundary) + 1, int(boundary) + 6))
        observed_boundaries = [int(row["physical_boundary"]) for row in ledger]
        if observed_boundaries != expected_boundaries:
            raise OneStepRunnerError(
                "%s tracking rows do not cover B+1 through B+5" % label
            )
        for physical_boundary, error_row in zip(
            observed_boundaries, error_ledger
        ):
            if not isinstance(error_row, list) or len(error_row) != 7:
                raise OneStepRunnerError(
                    "%s tracking error row is not one seven-vector" % label
                )
            errors = [float(value) for value in error_row]
            if any(not math.isfinite(value) for value in errors):
                raise OneStepRunnerError(
                    "%s tracking error row is non-finite" % label
                )
            output.append(
                {"physical_boundary": physical_boundary, "errors": errors}
            )
        return output

    nominal_rows = rows(nominal, "nominal")
    psf_rows = rows(psf, "psf")
    if bool(nominal.get("any_robot_to_selected_obstacle_contact")) != (
        nominal_any_contact_boundary is not None
    ):
        raise OneStepRunnerError(
            "nominal contact flag differs from tracking cutoff C_any_nom"
        )

    def window(
        *,
        input_rows: Sequence[Mapping[str, Any]],
        scope: str,
        cutoff: Optional[int],
    ) -> Dict[str, Any]:
        included = [
            row
            for row in input_rows
            if cutoff is None or int(row["physical_boundary"]) < int(cutoff)
        ]
        flattened = [
            float(value) for row in included for value in row["errors"]
        ]
        linf = None if not flattened else max(abs(value) for value in flattened)
        rmse = (
            None
            if not flattened
            else math.sqrt(
                sum(value * value for value in flattened) / len(flattened)
            )
        )
        admissible = bool(
            flattened
            and linf is not None
            and rmse is not None
            and linf <= thresholds["linf_rad_s"]
            and rmse <= thresholds["rmse_rad_s"]
        )
        return {
            "scope": scope,
            "cutoff_physical_boundary_exclusive": cutoff,
            "included_physical_boundaries": [
                int(row["physical_boundary"]) for row in included
            ],
            "linf_rad_s": linf,
            "rmse_rad_s": rmse,
            "thresholds": dict(thresholds),
            "admissible": admissible,
        }

    nominal_scope_key = (
        "nominal_with_selected_obstacle_contact"
        if nominal_any_contact_boundary is not None
        else "nominal_without_selected_obstacle_contact"
    )
    return {
        "scope_protocol": dict(scope_protocol),
        "nominal": window(
            input_rows=nominal_rows,
            scope=str(scope_protocol[nominal_scope_key]),
            cutoff=nominal_any_contact_boundary,
        ),
        "psf": window(
            input_rows=psf_rows,
            scope=str(scope_protocol["poisson_filtered_velocity"]),
            cutoff=None,
        ),
    }


def _diagnostic_and_classification_inputs(
    *,
    protocol: Mapping[str, Any],
    boundary: int,
    contact_boundary: int,
    boundary_filter: Mapping[str, Any],
    nominal: Mapping[str, Any],
    psf: Mapping[str, Any],
    static_field_thresholds: Mapping[str, Any],
) -> Dict[str, Any]:
    acceptance = protocol["acceptance"]
    trend_index = int(boundary_filter["trend_sample_index"])
    h_B = float(boundary_filter["ordered_sample_h_m2"][trend_index])
    predicted_end = float(
        nominal["physics_substep_ledger"][-1]["first_order_predicted_h_m2"][
            trend_index
        ]
    )
    actual_end_raw = nominal["physics_substep_ledger"][-1][
        "ordered_sample_h_m2"
    ][trend_index]
    actual_end = None if actual_end_raw is None else float(actual_end_raw)
    actual_h = [
        (
            None
            if row["ordered_sample_h_m2"][trend_index] is None
            else float(row["ordered_sample_h_m2"][trend_index])
        )
        for row in nominal["physics_substep_ledger"]
    ]
    psf_actual_end_raw = psf["physics_substep_ledger"][-1][
        "ordered_sample_h_m2"
    ][trend_index]
    psf_actual_end = (
        None if psf_actual_end_raw is None else float(psf_actual_end_raw)
    )
    psf_minus_nominal_end = (
        None
        if actual_end is None or psf_actual_end is None
        else psf_actual_end - actual_end
    )
    prediction_errors = [
        (
            None
            if row["prediction_error_m2"][trend_index] is None
            else float(row["prediction_error_m2"][trend_index])
        )
        for row in nominal["physics_substep_ledger"]
    ]
    nominal_start_d_sim = float(nominal["start_boundary"]["minimum_D_sim_m"])
    nominal_min_d_sim = float(nominal["minimum_D_sim_m"])
    nominal_contact = bool(nominal["any_target_contact"])
    trend_only = bool(
        predicted_end < h_B
        and actual_end is not None
        and actual_end < h_B
        and nominal_min_d_sim < nominal_start_d_sim
        and not nominal_contact
    )
    nominal_target_records = [
        record
        for row in nominal["physics_substep_ledger"]
        for record in row["target_physical_contact_records"]
    ]
    nominal_any_selected_records = [
        record
        for row in nominal["physics_substep_ledger"]
        for record in row[
            "robot_to_selected_obstacle_physical_contact_records"
        ]
    ]
    nominal_contact_boundary = (
        None
        if not nominal_target_records
        else min(int(record["physical_boundary"]) for record in nominal_target_records)
    )
    nominal_any_contact_boundary = (
        None
        if not nominal_any_selected_records
        else min(
            int(record["physical_boundary"])
            for record in nominal_any_selected_records
        )
    )
    nominal_contact_row = (
        None
        if nominal_contact_boundary is None
        else next(
            (
                row
                for row in nominal["physics_substep_ledger"]
                if int(row["physical_boundary"]) == nominal_contact_boundary
            ),
            None,
        )
    )
    safe_contact_row = (
        None
        if nominal_contact_boundary is None
        else next(
            (
                row
                for row in psf["physics_substep_ledger"]
                if int(row["physical_boundary"]) == nominal_contact_boundary
            ),
            None,
        )
    )
    if nominal_contact_boundary is not None and (
        nominal_contact_row is None or safe_contact_row is None
    ):
        raise OneStepRunnerError(
            "paired post-state ledgers lack the phase-correct nominal contact boundary"
        )
    nominal_selected_at_C = (
        []
        if nominal_contact_boundary is None
        else [
            record
            for row in nominal["physics_substep_ledger"]
            for record in row[
                "robot_to_selected_obstacle_physical_contact_records"
            ]
            if int(record["physical_boundary"]) == nominal_contact_boundary
        ]
    )
    safe_selected_at_C = (
        []
        if nominal_contact_boundary is None
        else [
            record
            for row in psf["physics_substep_ledger"]
            for record in row[
                "robot_to_selected_obstacle_physical_contact_records"
            ]
            if int(record["physical_boundary"]) == nominal_contact_boundary
        ]
    )
    nominal_clearance_at_C = (
        None
        if nominal_contact_row is None
        else min(
            [float(nominal_contact_row["post_state_minimum_D_sim_m"])]
            + [float(record["contact_distance_m"]) for record in nominal_selected_at_C]
            + ([0.0] if nominal_selected_at_C else [])
        )
    )
    safe_clearance_at_C = (
        None
        if safe_contact_row is None
        else min(
            [float(safe_contact_row["post_state_minimum_D_sim_m"])]
            + [float(record["contact_distance_m"]) for record in safe_selected_at_C]
            + ([0.0] if safe_selected_at_C else [])
        )
    )
    nominal_command_norm = float(nominal["motion"]["command_norm_rad_s"])
    safe_command_norm = float(psf["motion"]["command_norm_rad_s"])
    safe_to_nominal_ratio = (
        None
        if nominal_command_norm == 0.0
        else safe_command_norm / nominal_command_norm
    )
    static_windows = _static_field_admissibility_windows(
        protocol=protocol,
        boundary=boundary,
        nominal_any_contact_boundary=nominal_any_contact_boundary,
        nominal=nominal,
        psf=psf,
        thresholds=static_field_thresholds,
    )
    tracking_windows = _tracking_admissibility_windows(
        protocol=protocol,
        boundary=boundary,
        nominal_any_contact_boundary=nominal_any_contact_boundary,
        nominal=nominal,
        psf=psf,
    )
    return {
        "trend_sample_index": trend_index,
        "trend_sample_id": int(boundary_filter["trend_sample_id"]),
        "h_at_B_m2": h_B,
        "first_order_predicted_h_after_horizon_m2": predicted_end,
        "actual_h_after_each_physics_substep_m2": actual_h,
        "psf_h_after_horizon_m2": psf_actual_end,
        "psf_minus_nominal_h_after_horizon_m2": psf_minus_nominal_end,
        "psf_warning_sample_h_strictly_greater_than_nominal_at_horizon": bool(
            psf_minus_nominal_end is not None and psf_minus_nominal_end > 0.0
        ),
        "prediction_error_m2_without_acceptance_threshold": prediction_errors,
        "prediction_error_policy": "diagnostic_only_no_post_hoc_tolerance",
        "nominal_D_sim_at_B_m": nominal_start_d_sim,
        "nominal_minimum_D_sim_over_horizon_m": nominal_min_d_sim,
        "trend_only_conditions_all_true_without_target_contact": trend_only,
        "historical_identification_contact_boundary_C": int(contact_boundary),
        "historical_contact_boundary_within_horizon": bool(
            boundary < contact_boundary <= boundary + 5
        ),
        "counterfactual_nominal_first_target_contact_boundary_C_nom": (
            nominal_contact_boundary
        ),
        "counterfactual_nominal_first_any_robot_selected_obstacle_contact_boundary_C_any_nom": (
            nominal_any_contact_boundary
        ),
        "counterfactual_nominal_target_contact_within_horizon": bool(
            nominal_contact_boundary is not None
            and boundary < nominal_contact_boundary <= boundary + 5
        ),
        "nominal_clearance_at_C_nom_m": (
            nominal_clearance_at_C
        ),
        "psf_clearance_at_C_nom_m": (
            safe_clearance_at_C
        ),
        "static_field_admissibility_windows": static_windows,
        "tracking_admissibility_windows": tracking_windows,
        "local_motion_retention": {
            "filter_correction_norm_rad_s": float(
                boundary_filter["correction_norm"]
            ),
            "nominal_command_norm_rad_s": nominal_command_norm,
            "safe_command_norm_rad_s": safe_command_norm,
            "safe_to_nominal_command_norm_ratio": safe_to_nominal_ratio,
            "safe_measured_joint_motion_rad": float(
                psf["motion"]["measured_arm_motion_l2_rad"]
            ),
            "safe_command_integral_over_horizon_rad": float(
                psf["motion"]["command_integral_over_horizon_rad"]
            ),
            "safe_measured_motion_to_command_integral_ratio": psf["motion"][
                "measured_motion_to_command_integral_ratio"
            ],
            "thresholds": {
                key: float(acceptance[key])
                for key in (
                    "minimum_filter_correction_norm_rad_s",
                    "minimum_safe_command_norm_rad_s",
                    "minimum_safe_to_nominal_command_norm_ratio",
                    "minimum_safe_measured_joint_motion_rad",
                    "minimum_safe_measured_motion_to_command_integral_ratio",
                )
            },
            "interpretation": acceptance["local_motion_interpretation"],
        },
    }


def _require_boundary_B_admissible(
    measurement: Mapping[str, Any], runtime_protocol: Mapping[str, Any]
) -> Dict[str, Any]:
    h = measurement.get("ordered_sample_h_m2")
    validity = measurement.get("ordered_field_query_validity_and_reason")
    invalid_count = measurement.get("invalid_field_query_count")
    if (
        not isinstance(h, list)
        or not h
        or not isinstance(validity, list)
        or len(validity) != len(h)
        or isinstance(invalid_count, bool)
        or not isinstance(invalid_count, int)
        or invalid_count != sum(row.get("valid") is not True for row in validity)
        or invalid_count != 0
        or any(value is None for value in h)
        or any(row.get("valid") is not True or row.get("reason") is not None for row in validity)
        or not all(float(value) > 0.0 for value in h)
    ):
        raise OneStepRunnerError(
            "boundary B requires complete valid exact queries and strict h>0"
        )
    if float(measurement.get("minimum_D_sim_m", float("-inf"))) <= 0.0:
        raise OneStepRunnerError("boundary B requires strict positive simulator clearance")
    if measurement.get("any_robot_to_selected_obstacle_contact_present") is not False:
        raise OneStepRunnerError("boundary B already has robot-selected-obstacle contact")
    admissibility = runtime_protocol["admissibility"]
    observed = {
        "translation_m": float(measurement["selected_obstacle_translation_drift_m"]),
        "rotation_rad": float(measurement["selected_obstacle_rotation_drift_rad"]),
        "surface_m": float(measurement["selected_obstacle_surface_drift_m"]),
        "linear_speed_m_s": float(
            measurement["selected_obstacle_max_body_linear_speed_m_s"]
        ),
        "angular_speed_rad_s": float(
            measurement["selected_obstacle_max_body_angular_speed_rad_s"]
        ),
    }
    thresholds = {
        "translation_m": float(
            admissibility["max_selected_geom_translation_drift_m"]
        ),
        "rotation_rad": float(
            admissibility["max_selected_geom_rotation_drift_rad"]
        ),
        "surface_m": float(admissibility["max_selected_geom_surface_drift_m"]),
        "linear_speed_m_s": float(
            admissibility["max_selected_body_linear_speed_m_s"]
        ),
        "angular_speed_rad_s": float(
            admissibility["max_selected_body_angular_speed_rad_s"]
        ),
    }
    exceeded = [key for key in thresholds if observed[key] > thresholds[key]]
    if exceeded:
        raise OneStepRunnerError(
            "boundary B selected-obstacle static assumption is inadmissible at %s"
            % ", ".join(sorted(exceeded))
        )
    return {
        "strict_all_sample_h_positive": True,
        "strict_D_sim_positive": True,
        "zero_robot_selected_obstacle_contact": True,
        "invalid_field_query_count": int(invalid_count),
        "selected_obstacle_static_observed": observed,
        "selected_obstacle_static_thresholds": thresholds,
        "selected_obstacle_static_admissible": True,
        "QP_or_arm_physics_executed_before_this_gate": False,
        "passed": True,
    }


def _require_actionable_warning(
    identification: Mapping[str, Any], contact_boundary: int
) -> Dict[str, Any]:
    assessment = identification["shadow_replay"]["poisson_identification"][
        "contact_prediction_assessment"
    ]
    contact = assessment["first_link56_contact"]
    warning = assessment["primary_registered_warning"]
    if (
        warning.get("signal_kind") != "cbf_lhs_negative"
        or int(warning.get("geom_id", -1)) != int(contact["robot_geom_id"])
    ):
        raise OneStepRunnerError(
            "only a negative CBF lhs warning on the eventual contact geom authorizes Stage 13"
        )
    warning_boundary = int(warning["observation_index"]) + 1
    lead = int(contact_boundary) - warning_boundary
    next_filter = (
        (warning_boundary + PHYSICS_SUBSTEPS_PER_FILTER_UPDATE - 1)
        // PHYSICS_SUBSTEPS_PER_FILTER_UPDATE
    ) * PHYSICS_SUBSTEPS_PER_FILTER_UPDATE
    if (
        lead <= 0
        or lead != int(assessment["lead_physics_substeps"])
        or not math.isclose(
            float(assessment["lead_time_s"]),
            lead * PHYSICS_TIMESTEP_S,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or next_filter >= int(contact_boundary)
    ):
        raise OneStepRunnerError(
            "identification warning has no positive phase-correct scheduled lead"
        )
    return {
        "primary_registered_warning": dict(warning),
        "warning_physical_boundary": warning_boundary,
        "contact_physical_boundary": int(contact_boundary),
        "lead_physics_substeps": lead,
        "lead_time_s": lead * PHYSICS_TIMESTEP_S,
        "next_scheduled_filter_boundary": next_filter,
        "next_filter_boundary_strictly_before_contact": True,
        "invalid_query_warning_used_for_authorization": False,
        "passed": True,
    }


def _require_exact_shadow_construction(
    *,
    identification: Mapping[str, Any],
    construction: Mapping[str, Any],
    settled_integration_state_sha256: str,
    bundle: Any,
    full_robot_sampling: Mapping[str, Any],
    arm_dof_indices: Sequence[int],
    differential_audit_config: Mapping[str, Any],
) -> Dict[str, Any]:
    expected = identification["shadow_replay"].get("construction")
    if not isinstance(expected, Mapping):
        raise OneStepRunnerError("identification construction evidence is absent")
    expected_physical_model = _require_exact_physical_model_contract(
        expected.get("physical_model"), "identification construction"
    )
    fresh_physical_model = _require_exact_physical_model_contract(
        construction.get("physical_model"), "fresh construction"
    )
    exact_fields = (
        "active_obstacle_name",
        "contact_model_authority_sha256",
        "physical_model",
        "robot_root_body_name",
        "robot_root_body_ids",
        "arm_dof_indices",
        "resolved_geometry",
        "field_bundle",
        "full_robot_measurement_sampling",
        "static_field_drift_thresholds",
        "cbf_alpha_gain_per_s",
    )
    differences = [
        field
        for field in exact_fields
        if _canonical(expected.get(field)) != _canonical(construction.get(field))
    ]
    if differences:
        raise OneStepRunnerError(
            "fresh stable construction differs from identification at %s"
            % ", ".join(sorted(differences))
        )
    read_only = construction["complete_integration_state_read_only_audit"]
    expected_read_only = expected["complete_integration_state_read_only_audit"]
    differential = construction["settled_link56_differential_audit"]
    expected_differential = expected["settled_link56_differential_audit"]
    differential_validation = construction[
        "settled_link56_differential_audit_validation"
    ]
    expected_differential_validation = expected[
        "settled_link56_differential_audit_validation"
    ]
    from main.poisson_fullbody.jacobians import (
        DifferentialAuditError,
        validate_protected_sample_differential_audit,
    )

    try:
        fresh_validation = validate_protected_sample_differential_audit(
            differential,
            expected_samples=bundle.protected_samples.samples,
            expected_arm_dof_indices=arm_dof_indices,
            expected_integration_state_sha256=settled_integration_state_sha256,
            expected_differential_audit_config=differential_audit_config,
        )
        shadow_validation = validate_protected_sample_differential_audit(
            expected_differential,
            expected_samples=bundle.protected_samples.samples,
            expected_arm_dof_indices=arm_dof_indices,
            expected_integration_state_sha256=settled_integration_state_sha256,
            expected_differential_audit_config=differential_audit_config,
        )
    except (DifferentialAuditError, KeyError, TypeError, ValueError) as error:
        raise OneStepRunnerError(
            "fresh or shadow protected-sample differential audit is invalid"
        ) from error
    stable_differential_fields = (
        "specification_sha256",
        "binding_sha256",
        "classification_ledger_sha256",
    )
    if (
        read_only.get("before_sha256") != settled_integration_state_sha256
        or read_only.get("after_sha256") != settled_integration_state_sha256
        or read_only.get("exact_array_equal") is not True
        or expected_read_only.get("before_sha256")
        != settled_integration_state_sha256
        or expected_read_only.get("after_sha256")
        != settled_integration_state_sha256
        or expected_read_only.get("exact_array_equal") is not True
        or list(construction.get("arm_dof_indices", []))
        != [int(value) for value in arm_dof_indices]
        or construction["field_bundle"]["hashes"] != asdict(bundle.hashes)
        or construction["field_bundle"]["protected_sample_count"]
        != len(bundle.protected_samples.samples)
        or construction["field_bundle"]["protected_samples"]
        != [sample.to_dict() for sample in bundle.protected_samples.samples]
        or _canonical(construction["full_robot_measurement_sampling"])
        != _canonical(
            {
                key: value
                for key, value in full_robot_sampling.items()
                if key != "sample_ledger"
            }
        )
        or differential_validation != fresh_validation
        or expected_differential_validation != shadow_validation
        or fresh_validation.get("passed") is not True
        or shadow_validation.get("passed") is not True
        or any(
            differential.get(field) != expected_differential.get(field)
            for field in stable_differential_fields
        )
    ):
        raise OneStepRunnerError("fresh shadow construction binding is incomplete")
    return {
        "contact_model_authority_sha256": construction[
            "contact_model_authority_sha256"
        ],
        "physical_model": dict(fresh_physical_model),
        "resolved_geometry": construction["resolved_geometry"],
        "field_bundle_hashes": construction["field_bundle"]["hashes"],
        "field_bundle_sha256": bundle.hashes.bundle_sha256,
        "full_robot_measurement_sampling": dict(full_robot_sampling),
        "ordered_protected_sample_ledger": construction["field_bundle"][
            "protected_samples"
        ],
        "ordered_protected_sample_ledger_sha256": (
            bundle.hashes.protected_samples_sha256
        ),
        "protected_sample_count": len(bundle.protected_samples.samples),
        "protected_surface_components": construction["field_bundle"][
            "surface_components"
        ],
        "protected_sampling_epsilon_m": construction["field_bundle"][
            "protected_sampling_epsilon_m"
        ],
        "protected_sampling_maximum_surface_cover_radius_m": construction[
            "field_bundle"
        ]["protected_sampling_maximum_surface_cover_radius_m"],
        "protected_sampling_coverage_semantics": construction["field_bundle"][
            "protected_sampling_coverage_semantics"
        ],
        "arm_dof_indices": [int(value) for value in arm_dof_indices],
        "settled_mjstate_integration_sha256": settled_integration_state_sha256,
        "settled_link56_differential_audit_binding_sha256": differential[
            "binding_sha256"
        ],
        "settled_link56_differential_audit_classification_ledger_sha256": (
            differential["classification_ledger_sha256"]
        ),
        "settled_link56_differential_audit_validation": differential_validation,
        "fresh_stable_construction_binding_equals_identification": True,
    }


def _classify_complete_pair(
    *,
    protocol: Mapping[str, Any],
    boundary_admissibility: Mapping[str, Any],
    boundary_filter: Mapping[str, Any],
    nominal: Mapping[str, Any],
    psf: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    exact_paired_start: bool,
) -> Dict[str, Any]:
    acceptance = protocol["acceptance"]
    safe_residual_minimum = min(
        float(value) for value in boundary_filter["safe_CBF_residuals_m2_per_s"]
    )
    nominal_contact_boundary = diagnostics[
        "counterfactual_nominal_first_target_contact_boundary_C_nom"
    ]
    nominal_any_contact_boundary = diagnostics[
        "counterfactual_nominal_first_any_robot_selected_obstacle_contact_boundary_C_any_nom"
    ]
    nominal_clearance = diagnostics["nominal_clearance_at_C_nom_m"]
    safe_clearance = diagnostics["psf_clearance_at_C_nom_m"]
    motion_retention = diagnostics["local_motion_retention"]
    motion_thresholds = motion_retention["thresholds"]
    local_feasibility_conditions = {
        "nominal_velocity_within_registered_controller_envelope": True,
        "no_hidden_clipping": bool(
            nominal["hidden_clipping_applied"] is False
            and psf["hidden_clipping_applied"] is False
        ),
        "complete_exposure": bool(
            nominal["held_command_physics_substeps"] == 5
            and psf["held_command_physics_substeps"] == 5
        ),
        "boundary_B_preflight_admissible": bool(
            boundary_admissibility.get("passed") is True
        ),
        "exact_paired_start": bool(exact_paired_start is True),
        "qp_valid": bool(
            psf["filter_solve_count"] == 1
            and boundary_filter["one_CBF_row_per_exact_bound_sample"]["no_slack"]
            is True
            and boundary_filter["QP_status"] in ("solved", "solved inaccurate")
            and safe_residual_minimum
            >= float(acceptance["qp_safe_residual_minimum"])
        ),
        "both_tracking_valid": bool(
            diagnostics["tracking_admissibility_windows"]["nominal"][
                "admissible"
            ]
            and diagnostics["tracking_admissibility_windows"]["psf"][
                "admissible"
            ]
        ),
        "static_field_admissible": bool(
            diagnostics["static_field_admissibility_windows"]["nominal"][
                "admissible"
            ]
            and diagnostics["static_field_admissibility_windows"]["psf"][
                "admissible"
            ]
        ),
        "filter_correction_above_registered_minimum": bool(
            float(motion_retention["filter_correction_norm_rad_s"])
            >= float(motion_thresholds["minimum_filter_correction_norm_rad_s"])
        ),
        "safe_command_above_registered_minimum": bool(
            float(motion_retention["safe_command_norm_rad_s"])
            >= float(motion_thresholds["minimum_safe_command_norm_rad_s"])
        ),
        "safe_command_retains_registered_nominal_norm_fraction": bool(
            motion_retention["safe_to_nominal_command_norm_ratio"] is not None
            and float(motion_retention["safe_to_nominal_command_norm_ratio"])
            >= float(
                motion_thresholds["minimum_safe_to_nominal_command_norm_ratio"]
            )
        ),
        "safe_measured_motion_above_registered_minimum": bool(
            float(motion_retention["safe_measured_joint_motion_rad"])
            >= float(
                motion_thresholds["minimum_safe_measured_joint_motion_rad"]
            )
        ),
        "safe_measured_motion_retains_registered_command_integral_fraction": bool(
            motion_retention[
                "safe_measured_motion_to_command_integral_ratio"
            ]
            is not None
            and float(
                motion_retention[
                    "safe_measured_motion_to_command_integral_ratio"
                ]
            )
            >= float(
                motion_thresholds[
                    "minimum_safe_measured_motion_to_command_integral_ratio"
                ]
            )
        ),
        "safe_all_queries_valid": bool(
            psf["invalid_field_query_count"]
            <= int(acceptance["maximum_safe_arm_invalid_field_queries"])
        ),
        "safe_minimum_h_nonnegative": bool(
            psf["minimum_h_m2"] is not None and float(psf["minimum_h_m2"]) >= 0.0
        ),
        "safe_minimum_D_sim_strictly_positive": bool(
            all(
                float(row["minimum_D_sim_m"]) > 0.0
                for row in psf["physics_substep_ledger"]
            )
        ),
        "safe_all_robot_selected_obstacle_contact_absent": not bool(
            psf["any_robot_to_selected_obstacle_contact"]
        ),
    }
    target_valid = (
        diagnostics["actual_h_after_each_physics_substep_m2"][-1] is not None
    )
    psf_target_valid = diagnostics["psf_h_after_horizon_m2"] is not None
    predicted_decreased = bool(
        float(diagnostics["first_order_predicted_h_after_horizon_m2"])
        < float(diagnostics["h_at_B_m2"])
    )
    actual_decreased = bool(
        target_valid
        and float(diagnostics["actual_h_after_each_physics_substep_m2"][-1])
        < float(diagnostics["h_at_B_m2"])
    )
    dsim_decreased = bool(
        float(diagnostics["nominal_minimum_D_sim_over_horizon_m"])
        < float(diagnostics["nominal_D_sim_at_B_m"])
    )
    unsafe_trend_conditions = {
        "nominal_target_contact_absent": not bool(nominal["any_target_contact"]),
        "warning_sample_valid_in_both_arms_at_horizon": bool(
            target_valid and psf_target_valid
        ),
        "warning_sample_first_order_predicted_h_decreased": predicted_decreased,
        "nominal_warning_sample_actual_h_decreased": actual_decreased,
        "nominal_D_sim_decreased": dsim_decreased,
        "psf_warning_sample_h_strictly_greater_than_nominal_at_horizon": bool(
            diagnostics[
                "psf_warning_sample_h_strictly_greater_than_nominal_at_horizon"
            ]
        ),
    }
    contact_prevention_conditions = {
        "nominal_target_contact_within_horizon": bool(
            nominal["any_target_contact"]
            and diagnostics[
                "counterfactual_nominal_target_contact_within_horizon"
            ]
        ),
        "nominal_first_selected_obstacle_contact_matches_target_boundary": bool(
            nominal_any_contact_boundary is not None
            and nominal_contact_boundary is not None
            and int(nominal_any_contact_boundary) == int(nominal_contact_boundary)
        ),
        "safe_clearance_greater_at_nominal_target_contact_boundary": bool(
            nominal_contact_boundary is not None
            and nominal_clearance is not None
            and safe_clearance is not None
            and float(safe_clearance) > float(nominal_clearance)
        ),
    }
    if nominal["any_target_contact"]:
        reference_kind = "target_contact_within_horizon"
        reference_conditions = contact_prevention_conditions
    else:
        reference_kind = "unsafe_trend_without_target_contact"
        reference_conditions = unsafe_trend_conditions
    reference_passed = all(reference_conditions.values())
    local_passed = all(local_feasibility_conditions.values())
    stage_passed = bool(local_passed and reference_passed)
    contact_prevention_observed = bool(
        stage_passed and reference_kind == "target_contact_within_horizon"
    )
    if contact_prevention_observed:
        label = str(acceptance["strong_result_label"])
    elif stage_passed:
        label = str(acceptance["trend_only_label"])
    else:
        label = str(acceptance["other_outcome_label"])
    typed_reasons = sorted(
        {
            key
            for key, value in local_feasibility_conditions.items()
            if value is not True
        }
        | {
            key
            for key, value in reference_conditions.items()
            if value is not True
        }
    )
    return {
        "label": label,
        "stage_13_passed": stage_passed,
        "contact_prevention_observed": contact_prevention_observed,
        "claim_scope": (
            "one_100hz_interval_local_geometry_field_jacobian_qp_and_tracking_"
            "feasibility_not_task_success_or_full_episode_safety"
        ),
        "local_feasibility_conditions": local_feasibility_conditions,
        "nominal_reference": {
            "kind": reference_kind,
            "conditions": reference_conditions,
            "passed": reference_passed,
        },
        "typed_reasons": typed_reasons,
    }


def _preflight_inadmissible_outcome(
    *,
    bound_sample_count: int,
    source_prefix_callback_count: int,
    gripper_evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the complete no-QP terminal union for an out-of-envelope nominal."""

    gripper_keys = {
        "source_action_index",
        "source_action_7d",
        "source_action_sha256",
        "exact_gripper_value",
        "exact_gripper_value_sha256",
    }
    if set(gripper_evidence) != gripper_keys:
        raise OneStepRunnerError(
            "preflight-inadmissible terminal requires complete gripper evidence"
        )
    if (
        isinstance(bound_sample_count, bool)
        or not isinstance(bound_sample_count, int)
        or bound_sample_count <= 0
        or isinstance(source_prefix_callback_count, bool)
        or not isinstance(source_prefix_callback_count, int)
        or source_prefix_callback_count <= 0
    ):
        raise OneStepRunnerError(
            "preflight-inadmissible terminal requires complete source evidence"
        )
    return {
        "execution": {
            "outcome_kind": "preflight_inadmissible_nominal_velocity",
            "source_prefix_complete": True,
            "qp_executed": False,
            "paired_joint_velocity_physics_executed": False,
            "no_hidden_clipping": True,
        },
        "boundary_B_filter": None,
        "arms": [],
        "diagnostics": None,
        "boundary_B_measurement": None,
        "boundary_B_admissibility": None,
        "gripper_evidence": dict(gripper_evidence),
        "counts": {
            "bound_sample_count": int(bound_sample_count),
            "source_prefix_callback_count": int(source_prefix_callback_count),
            "nominal_physics_substep_count": 0,
            "psf_physics_substep_count": 0,
            "qp_solve_count": 0,
            "nominal_invalid_field_query_count": 0,
            "psf_invalid_field_query_count": 0,
            "nominal_target_contact_record_count": 0,
            "psf_robot_selected_obstacle_contact_record_count": 0,
        },
        "classification": {
            "label": (
                "inadmissible_nominal_velocity_outside_registered_"
                "controller_envelope"
            ),
            "stage_13_passed": False,
            "contact_prevention_observed": False,
            "claim_scope": (
                "registered_controller_envelope_inadmissibility_only_"
                "not_poisson_qp_or_contact_prevention"
            ),
            "local_feasibility_conditions": {
                "nominal_velocity_within_registered_controller_envelope": False,
                "no_hidden_clipping": True,
            },
            "nominal_reference": {
                "kind": "not_evaluated_preflight_inadmissible",
                "conditions": {},
                "passed": False,
            },
            "typed_reasons": [
                "nominal_velocity_outside_registered_controller_envelope"
            ],
        },
    }


def _artifact_identity(path: Path, value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "file_sha256": _file_sha256(path),
        "payload_sha256": str(value["result_payload_sha256"]),
        "schema_version": str(value["schema_version"]),
        "status": str(value["status"]),
    }


def _section_hashes(payload: Dict[str, Any]) -> None:
    payload["protocol_binding_sha256"] = _sha256_bytes(
        _canonical(payload["protocol_binding"])
    )
    payload["authority_sha256"] = _sha256_bytes(_canonical(payload["authority"]))
    payload["provenance_sha256"] = _sha256_bytes(_canonical(payload["provenance"]))
    payload["nominal_velocity_estimate_sha256"] = _sha256_bytes(
        _canonical(payload["nominal_velocity_estimate"])
    )
    payload["boundary_B_filter_sha256"] = _sha256_bytes(
        _canonical(payload["boundary_B_filter"])
    )
    payload["arm_ledger_sha256"] = _sha256_bytes(_canonical(payload["arms"]))
    payload["classification_ledger_sha256"] = _sha256_bytes(
        _canonical(payload["classification"])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/vlsa_poisson_link56_aegis_car_109.v1.jsonl"),
    )
    parser.add_argument(
        "--selection-protocol",
        type=Path,
        default=Path("configs/vlsa_poisson_link56_feasibility.v1.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/vlsa_poisson_one_step_counterfactual.v2.json"),
    )
    parser.add_argument("--historical-result-root", required=True, type=Path)
    parser.add_argument("--numeric-validation-result", required=True, type=Path)
    parser.add_argument("--parity-result", required=True, type=Path)
    parser.add_argument("--shadow-identification-result", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    arguments = parser.parse_args()

    root = arguments.repo_root.resolve()
    os.chdir(str(root))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    for import_root in (root / "main", root / "safelibero"):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

    output: Optional[Path] = None
    phase = "argument_and_output_validation"
    source_git: Optional[Mapping[str, Any]] = None
    allocation: Optional[Mapping[str, Any]] = None
    started = time.time()
    try:
        if arguments.case_id != DEFAULT_CASE_ID:
            raise OneStepRunnerError(
                "Stage 13 is registered only for %s" % DEFAULT_CASE_ID
            )
        output = _validated_new_output(arguments.output_root, arguments.run_id)

        from main.poisson_fullbody.contracts import (
            attach_payload_hash,
            load_hashed_json,
            publish_hashed_json,
            sha256_file,
        )
        from main.poisson_fullbody.controller_bridge import model_physics_contract
        from main.poisson_fullbody.field_bundle import build_static_field_bundle
        from main.poisson_fullbody.measurement import (
            clone_forwarded_state,
            resolve_collision_geom_sets,
        )
        from main.poisson_fullbody.one_step_counterfactual import (
            validate_one_step_counterfactual_result,
        )
        from main.poisson_fullbody.shadow_replay import load_historical_action_replay
        from main.poisson_fullbody.robot_samples import validate_rigid_roundtrip
        from main.poisson_fullbody.surface_sampling import (
            build_robot_collision_samples,
            validate_robot_sample_evidence,
        )
        from scripts.run_poisson_active_canary import (
            _allocation,
            _bound_historical_result_path,
            _git,
            _load_case,
            _require_identification_prerequisite,
            _require_numeric_prerequisite,
            _validate_replay_against_manifest,
        )
        from scripts.run_poisson_shadow_identification import (
            _load_bound_runtime_protocol,
            _prepare_shadow_runtime,
            _require_upstream_parity,
        )
        import main.evaluate_safelibero_aegis as evaluator
        import numpy as np

        phase = "clean_source_and_allocation_preflight"
        source_git = _git(root)
        source_git = dict(source_git, status_short=[])
        allocation = _allocation()

        phase = "protocol_manifest_and_prerequisite_validation"
        protocol_path, stage_protocol, protocol_raw_sha256 = _protocol_paths(
            root, arguments.protocol
        )
        protocol_semantic_sha256 = _sha256_bytes(_canonical(stage_protocol))
        manifest_path = (
            arguments.manifest.resolve()
            if arguments.manifest.is_absolute()
            else (root / arguments.manifest).resolve()
        )
        selection_path = (
            arguments.selection_protocol.resolve()
            if arguments.selection_protocol.is_absolute()
            else (root / arguments.selection_protocol).resolve()
        )
        manifest_path = _require_regular_file(manifest_path, "manifest")
        selection_path = _require_regular_file(selection_path, "selection protocol")
        case, case_row_sha256 = _load_case(manifest_path, arguments.case_id)
        runtime_protocol, runtime_hashes, selection, runtime_protocol_path = (
            _load_bound_runtime_protocol(
                root=root,
                case=case,
                selection_config_path=selection_path,
            )
        )
        frozen = stage_protocol["prerequisites"]
        registered_selection_path = (
            root / frozen["selection_protocol_relative_path"]
        ).resolve()
        if (
            selection_path != registered_selection_path
            or selection.get("schema_version")
            != frozen["selection_schema_version"]
            or selection.get("protocol_id") != frozen["selection_protocol_id"]
            or _file_sha256(selection_path)
            != frozen["selection_raw_file_sha256"]
        ):
            raise OneStepRunnerError("one-step protocol selection binding differs")
        if (
            _file_sha256(runtime_protocol_path)
            != frozen["runtime_raw_file_sha256"]
            or runtime_hashes.protocol_sha256
            != frozen["runtime_semantic_protocol_sha256"]
            or runtime_hashes.parameter_block_sha256
            != frozen["runtime_parameter_block_sha256"]
            or runtime_protocol["schema_version"] != frozen["runtime_schema_version"]
            or runtime_protocol["protocol_id"] != frozen["runtime_protocol_id"]
        ):
            raise OneStepRunnerError("one-step protocol runtime binding differs")
        if (
            stage_protocol["case"]["selected_obstacle_name"]
            != case["active_obstacle_name"]
            or stage_protocol["case"]["protected_robot_body_names"]
            != runtime_protocol["claim_scope"]["protected_robot_bodies"]
        ):
            raise OneStepRunnerError("one-step case/geometry binding differs")

        historical_root = arguments.historical_result_root.resolve()
        historical_path = _bound_historical_result_path(historical_root, case)
        replay = load_historical_action_replay(
            historical_path, expected_case_id=case["case_id"]
        )
        _validate_replay_against_manifest(replay, case)
        if len(replay.actions) != 237:
            raise OneStepRunnerError("registered canary must retain exactly 237 actions")
        manifest_sha256 = _file_sha256(manifest_path)
        numeric_path = _require_regular_file(
            arguments.numeric_validation_result.resolve(), "numeric prerequisite"
        )
        parity_path = _require_regular_file(
            arguments.parity_result.resolve(), "parity prerequisite"
        )
        identification_path = _require_regular_file(
            arguments.shadow_identification_result.resolve(),
            "identification prerequisite",
        )
        numeric = _require_numeric_prerequisite(
            numeric_path, source_commit=source_git["commit"]
        )
        parity = _require_upstream_parity(
            parity_path,
            case_id=case["case_id"],
            source_commit=source_git["commit"],
            historical_payload_sha256=replay.result_payload_sha256,
            manifest_sha256=manifest_sha256,
            manifest_row_sha256=case_row_sha256,
            expected_callback_count=25 * len(replay.actions),
            historical_provenance=replay.provenance(),
            expected_action_state_sha256_ledger=[
                step.simulator_state_sha256 for step in replay.steps
            ],
        )
        identification = _require_identification_prerequisite(
            identification_path,
            case=case,
            case_row_hash=case_row_sha256,
            source_commit=source_git["commit"],
            manifest_sha256=manifest_sha256,
            selection_sha256=_file_sha256(selection_path),
            runtime_protocol_raw_sha256=_file_sha256(runtime_protocol_path),
            runtime_protocol_semantic_sha256=runtime_hashes.protocol_sha256,
            runtime_parameter_block_sha256=runtime_hashes.parameter_block_sha256,
            alpha_gain_per_s=float(runtime_protocol["cbf"]["alpha_gain_per_s"]),
            static_drift_thresholds={
                "translation_m": float(
                    runtime_protocol["admissibility"][
                        "max_selected_geom_translation_drift_m"
                    ]
                ),
                "rotation_rad": float(
                    runtime_protocol["admissibility"][
                        "max_selected_geom_rotation_drift_rad"
                    ]
                ),
                "surface_m": float(
                    runtime_protocol["admissibility"][
                        "max_selected_geom_surface_drift_m"
                    ]
                ),
            },
            differential_audit_config=runtime_protocol["differential_audit"],
            replay=replay,
            parity=parity,
        )
        target_contact, contact_boundary, boundary = _contact_from_identification(
            identification, stage_protocol
        )
        warning_authorization = _require_actionable_warning(
            identification, contact_boundary
        )
        if boundary != int(
            warning_authorization["next_scheduled_filter_boundary"]
        ):
            raise OneStepRunnerError(
                "counterfactual B is not the first scheduled filter boundary after warning"
            )

        protocol_binding = {
            "raw_file_sha256": protocol_raw_sha256,
            "semantic_sha256": protocol_semantic_sha256,
            "schema_version": stage_protocol["schema_version"],
            "protocol_id": stage_protocol["protocol_id"],
            "result_schema_version": EXPECTED_RESULT_SCHEMA,
        }
        numeric_identity = _artifact_identity(numeric_path, numeric)
        parity_identity = _artifact_identity(parity_path, parity)
        identification_identity = _artifact_identity(
            identification_path, identification
        )
        authority = {
            "expected_code_commit": source_git["commit"],
            "expected_run_id": arguments.run_id,
            "expected_slurm_job_id": str(allocation["slurm_job_id"]),
            "expected_host": str(allocation["host_name"]),
            "expected_device": str(allocation["device"]["model"]),
            "protocol": {
                "path": str(protocol_path),
                "file_sha256": protocol_raw_sha256,
                "semantic_sha256": protocol_semantic_sha256,
                "schema_version": stage_protocol["schema_version"],
                "protocol_id": stage_protocol["protocol_id"],
            },
            "numeric_prerequisite": numeric_identity,
            "parity_prerequisite": parity_identity,
            "identification_prerequisite": identification_identity,
        }

        phase = "fresh_settled_field_and_shadow_construction"
        runtime = evaluator._runtime_imports(include_aegis=False)
        source_env = None
        try:
            (
                source_env,
                _,
                _,
                _,
                _,
                _,
                _,
                construction,
            ) = _prepare_shadow_runtime(
                evaluator=evaluator,
                runtime=runtime,
                case=case,
                replay=replay,
                protocol=runtime_protocol,
                protocol_hashes=runtime_hashes,
            )
            physical_model = _require_exact_physical_model_contract(
                model_physics_contract(source_env.sim), "fresh compiled model"
            )
            if (
                _canonical(construction.get("physical_model"))
                != _canonical(physical_model)
                or _canonical(construction.get("physical_model"))
                != _canonical(
                    identification["shadow_replay"]["construction"].get(
                        "physical_model"
                    )
                )
            ):
                raise OneStepRunnerError(
                    "fresh compiled physical model differs from identification construction"
                )
            joint_velocity_controller = _capture_joint_velocity_controller_authority(
                evaluator=evaluator,
                runtime=runtime,
                case=case,
                expected_physical_model=physical_model,
            )
            expected_joint_velocity_controller = stage_protocol.get(
                "joint_velocity_controller_authority", {}
            ).get("expected_restore_controller_contract")
            if _canonical(joint_velocity_controller) != _canonical(
                expected_joint_velocity_controller
            ):
                raise OneStepRunnerError(
                    "live joint-velocity controller differs from frozen H100 authority"
                )
            settled_capture = _capture_state(source_env, 0)
            raw_model, raw_data = _raw_model_data(source_env.sim)
            resolved_record = construction["resolved_geometry"]
            resolved = resolve_collision_geom_sets(
                raw_model,
                robot_root_body_ids=resolved_record["robot_root_body_ids"],
                obstacle_root_body_ids=resolved_record["obstacle_root_body_ids"],
                link56_body_ids=resolved_record["link56_body_ids"],
            )
            if (
                target_contact["robot_body_name"]
                not in stage_protocol["case"][
                    "required_nominal_contact_body_names"
                ]
                or int(target_contact["robot_body_id"])
                not in set(int(value) for value in resolved.link56_body_ids)
                or int(target_contact["robot_geom_id"])
                not in set(int(value) for value in resolved.link56_geom_ids)
                or int(target_contact["obstacle_body_id"])
                not in set(int(value) for value in resolved.obstacle_body_ids)
                or int(target_contact["obstacle_geom_id"])
                not in set(int(value) for value in resolved.obstacle_geom_ids)
                or case["active_obstacle_name"]
                != stage_protocol["case"]["selected_obstacle_name"]
            ):
                raise OneStepRunnerError(
                    "identification target is outside the protocol link5/6-selected-obstacle authority"
                )
            bundle = build_static_field_bundle(
                raw_model,
                raw_data,
                resolved=resolved,
                protocol=runtime_protocol,
                protocol_hashes=runtime_hashes,
            )
            forwarded = clone_forwarded_state(raw_model, raw_data)
            full_samples = build_robot_collision_samples(
                source_env.sim.model,
                forwarded,
                geom_ids=resolved.robot_geom_ids,
                epsilon_m=float(runtime_protocol["coverage"]["epsilon_m"]),
            )
            full_sample_ledger = [
                sample.to_dict() for sample in full_samples.samples
            ]
            full_sample_ledger_sha256 = _sha256_bytes(
                _canonical(full_sample_ledger)
            )
            if full_sample_ledger_sha256 != full_samples.sample_ledger_sha256:
                raise OneStepRunnerError(
                    "fresh full-robot sample ledger hash does not reconstruct"
                )
            full_robot_roundtrip = validate_rigid_roundtrip(
                full_samples.samples, forwarded
            )
            full_robot_sampling = {
                "sample_count": len(full_samples.samples),
                "sample_ledger": full_sample_ledger,
                "sample_ledger_sha256": full_sample_ledger_sha256,
                "geom_records": list(full_samples.geom_records),
                "epsilon_m": float(full_samples.epsilon_m),
                "maximum_surface_cover_radius_m": float(
                    full_samples.maximum_surface_cover_radius_m
                ),
                "coverage_semantics": full_samples.coverage_semantics,
                "rigid_roundtrip": full_robot_roundtrip,
            }
            try:
                full_robot_type_counts = validate_robot_sample_evidence(
                    full_robot_sampling,
                    resolved_geom_ids=resolved.robot_geom_ids,
                    resolved_geom_names=resolved.robot_geom_names,
                    resolved_body_ids=resolved.robot_body_ids,
                    roundtrip_field="rigid_roundtrip",
                )
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise OneStepRunnerError(
                    "fresh full-robot surface-sampling evidence is invalid"
                ) from error
            if (
                full_robot_type_counts != {"mesh": 11, "box": 4, "cylinder": 1}
                or full_robot_roundtrip.get("passed") is not True
                or _canonical(
                    {
                        key: value
                        for key, value in full_robot_sampling.items()
                        if key != "sample_ledger"
                    }
                )
                != _canonical(construction["full_robot_measurement_sampling"])
            ):
                raise OneStepRunnerError(
                    "fresh full-robot measurement sampling differs from identification"
                )
            arm_dof_indices = tuple(
                int(value) for value in source_env.robots[0]._ref_joint_vel_indexes
            )
            arm_qpos_indices = tuple(
                int(value) for value in source_env.robots[0]._ref_joint_pos_indexes
            )
            expected_arm_qpos_indices_raw = stage_protocol[
                "nominal_velocity_estimator"
            ].get("expected_arm_qpos_indices")
            if (
                not isinstance(expected_arm_qpos_indices_raw, list)
                or len(expected_arm_qpos_indices_raw) != 7
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in expected_arm_qpos_indices_raw
                )
                or len(set(expected_arm_qpos_indices_raw)) != 7
            ):
                raise OneStepRunnerError(
                    "frozen expected arm qpos index authority is invalid"
                )
            expected_arm_qpos_indices = tuple(expected_arm_qpos_indices_raw)
            if arm_qpos_indices != expected_arm_qpos_indices:
                raise OneStepRunnerError(
                    "source robot arm qpos registration differs from frozen protocol"
                )
            shadow_binding = _require_exact_shadow_construction(
                identification=identification,
                construction=construction,
                settled_integration_state_sha256=settled_capture[
                    "integration_state_sha256"
                ],
                bundle=bundle,
                full_robot_sampling=full_robot_sampling,
                arm_dof_indices=arm_dof_indices,
                differential_audit_config=runtime_protocol[
                    "differential_audit"
                ],
            )

            phase = "single_exact_OSC_prefix_and_nominal_derivation"
            exact_prefix = _capture_exact_prefix(
                env=source_env,
                evaluator=evaluator,
                runtime=runtime,
                replay=replay,
                parity=parity,
                identification=identification,
                boundary=boundary,
            )
            source_restore = _restore_source_to_boundary(
                source_env, exact_prefix["state_at_B"]
            )
            nominal_velocity = _derive_nominal_velocity(
                source_env,
                exact_prefix["state_at_B"],
                exact_prefix["state_at_B_plus_5"],
                arm_dof_indices,
                stage_protocol,
            )
            if list(expected_arm_qpos_indices) != nominal_velocity["arm_qpos_indices"]:
                raise OneStepRunnerError(
                    "source robot arm qpos registration differs from compiled hinge mapping"
                )
            source_action_index = boundary // PHYSICS_SUBSTEPS_PER_OSC_ACTION
            source_action = np.asarray(
                replay.actions[source_action_index], dtype=np.float64
            )
            if source_action.shape != (7,) or not np.all(np.isfinite(source_action)):
                raise OneStepRunnerError("source action/gripper evidence is invalid")
            gripper_value = float(source_action[6])
            gripper_evidence = {
                "source_action_index": int(source_action_index),
                "source_action_7d": source_action.tolist(),
                "source_action_sha256": _array_sha256(source_action),
                "exact_gripper_value": gripper_value,
                "exact_gripper_value_sha256": _sha256_bytes(
                    _canonical(gripper_value)
                ),
            }
            if nominal_velocity["arm_velocity_within_registered_bounds"] is not True:
                phase = "terminal_nominal_velocity_controller_envelope_preflight"
                raise _NominalVelocityPreflightInadmissible()
            from main.poisson_fullbody.joint_velocity_adapter import (
                normalized_joint_velocity_action,
            )

            qdot_nominal = np.asarray(
                nominal_velocity["qdot_nom_arm_slice"], dtype=np.float64
            )
            nominal_action = normalized_joint_velocity_action(
                qdot_nominal, gripper_value
            )
            boundary_measurement = _state_measurement(
                sim=source_env.sim,
                bundle=bundle,
                resolved=resolved,
                full_samples=full_samples,
                arm_dof_indices=arm_dof_indices,
                target_contact=target_contact,
                physical_boundary=boundary,
                commanded_arm_qdot=qdot_nominal,
                normalized_action=nominal_action,
                include_live_solver=False,
            )
            boundary_admissibility = _require_boundary_B_admissible(
                boundary_measurement, runtime_protocol
            )

            phase = "single_boundary_B_QP_solve"
            boundary_filter = _boundary_filter_evidence(
                source_env=source_env,
                bundle=bundle,
                arm_dof_indices=arm_dof_indices,
                arm_qpos_indices=arm_qpos_indices,
                qdot_nominal=qdot_nominal,
                target_contact=target_contact,
                runtime_protocol=runtime_protocol,
                stage_protocol=stage_protocol,
                warning_authorization=warning_authorization,
            )

            phase = "paired_five_substep_joint_velocity_physics"
            nominal_arm = _run_counterfactual_arm(
                arm_name="nominal",
                evaluator=evaluator,
                runtime=runtime,
                case=case,
                replay=replay,
                source_env=source_env,
                state_at_B=exact_prefix["state_at_B"],
                boundary=boundary,
                command_arm_qdot=qdot_nominal,
                gripper_command=gripper_value,
                bundle=bundle,
                resolved=resolved,
                full_samples=full_samples,
                target_contact=target_contact,
                runtime_protocol=runtime_protocol,
                boundary_filter=boundary_filter,
                expected_arm_qpos_indices=expected_arm_qpos_indices,
            )
            psf_arm = _run_counterfactual_arm(
                arm_name="psf",
                evaluator=evaluator,
                runtime=runtime,
                case=case,
                replay=replay,
                source_env=source_env,
                state_at_B=exact_prefix["state_at_B"],
                boundary=boundary,
                command_arm_qdot=boundary_filter["qdot_safe"],
                gripper_command=gripper_value,
                bundle=bundle,
                resolved=resolved,
                full_samples=full_samples,
                target_contact=target_contact,
                runtime_protocol=runtime_protocol,
                boundary_filter=boundary_filter,
                expected_arm_qpos_indices=expected_arm_qpos_indices,
            )
            arms = [nominal_arm, psf_arm]
            exact_paired_start = _require_exact_paired_arm_start(
                nominal=nominal_arm,
                psf=psf_arm,
                boundary_measurement=boundary_measurement,
            )
            diagnostics = _diagnostic_and_classification_inputs(
                protocol=stage_protocol,
                boundary=boundary,
                contact_boundary=contact_boundary,
                boundary_filter=boundary_filter,
                nominal=nominal_arm,
                psf=psf_arm,
                static_field_thresholds=boundary_admissibility[
                    "selected_obstacle_static_thresholds"
                ],
            )
            classification = _classify_complete_pair(
                protocol=stage_protocol,
                boundary_admissibility=boundary_admissibility,
                boundary_filter=boundary_filter,
                nominal=nominal_arm,
                psf=psf_arm,
                diagnostics=diagnostics,
                exact_paired_start=exact_paired_start,
            )
            execution = {
                "outcome_kind": "paired_counterfactual_complete",
                "source_prefix_complete": True,
                "qp_executed": True,
                "paired_joint_velocity_physics_executed": True,
                "no_hidden_clipping": True,
            }
        except _NominalVelocityPreflightInadmissible:
            terminal_outcome = _preflight_inadmissible_outcome(
                bound_sample_count=len(bundle.protected_samples.samples),
                source_prefix_callback_count=int(
                    exact_prefix["observed_callback_count"]
                ),
                gripper_evidence=gripper_evidence,
            )
            boundary_filter = terminal_outcome["boundary_B_filter"]
            arms = terminal_outcome["arms"]
            diagnostics = terminal_outcome["diagnostics"]
            boundary_measurement = terminal_outcome["boundary_B_measurement"]
            boundary_admissibility = terminal_outcome[
                "boundary_B_admissibility"
            ]
            gripper_evidence = terminal_outcome["gripper_evidence"]
            execution = terminal_outcome["execution"]
            classification = terminal_outcome["classification"]
        finally:
            if source_env is not None:
                source_env.close()

        phase = "pure_validation_and_atomic_publication"
        provenance = {
            "source": source_git,
            "allocation": {
                "execution_environment": allocation["execution_environment"],
                "slurm_job_id": str(allocation["slurm_job_id"]),
                "host_name": allocation["host_name"],
                "device": allocation["device"],
            },
            "manifest": {
                "path": str(manifest_path),
                "file_sha256": manifest_sha256,
                "row_sha256": case_row_sha256,
                "case_id": case["case_id"],
            },
            "selection": {
                "path": str(selection_path),
                "file_sha256": _file_sha256(selection_path),
                "schema_version": selection["schema_version"],
                "protocol_id": selection["protocol_id"],
            },
            "runtime_protocol": {
                "path": str(runtime_protocol_path),
                "raw_file_sha256": _file_sha256(runtime_protocol_path),
                "semantic_sha256": runtime_hashes.protocol_sha256,
                "parameter_block_sha256": runtime_hashes.parameter_block_sha256,
                "schema_version": runtime_protocol["schema_version"],
                "protocol_id": runtime_protocol["protocol_id"],
            },
            "historical_result": {
                "path": str(historical_path),
                "file_sha256": replay.result_file_sha256,
                "payload_sha256": replay.result_payload_sha256,
                "action_count": len(replay.actions),
                "executed_action_sequence_sha256": replay.executed_sequence_sha256,
                "policy_noise_schedule_sha256": replay.policy_noise_schedule_sha256,
            },
            "physical_model": dict(physical_model),
            "joint_velocity_controller": dict(joint_velocity_controller),
            "contact_model": {
                "authority_sha256": shadow_binding[
                    "contact_model_authority_sha256"
                ],
                "active_obstacle_name": case["active_obstacle_name"],
                "resolved_geometry": resolved.to_dict(),
            },
            "field_bundle": {
                "bundle_sha256": bundle.hashes.bundle_sha256,
                "protected_samples_sha256": bundle.hashes.protected_samples_sha256,
                "protocol_sha256": bundle.hashes.protocol_sha256,
                "parameter_block_sha256": bundle.hashes.parameter_block_sha256,
            },
            "execution": {
                "active_policy_query_count": 0,
                "policy_server_started": False,
                "source_prefix_replayed_once": True,
            },
            "python": {"executable": sys.executable, "version": platform.python_version()},
        }
        source_boundary = {
            "target_contact": target_contact,
            "contact_physical_boundary_C": int(contact_boundary),
            "filter_boundary_B": int(boundary),
            "exact_prefix": exact_prefix,
            "source_restore": source_restore,
            "gripper_evidence": gripper_evidence,
            "warning_authorization": warning_authorization,
            "boundary_B_admissibility": (
                None
                if boundary_admissibility is None
                else dict(boundary_admissibility, measurement=boundary_measurement)
            ),
            "shadow_construction_binding": shadow_binding,
        }
        if execution["outcome_kind"] == "preflight_inadmissible_nominal_velocity":
            counts = terminal_outcome["counts"]
        else:
            counts = {
                "bound_sample_count": int(boundary_filter["ordered_sample_count"]),
                "source_prefix_callback_count": int(
                    exact_prefix["observed_callback_count"]
                ),
                "nominal_physics_substep_count": len(
                    nominal_arm["physics_substep_ledger"]
                ),
                "psf_physics_substep_count": len(psf_arm["physics_substep_ledger"]),
                "qp_solve_count": int(psf_arm["filter_solve_count"]),
                "nominal_invalid_field_query_count": int(
                    nominal_arm["invalid_field_query_count"]
                ),
                "psf_invalid_field_query_count": int(
                    psf_arm["invalid_field_query_count"]
                ),
                "nominal_target_contact_record_count": sum(
                    len(row["target_physical_contact_records"])
                    for row in nominal_arm["physics_substep_ledger"]
                ),
                "psf_robot_selected_obstacle_contact_record_count": sum(
                    len(row["robot_to_selected_obstacle_physical_contact_records"])
                    for row in psf_arm["physics_substep_ledger"]
                ),
            }
        identification_assessment = identification["shadow_replay"][
            "poisson_identification"
        ]["contact_prediction_assessment"]
        identification_construction = identification["shadow_replay"][
            "construction"
        ]
        parity_callback = parity["callback_replay"]
        identification_callback = _identification_callback_authority(identification)
        if (
            identification_callback[
                "callback_state_read_only_after_sha256_ledger"
            ]
            != parity_callback["official_integration_state_sha256_ledger"]
            or identification_callback["callback_state_sequence_sha256"]
            != parity_callback["official_integration_state_sequence_sha256"]
        ):
            raise OneStepRunnerError(
                "identification callback state authority differs from exact parity"
            )
        identification_settled_state_sha256 = identification_construction[
            "complete_integration_state_read_only_audit"
        ]["before_sha256"]
        if (
            identification_settled_state_sha256
            != parity_callback["settled_official_integration_state"]["sha256"]
        ):
            raise OneStepRunnerError(
                "identification settled state authority differs from exact parity"
            )
        authority["dynamic_authority"] = {
            "case": {
                "case_id": case["case_id"],
                "manifest_file_sha256": manifest_sha256,
                "manifest_row_sha256": case_row_sha256,
            },
            "selection": {
                "relative_path": frozen["selection_protocol_relative_path"],
                "schema_version": selection["schema_version"],
                "file_sha256": _file_sha256(selection_path),
                "protocol_id": selection["protocol_id"],
            },
            "historical": {
                "result_file_sha256": replay.result_file_sha256,
                "result_payload_sha256": replay.result_payload_sha256,
                "action_count": len(replay.actions),
                "executed_action_sequence_sha256": replay.executed_sequence_sha256,
                "policy_noise_schedule_sha256": replay.policy_noise_schedule_sha256,
            },
            "runtime": {
                "raw_file_sha256": _file_sha256(runtime_protocol_path),
                "semantic_sha256": runtime_hashes.protocol_sha256,
                "parameter_block_sha256": runtime_hashes.parameter_block_sha256,
                "schema_version": runtime_protocol["schema_version"],
                "protocol_id": runtime_protocol["protocol_id"],
                "registered_parameters": {
                    key: runtime_protocol[key]
                    for key in ("admissibility", "coverage", "cbf", "qp", "cadence")
                },
            },
            "parity": {
                "settled_official_integration_state": parity_callback[
                    "settled_official_integration_state"
                ],
                "executed_action_count": parity_callback["executed_action_count"],
                "action_boundary_state_sha256_ledger": parity_callback[
                    "action_boundary_state_sha256_ledger"
                ],
                "state_sequence_sha256": parity_callback["state_sequence_sha256"],
                "official_integration_state_count": parity_callback[
                    "official_integration_state_count"
                ],
                "official_integration_state_sha256_ledger": parity_callback[
                    "official_integration_state_sha256_ledger"
                ],
                "official_integration_state_sequence_sha256": parity_callback[
                    "official_integration_state_sequence_sha256"
                ],
            },
            "identification": {
                **identification_callback,
                "first_link56_contact": identification_assessment[
                    "first_link56_contact"
                ],
                "primary_registered_warning": identification_assessment[
                    "primary_registered_warning"
                ],
                "contact_model_authority_sha256": identification_construction[
                    "contact_model_authority_sha256"
                ],
                "resolved_geometry": identification_construction[
                    "resolved_geometry"
                ],
                "field_bundle_hashes": identification_construction["field_bundle"][
                    "hashes"
                ],
                "full_robot_measurement_sampling": identification_construction[
                    "full_robot_measurement_sampling"
                ],
                "physical_model": identification_construction["physical_model"],
                "ordered_protected_sample_ledger_sha256": (
                    identification_construction["field_bundle"]["hashes"][
                        "protected_samples_sha256"
                    ]
                ),
                "protected_sample_count": identification_construction[
                    "field_bundle"
                ]["protected_sample_count"],
                "arm_dof_indices": identification_construction["arm_dof_indices"],
                "settled_mjstate_integration_sha256": identification_construction[
                    "complete_integration_state_read_only_audit"
                ]["before_sha256"],
                "differential_binding_sha256": identification_construction[
                    "settled_link56_differential_audit"
                ]["binding_sha256"],
                "differential_classification_ledger_sha256": (
                    identification_construction[
                        "settled_link56_differential_audit"
                    ]["classification_ledger_sha256"]
                ),
                "differential_validation": identification_construction[
                    "settled_link56_differential_audit_validation"
                ],
            },
        }
        candidate: Dict[str, Any] = {
            "schema_version": EXPECTED_RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "run_id": arguments.run_id,
            "case_id": case["case_id"],
            "protocol_binding": protocol_binding,
            "authority": authority,
            "provenance": provenance,
            "execution": execution,
            "source_boundary": source_boundary,
            "nominal_velocity_estimate": nominal_velocity,
            "boundary_B_filter": boundary_filter,
            "arms": arms,
            "diagnostics": diagnostics,
            "counts": counts,
            "classification": classification,
            "passed": bool(classification["stage_13_passed"]),
        }
        _section_hashes(candidate)
        hashed_candidate = attach_payload_hash(candidate)
        core_validation = validate_one_step_counterfactual_result(
            hashed_candidate,
            protocol=stage_protocol,
            protocol_raw_sha256=protocol_raw_sha256,
            expected_authority=authority,
        )
        result_path = output / "result.json"
        publish_hashed_json(result_path, candidate)
        loaded_result = load_hashed_json(result_path)
        reloaded_core = validate_one_step_counterfactual_result(
            loaded_result,
            protocol=stage_protocol,
            protocol_raw_sha256=protocol_raw_sha256,
            expected_authority=authority,
        )
        if _canonical(reloaded_core) != _canonical(core_validation):
            raise OneStepRunnerError("reloaded pure validation record differs")
        receipt = {
            "schema_version": (
                "vlsa_poisson_one_step_counterfactual_validation_receipt.v1"
            ),
            "status": "validated",
            "scientific_result": False,
            "run_id": arguments.run_id,
            "case_id": case["case_id"],
            "result": {
                "relative_path": "result.json",
                "file_sha256": sha256_file(result_path),
                "payload_sha256": loaded_result["result_payload_sha256"],
                "schema_version": loaded_result["schema_version"],
            },
            "protocol_binding_sha256": loaded_result["protocol_binding_sha256"],
            "authority_sha256": loaded_result["authority_sha256"],
            "provenance_sha256": loaded_result["provenance_sha256"],
            "nominal_velocity_estimate_sha256": loaded_result[
                "nominal_velocity_estimate_sha256"
            ],
            "boundary_B_filter_sha256": loaded_result[
                "boundary_B_filter_sha256"
            ],
            "arm_ledger_sha256": loaded_result["arm_ledger_sha256"],
            "classification_ledger_sha256": loaded_result[
                "classification_ledger_sha256"
            ],
            "core_validation": core_validation,
            "producer": {
                "code_commit": authority["expected_code_commit"],
                "slurm_job_id": authority["expected_slurm_job_id"],
                "host": authority["expected_host"],
                "device": authority["expected_device"],
            },
            "independent_consumer_required": True,
            "external_slurm_requirement": (
                "exact_producer_job_must_be_independently_confirmed_"
                "COMPLETED_with_exit_0_0"
            ),
        }
        receipt_path = output / "validation_receipt.json"
        publish_hashed_json(receipt_path, receipt)
        loaded_receipt = load_hashed_json(receipt_path)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "run_id": arguments.run_id,
                    "classification": classification["label"],
                    "passed": classification["stage_13_passed"],
                    "result": str(result_path),
                    "receipt": str(receipt_path),
                    "producer_slurm_job_id": authority["expected_slurm_job_id"],
                    "producer_host": authority["expected_host"],
                    "producer_device": authority["expected_device"],
                    "result_file_sha256": sha256_file(result_path),
                    "result_payload_sha256": loaded_result[
                        "result_payload_sha256"
                    ],
                    "receipt_file_sha256": sha256_file(receipt_path),
                    "receipt_payload_sha256": loaded_receipt[
                        "result_payload_sha256"
                    ],
                    "independent_consumer_required": True,
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        if output is not None and output.is_dir() and not output.is_symlink():
            try:
                from main.poisson_fullbody.contracts import publish_hashed_json

                failure = {
                    "schema_version": "vlsa_poisson_one_step_counterfactual_failure.v1",
                    "status": "failed",
                    "scientific_result": False,
                    "run_id": arguments.run_id,
                    "case_id": arguments.case_id,
                    "phase": phase,
                    "failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    },
                    "source_commit": (
                        None if source_git is None else source_git.get("commit")
                    ),
                    "slurm_job_id": (
                        os.environ.get("SLURM_JOB_ID")
                        if allocation is None
                        else allocation.get("slurm_job_id")
                    ),
                    "host": socket.gethostname(),
                    "partial_result_interpretation_prohibited": True,
                    "strong_or_trend_classification": None,
                    "result_json_written": bool((output / "result.json").exists()),
                    "validation_receipt_written": bool(
                        (output / "validation_receipt.json").exists()
                    ),
                    "timing": {
                        "started_unix": started,
                        "finished_unix": time.time(),
                        "elapsed_seconds": time.time() - started,
                    },
                }
                publish_hashed_json(output / "run_failure.json", failure)
            except Exception as publication_error:
                print(
                    "one-step failure receipt could not be published: %s"
                    % publication_error,
                    file=sys.stderr,
                )
        print(
            "one-step counterfactual failed during %s: %s" % (phase, error),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
