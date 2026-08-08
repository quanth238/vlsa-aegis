"""EmbodiSteer-inspired task-metric multi-CBF guidance for pi0.5 flow.

This opt-in primary-case test preserves the released Cartesian pi0.5/OSC
execution interface.  It adopts EmbodiSteer's scheduled task-preserving QP
metric and extends the paper's one barrier row to the registered simultaneous
L5/L6/L7/end-effector trajectory rows.  It is deliberately described as an
OSC-consistent surrogate, not as faithful joint-space EmbodiSteer.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .predictive_flow import (
    PROTECTED_BODY_NAMES,
    build_flow_guidance_envelope,
)


EMBODISTEER_FLOW_SCHEMA = "vlsa_embodisteer_multicbf_flow.v1"
EMBODISTEER_ENVELOPE_SCHEMA = "crfs_embodisteer_multicbf_guidance.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_embodisteer_flow_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("EmbodiSteer-flow config is invalid JSON") from error
    required = {
        "schema_version",
        "protocol_id",
        "paper_reference",
        "case_ids",
        "claim_scope",
        "nominal_policy",
        "action_protocol",
        "obstacle_geometry",
        "protected_body_names",
        "robot_geometry",
        "end_effector_geometry",
        "trajectory_model",
        "embodisteer_guidance",
        "verification",
        "success_definition",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("EmbodiSteer-flow config keys differ")
    if config["schema_version"] != EMBODISTEER_FLOW_SCHEMA:
        raise ValueError("EmbodiSteer-flow schema differs")
    if config["paper_reference"] != {
        "arxiv": "2606.12965",
        "adapted_equations": ["6", "14", "15", "16"],
        "fidelity": "osc_action_space_surrogate_not_full_joint_space_embodisteer",
    }:
        raise ValueError("EmbodiSteer paper mapping differs")
    if config["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("EmbodiSteer-flow case differs")
    if config["protected_body_names"] != PROTECTED_BODY_NAMES:
        raise ValueError("EmbodiSteer-flow protected bodies differ")
    if config["obstacle_geometry"] != "released_aegis_frozen_perception_mvee":
        raise ValueError("EmbodiSteer-flow obstacle differs")
    if config["action_protocol"] != {
        "execute_actions_per_query": 5,
        "model_action_horizon": 10,
        "modified_dimensions": [0, 1, 2],
        "preserve_gripper": True,
        "rotation": "zero_translational_protocol",
    }:
        raise ValueError("EmbodiSteer-flow action protocol differs")
    trajectory = config["trajectory_model"]
    if trajectory != {
        "activation_h_opt_m": 0.02,
        "clearance_tolerance_m": 1e-06,
        "finite_difference_action": 0.05,
        "finite_difference_scheme": "clipped_central_difference",
        "horizon_steps": 10,
        "optimizer_clearance_m": 0.01,
        "transition": "complete_cloned_safelibero_env_step_with_stateful_osc_pose_and_gripper",
    }:
        raise ValueError("EmbodiSteer-flow trajectory model differs")
    guidance = config["embodisteer_guidance"]
    if guidance != {
        "action_limit": 1.0,
        "barrier_decay_gamma": 0.9,
        "euler_steps": 10,
        "guidance_schedule": {
            "base_strength": 1.0,
            "beta": 50.0,
            "transition": 0.7,
        },
        "projection_residual_tolerance": 0.00005,
        "projection_sweeps_per_euler_step": 64,
        "relinearization_attempts": 1,
        "task_metric": {
            "joint_regularization_lambda": 0.01,
            "position_weight": 1.0,
            "source": "cloned_osc_end_effector_trajectory_jacobian",
        },
    }:
        raise ValueError("EmbodiSteer-flow guidance protocol differs")
    for value in guidance["guidance_schedule"].values():
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError("EmbodiSteer schedule is nonfinite")
    if config["verification"] != {
        "clone_state_tolerance": 1e-10,
        "execute_only_exactly_verified_chunk": True,
        "raw_simulator_contact_and_car_authority": True,
    }:
        raise ValueError("EmbodiSteer-flow verification differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def build_embodisteer_guidance_envelope(
    model: Mapping[str, Any],
    nominal_output_actions: Any,
    current_h_opt_m: Any,
    *,
    gamma: float,
    action_limit: float,
    projection_tolerance: float,
    joint_regularization_lambda: float,
    position_weight: float,
    guidance_schedule: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one task-metric projection containing every barrier row."""

    import numpy as np

    envelope, record = build_flow_guidance_envelope(
        model,
        nominal_output_actions,
        current_h_opt_m,
        gamma=gamma,
        action_limit=action_limit,
        projection_tolerance=projection_tolerance,
    )
    eef_jacobian = np.asarray(model["eef_jacobian"], dtype=np.float64)
    if eef_jacobian.shape != (10, 3, 30) or not np.all(
        np.isfinite(eef_jacobian)
    ):
        raise ValueError("EmbodiSteer end-effector Jacobian differs")
    task_jacobian = eef_jacobian.reshape(30, 30)
    regularization = float(joint_regularization_lambda)
    weight = float(position_weight)
    if regularization <= 0.0 or weight <= 0.0:
        raise ValueError("EmbodiSteer task metric weights must be positive")
    metric = weight * (task_jacobian.T @ task_jacobian) + regularization * np.eye(30)
    rows = np.asarray(envelope["delta_rows"], dtype=np.float64)
    directions = np.linalg.solve(metric, rows.T).T
    denominators = np.sum(rows * directions, axis=1)
    if (
        directions.shape != rows.shape
        or not np.all(np.isfinite(directions))
        or not np.all(denominators > 1.0e-12)
    ):
        raise ValueError("EmbodiSteer task-metric directions are invalid")
    condition = float(np.linalg.cond(metric))
    if not math.isfinite(condition):
        raise ValueError("EmbodiSteer task metric is singular")

    envelope["schema_version"] = EMBODISTEER_ENVELOPE_SCHEMA
    envelope["task_metric_directions"] = directions.tolist()
    envelope["task_metric_condition_number"] = condition
    envelope["guidance_schedule"] = {
        key: float(guidance_schedule[key])
        for key in ("base_strength", "beta", "transition")
    }
    record["paper_mapping"] = {
        "reference": "EmbodiSteer arXiv:2606.12965",
        "single_to_multi_constraint_extension": True,
        "paper_qp_equations": ["6", "14", "15"],
        "paper_schedule_equation": "16",
        "surrogate_limitation": "OSC action-space metric, not joint-space denoising",
    }
    record["task_metric"] = {
        "definition": "J_eef_trajectory.T @ W @ J_eef_trajectory + lambda * I",
        "shape": [30, 30],
        "position_weight": weight,
        "joint_regularization_lambda": regularization,
        "condition_number": condition,
        "sha256": hashlib.sha256(
            np.ascontiguousarray(metric, dtype="<f8").tobytes()
        ).hexdigest(),
        "direction_sha256": hashlib.sha256(
            np.ascontiguousarray(directions, dtype="<f8").tobytes()
        ).hexdigest(),
        "minimum_projection_denominator": float(np.min(denominators)),
    }
    record["guidance_schedule"] = dict(envelope["guidance_schedule"])
    return envelope, record
