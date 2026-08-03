#!/usr/bin/env python3
"""Run the lean native-OSC moving-obstacle link-5/link-6 experiment.

The archived successful AEGIS rollout is the immutable collision control.  The
live treatment executes those exact actions under the released ``OSC_POSE``
controller until the first byte-different Poisson-CBF output.  It then finishes
the task from its own observations.  Both registered cases always protect the
same link-5/link-6 surface union; historical link labels never select rows.

Unlike the superseded direct joint-velocity pilot, this runner does not replace
OSC when the filter is inactive.  Unlike the superseded frozen-field pilot, it
attaches the field to the measured rigid obstacle pose and includes obstacle
twist in the time-varying CBF inequality.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from scripts import run_poisson_fast_feasibility as fast
from scripts import run_poisson_osc_arm_link_canary as osc_support


class ReferenceGovernorRunnerError(RuntimeError):
    """The experiment apparatus is invalid."""


class ReferenceGovernorMethodFailure(RuntimeError):
    """The registered controller cannot produce the next safe OSC action."""


class RegisteredContactObserved(RuntimeError):
    """A registered original or shifted collision occurred after physics."""


def _load_protocol(path: Path, case_id: str) -> Dict[str, Any]:
    value = fast._json(path, "OSC reference-governor protocol")
    expected_top = {
        "schema_version",
        "protocol_id",
        "allowed_case_ids",
        "protected_robot_body_names",
        "task_conditioned_link_selection",
        "controller",
        "field",
        "qp",
        "policy",
        "acceptance",
        "result",
    }
    if set(value) != expected_top:
        raise ReferenceGovernorRunnerError("reference-governor protocol keys differ")
    if (
        value["schema_version"]
        != "vlsa_poisson_osc_reference_governor_protocol.v1"
        or case_id not in value["allowed_case_ids"]
        or value["allowed_case_ids"]
        != ["vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t3-e42"]
        or value["protected_robot_body_names"]
        != ["robot0_link5", "robot0_link6"]
        or value["task_conditioned_link_selection"] is not False
    ):
        raise ReferenceGovernorRunnerError("case or shared protected-link scope differs")
    if not (
        value["controller"]["executor"] == "released_native_OSC_POSE_20Hz"
        and value["controller"]["zero_correction_behavior"]
        == "source_action_byte_exact_passthrough"
        and value["controller"]["joint_velocity_predictor"]
        == "measured_qdot_plus_DLS_of_commanded_minus_measured_EEF_twist"
        and value["field"]["frame"]
        == "selected_obstacle_rigid_body_attached"
        and value["field"]["rebuild_or_inflate_field"] is False
        and value["qp"]["hard_cbf_constraints"] is True
        and value["qp"]["slack"] is False
        and value["qp"]["zero_or_cached_command_fallback"] is False
    ):
        raise ReferenceGovernorRunnerError("registered method semantics differ")
    return value


def _root_pose_twist(model: Any, data: Any, body_id: int) -> Tuple[Any, Any]:
    import mujoco
    import numpy as np

    from main.poisson_fullbody.rigid_obstacle_field import RigidPose, RigidTwist

    position = np.asarray(data.xpos[int(body_id)], dtype=np.float64).copy()
    rotation = np.asarray(data.xmat[int(body_id)], dtype=np.float64).reshape(3, 3).copy()
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
        raise ReferenceGovernorRunnerError("obstacle rigid-body twist is nonfinite")
    return (
        RigidPose(position, rotation),
        RigidTwist(
            linear_world_m_per_s=velocity[3:].copy(),
            angular_world_rad_per_s=velocity[:3].copy(),
        ),
    )


def _path_length(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return 0.0
    import numpy as np

    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ReferenceGovernorRunnerError("EEF path contains invalid points")
    return float(np.sum(np.linalg.norm(np.diff(array, axis=0), axis=1)))


def _classification(metrics: Mapping[str, Any]) -> Tuple[str, bool]:
    if metrics["registered_contact_seen"]:
        return "CONTACT_REMAINS_OR_SHIFTED", False
    if metrics["method_failure_seen"]:
        return "METHOD_INFEASIBLE_NO_STOP_COMMAND_EXECUTED", False
    if not metrics["material_correction_present"]:
        return "NO_MATERIAL_CORRECTION", False
    if not metrics["material_correction_before_historical_contact"]:
        return "CORRECTION_TOO_LATE", False
    if not metrics["paper_car_avoided"]:
        return "CAR_FAILURE_REMAINS", False
    if not metrics["useful_post_correction_motion"]:
        return "STOP_OR_STALL", False
    if not metrics["native_task_success_after_correction"]:
        return "CONTACT_PREVENTED_TASK_FAILED", False
    return "SAFE_TASK_SUCCESS_USEFUL_CORRECTION", True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/vlsa_poisson_osc_reference_governor.v1.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/vlsa_poisson_arm_contact_165.v1.jsonl"),
    )
    parser.add_argument("--historical-result-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()

    root = arguments.repo_root.resolve()
    os.chdir(str(root))
    for path in (root, root / "main", root / "safelibero"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    output: Optional[Path] = None
    result_path: Optional[Path] = None
    env: Any = None
    video: Any = None
    started = time.time()
    provenance: Dict[str, Any] = {}
    apparatus: Dict[str, Any] = {}
    treatment: Dict[str, Any] = {}
    try:
        import mujoco
        import numpy as np
        import main.evaluate_safelibero_aegis as evaluator
        from main.poisson_fullbody.contracts import publish_hashed_json
        from main.poisson_fullbody.feasibility_protocol import (
            load_feasibility_protocol,
        )
        from main.poisson_fullbody.field_bundle import build_static_field_bundle
        from main.poisson_fullbody.jacobians import evaluate_point_jacobians
        from main.poisson_fullbody.measurement import (
            clone_forwarded_state,
            resolve_collision_geom_sets,
        )
        from main.poisson_fullbody.osc_reference_governor import (
            OscPoseReferenceGovernor,
            osc_output_scale,
        )
        from main.poisson_fullbody.rigid_obstacle_field import (
            dynamic_cbf_rows,
            query_rigid_obstacle_field_batch,
        )
        from main.poisson_fullbody.shadow_replay import (
            load_historical_action_replay,
        )
        from scripts.run_poisson_shadow_parity import (
            _check_step,
            _observation_sha256,
            _prepare_environment,
        )

        output = fast._new_output(arguments.output_root.resolve(), arguments.run_id)
        result_path = output / "result.json"
        protocol_path = (
            arguments.protocol
            if arguments.protocol.is_absolute()
            else root / arguments.protocol
        ).resolve()
        manifest_path = (
            arguments.manifest
            if arguments.manifest.is_absolute()
            else root / arguments.manifest
        ).resolve()
        protocol = _load_protocol(protocol_path, arguments.case_id)
        case, case_row_sha256 = osc_support._manifest_case(
            manifest_path, arguments.case_id
        )
        if not (
            case.get("clean_task_success_canary_eligible") is True
            and case.get("settled_relevant_contact") is False
            and case.get("only_link56_selected_obstacle_robot_contact") is True
            and case.get("historical_outcome", {}).get("task_success") is True
            and case.get("historical_outcome", {}).get("paper_car_failure") is True
        ):
            raise ReferenceGovernorRunnerError(
                "case is not a verified task-successful link-5/link-6 collision control"
            )
        historical_path = osc_support._historical_path(
            arguments.historical_result_root.resolve(), case
        )
        replay = load_historical_action_replay(
            historical_path,
            expected_case_id=arguments.case_id,
            expected_arm="pi05_plus_aegis_translational",
        )
        historical = fast._json(historical_path, "historical AEGIS control")
        historical_binding = case["historical_result"]
        if not (
            replay.result_file_sha256 == historical_binding["file_sha256"]
            and replay.result_payload_sha256 == historical_binding["payload_sha256"]
            and replay.executed_sequence_sha256
            == historical_binding["executed_action_sequence_sha256"]
            and replay.historical_task_success is True
            and replay.historical_car_collision is True
        ):
            raise ReferenceGovernorRunnerError("historical control bytes differ")

        runtime_path = (root / protocol["field"]["runtime_protocol_path"]).resolve()
        runtime_protocol, runtime_hashes = load_feasibility_protocol(runtime_path)
        source = fast._git(root)
        allocation = fast._allocation()
        provenance = {
            "source": source,
            "allocation": allocation,
            "python": platform.python_version(),
            "run_id": arguments.run_id,
            "case_id": arguments.case_id,
            "protocol_file_sha256": fast._file_sha256(protocol_path),
            "manifest_file_sha256": fast._file_sha256(manifest_path),
            "case_row_sha256": case_row_sha256,
            "runtime_protocol_file_sha256": fast._file_sha256(runtime_path),
            "runtime_protocol_semantic_sha256": runtime_hashes.protocol_sha256,
            "historical_result_file_sha256": replay.result_file_sha256,
            "historical_result_payload_sha256": replay.result_payload_sha256,
            "historical_action_sequence_sha256": replay.executed_sequence_sha256,
        }

        runtime = evaluator._runtime_imports(include_aegis=True)
        runtime_case = osc_support._source_case(case)
        env, task, observation, goal_atoms, previous_goal = _prepare_environment(
            evaluator, runtime, runtime_case, replay
        )
        if tuple(previous_goal) != (False,):
            raise ReferenceGovernorRunnerError("task is already complete after settling")
        if not hasattr(env, "step_with_action_reference_intervention"):
            raise ReferenceGovernorRunnerError(
                "native OSC action-reference intervention wrapper is absent"
            )
        settled_state = np.asarray(
            env.sim.get_state().flatten(), dtype=np.float64
        ).copy()
        settled_state_sha256 = evaluator.array_sha256(settled_state)
        if settled_state_sha256 != case["pairing"]["settled_simulator_state_sha256"]:
            raise ReferenceGovernorRunnerError("settled simulator state differs")
        controller_record = osc_support._controller_record(env)
        if controller_record.get("original_osc_verified") is not True:
            raise ReferenceGovernorRunnerError("released native OSC is not active")
        controller_scale = osc_output_scale(env.robots[0].controller)

        obstacle_name, _ = evaluator._active_obstacle(env, observation)
        if obstacle_name != case["active_obstacle_name"]:
            raise ReferenceGovernorRunnerError("active obstacle differs")
        authority = evaluator._contact_model_authority(env, obstacle_name)
        obstacle_root_body_id = int(authority["active_obstacle_root_body_id"])
        robot_root_body_id = fast._body_id(
            env.sim.model, env.robots[0].robot_model.root_body
        )
        protected_names = tuple(protocol["protected_robot_body_names"])
        protected_ids = tuple(
            fast._body_id(env.sim.model, name) for name in protected_names
        )
        raw_model, raw_data = fast._raw_model_data(env.sim)
        resolved = resolve_collision_geom_sets(
            raw_model,
            robot_root_body_ids=(robot_root_body_id,),
            obstacle_root_body_ids=(obstacle_root_body_id,),
            link56_body_ids=protected_ids,
        )
        if set(int(value) for value in resolved.obstacle_body_ids) != {
            obstacle_root_body_id
        }:
            raise ReferenceGovernorRunnerError(
                "selected obstacle is not one rigid root-body geometry set"
            )
        official_before_bundle = fast._official_state(env.sim)
        bundle = build_static_field_bundle(
            raw_model,
            raw_data,
            resolved=resolved,
            protocol=runtime_protocol,
            protocol_hashes=runtime_hashes,
            protected_body_names=protected_names,
        )
        if not np.array_equal(official_before_bundle, fast._official_state(env.sim)):
            raise ReferenceGovernorRunnerError("field construction changed simulator state")
        if tuple(bundle.protected_body_names) != protected_names:
            raise ReferenceGovernorRunnerError("field protected-link union differs")
        reference_pose, settled_twist = _root_pose_twist(
            raw_model, raw_data, obstacle_root_body_id
        )
        arm_qvel_indices = tuple(
            int(value) for value in env.robots[0]._ref_joint_vel_indexes
        )
        if len(arm_qvel_indices) != 7:
            raise ReferenceGovernorRunnerError("Panda arm qvel order differs")
        site_id = fast._eef_site_id(env)
        contact_monitor = osc_support._RegisteredContactMonitor(
            env.sim,
            resolved,
            physics_substeps_per_action=25,
        )
        if contact_monitor.settled_contact:
            raise ReferenceGovernorRunnerError(
                "treatment begins in a registered forbidden contact"
            )

        planner = osc_support.HybridAegisPlanner(
            evaluator=evaluator,
            runtime=runtime,
            case=runtime_case,
            task_description=str(task.language),
            historical=historical,
            host=arguments.host,
            port=arguments.port,
            replan_steps=int(protocol["policy"]["replan_steps"]),
            model_action_horizon=int(protocol["policy"]["model_action_horizon"]),
        )
        qp = protocol["qp"]
        governor = OscPoseReferenceGovernor(
            damping=float(
                protocol["controller"]["damped_least_squares_lambda"]
            ),
            eps_abs=float(qp["eps_abs"]),
            eps_rel=float(qp["eps_rel"]),
            max_iter=int(qp["max_iterations"]),
            postcheck_tolerance=float(qp["postcheck_tolerance"]),
            material_action_correction=float(
                qp["material_action_correction_l2"]
            ),
        )
        video = osc_support.TreatmentVideo(
            evaluator,
            runtime["imageio"],
            output,
            fps=20,
        )
        video.append(observation, "settled_pre_action", -1)

        paper_car_key = "%s_pos" % obstacle_name
        if paper_car_key not in observation:
            raise ReferenceGovernorRunnerError("paper CAR observation is absent")
        settled_car_position = np.asarray(
            observation[paper_car_key], dtype=np.float64
        ).copy()
        car_threshold = float(
            protocol["acceptance"]["paper_car_displacement_threshold_m"]
        )
        action_trace: List[Dict[str, Any]] = []
        eef_path: List[List[float]] = []
        car_ledger: List[Dict[str, Any]] = []
        first_material_action: Optional[int] = None
        first_byte_divergence_action: Optional[int] = None
        material_correction_count = 0
        maximum_action_correction = 0.0
        nominal_identity_before_correction = True
        predivergence_parity_trace: List[Dict[str, Any]] = []
        registered_contact_seen = False
        method_failure: Optional[Dict[str, Any]] = None
        completed_action_count = 0
        observed_physics_substeps = 0
        task_success = False
        task_success_action: Optional[int] = None
        maximum_car_displacement = 0.0
        zero_pose_action_count_after_correction = 0
        pose_action_count_after_correction = 0
        terminal_kind = "maximum_action_count"

        maximum_actions = int(runtime_case["max_steps"])
        for source_index in range(maximum_actions):
            if planner.divergence_action is None and source_index >= len(replay.actions):
                terminal_kind = "historical_control_ended_without_material_correction"
                break
            nominal_action = np.asarray(
                planner.action(
                    env=env,
                    observation=observation,
                    source_index=source_index,
                ),
                dtype=np.float64,
            )
            if nominal_action.shape != (7,) or not np.all(np.isfinite(nominal_action)):
                raise ReferenceGovernorRunnerError("planner action is invalid")
            decision: Dict[str, Any] = {}

            def filter_reference(sim: Any, source_action: Any) -> Any:
                nonlocal first_material_action
                nonlocal first_byte_divergence_action
                nonlocal material_correction_count
                nonlocal maximum_action_correction
                nonlocal nominal_identity_before_correction
                current_pose, current_twist = _root_pose_twist(
                    raw_model, raw_data, obstacle_root_body_id
                )
                points, point_jacobians = evaluate_point_jacobians(
                    raw_model,
                    raw_data,
                    bundle.protected_samples.samples,
                    arm_qvel_indices,
                )
                queries = query_rigid_obstacle_field_batch(
                    bundle.field,
                    points,
                    reference_pose=reference_pose,
                    current_pose=current_pose,
                    current_twist=current_twist,
                )
                if not queries or any(not row.valid for row in queries):
                    raise ReferenceGovernorMethodFailure(
                        "moving rigid Poisson query is invalid"
                    )
                h = np.asarray([row.value_m2 for row in queries], dtype=np.float64)
                gradients = np.asarray(
                    [row.gradient_world_m for row in queries], dtype=np.float64
                )
                partial_time = np.asarray(
                    [row.partial_time_m2_per_s for row in queries],
                    dtype=np.float64,
                )
                if np.any(h <= 0.0):
                    raise ReferenceGovernorMethodFailure(
                        "protected sample is already outside h>0"
                    )
                cbf_rows, cbf_lower = dynamic_cbf_rows(
                    h,
                    gradients,
                    point_jacobians,
                    partial_time,
                    alpha_per_s=float(protocol["field"]["alpha_per_s"]),
                    margin_m2_per_s=float(
                        protocol["field"]["margin_m2_per_s"]
                    ),
                )
                eef_position, _, eef_jacobian = fast._eef_kinematics(
                    env, site_id, arm_qvel_indices
                )
                result = governor.filter_action(
                    source_action,
                    eef_jacobian=eef_jacobian,
                    measured_arm_qvel_rad_per_s=np.asarray(
                        raw_data.qvel,
                        dtype=np.float64,
                    )[list(arm_qvel_indices)],
                    cbf_rows_qdot=cbf_rows,
                    cbf_lower_m2_per_s=cbf_lower,
                    qdot_lower_rad_per_s=None,
                    qdot_upper_rad_per_s=None,
                    osc_output_scale=controller_scale,
                    control_dt_seconds=float(env.env.control_timestep),
                )
                decision.update(
                    {
                        "source_action_index": int(source_index),
                        "source_action": np.asarray(source_action).tolist(),
                        "minimum_h_m2": float(np.min(h)),
                        "maximum_abs_partial_h_per_s": float(
                            np.max(np.abs(partial_time))
                        ),
                        "obstacle_root_position_world_m": (
                            current_pose.position_world_m.tolist()
                        ),
                        "obstacle_linear_velocity_world_m_per_s": (
                            current_twist.linear_world_m_per_s.tolist()
                        ),
                        "obstacle_angular_velocity_world_rad_per_s": (
                            current_twist.angular_world_rad_per_s.tolist()
                        ),
                        "governor_reason": result.reason,
                        "governor_diagnostics": dict(result.diagnostics),
                    }
                )
                if not result.valid or result.action is None:
                    raise ReferenceGovernorMethodFailure(result.reason)
                filtered = np.asarray(result.action, dtype=np.float64)
                byte_identical = bool(
                    filtered.tobytes(order="C")
                    == np.asarray(source_action, dtype=np.float64).tobytes(order="C")
                )
                correction = float(
                    result.diagnostics["correction_l2_normalized_action"]
                )
                material = bool(result.diagnostics["material_correction"])
                decision.update(
                    {
                        "filtered_action": filtered.tolist(),
                        "action_byte_identical_to_source": byte_identical,
                        "correction_l2_normalized_action": correction,
                        "material_correction": material,
                    }
                )
                maximum_action_correction = max(
                    maximum_action_correction, correction
                )
                if first_material_action is None:
                    nominal_identity_before_correction = bool(
                        nominal_identity_before_correction
                        and (material or byte_identical)
                    )
                if not byte_identical and first_byte_divergence_action is None:
                    first_byte_divergence_action = int(source_index)
                    planner.mark_divergence(
                        action_index=source_index,
                        physical_boundary=source_index * 25,
                    )
                if material:
                    material_correction_count += 1
                    if first_material_action is None:
                        first_material_action = int(source_index)
                        eef_path.append(eef_position.tolist())
                return filtered

            def observe_poststep(sim: Any, substep_index: int) -> None:
                nonlocal observed_physics_substeps
                nonlocal registered_contact_seen
                snapshot = contact_monitor.observe_post_integration(
                    sim,
                    source_action_index=source_index,
                    physics_substep_index=int(substep_index),
                )
                observed_physics_substeps += 1
                if first_material_action is not None:
                    forwarded = clone_forwarded_state(raw_model, raw_data)
                    eef_path.append(
                        np.asarray(
                            forwarded.site_xpos[site_id], dtype=np.float64
                        ).tolist()
                    )
                if snapshot["literal_contact"]:
                    registered_contact_seen = True
                    raise RegisteredContactObserved()

            try:
                observation_after, reward, done, info = (
                    env.step_with_action_reference_intervention(
                        nominal_action,
                        filter_reference,
                        poststep_callback=observe_poststep,
                        expected_substeps=25,
                    )
                )
            except ReferenceGovernorMethodFailure as error:
                method_failure = {
                    "source_action_index": int(source_index),
                    "reason": str(error),
                    "decision": dict(decision),
                }
                terminal_kind = "reference_governor_method_failure_before_physics"
                break
            except RegisteredContactObserved:
                terminal_kind = "registered_original_or_shifted_contact"
                break

            completed_action_count += 1
            observation = observation_after
            action_trace.append(dict(decision))
            video.append(observation, "completed_action", source_index)
            state_hash = evaluator.array_sha256(env.sim.get_state().flatten())
            observation_hash = _observation_sha256(observation, evaluator, np)
            if first_byte_divergence_action is None:
                state_hash, observation_hash, previous_goal = _check_step(
                    evaluator=evaluator,
                    env=env,
                    observation=observation,
                    reward=reward,
                    done=done,
                    expected_step=replay.steps[source_index],
                    goal_atoms=goal_atoms,
                    previous_goal_values=previous_goal,
                    np=np,
                )
                predivergence_parity_trace.append(
                    {
                        "source_action_index": int(source_index),
                        "simulator_state_sha256": state_hash,
                        "observation_sha256": observation_hash,
                        "historical_match": True,
                    }
                )
                goal_all_satisfied = bool(done)
            else:
                goal = evaluator._goal_progress_snapshot(
                    env,
                    goal_atoms,
                    step=source_index,
                    previous_values=previous_goal,
                )
                if bool(done) is not bool(goal["all_satisfied"]):
                    raise ReferenceGovernorRunnerError(
                        "native done differs from BDDL goal"
                    )
                previous_goal = goal["values"]
                goal_all_satisfied = bool(goal["all_satisfied"])
            evaluator._update_eef_marker(
                env,
                evaluator._eef_proxy(runtime, observation),
            )
            car_position = np.asarray(
                observation[paper_car_key], dtype=np.float64
            )
            car_displacement = float(
                np.sum(np.abs(car_position - settled_car_position))
            )
            maximum_car_displacement = max(
                maximum_car_displacement, car_displacement
            )
            car_ledger.append(
                {
                    "source_action_index": int(source_index),
                    "l1_displacement_m": car_displacement,
                    "threshold_crossed": bool(car_displacement > car_threshold),
                }
            )
            if first_material_action is not None:
                pose_action_count_after_correction += 1
                if float(
                    np.linalg.norm(
                        np.asarray(decision["filtered_action"], dtype=np.float64)[:6]
                    )
                ) <= 1.0e-8:
                    zero_pose_action_count_after_correction += 1
            task_success = goal_all_satisfied
            if task_success:
                task_success_action = int(source_index)
                terminal_kind = "native_task_success"
                break

        video_record = video.close()
        video = None
        contact_result = contact_monitor.result()
        eef_path_length = _path_length(eef_path)
        zero_fraction = (
            1.0
            if pose_action_count_after_correction == 0
            else float(
                zero_pose_action_count_after_correction
                / pose_action_count_after_correction
            )
        )
        acceptance = protocol["acceptance"]
        historical_contact_action = int(
            case["first_sampled_link_contact_control_step"]
        )
        metrics = {
            "historical_control_task_success": True,
            "historical_control_link56_contact": True,
            "historical_control_paper_car_failure": True,
            "historical_contact_source_action": historical_contact_action,
            "material_correction_present": bool(first_material_action is not None),
            "first_byte_divergence_source_action": first_byte_divergence_action,
            "first_material_correction_source_action": first_material_action,
            "material_correction_before_historical_contact": bool(
                first_material_action is not None
                and first_material_action <= historical_contact_action
            ),
            "material_correction_count": int(material_correction_count),
            "maximum_action_correction_l2": float(maximum_action_correction),
            "native_identity_before_first_correction": bool(
                nominal_identity_before_correction
            ),
            "predivergence_state_reward_done_goal_parity_exact": bool(
                first_byte_divergence_action is not None
                and len(predivergence_parity_trace)
                == first_byte_divergence_action
                and all(
                    row["historical_match"] is True
                    for row in predivergence_parity_trace
                )
            ),
            "registered_contact_seen": bool(
                registered_contact_seen
                or contact_result["any_registered_forbidden_contact"]
            ),
            "any_robot_selected_obstacle_contact": bool(
                contact_result["any_robot_selected_obstacle_contact"]
            ),
            "any_link56_external_nonrobot_contact": bool(
                contact_result["any_link56_external_nonrobot_contact"]
            ),
            "paper_car_avoided": bool(
                maximum_car_displacement <= car_threshold
            ),
            "maximum_paper_car_l1_displacement_m": float(
                maximum_car_displacement
            ),
            "post_correction_eef_path_length_m": float(eef_path_length),
            "post_correction_zero_pose_action_fraction": float(zero_fraction),
            "useful_post_correction_motion": bool(
                eef_path_length
                >= float(acceptance["minimum_post_correction_eef_path_m"])
                and zero_fraction
                <= float(
                    acceptance[
                        "maximum_post_correction_zero_pose_action_fraction"
                    ]
                )
            ),
            "native_task_success": bool(task_success),
            "native_task_success_source_action": task_success_action,
            "native_task_success_after_correction": bool(
                task_success
                and first_material_action is not None
                and task_success_action is not None
                and task_success_action >= first_material_action
            ),
            "method_failure_seen": bool(method_failure is not None),
            "completed_source_action_count": int(completed_action_count),
            "observed_physics_substep_count": int(observed_physics_substeps),
        }
        classification, feasible = _classification(metrics)
        apparatus = {
            "controller": controller_record,
            "settled_state_sha256": settled_state_sha256,
            "protected_robot_body_names": list(bundle.protected_body_names),
            "task_conditioned_link_selection": False,
            "field_bundle_hashes": asdict(bundle.hashes),
            "field_diagnostics": asdict(bundle.diagnostics),
            "reference_obstacle_pose": {
                "root_body_id": obstacle_root_body_id,
                "position_world_m": reference_pose.position_world_m.tolist(),
                "rotation_world_from_body": (
                    reference_pose.rotation_world_from_body.tolist()
                ),
                "settled_linear_velocity_world_m_per_s": (
                    settled_twist.linear_world_m_per_s.tolist()
                ),
                "settled_angular_velocity_world_rad_per_s": (
                    settled_twist.angular_world_rad_per_s.tolist()
                ),
            },
            "moving_field_semantics": (
                "h(x,t)=h_ref(p_ref+R_ref R_t^T(x-p_t)); "
                "partial_t_h=-grad_world_dot(v+omega_cross_radius)"
            ),
            "contact_scope": contact_monitor.scope,
        }
        treatment = {
            "terminal_kind": terminal_kind,
            "planner": planner.record(),
            "action_trace": action_trace,
            "predivergence_parity_trace": predivergence_parity_trace,
            "method_failure": method_failure,
            "contact": contact_result,
            "paper_car": {
                "threshold_m": car_threshold,
                "ledger": car_ledger,
            },
            "video": video_record,
            "metrics": metrics,
        }
        candidate = {
            "schema_version": protocol["result"]["schema_version"],
            "status": "complete",
            "run_id": arguments.run_id,
            "case_id": arguments.case_id,
            "provenance": provenance,
            "protocol": protocol,
            "apparatus": apparatus,
            "historical_control": {
                "result_file_sha256": replay.result_file_sha256,
                "result_payload_sha256": replay.result_payload_sha256,
                "executed_action_sequence_sha256": (
                    replay.executed_sequence_sha256
                ),
                "task_success": True,
                "link56_contact": True,
                "paper_car_failure": True,
                "first_link56_contact_source_action": historical_contact_action,
            },
            "treatment": treatment,
            "classification": {
                "classification": classification,
                "feasible": feasible,
                "positive_requires_useful_correction_not_stopping": True,
            },
            "claim_scope": protocol["result"]["claim_scope"],
            "partial_output_interpreted": False,
            "timing": {
                "started_unix": started,
                "finished_unix": time.time(),
                "elapsed_seconds": time.time() - started,
            },
        }
        publish_hashed_json(result_path, candidate)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "classification": classification,
                    "feasible": feasible,
                    "result": str(result_path),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        if video is not None:
            try:
                video.close()
            except Exception:
                pass
        if output is None:
            try:
                output = fast._new_output(
                    arguments.output_root.resolve(), arguments.run_id
                )
                result_path = output / "result.json"
            except Exception:
                result_path = None
        if result_path is not None and not result_path.exists():
            try:
                from main.poisson_fullbody.contracts import publish_hashed_json

                publish_hashed_json(
                    result_path,
                    {
                        "schema_version": (
                            "vlsa_poisson_osc_reference_governor_result.v1"
                        ),
                        "status": "apparatus_failure",
                        "run_id": arguments.run_id,
                        "case_id": arguments.case_id,
                        "provenance": provenance,
                        "apparatus": apparatus,
                        "treatment": treatment,
                        "partial_output_interpreted": False,
                        "failure": {
                            "type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        },
                        "timing": {
                            "started_unix": started,
                            "finished_unix": time.time(),
                            "elapsed_seconds": time.time() - started,
                        },
                    },
                )
            except Exception as publication_error:
                print(
                    "failed to publish reference-governor failure: %s"
                    % publication_error,
                    file=sys.stderr,
                )
        print("OSC reference-governor run failed: %s" % error, file=sys.stderr)
        return 2
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
