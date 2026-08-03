#!/usr/bin/env python3
"""Run the minimal paper-faithful link-aware Poisson-CBF experiment.

Both arms start from the exact settled SafeLIBERO state and use the same live
pi0.5 + released AEGIS high-level actions.  Both then use the same Cartesian
to joint-velocity adapter and the same 100 Hz JOINT_VELOCITY controller.  The
only treatment difference is the hard Poisson-CBF QP over the fixed link-5 and
link-6 surface set.  The claim-bearing command is joint velocity throughout.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from scripts import run_poisson_closed_loop_canary as live
from scripts import run_poisson_fast_feasibility as fast


class DirectJointVelocityRunnerError(RuntimeError):
    """The direct experiment cannot support a scientific interpretation."""


def _manifest_case(path: Path, case_id: str) -> Tuple[Dict[str, Any], str]:
    matches: List[Tuple[Dict[str, Any], str]] = []
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            value = json.loads(raw)
            if value.get("case_id") == case_id:
                matches.append((value, fast._sha256(raw.rstrip(b"\r\n"))))
    if len(matches) != 1:
        raise DirectJointVelocityRunnerError(
            "manifest must contain exactly one %s row" % case_id
        )
    return matches[0]


def _validate_manifest_case_identity(
    case: Mapping[str, Any], case_contract: Mapping[str, Any]
) -> None:
    """Bind the frozen protocol to the immutable manifest's real schema."""

    manifest_bindings = {
        "active_obstacle_name": "selected_obstacle_name",
        "bddl_path": "bddl_path",
        "bddl_sha256": "bddl_sha256",
        "initial_states_path": "initial_states_path",
        "initial_states_sha256": "initial_states_sha256",
        "environment_seed": "environment_seed",
        "episode_index": "episode_index",
        "logical_task_index": "logical_task_index",
        "resolved_task_index": "resolved_task_index",
        "task_name": "task_name",
        "suite": "suite",
        "safety_level": "safety_level",
        "policy_noise_seed": "policy_noise_seed",
        "policy_noise_schedule_id": "policy_noise_schedule_id",
    }
    if any(
        case.get(manifest_key) != case_contract.get(protocol_key)
        for manifest_key, protocol_key in manifest_bindings.items()
    ):
        raise DirectJointVelocityRunnerError(
            "manifest case identity differs from the frozen protocol"
        )
    historical = case.get("historical_aegis_result")
    pairing = (
        historical.get("pairing") if isinstance(historical, Mapping) else None
    )
    nested_pairing_keys = (
        "policy_noise_schedule_sha256",
        "settled_simulator_state_sha256",
        "initial_observation_sha256",
        "initial_state_sha256",
    )
    if not isinstance(pairing, Mapping) or any(
        pairing.get(key) != case_contract.get(key) for key in nested_pairing_keys
    ):
        raise DirectJointVelocityRunnerError(
            "manifest historical pairing differs from the frozen protocol"
        )


def _controller_is_direct_joint_velocity(arm: Mapping[str, Any]) -> bool:
    restore = arm.get("restore")
    controller = restore.get("controller") if isinstance(restore, Mapping) else None
    return bool(
        isinstance(controller, Mapping)
        and controller.get("controller_name") == "JOINT_VELOCITY"
        and controller.get("controller_class_qualname")
        == "JointVelocityController"
        and controller.get("environment_action_dim") == 8
        and controller.get("arm_control_dim") == 7
        and controller.get("control_frequency_hz") == 100
    )


def _provider_complete(
    record: Mapping[str, Any], *, action_count: int, query_count: int
) -> bool:
    actions = record.get("high_level_action_trace")
    queries = record.get("policy_queries")
    if (
        not isinstance(actions, Sequence)
        or isinstance(actions, (str, bytes))
        or not isinstance(queries, Sequence)
        or isinstance(queries, (str, bytes))
        or len(actions) != action_count
        or len(queries) != query_count
    ):
        return False
    for local_index, row in enumerate(actions):
        if (
            not isinstance(row, Mapping)
            or row.get("local_action_index") != local_index
            or row.get("source_action_index") != local_index
            or row.get("query_index") != local_index // 5
            or row.get("query_chunk_offset") != local_index % 5
            or row.get("aegis_qp", {}).get("status") != "solved"
        ):
            return False
    for query_index, row in enumerate(queries):
        if (
            not isinstance(row, Mapping)
            or row.get("query_index") != query_index
            or row.get("local_action_index") != query_index * 5
            or row.get("source_action_index") != query_index * 5
            or row.get("returned_action_shape") != [10, 7]
        ):
            return False
    return bool(
        record.get("source_start_action") == 0
        and record.get("first_query_index") == 0
        and record.get("recorded_suffix_actions_executed") is False
    )


def _first_aegis_pair_exact(
    baseline: Mapping[str, Any], treatment: Mapping[str, Any]
) -> bool:
    baseline_rows = baseline.get("high_level_action_trace")
    treatment_rows = treatment.get("high_level_action_trace")
    if not baseline_rows or not treatment_rows:
        return False
    first_baseline = baseline_rows[0]
    first_treatment = treatment_rows[0]
    return bool(
        first_baseline.get("source_action_index") == 0
        and first_treatment.get("source_action_index") == 0
        and fast._canonical(
            live._aegis_input_projection(
                first_baseline, historical=False, include_nominal=True
            )
        )
        == fast._canonical(
            live._aegis_input_projection(
                first_treatment, historical=False, include_nominal=True
            )
        )
        and fast._canonical(
            live._aegis_output_projection(first_baseline, historical=False)
        )
        == fast._canonical(
            live._aegis_output_projection(first_treatment, historical=False)
        )
    )


def _video_complete(video: Mapping[str, Any], action_count: int) -> bool:
    return bool(
        video.get("decoded_successfully") is True
        and video.get("real_simulation_frames") is True
        and video.get("two_dimensional_safety_overlay") is False
        and video.get("source_action_index_start") == -1
        and video.get("source_action_index_end") == action_count - 1
        and video.get("frame_count") == action_count + 1
        and video.get("decoded_frame_count") == action_count + 1
    )


def _registered_contact_summary(arm: Mapping[str, Any]) -> Mapping[str, Any]:
    value = arm.get("registered_forbidden_contact")
    return value if isinstance(value, Mapping) else {}


def _metric_inputs(
    *,
    protocol: Mapping[str, Any],
    derived: Mapping[str, Any],
    baseline: Mapping[str, Any],
    treatment: Mapping[str, Any],
    providers: Mapping[str, Mapping[str, Any]],
    first_query_cache: Mapping[str, Any],
    videos: Mapping[str, Mapping[str, Any]],
    protected_body_names: Sequence[str],
    settled_state_hash: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    action_count = int(derived["horizon_action_count"])
    query_count = (action_count + 4) // 5
    expected_updates = action_count * 5
    expected_substeps = expected_updates * 5
    thresholds = derived["thresholds"]

    baseline_provider = providers["baseline"]
    treatment_provider = providers["psf"]
    baseline_contact = baseline["literal_contact"]
    treatment_contact = treatment["literal_contact"]
    baseline_registered = _registered_contact_summary(baseline)
    treatment_registered = _registered_contact_summary(treatment)
    baseline_contact_boundary = baseline_contact.get(
        "first_link56_physical_boundary"
    )
    material_rows = [
        row
        for row in treatment["activation_trace"]
        if row["filter_correction_l2_rad_s"]
        >= float(thresholds["minimum_correction_norm_rad_s"])
        and row["nominal_within_dynamic_joint_bounds"] is True
        and (
            baseline_contact_boundary is None
            or int(row["physical_boundary"]) < int(baseline_contact_boundary)
        )
    ]
    first_correction = (
        None if not material_rows else int(material_rows[0]["physical_boundary"])
    )
    first_task_success_source_action = treatment["task"].get(
        "first_task_success_source_action_index"
    )
    first_task_success_end_boundary = (
        None
        if first_task_success_source_action is None
        else (int(first_task_success_source_action) + 1) * 25
    )
    post_motion = fast._post_correction_motion(
        treatment["command_trace"],
        treatment["physics_trace"],
        first_correction_physical_boundary=first_correction,
        end_physical_boundary_exclusive=first_task_success_end_boundary,
        zero_command_norm_threshold_rad_s=float(
            thresholds["minimum_correction_norm_rad_s"]
        ),
    )
    task_rows_after_correction = [
        row
        for row in treatment["task"]["goal_progress_ledger"]
        if first_correction is not None
        and row.get("snapshot_kind") == "completed_high_level_post_step"
        and row.get("all_satisfied") is True
        and (int(row["source_action_index"]) + 1) * 25 > first_correction
    ]
    shared_links = list(protected_body_names) == [
        "robot0_link5",
        "robot0_link6",
    ]
    provider_complete = bool(
        _provider_complete(
            baseline_provider,
            action_count=action_count,
            query_count=query_count,
        )
        and _provider_complete(
            treatment_provider,
            action_count=action_count,
            query_count=query_count,
        )
    )
    qp_complete = bool(
        treatment["qp_solve_count"] == expected_updates
        and treatment["qp_postcheck_count"] == expected_updates
        and treatment["joint_limit_postcheck_count"] == expected_updates
        and treatment["precontact_execution_valid"] is True
    )
    full_exposure = bool(
        baseline["exposure_complete"] is True
        and treatment["exposure_complete"] is True
        and baseline["filter_update_count"] == expected_updates
        and treatment["filter_update_count"] == expected_updates
        and baseline["physics_substep_count"] == expected_substeps
        and treatment["physics_substep_count"] == expected_substeps
        and baseline["task"]["completed_source_action_count"] == action_count
        and treatment["task"]["completed_source_action_count"] == action_count
    )
    useful_motion = bool(
        post_motion["filter_correction_integral_rad"]
        >= float(thresholds["minimum_correction_integral_rad"])
        and post_motion["measured_joint_motion_integral_rad"]
        >= float(
            thresholds["minimum_post_correction_measured_joint_motion_integral_rad"]
        )
        and post_motion["cartesian_path_length_m"]
        >= float(thresholds["minimum_post_correction_eef_path_length_m"])
        and post_motion["executed_command_integral_rad"]
        >= float(
            thresholds["minimum_post_correction_executed_command_integral_rad"]
        )
        and post_motion["zero_command_fraction"]
        <= float(thresholds["maximum_post_correction_zero_command_fraction"])
    )
    treatment_no_registered_contact = bool(
        treatment_registered.get("settled_forbidden_contact") is False
        and treatment_registered.get("rollout_forbidden_contact") is False
        and treatment_registered.get("any_registered_forbidden_contact") is False
        and treatment_contact["any_robot_selected_obstacle_present"] is False
    )
    baseline_reproduced = bool(
        baseline_contact["link56_present"] is True
        and baseline_contact["first_link56_physical_boundary"]
        == baseline_contact["first_any_robot_physical_boundary"]
    )
    exact_settled_pair = bool(
        fast._pair_exact(baseline, treatment)
        and baseline["restore"]["settled_state_sha256"] == settled_state_hash
        and treatment["restore"]["settled_state_sha256"] == settled_state_hash
    )
    first_query_paired = bool(
        first_query_cache.get("contract_valid") is True
        and first_query_cache.get("first_query_index") == 0
    )
    released_aegis_valid = bool(
        provider_complete
        and baseline_provider.get(
            "first_aegis_input_binding_matches_historical"
        )
        is True
        and treatment_provider.get(
            "first_aegis_input_binding_matches_historical"
        )
        is True
        and _first_aegis_pair_exact(baseline_provider, treatment_provider)
    )
    own_observations_valid = bool(
        live._own_observation_chain_valid(baseline_provider, baseline)
        and live._own_observation_chain_valid(treatment_provider, treatment)
    )
    contact_cadence_valid = bool(
        baseline_registered.get("observed_physics_substeps")
        == expected_substeps
        and treatment_registered.get("observed_physics_substeps")
        == expected_substeps
        and baseline["physics_trace_row_count"] == expected_substeps
        and treatment["physics_trace_row_count"] == expected_substeps
        and all(
            "registered_forbidden_contact_observed" in row
            for arm in (baseline, treatment)
            for row in arm["physics_trace"]
        )
    )
    hard_rows_enforced = bool(
        qp_complete
        and treatment["pre_filter_field_observation_count"]
        == expected_updates
        and treatment["invalid_field_query_count"] == 0
        and treatment["nonpositive_post_state_field_query_count"] == 0
    )
    baseline_tracking = baseline["tracking_precontact"]
    treatment_tracking = treatment["tracking_full_window"]
    tracking_valid = bool(
        baseline_tracking["observation_count"] > 0
        and baseline_tracking["linf_rad_s"]
        <= float(thresholds["maximum_joint_velocity_tracking_linf_rad_s"])
        and baseline_tracking["rmse_rad_s"]
        <= float(thresholds["maximum_joint_velocity_tracking_rmse_rad_s"])
        and treatment_tracking["observation_count"] == expected_substeps
        and treatment_tracking["linf_rad_s"]
        <= float(thresholds["maximum_joint_velocity_tracking_linf_rad_s"])
        and treatment_tracking["rmse_rad_s"]
        <= float(thresholds["maximum_joint_velocity_tracking_rmse_rad_s"])
    )
    paper_car_complete = bool(
        baseline["paper_car"]["endpoint_ledger_complete"] is True
        and treatment["paper_car"]["endpoint_ledger_complete"] is True
        and len(baseline["paper_car"]["endpoint_ledger"]) == action_count
        and len(treatment["paper_car"]["endpoint_ledger"]) == action_count
    )
    metrics = {
        "exact_settled_pair_start": exact_settled_pair,
        "baseline_exposure_complete": bool(
            baseline["exposure_complete"] is True
        ),
        "treatment_exposure_complete": bool(
            treatment["exposure_complete"] is True
        ),
        "first_live_policy_query_paired": first_query_paired,
        "live_policy_contract_valid": bool(
            provider_complete
            and _video_complete(videos["baseline"], action_count)
            and _video_complete(videos["psf"], action_count)
        ),
        "released_aegis_contract_valid": released_aegis_valid,
        "own_observation_chains_valid": own_observations_valid,
        "direct_joint_velocity_controller_used": bool(
            _controller_is_direct_joint_velocity(baseline)
            and _controller_is_direct_joint_velocity(treatment)
        ),
        "fixed_link56_protected_set_used": shared_links,
        "hard_psf_rows_enforced": hard_rows_enforced,
        "shared_nominal_joint_bounds_inactive": bool(
            baseline["all_nominal_commands_within_dynamic_joint_bounds"]
            is True
            and treatment["all_nominal_commands_within_dynamic_joint_bounds"]
            is True
        ),
        "joint_velocity_tracking_valid": tracking_valid,
        "no_unregistered_fallback_executed": bool(
            full_exposure
            and baseline_provider.get("recorded_suffix_actions_executed")
            is False
            and treatment_provider.get("recorded_suffix_actions_executed")
            is False
        ),
        "every_physics_substep_contact_checked": contact_cadence_valid,
        "paper_car_endpoint_ledger_complete": paper_car_complete,
        "baseline_link56_contact_present": baseline_reproduced,
        "treatment_any_robot_selected_obstacle_contact_present": bool(
            treatment_contact["any_robot_selected_obstacle_present"]
            or treatment_registered.get(
                "any_robot_selected_obstacle_contact", False
            )
        ),
        "treatment_link56_shifted_external_contact_present": bool(
            treatment_registered.get(
                "any_link56_nonselected_external_contact", False
            )
        ),
        "treatment_method_stop": bool(
            treatment.get("method_terminated_early") is True
            and isinstance(treatment.get("method_stop"), Mapping)
        ),
        "treatment_stalled_after_correction": not useful_motion,
        "treatment_native_task_success": bool(
            treatment["task"]["ever_task_success_at_or_after_branch"]
            and treatment["task"]["terminal_task_success"]
        ),
        "treatment_task_success_after_correction": bool(
            task_rows_after_correction
        ),
        "treatment_paper_car_avoided": bool(
            treatment["paper_car"]["paper_collision_avoidance"] is True
        ),
        "material_correction_present": bool(material_rows),
        "material_correction_before_baseline_contact": bool(
            baseline_contact_boundary is not None and material_rows
        ),
        "maximum_correction_norm_rad_s": post_motion[
            "maximum_correction_norm_rad_s"
        ],
        "correction_integral_rad": post_motion[
            "filter_correction_integral_rad"
        ],
        "post_correction_measured_joint_motion_integral_rad": post_motion[
            "measured_joint_motion_integral_rad"
        ],
        "post_correction_eef_path_length_m": post_motion[
            "cartesian_path_length_m"
        ],
        "post_correction_executed_command_integral_rad": post_motion[
            "executed_command_integral_rad"
        ],
        "post_correction_zero_command_fraction": post_motion[
            "zero_command_fraction"
        ],
    }
    return metrics, post_motion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/vlsa_poisson_link56_aegis_car_109.v1.jsonl"),
    )
    parser.add_argument("--historical-result-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", required=True, type=int)
    arguments = parser.parse_args()

    root = arguments.repo_root.resolve()
    os.chdir(str(root))
    for path in (root, root / "main", root / "safelibero"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    output = None
    result_path = None
    source_env = None
    started = time.time()
    provenance: Dict[str, Any] = {}
    arms: Dict[str, Any] = {}
    policy_records: Dict[str, Any] = {}
    videos: Dict[str, Any] = {}
    try:
        import numpy as np
        import main.evaluate_safelibero_aegis as evaluator
        from main.poisson_fullbody.contracts import publish_hashed_json
        from main.poisson_fullbody.direct_joint_velocity_feasibility import (
            BASELINE_ARM,
            PSF_ARM,
            RESULT_SCHEMA,
            classify_direct_joint_velocity,
            validate_direct_joint_velocity_protocol,
        )
        from main.poisson_fullbody.feasibility_protocol import (
            load_feasibility_protocol,
        )
        from main.poisson_fullbody.field_bundle import build_static_field_bundle
        from main.poisson_fullbody.measurement import (
            clone_forwarded_state,
            resolve_collision_geom_sets,
        )
        from main.poisson_fullbody.robot_samples import validate_rigid_roundtrip
        from main.poisson_fullbody.shadow_replay import (
            load_historical_action_replay,
        )
        from main.poisson_fullbody.surface_sampling import (
            build_robot_collision_samples,
            validate_robot_sample_evidence,
        )
        from scripts.run_poisson_shadow_parity import _prepare_environment

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
        protocol = fast._json(protocol_path, "direct joint-velocity protocol")
        derived = validate_direct_joint_velocity_protocol(protocol)
        if arguments.case_id != derived["case_id"]:
            raise DirectJointVelocityRunnerError("CLI and protocol case IDs differ")
        case, case_row_sha256 = _manifest_case(manifest_path, arguments.case_id)
        case_contract = protocol["case"]
        _validate_manifest_case_identity(case, case_contract)
        if (
            fast._file_sha256(manifest_path)
            != protocol["selection_binding"]["manifest_sha256"]
            or case.get("protocol_config_sha256")
            != protocol["selection_binding"]["protocol_config_sha256"]
            or case["historical_aegis_result"]["pairing"][
                "manifest_row_sha256"
            ]
            != protocol["selection_binding"]["source_manifest_row_sha256"]
        ):
            raise DirectJointVelocityRunnerError("manifest or selection binding differs")
        selection_path = root / protocol["selection_binding"][
            "protocol_config_path"
        ]
        if (
            fast._file_sha256(selection_path)
            != protocol["selection_binding"]["protocol_config_sha256"]
        ):
            raise DirectJointVelocityRunnerError("selection file hash differs")

        runtime_path = root / protocol["runtime_binding"]["path"]
        if (
            fast._file_sha256(runtime_path)
            != protocol["runtime_binding"]["file_sha256"]
        ):
            raise DirectJointVelocityRunnerError("runtime file hash differs")
        runtime_protocol, runtime_hashes = load_feasibility_protocol(
            runtime_path,
            expected_protocol_sha256=protocol["runtime_binding"][
                "semantic_sha256"
            ],
        )
        if (
            runtime_hashes.parameter_block_sha256
            != protocol["runtime_binding"]["parameter_block_sha256"]
        ):
            raise DirectJointVelocityRunnerError("runtime parameter hash differs")

        historical_path = fast._historical_path(
            arguments.historical_result_root.resolve(), case
        )
        replay = load_historical_action_replay(
            historical_path,
            expected_case_id=arguments.case_id,
            expected_arm="pi05_plus_aegis_translational",
        )
        source_contract = protocol["historical_evaluation"]
        if (
            replay.result_file_sha256
            != source_contract["historical_result_file_sha256"]
            or replay.result_payload_sha256
            != source_contract["historical_result_payload_sha256"]
            or replay.executed_sequence_sha256
            != source_contract["historical_executed_action_sequence_sha256"]
            or len(replay.actions) != derived["horizon_action_count"]
        ):
            raise DirectJointVelocityRunnerError("historical source result differs")
        historical_result = fast._json(historical_path, "historical AEGIS result")
        source = fast._git(root)
        allocation = fast._allocation()
        checkpoint_identity = live._checkpoint_identity(protocol["online_policy"])
        provenance = {
            "source": source,
            "allocation": allocation,
            "python": platform.python_version(),
            "run_id": arguments.run_id,
            "case_id": arguments.case_id,
            "case_row_sha256": case_row_sha256,
            "protocol_file_sha256": fast._file_sha256(protocol_path),
            "historical_result_file_sha256": replay.result_file_sha256,
            "historical_result_payload_sha256": replay.result_payload_sha256,
            "historical_executed_action_sequence_sha256": (
                replay.executed_sequence_sha256
            ),
            "policy_checkpoint_identity": checkpoint_identity,
        }

        runtime = evaluator._runtime_imports(include_aegis=True)
        source_env, task, observation, _, previous_goal = _prepare_environment(
            evaluator, runtime, case, replay
        )
        settled_state = np.asarray(
            source_env.sim.get_state().flatten(), dtype=np.float64
        ).copy()
        settled_state_hash = evaluator.array_sha256(settled_state)
        if (
            settled_state_hash
            != case["historical_aegis_result"]["pairing"][
                "settled_simulator_state_sha256"
            ]
        ):
            raise DirectJointVelocityRunnerError("settled state hash differs")
        expected_goal_values = (False,)
        if tuple(previous_goal) != expected_goal_values:
            raise DirectJointVelocityRunnerError("initial task goal state differs")

        obstacle_name, _ = evaluator._active_obstacle(source_env, observation)
        if (
            obstacle_name != protocol["case"]["selected_obstacle_name"]
            or obstacle_name != case["active_obstacle_name"]
        ):
            raise DirectJointVelocityRunnerError("selected obstacle differs")
        authority = evaluator._contact_model_authority(source_env, obstacle_name)
        robot_root = fast._body_id(
            source_env.sim.model, source_env.robots[0].robot_model.root_body
        )
        protected_body_names = list(
            protocol["execution"]["protected_robot_body_names"]
        )
        link_ids = tuple(
            fast._body_id(source_env.sim.model, name)
            for name in protected_body_names
        )
        raw_model, raw_data = fast._raw_model_data(source_env.sim)
        resolved = resolve_collision_geom_sets(
            raw_model,
            robot_root_body_ids=(robot_root,),
            obstacle_root_body_ids=(
                int(authority["active_obstacle_root_body_id"]),
            ),
            link56_body_ids=link_ids,
        )
        official_before_field = fast._official_state(source_env.sim)
        bundle = build_static_field_bundle(
            raw_model,
            raw_data,
            resolved=resolved,
            protocol=runtime_protocol,
            protocol_hashes=runtime_hashes,
        )
        if not np.array_equal(
            official_before_field, fast._official_state(source_env.sim)
        ):
            raise DirectJointVelocityRunnerError(
                "field construction changed the settled state"
            )
        source_arm_qpos_indexes = tuple(
            int(value) for value in source_env.robots[0]._ref_joint_pos_indexes
        )
        if len(source_arm_qpos_indexes) != 7:
            raise DirectJointVelocityRunnerError(
                "settled Panda joint-position index contract differs"
            )
        settled_arm_qpos = np.asarray(
            source_env.sim.data.qpos[list(source_arm_qpos_indexes)],
            dtype=np.float64,
        )
        joint_q_min, joint_q_max = fast._joint_limits(source_env)
        joint_margin = float(
            runtime_protocol["qp"]["joint_position_margin_rad"]
        )
        joints_inside_margin = bool(
            np.all(settled_arm_qpos >= joint_q_min + joint_margin)
            and np.all(settled_arm_qpos <= joint_q_max - joint_margin)
        )
        strict_safe_samples = bool(bundle.diagnostics.minimum_initial_h_m2 > 0.0)
        physical_bounds_contain_zero = bool(
            all(
                float(lower) <= 0.0 <= float(upper)
                for lower, upper in zip(
                    runtime_protocol["qp"]["velocity_lower_rad_s"],
                    runtime_protocol["qp"]["velocity_upper_rad_s"],
                )
            )
        )
        zero_velocity_witness = {
            "schema_version": (
                "vlsa_poisson_direct_joint_velocity_feasibility_witness.v1"
            ),
            "minimum_settled_protected_h_m2": float(
                bundle.diagnostics.minimum_initial_h_m2
            ),
            "all_settled_protected_samples_strictly_safe": (
                strict_safe_samples
            ),
            "all_arm_joints_inside_registered_margin": joints_inside_margin,
            "physical_velocity_bounds_contain_zero": (
                physical_bounds_contain_zero
            ),
            "static_cbf_zero_velocity_residual_minimum_m2_per_s": float(
                runtime_protocol["cbf"]["alpha_gain_per_s"]
                * bundle.diagnostics.minimum_initial_h_m2
            ),
            "constructive_qp_feasible": bool(
                strict_safe_samples
                and joints_inside_margin
                and physical_bounds_contain_zero
            ),
            "zero_velocity_executed_as_fallback": False,
            "zero_velocity_counts_as_positive_outcome": False,
        }
        if zero_velocity_witness["constructive_qp_feasible"] is not True:
            raise DirectJointVelocityRunnerError(
                "settled direct joint-velocity QP lacks its constructive "
                "feasibility witness"
            )
        forwarded = clone_forwarded_state(raw_model, raw_data)
        full_samples = build_robot_collision_samples(
            source_env.sim.model,
            forwarded,
            geom_ids=resolved.robot_geom_ids,
            epsilon_m=float(runtime_protocol["coverage"]["epsilon_m"]),
        )
        roundtrip = validate_rigid_roundtrip(full_samples.samples, forwarded)
        full_sample_evidence = {
            "sample_count": len(full_samples.samples),
            "sample_ledger_sha256": full_samples.sample_ledger_sha256,
            "geom_records": list(full_samples.geom_records),
            "epsilon_m": full_samples.epsilon_m,
            "maximum_surface_cover_radius_m": (
                full_samples.maximum_surface_cover_radius_m
            ),
            "coverage_semantics": full_samples.coverage_semantics,
            "roundtrip": roundtrip,
        }
        validate_robot_sample_evidence(
            full_sample_evidence,
            resolved_geom_ids=resolved.robot_geom_ids,
            resolved_geom_names=resolved.robot_geom_names,
            resolved_body_ids=resolved.robot_body_ids,
            roundtrip_field="roundtrip",
        )
        settled_obstacle = fast._obstacle_state(
            raw_model, forwarded, bundle, resolved.obstacle_body_ids
        )
        if not fast._static_admissible([settled_obstacle], runtime_protocol):
            raise DirectJointVelocityRunnerError(
                "selected obstacle is not static at the settled boundary"
            )

        action_count = int(derived["horizon_action_count"])
        expected_updates = action_count * 5
        expected_substeps = expected_updates * 5
        query_count = (action_count + 4) // 5
        placeholder_actions = [[0.0] * 7 for _ in range(action_count)]
        paired_cache = live.PairedFirstQueryCache(first_query_index=0)
        server_metadata = None
        for (
            key,
            runner_arm,
            protocol_arm,
            slug,
            first_query_execution,
            paired_source,
        ) in (
            (
                "baseline",
                "joint_velocity_adapter_only",
                BASELINE_ARM,
                "adapter-only",
                live.FIRST_QUERY_LIVE_AND_CACHE,
                None,
            ),
            (
                "psf",
                "joint_velocity_adapter_plus_link56_psf",
                PSF_ARM,
                "adapter-plus-link56-poisson-cbf",
                live.FIRST_QUERY_PAIRED_CACHE_REUSE,
                BASELINE_ARM,
            ),
        ):
            client = runtime["websocket_client_policy"].WebsocketClientPolicy(
                arguments.host, arguments.port
            )
            metadata = evaluator._server_identity(client)
            if metadata.get("status") != "available":
                raise DirectJointVelocityRunnerError(
                    "pi0.5 server metadata is unavailable"
                )
            if server_metadata is None:
                server_metadata = metadata
            elif fast._canonical(server_metadata) != fast._canonical(metadata):
                raise DirectJointVelocityRunnerError(
                    "pi0.5 server metadata changed between arms"
                )
            provider = live.LiveAegisPolicy(
                evaluator=evaluator,
                runtime=runtime,
                client=client,
                case=case,
                task_description=str(task.language),
                historical_result=historical_result,
                first_query_index=0,
                replan_steps=int(case["replan_steps"]),
                model_action_horizon=int(case["model_action_horizon"]),
                historical_first_chunk_sha256_diagnostic=str(
                    historical_result["policy_queries"][0][
                        "returned_actions_sha256"
                    ]
                ),
                paired_first_query_cache=paired_cache,
                first_query_execution=first_query_execution,
                paired_first_query_source_arm=paired_source,
                arm_name=protocol_arm,
                source_start_action=0,
            )
            video = live.RolloutVideo(
                evaluator=evaluator,
                imageio=runtime["imageio"],
                run_root=output,
                arm_slug=slug,
                fps=int(protocol["videos"]["fps"]),
            )
            try:
                arm = fast._run_arm(
                    arm_name=runner_arm,
                    evaluator=evaluator,
                    runtime=runtime,
                    case=case,
                    source_env=source_env,
                    state_at_B=settled_state,
                    actions=placeholder_actions,
                    source_start_action=0,
                    bundle=bundle,
                    resolved=resolved,
                    full_samples=full_samples,
                    runtime_protocol=runtime_protocol,
                    nominal_activation_threshold_m2_per_s=-float(
                        runtime_protocol["qp"]["postcheck_cbf_tolerance"]
                    ),
                    qp_max_iterations=int(
                        runtime_protocol["qp"]["max_iterations"]
                    ),
                    expected_filter_updates=expected_updates,
                    expected_physics_substeps=expected_substeps,
                    expected_boundary_goal_values=expected_goal_values,
                    live_action_provider=provider,
                    rollout_frame_observer=video,
                    initial_live_observation=copy.deepcopy(observation),
                    full_clearance_observation_stride=25,
                    monitor_registered_forbidden_contacts=True,
                    record_paper_car=True,
                    record_qp_failure_as_method_stop=(key == "psf"),
                )
                arms[key] = arm
            except fast.FastArmExecutionFailure as error:
                arms[key] = dict(error.evidence)
                raise
            finally:
                active_exception = sys.exc_info()[0] is not None
                finalization_error = None
                try:
                    policy_records[key] = provider.record()
                except Exception as error:
                    policy_records[key] = {
                        "status": "finalization_failure",
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                    }
                    finalization_error = error
                try:
                    videos[key] = video.close()
                except Exception as error:
                    videos[key] = {
                        "status": "finalization_failure",
                        "failure_type": type(error).__name__,
                        "failure_message": str(error),
                    }
                    if finalization_error is None:
                        finalization_error = error
                if not active_exception and finalization_error is not None:
                    raise finalization_error

        cache_record = paired_cache.record()
        metrics, post_motion = _metric_inputs(
            protocol=protocol,
            derived=derived,
            baseline=arms["baseline"],
            treatment=arms["psf"],
            providers=policy_records,
            first_query_cache=cache_record,
            videos=videos,
            protected_body_names=protected_body_names,
            settled_state_hash=settled_state_hash,
        )
        classification = classify_direct_joint_velocity(metrics, protocol)
        protected_sample_rows = [
            sample.to_dict() for sample in bundle.protected_samples.samples
        ]
        implementation = {
            "producer_module": (
                "scripts.run_poisson_direct_joint_velocity_feasibility"
            ),
            "protocol_module": (
                "main.poisson_fullbody.direct_joint_velocity_feasibility"
            ),
            "cbf_qp_module": "main.poisson_fullbody.cbf_qp",
            "joint_velocity_adapter_module": (
                "main.poisson_fullbody.joint_velocity_adapter"
            ),
            "contact_monitor_module": (
                "main.poisson_fullbody.registered_contact_monitor"
            ),
            "arm_runner_module": "scripts.run_poisson_fast_feasibility",
            "controller": "JOINT_VELOCITY",
            "decision_variable": "joint_velocity_rad_s",
            "runtime_imports": [
                "main.poisson_fullbody.cbf_qp",
                "main.poisson_fullbody.joint_velocity_adapter",
                "main.poisson_fullbody.registered_contact_monitor",
            ],
            "source_sha256": {
                "producer": fast._file_sha256(Path(__file__).resolve()),
                "protocol": fast._file_sha256(
                    root
                    / "main/poisson_fullbody/direct_joint_velocity_feasibility.py"
                ),
                "arm_runner": fast._file_sha256(
                    root / "scripts/run_poisson_fast_feasibility.py"
                ),
                "cbf_qp": fast._file_sha256(
                    root / "main/poisson_fullbody/cbf_qp.py"
                ),
                "joint_velocity_adapter": fast._file_sha256(
                    root / "main/poisson_fullbody/joint_velocity_adapter.py"
                ),
                "contact_monitor": fast._file_sha256(
                    root / "main/poisson_fullbody/registered_contact_monitor.py"
                ),
            },
        }
        provider_evidence = {
            "implementation": implementation,
            "adapter_only": policy_records["baseline"],
            "adapter_plus_psf": policy_records["psf"],
        }
        candidate = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "protocol_id": protocol["protocol_id"],
            "run_id": arguments.run_id,
            "case_id": arguments.case_id,
            "provenance": provenance,
            "controller_contract": {
                "controller": "JOINT_VELOCITY",
                "system_model": "qdot_equals_v",
                "decision_variable": "joint_velocity_rad_s",
                "cbf_constraint": "grad_h_dot_J_qdot_plus_alpha_h_ge_0",
                "static_obstacle_partial_t_h": 0.0,
                "control_frequency_hz": 100,
                "physics_frequency_hz": 500,
                "controller_updates_per_high_level_action": 5,
                "physics_substeps_per_controller_update": 5,
                "hard_constraints": True,
                "safety_slack_used": False,
                "fallback_policy": "none_fail_closed_as_negative",
                "zero_velocity_is_feasibility_witness_not_acceptance": True,
                "acceptance_thresholds": {
                    "cbf_postcheck_tolerance_m2_per_s": float(
                        runtime_protocol["qp"]["postcheck_cbf_tolerance"]
                    ),
                    "velocity_bound_tolerance_rad_s": float(
                        runtime_protocol["qp"][
                            "postcheck_bound_tolerance_rad_s"
                        ]
                    ),
                    "material_correction_minimum_rad_s": float(
                        derived["thresholds"][
                            "minimum_correction_norm_rad_s"
                        ]
                    ),
                    "material_correction_integral_minimum_rad": float(
                        derived["thresholds"][
                            "minimum_correction_integral_rad"
                        ]
                    ),
                    "post_correction_joint_motion_minimum_rad": float(
                        derived["thresholds"][
                            "minimum_post_correction_measured_joint_motion_integral_rad"
                        ]
                    ),
                    "post_correction_eef_path_minimum_m": float(
                        derived["thresholds"][
                            "minimum_post_correction_eef_path_length_m"
                        ]
                    ),
                    "post_correction_executed_integral_minimum_rad": float(
                        derived["thresholds"][
                            "minimum_post_correction_executed_command_integral_rad"
                        ]
                    ),
                    "post_correction_zero_command_fraction_maximum": float(
                        derived["thresholds"][
                            "maximum_post_correction_zero_command_fraction"
                        ]
                    ),
                    "tracking_linf_maximum_rad_s": float(
                        derived["thresholds"][
                            "maximum_joint_velocity_tracking_linf_rad_s"
                        ]
                    ),
                    "tracking_rmse_maximum_rad_s": float(
                        derived["thresholds"][
                            "maximum_joint_velocity_tracking_rmse_rad_s"
                        ]
                    ),
                },
            },
            "policy_server": server_metadata,
            "pairing": {
                "settled_state_sha256": settled_state_hash,
                "first_query_cache": cache_record,
                "same_initial_state": metrics["exact_settled_pair_start"],
                "same_first_policy_query": metrics[
                    "first_live_policy_query_paired"
                ],
            },
            "field": {
                "protected_robot_body_names": protected_body_names,
                "selection_rule": (
                    "fixed_shared_link5_link6_surface_union_for_all_cases"
                ),
                "task_conditioned_link_selection": False,
                "historical_target_link_is_evaluation_only": True,
                "historical_target_link_body_names": list(
                    case["selection_evidence"][
                        "literal_protected_link_contact_bodies"
                    ]
                ),
                "bundle_hashes": asdict(bundle.hashes),
                "diagnostics": asdict(bundle.diagnostics),
                "resolved_geometry": resolved.to_dict(),
                "protected_samples": {
                    "sample_count": len(protected_sample_rows),
                    "sample_ledger_sha256": fast._sha256(
                        fast._canonical(protected_sample_rows)
                    ),
                    "samples": protected_sample_rows,
                },
                "full_robot_sampling": full_sample_evidence,
                "settled_selected_obstacle_state": settled_obstacle,
                "zero_velocity_feasibility_witness": zero_velocity_witness,
            },
            "providers": provider_evidence,
            "arms": {
                "adapter_only": arms["baseline"],
                "adapter_plus_psf": arms["psf"],
            },
            "videos": videos,
            "metrics": metrics,
            "post_correction_motion": post_motion,
            "classification": classification,
            "producer_classification_is_preliminary": True,
            "scientific_interpretation_requires_independent_consumer": True,
            "partial_output_interpreted": False,
            "claim_scope": protocol["result_contract"]["claim_scope"],
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
                    "classification": classification["classification"],
                    "feasible": classification["feasible"],
                    "result": str(result_path),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        if output is None:
            try:
                output = fast._new_output(
                    arguments.output_root.resolve(), arguments.run_id
                )
                result_path = output / "result.json"
            except Exception:
                output = None
        if result_path is not None and not result_path.exists():
            try:
                from main.poisson_fullbody.contracts import publish_hashed_json
                from main.poisson_fullbody.direct_joint_velocity_feasibility import (
                    RESULT_SCHEMA,
                )

                publish_hashed_json(
                    result_path,
                    {
                        "schema_version": RESULT_SCHEMA,
                        "status": "apparatus_or_method_failure",
                        "run_id": arguments.run_id,
                        "case_id": arguments.case_id,
                        "provenance": provenance,
                        "arms": arms,
                        "providers": policy_records,
                        "videos": videos,
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
                    "failed to publish direct failure artifact: %s"
                    % publication_error,
                    file=sys.stderr,
                )
        print(
            "direct joint-velocity feasibility failed: %s" % error,
            file=sys.stderr,
        )
        return 2
    finally:
        if source_env is not None:
            source_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
