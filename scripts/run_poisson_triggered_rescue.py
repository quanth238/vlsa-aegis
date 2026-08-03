#!/usr/bin/env python3
"""Run the minimal event-triggered direct-qdot Poisson-CBF rescue.

The source SafeLIBERO episode is replayed under unchanged native OSC.  Before
each 20 Hz source action, the runner constructs the same bounded first
joint-velocity command used by the accepted direct-qdot adapter and evaluates
the link-5/link-6 Poisson CBF.  The first candidate with an unsafe nominal CBF
residual and a material solved hard-QP correction becomes the trigger.  No
case-specific action index participates in that decision.

At the trigger state, two fresh JOINT_VELOCITY arms receive the identical
remaining recorded action suffix: adapter-only versus adapter plus the hard
Poisson-CBF QP.  A positive result still requires baseline contact, zero
original or shifted treatment contact, paper-CAR safety, useful motion, and
native task success.  This is controller-feasibility evidence, not a
closed-loop policy or population claim.
"""

from __future__ import annotations

import argparse
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


def _manifest_case(path: Path, case_id: str, expected_row_sha256: str) -> Tuple[Dict[str, Any], str]:
    from scripts import run_poisson_fast_feasibility as fast

    if path.is_symlink() or not path.is_file():
        raise fast.FastRunnerError("registered case manifest is unavailable")
    matches = []
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            value = json.loads(raw)
            if value.get("case_id") == case_id:
                matches.append((value, fast._sha256(raw.rstrip(b"\r\n"))))
    if len(matches) != 1:
        raise fast.FastRunnerError("manifest must contain exactly one requested case")
    case, digest = matches[0]
    if digest != expected_row_sha256:
        raise fast.FastRunnerError("manifest case-row hash differs")
    if case.get("settle_actions") != 20:
        raise fast.FastRunnerError("triggered rescue requires 20 settling actions")
    if case.get("study_partition") not in ("development", "heldout_evaluation"):
        raise fast.FastRunnerError("triggered rescue case partition differs")
    return case, digest


def _sample_records(samples: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "sample_id": int(sample.sample_id),
            "body_id": int(sample.body_id),
            "body_name": str(sample.body_name),
            "geom_id": int(sample.geom_id),
            "geom_name": str(sample.geom_name),
        }
        for sample in samples
    ]


def _post_correction_metrics(
    arm: Mapping[str, Any],
    *,
    first_boundary: Any,
) -> Dict[str, float]:
    from scripts import run_poisson_fast_feasibility as fast

    if first_boundary is None:
        return {
            "filter_correction_integral_rad": 0.0,
            "measured_joint_motion_integral_rad": 0.0,
            "cartesian_path_length_m": 0.0,
            "executed_command_integral_rad": 0.0,
            "zero_command_fraction": 1.0,
            "maximum_correction_norm_rad_s": 0.0,
        }
    return fast._post_correction_motion(
        arm["command_trace"],
        arm["physics_trace"],
        first_correction_physical_boundary=int(first_boundary),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/vlsa_poisson_triggered_rescue.v1.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/vlsa_poisson_link56_aegis_car_109.v1.jsonl"),
    )
    parser.add_argument("--historical-result-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id", default="vlsa-t1-goal-ii-t0-e05")
    arguments = parser.parse_args()

    root = arguments.repo_root.resolve()
    os.chdir(str(root))
    for path in (root, root / "main", root / "safelibero"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from scripts import run_poisson_fast_feasibility as fast
    from main.poisson_fullbody.triggered_rescue import (
        METRICS_SCHEMA,
        RESULT_SCHEMA,
        classify_triggered_rescue,
        evaluate_direct_qdot_trigger_candidate,
        validate_triggered_rescue_protocol,
    )

    try:
        output = fast._new_output(arguments.output_root, arguments.run_id)
    except fast.FastRunnerError as error:
        print("triggered rescue refused: %s" % error, file=sys.stderr)
        return 2
    result_path = output / "result.json"
    started = time.time()
    source_env = None
    provenance: Dict[str, Any] = {}
    trigger_rows: List[Dict[str, Any]] = []
    prefix_rows: List[Dict[str, Any]] = []
    arm_evidence: Dict[str, Any] = {}
    case: Dict[str, Any] = {}
    protocol: Dict[str, Any] = {}
    selected_protocol_id: Any = None
    try:
        import numpy as np
        from main.poisson_fullbody.cbf_qp import HardCbfQp, joint_velocity_bounds
        from main.poisson_fullbody.contracts import publish_hashed_json
        from main.poisson_fullbody.feasibility_protocol import load_feasibility_protocol
        from main.poisson_fullbody.field_bundle import build_static_field_bundle
        from main.poisson_fullbody.jacobians import evaluate_point_jacobians
        from main.poisson_fullbody.joint_velocity_adapter import TranslationalJointVelocityAdapter
        from main.poisson_fullbody.measurement import clone_forwarded_state, resolve_collision_geom_sets
        from main.poisson_fullbody.robot_samples import validate_rigid_roundtrip
        from main.poisson_fullbody.shadow_replay import load_historical_action_replay
        from main.poisson_fullbody.surface_sampling import (
            build_robot_collision_samples,
            validate_robot_sample_evidence,
        )
        from scripts.run_poisson_shadow_parity import (
            _check_step,
            _observation_sha256,
            _prepare_environment,
        )
        import main.evaluate_safelibero_aegis as evaluator

        protocol_path = arguments.protocol if arguments.protocol.is_absolute() else root / arguments.protocol
        manifest_path = arguments.manifest if arguments.manifest.is_absolute() else root / arguments.manifest
        protocol = fast._json(protocol_path.resolve(), "triggered-rescue protocol")
        selected_protocol_id = protocol.get("protocol_id")
        derived = validate_triggered_rescue_protocol(
            protocol, case_id=arguments.case_id
        )
        source = fast._git(root)
        allocation = fast._allocation()
        case, case_row_sha256 = _manifest_case(
            manifest_path.resolve(),
            arguments.case_id,
            str(derived["case"]["source_manifest_row_sha256"]),
        )
        if fast._file_sha256(manifest_path.resolve()) != protocol["selection_binding"]["manifest_sha256"]:
            raise fast.FastRunnerError("selection manifest raw hash differs")
        selection_path = root / protocol["selection_binding"]["protocol_config_path"]
        if fast._file_sha256(selection_path) != protocol["selection_binding"]["protocol_config_sha256"]:
            raise fast.FastRunnerError("selection protocol raw hash differs")
        if case.get("protocol_config_sha256") != protocol["selection_binding"]["protocol_config_sha256"]:
            raise fast.FastRunnerError("case-to-selection binding differs")

        runtime_path = root / protocol["runtime_binding"]["path"]
        if fast._file_sha256(runtime_path) != protocol["runtime_binding"]["file_sha256"]:
            raise fast.FastRunnerError("runtime protocol raw hash differs")
        runtime_protocol, runtime_hashes = load_feasibility_protocol(
            runtime_path,
            expected_protocol_sha256=protocol["runtime_binding"]["semantic_sha256"],
        )
        if runtime_hashes.parameter_block_sha256 != protocol["runtime_binding"]["parameter_block_sha256"]:
            raise fast.FastRunnerError("runtime parameter-block hash differs")

        historical_path = fast._historical_path(
            arguments.historical_result_root.resolve(), case
        )
        replay = load_historical_action_replay(
            historical_path,
            expected_case_id=arguments.case_id,
            expected_arm="pi05_plus_aegis_translational",
        )
        source_contract = derived["case"]
        if (
            replay.result_file_sha256 != source_contract["historical_result_file_sha256"]
            or replay.result_payload_sha256 != source_contract["historical_result_payload_sha256"]
            or replay.executed_sequence_sha256 != source_contract["historical_executed_action_sequence_sha256"]
            or len(replay.actions) != derived["horizon_action_count"]
        ):
            raise fast.FastRunnerError("historical replay differs from protocol")

        provenance = {
            "source": source,
            "allocation": allocation,
            "python": platform.python_version(),
            "run_id": arguments.run_id,
            "case_id": arguments.case_id,
            "case_row_sha256": case_row_sha256,
            "protocol_file_sha256": fast._file_sha256(protocol_path.resolve()),
            "historical_result_file_sha256": replay.result_file_sha256,
            "historical_result_payload_sha256": replay.result_payload_sha256,
            "historical_executed_action_sequence_sha256": replay.executed_sequence_sha256,
            "online_policy_query_count": 0,
        }

        runtime = evaluator._runtime_imports(include_aegis=False)
        source_env, _, observation, goal_atoms, previous_goal = _prepare_environment(
            evaluator, runtime, case, replay
        )
        obstacle_name, _ = evaluator._active_obstacle(source_env, observation)
        if (
            obstacle_name != source_contract["selected_obstacle_name"]
            or obstacle_name != case["active_obstacle_name"]
        ):
            raise fast.FastRunnerError("selected obstacle identity differs")
        paper_car_reference_position = np.asarray(
            observation["%s_pos" % obstacle_name], dtype=np.float64
        ).copy()
        if paper_car_reference_position.shape != (3,) or not np.all(
            np.isfinite(paper_car_reference_position)
        ):
            raise fast.FastRunnerError("settled paper-CAR reference is invalid")

        authority = evaluator._contact_model_authority(source_env, obstacle_name)
        robot_root = fast._body_id(
            source_env.sim.model, source_env.robots[0].robot_model.root_body
        )
        link_ids = tuple(
            fast._body_id(source_env.sim.model, name)
            for name in derived["protected_robot_body_names"]
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
        settled_official = fast._official_state(source_env.sim)
        bundle = build_static_field_bundle(
            raw_model,
            raw_data,
            resolved=resolved,
            protocol=runtime_protocol,
            protocol_hashes=runtime_hashes,
        )
        if not np.array_equal(settled_official, fast._official_state(source_env.sim)):
            raise fast.FastRunnerError("field construction changed simulator state")
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
            "maximum_surface_cover_radius_m": full_samples.maximum_surface_cover_radius_m,
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
        settled_obstacle_state = fast._obstacle_state(
            raw_model, forwarded, bundle, resolved.obstacle_body_ids
        )
        if not fast._static_admissible([settled_obstacle_state], runtime_protocol):
            raise fast.FastAssumptionInvalid(
                "selected obstacle is not static at field construction"
            )

        arm_dofs = tuple(
            int(value) for value in source_env.robots[0]._ref_joint_vel_indexes
        )
        arm_qpos = tuple(
            int(value) for value in source_env.robots[0]._ref_joint_pos_indexes
        )
        if len(arm_dofs) != 7 or len(arm_qpos) != 7:
            raise fast.FastRunnerError("Panda arm index contract differs")
        site_id = fast._eef_site_id(source_env)
        q_min, q_max = fast._joint_limits(source_env)
        adapter_cfg = runtime_protocol["adapter"]
        qp_cfg = runtime_protocol["qp"]
        qp = HardCbfQp(
            eps_abs=float(qp_cfg["eps_abs"]),
            eps_rel=float(qp_cfg["eps_rel"]),
            max_iter=int(derived["qp_max_iterations"]),
            postcheck_cbf_tolerance=float(qp_cfg["postcheck_cbf_tolerance"]),
            postcheck_bound_tolerance=float(
                qp_cfg["postcheck_bound_tolerance_rad_s"]
            ),
        )
        sample_records = _sample_records(bundle.protected_samples.samples)
        trigger = None
        obstacle_rows = [settled_obstacle_state]

        for source_index, expected_step in enumerate(replay.steps):
            if int(expected_step.step) != source_index:
                raise fast.FastRunnerError("historical source action index differs")
            source_action = np.asarray(expected_step.action, dtype=np.float64)
            if source_action.shape != (7,) or not np.all(np.isfinite(source_action)):
                raise fast.FastRunnerError("historical source action is invalid")
            if np.any(source_action[3:6] != 0.0):
                raise fast.FastRunnerError(
                    "translation-only trigger preview would discard rotation"
                )

            position, rotation, eef_jacobian = fast._eef_kinematics(
                source_env, site_id, arm_dofs
            )
            preview_adapter = TranslationalJointVelocityAdapter(
                damping=float(adapter_cfg["dls_damping"]),
                position_gain_s_inv=float(adapter_cfg["position_gain_per_s"]),
                orientation_gain_s_inv=float(
                    adapter_cfg["orientation_gain_per_s"]
                ),
            )
            preview_adapter.begin_high_level_action(
                source_action, position, rotation
            )
            measured_qdot = np.asarray(
                source_env.sim.data.qvel[list(arm_dofs)], dtype=np.float64
            )
            preview = preview_adapter.compute_inner_action(
                position,
                rotation,
                eef_jacobian,
                measured_joint_velocity=measured_qdot,
            )
            nominal = np.asarray(preview.qdot_physical, dtype=np.float64)
            current_forwarded = clone_forwarded_state(raw_model, raw_data)
            points, point_jacobians = evaluate_point_jacobians(
                raw_model,
                current_forwarded,
                bundle.protected_samples.samples,
                arm_dofs,
            )
            queries = [bundle.field.query(point) for point in points]
            invalid = [
                query
                for query in queries
                if not query.valid
                or query.value is None
                or query.gradient is None
            ]
            if invalid:
                raise fast.FastRunnerError(
                    "direct-qdot trigger field query is invalid before source action"
                )
            h = np.asarray([float(query.value) for query in queries], dtype=np.float64)
            if np.any(h <= 0.0):
                raise fast.FastRunnerError(
                    "direct-qdot trigger scan reached nonpositive h"
                )
            gradients = np.asarray(
                [query.gradient for query in queries], dtype=np.float64
            )
            rows = np.einsum("ni,nij->nj", gradients, point_jacobians)
            alpha = float(runtime_protocol["cbf"]["alpha_gain_per_s"])
            nominal_residuals = rows @ nominal + alpha * h
            nominal_minimum = float(np.min(nominal_residuals))
            scan_row: Dict[str, Any] = {
                "source_action_index": source_index,
                "physical_boundary": source_index * 25,
                "nominal_minimum_cbf_residual_m2_per_s": nominal_minimum,
                "nominal_threshold_m2_per_s": derived[
                    "nominal_trigger_threshold_m2_per_s"
                ],
                "bounded_nominal_qdot_rad_s": nominal.tolist(),
                "field_query_count": len(queries),
                "field_queries_all_valid_positive": True,
                "hard_qp_attempted": False,
                "triggered": False,
            }
            if nominal_minimum <= derived["nominal_trigger_threshold_m2_per_s"]:
                q = np.asarray(raw_data.qpos[list(arm_qpos)], dtype=np.float64)
                lower, upper = joint_velocity_bounds(
                    q,
                    q_min,
                    q_max,
                    qp_cfg["velocity_lower_rad_s"],
                    qp_cfg["velocity_upper_rad_s"],
                    alpha_joint=float(qp_cfg["joint_limit_alpha_per_s"]),
                    control_dt_seconds=0.01,
                    position_margin_rad=float(
                        qp_cfg["joint_position_margin_rad"]
                    ),
                )
                qp_result = qp.solve_from_field(
                    nominal,
                    h,
                    gradients,
                    point_jacobians,
                    lower,
                    upper,
                    alpha=alpha,
                    weight_diagonal=qp_cfg["weight_diagonal"],
                    require_safe_start=True,
                )
                scan_row["hard_qp_attempted"] = True
                scan_row["hard_qp_valid"] = bool(qp_result.valid)
                scan_row["hard_qp_reason"] = str(qp_result.reason)
                scan_row["hard_qp_diagnostics"] = dict(qp_result.diagnostics)
                if not qp_result.valid or qp_result.qdot_safe is None:
                    raise fast.FastRunnerError(
                        "first unsafe direct-qdot preview has no hard-QP command"
                    )
                safe = np.asarray(qp_result.qdot_safe, dtype=np.float64)
                candidate = evaluate_direct_qdot_trigger_candidate(
                    source_action_index=source_index,
                    physical_boundary=source_index * 25,
                    nominal_qdot=nominal,
                    safe_qdot=safe,
                    h=h,
                    gradients=gradients,
                    jacobians=point_jacobians,
                    alpha_per_s=alpha,
                    nominal_threshold_m2_per_s=derived[
                        "nominal_trigger_threshold_m2_per_s"
                    ],
                    material_correction_threshold_rad_s=derived[
                        "material_correction_threshold_rad_s"
                    ],
                    sample_records=sample_records,
                )
                scan_row.update(candidate)
                if candidate["triggered"]:
                    trigger = dict(scan_row)
                    trigger["native_goal_values_at_boundary"] = list(
                        previous_goal
                    )
                    trigger["simulator_state_sha256"] = evaluator.array_sha256(
                        source_env.sim.get_state().flatten()
                    )
                    trigger["official_state_raw_bytes_sha256"] = fast._sha256(
                        fast._official_state(source_env.sim).tobytes()
                    )
                    trigger_rows.append(dict(scan_row))
                    break
            trigger_rows.append(dict(scan_row))

            observation, reward, done, _ = source_env.step(expected_step.action)
            _, _, previous_goal = _check_step(
                evaluator=evaluator,
                env=source_env,
                observation=observation,
                reward=reward,
                done=done,
                expected_step=expected_step,
                goal_atoms=goal_atoms,
                previous_goal_values=previous_goal,
                np=np,
            )
            evaluator._update_eef_marker(
                source_env, evaluator._eef_proxy(runtime, observation)
            )
            prefix_forwarded = clone_forwarded_state(raw_model, raw_data)
            obstacle_state = fast._obstacle_state(
                raw_model,
                prefix_forwarded,
                bundle,
                resolved.obstacle_body_ids,
            )
            obstacle_rows.append(obstacle_state)
            if not fast._static_admissible(obstacle_rows, runtime_protocol):
                raise fast.FastAssumptionInvalid(
                    "selected obstacle left the static field before trigger"
                )
            prefix_rows.append(
                {
                    "source_action_index": source_index,
                    "returned_observation_sha256": _observation_sha256(
                        observation, evaluator, np
                    ),
                    "simulator_state_sha256": evaluator.array_sha256(
                        source_env.sim.get_state().flatten()
                    ),
                    "native_goal_values": list(previous_goal),
                    "selected_obstacle": obstacle_state,
                }
            )

        trigger_scan_complete = bool(
            trigger is not None or len(prefix_rows) == len(replay.actions)
        )
        if trigger is None:
            metrics = {
                "schema_version": METRICS_SCHEMA,
                "trigger_scan_complete": trigger_scan_complete,
                "trigger_found": False,
                "native_prefix_exact": len(prefix_rows) == len(replay.actions),
                "exact_paired_trigger_state": False,
                "full_recorded_episode_complete": False,
                "baseline_exposure_complete": False,
                "psf_exposure_complete": False,
                "baseline_link56_contact_present": False,
                "psf_any_robot_selected_obstacle_contact_present": False,
                "psf_shifted_link56_external_contact_present": False,
                "psf_contact_terminated": False,
                "material_correction_before_baseline_contact": False,
                "psf_paper_car_avoided": False,
                "psf_useful_post_correction_motion": False,
                "psf_task_success_after_correction": False,
                "psf_terminal_task_success": False,
                "psf_method_stop_or_stall": False,
                "all_psf_field_queries_valid": False,
                "all_psf_qps_solved_and_postchecked": False,
            }
            classification = classify_triggered_rescue(
                metrics, protocol, case_id=arguments.case_id
            )
            candidate = {
                "schema_version": RESULT_SCHEMA,
                "status": "complete",
                "protocol_id": selected_protocol_id,
                "run_id": arguments.run_id,
                "case_id": arguments.case_id,
                "provenance": provenance,
                "trigger_scan": {
                    "decision_variable": "bounded_physical_arm_joint_velocity_rad_s",
                    "rows": trigger_rows,
                    "first_actionable_trigger": None,
                    "complete": trigger_scan_complete,
                },
                "native_osc_prefix": {
                    "completed_action_count": len(prefix_rows),
                    "rows": prefix_rows,
                    "trigger_state_sha256": None,
                    "goal_values_at_trigger": None,
                },
                "metrics": metrics,
                "classification": classification,
                "arms": {},
                "partial_output_interpreted": False,
                "timing": {
                    "started_unix": started,
                    "finished_unix": time.time(),
                    "elapsed_seconds": time.time() - started,
                },
            }
            publish_hashed_json(result_path, candidate)
            print(json.dumps({"status": "complete", "outcome": classification["classification"], "result": str(result_path)}, sort_keys=True))
            return 0

        trigger_action = int(trigger["source_action_index"])
        trigger_state = np.asarray(
            source_env.sim.get_state().flatten(), dtype=np.float64
        ).copy()
        trigger_state_sha256 = evaluator.array_sha256(trigger_state)
        if trigger_state_sha256 != trigger["simulator_state_sha256"]:
            raise fast.FastRunnerError("latched trigger state changed before branching")
        suffix_actions = replay.actions[trigger_action:]
        if not suffix_actions:
            raise fast.FastRunnerError("trigger leaves no action suffix")
        if any(
            any(float(value) != 0.0 for value in action[3:6])
            for action in suffix_actions
        ):
            raise fast.FastRunnerError(
                "translation-only suffix would discard nonzero rotation"
            )
        expected_updates = len(suffix_actions) * 5
        expected_substeps = len(suffix_actions) * 25
        boundary_goal_values = tuple(trigger["native_goal_values_at_boundary"])

        adapter = fast._run_arm(
            arm_name="joint_velocity_adapter_only",
            evaluator=evaluator,
            runtime=runtime,
            case=case,
            source_env=source_env,
            state_at_B=trigger_state,
            actions=suffix_actions,
            source_start_action=trigger_action,
            bundle=bundle,
            resolved=resolved,
            full_samples=full_samples,
            runtime_protocol=runtime_protocol,
            nominal_activation_threshold_m2_per_s=derived[
                "nominal_trigger_threshold_m2_per_s"
            ],
            qp_max_iterations=derived["qp_max_iterations"],
            expected_filter_updates=expected_updates,
            expected_physics_substeps=expected_substeps,
            expected_boundary_goal_values=boundary_goal_values,
            monitor_registered_forbidden_contacts=True,
            record_paper_car=True,
            paper_car_reference_position=paper_car_reference_position,
            paper_car_obstacle_name_override=obstacle_name,
        )
        arm_evidence["adapter_only"] = adapter
        psf = fast._run_arm(
            arm_name="joint_velocity_adapter_plus_link56_psf",
            evaluator=evaluator,
            runtime=runtime,
            case=case,
            source_env=source_env,
            state_at_B=trigger_state,
            actions=suffix_actions,
            source_start_action=trigger_action,
            bundle=bundle,
            resolved=resolved,
            full_samples=full_samples,
            runtime_protocol=runtime_protocol,
            nominal_activation_threshold_m2_per_s=derived[
                "nominal_trigger_threshold_m2_per_s"
            ],
            qp_max_iterations=derived["qp_max_iterations"],
            expected_filter_updates=expected_updates,
            expected_physics_substeps=expected_substeps,
            expected_boundary_goal_values=boundary_goal_values,
            monitor_registered_forbidden_contacts=True,
            record_paper_car=True,
            paper_car_reference_position=paper_car_reference_position,
            paper_car_obstacle_name_override=obstacle_name,
            record_qp_failure_as_method_stop=True,
        )
        arm_evidence["adapter_plus_psf"] = psf

        exact_pair = fast._pair_exact(adapter, psf)
        adapter_contact_boundary = adapter["literal_contact"][
            "first_link56_physical_boundary"
        ]
        material_rows = [
            row
            for row in psf["activation_trace"]
            if row["filter_correction_l2_rad_s"]
            >= derived["material_correction_threshold_rad_s"]
        ]
        first_material_boundary = (
            None if not material_rows else int(material_rows[0]["physical_boundary"])
        )
        post_motion = _post_correction_metrics(
            psf, first_boundary=first_material_boundary
        )
        acceptance = derived["acceptance"]
        useful_motion = bool(
            first_material_boundary is not None
            and post_motion["filter_correction_integral_rad"]
            >= acceptance["minimum_correction_integral_rad"]
            and post_motion["measured_joint_motion_integral_rad"]
            >= acceptance[
                "minimum_post_correction_measured_joint_motion_integral_rad"
            ]
            and post_motion["cartesian_path_length_m"]
            >= acceptance["minimum_post_correction_eef_path_length_m"]
            and post_motion["zero_command_fraction"]
            <= acceptance["maximum_post_correction_zero_command_fraction"]
        )
        psf_registered = psf.get("registered_forbidden_contact")
        if not isinstance(psf_registered, Mapping):
            raise fast.FastRunnerError(
                "PSF arm lacks broad registered-contact measurement"
            )
        shifted_contact = bool(
            psf_registered.get("any_link56_nonselected_external_contact")
        )
        first_success = psf["task"]["first_task_success_source_action_index"]
        task_after_correction = bool(
            first_material_boundary is not None
            and first_success is not None
            and (int(first_success) + 1) * 25 > first_material_boundary
        )
        baseline_complete = bool(adapter["exposure_complete"])
        psf_complete = bool(psf["exposure_complete"])
        full_episode_complete = bool(
            len(prefix_rows) == trigger_action
            and baseline_complete
            and psf_complete
            and adapter["task"]["completed_source_action_count"]
            == len(suffix_actions)
            and psf["task"]["completed_source_action_count"]
            == len(suffix_actions)
            and trigger_action + len(suffix_actions) == len(replay.actions)
        )
        psf_qp_consistent = bool(
            psf["qp_solve_count"] == psf["filter_update_count"]
            and psf["qp_postcheck_count"] == psf["filter_update_count"]
            and psf["joint_limit_postcheck_count"] == psf["filter_update_count"]
        )
        psf_field_valid = bool(
            psf["invalid_field_query_count"] == 0
            and psf["precontact_field_queries_valid_and_positive"]
        )
        psf_method_stop_or_stall = bool(
            psf.get("method_terminated_early")
            or (
                not psf.get("contact_terminated_early")
                and first_material_boundary is not None
                and not useful_motion
            )
        )
        metrics = {
            "schema_version": METRICS_SCHEMA,
            "trigger_scan_complete": trigger_scan_complete,
            "trigger_found": True,
            "native_prefix_exact": len(prefix_rows) == trigger_action,
            "exact_paired_trigger_state": exact_pair,
            "full_recorded_episode_complete": full_episode_complete,
            "baseline_exposure_complete": baseline_complete,
            "psf_exposure_complete": psf_complete,
            "baseline_link56_contact_present": bool(
                adapter["literal_contact"]["link56_present"]
            ),
            "psf_any_robot_selected_obstacle_contact_present": bool(
                psf["literal_contact"]["any_robot_selected_obstacle_present"]
            ),
            "psf_shifted_link56_external_contact_present": shifted_contact,
            "psf_contact_terminated": bool(psf["contact_terminated_early"]),
            "material_correction_before_baseline_contact": bool(
                first_material_boundary is not None
                and adapter_contact_boundary is not None
                and first_material_boundary < int(adapter_contact_boundary)
            ),
            "psf_paper_car_avoided": bool(
                psf["paper_car"]["endpoint_ledger_complete"]
                and psf["paper_car"]["paper_collision_avoidance"]
                and psf["paper_car"][
                    "maximum_active_obstacle_l1_displacement_m"
                ]
                <= acceptance["paper_car_displacement_threshold_m"]
            ),
            "psf_useful_post_correction_motion": useful_motion,
            "psf_task_success_after_correction": task_after_correction,
            "psf_terminal_task_success": bool(
                psf["task"]["terminal_task_success"]
            ),
            "psf_method_stop_or_stall": psf_method_stop_or_stall,
            "all_psf_field_queries_valid": psf_field_valid,
            "all_psf_qps_solved_and_postchecked": psf_qp_consistent,
        }
        classification = classify_triggered_rescue(
            metrics, protocol, case_id=arguments.case_id
        )
        suffix_record = {
            "source_action_index_start": trigger_action,
            "source_action_index_end_inclusive": len(replay.actions) - 1,
            "action_count": len(suffix_actions),
            "actions": [list(action) for action in suffix_actions],
        }
        candidate = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "protocol_id": selected_protocol_id,
            "run_id": arguments.run_id,
            "case_id": arguments.case_id,
            "provenance": provenance,
            "trigger_scan": {
                "decision_variable": "bounded_physical_arm_joint_velocity_rad_s",
                "fixed_trigger_action_index_used": False,
                "case_identity_used_by_trigger": False,
                "historical_contact_action_used_by_trigger": False,
                "rows": trigger_rows,
                "first_actionable_trigger": trigger,
                "complete": trigger_scan_complete,
            },
            "native_osc_prefix": {
                "completed_action_count": len(prefix_rows),
                "rows": prefix_rows,
                "trigger_state_sha256": trigger_state_sha256,
                "goal_values_at_trigger": list(boundary_goal_values),
            },
            "paired_suffix": {
                "action_record_sha256": fast._sha256(
                    fast._canonical(suffix_record)
                ),
                "action_array_sha256": fast._sha256(
                    fast._canonical([list(action) for action in suffix_actions])
                ),
                "source_action_index_start": trigger_action,
                "source_action_index_end_inclusive": len(replay.actions) - 1,
                "action_count": len(suffix_actions),
                "expected_filter_updates_per_arm": expected_updates,
                "expected_physics_substeps_per_arm": expected_substeps,
                "online_policy_queries": 0,
            },
            "field": {
                "bundle_hashes": asdict(bundle.hashes),
                "diagnostics": asdict(bundle.diagnostics),
                "resolved_geometry": resolved.to_dict(),
                "full_robot_sampling": full_sample_evidence,
                "settled_obstacle_state": settled_obstacle_state,
            },
            "post_correction_motion": post_motion,
            "metrics": metrics,
            "classification": classification,
            "arms": arm_evidence,
            "claim_scope": protocol["result_contract"]["claim_scope"],
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
                    "outcome": classification["classification"],
                    "result": str(result_path),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as error:
        if isinstance(error, fast.FastArmExecutionFailure):
            arm_evidence[error.evidence.get("arm_name", "failed_arm")] = dict(
                error.evidence
            )
        try:
            from main.poisson_fullbody.contracts import publish_hashed_json

            failure = {
                "schema_version": RESULT_SCHEMA,
                "status": "apparatus_failure",
                "protocol_id": selected_protocol_id,
                "run_id": arguments.run_id,
                "case_id": arguments.case_id,
                "provenance": provenance,
                "trigger_scan": {
                    "rows": trigger_rows,
                    "complete": False,
                },
                "native_osc_prefix": prefix_rows,
                "arms": arm_evidence,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "partial_output_interpreted": False,
                "timing": {
                    "started_unix": started,
                    "finished_unix": time.time(),
                    "elapsed_seconds": time.time() - started,
                },
            }
            publish_hashed_json(result_path, failure)
        except Exception as publication_error:
            print(
                "triggered-rescue failure publication failed: %s"
                % publication_error,
                file=sys.stderr,
            )
        print("triggered rescue failed: %s" % error, file=sys.stderr)
        return 1
    finally:
        if source_env is not None:
            source_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
