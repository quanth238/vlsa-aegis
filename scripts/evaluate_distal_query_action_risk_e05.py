#!/usr/bin/env python3
"""Evaluate the query-aligned five-action Monte Carlo risk gate on E05."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_pncbf_backup_oracle_e05 import (
    _combine_primary_internal_records,
    _primary_internal_rollout,
)
from scripts.evaluate_distal_smooth_field_attribution_e05 import (
    InstrumentedContinuationProbe,
    _disable_images,
)
from scripts.evaluate_distal_counterfactual_field_e05 import FixedContinuationProbe
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require, _sha256,
)


def _public(value: Any) -> Any:
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, Mapping):
        return {str(key): _public(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    return value


def _allocation_record() -> dict[str, Any]:
    """Record either one allocated H100 or a CPU canary on an H100 host.

    MuJoCo/OSC execution is CPU-bound.  A canary may therefore use CPU cores
    on an H100 worker without consuming another user's allocated accelerator.
    Full diagnostic jobs continue to request exactly one H100.
    """

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isdigit():
        raise ValueError("query action-risk simulation requires Slurm")
    allocated = bool(os.environ.get("SLURM_JOB_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES"))
    if allocated:
        output = subprocess.check_output(
            [
                "nvidia-smi", "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=30,
        ).strip().splitlines()
        if len(output) != 1 or "H100" not in output[0]:
            raise ValueError("query action-risk GPU allocation must expose exactly one H100")
    else:
        if os.environ.get("QUERY_RISK_CPU_CANARY") != "1":
            raise ValueError("CPU-on-H100 execution is allowed only for an explicit canary")
        information = sorted(Path("/proc/driver/nvidia/gpus").glob("*/information"))
        output = [path.read_text().strip().replace("\n", "; ") for path in information]
        if len(output) != 8 or any("H100" not in item for item in output):
            raise ValueError("CPU canary is not on the registered eight-H100 worker")
    return {
        "slurm_job_id": job_id,
        "host": socket.gethostname(),
        "execution_mode": "allocated_H100" if allocated else "CPU_canary_on_H100_host",
        "gpu_device_allocated": allocated,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host_H100_inventory": output,
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
    }


def evaluate(
    *, repo_root: Path, population_manifest_path: Path, archived_path: Path,
    geometry_config_path: Path, experiment_config_path: Path,
    expected_commit: str, output_path: Path, candidate_limit: Optional[int] = None,
    case_id_override: Optional[str] = None,
    state_step_override: Optional[int] = None,
    query_index_override: Optional[int] = None,
    result_schema_override: Optional[str] = None,
    claim_scope_override: Optional[str] = None,
    population_binding: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    import time
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_SETTLE_ACTIONS, _active_obstacle, _build_environment,
        _runtime_imports, _settle, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.pure_backup import (
        orthonormal_local_frame, registered_directions, select_backup_with_fallback,
    )
    from main.multilink_ellipsoid.query_action_risk import (
        RESULT_SCHEMA, candidate_definitions, canonical, combine_row_minima,
        exact_safe, load_config, risk_from_row_minimum,
    )
    from main.multilink_ellipsoid.rollout import (
        _auxiliary_sim_snapshot, _base_env, _controller_snapshot,
        _dynamic_state_vector, _restore_auxiliary_sim_snapshot,
        _restore_controller_snapshot,
    )
    from main.multilink_ellipsoid.shadow import MultilinkEllipsoidShadow, load_shadow_config
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    started = time.perf_counter_ns()
    config = load_config(experiment_config_path)
    source = _git_identity(repo_root, expected_commit)
    allocation = _allocation_record()
    archived = _load(archived_path)
    target_case_id = config["case_id"] if case_id_override is None else str(case_id_override)
    _require(archived["case_id"] == target_case_id, "archived query-risk case differs")
    _require(archived["task_success"] is True, "archived E05 task did not succeed")
    rows = [row for row in read_jsonl(population_manifest_path)
            if row.get("case_id") == target_case_id]
    _require(len(rows) == 1, "E05 population row differs")
    case = rows[0]
    validate_case_row(case, repo_root)
    runtime = _runtime_imports(include_aegis=True)
    geometry_config = load_shadow_config(geometry_config_path)
    state_step = int(
        config["state"]["query_boundary_step"]
        if state_step_override is None else state_step_override
    )
    horizon = int(config["state"]["candidate_actions"])
    action_rows = {int(item["step"]): item for item in archived["actions"]}
    _require(set(range(state_step + horizon)).issubset(action_rows), "E05 action ledger differs")
    archived_actions = np.asarray(
        [action_rows[step]["executed"] for step in range(len(action_rows))],
        dtype=np.float64,
    )
    nominal = archived_actions[state_step:state_step + horizon]
    _require(nominal.shape == (5, 7), "E05 nominal five-action chunk differs")
    query_index = int(
        config["state"]["query_index"]
        if query_index_override is None else query_index_override
    )
    query = archived["policy_queries"][query_index]
    _require(int(query["query_index"]) == query_index,
             "E05 policy query binding differs")
    _require(int(query["rng_seed"]) == int(case["policy_noise_seed"]) + int(query["query_index"]),
             "E05 policy query seed differs")

    env = probe_env = None
    try:
        env, task, observation, _ = _build_environment(runtime, case, render_resolution=32)
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        probe_env, probe_task, probe_observation, _ = _build_environment(
            runtime, case, render_resolution=32
        )
        probe_observation = _settle(probe_env, probe_observation, TABLE_SETTLE_ACTIONS)
        _require(str(task.language) == str(probe_task.language), "E05 probe task differs")
        obstacle_name, _ = _active_obstacle(env, observation)
        obstacle_reference = np.asarray(
            observation["%s_pos" % obstacle_name], dtype=np.float64
        ).copy()
        perception = archived["perception"]
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(
            geometry_config,
            {
                "p2": perception["mvee_center"],
                "R2": perception["mvee_rotation"],
                "Q2_diag": perception["mvee_semiaxes"],
                "record": {"label": perception["obstacle_label"]},
            },
        )
        one_step = SlabbedEightConstraintProbe(
            probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        _disable_images(probe_env)
        instrumented = InstrumentedContinuationProbe(
            one_step, obstacle_name, obstacle_reference
        )
        for step in range(state_step):
            observation, _, done, _ = env.step(archived_actions[step].tolist())
            _require(not done, "E05 completed before query boundary")

        current = np.asarray(one_step.clearances(env)[:7], dtype=np.float64)
        current_car = float(np.sum(np.abs(
            np.asarray(observation["%s_pos" % obstacle_name], dtype=np.float64)
            - obstacle_reference
        )))
        _require(float(np.min(current)) >= float(config["state"]["minimum_initial_proxy_clearance_m"]),
                 "E05 query boundary is initially proxy-unsafe")
        _require(current_car <= float(config["state"]["maximum_initial_active_obstacle_l1_displacement_m"]),
                 "E05 query boundary already fails CAR")
        links = geometry._slabbed_links(env)
        active_row = int(np.argmin(current))
        normal = np.asarray(links[active_row].center) - np.asarray(geometry.obstacle.center)
        frame = orthonormal_local_frame(normal)
        definitions = candidate_definitions(nominal, frame, config)
        if candidate_limit is not None:
            _require(1 <= int(candidate_limit) <= len(definitions), "candidate canary limit differs")
            definitions = definitions[:int(candidate_limit)]
        full_protocol = candidate_limit is None

        base = _base_env(env)
        source_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()
        source_auxiliary = _auxiliary_sim_snapshot(env)
        source_controller = _controller_snapshot(env)
        source_clock = (int(base.timestep), float(base.cur_time), bool(base.done))
        source_dynamic = np.asarray(_dynamic_state_vector(env), dtype=np.float64).copy()
        source_hash = hashlib.sha256(source_dynamic.tobytes()).hexdigest()

        def restore_source() -> None:
            env.sim.set_state_from_flattened(source_state)
            env.sim.forward()
            _restore_auxiliary_sim_snapshot(env, source_auxiliary)
            _restore_controller_snapshot(env, source_controller)
            base.timestep, base.cur_time, base.done = source_clock
            _require(np.array_equal(_dynamic_state_vector(env), source_dynamic),
                     "E05 query snapshot restore differs")

        expected_substeps = int(config["state"]["expected_mujoco_substeps_per_action"])
        tolerance = float(config["state"]["boundary_equivalence_tolerance_m"])
        buffer_m = float(config["risk_target"]["safety_buffer_m"])
        car_limit = float(config["risk_target"]["paper_car_threshold_m"])
        backup_config = config["backup_policy"]
        records = []

        def summarize_rollout(record: Mapping[str, Any], *, exclude_k0: bool) -> dict[str, Any]:
            trace = np.asarray(record["clearance_trace_m"], dtype=np.float64)[:, :7]
            evaluated = trace[1:] if exclude_k0 else trace
            displacement = np.asarray(
                record.get("active_obstacle_l1_displacement_trace_m", []), dtype=np.float64
            )
            return {
                "row_minimum_clearance_m": np.min(evaluated, axis=0).tolist(),
                "minimum_clearance_m": float(np.min(evaluated)),
                "protected_contact_count": len(record["protected_contacts"]),
                "protected_contacts": record["protected_contacts"],
                "maximum_active_obstacle_l1_displacement_m": float(
                    record["maximum_active_obstacle_l1_displacement_m"]
                ),
                "sample_count": int(evaluated.shape[0]),
                "maximum_boundary_equivalence_error_m": float(
                    record["maximum_boundary_equivalence_error_m"]
                ),
                "clearance_trace_m": trace.tolist(),
                "substep_counts": list(record["substep_counts"]),
                "active_obstacle_l1_displacement_trace_m": displacement.tolist(),
            }

        determinism_runs = []
        for replay_index in range(2):
            restore_source()
            replay = _primary_internal_rollout(
                env, one_step, obstacle_name, obstacle_reference, nominal,
                expected_substeps=expected_substeps, boundary_tolerance=tolerance,
                step_base=state_step,
            )
            determinism_runs.append({
                "replay_index": replay_index,
                "next_state": np.asarray(_dynamic_state_vector(env), dtype=np.float64).copy(),
                "next_state_sha256": hashlib.sha256(
                    np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
                ).hexdigest(),
                "clearance_trace_m": np.asarray(replay["clearance_trace_m"], dtype=np.float64),
                "protected_contacts": replay["protected_contacts"],
                "maximum_active_obstacle_l1_displacement_m": float(
                    replay["maximum_active_obstacle_l1_displacement_m"]
                ),
            })
        determinism = {
            "next_state_maximum_absolute_error": float(np.max(np.abs(
                determinism_runs[0]["next_state"] - determinism_runs[1]["next_state"]
            ))),
            "clearance_trace_maximum_absolute_error_m": float(np.max(np.abs(
                determinism_runs[0]["clearance_trace_m"]
                - determinism_runs[1]["clearance_trace_m"]
            ))),
            "contacts_identical": bool(
                determinism_runs[0]["protected_contacts"]
                == determinism_runs[1]["protected_contacts"]
            ),
            "CAR_maximum_absolute_error_m": abs(
                determinism_runs[0]["maximum_active_obstacle_l1_displacement_m"]
                - determinism_runs[1]["maximum_active_obstacle_l1_displacement_m"]
            ),
            "next_state_sha256": [
                item["next_state_sha256"] for item in determinism_runs
            ],
        }
        restore_source()

        for definition in definitions:
            restore_source()
            restore_error = float(np.max(np.abs(_dynamic_state_vector(env) - source_dynamic)))
            candidate_actions = np.asarray(definition["actions"], dtype=np.float64)
            prefix = _primary_internal_rollout(
                env, one_step, obstacle_name, obstacle_reference, candidate_actions,
                expected_substeps=expected_substeps, boundary_tolerance=tolerance,
                step_base=state_step,
            )
            prefix_summary = summarize_rollout(prefix, exclude_k0=True)
            prefix_physical_veto = bool(
                prefix_summary["protected_contact_count"] > 0
                or prefix_summary["maximum_active_obstacle_l1_displacement_m"] > car_limit
            )
            backup_actual_records = []
            backup_decisions = []
            terminal_hold = None
            terminal_status = "UNSAFE_CONTACT_OR_CAR" if prefix_physical_veto else None
            terminal_reason = "candidate_prefix_physical_veto" if prefix_physical_veto else None

            for backup_step in range(int(backup_config["maximum_executed_actions"]) + 1):
                if terminal_status is not None:
                    break
                current_backup = np.asarray(one_step.clearances(env)[:7], dtype=np.float64)
                if float(np.min(current_backup)) >= float(backup_config["terminal_release_clearance_m"]):
                    hold_actions = np.zeros(
                        (int(backup_config["terminal_hold_actions"]), 7), dtype=np.float64
                    )
                    hold = instrumented.rollout_internal(
                        env, hold_actions, expected_substeps=expected_substeps,
                        boundary_tolerance=tolerance, step_base=state_step + horizon + backup_step,
                    )
                    hold_summary = summarize_rollout(hold, exclude_k0=False)
                    hold_safe = bool(
                        hold_summary["minimum_clearance_m"] >= buffer_m
                        and hold_summary["protected_contact_count"] == 0
                        and hold_summary["maximum_active_obstacle_l1_displacement_m"] <= car_limit
                    )
                    if hold_safe:
                        terminal_hold = hold_summary
                        terminal_status = "SAFE_TERMINAL"
                        terminal_reason = "release_margin_plus_verified_stable_hold"
                        break
                if backup_step >= int(backup_config["maximum_executed_actions"]):
                    terminal_status = "UNKNOWN_TIMEOUT"
                    terminal_reason = "registered_backup_horizon_exhausted"
                    break

                backup_links = geometry._slabbed_links(env)
                backup_active = int(np.argmin(current_backup))
                backup_normal = (
                    np.asarray(backup_links[backup_active].center)
                    - np.asarray(geometry.obstacle.center)
                )
                backup_normal /= np.linalg.norm(backup_normal)
                commands = [("hold", np.zeros(7, dtype=np.float64))]
                for direction_name, direction in registered_directions(backup_normal):
                    for amplitude in backup_config["amplitudes_action"]:
                        command = np.zeros(7, dtype=np.float64)
                        command[:3] = np.asarray(direction) * float(amplitude)
                        commands.append(("%s_amp_%s" % (direction_name, amplitude), command))
                _require(len(commands) == int(backup_config["candidate_count"]),
                         "registered backup candidate count differs")
                branches = []
                for order, (name, command) in enumerate(commands):
                    branch_actions = np.zeros(
                        (1 + int(backup_config["selection_lookahead_hold_actions"]), 7),
                        dtype=np.float64,
                    )
                    branch_actions[0] = command
                    branch = instrumented.rollout_internal(
                        env, branch_actions, expected_substeps=expected_substeps,
                        boundary_tolerance=tolerance,
                        step_base=state_step + horizon + backup_step,
                    )
                    branch_summary = summarize_rollout(branch, exclude_k0=False)
                    trace = np.asarray(branch["clearance_trace_m"], dtype=np.float64)[:, :7]
                    branches.append({
                        "name": name,
                        "order": order,
                        "first_action": command.tolist(),
                        "record": {
                            **branch_summary,
                            "future_minimum_clearance_m": float(np.min(trace[1:])),
                        },
                    })
                selected, selection_mode = select_backup_with_fallback(
                    branches, safety_buffer_m=buffer_m, paper_car_threshold_m=car_limit
                )
                selected_action = np.asarray(selected["first_action"], dtype=np.float64)
                actual = _primary_internal_rollout(
                    env, one_step, obstacle_name, obstacle_reference,
                    selected_action.reshape(1, 7), expected_substeps=expected_substeps,
                    boundary_tolerance=tolerance,
                    step_base=state_step + horizon + backup_step,
                )
                actual_summary = summarize_rollout(actual, exclude_k0=True)
                backup_actual_records.append(actual)
                backup_decisions.append({
                    "backup_step": backup_step,
                    "active_row": backup_active,
                    "normal": backup_normal.tolist(),
                    "candidate_count": len(branches),
                    "selection_mode": selection_mode,
                    "selected_name": selected["name"],
                    "selected_order": selected["order"],
                    "selected_action": selected["first_action"],
                    "selected_branch": selected["record"],
                    "actual": actual_summary,
                })
                if (
                    actual_summary["protected_contact_count"] > 0
                    or actual_summary["maximum_active_obstacle_l1_displacement_m"] > car_limit
                ):
                    terminal_status = "UNSAFE_CONTACT_OR_CAR"
                    terminal_reason = "executed_backup_physical_veto"
                    break

            backup_parts = []
            if backup_actual_records:
                combined_actual = _combine_primary_internal_records(backup_actual_records)
                backup_parts.append(
                    summarize_rollout(combined_actual, exclude_k0=False)["row_minimum_clearance_m"]
                )
            if terminal_hold is not None:
                backup_parts.append(terminal_hold["row_minimum_clearance_m"])
            if backup_parts:
                backup_row_min = combine_row_minima(*backup_parts)
                combined_row_min = combine_row_minima(
                    prefix_summary["row_minimum_clearance_m"], backup_row_min
                )
            else:
                backup_row_min = None
                combined_row_min = prefix_summary["row_minimum_clearance_m"]
            backup_contacts = sum(
                len(item["protected_contacts"]) for item in backup_actual_records
            ) + (0 if terminal_hold is None else terminal_hold["protected_contact_count"])
            backup_car = max(
                [0.0]
                + [float(item["maximum_active_obstacle_l1_displacement_m"])
                   for item in backup_actual_records]
                + ([] if terminal_hold is None else [
                    float(terminal_hold["maximum_active_obstacle_l1_displacement_m"])
                ])
            )
            physical_veto = bool(prefix_physical_veto or backup_contacts > 0 or backup_car > car_limit)
            record = {
                **definition,
                "source_snapshot_sha256": source_hash,
                "source_restore_maximum_error": restore_error,
                "prefix": prefix_summary,
                "backup": {
                    "executed_action_count": len(backup_actual_records),
                    "decisions": backup_decisions,
                    "terminal_hold": terminal_hold,
                    "row_minimum_clearance_m": backup_row_min,
                    "protected_contact_count": backup_contacts,
                    "maximum_active_obstacle_l1_displacement_m": backup_car,
                },
                "candidate_prefix_risk": risk_from_row_minimum(
                    prefix_summary["row_minimum_clearance_m"], buffer_m
                ),
                "backup_risk": None if backup_row_min is None else risk_from_row_minimum(
                    backup_row_min, buffer_m
                ),
                "combined_row_minimum_clearance_m": combined_row_min,
                "combined_risk": risk_from_row_minimum(combined_row_min, buffer_m),
                "terminal_status": terminal_status,
                "terminal_reason": terminal_reason,
                "physical_veto": physical_veto,
                "terminal_state_sha256": hashlib.sha256(
                    np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
                ).hexdigest(),
            }
            record["exact_safe"] = exact_safe(record)
            records.append(record)

        safe_count = sum(bool(item["exact_safe"]) for item in records)
        unsafe_count = sum(
            item["terminal_status"] == "UNSAFE_CONTACT_OR_CAR"
            or max(item["combined_risk"]) > 0.0
            for item in records
        )
        timeout_count = sum(item["terminal_status"] == "UNKNOWN_TIMEOUT" for item in records)
        proxy_safe_physical_collision_count = sum(
            max(item["combined_risk"]) <= 0.0 and item["physical_veto"] for item in records
        )
        gates = {
            "exact_snapshot_replay": bool(
                all(item["source_restore_maximum_error"] == 0.0 for item in records)
                and determinism["next_state_maximum_absolute_error"] == 0.0
                and determinism["clearance_trace_maximum_absolute_error_m"] == 0.0
                and determinism["contacts_identical"]
                and determinism["CAR_maximum_absolute_error_m"] == 0.0
            ),
            "query_boundary_is_initially_safe": float(np.min(current)) >= buffer_m,
            "nominal_future_is_unsafe": not records[0]["exact_safe"],
            "safe_and_unsafe_candidate_support": safe_count > 0 and unsafe_count > 0,
            "at_least_one_safe_terminal": safe_count > 0,
            "no_timeout_labeled_safe": all(
                not item["exact_safe"] for item in records
                if item["terminal_status"] == "UNKNOWN_TIMEOUT"
            ),
            "proxy_safe_physical_collision_count_zero": proxy_safe_physical_collision_count == 0,
            "all_terminal_statuses_registered": all(
                item["terminal_status"] in config["risk_target"]["terminal_statuses"]
                for item in records
            ),
            "all_replays_boundary_exact": all(
                item["prefix"]["maximum_boundary_equivalence_error_m"] <= tolerance
                for item in records
            ),
        }
        result = {
            "schema_version": RESULT_SCHEMA if result_schema_override is None else result_schema_override,
            "status": "complete",
            "scientific_result": bool(full_protocol),
            "execution_mode": "full_diagnostic" if full_protocol else "apparatus_canary",
            "claim_scope": config["claim_scope"] if claim_scope_override is None else claim_scope_override,
            "source": source,
            "allocation": allocation,
            "config": config,
            "archived_table1": {
                "path": str(archived_path),
                "file_sha256": _file_sha256(archived_path),
                "result_payload_sha256": archived["result_payload_sha256"],
                "read_only": True,
            },
            "policy_query": query,
            "determinism_replay": determinism,
            "nominal_five_action_chunk": nominal.tolist(),
            "nominal_five_action_chunk_sha256": hashlib.sha256(nominal.tobytes()).hexdigest(),
            "state": {
                "step": state_step,
                "source_snapshot_sha256": source_hash,
                "initial_clearance_m": current.tolist(),
                "initial_minimum_clearance_m": float(np.min(current)),
                "initial_active_obstacle_l1_displacement_m": current_car,
                "active_row": active_row,
                "local_frame": frame,
            },
            "candidate_count": len(records),
            "candidates": _public(records),
            "summary": {
                "safe_candidate_count": safe_count,
                "unsafe_candidate_count": unsafe_count,
                "unknown_timeout_count": timeout_count,
                "proxy_safe_physical_collision_count": proxy_safe_physical_collision_count,
                "terminal_status_counts": {
                    status: sum(item["terminal_status"] == status for item in records)
                    for status in config["risk_target"]["terminal_statuses"]
                },
                "row_active_witness_counts": [
                    sum(int(np.argmin(item["combined_row_minimum_clearance_m"])) == row
                        for item in records)
                    for row in range(7)
                ],
            },
            "gates": gates,
            "training_authorized": False,
            "interpretation": (
                "query_aligned_five_action_risk_diagnostic_pass"
                if full_protocol and all(gates.values())
                else "query_aligned_five_action_risk_apparatus_canary_pass"
                if not full_protocol and all(
                    gates[key] for key in (
                        "exact_snapshot_replay", "query_boundary_is_initially_safe",
                        "no_timeout_labeled_safe", "all_terminal_statuses_registered",
                        "all_replays_boundary_exact",
                    )
                )
                else "query_aligned_five_action_risk_diagnostic_no_go"
                if full_protocol else "query_aligned_five_action_risk_apparatus_canary_fail"
            ),
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        if result_schema_override is not None:
            result["base_method_config"] = result.pop("config")
            result["population_binding"] = dict(population_binding or {})
            result["interpretation"] = (
                "grouped_query_action_risk_state_pass"
                if full_protocol and all(gates.values())
                else "grouped_query_action_risk_state_no_go"
            )
        result["result_payload_sha256"] = _sha256(canonical(result))
        return result
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        population_manifest_path=args.population_manifest.resolve(),
        archived_path=args.archived.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
        candidate_limit=args.candidate_limit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "summary": result["summary"],
        "gates": result["gates"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
