#!/usr/bin/env python3
"""Run the exact E05 full-action-bound one-step recovery diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

from main.multilink_ellipsoid.full_bound_recovery import (
    RESULT_SCHEMA, full_bound_candidates, load_full_bound_config,
    transition_is_safe,
)
from scripts.collect_distal_boundary_generalization_moka10 import _canonical_action
from scripts.evaluate_distal_execution_margin_nn_e05 import _build_pair, _geometry
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _summary(
    transition: Mapping[str, Any], xyz: Any, nominal_xyz: Any,
    source: str, safe: bool,
) -> dict[str, Any]:
    import numpy as np

    gaps = np.asarray(transition["minimum_substep_clearance_m"][:7], dtype=np.float64)
    value = np.asarray(xyz, dtype=np.float64)
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    return {
        "source": source,
        "candidate_xyz": value.tolist(),
        "objective_l2_from_nominal": float(np.linalg.norm(value - nominal)),
        "minimum_substep_clearance_m": gaps.tolist(),
        "minimum_distal_substep_clearance_m": float(np.min(gaps)),
        "active_constraint_index": int(np.argmin(gaps)),
        "D_opt_proxy_safe": bool(np.all(gaps >= 0.0)),
        "D_sim_raw_safe": bool(
            int(transition["raw_protected_contact_count"]) == 0
            and float(transition["maximum_within_step_obstacle_l1_displacement_m"])
            <= 1.0e-4
        ),
        "verified_safe": bool(safe),
        "raw_protected_contact_count": int(transition["raw_protected_contact_count"]),
        "maximum_within_step_obstacle_l1_displacement_m": float(
            transition["maximum_within_step_obstacle_l1_displacement_m"]
        ),
        "next_state_sha256": str(transition["next_state_sha256"]),
        "env_step_wall_seconds": float(transition["env_step_wall_seconds"]),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        _runtime_imports, pairing_record, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.active import MultiConstraintQp
    from main.multilink_ellipsoid.obstacle_primitives import load_obstacle_primitive_config
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config
    from main.multilink_ellipsoid.sitl_candidate import finite_difference_rows

    started = time.perf_counter_ns()
    config = load_full_bound_config(args.config.resolve())
    source_settings = config["source"]
    _require(
        _file_sha256(args.population_manifest.resolve())
        == source_settings["table1_population_manifest_sha256"],
        "full-bound population manifest differs",
    )
    for path, key in (
        (args.prior_dataset.resolve(), "prior_dataset_sha256"),
        (args.prior_result.resolve(), "prior_result_sha256"),
        (args.prior_validation.resolve(), "prior_validation_sha256"),
    ):
        _require(path.is_file() and not path.is_symlink(), "prior grouped artifact missing")
        _require(_file_sha256(path) == source_settings[key], "prior grouped artifact differs")
    prior_dataset = _load(args.prior_dataset.resolve())
    prior_result = _load(args.prior_result.resolve())
    prior_validation = _load(args.prior_validation.resolve())
    e05_prior = next(
        item for item in prior_dataset["episode_results"]
        if item["case_id"] == config["case_id"]
    )
    _require(
        prior_result["decision"]["neural_training_authorized"] is False
        and prior_validation["neural_training_authorized"] is False
        and e05_prior["eligible"] is False
        and e05_prior["first_crossing_step"] == source_settings["action_step"],
        "prior grouped NO-GO trigger differs",
    )
    _require(
        _file_sha256(args.geometry_config.resolve())
        == config["geometry"]["distal_geometry_config_sha256"]
        and _file_sha256(args.exact_box_config.resolve())
        == config["geometry"]["exact_box_config_sha256"],
        "full-bound geometry inputs differ",
    )
    archived_path = args.archived_root.resolve() / source_settings["archived_relative_path"]
    _require(archived_path.is_file() and not archived_path.is_symlink(), "E05 archive missing")
    archived = _load(archived_path)
    _require(
        _file_sha256(archived_path) == source_settings["archived_file_sha256"]
        and archived.get("result_payload_sha256")
        == source_settings["archived_payload_sha256"]
        and len(archived.get("actions", [])) == source_settings["action_count"]
        and archived.get("action_invariance_ledger", {}).get("executed_sequence_sha256")
        == source_settings["executed_sequence_sha256"],
        "E05 Table-1 receipt differs",
    )
    rows = {item["case_id"]: item for item in read_jsonl(args.population_manifest.resolve())}
    case = rows[config["case_id"]]
    validate_case_row(case, args.repo_root.resolve())
    source = _git_identity(args.repo_root.resolve(), args.expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(args.geometry_config.resolve())
    exact_box_config = load_obstacle_primitive_config(args.exact_box_config.resolve())
    env = probe_env = None
    try:
        env, probe_env, task, observation, setup = _build_pair(runtime, case)
        _require(setup["obstacle_name"] == "moka_pot_obstacle_1", "E05 active obstacle differs")
        pairing = pairing_record(
            case=case,
            selected_initial_state=setup["selected_initial_state"],
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=setup["obstacle_name"],
            settled_simulator_state=np.asarray(
                env.sim.get_state().flatten(), dtype=np.float64
            ),
        )
        for key in (
            "manifest_row_sha256", "initial_state_sha256",
            "initial_observation_sha256", "settled_simulator_state_sha256",
            "settled_active_obstacle_position_sha256", "policy_noise_schedule_sha256",
        ):
            _require(pairing[key] == archived["pairing"][key], "E05 pairing differs: %s" % key)
        geometry, exact_boxes = _geometry(
            geometry_config=geometry_config,
            exact_box_config=exact_box_config,
            archived=archived,
            env=env,
            obstacle_name=setup["obstacle_name"],
        )
        probe = SubstepEightConstraintProbe(
            probe_env, geometry,
            active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1.0e-6,
            contact_distance_threshold_m=float(
                config["verification"]["raw_contact_distance_threshold_m"]
            ),
            obstacle_primitive_union=exact_boxes,
        )
        step = int(source_settings["action_step"])
        for index in range(step):
            env.step(_canonical_action(archived["actions"][index], index).tolist())
        nominal = _canonical_action(archived["actions"][step], step)
        nominal_transition = probe.transition(env, nominal)
        nominal_summary = _summary(
            nominal_transition, nominal[:3], nominal[:3], "nominal",
            transition_is_safe(nominal_transition, config),
        )

        epsilon = float(config["finite_difference"]["epsilon_action"])
        limit = float(config["action_space"]["action_limit"])
        plus_xyz = []
        minus_xyz = []
        plus_h = []
        minus_h = []
        finite_difference_records = []
        for dimension in range(3):
            positive = nominal.copy(); negative = nominal.copy()
            positive[dimension] = min(limit, positive[dimension] + epsilon)
            negative[dimension] = max(-limit, negative[dimension] - epsilon)
            plus = probe.transition(env, positive)
            minus = probe.transition(env, negative)
            plus_xyz.append(positive[:3].copy()); minus_xyz.append(negative[:3].copy())
            plus_h.append(plus["minimum_substep_clearance_m"][:7])
            minus_h.append(minus["minimum_substep_clearance_m"][:7])
            finite_difference_records.append({
                "dimension": dimension,
                "plus": _summary(plus, positive[:3], nominal[:3], "finite_difference_plus", transition_is_safe(plus, config)),
                "minus": _summary(minus, negative[:3], nominal[:3], "finite_difference_minus", transition_is_safe(minus, config)),
            })
        gradients = finite_difference_rows(
            plus_h, minus_h, plus_xyz, minus_xyz
        )
        nominal_h = np.asarray(
            nominal_transition["minimum_substep_clearance_m"][:7],
            dtype=np.float64,
        )
        lower = -nominal_h + gradients @ nominal[:3]
        optimizer = config["optimizer"]
        qp = MultiConstraintQp(
            eps_abs=float(optimizer["eps_abs"]),
            eps_rel=float(optimizer["eps_rel"]),
            max_iter=int(optimizer["max_iter"]),
            residual_tolerance=float(optimizer["residual_tolerance"]),
            bound_tolerance=float(optimizer["bound_tolerance_action"]),
        )
        qp_started = time.perf_counter_ns()
        qp_result = qp.solve(
            nominal[:3], np.eye(3), gradients, lower,
            -limit * np.ones(3), limit * np.ones(3),
        )
        qp_wall_seconds = (time.perf_counter_ns() - qp_started) * 1.0e-9
        qp_transition = None
        qp_summary = None
        if qp_result.valid and qp_result.qdot_safe is not None:
            qp_xyz = np.asarray(qp_result.qdot_safe, dtype=np.float64)
            qp_action = nominal.copy(); qp_action[:3] = qp_xyz
            qp_transition = probe.transition(env, qp_action)
            qp_summary = _summary(
                qp_transition, qp_xyz, nominal[:3], "finite_difference_qp",
                transition_is_safe(qp_transition, config),
            )

        candidate_records = []
        smallest = None
        for candidate_index, candidate in enumerate(
            full_bound_candidates(nominal[:3], config)
        ):
            action = nominal.copy(); action[:3] = candidate["xyz"]
            transition = probe.transition(env, action)
            safe = transition_is_safe(transition, config)
            record = _summary(
                transition, candidate["xyz"], nominal[:3],
                candidate["source"], safe,
            )
            record["candidate_index"] = candidate_index
            candidate_records.append(record)
            if safe and (
                smallest is None
                or (record["objective_l2_from_nominal"], candidate_index)
                < (smallest["objective_l2_from_nominal"], smallest["candidate_index"])
            ):
                smallest = record

        verified_options = []
        if smallest is not None:
            verified_options.append(smallest)
        if qp_summary is not None and qp_summary["verified_safe"]:
            option = dict(qp_summary); option["candidate_index"] = -1
            verified_options.append(option)
        executed = None
        execution_match = False
        if verified_options:
            accepted = min(
                verified_options,
                key=lambda item: (
                    item["objective_l2_from_nominal"],
                    item["candidate_index"],
                ),
            )
            accepted_action = nominal.copy()
            accepted_action[:3] = np.asarray(accepted["candidate_xyz"], dtype=np.float64)
            accepted_clone = probe.transition(env, accepted_action)
            env.step(accepted_action.tolist())
            executed_hash = hashlib.sha256(
                np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
            ).hexdigest()
            execution_match = bool(executed_hash == accepted_clone["next_state_sha256"])
            executed = {
                "source": accepted["source"],
                "action": accepted_action.tolist(),
                "clone_next_state_sha256": accepted_clone["next_state_sha256"],
                "executed_next_state_sha256": executed_hash,
                "exact_clone_execution_match": execution_match,
            }

        physical = bool(smallest is not None)
        qp_pass = bool(qp_summary is not None and qp_summary["verified_safe"])
        decision = {
            "physical_full_bound_recovery_exists": physical,
            "finite_difference_qp_recovery_pass": qp_pass,
            "execution_fidelity_pass": execution_match,
            "larger_region_collection_authorized": bool(physical and execution_match),
            "closed_loop_e05_authorized": False,
            "multi_step_required_before_larger_region_collection": bool(not physical),
        }
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "table1": {
                "path": str(archived_path),
                "file_sha256": _file_sha256(archived_path),
                "payload_sha256": archived["result_payload_sha256"],
                "modified": False,
            },
            "prior_grouped_gate": {
                "job_id": source_settings["prior_grouped_job_id"],
                "dataset_file_sha256": _file_sha256(args.prior_dataset.resolve()),
                "result_file_sha256": _file_sha256(args.prior_result.resolve()),
                "validation_file_sha256": _file_sha256(args.prior_validation.resolve()),
                "neural_training_authorized": False,
            },
            "pairing": pairing,
            "action_step": step,
            "nominal": nominal_summary,
            "finite_difference": {
                "records": finite_difference_records,
                "gradients_m_per_action": gradients.tolist(),
            },
            "qp": {
                "valid": bool(qp_result.valid),
                "reason": qp_result.reason,
                "diagnostics": dict(qp_result.diagnostics),
                "solve_wall_seconds": qp_wall_seconds,
                "lower": lower.tolist(),
                "candidate": qp_summary,
            },
            "global_search": {
                "candidate_count": len(candidate_records),
                "verified_safe_candidate_count": sum(
                    item["verified_safe"] for item in candidate_records
                ),
                "smallest_verified_safe_candidate": smallest,
                "candidates": candidate_records,
            },
            "executed_verified_recovery": executed,
            "decision": decision,
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = _hash_without(
            result, "result_payload_sha256"
        )
        return result
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--population-manifest", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--prior-dataset", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path, required=True)
    parser.add_argument("--prior-validation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args)
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result["decision"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
