#!/usr/bin/env python3
"""Audit a deterministic VLA-ledger-independent finite-candidate backup."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.evaluate_distal_pncbf_backup_oracle_e05 import _backup_summary
from scripts.evaluate_distal_smooth_field_attribution_e05 import (
    InstrumentedContinuationProbe,
    _disable_images,
)
from scripts.evaluate_distal_counterfactual_field_e05 import FixedContinuationProbe
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


RESULT_SCHEMA = "vlsa_distal_pncbf_pure_backup_audit_e05_result.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    geometry_config_path: Path,
    experiment_config_path: Path,
    source_result_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_SETTLE_ACTIONS,
        _active_obstacle,
        _build_environment,
        _runtime_imports,
        _settle,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.pure_backup import (
        load_pure_backup_config,
        registered_directions,
        select_verified_backup,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    config = load_pure_backup_config(experiment_config_path)
    registration = config["registered_source_result"]
    _require(_file_sha256(source_result_path) == registration["file_sha256"], "source file differs")
    source_result = _load(source_result_path)
    _require(source_result["result_payload_sha256"] == registration["result_payload_sha256"], "source payload differs")
    _require(source_result["source"]["commit"] == registration["source_commit"], "source commit differs")
    _require(source_result["interpretation"] == "finite_horizon_exact_backup_policy_value_pass", "source result differs")

    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file differs")
    _require(archived["result_payload_sha256"] == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    rows = [row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID]
    _require(len(rows) == 1, "manifest case differs")
    case = rows[0]
    validate_case_row(case, repo_root)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=True)
    geometry_config = load_shadow_config(geometry_config_path)
    by_step = {int(item["step"]): np.asarray(item["action"], dtype=np.float64) for item in source_result["actions"]}
    _require(set(by_step) == set(range(288)), "source action ledger differs")

    env = probe_env = None
    try:
        env, task, observation, _ = _build_environment(runtime, case, render_resolution=32)
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        probe_env, probe_task, probe_observation, _ = _build_environment(runtime, case, render_resolution=32)
        probe_observation = _settle(probe_env, probe_observation, TABLE_SETTLE_ACTIONS)
        _require(str(task.language) == str(probe_task.language), "probe task differs")
        obstacle_name, _ = _active_obstacle(env, observation)
        initial_obstacle = np.asarray(observation["%s_pos" % obstacle_name], dtype=np.float64).copy()
        perception = source_result["config"]["registered_inputs"]
        # Geometry is reconstructed from the immutable Table-1 perception record.
        archived_perception = archived["perception"]
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(
            geometry_config,
            {
                "p2": archived_perception["mvee_center"],
                "R2": archived_perception["mvee_rotation"],
                "Q2_diag": archived_perception["mvee_semiaxes"],
                "record": {"label": archived_perception["obstacle_label"]},
            },
        )
        one_step = SlabbedEightConstraintProbe(
            probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        _disable_images(probe_env)
        instrumented = InstrumentedContinuationProbe(
            one_step, obstacle_name, initial_obstacle
        )
        expected_substeps = int(config["measurement"]["expected_mujoco_substeps_per_action"])
        tolerance = float(config["measurement"]["boundary_equivalence_tolerance"])
        car_limit = float(config["paper_car_threshold_m"])
        buffer_m = float(config["safety_buffer_m"])
        target_steps = set(int(item) for item in config["decision_steps"])
        audit_states = []

        def candidate_record(name: str, order: int, first_action: Any, step: int) -> dict[str, Any]:
            actions = np.zeros((1 + int(config["candidate_family"]["terminal_hold_actions"]), 7), dtype=np.float64)
            actions[0] = np.asarray(first_action, dtype=np.float64)
            rollout = instrumented.rollout_internal(
                env,
                actions,
                expected_substeps=expected_substeps,
                boundary_tolerance=tolerance,
                step_base=step,
            )
            trace = np.asarray(rollout["clearance_trace_m"], dtype=np.float64)[:, :7]
            record = {
                "minimum_clearance_m": float(np.min(trace)),
                "future_minimum_clearance_m": float(np.min(trace[1:])),
                "row_minimum_clearance_m": np.min(trace, axis=0).tolist(),
                "protected_contact_count": len(rollout["protected_contacts"]),
                "maximum_active_obstacle_l1_displacement_m": float(rollout["maximum_active_obstacle_l1_displacement_m"]),
                "sample_count": int(trace.shape[0]),
                "maximum_boundary_equivalence_error_m": float(rollout["maximum_boundary_equivalence_error_m"]),
            }
            return {"name": name, "order": int(order), "first_action": actions[0].tolist(), "record": record}

        for step in range(289):
            if step in target_steps:
                source_before = np.asarray(_dynamic_state_vector(env), dtype=np.float64).copy()
                current = np.asarray(one_step.clearances(env)[:7], dtype=np.float64)
                active_row = int(np.argmin(current))
                links = geometry._slabbed_links(env)
                normal = np.asarray(links[active_row].center) - np.asarray(geometry.obstacle.center)
                normal /= float(np.linalg.norm(normal))
                definitions: list[tuple[str, Any]] = [("hold", np.zeros(7, dtype=np.float64))]
                for direction_name, direction in registered_directions(normal):
                    for amplitude in config["candidate_family"]["amplitudes_action"]:
                        action = np.zeros(7, dtype=np.float64)
                        action[:3] = np.asarray(direction, dtype=np.float64) * float(amplitude)
                        definitions.append(("%s_amp_%s" % (direction_name, amplitude), action))
                forward = [candidate_record(name, index, action, step) for index, (name, action) in enumerate(definitions)]
                reverse_raw = [candidate_record(name, index, action, step) for index, (name, action) in reversed(list(enumerate(definitions)))]
                reverse = sorted(reverse_raw, key=lambda item: item["order"])
                replay = candidate_record(definitions[0][0], 0, definitions[0][1], step)
                source_after = np.asarray(_dynamic_state_vector(env), dtype=np.float64)
                maximum_repeat_error = max(
                    abs(float(forward[0]["record"][key]) - float(replay["record"][key]))
                    for key in (
                        "minimum_clearance_m",
                        "future_minimum_clearance_m",
                        "maximum_active_obstacle_l1_displacement_m",
                    )
                )
                maximum_order_error = 0.0
                for left, right in zip(forward, reverse):
                    _require(left["name"] == right["name"], "candidate identity differs")
                    for key in (
                        "minimum_clearance_m",
                        "future_minimum_clearance_m",
                        "maximum_active_obstacle_l1_displacement_m",
                    ):
                        maximum_order_error = max(
                            maximum_order_error,
                            abs(float(left["record"][key]) - float(right["record"][key])),
                        )
                selected = select_verified_backup(
                    forward,
                    safety_buffer_m=buffer_m,
                    paper_car_threshold_m=car_limit,
                )
                audit_states.append(
                    {
                        "step": step,
                        "initial_clearance_m": current.tolist(),
                        "active_row": active_row,
                        "state_derived_normal": normal.tolist(),
                        "candidate_count": len(forward),
                        "safe_candidate_count": sum(
                            item["record"]["minimum_clearance_m"] >= buffer_m
                            and item["record"]["protected_contact_count"] == 0
                            and item["record"]["maximum_active_obstacle_l1_displacement_m"] <= car_limit
                            for item in forward
                        ),
                        "selected": selected,
                        "repeat_replay_maximum_error": maximum_repeat_error,
                        "candidate_order_maximum_error": maximum_order_error,
                        "source_state_maximum_mutation": float(np.max(np.abs(source_after - source_before))),
                        "ledger_independence": {
                            "policy_function_arguments": ["complete_physical_state", "fixed_config"],
                            "vla_chunk_argument": False,
                            "proposal_ledger_argument": False,
                            "candidate_family_sha256": hashlib.sha256(_canonical([(name, np.asarray(action).tolist()) for name, action in definitions])).hexdigest(),
                        },
                        "candidates": forward,
                    }
                )
            if step < 288:
                observation, _, _, _ = env.step(by_step[step].tolist())

        gates = {
            "candidate_order_invariant": all(item["candidate_order_maximum_error"] <= tolerance for item in audit_states),
            "repeat_replay_deterministic": all(item["repeat_replay_maximum_error"] <= tolerance for item in audit_states),
            "source_state_unmodified": all(item["source_state_maximum_mutation"] == 0.0 for item in audit_states),
            "ledger_independent_by_construction": all(
                not item["ledger_independence"]["vla_chunk_argument"]
                and not item["ledger_independence"]["proposal_ledger_argument"]
                for item in audit_states
            ),
            "safe_candidate_support_all_states": all(item["safe_candidate_count"] > 0 for item in audit_states),
            "selected_terminal_hold_safe_all_states": all(item["selected"] is not None for item in audit_states),
        }
        passed = all(gates.values())
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "case_id": CASE_ID,
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "source_result": {
                "path": str(source_result_path),
                "file_sha256": _file_sha256(source_result_path),
                "result_payload_sha256": source_result["result_payload_sha256"],
                "used_only_to_reconstruct_physical_states": True,
                "forbidden_from_backup_action_and_score": True,
            },
            "audit_states": audit_states,
            "gates": gates,
            "interpretation": (
                "pure_backup_representative_state_audit_pass"
                if passed
                else "pure_backup_representative_state_audit_no_go"
            ),
            "primary_problem_solved": passed,
        }
        result["result_payload_sha256"] = _sha256(_canonical(result))
        return result
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        source_result_path=args.source_result.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({"interpretation": result["interpretation"], "gates": result["gates"], "result_payload_sha256": result["result_payload_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
