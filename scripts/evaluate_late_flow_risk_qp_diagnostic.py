#!/usr/bin/env python3
"""Run one saved-query late-flow pullback QP without executing its output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.late_flow_risk_qp import (
    RESULT_SCHEMA, canonical, central_difference_batches,
    correction_residual_bank, load_config, payload_sha256,
    pullback_jacobian, solve_pullback_qp,
)
from scripts.evaluate_learned_risk_sqp_feasibility import _predictor
from scripts.evaluate_terminalized_late_flow_task_success import (
    LiveTerminalizedSelector, RawRobotContactMonitor, _allocation_record,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256, ARCHIVED_PAYLOAD_SHA256, CASE_ID,
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _branch_envelope(residuals: Any, *, label: str) -> dict[str, Any]:
    import numpy as np

    value = np.asarray(residuals, dtype=np.float64)
    _require(value.shape == (13, 10, 3), "late-flow QP residual bank differs")
    _require(np.array_equal(value[0], np.zeros((10, 3))),
             "late-flow QP nominal residual differs")
    names = ["nominal"] + [f"{label}_branch_{index:02d}" for index in range(1, 13)]
    return {
        "schema_version": "crfs_terminal_branching.v1",
        "action_horizon": 10,
        "action_dimensions": [0, 1, 2],
        "candidate_names": names,
        "candidate_output_residuals": value.tolist(),
        "branch_after_euler_step": 8,
    }


def _score_bank(predict: Any, terminal: Any) -> list[list[float]]:
    import numpy as np

    bank = np.asarray(terminal, dtype=np.float64)
    _require(bank.shape == (13, 10, 7), "late-flow QP terminal bank differs")
    output = []
    for actions in bank:
        effective = actions[:5].copy()
        effective[:, :3] = np.clip(effective[:, :3], -1.0, 1.0)
        output.append(predict(effective))
    return output


def _scientific_view(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items()
        if key not in {"allocation", "result_payload_sha256", "source", "wall_seconds"}
    }


def evaluate(
    *, repo_root: Path, manifest_path: Path, archived_path: Path,
    config_path: Path, expected_commit: str, host: str, port: int,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION, TABLE_SETTLE_ACTIONS,
        _active_obstacle, _build_environment, _contact_model_authority,
        _policy_observation, _runtime_imports, _server_identity, _settle,
        array_sha256, pairing_record, query_seed, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector

    started = time.perf_counter_ns()
    config = load_config(config_path)
    source = _git_identity(repo_root, expected_commit)
    allocation = _allocation_record(require_h100=True)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256,
             "late-flow QP archived file differs")
    archived = _load(archived_path)
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256,
             "late-flow QP archived payload differs")
    source_path = Path(config["source_query"]["result_path"])
    _require(_file_sha256(source_path) == config["source_query"]["result_file_sha256"],
             "late-flow QP source result file differs")
    source_query = _load(source_path)
    _require(
        source_query.get("result_payload_sha256")
        == config["source_query"]["result_payload_sha256"]
        and source_query["episode"]["terminal_reason"]
        == config["source_query"]["terminal_reason"]
        and int(source_query["episode"]["action_count"])
        == int(config["source_query"]["replay_action_count"]),
        "late-flow QP source result differs",
    )
    matches = [
        row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID
    ]
    _require(len(matches) == 1, "late-flow QP manifest differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    runtime = _runtime_imports(include_aegis=False)
    env = None
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION,
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        obstacle_name, _ = _active_obstacle(env, observation)
        initial_obstacle = np.asarray(
            observation[obstacle_name + "_pos"], dtype=np.float64,
        ).copy()
        pairing = pairing_record(
            case=case, selected_initial_state=selected_initial_state,
            settled_observation=observation, task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=np.asarray(
                env.sim.get_state().flatten(), dtype=np.float64,
            ),
        )
        _require(pairing == source_query["pairing"],
                 "late-flow QP pairing differs")
        monitor = RawRobotContactMonitor(
            env, initial_obstacle, _contact_model_authority(env, obstacle_name),
            source_query["config"]["physical_contact_groups"],
        )
        for row in source_query["episode"]["actions"]:
            observation, _, done, _, internal = monitor.execute(
                row["executed_action"], step=int(row["step"]),
            )
            _require(not done and int(internal["substep_count"]) == 25,
                     "late-flow QP saved prefix differs")
        physical = monitor.summary()
        _require(
            int(physical["robot_contact_sample_count"]) == 0
            and physical["first_paper_CAR"] is None,
            "late-flow QP query reconstruction is physically invalid",
        )
        state_hash = hashlib.sha256(
            np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
        ).hexdigest()
        _require(
            state_hash == source_query["episode"]["terminal_dynamic_state_sha256"],
            "late-flow QP query state differs",
        )
        client = runtime["websocket_client_policy"].WebsocketClientPolicy(
            host, int(port),
        )
        server_identity = _server_identity(client)
        selector = LiveTerminalizedSelector(
            repo_root=repo_root, runtime=runtime, env=env,
            obstacle_name=obstacle_name, archived=archived,
            config=config, client=client,
        )
        seed = query_seed(
            int(case["policy_noise_seed"]),
            int(config["source_query"]["query_index"]),
        )
        request_base = _policy_observation(
            runtime, observation, task_description=str(task.language),
            resize_size=224, rng_seed=seed,
        )
        epsilon = float(config["numerical_pullback"]["central_difference_action"])
        scored_batches = []
        ordinary = None
        server_timings = []
        for batch in central_difference_batches(epsilon=epsilon):
            policy_input = dict(request_base)
            policy_input["__crfs__"] = dict(request_base["__crfs__"])
            policy_input["__crfs__"]["terminal_branching"] = _branch_envelope(
                batch["residuals"], label=f"pullback_b{batch['batch_index']}",
            )
            response = client.infer(policy_input)
            current_ordinary = np.asarray(response["actions"], dtype=np.float64)
            terminal = np.asarray(
                response["terminal_branching"]["terminal_output_actions"],
                dtype=np.float64,
            )
            _require(current_ordinary.shape == (10, 7),
                     "late-flow QP ordinary action differs")
            _require(np.array_equal(terminal[0], current_ordinary),
                     "late-flow QP nominal terminal branch differs")
            if ordinary is None:
                ordinary = current_ordinary
                _, predict = _predictor(selector, ordinary)
            else:
                _require(np.array_equal(ordinary, current_ordinary),
                         "late-flow QP ordinary request is not exact")
            scored_batches.append({
                "batch_index": int(batch["batch_index"]),
                "terminal_bank_sha256": array_sha256(terminal),
                "risk_by_branch_and_row": _score_bank(predict, terminal),
            })
            server_timings.append(response.get("server_timing"))
        _require(ordinary is not None, "late-flow QP ordinary action is missing")
        nominal_risk, jacobian = pullback_jacobian(
            scored_batches=scored_batches, epsilon=epsilon,
        )
        qp_config = config["QP"]
        qp = solve_pullback_qp(
            nominal_risk=nominal_risk, jacobian=jacobian,
            constraint_rows=qp_config["constraint_rows"],
            risk_margin=float(qp_config["risk_margin"]),
            linearized_tightening=float(qp_config["linearized_tightening"]),
            trust_region_action=float(qp_config["trust_region_action"]),
        )
        nonlinear = None
        if qp["QP_valid"] and qp["correction"] is not None:
            final_input = dict(request_base)
            final_input["__crfs__"] = dict(request_base["__crfs__"])
            final_input["__crfs__"]["terminal_branching"] = _branch_envelope(
                correction_residual_bank(qp["correction"]), label="qp_recheck",
            )
            final_response = client.infer(final_input)
            final_ordinary = np.asarray(final_response["actions"], dtype=np.float64)
            final_bank = np.asarray(
                final_response["terminal_branching"]["terminal_output_actions"],
                dtype=np.float64,
            )
            _require(np.array_equal(final_ordinary, ordinary),
                     "late-flow QP recheck ordinary action differs")
            risk = _score_bank(predict, final_bank)[1]
            rows = [int(row) for row in qp_config["constraint_rows"]]
            nonlinear = {
                "terminal_action_sha256": array_sha256(final_bank[1]),
                "predicted_risk_by_row": risk,
                "maximum_constrained_risk": max(float(risk[row]) for row in rows),
                "constraints_satisfied": bool(
                    max(float(risk[row]) for row in rows)
                    <= -float(qp_config["risk_margin"])
                ),
                "server_timing": final_response.get("server_timing"),
            }
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "diagnostic_only": True,
            "source": source,
            "allocation": allocation,
            "config": config,
            "case_id": CASE_ID,
            "pairing": pairing,
            "query_dynamic_state_sha256": state_hash,
            "source_prefix_replay_action_count": len(source_query["episode"]["actions"]),
            "new_label_count": 0,
            "simulator_candidate_rollout_count": 0,
            "fixed_candidate_selection_count": 0,
            "QP_action_executed": False,
            "policy_server": server_identity,
            "rng_seed": int(seed),
            "ordinary_terminal_actions_sha256": array_sha256(ordinary),
            "nominal_predicted_risk_by_row": nominal_risk,
            "pullback_jacobian": jacobian,
            "pullback_jacobian_sha256": hashlib.sha256(
                np.asarray(jacobian, dtype=np.float64).tobytes()
            ).hexdigest(),
            "derivative_probe_batches": scored_batches,
            "derivative_server_timings": server_timings,
            "QP": qp,
            "nonlinear_terminal_recheck": nonlinear,
            "predicted_feasible": bool(
                nonlinear is not None and nonlinear["constraints_satisfied"]
            ),
            "control_authorized": False,
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["scientific_view"] = _scientific_view(result)
        result["result_payload_sha256"] = payload_sha256(
            result, "result_payload_sha256",
        )
        return result
    finally:
        if env is not None:
            env.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = evaluate(
        repo_root=args.repo_root.resolve(), manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit, host=args.host, port=args.port,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "status": value["status"],
        "nominal_max": max(value["nominal_predicted_risk_by_row"][:5]),
        "QP_valid": value["QP"]["QP_valid"],
        "predicted_feasible": value["predicted_feasible"],
        "final_max": (
            None if value["nonlinear_terminal_recheck"] is None
            else value["nonlinear_terminal_recheck"]["maximum_constrained_risk"]
        ),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
