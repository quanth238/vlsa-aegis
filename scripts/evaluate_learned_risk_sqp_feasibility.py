#!/usr/bin/env python3
"""Compare EE-only and EE+palm+L5 learned-risk SQPs without execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.learned_risk_sqp import (
    RESULT_SCHEMA, canonical, load_config, payload_sha256,
    scientific_view, solve_sqp,
)
from scripts.evaluate_terminalized_late_flow_task_success import (
    LiveTerminalizedSelector, RawRobotContactMonitor, _allocation_record,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256, ARCHIVED_PAYLOAD_SHA256, CASE_ID,
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _predictor(selector: LiveTerminalizedSelector, ordinary: Any):
    import copy
    import numpy as np

    from main.multilink_ellipsoid.compact_safety_coordinate_q import (
        MODEL_ROWS, safety_coordinate_feature,
    )
    from main.multilink_ellipsoid.compact_selector_ablation import (
        predict_serialized_mlp_float32,
    )

    exact = selector._current_exact_case()
    ordinary = np.asarray(ordinary, dtype=np.float64)
    _require(ordinary.shape == (10, 7), "learned-risk SQP ordinary differs")
    feature_case = copy.deepcopy(exact)
    feature_case["source_nominal_five_action_chunk"] = ordinary[:5].tolist()

    def predict(candidate_actions: Any) -> list[float]:
        actions = np.asarray(candidate_actions, dtype=np.float64)
        _require(actions.shape == (5, 7),
                 "learned-risk SQP candidate differs")
        candidate = {"source_executed_actions": actions.tolist()}
        features = [
            safety_coordinate_feature(
                feature_case, candidate, row, translation_scale=0.05,
            )
            for row in MODEL_ROWS
        ]
        values = predict_serialized_mlp_float32(features, selector.state_payload)
        _require(len(values) == 7 and all(len(value) == 1 for value in values),
                 "learned-risk SQP model output differs")
        return [float(value[0]) for value in values]

    return exact, predict


def evaluate(
    *, repo_root: Path, manifest_path: Path, archived_path: Path,
    config_path: Path, expected_commit: str, replica: str,
    host: str, port: int, output_path: Path,
    frozen_producer_result_path: Optional[Path] = None,
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
    _require(replica in ("producer", "replay"),
             "learned-risk SQP replica differs")
    _require((replica == "replay") is (frozen_producer_result_path is not None),
             "learned-risk SQP producer binding differs")
    source = _git_identity(repo_root, expected_commit)
    allocation = _allocation_record(require_h100=replica == "producer")
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256,
             "learned-risk SQP archive differs")
    archived = _load(archived_path)
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256,
             "learned-risk SQP archive payload differs")
    source_query_path = Path(config["source_query"]["result_path"])
    _require(
        _file_sha256(source_query_path)
        == config["source_query"]["result_file_sha256"],
        "learned-risk SQP source result file differs",
    )
    source_query = _load(source_query_path)
    _require(
        source_query.get("result_payload_sha256")
        == config["source_query"]["result_payload_sha256"]
        and source_query["episode"]["terminal_reason"]
        == config["source_query"]["terminal_reason"]
        and source_query["episode"]["action_count"]
        == config["source_query"]["replay_action_count"],
        "learned-risk SQP source result differs",
    )
    rows = [row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID]
    _require(len(rows) == 1, "learned-risk SQP manifest differs")
    case = rows[0]
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
                 "learned-risk SQP pairing differs")
        monitor = RawRobotContactMonitor(
            env, initial_obstacle, _contact_model_authority(env, obstacle_name),
            source_query["config"]["physical_contact_groups"],
        )
        for row in source_query["episode"]["actions"]:
            observation, _, done, _, internal = monitor.execute(
                row["executed_action"], step=int(row["step"]),
            )
            _require(not done and int(internal["substep_count"]) == 25,
                     "learned-risk SQP prefix differs")
        physical = monitor.summary()
        _require(
            int(physical["robot_contact_sample_count"]) == 0
            and physical["first_paper_CAR"] is None,
            "learned-risk SQP query reconstruction is physically invalid",
        )
        state_hash = hashlib.sha256(
            np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
        ).hexdigest()
        _require(
            state_hash == source_query["episode"]["terminal_dynamic_state_sha256"],
            "learned-risk SQP query state differs",
        )
        producer_result = None
        producer_binding = None
        server_identity = None
        if replica == "producer":
            client = runtime["websocket_client_policy"].WebsocketClientPolicy(
                host, int(port),
            )
            server_identity = _server_identity(client)
            policy_input = _policy_observation(
                runtime, observation, task_description=str(task.language),
                resize_size=224,
                rng_seed=query_seed(
                    int(case["policy_noise_seed"]),
                    int(config["source_query"]["query_index"]),
                ),
            )
            response = client.infer(policy_input)
            ordinary = np.asarray(response["actions"], dtype=np.float64)
            policy_timing = response.get("server_timing")
        else:
            producer_result = _load(frozen_producer_result_path)
            _require(
                producer_result.get("schema_version") == RESULT_SCHEMA
                and producer_result.get("status") == "complete"
                and producer_result.get("replica") == "producer"
                and producer_result.get("result_payload_sha256")
                == payload_sha256(producer_result, "result_payload_sha256"),
                "learned-risk SQP frozen producer differs",
            )
            ordinary = np.asarray(
                producer_result["ordinary_terminal_actions"], dtype=np.float64,
            )
            policy_timing = None
            producer_binding = {
                "path": str(frozen_producer_result_path),
                "file_sha256": _file_sha256(frozen_producer_result_path),
                "payload_sha256": producer_result["result_payload_sha256"],
            }
        _require(ordinary.shape == (10, 7),
                 "learned-risk SQP ordinary shape differs")
        selector = LiveTerminalizedSelector(
            repo_root=repo_root, runtime=runtime, env=env,
            obstacle_name=obstacle_name, archived=archived,
            config=config, client=None,
        )
        exact, predict = _predictor(selector, ordinary)
        nominal_risk = predict(ordinary[:5])
        settings = config["SQP"]
        arms = {
            name: solve_sqp(
                ordinary[:5], predict, constraint_rows=constraint_rows,
                maximum_iterations=int(settings["maximum_iterations"]),
                finite_difference_action=float(settings["finite_difference_action"]),
                trust_region_action=float(settings["trust_region_action"]),
                maximum_total_correction_action=float(
                    settings["maximum_total_correction_action"]
                ),
                line_search_fractions=settings["line_search_fractions"],
                risk_margin=float(settings["risk_margin"]),
            )
            for name, constraint_rows in settings["arms"].items()
        }
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "replica": replica,
            "source": source,
            "allocation": allocation,
            "config": config,
            "case_id": CASE_ID,
            "pairing": pairing,
            "pairing_sha256": hashlib.sha256(canonical(pairing)).hexdigest(),
            "source_query_result": {
                "path": str(source_query_path),
                "file_sha256": config["source_query"]["result_file_sha256"],
                "payload_sha256": config["source_query"]["result_payload_sha256"],
            },
            "query_dynamic_state_sha256": state_hash,
            "ordinary_terminal_actions": ordinary.tolist(),
            "ordinary_terminal_actions_sha256": array_sha256(ordinary),
            "nominal_predicted_risk_by_row": nominal_risk,
            "initial_exact_row_slack": exact["exact_group_target"][
                "initial_row_normalized_radial_slack"
            ],
            "arms": arms,
            "candidate_future_rollout_count": 0,
            "QP_action_executed": False,
            "policy_server": server_identity,
            "policy_server_timing": policy_timing,
            "producer_binding": producer_binding,
            "diagnostic_only": True,
            "control_authorized": False,
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["scientific_view"] = scientific_view(result)
        if producer_result is not None:
            _require(result["scientific_view"] == producer_result["scientific_view"],
                     "learned-risk SQP scientific replay differs")
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
    parser.add_argument("--replica", choices=("producer", "replay"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-producer-result", type=Path)
    args = parser.parse_args(argv)
    value = evaluate(
        repo_root=args.repo_root.resolve(), manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit, replica=args.replica,
        host=args.host, port=args.port, output_path=args.output.resolve(),
        frozen_producer_result_path=(
            None if args.frozen_producer_result is None
            else args.frozen_producer_result.resolve()
        ),
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "status": value["status"], "replica": value["replica"],
        "nominal_max_primary": max(value["nominal_predicted_risk_by_row"][:5]),
        "arms": {
            name: {
                "feasible": arm["nonlinear_constraints_satisfied"],
                "all_primary_safe": arm["all_primary_rows_safe"],
                "final_max": arm["final_maximum_constrained_risk"],
            }
            for name, arm in value["arms"].items()
        },
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
