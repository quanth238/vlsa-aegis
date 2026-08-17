#!/usr/bin/env python3
"""Validate late branching while scoring only terminal executable chunks.

This canary replays one immutable SafeLIBERO prefix to obtain a real policy
observation.  It performs one ordinary pi0.5 query and one 13-branch query
with the identical observation and RNG seed.  It does not execute candidate
branches in MuJoCo and does not evaluate or tune the frozen risk model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    EXPECTED_ACTION_HORIZON,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
)


def _archived_action(records: Sequence[Mapping[str, Any]], step: int) -> Any:
    import numpy as np

    record = records[step]
    _require(int(record["step"]) == step, "archived action index differs")
    action = np.asarray(record["env_step_input"], dtype=np.float64)
    _require(action.shape == (7,), "archived action shape differs")
    return action


def _candidate_bank(nominal: Any, names: Sequence[str]) -> Any:
    """Construct the frozen 13-bank in an identity local frame.

    The canary tests sampler semantics, not geometric direction quality.  The
    actual pilot supplies the registered robot-obstacle local frame.
    """

    import numpy as np

    from main.multilink_ellipsoid.active_boundary_search import candidate_definitions

    search_config = {
        "finite_search": {
            "action_coordinates": "physical_output_action_displacement",
            "spatial_basis": ["normal", "tangent_up", "tangent_side"],
            "coefficient_grid": [-1, 0, 1],
            "exclude_all_zero": True,
            "spatial_direction_count": 26,
            "correction_l2_action": 2.0,
            "action_limit": 1.0,
            "include_nominal": True,
            "candidate_count_per_job": 27,
            "endpoint_preservation": False,
            "preserve_rotation_and_gripper": True,
            "robust_boundary_margin_m": 0.001,
            "interpretation": "sampler_parity_canary_only",
        }
    }
    local_frame = {
        "normal": [1.0, 0.0, 0.0],
        "tangent_up": [0.0, 1.0, 0.0],
        "tangent_side": [0.0, 0.0, 1.0],
    }
    definitions = candidate_definitions(
        np.asarray(nominal[:5], dtype=np.float64),
        local_frame,
        search_config,
        "front_loaded",
    )
    by_name = {str(item["name"]): item for item in definitions}
    _require(set(names).issubset(by_name), "frozen canary candidates are missing")
    candidates = []
    for name in names:
        candidate = np.asarray(nominal, dtype=np.float64).copy()
        candidate[:5] = np.asarray(by_name[name]["actions"], dtype=np.float64)
        candidates.append(candidate)
    bank = np.asarray(candidates, dtype=np.float64)
    _require(bank.shape == (13, 10, 7), "frozen canary bank shape differs")
    return bank


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    experiment_config_path: Path,
    expected_commit: str,
    replica: str,
    host: str,
    port: int,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        _active_obstacle,
        _build_environment,
        _policy_observation,
        _runtime_imports,
        _server_identity,
        _settle,
        array_sha256,
        pairing_record,
        query_seed,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.shadow import allocation_record
    from main.multilink_ellipsoid.terminal_branching import (
        FROZEN_CANDIDATE_NAMES,
        RESULT_SCHEMA,
        build_envelope,
        load_config,
        payload_sha256,
    )

    started = time.perf_counter_ns()
    config = load_config(experiment_config_path)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    archived_actions = archived.get("actions")
    _require(
        isinstance(archived_actions, list)
        and len(archived_actions) == EXPECTED_ACTION_HORIZON,
        "Table-1 action horizon differs",
    )
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary manifest row differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    canary = config["sampler_canary"]
    _require(canary["case_id"] == CASE_ID, "sampler canary case differs")
    query_step = int(canary["query_step"])
    query_index = int(canary["query_index"])
    runtime = _runtime_imports(include_aegis=False)
    env = None
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        settled_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
        active_obstacle_name, _ = _active_obstacle(env, observation)
        pairing = pairing_record(
            case=case,
            selected_initial_state=selected_initial_state,
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=active_obstacle_name,
            settled_simulator_state=settled_state,
        )
        for step in range(query_step):
            observation, _, done, _ = env.step(
                _archived_action(archived_actions, step).tolist()
            )
            _require(not bool(done), "archived prefix completed before canary query")

        client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, port)
        server_identity = _server_identity(client)
        seed = query_seed(int(case["policy_noise_seed"]), query_index)
        ordinary_input = _policy_observation(
            runtime,
            observation,
            task_description=str(task.language),
            resize_size=224,
            rng_seed=seed,
        )
        ordinary_started = time.perf_counter_ns()
        ordinary_response = client.infer(ordinary_input)
        ordinary_seconds = (time.perf_counter_ns() - ordinary_started) * 1.0e-9
        ordinary = np.asarray(ordinary_response["actions"], dtype=np.float64)
        _require(ordinary.shape == (10, 7), "ordinary terminal chunk differs")

        candidates = _candidate_bank(ordinary, FROZEN_CANDIDATE_NAMES)
        envelope = build_envelope(
            ordinary,
            candidates,
            FROZEN_CANDIDATE_NAMES,
            branch_after_euler_step=int(config["flow"]["branch_after_euler_step"]),
        )
        branch_input = _policy_observation(
            runtime,
            observation,
            task_description=str(task.language),
            resize_size=224,
            rng_seed=seed,
        )
        branch_input["__crfs__"]["terminal_branching"] = envelope
        branch_started = time.perf_counter_ns()
        branch_response = client.infer(branch_input)
        branch_seconds = (time.perf_counter_ns() - branch_started) * 1.0e-9
        diagnostic = branch_response.get("terminal_branching")
        _require(isinstance(diagnostic, dict), "terminal branch diagnostic is missing")
        terminal = np.asarray(diagnostic["terminal_output_actions"], dtype=np.float64)
        _require(terminal.shape == (13, 10, 7), "terminal branch output bank differs")
        branch_zero = np.asarray(branch_response["actions"], dtype=np.float64)
        _require(branch_zero.shape == (10, 7), "terminal branch-zero output differs")
        branch_zero_difference = float(np.max(np.abs(branch_zero - ordinary)))
        internal_branch_zero_difference = float(
            np.max(np.abs(terminal[0] - ordinary))
        )
        diversity = float(np.max(np.abs(terminal[1:] - terminal[0:1])))
        batched_zero_model_drift = float(
            diagnostic["batched_zero_branch_max_abs_model_difference"]
        )
        _require(
            np.isfinite(batched_zero_model_drift)
            and batched_zero_model_drift >= 0.0,
            "batched zero-branch diagnostic differs",
        )
        threshold = float(canary["require_branch_zero_ordinary_action_max_abs"])
        gates = {
            "branch_zero_matches_ordinary": bool(
                branch_zero_difference <= threshold
                and internal_branch_zero_difference <= threshold
            ),
            "all_terminal_actions_finite": bool(np.all(np.isfinite(terminal))),
            "nonzero_branch_diversity": bool(diversity > 1.0e-12),
            "candidate_names_exact": diagnostic.get("candidate_names")
            == list(FROZEN_CANDIDATE_NAMES),
            "candidate_count_exact": int(terminal.shape[0]) == 13,
            "critic_not_run_inside_sampler": diagnostic.get(
                "risk_scored_inside_sampler"
            ) is False and diagnostic.get("selected_candidate") is None,
        }
        scientific_view = {
            "case_id": CASE_ID,
            "query_step": query_step,
            "query_index": query_index,
            "rng_seed": seed,
            "pairing": pairing,
            "candidate_names": list(FROZEN_CANDIDATE_NAMES),
            "ordinary_action_sha256": array_sha256(ordinary),
            "terminal_bank_sha256": array_sha256(terminal),
            "terminal_branch_hashes": [array_sha256(item) for item in terminal],
            "branch_zero_max_abs_difference": branch_zero_difference,
            "internal_branch_zero_max_abs_difference": internal_branch_zero_difference,
            "maximum_terminal_branch_diversity": diversity,
            "batched_zero_branch_max_abs_model_difference": (
                batched_zero_model_drift
            ),
            "gates": gates,
        }
        result: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA,
            "status": "passing" if all(gates.values()) else "scientific_no_go",
            "replica": str(replica),
            "scientific_result": True,
            "source": source,
            "allocation": allocation,
            "experiment_config": {
                "path": str(experiment_config_path),
                "file_sha256": config["config_file_sha256"],
                "payload_sha256": config["config_payload_sha256"],
            },
            "archived_source": {
                "path": str(archived_path),
                "file_sha256": ARCHIVED_FILE_SHA256,
                "payload_sha256": ARCHIVED_PAYLOAD_SHA256,
            },
            "server_identity": server_identity,
            "timing": {
                "ordinary_inference_seconds": ordinary_seconds,
                "terminal_branch_inference_seconds": branch_seconds,
                "total_wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
            },
            "scientific_view": scientific_view,
        }
        result["result_payload_sha256"] = payload_sha256(
            result, "result_payload_sha256"
        )
        return result
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--replica", choices=("producer", "replay"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        expected_commit=args.expected_commit,
        replica=args.replica,
        host=args.host,
        port=args.port,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "replica": result["replica"],
        "result_payload_sha256": result["result_payload_sha256"],
        "gates": result["scientific_view"]["gates"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
