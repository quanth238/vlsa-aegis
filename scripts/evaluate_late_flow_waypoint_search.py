#!/usr/bin/env python3
"""Search smooth late-flow waypoint branches without simulator lookahead."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.late_flow_waypoint_search import (
    RESULT_SCHEMA, canonical, load_config, payload_sha256,
    summarize_predictions, waypoint_batches,
)
from scripts.evaluate_learned_risk_sqp_feasibility import _predictor
from scripts.evaluate_terminalized_late_flow_task_success import (
    LiveTerminalizedSelector, RawRobotContactMonitor, _allocation_record,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256, ARCHIVED_PAYLOAD_SHA256, CASE_ID,
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _branch_envelope(batch: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    residuals = np.asarray(batch["residuals"], dtype=np.float64)
    _require(residuals.shape == (13, 10, 3),
             "late-flow waypoint residual bank differs")
    _require(np.array_equal(residuals[0], np.zeros((10, 3))),
             "late-flow waypoint nominal residual differs")
    names = ["nominal"] + [str(row["candidate_name"]) for row in batch["records"]]
    _require(len(names) == 13 and len(set(names)) == 13,
             "late-flow waypoint names differ")
    return {
        "schema_version": "crfs_terminal_branching.v1",
        "action_horizon": 10,
        "action_dimensions": [0, 1, 2],
        "candidate_names": names,
        "candidate_output_residuals": residuals.tolist(),
        "branch_after_euler_step": 8,
    }


def _scientific_view(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(result["case_id"]),
        "pairing": result["pairing"],
        "query_dynamic_state_sha256": str(result["query_dynamic_state_sha256"]),
        "source_prefix_replay_action_count": int(
            result["source_prefix_replay_action_count"]
        ),
        "rng_seed": int(result["rng_seed"]),
        "ordinary_terminal_actions_sha256": str(
            result["ordinary_terminal_actions_sha256"]
        ),
        "outward_world": list(result["outward_world"]),
        "candidate_records_sha256": str(result["candidate_records_sha256"]),
        "terminal_bank_hashes_sha256": str(
            result["terminal_bank_hashes_sha256"]
        ),
        "summary": result["summary"],
        "new_label_count": int(result["new_label_count"]),
        "simulator_candidate_rollout_count": int(
            result["simulator_candidate_rollout_count"]
        ),
        "action_executed": bool(result["action_executed"]),
        "control_authorized": bool(result["control_authorized"]),
    }


def evaluate(
    *, repo_root: Path, manifest_path: Path, archived_path: Path,
    config_path: Path, expected_commit: str, replica: str,
    host: str, port: int,
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
    _require(replica in ("producer", "replica"),
             "late-flow waypoint replica differs")
    source = _git_identity(repo_root, expected_commit)
    allocation = _allocation_record(require_h100=True)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256,
             "late-flow waypoint archived file differs")
    archived = _load(archived_path)
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256,
             "late-flow waypoint archived payload differs")
    source_path = Path(config["source_query"]["result_path"])
    _require(_file_sha256(source_path) == config["source_query"]["result_file_sha256"],
             "late-flow waypoint source result file differs")
    source_query = _load(source_path)
    _require(
        source_query.get("result_payload_sha256")
        == config["source_query"]["result_payload_sha256"]
        and source_query["episode"]["terminal_reason"]
        == config["source_query"]["terminal_reason"]
        and int(source_query["episode"]["action_count"])
        == int(config["source_query"]["replay_action_count"]),
        "late-flow waypoint source result differs",
    )
    matches = [
        row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID
    ]
    _require(len(matches) == 1, "late-flow waypoint manifest differs")
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
                 "late-flow waypoint pairing differs")
        monitor = RawRobotContactMonitor(
            env, initial_obstacle, _contact_model_authority(env, obstacle_name),
            source_query["config"]["physical_contact_groups"],
        )
        for row in source_query["episode"]["actions"]:
            observation, _, done, _, internal = monitor.execute(
                row["executed_action"], step=int(row["step"]),
            )
            _require(not done and int(internal["substep_count"]) == 25,
                     "late-flow waypoint saved prefix differs")
        physical = monitor.summary()
        _require(
            int(physical["robot_contact_sample_count"]) == 0
            and physical["first_paper_CAR"] is None,
            "late-flow waypoint query reconstruction is physically invalid",
        )
        state_hash = hashlib.sha256(
            np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
        ).hexdigest()
        _require(
            state_hash == source_query["episode"]["terminal_dynamic_state_sha256"],
            "late-flow waypoint query state differs",
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
            int(case["policy_noise_seed"]), int(config["source_query"]["query_index"]),
        )
        request_base = _policy_observation(
            runtime, observation, task_description=str(task.language),
            resize_size=224, rng_seed=seed,
        )
        outward = np.asarray(observation["robot0_eef_pos"], dtype=np.float64) - np.asarray(
            observation[obstacle_name + "_pos"], dtype=np.float64,
        )
        batches = waypoint_batches(
            outward_world=outward,
            seed=int(config["waypoint_search"]["sampling_seed"]),
            trust_region_action=float(config["waypoint_search"]["trust_region_action"]),
            translation_scale_m=float(
                config["waypoint_search"]["translation_scale_m_per_action_unit"]
            ),
        )
        ordinary = None
        predict = None
        nominal_risk = None
        records = []
        terminal_hashes = []
        server_timings = []
        for batch in batches:
            policy_input = dict(request_base)
            policy_input["__crfs__"] = dict(request_base["__crfs__"])
            policy_input["__crfs__"]["terminal_branching"] = _branch_envelope(batch)
            response = client.infer(policy_input)
            current_ordinary = np.asarray(response["actions"], dtype=np.float64)
            terminal = np.asarray(
                response["terminal_branching"]["terminal_output_actions"],
                dtype=np.float64,
            )
            _require(
                current_ordinary.shape == (10, 7)
                and terminal.shape == (13, 10, 7)
                and np.array_equal(terminal[0], current_ordinary),
                "late-flow waypoint terminal bank differs",
            )
            if ordinary is None:
                ordinary = current_ordinary
                _, predict = _predictor(selector, ordinary)
                nominal_effective = ordinary[:5].copy()
                nominal_effective[:, :3] = np.clip(
                    nominal_effective[:, :3], -1.0, 1.0,
                )
                nominal_risk = predict(nominal_effective)
            else:
                _require(np.array_equal(ordinary, current_ordinary),
                         "late-flow waypoint ordinary request is not exact")
            _require(predict is not None and nominal_risk is not None,
                     "late-flow waypoint predictor differs")
            terminal_hashes.append(array_sha256(terminal))
            server_timings.append(response.get("server_timing"))
            for metadata in batch["records"]:
                branch_index = int(metadata["branch_index"])
                effective = terminal[branch_index, :5].copy()
                effective[:, :3] = np.clip(effective[:, :3], -1.0, 1.0)
                ordinary_effective = ordinary[:5].copy()
                ordinary_effective[:, :3] = np.clip(
                    ordinary_effective[:, :3], -1.0, 1.0,
                )
                correction = effective[:, :3] - ordinary_effective[:, :3]
                second = np.diff(correction, n=2, axis=0)
                records.append({
                    **dict(metadata),
                    "terminal_action_sha256": array_sha256(effective),
                    "effective_first_five_actions": effective.tolist(),
                    "effective_correction_l2_action": float(
                        np.linalg.norm(correction)
                    ),
                    "effective_correction_linf_action": float(
                        np.max(np.abs(correction))
                    ),
                    "effective_endpoint_sum_action": np.sum(
                        correction, axis=0,
                    ).tolist(),
                    "effective_smoothness": float(np.linalg.norm(second)),
                    "predicted_risk_by_row": predict(effective),
                })
        _require(
            ordinary is not None and nominal_risk is not None
            and len(records) == 504 and len(terminal_hashes) == 42,
            "late-flow waypoint result population differs",
        )
        summary = summarize_predictions(
            nominal_risk=nominal_risk, records=records,
            physical_rows=config["scoring"]["physical_constraint_rows"],
            diagnostic_rows=config["scoring"]["diagnostic_proxy_rows"],
            threshold=float(config["scoring"]["predicted_safe_threshold"]),
        )
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "diagnostic_only": True,
            "replica": replica,
            "source": source,
            "allocation": allocation,
            "config": config,
            "case_id": CASE_ID,
            "pairing": pairing,
            "query_dynamic_state_sha256": state_hash,
            "source_prefix_replay_action_count": len(source_query["episode"]["actions"]),
            "new_label_count": 0,
            "simulator_candidate_rollout_count": 0,
            "action_executed": False,
            "control_authorized": False,
            "policy_server": server_identity,
            "rng_seed": int(seed),
            "outward_world": outward.tolist(),
            "ordinary_terminal_actions_sha256": array_sha256(ordinary),
            "candidate_records": records,
            "candidate_records_sha256": hashlib.sha256(canonical(records)).hexdigest(),
            "terminal_bank_hashes": terminal_hashes,
            "terminal_bank_hashes_sha256": hashlib.sha256(
                canonical(terminal_hashes)
            ).hexdigest(),
            "server_timings": server_timings,
            "summary": summary,
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
    parser.add_argument("--replica", choices=("producer", "replica"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = evaluate(
        repo_root=args.repo_root.resolve(), manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit, replica=args.replica,
        host=args.host, port=args.port,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({
        "status": value["status"],
        "replica": value["replica"],
        "candidate_count": value["summary"]["candidate_count"],
        "physical_safe_count": value["summary"]["predicted_physical_safe_count"],
        "proxy_inclusive_safe_count": value["summary"][
            "predicted_proxy_inclusive_safe_count"
        ],
        "minimum_proxy_inclusive_risk": value["summary"][
            "minimum_proxy_inclusive_risk_candidate"
        ]["maximum_proxy_inclusive_risk"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
