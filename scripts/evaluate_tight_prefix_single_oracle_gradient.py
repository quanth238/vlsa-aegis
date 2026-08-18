#!/usr/bin/env python3
"""Run or exactly replay the one-case E15 oracle/critic gradient test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_tight_prefix_oracle_flow_gradient import (
    _evaluate_bank,
    _load_bound_artifacts,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _bound_inputs(
    repo_root: Path, config: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    case = config["case"]
    adapter = {
        "source": config["source"],
        "cases": [{
            "case_id": case["case_id"],
            "split": "test",
            "tight_case_file_sha256": case["tight_case_file_sha256"],
        }],
    }
    tight_config, _, model_record, exact_records = _load_bound_artifacts(
        repo_root=repo_root, config=adapter,
    )
    exact_record = exact_records[case["case_id"]]
    _require(
        exact_record.get("result_payload_sha256")
        == case["tight_case_payload_sha256"],
        "single-oracle-gradient exact case payload differs",
    )
    case_config = tight_config["cases"][int(case["tight_dataset_case_index"])]
    _require(
        case_config["case_id"] == case["case_id"]
        and int(case_config["state_step"]) == int(case["state_step"])
        and case_config["split"] == "test",
        "single-oracle-gradient dataset case differs",
    )
    return tight_config, model_record, exact_record["case"], case_config


def _phase(
    *, repo_root: Path, tight_config: Mapping[str, Any],
    case_config: Mapping[str, Any], exact_source_case: Mapping[str, Any],
    state_payload: Mapping[str, Any], primary_rows: Sequence[int],
    car_limit: float, names: Sequence[str], terminal_actions: Any,
    ordinary_actions: Any,
) -> list[dict[str, Any]]:
    return _evaluate_bank(
        repo_root=repo_root, tight_config=tight_config,
        case_config=case_config, names=names,
        terminal_actions=terminal_actions, ordinary_actions=ordinary_actions,
        exact_source_case=exact_source_case, state_payload=state_payload,
        primary_rows=primary_rows, car_limit=car_limit,
    )["records"]


def _physical_false_safes(phases: Mapping[str, Any]) -> int:
    return sum(
        int(record["represented_geometry_physical_false_safe"])
        for value in phases.values()
        if isinstance(value, list)
        for record in value
        if isinstance(record, Mapping)
        and "represented_geometry_physical_false_safe" in record
    )


def _decision(
    *, raw_records: Sequence[Mapping[str, Any]],
    oracle_records: Optional[Sequence[Mapping[str, Any]]] = None,
    comparison_metrics: Optional[Mapping[str, Any]] = None,
) -> str:
    if float(raw_records[0]["hard_primary_future_risk"]) <= 0.0:
        return "unsuitable_raw_terminal_not_unsafe"
    if oracle_records is None:
        return "oracle_gradient_unavailable"
    if not any(bool(record["physical_primary_safe"]) for record in oracle_records):
        return "oracle_local_correction_no_go"
    if comparison_metrics is None:
        return "comparison_missing"
    if bool(comparison_metrics["learned_safe"]):
        return "oracle_and_MLP_succeed"
    return "oracle_succeeds_MLP_fails"


def _public_result(
    *, repo_root: Path, config_path: Path, config: Mapping[str, Any],
    expected_commit: str, replica: str, model_sha256: str,
    action_bundle: Mapping[str, Any], phase_records: Mapping[str, Any],
    exact_gradient: Optional[Sequence[float]],
    learned_gradient: Optional[Sequence[float]],
    selected_radius: Optional[float], direction_audit: Optional[Mapping[str, Any]],
    metrics: Optional[Mapping[str, Any]], server_identity: Optional[Mapping[str, Any]],
    frozen_producer_binding: Optional[Mapping[str, Any]], started_ns: int,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.pi05_palm_l6_source_selection import (
        cpu_allocation_record,
    )
    from main.multilink_ellipsoid.shadow import allocation_record
    from main.multilink_ellipsoid.tight_prefix_single_oracle_gradient import (
        RESULT_SCHEMA, file_sha256, payload_sha256,
    )

    raw = phase_records["raw"]
    oracle = phase_records.get("oracle_line")
    decision = _decision(
        raw_records=raw, oracle_records=oracle, comparison_metrics=metrics,
    )
    false_safes = _physical_false_safes(phase_records)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete_single_oracle_gradient",
        "scientific_result": True,
        "replica": replica,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": (
            allocation_record() if replica == "producer"
            else cpu_allocation_record()
        ),
        "config_file_sha256": file_sha256(config_path),
        "config_payload_sha256": config["config_payload_sha256"],
        "server_identity": server_identity,
        "frozen_producer_binding": frozen_producer_binding,
        "frozen_model_sha256": model_sha256,
        "case_id": config["case"]["case_id"],
        "state_step": int(config["case"]["state_step"]),
        "action_bundle": action_bundle,
        "phase_records": phase_records,
        "exact_gradient": None if exact_gradient is None else list(exact_gradient),
        "learned_gradient": None if learned_gradient is None else list(learned_gradient),
        "selected_exact_safe_oracle_radius_l2": selected_radius,
        "direction_audit": direction_audit,
        "comparison_metrics": metrics,
        "summary": {
            "decision": decision,
            "raw_terminal_unsafe": bool(
                float(raw[0]["hard_primary_future_risk"]) > 0.0
            ),
            "exact_safe_oracle_within_radius": bool(
                oracle is not None
                and any(record["physical_primary_safe"] for record in oracle)
            ),
            "oracle_safe": None if metrics is None else bool(metrics["oracle_safe"]),
            "MLP_safe": None if metrics is None else bool(metrics["learned_safe"]),
            "physical_false_safe_count": int(false_safes),
            "full_episode_authorized": decision == "oracle_and_MLP_succeed",
            "retraining_authorized": False,
        },
        "model_training_performed": False,
        "QP_or_full_episode_performed": False,
        "timing": {"total_wall_seconds": (time.perf_counter_ns() - started_ns) * 1e-9},
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256",
    )
    return result


def run_producer(
    *, repo_root: Path, config_path: Path, expected_commit: str,
    host: str, port: int,
) -> dict[str, Any]:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION, TABLE_SETTLE_ACTIONS, _active_obstacle,
        _build_environment, _policy_observation, _runtime_imports, _server_identity,
        _settle, array_sha256, query_seed, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.tight_prefix_oracle_flow_gradient import (
        central_gradient, finite_difference_residuals, terminal_branch_envelope,
    )
    from main.multilink_ellipsoid.tight_prefix_single_oracle_gradient import (
        comparison_metrics, comparison_residuals, load_config,
        oracle_line_residuals, terminal_branch_batches,
    )

    started = time.perf_counter_ns()
    config = load_config(config_path)
    tight, model, exact_case, case_config = _bound_inputs(repo_root, config)
    state_payload = model["state_payload"]
    primary_rows = tuple(config["finite_difference"]["primary_rows"])
    car_limit = float(config["gate"]["paper_CAR_threshold_m"])
    runtime = _runtime_imports(include_aegis=False)
    manifest = read_jsonl(repo_root / tight["source"]["population_manifest"])
    by_id = {str(row["case_id"]): row for row in manifest}
    case = by_id[config["case"]["case_id"]]
    validate_case_row(case, repo_root)
    client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, port)
    server = _server_identity(client)
    env = None

    def terminal_query(
        observation: Mapping[str, Any], task: Any, seed: int, ordinary: Any,
        names: Sequence[str], residuals: Sequence[Any],
    ) -> tuple[Any, dict[str, Any]]:
        batches = terminal_branch_batches(names, residuals)
        scientific = []
        batch_receipts = []
        for batch in batches:
            envelope = terminal_branch_envelope(
                batch["request_names"], batch["request_residuals"],
                branch_after_euler_step=int(
                    config["flow"]["branch_after_euler_step"]
                ),
            )
            request = _policy_observation(
                runtime, observation, task_description=str(task.language),
                resize_size=224, rng_seed=seed,
            )
            request["__crfs__"]["terminal_branching"] = envelope
            diagnostic = client.infer(request)["terminal_branching"]
            terminal = np.asarray(
                diagnostic["terminal_output_actions"], dtype=np.float64,
            )
            count = len(batch["scientific_names"])
            _require(
                diagnostic["candidate_names"] == batch["request_names"]
                and terminal.shape == (13, 10, 7)
                and float(np.max(np.abs(terminal[0] - ordinary))) == 0.0,
                "single-oracle-gradient branch-zero pairing differs",
            )
            scientific.append(terminal[1:1 + count])
            batch_receipts.append({
                "batch_index": batch["batch_index"],
                "scientific_names": batch["scientific_names"],
                "request_names": batch["request_names"],
                "batched_zero_branch_max_abs_model_difference": float(
                    diagnostic["batched_zero_branch_max_abs_model_difference"]
                ),
            })
        terminal = np.concatenate(scientific, axis=0)
        _require(
            terminal.shape == (len(names), 10, 7),
            "single-oracle-gradient packed terminal result differs",
        )
        return terminal, {
            "candidate_names": list(names),
            "requested_residual_sha256": array_sha256(
                np.asarray(residuals, dtype=np.float64)
            ),
            "terminal_action_sha256": array_sha256(terminal),
            "branch_zero_exact": True,
            "batch_count": len(batch_receipts),
            "batches": batch_receipts,
        }

    try:
        env, task, observation, _ = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION,
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        source = _load(Path(case_config["source_result"]))
        active, _ = _active_obstacle(env, observation)
        _require(
            active == source["population_binding"]["selection"]["active_obstacle_name"],
            "single-oracle-gradient active obstacle differs",
        )
        binding = source["archived_table1"]
        archived_path = Path(binding["path"])
        archived = _load(archived_path)
        _require(
            binding.get("read_only") is True
            and _file_sha256(archived_path) == binding["file_sha256"]
            and archived.get("result_payload_sha256") == binding["result_payload_sha256"],
            "single-oracle-gradient archived prefix differs",
        )
        action_rows = {int(row["step"]): row for row in archived["actions"]}
        state_step = int(config["case"]["state_step"])
        for step in range(state_step):
            observation, _, done, _ = env.step(action_rows[step]["executed"])
            _require(not bool(done), "single-oracle-gradient prefix ended early")
        state_hash = hashlib.sha256(
            np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
        ).hexdigest()
        _require(
            state_hash == exact_case["source_snapshot_sha256"]
            == source["state"]["source_snapshot_sha256"],
            "single-oracle-gradient warning state differs",
        )
        seed = query_seed(int(case["policy_noise_seed"]), state_step // 5)
        ordinary_request = _policy_observation(
            runtime, observation, task_description=str(task.language),
            resize_size=224, rng_seed=seed,
        )
        ordinary = np.asarray(client.infer(ordinary_request)["actions"], dtype=np.float64)
        _require(ordinary.shape == (10, 7), "single-oracle-gradient ordinary action differs")

        action_bundle: dict[str, Any] = {
            "state_sha256": state_hash,
            "query_seed": int(seed),
            "ordinary_terminal_actions": ordinary.tolist(),
            "ordinary_terminal_action_sha256": array_sha256(ordinary),
            "raw": {"names": ["raw_terminal_pi05"], "terminal_actions": [ordinary.tolist()]},
        }
        phases: dict[str, Any] = {}
        phases["raw"] = _phase(
            repo_root=repo_root, tight_config=tight, case_config=case_config,
            exact_source_case=exact_case, state_payload=state_payload,
            primary_rows=primary_rows, car_limit=car_limit,
            names=action_bundle["raw"]["names"],
            terminal_actions=action_bundle["raw"]["terminal_actions"],
            ordinary_actions=ordinary,
        )
        if float(phases["raw"][0]["hard_primary_future_risk"]) <= 0.0:
            return _public_result(
                repo_root=repo_root, config_path=config_path, config=config,
                expected_commit=expected_commit, replica="producer",
                model_sha256=model["model_sha256"], action_bundle=action_bundle,
                phase_records=phases, exact_gradient=None, learned_gradient=None,
                selected_radius=None, direction_audit=None, metrics=None,
                server_identity=server, frozen_producer_binding=None,
                started_ns=started,
            )

        epsilon = float(config["finite_difference"]["epsilon"])
        fd_names, fd_residuals = finite_difference_residuals(
            np.zeros((10, 3), dtype=np.float64), epsilon=epsilon,
        )
        fd_terminal, fd_receipt = terminal_query(
            observation, task, seed, ordinary, fd_names, fd_residuals,
        )
        action_bundle["finite_difference"] = {
            "names": fd_names, "terminal_actions": fd_terminal.tolist(),
            "policy_receipt": fd_receipt,
        }
        phases["finite_difference"] = _phase(
            repo_root=repo_root, tight_config=tight, case_config=case_config,
            exact_source_case=exact_case, state_payload=state_payload,
            primary_rows=primary_rows, car_limit=car_limit, names=fd_names,
            terminal_actions=fd_terminal, ordinary_actions=ordinary,
        )
        exact_gradient = central_gradient(
            phases["finite_difference"], key="hard_primary_future_risk",
            epsilon=epsilon,
        )
        learned_gradient = central_gradient(
            phases["finite_difference"], key="smooth_global_prediction",
            epsilon=epsilon,
        )

        line_names, line_residuals = oracle_line_residuals(
            exact_gradient, config["support"]["oracle_line_radii_l2"],
        )
        line_terminal, line_receipt = terminal_query(
            observation, task, seed, ordinary, line_names, line_residuals,
        )
        action_bundle["oracle_line"] = {
            "names": line_names, "terminal_actions": line_terminal.tolist(),
            "policy_receipt": line_receipt,
        }
        phases["oracle_line"] = _phase(
            repo_root=repo_root, tight_config=tight, case_config=case_config,
            exact_source_case=exact_case, state_payload=state_payload,
            primary_rows=primary_rows, car_limit=car_limit, names=line_names,
            terminal_actions=line_terminal, ordinary_actions=ordinary,
        )
        safe_indices = [
            index for index, record in enumerate(phases["oracle_line"])
            if bool(record["physical_primary_safe"])
        ]
        if not safe_indices:
            return _public_result(
                repo_root=repo_root, config_path=config_path, config=config,
                expected_commit=expected_commit, replica="producer",
                model_sha256=model["model_sha256"], action_bundle=action_bundle,
                phase_records=phases, exact_gradient=exact_gradient,
                learned_gradient=learned_gradient, selected_radius=None,
                direction_audit=None, metrics=None, server_identity=server,
                frozen_producer_binding=None, started_ns=started,
            )

        safe_index = safe_indices[0]
        selected_radius = float(config["support"]["oracle_line_radii_l2"][safe_index])
        other_names, other_residuals, direction_audit = comparison_residuals(
            learned_gradient, exact_gradient, radius=selected_radius,
            random_seed=int(config["comparison"]["random_seed"]),
        )
        other_terminal, comparison_receipt = terminal_query(
            observation, task, seed, ordinary, other_names, other_residuals,
        )
        names = [
            "comparison_anchor", "learned_down", "oracle_down", "oracle_up",
            "random_0", "random_1", "random_2", "random_3",
        ]
        terminal = np.concatenate([
            other_terminal[:2], line_terminal[safe_index:safe_index + 1],
            other_terminal[2:],
        ], axis=0)
        action_bundle["comparison"] = {
            "names": names, "terminal_actions": terminal.tolist(),
            "policy_receipt": comparison_receipt,
            "oracle_action_source": line_names[safe_index],
        }
        phases["comparison"] = _phase(
            repo_root=repo_root, tight_config=tight, case_config=case_config,
            exact_source_case=exact_case, state_payload=state_payload,
            primary_rows=primary_rows, car_limit=car_limit, names=names,
            terminal_actions=terminal, ordinary_actions=ordinary,
        )
        metrics = comparison_metrics(phases["comparison"])
        return _public_result(
            repo_root=repo_root, config_path=config_path, config=config,
            expected_commit=expected_commit, replica="producer",
            model_sha256=model["model_sha256"], action_bundle=action_bundle,
            phase_records=phases, exact_gradient=exact_gradient,
            learned_gradient=learned_gradient, selected_radius=selected_radius,
            direction_audit=direction_audit, metrics=metrics,
            server_identity=server, frozen_producer_binding=None,
            started_ns=started,
        )
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def run_replay(
    *, repo_root: Path, config_path: Path, expected_commit: str,
    producer_path: Path, accepted_producer_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.tight_prefix_oracle_flow_gradient import (
        central_gradient,
    )
    from main.multilink_ellipsoid.tight_prefix_single_oracle_gradient import (
        RESULT_SCHEMA, comparison_metrics, load_config, payload_sha256,
    )

    started = time.perf_counter_ns()
    config = load_config(config_path)
    producer = _load(producer_path)
    _require(
        producer.get("schema_version") == RESULT_SCHEMA
        and producer.get("replica") == "producer"
        and producer.get("source", {}).get("commit") == accepted_producer_commit
        and producer.get("result_payload_sha256")
        == payload_sha256(producer, "result_payload_sha256")
        and producer.get("config_payload_sha256") == config["config_payload_sha256"],
        "single-oracle-gradient producer binding differs",
    )
    tight, model, exact_case, case_config = _bound_inputs(repo_root, config)
    primary_rows = tuple(config["finite_difference"]["primary_rows"])
    car_limit = float(config["gate"]["paper_CAR_threshold_m"])
    state_payload = model["state_payload"]
    bundle = producer["action_bundle"]
    ordinary = bundle["ordinary_terminal_actions"]
    phases: dict[str, Any] = {}
    for phase_name in ("raw", "finite_difference", "oracle_line", "comparison"):
        if phase_name not in bundle:
            continue
        phase_bundle = bundle[phase_name]
        phases[phase_name] = _phase(
            repo_root=repo_root, tight_config=tight, case_config=case_config,
            exact_source_case=exact_case, state_payload=state_payload,
            primary_rows=primary_rows, car_limit=car_limit,
            names=phase_bundle["names"],
            terminal_actions=phase_bundle["terminal_actions"],
            ordinary_actions=ordinary,
        )
    exact_gradient = None
    learned_gradient = None
    if "finite_difference" in phases:
        epsilon = float(config["finite_difference"]["epsilon"])
        exact_gradient = central_gradient(
            phases["finite_difference"], key="hard_primary_future_risk",
            epsilon=epsilon,
        )
        learned_gradient = central_gradient(
            phases["finite_difference"], key="smooth_global_prediction",
            epsilon=epsilon,
        )
    metrics = None
    if "comparison" in phases:
        metrics = comparison_metrics(phases["comparison"])
    return _public_result(
        repo_root=repo_root, config_path=config_path, config=config,
        expected_commit=expected_commit, replica="replay",
        model_sha256=model["model_sha256"], action_bundle=bundle,
        phase_records=phases, exact_gradient=exact_gradient,
        learned_gradient=learned_gradient,
        selected_radius=producer["selected_exact_safe_oracle_radius_l2"],
        direction_audit=producer["direction_audit"], metrics=metrics,
        server_identity=None,
        frozen_producer_binding={
            "path": str(producer_path), "file_sha256": _file_sha256(producer_path),
            "payload_sha256": producer["result_payload_sha256"],
            "accepted_commit": accepted_producer_commit,
        },
        started_ns=started,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--replica", choices=("producer", "replay"), required=True)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--producer", type=Path)
    parser.add_argument("--accepted-producer-commit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.replica == "producer":
        _require(args.host is not None and args.port is not None,
                 "single-oracle-gradient producer server differs")
        result = run_producer(
            repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
            expected_commit=args.expected_commit, host=args.host, port=args.port,
        )
    else:
        _require(
            args.producer is not None and args.accepted_producer_commit is not None,
            "single-oracle-gradient replay producer binding differs",
        )
        result = run_replay(
            repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
            expected_commit=args.expected_commit,
            producer_path=args.producer.resolve(),
            accepted_producer_commit=args.accepted_producer_commit,
        )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"], "decision": result["summary"]["decision"],
        "raw_terminal_unsafe": result["summary"]["raw_terminal_unsafe"],
        "exact_safe_oracle_within_radius": result["summary"]["exact_safe_oracle_within_radius"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
