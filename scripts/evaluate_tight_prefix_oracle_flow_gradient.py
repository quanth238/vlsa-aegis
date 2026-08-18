#!/usr/bin/env python3
"""Run the three-root late-flow physical-oracle gradient canary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.audit_distal_compiled_box_risk_target import _evaluate_case
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


PRIMARY_GROUPS = (
    "palm", "finger1_base", "finger1_pad", "finger2_base", "finger2_pad",
    "L5",
)


def _numpy_model_prediction(
    features: Sequence[Sequence[float]], state_payload: Mapping[str, Any],
) -> Any:
    """Reproduce the frozen float32 SiLU MLP without training or autograd."""
    import numpy as np

    x = np.asarray(features, dtype=np.float32)
    mean = np.asarray(state_payload["feature_mean"], dtype=np.float32)
    scale = np.asarray(state_payload["feature_scale"], dtype=np.float32)
    target_mean = np.asarray(state_payload["target_mean"], dtype=np.float32)
    target_scale = np.asarray(state_payload["target_scale"], dtype=np.float32)
    state = state_payload["state_dict"]
    _require(x.ndim == 2 and x.shape[1] == len(mean) == 17,
             "oracle-flow-gradient model feature differs")
    value = (x - mean) / scale
    for index, prefix in enumerate(("0", "2", "4")):
        weight = np.asarray(state[prefix + ".weight"], dtype=np.float32)
        bias = np.asarray(state[prefix + ".bias"], dtype=np.float32)
        value = value @ weight.T + bias
        if index < 2:
            value = value / (1.0 + np.exp(-value))
    return value * target_scale + target_mean


def _critic_score(
    exact_case: Mapping[str, Any], actions: Sequence[Sequence[float]],
    state_payload: Mapping[str, Any], *, primary_rows: Sequence[int],
    beta: float = 20.0,
) -> dict[str, Any]:
    import numpy as np
    from main.multilink_ellipsoid.tight_prefix_obstacle_frame_q import (
        MODEL_ROWS, obstacle_frame_feature,
    )

    candidate = {"source_executed_actions": actions}
    features = [
        obstacle_frame_feature(
            exact_case, candidate, int(row), translation_scale=0.05,
            model_rows=MODEL_ROWS,
        )
        for row in primary_rows
    ]
    prediction = _numpy_model_prediction(features, state_payload).reshape(-1)
    maximum = float(np.max(float(beta) * prediction))
    global_value = float(
        (maximum + math.log(float(np.sum(np.exp(
            float(beta) * prediction - maximum
        )))) - math.log(len(prediction))) / float(beta)
    )
    return {
        "per_row_prediction": prediction.astype(float).tolist(),
        "smooth_global_prediction": global_value,
    }


def _candidate_summary(
    candidate: Mapping[str, Any], exact_source_case: Mapping[str, Any],
    state_payload: Mapping[str, Any], *, primary_rows: Sequence[int],
    car_limit: float, clipping_delta: float,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.tight_prefix_risk_q_diagnostic import (
        row_future_risks,
    )

    row_risks = row_future_risks(candidate, tuple(range(10)))
    hard = max(float(row_risks[int(row)]) for row in primary_rows)
    target = candidate["exact_group_target"]
    contacts = sum(
        int(target["group_contact_sample_count"][group])
        for group in PRIMARY_GROUPS
    )
    car = float(candidate["replayed_maximum_CAR_m"])
    known = bool(target["known_outcome"])
    safe = bool(known and hard <= 0.0 and contacts == 0 and car <= car_limit)
    score = _critic_score(
        exact_source_case, candidate["source_executed_actions"], state_payload,
        primary_rows=primary_rows,
    )
    return {
        "name": str(candidate["name"]),
        "known_outcome": known,
        "hard_primary_future_risk": float(hard),
        "primary_row_future_risks": [float(row_risks[row]) for row in primary_rows],
        "physical_primary_safe": safe,
        "primary_contact_sample_count": int(contacts),
        "replayed_maximum_CAR_m": car,
        "translation_clipping_max_abs": float(clipping_delta),
        "represented_geometry_physical_false_safe": bool(
            hard <= 0.0 and contacts > 0
        ),
        **score,
    }


def _action_overrides(
    names: Sequence[str], terminal_actions: Any, ordinary_actions: Any,
) -> tuple[list[dict[str, Any]], list[float]]:
    import numpy as np

    bank = np.asarray(terminal_actions, dtype=np.float64)
    ordinary = np.asarray(ordinary_actions, dtype=np.float64)
    _require(bank.shape == (len(names), 10, 7) and ordinary.shape == (10, 7),
             "oracle-flow-gradient terminal action shape differs")
    overrides = []
    clipping = []
    for order, (name, terminal) in enumerate(zip(names, bank)):
        effective = terminal[:5].copy()
        before = effective[:, :3].copy()
        effective[:, :3] = np.clip(effective[:, :3], -1.0, 1.0)
        delta = float(np.max(np.abs(effective[:, :3] - before)))
        clipping.append(delta)
        correction = effective[:, :3] - np.clip(ordinary[:5, :3], -1.0, 1.0)
        overrides.append({
            "name": str(name),
            "actions": effective.tolist(),
            "requested_alpha": float(np.linalg.norm(correction)),
            "effective_correction_l2_action": float(np.linalg.norm(correction)),
            "metadata": {"terminalized_late_flow": True, "order": int(order)},
        })
    return overrides, clipping


def _evaluate_bank(
    *, repo_root: Path, tight_config: Mapping[str, Any],
    case_config: Mapping[str, Any], names: Sequence[str], terminal_actions: Any,
    ordinary_actions: Any, exact_source_case: Mapping[str, Any],
    state_payload: Mapping[str, Any], primary_rows: Sequence[int],
    car_limit: float,
) -> dict[str, Any]:
    overrides, clipping = _action_overrides(names, terminal_actions, ordinary_actions)
    selection = dict(case_config)
    selection["candidate_names"] = list(names)
    selection["slab_initialization"] = "query_state_matching_source"
    evaluated = _evaluate_case(
        repo_root=repo_root,
        population_manifest=repo_root / tight_config["source"]["population_manifest"],
        geometry_config_path=(
            repo_root / tight_config["source"]["legacy_replay_geometry_config"]
        ),
        case_config=selection,
        audit_config={
            "rollout_scope": tight_config["rollout_scope"],
            "gate": tight_config["gate"],
            "exact_group_target": tight_config["exact_group_target"],
            "candidate_action_overrides": overrides,
        },
    )
    _require(
        evaluated.get("state_hash_matches") is True
        and evaluated.get("source_replay_exact") is True
        and [row["name"] for row in evaluated["candidates"]] == list(names),
        "oracle-flow-gradient exact bank replay differs",
    )
    summaries = [
        _candidate_summary(
            candidate, exact_source_case, state_payload,
            primary_rows=primary_rows, car_limit=car_limit,
            clipping_delta=clipping[index],
        )
        for index, candidate in enumerate(evaluated["candidates"])
    ]
    return {"exact_case": evaluated, "records": summaries}


def _source_bank_residuals(exact_case: Mapping[str, Any]) -> tuple[list[str], Any]:
    import numpy as np

    nominal = np.asarray(exact_case["source_nominal_five_action_chunk"], dtype=np.float64)
    candidates = exact_case["candidates"]
    names = [str(candidate["name"]) for candidate in candidates]
    residuals = np.zeros((len(candidates), 10, 3), dtype=np.float64)
    for index, candidate in enumerate(candidates):
        effective = np.asarray(candidate["source_executed_actions"], dtype=np.float64)
        residuals[index, :5] = effective[:, :3] - nominal[:, :3]
    _require(
        nominal.shape == (5, 7) and residuals.shape == (13, 10, 3)
        and float(np.max(np.abs(residuals[0]))) == 0.0,
        "oracle-flow-gradient frozen source bank differs",
    )
    return names, residuals


def _load_bound_artifacts(
    *, repo_root: Path, config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    from main.multilink_ellipsoid.terminal_branching import (
        VALIDATION_SCHEMA as TERMINAL_VALIDATION_SCHEMA,
        payload_sha256 as terminal_payload,
    )
    from main.multilink_ellipsoid.tight_prefix_obstacle_frame_q import (
        RESULT_SCHEMA as TRAINING_RESULT_SCHEMA,
        VALIDATION_SCHEMA as TRAINING_VALIDATION_SCHEMA,
        load_config as load_training_config,
        payload_sha256 as training_payload,
    )
    from main.multilink_ellipsoid.tight_prefix_risk_dataset import (
        CASE_SCHEMA as DATASET_CASE_SCHEMA,
        VALIDATION_SCHEMA as DATASET_VALIDATION_SCHEMA,
        load_config as load_dataset_config,
        payload_sha256 as dataset_payload,
    )

    source = config["source"]
    dataset_path = repo_root / source["tight_dataset_config"]
    _require(_file_sha256(dataset_path) == source["tight_dataset_config_file_sha256"],
             "oracle-flow-gradient dataset config file differs")
    dataset = load_dataset_config(dataset_path, repo_root=repo_root)
    _require(dataset["config_payload_sha256"] == source["tight_dataset_config_payload_sha256"],
             "oracle-flow-gradient dataset config payload differs")
    dataset_validation_path = Path(source["tight_dataset_validation"])
    dataset_validation = _load(dataset_validation_path)
    _require(
        _file_sha256(dataset_validation_path)
        == source["tight_dataset_validation_file_sha256"]
        and dataset_validation.get("schema_version") == DATASET_VALIDATION_SCHEMA
        and dataset_validation.get("validation_payload_sha256")
        == source["tight_dataset_validation_payload_sha256"]
        == dataset_payload(dataset_validation)
        and dataset_validation.get("dataset_gate_pass") is True,
        "oracle-flow-gradient dataset validation differs",
    )

    training_config_path = repo_root / source["obstacle_frame_training_config"]
    _require(
        _file_sha256(training_config_path)
        == source["obstacle_frame_training_config_file_sha256"],
        "oracle-flow-gradient training config differs",
    )
    training_config = load_training_config(training_config_path)
    result_path = Path(source["obstacle_frame_result"])
    result = _load(result_path)
    validation_path = Path(source["obstacle_frame_validation"])
    validation = _load(validation_path)
    _require(
        _file_sha256(result_path) == source["obstacle_frame_result_file_sha256"]
        and result.get("schema_version") == TRAINING_RESULT_SCHEMA
        and result.get("result_payload_sha256")
        == source["obstacle_frame_result_payload_sha256"]
        == training_payload(result, "result_payload_sha256")
        and _file_sha256(validation_path)
        == source["obstacle_frame_validation_file_sha256"]
        and validation.get("schema_version") == TRAINING_VALIDATION_SCHEMA
        and validation.get("validation_payload_sha256")
        == source["obstacle_frame_validation_payload_sha256"]
        == training_payload(validation, "validation_payload_sha256")
        and validation.get("independent_training_exact") is True,
        "oracle-flow-gradient frozen training artifacts differ",
    )
    model = result["arms"]["obstacle_frame_17D"]["model"]
    _require(
        model["model_sha256"] == source["obstacle_frame_model_sha256"],
        "oracle-flow-gradient model hash differs",
    )
    terminal_path = Path(source["terminal_sampler_validation"])
    terminal = _load(terminal_path)
    _require(
        _file_sha256(terminal_path) == source["terminal_sampler_validation_file_sha256"]
        and terminal.get("schema_version") == TERMINAL_VALIDATION_SCHEMA
        and terminal.get("validation_payload_sha256")
        == source["terminal_sampler_validation_payload_sha256"]
        == terminal_payload(terminal, "validation_payload_sha256")
        and terminal.get("status") == "passing"
        and terminal.get("scientific_result") is True
        and all(bool(value) for value in terminal.get("checks", {}).values()),
        "oracle-flow-gradient terminal sampler validation differs",
    )

    exact_cases = {}
    for selected in config["cases"]:
        path = Path(source["tight_dataset_producer_dir"]) / (selected["case_id"] + ".json")
        record = _load(path)
        _require(
            _file_sha256(path) == selected["tight_case_file_sha256"]
            and record.get("schema_version") == DATASET_CASE_SCHEMA
            and record.get("case_id") == selected["case_id"]
            and record.get("split") == "validation"
            and record.get("result_payload_sha256") == dataset_payload(record),
            "oracle-flow-gradient exact case artifact differs",
        )
        exact_cases[selected["case_id"]] = record
    return dataset, training_config, model, exact_cases


def run(
    *, repo_root: Path, config_path: Path, expected_commit: str,
    replica: str, host: str, port: int, run_root: Path,
) -> dict[str, Any]:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION, TABLE_SETTLE_ACTIONS, _active_obstacle,
        _build_environment, _policy_observation, _runtime_imports, _server_identity,
        _settle, array_sha256, query_seed, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import allocation_record
    from main.multilink_ellipsoid.tight_prefix_oracle_flow_gradient import (
        CASE_SCHEMA, case_direction_metrics, central_gradient,
        choose_closest_unsafe_safe_pair, choose_local_crossing,
        comparison_residuals, file_sha256, finite_difference_residuals,
        interpolation_residuals, load_config, payload_sha256,
        terminal_branch_envelope,
    )

    started = time.perf_counter_ns()
    _require(replica in ("producer", "replay"), "oracle-flow-gradient replica differs")
    config = load_config(config_path)
    tight_config, _, model_record, exact_records = _load_bound_artifacts(
        repo_root=repo_root, config=config,
    )
    state_payload = model_record["state_payload"]
    primary_rows = tuple(config["finite_difference"]["primary_rows"])
    car_limit = float(config["gate"]["paper_CAR_threshold_m"])
    runtime = _runtime_imports(include_aegis=False)
    manifest_rows = read_jsonl(repo_root / tight_config["source"]["population_manifest"])
    manifest_by_id = {str(row["case_id"]): row for row in manifest_rows}
    client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, port)
    server = _server_identity(client)

    def terminal_query(
        observation: Mapping[str, Any], task: Any, seed: int,
        ordinary: Any, names: Sequence[str], residuals: Sequence[Any],
    ) -> tuple[Any, dict[str, Any]]:
        requested = np.asarray(residuals, dtype=np.float64)
        labels = list(names)
        prepend = not bool(np.array_equal(
            requested[0], np.zeros((10, 3), dtype=np.float64)
        ))
        if prepend:
            labels = ["ordinary_reference"] + labels
            requested = np.concatenate([
                np.zeros((1, 10, 3), dtype=np.float64), requested,
            ], axis=0)
        envelope = terminal_branch_envelope(
            labels, requested,
            branch_after_euler_step=int(config["flow"]["branch_after_euler_step"]),
        )
        request = _policy_observation(
            runtime, observation, task_description=str(task.language),
            resize_size=224, rng_seed=seed,
        )
        request["__crfs__"]["terminal_branching"] = envelope
        response = client.infer(request)
        diagnostic = response["terminal_branching"]
        terminal = np.asarray(diagnostic["terminal_output_actions"], dtype=np.float64)
        _require(
            diagnostic["candidate_names"] == labels
            and terminal.shape == (len(labels), 10, 7)
            and float(np.max(np.abs(terminal[0] - ordinary))) == 0.0,
            "oracle-flow-gradient branch-zero pairing differs",
        )
        if prepend:
            terminal = terminal[1:]
        return terminal, {
            "candidate_names": list(names),
            "requested_residual_sha256": array_sha256(np.asarray(residuals, dtype=np.float64)),
            "terminal_action_sha256": array_sha256(terminal),
            "branch_zero_exact": True,
            "batched_zero_branch_max_abs_model_difference": float(
                diagnostic["batched_zero_branch_max_abs_model_difference"]
            ),
        }

    case_results = []
    for case_index, selected in enumerate(config["cases"]):
        case_id = str(selected["case_id"])
        tight_index = int(selected["tight_dataset_case_index"])
        case_config = dict(tight_config["cases"][tight_index])
        _require(
            case_config["case_id"] == case_id
            and int(case_config["state_step"]) == int(selected["state_step"])
            and case_config["split"] == "validation",
            "oracle-flow-gradient selected tight case differs",
        )
        exact_record = exact_records[case_id]
        exact_source_case = exact_record["case"]
        source = _load(Path(case_config["source_result"]))
        case = manifest_by_id[case_id]
        validate_case_row(case, repo_root)
        env = None
        try:
            env, task, observation, _ = _build_environment(
                runtime, case, render_resolution=TABLE_RENDER_RESOLUTION,
            )
            observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
            active_obstacle, _ = _active_obstacle(env, observation)
            _require(
                active_obstacle
                == source["population_binding"]["selection"]["active_obstacle_name"],
                "oracle-flow-gradient active obstacle differs",
            )
            archived_binding = source["archived_table1"]
            archived_path = Path(archived_binding["path"])
            archived = _load(archived_path)
            _require(
                archived_binding.get("read_only") is True
                and _file_sha256(archived_path)
                == archived_binding["file_sha256"]
                and archived.get("result_payload_sha256")
                == archived_binding["result_payload_sha256"]
                and archived.get("case_id") == case_id,
                "oracle-flow-gradient archived action ledger differs",
            )
            action_rows = {
                int(row["step"]): row for row in archived["actions"]
            }
            state_step = int(selected["state_step"])
            _require(
                set(range(state_step)).issubset(action_rows),
                "oracle-flow-gradient archived action prefix differs",
            )
            for step in range(state_step):
                observation, _, done, _ = env.step(action_rows[step]["executed"])
                _require(not bool(done), "oracle-flow-gradient prefix ended early")
            state_hash = hashlib.sha256(
                np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes()
            ).hexdigest()
            _require(
                state_hash == source["state"]["source_snapshot_sha256"]
                == exact_source_case["source_snapshot_sha256"],
                "oracle-flow-gradient live state differs",
            )
            _require(state_step % 5 == 0, "oracle-flow-gradient query boundary differs")
            seed = query_seed(int(case["policy_noise_seed"]), state_step // 5)
            ordinary_request = _policy_observation(
                runtime, observation, task_description=str(task.language),
                resize_size=224, rng_seed=seed,
            )
            ordinary = np.asarray(client.infer(ordinary_request)["actions"], dtype=np.float64)
            _require(ordinary.shape == (10, 7), "oracle-flow-gradient ordinary action differs")

            bank_names, bank_residuals = _source_bank_residuals(exact_source_case)
            bank_terminal, bank_policy = terminal_query(
                observation, task, seed, ordinary, bank_names, bank_residuals,
            )
            bank = _evaluate_bank(
                repo_root=repo_root, tight_config=tight_config,
                case_config=case_config, names=bank_names,
                terminal_actions=bank_terminal, ordinary_actions=ordinary,
                exact_source_case=exact_source_case, state_payload=state_payload,
                primary_rows=primary_rows, car_limit=car_limit,
            )
            pair = choose_closest_unsafe_safe_pair(
                bank["records"], bank_residuals,
                maximum_distance=float(config["localization"]["maximum_pair_distance_l2"]),
            )
            record: dict[str, Any] = {
                "case_id": case_id,
                "role": selected["role"],
                "state_step": state_step,
                "query_index": state_step // 5,
                "rng_seed": int(seed),
                "state_sha256": state_hash,
                "ordinary_action_sha256": array_sha256(ordinary),
                "bank_policy": bank_policy,
                "bank_records": bank["records"],
                "bank_pair": pair,
                "eligible": False,
                "ineligibility_reason": None,
            }
            if selected["role"] == "far_safe_control":
                record["ineligibility_reason"] = "preregistered_far_safe_control"
                case_results.append(record)
                continue
            if pair is None:
                record["ineligibility_reason"] = "no_frozen_bank_unsafe_safe_pair_within_radius"
                case_results.append(record)
                continue

            interpolation_names, interpolation_values = interpolation_residuals(
                bank_residuals[pair["unsafe_index"]],
                bank_residuals[pair["safe_index"]],
                segment_count=int(config["localization"]["interpolation_segment_count"]),
            )
            interpolation_terminal, interpolation_policy = terminal_query(
                observation, task, seed, ordinary,
                interpolation_names, interpolation_values,
            )
            interpolation = _evaluate_bank(
                repo_root=repo_root, tight_config=tight_config,
                case_config=case_config, names=interpolation_names,
                terminal_actions=interpolation_terminal, ordinary_actions=ordinary,
                exact_source_case=exact_source_case, state_payload=state_payload,
                primary_rows=primary_rows, car_limit=car_limit,
            )
            crossing = choose_local_crossing(
                interpolation["records"], interpolation_values,
                minimum_distance=float(config["localization"]["minimum_local_witness_distance_l2"]),
                maximum_distance=float(config["localization"]["maximum_local_witness_distance_l2"]),
            )
            record.update({
                "interpolation_policy": interpolation_policy,
                "interpolation_records": interpolation["records"],
                "local_crossing": crossing,
                "interpolation_endpoint_terminal_max_abs_error": max(
                    float(np.max(np.abs(
                        interpolation_terminal[0] - bank_terminal[pair["unsafe_index"]]
                    ))),
                    float(np.max(np.abs(
                        interpolation_terminal[-1] - bank_terminal[pair["safe_index"]]
                    ))),
                ),
            })
            if crossing is None:
                record["ineligibility_reason"] = "no_adjacent_local_unsafe_safe_crossing"
                case_results.append(record)
                continue

            anchor_residual = np.asarray(
                interpolation_values[crossing["unsafe_index"]], dtype=np.float64,
            )
            witness_distance = float(crossing["witness_distance_l2"])
            epsilon = min(
                float(config["finite_difference"]["maximum_epsilon"]),
                float(config["finite_difference"]["witness_fraction"]) * witness_distance,
            )
            if epsilon < float(config["finite_difference"]["minimum_epsilon"]):
                record["ineligibility_reason"] = "finite_difference_epsilon_below_minimum"
                case_results.append(record)
                continue
            fd_names, fd_values = finite_difference_residuals(
                anchor_residual, epsilon=epsilon,
            )
            fd_terminal, fd_policy = terminal_query(
                observation, task, seed, ordinary, fd_names, fd_values,
            )
            fd = _evaluate_bank(
                repo_root=repo_root, tight_config=tight_config,
                case_config=case_config, names=fd_names,
                terminal_actions=fd_terminal, ordinary_actions=ordinary,
                exact_source_case=exact_source_case, state_payload=state_payload,
                primary_rows=primary_rows, car_limit=car_limit,
            )
            oracle_gradient = central_gradient(
                fd["records"], key="hard_primary_future_risk", epsilon=epsilon,
            )
            learned_gradient = central_gradient(
                fd["records"], key="smooth_global_prediction", epsilon=epsilon,
            )
            comparison_names, comparison_values, direction_audit = comparison_residuals(
                anchor_residual, learned_gradient, oracle_gradient,
                correction_norm=witness_distance,
                random_seed=int(config["comparison"]["random_seed"]) + case_index,
            )
            comparison_terminal, comparison_policy = terminal_query(
                observation, task, seed, ordinary,
                comparison_names, comparison_values,
            )
            comparison = _evaluate_bank(
                repo_root=repo_root, tight_config=tight_config,
                case_config=case_config, names=comparison_names,
                terminal_actions=comparison_terminal, ordinary_actions=ordinary,
                exact_source_case=exact_source_case, state_payload=state_payload,
                primary_rows=primary_rows, car_limit=car_limit,
            )
            metrics = case_direction_metrics(comparison["records"])
            record.update({
                "eligible": True,
                "ineligibility_reason": None,
                "finite_difference_epsilon": float(epsilon),
                "finite_difference_policy": fd_policy,
                "finite_difference_records": fd["records"],
                "finite_difference_anchor_terminal_max_abs_error": float(
                    np.max(np.abs(
                        fd_terminal[0]
                        - interpolation_terminal[crossing["unsafe_index"]]
                    ))
                ),
                "oracle_gradient": oracle_gradient,
                "learned_gradient": learned_gradient,
                "direction_audit": direction_audit,
                "comparison_policy": comparison_policy,
                "comparison_records": comparison["records"],
                "comparison_anchor_terminal_max_abs_error": float(
                    np.max(np.abs(comparison_terminal[0] - fd_terminal[0]))
                ),
                "metrics": metrics,
            })
            case_results.append(record)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

    eligible = [row for row in case_results if row["eligible"]]
    boundary_eligible = [row for row in eligible if row["role"] == "boundary"]
    physical_false_safes = sum(
        int(candidate["represented_geometry_physical_false_safe"])
        for case in case_results
        for phase in (
            "bank_records", "interpolation_records",
            "finite_difference_records", "comparison_records",
        )
        for candidate in case.get(phase, [])
    )
    checks = {
        "case_count": len(case_results) == int(config["gate"]["required_case_count"]),
        "eligible_boundary_roots": len(boundary_eligible)
        >= int(config["gate"]["minimum_eligible_boundary_root_count"]),
        "oracle_descent": bool(boundary_eligible) and all(
            row["metrics"]["oracle_descends"] for row in boundary_eligible
        ),
        "oracle_beats_opposite": bool(boundary_eligible) and all(
            row["metrics"]["oracle_beats_opposite"] for row in boundary_eligible
        ),
        "oracle_random_advantage": bool(boundary_eligible) and all(
            float(row["metrics"]["oracle_random_advantage_rate"])
            >= float(config["gate"]["minimum_oracle_random_advantage_rate"])
            for row in boundary_eligible
        ),
        "oracle_safe_conversion": sum(
            int(row["metrics"]["oracle_safe_conversion"])
            for row in boundary_eligible
        ) >= int(config["gate"]["minimum_oracle_safe_conversion_count"]),
        "zero_represented_geometry_physical_false_safes": physical_false_safes == 0,
    }
    canary_pass = bool(all(checks.values()))
    result = {
        "schema_version": CASE_SCHEMA,
        "status": "complete_oracle_flow_gradient_canary",
        "scientific_result": True,
        "replica": replica,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": allocation_record(),
        "config_file_sha256": file_sha256(config_path),
        "config_payload_sha256": config["config_payload_sha256"],
        "server_identity": server,
        "frozen_model_sha256": model_record["model_sha256"],
        "case_results": case_results,
        "summary": {
            "case_count": len(case_results),
            "eligible_boundary_root_count": len(boundary_eligible),
            "physical_false_safe_count": physical_false_safes,
            "oracle_safe_conversion_count": sum(
                int(row["metrics"]["oracle_safe_conversion"])
                for row in boundary_eligible
            ),
            "learned_safe_conversion_count": sum(
                int(row["metrics"]["learned_safe_conversion"])
                for row in boundary_eligible
            ),
            "checks": checks,
            "oracle_steering_canary_pass": canary_pass,
        },
        "model_training_performed": False,
        "QP_or_full_episode_performed": False,
        "timing": {
            "total_wall_seconds": (time.perf_counter_ns() - started) * 1e-9,
        },
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--replica", choices=("producer", "replay"), required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit, replica=args.replica,
        host=args.host, port=args.port, run_root=args.run_root.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "eligible_boundary_root_count": result["summary"]["eligible_boundary_root_count"],
        "oracle_safe_conversion_count": result["summary"]["oracle_safe_conversion_count"],
        "oracle_steering_canary_pass": result["summary"]["oracle_steering_canary_pass"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
