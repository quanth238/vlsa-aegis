#!/usr/bin/env python3
"""Train and exactly audit a three-link contact ranker on the E05 suffix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.contact_ranker import (
    CONTACT_RANKER_SCHEMA,
    PROTECTED_LINKS,
    calibrate_zero_false_safe_threshold,
    link_contact_labels,
    load_contact_ranker_config,
    select_minimal_predicted_safe,
)
from scripts.collect_distal_boundary_generalization_moka10 import (
    _restore_env,
    _snapshot_env,
)
from scripts.evaluate_distal_execution_margin_nn_e05 import (
    _build_pair,
    _source_action,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    CASE_ID,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


DATASET_SCHEMA = "vlsa_distal_contact_ranker_e05_dataset.v1"
RESULT_SCHEMA = "vlsa_distal_contact_ranker_e05_result.v1"


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _successful_query_action(item: Mapping[str, Any], step: int) -> Any:
    import numpy as np

    _require(int(item.get("step", -1)) == step, "successful action step differs")
    direct = np.asarray(item.get("nominal_released_aegis_action"), dtype=np.float64)
    nested = np.asarray(
        item.get("filter", {}).get("nominal_released_aegis_action"),
        dtype=np.float64,
    )
    _require(
        direct.shape == (7,) and np.all(np.isfinite(direct))
        and np.array_equal(direct, nested),
        "released AEGIS query action differs",
    )
    return direct


def _split_for_step(config: Mapping[str, Any], step: int) -> str:
    groups = config["state_groups"]
    matches = [
        name.replace("_steps", "")
        for name in ("train_steps", "validation_steps", "test_steps")
        if step in groups[name]
    ]
    _require(len(matches) == 1, "state does not belong to exactly one split")
    return matches[0]


def _matched_source_next_hash(item: Mapping[str, Any], executed_xyz: Any) -> str:
    import numpy as np

    matches = []
    for attempt in item["filter"]["verification"]["attempts"]:
        candidate = np.asarray(attempt.get("candidate_xyz"), dtype=np.float64)
        if candidate.shape == (3,) and np.array_equal(candidate, executed_xyz):
            matches.append(str(attempt["next_state_sha256"]))
    unique = sorted(set(matches))
    _require(len(unique) == 1, "successful action lacks one recorded next-state hash")
    return unique[0]


def _build_model(torch: Any, input_count: int, widths: Sequence[int]) -> Any:
    layers = []
    previous = input_count
    for width in widths:
        layers.extend((torch.nn.Linear(previous, int(width)), torch.nn.SiLU()))
        previous = int(width)
    layers.append(torch.nn.Linear(previous, 3))
    return torch.nn.Sequential(*layers)


def _risk_metrics(records: Sequence[Mapping[str, Any]], risks: Any, threshold: float) -> dict[str, Any]:
    import numpy as np

    unsafe = np.asarray([not bool(item["D_sim_raw_safe"]) for item in records], dtype=bool)
    values = np.asarray(risks, dtype=np.float64)
    predicted_safe = values < float(threshold)
    false_safe = np.logical_and(predicted_safe, unsafe)
    false_unsafe = np.logical_and(~predicted_safe, ~unsafe)
    safe_risks = values[~unsafe]
    unsafe_risks = values[unsafe]
    auc = None
    if safe_risks.size and unsafe_risks.size:
        comparisons = [
            1.0 if bad > good else 0.5 if bad == good else 0.0
            for bad in unsafe_risks for good in safe_risks
        ]
        auc = float(np.mean(comparisons))
    return {
        "candidate_count": len(records),
        "raw_safe_count": int(np.count_nonzero(~unsafe)),
        "raw_unsafe_count": int(np.count_nonzero(unsafe)),
        "predicted_safe_count": int(np.count_nonzero(predicted_safe)),
        "false_safe_count": int(np.count_nonzero(false_safe)),
        "false_unsafe_count": int(np.count_nonzero(false_unsafe)),
        "any_contact_risk_auc": auc,
        "minimum_risk": float(np.min(values)),
        "maximum_risk": float(np.max(values)),
    }


def _train(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any], model_path: Path) -> tuple[Any, dict[str, Any], Any]:
    import numpy as np
    import torch

    features = np.asarray([item["feature_vector"] for item in records], dtype=np.float64)
    labels = np.asarray([item["link_contact_labels"] for item in records], dtype=np.float64)
    splits = np.asarray([item["split"] for item in records])
    train_mask = splits == "train"
    validation_mask = splits == "validation"
    _require(features.shape[1] == 33 and labels.shape == (len(records), 3), "training arrays differ")
    settings = config["training"]
    mean = np.mean(features[train_mask], axis=0)
    deviation = np.maximum(
        np.std(features[train_mask], axis=0),
        float(settings["minimum_standard_deviation"]),
    )
    normalized = (features - mean) / deviation
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    # The pinned simulator environment predates sm90.  This network is tiny,
    # so train deterministically on CPU inside the H100 allocation rather than
    # change the simulator's Python environment.
    device = torch.device("cpu")
    model = _build_model(torch, 33, config["network"]["hidden_widths"]).to(
        device=device, dtype=torch.float64
    )
    x = torch.as_tensor(normalized, dtype=torch.float64, device=device)
    y = torch.as_tensor(labels, dtype=torch.float64, device=device)
    train = torch.as_tensor(train_mask, dtype=torch.bool, device=device)
    validation = torch.as_tensor(validation_mask, dtype=torch.bool, device=device)
    positive = np.sum(labels[train_mask], axis=0)
    negative = int(np.count_nonzero(train_mask)) - positive
    positive_weight = np.where(positive > 0, negative / np.maximum(positive, 1.0), 1.0)
    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(positive_weight, dtype=torch.float64, device=device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    best = None
    best_epoch = None
    best_validation = math.inf
    patience = 0
    started = time.perf_counter_ns()
    for epoch in range(int(settings["epochs"])):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(x[train]), y[train]); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(x[validation]), y[validation]).cpu())
        if validation_loss < best_validation - 1.0e-12:
            best_validation = validation_loss
            best_epoch = epoch
            best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= int(settings["early_stopping_patience"]):
            break
    _require(best is not None, "contact-ranker training produced no model")
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x)).detach().cpu().numpy()
    artifact = {
        "schema_version": "vlsa_distal_contact_ranker_e05_model.v1",
        "model_state_dict": best,
        "feature_mean": mean,
        "feature_standard_deviation": deviation,
        "hidden_widths": list(config["network"]["hidden_widths"]),
        "protected_links": list(PROTECTED_LINKS),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.parent / (".%s.%d.tmp" % (model_path.name, os.getpid()))
    torch.save(artifact, temporary)
    os.replace(temporary, model_path)
    audit = {
        "seed": seed,
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch + 1),
        "best_validation_bce": float(best_validation),
        "train_positive_count_by_link": positive.astype(int).tolist(),
        "train_negative_count_by_link": negative.astype(int).tolist(),
        "positive_weight_by_link": positive_weight.tolist(),
        "wall_seconds": (time.perf_counter_ns() - started) * 1e-9,
        "training_device": "cpu_inside_H100_allocation",
        "torch_version": str(torch.__version__),
        "model_path": str(model_path),
        "model_file_sha256": _file_sha256(model_path),
    }
    return model, {"mean": mean, "deviation": deviation, "audit": audit}, probabilities


def _predict_risk_and_gradient(model: Any, feature: Sequence[float], mean: Any, deviation: Any) -> tuple[float, Any]:
    import numpy as np
    import torch

    device = next(model.parameters()).device
    raw = torch.as_tensor(feature, dtype=torch.float64, device=device).requires_grad_(True)
    normalized = (raw - torch.as_tensor(mean, dtype=torch.float64, device=device)) / torch.as_tensor(
        deviation, dtype=torch.float64, device=device
    )
    risk = torch.max(torch.sigmoid(model(normalized[None, :]))[0])
    gradient = torch.autograd.grad(risk, raw)[0][-3:]
    return float(risk.detach().cpu()), gradient.detach().cpu().numpy().astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--successful-sitl", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); started = time.perf_counter_ns()

    import numpy as np
    import torch
    from main.evaluate_safelibero_aegis import (
        _runtime_imports, pairing_record, read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.execution_margin_nn import feature_vector
    from main.multilink_ellipsoid.oracle_affine import SubstepEightConstraintProbe, oracle_candidate_xyz
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    config = load_contact_ranker_config(args.config.resolve())
    identities = config["immutable_sources"]
    _require(_file_sha256(args.archived.resolve()) == identities["archived_table1_file_sha256"], "Table-1 source differs")
    _require(_file_sha256(args.successful_sitl.resolve()) == identities["successful_sitl_file_sha256"], "successful SITL source differs")
    successful = _load(args.successful_sitl.resolve()); archived = _load(args.archived.resolve())
    _require(successful.get("result_payload_sha256") == identities["successful_sitl_payload_sha256"], "successful SITL payload differs")
    _require(successful.get("primary_problem_solved") is True and successful.get("case_id") == CASE_ID, "successful SITL contract differs")
    _require(_file_sha256(args.geometry_config.resolve()) == identities["geometry_config_file_sha256"], "geometry config differs")
    geometry_config = load_shadow_config(args.geometry_config.resolve())
    rows = [row for row in read_jsonl(args.manifest.resolve()) if row.get("case_id") == CASE_ID]
    _require(len(rows) == 1, "E05 manifest row is not unique"); case = rows[0]; validate_case_row(case, args.repo_root.resolve())
    source = _git_identity(args.repo_root.resolve(), args.expected_commit)
    allocation = allocation_record(); runtime = _runtime_imports(include_aegis=False)
    env = probe_env = None
    try:
        env, probe_env, task, observation, setup = _build_pair(runtime, case)
        pairing = pairing_record(
            case=case, selected_initial_state=setup["selected_initial_state"],
            settled_observation=observation, task_description=str(task.language),
            active_obstacle_name=setup["obstacle_name"],
            settled_simulator_state=np.asarray(env.sim.get_state().flatten(), dtype=np.float64),
        )
        for key in (
            "manifest_row_sha256", "initial_state_sha256", "initial_observation_sha256",
            "settled_simulator_state_sha256", "settled_active_obstacle_position_sha256",
            "policy_noise_schedule_sha256",
        ):
            _require(pairing[key] == successful["pairing"][key], "pairing differs: %s" % key)
        from main.multilink_ellipsoid.shadow import MultilinkEllipsoidShadow
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
        _require(
            geometry.geometry_record(env)["distal_ellipsoid_count"] == 7,
            "distal geometry count differs",
        )
        probe = SubstepEightConstraintProbe(
            probe_env, geometry, active_obstacle_name=setup["obstacle_name"],
            quadratic_tolerance=1e-6, contact_distance_threshold_m=0.0,
            obstacle_primitive_union=None,
        )
        actions = successful["actions"]
        first = config["state_groups"]["collect_steps"][0]
        for step in range(first):
            env.step(_source_action(actions[step], step).tolist())
        records = []; states = []; snapshots = {}
        candidate_config = {"candidate_set": config["candidate_set"]}
        for step in config["state_groups"]["collect_steps"]:
            snapshots[step] = _snapshot_env(env)
            query = _successful_query_action(actions[step], step)
            executed = _source_action(actions[step], step)
            candidates = oracle_candidate_xyz(query[:3], candidate_config)
            if not any(np.array_equal(item["xyz"], executed[:3]) for item in candidates):
                candidates.append({"source": "successful_oracle_action", "xyz": executed[:3].copy()})
            else:
                for item in candidates:
                    if np.array_equal(item["xyz"], executed[:3]):
                        item["source"] = "successful_oracle_action"
            state_records = []
            for candidate_index, candidate in enumerate(candidates):
                action = query.copy(); action[:3] = candidate["xyz"]
                transition = probe.transition(env, action)
                events = transition["raw_protected_contact_events"]
                labels = link_contact_labels(events)
                raw_safe = bool(not events)
                record = {
                    "record_index": len(records), "state_step": int(step),
                    "split": _split_for_step(config, step), "candidate_index": int(candidate_index),
                    "candidate_source": str(candidate["source"]),
                    "nominal_xyz": query[:3].tolist(), "candidate_xyz": np.asarray(candidate["xyz"]).tolist(),
                    "feature_vector": feature_vector(transition["substeps"][0], query[:3], candidate["xyz"]).tolist(),
                    "link_contact_labels": labels, "D_sim_raw_safe": raw_safe,
                    "D_opt_minimum_proxy_clearance_m": float(np.min(transition["minimum_substep_clearance_m"][:7])),
                    "raw_protected_contact_count": int(transition["raw_protected_contact_count"]),
                    "maximum_within_step_obstacle_l1_displacement_m": float(transition["maximum_within_step_obstacle_l1_displacement_m"]),
                    "next_state_sha256": str(transition["next_state_sha256"]),
                    "next_eef_position_m": list(transition["next_eef_position_m"]),
                    "env_step_wall_seconds": float(transition["env_step_wall_seconds"]),
                }
                records.append(record); state_records.append(record)
            oracle_rows = [item for item in state_records if item["candidate_source"] == "successful_oracle_action"]
            _require(len(oracle_rows) == 1, "state lacks one task-valid oracle candidate")
            env.step(executed.tolist())
            actual_hash = _sha256(np.asarray(_dynamic_state_vector(env), dtype=np.float64).tobytes())
            expected_hash = _matched_source_next_hash(actions[step], executed[:3])
            _require(actual_hash == expected_hash == oracle_rows[0]["next_state_sha256"], "successful state transition differs")
            states.append({
                "state_step": int(step), "split": _split_for_step(config, step),
                "candidate_count": len(state_records),
                "raw_safe_count": sum(item["D_sim_raw_safe"] for item in state_records),
                "raw_unsafe_count": sum(not item["D_sim_raw_safe"] for item in state_records),
                "successful_oracle_raw_safe": oracle_rows[0]["D_sim_raw_safe"],
                "successful_oracle_candidate_xyz": oracle_rows[0]["candidate_xyz"],
                "successful_oracle_next_state_sha256": oracle_rows[0]["next_state_sha256"],
            })
        split_counts = {
            name: {
                "candidate_count": sum(item["split"] == name for item in records),
                "raw_safe_count": sum(item["split"] == name and item["D_sim_raw_safe"] for item in records),
                "raw_unsafe_count": sum(item["split"] == name and not item["D_sim_raw_safe"] for item in records),
            }
            for name in ("train", "validation", "test")
        }
        data_gate = bool(
            all(value["raw_safe_count"] > 0 and value["raw_unsafe_count"] > 0 for value in split_counts.values())
            and all(item["raw_safe_count"] > 0 and item["raw_unsafe_count"] > 0 for item in states if item["split"] == "test")
        )
        dataset = {
            "schema_version": DATASET_SCHEMA, "status": "complete", "scientific_result": True,
            "claim_scope": config["claim_scope"], "source": source, "allocation": allocation,
            "config": config, "pairing": pairing, "states": states, "records": records,
            "split_counts": split_counts,
            "label_semantics": "exact_nonpositive_MuJoCo_contact_during_every_internal_OSC_substep",
            "D_opt_semantics": "seven_L5_L6_L7_ellipsoids_vs_released_AEGIS_obstacle_MVEE_diagnostic_only",
            "D_sim_semantics": "any_nonpositive_raw_L5_L6_L7_contact_over_every_internal_OSC_substep",
            "dataset_gate_pass": data_gate,
            "total_clone_env_step_wall_seconds": float(sum(item["env_step_wall_seconds"] for item in records)),
        }
        dataset["dataset_payload_sha256"] = _hash_without(dataset, "dataset_payload_sha256")
        _atomic_write(args.dataset.resolve(), dataset)
        if not data_gate:
            result = {
                "schema_version": RESULT_SCHEMA, "status": "complete", "scientific_result": True,
                "source": source, "allocation": allocation, "config": config,
                "dataset_file_sha256": _file_sha256(args.dataset.resolve()),
                "dataset_gate_pass": False, "training_performed": False,
                "decision": {"ranking_feasibility_go": False, "gradient_steering_go": False,
                             "stop_reason": "exact_contact_dataset_gate_failed"},
            }
        else:
            model, model_state, probabilities = _train(records, config, args.model.resolve())
            risks = np.max(probabilities, axis=1)
            validation_indexes = [i for i, item in enumerate(records) if item["split"] == "validation"]
            threshold = calibrate_zero_false_safe_threshold(
                [risks[i] for i in validation_indexes],
                [not records[i]["D_sim_raw_safe"] for i in validation_indexes],
            )
            metrics = {
                name: _risk_metrics(
                    [item for item in records if item["split"] == name],
                    [risks[i] for i, item in enumerate(records) if item["split"] == name],
                    threshold,
                )
                for name in ("train", "validation", "test")
            }
            test_selection = []
            gradient_audits = []
            inference_times = []
            for step in config["state_groups"]["test_steps"]:
                indexed = [(i, item) for i, item in enumerate(records) if item["state_step"] == step]
                local_records = [item for _, item in indexed]
                local_risks = [risks[i] for i, _ in indexed]
                selected_local = select_minimal_predicted_safe(local_records, local_risks, threshold)
                selected = None if selected_local is None else local_records[selected_local]
                fresh = None
                _restore_env(env, snapshots[step])
                query = _successful_query_action(actions[step], step)
                if selected is not None:
                    action = query.copy(); action[:3] = selected["candidate_xyz"]
                    transition = probe.transition(env, action)
                    fresh = {
                        "D_sim_raw_safe": bool(transition["raw_protected_contact_count"] == 0),
                        "raw_protected_contact_count": int(transition["raw_protected_contact_count"]),
                        "maximum_within_step_obstacle_l1_displacement_m": float(transition["maximum_within_step_obstacle_l1_displacement_m"]),
                        "next_state_sha256": transition["next_state_sha256"],
                        "matches_collected_transition": transition["next_state_sha256"] == selected["next_state_sha256"],
                    }
                test_selection.append({
                    "state_step": step, "selected": selected is not None,
                    "candidate_source": None if selected is None else selected["candidate_source"],
                    "candidate_xyz": None if selected is None else selected["candidate_xyz"],
                    "risk": None if selected is None else float(local_risks[selected_local]),
                    "correction_l2": None if selected is None else float(np.linalg.norm(np.asarray(selected["candidate_xyz"]) - query[:3])),
                    "fresh_exact_verification": fresh,
                })
                nominal_record = next(item for item in local_records if item["candidate_source"] == "nominal")
                t0 = time.perf_counter_ns()
                nominal_risk, gradient = _predict_risk_and_gradient(
                    model, nominal_record["feature_vector"], model_state["mean"], model_state["deviation"]
                )
                inference_times.append((time.perf_counter_ns() - t0) * 1e-6)
                norm = float(np.linalg.norm(gradient)); attempts = []
                for amount in config["gradient_audit"]["step_sizes_action"]:
                    candidate_xyz = query[:3].copy()
                    if norm > float(config["gradient_audit"]["minimum_gradient_norm"]):
                        candidate_xyz = np.clip(candidate_xyz - float(amount) * gradient / norm, -1.0, 1.0)
                    action = query.copy(); action[:3] = candidate_xyz
                    transition = probe.transition(env, action)
                    attempts.append({
                        "step_size_action": float(amount), "candidate_xyz": candidate_xyz.tolist(),
                        "D_sim_raw_safe": bool(transition["raw_protected_contact_count"] == 0),
                        "raw_protected_contact_count": int(transition["raw_protected_contact_count"]),
                        "maximum_within_step_obstacle_l1_displacement_m": float(transition["maximum_within_step_obstacle_l1_displacement_m"]),
                    })
                gradient_audits.append({
                    "state_step": step, "nominal_risk": nominal_risk,
                    "risk_gradient": gradient.tolist(), "gradient_norm": norm,
                    "attempts": attempts, "at_least_one_exact_safe": any(item["D_sim_raw_safe"] for item in attempts),
                })
            ranking_go = bool(
                metrics["test"]["false_safe_count"] == 0
                and metrics["test"]["predicted_safe_count"] > 0
                and all(item["fresh_exact_verification"] is not None
                        and item["fresh_exact_verification"]["D_sim_raw_safe"]
                        and item["fresh_exact_verification"]["matches_collected_transition"]
                        for item in test_selection)
            )
            gradient_go = bool(all(item["at_least_one_exact_safe"] for item in gradient_audits))
            result = {
                "schema_version": RESULT_SCHEMA, "status": "complete", "scientific_result": True,
                "claim_scope": config["claim_scope"], "source": source, "allocation": allocation,
                "config": config, "dataset_file_sha256": _file_sha256(args.dataset.resolve()),
                "dataset_payload_sha256": dataset["dataset_payload_sha256"],
                "dataset_gate_pass": True, "training_performed": True,
                "training": model_state["audit"], "calibrated_safe_risk_threshold": threshold,
                "metrics": metrics, "test_selection": test_selection,
                "gradient_audit": gradient_audits,
                "inference_time_ms": {"mean": float(np.mean(inference_times)), "maximum": float(np.max(inference_times))},
                "decision": {
                    "ranking_feasibility_go": ranking_go,
                    "gradient_steering_go": gradient_go,
                    "closed_loop_task_success_demonstrated": False,
                    "stop_reason": None if ranking_go else "held_out_exact_contact_ranking_gate_failed",
                },
                "wall_seconds": (time.perf_counter_ns() - started) * 1e-9,
            }
        result["result_payload_sha256"] = _hash_without(result, "result_payload_sha256")
        _atomic_write(args.output.resolve(), result)
        print(json.dumps(result["decision"], sort_keys=True))
    finally:
        if probe_env is not None: probe_env.close()
        if env is not None: env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
