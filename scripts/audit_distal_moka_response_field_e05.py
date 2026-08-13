#!/usr/bin/env python3
"""Replay and diagnose the frozen E05 Moka response model without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_counterfactual_field_e05 import (
    _actions_with_correction,
    _archived_action,
    _disable_images,
)
from scripts.evaluate_distal_moka_response_field_e05 import (
    MokaContinuationProbe,
    _direction_metrics,
    _fit_rows,
    _soft_direction,
    _state_context_groups,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    EXPECTED_ACTION_HORIZON,
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


RESULT_SCHEMA = "vlsa_distal_moka_response_field_e05_audit_result.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_audit_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("protocol_id") != "vlsa-distal-moka-response-field-e05-audit-v1":
        raise ValueError("Moka response audit protocol differs")
    if value.get("case_ids") != [CASE_ID]:
        raise ValueError("Moka response audit case differs")
    if value.get("forbidden") != [
        "model_training",
        "model_parameter_change",
        "larger_correction",
        "qp",
        "corrected_execution",
        "closed_loop",
    ]:
        raise ValueError("Moka response audit forbidden operations differ")
    if value.get("replay") != {
        "paired_direction_count_per_state": 32,
        "paired_perturbation_action": 0.05,
        "state_steps": [178, 179, 180, 181, 182, 183, 184, 185, 186],
    }:
        raise ValueError("Moka response audit replay differs")
    value["config_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return value


def _cosine_rows(prediction: Any, target: Any) -> Any:
    import numpy as np

    predicted = np.asarray(prediction, dtype=np.float64).reshape(-1, 15)
    exact = np.asarray(target, dtype=np.float64).reshape(-1, 15)
    numerator = np.sum(predicted * exact, axis=1)
    denominator = np.linalg.norm(predicted, axis=1) * np.linalg.norm(exact, axis=1)
    output = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = denominator > 1.0e-20
    output[valid] = numerator[valid] / denominator[valid]
    return output


def _summary(values: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": int(array.size), "finite_count": 0}
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "p95": float(np.quantile(finite, 0.95)),
    }


def _aggregate_split(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import numpy as np

    values_predicted = np.concatenate(
        [np.asarray(state["predicted_values_m"]).reshape(-1) for state in states]
    )
    values_exact = np.concatenate(
        [np.asarray(state["base_values_m"]).reshape(-1) for state in states]
    )
    fit_prediction = np.concatenate(
        [np.asarray(state["fit_prediction"]).reshape(-1) for state in states]
    )
    fit_target = np.concatenate(
        [np.asarray(state["fit_targets_m_per_action"]).reshape(-1) for state in states]
    )
    held_prediction = np.concatenate(
        [np.asarray(state["heldout_prediction"]).reshape(-1) for state in states]
    )
    exact_held_prediction = np.concatenate(
        [np.asarray(state["exact_heldout_prediction"]).reshape(-1) for state in states]
    )
    held_target = np.concatenate(
        [np.asarray(state["heldout_targets_m_per_action"]).reshape(-1) for state in states]
    )
    row_cosines = np.concatenate(
        [_cosine_rows(state["predicted_rows"], state["fit_gradients_m_per_action"]) for state in states]
    )
    value_error = values_predicted - values_exact
    return {
        "state_count": len(states),
        "value_rmse_m": float(np.sqrt(np.mean(value_error ** 2))),
        "value_bias_m": float(np.mean(value_error)),
        "value_maximum_optimism_m": float(np.max(value_error)),
        "value_false_safe_count": int(np.sum((values_predicted >= 0.0) & (values_exact < 0.0))),
        "fit_directional": _direction_metrics(fit_prediction, fit_target),
        "heldout_directional": _direction_metrics(held_prediction, held_target),
        "local_ridge_heldout_directional": _direction_metrics(
            exact_held_prediction, held_target
        ),
        "row_gradient_cosine": _summary(row_cosines),
        "smooth_direction_cosine": _summary(
            [state["smooth_direction_cosine"] for state in states]
        ),
    }


def _state_support(states: Sequence[Mapping[str, Any]], train_steps: Sequence[int]) -> dict[str, Any]:
    import numpy as np

    train = [state for state in states if int(state["step"]) in set(train_steps)]
    group_names = list(train[0]["context_groups"])
    output = []
    for state in states:
        groups = {}
        for name in group_names:
            train_matrix = np.asarray(
                [item["context_groups"][name] for item in train], dtype=np.float64
            )
            value = np.asarray(state["context_groups"][name], dtype=np.float64)
            mean = np.mean(train_matrix, axis=0)
            scale = np.std(train_matrix, axis=0)
            model_scale = np.where(scale >= 1.0e-6, scale, 1.0)
            z = (value - mean) / model_scale
            nearest = min(
                float(np.sqrt(np.mean(((value - row) / model_scale) ** 2)))
                for row in train_matrix
            )
            lower = np.min(train_matrix, axis=0)
            upper = np.max(train_matrix, axis=0)
            tolerance = 1.0e-9 + 1.0e-6 * np.maximum(1.0, np.maximum(np.abs(lower), np.abs(upper)))
            outside = (value < lower - tolerance) | (value > upper + tolerance)
            groups[name] = {
                "dimension": int(value.size),
                "model_z_rms": float(np.sqrt(np.mean(z ** 2))),
                "model_z_maximum_absolute": float(np.max(np.abs(z))),
                "model_z_count_above_3": int(np.sum(np.abs(z) > 3.0)),
                "nearest_train_model_z_rms": nearest,
                "outside_train_range_count": int(np.sum(outside)),
                "outside_train_range_fraction": float(np.mean(outside)),
                "constant_train_dimension_count": int(np.sum(scale < 1.0e-6)),
            }
        output.append({"step": int(state["step"]), "split": state["split"], "groups": groups})
    return {"training_steps": list(train_steps), "states": output}


def _near_active_report(
    states: Sequence[Mapping[str, Any]],
    link_names: Sequence[str],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    import numpy as np

    rows = []
    for state in states:
        base = np.asarray(state["base_values_m"], dtype=np.float64)
        predicted_values = np.asarray(state["predicted_values_m"], dtype=np.float64)
        exact_rows = np.asarray(state["fit_gradients_m_per_action"], dtype=np.float64)
        predicted_rows = np.asarray(state["predicted_rows"], dtype=np.float64)
        row_cosine = _cosine_rows(predicted_rows, exact_rows).reshape(20, 7)
        worst = float(np.min(base))
        for offset in range(20):
            for row in range(7):
                value = float(base[offset, row])
                rows.append(
                    {
                        "state_step": int(state["step"]),
                        "split": state["split"],
                        "action_offset": offset,
                        "absolute_action": int(state["step"]) + offset,
                        "robot_row": row,
                        "link_name": str(link_names[row]),
                        "margin_m": value,
                        "relative_to_worst_m": value - worst,
                        "primitive_index": int(state["primitive_indexes"][offset, row]),
                        "primitive_name": str(state["primitive_names"][offset][row]),
                        "primitive_second_gap_m": float(state["primitive_second_gap_m"][offset, row]),
                        "paired_primitive_switch_fraction": float(
                            state["paired_primitive_switch_fraction"][offset, row]
                        ),
                        "predicted_value_m": float(predicted_values[offset, row]),
                        "value_error_m": float(predicted_values[offset, row] - value),
                        "exact_gradient_norm_m_per_action": float(np.linalg.norm(exact_rows[offset, row])),
                        "predicted_gradient_norm_m_per_action": float(np.linalg.norm(predicted_rows[offset, row])),
                        "gradient_cosine": float(row_cosine[offset, row]) if math.isfinite(float(row_cosine[offset, row])) else None,
                    }
                )
    reports = {}
    for threshold in thresholds:
        selected = [row for row in rows if row["relative_to_worst_m"] <= float(threshold)]
        reports[str(threshold)] = {
            "count": len(selected),
            "by_split": {
                split: sum(row["split"] == split for row in selected)
                for split in ("train", "validation", "test")
            },
            "by_link": {
                link: sum(row["link_name"] == link for row in selected)
                for link in sorted(set(link_names))
            },
            "gradient_cosine": _summary(
                [row["gradient_cosine"] for row in selected if row["gradient_cosine"] is not None]
            ),
            "value_error_m": _summary([row["value_error_m"] for row in selected]),
            "primitive_second_gap_m": _summary(
                [row["primitive_second_gap_m"] for row in selected]
            ),
            "paired_primitive_switch_fraction": _summary(
                [row["paired_primitive_switch_fraction"] for row in selected]
            ),
            "rows": selected,
        }
    boundary = [row for row in rows if abs(row["margin_m"]) <= 0.005]
    reports["physical_boundary_abs_0.005"] = {
        "count": len(boundary),
        "rows": boundary,
    }
    return reports


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    geometry_config_path: Path,
    experiment_config_path: Path,
    audit_config_path: Path,
    frozen_result_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        _active_obstacle,
        _build_environment,
        _runtime_imports,
        _settle,
        pairing_record,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.moka_response_field import (
        load_frozen_response_model,
        load_moka_response_config,
        predict_response,
    )
    from main.multilink_ellipsoid.multi_witness_counterfactual import bounded_paired_directions
    from main.multilink_ellipsoid.obstacle_proxy_audit import compiled_obstacle_boxes
    from main.multilink_ellipsoid.shadow import MultilinkEllipsoidShadow, allocation_record, load_shadow_config
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    started = time.perf_counter_ns()
    audit_config = _load_audit_config(audit_config_path)
    experiment_config = load_moka_response_config(experiment_config_path)
    frozen = _load(frozen_result_path)
    expected = audit_config["frozen_producer"]
    _require(_file_sha256(frozen_result_path) == expected["result_file_sha256"], "frozen result file differs")
    _require(frozen.get("result_payload_sha256") == expected["result_payload_sha256"], "frozen result payload differs")
    _require(frozen.get("source", {}).get("commit") == expected["commit"], "frozen producer commit differs")
    bundle = load_frozen_response_model(torch, frozen["model"]["state_payload"])
    model_payload_sha256 = hashlib.sha256(_canonical(frozen["model"]["state_payload"])).hexdigest()
    _require(model_payload_sha256 == frozen["model"]["model_sha256"], "frozen model hash differs")
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    archived_actions = archived.get("actions")
    _require(isinstance(archived_actions, list) and len(archived_actions) == EXPECTED_ACTION_HORIZON, "Table-1 horizon differs")
    matches = [row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "Moka manifest case differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    geometry_config = load_shadow_config(geometry_config_path)
    runtime = _runtime_imports(include_aegis=False)
    env = None
    probe_env = None
    try:
        env, task, observation, selected_initial_state = _build_environment(runtime, case, render_resolution=TABLE_RENDER_RESOLUTION)
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        probe_env, probe_task, probe_observation, probe_initial_state = _build_environment(runtime, case, render_resolution=32)
        probe_observation = _settle(probe_env, probe_observation, TABLE_SETTLE_ACTIONS)
        _require(str(task.language) == str(probe_task.language), "Moka probe task differs")
        _require(np.array_equal(np.asarray(selected_initial_state), np.asarray(probe_initial_state)), "Moka initial state differs")
        obstacle_name, _ = _active_obstacle(env, observation)
        pairing = pairing_record(
            case=case,
            selected_initial_state=selected_initial_state,
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=np.asarray(env.sim.get_state().flatten()),
        )
        for key in ("manifest_row_sha256", "initial_state_sha256", "initial_observation_sha256", "settled_simulator_state_sha256"):
            _require(pairing[key] == frozen["pairing"][key], "Moka audit pairing differs: %s" % key)
        disabled_images = {"main": _disable_images(env), "probe": _disable_images(probe_env)}
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
        one_step = SlabbedEightConstraintProbe(probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name)
        probe = MokaContinuationProbe(one_step, obstacle_name)
        directions_count = int(audit_config["replay"]["paired_direction_count_per_state"])
        epsilon = float(audit_config["replay"]["paired_perturbation_action"])
        fit_count = int(experiment_config["sampling"]["fit_direction_count"])
        state_steps = audit_config["replay"]["state_steps"]
        split_config = experiment_config["state_split"]
        records = []
        total_rollouts = 0
        total_probe_wall = 0.0
        for step in range(max(state_steps) + 1):
            if step in state_steps:
                base_actions = np.asarray(
                    [_archived_action(archived_actions, index) for index in range(step, step + 20)],
                    dtype=np.float64,
                )
                boxes = compiled_obstacle_boxes(env, obstacle_name)
                groups = _state_context_groups(env, base_actions, boxes)
                context = np.concatenate(list(groups.values()))
                directions = bounded_paired_directions(
                    directions_count,
                    int(experiment_config["sampling"]["direction_seed"]) + step,
                    base_actions[:5, :3],
                    epsilon,
                    float(experiment_config["action_space"]["action_limit"]),
                )
                base = probe.rollout(env, base_actions, step_base=step, witness_details=True)
                total_rollouts += 1
                total_probe_wall += float(base["env_step_wall_seconds"])
                plus = []
                minus = []
                plus_primitive_indexes = []
                minus_primitive_indexes = []
                for direction in directions:
                    positive = probe.rollout(
                        env,
                        _actions_with_correction(base_actions, epsilon * direction, float(experiment_config["action_space"]["action_limit"])),
                        step_base=step,
                        witness_details=True,
                    )
                    negative = probe.rollout(
                        env,
                        _actions_with_correction(base_actions, -epsilon * direction, float(experiment_config["action_space"]["action_limit"])),
                        step_base=step,
                        witness_details=True,
                    )
                    plus.append(positive["trace_m"])
                    minus.append(negative["trace_m"])
                    plus_primitive_indexes.append(positive["primitive_indexes"])
                    minus_primitive_indexes.append(negative["primitive_indexes"])
                    total_rollouts += 2
                    total_probe_wall += float(positive["env_step_wall_seconds"]) + float(negative["env_step_wall_seconds"])
                plus = np.asarray(plus)
                minus = np.asarray(minus)
                fit = _fit_rows(directions[:fit_count], plus[:fit_count], minus[:fit_count], epsilon, float(experiment_config["sampling"]["ridge"]))
                heldout_targets = ((plus[fit_count:] - minus[fit_count:]) / (2.0 * epsilon)).reshape(directions_count - fit_count, 140)
                predicted_values, predicted_rows = predict_response(bundle, context)
                fit_prediction = directions[:fit_count].dot(predicted_rows.reshape(140, 15).T)
                heldout_prediction = directions[fit_count:].dot(predicted_rows.reshape(140, 15).T)
                exact_heldout_prediction = directions[fit_count:].dot(
                    fit["gradients"].reshape(140, 15).T
                )
                primitive_switch_fraction = np.mean(
                    np.asarray(plus_primitive_indexes, dtype=np.int64)
                    != np.asarray(minus_primitive_indexes, dtype=np.int64),
                    axis=0,
                )
                exact_direction = _soft_direction(base["trace_m"], fit["gradients"].reshape(20, 7, 15), float(experiment_config["score"]["softmin_temperature_m"]))
                predicted_direction = _soft_direction(predicted_values, predicted_rows, float(experiment_config["score"]["softmin_temperature_m"]))
                if step in split_config["train_steps"]:
                    split_name = "train"
                elif step in split_config["validation_steps"]:
                    split_name = "validation"
                else:
                    split_name = "test"
                records.append(
                    {
                        "step": step,
                        "split": split_name,
                        "context_groups": groups,
                        "base_values_m": base["trace_m"],
                        "predicted_values_m": predicted_values,
                        "fit_targets_m_per_action": fit["targets"],
                        "fit_gradients_m_per_action": fit["gradients"].reshape(20, 7, 15),
                        "fit_prediction": fit_prediction,
                        "heldout_targets_m_per_action": heldout_targets,
                        "heldout_prediction": heldout_prediction,
                        "exact_heldout_prediction": exact_heldout_prediction,
                        "predicted_rows": predicted_rows,
                        "smooth_direction_cosine": float(np.dot(exact_direction, predicted_direction)),
                        "primitive_indexes": base["primitive_indexes"],
                        "primitive_names": base["primitive_names"],
                        "primitive_second_gap_m": base["primitive_second_gap_m"],
                        "paired_primitive_switch_fraction": primitive_switch_fraction,
                        "base_contact_count": len(base["protected_contacts"]),
                        "base_exact_overlap_count": int(sum(base["exact_box_overlap"])),
                    }
                )
            action = np.asarray(_archived_action(archived_actions, step), dtype=np.float64)
            observation, _, done, _ = env.step(action.tolist())
            _require(not bool(done) or step >= max(state_steps), "Moka audit episode ended early")

        public_states = []
        for state in records:
            public_states.append(
                {
                    "step": state["step"],
                    "split": state["split"],
                    "value_rmse_m": float(np.sqrt(np.mean((state["predicted_values_m"] - state["base_values_m"]) ** 2))),
                    "value_bias_m": float(np.mean(state["predicted_values_m"] - state["base_values_m"])),
                    "fit_directional": _direction_metrics(state["fit_prediction"], state["fit_targets_m_per_action"]),
                    "heldout_directional": _direction_metrics(state["heldout_prediction"], state["heldout_targets_m_per_action"]),
                    "local_ridge_heldout_directional": _direction_metrics(
                        state["exact_heldout_prediction"],
                        state["heldout_targets_m_per_action"],
                    ),
                    "row_gradient_cosine": _summary(_cosine_rows(state["predicted_rows"], state["fit_gradients_m_per_action"])),
                    "smooth_direction_cosine": state["smooth_direction_cosine"],
                    "base_minimum_m": float(np.min(state["base_values_m"])),
                    "base_contact_count": state["base_contact_count"],
                    "base_exact_overlap_count": state["base_exact_overlap_count"],
                }
            )
        split_reports = {
            split: _aggregate_split([state for state in records if state["split"] == split])
            for split in ("train", "validation", "test")
        }
        support = _state_support(records, split_config["train_steps"])
        link_names = [item["body_name"] for item in frozen["geometry"]["distal_ellipsoids"]]
        near_active = _near_active_report(records, link_names, audit_config["audit"]["active_thresholds_m"])
        train_pass = bool(
            split_reports["train"]["heldout_directional"]["cosine"] >= 0.8
            and split_reports["train"]["heldout_directional"]["sign_accuracy"] >= 0.75
        )
        validation_pass = bool(
            split_reports["validation"]["heldout_directional"]["cosine"] >= 0.8
            and split_reports["validation"]["heldout_directional"]["sign_accuracy"] >= 0.75
        )
        test_pass = bool(
            split_reports["test"]["heldout_directional"]["cosine"] >= 0.8
            and split_reports["test"]["heldout_directional"]["sign_accuracy"] >= 0.75
        )
        test_support = [item for item in support["states"] if item["split"] == "test"]
        unsupported_groups = sorted(
            {
                name
                for item in test_support
                for name, group in item["groups"].items()
                if group["model_z_maximum_absolute"] > float(audit_config["audit"]["state_support_z_threshold"])
            }
        )
        if not train_pass:
            diagnosis = "training_response_underfit"
        elif not validation_pass and unsupported_groups:
            diagnosis = "state_coverage_and_generalization_failure"
        elif not validation_pass:
            diagnosis = "validation_generalization_or_representation_failure"
        elif not test_pass and unsupported_groups:
            diagnosis = "heldout_state_coverage_failure"
        elif not test_pass:
            diagnosis = "heldout_response_generalization_failure"
        else:
            diagnosis = "response_fit_passed_requires_separate_control_gate"
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "claim_scope": audit_config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "audit_config": audit_config,
            "experiment_config": experiment_config,
            "pairing": pairing,
            "disabled_images": disabled_images,
            "frozen_model": {
                "producer_result_path": str(frozen_result_path),
                "producer_result_file_sha256": _file_sha256(frozen_result_path),
                "producer_result_payload_sha256": frozen["result_payload_sha256"],
                "model_sha256": frozen["model"]["model_sha256"],
                "reconstructed_model_payload_sha256": model_payload_sha256,
                "parameters_modified": False,
                "training_called": False,
            },
            "state_reports": public_states,
            "split_reports": split_reports,
            "state_support": support,
            "near_active_witnesses": near_active,
            "diagnosis": {
                "classification": diagnosis,
                "train_response_gate_pass": train_pass,
                "validation_response_gate_pass": validation_pass,
                "test_response_gate_pass": test_pass,
                "unsupported_test_input_groups": unsupported_groups,
                "correction_or_qp_attempted": False,
            },
            "rollout_count": total_rollouts,
            "probe_env_step_wall_seconds": total_probe_wall,
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = _sha256(_canonical(result))
        return result
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--audit-config", type=Path, required=True)
    parser.add_argument("--frozen-result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        audit_config_path=args.audit_config.resolve(),
        frozen_result_path=args.frozen_result.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({"diagnosis": result["diagnosis"], "result_payload_sha256": result["result_payload_sha256"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
