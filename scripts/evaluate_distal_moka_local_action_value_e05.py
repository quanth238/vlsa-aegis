#!/usr/bin/env python3
"""Matched scalar-response versus nonlinear local action-value experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
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


RESULT_SCHEMA = "vlsa_distal_moka_local_action_value_e05_result.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
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


def load_config(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    _require(
        value.get("protocol_id") == "vlsa-distal-moka-local-action-value-e05-v1",
        "local action-value protocol differs",
    )
    _require(value.get("case_ids") == [CASE_ID], "local action-value case differs")
    _require(value.get("state") == {"step": 182}, "local action-value state differs")
    sampling = value.get("sampling", {})
    _require(float(sampling.get("radius_action")) == 0.0125, "radius differs")
    _require(sampling.get("train_direction_indexes") == [0, 20], "train split differs")
    _require(sampling.get("validation_direction_indexes") == [20, 24], "validation split differs")
    _require(sampling.get("test_direction_indexes") == [24, 32], "test split differs")
    _require(
        value.get("input", {}).get("model_input_dimension") == 40,
        "local action-value input differs",
    )
    _require(
        value.get("forbidden")
        == [
            "additional_state_input",
            "direction_change",
            "radius_change",
            "geometry_change",
            "horizon_change",
            "qp",
            "corrected_execution",
            "closed_loop",
        ],
        "local action-value forbidden operations differ",
    )
    output = json.loads(_canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def _repeat_queries(witness: Any, queries: Any) -> Any:
    import numpy as np

    rows = np.asarray(witness, dtype=np.float64)
    directions = np.asarray(queries, dtype=np.float64)
    if rows.shape != (140, 25) or directions.ndim != 2 or directions.shape[1] != 15:
        raise ValueError("local action-value query arrays differ")
    tiled_rows = np.tile(rows[None, :, :], (directions.shape[0], 1, 1))
    tiled_queries = np.tile(directions[:, None, :], (1, 140, 1))
    return np.concatenate([tiled_rows, tiled_queries], axis=2).reshape(-1, 40)


def _response_report(prediction: Any, exact: Any, active: Any) -> dict[str, Any]:
    import numpy as np

    predicted = np.asarray(prediction, dtype=np.float64).reshape(-1, 140)
    target = np.asarray(exact, dtype=np.float64).reshape(predicted.shape)
    mask = np.asarray(active, dtype=bool).reshape(140)
    return {
        "all_witnesses": _direction_metrics(predicted, target),
        "near_active_witnesses": _direction_metrics(predicted[:, mask], target[:, mask]),
    }


def _branch_metrics(
    predicted_plus: Any,
    predicted_minus: Any,
    exact_plus: Any,
    exact_minus: Any,
) -> dict[str, Any]:
    import numpy as np

    pp = np.asarray(predicted_plus, dtype=np.float64).reshape(-1, 140)
    pm = np.asarray(predicted_minus, dtype=np.float64).reshape(-1, 140)
    ep = np.asarray(exact_plus, dtype=np.float64).reshape(pp.shape)
    em = np.asarray(exact_minus, dtype=np.float64).reshape(pp.shape)
    predicted_worst = np.stack([np.min(pp, axis=1), np.min(pm, axis=1)], axis=1)
    exact_worst = np.stack([np.min(ep, axis=1), np.min(em, axis=1)], axis=1)
    safer = np.argmax(predicted_worst, axis=1) == np.argmax(exact_worst, axis=1)
    return {
        "direction_count": int(pp.shape[0]),
        "safer_branch_accuracy": float(np.mean(safer)),
        "worst_margin_rmse_m": float(
            np.sqrt(np.mean((predicted_worst - exact_worst) ** 2))
        ),
        "predicted_worst_m": predicted_worst.tolist(),
        "exact_worst_m": exact_worst.tolist(),
    }


def _model_gate(config: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    gate = config["gate"]
    response = report["test_response"]
    branches = report["test_branches"]
    checks = {
        "test_response_cosine": float(response["all_witnesses"]["cosine"])
        >= float(gate["minimum_test_response_cosine"]),
        "test_response_sign": float(response["all_witnesses"]["sign_accuracy"])
        >= float(gate["minimum_test_response_sign_accuracy"]),
        "near_active_response_cosine": float(response["near_active_witnesses"]["cosine"])
        >= float(gate["minimum_near_active_response_cosine"]),
        "near_active_response_sign": float(response["near_active_witnesses"]["sign_accuracy"])
        >= float(gate["minimum_near_active_response_sign_accuracy"]),
        "safer_branch_accuracy": float(branches["safer_branch_accuracy"])
        >= float(gate["minimum_safer_branch_accuracy"]),
        "worst_margin_rmse": float(branches["worst_margin_rmse_m"])
        <= float(gate["maximum_test_worst_margin_rmse_m"]),
    }
    return {"checks": checks, "pass": bool(all(checks.values()))}


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    geometry_config_path: Path,
    experiment_config_path: Path,
    local_config_path: Path,
    reference_result_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    import numpy as np

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
    from main.multilink_ellipsoid.moka_local_action_value import (
        predict_scalar,
        train_scalar_model,
    )
    from main.multilink_ellipsoid.moka_response_field import (
        load_moka_response_config,
        witness_features,
    )
    from main.multilink_ellipsoid.multi_witness_counterfactual import (
        bounded_paired_directions,
    )
    from main.multilink_ellipsoid.obstacle_proxy_audit import compiled_obstacle_boxes
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    started = time.perf_counter_ns()
    config = load_config(local_config_path)
    experiment_config = load_moka_response_config(experiment_config_path)
    reference = _load(reference_result_path)
    _require(
        _file_sha256(reference_result_path) == config["reference"]["result_file_sha256"],
        "reference radius file differs",
    )
    _require(
        reference.get("result_payload_sha256")
        == config["reference"]["result_payload_sha256"],
        "reference radius payload differs",
    )
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    archived_actions = archived.get("actions")
    _require(isinstance(archived_actions, list) and len(archived_actions) == EXPECTED_ACTION_HORIZON, "Table-1 horizon differs")
    matches = [row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "local action-value manifest case differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    geometry_config = load_shadow_config(geometry_config_path)
    runtime = _runtime_imports(include_aegis=False)
    env = None
    probe_env = None
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        probe_env, probe_task, probe_observation, probe_initial_state = _build_environment(
            runtime, case, render_resolution=32
        )
        probe_observation = _settle(probe_env, probe_observation, TABLE_SETTLE_ACTIONS)
        _require(str(task.language) == str(probe_task.language), "probe task differs")
        _require(np.array_equal(np.asarray(selected_initial_state), np.asarray(probe_initial_state)), "probe initial state differs")
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
            _require(pairing[key] == reference["pairing"][key], "pairing differs: %s" % key)
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
        one_step = SlabbedEightConstraintProbe(
            probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        probe = MokaContinuationProbe(one_step, obstacle_name)
        sampling = config["sampling"]
        step_target = int(config["state"]["step"])
        probe_wall = 0.0
        for step in range(step_target + 1):
            if step == step_target:
                base_actions = np.asarray(
                    [_archived_action(archived_actions, index) for index in range(step, step + 20)],
                    dtype=np.float64,
                )
                boxes = compiled_obstacle_boxes(env, obstacle_name)
                _require(len(boxes) == int(experiment_config["geometry"]["expected_compiled_moka_box_count"]), "compiled Moka boxes differ")
                context = base_actions[:5, :3].reshape(15)
                directions = bounded_paired_directions(
                    int(sampling["paired_direction_count"]),
                    int(sampling["direction_seed"]) + step,
                    base_actions[:5, :3],
                    float(sampling["generation_radius_action"]),
                    float(experiment_config["action_space"]["action_limit"]),
                )
                direction_hash = hashlib.sha256(np.asarray(directions, dtype="<f8").tobytes(order="C")).hexdigest()
                _require(direction_hash == config["reference"]["directions_float64_sha256"], "direction hash differs")
                base = probe.rollout(env, base_actions, step_base=step)
                probe_wall += float(base["env_step_wall_seconds"])
                plus = []
                minus = []
                radius = float(sampling["radius_action"])
                for direction in directions:
                    positive = probe.rollout(
                        env,
                        _actions_with_correction(base_actions, radius * direction, float(experiment_config["action_space"]["action_limit"])),
                        step_base=step,
                    )
                    negative = probe.rollout(
                        env,
                        _actions_with_correction(base_actions, -radius * direction, float(experiment_config["action_space"]["action_limit"])),
                        step_base=step,
                    )
                    plus.append(positive["trace_m"])
                    minus.append(negative["trace_m"])
                    probe_wall += float(positive["env_step_wall_seconds"])
                    probe_wall += float(negative["env_step_wall_seconds"])
            action = np.asarray(_archived_action(archived_actions, step), dtype=np.float64)
            observation, _, done, _ = env.step(action.tolist())
            _require(not bool(done) or step >= step_target, "episode ended early")

        base_values = np.asarray(base["trace_m"], dtype=np.float64).reshape(140)
        plus_values = np.asarray(plus, dtype=np.float64).reshape(32, 140)
        minus_values = np.asarray(minus, dtype=np.float64).reshape(32, 140)
        exact_response = (plus_values - minus_values) / (2.0 * radius)
        active = base_values <= float(np.min(base_values)) + float(sampling["near_active_threshold_m"])
        witnesses = witness_features(context)
        _require(witnesses.shape == (140, 25), "compact witness input differs")
        train_start, train_stop = sampling["train_direction_indexes"]
        validation_start, validation_stop = sampling["validation_direction_indexes"]
        test_start, test_stop = sampling["test_direction_indexes"]

        response_train_x = _repeat_queries(witnesses, directions[train_start:train_stop])
        response_validation_x = _repeat_queries(witnesses, directions[validation_start:validation_stop])
        response_model = train_scalar_model(
            response_train_x,
            exact_response[train_start:train_stop].reshape(-1),
            response_validation_x,
            exact_response[validation_start:validation_stop].reshape(-1),
            config["model"],
        )

        def value_features(start: int, stop: int) -> Any:
            return np.concatenate(
                [
                    _repeat_queries(witnesses, radius * directions[start:stop]),
                    _repeat_queries(witnesses, -radius * directions[start:stop]),
                ],
                axis=0,
            )

        value_model = train_scalar_model(
            value_features(train_start, train_stop),
            np.concatenate([plus_values[train_start:train_stop].reshape(-1), minus_values[train_start:train_stop].reshape(-1)]),
            value_features(validation_start, validation_stop),
            np.concatenate([plus_values[validation_start:validation_stop].reshape(-1), minus_values[validation_start:validation_stop].reshape(-1)]),
            config["model"],
        )

        def response_predictions(bundle: Mapping[str, Any], start: int, stop: int) -> Any:
            return predict_scalar(bundle, _repeat_queries(witnesses, directions[start:stop])).reshape(stop - start, 140)

        def value_predictions(start: int, stop: int) -> tuple[Any, Any]:
            positive = predict_scalar(value_model, _repeat_queries(witnesses, radius * directions[start:stop])).reshape(stop - start, 140)
            negative = predict_scalar(value_model, _repeat_queries(witnesses, -radius * directions[start:stop])).reshape(stop - start, 140)
            return positive, negative

        reports = {}
        response_test = response_predictions(response_model, test_start, test_stop)
        response_plus = base_values[None, :] + radius * response_test
        response_minus = base_values[None, :] - radius * response_test
        reports["direction_conditioned_scalar"] = {
            "train_response": _response_report(response_predictions(response_model, train_start, train_stop), exact_response[train_start:train_stop], active),
            "validation_response": _response_report(response_predictions(response_model, validation_start, validation_stop), exact_response[validation_start:validation_stop], active),
            "test_response": _response_report(response_test, exact_response[test_start:test_stop], active),
            "test_branches": _branch_metrics(response_plus, response_minus, plus_values[test_start:test_stop], minus_values[test_start:test_stop]),
            "model": {
                "best_epoch": response_model["best_epoch"],
                "best_validation_loss": response_model["best_validation_loss"],
                "model_sha256": response_model["model_sha256"],
                "parameter_count": response_model["parameter_count"],
                "state_payload": response_model["state_payload"],
            },
        }
        value_plus, value_minus = value_predictions(test_start, test_stop)
        value_response = (value_plus - value_minus) / (2.0 * radius)
        train_value_plus, train_value_minus = value_predictions(train_start, train_stop)
        validation_value_plus, validation_value_minus = value_predictions(validation_start, validation_stop)
        reports["nonlinear_local_action_value"] = {
            "train_response": _response_report((train_value_plus - train_value_minus) / (2.0 * radius), exact_response[train_start:train_stop], active),
            "validation_response": _response_report((validation_value_plus - validation_value_minus) / (2.0 * radius), exact_response[validation_start:validation_stop], active),
            "test_response": _response_report(value_response, exact_response[test_start:test_stop], active),
            "test_branches": _branch_metrics(value_plus, value_minus, plus_values[test_start:test_stop], minus_values[test_start:test_stop]),
            "model": {
                "best_epoch": value_model["best_epoch"],
                "best_validation_loss": value_model["best_validation_loss"],
                "model_sha256": value_model["model_sha256"],
                "parameter_count": value_model["parameter_count"],
                "state_payload": value_model["state_payload"],
            },
        }
        for report in reports.values():
            report["gate"] = _model_gate(config, report)
        passing = [name for name, report in reports.items() if report["gate"]["pass"]]
        if passing:
            interpretation = "nonlinear_local_prediction_representation_passes"
        else:
            interpretation = "neither_scalar_nor_nonlinear_local_representation_passes"
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "experiment_config_sha256": experiment_config["config_file_sha256"],
            "reference": {
                "path": str(reference_result_path),
                "result_file_sha256": _file_sha256(reference_result_path),
                "result_payload_sha256": reference["result_payload_sha256"],
            },
            "pairing": pairing,
            "disabled_images": disabled_images,
            "direction_binding": {
                "count": int(directions.shape[0]),
                "dimension": int(directions.shape[1]),
                "directions_float64_sha256": direction_hash,
                "radius_action": radius,
            },
            "split": {
                "train_direction_indexes": list(range(train_start, train_stop)),
                "validation_direction_indexes": list(range(validation_start, validation_stop)),
                "test_direction_indexes": list(range(test_start, test_stop)),
            },
            "state": {
                "step": step_target,
                "base_minimum_m": float(np.min(base_values)),
                "base_contact_count": len(base["protected_contacts"]),
                "near_active_witness_count": int(np.sum(active)),
            },
            "model_reports": reports,
            "passing_models": passing,
            "overall_gate_pass": bool(passing),
            "interpretation": interpretation,
            "rollout_count": 65,
            "probe_env_step_wall_seconds": probe_wall,
            "correction_or_qp_attempted": False,
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
    parser.add_argument("--local-config", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        local_config_path=args.local_config.resolve(),
        reference_result_path=args.reference_result.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({"interpretation": result["interpretation"], "overall_gate_pass": result["overall_gate_pass"], "passing_models": result["passing_models"], "result_payload_sha256": result["result_payload_sha256"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
