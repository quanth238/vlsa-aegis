#!/usr/bin/env python3
"""Frozen 64-direction audit of the E05 Moka paired-ranking hypothesis."""

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
from scripts.evaluate_distal_moka_response_field_e05 import MokaContinuationProbe
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


RESULT_SCHEMA = "vlsa_distal_moka_frozen_ranker_audit_e05_result.v1"


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
        value.get("protocol_id") == "vlsa-distal-moka-frozen-ranker-audit-e05-v1",
        "frozen ranker protocol differs",
    )
    _require(value.get("case_ids") == [CASE_ID], "frozen ranker case differs")
    _require(value.get("state") == {"step": 182}, "frozen ranker state differs")
    sampling = value.get("sampling", {})
    _require(float(sampling.get("radius_action")) == 0.0125, "ranker radius differs")
    _require(int(sampling.get("paired_direction_count")) == 64, "ranker direction count differs")
    _require(sampling.get("best_of_n_prefixes") == [4, 8, 16, 32, 64], "Best-of-N prefixes differ")
    _require(int(sampling.get("random_sign_trials")) == 10000, "random trials differ")
    _require(
        value.get("forbidden")
        == [
            "model_training",
            "model_parameter_change",
            "input_change",
            "loss_change",
            "state_change",
            "radius_change",
            "geometry_change",
            "horizon_change",
            "qp",
            "corrected_execution",
            "closed_loop",
        ],
        "frozen ranker forbidden operations differ",
    )
    output = json.loads(_canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def _repeat_queries(witness: Any, queries: Any) -> Any:
    import numpy as np

    rows = np.asarray(witness, dtype=np.float64)
    directions = np.asarray(queries, dtype=np.float64)
    _require(rows.shape == (140, 25), "ranker witness features differ")
    _require(directions.ndim == 2 and directions.shape[1] == 15, "ranker queries differ")
    row_values = np.tile(rows[None, :, :], (directions.shape[0], 1, 1))
    query_values = np.tile(directions[:, None, :], (1, 140, 1))
    return np.concatenate([row_values, query_values], axis=2).reshape(-1, 40)


def _one_sided_binomial_p(successes: int, trials: int) -> float:
    return float(
        sum(math.comb(int(trials), value) for value in range(int(successes), int(trials) + 1))
        / float(2 ** int(trials))
    )


def _gain_summary(values: Any) -> dict[str, Any]:
    import numpy as np

    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(raw.size > 0 and np.all(np.isfinite(raw)), "ranker gain summary differs")
    return {
        "count": int(raw.size),
        "mean_m": float(np.mean(raw)),
        "median_m": float(np.median(raw)),
        "minimum_m": float(np.min(raw)),
        "maximum_m": float(np.max(raw)),
        "positive_rate": float(np.mean(raw > 0.0)),
    }


def _best_of_n(predicted: Any, exact: Any, prefixes: Sequence[int]) -> dict[str, Any]:
    import numpy as np

    predicted_values = np.asarray(predicted, dtype=np.float64).reshape(-1)
    exact_values = np.asarray(exact, dtype=np.float64).reshape(-1)
    _require(predicted_values.shape == exact_values.shape, "Best-of-N arrays differ")
    output = {}
    for count in prefixes:
        size = int(count)
        chosen = int(np.argmax(predicted_values[:size]))
        order = np.argsort(-exact_values[:size], kind="stable")
        rank = int(np.flatnonzero(order == chosen)[0]) + 1
        best = int(order[0])
        percentile = 1.0 if size == 1 else 1.0 - float(rank - 1) / float(size - 1)
        output[str(size)] = {
            "predicted_choice_index": chosen,
            "exact_best_index": best,
            "top1_match": bool(chosen == best),
            "exact_rank": rank,
            "exact_percentile": percentile,
            "selected_exact_margin_m": float(exact_values[chosen]),
            "best_exact_margin_m": float(exact_values[best]),
            "regret_m": float(exact_values[best] - exact_values[chosen]),
        }
    return output


def _gate(config: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = config["gate"]
    checks = {
        "branch_accuracy": float(metrics["branch_accuracy"])
        >= float(thresholds["minimum_branch_accuracy"]),
        "binomial_significance": float(metrics["one_sided_binomial_p"])
        < float(thresholds["maximum_one_sided_binomial_p"]),
        "positive_near_active_l5_gain": float(
            metrics["selected_near_active_l5_gain"]["mean_m"]
        )
        > float(thresholds["minimum_mean_selected_near_active_l5_gain_m"]),
    }
    return {"checks": checks, "pass": bool(all(checks.values()))}


def audit(
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
    from main.multilink_ellipsoid.moka_local_action_value import (
        load_frozen_scalar_model,
        predict_scalar,
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
    config = load_config(audit_config_path)
    experiment_config = load_moka_response_config(experiment_config_path)
    frozen_result = _load(frozen_result_path)
    _require(
        _file_sha256(frozen_result_path) == config["frozen_model"]["result_file_sha256"],
        "frozen result file differs",
    )
    _require(
        frozen_result.get("result_payload_sha256")
        == config["frozen_model"]["result_payload_sha256"],
        "frozen result payload differs",
    )
    frozen_record = frozen_result["model_reports"][config["frozen_model"]["arm"]]["model"]
    _require(frozen_record["model_sha256"] == config["frozen_model"]["model_sha256"], "frozen model hash differs")
    model = load_frozen_scalar_model(torch, frozen_record["state_payload"])
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    archived_actions = archived.get("actions")
    _require(isinstance(archived_actions, list) and len(archived_actions) == EXPECTED_ACTION_HORIZON, "Table-1 horizon differs")
    matches = [row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "frozen ranker manifest case differs")
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
            _require(pairing[key] == frozen_result["pairing"][key], "pairing differs: %s" % key)
        disabled_images = {"main": _disable_images(env), "probe": _disable_images(probe_env)}
        perception = archived["perception"]
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(
            geometry_config,
            {"p2": perception["mvee_center"], "R2": perception["mvee_rotation"], "Q2_diag": perception["mvee_semiaxes"], "record": {"label": perception["obstacle_label"]}},
        )
        one_step = SlabbedEightConstraintProbe(probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name)
        probe = MokaContinuationProbe(one_step, obstacle_name)
        sampling = config["sampling"]
        step_target = int(config["state"]["step"])
        probe_wall = 0.0
        for step in range(step_target + 1):
            if step == step_target:
                base_actions = np.asarray([_archived_action(archived_actions, index) for index in range(step, step + 20)], dtype=np.float64)
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
                development_directions = bounded_paired_directions(
                    32,
                    2026081310 + step,
                    base_actions[:5, :3],
                    float(sampling["generation_radius_action"]),
                    float(experiment_config["action_space"]["action_limit"]),
                )
                _require(
                    not any(
                        np.array_equal(candidate, prior)
                        or np.array_equal(candidate, -prior)
                        for candidate in directions
                        for prior in development_directions
                    ),
                    "new ranker directions overlap development directions",
                )
                base = probe.rollout(env, base_actions, step_base=step)
                probe_wall += float(base["env_step_wall_seconds"])
                plus = []
                minus = []
                radius = float(sampling["radius_action"])
                for direction in directions:
                    positive = probe.rollout(env, _actions_with_correction(base_actions, radius * direction, float(experiment_config["action_space"]["action_limit"])), step_base=step)
                    negative = probe.rollout(env, _actions_with_correction(base_actions, -radius * direction, float(experiment_config["action_space"]["action_limit"])), step_base=step)
                    plus.append(positive["trace_m"])
                    minus.append(negative["trace_m"])
                    probe_wall += float(positive["env_step_wall_seconds"])
                    probe_wall += float(negative["env_step_wall_seconds"])
            action = np.asarray(_archived_action(archived_actions, step), dtype=np.float64)
            observation, _, done, _ = env.step(action.tolist())
            _require(not bool(done) or step >= step_target, "episode ended early")

        base_values = np.asarray(base["trace_m"], dtype=np.float64).reshape(140)
        plus_values = np.asarray(plus, dtype=np.float64).reshape(64, 140)
        minus_values = np.asarray(minus, dtype=np.float64).reshape(64, 140)
        active = base_values <= float(np.min(base_values)) + float(sampling["near_active_threshold_m"])
        active_l5 = active.copy()
        for index in range(140):
            if index % 7 >= 3:
                active_l5[index] = False
        _require(int(np.sum(active_l5)) == 6, "nominal near-active L5 witness count differs")
        features = witness_features(context)
        prediction = predict_scalar(model, _repeat_queries(features, directions)).reshape(64, 140)
        predicted_plus = base_values[None, :] + radius * prediction
        predicted_minus = base_values[None, :] - radius * prediction
        predicted_worst = np.stack([np.min(predicted_plus, axis=1), np.min(predicted_minus, axis=1)], axis=1)
        exact_worst = np.stack([np.min(plus_values, axis=1), np.min(minus_values, axis=1)], axis=1)
        predicted_sign = np.argmax(predicted_worst, axis=1)
        exact_sign = np.argmax(exact_worst, axis=1)
        correct = predicted_sign == exact_sign
        selected_exact = exact_worst[np.arange(64), predicted_sign]
        selected_predicted = predicted_worst[np.arange(64), predicted_sign]
        exact_selected_values = np.where(predicted_sign[:, None] == 1, minus_values, plus_values)
        selected_gain = selected_exact - float(np.min(base_values))
        selected_active_gain = np.min(exact_selected_values[:, active_l5], axis=1) - float(np.min(base_values[active_l5]))

        predicted_active = np.stack([np.min(predicted_plus[:, active_l5], axis=1), np.min(predicted_minus[:, active_l5], axis=1)], axis=1)
        exact_active = np.stack([np.min(plus_values[:, active_l5], axis=1), np.min(minus_values[:, active_l5], axis=1)], axis=1)
        active_accuracy = float(np.mean(np.argmax(predicted_active, axis=1) == np.argmax(exact_active, axis=1)))
        successes = int(np.sum(correct))
        rng = np.random.default_rng(int(sampling["random_sign_seed"]))
        random_signs = rng.integers(0, 2, size=(int(sampling["random_sign_trials"]), 64))
        random_selected = np.take_along_axis(exact_worst[None, :, :], random_signs[:, :, None], axis=2).squeeze(2)
        random_active_selected = np.take_along_axis(exact_active[None, :, :], random_signs[:, :, None], axis=2).squeeze(2)
        random_summary = {
            "expected_branch_accuracy": 0.5,
            "trial_count": int(sampling["random_sign_trials"]),
            "mean_selected_clearance_gain_m": float(np.mean(random_selected - float(np.min(base_values)))),
            "mean_selected_near_active_l5_gain_m": float(np.mean(random_active_selected - float(np.min(base_values[active_l5])))),
            "model_mean_gain_percentile": float(np.mean(np.mean(random_selected - float(np.min(base_values)), axis=1) <= np.mean(selected_gain))),
        }
        metrics = {
            "successes": successes,
            "direction_count": 64,
            "branch_accuracy": float(np.mean(correct)),
            "wrong_direction_rate": float(np.mean(~correct)),
            "one_sided_binomial_p": _one_sided_binomial_p(successes, 64),
            "near_active_l5_branch_accuracy": active_accuracy,
            "selected_clearance_gain": _gain_summary(selected_gain),
            "selected_near_active_l5_gain": _gain_summary(selected_active_gain),
            "random_sign_baseline": random_summary,
        }
        metrics["gate"] = _gate(config, metrics)
        best_of_n = _best_of_n(selected_predicted, selected_exact, sampling["best_of_n_prefixes"])
        all_exact = np.concatenate([exact_worst[:, 0], exact_worst[:, 1]])
        selected_index = int(np.argmax(selected_predicted))
        global_best_index = int(np.argmax(all_exact))
        best_of_n["64"]["best_of_128_exact_margin_m"] = float(all_exact[global_best_index])
        best_of_n["64"]["regret_to_best_of_128_m"] = float(all_exact[global_best_index] - selected_exact[selected_index])
        best_of_n["64"]["top1_match_best_of_128"] = bool(
            (global_best_index == selected_index and predicted_sign[selected_index] == 0)
            or (global_best_index == 64 + selected_index and predicted_sign[selected_index] == 1)
        )
        if metrics["gate"]["pass"]:
            interpretation = "frozen_paired_ranking_signal_passes_one_state_gate"
        else:
            interpretation = "frozen_paired_ranking_signal_fails_large_direction_gate"
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "experiment_config_sha256": experiment_config["config_file_sha256"],
            "frozen_model": {
                "result_path": str(frozen_result_path),
                "result_file_sha256": _file_sha256(frozen_result_path),
                "result_payload_sha256": frozen_result["result_payload_sha256"],
                "model_sha256": frozen_record["model_sha256"],
            },
            "pairing": pairing,
            "disabled_images": disabled_images,
            "direction_binding": {
                "count": 64,
                "dimension": 15,
                "direction_seed": int(sampling["direction_seed"]),
                "effective_direction_seed": int(sampling["direction_seed"]) + step_target,
                "directions_float64_sha256": hashlib.sha256(np.asarray(directions, dtype="<f8").tobytes(order="C")).hexdigest(),
                "radius_action": radius,
                "disjoint_from_32_development_directions_up_to_sign": True,
            },
            "state": {
                "step": step_target,
                "base_minimum_m": float(np.min(base_values)),
                "base_contact_count": len(base["protected_contacts"]),
                "near_active_l5_witness_count": int(np.sum(active_l5)),
            },
            "metrics": metrics,
            "best_of_n": best_of_n,
            "overall_gate_pass": metrics["gate"]["pass"],
            "interpretation": interpretation,
            "rollout_count": 129,
            "probe_env_step_wall_seconds": probe_wall,
            "training_or_control_attempted": False,
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
    result = audit(
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
    print(json.dumps({"interpretation": result["interpretation"], "overall_gate_pass": result["overall_gate_pass"], "branch_accuracy": result["metrics"]["branch_accuracy"], "one_sided_binomial_p": result["metrics"]["one_sided_binomial_p"], "result_payload_sha256": result["result_payload_sha256"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
