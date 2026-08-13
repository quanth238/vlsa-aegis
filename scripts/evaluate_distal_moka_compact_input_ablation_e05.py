#!/usr/bin/env python3
"""One-state, action-only memorization test for the E05 Moka response MLP."""

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


RESULT_SCHEMA = "vlsa_distal_moka_compact_input_ablation_e05_result.v1"


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


def load_ablation_config(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    _require(
        value.get("protocol_id")
        == "vlsa-distal-moka-compact-input-ablation-e05-v1",
        "compact-input protocol differs",
    )
    _require(value.get("case_ids") == [CASE_ID], "compact-input case differs")
    _require(value.get("state", {}).get("step") == 182, "compact-input state differs")
    _require(
        value.get("input", {}).get("context_groups")
        == ["nominal_first_five_xyz"],
        "compact-input context differs",
    )
    _require(
        value.get("input", {}).get("context_dimension") == 15,
        "compact-input context dimension differs",
    )
    _require(
        value.get("input", {}).get("model_input_dimension") == 25,
        "compact-input model dimension differs",
    )
    _require(
        value.get("forbidden")
        == [
            "additional_input_group",
            "loss_change",
            "optimizer_change",
            "architecture_change",
            "larger_correction",
            "qp",
            "corrected_execution",
            "closed_loop",
        ],
        "compact-input forbidden operations differ",
    )
    output = json.loads(_canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


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
    _require(finite.size > 0, "compact-input summary has no finite values")
    return {
        "count": int(array.size),
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def _metrics(
    *,
    directions: Any,
    targets: Any,
    predicted_rows: Any,
) -> dict[str, Any]:
    import numpy as np

    design = np.asarray(directions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64).reshape(design.shape[0], -1)
    rows = np.asarray(predicted_rows, dtype=np.float64).reshape(target.shape[1], 15)
    prediction = design.dot(rows.T)
    return _direction_metrics(prediction, target)


def _gate(
    config: Mapping[str, Any],
    compact: Mapping[str, Any],
    ridge: Mapping[str, Any],
) -> dict[str, Any]:
    gate = config["gate"]
    fit_rmse_ratio = float(compact["fit"]["rmse_m_per_action"]) / max(
        float(ridge["fit"]["rmse_m_per_action"]), 1.0e-12
    )
    heldout_rmse_ratio = float(compact["heldout"]["rmse_m_per_action"]) / max(
        float(ridge["heldout"]["rmse_m_per_action"]), 1.0e-12
    )
    heldout_cosine_delta = float(compact["heldout"]["cosine"]) - float(
        ridge["heldout"]["cosine"]
    )
    checks = {
        "fit_response_cosine": bool(
            float(compact["fit"]["cosine"])
            >= float(gate["minimum_fit_response_cosine"])
        ),
        "fit_sign_accuracy": bool(
            float(compact["fit"]["sign_accuracy"])
            >= float(gate["minimum_fit_sign_accuracy"])
        ),
        "fit_rmse_relative_to_ridge": bool(
            fit_rmse_ratio
            <= float(gate["maximum_fit_rmse_ratio_to_local_ridge"])
        ),
        "row_gradient_cosine_to_ridge": bool(
            float(compact["row_gradient_cosine_to_ridge"]["mean"])
            >= float(gate["minimum_row_gradient_cosine_to_local_ridge"])
        ),
        "heldout_cosine_relative_to_ridge": bool(
            heldout_cosine_delta
            >= float(gate["minimum_heldout_cosine_relative_to_local_ridge"])
        ),
        "heldout_rmse_relative_to_ridge": bool(
            heldout_rmse_ratio
            <= float(gate["maximum_heldout_rmse_ratio_to_local_ridge"])
        ),
    }
    return {
        "checks": checks,
        "fit_rmse_ratio_to_ridge": fit_rmse_ratio,
        "heldout_cosine_delta_from_ridge": heldout_cosine_delta,
        "heldout_rmse_ratio_to_ridge": heldout_rmse_ratio,
        "pass": bool(all(checks.values())),
    }


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    geometry_config_path: Path,
    experiment_config_path: Path,
    ablation_config_path: Path,
    frozen_audit_path: Path,
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
    from main.multilink_ellipsoid.moka_response_field import (
        load_moka_response_config,
        predict_response,
        train_response_model,
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
    config = load_ablation_config(ablation_config_path)
    experiment_config = load_moka_response_config(experiment_config_path)
    _require(config["model"] == experiment_config["model"], "model settings changed")
    sampling = config["sampling"]
    _require(
        sampling["direction_seed"] == experiment_config["sampling"]["direction_seed"],
        "direction seed changed",
    )
    _require(
        sampling["fit_direction_count"]
        == experiment_config["sampling"]["fit_direction_count"],
        "fit count changed",
    )
    _require(
        sampling["paired_direction_count"]
        == experiment_config["sampling"]["paired_direction_count_per_state"],
        "paired count changed",
    )
    _require(
        sampling["paired_perturbation_action"]
        == experiment_config["sampling"]["paired_perturbation_action"],
        "paired perturbation changed",
    )
    frozen_audit = _load(frozen_audit_path)
    frozen_expected = config["frozen_audit"]
    _require(
        _file_sha256(frozen_audit_path) == frozen_expected["result_file_sha256"],
        "frozen audit file differs",
    )
    _require(
        frozen_audit.get("result_payload_sha256")
        == frozen_expected["result_payload_sha256"],
        "frozen audit payload differs",
    )
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    archived = _load(archived_path)
    _require(
        _file_sha256(archived_path) == ARCHIVED_FILE_SHA256,
        "Table-1 file hash differs",
    )
    _require(
        archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256,
        "Table-1 payload differs",
    )
    archived_actions = archived.get("actions")
    _require(
        isinstance(archived_actions, list)
        and len(archived_actions) == EXPECTED_ACTION_HORIZON,
        "Table-1 horizon differs",
    )
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "compact-input manifest case differs")
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
        _require(
            np.array_equal(
                np.asarray(selected_initial_state), np.asarray(probe_initial_state)
            ),
            "probe initial state differs",
        )
        obstacle_name, _ = _active_obstacle(env, observation)
        pairing = pairing_record(
            case=case,
            selected_initial_state=selected_initial_state,
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=np.asarray(env.sim.get_state().flatten()),
        )
        for key in (
            "manifest_row_sha256",
            "initial_state_sha256",
            "initial_observation_sha256",
            "settled_simulator_state_sha256",
        ):
            _require(pairing[key] == frozen_audit["pairing"][key], "pairing differs: %s" % key)
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
            probe_env,
            geometry,
            clearance_m=0.0,
            active_obstacle_name=obstacle_name,
        )
        probe = MokaContinuationProbe(one_step, obstacle_name)
        state_step = int(config["state"]["step"])
        for step in range(state_step + 1):
            if step == state_step:
                base_actions = np.asarray(
                    [
                        _archived_action(archived_actions, index)
                        for index in range(step, step + 20)
                    ],
                    dtype=np.float64,
                )
                boxes = compiled_obstacle_boxes(env, obstacle_name)
                _require(
                    len(boxes)
                    == int(experiment_config["geometry"]["expected_compiled_moka_box_count"]),
                    "compiled Moka box count differs",
                )
                compact_context = base_actions[:5, :3].reshape(15)
                directions = bounded_paired_directions(
                    int(sampling["paired_direction_count"]),
                    int(sampling["direction_seed"]) + step,
                    base_actions[:5, :3],
                    float(sampling["paired_perturbation_action"]),
                    float(experiment_config["action_space"]["action_limit"]),
                )
                base = probe.rollout(env, base_actions, step_base=step)
                plus = []
                minus = []
                probe_wall = float(base["env_step_wall_seconds"])
                for direction in directions:
                    positive = probe.rollout(
                        env,
                        _actions_with_correction(
                            base_actions,
                            float(sampling["paired_perturbation_action"]) * direction,
                            float(experiment_config["action_space"]["action_limit"]),
                        ),
                        step_base=step,
                    )
                    negative = probe.rollout(
                        env,
                        _actions_with_correction(
                            base_actions,
                            -float(sampling["paired_perturbation_action"]) * direction,
                            float(experiment_config["action_space"]["action_limit"]),
                        ),
                        step_base=step,
                    )
                    plus.append(positive["trace_m"])
                    minus.append(negative["trace_m"])
                    probe_wall += float(positive["env_step_wall_seconds"])
                    probe_wall += float(negative["env_step_wall_seconds"])
            action = np.asarray(_archived_action(archived_actions, step), dtype=np.float64)
            observation, _, done, _ = env.step(action.tolist())
            _require(not bool(done) or step >= state_step, "episode ended early")

        plus_array = np.asarray(plus, dtype=np.float64)
        minus_array = np.asarray(minus, dtype=np.float64)
        fit_count = int(sampling["fit_direction_count"])
        fit = _fit_rows(
            directions[:fit_count],
            plus_array[:fit_count],
            minus_array[:fit_count],
            float(sampling["paired_perturbation_action"]),
            float(sampling["ridge"]),
        )
        heldout_targets = (
            plus_array[fit_count:] - minus_array[fit_count:]
        ) / (2.0 * float(sampling["paired_perturbation_action"]))
        train_state = {
            "context": compact_context,
            "base_values_m": base["trace_m"],
            "fit_directions": directions[:fit_count],
            "fit_targets_m_per_action": fit["targets"],
        }
        model = train_response_model([train_state], [train_state], config["model"])
        predicted_values, predicted_rows = predict_response(model, compact_context)
        ridge_rows = np.asarray(fit["gradients"], dtype=np.float64).reshape(20, 7, 15)
        compact_fit = _metrics(
            directions=directions[:fit_count],
            targets=fit["targets"],
            predicted_rows=predicted_rows,
        )
        compact_heldout = _metrics(
            directions=directions[fit_count:],
            targets=heldout_targets,
            predicted_rows=predicted_rows,
        )
        ridge_fit = _metrics(
            directions=directions[:fit_count],
            targets=fit["targets"],
            predicted_rows=ridge_rows,
        )
        ridge_heldout = _metrics(
            directions=directions[fit_count:],
            targets=heldout_targets,
            predicted_rows=ridge_rows,
        )
        row_cosines = _cosine_rows(predicted_rows, ridge_rows)
        base_values = np.asarray(base["trace_m"], dtype=np.float64)
        active = base_values <= float(np.min(base_values)) + float(
            sampling["near_active_threshold_m"]
        )
        active_cosines = _cosine_rows(predicted_rows[active], ridge_rows[active])
        compact_report = {
            "fit": compact_fit,
            "heldout": compact_heldout,
            "row_gradient_cosine_to_ridge": _summary(row_cosines),
            "near_active_row_gradient_cosine_to_ridge": _summary(active_cosines),
            "value_rmse_m": float(
                np.sqrt(np.mean((predicted_values - base_values) ** 2))
            ),
            "value_bias_m": float(np.mean(predicted_values - base_values)),
        }
        ridge_report = {
            "fit": ridge_fit,
            "heldout": ridge_heldout,
            "design_rank": int(np.linalg.matrix_rank(directions[:fit_count])),
            "design_condition_number": float(
                np.linalg.cond(directions[:fit_count].T.dot(directions[:fit_count]))
            ),
        }
        gate = _gate(config, compact_report, ridge_report)
        frozen_step = next(
            item for item in frozen_audit["state_reports"] if item["step"] == state_step
        )
        if gate["pass"]:
            interpretation = "compact_input_one_state_memorization_pass"
        elif (
            compact_fit["cosine"] >= 0.9
            and compact_report["near_active_row_gradient_cosine_to_ridge"]["mean"] < 0.9
        ):
            interpretation = "global_memorization_pass_near_active_rows_fail"
        elif ridge_heldout["cosine"] < 0.8:
            interpretation = "input_only_ablation_fails_and_local_secant_teacher_is_nonlinear"
        else:
            interpretation = "input_only_ablation_fails_to_match_local_linear_teacher"
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "experiment_config_sha256": experiment_config["config_file_sha256"],
            "frozen_audit": {
                "path": str(frozen_audit_path),
                "result_file_sha256": _file_sha256(frozen_audit_path),
                "result_payload_sha256": frozen_audit["result_payload_sha256"],
                "step_182_original_metrics": frozen_step,
            },
            "pairing": pairing,
            "disabled_images": disabled_images,
            "input_ablation": {
                "context_dimension": int(compact_context.size),
                "model_input_dimension": int(len(model["feature_mean"])),
                "original_model_input_dimension": int(
                    len(frozen_audit["frozen_model"].get("model_input", []))
                    if "model_input" in frozen_audit["frozen_model"]
                    else 1055
                ),
                "only_changed_factor": "input_representation",
                "single_state_action_context_is_constant_after_normalization": True,
            },
            "state": {
                "step": state_step,
                "base_minimum_m": float(np.min(base_values)),
                "base_contact_count": len(base["protected_contacts"]),
                "near_active_witness_count": int(np.sum(active)),
            },
            "model": {
                "best_epoch": int(model["best_epoch"]),
                "best_validation_loss": float(model["best_validation_loss"]),
                "model_sha256": model["model_sha256"],
                "state_payload": model["state_payload"],
                "parameters": int(
                    sum(parameter.numel() for parameter in model["model"].parameters())
                ),
            },
            "compact_mlp": compact_report,
            "local_ridge": ridge_report,
            "gate": gate,
            "interpretation": interpretation,
            "rollout_count": 1 + 2 * int(sampling["paired_direction_count"]),
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
    parser.add_argument("--ablation-config", type=Path, required=True)
    parser.add_argument("--frozen-audit", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        ablation_config_path=args.ablation_config.resolve(),
        frozen_audit_path=args.frozen_audit.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "gate_pass": result["gate"]["pass"],
                "interpretation": result["interpretation"],
                "result_payload_sha256": result["result_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
