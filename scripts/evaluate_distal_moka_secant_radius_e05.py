#!/usr/bin/env python3
"""Matched compact-input secant-radius ablation at E05 step 182."""

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
from scripts.evaluate_distal_moka_compact_input_ablation_e05 import (
    _cosine_rows,
    _metrics,
    _summary,
)
from scripts.evaluate_distal_moka_response_field_e05 import (
    MokaContinuationProbe,
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


RESULT_SCHEMA = "vlsa_distal_moka_secant_radius_e05_result.v1"


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


def load_radius_config(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    _require(
        value.get("protocol_id") == "vlsa-distal-moka-secant-radius-e05-v1",
        "secant-radius protocol differs",
    )
    _require(value.get("case_ids") == [CASE_ID], "secant-radius case differs")
    _require(value.get("state") == {"step": 182}, "secant-radius state differs")
    _require(
        value.get("sampling", {}).get("test_radii_action") == [0.025, 0.0125],
        "secant radii differ",
    )
    _require(
        value.get("direction_protocol")
        == {
            "generation_radius_action": 0.05,
            "reuse_exact_directions_across_radii": True,
        },
        "secant direction protocol differs",
    )
    _require(
        value.get("input")
        == {
            "context_dimension": 15,
            "context_groups": ["nominal_first_five_xyz"],
            "model_input_dimension": 25,
        },
        "secant compact input differs",
    )
    _require(
        value.get("forbidden")
        == [
            "input_change",
            "direction_change",
            "loss_change",
            "optimizer_change",
            "architecture_change",
            "horizon_change",
            "qp",
            "corrected_execution",
            "closed_loop",
        ],
        "secant forbidden operations differ",
    )
    output = json.loads(_canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def _radius_gate(config: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    threshold = config["gate"]
    checks = {
        "mlp_fit_cosine": bool(
            float(report["compact_mlp"]["fit"]["cosine"])
            >= float(threshold["minimum_fit_response_cosine"])
        ),
        "mlp_fit_sign": bool(
            float(report["compact_mlp"]["fit"]["sign_accuracy"])
            >= float(threshold["minimum_fit_sign_accuracy"])
        ),
        "mlp_heldout_cosine": bool(
            float(report["compact_mlp"]["heldout"]["cosine"])
            >= float(threshold["minimum_heldout_response_cosine"])
        ),
        "mlp_heldout_sign": bool(
            float(report["compact_mlp"]["heldout"]["sign_accuracy"])
            >= float(threshold["minimum_heldout_sign_accuracy"])
        ),
        "near_active_row_cosine": bool(
            float(
                report["compact_mlp"]["near_active_row_gradient_cosine_to_ridge"][
                    "mean"
                ]
            )
            >= float(threshold["minimum_near_active_row_cosine_to_ridge"])
        ),
        "ridge_heldout_cosine": bool(
            float(report["local_ridge"]["heldout"]["cosine"])
            >= float(threshold["minimum_ridge_heldout_response_cosine"])
        ),
        "ridge_heldout_sign": bool(
            float(report["local_ridge"]["heldout"]["sign_accuracy"])
            >= float(threshold["minimum_ridge_heldout_sign_accuracy"])
        ),
    }
    return {"checks": checks, "pass": bool(all(checks.values()))}


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    geometry_config_path: Path,
    experiment_config_path: Path,
    radius_config_path: Path,
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
    config = load_radius_config(radius_config_path)
    experiment_config = load_moka_response_config(experiment_config_path)
    reference = _load(reference_result_path)
    _require(
        _file_sha256(reference_result_path) == config["reference"]["result_file_sha256"],
        "reference compact result file differs",
    )
    _require(
        reference.get("result_payload_sha256")
        == config["reference"]["result_payload_sha256"],
        "reference compact result payload differs",
    )
    _require(config["model"] == experiment_config["model"], "model changed")
    sampling = config["sampling"]
    _require(
        sampling["direction_seed"] == experiment_config["sampling"]["direction_seed"],
        "direction seed changed",
    )
    _require(
        sampling["fit_direction_count"]
        == experiment_config["sampling"]["fit_direction_count"],
        "fit direction count changed",
    )
    _require(
        sampling["paired_direction_count"]
        == experiment_config["sampling"]["paired_direction_count_per_state"],
        "paired direction count changed",
    )
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    archived_actions = archived.get("actions")
    _require(
        isinstance(archived_actions, list)
        and len(archived_actions) == EXPECTED_ACTION_HORIZON,
        "Table-1 horizon differs",
    )
    matches = [
        row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID
    ]
    _require(len(matches) == 1, "secant-radius manifest case differs")
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
            np.array_equal(np.asarray(selected_initial_state), np.asarray(probe_initial_state)),
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
        step_target = int(config["state"]["step"])
        for step in range(step_target + 1):
            if step == step_target:
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
                    "compiled Moka boxes differ",
                )
                compact_context = base_actions[:5, :3].reshape(15)
                directions = bounded_paired_directions(
                    int(sampling["paired_direction_count"]),
                    int(sampling["direction_seed"]) + step,
                    base_actions[:5, :3],
                    float(config["direction_protocol"]["generation_radius_action"]),
                    float(experiment_config["action_space"]["action_limit"]),
                )
                directions_hash = hashlib.sha256(
                    np.asarray(directions, dtype="<f8").tobytes(order="C")
                ).hexdigest()
                base = probe.rollout(env, base_actions, step_base=step)
                probe_wall = float(base["env_step_wall_seconds"])
                radius_rollouts = {}
                for radius in sampling["test_radii_action"]:
                    plus = []
                    minus = []
                    for direction in directions:
                        positive = probe.rollout(
                            env,
                            _actions_with_correction(
                                base_actions,
                                float(radius) * direction,
                                float(experiment_config["action_space"]["action_limit"]),
                            ),
                            step_base=step,
                        )
                        negative = probe.rollout(
                            env,
                            _actions_with_correction(
                                base_actions,
                                -float(radius) * direction,
                                float(experiment_config["action_space"]["action_limit"]),
                            ),
                            step_base=step,
                        )
                        plus.append(positive["trace_m"])
                        minus.append(negative["trace_m"])
                        probe_wall += float(positive["env_step_wall_seconds"])
                        probe_wall += float(negative["env_step_wall_seconds"])
                    radius_rollouts[str(radius)] = {
                        "plus": np.asarray(plus, dtype=np.float64),
                        "minus": np.asarray(minus, dtype=np.float64),
                    }
            action = np.asarray(_archived_action(archived_actions, step), dtype=np.float64)
            observation, _, done, _ = env.step(action.tolist())
            _require(not bool(done) or step >= step_target, "episode ended early")

        base_values = np.asarray(base["trace_m"], dtype=np.float64)
        active = base_values <= float(np.min(base_values)) + float(
            sampling["near_active_threshold_m"]
        )
        fit_count = int(sampling["fit_direction_count"])
        reports = {}
        internal_rows = {}
        for radius in sampling["test_radii_action"]:
            rollouts = radius_rollouts[str(radius)]
            fit = _fit_rows(
                directions[:fit_count],
                rollouts["plus"][:fit_count],
                rollouts["minus"][:fit_count],
                float(radius),
                float(sampling["ridge"]),
            )
            heldout = (rollouts["plus"][fit_count:] - rollouts["minus"][fit_count:]) / (
                2.0 * float(radius)
            )
            train_state = {
                "context": compact_context,
                "base_values_m": base_values,
                "fit_directions": directions[:fit_count],
                "fit_targets_m_per_action": fit["targets"],
            }
            model = train_response_model([train_state], [train_state], config["model"])
            predicted_values, predicted_rows = predict_response(model, compact_context)
            ridge_rows = np.asarray(fit["gradients"], dtype=np.float64).reshape(20, 7, 15)
            compact = {
                "fit": _metrics(
                    directions=directions[:fit_count],
                    targets=fit["targets"],
                    predicted_rows=predicted_rows,
                ),
                "heldout": _metrics(
                    directions=directions[fit_count:],
                    targets=heldout,
                    predicted_rows=predicted_rows,
                ),
                "row_gradient_cosine_to_ridge": _summary(
                    _cosine_rows(predicted_rows, ridge_rows)
                ),
                "near_active_row_gradient_cosine_to_ridge": _summary(
                    _cosine_rows(predicted_rows[active], ridge_rows[active])
                ),
                "value_rmse_m": float(
                    np.sqrt(np.mean((predicted_values - base_values) ** 2))
                ),
                "value_bias_m": float(np.mean(predicted_values - base_values)),
            }
            ridge = {
                "fit": _metrics(
                    directions=directions[:fit_count],
                    targets=fit["targets"],
                    predicted_rows=ridge_rows,
                ),
                "heldout": _metrics(
                    directions=directions[fit_count:],
                    targets=heldout,
                    predicted_rows=ridge_rows,
                ),
                "design_rank": int(np.linalg.matrix_rank(directions[:fit_count])),
                "design_condition_number": float(
                    np.linalg.cond(directions[:fit_count].T.dot(directions[:fit_count]))
                ),
            }
            report = {
                "radius_action": float(radius),
                "compact_mlp": compact,
                "local_ridge": ridge,
                "model": {
                    "best_epoch": int(model["best_epoch"]),
                    "best_validation_loss": float(model["best_validation_loss"]),
                    "model_sha256": model["model_sha256"],
                    "state_payload": model["state_payload"],
                    "parameters": int(
                        sum(parameter.numel() for parameter in model["model"].parameters())
                    ),
                },
            }
            report["gate"] = _radius_gate(config, report)
            reports[str(radius)] = report
            internal_rows[str(radius)] = {
                "ridge": ridge_rows,
                "mlp": predicted_rows,
            }

        radius_names = [str(value) for value in sampling["test_radii_action"]]
        first_rows = internal_rows[radius_names[0]]
        second_rows = internal_rows[radius_names[1]]
        cross_radius = {
            "ridge_row_cosine": _summary(
                _cosine_rows(first_rows["ridge"], second_rows["ridge"])
            ),
            "mlp_row_cosine": _summary(
                _cosine_rows(first_rows["mlp"], second_rows["mlp"])
            ),
            "near_active_ridge_row_cosine": _summary(
                _cosine_rows(first_rows["ridge"][active], second_rows["ridge"][active])
            ),
            "near_active_mlp_row_cosine": _summary(
                _cosine_rows(first_rows["mlp"][active], second_rows["mlp"][active])
            ),
        }
        passing = [name for name, report in reports.items() if report["gate"]["pass"]]
        if passing:
            interpretation = "smaller_secant_radius_passes_local_response_gate"
        elif all(
            float(report["local_ridge"]["heldout"]["cosine"])
            < float(config["gate"]["minimum_ridge_heldout_response_cosine"])
            for report in reports.values()
        ):
            interpretation = "smaller_secants_do_not_rescue_affine_response_teacher"
        else:
            interpretation = "teacher_improves_but_compact_mlp_still_fails"
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
                "radius_action": reference["config"]["sampling"]["paired_perturbation_action"],
                "compact_mlp": reference["compact_mlp"],
                "local_ridge": reference["local_ridge"],
            },
            "pairing": pairing,
            "disabled_images": disabled_images,
            "direction_binding": {
                "count": int(directions.shape[0]),
                "dimension": int(directions.shape[1]),
                "directions_float64_sha256": directions_hash,
                "generation_radius_action": float(
                    config["direction_protocol"]["generation_radius_action"]
                ),
                "identical_for_all_test_radii": True,
            },
            "state": {
                "step": step_target,
                "base_minimum_m": float(np.min(base_values)),
                "base_contact_count": len(base["protected_contacts"]),
                "near_active_witness_count": int(np.sum(active)),
            },
            "radius_reports": reports,
            "cross_radius": cross_radius,
            "passing_radii": passing,
            "overall_gate_pass": bool(passing),
            "interpretation": interpretation,
            "rollout_count": 1
            + 2
            * int(sampling["paired_direction_count"])
            * len(sampling["test_radii_action"]),
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
    parser.add_argument("--radius-config", type=Path, required=True)
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
        radius_config_path=args.radius_config.resolve(),
        reference_result_path=args.reference_result.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "interpretation": result["interpretation"],
                "overall_gate_pass": result["overall_gate_pass"],
                "passing_radii": result["passing_radii"],
                "result_payload_sha256": result["result_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
