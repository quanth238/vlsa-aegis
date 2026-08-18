#!/usr/bin/env python3
"""Audit compact-critic action gradients against immutable exact risk secants."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_tight_prefix_risk_q_diagnostic import load_samples


def _reconstruct_model(
    trained: Mapping[str, Any], training_config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.whole_body_q_only_diagnostic import build_model

    record = trained["compact_shared_7D"]["model"]
    payload = record["state_payload"]
    model = build_model(
        torch, int(payload["input_dimension"]), int(payload["output_dimension"]),
        training_config["model"]["hidden_widths"],
    )
    state = {
        name: torch.as_tensor(value, dtype=torch.float32)
        for name, value in payload["state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    model = model.double().cpu().eval()
    return {
        "model": model,
        "feature_mean": np.asarray(payload["feature_mean"], dtype=np.float64),
        "feature_scale": np.asarray(payload["feature_scale"], dtype=np.float64),
        "target_mean": np.asarray(payload["target_mean"], dtype=np.float64),
        "target_scale": np.asarray(payload["target_scale"], dtype=np.float64),
        "model_sha256": str(record["model_sha256"]),
        "state_payload_sha256": hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _value_and_gradient(
    bundle: Mapping[str, Any], feature: Sequence[float], *, epsilon: float,
) -> tuple[float, list[float], float]:
    import numpy as np
    import torch

    model = bundle["model"]
    mean = torch.as_tensor(bundle["feature_mean"], dtype=torch.float64)
    scale = torch.as_tensor(bundle["feature_scale"], dtype=torch.float64)
    target_mean = torch.as_tensor(bundle["target_mean"], dtype=torch.float64)
    target_scale = torch.as_tensor(bundle["target_scale"], dtype=torch.float64)
    raw = torch.as_tensor(feature, dtype=torch.float64).clone().requires_grad_(True)
    normalized = (raw - mean) / scale
    value = (model(normalized) * target_scale + target_mean).reshape(())
    gradient = torch.autograd.grad(value, raw)[0]

    z = ((raw.detach() - mean) / scale).detach()

    def evaluate(item: Any) -> float:
        return float((model(item) * target_scale + target_mean).reshape(()).item())

    analytic_z = gradient.detach() * scale
    maximum_error = 0.0
    for index in range(2, 7):
        offset = torch.zeros_like(z)
        offset[index] = float(epsilon)
        finite = (evaluate(z + offset) - evaluate(z - offset)) / (2.0 * epsilon)
        maximum_error = max(
            maximum_error, abs(float(analytic_z[index].item()) - float(finite)),
        )
    output = gradient.detach().cpu().numpy().astype(np.float64)
    return float(value.item()), output.tolist(), float(maximum_error)


def _exact_cases(
    *, repo_root: Path, training_config: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    from main.multilink_ellipsoid.tight_prefix_risk_dataset import (
        load_config as load_dataset_config,
    )

    dataset_path = repo_root / training_config["source"]["dataset_config"]
    dataset = load_dataset_config(dataset_path, repo_root=repo_root)
    producer = Path(training_config["source"]["producer_dir"])
    output = {}
    for item in dataset["cases"]:
        case_id = str(item["case_id"])
        record = _load(producer / (case_id + ".json"))
        _require(record.get("case_id") == case_id, "critic gradient case differs")
        output[case_id] = record["case"]
    _require(len(output) == 24, "critic gradient case count differs")
    return output


def _candidate_xyz(exact: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    output = {}
    for candidate in exact["candidates"]:
        actions = np.asarray(candidate["source_executed_actions"], dtype=np.float64)
        _require(actions.shape == (5, 7), "critic gradient action shape differs")
        output[str(candidate["name"])] = actions[:, :3].reshape(-1)
    _require(len(output) == 13 and "nominal" in output, "critic gradient bank differs")
    return output


def run(
    *, repo_root: Path, config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    import numpy as np
    from main.multilink_ellipsoid.pi05_palm_l6_source_selection import (
        cpu_allocation_record,
    )
    from main.multilink_ellipsoid.tight_prefix_critic_gradient_audit import (
        RESULT_SCHEMA, aggregate_pair_metrics, canonical, feasibility_verdict,
        file_sha256, load_config, pair_direction_record, payload_sha256,
        projection_geometry, smoothmax_weights, unsafe_recall,
    )
    from main.multilink_ellipsoid.tight_prefix_risk_q_diagnostic import (
        RESULT_SCHEMA as TRAINING_RESULT_SCHEMA,
        VALIDATION_SCHEMA as TRAINING_VALIDATION_SCHEMA,
        load_config as load_training_config,
        payload_sha256 as training_payload,
    )

    config = load_config(config_path)
    source = config["source"]
    training_config_path = repo_root / source["training_config"]
    _require(
        file_sha256(training_config_path) == source["training_config_file_sha256"],
        "critic gradient training config file differs",
    )
    training_config = load_training_config(training_config_path)
    _require(
        training_config["config_payload_sha256"]
        == source["training_config_payload_sha256"],
        "critic gradient training config payload differs",
    )
    result_path = Path(source["training_result"])
    validation_path = Path(source["training_validation"])
    _require(
        _file_sha256(result_path) == source["training_result_file_sha256"]
        and _file_sha256(validation_path)
        == source["training_validation_file_sha256"],
        "critic gradient source file differs",
    )
    trained = _load(result_path)
    validated = _load(validation_path)
    _require(
        trained.get("schema_version") == TRAINING_RESULT_SCHEMA
        and trained.get("result_payload_sha256")
        == source["training_result_payload_sha256"]
        == training_payload(trained, "result_payload_sha256")
        and validated.get("schema_version") == TRAINING_VALIDATION_SCHEMA
        and validated.get("validation_payload_sha256")
        == source["training_validation_payload_sha256"]
        == training_payload(validated, "validation_payload_sha256")
        and validated.get("independent_training_exact") is True
        and validated.get("model_sha256") == source["model_sha256"]
        == trained["compact_shared_7D"]["model"]["model_sha256"],
        "critic gradient validated model differs",
    )
    samples, sample_source = load_samples(
        repo_root=repo_root, config=training_config,
    )
    exact_cases = _exact_cases(
        repo_root=repo_root, training_config=training_config,
    )
    actions = {
        state_id: _candidate_xyz(exact)
        for state_id, exact in exact_cases.items()
    }
    model = _reconstruct_model(trained, training_config)
    _require(
        model["model_sha256"] == model["state_payload_sha256"]
        == source["model_sha256"],
        "critic gradient reconstructed model hash differs",
    )

    audit = config["audit"]
    primary_rows = tuple(int(row) for row in audit["primary_rows"])
    beta = float(audit["smoothmax_beta"])
    finite_epsilon = float(audit["normalized_feature_finite_difference_epsilon"])
    translation_scale = float(training_config["feature"][
        "translation_scale_m_per_action_unit"
    ])
    predictions = trained["compact_shared_7D"]["predictions"]
    grouped: dict[str, dict[str, dict[str, dict[int, dict[str, Any]]]]] = {}
    maximum_prediction_error = 0.0
    maximum_finite_difference_error = 0.0
    projection_cache: dict[tuple[str, int], tuple[float, list[float], float]] = {}
    for split in audit["report_splits"]:
        _require(
            len(samples[split]) == len(predictions[split]),
            "critic gradient prediction count differs",
        )
        grouped[split] = {}
        for sample, stored_prediction in zip(samples[split], predictions[split]):
            state_id = str(sample["state_id"])
            candidate_name = str(sample["candidate_name"])
            row = int(sample["row_index"])
            value, gradient_feature, finite_error = _value_and_gradient(
                model, sample["feature"], epsilon=finite_epsilon,
            )
            maximum_prediction_error = max(
                maximum_prediction_error,
                abs(value - float(stored_prediction[0])),
            )
            maximum_finite_difference_error = max(
                maximum_finite_difference_error, finite_error,
            )
            key = (state_id, row)
            if key not in projection_cache:
                projection_cache[key] = projection_geometry(exact_cases[state_id], row)
            _, normal_list, support = projection_cache[key]
            normal = np.asarray(normal_list, dtype=np.float64)
            action_gradient = np.concatenate([
                float(gradient_feature[2 + step])
                * translation_scale * normal / float(support)
                for step in range(5)
            ])
            grouped[split].setdefault(state_id, {}).setdefault(
                candidate_name, {}
            )[row] = {
                "actual": float(sample["risk"]),
                "predicted": float(value),
                "action_gradient": action_gradient.tolist(),
            }

    numerical = {
        "maximum_stored_prediction_absolute_error": float(maximum_prediction_error),
        "stored_prediction_tolerance": float(
            audit["stored_prediction_maximum_absolute_error"]
        ),
        "stored_prediction_pass": bool(
            maximum_prediction_error
            <= float(audit["stored_prediction_maximum_absolute_error"])
        ),
        "maximum_autograd_finite_difference_absolute_error": float(
            maximum_finite_difference_error
        ),
        "finite_difference_tolerance": float(
            audit["finite_difference_maximum_absolute_error"]
        ),
        "autograd_finite_difference_pass": bool(
            maximum_finite_difference_error
            <= float(audit["finite_difference_maximum_absolute_error"])
        ),
    }
    numerical["numerical_gradient_pass"] = bool(
        numerical["stored_prediction_pass"]
        and numerical["autograd_finite_difference_pass"]
    )

    trigger_by_split = {}
    trigger_by_group = {}
    pair_records_by_split = {}
    pair_metrics_by_split = {}
    state_gradient_records = {}
    for split in audit["report_splits"]:
        candidate_trigger = []
        group_trigger = {key: [] for key in training_config["group_rows"]}
        pair_records = []
        state_gradient_records[split] = []
        for state_id, state_candidates in grouped[split].items():
            for candidate_name, rows in state_candidates.items():
                _require(
                    sorted(rows) == list(range(10)),
                    "critic gradient row population differs",
                )
                candidate_trigger.append({
                    "actual": max(rows[row]["actual"] for row in primary_rows),
                    "predicted": max(
                        rows[row]["predicted"] for row in primary_rows
                    ),
                })
                for group, group_rows in training_config["group_rows"].items():
                    group_trigger[group].append({
                        "actual": max(rows[int(row)]["actual"] for row in group_rows),
                        "predicted": max(
                            rows[int(row)]["predicted"] for row in group_rows
                        ),
                    })

            nominal = state_candidates["nominal"]
            nominal_predictions = [
                nominal[row]["predicted"] for row in primary_rows
            ]
            weights = smoothmax_weights(nominal_predictions, beta)
            global_gradient = sum(
                float(weight) * np.asarray(
                    nominal[row]["action_gradient"], dtype=np.float64,
                )
                for weight, row in zip(weights, primary_rows)
            )
            active_row = int(primary_rows[int(np.argmax(nominal_predictions))])
            state_gradient_records[split].append({
                "state_id": state_id,
                "predicted_active_row": active_row,
                "predicted_active_row_risk": float(max(nominal_predictions)),
                "global_gradient_l2": float(np.linalg.norm(global_gradient)),
                "smoothmax_weights": [float(value) for value in weights],
            })
            nominal_actions = actions[state_id]["nominal"]
            nominal_true = max(
                nominal[row]["actual"] for row in primary_rows
            )
            for minus_name, plus_name in audit["candidate_pairs"]:
                minus_delta = actions[state_id][minus_name] - nominal_actions
                plus_delta = actions[state_id][plus_name] - nominal_actions
                symmetry_error = float(np.max(np.abs(minus_delta + plus_delta)))
                minus_true = max(
                    state_candidates[minus_name][row]["actual"]
                    for row in primary_rows
                )
                plus_true = max(
                    state_candidates[plus_name][row]["actual"]
                    for row in primary_rows
                )
                pair_records.append(pair_direction_record(
                    state_id=state_id,
                    pair_names=[minus_name, plus_name],
                    symmetry_error=symmetry_error,
                    predicted_minus_score=float(global_gradient @ minus_delta),
                    predicted_plus_score=float(global_gradient @ plus_delta),
                    nominal_true_risk=float(nominal_true),
                    minus_true_risk=float(minus_true),
                    plus_true_risk=float(plus_true),
                    symmetry_tolerance=float(
                        audit["post_clipping_symmetry_maximum_absolute_error"]
                    ),
                    score_tolerance=float(
                        audit["gradient_score_tie_absolute_tolerance"]
                    ),
                    risk_tolerance=float(audit["risk_tie_absolute_tolerance"]),
                ))
        trigger_by_split[split] = unsafe_recall(candidate_trigger)
        trigger_by_group[split] = {
            group: unsafe_recall(items) for group, items in group_trigger.items()
        }
        pair_records_by_split[split] = pair_records
        pair_metrics_by_split[split] = aggregate_pair_metrics(pair_records)

    decision_split = str(audit["decision_split"])
    verdict = feasibility_verdict(
        trigger=trigger_by_split[decision_split],
        pairs=pair_metrics_by_split[decision_split],
        numerical_gradient_pass=bool(numerical["numerical_gradient_pass"]),
        gate=config["feasibility_gate"],
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete_zero_simulation_critic_gradient_audit",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": cpu_allocation_record(),
        "config": config,
        "source_artifacts": {
            "training_result": source["training_result"],
            "training_validation": source["training_validation"],
            "model_sha256": source["model_sha256"],
            "sample_source": sample_source,
        },
        "numerical_gradient_audit": numerical,
        "structural_action_span": config["structural_audit"],
        "unsafe_trigger_by_split": trigger_by_split,
        "unsafe_trigger_by_group_and_split": trigger_by_group,
        "pair_metrics_by_split": pair_metrics_by_split,
        "pair_records_by_split": pair_records_by_split,
        "state_gradient_records_by_split": state_gradient_records,
        "feasibility_verdict_from_validation_only": verdict,
        "new_simulator_rollout_count": 0,
        "model_retraining_performed": False,
        "action_or_flow_correction_executed": False,
        "test_used_for_gate_or_tuning": False,
        "paper_or_safety_claim_authorized": False,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "numerical_gradient_audit": result["numerical_gradient_audit"],
        "validation_unsafe_trigger": result["unsafe_trigger_by_split"][
            "validation"
        ],
        "validation_pair_metrics": result["pair_metrics_by_split"][
            "validation"
        ],
        "feasibility_verdict": result[
            "feasibility_verdict_from_validation_only"
        ],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
