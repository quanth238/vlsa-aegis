#!/usr/bin/env python3
"""Train prediction-only MLP diagnostics from immutable whole-body artifacts."""

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


def load_samples(
    *, repo_root: Path, config: Mapping[str, Any],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    from main.multilink_ellipsoid.exact_group_boundary import (
        CASE_SCHEMA, load_cases, load_config as load_cohort_config,
        payload_sha256 as source_payload_sha256,
    )
    from main.multilink_ellipsoid.generic_l5_9d_capacity import feature_vector
    from main.multilink_ellipsoid.whole_body_q_only_diagnostic import (
        CLAIMED_ROWS, direct_l5_33d_feature, row_future_risk,
        shared_constraint_feature,
    )

    output = {
        split: {"l5": [], "shared": []}
        for split in config["dataset"]["evaluation_splits"]
    }
    case_records = []
    excluded = []
    for source in config["sources"]:
        cohort_path = repo_root / source["cohort_config"]
        _require(
            _file_sha256(cohort_path) == source["cohort_config_file_sha256"],
            "whole-body diagnostic cohort config differs",
        )
        cohort = load_cohort_config(cohort_path)
        selections = load_cases(repo_root / cohort["selection_manifest"], cohort)
        root = Path(source["artifact_root"])
        validation_path = root / "validation.json"
        _require(
            _file_sha256(validation_path) == source["validation_file_sha256"],
            "whole-body diagnostic validation file differs",
        )
        validation = _load(validation_path)
        _require(
            validation.get("validation_payload_sha256")
            == source["validation_payload_sha256"]
            == source_payload_sha256(
                validation, key="validation_payload_sha256",
            )
            and bool(validation["independent_replay"][
                "exact_scientific_reproduction"
            ])
            and int(validation["summary"]["physical_false_safe_count"]) == 0,
            "whole-body diagnostic validation payload differs",
        )
        for selection in selections:
            split = str(selection["split"])
            if split not in output:
                continue
            case_id = str(selection["case_id"])
            path = root / "producer" / (case_id + ".json")
            record = _load(path)
            _require(
                record.get("schema_version") == CASE_SCHEMA
                and record.get("case_id") == case_id
                and record.get("source", {}).get("commit")
                == source["artifact_commit"]
                and record.get("result_payload_sha256")
                == source_payload_sha256(record),
                "whole-body diagnostic case payload differs",
            )
            exact = record["exact_case"]
            _require(
                bool(exact["source_replay_exact"])
                and bool(exact["state_hash_matches"])
                and bool(exact["exact_group_target"][
                    "robot_primitive_certificate_pass"
                ]),
                "whole-body diagnostic case replay differs",
            )
            initial = exact["exact_group_target"]
            safe = all(
                float(initial["initial_group_normalized_radial_slack"][group]) > 0.0
                and int(initial["initial_group_contact_sample_count"][group]) == 0
                for group in ("palm", "L5", "L6", "L7")
            )
            if not safe:
                excluded.append({
                    "case_id": case_id, "split": split,
                    "reason": "initially_unsafe_recovery_audit",
                })
                continue
            context = exact["physical_context"]
            known = 0
            for order, candidate in enumerate(exact["candidates"]):
                if not bool(candidate["exact_group_target"]["known_outcome"]):
                    continue
                base_9d = feature_vector(
                    eef_position_m=context["eef_position_m"],
                    candidate_actions=candidate["source_executed_actions"],
                    obstacle_center_m=context["obstacle"]["center_m"],
                    obstacle_semiaxes_m=context["obstacle"]["semiaxes_m"],
                    translation_scale_m_per_action_unit=float(
                        config["features"]["translation_scale_m_per_action_unit"]
                    ),
                )
                common = {
                    "state_id": case_id,
                    "split": split,
                    "candidate_name": str(candidate["name"]),
                    "candidate_order": int(order),
                    "applied_correction_l2_action": float(candidate[
                        "source_effective_post_AEGIS_correction_l2_action"
                    ]),
                }
                risk_rows = [row_future_risk(candidate, row) for row in (2, 3, 4)]
                output[split]["l5"].append({
                    **common,
                    "feature_9d": base_9d,
                    "feature_33d": direct_l5_33d_feature(base_9d, exact),
                    "risk_rows": risk_rows,
                    "global_risk": max(risk_rows),
                })
                for row in CLAIMED_ROWS:
                    output[split]["shared"].append({
                        **common,
                        "row_index": int(row),
                        "balance_id": f"{case_id}|row-{row}",
                        "feature": shared_constraint_feature(exact, candidate, row),
                        "risk": row_future_risk(candidate, row),
                    })
                known += 1
            case_records.append({
                "case_id": case_id,
                "split": split,
                "source": source["name"],
                "known_candidate_count": known,
                "payload_sha256": record["result_payload_sha256"],
            })
    _require(output["train"]["l5"] and output["validation"]["l5"],
             "whole-body diagnostic eligible samples are empty")
    return output, {
        "cases": case_records,
        "excluded_initially_unsafe_cases": excluded,
        "test_artifacts_accessed": False,
    }


def _arrays(samples: Sequence[Mapping[str, Any]], feature_key: str,
            target_key: str) -> tuple[Any, Any, list[str]]:
    import numpy as np

    target = [sample[target_key] for sample in samples]
    y = np.asarray(target, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    return (
        np.asarray([sample[feature_key] for sample in samples], dtype=np.float64),
        y,
        [str(sample.get("balance_id", sample["state_id"])) for sample in samples],
    )


def _predict(bundle: Mapping[str, Any], features: Any) -> Any:
    import numpy as np
    import torch

    raw = np.asarray(features, dtype=np.float64)
    normalized = torch.as_tensor(
        (raw - bundle["feature_mean"]) / bundle["feature_scale"],
        dtype=torch.float32, device=bundle["device"],
    )
    bundle["model"].eval()
    with torch.no_grad():
        value = bundle["model"](normalized).detach().cpu().numpy()
    return value * bundle["target_scale"] + bundle["target_mean"]


def train_arm(
    train_samples: Sequence[Mapping[str, Any]], *, feature_key: str,
    target_key: str, input_dimension: int, output_dimension: int,
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.generic_l5_9d_capacity import (
        canonical, state_balanced_weights, weighted_mean_scale,
    )
    from main.multilink_ellipsoid.whole_body_q_only_diagnostic import build_model

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("whole-body diagnostic requires exactly one H100")
    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name:
        raise RuntimeError("whole-body diagnostic requires an H100")
    train_x, train_y, balance_ids = _arrays(
        train_samples, feature_key, target_key,
    )
    _require(
        train_x.shape[1] == int(input_dimension)
        and train_y.shape[1] == int(output_dimension),
        "whole-body diagnostic training shape differs",
    )
    weights = np.asarray(state_balanced_weights(balance_ids), dtype=np.float64)
    feature_mean, feature_scale = weighted_mean_scale(
        train_x, weights, 1.0e-6, fallback_scale=1.0,
    )
    target_mean, target_scale = weighted_mean_scale(
        train_y, weights, float(model_config["minimum_target_scale"]),
        fallback_scale=float(model_config["minimum_target_scale"]),
    )
    seed = int(model_config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(8, int(model_config["cpu_threads"])))
    device = torch.device("cuda:0")

    def tensor(value: Any) -> Any:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    tx = tensor((train_x - feature_mean) / feature_scale)
    ty = tensor((train_y - target_mean) / target_scale)
    physical_target = tensor(train_y)
    sample_weight = tensor(weights[:, None])
    model = build_model(
        torch, input_dimension, output_dimension, model_config["hidden_widths"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_loss = math.inf
    for _ in range(int(model_config["epochs"])):
        model.train()
        element = torch.nn.functional.smooth_l1_loss(
            model(tx), ty, beta=float(model_config["huber_beta_normalized"]),
            reduction="none",
        )
        boundary = 1.0 + float(model_config["boundary_weight_multiplier"]) * torch.exp(
            -torch.abs(physical_target) / float(model_config["boundary_scale"])
        )
        combined = sample_weight * boundary
        loss = torch.sum(combined * element) / torch.sum(combined)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(model_config["gradient_clip_norm"]),
        )
        optimizer.step()
        final_loss = float(loss.item())
    model.eval()
    state_payload = {
        "input_dimension": int(input_dimension),
        "output_dimension": int(output_dimension),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "target_mean": target_mean.tolist(),
        "target_scale": target_scale.tolist(),
        "state_dict": {
            name: value.detach().cpu().numpy().tolist()
            for name, value in sorted(model.state_dict().items())
        },
    }
    return {
        "bundle": {
            "model": model, "device": device,
            "feature_mean": feature_mean, "feature_scale": feature_scale,
            "target_mean": target_mean, "target_scale": target_scale,
        },
        "state_payload": state_payload,
        "model_sha256": hashlib.sha256(canonical(state_payload)).hexdigest(),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "device_name": device_name,
        "epochs_completed": int(model_config["epochs"]),
        "final_train_loss": final_loss,
    }


def shared_metrics(
    samples: Sequence[Mapping[str, Any]], predictions: Sequence[Sequence[float]],
    *, group_rows: Mapping[str, Sequence[int]], near_boundary_abs_risk: float,
) -> dict[str, Any]:
    import numpy as np
    from main.multilink_ellipsoid.generic_l5_9d_capacity import spearman

    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    actual = np.asarray([sample["risk"] for sample in samples], dtype=np.float64)
    _require(len(predicted) == len(samples), "shared diagnostic prediction shape differs")
    by_candidate: dict[tuple[str, str], dict[str, Any]] = {}
    row_error = {str(row): [] for rows in group_rows.values() for row in rows}
    for index, sample in enumerate(samples):
        row = int(sample["row_index"])
        row_error[str(row)].append(float(predicted[index] - actual[index]))
        key = (str(sample["state_id"]), str(sample["candidate_name"]))
        entry = by_candidate.setdefault(key, {
            "state_id": key[0], "candidate_name": key[1],
            "candidate_order": int(sample["candidate_order"]),
            "correction": float(sample["applied_correction_l2_action"]),
            "actual": {}, "predicted": {},
        })
        entry["actual"][row] = float(actual[index])
        entry["predicted"][row] = float(predicted[index])
    candidates = list(by_candidate.values())
    required_rows = sorted({int(row) for rows in group_rows.values() for row in rows})
    _require(
        all(sorted(item["actual"]) == required_rows for item in candidates),
        "shared diagnostic candidate rows differ",
    )

    def aggregate(rows: Sequence[int]) -> dict[str, Any]:
        target = np.asarray([
            max(item["actual"][int(row)] for row in rows) for item in candidates
        ], dtype=np.float64)
        estimate = np.asarray([
            max(item["predicted"][int(row)] for row in rows) for item in candidates
        ], dtype=np.float64)
        actual_safe = target <= 0.0
        predicted_safe = estimate <= 0.0
        near = np.abs(target) <= float(near_boundary_abs_risk)
        return {
            "candidate_count": len(candidates),
            "RMSE": float(np.sqrt(np.mean((estimate - target) ** 2))),
            "MAE": float(np.mean(np.abs(estimate - target))),
            "false_safe_count": int(np.sum(predicted_safe & ~actual_safe)),
            "false_unsafe_count": int(np.sum(~predicted_safe & actual_safe)),
            "actual_safe_count": int(np.sum(actual_safe)),
            "predicted_safe_count": int(np.sum(predicted_safe)),
            "safe_recall": None if not np.any(actual_safe) else float(
                np.sum(predicted_safe & actual_safe) / np.sum(actual_safe)
            ),
            "near_boundary_count": int(np.sum(near)),
            "near_boundary_RMSE": None if not np.any(near) else float(
                np.sqrt(np.mean((estimate[near] - target[near]) ** 2))
            ),
            "rank_spearman": spearman(target.tolist(), estimate.tolist()),
        }

    groups = {group: aggregate(rows) for group, rows in group_rows.items()}
    global_metrics = aggregate(required_rows)
    state_rows = []
    for state_id in sorted({item["state_id"] for item in candidates}):
        items = [item for item in candidates if item["state_id"] == state_id]
        for item in items:
            item["actual_global"] = max(item["actual"].values())
            item["predicted_global"] = max(item["predicted"].values())
        accepted = [item for item in items if item["predicted_global"] <= 0.0]
        selected = None if not accepted else min(
            accepted, key=lambda item: (item["correction"], item["candidate_order"]),
        )
        state_rows.append({
            "state_id": state_id,
            "known_candidate_count": len(items),
            "actual_safe_candidate_count": sum(
                item["actual_global"] <= 0.0 for item in items
            ),
            "predicted_safe_candidate_count": len(accepted),
            "predicted_safe_support": bool(accepted),
            "selected_candidate": None if selected is None else selected["candidate_name"],
            "selected_actual_safe": None if selected is None else bool(
                selected["actual_global"] <= 0.0
            ),
        })
    global_metrics["supported_state_count"] = sum(
        row["predicted_safe_support"] for row in state_rows
    )
    global_metrics["selected_exact_safe_state_count"] = sum(
        row["selected_actual_safe"] is True for row in state_rows
    )
    global_metrics["states"] = state_rows
    return {
        "row_RMSE": {
            row: float(np.sqrt(np.mean(np.asarray(errors) ** 2)))
            for row, errors in row_error.items()
        },
        "per_group": groups,
        "global": global_metrics,
    }


def run(*, repo_root: Path, config_path: Path, expected_commit: str) -> dict[str, Any]:
    from main.multilink_ellipsoid.generic_l5_9d_capacity import diagnostic_metrics
    from main.multilink_ellipsoid.shadow import allocation_record
    from main.multilink_ellipsoid.whole_body_q_only_diagnostic import (
        RESULT_SCHEMA, load_config, payload_sha256,
    )

    config = load_config(config_path)
    samples, source_record = load_samples(repo_root=repo_root, config=config)
    model_config = config["model"]
    arms = {}
    for arm_name, feature_key in (
        ("relative_endpoint_9D", "feature_9d"),
        ("direct_L5_OSC_33D", "feature_33d"),
    ):
        arm_config = config["arms"][arm_name]
        trained = train_arm(
            samples["train"]["l5"], feature_key=feature_key,
            target_key="risk_rows",
            input_dimension=int(arm_config["input_dimension"]),
            output_dimension=int(arm_config["output_dimension"]),
            model_config=model_config,
        )
        predictions = {}
        metrics = {}
        for split in config["dataset"]["evaluation_splits"]:
            x, _, _ = _arrays(samples[split]["l5"], feature_key, "risk_rows")
            prediction = _predict(trained["bundle"], x).tolist()
            predictions[split] = prediction
            metrics[split] = diagnostic_metrics(
                samples[split]["l5"], prediction,
                near_boundary_abs_risk=float(config["metrics"][
                    "near_boundary_abs_risk"
                ]),
            )
        arms[arm_name] = {
            "model": {key: value for key, value in trained.items() if key != "bundle"},
            "predictions": predictions,
            "metrics": metrics,
        }
    shared_config = config["arms"]["shared_constraint_135D"]
    trained = train_arm(
        samples["train"]["shared"], feature_key="feature", target_key="risk",
        input_dimension=int(shared_config["input_dimension"]),
        output_dimension=int(shared_config["output_dimension"]),
        model_config=model_config,
    )
    shared_predictions = {}
    shared_split_metrics = {}
    for split in config["dataset"]["evaluation_splits"]:
        x, _, _ = _arrays(samples[split]["shared"], "feature", "risk")
        prediction = _predict(trained["bundle"], x).tolist()
        shared_predictions[split] = prediction
        shared_split_metrics[split] = shared_metrics(
            samples[split]["shared"], prediction,
            group_rows=shared_config["group_rows"],
            near_boundary_abs_risk=float(config["metrics"][
                "near_boundary_abs_risk"
            ]),
        )
    arms["shared_constraint_135D"] = {
        "model": {key: value for key, value in trained.items() if key != "bundle"},
        "predictions": shared_predictions,
        "metrics": shared_split_metrics,
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete_diagnostic_not_safety_authorization",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": allocation_record(),
        "config": config,
        "source_artifacts": source_record,
        "dataset": {
            "eligible_state_count": {
                split: len({sample["state_id"] for sample in samples[split]["l5"]})
                for split in config["dataset"]["evaluation_splits"]
            },
            "known_candidate_count": {
                split: len(samples[split]["l5"])
                for split in config["dataset"]["evaluation_splits"]
            },
            "shared_row_sample_count": {
                split: len(samples[split]["shared"])
                for split in config["dataset"]["evaluation_splits"]
            },
            "unknown_timeouts_censored": True,
            "test_artifacts_accessed": False,
            "normalization_fit_on_train_only": True,
            "validation_not_used_for_checkpoint_selection": True,
        },
        "arms": arms,
        "diagnostic_only": True,
        "training_coverage_gate_pass": False,
        "test_authorized": False,
        "correction_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
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
        "dataset": result["dataset"],
        "metrics": {name: arm["metrics"] for name, arm in result["arms"].items()},
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
