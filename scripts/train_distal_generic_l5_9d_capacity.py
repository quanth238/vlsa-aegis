#!/usr/bin/env python3
"""Run a leave-one-state-out 9D L5 future-risk capacity diagnostic."""

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
    config: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    from main.multilink_ellipsoid.exact_group_boundary import (
        CASE_SCHEMA, VALIDATION_SCHEMA as SOURCE_VALIDATION_SCHEMA,
        payload_sha256 as source_payload_sha256,
    )
    from main.multilink_ellipsoid.generic_l5_9d_capacity import feature_vector

    source = config["sources"]
    validation_path = Path(source["validation_file"])
    _require(
        _file_sha256(validation_path) == source["validation_file_sha256"],
        "generic L5 9D source validation file differs",
    )
    validation = _load(validation_path)
    _require(
        validation["schema_version"] == SOURCE_VALIDATION_SCHEMA
        and validation["validation_payload_sha256"]
        == source["validation_payload_sha256"]
        == source_payload_sha256(validation, "validation_payload_sha256")
        and bool(validation["summary"]["source_state_hash_exact"])
        and int(validation["summary"]["physical_false_safe_count"]) == 0,
        "generic L5 9D source validation differs",
    )
    producer_root = Path(source["producer_root"])
    prevention_ids = [str(item) for item in config["dataset"]["prevention_case_ids"]]
    prevention = set(prevention_ids)
    recovery = str(config["dataset"]["recovery_case_id"])
    row_indices = [int(item) for item in config["dataset"]["L5_row_indices"]]
    scale = float(config["features"]["translation_scale_m_per_action_unit"])
    grouped: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in prevention_ids}
    recovery_samples: list[dict[str, Any]] = []
    case_records = []
    for case_id in sorted(prevention | {recovery}):
        binding = source["case_artifacts"][case_id]
        path = producer_root / binding["filename"]
        _require(_file_sha256(path) == binding["file_sha256"], "generic L5 case file differs")
        result = _load(path)
        _require(
            result["schema_version"] == CASE_SCHEMA
            and result["case_id"] == case_id
            and result["result_payload_sha256"] == binding["payload_sha256"]
            == source_payload_sha256(result),
            "generic L5 case payload differs",
        )
        exact = result["exact_case"]
        context = exact["physical_context"]
        _require(
            bool(exact["source_replay_exact"])
            and bool(exact["state_hash_matches"])
            and bool(exact["exact_group_target"]["robot_primitive_certificate_pass"]),
            "generic L5 case replay differs",
        )
        known_count = 0
        for order, candidate in enumerate(exact["candidates"]):
            if not bool(candidate["exact_group_target"]["known_outcome"]):
                continue
            row_slack = candidate["compiled_box_row_minimum_normalized_radial_slack"]
            _require(len(row_slack) == 7, "generic L5 row target shape differs")
            risk_rows = [-float(row_slack[index]) for index in row_indices]
            actions = candidate["source_executed_actions"]
            sample = {
                "state_id": case_id,
                "candidate_name": str(candidate["name"]),
                "candidate_order": int(order),
                "feature": feature_vector(
                    eef_position_m=context["eef_position_m"],
                    candidate_actions=actions,
                    obstacle_center_m=context["obstacle"]["center_m"],
                    obstacle_semiaxes_m=context["obstacle"]["semiaxes_m"],
                    translation_scale_m_per_action_unit=scale,
                ),
                "risk_rows": risk_rows,
                "global_risk": max(risk_rows),
                "source_terminal_status": candidate["source_terminal_status"],
                "applied_correction_l2_action": float(
                    candidate["source_effective_post_AEGIS_correction_l2_action"]
                ),
            }
            (recovery_samples if case_id == recovery else grouped[case_id]).append(sample)
            known_count += 1
        expected = int(config["dataset"]["expected_known_candidate_counts"][case_id])
        _require(known_count == expected, "generic L5 known candidate count differs")
        case_records.append({
            "case_id": case_id, "file": str(path),
            "file_sha256": binding["file_sha256"],
            "payload_sha256": binding["payload_sha256"],
            "known_candidate_count": known_count,
            "category": "recovery" if case_id == recovery else "prevention",
        })
    return grouped, recovery_samples, {
        "source_validation_file_sha256": source["validation_file_sha256"],
        "source_validation_payload_sha256": source["validation_payload_sha256"],
        "cases": case_records,
    }


def arrays(samples: Sequence[Mapping[str, Any]]) -> tuple[Any, Any, list[str]]:
    import numpy as np

    return (
        np.asarray([sample["feature"] for sample in samples], dtype=np.float64),
        np.asarray([sample["risk_rows"] for sample in samples], dtype=np.float64),
        [str(sample["state_id"]) for sample in samples],
    )


def train_fold(
    train_samples: Sequence[Mapping[str, Any]],
    validation_samples: Sequence[Mapping[str, Any]],
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.generic_l5_9d_capacity import (
        build_model, canonical, state_balanced_weights, weighted_mean_scale,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("generic L5 9D training requires exactly one H100")
    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name:
        raise RuntimeError("generic L5 9D training requires an H100")
    train_x, train_y, train_state_ids = arrays(train_samples)
    validation_x, _, _ = arrays(validation_samples)
    weights = np.asarray(state_balanced_weights(train_state_ids), dtype=np.float64)
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
    model = build_model(torch, model_config["hidden_widths"]).to(device)
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
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "target_mean": target_mean.tolist(),
        "target_scale": target_scale.tolist(),
        "state_dict": {
            name: value.detach().cpu().numpy().tolist()
            for name, value in sorted(model.state_dict().items())
        },
    }
    bundle = {
        "model": model, "device": device,
        "feature_mean": feature_mean, "feature_scale": feature_scale,
        "target_mean": target_mean, "target_scale": target_scale,
    }
    return {
        "bundle": bundle,
        "state_payload": state_payload,
        "model_sha256": hashlib.sha256(canonical(state_payload)).hexdigest(),
        "parameter_count": int(sum(item.numel() for item in model.parameters())),
        "device_name": device_name,
        "epochs_completed": int(model_config["epochs"]),
        "final_train_loss": final_loss,
        "train_prediction": predict(bundle, train_x).tolist(),
        "validation_prediction": predict(bundle, validation_x).tolist(),
    }


def load_bundle(torch: Any, payload: Mapping[str, Any], model_config: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    from main.multilink_ellipsoid.generic_l5_9d_capacity import build_model

    model = build_model(torch, model_config["hidden_widths"])
    template = model.state_dict()
    _require(set(payload["state_dict"]) == set(template), "generic L5 model keys differ")
    state = {}
    for name, value in template.items():
        raw = torch.as_tensor(payload["state_dict"][name], dtype=value.dtype)
        _require(raw.shape == value.shape, "generic L5 model shape differs")
        state[name] = raw
    model.load_state_dict(state)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return {
        "model": model, "device": device,
        "feature_mean": np.asarray(payload["feature_mean"], dtype=np.float64),
        "feature_scale": np.asarray(payload["feature_scale"], dtype=np.float64),
        "target_mean": np.asarray(payload["target_mean"], dtype=np.float64),
        "target_scale": np.asarray(payload["target_scale"], dtype=np.float64),
    }


def predict(bundle: Mapping[str, Any], features: Any) -> Any:
    import numpy as np
    import torch

    raw = np.asarray(features, dtype=np.float64)
    normalized = torch.as_tensor(
        (raw - bundle["feature_mean"]) / bundle["feature_scale"],
        dtype=torch.float32, device=bundle["device"],
    )
    bundle["model"].eval()
    with torch.no_grad():
        output = bundle["model"](normalized).detach().cpu().numpy()
    return output * bundle["target_scale"] + bundle["target_mean"]


def run(*, repo_root: Path, config_path: Path, expected_commit: str) -> dict[str, Any]:
    from main.multilink_ellipsoid.generic_l5_9d_capacity import (
        RESULT_SCHEMA, diagnostic_metrics, load_config, payload_sha256,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    grouped, recovery_samples, source_record = load_samples(config)
    folds = []
    all_predictions = []
    all_samples = []
    for held_out in config["dataset"]["prevention_case_ids"]:
        train_samples = [
            sample for state_id, samples in grouped.items() if state_id != held_out
            for sample in samples
        ]
        validation_samples = list(grouped[held_out])
        trained = train_fold(train_samples, validation_samples, config["model"])
        train_metrics = diagnostic_metrics(
            train_samples, trained["train_prediction"],
            near_boundary_abs_risk=float(config["metrics"]["near_boundary_abs_risk"]),
        )
        metrics = diagnostic_metrics(
            validation_samples, trained["validation_prediction"],
            near_boundary_abs_risk=float(config["metrics"]["near_boundary_abs_risk"]),
        )
        folds.append({
            "held_out_state_id": held_out,
            "training_state_ids": sorted(set(grouped) - {held_out}),
            "training_sample_count": len(train_samples),
            "validation_sample_count": len(validation_samples),
            "model": {key: value for key, value in trained.items() if key not in (
                "bundle", "train_prediction", "validation_prediction",
            )},
            "train_prediction": trained["train_prediction"],
            "validation_prediction": trained["validation_prediction"],
            "train_metrics": train_metrics,
            "metrics": metrics,
        })
        all_predictions.extend(trained["validation_prediction"])
        all_samples.extend(validation_samples)
    aggregate = diagnostic_metrics(
        all_samples, all_predictions,
        near_boundary_abs_risk=float(config["metrics"]["near_boundary_abs_risk"]),
    )
    all_prevention = [sample for samples in grouped.values() for sample in samples]
    recovery_fit = train_fold(all_prevention, recovery_samples, config["model"])
    recovery_metrics = diagnostic_metrics(
        recovery_samples, recovery_fit["validation_prediction"],
        near_boundary_abs_risk=float(config["metrics"]["near_boundary_abs_risk"]),
    )
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": identity, "allocation": allocation_record(), "config": config,
        "source_artifacts": source_record,
        "dataset": {
            "prevention_state_count": len(grouped),
            "prevention_known_sample_count": len(all_prevention),
            "recovery_state_count": 1,
            "recovery_known_sample_count": len(recovery_samples),
            "unknown_timeouts_masked": True,
            "candidate_groups_preserved": True,
            "leave_one_state_out": True,
        },
        "folds": folds,
        "leave_one_state_out_metrics": aggregate,
        "recovery_diagnostic": {
            "state_id": config["dataset"]["recovery_case_id"],
            "model": {key: value for key, value in recovery_fit.items() if key not in (
                "bundle", "train_prediction", "validation_prediction",
            )},
            "prediction": recovery_fit["validation_prediction"],
            "metrics": recovery_metrics,
            "excluded_from_prevention_claim": True,
        },
        "interpretation": "capacity_diagnostic_only_no_control_authority",
        "training_dataset_gate_pass": False,
        "correction_authorized": False, "QP_authorized": False,
        "calibration_authorized": False, "closed_loop_authorized": False,
    }
    result["result_payload_sha256"] = payload_sha256(result, "result_payload_sha256")
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
        "metrics": result["leave_one_state_out_metrics"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
