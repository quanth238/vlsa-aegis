#!/usr/bin/env python3
"""Run the matched 9D versus causal 33D L5 context ablation."""

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


def _feature_key(arm_name: str) -> str:
    return "feature_9d" if arm_name == "relative_endpoint_9D" else "feature_33d"


def arrays(samples: Sequence[Mapping[str, Any]], arm_name: str) -> tuple[Any, Any, list[str]]:
    import numpy as np

    key = _feature_key(arm_name)
    return (
        np.asarray([sample[key] for sample in samples], dtype=np.float64),
        np.asarray([sample["risk_rows"] for sample in samples], dtype=np.float64),
        [str(sample["state_id"]) for sample in samples],
    )


def load_ablation_samples(
    *, repo_root: Path, config: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    from main.multilink_ellipsoid.exact_group_boundary import payload_sha256 as source_payload_sha256
    from main.multilink_ellipsoid.prospective_l5_context_ablation import direct_l5_osc_feature
    from main.multilink_ellipsoid.prospective_l5_q_only_diagnostic import (
        RESULT_SCHEMA as FROZEN_RESULT_SCHEMA,
        VALIDATION_SCHEMA as FROZEN_VALIDATION_SCHEMA,
        load_config as load_base_config,
        payload_sha256 as frozen_payload_sha256,
    )
    from scripts.train_distal_prospective_l5_q_only_diagnostic import load_samples

    source = config["sources"]
    base_config_path = repo_root / source["base_config_relative_path"]
    base_config = load_base_config(base_config_path)
    frozen_result_path = Path(source["frozen_result_file"])
    frozen_validation_path = Path(source["frozen_validation_file"])
    _require(_file_sha256(frozen_result_path) == source["frozen_result_file_sha256"], "frozen 9D result file differs")
    _require(_file_sha256(frozen_validation_path) == source["frozen_validation_file_sha256"], "frozen 9D validation file differs")
    frozen_result = _load(frozen_result_path)
    frozen_validation = _load(frozen_validation_path)
    _require(
        frozen_result["schema_version"] == FROZEN_RESULT_SCHEMA
        and frozen_result["result_payload_sha256"] == source["frozen_result_payload_sha256"]
        == frozen_payload_sha256(frozen_result, "result_payload_sha256")
        and frozen_result["config"] == base_config,
        "frozen 9D result payload differs",
    )
    _require(
        frozen_validation["schema_version"] == FROZEN_VALIDATION_SCHEMA
        and frozen_validation["validation_payload_sha256"] == source["frozen_validation_payload_sha256"]
        == frozen_payload_sha256(frozen_validation, "validation_payload_sha256")
        and frozen_validation["result_payload_sha256"] == frozen_result["result_payload_sha256"]
        and float(frozen_validation["frozen_prediction_replay_maximum_error"]) <= 1.0e-9
        and float(frozen_validation["independent_retrain_prediction_maximum_error"]) <= 1.0e-9,
        "frozen 9D validation payload differs",
    )
    samples, case_record = load_samples(base_config)
    exact_by_state: dict[str, Mapping[str, Any]] = {}
    for state_id, binding in base_config["sources"]["case_artifacts"].items():
        path = Path(base_config["sources"]["producer_root"]) / binding["filename"]
        case = _load(path)
        _require(
            case["result_payload_sha256"] == binding["payload_sha256"]
            == source_payload_sha256(case),
            "context ablation case payload differs",
        )
        exact_by_state[str(state_id)] = case["exact_case"]
    primitive_indices = config["arms"]["direct_L5_OSC_33D"]["initial_L5_primitive_row_indices"]
    slack_indices = config["arms"]["direct_L5_OSC_33D"]["initial_exact_slack_indices"]
    augmented: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "validation", "test")}
    for split in augmented:
        for sample in samples[split]:
            state_id = str(sample["state_id"])
            item = dict(sample)
            item["feature_9d"] = [float(value) for value in sample["feature"]]
            item["feature_33d"] = direct_l5_osc_feature(
                item["feature_9d"], exact_by_state[state_id],
                primitive_indices=primitive_indices, slack_indices=slack_indices,
            )
            augmented[split].append(item)
    source_record = {
        "base_config_file": str(base_config_path),
        "base_config_file_sha256": base_config["config_file_sha256"],
        "base_config_payload_sha256": base_config["config_payload_sha256"],
        "frozen_result_file": str(frozen_result_path),
        "frozen_result_file_sha256": source["frozen_result_file_sha256"],
        "frozen_result_payload_sha256": source["frozen_result_payload_sha256"],
        "frozen_validation_file": str(frozen_validation_path),
        "frozen_validation_file_sha256": source["frozen_validation_file_sha256"],
        "frozen_validation_payload_sha256": source["frozen_validation_payload_sha256"],
        "cases": case_record["cases"],
    }
    return augmented, source_record, frozen_result


def train_arm(
    train_samples: Sequence[Mapping[str, Any]],
    validation_samples: Sequence[Mapping[str, Any]],
    *, arm_name: str, model_config: Mapping[str, Any], input_dimension: int,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.generic_l5_9d_capacity import (
        canonical, state_balanced_weights, weighted_mean_scale,
    )
    from main.multilink_ellipsoid.prospective_l5_context_ablation import build_model

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("prospective L5 context ablation requires exactly one H100")
    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name:
        raise RuntimeError("prospective L5 context ablation requires an H100")
    train_x, train_y, train_state_ids = arrays(train_samples, arm_name)
    validation_x, _, _ = arrays(validation_samples, arm_name)
    weights = np.asarray(state_balanced_weights(train_state_ids), dtype=np.float64)
    feature_mean, feature_scale = weighted_mean_scale(train_x, weights, 1.0e-6, fallback_scale=1.0)
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
    model = build_model(torch, input_dimension, model_config["hidden_widths"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    final_loss = math.inf
    for _ in range(int(model_config["epochs"])):
        model.train()
        element = torch.nn.functional.smooth_l1_loss(
            model(tx), ty, beta=float(model_config["huber_beta_normalized"]), reduction="none",
        )
        boundary = 1.0 + float(model_config["boundary_weight_multiplier"]) * torch.exp(
            -torch.abs(physical_target) / float(model_config["boundary_scale"])
        )
        combined = sample_weight * boundary
        loss = torch.sum(combined * element) / torch.sum(combined)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(model_config["gradient_clip_norm"]))
        optimizer.step()
        final_loss = float(loss.item())
    model.eval()
    state_payload = {
        "input_dimension": int(input_dimension),
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


def load_bundle(
    torch: Any, payload: Mapping[str, Any], model_config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    from main.multilink_ellipsoid.prospective_l5_context_ablation import build_model

    input_dimension = int(payload["input_dimension"])
    model = build_model(torch, input_dimension, model_config["hidden_widths"])
    template = model.state_dict()
    _require(set(payload["state_dict"]) == set(template), "context ablation model keys differ")
    state = {}
    for name, value in template.items():
        raw = torch.as_tensor(payload["state_dict"][name], dtype=value.dtype)
        _require(raw.shape == value.shape, "context ablation model shape differs")
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


def feature_distribution_audit(
    samples: Mapping[str, Sequence[Mapping[str, Any]]], arm_name: str,
) -> dict[str, Any]:
    import numpy as np

    train_x, _, _ = arrays(samples["train"], arm_name)
    lower, upper = np.min(train_x, axis=0), np.max(train_x, axis=0)
    output: dict[str, Any] = {
        "input_dimension": int(train_x.shape[1]),
        "train_minimum": lower.tolist(),
        "train_maximum": upper.tolist(),
    }
    for split in ("train", "validation", "test"):
        values, _, _ = arrays(samples[split], arm_name)
        outside = (values < lower[None, :]) | (values > upper[None, :])
        output[split] = {
            "element_outside_train_range_fraction": float(np.mean(outside)),
            "sample_with_any_outside_train_range_fraction": float(np.mean(np.any(outside, axis=1))),
        }
    return output


def run(*, repo_root: Path, config_path: Path, expected_commit: str) -> dict[str, Any]:
    import numpy as np
    from main.multilink_ellipsoid.generic_l5_9d_capacity import diagnostic_metrics
    from main.multilink_ellipsoid.prospective_l5_context_ablation import (
        RESULT_SCHEMA, classify_ablation, load_config, payload_sha256,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    samples, source_record, frozen_result = load_ablation_samples(repo_root=repo_root, config=config)
    base_model_config = frozen_result["config"]["model"]
    arms: dict[str, Any] = {}
    for arm_name, arm_config in config["arms"].items():
        trained = train_arm(
            samples["train"], samples["validation"], arm_name=arm_name,
            model_config=base_model_config, input_dimension=int(arm_config["input_dimension"]),
        )
        predictions = {
            "train": trained["train_prediction"],
            "validation": trained["validation_prediction"],
            "test": predict(trained["bundle"], arrays(samples["test"], arm_name)[0]).tolist(),
        }
        metrics = {
            split: diagnostic_metrics(
                samples[split], predictions[split],
                near_boundary_abs_risk=float(frozen_result["config"]["metrics"]["near_boundary_abs_risk"]),
            ) for split in ("train", "validation", "test")
        }
        arms[arm_name] = {
            "model": {key: value for key, value in trained.items() if key not in ("bundle", "train_prediction", "validation_prediction")},
            "predictions": predictions,
            "metrics": metrics,
            "feature_distribution_audit": feature_distribution_audit(samples, arm_name),
        }
    frozen_error = 0.0
    for split in ("train", "validation", "test"):
        frozen_error = max(frozen_error, float(np.max(np.abs(
            np.asarray(arms["relative_endpoint_9D"]["predictions"][split], dtype=np.float64)
            - np.asarray(frozen_result["predictions"][split], dtype=np.float64)
        ))))
    _require(frozen_error <= 1.0e-9, "matched 9D arm does not reproduce ADR-0169")
    interpretation, checks = classify_ablation(
        arms["relative_endpoint_9D"]["metrics"],
        arms["direct_L5_OSC_33D"]["metrics"], config["decision"],
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": identity,
        "allocation": allocation_record(),
        "config": config,
        "source_artifacts": source_record,
        "dataset": {
            "state_count": {split: len(frozen_result["config"]["dataset"]["split_case_ids"][split]) for split in ("train", "validation", "test")},
            "known_sample_count": {split: len(samples[split]) for split in ("train", "validation", "test")},
            "unknown_timeouts_masked": True,
            "same_samples_and_targets_between_arms": True,
            "opened_test_reused_for_root_cause_only": True,
        },
        "frozen_9D_prediction_maximum_error": frozen_error,
        "arms": arms,
        "decision_checks": checks,
        "interpretation": interpretation,
        "new_generalization_evidence": False,
        "training_dataset_gate_pass": False,
        "correction_authorized": False,
        "QP_authorized": False,
        "calibration_authorized": False,
        "closed_loop_authorized": False,
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
        "interpretation": result["interpretation"],
        "metrics": {name: arm["metrics"] for name, arm in result["arms"].items()},
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
