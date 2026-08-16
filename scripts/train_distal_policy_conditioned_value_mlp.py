#!/usr/bin/env python3
"""Train matched Q-only and PNCBF-style shared Q/V future-risk MLPs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _flatten(dataset: Mapping[str, Any], sample_key: str, split: str) -> Sequence[Mapping[str, Any]]:
    return [
        sample
        for case in dataset["case_samples"] if str(case["split"]) == split
        for sample in case[sample_key]
    ]


def _arrays(samples: Sequence[Mapping[str, Any]], with_action: bool) -> Tuple[Any, ...]:
    import numpy as np

    states = np.asarray([sample["state_feature"] for sample in samples], dtype=np.float64)
    targets = np.asarray([sample["target"] for sample in samples], dtype=np.float64)
    if with_action:
        actions = np.asarray([sample["action_feature"] for sample in samples], dtype=np.float64)
        return states, actions, targets
    return states, targets


def _predict(bundle: Mapping[str, Any], states: Any, actions: Optional[Any]) -> Any:
    import numpy as np
    import torch

    state = torch.as_tensor(
        (np.asarray(states) - bundle["state_mean"]) / bundle["state_scale"],
        dtype=torch.float32, device=bundle["device"],
    )
    with torch.no_grad():
        if actions is None:
            normalized = bundle["model"].v(state)
        else:
            action = torch.as_tensor(
                (np.asarray(actions) - bundle["action_mean"]) / bundle["action_scale"],
                dtype=torch.float32, device=bundle["device"],
            )
            normalized = bundle["model"].q(state, action)
    return (
        normalized.cpu().numpy() * bundle["target_scale"] + bundle["target_mean"]
    ).reshape(-1)


def _train_arm(
    q_train: Sequence[Mapping[str, Any]], v_train: Sequence[Mapping[str, Any]],
    model_config: Mapping[str, Any], arm: str,
) -> Dict[str, Any]:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.policy_conditioned_value_model import (
        build_model, canonical, state_balanced_weights, weighted_mean_scale,
    )

    if arm not in ("q_only", "q_plus_v"):
        raise ValueError("policy-value training arm differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("policy-value training requires exactly one allocation GPU")
    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name:
        raise RuntimeError("policy-value training requires an H100")
    q_state, q_action, q_target = _arrays(q_train, True)
    v_state, v_target = _arrays(v_train, False)
    if q_state.shape[1] != v_state.shape[1]:
        raise ValueError("policy-value state dimensions differ")
    q_weight = np.asarray(state_balanced_weights(q_train), dtype=np.float64)
    v_weight = np.asarray(state_balanced_weights(v_train), dtype=np.float64)
    state_mean, state_scale = weighted_mean_scale(
        q_state, q_weight, 1.0e-6, 1.0,
    )
    action_mean, action_scale = weighted_mean_scale(
        q_action, q_weight, 1.0e-6, 1.0,
    )
    target_mean, target_scale = weighted_mean_scale(
        q_target[:, None], q_weight,
        float(model_config["minimum_target_scale"]),
        float(model_config["minimum_target_scale"]),
    )
    seed = int(model_config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(8, int(model_config["cpu_threads"])))
    device = torch.device("cuda:0")
    model = build_model(
        torch, q_state.shape[1], q_action.shape[1],
        int(model_config["hidden_width"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )

    def tensor(value: Any) -> Any:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    qs = tensor((q_state - state_mean) / state_scale)
    qa = tensor((q_action - action_mean) / action_scale)
    qt = tensor((q_target[:, None] - target_mean) / target_scale)
    qw = tensor(q_weight[:, None])
    vs = tensor((v_state - state_mean) / state_scale)
    vt = tensor((v_target[:, None] - target_mean) / target_scale)
    vw = tensor(v_weight[:, None])
    final_q = math.inf
    final_v = 0.0
    for _ in range(int(model_config["epochs"])):
        model.train()
        q_element = torch.nn.functional.smooth_l1_loss(
            model.q(qs, qa), qt,
            beta=float(model_config["huber_beta_normalized"]), reduction="none",
        )
        q_loss = torch.sum(qw * q_element) / torch.sum(qw)
        if arm == "q_plus_v":
            v_element = torch.nn.functional.smooth_l1_loss(
                model.v(vs), vt,
                beta=float(model_config["huber_beta_normalized"]), reduction="none",
            )
            v_loss = torch.sum(vw * v_element) / torch.sum(vw)
            loss = q_loss + float(model_config["value_loss_weight"]) * v_loss
            final_v = float(v_loss.item())
        else:
            loss = q_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(model_config["gradient_clip_norm"]),
        )
        optimizer.step()
        final_q = float(q_loss.item())
    model.eval()
    state_payload = {
        "arm": arm,
        "state_dimension": int(q_state.shape[1]),
        "action_dimension": int(q_action.shape[1]),
        "state_mean": state_mean.tolist(), "state_scale": state_scale.tolist(),
        "action_mean": action_mean.tolist(), "action_scale": action_scale.tolist(),
        "target_mean": target_mean.tolist(), "target_scale": target_scale.tolist(),
        "state_dict": {
            name: value.detach().cpu().numpy().tolist()
            for name, value in sorted(model.state_dict().items())
        },
    }
    bundle = {
        "model": model, "device": device,
        "state_mean": state_mean, "state_scale": state_scale,
        "action_mean": action_mean, "action_scale": action_scale,
        "target_mean": target_mean, "target_scale": target_scale,
    }
    return {
        "bundle": bundle,
        "model_sha256": hashlib.sha256(canonical(state_payload)).hexdigest(),
        "state_payload": state_payload,
        "device_name": device_name,
        "parameter_count": int(sum(item.numel() for item in model.parameters())),
        "epochs_completed": int(model_config["epochs"]),
        "final_q_loss": final_q, "final_v_loss": final_v,
    }


def run(*, repo_root: Path, config_path: Path, expected_commit: str) -> Mapping[str, Any]:
    from main.multilink_ellipsoid.policy_conditioned_value_dataset import (
        DATASET_SCHEMA, payload_sha256 as dataset_payload_sha256,
    )
    from main.multilink_ellipsoid.policy_conditioned_value_model import (
        RESULT_SCHEMA, load_config, payload_sha256, risk_metrics,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    source = config["source_dataset"]
    path = Path(source["file"])
    _require(_file_sha256(path) == source["file_sha256"], "policy-value dataset file differs")
    dataset = _load(path)
    _require(
        dataset["schema_version"] == DATASET_SCHEMA
        and dataset["dataset_payload_sha256"] == source["payload_sha256"]
        == dataset_payload_sha256(dataset, "dataset_payload_sha256")
        and bool(dataset["training_authorized"])
        and bool(dataset["readiness_checks"]["training_ready"]),
        "policy-value dataset is not training ready",
    )
    q_samples = {
        split: _flatten(dataset, "query_samples", split)
        for split in ("train", "validation", "test")
    }
    v_samples = {
        split: _flatten(dataset, "value_samples", split)
        for split in ("train", "validation", "test")
    }
    _require(all(q_samples.values()) and all(v_samples.values()), "policy-value split is empty")
    arms = {}
    for arm in config["model"]["arms"]:
        trained = _train_arm(q_samples["train"], v_samples["train"], config["model"], arm)
        metrics = {}
        q_predictions = {}
        v_predictions = {}
        for split in ("train", "validation", "test"):
            qs, qa, _ = _arrays(q_samples[split], True)
            vs, _ = _arrays(v_samples[split], False)
            q_predictions[split] = _predict(trained["bundle"], qs, qa).tolist()
            v_predictions[split] = _predict(trained["bundle"], vs, None).tolist()
            metrics[split] = {
                "query_Q": risk_metrics(
                    q_samples[split], q_predictions[split],
                    float(config["metrics"]["near_boundary_abs_risk"]),
                ),
                "policy_V": risk_metrics(
                    v_samples[split], v_predictions[split],
                    float(config["metrics"]["near_boundary_abs_risk"]),
                ),
            }
        public_model = {
            key: value for key, value in trained.items() if key != "bundle"
        }
        arms[arm] = {
            "model": public_model,
            "query_predictions": q_predictions,
            "value_predictions": v_predictions,
            "metrics": metrics,
        }
    checks = {}
    for arm, record in arms.items():
        for split in ("validation", "test"):
            metrics = record["metrics"][split]["query_Q"]
            checks[arm + "/" + split + "/zero_false_safe"] = (
                int(metrics["false_safe_count"]) == 0
            )
            checks[arm + "/" + split + "/safe_support"] = (
                int(metrics["safe_support_state_count"])
                == int(metrics["recoverable_state_count"])
            )
            near_rmse = metrics["near_boundary_rmse"]
            checks[arm + "/" + split + "/near_boundary_RMSE"] = (
                near_rmse is not None
                and float(near_rmse) <= float(config["decision"]["maximum_near_boundary_RMSE"])
            )
    q_only = arms["q_only"]["metrics"]["test"]["query_Q"]
    q_plus_v = arms["q_plus_v"]["metrics"]["test"]["query_Q"]
    checks["q_plus_v_not_worse_than_q_only"] = bool(
        q_plus_v["rmse"] <= q_only["rmse"]
        and q_plus_v["false_safe_count"] <= q_only["false_safe_count"]
        and q_plus_v["safe_support_state_count"] >= q_only["safe_support_state_count"]
    )
    gate_pass = bool(checks and all(checks.values()))
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "policy_value_prediction_gate_pass" if gate_pass else "prediction_no_go",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": identity,
        "allocation": allocation_record(),
        "config_file": str(config_path),
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "dataset": {
            "file": str(path), "file_sha256": source["file_sha256"],
            "payload_sha256": source["payload_sha256"],
            "normalization_fit_on_train_only": True,
            "episode_groups_preserved": True,
            "fixed_epoch_model": True,
            "test_accessed_once_after_training": True,
        },
        "learned_object": "policy_conditioned_future_violation_not_OSC_execution",
        "arms": arms,
        "decision_checks": checks,
        "prediction_gate_pass": gate_pass,
        "correction_authorized": False,
        "QP_authorized": False,
        "denoising_authorized": False,
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
        "status": result["status"],
        "decision_checks": result["decision_checks"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
