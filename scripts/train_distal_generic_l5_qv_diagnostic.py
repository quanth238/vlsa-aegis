#!/usr/bin/env python3
"""Train matched Q-only and shared Q-plus-V direct-L5 diagnostics."""

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


def load_samples(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from main.multilink_ellipsoid.exact_group_boundary import (
        CASE_SCHEMA, VALIDATION_SCHEMA as SOURCE_VALIDATION_SCHEMA,
        payload_sha256 as source_payload_sha256,
    )
    from main.multilink_ellipsoid.generic_l5_qv_diagnostic import (
        action_endpoint_feature, direct_l5_state_feature,
    )

    source = config["sources"]
    validation_path = Path(source["validation_file"])
    _require(
        _file_sha256(validation_path) == source["validation_file_sha256"],
        "generic L5 Q/V source validation file differs",
    )
    validation = _load(validation_path)
    _require(
        validation["schema_version"] == SOURCE_VALIDATION_SCHEMA
        and validation["validation_payload_sha256"]
        == source["validation_payload_sha256"]
        == source_payload_sha256(validation, "validation_payload_sha256")
        and bool(validation["summary"]["trajectory_policy_value_apparatus_pass"])
        and bool(validation["summary"]["trajectory_context_complete"])
        and float(validation["summary"]["trajectory_maximum_bellman_residual"]) == 0.0
        and bool(validation["summary"]["source_state_hash_exact"])
        and int(validation["summary"]["physical_false_safe_count"]) == 0,
        "generic L5 Q/V source validation differs",
    )
    row_indices = [int(item) for item in config["dataset"]["L5_row_indices"]]
    scale = float(config["features"]["translation_scale_m_per_action_unit"])
    root = Path(source["producer_root"])
    cases = {}
    case_records = []
    for case_id, binding in sorted(source["case_artifacts"].items()):
        path = root / binding["filename"]
        _require(_file_sha256(path) == binding["file_sha256"], "generic L5 Q/V case file differs")
        result = _load(path)
        _require(
            result["schema_version"] == CASE_SCHEMA
            and result["case_id"] == case_id
            and result["result_payload_sha256"] == binding["payload_sha256"]
            == source_payload_sha256(result)
            and bool(result["exact_case"]["state_hash_matches"]),
            "generic L5 Q/V case payload differs",
        )
        query_samples = []
        value_samples = []
        unknown = 0
        eligible = 0
        for candidate_order, candidate in enumerate(result["exact_case"]["candidates"]):
            exact = candidate["exact_group_target"]
            boundaries = exact["action_boundaries"]
            records = exact["trajectory_policy_value"]["records"]
            _require(len(boundaries) == len(records), "generic L5 Q/V boundary count differs")
            if bool(exact["known_outcome"]):
                query_samples.append({
                    "state_id": case_id,
                    "candidate_name": str(candidate["name"]),
                    "candidate_order": int(candidate_order),
                    "state_feature": direct_l5_state_feature(
                        boundaries[0], row_indices=row_indices,
                    ),
                    "action_feature": action_endpoint_feature(
                        candidate["source_executed_actions"],
                        translation_scale_m_per_action_unit=scale,
                    ),
                    "target": float(exact["group_future_violation"]["L5"]),
                    "source_terminal_status": str(candidate["source_terminal_status"]),
                })
            else:
                unknown += 1
            for boundary, record in zip(boundaries, records):
                _require(
                    int(boundary["action_offset"]) == int(record["action_offset"])
                    and str(boundary["phase"]) == str(record["phase"]),
                    "generic L5 Q/V boundary binding differs",
                )
                if not bool(record["training_sample_eligible"]):
                    continue
                value_samples.append({
                    "state_id": case_id,
                    "trajectory_id": f"{case_id}/{candidate['name']}",
                    "candidate_name": str(candidate["name"]),
                    "action_offset": int(record["action_offset"]),
                    "phase": str(record["phase"]),
                    "state_feature": direct_l5_state_feature(
                        boundary, row_indices=row_indices,
                    ),
                    "target": float(record["value"]["L5"]),
                })
                eligible += 1
        expected = config["dataset"]["expected_counts"][case_id]
        _require(
            len(query_samples) == int(expected["known_query"])
            and unknown == int(expected["unknown_query"])
            and eligible == int(expected["eligible_value"]),
            "generic L5 Q/V case counts differ",
        )
        cases[case_id] = {"query": query_samples, "value": value_samples}
        case_records.append({
            "case_id": case_id, "file": str(path),
            "file_sha256": binding["file_sha256"],
            "payload_sha256": binding["payload_sha256"],
            "known_query_count": len(query_samples),
            "unknown_query_count": unknown,
            "eligible_value_count": eligible,
        })
    expected_ids = set(
        config["dataset"]["boundary_case_ids"]
        + config["dataset"]["safe_auxiliary_case_ids"]
        + config["dataset"]["recovery_case_ids"]
    )
    _require(set(cases) == expected_ids, "generic L5 Q/V case population differs")
    return cases, {
        "source_validation_file": str(validation_path),
        "source_validation_file_sha256": source["validation_file_sha256"],
        "source_validation_payload_sha256": source["validation_payload_sha256"],
        "cases": case_records,
    }


def q_arrays(samples: Sequence[Mapping[str, Any]]) -> tuple[Any, Any, Any, list[str]]:
    import numpy as np
    return (
        np.asarray([item["state_feature"] for item in samples], dtype=np.float64),
        np.asarray([item["action_feature"] for item in samples], dtype=np.float64),
        np.asarray([item["target"] for item in samples], dtype=np.float64),
        [str(item["state_id"]) for item in samples],
    )


def v_arrays(samples: Sequence[Mapping[str, Any]]) -> tuple[Any, Any, list[str]]:
    import numpy as np
    return (
        np.asarray([item["state_feature"] for item in samples], dtype=np.float64),
        np.asarray([item["target"] for item in samples], dtype=np.float64),
        [str(item["state_id"]) for item in samples],
    )


def train_arm(
    q_train: Sequence[Mapping[str, Any]], v_train: Sequence[Mapping[str, Any]],
    q_validation: Sequence[Mapping[str, Any]], model_config: Mapping[str, Any],
    *, arm: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.generic_l5_qv_diagnostic import (
        build_model, canonical, state_balanced_weights, weighted_mean_scale,
    )

    if arm not in ("q_only", "q_plus_v"):
        raise ValueError("generic L5 Q/V arm differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("generic L5 Q/V training requires exactly one H100")
    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name:
        raise RuntimeError("generic L5 Q/V training requires an H100")
    q_state, q_action, q_target, q_ids = q_arrays(q_train)
    v_state, v_target, v_ids = v_arrays(v_train)
    validation_state, validation_action, _, _ = q_arrays(q_validation)
    q_weights = np.asarray(state_balanced_weights(q_ids), dtype=np.float64)
    v_weights = np.asarray(state_balanced_weights(v_ids), dtype=np.float64)
    state_mean, state_scale = weighted_mean_scale(
        q_state, q_weights, 1.0e-6, fallback_scale=1.0,
    )
    action_mean, action_scale = weighted_mean_scale(
        q_action, q_weights, 1.0e-6, fallback_scale=1.0,
    )
    target_mean, target_scale = weighted_mean_scale(
        q_target[:, None], q_weights, float(model_config["minimum_target_scale"]),
        fallback_scale=float(model_config["minimum_target_scale"]),
    )
    target_mean = target_mean.reshape(1)
    target_scale = target_scale.reshape(1)
    seed = int(model_config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(8, int(model_config["cpu_threads"])))
    device = torch.device("cuda:0")

    def tensor(value: Any) -> Any:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    qs = tensor((q_state - state_mean) / state_scale)
    qa = tensor((q_action - action_mean) / action_scale)
    qt = tensor((q_target[:, None] - target_mean) / target_scale)
    qphysical = tensor(q_target[:, None])
    qw = tensor(q_weights[:, None])
    vs = tensor((v_state - state_mean) / state_scale)
    vt = tensor((v_target[:, None] - target_mean) / target_scale)
    vphysical = tensor(v_target[:, None])
    vw = tensor(v_weights[:, None])
    model = build_model(torch, int(model_config["hidden_width"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    beta = float(model_config["huber_beta_normalized"])
    multiplier = float(model_config["boundary_weight_multiplier"])
    boundary_scale = float(model_config["boundary_scale"])
    final_q = math.inf
    final_v = 0.0
    for _ in range(int(model_config["epochs"])):
        model.train()
        q_element = torch.nn.functional.smooth_l1_loss(
            model.q(qs, qa), qt, beta=beta, reduction="none",
        )
        q_boundary = 1.0 + multiplier * torch.exp(-torch.abs(qphysical) / boundary_scale)
        q_loss = torch.sum(qw * q_boundary * q_element) / torch.sum(qw * q_boundary)
        if arm == "q_plus_v":
            v_element = torch.nn.functional.smooth_l1_loss(
                model.v(vs), vt, beta=beta, reduction="none",
            )
            v_boundary = 1.0 + multiplier * torch.exp(-torch.abs(vphysical) / boundary_scale)
            v_loss = torch.sum(vw * v_boundary * v_element) / torch.sum(vw * v_boundary)
            loss = q_loss + float(model_config["value_loss_weight"]) * v_loss
            final_v = float(v_loss.item())
        else:
            loss = q_loss
            final_v = 0.0
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
        "state_payload": state_payload,
        "model_sha256": hashlib.sha256(canonical(state_payload)).hexdigest(),
        "parameter_count": int(sum(item.numel() for item in model.parameters())),
        "device_name": device_name,
        "epochs_completed": int(model_config["epochs"]),
        "final_q_loss": final_q, "final_v_loss": final_v,
        "train_prediction": predict_q(bundle, q_state, q_action).tolist(),
        "validation_prediction": predict_q(
            bundle, validation_state, validation_action,
        ).tolist(),
    }


def load_bundle(torch: Any, payload: Mapping[str, Any], model_config: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    from main.multilink_ellipsoid.generic_l5_qv_diagnostic import build_model
    device = torch.device("cuda:0")
    model = build_model(torch, int(model_config["hidden_width"])).to(device)
    current = model.state_dict()
    converted = {
        name: torch.as_tensor(value, dtype=current[name].dtype, device=device)
        for name, value in payload["state_dict"].items()
    }
    model.load_state_dict(converted)
    model.eval()
    return {
        "model": model, "device": device,
        "state_mean": np.asarray(payload["state_mean"], dtype=np.float64),
        "state_scale": np.asarray(payload["state_scale"], dtype=np.float64),
        "action_mean": np.asarray(payload["action_mean"], dtype=np.float64),
        "action_scale": np.asarray(payload["action_scale"], dtype=np.float64),
        "target_mean": np.asarray(payload["target_mean"], dtype=np.float64),
        "target_scale": np.asarray(payload["target_scale"], dtype=np.float64),
    }


def predict_q(bundle: Mapping[str, Any], state: Any, action: Any) -> Any:
    import numpy as np
    import torch
    normalized_state = torch.as_tensor(
        (np.asarray(state) - bundle["state_mean"]) / bundle["state_scale"],
        dtype=torch.float32, device=bundle["device"],
    )
    normalized_action = torch.as_tensor(
        (np.asarray(action) - bundle["action_mean"]) / bundle["action_scale"],
        dtype=torch.float32, device=bundle["device"],
    )
    with torch.no_grad():
        normalized = bundle["model"].q(normalized_state, normalized_action).cpu().numpy()
    return (normalized * bundle["target_scale"] + bundle["target_mean"]).reshape(-1)


def run(*, repo_root: Path, config_path: Path, expected_commit: str) -> dict[str, Any]:
    from main.multilink_ellipsoid.generic_l5_qv_diagnostic import (
        RESULT_SCHEMA, diagnostic_metrics, load_config, payload_sha256,
    )

    config = load_config(config_path)
    identity = _git_identity(repo_root)
    _require(identity["commit"] == expected_commit and not identity["dirty"], "generic L5 Q/V source is not immutable")
    cases, source_record = load_samples(config)
    boundary_ids = [str(item) for item in config["dataset"]["boundary_case_ids"]]
    auxiliary_ids = [str(item) for item in config["dataset"]["safe_auxiliary_case_ids"]]
    recovery_ids = [str(item) for item in config["dataset"]["recovery_case_ids"]]
    folds = []
    aggregate = {arm: {"samples": [], "predictions": []} for arm in config["model"]["arms"]}
    for held_out in boundary_ids:
        training_ids = [item for item in boundary_ids + auxiliary_ids if item != held_out]
        q_train = [sample for case_id in training_ids for sample in cases[case_id]["query"]]
        v_train = [sample for case_id in training_ids for sample in cases[case_id]["value"]]
        q_validation = cases[held_out]["query"]
        arm_records = {}
        for arm in config["model"]["arms"]:
            trained = train_arm(
                q_train, v_train, q_validation, config["model"], arm=arm,
            )
            train_metrics = diagnostic_metrics(q_train, trained["train_prediction"])
            validation_metrics = diagnostic_metrics(q_validation, trained["validation_prediction"])
            arm_records[arm] = {
                "model": {key: value for key, value in trained.items() if key not in (
                    "bundle", "train_prediction", "validation_prediction",
                )},
                "train_prediction": trained["train_prediction"],
                "validation_prediction": trained["validation_prediction"],
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
            }
            aggregate[arm]["samples"].extend(q_validation)
            aggregate[arm]["predictions"].extend(trained["validation_prediction"])
        folds.append({
            "held_out_state_id": held_out,
            "training_state_ids": training_ids,
            "training_query_count": len(q_train),
            "training_value_count": len(v_train),
            "validation_query_count": len(q_validation),
            "arms": arm_records,
        })
    aggregate_metrics = {
        arm: diagnostic_metrics(record["samples"], record["predictions"])
        for arm, record in aggregate.items()
    }
    q_only = aggregate_metrics["q_only"]
    q_plus_v = aggregate_metrics["q_plus_v"]
    rule = config["decision"]["q_plus_v_improvement_rule"]
    mechanism_pass = bool(
        q_plus_v["rmse"] <= float(rule["maximum_RMSE_ratio"]) * q_only["rmse"]
        and q_plus_v["false_safe_count"] < q_only["false_safe_count"]
        and q_plus_v["safe_support_state_count"] >= q_only["safe_support_state_count"]
    )
    all_fit_ids = boundary_ids + auxiliary_ids
    recovery_records = {}
    q_train_all = [sample for case_id in all_fit_ids for sample in cases[case_id]["query"]]
    v_train_all = [sample for case_id in all_fit_ids for sample in cases[case_id]["value"]]
    recovery_samples = [sample for case_id in recovery_ids for sample in cases[case_id]["query"]]
    for arm in config["model"]["arms"]:
        trained = train_arm(
            q_train_all, v_train_all, recovery_samples, config["model"], arm=arm,
        )
        recovery_records[arm] = {
            "model": {key: value for key, value in trained.items() if key not in (
                "bundle", "train_prediction", "validation_prediction",
            )},
            "prediction": trained["validation_prediction"],
            "metrics": diagnostic_metrics(recovery_samples, trained["validation_prediction"]),
        }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "claim_scope": config["claim_scope"],
        "config_file": str(config_path),
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "source_commit": expected_commit,
        "source": source_record,
        "folds": folds,
        "aggregate_metrics": aggregate_metrics,
        "recovery_diagnostic": recovery_records,
        "q_plus_v_mechanism_pass": mechanism_pass,
        "interpretation": (
            "trajectory_value_supervision_reduces_unseen_state_offset"
            if mechanism_pass else "trajectory_value_supervision_does_not_resolve_unseen_state_offset"
        ),
        "training_dataset_gate_pass": False,
        "correction_authorized": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
        "result_payload_sha256": "",
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
        repo_root=args.repo_root, config_path=args.config,
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "interpretation": result["interpretation"],
        "aggregate_metrics": result["aggregate_metrics"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
