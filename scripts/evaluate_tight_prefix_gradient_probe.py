#!/usr/bin/env python3
"""Run matched exact five-action probes around unsafe validation anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.audit_distal_compiled_box_risk_target import _evaluate_case
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_tight_prefix_risk_q_diagnostic import load_samples


def _allocation_record(torch: Any) -> dict[str, Any]:
    _require(os.environ.get("SLURM_JOB_ID") is not None,
             "tight gradient-probe requires Slurm")
    _require(torch.cuda.is_available(),
             "tight gradient-probe requires allocated CUDA")
    device_name = str(torch.cuda.get_device_name(0))
    _require("H100" in device_name, "tight gradient-probe requires H100")
    return {
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(),
        "device": device_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
    }


def _load_sources(
    *, repo_root: Path, config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    from main.multilink_ellipsoid.tight_prefix_gradient_probe import file_sha256
    from main.multilink_ellipsoid.tight_prefix_risk_dataset import (
        CASE_SCHEMA as DATASET_CASE_SCHEMA,
        load_config as load_dataset_config,
        payload_sha256 as dataset_payload,
    )
    from main.multilink_ellipsoid.tight_prefix_risk_q_diagnostic import (
        RESULT_SCHEMA as TRAINING_RESULT_SCHEMA,
        VALIDATION_SCHEMA as TRAINING_VALIDATION_SCHEMA,
        load_config as load_training_config,
        payload_sha256 as training_payload,
    )

    source = config["source"]
    training_config_path = repo_root / source["training_config"]
    dataset_config_path = repo_root / source["dataset_config"]
    _require(
        file_sha256(training_config_path)
        == source["training_config_file_sha256"]
        and file_sha256(dataset_config_path)
        == source["dataset_config_file_sha256"],
        "tight gradient-probe config binding differs",
    )
    training_config = load_training_config(training_config_path)
    dataset_config = load_dataset_config(dataset_config_path, repo_root=repo_root)
    _require(
        training_config["config_payload_sha256"]
        == source["training_config_payload_sha256"]
        and dataset_config["config_payload_sha256"]
        == source["dataset_config_payload_sha256"],
        "tight gradient-probe config payload differs",
    )
    result_path = Path(source["training_result"])
    validation_path = Path(source["training_validation"])
    _require(
        _file_sha256(result_path) == source["training_result_file_sha256"]
        and _file_sha256(validation_path)
        == source["training_validation_file_sha256"],
        "tight gradient-probe model file differs",
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
        and trained["compact_shared_7D"]["model"]["model_sha256"]
        == source["model_sha256"],
        "tight gradient-probe validated model differs",
    )
    artifacts = {}
    producer_dir = Path(source["dataset_producer_dir"])
    for case in config["cases"]:
        case_id = str(case["case_id"])
        record = _load(producer_dir / (case_id + ".json"))
        _require(
            record.get("schema_version") == DATASET_CASE_SCHEMA
            and record.get("case_id") == case_id
            and record.get("split") == "validation"
            and record.get("source", {}).get("commit")
            == source["dataset_artifact_commit"]
            and record.get("result_payload_sha256") == dataset_payload(record),
            "tight gradient-probe dataset case differs",
        )
        artifacts[case_id] = record
    return training_config, dataset_config, trained, artifacts


def _model_bundle(
    *, torch: Any, training_config: Mapping[str, Any],
    trained: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    from main.multilink_ellipsoid.whole_body_q_only_diagnostic import build_model

    record = trained["compact_shared_7D"]["model"]
    payload = record["state_payload"]
    model = build_model(
        torch, int(payload["input_dimension"]), int(payload["output_dimension"]),
        training_config["model"]["hidden_widths"],
    )
    model.load_state_dict({
        name: torch.as_tensor(value, dtype=torch.float32)
        for name, value in payload["state_dict"].items()
    }, strict=True)
    model = model.float().cuda().eval()
    return {
        "model": model,
        "feature_mean": torch.as_tensor(
            np.asarray(payload["feature_mean"], dtype=np.float32),
            device="cuda", dtype=torch.float32,
        ),
        "feature_scale": torch.as_tensor(
            np.asarray(payload["feature_scale"], dtype=np.float32),
            device="cuda", dtype=torch.float32,
        ),
        "target_mean": torch.as_tensor(
            np.asarray(payload["target_mean"], dtype=np.float32),
            device="cuda", dtype=torch.float32,
        ),
        "target_scale": torch.as_tensor(
            np.asarray(payload["target_scale"], dtype=np.float32),
            device="cuda", dtype=torch.float32,
        ),
        "model_sha256": str(record["model_sha256"]),
    }


def _prediction_map(
    *, repo_root: Path, training_config: Mapping[str, Any],
    trained: Mapping[str, Any],
) -> dict[tuple[str, str, int], float]:
    samples, _ = load_samples(repo_root=repo_root, config=training_config)
    output = {}
    predictions = trained["compact_shared_7D"]["predictions"]
    for sample, prediction in zip(samples["validation"], predictions["validation"]):
        key = (
            str(sample["state_id"]), str(sample["candidate_name"]),
            int(sample["row_index"]),
        )
        output[key] = float(prediction[0])
    return output


def _critic_gradient(
    *, torch: Any, bundle: Mapping[str, Any], exact_case: Mapping[str, Any],
    anchor_name: str, anchor_actions: Sequence[Sequence[float]],
    stored_predictions: Mapping[tuple[str, str, int], float],
    case_id: str, primary_rows: Sequence[int], beta: float,
    translation_scale: float,
) -> dict[str, Any]:
    import numpy as np
    from main.multilink_ellipsoid.tight_prefix_critic_gradient_audit import (
        projection_geometry,
    )

    actions = torch.as_tensor(
        np.asarray(anchor_actions, dtype=np.float32),
        device="cuda", dtype=torch.float32,
    ).clone().requires_grad_(True)
    nominal_first = np.asarray(
        exact_case["source_nominal_five_action_chunk"][0][:3],
        dtype=np.float32,
    )
    predictions = []
    stored = []
    geometry_records = []
    for row in primary_rows:
        slack, normal_record, support = projection_geometry(exact_case, int(row))
        normal = torch.as_tensor(
            np.asarray(normal_record, dtype=np.float32),
            device="cuda", dtype=torch.float32,
        )
        nominal_projection = float(
            float(translation_scale) * float(normal.detach().cpu().numpy() @ nominal_first)
            / float(support)
        )
        effective_projection = (
            float(translation_scale) * (actions[:, :3] @ normal)
            / float(support)
        )
        feature = torch.cat((
            torch.as_tensor(
                [float(slack), nominal_projection], device="cuda",
                dtype=torch.float32,
            ),
            effective_projection,
        ))
        normalized = (
            feature - bundle["feature_mean"]
        ) / bundle["feature_scale"]
        value = (
            bundle["model"](normalized)
            * bundle["target_scale"] + bundle["target_mean"]
        ).reshape(())
        predictions.append(value)
        stored.append(float(stored_predictions[(case_id, anchor_name, int(row))]))
        geometry_records.append({
            "row_index": int(row), "initial_slack": float(slack),
            "normal": [float(item) for item in normal_record],
            "support_radius_m": float(support),
        })
    vector = torch.stack(predictions)
    global_value = (
        torch.logsumexp(float(beta) * vector, dim=0)
        - math.log(len(primary_rows))
    ) / float(beta)
    gradient = torch.autograd.grad(global_value, actions)[0][:, :3]
    predicted = vector.detach().cpu().numpy().astype(np.float64)
    stored_array = np.asarray(stored, dtype=np.float64)
    return {
        "global_smooth_risk": float(global_value.detach().cpu().item()),
        "per_row_prediction": predicted.tolist(),
        "per_row_stored_prediction": stored_array.tolist(),
        "maximum_stored_prediction_absolute_error": float(
            np.max(np.abs(predicted - stored_array))
        ),
        "raw_action_gradient": gradient.detach().cpu().numpy().astype(
            np.float64
        ).tolist(),
        "raw_action_gradient_l2": float(torch.linalg.vector_norm(
            gradient
        ).detach().cpu().item()),
        "geometry": geometry_records,
    }


def _primary_risk(candidate: Mapping[str, Any], primary_rows: Sequence[int]) -> float:
    from main.multilink_ellipsoid.tight_prefix_risk_q_diagnostic import (
        row_future_risks,
    )

    return float(max(row_future_risks(candidate, primary_rows)))


def evaluate_case(
    *, repo_root: Path, config_path: Path, case_index: int,
    expected_commit: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from main.multilink_ellipsoid.tight_prefix_gradient_probe import (
        CASE_SCHEMA, load_config, payload_sha256, symmetric_probe_actions,
    )
    from main.multilink_ellipsoid.tight_prefix_risk_dataset import (
        load_config as load_dataset_config,
    )

    config = load_config(config_path)
    _require(0 <= int(case_index) < len(config["cases"]),
             "tight gradient-probe case index differs")
    training_config, dataset_config, trained, artifacts = _load_sources(
        repo_root=repo_root, config=config,
    )
    selected = config["cases"][int(case_index)]
    case_id = str(selected["case_id"])
    exact_record = artifacts[case_id]
    exact_case = exact_record["case"]
    case_config = next(
        item for item in dataset_config["cases"]
        if str(item["case_id"]) == case_id
    )
    bundle = _model_bundle(
        torch=torch, training_config=training_config, trained=trained,
    )
    stored_predictions = _prediction_map(
        repo_root=repo_root, training_config=training_config, trained=trained,
    )
    primary_rows = [int(row) for row in config["probe"]["primary_rows"]]
    source_by_name = {
        str(candidate["name"]): candidate
        for candidate in exact_case["candidates"]
    }
    anchor_records = []
    overrides = []
    for anchor_order, anchor_name in enumerate(
        selected["unsafe_anchor_candidates"]
    ):
        anchor = source_by_name[str(anchor_name)]
        anchor_actions = np.asarray(
            anchor["source_executed_actions"], dtype=np.float64,
        )
        anchor_true_risk = _primary_risk(anchor, primary_rows)
        _require(anchor_true_risk > 0.0,
                 "tight gradient-probe anchor is not unsafe")
        critic = _critic_gradient(
            torch=torch, bundle=bundle, exact_case=exact_case,
            anchor_name=str(anchor_name), anchor_actions=anchor_actions,
            stored_predictions=stored_predictions, case_id=case_id,
            primary_rows=primary_rows,
            beta=float(config["probe"]["smoothmax_beta"]),
            translation_scale=float(training_config["feature"][
                "translation_scale_m_per_action_unit"
            ]),
        )
        seed_digest = hashlib.sha256(
            (case_id + "|" + str(anchor_name)).encode("utf-8")
        ).hexdigest()
        seed = int(config["probe"]["random_seed"]) + int(seed_digest[:8], 16)
        random = np.random.default_rng(seed).normal(size=(5, 3))
        probes = symmetric_probe_actions(
            anchor_actions, critic["raw_action_gradient"], random,
            requested_radius=float(config["probe"][
                "requested_translation_l2_action"
            ]),
            minimum_radius=float(config["probe"][
                "minimum_translation_l2_action"
            ]),
            bound_headroom=float(config["probe"]["symmetric_bound_headroom"]),
        )
        anchor_key = "anchor-%d" % int(anchor_order)
        for probe_name in config["probe"]["probe_names"]:
            name = anchor_key + "__" + str(probe_name)
            overrides.append({
                "name": name,
                "actions": probes[str(probe_name)],
                "requested_alpha": float(probes["radius_l2_action"]),
                "effective_correction_l2_action": float(
                    probes["radius_l2_action"]
                ),
                "metadata": {
                    "anchor_name": str(anchor_name),
                    "anchor_order": int(anchor_order),
                    "probe_name": str(probe_name),
                },
            })
        anchor_records.append({
            "anchor_name": str(anchor_name),
            "anchor_order": int(anchor_order),
            "anchor_actions": anchor_actions.tolist(),
            "anchor_true_primary_risk": float(anchor_true_risk),
            "critic": critic,
            "probe_construction": probes,
        })
    replay_case_config = dict(case_config)
    replay_case_config["candidate_names"] = [item["name"] for item in overrides]
    replay_case_config["slab_initialization"] = "query_state_matching_source"
    evaluated = _evaluate_case(
        repo_root=repo_root,
        population_manifest=repo_root / dataset_config["source"][
            "population_manifest"
        ],
        geometry_config_path=repo_root / dataset_config["source"][
            "legacy_replay_geometry_config"
        ],
        case_config=replay_case_config,
        audit_config={
            "rollout_scope": "candidate_five_action_prefix_only",
            "gate": dataset_config["gate"],
            "exact_group_target": dataset_config["exact_group_target"],
            "candidate_action_overrides": overrides,
        },
    )
    evaluated_by_name = {
        str(candidate["name"]): candidate
        for candidate in evaluated["candidates"]
    }
    for anchor in anchor_records:
        prefix = "anchor-%d__" % int(anchor["anchor_order"])
        risks = {
            probe_name: _primary_risk(
                evaluated_by_name[prefix + probe_name], primary_rows,
            )
            for probe_name in config["probe"]["probe_names"]
        }
        anchor_risk = float(anchor["anchor_true_primary_risk"])
        down = float(risks["gradient_down"])
        anchor["exact_probe_primary_risk"] = risks
        anchor["direction_correct"] = bool(
            down < float(risks["gradient_up"]) - 1e-10
        )
        anchor["gradient_down_descends"] = bool(down < anchor_risk - 1e-10)
        anchor["gradient_down_beats_random"] = bool(
            down < float(risks["matched_random"]) - 1e-10
        )
        anchor["gradient_down_converts_safe"] = bool(down <= 0.0)
        anchor["down_risk_change"] = float(down - anchor_risk)
        anchor["anchor_prediction_absolute_error"] = float(
            anchor["critic"]["maximum_stored_prediction_absolute_error"]
        )
        anchor["post_clipping_symmetry_error"] = float(
            anchor["probe_construction"]["post_clipping_symmetry_error"]
        )

    output = {
        "schema_version": CASE_SCHEMA,
        "status": "complete_exact_gradient_probe_case",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": _allocation_record(torch),
        "case_index": int(case_index),
        "case_id": case_id,
        "split": "validation",
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "model_sha256": bundle["model_sha256"],
        "anchor_records": anchor_records,
        "exact_rollout": evaluated,
        "new_simulator_rollout_count": len(overrides),
        "training_or_policy_query_performed": False,
        "QP_or_flow_guidance_performed": False,
        "test_split_opened": False,
        "paper_or_safety_claim_authorized": False,
    }
    output["result_payload_sha256"] = payload_sha256(
        output, "result_payload_sha256",
    )
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_case(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        case_index=args.case_index, expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "case_id": result["case_id"],
        "anchor_count": len(result["anchor_records"]),
        "new_simulator_rollout_count": result["new_simulator_rollout_count"],
        "anchor_summary": [
            {
                "anchor": item["anchor_name"],
                "anchor_risk": item["anchor_true_primary_risk"],
                "probe_risk": item["exact_probe_primary_risk"],
                "direction_correct": item["direction_correct"],
                "descent": item["gradient_down_descends"],
            }
            for item in result["anchor_records"]
        ],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
