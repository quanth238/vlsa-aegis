#!/usr/bin/env python3
"""Generate and exactly evaluate frozen-critic guidance one query earlier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.audit_distal_compiled_box_risk_target import _evaluate_case
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


PRIMARY_GROUPS = (
    "palm", "finger1_base", "finger1_pad", "finger2_base",
    "finger2_pad", "L5",
)


def _allocation_record() -> dict[str, Any]:
    _require(os.environ.get("SLURM_JOB_ID") is not None,
             "early gradient-probe requires Slurm")
    devices = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    _require(len(devices) == 1 and "H100" in devices[0],
             "early gradient-probe requires one H100")
    return {
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(), "device": devices[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
    }


def _load_prep(
    *, config: Mapping[str, Any], case_index: int, prep_path: Path,
    accepted_prep_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.tight_prefix_early_gradient_probe import (
        PREP_SCHEMA, payload_sha256,
    )

    selected = config["cases"][int(case_index)]
    prep = _load(prep_path)
    _require(
        prep.get("schema_version") == PREP_SCHEMA
        and prep.get("source", {}).get("commit") == accepted_prep_commit
        and prep.get("case_index") == int(case_index)
        and prep.get("case_id") == selected["case_id"]
        and prep.get("early_state_step") == selected["early_state_step"]
        and prep.get("config_file_sha256") == config["config_file_sha256"]
        and prep.get("config_payload_sha256") == config["config_payload_sha256"]
        and prep.get("result_payload_sha256")
        == payload_sha256(prep, "result_payload_sha256")
        and prep.get("new_policy_query_count") == 0
        and prep.get("model_training_performed") is False,
        "early gradient-probe prep differs",
    )
    return prep


def _load_model_sources(
    *, repo_root: Path, config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from main.multilink_ellipsoid.tight_prefix_early_gradient_probe import (
        file_sha256,
    )
    from main.multilink_ellipsoid.tight_prefix_risk_q_diagnostic import (
        RESULT_SCHEMA as TRAINING_RESULT_SCHEMA,
        VALIDATION_SCHEMA as TRAINING_VALIDATION_SCHEMA,
        load_config as load_training_config,
        payload_sha256 as training_payload,
    )

    source = config["source"]
    training_path = repo_root / source["training_config"]
    _require(
        file_sha256(training_path) == source["training_config_file_sha256"],
        "early gradient-probe training config file differs",
    )
    training = load_training_config(training_path)
    _require(
        training["config_payload_sha256"]
        == source["training_config_payload_sha256"],
        "early gradient-probe training config payload differs",
    )
    result_path = Path(source["training_result"])
    validation_path = Path(source["training_validation"])
    _require(
        _file_sha256(result_path) == source["training_result_file_sha256"]
        and _file_sha256(validation_path)
        == source["training_validation_file_sha256"],
        "early gradient-probe model files differ",
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
        "early gradient-probe frozen model differs",
    )
    return training, trained


def _row_risk(candidate: Mapping[str, Any], rows: Sequence[int]) -> float:
    from main.multilink_ellipsoid.tight_prefix_risk_q_diagnostic import (
        row_future_risks,
    )
    return float(max(row_future_risks(candidate, rows)))


def _candidate_safe(
    candidate: Mapping[str, Any], rows: Sequence[int], car_limit: float,
) -> bool:
    exact = candidate["exact_group_target"]
    contacts = sum(
        int(exact["group_contact_sample_count"][group])
        for group in PRIMARY_GROUPS
    )
    return bool(
        _row_risk(candidate, rows) <= 0.0 and contacts == 0
        and float(candidate["replayed_maximum_CAR_m"]) <= float(car_limit)
    )


def generate_definitions(
    *, repo_root: Path, config_path: Path, case_index: int,
    prep_path: Path, expected_commit: str, accepted_prep_commit: str,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from main.multilink_ellipsoid.terminal_branching import score_terminal_bank
    from main.multilink_ellipsoid.tight_prefix_early_gradient_probe import (
        load_config, payload_sha256, symmetric_multi_probe_actions,
    )
    from scripts.evaluate_tight_prefix_gradient_probe import (
        _critic_gradient, _model_bundle,
    )

    config = load_config(config_path)
    _require(0 <= int(case_index) < len(config["cases"]),
             "early gradient-probe case index differs")
    prep = _load_prep(
        config=config, case_index=case_index, prep_path=prep_path,
        accepted_prep_commit=accepted_prep_commit,
    )
    training, trained = _load_model_sources(repo_root=repo_root, config=config)
    exact_case = prep["exact_case"]
    nominal = exact_case["candidates"][0]
    _require(nominal["name"] == "nominal",
             "early gradient-probe nominal differs")
    nominal_actions = np.asarray(
        nominal["source_executed_actions"], dtype=np.float64,
    )
    primary_rows = [int(row) for row in config["probe"]["primary_rows"]]
    bundle = _model_bundle(torch=torch, training_config=training, trained=trained)
    ordinary = np.zeros((10, 7), dtype=np.float64)
    ordinary[:5] = nominal_actions
    terminal = ordinary[None, :, :].copy()
    score = score_terminal_bank(
        exact_case=exact_case, ordinary_terminal_actions=ordinary,
        terminal_action_bank=terminal, candidate_names=["nominal"],
        required_candidate_names=["nominal"],
        state_payload=trained["compact_shared_7D"]["model"]["state_payload"],
        primary_rows=primary_rows, model_rows=list(range(10)),
        translation_scale=float(training["feature"][
            "translation_scale_m_per_action_unit"
        ]),
    )
    stored = {
        (prep["case_id"], "nominal", row): float(
            score["records"][0]["predicted_by_row"][str(row)]
        ) for row in primary_rows
    }
    critic = _critic_gradient(
        torch=torch, bundle=bundle, exact_case=exact_case,
        anchor_name="nominal", anchor_actions=nominal_actions,
        stored_predictions=stored, case_id=prep["case_id"],
        primary_rows=primary_rows,
        beta=float(config["probe"]["smoothmax_beta"]),
        translation_scale=float(training["feature"][
            "translation_scale_m_per_action_unit"
        ]),
    )
    predicted_hard = max(float(value) for value in critic["per_row_prediction"])
    exact_target = exact_case["exact_group_target"]
    initial_safe = bool(
        all(
            float(exact_target["initial_row_normalized_radial_slack"][row]) > 0.0
            for row in primary_rows
        )
        and all(
            int(exact_target["initial_group_contact_sample_count"][group]) == 0
            for group in PRIMARY_GROUPS
        )
    )
    nominal_safe = _candidate_safe(
        nominal, primary_rows, float(config["gate"]["paper_car_threshold_m"]),
    )
    trigger_margin = float(config["probe"]["trigger"]["margin"])
    triggered = bool(predicted_hard > -trigger_margin)
    eligible = bool(initial_safe and nominal_safe and triggered)
    construction = None
    overrides = []
    if eligible:
        seed = int(config["probe"]["random_seed"]) + int(
            hashlib.sha256(prep["case_id"].encode("utf-8")).hexdigest()[:8], 16
        )
        random = np.random.default_rng(seed).normal(size=(
            int(config["probe"]["random_direction_count"]), 5, 3,
        ))
        construction = symmetric_multi_probe_actions(
            nominal_actions, critic["raw_action_gradient"], random,
            requested_radius=float(config["probe"][
                "requested_translation_l2_action"
            ]),
            minimum_radius=float(config["probe"][
                "minimum_translation_l2_action"
            ]),
            bound_headroom=float(config["probe"]["symmetric_bound_headroom"]),
        )
        definitions = [
            ("gradient_down", construction["gradient_down"]),
            ("gradient_up", construction["gradient_up"]),
        ]
        definitions.extend(
            ("matched_random_%d" % index, actions)
            for index, actions in enumerate(construction["matched_random"])
        )
        overrides = [{
            "name": name, "actions": actions,
            "requested_alpha": construction["radius_l2_action"],
            "effective_correction_l2_action": construction["radius_l2_action"],
            "metadata": {"probe_name": name, "anchor_name": "nominal"},
        } for name, actions in definitions]
    result = {
        "schema_version": "vlsa_tight_prefix_early_gradient_definitions.v1",
        "status": "complete_H100_early_gradient_definitions",
        "source": _git_identity(repo_root, expected_commit),
        "case_index": int(case_index), "case_id": prep["case_id"],
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "model_sha256": bundle["model_sha256"],
        "initial_primary_safe": initial_safe,
        "nominal_exact_safe": nominal_safe,
        "nominal_exact_primary_risk": _row_risk(nominal, primary_rows),
        "nominal_predicted_primary_hard": predicted_hard,
        "trigger_margin": trigger_margin, "triggered": triggered,
        "eligible": eligible, "critic": critic,
        "probe_construction": construction, "action_overrides": overrides,
        "simulator_rollout_count": 0, "training_performed": False,
    }
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256",
    )
    return result


def evaluate_case(
    *, repo_root: Path, config_path: Path, case_index: int,
    prep_path: Path, definitions_path: Path, expected_commit: str,
    accepted_prep_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.tight_prefix_early_gradient_probe import (
        CASE_SCHEMA, load_config, payload_sha256, scientific_view,
    )
    from main.multilink_ellipsoid.tight_prefix_risk_dataset import (
        load_config as load_dataset_config,
    )

    config = load_config(config_path)
    prep = _load_prep(
        config=config, case_index=case_index, prep_path=prep_path,
        accepted_prep_commit=accepted_prep_commit,
    )
    definitions = _load(definitions_path)
    _require(
        definitions.get("schema_version")
        == "vlsa_tight_prefix_early_gradient_definitions.v1"
        and definitions.get("source", {}).get("commit") == expected_commit
        and definitions.get("case_index") == int(case_index)
        and definitions.get("case_id") == prep["case_id"]
        and definitions.get("config_file_sha256") == config["config_file_sha256"]
        and definitions.get("config_payload_sha256") == config["config_payload_sha256"]
        and definitions.get("model_sha256") == config["source"]["model_sha256"]
        and definitions.get("result_payload_sha256")
        == payload_sha256(definitions, "result_payload_sha256"),
        "early gradient-probe definitions differ",
    )
    exact_rollout = None
    risks = {}
    safe = {}
    overrides = definitions["action_overrides"]
    if definitions["eligible"]:
        dataset = load_dataset_config(
            repo_root / config["source"]["tight_dataset_config"],
            repo_root=repo_root,
        )
        source_curve = prep["source_curve"]
        exact_rollout = _evaluate_case(
            repo_root=repo_root,
            population_manifest=repo_root / config["source"]["population_manifest"],
            geometry_config_path=repo_root / config["source"][
                "legacy_replay_geometry_config"
            ],
            case_config={
                "case_id": prep["case_id"],
                "state_step": prep["early_state_step"],
                "source_result": source_curve["path"],
                "source_result_file_sha256": source_curve["file_sha256"],
                "source_result_payload_sha256": source_curve[
                    "result_payload_sha256"
                ],
                "slab_initialization": "query_state_matching_source",
                "candidate_names": [item["name"] for item in overrides],
            },
            audit_config={
                "rollout_scope": "candidate_five_action_prefix_only",
                "gate": dataset["gate"],
                "exact_group_target": dataset["exact_group_target"],
                "candidate_action_overrides": overrides,
            },
        )
        by_name = {item["name"]: item for item in exact_rollout["candidates"]}
        for name in by_name:
            risks[name] = _row_risk(
                by_name[name], config["probe"]["primary_rows"],
            )
            safe[name] = _candidate_safe(
                by_name[name], config["probe"]["primary_rows"],
                float(config["gate"]["paper_car_threshold_m"]),
            )
    nominal_risk = float(definitions["nominal_exact_primary_risk"])
    down = risks.get("gradient_down")
    random_names = [
        "matched_random_%d" % index
        for index in range(int(config["probe"]["random_direction_count"]))
    ]
    record = {
        "case_id": prep["case_id"], "split": "validation",
        "early_state_step": prep["early_state_step"],
        "initial_primary_safe": definitions["initial_primary_safe"],
        "nominal_exact_safe": definitions["nominal_exact_safe"],
        "nominal_exact_primary_risk": nominal_risk,
        "nominal_predicted_primary_hard": definitions[
            "nominal_predicted_primary_hard"
        ],
        "triggered": definitions["triggered"],
        "eligible": definitions["eligible"],
        "exact_probe_primary_risk": risks,
        "exact_probe_safe": safe,
        "gradient_down_exact_safe": bool(
            definitions["eligible"] and safe.get("gradient_down", False)
        ),
        "gradient_down_descends": bool(
            definitions["eligible"] and down < nominal_risk - 1e-10
        ),
        "gradient_down_beats_up": bool(
            definitions["eligible"]
            and down < float(risks["gradient_up"]) - 1e-10
        ),
        "gradient_down_beats_random": [
            bool(down < float(risks[name]) - 1e-10)
            for name in random_names
        ] if definitions["eligible"] else [],
        "down_risk_change": (
            0.0 if not definitions["eligible"] else float(down - nominal_risk)
        ),
        "critic": definitions["critic"],
        "probe_construction": definitions["probe_construction"],
    }
    result = {
        "schema_version": CASE_SCHEMA,
        "status": "complete_early_query_gradient_probe_case",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": _git_identity(repo_root, expected_commit),
        "allocation": _allocation_record(),
        "prep_binding": {
            "path": str(prep_path), "file_sha256": _file_sha256(prep_path),
            "result_payload_sha256": prep["result_payload_sha256"],
            "source_commit": prep["source"]["commit"],
        },
        "case_index": int(case_index), "case_id": prep["case_id"],
        "split": "validation",
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "model_sha256": definitions["model_sha256"],
        "record": record, "exact_rollout": exact_rollout,
        "new_simulator_rollout_count": len(overrides),
        "new_policy_query_count": 0, "training_performed": False,
        "QP_or_analytical_guidance_performed": False,
        "test_split_opened": False,
    }
    result["scientific_view"] = scientific_view(result)
    result["result_payload_sha256"] = payload_sha256(
        result, "result_payload_sha256",
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--accepted-prep-commit", required=True)
    parser.add_argument("--mode", choices=("generate", "evaluate"), required=True)
    parser.add_argument("--definitions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "generate":
        _require(args.definitions is None,
                 "early gradient generation definitions differ")
        result = generate_definitions(
            repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
            case_index=args.case_index, prep_path=args.prep.resolve(),
            expected_commit=args.expected_commit,
            accepted_prep_commit=args.accepted_prep_commit,
        )
    else:
        _require(args.definitions is not None,
                 "early gradient definitions are absent")
        result = evaluate_case(
            repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
            case_index=args.case_index, prep_path=args.prep.resolve(),
            definitions_path=args.definitions.resolve(),
            expected_commit=args.expected_commit,
            accepted_prep_commit=args.accepted_prep_commit,
        )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "case_id": result["case_id"],
        "eligible": result.get("eligible", result.get("record", {}).get("eligible")),
        "new_simulator_rollout_count": result.get("new_simulator_rollout_count", 0),
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
