#!/usr/bin/env python3
"""Matched Test A/Test B audit of L5 action versus state generalization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_l5_action_risk import _arrays, _load_samples


def _state_ids(samples: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(item["state_id"]) for item in samples})


def audit(
    *, repo_root: Path, config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.l5_action_risk import (
        AUDIT_SCHEMA, load_config as load_model_config, predict, train_model,
    )
    from main.multilink_ellipsoid.l5_state_generalization import (
        RESULT_SCHEMA, canonical, load_config, payload_sha256,
        prediction_diagnostic, prediction_gate,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    source = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    source_config_path = repo_root / config["source"]["base_model_config"]
    _require(
        _file_sha256(source_config_path)
        == config["source"]["base_model_config_file_sha256"],
        "L5 state-generalization base model config differs",
    )
    model_config = load_model_config(source_config_path)
    audit_path = Path(config["source"]["dataset_audit"])
    dataset_audit = _load(audit_path)
    _require(
        _file_sha256(audit_path)
        == config["source"]["dataset_audit_file_sha256"]
        and dataset_audit["result_payload_sha256"]
        == config["source"]["dataset_audit_payload_sha256"]
        and dataset_audit["schema_version"] == AUDIT_SCHEMA
        and dataset_audit["MLP_training_authorized"] is True
        and all(dataset_audit["gates"].values()),
        "L5 state-generalization dataset audit differs",
    )
    samples = _load_samples(model_config)
    fit = [
        item for item in samples["train"]
        if item["temporal_profile"] in {None, "constant"}
    ]
    test_a = [
        item for item in samples["train"]
        if item["temporal_profile"] == "front_loaded"
    ]
    test_b = [
        item for item in samples["validation"]
        if item["temporal_profile"] == "front_loaded"
    ]
    _require(fit and test_a and test_b, "L5 matched split is empty")
    fit_states = _state_ids(fit)
    test_a_states = _state_ids(test_a)
    test_b_states = _state_ids(test_b)
    _require(
        fit_states == test_a_states
        and not set(fit_states).intersection(test_b_states),
        "L5 matched state partition differs",
    )
    train_x, train_y = _arrays(fit)
    test_a_x, test_a_y = _arrays(test_a)
    bundle = train_model(
        train_x, train_y, test_a_x, test_a_y, model_config["model"]
    )
    split_samples = {"fit": fit, "test_A": test_a, "test_B": test_b}
    predictions = {}
    metrics = {}
    gates = {}
    for split_name, split_items in split_samples.items():
        features, _ = _arrays(split_items)
        prediction = predict(bundle, features)
        predictions[split_name] = prediction.tolist()
        metrics[split_name] = prediction_diagnostic(prediction, split_items)
        gates[split_name] = prediction_gate(
            metrics[split_name], config["diagnostic_gate"]
        )
    test_a_pass = all(gates["test_A"].values())
    test_b_pass = all(gates["test_B"].values())
    if test_a_pass and not test_b_pass:
        interpretation = "state_generalization_is_primary_observed_failure"
    elif not test_a_pass:
        interpretation = (
            "known_state_action_interpolation_also_fails_"
            "representation_or_action_coverage_not_excluded"
        )
    else:
        interpretation = "matched_action_and_state_prediction_gate_pass"
    state_payload = bundle["state_payload"]
    _require(
        bundle["model_sha256"]
        == hashlib.sha256(canonical(state_payload)).hexdigest(),
        "L5 matched model payload differs",
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": source,
        "allocation": allocation_record(),
        "config": config,
        "base_model_config": model_config,
        "dataset": {
            "audit_path": str(audit_path),
            "audit_file_sha256": _file_sha256(audit_path),
            "audit_payload_sha256": dataset_audit["result_payload_sha256"],
            "reserved_test_episodes_unopened": True,
            "new_simulation": False,
        },
        "split": {
            "sample_counts": {
                name: len(items) for name, items in split_samples.items()
            },
            "state_ids": {
                name: _state_ids(items) for name, items in split_samples.items()
            },
            "fit_and_test_A_share_states": fit_states == test_a_states,
            "test_B_states_disjoint": not set(fit_states).intersection(test_b_states),
            "test_action_profile": "front_loaded",
        },
        "model": {
            key: value for key, value in bundle.items()
            if key not in {
                "model", "device", "feature_mean", "feature_scale",
                "target_mean", "target_scale",
            }
        },
        "predictions": predictions,
        "metrics": metrics,
        "gates": gates,
        "test_A_pass": test_a_pass,
        "test_B_pass": test_b_pass,
        "interpretation": interpretation,
        "AEGIS_EE_compatibility_evaluated": False,
        "QP_authorized": False,
        "closed_loop_authorized": False,
    }
    result["result_payload_sha256"] = payload_sha256(result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "sample_counts": result["split"]["sample_counts"],
        "test_A": result["metrics"]["test_A"],
        "test_B": result["metrics"]["test_B"],
        "gates": result["gates"],
        "model_sha256": result["model"]["model_sha256"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
