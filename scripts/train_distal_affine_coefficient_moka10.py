#!/usr/bin/env python3
"""Train the authorized state-conditioned affine-row MLP on H100."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.affine_coefficient_model import (
    AFFINE_COEFFICIENT_DATASET_SCHEMA,
    AFFINE_COEFFICIENT_TRAINING_SCHEMA,
    affine_values,
    load_affine_coefficient_config,
    save_affine_coefficient_model,
    train_affine_coefficient_model,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-result", type=Path, required=True)
    parser.add_argument("--dataset-validation", type=Path, required=True)
    parser.add_argument("--target-analysis", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_affine_coefficient_config(args.config.resolve())
    dataset = _load(args.dataset.resolve())
    result = _load(args.dataset_result.resolve())
    validation = _load(args.dataset_validation.resolve())
    target_analysis = _load(args.target_analysis.resolve())
    _require(
        dataset.get("schema_version") == AFFINE_COEFFICIENT_DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == _hash_without(dataset, "dataset_payload_sha256")
        and result.get("dataset", {}).get("file_sha256")
        == _file_sha256(args.dataset.resolve())
        and result.get("decision", {}).get("neural_training_authorized") is False
        and validation.get("neural_training_authorized") is False
        and target_analysis.get("schema_version")
        == "vlsa_distal_affine_coefficient_target_analysis_result.v1"
        and target_analysis.get("dataset_file_sha256")
        == _file_sha256(args.dataset.resolve())
        and target_analysis.get("decision", {}).get(
            "neural_training_authorized"
        ) is True,
        "affine-coefficient dataset did not authorize training",
    )
    states = dataset["state_records"]
    model, model_state, audit = train_affine_coefficient_model(states, config)
    del model
    predicted = model_state.pop("predicted")
    validation_indexes = [
        index for index, state in enumerate(states) if state["split"] == "validation"
    ]
    calibration_over = [[] for _ in range(7)]
    for index in validation_indexes:
        state = states[index]
        for xyz, true in zip(
            state["candidate_first_xyz"], state["candidate_minimum_distal_margin_m"]
        ):
            lower = affine_values(
                predicted["nominal_margin_m"][index],
                predicted["gradient_m_per_action"][index],
                predicted["state_conditioned_error_m"][index],
                xyz, state["nominal_first_action"][:3],
            )
            for row in range(7):
                calibration_over[row].append(float(lower[row] - true[row]))
    padding = float(config["calibration"]["fixed_padding_m"])
    calibration = np.asarray([
        max(0.0, max(values)) + padding for values in calibration_over
    ], dtype=np.float64)
    model_state["calibration_m"] = calibration

    per_case = {}
    total_false_safe_rows = 0
    total_false_safe_candidates = 0
    all_test_cosines = []
    for case_id in sorted({state["case_id"] for state in states if state["split"] == "test"}):
        indexes = [
            index for index, state in enumerate(states)
            if state["split"] == "test" and state["case_id"] == case_id
        ]
        b_error = []
        e_error = []
        cosines = []
        false_rows = 0
        false_candidates = 0
        for index in indexes:
            state = states[index]
            target = state["coefficient_target"]
            b_error.extend(
                predicted["nominal_margin_m"][index]
                - np.asarray(target["nominal_margin_m"], dtype=np.float64)
            )
            e_error.extend(
                predicted["state_conditioned_error_m"][index]
                - np.asarray(target["state_conditioned_error_m"], dtype=np.float64)
            )
            true_a = np.asarray(target["gradient_m_per_action"], dtype=np.float64)
            learned_a = predicted["gradient_m_per_action"][index]
            for row in range(7):
                denominator = float(
                    np.linalg.norm(true_a[row]) * np.linalg.norm(learned_a[row])
                )
                if denominator > 1.0e-12:
                    cosines.append(float(np.dot(true_a[row], learned_a[row]) / denominator))
            for xyz, true in zip(
                state["candidate_first_xyz"], state["candidate_minimum_distal_margin_m"]
            ):
                lower = affine_values(
                    predicted["nominal_margin_m"][index],
                    predicted["gradient_m_per_action"][index],
                    predicted["state_conditioned_error_m"][index] + calibration,
                    xyz, state["nominal_first_action"][:3],
                )
                false = np.logical_and(lower >= 0.0, np.asarray(true) < 0.0)
                false_rows += int(np.count_nonzero(false))
                false_candidates += int(np.any(false))
        all_test_cosines.extend(cosines)
        total_false_safe_rows += false_rows
        total_false_safe_candidates += false_candidates
        per_case[case_id] = {
            "state_count": len(indexes),
            "nominal_margin_rmse_m": float(np.sqrt(np.mean(np.asarray(b_error) ** 2))),
            "state_conditioned_error_rmse_m": float(
                np.sqrt(np.mean(np.asarray(e_error) ** 2))
            ),
            "gradient_cosine_count": len(cosines),
            "gradient_cosine_mean": None if not cosines else float(np.mean(cosines)),
            "gradient_cosine_minimum": None if not cosines else float(np.min(cosines)),
            "sampled_false_safe_row_count": false_rows,
            "sampled_false_safe_candidate_count": false_candidates,
        }
    false_safe_gate = bool(
        total_false_safe_candidates
        == int(config["decision_gate"]["test_sampled_false_safe_candidate_count"])
    )
    cosine_mean = None if not all_test_cosines else float(np.mean(all_test_cosines))
    gradient_gate = bool(
        cosine_mean is not None
        and cosine_mean >= float(
            config["decision_gate"]["test_gradient_cosine_similarity_mean_minimum"]
        )
    )
    model_identity = save_affine_coefficient_model(args.model.resolve(), model_state)
    output = {
        "schema_version": AFFINE_COEFFICIENT_TRAINING_SCHEMA,
        "status": "complete", "scientific_result": False,
        "source": _git_identity(args.repo_root.resolve(), args.expected_commit),
        "config_file_sha256": config["config_file_sha256"],
        "dataset": {
            "path": str(args.dataset.resolve()),
            "file_sha256": _file_sha256(args.dataset.resolve()),
            "payload_sha256": dataset["dataset_payload_sha256"],
            "state_count": len(states),
        },
        "target_analysis": {
            "path": str(args.target_analysis.resolve()),
            "file_sha256": _file_sha256(args.target_analysis.resolve()),
            "payload_sha256": target_analysis["analysis_payload_sha256"],
        },
        "model_artifact": model_identity,
        "training": audit,
        "calibration_m": calibration.tolist(),
        "test_metrics": {
            "per_case": per_case,
            "gradient_cosine_mean": cosine_mean,
            "sampled_false_safe_row_count": total_false_safe_rows,
            "sampled_false_safe_candidate_count": total_false_safe_candidates,
            "gradient_gate_pass": gradient_gate,
            "false_safe_gate_pass": false_safe_gate,
            "learned_evaluation_authorized": bool(gradient_gate and false_safe_gate),
        },
    }
    output["training_payload_sha256"] = _hash_without(
        output, "training_payload_sha256"
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output["test_metrics"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
