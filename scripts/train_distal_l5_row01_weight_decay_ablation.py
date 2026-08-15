#!/usr/bin/env python3
"""Run the frozen 354D L5 row-0/1 weight-decay-only ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_l5_row01_input_ablation import (
    _complete_arrays, _model_config, _model_record, load_augmented_samples,
)


def _source(
    config: Mapping[str, Any], repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from main.multilink_ellipsoid.l5_row01_input_ablation import (
        load_config as load_input_config, payload_sha256 as input_payload_sha256,
    )

    source = config["source"]
    input_path = repo_root / source["input_ablation_config"]
    _require(
        _file_sha256(input_path) == source["input_ablation_config_file_sha256"],
        "weight-decay source config differs",
    )
    input_config = load_input_config(input_path)
    samples, _ = load_augmented_samples(input_config, repo_root)
    frozen_path = Path(source["frozen_input_ablation_result"])
    _require(
        _file_sha256(frozen_path)
        == source["frozen_input_ablation_result_file_sha256"],
        "weight-decay frozen result file differs",
    )
    frozen = _load(frozen_path)
    _require(
        frozen["result_payload_sha256"]
        == source["frozen_input_ablation_result_payload_sha256"]
        == input_payload_sha256(frozen),
        "weight-decay frozen result payload differs",
    )
    return input_config, samples, frozen


def run(
    *, repo_root: Path, config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.l5_row01_input_ablation import (
        ablation_metrics, train_complete_model,
    )
    from main.multilink_ellipsoid.l5_row01_selection import predict
    from main.multilink_ellipsoid.l5_row01_weight_decay_ablation import (
        RESULT_SCHEMA, classify_effect, load_config, payload_sha256,
    )
    from main.multilink_ellipsoid.shadow import allocation_record

    source_identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    input_config, samples, frozen = _source(config, repo_root)
    train_x, train_y = _complete_arrays(samples["train"])
    validation_x, validation_y = _complete_arrays(samples["validation"])
    base_model_config = _model_config(input_config, 354)
    # Selection metrics in the frozen experiment were preregistered at these values.
    random_seed = 20260814
    random_draws = 1024
    arms = {}
    maximum_established_reproduction_error = 0.0
    frozen_predictions = {
        split: np.asarray(
            frozen["arms"]["complete_physical_OSC_354D"]["predictions"][split],
            dtype=np.float64,
        )
        for split in ("train", "validation")
    }
    for name in ("no_weight_decay", "established_weight_decay"):
        model_config = dict(base_model_config)
        model_config["weight_decay"] = float(config["arms"][name]["weight_decay"])
        bundle = train_complete_model(
            train_x, train_y, validation_x, validation_y, model_config,
        )
        predictions = {}
        metrics = {}
        for split, features in (
            ("train", train_x), ("validation", validation_x),
        ):
            prediction = predict(bundle, features)
            predictions[split] = prediction.tolist()
            metrics[split] = ablation_metrics(
                prediction, samples[split], random_seed=random_seed,
                random_draws=random_draws,
            )
            if name == "established_weight_decay":
                maximum_established_reproduction_error = max(
                    maximum_established_reproduction_error,
                    float(np.max(np.abs(prediction - frozen_predictions[split]))),
                )
        arms[name] = {
            "weight_decay": model_config["weight_decay"],
            "model_config": model_config,
            "model": _model_record(bundle),
            "predictions": predictions,
            "metrics": metrics,
        }
    _require(
        maximum_established_reproduction_error
        <= float(config["matched"][
            "established_prediction_reproduction_maximum_error_m"
        ]),
        "established weight-decay arm does not reproduce the frozen result",
    )
    interpretation, deltas = classify_effect(
        arms["no_weight_decay"]["metrics"]["validation"],
        arms["established_weight_decay"]["metrics"]["validation"],
        minimum_relative_validation_improvement=float(
            config["decision"]["minimum_relative_validation_RMSE_improvement"]
        ),
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": source_identity,
        "allocation": allocation_record(),
        "config": config,
        "dataset": {
            "train_label_count": int(train_y.shape[0]),
            "grouped_validation_label_count": int(validation_y.shape[0]),
            "same_labels_splits_inputs_architecture_loss_seed_optimizer_schedule": True,
            "only_weight_decay_differs": True,
            "reserved_test_episodes_unopened": True,
        },
        "arms": arms,
        "maximum_established_prediction_reproduction_error_m":
        maximum_established_reproduction_error,
        "comparison": deltas,
        "interpretation": interpretation,
        "prediction_or_control_gate_pass": False,
        "fresh_exact_replay_authorized": False,
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
    result = run(
        repo_root=args.repo_root.resolve(), config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "comparison": result["comparison"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
