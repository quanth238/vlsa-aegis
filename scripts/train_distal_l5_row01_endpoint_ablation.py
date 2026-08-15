#!/usr/bin/env python3
"""Train the six-dimensional EE start/end L5 row-0/1 risk ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_l5_row01_input_ablation import (
    _model_config, _model_record, load_augmented_samples,
)


def load_endpoint_samples(
    config: Mapping[str, Any], repo_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    import numpy as np

    from main.multilink_ellipsoid.l5_row01_endpoint_ablation import (
        endpoint_feature_vector,
    )
    from main.multilink_ellipsoid.l5_row01_input_ablation import (
        load_config as load_input_config, payload_sha256 as input_payload_sha256,
    )

    source = config["source"]
    input_path = repo_root / source["input_ablation_config"]
    _require(
        _file_sha256(input_path) == source["input_ablation_config_file_sha256"],
        "endpoint source config differs",
    )
    input_config = load_input_config(input_path)
    samples, _ = load_augmented_samples(input_config, repo_root)
    cache = {}
    scale = float(config["endpoint"]["translation_scale_m_per_action_unit"])
    for split in ("train", "validation"):
        for sample in samples[split]:
            path = Path(sample["source_result_path"])
            if path not in cache:
                cache[path] = _load(path)
            result = cache[path]
            matches = [
                candidate for candidate in result["candidates"]
                if candidate["name"] == sample["candidate_name"]
                and int(candidate["order"]) == int(sample["candidate_order"])
            ]
            _require(len(matches) == 1, "endpoint candidate differs")
            candidate = matches[0]
            _require(np.array_equal(
                np.asarray(candidate["actions"], dtype=np.float64),
                np.asarray(sample["actions"], dtype=np.float64),
            ), "endpoint candidate actions differ")
            sample["endpoint_feature_vector"] = endpoint_feature_vector(
                eef_position_m=result["state"]["physical_context"]["eef_position_m"],
                candidate_actions=candidate["actions"],
                translation_scale_m_per_action_unit=scale,
            )
    frozen_path = Path(source["frozen_input_ablation_result"])
    _require(
        _file_sha256(frozen_path)
        == source["frozen_input_ablation_result_file_sha256"],
        "endpoint frozen result file differs",
    )
    frozen = _load(frozen_path)
    _require(
        frozen["result_payload_sha256"]
        == source["frozen_input_ablation_result_payload_sha256"]
        == input_payload_sha256(frozen),
        "endpoint frozen result payload differs",
    )
    return samples, frozen, input_config


def endpoint_arrays(samples: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
    import numpy as np
    return (
        np.asarray(
            [sample["endpoint_feature_vector"] for sample in samples],
            dtype=np.float64,
        ),
        np.asarray([sample["risk_row01"] for sample in samples], dtype=np.float64),
    )


def run(
    *, repo_root: Path, config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.l5_row01_endpoint_ablation import (
        ENDPOINT_DIMENSION, RESULT_SCHEMA, classify, load_config, payload_sha256,
    )
    from main.multilink_ellipsoid.l5_row01_input_ablation import ablation_metrics
    from main.multilink_ellipsoid.l5_row01_selection import predict, train_model
    from main.multilink_ellipsoid.shadow import allocation_record

    source_identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    samples, frozen, input_config = load_endpoint_samples(config, repo_root)
    train_x, train_y = endpoint_arrays(samples["train"])
    validation_x, validation_y = endpoint_arrays(samples["validation"])
    _require(
        train_x.shape == (43, ENDPOINT_DIMENSION)
        and validation_x.shape == (29, ENDPOINT_DIMENSION),
        "endpoint arrays differ",
    )
    model_config = _model_config(input_config, ENDPOINT_DIMENSION)
    model_config.update(config["matched_model"])
    bundle = train_model(
        train_x, train_y, validation_x, validation_y, model_config,
    )
    predictions = {}
    metrics = {}
    for split, features in (("train", train_x), ("validation", validation_x)):
        prediction = predict(bundle, features)
        predictions[split] = prediction.tolist()
        metrics[split] = ablation_metrics(
            prediction, samples[split], random_seed=20260814, random_draws=1024,
        )
    complete_metrics = frozen["arms"]["complete_physical_OSC_354D"]["metrics"]
    interpretation, comparison = classify(
        metrics["validation"], complete_metrics["validation"],
        minimum_relative_rmse_improvement=float(config["decision"][
            "minimum_relative_validation_RMSE_improvement_over_complete_354D"
        ]),
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
            "train_label_count": len(samples["train"]),
            "grouped_validation_label_count": len(samples["validation"]),
            "labels_and_episode_splits_identical_to_frozen_354D": True,
            "reserved_test_episodes_unopened": True,
        },
        "endpoint_6D": {
            "model_config": model_config,
            "model": _model_record(bundle),
            "predictions": predictions,
            "metrics": metrics,
        },
        "frozen_complete_354D_metrics": complete_metrics,
        "comparison": comparison,
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
