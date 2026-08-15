#!/usr/bin/env python3
"""Train the 9D obstacle-relative EE endpoint L5 risk ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_l5_row01_endpoint_ablation import (
    load_endpoint_samples,
)
from scripts.train_distal_l5_row01_input_ablation import _model_config, _model_record


def load_samples(
    config: Mapping[str, Any], repo_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    import numpy as np
    from main.multilink_ellipsoid.l5_row01_endpoint_ablation import (
        payload_sha256 as endpoint_payload_sha256,
    )
    from main.multilink_ellipsoid.l5_row01_obstacle_endpoint_ablation import (
        obstacle_endpoint_feature_vector,
    )

    endpoint_source_config = dict(config)
    endpoint_source_config["endpoint"] = {
        "translation_scale_m_per_action_unit":
        config["features"]["translation_scale_m_per_action_unit"]
    }
    samples, complete_frozen, input_config = load_endpoint_samples(
        endpoint_source_config, repo_root
    )
    source = config["source"]
    endpoint_path = Path(source["frozen_endpoint_result"])
    _require(
        _file_sha256(endpoint_path) == source["frozen_endpoint_result_file_sha256"],
        "obstacle-endpoint frozen endpoint file differs",
    )
    endpoint_frozen = _load(endpoint_path)
    _require(
        endpoint_frozen["result_payload_sha256"]
        == source["frozen_endpoint_result_payload_sha256"]
        == endpoint_payload_sha256(endpoint_frozen),
        "obstacle-endpoint frozen endpoint payload differs",
    )
    cache = {}
    scale = float(config["features"]["translation_scale_m_per_action_unit"])
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
            _require(len(matches) == 1, "obstacle-endpoint candidate differs")
            candidate = matches[0]
            _require(np.array_equal(
                np.asarray(candidate["actions"], dtype=np.float64),
                np.asarray(sample["actions"], dtype=np.float64),
            ), "obstacle-endpoint action differs")
            context = result["state"]["physical_context"]
            sample["obstacle_endpoint_feature_vector"] = (
                obstacle_endpoint_feature_vector(
                    eef_position_m=context["eef_position_m"],
                    candidate_actions=candidate["actions"],
                    obstacle_center_m=context["obstacle"]["center_m"],
                    obstacle_semiaxes_m=context["obstacle"]["semiaxes_m"],
                    translation_scale_m_per_action_unit=scale,
                )
            )
    return samples, complete_frozen, endpoint_frozen


def arrays(samples: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
    import numpy as np
    return (
        np.asarray(
            [item["obstacle_endpoint_feature_vector"] for item in samples],
            dtype=np.float64,
        ),
        np.asarray([item["risk_row01"] for item in samples], dtype=np.float64),
    )


def run(
    *, repo_root: Path, config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    from main.multilink_ellipsoid.l5_row01_input_ablation import ablation_metrics
    from main.multilink_ellipsoid.l5_row01_obstacle_endpoint_ablation import (
        INPUT_DIMENSION, RESULT_SCHEMA, classify, load_config, payload_sha256,
    )
    from main.multilink_ellipsoid.l5_row01_selection import predict, train_model
    from main.multilink_ellipsoid.shadow import allocation_record

    identity = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    samples, complete_frozen, endpoint_frozen = load_samples(config, repo_root)
    train_x, train_y = arrays(samples["train"])
    validation_x, validation_y = arrays(samples["validation"])
    _require(
        train_x.shape == (43, INPUT_DIMENSION)
        and validation_x.shape == (29, INPUT_DIMENSION),
        "obstacle-endpoint arrays differ",
    )
    base_config = _model_config(complete_frozen["config"], INPUT_DIMENSION)
    base_config.update(config["matched_model"])
    bundle = train_model(
        train_x, train_y, validation_x, validation_y, base_config,
    )
    predictions = {}
    metrics = {}
    for split, features in (("train", train_x), ("validation", validation_x)):
        prediction = predict(bundle, features)
        predictions[split] = prediction.tolist()
        metrics[split] = ablation_metrics(
            prediction, samples[split], random_seed=20260814, random_draws=1024,
        )
    endpoint_metrics = endpoint_frozen["endpoint_6D"]["metrics"]
    complete_metrics = complete_frozen["arms"]["complete_physical_OSC_354D"]["metrics"]
    interpretation, comparison = classify(
        metrics["validation"], endpoint_metrics["validation"],
        minimum_relative_rmse_improvement=float(config["decision"][
            "minimum_relative_validation_RMSE_improvement_over_endpoint_6D"
        ]),
    )
    result = {
        "schema_version": RESULT_SCHEMA, "status": "complete",
        "scientific_result": True, "claim_scope": config["claim_scope"],
        "source": identity, "allocation": allocation_record(), "config": config,
        "dataset": {
            "train_label_count": len(samples["train"]),
            "grouped_validation_label_count": len(samples["validation"]),
            "labels_and_splits_identical": True,
            "reserved_test_episodes_unopened": True,
        },
        "obstacle_relative_endpoint_9D": {
            "model_config": base_config, "model": _model_record(bundle),
            "predictions": predictions, "metrics": metrics,
        },
        "frozen_endpoint_6D_metrics": endpoint_metrics,
        "frozen_complete_354D_metrics": complete_metrics,
        "comparison": comparison, "interpretation": interpretation,
        "prediction_or_control_gate_pass": False,
        "fresh_exact_replay_authorized": False,
        "QP_authorized": False, "closed_loop_authorized": False,
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
