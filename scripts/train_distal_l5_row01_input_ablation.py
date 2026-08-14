#!/usr/bin/env python3
"""Run the frozen compact-versus-complete L5 row01 input ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_l5_row01_selection import _arrays, load_samples


def _model_config(config: Mapping[str, Any], input_dimension: int) -> dict[str, Any]:
    matched = config["matched"]
    return {
        "class": "matched_action_conditioned_two_output_MLP",
        "input_dimension": int(input_dimension),
        "output_count": 2,
        "hidden_widths": matched["hidden_widths"],
        "activation": matched["activation"],
        "seed": matched["seed"],
        "cpu_threads": matched["cpu_threads"],
        "epochs": matched["epochs"],
        "learning_rate": matched["learning_rate"],
        "weight_decay": matched["weight_decay"],
        "gradient_clip_norm": matched["gradient_clip_norm"],
        "huber_beta_normalized": matched["huber_beta_normalized"],
        "boundary_scale_m": matched["boundary_scale_m"],
        "boundary_weight_multiplier": matched["boundary_weight_multiplier"],
        "minimum_target_scale_m": matched["minimum_target_scale_m"],
    }


def load_augmented_samples(
    config: Mapping[str, Any], repo_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    import numpy as np

    from main.multilink_ellipsoid.l5_row01_input_ablation import complete_feature_vector
    from main.multilink_ellipsoid.l5_row01_selection import (
        load_config as load_row01_config, payload_sha256 as row01_payload_sha256,
    )

    source = config["source"]
    row01_path = repo_root / source["row01_config"]
    _require(_file_sha256(row01_path) == source["row01_config_file_sha256"],
             "input-ablation row01 config file differs")
    row01_config = load_row01_config(row01_path)
    samples = load_samples(row01_config)
    cache = {}
    for split in ("train", "validation"):
        for sample in samples[split]:
            path = Path(sample["source_result_path"])
            if path not in cache:
                _require(_file_sha256(path) == sample["source_result_file_sha256"],
                         "input-ablation source result file differs")
                cache[path] = _load(path)
            result = cache[path]
            _require(
                result["result_payload_sha256"]
                == sample["source_result_payload_sha256"],
                "input-ablation source result payload differs",
            )
            matches = [
                candidate for candidate in result["candidates"]
                if candidate["name"] == sample["candidate_name"]
                and int(candidate["order"]) == int(sample["candidate_order"])
            ]
            _require(len(matches) == 1, "input-ablation candidate differs")
            candidate = matches[0]
            _require(np.array_equal(
                np.asarray(candidate["actions"], dtype=np.float64),
                np.asarray(sample["actions"], dtype=np.float64),
            ), "input-ablation candidate action differs")
            sample["complete_feature_vector"] = complete_feature_vector(
                physical_context=result["state"]["physical_context"],
                nominal_actions=result["nominal_five_action_chunk"],
                candidate_actions=candidate["actions"],
                aegis=candidate["aegis_consistency"],
            )
    frozen_path = Path(source["frozen_compact_result"])
    _require(_file_sha256(frozen_path) == source["frozen_compact_result_file_sha256"],
             "input-ablation frozen compact result file differs")
    frozen = _load(frozen_path)
    _require(
        frozen["result_payload_sha256"] == source["frozen_compact_result_payload_sha256"]
        and frozen["result_payload_sha256"] == row01_payload_sha256(frozen),
        "input-ablation frozen compact payload differs",
    )
    return samples, frozen


def _complete_arrays(samples: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
    import numpy as np
    return (
        np.asarray(
            [sample["complete_feature_vector"] for sample in samples],
            dtype=np.float64,
        ),
        np.asarray([sample["risk_row01"] for sample in samples], dtype=np.float64),
    )


def _model_record(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in bundle.items()
        if key not in {
            "model", "device", "feature_mean", "feature_scale",
            "target_mean", "target_scale",
        }
    }


def run(
    *, repo_root: Path, config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.l5_row01_input_ablation import (
        RESULT_SCHEMA, ablation_metrics, load_config, payload_sha256,
        train_complete_model,
    )
    from main.multilink_ellipsoid.l5_row01_selection import predict, train_model
    from main.multilink_ellipsoid.shadow import allocation_record

    source = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    samples, frozen = load_augmented_samples(config, repo_root)
    compact_train_x, train_y = _arrays(samples["train"])
    compact_validation_x, validation_y = _arrays(samples["validation"])
    complete_train_x, complete_train_y = _complete_arrays(samples["train"])
    complete_validation_x, complete_validation_y = _complete_arrays(
        samples["validation"]
    )
    _require(
        np.array_equal(train_y, complete_train_y)
        and np.array_equal(validation_y, complete_validation_y),
        "input-ablation labels differ between arms",
    )
    compact_bundle = train_model(
        compact_train_x, train_y, compact_validation_x, validation_y,
        _model_config(config, 86),
    )
    complete_bundle = train_complete_model(
        complete_train_x, train_y, complete_validation_x, validation_y,
        _model_config(config, 354),
    )
    random_seed = int(frozen["config"]["selection"]["random_seed"])
    random_draws = int(frozen["config"]["selection"]["random_draws_per_state"])
    arms = {}
    maximum_compact_reproduction_error = 0.0
    for name, bundle, arrays in (
        ("compact_86D", compact_bundle, (compact_train_x, compact_validation_x)),
        ("complete_physical_OSC_354D", complete_bundle,
         (complete_train_x, complete_validation_x)),
    ):
        predictions = {}
        metrics = {}
        for split, features in zip(("train", "validation"), arrays):
            prediction = predict(bundle, features)
            predictions[split] = prediction.tolist()
            metrics[split] = ablation_metrics(
                prediction, samples[split], random_seed=random_seed,
                random_draws=random_draws,
            )
            if name == "compact_86D":
                stored = np.asarray(frozen["predictions"][split], dtype=np.float64)
                maximum_compact_reproduction_error = max(
                    maximum_compact_reproduction_error,
                    float(np.max(np.abs(prediction - stored))),
                )
        arms[name] = {
            "input_dimension": int(arrays[0].shape[1]),
            "model": _model_record(bundle),
            "predictions": predictions,
            "metrics": metrics,
        }
    _require(
        maximum_compact_reproduction_error
        <= float(config["matched"]["compact_reproduction_maximum_error_m"]),
        "input-ablation compact reproduction differs",
    )
    compact = arms["compact_86D"]["metrics"]["validation"]
    complete = arms["complete_physical_OSC_354D"]["metrics"]["validation"]
    gates = {
        "compact_arm_exactly_reproduces_frozen_result":
        maximum_compact_reproduction_error <= 1.0e-9,
        "zero_complete_input_row01_false_safes":
        complete["row01_false_safe_count"] == 0,
        "complete_input_safe_support_all_four_recoverable_states":
        complete["supported_recoverable_state_count"]
        == complete["recoverable_state_count"] == 4,
        "all_complete_input_selected_candidates_exact_all_seven_safe":
        complete["all_selected_exact_all_seven_safe"],
        "complete_input_selected_safe_rate_better_than_seeded_random":
        complete["selected_exact_safe_rate"]
        > complete["seeded_random_exact_safe_rate"],
        "complete_input_validation_RMSE_below_compact":
        complete["rmse_m"] < compact["rmse_m"],
        "complete_input_near_boundary_RMSE_at_most_1mm":
        complete["near_boundary_rmse_m"] is not None
        and complete["near_boundary_rmse_m"]
        <= float(config["gates"]["complete_input_near_boundary_RMSE_maximum_m"]),
        "reserved_test_episodes_unopened": True,
    }
    prediction_gate_pass = all(gates.values())
    complete_train = arms["complete_physical_OSC_354D"]["metrics"]["train"]
    compact_train = arms["compact_86D"]["metrics"]["train"]
    if prediction_gate_pass:
        interpretation = "complete_input_passes_omitted_variable_hypothesis_supported"
    elif (
        complete["rmse_m"] < compact["rmse_m"]
        and complete["row01_false_safe_count"] > 0
    ):
        interpretation = "complete_input_improves_but_false_safes_remain"
    elif complete_train["rmse_m"] <= 0.001 and compact_train["rmse_m"] <= 0.001:
        interpretation = "both_fit_training_but_fail_grouped_validation"
    else:
        interpretation = "complete_input_fails_training_target_or_representation"
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "source": source,
        "allocation": allocation_record(),
        "config": config,
        "dataset": {
            "sample_counts": {split: len(samples[split]) for split in ("train", "validation")},
            "labels_bitwise_identical_between_arms": True,
            "episode_splits_identical_between_arms": True,
            "candidate_actions_identical_between_arms": True,
            "reserved_test_episodes_unopened": True,
        },
        "arms": arms,
        "maximum_compact_reproduction_error_m": maximum_compact_reproduction_error,
        "gates": gates,
        "prediction_gate_pass": prediction_gate_pass,
        "fresh_exact_replay_authorized": prediction_gate_pass,
        "QP_authorized": False,
        "closed_loop_authorized": False,
        "interpretation": interpretation,
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
        "gates": result["gates"],
        "compact_validation": result["arms"]["compact_86D"]["metrics"]["validation"],
        "complete_validation": result["arms"]["complete_physical_OSC_354D"]["metrics"]["validation"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
