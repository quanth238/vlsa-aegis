#!/usr/bin/env python3
"""Run the frozen-data relative-representation root-cause audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def load_samples(config: Mapping[str, Any], repo_root: Path) -> tuple[Any, Any]:
    import numpy as np

    from main.multilink_ellipsoid.l5_row01_input_ablation import (
        load_config as load_input_config, payload_sha256 as input_payload_sha256,
    )
    from main.multilink_ellipsoid.l5_row01_relative_root_cause import (
        relative_feature_vector, witness_phase,
    )
    from scripts.train_distal_l5_row01_input_ablation import load_augmented_samples

    source = config["source"]
    input_path = repo_root / source["input_ablation_config"]
    _require(_file_sha256(input_path) == source["input_ablation_config_file_sha256"],
             "relative audit input-ablation config differs")
    input_config = load_input_config(input_path)
    samples, _ = load_augmented_samples(input_config, repo_root)
    cache = {}
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
            _require(len(matches) == 1, "relative audit candidate differs")
            candidate = matches[0]
            _require(np.array_equal(
                np.asarray(candidate["actions"], dtype=np.float64),
                np.asarray(sample["actions"], dtype=np.float64),
            ), "relative audit candidate action differs")
            sample["relative_feature_vector"] = relative_feature_vector(
                physical_context=result["state"]["physical_context"],
                nominal_actions=result["nominal_five_action_chunk"],
                candidate_actions=candidate["actions"],
            )
            sample["witness_phase"] = witness_phase(candidate)
    frozen_path = Path(source["frozen_input_ablation_result"])
    _require(_file_sha256(frozen_path) == source["frozen_input_ablation_result_file_sha256"],
             "relative audit frozen result file differs")
    frozen = _load(frozen_path)
    _require(
        frozen["result_payload_sha256"]
        == source["frozen_input_ablation_result_payload_sha256"]
        and frozen["result_payload_sha256"] == input_payload_sha256(frozen),
        "relative audit frozen result payload differs",
    )
    return samples, frozen


def arrays(samples: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
    import numpy as np
    return (
        np.asarray([item["relative_feature_vector"] for item in samples], dtype=np.float64),
        np.asarray([item["risk_row01"] for item in samples], dtype=np.float64),
    )


def _model_record(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in bundle.items() if key not in {
        "model", "device", "feature_mean", "feature_scale",
        "target_mean", "target_scale",
    }}


def run(repo_root: Path, config_path: Path, expected_commit: str) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.l5_row01_input_ablation import ablation_metrics
    from main.multilink_ellipsoid.l5_row01_relative_root_cause import (
        RESULT_SCHEMA, knn_predict, load_config, payload_sha256, ridge_predict,
        support_audit,
    )
    from main.multilink_ellipsoid.l5_row01_selection import predict, train_model
    from main.multilink_ellipsoid.shadow import allocation_record

    source = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    samples, frozen = load_samples(config, repo_root)
    train_x, train_y = arrays(samples["train"])
    validation_x, validation_y = arrays(samples["validation"])
    _require(train_x.shape == (43, 134) and validation_x.shape == (29, 134),
             "relative audit arrays differ")
    model_config = dict(config["matched_MLP"])
    bundle = train_model(train_x, train_y, validation_x, validation_y, model_config)
    random_seed = 20260814
    random_draws = 1024
    predictions = {
        "relative_MLP": {
            "train": predict(bundle, train_x),
            "validation": predict(bundle, validation_x),
        },
        "relative_KNN": {
            "validation": knn_predict(
                train_x, train_y, validation_x, config["baselines"]["KNN"]["k"]
            ),
        },
        "relative_ridge": {
            "validation": ridge_predict(
                train_x, train_y, validation_x,
                config["baselines"]["ridge"]["lambda"],
            ),
        },
    }
    reports = {}
    for name, split_predictions in predictions.items():
        reports[name] = {}
        for split, values in split_predictions.items():
            reports[name][split] = ablation_metrics(
                values, samples[split], random_seed=random_seed,
                random_draws=random_draws,
            )
    relative_validation = reports["relative_MLP"]["validation"]
    complete_validation = frozen["arms"]["complete_physical_OSC_354D"]["metrics"]["validation"]
    gates = {
        "zero_relative_false_safes": relative_validation["row01_false_safe_count"] == 0,
        "relative_safe_support_all_four_recoverable_states":
        relative_validation["supported_recoverable_state_count"]
        == relative_validation["recoverable_state_count"] == 4,
        "relative_exact_safe_selection_all_four":
        relative_validation["selected_exact_safe_recoverable_state_count"] == 4,
        "relative_near_boundary_RMSE_at_most_1mm":
        relative_validation["near_boundary_rmse_m"] is not None
        and relative_validation["near_boundary_rmse_m"] <= 0.001,
        "relative_validation_RMSE_below_complete_354D":
        relative_validation["rmse_m"] < complete_validation["rmse_m"],
        "reserved_test_episodes_unopened": True,
    }
    audit = support_audit(
        train_x, train_y, samples["train"], validation_x, validation_y,
        samples["validation"],
    )
    if all(gates.values()):
        interpretation = "relative_representation_passes_strict_gate"
    elif (
        relative_validation["rmse_m"] < complete_validation["rmse_m"]
        and relative_validation["row01_false_safe_count"]
        < complete_validation["row01_false_safe_count"]
    ):
        interpretation = "relative_representation_improves_but_fails_safety"
    elif reports["relative_MLP"]["train"]["rmse_m"] <= 0.001:
        interpretation = "relative_fits_training_but_grouped_state_support_fails"
    else:
        interpretation = "relative_feature_or_target_representation_fails_training"
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "source": source,
        "allocation": allocation_record(),
        "claim_scope": config["claim_scope"],
        "config": config,
        "dataset": {
            "sample_counts": {"train": 43, "validation": 29},
            "unique_state_counts": {
                split: len({item["state_id"] for item in samples[split]})
                for split in ("train", "validation")
            },
            "labels_and_splits_unchanged": True,
            "reserved_test_episodes_unopened": True,
        },
        "frozen_complete_354D_validation": complete_validation,
        "relative_model": _model_record(bundle),
        "predictions": {
            name: {split: values.tolist() for split, values in splits.items()}
            for name, splits in predictions.items()
        },
        "reports": reports,
        "support_audit": audit,
        "gates": gates,
        "prediction_gate_pass": all(gates.values()),
        "fresh_exact_replay_authorized": False,
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
        args.repo_root.resolve(), args.config.resolve(), args.expected_commit
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "relative_MLP_train": result["reports"]["relative_MLP"]["train"],
        "relative_MLP_validation": result["reports"]["relative_MLP"]["validation"],
        "relative_KNN_validation": result["reports"]["relative_KNN"]["validation"],
        "relative_ridge_validation": result["reports"]["relative_ridge"]["validation"],
        "gates": result["gates"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
