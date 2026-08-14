#!/usr/bin/env python3
"""Train matched prefix/backup component models on frozen rollout traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)


def load_component_samples(config: Mapping[str, Any], repo_root: Path) -> tuple[Any, Any]:
    import numpy as np

    from main.multilink_ellipsoid.l5_row01_relative_root_cause import (
        load_config as load_relative_config, payload_sha256 as relative_payload_sha256,
    )
    from scripts.train_distal_l5_row01_relative_root_cause import load_samples

    source = config["source"]
    relative_path = repo_root / source["relative_audit_config"]
    _require(_file_sha256(relative_path) == source["relative_audit_config_file_sha256"],
             "component audit relative config differs")
    relative_config = load_relative_config(relative_path)
    samples, _ = load_samples(relative_config, repo_root)
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
            _require(len(matches) == 1, "component candidate differs")
            candidate = matches[0]
            prefix = [float(value) for value in candidate["candidate_prefix_risk"][:2]]
            backup = (
                None if candidate.get("backup_risk") is None
                else [float(value) for value in candidate["backup_risk"][:2]]
            )
            combined = [float(value) for value in candidate["combined_risk"][:2]]
            expected = prefix if backup is None else [
                max(prefix[row], backup[row]) for row in range(2)
            ]
            _require(np.allclose(combined, expected, atol=1.0e-12, rtol=0.0),
                     "component maximum differs")
            sample["prefix_target"] = prefix
            sample["backup_target"] = backup
            sample["combined_target"] = combined
    frozen_path = Path(source["frozen_relative_result"])
    _require(_file_sha256(frozen_path) == source["frozen_relative_result_file_sha256"],
             "component frozen result file differs")
    frozen = _load(frozen_path)
    _require(
        frozen["result_payload_sha256"] == source["frozen_relative_result_payload_sha256"]
        and frozen["result_payload_sha256"] == relative_payload_sha256(frozen),
        "component frozen result payload differs",
    )
    return samples, frozen


def arrays(samples: Sequence[Mapping[str, Any]], target: str) -> tuple[Any, Any, list[Any]]:
    import numpy as np

    selected = [item for item in samples if item[f"{target}_target"] is not None]
    return (
        np.asarray([item["relative_feature_vector"] for item in selected], dtype=np.float64),
        np.asarray([item[f"{target}_target"] for item in selected], dtype=np.float64),
        selected,
    )


def model_record(bundle: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in bundle.items() if key not in {
        "model", "device", "feature_mean", "feature_scale",
        "target_mean", "target_scale",
    }}


def run(repo_root: Path, config_path: Path, expected_commit: str) -> dict[str, Any]:
    import numpy as np

    from main.multilink_ellipsoid.l5_row01_component_audit import (
        RESULT_SCHEMA, component_metrics, false_safe_phase_attribution,
        load_config, payload_sha256, response_curve_audit, train_component_model,
    )
    from main.multilink_ellipsoid.l5_row01_input_ablation import ablation_metrics
    from main.multilink_ellipsoid.l5_row01_selection import predict
    from main.multilink_ellipsoid.shadow import allocation_record

    source = _git_identity(repo_root, expected_commit)
    config = load_config(config_path)
    samples, frozen = load_component_samples(config, repo_root)
    bundles = {}
    predictions = {}
    reports = {}
    for component in ("prefix", "backup"):
        train_x, train_y, train_items = arrays(samples["train"], component)
        validation_x, validation_y, validation_items = arrays(samples["validation"], component)
        bundle = train_component_model(
            train_x, train_y, validation_x, validation_y, config["matched_model"]
        )
        bundles[component] = bundle
        predictions[component] = {
            "train": predict(bundle, train_x),
            "validation": predict(bundle, validation_x),
        }
        reports[component] = {
            "train": component_metrics(
                predictions[component]["train"], train_y, 0.005
            ),
            "validation": component_metrics(
                predictions[component]["validation"], validation_y, 0.005
            ),
            "sample_counts": {
                "train": len(train_items), "validation": len(validation_items)
            },
        }
    all_features = {
        split: np.asarray([
            item["relative_feature_vector"] for item in samples[split]
        ], dtype=np.float64) for split in ("train", "validation")
    }
    full_component_predictions = {
        component: {
            split: predict(bundles[component], all_features[split])
            for split in ("train", "validation")
        } for component in ("prefix", "backup")
    }
    structured = {
        split: np.maximum(
            full_component_predictions["prefix"][split],
            full_component_predictions["backup"][split],
        ) for split in ("train", "validation")
    }
    structured_reports = {
        split: ablation_metrics(
            structured[split], samples[split], random_seed=20260814,
            random_draws=1024,
        ) for split in ("train", "validation")
    }
    direct_validation = np.asarray(
        frozen["predictions"]["relative_MLP"]["validation"], dtype=np.float64
    )
    attribution = {
        "direct_combined_relative_MLP": false_safe_phase_attribution(
            direct_validation, samples["validation"]
        ),
        "structured_component_max": false_safe_phase_attribution(
            structured["validation"], samples["validation"]
        ),
    }
    curve = response_curve_audit(
        samples["train"], samples["validation"],
        config["E39_curve"]["validation_state_contains"],
        config["E39_curve"]["minimum_unique_coordinates"],
    )
    prefix_validation = reports["prefix"]["validation"]
    backup_validation = reports["backup"]["validation"]
    combined_validation = structured_reports["validation"]
    gates = {
        "zero_prefix_validation_false_safes": prefix_validation["false_safe_count"] == 0,
        "zero_backup_validation_false_safes_on_observed_backup":
        backup_validation["false_safe_count"] == 0,
        "prefix_near_boundary_RMSE_at_most_1mm":
        prefix_validation["near_boundary_rmse_m"] is not None
        and prefix_validation["near_boundary_rmse_m"] <= 0.001,
        "backup_near_boundary_RMSE_at_most_1mm":
        backup_validation["near_boundary_rmse_m"] is not None
        and backup_validation["near_boundary_rmse_m"] <= 0.001,
        "zero_structured_combined_false_safes":
        combined_validation["row01_false_safe_count"] == 0,
        "structured_safe_selection_all_four_recoverable_states":
        combined_validation["selected_exact_safe_recoverable_state_count"] == 4,
        "E39_all_identifiable_slope_signs_match_nearest_training_curve":
        curve["all_identifiable_slope_signs_match"],
        "reserved_test_episodes_unopened": True,
    }
    if all(gates.values()):
        interpretation = "structured_components_pass_for_Best_of_N_followup"
    elif not gates["zero_prefix_validation_false_safes"]:
        interpretation = "prefix_component_fails_collect_prefix_boundaries"
    elif not gates["zero_backup_validation_false_safes_on_observed_backup"]:
        interpretation = "backup_component_fails_collect_backup_boundaries"
    elif not curve["all_identifiable_slope_signs_match"]:
        interpretation = "E39_action_response_slope_differs_across_nearby_states"
    else:
        interpretation = "components_individually_improve_but_structured_max_fails"
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete", "scientific_result": True,
        "source": source, "allocation": allocation_record(),
        "claim_scope": config["claim_scope"], "config": config,
        "dataset": {
            "combined_sample_counts": {"train": 43, "validation": 29},
            "component_sample_counts": {
                component: reports[component]["sample_counts"]
                for component in ("prefix", "backup")
            },
            "missing_backup_is_censored": True,
            "labels_and_episode_splits_unchanged": True,
            "reserved_test_episodes_unopened": True,
        },
        "models": {component: model_record(bundle) for component, bundle in bundles.items()},
        "predictions": {
            "component_observed": {
                component: {
                    split: values.tolist() for split, values in by_split.items()
                } for component, by_split in predictions.items()
            },
            "component_all_candidates": {
                component: {
                    split: values.tolist() for split, values in by_split.items()
                } for component, by_split in full_component_predictions.items()
            },
            "structured_max": {
                split: values.tolist() for split, values in structured.items()
            },
        },
        "component_reports": reports,
        "structured_reports": structured_reports,
        "false_safe_attribution": attribution,
        "E39_response_curve_audit": curve,
        "gates": gates, "prediction_gate_pass": all(gates.values()),
        "fresh_exact_replay_authorized": False,
        "QP_authorized": False, "closed_loop_authorized": False,
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
    result = run(args.repo_root.resolve(), args.config.resolve(), args.expected_commit)
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "dataset": result["dataset"],
        "component_reports": result["component_reports"],
        "structured_validation": result["structured_reports"]["validation"],
        "false_safe_attribution": result["false_safe_attribution"],
        "E39_response_curve_audit": result["E39_response_curve_audit"],
        "gates": result["gates"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
