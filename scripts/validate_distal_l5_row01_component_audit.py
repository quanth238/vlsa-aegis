#!/usr/bin/env python3
"""Independent replay for the prefix/backup component audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def validate(repo_root: Path, config_path: Path, result_path: Path,
             producer_commit: str, validator_commit: str) -> dict[str, Any]:
    import numpy as np
    import torch

    from main.multilink_ellipsoid.l5_row01_component_audit import (
        VALIDATION_SCHEMA, component_metrics, false_safe_phase_attribution,
        load_config, payload_sha256, response_curve_audit,
    )
    from main.multilink_ellipsoid.l5_row01_input_ablation import ablation_metrics
    from main.multilink_ellipsoid.l5_row01_relative_root_cause import load_relative_bundle
    from main.multilink_ellipsoid.l5_row01_selection import predict
    from scripts.train_distal_l5_row01_component_audit import (
        arrays, load_component_samples,
    )

    config = load_config(config_path)
    result = _load(result_path)
    _require(result["source"]["commit"] == producer_commit,
             "component validator producer commit differs")
    _require(result["result_payload_sha256"] == payload_sha256(result),
             "component validator result payload differs")
    samples, frozen = load_component_samples(config, repo_root)
    bundles = {
        component: load_relative_bundle(
            torch, result["models"][component]["state_payload"], config["matched_model"]
        ) for component in ("prefix", "backup")
    }
    maximum_error = 0.0
    component_reports = {}
    full_predictions = {}
    for component in ("prefix", "backup"):
        component_reports[component] = {}
        full_predictions[component] = {}
        for split in ("train", "validation"):
            x, y, items = arrays(samples[split], component)
            values = predict(bundles[component], x)
            stored = np.asarray(
                result["predictions"]["component_observed"][component][split],
                dtype=np.float64,
            )
            maximum_error = max(maximum_error, float(np.max(np.abs(values - stored))))
            component_reports[component][split] = component_metrics(values, y, 0.005)
            all_x = np.asarray([
                item["relative_feature_vector"] for item in samples[split]
            ], dtype=np.float64)
            full_predictions[component][split] = predict(bundles[component], all_x)
            stored_full = np.asarray(
                result["predictions"]["component_all_candidates"][component][split],
                dtype=np.float64,
            )
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(full_predictions[component][split] - stored_full))),
            )
        component_reports[component]["sample_counts"] = result["component_reports"][component]["sample_counts"]
    structured = {
        split: np.maximum(
            full_predictions["prefix"][split], full_predictions["backup"][split]
        ) for split in ("train", "validation")
    }
    structured_reports = {
        split: ablation_metrics(
            structured[split], samples[split], random_seed=20260814, random_draws=1024
        ) for split in ("train", "validation")
    }
    attribution = {
        "direct_combined_relative_MLP": false_safe_phase_attribution(
            np.asarray(frozen["predictions"]["relative_MLP"]["validation"]),
            samples["validation"],
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
    _require(maximum_error <= 1.0e-9, "component prediction replay differs")
    _require(component_reports == result["component_reports"],
             "component report replay differs")
    _require(structured_reports == result["structured_reports"],
             "structured report replay differs")
    _require(attribution == result["false_safe_attribution"],
             "component attribution replay differs")
    _require(curve == result["E39_response_curve_audit"],
             "response curve replay differs")
    validation = {
        "schema_version": VALIDATION_SCHEMA, "status": "passing",
        "producer_commit": producer_commit, "validator_commit": validator_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "maximum_prediction_replay_error_m": maximum_error,
        "component_reports": component_reports,
        "structured_reports": structured_reports,
        "false_safe_attribution": attribution,
        "E39_response_curve_audit": curve,
        "interpretation": result["interpretation"],
    }
    validation["validation_payload_sha256"] = payload_sha256(validation)
    return validation


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(
        args.repo_root.resolve(), args.config.resolve(), args.result.resolve(),
        args.producer_commit, args.validator_commit,
    )
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({
        "interpretation": output["interpretation"],
        "maximum_prediction_replay_error_m": output["maximum_prediction_replay_error_m"],
        "validation_payload_sha256": output["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
