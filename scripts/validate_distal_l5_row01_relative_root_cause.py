#!/usr/bin/env python3
"""Independently replay the frozen relative-representation audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def validate(
    repo_root: Path, config_path: Path, result_path: Path,
    producer_commit: str, validator_commit: str,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from main.multilink_ellipsoid.l5_row01_input_ablation import ablation_metrics
    from main.multilink_ellipsoid.l5_row01_relative_root_cause import (
        VALIDATION_SCHEMA, knn_predict, load_config, load_relative_bundle,
        payload_sha256, ridge_predict, support_audit,
    )
    from main.multilink_ellipsoid.l5_row01_selection import predict
    from scripts.train_distal_l5_row01_relative_root_cause import arrays, load_samples

    config = load_config(config_path)
    result = _load(result_path)
    _require(result["source"]["commit"] == producer_commit,
             "relative validation producer commit differs")
    _require(result["result_payload_sha256"] == payload_sha256(result),
             "relative validation result payload differs")
    samples, _ = load_samples(config, repo_root)
    train_x, train_y = arrays(samples["train"])
    validation_x, validation_y = arrays(samples["validation"])
    bundle = load_relative_bundle(
        torch, result["relative_model"]["state_payload"], config["matched_MLP"]
    )
    replay = {
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
    maximum_error = 0.0
    reports = {}
    for name, splits in replay.items():
        reports[name] = {}
        for split, values in splits.items():
            stored = np.asarray(result["predictions"][name][split], dtype=np.float64)
            maximum_error = max(maximum_error, float(np.max(np.abs(values - stored))))
            reports[name][split] = ablation_metrics(
                values, samples[split], random_seed=20260814, random_draws=1024
            )
    _require(maximum_error <= 1.0e-9, "relative prediction replay differs")
    _require(reports == result["reports"], "relative metric replay differs")
    audit = support_audit(
        train_x, train_y, samples["train"], validation_x, validation_y,
        samples["validation"],
    )
    _require(audit == result["support_audit"], "relative support audit differs")
    validation = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "passing",
        "producer_commit": producer_commit,
        "validator_commit": validator_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "maximum_prediction_replay_error_m": maximum_error,
        "reports": reports,
        "support_audit": audit,
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
