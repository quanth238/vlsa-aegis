#!/usr/bin/env python3
"""Independently replay the matched L5 row01 input ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _git_identity, _load, _require,
)
from scripts.train_distal_l5_row01_input_ablation import (
    _arrays, _complete_arrays, _model_config, load_augmented_samples,
)


def validate(
    *, repo_root: Path, result_path: Path, config_path: Path,
    producer_commit: str, validator_commit: str,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from main.multilink_ellipsoid.l5_row01_input_ablation import (
        RESULT_SCHEMA, VALIDATION_SCHEMA, ablation_metrics, canonical,
        load_complete_bundle, load_config, payload_sha256,
    )
    from main.multilink_ellipsoid.l5_row01_selection import (
        load_frozen_bundle, predict,
    )

    _git_identity(repo_root, validator_commit)
    config = load_config(config_path)
    result = _load(result_path)
    _require(result["schema_version"] == RESULT_SCHEMA,
             "input-ablation result schema differs")
    _require(result["source"]["commit"] == producer_commit,
             "input-ablation producer differs")
    _require(result["result_payload_sha256"] == payload_sha256(result),
             "input-ablation result payload differs")
    _require(result["config"] == config, "input-ablation embedded config differs")
    samples, _frozen = load_augmented_samples(config, repo_root)
    random_seed = 20260814
    random_draws = 1024
    maximum_error = 0.0
    reports = {}
    for arm in ("compact_86D", "complete_physical_OSC_354D"):
        arm_result = result["arms"][arm]
        payload = arm_result["model"]["state_payload"]
        _require(
            arm_result["model"]["model_sha256"]
            == hashlib.sha256(canonical(payload)).hexdigest(),
            "input-ablation model hash differs",
        )
        bundle = (
            load_frozen_bundle(torch, payload, _model_config(config, 86))
            if arm == "compact_86D" else
            load_complete_bundle(torch, payload, _model_config(config, 354))
        )
        reports[arm] = {}
        for split in ("train", "validation"):
            features, _ = (
                _arrays(samples[split]) if arm == "compact_86D"
                else _complete_arrays(samples[split])
            )
            replay = predict(bundle, features)
            stored = np.asarray(arm_result["predictions"][split], dtype=np.float64)
            maximum_error = max(
                maximum_error, float(np.max(np.abs(replay - stored)))
            )
            reports[arm][split] = ablation_metrics(
                replay, samples[split], random_seed=random_seed,
                random_draws=random_draws,
            )
    _require(maximum_error <= 1.0e-9, "input-ablation replay differs")
    _require(
        reports == {name: value["metrics"] for name, value in result["arms"].items()},
        "input-ablation metrics replay differs",
    )
    output = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "validated",
        "scientific_result": True,
        "producer_commit": producer_commit,
        "validator_commit": validator_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "model_sha256_by_arm": {
            name: value["model"]["model_sha256"]
            for name, value in result["arms"].items()
        },
        "maximum_prediction_replay_error_m": maximum_error,
        "metrics": reports,
        "gates": result["gates"],
        "prediction_gate_pass": result["prediction_gate_pass"],
        "fresh_exact_replay_authorized": result["fresh_exact_replay_authorized"],
        "QP_authorized": False,
        "closed_loop_authorized": False,
        "interpretation": result["interpretation"],
    }
    output["validation_payload_sha256"] = payload_sha256(output)
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--validator-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = validate(
        repo_root=args.repo_root.resolve(), result_path=args.result.resolve(),
        config_path=args.config.resolve(), producer_commit=args.producer_commit,
        validator_commit=args.validator_commit,
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
