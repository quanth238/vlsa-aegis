#!/usr/bin/env python3
"""Independent replay for the exact-anchor L5 action-response diagnostic."""

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

    from main.multilink_ellipsoid.l5_row01_oracle_anchor_delta import (
        VALIDATION_SCHEMA, load_bundle, load_config, payload_sha256, predict,
        response_metrics,
    )
    from scripts.train_distal_l5_row01_oracle_anchor_delta import (
        arrays, load_anchored_samples,
    )

    config = load_config(config_path)
    result = _load(result_path)
    _require(result["source"]["commit"] == producer_commit,
             "oracle-anchor validator producer commit differs")
    _require(result["result_payload_sha256"] == payload_sha256(result),
             "oracle-anchor result payload differs")
    samples, _ = load_anchored_samples(config, repo_root)
    bundle = load_bundle(torch, result["model"]["state_payload"], config["model"])
    reports = {}
    maximum_error = 0.0
    zero_error = 0.0
    for split in ("train", "validation"):
        context, residual, _, _ = arrays(samples[split])
        values = predict(bundle, context, residual)
        stored = np.asarray(result["predictions"][split], dtype=np.float64)
        maximum_error = max(maximum_error, float(np.max(np.abs(values - stored))))
        zero = predict(bundle, context, np.zeros_like(residual))
        zero_error = max(zero_error, float(np.max(np.abs(zero))))
        reports[split] = response_metrics(
            values, samples[split],
            sign_deadband_m=float(config["evaluation"]["improvement_sign_deadband_m"]),
            random_seed=int(config["evaluation"]["random_seed"]),
            random_draws=int(config["evaluation"]["random_draws_per_state"]),
        )
    _require(maximum_error <= 1.0e-9,
             "oracle-anchor prediction replay differs")
    _require(zero_error <= float(config["gates"]["architectural_zero_response_maximum_error_m"]),
             "oracle-anchor zero-response architecture differs")
    _require(reports == result["reports"],
             "oracle-anchor report replay differs")
    validation = {
        "schema_version": VALIDATION_SCHEMA, "status": "passing",
        "producer_commit": producer_commit, "validator_commit": validator_commit,
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "maximum_prediction_replay_error_m": maximum_error,
        "architectural_zero_response_maximum_error_m": zero_error,
        "reports": reports, "gates": result["gates"],
        "response_gate_pass": result["response_gate_pass"],
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
        "architectural_zero_response_maximum_error_m": output["architectural_zero_response_maximum_error_m"],
        "validation_payload_sha256": output["validation_payload_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
