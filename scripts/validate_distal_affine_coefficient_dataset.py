#!/usr/bin/env python3
"""Independently validate the multi-state affine coefficient target gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from main.multilink_ellipsoid.affine_coefficient_model import (
    AFFINE_COEFFICIENT_DATASET_RESULT_SCHEMA,
    AFFINE_COEFFICIENT_DATASET_SCHEMA,
    affine_values,
    load_affine_coefficient_config,
)
from main.multilink_ellipsoid.two_step_margin import load_selected_manifest
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write, _file_sha256, _load, _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_affine_coefficient_config(args.config.resolve())
    selected = load_selected_manifest(args.selected_manifest.resolve(), config)
    result = _load(args.result.resolve())
    dataset = _load(args.dataset.resolve())
    _require(
        result.get("schema_version") == AFFINE_COEFFICIENT_DATASET_RESULT_SCHEMA
        and dataset.get("schema_version") == AFFINE_COEFFICIENT_DATASET_SCHEMA,
        "affine-coefficient dataset schemas differ",
    )
    _require(
        result.get("source", {}).get("commit") == args.expected_commit
        and dataset.get("source_commit") == args.expected_commit,
        "affine-coefficient dataset source differs",
    )
    _require(
        result.get("result_payload_sha256") == _hash_without(result, "result_payload_sha256")
        and dataset.get("dataset_payload_sha256")
        == _hash_without(dataset, "dataset_payload_sha256")
        and result.get("dataset", {}).get("file_sha256")
        == _file_sha256(args.dataset.resolve()),
        "affine-coefficient dataset binding differs",
    )
    states = dataset.get("state_records", [])
    expected_states = len(selected) * int(
        config["state_sampling"]["expected_states_per_episode"]
    )
    _require(
        len(states) == expected_states
        and [item.get("state_index") for item in states] == list(range(expected_states)),
        "affine-coefficient state count/index differs",
    )
    selected_by_case = {item["case_id"]: item for item in selected}
    gate = True
    for state in states:
        row = selected_by_case[state["case_id"]]
        _require(
            state["split"] == row["split"]
            and state["task_level_group_id"] == row["task_level_group_id"],
            "affine-coefficient complete episode split leaked",
        )
        candidates = np.asarray(state["candidate_first_xyz"], dtype=np.float64)
        margins = np.asarray(
            state["candidate_minimum_distal_margin_m"], dtype=np.float64
        )
        target = state.get("coefficient_target")
        if not state.get("gate_pass") or target is None:
            gate = False
            continue
        _require(
            candidates.shape == (
                config["action_sampling"]["expected_grid_action_count_per_state"], 3
            ) and margins.shape == (len(candidates), 7),
            "affine-coefficient grid arrays differ",
        )
        predicted = np.asarray([
            affine_values(
                target["nominal_margin_m"], target["gradient_m_per_action"],
                target["state_conditioned_error_m"], xyz,
                state["nominal_first_action"][:3],
            ) for xyz in candidates
        ])
        _require(
            float(np.max(predicted - margins)) <= 1.0e-8
            and not np.any(np.logical_and(predicted >= 0.0, margins < 0.0)),
            "affine-coefficient lower envelope postcheck differs",
        )
        exact = state["oracle_qp_exact_verification"]
        _require(
            state["oracle_qp"]["valid"]
            and state["oracle_qp"]["diagnostics"]["input_constraint_count"] == 7
            and exact["D_opt_seven_distal_safe"]
            and exact["released_AEGIS_EE_proxy_safe"]
            and exact["D_sim_raw_safe"]
            and exact["maximum_within_step_obstacle_l1_displacement_m"] <= 1.0e-4,
            "affine-coefficient exact oracle verification differs",
        )
    episodes = dataset.get("episode_results", [])
    _require(
        len(episodes) == len(selected)
        and {item["case_id"] for item in episodes} == set(selected_by_case),
        "affine-coefficient episode set differs",
    )
    gate = bool(gate and all(item.get("gate_pass") for item in episodes))
    _require(
        bool(dataset["summary"]["dataset_gate_pass"]) is gate
        and bool(result["decision"]["dataset_gate_pass"]) is gate
        and bool(result["decision"]["neural_training_authorized"]) is gate,
        "affine-coefficient dataset decision differs",
    )
    output = {
        "schema_version": "vlsa_distal_affine_coefficient_moka10_dataset_validation.v1",
        "status": "valid", "scientific_result": False,
        "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "dataset_file_sha256": _file_sha256(args.dataset.resolve()),
        "config_file_sha256": _file_sha256(args.config.resolve()),
        "selected_manifest_file_sha256": _file_sha256(args.selected_manifest.resolve()),
        "episode_count": len(episodes), "state_count": len(states),
        "neural_training_authorized": gate,
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps({"status": "valid", "neural_training_authorized": gate}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
