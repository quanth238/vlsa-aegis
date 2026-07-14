#!/usr/bin/env python3
"""Calibrate OSC translation and held-out D_opt to D_sim transfer."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from crfs_harness.artifacts import atomic_write_json, validate_jsonl_unique
from crfs_oracle.measurement import signed_distance_point_to_oriented_box
from crfs_oracle.runner import SafeLiberoCase, oracle_config_from_mapping


def predict_clearance(start, actions, response, boxes, radius, samples_per_segment=26):
    waypoints = [np.asarray(start, dtype=np.float64)]
    for action in actions:
        waypoints.append(waypoints[-1] + response @ np.asarray(action[:3], dtype=np.float64))
    minimum = float("inf")
    for first, second in zip(waypoints[:-1], waypoints[1:]):
        for alpha in np.linspace(0.0, 1.0, samples_per_segment):
            point = (1.0 - alpha) * first + alpha * second
            for box in boxes:
                distance = signed_distance_point_to_oriented_box(
                    point,
                    box["center_m"],
                    np.asarray(box["rotation_world"]).reshape(3, 3),
                    box["half_size_m"],
                ) - radius
                minimum = min(minimum, float(distance))
    return minimum, waypoints[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-states", type=int, default=10)
    parser.add_argument("--heldout-prefixes-per-state", type=int, default=2)
    parser.add_argument("--probe-amplitude", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    cases, errors = validate_jsonl_unique(args.manifest, "case_id")
    selected = cases[: args.calibration_states]
    if errors or len(selected) < args.calibration_states:
        raise SystemExit(f"invalid or short manifest: {errors}")
    value = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = oracle_config_from_mapping(
        value,
        host="unused",
        port=0,
        checkpoint_id="not-used-in-H04",
        checkpoint_sha256="not-used-in-H04",
        output_root=str(Path(args.output).parent),
        run_id="h04-calibration",
    )
    environment = SafeLiberoCase(selected[0], config)
    inputs = []
    outputs = []
    probe_records = []
    try:
        for case in selected:
            environment.configure_case(case)
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    actions = np.zeros((5, 7), dtype=np.float64)
                    actions[:, 6] = -1.0
                    actions[0, axis] = sign * args.probe_amplitude
                    rollout = environment.rollout(actions)
                    displacement = np.asarray(rollout["end_eef_m"]) - np.asarray(rollout["start_eef_m"])
                    inputs.append(actions[0, :3])
                    outputs.append(displacement)
                    probe_records.append(
                        {
                            "case_id": case["case_id"],
                            "episode_index": case["episode_index"],
                            "action": actions[0, :3].tolist(),
                            "displacement_m": displacement.tolist(),
                        }
                    )
        x = np.asarray(inputs, dtype=np.float64)
        y = np.asarray(outputs, dtype=np.float64)
        response = np.linalg.lstsq(x, y, rcond=None)[0].T

        rng = np.random.default_rng(args.seed)
        heldout = []
        for case in selected:
            environment.configure_case(case)
            for _ in range(args.heldout_prefixes_per_state):
                actions = np.zeros((5, 7), dtype=np.float64)
                actions[:, :3] = rng.uniform(-args.probe_amplitude, args.probe_amplitude, size=(5, 3))
                actions[:, 6] = -1.0
                rollout = environment.rollout(actions)
                predicted_clearance, predicted_endpoint = predict_clearance(
                    rollout["start_eef_center_m"],
                    actions,
                    response,
                    rollout["branch_obstacle_boxes"],
                    config.eef_radius_m,
                )
                actual_endpoint = np.asarray(rollout["end_eef_m"], dtype=np.float64)
                predicted_site_endpoint = np.asarray(rollout["start_eef_m"], dtype=np.float64)
                predicted_site_endpoint += sum((response @ action[:3] for action in actions), np.zeros(3))
                heldout.append(
                    {
                        "case_id": case["case_id"],
                        "episode_index": case["episode_index"],
                        "actions": actions.tolist(),
                        "d_opt_m": predicted_clearance,
                        "d_sim_m": float(rollout["clearance_m"]),
                        "clearance_error_m": predicted_clearance - float(rollout["clearance_m"]),
                        "endpoint_error_m": float(np.linalg.norm(predicted_site_endpoint - actual_endpoint)),
                        "predicted_center_endpoint_m": predicted_endpoint.tolist(),
                        "physical_contact": bool(rollout["contact"]),
                    }
                )
    finally:
        environment.close()

    clearance_errors = np.abs([record["clearance_error_m"] for record in heldout])
    endpoint_errors = np.asarray([record["endpoint_error_m"] for record in heldout])
    false_safe = sum(record["d_opt_m"] >= 0.01 and record["d_sim_m"] < 0.0 for record in heldout)
    median_endpoint = float(np.median(endpoint_errors))
    p95_clearance = float(np.quantile(clearance_errors, 0.95))
    passed = bool(median_endpoint <= 0.005 and p95_clearance <= 0.01 and false_safe == 0)
    result = {
        "schema_version": "1.0",
        "gate": "H04",
        "status": "passed" if passed else "failed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "python_version": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "calibration_states": len(selected),
        "probe_amplitude": args.probe_amplitude,
        "response_matrix_m_per_action": response.tolist(),
        "heldout_prefixes": len(heldout),
        "median_endpoint_error_m": median_endpoint,
        "p95_absolute_clearance_error_m": p95_clearance,
        "false_safe_at_1cm_margin": false_safe,
        "thresholds": {
            "max_median_endpoint_error_m": 0.005,
            "max_p95_absolute_clearance_error_m": 0.01,
            "max_false_safe_at_1cm_margin": 0,
        },
        "probe_records": probe_records,
        "heldout_records": heldout,
    }
    atomic_write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if not key.endswith("records")}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
