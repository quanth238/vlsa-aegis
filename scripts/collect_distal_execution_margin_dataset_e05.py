#!/usr/bin/env python3
"""Collect the preregistered E05 oracle dataset before neural training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Optional, Sequence

from scripts.evaluate_distal_execution_margin_nn_e05 import (
    _build_pair,
    _collect_dataset,
    _geometry,
    _verify_sources,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    CASE_ID,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _require,
    _sha256,
)


DATASET_RESULT_SCHEMA = "vlsa_distal_execution_margin_dataset_e05_result.v1"


def collect(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    false_safe_result_path: Path,
    obstacle_discovery_result_path: Path,
    geometry_config_path: Path,
    exact_box_config_path: Path,
    config_path: Path,
    expected_commit: str,
    dataset_path: Path,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        _runtime_imports,
        pairing_record,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.execution_margin_nn import (
        load_execution_margin_config,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        EXACT_BOX_OBSTACLE_SCHEMA,
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.shadow import allocation_record, load_shadow_config

    started = time.perf_counter_ns()
    config = load_execution_margin_config(config_path)
    geometry_config = load_shadow_config(geometry_config_path)
    exact_box_config = load_obstacle_primitive_config(exact_box_config_path)
    _require(
        exact_box_config["schema_version"] == EXACT_BOX_OBSTACLE_SCHEMA
        and exact_box_config["config_payload_sha256"]
        == config["immutable_sources"]["exact_box_config_payload_sha256"],
        "exact-box config payload identity differs",
    )
    archived, false_safe, discovery = _verify_sources(
        archived_path=archived_path,
        false_safe_result_path=false_safe_result_path,
        obstacle_discovery_result_path=obstacle_discovery_result_path,
        geometry_config_path=geometry_config_path,
        exact_box_config_path=exact_box_config_path,
        config=config,
    )
    matches = [
        row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID
    ]
    _require(len(matches) == 1, "primary dataset manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    env = probe_env = None
    try:
        env, probe_env, task, observation, setup = _build_pair(runtime, case)
        _require(
            setup["obstacle_name"] == archived["obstacle"]["active_name"],
            "active obstacle differs from Table 1",
        )
        pairing = pairing_record(
            case=case,
            selected_initial_state=setup["selected_initial_state"],
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=setup["obstacle_name"],
            settled_simulator_state=np.asarray(
                env.sim.get_state().flatten(), dtype=np.float64
            ),
        )
        for key in (
            "manifest_row_sha256",
            "initial_state_sha256",
            "initial_observation_sha256",
            "settled_simulator_state_sha256",
            "settled_active_obstacle_position_sha256",
            "policy_noise_schedule_sha256",
        ):
            _require(
                pairing[key] == archived["pairing"][key],
                "dataset pairing field differs: %s" % key,
            )
        geometry, exact_boxes = _geometry(
            geometry_config=geometry_config,
            exact_box_config=exact_box_config,
            archived=archived,
            env=env,
            obstacle_name=setup["obstacle_name"],
        )
        geometry_record = geometry.geometry_record(env)
        exact_box_record = exact_boxes.geometry_record(env)
        records, state_records, summary = _collect_dataset(
            env=env,
            probe_env=probe_env,
            geometry=geometry,
            exact_boxes=exact_boxes,
            obstacle_name=setup["obstacle_name"],
            actions=false_safe["actions"],
            config=config,
        )
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()
    dataset = {
        "schema_version": "vlsa_distal_execution_margin_nn_e05_dataset.v1",
        "case_id": CASE_ID,
        "config_file_sha256": config["config_file_sha256"],
        "config_payload_sha256": config["config_payload_sha256"],
        "source_commit": source["commit"],
        "constraint_order": [
            "L5_part_0",
            "L5_part_1",
            "L5_part_2",
            "L6_part_0",
            "L6_part_1",
            "L7_part_0",
            "L7_part_1",
        ],
        "summary": summary,
        "states": state_records,
        "records": records,
    }
    dataset["dataset_payload_sha256"] = _sha256(
        json.dumps(
            dataset,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    _atomic_write(dataset_path, dataset)
    gate_pass = bool(summary["oracle_analysis_gate_pass"])
    result = {
        "schema_version": DATASET_RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "case_id": CASE_ID,
        "claim_scope": (
            "single_E05_privileged_exact_box_cloned_OSC_dataset_and_"
            "pretraining_oracle_analysis_gate_not_neural_training_not_policy_efficacy"
        ),
        "source": source,
        "allocation": allocation,
        "config": config,
        "pairing": pairing,
        "immutable_sources": {
            "archived_table1": str(archived_path),
            "false_safe_action_ledger": str(false_safe_result_path),
            "exact_box_discovery": str(obstacle_discovery_result_path),
            "discovery_schema_version": discovery.get("schema_version"),
            "read_only": True,
        },
        "probe_environment": {
            "osc_controller": "OSC_POSE",
            "control_frequency_hz": 20,
            "internal_substep_capture": (
                "interval_start_and_every_internal_MuJoCo_step"
            ),
            "disabled_image_observable_count": setup["disabled_images"],
        },
        "geometry": {
            "accepted_robot_and_released_ee": geometry_record,
            "exact_obstacle_boxes": exact_box_record,
        },
        "dataset": {
            "path": str(dataset_path),
            "file_sha256": _file_sha256(dataset_path),
            "payload_sha256": dataset["dataset_payload_sha256"],
            "sample_count": len(records),
        },
        "dataset_summary": summary,
        "decision": {
            "geometry_authority_pass": bool(summary["geometry_authority_pass"]),
            "recoverable_exact_proxy_crossing_steps": summary[
                "recoverable_exact_proxy_crossing_steps"
            ],
            "primary_projection_step": config["state_groups"][
                "primary_projection_step"
            ],
            "primary_local_jointly_safe_candidate_exists": bool(
                summary["primary_local_jointly_safe_candidate_exists"]
            ),
            "oracle_analysis_gate_pass": gate_pass,
            "neural_training_authorized": gate_pass,
        },
        "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        "failure": None,
    }
    result["result_payload_sha256"] = _sha256(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--false-safe-result", type=Path, required=True)
    parser.add_argument("--obstacle-discovery-result", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = collect(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        false_safe_result_path=args.false_safe_result.resolve(),
        obstacle_discovery_result_path=args.obstacle_discovery_result.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        exact_box_config_path=args.exact_box_config.resolve(),
        config_path=args.config.resolve(),
        expected_commit=args.expected_commit,
        dataset_path=args.dataset.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(json.dumps(result["decision"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
