#!/usr/bin/env python3
"""Fresh action-ledger replay for the fixed E05 PNCBF-style backup oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_pncbf_backup_oracle_e05 import RESULT_SCHEMA, _canonical
from scripts.evaluate_distal_smooth_field_attribution_e05 import (
    InstrumentedContinuationProbe,
    _disable_images,
)
from scripts.evaluate_distal_counterfactual_field_e05 import FixedContinuationProbe
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    _file_sha256,
    _load,
    _require,
)


VALIDATION_SCHEMA = "vlsa_distal_pncbf_backup_oracle_e05_validation.v1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate(
    *, repo_root: Path, result_path: Path, manifest_path: Path,
    archived_path: Path, geometry_config_path: Path,
    experiment_config_path: Path, expected_commit: str,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION, TABLE_SETTLE_ACTIONS, _active_obstacle,
        _build_environment, _runtime_imports, _settle, array_sha256,
        read_jsonl, validate_case_row,
    )
    from main.multilink_ellipsoid.pncbf_backup import (
        load_config, physical_safe, select_repulsive_candidate,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow, allocation_record, load_shadow_config,
    )
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe

    result = json.loads(result_path.read_text())
    _require(result["schema_version"] == RESULT_SCHEMA, "producer schema differs")
    payload = dict(result)
    claimed = payload.pop("result_payload_sha256")
    _require(claimed == _sha256(_canonical(payload)), "producer self-hash differs")
    _require(result["source"]["commit"] == expected_commit, "producer commit differs")
    _require(result["scientific_result"] is True, "producer is not scientific")
    _require("H100" in result["allocation"]["device"]["name"], "producer was not H100")
    _require(result["case_id"] == CASE_ID, "producer case differs")
    config = load_config(experiment_config_path)
    _require(
        config["config_payload_sha256"] == result["config"]["config_payload_sha256"],
        "producer config differs",
    )
    car_limit = float(config["measurement"]["paper_car_threshold_m"])
    expected_substeps = int(config["measurement"]["expected_mujoco_substeps_per_action"])
    tolerance = float(config["measurement"]["boundary_equivalence_tolerance"])
    for query in result["policy_queries"]:
        returned = np.asarray(query["returned_actions"], dtype=np.float64)
        _require(returned.shape == (10, 7), "stored policy chunk differs")
        _require(array_sha256(returned) == query["returned_actions_sha256"], "policy chunk hash differs")
    for window in result["windows"]:
        expected_count = int(window["action_count"])
        for record in [window["nominal"]] + [item["record"] for item in window["candidates"]]:
            _require(record["sample_count"] == 1 + expected_substeps * expected_count, "window sample count differs")
            _require(record["substep_counts"] == [expected_substeps] * expected_count, "window substeps differ")
            _require(record["maximum_boundary_equivalence_error_m"] <= tolerance, "window boundary equivalence differs")
        if window["selected"] is not None:
            _require(physical_safe(window["selected"], car_limit), "selected window is physically unsafe")
        if window["selected_source"] == "hysteretic_normal_repulsion":
            expected = select_repulsive_candidate(
                window["candidates"],
                nominal_clearance_m=float(window["nominal"]["minimum_clearance_m"]),
                activation_clearance_m=float(config["warning"]["activation_clearance_m"]),
                paper_car_threshold_m=car_limit,
            )
            _require(expected is not None, "selected repulsion is not reproducible")
            selected_actions = np.asarray(
                [row["action"] for row in result["actions"] if window["step"] <= row["step"] < window["step"] + window["executed_action_count"]],
                dtype=np.float64,
            )
            _require(
                np.array_equal(
                    selected_actions,
                    np.asarray(expected["actions"], dtype=np.float64)[: int(window["executed_action_count"])],
                ),
                "selected repulsion actions differ",
            )

    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file differs")
    _require(archived["result_payload_sha256"] == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    rows = [row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID]
    _require(len(rows) == 1, "manifest case differs")
    case = rows[0]
    validate_case_row(case, repo_root)
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(geometry_config_path)
    env = probe_env = None
    try:
        env, task, observation, initial_state = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        probe_env, probe_task, probe_observation, probe_state = _build_environment(
            runtime, case, render_resolution=32
        )
        probe_observation = _settle(probe_env, probe_observation, TABLE_SETTLE_ACTIONS)
        _require(np.array_equal(initial_state, probe_state), "validator probe state differs")
        _require(str(task.language) == str(probe_task.language), "validator task differs")
        obstacle_name, _ = _active_obstacle(env, observation)
        initial_obstacle = np.asarray(observation["%s_pos" % obstacle_name], dtype=np.float64)
        perception = archived["perception"]
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(
            geometry_config,
            {"p2": perception["mvee_center"], "R2": perception["mvee_rotation"],
             "Q2_diag": perception["mvee_semiaxes"],
             "record": {"label": perception["obstacle_label"]}},
        )
        one_step = SlabbedEightConstraintProbe(
            probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name
        )
        _disable_images(probe_env)
        instrumented = InstrumentedContinuationProbe(
            one_step, obstacle_name, initial_obstacle
        )
        minimum_clearance = float("inf")
        protected_contacts = []
        maximum_car = 0.0
        success_step = None
        records = result["actions"]
        _require([item["step"] for item in records] == list(range(len(records))), "action ledger is not contiguous")
        for item in records:
            step = int(item["step"])
            command = np.asarray(item["action"], dtype=np.float64)
            evidence = instrumented.rollout_internal(
                env, command.reshape(1, 7), expected_substeps=expected_substeps,
                boundary_tolerance=tolerance, step_base=step,
            )
            minimum_clearance = min(minimum_clearance, float(evidence["minimum_clearance_m"]))
            protected_contacts.extend(evidence["protected_contacts"])
            maximum_car = max(maximum_car, float(evidence["maximum_active_obstacle_l1_displacement_m"]))
            observation, reward, done, _ = env.step(command.tolist())
            _require(array_sha256(_dynamic_state_vector(env)) == item["next_state_sha256"], "fresh replay state hash differs")
            _require(bool(done) == bool(item["done"]), "fresh replay task signal differs")
            if done and success_step is None:
                success_step = step
        replay_primary = bool(
            success_step is not None and len(protected_contacts) == 0
            and maximum_car <= car_limit and result["failure"] is None
        )
        _require(replay_primary == result["primary_problem_solved"], "fresh replay primary gate differs")
        video = Path(result["video"]["path"])
        final = Path(result["final_jpg"]["path"])
        _require(video.is_file() and final.is_file(), "producer media is missing")
        _require(_file_sha256(video) == result["video"]["file_sha256"], "producer video hash differs")
        _require(_file_sha256(final) == result["final_jpg"]["file_sha256"], "producer final frame hash differs")
        validation = {
            "schema_version": VALIDATION_SCHEMA,
            "status": "validated",
            "scientific_result": True,
            "producer_commit": expected_commit,
            "validator_source": {"commit": expected_commit},
            "allocation": allocation_record(),
            "result_file_sha256": _file_sha256(result_path),
            "result_payload_sha256": claimed,
            "fresh_replay": {
                "action_count": len(records),
                "minimum_L5_L7_proxy_clearance_m": minimum_clearance,
                "protected_contact_event_count": len(protected_contacts),
                "maximum_active_obstacle_l1_displacement_m": maximum_car,
                "native_task_success_step": success_step,
            },
            "primary_problem_solved": replay_primary,
            "interpretation": result["interpretation"],
        }
        validation["validation_payload_sha256"] = _sha256(_canonical(validation))
        return validation
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = validate(
        repo_root=args.repo_root.resolve(), result_path=args.result.resolve(),
        manifest_path=args.manifest.resolve(), archived_path=args.archived.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        experiment_config_path=args.experiment_config.resolve(),
        expected_commit=args.expected_commit,
    )
    _atomic_write(args.output.resolve(), value)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
