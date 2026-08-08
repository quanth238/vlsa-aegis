#!/usr/bin/env python3
"""Run the one-state substep geometry and oracle-affine audit on E05."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Optional, Sequence

from scripts.evaluate_distal_sitl_candidate_e05 import _disable_probe_images
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    EXPECTED_ACTION_HORIZON,
    _atomic_write,
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    false_safe_result_path: Path,
    geometry_config_path: Path,
    audit_config_path: Path,
    obstacle_primitive_config_path: Optional[Path],
    expected_commit: str,
    output_path: Path,
) -> dict[str, Any]:
    import numpy as np

    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION,
        TABLE_SETTLE_ACTIONS,
        _active_obstacle,
        _build_environment,
        _runtime_imports,
        _settle,
        pairing_record,
        read_jsonl,
        validate_case_row,
    )
    from main.multilink_ellipsoid.oracle_affine import (
        ORACLE_AFFINE_RESULT_SCHEMA,
        load_oracle_affine_config,
        run_oracle_affine_audit,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        OBSTACLE_PRIMITIVE_RESULT_SCHEMA,
        ConservativeObstaclePrimitiveUnion,
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.rollout import _dynamic_state_vector
    from main.multilink_ellipsoid.shadow import (
        MultilinkEllipsoidShadow,
        allocation_record,
        load_shadow_config,
    )

    started = time.perf_counter_ns()
    archived = _load(archived_path)
    _require(
        _file_sha256(archived_path) == ARCHIVED_FILE_SHA256,
        "archived Table-1 file hash differs",
    )
    _require(
        archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256,
        "archived Table-1 payload identity differs",
    )
    archived_actions = archived.get("actions")
    _require(
        isinstance(archived_actions, list)
        and len(archived_actions) == EXPECTED_ACTION_HORIZON,
        "archived successful AEGIS action horizon differs",
    )
    false_safe = _load(false_safe_result_path)
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary oracle-affine manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)
    geometry_config = load_shadow_config(geometry_config_path)
    audit_config = load_oracle_affine_config(audit_config_path)
    obstacle_primitive_config = (
        None
        if obstacle_primitive_config_path is None
        else load_obstacle_primitive_config(obstacle_primitive_config_path)
    )
    _require(geometry_config["case_ids"] == [CASE_ID], "geometry config case differs")
    _require(audit_config["case_ids"] == [CASE_ID], "audit config case differs")
    if obstacle_primitive_config is not None:
        _require(
            obstacle_primitive_config["case_ids"] == [CASE_ID],
            "obstacle primitive config case differs",
        )
        base_identity = obstacle_primitive_config["base_audit_config"]
        _require(
            base_identity["config_file_sha256"]
            == audit_config["config_file_sha256"]
            and base_identity["config_payload_sha256"]
            == audit_config["config_payload_sha256"],
            "obstacle primitive base audit config identity differs",
        )
    false_safe_source = audit_config["false_safe_source"]
    _require(
        _file_sha256(false_safe_result_path) == false_safe_source["file_sha256"],
        "false-safe source file hash differs",
    )
    _require(
        false_safe.get("result_payload_sha256")
        == false_safe_source["result_payload_sha256"],
        "false-safe source payload identity differs",
    )
    _require(
        false_safe.get("schema_version")
        == "vlsa_distal_sitl_candidate_e05_result.v1"
        and false_safe.get("status") == "complete"
        and false_safe.get("scientific_result") is True
        and false_safe.get("case_id") == CASE_ID,
        "false-safe source result identity differs",
    )
    _require(
        str(false_safe.get("allocation", {}).get("slurm_job_id"))
        == false_safe_source["slurm_job_id"],
        "false-safe source Slurm identity differs",
    )
    false_safe_actions = false_safe.get("actions")
    _require(
        isinstance(false_safe_actions, list)
        and len(false_safe_actions) == EXPECTED_ACTION_HORIZON,
        "false-safe source action horizon differs",
    )
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    runtime = _runtime_imports(include_aegis=False)
    env = None
    probe_env = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        env, task, observation, selected_initial_state = _build_environment(
            runtime, case, render_resolution=TABLE_RENDER_RESOLUTION
        )
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        probe_env, probe_task, probe_observation, probe_initial_state = _build_environment(
            runtime, case, render_resolution=32
        )
        probe_observation = _settle(
            probe_env, probe_observation, TABLE_SETTLE_ACTIONS
        )
        _require(str(probe_task.language) == str(task.language), "probe task differs")
        _require(
            np.array_equal(
                np.asarray(probe_initial_state), np.asarray(selected_initial_state)
            ),
            "probe initial state differs",
        )
        _require(
            np.array_equal(
                np.asarray(probe_env.sim.get_state().flatten()),
                np.asarray(env.sim.get_state().flatten()),
            ),
            "settled probe simulator state differs",
        )
        obstacle_name, _ = _active_obstacle(env, observation)
        probe_obstacle_name, _ = _active_obstacle(probe_env, probe_observation)
        _require(probe_obstacle_name == obstacle_name, "probe obstacle differs")
        _require(
            obstacle_name == archived["obstacle"]["active_name"],
            "active obstacle differs from Table 1",
        )
        settled_state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
        pairing = pairing_record(
            case=case,
            selected_initial_state=selected_initial_state,
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=settled_state,
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
                "oracle-affine pairing field differs: %s" % key,
            )
        disabled_main_images = _disable_probe_images(env)
        disabled_probe_images = _disable_probe_images(probe_env)

        perception = archived["perception"]
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(
            geometry_config,
            {
                "p2": perception["mvee_center"],
                "R2": perception["mvee_rotation"],
                "Q2_diag": perception["mvee_semiaxes"],
                "record": {"label": perception["obstacle_label"]},
            },
        )
        geometry_record = geometry.geometry_record(env)
        _require(
            geometry_record["distal_ellipsoid_count"] == 7,
            "oracle-affine distal geometry count differs",
        )
        _require(
            geometry_record["total_constraint_geometry_count"] == 8,
            "oracle-affine total geometry count differs",
        )

        audit_step = int(audit_config["audit_step"])
        source_audit_action = false_safe_actions[audit_step]
        source_filter = source_audit_action.get("filter", {})
        source_clearance = np.asarray(
            source_filter.get("nominal_next_clearance_m"), dtype=np.float64
        )
        source_contacts = source_audit_action.get("protected_link_contact_events")
        _require(
            int(source_audit_action.get("step", -1)) == audit_step
            and source_clearance.shape == (8,)
            and np.all(np.isfinite(source_clearance))
            and np.all(source_clearance > 0.0)
            and source_filter.get("nominal_safe") is True
            and source_filter.get("modified") is False,
            "registered source action is not the positive-clearance false-safe",
        )
        _require(
            isinstance(source_contacts, list)
            and any(
                event.get("other", {}).get("geom_name")
                == "robot0_link6_collision"
                and float(event.get("distance", 1.0)) <= 0.0
                for event in source_contacts
            ),
            "registered source action lacks direct nonpositive L6 contact",
        )
        prefix_hashes = []
        for index in range(audit_step):
            source_action = false_safe_actions[index]
            _require(
                int(source_action["step"]) == index,
                "false-safe prefix action indexes differ",
            )
            action = np.asarray(
                source_action["executed_sitl_action"], dtype=np.float64
            )
            _require(
                action.shape == (7,)
                and np.array_equal(
                    action,
                    np.asarray(
                        source_action["filter"]["executed_action"],
                        dtype=np.float64,
                    ),
                ),
                "false-safe prefix executed-action binding differs",
            )
            env.step(action.tolist())
            prefix_hashes.append(
                _sha256(
                    json.dumps(
                        action.tolist(), separators=(",", ":"), allow_nan=False
                    ).encode("utf-8")
                )
            )
        state_vector = _dynamic_state_vector(env)
        audit_state_sha256 = _sha256(state_vector.tobytes())
        nominal = np.asarray(
            source_audit_action["executed_sitl_action"], dtype=np.float64
        )
        _require(
            nominal.shape == (7,)
            and np.array_equal(
                nominal,
                np.asarray(source_filter["executed_action"], dtype=np.float64),
            ),
            "false-safe audit action binding differs",
        )
        obstacle_primitive_union = (
            None
            if obstacle_primitive_config is None
            else ConservativeObstaclePrimitiveUnion(
                obstacle_primitive_config, env, obstacle_name
            )
        )
        obstacle_primitive_record = (
            None
            if obstacle_primitive_union is None
            else obstacle_primitive_union.geometry_record(env)
        )
        analytic_shadow = geometry.evaluate(env, nominal.tolist(), step=audit_step)
        audit = run_oracle_affine_audit(
            audit_config,
            geometry,
            env,
            probe_env,
            obstacle_name,
            nominal.tolist(),
            obstacle_primitive_union=obstacle_primitive_union,
        )
        result = {
            "schema_version": (
                ORACLE_AFFINE_RESULT_SCHEMA
                if obstacle_primitive_config is None
                else OBSTACLE_PRIMITIVE_RESULT_SCHEMA
            ),
            "status": "complete",
            "scientific_result": True,
            "case_id": CASE_ID,
            "claim_scope": (
                audit_config["claim_scope"]
                if obstacle_primitive_config is None
                else obstacle_primitive_config["claim_scope"]
            ),
            "source": source,
            "allocation": allocation,
            "config": audit_config,
            "obstacle_primitive_config": obstacle_primitive_config,
            "geometry_config": geometry_config,
            "archived_table1": {
                "path": str(archived_path),
                "file_sha256": ARCHIVED_FILE_SHA256,
                "payload_sha256": ARCHIVED_PAYLOAD_SHA256,
                "read_only": True,
                "original_outcome": {
                    "first_link5_contact_step": 187,
                    "first_paper_car_step": 188,
                    "native_task_success_step": 236,
                },
            },
            "false_safe_source": {
                "path": str(false_safe_result_path),
                "file_sha256": false_safe_source["file_sha256"],
                "payload_sha256": false_safe_source[
                    "result_payload_sha256"
                ],
                "slurm_job_id": false_safe_source["slurm_job_id"],
                "read_only": True,
                "action_count": len(false_safe_actions),
                "audit_step_endpoint_clearance_m": source_clearance.tolist(),
                "audit_step_protected_contact_events": source_contacts,
            },
            "pairing": pairing,
            "probe_environment": {
                "same_bddl_task_episode_and_settled_state": True,
                "main_disabled_image_observable_count": disabled_main_images,
                "probe_disabled_image_observable_count": disabled_probe_images,
                "osc_controller": "OSC_POSE",
                "control_frequency_hz": 20,
                "internal_substep_capture": audit_config["substep_measurement"][
                    "capture_hook"
                ],
            },
            "geometry": (
                geometry_record
                if obstacle_primitive_config is None
                else {
                    "accepted_robot_and_released_ee": geometry_record,
                    "conservative_obstacle_union": obstacle_primitive_record,
                }
            ),
            "audit_state": {
                "step": audit_step,
                "immutable_prefix_action_count": audit_step,
                "immutable_prefix_source": (
                    "job_37109_executed_sitl_action_ledger"
                ),
                "immutable_prefix_action_sha256": prefix_hashes,
                "dynamic_state_sha256": audit_state_sha256,
                "nominal_action": nominal.tolist(),
            },
            (
                "analytic_shadow_at_audit_state"
                if obstacle_primitive_config is None
                else "released_aegis_analytic_shadow_diagnostic"
            ): analytic_shadow,
            "oracle_affine_audit": audit,
            "research_direction_go": bool(
                audit["decision"]["research_direction_go"]
            ),
            "stop_reason": audit["decision"]["stop_reason"],
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
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def main(argv: Sequence[str] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--false-safe-result", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--audit-config", type=Path, required=True)
    parser.add_argument("--obstacle-primitive-config", type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        false_safe_result_path=args.false_safe_result.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        audit_config_path=args.audit_config.resolve(),
        obstacle_primitive_config_path=(
            None
            if args.obstacle_primitive_config is None
            else args.obstacle_primitive_config.resolve()
        ),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "research_direction_go": result["research_direction_go"],
                "stop_reason": result["stop_reason"],
                "output": str(args.output.resolve()),
                "result_payload_sha256": result["result_payload_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
