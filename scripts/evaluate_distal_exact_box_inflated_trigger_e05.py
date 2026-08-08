#!/usr/bin/env python3
"""Scan the immutable E05 ledger and test the first 8 mm exact-box crossing."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

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


def _verify_result_source(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    result = _load(path)
    _require(_file_sha256(path) == expected["file_sha256"], "prior result file hash differs")
    _require(
        result.get("result_payload_sha256") == expected["result_payload_sha256"]
        and result.get("schema_version") == expected["schema_version"]
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and str(result.get("allocation", {}).get("slurm_job_id"))
        == expected["slurm_job_id"],
        "prior result identity differs",
    )
    return {
        "path": str(path),
        **expected,
        "read_only": True,
    }


def evaluate(
    *,
    repo_root: Path,
    manifest_path: Path,
    archived_path: Path,
    false_safe_result_path: Path,
    geometry_config_path: Path,
    base_audit_config_path: Path,
    exact_box_config_path: Path,
    trigger_config_path: Path,
    obstacle_discovery_result_path: Path,
    fixed_margin_result_path: Path,
    exact_box_result_path: Path,
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
    from main.multilink_ellipsoid.inflated_trigger import (
        INFLATED_TRIGGER_RESULT_SCHEMA,
        intervention_pass,
        is_first_crossing_candidate,
        load_inflated_trigger_config,
    )
    from main.multilink_ellipsoid.obstacle_primitives import (
        OBSTACLE_PRIMITIVE_RESULT_SCHEMA,
        ExactObstacleBoxUnion,
        load_obstacle_primitive_config,
    )
    from main.multilink_ellipsoid.oracle_affine import (
        SubstepEightConstraintProbe,
        load_oracle_affine_config,
        run_oracle_affine_audit,
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
        _file_sha256(archived_path) == ARCHIVED_FILE_SHA256
        and archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256,
        "archived Table-1 identity differs",
    )
    false_safe = _load(false_safe_result_path)
    trigger_config = load_inflated_trigger_config(trigger_config_path)
    base_audit_config = load_oracle_affine_config(base_audit_config_path)
    exact_box_config = load_obstacle_primitive_config(exact_box_config_path)
    geometry_config = load_shadow_config(geometry_config_path)
    _require(
        trigger_config["base_audit_config"]["config_file_sha256"]
        == base_audit_config["config_file_sha256"]
        and trigger_config["base_audit_config"]["config_payload_sha256"]
        == base_audit_config["config_payload_sha256"],
        "inflated-trigger base audit binding differs",
    )
    _require(
        trigger_config["exact_box_config"]["config_file_sha256"]
        == exact_box_config["config_file_sha256"]
        and trigger_config["exact_box_config"]["config_payload_sha256"]
        == exact_box_config["config_payload_sha256"],
        "inflated-trigger exact-box binding differs",
    )
    prior_sources = {
        "fixed_margin": _verify_result_source(
            fixed_margin_result_path,
            trigger_config["prior_results"]["fixed_margin"],
        ),
        "exact_box": _verify_result_source(
            exact_box_result_path,
            trigger_config["prior_results"]["exact_box"],
        ),
    }
    discovery = _load(obstacle_discovery_result_path)
    discovery_expected = exact_box_config["discovery_source"]
    _require(
        _file_sha256(obstacle_discovery_result_path)
        == discovery_expected["file_sha256"]
        and discovery.get("result_payload_sha256")
        == discovery_expected["result_payload_sha256"]
        and discovery.get("schema_version") == OBSTACLE_PRIMITIVE_RESULT_SCHEMA
        and str(discovery.get("allocation", {}).get("slurm_job_id"))
        == discovery_expected["slurm_job_id"],
        "obstacle discovery identity differs",
    )
    false_source = base_audit_config["false_safe_source"]
    _require(
        _file_sha256(false_safe_result_path) == false_source["file_sha256"]
        and false_safe.get("result_payload_sha256")
        == false_source["result_payload_sha256"]
        and false_safe.get("schema_version")
        == "vlsa_distal_sitl_candidate_e05_result.v1"
        and false_safe.get("case_id") == CASE_ID
        and str(false_safe.get("allocation", {}).get("slurm_job_id"))
        == false_source["slurm_job_id"],
        "immutable action ledger identity differs",
    )
    actions = false_safe.get("actions")
    _require(
        isinstance(actions, list) and len(actions) == EXPECTED_ACTION_HORIZON,
        "immutable action ledger length differs",
    )
    rows = read_jsonl(manifest_path)
    matches = [row for row in rows if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "primary manifest row is not unique")
    case = matches[0]
    validate_case_row(case, repo_root)
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
            np.array_equal(np.asarray(probe_initial_state), np.asarray(selected_initial_state))
            and np.array_equal(
                np.asarray(probe_env.sim.get_state().flatten()),
                np.asarray(env.sim.get_state().flatten()),
            ),
            "settled probe simulator state differs",
        )
        obstacle_name, _ = _active_obstacle(env, observation)
        probe_obstacle_name, _ = _active_obstacle(probe_env, probe_observation)
        _require(probe_obstacle_name == obstacle_name, "probe obstacle differs")
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
            _require(pairing[key] == archived["pairing"][key], "pairing field differs: %s" % key)
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
        obstacle_union = ExactObstacleBoxUnion(
            exact_box_config, env, obstacle_name
        )
        measurement = base_audit_config["substep_measurement"]
        probe = SubstepEightConstraintProbe(
            probe_env,
            geometry,
            active_obstacle_name=obstacle_name,
            quadratic_tolerance=float(
                measurement["contact_proxy_quadratic_tolerance"]
            ),
            contact_distance_threshold_m=float(
                measurement["raw_contact_distance_threshold_m"]
            ),
            obstacle_primitive_union=obstacle_union,
        )
        shell_m = float(trigger_config["rounded_box_shell"]["shell_m"])
        scan_settings = trigger_config["trigger_scan"]
        scan_records = []
        trigger_step = None
        trigger_action = None
        prefix_hashes = []
        for step in range(
            int(scan_settings["start_step"]),
            int(scan_settings["end_step_inclusive"]) + 1,
        ):
            source_action = actions[step]
            _require(int(source_action.get("step", -1)) == step, "action index differs")
            action = np.asarray(source_action.get("executed_sitl_action"), dtype=np.float64)
            _require(
                action.shape == (7,)
                and np.all(np.isfinite(action))
                and np.array_equal(
                    action,
                    np.asarray(source_action.get("filter", {}).get("executed_action"), dtype=np.float64),
                ),
                "immutable executed action binding differs",
            )
            current = np.asarray(probe.clearances(env), dtype=np.float64)
            transition = probe.transition(env, action.tolist())
            nominal_minimum = np.asarray(
                transition["minimum_substep_clearance_m"], dtype=np.float64
            )
            selected = is_first_crossing_candidate(
                current, nominal_minimum, shell_m
            )
            scan_records.append(
                {
                    "step": step,
                    "state_sha256": hashlib.sha256(
                        _dynamic_state_vector(env).tobytes()
                    ).hexdigest(),
                    "action": action.tolist(),
                    "current_exact_clearance_m": current.tolist(),
                    "current_inflated_clearance_m": (current - shell_m).tolist(),
                    "nominal_minimum_exact_clearance_m": nominal_minimum.tolist(),
                    "nominal_minimum_inflated_clearance_m": (
                        nominal_minimum - shell_m
                    ).tolist(),
                    "nominal_raw_protected_contact_count": int(
                        transition["raw_protected_contact_count"]
                    ),
                    "nominal_maximum_obstacle_l1_displacement_m": float(
                        transition["maximum_within_step_obstacle_l1_displacement_m"]
                    ),
                    "selected_first_crossing": selected,
                }
            )
            if selected:
                trigger_step = step
                trigger_action = action.copy()
                break
            env.step(action.tolist())
            _require(
                hashlib.sha256(_dynamic_state_vector(env).tobytes()).hexdigest()
                == transition["next_state_sha256"],
                "immutable prefix execution differs from exact clone",
            )
            prefix_hashes.append(
                _sha256(
                    json.dumps(
                        action.tolist(), separators=(",", ":"), allow_nan=False
                    ).encode("utf-8")
                )
            )
        _require(trigger_step is not None and trigger_action is not None, "no first crossing trigger found")
        effective_audit = copy.deepcopy(base_audit_config)
        effective_audit.pop("config_file_sha256", None)
        effective_audit.pop("config_payload_sha256", None)
        effective_audit["affine_model"]["clearance_target_m"] = shell_m
        audit = run_oracle_affine_audit(
            effective_audit,
            geometry,
            env,
            probe_env,
            obstacle_name,
            trigger_action.tolist(),
            obstacle_primitive_union=obstacle_union,
        )
        passed = intervention_pass(audit)
        decision = audit["decision"]
        if not decision["local_jointly_raw_and_proxy_safe_candidate_exists"]:
            stop_reason = "no_jointly_raw_and_inflated_proxy_safe_candidate_at_first_crossing"
        elif not decision["affine_candidate_gate_pass"]:
            stop_reason = "inflated_clearance_map_not_conservatively_affine_on_candidates"
        elif not decision["qp_valid"]:
            stop_reason = "inflated_first_crossing_qp_infeasible"
        elif not (decision["qp_exact_raw_safe"] and decision["qp_exact_proxy_safe"]):
            stop_reason = "inflated_first_crossing_qp_failed_exact_verification"
        else:
            stop_reason = None
        result = {
            "schema_version": INFLATED_TRIGGER_RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "case_id": CASE_ID,
            "claim_scope": trigger_config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "trigger_config": trigger_config,
            "base_audit_config": base_audit_config,
            "exact_box_config": exact_box_config,
            "effective_audit_settings": effective_audit,
            "prior_results": prior_sources,
            "obstacle_discovery_source": {
                "path": str(obstacle_discovery_result_path),
                **discovery_expected,
                "read_only": True,
            },
            "archived_table1": {
                "path": str(archived_path),
                "file_sha256": ARCHIVED_FILE_SHA256,
                "payload_sha256": ARCHIVED_PAYLOAD_SHA256,
                "read_only": True,
            },
            "action_ledger": {
                "path": str(false_safe_result_path),
                "file_sha256": false_source["file_sha256"],
                "payload_sha256": false_source["result_payload_sha256"],
                "slurm_job_id": false_source["slurm_job_id"],
                "action_count": len(actions),
                "read_only": True,
            },
            "pairing": pairing,
            "probe_environment": {
                "main_disabled_image_observable_count": disabled_main_images,
                "probe_disabled_image_observable_count": disabled_probe_images,
                "osc_controller": "OSC_POSE",
                "control_frequency_hz": 20,
            },
            "geometry": {
                "accepted_robot_and_released_ee": geometry.geometry_record(env),
                "exact_obstacle_union": obstacle_union.geometry_record(env),
                "rounded_box_shell_m": shell_m,
            },
            "trigger_scan": {
                "selected_step": trigger_step,
                "scanned_step_count": len(scan_records),
                "immutable_prefix_action_count": len(prefix_hashes),
                "immutable_prefix_action_sha256": prefix_hashes,
                "records": scan_records,
            },
            "oracle_affine_audit": audit,
            "inflated_first_crossing_test_pass": passed,
            "stop_reason": stop_reason,
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
    parser.add_argument("--base-audit-config", type=Path, required=True)
    parser.add_argument("--exact-box-config", type=Path, required=True)
    parser.add_argument("--trigger-config", type=Path, required=True)
    parser.add_argument("--obstacle-discovery-result", type=Path, required=True)
    parser.add_argument("--fixed-margin-result", type=Path, required=True)
    parser.add_argument("--exact-box-result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(
        repo_root=args.repo_root.resolve(),
        manifest_path=args.manifest.resolve(),
        archived_path=args.archived.resolve(),
        false_safe_result_path=args.false_safe_result.resolve(),
        geometry_config_path=args.geometry_config.resolve(),
        base_audit_config_path=args.base_audit_config.resolve(),
        exact_box_config_path=args.exact_box_config.resolve(),
        trigger_config_path=args.trigger_config.resolve(),
        obstacle_discovery_result_path=args.obstacle_discovery_result.resolve(),
        fixed_margin_result_path=args.fixed_margin_result.resolve(),
        exact_box_result_path=args.exact_box_result.resolve(),
        expected_commit=args.expected_commit,
        output_path=args.output.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_step": result["trigger_scan"]["selected_step"],
                "inflated_first_crossing_test_pass": result[
                    "inflated_first_crossing_test_pass"
                ],
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
