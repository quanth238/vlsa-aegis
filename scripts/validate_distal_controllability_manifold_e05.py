#!/usr/bin/env python3
"""Independent arithmetic and fresh-rollout validator for the E05 oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_controllability_manifold_e05 import (
    PhysicalProbe,
    _archived_action,
    _atomic_write,
    _score_record,
    load_config,
)
from scripts.evaluate_distal_counterfactual_field_e05 import _disable_images
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    EXPECTED_ACTION_HORIZON,
    _file_sha256,
    _load,
    _require,
    _sha256,
)


SCHEMA = "vlsa_distal_controllability_manifold_e05_validation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _close(left: float, right: float, tolerance: float = 1.0e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=float(tolerance))


def _manual_gate(record: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "exact_box_nonoverlap": int(record["exact_overlap_sample_count"]) == 0
        and float(record["minimum_exact_normalized_radial_slack"]) >= 0.0,
        "zero_protected_contact": int(record["protected_contact_count"]) == 0,
        "paper_car": float(record["maximum_active_obstacle_l1_displacement_m"])
        <= float(gate["paper_car_threshold_m"]),
        "terminal_eef": float(record["terminal_eef_error_m"])
        <= float(gate["maximum_terminal_eef_error_m"]),
        "task_progress": float(record["task_progress_ratio"])
        >= float(gate["minimum_task_progress_ratio"]),
    }


def validate(*, repo_root: Path, manifest_path: Path, archived_path: Path, geometry_config_path: Path, experiment_config_path: Path, result_path: Path, expected_commit: str) -> dict[str, Any]:
    import numpy as np
    from main.evaluate_safelibero_aegis import TABLE_RENDER_RESOLUTION, TABLE_SETTLE_ACTIONS, _active_obstacle, _build_environment, _runtime_imports, _settle, read_jsonl, validate_case_row
    from main.multilink_ellipsoid.detour_manifold import finite_soft_library
    from main.multilink_ellipsoid.shadow import MultilinkEllipsoidShadow, allocation_record, load_shadow_config
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe, _obstacle_root_body_id

    result = _load(result_path)
    _require(result.get("schema_version") == "vlsa_distal_controllability_manifold_e05_result.v1", "result schema differs")
    payload = dict(result)
    claimed = payload.pop("result_payload_sha256")
    _require(_sha256(_canonical(payload)) == claimed, "result payload hash differs")
    _require(result["source"]["commit"] == expected_commit, "result commit differs")
    _require(not bool(result["source"]["dirty"]), "result source is dirty")
    config = load_config(experiment_config_path)
    _require(result["config"]["config_payload_sha256"] == config["config_payload_sha256"], "result config differs")
    _require(len(finite_soft_library()) == int(config["manifold"]["finite_library_count"]), "finite library count differs")
    control = result["controllability"]
    control_checks = {
        "initial_exact_safe": not bool(control["initial_exact_overlap"]),
        "initial_contact_free": int(control["initial_protected_contact_count"]) == 0,
        "command_influence_precedes_violation": int(control["first_influence_sample_index"]) < int(control["first_physical_violation_sample_index"]),
        "cartesian_authority": float(control["maximum_previolation_link_center_change_m"]) >= float(config["authority_audit"]["minimum_previolation_link_center_change_m"]),
        "genuine_future_physical_violation": int(control["nominal_protected_contact_count"]) > 0,
    }
    _require(control["gate"]["checks"] == control_checks, "controllability checks differ")
    _require(bool(control["gate"]["pass"]) is bool(all(control_checks.values())), "controllability gate differs")
    for section in list(result["search"].get("continuous", {}).values()) + ([result["search"]["finite_library"]] if result["search"].get("finite_library") else []):
        passed = 0
        for row in section.get("internal_finalists", []):
            checks = _manual_gate(row, config["gate"])
            _require(row["gate"]["checks"] == checks, "candidate checks differ")
            _require(bool(row["gate"]["pass"]) is bool(all(checks.values())), "candidate gate differs")
            passed += int(bool(row["gate"]["pass"]))
        _require(int(section.get("safe_support_count", 0)) == passed, "safe support count differs")
        _require(bool(section.get("pass", False)) is bool(passed > 0), "section pass differs")

    # Freshly replay the best internally verified finalist in each search arm.
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    actions_ledger = archived["actions"]
    _require(len(actions_ledger) == EXPECTED_ACTION_HORIZON, "Table-1 horizon differs")
    matches = [row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "validator manifest case differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(geometry_config_path)
    env = None
    probe_env = None
    replayed = []
    try:
        env, task, observation, selected = _build_environment(runtime, case, render_resolution=TABLE_RENDER_RESOLUTION)
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        probe_env, probe_task, probe_observation, probe_selected = _build_environment(runtime, case, render_resolution=32)
        probe_observation = _settle(probe_env, probe_observation, TABLE_SETTLE_ACTIONS)
        _require(np.array_equal(np.asarray(selected), np.asarray(probe_selected)), "validator initial state differs")
        obstacle_name, _ = _active_obstacle(env, observation)
        obstacle_id = _obstacle_root_body_id(env.sim.model, obstacle_name)
        obstacle_reference = np.asarray(env.sim.data.xpos[obstacle_id], dtype=np.float64).copy()
        perception = archived["perception"]
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(geometry_config, {"p2": perception["mvee_center"], "R2": perception["mvee_rotation"], "Q2_diag": perception["mvee_semiaxes"], "record": {"label": perception["obstacle_label"]}})
        one_step = SlabbedEightConstraintProbe(probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name)
        probe = PhysicalProbe(one_step, obstacle_name, obstacle_reference)
        for step in range(182):
            observation, _, done, _ = env.step(np.asarray(_archived_action(actions_ledger, step), dtype=np.float64).tolist())
            _require(not bool(done), "validator episode ended early")
        nominal = result["nominal_internal"]
        start_eef = np.asarray(nominal["samples"][0]["eef_position_m"], dtype=np.float64)
        target_eef = np.asarray(nominal["terminal_eef_position_m"], dtype=np.float64)
        nominal_delta = target_eef - start_eef
        sections = [("continuous_%s" % name, value) for name, value in result["search"].get("continuous", {}).items()]
        if result["search"].get("finite_library"):
            sections.append(("finite_library", result["search"]["finite_library"]))
        for name, section in sections:
            rows = section.get("internal_finalists", [])
            if not rows:
                continue
            stored = min(rows, key=lambda item: float(item["objective"]))
            fresh = probe.rollout(env, np.asarray(stored["actions"], dtype=np.float64), internal=True, step_base=182)
            fresh = _score_record(fresh, np.asarray(stored["effective_correction"], dtype=np.float64), start_eef, target_eef, nominal_delta, config)
            keys = ("minimum_proxy_margin_m", "minimum_exact_normalized_radial_slack", "exact_overlap_sample_count", "protected_contact_count", "maximum_active_obstacle_l1_displacement_m", "terminal_eef_error_m", "task_progress_ratio")
            for key in keys:
                if isinstance(stored[key], int):
                    _require(int(fresh[key]) == int(stored[key]), "fresh %s %s differs" % (name, key))
                else:
                    _require(_close(float(fresh[key]), float(stored[key]), 1.0e-10), "fresh %s %s differs" % (name, key))
            checks = _manual_gate(fresh, config["gate"])
            _require(bool(all(checks.values())) is bool(stored["gate"]["pass"]), "fresh candidate decision differs")
            replayed.append({"arm": name, "stored_pass": bool(stored["gate"]["pass"]), "fresh_pass": bool(all(checks.values())), "fresh_minimum_exact_normalized_radial_slack": float(fresh["minimum_exact_normalized_radial_slack"]), "fresh_protected_contact_count": int(fresh["protected_contact_count"])})
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()
    output = {
        "schema_version": SCHEMA,
        "status": "complete",
        "allocation": allocation_record(),
        "result_path": str(result_path),
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": claimed,
        "controllability_gate_recomputed": bool(all(control_checks.values())),
        "fresh_replays": replayed,
        "fresh_replay_count": len(replayed),
        "validation_pass": True,
    }
    output["validation_payload_sha256"] = _sha256(_canonical(output))
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = validate(repo_root=args.repo_root.resolve(), manifest_path=args.manifest.resolve(), archived_path=args.archived.resolve(), geometry_config_path=args.geometry_config.resolve(), experiment_config_path=args.experiment_config.resolve(), result_path=args.result.resolve(), expected_commit=args.expected_commit)
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({"validation_pass": value["validation_pass"], "fresh_replay_count": value["fresh_replay_count"], "validation_payload_sha256": value["validation_payload_sha256"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
