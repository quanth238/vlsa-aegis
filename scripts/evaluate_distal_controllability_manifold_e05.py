#!/usr/bin/env python3
"""No-learning controllability and task-relative detour oracle at E05/182."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_counterfactual_field_e05 import (
    _archived_action,
    _disable_images,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    ARCHIVED_FILE_SHA256,
    ARCHIVED_PAYLOAD_SHA256,
    CASE_ID,
    EXPECTED_ACTION_HORIZON,
    _file_sha256,
    _git_identity,
    _load,
    _require,
    _sha256,
)


RESULT_SCHEMA = "vlsa_distal_controllability_manifold_e05_result.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _public(value: Any) -> Any:
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    _require(value.get("protocol_id") == "vlsa-distal-controllability-manifold-e05-v1", "manifold protocol differs")
    _require(value.get("case_ids") == [CASE_ID], "manifold case differs")
    _require(value.get("state") == {"step": 182, "evaluation_horizon_actions": 20, "corrected_prefix_actions": 5, "expected_mujoco_substeps_per_action": 25}, "manifold state differs")
    _require(value["manifold"]["continuous_arms"] == ["soft_free_5d", "endpoint_preserving_4d"], "manifold arms differ")
    _require(int(value["manifold"]["finite_library_count"]) == 54, "manifold library count differs")
    output = json.loads(_canonical(value).decode())
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


class PhysicalProbe:
    def __init__(self, one_step_probe: Any, obstacle_name: str, obstacle_reference: Any) -> None:
        self.one_step_probe = one_step_probe
        self.obstacle_name = str(obstacle_name)
        self.obstacle_reference = obstacle_reference

    @property
    def env(self) -> Any:
        return self.one_step_probe.probe_env

    def rollout(self, main_env: Any, actions: Any, *, internal: bool, step_base: int = 182) -> dict[str, Any]:
        import numpy as np
        from main.multilink_ellipsoid.moka_response_field import compiled_box_ellipsoids, multi_primitive_link_margins
        from main.multilink_ellipsoid.obstacle_proxy_audit import compiled_obstacle_boxes, evaluate_obstacle_representations
        from main.multilink_ellipsoid.predictive_flow import _eef_site_id
        from main.multilink_ellipsoid.sitl_candidate import _obstacle_root_body_id, _protected_contact_evidence

        commands = np.asarray(actions, dtype=np.float64)
        _require(commands.shape == (20, 7), "manifold rollout actions differ")
        synchronization = self.one_step_probe.synchronize(main_env)
        obstacle_id = _obstacle_root_body_id(self.env.sim.model, self.obstacle_name)
        reference = np.asarray(self.obstacle_reference, dtype=np.float64).reshape(3)
        samples = []
        contact_events = []

        def measure(action_offset: int, substep: int) -> None:
            links = self.one_step_probe.geometry._slabbed_links(self.env)
            boxes = compiled_obstacle_boxes(self.env, self.obstacle_name)
            proxy = multi_primitive_link_margins(links, compiled_box_ellipsoids(boxes))
            exact = evaluate_obstacle_representations(links, self.one_step_probe.geometry.obstacle, boxes)
            evidence = _protected_contact_evidence(self.env, self.obstacle_name)
            sample_index = len(samples)
            for event in evidence["events"]:
                contact_events.append({"sample_index": sample_index, "action_offset": action_offset, "substep": substep, **event})
            obstacle = np.asarray(self.env.sim.data.xpos[obstacle_id], dtype=np.float64)
            samples.append({
                "sample_index": sample_index,
                "action_offset": int(action_offset),
                "substep": int(substep),
                "proxy_margins_m": proxy,
                "minimum_proxy_margin_m": float(np.min(proxy)),
                "exact_overlap": bool(exact["compiled_box_union_any_exact_solid_overlap"]),
                "exact_overlap_count": int(exact["compiled_box_union_exact_overlap_count"]),
                "minimum_exact_normalized_radial_slack": float(exact["compiled_box_union_minimum_normalized_radial_slack"]),
                "protected_contact_count": len(evidence["events"]),
                "link_centers_m": np.asarray([link.center for link in links], dtype=np.float64),
                "eef_position_m": np.asarray(self.env.sim.data.site_xpos[_eef_site_id(self.env)], dtype=np.float64).copy(),
                "active_obstacle_l1_displacement_m": float(np.sum(np.abs(obstacle - reference))),
            })

        self.env.sim.forward()
        measure(-1, -1)
        substep_counts = []
        started = time.perf_counter_ns()
        original_step = self.env.sim.step
        for action_offset, command in enumerate(commands):
            before = len(samples)
            if internal:
                def instrumented_step(*args: Any, **kwargs: Any) -> Any:
                    output = original_step(*args, **kwargs)
                    measure(action_offset, len(samples) - before)
                    return output
                self.env.sim.step = instrumented_step
                try:
                    self.env.step(command.tolist())
                finally:
                    self.env.sim.step = original_step
                count = len(samples) - before
            else:
                self.env.step(command.tolist())
                measure(action_offset, -1)
                count = 1
            substep_counts.append(count)
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        proxy = np.asarray([sample["proxy_margins_m"] for sample in samples], dtype=np.float64)
        centers = np.asarray([sample["link_centers_m"] for sample in samples], dtype=np.float64)
        overlaps = [sample for sample in samples if sample["exact_overlap"]]
        physical = [sample for sample in samples if sample["exact_overlap"] or sample["protected_contact_count"] > 0]
        return {
            "internal": bool(internal),
            "sample_count": len(samples),
            "substep_counts": substep_counts,
            "samples": samples,
            "minimum_proxy_margin_m": float(np.min(proxy)),
            "proxy_row_minima_m": np.min(proxy, axis=0),
            "minimum_exact_normalized_radial_slack": float(min(sample["minimum_exact_normalized_radial_slack"] for sample in samples)),
            "exact_overlap_sample_count": len(overlaps),
            "protected_contact_count": sum(1 for sample in samples if sample["protected_contact_count"] > 0),
            "protected_contact_events": contact_events,
            "physical_violation_sample_count": len(physical),
            "first_physical_violation_sample_index": None if not physical else int(physical[0]["sample_index"]),
            "first_proxy_violation_sample_index": next((int(sample["sample_index"]) for sample in samples if sample["minimum_proxy_margin_m"] < 0.0), None),
            "maximum_active_obstacle_l1_displacement_m": float(max(sample["active_obstacle_l1_displacement_m"] for sample in samples)),
            "terminal_eef_position_m": np.asarray(samples[-1]["eef_position_m"], dtype=np.float64),
            "link_centers_m": centers,
            "synchronization": synchronization,
            "env_step_wall_seconds": elapsed,
        }


def _candidate_actions(base_actions: Any, matrix: Any, coefficients: Any, action_limit: float, maximum_norm: float) -> tuple[Any, Any, Any, bool]:
    import numpy as np
    nominal = np.asarray(base_actions, dtype=np.float64)
    raw = np.asarray(matrix, dtype=np.float64).dot(np.asarray(coefficients, dtype=np.float64)).reshape(5, 3)
    norm = float(np.linalg.norm(raw))
    if norm > float(maximum_norm):
        raw *= float(maximum_norm) / norm
    actions = nominal.copy()
    actions[:5, :3] = np.clip(actions[:5, :3] + raw, -float(action_limit), float(action_limit))
    effective = actions[:5, :3] - nominal[:5, :3]
    return actions, raw, effective, bool(np.max(np.abs(effective - raw)) > 1.0e-12)


def _score_record(record: Mapping[str, Any], effective: Any, start_eef: Any, target_eef: Any, nominal_delta: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    terminal = np.asarray(record["terminal_eef_position_m"], dtype=np.float64)
    target = np.asarray(target_eef, dtype=np.float64)
    start = np.asarray(start_eef, dtype=np.float64)
    delta = np.asarray(nominal_delta, dtype=np.float64)
    denominator = float(np.dot(delta, delta))
    terminal_error = float(np.linalg.norm(terminal - target))
    progress = 0.0 if denominator <= 1.0e-12 else float(np.dot(terminal - start, delta) / denominator)
    output = dict(record)
    output["terminal_eef_error_m"] = terminal_error
    output["task_progress_ratio"] = progress
    output["correction_l2_action"] = float(np.linalg.norm(np.asarray(effective, dtype=np.float64)))
    return output


def _objective(record: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    gate = config["gate"]
    slack_penalty = 100.0 * max(0.0, -float(record["minimum_exact_normalized_radial_slack"]))
    contact_penalty = 10.0 * float(record["protected_contact_count"])
    car_penalty = 1000.0 * max(0.0, float(record["maximum_active_obstacle_l1_displacement_m"]) - float(gate["paper_car_threshold_m"]))
    terminal_penalty = 100.0 * max(0.0, float(record["terminal_eef_error_m"]) - float(gate["maximum_terminal_eef_error_m"]))
    progress_penalty = 10.0 * max(0.0, float(gate["minimum_task_progress_ratio"]) - float(record["task_progress_ratio"]))
    regularizer = 0.001 * float(record["correction_l2_action"])
    return slack_penalty + contact_penalty + car_penalty + terminal_penalty + progress_penalty + regularizer - 0.001 * float(record["minimum_exact_normalized_radial_slack"])


def evaluate(*, repo_root: Path, manifest_path: Path, archived_path: Path, geometry_config_path: Path, experiment_config_path: Path, expected_commit: str) -> dict[str, Any]:
    import numpy as np
    from scipy.optimize import differential_evolution
    from main.evaluate_safelibero_aegis import TABLE_RENDER_RESOLUTION, TABLE_SETTLE_ACTIONS, _active_obstacle, _build_environment, _runtime_imports, _settle, pairing_record, read_jsonl, validate_case_row
    from main.multilink_ellipsoid.detour_manifold import candidate_gate, controllability_gate, finite_soft_library, task_relative_basis
    from main.multilink_ellipsoid.obstacle_proxy_audit import compiled_obstacle_boxes
    from main.multilink_ellipsoid.shadow import MultilinkEllipsoidShadow, allocation_record, load_shadow_config
    from main.multilink_ellipsoid.sitl_candidate import SlabbedEightConstraintProbe, _obstacle_root_body_id

    started = time.perf_counter_ns()
    config = load_config(experiment_config_path)
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file hash differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    archived_actions = archived.get("actions")
    _require(isinstance(archived_actions, list) and len(archived_actions) == EXPECTED_ACTION_HORIZON, "Table-1 horizon differs")
    matches = [row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID]
    _require(len(matches) == 1, "manifold manifest case differs")
    case = matches[0]
    validate_case_row(case, repo_root)
    runtime = _runtime_imports(include_aegis=False)
    geometry_config = load_shadow_config(geometry_config_path)
    env = None
    probe_env = None
    try:
        env, task, observation, selected_initial_state = _build_environment(runtime, case, render_resolution=TABLE_RENDER_RESOLUTION)
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        probe_env, probe_task, probe_observation, probe_initial_state = _build_environment(runtime, case, render_resolution=32)
        probe_observation = _settle(probe_env, probe_observation, TABLE_SETTLE_ACTIONS)
        _require(str(task.language) == str(probe_task.language), "probe task differs")
        _require(np.array_equal(np.asarray(selected_initial_state), np.asarray(probe_initial_state)), "probe initial state differs")
        obstacle_name, _ = _active_obstacle(env, observation)
        obstacle_id = _obstacle_root_body_id(env.sim.model, obstacle_name)
        obstacle_reference = np.asarray(env.sim.data.xpos[obstacle_id], dtype=np.float64).copy()
        pairing = pairing_record(case=case, selected_initial_state=selected_initial_state, settled_observation=observation, task_description=str(task.language), active_obstacle_name=obstacle_name, settled_simulator_state=np.asarray(env.sim.get_state().flatten()))
        disabled_images = {"main": _disable_images(env), "probe": _disable_images(probe_env)}
        perception = archived["perception"]
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(geometry_config, {"p2": perception["mvee_center"], "R2": perception["mvee_rotation"], "Q2_diag": perception["mvee_semiaxes"], "record": {"label": perception["obstacle_label"]}})
        one_step = SlabbedEightConstraintProbe(probe_env, geometry, clearance_m=0.0, active_obstacle_name=obstacle_name)
        probe = PhysicalProbe(one_step, obstacle_name, obstacle_reference)
        step_target = int(config["state"]["step"])
        for step in range(step_target + 1):
            if step == step_target:
                base_actions = np.asarray([_archived_action(archived_actions, index) for index in range(step, step + 20)], dtype=np.float64)
                boxes = compiled_obstacle_boxes(env, obstacle_name)
                _require(len(boxes) == 15, "compiled Moka box count differs")
                break
            observation, _, done, _ = env.step(np.asarray(_archived_action(archived_actions, step), dtype=np.float64).tolist())
            _require(not bool(done), "episode ended before manifold state")

        basis = task_relative_basis(base_actions[:5, :3])
        nominal_internal = probe.rollout(env, base_actions, internal=True, step_base=step_target)
        expected = int(config["state"]["expected_mujoco_substeps_per_action"])
        _require(all(int(count) == expected for count in nominal_internal["substep_counts"]), "nominal internal substeps differ")
        start_eef = np.asarray(nominal_internal["samples"][0]["eef_position_m"], dtype=np.float64)
        target_eef = np.asarray(nominal_internal["terminal_eef_position_m"], dtype=np.float64)
        nominal_delta = target_eef - start_eef

        authority_records = []
        first_violation = nominal_internal["first_physical_violation_sample_index"]
        _require(first_violation is not None, "nominal continuation lacks registered physical violation")
        maximum_pre = 0.0
        first_influence = len(nominal_internal["samples"]) + 1
        authority = config["authority_audit"]
        for basis_index in range(5):
            for sign in (-1.0, 1.0):
                coefficients = np.zeros(5, dtype=np.float64)
                coefficients[basis_index] = sign * float(authority["coefficient_amplitude_action"])
                actions, raw, effective, clipped = _candidate_actions(base_actions, basis["soft_matrix"], coefficients, 1.0, float(config["manifold"]["maximum_effective_correction_l2_action"]))
                record = probe.rollout(env, actions, internal=True, step_base=step_target)
                difference = np.linalg.norm(np.asarray(record["link_centers_m"]) - np.asarray(nominal_internal["link_centers_m"]), axis=2)
                influenced = np.flatnonzero(np.max(difference, axis=1) > float(authority["influence_tolerance_m"]))
                first = len(record["samples"]) + 1 if influenced.size == 0 else int(influenced[0])
                pre = float(np.max(difference[: int(first_violation)]))
                maximum_pre = max(maximum_pre, pre)
                first_influence = min(first_influence, first)
                authority_records.append({"basis_index": basis_index, "basis_name": basis["soft_names"][basis_index], "sign": sign, "coefficients": coefficients, "effective_correction_l2_action": float(np.linalg.norm(effective)), "action_bound_clipped": clipped, "first_influence_sample_index": first, "maximum_previolation_link_center_change_m": pre, "minimum_proxy_margin_m": record["minimum_proxy_margin_m"], "minimum_exact_normalized_radial_slack": record["minimum_exact_normalized_radial_slack"], "physical_violation_sample_count": record["physical_violation_sample_count"]})
        control_record = {
            "initial_proxy_margin_m": float(nominal_internal["samples"][0]["minimum_proxy_margin_m"]),
            "initial_exact_overlap": bool(nominal_internal["samples"][0]["exact_overlap"]),
            "initial_protected_contact_count": int(nominal_internal["samples"][0]["protected_contact_count"]),
            "first_proxy_violation_sample_index": nominal_internal["first_proxy_violation_sample_index"],
            "first_physical_violation_sample_index": int(first_violation),
            "first_influence_sample_index": int(first_influence),
            "maximum_previolation_link_center_change_m": maximum_pre,
            "nominal_physical_violation_count": int(nominal_internal["physical_violation_sample_count"]),
            "nominal_exact_overlap_sample_count": int(nominal_internal["exact_overlap_sample_count"]),
            "nominal_protected_contact_count": int(nominal_internal["protected_contact_count"]),
            "nominal_minimum_proxy_margin_m": float(nominal_internal["minimum_proxy_margin_m"]),
            "nominal_minimum_exact_normalized_radial_slack": float(nominal_internal["minimum_exact_normalized_radial_slack"]),
            "authority_probes": authority_records,
        }
        control_record["gate"] = controllability_gate(control_record, authority)

        search_report: dict[str, Any] = {"attempted": False, "reason": "controllability_gate_failed", "continuous": {}, "finite_library": {}}
        rollout_count = 11
        if control_record["gate"]["pass"]:
            search_report = {"attempted": True, "reason": "completed", "continuous": {}, "finite_library": {}}
            bound = tuple(float(value) for value in config["manifold"]["coefficient_bounds_action"])
            finalist_count = int(config["manifold"]["internal_finalist_count_per_continuous_arm"])

            def run_arm(name: str, matrix: Any) -> dict[str, Any]:
                nonlocal rollout_count
                dimension = int(np.asarray(matrix).shape[1])
                cache: dict[tuple[float, ...], dict[str, Any]] = {}
                def evaluate_coefficients(values: Any) -> dict[str, Any]:
                    nonlocal rollout_count
                    key = tuple(float(value) for value in np.asarray(values, dtype=np.float64))
                    if key not in cache:
                        actions, raw, effective, clipped = _candidate_actions(base_actions, matrix, key, 1.0, float(config["manifold"]["maximum_effective_correction_l2_action"]))
                        boundary = probe.rollout(env, actions, internal=False, step_base=step_target)
                        rollout_count += 1
                        scored = _score_record(boundary, effective, start_eef, target_eef, nominal_delta, config)
                        scored.update({"coefficients": list(key), "raw_correction": raw, "effective_correction": effective, "action_bound_clipped": clipped, "objective": None, "actions": actions})
                        scored["objective"] = _objective(scored, config)
                        cache[key] = scored
                    return cache[key]
                optimize = differential_evolution(lambda values: float(evaluate_coefficients(values)["objective"]), bounds=[bound] * dimension, seed=int(config["manifold"]["differential_evolution_seed"]) + dimension, popsize=int(config["manifold"]["population_multiplier"]), maxiter=int(config["manifold"]["maximum_generations"]), polish=bool(config["manifold"]["polish"]), workers=1, updating="immediate")
                ranked = sorted(cache.values(), key=lambda item: float(item["objective"]))
                internal_rows = []
                for boundary in ranked[:finalist_count]:
                    internal = probe.rollout(env, boundary["actions"], internal=True, step_base=step_target)
                    rollout_count += 1
                    scored = _score_record(internal, boundary["effective_correction"], start_eef, target_eef, nominal_delta, config)
                    scored.update({"coefficients": boundary["coefficients"], "actions": boundary["actions"], "effective_correction": boundary["effective_correction"], "action_bound_clipped": boundary["action_bound_clipped"], "objective": _objective(scored, config)})
                    scored["gate"] = candidate_gate(scored, config["gate"])
                    internal_rows.append(scored)
                passed = [row for row in internal_rows if row["gate"]["pass"]]
                return {"dimension": dimension, "optimizer_success": bool(optimize.success), "optimizer_message": str(optimize.message), "boundary_evaluation_count": len(cache), "best_boundary": _public(ranked[0]), "internal_finalists": _public(internal_rows), "safe_support_count": len(passed), "pass": bool(passed)}

            search_report["continuous"]["soft_free_5d"] = run_arm("soft_free_5d", basis["soft_matrix"])
            search_report["continuous"]["endpoint_preserving_4d"] = run_arm("endpoint_preserving_4d", basis["endpoint_matrix"])

            library_boundary = []
            for item in finite_soft_library():
                actions, raw, effective, clipped = _candidate_actions(base_actions, basis["soft_matrix"], item["coefficients"], 1.0, float(config["manifold"]["maximum_effective_correction_l2_action"]))
                boundary = probe.rollout(env, actions, internal=False, step_base=step_target)
                rollout_count += 1
                scored = _score_record(boundary, effective, start_eef, target_eef, nominal_delta, config)
                scored.update({"library_index": item["index"], "family": item["family"], "coefficients": item["coefficients"], "effective_correction": effective, "actions": actions, "action_bound_clipped": clipped})
                scored["objective"] = _objective(scored, config)
                library_boundary.append(scored)
            ranked_library = sorted(library_boundary, key=lambda item: float(item["objective"]))
            library_internal = []
            for boundary in ranked_library:
                internal = probe.rollout(env, boundary["actions"], internal=True, step_base=step_target)
                rollout_count += 1
                scored = _score_record(internal, boundary["effective_correction"], start_eef, target_eef, nominal_delta, config)
                scored.update({"library_index": boundary["library_index"], "family": boundary["family"], "coefficients": boundary["coefficients"], "actions": boundary["actions"], "effective_correction": boundary["effective_correction"], "action_bound_clipped": boundary["action_bound_clipped"]})
                scored["objective"] = _objective(scored, config)
                scored["gate"] = candidate_gate(scored, config["gate"])
                library_internal.append(scored)
            library_passed = [row for row in library_internal if row["gate"]["pass"]]
            search_report["finite_library"] = {"candidate_count": len(library_boundary), "best_boundary": _public(ranked_library[0]), "internal_finalists": _public(library_internal), "safe_support_count": len(library_passed), "pass": bool(library_passed)}

        continuous_pass = any(bool(value.get("pass")) for value in search_report.get("continuous", {}).values())
        library_pass = bool(search_report.get("finite_library", {}).get("pass"))
        interpretation = "controllability_gate_failed" if not control_record["gate"]["pass"] else ("continuous_and_finite_detour_support" if continuous_pass and library_pass else "continuous_support_but_finite_library_misses" if continuous_pass else "tested_task_relative_manifold_has_no_verified_support")
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "scientific_result": True,
            "claim_scope": config["claim_scope"],
            "source": source,
            "allocation": allocation,
            "config": config,
            "pairing": pairing,
            "disabled_images": disabled_images,
            "basis": _public(basis),
            "nominal_internal": _public(nominal_internal),
            "controllability": _public(control_record),
            "search": _public(search_report),
            "continuous_manifold_support": continuous_pass,
            "finite_library_support": library_pass,
            "interpretation": interpretation,
            "rollout_count": rollout_count,
            "training_or_primary_control_attempted": False,
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = _sha256(_canonical(result))
        return result
    finally:
        if probe_env is not None:
            probe_env.close()
        if env is not None:
            env.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = evaluate(repo_root=args.repo_root.resolve(), manifest_path=args.manifest.resolve(), archived_path=args.archived.resolve(), geometry_config_path=args.geometry_config.resolve(), experiment_config_path=args.experiment_config.resolve(), expected_commit=args.expected_commit)
    _atomic_write(args.output.resolve(), value)
    print(json.dumps({"interpretation": value["interpretation"], "controllability_gate": value["controllability"]["gate"]["pass"], "continuous_manifold_support": value["continuous_manifold_support"], "finite_library_support": value["finite_library_support"], "rollout_count": value["rollout_count"], "result_payload_sha256": value["result_payload_sha256"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
