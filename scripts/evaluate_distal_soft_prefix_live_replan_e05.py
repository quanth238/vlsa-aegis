#!/usr/bin/env python3
"""Execute one registered contact-free prefix, then release to live pi0.5."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from scripts.evaluate_distal_smooth_field_attribution_e05 import (
    _archived_actions,
    _public,
    _validate_registered,
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


RESULT_SCHEMA = "vlsa_distal_soft_prefix_live_replan_e05_result.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


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
    _require(value.get("protocol_id") == "vlsa-distal-soft-prefix-live-replan-e05-v1", "live-replan protocol differs")
    _require(value.get("case_ids") == [CASE_ID], "live-replan case differs")
    state = value["state_protocol"]
    _require(state["repulsive_prefix_steps"] == list(range(182, 187)), "repulsive prefix differs")
    _require(int(state["live_replanning_start_step"]) == 187, "live release differs")
    _require(int(state["first_live_policy_query_index"]) == 37, "live query differs")
    _require(int(state["execute_actions_per_query"]) == 5, "replan interval differs")
    _require(value["control"]["additional_l5_l7_intervention_after_prefix"] is False, "distal intervention must be disabled after prefix")
    output = json.loads(_canonical(value).decode())
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


class InternalSafetyMonitor:
    """Read-only all-substep physical and ellipsoid diagnostics."""

    def __init__(self, env: Any, geometry: Any, obstacle_name: str, obstacle_reference: Any) -> None:
        import numpy as np
        from main.multilink_ellipsoid.sitl_candidate import _obstacle_root_body_id

        self.env = env
        self.geometry = geometry
        self.obstacle_name = str(obstacle_name)
        self.reference = np.asarray(obstacle_reference, dtype=np.float64).reshape(3)
        self.obstacle_id = _obstacle_root_body_id(env.sim.model, self.obstacle_name)
        self.total_samples = 0
        self.minimum_proxy_margin_m: Optional[float] = None
        self.minimum_exact_slack: Optional[float] = None
        self.exact_overlap_samples = 0
        self.protected_contact_samples = 0
        self.protected_contact_events: list[dict[str, Any]] = []
        self.maximum_displacement_m = 0.0
        self.first_protected_contact = None
        self.first_exact_overlap = None

    def _measure(self, step: int, substep: int, record: dict[str, Any]) -> None:
        import numpy as np
        from main.multilink_ellipsoid.moka_response_field import compiled_box_ellipsoids, multi_primitive_link_margins
        from main.multilink_ellipsoid.obstacle_proxy_audit import compiled_obstacle_boxes, evaluate_obstacle_representations
        from main.multilink_ellipsoid.sitl_candidate import _protected_contact_evidence

        links = self.geometry._slabbed_links(self.env)
        boxes = compiled_obstacle_boxes(self.env, self.obstacle_name)
        proxy = np.asarray(multi_primitive_link_margins(links, compiled_box_ellipsoids(boxes)), dtype=np.float64)
        exact = evaluate_obstacle_representations(links, self.geometry.obstacle, boxes)
        evidence = _protected_contact_evidence(self.env, self.obstacle_name)
        obstacle = np.asarray(self.env.sim.data.xpos[self.obstacle_id], dtype=np.float64)
        displacement = float(np.sum(np.abs(obstacle - self.reference)))
        proxy_min = float(np.min(proxy))
        exact_min = float(exact["compiled_box_union_minimum_normalized_radial_slack"])
        overlap = bool(exact["compiled_box_union_any_exact_solid_overlap"])
        contact = bool(evidence["events"])
        self.total_samples += 1
        self.minimum_proxy_margin_m = proxy_min if self.minimum_proxy_margin_m is None else min(self.minimum_proxy_margin_m, proxy_min)
        self.minimum_exact_slack = exact_min if self.minimum_exact_slack is None else min(self.minimum_exact_slack, exact_min)
        self.maximum_displacement_m = max(self.maximum_displacement_m, displacement)
        record["minimum_proxy_margin_m"] = min(record["minimum_proxy_margin_m"], proxy_min)
        record["minimum_exact_normalized_radial_slack"] = min(record["minimum_exact_normalized_radial_slack"], exact_min)
        record["maximum_active_obstacle_l1_displacement_m"] = max(record["maximum_active_obstacle_l1_displacement_m"], displacement)
        if overlap:
            self.exact_overlap_samples += 1
            record["exact_overlap_samples"] += 1
            if self.first_exact_overlap is None:
                self.first_exact_overlap = {"step": int(step), "substep": int(substep)}
        if contact:
            self.protected_contact_samples += 1
            record["protected_contact_samples"] += 1
            if self.first_protected_contact is None:
                self.first_protected_contact = {"step": int(step), "substep": int(substep)}
            for event in evidence["events"]:
                self.protected_contact_events.append({"step": int(step), "substep": int(substep), **_public(event)})

    def execute(self, command: Any, *, step: int) -> tuple[Any, float, bool, Any, dict[str, Any]]:
        import numpy as np

        action = np.asarray(command, dtype=np.float64)
        record = {
            "step": int(step), "substep_count": 0,
            "minimum_proxy_margin_m": float("inf"),
            "minimum_exact_normalized_radial_slack": float("inf"),
            "exact_overlap_samples": 0, "protected_contact_samples": 0,
            "maximum_active_obstacle_l1_displacement_m": 0.0,
        }
        original_step = self.env.sim.step

        def instrumented_step(*args: Any, **kwargs: Any) -> Any:
            output = original_step(*args, **kwargs)
            record["substep_count"] += 1
            self._measure(step, record["substep_count"], record)
            return output

        self.env.sim.step = instrumented_step
        try:
            observation, reward, done, info = self.env.step(action.tolist())
        finally:
            self.env.sim.step = original_step
        return observation, float(reward), bool(done), info, record

    def summary(self) -> dict[str, Any]:
        return {
            "sample_count": self.total_samples,
            "minimum_proxy_margin_m": self.minimum_proxy_margin_m,
            "minimum_exact_normalized_radial_slack": self.minimum_exact_slack,
            "exact_overlap_sample_count": self.exact_overlap_samples,
            "protected_contact_sample_count": self.protected_contact_samples,
            "protected_contact_events": self.protected_contact_events,
            "first_exact_overlap": self.first_exact_overlap,
            "first_protected_contact": self.first_protected_contact,
            "maximum_active_obstacle_l1_displacement_m": self.maximum_displacement_m,
        }


def evaluate(
    *, repo_root: Path, manifest_path: Path, archived_path: Path,
    controllability_path: Path, geometry_config_path: Path,
    experiment_config_path: Path, expected_commit: str, host: str,
    port: int, output_path: Path,
) -> dict[str, Any]:
    import numpy as np
    from main.evaluate_safelibero_aegis import (
        TABLE_RENDER_RESOLUTION, TABLE_SETTLE_ACTIONS, TABLE_VIDEO_FPS,
        _active_obstacle, _aegis_action, _build_environment,
        _contact_model_authority, _detailed_active_obstacle_contacts,
        _eef_proxy, _goal_progress_definition, _goal_progress_snapshot,
        _goal_progress_summary, _is_protected_event, _policy_observation,
        _processed_image, _runtime_imports, _server_identity, _settle,
        array_sha256, max_steps_for_case, pairing_record, query_seed,
        read_jsonl, translational_action, validate_case_row,
    )
    from main.multilink_ellipsoid.shadow import MultilinkEllipsoidShadow, allocation_record, load_shadow_config

    started = time.perf_counter_ns()
    config = load_config(experiment_config_path)
    registered = config["registered_inputs"]["controllability_result"]
    _require(_file_sha256(controllability_path) == registered["file_sha256"], "controllability file differs")
    control = _validate_registered(controllability_path, registered, "vlsa_distal_controllability_manifold_e05_result.v1")
    source = _git_identity(repo_root, expected_commit)
    allocation = allocation_record()
    archived = _load(archived_path)
    _require(_file_sha256(archived_path) == ARCHIVED_FILE_SHA256, "Table-1 file differs")
    _require(archived.get("result_payload_sha256") == ARCHIVED_PAYLOAD_SHA256, "Table-1 payload differs")
    _require(len(archived["actions"]) == EXPECTED_ACTION_HORIZON, "Table-1 horizon differs")
    rows = [row for row in read_jsonl(manifest_path) if row.get("case_id") == CASE_ID]
    _require(len(rows) == 1, "manifest case differs")
    case = rows[0]
    validate_case_row(case, repo_root)
    runtime = _runtime_imports(include_aegis=True)
    geometry_config = load_shadow_config(geometry_config_path)
    candidates = control["search"]["continuous"][config["registered_inputs"]["selected_arm"]]["internal_finalists"]
    target = np.asarray(config["registered_inputs"]["selected_coefficients"], dtype=np.float64)
    selected_rows = [row for row in candidates if np.array_equal(np.asarray(row["coefficients"], dtype=np.float64), target)]
    _require(len(selected_rows) == 1, "registered repulsive candidate differs")
    selected = selected_rows[0]
    _require(selected["protected_contact_count"] == 0, "registered candidate is not raw-contact-free")
    _require(selected["maximum_active_obstacle_l1_displacement_m"] <= float(config["measurement"]["paper_car_threshold_m"]), "registered candidate fails CAR")
    candidate_actions = np.asarray(selected["actions"], dtype=np.float64)
    _require(candidate_actions.shape == (20, 7), "registered action shape differs")
    prefix = candidate_actions[:5].copy()
    prefix_expected_end = [sample for sample in selected["samples"] if int(sample["action_offset"]) == 4][-1]
    nominal_prefix_end = [sample for sample in control["nominal_internal"]["samples"] if int(sample["action_offset"]) == 4][-1]
    expected_prefix_displacement = float(np.linalg.norm(np.asarray(prefix_expected_end["eef_position_m"]) - np.asarray(nominal_prefix_end["eef_position_m"])))

    env = None
    video_writer = None
    video_partial = output_path.with_name("episode.partial.mp4")
    video_final = output_path.with_name("episode.mp4")
    final_jpg = output_path.with_name("final.jpg")
    try:
        env, task, observation, selected_initial_state = _build_environment(runtime, case, render_resolution=TABLE_RENDER_RESOLUTION)
        observation = _settle(env, observation, TABLE_SETTLE_ACTIONS)
        obstacle_name, _ = _active_obstacle(env, observation)
        initial_obstacle_position = np.asarray(observation["%s_pos" % obstacle_name], dtype=np.float64).copy()
        pairing = pairing_record(case=case, selected_initial_state=selected_initial_state, settled_observation=observation, task_description=str(task.language), active_obstacle_name=obstacle_name, settled_simulator_state=np.asarray(env.sim.get_state().flatten()))
        for key in ("manifest_row_sha256", "initial_state_sha256", "initial_observation_sha256", "settled_simulator_state_sha256", "settled_active_obstacle_position_sha256", "policy_noise_schedule_sha256"):
            _require(pairing[key] == archived["pairing"][key], "pairing differs: %s" % key)
        perception = archived["perception"]
        geometry = MultilinkEllipsoidShadow.from_aegis_geometry(geometry_config, {"p2": perception["mvee_center"], "R2": perception["mvee_rotation"], "Q2_diag": perception["mvee_semiaxes"], "record": {"label": perception["obstacle_label"]}})
        monitor = InternalSafetyMonitor(env, geometry, obstacle_name, initial_obstacle_position)
        contact_authority = _contact_model_authority(env, obstacle_name)
        goal_definition, goal_atoms = _goal_progress_definition(env)
        initial_goal = _goal_progress_snapshot(env, goal_atoms, step=-1, previous_values=None)
        previous_goal_values = initial_goal["values"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = runtime["imageio"].get_writer(str(video_partial), fps=TABLE_VIDEO_FPS, codec="libx264", macro_block_size=None, pixelformat="yuv420p", output_params=["-crf", "18", "-movflags", "+faststart"])
        terminal_frame = _processed_image(observation, "agentview_image")
        video_writer.append_data(terminal_frame)
        action_records = []

        def finish_action(action: Any, step: int, source_name: str, reward: float, done: bool, internal: Optional[Mapping[str, Any]]) -> None:
            nonlocal observation, previous_goal_values, terminal_frame
            goal = _goal_progress_snapshot(env, goal_atoms, step=step, previous_values=previous_goal_values)
            previous_goal_values = goal["values"]
            _require(goal["all_satisfied"] is bool(done), "goal vector and done differ")
            contacts = _detailed_active_obstacle_contacts(env, obstacle_name, step=step, contact_authority=contact_authority)
            protected = [event for event in contacts["events"] if event.get("other", {}).get("classification") == "robot" and _is_protected_event(event)]
            displacement = float(np.sum(np.abs(np.asarray(observation["%s_pos" % obstacle_name], dtype=np.float64) - initial_obstacle_position)))
            terminal_frame = _processed_image(observation, "agentview_image")
            video_writer.append_data(terminal_frame)
            action_records.append({"step": int(step), "source": source_name, "action": np.asarray(action).tolist(), "reward": reward, "done": bool(done), "goal_progress": goal, "boundary_protected_contact_events": _public(protected), "active_obstacle_l1_displacement_m": displacement, "internal": None if internal is None else _public(internal)})

        for step in range(182):
            action = _archived_actions(archived, step, step)[0]
            observation, reward, done, _ = env.step(action.tolist())
            finish_action(action, step, "immutable_archived_aegis_prefix", float(reward), bool(done), None)
            _require(not done, "archived prefix completed early")
        for offset, action in enumerate(prefix):
            step = 182 + offset
            observation, reward, done, _, internal = monitor.execute(action, step=step)
            finish_action(action, step, "registered_contact_free_repulsive_prefix", reward, done, internal)
            _require(not done, "repulsive prefix completed task before release")
        observed_prefix_end = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
        prefix_replay_error = float(np.max(np.abs(observed_prefix_end - np.asarray(prefix_expected_end["eef_position_m"]))))

        client = runtime["websocket_client_policy"].WebsocketClientPolicy(host, int(port))
        server_identity = _server_identity(client)
        proxy = _eef_proxy(runtime, observation)
        p2 = np.asarray(perception["mvee_center"], dtype=np.float64)
        z_fixed = p2 - np.asarray(proxy["p1"], dtype=np.float64)
        z_fixed /= np.linalg.norm(z_fixed)
        released_geometry = {"p2": p2, "R2": np.asarray(perception["mvee_rotation"]), "Q2_diag": np.asarray(perception["mvee_semiaxes"]), "z_fixed": z_fixed}
        q1_diag = np.asarray([0.06, 0.12, 0.11])
        plan: collections.deque[Any] = collections.deque()
        policy_queries = []
        live_aegis_records = []
        query_index = int(config["state_protocol"]["first_live_policy_query_index"])
        for step in range(187, max_steps_for_case(case)):
            if not plan:
                seed = query_seed(int(case["policy_noise_seed"]), query_index)
                policy_input = _policy_observation(runtime, observation, task_description=str(task.language), resize_size=224, rng_seed=seed)
                query_started = time.perf_counter_ns()
                response = client.infer(policy_input)
                query_wall = (time.perf_counter_ns() - query_started) * 1.0e-9
                returned = np.asarray(response["actions"], dtype=np.float64)
                _require(returned.shape == (int(case["model_action_horizon"]), 7), "live chunk differs")
                plan.extend(returned[index].copy() for index in range(int(case["replan_steps"])))
                policy_queries.append({"query_index": query_index, "rng_seed": seed, "returned_actions_sha256": array_sha256(returned), "wall_seconds": query_wall, "server_timing": response.get("server_timing")})
                query_index += 1
            raw = np.asarray(plan.popleft(), dtype=np.float64)
            nominal_action = translational_action(raw)
            executed, qp_record = _aegis_action(runtime, nominal_translational=nominal_action, proxy=proxy, geometry=released_geometry, q1_diag=q1_diag, diagnostics_enabled=True)
            observation, reward, done, _, internal = monitor.execute(executed, step=step)
            finish_action(executed, step, "fresh_frozen_pi05_released_aegis", reward, done, internal)
            live_aegis_records.append({"step": step, "nominal_raw": raw.tolist(), "nominal_translational": nominal_action, "executed": executed, "qp": qp_record})
            proxy = _eef_proxy(runtime, observation)
            if done:
                break

        video_writer.close()
        video_writer = None
        video_partial.replace(video_final)
        runtime["imageio"].imwrite(str(final_jpg), terminal_frame)
        goal_summary = _goal_progress_summary(initial_goal, action_records)
        native_success = goal_summary["first_all_satisfied_step"] is not None
        safety = monitor.summary()
        car_pass = safety["maximum_active_obstacle_l1_displacement_m"] <= float(config["measurement"]["paper_car_threshold_m"])
        contact_pass = safety["protected_contact_sample_count"] == 0
        primary = bool(native_success and car_pass and contact_pass)
        interpretation = "one_repulsive_prefix_then_live_vla_replanning_safe_task_success" if primary else "live_vla_replanning_returns_to_protected_collision" if not contact_pass else "contact_free_but_paper_car_failed" if not car_pass else "physically_safe_but_live_vla_task_recovery_failed"
        result = {
            "schema_version": RESULT_SCHEMA, "status": "complete", "scientific_result": True,
            "case_id": CASE_ID, "claim_scope": config["claim_scope"], "source": source, "allocation": allocation, "config": config,
            "archived_table1": {"path": str(archived_path), "file_sha256": ARCHIVED_FILE_SHA256, "payload_sha256": ARCHIVED_PAYLOAD_SHA256, "read_only": True},
            "registered_controllability": {"path": str(controllability_path), "file_sha256": _file_sha256(controllability_path), "payload_sha256": control["result_payload_sha256"]},
            "pairing": pairing, "policy_server": server_identity,
            "repulsive_prefix": {"coefficients": target.tolist(), "actions": prefix.tolist(), "actions_sha256": array_sha256(prefix), "correction_l2_action": selected["correction_l2_action"], "registered_twenty_action_terminal_eef_error_m": selected["terminal_eef_error_m"], "expected_immediate_eef_displacement_from_nominal_prefix_m": expected_prefix_displacement, "observed_end_eef_position_m": observed_prefix_end.tolist(), "replay_max_abs_error_m": prefix_replay_error},
            "release": {"step": 187, "aegis_z_fixed_reinitialized": z_fixed.tolist(), "additional_distal_intervention": False},
            "policy_query_count": len(policy_queries), "policy_queries": policy_queries, "live_aegis_records": live_aegis_records,
            "action_count": len(action_records), "actions": action_records,
            "goal_progress": {**goal_definition, "initial": initial_goal, "summary": goal_summary},
            "physical_safety_after_activation": safety, "ellipsoid_diagnostic_pass": safety["exact_overlap_sample_count"] == 0,
            "raw_contact_pass": contact_pass, "paper_car_pass": car_pass, "native_task_success": native_success,
            "native_task_success_step": goal_summary["first_all_satisfied_step"], "primary_problem_solved": primary,
            "interpretation": interpretation,
            "safe_set_pilot": {"membership_observed": primary, "object": config["safe_set_interpretation"]["object"], "formal_invariance_claimed": False},
            "video": {"path": str(video_final), "file_sha256": _file_sha256(video_final), "frames_written": len(action_records) + 1, "fps": TABLE_VIDEO_FPS},
            "final_jpg": {"path": str(final_jpg), "file_sha256": _file_sha256(final_jpg)},
            "wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
        }
        result["result_payload_sha256"] = _sha256(_canonical(result))
        return result
    finally:
        if video_writer is not None:
            video_writer.close()
        if env is not None:
            env.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--controllability-result", type=Path, required=True)
    parser.add_argument("--geometry-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(repo_root=args.repo_root.resolve(), manifest_path=args.manifest.resolve(), archived_path=args.archived.resolve(), controllability_path=args.controllability_result.resolve(), geometry_config_path=args.geometry_config.resolve(), experiment_config_path=args.experiment_config.resolve(), expected_commit=args.expected_commit, host=args.host, port=args.port, output_path=args.output.resolve())
    _atomic_write(args.output.resolve(), result)
    print(json.dumps({"interpretation": result["interpretation"], "primary_problem_solved": result["primary_problem_solved"], "task_success": result["native_task_success"], "raw_contact_pass": result["raw_contact_pass"], "paper_car_pass": result["paper_car_pass"], "result_payload_sha256": result["result_payload_sha256"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
