"""One-case paired CRFS oracle experiment derived from the baseline runner."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from crfs_harness.artifacts import atomic_write_json, content_hash, valid_completion
from crfs_harness.manifest import validate_case
from crfs_harness.official_state import verify_official_state_bindings

from .measurement import GeomClearanceMonitor, resolve_crfs_geom_groups
from .projection import ProjectionResult, solve_kinematic_projection

LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float64)


def _progress(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float64)
    return quat[:3] * 2.0 * math.acos(quat[3]) / denominator


def policy_observation(obs: dict[str, Any], prompt: str, resize_size: int) -> dict[str, Any]:
    from openpi_client import image_tools

    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return {
        "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size)),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist, resize_size, resize_size)
        ),
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], quat_to_axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
        "prompt": str(prompt),
    }


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _git_state(root: Path) -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    return commit, dirty


def _active_obstacle(env, observation: dict[str, Any]) -> str:
    candidates = []
    for name in sorted(env.env.objects_dict):
        if "obstacle" not in name or f"{name}_pos" not in observation:
            continue
        position = np.asarray(observation[f"{name}_pos"], dtype=np.float64)
        if position[2] > 0 and -0.5 < position[0] < 0.5 and -0.5 < position[1] < 0.5:
            candidates.append(name)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one active workspace obstacle, found {candidates}")
    return candidates[0]


@dataclass(frozen=True)
class OracleConfig:
    host: str
    port: int
    resize_size: int
    settle_steps: int
    executed_prefix: int
    action_horizon: int
    action_dim: int
    sampler_steps: int
    intervention_step: int
    safety_margin_m: float
    distance_limit_m: float
    eef_radius_m: float
    measurement_repeats: int
    stop_after_measurement: bool
    response_matrix_m_per_action: tuple[tuple[float, ...], ...] | None
    optimizer_max_iterations: int
    checkpoint_id: str
    checkpoint_sha256: str
    output_root: str
    run_id: str


def oracle_config_from_mapping(
    value: dict[str, Any],
    *,
    host: str,
    port: int,
    checkpoint_id: str,
    checkpoint_sha256: str,
    output_root: str,
    run_id: str,
) -> OracleConfig:
    """Build the runtime config once for single-case and batched launchers."""
    return OracleConfig(
        host=host,
        port=port,
        resize_size=int(value["resize_size"]),
        settle_steps=int(value["settle_steps"]),
        executed_prefix=int(value["executed_prefix"]),
        action_horizon=int(value["action_horizon"]),
        action_dim=int(value["action_dim"]),
        sampler_steps=int(value["sampler_steps"]),
        intervention_step=int(value["intervention_step"]),
        safety_margin_m=float(value["safety_margin_m"]),
        distance_limit_m=float(value["distance_limit_m"]),
        eef_radius_m=float(value["eef_radius_m"]),
        measurement_repeats=int(value.get("measurement_repeats", 2)),
        stop_after_measurement=bool(value.get("stop_after_measurement", False)),
        response_matrix_m_per_action=(
            tuple(tuple(float(item) for item in row) for row in value["response_matrix_m_per_action"])
            if value.get("response_matrix_m_per_action") is not None
            else None
        ),
        optimizer_max_iterations=int(value["optimizer_max_iterations"]),
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=checkpoint_sha256,
        output_root=output_root,
        run_id=run_id,
    )


class SafeLiberoCase:
    """Reset/replay wrapper that retains the baseline task and controller."""

    def __init__(self, case: dict[str, Any], config: OracleConfig) -> None:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self._generated_source = "source_estimand" in case
        self._official_state_bound = case.get("schema_version") == "3.0"
        if self._generated_source and self._official_state_bound:
            raise ValueError("A case cannot be both generated and official-state-bound")
        if self._official_state_bound:
            manifest_errors = validate_case(case)
            if manifest_errors:
                raise ValueError(
                    "Invalid official saved-state manifest record: "
                    + "; ".join(manifest_errors)
                )
        if self._generated_source:
            if case.get("source_estimand") != "task0_single_obstacle_generated_v1":
                raise ValueError("Unsupported generated source estimand")
            if case.get("safety_level") == "II":
                raise ValueError("Generated source states must never be labeled SafeLIBERO Level II")
            from crfs_oracle.generated_source import load_generated_source_bundle

            suite = benchmark.get_benchmark_dict()[case["task_suite"]]()
            self._generated_bundle = load_generated_source_bundle(case)
        else:
            suite = benchmark.get_benchmark_dict()[case["task_suite"]](safety_level=case["safety_level"])
            self._generated_bundle = None
        self._task = suite.get_task(int(case["task_index"]))
        self._suite = suite
        self._task_suite = str(case["task_suite"])
        self._safety_level = None if self._generated_source else str(case["safety_level"])
        self._task_index = int(case["task_index"])
        self._source_estimand = str(case["source_estimand"]) if self._generated_source else None
        self._init_states_root = (
            None if self._generated_source else Path(get_libero_path("init_states"))
        )
        self._init_state = (
            None
            if self._generated_source
            else self._load_released_init_state(case)
        )
        bddl_path = Path(get_libero_path("bddl_files")) / self._task.problem_folder / self._task.bddl_file
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=config.resize_size,
            camera_widths=config.resize_size,
            camera_depths=True,
            # The scene is immutable across paired branches. A soft reset
            # resets simulator/controller state without recompiling MuJoCo.
            hard_reset=False,
        )
        self.env.seed(int(case["environment_seed"]))
        self._environment_seed = int(case["environment_seed"])
        self.config = config
        self.obstacle_name: str | None = None
        self.eef_geoms: tuple[str, ...] = ()
        self.obstacle_geoms: tuple[str, ...] = ()

    def _released_init_state_path(self) -> Path:
        if self._generated_source or self._init_states_root is None:
            raise RuntimeError("Released init-state path requested for a generated source")
        filename = str(self._task.init_states_file)
        suffix = ".pruned_init"
        if not filename.endswith(suffix):
            raise ValueError(f"SafeLIBERO task has an unsupported init-state filename: {filename}")
        level_filename = f"{filename[:-len(suffix)]}_level_{self._safety_level}{suffix}"
        return self._init_states_root / str(self._task.problem_folder) / level_filename

    def _load_released_init_state(self, case: dict[str, Any]) -> np.ndarray:
        if self._generated_source:
            raise RuntimeError("Released init-state loader called for a generated source")
        states = self._suite.get_task_init_states(self._task_index)
        row = np.asarray(states[int(case["episode_index"])])
        if not self._official_state_bound:
            return row

        manifest_errors = validate_case(case)
        if manifest_errors:
            raise ValueError(
                "Invalid official saved-state manifest record: " + "; ".join(manifest_errors)
            )
        row = np.ascontiguousarray(row)
        binding_errors = verify_official_state_bindings(
            case,
            actual_task_name=str(self._task.name),
            init_states_root=self._init_states_root,
            init_state_path=self._released_init_state_path(),
            row=row,
        )
        if binding_errors:
            raise ValueError(
                "Official saved-state identity verification failed: "
                + "; ".join(binding_errors)
            )
        return row

    def configure_case(self, case: dict[str, Any]) -> None:
        """Select another saved state without recompiling the identical task."""
        generated_source = "source_estimand" in case
        if generated_source != self._generated_source:
            raise ValueError("Shared environment cannot mix released and generated state sources")
        official_state_bound = case.get("schema_version") == "3.0"
        if official_state_bound != self._official_state_bound:
            raise ValueError(
                "Shared environment cannot mix legacy released and official-state-bound records"
            )
        if self._generated_source:
            identity = (
                str(case["task_suite"]),
                int(case["task_index"]),
                str(case["source_estimand"]),
            )
            expected = (self._task_suite, self._task_index, self._source_estimand)
            if identity != expected:
                raise ValueError(f"Shared environment task/source mismatch: expected {expected}, got {identity}")
            if case.get("safety_level") == "II":
                raise ValueError("Generated source states must never be labeled SafeLIBERO Level II")
            from crfs_oracle.generated_source import load_generated_source_bundle

            self._generated_bundle = load_generated_source_bundle(case)
            self._environment_seed = int(case["environment_seed"])
            self.env.seed(self._environment_seed)
            self.obstacle_name = None
            self.eef_geoms = ()
            self.obstacle_geoms = ()
            return
        identity = (str(case["task_suite"]), str(case["safety_level"]), int(case["task_index"]))
        expected = (self._task_suite, self._safety_level, self._task_index)
        if identity != expected:
            raise ValueError(f"Shared environment task mismatch: expected {expected}, got {identity}")
        self._init_state = self._load_released_init_state(case)
        self._environment_seed = int(case["environment_seed"])
        self.env.seed(self._environment_seed)
        self.obstacle_name = None
        self.eef_geoms = ()
        self.obstacle_geoms = ()

    @property
    def prompt(self) -> str:
        return str(self._task.language)

    def reset_and_settle(self) -> dict[str, Any]:
        if self._generated_source:
            if self.config.settle_steps != 20:
                raise ValueError("Generated source replay requires the recorded 20-step settle history")
            if self._generated_bundle is None:
                raise RuntimeError("Generated source bundle is not loaded")
            from crfs_oracle.generated_source import restore_generated_source_branch

            self.env.seed(self._environment_seed)
            observation = restore_generated_source_branch(self.env, self._generated_bundle)
        else:
            self.env.seed(self._environment_seed)
            self.env.reset()
            self.env.set_init_state(self._init_state.copy())
            for _ in range(self.config.settle_steps):
                self.env.step_with_substep_callback(
                    LIBERO_DUMMY_ACTION.tolist(),
                    lambda _sim, _substep: None,
                    update_observables=False,
                    collect_observations=False,
                )
            # Sensor evaluation has no effect on physics. Render exactly once at
            # the settled branch point rather than on all 20 dummy control steps.
            self.env._update_observables(force=True)
            observation = self.env.env._get_observations()
        obstacle_name = _active_obstacle(self.env, observation)
        eef_geoms, obstacle_geoms = resolve_crfs_geom_groups(self.env, obstacle_name)
        if self._generated_source:
            from crfs_oracle.generated_source import verify_generated_source_branch

            errors = verify_generated_source_branch(
                self._generated_bundle,
                env=self.env,
                observation=observation,
                active_obstacle_name=obstacle_name,
            )
            if errors:
                raise RuntimeError(
                    "Generated source branch verification failed: " + "; ".join(errors)
                )
        self.obstacle_name = obstacle_name
        self.eef_geoms = eef_geoms
        self.obstacle_geoms = obstacle_geoms
        return observation

    def rollout(
        self,
        actions: np.ndarray,
        *,
        tracked_body_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Replay one action prefix, optionally tracking selected MuJoCo bodies.

        The default call preserves the original return surface.  Body tracking
        is opt-in so baseline H03--H05 callers do not pay for, or observe, the
        additional reach-diagnostic measurements.
        """
        prefix = np.asarray(actions, dtype=np.float64)
        if prefix.shape != (self.config.executed_prefix, 7):
            raise ValueError(f"Expected action prefix shape ({self.config.executed_prefix}, 7), got {prefix.shape}")
        tracked_names = None
        if tracked_body_names is not None:
            tracked_names = tuple(str(name) for name in tracked_body_names)
            if not tracked_names or any(not name for name in tracked_names):
                raise ValueError("tracked_body_names must contain non-empty names")
            if len(set(tracked_names)) != len(tracked_names):
                raise ValueError("tracked_body_names must be unique")
        observation = self.reset_and_settle()
        start_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
        eef_site_id = int(self.env.robots[0].eef_site_id)
        branch_eef = np.asarray(
            self.env.sim.data.site_xpos[eef_site_id], dtype=np.float64
        ).copy()
        site_rotation = np.asarray(self.env.sim.data.site_xmat[eef_site_id], dtype=np.float64).reshape(3, 3)
        start_eef_center = (
            branch_eef
            + site_rotation @ np.asarray((0.0, 0.0, -0.08), dtype=np.float64)
        )

        tracked_body_ids: dict[str, int] = {}
        tracked_branch_positions: dict[str, np.ndarray] = {}
        tracked_maximum_displacements: dict[str, float] = {}
        tracked_substep_samples = 0
        if tracked_names is not None:
            body_ids = getattr(self.env.env, "obj_body_id", None)
            if body_ids is None:
                raise RuntimeError("SafeLIBERO domain exposes no obj_body_id mapping")
            missing = [name for name in tracked_names if name not in body_ids]
            if missing:
                raise ValueError(f"Unknown tracked SafeLIBERO bodies: {missing}")
            tracked_body_ids = {name: int(body_ids[name]) for name in tracked_names}
            tracked_branch_positions = {
                name: np.asarray(
                    self.env.sim.data.body_xpos[body_id], dtype=np.float64
                ).copy()
                for name, body_id in tracked_body_ids.items()
            }
            tracked_maximum_displacements = {name: 0.0 for name in tracked_names}
        import mujoco

        box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
        branch_obstacle_boxes = []
        for name in self.obstacle_geoms:
            geom_id = int(self.env.sim.model.geom_name2id(name))
            if int(self.env.sim.model.geom_type[geom_id]) != box_type:
                continue
            branch_obstacle_boxes.append(
                {
                    "name": name,
                    "center_m": np.asarray(self.env.sim.data.geom_xpos[geom_id], dtype=np.float64).tolist(),
                    "rotation_world": np.asarray(
                        self.env.sim.data.geom_xmat[geom_id], dtype=np.float64
                    ).reshape(-1).tolist(),
                    "half_size_m": np.asarray(
                        self.env.sim.model.geom_size[geom_id], dtype=np.float64
                    ).tolist(),
                }
            )
        monitor = GeomClearanceMonitor(
            self.env.sim,
            self.eef_geoms,
            self.obstacle_geoms,
            distance_limit_m=self.config.distance_limit_m,
            eef_site_id=int(self.env.robots[0].eef_site_id),
            eef_center_offset_local_m=(0.0, 0.0, -0.08),
            eef_radius_m=self.config.eef_radius_m,
        )
        # Include the branch point itself before executing the first action.
        monitor.observe(self.env.sim, -1)
        done = False
        final_task_success = False
        task_success_during_prefix = False
        for action_index, action in enumerate(prefix):
            def observe_global_substep(sim, substep_index, *, _action_index=action_index):
                nonlocal tracked_substep_samples
                monitor.observe(sim, _action_index * 25 + int(substep_index))
                if tracked_names is not None:
                    for name, body_id in tracked_body_ids.items():
                        position = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)
                        displacement = float(
                            np.linalg.norm(position - tracked_branch_positions[name])
                        )
                        tracked_maximum_displacements[name] = max(
                            tracked_maximum_displacements[name], displacement
                        )
                    tracked_substep_samples += 1

            _, _, done, _ = self.env.step_with_substep_callback(
                action.tolist(),
                observe_global_substep,
                update_observables=False,
                collect_observations=False,
            )
            final_task_success = bool(done or self.env.check_success())
            task_success_during_prefix = bool(
                task_success_during_prefix or final_task_success
            )
        measurement = monitor.result()
        if measurement.conservative_clearance_m is None:
            raise RuntimeError("Controlled sphere/box metric found no obstacle box geoms")
        end_eef = np.asarray(
            self.env.sim.data.site_xpos[int(self.env.robots[0].eef_site_id)], dtype=np.float64
        ).copy()
        result = {
            # Eq. (3) in main.tex: signed obstacle distance at the conservative
            # EEF-sphere center, minus the preregistered EEF radius.
            "clearance_m": measurement.conservative_clearance_m,
            "raw_mujoco_clearance_m": measurement.min_clearance_m,
            "contact": measurement.contact,
            "measurement_samples": measurement.samples,
            "minimum_geom_pair": ("crfs_eef_sphere", measurement.conservative_obstacle_geom),
            "raw_mujoco_minimum_geom_pair": measurement.min_pair,
            "measurement": measurement.to_dict(),
            "start_eef_m": start_eef.tolist(),
            "start_eef_center_m": start_eef_center.tolist(),
            "end_eef_m": end_eef.tolist(),
            "branch_obstacle_boxes": branch_obstacle_boxes,
            # Preserve the baseline final-state field and add a monotone prefix
            # diagnostic for R03A's pregrasp terminal criterion.
            "task_success": final_task_success,
            "task_success_during_prefix": task_success_during_prefix,
        }
        if tracked_names is not None:
            expected_substeps = int(prefix.shape[0]) * 25
            if tracked_substep_samples != expected_substeps:
                raise RuntimeError(
                    "tracked-body callback did not observe every physics substep: "
                    f"expected {expected_substeps}, got {tracked_substep_samples}"
                )
            bodies = {}
            for name, body_id in tracked_body_ids.items():
                end_position = np.asarray(
                    self.env.sim.data.body_xpos[body_id], dtype=np.float64
                ).copy()
                endpoint_displacement = float(
                    np.linalg.norm(end_position - tracked_branch_positions[name])
                )
                maximum_displacement = max(
                    tracked_maximum_displacements[name], endpoint_displacement
                )
                bodies[name] = {
                    "body_id": body_id,
                    "branch_world_m": tracked_branch_positions[name].tolist(),
                    "end_world_m": end_position.tolist(),
                    "maximum_displacement_m": maximum_displacement,
                    "endpoint_displacement_m": endpoint_displacement,
                }
            result["tracked_body_motion"] = {
                "substep_samples": tracked_substep_samples,
                "branch_eef_world_m": branch_eef.tolist(),
                "end_eef_world_m": end_eef.tolist(),
                "bodies": bodies,
            }
        return result

    def close(self) -> None:
        self.env.close()


def _trial(rollout: dict[str, Any], nominal_endpoint: np.ndarray) -> dict[str, Any]:
    endpoint = np.asarray(rollout["end_eef_m"], dtype=np.float64)
    return {
        **rollout,
        "safe": bool(float(rollout["clearance_m"]) >= 0.0 and not rollout["contact"]),
        "endpoint_error_m": float(np.linalg.norm(endpoint - nominal_endpoint)),
    }


def _random_equal_norm(correction: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_correction = rng.normal(size=correction.shape)
    random_correction -= random_correction.mean(axis=0, keepdims=True)
    norm = float(np.linalg.norm(random_correction))
    target_norm = float(np.linalg.norm(correction))
    if target_norm == 0.0:
        return np.zeros_like(correction)
    if norm == 0.0:
        raise RuntimeError("Degenerate random control direction")
    return random_correction * target_norm / norm


def _infer(client, observation: dict[str, Any], noise: np.ndarray, config: OracleConfig, **controls) -> dict:
    request = copy.deepcopy(observation)
    request["__crfs__"] = {
        "noise": noise,
        "intervention_step": config.intervention_step,
        "return_trace": True,
        **controls,
    }
    return client.infer(request)


def _determinism_check(first: dict, second: dict) -> dict[str, Any]:
    action_equal = bool(np.array_equal(np.asarray(first["actions"]), np.asarray(second["actions"])))
    trace_equal = bool(
        all(
            np.array_equal(np.asarray(first["crfs_trace"][key]), np.asarray(second["crfs_trace"][key]))
            for key in sorted(first["crfs_trace"])
        )
    )
    return {"policy_actions_exact": action_equal, "policy_trace_exact": trace_equal, "passed": action_equal and trace_equal}


def run_case(
    case: dict[str, Any],
    config: OracleConfig,
    *,
    repo_root: str | Path,
    client=None,
    environment: SafeLiberoCase | None = None,
) -> tuple[Path, str]:
    from openpi_client import websocket_client_policy

    root = Path(repo_root).resolve()
    output = Path(config.output_root) / config.run_id / str(case["case_id"]) / "results.json"
    measurement_output = output.parent / "measurement-audit.json"
    if config.stop_after_measurement and measurement_output.exists():
        with measurement_output.open(encoding="utf-8") as handle:
            existing_measurement = json.load(handle)
        if existing_measurement.get("status") == "passed":
            return measurement_output, "skipped_valid_measurement"
    if valid_completion(output):
        return output, "skipped_valid_completion"

    git_commit, git_dirty = _git_state(root)
    noise_rng = np.random.default_rng(int(case["policy_seed"]))
    noise = noise_rng.normal(size=(config.action_horizon, config.action_dim)).astype(np.float32)
    if client is None:
        client = websocket_client_policy.WebsocketClientPolicy(config.host, config.port)
    _progress("policy_client_connected", case_id=case["case_id"])
    owns_environment = environment is None
    if environment is None:
        environment = SafeLiberoCase(case, config)
    else:
        environment.configure_case(case)
    _progress("safelibero_environment_ready", case_id=case["case_id"])
    try:
        initial_observation = environment.reset_and_settle()
        _progress("branch_state_ready", obstacle=environment.obstacle_name)
        policy_input = policy_observation(initial_observation, environment.prompt, config.resize_size)
        nominal_reply = _infer(client, policy_input, noise, config, intervention_mode="none")
        _progress("nominal_policy_first_complete")
        repeated_reply = _infer(client, policy_input, noise, config, intervention_mode="none")
        _progress("nominal_policy_replay_complete")
        determinism = _determinism_check(nominal_reply, repeated_reply)
        if not determinism["passed"]:
            raise RuntimeError(f"Fixed observation/noise policy replay is not exact: {determinism}")

        nominal_actions = np.asarray(nominal_reply["actions"], dtype=np.float64)[: config.executed_prefix, :7]
        if config.measurement_repeats < 2:
            raise ValueError("measurement_repeats must be at least two")
        measurement_rollouts = [environment.rollout(nominal_actions) for _ in range(config.measurement_repeats)]
        nominal_rollout = measurement_rollouts[0]
        repeated_rollout = measurement_rollouts[1]
        expected_samples = 1 + config.executed_prefix * 25
        clearance_values = np.asarray(
            [float(rollout["clearance_m"]) for rollout in measurement_rollouts], dtype=np.float64
        )
        endpoint_values = np.asarray(
            [rollout["end_eef_m"] for rollout in measurement_rollouts], dtype=np.float64
        )
        simulator_deterministic = bool(
            np.max(np.ptp(endpoint_values, axis=0)) <= 1e-9
            and float(np.ptp(clearance_values)) <= 1e-9
        )
        sample_counts_correct = all(
            int(rollout["measurement_samples"]) == expected_samples for rollout in measurement_rollouts
        )
        # A conservative proxy may become negative before physical contact.
        # The unsafe converse is forbidden: physical contact with positive
        # conservative clearance would prove that the sphere is not conservative.
        contact_conservative = all(
            not bool(rollout["contact"]) or float(rollout["clearance_m"]) <= 1e-4
            for rollout in measurement_rollouts
        )
        measurement_passed = simulator_deterministic and sample_counts_correct and contact_conservative
        measurement_audit = {
            "schema_version": "1.0",
            "gate": "H03",
            "status": "passed" if measurement_passed else "failed",
            "case_id": case["case_id"],
            "run_id": config.run_id,
            "obstacle_name": environment.obstacle_name,
            "expected_samples": expected_samples,
            "repeats": config.measurement_repeats,
            "max_clearance_variation_m": float(np.ptp(clearance_values)),
            "max_endpoint_coordinate_variation_m": float(np.max(np.ptp(endpoint_values, axis=0))),
            "simulator_deterministic": simulator_deterministic,
            "sample_counts_correct": sample_counts_correct,
            "contact_conservative": contact_conservative,
            "raw_mujoco_tracker_advisory_only": True,
            "runs": measurement_rollouts,
        }
        atomic_write_json(measurement_output, measurement_audit)
        _progress(
            "nominal_simulator_replay_complete",
            clearance_m=float(nominal_rollout["clearance_m"]),
            raw_mujoco_clearance_m=float(nominal_rollout["raw_mujoco_clearance_m"]),
            contact=bool(nominal_rollout["contact"]),
            raw_mj_contact_consistent=bool(
                nominal_rollout["measurement"]["raw_mj_contact_consistent"]
            ),
            min_pair=nominal_rollout["minimum_geom_pair"],
            repeats=config.measurement_repeats,
        )
        if not simulator_deterministic:
            raise RuntimeError("Reset + initial-state + settle replay is not deterministic")
        if not sample_counts_correct:
            raise RuntimeError(
                f"Physics-substep audit expected {expected_samples} samples in every repeat"
            )
        if not contact_conservative:
            raise RuntimeError(
                "H03 conservative proxy missed a physical contact; refusing projection"
            )
        if config.stop_after_measurement:
            return measurement_output, "measurement_only_completed"
        nominal_endpoint = np.asarray(nominal_rollout["end_eef_m"], dtype=np.float64)

        if nominal_rollout["clearance_m"] >= 0.0 and not nominal_rollout["contact"]:
            repair = ProjectionResult(
                feasible=True,
                actions=nominal_actions.tolist(),
                correction=np.zeros((config.executed_prefix, 3)).tolist(),
                objective=0.0,
                verified_clearance_m=float(nominal_rollout["clearance_m"]),
                endpoint_error_m=0.0,
                optimizer_success=True,
                optimizer_status=0,
                optimizer_message="nominal prefix already safe",
                evaluations=0,
            )
        else:
            _progress("projection_started")
            if config.response_matrix_m_per_action is None:
                raise RuntimeError("H05 projection requires the frozen H04 response matrix")
            repair = solve_kinematic_projection(
                nominal_actions,
                start_eef_center_m=np.asarray(nominal_rollout["start_eef_center_m"]),
                response_matrix=np.asarray(config.response_matrix_m_per_action),
                obstacle_boxes=nominal_rollout["branch_obstacle_boxes"],
                eef_radius_m=config.eef_radius_m,
                safety_margin_m=config.safety_margin_m,
                max_iterations=config.optimizer_max_iterations,
            )
            _progress(
                "projection_complete",
                feasible=repair.feasible,
                evaluations=repair.evaluations,
                clearance_m=repair.verified_clearance_m,
            )

        provenance = {
            "evidence_tier": "real_safelibero_preliminary",
            "research_limitations": [
                "released SafeLIBERO obstacles are movable rather than the preregistered static asymmetric-convex pilot",
            ],
            "d_opt_model": "frozen H04 response matrix plus static branch oriented-box geometry",
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "baseline_commit": "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b",
            "python_version": platform.python_version(),
            "host": socket.gethostname(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_suite": case["task_suite"],
            "safety_level": case["safety_level"],
            "task_index": case["task_index"],
            "episode_index": case["episode_index"],
            "environment_seed": case["environment_seed"],
            "policy_seed": case["policy_seed"],
            "random_control_seed": case["random_control_seed"],
            "group_id": case["group_id"],
            "checkpoint_id": config.checkpoint_id,
            "checkpoint_sha256": config.checkpoint_sha256,
            "noise_sha256": _array_hash(noise),
            "sampler_steps": config.sampler_steps,
            "intervention_step": config.intervention_step,
            "action_frame": "world-frame OSC translation delta",
            "correction_space": "physical",
            "obstacle_name": environment.obstacle_name,
            "eef_geoms": list(environment.eef_geoms),
            "obstacle_geoms": list(environment.obstacle_geoms),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "gpu_type": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "policy_determinism": determinism,
            "simulator_replay_exact": simulator_deterministic,
        }
        config_value = {**config.__dict__, "output_root": "<declared-output-root>"}
        base_result = {
            "schema_version": "1.0",
            "case_id": case["case_id"],
            "run_id": config.run_id,
            "config_hash": content_hash(config_value),
            "provenance": provenance,
            "repair": repair.to_dict(),
        }
        if not repair.feasible or repair.actions is None or repair.correction is None:
            result = {**base_result, "status": "infeasible", "trials": {"nominal": _trial(nominal_rollout, nominal_endpoint)}}
            atomic_write_json(output, result)
            return output, "infeasible"

        repaired_actions = np.asarray(repair.actions, dtype=np.float64)
        correction = np.asarray(repair.correction, dtype=np.float32)
        random_correction = _random_equal_norm(correction, int(case["random_control_seed"])).astype(np.float32)
        direct_rollout = environment.rollout(repaired_actions)
        _progress("direct_repair_rollout_complete", clearance_m=float(direct_rollout["clearance_m"]))
        oracle_reply = _infer(
            client,
            policy_input,
            noise,
            config,
            intervention_mode="residual",
            correction_space="physical",
            correction=correction,
        )
        random_reply = _infer(
            client,
            policy_input,
            noise,
            config,
            intervention_mode="residual",
            correction_space="physical",
            correction=random_correction,
        )
        bridge_reply = _infer(
            client,
            policy_input,
            noise,
            config,
            intervention_mode="bridge_edit",
            correction_space="physical",
            correction=correction,
        )
        oracle_rollout = environment.rollout(np.asarray(oracle_reply["actions"])[: config.executed_prefix, :7])
        random_rollout = environment.rollout(np.asarray(random_reply["actions"])[: config.executed_prefix, :7])
        bridge_rollout = environment.rollout(np.asarray(bridge_reply["actions"])[: config.executed_prefix, :7])
        _progress("paired_intervention_rollouts_complete")
        result = {
            **base_result,
            "status": "completed",
            "trials": {
                "nominal": _trial(nominal_rollout, nominal_endpoint),
                "direct_repair": _trial(direct_rollout, nominal_endpoint),
                "random_residual": _trial(random_rollout, nominal_endpoint),
                "oracle_residual": _trial(oracle_rollout, nominal_endpoint),
                "bridge_edit": _trial(bridge_rollout, nominal_endpoint),
            },
        }
        atomic_write_json(output, result)
        return output, "completed"
    finally:
        if owns_environment:
            environment.close()
