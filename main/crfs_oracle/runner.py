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
from typing import Any

import numpy as np

from crfs_harness.artifacts import atomic_write_json, content_hash, valid_completion

from .measurement import GeomClearanceMonitor, resolve_crfs_geom_groups
from .projection import ProjectionResult, solve_simulator_projection

LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float64)


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
    optimizer_max_iterations: int
    checkpoint_id: str
    checkpoint_sha256: str
    output_root: str
    run_id: str


class SafeLiberoCase:
    """Reset/replay wrapper that retains the baseline task and controller."""

    def __init__(self, case: dict[str, Any], config: OracleConfig) -> None:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[case["task_suite"]](safety_level=case["safety_level"])
        self._task = suite.get_task(int(case["task_index"]))
        self._init_state = np.asarray(suite.get_task_init_states(int(case["task_index"]))[int(case["episode_index"])])
        bddl_path = Path(get_libero_path("bddl_files")) / self._task.problem_folder / self._task.bddl_file
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=config.resize_size,
            camera_widths=config.resize_size,
            camera_depths=True,
        )
        self.env.seed(int(case["environment_seed"]))
        self._environment_seed = int(case["environment_seed"])
        self.config = config
        self.obstacle_name: str | None = None
        self.eef_geoms: tuple[str, ...] = ()
        self.obstacle_geoms: tuple[str, ...] = ()

    @property
    def prompt(self) -> str:
        return str(self._task.language)

    def reset_and_settle(self) -> dict[str, Any]:
        self.env.seed(self._environment_seed)
        self.env.reset()
        observation = self.env.set_init_state(self._init_state.copy())
        for _ in range(self.config.settle_steps):
            observation, _, _, _ = self.env.step(LIBERO_DUMMY_ACTION.tolist())
        obstacle_name = _active_obstacle(self.env, observation)
        eef_geoms, obstacle_geoms = resolve_crfs_geom_groups(self.env, obstacle_name)
        self.obstacle_name = obstacle_name
        self.eef_geoms = eef_geoms
        self.obstacle_geoms = obstacle_geoms
        return observation

    def rollout(self, actions: np.ndarray) -> dict[str, Any]:
        prefix = np.asarray(actions, dtype=np.float64)
        if prefix.shape != (self.config.executed_prefix, 7):
            raise ValueError(f"Expected action prefix shape ({self.config.executed_prefix}, 7), got {prefix.shape}")
        observation = self.reset_and_settle()
        start_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
        monitor = GeomClearanceMonitor(
            self.env.sim,
            self.eef_geoms,
            self.obstacle_geoms,
            distance_limit_m=self.config.distance_limit_m,
        )
        # Include the branch point itself before executing the first action.
        monitor.observe(self.env.sim, -1)
        done = False
        for action in prefix:
            observation, _, done, _ = self.env.step_with_substep_callback(action.tolist(), monitor.observe)
        measurement = monitor.result()
        end_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
        return {
            "clearance_m": measurement.min_clearance_m,
            "contact": measurement.contact,
            "measurement_samples": measurement.samples,
            "minimum_geom_pair": measurement.min_pair,
            "start_eef_m": start_eef.tolist(),
            "end_eef_m": end_eef.tolist(),
            "task_success": bool(done or self.env.check_success()),
        }

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


def run_case(case: dict[str, Any], config: OracleConfig, *, repo_root: str | Path) -> tuple[Path, str]:
    from openpi_client import websocket_client_policy

    root = Path(repo_root).resolve()
    output = Path(config.output_root) / config.run_id / str(case["case_id"]) / "results.json"
    if valid_completion(output):
        return output, "skipped_valid_completion"

    git_commit, git_dirty = _git_state(root)
    noise_rng = np.random.default_rng(int(case["policy_seed"]))
    noise = noise_rng.normal(size=(config.action_horizon, config.action_dim)).astype(np.float32)
    client = websocket_client_policy.WebsocketClientPolicy(config.host, config.port)
    environment = SafeLiberoCase(case, config)
    try:
        initial_observation = environment.reset_and_settle()
        policy_input = policy_observation(initial_observation, environment.prompt, config.resize_size)
        nominal_reply = _infer(client, policy_input, noise, config, intervention_mode="none")
        repeated_reply = _infer(client, policy_input, noise, config, intervention_mode="none")
        determinism = _determinism_check(nominal_reply, repeated_reply)
        if not determinism["passed"]:
            raise RuntimeError(f"Fixed observation/noise policy replay is not exact: {determinism}")

        nominal_actions = np.asarray(nominal_reply["actions"], dtype=np.float64)[: config.executed_prefix, :7]
        nominal_rollout = environment.rollout(nominal_actions)
        repeated_rollout = environment.rollout(nominal_actions)
        simulator_deterministic = bool(
            np.allclose(nominal_rollout["end_eef_m"], repeated_rollout["end_eef_m"], atol=1e-9, rtol=0.0)
            and math.isclose(
                float(nominal_rollout["clearance_m"]),
                float(repeated_rollout["clearance_m"]),
                abs_tol=1e-9,
                rel_tol=0.0,
            )
        )
        if not simulator_deterministic:
            raise RuntimeError("Reset + initial-state + settle replay is not deterministic")
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
            repair = solve_simulator_projection(
                nominal_actions,
                environment.rollout,
                safety_margin_m=config.safety_margin_m,
                max_iterations=config.optimizer_max_iterations,
            )

        provenance = {
            "evidence_tier": "real_safelibero_preliminary",
            "research_limitations": [
                "optimizer currently queries simulator geometry directly; D_opt/D_sim independence is not yet established",
                "released SafeLIBERO obstacles are movable rather than the preregistered static asymmetric-convex pilot",
            ],
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
        environment.close()
