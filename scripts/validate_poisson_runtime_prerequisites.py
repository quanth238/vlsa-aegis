#!/usr/bin/env python3
"""Allocation-backed prerequisite probe for the static full-body PSF pilot.

This script does not run a policy, AEGIS, or a Poisson safety filter.  It
restores one frozen SafeLIBERO state under both the released OSC_POSE
controller and Robosuite's JOINT_VELOCITY controller, then records the runtime
facts that the proposed pilot depends on:

* controller action dimensions and scaling;
* robot joint indexes and neutral-settling state differences;
* active-obstacle collision geoms from frozen MuJoCo authority; and
* availability of an arbitrary body-point translational Jacobian.

The output is written atomically and is implementation evidence only.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


SCHEMA_VERSION = "vlsa_poisson_runtime_prerequisites.v1"


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _array_sha256(value, np):
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(header + b"\n" + array.tobytes())


def _json_safe(value, np):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("non-finite floating-point value")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item, np) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, np) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), np)
    if isinstance(value, np.generic):
        return _json_safe(value.item(), np)
    return str(value)


def _atomic_json(path, payload, np):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            _json_safe(payload, np),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(partial), str(path))


def _git_record(repo_root):
    def run(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo_root)] + list(args),
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short").splitlines(),
    }


def _package_versions(names):
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - Python < 3.8 fallback
        import importlib_metadata as metadata

    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _load_case(manifest_path, case_id):
    matches = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("case_id") == case_id:
                matches.append((line_number, row))
    if len(matches) != 1:
        raise RuntimeError(
            "expected one manifest row for %s, found %d"
            % (case_id, len(matches))
        )
    return matches[0]


def _name(model, kind, index):
    method = getattr(model, "%s_id2name" % kind, None)
    if method is None:
        return None
    value = method(int(index))
    return None if value is None else str(value)


def _descends_from(model, body_id, root_body_id):
    current = int(body_id)
    seen = set()
    while current not in seen:
        if current == int(root_body_id):
            return True
        seen.add(current)
        if current == 0:
            return False
        current = int(model.body_parentid[current])
    raise RuntimeError("cycle in MuJoCo body-parent topology")


def _geom_records(env, root_body_id, np):
    model = env.sim.model
    records = []
    for geom_id in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[geom_id])
        if not _descends_from(model, body_id, root_body_id):
            continue
        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])
        records.append(
            {
                "geom_id": geom_id,
                "geom_name": _name(model, "geom", geom_id),
                "body_id": body_id,
                "body_name": _name(model, "body", body_id),
                "type": int(model.geom_type[geom_id]),
                "group": int(model.geom_group[geom_id]),
                "contype": contype,
                "conaffinity": conaffinity,
                "collision_enabled": bool(contype or conaffinity),
                "size": np.asarray(model.geom_size[geom_id], dtype=float),
                "world_position": np.asarray(
                    env.sim.data.geom_xpos[geom_id], dtype=float
                ),
                "world_rotation": np.asarray(
                    env.sim.data.geom_xmat[geom_id], dtype=float
                ).reshape(3, 3),
            }
        )
    return records


def _controller_record(env, np):
    robot = env.robots[0]
    controller = robot.controller
    low, high = robot.action_limits
    return {
        "name": str(controller.name),
        "environment_action_dim": int(env.env.action_dim),
        "arm_control_dim": int(controller.control_dim),
        "gripper_dof": int(robot.gripper.dof),
        "environment_action_low": np.asarray(low, dtype=float),
        "environment_action_high": np.asarray(high, dtype=float),
        "controller_input_min": np.asarray(controller.input_min, dtype=float),
        "controller_input_max": np.asarray(controller.input_max, dtype=float),
        "controller_output_min": np.asarray(controller.output_min, dtype=float),
        "controller_output_max": np.asarray(controller.output_max, dtype=float),
        "control_frequency_hz": float(env.env.control_freq),
        "control_timestep_seconds": float(env.env.control_timestep),
        "model_timestep_seconds": float(env.env.model_timestep),
        "physics_substeps_per_control_step": int(
            env.env.control_timestep / env.env.model_timestep
        ),
        "arm_qpos_indexes": [int(value) for value in robot._ref_joint_pos_indexes],
        "arm_qvel_indexes": [int(value) for value in robot._ref_joint_vel_indexes],
    }


def _arbitrary_point_jacobian(env, body_name, np):
    import mujoco

    model = env.sim.model
    data = env.sim.data
    body_id = int(model.body_name2id(body_name))
    point = np.asarray(data.body_xpos[body_id], dtype=float).copy()
    jacp = np.zeros((3, int(model.nv)), dtype=float)
    jacr = np.zeros((3, int(model.nv)), dtype=float)
    raw_model = getattr(model, "_model", model)
    raw_data = getattr(data, "_data", data)
    mujoco.mj_jac(raw_model, raw_data, jacp, jacr, point, body_id)
    robot = env.robots[0]
    arm_columns = [int(value) for value in robot._ref_joint_vel_indexes]
    arm_jacobian = jacp[:, arm_columns]
    return {
        "api": "mujoco.mj_jac",
        "body_name": body_name,
        "body_id": body_id,
        "world_point": point,
        "full_shape": list(jacp.shape),
        "arm_shape": list(arm_jacobian.shape),
        "arm_rank": int(np.linalg.matrix_rank(arm_jacobian)),
        "arm_frobenius_norm": float(np.linalg.norm(arm_jacobian)),
        "arm_jacobian": arm_jacobian,
    }


def _build_and_settle(evaluator, runtime, case, controller_name, np):
    suite_name = evaluator.normalize_suite_name(str(case["suite"]))
    benchmark_dict = runtime["benchmark"].get_benchmark_dict()
    task_suite = benchmark_dict[suite_name](
        safety_level=str(case["safety_level"])
    )
    runtime_index = evaluator._find_runtime_task_index(task_suite, case)
    task = task_suite.get_task(runtime_index)
    bddl_path = Path(task_suite.get_task_bddl_file_path(runtime_index))
    initial_states = task_suite.get_task_init_states(runtime_index)
    initial_state = np.asarray(
        initial_states[int(case["episode_index"])], dtype=float
    ).copy()

    runtime["np"].random.seed(int(case["environment_seed"]))
    env = runtime["OffScreenRenderEnv"](
        bddl_file_name=bddl_path,
        camera_heights=64,
        camera_widths=64,
        camera_depths=True,
        controller=controller_name,
    )
    try:
        env.seed(int(case["environment_seed"]))
        env.reset()
        observation = env.set_init_state(initial_state)
        state_before_settle = np.asarray(
            env.sim.get_state().flatten(), dtype=float
        ).copy()
        neutral_action = np.zeros(int(env.env.action_dim), dtype=float)
        for _ in range(int(case["settle_actions"])):
            observation, _, _, _ = env.step(neutral_action)
        state_after_settle = np.asarray(
            env.sim.get_state().flatten(), dtype=float
        ).copy()
        robot = env.robots[0]
        qpos_indexes = [int(value) for value in robot._ref_joint_pos_indexes]
        qvel_indexes = [int(value) for value in robot._ref_joint_vel_indexes]
        active_obstacle, candidates = evaluator._active_obstacle(
            env, observation
        )
        authority = evaluator._contact_model_authority(env, active_obstacle)
        root_body_id = int(authority["active_obstacle_root_body_id"])
        geoms = _geom_records(env, root_body_id, np)
        output = {
            "controller": _controller_record(env, np),
            "neutral_action": neutral_action,
            "state_before_settle_sha256": _array_sha256(
                state_before_settle, np
            ),
            "state_after_settle_sha256": _array_sha256(
                state_after_settle, np
            ),
            "state_after_settle": state_after_settle,
            "arm_qpos_after_settle": np.asarray(
                env.sim.data.qpos[qpos_indexes], dtype=float
            ),
            "arm_qvel_after_settle": np.asarray(
                env.sim.data.qvel[qvel_indexes], dtype=float
            ),
            "active_obstacle": {
                "name": active_obstacle,
                "candidates": candidates,
                "root_body_id": root_body_id,
                "root_body_name": _name(
                    env.sim.model, "body", root_body_id
                ),
                "collision_geoms": [
                    row for row in geoms if row["collision_enabled"]
                ],
                "noncollision_geoms": [
                    row for row in geoms if not row["collision_enabled"]
                ],
            },
            "link5_jacobian": _arbitrary_point_jacobian(
                env, "robot0_link5", np
            ),
            "link6_jacobian": _arbitrary_point_jacobian(
                env, "robot0_link6", np
            ),
        }
        return env, output
    except Exception:
        env.close()
        raise


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/vlsa_table1_population.jsonl"),
    )
    parser.add_argument(
        "--case-id", default="vlsa-t1-goal-ii-t0-e05"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path

    sys.path.insert(0, str(repo_root / "safelibero"))
    sys.path.insert(0, str(repo_root / "main"))
    evaluator = importlib.import_module("evaluate_safelibero_aegis")
    runtime = evaluator._runtime_imports(include_aegis=False)
    np = runtime["np"]
    line_number, case = _load_case(manifest_path, args.case_id)

    started = time.time()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "case_id": args.case_id,
        "manifest": {
            "path": str(manifest_path),
            "line_number": line_number,
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "provenance": {
            "git": _git_record(repo_root),
            "host": os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "packages": _package_versions(
                ["mujoco", "robosuite", "numpy", "scipy", "cvxpy", "osqp"]
            ),
        },
        "timing": {"started_unix": started},
    }
    osc_env = None
    joint_env = None
    try:
        osc_env, osc = _build_and_settle(
            evaluator, runtime, case, "OSC_POSE", np
        )
        joint_env, joint = _build_and_settle(
            evaluator, runtime, case, "JOINT_VELOCITY", np
        )
        state_difference = np.asarray(
            joint["state_after_settle"], dtype=float
        ) - np.asarray(osc["state_after_settle"], dtype=float)
        qpos_difference = np.asarray(
            joint["arm_qpos_after_settle"], dtype=float
        ) - np.asarray(osc["arm_qpos_after_settle"], dtype=float)

        checks = {
            "osc_action_dim_is_7": (
                osc["controller"]["environment_action_dim"] == 7
            ),
            "joint_velocity_action_dim_is_8": (
                joint["controller"]["environment_action_dim"] == 8
            ),
            "joint_velocity_has_seven_arm_controls": (
                joint["controller"]["arm_control_dim"] == 7
            ),
            "joint_velocity_output_is_half_rad_per_second": bool(
                np.allclose(
                    joint["controller"]["controller_output_min"], -0.5
                )
                and np.allclose(
                    joint["controller"]["controller_output_max"], 0.5
                )
            ),
            "active_obstacle_identity_matches_between_controllers": (
                osc["active_obstacle"]["name"]
                == joint["active_obstacle"]["name"]
            ),
            "active_obstacle_has_collision_geoms": bool(
                osc["active_obstacle"]["collision_geoms"]
            ),
            "link5_arbitrary_point_jacobian_available": (
                osc["link5_jacobian"]["arm_shape"] == [3, 7]
                and osc["link5_jacobian"]["arm_frobenius_norm"] > 0.0
            ),
            "link6_arbitrary_point_jacobian_available": (
                osc["link6_jacobian"]["arm_shape"] == [3, 7]
                and osc["link6_jacobian"]["arm_frobenius_norm"] > 0.0
            ),
        }
        payload.update(
            {
                "status": "complete",
                "case": case,
                "osc_pose": osc,
                "joint_velocity": joint,
                "controller_settling_comparison": {
                    "state_l2": float(np.linalg.norm(state_difference)),
                    "state_linf": float(
                        np.linalg.norm(state_difference, ord=np.inf)
                    ),
                    "arm_qpos_l2_mixed_rad": float(
                        np.linalg.norm(qpos_difference)
                    ),
                    "arm_qpos_linf_mixed_rad": float(
                        np.linalg.norm(qpos_difference, ord=np.inf)
                    ),
                    "states_identical": bool(
                        np.array_equal(
                            osc["state_after_settle"],
                            joint["state_after_settle"],
                        )
                    ),
                },
                "checks": checks,
                "all_prerequisite_checks_passed": all(checks.values()),
            }
        )
    except Exception as error:
        payload.update(
            {
                "status": "failed",
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            }
        )
    finally:
        if osc_env is not None:
            osc_env.close()
        if joint_env is not None:
            joint_env.close()
        payload["timing"].update(
            {
                "finished_unix": time.time(),
                "elapsed_seconds": time.time() - started,
            }
        )
        _atomic_json(args.output.resolve(), payload, np)

    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
