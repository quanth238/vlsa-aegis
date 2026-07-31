#!/usr/bin/env python3
"""Allocation-backed exact replay and substep-observer parity canary.

This job does not execute a Poisson correction.  It establishes two necessary
preconditions before an active controller is allowed to run:

1. the unchanged OSC path exactly reproduces every historical post-step
   simulator-state hash from one completed Table-1 AEGIS rollout; and
2. the opt-in 1 x 25 substep callback path is exactly equivalent to the
   ordinary environment step for state, observations, reward, success, and
   executed horizon.

The callback only hashes state.  Field construction and CBF queries are added
in a later shadow stage after this zero-mutation apparatus gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, Mapping, Sequence, Tuple


SCHEMA_VERSION = "vlsa_poisson_shadow_parity.v1"
DEFAULT_CASE_ID = "vlsa-t1-goal-ii-t0-e05"


class ShadowParityError(RuntimeError):
    """A registered exact-parity invariant did not hold."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_record(root: Path) -> Dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root)] + list(arguments),
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short").splitlines(),
    }


def _gpu_inventory() -> Dict[str, Any]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total",
            "--format=csv,noheader,nounits",
        ],
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=30,
    )
    devices = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ShadowParityError("unexpected nvidia-smi row: %r" % line)
        devices.append(
            {
                "name": parts[0],
                "uuid": parts[1],
                "memory_total_mib": int(parts[2]),
            }
        )
    if not devices or not all("H100" in row["name"] for row in devices):
        raise ShadowParityError("the allocation does not expose only H100 GPUs")
    return {"devices": devices}


def _package_versions(names: Sequence[str]) -> Dict[str, Any]:
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _load_case(manifest: Path, case_id: str) -> Tuple[int, Dict[str, Any], str]:
    matches = []
    with manifest.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ShadowParityError(
                    "manifest line %d is invalid JSON" % line_number
                ) from error
            if row.get("case_id") == case_id:
                matches.append((line_number, row, _sha256(raw_line.rstrip(b"\r\n"))))
    if len(matches) != 1:
        raise ShadowParityError(
            "expected exactly one manifest row for %s, found %d"
            % (case_id, len(matches))
        )
    line_number, row, row_hash = matches[0]
    if row.get("split") != "bringup_canary" or row.get("study_partition") != "development":
        raise ShadowParityError("shadow parity is restricted to the bring-up canary")
    if row.get("settle_actions") != 20:
        raise ShadowParityError("the canary must retain exactly 20 settle actions")
    return line_number, row, row_hash


def _fingerprint(value: Any, evaluator: Any, np: Any) -> Any:
    """Return a deterministic, compact contract for an observation value."""

    if isinstance(value, Mapping):
        return {
            str(key): _fingerprint(item, evaluator, np)
            for key, item in sorted(value.items(), key=lambda row: str(row[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_fingerprint(item, evaluator, np) for item in value]
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "kind": "array",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": evaluator.array_sha256(array),
        }
    if isinstance(value, np.generic):
        return _fingerprint(value.item(), evaluator, np)
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ShadowParityError("observation contains a non-finite scalar")
        return value
    raise ShadowParityError("unsupported observation value %s" % type(value).__name__)


def _observation_sha256(observation: Mapping[str, Any], evaluator: Any, np: Any) -> str:
    return _sha256(_canonical(_fingerprint(observation, evaluator, np)))


def _prepare_environment(
    evaluator: Any,
    runtime: Mapping[str, Any],
    case: Mapping[str, Any],
    replay: Any,
) -> Tuple[Any, Any, Mapping[str, Any], Sequence[Sequence[str]], Sequence[bool]]:
    np = runtime["np"]
    env, task, observation, selected_initial_state = evaluator._build_environment(
        runtime,
        case,
        render_resolution=evaluator.TABLE_RENDER_RESOLUTION,
    )
    proxy = evaluator._eef_proxy(runtime, observation)
    evaluator._update_eef_marker(env, proxy)
    observation = evaluator._settle(env, observation, 20)
    obstacle_name, _ = evaluator._active_obstacle(env, observation)
    settled_state = np.asarray(env.sim.get_state().flatten(), dtype=float).copy()
    pairing = evaluator.pairing_record(
        case=case,
        selected_initial_state=selected_initial_state,
        settled_observation=observation,
        task_description=str(task.language),
        active_obstacle_name=obstacle_name,
        settled_simulator_state=settled_state,
    )
    observed = {
        "settled_simulator_state_sha256": pairing.get(
            "settled_simulator_state_sha256"
        ),
        "initial_observation_sha256": pairing.get("initial_observation_sha256"),
        "policy_noise_schedule_sha256": pairing.get(
            "policy_noise_schedule_sha256"
        ),
    }
    expected = {
        "settled_simulator_state_sha256": replay.settled_simulator_state_sha256,
        "initial_observation_sha256": replay.initial_observation_sha256,
        "policy_noise_schedule_sha256": replay.policy_noise_schedule_sha256,
    }
    if observed != expected:
        env.close()
        raise ShadowParityError(
            "settled pairing differs: expected %r, observed %r" % (expected, observed)
        )
    goal_definition, goal_atoms = evaluator._goal_progress_definition(env)
    del goal_definition
    initial_goal = evaluator._goal_progress_snapshot(
        env, goal_atoms, step=-1, previous_values=None
    )
    return env, task, observation, goal_atoms, initial_goal["values"]


def _check_step(
    *,
    evaluator: Any,
    env: Any,
    observation: Mapping[str, Any],
    reward: Any,
    done: Any,
    expected_step: Any,
    goal_atoms: Sequence[Sequence[str]],
    previous_goal_values: Sequence[bool],
    np: Any,
) -> Tuple[str, str, Sequence[bool]]:
    state_hash = evaluator.array_sha256(env.sim.get_state().flatten())
    if state_hash != expected_step.simulator_state_sha256:
        raise ShadowParityError(
            "simulator state differs at step %d: expected %s, observed %s"
            % (expected_step.step, expected_step.simulator_state_sha256, state_hash)
        )
    if float(reward) != expected_step.reward or bool(done) is not expected_step.done:
        raise ShadowParityError(
            "reward/done differs at step %d" % expected_step.step
        )
    goal = evaluator._goal_progress_snapshot(
        env,
        goal_atoms,
        step=expected_step.step,
        previous_values=previous_goal_values,
    )
    if tuple(goal["values"]) != expected_step.goal_values:
        raise ShadowParityError("goal vector differs at step %d" % expected_step.step)
    if goal["all_satisfied"] is not bool(done):
        raise ShadowParityError("goal vector and done disagree at step %d" % expected_step.step)
    observation_hash = _observation_sha256(observation, evaluator, np)
    return state_hash, observation_hash, goal["values"]


def _run_ordinary(
    evaluator: Any,
    runtime: Mapping[str, Any],
    case: Mapping[str, Any],
    replay: Any,
) -> Dict[str, Any]:
    env = None
    try:
        env, _, observation, goal_atoms, previous_goal = _prepare_environment(
            evaluator, runtime, case, replay
        )
        state_hashes = []
        observation_hashes = []
        for expected_step in replay.steps:
            observation, reward, done, _ = env.step(expected_step.action)
            state_hash, observation_hash, previous_goal = _check_step(
                evaluator=evaluator,
                env=env,
                observation=observation,
                reward=reward,
                done=done,
                expected_step=expected_step,
                goal_atoms=goal_atoms,
                previous_goal_values=previous_goal,
                np=runtime["np"],
            )
            state_hashes.append(state_hash)
            observation_hashes.append(observation_hash)
            proxy = evaluator._eef_proxy(runtime, observation)
            evaluator._update_eef_marker(env, proxy)
            if done and expected_step.step != len(replay.steps) - 1:
                raise ShadowParityError("ordinary replay terminated early")
        if not replay.steps[-1].done:
            raise ShadowParityError("registered canary did not terminate in task success")
        return {
            "executed_action_count": len(state_hashes),
            "expected_state_match_count": len(state_hashes),
            "state_sequence_sha256": _sha256(_canonical(state_hashes)),
            "observation_sequence_sha256": _sha256(_canonical(observation_hashes)),
            "terminal_simulator_state_sha256": state_hashes[-1],
            "state_hashes": state_hashes,
            "observation_hashes": observation_hashes,
        }
    finally:
        if env is not None:
            env.close()


def _run_callback(
    evaluator: Any,
    runtime: Mapping[str, Any],
    case: Mapping[str, Any],
    replay: Any,
    ordinary: Mapping[str, Any],
) -> Dict[str, Any]:
    env = None
    try:
        env, _, observation, goal_atoms, previous_goal = _prepare_environment(
            evaluator, runtime, case, replay
        )
        state_hashes = []
        observation_hashes = []
        substep_trace = []
        for expected_step in replay.steps:
            current_substeps = []

            def callback(sim: Any, substep_index: int) -> None:
                current_substeps.append(
                    {
                        "substep": int(substep_index),
                        "simulator_state_sha256": evaluator.array_sha256(
                            sim.get_state().flatten()
                        ),
                    }
                )

            observation, reward, done, _ = env.step_with_substep_callback(
                expected_step.action,
                callback,
                expected_substeps=25,
            )
            if len(current_substeps) != 25 or [row["substep"] for row in current_substeps] != list(range(25)):
                raise ShadowParityError(
                    "callback cadence differs at step %d" % expected_step.step
                )
            state_hash, observation_hash, previous_goal = _check_step(
                evaluator=evaluator,
                env=env,
                observation=observation,
                reward=reward,
                done=done,
                expected_step=expected_step,
                goal_atoms=goal_atoms,
                previous_goal_values=previous_goal,
                np=runtime["np"],
            )
            if state_hash != ordinary["state_hashes"][expected_step.step]:
                raise ShadowParityError(
                    "callback and ordinary states differ at step %d" % expected_step.step
                )
            if observation_hash != ordinary["observation_hashes"][expected_step.step]:
                raise ShadowParityError(
                    "callback and ordinary observations differ at step %d"
                    % expected_step.step
                )
            state_hashes.append(state_hash)
            observation_hashes.append(observation_hash)
            substep_trace.append(current_substeps)
            proxy = evaluator._eef_proxy(runtime, observation)
            evaluator._update_eef_marker(env, proxy)
            if done and expected_step.step != len(replay.steps) - 1:
                raise ShadowParityError("callback replay terminated early")
        return {
            "executed_action_count": len(state_hashes),
            "callback_count": sum(len(row) for row in substep_trace),
            "expected_callback_count": 25 * len(replay.steps),
            "state_sequence_sha256": _sha256(_canonical(state_hashes)),
            "observation_sequence_sha256": _sha256(_canonical(observation_hashes)),
            "substep_trace_sha256": _sha256(_canonical(substep_trace)),
            "first_substep": substep_trace[0][0],
            "last_substep": substep_trace[-1][-1],
            "terminal_simulator_state_sha256": state_hashes[-1],
        }
    finally:
        if env is not None:
            env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("manifests/vlsa_poisson_link56_aegis_car_109.v1.jsonl"))
    parser.add_argument("--historical-result-root", type=Path, required=True)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    root = arguments.repo_root.resolve()
    manifest = arguments.manifest
    if not manifest.is_absolute():
        manifest = root / manifest
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "safelibero"))
    sys.path.insert(0, str(root / "main"))

    from main.poisson_fullbody.contracts import publish_hashed_json
    from main.poisson_fullbody.shadow_replay import load_historical_action_replay

    started = time.time()
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        "scientific_result": False,
        "evidence_tier": "allocation_backed_exact_simulator_parity",
        "claim_limit": "apparatus validation only; no Poisson correction or safety efficacy was tested",
        "case_id": arguments.case_id,
        "timing": {"started_unix": started},
    }
    try:
        source = _git_record(root)
        if source["status_short"]:
            raise ShadowParityError("shadow parity requires a clean source tree")
        if not os.environ.get("SLURM_JOB_ID"):
            raise ShadowParityError("shadow parity must run inside a Slurm allocation")
        gpu = _gpu_inventory()
        line_number, case, row_hash = _load_case(manifest, arguments.case_id)
        historical_record = case.get("historical_aegis_result")
        if not isinstance(historical_record, dict):
            raise ShadowParityError("manifest lacks the historical AEGIS binding")
        relative = historical_record.get("source_relative_path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ShadowParityError("historical result relative path is invalid")
        historical_path = arguments.historical_result_root.resolve() / relative
        if _file_sha256(historical_path) != historical_record.get("raw_file_sha256"):
            raise ShadowParityError("historical result raw SHA-256 differs")
        replay = load_historical_action_replay(
            historical_path, expected_case_id=arguments.case_id
        )
        if replay.result_payload_sha256 != historical_record.get("result_payload_sha256"):
            raise ShadowParityError("historical result payload binding differs")

        evaluator = importlib.import_module("evaluate_safelibero_aegis")
        runtime = evaluator._runtime_imports(include_aegis=False)
        ordinary = _run_ordinary(evaluator, runtime, case, replay)
        callback = _run_callback(evaluator, runtime, case, replay, ordinary)
        historical_state_sequence = [
            step.simulator_state_sha256 for step in replay.steps
        ]
        expected_sequence_hash = _sha256(_canonical(historical_state_sequence))
        if not (
            ordinary["state_sequence_sha256"]
            == callback["state_sequence_sha256"]
            == expected_sequence_hash
        ):
            raise ShadowParityError("final state-sequence identities disagree")
        if callback["callback_count"] != callback["expected_callback_count"]:
            raise ShadowParityError("callback exposure is incomplete")
        payload.update(
            {
                "status": "passed",
                "provenance": {
                    "source": source,
                    "manifest_path": str(manifest),
                    "manifest_sha256": _file_sha256(manifest),
                    "manifest_line_number": line_number,
                    "manifest_row_sha256": row_hash,
                    "historical_result_path": str(historical_path),
                    "historical_result_file_sha256": replay.result_file_sha256,
                    "historical_result_payload_sha256": replay.result_payload_sha256,
                    "host": socket.gethostname(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
                    "python_executable": sys.executable,
                    "python_version": platform.python_version(),
                    "packages": _package_versions(("numpy", "mujoco", "robosuite", "scipy")),
                    "gpu": gpu,
                },
                "historical": replay.provenance(),
                "ordinary_replay": {
                    key: value
                    for key, value in ordinary.items()
                    if key not in {"state_hashes", "observation_hashes"}
                },
                "callback_replay": callback,
                "acceptance": {
                    "all_historical_post_step_states_exact": True,
                    "ordinary_and_callback_states_exact": True,
                    "ordinary_and_callback_observations_exact": True,
                    "reward_done_goal_exact": True,
                    "full_callback_exposure": True,
                    "ordinary_step_path_unmodified": True,
                },
            }
        )
    except Exception as error:
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    payload["timing"].update(
        {
            "finished_unix": time.time(),
            "elapsed_seconds": time.time() - started,
        }
    )
    publish_hashed_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}, sort_keys=True))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
