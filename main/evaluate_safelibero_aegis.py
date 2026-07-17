#!/usr/bin/env python3
"""Reproduce the released translational SafeLIBERO evaluation, one case at a time.

This runner is deliberately a thin evidence-producing wrapper around the
algorithm released in ``main/main_aegis_translational.py`` and
``main/utils.py``.  It supports the two Table-1 arms needed for a paired
reproduction:

* ``pi05``: nominal pi0.5 translation and gripper, with rotation zeroed.
* ``aegis``: the same nominal actions passed through the released
  six-variable translational CBF-QP.

The VLM selector is not re-run.  The AEGIS arm requires a Codex label frozen
against the exact post-settle agent-view array hash before the outcome run.
Simulator, OpenPI, GroundingDINO, CVXPY, and rendering imports intentionally
live inside runtime functions so protocol/unit tests can import this module
without those dependencies.
"""

from __future__ import annotations

import argparse
import collections
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence


RESULT_SCHEMA = "vlsa_table1_episode_result.v1"
LABEL_SCHEMA = "vlsa_table1_codex_label.v1"
CAPTURE_SCHEMA = "vlsa_table1_label_capture.v1"
MANIFEST_SCHEMA = "vlsa_table1_population_case.v1"
UPSTREAM_COMMIT = "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b"
PROTOCOL_ID = "vlsa-table1-translational-1600-v1"
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
PAPER_COLLISION_THRESHOLD_M = 0.001
TABLE_ENVIRONMENT_SEED = 7
TABLE_SETTLE_ACTIONS = 20
TABLE_RENDER_RESOLUTION = 1024
TABLE_POLICY_RESIZE = 224
TABLE_MODEL_ACTION_HORIZON = 10
TABLE_REPLAN_STEPS = 5
TABLE_VIDEO_FPS = 30
TRANSLATIONAL_FAIL_OPEN = "corrected_translational_nominal"
UPSTREAM_EMPTY_PERCEPTION_FALLBACK = "raw_nominal_including_rotation"
ALLOWED_LABELS = {
    "yellow rectangular book",
    "blue moka pot",
    "red mug",
    "white storage box",
    "black wine bottle",
    "red milk carton",
}
LONG_EXTRA_LABELS = {"gray rectangular binder"}
_GROUNDING_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
SUITE_MAX_STEPS = {
    "safelibero_spatial": 300,
    "safelibero_goal": 300,
    "safelibero_object": 300,
    "safelibero_long": 550,
}


class ProtocolError(RuntimeError):
    """The immutable experiment contract or source binding is invalid."""


class ApparatusError(RuntimeError):
    """The requested scientific arm could not be validly executed."""


class MethodFailure(RuntimeError):
    """The released method reached a validly observed hard failure."""


def _is_precontrol_geometry_failure(result: Mapping[str, Any]) -> bool:
    failure = result.get("method_failure")
    return bool(
        result.get("status") == "method_failure"
        and isinstance(failure, Mapping)
        and failure.get("component") == "aegis_geometry"
        and failure.get("phase") == "precontrol"
        and failure.get("step") == 0
        and failure.get("safety_by_no_execution") is True
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return the stable JSON representation used by result receipts."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    """Hash exact array dtype, shape, and C-order bytes.

    This is intentionally not a hash of an encoder-dependent PNG byte stream.
    The capture script and evaluator both use this function, so a frozen label
    is bound to the exact pixels consumed by the released perception pipeline.
    """

    import numpy as np

    value = np.ascontiguousarray(array)
    header = canonical_json_bytes(
        {"dtype": value.dtype.str, "shape": list(value.shape)}
    )
    digest = hashlib.sha256()
    digest.update(b"vlsa-table1-array-v1\0")
    digest.update(header)
    digest.update(b"\0")
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> str:
    """Write JSON atomically and return the payload SHA-256."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256_bytes(payload)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ProtocolError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(row, dict):
                raise ProtocolError(
                    f"{path}:{line_number}: row must be a JSON object"
                )
            rows.append(row)
    return rows


def normalize_suite_name(name: str) -> str:
    aliases = {
        "safelibero_10": "safelibero_long",
        "libero_10": "safelibero_long",
        "long": "safelibero_long",
        "spatial": "safelibero_spatial",
        "goal": "safelibero_goal",
        "object": "safelibero_object",
    }
    canonical = aliases.get(name, name)
    if canonical not in SUITE_MAX_STEPS:
        raise ProtocolError(f"unsupported SafeLIBERO suite: {name}")
    return canonical


def max_steps_for_case(case: Mapping[str, Any]) -> int:
    suite = normalize_suite_name(str(case["suite"]))
    registered = int(case.get("max_steps", SUITE_MAX_STEPS[suite]))
    expected = SUITE_MAX_STEPS[suite]
    if registered != expected:
        raise ProtocolError(
            f"{case.get('case_id')}: max_steps={registered}, expected {expected}"
        )
    return registered


def translational_action(action: Sequence[float]) -> list[float]:
    """Keep nominal XYZ and gripper while zeroing rotation."""

    if len(action) < 7:
        raise ApparatusError(f"pi0.5 returned {len(action)} action values, need 7")
    output = [0.0] * 7
    output[0:3] = [float(action[index]) for index in range(3)]
    output[6] = float(action[6])
    if not all(math.isfinite(value) for value in output):
        raise ApparatusError("pi0.5 returned a non-finite action")
    return output


def legacy_ets_steps(executed_action_count: int, task_success: bool) -> int:
    """Mirror the released zero-based ``t`` counter used for Table 1.

    On success, the released loop breaks before incrementing ``t``; therefore
    an episode succeeding on its Nth executed action records N-1.  A timeout
    records the full horizon.  We also publish ``executed_action_count`` so the
    physical number of applied controls is not lost.
    """

    if executed_action_count < 0:
        raise ValueError("executed_action_count must be non-negative")
    if task_success and executed_action_count:
        return executed_action_count - 1
    return executed_action_count


def query_seed(base_seed: int, query_index: int) -> int:
    value = int(base_seed) + int(query_index)
    if not 0 <= value <= 0xFFFFFFFF:
        raise ProtocolError("per-query policy seed is outside uint32")
    return value


def policy_noise_schedule(case: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze every query seed that could be consumed within the horizon."""

    max_steps = max_steps_for_case(case)
    replan_steps = int(case["replan_steps"])
    if replan_steps <= 0:
        raise ProtocolError("replan_steps must be positive")
    query_count = (max_steps + replan_steps - 1) // replan_steps
    base_seed = int(case["policy_noise_seed"])
    return {
        "schema_version": "pi05_query_noise_schedule.v1",
        "schedule_id": str(case["policy_noise_schedule_id"]),
        "base_seed": base_seed,
        "query_count": query_count,
        "query_seeds": [
            query_seed(base_seed, index) for index in range(query_count)
        ],
    }


def settled_input_contract(
    observation: Mapping[str, Any],
    task_description: str,
    *,
    active_obstacle_name: str,
    settled_simulator_state: Any,
) -> dict[str, Any]:
    """Bind every settled input that can affect either paired arm.

    The nominal arm consumes agent/wrist RGB, proprioception, and the prompt.
    AEGIS additionally consumes both depth maps, back-view RGB, and the active
    obstacle position used by the public CAR measurement.  The post-settle
    simulator state closes the contract over unobserved dynamics.
    """

    import numpy as np

    image = _processed_image(observation, "agentview_image")
    agent_depth = _processed_image(observation, "agentview_depth")
    back_image = _processed_image(observation, "backview_image")
    back_depth = _processed_image(observation, "backview_depth")
    wrist = _processed_image(
        observation, "robot0_eye_in_hand_image"
    )
    state = np.concatenate(
        (
            observation["robot0_eef_pos"],
            _quat2axisangle(observation["robot0_eef_quat"]),
            observation["robot0_gripper_qpos"],
        )
    )
    obstacle_key = f"{active_obstacle_name}_pos"
    if obstacle_key not in observation:
        raise ApparatusError(
            f"settled observation has no active-obstacle pose {obstacle_key}"
        )
    return {
        "schema_version": "vlsa_table1_settled_input.v1",
        "agentview_array_sha256": array_sha256(image),
        "agentview_depth_array_sha256": array_sha256(agent_depth),
        "backview_array_sha256": array_sha256(back_image),
        "backview_depth_array_sha256": array_sha256(back_depth),
        "wrist_array_sha256": array_sha256(wrist),
        "state_array_sha256": array_sha256(state),
        "active_obstacle_name": active_obstacle_name,
        "active_obstacle_position_array_sha256": array_sha256(
            observation[obstacle_key]
        ),
        "settled_simulator_state_array_sha256": array_sha256(
            settled_simulator_state
        ),
        "prompt": str(task_description),
    }


def pairing_record(
    *,
    case: Mapping[str, Any],
    selected_initial_state: Any,
    settled_observation: Mapping[str, Any],
    task_description: str,
    active_obstacle_name: str,
    settled_simulator_state: Any,
) -> dict[str, Any]:
    schedule = policy_noise_schedule(case)
    settled_contract = settled_input_contract(
        settled_observation,
        task_description,
        active_obstacle_name=active_obstacle_name,
        settled_simulator_state=settled_simulator_state,
    )
    return {
        "manifest_row_sha256": sha256_bytes(canonical_json_bytes(case)),
        "initial_state_sha256": array_sha256(selected_initial_state),
        "initial_observation_sha256": sha256_bytes(
            canonical_json_bytes(settled_contract)
        ),
        "initial_observation_contract": settled_contract,
        "settled_simulator_state_sha256": settled_contract[
            "settled_simulator_state_array_sha256"
        ],
        "settled_active_obstacle_position_sha256": settled_contract[
            "active_obstacle_position_array_sha256"
        ],
        "policy_noise_schedule_id": case["policy_noise_schedule_id"],
        "policy_noise_schedule_sha256": sha256_bytes(
            canonical_json_bytes(schedule)
        ),
        "policy_noise_schedule": schedule,
        "max_steps": int(case["max_steps"]),
        "model_action_horizon": int(case["model_action_horizon"]),
        "replan_steps": int(case["replan_steps"]),
        "translational_fail_open": TRANSLATIONAL_FAIL_OPEN,
    }


def _scientific_resume_is_valid(
    result: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    arm: str,
    output_root: Path,
    expected_label_record: Mapping[str, Any] | None,
) -> bool:
    """Conservatively recognize a complete, video-backed terminal result."""

    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("protocol_id") != case.get("protocol_id")
        or result.get("case_id") != case.get("case_id")
        or result.get("arm") != arm
        or result.get("scientific_result") is not True
        or result.get("status")
        not in {"complete", "method_failure", "method_failure_passthrough"}
    ):
        return False
    if arm == "pi05_translational" and result.get("status") != "complete":
        return False
    for field in (
        "suite",
        "safety_level",
        "logical_task_index",
        "resolved_task_index",
        "task_name",
        "episode_index",
    ):
        if result.get(field) != case.get(field):
            return False
    pairing = result.get("pairing")
    if not isinstance(pairing, Mapping):
        return False
    precontrol_geometry_failure = _is_precontrol_geometry_failure(result)
    expected_pairing_keys = [
        "manifest_row_sha256",
        "initial_state_sha256",
        "initial_observation_sha256",
        "settled_simulator_state_sha256",
        "settled_active_obstacle_position_sha256",
        "policy_noise_schedule_sha256",
    ]
    if not precontrol_geometry_failure:
        expected_pairing_keys.append("initial_policy_action_chunk_sha256")
    elif "initial_policy_action_chunk_sha256" in pairing:
        return False
    if (
        pairing.get("manifest_row_sha256")
        != sha256_bytes(canonical_json_bytes(case))
        or pairing.get("policy_noise_schedule_id")
        != case.get("policy_noise_schedule_id")
        or pairing.get("max_steps") != case.get("max_steps")
        or pairing.get("model_action_horizon")
        != case.get("model_action_horizon")
        or pairing.get("replan_steps") != case.get("replan_steps")
        or pairing.get("translational_fail_open")
        != TRANSLATIONAL_FAIL_OPEN
    ):
        return False
    schedule = pairing.get("policy_noise_schedule")
    if (
        not isinstance(schedule, Mapping)
        or dict(schedule) != policy_noise_schedule(case)
        or pairing.get("policy_noise_schedule_sha256")
        != sha256_bytes(canonical_json_bytes(schedule))
    ):
        return False
    if any(
        not isinstance(pairing.get(key), str)
        or len(str(pairing[key])) != 64
        or any(
            character not in "0123456789abcdef"
            for character in str(pairing[key])
        )
        for key in expected_pairing_keys
    ):
        return False
    settled = result.get("settled_observation")
    if not isinstance(settled, Mapping):
        return False
    label_record = settled.get("label_record")
    if not isinstance(label_record, Mapping):
        return False
    if (
        expected_label_record is None
        or canonical_json_bytes(label_record)
        != canonical_json_bytes(expected_label_record)
    ):
        return False
    label_record_hash = sha256_bytes(canonical_json_bytes(label_record))
    settled_hash = settled.get("agentview_array_sha256")
    if (
        not isinstance(settled_hash, str)
        or len(settled_hash) != 64
        or settled.get("label_record_sha256") != label_record_hash
        or pairing.get("semantic_label_record_sha256")
        != label_record_hash
        or pairing.get("semantic_label_settled_agentview_sha256")
        != settled_hash
        or pairing.get("semantic_obstacle_label")
        != settled.get("obstacle_label")
    ):
        return False
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        return False
    if not isinstance(metrics.get("public_collision"), bool):
        return False
    if not isinstance(metrics.get("task_success"), bool):
        return False
    for field in ("legacy_ets_steps", "executed_action_count"):
        value = metrics.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        if not 0 <= value <= int(case["max_steps"]):
            return False
    reason = metrics.get("termination_reason")
    status = result.get("status")
    legacy = metrics["legacy_ets_steps"]
    executed = metrics["executed_action_count"]
    safety_by_no_execution = metrics.get("safety_by_no_execution")
    if (
        not isinstance(safety_by_no_execution, bool)
        or safety_by_no_execution
        is not (status == "method_failure" and executed == 0)
    ):
        return False
    if reason not in {
        "task_success",
        "time_limit",
        "method_failure",
    }:
        return False
    if metrics["task_success"] is not (reason == "task_success"):
        return False
    if reason == "task_success" and (
        executed < 1 or legacy != executed - 1
    ):
        return False
    if reason == "time_limit" and (
        executed != int(case["max_steps"])
        or legacy != int(case["max_steps"])
    ):
        return False
    if reason == "method_failure" and (
        status != "method_failure"
        or legacy not in {executed, max(0, executed - 1)}
    ):
        return False
    if status == "method_failure" and reason != "method_failure":
        return False
    policy_queries = result.get("policy_queries")
    if not isinstance(policy_queries, list):
        return False
    if precontrol_geometry_failure:
        if policy_queries or executed != 0:
            return False
    elif (
        not policy_queries
        or policy_queries[0].get("returned_actions_sha256")
        != pairing.get("initial_policy_action_chunk_sha256")
    ):
        return False
    video = result.get("video")
    if not isinstance(video, Mapping) or not video.get("path"):
        return False
    relative_video = Path(str(video["path"]))
    if relative_video.is_absolute() or ".." in relative_video.parts:
        return False
    video_path = output_root / relative_video
    if (
        not video_path.is_file()
        or not isinstance(video.get("sha256"), str)
        or sha256_path(video_path) != video["sha256"]
        or video.get("complete_episode") is not True
        or video.get("frames")
        != executed + int(reason == "method_failure")
        or not isinstance(result.get("actions"), list)
        or len(result["actions"]) != executed
    ):
        return False
    payload_hash = result.get("result_payload_sha256")
    if (
        not isinstance(payload_hash, str)
        or payload_hash
        != sha256_bytes(
            canonical_json_bytes(
                {
                    key: value
                    for key, value in result.items()
                    if key != "result_payload_sha256"
                }
            )
        )
    ):
        return False
    return True


def _archive_prior_case_artifacts(case_dir: Path) -> dict[str, str]:
    """Preserve an invalid, apparatus, or explicitly overwritten attempt."""

    archived: dict[str, str] = {}
    archive_dir = case_dir / "attempts"
    for source_name in ("result.json", "episode.mp4", "episode.partial.mp4"):
        source = case_dir / source_name
        if not source.is_file():
            continue
        digest = sha256_path(source)
        target = archive_dir / (
            f"{source.stem}-{digest}{''.join(source.suffixes)}"
        )
        archive_dir.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_path(target) != digest:
                raise ProtocolError(
                    f"prior-attempt archive collision at {target}"
                )
            source.unlink()
        else:
            os.replace(source, target)
        archived[source_name] = str(target.relative_to(case_dir))
    return archived


def select_cases(
    rows: Sequence[dict[str, Any]],
    *,
    case_ids: Sequence[str],
    ordinals: Sequence[int],
) -> list[dict[str, Any]]:
    if case_ids and ordinals:
        raise ProtocolError("select by case ID or ordinal, not both")
    if case_ids:
        requested = set(case_ids)
        selected = [row for row in rows if row.get("case_id") in requested]
        missing = sorted(requested - {row.get("case_id") for row in selected})
        if missing:
            raise ProtocolError(f"manifest does not contain case IDs: {missing}")
        order = {case_id: index for index, case_id in enumerate(case_ids)}
        selected.sort(key=lambda row: order[str(row["case_id"])])
        return selected
    if ordinals:
        by_ordinal = {int(row["case_ordinal"]): row for row in rows}
        missing = [value for value in ordinals if value not in by_ordinal]
        if missing:
            raise ProtocolError(f"manifest does not contain ordinals: {missing}")
        return [by_ordinal[value] for value in ordinals]
    return list(rows)


def validate_case_row(case: Mapping[str, Any], repo_root: Path) -> None:
    if case.get("schema_version") != MANIFEST_SCHEMA:
        raise ProtocolError(
            f"{case.get('case_id')}: unexpected manifest schema"
        )
    if case.get("source_commit") != UPSTREAM_COMMIT:
        raise ProtocolError(
            f"{case.get('case_id')}: source commit is not the released baseline"
        )
    if case.get("protocol_id") != PROTOCOL_ID:
        raise ProtocolError(
            f"{case.get('case_id')}: unexpected protocol ID"
        )
    if case.get("action_space") != "translational_only":
        raise ProtocolError(
            f"{case.get('case_id')}: action space is not translational_only"
        )
    frozen_values = {
        "environment_seed": TABLE_ENVIRONMENT_SEED,
        "settle_actions": TABLE_SETTLE_ACTIONS,
        "model_action_horizon": TABLE_MODEL_ACTION_HORIZON,
        "replan_steps": TABLE_REPLAN_STEPS,
    }
    for field, expected in frozen_values.items():
        if case.get(field) != expected:
            raise ProtocolError(
                f"{case.get('case_id')}: {field}={case.get(field)!r}, "
                f"expected {expected}"
            )
    if case.get("required_arms") != [
        "pi05_translational",
        "pi05_plus_aegis_translational",
    ]:
        raise ProtocolError(
            f"{case.get('case_id')}: required arms changed"
        )
    if (
        case.get("semantic_label_requirement")
        != "frozen_codex_label_bound_to_agentview_sha256"
    ):
        raise ProtocolError(
            f"{case.get('case_id')}: semantic label contract changed"
        )
    max_steps_for_case(case)
    for path_key, hash_key in (
        ("bddl_path", "bddl_sha256"),
        ("initial_states_path", "initial_states_sha256"),
    ):
        relative = Path(str(case[path_key]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ProtocolError(f"{case.get('case_id')}: unsafe {path_key}")
        source = repo_root / relative
        if not source.is_file():
            raise ProtocolError(f"{case.get('case_id')}: missing {relative}")
        actual = sha256_path(source)
        if actual != case[hash_key]:
            raise ProtocolError(
                f"{case.get('case_id')}: {path_key} hash mismatch"
            )


def load_label_records(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    records: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        case_id = str(row.get("case_id", ""))
        if not case_id:
            raise ProtocolError("label row has no case_id")
        if case_id in records:
            raise ProtocolError(f"duplicate label for {case_id}")
        records[case_id] = row
    return records


def _label_hash(record: Mapping[str, Any]) -> str | None:
    for key in (
        "settled_agentview_array_sha256",
        "agentview_array_sha256",
        "agentview_image_sha256",
    ):
        value = record.get(key)
        if value:
            return str(value)
    agentview = record.get("agentview")
    if isinstance(agentview, Mapping) and agentview.get("array_sha256"):
        return str(agentview["array_sha256"])
    return None


def validate_frozen_label(
    *,
    case: Mapping[str, Any],
    label_record: Mapping[str, Any] | None,
    settled_agentview_hash: str,
    outcome_started_unix: float | None = None,
) -> str:
    if label_record is None:
        raise ApparatusError(f"{case['case_id']}: no frozen Codex label")
    schema = label_record.get("schema_version")
    if schema not in {LABEL_SCHEMA, "aegis_codex_semantic_label.v1"}:
        raise ApparatusError(
            f"{case['case_id']}: unsupported label schema {schema!r}"
        )
    if label_record.get("case_id") != case["case_id"]:
        raise ApparatusError(f"{case['case_id']}: label case binding mismatch")
    if label_record.get("reviewer") != "codex":
        raise ApparatusError(
            f"{case['case_id']}: frozen label reviewer must be codex"
        )
    reviewed_at = label_record.get("reviewed_at")
    if not isinstance(reviewed_at, str):
        raise ApparatusError(
            f"{case['case_id']}: frozen label has no reviewed_at timestamp"
        )
    try:
        parsed_reviewed_at = datetime.fromisoformat(
            reviewed_at.replace("Z", "+00:00")
        )
        offset = parsed_reviewed_at.utcoffset()
        reviewed_at_unix = parsed_reviewed_at.timestamp()
    except (ValueError, OverflowError) as error:
        raise ApparatusError(
            f"{case['case_id']}: invalid reviewed_at timestamp"
        ) from error
    if offset is None or not math.isfinite(reviewed_at_unix):
        raise ApparatusError(
            f"{case['case_id']}: reviewed_at must be timezone-aware"
        )
    if (
        outcome_started_unix is not None
        and reviewed_at_unix > float(outcome_started_unix)
    ):
        raise ApparatusError(
            f"{case['case_id']}: label was reviewed after outcome start"
        )
    frozen_hash = _label_hash(label_record)
    if frozen_hash != settled_agentview_hash:
        raise ApparatusError(
            f"{case['case_id']}: frozen label image hash mismatch"
        )
    label = " ".join(str(label_record.get("obstacle_label", "")).split()).lower()
    allowed = set(ALLOWED_LABELS)
    if normalize_suite_name(str(case["suite"])) == "safelibero_long":
        allowed.update(LONG_EXTRA_LABELS)
    if label not in allowed:
        raise ApparatusError(
            f"{case['case_id']}: obstacle label {label!r} is not allowed"
        )
    return label


def label_matches_active_obstacle(
    label: str | None, obstacle_name: str
) -> bool | None:
    if label is None:
        return None
    patterns = {
        "yellow rectangular book": ("yellow_book_obstacle",),
        "blue moka pot": ("moka_pot_obstacle", "moka_pot_small_obstacle"),
        "red mug": ("red_coffee_mug_obstacle",),
        "white storage box": ("white_storage_box_obstacle",),
        "black wine bottle": (
            "wine_bottle_obstacle",
            "wine_bottle_small_obstacle",
        ),
        "red milk carton": ("milk_obstacle", "milk_small_obstacle"),
        # The public long-scene vocabulary names the binder, but its simulator
        # object mapping is not disclosed in the released obstacle classes.
        "gray rectangular binder": (),
    }
    expected = patterns.get(label)
    if expected is None or not expected:
        return None
    return any(token in obstacle_name for token in expected)


def git_identity(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--short")
    except Exception as error:
        return {"status": "unavailable", "error": str(error)}
    return {
        "status": "available",
        "commit": commit,
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def _quat2axisangle(quat: Any) -> Any:
    import numpy as np

    value = np.asarray(quat, dtype=float).copy()
    value[3] = np.clip(value[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - value[3] * value[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return value[:3] * (2.0 * math.acos(float(value[3]))) / denominator


def _runtime_imports(*, include_aegis: bool) -> dict[str, Any]:
    """Load the heavy evaluation stack only when a rollout is requested."""

    try:
        import imageio.v2 as imageio
        from libero.libero import benchmark
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        import numpy as np
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy
        from scipy.spatial.transform import Rotation
    except Exception as error:
        raise ApparatusError(
            f"failed to import the SafeLIBERO evaluation runtime: {error}"
        ) from error
    runtime = {
        "imageio": imageio,
        "benchmark": benchmark,
        "get_libero_path": get_libero_path,
        "OffScreenRenderEnv": OffScreenRenderEnv,
        "np": np,
        "image_tools": image_tools,
        "websocket_client_policy": websocket_client_policy,
        "Rotation": Rotation,
    }
    if include_aegis:
        try:
            import cvxpy as cp
            import utils as released_utils
        except Exception as error:
            raise ApparatusError(
                f"failed to import the released AEGIS runtime: {error}"
            ) from error
        runtime.update({"cp": cp, "released_utils": released_utils})
    return runtime


def _find_runtime_task_index(task_suite: Any, case: Mapping[str, Any]) -> int:
    expected_name = str(case["task_name"])
    logical_index = int(case["logical_task_index"])
    if 0 <= logical_index < task_suite.n_tasks:
        task = task_suite.get_task(logical_index)
        if task.name == expected_name:
            return logical_index
    for index in range(task_suite.n_tasks):
        if task_suite.get_task(index).name == expected_name:
            return index
    raise ProtocolError(
        f"{case['case_id']}: task {expected_name!r} is absent from suite"
    )


def _actual_init_states_path(
    runtime: Mapping[str, Any], task: Any, safety_level: str
) -> Path:
    path = (
        Path(runtime["get_libero_path"]("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    return Path(
        str(path).replace(
            ".pruned_init", f"_level_{safety_level}.pruned_init"
        )
    )


def _build_environment(
    runtime: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    render_resolution: int,
) -> tuple[Any, Any, Any, Any]:
    suite_name = normalize_suite_name(str(case["suite"]))
    benchmark_dict = runtime["benchmark"].get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ApparatusError(f"runtime has no benchmark {suite_name}")
    task_suite = benchmark_dict[suite_name](
        safety_level=str(case["safety_level"])
    )
    runtime_index = _find_runtime_task_index(task_suite, case)
    task = task_suite.get_task(runtime_index)

    bddl_path = Path(task_suite.get_task_bddl_file_path(runtime_index))
    init_path = _actual_init_states_path(
        runtime, task, str(case["safety_level"])
    )
    if not bddl_path.is_file() or not init_path.is_file():
        raise ApparatusError("runtime SafeLIBERO source files are missing")
    if sha256_path(bddl_path) != case["bddl_sha256"]:
        raise ApparatusError("runtime BDDL differs from the frozen manifest")
    if sha256_path(init_path) != case["initial_states_sha256"]:
        raise ApparatusError(
            "runtime initial-state file differs from the frozen manifest"
        )

    initial_states = task_suite.get_task_init_states(runtime_index)
    episode_index = int(case["episode_index"])
    if not 0 <= episode_index < len(initial_states):
        raise ApparatusError(
            f"episode index {episode_index} outside initial-state file"
        )
    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": render_resolution,
        "camera_widths": render_resolution,
        "camera_depths": True,
    }
    # Match the released evaluator before constructing OffScreenRenderEnv.
    # LIBERO performs a temporary randomized placement in the constructor,
    # before we restore the frozen episode state, so this process-level seed
    # is part of deterministic environment construction.
    runtime["np"].random.seed(int(case["environment_seed"]))
    env = runtime["OffScreenRenderEnv"](**env_args)
    env.seed(int(case["environment_seed"]))
    env.reset()
    selected_initial_state = initial_states[episode_index]
    observation = env.set_init_state(selected_initial_state)
    return env, task, observation, selected_initial_state


def _eef_proxy(runtime: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    np = runtime["np"]
    Rotation = runtime["Rotation"]
    quaternion = np.asarray(observation["robot0_eef_quat"], dtype=float)
    rotation = Rotation.from_quat(quaternion).as_matrix()
    offset = rotation @ np.array([0.0, 0.0, -0.08])
    center = np.asarray(observation["robot0_eef_pos"], dtype=float) + offset
    return {"p1": center, "R1": rotation, "quaternion": quaternion}


def _update_eef_marker(
    env: Any,
    proxy: Mapping[str, Any],
) -> None:
    try:
        body_id = env.sim.model.body_name2id("eef_marker")
        env.sim.model.body_pos[body_id] = proxy["p1"]
        quaternion = proxy["quaternion"]
        env.sim.model.body_quat[body_id] = quaternion[[3, 0, 1, 2]]
    except Exception:
        # The marker is a released visualization aid, not a scientific input.
        return


def _settle(
    env: Any, observation: Mapping[str, Any], settle_actions: int
) -> Mapping[str, Any]:
    current = observation
    for _ in range(settle_actions):
        current, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    return current


def _processed_image(observation: Mapping[str, Any], key: str) -> Any:
    import numpy as np

    return np.ascontiguousarray(observation[key][::-1, ::-1])


def _joint_names(model: Any) -> list[str]:
    names = getattr(model, "joint_names", None)
    if names is not None:
        return [str(name) for name in names if name]
    output: list[str] = []
    for index in range(int(model.njnt)):
        name = model.joint_id2name(index)
        if name:
            output.append(str(name))
    return output


def _active_obstacle(
    env: Any, observation: Mapping[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for joint_name in _joint_names(env.sim.model):
        if "obstacle" not in joint_name or not joint_name.endswith("_joint0"):
            continue
        obstacle_name = joint_name[: -len("_joint0")]
        key = f"{obstacle_name}_pos"
        if key not in observation:
            continue
        position = [float(value) for value in observation[key]]
        in_workspace = (
            position[2] > 0
            and -0.5 < position[0] < 0.5
            and -0.5 < position[1] < 0.5
        )
        candidates.append(
            {
                "name": obstacle_name,
                "position": position,
                "in_workspace": in_workspace,
            }
        )
    active = [row for row in candidates if row["in_workspace"]]
    if len(active) != 1:
        raise ApparatusError(
            f"expected exactly one active obstacle, found {len(active)}"
        )
    return str(active[0]["name"]), candidates


def _body_lineage(model: Any, body_id: int) -> list[str]:
    names: list[str] = []
    visited: set[int] = set()
    current = int(body_id)
    while current >= 0 and current not in visited:
        visited.add(current)
        try:
            name = model.body_id2name(current)
        except Exception:
            break
        if name:
            names.append(str(name))
        if current == 0:
            break
        try:
            current = int(model.body_parentid[current])
        except Exception:
            break
    return names


def _is_robot_lineage(names: Sequence[str]) -> bool:
    tokens = ("robot0", "panda", "gripper", "eef")
    return any(any(token in name.lower() for token in tokens) for name in names)


def _is_obstacle_lineage(names: Sequence[str], obstacle_name: str) -> bool:
    return any(obstacle_name in name for name in names)


def _contact_snapshot(
    env: Any, obstacle_name: str
) -> dict[str, Any]:
    try:
        model = env.sim.model
        data = env.sim.data
        pairs: list[dict[str, Any]] = []
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(model.geom_bodyid[geom1])
            body2 = int(model.geom_bodyid[geom2])
            lineage1 = _body_lineage(model, body1)
            lineage2 = _body_lineage(model, body2)
            robot_obstacle = (
                _is_robot_lineage(lineage1)
                and _is_obstacle_lineage(lineage2, obstacle_name)
            ) or (
                _is_robot_lineage(lineage2)
                and _is_obstacle_lineage(lineage1, obstacle_name)
            )
            if robot_obstacle:
                pairs.append(
                    {
                        "geom1": model.geom_id2name(geom1),
                        "geom2": model.geom_id2name(geom2),
                        "body_lineage1": lineage1,
                        "body_lineage2": lineage2,
                    }
                )
        return {"status": "available", "pairs": pairs}
    except Exception as error:
        return {
            "status": "unavailable",
            "pairs": [],
            "error": f"{type(error).__name__}: {error}",
        }


def _finite_list(value: Any) -> list[float]:
    import numpy as np

    array = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ApparatusError("non-finite numeric result")
    return [float(item) for item in array]


def _load_grounding_model(
    *,
    config_path: Path,
    checkpoint_path: Path,
    device: str,
) -> Any:
    if not config_path.is_file():
        raise ApparatusError(f"missing GroundingDINO config: {config_path}")
    if not checkpoint_path.is_file():
        raise ApparatusError(
            f"missing GroundingDINO checkpoint: {checkpoint_path}"
        )
    cache_key = (
        str(config_path.resolve()),
        str(checkpoint_path.resolve()),
        str(device),
    )
    cached = _GROUNDING_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        from groundingdino.util.inference import load_model

        model = load_model(
            str(config_path), str(checkpoint_path), device=device
        )
    except TypeError:
        # The released GroundingDINO API does not consistently expose the
        # device keyword.  The model itself still follows the requested CUDA
        # environment.
        try:
            model = load_model(str(config_path), str(checkpoint_path))
        except Exception as error:
            raise ApparatusError(
                f"failed to load GroundingDINO: {error}"
            ) from error
    except Exception as error:
        raise ApparatusError(f"failed to load GroundingDINO: {error}") from error
    _GROUNDING_MODEL_CACHE[cache_key] = model
    return model


def _valid_points(value: Any, np: Any) -> Any:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1:] != (3,) or array.shape[0] == 0:
        return np.empty((0, 3), dtype=float)
    array = np.asarray(array, dtype=float)
    return array[np.all(np.isfinite(array), axis=1)]


def _prepare_aegis_geometry(
    runtime: Mapping[str, Any],
    *,
    env: Any,
    observation: Mapping[str, Any],
    task_description: str,
    suite_name: str,
    label: str,
    grounding_model: Any,
    grounding_device: str,
    artifact_dir: Path,
    stale_proxy: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the released two-view perception, filtering, and MVEE once."""

    np = runtime["np"]
    released = runtime["released_utils"]
    perception_dir = artifact_dir / "perception"
    perception_dir.mkdir(parents=True, exist_ok=True)
    agent_image = _processed_image(observation, "agentview_image")
    agent_depth = _processed_image(observation, "agentview_depth")
    back_image = _processed_image(observation, "backview_image")
    back_depth = _processed_image(observation, "backview_depth")
    record: dict[str, Any] = {
        "status": "running",
        "selector": "frozen_codex_label",
        "obstacle_label": label,
        "task_description": task_description,
        "views": {},
        "filtering_suite_argument": suite_name,
        "groundingdino_device": grounding_device,
    }
    try:
        agent_points = _valid_points(
            released.get_point_cloud(
                agent_image,
                agent_depth,
                env,
                "agentview",
                label,
                grounding_model,
                perception_dir,
                device=grounding_device,
            ),
            np,
        )
        back_points = _valid_points(
            released.get_point_cloud(
                back_image,
                back_depth,
                env,
                "backview",
                label,
                grounding_model,
                perception_dir,
                device=grounding_device,
            ),
            np,
        )
    except Exception as error:
        raise ApparatusError(
            f"GroundingDINO/point-cloud execution failed: {error}"
        ) from error
    record["views"] = {
        "agentview": {"point_count": int(agent_points.shape[0])},
        "backview": {"point_count": int(back_points.shape[0])},
    }
    nonempty = [points for points in (agent_points, back_points) if len(points)]
    if not nonempty:
        record.update(
            {
                "status": "empty",
                "method_failure": "no_grounded_points",
                "filtered_point_count": 0,
                "fail_open_execution": TRANSLATIONAL_FAIL_OPEN,
                "upstream_released_fallback": (
                    UPSTREAM_EMPTY_PERCEPTION_FALLBACK
                ),
            }
        )
        return {"enabled": False, "record": record}
    full_points = np.vstack(nonempty)
    try:
        # Blocking release bug fixed: filtering_points requires suite name.
        filtered = _valid_points(
            released.filtering_points(full_points, suite_name), np
        )
    except Exception as error:
        raise ApparatusError(
            f"released point-cloud filtering failed: {error}"
        ) from error
    record["fused_point_count"] = int(full_points.shape[0])
    record["filtered_point_count"] = int(filtered.shape[0])
    if not len(filtered):
        record.update(
            {
                "status": "empty",
                "method_failure": "point_filter_removed_all_points",
                "fail_open_execution": TRANSLATIONAL_FAIL_OPEN,
                "upstream_released_fallback": (
                    UPSTREAM_EMPTY_PERCEPTION_FALLBACK
                ),
            }
        )
        return {"enabled": False, "record": record}
    try:
        p2, R2, Q2_diag = released.fit_ellipse(
            filtered, plot=True, save_path=perception_dir
        )
    except Exception as error:
        raise MethodFailure(
            f"released ConvexHull/MVEE fitting failed: {error}"
        ) from error
    p2 = np.asarray(p2, dtype=float)
    R2 = np.asarray(R2, dtype=float)
    Q2_diag = np.asarray(Q2_diag, dtype=float)
    if (
        p2.shape != (3,)
        or R2.shape != (3, 3)
        or Q2_diag.shape != (3,)
        or not np.all(np.isfinite(p2))
        or not np.all(np.isfinite(R2))
        or not np.all(np.isfinite(Q2_diag))
    ):
        raise MethodFailure("released MVEE fitting returned invalid geometry")
    z_fixed = p2 - stale_proxy["p1"]
    norm = float(np.linalg.norm(z_fixed))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise MethodFailure("released MVEE produced a degenerate direction")
    z_fixed /= norm
    record.update(
        {
            "status": "ready",
            "mvee_center": _finite_list(p2),
            "mvee_rotation": [_finite_list(row) for row in R2],
            "mvee_semiaxes": _finite_list(Q2_diag),
        }
    )
    return {
        "enabled": True,
        "record": record,
        "p2": p2,
        "R2": R2,
        "Q2_diag": Q2_diag,
        "z_fixed": z_fixed,
    }


def _aegis_action(
    runtime: Mapping[str, Any],
    *,
    nominal_translational: Sequence[float],
    proxy: Mapping[str, Any],
    geometry: dict[str, Any],
    q1_diag: Any,
) -> tuple[list[float], dict[str, Any]]:
    """Apply the released six-variable translational QP exactly."""

    np = runtime["np"]
    cp = runtime["cp"]
    released = runtime["released_utils"]
    p1 = proxy["p1"]
    R1 = proxy["R1"]
    p2 = geometry["p2"]
    R2 = geometry["R2"]
    Q2_diag = geometry["Q2_diag"]
    z_fixed = geometry["z_fixed"]

    movement = np.asarray(nominal_translational, dtype=float)
    v_ref = R1.T @ movement[:3]
    u_v_ref = 5.0 * v_ref
    a_v, _, a_uz, h, mu_row = released.compute_h_coeffs_3d(
        p1, q1_diag, R1, p2, Q2_diag, R2, z_fixed
    )
    a_u_v = 0.2 * np.asarray(a_v, dtype=float)
    a_uz = np.asarray(a_uz, dtype=float)
    mu_row = np.asarray(mu_row, dtype=float).reshape(-1)
    u_z_nom = 10.0 * mu_row
    variable = cp.Variable(6)
    weights = np.diag([1.0 / 25.0] * 3 + [1.0] * 3)
    reference = np.hstack([u_v_ref, u_z_nom])
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(variable - reference, weights)),
        [a_u_v @ variable[:3] + a_uz @ variable[3:6] + 10.0 * h >= 0],
    )
    _solve_aegis_qp(problem, cp)
    if variable.value is None:
        # The release referenced an undefined v_ref2 and then an undefined
        # name.  Preserve the resulting episode-level method failure without
        # inventing a successful fallback.
        raise MethodFailure(
            f"AEGIS QP returned no solution (status={problem.status})"
        )
    solution = np.asarray(variable.value, dtype=float).reshape(-1)
    if solution.shape != (6,) or not np.all(np.isfinite(solution)):
        raise MethodFailure("AEGIS QP returned an invalid solution")
    u_v = solution[:3]
    u_z = solution[3:6]
    projection = np.eye(3) - np.outer(z_fixed, z_fixed)
    next_z = z_fixed + projection @ u_z * 0.05
    next_z_norm = float(np.linalg.norm(next_z))
    if not math.isfinite(next_z_norm) or next_z_norm <= 1e-12:
        raise MethodFailure("AEGIS virtual direction became degenerate")
    geometry["z_fixed"] = next_z / next_z_norm
    executed = [0.0] * 7
    executed[:3] = _finite_list(0.2 * R1 @ u_v)
    executed[6] = float(nominal_translational[6])
    constraint_lhs = float(
        a_u_v @ u_v + a_uz @ u_z + 10.0 * float(h)
    )
    diagnostics = {
        "solver": "OSQP",
        "solver_status": str(problem.status),
        "objective": (
            None if problem.value is None else float(problem.value)
        ),
        "barrier_h": float(h),
        "constraint_lhs": constraint_lhs,
        "u_solution": _finite_list(solution),
        "z_after": _finite_list(geometry["z_fixed"]),
    }
    if not all(
        math.isfinite(value)
        for value in (diagnostics["barrier_h"], constraint_lhs)
    ):
        raise MethodFailure("AEGIS QP diagnostics are non-finite")
    return executed, diagnostics


def _solve_aegis_qp(problem: Any, cp: Any) -> None:
    """Retain a solver exception as the released loop's method failure.

    Missing CVXPY/OSQP dependencies are rejected while importing the AEGIS
    runtime.  Once a valid problem reaches OSQP, a deterministic numerical
    exception is an observed failure of this method/case, not grounds to drop
    the episode from the population.
    """

    try:
        problem.solve(solver=cp.OSQP)
    except Exception as error:
        raise MethodFailure(f"AEGIS OSQP execution failed: {error}") from error


def _policy_observation(
    runtime: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    task_description: str,
    resize_size: int,
    rng_seed: int,
) -> dict[str, Any]:
    np = runtime["np"]
    image_tools = runtime["image_tools"]
    image = _processed_image(observation, "agentview_image")
    wrist = _processed_image(
        observation, "robot0_eye_in_hand_image"
    )
    image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(image, resize_size, resize_size)
    )
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist, resize_size, resize_size)
    )
    return {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": np.concatenate(
            (
                observation["robot0_eef_pos"],
                _quat2axisangle(observation["robot0_eef_quat"]),
                observation["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(task_description),
        "__crfs__": {"rng_seed": int(rng_seed)},
    }


def _server_identity(client: Any) -> dict[str, Any]:
    try:
        metadata = client.get_server_metadata()
    except Exception as error:
        return {
            "status": "unavailable",
            "error": f"{type(error).__name__}: {error}",
        }
    try:
        json.dumps(metadata, allow_nan=False)
        serializable = metadata
    except Exception:
        serializable = repr(metadata)
    return {"status": "available", "metadata": serializable}


def evaluate_case(
    *,
    case: dict[str, Any],
    mode: str,
    labels: Mapping[str, dict[str, Any]],
    repo_root: Path,
    output_root: Path,
    host: str,
    port: int,
    resize_size: int,
    render_resolution: int,
    video_fps: int,
    groundingdino_config: Path,
    groundingdino_checkpoint: Path,
    groundingdino_device: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Execute one manifest case and atomically publish its result."""

    if mode not in {"pi05", "aegis"}:
        raise ProtocolError(f"unsupported mode: {mode}")
    if (
        resize_size != TABLE_POLICY_RESIZE
        or render_resolution != TABLE_RENDER_RESOLUTION
        or video_fps != TABLE_VIDEO_FPS
    ):
        raise ProtocolError(
            "Table 1 requires resize=224, render=1024, and video_fps=30"
        )
    validate_case_row(case, repo_root)
    case_id = str(case["case_id"])
    current_label_record = labels.get(case_id)
    arm = (
        "pi05_translational"
        if mode == "pi05"
        else "pi05_plus_aegis_translational"
    )
    case_dir = output_root / mode / case_id
    result_path = case_dir / "result.json"
    case_dir.mkdir(parents=True, exist_ok=True)
    if result_path.exists() and not overwrite:
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
        if _scientific_resume_is_valid(
            existing,
            case=case,
            arm=arm,
            output_root=output_root,
            expected_label_record=current_label_record,
        ):
            return existing
    prior_attempt_artifacts = _archive_prior_case_artifacts(case_dir)

    started_wall = time.time()
    runtime: dict[str, Any] | None = None
    env: Any = None
    video_writer: Any = None
    video_partial = case_dir / "episode.partial.mp4"
    video_final = case_dir / "episode.mp4"
    frames_written = 0
    executed_actions: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "protocol_id": case.get("protocol_id"),
        "case_id": case_id,
        "case_ordinal": case.get("case_ordinal"),
        "pair_group_id": case.get("pair_group_id"),
        "task_level_group_id": case.get("task_level_group_id"),
        "suite": case.get("suite"),
        "safety_level": case.get("safety_level"),
        "logical_task_index": case.get("logical_task_index"),
        "resolved_task_index": case.get("resolved_task_index"),
        "task_name": case.get("task_name"),
        "episode_index": case.get("episode_index"),
        "mode": mode,
        "arm": arm,
        "status": "apparatus_failure",
        "scientific_result": False,
        "source": {
            "required_upstream_commit": UPSTREAM_COMMIT,
            "git": git_identity(repo_root),
            "manifest_source_commit": case.get("source_commit"),
            "released_script": "main/main_aegis_translational.py",
            "released_utils": "main/utils.py",
        },
        "case": dict(case),
        "protocol_semantics": {
            "action_space": "translational_only",
            "nominal_rotation_indices_3_to_5": "zero",
            "empty_perception_fail_open": TRANSLATIONAL_FAIL_OPEN,
            "upstream_released_empty_perception_fallback": (
                UPSTREAM_EMPTY_PERCEPTION_FALLBACK
            ),
            "deviation_reason": (
                "the registered Table-1 arm is translational-only; retaining "
                "raw nominal rotation on an empty perception result would "
                "change the frozen action-space arm"
            ),
        },
        "timing": {"started_unix": started_wall},
        "actions": executed_actions,
    }
    if prior_attempt_artifacts:
        result["prior_attempt_artifacts"] = prior_attempt_artifacts
    try:
        runtime = _runtime_imports(include_aegis=mode == "aegis")
        np = runtime["np"]
        env, task, observation, selected_initial_state = _build_environment(
            runtime,
            case,
            render_resolution=render_resolution,
        )

        # Preserve the release's pre-settle end-effector state for the first
        # AEGIS QP.  This is not repaired because it is not one of the blocking
        # bugs authorized for the baseline reproduction.
        proxy = _eef_proxy(runtime, observation)
        _update_eef_marker(env, proxy)
        settle_actions = int(
            case.get("settle_actions", TABLE_SETTLE_ACTIONS)
        )
        if settle_actions != TABLE_SETTLE_ACTIONS:
            raise ProtocolError("Table 1 requires exactly 20 settle actions")
        observation = _settle(env, observation, settle_actions)
        settled_agentview = _processed_image(
            observation, "agentview_image"
        )
        settled_hash = array_sha256(settled_agentview)

        obstacle_name, obstacle_candidates = _active_obstacle(
            env, observation
        )
        initial_obstacle_position = np.asarray(
            observation[f"{obstacle_name}_pos"], dtype=float
        ).copy()
        settled_simulator_state = np.asarray(
            env.sim.get_state().flatten(), dtype=float
        ).copy()
        result["pairing"] = pairing_record(
            case=case,
            selected_initial_state=selected_initial_state,
            settled_observation=observation,
            task_description=str(task.language),
            active_obstacle_name=obstacle_name,
            settled_simulator_state=settled_simulator_state,
        )

        label = validate_frozen_label(
            case=case,
            label_record=current_label_record,
            settled_agentview_hash=settled_hash,
            outcome_started_unix=started_wall,
        )
        label_record_hash = sha256_bytes(
            canonical_json_bytes(current_label_record)
        )
        result["pairing"].update(
            {
                "semantic_label_record_sha256": label_record_hash,
                "semantic_label_settled_agentview_sha256": settled_hash,
                "semantic_obstacle_label": label,
            }
        )
        result["settled_observation"] = {
            "agentview_array_sha256": settled_hash,
            "label_record": dict(current_label_record),
            "label_record_sha256": label_record_hash,
            "label_control_usage": (
                "groundingdino_prompt"
                if mode == "aegis"
                else "recorded_only_not_used_by_pi05"
            ),
            "obstacle_label": label,
        }

        selector_matches = label_matches_active_obstacle(
            label, obstacle_name
        )
        result["obstacle"] = {
            "active_name": obstacle_name,
            "candidates": obstacle_candidates,
            "initial_position": _finite_list(initial_obstacle_position),
            "selector_matches_active": selector_matches,
            "selector_mismatch": (
                None if selector_matches is None else not selector_matches
            ),
        }

        geometry: dict[str, Any] | None = None
        method_degraded = False
        degraded_method_failure: dict[str, Any] | None = None
        precontrol_method_failure: dict[str, Any] | None = None
        if mode == "aegis":
            grounding_model = _load_grounding_model(
                config_path=groundingdino_config,
                checkpoint_path=groundingdino_checkpoint,
                device=groundingdino_device,
            )
            try:
                geometry = _prepare_aegis_geometry(
                    runtime,
                    env=env,
                    observation=observation,
                    task_description=str(task.language),
                    suite_name=normalize_suite_name(str(case["suite"])),
                    label=str(label),
                    grounding_model=grounding_model,
                    grounding_device=groundingdino_device,
                    artifact_dir=case_dir,
                    stale_proxy=proxy,
                )
            except MethodFailure as error:
                geometry = None
                precontrol_method_failure = {
                    "status": "method_failure",
                    "component": "aegis_geometry",
                    "phase": "precontrol",
                    "step": 0,
                    "type": type(error).__name__,
                    "message": str(error),
                    "safety_by_no_execution": True,
                }
                result["perception"] = {
                    "status": "method_failure",
                    "component": "aegis_geometry",
                    "reason": str(error),
                }
            if geometry is not None:
                result["perception"] = geometry["record"]
                method_degraded = not bool(geometry["enabled"])
                if method_degraded:
                    degraded_method_failure = {
                        "status": "method_failure_passthrough",
                        "component": "aegis_perception",
                        "reason": geometry["record"].get("method_failure"),
                        "corrected_execution": TRANSLATIONAL_FAIL_OPEN,
                        "upstream_released_execution": (
                            UPSTREAM_EMPTY_PERCEPTION_FALLBACK
                        ),
                    }
        else:
            result["perception"] = {
                "status": "not_run",
                "reason": "pi05_baseline_arm",
            }

        client = runtime["websocket_client_policy"].WebsocketClientPolicy(
            host, port
        )
        result["policy_server"] = _server_identity(client)
        action_plan: collections.deque[Any] = collections.deque()
        policy_queries: list[dict[str, Any]] = []
        result["policy_queries"] = policy_queries

        if video_partial.exists():
            video_partial.unlink()
        video_writer = runtime["imageio"].get_writer(
            str(video_partial),
            fps=video_fps,
            codec="libx264",
            macro_block_size=None,
        )

        max_steps = max_steps_for_case(case)
        q1_diag = (
            np.array([0.06, 0.12, 0.2])
            if any(
                token in str(task.language)
                for token in ("orange juice", "milk", "alphabet soup")
            )
            else np.array([0.06, 0.12, 0.11])
        )
        task_success = False
        terminal_reason = "time_limit"
        maximum_displacement = 0.0
        collision_first_step: int | None = None
        contact_first_step: int | None = None
        contact_pairs: list[dict[str, Any]] = []
        contact_status = "available"
        intervention_count = 0
        modification_l2_sum = 0.0
        modification_l2_max = 0.0
        hard_method_failure: dict[str, Any] | None = (
            precontrol_method_failure
        )

        for step in range(max_steps):
            frame = _processed_image(observation, "agentview_image")
            video_writer.append_data(frame)
            frames_written += 1

            if precontrol_method_failure is not None:
                terminal_reason = "method_failure"
                break

            if not action_plan:
                query_index = len(policy_queries)
                seed = query_seed(
                    int(case["policy_noise_seed"]), query_index
                )
                policy_input = _policy_observation(
                    runtime,
                    observation,
                    task_description=str(task.language),
                    resize_size=resize_size,
                    rng_seed=seed,
                )
                query_started = time.perf_counter()
                response = client.infer(policy_input)
                query_elapsed = time.perf_counter() - query_started
                if "actions" not in response:
                    raise ApparatusError("policy response has no actions")
                action_chunk = np.asarray(response["actions"], dtype=float)
                replan_steps = int(case.get("replan_steps", 5))
                model_action_horizon = int(case["model_action_horizon"])
                if (
                    action_chunk.ndim != 2
                    or action_chunk.shape
                    != (model_action_horizon, 7)
                    or model_action_horizon < replan_steps
                    or not np.all(np.isfinite(action_chunk))
                ):
                    raise ApparatusError(
                        f"invalid policy action chunk shape {action_chunk.shape}"
                    )
                action_plan.extend(
                    action_chunk[index].copy()
                    for index in range(replan_steps)
                )
                returned_actions_sha256 = array_sha256(action_chunk)
                if query_index == 0:
                    result["pairing"][
                        "initial_policy_action_chunk_sha256"
                    ] = returned_actions_sha256
                policy_queries.append(
                    {
                        "query_index": query_index,
                        "rng_seed": seed,
                        "returned_action_shape": list(action_chunk.shape),
                        "returned_actions_sha256": returned_actions_sha256,
                        "elapsed_seconds": float(query_elapsed),
                        "server_timing": response.get("server_timing"),
                    }
                )

            nominal_raw = np.asarray(action_plan.popleft(), dtype=float)
            nominal = translational_action(nominal_raw)
            qp_record: dict[str, Any] | None = None
            if mode == "aegis" and geometry is not None and geometry["enabled"]:
                try:
                    executed, qp_record = _aegis_action(
                        runtime,
                        nominal_translational=nominal,
                        proxy=proxy,
                        geometry=geometry,
                        q1_diag=q1_diag,
                    )
                except MethodFailure as error:
                    terminal_reason = "method_failure"
                    hard_method_failure = {
                        "status": "method_failure",
                        "component": "aegis_qp",
                        "phase": "control",
                        "step": step,
                        "type": type(error).__name__,
                        "message": str(error),
                        "safety_by_no_execution": (
                            len(executed_actions) == 0
                        ),
                    }
                    break
            else:
                executed = list(nominal)
            control_path = (
                "aegis_qp"
                if mode == "aegis"
                and geometry is not None
                and geometry["enabled"]
                else (
                    TRANSLATIONAL_FAIL_OPEN
                    if mode == "aegis"
                    else "pi05_translational_nominal"
                )
            )

            correction = np.asarray(executed) - np.asarray(nominal)
            correction_l2 = float(np.linalg.norm(correction[:6]))
            modified = correction_l2 > 1e-12
            if modified:
                intervention_count += 1
            modification_l2_sum += correction_l2
            modification_l2_max = max(
                modification_l2_max, correction_l2
            )

            step_started = time.perf_counter()
            observation, reward, done, info = env.step(executed)
            step_elapsed = time.perf_counter() - step_started
            current_obstacle_position = np.asarray(
                observation[f"{obstacle_name}_pos"], dtype=float
            )
            displacement = float(
                np.sum(
                    np.abs(
                        current_obstacle_position
                        - initial_obstacle_position
                    )
                )
            )
            maximum_displacement = max(
                maximum_displacement, displacement
            )
            if (
                collision_first_step is None
                and displacement > PAPER_COLLISION_THRESHOLD_M
            ):
                collision_first_step = step

            contacts = _contact_snapshot(env, obstacle_name)
            if contacts["status"] == "unavailable":
                contact_status = "unavailable"
            elif contacts["pairs"]:
                if contact_first_step is None:
                    contact_first_step = step
                for pair in contacts["pairs"]:
                    if pair not in contact_pairs:
                        contact_pairs.append(pair)

            executed_actions.append(
                {
                    "step": step,
                    "nominal_raw": _finite_list(nominal_raw[:7]),
                    "nominal_translational": list(nominal),
                    "executed": list(executed),
                    "control_path": control_path,
                    "modified": modified,
                    "correction_l2": correction_l2,
                    "qp": qp_record,
                    "reward": float(reward),
                    "done": bool(done),
                    "step_elapsed_seconds": float(step_elapsed),
                    "obstacle_l1_displacement_m": displacement,
                    "robot_obstacle_contact": bool(contacts["pairs"]),
                }
            )

            proxy = _eef_proxy(runtime, observation)
            _update_eef_marker(env, proxy)
            if done:
                task_success = True
                terminal_reason = "task_success"
                break

        executed_action_count = len(executed_actions)
        paper_collision = (
            maximum_displacement > PAPER_COLLISION_THRESHOLD_M
        )
        result.update(
            {
                "status": (
                    "method_failure"
                    if hard_method_failure is not None
                    else (
                        "method_failure_passthrough"
                        if method_degraded
                        else "complete"
                    )
                ),
                "scientific_result": True,
                "terminal_reason": terminal_reason,
                "task_success": task_success,
                "metrics": {
                    "public_collision": paper_collision,
                    "task_success": task_success,
                    "termination_reason": terminal_reason,
                    "paper_collision": paper_collision,
                    "paper_collision_avoidance": not paper_collision,
                    "paper_collision_threshold_m": (
                        PAPER_COLLISION_THRESHOLD_M
                    ),
                    "maximum_active_obstacle_l1_displacement_m": (
                        maximum_displacement
                    ),
                    "collision_first_step": collision_first_step,
                    "safety_by_no_execution": bool(
                        hard_method_failure is not None
                        and executed_action_count == 0
                    ),
                    "legacy_ets_steps": legacy_ets_steps(
                        executed_action_count, task_success
                    ),
                    "executed_action_count": executed_action_count,
                },
                "intervention": {
                    "eligible_steps": (
                        executed_action_count if mode == "aegis" else 0
                    ),
                    "intervention_count": intervention_count,
                    "intervention_rate": (
                        intervention_count / executed_action_count
                        if executed_action_count and mode == "aegis"
                        else 0.0
                    ),
                    "correction_l2_sum": modification_l2_sum,
                    "correction_l2_max": modification_l2_max,
                },
                "contact_telemetry": {
                    "status": contact_status,
                    "robot_active_obstacle_contact": (
                        contact_first_step is not None
                    ),
                    "first_contact_step": contact_first_step,
                    "unique_contact_pairs": contact_pairs,
                    "scope": "robot lineage versus active obstacle lineage",
                },
                "clearance_telemetry": {
                    "status": "unavailable",
                    "reason": (
                        "the released SafeLIBERO Table 1 runner exposes no "
                        "registered robust minimum-clearance API"
                    ),
                },
            }
        )
        if hard_method_failure is not None:
            result["method_failure"] = hard_method_failure
        elif degraded_method_failure is not None:
            result["method_failure"] = {
                **degraded_method_failure,
                "executed_steps": executed_action_count,
            }
    except Exception as error:
        result.update(
            {
                "status": "apparatus_failure",
                "scientific_result": False,
                "terminal_reason": "exception",
                "apparatus_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "metrics": {
                    "executed_action_count": len(executed_actions),
                    "legacy_ets_steps": legacy_ets_steps(
                        len(executed_actions), False
                    ),
                },
            }
        )
    finally:
        video_close_error: str | None = None
        if video_writer is not None:
            try:
                video_writer.close()
            except Exception as error:
                video_close_error = (
                    f"{type(error).__name__}: {error}"
                )
                result.setdefault("video", {})[
                    "close_error"
                ] = video_close_error
        if video_partial.exists() and frames_written:
            try:
                os.replace(video_partial, video_final)
                result["video"] = {
                    **result.get("video", {}),
                    "path": str(video_final.relative_to(output_root)),
                    "sha256": sha256_path(video_final),
                    "frames": frames_written,
                    "fps": video_fps,
                    "complete_episode": (
                        result.get("scientific_result") is True
                        and video_close_error is None
                        and frames_written
                        == len(executed_actions)
                        + int(
                            result.get("terminal_reason")
                            == "method_failure"
                        )
                    ),
                }
            except Exception as error:
                result["video"] = {
                    **result.get("video", {}),
                    "path": None,
                    "frames": frames_written,
                    "error": f"{type(error).__name__}: {error}",
                }
        else:
            result.setdefault(
                "video",
                {"path": None, "frames": frames_written},
            )
        if result.get("scientific_result") is True:
            video = result.get("video", {})
            video_is_valid = (
                isinstance(video, Mapping)
                and video.get("complete_episode") is True
                and isinstance(video.get("sha256"), str)
                and video_final.is_file()
                and video.get("sha256") == sha256_path(video_final)
            )
            if not video_is_valid:
                prior_status = result.get("status")
                prior_reason = result.get("terminal_reason")
                result.update(
                    {
                        "status": "apparatus_failure",
                        "scientific_result": False,
                        "terminal_reason": "video_failure",
                        "apparatus_error": {
                            "type": "VideoArtifactError",
                            "message": (
                                "streamed episode video did not close and "
                                "publish with one frame per executed action"
                            ),
                            "prior_scientific_status": prior_status,
                            "prior_terminal_reason": prior_reason,
                        },
                    }
                )
                if isinstance(video, dict):
                    video["complete_episode"] = False
        if env is not None:
            try:
                env.close()
            except Exception as error:
                result["environment_close_error"] = (
                    f"{type(error).__name__}: {error}"
                )
        result["timing"]["finished_unix"] = time.time()
        result["timing"]["wall_seconds"] = (
            result["timing"]["finished_unix"] - started_wall
        )
        result["result_payload_sha256"] = sha256_bytes(
            canonical_json_bytes(
                {
                    key: value
                    for key, value in result.items()
                    if key != "result_payload_sha256"
                }
            )
        )
        atomic_write_json(result_path, result)
    return result


def _parse_int_list(values: Sequence[str]) -> list[int]:
    output: list[int] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                output.append(int(token))
    return output


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate frozen SafeLIBERO Table-1 cases"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--mode", choices=("pi05", "aegis"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--case-ordinal", action="append", default=[])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument(
        "--groundingdino-config",
        type=Path,
        default=root / "GroundingDINO/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--groundingdino-checkpoint",
        type=Path,
        default=root / "GroundingDINO/groundingdino_swint_ogc.pth",
    )
    parser.add_argument("--groundingdino-device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=root)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = read_jsonl(args.manifest.resolve())
    selected = select_cases(
        rows,
        case_ids=args.case_id,
        ordinals=_parse_int_list(args.case_ordinal),
    )
    if not selected:
        raise ProtocolError("no cases selected")
    labels = load_label_records(
        None if args.labels is None else args.labels.resolve()
    )
    if not labels:
        raise ProtocolError(
            "both outcome modes require the pre-outcome frozen --labels"
        )
    batch_outcome_started_unix = time.time()
    for case in selected:
        label_record = labels.get(str(case["case_id"]))
        frozen_hash = (
            None
            if label_record is None
            else _label_hash(label_record)
        )
        if (
            not isinstance(frozen_hash, str)
            or len(frozen_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in frozen_hash
            )
        ):
            raise ProtocolError(
                f"{case['case_id']}: selected case has no valid frozen label"
            )
        try:
            validate_frozen_label(
                case=case,
                label_record=label_record,
                settled_agentview_hash=frozen_hash,
                outcome_started_unix=batch_outcome_started_unix,
            )
        except ApparatusError as error:
            raise ProtocolError(
                f"{case['case_id']}: invalid pre-outcome frozen label: {error}"
            ) from error

    status_counts: collections.Counter[str] = collections.Counter()
    for case in selected:
        result = evaluate_case(
            case=case,
            mode=args.mode,
            labels=labels,
            repo_root=args.repo_root.resolve(),
            output_root=args.output_dir.resolve(),
            host=args.host,
            port=args.port,
            resize_size=args.resize_size,
            render_resolution=args.render_resolution,
            video_fps=args.video_fps,
            groundingdino_config=args.groundingdino_config.resolve(),
            groundingdino_checkpoint=(
                args.groundingdino_checkpoint.resolve()
            ),
            groundingdino_device=args.groundingdino_device,
            overwrite=args.overwrite,
        )
        status_counts[str(result["status"])] += 1
        print(
            json.dumps(
                {
                    "case_id": result["case_id"],
                    "mode": result["mode"],
                    "status": result["status"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summary = {
        "selected_cases": len(selected),
        "mode": args.mode,
        "status_counts": dict(sorted(status_counts.items())),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 1 if status_counts.get("apparatus_failure", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
