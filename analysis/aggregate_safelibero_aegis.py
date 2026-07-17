#!/usr/bin/env python3
"""Strictly aggregate the complete paired VLSA Table 1 population.

No result is silently excluded. Algorithmic method failures remain in every
denominator, while apparatus failures and incomplete populations prevent an
aggregate from being published.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


CONFIG_SCHEMA = "vlsa_table1_translational_protocol.v1"
MANIFEST_SCHEMA = "vlsa_table1_population_case.v1"
RECEIPT_SCHEMA = "vlsa_table1_population_receipt.v1"
OUTPUT_SCHEMA = "vlsa_table1_population_summary.v1"
LABEL_SCHEMAS = {
    "vlsa_table1_codex_label.v1",
    "aegis_codex_semantic_label.v1",
}
ALLOWED_LABELS = {
    "yellow rectangular book",
    "blue moka pot",
    "red mug",
    "white storage box",
    "black wine bottle",
    "red milk carton",
}
LONG_EXTRA_LABELS = {"gray rectangular binder"}
NOISE_SCHEMA = "pi05_query_noise_schedule.v1"
SETTLED_INPUT_SCHEMA = "vlsa_table1_settled_input.v1"
TRANSLATIONAL_FAIL_OPEN = "corrected_translational_nominal"
PAPER_COLLISION_THRESHOLD_M = 0.001


class AggregationError(RuntimeError):
    """Raised when the population is incomplete or scientifically invalid."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_record_sha256(value: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AggregationError(f"{path} must contain a JSON object")
    return value


def load_jsonl_bytes(
    payload: bytes, *, source: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise AggregationError(
                f"{source}:{line_number} is invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise AggregationError(
                f"{source}:{line_number} must be a JSON object"
            )
        rows.append(value)
    return rows


def load_result_records(path: Path) -> list[dict[str, Any]]:
    """Load one pretty-printed result JSON or a JSONL result shard."""

    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return load_jsonl_bytes(payload, source=str(path))
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list) and all(
        isinstance(row, dict) for row in value
    ):
        return list(value)
    raise AggregationError(
        f"{path} must contain a result object, result-object list, or JSONL"
    )


def expand_result_paths(paths: Iterable[Path]) -> list[Path]:
    """Expand immutable run roots without relying on a 3,200-file argv."""

    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(path.rglob("result.json")))
        elif path.is_file():
            expanded.append(path)
        else:
            raise AggregationError(f"result path does not exist: {path}")
    if not expanded:
        raise AggregationError("no result files were found")
    return expanded


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _finite_scalar(
    value: Any,
    *,
    label: str,
    nonnegative: bool = False,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (nonnegative and float(value) < 0.0)
    ):
        qualifier = " nonnegative" if nonnegative else ""
        raise AggregationError(f"{label} must be a finite{qualifier} number")
    return float(value)


def _finite_vector(
    value: Any,
    *,
    length: int,
    label: str,
) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise AggregationError(f"{label} must contain {length} values")
    return [
        _finite_scalar(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    ]


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AggregationError(f"{label} must be a lowercase SHA-256")
    return value


def _result_payload_sha256(result: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                key: value
                for key, value in result.items()
                if key != "result_payload_sha256"
                and not str(key).startswith("_")
            }
        )
    )


def _validate_precontrol_geometry_failure(
    result: Mapping[str, Any],
    *,
    aegis_arm: str,
    pairing: Mapping[str, Any],
) -> bool:
    """Recognize the sole outcome allowed before the first policy query."""

    failure = result.get("method_failure")
    if (
        not isinstance(failure, Mapping)
        or failure.get("component") != "aegis_geometry"
    ):
        return False

    metrics = result.get("metrics")
    video = result.get("video")
    exact_contract = (
        result.get("arm") == aegis_arm
        and result.get("status") == "method_failure"
        and failure.get("status") == "method_failure"
        and failure.get("phase") == "precontrol"
        and _is_int(failure.get("step"))
        and failure.get("step") == 0
        and failure.get("safety_by_no_execution") is True
        and result.get("terminal_reason") == "method_failure"
        and result.get("task_success") is False
        and isinstance(metrics, Mapping)
        and _is_int(metrics.get("executed_action_count"))
        and metrics.get("executed_action_count") == 0
        and _is_int(metrics.get("legacy_ets_steps"))
        and metrics.get("legacy_ets_steps") == 0
        and metrics.get("termination_reason") == "method_failure"
        and metrics.get("task_success") is False
        and metrics.get("safety_by_no_execution") is True
        and result.get("actions") == []
        and result.get("policy_queries") == []
        and isinstance(video, Mapping)
        and _is_int(video.get("frames"))
        and video.get("frames") == 1
        and "initial_policy_action_chunk_sha256" not in pairing
    )
    if not exact_contract:
        raise AggregationError(
            f"{result.get('case_id')}/{result.get('arm')}: precontrol "
            "AEGIS geometry failure contract is invalid"
        )
    return True


def _label_image_sha256(record: Mapping[str, Any]) -> str | None:
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


def expected_policy_noise_schedule(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    max_steps = int(manifest["max_steps"])
    replan_steps = int(manifest["replan_steps"])
    if max_steps <= 0 or replan_steps <= 0:
        raise AggregationError(
            f"{manifest.get('case_id')}: invalid policy horizon"
        )
    query_count = (max_steps + replan_steps - 1) // replan_steps
    base_seed = int(manifest["policy_noise_seed"])
    seeds = [base_seed + index for index in range(query_count)]
    if any(not 0 <= seed <= 0xFFFFFFFF for seed in seeds):
        raise AggregationError(
            f"{manifest.get('case_id')}: query seed is outside uint32"
        )
    return {
        "schema_version": NOISE_SCHEMA,
        "schedule_id": str(manifest["policy_noise_schedule_id"]),
        "base_seed": base_seed,
        "query_count": query_count,
        "query_seeds": seeds,
    }


def _validate_label(
    result: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    pairing: Mapping[str, Any],
) -> None:
    case_id = str(manifest["case_id"])
    arm = str(result.get("arm"))
    settled = result.get("settled_observation")
    if not isinstance(settled, Mapping):
        raise AggregationError(
            f"{case_id}/{arm}: settled observation is missing"
        )
    settled_hash = _require_sha256(
        settled.get("agentview_array_sha256"),
        label=f"{case_id}/{arm}/settled agentview",
    )
    record = settled.get("label_record")
    if not isinstance(record, Mapping):
        raise AggregationError(
            f"{case_id}/{arm}: frozen label record is missing"
        )
    if (
        record.get("schema_version") not in LABEL_SCHEMAS
        or record.get("case_id") != case_id
        or record.get("reviewer") != "codex"
    ):
        raise AggregationError(
            f"{case_id}/{arm}: frozen label identity is invalid"
        )
    reviewed_at = record.get("reviewed_at")
    try:
        parsed = datetime.fromisoformat(
            str(reviewed_at).replace("Z", "+00:00")
        )
        reviewed_unix = parsed.timestamp()
    except (ValueError, OverflowError):
        raise AggregationError(
            f"{case_id}/{arm}: frozen label timestamp is invalid"
        ) from None
    timing = result.get("timing")
    started = (
        timing.get("started_unix")
        if isinstance(timing, Mapping)
        else None
    )
    if (
        parsed.utcoffset() is None
        or not isinstance(started, (int, float))
        or isinstance(started, bool)
        or not math.isfinite(float(started))
        or reviewed_unix > float(started)
    ):
        raise AggregationError(
            f"{case_id}/{arm}: label was not frozen before the outcome"
        )
    if _label_image_sha256(record) != settled_hash:
        raise AggregationError(
            f"{case_id}/{arm}: label image binding changed"
        )
    label = " ".join(str(record.get("obstacle_label", "")).split()).lower()
    allowed = set(ALLOWED_LABELS)
    if manifest.get("suite") == "safelibero_long":
        allowed.update(LONG_EXTRA_LABELS)
    if label not in allowed:
        raise AggregationError(
            f"{case_id}/{arm}: obstacle label is outside the frozen vocabulary"
        )
    record_hash = canonical_record_sha256(dict(record))
    if (
        settled.get("label_record_sha256") != record_hash
        or pairing.get("semantic_label_record_sha256") != record_hash
        or pairing.get("semantic_label_settled_agentview_sha256")
        != settled_hash
        or pairing.get("semantic_obstacle_label") != label
        or settled.get("obstacle_label") != label
    ):
        raise AggregationError(
            f"{case_id}/{arm}: label payload bindings disagree"
        )


def _validate_noise_and_queries(
    result: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    pairing: Mapping[str, Any],
    executed: int,
    precontrol_geometry_failure: bool,
) -> None:
    case_id = str(manifest["case_id"])
    arm = str(result.get("arm"))
    expected = expected_policy_noise_schedule(manifest)
    schedule = pairing.get("policy_noise_schedule")
    if (
        not isinstance(schedule, Mapping)
        or dict(schedule) != expected
        or pairing.get("policy_noise_schedule_sha256")
        != canonical_record_sha256(expected)
    ):
        raise AggregationError(
            f"{case_id}/{arm}: policy-noise schedule is invalid"
        )
    queries = result.get("policy_queries")
    if not isinstance(queries, list):
        raise AggregationError(
            f"{case_id}/{arm}: policy-query ledger is missing"
        )
    if precontrol_geometry_failure:
        if queries:
            raise AggregationError(
                f"{case_id}/{arm}: precontrol geometry failure queried policy"
            )
        if "initial_policy_action_chunk_sha256" in pairing:
            raise AggregationError(
                f"{case_id}/{arm}: precontrol geometry failure has an "
                "initial policy action hash"
            )
        return
    attempted_steps = executed + int(result.get("status") == "method_failure")
    expected_query_count = (
        attempted_steps + int(manifest["replan_steps"]) - 1
    ) // int(manifest["replan_steps"])
    if len(queries) != expected_query_count:
        raise AggregationError(
            f"{case_id}/{arm}: policy-query count is invalid"
        )
    for index, query in enumerate(queries):
        if not isinstance(query, Mapping):
            raise AggregationError(
                f"{case_id}/{arm}: policy query {index} is invalid"
            )
        if (
            query.get("query_index") != index
            or query.get("rng_seed") != expected["query_seeds"][index]
            or query.get("returned_action_shape")
            != [int(manifest["model_action_horizon"]), 7]
        ):
            raise AggregationError(
                f"{case_id}/{arm}: policy query {index} changed"
            )
        _require_sha256(
            query.get("returned_actions_sha256"),
            label=f"{case_id}/{arm}/query-{index}",
        )
    if not queries:
        raise AggregationError(
            f"{case_id}/{arm}: outcome has no initial policy query"
        )
    if (
        pairing.get("initial_policy_action_chunk_sha256")
        != queries[0]["returned_actions_sha256"]
    ):
        raise AggregationError(
            f"{case_id}/{arm}: initial policy action chunk is not bound"
        )


def _validate_video_metadata(
    result: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    executed: int,
) -> None:
    case_id = str(manifest["case_id"])
    arm = str(result.get("arm"))
    video = result.get("video")
    if not isinstance(video, Mapping):
        raise AggregationError(f"{case_id}/{arm}: video record is missing")
    raw_path = video.get("path")
    relative = Path(str(raw_path))
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise AggregationError(
            f"{case_id}/{arm}: video path is unsafe"
        )
    _require_sha256(
        video.get("sha256"), label=f"{case_id}/{arm}/video"
    )
    if (
        video.get("complete_episode") is not True
        or video.get("fps") != 30
        or video.get("frames")
        != executed + int(result.get("status") == "method_failure")
    ):
        raise AggregationError(
            f"{case_id}/{arm}: video metadata is incomplete"
        )


def load_protocol(
    config_path: Path,
    receipt_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise AggregationError("unexpected protocol config schema")
    receipt = load_json(receipt_path)
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise AggregationError("unexpected manifest receipt schema")
    if receipt.get("protocol_id") != config.get("protocol_id"):
        raise AggregationError("receipt/config protocol mismatch")
    config_sha256 = sha256_bytes(config_raw)
    if receipt.get("protocol_config_sha256") != config_sha256:
        raise AggregationError("protocol config hash does not match receipt")

    manifest_raw = manifest_path.read_bytes()
    if receipt.get("manifest_sha256") != sha256_bytes(manifest_raw):
        raise AggregationError("manifest hash does not match receipt")
    rows = load_jsonl_bytes(manifest_raw, source=str(manifest_path))
    expected_cases = int(config["population"]["expected_cases"])
    if (
        len(rows) != expected_cases
        or receipt.get("manifest_rows") != expected_cases
    ):
        raise AggregationError(
            f"manifest has {len(rows)} cases, expected {expected_cases}"
        )

    seen: set[str] = set()
    group_counts: Counter[str] = Counter()
    for ordinal, row in enumerate(rows):
        case_id = row.get("case_id")
        if row.get("schema_version") != MANIFEST_SCHEMA:
            raise AggregationError(f"{case_id}: unexpected manifest schema")
        if (
            row.get("protocol_id") != config["protocol_id"]
            or row.get("protocol_config_sha256") != config_sha256
            or row.get("source_commit")
            != config["source"]["upstream_commit"]
            or row.get("case_ordinal") != ordinal
        ):
            raise AggregationError(f"{case_id}: manifest identity mismatch")
        if not isinstance(case_id, str) or case_id in seen:
            raise AggregationError(f"duplicate or invalid case ID: {case_id}")
        seen.add(case_id)
        group_counts[row["task_level_group_id"]] += 1
        if row.get("required_arms") != config["arms"]:
            raise AggregationError(f"{case_id}: required arms changed")
    expected_group_size = int(
        config["population"]["expected_cases_per_task_level_group"]
    )
    if (
        len(group_counts) != 32
        or set(group_counts.values()) != {expected_group_size}
    ):
        raise AggregationError(
            f"manifest group sizes are invalid: {dict(group_counts)}"
        )
    return config, rows


def _validate_metrics(
    result: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    case_id = manifest["case_id"]
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise AggregationError(f"{case_id}: metrics are missing")
    for key in (
        "public_collision",
        "paper_collision",
        "paper_collision_avoidance",
        "task_success",
    ):
        if not _is_bool(metrics.get(key)):
            raise AggregationError(f"{case_id}: {key} must be Boolean")
    for key in ("legacy_ets_steps", "executed_action_count"):
        if not _is_int(metrics.get(key)):
            raise AggregationError(f"{case_id}: {key} must be an integer")

    max_steps = int(manifest["max_steps"])
    legacy = metrics["legacy_ets_steps"]
    executed = metrics["executed_action_count"]
    if not (0 <= legacy <= max_steps and 0 <= executed <= max_steps):
        raise AggregationError(f"{case_id}: step metrics exceed the horizon")

    reason = metrics.get("termination_reason")
    status = result["status"]
    if reason not in {"task_success", "time_limit", "method_failure"}:
        raise AggregationError(f"{case_id}: invalid termination reason")
    if status == "method_failure" and reason != "method_failure":
        raise AggregationError(
            f"{case_id}: hard method failure needs method_failure termination"
        )
    if metrics["task_success"] is not (reason == "task_success"):
        raise AggregationError(
            f"{case_id}: success and termination reason disagree"
        )
    if reason == "task_success":
        if executed < 1 or legacy != executed - 1:
            raise AggregationError(
                f"{case_id}: successful legacy ETS must be actions minus one"
            )
    elif reason == "time_limit":
        if executed != max_steps or legacy != max_steps:
            raise AggregationError(
                f"{case_id}: time-limit episode must consume its horizon"
            )
    else:
        if status != "method_failure" or legacy not in {
            executed,
            max(0, executed - 1),
        }:
            raise AggregationError(
                f"{case_id}: invalid retained method-failure step count"
            )
    threshold = metrics.get("paper_collision_threshold_m")
    displacement = metrics.get(
        "maximum_active_obstacle_l1_displacement_m"
    )
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or float(threshold) != PAPER_COLLISION_THRESHOLD_M
        or not isinstance(displacement, (int, float))
        or isinstance(displacement, bool)
        or not math.isfinite(float(displacement))
        or float(displacement) < 0.0
    ):
        raise AggregationError(
            f"{case_id}: collision telemetry is invalid"
        )
    recomputed_collision = (
        float(displacement) > PAPER_COLLISION_THRESHOLD_M
    )
    if (
        metrics["public_collision"] is not recomputed_collision
        or metrics["paper_collision"] is not recomputed_collision
        or metrics["paper_collision_avoidance"] is recomputed_collision
    ):
        raise AggregationError(
            f"{case_id}: collision metric does not match displacement"
        )
    if (
        result.get("task_success") is not metrics["task_success"]
        or result.get("terminal_reason") != reason
    ):
        raise AggregationError(
            f"{case_id}: top-level outcome disagrees with metrics"
        )
    actions = result.get("actions")
    if not isinstance(actions, list) or len(actions) != executed:
        raise AggregationError(
            f"{case_id}: executed-action ledger length is invalid"
        )
    arm = str(result.get("arm"))
    action_displacements: list[float] = []
    action_contacts: list[bool] = []
    correction_l2_values: list[float] = []
    modified_count = 0
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping) or action.get("step") != index:
            raise AggregationError(
                f"{case_id}/{arm}: action {index} is invalid"
            )
        expected_paths = (
            {"pi05_translational_nominal"}
            if arm == "pi05_translational"
            else {"aegis_qp", TRANSLATIONAL_FAIL_OPEN}
        )
        if action.get("control_path") not in expected_paths:
            raise AggregationError(
                f"{case_id}/{arm}: action {index} has invalid control path"
            )
        nominal_raw = _finite_vector(
            action.get("nominal_raw"),
            length=7,
            label=f"{case_id}/{arm}/action-{index}/nominal_raw",
        )
        nominal = _finite_vector(
            action.get("nominal_translational"),
            length=7,
            label=f"{case_id}/{arm}/action-{index}/nominal_translational",
        )
        applied = _finite_vector(
            action.get("executed"),
            length=7,
            label=f"{case_id}/{arm}/action-{index}/executed",
        )
        expected_nominal = nominal_raw[:3] + [0.0, 0.0, 0.0] + [
            nominal_raw[6]
        ]
        if any(
            not math.isclose(
                observed, expected, rel_tol=0.0, abs_tol=1e-12
            )
            for observed, expected in zip(nominal, expected_nominal)
        ):
            raise AggregationError(
                f"{case_id}/{arm}: action {index} changed the "
                "translational nominal"
            )
        correction_l2 = _finite_scalar(
            action.get("correction_l2"),
            label=f"{case_id}/{arm}/action-{index}/correction_l2",
            nonnegative=True,
        )
        recomputed_l2 = math.sqrt(
            sum(
                (applied[axis] - nominal[axis]) ** 2
                for axis in range(6)
            )
        )
        if not math.isclose(
            correction_l2, recomputed_l2, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise AggregationError(
                f"{case_id}/{arm}: action {index} correction norm changed"
            )
        modified = action.get("modified")
        if not _is_bool(modified) or modified is not (
            correction_l2 > 1e-12
        ):
            raise AggregationError(
                f"{case_id}/{arm}: action {index} modification flag changed"
            )
        control_path = action["control_path"]
        if control_path in {
            "pi05_translational_nominal",
            TRANSLATIONAL_FAIL_OPEN,
        }:
            if action.get("qp") is not None or modified:
                raise AggregationError(
                    f"{case_id}/{arm}: action {index} fail-open/nominal "
                    "path cannot contain a QP correction"
                )
        elif not isinstance(action.get("qp"), Mapping):
            raise AggregationError(
                f"{case_id}/{arm}: action {index} lacks QP diagnostics"
            )
        if arm == "pi05_translational" and modified:
            raise AggregationError(
                f"{case_id}: baseline action {index} was modified"
            )
        _finite_scalar(
            action.get("reward"),
            label=f"{case_id}/{arm}/action-{index}/reward",
        )
        _finite_scalar(
            action.get("step_elapsed_seconds"),
            label=f"{case_id}/{arm}/action-{index}/step time",
            nonnegative=True,
        )
        displacement_at_step = _finite_scalar(
            action.get("obstacle_l1_displacement_m"),
            label=f"{case_id}/{arm}/action-{index}/obstacle displacement",
            nonnegative=True,
        )
        contact_at_step = action.get("robot_obstacle_contact")
        done_at_step = action.get("done")
        if not _is_bool(contact_at_step) or not _is_bool(done_at_step):
            raise AggregationError(
                f"{case_id}/{arm}: action {index} contact/done is invalid"
            )
        if done_at_step and index != executed - 1:
            raise AggregationError(
                f"{case_id}/{arm}: action ledger continues after done"
            )
        action_displacements.append(displacement_at_step)
        action_contacts.append(contact_at_step)
        correction_l2_values.append(correction_l2)
        modified_count += int(modified)

    recomputed_maximum = max(action_displacements, default=0.0)
    if not math.isclose(
        float(displacement),
        recomputed_maximum,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise AggregationError(
            f"{case_id}/{arm}: maximum obstacle displacement was not "
            "derived from the action ledger"
        )
    recomputed_first_collision = next(
        (
            index
            for index, value in enumerate(action_displacements)
            if value > PAPER_COLLISION_THRESHOLD_M
        ),
        None,
    )
    if metrics.get("collision_first_step") != recomputed_first_collision:
        raise AggregationError(
            f"{case_id}/{arm}: collision first-step telemetry changed"
        )
    recomputed_success = bool(actions and actions[-1]["done"])
    if recomputed_success is not metrics["task_success"]:
        raise AggregationError(
            f"{case_id}/{arm}: task success was not derived from done"
        )

    intervention = result.get("intervention")
    if not isinstance(intervention, Mapping):
        raise AggregationError(
            f"{case_id}/{arm}: intervention telemetry is missing"
        )
    expected_eligible = executed if arm != "pi05_translational" else 0
    expected_rate = (
        modified_count / executed
        if executed and arm != "pi05_translational"
        else 0.0
    )
    observed_rate = _finite_scalar(
        intervention.get("intervention_rate"),
        label=f"{case_id}/{arm}/intervention rate",
        nonnegative=True,
    )
    observed_sum = _finite_scalar(
        intervention.get("correction_l2_sum"),
        label=f"{case_id}/{arm}/correction sum",
        nonnegative=True,
    )
    observed_max = _finite_scalar(
        intervention.get("correction_l2_max"),
        label=f"{case_id}/{arm}/correction maximum",
        nonnegative=True,
    )
    if (
        intervention.get("eligible_steps") != expected_eligible
        or intervention.get("intervention_count") != modified_count
        or not math.isclose(
            observed_rate, expected_rate, rel_tol=1e-12, abs_tol=1e-12
        )
        or not math.isclose(
            observed_sum,
            sum(correction_l2_values),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            observed_max,
            max(correction_l2_values, default=0.0),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise AggregationError(
            f"{case_id}/{arm}: intervention summary does not match actions"
        )

    contact = result.get("contact_telemetry")
    if (
        not isinstance(contact, Mapping)
        or contact.get("status") not in {"available", "unavailable"}
        or not isinstance(contact.get("unique_contact_pairs"), list)
    ):
        raise AggregationError(
            f"{case_id}/{arm}: contact telemetry is invalid"
        )
    first_contact = next(
        (index for index, value in enumerate(action_contacts) if value),
        None,
    )
    if (
        contact.get("robot_active_obstacle_contact")
        is not (first_contact is not None)
        or contact.get("first_contact_step") != first_contact
    ):
        raise AggregationError(
            f"{case_id}/{arm}: contact summary does not match actions"
        )
    return metrics


def validate_result(
    result: dict[str, Any],
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    case_id = manifest["case_id"]
    arm = result.get("arm")
    payload_hash = _require_sha256(
        result.get("result_payload_sha256"),
        label=f"{case_id}/{arm}/result payload",
    )
    if payload_hash != _result_payload_sha256(result):
        raise AggregationError(
            f"{case_id}/{arm}: result payload hash mismatch"
        )
    if (
        result.get("schema_version")
        != config["result_contract"]["schema_version"]
        or result.get("protocol_id") != config["protocol_id"]
        or result.get("case_id") != case_id
        or arm not in config["arms"]
    ):
        raise AggregationError(f"{case_id}: result identity mismatch")
    accepted = set(
        config["result_contract"]["accepted_terminal_statuses"]
    )
    status = result.get("status")
    if status not in accepted:
        raise AggregationError(
            f"{case_id}/{arm}: non-scientific or nonterminal status {status!r}"
        )
    if result.get("scientific_result") is not True:
        raise AggregationError(
            f"{case_id}/{arm}: scientific_result must be true"
        )
    if arm == config["arms"][0] and status != "complete":
        raise AggregationError(
            f"{case_id}: nominal baseline cannot be a method failure"
        )
    method_failure = result.get("method_failure")
    if status == "method_failure":
        if (
            not isinstance(method_failure, Mapping)
            or method_failure.get("status") != "method_failure"
        ):
            raise AggregationError(
                f"{case_id}/{arm}: hard method failure is not documented"
            )
    elif status == "method_failure_passthrough":
        if (
            not isinstance(method_failure, Mapping)
            or method_failure.get("status")
            != "method_failure_passthrough"
            or method_failure.get("corrected_execution")
            != TRANSLATIONAL_FAIL_OPEN
            or method_failure.get("upstream_released_execution")
            != "raw_nominal_including_rotation"
        ):
            raise AggregationError(
                f"{case_id}/{arm}: fail-open method failure is not explicit"
            )

    identity_fields = (
        "suite",
        "safety_level",
        "logical_task_index",
        "resolved_task_index",
        "task_name",
        "episode_index",
    )
    for field in identity_fields:
        if result.get(field) != manifest.get(field):
            raise AggregationError(
                f"{case_id}/{arm}: {field} differs from manifest"
            )

    pairing = result.get("pairing")
    if not isinstance(pairing, dict):
        raise AggregationError(f"{case_id}/{arm}: pairing record is missing")
    precontrol_geometry_failure = _validate_precontrol_geometry_failure(
        result,
        aegis_arm=str(config["arms"][1]),
        pairing=pairing,
    )
    expected_manifest_hash = canonical_record_sha256(manifest)
    if pairing.get("manifest_row_sha256") != expected_manifest_hash:
        raise AggregationError(
            f"{case_id}/{arm}: manifest-row binding changed"
        )
    for field in (
        "initial_state_sha256",
        "initial_observation_sha256",
        "settled_simulator_state_sha256",
        "settled_active_obstacle_position_sha256",
        "policy_noise_schedule_sha256",
        "semantic_label_record_sha256",
        "semantic_label_settled_agentview_sha256",
    ):
        _require_sha256(
            pairing.get(field), label=f"{case_id}/{arm}/{field}"
        )
    if precontrol_geometry_failure:
        if "initial_policy_action_chunk_sha256" in pairing:
            raise AggregationError(
                f"{case_id}/{arm}: precontrol geometry failure has an "
                "initial policy action hash"
            )
    else:
        _require_sha256(
            pairing.get("initial_policy_action_chunk_sha256"),
            label=(
                f"{case_id}/{arm}/initial_policy_action_chunk_sha256"
            ),
        )
    if not isinstance(pairing.get("semantic_obstacle_label"), str) or not (
        pairing["semantic_obstacle_label"].strip()
    ):
        raise AggregationError(
            f"{case_id}/{arm}: semantic obstacle label is missing"
        )
    if (
        pairing.get("policy_noise_schedule_id")
        != manifest["policy_noise_schedule_id"]
        or pairing.get("max_steps") != manifest["max_steps"]
        or pairing.get("model_action_horizon")
        != manifest["model_action_horizon"]
        or pairing.get("replan_steps") != manifest["replan_steps"]
        or pairing.get("translational_fail_open")
        != TRANSLATIONAL_FAIL_OPEN
    ):
        raise AggregationError(
            f"{case_id}/{arm}: pairing protocol differs from manifest"
        )
    settled_contract = pairing.get("initial_observation_contract")
    if (
        not isinstance(settled_contract, Mapping)
        or settled_contract.get("schema_version")
        != SETTLED_INPUT_SCHEMA
        or pairing.get("initial_observation_sha256")
        != canonical_record_sha256(dict(settled_contract))
        or pairing.get("settled_simulator_state_sha256")
        != settled_contract.get(
            "settled_simulator_state_array_sha256"
        )
        or pairing.get("settled_active_obstacle_position_sha256")
        != settled_contract.get(
            "active_obstacle_position_array_sha256"
        )
    ):
        raise AggregationError(
            f"{case_id}/{arm}: settled input contract is invalid"
        )
    for field in (
        "agentview_array_sha256",
        "agentview_depth_array_sha256",
        "backview_array_sha256",
        "backview_depth_array_sha256",
        "wrist_array_sha256",
        "state_array_sha256",
        "active_obstacle_position_array_sha256",
        "settled_simulator_state_array_sha256",
    ):
        _require_sha256(
            settled_contract.get(field),
            label=f"{case_id}/{arm}/settled/{field}",
        )
    if (
        not isinstance(settled_contract.get("active_obstacle_name"), str)
        or not settled_contract["active_obstacle_name"]
        or not isinstance(settled_contract.get("prompt"), str)
        or not settled_contract["prompt"]
    ):
        raise AggregationError(
            f"{case_id}/{arm}: settled non-array inputs are missing"
        )

    validated = dict(result)
    validated["metrics"] = _validate_metrics(result, manifest)
    _validate_label(result, manifest=manifest, pairing=pairing)
    _validate_noise_and_queries(
        result,
        manifest=manifest,
        pairing=pairing,
        executed=validated["metrics"]["executed_action_count"],
        precontrol_geometry_failure=precontrol_geometry_failure,
    )
    _validate_video_metadata(
        result,
        manifest=manifest,
        executed=validated["metrics"]["executed_action_count"],
    )
    return validated


def load_results(
    result_paths: Iterable[Path],
    *,
    config: dict[str, Any],
    manifests: list[dict[str, Any]],
    verify_video_files: bool = True,
) -> dict[tuple[str, str], dict[str, Any]]:
    manifest_by_id = {row["case_id"]: row for row in manifests}
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for path in expand_result_paths(result_paths):
        for raw in load_result_records(path):
            case_id = raw.get("case_id")
            arm = raw.get("arm")
            manifest = manifest_by_id.get(case_id)
            if manifest is None:
                raise AggregationError(f"unexpected case result: {case_id}")
            key = (case_id, arm)
            if key in results:
                raise AggregationError(f"duplicate result: {key}")
            validated = validate_result(
                raw, config=config, manifest=manifest
            )
            artifact_root = path.parent
            if (
                path.name == "result.json"
                and path.parent.parent.name in {"pi05", "aegis"}
            ):
                artifact_root = path.parents[2]
            video_path: Path | None = None
            if verify_video_files:
                relative_video = Path(str(validated["video"]["path"]))
                root = artifact_root.resolve()
                candidate = (root / relative_video).resolve()
                try:
                    candidate.relative_to(root)
                except ValueError as error:
                    raise AggregationError(
                        f"{case_id}/{arm}: video escapes artifact root"
                    ) from error
                if (
                    not candidate.is_file()
                    or sha256_path(candidate) != validated["video"]["sha256"]
                ):
                    raise AggregationError(
                        f"{case_id}/{arm}: video file/hash mismatch"
                    )
                video_path = candidate
            validated["_result_source_path"] = str(path.resolve())
            validated["_artifact_root"] = str(artifact_root.resolve())
            validated["_resolved_video_path"] = (
                None if video_path is None else str(video_path)
            )
            results[key] = validated

    expected = {
        (row["case_id"], arm)
        for row in manifests
        for arm in config["arms"]
    }
    actual = set(results)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AggregationError(
            f"incomplete paired population: missing={len(missing)} "
            f"{missing[:5]}, extra={len(extra)} {extra[:5]}"
        )
    return results


def validate_pairs(
    *,
    config: dict[str, Any],
    manifests: list[dict[str, Any]],
    results: dict[tuple[str, str], dict[str, Any]],
) -> None:
    paired_fields = (
        "initial_state_sha256",
        "initial_observation_sha256",
        "initial_observation_contract",
        "settled_simulator_state_sha256",
        "settled_active_obstacle_position_sha256",
        "policy_noise_schedule_id",
        "policy_noise_schedule_sha256",
        "policy_noise_schedule",
        "semantic_label_record_sha256",
        "semantic_label_settled_agentview_sha256",
        "semantic_obstacle_label",
        "max_steps",
        "model_action_horizon",
        "replan_steps",
        "translational_fail_open",
    )
    first_arm, second_arm = config["arms"]
    for manifest in manifests:
        case_id = manifest["case_id"]
        first_result = results[(case_id, first_arm)]
        second_result = results[(case_id, second_arm)]
        first = first_result["pairing"]
        second = second_result["pairing"]
        precontrol_geometry_failure = (
            _validate_precontrol_geometry_failure(
                second_result,
                aegis_arm=str(second_arm),
                pairing=second,
            )
        )
        changed = [
            field
            for field in paired_fields
            if first.get(field) != second.get(field)
        ]
        if changed:
            raise AggregationError(
                f"{case_id}: arms are not paired for {changed}"
            )
        baseline_action_hash = _require_sha256(
            first.get("initial_policy_action_chunk_sha256"),
            label=(
                f"{case_id}/{first_arm}/"
                "initial_policy_action_chunk_sha256"
            ),
        )
        baseline_queries = first_result.get("policy_queries")
        if (
            not isinstance(baseline_queries, list)
            or not baseline_queries
            or not isinstance(baseline_queries[0], Mapping)
            or baseline_queries[0].get("returned_actions_sha256")
            != baseline_action_hash
        ):
            raise AggregationError(
                f"{case_id}: baseline initial policy action is not bound"
            )
        if precontrol_geometry_failure:
            if "initial_policy_action_chunk_sha256" in second:
                raise AggregationError(
                    f"{case_id}: precontrol geometry failure has an "
                    "initial policy action hash"
                )
        else:
            aegis_action_hash = _require_sha256(
                second.get("initial_policy_action_chunk_sha256"),
                label=(
                    f"{case_id}/{second_arm}/"
                    "initial_policy_action_chunk_sha256"
                ),
            )
            aegis_queries = second_result.get("policy_queries")
            if (
                not isinstance(aegis_queries, list)
                or not aegis_queries
                or not isinstance(aegis_queries[0], Mapping)
                or aegis_queries[0].get("returned_actions_sha256")
                != aegis_action_hash
            ):
                raise AggregationError(
                    f"{case_id}: AEGIS initial policy action is not bound"
                )
            if baseline_action_hash != aegis_action_hash:
                raise AggregationError(
                    f"{case_id}: arms are not paired for "
                    "['initial_policy_action_chunk_sha256']"
                )


def summarize_episode_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise AggregationError("cannot summarize an empty result group")
    metrics = [row["metrics"] for row in rows]
    status_counts = Counter(row["status"] for row in rows)
    return {
        "episodes": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "retained_method_failures": sum(
            count
            for status, count in status_counts.items()
            if status.startswith("method_failure")
        ),
        "car_percent": 100.0
        * sum(not row["public_collision"] for row in metrics)
        / len(metrics),
        "tsr_percent": 100.0
        * sum(row["task_success"] for row in metrics)
        / len(metrics),
        "legacy_ets_steps_mean": sum(
            row["legacy_ets_steps"] for row in metrics
        )
        / len(metrics),
        "executed_action_count_mean": sum(
            row["executed_action_count"] for row in metrics
        )
        / len(metrics),
    }


def mean_summaries(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not summaries:
        raise AggregationError("cannot average an empty summary collection")
    status_counts: Counter[str] = Counter()
    for summary in summaries:
        status_counts.update(summary["status_counts"])
    count = len(summaries)
    return {
        "groups": count,
        "episodes": sum(summary["episodes"] for summary in summaries),
        "status_counts": dict(sorted(status_counts.items())),
        "retained_method_failures": sum(
            summary["retained_method_failures"] for summary in summaries
        ),
        "car_percent": sum(
            summary["car_percent"] for summary in summaries
        )
        / count,
        "tsr_percent": sum(
            summary["tsr_percent"] for summary in summaries
        )
        / count,
        "legacy_ets_steps_mean": sum(
            summary["legacy_ets_steps_mean"] for summary in summaries
        )
        / count,
        "executed_action_count_mean": sum(
            summary["executed_action_count_mean"]
            for summary in summaries
        )
        / count,
    }


def aggregate(
    *,
    config: dict[str, Any],
    manifests: list[dict[str, Any]],
    results: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    manifest_by_id = {row["case_id"]: row for row in manifests}
    group_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    suite_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    overall: dict[str, dict[str, Any]] = {}

    for arm in config["arms"]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case_id in manifest_by_id:
            group_id = manifest_by_id[case_id]["task_level_group_id"]
            grouped[group_id].append(results[(case_id, arm)])
        arm_groups = {
            group_id: summarize_episode_rows(rows)
            for group_id, rows in sorted(grouped.items())
        }
        group_summaries[arm] = arm_groups

        arm_suites: dict[str, dict[str, Any]] = {}
        for suite in config["population"]["suite_order"]:
            suite_short = suite.removeprefix("safelibero_")
            summaries = [
                value
                for group_id, value in arm_groups.items()
                if group_id.startswith(f"vlsa-t1-{suite_short}-")
            ]
            if len(summaries) != 8:
                raise AggregationError(
                    f"{arm}/{suite}: expected eight group summaries"
                )
            arm_suites[suite] = mean_summaries(summaries)
        suite_summaries[arm] = arm_suites
        overall[arm] = mean_summaries(list(arm_suites.values()))

    first_arm, second_arm = config["arms"]
    transitions = Counter()
    for manifest in manifests:
        case_id = manifest["case_id"]
        baseline = results[(case_id, first_arm)]["metrics"]
        aegis = results[(case_id, second_arm)]["metrics"]
        transitions[
            (
                "baseline_safe" if not baseline["public_collision"] else "baseline_collision",
                "aegis_safe" if not aegis["public_collision"] else "aegis_collision",
            )
        ] += 1
        transitions[
            (
                "baseline_success" if baseline["task_success"] else "baseline_failure",
                "aegis_success" if aegis["task_success"] else "aegis_failure",
            )
        ] += 1

    target_differences: dict[str, dict[str, dict[str, float]]] = {}
    for arm in config["arms"]:
        arm_differences: dict[str, dict[str, float]] = {}
        targets = config["published_table1_targets"][arm]
        for scope in [*config["population"]["suite_order"], "average"]:
            observed = (
                overall[arm] if scope == "average" else suite_summaries[arm][scope]
            )
            target = targets[scope]
            arm_differences[scope] = {
                "car_percent_delta": observed["car_percent"]
                - target["car_percent"],
                "tsr_percent_delta": observed["tsr_percent"]
                - target["tsr_percent"],
                "legacy_ets_steps_delta": observed[
                    "legacy_ets_steps_mean"
                ]
                - target["ets_steps"],
            }
        target_differences[arm] = arm_differences

    return {
        "schema_version": OUTPUT_SCHEMA,
        "status": "complete_population_validated",
        "protocol_id": config["protocol_id"],
        "accepted_result_payloads_sha256": sha256_bytes(
            canonical_json_bytes(
                [
                    {
                        "case_id": case_id,
                        "arm": arm,
                        "result_payload_sha256": results[
                            (case_id, arm)
                        ]["result_payload_sha256"],
                    }
                    for case_id, arm in sorted(results)
                ]
            )
        ),
        "population": {
            "cases": len(manifests),
            "arms": len(config["arms"]),
            "results": len(results),
            "task_level_groups": 32,
            "no_results_dropped": True,
        },
        "metric_semantics": {
            "car": config["metrics"]["car"]["collision_definition"],
            "tsr": config["metrics"]["tsr"]["success_definition"],
            "table_ets": config["metrics"]["ets"]["table_value"],
            "executed_actions_reported_separately": True,
        },
        "task_level_groups": group_summaries,
        "suites": suite_summaries,
        "average": overall,
        "paired_transitions": {
            " | ".join(key): count
            for key, count in sorted(transitions.items())
        },
        "published_table1_differences": target_differences,
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(
            value,
            stream,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config, manifests = load_protocol(
        args.config, args.receipt, args.manifest
    )
    results = load_results(
        args.results, config=config, manifests=manifests
    )
    validate_pairs(
        config=config, manifests=manifests, results=results
    )
    summary = aggregate(
        config=config, manifests=manifests, results=results
    )
    if not all(
        math.isfinite(value)
        for arm in summary["average"].values()
        for key, value in arm.items()
        if key.endswith("_percent")
        or key.endswith("_mean")
    ):
        raise AggregationError("aggregate contains nonfinite metrics")
    atomic_json(args.output, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
