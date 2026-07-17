#!/usr/bin/env python3
"""Strictly aggregate the complete paired VLSA Table 1 population.

No result is silently excluded. Algorithmic method failures remain in every
denominator, while apparatus failures and incomplete populations prevent an
aggregate from being published.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


CONFIG_SCHEMA = "vlsa_table1_translational_protocol.v1"
MANIFEST_SCHEMA = "vlsa_table1_population_case.v1"
RECEIPT_SCHEMA = "vlsa_table1_population_receipt.v1"
OUTPUT_SCHEMA = "vlsa_table1_population_summary.v1"


class AggregationError(RuntimeError):
    """Raised when the population is incomplete or scientifically invalid."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AggregationError(f"{label} must be a lowercase SHA-256")
    return value


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
    for key in ("public_collision", "task_success"):
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
    return metrics


def validate_result(
    result: dict[str, Any],
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    case_id = manifest["case_id"]
    arm = result.get("arm")
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
    expected_manifest_hash = canonical_record_sha256(manifest)
    if pairing.get("manifest_row_sha256") != expected_manifest_hash:
        raise AggregationError(
            f"{case_id}/{arm}: manifest-row binding changed"
        )
    for field in (
        "initial_state_sha256",
        "initial_observation_sha256",
        "policy_noise_schedule_sha256",
        "semantic_label_record_sha256",
        "semantic_label_settled_agentview_sha256",
    ):
        _require_sha256(
            pairing.get(field), label=f"{case_id}/{arm}/{field}"
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
    ):
        raise AggregationError(
            f"{case_id}/{arm}: pairing protocol differs from manifest"
        )

    validated = dict(result)
    validated["metrics"] = _validate_metrics(result, manifest)
    return validated


def load_results(
    result_paths: Iterable[Path],
    *,
    config: dict[str, Any],
    manifests: list[dict[str, Any]],
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
            results[key] = validate_result(
                raw, config=config, manifest=manifest
            )

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
        "policy_noise_schedule_id",
        "policy_noise_schedule_sha256",
        "semantic_label_record_sha256",
        "semantic_label_settled_agentview_sha256",
        "semantic_obstacle_label",
        "max_steps",
        "model_action_horizon",
        "replan_steps",
    )
    first_arm, second_arm = config["arms"]
    for manifest in manifests:
        case_id = manifest["case_id"]
        first = results[(case_id, first_arm)]["pairing"]
        second = results[(case_id, second_arm)]["pairing"]
        changed = [
            field
            for field in paired_fields
            if first[field] != second[field]
        ]
        if changed:
            raise AggregationError(
                f"{case_id}: arms are not paired for {changed}"
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
