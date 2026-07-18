#!/usr/bin/env python3
"""Render a fail-closed SafeLIBERO Table-1 comparison.

The renderer accepts only the complete post-publication v2 population summary,
its terminal analysis-v2 publication receipt, and the exact frozen
translational protocol containing the paper targets. It does not read episode
results, repair an aggregate, or infer missing values. Every receipt binding,
count hierarchy, and paired transition is checked before any output directory
is created.
"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


SUMMARY_SCHEMA = "vlsa_table1_population_summary.v2"
SUMMARY_STATUS = "complete_postpublication_analysis_v2"
ANALYSIS_RECEIPT_SCHEMA = (
    "vlsa_table1_postpublication_analysis_v2_receipt.v1"
)
ANALYSIS_RECEIPT_STATUS = "published_derived_analysis_v2"
TARGET_CONFIG_SCHEMA = "vlsa_table1_translational_protocol.v1"
RECEIPT_SCHEMA = "vlsa_table1_table_report.v1"
PROTOCOL_ID = "vlsa-table1-translational-1600-v1"
EXPECTED_RUNTIME_COMMIT = "1592aa59361f431ba96c6ddcbebcb596f6c20853"
EXPECTED_RUN_ID = "vlsa-table1-contact-authority-population-20260718a"
ANALYSIS_PUBLICATION_ROOT = Path(
    "/mnt/data/quanth/experiments/vlsa-aegis-table1-analysis-v2"
)
ANALYSIS_PUBLISHER_JOB_ID = "28610"
EXPECTED_TARGET_CONFIG_SHA256 = (
    "7dec2cdd473e6809a755d908711efd457c5fdd36ca9655ab6a15e0c9c4bdbfa6"
)
ARMS = (
    "pi05_translational",
    "pi05_plus_aegis_translational",
)
SUITES = (
    "safelibero_spatial",
    "safelibero_goal",
    "safelibero_object",
    "safelibero_long",
)
SCOPES = (*SUITES, "average")
SUITE_LABELS = {
    "safelibero_spatial": "Spatial",
    "safelibero_goal": "Goal",
    "safelibero_object": "Object",
    "safelibero_long": "Long",
    "average": "Average",
}
METHOD_LABELS_MD = {
    "pi05_translational": "π0.5",
    "pi05_plus_aegis_translational": "π0.5+AEGIS",
}
METHOD_LABELS_TEX = {
    "pi05_translational": r"$\pi_{0.5}$",
    "pi05_plus_aegis_translational": r"$\pi_{0.5}+\mathrm{AEGIS}$",
}
EXPECTED_CASES = 1600
EXPECTED_RESULTS = 3200
EXPECTED_GROUPS = 32
EXPECTED_SUITE_CASES = 400
EXPECTED_GROUP_CASES = 50
EXPECTED_GROUP_IDS = frozenset(
    f"vlsa-t1-{suite[len('safelibero_'):]}-{level}-t{task}"
    for suite in SUITES
    for level in ("i", "ii")
    for task in range(4)
)
SCIENTIFIC_STATUSES = {
    "complete",
    "method_failure",
    "method_failure_passthrough",
}
JOINT_OUTCOMES = (
    "safe_task_success",
    "safe_task_failure",
    "collision_task_success",
    "collision_task_failure",
)
CAR_TRANSITIONS = (
    "baseline_safe_to_aegis_safe",
    "baseline_safe_to_aegis_collision",
    "baseline_collision_to_aegis_safe",
    "baseline_collision_to_aegis_collision",
)
TASK_TRANSITIONS = (
    "baseline_success_to_aegis_success",
    "baseline_success_to_aegis_failure",
    "baseline_failure_to_aegis_success",
    "baseline_failure_to_aegis_failure",
)
JOINT_TRANSITIONS = tuple(
    f"baseline_{baseline}_to_aegis_{aegis}"
    for baseline in JOINT_OUTCOMES
    for aegis in JOINT_OUTCOMES
)
METRIC_SPECS = (
    ("CAR", "car", "percent"),
    ("TSR", "tsr", "percent"),
    ("ETS", "legacy_ets_steps", "mean"),
)


class TableReportError(RuntimeError):
    """Raised when inputs are not a complete validated population."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise TableReportError(f"JSON contains nonfinite number {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise TableReportError(f"JSON contains duplicate key {key!r}")
        output[key] = value
    return output


def load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise TableReportError(f"{label} is missing or symlinked")
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            parse_float=Decimal,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except TableReportError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TableReportError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise TableReportError(f"{label} must be a JSON object")
    return value, raw


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TableReportError(f"{label} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: Sequence[str] | set[str],
    *,
    label: str,
) -> None:
    expected_set = set(expected)
    if set(value) != expected_set:
        missing = sorted(expected_set - set(value))
        extra = sorted(set(value) - expected_set)
        raise TableReportError(
            f"{label} keys differ: missing={missing}, extra={extra}"
        )


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TableReportError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, *, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise TableReportError(f"{label} must be a finite number")
    result = Decimal(str(value))
    if not result.is_finite():
        raise TableReportError(f"{label} must be a finite number")
    return result


def _sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TableReportError(f"{label} must be lowercase SHA-256")
    return value


def _close(
    observed: Any,
    expected: Decimal,
    *,
    label: str,
    tolerance: Decimal = Decimal("1e-9"),
) -> Decimal:
    value = _number(observed, label=label)
    if abs(value - expected) > tolerance:
        raise TableReportError(
            f"{label} differs: observed={value}, expected={expected}"
        )
    return value


def _validated_status_counts(
    value: Any,
    *,
    denominator: int,
    arm: str,
    label: str,
) -> dict[str, int]:
    counts = _mapping(value, label=label)
    if not set(counts).issubset(SCIENTIFIC_STATUSES):
        unsafe = sorted(set(counts) - SCIENTIFIC_STATUSES)
        raise TableReportError(
            f"{label} contains apparatus/non-scientific statuses: {unsafe}"
        )
    normalized = {
        status: _integer(count, label=f"{label}/{status}")
        for status, count in counts.items()
    }
    if sum(normalized.values()) != denominator:
        raise TableReportError(f"{label} does not sum to {denominator}")
    if arm == ARMS[0] and normalized != {"complete": denominator}:
        raise TableReportError(
            f"{label} nominal baseline contains a method failure"
        )
    return normalized


def _validated_rate(
    value: Any,
    *,
    denominator: int,
    label: str,
) -> dict[str, Any]:
    record = _mapping(value, label=label)
    _exact_keys(
        record,
        {"success_count", "failure_count", "denominator", "percent"},
        label=label,
    )
    success = _integer(record["success_count"], label=f"{label}/success")
    failure = _integer(record["failure_count"], label=f"{label}/failure")
    exact_denominator = _integer(
        record["denominator"], label=f"{label}/denominator"
    )
    if exact_denominator != denominator or success + failure != denominator:
        raise TableReportError(f"{label} exact denominator is inconsistent")
    expected_percent = Decimal(100 * success) / Decimal(denominator)
    percent = _close(
        record["percent"],
        expected_percent,
        label=f"{label}/percent",
    )
    return {
        "success_count": success,
        "failure_count": failure,
        "denominator": denominator,
        "percent": percent,
    }


def _validated_mean(
    value: Any,
    *,
    denominator: int,
    label: str,
) -> dict[str, Any]:
    record = _mapping(value, label=label)
    _exact_keys(record, {"sum", "denominator", "mean"}, label=label)
    total = _integer(record["sum"], label=f"{label}/sum")
    exact_denominator = _integer(
        record["denominator"], label=f"{label}/denominator"
    )
    if exact_denominator != denominator:
        raise TableReportError(f"{label} denominator differs")
    mean = _close(
        record["mean"],
        Decimal(total) / Decimal(denominator),
        label=f"{label}/mean",
    )
    return {"sum": total, "denominator": denominator, "mean": mean}


def validate_metric_block(
    value: Any,
    *,
    arm: str,
    denominator: int,
    expected_groups: int | None,
    label: str,
) -> dict[str, Any]:
    block = _mapping(value, label=label)
    episodes = _integer(block.get("episodes"), label=f"{label}/episodes")
    if episodes != denominator:
        raise TableReportError(
            f"{label} has {episodes} episodes, expected {denominator}"
        )
    if expected_groups is not None:
        groups = _integer(block.get("groups"), label=f"{label}/groups")
        if groups != expected_groups:
            raise TableReportError(
                f"{label} has {groups} groups, expected {expected_groups}"
            )
    status_counts = _validated_status_counts(
        block.get("status_counts"),
        denominator=denominator,
        arm=arm,
        label=f"{label}/status_counts",
    )
    retained = _integer(
        block.get("retained_method_failures"),
        label=f"{label}/retained_method_failures",
    )
    expected_retained = sum(
        count
        for status, count in status_counts.items()
        if status.startswith("method_failure")
    )
    if retained != expected_retained:
        raise TableReportError(
            f"{label} retained-method-failure count differs"
        )
    car = _validated_rate(
        block.get("car"), denominator=denominator, label=f"{label}/car"
    )
    tsr = _validated_rate(
        block.get("tsr"), denominator=denominator, label=f"{label}/tsr"
    )
    ets = _validated_mean(
        block.get("legacy_ets_steps"),
        denominator=denominator,
        label=f"{label}/legacy_ets_steps",
    )
    actions = _validated_mean(
        block.get("executed_action_count"),
        denominator=denominator,
        label=f"{label}/executed_action_count",
    )
    _close(
        block.get("car_percent"),
        car["percent"],
        label=f"{label}/car_percent",
    )
    _close(
        block.get("tsr_percent"),
        tsr["percent"],
        label=f"{label}/tsr_percent",
    )
    _close(
        block.get("legacy_ets_steps_mean"),
        ets["mean"],
        label=f"{label}/legacy_ets_steps_mean",
    )
    _close(
        block.get("executed_action_count_mean"),
        actions["mean"],
        label=f"{label}/executed_action_count_mean",
    )
    joint_raw = _mapping(
        block.get("joint_outcome_counts"),
        label=f"{label}/joint_outcome_counts",
    )
    _exact_keys(
        joint_raw, set(JOINT_OUTCOMES), label=f"{label}/joint_outcome_counts"
    )
    joint = {
        key: _integer(
            joint_raw[key], label=f"{label}/joint_outcome_counts/{key}"
        )
        for key in JOINT_OUTCOMES
    }
    if sum(joint.values()) != denominator:
        raise TableReportError(f"{label} joint outcomes do not sum")
    if (
        joint["safe_task_success"] + joint["safe_task_failure"]
        != car["success_count"]
        or joint["safe_task_success"] + joint["collision_task_success"]
        != tsr["success_count"]
    ):
        raise TableReportError(
            f"{label} joint outcomes disagree with CAR/TSR"
        )
    return {
        "episodes": denominator,
        "groups": expected_groups,
        "status_counts": status_counts,
        "retained_method_failures": retained,
        "car": car,
        "tsr": tsr,
        "legacy_ets_steps": ets,
        "executed_action_count": actions,
        "joint_outcome_counts": joint,
    }


def _sum_blocks(
    children: Sequence[Mapping[str, Any]],
    parent: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if sum(child["episodes"] for child in children) != parent["episodes"]:
        raise TableReportError(f"{label} episode hierarchy differs")
    statuses: Counter[str] = Counter()
    for child in children:
        statuses.update(child["status_counts"])
    if dict(statuses) != dict(parent["status_counts"]):
        raise TableReportError(f"{label} status hierarchy differs")
    if (
        sum(child["retained_method_failures"] for child in children)
        != parent["retained_method_failures"]
    ):
        raise TableReportError(f"{label} method-failure hierarchy differs")
    for metric, total_field in (
        ("car", "success_count"),
        ("tsr", "success_count"),
        ("legacy_ets_steps", "sum"),
        ("executed_action_count", "sum"),
    ):
        if (
            sum(child[metric][total_field] for child in children)
            != parent[metric][total_field]
        ):
            raise TableReportError(f"{label}/{metric} hierarchy differs")
    for outcome in JOINT_OUTCOMES:
        if (
            sum(child["joint_outcome_counts"][outcome] for child in children)
            != parent["joint_outcome_counts"][outcome]
        ):
            raise TableReportError(f"{label}/{outcome} hierarchy differs")


def validate_targets(
    config: Mapping[str, Any],
    *,
    raw_sha256: str,
) -> dict[str, dict[str, dict[str, Decimal]]]:
    if raw_sha256 != EXPECTED_TARGET_CONFIG_SHA256:
        raise TableReportError(
            "paper-target config hash differs from the frozen protocol"
        )
    if (
        config.get("schema_version") != TARGET_CONFIG_SCHEMA
        or config.get("protocol_id") != PROTOCOL_ID
        or tuple(config.get("arms", ())) != ARMS
    ):
        raise TableReportError("paper-target protocol identity differs")
    population = _mapping(
        config.get("population"), label="paper targets/population"
    )
    if (
        tuple(population.get("suite_order", ())) != SUITES
        or population.get("expected_cases") != EXPECTED_CASES
        or population.get("expected_cases_per_task_level_group")
        != EXPECTED_GROUP_CASES
    ):
        raise TableReportError("paper-target population contract differs")
    targets_raw = _mapping(
        config.get("published_table1_targets"),
        label="published_table1_targets",
    )
    _exact_keys(targets_raw, set(ARMS), label="published_table1_targets")
    targets: dict[str, dict[str, dict[str, Decimal]]] = {}
    for arm in ARMS:
        arm_raw = _mapping(
            targets_raw[arm], label=f"published_table1_targets/{arm}"
        )
        _exact_keys(
            arm_raw, set(SCOPES), label=f"published_table1_targets/{arm}"
        )
        arm_targets: dict[str, dict[str, Decimal]] = {}
        for scope in SCOPES:
            record = _mapping(
                arm_raw[scope],
                label=f"published_table1_targets/{arm}/{scope}",
            )
            _exact_keys(
                record,
                {"car_percent", "tsr_percent", "ets_steps"},
                label=f"published_table1_targets/{arm}/{scope}",
            )
            normalized = {
                key: _number(
                    record[key],
                    label=f"published_table1_targets/{arm}/{scope}/{key}",
                )
                for key in ("car_percent", "tsr_percent", "ets_steps")
            }
            if not (
                Decimal(0) <= normalized["car_percent"] <= Decimal(100)
                and Decimal(0) <= normalized["tsr_percent"] <= Decimal(100)
                and normalized["ets_steps"] >= 0
            ):
                raise TableReportError(
                    f"published_table1_targets/{arm}/{scope} is out of range"
                )
            arm_targets[scope] = normalized
        for metric in ("car_percent", "tsr_percent", "ets_steps"):
            arithmetic = sum(
                arm_targets[suite][metric] for suite in SUITES
            ) / Decimal(len(SUITES))
            rounded = arithmetic.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            if arm_targets["average"][metric] != rounded:
                raise TableReportError(
                    f"published {arm}/{metric} average is not the "
                    "two-decimal arithmetic suite average"
                )
        targets[arm] = arm_targets
    return targets


def _validate_transition_counts(
    value: Any,
    keys: Sequence[str],
    *,
    denominator: int,
    label: str,
) -> dict[str, int]:
    record = _mapping(value, label=label)
    _exact_keys(record, set(keys), label=label)
    counts = {
        key: _integer(record[key], label=f"{label}/{key}") for key in keys
    }
    if sum(counts.values()) != denominator:
        raise TableReportError(f"{label} does not sum to {denominator}")
    return counts


def _validate_transitions(
    value: Any,
    *,
    baseline: Mapping[str, Any],
    aegis: Mapping[str, Any],
    denominator: int,
    label: str,
) -> dict[str, dict[str, int]]:
    record = _mapping(value, label=label)
    _exact_keys(
        record,
        {"denominator", "arms", "car", "task", "joint"},
        label=label,
    )
    if _integer(record.get("denominator"), label=f"{label}/denominator") != denominator:
        raise TableReportError(f"{label} denominator differs")
    if record.get("arms") != {"baseline": ARMS[0], "aegis": ARMS[1]}:
        raise TableReportError(f"{label} arm identity differs")
    car = _validate_transition_counts(
        record.get("car"), CAR_TRANSITIONS, denominator=denominator, label=f"{label}/car"
    )
    task = _validate_transition_counts(
        record.get("task"), TASK_TRANSITIONS, denominator=denominator, label=f"{label}/task"
    )
    joint = _validate_transition_counts(
        record.get("joint"),
        JOINT_TRANSITIONS,
        denominator=denominator,
        label=f"{label}/joint",
    )
    projected_car: Counter[str] = Counter()
    projected_task: Counter[str] = Counter()
    for baseline_outcome in JOINT_OUTCOMES:
        for aegis_outcome in JOINT_OUTCOMES:
            count = joint[
                f"baseline_{baseline_outcome}_to_aegis_{aegis_outcome}"
            ]
            projected_car[
                (
                    "baseline_"
                    + (
                        "safe"
                        if baseline_outcome.startswith("safe_")
                        else "collision"
                    )
                    + "_to_aegis_"
                    + (
                        "safe"
                        if aegis_outcome.startswith("safe_")
                        else "collision"
                    )
                )
            ] += count
            projected_task[
                (
                    "baseline_"
                    + (
                        "success"
                        if baseline_outcome.endswith("_success")
                        else "failure"
                    )
                    + "_to_aegis_"
                    + (
                        "success"
                        if aegis_outcome.endswith("_success")
                        else "failure"
                    )
                )
            ] += count
    if (
        car != {key: projected_car[key] for key in CAR_TRANSITIONS}
        or task != {key: projected_task[key] for key in TASK_TRANSITIONS}
    ):
        raise TableReportError(
            f"{label} CAR/task transitions are not exact joint projections"
        )
    if (
        car["baseline_safe_to_aegis_safe"]
        + car["baseline_safe_to_aegis_collision"]
        != baseline["car"]["success_count"]
        or car["baseline_safe_to_aegis_safe"]
        + car["baseline_collision_to_aegis_safe"]
        != aegis["car"]["success_count"]
        or task["baseline_success_to_aegis_success"]
        + task["baseline_success_to_aegis_failure"]
        != baseline["tsr"]["success_count"]
        or task["baseline_success_to_aegis_success"]
        + task["baseline_failure_to_aegis_success"]
        != aegis["tsr"]["success_count"]
    ):
        raise TableReportError(f"{label} CAR/TSR transitions disagree")
    for baseline_outcome in JOINT_OUTCOMES:
        if (
            sum(
                joint[
                    f"baseline_{baseline_outcome}_to_aegis_{aegis_outcome}"
                ]
                for aegis_outcome in JOINT_OUTCOMES
            )
            != baseline["joint_outcome_counts"][baseline_outcome]
        ):
            raise TableReportError(
                f"{label} baseline joint transition marginal differs"
            )
    for aegis_outcome in JOINT_OUTCOMES:
        if (
            sum(
                joint[
                    f"baseline_{baseline_outcome}_to_aegis_{aegis_outcome}"
                ]
                for baseline_outcome in JOINT_OUTCOMES
            )
            != aegis["joint_outcome_counts"][aegis_outcome]
        ):
            raise TableReportError(
                f"{label} AEGIS joint transition marginal differs"
            )
    return {"car": car, "task": task, "joint": joint}


def _validate_differences(
    value: Any,
    *,
    blocks: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    differences = _mapping(value, label="method_differences")
    _exact_keys(differences, set(SCOPES), label="method_differences")
    for scope in SCOPES:
        record = _mapping(
            differences[scope], label=f"method_differences/{scope}"
        )
        _exact_keys(
            record,
            {
                "denominator",
                "car_success_count_delta",
                "car_percentage_point_delta",
                "tsr_success_count_delta",
                "tsr_percentage_point_delta",
                "legacy_ets_steps_sum_delta",
                "legacy_ets_steps_mean_delta",
                "executed_action_count_sum_delta",
                "executed_action_count_mean_delta",
            },
            label=f"method_differences/{scope}",
        )
        baseline = blocks[ARMS[0]][scope]
        aegis = blocks[ARMS[1]][scope]
        if _integer(
            record.get("denominator"),
            label=f"method_differences/{scope}/denominator",
        ) != baseline["episodes"]:
            raise TableReportError(
                f"method_differences/{scope} denominator differs"
            )
        integer_expected = {
            "car_success_count_delta": (
                aegis["car"]["success_count"]
                - baseline["car"]["success_count"]
            ),
            "tsr_success_count_delta": (
                aegis["tsr"]["success_count"]
                - baseline["tsr"]["success_count"]
            ),
            "legacy_ets_steps_sum_delta": (
                aegis["legacy_ets_steps"]["sum"]
                - baseline["legacy_ets_steps"]["sum"]
            ),
            "executed_action_count_sum_delta": (
                aegis["executed_action_count"]["sum"]
                - baseline["executed_action_count"]["sum"]
            ),
        }
        for key, expected in integer_expected.items():
            if _integer(
                record.get(key),
                label=f"method_differences/{scope}/{key}",
                minimum=min(0, expected),
            ) != expected:
                raise TableReportError(
                    f"method_differences/{scope}/{key} differs"
                )
        decimal_expected = {
            "car_percentage_point_delta": (
                aegis["car"]["percent"] - baseline["car"]["percent"]
            ),
            "tsr_percentage_point_delta": (
                aegis["tsr"]["percent"] - baseline["tsr"]["percent"]
            ),
            "legacy_ets_steps_mean_delta": (
                aegis["legacy_ets_steps"]["mean"]
                - baseline["legacy_ets_steps"]["mean"]
            ),
            "executed_action_count_mean_delta": (
                aegis["executed_action_count"]["mean"]
                - baseline["executed_action_count"]["mean"]
            ),
        }
        for key, expected in decimal_expected.items():
            _close(
                record.get(key),
                expected,
                label=f"method_differences/{scope}/{key}",
            )


def validate_summary(
    summary: Mapping[str, Any],
    *,
    targets: Mapping[str, Mapping[str, Mapping[str, Decimal]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("status") != SUMMARY_STATUS
        or summary.get("protocol_id") != PROTOCOL_ID
    ):
        raise TableReportError(
            "summary is not the complete post-publication v2 population"
        )
    population = _mapping(summary.get("population"), label="population")
    if (
        population.get("cases") != EXPECTED_CASES
        or population.get("arms") != len(ARMS)
        or population.get("results") != EXPECTED_RESULTS
        or population.get("task_level_groups") != EXPECTED_GROUPS
        or population.get("no_results_dropped") is not True
    ):
        raise TableReportError(
            "summary is partial or does not retain every result"
        )
    accepted_ledger = _sha(
        summary.get("accepted_result_payloads_sha256"),
        label="accepted result ledger",
    )
    source = _mapping(summary.get("source_v1"), label="source_v1")
    if source.get("v1_source_git_commit") != EXPECTED_RUNTIME_COMMIT:
        raise TableReportError("summary source runtime commit differs")
    if source.get("v1_run_id") != EXPECTED_RUN_ID:
        raise TableReportError("summary source run ID differs")
    required_source_hashes = {
        "v1_publication_receipt_sha256",
        "v1_population_summary_sha256",
        "v1_prepublish_receipt_sha256",
        "v1_accepted_result_payloads_sha256",
    }
    if not required_source_hashes.issubset(source):
        raise TableReportError("summary source-v1 binding is incomplete")
    for key, value in source.items():
        if key.endswith("_sha256"):
            _sha(value, label=f"source_v1/{key}")
    if source.get("v1_accepted_result_payloads_sha256") != accepted_ledger:
        raise TableReportError("summary accepted-result ledger differs")
    claim = _mapping(summary.get("claim_scope"), label="claim_scope")
    required_claims = {
        "openvla_oft_included": False,
        "paper_semantic_selector_reproduced": False,
        "paper_exact_end_to_end_reproduction_claimed": False,
        "clearance_available": False,
        "minimum_clearance_claimed": False,
    }
    for key, expected in required_claims.items():
        if claim.get(key) is not expected:
            raise TableReportError(f"claim_scope/{key} differs")
    required_claim_strings = {
        "method_label": (
            "pi0.5 + AEGIS translational conditioned on frozen "
            "per-case Codex obstacle labels"
        ),
        "baseline_method_label": "pi0.5 translational",
        "aegis_method_label": (
            "pi0.5 + AEGIS translational conditioned on frozen "
            "per-case Codex obstacle labels"
        ),
        "population_scope": (
            "complete frozen 1,600-case SafeLIBERO population for "
            "the two registered translational arms"
        ),
    }
    for key, expected in required_claim_strings.items():
        if claim.get(key) != expected:
            raise TableReportError(f"claim_scope/{key} differs")
    if (
        claim.get("table_scope")
        != "two-row translational Table-1 reproduction: pi0.5 and pi0.5+AEGIS only"
    ):
        raise TableReportError("claim_scope/table_scope differs")
    semantics = _mapping(
        summary.get("metric_semantics"), label="metric_semantics"
    )
    expected_semantics = {
        "car": (
            "maximum active-obstacle L1 position displacement strictly "
            "greater than 0.001 m"
        ),
        "tsr": "SafeLIBERO simulator done",
        "table_ets": (
            "mean released zero-based control-loop counter legacy_ets_steps"
        ),
        "executed_actions_reported_separately": True,
    }
    _exact_keys(
        semantics, set(expected_semantics), label="metric_semantics"
    )
    for key, expected in expected_semantics.items():
        if semantics.get(key) != expected:
            raise TableReportError(f"metric_semantics/{key} differs")

    groups_raw = _mapping(
        summary.get("task_level_groups"), label="task_level_groups"
    )
    suites_raw = _mapping(summary.get("suites"), label="suites")
    average_raw = _mapping(summary.get("average"), label="average")
    _exact_keys(groups_raw, set(ARMS), label="task_level_groups")
    _exact_keys(suites_raw, set(ARMS), label="suites")
    _exact_keys(average_raw, set(ARMS), label="average")
    blocks: dict[str, dict[str, dict[str, Any]]] = {}
    group_names: set[str] | None = None
    for arm in ARMS:
        arm_groups_raw = _mapping(
            groups_raw[arm], label=f"task_level_groups/{arm}"
        )
        if len(arm_groups_raw) != EXPECTED_GROUPS:
            raise TableReportError(
                f"task_level_groups/{arm} does not have 32 groups"
            )
        if set(arm_groups_raw) != EXPECTED_GROUP_IDS:
            raise TableReportError(
                f"task_level_groups/{arm} identities differ from the "
                "frozen 32-group protocol"
            )
        if group_names is None:
            group_names = set(arm_groups_raw)
        elif set(arm_groups_raw) != group_names:
            raise TableReportError("arms use different task-level groups")
        arm_group_blocks = {
            group_id: validate_metric_block(
                block,
                arm=arm,
                denominator=EXPECTED_GROUP_CASES,
                expected_groups=None,
                label=f"task_level_groups/{arm}/{group_id}",
            )
            for group_id, block in arm_groups_raw.items()
        }
        arm_suites_raw = _mapping(
            suites_raw[arm], label=f"suites/{arm}"
        )
        _exact_keys(arm_suites_raw, set(SUITES), label=f"suites/{arm}")
        arm_blocks: dict[str, dict[str, Any]] = {}
        for suite in SUITES:
            suite_block = validate_metric_block(
                arm_suites_raw[suite],
                arm=arm,
                denominator=EXPECTED_SUITE_CASES,
                expected_groups=8,
                label=f"suites/{arm}/{suite}",
            )
            prefix = f"vlsa-t1-{suite[len('safelibero_'):]}-"
            selected = [
                block
                for group_id, block in arm_group_blocks.items()
                if group_id.startswith(prefix)
            ]
            if len(selected) != 8:
                raise TableReportError(
                    f"task_level_groups/{arm}/{suite} does not have 8 groups"
                )
            _sum_blocks(
                selected,
                suite_block,
                label=f"task_level_groups-to-suite/{arm}/{suite}",
            )
            arm_blocks[suite] = suite_block
        average = validate_metric_block(
            average_raw[arm],
            arm=arm,
            denominator=EXPECTED_CASES,
            expected_groups=EXPECTED_GROUPS,
            label=f"average/{arm}",
        )
        _sum_blocks(
            [arm_blocks[suite] for suite in SUITES],
            average,
            label=f"suites-to-average/{arm}",
        )
        arm_blocks["average"] = average
        blocks[arm] = arm_blocks

    differences = _mapping(
        summary.get("published_table1_differences"),
        label="published_table1_differences",
    )
    _exact_keys(
        differences, set(ARMS), label="published_table1_differences"
    )
    for arm in ARMS:
        arm_differences = _mapping(
            differences[arm],
            label=f"published_table1_differences/{arm}",
        )
        _exact_keys(
            arm_differences,
            set(SCOPES),
            label=f"published_table1_differences/{arm}",
        )
        for scope in SCOPES:
            record = _mapping(
                arm_differences[scope],
                label=f"published_table1_differences/{arm}/{scope}",
            )
            _exact_keys(
                record,
                {
                    "car_percent_delta",
                    "tsr_percent_delta",
                    "legacy_ets_steps_delta",
                },
                label=f"published_table1_differences/{arm}/{scope}",
            )
            expected = {
                "car_percent_delta": (
                    blocks[arm][scope]["car"]["percent"]
                    - targets[arm][scope]["car_percent"]
                ),
                "tsr_percent_delta": (
                    blocks[arm][scope]["tsr"]["percent"]
                    - targets[arm][scope]["tsr_percent"]
                ),
                "legacy_ets_steps_delta": (
                    blocks[arm][scope]["legacy_ets_steps"]["mean"]
                    - targets[arm][scope]["ets_steps"]
                ),
            }
            for key, expected_value in expected.items():
                _close(
                    record.get(key),
                    expected_value,
                    label=(
                        f"published_table1_differences/{arm}/{scope}/{key}"
                    ),
                )
    _validate_differences(summary.get("method_differences"), blocks=blocks)

    transitions = _mapping(
        summary.get("paired_transitions"), label="paired_transitions"
    )
    _exact_keys(
        transitions, {"overall", "suites"}, label="paired_transitions"
    )
    suite_transitions = _mapping(
        transitions.get("suites"), label="paired_transitions/suites"
    )
    _exact_keys(
        suite_transitions, set(SUITES), label="paired_transitions/suites"
    )
    overall_transitions = _validate_transitions(
        transitions.get("overall"),
        baseline=blocks[ARMS[0]]["average"],
        aegis=blocks[ARMS[1]]["average"],
        denominator=EXPECTED_CASES,
        label="paired_transitions/overall",
    )
    validated_suite_transitions = []
    for suite in SUITES:
        validated_suite_transitions.append(_validate_transitions(
            suite_transitions[suite],
            baseline=blocks[ARMS[0]][suite],
            aegis=blocks[ARMS[1]][suite],
            denominator=EXPECTED_SUITE_CASES,
            label=f"paired_transitions/suites/{suite}",
        ))
    for component, keys in (
        ("car", CAR_TRANSITIONS),
        ("task", TASK_TRANSITIONS),
        ("joint", JOINT_TRANSITIONS),
    ):
        for key in keys:
            if (
                sum(
                    suite[component][key]
                    for suite in validated_suite_transitions
                )
                != overall_transitions[component][key]
            ):
                raise TableReportError(
                    "suite paired transitions do not sum to overall "
                    f"for {component}/{key}"
                )
    return blocks


def validate_analysis_receipt(
    receipt: Mapping[str, Any],
    *,
    summary: Mapping[str, Any],
    summary_path: Path,
    summary_raw: bytes,
) -> tuple[str, str]:
    """Bind the summary bytes to the terminal analysis-v2 publication."""

    payload_sha256 = _sha(
        receipt.get("receipt_payload_sha256"),
        label="analysis receipt payload",
    )
    payload = dict(receipt)
    payload.pop("receipt_payload_sha256", None)
    if sha256_bytes(canonical_json_bytes(payload)) != payload_sha256:
        raise TableReportError("analysis receipt payload hash differs")
    if (
        receipt.get("schema_version") != ANALYSIS_RECEIPT_SCHEMA
        or receipt.get("status") != ANALYSIS_RECEIPT_STATUS
        or receipt.get("scientific_result") is not False
    ):
        raise TableReportError(
            "analysis receipt is not the terminal derived publication"
        )
    population = _mapping(
        receipt.get("population"), label="analysis receipt/population"
    )
    if (
        population.get("cases") != EXPECTED_CASES
        or population.get("results") != EXPECTED_RESULTS
        or population.get("no_cases_dropped") is not True
        or population.get("accepted_result_payloads_sha256")
        != summary.get("accepted_result_payloads_sha256")
    ):
        raise TableReportError(
            "analysis receipt population is incomplete or differently bound"
        )
    if receipt.get("source_v1") != summary.get("source_v1"):
        raise TableReportError(
            "analysis receipt source-v1 binding differs from summary"
        )
    if receipt.get("claim_scope") != summary.get("claim_scope"):
        raise TableReportError(
            "analysis receipt claim scope differs from summary"
        )
    artifacts = _mapping(
        receipt.get("artifacts"), label="analysis receipt/artifacts"
    )
    descriptor = _mapping(
        artifacts.get("summary"),
        label="analysis receipt/artifacts/summary",
    )
    _exact_keys(
        descriptor,
        {"path", "sha256", "bytes"},
        label="analysis receipt/artifacts/summary",
    )
    recorded_path = descriptor.get("path")
    if not isinstance(recorded_path, str) or not recorded_path:
        raise TableReportError(
            "analysis receipt summary path must be non-empty"
        )
    provenance_path = Path(recorded_path)
    expected_provenance_path = (
        ANALYSIS_PUBLICATION_ROOT
        / f"{EXPECTED_RUN_ID}-publisher-{ANALYSIS_PUBLISHER_JOB_ID}"
        / "population-summary-v2.json"
    )
    if (
        not provenance_path.is_absolute()
        or ".." in provenance_path.parts
        or recorded_path != str(expected_provenance_path)
    ):
        raise TableReportError(
            "analysis receipt summary path has unsafe/unexpected topology"
        )
    if summary_path.name != "population-summary-v2.json":
        raise TableReportError(
            "supplied relocated summary basename differs from publication"
        )
    if (
        _sha(
            descriptor.get("sha256"),
            label="analysis receipt summary SHA",
        )
        != sha256_bytes(summary_raw)
        or _integer(
            descriptor.get("bytes"),
            label="analysis receipt summary bytes",
        )
        != len(summary_raw)
    ):
        raise TableReportError(
            "analysis receipt summary bytes/SHA differ from supplied summary"
        )
    return payload_sha256, recorded_path


def _decimal_text(value: Decimal, *, signed: bool = False) -> str:
    rounded = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{rounded:+.2f}" if signed else f"{rounded:.2f}"


def _paper_value(
    metric_key: str, target: Mapping[str, Decimal]
) -> str:
    key = {
        "car": "car_percent",
        "tsr": "tsr_percent",
        "legacy_ets_steps": "ets_steps",
    }[metric_key]
    suffix = "%" if metric_key in {"car", "tsr"} else ""
    return f"{_decimal_text(target[key])}{suffix}"


def _reproduced_value(
    metric_key: str, block: Mapping[str, Any], *, latex: bool = False
) -> str:
    record = block[metric_key]
    if metric_key in {"car", "tsr"}:
        percent = _decimal_text(record["percent"])
        evidence = f"{record['success_count']}/{record['denominator']}"
        return (
            rf"{percent}\%\,({evidence})"
            if latex
            else f"{percent}% ({evidence})"
        )
    return (
        rf"{_decimal_text(record['mean'])}\,({record['sum']}/{record['denominator']})"
        if latex
        else (
            f"{_decimal_text(record['mean'])} "
            f"({record['sum']}/{record['denominator']})"
        )
    )


def _target_delta(
    metric_key: str,
    block: Mapping[str, Any],
    target: Mapping[str, Decimal],
) -> Decimal:
    if metric_key == "car":
        return block["car"]["percent"] - target["car_percent"]
    if metric_key == "tsr":
        return block["tsr"]["percent"] - target["tsr_percent"]
    return block["legacy_ets_steps"]["mean"] - target["ets_steps"]


def render_markdown(
    *,
    blocks: Mapping[str, Mapping[str, Mapping[str, Any]]],
    targets: Mapping[str, Mapping[str, Mapping[str, Decimal]]],
    summary_sha256: str,
    analysis_receipt_sha256: str,
    summary_provenance_path: str,
    local_summary_path: str,
    target_sha256: str,
) -> str:
    lines = [
        "# SafeLIBERO translational Table-1 comparison",
        "",
        (
            "Complete paired population: **1,600 cases / 3,200 results**, "
            "with every scientific method failure retained and no apparatus "
            "status in the validated summary."
        ),
        "",
        (
            f"Input summary SHA-256: `{summary_sha256}`  \n"
            f"Analysis-v2 receipt SHA-256: `{analysis_receipt_sha256}`  \n"
            f"Receipt-recorded summary path: `{summary_provenance_path}`  \n"
            f"Validated local mirror path: `{local_summary_path}`  \n"
            f"Frozen paper-target config SHA-256: `{target_sha256}`"
        ),
        "",
    ]
    for metric_label, metric_key, _ in METRIC_SPECS:
        lines.extend(
            [
                f"## {metric_label}",
                "",
                (
                    "| Method | Row | Spatial | Goal | Object | Long | "
                    "Average |"
                ),
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for arm in ARMS:
            paper = [
                _paper_value(metric_key, targets[arm][scope])
                for scope in SCOPES
            ]
            reproduced = [
                _reproduced_value(metric_key, blocks[arm][scope])
                for scope in SCOPES
            ]
            unit = " pp" if metric_key in {"car", "tsr"} else ""
            deltas = [
                f"{_decimal_text(_target_delta(metric_key, blocks[arm][scope], targets[arm][scope]), signed=True)}{unit}"
                for scope in SCOPES
            ]
            method = METHOD_LABELS_MD[arm]
            lines.append(
                f"| {method} | Paper | " + " | ".join(paper) + " |"
            )
            lines.append(
                f"| {method} | Reproduced (exact) | "
                + " | ".join(reproduced)
                + " |"
            )
            lines.append(
                f"| {method} | Δ reproduced−paper | "
                + " | ".join(deltas)
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## Direct paired method effect",
            "",
            (
                "Δ below is **π0.5+AEGIS − π0.5** on the same frozen "
                "population."
            ),
            "",
            "| Metric | Spatial | Goal | Object | Long | Average |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric_label, metric_key, _ in METRIC_SPECS:
        values: list[str] = []
        for scope in SCOPES:
            baseline = blocks[ARMS[0]][scope][metric_key]
            aegis = blocks[ARMS[1]][scope][metric_key]
            field = "percent" if metric_key in {"car", "tsr"} else "mean"
            unit = " pp" if metric_key in {"car", "tsr"} else " steps"
            values.append(
                f"{_decimal_text(aegis[field] - baseline[field], signed=True)}{unit}"
            )
        lines.append(f"| {metric_label} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Joint CAR-defined safety and task outcome",
            "",
            (
                "| Method | Safe + success | Safe + task failure | "
                "Collision + success | Collision + task failure | Denominator |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in ARMS:
        block = blocks[arm]["average"]
        cells = []
        for outcome in JOINT_OUTCOMES:
            count = block["joint_outcome_counts"][outcome]
            percent = Decimal(100 * count) / Decimal(block["episodes"])
            cells.append(f"{count} ({_decimal_text(percent)}%)")
        lines.append(
            f"| {METHOD_LABELS_MD[arm]} | "
            + " | ".join(cells)
            + f" | {block['episodes']} |"
        )

    lines.extend(
        [
            "",
            "## Scope and caveats",
            "",
            (
                "- CAR is the released active-obstacle displacement test "
                "(maximum L1 displacement strictly greater than 1 mm means "
                "collision). It is **not** a direct contact or clearance metric."
            ),
            (
                "- “Safe” in the joint table means CAR-defined safe. Sampled "
                "MuJoCo contacts are a separate diagnostic and cannot be "
                "reconstructed from this aggregate table."
            ),
            (
                "- Continuous minimum clearance/SDF is unavailable; this "
                "report makes no minimum-clearance claim."
            ),
            (
                "- The AEGIS arm uses frozen per-case Codex obstacle labels. "
                "The paper's GLM-4.5V semantic selector was not reproduced, "
                "so this is not an exact end-to-end paper reproduction."
            ),
            (
                "- Paper values are printed targets, so exact paper "
                "numerators are unavailable. In particular, the printed "
                "Long-suite AEGIS CAR of 79.63% is not representable as an "
                "integer numerator over 400 cases."
            ),
            (
                "- The Average column is the arithmetic mean over the four "
                "equal-size suites; reproduced counts also aggregate exactly "
                "over 1,600 cases."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_latex(
    *,
    blocks: Mapping[str, Mapping[str, Mapping[str, Any]]],
    targets: Mapping[str, Mapping[str, Mapping[str, Decimal]]],
) -> str:
    lines = [
        "% Auto-generated only from a validated 1,600-case / 3,200-result summary.",
        "% Requires: \\usepackage{booktabs,graphicx}",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{SafeLIBERO translational Table-1 targets and complete-population reproduction. Reproduced CAR/TSR cells show percentage (successes/denominator); ETS cells show mean (sum/denominator).}",
        "\\label{tab:safelibero_table1_reproduction}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lllrrrrr}",
        "\\toprule",
        "Metric & Method & Row & Spatial & Goal & Object & Long & Average \\\\",
        "\\midrule",
    ]
    first_metric = True
    for metric_label, metric_key, _ in METRIC_SPECS:
        if not first_metric:
            lines.append("\\midrule")
        first_metric = False
        for arm in ARMS:
            paper = [
                _paper_value(metric_key, targets[arm][scope]).replace(
                    "%", r"\%"
                )
                for scope in SCOPES
            ]
            reproduced = [
                _reproduced_value(
                    metric_key, blocks[arm][scope], latex=True
                )
                for scope in SCOPES
            ]
            deltas = [
                (
                    "$"
                    + _decimal_text(
                        _target_delta(
                            metric_key,
                            blocks[arm][scope],
                            targets[arm][scope],
                        ),
                        signed=True,
                    )
                    + r"\,\mathrm{pp}$"
                    if metric_key in {"car", "tsr"}
                    else _decimal_text(
                        _target_delta(
                            metric_key,
                            blocks[arm][scope],
                            targets[arm][scope],
                        ),
                        signed=True,
                    )
                )
                for scope in SCOPES
            ]
            method = METHOD_LABELS_TEX[arm]
            lines.append(
                " & ".join([metric_label, method, "Paper", *paper]) + r" \\"
            )
            lines.append(
                " & ".join(
                    [metric_label, method, "Reproduced", *reproduced]
                )
                + r" \\"
            )
            lines.append(
                " & ".join(
                    [
                        metric_label,
                        method,
                        r"$\Delta$ (reproduced$-$paper)",
                        *deltas,
                    ]
                )
                + r" \\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
            "",
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            "\\caption{Direct paired method effect and joint CAR-defined safety/task outcomes.}",
            "\\label{tab:safelibero_table1_joint}",
            "\\begin{tabular}{lrrrrr}",
            "\\toprule",
            r"Metric & Spatial & Goal & Object & Long & Average \\",
            "\\midrule",
        ]
    )
    for metric_label, metric_key, _ in METRIC_SPECS:
        values = []
        for scope in SCOPES:
            baseline = blocks[ARMS[0]][scope][metric_key]
            aegis = blocks[ARMS[1]][scope][metric_key]
            field = "percent" if metric_key in {"car", "tsr"} else "mean"
            difference = _decimal_text(
                aegis[field] - baseline[field], signed=True
            )
            values.append(
                f"${difference}" + r"\,\mathrm{pp}$"
                if metric_key in {"car", "tsr"}
                else difference
            )
        lines.append(
            " & ".join([f"{metric_label} (AEGIS$-$baseline)", *values])
            + r" \\"
        )
    lines.extend(
        [
            "\\midrule",
            (
                r"Method & Safe+success & Safe+failure & "
                r"Collision+success & Collision+failure & Denominator \\"
            ),
            "\\midrule",
        ]
    )
    for arm in ARMS:
        block = blocks[arm]["average"]
        cells = []
        for outcome in JOINT_OUTCOMES:
            count = block["joint_outcome_counts"][outcome]
            percent = Decimal(100 * count) / Decimal(block["episodes"])
            cells.append(
                rf"{count}\,({_decimal_text(percent)}\%)"
            )
        lines.append(
            " & ".join(
                [METHOD_LABELS_TEX[arm], *cells, str(block["episodes"])]
            )
            + r" \\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
            "\\begin{quote}",
            "\\footnotesize",
            (
                "\\textbf{Scope and caveats.} CAR is the released active-"
                "obstacle displacement test (strictly greater than 1~mm), "
                "not direct contact or clearance. ``Safe'' in the joint table "
                "means CAR-defined safe; sampled MuJoCo contacts are separate "
                "diagnostics. Continuous minimum clearance/SDF is unavailable. "
                "The AEGIS arm is conditioned on frozen per-case Codex labels; "
                "the paper's GLM-4.5V selector was not reproduced. Paper "
                "numerators are unavailable, and the printed Long-suite AEGIS "
                "CAR of 79.63\\% is not representable over 400 cases."
            ),
            "\\end{quote}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_atomic(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, delete=False
    ) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": sha256_path(path),
        "bytes": path.stat().st_size,
    }


def build_report(
    *,
    summary_path: Path,
    analysis_receipt_path: Path,
    paper_targets_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists() or output_root.is_symlink():
        raise TableReportError("output root must be unused")
    summary, summary_raw = load_json(summary_path, label="population summary")
    analysis_receipt, analysis_receipt_raw = load_json(
        analysis_receipt_path, label="analysis-v2 receipt"
    )
    targets_config, targets_raw = load_json(
        paper_targets_path, label="paper-target config"
    )
    summary_sha256 = sha256_bytes(summary_raw)
    analysis_receipt_sha256 = sha256_bytes(analysis_receipt_raw)
    targets_sha256 = sha256_bytes(targets_raw)
    targets = validate_targets(targets_config, raw_sha256=targets_sha256)
    blocks = validate_summary(summary, targets=targets)
    (
        analysis_receipt_payload_sha256,
        summary_provenance_path,
    ) = validate_analysis_receipt(
        analysis_receipt,
        summary=summary,
        summary_path=summary_path,
        summary_raw=summary_raw,
    )
    local_summary_path = str(summary_path.resolve())
    markdown = render_markdown(
        blocks=blocks,
        targets=targets,
        summary_sha256=summary_sha256,
        analysis_receipt_sha256=analysis_receipt_sha256,
        summary_provenance_path=summary_provenance_path,
        local_summary_path=local_summary_path,
        target_sha256=targets_sha256,
    ).encode("utf-8")
    latex = render_latex(blocks=blocks, targets=targets).encode("utf-8")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    markdown_path = output_root / "table1-comparison.md"
    latex_path = output_root / "table1-comparison.tex"
    receipt_path = output_root / "table1-report-receipt.json"
    _write_atomic(markdown_path, markdown)
    _write_atomic(latex_path, latex)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "rendered_from_complete_validated_population",
        "scientific_result": False,
        "inputs": {
            "population_summary_sha256": summary_sha256,
            "analysis_v2_receipt_sha256": analysis_receipt_sha256,
            "analysis_v2_receipt_payload_sha256": (
                analysis_receipt_payload_sha256
            ),
            "analysis_v2_summary_provenance_path": (
                summary_provenance_path
            ),
            "local_population_summary_path": local_summary_path,
            "paper_target_config_sha256": targets_sha256,
            "accepted_result_payloads_sha256": summary[
                "accepted_result_payloads_sha256"
            ],
            "source_v1_run_id": summary["source_v1"]["v1_run_id"],
            "source_v1_git_commit": summary["source_v1"][
                "v1_source_git_commit"
            ],
        },
        "validation": {
            "cases": EXPECTED_CASES,
            "results": EXPECTED_RESULTS,
            "task_level_groups": EXPECTED_GROUPS,
            "apparatus_failure_count": 0,
            "no_results_dropped": True,
            "paper_targets_hash_frozen": True,
        },
        "artifacts": {
            "markdown": _artifact(markdown_path),
            "latex": _artifact(latex_path),
        },
        "claim_scope": {
            "methods": list(ARMS),
            "semantic_selector_reproduced": False,
            "minimum_clearance_available": False,
            "car_is_direct_contact": False,
        },
    }
    receipt["receipt_payload_sha256"] = sha256_bytes(
        canonical_json_bytes(receipt)
    )
    _write_atomic(
        receipt_path,
        (
            json.dumps(
                receipt,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render Markdown and Overleaf-ready LaTeX from an exact complete "
            "SafeLIBERO post-publication summary"
        )
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--analysis-receipt", type=Path, required=True)
    parser.add_argument("--paper-targets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = build_report(
        summary_path=args.summary,
        analysis_receipt_path=args.analysis_receipt,
        paper_targets_path=args.paper_targets,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_payload_sha256": receipt[
                    "receipt_payload_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
