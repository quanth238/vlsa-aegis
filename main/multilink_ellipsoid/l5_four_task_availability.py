"""Contracts for the immutable four-task/two-level L5 availability audit."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_l5_four_task_availability_audit.v1"
RESULT_SCHEMA = "vlsa_distal_l5_four_task_availability_audit_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_l5_four_task_availability_audit_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("result_payload_sha256", None)
    payload.pop("validation_payload_sha256", None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "protocol_id", "claim_scope", "source",
        "population", "clean_L5_eligibility", "fold_gate",
        "next_if_infeasible", "forbidden",
    }:
        raise ValueError("four-task availability config keys differ")
    population = value["population"]
    gate = value["fold_gate"]
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"]
        != "vlsa-distal-l5-four-task-availability-audit-v1"
        or population["suite"] != "safelibero_goal"
        or population["logical_tasks"] != [0, 1, 2, 3]
        or population["safety_levels"] != ["I", "II"]
        or population["expected_case_count"] != 400
        or gate["minimum_eligible_test_cases_per_level"] != 1
        or gate["minimum_distinct_eligible_training_tasks"] != 2
    ):
        raise ValueError("four-task availability protocol differs")
    if repo_root is not None:
        manifest = Path(repo_root) / value["source"]["population_manifest"]
        if file_sha256(manifest) != value["source"][
            "population_manifest_file_sha256"
        ]:
            raise ValueError("four-task population manifest differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def load_population(
    path: Path, config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [
        row for row in rows
        if row["suite"] == config["population"]["suite"]
        and int(row["logical_task_index"])
        in config["population"]["logical_tasks"]
        and row["safety_level"] in config["population"]["safety_levels"]
    ]
    if len(selected) != int(config["population"]["expected_case_count"]):
        raise ValueError("four-task population count differs")
    identities = [str(row["case_id"]) for row in selected]
    if len(identities) != len(set(identities)):
        raise ValueError("four-task population identities repeat")
    expected = int(config["population"]["episodes_per_task_level"])
    counts = Counter(
        (int(row["logical_task_index"]), str(row["safety_level"]))
        for row in selected
    )
    if any(
        counts[(task, level)] != expected
        for task in config["population"]["logical_tasks"]
        for level in config["population"]["safety_levels"]
    ):
        raise ValueError("four-task task-level counts differ")
    return sorted(selected, key=lambda row: int(row["case_ordinal"]))


def archived_result_path(
    table1_root: Path, row: Mapping[str, Any],
) -> Path:
    task_ordinal = int(row["case_ordinal"]) // 50
    return (
        Path(table1_root) / "tasks" / f"task-{task_ordinal}" / "results"
        / "aegis" / str(row["case_id"]) / "result.json"
    )


def summarize_records(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, Any]:
    tasks = [int(item) for item in config["population"]["logical_tasks"]]
    levels = [str(item) for item in config["population"]["safety_levels"]]
    by_task_level = []
    for task in tasks:
        for level in levels:
            subset = [
                record for record in records
                if int(record["logical_task_index"]) == task
                and str(record["safety_level"]) == level
            ]
            counts = Counter(str(record["classification"]) for record in subset)
            eligible = [record for record in subset if bool(record["eligible"])]
            by_task_level.append({
                "logical_task_index": task,
                "safety_level": level,
                "case_count": len(subset),
                "classification_counts": dict(sorted(counts.items())),
                "eligible_clean_L5_case_count": len(eligible),
                "eligible_clean_L5_case_ids": [
                    str(record["case_id"]) for record in eligible
                ],
            })
    gate = config["fold_gate"]
    folds = []
    for held_out in tasks:
        test = [
            record for record in records
            if int(record["logical_task_index"]) == held_out
            and bool(record["eligible"])
        ]
        train = [
            record for record in records
            if int(record["logical_task_index"]) != held_out
            and bool(record["eligible"])
        ]
        test_by_level = {
            level: sum(str(record["safety_level"]) == level for record in test)
            for level in levels
        }
        train_tasks = sorted({
            int(record["logical_task_index"]) for record in train
        })
        feasible = bool(
            all(
                count >= int(gate["minimum_eligible_test_cases_per_level"])
                for count in test_by_level.values()
            )
            and len(train_tasks)
            >= int(gate["minimum_distinct_eligible_training_tasks"])
        )
        folds.append({
            "held_out_logical_task_index": held_out,
            "held_out_levels": levels,
            "eligible_test_case_count_by_level": test_by_level,
            "eligible_test_case_count": len(test),
            "eligible_training_case_count": len(train),
            "distinct_eligible_training_tasks": train_tasks,
            "both_held_out_levels_excluded_from_training": True,
            "fold_feasible": feasible,
        })
    all_feasible = all(item["fold_feasible"] for item in folds)
    return {
        "case_count": len(records),
        "classification_counts": dict(sorted(Counter(
            str(record["classification"]) for record in records
        ).items())),
        "eligible_clean_L5_case_count": sum(
            bool(record["eligible"]) for record in records
        ),
        "task_level_coverage": by_task_level,
        "folds": folds,
        "all_four_task_folds_feasible": all_feasible,
        "next_action": (
            "freeze_four_folds_then_collect_two_sided_boundaries"
            if all_feasible else config["next_if_infeasible"]
        ),
    }
