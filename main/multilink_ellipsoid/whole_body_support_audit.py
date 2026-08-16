"""Read-only coverage audit for whole-body future-risk artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


AUDIT_CONFIG_SCHEMA = "vlsa_distal_whole_body_support_audit.v1"
AUDIT_SCHEMA = "vlsa_distal_whole_body_support_audit_result.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def load_audit_config(path: Any) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get("schema_version") != AUDIT_CONFIG_SCHEMA:
        raise ValueError("whole-body support audit schema differs")
    if value.get("protocol_id") != "vlsa-distal-whole-body-support-audit-v1":
        raise ValueError("whole-body support audit protocol differs")
    if value.get("group_order") != [
        "end_effector", "palm", "L5", "L6", "L7",
    ]:
        raise ValueError("whole-body support group order differs")
    if value.get("physical_group_order") != ["palm", "L5", "L6", "L7"]:
        raise ValueError("whole-body physical group order differs")
    if value.get("robot_rows") != {
        "end_effector": [0], "palm": [1], "L5": [2, 3, 4],
        "L6": [5, 6], "L7": [7, 8],
    }:
        raise ValueError("whole-body support row binding differs")
    if value.get("two_sided_rule") != "strict_Q_negative_and_positive_known_outcomes":
        raise ValueError("whole-body support two-sided rule differs")
    if value.get("timeout_rule") != "UNKNOWN_excluded_from_safe_unsafe_support":
        raise ValueError("whole-body support timeout rule differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def _blank_coverage() -> dict[str, Any]:
    return {
        "known_candidate_count": 0,
        "safe_candidate_count": 0,
        "unsafe_candidate_count": 0,
        "near_boundary_candidate_count": 0,
        "active_witness_candidate_count": 0,
        "two_sided_state_count": 0,
        "two_sided_state_ids": [],
        "state_count": 0,
        "episode_count": 0,
    }


def _strict_two_sided(values: Sequence[float]) -> bool:
    return any(value < 0.0 for value in values) and any(
        value > 0.0 for value in values
    )


def _support_classification(*, safe: int, unsafe: int) -> str:
    if safe > 0 and unsafe > 0:
        return "two_sided"
    if safe > 0:
        return "safe_only"
    if unsafe > 0:
        return "unsafe_only"
    return "no_known_outcome"


def _minimum_row_slacks(candidate: Mapping[str, Any], row_count: int) -> list[float]:
    trace = candidate["exact_group_target"]["trace"]
    if not trace:
        raise ValueError("known candidate has no exact trace")
    output = [float("inf")] * row_count
    for sample in trace:
        values = sample["row_normalized_radial_slack"]
        if len(values) != row_count:
            raise ValueError("whole-body trace row count differs")
        output = [min(left, float(right)) for left, right in zip(output, values)]
    return output


def _raw_contact_pair_records(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for event in candidate.get("raw_protected_contacts", []):
        protected = str(event.get("protected_geom_name", ""))
        obstacle = str(event.get("obstacle_geom_name", ""))
        if not protected or not obstacle:
            raise ValueError("whole-body raw contact lacks compiled geom names")
        output.append({
            "candidate_name": str(candidate["name"]),
            "pair": "%s ↔ %s" % (protected, obstacle),
            "protected_geom_name": protected,
            "obstacle_geom_name": obstacle,
            "phase": event.get("phase"),
            "action_offset": event.get("action_offset"),
            "substep": event.get("substep"),
            "distance_m": float(event["distance_m"]),
        })
    return output


def audit_cases(
    cases: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, Any]:
    groups = list(config["group_order"])
    physical_groups = list(config["physical_group_order"])
    robot_rows = {key: list(value) for key, value in config["robot_rows"].items()}
    row_count = sum(len(robot_rows[group]) for group in groups)
    row_to_group = {
        int(row): group for group in groups for row in robot_rows[group]
    }
    near = float(config["near_boundary_abs_slack"])
    car_threshold = float(config["paper_car_threshold_m"])
    consistency_tolerance = float(config["risk_trace_consistency_tolerance"])

    per_case = []
    for case in cases:
        exact_case = case["exact_case"]
        initial = exact_case["exact_group_target"]
        initial_slack = {
            group: float(initial["initial_group_normalized_radial_slack"][group])
            for group in groups
        }
        initial_contact = {
            group: int(initial["initial_group_contact_sample_count"][group])
            for group in groups
        }
        known = [
            candidate for candidate in exact_case["candidates"]
            if bool(candidate["exact_group_target"]["known_outcome"])
        ]
        group_values = {group: [] for group in groups}
        row_values = {row: [] for row in range(row_count)}
        group_counts = {group: _blank_coverage() for group in groups}
        row_counts = {str(row): _blank_coverage() for row in range(row_count)}
        represented_global_safe = 0
        represented_global_unsafe = 0
        physical_global_safe = 0
        physical_global_unsafe = 0
        active_group_counts = {group: 0 for group in groups}
        active_row_counts = {str(row): 0 for row in range(row_count)}
        raw_contact_pair_records = []

        for candidate in known:
            target = candidate["exact_group_target"]
            group_q = {
                group: float(target["group_future_violation"][group])
                for group in groups
            }
            group_contact = {
                group: int(target["group_contact_sample_count"][group])
                for group in groups
            }
            row_slack = _minimum_row_slacks(candidate, row_count)
            row_q = [-value for value in row_slack]
            for group in groups:
                trace_q = -min(row_slack[row] for row in robot_rows[group])
                if abs(trace_q - group_q[group]) > consistency_tolerance:
                    raise ValueError("stored group risk differs from exact row trace")
                group_values[group].append(group_q[group])
                counts = group_counts[group]
                counts["known_candidate_count"] += 1
                safe = group_q[group] <= 0.0 and group_contact[group] == 0
                counts["safe_candidate_count"] += int(safe)
                counts["unsafe_candidate_count"] += int(not safe)
                counts["near_boundary_candidate_count"] += int(
                    abs(group_q[group]) <= near
                )
            for row, value in enumerate(row_q):
                row_values[row].append(value)
                counts = row_counts[str(row)]
                counts["known_candidate_count"] += 1
                counts["safe_candidate_count"] += int(value <= 0.0)
                counts["unsafe_candidate_count"] += int(value > 0.0)
                counts["near_boundary_candidate_count"] += int(abs(value) <= near)

            active_group = max(groups, key=lambda group: group_q[group])
            active_row = max(range(row_count), key=lambda row: row_q[row])
            active_group_counts[active_group] += 1
            active_row_counts[str(active_row)] += 1

            represented_safe = all(
                group_q[group] <= 0.0 and group_contact[group] == 0
                for group in groups
            )
            represented_unsafe = any(
                group_q[group] > 0.0 or group_contact[group] > 0
                for group in groups
            )
            physical_safe = bool(
                all(
                    group_q[group] <= 0.0 and group_contact[group] == 0
                    for group in physical_groups
                )
                and int(candidate["raw_protected_contact_sample_count"]) == 0
                and float(candidate["replayed_maximum_CAR_m"]) <= car_threshold
                and not bool(candidate["replayed_physical_veto"])
            )
            physical_unsafe = bool(
                any(
                    group_q[group] > 0.0 or group_contact[group] > 0
                    for group in physical_groups
                )
                or int(candidate["raw_protected_contact_sample_count"]) > 0
                or float(candidate["replayed_maximum_CAR_m"]) > car_threshold
                or bool(candidate["replayed_physical_veto"])
            )
            represented_global_safe += int(represented_safe)
            represented_global_unsafe += int(represented_unsafe)
            physical_global_safe += int(physical_safe)
            physical_global_unsafe += int(physical_unsafe)
            raw_contact_pair_records.extend(_raw_contact_pair_records(candidate))

        for group in groups:
            group_counts[group]["active_witness_candidate_count"] = (
                active_group_counts[group]
            )
            group_counts[group]["two_sided_support"] = _strict_two_sided(
                group_values[group]
            )
            group_counts[group]["initially_safe"] = bool(
                initial_slack[group] > 0.0 and initial_contact[group] == 0
            )
            group_counts[group]["support_classification"] = (
                _support_classification(
                    safe=int(group_counts[group]["safe_candidate_count"]),
                    unsafe=int(group_counts[group]["unsafe_candidate_count"]),
                )
            )
            group_counts[group]["timeout_limited"] = bool(
                len(exact_case["candidates"]) > len(known)
            )
        for row in range(row_count):
            row_counts[str(row)]["active_witness_candidate_count"] = (
                active_row_counts[str(row)]
            )
            row_counts[str(row)]["constraint_group"] = row_to_group[row]
            row_counts[str(row)]["two_sided_support"] = _strict_two_sided(
                row_values[row]
            )

        per_case.append({
            "case_id": str(case["case_id"]),
            "episode_group_id": str(case["selection"]["episode_group_id"]),
            "split": str(case["selection"]["split"]),
            "target_group": str(case["selection"]["target_group"]),
            "candidate_count": len(exact_case["candidates"]),
            "known_candidate_count": len(known),
            "unknown_timeout_count": len(exact_case["candidates"]) - len(known),
            "initial_group_slack": initial_slack,
            "initial_group_contact_count": initial_contact,
            "initially_safe_across_physical_groups": all(
                initial_slack[group] > 0.0 and initial_contact[group] == 0
                for group in physical_groups
            ),
            "per_group": group_counts,
            "per_row": row_counts,
            "constraint_report": {
                group: {
                    "initially_safe": bool(
                        group_counts[group]["initially_safe"]
                    ),
                    "safe_candidate_count": int(
                        group_counts[group]["safe_candidate_count"]
                    ),
                    "unsafe_candidate_count": int(
                        group_counts[group]["unsafe_candidate_count"]
                    ),
                    "unknown_candidate_count": (
                        len(exact_case["candidates"]) - len(known)
                    ),
                    "two_sided_support": bool(
                        group_counts[group]["two_sided_support"]
                    ),
                    "support_classification": str(
                        group_counts[group]["support_classification"]
                    ),
                    "timeout_limited": bool(
                        group_counts[group]["timeout_limited"]
                    ),
                }
                for group in groups
            },
            "raw_contact_pair_sample_count": len(raw_contact_pair_records),
            "raw_contact_pairs": sorted({
                str(item["pair"]) for item in raw_contact_pair_records
            }),
            "raw_contact_pair_records": raw_contact_pair_records,
            "global_support": {
                "represented_safe_candidate_count": represented_global_safe,
                "represented_unsafe_candidate_count": represented_global_unsafe,
                "represented_two_sided_support": bool(
                    represented_global_safe and represented_global_unsafe
                ),
                "physical_safe_candidate_count": physical_global_safe,
                "physical_unsafe_candidate_count": physical_global_unsafe,
                "physical_two_sided_support": bool(
                    physical_global_safe and physical_global_unsafe
                ),
            },
        })

    split_summary = {}
    for split in ("train", "validation", "test"):
        split_cases = [row for row in per_case if row["split"] == split]
        groups_summary = {group: _blank_coverage() for group in groups}
        rows_summary = {str(row): _blank_coverage() for row in range(row_count)}
        for group in groups:
            target = groups_summary[group]
            target["state_count"] = len(split_cases)
            target["episode_count"] = len({
                row["episode_group_id"] for row in split_cases
            })
            for row in split_cases:
                source = row["per_group"][group]
                for key in (
                    "known_candidate_count", "safe_candidate_count",
                    "unsafe_candidate_count", "near_boundary_candidate_count",
                    "active_witness_candidate_count",
                ):
                    target[key] += int(source[key])
                if source["two_sided_support"]:
                    target["two_sided_state_ids"].append(row["case_id"])
            target["two_sided_state_count"] = len(target["two_sided_state_ids"])
            episode_rows: dict[str, list[dict[str, Any]]] = {}
            for case_row in split_cases:
                episode_rows.setdefault(
                    str(case_row["episode_group_id"]), []
                ).append(case_row)
            episode_classes = {
                "initially_safe": [],
                "initially_unsafe": [],
                "safe_only": [],
                "unsafe_only": [],
                "two_sided": [],
                "mixed_across_one_sided_states": [],
                "no_known_outcome": [],
                "timeout_limited": [],
            }
            for episode_id, state_rows in episode_rows.items():
                prevention = [
                    row for row in state_rows
                    if bool(row["per_group"][group]["initially_safe"])
                ]
                if prevention:
                    episode_classes["initially_safe"].append(episode_id)
                else:
                    episode_classes["initially_unsafe"].append(episode_id)
                if any(int(row["unknown_timeout_count"]) > 0 for row in state_rows):
                    episode_classes["timeout_limited"].append(episode_id)
                if not prevention:
                    continue
                state_classes = {
                    str(row["per_group"][group]["support_classification"])
                    for row in prevention
                }
                if "two_sided" in state_classes:
                    episode_classes["two_sided"].append(episode_id)
                elif "safe_only" in state_classes and "unsafe_only" in state_classes:
                    episode_classes["mixed_across_one_sided_states"].append(
                        episode_id
                    )
                elif "safe_only" in state_classes:
                    episode_classes["safe_only"].append(episode_id)
                elif "unsafe_only" in state_classes:
                    episode_classes["unsafe_only"].append(episode_id)
                else:
                    episode_classes["no_known_outcome"].append(episode_id)
            target["episode_group_classification"] = {
                "%s_episode_count" % key: len(value)
                for key, value in episode_classes.items()
            }
            target["episode_group_ids_by_classification"] = episode_classes
        for row_index in range(row_count):
            target = rows_summary[str(row_index)]
            target["constraint_group"] = row_to_group[row_index]
            target["state_count"] = len(split_cases)
            target["episode_count"] = len({
                row["episode_group_id"] for row in split_cases
            })
            for row in split_cases:
                source = row["per_row"][str(row_index)]
                for key in (
                    "known_candidate_count", "safe_candidate_count",
                    "unsafe_candidate_count", "near_boundary_candidate_count",
                    "active_witness_candidate_count",
                ):
                    target[key] += int(source[key])
                if source["two_sided_support"]:
                    target["two_sided_state_ids"].append(row["case_id"])
            target["two_sided_state_count"] = len(target["two_sided_state_ids"])
        split_summary[split] = {
            "case_count": len(split_cases),
            "unknown_timeout_count": sum(
                int(row["unknown_timeout_count"]) for row in split_cases
            ),
            "per_group": groups_summary,
            "per_row": rows_summary,
            "represented_global_safe_support_state_count": sum(
                row["global_support"]["represented_safe_candidate_count"] > 0
                for row in split_cases
            ),
            "represented_global_two_sided_state_count": sum(
                row["global_support"]["represented_two_sided_support"]
                for row in split_cases
            ),
            "physical_global_safe_support_state_count": sum(
                row["global_support"]["physical_safe_candidate_count"] > 0
                for row in split_cases
            ),
            "physical_global_two_sided_state_count": sum(
                row["global_support"]["physical_two_sided_support"]
                for row in split_cases
            ),
        }

    return {
        "case_count": len(per_case),
        "candidate_count": sum(row["candidate_count"] for row in per_case),
        "known_candidate_count": sum(row["known_candidate_count"] for row in per_case),
        "unknown_timeout_count": sum(row["unknown_timeout_count"] for row in per_case),
        "per_case": per_case,
        "split_summary": split_summary,
    }
