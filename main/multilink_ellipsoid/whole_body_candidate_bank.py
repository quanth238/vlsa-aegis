"""Deterministic coverage-preserving reduction of the local-frame grid."""

from __future__ import annotations

import itertools
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from main.multilink_ellipsoid.whole_body_support_audit import audit_cases


GRID_PATTERN = re.compile(
    r"^grid_(m1|z0|p1)_(m1|z0|p1)_(m1|z0|p1)_front_loaded_r2\.0$"
)
TOKEN_VALUE = {"m1": -1, "z0": 0, "p1": 1}
VALUE_TOKEN = {-1: "m1", 0: "z0", 1: "p1"}


def load_bank_config(path: Any) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get("schema_version") != "vlsa_distal_whole_body_candidate_bank.v1":
        raise ValueError("whole-body candidate-bank config schema differs")
    if value.get("protocol_id") != "vlsa-distal-whole-body-candidate-bank-v1":
        raise ValueError("whole-body candidate-bank protocol differs")
    if value.get("development_split") != "train":
        raise ValueError("whole-body candidate-bank development split differs")
    if value.get("diagnostic_only_splits") != ["validation", "test"]:
        raise ValueError("whole-body candidate-bank diagnostic splits differ")
    if int(value.get("source_candidate_count", 0)) != 27:
        raise ValueError("whole-body candidate-bank source size differs")
    if int(value.get("selected_candidate_count", 0)) != 13:
        raise ValueError("whole-body candidate-bank selected size differs")
    if value.get("pair_rule") != "nominal_plus_six_exact_opposite_direction_pairs":
        raise ValueError("whole-body candidate-bank pair rule differs")
    if value.get("validation_test_outcomes_influence_selection") is not False:
        raise ValueError("holdout outcomes may not influence candidate-bank selection")
    output = json.loads(json.dumps(value, sort_keys=True))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    return output


def candidate_vector(name: str) -> tuple[int, int, int]:
    match = GRID_PATTERN.match(name)
    if match is None:
        raise ValueError("local-frame grid candidate name differs")
    return tuple(TOKEN_VALUE[token] for token in match.groups())


def candidate_name(vector: Sequence[int]) -> str:
    return "grid_{}_{}_{}_front_loaded_r2.0".format(
        *(VALUE_TOKEN[int(value)] for value in vector)
    )


def opposite_name(name: str) -> str:
    return candidate_name([-value for value in candidate_vector(name)])


def _case_view(case: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    selected = set(names)
    value = dict(case)
    exact = dict(case["exact_case"])
    exact["candidates"] = [
        row for row in case["exact_case"]["candidates"] if row["name"] in selected
    ]
    if len(exact["candidates"]) != len(names):
        raise ValueError("candidate bank is not present in every case")
    value["exact_case"] = exact
    return value


def _support_signature(case_summary: Mapping[str, Any]) -> dict[str, bool]:
    output = {}
    for group, value in case_summary["per_group"].items():
        output["group/{}/safe".format(group)] = value["safe_candidate_count"] > 0
        output["group/{}/unsafe".format(group)] = value["unsafe_candidate_count"] > 0
        output["group/{}/two_sided".format(group)] = bool(value["two_sided_support"])
    for row, value in case_summary["per_row"].items():
        output["row/{}/safe".format(row)] = value["safe_candidate_count"] > 0
        output["row/{}/unsafe".format(row)] = value["unsafe_candidate_count"] > 0
        output["row/{}/two_sided".format(row)] = bool(value["two_sided_support"])
    global_support = case_summary["global_support"]
    for kind in ("represented", "physical"):
        output["global/{}/safe".format(kind)] = (
            global_support["{}_safe_candidate_count".format(kind)] > 0
        )
        output["global/{}/unsafe".format(kind)] = (
            global_support["{}_unsafe_candidate_count".format(kind)] > 0
        )
        output["global/{}/two_sided".format(kind)] = bool(
            global_support["{}_two_sided_support".format(kind)]
        )
    return output


def _coverage_requirements(
    full_summary: Mapping[str, Any],
) -> dict[str, dict[str, bool]]:
    requirements = {}
    for case in full_summary["per_case"]:
        signature = _support_signature(case)
        requirements[case["case_id"]] = {
            key: value for key, value in signature.items() if value
        }
    return requirements


def _preserves(
    subset_summary: Mapping[str, Any],
    requirements: Mapping[str, Mapping[str, bool]],
) -> bool:
    by_id = {row["case_id"]: row for row in subset_summary["per_case"]}
    for case_id, required in requirements.items():
        observed = _support_signature(by_id[case_id])
        if any(not observed.get(key, False) for key in required):
            return False
    return True


def _direction_determinant(pair_representatives: Sequence[str]) -> float:
    matrix = [[0.0] * 3 for _ in range(3)]
    for name in pair_representatives:
        vector = [float(value) for value in candidate_vector(name)]
        norm = math.sqrt(sum(value * value for value in vector))
        unit = [value / norm for value in vector]
        for row in range(3):
            for column in range(3):
                matrix[row][column] += unit[row] * unit[column]
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _score(
    cases: Sequence[Mapping[str, Any]], summary: Mapping[str, Any],
    pair_representatives: Sequence[str], names: Sequence[str],
) -> tuple[Any, ...]:
    known = int(summary["known_candidate_count"])
    near = sum(
        int(value["near_boundary_candidate_count"])
        for case in summary["per_case"]
        for value in list(case["per_group"].values()) + list(case["per_row"].values())
        if bool(value["two_sided_support"])
    )
    selected = set(names)
    effective_ratio_sum = 0.0
    effective_ratio_count = 0
    for case in cases:
        for candidate in case["exact_case"]["candidates"]:
            if candidate["name"] not in selected or candidate["name"] == "nominal":
                continue
            if not bool(candidate["exact_group_target"]["known_outcome"]):
                continue
            requested = float(candidate["requested_alpha"])
            if requested <= 0.0:
                continue
            effective_ratio_sum += float(
                candidate["source_effective_post_AEGIS_correction_l2_action"]
            ) / requested
            effective_ratio_count += 1
    effective_ratio = (
        effective_ratio_sum / effective_ratio_count if effective_ratio_count else 0.0
    )
    return (
        known,
        near,
        round(_direction_determinant(pair_representatives), 12),
        round(effective_ratio, 12),
        tuple(sorted(names)),
    )


def select_bank(
    cases: Sequence[Mapping[str, Any]], audit_config: Mapping[str, Any],
    bank_size: int = 13, development_split: str = "train",
) -> dict[str, Any]:
    development = [
        case for case in cases if case["selection"]["split"] == development_split
    ]
    diagnostics = [
        case for case in cases if case["selection"]["split"] != development_split
    ]
    if not development:
        raise ValueError("candidate-bank development split is empty")
    names = [row["name"] for row in development[0]["exact_case"]["candidates"]]
    if names[0] != "nominal" or len(names) != 27:
        raise ValueError("candidate-bank source grid differs")
    if any(
        [row["name"] for row in case["exact_case"]["candidates"]] != names
        for case in cases
    ):
        raise ValueError("candidate ordering differs across cases")
    if bank_size != 13:
        raise ValueError("candidate-bank target size differs")

    non_nominal = names[1:]
    pairs = []
    seen = set()
    for name in non_nominal:
        if name in seen:
            continue
        opposite = opposite_name(name)
        if opposite not in non_nominal:
            raise ValueError("candidate opposite direction is missing")
        pair = tuple(sorted((name, opposite)))
        pairs.append(pair)
        seen.update(pair)
    if len(pairs) != 13:
        raise ValueError("candidate opposite-pair count differs")

    full_development_summary = audit_cases(development, audit_config)
    requirements = _coverage_requirements(full_development_summary)
    feasible = []
    for chosen_pairs in itertools.combinations(pairs, 6):
        selected_names = ["nominal"] + [
            name for pair in chosen_pairs for name in pair
        ]
        subset_summary = audit_cases(
            [_case_view(case, selected_names) for case in development], audit_config,
        )
        if not _preserves(subset_summary, requirements):
            continue
        representatives = [pair[0] for pair in chosen_pairs]
        feasible.append((
            _score(development, subset_summary, representatives, selected_names),
            selected_names,
            subset_summary,
        ))
    if not feasible:
        return {
            "selection_status": "no_coverage_preserving_13_candidate_bank",
            "full_development_summary": full_development_summary,
            "coverage_requirements": requirements,
            "feasible_bank_count": 0,
            "selected_candidate_names": [],
        }
    feasible.sort(key=lambda item: item[0], reverse=True)
    score, selected_names, selected_development_summary = feasible[0]
    diagnostic_full = audit_cases(diagnostics, audit_config)
    diagnostic_selected = audit_cases(
        [_case_view(case, selected_names) for case in diagnostics], audit_config,
    )
    return {
        "selection_status": "coverage_preserving_13_candidate_bank_selected",
        "development_split_only": True,
        "selection_objective_order": [
            "known_outcome_count", "near_boundary_count_on_supported_constraints",
            "direction_covariance_determinant", "mean_effective_requested_ratio",
            "deterministic_lexicographic_tie_break",
        ],
        "candidate_pair_rule": "nominal_plus_six_exact_opposite_direction_pairs",
        "feasible_bank_count": len(feasible),
        "selected_score": {
            "known_outcome_count": score[0],
            "near_boundary_count_on_supported_constraints": score[1],
            "direction_covariance_determinant": score[2],
            "mean_effective_requested_ratio": score[3],
        },
        "selected_candidate_names": selected_names,
        "coverage_requirements": requirements,
        "full_development_summary": full_development_summary,
        "selected_development_summary": selected_development_summary,
        "diagnostic_full_holdout_summary": diagnostic_full,
        "diagnostic_selected_holdout_summary": diagnostic_selected,
    }
