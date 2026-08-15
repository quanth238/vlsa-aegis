"""Realized post-AEGIS monotonicity contracts for scalar candidate risk."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_empirical_risk_monotonicity.v1"
RESULT_SCHEMA = "vlsa_distal_empirical_risk_monotonicity_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_empirical_risk_monotonicity_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "dataset_config",
        "dataset_config_file_sha256", "risk_field", "positive_is_unsafe",
        "realized_coordinate", "alignment", "monotonicity", "gate",
        "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("empirical risk monotonicity config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("empirical risk monotonicity config schema differs")
    if value["protocol_id"] != "vlsa-distal-empirical-risk-monotonicity-v1":
        raise ValueError("empirical risk monotonicity protocol differs")
    if value["risk_field"] != "compiled_box_risk":
        raise ValueError("empirical risk monotonicity target differs")
    if value["positive_is_unsafe"] is not True:
        raise ValueError("empirical risk monotonicity sign differs")
    expected_alignment = {
        "minimum_cosine": 0.99,
        "maximum_off_axis_fraction": 0.05,
        "minimum_effective_alpha_increment": 1e-09,
        "exclude_source_clipping": True,
    }
    if value["alignment"] != expected_alignment:
        raise ValueError("empirical risk monotonicity alignment differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = sha256(raw)
    output["config_payload_sha256"] = sha256(canonical(value))
    return output


def _flatten_actions(actions: Sequence[Sequence[float]]) -> list[float]:
    if len(actions) != 5 or any(len(row) != 7 for row in actions):
        raise ValueError("empirical risk monotonicity action shape differs")
    values = [float(item) for row in actions for item in row]
    if any(not math.isfinite(item) for item in values):
        raise ValueError("empirical risk monotonicity action is nonfinite")
    return values


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _ideal_direction(source_candidate: Mapping[str, Any]) -> list[float]:
    direction = [float(item) for item in source_candidate["direction"]]
    profile = [float(item) for item in source_candidate["temporal_profile"]]
    if len(direction) != 3 or len(profile) != 5:
        raise ValueError("empirical risk monotonicity registered profile differs")
    ideal = []
    for weight in profile:
        ideal.extend([weight * item for item in direction])
        ideal.extend([0.0, 0.0, 0.0, 0.0])
    norm = _norm(ideal)
    if abs(norm - 1.0) > 1.0e-9:
        raise ValueError("empirical risk monotonicity profile is not unit norm")
    return ideal


def case_report(
    result: Mapping[str, Any], source_curve: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    empirical = result["empirical_case"]
    source_by_name = {str(row["name"]): row for row in source_curve["candidates"]}
    candidates = empirical["candidates"]
    baseline = next(
        row for row in candidates if float(row["requested_alpha"]) == 0.0
    )
    baseline_actions = _flatten_actions(baseline["source_executed_actions"])
    records = []
    for candidate in candidates:
        name = str(candidate["name"])
        source = source_by_name[name]
        requested = float(candidate["requested_alpha"])
        actions = _flatten_actions(candidate["source_executed_actions"])
        delta = [item - base for item, base in zip(actions, baseline_actions)]
        delta_norm = _norm(delta)
        known = candidate["source_terminal_status"] != "UNKNOWN_TIMEOUT"
        clipped = bool(source.get("clipped", False))
        if requested == 0.0:
            effective_alpha = 0.0
            cosine = 1.0
            off_axis = 0.0
            off_axis_fraction = 0.0
            direction_preserved = True
        else:
            ideal = _ideal_direction(source)
            effective_alpha = _dot(delta, ideal)
            residual = [
                item - effective_alpha * basis
                for item, basis in zip(delta, ideal)
            ]
            off_axis = _norm(residual)
            cosine = (
                -1.0 if delta_norm == 0.0
                else _dot(delta, ideal) / delta_norm
            )
            off_axis_fraction = (
                math.inf if delta_norm == 0.0 else off_axis / delta_norm
            )
            direction_preserved = bool(
                cosine >= float(config["alignment"]["minimum_cosine"])
                and off_axis_fraction
                <= float(config["alignment"]["maximum_off_axis_fraction"])
                and effective_alpha > 0.0
            )
        eligible = bool(
            known and direction_preserved
            and not (
                bool(config["alignment"]["exclude_source_clipping"])
                and clipped
            )
        )
        records.append({
            "name": name,
            "requested_alpha": requested,
            "effective_outward_alpha": float(effective_alpha),
            "realized_correction_l2": float(delta_norm),
            "alignment_cosine": float(cosine),
            "off_axis_l2": float(off_axis),
            "off_axis_fraction": float(off_axis_fraction),
            "source_clipped": clipped,
            "known_non_timeout": known,
            "direction_preserved": direction_preserved,
            "eligible_for_monotonicity": eligible,
            "risk": float(candidate[config["risk_field"]]),
        })
    eligible = sorted(
        (row for row in records if row["eligible_for_monotonicity"]),
        key=lambda row: (row["effective_outward_alpha"], row["requested_alpha"]),
    )
    pairs = []
    tolerance = float(config["monotonicity"]["risk_tolerance"])
    minimum_increment = float(
        config["alignment"]["minimum_effective_alpha_increment"]
    )
    for left, right in zip(eligible, eligible[1:]):
        increment = float(
            right["effective_outward_alpha"] - left["effective_outward_alpha"]
        )
        if increment < minimum_increment:
            continue
        risk_delta = float(right["risk"] - left["risk"])
        pairs.append({
            "left": left["name"],
            "right": right["name"],
            "effective_alpha_increment": increment,
            "risk_delta": risk_delta,
            "monotonicity_violation": risk_delta > tolerance,
        })
    return {
        "case_id": result["case_id"],
        "split": result["split"],
        "state_step": int(result["state_step"]),
        "candidate_records": records,
        "eligible_pair_count": len(pairs),
        "monotonicity_violation_count": sum(
            bool(row["monotonicity_violation"]) for row in pairs
        ),
        "pairs": pairs,
    }


def classify(
    reports: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
    *, dataset_strict_gate_pass: bool,
) -> dict[str, Any]:
    splits = {}
    for split in ("train", "validation"):
        rows = [row for row in reports if row["split"] == split]
        pair_count = sum(int(row["eligible_pair_count"]) for row in rows)
        violations = sum(
            int(row["monotonicity_violation_count"]) for row in rows
        )
        splits[split] = {
            "state_count": len(rows),
            "states_with_eligible_pair": sum(
                int(row["eligible_pair_count"]) > 0 for row in rows
            ),
            "eligible_pair_count": pair_count,
            "monotonicity_violation_count": violations,
            "monotonic_pair_fraction": (
                None if pair_count == 0 else (pair_count - violations) / pair_count
            ),
            "unknown_timeout_candidate_count": sum(
                not bool(candidate["known_non_timeout"])
                for row in rows for candidate in row["candidate_records"]
            ),
            "direction_or_clipping_excluded_candidate_count": sum(
                bool(candidate["known_non_timeout"])
                and not bool(candidate["eligible_for_monotonicity"])
                for row in rows for candidate in row["candidate_records"]
            ),
        }
    gate = config["gate"]
    gates = {
        "dataset_strict_gate": bool(dataset_strict_gate_pass),
        "train_state_support": splits["train"]["states_with_eligible_pair"]
        >= int(gate["minimum_train_states_with_eligible_pair"]),
        "validation_state_support": splits["validation"]["states_with_eligible_pair"]
        >= int(gate["minimum_validation_states_with_eligible_pair"]),
        "train_pair_support": splits["train"]["eligible_pair_count"]
        >= int(gate["minimum_train_eligible_pairs"]),
        "validation_pair_support": splits["validation"]["eligible_pair_count"]
        >= int(gate["minimum_validation_eligible_pairs"]),
        "zero_train_monotonicity_violations": (
            not gate["require_zero_train_monotonicity_violations"]
            or splits["train"]["monotonicity_violation_count"] == 0
        ),
        "zero_validation_monotonicity_violations": (
            not gate["require_zero_validation_monotonicity_violations"]
            or splits["validation"]["monotonicity_violation_count"] == 0
        ),
    }
    return {
        "splits": splits,
        "gates": gates,
        "monotonic_regularization_authorized": bool(all(gates.values())),
    }
