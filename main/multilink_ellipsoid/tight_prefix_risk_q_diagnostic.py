"""Contracts for compact Q-only training on tight five-action prefix risk."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_tight_prefix_risk_q_diagnostic.v1"
RESULT_SCHEMA = "vlsa_tight_prefix_risk_q_diagnostic_result.v1"
VALIDATION_SCHEMA = "vlsa_tight_prefix_risk_q_diagnostic_validation.v1"
MODEL_ROWS = tuple(range(10))
PRIMARY_ROWS = tuple(range(8))
DIAGNOSTIC_ROWS = (8, 9)
INPUT_DIMENSION = 7


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    expected = {
        "schema_version", "protocol_id", "claim_scope", "source",
        "dataset", "feature", "group_rows", "primary_rows",
        "diagnostic_rows", "model", "evaluation", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("tight prefix Q config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-tight-prefix-risk-q-diagnostic-v1"
    ):
        raise ValueError("tight prefix Q protocol differs")
    dataset = value["dataset"]
    if (
        dataset.get("fit_splits") != ["train"]
        or dataset.get("evaluation_splits") != ["train", "validation", "test"]
        or dataset.get("exclude_initially_unsafe_from_fit") is not True
        or dataset.get("test_status") != "already_opened_diagnostic"
    ):
        raise ValueError("tight prefix Q dataset protocol differs")
    feature = value["feature"]
    if (
        feature.get("input_dimension") != INPUT_DIMENSION
        or feature.get("model_rows") != list(MODEL_ROWS)
        or feature.get("definition")
        != "[initial_exact_radial_slack,nominal_first_step_outward_projection,"
        "five_effective_step_outward_projections]"
        or float(feature.get("translation_scale_m_per_action_unit", -1.0))
        != 0.05
    ):
        raise ValueError("tight prefix Q feature differs")
    groups = value["group_rows"]
    if groups != {
        "tight_end_effector": [0, 1, 2, 3, 4],
        "palm": [0],
        "finger1_base": [1],
        "finger1_pad": [2],
        "finger2_base": [3],
        "finger2_pad": [4],
        "L5": [5, 6, 7],
        "L6_diagnostic": [8, 9],
    }:
        raise ValueError("tight prefix Q group rows differ")
    if (
        value["primary_rows"] != list(PRIMARY_ROWS)
        or value["diagnostic_rows"] != list(DIAGNOSTIC_ROWS)
    ):
        raise ValueError("tight prefix Q row authority differs")
    model = value["model"]
    if (
        model.get("hidden_widths") != [32, 32]
        or model.get("seed") != 20260814
        or float(model.get("weight_decay", -1.0)) != 0.0001
        or model.get("checkpoint") != "fixed_final_epoch_no_selection"
    ):
        raise ValueError("tight prefix Q model differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def scientific_view(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items()
        if key not in {"allocation", "result_payload_sha256", "source"}
    }


def row_future_risks(candidate: Mapping[str, Any], rows: Sequence[int]) -> list[float]:
    """Return Q=-min(h) for every requested row in the five-action trace."""
    requested = tuple(int(row) for row in rows)
    trace = candidate["exact_group_target"]["trace"]
    if not trace or len(set(requested)) != len(requested):
        raise ValueError("tight prefix Q trace differs")
    width = len(trace[0]["row_normalized_radial_slack"])
    if (
        any(row < 0 or row >= width for row in requested)
        or any(len(sample["row_normalized_radial_slack"]) != width for sample in trace)
    ):
        raise ValueError("tight prefix Q trace row differs")
    return [
        -min(
            float(sample["row_normalized_radial_slack"][row])
            for sample in trace
        )
        for row in requested
    ]
