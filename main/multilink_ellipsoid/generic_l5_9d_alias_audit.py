"""Contracts for the no-training 9D L5 offset and support audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_generic_l5_9d_alias_audit.v1"
RESULT_SCHEMA = "vlsa_distal_generic_l5_9d_alias_audit_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_generic_l5_9d_alias_audit_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "protocol_id", "claim_scope", "sources", "audit",
        "decision", "forbidden",
    }:
        raise ValueError("generic L5 9D alias-audit config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-distal-generic-l5-9d-alias-audit-v1"
        or value["audit"]["prevention_state_count"] != 4
        or value["audit"]["oracle_anchor"]
        != "exact_nominal_row_risk_plus_frozen_predicted_candidate_delta"
    ):
        raise ValueError("generic L5 9D alias-audit protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def oracle_anchor_predictions(
    samples: Sequence[Mapping[str, Any]],
    predictions: Sequence[Sequence[float]],
) -> list[list[float]]:
    import numpy as np

    target = np.asarray([sample["risk_rows"] for sample in samples], dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    nominal = [
        index for index, sample in enumerate(samples)
        if str(sample["candidate_name"]) == "nominal"
    ]
    if target.shape != predicted.shape or len(nominal) != 1:
        raise ValueError("oracle-anchor input differs")
    anchor = nominal[0]
    return (predicted + target[anchor] - predicted[anchor]).tolist()


def direct_l5_context_vector(physical_context: Mapping[str, Any]) -> list[float]:
    obstacle = [float(item) for item in physical_context["obstacle"]["center_m"]]
    rows = physical_context["geometry_rows"][:3]
    if len(rows) != 3 or any(row["body_name"] != "robot0_link5" for row in rows):
        raise ValueError("direct L5 geometry rows differ")
    output = []
    for row in rows:
        center = [float(item) for item in row["center_m"]]
        rotation = [float(item) for line in row["rotation"] for item in line]
        semiaxes = [float(item) for item in row["semiaxes_m"]]
        output.extend([center[index] - obstacle[index] for index in range(3)])
        output.extend(rotation)
        output.extend(semiaxes)
    if len(output) != 45 or not all(math.isfinite(item) for item in output):
        raise ValueError("direct L5 context vector differs")
    return output


def standardized_rows(values: Sequence[Sequence[float]]) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    scale = np.std(array, axis=0)
    scale = np.where(scale >= 1.0e-9, scale, 1.0)
    return (array - np.mean(array, axis=0)) / scale


def rms_distance(left: Any, right: Any) -> float:
    import numpy as np

    return float(np.sqrt(np.mean((np.asarray(left) - np.asarray(right)) ** 2)))


def correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    import numpy as np

    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])
