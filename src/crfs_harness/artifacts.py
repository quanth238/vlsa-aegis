"""Atomic experiment artifacts and dependency-free schema checks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CASE_RESULT_REQUIRED = {
    "schema_version",
    "case_id",
    "run_id",
    "status",
    "config_hash",
    "provenance",
    "repair",
    "trials",
}
TRIAL_NAMES = {"nominal", "direct_repair", "random_residual", "oracle_residual", "bridge_edit"}
FINAL_STATUSES = {"completed", "infeasible", "failed"}
RUNTIME_TRANSPORT_FIELDS = {"host", "port", "output_root", "run_id"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def scientific_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove allocation/transport routing from a scientific config identity."""
    return {
        key: item
        for key, item in value.items()
        if key not in RUNTIME_TRANSPORT_FIELDS
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_case_result(value: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = CASE_RESULT_REQUIRED - set(value)
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")
    status = value.get("status")
    if status not in FINAL_STATUSES:
        errors.append(f"invalid final status: {status!r}")
    for key in ("case_id", "run_id", "config_hash"):
        if not isinstance(value.get(key), str) or not value.get(key):
            errors.append(f"{key} must be a non-empty string")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("provenance must be an object")
    trials = value.get("trials")
    if not isinstance(trials, Mapping):
        errors.append("trials must be an object")
    elif status == "completed":
        absent_trials = TRIAL_NAMES - set(trials)
        if absent_trials:
            errors.append(f"completed result missing paired trials: {sorted(absent_trials)}")
        for name, trial in trials.items():
            errors.extend(_validate_trial(name, trial))
    repair = value.get("repair")
    if not isinstance(repair, Mapping):
        errors.append("repair must be an object")
    elif status in {"completed", "infeasible"} and not isinstance(repair.get("feasible"), bool):
        errors.append("repair.feasible must be boolean")
    return errors


def _validate_trial(name: str, value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"trial {name!r} must be an object"]
    errors = []
    for key in ("clearance_m", "endpoint_error_m"):
        if not isinstance(value.get(key), (int, float)):
            errors.append(f"trial {name!r}.{key} must be numeric")
    if not isinstance(value.get("safe"), bool):
        errors.append(f"trial {name!r}.safe must be boolean")
    return errors


def valid_completion(path: str | Path) -> bool:
    try:
        value = load_json(path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return (
        isinstance(value, Mapping)
        and value.get("status") in {"completed", "infeasible"}
        and not validate_case_result(value)
    )


def validate_jsonl_unique(path: str | Path, key: str) -> tuple[list[dict[str, Any]], list[str]]:
    values: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[Any] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"line {line_number}: invalid JSON: {error}")
                continue
            if not isinstance(value, dict):
                errors.append(f"line {line_number}: expected an object")
                continue
            identity = value.get(key)
            if not identity:
                errors.append(f"line {line_number}: missing {key}")
            elif identity in seen:
                errors.append(f"line {line_number}: duplicate {key}={identity!r}")
            else:
                seen.add(identity)
            values.append(value)
    return values, errors
