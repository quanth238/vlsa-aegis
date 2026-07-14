"""Portable identity checks for released SafeLIBERO saved-state rows.

The official-state manifest path deliberately binds portable names and bytes,
never an installation-specific absolute path.  This module stays free of
NumPy and PyTorch imports so manifest validation remains part of the local,
dependency-free gate.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


def canonical_init_state_relative_path(value: str) -> str:
    """Return one canonical, root-relative POSIX path or fail closed."""

    if not isinstance(value, str) or not value:
        raise ValueError("init_state_relative_path must be a non-empty string")
    if "\\" in value:
        raise ValueError("init_state_relative_path must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("init_state_relative_path must stay below the init-state root")
    canonical = path.as_posix()
    if canonical != value:
        raise ValueError("init_state_relative_path must be canonical POSIX text")
    return canonical


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def init_state_row_identity(row: Any) -> dict[str, Any]:
    """Describe one C-contiguous array row without importing an array library."""

    dtype = getattr(row, "dtype", None)
    dtype_text = getattr(dtype, "str", None)
    if not isinstance(dtype_text, str) or not dtype_text:
        raise ValueError("init-state row must expose a canonical dtype.str")
    shape_value = getattr(row, "shape", None)
    if shape_value is None:
        raise ValueError("init-state row must expose shape")
    try:
        shape = [int(value) for value in shape_value]
    except (TypeError, ValueError) as error:
        raise ValueError("init-state row shape must contain integers") from error
    if not shape or any(value < 0 for value in shape):
        raise ValueError("init-state row shape must be non-empty and non-negative")
    tobytes = getattr(row, "tobytes", None)
    if not callable(tobytes):
        raise ValueError("init-state row must expose tobytes")
    try:
        payload = tobytes(order="C")
    except TypeError:
        payload = tobytes()
    if not isinstance(payload, bytes):
        raise ValueError("init-state row tobytes must return bytes")
    return {
        "init_state_row_dtype": dtype_text,
        "init_state_row_shape": shape,
        "init_state_row_bytes_sha256": hashlib.sha256(payload).hexdigest(),
    }


def verify_official_state_bindings(
    case: Mapping[str, Any],
    *,
    actual_task_name: str,
    init_states_root: str | Path,
    init_state_path: str | Path,
    row: Any,
) -> list[str]:
    """Recompute every schema-3 official-state binding from runtime inputs."""

    errors: list[str] = []
    if case.get("schema_version") != "3.0":
        return ["official-state verification requires schema_version 3.0"]
    if case.get("task_name") != actual_task_name:
        errors.append("task_name differs from the selected SafeLIBERO task")

    root = Path(init_states_root).resolve()
    path = Path(init_state_path).resolve()
    try:
        relative_path = path.relative_to(root).as_posix()
        relative_path = canonical_init_state_relative_path(relative_path)
    except (OSError, ValueError) as error:
        errors.append(f"init-state path is outside or invalid below the configured root: {error}")
        relative_path = None
    if relative_path is not None and case.get("init_state_relative_path") != relative_path:
        errors.append("init_state_relative_path differs from the selected SafeLIBERO file")

    try:
        actual_file_sha256 = file_sha256(path)
    except OSError as error:
        errors.append(f"cannot hash selected init-state file: {error}")
    else:
        if case.get("init_state_file_sha256") != actual_file_sha256:
            errors.append("init_state_file_sha256 differs from selected file bytes")

    try:
        actual_row = init_state_row_identity(row)
    except ValueError as error:
        errors.append(f"cannot identify selected init-state row: {error}")
    else:
        for key, actual in actual_row.items():
            if case.get(key) != actual:
                errors.append(f"{key} differs from selected row")
    return errors
