#!/usr/bin/env python3
"""Small, dependency-free helpers for immutable AEGIS run receipts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


SHA256_LENGTH = 64


class ReceiptError(RuntimeError):
    """Raised when immutable infrastructure evidence is absent or inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReceiptError(f"{label} must be a lowercase SHA-256")
    return value


def load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReceiptError(f"{label} is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ReceiptError(f"{label} must contain one JSON object")
    return value


def verify_payload_sha256(
    value: Mapping[str, Any],
    *,
    field: str,
    label: str,
) -> str:
    expected = require_sha256(value.get(field), label=f"{label}/{field}")
    payload = {key: item for key, item in value.items() if key != field}
    observed = sha256_bytes(canonical_json_bytes(payload))
    if observed != expected:
        raise ReceiptError(f"{label} payload SHA-256 does not match")
    return observed


def load_tsv_contract(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file():
        raise ReceiptError(f"run contract is missing or symlinked: {path}")
    output: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0] or fields[0] in output:
            raise ReceiptError(
                f"run contract line {line_number} is malformed or duplicated"
            )
        output[fields[0]] = fields[1]
    return output


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish JSON and never replace an existing artifact."""

    if path.exists() or path.is_symlink():
        raise ReceiptError(f"immutable receipt already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ReceiptError(
            f"receipt parent must be an existing real directory: {path.parent}"
        )
    payload = json.dumps(
        dict(value),
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ReceiptError(
                f"immutable receipt was concurrently created: {path}"
            ) from error
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
