"""Immutable artifact helpers for opt-in Poisson feasibility runs."""

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


RESULT_HASH_FIELD = "result_payload_sha256"


class ArtifactContractError(RuntimeError):
    """Raised when a final scientific artifact is absent or inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactContractError("%s must be a lowercase SHA-256" % label)
    return value


def attach_payload_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str = RESULT_HASH_FIELD,
) -> Dict[str, Any]:
    if hash_field in payload:
        raise ArtifactContractError("payload already contains %s" % hash_field)
    output = dict(payload)
    output[hash_field] = sha256_bytes(canonical_json_bytes(output))
    return output


def verify_payload_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str = RESULT_HASH_FIELD,
) -> str:
    expected = require_sha256(payload.get(hash_field), hash_field)
    unhashed = {key: value for key, value in payload.items() if key != hash_field}
    observed = sha256_bytes(canonical_json_bytes(unhashed))
    if observed != expected:
        raise ArtifactContractError("%s does not match payload" % hash_field)
    return observed


def load_hashed_json(
    path: Path,
    *,
    hash_field: str = RESULT_HASH_FIELD,
) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactContractError("final JSON is missing or symlinked: %s" % path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactContractError("final JSON is invalid: %s" % path) from error
    if not isinstance(value, dict):
        raise ArtifactContractError("final JSON must contain one object")
    verify_payload_hash(value, hash_field=hash_field)
    return value


def _encoded_pretty_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def publish_hashed_json(
    path: Path,
    payload_without_hash: Mapping[str, Any],
    *,
    hash_field: str = RESULT_HASH_FIELD,
) -> str:
    """Fsync and atomically publish a final, never-overwritten JSON object.

    An already-valid byte-identical artifact is an idempotent no-op.  Any
    other existing path is a conflict, including a valid artifact from a
    different attempt.
    """

    final_payload = attach_payload_hash(payload_without_hash, hash_field=hash_field)
    encoded = _encoded_pretty_json(final_payload)
    if path.exists() or path.is_symlink():
        existing = load_hashed_json(path, hash_field=hash_field)
        if _encoded_pretty_json(existing) == encoded and path.read_bytes() == encoded:
            return "identical_existing"
        raise ArtifactContractError("refusing to overwrite immutable final: %s" % path)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ArtifactContractError("final parent must be an existing real directory")

    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=".%s." % path.name,
            suffix=".partial",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        staged = json.loads(temporary.read_text(encoding="utf-8"))
        if not isinstance(staged, dict):
            raise ArtifactContractError("staged final is not one JSON object")
        verify_payload_hash(staged, hash_field=hash_field)
        try:
            os.link(str(temporary), str(path))
        except FileExistsError as error:
            raise ArtifactContractError(
                "final was concurrently created: %s" % path
            ) from error
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return "published"


def validate_artifact_reference(root: Path, reference: Mapping[str, Any]) -> Path:
    relative = reference.get("relative_path")
    if not isinstance(relative, str) or not relative:
        raise ArtifactContractError("artifact relative_path is invalid")
    relative_path = Path(relative)
    parts = relative_path.parts
    if relative_path.is_absolute() or not parts or any(
        part in ("", ".", "..") for part in parts
    ):
        raise ArtifactContractError("artifact relative_path is invalid")
    if root.is_symlink() or not root.is_dir():
        raise ArtifactContractError("artifact root must be an existing real directory")

    # Traverse from an already-open root and refuse symbolic links at every
    # component.  Resolving first is insufficient: ``Path.resolve`` erases the
    # evidence that a reference escaped through an in-root symlink.  Hash the
    # same no-follow file descriptor that was checked so the validation cannot
    # be redirected between ``is_symlink`` and ``open``.
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    opened = []
    artifact_fd = None
    try:
        current_fd = os.open(str(root), directory_flags | nofollow)
        opened.append(current_fd)
        for component in parts[:-1]:
            current_fd = os.open(
                component,
                directory_flags | nofollow,
                dir_fd=current_fd,
            )
            opened.append(current_fd)
        artifact_fd = os.open(parts[-1], os.O_RDONLY | nofollow, dir_fd=current_fd)
        metadata = os.fstat(artifact_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactContractError("referenced artifact is not a regular file")
    except (OSError, ValueError, ArtifactContractError) as error:
        if artifact_fd is not None:
            os.close(artifact_fd)
        for fd in reversed(opened):
            os.close(fd)
        raise ArtifactContractError(
            "referenced artifact is missing, invalid, or symlinked"
        ) from error

    expected_size = reference.get("bytes")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
        for fd in reversed(opened):
            os.close(fd)
        if artifact_fd is not None:
            os.close(artifact_fd)
        raise ArtifactContractError("artifact byte count is invalid")
    try:
        if metadata.st_size != expected_size:
            raise ArtifactContractError("artifact byte count differs")
        expected_hash = require_sha256(reference.get("sha256"), "artifact sha256")
        digest = hashlib.sha256()
        with os.fdopen(os.dup(artifact_fd), "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ArtifactContractError("artifact SHA-256 differs")
    finally:
        if artifact_fd is not None:
            os.close(artifact_fd)
        for fd in reversed(opened):
            os.close(fd)
    return root.absolute().joinpath(*parts)


def validate_resume_identity(
    result: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    identity_fields: Iterable[str],
) -> None:
    def nested_value(value: Mapping[str, Any], field: str) -> Any:
        current: Any = value
        for component in field.split("."):
            if not isinstance(current, Mapping) or component not in current:
                raise ArtifactContractError(
                    "resume identity is missing %s" % field
                )
            current = current[component]
        return current

    for field in identity_fields:
        if nested_value(result, field) != nested_value(expected, field):
            raise ArtifactContractError("resume identity differs at %s" % field)


def final_is_resumable(
    final_path: Path,
    expected_identity: Mapping[str, Any],
    *,
    identity_fields: Iterable[str],
    artifact_root: Optional[Path] = None,
) -> Tuple[bool, Optional[str]]:
    """Return true only for a fully valid final with exact provenance."""

    try:
        result = load_hashed_json(final_path)
        # Import lazily to avoid a module cycle: result_schema itself uses the
        # hashing helpers above.  A matching self-hash is necessary but not
        # sufficient for resume; the final must satisfy the complete episode
        # schema and scientific invariants as well.
        from main.poisson_fullbody.result_schema import validate_episode_result

        validate_episode_result(result)
        validate_resume_identity(
            result, expected_identity, identity_fields=identity_fields
        )
        references = result.get("artifact_references")
        if not isinstance(references, list):
            raise ArtifactContractError("artifact_references must be a list")
        if references and artifact_root is None:
            raise ArtifactContractError(
                "artifact_root is required to verify referenced artifacts"
            )
        if artifact_root is not None:
            for reference in references:
                if not isinstance(reference, dict):
                    raise ArtifactContractError("artifact reference must be an object")
                validate_artifact_reference(artifact_root, reference)
    except ArtifactContractError as error:
        return False, str(error)
    return True, None
