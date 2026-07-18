#!/usr/bin/env python3
"""Fetch exactly one hash-bound VLSA Table-1 episode video.

The postpublication bundle is local metadata.  The selected MP4 remains under
the immutable VinUni run root until this tool probes and copies that one file.
No wildcard, directory, recursive, overwrite, or deletion interface exists.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


ROW_SCHEMA = "vlsa_table1_video_transfer_row.v2"
RECEIPT_SCHEMA = "vlsa_table1_video_transfer_receipt.v2"
EXPECTED_MANIFEST_NAME = "video-transfer-manifest.jsonl"
EXPECTED_ROWS = 3200
EXPECTED_ARMS = (
    "pi05_translational",
    "pi05_plus_aegis_translational",
)
ARM_MODES = {
    "pi05_translational": "pi05",
    "pi05_plus_aegis_translational": "aegis",
}
ROW_FIELDS = {
    "schema_version",
    "case_ordinal",
    "task_index",
    "task_level_group_id",
    "case_id",
    "arm",
    "mode",
    "status",
    "result",
    "video",
    "publisher_v2_inventory_item",
}
RESULT_FIELDS = {"path", "bytes", "sha256", "result_payload_sha256"}
VIDEO_FIELDS = {
    "path",
    "bytes",
    "sha256",
    "frames",
    "fps",
    "complete_episode",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SAFE_REMOTE_ROOT_RE = re.compile(r"^/[A-Za-z0-9._/-]+$")
SAFE_HOST_RE = re.compile(r"^[A-Za-z0-9._-]+$")
RUN_ROOT_PREFIX = PurePosixPath(
    "/mnt/data/quanth/experiments/vlsa-aegis-table1"
)


class FetchError(RuntimeError):
    """Raised when metadata, transport, or publication is unsafe."""


@dataclass(frozen=True)
class RemoteFileInfo:
    bytes: int
    sha256: str
    regular: bool = True
    symlink: bool = False


@dataclass(frozen=True)
class SelectedVideo:
    case_id: str
    arm: str
    run_root: str
    relative_path: str
    bytes: int
    sha256: str

    @property
    def remote_path(self) -> str:
        return self.run_root.rstrip("/") + "/" + self.relative_path


class VideoTransport:
    """Minimal transport interface used by production and fake transports."""

    def probe(self, run_root: str, relative_path: str) -> RemoteFileInfo:
        raise NotImplementedError

    def fetch(
        self,
        run_root: str,
        relative_path: str,
        expected_bytes: int,
        expected_sha256: str,
        destination_fd: int,
    ) -> None:
        raise NotImplementedError


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


def _stable_read(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise FetchError("{} is missing, not a file, or symlinked: {}".format(label, path))
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as error:
        raise FetchError("{} cannot be opened safely: {}".format(label, path)) from error
    chunks = []
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise FetchError("{} is not a regular file".format(label))
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        getattr(before, "st_mtime_ns", int(before.st_mtime * 1e9)),
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        getattr(after, "st_mtime_ns", int(after.st_mtime * 1e9)),
    )
    try:
        path_stat = os.stat(str(path), follow_symlinks=False)
    except OSError as error:
        raise FetchError("{} disappeared while being read".format(label)) from error
    path_identity = (
        path_stat.st_dev,
        path_stat.st_ino,
        path_stat.st_size,
        getattr(path_stat, "st_mtime_ns", int(path_stat.st_mtime * 1e9)),
    )
    payload = b"".join(chunks)
    if (
        before_identity != after_identity
        or path_identity != after_identity
        or stat.S_ISLNK(path_stat.st_mode)
        or len(payload) != after.st_size
    ):
        raise FetchError("{} changed while being read".format(label))
    return payload


def stable_file_sha256_and_size(path: Path, label: str) -> Tuple[str, int]:
    payload = _stable_read(path, label)
    return sha256_bytes(payload), len(payload)


def stable_fd_sha256_and_size(descriptor: int, label: str) -> Tuple[str, int]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise FetchError("{} is not a regular file".format(label))
    digest = hashlib.sha256()
    observed_size = 0
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        observed_size += len(chunk)
    after = os.fstat(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        getattr(before, "st_mtime_ns", int(before.st_mtime * 1e9)),
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        getattr(after, "st_mtime_ns", int(after.st_mtime * 1e9)),
    )
    if before_identity != after_identity or observed_size != after.st_size:
        raise FetchError("{} changed while being read".format(label))
    return digest.hexdigest(), observed_size


def _require_path_matches_fd(path: Path, descriptor: int, label: str) -> None:
    try:
        path_stat = os.stat(str(path), follow_symlinks=False)
    except OSError as error:
        raise FetchError("{} pathname disappeared".format(label)) from error
    descriptor_stat = os.fstat(descriptor)
    if (
        stat.S_ISLNK(path_stat.st_mode)
        or not stat.S_ISREG(path_stat.st_mode)
        or (path_stat.st_dev, path_stat.st_ino)
        != (descriptor_stat.st_dev, descriptor_stat.st_ino)
    ):
        raise FetchError("{} pathname no longer names its open file".format(label))


def _require_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FetchError("{} must be an object".format(label))
    return value


def _require_exact_fields(
    value: Mapping[str, Any], expected: Iterable[str], label: str
) -> None:
    expected_set = set(expected)
    observed_set = set(value)
    if observed_set != expected_set:
        raise FetchError(
            "{} fields differ: missing={}, extra={}".format(
                label,
                sorted(expected_set - observed_set),
                sorted(observed_set - expected_set),
            )
        )


def _require_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise FetchError("{} must be a positive integer".format(label))
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FetchError("{} must be a nonnegative integer".format(label))
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise FetchError("{} must be a lowercase SHA-256".format(label))
    return value


def _require_safe_component(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or SAFE_COMPONENT_RE.fullmatch(value) is None
        or value in {".", ".."}
    ):
        raise FetchError("{} is not a safe path component".format(label))
    return value


def _require_safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise FetchError("{} must be a nonempty POSIX path".format(label))
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise FetchError("{} must be a normalized relative POSIX path".format(label))
    if any(
        component in {"", ".", ".."}
        or SAFE_COMPONENT_RE.fullmatch(component) is None
        for component in path.parts
    ):
        raise FetchError("{} contains an unsafe path component".format(label))
    return value


def _load_json_object(path: Path, label: str) -> Tuple[Dict[str, Any], bytes]:
    payload = _stable_read(path, label)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FetchError("{} is invalid JSON".format(label)) from error
    if not isinstance(value, dict):
        raise FetchError("{} must contain one JSON object".format(label))
    return value, payload


def _verify_receipt_payload(receipt: Mapping[str, Any]) -> None:
    expected = _require_sha256(
        receipt.get("receipt_payload_sha256"), "receipt payload SHA-256"
    )
    unsigned = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_payload_sha256"
    }
    observed = sha256_bytes(canonical_json_bytes(unsigned))
    if observed != expected:
        raise FetchError("receipt payload SHA-256 does not verify")


def _bound_run_root(receipt: Mapping[str, Any]) -> str:
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise FetchError("only {} is accepted".format(RECEIPT_SCHEMA))
    if receipt.get("status") != "validated":
        raise FetchError("video-transfer receipt is not validated")
    _verify_receipt_payload(receipt)
    source = _require_object(receipt.get("source"), "receipt/source")
    run_id = _require_safe_component(source.get("run_id"), "receipt/source/run_id")
    source_commit = source.get("source_git_commit")
    if not isinstance(source_commit, str) or GIT_COMMIT_RE.fullmatch(source_commit) is None:
        raise FetchError("receipt/source/source_git_commit is invalid")
    run_root_value = source.get("run_root")
    if (
        not isinstance(run_root_value, str)
        or SAFE_REMOTE_ROOT_RE.fullmatch(run_root_value) is None
    ):
        raise FetchError("receipt/source/run_root is not a safe absolute path")
    run_root = PurePosixPath(run_root_value)
    if run_root.as_posix() != run_root_value or ".." in run_root.parts:
        raise FetchError("receipt/source/run_root is not normalized")
    if run_root.parent != RUN_ROOT_PREFIX or run_root.name != run_id:
        raise FetchError("receipt/source/run_root is outside the registered experiment root")
    publication = _require_object(
        source.get("publication_receipt"), "receipt/source/publication_receipt"
    )
    publication_path_value = publication.get("path")
    if (
        not isinstance(publication_path_value, str)
        or SAFE_REMOTE_ROOT_RE.fullmatch(publication_path_value) is None
    ):
        raise FetchError("receipt/source/publication_receipt/path is unsafe")
    publication_path = PurePosixPath(publication_path_value)
    if (
        publication_path.as_posix() != publication_path_value
        or ".." in publication_path.parts
        or publication_path == run_root
        or tuple(publication_path.parts[: len(run_root.parts)]) != run_root.parts
    ):
        raise FetchError("publication receipt is not bound under the immutable run root")
    _require_sha256(
        publication.get("sha256"), "receipt/source/publication_receipt/sha256"
    )
    _require_sha256(
        publication.get("receipt_payload_sha256"),
        "receipt/source/publication_receipt/receipt_payload_sha256",
    )
    return run_root_value


def _load_and_verify_manifest(
    manifest_path: Path, receipt: Mapping[str, Any]
) -> Sequence[Mapping[str, Any]]:
    manifest_payload = _stable_read(manifest_path, "video-transfer manifest")
    transfer = _require_object(
        receipt.get("transfer_manifest"), "receipt/transfer_manifest"
    )
    if transfer.get("path") != EXPECTED_MANIFEST_NAME:
        raise FetchError("receipt binds an unexpected transfer-manifest path")
    if manifest_path.name != EXPECTED_MANIFEST_NAME:
        raise FetchError("local manifest filename differs from the receipt")
    expected_rows = _require_positive_int(
        transfer.get("rows"), "receipt/transfer_manifest/rows"
    )
    if expected_rows != EXPECTED_ROWS:
        raise FetchError("transfer manifest must contain exactly 3,200 rows")
    expected_sha = _require_sha256(
        transfer.get("sha256"), "receipt/transfer_manifest/sha256"
    )
    if sha256_bytes(manifest_payload) != expected_sha:
        raise FetchError("transfer-manifest file SHA-256 does not verify")

    rows = []
    for line_number, raw_line in enumerate(manifest_payload.splitlines(), 1):
        if not raw_line.strip():
            raise FetchError("manifest contains a blank row at line {}".format(line_number))
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FetchError(
                "manifest line {} is invalid JSON".format(line_number)
            ) from error
        if not isinstance(row, dict):
            raise FetchError("manifest line {} is not an object".format(line_number))
        rows.append(row)
    if len(rows) != expected_rows:
        raise FetchError(
            "manifest row count differs: observed={}, expected={}".format(
                len(rows), expected_rows
            )
        )
    expected_canonical_rows = _require_sha256(
        transfer.get("canonical_rows_sha256"),
        "receipt/transfer_manifest/canonical_rows_sha256",
    )
    if sha256_bytes(canonical_json_bytes(rows)) != expected_canonical_rows:
        raise FetchError("transfer-manifest canonical row SHA-256 does not verify")
    return rows


def _validate_row(row: Mapping[str, Any], line_number: int) -> Tuple[str, str]:
    label = "manifest line {}".format(line_number)
    _require_exact_fields(row, ROW_FIELDS, label)
    if row.get("schema_version") != ROW_SCHEMA:
        raise FetchError("{} uses an unsupported schema".format(label))
    case_id = _require_safe_component(row.get("case_id"), "{}/case_id".format(label))
    arm = row.get("arm")
    if arm not in EXPECTED_ARMS:
        raise FetchError("{}/arm is unsupported".format(label))
    mode = row.get("mode")
    if mode != ARM_MODES[arm]:
        raise FetchError("{}/mode does not match its arm".format(label))
    ordinal = _require_nonnegative_int(
        row.get("case_ordinal"), "{}/case_ordinal".format(label)
    )
    if ordinal >= 1600:
        raise FetchError("{}/case_ordinal is outside [0, 1599]".format(label))
    task_index = _require_nonnegative_int(
        row.get("task_index"), "{}/task_index".format(label)
    )
    if task_index != ordinal // 50 or task_index >= 32:
        raise FetchError("{}/task_index does not match case_ordinal".format(label))
    _require_safe_component(
        row.get("task_level_group_id"), "{}/task_level_group_id".format(label)
    )
    if not isinstance(row.get("status"), str) or not row["status"]:
        raise FetchError("{}/status must be a nonempty string".format(label))

    result = _require_object(row.get("result"), "{}/result".format(label))
    _require_exact_fields(result, RESULT_FIELDS, "{}/result".format(label))
    result_path = _require_safe_relative_path(
        result.get("path"), "{}/result/path".format(label)
    )
    expected_prefix = "tasks/task-{}/results/{}/{}/".format(
        task_index, mode, case_id
    )
    if result_path != expected_prefix + "result.json":
        raise FetchError("{}/result/path is not the canonical case path".format(label))
    _require_positive_int(result.get("bytes"), "{}/result/bytes".format(label))
    _require_sha256(result.get("sha256"), "{}/result/sha256".format(label))
    _require_sha256(
        result.get("result_payload_sha256"),
        "{}/result/result_payload_sha256".format(label),
    )

    video = _require_object(row.get("video"), "{}/video".format(label))
    _require_exact_fields(video, VIDEO_FIELDS, "{}/video".format(label))
    video_path = _require_safe_relative_path(
        video.get("path"), "{}/video/path".format(label)
    )
    if video_path != expected_prefix + "episode.mp4":
        raise FetchError("{}/video/path is not the canonical case path".format(label))
    _require_positive_int(video.get("bytes"), "{}/video/bytes".format(label))
    _require_sha256(video.get("sha256"), "{}/video/sha256".format(label))
    _require_positive_int(video.get("frames"), "{}/video/frames".format(label))
    _require_positive_int(video.get("fps"), "{}/video/fps".format(label))
    if video.get("complete_episode") is not True:
        raise FetchError("{}/video is not a complete episode".format(label))
    _require_object(
        row.get("publisher_v2_inventory_item"),
        "{}/publisher_v2_inventory_item".format(label),
    )
    return case_id, arm


def select_video(
    manifest_path: Path,
    receipt_path: Path,
    case_id: str,
    arm: str,
) -> SelectedVideo:
    case_id = _require_safe_component(case_id, "requested case_id")
    if arm not in EXPECTED_ARMS:
        raise FetchError("requested arm is unsupported")
    receipt, _ = _load_json_object(receipt_path, "video-transfer receipt")
    run_root = _bound_run_root(receipt)
    rows = _load_and_verify_manifest(manifest_path, receipt)

    matches = []
    seen = set()
    for line_number, row in enumerate(rows, 1):
        identity = _validate_row(row, line_number)
        if identity in seen:
            raise FetchError(
                "manifest contains duplicate case/arm row: {}/{}".format(*identity)
            )
        seen.add(identity)
        if identity == (case_id, arm):
            matches.append(row)
    if not matches:
        raise FetchError("requested case/arm is absent from the manifest")
    if len(matches) != 1:
        raise FetchError("requested case/arm is ambiguous in the manifest")
    video = matches[0]["video"]
    return SelectedVideo(
        case_id=case_id,
        arm=arm,
        run_root=run_root,
        relative_path=str(video["path"]),
        bytes=int(video["bytes"]),
        sha256=str(video["sha256"]),
    )


class ScpTransport(VideoTransport):
    """Probe and stream one selected file over SSH."""

    _PROBE_SCRIPT = r"""set -eu
root=$1
relative=$2
[ -d "$root" ] && [ ! -L "$root" ] || exit 20
case "$relative" in
  /*|*..*|*[!A-Za-z0-9._/-]*) exit 21 ;;
esac
old_ifs=$IFS
IFS=/
set -- $relative
IFS=$old_ifs
candidate=$root
for component in "$@"; do
  [ -n "$component" ] && [ "$component" != "." ] && [ "$component" != ".." ] || exit 21
  candidate=$candidate/$component
  [ ! -L "$candidate" ] || exit 22
done
[ -f "$candidate" ] || exit 23
resolved_root=$(readlink -f -- "$root")
resolved_candidate=$(readlink -f -- "$candidate")
case "$resolved_candidate" in
  "$resolved_root"/*) ;;
  *) exit 24 ;;
esac
bytes=$(stat -c %s -- "$candidate")
sha=$(sha256sum -- "$candidate" | awk '{print $1}')
printf 'bytes=%s\nsha256=%s\n' "$bytes" "$sha"
"""

    _FETCH_SCRIPT = r"""set -eu
root=$1
relative=$2
expected_bytes=$3
[ -d "$root" ] && [ ! -L "$root" ] || exit 20
case "$relative" in
  /*|*..*|*[!A-Za-z0-9._/-]*) exit 21 ;;
esac
case "$expected_bytes" in
  ''|*[!0-9]*) exit 21 ;;
esac
old_ifs=$IFS
IFS=/
set -- $relative
IFS=$old_ifs
candidate=$root
for component in "$@"; do
  [ -n "$component" ] && [ "$component" != "." ] && [ "$component" != ".." ] || exit 21
  candidate=$candidate/$component
  [ ! -L "$candidate" ] || exit 22
done
[ -f "$candidate" ] || exit 23
resolved_root=$(readlink -f -- "$root")
resolved_candidate=$(readlink -f -- "$candidate")
case "$resolved_candidate" in
  "$resolved_root"/*) ;;
  *) exit 24 ;;
esac
exec 3< "$candidate"
opened_candidate=$(readlink -f -- "/proc/$$/fd/3")
case "$opened_candidate" in
  "$resolved_root"/*) ;;
  *) exit 24 ;;
esac
[ "$opened_candidate" = "$resolved_candidate" ] || exit 25
bytes=$(stat -Lc %s -- "/proc/$$/fd/3")
[ "$bytes" = "$expected_bytes" ] || exit 26
cat <&3
"""

    def __init__(self, host: str = "vinuni") -> None:
        if SAFE_HOST_RE.fullmatch(host) is None:
            raise FetchError("SSH host must be one safe configured alias")
        self.host = host

    def probe(self, run_root: str, relative_path: str) -> RemoteFileInfo:
        command = [
            "ssh",
            "--",
            self.host,
            "sh",
            "-s",
            "--",
            run_root,
            relative_path,
        ]
        completed = subprocess.run(
            command,
            input=self._PROBE_SCRIPT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise FetchError(
                "remote file probe failed safely (exit {}): {}".format(
                    completed.returncode, completed.stderr.strip()
                )
            )
        fields: Dict[str, str] = {}
        for line in completed.stdout.splitlines():
            key, separator, value = line.partition("=")
            if not separator or key in fields:
                raise FetchError("remote probe returned an ambiguous record")
            fields[key] = value
        if set(fields) != {"bytes", "sha256"}:
            raise FetchError("remote probe returned unexpected fields")
        try:
            size = int(fields["bytes"])
        except ValueError as error:
            raise FetchError("remote probe returned an invalid byte count") from error
        return RemoteFileInfo(
            bytes=_require_positive_int(size, "remote video bytes"),
            sha256=_require_sha256(fields["sha256"], "remote video SHA-256"),
        )

    def fetch(
        self,
        run_root: str,
        relative_path: str,
        expected_bytes: int,
        expected_sha256: str,
        destination_fd: int,
    ) -> None:
        if (
            SAFE_REMOTE_ROOT_RE.fullmatch(run_root) is None
            or _require_safe_relative_path(relative_path, "remote MP4 path")
            != relative_path
        ):
            raise FetchError("remote MP4 path is unsafe")
        expected_bytes = _require_positive_int(
            expected_bytes, "expected remote video bytes"
        )
        _require_sha256(expected_sha256, "expected remote video SHA-256")
        descriptor_stat = os.fstat(destination_fd)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise FetchError("download destination is not an open regular file")
        completed = subprocess.run(
            [
                "ssh",
                "--",
                self.host,
                "sh",
                "-s",
                "--",
                run_root,
                relative_path,
                str(expected_bytes),
            ],
            input=self._FETCH_SCRIPT,
            text=True,
            stdout=destination_fd,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise FetchError(
                "single-file SSH stream failed (exit {}): {}".format(
                    completed.returncode,
                    completed.stderr.strip(),
                )
            )


def _prepare_output(output: Path) -> Path:
    output = output.absolute()
    if output.suffix.lower() != ".mp4":
        raise FetchError("output must have an .mp4 suffix")
    _require_safe_component(output.name, "output filename")
    if os.path.lexists(str(output)):
        raise FetchError("output already exists; overwrite is not supported")
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir():
        raise FetchError("output parent must be an existing real directory")
    current = Path(parent.anchor)
    for component in parent.parts[1:]:
        current = current / component
        try:
            component_stat = os.lstat(str(current))
        except OSError as error:
            raise FetchError(
                "output parent ancestry cannot be verified: {}".format(current)
            ) from error
        if stat.S_ISLNK(component_stat.st_mode):
            raise FetchError(
                "output parent ancestry contains a symlink: {}".format(current)
            )
    return output


def fetch_selected_video(
    selected: SelectedVideo,
    output: Path,
    transport: VideoTransport,
) -> Path:
    output = _prepare_output(output)
    remote = transport.probe(selected.run_root, selected.relative_path)
    if not remote.regular or remote.symlink:
        raise FetchError("remote selected video is not a real regular file")
    if remote.bytes != selected.bytes or remote.sha256 != selected.sha256:
        raise FetchError("remote selected video differs from its manifest identity")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.fetch-".format(output.name),
        suffix=".partial",
        dir=str(output.parent),
    )
    temporary = Path(temporary_name)
    published = False
    try:
        transport.fetch(
            selected.run_root,
            selected.relative_path,
            selected.bytes,
            selected.sha256,
            descriptor,
        )
        _require_path_matches_fd(temporary, descriptor, "downloaded video")
        observed_sha, observed_size = stable_fd_sha256_and_size(
            descriptor, "downloaded video"
        )
        if observed_size != selected.bytes or observed_sha != selected.sha256:
            raise FetchError("downloaded video differs from its manifest identity")
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        _require_path_matches_fd(temporary, descriptor, "downloaded video")
        try:
            os.link(str(temporary), str(output), follow_symlinks=False)
        except FileExistsError as error:
            raise FetchError("output appeared during transfer; refusing to replace it") from error
        published = True
        _require_path_matches_fd(output, descriptor, "published video")
        directory_descriptor = os.open(str(output.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if os.path.lexists(str(temporary)):
            os.unlink(str(temporary))
        os.close(descriptor)
    if not published:
        raise FetchError("video was not atomically published")
    final_sha, final_size = stable_file_sha256_and_size(output, "published video")
    if final_size != selected.bytes or final_sha != selected.sha256:
        raise FetchError("published video failed its final identity check")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch one hash-bound VLSA Table-1 MP4; never a directory or wildcard."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--arm", required=True, choices=EXPECTED_ARMS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--host", default="vinuni")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selected = select_video(
            manifest_path=args.manifest,
            receipt_path=args.receipt,
            case_id=args.case_id,
            arm=args.arm,
        )
        output = fetch_selected_video(
            selected=selected,
            output=args.output,
            transport=ScpTransport(args.host),
        )
    except FetchError as error:
        print("video fetch refused: {}".format(error), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "fetched_one_video",
                "case_id": selected.case_id,
                "arm": selected.arm,
                "output": str(output),
                "bytes": selected.bytes,
                "sha256": selected.sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
