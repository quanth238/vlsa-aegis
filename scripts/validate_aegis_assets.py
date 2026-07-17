#!/usr/bin/env python3
"""Allocation-side source and asset preflight for the AEGIS reproduction.

The script deliberately uses only the Python standard library.  It is safe to
import in local structural tests, but production invocations belong inside a
Slurm allocation.  A successful invocation writes one immutable JSON receipt;
an existing receipt is never replaced.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable


CONFIG_SCHEMA = "vlsa_table1_translational_protocol.v1"
MANIFEST_SCHEMA = "vlsa_table1_population_case.v1"
RECEIPT_SCHEMA = "vlsa_table1_population_receipt.v1"
PREFLIGHT_SCHEMA = "vlsa_table1_allocation_preflight.v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")

DEFAULT_PI05_CHECKPOINT = Path(
    "/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero"
)
DEFAULT_DINO_CONFIG = Path(
    "/mnt/data/quanth/cache/uv/archive-v0/hHOpLbugg_lAlUaF/"
    "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
DEFAULT_DINO_CHECKPOINT = Path(
    "/mnt/data/quanth/cache/aegis/groundingdino/"
    "groundingdino_swint_ogc.pth"
)

# The large OCDBT objects are identified by the exact object names and sizes
# frozen on VinUni.  Orbax metadata and manifests are content-hashed below.
# Passing --expected-pi05-tree-sha256 additionally performs a full 15 GB
# content hash when a preregistered tree digest is available.
PI05_METADATA_FILES: dict[str, tuple[int, str]] = {
    "assets/physical-intelligence/libero/norm_stats.json": (
        1914,
        "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
    ),
    "params/_METADATA": (
        23493,
        "303a4e354814928e1d29b75e310f2c1ac7e7e29b62f48395b631045ca1cffc73",
    ),
    "params/_sharding": (
        17952,
        "63f4c57ba6ff10f4132a639b9942eabcc26942eb34081be1d93bf7ddab816501",
    ),
    "params/array_metadatas/process_0": (
        9062,
        "2b29474f08aa50922da11074deb4d7f35e30f0071a088a30f40df88edc1ebdb0",
    ),
    "params/manifest.ocdbt": (
        120,
        "65246951e69bd2b5118e609646bc9e8c439229bccbc1325643833aeb74f77104",
    ),
    "params/ocdbt.process_0/manifest.ocdbt": (
        322,
        "3bf70fbb0fac151675595b33aeb8203139e8809de17d606574ab606a758b3591",
    ),
    "params/d/98a77a52a8eb845ae4830eb7fe983979": (
        42307,
        "93b39327a1552b06b1b6ce5190d58131297bcca66c3f2a099bbb2882df7759af",
    ),
    "params/ocdbt.process_0/d/2b6985f48e9da86f68627a7608c5bc25": (
        1077,
        "3b81b2e8afe1456a4584d19dcf600f2bb6f27beff9f061b336f06657a11d5cb5",
    ),
    "params/ocdbt.process_0/d/bc613ff288a162563e622d01bb60622a": (
        217,
        "1e99d5876e6b8db2e12c43c8079f6b98cffac2c26bbd84e59cc505e983d146d2",
    ),
    "params/ocdbt.process_0/d/bda87d9791f23df771cd2d15293780cc": (
        2926522,
        "9c717ff13084524b74f330803421b8a261bc3009a477c933e1e1a1477a796fc3",
    ),
}

PI05_DATA_FILES: dict[str, int] = {
    "params/ocdbt.process_0/d/0eaaecefaa9720d30a32cc56e65fd345": 2449323387,
    "params/ocdbt.process_0/d/155391c1cbf93a1be16266de052e5b48": 2307885530,
    "params/ocdbt.process_0/d/475fab3ee8821662585a8cde3eb32e22": 2150232827,
    "params/ocdbt.process_0/d/6c54da5a6f62c20a09c0f3f8c3329e00": 1608436908,
    "params/ocdbt.process_0/d/896bf93c5cf2e8ddd274a6ea0a2feec0": 2240080987,
    "params/ocdbt.process_0/d/efbb46173882cb35ed41ffe0c2db8a5e": 1680102856,
}

DINO_FILES: dict[str, tuple[int, str]] = {
    "config": (
        1006,
        "172e80017f9395668a9cb5d1b8bd9d061c0e360471c6ed673c83b69bb14399f1",
    ),
    "checkpoint": (
        693997677,
        "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
    ),
}

ALLOWED_LABELS = {
    "yellow rectangular book",
    "blue moka pot",
    "red mug",
    "white storage box",
    "black wine bottle",
    "red milk carton",
}
LONG_EXTRA_LABELS = {"gray rectangular binder"}


class PreflightError(RuntimeError):
    """Raised when a frozen input differs from the preregistered contract."""


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
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, *, label: str) -> str:
    if not SHA256_PATTERN.fullmatch(value):
        raise PreflightError(f"{label} must be a lowercase SHA-256")
    return value


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise PreflightError(f"{label} is missing, symlinked, or not a file: {path}")
    return path


def validate_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    _regular_file(path, label=label)
    observed_size = path.stat().st_size
    if observed_size != expected_size:
        raise PreflightError(
            f"{label} size changed: {observed_size}, expected {expected_size}"
        )
    observed_sha256 = sha256_path(path)
    if observed_sha256 != _require_sha256(expected_sha256, label=label):
        raise PreflightError(f"{label} SHA-256 changed")
    return {
        "path": str(path.resolve()),
        "bytes": observed_size,
        "sha256": observed_sha256,
    }


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise PreflightError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def validate_source(
    repo_root: Path,
    *,
    expected_commit: str,
    upstream_commit: str,
) -> dict[str, Any]:
    if not COMMIT_PATTERN.fullmatch(expected_commit):
        raise PreflightError("expected source commit must be 40 lowercase hex digits")
    if not COMMIT_PATTERN.fullmatch(upstream_commit):
        raise PreflightError("upstream commit must be 40 lowercase hex digits")
    observed_commit = _git(repo_root, "rev-parse", "HEAD")
    if observed_commit != expected_commit:
        raise PreflightError(
            f"source commit changed: {observed_commit}, expected {expected_commit}"
        )
    dirty = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if dirty:
        raise PreflightError(f"source tree is not clean: {dirty.splitlines()[0]}")
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "merge-base",
            "--is-ancestor",
            upstream_commit,
            expected_commit,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if ancestry.returncode != 0:
        raise PreflightError("the frozen upstream release is not an ancestor of HEAD")
    changed_paths = _git(
        repo_root,
        "diff",
        "--name-only",
        f"{upstream_commit}..{expected_commit}",
    ).splitlines()
    return {
        "repo_root": str(repo_root.resolve()),
        "git_commit": observed_commit,
        "git_dirty": False,
        "upstream_commit": upstream_commit,
        "upstream_is_ancestor": True,
        "changed_paths_from_upstream": changed_paths,
    }


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _regular_file(path, label=label).read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise PreflightError(f"{label} must contain one JSON object")
    return value, raw


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = _regular_file(path, label="population manifest").read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PreflightError(
                f"population manifest line {line_number} is invalid"
            ) from error
        if not isinstance(value, dict):
            raise PreflightError(
                f"population manifest line {line_number} is not an object"
            )
        rows.append(value)
    return rows, raw


def _bound_repo_path(repo_root: Path, relative: str, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PreflightError(f"{label} path is missing")
    candidate = repo_root / relative
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as error:
        raise PreflightError(f"{label} escapes the source tree") from error
    if candidate.is_symlink() or not candidate.is_file():
        raise PreflightError(f"{label} is missing or symlinked: {relative}")
    return candidate


def validate_protocol(
    repo_root: Path,
    *,
    config_path: Path,
    manifest_path: Path,
    receipt_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config, config_raw = _load_json(config_path, label="protocol config")
    receipt, _ = _load_json(receipt_path, label="manifest receipt")
    rows, manifest_raw = _load_jsonl(manifest_path)
    config_sha256 = sha256_bytes(config_raw)
    manifest_sha256 = sha256_bytes(manifest_raw)

    if config.get("schema_version") != CONFIG_SCHEMA:
        raise PreflightError("unexpected protocol config schema")
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise PreflightError("unexpected manifest receipt schema")
    if receipt.get("protocol_config_sha256") != config_sha256:
        raise PreflightError("protocol config differs from the manifest receipt")
    if receipt.get("manifest_sha256") != manifest_sha256:
        raise PreflightError("population manifest differs from its receipt")
    expected_rows = int(config["population"]["expected_cases"])
    if (
        expected_rows != 1600
        or receipt.get("manifest_rows") != expected_rows
        or len(rows) != expected_rows
    ):
        raise PreflightError("population is not the frozen 1,600-case set")

    seen: set[str] = set()
    source_hashes: dict[str, str] = {}
    for row in rows:
        case_id = row.get("case_id")
        if (
            row.get("schema_version") != MANIFEST_SCHEMA
            or row.get("protocol_id") != config["protocol_id"]
            or row.get("protocol_config_sha256") != config_sha256
            or not isinstance(case_id, str)
            or case_id in seen
        ):
            raise PreflightError(f"invalid or duplicate manifest case: {case_id!r}")
        seen.add(case_id)
        for path_key, hash_key, label in (
            ("bddl_path", "bddl_sha256", "BDDL source"),
            (
                "initial_states_path",
                "initial_states_sha256",
                "initial-state source",
            ),
        ):
            relative = row.get(path_key)
            expected_hash = row.get(hash_key)
            if not isinstance(expected_hash, str):
                raise PreflightError(f"{case_id}: {hash_key} is missing")
            expected_hash = _require_sha256(
                expected_hash, label=f"{case_id}/{hash_key}"
            )
            if relative not in source_hashes:
                source_path = _bound_repo_path(
                    repo_root, relative, label=f"{case_id}/{label}"
                )
                source_hashes[relative] = sha256_path(source_path)
            if source_hashes[relative] != expected_hash:
                raise PreflightError(f"{case_id}: {label} hash changed")

    return config, rows, {
        "config": {
            "path": str(config_path.resolve()),
            "sha256": config_sha256,
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha256,
            "rows": len(rows),
        },
        "receipt": {
            "path": str(receipt_path.resolve()),
            "sha256": sha256_path(receipt_path),
        },
        "frozen_safelibero_sources": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(source_hashes.items())
        ],
    }


def _tree_content_sha256(root: Path, relative_paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        path = _regular_file(root / relative, label=f"pi0.5 asset {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_path(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_pi05_checkpoint(
    checkpoint: Path,
    *,
    expected_tree_sha256: str | None,
) -> dict[str, Any]:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise PreflightError(f"pi0.5 checkpoint is missing or symlinked: {checkpoint}")
    expected_paths = set(PI05_METADATA_FILES) | set(PI05_DATA_FILES)
    observed_paths = {
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    if observed_paths != expected_paths:
        missing = sorted(expected_paths - observed_paths)
        extra = sorted(observed_paths - expected_paths)
        raise PreflightError(
            f"pi0.5 checkpoint inventory changed; missing={missing}, extra={extra}"
        )

    metadata: list[dict[str, Any]] = []
    for relative, (size, digest) in sorted(PI05_METADATA_FILES.items()):
        record = validate_file(
            checkpoint / relative,
            expected_size=size,
            expected_sha256=digest,
            label=f"pi0.5 metadata {relative}",
        )
        record["relative_path"] = relative
        metadata.append(record)

    data_inventory: list[dict[str, Any]] = []
    for relative, expected_size in sorted(PI05_DATA_FILES.items()):
        path = _regular_file(
            checkpoint / relative, label=f"pi0.5 data object {relative}"
        )
        size = path.stat().st_size
        if size != expected_size:
            raise PreflightError(
                f"pi0.5 data object {relative} size changed: "
                f"{size}, expected {expected_size}"
            )
        data_inventory.append(
            {"relative_path": relative, "bytes": size}
        )

    structural_fingerprint = sha256_bytes(
        canonical_json_bytes(
            {
                "metadata": [
                    {
                        "relative_path": item["relative_path"],
                        "bytes": item["bytes"],
                        "sha256": item["sha256"],
                    }
                    for item in metadata
                ],
                "data_inventory": data_inventory,
            }
        )
    )
    result: dict[str, Any] = {
        "path": str(checkpoint.resolve()),
        "format": "jax_orbax_ocdbt",
        "metadata": metadata,
        "data_inventory": data_inventory,
        "structural_fingerprint_sha256": structural_fingerprint,
        "full_content_tree_sha256": None,
        "full_content_hash_verified": False,
    }
    if expected_tree_sha256 is not None:
        expected_tree_sha256 = _require_sha256(
            expected_tree_sha256, label="pi0.5 full tree"
        )
        observed_tree_sha256 = _tree_content_sha256(
            checkpoint, expected_paths
        )
        if observed_tree_sha256 != expected_tree_sha256:
            raise PreflightError("pi0.5 full checkpoint content hash changed")
        result["full_content_tree_sha256"] = observed_tree_sha256
        result["full_content_hash_verified"] = True
    return result


def validate_evaluation_assets(
    *,
    cases: list[dict[str, Any]],
    required_case_ordinals: list[int],
    require_complete_label_population: bool,
    pi05_checkpoint: Path,
    expected_pi05_tree_sha256: str | None,
    dino_config: Path,
    dino_checkpoint: Path,
    label_manifest: Path,
    expected_label_manifest_sha256: str,
) -> dict[str, Any]:
    label_manifest = _regular_file(
        label_manifest, label="frozen Codex label manifest"
    )
    label_sha256 = sha256_path(label_manifest)
    if label_sha256 != _require_sha256(
        expected_label_manifest_sha256, label="frozen Codex label manifest"
    ):
        raise PreflightError("frozen Codex label manifest hash changed")
    label_rows, _ = _load_jsonl(label_manifest)
    cases_by_id = {str(case["case_id"]): case for case in cases}
    labels_by_id: dict[str, dict[str, Any]] = {}
    for row in label_rows:
        case_id = row.get("case_id")
        if (
            row.get("schema_version")
            not in {
                "vlsa_table1_codex_label.v1",
                "aegis_codex_semantic_label.v1",
            }
            or not isinstance(case_id, str)
            or case_id not in cases_by_id
            or case_id in labels_by_id
        ):
            raise PreflightError(
                f"invalid, duplicate, or unexpected frozen label: {case_id!r}"
            )
        image_hash = None
        for key in (
            "settled_agentview_array_sha256",
            "agentview_array_sha256",
            "agentview_image_sha256",
        ):
            if row.get(key):
                image_hash = row[key]
                break
        if image_hash is None and isinstance(row.get("agentview"), dict):
            image_hash = row["agentview"].get("array_sha256")
        if not isinstance(image_hash, str):
            raise PreflightError(f"{case_id}: settled-image hash is missing")
        _require_sha256(image_hash, label=f"{case_id}/settled image")

        label = " ".join(
            str(row.get("obstacle_label", "")).split()
        ).lower()
        allowed = set(ALLOWED_LABELS)
        if cases_by_id[case_id].get("suite") == "safelibero_long":
            allowed.update(LONG_EXTRA_LABELS)
        if label not in allowed:
            raise PreflightError(
                f"{case_id}: obstacle label {label!r} is not preregistered"
            )
        reviewed_at = row.get("reviewed_at")
        if row.get("reviewer") != "codex" or not isinstance(
            reviewed_at, str
        ):
            raise PreflightError(
                f"{case_id}: Codex reviewer identity or review time is missing"
            )
        try:
            parsed_reviewed_at = datetime.fromisoformat(
                reviewed_at.replace("Z", "+00:00")
            )
        except ValueError as error:
            raise PreflightError(
                f"{case_id}: Codex review time is not ISO-8601"
            ) from error
        if (
            parsed_reviewed_at.tzinfo is None
            or parsed_reviewed_at.utcoffset() is None
        ):
            raise PreflightError(
                f"{case_id}: Codex review time must include a timezone"
            )
        labels_by_id[case_id] = row
    if require_complete_label_population:
        required_case_ids = set(cases_by_id)
    else:
        required_case_ids = set()
        for ordinal in required_case_ordinals:
            if ordinal < 0 or ordinal >= len(cases):
                raise PreflightError(
                    f"required case ordinal is outside the manifest: {ordinal}"
                )
            required_case_ids.add(str(cases[ordinal]["case_id"]))
    if not required_case_ids:
        raise PreflightError(
            "evaluation preflight must bind at least one required label"
        )
    missing = sorted(required_case_ids - set(labels_by_id))
    if missing:
        raise PreflightError(
            f"frozen Codex label manifest is missing required cases: {missing[:3]}"
        )
    return {
        "pi05_checkpoint": validate_pi05_checkpoint(
            pi05_checkpoint,
            expected_tree_sha256=expected_pi05_tree_sha256,
        ),
        "groundingdino": {
            "config": validate_file(
                dino_config,
                expected_size=DINO_FILES["config"][0],
                expected_sha256=DINO_FILES["config"][1],
                label="GroundingDINO config",
            ),
            "checkpoint": validate_file(
                dino_checkpoint,
                expected_size=DINO_FILES["checkpoint"][0],
                expected_sha256=DINO_FILES["checkpoint"][1],
                label="GroundingDINO checkpoint",
            ),
        },
        "frozen_codex_labels": {
            "path": str(label_manifest.resolve()),
            "sha256": label_sha256,
            "rows": len(label_rows),
            "required_cases": len(required_case_ids),
            "all_population_cases_bound": (
                set(labels_by_id) == set(cases_by_id)
            ),
        },
    }


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    """Publish complete JSON atomically and refuse to replace an artifact."""

    if path.exists() or path.is_symlink():
        raise PreflightError(f"preflight receipt already exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise PreflightError(
            f"preflight receipt parent must be an existing directory: {path.parent}"
        )
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise PreflightError(
                f"preflight receipt was concurrently created: {path}"
            ) from error
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Validate clean source, the immutable SafeLIBERO population, and "
            "the assets required by one allocation."
        )
    )
    parser.add_argument(
        "--profile", choices=("capture", "evaluation"), required=True
    )
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_root / "configs/vlsa_table1_translational.json",
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--manifest-receipt",
        type=Path,
        default=default_root
        / "manifests/vlsa_table1_population.receipt.json",
    )
    parser.add_argument(
        "--pi05-checkpoint",
        type=Path,
        default=DEFAULT_PI05_CHECKPOINT,
    )
    parser.add_argument("--expected-pi05-tree-sha256")
    parser.add_argument(
        "--dino-config", type=Path, default=DEFAULT_DINO_CONFIG
    )
    parser.add_argument(
        "--dino-checkpoint", type=Path, default=DEFAULT_DINO_CHECKPOINT
    )
    parser.add_argument("--label-manifest", type=Path)
    parser.add_argument("--expected-label-manifest-sha256")
    parser.add_argument(
        "--required-case-ordinal",
        action="append",
        type=int,
        default=[],
    )
    parser.add_argument(
        "--require-complete-label-population",
        action="store_true",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        repo_root = args.repo_root.resolve()
        config, rows, protocol = validate_protocol(
            repo_root,
            config_path=args.config.resolve(),
            manifest_path=args.manifest.resolve(),
            receipt_path=args.manifest_receipt.resolve(),
        )
        source = validate_source(
            repo_root,
            expected_commit=args.expected_commit,
            upstream_commit=config["source"]["upstream_commit"],
        )
        assets: dict[str, Any] = {
            "profile": args.profile,
            "policy_model_executed": False,
            "groundingdino_executed": False,
            "qp_executed": False,
        }
        if args.profile == "evaluation":
            if (
                args.label_manifest is None
                or args.expected_label_manifest_sha256 is None
            ):
                raise PreflightError(
                    "evaluation requires a frozen label manifest and its "
                    "preregistered SHA-256"
                )
            assets.update(
                validate_evaluation_assets(
                    cases=rows,
                    required_case_ordinals=args.required_case_ordinal,
                    require_complete_label_population=(
                        args.require_complete_label_population
                    ),
                    pi05_checkpoint=args.pi05_checkpoint.resolve(),
                    expected_pi05_tree_sha256=(
                        args.expected_pi05_tree_sha256
                    ),
                    dino_config=args.dino_config.resolve(),
                    dino_checkpoint=args.dino_checkpoint.resolve(),
                    label_manifest=args.label_manifest.resolve(),
                    expected_label_manifest_sha256=(
                        args.expected_label_manifest_sha256
                    ),
                )
            )

        result = {
            "schema_version": PREFLIGHT_SCHEMA,
            "status": "passed",
            "scientific_result": False,
            "profile": args.profile,
            "protocol_id": config["protocol_id"],
            "source": source,
            "protocol": protocol,
            "assets": assets,
            "case_count": len(rows),
        "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "host": os.environ.get("SLURMD_NODENAME"),
            },
        }
        if args.profile == "evaluation":
            result["assets"]["groundingdino"]["device"] = os.environ.get(
                "GROUNDINGDINO_DEVICE", "unrecorded"
            )
        result["receipt_payload_sha256"] = sha256_bytes(
            canonical_json_bytes(result)
        )
        write_json_exclusive(args.output.resolve(), result)
        json.dump(
            {
                "status": "passed",
                "profile": args.profile,
                "output": str(args.output.resolve()),
                "receipt_payload_sha256": result["receipt_payload_sha256"],
            },
            sys.stdout,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 0
    except (KeyError, OSError, PreflightError) as error:
        print(f"asset preflight failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
