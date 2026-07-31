#!/usr/bin/env python3
"""Build the immutable 109-case link-5/6 Poisson feasibility manifest.

The input case audit is deliberately external to Git because it is an output
of the completed 3,200-rollout Table-1 reproduction.  This builder binds that
audit to the checked-in 1,600-case population by SHA-256 and by an exact
one-to-one case join.  Selection is literal: a semicolon-delimited contact
body token must equal ``robot0_link5`` or ``robot0_link6``.

Only the Python standard library is used, and the emitted JSONL order and
canonical encoding are part of the protocol contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


CONFIG_SCHEMA = "vlsa_poisson_link56_feasibility_protocol.v1"
MANIFEST_SCHEMA = "vlsa_poisson_link56_case.v1"
RECEIPT_SCHEMA = "vlsa_poisson_link56_manifest_receipt.v1"

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")

_REQUIRED_AUDIT_COLUMNS = {
    "case_id",
    "suite",
    "task_level_group_id",
    "task_name",
    "safety_level",
    "episode_index",
    "active_obstacle_name",
    "frozen_obstacle_label",
    "aegis_collision",
    "aegis_task_success",
    "aegis_settled_relevant_contact",
    "aegis_postcontrol_robot_contact",
    "aegis_geometry_complete",
    "aegis_all_actions_solved_qp",
    "observed_car_class",
    "refined_collision_mechanism",
    "collision_first_step",
    "first_relevant_contact_step",
    "barrier_h_at_first_relevant_contact",
    "constraint_lhs_at_first_relevant_contact",
    "barrier_h_at_car_crossing",
    "robot_contact_bodies",
}

_JOIN_IDENTITY_FIELDS = (
    "suite",
    "task_level_group_id",
    "task_name",
    "safety_level",
    "episode_index",
)


class ManifestBuildError(RuntimeError):
    """Raised when a source or generated artifact violates the protocol."""


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


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ManifestBuildError("{} must be a lowercase SHA-256".format(label))
    return value


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestBuildError("{} must be an integer".format(label))
    return value


def _parse_bool(value: str, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ManifestBuildError("{} must be exactly True or False".format(label))


def _parse_optional_int(value: str, label: str) -> Optional[int]:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as error:
        raise ManifestBuildError("{} must be an integer or empty".format(label)) from error


def _parse_optional_float(value: str, label: str) -> Optional[float]:
    if value == "":
        return None
    try:
        result = float(value)
    except ValueError as error:
        raise ManifestBuildError("{} must be finite or empty".format(label)) from error
    if not math.isfinite(result):
        raise ManifestBuildError("{} must be finite or empty".format(label))
    return result


def literal_contact_bodies(value: str) -> Tuple[str, ...]:
    """Parse exact body tokens without substring or regex matching."""

    if value == "":
        return ()
    bodies = tuple(part.strip() for part in value.split(";"))
    if any(not body for body in bodies):
        raise ManifestBuildError("robot_contact_bodies contains an empty token")
    if len(set(bodies)) != len(bodies):
        raise ManifestBuildError("robot_contact_bodies contains a duplicate token")
    return bodies


def _validated_contact_lineage(
    value: Any, *, case_id: str, pair_index: int, field: str
) -> Tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(name, str) or not name for name in value)
    ):
        raise ManifestBuildError(
            "{} historical contact pair {} {} is invalid".format(
                case_id, pair_index, field
            )
        )
    lineage = tuple(value)
    if len(set(lineage)) != len(lineage):
        raise ManifestBuildError(
            "{} historical contact pair {} {} contains duplicate bodies".format(
                case_id, pair_index, field
            )
        )
    return lineage


def _validated_contact_geom_name(
    value: Any, *, case_id: str, pair_index: int, field: str
) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ManifestBuildError(
            "{} historical contact pair {} {} is invalid".format(
                case_id, pair_index, field
            )
        )
    return value


def matched_direct_contact_pair_evidence(
    *,
    unique_pairs: Any,
    active_obstacle_name: Any,
    audited_links: Set[str],
    case_id: str,
) -> List[Dict[str, Any]]:
    """Bind each audited protected link to the active obstacle in one pair.

    ``body_lineage1`` and ``body_lineage2`` are ordered from the geom's
    directly attached body to ``world``.  A protected link therefore counts
    only when it is the first body on one side of a *single* historical pair;
    seeing that link in one pair and the active obstacle in another is not
    evidence of their contact.  The obstacle root-name rule is the exact
    naming contract used by the frozen Table-1 contact authority: the root is
    either the task-object name or starts with ``<task-object-name>_``.
    """

    protected_links = {"robot0_link5", "robot0_link6"}
    if (
        not isinstance(active_obstacle_name, str)
        or not active_obstacle_name
        or not audited_links
        or not audited_links.issubset(protected_links)
    ):
        raise ManifestBuildError(
            "{} direct-contact binding inputs are invalid".format(case_id)
        )
    if not isinstance(unique_pairs, list):
        raise ManifestBuildError(
            "{} historical contact pairs are invalid".format(case_id)
        )

    evidence: List[Dict[str, Any]] = []
    seen_pair_hashes: Set[str] = set()
    matched_links: Set[str] = set()
    required_fields = {"geom1", "geom2", "body_lineage1", "body_lineage2"}
    for pair_index, pair in enumerate(unique_pairs):
        if not isinstance(pair, dict) or set(pair) != required_fields:
            raise ManifestBuildError(
                "{} historical contact pair {} has invalid fields".format(
                    case_id, pair_index
                )
            )
        lineages = (
            _validated_contact_lineage(
                pair["body_lineage1"],
                case_id=case_id,
                pair_index=pair_index,
                field="body_lineage1",
            ),
            _validated_contact_lineage(
                pair["body_lineage2"],
                case_id=case_id,
                pair_index=pair_index,
                field="body_lineage2",
            ),
        )
        geom_names = (
            _validated_contact_geom_name(
                pair["geom1"],
                case_id=case_id,
                pair_index=pair_index,
                field="geom1",
            ),
            _validated_contact_geom_name(
                pair["geom2"],
                case_id=case_id,
                pair_index=pair_index,
                field="geom2",
            ),
        )
        pair_sha256 = sha256_bytes(canonical_json_bytes(pair))
        if pair_sha256 in seen_pair_hashes:
            raise ManifestBuildError(
                "{} historical contact pairs are not unique".format(case_id)
            )
        seen_pair_hashes.add(pair_sha256)

        orientations: List[Dict[str, Any]] = []
        for protected_side in (0, 1):
            obstacle_side = 1 - protected_side
            protected_lineage = lineages[protected_side]
            obstacle_lineage = lineages[obstacle_side]
            protected_body = protected_lineage[0]
            if protected_body not in audited_links:
                continue
            obstacle_root_candidates = [
                body
                for body in obstacle_lineage
                if body == active_obstacle_name
                or body.startswith(active_obstacle_name + "_")
            ]
            if not obstacle_root_candidates:
                continue
            # The lineage is contact-body-to-world, so the last matching task
            # body is the active object's root when nested object bodies exist.
            obstacle_root_body = obstacle_root_candidates[-1]
            orientations.append(
                {
                    "historical_unique_contact_pair_sha256": pair_sha256,
                    "protected_link_body_name": protected_body,
                    "protected_link_geom_name": geom_names[protected_side],
                    "active_obstacle_name": active_obstacle_name,
                    "active_obstacle_contact_body_name": obstacle_lineage[0],
                    "active_obstacle_root_body_name": obstacle_root_body,
                    "active_obstacle_geom_name": geom_names[obstacle_side],
                    "protected_link_body_lineage": list(protected_lineage),
                    "active_obstacle_body_lineage": list(obstacle_lineage),
                }
            )
        if len(orientations) > 1:
            raise ManifestBuildError(
                "{} historical contact pair {} has ambiguous sides".format(
                    case_id, pair_index
                )
            )
        if orientations:
            evidence.append(orientations[0])
            matched_links.add(orientations[0]["protected_link_body_name"])

    missing_links = sorted(audited_links - matched_links)
    if missing_links:
        raise ManifestBuildError(
            "{} historical result has no direct active-obstacle pair for {}".format(
                case_id, ", ".join(missing_links)
            )
        )
    return sorted(evidence, key=canonical_json_bytes)


def semantic_task_family_id(suite: str, task_name: str) -> str:
    """Return the registered split-block ID for one semantic task family."""

    payload = suite.encode("utf-8") + b"\0" + task_name.encode("utf-8")
    return sha256_bytes(payload)


def validate_runtime_protocol_binding(
    config_path: Path, config: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate the separately versioned numerical protocol and both hashes."""

    binding = config.get("runtime_protocol")
    if not isinstance(binding, dict):
        raise ManifestBuildError("runtime_protocol binding is missing")
    required = {
        "relative_path",
        "schema_version",
        "protocol_id",
        "semantic_protocol_sha256",
        "parameter_block_sha256",
        "raw_file_sha256",
        "selection_status",
    }
    if set(binding) != required:
        raise ManifestBuildError("runtime_protocol binding fields are invalid")
    relative = binding.get("relative_path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ManifestBuildError("runtime protocol path must be relative")
    candidate = config_path.resolve().parent.parent / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ManifestBuildError("runtime protocol file is missing or symlinked")
    if sha256_path(candidate) != _require_sha256(
        binding.get("raw_file_sha256"), "runtime protocol raw file hash"
    ):
        raise ManifestBuildError("runtime protocol raw file SHA-256 differs")
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestBuildError("runtime protocol JSON is invalid") from error
    if not isinstance(payload, dict):
        raise ManifestBuildError("runtime protocol must contain one object")
    if payload.get("schema_version") != binding.get("schema_version"):
        raise ManifestBuildError("runtime protocol schema binding differs")
    if payload.get("protocol_id") != binding.get("protocol_id"):
        raise ManifestBuildError("runtime protocol ID binding differs")
    semantic_hash = sha256_bytes(canonical_json_bytes(payload))
    if semantic_hash != _require_sha256(
        binding.get("semantic_protocol_sha256"),
        "runtime semantic protocol hash",
    ):
        raise ManifestBuildError("runtime semantic protocol SHA-256 differs")
    selection = payload.get("parameter_selection")
    if not isinstance(selection, dict):
        raise ManifestBuildError("runtime parameter selection is invalid")
    sections = selection.get("parameter_sections")
    if not isinstance(sections, list) or not sections or any(
        not isinstance(section, str) or section not in payload for section in sections
    ):
        raise ManifestBuildError("runtime parameter section registry is invalid")
    parameter_hash = sha256_bytes(
        canonical_json_bytes({section: payload[section] for section in sections})
    )
    for observed, declared, label in (
        (
            parameter_hash,
            selection.get("parameter_block_sha256"),
            "runtime-declared parameter hash",
        ),
        (
            parameter_hash,
            binding.get("parameter_block_sha256"),
            "runtime-bound parameter hash",
        ),
    ):
        if observed != _require_sha256(declared, label):
            raise ManifestBuildError("runtime parameter block SHA-256 differs")
    if binding.get("selection_status") != "development_canary_not_heldout_frozen":
        raise ManifestBuildError("runtime parameter selection status is unsupported")
    return payload


def load_config(config_path: Path) -> Tuple[Dict[str, Any], str]:
    raw = config_path.read_bytes()
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ManifestBuildError("invalid protocol config JSON") from error
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ManifestBuildError("unexpected protocol config schema")
    validate_runtime_protocol_binding(config_path, config)

    source = config.get("source", {})
    _require_sha256(source.get("table1_manifest_sha256"), "source manifest hash")
    _require_sha256(source.get("case_audit_sha256"), "source audit hash")
    if _require_int(source.get("table1_manifest_rows"), "source manifest rows") != 1600:
        raise ManifestBuildError("source population must contain 1,600 cases")
    if _require_int(source.get("case_audit_rows"), "source audit rows") != 1600:
        raise ManifestBuildError("source audit must contain 1,600 cases")
    historical = source.get("historical_results", {})
    for field in (
        "population_summary_sha256",
        "failure_case_ledger_sha256",
        "accepted_result_payloads_sha256",
        "source_run_contract_sha256",
        "pi05_tree_sha256",
        "pi05_hash_receipt_sha256",
    ):
        _require_sha256(historical.get(field), "historical results " + field)
    if (
        not isinstance(historical.get("source_git_commit"), str)
        or _HEX_GIT_COMMIT.fullmatch(historical["source_git_commit"]) is None
    ):
        raise ManifestBuildError("historical results source_git_commit is invalid")
    if _require_int(
        historical.get("expected_aegis_result_files"),
        "historical AEGIS result files",
    ) != 1600:
        raise ManifestBuildError("historical root must contain 1,600 AEGIS results")
    for field in (
        "root_identifier",
        "aegis_result_relative_glob",
        "aegis_result_relative_pattern",
        "result_schema_version",
        "accepted_status",
        "source_run_contract_relative_path",
        "pi05_hash_receipt_schema_version",
        "population_summary_relative_path",
        "failure_case_ledger_relative_path",
    ):
        if not isinstance(historical.get(field), str) or not historical[field]:
            raise ManifestBuildError("historical results {} is invalid".format(field))
    if Path(historical["population_summary_relative_path"]).is_absolute():
        raise ManifestBuildError("historical population summary path must be relative")
    if Path(historical["source_run_contract_relative_path"]).is_absolute():
        raise ManifestBuildError("historical run contract path must be relative")
    if Path(historical["failure_case_ledger_relative_path"]).is_absolute():
        raise ManifestBuildError("historical failure ledger path must be relative")
    if Path(historical["aegis_result_relative_glob"]).is_absolute():
        raise ManifestBuildError("historical result glob must be relative")
    if "{case_id}" not in historical["aegis_result_relative_pattern"]:
        raise ManifestBuildError("historical result pattern must contain {case_id}")

    population = config.get("population", {})
    if _require_int(population.get("expected_cases"), "target cases") != 109:
        raise ManifestBuildError("target population must contain 109 cases")
    if _require_int(
        population.get("expected_aegis_task_failures"), "target task failures"
    ) != 62:
        raise ManifestBuildError("target population must contain 62 AEGIS task failures")
    disclosure = population.get("outcome_conditioned_disclosure", {})
    if disclosure.get("outcome_conditioned") is not True:
        raise ManifestBuildError("outcome-conditioned population must be disclosed")
    if disclosure.get("selection_uses_aegis_outcomes") is not True:
        raise ManifestBuildError("outcome selection must be disclosed")

    strata = population.get("strata", {})
    if set(strata) != {
        "primary_static_positive_aegis_barrier",
        "support_contact_after_settling_stress",
        "nonpositive_aegis_barrier_recovery_stress",
    }:
        raise ManifestBuildError("unexpected stratum registry")
    if sum(_require_int(row.get("expected_cases"), key) for key, row in strata.items()) != 109:
        raise ManifestBuildError("stratum case counts must sum to 109")
    if sum(
        _require_int(row.get("expected_aegis_task_failures"), key)
        for key, row in strata.items()
    ) != 62:
        raise ManifestBuildError("stratum task failures must sum to 62")

    split = config.get("split", {})
    expected_split_counts = {
        "bringup_canary": 5,
        "parameter_freeze": 20,
        "heldout_evaluation": 84,
    }
    for split_name, expected in expected_split_counts.items():
        split_contract = split.get(split_name, {})
        actual = _require_int(split_contract.get("expected_cases"), split_name)
        if actual != expected:
            raise ManifestBuildError(
                "{} split must contain {} cases".format(split_name, expected)
            )
        split_failures = _require_int(
            split_contract.get("expected_aegis_task_failures"),
            split_name + " task failures",
        )
        expected_strata = split_contract.get("expected_strata", {})
        expected_suites = split_contract.get("expected_suites", {})
        if set(expected_strata) != set(strata):
            raise ManifestBuildError("{} has an invalid stratum distribution".format(split_name))
        if sum(_require_int(value, split_name + " stratum") for value in expected_strata.values()) != expected:
            raise ManifestBuildError("{} stratum counts do not sum to its size".format(split_name))
        if set(expected_suites) != {
            "safelibero_spatial",
            "safelibero_goal",
            "safelibero_object",
            "safelibero_long",
        }:
            raise ManifestBuildError("{} has an invalid suite distribution".format(split_name))
        if sum(_require_int(value, split_name + " suite") for value in expected_suites.values()) != expected:
            raise ManifestBuildError("{} suite counts do not sum to its size".format(split_name))
        if split_failures < 0 or split_failures > expected:
            raise ManifestBuildError("{} task failures are invalid".format(split_name))
    if _require_int(
        split.get("development_pool", {}).get("expected_cases"),
        "development pool",
    ) != 25:
        raise ManifestBuildError("development pool must contain 25 cases")

    max_steps = config.get("execution", {}).get("max_steps_by_suite", {})
    expected_horizons = {
        "safelibero_spatial": 300,
        "safelibero_goal": 300,
        "safelibero_object": 300,
        "safelibero_long": 550,
    }
    if max_steps != expected_horizons:
        raise ManifestBuildError("suite horizons must preserve 300/300/300/550")

    arms = config.get("required_arms")
    expected_arms = [
        "joint_velocity_adapter_only",
        "joint_velocity_psf_link56",
    ]
    if arms != expected_arms:
        raise ManifestBuildError(
            "the first causal study requires exactly the matched adapter-only "
            "and link56 Poisson arms"
        )
    expected_results = _require_int(
        config.get("result_contract", {}).get("expected_results"),
        "expected results",
    )
    if expected_results != 109 * len(arms):
        raise ManifestBuildError("result count must equal cases times arms")
    return config, sha256_bytes(raw)


def _check_file_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_path(path)
    if actual != expected:
        raise ManifestBuildError(
            "{} SHA-256 mismatch: expected {}, got {}".format(label, expected, actual)
        )


def load_source_population(
    path: Path, config: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    source = config["source"]
    _check_file_hash(path, source["table1_manifest_sha256"], "source manifest")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.endswith("\n"):
                raise ManifestBuildError(
                    "source manifest line {} lacks a terminal newline".format(line_number)
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ManifestBuildError(
                    "invalid source manifest JSON on line {}".format(line_number)
                ) from error
            if not isinstance(row, dict):
                raise ManifestBuildError("source manifest rows must be objects")
            rows.append(row)

    expected_rows = int(source["table1_manifest_rows"])
    if len(rows) != expected_rows:
        raise ManifestBuildError(
            "source manifest has {} rows, expected {}".format(len(rows), expected_rows)
        )
    case_ids = [row.get("case_id") for row in rows]
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise ManifestBuildError("source manifest contains an invalid case_id")
    if len(set(case_ids)) != len(case_ids):
        raise ManifestBuildError("source manifest case IDs are not unique")
    if [row.get("case_ordinal") for row in rows] != list(range(expected_rows)):
        raise ManifestBuildError("source manifest ordinals are not exact and contiguous")
    if any(
        row.get("schema_version") != source["table1_manifest_schema_version"]
        for row in rows
    ):
        raise ManifestBuildError("source manifest schema mismatch")
    if any(row.get("protocol_id") != source["table1_protocol_id"] for row in rows):
        raise ManifestBuildError("source manifest protocol mismatch")
    return rows


def load_source_audit(
    path: Path, config: Mapping[str, Any]
) -> List[Dict[str, str]]:
    source = config["source"]
    _check_file_hash(path, source["case_audit_sha256"], "source audit")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ManifestBuildError("source audit has no CSV header")
        missing = sorted(_REQUIRED_AUDIT_COLUMNS - set(reader.fieldnames))
        if missing:
            raise ManifestBuildError(
                "source audit is missing columns: {}".format(", ".join(missing))
            )
        rows = [dict(row) for row in reader]

    expected_rows = int(source["case_audit_rows"])
    if len(rows) != expected_rows:
        raise ManifestBuildError(
            "source audit has {} rows, expected {}".format(len(rows), expected_rows)
        )
    case_ids = [row["case_id"] for row in rows]
    if any(not case_id for case_id in case_ids):
        raise ManifestBuildError("source audit contains an empty case_id")
    if len(set(case_ids)) != len(case_ids):
        raise ManifestBuildError("source audit case IDs are not unique")
    return rows


def _load_json_object(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ManifestBuildError("{} is missing or symlinked: {}".format(label, path))
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestBuildError("{} is not valid JSON".format(label)) from error
    if not isinstance(value, dict):
        raise ManifestBuildError("{} must contain one JSON object".format(label))
    return value


def load_historical_result_index(
    root: Path,
    config: Mapping[str, Any],
    source_case_ids: Set[str],
) -> Dict[str, Path]:
    """Validate the complete historical root and index its AEGIS results."""

    if root.is_symlink() or not root.is_dir():
        raise ManifestBuildError("historical result root is missing or symlinked")
    historical = config["source"]["historical_results"]

    run_contract_path = root / historical["source_run_contract_relative_path"]
    _check_file_hash(
        run_contract_path,
        historical["source_run_contract_sha256"],
        "historical source run contract",
    )
    try:
        run_contract_rows = [
            line.split("\t", 1)
            for line in run_contract_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, UnicodeDecodeError) as error:
        raise ManifestBuildError("historical source run contract is unreadable") from error
    if any(len(row) != 2 for row in run_contract_rows):
        raise ManifestBuildError("historical source run contract is not two-column TSV")
    run_contract = dict(run_contract_rows)
    if len(run_contract) != len(run_contract_rows):
        raise ManifestBuildError("historical source run contract has duplicate keys")
    expected_run_contract = {
        "schema_version": "vlsa_table1_run_contract.v1",
        "run_id": config["source"]["table1_run_id"],
        "git_commit": historical["source_git_commit"],
        "pi05_tree_sha256": historical["pi05_tree_sha256"],
        "pi05_hash_receipt_sha256": historical["pi05_hash_receipt_sha256"],
    }
    for key, expected_value in expected_run_contract.items():
        if run_contract.get(key) != expected_value:
            raise ManifestBuildError(
                "historical source run contract {} differs".format(key)
            )

    summary_path = root / historical["population_summary_relative_path"]
    _check_file_hash(
        summary_path,
        historical["population_summary_sha256"],
        "historical population summary",
    )
    summary = _load_json_object(summary_path, "historical population summary")
    if summary.get("schema_version") != "vlsa_table1_population_summary.v1":
        raise ManifestBuildError("historical population summary schema mismatch")
    if summary.get("protocol_id") != config["source"]["table1_protocol_id"]:
        raise ManifestBuildError("historical population summary protocol mismatch")
    if summary.get("status") != "complete_population_validated":
        raise ManifestBuildError("historical population summary is not complete")
    if (
        summary.get("accepted_result_payloads_sha256")
        != historical["accepted_result_payloads_sha256"]
    ):
        raise ManifestBuildError("accepted-result payload ledger SHA-256 mismatch")
    expected_population = {
        "arms": 2,
        "cases": 1600,
        "no_results_dropped": True,
        "results": 3200,
        "task_level_groups": 32,
    }
    if summary.get("population") != expected_population:
        raise ManifestBuildError("historical population summary is not the complete population")

    failure_ledger_path = root / historical["failure_case_ledger_relative_path"]
    _check_file_hash(
        failure_ledger_path,
        historical["failure_case_ledger_sha256"],
        "historical failure case ledger",
    )

    paths = sorted(root.glob(historical["aegis_result_relative_glob"]))
    expected = int(historical["expected_aegis_result_files"])
    if len(paths) != expected:
        raise ManifestBuildError(
            "historical root has {} AEGIS result files, expected {}".format(
                len(paths), expected
            )
        )
    by_case: Dict[str, Path] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ManifestBuildError("historical result is missing or symlinked: {}".format(path))
        case_id = path.parent.name
        if case_id in by_case:
            raise ManifestBuildError(
                "historical root has multiple AEGIS results for {}".format(case_id)
            )
        by_case[case_id] = path
    if set(by_case) != source_case_ids:
        missing = sorted(source_case_ids - set(by_case))
        extra = sorted(set(by_case) - source_case_ids)
        raise ManifestBuildError(
            "historical AEGIS results do not bind one-to-one to the source population; "
            "missing={}, extra={}".format(missing[:5], extra[:5])
        )
    return by_case


def _historical_result_payload_sha256(result: Mapping[str, Any]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                key: value
                for key, value in result.items()
                if key != "result_payload_sha256" and not str(key).startswith("_")
            }
        )
    )


def validate_historical_aegis_result(
    *,
    path: Path,
    root: Path,
    source_row: Mapping[str, Any],
    audit_row: Mapping[str, str],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate one historical result and return its portable binding."""

    case_id = source_row["case_id"]
    raw = path.read_bytes()
    raw_file_sha256 = sha256_bytes(raw)
    try:
        result = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestBuildError(
            "historical AEGIS result is invalid JSON for {}".format(case_id)
        ) from error
    if not isinstance(result, dict):
        raise ManifestBuildError("historical AEGIS result is not an object for {}".format(case_id))

    historical = config["source"]["historical_results"]
    if result.get("schema_version") != historical["result_schema_version"]:
        raise ManifestBuildError("{} historical result schema mismatch".format(case_id))
    if result.get("protocol_id") != config["source"]["table1_protocol_id"]:
        raise ManifestBuildError("{} historical result protocol mismatch".format(case_id))
    if result.get("case_id") != case_id or result.get("case_ordinal") != source_row["case_ordinal"]:
        raise ManifestBuildError("{} historical result identity mismatch".format(case_id))
    if result.get("arm") != config["source"]["source_arm"] or result.get("mode") != "aegis":
        raise ManifestBuildError("{} is not the registered historical AEGIS arm".format(case_id))
    if result.get("status") != historical["accepted_status"]:
        raise ManifestBuildError("{} historical result is not complete".format(case_id))
    if result.get("scientific_result") is not True:
        raise ManifestBuildError("{} historical result is not scientific".format(case_id))
    if result.get("case") != source_row:
        raise ManifestBuildError("{} historical result embeds a different manifest row".format(case_id))

    payload_sha256 = _require_sha256(
        result.get("result_payload_sha256"), case_id + " result payload"
    )
    if _historical_result_payload_sha256(result) != payload_sha256:
        raise ManifestBuildError("{} historical result payload hash mismatch".format(case_id))

    source = result.get("source")
    if not isinstance(source, dict):
        raise ManifestBuildError("{} historical result source is invalid".format(case_id))
    git = source.get("git")
    if (
        not isinstance(git, dict)
        or git.get("commit") != historical["source_git_commit"]
        or git.get("dirty") is not False
    ):
        raise ManifestBuildError("{} historical result Git binding mismatch".format(case_id))
    if source.get("manifest_source_commit") != source_row["source_commit"]:
        raise ManifestBuildError("{} historical source-manifest commit mismatch".format(case_id))

    ledger = result.get("action_invariance_ledger")
    actions = result.get("actions")
    metrics = result.get("metrics")
    if not isinstance(ledger, dict) or not isinstance(actions, list) or not isinstance(metrics, dict):
        raise ManifestBuildError("{} historical action ledger is invalid".format(case_id))
    action_count = _require_int(ledger.get("action_count"), case_id + " action count")
    if action_count != len(actions) or metrics.get("executed_action_count") != action_count:
        raise ManifestBuildError("{} historical action count mismatch".format(case_id))
    executed_sequence_sha256 = _require_sha256(
        ledger.get("executed_sequence_sha256"), case_id + " executed sequence"
    )
    recomputed_executed_sequence_sha256 = sha256_bytes(
        canonical_json_bytes([action.get("executed") for action in actions])
    )
    if executed_sequence_sha256 != recomputed_executed_sequence_sha256:
        raise ManifestBuildError("{} historical executed-action ledger is stale".format(case_id))
    historical_task_success = _parse_bool(
        audit_row["aegis_task_success"], case_id + " audit task success"
    )
    if (
        result.get("task_success") != historical_task_success
        or metrics.get("task_success") != historical_task_success
    ):
        raise ManifestBuildError("{} historical task outcome mismatch".format(case_id))
    if metrics.get("paper_collision") is not True:
        raise ManifestBuildError("{} historical result does not establish CAR failure".format(case_id))
    audit_collision_first_step = _parse_optional_int(
        audit_row["collision_first_step"], case_id + " audit collision first step"
    )
    if metrics.get("collision_first_step") != audit_collision_first_step:
        raise ManifestBuildError("{} historical collision step mismatch".format(case_id))

    contact_telemetry = result.get("contact_telemetry")
    if (
        not isinstance(contact_telemetry, dict)
        or contact_telemetry.get("status") != "available"
        or contact_telemetry.get("robot_active_obstacle_contact") is not True
    ):
        raise ManifestBuildError("{} historical robot-contact telemetry is invalid".format(case_id))
    unique_pairs = contact_telemetry.get("unique_contact_pairs")
    audited_links = set(literal_contact_bodies(audit_row["robot_contact_bodies"])) & {
        "robot0_link5",
        "robot0_link6",
    }
    direct_contact_evidence = matched_direct_contact_pair_evidence(
        unique_pairs=unique_pairs,
        active_obstacle_name=audit_row["active_obstacle_name"],
        audited_links=audited_links,
        case_id=case_id,
    )

    pairing = result.get("pairing")
    if not isinstance(pairing, dict):
        raise ManifestBuildError("{} historical pairing record is invalid".format(case_id))
    pairing_hash_fields = (
        "policy_noise_schedule_sha256",
        "settled_simulator_state_sha256",
        "initial_state_sha256",
        "initial_observation_sha256",
        "manifest_row_sha256",
        "semantic_label_record_sha256",
    )
    pairing_hashes = {
        field: _require_sha256(pairing.get(field), "{} pairing {}".format(case_id, field))
        for field in pairing_hash_fields
    }
    expected_manifest_row_sha256 = sha256_bytes(canonical_json_bytes(source_row))
    if pairing_hashes["manifest_row_sha256"] != expected_manifest_row_sha256:
        raise ManifestBuildError("{} pairing manifest-row hash mismatch".format(case_id))
    if pairing.get("policy_noise_schedule_id") != source_row["policy_noise_schedule_id"]:
        raise ManifestBuildError("{} pairing policy-noise schedule mismatch".format(case_id))
    policy_noise_schedule = pairing.get("policy_noise_schedule")
    if (
        not isinstance(policy_noise_schedule, dict)
        or sha256_bytes(canonical_json_bytes(policy_noise_schedule))
        != pairing_hashes["policy_noise_schedule_sha256"]
    ):
        raise ManifestBuildError("{} policy-noise schedule hash mismatch".format(case_id))
    if (
        pairing.get("max_steps") != source_row["max_steps"]
        or pairing.get("model_action_horizon") != source_row["model_action_horizon"]
        or pairing.get("replan_steps") != source_row["replan_steps"]
    ):
        raise ManifestBuildError("{} historical pairing execution mismatch".format(case_id))
    if pairing.get("semantic_obstacle_label") != audit_row["frozen_obstacle_label"]:
        raise ManifestBuildError("{} historical semantic label mismatch".format(case_id))
    initial_observation_contract = pairing.get("initial_observation_contract")
    if (
        not isinstance(initial_observation_contract, dict)
        or initial_observation_contract.get("active_obstacle_name")
        != audit_row["active_obstacle_name"]
    ):
        raise ManifestBuildError("{} historical active-obstacle binding mismatch".format(case_id))
    if (
        sha256_bytes(canonical_json_bytes(initial_observation_contract))
        != pairing_hashes["initial_observation_sha256"]
    ):
        raise ManifestBuildError("{} initial-observation hash mismatch".format(case_id))
    if (
        initial_observation_contract.get("settled_simulator_state_array_sha256")
        != pairing_hashes["settled_simulator_state_sha256"]
    ):
        raise ManifestBuildError("{} settled simulator-state hash mismatch".format(case_id))

    relative_path = path.relative_to(root).as_posix()
    registered_pattern = historical["aegis_result_relative_pattern"].format(
        case_id=case_id
    )
    return {
        "source_identifier": "{}::aegis::{}".format(
            historical["root_identifier"], case_id
        ),
        "source_relative_path": relative_path,
        "source_relative_pattern": registered_pattern,
        "result_payload_sha256": payload_sha256,
        "raw_file_sha256": raw_file_sha256,
        "action_count": action_count,
        "action_invariance_ledger": {
            "action_count": action_count,
            "executed_sequence_sha256": executed_sequence_sha256,
        },
        "direct_protected_link_active_obstacle_contact": {
            "source_field": "contact_telemetry.unique_contact_pairs",
            "active_obstacle_name": audit_row["active_obstacle_name"],
            "required_protected_link_body_names": sorted(audited_links),
            "matched_pairs": direct_contact_evidence,
            "matched_pairs_sha256": sha256_bytes(
                canonical_json_bytes(direct_contact_evidence)
            ),
        },
        "pairing": {
            "policy_noise_schedule_sha256": pairing_hashes[
                "policy_noise_schedule_sha256"
            ],
            "settled_simulator_state_sha256": pairing_hashes[
                "settled_simulator_state_sha256"
            ],
            "initial_state_sha256": pairing_hashes["initial_state_sha256"],
            "initial_observation_sha256": pairing_hashes[
                "initial_observation_sha256"
            ],
            "manifest_row_sha256": pairing_hashes["manifest_row_sha256"],
            "semantic_label_record_sha256": pairing_hashes[
                "semantic_label_record_sha256"
            ],
        },
        "semantic_label_record_sha256": pairing_hashes[
            "semantic_label_record_sha256"
        ],
    }


def validate_source_join(
    source_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, str]],
) -> Dict[str, Mapping[str, str]]:
    """Validate and return the exact one-to-one audit join by case ID."""

    source_by_id: Dict[str, Mapping[str, Any]] = {}
    for row in source_rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ManifestBuildError("source join contains an invalid case_id")
        if case_id in source_by_id:
            raise ManifestBuildError("duplicate source case_id in join: {}".format(case_id))
        source_by_id[case_id] = row

    audit_by_id: Dict[str, Mapping[str, str]] = {}
    for row in audit_rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ManifestBuildError("audit join contains an invalid case_id")
        if case_id in audit_by_id:
            raise ManifestBuildError("duplicate audit case_id in join: {}".format(case_id))
        audit_by_id[case_id] = row

    source_ids = set(source_by_id)
    audit_ids = set(audit_by_id)
    if source_ids != audit_ids:
        missing_audit = sorted(source_ids - audit_ids)
        extra_audit = sorted(audit_ids - source_ids)
        raise ManifestBuildError(
            "source/audit case join is not one-to-one; missing audit={}, extra audit={}".format(
                missing_audit[:5], extra_audit[:5]
            )
        )

    for case_id in sorted(source_ids):
        source_row = source_by_id[case_id]
        audit_row = audit_by_id[case_id]
        for field in _JOIN_IDENTITY_FIELDS:
            source_value = source_row.get(field)
            audit_value: Any = audit_row.get(field)
            if field == "episode_index":
                try:
                    audit_value = int(str(audit_value))
                except ValueError as error:
                    raise ManifestBuildError(
                        "{} has invalid audit episode_index".format(case_id)
                    ) from error
            if source_value != audit_value:
                raise ManifestBuildError(
                    "{} identity mismatch for {}: source={!r}, audit={!r}".format(
                        case_id, field, source_value, audit_value
                    )
                )
    return audit_by_id


def _classify_stratum(row: Mapping[str, str]) -> str:
    case_id = row["case_id"]
    settled = _parse_bool(
        row["aegis_settled_relevant_contact"],
        "{} aegis_settled_relevant_contact".format(case_id),
    )
    barrier_h = _parse_optional_float(
        row["barrier_h_at_first_relevant_contact"],
        "{} barrier_h_at_first_relevant_contact".format(case_id),
    )
    if settled:
        return "support_contact_after_settling_stress"
    if barrier_h is None:
        raise ManifestBuildError(
            "{} has no settled contact and no first-contact barrier".format(case_id)
        )
    if barrier_h > 0.0:
        return "primary_static_positive_aegis_barrier"
    return "nonpositive_aegis_barrier_recovery_stress"


def _development_family_registry(config: Mapping[str, Any]) -> Set[Tuple[str, str]]:
    registry: Set[Tuple[str, str]] = set()
    for entry in config["split"]["development_pool"]["task_families"]:
        key = (entry.get("suite"), entry.get("task_name"))
        if not all(isinstance(item, str) and item for item in key):
            raise ManifestBuildError("development task family is invalid")
        if key in registry:
            raise ManifestBuildError("duplicate development task family")
        registry.add(key)  # type: ignore[arg-type]
    return registry


def _selected_audit_row(
    row: Mapping[str, str], config: Mapping[str, Any]
) -> bool:
    rule = config["population"]["selection_rule"]
    bodies = set(literal_contact_bodies(row["robot_contact_bodies"]))
    protected = set(rule["literal_robot_contact_body_any_of"])
    return (
        _parse_bool(row["aegis_collision"], row["case_id"] + " aegis_collision")
        == rule["aegis_collision"]
        and _parse_bool(
            row["aegis_postcontrol_robot_contact"],
            row["case_id"] + " aegis_postcontrol_robot_contact",
        )
        == rule["aegis_postcontrol_robot_contact"]
        and bool(bodies & protected)
    )


def build_rows(
    *,
    config_path: Path,
    source_population_path: Path,
    source_audit_path: Path,
    historical_result_root: Path,
) -> List[Dict[str, Any]]:
    config, config_sha256 = load_config(config_path)
    source_rows = load_source_population(source_population_path, config)
    audit_rows = load_source_audit(source_audit_path, config)
    audit_by_id = validate_source_join(source_rows, audit_rows)
    historical_result_by_id = load_historical_result_index(
        historical_result_root,
        config,
        {row["case_id"] for row in source_rows},
    )

    development_families = _development_family_registry(config)
    canary_ids = set(config["split"]["bringup_canary"]["case_ids"])
    if len(canary_ids) != 5:
        raise ManifestBuildError("bring-up canary IDs must be five distinct cases")
    disclosure = config["population"]["outcome_conditioned_disclosure"]
    protected_bodies = set(
        config["population"]["selection_rule"]["literal_robot_contact_body_any_of"]
    )

    rows: List[Dict[str, Any]] = []
    family_counts: Dict[Tuple[str, str], int] = {}
    for source_row in source_rows:
        audit_row = audit_by_id[source_row["case_id"]]
        if not _selected_audit_row(audit_row, config):
            continue

        case_id = source_row["case_id"]
        bodies = literal_contact_bodies(audit_row["robot_contact_bodies"])
        if not (set(bodies) & protected_bodies):
            raise ManifestBuildError("{} failed literal link selection".format(case_id))
        for field in ("aegis_geometry_complete", "aegis_all_actions_solved_qp"):
            if not _parse_bool(audit_row[field], "{} {}".format(case_id, field)):
                raise ManifestBuildError("{} selected row has false {}".format(case_id, field))

        stratum = _classify_stratum(audit_row)
        stratum_contract = config["population"]["strata"][stratum]
        if audit_row["observed_car_class"] != stratum_contract["observed_car_class"]:
            raise ManifestBuildError("{} observed CAR class disagrees with stratum".format(case_id))
        if (
            audit_row["refined_collision_mechanism"]
            != stratum_contract["refined_collision_mechanism"]
        ):
            raise ManifestBuildError("{} mechanism label disagrees with stratum".format(case_id))

        suite = source_row["suite"]
        task_name = source_row["task_name"]
        family_key = (suite, task_name)
        family_counts[family_key] = family_counts.get(family_key, 0) + 1
        in_development = family_key in development_families
        if case_id in canary_ids:
            if not in_development:
                raise ManifestBuildError("{} canary is outside development families".format(case_id))
            split_name = "bringup_canary"
        elif in_development:
            split_name = "parameter_freeze"
        else:
            split_name = "heldout_evaluation"

        expected_max_steps = int(config["execution"]["max_steps_by_suite"][suite])
        if source_row["max_steps"] != expected_max_steps:
            raise ManifestBuildError("{} has an invalid max_steps".format(case_id))
        if source_row["settle_actions"] != config["execution"]["settle_actions"]:
            raise ManifestBuildError("{} has an invalid settle action count".format(case_id))

        barrier_h = _parse_optional_float(
            audit_row["barrier_h_at_first_relevant_contact"],
            case_id + " barrier_h_at_first_relevant_contact",
        )
        constraint_lhs = _parse_optional_float(
            audit_row["constraint_lhs_at_first_relevant_contact"],
            case_id + " constraint_lhs_at_first_relevant_contact",
        )
        car_h = _parse_optional_float(
            audit_row["barrier_h_at_car_crossing"],
            case_id + " barrier_h_at_car_crossing",
        )
        task_success = _parse_bool(
            audit_row["aegis_task_success"], case_id + " aegis_task_success"
        )
        settled = _parse_bool(
            audit_row["aegis_settled_relevant_contact"],
            case_id + " aegis_settled_relevant_contact",
        )
        historical_result = validate_historical_aegis_result(
            path=historical_result_by_id[case_id],
            root=historical_result_root,
            source_row=source_row,
            audit_row=audit_row,
            config=config,
        )

        row = {
            "schema_version": MANIFEST_SCHEMA,
            "protocol_id": config["protocol_id"],
            "protocol_config_sha256": config_sha256,
            "case_ordinal": len(rows),
            "case_id": case_id,
            "pair_group_id": "vlsa-poisson-link56:" + case_id,
            "source_case_ordinal": source_row["case_ordinal"],
            "source_protocol_id": source_row["protocol_id"],
            "source_protocol_config_sha256": source_row["protocol_config_sha256"],
            "source_commit": source_row["source_commit"],
            "source_manifest_sha256": config["source"]["table1_manifest_sha256"],
            "source_manifest_row_sha256": sha256_bytes(canonical_json_bytes(source_row)),
            "source_audit_sha256": config["source"]["case_audit_sha256"],
            "source_audit_row_sha256": sha256_bytes(canonical_json_bytes(audit_row)),
            "source_table1_run_id": config["source"]["table1_run_id"],
            "historical_aegis_result": historical_result,
            "suite": suite,
            "safety_level": source_row["safety_level"],
            "logical_task_index": source_row["logical_task_index"],
            "resolved_task_index": source_row["resolved_task_index"],
            "task_name": task_name,
            "task_level_group_id": source_row["task_level_group_id"],
            "task_family_id": semantic_task_family_id(suite, task_name),
            "episode_index": source_row["episode_index"],
            "environment_seed": source_row["environment_seed"],
            "policy_noise_seed": source_row["policy_noise_seed"],
            "policy_noise_schedule_id": source_row["policy_noise_schedule_id"],
            "model_action_horizon": source_row["model_action_horizon"],
            "replan_steps": source_row["replan_steps"],
            "settle_actions": source_row["settle_actions"],
            "max_steps": source_row["max_steps"],
            "source_policy_action_space": source_row["action_space"],
            "active_execution_action_space": "joint_velocity_7d_plus_gripper",
            "required_arms": list(config["required_arms"]),
            "bddl_path": source_row["bddl_path"],
            "bddl_sha256": source_row["bddl_sha256"],
            "initial_states_path": source_row["initial_states_path"],
            "initial_states_sha256": source_row["initial_states_sha256"],
            "semantic_label_requirement": source_row["semantic_label_requirement"],
            "split": split_name,
            "study_partition": (
                "development" if in_development else "heldout_evaluation"
            ),
            "stratum": stratum,
            "active_obstacle_name": audit_row["active_obstacle_name"],
            "frozen_obstacle_label": audit_row["frozen_obstacle_label"],
            "selection_evidence": {
                "source_arm": config["source"]["source_arm"],
                "aegis_collision": True,
                "aegis_task_success": task_success,
                "aegis_task_failure": not task_success,
                "aegis_settled_relevant_contact": settled,
                "aegis_postcontrol_robot_contact": True,
                "aegis_geometry_complete": True,
                "aegis_all_actions_solved_qp": True,
                "literal_robot_contact_bodies": list(bodies),
                "literal_protected_link_contact_bodies": sorted(
                    set(bodies) & protected_bodies
                ),
                "observed_car_class": audit_row["observed_car_class"],
                "refined_collision_mechanism": audit_row[
                    "refined_collision_mechanism"
                ],
                "collision_first_step": _parse_optional_int(
                    audit_row["collision_first_step"], case_id + " collision_first_step"
                ),
                "first_relevant_contact_step": _parse_optional_int(
                    audit_row["first_relevant_contact_step"],
                    case_id + " first_relevant_contact_step",
                ),
                "released_aegis_barrier_h_at_first_relevant_contact": barrier_h,
                "released_aegis_constraint_lhs_at_first_relevant_contact": constraint_lhs,
                "released_aegis_barrier_h_at_car_crossing": car_h,
            },
            "outcome_conditioned": True,
            "population_kind": disclosure["population_kind"],
            "claim_scope": disclosure["claim_scope"],
            "prohibited_interpretation": disclosure["prohibited_interpretation"],
        }
        rows.append(row)

    _validate_generated_rows(rows, config, family_counts, development_families, canary_ids)
    return rows


def _validate_generated_rows(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    family_counts: Mapping[Tuple[str, str], int],
    development_families: Set[Tuple[str, str]],
    canary_ids: Set[str],
) -> None:
    expected_cases = int(config["population"]["expected_cases"])
    if len(rows) != expected_cases:
        raise ManifestBuildError(
            "selected {} cases, expected {}".format(len(rows), expected_cases)
        )
    case_ids = [row["case_id"] for row in rows]
    if len(set(case_ids)) != expected_cases:
        raise ManifestBuildError("target case IDs are not unique")
    if set(case_ids) & canary_ids != canary_ids:
        raise ManifestBuildError("one or more registered canaries were not selected")

    strata = config["population"]["strata"]
    for stratum, contract in strata.items():
        selected = [row for row in rows if row["stratum"] == stratum]
        if len(selected) != int(contract["expected_cases"]):
            raise ManifestBuildError(
                "{} has {} cases, expected {}".format(
                    stratum, len(selected), contract["expected_cases"]
                )
            )
        failures = sum(row["selection_evidence"]["aegis_task_failure"] for row in selected)
        if failures != int(contract["expected_aegis_task_failures"]):
            raise ManifestBuildError(
                "{} has {} task failures, expected {}".format(
                    stratum, failures, contract["expected_aegis_task_failures"]
                )
            )

    total_failures = sum(row["selection_evidence"]["aegis_task_failure"] for row in rows)
    if total_failures != int(config["population"]["expected_aegis_task_failures"]):
        raise ManifestBuildError("target AEGIS task-failure total is invalid")

    for split_name in ("bringup_canary", "parameter_freeze", "heldout_evaluation"):
        split_rows = [row for row in rows if row["split"] == split_name]
        count = len(split_rows)
        split_contract = config["split"][split_name]
        expected = int(split_contract["expected_cases"])
        if count != expected:
            raise ManifestBuildError(
                "{} has {} cases, expected {}".format(split_name, count, expected)
            )
        failures = sum(
            row["selection_evidence"]["aegis_task_failure"] for row in split_rows
        )
        if failures != int(split_contract["expected_aegis_task_failures"]):
            raise ManifestBuildError("{} task-failure distribution is invalid".format(split_name))
        actual_strata = {
            name: sum(row["stratum"] == name for row in split_rows)
            for name in config["population"]["strata"]
        }
        if actual_strata != split_contract["expected_strata"]:
            raise ManifestBuildError("{} stratum distribution is invalid".format(split_name))
        actual_suites = {
            name: sum(row["suite"] == name for row in split_rows)
            for name in config["execution"]["max_steps_by_suite"]
        }
        if actual_suites != split_contract["expected_suites"]:
            raise ManifestBuildError("{} suite distribution is invalid".format(split_name))

    development_rows = [row for row in rows if row["study_partition"] == "development"]
    if len(development_rows) != int(config["split"]["development_pool"]["expected_cases"]):
        raise ManifestBuildError("development pool size is invalid")

    registered_expected: Dict[Tuple[str, str], int] = {}
    for entry in config["split"]["development_pool"]["task_families"]:
        registered_expected[(entry["suite"], entry["task_name"])] = int(
            entry["expected_cases"]
        )
    if set(registered_expected) != development_families:
        raise ManifestBuildError("development family registry is inconsistent")
    for key, expected in registered_expected.items():
        if family_counts.get(key, 0) != expected:
            raise ManifestBuildError(
                "development family {!r} has {}, expected {}".format(
                    key, family_counts.get(key, 0), expected
                )
            )

    family_partitions: Dict[str, Set[str]] = {}
    for row in rows:
        family_partitions.setdefault(row["task_family_id"], set()).add(
            row["study_partition"]
        )
    leaked = sorted(family for family, parts in family_partitions.items() if len(parts) != 1)
    if leaked:
        raise ManifestBuildError(
            "semantic task families cross development/evaluation: {}".format(leaked)
        )


def manifest_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def build_receipt(
    *,
    config_path: Path,
    source_population_path: Path,
    source_audit_path: Path,
    rows: Sequence[Mapping[str, Any]],
    payload: bytes,
) -> Dict[str, Any]:
    config, config_sha256 = load_config(config_path)
    split_counts = {
        name: sum(row["split"] == name for row in rows)
        for name in ("bringup_canary", "parameter_freeze", "heldout_evaluation")
    }
    split_task_failures = {
        name: sum(
            row["split"] == name
            and row["selection_evidence"]["aegis_task_failure"]
            for row in rows
        )
        for name in ("bringup_canary", "parameter_freeze", "heldout_evaluation")
    }
    split_stratum_counts = {
        split_name: {
            stratum: sum(
                row["split"] == split_name and row["stratum"] == stratum
                for row in rows
            )
            for stratum in config["population"]["strata"]
        }
        for split_name in ("bringup_canary", "parameter_freeze", "heldout_evaluation")
    }
    split_suite_counts = {
        split_name: {
            suite: sum(
                row["split"] == split_name and row["suite"] == suite for row in rows
            )
            for suite in config["execution"]["max_steps_by_suite"]
        }
        for split_name in ("bringup_canary", "parameter_freeze", "heldout_evaluation")
    }
    stratum_counts = {
        name: sum(row["stratum"] == name for row in rows)
        for name in config["population"]["strata"]
    }
    stratum_task_failures = {
        name: sum(
            row["stratum"] == name
            and row["selection_evidence"]["aegis_task_failure"]
            for row in rows
        )
        for name in config["population"]["strata"]
    }
    historical = config["source"]["historical_results"]
    historical_bindings = [
        {
            "case_id": row["case_id"],
            "result_payload_sha256": row["historical_aegis_result"][
                "result_payload_sha256"
            ],
            "raw_file_sha256": row["historical_aegis_result"]["raw_file_sha256"],
            "source_identifier": row["historical_aegis_result"]["source_identifier"],
        }
        for row in rows
    ]
    return {
        "schema_version": RECEIPT_SCHEMA,
        "protocol_id": config["protocol_id"],
        "protocol_config_sha256": config_sha256,
        "manifest_schema_version": MANIFEST_SCHEMA,
        "manifest_rows": len(rows),
        "manifest_sha256": sha256_bytes(payload),
        "first_case_id": rows[0]["case_id"],
        "last_case_id": rows[-1]["case_id"],
        "source_manifest_sha256": sha256_path(source_population_path),
        "source_audit_sha256": sha256_path(source_audit_path),
        "historical_result_root_identifier": historical["root_identifier"],
        "historical_aegis_results_bound": len(historical_bindings),
        "historical_result_bindings_sha256": sha256_bytes(
            canonical_json_bytes(historical_bindings)
        ),
        "population_summary_sha256": historical["population_summary_sha256"],
        "failure_case_ledger_sha256": historical["failure_case_ledger_sha256"],
        "accepted_result_payloads_sha256": historical[
            "accepted_result_payloads_sha256"
        ],
        "source_run_contract_sha256": historical[
            "source_run_contract_sha256"
        ],
        "pi05_tree_sha256": historical["pi05_tree_sha256"],
        "pi05_hash_receipt_sha256": historical[
            "pi05_hash_receipt_sha256"
        ],
        "full_source_join_rows_validated": int(config["source"]["case_audit_rows"]),
        "exact_source_join_rows": len(rows),
        "aegis_task_failures": sum(
            row["selection_evidence"]["aegis_task_failure"] for row in rows
        ),
        "stratum_counts": stratum_counts,
        "stratum_task_failures": stratum_task_failures,
        "split_counts": split_counts,
        "split_task_failures": split_task_failures,
        "split_stratum_counts": split_stratum_counts,
        "split_suite_counts": split_suite_counts,
        "development_pool_cases": sum(
            row["study_partition"] == "development" for row in rows
        ),
        "semantic_task_families": len(
            {row["task_family_id"] for row in rows}
        ),
        "required_arms": list(config["required_arms"]),
        "paired_results_required": sum(len(row["required_arms"]) for row in rows),
        "outcome_conditioned_disclosure": dict(
            config["population"]["outcome_conditioned_disclosure"]
        ),
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary_path), str(path))
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_root / "configs/vlsa_poisson_link56_feasibility.v1.json",
    )
    parser.add_argument("--source-population", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--historical-result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    source_population_path = args.source_population.resolve()
    source_audit_path = args.source_audit.resolve()
    historical_result_root = args.historical_result_root.resolve()
    rows = build_rows(
        config_path=config_path,
        source_population_path=source_population_path,
        source_audit_path=source_audit_path,
        historical_result_root=historical_result_root,
    )
    payload = manifest_bytes(rows)
    receipt = build_receipt(
        config_path=config_path,
        source_population_path=source_population_path,
        source_audit_path=source_audit_path,
        rows=rows,
        payload=payload,
    )
    _atomic_write_bytes(args.output.resolve(), payload)
    _atomic_write_bytes(
        args.receipt_output.resolve(),
        json.dumps(receipt, sort_keys=True, indent=2).encode("utf-8") + b"\n",
    )
    json.dump(receipt, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
