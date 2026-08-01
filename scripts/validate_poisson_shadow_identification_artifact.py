#!/usr/bin/env python3
"""Read-only independent consumer for shadow-identification v3 artifacts.

The producer and this consumer must run in different Slurm allocations.  This
program never publishes or modifies an experiment artifact: its only output is
one typed JSON report on stdout (or one rejection on stderr).  A validated
producer failure is diagnostic-only.  A passed producer authorizes the next
one-step gate only when all immutable inputs, the complete serialized replay,
and an independently reconstructed phase-correct warning are valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EXPECTED_SCHEMA = "vlsa_poisson_shadow_identification.v3"
EXPECTED_NUMERIC_SCHEMA = "vlsa_poisson_numeric_validation.v1"
EXPECTED_PARITY_SCHEMA = "vlsa_poisson_shadow_parity.v2"
EXPECTED_CASE_ID = "vlsa-t1-goal-ii-t0-e05"
CONTACT_DEFINITION = "mujoco_contact_dist_le_0"
ACTION_COUNT = 237
INNER_UPDATES_PER_ACTION = 5
PHYSICS_SUBSTEPS_PER_INNER_UPDATE = 5
CALLBACKS_PER_ACTION = (
    INNER_UPDATES_PER_ACTION * PHYSICS_SUBSTEPS_PER_INNER_UPDATE
)
CALLBACK_COUNT = ACTION_COUNT * CALLBACKS_PER_ACTION
PHYSICS_TIMESTEP_S = 0.002
FILTER_UPDATE_PERIOD = 5
CONSUMER_SCHEMA = "vlsa_poisson_shadow_identification_consumer.v1"
IDENTIFICATION_EVIDENCE_TIER = (
    "allocation_backed_exact_replay_shadow_identification"
)
IDENTIFICATION_CLAIM_LIMIT = (
    "read-only temporal identification only; no action was corrected and "
    "no active safety or utility efficacy was tested"
)
EXPECTED_PRODUCER_JOB_NAME = "vlsa-poisson-identify"
NUMERIC_EVIDENCE_TIER = "allocation_backed_implementation_validation"
NUMERIC_CLAIM_LIMIT = (
    "synthetic and numerical tests do not establish SafeLIBERO safety efficacy"
)
EXPECTED_NUMERIC_JOB_NAME = "vlsa-poisson-numerics"
PARITY_EVIDENCE_TIER = "allocation_backed_exact_simulator_parity"
PARITY_CLAIM_LIMIT = (
    "apparatus validation only; no Poisson correction or safety efficacy was tested"
)
EXPECTED_PARITY_JOB_NAME = "vlsa-poisson-shadow"
REGISTERED_EVALUATION_PYTHON = (
    "/mnt/data/quanth/venvs/safety_vla/main/bin/python"
)
DIFFERENTIAL_FAILURE_SCHEMA = (
    "vlsa_poisson_shadow_identification_differential_audit_failure_evidence.v1"
)
DIFFERENTIAL_FAILURE_PHASE = (
    "settled_link56_protected_sample_differential_audit_validation"
)
DIFFERENTIAL_FAILURE_TYPE = (
    "ShadowIdentificationDifferentialAuditFailure"
)
DIFFERENTIAL_FAILURE_MESSAGE = (
    "settled link-5/6 protected-sample differential audit did not pass"
)
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

PRODUCER_PROVENANCE_FIELDS = frozenset(
    (
        "source",
        "manifest_path",
        "manifest_sha256",
        "manifest_line_number",
        "manifest_row_sha256",
        "selection_config_path",
        "selection_config_sha256",
        "selection_protocol_id",
        "runtime_protocol_path",
        "runtime_protocol_raw_sha256",
        "runtime_protocol_semantic_sha256",
        "runtime_parameter_block_sha256",
        "historical_result_path",
        "historical_result_file_sha256",
        "historical_result_payload_sha256",
        "upstream_parity_path",
        "upstream_parity_payload_sha256",
        "host",
        "slurm_job_id",
        "slurm_job_name",
        "python_executable",
        "python_version",
        "packages",
        "gpu",
    )
)
SHADOW_REPLAY_FIELDS = frozenset(
    (
        "executed_action_count",
        "callback_count",
        "expected_callback_count",
        "action_boundary_state_sha256_ledger",
        "state_sequence_sha256",
        "observation_sequence_sha256",
        "callback_state_read_only_ledger",
        "callback_state_read_only_ledger_sha256",
        "callback_state_sequence_sha256",
        "terminal_simulator_state_sha256",
        "construction",
        "measurement",
        "monitor_static_drift_exception_count",
        "monitor_static_drift_exceptions",
        "poisson_identification",
    )
)
NUMERIC_TOP_FIELDS = frozenset(
    (
        "schema_version",
        "status",
        "scientific_result",
        "evidence_tier",
        "claim_limit",
        "created_at_unix",
        "duration_seconds",
        "source",
        "runtime",
        "tests",
        "acceptance",
        "result_payload_sha256",
    )
)
NUMERIC_RUNTIME_FIELDS = frozenset(
    (
        "python_executable",
        "python_version",
        "host",
        "slurm_job_id",
        "slurm_job_name",
        "slurm_partition",
        "cuda_visible_devices",
        "production_grid_validation_enabled",
        "allocated_gpu_inventory",
        "packages",
    )
)
NUMERIC_TEST_FIELDS = frozenset(
    (
        "modules",
        "minimum_tests",
        "tests_run",
        "failure_count",
        "error_count",
        "skip_count",
        "failures",
        "errors",
        "skipped",
        "output",
    )
)
PARITY_TOP_FIELDS = frozenset(
    (
        "schema_version",
        "status",
        "scientific_result",
        "evidence_tier",
        "claim_limit",
        "case_id",
        "timing",
        "provenance",
        "historical",
        "ordinary_replay",
        "callback_replay",
        "acceptance",
        "result_payload_sha256",
    )
)
PARITY_PROVENANCE_FIELDS = frozenset(
    (
        "source",
        "manifest_path",
        "manifest_sha256",
        "manifest_line_number",
        "manifest_row_sha256",
        "historical_result_path",
        "historical_result_file_sha256",
        "historical_result_payload_sha256",
        "host",
        "slurm_job_id",
        "slurm_job_name",
        "python_executable",
        "python_version",
        "packages",
        "gpu",
    )
)

IDENTIFICATION_ACCEPTANCE_FIELDS = frozenset(
    (
        "upstream_exact_parity_same_clean_commit",
        "all_237_historical_actions_executed",
        "all_5925_callbacks_observed",
        "historical_state_reward_done_goal_exact",
        "upstream_observation_sequence_exact",
        "complete_mujoco_integration_state_unchanged_by_construction",
        "complete_mujoco_integration_state_unchanged_by_callback",
        "all_link56_protected_sample_point_jacobians_validated",
        "all_link56_protected_sample_field_chain_rules_validated",
        "static_queries_stop_at_first_registered_drift",
        "contact_authority_is_mujoco_nonpositive_distance",
        "no_action_or_control_mutation",
        "no_active_safety_efficacy_claim",
    )
)
NUMERIC_ACCEPTANCE_FIELDS = frozenset(
    (
        "unittest_success",
        "zero_skips",
        "minimum_test_count_met",
        "clean_source",
        "inside_slurm_allocation",
        "allocated_gpu_visible",
        "allocated_devices_are_h100",
        "production_grid_validation_enabled",
        "production_grid_test_module_included",
    )
)
PARITY_ACCEPTANCE_FIELDS = frozenset(
    (
        "all_historical_post_step_states_exact",
        "ordinary_and_callback_states_exact",
        "ordinary_and_callback_observations_exact",
        "reward_done_goal_exact",
        "full_callback_exposure",
        "ordinary_step_path_unmodified",
    )
)
REQUIRED_NUMERIC_TEST_MODULE = (
    "tests.test_poisson_shadow_identification_artifact_validator"
)


class ShadowIdentificationArtifactError(RuntimeError):
    """Raised when an input cannot support even a diagnostic interpretation."""


def _canonical(value: Any, *, ensure_ascii: bool = False) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=ensure_ascii,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as error:
        raise ShadowIdentificationArtifactError(
            "artifact contains a non-canonical or non-finite value"
        ) from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ShadowIdentificationArtifactError(
            "cannot hash required file: %s" % path
        ) from error
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ShadowIdentificationArtifactError(
            "%s must be one lowercase SHA-256" % label
        )
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ShadowIdentificationArtifactError("%s must be an object" % label)
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ShadowIdentificationArtifactError(
            "%s must be an integer >= %d" % (label, minimum)
        )
    return int(value)


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ShadowIdentificationArtifactError(
            "%s must be a finite number" % label
        )
    return float(value)


def _validate_timing(value: Any, label: str) -> None:
    timing = _mapping(value, label)
    if set(timing) != {"started_unix", "finished_unix", "elapsed_seconds"}:
        raise ShadowIdentificationArtifactError("%s fields differ" % label)
    started = _finite(timing.get("started_unix"), label + ".started_unix")
    finished = _finite(timing.get("finished_unix"), label + ".finished_unix")
    elapsed = _finite(timing.get("elapsed_seconds"), label + ".elapsed_seconds")
    if elapsed < 0.0 or finished < started or not math.isclose(
        finished - started, elapsed, rel_tol=1e-6, abs_tol=1e-3
    ):
        raise ShadowIdentificationArtifactError("%s is inconsistent" % label)


def _finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ShadowIdentificationArtifactError(
                "%s contains a non-finite number" % label
            )
        return
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, list):
        for item in value:
            _finite_tree(item, label)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ShadowIdentificationArtifactError(
                    "%s contains a non-string key" % label
                )
            _finite_tree(item, label)
        return
    raise ShadowIdentificationArtifactError(
        "%s contains unsupported value type %s"
        % (label, type(value).__name__)
    )


def _reject_duplicate_pairs(
    pairs: Sequence[Tuple[str, Any]],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON key %r" % key)
        output[key] = value
    return output


def _require_real_file(path: Path, label: str) -> Path:
    candidate = Path(path)
    if ".." in candidate.parts:
        raise ShadowIdentificationArtifactError(
            "%s must not contain parent traversal" % label
        )
    absolute = candidate if candidate.is_absolute() else Path.cwd() / candidate
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise ShadowIdentificationArtifactError(
                "%s must not traverse symbolic links" % label
            )
    if not absolute.is_file():
        raise ShadowIdentificationArtifactError(
            "%s must be an existing nonsymlink regular file" % label
        )
    return absolute.resolve(strict=True)


def _require_real_directory(path: Path, label: str) -> Path:
    candidate = Path(path)
    if ".." in candidate.parts:
        raise ShadowIdentificationArtifactError(
            "%s must not contain parent traversal" % label
        )
    absolute = candidate if candidate.is_absolute() else Path.cwd() / candidate
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise ShadowIdentificationArtifactError(
                "%s must not traverse symbolic links" % label
            )
    if not absolute.is_dir():
        raise ShadowIdentificationArtifactError(
            "%s must be an existing nonsymlink directory" % label
        )
    return absolute.resolve(strict=True)


def _load_strict_json(path: Path, label: str) -> Tuple[Path, Dict[str, Any]]:
    real = _require_real_file(path, label)
    try:
        value = json.loads(
            real.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError("non-finite JSON constant %r" % item)
            ),
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise ShadowIdentificationArtifactError(
            "%s is not strict finite duplicate-free JSON" % label
        ) from error
    if not isinstance(value, dict):
        raise ShadowIdentificationArtifactError(
            "%s must contain one JSON object" % label
        )
    _finite_tree(value, label)
    return real, value


def _load_hashed_json(
    path: Path, label: str, *, ensure_ascii: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    real, value = _load_strict_json(path, label)
    expected = _require_sha256(
        value.get("result_payload_sha256"),
        label + ".result_payload_sha256",
    )
    unhashed = {
        key: item
        for key, item in value.items()
        if key != "result_payload_sha256"
    }
    observed = _sha256_bytes(_canonical(unhashed, ensure_ascii=ensure_ascii))
    if observed != expected:
        raise ShadowIdentificationArtifactError(
            "%s canonical payload hash differs" % label
        )
    return value, {
        "path": str(real),
        "file_sha256": _sha256_file(real),
        "payload_sha256": observed,
        "schema_version": value.get("schema_version"),
    }


def _require_all_true(
    value: Any, expected_fields: frozenset, label: str
) -> None:
    acceptance = _mapping(value, label + ".acceptance")
    if set(acceptance) != set(expected_fields):
        raise ShadowIdentificationArtifactError(
            "%s acceptance fields differ" % label
        )
    if any(acceptance[field] is not True for field in expected_fields):
        raise ShadowIdentificationArtifactError(
            "%s contains a false or non-boolean acceptance" % label
        )


def _require_clean_source(value: Any, expected_commit: str, label: str) -> None:
    source = _mapping(value, label + ".source")
    if (
        set(source) != {"commit", "branch", "status_short"}
        or source.get("commit") != expected_commit
        or source.get("status_short") != []
        or not isinstance(source.get("branch"), str)
        or not source.get("branch")
    ):
        raise ShadowIdentificationArtifactError(
            "%s is not from the exact clean source commit" % label
        )


def _load_case(
    manifest_path: Path, case_id: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    real = _require_real_file(manifest_path, "case manifest")
    matches = []
    try:
        with real.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                stripped = raw_line.rstrip(b"\r\n")
                if not stripped:
                    continue
                try:
                    row = json.loads(
                        stripped.decode("utf-8"),
                        object_pairs_hook=_reject_duplicate_pairs,
                        parse_constant=lambda item: (_ for _ in ()).throw(
                            ValueError("non-finite JSON constant %r" % item)
                        ),
                    )
                except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
                    raise ShadowIdentificationArtifactError(
                        "manifest line %d is invalid" % line_number
                    ) from error
                if not isinstance(row, dict):
                    raise ShadowIdentificationArtifactError(
                        "manifest line %d is not an object" % line_number
                    )
                _finite_tree(row, "manifest line %d" % line_number)
                if row.get("case_id") == case_id:
                    matches.append(
                        (
                            line_number,
                            row,
                            _sha256_bytes(stripped),
                        )
                    )
    except OSError as error:
        raise ShadowIdentificationArtifactError("cannot read case manifest") from error
    if len(matches) != 1:
        raise ShadowIdentificationArtifactError(
            "expected exactly one manifest row for %s, found %d"
            % (case_id, len(matches))
        )
    line_number, row, row_sha256 = matches[0]
    if (
        row.get("split") != "bringup_canary"
        or row.get("study_partition") != "development"
        or row.get("settle_actions") != 20
    ):
        raise ShadowIdentificationArtifactError(
            "manifest row is not the registered settled bring-up canary"
        )
    return row, {
        "path": str(real),
        "file_sha256": _sha256_file(real),
        "line_number": line_number,
        "row_sha256": row_sha256,
    }


def _validate_package_inventory(
    value: Any, expected_names: frozenset, label: str
) -> None:
    packages = _mapping(value, label)
    if set(packages) != set(expected_names) or any(
        not isinstance(version, str) or not version
        for version in packages.values()
    ):
        raise ShadowIdentificationArtifactError(
            "%s package inventory differs" % label
        )


def _validate_single_h100_devices(
    value: Any, *, expected_device: str, label: str
) -> Dict[str, Any]:
    devices = value
    if not isinstance(devices, list) or len(devices) != 1:
        raise ShadowIdentificationArtifactError(
            "%s must expose exactly one H100 device" % label
        )
    device = _mapping(devices[0], label + ".devices[0]")
    if set(device) != {"name", "uuid", "driver_version"}:
        raise ShadowIdentificationArtifactError(
            "%s device fields differ" % label
        )
    if (
        device.get("name") != expected_device
        or "H100" not in expected_device
        or not isinstance(device.get("uuid"), str)
        or not device.get("uuid")
        or not isinstance(device.get("driver_version"), str)
        or not device.get("driver_version")
    ):
        raise ShadowIdentificationArtifactError(
            "%s H100 identity differs from external authority" % label
        )
    return dict(device)


def _validate_numeric(
    numeric: Mapping[str, Any],
    expected_commit: str,
    *,
    expected_job_id: str,
    expected_host: str,
    expected_device: str,
) -> None:
    if (
        set(numeric) != set(NUMERIC_TOP_FIELDS)
        or numeric.get("schema_version") != EXPECTED_NUMERIC_SCHEMA
        or numeric.get("status") != "passed"
        or numeric.get("scientific_result") is not False
        or numeric.get("evidence_tier") != NUMERIC_EVIDENCE_TIER
        or numeric.get("claim_limit") != NUMERIC_CLAIM_LIMIT
    ):
        raise ShadowIdentificationArtifactError(
            "numeric prerequisite has wrong schema or status"
        )
    _require_clean_source(numeric.get("source"), expected_commit, "numeric prerequisite")
    _require_all_true(
        numeric.get("acceptance"),
        NUMERIC_ACCEPTANCE_FIELDS,
        "numeric prerequisite",
    )
    _integer(numeric.get("created_at_unix"), "numeric created_at_unix", minimum=1)
    if _finite(numeric.get("duration_seconds"), "numeric duration_seconds") < 0.0:
        raise ShadowIdentificationArtifactError(
            "numeric duration_seconds must be nonnegative"
        )
    runtime = _mapping(numeric.get("runtime"), "numeric prerequisite.runtime")
    if set(runtime) != set(NUMERIC_RUNTIME_FIELDS):
        raise ShadowIdentificationArtifactError(
            "numeric runtime fields differ"
        )
    if (
        runtime.get("python_executable") != REGISTERED_EVALUATION_PYTHON
        or not isinstance(runtime.get("python_version"), str)
        or not runtime.get("python_version").startswith("3.8.")
        or runtime.get("host") != expected_host
        or runtime.get("slurm_job_id") != expected_job_id
        or runtime.get("slurm_job_name") != EXPECTED_NUMERIC_JOB_NAME
        or runtime.get("slurm_partition") != "mig"
        or not isinstance(runtime.get("cuda_visible_devices"), str)
        or not runtime.get("cuda_visible_devices")
        or runtime.get("production_grid_validation_enabled") is not True
    ):
        raise ShadowIdentificationArtifactError(
            "numeric runtime differs from registered allocation authority"
        )
    inventory = _mapping(
        runtime.get("allocated_gpu_inventory"),
        "numeric allocated GPU inventory",
    )
    if (
        set(inventory) != {"available", "devices", "error"}
        or inventory.get("available") is not True
        or inventory.get("error") is not None
    ):
        raise ShadowIdentificationArtifactError(
            "numeric allocated GPU inventory is not a complete success"
        )
    _validate_single_h100_devices(
        inventory.get("devices"),
        expected_device=expected_device,
        label="numeric allocation",
    )
    _validate_package_inventory(
        runtime.get("packages"),
        frozenset(("numpy", "scipy", "mujoco", "cvxpy", "osqp")),
        "numeric runtime",
    )
    tests = _mapping(numeric.get("tests"), "numeric prerequisite.tests")
    if set(tests) != set(NUMERIC_TEST_FIELDS):
        raise ShadowIdentificationArtifactError("numeric tests fields differ")
    try:
        from scripts.run_poisson_numeric_validation import DEFAULT_TEST_MODULES
    except ImportError as error:
        raise ShadowIdentificationArtifactError(
            "cannot load registered numeric module inventory"
        ) from error
    tests_run = _integer(tests.get("tests_run"), "numeric tests_run")
    output = tests.get("output")
    ran_match = (
        re.search(r"(?:^|\n)Ran ([0-9]+) tests? in ", output)
        if isinstance(output, str)
        else None
    )
    if (
        tests.get("modules") != list(DEFAULT_TEST_MODULES)
        or tests.get("minimum_tests") != 30
        or tests_run < 30
        or tests.get("failure_count") != 0
        or tests.get("error_count") != 0
        or tests.get("skip_count") != 0
        or tests.get("failures") != []
        or tests.get("errors") != []
        or tests.get("skipped") != []
        or ran_match is None
        or int(ran_match.group(1)) != tests_run
        or REQUIRED_NUMERIC_TEST_MODULE not in tests.get("modules", [])
    ):
        raise ShadowIdentificationArtifactError(
            "numeric prerequisite test execution evidence is incomplete"
        )


def _validate_parity_envelope(
    parity: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_case_id: str,
    expected_job_id: str,
    expected_host: str,
    expected_device: str,
) -> None:
    if (
        set(parity) != set(PARITY_TOP_FIELDS)
        or parity.get("schema_version") != EXPECTED_PARITY_SCHEMA
        or parity.get("status") != "passed"
        or parity.get("scientific_result") is not False
        or parity.get("evidence_tier") != PARITY_EVIDENCE_TIER
        or parity.get("claim_limit") != PARITY_CLAIM_LIMIT
        or parity.get("case_id") != expected_case_id
    ):
        raise ShadowIdentificationArtifactError(
            "exact-parity prerequisite has wrong envelope"
        )
    _validate_timing(parity.get("timing"), "exact-parity timing")
    _require_all_true(
        parity.get("acceptance"), PARITY_ACCEPTANCE_FIELDS, "exact parity"
    )
    provenance = _mapping(parity.get("provenance"), "exact-parity provenance")
    if set(provenance) != set(PARITY_PROVENANCE_FIELDS):
        raise ShadowIdentificationArtifactError(
            "exact-parity provenance fields differ"
        )
    _require_clean_source(provenance.get("source"), expected_commit, "exact parity")
    if (
        provenance.get("host") != expected_host
        or provenance.get("slurm_job_id") != expected_job_id
        or provenance.get("slurm_job_name") != EXPECTED_PARITY_JOB_NAME
        or provenance.get("python_executable") != REGISTERED_EVALUATION_PYTHON
        or not isinstance(provenance.get("python_version"), str)
        or not provenance.get("python_version").startswith("3.8.")
    ):
        raise ShadowIdentificationArtifactError(
            "exact-parity runtime differs from registered allocation authority"
        )
    _validate_package_inventory(
        provenance.get("packages"),
        frozenset(("numpy", "mujoco", "robosuite", "scipy")),
        "exact-parity provenance",
    )
    gpu = _mapping(provenance.get("gpu"), "exact-parity GPU inventory")
    if set(gpu) != {"devices"}:
        raise ShadowIdentificationArtifactError(
            "exact-parity GPU inventory fields differ"
        )
    _validate_single_h100_devices(
        gpu.get("devices"),
        expected_device=expected_device,
        label="exact-parity allocation",
    )
    _mapping(parity.get("ordinary_replay"), "exact-parity ordinary replay")
    _mapping(parity.get("callback_replay"), "exact-parity callback replay")


def _load_bound_prerequisites(
    *,
    numeric_path: Path,
    parity_path: Path,
    manifest_path: Path,
    selection_path: Path,
    runtime_path: Path,
    historical_result_root: Path,
    expected_code_commit: str,
    expected_case_id: str,
    expected_numeric_job_id: str,
    expected_numeric_host: str,
    expected_numeric_device: str,
    expected_parity_job_id: str,
    expected_parity_host: str,
    expected_parity_device: str,
) -> Dict[str, Any]:
    """Load every immutable authority needed by a retained producer outcome."""

    numeric, numeric_identity = _load_hashed_json(
        numeric_path, "numeric prerequisite", ensure_ascii=True
    )
    _validate_numeric(
        numeric,
        expected_code_commit,
        expected_job_id=expected_numeric_job_id,
        expected_host=expected_numeric_host,
        expected_device=expected_numeric_device,
    )
    case, manifest_identity = _load_case(manifest_path, expected_case_id)

    selection_real, selection = _load_strict_json(
        selection_path, "selection protocol"
    )
    selection_identity = {
        "path": str(selection_real),
        "file_sha256": _sha256_file(selection_real),
        "schema_version": selection.get("schema_version"),
        "protocol_id": selection.get("protocol_id"),
    }
    if (
        selection_identity["file_sha256"] != case.get("protocol_config_sha256")
        or not isinstance(selection.get("protocol_id"), str)
        or not selection.get("protocol_id")
    ):
        raise ShadowIdentificationArtifactError(
            "selection protocol byte binding differs from the manifest"
        )
    runtime_record = _mapping(
        selection.get("runtime_protocol"), "selection runtime_protocol"
    )
    relative_runtime = runtime_record.get("relative_path")
    if (
        not isinstance(relative_runtime, str)
        or not relative_runtime
        or Path(relative_runtime).is_absolute()
        or ".." in Path(relative_runtime).parts
    ):
        raise ShadowIdentificationArtifactError(
            "selection runtime protocol path is invalid"
        )
    runtime_real = _require_real_file(runtime_path, "runtime protocol")
    if runtime_real != (selection_real.parents[1] / relative_runtime).resolve():
        raise ShadowIdentificationArtifactError(
            "runtime protocol path differs from selection"
        )
    runtime_file_sha256 = _sha256_file(runtime_real)
    if runtime_file_sha256 != runtime_record.get("raw_file_sha256"):
        raise ShadowIdentificationArtifactError(
            "runtime protocol raw byte binding differs"
        )
    try:
        from main.poisson_fullbody.feasibility_protocol import (
            FeasibilityProtocolError,
            load_feasibility_protocol,
        )

        runtime_protocol, runtime_hashes = load_feasibility_protocol(
            runtime_real,
            expected_protocol_sha256=runtime_record.get(
                "semantic_protocol_sha256"
            ),
        )
    except FeasibilityProtocolError as error:
        raise ShadowIdentificationArtifactError(
            "runtime protocol is invalid: %s" % error
        ) from error
    if (
        runtime_hashes.parameter_block_sha256
        != runtime_record.get("parameter_block_sha256")
    ):
        raise ShadowIdentificationArtifactError(
            "runtime protocol parameter-block binding differs"
        )
    runtime_identity = {
        "path": str(runtime_real),
        "file_sha256": runtime_file_sha256,
        "semantic_protocol_sha256": runtime_hashes.protocol_sha256,
        "parameter_block_sha256": runtime_hashes.parameter_block_sha256,
        "schema_version": runtime_protocol.get("schema_version"),
        "protocol_id": runtime_protocol.get("protocol_id"),
    }

    historical_binding = _mapping(
        case.get("historical_aegis_result"), "manifest historical binding"
    )
    relative_historical = historical_binding.get("source_relative_path")
    if (
        not isinstance(relative_historical, str)
        or not relative_historical
        or Path(relative_historical).is_absolute()
        or ".." in Path(relative_historical).parts
    ):
        raise ShadowIdentificationArtifactError(
            "historical result relative path is invalid"
        )
    historical_root = _require_real_directory(
        historical_result_root, "historical result root"
    )
    historical_path = historical_root / relative_historical
    if _sha256_file(
        _require_real_file(historical_path, "historical result")
    ) != historical_binding.get("raw_file_sha256"):
        raise ShadowIdentificationArtifactError(
            "historical result raw byte binding differs"
        )
    try:
        from main.poisson_fullbody.shadow_replay import (
            ReplayContractError,
            load_historical_action_replay,
        )

        replay = load_historical_action_replay(
            historical_path, expected_case_id=expected_case_id
        )
    except ReplayContractError as error:
        raise ShadowIdentificationArtifactError(
            "historical replay is invalid: %s" % error
        ) from error
    if (
        len(replay.steps) != ACTION_COUNT
        or replay.result_payload_sha256
        != historical_binding.get("result_payload_sha256")
    ):
        raise ShadowIdentificationArtifactError(
            "historical replay is not the exact 237-action canary"
        )
    historical_identity = {
        "path": str(historical_path.resolve()),
        "file_sha256": replay.result_file_sha256,
        "payload_sha256": replay.result_payload_sha256,
        "executed_sequence_sha256": replay.executed_sequence_sha256,
        "action_count": len(replay.steps),
    }

    parity, parity_identity = _load_hashed_json(
        parity_path, "exact-parity prerequisite"
    )
    _validate_parity_envelope(
        parity,
        expected_commit=expected_code_commit,
        expected_case_id=expected_case_id,
        expected_job_id=expected_parity_job_id,
        expected_host=expected_parity_host,
        expected_device=expected_parity_device,
    )
    try:
        from scripts.run_poisson_shadow_identification import (
            ShadowIdentificationRunnerError,
            _require_upstream_parity,
        )

        deep_parity = _require_upstream_parity(
            Path(parity_path).resolve(),
            case_id=expected_case_id,
            source_commit=expected_code_commit,
            historical_payload_sha256=replay.result_payload_sha256,
            manifest_sha256=manifest_identity["file_sha256"],
            manifest_row_sha256=manifest_identity["row_sha256"],
            expected_callback_count=CALLBACK_COUNT,
            historical_provenance=replay.provenance(),
            expected_action_state_sha256_ledger=[
                step.simulator_state_sha256 for step in replay.steps
            ],
        )
    except ShadowIdentificationRunnerError as error:
        raise ShadowIdentificationArtifactError(
            "exact-parity prerequisite is invalid: %s" % error
        ) from error
    if _canonical(deep_parity) != _canonical(parity):
        raise ShadowIdentificationArtifactError(
            "strict and deep exact-parity loads differ"
        )
    return {
        "numeric": numeric,
        "numeric_identity": numeric_identity,
        "case": case,
        "manifest_identity": manifest_identity,
        "selection": selection,
        "selection_identity": selection_identity,
        "runtime_protocol": runtime_protocol,
        "runtime_identity": runtime_identity,
        "replay": replay,
        "historical_identity": historical_identity,
        "parity": parity,
        "parity_identity": parity_identity,
    }


def _validate_producer_authority(
    provenance: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_job_id: str,
    expected_host: str,
    expected_device: str,
) -> Dict[str, Any]:
    if set(provenance) != set(PRODUCER_PROVENANCE_FIELDS):
        raise ShadowIdentificationArtifactError(
            "identification producer provenance fields differ"
        )
    _require_clean_source(
        provenance.get("source"), expected_commit, "identification producer"
    )
    if (
        provenance.get("slurm_job_id") != expected_job_id
        or provenance.get("host") != expected_host
        or provenance.get("slurm_job_name") != EXPECTED_PRODUCER_JOB_NAME
        or provenance.get("python_executable") != REGISTERED_EVALUATION_PYTHON
        or not isinstance(provenance.get("python_version"), str)
        or not provenance.get("python_version").startswith("3.8.")
    ):
        raise ShadowIdentificationArtifactError(
            "identification producer runtime differs from registered authority"
        )
    packages = _mapping(
        provenance.get("packages"), "identification producer.packages"
    )
    if set(packages) != {"numpy", "mujoco", "robosuite", "scipy"} or any(
        value is not None and (not isinstance(value, str) or not value)
        for value in packages.values()
    ):
        raise ShadowIdentificationArtifactError(
            "identification producer package inventory differs"
        )
    gpu = _mapping(provenance.get("gpu"), "identification producer.gpu")
    if set(gpu) != {"devices"}:
        raise ShadowIdentificationArtifactError(
            "identification producer GPU inventory fields differ"
        )
    devices = gpu.get("devices")
    if not isinstance(devices, list) or len(devices) != 1:
        raise ShadowIdentificationArtifactError(
            "identification producer must expose exactly one H100 device"
        )
    device = _mapping(devices[0], "identification producer.gpu.devices[0]")
    if set(device) != {"name", "uuid", "driver_version"}:
        raise ShadowIdentificationArtifactError(
            "identification producer device fields differ"
        )
    if (
        device.get("name") != expected_device
        or "H100" not in expected_device
        or not isinstance(device.get("uuid"), str)
        or not device.get("uuid")
        or not isinstance(device.get("driver_version"), str)
        or not device.get("driver_version")
    ):
        raise ShadowIdentificationArtifactError(
            "identification producer H100 device differs from external authority"
        )
    return dict(device)


def _validate_failure_envelope(
    result: Mapping[str, Any],
    *,
    result_identity: Mapping[str, Any],
    expected_case_id: str,
    expected_commit: str,
    expected_job_id: str,
    expected_host: str,
    expected_device: str,
    external_authority: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    allowed = {
        "schema_version",
        "status",
        "scientific_result",
        "evidence_tier",
        "claim_limit",
        "case_id",
        "phase",
        "timing",
        "provenance",
        "historical",
        "shadow_replay",
        "acceptance",
        "failure_evidence",
        "failure",
        "result_payload_sha256",
    }
    if set(result) - allowed:
        raise ShadowIdentificationArtifactError(
            "failed producer artifact has unknown top-level fields"
        )
    if (
        result.get("schema_version") != EXPECTED_SCHEMA
        or result.get("status") != "failed"
        or result.get("scientific_result") is not False
        or result.get("case_id") != expected_case_id
        or result.get("evidence_tier") != IDENTIFICATION_EVIDENCE_TIER
        or result.get("claim_limit") != IDENTIFICATION_CLAIM_LIMIT
    ):
        raise ShadowIdentificationArtifactError(
            "failed producer artifact has wrong schema, status, or case"
        )
    phase = result.get("phase")
    if not isinstance(phase, str) or not phase:
        raise ShadowIdentificationArtifactError(
            "failed producer artifact lacks a typed phase"
        )
    failure = _mapping(result.get("failure"), "failed producer.failure")
    if set(failure) != {"type", "message", "traceback"} or any(
        not isinstance(failure.get(field), str) or not failure.get(field)
        for field in ("type", "message", "traceback")
    ):
        raise ShadowIdentificationArtifactError(
            "failed producer artifact has a malformed failure record"
        )
    _validate_timing(result.get("timing"), "failed producer.timing")

    authority_status = "unavailable_before_authority_binding"
    producer_device: Optional[Dict[str, Any]] = None
    provenance = result.get("provenance")
    if provenance is not None:
        provenance = _mapping(provenance, "failed producer.provenance")
        producer_device = _validate_producer_authority(
            provenance,
            expected_commit=expected_commit,
            expected_job_id=expected_job_id,
            expected_host=expected_host,
            expected_device=expected_device,
        )
        authority_status = "exact_external_authority_bound"

    acceptance = result.get("acceptance")
    if acceptance is not None:
        acceptance = _mapping(acceptance, "failed producer.acceptance")
        if (
            set(acceptance) != set(IDENTIFICATION_ACCEPTANCE_FIELDS)
            or any(not isinstance(value, bool) for value in acceptance.values())
            or all(acceptance.values())
        ):
            raise ShadowIdentificationArtifactError(
                "failed producer acceptance is malformed or falsely all-true"
            )

    failure_evidence = result.get("failure_evidence")
    diagnostic_kind = "producer_failed_before_complete_shadow"
    failure_validation: Optional[Dict[str, Any]] = None
    typed_differential_failure = bool(
        failure.get("type") == DIFFERENTIAL_FAILURE_TYPE
        or phase == DIFFERENTIAL_FAILURE_PHASE
    )
    if (failure_evidence is not None) is not typed_differential_failure:
        raise ShadowIdentificationArtifactError(
            "differential-audit failure evidence and typed failure cause differ"
        )
    if failure_evidence is not None:
        if (
            failure.get("type") != DIFFERENTIAL_FAILURE_TYPE
            or failure.get("message") != DIFFERENTIAL_FAILURE_MESSAGE
            or phase != DIFFERENTIAL_FAILURE_PHASE
            or "shadow_replay" in result
            or "acceptance" in result
        ):
            raise ShadowIdentificationArtifactError(
                "differential-audit failure evidence violates phase causality"
            )
        if external_authority is None:
            raise ShadowIdentificationArtifactError(
                "retained differential failure lacks external immutable authorities"
            )
        evidence = _mapping(
            failure_evidence, "failed producer.failure_evidence"
        )
        expected_fields = {
            "schema_version",
            "phase",
            "failure_kind",
            "scientific_result",
            "active_physics_authorized",
            "settled_link56_differential_audit",
            "settled_link56_differential_audit_validation",
            "interpretation",
            "authority_binding",
        }
        if set(evidence) != expected_fields:
            raise ShadowIdentificationArtifactError(
                "differential-audit failure evidence fields differ"
            )
        if (
            evidence.get("schema_version") != DIFFERENTIAL_FAILURE_SCHEMA
            or evidence.get("phase") != DIFFERENTIAL_FAILURE_PHASE
            or phase != DIFFERENTIAL_FAILURE_PHASE
            or evidence.get("failure_kind")
            != "registered_protected_sample_differential_audit_failed"
            or evidence.get("scientific_result") is not False
            or evidence.get("active_physics_authorized") is not False
            or not isinstance(evidence.get("interpretation"), str)
            or not evidence.get("interpretation")
            or provenance is None
        ):
            raise ShadowIdentificationArtifactError(
                "differential-audit failure evidence semantics differ"
            )
        binding = _mapping(
            evidence.get("authority_binding"),
            "failed producer.failure_evidence.authority_binding",
        )
        required_binding = {
            "source",
            "manifest_sha256",
            "manifest_row_sha256",
            "selection_config_sha256",
            "runtime_protocol_raw_sha256",
            "runtime_protocol_semantic_sha256",
            "runtime_parameter_block_sha256",
            "historical_result_file_sha256",
            "historical_result_payload_sha256",
            "upstream_parity_payload_sha256",
            "host",
            "slurm_job_id",
            "slurm_job_name",
        }
        if set(binding) != required_binding:
            raise ShadowIdentificationArtifactError(
                "differential-audit failure authority binding fields differ"
            )
        expected_binding = {
            field: provenance.get(field) for field in required_binding
        }
        expected_binding["source"] = provenance.get("source")
        if _canonical(binding) != _canonical(expected_binding):
            raise ShadowIdentificationArtifactError(
                "differential-audit failure authority binding differs"
            )
        replay = external_authority["replay"]
        expected_external_binding = {
            "source": provenance.get("source"),
            "manifest_sha256": external_authority["manifest_identity"][
                "file_sha256"
            ],
            "manifest_row_sha256": external_authority["manifest_identity"][
                "row_sha256"
            ],
            "selection_config_sha256": external_authority[
                "selection_identity"
            ]["file_sha256"],
            "runtime_protocol_raw_sha256": external_authority[
                "runtime_identity"
            ]["file_sha256"],
            "runtime_protocol_semantic_sha256": external_authority[
                "runtime_identity"
            ]["semantic_protocol_sha256"],
            "runtime_parameter_block_sha256": external_authority[
                "runtime_identity"
            ]["parameter_block_sha256"],
            "historical_result_file_sha256": replay.result_file_sha256,
            "historical_result_payload_sha256": replay.result_payload_sha256,
            "upstream_parity_payload_sha256": external_authority[
                "parity_identity"
            ]["payload_sha256"],
            "host": expected_host,
            "slurm_job_id": expected_job_id,
            "slurm_job_name": provenance.get("slurm_job_name"),
        }
        if _canonical(binding) != _canonical(expected_external_binding):
            raise ShadowIdentificationArtifactError(
                "differential-audit failure differs from external immutable bindings"
            )
        expected_paths = {
            "manifest_path": external_authority["manifest_identity"]["path"],
            "selection_config_path": external_authority["selection_identity"][
                "path"
            ],
            "runtime_protocol_path": external_authority["runtime_identity"][
                "path"
            ],
            "historical_result_path": external_authority[
                "historical_identity"
            ]["path"],
            "upstream_parity_path": external_authority["parity_identity"][
                "path"
            ],
        }
        if any(
            provenance.get(field) != expected
            for field, expected in expected_paths.items()
        ):
            raise ShadowIdentificationArtifactError(
                "differential-audit failure input paths differ from external authority"
            )
        if _canonical(result.get("historical")) != _canonical(
            replay.provenance()
        ):
            raise ShadowIdentificationArtifactError(
                "differential-audit failure historical replay binding differs"
            )
        audit = _mapping(
            evidence.get("settled_link56_differential_audit"),
            "failed differential audit",
        )
        receipt = _mapping(
            evidence.get("settled_link56_differential_audit_validation"),
            "failed differential audit receipt",
        )
        try:
            from main.poisson_fullbody.jacobians import (
                DifferentialAuditError,
                inspect_protected_sample_differential_audit,
            )

            integration = _mapping(
                audit.get("integration_state"),
                "failed differential audit.integration_state",
            )
            ordered_samples = audit.get("ordered_samples")
            if (
                not isinstance(ordered_samples, list)
                or len(ordered_samples) != 1531
                or [sample.get("sample_id") for sample in ordered_samples]
                != list(range(1531))
                or set(sample.get("body_name") for sample in ordered_samples)
                != set(
                    external_authority["runtime_protocol"]["claim_scope"][
                        "protected_robot_bodies"
                    ]
                )
                or any(
                    sample.get("source") != "collision_geom_surface"
                    for sample in ordered_samples
                )
            ):
                raise ShadowIdentificationArtifactError(
                    "retained failed audit protected-sample authority differs"
                )
            if _canonical(audit.get("differential_audit_config")) != _canonical(
                external_authority["runtime_protocol"]["differential_audit"]
            ):
                raise ShadowIdentificationArtifactError(
                    "retained failed audit config differs from frozen runtime protocol"
                )
            reconstructed = inspect_protected_sample_differential_audit(
                audit,
                expected_samples=ordered_samples,
                expected_arm_dof_indices=list(range(7)),
                expected_integration_state_sha256=integration.get(
                    "source_initial_sha256"
                ),
                expected_differential_audit_config=external_authority[
                    "runtime_protocol"
                ]["differential_audit"],
            )
        except (DifferentialAuditError, KeyError, TypeError, ValueError) as error:
            raise ShadowIdentificationArtifactError(
                "retained failed differential audit is malformed: %s" % error
            ) from error
        if reconstructed.get("passed") is not False or _canonical(
            reconstructed
        ) != _canonical(receipt):
            raise ShadowIdentificationArtifactError(
                "retained failed differential audit receipt differs"
            )
        failure_validation = dict(reconstructed)
        # The external inputs bind the protocol, history, parity, and producer
        # allocation, but they do not independently supply the settled-state
        # hash or exact protected-sample point ledger.  This reconstruction can
        # establish producer-local receipt consistency only; it cannot promote
        # that failure into a verified mechanism diagnosis.
        diagnostic_kind = (
            "producer_reported_differential_audit_failure_with_"
            "unbound_sample_state_authority"
        )

    return {
        "schema_version": CONSUMER_SCHEMA,
        "status": "validated_negative",
        "producer_status": "failed",
        "stage_13_authorized": False,
        "authorization_scope": None,
        "diagnostic_kind": diagnostic_kind,
        "case_id": expected_case_id,
        "identification_result": dict(result_identity),
        "producer_authority_status": authority_status,
        "producer_device": producer_device,
        "producer_failure": {
            "phase": phase,
            "type": failure["type"],
            "message": failure["message"],
        },
        "retained_differential_audit_validation": failure_validation,
        "retained_differential_audit_validation_scope": (
            "producer_local_structural_and_receipt_consistency_only"
            if failure_validation is not None
            else None
        ),
        "external_sample_state_authority_bound": False,
        "mechanism_diagnosis_verified": False,
        "partial_shadow_interpreted": False,
        "active_physics_authorized": False,
        "active_physics_executed": False,
    }


def _reconstruct_warning_contact(
    shadow: Mapping[str, Any]
) -> Dict[str, Any]:
    identification = _mapping(
        shadow.get("poisson_identification"), "Poisson identification"
    )
    signals = _mapping(identification.get("signals"), "identification signals")
    assessment = _mapping(
        identification.get("contact_prediction_assessment"),
        "contact prediction assessment",
    )
    warning = signals.get(
        "first_observed_minimum_cbf_lhs_negative_on_contact_geom"
    )
    contact = assessment.get("first_link56_contact")
    if contact is None:
        return {
            "authorization": False,
            "diagnostic_kind": "no_link56_contact_outcome",
        }
    contact = _mapping(contact, "first link56 contact")
    if contact.get("observation_index") is None:
        return {
            "authorization": False,
            "diagnostic_kind": "link56_contact_already_present_after_settling",
        }
    if warning is None:
        return {
            "authorization": False,
            "diagnostic_kind": "no_cbf_lhs_negative_warning_on_contact_geom",
        }
    warning = _mapping(warning, "contact-geom negative-CBF warning")
    primary = assessment.get("primary_registered_warning")
    if _canonical(primary) != _canonical(warning):
        return {
            "authorization": False,
            "diagnostic_kind": "primary_warning_is_not_cbf_lhs_negative",
        }
    if warning.get("signal_kind") != "cbf_lhs_negative":
        raise ShadowIdentificationArtifactError(
            "selected warning is not the registered cbf_lhs_negative signal"
        )
    warning_evidence = _mapping(warning.get("evidence"), "warning evidence")
    lhs = _finite(
        warning_evidence.get("observed_cbf_lhs_m2_per_s"),
        "warning evidence CBF lhs",
    )
    if lhs >= 0.0:
        raise ShadowIdentificationArtifactError(
            "cbf_lhs_negative warning evidence is not negative"
        )
    warning_observation = _integer(
        warning.get("observation_index"), "warning observation index"
    )
    if warning_observation >= CALLBACK_COUNT:
        raise ShadowIdentificationArtifactError(
            "warning observation index exceeds complete callback exposure"
        )
    if (
        warning.get("high_level_index") != warning_observation // CALLBACKS_PER_ACTION
        or warning.get("physics_substep_index")
        != warning_observation % CALLBACKS_PER_ACTION
    ):
        raise ShadowIdentificationArtifactError("warning cadence differs")
    contact_observation = _integer(
        contact.get("observation_index"), "contact observation index"
    )
    if contact_observation >= CALLBACK_COUNT:
        raise ShadowIdentificationArtifactError(
            "contact observation index exceeds complete callback exposure"
        )
    if contact.get("is_physical_nonpositive_distance_contact") is not True:
        raise ShadowIdentificationArtifactError(
            "selected contact is not a MuJoCo nonpositive-distance contact"
        )
    phase = contact.get("source_phase")
    if phase == "post_integration_recomputed":
        contact_boundary = contact_observation + 1
    elif phase == "live_solver_phase_preintegration_geometry":
        contact_boundary = contact_observation
    else:
        raise ShadowIdentificationArtifactError(
            "contact source phase cannot define a physical boundary"
        )
    warning_boundary = warning_observation + 1
    lead = contact_boundary - warning_boundary
    recorded_lead = assessment.get("lead_physics_substeps")
    recorded_lead_time = assessment.get("lead_time_s")
    if (
        isinstance(recorded_lead, bool)
        or not isinstance(recorded_lead, int)
        or recorded_lead != lead
        or not math.isclose(
            _finite(recorded_lead_time, "recorded warning lead time"),
            lead * PHYSICS_TIMESTEP_S,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ShadowIdentificationArtifactError(
            "recorded warning lead differs from independent phase arithmetic"
        )
    if (
        warning.get("geom_id") != contact.get("robot_geom_id")
        or warning.get("geom_name") != contact.get("robot_geom_name")
        or not isinstance(warning.get("geom_name"), str)
        or not warning.get("geom_name")
    ):
        raise ShadowIdentificationArtifactError(
            "warning and contact robot geometry differ"
        )
    boundary = (
        (warning_boundary + FILTER_UPDATE_PERIOD - 1)
        // FILTER_UPDATE_PERIOD
    ) * FILTER_UPDATE_PERIOD
    reconstruction = {
        "signal_kind": "cbf_lhs_negative",
        "warning_observation_index": warning_observation,
        "warning_physical_boundary_W": warning_boundary,
        "contact_observation_index": contact_observation,
        "contact_source_phase": phase,
        "contact_physical_boundary_C": contact_boundary,
        "lead_physics_substeps": lead,
        "lead_time_s": lead * PHYSICS_TIMESTEP_S,
        "physics_substeps_per_filter_update": FILTER_UPDATE_PERIOD,
        "first_scheduled_filter_boundary_B": boundary,
        "B_minus_W_physics_substeps": boundary - warning_boundary,
        "C_minus_B_physics_substeps": contact_boundary - boundary,
        "warning_robot_geom_id": warning.get("geom_id"),
        "warning_robot_geom_name": warning.get("geom_name"),
        "contact_robot_geom_id": contact.get("robot_geom_id"),
        "contact_robot_geom_name": contact.get("robot_geom_name"),
        "negative_cbf_lhs_m2_per_s": lhs,
    }
    if lead <= 0:
        return {
            "authorization": False,
            "diagnostic_kind": "nonpositive_phase_correct_warning_lead",
            "timing_reconstruction": reconstruction,
        }
    if not (
        boundary >= warning_boundary
        and boundary - warning_boundary < FILTER_UPDATE_PERIOD
        and boundary < contact_boundary
    ):
        return {
            "authorization": False,
            "diagnostic_kind": "no_scheduled_100hz_update_before_contact",
            "timing_reconstruction": reconstruction,
        }
    if assessment.get("assessment") != "registered_warning_preceded_link56_contact":
        raise ShadowIdentificationArtifactError(
            "contact assessment differs from positive warning lead"
        )
    reconstruction.update(
        {
            "warning_contact_geom_equal": True,
            "lead_strictly_positive": True,
            "B_at_or_after_W": True,
            "B_minus_W_strictly_less_than_period": True,
            "B_strictly_before_C": True,
        }
    )
    return {
        "authorization": True,
        "diagnostic_kind": "actionable_cbf_lhs_warning",
        "timing_reconstruction": reconstruction,
    }


def validate_shadow_identification_artifact(
    *,
    identification_path: Path,
    numeric_path: Path,
    parity_path: Path,
    manifest_path: Path,
    selection_path: Path,
    runtime_path: Path,
    historical_result_root: Path,
    expected_code_commit: str,
    expected_case_id: str,
    expected_producer_job_id: str,
    expected_producer_host: str,
    expected_producer_device: str,
    expected_numeric_job_id: str,
    expected_numeric_host: str,
    expected_numeric_device: str,
    expected_parity_job_id: str,
    expected_parity_host: str,
    expected_parity_device: str,
) -> Dict[str, Any]:
    """Validate one final producer artifact without writing any file."""

    if (
        not isinstance(expected_code_commit, str)
        or COMMIT_PATTERN.fullmatch(expected_code_commit) is None
    ):
        raise ShadowIdentificationArtifactError(
            "expected code commit must be one lowercase 40-character commit"
        )
    for value, label in (
        (expected_case_id, "expected case ID"),
        (expected_producer_job_id, "expected producer job ID"),
        (expected_producer_host, "expected producer host"),
        (expected_producer_device, "expected producer device"),
        (expected_numeric_job_id, "expected numeric job ID"),
        (expected_numeric_host, "expected numeric host"),
        (expected_numeric_device, "expected numeric device"),
        (expected_parity_job_id, "expected parity job ID"),
        (expected_parity_host, "expected parity host"),
        (expected_parity_device, "expected parity device"),
    ):
        if not isinstance(value, str) or not value:
            raise ShadowIdentificationArtifactError("%s must be nonempty" % label)
    if any(
        not job_id.isdigit()
        for job_id in (
            expected_producer_job_id,
            expected_numeric_job_id,
            expected_parity_job_id,
        )
    ):
        raise ShadowIdentificationArtifactError(
            "expected job IDs must identify exact numeric Slurm jobs"
        )
    if expected_case_id != EXPECTED_CASE_ID:
        raise ShadowIdentificationArtifactError(
            "consumer is registered only for %s" % EXPECTED_CASE_ID
        )

    result, result_identity = _load_hashed_json(
        identification_path, "shadow-identification result"
    )
    if result.get("status") == "failed":
        external_authority = None
        if result.get("failure_evidence") is not None:
            external_authority = _load_bound_prerequisites(
                numeric_path=numeric_path,
                parity_path=parity_path,
                manifest_path=manifest_path,
                selection_path=selection_path,
                runtime_path=runtime_path,
                historical_result_root=historical_result_root,
                expected_code_commit=expected_code_commit,
                expected_case_id=expected_case_id,
                expected_numeric_job_id=expected_numeric_job_id,
                expected_numeric_host=expected_numeric_host,
                expected_numeric_device=expected_numeric_device,
                expected_parity_job_id=expected_parity_job_id,
                expected_parity_host=expected_parity_host,
                expected_parity_device=expected_parity_device,
            )
        return _validate_failure_envelope(
            result,
            result_identity=result_identity,
            expected_case_id=expected_case_id,
            expected_commit=expected_code_commit,
            expected_job_id=expected_producer_job_id,
            expected_host=expected_producer_host,
            expected_device=expected_producer_device,
            external_authority=external_authority,
        )
    expected_passed_fields = {
        "schema_version",
        "status",
        "scientific_result",
        "evidence_tier",
        "claim_limit",
        "case_id",
        "phase",
        "timing",
        "provenance",
        "historical",
        "shadow_replay",
        "acceptance",
        "result_payload_sha256",
    }
    if (
        set(result) != expected_passed_fields
        or
        result.get("schema_version") != EXPECTED_SCHEMA
        or result.get("status") != "passed"
        or result.get("phase") != "complete"
        or result.get("scientific_result") is not False
        or result.get("case_id") != expected_case_id
        or result.get("evidence_tier") != IDENTIFICATION_EVIDENCE_TIER
        or result.get("claim_limit") != IDENTIFICATION_CLAIM_LIMIT
        or "failure" in result
        or "failure_evidence" in result
    ):
        raise ShadowIdentificationArtifactError(
            "identification result is neither a complete pass nor a valid failure"
        )
    _validate_timing(result.get("timing"), "passed producer.timing")

    numeric, numeric_identity = _load_hashed_json(
        numeric_path, "numeric prerequisite", ensure_ascii=True
    )
    _validate_numeric(
        numeric,
        expected_code_commit,
        expected_job_id=expected_numeric_job_id,
        expected_host=expected_numeric_host,
        expected_device=expected_numeric_device,
    )
    case, manifest_identity = _load_case(manifest_path, expected_case_id)
    selection_real, selection = _load_strict_json(
        selection_path, "selection protocol"
    )
    selection_identity = {
        "path": str(selection_real),
        "file_sha256": _sha256_file(selection_real),
        "schema_version": selection.get("schema_version"),
        "protocol_id": selection.get("protocol_id"),
    }
    if (
        selection_identity["file_sha256"] != case.get("protocol_config_sha256")
        or not isinstance(selection.get("protocol_id"), str)
        or not selection.get("protocol_id")
    ):
        raise ShadowIdentificationArtifactError(
            "selection protocol byte binding differs from the manifest"
        )
    runtime_record = _mapping(
        selection.get("runtime_protocol"), "selection runtime_protocol"
    )
    relative_runtime = runtime_record.get("relative_path")
    if (
        not isinstance(relative_runtime, str)
        or not relative_runtime
        or Path(relative_runtime).is_absolute()
        or ".." in Path(relative_runtime).parts
    ):
        raise ShadowIdentificationArtifactError(
            "selection runtime protocol path is invalid"
        )
    runtime_real = _require_real_file(runtime_path, "runtime protocol")
    expected_runtime_real = (
        selection_real.parents[1] / relative_runtime
    ).resolve()
    if runtime_real != expected_runtime_real:
        raise ShadowIdentificationArtifactError(
            "runtime protocol path differs from selection"
        )
    runtime_file_sha256 = _sha256_file(runtime_real)
    if runtime_file_sha256 != runtime_record.get("raw_file_sha256"):
        raise ShadowIdentificationArtifactError(
            "runtime protocol raw byte binding differs"
        )
    try:
        from main.poisson_fullbody.feasibility_protocol import (
            FeasibilityProtocolError,
            load_feasibility_protocol,
        )

        runtime_protocol, runtime_hashes = load_feasibility_protocol(
            runtime_real,
            expected_protocol_sha256=runtime_record.get(
                "semantic_protocol_sha256"
            ),
        )
    except FeasibilityProtocolError as error:
        raise ShadowIdentificationArtifactError(
            "runtime protocol is invalid: %s" % error
        ) from error
    if (
        runtime_hashes.parameter_block_sha256
        != runtime_record.get("parameter_block_sha256")
    ):
        raise ShadowIdentificationArtifactError(
            "runtime protocol parameter-block binding differs"
        )
    runtime_identity = {
        "path": str(runtime_real),
        "file_sha256": runtime_file_sha256,
        "semantic_protocol_sha256": runtime_hashes.protocol_sha256,
        "parameter_block_sha256": runtime_hashes.parameter_block_sha256,
        "schema_version": runtime_protocol.get("schema_version"),
        "protocol_id": runtime_protocol.get("protocol_id"),
    }

    historical_binding = _mapping(
        case.get("historical_aegis_result"), "manifest historical binding"
    )
    relative_historical = historical_binding.get("source_relative_path")
    if (
        not isinstance(relative_historical, str)
        or not relative_historical
        or Path(relative_historical).is_absolute()
        or ".." in Path(relative_historical).parts
    ):
        raise ShadowIdentificationArtifactError(
            "historical result relative path is invalid"
        )
    historical_root = _require_real_directory(
        historical_result_root, "historical result root"
    )
    historical_path = historical_root / relative_historical
    if _sha256_file(_require_real_file(historical_path, "historical result")) != historical_binding.get(
        "raw_file_sha256"
    ):
        raise ShadowIdentificationArtifactError(
            "historical result raw byte binding differs"
        )
    try:
        from main.poisson_fullbody.shadow_replay import (
            ReplayContractError,
            load_historical_action_replay,
        )

        replay = load_historical_action_replay(
            historical_path, expected_case_id=expected_case_id
        )
    except ReplayContractError as error:
        raise ShadowIdentificationArtifactError(
            "historical replay is invalid: %s" % error
        ) from error
    if (
        len(replay.steps) != ACTION_COUNT
        or replay.result_payload_sha256
        != historical_binding.get("result_payload_sha256")
    ):
        raise ShadowIdentificationArtifactError(
            "historical replay is not the exact 237-action canary"
        )
    historical_identity = {
        "path": str(historical_path.resolve()),
        "file_sha256": replay.result_file_sha256,
        "payload_sha256": replay.result_payload_sha256,
        "executed_sequence_sha256": replay.executed_sequence_sha256,
        "action_count": len(replay.steps),
    }

    parity, parity_identity = _load_hashed_json(
        parity_path, "exact-parity prerequisite"
    )
    _validate_parity_envelope(
        parity,
        expected_commit=expected_code_commit,
        expected_case_id=expected_case_id,
        expected_job_id=expected_parity_job_id,
        expected_host=expected_parity_host,
        expected_device=expected_parity_device,
    )
    try:
        from scripts.run_poisson_shadow_identification import (
            ShadowIdentificationRunnerError,
            _require_upstream_parity,
        )

        deep_parity = _require_upstream_parity(
            Path(parity_path).resolve(),
            case_id=expected_case_id,
            source_commit=expected_code_commit,
            historical_payload_sha256=replay.result_payload_sha256,
            manifest_sha256=manifest_identity["file_sha256"],
            manifest_row_sha256=manifest_identity["row_sha256"],
            expected_callback_count=CALLBACK_COUNT,
            historical_provenance=replay.provenance(),
            expected_action_state_sha256_ledger=[
                step.simulator_state_sha256 for step in replay.steps
            ],
        )
    except ShadowIdentificationRunnerError as error:
        raise ShadowIdentificationArtifactError(
            "exact-parity prerequisite is invalid: %s" % error
        ) from error
    if _canonical(deep_parity) != _canonical(parity):
        raise ShadowIdentificationArtifactError(
            "strict and deep exact-parity loads differ"
        )

    provenance = _mapping(result.get("provenance"), "identification provenance")
    producer_device = _validate_producer_authority(
        provenance,
        expected_commit=expected_code_commit,
        expected_job_id=expected_producer_job_id,
        expected_host=expected_producer_host,
        expected_device=expected_producer_device,
    )
    expected_provenance = {
        "manifest_path": manifest_identity["path"],
        "manifest_sha256": manifest_identity["file_sha256"],
        "manifest_line_number": manifest_identity["line_number"],
        "manifest_row_sha256": manifest_identity["row_sha256"],
        "selection_config_path": selection_identity["path"],
        "selection_config_sha256": selection_identity["file_sha256"],
        "selection_protocol_id": selection_identity["protocol_id"],
        "runtime_protocol_path": runtime_identity["path"],
        "runtime_protocol_raw_sha256": runtime_identity["file_sha256"],
        "runtime_protocol_semantic_sha256": runtime_identity[
            "semantic_protocol_sha256"
        ],
        "runtime_parameter_block_sha256": runtime_identity[
            "parameter_block_sha256"
        ],
        "historical_result_path": historical_identity["path"],
        "historical_result_file_sha256": historical_identity["file_sha256"],
        "historical_result_payload_sha256": historical_identity[
            "payload_sha256"
        ],
        "upstream_parity_path": parity_identity["path"],
        "upstream_parity_payload_sha256": parity_identity["payload_sha256"],
    }
    differences = sorted(
        field
        for field, expected in expected_provenance.items()
        if provenance.get(field) != expected
    )
    if differences:
        raise ShadowIdentificationArtifactError(
            "identification immutable binding differs at %s"
            % ", ".join(differences)
        )
    if _canonical(result.get("historical")) != _canonical(replay.provenance()):
        raise ShadowIdentificationArtifactError(
            "identification historical replay binding differs"
        )
    _require_all_true(
        result.get("acceptance"),
        IDENTIFICATION_ACCEPTANCE_FIELDS,
        "shadow identification",
    )

    shadow = _mapping(result.get("shadow_replay"), "shadow replay")
    if set(shadow) != set(SHADOW_REPLAY_FIELDS):
        raise ShadowIdentificationArtifactError(
            "shadow replay top-level fields differ"
        )
    measurement = _mapping(shadow.get("measurement"), "shadow measurement")
    identification = _mapping(
        shadow.get("poisson_identification"), "Poisson identification"
    )
    if (
        shadow.get("executed_action_count") != ACTION_COUNT
        or shadow.get("callback_count") != CALLBACK_COUNT
        or shadow.get("expected_callback_count") != CALLBACK_COUNT
        or measurement.get("observed_physics_substeps") != CALLBACK_COUNT
        or identification.get("observed_callback_count") != CALLBACK_COUNT
        or identification.get("expected_callback_count") != CALLBACK_COUNT
        or measurement.get("first_index") != [0, 0, 0]
        or measurement.get("last_index") != [236, 4, 4]
        or len(shadow.get("action_boundary_state_sha256_ledger", []))
        != ACTION_COUNT
        or len(shadow.get("callback_state_read_only_ledger", []))
        != CALLBACK_COUNT
        or len(identification.get("trace", [])) != CALLBACK_COUNT
    ):
        raise ShadowIdentificationArtifactError(
            "identification exposure or typed cadence endpoints are incomplete"
        )
    expected_action_states = [
        step.simulator_state_sha256 for step in replay.steps
    ]
    parity_callback = _mapping(
        parity.get("callback_replay"), "exact-parity callback replay"
    )
    callback_ledger = shadow.get("callback_state_read_only_ledger")
    callback_ledger_sha256 = _sha256_bytes(_canonical(callback_ledger))
    if (
        shadow.get("action_boundary_state_sha256_ledger")
        != expected_action_states
        or shadow.get("state_sequence_sha256")
        != parity_callback.get("state_sequence_sha256")
        or shadow.get("observation_sequence_sha256")
        != parity_callback.get("observation_sequence_sha256")
        or shadow.get("terminal_simulator_state_sha256")
        != parity_callback.get("terminal_simulator_state_sha256")
        or [record.get("after_sha256") for record in callback_ledger]
        != parity_callback.get("official_integration_state_sha256_ledger")
        or shadow.get("callback_state_read_only_ledger_sha256")
        != callback_ledger_sha256
        or shadow.get("callback_state_sequence_sha256")
        != parity_callback.get("official_integration_state_sequence_sha256")
    ):
        raise ShadowIdentificationArtifactError(
            "identification replay ledgers differ from parity or history"
        )
    try:
        from main.poisson_fullbody.shadow_identification import (
            ShadowIdentificationError,
            validate_shadow_replay_record,
        )

        validate_shadow_replay_record(
            shadow,
            action_count=ACTION_COUNT,
            inner_updates_per_high_level_action=INNER_UPDATES_PER_ACTION,
            physics_substeps_per_inner_update=(
                PHYSICS_SUBSTEPS_PER_INNER_UPDATE
            ),
            physics_timestep_s=PHYSICS_TIMESTEP_S,
            contact_definition=CONTACT_DEFINITION,
            alpha_gain_per_s=float(runtime_protocol["cbf"]["alpha_gain_per_s"]),
            static_drift_thresholds={
                "translation_m": float(
                    runtime_protocol["admissibility"][
                        "max_selected_geom_translation_drift_m"
                    ]
                ),
                "rotation_rad": float(
                    runtime_protocol["admissibility"][
                        "max_selected_geom_rotation_drift_rad"
                    ]
                ),
                "surface_m": float(
                    runtime_protocol["admissibility"][
                        "max_selected_geom_surface_drift_m"
                    ]
                ),
            },
            differential_audit_config=runtime_protocol["differential_audit"],
        )
    except (ShadowIdentificationError, KeyError, TypeError, ValueError) as error:
        raise ShadowIdentificationArtifactError(
            "complete pure shadow replay validation failed: %s" % error
        ) from error

    timing = _reconstruct_warning_contact(shadow)
    construction = _mapping(shadow.get("construction"), "shadow construction")
    differential = _mapping(
        construction.get("settled_link56_differential_audit_validation"),
        "differential audit validation",
    )
    counts = _mapping(differential.get("counts"), "differential audit counts")
    if differential.get("passed") is not True:
        raise ShadowIdentificationArtifactError(
            "complete producer contains a failed differential audit"
        )
    authorization = timing.get("authorization") is True
    return {
        "schema_version": CONSUMER_SCHEMA,
        "status": "validated" if authorization else "validated_negative",
        "producer_status": "passed",
        "stage_13_authorized": authorization,
        "authorization_scope": (
            "stage_13_one_step_counterfactual_preflight_only"
            if authorization
            else None
        ),
        "diagnostic_kind": timing["diagnostic_kind"],
        "case_id": expected_case_id,
        "source_commit": expected_code_commit,
        "producer": {
            "slurm_job_id": expected_producer_job_id,
            "host": expected_producer_host,
            "device": producer_device,
        },
        "identification_result": result_identity,
        "numeric_prerequisite": numeric_identity,
        "numeric_allocation": {
            "slurm_job_id": expected_numeric_job_id,
            "host": expected_numeric_host,
            "device": expected_numeric_device,
        },
        "parity_prerequisite": parity_identity,
        "parity_allocation": {
            "slurm_job_id": expected_parity_job_id,
            "host": expected_parity_host,
            "device": expected_parity_device,
        },
        "manifest": manifest_identity,
        "selection_protocol": selection_identity,
        "runtime_protocol": runtime_identity,
        "historical_result": historical_identity,
        "exposure": {
            "action_count": ACTION_COUNT,
            "callback_count": CALLBACK_COUNT,
            "measurement_first_index": [0, 0, 0],
            "measurement_last_index": [236, 4, 4],
        },
        "differential_audit": {
            "passed": True,
            "sample_count": counts.get("sample_count"),
            "point_jacobian_passed_sample_count": counts.get(
                "point_jacobian_passed_sample_count"
            ),
            "passed_coupled_direction_count": counts.get(
                "passed_coupled_direction_count"
            ),
            "required_coupled_direction_count": counts.get(
                "required_coupled_direction_count"
            ),
            "fresh_pure_reconstruction": "passed",
        },
        "timing_reconstruction": timing.get("timing_reconstruction"),
        "all_acceptance_true": True,
        "partial_output_interpreted": False,
        # Identification can unlock only the registered Stage-13 one-step
        # counterfactual.  It cannot authorize the later active canary.
        "active_physics_authorized": False,
        "active_physics_executed": False,
        "full_active_canary_authorized": False,
        "external_slurm_requirement": (
            "independently confirm numeric, parity, producer, and consumer "
            "top-level COMPLETED|0:0"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--identification-result", required=True, type=Path)
    parser.add_argument("--numeric-prerequisite", required=True, type=Path)
    parser.add_argument("--parity-prerequisite", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selection-protocol", required=True, type=Path)
    parser.add_argument("--runtime-protocol", required=True, type=Path)
    parser.add_argument("--historical-result-root", required=True, type=Path)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--expected-case-id", default=EXPECTED_CASE_ID)
    parser.add_argument("--expected-producer-job-id", required=True)
    parser.add_argument("--expected-producer-host", required=True)
    parser.add_argument("--expected-producer-device", required=True)
    parser.add_argument("--expected-numeric-job-id", required=True)
    parser.add_argument("--expected-numeric-host", required=True)
    parser.add_argument("--expected-numeric-device", required=True)
    parser.add_argument("--expected-parity-job-id", required=True)
    parser.add_argument("--expected-parity-host", required=True)
    parser.add_argument("--expected-parity-device", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    consumer_job = os.environ.get("SLURM_JOB_ID")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if not consumer_job:
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "error": "consumer must run inside a separate Slurm allocation",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    prerequisite_jobs = (
        arguments.expected_numeric_job_id,
        arguments.expected_parity_job_id,
        arguments.expected_producer_job_id,
    )
    if consumer_job in prerequisite_jobs or len(set(prerequisite_jobs)) != 3:
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "error": "consumer, numeric, parity, and producer Slurm jobs must be distinct",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    if (
        cpus is None
        or not cpus.isdigit()
        or int(cpus) < 1
        or int(cpus) > 2
    ):
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "error": "consumer allocation must expose one or two CPUs",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    gpu_environment = tuple(
        os.environ.get(field, "")
        for field in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE")
    )
    if any(value not in ("", "0", "(null)") for value in gpu_environment):
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "error": "consumer must use a CPU-only Slurm allocation",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    if (
        os.environ.get("EVALUATION_PYTHON") != REGISTERED_EVALUATION_PYTHON
        or sys.version_info[:2] != (3, 8)
        or Path(sys.executable).resolve()
        != Path(REGISTERED_EVALUATION_PYTHON).resolve()
    ):
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "error": "consumer interpreter differs from registered Python 3.8 evaluation runtime",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        repo_root = _require_real_directory(
            arguments.repo_root, "consumer source root"
        )
        observed_commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()
        observed_status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--short"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).splitlines()
    except (OSError, subprocess.SubprocessError, ShadowIdentificationArtifactError) as error:
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "error": "consumer clean-source inspection failed: %s" % error,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    if observed_commit != arguments.expected_code_commit or observed_status:
        print(
            json.dumps(
                {
                    "status": "rejected",
                    "error": "consumer is not running from the exact clean expected commit",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        report = validate_shadow_identification_artifact(
            identification_path=arguments.identification_result,
            numeric_path=arguments.numeric_prerequisite,
            parity_path=arguments.parity_prerequisite,
            manifest_path=arguments.manifest,
            selection_path=arguments.selection_protocol,
            runtime_path=arguments.runtime_protocol,
            historical_result_root=arguments.historical_result_root,
            expected_code_commit=arguments.expected_code_commit,
            expected_case_id=arguments.expected_case_id,
            expected_producer_job_id=arguments.expected_producer_job_id,
            expected_producer_host=arguments.expected_producer_host,
            expected_producer_device=arguments.expected_producer_device,
            expected_numeric_job_id=arguments.expected_numeric_job_id,
            expected_numeric_host=arguments.expected_numeric_host,
            expected_numeric_device=arguments.expected_numeric_device,
            expected_parity_job_id=arguments.expected_parity_job_id,
            expected_parity_host=arguments.expected_parity_host,
            expected_parity_device=arguments.expected_parity_device,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema_version": CONSUMER_SCHEMA,
                    "status": "rejected",
                    "stage_13_authorized": False,
                    "active_physics_authorized": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    report["consumer_slurm_job_id"] = consumer_job
    report["producer_and_consumer_jobs_distinct"] = True
    report["writes_experiment_artifacts"] = False
    report["consumer_python_executable"] = sys.executable
    report["consumer_python_version"] = "%d.%d.%d" % sys.version_info[:3]
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
