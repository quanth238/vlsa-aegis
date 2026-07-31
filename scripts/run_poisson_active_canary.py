#!/usr/bin/env python3
"""Allocation-only paired active canary for the static link-5/6 Poisson PSF.

The staged canary compares exactly two arms from one restored OSC-settled
state: ``joint_velocity_adapter_only`` and ``joint_velocity_psf_link56``.
Both consume the same 237 recorded actions from the completed historical
``pi0.5+AEGIS`` episode.  No policy server is contacted and no new OpenPI
query is made; this experiment therefore tests a Poisson filter *after* the
historical AEGIS action sequence, not raw pi0.5.

Real SafeLIBERO execution is refused outside a clean H100 Slurm allocation.
Unexpected apparatus failures publish a non-scientific run receipt.  Episode
``result.json`` is published only through the strict v2 scientific validator.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
try:
    from importlib import metadata
except ImportError:  # pragma: no cover - Python 3.8 allocation fallback
    import importlib_metadata as metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_CASE_ID = "vlsa-t1-goal-ii-t0-e05"
ARMS = ("joint_velocity_adapter_only", "joint_velocity_psf_link56")
EXPECTED_NUMERIC_SCHEMA = "vlsa_poisson_numeric_validation.v1"
EXPECTED_PARITY_SCHEMA = "vlsa_poisson_shadow_parity.v1"
EXPECTED_IDENTIFICATION_SCHEMA = "vlsa_poisson_shadow_identification.v2"
D_SIM_SEMANTICS = (
    "union_of_settled_live_solver_and_forwarded_post_state_nonpositive_contacts_"
    "plus_exact_obb_coverage_lower_bound"
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ActiveRunnerError(RuntimeError):
    pass


def _validated_run_output(output_root: Path, run_id: str) -> Path:
    if not isinstance(run_id, str) or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ActiveRunnerError(
            "run ID must be 1-128 portable alphanumeric/dot/underscore/hyphen characters"
        )
    raw_root = Path(output_root)
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ActiveRunnerError("output root must be an existing real directory")
    resolved_root = raw_root.resolve()
    output = resolved_root / run_id
    if output.parent != resolved_root or output.is_symlink():
        raise ActiveRunnerError("run output must remain inside the registered root")
    return output


class ArmTermination(RuntimeError):
    def __init__(self, completion_class: str, terminal_reason: str) -> None:
        super().__init__(terminal_reason)
        self.completion_class = str(completion_class)
        self.terminal_reason = str(terminal_reason)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ActiveRunnerError("%s is missing or symlinked" % label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ActiveRunnerError("%s is invalid JSON" % label) from error
    if not isinstance(value, dict):
        raise ActiveRunnerError("%s must contain one JSON object" % label)
    return value


def _git(root: Path) -> Dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root)] + list(arguments),
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()

    status = run("status", "--short")
    if status:
        raise ActiveRunnerError("scientific active execution requires a clean worktree")
    baseline = "1592aa59361f431ba96c6ddcbebcb596f6c20853"
    ancestor = subprocess.call(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", baseline, "HEAD"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestor != 0:
        raise ActiveRunnerError("registered Table-1 baseline is not an ancestor")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current")}


def _allocation() -> Dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isdigit():
        raise ActiveRunnerError("active SafeLIBERO canary requires a Slurm allocation")
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=30,
    ).strip().splitlines()
    if len(query) != 1:
        raise ActiveRunnerError("staged canary requires exactly one visible GPU")
    parts = [part.strip() for part in query[0].split(",")]
    if len(parts) != 3 or "H100" not in parts[0]:
        raise ActiveRunnerError("staged canary requires one H100 GPU")
    cuda_runtime = None
    try:
        import torch

        cuda_runtime = str(torch.version.cuda)
    except Exception:
        cuda_runtime = "unavailable_but_h100_verified_by_nvidia_smi"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    visible_ids = [value.strip() for value in visible.split(",") if value.strip()]
    return {
        "execution_environment": "slurm_allocation",
        "slurm_job_id": job_id,
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "slurm_step_id": os.environ.get("SLURM_STEP_ID", "batch"),
        "host_name": socket.gethostname(),
        "process_id": os.getpid(),
        "device": {
            "device_type": "cuda",
            "visible_device_ids": visible_ids or ["0"],
            "model": parts[0],
            "uuid": parts[1],
            "driver_version": parts[2],
            "cuda_runtime_version": cuda_runtime,
        },
    }


def _package_version(name: str) -> str:
    try:
        return str(metadata.version(name))
    except metadata.PackageNotFoundError:
        return "unavailable"


def _load_case(path: Path, case_id: str) -> Tuple[Dict[str, Any], str]:
    matches = []
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.strip():
                continue
            value = json.loads(raw)
            if value.get("case_id") == case_id:
                matches.append((value, _sha256(raw.rstrip(b"\r\n"))))
    if len(matches) != 1:
        raise ActiveRunnerError("expected exactly one manifest canary row")
    case, row_hash = matches[0]
    if (
        case.get("split") != "bringup_canary"
        or case.get("study_partition") != "development"
        or case.get("settle_actions") != 20
    ):
        raise ActiveRunnerError("active execution is restricted to the registered bring-up canary")
    return case, row_hash


def _checkpoint_identity(
    path: Path,
    *,
    expected_tree_sha256: str,
    expected_receipt_file_sha256: str,
    expected_schema_version: str,
) -> Dict[str, str]:
    receipt_file_sha256 = _file_sha256(path)
    if receipt_file_sha256 != expected_receipt_file_sha256:
        raise ActiveRunnerError(
            "checkpoint receipt is not the receipt frozen by the historical Table-1 run"
        )
    receipt = _json(path, "checkpoint receipt")
    if receipt.get("schema_version") != expected_schema_version:
        raise ActiveRunnerError("checkpoint receipt schema differs from source contract")
    expected = receipt.get("receipt_payload_sha256")
    payload = {key: value for key, value in receipt.items() if key != "receipt_payload_sha256"}
    if expected != _sha256(_canonical(payload)):
        raise ActiveRunnerError("checkpoint receipt payload hash differs")
    checkpoint = receipt.get("checkpoint")
    if (
        receipt.get("status") != "passed"
        or not isinstance(checkpoint, dict)
        or checkpoint.get("full_content_hash_verified") is not True
    ):
        raise ActiveRunnerError("checkpoint receipt is not a passed full-content receipt")
    norm = [
        row
        for row in checkpoint.get("metadata", [])
        if isinstance(row, dict)
        and row.get("relative_path")
        == "assets/physical-intelligence/libero/norm_stats.json"
    ]
    if len(norm) != 1:
        raise ActiveRunnerError("checkpoint receipt lacks the unique LIBERO norm statistics")
    if checkpoint.get("full_content_tree_sha256") != expected_tree_sha256:
        raise ActiveRunnerError(
            "checkpoint content tree differs from the historical Table-1 run contract"
        )
    return {
        "checkpoint_sha256": str(checkpoint["full_content_tree_sha256"]),
        "normalization_statistics_sha256": str(norm[0]["sha256"]),
        "receipt_payload_sha256": str(expected),
        "receipt_file_sha256": receipt_file_sha256,
        "execution_semantics": (
            "historical_pi05_source_identity_only_no_checkpoint_loaded_or_queried"
        ),
    }


def _load_historical_source_contract(
    historical_root: Path,
    source: Mapping[str, Any],
) -> Dict[str, Any]:
    historical = source.get("historical_results")
    if not isinstance(historical, Mapping):
        raise ActiveRunnerError("selection protocol lacks historical-results contract")
    relative = historical.get("source_run_contract_relative_path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or any(part in ("", ".", "..") for part in Path(relative).parts)
    ):
        raise ActiveRunnerError("historical run-contract relative path is invalid")
    path = historical_root / relative
    if path.is_symlink() or not path.is_file():
        raise ActiveRunnerError("historical run-contract is missing or symlinked")
    raw_sha256 = _file_sha256(path)
    if raw_sha256 != historical.get("source_run_contract_sha256"):
        raise ActiveRunnerError("historical run-contract SHA-256 differs")
    fields: Dict[str, str] = {}
    try:
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            parts = raw_line.split("\t")
            if len(parts) != 2 or not parts[0] or parts[0] in fields:
                raise ActiveRunnerError(
                    "historical run-contract line %d is invalid" % line_number
                )
            fields[parts[0]] = parts[1]
    except (OSError, UnicodeDecodeError) as error:
        raise ActiveRunnerError("historical run-contract is unreadable") from error
    expected = {
        "schema_version": "vlsa_table1_run_contract.v1",
        "run_id": source.get("table1_run_id"),
        "run_stage": "population",
        "git_commit": historical.get("source_git_commit"),
        "manifest_sha256": source.get("table1_manifest_sha256"),
        "pi05_tree_sha256": historical.get("pi05_tree_sha256"),
        "pi05_hash_receipt_sha256": historical.get(
            "pi05_hash_receipt_sha256"
        ),
    }
    for key, value in expected.items():
        if fields.get(key) != value:
            raise ActiveRunnerError(
                "historical run-contract differs at %s" % key
            )
    return {
        "relative_path": relative,
        "sha256": raw_sha256,
        "schema_version": fields["schema_version"],
        "run_id": fields["run_id"],
        "source_git_commit": fields["git_commit"],
        "pi05_tree_sha256": fields["pi05_tree_sha256"],
        "pi05_hash_receipt_sha256": fields["pi05_hash_receipt_sha256"],
    }


def _bound_historical_result_path(
    historical_root: Path,
    case: Mapping[str, Any],
) -> Path:
    record = case.get("historical_aegis_result")
    if not isinstance(record, Mapping):
        raise ActiveRunnerError("manifest case lacks historical AEGIS binding")
    relative = record.get("source_relative_path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or any(part in ("", ".", "..") for part in Path(relative).parts)
    ):
        raise ActiveRunnerError("manifest historical result path is invalid")
    if historical_root.is_symlink() or not historical_root.is_dir():
        raise ActiveRunnerError("historical result root is missing or symlinked")
    current = historical_root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ActiveRunnerError("historical result path traverses a symlink")
    if not current.is_file():
        raise ActiveRunnerError("manifest-bound historical result is missing")
    if _file_sha256(current) != record.get("raw_file_sha256"):
        raise ActiveRunnerError("manifest-bound historical result raw SHA-256 differs")
    return current


def _validate_replay_against_manifest(
    replay: Any,
    case: Mapping[str, Any],
) -> None:
    record = case.get("historical_aegis_result")
    if not isinstance(record, Mapping):
        raise ActiveRunnerError("manifest case lacks historical AEGIS binding")
    ledger = record.get("action_invariance_ledger")
    pairing = record.get("pairing")
    if not isinstance(ledger, Mapping) or not isinstance(pairing, Mapping):
        raise ActiveRunnerError("manifest historical action or pairing binding is absent")
    observed = {
        "result_payload_sha256": replay.result_payload_sha256,
        "raw_file_sha256": replay.result_file_sha256,
        "action_count": len(replay.actions),
        "ledger_action_count": len(replay.actions),
        "executed_sequence_sha256": replay.executed_sequence_sha256,
        "settled_simulator_state_sha256": replay.settled_simulator_state_sha256,
        "initial_observation_sha256": replay.initial_observation_sha256,
        "policy_noise_schedule_sha256": replay.policy_noise_schedule_sha256,
    }
    expected = {
        "result_payload_sha256": record.get("result_payload_sha256"),
        "raw_file_sha256": record.get("raw_file_sha256"),
        "action_count": record.get("action_count"),
        "ledger_action_count": ledger.get("action_count"),
        "executed_sequence_sha256": ledger.get("executed_sequence_sha256"),
        "settled_simulator_state_sha256": pairing.get(
            "settled_simulator_state_sha256"
        ),
        "initial_observation_sha256": pairing.get("initial_observation_sha256"),
        "policy_noise_schedule_sha256": pairing.get(
            "policy_noise_schedule_sha256"
        ),
    }
    if observed != expected:
        differences = sorted(
            key for key in expected if observed.get(key) != expected.get(key)
        )
        raise ActiveRunnerError(
            "historical replay differs from manifest binding at %s"
            % ", ".join(differences)
        )
    if replay.historical_task_success is not True:
        raise ActiveRunnerError("registered first canary must historically succeed")
    if not replay.steps or replay.steps[-1].done is not True:
        raise ActiveRunnerError(
            "registered 237-step source must end at its historical success terminal"
        )


def _require_true_acceptance(
    value: Any,
    required_fields: Sequence[str],
    label: str,
) -> None:
    if not isinstance(value, Mapping) or not all(
        value.get(field) is True for field in required_fields
    ):
        raise ActiveRunnerError("%s acceptance is incomplete" % label)


def _require_full_robot_sampling_evidence(
    resolved: Any,
    evidence: Any,
    *,
    roundtrip_field: str,
) -> None:
    from main.poisson_fullbody.surface_sampling import (
        validate_robot_sample_evidence,
    )

    if not isinstance(resolved, Mapping):
        raise ActiveRunnerError("authoritative resolved geometry evidence is absent")
    try:
        type_counts = validate_robot_sample_evidence(
            evidence,
            resolved_geom_ids=resolved["robot_geom_ids"],
            resolved_geom_names=resolved["robot_geom_names"],
            resolved_body_ids=resolved["robot_body_ids"],
            roundtrip_field=roundtrip_field,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ActiveRunnerError(
            "full-robot surface-sampling evidence is invalid: %s" % error
        ) from error
    if type_counts != {"mesh": 11, "box": 4, "cylinder": 1}:
        raise ActiveRunnerError(
            "first-canary robot collision geometry type counts changed"
        )
    records = evidence.get("geom_records")
    cylinder_records = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("geom_type_name") == "cylinder"
    ]
    if len(cylinder_records) != 1:
        raise ActiveRunnerError("first-canary pedestal cylinder is not unique")
    cylinder = cylinder_records[0]
    if (
        cylinder.get("geom_id") != 84
        or cylinder.get("geom_name") != "mount0_pedestal_col"
        or cylinder.get("geom_type_id") != 5
        or cylinder.get("geom_size") != [0.18, 0.31, 0.0]
    ):
        raise ActiveRunnerError(
            "first-canary pedestal-cylinder identity or dimensions changed"
        )


def _require_numeric_prerequisite(path: Path, *, source_commit: str) -> Dict[str, Any]:
    from main.poisson_fullbody.contracts import load_hashed_json

    result = load_hashed_json(path)
    if (
        result.get("schema_version") != EXPECTED_NUMERIC_SCHEMA
        or result.get("status") != "passed"
        or result.get("scientific_result") is not False
    ):
        raise ActiveRunnerError("numeric prerequisite has wrong schema or status")
    source = result.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("commit") != source_commit
        or source.get("status_short") != []
    ):
        raise ActiveRunnerError(
            "numeric prerequisite is not from this exact clean commit"
        )
    _require_true_acceptance(
        result.get("acceptance"),
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
        ),
        "numeric prerequisite",
    )
    tests = result.get("tests")
    if not isinstance(tests, Mapping) or tests.get("skip_count") != 0:
        raise ActiveRunnerError("numeric prerequisite did not retain zero-skip evidence")
    return result


def _require_identification_prerequisite(
    path: Path,
    *,
    case: Mapping[str, Any],
    case_row_hash: str,
    source_commit: str,
    manifest_sha256: str,
    selection_sha256: str,
    runtime_protocol_raw_sha256: str,
    runtime_protocol_semantic_sha256: str,
    runtime_parameter_block_sha256: str,
    alpha_gain_per_s: float,
    static_drift_thresholds: Mapping[str, float],
    replay: Any,
    parity: Mapping[str, Any],
) -> Dict[str, Any]:
    from main.poisson_fullbody.contracts import load_hashed_json
    from main.poisson_fullbody.shadow_identification import (
        ShadowIdentificationError,
        registered_filter_update_available_before_contact,
        rollout_contact_boundary_index,
        validate_shadow_replay_record,
    )

    result = load_hashed_json(path)
    if (
        result.get("schema_version") != EXPECTED_IDENTIFICATION_SCHEMA
        or result.get("status") != "passed"
        or result.get("scientific_result") is not False
        or result.get("case_id") != case["case_id"]
    ):
        raise ActiveRunnerError(
            "shadow-identification prerequisite has wrong schema, status, or case"
        )
    provenance = result.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ActiveRunnerError("shadow-identification provenance is absent")
    source = provenance.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("commit") != source_commit
        or source.get("status_short") != []
    ):
        raise ActiveRunnerError(
            "shadow-identification prerequisite is not from this exact clean commit"
        )
    expected = {
        "manifest_sha256": manifest_sha256,
        "manifest_row_sha256": case_row_hash,
        "selection_config_sha256": selection_sha256,
        "runtime_protocol_raw_sha256": runtime_protocol_raw_sha256,
        "runtime_protocol_semantic_sha256": runtime_protocol_semantic_sha256,
        "runtime_parameter_block_sha256": runtime_parameter_block_sha256,
        "historical_result_file_sha256": replay.result_file_sha256,
        "historical_result_payload_sha256": replay.result_payload_sha256,
        "upstream_parity_payload_sha256": parity["result_payload_sha256"],
    }
    differences = sorted(
        field for field, value in expected.items() if provenance.get(field) != value
    )
    if differences:
        raise ActiveRunnerError(
            "shadow-identification binding differs at %s"
            % ", ".join(differences)
        )
    if result.get("historical") != replay.provenance():
        raise ActiveRunnerError(
            "shadow-identification historical replay identity differs"
        )
    _require_true_acceptance(
        result.get("acceptance"),
        (
            "upstream_exact_parity_same_clean_commit",
            "all_237_historical_actions_executed",
            "all_5925_callbacks_observed",
            "historical_state_reward_done_goal_exact",
            "upstream_observation_sequence_exact",
            "complete_mujoco_integration_state_unchanged_by_construction",
            "complete_mujoco_integration_state_unchanged_by_callback",
            "static_queries_stop_at_first_registered_drift",
            "contact_authority_is_mujoco_nonpositive_distance",
            "no_action_or_control_mutation",
            "no_active_safety_efficacy_claim",
        ),
        "shadow-identification prerequisite",
    )
    shadow = result.get("shadow_replay")
    expected_callbacks = 25 * len(replay.actions)
    if (
        not isinstance(shadow, Mapping)
        or shadow.get("executed_action_count") != len(replay.actions)
        or shadow.get("callback_count") != expected_callbacks
        or shadow.get("expected_callback_count") != expected_callbacks
    ):
        raise ActiveRunnerError(
            "shadow-identification prerequisite exposure is incomplete"
        )
    parity_callback = parity.get("callback_replay")
    if (
        not isinstance(parity_callback, Mapping)
        or shadow.get("state_sequence_sha256")
        != parity_callback.get("state_sequence_sha256")
        or shadow.get("observation_sequence_sha256")
        != parity_callback.get("observation_sequence_sha256")
    ):
        raise ActiveRunnerError(
            "shadow-identification replay hashes differ from exact parity"
        )
    try:
        validate_shadow_replay_record(
            shadow,
            action_count=len(replay.actions),
            inner_updates_per_high_level_action=5,
            physics_substeps_per_inner_update=5,
            physics_timestep_s=0.002,
            contact_definition="mujoco_contact_dist_le_0",
            alpha_gain_per_s=alpha_gain_per_s,
            static_drift_thresholds=static_drift_thresholds,
        )
    except ShadowIdentificationError as error:
        raise ActiveRunnerError(
            "shadow-identification serialized replay is invalid: %s" % error
        ) from error
    construction = shadow.get("construction")
    if not isinstance(construction, Mapping):
        raise ActiveRunnerError(
            "shadow-identification construction evidence is absent"
        )
    _require_full_robot_sampling_evidence(
        construction.get("resolved_geometry"),
        construction.get("full_robot_measurement_sampling"),
        roundtrip_field="rigid_roundtrip",
    )
    identification = shadow.get("poisson_identification")
    assessment = (
        identification.get("contact_prediction_assessment")
        if isinstance(identification, Mapping)
        else None
    )
    if (
        not isinstance(assessment, Mapping)
        or assessment.get("assessment")
        != "registered_warning_preceded_link56_contact"
    ):
        raise ActiveRunnerError(
            "shadow identification did not establish a pre-contact registered warning"
        )
    lead_substeps = assessment.get("lead_physics_substeps")
    lead_time = assessment.get("lead_time_s")
    primary = assessment.get("primary_registered_warning")
    contact = assessment.get("first_link56_contact")
    if (
        isinstance(lead_substeps, bool)
        or not isinstance(lead_substeps, int)
        or lead_substeps <= 0
        or isinstance(lead_time, bool)
        or not isinstance(lead_time, (int, float))
        or not math.isfinite(float(lead_time))
        or float(lead_time) <= 0.0
        or not math.isclose(
            float(lead_time), float(lead_substeps) * 0.002, rel_tol=0.0, abs_tol=1e-12
        )
        or not isinstance(primary, Mapping)
        or not isinstance(contact, Mapping)
        or primary.get("geom_id") != contact.get("robot_geom_id")
        or int(primary.get("observation_index", -1)) + lead_substeps
        != rollout_contact_boundary_index(
            int(contact.get("observation_index", -2)),
            str(contact.get("source_phase")),
        )
    ):
        raise ActiveRunnerError(
            "shadow warning lead or registered contact-geom identity is invalid"
        )
    if not registered_filter_update_available_before_contact(
        warning_observation_index=int(primary["observation_index"]),
        contact_observation_index=int(contact["observation_index"]),
        contact_source_phase=str(contact["source_phase"]),
        physics_substeps_per_filter_update=5,
    ):
        raise ActiveRunnerError(
            "shadow warning leaves no scheduled 100 Hz filter update before contact"
        )
    return result


def _require_identification_matches_active_construction(
    identification_prerequisite: Mapping[str, Any],
    *,
    case: Mapping[str, Any],
    active_obstacle_name: str,
    contact_model_authority_sha256: str,
    robot_root_body_name: str,
    robot_root_body_ids: Sequence[int],
    arm_dof_indices: Sequence[int],
    resolved_geometry: Mapping[str, Any],
    field_bundle: Any,
    full_robot_sampling: Mapping[str, Any],
    settled_integration_state_sha256: str,
    settled_integration_state_length: int,
) -> None:
    """Bind shadow authorization to the exact field rebuilt for active physics."""

    shadow = identification_prerequisite.get("shadow_replay")
    construction = shadow.get("construction") if isinstance(shadow, Mapping) else None
    if not isinstance(construction, Mapping):
        raise ActiveRunnerError(
            "shadow-identification construction evidence is absent"
        )
    expected_obstacle = case.get("active_obstacle_name")
    expected_field_bundle = {
        "protocol_id": field_bundle.protocol_id,
        "protected_body_ids": list(field_bundle.protected_body_ids),
        "protected_body_names": list(field_bundle.protected_body_names),
        "diagnostics": asdict(field_bundle.diagnostics),
        "hashes": asdict(field_bundle.hashes),
        "surface_components": [
            asdict(value) for value in field_bundle.protected_samples.components
        ],
        "protected_sample_count": len(field_bundle.protected_samples.samples),
    }
    expected_sampling = dict(full_robot_sampling)
    roundtrip = expected_sampling.pop("roundtrip", None)
    if roundtrip is None:
        raise ActiveRunnerError(
            "active full-robot sampling omitted its rigid roundtrip audit"
        )
    expected_sampling["rigid_roundtrip"] = roundtrip
    expected = {
        "active_obstacle_name": expected_obstacle,
        "contact_model_authority_sha256": contact_model_authority_sha256,
        "robot_root_body_name": robot_root_body_name,
        "robot_root_body_ids": list(robot_root_body_ids),
        "arm_dof_indices": list(arm_dof_indices),
        "resolved_geometry": dict(resolved_geometry),
        "field_bundle": expected_field_bundle,
        "full_robot_measurement_sampling": expected_sampling,
    }
    if active_obstacle_name != expected_obstacle:
        raise ActiveRunnerError(
            "active selected obstacle differs from the immutable manifest"
        )
    differences = [
        field
        for field, value in expected.items()
        if _canonical(construction.get(field)) != _canonical(value)
    ]
    if differences:
        raise ActiveRunnerError(
            "shadow identification differs from active construction at %s"
            % ", ".join(sorted(differences))
        )
    read_only = construction.get("complete_integration_state_read_only_audit")
    if not isinstance(read_only, Mapping):
        raise ActiveRunnerError(
            "shadow construction read-only state audit is absent"
        )
    if (
        read_only.get("mujoco_state_specification") != "mjSTATE_INTEGRATION"
        or read_only.get("state_vector_length")
        != int(settled_integration_state_length)
        or read_only.get("before_sha256") != settled_integration_state_sha256
        or read_only.get("after_sha256") != settled_integration_state_sha256
        or read_only.get("exact_array_equal") is not True
    ):
        raise ActiveRunnerError(
            "shadow identification and active settled MuJoCo state differ"
        )


RESUME_IDENTITY_FIELDS = (
    "run_id",
    "protocol_id",
    "case_id",
    "arm",
    "provenance.code_commit",
    "provenance.code_dirty",
    "provenance.run_contract_sha256",
    "provenance.manifest_sha256",
    "provenance.manifest_record_sha256",
    "provenance.protocol_config_sha256",
    "provenance.runtime_protocol_raw_sha256",
    "provenance.runtime_protocol_semantic_sha256",
    "case_identity.manifest_case_ordinal",
    "case_identity.initial_states_sha256",
    "case_identity.bddl_sha256",
    "case_identity.active_obstacle_name",
    "case_identity.registered_source_exposure_high_level_steps",
    "pairing.source_settled_observation_sha256",
    "pairing.policy_noise_schedule_sha256",
    "pairing.policy_query_schedule_sha256",
    "pairing.nominal_high_level_action_ledger_sha256",
    "pairing.source_exposure_high_level_steps",
    "runtime.protocol_parameter_block_sha256",
    "runtime.model.model_configuration_sha256",
    "runtime.model.checkpoint_sha256",
    "runtime.model.normalization_statistics_sha256",
    "runtime.sampler.configuration_sha256",
    "runtime.sampler.source_policy_query_count",
    "runtime.sampler.active_policy_query_count",
    "runtime.sampler.active_policy_rng_exercised",
)
def _expected_resume_identity(
    *,
    root: Path,
    run_id: str,
    case: Mapping[str, Any],
    case_row_hash: str,
    manifest_path: Path,
    selection_path: Path,
    runtime_protocol_path: Path,
    protocol_hashes: Any,
    source_git: Mapping[str, Any],
    checkpoint: Mapping[str, str],
    replay: Any,
    field_sha256: str,
    run_contract_sha256: str,
    arm: str,
) -> Dict[str, Any]:
    from main.poisson_fullbody.result_schema import PROTOCOL_ID

    model_config = {
        "checkpoint_receipt_payload_sha256": checkpoint["receipt_payload_sha256"],
        "checkpoint_receipt_file_sha256": checkpoint["receipt_file_sha256"],
        "source_historical_arm": replay.arm,
        "execution_semantics": checkpoint["execution_semantics"],
    }
    sampler_config = {
        "source": replay.provenance(),
        "active_policy_queries": 0,
        "active_policy_rng_exercised": False,
    }
    return {
        "run_id": run_id,
        "protocol_id": PROTOCOL_ID,
        "case_id": case["case_id"],
        "arm": arm,
        "provenance": {
            "code_commit": source_git["commit"],
            "code_dirty": False,
            "run_contract_sha256": run_contract_sha256,
            "manifest_sha256": _file_sha256(manifest_path),
            "manifest_record_sha256": case_row_hash,
            "protocol_config_sha256": _file_sha256(selection_path),
            "runtime_protocol_raw_sha256": _file_sha256(runtime_protocol_path),
            "runtime_protocol_semantic_sha256": protocol_hashes.protocol_sha256,
        },
        "case_identity": {
            "manifest_case_ordinal": int(case["case_ordinal"]),
            "initial_states_sha256": case["initial_states_sha256"],
            "bddl_sha256": case["bddl_sha256"],
            "active_obstacle_name": case["active_obstacle_name"],
            "registered_source_exposure_high_level_steps": len(replay.actions),
        },
        "pairing": {
            "source_settled_observation_sha256": replay.initial_observation_sha256,
            "policy_noise_schedule_sha256": replay.policy_noise_schedule_sha256,
            "policy_query_schedule_sha256": replay.source_policy_query_schedule_sha256,
            "nominal_high_level_action_ledger_sha256": replay.executed_sequence_sha256,
            "field_sha256": field_sha256,
            "source_exposure_high_level_steps": len(replay.actions),
        },
        "runtime": {
            "protocol_parameter_block_sha256": protocol_hashes.parameter_block_sha256,
            "model": {
                "model_configuration_sha256": _sha256(_canonical(model_config)),
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "normalization_statistics_sha256": checkpoint[
                    "normalization_statistics_sha256"
                ],
            },
            "sampler": {
                "configuration_sha256": _sha256(_canonical(sampler_config)),
                "source_policy_query_count": replay.source_policy_query_count,
                "active_policy_query_count": 0,
                "active_policy_rng_exercised": False,
            },
        },
    }


def _validate_complete_run_receipt(
    *,
    receipt_path: Path,
    output: Path,
    expected_receipt_identity: Mapping[str, Any],
    expected_arm_identities: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    from main.poisson_fullbody.contracts import (
        final_is_resumable,
        load_hashed_json,
        sha256_file,
    )
    from main.poisson_fullbody.result_schema import validate_active_canary_pair

    receipt = load_hashed_json(receipt_path)
    if receipt.get("status") == "failed":
        raise ActiveRunnerError(
            "this run ID already has a failed immutable receipt; choose a new run ID"
        )
    if (
        receipt.get("schema_version")
        != "vlsa_poisson_active_canary_run_receipt.v2"
        or receipt.get("status") != "complete"
        or receipt.get("scientific_result") is not False
    ):
        raise ActiveRunnerError(
            "existing run receipt is not a reusable complete v2 receipt"
        )
    if receipt.get("identity") != expected_receipt_identity:
        raise ActiveRunnerError(
            "existing complete run receipt differs from current full identity"
        )
    results: Dict[str, Any] = {}
    arm_records = receipt.get("arm_results")
    if not isinstance(arm_records, Mapping) or set(arm_records) != set(ARMS):
        raise ActiveRunnerError("complete receipt arm-result inventory differs")
    for arm in ARMS:
        expected_relative = "%s/%s/result.json" % (
            expected_receipt_identity["case_id"],
            arm,
        )
        record = arm_records.get(arm)
        if not isinstance(record, Mapping) or record.get("relative_path") != expected_relative:
            raise ActiveRunnerError("complete receipt arm path differs for %s" % arm)
        final_path = output / expected_relative
        resumable, reason = final_is_resumable(
            final_path,
            expected_arm_identities[arm],
            identity_fields=RESUME_IDENTITY_FIELDS,
            artifact_root=output,
        )
        if not resumable:
            raise ActiveRunnerError(
                "complete receipt arm %s is not resumable: %s" % (arm, reason)
            )
        if sha256_file(final_path) != record.get("sha256"):
            raise ActiveRunnerError("complete receipt arm file hash differs for %s" % arm)
        results[arm] = load_hashed_json(final_path)
    pair_relative = "%s/pair_result.json" % expected_receipt_identity["case_id"]
    if receipt.get("pair_result_relative_path") != pair_relative:
        raise ActiveRunnerError("complete receipt pair-result path differs")
    pair_path = output / pair_relative
    pair = load_hashed_json(pair_path)
    if sha256_file(pair_path) != receipt.get("pair_result_sha256"):
        raise ActiveRunnerError("complete receipt pair-result file hash differs")
    expected_pair = dict(
        validate_active_canary_pair(
            results["joint_velocity_adapter_only"],
            results["joint_velocity_psf_link56"],
        )
    )
    expected_pair.update(
        {
            "status": "complete",
            "scientific_result": False,
            "four_arm_109_case_study_complete": False,
            "run_contract_sha256": expected_receipt_identity[
                "run_contract_sha256"
            ],
        }
    )
    observed_pair_without_hash = {
        key: value for key, value in pair.items() if key != "result_payload_sha256"
    }
    if observed_pair_without_hash != expected_pair:
        raise ActiveRunnerError("complete receipt pair-result payload differs")
    return results


def _historical_result_path(root: Path, relative_pattern: str, case_id: str) -> Path:
    relative = relative_pattern.format(case_id=case_id)
    direct = root / relative
    if direct.is_file() and not direct.is_symlink():
        return direct
    matches = list(root.glob("tasks/task-*/results/aegis/%s/result.json" % case_id))
    if len(matches) != 1:
        raise ActiveRunnerError("historical result root does not resolve one AEGIS result")
    return matches[0]


def _measurement(value: Optional[float], units: str, reason: str) -> Dict[str, Any]:
    if value is None:
        return {"available": False, "value": None, "units": units, "reason": reason}
    return {"available": True, "value": float(value), "units": units, "reason": None}


def _raw_model_data(sim: Any) -> Tuple[Any, Any]:
    return (
        getattr(sim.model, "_model", sim.model),
        getattr(sim.data, "_data", sim.data),
    )


def _official_integration_vector(sim: Any) -> Any:
    import mujoco
    import numpy as np

    model, data = _raw_model_data(sim)
    specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    value = np.empty(
        int(mujoco.mj_stateSize(model, specification)), dtype=np.float64
    )
    mujoco.mj_getState(model, data, value, specification)
    return value


def _eef_site_id(env: Any) -> int:
    robot = env.robots[0]
    candidates = []
    value = getattr(robot, "eef_site_id", None)
    if isinstance(value, int) and not isinstance(value, bool):
        candidates.append(int(value))
    sites = getattr(getattr(robot, "gripper", None), "important_sites", None)
    if isinstance(sites, Mapping):
        name = sites.get("grip_site")
        if isinstance(name, str) and name:
            try:
                candidates.append(int(env.sim.model.site_name2id(name)))
            except Exception:
                pass
    candidates = sorted(set(value for value in candidates if value >= 0))
    if len(candidates) != 1:
        raise ActiveRunnerError("cannot resolve one authoritative Panda grip site")
    return candidates[0]


def _eef_kinematics(env: Any, site_id: int, arm_dof_indices: Sequence[int]) -> Tuple[Any, Any, Any]:
    import mujoco
    import numpy as np

    model, data = _raw_model_data(env.sim)
    position = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    rotation = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
    jac_pos = np.zeros((3, int(model.nv)), dtype=np.float64)
    jac_rot = np.zeros((3, int(model.nv)), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jac_pos, jac_rot, int(site_id))
    indexes = np.asarray(arm_dof_indices, dtype=np.int64)
    jacobian = np.vstack((jac_pos[:, indexes], jac_rot[:, indexes]))
    if jacobian.shape != (6, 7) or not np.all(np.isfinite(jacobian)):
        raise ActiveRunnerError("MuJoCo grip-site Jacobian is invalid")
    return position, rotation, jacobian


def _joint_limits(env: Any) -> Tuple[Any, Any]:
    import numpy as np

    robot = env.robots[0]
    joint_ids = getattr(robot, "_ref_joint_indexes", None)
    if joint_ids is None or len(joint_ids) != 7:
        raise ActiveRunnerError("Panda joint IDs are unavailable")
    model, _ = _raw_model_data(env.sim)
    ranges = np.asarray(model.jnt_range[np.asarray(joint_ids, dtype=np.int64)], dtype=np.float64)
    if ranges.shape != (7, 2) or np.any(ranges[:, 0] >= ranges[:, 1]):
        raise ActiveRunnerError("Panda joint ranges are invalid")
    return ranges[:, 0].copy(), ranges[:, 1].copy()


def _eef_jacobian_audit(
    env: Any,
    site_id: int,
    arm_qpos_indices: Sequence[int],
    arm_dof_indices: Sequence[int],
) -> Dict[str, Any]:
    """Finite-difference the linear rows on a separate integration-state clone."""

    import mujoco
    import numpy as np
    from main.poisson_fullbody.measurement import copy_integration_state

    model, data = _raw_model_data(env.sim)
    clone = copy_integration_state(model, data)
    mujoco.mj_forward(model, clone)
    jac_pos = np.zeros((3, int(model.nv)), dtype=np.float64)
    jac_rot = np.zeros((3, int(model.nv)), dtype=np.float64)
    mujoco.mj_jacSite(model, clone, jac_pos, jac_rot, int(site_id))
    analytic = jac_pos[:, np.asarray(arm_dof_indices, dtype=np.int64)]
    original = np.asarray(clone.qpos, dtype=np.float64).copy()
    numerical = np.zeros((3, 7), dtype=np.float64)
    delta = 1e-6
    for column, qpos_index in enumerate(arm_qpos_indices):
        clone.qpos[:] = original
        clone.qpos[int(qpos_index)] += delta
        mujoco.mj_forward(model, clone)
        plus = np.asarray(clone.site_xpos[site_id], dtype=np.float64).copy()
        clone.qpos[:] = original
        clone.qpos[int(qpos_index)] -= delta
        mujoco.mj_forward(model, clone)
        minus = np.asarray(clone.site_xpos[site_id], dtype=np.float64).copy()
        numerical[:, column] = (plus - minus) / (2.0 * delta)
    difference = analytic - numerical
    maximum = float(np.max(np.abs(difference)))
    site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, int(site_id))
    if not site_name:
        raise ActiveRunnerError("authoritative grip site has no MuJoCo name")
    record = {
        "site_id": int(site_id),
        "site_name": str(site_name),
        "jacobian_row_order": "linear_xyz_then_angular_xyz",
        "shape": [6, 7],
        "linear_finite_difference_delta_rad": delta,
        "maximum_linear_jacobian_error_m_per_rad": maximum,
        "absolute_tolerance_m_per_rad": 2e-6,
        "passed": bool(maximum <= 2e-6),
        "live_state_mutated": False,
    }
    if not record["passed"]:
        raise ActiveRunnerError("grip-site Jacobian finite-difference audit failed")
    return record


def _forwarded_eef_position(env: Any, site_id: int) -> Any:
    import numpy as np
    from main.poisson_fullbody.measurement import clone_forwarded_state

    model, data = _raw_model_data(env.sim)
    clone = clone_forwarded_state(model, data)
    return np.asarray(clone.site_xpos[int(site_id)], dtype=np.float64).copy()


def _monitor_summary(monitor: Any, physics_substeps: int) -> Dict[str, Any]:
    if physics_substeps:
        measured = monitor.result()
        settled = measured.settled_state
        drift = measured.obstacle_pose_drift
        return {
            "D_sim_min_m": float(measured.D_sim_min_m),
            "any_contact": bool(measured.any_robot_obstacle_contact),
            "link56_contact": bool(measured.link56_obstacle_contact),
            "total": int(measured.total_physical_contact_point_record_count),
            "settled": int(settled.physical_contact_point_record_count),
            "rollout": int(measured.rollout_phase_physical_contact_point_record_count),
            "live": int(measured.live_solver_nonpositive_contact_point_record_count),
            "post": int(measured.post_state_physical_contact_point_record_count),
            "first": None if measured.first_physical_contact_point_record is None else measured.first_physical_contact_point_record.to_dict(),
            "first_settled": None if settled.first_physical_contact_point_record is None else settled.first_physical_contact_point_record.to_dict(),
            "first_live": None if measured.first_live_solver_physical_contact_point_record is None else measured.first_live_solver_physical_contact_point_record.to_dict(),
            "first_post": None if measured.first_post_state_physical_contact_point_record is None else measured.first_post_state_physical_contact_point_record.to_dict(),
            "translation_drift_m": float(drift.maximum_translation_m),
            "rotation_drift_rad": float(drift.maximum_rotation_rad),
            "surface_drift_m": float(drift.maximum_surface_point_displacement_m),
            "record": measured.to_dict(),
        }
    settled = monitor.settled_state
    count = int(settled.physical_contact_point_record_count)
    first = None if settled.first_physical_contact_point_record is None else settled.first_physical_contact_point_record.to_dict()
    return {
        "D_sim_min_m": float(settled.D_sim_m),
        "any_contact": bool(settled.any_robot_obstacle_contact),
        "link56_contact": bool(settled.link56_obstacle_contact),
        "total": count,
        "settled": count,
        "rollout": 0,
        "live": 0,
        "post": 0,
        "first": first,
        "first_settled": first,
        "first_live": None,
        "first_post": None,
        "translation_drift_m": 0.0,
        "rotation_drift_rad": 0.0,
        "surface_drift_m": 0.0,
        "record": {"settled_state": settled.to_dict(), "observed_physics_substeps": 0},
    }


def _run_arm(
    *,
    arm: str,
    evaluator: Any,
    runtime: Mapping[str, Any],
    case: Mapping[str, Any],
    replay: Any,
    source_env: Any,
    source_settled_observation: Mapping[str, Any],
    settled_state: Any,
    bundle: Any,
    resolved: Any,
    full_samples: Any,
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    import numpy as np
    from main.poisson_fullbody.active_canary import (
        ControllerTrackingInvalid,
        PrefixCounters,
        TrackingAudit,
        classify_filter_failure,
        paper_car_endpoint,
    )
    from main.poisson_fullbody.cbf_qp import HardCbfQp, joint_velocity_bounds
    from main.poisson_fullbody.controller_bridge import (
        restore_osc_settled_state_into_joint_velocity_env,
    )
    from main.poisson_fullbody.geometry import minimum_point_to_oriented_boxes_distance
    from main.poisson_fullbody.jacobians import evaluate_point_jacobians
    from main.poisson_fullbody.joint_velocity_adapter import (
        TranslationalJointVelocityAdapter,
        normalized_joint_velocity_action,
    )
    from main.poisson_fullbody.measurement import (
        FullRobotObstacleMonitor,
        StaticObstacleDriftInadmissible,
        clone_forwarded_state,
    )
    from scripts.run_poisson_shadow_parity import _observation_sha256

    env = None
    try:
        env, task, initial_observation, selected_initial_state = evaluator._build_environment(
            runtime,
            case,
            render_resolution=evaluator.TABLE_RENDER_RESOLUTION,
            controller="JOINT_VELOCITY",
            control_frequency_hz=100,
        )
        initial_proxy = evaluator._eef_proxy(runtime, initial_observation)
        evaluator._update_eef_marker(env, initial_proxy)
        admissibility = protocol["admissibility"]
        restore = restore_osc_settled_state_into_joint_velocity_env(
            source_env,
            env,
            settled_state,
            max_arm_qpos_error_rad=float(admissibility["max_state_restore_qpos_error_rad"]),
            max_arm_qvel_error_rad_s=float(admissibility["max_state_restore_qvel_error_rad_s"]),
        )
        if (
            not restore["official_integration_state_available"]
            or restore["official_integration_state_sha256"]
            != restore["target_official_integration_state_sha256"]
        ):
            raise ActiveRunnerError("target did not receive the complete official settled state")
        _, target_raw_data = _raw_model_data(env.sim)
        settled_simulator_time_s = float(target_raw_data.time)
        observation = env.env._get_observations()
        active_initial_observation_sha256 = _observation_sha256(
            observation, evaluator, np
        )
        active_name, _ = evaluator._active_obstacle(env, observation)
        if active_name != case["active_obstacle_name"]:
            raise ActiveRunnerError("target active obstacle differs after restore")
        obstacle_position_key = active_name + "_pos"
        source_settled_obstacle_position = np.asarray(
            source_settled_observation[obstacle_position_key], dtype=np.float64
        ).copy()
        target_settled_obstacle_position = np.asarray(
            observation[obstacle_position_key], dtype=np.float64
        ).copy()
        source_position_sha256 = evaluator.array_sha256(
            source_settled_obstacle_position
        )
        target_position_sha256 = evaluator.array_sha256(
            target_settled_obstacle_position
        )
        if source_position_sha256 != replay.settled_active_obstacle_position_sha256:
            raise ActiveRunnerError(
                "source settled active-obstacle position differs from historical pairing"
            )
        if (
            target_position_sha256 != source_position_sha256
            or not np.array_equal(
                target_settled_obstacle_position,
                source_settled_obstacle_position,
            )
        ):
            raise ActiveRunnerError(
                "target settled active-obstacle position differs from source"
            )
        arm_dofs = tuple(int(value) for value in env.robots[0]._ref_joint_vel_indexes)
        arm_qpos = tuple(int(value) for value in env.robots[0]._ref_joint_pos_indexes)
        if len(arm_dofs) != 7 or len(arm_qpos) != 7:
            raise ActiveRunnerError("Panda arm index contract differs")
        site_id = _eef_site_id(env)
        q_min, q_max = _joint_limits(env)
        eef_jacobian_audit = _eef_jacobian_audit(
            env, site_id, arm_qpos, arm_dofs
        )
        safety = protocol["safety"]
        monitor = FullRobotObstacleMonitor(
            env.sim,
            resolved,
            full_samples.samples,
            certified_coverage_radius_m=full_samples.maximum_surface_cover_radius_m,
            max_selected_geom_surface_drift_m=float(admissibility["max_selected_geom_surface_drift_m"]),
            max_selected_geom_translation_drift_m=float(admissibility["max_selected_geom_translation_drift_m"]),
            max_selected_geom_rotation_drift_rad=float(admissibility["max_selected_geom_rotation_drift_rad"]),
            max_settled_obstacle_linear_speed_m_per_s=float(admissibility["max_selected_body_linear_speed_m_s"]),
            max_settled_obstacle_angular_speed_rad_per_s=float(admissibility["max_selected_body_angular_speed_rad_s"]),
            require_settled_static_motion=(arm != "joint_velocity_adapter_only"),
            terminate_on_static_drift=(arm != "joint_velocity_adapter_only"),
            near_contact_tolerance_m=float(safety["contact_margin_m"]),
            inner_updates_per_high_level_action=5,
            physics_substeps_per_inner_update=5,
        )
        settled_motion_admissible = bool(
            monitor.settled_state.obstacle_motion_admissibility.admissible
        )
        counters = PrefixCounters()
        tracking = TrackingAudit(
            maximum_linf_rad_s=float(admissibility["max_joint_velocity_tracking_linf_rad_s"]),
            maximum_rmse_rad_s=float(admissibility["max_joint_velocity_tracking_rmse_rad_s"]),
        )
        adapter_cfg = protocol["adapter"]
        adapter = TranslationalJointVelocityAdapter(
            damping=float(adapter_cfg["dls_damping"]),
            position_gain_s_inv=float(adapter_cfg["position_gain_per_s"]),
            orientation_gain_s_inv=float(adapter_cfg["orientation_gain_per_s"]),
        )
        qp_cfg = protocol["qp"]
        qp = HardCbfQp(
            eps_abs=float(qp_cfg["eps_abs"]),
            eps_rel=float(qp_cfg["eps_rel"]),
            max_iter=int(qp_cfg["max_iterations"]),
            postcheck_cbf_tolerance=float(qp_cfg["postcheck_cbf_tolerance"]),
            postcheck_bound_tolerance=float(qp_cfg["postcheck_bound_tolerance_rad_s"]),
        )
        goal_definition, goal_atoms = evaluator._goal_progress_definition(env)
        initial_goal = evaluator._goal_progress_snapshot(
            env, goal_atoms, step=-1, previous_values=None
        )
        previous_goal_values = initial_goal["values"]
        goal_ledger = [dict(initial_goal, snapshot_kind="settled_pre_action")]
        task_latched = bool(initial_goal["all_satisfied"])
        initial_task_success = bool(initial_goal["all_satisfied"])
        first_task_success_index = None
        terminal_task_success = initial_task_success
        terminal_goal_fraction = float(initial_goal["fraction"])
        maximum_goal_fraction = float(initial_goal["fraction"])
        goal_regressions = 0
        horizon = int(getattr(env.env, "horizon", 0))
        ignore_done = bool(getattr(env.env, "ignore_done", False))
        if (
            not ignore_done
            and horizon <= int(env.env.timestep) + len(replay.actions)
        ):
            raise ActiveRunnerError(
                "robosuite task horizon cannot preserve the registered source exposure"
            )
        settled_obstacle_position = target_settled_obstacle_position.copy()
        post_step_obstacle_positions = []

        # Shared field safe-start is checked before either Poisson physics
        # update. Adapter-only records the same preflight but does not use it
        # as a control-law gate.
        initial_points, _ = evaluate_point_jacobians(
            env.sim.model,
            env.sim.data,
            bundle.protected_samples.samples,
            arm_dofs,
        )
        initial_queries = [bundle.field.query(point) for point in initial_points]
        initial_valid_h = [
            float(query.value)
            for query in initial_queries
            if query.valid and query.value is not None
        ]
        safe_start = bool(
            initial_queries
            and all(
                query.valid
                and query.value is not None
                and float(query.value) > 0.0
                for query in initial_queries
            )
        )
        completion = "executed"
        terminal_reason = "registered_source_exposure_completed"
        if arm == "joint_velocity_psf_link56" and not settled_motion_admissible:
            completion = "static_obstacle_inadmissible"
            terminal_reason = "settled_selected_obstacle_motion_inadmissible"
        elif arm == "joint_velocity_psf_link56" and not safe_start:
            completion = "safe_start_inadmissible"
            terminal_reason = "initial_link56_poisson_samples_not_strictly_positive"

        entered_source_actions: List[List[float]] = []
        completed_source_actions: List[List[float]] = []
        nominal_ledger: List[Dict[str, Any]] = []
        executed_ledger: List[Dict[str, Any]] = []
        inner_trace: List[Dict[str, Any]] = []
        physics_trace: List[Dict[str, Any]] = []
        fail_closed_attempt_trace: List[Dict[str, Any]] = []
        realized_cbf_trace: List[Dict[str, Any]] = []
        nominal_motion = 0.0
        safe_motion = 0.0
        measured_motion = 0.0
        correction_motion = 0.0
        zero_commands = 0
        exposed_command_keys = set()
        pending_command = {"nominal": None, "executed": None, "key": None}
        eef_path = 0.0
        previous_eef_position = _forwarded_eef_position(env, site_id)
        settled_eef_position = previous_eef_position.copy()
        initial_h_minimum = (
            min(initial_valid_h)
            if len(initial_valid_h) == len(initial_queries) and initial_valid_h
            else None
        )
        h_minimum = initial_h_minimum
        initial_dopt_minimum = min(
            minimum_point_to_oriented_boxes_distance(point, bundle.obstacle_boxes)
            for point in initial_points
        )
        dopt_minimum = initial_dopt_minimum
        nominal_residual_minimum = None
        safe_raw_residual_minimum = None
        safe_normalized_residual_minimum = None
        realized_residual_minimum = None
        first_negative_realized_residual = None
        negative_realized_residual_count = 0
        realized_residual_evaluation_count = 0
        invalid_field_queries = 0
        nonpositive_runtime_field_queries = 0
        optimizer_counts = {
            "attempt_count": 0,
            "solved_count": 0,
            "infeasible_count": 0,
            "solver_failure_count": 0,
            "postcheck_failure_count": 0,
        }

        if completion == "executed":
            try:
                for high_index, source_action_tuple in enumerate(replay.actions):
                    source_action = np.asarray(source_action_tuple, dtype=np.float64)
                    counters.enter_high_level(high_index)
                    entered_source_actions.append(source_action.tolist())
                    position, rotation, _ = _eef_kinematics(env, site_id, arm_dofs)
                    adapter.begin_high_level_action(source_action, position, rotation)

                    def provider(sim: Any, inner_index: int) -> Any:
                        nonlocal nominal_motion, safe_motion, correction_motion
                        nonlocal zero_commands
                        nonlocal h_minimum, dopt_minimum
                        nonlocal nominal_residual_minimum, safe_raw_residual_minimum
                        nonlocal safe_normalized_residual_minimum, invalid_field_queries
                        nonlocal nonpositive_runtime_field_queries
                        counters.enter_inner(high_index, inner_index)
                        position_now, rotation_now, eef_jacobian = _eef_kinematics(
                            env, site_id, arm_dofs
                        )
                        qvel = np.asarray(env.sim.data.qvel[list(arm_dofs)], dtype=np.float64)
                        nominal_step = adapter.compute_inner_action(
                            position_now,
                            rotation_now,
                            eef_jacobian,
                            measured_joint_velocity=qvel,
                        )
                        nominal = np.asarray(nominal_step.qdot_physical, dtype=np.float64)
                        executed = nominal.copy()
                        qp_record = None
                        provider_h_minimum = None
                        provider_nominal_residual_minimum = None
                        provider_safe_residual_minimum = None
                        points, point_jacobians = evaluate_point_jacobians(
                            env.sim.model,
                            env.sim.data,
                            bundle.protected_samples.samples,
                            arm_dofs,
                        )
                        dopt_now = min(
                            minimum_point_to_oriented_boxes_distance(point, bundle.obstacle_boxes)
                            for point in points
                        )
                        dopt_minimum = min(dopt_minimum, float(dopt_now))
                        if arm == "joint_velocity_psf_link56":
                            queries = [bundle.field.query(point) for point in points]
                            invalid = [query for query in queries if not query.valid]
                            if invalid:
                                invalid_field_queries += len(invalid)
                                fail_closed_attempt_trace.append(
                                    {
                                        "high_level_index": high_index,
                                        "inner_control_index": inner_index,
                                        "completion_class": "field_invalid",
                                        "invalid_query_count": len(invalid),
                                        "first_invalid_reason": str(
                                            invalid[0].reason.value
                                        ),
                                        "D_opt_min_m": float(dopt_now),
                                        "minimum_h_m2": None,
                                        "physics_executed_after_attempt": False,
                                    }
                                )
                                raise ArmTermination(
                                    "field_invalid",
                                    "runtime_poisson_field_query_invalid:%s"
                                    % str(invalid[0].reason.value),
                                )
                            h = np.asarray([float(query.value) for query in queries], dtype=np.float64)
                            provider_h_minimum = float(np.min(h))
                            gradients = np.asarray([query.gradient for query in queries], dtype=np.float64)
                            if np.any(h <= 0.0):
                                h_minimum = min(float(h_minimum), float(np.min(h)))
                                nonpositive_runtime_field_queries += int(
                                    np.count_nonzero(h <= 0.0)
                                )
                                fail_closed_attempt_trace.append(
                                    {
                                        "high_level_index": high_index,
                                        "inner_control_index": inner_index,
                                        "completion_class": "barrier_invariance_lost",
                                        "minimum_h_m2": float(np.min(h)),
                                        "D_opt_min_m": float(dopt_now),
                                        "nonpositive_query_count": int(
                                            np.count_nonzero(h <= 0.0)
                                        ),
                                        "physics_executed_after_attempt": False,
                                    }
                                )
                                raise ArmTermination(
                                    "barrier_invariance_lost",
                                    "runtime_poisson_invariance_lost_nonpositive_h",
                                )
                            h_minimum = min(float(h_minimum), float(np.min(h)))
                            q = np.asarray(env.sim.data.qpos[list(arm_qpos)], dtype=np.float64)
                            lower, upper = joint_velocity_bounds(
                                q,
                                q_min,
                                q_max,
                                qp_cfg["velocity_lower_rad_s"],
                                qp_cfg["velocity_upper_rad_s"],
                                alpha_joint=float(qp_cfg["joint_limit_alpha_per_s"]),
                                control_dt_seconds=0.01,
                                position_margin_rad=float(qp_cfg["joint_position_margin_rad"]),
                            )
                            result = qp.solve_from_field(
                                nominal,
                                h,
                                gradients,
                                point_jacobians,
                                lower,
                                upper,
                                alpha=float(protocol["cbf"]["alpha_gain_per_s"]),
                                weight_diagonal=qp_cfg["weight_diagonal"],
                                require_safe_start=True,
                            )
                            optimizer_counts["attempt_count"] += 1
                            if not result.valid:
                                decision = classify_filter_failure(result.reason, result.diagnostics)
                                key = {
                                    "infeasible": "infeasible_count",
                                    "solver_failure": "solver_failure_count",
                                    "postcheck_failure": "postcheck_failure_count",
                                }.get(decision.optimizer_outcome, "solver_failure_count")
                                optimizer_counts[key] += 1
                                fail_closed_attempt_trace.append(
                                    {
                                        "high_level_index": high_index,
                                        "inner_control_index": inner_index,
                                        "completion_class": decision.completion_class,
                                        "filter_reason": result.reason,
                                        "optimizer_outcome": decision.optimizer_outcome,
                                        "diagnostics": dict(result.diagnostics),
                                        "D_opt_min_m": float(dopt_now),
                                        "minimum_h_m2": provider_h_minimum,
                                        "physics_executed_after_attempt": False,
                                    }
                                )
                                raise ArmTermination(
                                    decision.completion_class,
                                    "hard_qp_fail_closed:%s" % decision.terminal_reason,
                                )
                            optimizer_counts["solved_count"] += 1
                            executed = np.asarray(result.qdot_safe, dtype=np.float64)
                            alpha = float(protocol["cbf"]["alpha_gain_per_s"])
                            rows = np.einsum("ni,nij->nj", gradients, point_jacobians)
                            nominal_raw = rows @ nominal + alpha * h
                            safe_raw = rows @ executed + alpha * h
                            nominal_min = float(np.min(nominal_raw))
                            safe_min = float(np.min(safe_raw))
                            provider_nominal_residual_minimum = nominal_min
                            provider_safe_residual_minimum = safe_min
                            independent = result.diagnostics.get(
                                "minimum_raw_cbf_residual_m2_per_s"
                            )
                            if independent is None or not math.isclose(
                                safe_min, float(independent), rel_tol=0.0, abs_tol=1e-12
                            ):
                                raise ActiveRunnerError("raw CBF residual audit differs from QP")
                            nominal_residual_minimum = nominal_min if nominal_residual_minimum is None else min(nominal_residual_minimum, nominal_min)
                            safe_raw_residual_minimum = safe_min if safe_raw_residual_minimum is None else min(safe_raw_residual_minimum, safe_min)
                            normalized = float(result.diagnostics["minimum_normalized_cbf_residual"])
                            safe_normalized_residual_minimum = normalized if safe_normalized_residual_minimum is None else min(safe_normalized_residual_minimum, normalized)
                            qp_record = dict(result.diagnostics)
                        execution_record = adapter.record_executed_joint_velocity(executed)
                        tracking.register_command(
                            high_level_index=high_index,
                            inner_control_index=inner_index,
                            qdot_command_rad_s=executed,
                        )
                        nominal_norm = float(np.linalg.norm(nominal))
                        safe_norm = float(np.linalg.norm(executed))
                        correction_norm = float(np.linalg.norm(executed - nominal))
                        pending_command["nominal"] = nominal.copy()
                        pending_command["executed"] = executed.copy()
                        pending_command["key"] = (high_index, inner_index)
                        nominal_ledger.append(
                            {"high_level_index": high_index, "inner_control_index": inner_index, "qdot_rad_s": nominal.tolist()}
                        )
                        executed_ledger.append(
                            {"high_level_index": high_index, "inner_control_index": inner_index, "qdot_rad_s": executed.tolist()}
                        )
                        inner_trace.append(
                            {
                                "high_level_index": high_index,
                                "inner_control_index": inner_index,
                                "source_action": source_action.tolist(),
                                "adapter": nominal_step.to_record(),
                                "execution": execution_record,
                                "D_opt_min_m": float(dopt_now),
                                "minimum_h_m2": provider_h_minimum,
                                "minimum_nominal_cbf_residual_m2_per_s": (
                                    provider_nominal_residual_minimum
                                ),
                                "minimum_safe_cbf_residual_m2_per_s": (
                                    provider_safe_residual_minimum
                                ),
                                "qp": qp_record,
                            }
                        )
                        return normalized_joint_velocity_action(
                            executed, float(source_action[6])
                        )

                    def callback(sim: Any, inner_index: int, physics_index: int) -> None:
                        nonlocal measured_motion, nominal_motion, safe_motion
                        nonlocal correction_motion, zero_commands
                        nonlocal eef_path, previous_eef_position, h_minimum
                        nonlocal dopt_minimum, realized_residual_minimum
                        nonlocal first_negative_realized_residual
                        nonlocal negative_realized_residual_count
                        nonlocal realized_residual_evaluation_count
                        nonlocal invalid_field_queries
                        nonlocal nonpositive_runtime_field_queries
                        # The integration already happened. Count it before any
                        # measurement, static-drift, or tracking exception.
                        counters.complete_physics(high_index, inner_index, physics_index)
                        qvel_now = np.asarray(
                            env.sim.data.qvel[list(arm_dofs)], dtype=np.float64
                        )
                        measured_motion += float(np.linalg.norm(qvel_now)) * 0.002
                        eef_position_now = _forwarded_eef_position(env, site_id)
                        eef_path += float(
                            np.linalg.norm(eef_position_now - previous_eef_position)
                        )
                        previous_eef_position = eef_position_now
                        key = (high_index, inner_index)
                        if pending_command["key"] != key:
                            raise ActiveRunnerError(
                                "physics callback has no matching issued command"
                            )
                        pending_nominal = pending_command["nominal"]
                        pending_executed = pending_command["executed"]
                        physics_trace.append(
                            {
                                "high_level_index": high_index,
                                "inner_control_index": inner_index,
                                "physics_substep_index": physics_index,
                                "nominal_joint_velocity_command_rad_s": (
                                    pending_nominal.tolist()
                                ),
                                "issued_joint_velocity_command_rad_s": (
                                    pending_executed.tolist()
                                ),
                                "measured_arm_joint_velocity_rad_s": (
                                    qvel_now.tolist()
                                ),
                                "forwarded_eef_position_m": (
                                    eef_position_now.tolist()
                                ),
                            }
                        )
                        nominal_motion += float(np.linalg.norm(pending_nominal)) * 0.002
                        safe_motion += float(np.linalg.norm(pending_executed)) * 0.002
                        correction_motion += float(
                            np.linalg.norm(pending_executed - pending_nominal)
                        ) * 0.002
                        if key not in exposed_command_keys:
                            exposed_command_keys.add(key)
                            zero_commands += int(
                                float(np.linalg.norm(pending_executed)) <= 1e-12
                            )
                        drift_error = None
                        tracking_error = None
                        realized_error = None
                        try:
                            monitor.observe_post_integration(
                                sim,
                                high_level_index=high_index,
                                inner_control_index=inner_index,
                                physics_substep_index=physics_index,
                            )
                        except StaticObstacleDriftInadmissible as error:
                            drift_error = error
                        try:
                            tracking.observe_post_integration(
                                high_level_index=high_index,
                                inner_control_index=inner_index,
                                physics_substep_index=physics_index,
                                measured_qvel_rad_s=qvel_now,
                            )
                        except ControllerTrackingInvalid as error:
                            tracking_error = error
                        if arm == "joint_velocity_psf_link56" and drift_error is not None:
                            realized_cbf_trace.append(
                                {
                                    "high_level_index": high_index,
                                    "inner_control_index": inner_index,
                                    "physics_substep_index": physics_index,
                                    "valid": False,
                                    "failure": "static_selected_obstacle_drift_invalidated_field",
                                    "query_count": 0,
                                }
                            )
                        if arm == "joint_velocity_psf_link56" and drift_error is None:
                            raw_model, raw_data = _raw_model_data(env.sim)
                            forwarded_data = clone_forwarded_state(raw_model, raw_data)
                            realized_points, realized_jacobians = evaluate_point_jacobians(
                                raw_model,
                                forwarded_data,
                                bundle.protected_samples.samples,
                                arm_dofs,
                            )
                            dopt_realized = min(
                                minimum_point_to_oriented_boxes_distance(
                                    point, bundle.obstacle_boxes
                                )
                                for point in realized_points
                            )
                            dopt_minimum = min(dopt_minimum, float(dopt_realized))
                            realized_queries = [
                                bundle.field.query(point) for point in realized_points
                            ]
                            invalid_realized = [
                                query for query in realized_queries if not query.valid
                            ]
                            realized_row = {
                                "high_level_index": high_index,
                                "inner_control_index": inner_index,
                                "physics_substep_index": physics_index,
                                "D_opt_min_m": float(dopt_realized),
                                "query_count": len(realized_queries),
                            }
                            if invalid_realized:
                                invalid_field_queries += len(invalid_realized)
                                realized_row.update(
                                    {
                                        "valid": False,
                                        "failure": "post_state_field_invalid",
                                        "invalid_query_count": len(invalid_realized),
                                        "first_invalid_reason": str(
                                            invalid_realized[0].reason.value
                                        ),
                                    }
                                )
                                realized_error = ArmTermination(
                                    "field_invalid",
                                    "post_state_poisson_field_query_invalid:%s"
                                    % str(invalid_realized[0].reason.value),
                                )
                            else:
                                realized_h = np.asarray(
                                    [float(query.value) for query in realized_queries],
                                    dtype=np.float64,
                                )
                                h_minimum = min(
                                    float(h_minimum), float(np.min(realized_h))
                                )
                                if np.any(realized_h <= 0.0):
                                    count = int(np.count_nonzero(realized_h <= 0.0))
                                    nonpositive_runtime_field_queries += count
                                    realized_row.update(
                                        {
                                            "valid": False,
                                            "failure": "post_state_nonpositive_h",
                                            "minimum_h_m2": float(np.min(realized_h)),
                                            "nonpositive_query_count": count,
                                        }
                                    )
                                    realized_error = ArmTermination(
                                        "barrier_invariance_lost",
                                        "post_state_poisson_invariance_lost_nonpositive_h",
                                    )
                                else:
                                    realized_gradients = np.asarray(
                                        [query.gradient for query in realized_queries],
                                        dtype=np.float64,
                                    )
                                    realized_rows = np.einsum(
                                        "ni,nij->nj",
                                        realized_gradients,
                                        realized_jacobians,
                                    )
                                    realized_residuals = (
                                        realized_rows @ qvel_now
                                        + float(protocol["cbf"]["alpha_gain_per_s"])
                                        * realized_h
                                    )
                                    realized_minimum = float(
                                        np.min(realized_residuals)
                                    )
                                    realized_residual_evaluation_count += len(
                                        realized_residuals
                                    )
                                    realized_residual_minimum = (
                                        realized_minimum
                                        if realized_residual_minimum is None
                                        else min(
                                            realized_residual_minimum,
                                            realized_minimum,
                                        )
                                    )
                                    realized_row.update(
                                        {
                                            "valid": True,
                                            "minimum_h_m2": float(np.min(realized_h)),
                                            "minimum_realized_cbf_residual_m2_per_s": realized_minimum,
                                        }
                                    )
                                    if realized_minimum < 0.0:
                                        negative_realized_residual_count += 1
                                        crossing = {
                                            "high_level_index": high_index,
                                            "inner_control_index": inner_index,
                                            "physics_substep_index": physics_index,
                                            "minimum_realized_cbf_residual_m2_per_s": realized_minimum,
                                            "minimum_sample_index": int(
                                                np.argmin(realized_residuals)
                                            ),
                                            "minimum_h_m2": float(
                                                realized_h[
                                                    int(np.argmin(realized_residuals))
                                                ]
                                            ),
                                        }
                                        if first_negative_realized_residual is None:
                                            first_negative_realized_residual = crossing
                                        realized_row["first_negative_crossing"] = crossing
                                        realized_error = ArmTermination(
                                            "barrier_invariance_lost",
                                            "post_state_realized_cbf_residual_negative",
                                        )
                            realized_cbf_trace.append(realized_row)
                        if drift_error is not None:
                            raise ArmTermination(
                                "static_obstacle_inadmissible",
                                "runtime_selected_obstacle_static_drift",
                            )
                        if realized_error is not None:
                            fail_closed_attempt_trace.append(
                                {
                                    "high_level_index": high_index,
                                    "inner_control_index": inner_index,
                                    "physics_substep_index": physics_index,
                                    "completion_class": realized_error.completion_class,
                                    "terminal_reason": realized_error.terminal_reason,
                                    "physics_executed_before_attempt": True,
                                    "further_physics_executed_after_attempt": False,
                                }
                            )
                            raise realized_error
                        if tracking_error is not None:
                            raise ArmTermination(
                                "controller_tracking_invalid",
                                "runtime_joint_velocity_tracking_threshold_crossed",
                            )

                    observation, _, done, _ = env.step_grouped_actions_with_substep_callback(
                        provider,
                        callback,
                        expected_inner_updates=5,
                        expected_substeps_per_inner=5,
                        expected_high_level_dt=0.05,
                    )
                    if bool(env.env.done):
                        raise ActiveRunnerError(
                            "internal robosuite horizon terminated the fixed replay"
                        )
                    completed_source_actions.append(source_action.tolist())
                    post_step_obstacle_positions.append(
                        np.asarray(observation[active_name + "_pos"], dtype=np.float64).tolist()
                    )
                    goal = evaluator._goal_progress_snapshot(
                        env,
                        goal_atoms,
                        step=high_index,
                        previous_values=previous_goal_values,
                    )
                    if bool(done) != bool(goal["all_satisfied"]):
                        raise ActiveRunnerError(
                            "returned task-success done differs from native goal conjunction"
                        )
                    goal_ledger.append(
                        dict(goal, snapshot_kind="completed_high_level_post_step")
                    )
                    previous_goal_values = goal["values"]
                    task_latched = bool(task_latched or goal["all_satisfied"] or done)
                    terminal_task_success = bool(goal["all_satisfied"])
                    terminal_goal_fraction = float(goal["fraction"])
                    if goal["all_satisfied"] and first_task_success_index is None:
                        first_task_success_index = high_index
                    maximum_goal_fraction = max(maximum_goal_fraction, float(goal["fraction"]))
                    goal_regressions += len(goal["regressed_indices"])
                tracking.require_complete(len(replay.actions) * 5)
            except ArmTermination as stop:
                completion = stop.completion_class
                terminal_reason = stop.terminal_reason

        exposure_complete = counters.exposure_complete(len(replay.actions)) and completion == "executed"
        terminal_simulator_time_s = float(target_raw_data.time)
        expected_physics_exposure_s = float(counters.physics_substeps) * 0.002
        observed_physics_exposure_s = (
            terminal_simulator_time_s - settled_simulator_time_s
        )
        if not math.isclose(
            observed_physics_exposure_s,
            expected_physics_exposure_s,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ActiveRunnerError(
                "MuJoCo clock exposure differs from completed 2 ms physics count"
            )
        if exposure_complete and len(post_step_obstacle_positions) != len(replay.actions):
            raise ActiveRunnerError(
                "complete CAR endpoint lacks one post-step position per source action"
            )
        monitor_record = _monitor_summary(monitor, counters.physics_substeps)
        static_admissible = bool(
            settled_motion_admissible
            and monitor_record["translation_drift_m"]
            <= float(admissibility["max_selected_geom_translation_drift_m"])
            and monitor_record["rotation_drift_rad"]
            <= float(admissibility["max_selected_geom_rotation_drift_rad"])
            and monitor_record["surface_drift_m"]
            <= float(admissibility["max_selected_geom_surface_drift_m"])
        )
        car = paper_car_endpoint(
            settled_obstacle_position,
            post_step_obstacle_positions,
            fixed_exposure_complete=exposure_complete,
        )
        car_position_ledger = {
            "settled_active_obstacle_root_position_m": settled_obstacle_position.tolist(),
            "settled_active_obstacle_position_sha256": target_position_sha256,
            "source_settled_active_obstacle_position_sha256": source_position_sha256,
            "historical_settled_active_obstacle_position_sha256": (
                replay.settled_active_obstacle_position_sha256
            ),
            "completed_post_step_positions": [
                {
                    "high_level_index": index,
                    "position_m": list(position),
                    "l1_displacement_from_settled_m": float(
                        np.sum(
                            np.abs(
                                np.asarray(position, dtype=np.float64)
                                - settled_obstacle_position
                            )
                        )
                    ),
                }
                for index, position in enumerate(post_step_obstacle_positions)
            ],
        }
        car_position_ledger_sha = _sha256(_canonical(car_position_ledger))
        car.update(
            {
                "exposure_scope": "registered_historical_source_exposure_not_full_table1_suite_horizon",
                "position_ledger_sha256": car_position_ledger_sha,
            }
        )
        state_before_observation_sync = _official_integration_vector(env.sim)
        env.env._update_observables(force=True)
        observation = env.env._get_observations()
        state_after_observation_sync = _official_integration_vector(env.sim)
        if not np.array_equal(
            state_before_observation_sync, state_after_observation_sync
        ):
            raise ActiveRunnerError(
                "terminal observation synchronization changed integration state"
            )
        if not exposure_complete:
            diagnostic_step = counters.high_level_steps - 1
            terminal_inner_global = counters.inner_control_steps - 1
            terminal_inner_index = (
                terminal_inner_global % 5 if terminal_inner_global >= 0 else None
            )
            completed_substeps_in_terminal_inner = (
                counters.physics_substeps - 5 * terminal_inner_global
                if terminal_inner_global >= 0
                else 0
            )
            terminal_goal = evaluator._goal_progress_snapshot(
                env,
                goal_atoms,
                step=diagnostic_step,
                previous_values=previous_goal_values,
            )
            goal_ledger.append(
                dict(
                    terminal_goal,
                    snapshot_kind="partial_prefix_terminal_state_diagnostic",
                    terminal_prefix_coordinate={
                        "entered_high_level_index": (
                            counters.high_level_steps - 1
                            if counters.high_level_steps > 0
                            else None
                        ),
                        "entered_inner_control_index": terminal_inner_index,
                        "completed_physics_substeps_in_entered_inner": (
                            completed_substeps_in_terminal_inner
                        ),
                        "completed_high_level_action_count": len(
                            completed_source_actions
                        ),
                    },
                )
            )
            terminal_task_success = bool(terminal_goal["all_satisfied"])
            terminal_goal_fraction = float(terminal_goal["fraction"])
            task_latched = bool(task_latched or terminal_task_success)
            maximum_goal_fraction = max(
                maximum_goal_fraction, terminal_goal_fraction
            )
            goal_regressions += len(terminal_goal["regressed_indices"])
            if terminal_task_success and first_task_success_index is None:
                first_task_success_index = diagnostic_step
        terminal_observation_hash = _observation_sha256(observation, evaluator, np)
        terminal_state_hash = evaluator.array_sha256(state_after_observation_sync)
        terminal_legacy_state_hash = evaluator.array_sha256(
            env.sim.get_state().flatten()
        )
        command_count = len(exposed_command_keys)
        all_issued_arm_joint_commands_zero = bool(
            command_count > 0 and zero_commands == command_count
        )
        optimizer_terminal = "not_applicable"
        if arm != "joint_velocity_adapter_only":
            if optimizer_counts["postcheck_failure_count"]:
                optimizer_terminal = "postcheck_failure"
            elif optimizer_counts["solver_failure_count"]:
                optimizer_terminal = "solver_failure"
            elif optimizer_counts["infeasible_count"]:
                optimizer_terminal = "infeasible"
            elif optimizer_counts["solved_count"]:
                optimizer_terminal = "solved"
            else:
                optimizer_terminal = "not_reached"
        return {
            "arm": arm,
            "completion_class": completion,
            "terminal_reason": terminal_reason,
            "exposure_complete": exposure_complete,
            "restore": restore,
            "selected_initial_state_sha256": evaluator.array_sha256(selected_initial_state),
            "active_initial_observation_sha256": active_initial_observation_sha256,
            "goal_definition": goal_definition,
            "goal_progress_ledger": goal_ledger,
            "initial_protected_sample_audit": {
                "protected_sample_count": len(initial_points),
                "field_query_count": len(initial_queries),
                "valid_field_query_count": len(initial_valid_h),
                "minimum_h_m2": initial_h_minimum,
                "D_opt_min_m": float(initial_dopt_minimum),
                "strict_safe_start": safe_start,
            },
            "prefix": counters.record(),
            "physics_clock": {
                "settled_time_s": settled_simulator_time_s,
                "terminal_time_s": terminal_simulator_time_s,
                "observed_exposure_s": observed_physics_exposure_s,
                "expected_from_completed_substeps_s": expected_physics_exposure_s,
                "exact_count_consistent_within_abs_1e_10_s": True,
            },
            "entered_source_actions": entered_source_actions,
            "completed_source_actions": completed_source_actions,
            "nominal_ledger": nominal_ledger,
            "executed_ledger": executed_ledger,
            "inner_trace": inner_trace,
            "physics_trace": physics_trace,
            "fail_closed_attempt_trace": fail_closed_attempt_trace,
            "realized_cbf_trace": realized_cbf_trace,
            "monitor": monitor_record,
            "paper_car": car,
            "paper_car_position_ledger": car_position_ledger,
            "task": {
                "fixed_exposure_available": exposure_complete,
                "outcome_scope": "registered_source_outcome_dependent_exposure_not_full_table1_suite_horizon",
                "ever_task_success_within_registered_source_exposure": bool(task_latched) if exposure_complete else None,
                "terminal_task_success_within_registered_source_exposure": bool(terminal_task_success) if exposure_complete else None,
                "first_task_success_high_level_index": first_task_success_index if exposure_complete else None,
                "initial_task_success": initial_task_success,
                "prefix_task_success_latched": bool(task_latched),
                "prefix_first_task_success_high_level_index": first_task_success_index,
                "prefix_terminal_task_success": bool(terminal_task_success),
                "terminal_goal_fraction": terminal_goal_fraction,
                "maximum_goal_fraction": maximum_goal_fraction,
                "goal_regression_count": goal_regressions,
                "unavailable_reason": None if exposure_complete else "fixed_exposure_incomplete",
            },
            "usefulness": {
                "nominal_joint_motion_integral_rad": nominal_motion,
                "safe_joint_motion_integral_rad": safe_motion,
                "measured_joint_motion_integral_rad": measured_motion,
                "eef_path_length_m": eef_path,
                "correction_integral_rad": correction_motion,
                "motion_retention_ratio": safe_motion / nominal_motion if nominal_motion > 0.0 else 0.0,
                "zero_motion_fraction": zero_commands / command_count if command_count else 1.0,
                "all_issued_arm_joint_commands_zero": (
                    all_issued_arm_joint_commands_zero
                ),
                "safety_by_no_execution": counters.physics_substeps == 0,
            },
            "validity": {
                "poisson_safe_start": safe_start,
                "static_admissible": static_admissible,
                "coverage_audit_passed": True,
                "obstacle_translation_drift_m": monitor_record["translation_drift_m"],
                "obstacle_rotation_drift_rad": monitor_record["rotation_drift_rad"],
                "invalid_field_query_count": invalid_field_queries,
                "nonpositive_runtime_field_query_count": nonpositive_runtime_field_queries,
                "negative_realized_cbf_residual_count": negative_realized_residual_count,
                "realized_cbf_observed_physics_substep_count": len(
                    realized_cbf_trace
                ),
                "realized_cbf_residual_evaluation_count": (
                    realized_residual_evaluation_count
                ),
                "first_negative_realized_cbf_residual": (
                    first_negative_realized_residual
                ),
                "qp_infeasible_count": optimizer_counts["infeasible_count"],
                "qp_solver_failure_count": optimizer_counts["solver_failure_count"],
                "qp_postcheck_failure_count": optimizer_counts["postcheck_failure_count"],
                "maximum_velocity_tracking_error_rad_s": tracking.maximum_observed_linf_rad_s,
                "velocity_tracking_rmse_rad_s": tracking.cumulative_rmse_rad_s,
                "velocity_tracking_linf_threshold_rad_s": tracking.maximum_linf_rad_s,
                "velocity_tracking_rmse_threshold_rad_s": tracking.maximum_rmse_rad_s,
                "velocity_tracking_gate_semantics": (
                    "empirical_controller_fidelity_gate_not_cbf_safety_certificate"
                ),
                "velocity_tracking_observed_physics_substep_count": tracking.summary()[
                    "observed_physics_substep_count"
                ],
                "first_velocity_tracking_threshold_crossing": tracking.summary()[
                    "first_threshold_crossing"
                ],
            },
            "optimizer_counts": optimizer_counts,
            "optimizer_terminal_status": optimizer_terminal,
            "minimums": {
                "h_m2": h_minimum if arm == "joint_velocity_psf_link56" else None,
                "D_opt_m": dopt_minimum,
                "nominal_raw_cbf_residual_m2_per_s": nominal_residual_minimum,
                "safe_raw_cbf_residual_m2_per_s": safe_raw_residual_minimum,
                "safe_normalized_cbf_residual": safe_normalized_residual_minimum,
                "realized_raw_cbf_residual_m2_per_s": realized_residual_minimum,
            },
            "realized_cbf_audit": {
                "semantics": "every_completed_2ms_post_state_grad_h_T_J_qvel_actual_plus_alpha_h",
                "observed_physics_substep_count": len(realized_cbf_trace),
                "residual_evaluation_count": realized_residual_evaluation_count,
                "negative_residual_callback_count": negative_realized_residual_count,
                "first_negative_residual": first_negative_realized_residual,
            },
            "eef_path_audit": {
                "semantics": "settled_forwarded_grip_site_plus_every_completed_2ms_post_state",
                "position_sample_count": 1 + counters.physics_substeps,
                "settled_forwarded_eef_position_m": (
                    settled_eef_position.tolist()
                ),
            },
            "tracking": tracking.summary(),
            "eef_reference_site": eef_jacobian_audit,
            "terminal_simulator_state_sha256": terminal_state_hash,
            "terminal_legacy_flattened_state_sha256": terminal_legacy_state_hash,
            "terminal_observation_sha256": terminal_observation_hash,
            "task_language": str(task.language),
        }
    finally:
        if env is not None:
            env.close()


def _scientific_payload(
    *,
    root: Path,
    run_id: str,
    case: Mapping[str, Any],
    case_row_hash: str,
    manifest_path: Path,
    selection_path: Path,
    runtime_protocol_path: Path,
    protocol_hashes: Any,
    checkpoint: Mapping[str, str],
    replay: Any,
    allocation: Mapping[str, Any],
    source_git: Mapping[str, Any],
    source_pairing: Mapping[str, Any],
    bundle: Any,
    resolved: Any,
    arm_outcome: Mapping[str, Any],
    artifact_reference: Mapping[str, Any],
    run_contract_sha256: str,
) -> Dict[str, Any]:
    from main.poisson_fullbody.contracts import canonical_json_bytes, sha256_bytes
    from main.poisson_fullbody.result_schema import (
        BASELINE_COMMIT,
        CODE_REPOSITORY,
        BASELINE_REPOSITORY,
        MEASUREMENT_SCHEMA_VERSION,
        PROTOCOL_ID,
        RESULT_SCHEMA_VERSION,
    )

    arm = arm_outcome["arm"]
    optimizer = arm_outcome["optimizer_counts"]
    prefix = arm_outcome["prefix"]
    full_source_hash = replay.executed_sequence_sha256
    entered_hash = _sha256(_canonical(arm_outcome["entered_source_actions"]))
    completed_hash = _sha256(_canonical(arm_outcome["completed_source_actions"]))
    # The historical full ledger uses the same canonical list-of-lists hash.
    if arm_outcome["exposure_complete"] and (
        entered_hash != full_source_hash or completed_hash != full_source_hash
    ):
        raise ActiveRunnerError("complete action prefix hashes differ from historical source")
    seeds = [
        {"scope": "simulator", "name": "environment", "value": int(case["environment_seed"])},
        {"scope": "numpy_rng", "name": "environment_construction_np_random_seed", "value": int(case["environment_seed"])},
        {"scope": "historical_policy_noise", "name": "historical_pi05_query_noise_not_reexecuted", "value": int(case["policy_noise_seed"])},
    ]
    controller_contract = arm_outcome["restore"]["controller"]
    controller_initial_sha = arm_outcome["restore"]["controller_software_state"]["sha256"]
    pairing_key = {
        "pair_group_id": case["pair_group_id"],
        "official_settled_state": arm_outcome["restore"]["official_integration_state_sha256"],
        "historical_action_ledger": full_source_hash,
        "field_sha256": bundle.hashes.field_sha256,
        "source_observation_sha256": replay.initial_observation_sha256,
    }
    intervention_enabled = arm != "joint_velocity_adapter_only"
    minimums = arm_outcome["minimums"]
    monitor = arm_outcome["monitor"]
    exposure_complete = arm_outcome["exposure_complete"]
    measurement_config = {
        "resolved_geometry": resolved.to_dict(),
        "D_sim_semantics": D_SIM_SEMANTICS,
    }
    optimizer_config = {
        "runtime_protocol_parameter_block_sha256": protocol_hashes.parameter_block_sha256,
        "arm": arm,
        "solver": "osqp",
    }
    sampler_config = {
        "source": replay.provenance(),
        "active_policy_queries": 0,
        "active_policy_rng_exercised": False,
    }
    model_config = {
        "checkpoint_receipt_payload_sha256": checkpoint["receipt_payload_sha256"],
        "checkpoint_receipt_file_sha256": checkpoint["receipt_file_sha256"],
        "source_historical_arm": replay.arm,
        "execution_semantics": checkpoint["execution_semantics"],
    }
    controller_config = controller_contract
    intervention_config = {
        "arm": arm,
        "field_sha256": bundle.hashes.field_sha256,
        "protected": [] if not intervention_enabled else ["robot0_link5", "robot0_link6"],
    }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "scientific_result": True,
        "completion_class": arm_outcome["completion_class"],
        "run_id": run_id,
        "protocol_id": PROTOCOL_ID,
        "case_id": case["case_id"],
        "arm": arm,
        "provenance": {
            "code_repository": CODE_REPOSITORY,
            "code_commit": source_git["commit"],
            "code_dirty": False,
            "code_dirty_state_sha256": sha256_bytes(b""),
            "code_dirty_state_semantics": "sha256_empty_bytes_for_required_clean_git_worktree_v1",
            "baseline_repository": BASELINE_REPOSITORY,
            "baseline_commit": BASELINE_COMMIT,
            "run_contract_sha256": run_contract_sha256,
            "manifest_sha256": _file_sha256(manifest_path),
            "manifest_record_sha256": case_row_hash,
            "protocol_config_sha256": _file_sha256(selection_path),
            "runtime_protocol_raw_sha256": _file_sha256(runtime_protocol_path),
            "runtime_protocol_semantic_sha256": protocol_hashes.protocol_sha256,
        },
        "case_identity": {
            "dataset": "SafeLIBERO",
            "manifest_case_id": case["case_id"],
            "manifest_case_ordinal": int(case["case_ordinal"]),
            "pair_group_id": case["pair_group_id"],
            "suite": case["suite"],
            "safety_level": case["safety_level"],
            "logical_task_index": int(case["logical_task_index"]),
            "resolved_task_index": int(case["resolved_task_index"]),
            "episode_index": int(case["episode_index"]),
            "task_name": case["task_name"],
            "task_family_id": case["task_family_id"],
            "task_level_group_id": case["task_level_group_id"],
            "bddl_relative_path": case["bddl_path"],
            "bddl_sha256": case["bddl_sha256"],
            "initial_states_relative_path": case["initial_states_path"],
            "initial_states_sha256": case["initial_states_sha256"],
            "initial_state_record_sha256": arm_outcome["selected_initial_state_sha256"],
            "active_obstacle_name": case["active_obstacle_name"],
            "split": case["split"],
            "stratum": case["stratum"],
            "suite_horizon_high_level_steps": int(case["max_steps"]),
            "registered_source_exposure_high_level_steps": len(replay.actions),
        },
        "seeds": {
            "inventory_complete": True,
            "entries": seeds,
            "ledger_sha256": sha256_bytes(canonical_json_bytes(seeds)),
        },
        "pairing": {
            "pair_group_id": case["pair_group_id"],
            "pairing_key_sha256": _sha256(_canonical(pairing_key)),
            "restored_settled_state_sha256": arm_outcome["restore"]["official_integration_state_sha256"],
            "source_settled_observation_sha256": replay.initial_observation_sha256,
            "active_joint_velocity_initial_observation_sha256": arm_outcome["active_initial_observation_sha256"],
            "policy_noise_schedule_sha256": replay.policy_noise_schedule_sha256,
            "policy_query_schedule_sha256": replay.source_policy_query_schedule_sha256,
            "nominal_high_level_action_ledger_sha256": full_source_hash,
            "settling_action_ledger_sha256": _sha256(_canonical([[0.0] * 7] * 20)),
            "controller_initial_state_sha256": controller_initial_sha,
            "compiled_physical_model_sha256": arm_outcome["restore"]["physical_model_sha256"],
            "compiled_mjb_sha256": arm_outcome["restore"]["compiled_mjb_sha256"],
            "field_sha256": bundle.hashes.field_sha256,
            "source_exposure_high_level_steps": len(replay.actions),
            "inner_updates_per_high_level_action": 5,
            "high_level_dt_seconds": 0.05,
            "inner_dt_seconds": 0.01,
            "physics_dt_seconds": 0.002,
        },
        "runtime": {
            "protocol_parameter_block_sha256": protocol_hashes.parameter_block_sha256,
            "model": {
                "policy_name": "pi0.5",
                "model_configuration_sha256": _sha256(_canonical(model_config)),
                "checkpoint_identifier": "openpi-pi05-libero",
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "checkpoint_hash_semantics": "sha256_content_tree_v1",
                "normalization_statistics_sha256": checkpoint["normalization_statistics_sha256"],
            },
            "sampler": {
                "implementation": "frozen_historical_pi05_plus_aegis_executed_action_replay",
                "configuration_sha256": _sha256(_canonical(sampler_config)),
                "source_action_horizon": 10,
                "source_replan_steps": 5,
                "source_policy_query_count": replay.source_policy_query_count,
                "active_policy_query_count": 0,
                "active_policy_rng_exercised": False,
                "source_action_semantics": "historical_pi05_plus_aegis_translational_exact_actions_executed_not_nominal_raw",
                "deterministic_replay": True,
            },
            "intervention": {
                "enabled": intervention_enabled,
                "mode": arm,
                "configuration_sha256": _sha256(_canonical(intervention_config)),
                "protected_robot_bodies": [] if not intervention_enabled else ["robot0_link5", "robot0_link6"],
                "static_selected_obstacle_only": True,
                "fail_closed": True,
            },
            "controller": {
                "name": "JOINT_VELOCITY",
                "implementation_version": "robosuite-" + _package_version("robosuite"),
                "configuration_sha256": _sha256(_canonical(controller_config)),
                "controlled_joint_count": 7,
                "command_dimension": 8,
                "normalized_limit_abs": 1.0,
                "physical_velocity_limit_rad_s": 0.5,
            },
            "action_space": {
                "source_policy_frame": "mujoco_world_frame_cartesian_delta",
                "controller_command_frame": "panda_joint_velocity_coordinates",
                "source_policy_normalization_space": "openpi_output_after_libero_action_unnormalization",
                "controller_normalization_space": "robosuite_joint_velocity_normalized_minus1_plus1",
                "physical_displacement_conversion": "scale_only_never_subtract_normalization_mean",
                "gripper_semantics": "unchanged_vla_command",
            },
            "measurement": {
                "schema_version": MEASUREMENT_SCHEMA_VERSION,
                "implementation_sha256": _file_sha256(root / "main/poisson_fullbody/measurement.py"),
                "configuration_sha256": _sha256(_canonical(measurement_config)),
                "simulator": "MuJoCo",
                "simulator_version": _package_version("mujoco"),
                "robosuite_version": _package_version("robosuite"),
                "physics_timestep_seconds": 0.002,
                "contact_definition": "mujoco_contact_dist_le_zero",
                "contact_sources": ["settled_state", "live_solver_state", "forwarded_post_integration_state"],
                "D_opt_semantics": "minimum_registered_surface_sample_signed_distance_to_selected_obstacle_obb_union",
                "D_sim_semantics": D_SIM_SEMANTICS,
                "optimizer_and_simulator_clearance_distinct": True,
            },
        },
        "optimizer": {
            "enabled": intervention_enabled,
            "solver": "osqp",
            "solver_version": _package_version("osqp"),
            "configuration_sha256": _sha256(_canonical(optimizer_config)),
            **optimizer,
            "terminal_status": arm_outcome["optimizer_terminal_status"],
            "minimum_normalized_cbf_residual": minimums["safe_normalized_cbf_residual"],
            "minimum_raw_cbf_residual_m2_per_s": minimums["safe_raw_cbf_residual_m2_per_s"],
            "normalized_cbf_postcheck_tolerance": 5e-7,
        },
        "allocation": dict(allocation),
        "execution": {
            "high_level_steps": int(prefix["high_level_steps"]),
            "inner_control_steps": int(prefix["inner_control_steps"]),
            "physics_substeps": int(prefix["physics_substeps"]),
            "physics_exposure_seconds": float(prefix["physics_exposure_seconds"]),
            "exposure_complete": exposure_complete,
            "fail_closed_no_further_physics": not exposure_complete,
            "terminal_reason": arm_outcome["terminal_reason"],
            "completed_high_level_steps": len(
                arm_outcome["completed_source_actions"]
            ),
            "terminal_entered_high_level_index": (
                int(prefix["high_level_steps"]) - 1
                if int(prefix["high_level_steps"]) > 0
                else None
            ),
            "terminal_entered_inner_control_index": (
                (int(prefix["inner_control_steps"]) - 1) % 5
                if int(prefix["inner_control_steps"]) > 0
                else None
            ),
            "completed_physics_substeps_in_terminal_inner_control": (
                int(prefix["physics_substeps"])
                - 5 * (int(prefix["inner_control_steps"]) - 1)
                if int(prefix["inner_control_steps"]) > 0
                else 0
            ),
            "source_action_prefix_semantics": (
                "entered_hash_counts_provider_entered_actions_completed_hash_counts_only_"
                "fully_returned_high_level_post_steps"
            ),
            "planned_source_action_ledger_sha256": full_source_hash,
            "entered_source_action_prefix_sha256": entered_hash,
            "completed_high_level_source_action_prefix_sha256": completed_hash,
            "nominal_joint_velocity_ledger_sha256": _sha256(_canonical(arm_outcome["nominal_ledger"])),
            "executed_controller_action_ledger_sha256": _sha256(_canonical(arm_outcome["executed_ledger"])),
            "terminal_simulator_state_sha256": arm_outcome["terminal_simulator_state_sha256"],
            "terminal_simulator_state_semantics": "complete_official_mujoco_mjSTATE_INTEGRATION",
            "terminal_observation_sha256": arm_outcome["terminal_observation_sha256"],
            "terminal_observation_semantics": "current_state_synchronized_without_additional_physics",
        },
        "endpoints": {
            "safety": {
                "h_min": _measurement(minimums["h_m2"], "poisson_field_m2", "adapter_intervention_disabled"),
                "minimum_nominal_cbf_residual": _measurement(minimums["nominal_raw_cbf_residual_m2_per_s"], "poisson_cbf_m2_per_s", "no_solved_poisson_qp"),
                "minimum_safe_cbf_residual": _measurement(minimums["safe_raw_cbf_residual_m2_per_s"], "poisson_cbf_m2_per_s", "no_solved_poisson_qp"),
                "minimum_realized_cbf_residual": _measurement(
                    minimums["realized_raw_cbf_residual_m2_per_s"],
                    "poisson_cbf_m2_per_s",
                    "adapter_intervention_disabled_or_no_valid_completed_post_state_query",
                ),
                "D_opt_min_m": _measurement(minimums["D_opt_m"], "m", "no_protected_surface_query"),
                "D_sim_min_m": _measurement(monitor["D_sim_min_m"], "m", "simulator_measurement_unavailable"),
                "D_sim_semantics": D_SIM_SEMANTICS,
                "any_robot_obstacle_contact": monitor["any_contact"],
                "link56_obstacle_contact": monitor["link56_contact"],
                "contact_observation_complete": exposure_complete,
                "contact_absence_right_censored": bool(not exposure_complete and not monitor["any_contact"]),
                "total_physical_contact_point_record_count": monitor["total"],
                "settled_physical_contact_point_record_count": monitor["settled"],
                "rollout_phase_physical_contact_point_record_count": monitor["rollout"],
                "live_solver_nonpositive_contact_point_record_count": monitor["live"],
                "post_state_physical_contact_point_record_count": monitor["post"],
                "first_physical_contact_point_record": monitor["first"],
                "first_settled_physical_contact_point_record": monitor["first_settled"],
                "first_live_solver_physical_contact_point_record": monitor["first_live"],
                "first_post_state_physical_contact_point_record": monitor["first_post"],
                "paper_car": arm_outcome["paper_car"],
            },
            "task": arm_outcome["task"],
            "usefulness": arm_outcome["usefulness"],
            "validity": arm_outcome["validity"],
        },
        "artifact_references": [dict(artifact_reference)],
    }


def _validate_trace_payload_against_scientific(
    trace_payload: Mapping[str, Any],
    scientific_payload: Mapping[str, Any],
) -> None:
    """Recompute the scientific projection from the serialized arm trace.

    The episode schema validates the compact result.  This second, independent
    gate prevents a valid-looking result from disagreeing with the immutable
    2 ms trace that it cites.
    """

    def require(condition: bool, label: str) -> None:
        if not condition:
            raise ActiveRunnerError("arm trace/scientific mismatch: " + label)

    def json_equal(left: Any, right: Any) -> bool:
        return _canonical(left) == _canonical(right)

    outcome = trace_payload.get("outcome")
    require(isinstance(outcome, Mapping), "trace.outcome missing")
    require(trace_payload.get("run_id") == scientific_payload.get("run_id"), "run_id")
    require(trace_payload.get("case_id") == scientific_payload.get("case_id"), "case_id")
    require(trace_payload.get("arm") == scientific_payload.get("arm"), "arm")
    require(outcome.get("arm") == scientific_payload.get("arm"), "outcome arm")
    require(
        outcome.get("completion_class") == scientific_payload.get("completion_class"),
        "completion_class",
    )

    execution = scientific_payload["execution"]
    prefix = outcome["prefix"]
    for field in (
        "high_level_steps",
        "inner_control_steps",
        "physics_substeps",
        "physics_exposure_seconds",
    ):
        require(execution[field] == prefix[field], "execution." + field)
    high_steps = int(prefix["high_level_steps"])
    inner_steps = int(prefix["inner_control_steps"])
    physics_steps = int(prefix["physics_substeps"])
    entered_actions = outcome["entered_source_actions"]
    completed_actions = outcome["completed_source_actions"]
    require(len(entered_actions) == high_steps, "entered source-action count")
    require(
        len(completed_actions) == int(execution["completed_high_level_steps"]),
        "completed source-action count",
    )
    require(
        json_equal(completed_actions, entered_actions[: len(completed_actions)]),
        "completed actions are not an entered-action prefix",
    )
    require(
        _sha256(_canonical(entered_actions))
        == execution["entered_source_action_prefix_sha256"],
        "entered source-action hash",
    )
    require(
        _sha256(_canonical(completed_actions))
        == execution["completed_high_level_source_action_prefix_sha256"],
        "completed source-action hash",
    )
    require(
        _sha256(_canonical(outcome["nominal_ledger"]))
        == execution["nominal_joint_velocity_ledger_sha256"],
        "nominal command-ledger hash",
    )
    require(
        _sha256(_canonical(outcome["executed_ledger"]))
        == execution["executed_controller_action_ledger_sha256"],
        "executed command-ledger hash",
    )
    require(
        len(outcome["nominal_ledger"])
        == len(outcome["executed_ledger"])
        == len(outcome["inner_trace"])
        == (physics_steps + 4) // 5,
        "issued command-ledger count",
    )
    for global_index, (nominal_row, executed_row, inner_row) in enumerate(
        zip(outcome["nominal_ledger"], outcome["executed_ledger"], outcome["inner_trace"])
    ):
        coordinate = (global_index // 5, global_index % 5)
        require(
            (nominal_row["high_level_index"], nominal_row["inner_control_index"])
            == coordinate,
            "nominal command coordinates",
        )
        require(
            (executed_row["high_level_index"], executed_row["inner_control_index"])
            == coordinate,
            "executed command coordinates",
        )
        require(
            (inner_row["high_level_index"], inner_row["inner_control_index"])
            == coordinate,
            "inner trace coordinates",
        )
    require(
        execution["terminal_simulator_state_sha256"]
        == outcome["terminal_simulator_state_sha256"],
        "terminal simulator-state hash",
    )
    require(
        execution["terminal_observation_sha256"]
        == outcome["terminal_observation_sha256"],
        "terminal observation hash",
    )

    clock = outcome["physics_clock"]
    require(
        math.isclose(
            float(clock["observed_exposure_s"]),
            physics_steps * 0.002,
            rel_tol=0.0,
            abs_tol=1e-10,
        ),
        "MuJoCo clock exposure",
    )
    require(
        float(clock["expected_from_completed_substeps_s"])
        == physics_steps * 0.002,
        "expected 2 ms exposure",
    )
    physics_trace = outcome["physics_trace"]
    require(len(physics_trace) == physics_steps, "physics trace 2 ms count")
    settled_eef = outcome["eef_path_audit"][
        "settled_forwarded_eef_position_m"
    ]
    require(
        isinstance(settled_eef, list)
        and len(settled_eef) == 3
        and all(math.isfinite(float(value)) for value in settled_eef),
        "settled forwarded EEF position",
    )
    require(
        int(outcome["eef_path_audit"]["position_sample_count"])
        == 1 + physics_steps,
        "500 Hz end-effector path sample count",
    )

    recomputed_nominal_motion = 0.0
    recomputed_safe_motion = 0.0
    recomputed_measured_motion = 0.0
    recomputed_correction_motion = 0.0
    recomputed_eef_path = 0.0
    tracking_squared_error = 0.0
    tracking_maximum_linf = 0.0
    recomputed_first_tracking_crossing = None
    previous_position = [float(value) for value in settled_eef]
    tracking_summary = outcome["tracking"]
    linf_threshold = float(tracking_summary["maximum_linf_threshold_rad_s"])
    rmse_threshold = float(tracking_summary["maximum_rmse_threshold_rad_s"])

    def norm(vector: Sequence[float]) -> float:
        return math.sqrt(sum(float(value) ** 2 for value in vector))

    for global_substep, row in enumerate(physics_trace):
        coordinate = (
            global_substep // 25,
            (global_substep // 5) % 5,
            global_substep % 5,
        )
        require(
            (
                int(row["high_level_index"]),
                int(row["inner_control_index"]),
                int(row["physics_substep_index"]),
            )
            == coordinate,
            "physics trace coordinates",
        )
        command_index = global_substep // 5
        nominal = row["nominal_joint_velocity_command_rad_s"]
        issued = row["issued_joint_velocity_command_rad_s"]
        measured = row["measured_arm_joint_velocity_rad_s"]
        position = row["forwarded_eef_position_m"]
        for label, vector, length in (
            ("nominal command", nominal, 7),
            ("issued command", issued, 7),
            ("measured arm qvel", measured, 7),
            ("forwarded EEF position", position, 3),
        ):
            require(
                isinstance(vector, list)
                and len(vector) == length
                and all(math.isfinite(float(value)) for value in vector),
                "physics trace " + label,
            )
        require(
            json_equal(
                nominal,
                outcome["nominal_ledger"][command_index]["qdot_rad_s"],
            ),
            "physics trace nominal-command binding",
        )
        require(
            json_equal(
                issued,
                outcome["executed_ledger"][command_index]["qdot_rad_s"],
            ),
            "physics trace issued-command binding",
        )
        difference = [
            float(safe_value) - float(nominal_value)
            for safe_value, nominal_value in zip(issued, nominal)
        ]
        tracking_error = [
            float(measured_value) - float(command_value)
            for measured_value, command_value in zip(measured, issued)
        ]
        recomputed_nominal_motion += norm(nominal) * 0.002
        recomputed_safe_motion += norm(issued) * 0.002
        recomputed_measured_motion += norm(measured) * 0.002
        recomputed_correction_motion += norm(difference) * 0.002
        recomputed_eef_path += norm(
            [
                float(value) - float(reference)
                for value, reference in zip(position, previous_position)
            ]
        )
        previous_position = [float(value) for value in position]
        row_linf = max(abs(value) for value in tracking_error)
        tracking_squared_error += sum(value * value for value in tracking_error)
        tracking_maximum_linf = max(tracking_maximum_linf, row_linf)
        cumulative_rmse = math.sqrt(
            tracking_squared_error / (7 * (global_substep + 1))
        )
        if (
            recomputed_first_tracking_crossing is None
            and (row_linf > linf_threshold or cumulative_rmse > rmse_threshold)
        ):
            recomputed_first_tracking_crossing = {
                "high_level_index": coordinate[0],
                "inner_control_index": coordinate[1],
                "physics_substep_index": coordinate[2],
                "error_linf_rad_s": row_linf,
                "cumulative_rmse_rad_s": cumulative_rmse,
                "linf_threshold_rad_s": linf_threshold,
                "rmse_threshold_rad_s": rmse_threshold,
                "command_rad_s": [float(value) for value in issued],
                "measured_rad_s": [float(value) for value in measured],
            }

    def require_close(actual: Any, expected: float, label: str) -> None:
        require(
            math.isclose(
                float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-14
            ),
            label,
        )

    usefulness = outcome["usefulness"]
    require_close(
        usefulness["nominal_joint_motion_integral_rad"],
        recomputed_nominal_motion,
        "nominal joint-motion integral",
    )
    require_close(
        usefulness["safe_joint_motion_integral_rad"],
        recomputed_safe_motion,
        "safe joint-motion integral",
    )
    require_close(
        usefulness["measured_joint_motion_integral_rad"],
        recomputed_measured_motion,
        "measured joint-motion integral",
    )
    require_close(
        usefulness["correction_integral_rad"],
        recomputed_correction_motion,
        "correction integral",
    )
    require_close(
        usefulness["eef_path_length_m"],
        recomputed_eef_path,
        "500 Hz EEF path length",
    )
    command_count = len(outcome["executed_ledger"])
    zero_command_count = sum(
        norm(row["qdot_rad_s"]) <= 1e-12 for row in outcome["executed_ledger"]
    )
    require_close(
        usefulness["zero_motion_fraction"],
        zero_command_count / command_count if command_count else 1.0,
        "zero-command fraction",
    )
    require(
        bool(usefulness["all_issued_arm_joint_commands_zero"])
        == bool(command_count and zero_command_count == command_count),
        "all-issued-commands-zero endpoint",
    )
    require(
        bool(usefulness["safety_by_no_execution"]) == (physics_steps == 0),
        "safety-by-no-execution endpoint",
    )
    require_close(
        usefulness["motion_retention_ratio"],
        (
            recomputed_safe_motion / recomputed_nominal_motion
            if recomputed_nominal_motion > 0.0
            else 0.0
        ),
        "motion-retention ratio",
    )

    recomputed_tracking_rmse = (
        math.sqrt(tracking_squared_error / (7 * physics_steps))
        if physics_steps
        else 0.0
    )
    require_close(
        tracking_summary["maximum_linf_error_rad_s"],
        tracking_maximum_linf,
        "tracking Linf",
    )
    require_close(
        tracking_summary["cumulative_rmse_rad_s"],
        recomputed_tracking_rmse,
        "tracking cumulative RMSE",
    )
    require(
        json_equal(
            tracking_summary["first_threshold_crossing"],
            recomputed_first_tracking_crossing,
        ),
        "tracking first-threshold crossing recomputation",
    )

    endpoints = scientific_payload["endpoints"]
    require(json_equal(endpoints["task"], outcome["task"]), "task endpoint")
    require(
        json_equal(endpoints["usefulness"], outcome["usefulness"]),
        "usefulness endpoint",
    )
    require(json_equal(endpoints["validity"], outcome["validity"]), "validity endpoint")
    require(json_equal(endpoints["safety"]["paper_car"], outcome["paper_car"]), "CAR endpoint")

    car_ledger = outcome["paper_car_position_ledger"]
    require(
        _sha256(_canonical(car_ledger))
        == endpoints["safety"]["paper_car"]["position_ledger_sha256"],
        "CAR position-ledger hash",
    )
    car_rows = car_ledger["completed_post_step_positions"]
    require(len(car_rows) == len(completed_actions), "CAR completed-position count")
    positions = []
    settled_position = car_ledger["settled_active_obstacle_root_position_m"]
    for index, row in enumerate(car_rows):
        require(int(row["high_level_index"]) == index, "CAR position index")
        position = row["position_m"]
        displacement = sum(
            abs(float(value) - float(reference))
            for value, reference in zip(position, settled_position)
        )
        require(
            math.isclose(
                displacement,
                float(row["l1_displacement_from_settled_m"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "CAR per-step displacement",
        )
        positions.append(position)
    from main.poisson_fullbody.active_canary import paper_car_endpoint

    recomputed_car = paper_car_endpoint(
        settled_position,
        positions,
        fixed_exposure_complete=bool(outcome["exposure_complete"]),
    )
    for field, value in recomputed_car.items():
        require(outcome["paper_car"].get(field) == value, "CAR recomputation." + field)

    monitor = outcome["monitor"]
    safety = endpoints["safety"]
    monitor_projection = {
        "D_sim_min_m": monitor["D_sim_min_m"],
        "any_robot_obstacle_contact": monitor["any_contact"],
        "link56_obstacle_contact": monitor["link56_contact"],
        "total_physical_contact_point_record_count": monitor["total"],
        "settled_physical_contact_point_record_count": monitor["settled"],
        "rollout_phase_physical_contact_point_record_count": monitor["rollout"],
        "live_solver_nonpositive_contact_point_record_count": monitor["live"],
        "post_state_physical_contact_point_record_count": monitor["post"],
        "first_physical_contact_point_record": monitor["first"],
        "first_settled_physical_contact_point_record": monitor["first_settled"],
        "first_live_solver_physical_contact_point_record": monitor["first_live"],
        "first_post_state_physical_contact_point_record": monitor["first_post"],
    }
    for field, value in monitor_projection.items():
        if field == "D_sim_min_m":
            require(
                safety[field]["available"] and safety[field]["value"] == value,
                "monitor projection." + field,
            )
        else:
            require(json_equal(safety[field], value), "monitor projection." + field)
    require(
        int(monitor["total"]) == int(monitor["settled"]) + int(monitor["rollout"]),
        "contact phase total",
    )
    require(
        int(monitor["rollout"]) == int(monitor["live"]) + int(monitor["post"]),
        "rollout phase total",
    )
    if physics_steps:
        measurement_record = monitor["record"]
        require(
            int(measurement_record["observed_physics_substeps"]) == physics_steps,
            "monitor observed physics count",
        )
        for field in (
            "total_physical_contact_point_record_count",
            "rollout_phase_physical_contact_point_record_count",
            "live_solver_nonpositive_contact_point_record_count",
            "post_state_physical_contact_point_record_count",
        ):
            projected = {
                "total_physical_contact_point_record_count": "total",
                "rollout_phase_physical_contact_point_record_count": "rollout",
                "live_solver_nonpositive_contact_point_record_count": "live",
                "post_state_physical_contact_point_record_count": "post",
            }[field]
            require(
                int(measurement_record[field]) == int(monitor[projected]),
                "serialized monitor." + field,
            )

    optimizer = scientific_payload["optimizer"]
    for field, value in outcome["optimizer_counts"].items():
        require(optimizer[field] == value, "optimizer." + field)
    require(
        optimizer["terminal_status"] == outcome["optimizer_terminal_status"],
        "optimizer terminal status",
    )
    minimum_mapping = {
        "h_min": "h_m2",
        "minimum_nominal_cbf_residual": "nominal_raw_cbf_residual_m2_per_s",
        "minimum_safe_cbf_residual": "safe_raw_cbf_residual_m2_per_s",
        "minimum_realized_cbf_residual": "realized_raw_cbf_residual_m2_per_s",
        "D_opt_min_m": "D_opt_m",
    }
    for endpoint_name, outcome_name in minimum_mapping.items():
        record = safety[endpoint_name]
        value = outcome["minimums"][outcome_name]
        require(bool(record["available"]) == (value is not None), endpoint_name + " availability")
        require(record["value"] == value, endpoint_name + " value")

    initial_audit = outcome["initial_protected_sample_audit"]
    require(
        int(initial_audit["protected_sample_count"])
        == int(initial_audit["field_query_count"])
        and 0
        <= int(initial_audit["valid_field_query_count"])
        <= int(initial_audit["field_query_count"]),
        "initial protected-sample query counts",
    )
    require(
        bool(initial_audit["strict_safe_start"])
        == bool(outcome["validity"]["poisson_safe_start"]),
        "initial strict safe-start",
    )
    dopt_values = [float(initial_audit["D_opt_min_m"])]
    dopt_values.extend(
        float(row["D_opt_min_m"])
        for row in outcome["inner_trace"]
        if row.get("D_opt_min_m") is not None
    )
    dopt_values.extend(
        float(row["D_opt_min_m"])
        for row in outcome["fail_closed_attempt_trace"]
        if row.get("D_opt_min_m") is not None
    )
    dopt_values.extend(
        float(row["D_opt_min_m"])
        for row in outcome["realized_cbf_trace"]
        if row.get("D_opt_min_m") is not None
    )
    require_close(
        outcome["minimums"]["D_opt_m"],
        min(dopt_values),
        "D_opt trace minimum",
    )
    if outcome["arm"] == "joint_velocity_psf_link56":
        h_values = []
        if initial_audit.get("minimum_h_m2") is not None:
            h_values.append(float(initial_audit["minimum_h_m2"]))
        h_values.extend(
            float(row["minimum_h_m2"])
            for row in outcome["inner_trace"]
            if row.get("minimum_h_m2") is not None
        )
        h_values.extend(
            float(row["minimum_h_m2"])
            for row in outcome["fail_closed_attempt_trace"]
            if row.get("minimum_h_m2") is not None
        )
        h_values.extend(
            float(row["minimum_h_m2"])
            for row in outcome["realized_cbf_trace"]
            if row.get("minimum_h_m2") is not None
        )
        require(
            outcome["minimums"]["h_m2"]
            == (min(h_values) if h_values else None),
            "h trace minimum",
        )
        nominal_residual_values = [
            float(row["minimum_nominal_cbf_residual_m2_per_s"])
            for row in outcome["inner_trace"]
            if row.get("minimum_nominal_cbf_residual_m2_per_s") is not None
        ]
        safe_residual_values = [
            float(row["minimum_safe_cbf_residual_m2_per_s"])
            for row in outcome["inner_trace"]
            if row.get("minimum_safe_cbf_residual_m2_per_s") is not None
        ]
        require(
            outcome["minimums"]["nominal_raw_cbf_residual_m2_per_s"]
            == (min(nominal_residual_values) if nominal_residual_values else None),
            "nominal CBF trace minimum",
        )
        require(
            outcome["minimums"]["safe_raw_cbf_residual_m2_per_s"]
            == (min(safe_residual_values) if safe_residual_values else None),
            "safe CBF trace minimum",
        )

    tracking = outcome["tracking"]
    require(int(tracking["observed_physics_substep_count"]) == physics_steps, "tracking 2 ms count")
    require(
        int(tracking["command_count"]) == len(outcome["executed_ledger"]),
        "tracking command count",
    )
    require(
        tracking["first_threshold_crossing"]
        == outcome["validity"]["first_velocity_tracking_threshold_crossing"],
        "tracking first crossing",
    )

    realized_trace = outcome["realized_cbf_trace"]
    realized_audit = outcome["realized_cbf_audit"]
    if outcome["arm"] == "joint_velocity_psf_link56":
        require(len(realized_trace) == physics_steps, "realized CBF 2 ms trace count")
        evaluated = sum(
            int(row["query_count"]) for row in realized_trace if row.get("valid") is True
        )
        negative_rows = [
            row
            for row in realized_trace
            if row.get("valid") is True
            and float(row["minimum_realized_cbf_residual_m2_per_s"]) < 0.0
        ]
        realized_values = [
            float(row["minimum_realized_cbf_residual_m2_per_s"])
            for row in realized_trace
            if row.get("valid") is True
        ]
        require(
            int(realized_audit["residual_evaluation_count"]) == evaluated,
            "realized CBF evaluation count",
        )
        require(
            int(realized_audit["negative_residual_callback_count"])
            == len(negative_rows),
            "realized CBF negative count",
        )
        require(
            outcome["minimums"]["realized_raw_cbf_residual_m2_per_s"]
            == (min(realized_values) if realized_values else None),
            "realized CBF minimum",
        )
        first_negative = (
            negative_rows[0].get("first_negative_crossing") if negative_rows else None
        )
        require(
            realized_audit["first_negative_residual"] == first_negative,
            "realized CBF first negative crossing",
        )
    else:
        require(not realized_trace, "adapter-only realized CBF trace must be empty")
        require(
            int(realized_audit["observed_physics_substep_count"]) == 0
            and int(realized_audit["residual_evaluation_count"]) == 0,
            "adapter-only realized CBF audit must be empty",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("manifests/vlsa_poisson_link56_aegis_car_109.v1.jsonl"))
    parser.add_argument("--selection-protocol", type=Path, default=Path("configs/vlsa_poisson_link56_feasibility.v1.json"))
    parser.add_argument("--historical-result-root", required=True, type=Path)
    parser.add_argument("--checkpoint-receipt", required=True, type=Path)
    parser.add_argument("--numeric-validation-result", required=True, type=Path)
    parser.add_argument("--parity-result", required=True, type=Path)
    parser.add_argument("--shadow-identification-result", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    arguments = parser.parse_args()

    root = arguments.repo_root.resolve()
    os.chdir(str(root))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    for import_root in (root / "main", root / "safelibero"):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))
    try:
        output = _validated_run_output(arguments.output_root, arguments.run_id)
    except ActiveRunnerError as error:
        print("active canary refused: %s" % error, file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "run_receipt.json"
    started = time.time()
    try:
        from main.poisson_fullbody.contracts import (
            attach_payload_hash,
            load_hashed_json,
            publish_hashed_json,
            sha256_file,
        )
        from main.poisson_fullbody.active_canary import (
            fixed_exposure_action_contract,
            ledger_sha256,
        )
        from main.poisson_fullbody.field_bundle import build_static_field_bundle
        from main.poisson_fullbody.measurement import (
            FullRobotObstacleMonitor,
            clone_forwarded_state,
            resolve_collision_geom_sets,
        )
        from main.poisson_fullbody.result_schema import (
            publish_episode_result,
            validate_active_canary_pair,
            validate_episode_result,
        )
        from main.poisson_fullbody.robot_samples import validate_rigid_roundtrip
        from main.poisson_fullbody.shadow_replay import load_historical_action_replay
        from main.poisson_fullbody.surface_sampling import build_robot_collision_samples
        from scripts.run_poisson_shadow_identification import (
            _load_bound_runtime_protocol,
            _model_body_id,
            _official_integration_state,
            _require_upstream_parity,
        )
        from scripts.run_poisson_shadow_parity import _prepare_environment
        import main.evaluate_safelibero_aegis as evaluator

        if receipt_path.exists() or receipt_path.is_symlink():
            existing_receipt = load_hashed_json(receipt_path)
            if existing_receipt.get("status") == "failed":
                raise ActiveRunnerError(
                    "this run ID already has a failed immutable receipt; choose a new run ID"
                )
            if existing_receipt.get("status") != "complete":
                raise ActiveRunnerError(
                    "this run ID already has an unsupported immutable receipt"
                )

        source_git = _git(root)
        allocation = _allocation()
        manifest_path = (root / arguments.manifest).resolve() if not arguments.manifest.is_absolute() else arguments.manifest.resolve()
        selection_path = (root / arguments.selection_protocol).resolve() if not arguments.selection_protocol.is_absolute() else arguments.selection_protocol.resolve()
        case, case_row_hash = _load_case(manifest_path, arguments.case_id)
        protocol, protocol_hashes, selection, runtime_protocol_path = _load_bound_runtime_protocol(
            root=root,
            case=case,
            selection_config_path=selection_path,
        )
        if selection.get("required_arms") != list(ARMS):
            raise ActiveRunnerError("selection protocol required-arm declaration differs")
        historical_root = arguments.historical_result_root.resolve()
        source_contract = _load_historical_source_contract(
            historical_root, selection["source"]
        )
        source_historical = selection["source"]["historical_results"]
        checkpoint = _checkpoint_identity(
            arguments.checkpoint_receipt.resolve(),
            expected_tree_sha256=source_historical["pi05_tree_sha256"],
            expected_receipt_file_sha256=source_historical[
                "pi05_hash_receipt_sha256"
            ],
            expected_schema_version=source_historical[
                "pi05_hash_receipt_schema_version"
            ],
        )
        historical_path = _bound_historical_result_path(historical_root, case)
        replay = load_historical_action_replay(
            historical_path, expected_case_id=case["case_id"]
        )
        _validate_replay_against_manifest(replay, case)
        if len(replay.actions) != 237:
            raise ActiveRunnerError("first active canary must retain its exact 237-action source exposure")
        frozen_actions = fixed_exposure_action_contract(
            replay.actions,
            expected_count=237,
            expected_sha256=replay.executed_sequence_sha256,
        )
        rotation_components_exactly_zero = all(
            all(float(value) == 0.0 for value in action[3:6])
            for action in frozen_actions
        )
        if not rotation_components_exactly_zero:
            raise ActiveRunnerError(
                "translation-only adapter would discard nonzero historical rotation commands"
            )
        manifest_sha256 = _file_sha256(manifest_path)
        numeric_prerequisite = _require_numeric_prerequisite(
            arguments.numeric_validation_result.resolve(),
            source_commit=source_git["commit"],
        )
        parity_prerequisite = _require_upstream_parity(
            arguments.parity_result.resolve(),
            case_id=case["case_id"],
            source_commit=source_git["commit"],
            historical_payload_sha256=replay.result_payload_sha256,
            manifest_sha256=manifest_sha256,
            manifest_row_sha256=case_row_hash,
            expected_callback_count=25 * len(replay.actions),
        )
        identification_prerequisite = _require_identification_prerequisite(
            arguments.shadow_identification_result.resolve(),
            case=case,
            case_row_hash=case_row_hash,
            source_commit=source_git["commit"],
            manifest_sha256=manifest_sha256,
            selection_sha256=_file_sha256(selection_path),
            runtime_protocol_raw_sha256=_file_sha256(runtime_protocol_path),
            runtime_protocol_semantic_sha256=protocol_hashes.protocol_sha256,
            runtime_parameter_block_sha256=protocol_hashes.parameter_block_sha256,
            alpha_gain_per_s=float(protocol["cbf"]["alpha_gain_per_s"]),
            static_drift_thresholds={
                "translation_m": float(
                    protocol["admissibility"][
                        "max_selected_geom_translation_drift_m"
                    ]
                ),
                "rotation_rad": float(
                    protocol["admissibility"][
                        "max_selected_geom_rotation_drift_rad"
                    ]
                ),
                "surface_m": float(
                    protocol["admissibility"][
                        "max_selected_geom_surface_drift_m"
                    ]
                ),
            },
            replay=replay,
            parity=parity_prerequisite,
        )
        run_contract = {
            "schema_version": "vlsa_poisson_staged_two_arm_canary_run.v1",
            "study_completion": "staged_first_canary_only_not_four_arm_109_case_study",
            "case_id": case["case_id"],
            "arms": list(ARMS),
            "source_action_count": len(replay.actions),
            "source_action_sha256": replay.executed_sequence_sha256,
            "active_domain_separated_source_action_sha256": ledger_sha256(
                frozen_actions
            ),
            "source_rotation_components_3_to_5_exactly_zero": True,
            "source_gripper_component_preserved_exactly": True,
            "source_action_semantics": replay.provenance()["replay_semantics"],
            "active_policy_query_count": 0,
            "protocol_parameter_block_sha256": protocol_hashes.parameter_block_sha256,
            "historical_source_run_contract_sha256": source_contract["sha256"],
            "historical_pi05_tree_sha256": source_contract["pi05_tree_sha256"],
            "historical_pi05_hash_receipt_sha256": source_contract[
                "pi05_hash_receipt_sha256"
            ],
            "numeric_prerequisite_payload_sha256": numeric_prerequisite[
                "result_payload_sha256"
            ],
            "parity_prerequisite_payload_sha256": parity_prerequisite[
                "result_payload_sha256"
            ],
            "identification_prerequisite_payload_sha256": (
                identification_prerequisite["result_payload_sha256"]
            ),
            "code_commit": source_git["commit"],
        }
        run_contract_sha = _sha256(_canonical(run_contract))
        receipt_identity = {
            "run_id": arguments.run_id,
            "case_id": case["case_id"],
            "run_contract_sha256": run_contract_sha,
            "code_commit": source_git["commit"],
            "manifest_sha256": manifest_sha256,
            "manifest_record_sha256": case_row_hash,
            "selection_config_sha256": _file_sha256(selection_path),
            "runtime_protocol_raw_sha256": _file_sha256(runtime_protocol_path),
            "runtime_protocol_semantic_sha256": protocol_hashes.protocol_sha256,
            "runtime_parameter_block_sha256": protocol_hashes.parameter_block_sha256,
            "checkpoint_tree_sha256": checkpoint["checkpoint_sha256"],
            "checkpoint_receipt_file_sha256": checkpoint["receipt_file_sha256"],
            "historical_source_run_contract_sha256": source_contract["sha256"],
            "historical_result_file_sha256": replay.result_file_sha256,
            "historical_result_payload_sha256": replay.result_payload_sha256,
            "source_action_ledger_sha256": replay.executed_sequence_sha256,
            "numeric_prerequisite_payload_sha256": numeric_prerequisite[
                "result_payload_sha256"
            ],
            "parity_prerequisite_payload_sha256": parity_prerequisite[
                "result_payload_sha256"
            ],
            "identification_prerequisite_payload_sha256": (
                identification_prerequisite["result_payload_sha256"]
            ),
        }
        if receipt_path.exists():
            early_arm_identities = {
                arm: _expected_resume_identity(
                    root=root,
                    run_id=arguments.run_id,
                    case=case,
                    case_row_hash=case_row_hash,
                    manifest_path=manifest_path,
                    selection_path=selection_path,
                    runtime_protocol_path=runtime_protocol_path,
                    protocol_hashes=protocol_hashes,
                    source_git=source_git,
                    checkpoint=checkpoint,
                    replay=replay,
                    field_sha256="",
                    run_contract_sha256=run_contract_sha,
                    arm=arm,
                )
                for arm in ARMS
            }
            _validate_complete_run_receipt(
                receipt_path=receipt_path,
                output=output,
                expected_receipt_identity=receipt_identity,
                expected_arm_identities=early_arm_identities,
            )
            print(
                json.dumps(
                    {
                        "status": "complete_reused",
                        "run_id": arguments.run_id,
                        "receipt": str(receipt_path),
                    },
                    sort_keys=True,
                )
            )
            return 0

        runtime = evaluator._runtime_imports(include_aegis=False)
        source_env = None
        try:
            source_env, task, source_observation, goal_atoms, previous_goal = _prepare_environment(
                evaluator, runtime, case, replay
            )
            del goal_atoms, previous_goal
            settled_state = runtime["np"].asarray(
                source_env.sim.get_state().flatten(), dtype=runtime["np"].float64
            ).copy()
            source_pairing = replay.provenance()
            obstacle_name, _ = evaluator._active_obstacle(source_env, source_observation)
            if obstacle_name != case.get("active_obstacle_name"):
                raise ActiveRunnerError(
                    "active selected obstacle differs from the immutable manifest"
                )
            authority = evaluator._contact_model_authority(source_env, obstacle_name)
            robot_root_name = source_env.robots[0].robot_model.root_body
            robot_root = _model_body_id(source_env.sim.model, robot_root_name)
            if robot_root not in set(
                int(value) for value in authority["robot_body_ids"]
            ):
                raise ActiveRunnerError(
                    "robot root is absent from the contact-model authority"
                )
            arm_dof_indices = tuple(
                int(value)
                for value in source_env.robots[0]._ref_joint_vel_indexes
            )
            link_ids = tuple(
                _model_body_id(source_env.sim.model, name)
                for name in protocol["claim_scope"]["protected_robot_bodies"]
            )
            raw_model, raw_data = _raw_model_data(source_env.sim)
            resolved = resolve_collision_geom_sets(
                raw_model,
                robot_root_body_ids=(robot_root,),
                obstacle_root_body_ids=(int(authority["active_obstacle_root_body_id"]),),
                link56_body_ids=link_ids,
            )
            source_official_before = _official_integration_state(source_env.sim, runtime["np"])
            bundle = build_static_field_bundle(
                raw_model,
                raw_data,
                resolved=resolved,
                protocol=protocol,
                protocol_hashes=protocol_hashes,
            )
            forwarded = clone_forwarded_state(raw_model, raw_data)
            full_samples = build_robot_collision_samples(
                source_env.sim.model,
                forwarded,
                geom_ids=resolved.robot_geom_ids,
                epsilon_m=float(protocol["coverage"]["epsilon_m"]),
            )
            sampled_geom_ids = tuple(
                int(record["geom_id"]) for record in full_samples.geom_records
            )
            if sampled_geom_ids != tuple(
                int(value) for value in resolved.robot_geom_ids
            ):
                raise ActiveRunnerError(
                    "full-robot measurement samples differ from authoritative resolved geoms"
                )
            roundtrip = validate_rigid_roundtrip(full_samples.samples, forwarded)
            if not roundtrip["passed"]:
                raise ActiveRunnerError("full-robot measurement surface roundtrip failed")
            full_sampling_evidence = {
                "sample_count": len(full_samples.samples),
                "sample_ledger_sha256": full_samples.sample_ledger_sha256,
                "geom_records": list(full_samples.geom_records),
                "epsilon_m": full_samples.epsilon_m,
                "maximum_surface_cover_radius_m": (
                    full_samples.maximum_surface_cover_radius_m
                ),
                "coverage_semantics": full_samples.coverage_semantics,
                "roundtrip": roundtrip,
            }
            _require_full_robot_sampling_evidence(
                resolved.to_dict(),
                full_sampling_evidence,
                roundtrip_field="roundtrip",
            )
            source_official_after = _official_integration_state(source_env.sim, runtime["np"])
            if not runtime["np"].array_equal(source_official_before, source_official_after):
                raise ActiveRunnerError("field construction changed source integration state")
            _require_identification_matches_active_construction(
                identification_prerequisite,
                case=case,
                active_obstacle_name=obstacle_name,
                contact_model_authority_sha256=authority["authority_sha256"],
                robot_root_body_name=robot_root_name,
                robot_root_body_ids=(robot_root,),
                arm_dof_indices=arm_dof_indices,
                resolved_geometry=resolved.to_dict(),
                field_bundle=bundle,
                full_robot_sampling=full_sampling_evidence,
                settled_integration_state_sha256=evaluator.array_sha256(
                    source_official_before
                ),
                settled_integration_state_length=int(
                    source_official_before.size
                ),
            )

            results = {}
            for arm in ARMS:
                arm_dir = output / case["case_id"] / arm
                arm_dir.mkdir(parents=True, exist_ok=True)
                final_path = arm_dir / "result.json"
                if final_path.exists():
                    raise ActiveRunnerError(
                        "partial arm result exists without a complete immutable run "
                        "receipt; choose a new run ID instead of mixing runtime state"
                    )
                try:
                    outcome = _run_arm(
                        arm=arm,
                        evaluator=evaluator,
                        runtime=runtime,
                        case=case,
                        replay=replay,
                        source_env=source_env,
                        source_settled_observation=source_observation,
                        settled_state=settled_state,
                        bundle=bundle,
                        resolved=resolved,
                        full_samples=full_samples,
                        protocol=protocol,
                    )
                except Exception as error:
                    arm_error = {
                        "schema_version": "vlsa_poisson_active_arm_error.v1",
                        "status": "failed_nonfinal",
                        "scientific_result": False,
                        "run_id": arguments.run_id,
                        "case_id": case["case_id"],
                        "arm": arm,
                        "retained_not_silently_dropped": True,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                    publish_hashed_json(arm_dir / "arm_error.json", arm_error)
                    raise
                trace_payload = {
                    "schema_version": "vlsa_poisson_active_arm_trace.v2",
                    "scientific_result": False,
                    "run_id": arguments.run_id,
                    "case_id": case["case_id"],
                    "arm": arm,
                    "source_replay": replay.provenance(),
                    "staged_scope": "first_two_arm_canary_only",
                    "field_bundle_hashes": asdict(bundle.hashes),
                    "field_bundle_diagnostics": asdict(bundle.diagnostics),
                    "resolved_geometry": resolved.to_dict(),
                    "full_robot_surface_sampling": full_sampling_evidence,
                    "outcome": outcome,
                }
                trace_path = arm_dir / "trace.json"
                publish_hashed_json(trace_path, trace_payload)
                serialized_trace = load_hashed_json(trace_path)
                reference = {
                    "relative_path": str(trace_path.relative_to(output)),
                    "bytes": trace_path.stat().st_size,
                    "sha256": sha256_file(trace_path),
                    "artifact_type": "active_arm_audit_trace",
                    "media_type": "application/json",
                }
                payload = _scientific_payload(
                    root=root,
                    run_id=arguments.run_id,
                    case=case,
                    case_row_hash=case_row_hash,
                    manifest_path=manifest_path,
                    selection_path=selection_path,
                    runtime_protocol_path=runtime_protocol_path,
                    protocol_hashes=protocol_hashes,
                    checkpoint=checkpoint,
                    replay=replay,
                    allocation=allocation,
                    source_git=source_git,
                    source_pairing=source_pairing,
                    bundle=bundle,
                    resolved=resolved,
                    arm_outcome=outcome,
                    artifact_reference=reference,
                    run_contract_sha256=run_contract_sha,
                )
                _validate_trace_payload_against_scientific(
                    serialized_trace,
                    payload,
                )
                # Validate the exact object that will be atomically published.  This
                # deliberately runs before publication so every fail-closed partial
                # completion is subject to the same scientific schema as a full run.
                validate_episode_result(attach_payload_hash(payload))
                publish_episode_result(final_path, payload, artifact_root=output)
                results[arm] = load_hashed_json(final_path)
        finally:
            if source_env is not None:
                source_env.close()

        pair = validate_active_canary_pair(
            results["joint_velocity_adapter_only"],
            results["joint_velocity_psf_link56"],
        )
        pair_payload = dict(pair)
        pair_payload.update(
            {
                "status": "complete",
                "scientific_result": False,
                "four_arm_109_case_study_complete": False,
                "run_contract_sha256": run_contract_sha,
            }
        )
        pair_path = output / case["case_id"] / "pair_result.json"
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        publish_hashed_json(pair_path, pair_payload)
        run_receipt = {
            "schema_version": "vlsa_poisson_active_canary_run_receipt.v2",
            "status": "complete",
            "scientific_result": False,
            "run_id": arguments.run_id,
            "case_id": case["case_id"],
            "staged_scope": "two_arms_one_canary_not_four_arm_109_case_study",
            "identity": receipt_identity,
            "arm_results": {
                arm: {
                    "relative_path": "%s/%s/result.json" % (case["case_id"], arm),
                    "sha256": sha256_file(
                        output / case["case_id"] / arm / "result.json"
                    ),
                }
                for arm in ARMS
            },
            "pair_result_relative_path": str(pair_path.relative_to(output)),
            "pair_result_sha256": _file_sha256(pair_path),
            "elapsed_seconds": time.time() - started,
        }
        publish_hashed_json(receipt_path, run_receipt)
        return 0
    except Exception as error:
        failure = {
            "schema_version": "vlsa_poisson_active_canary_run_receipt.v2",
            "status": "failed",
            "scientific_result": False,
            "run_id": arguments.run_id,
            "case_id": arguments.case_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        }
        try:
            from main.poisson_fullbody.contracts import publish_hashed_json

            if not receipt_path.exists():
                publish_hashed_json(receipt_path, failure)
        except Exception:
            pass
        print("active canary failed: %s: %s" % (type(error).__name__, error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
