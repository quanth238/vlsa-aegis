"""Fail-closed CPU publication for the one-case CFS-00A canary.

The source H100 allocation owns raw evidence only.  This module runs in the
independent CPU ``afterany`` allocation, reparses the telemetry and focused
test log, semantically validates every present raw scientific payload, builds
the envelope twice, and is the sole owner of the atomic ``results.json``
publication.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from .r05a_full_lifetime_telemetry import parse_full_lifetime_telemetry
from .r05a_sampled_current_canary import (
    _atomic_write_json,
    _host_envelope,
    _load_hashed_object,
    _load_object,
    _publication_directory_binding,
    _regular_file_identity,
    _require_exact_child_directory,
    _require_unchanged_publication_directories,
    _resolve_exact,
    _validate_schema,
    file_sha256,
)


SCHEMA_VERSION = "1.0"
ARTIFACT_TYPE = "r05a_constrained_flow_canary_envelope"
CASE_ID = "crfs-1069f29a8d76463a"
SOURCE_NODE = "worker-1"
EXPERIMENT_ROOT = Path("/mnt/data/quanth/experiments/crfs-oracle")
APPARATUS_CONFIG_PATH = (
    "configs/experiments/r05a_constrained_flow_canary_apparatus.json"
)
SCIENTIFIC_CONFIG_PATH = "configs/experiments/r05a_constrained_flow_canary.json"
LEGACY_CONFIG_PATH = "configs/experiments/r05a_inverse_flow_canary.json"
ENVELOPE_SCHEMA_PATH = "schemas/r05a-constrained-flow-canary-envelope.schema.json"
SEMANTIC_VALIDATOR_PATH = "main/crfs_oracle/r05a_constrained_flow_validation.py"
ALLOCATION_TEST_REGISTRY_PATH = (
    "main/crfs_oracle/r05a_constrained_flow_allocation_tests.json"
)
RELEASE_DECISION_PATH = (
    "docs/decisions/0046-require-fresh-cfs00a-release-bound-to-runtime-evidence.md"
)
HISTORICAL_RELEASE_DECISION_PATH = (
    "docs/decisions/0041-require-exact-constrained-flow-canary-release-identity.md"
)
RUNTIME_IDENTITY_DECISION_PATH = (
    "docs/decisions/0045-accept-runtime-identity-regression.md"
)
RUNTIME_IDENTITY_EVIDENCE_PATH = (
    "evidence/r05a/runtime-identity-regression-20260716a.json"
)
RUNTIME_IDENTITY_PREFLIGHT_PATH = (
    "evidence/r05a/runtime-identity-preflight-20260715T221453Z.txt"
)
RELEASE_BRANCH_REF = "refs/remotes/origin/agent/crfs-oracle-harness"
RELEASE_ONLY_PATHS = (
    SCIENTIFIC_CONFIG_PATH,
    APPARATUS_CONFIG_PATH,
    RELEASE_DECISION_PATH,
)

SOURCE_RESOURCE_CONTRACT = {
    "partition": "main",
    "account": "normal",
    "qos": "normal",
    "gpus": 1,
    "cpus_per_task": 8,
    "host_memory_mib": 65536,
    "time_limit": "02:00:00",
    "array": "0-0%1",
    "requeue": False,
    "validator_partition": "main",
    "validator_account": "normal",
    "validator_qos": "normal",
    "validator_cpus": 2,
    "validator_host_memory_mib": 8192,
    "validator_time_limit": "00:15:00",
    "validator_gpus": 0,
    "validator_dependency": "afterany",
}
APPARATUS_RESOURCE_CONTRACT = {"source_host": SOURCE_NODE, **SOURCE_RESOURCE_CONTRACT}
RUNTIME_IDENTITY_CONTRACT = {
    "validation_helper": "scripts/hpc/lib/r05a_runtime_identity.sh",
    "validation_is_shell_only": True,
    "interpreter_invocation_during_validation_allowed": False,
    "openpi_python": {
        "public_path": "/mnt/data/quanth/venvs/openpi/bin/python",
        "direct_link_target": "/mnt/data/quanth/anaconda3/bin/python",
        "resolved_executable": "/mnt/data/quanth/anaconda3/bin/python3.11",
        "resolved_sha256": "c71718900fe84a9124d39abdd9d68d029930e0dcff1764686d8d6aad97216bc9",
    },
    "libero_python": {
        "public_path": "/mnt/data/quanth/venvs/openpi-libero-client/bin/python",
        "direct_link_target": "/home/quanth/.local/share/uv/python/cpython-3.8-linux-x86_64-gnu/bin/python3.8",
        "resolved_executable": "/home/quanth/.local/share/uv/python/cpython-3.8.20-linux-x86_64-gnu/bin/python3.8",
        "resolved_sha256": "c70efda0ee43d9a0014ee570cad3abb4f46b0c11f6ea88f7c467a91faafd4f62",
    },
}
RUNTIME_IDENTITY_EVIDENCE_BINDING = {
    "decision_path": RUNTIME_IDENTITY_DECISION_PATH,
    "decision_sha256": "3672cfac46d8ffbd5a224e837e871bdc7010884907f367b6e418471a68c1d98e",
    "evidence_path": RUNTIME_IDENTITY_EVIDENCE_PATH,
    "evidence_sha256": "ba83d7d696310456b696ddd0a846e5a6b0554b994204ecf7c568e2b565508237",
    "preflight_path": RUNTIME_IDENTITY_PREFLIGHT_PATH,
    "preflight_sha256": "c351ec194cf838829e82105f1343de242199e93a855b9ce45862b64a6b955221",
    "release_commit": "8415b659a46699757de1e99558713e56b95255b5",
    "job_id": "28043",
    "job_state": "COMPLETED",
    "job_exit_code": "0:0",
    "source_host": SOURCE_NODE,
    "result_sha256": "3bda039cd94bf283ebd2b2d9ff1839ce411037a729e6664efaabff20ce38b0de",
    "shell_only": True,
    "gpus_allocated": 0,
    "cfs_runtime_integration_evaluated": False,
    "h100_submission_authorized_by_evidence": False,
}
VINUNI_H100_GUIDE_CONTRACT = {
    "title": "2026-05-03 - VinUni H100 Server Guide.md",
    "local_reference_path": "/Users/quanth238/Library/Mobile Documents/iCloud~md~obsidian/Documents/LLM Knowledge Base/10 Raw/articles/research-infrastructure/2026-05-03 - VinUni H100 Server Guide.md",
    "sha256": "acee44c535e2fc25f8986e41efe233f21683a71c7fb5fa0ae726f0dae573b108",
    "line_count": 1298,
    "login_node_role": "control_plane_only",
    "allocation_compute_only": True,
    "live_preflight_overrides_examples": True,
    "free_h100_required_before_submission": True,
    "reroute_when_worker_1_busy": False,
    "shared_storage_stop_percent": 90,
}

# Every file whose bytes can influence raw production, validation, source-task
# accounting, or publication is rebound in the pre-release source contract.
BOUND_REPOSITORY_PATHS = frozenset(
    {
        SCIENTIFIC_CONFIG_PATH,
        APPARATUS_CONFIG_PATH,
        LEGACY_CONFIG_PATH,
        ENVELOPE_SCHEMA_PATH,
        SEMANTIC_VALIDATOR_PATH,
        ALLOCATION_TEST_REGISTRY_PATH,
        "manifests/r05a_inverse_flow_teacher_smoke.jsonl",
        "docs/decisions/0040-preregister-same-budget-constrained-flow-diagnostic.md",
        HISTORICAL_RELEASE_DECISION_PATH,
        RUNTIME_IDENTITY_DECISION_PATH,
        RUNTIME_IDENTITY_EVIDENCE_PATH,
        RUNTIME_IDENTITY_PREFLIGHT_PATH,
        RELEASE_DECISION_PATH,
        "schemas/r05a-inverse-flow-canary.schema.json",
        "main/crfs_oracle/r05a_canary.py",
        "main/crfs_oracle/r05a_constrained_flow_canary.py",
        "main/crfs_oracle/r05a_constrained_flow_publication.py",
        "main/crfs_oracle/r05a_full_lifetime_telemetry.py",
        "main/crfs_oracle/r05a_sampled_current_canary.py",
        "main/crfs_oracle/r05a_constrained_flow_allocation_tests.json",
        "main/run_crfs_r05a_canary.py",
        "main/run_crfs_r05a_constrained_flow_canary.py",
        "main/publish_crfs_r05a_constrained_flow_canary.py",
        "openpi/scripts/serve_cfs_policy.py",
        "openpi/scripts/serve_policy.py",
        "openpi/src/openpi/models_pytorch/crfs_inverse_control.py",
        "openpi/src/openpi/models_pytorch/crfs_linearized_control.py",
        "openpi/src/openpi/models_pytorch/pi0_pytorch.py",
        "openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/configuration_gemma.py",
        "openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py",
        "openpi/src/openpi/models_pytorch/transformers_replace/models/paligemma/modeling_paligemma.py",
        "openpi/src/openpi/models_pytorch/transformers_replace/models/siglip/check.py",
        "openpi/src/openpi/models_pytorch/transformers_replace/models/siglip/modeling_siglip.py",
        "openpi/src/openpi/policies/crfs_constrained_flow_adapter.py",
        "openpi/src/openpi/policies/policy.py",
        "scripts/hpc/lib/cgroup_v2_full_lifetime_monitor.sh",
        "scripts/hpc/lib/r05a_allocation_tests.sh",
        "scripts/hpc/lib/r05a_runtime_identity.sh",
        "scripts/hpc/lib/slurm_exact_array_task_status.sh",
        "scripts/hpc/preflight.sh",
        "scripts/hpc/prepare_jsonschema_overlay.sh",
        "scripts/hpc/prepare_transformers_overlay.sh",
        "scripts/hpc/run_r05a_constrained_flow_workload.sh",
        "scripts/hpc/run_r05a_constrained_flow_canary.sh",
        "scripts/hpc/validate_r05a_constrained_flow_canary.sh",
        "scripts/hpc/submit_r05a_constrained_flow_canary.sh",
        "slurm/r05a_constrained_flow_canary_h100.sbatch",
        "slurm/r05a_constrained_flow_canary_validate_cpu.sbatch",
        "tests/test_inverse_flow_control.py",
        "tests/test_inverse_flow_sampler.py",
        "tests/test_inverse_flow_policy.py",
        "tests/test_r05a_canary.py",
        "tests/test_linearized_flow_control.py",
        "tests/test_constrained_flow_adapter.py",
        "tests/test_r05a_constrained_flow_canary.py",
        "tests/test_r05a_constrained_flow_validation.py",
        "tests/test_r05a_constrained_flow_hpc_contract.py",
        "tests/test_r05a_constrained_flow_publication.py",
        "tests/test_r05a_runtime_identity_regression_evidence.py",
    }
)

SOURCE_CONTRACT_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "run_id",
    "git_commit",
    "git_dirty",
    "source_node",
    "gpu_slurm_array_job_id",
    "gpu_slurm_array_task_id",
    "exact_gpu_task_id",
    "resources",
    "artifact_paths",
    "frozen_bindings",
    "repository_file_sha256",
    "held_gpu_submission_path",
    "held_gpu_submission_sha256",
    "live_preflight_path",
    "live_preflight_sha256",
    "scientific_claim_allowed",
    "infeasibility_claim_allowed",
    "simulator_efficacy_claim_allowed",
    "probe_training_authorized",
    "timestamp_utc",
}
HELD_GPU_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "run_id",
    "git_commit",
    "gpu_slurm_array_job_id",
    "gpu_slurm_array_task_id",
    "exact_gpu_task_id",
    "source_node",
    "resources",
    "released_at_receipt_time",
    "scientific_claim_allowed",
    "infeasibility_claim_allowed",
    "simulator_efficacy_claim_allowed",
    "probe_training_authorized",
    "timestamp_utc",
}
SUBMISSION_KEYS = {
    "schema_version",
    "artifact_role",
    "status",
    "run_id",
    "git_commit",
    "source_node",
    "gpu_slurm_array_job_id",
    "gpu_slurm_array_task_id",
    "cpu_afterany_job_id",
    "dependency",
    "source_contract_path",
    "source_contract_sha256",
    "expected_result",
    "expected_validation_receipt",
    "scientific_claim_allowed",
    "infeasibility_claim_allowed",
    "simulator_efficacy_claim_allowed",
    "probe_training_authorized",
    "timestamp_utc",
}
EXECUTION_RELEASE_KEYS = {
    "schema_version",
    "artifact_role",
    "decision_artifact",
    "accepted_implementation_commit",
    "run_id",
    "single_submission",
    "source_host",
    "resources",
    "release_only_parent_required",
    "allowed_release_diff_paths",
    "automatic_resubmission_allowed",
    "automatic_next_experiment_allowed",
}

# The raw historical result deliberately predates allocation telemetry
# finalization.  These exact errors are the only omissions accepted from its
# independent semantic validator; all scientific fields are still checked.
LEGACY_RAW_DEFERRED_ERRORS = [
    "allocation dependency-backed focused tests did not pass",
    "host cgroup path provenance is missing",
    "memory record is missing",
]
OOM_SIGNAL = re.compile(
    r"out[ -]?of[ -]?memory|cuda(?:[^a-z0-9]|_)*(?:oom|error[^a-z0-9]*2)"
    r"|cublas_status_alloc_failed|cudnn_status_alloc_failed|outofmemoryerror"
    r"|memoryerror|std::bad_alloc|cannot allocate memory",
    flags=re.IGNORECASE,
)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _validate_execution_release(
    apparatus_config: Mapping[str, Any], *, run_id: str
) -> dict[str, Any]:
    if apparatus_config.get("ready_to_run") is not True or apparatus_config.get(
        "blocked_on"
    ) != []:
        raise ValueError("constrained-flow apparatus config is not released")
    if apparatus_config.get("resource_contract") != APPARATUS_RESOURCE_CONTRACT:
        raise ValueError("constrained-flow apparatus resource contract changed")
    if apparatus_config.get("runtime_identity_contract") != RUNTIME_IDENTITY_CONTRACT:
        raise ValueError("constrained-flow runtime identity contract changed")
    if (
        apparatus_config.get("runtime_identity_evidence_binding")
        != RUNTIME_IDENTITY_EVIDENCE_BINDING
    ):
        raise ValueError("constrained-flow runtime identity evidence binding changed")
    if apparatus_config.get("vinuni_h100_guide_contract") != VINUNI_H100_GUIDE_CONTRACT:
        raise ValueError("constrained-flow VinUni H100 guide contract changed")
    release = apparatus_config.get("execution_release")
    if type(release) is not dict or set(release) != EXECUTION_RELEASE_KEYS:
        raise ValueError("constrained-flow execution release keys changed")
    expected = {
        "schema_version": "1.0",
        "artifact_role": "r05a_constrained_flow_canary_execution_release",
        "decision_artifact": RELEASE_DECISION_PATH,
        "single_submission": True,
        "source_host": SOURCE_NODE,
        "resources": SOURCE_RESOURCE_CONTRACT,
        "release_only_parent_required": True,
        "allowed_release_diff_paths": list(RELEASE_ONLY_PATHS),
        "automatic_resubmission_allowed": False,
        "automatic_next_experiment_allowed": False,
    }
    for key, wanted in expected.items():
        if release.get(key) != wanted or type(release.get(key)) is not type(wanted):
            raise ValueError(f"constrained-flow execution release {key} changed")
    if not _is_commit(release.get("accepted_implementation_commit")):
        raise ValueError("constrained-flow accepted implementation commit is invalid")
    registered = release.get("run_id")
    if (
        not isinstance(registered, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", registered) is None
        or len(registered) > 128
        or registered != run_id
    ):
        raise ValueError("constrained-flow run id is not the released identity")
    return dict(release)


def _validate_release_commit(
    repository: Path,
    *,
    observed_commit: str,
    execution_release: Mapping[str, Any],
) -> None:
    accepted = execution_release["accepted_implementation_commit"]
    if _git_output(repository, "rev-parse", "HEAD") != observed_commit:
        raise ValueError("constrained-flow release HEAD changed")
    if _git_output(repository, "rev-parse", RELEASE_BRANCH_REF) != observed_commit:
        raise ValueError("constrained-flow origin release ref changed")
    parents = _git_output(
        repository, "rev-list", "--parents", "-n", "1", observed_commit
    ).split()
    if parents != [observed_commit, accepted]:
        raise ValueError("constrained-flow release is not the direct implementation child")
    changed = frozenset(
        line
        for line in _git_output(
            repository, "diff", "--name-only", accepted, observed_commit
        ).splitlines()
        if line
    )
    if changed != frozenset(RELEASE_ONLY_PATHS):
        raise ValueError("constrained-flow release commit changed non-release files")


def _validate_apparatus_bindings(
    repository: Path,
    apparatus: Mapping[str, Any],
) -> tuple[str, str]:
    if apparatus.get("schema_version") != "1.0" or apparatus.get(
        "experiment_identity"
    ) != "IFT-00B/CFS-00A":
        raise ValueError("constrained-flow apparatus identity changed")
    schema_path = repository / ENVELOPE_SCHEMA_PATH
    schema_sha = file_sha256(schema_path)
    if apparatus.get("envelope_schema") != {
        "path": ENVELOPE_SCHEMA_PATH,
        "sha256": schema_sha,
    }:
        raise ValueError("constrained-flow envelope schema binding changed")
    for field, relative in (
        ("scientific_config", SCIENTIFIC_CONFIG_PATH),
        ("legacy_scientific_config", LEGACY_CONFIG_PATH),
        ("semantic_validator", SEMANTIC_VALIDATOR_PATH),
    ):
        if apparatus.get(field) != {
            "path": relative,
            "sha256": file_sha256(repository / relative),
        }:
            raise ValueError(f"constrained-flow apparatus {field} binding changed")
    evidence_binding = apparatus.get("runtime_identity_evidence_binding")
    if evidence_binding != RUNTIME_IDENTITY_EVIDENCE_BINDING:
        raise ValueError("constrained-flow runtime identity evidence binding changed")
    for field in ("decision", "evidence", "preflight"):
        relative = evidence_binding[f"{field}_path"]
        if file_sha256(repository / relative) != evidence_binding[f"{field}_sha256"]:
            raise ValueError(f"constrained-flow runtime identity {field} bytes changed")
    runtime_evidence = _load_object(
        repository / RUNTIME_IDENTITY_EVIDENCE_PATH,
        label="runtime identity terminal evidence",
    )
    if any(
        (
            runtime_evidence.get("release_commit")
            != RUNTIME_IDENTITY_EVIDENCE_BINDING["release_commit"],
            runtime_evidence.get("source_host") != SOURCE_NODE,
            runtime_evidence.get("job", {}).get("job_id") != "28043",
            runtime_evidence.get("job", {}).get("state") != "COMPLETED",
            runtime_evidence.get("job", {}).get("exit_code") != "0:0",
            runtime_evidence.get("job", {}).get("allocated_gpus") != 0,
            runtime_evidence.get("execution", {}).get("shell_only") is not True,
            runtime_evidence.get("execution", {}).get("openpi_python_executed")
            is not False,
            runtime_evidence.get("execution", {}).get("libero_python_executed")
            is not False,
            runtime_evidence.get("interpretation", {}).get(
                "cfs_runtime_integration_evaluated"
            )
            is not False,
            runtime_evidence.get("interpretation", {}).get(
                "h100_submission_authorized_by_this_result"
            )
            is not False,
            runtime_evidence.get("immutable_artifacts", {})
            .get("result", {})
            .get("sha256")
            != RUNTIME_IDENTITY_EVIDENCE_BINDING["result_sha256"],
        )
    ):
        raise ValueError("constrained-flow runtime identity terminal evidence changed")
    scientific = _load_object(
        repository / SCIENTIFIC_CONFIG_PATH,
        label="constrained-flow scientific config",
    )
    if any(
        (
            scientific.get("config_status") != "released_exact_single_canary",
            scientific.get("ready_to_run") is not True,
            scientific.get("blocked_on") != [],
            scientific.get("execution_release") != apparatus.get("execution_release"),
            not isinstance(scientific.get("preregistration"), Mapping),
            scientific.get("preregistration", {}).get(
                "h100_submission_authorized"
            )
            is not True,
        )
    ):
        raise ValueError("constrained-flow scientific release bookkeeping changed")
    registry_sha = file_sha256(repository / ALLOCATION_TEST_REGISTRY_PATH)
    allocation = apparatus.get("allocation_test_contract")
    counts = _load_allocation_test_counts(repository / ALLOCATION_TEST_REGISTRY_PATH)
    if not isinstance(allocation, Mapping) or any(
        (
            allocation.get("registry_path") != ALLOCATION_TEST_REGISTRY_PATH,
            allocation.get("registry_sha256") != registry_sha,
            allocation.get("expected_counts") != counts,
            allocation.get("zero_skips_required") is not True,
            allocation.get("cpu_validator_rehashes_and_reparses") is not True,
        )
    ):
        raise ValueError("constrained-flow allocation-test binding changed")
    return schema_sha, registry_sha


def _load_allocation_test_counts(path: str | Path) -> dict[str, int]:
    value = _load_object(path, label="constrained-flow allocation-test registry")
    if set(value) != {"schema_version", "suites"} or value.get("schema_version") != "1.0":
        raise ValueError("constrained-flow allocation-test registry shape changed")
    suites = value.get("suites")
    if not isinstance(suites, list) or not suites:
        raise ValueError("constrained-flow allocation-test registry is empty")
    counts: dict[str, int] = {}
    for record in suites:
        if not isinstance(record, Mapping) or set(record) != {
            "pattern",
            "expected_tests",
        }:
            raise ValueError("constrained-flow allocation-test suite shape changed")
        name = record.get("pattern")
        count = record.get("expected_tests")
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or name in counts
            or type(count) is not int
            or count <= 0
        ):
            raise ValueError("constrained-flow allocation-test suite is invalid")
        counts[name] = count
    return counts


def _parse_allocation_test_log(
    path: str | Path, *, expected_counts: Mapping[str, int]
) -> tuple[str, dict[str, int]]:
    test_path = Path(path)
    digest = file_sha256(test_path)
    lines = test_path.read_text(encoding="utf-8").splitlines()
    observed: dict[str, int] = {}
    for suite, expected in expected_counts.items():
        marker = (
            f"verified_test_suite={suite} expected={expected} "
            f"observed={expected} skips=0 status=passed"
        )
        if [line for line in lines if line.startswith(f"verified_test_suite={suite} ")] != [marker]:
            raise ValueError(f"focused-test marker changed for {suite}")
        if sum(line.startswith(f"[{suite}] Ran {expected} test") for line in lines) != 1:
            raise ValueError(f"focused-test unittest count changed for {suite}")
        if lines.count(f"[{suite}] OK") != 1:
            raise ValueError(f"focused-test status changed for {suite}")
        if any(
            "skipped=" in line
            or line.startswith(f"[{suite}] FAILED")
            or line.startswith(f"[{suite}] ERROR")
            for line in lines
            if line.startswith(f"[{suite}] ")
        ):
            raise ValueError(f"focused-test suite skipped or failed: {suite}")
        observed[suite] = expected
    if sum(line.startswith("verified_test_suite=") for line in lines) != len(
        expected_counts
    ):
        raise ValueError("focused-test log contains an unregistered suite marker")
    return digest, observed


def _expected_frozen_bindings(repository: Path) -> dict[str, str]:
    """Return the independently rehashed source identities in the receipt."""

    return {
        "constrained_flow_config_sha256": file_sha256(
            repository / SCIENTIFIC_CONFIG_PATH
        ),
        "legacy_scientific_config_sha256": file_sha256(
            repository / LEGACY_CONFIG_PATH
        ),
        "manifest_sha256": file_sha256(
            repository / "manifests/r05a_inverse_flow_teacher_smoke.jsonl"
        ),
        "source_r02_sha256": "055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593",
        "checkpoint_sha256": "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",
        "envelope_schema_sha256": file_sha256(repository / ENVELOPE_SCHEMA_PATH),
        "adr0040_sha256": file_sha256(
            repository
            / "docs/decisions/0040-preregister-same-budget-constrained-flow-diagnostic.md"
        ),
        "historical_adr0041_sha256": file_sha256(
            repository / HISTORICAL_RELEASE_DECISION_PATH
        ),
        "adr0045_sha256": file_sha256(repository / RUNTIME_IDENTITY_DECISION_PATH),
        "adr0046_sha256": file_sha256(repository / RELEASE_DECISION_PATH),
        "runtime_identity_evidence_sha256": file_sha256(
            repository / RUNTIME_IDENTITY_EVIDENCE_PATH
        ),
        "runtime_identity_preflight_sha256": file_sha256(
            repository / RUNTIME_IDENTITY_PREFLIGHT_PATH
        ),
        "runtime_identity_release_commit": RUNTIME_IDENTITY_EVIDENCE_BINDING[
            "release_commit"
        ],
        "runtime_identity_job_id": RUNTIME_IDENTITY_EVIDENCE_BINDING["job_id"],
        "runtime_identity_result_sha256": RUNTIME_IDENTITY_EVIDENCE_BINDING[
            "result_sha256"
        ],
        "vinuni_h100_guide_sha256": VINUNI_H100_GUIDE_CONTRACT["sha256"],
        "normalization_asset_sha256": "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
        "baseline_revision": "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b",
        "historical_inverse_control_sha256": file_sha256(
            repository / "openpi/src/openpi/models_pytorch/crfs_inverse_control.py"
        ),
        "pi05_sampler_sha256": file_sha256(
            repository / "openpi/src/openpi/models_pytorch/pi0_pytorch.py"
        ),
        "policy_boundary_sha256": file_sha256(
            repository / "openpi/src/openpi/policies/policy.py"
        ),
        "ordinary_policy_server_sha256": file_sha256(
            repository / "openpi/scripts/serve_policy.py"
        ),
        "transformers_source_bundle_sha256": "430b00a688e12ff457cdd65929bd164fd001ffa1716dc83589d5388806d2bb33",
        "transformers_replacement_bundle_sha256": "2e1b546bdf42e9872c84734b2d5baf52bd411664685922458e734732c8434098",
        "transformers_overlay_bundle_sha256": "24be8ac6749a4cf7e19c261b14b39e951a499ec61b0602d0badcc4354171d261",
        "openpi_python_public_path": RUNTIME_IDENTITY_CONTRACT["openpi_python"]["public_path"],
        "openpi_python_direct_link_target": RUNTIME_IDENTITY_CONTRACT["openpi_python"]["direct_link_target"],
        "openpi_python_resolved_executable": RUNTIME_IDENTITY_CONTRACT["openpi_python"]["resolved_executable"],
        "openpi_python_resolved_sha256": RUNTIME_IDENTITY_CONTRACT["openpi_python"]["resolved_sha256"],
        "libero_python_public_path": RUNTIME_IDENTITY_CONTRACT["libero_python"]["public_path"],
        "libero_python_direct_link_target": RUNTIME_IDENTITY_CONTRACT["libero_python"]["direct_link_target"],
        "libero_python_resolved_executable": RUNTIME_IDENTITY_CONTRACT["libero_python"]["resolved_executable"],
        "libero_python_resolved_sha256": RUNTIME_IDENTITY_CONTRACT["libero_python"]["resolved_sha256"],
    }


def _validate_source_contract(
    contract_path: Path,
    *,
    expected_sha256: str,
    repository: Path,
    source_job_id: str,
    expected_git_commit: str,
) -> tuple[dict[str, Any], Path, Path]:
    if not _is_sha256(expected_sha256):
        raise ValueError("external source-contract SHA-256 is invalid")
    root = EXPERIMENT_ROOT.absolute()
    if (
        not contract_path.is_absolute()
        or contract_path.name != "source-contract.json"
        or contract_path.parent.parent != root
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", contract_path.parent.name)
        is None
    ):
        raise ValueError("constrained-flow source contract escaped the experiment root")
    run_root = root / contract_path.parent.name
    _require_exact_child_directory(run_root, root, label="run root")
    _resolve_exact(contract_path, run_root / "source-contract.json", label="source contract")
    contract, observed_sha = _load_hashed_object(contract_path, label="source contract")
    if observed_sha != expected_sha256:
        raise ValueError("source contract differs from externally supplied SHA-256")
    if set(contract) != SOURCE_CONTRACT_KEYS:
        raise ValueError("constrained-flow source contract keys changed")
    identity = {
        "schema_version": "1.0",
        "artifact_role": "r05a_constrained_flow_canary_source_contract",
        "status": "gpu_held_sources_bound_before_cpu_submission",
        "git_commit": expected_git_commit,
        "git_dirty": False,
        "source_node": SOURCE_NODE,
        "gpu_slurm_array_job_id": source_job_id,
        "gpu_slurm_array_task_id": 0,
        "exact_gpu_task_id": f"{source_job_id}_0",
        "resources": SOURCE_RESOURCE_CONTRACT,
        "scientific_claim_allowed": False,
        "infeasibility_claim_allowed": False,
        "simulator_efficacy_claim_allowed": False,
        "probe_training_authorized": False,
    }
    for key, wanted in identity.items():
        if contract.get(key) != wanted or type(contract.get(key)) is not type(wanted):
            raise ValueError(f"constrained-flow source contract {key} changed")
    run_id = contract.get("run_id")
    if run_id != run_root.name:
        raise ValueError("constrained-flow source contract run id changed")
    case_dir = run_root / CASE_ID
    _require_exact_child_directory(case_dir, run_root, label="case directory")
    expected_paths = {
        "run_root": str(run_root),
        "case_dir": str(case_dir),
        "legacy_payload": str(case_dir / "canary-payload.json"),
        "constrained_flow_payload": str(case_dir / "constrained-flow-payload.json"),
        "host_telemetry": str(case_dir / "host-cgroup-sampled-current.tsv"),
        "gpu_samples": str(case_dir / "gpu-memory-samples.csv"),
        "allocation_tests_log": str(case_dir / "allocation-focused-tests.log"),
        "hidden_candidate": str(case_dir / ".results.candidate.json"),
        "result": str(case_dir / "results.json"),
        "validation_receipt": str(run_root / "cpu-afterany-validation.json"),
    }
    if contract.get("artifact_paths") != expected_paths:
        raise ValueError("constrained-flow source artifact paths changed")
    live_preflight = run_root / "vinuni-preflight.txt"
    _resolve_exact(
        contract.get("live_preflight_path", ""),
        live_preflight,
        label="fresh VinUni preflight",
    )
    if file_sha256(live_preflight) != contract.get("live_preflight_sha256"):
        raise ValueError("fresh VinUni preflight binding changed")
    frozen_bindings = _expected_frozen_bindings(repository)
    if contract.get("frozen_bindings") != frozen_bindings:
        raise ValueError("constrained-flow frozen source bindings changed")
    hashes = contract.get("repository_file_sha256")
    if type(hashes) is not dict or set(hashes) != BOUND_REPOSITORY_PATHS:
        raise ValueError("constrained-flow repository binding set changed")
    for relative, recorded in hashes.items():
        if not _is_sha256(recorded):
            raise ValueError(f"invalid repository hash for {relative}")
        candidate = repository / relative
        if candidate.resolve(strict=True) != candidate:
            raise ValueError(f"repository binding escaped: {relative}")
        if file_sha256(candidate) != recorded:
            raise ValueError(f"repository binding changed: {relative}")
    held_path = run_root / "held-gpu-submission.json"
    _resolve_exact(
        contract.get("held_gpu_submission_path", ""),
        held_path,
        label="held GPU receipt",
    )
    held, held_sha = _load_hashed_object(held_path, label="held GPU receipt")
    if held_sha != contract.get("held_gpu_submission_sha256") or set(held) != HELD_GPU_KEYS:
        raise ValueError("held GPU receipt binding changed")
    held_expected = {
        "schema_version": "1.0",
        "artifact_role": "r05a_constrained_flow_canary_held_gpu_submission",
        "status": "sbatch_returned_held_gpu_id",
        "run_id": run_id,
        "git_commit": expected_git_commit,
        "gpu_slurm_array_job_id": source_job_id,
        "gpu_slurm_array_task_id": 0,
        "exact_gpu_task_id": f"{source_job_id}_0",
        "source_node": SOURCE_NODE,
        "resources": SOURCE_RESOURCE_CONTRACT,
        "released_at_receipt_time": False,
        "scientific_claim_allowed": False,
        "infeasibility_claim_allowed": False,
        "simulator_efficacy_claim_allowed": False,
        "probe_training_authorized": False,
    }
    for key, wanted in held_expected.items():
        if held.get(key) != wanted or type(held.get(key)) is not type(wanted):
            raise ValueError(f"held GPU receipt {key} changed")
    return contract, run_root, case_dir


def _validate_submission(
    submission_path: Path,
    *,
    contract: Mapping[str, Any],
    contract_path: Path,
    contract_sha256: str,
    source_job_id: str,
    publisher_job_id: str,
    result_path: Path,
    receipt_path: Path,
) -> str:
    submission, submission_sha = _load_hashed_object(
        submission_path, label="atomic submission receipt"
    )
    if set(submission) != SUBMISSION_KEYS:
        raise ValueError("constrained-flow atomic submission keys changed")
    expected = {
        "schema_version": "1.0",
        "artifact_role": "r05a_constrained_flow_canary_atomic_submission",
        "status": "cpu_afterany_registered_gpu_held",
        "run_id": contract["run_id"],
        "git_commit": contract["git_commit"],
        "source_node": SOURCE_NODE,
        "gpu_slurm_array_job_id": source_job_id,
        "gpu_slurm_array_task_id": 0,
        "cpu_afterany_job_id": publisher_job_id,
        "dependency": f"afterany:{source_job_id}",
        "source_contract_path": str(contract_path),
        "source_contract_sha256": contract_sha256,
        "expected_result": str(result_path),
        "expected_validation_receipt": str(receipt_path),
        "scientific_claim_allowed": False,
        "infeasibility_claim_allowed": False,
        "simulator_efficacy_claim_allowed": False,
        "probe_training_authorized": False,
    }
    for key, wanted in expected.items():
        if submission.get(key) != wanted or type(submission.get(key)) is not type(wanted):
            raise ValueError(f"constrained-flow atomic submission {key} changed")
    return submission_sha


def _validate_legacy_payload(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    source_job_id: str,
    expected_git_commit: str,
) -> tuple[dict[str, Any], list[str], int, int, str]:
    if set(payload) != {
        "schema_version",
        "payload_type",
        "complete_result_requires_memory_finalization",
        "result_without_memory",
    }:
        raise ValueError("legacy raw payload keys changed")
    if payload.get("schema_version") != "1.0" or payload.get("payload_type") != (
        "r05a_inverse_flow_allocation_canary_payload"
    ):
        raise ValueError("legacy raw payload identity changed")
    if payload.get("complete_result_requires_memory_finalization") is not True:
        raise ValueError("legacy raw payload no longer awaits CPU finalization")
    result = payload.get("result_without_memory")
    if type(result) is not dict or "memory" in result:
        raise ValueError("legacy raw result shape changed")
    if result.get("run_id") != run_id or result.get("case_id") != CASE_ID:
        raise ValueError("legacy raw run or case identity changed")
    provenance = result.get("provenance")
    if not isinstance(provenance, Mapping) or not all(
        (
            provenance.get("git_commit") == expected_git_commit,
            provenance.get("git_dirty") is False,
            str(provenance.get("slurm_job_id")) == source_job_id,
            str(provenance.get("slurm_array_job_id")) == source_job_id,
            str(provenance.get("slurm_array_task_id")) == "0",
            str(provenance.get("host", "")).split(".", 1)[0] == SOURCE_NODE,
        )
    ):
        raise ValueError("legacy raw provenance differs from exact source task")
    from . import r05a_canary as legacy

    errors = legacy.validate_r05a_canary_result(copy.deepcopy(result))
    if errors != LEGACY_RAW_DEFERRED_ERRORS:
        raise ValueError(
            "legacy raw semantic validation returned changed errors: " + "; ".join(errors)
        )
    first = legacy._trace_from_record(
        result["solver"]["first"]["trace"], name="legacy solver.first.trace"
    )
    duplicate = legacy._trace_from_record(
        result["solver"]["duplicate"]["trace"], name="legacy solver.duplicate.trace"
    )
    allocated, reserved = legacy._artifact_cuda_memory_peaks(first, duplicate)
    uuid = str(provenance.get("allocation_gpu_uuid", ""))
    if not uuid.startswith("GPU-"):
        raise ValueError("legacy raw allocation GPU UUID is missing")
    return result, errors, allocated, reserved, uuid


def _raw_record(
    *,
    path: Path,
    present: bool,
    sha256: str | None,
    payload_type: str | None,
    payload_variant: str | None,
    status: str | None,
    semantically_validated: bool,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "present": present,
        "sha256": sha256,
        "payload_type": payload_type,
        "payload_variant": payload_variant,
        "status": status,
        "bytes_unmodified_by_wrapper": True,
        "semantically_validated": semantically_validated,
    }


def _load_raw_payload_pair(
    *,
    constrained_path: Path,
    legacy_path: Path,
    case_dir: Path,
) -> tuple[dict[str, Any], str, dict[str, Any] | None, str | None]:
    """Load exact raw files and enforce the variant-dependent legacy contract."""

    _resolve_exact(
        constrained_path,
        case_dir / "constrained-flow-payload.json",
        label="constrained flow payload",
    )
    cfs_payload, cfs_sha = _load_hashed_object(
        constrained_path, label="constrained-flow raw payload"
    )
    variant = cfs_payload.get("payload_variant")
    if variant not in {"complete_comparison", "terminal_apparatus_failure"}:
        raise ValueError("constrained-flow raw payload variant changed")
    if variant == "terminal_apparatus_failure":
        failure = cfs_payload.get("failure")
        if isinstance(failure, Mapping) and OOM_SIGNAL.search(
            "\n".join(
                str(failure.get(key, "")) for key in ("error_type", "message")
            )
        ):
            raise ValueError(
                "terminal constrained-flow payload records an out-of-memory failure"
            )
    if not legacy_path.is_absolute() or legacy_path != case_dir / "canary-payload.json":
        raise ValueError("legacy payload escaped its immutable path")
    legacy_payload: dict[str, Any] | None = None
    legacy_sha: str | None = None
    if variant == "complete_comparison":
        _resolve_exact(legacy_path, case_dir / "canary-payload.json", label="legacy payload")
        legacy_payload, legacy_sha = _load_hashed_object(
            legacy_path, label="legacy raw payload"
        )
        if cfs_payload.get("legacy_payload_sha256") != legacy_sha:
            raise ValueError("CFS payload legacy-byte binding changed")
        return cfs_payload, cfs_sha, legacy_payload, legacy_sha
    legacy_record = cfs_payload.get("legacy_payload")
    if not isinstance(legacy_record, Mapping) or legacy_record.get("path") != str(
        legacy_path
    ):
        raise ValueError("terminal CFS payload legacy-path record changed")
    if legacy_record.get("exists") is True:
        _resolve_exact(legacy_path, case_dir / "canary-payload.json", label="legacy payload")
        legacy_payload, legacy_sha = _load_hashed_object(
            legacy_path, label="legacy raw payload"
        )
        if legacy_record.get("sha256") != legacy_sha:
            raise ValueError("terminal CFS payload legacy hash changed")
    elif not (
        legacy_record.get("exists") is False
        and legacy_record.get("sha256") is None
        and not os.path.lexists(legacy_path)
    ):
        raise ValueError("terminal CFS payload legacy absence is inconsistent")
    return cfs_payload, cfs_sha, legacy_payload, legacy_sha


def build_constrained_flow_envelope(
    *,
    legacy_payload_path: str | Path,
    constrained_flow_payload_path: str | Path,
    host_telemetry_path: str | Path,
    gpu_samples_path: str | Path,
    allocation_tests_log: str | Path,
    source_contract_path: str | Path,
    expected_source_contract_sha256: str,
    submission_path: str | Path,
    source_job_id: str,
    source_job_state: str,
    source_exit_code: str,
    publisher_job_id: str,
    expected_git_commit: str,
    result_path: str | Path,
    receipt_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Rebuild one accepted envelope entirely from immutable raw inputs."""

    if re.fullmatch(r"[1-9][0-9]*", source_job_id) is None:
        raise ValueError("source job id is invalid")
    if re.fullmatch(r"[1-9][0-9]*", publisher_job_id) is None:
        raise ValueError("publisher job id is invalid")
    if not _is_commit(expected_git_commit):
        raise ValueError("expected git commit is invalid")
    if source_job_state != "COMPLETED" or source_exit_code != "0:0":
        raise ValueError("source H100 task was not successful")
    repository = Path(repo_root).resolve(strict=True)
    observed_commit = _git_output(repository, "rev-parse", "HEAD")
    if observed_commit != expected_git_commit or _git_output(
        repository, "status", "--porcelain"
    ):
        raise ValueError("publisher checkout differs from the clean expected commit")
    apparatus, apparatus_sha = _load_hashed_object(
        repository / APPARATUS_CONFIG_PATH, label="constrained-flow apparatus config"
    )
    execution_release = _validate_execution_release(apparatus, run_id=Path(source_contract_path).parent.name)
    _validate_release_commit(
        repository,
        observed_commit=observed_commit,
        execution_release=execution_release,
    )
    schema_sha, registry_sha = _validate_apparatus_bindings(repository, apparatus)

    contract_path = Path(source_contract_path)
    contract, run_root, case_dir = _validate_source_contract(
        contract_path,
        expected_sha256=expected_source_contract_sha256,
        repository=repository,
        source_job_id=source_job_id,
        expected_git_commit=expected_git_commit,
    )
    if contract["repository_file_sha256"].get(APPARATUS_CONFIG_PATH) != apparatus_sha:
        raise ValueError("apparatus config differs from source contract")
    if contract["repository_file_sha256"].get(ENVELOPE_SCHEMA_PATH) != schema_sha:
        raise ValueError("envelope schema differs from source contract")

    expected_files = {
        "constrained flow payload": case_dir / "constrained-flow-payload.json",
        "host telemetry": case_dir / "host-cgroup-sampled-current.tsv",
        "GPU samples": case_dir / "gpu-memory-samples.csv",
        "allocation tests": case_dir / "allocation-focused-tests.log",
        "submission": run_root / "submission.json",
    }
    supplied = {
        "constrained flow payload": constrained_flow_payload_path,
        "host telemetry": host_telemetry_path,
        "GPU samples": gpu_samples_path,
        "allocation tests": allocation_tests_log,
        "submission": submission_path,
    }
    resolved = {
        label: _resolve_exact(supplied[label], expected, label=label)
        for label, expected in expected_files.items()
    }
    legacy_path = Path(legacy_payload_path)
    if not legacy_path.is_absolute() or legacy_path != case_dir / "canary-payload.json":
        raise ValueError("legacy payload escaped its immutable path")
    destination = Path(result_path)
    receipt = Path(receipt_path)
    if destination != case_dir / "results.json" or receipt != run_root / (
        "cpu-afterany-validation.json"
    ):
        raise ValueError("constrained-flow publication paths changed")
    submission_sha = _validate_submission(
        resolved["submission"],
        contract=contract,
        contract_path=contract_path,
        contract_sha256=expected_source_contract_sha256,
        source_job_id=source_job_id,
        publisher_job_id=publisher_job_id,
        result_path=destination,
        receipt_path=receipt,
    )

    host_summary = parse_full_lifetime_telemetry(
        resolved["host telemetry"], expected_job_id=source_job_id
    )
    if file_sha256(resolved["host telemetry"]) != host_summary["raw_trace_sha256"]:
        raise ValueError("host telemetry changed while parsed")
    from . import r05a_canary as legacy

    gpu_count, gpu_uuid, compute_high, device_high, gpu_sha = legacy._read_gpu_samples(
        resolved["GPU samples"]
    )
    if file_sha256(resolved["GPU samples"]) != gpu_sha:
        raise ValueError("GPU samples changed while parsed")
    counts = _load_allocation_test_counts(repository / ALLOCATION_TEST_REGISTRY_PATH)
    tests_sha, observed_counts = _parse_allocation_test_log(
        resolved["allocation tests"], expected_counts=counts
    )
    if file_sha256(resolved["allocation tests"]) != tests_sha:
        raise ValueError("allocation-test log changed while parsed")

    cfs_payload, cfs_sha, legacy_payload, legacy_sha = _load_raw_payload_pair(
        constrained_path=resolved["constrained flow payload"],
        legacy_path=legacy_path,
        case_dir=case_dir,
    )
    variant = cfs_payload.get("payload_variant")
    legacy_result: dict[str, Any] | None = None
    legacy_errors: list[str] = []
    process_allocated: int | None = None
    process_reserved: int | None = None
    legacy_uuid: str | None = None
    if legacy_payload is not None:
        legacy_result, legacy_errors, process_allocated, process_reserved, legacy_uuid = (
            _validate_legacy_payload(
                legacy_payload,
                run_id=contract["run_id"],
                source_job_id=source_job_id,
                expected_git_commit=expected_git_commit,
            )
        )
        if legacy_uuid != gpu_uuid:
            raise ValueError("GPU raw samples differ from legacy allocation UUID")
    constrained_config = _load_object(
        repository / SCIENTIFIC_CONFIG_PATH,
        label="constrained-flow scientific config",
    )
    from .r05a_constrained_flow_validation import (
        validate_constrained_flow_payload_or_raise,
    )

    recomputed_status = validate_constrained_flow_payload_or_raise(
        cfs_payload,
        constrained_config,
        legacy_payload=legacy_payload,
        expected_run_id=contract["run_id"],
        constrained_config_path=repository / SCIENTIFIC_CONFIG_PATH,
        legacy_config_path=repository / LEGACY_CONFIG_PATH,
    )
    if cfs_payload.get("status") != recomputed_status:
        raise ValueError("CFS raw status differs from independent reconstruction")
    if variant == "terminal_apparatus_failure":
        provenance = cfs_payload.get("provenance")
        if not isinstance(provenance, Mapping) or not all(
            (
                provenance.get("git_commit") == expected_git_commit,
                str(provenance.get("source_node", "")).split(".", 1)[0] == SOURCE_NODE,
                str(provenance.get("slurm_job_id")) == source_job_id,
                str(provenance.get("slurm_array_job_id")) == source_job_id,
                str(provenance.get("slurm_array_task_id")) == "0",
            )
        ):
            raise ValueError("terminal CFS payload provenance changed")
    simulator_use = {
        "setup_only": True,
        "policy_generated_action_steps_executed": 0,
        "teacher_generated_action_steps_executed": 0,
        "efficacy_rollouts_executed": 0,
        "simulator_efficacy_evaluated": False,
    }
    hidden_candidate = case_dir / ".results.candidate.json"
    legacy_present = legacy_payload is not None
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "gate": "CFS-00A",
        "experiment_identity": "IFT-00B/CFS-00A",
        "case_id": CASE_ID,
        "run_id": contract["run_id"],
        "status": recomputed_status,
        "evidence_tier": "real_pi05_allocation_transport_mechanism_canary_only",
        "apparatus_config_sha256": apparatus_sha,
        "envelope_schema_sha256": schema_sha,
        "source_job": {
            "source_commit": expected_git_commit,
            "source_tree_dirty": False,
            "array_job_id": source_job_id,
            "array_task_id": 0,
            "exact_task_id": f"{source_job_id}_0",
            "state": source_job_state,
            "exit_code": source_exit_code,
            "node": SOURCE_NODE,
            "partition": "main",
            "account": "normal",
            "qos": "normal",
            "requested_gpus": 1,
            "requested_cpus_per_task": 8,
            "requested_host_memory_mib": 65536,
            "time_limit": "02:00:00",
            "array": "0-0%1",
            "requeue": False,
        },
        "raw_payloads": {
            "legacy": _raw_record(
                path=legacy_path,
                present=legacy_present,
                sha256=legacy_sha,
                payload_type=(legacy_payload or {}).get("payload_type"),
                payload_variant=(
                    "historical_inverse_flow_payload" if legacy_present else None
                ),
                status=(legacy_result or {}).get("status"),
                semantically_validated=legacy_present,
            ),
            "constrained_flow": _raw_record(
                path=resolved["constrained flow payload"],
                present=True,
                sha256=cfs_sha,
                payload_type=cfs_payload.get("payload_type"),
                payload_variant=variant,
                status=cfs_payload.get("status"),
                semantically_validated=True,
            ),
            "every_present_payload_rehashed": True,
            "gpu_wrote_raw_only": True,
        },
        "allocation_tests": {
            "log_path": str(resolved["allocation tests"]),
            "log_sha256": tests_sha,
            "registry_path": ALLOCATION_TEST_REGISTRY_PATH,
            "registry_sha256": registry_sha,
            "expected_counts": counts,
            "observed_counts": observed_counts,
            "zero_skips": True,
            "exit_code": 0,
            "rehashed_and_reparsed_by_cpu_validator": True,
        },
        "semantic_validation": {
            "legacy_validator": "crfs_oracle.r05a_canary.validate_r05a_canary_result",
            "legacy_validator_module_sha256": file_sha256(
                repository / "main/crfs_oracle/r05a_canary.py"
            ),
            "legacy_validation_performed": legacy_present,
            "legacy_returned_errors": legacy_errors,
            "legacy_returned_errors_exact_allowlist": bool(
                legacy_present and legacy_errors == LEGACY_RAW_DEFERRED_ERRORS
            ),
            "constrained_flow_validator": "crfs_oracle.r05a_constrained_flow_validation.validate_constrained_flow_payload_or_raise",
            "constrained_flow_validator_module_sha256": file_sha256(
                repository / SEMANTIC_VALIDATOR_PATH
            ),
            "constrained_flow_validation_performed": True,
            "constrained_flow_returned_errors": [],
            "recomputed_status": recomputed_status,
            "stored_gpu_pass_booleans_trusted": False,
        },
        "telemetry": {
            "host": _host_envelope(host_summary),
            "gpu": {
                "allocation_gpu_uuid": gpu_uuid,
                "raw_samples_path": str(resolved["GPU samples"]),
                "raw_samples_sha256": gpu_sha,
                "sample_count": gpu_count,
                "process_peak_available": legacy_present,
                "process_peak_owner": (
                    "historical_arm_a_trace_only"
                    if legacy_present
                    else "unavailable_no_legacy_trace"
                ),
                "process_peak_allocated_bytes": process_allocated,
                "process_peak_reserved_bytes": process_reserved,
                "sampled_compute_high_water_mib": compute_high,
                "sampled_device_high_water_mib": device_high,
                "sampled_device_high_water_scope": (
                    "full_cfs_policy_lifecycle_periodic_lower_bound"
                ),
                "sampled_values_are_periodic_not_continuous_peaks": True,
                "out_of_memory_observed": False,
            },
            "all_raw_files_rehashed_by_cpu_validator": True,
            "telemetry_contract_passed": True,
        },
        "simulator_use": simulator_use,
        "publication": {
            "source_contract_receipt_path": str(contract_path),
            "source_contract_receipt_sha256": expected_source_contract_sha256,
            "externally_supplied_expected_receipt_sha256": expected_source_contract_sha256,
            "atomic_submission_receipt_path": str(resolved["submission"]),
            "atomic_submission_receipt_sha256": submission_sha,
            "hidden_candidate_path": str(hidden_candidate),
            "published_result_path": str(destination),
            "post_publication_validation_receipt_path": str(receipt),
            "cpu_validator_job_id": publisher_job_id,
            "dependency": f"afterany:{source_job_id}",
            "cpu_validator_gpu_count": 0,
            "source_job_exit_code": source_exit_code,
            "gpu_job_created_result_candidate": False,
            "gpu_job_published_results_json": False,
            "candidate_created_by_cpu_validator": True,
            "candidate_validated_before_publication": True,
            "candidate_revalidated_before_publication": True,
            "cpu_validator_is_sole_publisher": True,
            "raw_files_unchanged_after_candidate": True,
            "atomic_candidate_to_result_rename": True,
            "validation_passed": True,
        },
        "interpretation": {
            "classification": recomputed_status,
            "one_case_transport_mechanism_claim_allowed": recomputed_status
            == "mechanism_pass",
            "simulator_efficacy_claim_allowed": False,
            "collision_or_progress_claim_allowed": False,
            "infeasibility_claim_allowed": False,
            "student_learning_claim_allowed": False,
            "probe_training_authorized": False,
            "automatic_next_gate_allowed": False,
        },
    }
    errors = _validate_schema(envelope, repository / ENVELOPE_SCHEMA_PATH)
    if errors:
        raise ValueError("constrained-flow envelope schema failed: " + "; ".join(errors))
    return envelope


def publish_constrained_flow_envelope(**kwargs: Any) -> tuple[Path, Path]:
    """Build twice, bind file identities, and atomically publish one envelope."""

    destination = Path(kwargs["result_path"])
    receipt = Path(kwargs["receipt_path"])
    candidate = destination.parent / ".results.candidate.json"
    if destination.exists() or candidate.exists() or receipt.exists():
        raise ValueError("publication paths must not pre-exist")
    envelope = build_constrained_flow_envelope(**kwargs)
    binding = _publication_directory_binding(destination, receipt, candidate)
    _atomic_write_json(candidate, envelope)
    try:
        _require_unchanged_publication_directories(
            destination, receipt, candidate, binding
        )
        candidate_value, candidate_sha = _load_hashed_object(
            candidate, label="hidden result candidate"
        )
        rebuilt = build_constrained_flow_envelope(**kwargs)
        if candidate_value != rebuilt:
            raise ValueError("candidate differs after raw-file rehash and rebuild")
        errors = _validate_schema(
            candidate_value, Path(kwargs["repo_root"]) / ENVELOPE_SCHEMA_PATH
        )
        if errors:
            raise ValueError("candidate revalidation failed: " + "; ".join(errors))
        fingerprints = [
            (
                candidate_value["raw_payloads"]["constrained_flow"]["path"],
                candidate_value["raw_payloads"]["constrained_flow"]["sha256"],
                "constrained-flow payload",
            ),
            (
                candidate_value["allocation_tests"]["log_path"],
                candidate_value["allocation_tests"]["log_sha256"],
                "allocation-test log",
            ),
            (
                candidate_value["telemetry"]["host"]["raw_trace_path"],
                candidate_value["telemetry"]["host"]["raw_trace_sha256"],
                "host telemetry",
            ),
            (
                candidate_value["telemetry"]["gpu"]["raw_samples_path"],
                candidate_value["telemetry"]["gpu"]["raw_samples_sha256"],
                "GPU samples",
            ),
            (
                candidate_value["publication"]["source_contract_receipt_path"],
                candidate_value["publication"]["source_contract_receipt_sha256"],
                "source contract",
            ),
            (
                candidate_value["publication"]["atomic_submission_receipt_path"],
                candidate_value["publication"]["atomic_submission_receipt_sha256"],
                "submission receipt",
            ),
        ]
        legacy_record = candidate_value["raw_payloads"]["legacy"]
        if legacy_record["present"]:
            fingerprints.append(
                (legacy_record["path"], legacy_record["sha256"], "legacy payload")
            )
        for source, expected, label in fingerprints:
            if file_sha256(source) != expected:
                raise ValueError(f"{label} changed after candidate creation")
        identity_before = _regular_file_identity(candidate, label="hidden result candidate")
        value_before, sha_before = _load_hashed_object(
            candidate, label="hidden result candidate"
        )
        identity_after = _regular_file_identity(candidate, label="hidden result candidate")
        if identity_before != identity_after or value_before != candidate_value or sha_before != candidate_sha:
            raise ValueError("hidden result candidate changed during final validation")
        _require_unchanged_publication_directories(
            destination, receipt, candidate, binding
        )
    except Exception:
        candidate.unlink(missing_ok=True)
        raise
    try:
        os.replace(candidate, destination)
        _require_unchanged_publication_directories(
            destination, receipt, candidate, binding
        )
        if _regular_file_identity(destination, label="published result") != identity_after:
            raise ValueError("published result identity differs from candidate")
        published, published_sha = _load_hashed_object(
            destination, label="published result"
        )
        if published != candidate_value or published_sha != candidate_sha:
            raise ValueError("published result bytes differ from candidate")
        errors = _validate_schema(
            published, Path(kwargs["repo_root"]) / ENVELOPE_SCHEMA_PATH
        )
        if errors:
            raise ValueError("published result schema failed: " + "; ".join(errors))
        publication_receipt = {
            "schema_version": SCHEMA_VERSION,
            "artifact_role": "r05a_constrained_flow_canary_cpu_publication",
            "source_job_id": kwargs["source_job_id"],
            "publisher_job_id": kwargs["publisher_job_id"],
            "source_contract_path": str(kwargs["source_contract_path"]),
            "source_contract_sha256_external": kwargs[
                "expected_source_contract_sha256"
            ],
            "submission_path": candidate_value["publication"][
                "atomic_submission_receipt_path"
            ],
            "submission_sha256": candidate_value["publication"][
                "atomic_submission_receipt_sha256"
            ],
            "result_path": str(destination),
            "result_sha256": candidate_sha,
            "passed": True,
            "published": True,
            "errors": [],
            "one_case_transport_mechanism_claim_allowed": candidate_value[
                "interpretation"
            ]["one_case_transport_mechanism_claim_allowed"],
            "simulator_efficacy_claim_allowed": False,
            "probe_training_authorized": False,
        }
        _atomic_write_json(receipt, publication_receipt)
        _require_unchanged_publication_directories(
            destination, receipt, candidate, binding
        )
        final_value, final_sha = _load_hashed_object(
            destination, label="published result"
        )
        if (
            _regular_file_identity(destination, label="published result")
            != identity_after
            or final_value != candidate_value
            or final_sha != candidate_sha
        ):
            raise ValueError("published result changed before receipt finalization")
    except Exception:
        destination.unlink(missing_ok=True)
        receipt.unlink(missing_ok=True)
        raise
    return destination, receipt


__all__ = [
    "APPARATUS_CONFIG_PATH",
    "APPARATUS_RESOURCE_CONTRACT",
    "BOUND_REPOSITORY_PATHS",
    "CASE_ID",
    "ENVELOPE_SCHEMA_PATH",
    "EXECUTION_RELEASE_KEYS",
    "LEGACY_RAW_DEFERRED_ERRORS",
    "RELEASE_DECISION_PATH",
    "RELEASE_ONLY_PATHS",
    "RUNTIME_IDENTITY_CONTRACT",
    "SOURCE_RESOURCE_CONTRACT",
    "build_constrained_flow_envelope",
    "publish_constrained_flow_envelope",
]
