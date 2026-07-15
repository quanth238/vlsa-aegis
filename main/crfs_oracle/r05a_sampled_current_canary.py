"""CPU-only publication for the R05A sampled-current canary envelope.

The H100 source allocation owns only raw artifacts.  This module runs in the
independent CPU ``afterany`` allocation, rehashes and reparses every raw input,
constructs a v2 envelope around the byte-unchanged scientific payload, validates
the hidden candidate twice, and only then permits an atomic final publication.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Any, Mapping

from .r05a_full_lifetime_telemetry import parse_full_lifetime_telemetry


SCHEMA_VERSION = "2.0"
ARTIFACT_TYPE = "r05a_sampled_current_canary_envelope"
CASE_ID = "crfs-1069f29a8d76463a"
SOURCE_NODE = "worker-1"
EXPERIMENT_ROOT = Path("/mnt/data/quanth/experiments/crfs-oracle")
APPARATUS_CONFIG_PATH = "configs/experiments/r05a_sampled_current_canary_apparatus.json"
ENVELOPE_SCHEMA_PATH = "schemas/r05a-sampled-current-canary-envelope.schema.json"
ADR0036_PATH = "docs/decisions/0036-preregister-r05a-full-lifetime-sampled-current-canary.md"
RELEASE_DECISION_PATH = (
    "docs/decisions/0037-require-exact-single-canary-release-identity.md"
)
RELEASE_BRANCH_REF = "refs/remotes/origin/agent/crfs-oracle-harness"
RELEASE_ONLY_PATHS = (
    APPARATUS_CONFIG_PATH,
    RELEASE_DECISION_PATH,
)

FROZEN_BINDINGS: dict[str, str] = {
    "scientific_config_file_sha256": "c31401867f3cdce2b3f443ad021c39dfb812f573b570e1e7434e1f149f79abfb",
    "scientific_config_projection_sha256": "9e2ff74cda8ac3b5d4098942a82cc3352cebea3f4b8e1fc1d326c2dc24d2887d",
    "manifest_sha256": "bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633",
    "adr0028_sha256": "d2a00b1b049e92bb1ec8f11d60fa447e5e8bd5109cb8cf3f3fbb08f89498656f",
    "historical_scientific_schema_sha256": "e1681f865f81f2986945d10fb14073fe4cd9e78cbc5026f6749ab321b9cfb7a7",
    "source_r02_sha256": "055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593",
    "source_r02_config_sha256": "c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e",
    "source_r03_summary_sha256": "dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e",
    "source_r03_ordered_result_set_digest": "fa79a132f2fecbedab5917111ef71f022e52c933d8e535aaca118add5f5b7895",
    "checkpoint_sha256": "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",
    "normalization_asset_sha256": "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
    "baseline_revision": "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b",
    "r05a_canary_module_sha256": "9d4402ccb92835bc1af2a3e04fe54c0b94839f41beb2971022216f73ec67fede",
    "r05a_live_entrypoint_sha256": "247bd20e48ffe2228d2633d2f667a2791371163cca1c43ad7d278f4fbfd7439a",
    "inverse_control_implementation_sha256": "965082822466774e0a86eeb6f2d178c9e5f3d6d02bca5090c4e1fcc77bc52aa8",
    "pi05_sampler_sha256": "80366dcc7b2ddc598717d4c71c0e68e46312a1e5ffd3fc04479d5599433f4c55",
    "policy_boundary_sha256": "d16767ff2073d5c177cdfcc06dc05dcbf7cdb9ef2a2a0150023f7953b03508b9",
}

BOUND_REPOSITORY_PATHS = frozenset(
    {
        "configs/experiments/r05a_inverse_flow_canary.json",
        APPARATUS_CONFIG_PATH,
        "manifests/r05a_inverse_flow_teacher_smoke.jsonl",
        "docs/decisions/0028-pivot-to-inverse-flow-transport.md",
        ADR0036_PATH,
        RELEASE_DECISION_PATH,
        "schemas/r05a-inverse-flow-canary.schema.json",
        ENVELOPE_SCHEMA_PATH,
        "evidence/r03/r03-summary.json",
        "main/crfs_oracle/r05a_canary.py",
        "main/crfs_oracle/r05a_allocation_tests.json",
        "main/crfs_oracle/r05a_full_lifetime_telemetry.py",
        "main/crfs_oracle/r05a_sampled_current_canary.py",
        "main/run_crfs_r05a_canary.py",
        "main/publish_crfs_r05a_sampled_current_canary.py",
        "openpi/src/openpi/models_pytorch/crfs_inverse_control.py",
        "openpi/src/openpi/models_pytorch/pi0_pytorch.py",
        "openpi/src/openpi/policies/policy.py",
        "scripts/hpc/lib/cgroup_v2_full_lifetime_monitor.sh",
        "scripts/hpc/lib/r05a_allocation_tests.sh",
        "scripts/hpc/run_r05a_canary.sh",
        "scripts/hpc/run_r05a_sampled_current_canary.sh",
        "scripts/hpc/validate_r05a_sampled_current_canary.sh",
        "scripts/hpc/submit_r05a_sampled_current_canary.sh",
        "slurm/r05a_sampled_current_canary_h100.sbatch",
        "slurm/r05a_sampled_current_canary_validate_cpu.sbatch",
    }
)

LEGACY_MEMORY_ONLY_ERRORS = [
    "host cgroup path provenance is missing",
    "memory record is missing",
]
PRE_MEMORY_APPARATUS_KEYS = {
    "passed_before_memory_finalization",
    "teacher_invariants_passed",
    "determinism_passed",
    "canonical_replay_passed",
    "zero_and_frozen_passed",
    "compiled_eager_path_seam_passed",
    "pairing_passed",
    "checkpoint_scale_passed",
    "clean_nonconvergence",
    "memory_pending",
}
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
    "scientific_claim_allowed",
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
    "probe_training_authorized",
    "timestamp_utc",
}
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
    "probe_training_authorized",
    "timestamp_utc",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: str | Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} cannot be read as JSON: {error}") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_hashed_object(
    path: str | Path, *, label: str
) -> tuple[dict[str, Any], str]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} cannot be read as JSON: {error}") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _validate_execution_release(
    apparatus_config: Mapping[str, Any], *, run_id: str
) -> dict[str, Any]:
    """Validate one exact, preregistered execution identity without imports."""

    if apparatus_config.get("ready_to_run") is not True or apparatus_config.get(
        "blocked_on"
    ) != []:
        raise ValueError("sampled-current apparatus config is not released")
    release = apparatus_config.get("execution_release")
    if type(release) is not dict or set(release) != EXECUTION_RELEASE_KEYS:
        raise ValueError("sampled-current execution release keys changed")
    expected = {
        "schema_version": "1.0",
        "artifact_role": "r05a_single_canary_execution_release",
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
        observed = release.get(key)
        if observed != wanted or type(observed) is not type(wanted):
            raise ValueError(f"sampled-current execution release {key} changed")
    accepted_commit = release.get("accepted_implementation_commit")
    if not _is_commit(accepted_commit):
        raise ValueError("sampled-current accepted implementation commit is invalid")
    registered_run_id = release.get("run_id")
    if (
        not isinstance(registered_run_id, str)
        or len(registered_run_id) > 128
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", registered_run_id) is None
        or registered_run_id != run_id
    ):
        raise ValueError("sampled-current run id is not the exact released identity")
    expected_apparatus_resources = {"source_host": SOURCE_NODE, **SOURCE_RESOURCE_CONTRACT}
    observed_apparatus_resources = apparatus_config.get("resource_contract")
    if (
        observed_apparatus_resources != expected_apparatus_resources
        or type(observed_apparatus_resources) is not dict
    ):
        raise ValueError("sampled-current apparatus resource contract changed")
    return dict(release)


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _release_decision_appendix(execution_release: Mapping[str, Any]) -> str:
    resources = json.dumps(
        execution_release["resources"], sort_keys=True, separators=(",", ":")
    )
    return (
        "\n## Exact execution release\n\n"
        "Execution authorization: one preregistered IFT-00A canary submission only.\n\n"
        f"- Accepted implementation commit: `{execution_release['accepted_implementation_commit']}`.\n"
        f"- Immutable run ID: `{execution_release['run_id']}`.\n"
        f"- Source host: `{execution_release['source_host']}`.\n"
        f"- Resources (canonical JSON): `{resources}`.\n"
        "- Single submission: `true`.\n"
        "- Automatic resubmission: `false`.\n"
        "- Automatic next experiment: `false`.\n"
        "- Simulator efficacy claim authorized: `false`.\n"
        "- Probe or MLP training authorized: `false`.\n\n"
        "This appendix authorizes only the frozen one-case mechanism canary. "
        "It does not authorize IFT-01, solver tuning, a simulator efficacy claim, "
        "label collection, probe training, or MLP training.\n"
    )


def _validate_release_commit(
    repository: Path,
    *,
    observed_commit: str,
    execution_release: Mapping[str, Any],
) -> frozenset[str]:
    """Require a direct, origin-pinned release-only child commit."""

    accepted_implementation_commit = execution_release.get(
        "accepted_implementation_commit"
    )
    if not _is_commit(observed_commit) or not _is_commit(accepted_implementation_commit):
        raise ValueError("sampled-current release commit identity is invalid")
    if _git_output(repository, "rev-parse", "HEAD") != observed_commit:
        raise ValueError("sampled-current release HEAD changed")
    if _git_output(repository, "rev-parse", RELEASE_BRANCH_REF) != observed_commit:
        raise ValueError("sampled-current origin release ref changed")
    parents = _git_output(
        repository, "rev-list", "--parents", "-n", "1", observed_commit
    ).split()
    if parents != [observed_commit, accepted_implementation_commit]:
        raise ValueError("sampled-current release is not the direct implementation child")
    for commit in (accepted_implementation_commit, observed_commit):
        tree_record = _git_output(
            repository, "ls-tree", commit, "--", APPARATUS_CONFIG_PATH
        ).split()
        if len(tree_record) < 3 or tree_record[:2] != ["100644", "blob"]:
            raise ValueError("sampled-current apparatus config git mode changed")
    changed = frozenset(
        line
        for line in _git_output(
            repository,
            "diff",
            "--name-only",
            accepted_implementation_commit,
            observed_commit,
        ).splitlines()
        if line
    )
    required = {APPARATUS_CONFIG_PATH, RELEASE_DECISION_PATH}
    if not changed or not required.issubset(changed) or not changed.issubset(
        RELEASE_ONLY_PATHS
    ):
        raise ValueError("sampled-current release commit changed non-release files")
    try:
        parent_config = json.loads(
            _git_output(
                repository,
                "show",
                f"{accepted_implementation_commit}:{APPARATUS_CONFIG_PATH}",
            )
        )
        release_config = _load_object(
            repository / APPARATUS_CONFIG_PATH,
            label="sampled-current release apparatus config",
        )
    except json.JSONDecodeError as error:
        raise ValueError("sampled-current parent apparatus config is invalid") from error
    if (
        type(parent_config) is not dict
        or parent_config.get("ready_to_run") is not False
        or not isinstance(parent_config.get("blocked_on"), list)
        or not parent_config["blocked_on"]
        or "execution_release" in parent_config
    ):
        raise ValueError("sampled-current implementation parent was not unreleased")
    for value in (parent_config, release_config):
        value.pop("ready_to_run", None)
        value.pop("blocked_on", None)
        value.pop("execution_release", None)
    canonical_parent = json.dumps(parent_config, sort_keys=True, separators=(",", ":"))
    canonical_release = json.dumps(release_config, sort_keys=True, separators=(",", ":"))
    if canonical_release != canonical_parent:
        raise ValueError("sampled-current release changed non-release apparatus content")
    parent_decision = _git_bytes(
        repository,
        "show",
        f"{accepted_implementation_commit}:{RELEASE_DECISION_PATH}",
    )
    try:
        release_decision = (repository / RELEASE_DECISION_PATH).read_bytes()
    except OSError as error:
        raise ValueError("sampled-current release decision cannot be read") from error
    expected_decision = parent_decision + _release_decision_appendix(
        execution_release
    ).encode("utf-8")
    if release_decision != expected_decision:
        raise ValueError("sampled-current release decision is not the exact appendix")
    return changed


def _atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False) as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)
    return destination


def _regular_file_identity(path: str | Path, *, label: str) -> tuple[int, int, int]:
    """Return a stable local identity for one non-symlink regular file."""

    candidate = Path(path)
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise ValueError(f"{label} cannot be inspected: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a non-symlink regular file")
    return metadata.st_dev, metadata.st_ino, metadata.st_size


def _directory_identity(path: str | Path, *, label: str) -> tuple[int, int, str]:
    """Bind one non-symlink directory to its inode and resolved location."""

    candidate = Path(path)
    try:
        metadata_before = candidate.lstat()
        resolved = candidate.resolve(strict=True)
        metadata_after = candidate.lstat()
    except OSError as error:
        raise ValueError(f"{label} cannot be inspected: {error}") from error
    if (
        stat.S_ISLNK(metadata_before.st_mode)
        or not stat.S_ISDIR(metadata_before.st_mode)
        or stat.S_ISLNK(metadata_after.st_mode)
        or not stat.S_ISDIR(metadata_after.st_mode)
    ):
        raise ValueError(f"{label} must be a non-symlink directory")
    before = metadata_before.st_dev, metadata_before.st_ino
    after = metadata_after.st_dev, metadata_after.st_ino
    if before != after:
        raise ValueError(f"{label} identity changed while it was resolved")
    return metadata_after.st_dev, metadata_after.st_ino, str(resolved)


def _publication_directory_binding(
    destination: Path, receipt: Path, candidate: Path
) -> tuple[tuple[int, int, str], tuple[int, int, str]]:
    """Bind the exact run-root/case-directory publication hierarchy."""

    case_dir = destination.parent
    run_root = receipt.parent
    if (
        not destination.is_absolute()
        or not receipt.is_absolute()
        or destination.name != "results.json"
        or receipt.name != "cpu-afterany-validation.json"
        or candidate != case_dir / ".results.candidate.json"
        or case_dir.name != CASE_ID
        or case_dir.parent != run_root
    ):
        raise ValueError("publication paths escaped the immutable run hierarchy")
    run_binding = _directory_identity(run_root, label="publication run root")
    case_binding = _directory_identity(case_dir, label="publication case directory")
    if Path(case_binding[2]).parent != Path(run_binding[2]):
        raise ValueError("publication case directory escaped the resolved run root")
    return run_binding, case_binding


def _require_unchanged_publication_directories(
    destination: Path,
    receipt: Path,
    candidate: Path,
    expected: tuple[tuple[int, int, str], tuple[int, int, str]],
) -> None:
    if _publication_directory_binding(destination, receipt, candidate) != expected:
        raise ValueError("publication directory binding changed")


def _resolve_exact(path: str | Path, expected: Path, *, label: str) -> Path:
    supplied = Path(path)
    wanted = expected.absolute()
    if not supplied.is_absolute() or supplied != wanted:
        raise ValueError(f"{label} escaped its immutable path")
    try:
        metadata = supplied.lstat()
        observed = supplied.resolve(strict=True)
        resolved_wanted = wanted.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} cannot be resolved: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a non-symlink regular file")
    if observed != resolved_wanted:
        raise ValueError(f"{label} escaped its immutable path")
    return supplied


def _require_exact_child_directory(path: Path, parent: Path, *, label: str) -> Path:
    """Require one real directory immediately below a trusted real parent."""

    try:
        metadata = path.lstat()
        observed = path.resolve(strict=True)
        observed_parent = parent.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} cannot be resolved: {error}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a non-symlink directory")
    if observed.parent != observed_parent or observed.name != path.name:
        raise ValueError(f"{label} escaped its immutable parent")
    return path


def _validate_source_contract(
    contract_path: Path,
    *,
    expected_sha256: str,
    repo_root: Path,
    source_job_id: str,
) -> tuple[dict[str, Any], Path, Path]:
    if not _is_sha256(expected_sha256):
        raise ValueError("external source-contract SHA-256 is invalid")
    if (
        not contract_path.is_absolute()
        or contract_path.name != "source-contract.json"
        or contract_path.parent.parent != EXPERIMENT_ROOT
        or re.fullmatch(r"[A-Za-z0-9._-]+", contract_path.parent.name) is None
    ):
        raise ValueError("source contract escaped the immutable experiment root")
    path_run_id = contract_path.parent.name
    run_root = EXPERIMENT_ROOT / path_run_id
    _require_exact_child_directory(run_root, EXPERIMENT_ROOT, label="run root")
    _resolve_exact(contract_path, run_root / "source-contract.json", label="source contract")
    contract, contract_sha256 = _load_hashed_object(
        contract_path, label="source contract"
    )
    if contract_sha256 != expected_sha256:
        raise ValueError("source contract differs from the externally supplied SHA-256")
    if set(contract) != SOURCE_CONTRACT_KEYS:
        raise ValueError("source contract top-level keys changed")
    identity = {
        "schema_version": "2.0",
        "artifact_role": "r05a_sampled_current_canary_source_contract",
        "status": "gpu_held_sources_bound_before_cpu_submission",
        "git_dirty": False,
        "source_node": SOURCE_NODE,
        "gpu_slurm_array_job_id": source_job_id,
        "gpu_slurm_array_task_id": 0,
        "exact_gpu_task_id": f"{source_job_id}_0",
        "resources": SOURCE_RESOURCE_CONTRACT,
        "frozen_bindings": FROZEN_BINDINGS,
        "scientific_claim_allowed": False,
        "probe_training_authorized": False,
    }
    for key, expected in identity.items():
        if contract.get(key) != expected or type(contract.get(key)) is not type(expected):
            raise ValueError(f"source contract {key} changed")
    if not _is_commit(contract.get("git_commit")):
        raise ValueError("source contract git commit is invalid")
    run_id = contract.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9._-]+", run_id) is None:
        raise ValueError("source contract run id is invalid")
    if run_id != path_run_id:
        raise ValueError("source contract run id differs from its immutable path")
    case_dir = run_root / CASE_ID
    _require_exact_child_directory(case_dir, run_root, label="case directory")
    expected_paths = {
        "run_root": str(run_root),
        "case_dir": str(case_dir),
        "payload": str(case_dir / "canary-payload.json"),
        "host_telemetry": str(case_dir / "host-cgroup-sampled-current.tsv"),
        "gpu_samples": str(case_dir / "gpu-memory-samples.csv"),
        "allocation_tests_log": str(case_dir / "allocation-focused-tests.log"),
        "hidden_candidate": str(case_dir / ".results.candidate.json"),
        "result": str(case_dir / "results.json"),
        "validation_receipt": str(run_root / "cpu-afterany-validation.json"),
    }
    if contract.get("artifact_paths") != expected_paths:
        raise ValueError("source contract artifact paths changed")
    repository_hashes = contract.get("repository_file_sha256")
    if type(repository_hashes) is not dict or set(repository_hashes) != BOUND_REPOSITORY_PATHS:
        raise ValueError("source contract repository binding set changed")
    repository = repo_root.resolve(strict=True)
    for relative, recorded in repository_hashes.items():
        if not _is_sha256(recorded):
            raise ValueError(f"source contract hash is invalid for {relative}")
        candidate = (repository / relative).resolve(strict=True)
        if candidate != repository / relative:
            raise ValueError(f"source contract repository path escaped: {relative}")
        if file_sha256(candidate) != recorded:
            raise ValueError(f"source contract repository hash changed: {relative}")
    frozen_repo_hashes = {
        "configs/experiments/r05a_inverse_flow_canary.json": FROZEN_BINDINGS[
            "scientific_config_file_sha256"
        ],
        "manifests/r05a_inverse_flow_teacher_smoke.jsonl": FROZEN_BINDINGS[
            "manifest_sha256"
        ],
        "docs/decisions/0028-pivot-to-inverse-flow-transport.md": FROZEN_BINDINGS[
            "adr0028_sha256"
        ],
        "schemas/r05a-inverse-flow-canary.schema.json": FROZEN_BINDINGS[
            "historical_scientific_schema_sha256"
        ],
        "evidence/r03/r03-summary.json": FROZEN_BINDINGS[
            "source_r03_summary_sha256"
        ],
        "main/crfs_oracle/r05a_canary.py": FROZEN_BINDINGS[
            "r05a_canary_module_sha256"
        ],
        "main/run_crfs_r05a_canary.py": FROZEN_BINDINGS[
            "r05a_live_entrypoint_sha256"
        ],
        "openpi/src/openpi/models_pytorch/crfs_inverse_control.py": FROZEN_BINDINGS[
            "inverse_control_implementation_sha256"
        ],
        "openpi/src/openpi/models_pytorch/pi0_pytorch.py": FROZEN_BINDINGS[
            "pi05_sampler_sha256"
        ],
        "openpi/src/openpi/policies/policy.py": FROZEN_BINDINGS[
            "policy_boundary_sha256"
        ],
    }
    for relative, expected in frozen_repo_hashes.items():
        if repository_hashes.get(relative) != expected:
            raise ValueError(f"frozen science binding changed: {relative}")
    held_path = run_root / "held-gpu-submission.json"
    _resolve_exact(
        contract.get("held_gpu_submission_path", ""),
        held_path,
        label="held GPU receipt",
    )
    held_sha = contract.get("held_gpu_submission_sha256")
    held, observed_held_sha = _load_hashed_object(
        held_path, label="held GPU receipt"
    )
    if not _is_sha256(held_sha) or observed_held_sha != held_sha:
        raise ValueError("held GPU receipt binding changed")
    if set(held) != HELD_GPU_KEYS:
        raise ValueError("held GPU receipt keys changed")
    held_expected = {
        "schema_version": "2.0",
        "artifact_role": "r05a_sampled_current_canary_held_gpu_submission",
        "status": "sbatch_returned_held_gpu_id",
        "run_id": run_id,
        "git_commit": contract["git_commit"],
        "gpu_slurm_array_job_id": source_job_id,
        "gpu_slurm_array_task_id": 0,
        "exact_gpu_task_id": f"{source_job_id}_0",
        "source_node": SOURCE_NODE,
        "resources": SOURCE_RESOURCE_CONTRACT,
        "released_at_receipt_time": False,
        "scientific_claim_allowed": False,
        "probe_training_authorized": False,
    }
    for key, expected in held_expected.items():
        if held.get(key) != expected or type(held.get(key)) is not type(expected):
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
) -> tuple[dict[str, Any], str]:
    submission, submission_sha256 = _load_hashed_object(
        submission_path, label="atomic submission receipt"
    )
    if set(submission) != SUBMISSION_KEYS:
        raise ValueError("atomic submission receipt keys changed")
    expected = {
        "schema_version": "2.0",
        "artifact_role": "r05a_sampled_current_canary_atomic_submission",
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
        "probe_training_authorized": False,
    }
    for key, value in expected.items():
        if submission.get(key) != value or type(submission.get(key)) is not type(value):
            raise ValueError(f"atomic submission receipt {key} changed")
    return submission, submission_sha256


def _host_envelope(summary: Mapping[str, Any]) -> dict[str, Any]:
    native_state = summary["native_memory_peak_state"]
    if native_state == "readable_integer":
        native_state = "readable_positive_integer"
    lifecycle = summary["lifecycle_monotonic_ns"]
    return {
        "measurement_kind": (
            "job_scope_sampled_memory_current_lower_bound_with_hard_limit_and_hierarchical_events"
        ),
        "cgroup_version": 2,
        "kernel_osrelease": summary["kernel_osrelease"],
        "membership_path": summary["membership_path"],
        "mount_root": summary["mount_root"],
        "mount_point": summary["mount_point"],
        "job_scope_path": summary["job_scope_path"],
        "stopped_at_exact_job_boundary": True,
        "shared_ancestor_read": False,
        "sibling_or_descendant_enumeration": False,
        "memory_events_hierarchy_state": "hierarchical",
        "mount_memory_localevents": False,
        "raw_trace_path": summary["raw_trace_path"],
        "raw_trace_sha256": summary["raw_trace_sha256"],
        "sample_interval_requested_ms": summary["sample_interval_requested_ms"],
        "sample_count": summary["sample_count"],
        "first_monotonic_timestamp_ns": summary["first_sample_monotonic_ns"],
        "last_monotonic_timestamp_ns": summary["last_sample_monotonic_ns"],
        "timestamps_strictly_increasing": True,
        "maximum_adjacent_gap_ns": summary["maximum_adjacent_gap_ns"],
        "host_cgroup_sampled_current_high_water_bytes": summary[
            "host_cgroup_sampled_current_high_water_bytes"
        ],
        "host_cgroup_sampled_current_is_lower_bound_not_peak": True,
        "memory_max_before_bytes": summary["memory_max_before_bytes"],
        "memory_max_after_bytes": summary["memory_max_after_bytes"],
        "memory_max_unchanged": True,
        "memory_events_before": summary["memory_events_before"],
        "memory_events_after": summary["memory_events_after"],
        "memory_events_deltas": summary["memory_events_deltas"],
        "required_event_deltas_zero": {key: True for key in ("max", "oom", "oom_kill")},
        "memory_events_local_recorded_as_diagnostic_only": True,
        "native_memory_peak_state": native_state,
        "native_memory_peak_bytes": summary["native_memory_peak_value_bytes"],
        "lifecycle_markers": {
            "monitor_ready_ns": lifecycle["monitor_ready"],
            "policy_server_launch_ns": lifecycle["policy_launch"],
            "policy_server_cleanup_complete_ns": lifecycle["policy_cleanup_complete"],
            "gpu_monitor_cleanup_complete_ns": lifecycle["gpu_monitor_cleanup_complete"],
            "workload_cleanup_complete_ns": lifecycle["workload_cleanup_complete"],
            "monitor_stop_observed_ns": lifecycle["monitor_stop_observed"],
        },
        "first_sample_no_later_than_policy_launch": True,
        "last_sample_no_earlier_than_workload_cleanup": True,
        "monitor_ready_before_setup_python": True,
        "sealed_after_workload_cleanup": True,
    }


def _validate_schema(value: Mapping[str, Any], schema_path: Path) -> list[str]:
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema is required for sampled-current publication"]
    try:
        schema = _load_object(schema_path, label="sampled-current envelope schema")
        validator = jsonschema.Draft202012Validator(schema)
    except (ValueError, jsonschema.SchemaError) as error:
        return [f"cannot load sampled-current envelope schema: {error}"]
    return [
        "schema: " + error.message
        for error in sorted(
            validator.iter_errors(value),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        )
    ]


def build_sampled_current_envelope(
    *,
    payload_path: str | Path,
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
    result_path: str | Path,
    receipt_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build one fully recomputed envelope without writing it."""

    if re.fullmatch(r"[1-9][0-9]*", source_job_id) is None:
        raise ValueError("source job id is invalid")
    if re.fullmatch(r"[1-9][0-9]*", publisher_job_id) is None:
        raise ValueError("publisher job id is invalid")
    if source_job_state != "COMPLETED" or source_exit_code != "0:0":
        raise ValueError("source H100 task was not successful")
    repository = Path(repo_root).resolve(strict=True)
    observed_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    observed_dirty = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    contract_path = Path(source_contract_path)
    contract, run_root, case_dir = _validate_source_contract(
        contract_path,
        expected_sha256=expected_source_contract_sha256,
        repo_root=repository,
        source_job_id=source_job_id,
    )
    if observed_commit != contract["git_commit"] or observed_dirty:
        raise ValueError("publisher checkout differs from the clean source-contract commit")
    expected_files = {
        "payload": case_dir / "canary-payload.json",
        "host telemetry": case_dir / "host-cgroup-sampled-current.tsv",
        "GPU samples": case_dir / "gpu-memory-samples.csv",
        "allocation tests": case_dir / "allocation-focused-tests.log",
        "submission": run_root / "submission.json",
    }
    supplied = {
        "payload": payload_path,
        "host telemetry": host_telemetry_path,
        "GPU samples": gpu_samples_path,
        "allocation tests": allocation_tests_log,
        "submission": submission_path,
    }
    resolved: dict[str, Path] = {}
    for label, expected in expected_files.items():
        resolved[label] = _resolve_exact(supplied[label], expected, label=label)
    final_result = Path(result_path)
    final_receipt = Path(receipt_path)
    if final_result != case_dir / "results.json":
        raise ValueError("final result path changed")
    if final_receipt != run_root / "cpu-afterany-validation.json":
        raise ValueError("publication receipt path changed")
    _, submission_sha256 = _validate_submission(
        resolved["submission"],
        contract=contract,
        contract_path=contract_path,
        contract_sha256=expected_source_contract_sha256,
        source_job_id=source_job_id,
        publisher_job_id=publisher_job_id,
        result_path=final_result,
        receipt_path=final_receipt,
    )

    host_summary = parse_full_lifetime_telemetry(
        resolved["host telemetry"], expected_job_id=source_job_id
    )
    if file_sha256(resolved["host telemetry"]) != host_summary["raw_trace_sha256"]:
        raise ValueError("host telemetry changed while it was parsed")
    payload, payload_sha256 = _load_hashed_object(
        resolved["payload"], label="scientific payload"
    )
    if set(payload) != {
        "schema_version",
        "payload_type",
        "complete_result_requires_memory_finalization",
        "result_without_memory",
    }:
        raise ValueError("scientific payload keys changed")
    if payload.get("schema_version") != "1.0" or payload.get("payload_type") != (
        "r05a_inverse_flow_allocation_canary_payload"
    ):
        raise ValueError("scientific payload identity changed")
    if payload.get("complete_result_requires_memory_finalization") is not True:
        raise ValueError("scientific payload no longer awaits memory finalization")
    result = payload.get("result_without_memory")
    if type(result) is not dict or "memory" in result:
        raise ValueError("scientific payload contains a legacy memory record")
    if result.get("run_id") != contract["run_id"] or result.get("case_id") != CASE_ID:
        raise ValueError("scientific payload run or case identity changed")
    result_provenance = result.get("provenance")
    if not isinstance(result_provenance, Mapping) or not (
        result_provenance.get("git_commit") == contract["git_commit"]
        and result_provenance.get("git_dirty") is False
        and str(result_provenance.get("slurm_job_id")) == source_job_id
        and str(result_provenance.get("slurm_array_job_id")) == source_job_id
        and str(result_provenance.get("slurm_array_task_id")) == "0"
        and result_provenance.get("partition") == "main"
        and str(result_provenance.get("host", "")).split(".", 1)[0] == SOURCE_NODE
    ):
        raise ValueError("scientific payload provenance differs from the exact source task")

    # Deferred import keeps the raw telemetry parser dependency-free locally.
    from . import r05a_canary as legacy

    test_sha, test_counts = legacy._parse_allocation_test_log(resolved["allocation tests"])
    if file_sha256(resolved["allocation tests"]) != test_sha:
        raise ValueError("allocation test log changed while it was parsed")
    gpu_count, gpu_uuid, compute_high, device_high, gpu_sha = legacy._read_gpu_samples(
        resolved["GPU samples"]
    )
    if file_sha256(resolved["GPU samples"]) != gpu_sha:
        raise ValueError("GPU samples changed while they were parsed")
    view = copy.deepcopy(result)
    provenance = view.get("provenance")
    if type(provenance) is not dict:
        raise ValueError("scientific result provenance is missing")
    if "allocation_runtime_tests" in provenance or "gpu_samples_path" in provenance:
        raise ValueError("scientific payload already contains post-payload allocation metadata")
    provenance["allocation_runtime_tests"] = {
        "command_role": "dependency_backed_focused_tests_before_policy_server",
        "exit_code": 0,
        "log_path": str(resolved["allocation tests"]),
        "log_sha256": test_sha,
        "expected_counts": dict(legacy.ALLOCATION_TEST_COUNTS),
        "observed_counts": dict(test_counts),
        "registry_path": "main/crfs_oracle/r05a_allocation_tests.json",
        "registry_sha256": legacy.ALLOCATION_TEST_REGISTRY_SHA256,
        "zero_skips": True,
    }
    provenance["gpu_samples_path"] = str(resolved["GPU samples"])
    apparatus = view.get("apparatus")
    original_apparatus = result.get("apparatus")
    if type(apparatus) is not dict or type(original_apparatus) is not dict:
        raise ValueError("scientific pre-memory apparatus summary is missing")
    if set(apparatus) != PRE_MEMORY_APPARATUS_KEYS or apparatus.get("memory_pending") is not True:
        raise ValueError("scientific pre-memory apparatus summary changed shape")
    apparatus["memory_pending"] = False
    apparatus["memory_passed"] = True
    apparatus["passed"] = apparatus["passed_before_memory_finalization"]
    legacy_errors = legacy.validate_r05a_canary_result(view)
    if legacy_errors != LEGACY_MEMORY_ONLY_ERRORS:
        raise ValueError(
            "scientific validation returned non-memory or changed errors: "
            + "; ".join(legacy_errors)
        )

    first_trace = legacy._trace_from_record(
        result["solver"]["first"]["trace"], name="solver.first.trace"
    )
    duplicate_trace = legacy._trace_from_record(
        result["solver"]["duplicate"]["trace"], name="solver.duplicate.trace"
    )
    process_allocated, process_reserved = legacy._artifact_cuda_memory_peaks(
        first_trace, duplicate_trace
    )
    if result.get("provenance", {}).get("allocation_gpu_uuid") != gpu_uuid:
        raise ValueError("GPU raw samples differ from payload allocation UUID")
    pairing_checks = result.get("pairing", {}).get("checks")
    source_pairing_passed = bool(
        isinstance(pairing_checks, Mapping)
        and pairing_checks
        and all(value is True for value in pairing_checks.values())
    )
    if not source_pairing_passed:
        raise ValueError("scientific payload source pairing did not pass")
    simulator = result.get("simulator_use")
    expected_simulator = {
        "setup": "one_reset_plus_20_dummy_settle_control_steps",
        "policy_generated_action_steps_executed": 0,
        "teacher_generated_action_steps_executed": 0,
        "efficacy_rollouts_executed": 0,
        "simulator_efficacy_evaluated": False,
    }
    if simulator != expected_simulator:
        raise ValueError("scientific payload simulator-use contract changed")
    status = result.get("status")
    if status not in legacy.EXPECTED_RESULT_STATUSES:
        raise ValueError("scientific payload status changed")

    config_path = repository / APPARATUS_CONFIG_PATH
    schema_path = repository / ENVELOPE_SCHEMA_PATH
    apparatus_config, config_sha = _load_hashed_object(
        config_path, label="sampled-current apparatus config"
    )
    release = _validate_execution_release(
        apparatus_config, run_id=contract["run_id"]
    )
    _validate_release_commit(
        repository,
        observed_commit=observed_commit,
        execution_release=release,
    )
    schema_sha = file_sha256(schema_path)
    if apparatus_config.get("envelope_schema_sha256") != schema_sha:
        raise ValueError("apparatus config envelope-schema binding changed")
    if contract["repository_file_sha256"].get(APPARATUS_CONFIG_PATH) != config_sha:
        raise ValueError("apparatus config differs from the source contract")
    if contract["repository_file_sha256"].get(ENVELOPE_SCHEMA_PATH) != schema_sha:
        raise ValueError("envelope schema differs from the source contract")
    hidden_candidate = case_dir / ".results.candidate.json"
    classification = {
        "completed_converged": "mechanism_pass_only",
        "completed_nonconverged": "accepted_frozen_teacher_nonconvergence_stop",
        "completed_apparatus_failure": "apparatus_inconclusive",
    }[status]
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "gate": "R05A",
        "case_id": CASE_ID,
        "run_id": contract["run_id"],
        "status": status,
        "evidence_tier": "real_pi05_allocation_apparatus_canary_only",
        "scientific_claim_allowed": False,
        "probe_training_authorized": False,
        "ift01_authorized": False,
        "apparatus_config_sha256": config_sha,
        "envelope_schema_sha256": schema_sha,
        "frozen_bindings": dict(FROZEN_BINDINGS),
        "source_job": {
            "source_commit": contract["git_commit"],
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
        "allocation_tests": {
            "log_path": str(resolved["allocation tests"]),
            "log_sha256": test_sha,
            "registry_path": "main/crfs_oracle/r05a_allocation_tests.json",
            "registry_sha256": legacy.ALLOCATION_TEST_REGISTRY_SHA256,
            "expected_counts": dict(legacy.ALLOCATION_TEST_COUNTS),
            "observed_counts": dict(test_counts),
            "zero_skips": True,
            "exit_code": 0,
            "rehashed_and_reparsed_by_cpu_validator": True,
        },
        "scientific_payload": {
            "path": str(resolved["payload"]),
            "sha256": payload_sha256,
            "payload_type": payload["payload_type"],
            "embedded_result_member": "result_without_memory",
            "status": status,
            "scientific_config_hash": result.get("config_hash"),
            "bytes_unmodified_by_apparatus_wrapper": True,
            "legacy_memory_record_present": False,
            "sampled_current_written_to_legacy_peak_field": False,
            "direct_action_delta_applied_after_sampling": False,
            "source_pairing_passed": True,
            "teacher_generated_action_steps_executed": 0,
        },
        "semantic_validation": {
            "validator": "crfs_oracle.r05a_canary.validate_r05a_canary_result",
            "validator_module_sha256": FROZEN_BINDINGS["r05a_canary_module_sha256"],
            "legacy_view_allocation_tests_injected_after_independent_validation": True,
            "legacy_view_gpu_samples_path_injected_after_independent_validation": True,
            "legacy_view_host_memory_fabricated": False,
            "pre_memory_apparatus_summary_exact": True,
            "legacy_view_apparatus_memory_fields_adapted": True,
            "legacy_view_apparatus_memory_pending": False,
            "legacy_view_apparatus_memory_passed": True,
            "legacy_view_apparatus_passed_copied_from_pre_memory_result": True,
            "returned_legacy_errors": list(legacy_errors),
            "returned_legacy_errors_exact_allowlist": True,
            "nonmemory_error_count": 0,
            "recomputed_status": status,
        },
        "telemetry": {
            "host": _host_envelope(host_summary),
            "gpu": {
                "allocation_gpu_uuid": gpu_uuid,
                "raw_samples_path": str(resolved["GPU samples"]),
                "raw_samples_sha256": gpu_sha,
                "sample_count": gpu_count,
                "process_peak_allocated_bytes": process_allocated,
                "process_peak_reserved_bytes": process_reserved,
                "sampled_compute_high_water_mib": compute_high,
                "sampled_device_high_water_mib": device_high,
                "sampled_values_are_periodic_not_continuous_peaks": True,
                "out_of_memory_observed": False,
            },
            "all_raw_files_rehashed_by_cpu_validator": True,
            "telemetry_contract_passed": True,
        },
        "simulator_use": expected_simulator,
        "publication": {
            "source_contract_receipt_path": str(contract_path),
            "source_contract_receipt_sha256": expected_source_contract_sha256,
            "externally_supplied_expected_receipt_sha256": expected_source_contract_sha256,
            "atomic_submission_receipt_path": str(resolved["submission"]),
            "atomic_submission_receipt_sha256": submission_sha256,
            "hidden_candidate_path": str(hidden_candidate),
            "published_result_path": str(final_result),
            "post_publication_validation_receipt_path": str(final_receipt),
            "cpu_validator_job_id": publisher_job_id,
            "dependency": f"afterany:{source_job_id}",
            "cpu_validator_partition": "main",
            "cpu_validator_account": "normal",
            "cpu_validator_qos": "normal",
            "cpu_validator_cpus_per_task": 2,
            "cpu_validator_host_memory_mib": 8192,
            "cpu_validator_time_limit": "00:15:00",
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
            "classification": classification,
            "simulator_efficacy_claim_allowed": False,
            "inverse_flow_infeasibility_claim_allowed": False,
            "student_learning_claim_allowed": False,
            "automatic_next_gate_allowed": False,
        },
    }
    schema_errors = _validate_schema(envelope, schema_path)
    if schema_errors:
        raise ValueError("sampled-current envelope schema failed: " + "; ".join(schema_errors))
    return envelope


def publish_sampled_current_envelope(
    *,
    payload_path: str | Path,
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
    result_path: str | Path,
    receipt_path: str | Path,
    repo_root: str | Path,
) -> tuple[Path, Path]:
    """Build, validate twice, and atomically publish one final envelope."""

    destination = Path(result_path)
    receipt = Path(receipt_path)
    candidate = destination.parent / ".results.candidate.json"
    if destination.exists() or candidate.exists() or receipt.exists():
        raise ValueError("publication paths must not pre-exist")
    keyword = {
        "payload_path": payload_path,
        "host_telemetry_path": host_telemetry_path,
        "gpu_samples_path": gpu_samples_path,
        "allocation_tests_log": allocation_tests_log,
        "source_contract_path": source_contract_path,
        "expected_source_contract_sha256": expected_source_contract_sha256,
        "submission_path": submission_path,
        "source_job_id": source_job_id,
        "source_job_state": source_job_state,
        "source_exit_code": source_exit_code,
        "publisher_job_id": publisher_job_id,
        "result_path": result_path,
        "receipt_path": receipt_path,
        "repo_root": repo_root,
    }
    envelope = build_sampled_current_envelope(**keyword)
    directory_binding = _publication_directory_binding(destination, receipt, candidate)
    _atomic_write_json(candidate, envelope)
    try:
        _require_unchanged_publication_directories(
            destination, receipt, candidate, directory_binding
        )
        candidate_value, candidate_sha256 = _load_hashed_object(
            candidate, label="hidden result candidate"
        )
        rebuilt = build_sampled_current_envelope(**keyword)
        if candidate_value != rebuilt:
            raise ValueError(
                "candidate differs after raw-file rehash and independent rebuild"
            )
        schema_errors = _validate_schema(
            candidate_value, Path(repo_root) / ENVELOPE_SCHEMA_PATH
        )
        if schema_errors:
            raise ValueError(
                "candidate revalidation failed: " + "; ".join(schema_errors)
            )
        final_fingerprints = (
            (
                candidate_value["scientific_payload"]["path"],
                candidate_value["scientific_payload"]["sha256"],
                "scientific payload",
            ),
            (
                candidate_value["allocation_tests"]["log_path"],
                candidate_value["allocation_tests"]["log_sha256"],
                "allocation test log",
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
                "atomic submission receipt",
            ),
        )
        for source_path, expected_sha256, label in final_fingerprints:
            if file_sha256(source_path) != expected_sha256:
                raise ValueError(f"{label} changed after candidate creation")
        candidate_identity_before = _regular_file_identity(
            candidate, label="hidden result candidate"
        )
        candidate_value_before_rename, candidate_sha256_before_rename = (
            _load_hashed_object(candidate, label="hidden result candidate")
        )
        candidate_identity_after = _regular_file_identity(
            candidate, label="hidden result candidate"
        )
        if candidate_identity_before != candidate_identity_after:
            raise ValueError("hidden result candidate identity changed during final validation")
        if (
            candidate_sha256_before_rename != candidate_sha256
            or candidate_value_before_rename != candidate_value
        ):
            raise ValueError("hidden result candidate bytes changed after validation")
        _require_unchanged_publication_directories(
            destination, receipt, candidate, directory_binding
        )
    except Exception:
        candidate.unlink(missing_ok=True)
        raise
    try:
        os.replace(candidate, destination)
        _require_unchanged_publication_directories(
            destination, receipt, candidate, directory_binding
        )
        if _regular_file_identity(
            destination, label="published result"
        ) != candidate_identity_after:
            raise ValueError("published result identity differs from validated candidate")
        published_value, published_sha256 = _load_hashed_object(
            destination, label="published result"
        )
        if published_sha256 != candidate_sha256 or published_value != candidate_value:
            raise ValueError("published result bytes differ from validated candidate")
        published_schema_errors = _validate_schema(
            published_value, Path(repo_root) / ENVELOPE_SCHEMA_PATH
        )
        if published_schema_errors:
            raise ValueError(
                "published result revalidation failed: "
                + "; ".join(published_schema_errors)
            )
        if _regular_file_identity(
            destination, label="published result"
        ) != candidate_identity_after:
            raise ValueError("published result identity changed during revalidation")
        _require_unchanged_publication_directories(
            destination, receipt, candidate, directory_binding
        )
        publication_receipt = {
            "schema_version": SCHEMA_VERSION,
            "artifact_role": "r05a_sampled_current_canary_cpu_publication",
            "source_job_id": source_job_id,
            "publisher_job_id": publisher_job_id,
            "source_contract_path": str(source_contract_path),
            "source_contract_sha256_external": expected_source_contract_sha256,
            "submission_path": candidate_value["publication"][
                "atomic_submission_receipt_path"
            ],
            "submission_sha256": candidate_value["publication"][
                "atomic_submission_receipt_sha256"
            ],
            "result_path": str(destination),
            "result_sha256": candidate_sha256,
            "passed": True,
            "published": True,
            "errors": [],
            "scientific_claim_allowed": False,
            "probe_training_authorized": False,
        }
        _atomic_write_json(receipt, publication_receipt)
        _require_unchanged_publication_directories(
            destination, receipt, candidate, directory_binding
        )
        final_value, final_sha256 = _load_hashed_object(
            destination, label="published result"
        )
        if (
            _regular_file_identity(destination, label="published result")
            != candidate_identity_after
            or final_sha256 != candidate_sha256
            or final_value != candidate_value
        ):
            raise ValueError("published result changed before receipt finalization")
    except Exception:
        # A final envelope is valid only together with its post-publication
        # receipt.  Roll back files created by this call if that pair cannot be
        # completed; pre-existing paths were rejected above.
        destination.unlink(missing_ok=True)
        receipt.unlink(missing_ok=True)
        raise
    return destination, receipt


__all__ = [
    "BOUND_REPOSITORY_PATHS",
    "EXECUTION_RELEASE_KEYS",
    "FROZEN_BINDINGS",
    "LEGACY_MEMORY_ONLY_ERRORS",
    "RELEASE_DECISION_PATH",
    "RELEASE_ONLY_PATHS",
    "SOURCE_RESOURCE_CONTRACT",
    "build_sampled_current_envelope",
    "file_sha256",
    "publish_sampled_current_envelope",
]
