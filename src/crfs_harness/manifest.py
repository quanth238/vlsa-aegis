"""Immutable case-manifest construction and deterministic seed schedules."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from typing import Any

from .artifacts import canonical_json


GENERATED_SOURCE_ESTIMAND = "task0_single_obstacle_generated_v1"
GENERATED_SAFETY_LEVEL = "generated"
GENERATED_TASK_SUITE = "safelibero_spatial"
GENERATED_TASK_INDEX = 0
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def deterministic_seed(namespace: str, *parts: object) -> int:
    material = canonical_json([namespace, *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & 0x7FFF_FFFF


def case_id(task_suite: str, safety_level: str, task_index: int, episode_index: int, policy_seed: int) -> str:
    payload = {
        "task_suite": task_suite,
        "safety_level": safety_level,
        "task_index": task_index,
        "episode_index": episode_index,
        "policy_seed": policy_seed,
    }
    return f"crfs-{hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]}"


def build_cases(
    task_suite: str,
    safety_level: str,
    task_index: int,
    episode_indices: Iterable[int],
    seeds_per_episode: int,
    seed_namespace: str,
) -> list[dict[str, Any]]:
    if seeds_per_episode <= 0:
        raise ValueError("seeds_per_episode must be positive")
    cases: list[dict[str, Any]] = []
    for episode_index in episode_indices:
        for seed_index in range(seeds_per_episode):
            policy_seed = deterministic_seed(seed_namespace, task_suite, safety_level, task_index, episode_index, seed_index)
            cases.append(
                {
                    "schema_version": "1.0",
                    "case_id": case_id(task_suite, safety_level, task_index, episode_index, policy_seed),
                    "task_suite": task_suite,
                    "safety_level": safety_level,
                    "task_index": task_index,
                    "episode_index": episode_index,
                    "environment_seed": deterministic_seed(
                        "environment", task_suite, safety_level, task_index, episode_index
                    ),
                    "policy_seed": policy_seed,
                    "random_control_seed": deterministic_seed("random-control", policy_seed),
                    "group_id": f"{task_suite}:{safety_level}:{task_index}:{episode_index}",
                }
            )
    return cases


def generated_source_state_id(source_branch_sha256: str) -> str:
    """Return the immutable identity of one generated simulator branch.

    A flattened MuJoCo state does not contain model-level randomized fixture
    transforms.  The caller therefore supplies the complete branch digest,
    which also binds the finalized model, BDDL, observation, and geometry.
    Identity never depends on generation index, path, acceptance, or outcome.
    """

    if not _SHA256_PATTERN.fullmatch(str(source_branch_sha256)):
        raise ValueError("source_branch_sha256 must be 64 lowercase hexadecimal characters")
    return f"gsrc-{source_branch_sha256[:16]}"


def generated_case_id(
    source_estimand: str,
    source_branch_sha256: str,
    policy_seed: int,
) -> str:
    """Bind a policy draw to one generated state without path dependence."""

    generated_source_state_id(source_branch_sha256)
    if not isinstance(policy_seed, int) or policy_seed < 0:
        raise ValueError("policy_seed must be a non-negative integer")
    payload = {
        "source_estimand": str(source_estimand),
        "source_branch_sha256": source_branch_sha256,
        "policy_seed": policy_seed,
    }
    return f"crfs-{hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]}"


def build_generated_cases(
    *,
    task_suite: str,
    task_index: int,
    sources: Iterable[Mapping[str, Any]],
    seeds_per_source: int,
    seed_namespace: str,
) -> list[dict[str, Any]]:
    """Build schema-v2 policy cases from independently generated sources.

    ``sources`` contains only source identity and provenance. Scientific
    outcomes such as collision, clearance, progress, planner feasibility, or
    probe scores are deliberately not accepted by this constructor.
    """

    if seeds_per_source <= 0:
        raise ValueError("seeds_per_source must be positive")
    if task_suite != GENERATED_TASK_SUITE or task_index != GENERATED_TASK_INDEX:
        raise ValueError(
            "task0_single_obstacle_generated_v1 is bound to "
            f"{GENERATED_TASK_SUITE} task index {GENERATED_TASK_INDEX}"
        )
    cases: list[dict[str, Any]] = []
    seen_branches: set[str] = set()
    seen_source_indices: set[int] = set()
    for source in sources:
        allowed = {
            "source_index",
            "source_estimand",
            "source_state_id",
            "source_branch_sha256",
            "source_state_sha256",
            "source_bundle_sha256",
            "source_bundle_path",
            "environment_seed",
        }
        unexpected = set(source) - allowed
        if unexpected:
            raise ValueError(f"generated source contains non-identity fields: {sorted(unexpected)}")
        source_estimand = str(source.get("source_estimand", ""))
        if source_estimand != GENERATED_SOURCE_ESTIMAND:
            raise ValueError(f"source_estimand must be {GENERATED_SOURCE_ESTIMAND}")
        branch_sha256 = str(source.get("source_branch_sha256", ""))
        state_id = generated_source_state_id(branch_sha256)
        if source.get("source_state_id") != state_id:
            raise ValueError(f"source_state_id does not match source_branch_sha256; expected {state_id}")
        if branch_sha256 in seen_branches:
            raise ValueError(f"duplicate generated source branch: {state_id}")
        seen_branches.add(branch_sha256)
        state_sha256 = str(source.get("source_state_sha256", ""))
        if not _SHA256_PATTERN.fullmatch(state_sha256):
            raise ValueError("source_state_sha256 must be 64 lowercase hexadecimal characters")
        bundle_sha256 = str(source.get("source_bundle_sha256", ""))
        if not _SHA256_PATTERN.fullmatch(bundle_sha256):
            raise ValueError("source_bundle_sha256 must be 64 lowercase hexadecimal characters")
        bundle_path = source.get("source_bundle_path")
        if not isinstance(bundle_path, str) or not bundle_path:
            raise ValueError("source_bundle_path must be a non-empty string")
        source_index = source.get("source_index")
        environment_seed = source.get("environment_seed")
        for name, value in (("source_index", source_index), ("environment_seed", environment_seed)):
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if source_index in seen_source_indices:
            raise ValueError(f"duplicate generated source_index: {source_index}")
        seen_source_indices.add(source_index)
        group_id = f"{source_estimand}:{state_id}"
        for seed_index in range(seeds_per_source):
            policy_seed = deterministic_seed(
                seed_namespace,
                source_estimand,
                branch_sha256,
                seed_index,
            )
            cases.append(
                {
                    "schema_version": "2.0",
                    "case_id": generated_case_id(source_estimand, branch_sha256, policy_seed),
                    "task_suite": task_suite,
                    "safety_level": GENERATED_SAFETY_LEVEL,
                    "task_index": task_index,
                    "episode_index": source_index,
                    "environment_seed": environment_seed,
                    "policy_seed": policy_seed,
                    "random_control_seed": deterministic_seed("random-control", policy_seed),
                    "group_id": group_id,
                    "source_index": source_index,
                    "source_estimand": source_estimand,
                    "source_state_id": state_id,
                    "source_branch_sha256": branch_sha256,
                    "source_state_sha256": state_sha256,
                    "source_bundle_sha256": bundle_sha256,
                    "source_bundle_path": bundle_path,
                }
            )
    return cases


def validate_case(case: Mapping[str, Any]) -> list[str]:
    if case.get("schema_version") == "2.0":
        return _validate_generated_case(case)
    required = {
        "schema_version",
        "case_id",
        "task_suite",
        "safety_level",
        "task_index",
        "episode_index",
        "environment_seed",
        "policy_seed",
        "random_control_seed",
        "group_id",
    }
    errors = []
    missing = required - set(case)
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
    if case.get("safety_level") not in {"I", "II"}:
        errors.append("safety_level must be I or II")
    for key in ("task_index", "episode_index", "environment_seed", "policy_seed", "random_control_seed"):
        if not isinstance(case.get(key), int) or case[key] < 0:
            errors.append(f"{key} must be a non-negative integer")
    expected = None
    if not errors:
        expected = case_id(
            str(case["task_suite"]),
            str(case["safety_level"]),
            int(case["task_index"]),
            int(case["episode_index"]),
            int(case["policy_seed"]),
        )
    if expected is not None and case.get("case_id") != expected:
        errors.append(f"case_id does not match canonical identity; expected {expected}")
    return errors


def _validate_generated_case(case: Mapping[str, Any]) -> list[str]:
    required = {
        "schema_version",
        "case_id",
        "task_suite",
        "safety_level",
        "task_index",
        "episode_index",
        "environment_seed",
        "policy_seed",
        "random_control_seed",
        "group_id",
        "source_index",
        "source_estimand",
        "source_state_id",
        "source_branch_sha256",
        "source_state_sha256",
        "source_bundle_sha256",
        "source_bundle_path",
    }
    errors: list[str] = []
    missing = required - set(case)
    if missing:
        errors.append(f"missing generated fields: {sorted(missing)}")
    unexpected = set(case) - required
    if unexpected:
        errors.append(f"unexpected generated fields: {sorted(unexpected)}")
    if case.get("safety_level") != GENERATED_SAFETY_LEVEL:
        errors.append("generated cases must use safety_level='generated', never Level I or II")
    if case.get("source_estimand") != GENERATED_SOURCE_ESTIMAND:
        errors.append(f"source_estimand must be {GENERATED_SOURCE_ESTIMAND}")
    if case.get("task_suite") != GENERATED_TASK_SUITE or case.get("task_index") != GENERATED_TASK_INDEX:
        errors.append(
            "generated source estimand is bound to "
            f"{GENERATED_TASK_SUITE} task index {GENERATED_TASK_INDEX}"
        )
    for key in (
        "task_index",
        "episode_index",
        "source_index",
        "environment_seed",
        "policy_seed",
        "random_control_seed",
    ):
        value = case.get(key)
        if not isinstance(value, int) or value < 0:
            errors.append(f"{key} must be a non-negative integer")
    if isinstance(case.get("episode_index"), int) and isinstance(case.get("source_index"), int):
        if case["episode_index"] != case["source_index"]:
            errors.append("episode_index must equal source_index for generated cases")
    branch_sha256 = case.get("source_branch_sha256")
    state_sha256 = case.get("source_state_sha256")
    bundle_sha256 = case.get("source_bundle_sha256")
    if not isinstance(branch_sha256, str) or not _SHA256_PATTERN.fullmatch(branch_sha256):
        errors.append("source_branch_sha256 must be 64 lowercase hexadecimal characters")
    if not isinstance(state_sha256, str) or not _SHA256_PATTERN.fullmatch(state_sha256):
        errors.append("source_state_sha256 must be 64 lowercase hexadecimal characters")
    if not isinstance(bundle_sha256, str) or not _SHA256_PATTERN.fullmatch(bundle_sha256):
        errors.append("source_bundle_sha256 must be 64 lowercase hexadecimal characters")
    if not isinstance(case.get("source_bundle_path"), str) or not case.get("source_bundle_path"):
        errors.append("source_bundle_path must be a non-empty string")
    expected_state_id = None
    if isinstance(branch_sha256, str) and _SHA256_PATTERN.fullmatch(branch_sha256):
        expected_state_id = generated_source_state_id(branch_sha256)
        if case.get("source_state_id") != expected_state_id:
            errors.append(f"source_state_id does not match canonical identity; expected {expected_state_id}")
    if expected_state_id is not None and case.get("group_id") != (
        f"{GENERATED_SOURCE_ESTIMAND}:{expected_state_id}"
    ):
        errors.append("group_id does not match generated source-state identity")
    if (
        isinstance(branch_sha256, str)
        and _SHA256_PATTERN.fullmatch(branch_sha256)
        and isinstance(case.get("policy_seed"), int)
        and case["policy_seed"] >= 0
    ):
        expected_case_id = generated_case_id(
            GENERATED_SOURCE_ESTIMAND,
            branch_sha256,
            case["policy_seed"],
        )
        if case.get("case_id") != expected_case_id:
            errors.append(f"case_id does not match canonical identity; expected {expected_case_id}")
    return errors
