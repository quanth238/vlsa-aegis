"""Immutable case-manifest construction and deterministic seed schedules."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

from .artifacts import canonical_json


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


def validate_case(case: Mapping[str, Any]) -> list[str]:
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
