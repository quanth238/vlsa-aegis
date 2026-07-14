"""Oracle-stage aggregation with explicit evaluation populations."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def aggregate_results(results: Sequence[Mapping[str, Any]], bootstrap_seed: int = 0, samples: int = 2000) -> dict[str, Any]:
    completed = [result for result in results if result.get("status") == "completed"]
    colliding = [result for result in completed if not result["trials"]["nominal"]["safe"]]
    feasible = [result for result in colliding if result["repair"].get("feasible")]
    all_colliding_count = len(colliding) + sum(result.get("status") == "infeasible" for result in results)

    oracle_rate = _rate(feasible, lambda result: result["trials"]["oracle_residual"]["safe"])
    random_rate = _rate(feasible, lambda result: result["trials"]["random_residual"]["safe"])
    direct_rate = _rate(feasible, lambda result: result["trials"]["direct_repair"]["safe"])
    bridge_rate = _rate(feasible, lambda result: result["trials"]["bridge_edit"]["safe"])
    paired_values = [
        int(result["trials"]["oracle_residual"]["safe"]) - int(result["trials"]["random_residual"]["safe"])
        for result in feasible
    ]
    groups = _group_values(feasible, paired_values)
    ci = clustered_bootstrap_mean(groups, bootstrap_seed, samples) if groups else [None, None]
    overall_oracle_rescues = sum(result["trials"]["oracle_residual"]["safe"] for result in feasible)
    return {
        "schema_version": "1.0",
        "counts": {
            "input_results": len(results),
            "completed": len(completed),
            "colliding_completed": len(colliding),
            "feasible_colliding": len(feasible),
            "all_colliding_including_infeasible": all_colliding_count,
        },
        "feasible_conditioned": {
            "oracle_rescue_rate": oracle_rate,
            "random_rescue_rate": random_rate,
            "direct_repair_safety_rate": direct_rate,
            "bridge_edit_rescue_rate": bridge_rate,
            "oracle_minus_random": None if oracle_rate is None or random_rate is None else oracle_rate - random_rate,
            "oracle_minus_random_cluster_bootstrap_95ci": ci,
            "median_oracle_endpoint_error_m": _median(
                [result["trials"]["oracle_residual"]["endpoint_error_m"] for result in feasible]
            ),
        },
        "overall": {
            "oracle_rescue_rate_all_colliding": (
                overall_oracle_rescues / all_colliding_count if all_colliding_count else None
            ),
            "projection_verified_coverage": len(feasible) / all_colliding_count if all_colliding_count else None,
        },
    }


def clustered_bootstrap_mean(groups: Mapping[str, Sequence[float]], seed: int, samples: int) -> list[float]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    keys = sorted(groups)
    if not keys:
        raise ValueError("At least one group is required")
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [rng.choice(keys) for _ in keys]
        values = [value for key in selected for value in groups[key]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    lower = estimates[int(0.025 * (len(estimates) - 1))]
    upper = estimates[int(0.975 * (len(estimates) - 1))]
    return [lower, upper]


def _group_values(results: Sequence[Mapping[str, Any]], values: Sequence[float]) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = {}
    for result, value in zip(results, values, strict=True):
        group = str(result["provenance"].get("group_id", result["case_id"]))
        groups.setdefault(group, []).append(value)
    return groups


def _rate(results: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]) -> float | None:
    return sum(bool(predicate(result)) for result in results) / len(results) if results else None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])
