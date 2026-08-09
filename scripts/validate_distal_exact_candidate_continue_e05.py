#!/usr/bin/env python3
"""Validate the unsafe suffix-continuation E05 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.exact_candidate_continue import (
    CONTINUE_RESULT_SCHEMA,
    CONTINUE_VALIDATION_SCHEMA,
    load_continue_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    _atomic_write,
    _file_sha256,
    _load,
    _require,
)


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_continue_config(args.config.resolve())
    result = _load(args.result.resolve())
    _require(
        result.get("schema_version") == CONTINUE_RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and result.get("source", {}).get("commit") == args.expected_commit
        and result.get("source", {}).get("dirty") is False
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256"),
        "continuation result identity differs",
    )
    _require(
        result.get("config", {}).get("config_payload_sha256")
        == config["config_payload_sha256"],
        "continuation config differs",
    )
    allocation = result.get("allocation", {})
    _require(
        "H100" in str(allocation.get("device", {}).get("name"))
        and str(allocation.get("slurm_job_id", "")).isdigit(),
        "continuation result is not H100 allocation-backed",
    )
    prerequisite = result.get("prerequisite", {})
    _require(
        prerequisite.get("result_file_sha256")
        == config["prerequisite"]["exact_candidate_result_file_sha256"]
        and prerequisite.get("result_payload_sha256")
        == config["prerequisite"]["exact_candidate_result_payload_sha256"]
        and prerequisite.get("validation_file_sha256")
        == config["prerequisite"]["exact_candidate_validation_file_sha256"]
        and prerequisite.get("validated_empty_safe_set_step") == 187,
        "continuation prerequisite differs",
    )
    records = result.get("actions", [])
    _require(
        isinstance(records, list)
        and [item.get("step") for item in records] == list(range(len(records)))
        and 187 < len(records) <= config["primary_case"]["expected_action_count"],
        "continuation action ledger differs",
    )
    prefix = records[:187]
    suffix = records[187:]
    _require(
        len(prefix) == 187
        and all(item.get("executed") is True for item in prefix)
        and result.get("prefix_replay", {}).get("executed_action_count") == 187
        and result.get("prefix_replay", {}).get(
            "all_dynamic_state_hashes_match"
        ) is True
        and float(result.get("prefix_replay", {}).get(
            "maximum_obstacle_displacement_error_m", 1.0
        )) <= 1.0e-10,
        "continuation prefix replay differs",
    )
    for item in suffix:
        _require(
            item.get("executed") is True
            and item.get("safety_filter_bypassed") is True
            and item.get("unsafe_continuation") is True
            and item.get("candidate_search_used") is False
            and item.get("affine_QP_used") is False
            and item.get("nominal_action") == item.get("executed_action"),
            "unsafe suffix contract differs at step %s" % item.get("step"),
        )
    def first_step(predicate: Any) -> Optional[int]:
        return next((int(item["step"]) for item in suffix if predicate(item)), None)

    first_proxy = first_step(lambda item: min(
        float(value) for value in item["executed_measurement"][
            "minimum_all_eight_substep_clearance_m"
        ]
    ) < 0.0)
    first_contact = first_step(lambda item: int(
        item["executed_measurement"]["raw_protected_contact_count"]
    ) > 0)
    first_car = first_step(lambda item: float(
        item["active_obstacle_l1_displacement_m"]
    ) > 1.0e-3)
    task_success_step = next((
        int(item["step"]) for item in records
        if item["executed_measurement"]["done"] is True
    ), None)
    continuation = result.get("continuation", {})
    _require(
        continuation.get("start_step") == 187
        and continuation.get("action_count") == len(suffix)
        and continuation.get("total_executed_action_count") == len(records)
        and continuation.get("safety_filter_bypassed") is True
        and continuation.get("candidate_search_used") is False
        and continuation.get("affine_QP_used") is False
        and continuation.get("first_proxy_violation_step") == first_proxy
        and continuation.get("first_protected_contact_step") == first_contact
        and continuation.get("first_paper_CAR_step") == first_car
        and continuation.get("native_task_success_step") == task_success_step
        and continuation.get("native_task_success")
        is (task_success_step is not None),
        "continuation summary differs",
    )
    decision = result.get("decision", {})
    _require(
        decision.get("continued_after_empty_safe_set") is True
        and decision.get("native_task_success") is (task_success_step is not None)
        and decision.get("collision_free")
        is (first_contact is None and first_car is None)
        and decision.get("safety_success") is False
        and decision.get("deployable_method_demonstrated") is False
        and decision.get("neural_training_authorized") is False,
        "continuation decision differs",
    )
    output = {
        "schema_version": CONTINUE_VALIDATION_SCHEMA,
        "status": "validated",
        "scientific_result": False,
        "expected_commit": args.expected_commit,
        "result_file_sha256": _file_sha256(args.result.resolve()),
        "result_payload_sha256": result["result_payload_sha256"],
        "total_executed_action_count": len(records),
        "unsafe_suffix_action_count": len(suffix),
        "first_proxy_violation_step": first_proxy,
        "first_protected_contact_step": first_contact,
        "first_paper_CAR_step": first_car,
        "native_task_success": task_success_step is not None,
        "native_task_success_step": task_success_step,
        "continued_after_empty_safe_set": True,
        "safety_success": False,
    }
    _atomic_write(args.output.resolve(), output)
    print(json.dumps(output, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
