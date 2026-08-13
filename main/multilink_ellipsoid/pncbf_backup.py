"""Pure contracts for the fixed warning-triggered repulsive backup oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if value.get("protocol_id") not in {
        "vlsa-distal-pncbf-backup-oracle-e05-v1",
        "vlsa-distal-pncbf-backup-frozen-proposal-e05-v1",
    }:
        raise ValueError("PNCBF backup protocol differs")
    if value.get("case_ids") != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("PNCBF backup case differs")
    if value["state_protocol"]["registered_initial_prefix_steps"] != list(range(182, 187)):
        raise ValueError("PNCBF backup initial prefix differs")
    if value["state_protocol"]["lookahead_actions"] != 5 or value["state_protocol"]["execute_prefix_actions"] != 5:
        raise ValueError("PNCBF backup horizon differs")
    if value["backup_policy"]["candidate_correction_l2_action"] != [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        raise ValueError("PNCBF backup scale grid differs")
    if value["backup_policy"]["additional_direction_or_detour_modes"] is not False:
        raise ValueError("PNCBF backup must contain only normal repulsion")
    if value["backup_policy"]["physically_unsafe_nominal_forces_backup"] is not True:
        raise ValueError("PNCBF backup must reject physically unsafe nominal actions")
    if value.get("protocol_id") == "vlsa-distal-pncbf-backup-frozen-proposal-e05-v1":
        proposal = value["registered_inputs"].get("task_successful_proposal_result", {})
        if proposal.get("slurm_job_id") != "39354":
            raise ValueError("frozen proposal source differs")
    if not 0.0 < float(value["warning"]["activation_clearance_m"]) < float(value["warning"]["release_clearance_m"]):
        raise ValueError("PNCBF backup hysteresis thresholds differ")
    output = json.loads(_canonical(value).decode())
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(_canonical(value)).hexdigest()
    return output


def update_latch(active: bool, minimum_clearance_m: float, activation_m: float, release_m: float) -> bool:
    """Schmitt-trigger warning: activate low, release only after clear."""

    if active:
        return not float(minimum_clearance_m) >= float(release_m)
    return float(minimum_clearance_m) < float(activation_m)


def physical_safe(record: Mapping[str, Any], paper_car_threshold_m: float) -> bool:
    return bool(
        len(record["protected_contacts"]) == 0
        and float(record["maximum_active_obstacle_l1_displacement_m"])
        <= float(paper_car_threshold_m)
    )


def future_clearance(record: Mapping[str, Any]) -> float:
    """Minimum after command authority begins; exclude the immutable k=0 row."""
    trace = record.get("clearance_trace_m")
    if trace is None:
        return float(record.get("future_minimum_clearance_m", record["minimum_clearance_m"]))
    if len(trace) <= 1:
        raise ValueError("future clearance requires at least one executed sample")
    return min(float(value) for row in trace[1:] for value in row)


def select_repulsive_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *, nominal_clearance_m: float,
    activation_clearance_m: float,
    paper_car_threshold_m: float,
) -> Mapping[str, Any] | None:
    safe = [
        item
        for item in candidates
        if physical_safe(item["record"], paper_car_threshold_m)
        and float(item["record"].get("future_minimum_clearance_m", item["record"]["minimum_clearance_m"]))
        > float(nominal_clearance_m) + 1.0e-9
    ]
    buffered = [
        item
        for item in safe
        if float(item["record"].get("future_minimum_clearance_m", item["record"]["minimum_clearance_m"]))
        >= float(activation_clearance_m)
    ]
    if buffered:
        return min(buffered, key=lambda item: (float(item["correction_l2_action"]), -float(item["record"].get("future_minimum_clearance_m", item["record"]["minimum_clearance_m"]))))
    if safe:
        return max(safe, key=lambda item: (float(item["record"].get("future_minimum_clearance_m", item["record"]["minimum_clearance_m"])), -float(item["correction_l2_action"])))
    return None
