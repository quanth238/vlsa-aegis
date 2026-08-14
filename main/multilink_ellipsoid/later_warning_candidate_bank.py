"""Contracts for the opened E42 step-105 candidate-bank recoverability gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_distal_later_warning_candidate_bank.v1"
RESULT_SCHEMA = "vlsa_distal_later_warning_candidate_bank_result.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "source_result",
        "source_result_schema", "source_result_commit",
        "source_result_file_sha256", "source_result_payload_sha256",
        "selection_manifest", "selection_manifest_file_sha256", "case_id",
        "case_index", "state_step", "query_index", "base_risk_config",
        "base_risk_config_file_sha256", "normal_curve_config",
        "normal_curve_config_file_sha256", "candidate_bank",
        "source_state_replay_tolerance_m", "selection_rule",
        "unknown_timeout_handling", "gate", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("later-warning candidate-bank config keys differ")
    if value["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("later-warning candidate-bank schema differs")
    if value["protocol_id"] != "vlsa-distal-later-warning-candidate-bank-v1":
        raise ValueError("later-warning candidate-bank protocol differs")
    if value["case_id"] != "vlsa-t1-goal-ii-t3-e42":
        raise ValueError("later-warning case differs")
    if int(value["case_index"]) != 1:
        raise ValueError("later-warning case index differs")
    if int(value["state_step"]) != 105 or int(value["query_index"]) != 21:
        raise ValueError("later-warning state binding differs")
    if float(value["source_state_replay_tolerance_m"]) != 1.0e-9:
        raise ValueError("later-warning replay tolerance differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def _one(rows: Sequence[Mapping[str, Any]], *, label: str) -> Mapping[str, Any]:
    if len(rows) != 1:
        raise ValueError("later-warning %s count differs" % label)
    return rows[0]


def derive_replay_archive(
    source: Mapping[str, Any], sealed: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Create the immutable action ledger needed to replay the changed step-105 state.

    The clean complete-episode artifact stores actual executed commands but not a
    standalone MuJoCo snapshot.  Replaying those commands from the paired initial
    state recovers the physical/controller state.  The exact raw and post-AEGIS
    nominal chunk is reconstructed from the recorded query's QP records.
    """

    import numpy as np

    if source.get("schema_version") != config["source_result_schema"]:
        raise ValueError("later-warning source schema differs")
    if source.get("case_id") != config["case_id"]:
        raise ValueError("later-warning source case differs")
    if source.get("source", {}).get("commit") != config["source_result_commit"]:
        raise ValueError("later-warning source commit differs")
    if source.get("result_payload_sha256") != config["source_result_payload_sha256"]:
        raise ValueError("later-warning source payload differs")
    if sealed.get("case_id") != config["case_id"] or sealed.get("task_success") is not True:
        raise ValueError("later-warning sealed baseline differs")

    state_step = int(config["state_step"])
    query_index = int(config["query_index"])
    action_by_step = {int(row["step"]): row for row in source["actions"]}
    if not set(range(state_step)).issubset(action_by_step):
        raise ValueError("later-warning executed history differs")
    query = _one(
        [row for row in source["policy_queries"]
         if int(row["query_index"]) == query_index],
        label="policy query",
    )
    qp_records = list(query["nominal_qp_records"])
    if len(qp_records) != 5:
        raise ValueError("later-warning nominal QP horizon differs")
    raw = np.asarray(
        [row["context"]["nominal_translational"] for row in qp_records],
        dtype=np.float64,
    )
    post = np.asarray(
        [row["context"]["executed_action"] for row in qp_records],
        dtype=np.float64,
    )
    if raw.shape != (5, 7) or post.shape != (5, 7):
        raise ValueError("later-warning nominal action shape differs")
    z_before = np.asarray(qp_records[0]["z_before"], dtype=np.float64)
    if z_before.shape != (3,) or not np.all(np.isfinite(z_before)):
        raise ValueError("later-warning AEGIS memory differs")

    rows = []
    for step in range(state_step):
        executed = np.asarray(action_by_step[step]["action"], dtype=np.float64)
        if executed.shape != (7,) or not np.all(np.isfinite(executed)):
            raise ValueError("later-warning replay action differs")
        row = {
            "step": step,
            "executed": executed.tolist(),
            "nominal_translational": executed.tolist(),
            "qp": {},
        }
        rows.append(row)
    rows[-1]["qp"] = {"z_after": z_before.tolist()}
    for offset, qp in enumerate(qp_records):
        rows.append({
            "step": state_step + offset,
            "executed": post[offset].tolist(),
            "nominal_translational": raw[offset].tolist(),
            "qp": qp,
        })

    placeholder_queries = [
        {"query_index": index, "rng_seed": None}
        for index in range(query_index)
    ]
    placeholder_queries.append(query)
    derived = {
        "schema_version": "vlsa_distal_later_warning_replay_ledger.v1",
        "case_id": config["case_id"],
        "task_success": True,
        "perception": sealed["perception"],
        "actions": rows,
        "policy_queries": placeholder_queries,
        "source_binding": {
            "result_payload_sha256": source["result_payload_sha256"],
            "state_step": state_step,
            "query_index": query_index,
            "returned_actions_sha256": query["returned_actions_sha256"],
        },
    }
    derived["result_payload_sha256"] = hashlib.sha256(canonical(derived)).hexdigest()
    return derived


def minimum_actual_intervention(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    safe = [row for row in candidates if bool(row["exact_safe"])]
    if not safe:
        return None
    selected = min(
        safe,
        key=lambda row: (
            float(row["effective_post_AEGIS_correction_l2_action"]),
            int(row["order"]),
        ),
    )
    return {
        "name": str(selected["name"]),
        "order": int(selected["order"]),
        "requested_alpha": float(selected["requested_alpha"]),
        "effective_post_AEGIS_correction_l2_action": float(
            selected["effective_post_AEGIS_correction_l2_action"]
        ),
        "combined_worst_risk_m": max(
            float(value) for value in selected["combined_risk"]
        ),
        "terminal_status": str(selected["terminal_status"]),
    }
