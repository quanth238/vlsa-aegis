#!/usr/bin/env python3
"""Audit the six frozen whole-body Q-only selected-action false-safes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any, Optional, Sequence

from main.multilink_ellipsoid.whole_body_false_safe_audit import (
    RESULT_SCHEMA, attribute_selection, selected_false_safe_states, summarize,
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _payload_sha256(value: dict[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return _sha256(_canonical(payload))


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(value) + b"\n")
    os.replace(temporary, path)


def audit(*, config_path: Path) -> dict[str, Any]:
    config = _load(config_path)
    _require(
        config["schema_version"]
        == "vlsa_distal_whole_body_false_safe_audit_config.v1",
        "false-safe audit config schema differs",
    )
    prediction_path = Path(config["prediction_result"]["path"])
    _require(
        _file_sha256(prediction_path)
        == config["prediction_result"]["file_sha256"],
        "prediction result file differs",
    )
    prediction = _load(prediction_path)
    _require(
        prediction["result_payload_sha256"]
        == config["prediction_result"]["payload_sha256"],
        "prediction result payload differs",
    )
    selections = selected_false_safe_states(prediction)
    expected = config["expected_selected_false_safe_state_ids"]
    _require(
        [row["state_id"] for row in selections] == expected,
        "selected false-safe population differs",
    )
    source_by_case = {
        row["case_id"]: row for row in prediction["source_artifacts"]["cases"]
    }
    rows = []
    bindings = []
    for selection in selections:
        state_id = selection["state_id"]
        source = source_by_case[state_id]
        source_root = Path(config["source_roots"][source["source"]])
        case_path = source_root / "producer" / (state_id + ".json")
        case = _load(case_path)
        _require(
            case["result_payload_sha256"] == source["payload_sha256"],
            "case payload differs from fixed training source",
        )
        rows.append(attribute_selection(selection, case))
        bindings.append({
            "case_id": state_id,
            "source": source["source"],
            "path": str(case_path),
            "file_sha256": _file_sha256(case_path),
            "payload_sha256": case["result_payload_sha256"],
        })
    value = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": True,
        "claim_scope": config["claim_scope"],
        "allocation": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "host": socket.gethostname(),
            "device": "cpu_read_only_audit",
        },
        "prediction_result": {
            "path": str(prediction_path),
            "file_sha256": _file_sha256(prediction_path),
            "payload_sha256": prediction["result_payload_sha256"],
        },
        "source_bindings": bindings,
        "rows": rows,
        "summary": summarize(rows),
        "interpretation": (
            "false_safes_span_L6_L5_and_palm_transfer"
            if len({row["active_physical_witness"] for row in rows}) > 1
            else "false_safes_are_single_constraint_dominated"
        ),
        "correction_authorized": False,
    }
    value["audit_payload_sha256"] = _payload_sha256(
        value, "audit_payload_sha256"
    )
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = audit(config_path=args.config.resolve())
    _atomic_write(args.output.resolve(), value)
    print(json.dumps(value["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
