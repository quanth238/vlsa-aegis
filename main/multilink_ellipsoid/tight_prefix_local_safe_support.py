"""Pure contracts for auditing exact-safe support inside a correction ball."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


CONFIG_SCHEMA = "vlsa_tight_prefix_local_safe_support.v1"
RESULT_SCHEMA = "vlsa_tight_prefix_local_safe_support_result.v1"
VALIDATION_SCHEMA = "vlsa_tight_prefix_local_safe_support_validation.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    if set(value) != {
        "schema_version", "protocol_id", "claim_scope", "source", "audit",
        "prior_probe_anchors", "forbidden",
    }:
        raise ValueError("local safe-support config keys differ")
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"]
        != "vlsa-tight-prefix-local-safe-support-v1"
    ):
        raise ValueError("local safe-support protocol differs")
    audit = value["audit"]
    if audit != {
        "report_splits": ["train", "validation", "test"],
        "decision_split": "validation",
        "test_status": "already_opened_diagnostic",
        "primary_rows": list(range(8)),
        "diagnostic_rows": [8, 9],
        "translation_correction_radius_l2_action": 0.25,
        "distance_tolerance": 1e-12,
        "non_translation_action_tolerance": 1e-12,
        "anchor_definition": "maximum_primary_exact_row_Q_gt_0",
        "safe_definition": (
            "all_primary_exact_row_Q_le_0_and_no_primary_contact_and_CAR_pass"
        ),
        "grouping": "complete_saved_state_and_candidate_chunk",
    }:
        raise ValueError("local safe-support definition differs")
    anchors = value["prior_probe_anchors"]
    if len(anchors) != 6 or len({
        (item["case_id"], item["candidate_name"]) for item in anchors
    }) != 6:
        raise ValueError("local safe-support prior anchors differ")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def nearest_safe_support(
    *, anchor_name: str, anchor_actions: Sequence[Sequence[float]],
    safe_candidates: Sequence[Mapping[str, Any]], radius: float,
    tolerance: float, non_translation_tolerance: float,
) -> dict[str, Any]:
    import numpy as np

    anchor = np.asarray(anchor_actions, dtype=np.float64)
    if anchor.shape != (5, 7) or not np.all(np.isfinite(anchor)):
        raise ValueError("local safe-support anchor action differs")
    records = []
    inaccessible_count = 0
    for candidate in safe_candidates:
        actions = np.asarray(candidate["actions"], dtype=np.float64)
        if actions.shape != (5, 7) or not np.all(np.isfinite(actions)):
            raise ValueError("local safe-support candidate action differs")
        delta = actions[:, :3] - anchor[:, :3]
        non_translation_delta = actions[:, 3:] - anchor[:, 3:]
        non_translation_maximum = float(np.max(np.abs(non_translation_delta)))
        if non_translation_maximum > float(non_translation_tolerance):
            inaccessible_count += 1
            continue
        records.append({
            "candidate_name": str(candidate["name"]),
            "translation_distance_l2_action": float(np.linalg.norm(delta)),
            "translation_distance_linf_action": float(np.max(np.abs(delta))),
            "non_translation_maximum_absolute_delta": non_translation_maximum,
        })
    records.sort(key=lambda item: (
        item["translation_distance_l2_action"], item["candidate_name"],
    ))
    nearest = None if not records else records[0]
    supported = bool(
        nearest is not None
        and float(nearest["translation_distance_l2_action"])
        <= float(radius) + float(tolerance)
    )
    return {
        "anchor_name": str(anchor_name),
        "safe_candidate_count": len(records),
        "inaccessible_safe_candidate_count": int(inaccessible_count),
        "nearest_safe_candidate": nearest,
        "safe_candidate_within_radius": supported,
        "correction_radius_l2_action": float(radius),
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    distances = [
        float(item["nearest_safe_candidate"][
            "translation_distance_l2_action"
        ])
        for item in records if item["nearest_safe_candidate"] is not None
    ]
    supported = [item for item in records if item["safe_candidate_within_radius"]]
    return {
        "unsafe_anchor_count": len(records),
        "anchor_with_any_known_safe_count": len(distances),
        "local_safe_supported_anchor_count": len(supported),
        "local_safe_supported_root_count": len({
            str(item["case_id"]) for item in supported
        }),
        "minimum_nearest_safe_distance_l2_action": (
            None if not distances else min(distances)
        ),
        "median_nearest_safe_distance_l2_action": (
            None if not distances else float(statistics.median(distances))
        ),
    }


def scientific_view(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in result.items()
        if key not in {"source", "allocation", "result_payload_sha256"}
    }
