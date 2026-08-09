"""Exact-contact MLP helpers for the opt-in E05 feasibility diagnostic.

The targets in this module are simulator contact events, not ellipsoid
clearances.  Keeping that distinction explicit prevents a classifier score
from being reported as a physical distance or a CBF certificate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence


CONTACT_RANKER_SCHEMA = "vlsa_distal_contact_ranker_e05.v1"
PROTECTED_LINKS = ("robot0_link5", "robot0_link6", "robot0_link7")


def load_contact_ranker_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "protocol_id",
        "case_id",
        "claim_scope",
        "immutable_sources",
        "state_groups",
        "candidate_set",
        "network",
        "training",
        "calibration",
        "gradient_audit",
        "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("contact-ranker config keys differ")
    if config["schema_version"] != CONTACT_RANKER_SCHEMA:
        raise ValueError("contact-ranker schema differs")
    if config["case_id"] != "vlsa-t1-goal-ii-t0-e05":
        raise ValueError("contact-ranker case differs")
    groups = config["state_groups"]
    if set(groups) != {"collect_steps", "train_steps", "validation_steps", "test_steps"}:
        raise ValueError("contact-ranker state-group keys differ")
    collect = [int(value) for value in groups["collect_steps"]]
    partitions = {
        name: [int(value) for value in groups[name]]
        for name in ("train_steps", "validation_steps", "test_steps")
    }
    flattened = sum(partitions.values(), [])
    if sorted(collect) != sorted(flattened) or len(flattened) != len(set(flattened)):
        raise ValueError("contact-ranker states are not a disjoint complete split")
    if collect != list(range(184, 193)):
        raise ValueError("contact-ranker must collect the registered E05 suffix")
    if config["network"] != {
        "hidden_widths": [64, 64],
        "input_count": 33,
        "output_count": 3,
        "output_semantics": [
            "L5_contact_probability",
            "L6_contact_probability",
            "L7_contact_probability",
        ],
    }:
        raise ValueError("contact-ranker network differs")
    training = config["training"]
    if set(training) != {
        "seed", "epochs", "learning_rate", "weight_decay",
        "early_stopping_patience", "minimum_standard_deviation",
    }:
        raise ValueError("contact-ranker training keys differ")
    for key in ("learning_rate", "minimum_standard_deviation"):
        value = float(training[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("contact-ranker training value is invalid")
    return config


def link_contact_labels(events: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return one exact binary label for each protected body."""

    contacted = set()
    for event in events:
        body = str(event.get("protected_body_name", ""))
        if body not in PROTECTED_LINKS:
            raise ValueError("unexpected protected contact body: %s" % body)
        contacted.add(body)
    return [int(body in contacted) for body in PROTECTED_LINKS]


def calibrate_zero_false_safe_threshold(
    risks: Sequence[float], unsafe: Sequence[bool]
) -> float:
    """Largest strict threshold below every validation unsafe prediction."""

    if len(risks) != len(unsafe) or not risks:
        raise ValueError("calibration arrays differ or are empty")
    unsafe_risks = [float(risk) for risk, label in zip(risks, unsafe) if bool(label)]
    if not unsafe_risks:
        raise ValueError("validation split lacks an unsafe candidate")
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in risks):
        raise ValueError("calibration risk is invalid")
    minimum = min(unsafe_risks)
    if minimum <= 0.0:
        return 0.0
    # ``math.nextafter`` is absent from the cluster's Python 3.8 build.  Risks
    # are finite, positive IEEE-754 doubles, so decrementing the bit pattern is
    # exactly the next representable value toward zero.
    bits = struct.unpack(">Q", struct.pack(">d", minimum))[0]
    return struct.unpack(">d", struct.pack(">Q", bits - 1))[0]


def select_minimal_predicted_safe(
    records: Sequence[Mapping[str, Any]],
    risks: Sequence[float],
    threshold: float,
) -> int | None:
    """Choose the least Cartesian correction among conservatively safe rows."""

    if len(records) != len(risks):
        raise ValueError("selection arrays differ")
    candidates = []
    for index, (record, risk) in enumerate(zip(records, risks)):
        if float(risk) >= float(threshold):
            continue
        nominal = [float(value) for value in record["nominal_xyz"]]
        candidate = [float(value) for value in record["candidate_xyz"]]
        if len(nominal) != 3 or len(candidate) != 3:
            raise ValueError("selection action dimension differs")
        squared = sum((value - base) ** 2 for value, base in zip(candidate, nominal))
        candidates.append((squared, float(risk), index))
    return None if not candidates else min(candidates)[2]
