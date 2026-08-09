#!/usr/bin/env python3
"""Validate the no-training action-188 boundary-capacity dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from main.multilink_ellipsoid.boundary_capacity import (
    BOUNDARY_DATASET_RESULT_SCHEMA,
    BOUNDARY_DATASET_SCHEMA,
    load_boundary_capacity_config,
)
from scripts.replay_distal_three_ellipsoid_multicbf import (
    CASE_ID,
    _atomic_write,
    _file_sha256,
    _load,
    _require,
)


VALIDATION_SCHEMA = "vlsa_distal_boundary_capacity_e05_dataset_validation.v1"


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def validate(
    result_path: Path,
    *,
    expected_commit: str,
    expected_config: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    result = _load(result_path)
    _require(
        result.get("schema_version") == BOUNDARY_DATASET_RESULT_SCHEMA
        and result.get("status") == "complete"
        and result.get("scientific_result") is True
        and result.get("case_id") == CASE_ID
        and result.get("failure") is None
        and result.get("result_payload_sha256")
        == _hash_without(result, "result_payload_sha256"),
        "boundary-capacity dataset result identity differs",
    )
    _require(result.get("config") == expected_config, "boundary config differs")
    source = result.get("source", {})
    allocation = result.get("allocation", {})
    _require(
        source.get("commit") == expected_commit
        and source.get("dirty") is False
        and str(allocation.get("slurm_job_id", "")).isdigit()
        and "H100" in str(allocation.get("device", {}).get("name", "")),
        "boundary dataset source or H100 allocation differs",
    )
    geometry = result.get("geometry", {})
    _require(
        geometry.get("accepted_robot_and_released_ee", {}).get(
            "distal_ellipsoid_count"
        )
        == 7
        and geometry.get("exact_obstacle_boxes", {}).get("exact_box_count") == 15
        and geometry.get("exact_obstacle_boxes", {}).get("geometric_inflation")
        == 0.0,
        "boundary dataset geometry differs",
    )
    identity = result.get("dataset", {})
    dataset_path = Path(identity.get("path", "")).resolve()
    _require(
        dataset_path.is_file()
        and not dataset_path.is_symlink()
        and dataset_path.parent == result_path.parent.resolve()
        and identity.get("file_sha256") == _file_sha256(dataset_path),
        "boundary dataset file identity differs",
    )
    dataset = _load(dataset_path)
    settings = expected_config["sampling"]
    records = dataset.get("records", [])
    _require(
        dataset.get("schema_version") == BOUNDARY_DATASET_SCHEMA
        and dataset.get("dataset_payload_sha256")
        == _hash_without(dataset, "dataset_payload_sha256")
        == identity.get("payload_sha256")
        and dataset.get("source_commit") == expected_commit
        and dataset.get("config_file_sha256")
        == expected_config["config_file_sha256"]
        and dataset.get("config_payload_sha256")
        == expected_config["config_payload_sha256"]
        and len(records)
        == identity.get("record_count")
        == settings["expected_grid_action_count"]
        + settings["expected_gradient_probe_count"],
        "boundary dataset payload or dimensions differ",
    )
    groups: dict[str, set[str]] = {}
    group_categories: dict[str, set[str]] = {}
    grid_records = []
    anchor_records = []
    probe_records = []
    for record in records:
        _require(
            len(record.get("feature_vector", [])) == 33
            and len(record.get("current_clearance_m", [])) == 7
            and len(record.get("minimum_substep_clearance_m", [])) == 7
            and len(record.get("minimum_substep_witnesses", [])) == 7
            and record.get("split") in {"train", "validation", "test"}
            and isinstance(record.get("D_opt_proxy_safe"), bool)
            and isinstance(record.get("D_sim_raw_safe"), bool)
            and isinstance(record.get("geometry_consistency"), dict),
            "boundary dataset record is incomplete",
        )
        group = str(record.get("group_id"))
        groups.setdefault(group, set()).add(str(record["split"]))
        group_categories.setdefault(group, set()).add(str(record.get("category")))
        current = np.asarray(record["current_clearance_m"], dtype=np.float64)
        minimum = np.asarray(
            record["minimum_substep_clearance_m"], dtype=np.float64
        )
        expected_raw_safe = bool(
            record.get("raw_protected_contact_count") == 0
            and record.get("maximum_within_step_obstacle_l1_displacement_m", 1.0)
            <= 1.0e-4
        )
        _require(
            np.all(np.isfinite(current))
            and np.all(np.isfinite(minimum))
            and np.all(minimum <= current + 1.0e-12)
            and record["D_opt_proxy_safe"] is bool(np.all(minimum >= 0.0))
            and record["D_sim_raw_safe"] is expected_raw_safe,
            "boundary D_opt/D_sim record is inconsistent",
        )
        if record.get("source") == "grid":
            grid_records.append(record)
            if record.get("gradient_m_per_action") is not None:
                anchor_records.append(record)
        elif record.get("source") == "gradient_probe":
            probe_records.append(record)
        else:
            raise ValueError("boundary dataset record source differs")
    _require(
        all(len(splits) == 1 for splits in groups.values())
        and all(len(categories) == 1 for categories in group_categories.values())
        and len(grid_records) == settings["expected_grid_action_count"]
        and len(anchor_records) == settings["gradient_anchor_count"]
        and len(probe_records) == settings["expected_gradient_probe_count"]
        and all(
            len(record.get("gradient_m_per_action", [])) == 7
            and len(record.get("gradient_valid_rows", [])) == 7
            for record in anchor_records
        ),
        "boundary dataset grouping or gradient dimensions differ",
    )
    for anchor in anchor_records:
        selected = [
            item
            for item in probe_records
            if item.get("anchor_grid_index") == anchor.get("grid_index")
        ]
        _require(
            len(selected) == 6
            and {(item["probe_dimension"], item["probe_sign"]) for item in selected}
            == {(dimension, sign) for dimension in range(3) for sign in (-1, 1)}
            and all(item["group_id"] == anchor["group_id"] for item in selected),
            "boundary finite-difference group is incomplete",
        )
        computed = np.zeros((7, 3), dtype=np.float64)
        expected_valid = np.ones(7, dtype=bool)
        anchor_witness = anchor["minimum_substep_witnesses"]
        for dimension in range(3):
            minus = next(
                item
                for item in selected
                if item["probe_dimension"] == dimension
                and item["probe_sign"] == -1
            )
            plus = next(
                item
                for item in selected
                if item["probe_dimension"] == dimension
                and item["probe_sign"] == 1
            )
            computed[:, dimension] = (
                np.asarray(plus["minimum_substep_clearance_m"], dtype=np.float64)
                - np.asarray(minus["minimum_substep_clearance_m"], dtype=np.float64)
            ) / (2.0 * settings["finite_difference_epsilon_action"])
            for row in range(7):
                expected_valid[row] = bool(
                    expected_valid[row]
                    and anchor_witness[row]
                    == minus["minimum_substep_witnesses"][row]
                    == plus["minimum_substep_witnesses"][row]
                )
        _require(
            np.allclose(
                computed,
                np.asarray(anchor["gradient_m_per_action"], dtype=np.float64),
                rtol=0.0,
                atol=1.0e-12,
            )
            and np.array_equal(
                expected_valid,
                np.asarray(anchor["gradient_valid_rows"], dtype=bool),
            ),
            "boundary finite-difference gradient or witness validity differs",
        )
    summary = result.get("dataset_summary", {})
    categories = summary.get("grid_category_counts", {})
    stable = summary.get("stable_critical_gradient_anchor_count_by_split", {})
    nominal = summary.get("nominal_transition", {})
    expected_gate = bool(
        categories.get("boundary_safe", 0)
        >= settings["gradient_anchor_safe_count"]
        and categories.get("boundary_unsafe", 0)
        >= settings["gradient_anchor_unsafe_count"]
        and min((stable.get(name, 0) for name in ("train", "validation", "test")))
        >= 1
    )
    recomputed_categories = {
        name: sum(record["category"] == name for record in grid_records)
        for name in (
            "boundary_safe",
            "boundary_unsafe",
            "far_safe",
            "far_unsafe",
        )
    }
    recomputed_stable = {
        name: sum(
            record["split"] == name
            and bool(record["gradient_valid_rows"][1])
            for record in anchor_records
        )
        for name in ("train", "validation", "test")
    }
    _require(
        summary.get("grid_action_count") == len(grid_records)
        and summary.get("gradient_anchor_count") == len(anchor_records)
        and summary.get("gradient_probe_count") == len(probe_records)
        and summary.get("record_count") == len(records)
        and len(nominal.get("start_clearance_m", [])) == 7
        and min(nominal["start_clearance_m"]) >= 0.0
        and nominal.get("minimum_substep_clearance_m", [0.0, 0.0])[1] < 0.0
        and categories == recomputed_categories
        and stable == recomputed_stable
        and summary.get("dataset_capacity_gate_pass") is expected_gate,
        "boundary dataset summary or registered crossing differs",
    )
    decision = result.get("decision", {})
    _require(
        decision.get("dataset_capacity_gate_pass") is expected_gate
        and decision.get("neural_training_authorized") is expected_gate
        and ((decision.get("stop_reason") is None) is expected_gate),
        "boundary dataset decision differs",
    )
    return {
        "schema_version": VALIDATION_SCHEMA,
        "status": "valid",
        "scientific_result": False,
        "case_id": CASE_ID,
        "source_commit": expected_commit,
        "slurm_job_id": allocation["slurm_job_id"],
        "result_file_sha256": _file_sha256(result_path),
        "result_payload_sha256": result["result_payload_sha256"],
        "dataset_file_sha256": identity["file_sha256"],
        "dataset_payload_sha256": identity["payload_sha256"],
        "dataset_capacity_gate_pass": expected_gate,
        "neural_training_authorized": expected_gate,
        "checks": {
            "clean_H100_source": True,
            "no_training_before_dataset_gate": True,
            "same_state_exact_OSC_labels": True,
            "grouped_boundary_stratification": True,
            "D_opt_D_sim_separation": True,
            "finite_difference_witness_audit": True,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = validate(
        args.result.resolve(),
        expected_commit=args.expected_commit,
        expected_config=load_boundary_capacity_config(args.config.resolve()),
    )
    _atomic_write(args.output.resolve(), record)
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
