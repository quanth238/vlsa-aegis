"""Contact-consistent adaptive MuJoCo distance diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .native_geom_margin import PROTECTED_BODY_NAMES, _native_model_data
from .oracle_affine import SubstepEightConstraintProbe, _body_lineage


CONFIG_SCHEMA = "vlsa_distal_adaptive_native_distance_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_adaptive_native_distance_moka10_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_adaptive_native_distance_moka10_validation.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return _sha256(_canonical(payload))


def load_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("adaptive native-distance config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "population", "measurement", "gate", "decision",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("adaptive native-distance config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-adaptive-native-distance-moka10-v1"
    ):
        raise ValueError("adaptive native-distance protocol differs")
    if set(config["immutable_source"]) != {
        "source_population_manifest_sha256", "test_manifest_sha256",
        "expanded_dataset_file_sha256", "expanded_dataset_payload_sha256",
        "geometry_config_file_sha256", "exact_box_config_file_sha256",
        "prior_inventory_result_file_sha256",
        "prior_inventory_result_payload_sha256",
        "prior_inventory_validation_file_sha256",
    }:
        raise ValueError("adaptive native-distance immutable source differs")
    if config["population"] != {
        "case_ids": [
            "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10",
            "vlsa-t1-goal-ii-t0-e15",
        ],
        "expected_state_count": 15,
        "state_identity": (
            "exact_paired_settled_state_plus_canonical_archived_action_"
            "prefix_replayed_in_current_runtime"
        ),
        "primary_contact_witness_steps": [187, 188],
    }:
        raise ValueError("adaptive native-distance population differs")
    measurement = config["measurement"]
    if measurement != {
        "distance_api": "mujoco.mj_geomDistance",
        "overlap_distmax_m": 0.0,
        "positive_distmax_sequence_m": [
            0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064,
        ],
        "query_tolerance_m": 1.0e-6,
        "repeated_uncensored_consistency_tolerance_m": 1.0e-5,
        "raw_contact_distance_threshold_m": 0.0,
        "horizon_actions": 2,
        "temporal_resolution": "every_internal_MuJoCo_substep_in_cloned_OSC",
        "right_censored_value_semantics": "safe_lower_bound_not_exact_target",
        "contact_consistency": (
            "distmax_zero_negative_iff_registered_raw_pair_contact"
        ),
    }:
        raise ValueError("adaptive native-distance measurement differs")
    if config["gate"] != {
        "semantic_protected_group_inventory_hash_count": 1,
        "maximum_negative_without_raw_pair_contact_count": 0,
        "maximum_raw_pair_contact_without_negative_count": 0,
        "maximum_cutoff_induced_negative_count": 0,
        "maximum_repeated_uncensored_inconsistency_count": 0,
        "required_registered_raw_contact_count": 1,
        "required_initially_safe_test_state_count": 15,
        "training_in_this_gate": False,
        "QP_in_this_gate": False,
        "closed_loop_in_this_gate": False,
    }:
        raise ValueError("adaptive native-distance gate differs")
    if config["decision"] != {
        "pass": "preregister_full_grouped_physical_candidate_target_collection",
        "failure": "reject_mj_geomDistance_and_use_contact_or_external_geometry_oracle",
        "closed_loop_E05_authorized": False,
    }:
        raise ValueError("adaptive native-distance decision differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def _name(model: Any, kind: str, identifier: int) -> str:
    value = getattr(model, "%s_id2name" % kind)(int(identifier))
    return "" if value is None else str(value)


def _raw_pair_contacts(
    env: Any, inventory: Mapping[str, Any], threshold_m: float,
) -> tuple[set[tuple[int, int]], list[dict[str, Any]]]:
    model = env.sim.model
    obstacle_root = int(inventory["active_obstacle_root_body_id"])
    protected_ids = {
        int(model.body_name2id(name)): str(name) for name in PROTECTED_BODY_NAMES
    }
    pairs = set()
    records = []
    for contact_index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[contact_index]
        distance = float(contact.dist)
        if distance > float(threshold_m):
            continue
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        if body1 in protected_ids and obstacle_root in _body_lineage(model, body2):
            protected_geom, obstacle_geom = geom1, geom2
        elif body2 in protected_ids and obstacle_root in _body_lineage(model, body1):
            protected_geom, obstacle_geom = geom2, geom1
        else:
            continue
        pairs.add((protected_geom, obstacle_geom))
        records.append({
            "contact_index": int(contact_index),
            "protected_geom_id": protected_geom,
            "protected_geom_name": _name(model, "geom", protected_geom),
            "obstacle_geom_id": obstacle_geom,
            "obstacle_geom_name": _name(model, "geom", obstacle_geom),
            "contact_distance_m": distance,
        })
    return pairs, records


def _pair_query(
    native_model: Any, native_data: Any, protected_geom: int,
    obstacle_geom: int, distmax_m: float,
) -> dict[str, Any]:
    import mujoco
    import numpy as np

    fromto = np.zeros(6, dtype=np.float64)
    distance = float(mujoco.mj_geomDistance(
        native_model, native_data, int(protected_geom), int(obstacle_geom),
        float(distmax_m), fromto,
    ))
    return {
        "distmax_m": float(distmax_m), "distance_m": distance,
        "fromto_m": fromto.tolist(),
        "finite": bool(math.isfinite(distance) and np.all(np.isfinite(fromto))),
    }


def measure_adaptive_groups(
    env: Any, inventory: Mapping[str, Any], measurement: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure contact sign at zero then bracket positive safe clearance."""

    native_model, native_data = _native_model_data(env)
    tolerance = float(measurement["query_tolerance_m"])
    repeated_tolerance = float(
        measurement["repeated_uncensored_consistency_tolerance_m"]
    )
    raw_pairs, raw_records = _raw_pair_contacts(
        env, inventory, float(measurement["raw_contact_distance_threshold_m"])
    )
    registered_pairs = {
        (int(group["protected_geom_id"]), int(obstacle_geom))
        for group in inventory["groups"]
        for obstacle_geom in group["obstacle_geom_ids"]
    }
    group_records = []
    pair_records = []
    for group in inventory["groups"]:
        group_pairs = []
        for obstacle_geom in group["obstacle_geom_ids"]:
            pair = (int(group["protected_geom_id"]), int(obstacle_geom))
            zero = _pair_query(
                native_model, native_data, pair[0], pair[1],
                float(measurement["overlap_distmax_m"]),
            )
            zero_negative = bool(zero["distance_m"] < -tolerance)
            raw_contact = pair in raw_pairs
            positive_queries = []
            first_uncensored = None
            cutoff_negative = False
            uncensored_values = []
            for cutoff in measurement["positive_distmax_sequence_m"]:
                query = _pair_query(
                    native_model, native_data, pair[0], pair[1], float(cutoff)
                )
                uncensored = bool(query["distance_m"] < float(cutoff) - tolerance)
                query["uncensored"] = uncensored
                positive_queries.append(query)
                if query["distance_m"] < -tolerance and not zero_negative:
                    cutoff_negative = True
                if uncensored:
                    uncensored_values.append(float(query["distance_m"]))
                    if first_uncensored is None:
                        first_uncensored = query
            repeated_consistent = bool(
                not uncensored_values
                or max(uncensored_values) - min(uncensored_values)
                <= repeated_tolerance
            )
            if zero_negative:
                value = float(zero["distance_m"])
                value_kind = "overlap_at_zero_cutoff"
                right_censored = False
            elif first_uncensored is not None and first_uncensored["distance_m"] >= 0.0:
                value = float(first_uncensored["distance_m"])
                value_kind = "positive_first_uncensored"
                right_censored = False
            else:
                value = float(measurement["positive_distmax_sequence_m"][-1])
                value_kind = "positive_right_censored_lower_bound"
                right_censored = True
            record = {
                "protected_geom_id": pair[0],
                "protected_geom_name": str(group["protected_geom_name"]),
                "obstacle_geom_id": pair[1],
                "obstacle_geom_name": _name(env.sim.model, "geom", pair[1]),
                "zero_query": zero, "positive_queries": positive_queries,
                "zero_negative": zero_negative, "raw_pair_contact": raw_contact,
                "negative_without_raw_pair_contact": bool(
                    zero_negative and not raw_contact
                ),
                "raw_pair_contact_without_negative": bool(
                    raw_contact and not zero_negative
                ),
                "cutoff_induced_negative": cutoff_negative,
                "repeated_uncensored_consistent": repeated_consistent,
                "adaptive_value_m": value, "value_kind": value_kind,
                "right_censored": right_censored,
            }
            pair_records.append(record)
            group_pairs.append(record)
        closest = min(group_pairs, key=lambda item: item["adaptive_value_m"])
        group_records.append({
            "group_index": int(group["group_index"]),
            "body_name": str(group["body_name"]),
            "protected_geom_name": str(group["protected_geom_name"]),
            "adaptive_margin_m": float(closest["adaptive_value_m"]),
            "closest_pair": closest,
            "right_censored": bool(closest["right_censored"]),
        })
    unregistered_raw = sorted(raw_pairs - registered_pairs)
    values = [float(item["adaptive_margin_m"]) for item in group_records]
    return {
        "group_records": group_records, "group_margins_m": values,
        "global_minimum_margin_m": float(min(values)),
        "pair_records": pair_records,
        "raw_protected_contact_count": len(raw_records),
        "raw_protected_contact_records": raw_records,
        "unregistered_raw_pair_count": len(unregistered_raw),
        "negative_without_raw_pair_contact_count": sum(
            item["negative_without_raw_pair_contact"] for item in pair_records
        ),
        "raw_pair_contact_without_negative_count": sum(
            item["raw_pair_contact_without_negative"] for item in pair_records
        ),
        "cutoff_induced_negative_count": sum(
            item["cutoff_induced_negative"] for item in pair_records
        ),
        "repeated_uncensored_inconsistency_count": sum(
            not item["repeated_uncensored_consistent"] for item in pair_records
        ),
        "all_queries_finite": all(
            item["zero_query"]["finite"]
            and all(query["finite"] for query in item["positive_queries"])
            for item in pair_records
        ),
    }


class AdaptiveNativeSubstepProbe(SubstepEightConstraintProbe):
    """Capture adaptive physical margins at every cloned OSC substep."""

    def __init__(
        self, *args: Any, native_inventory: Mapping[str, Any],
        adaptive_measurement: Mapping[str, Any], **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.native_inventory = dict(native_inventory)
        self.adaptive_measurement = dict(adaptive_measurement)

    def _capture(self, env: Any, substep_index: int, phase: str) -> dict[str, Any]:
        output = super()._capture(env, substep_index, phase)
        output["adaptive_native"] = measure_adaptive_groups(
            env, self.native_inventory, self.adaptive_measurement
        )
        return output

    def _transition_from_current_probe(
        self, action: Sequence[float], *, synchronization: Mapping[str, Any]
    ) -> dict[str, Any]:
        import numpy as np

        output = super()._transition_from_current_probe(
            action, synchronization=synchronization
        )
        trace = [item["adaptive_native"] for item in output["substeps"]]
        margins = np.asarray(
            [item["group_margins_m"] for item in trace], dtype=np.float64
        )
        output["adaptive_native"] = {
            "initial_group_margins_m": margins[0].tolist(),
            "minimum_substep_group_margins_m": np.min(margins, axis=0).tolist(),
            "initial_global_minimum_margin_m": float(np.min(margins[0])),
            "minimum_substep_global_margin_m": float(np.min(margins)),
            "raw_protected_contact_count": sum(
                item["raw_protected_contact_count"] for item in trace
            ),
            "negative_without_raw_pair_contact_count": sum(
                item["negative_without_raw_pair_contact_count"] for item in trace
            ),
            "raw_pair_contact_without_negative_count": sum(
                item["raw_pair_contact_without_negative_count"] for item in trace
            ),
            "cutoff_induced_negative_count": sum(
                item["cutoff_induced_negative_count"] for item in trace
            ),
            "repeated_uncensored_inconsistency_count": sum(
                item["repeated_uncensored_inconsistency_count"] for item in trace
            ),
            "unregistered_raw_pair_count": sum(
                item["unregistered_raw_pair_count"] for item in trace
            ),
            "all_queries_finite": all(item["all_queries_finite"] for item in trace),
        }
        return output


def result_payload(value: Mapping[str, Any]) -> str:
    return _hash_without(value, "result_payload_sha256")
