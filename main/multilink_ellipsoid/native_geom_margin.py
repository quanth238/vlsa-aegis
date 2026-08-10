"""Native MuJoCo L5--L7 collision-geometry distance inventory.

This module is opt-in.  It never changes the ordinary ``env.step`` path and
keeps compiled-geometry distance (D_sim) separate from ellipsoid support gaps
(D_opt).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .oracle_affine import SubstepEightConstraintProbe, _body_lineage


CONFIG_SCHEMA = "vlsa_distal_native_geom_inventory_moka10.v1"
RESULT_SCHEMA = "vlsa_distal_native_geom_inventory_moka10_result.v1"
VALIDATION_SCHEMA = "vlsa_distal_native_geom_inventory_moka10_validation.v1"
PROTECTED_BODY_NAMES = ("robot0_link5", "robot0_link6", "robot0_link7")


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
        raise ValueError("native-geometry inventory config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope", "immutable_source",
        "population", "target_definition", "measurement", "gate", "decision",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("native-geometry inventory config keys differ")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["protocol_id"]
        != "vlsa-distal-native-geom-inventory-moka10-v1"
    ):
        raise ValueError("native-geometry inventory protocol differs")
    source_keys = {
        "source_population_manifest_sha256",
        "boundary_generalization_manifest_sha256",
        "same_task_expansion_manifest_sha256",
        "targeted_expansion_manifest_sha256",
        "expanded_dataset_file_sha256", "expanded_dataset_payload_sha256",
        "geometry_config_file_sha256", "exact_box_config_file_sha256",
    }
    if set(config["immutable_source"]) != source_keys:
        raise ValueError("native-geometry immutable-source keys differ")
    if config["population"] != {
        "expected_state_count": 85,
        "expected_split_counts": {"train": 60, "validation": 10, "test": 15},
        "expected_complete_episode_count": 17,
        "test_case_ids": [
            "vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t0-e10",
            "vlsa-t1-goal-ii-t0-e15",
        ],
    }:
        raise ValueError("native-geometry inventory population differs")
    if config["target_definition"] != {
        "protected_bodies": list(PROTECTED_BODY_NAMES),
        "grouping": "one_row_per_actual_compiled_protected_collision_geom",
        "obstacle_scope": "collision_eligible_geoms_in_active_obstacle_body_lineage",
        "group_margin": "minimum_signed_mj_geomDistance_to_registered_obstacle_geoms",
        "link_and_global_minima": "aggregates_only_not_training_targets",
        "ellipsoid_margins": "D_opt_comparison_only",
        "raw_contacts": "independent_D_sim_witness",
    }:
        raise ValueError("native-geometry target definition differs")
    measurement = config["measurement"]
    if set(measurement) != {
        "distance_api", "distance_max_m", "distance_censor_tolerance_m",
        "safe_distance_m", "contact_distance_threshold_m",
        "contact_distance_consistency_tolerance_m", "horizon_actions",
        "temporal_resolution", "include_k0_only_for_prevention_cohort",
        "primary_contact_witness_steps",
    }:
        raise ValueError("native-geometry measurement keys differ")
    if measurement["distance_api"] != "mujoco.mj_geomDistance":
        raise ValueError("native-geometry distance API differs")
    for key in (
        "distance_max_m", "distance_censor_tolerance_m",
        "contact_distance_consistency_tolerance_m",
    ):
        value = measurement[key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("measurement.%s must be finite and positive" % key)
    if float(measurement["safe_distance_m"]) != 0.0:
        raise ValueError("native-geometry discovery freezes zero safety margin")
    if float(measurement["contact_distance_threshold_m"]) != 0.0:
        raise ValueError("native-geometry contact threshold differs")
    if measurement["horizon_actions"] != 2 or measurement[
        "temporal_resolution"
    ] != "every_internal_MuJoCo_substep_in_cloned_OSC":
        raise ValueError("native-geometry horizon differs")
    if measurement["primary_contact_witness_steps"] != [187, 188]:
        raise ValueError("native-geometry contact witness steps differ")
    if config["gate"] != {
        "all_protected_geoms_have_registered_obstacle_pairs": True,
        "semantic_protected_group_inventory_consistent_across_episodes": True,
        "all_queries_finite_and_uncensored": True,
        "every_raw_contact_pair_registered": True,
        "every_raw_contact_distance_consistent": True,
        "required_primary_contact_witness_count": 1,
        "required_test_initially_native_safe_state_count": 15,
        "training_in_this_gate": False,
        "QP_in_this_gate": False,
        "closed_loop_in_this_gate": False,
    }:
        raise ValueError("native-geometry inventory gate differs")
    if config["decision"] != {
        "pass": "freeze_physical_group_inventory_then_collect_candidate_rollout_targets",
        "initially_unsafe_state": "separate_recovery_cohort_or_select_earlier_state",
        "failure": "repair_native_target_registration_before_learning",
        "closed_loop_E05_authorized": False,
    }:
        raise ValueError("native-geometry decision differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def _name(model: Any, kind: str, identifier: int) -> str:
    method = getattr(model, "%s_id2name" % kind)
    value = method(int(identifier))
    return "" if value is None else str(value)


def _obstacle_root_id(model: Any, active_obstacle_name: str) -> int:
    for name in (str(active_obstacle_name), "%s_main" % active_obstacle_name):
        try:
            return int(model.body_name2id(name))
        except (KeyError, ValueError):
            continue
    raise ValueError("active obstacle body is unavailable")


def _collision_eligible(model: Any, first: int, second: int) -> bool:
    first_type = int(model.geom_contype[first])
    first_affinity = int(model.geom_conaffinity[first])
    second_type = int(model.geom_contype[second])
    second_affinity = int(model.geom_conaffinity[second])
    return bool(
        (first_type & second_affinity) != 0
        or (second_type & first_affinity) != 0
    )


def compiled_group_inventory(env: Any, active_obstacle_name: str) -> dict[str, Any]:
    """Discover physical rows without assuming a seven-row proxy mapping."""

    model = env.sim.model
    obstacle_root = _obstacle_root_id(model, active_obstacle_name)
    protected_ids = {
        str(name): int(model.body_name2id(name)) for name in PROTECTED_BODY_NAMES
    }
    obstacle_geoms = [
        geom_id for geom_id in range(int(model.ngeom))
        if obstacle_root in _body_lineage(model, int(model.geom_bodyid[geom_id]))
    ]
    groups = []
    for body_name in PROTECTED_BODY_NAMES:
        body_id = protected_ids[body_name]
        for protected_geom in range(int(model.ngeom)):
            if int(model.geom_bodyid[protected_geom]) != body_id:
                continue
            pairs = [
                obstacle_geom for obstacle_geom in obstacle_geoms
                if _collision_eligible(model, protected_geom, obstacle_geom)
            ]
            if not pairs:
                continue
            groups.append({
                "group_index": len(groups),
                "body_name": body_name,
                "protected_geom_id": int(protected_geom),
                "protected_geom_name": _name(model, "geom", protected_geom),
                "obstacle_geom_ids": [int(item) for item in pairs],
                "obstacle_geom_names": [_name(model, "geom", item) for item in pairs],
            })
    represented = {item["body_name"] for item in groups}
    if represented != set(PROTECTED_BODY_NAMES):
        raise ValueError("one or more protected bodies lack collision-eligible geoms")
    semantic = [
        {"body_name": item["body_name"],
         "protected_geom_name": item["protected_geom_name"]}
        for item in groups
    ]
    registered_obstacle_geoms = {
        int(obstacle_geom) for group in groups
        for obstacle_geom in group["obstacle_geom_ids"]
    }
    return {
        "active_obstacle_name": str(active_obstacle_name),
        "active_obstacle_root_body_id": int(obstacle_root),
        "groups": groups,
        "semantic_protected_groups": semantic,
        "semantic_protected_group_sha256": _sha256(_canonical(semantic)),
        "obstacle_collision_geom_count": len(registered_obstacle_geoms),
    }


def _native_model_data(env: Any) -> tuple[Any, Any]:
    model = getattr(env.sim.model, "_model", None)
    data = getattr(env.sim.data, "_data", None)
    if model is None or data is None:
        raise ValueError("official MuJoCo model/data handles are unavailable")
    return model, data


def measure_native_groups(
    env: Any, inventory: Mapping[str, Any], *, distance_max_m: float,
    distance_censor_tolerance_m: float, safe_distance_m: float,
    contact_distance_threshold_m: float,
    contact_distance_consistency_tolerance_m: float,
) -> dict[str, Any]:
    """Measure signed group margins and cross-check active raw contacts."""

    import mujoco
    import numpy as np

    wrapper_model = env.sim.model
    native_model, native_data = _native_model_data(env)
    pair_queries: dict[tuple[int, int], dict[str, Any]] = {}
    group_records = []
    all_uncensored = True
    for group in inventory["groups"]:
        pair_records = []
        for obstacle_geom in group["obstacle_geom_ids"]:
            fromto = np.zeros(6, dtype=np.float64)
            distance = float(mujoco.mj_geomDistance(
                native_model, native_data, int(group["protected_geom_id"]),
                int(obstacle_geom), float(distance_max_m), fromto,
            ))
            finite = bool(math.isfinite(distance) and np.all(np.isfinite(fromto)))
            uncensored = bool(
                finite
                and distance < float(distance_max_m) - float(distance_censor_tolerance_m)
            )
            all_uncensored = bool(all_uncensored and uncensored)
            record = {
                "protected_geom_id": int(group["protected_geom_id"]),
                "protected_geom_name": str(group["protected_geom_name"]),
                "obstacle_geom_id": int(obstacle_geom),
                "obstacle_geom_name": _name(wrapper_model, "geom", obstacle_geom),
                "signed_distance_m": distance,
                "margin_m": distance - float(safe_distance_m),
                "fromto_m": fromto.tolist(),
                "finite": finite, "uncensored": uncensored,
            }
            pair_records.append(record)
            pair_queries[(int(group["protected_geom_id"]), int(obstacle_geom))] = record
        if not pair_records:
            raise ValueError("native protected group has no registered pairs")
        closest = min(pair_records, key=lambda item: item["signed_distance_m"])
        group_records.append({
            "group_index": int(group["group_index"]),
            "body_name": str(group["body_name"]),
            "protected_geom_id": int(group["protected_geom_id"]),
            "protected_geom_name": str(group["protected_geom_name"]),
            "margin_m": float(closest["margin_m"]),
            "closest_pair": closest,
            "registered_pair_count": len(pair_records),
        })
    obstacle_root = int(inventory["active_obstacle_root_body_id"])
    protected_ids = {
        int(wrapper_model.body_name2id(name)): str(name)
        for name in PROTECTED_BODY_NAMES
    }
    contact_records = []
    every_registered = True
    every_consistent = True
    for contact_index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[contact_index]
        contact_distance = float(contact.dist)
        if contact_distance > float(contact_distance_threshold_m):
            continue
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = int(wrapper_model.geom_bodyid[geom1])
        body2 = int(wrapper_model.geom_bodyid[geom2])
        if body1 in protected_ids and obstacle_root in _body_lineage(wrapper_model, body2):
            protected_geom, obstacle_geom = geom1, geom2
            body_name = protected_ids[body1]
        elif body2 in protected_ids and obstacle_root in _body_lineage(wrapper_model, body1):
            protected_geom, obstacle_geom = geom2, geom1
            body_name = protected_ids[body2]
        else:
            continue
        query = pair_queries.get((protected_geom, obstacle_geom))
        registered = query is not None
        error = (
            None if query is None
            else abs(float(query["signed_distance_m"]) - contact_distance)
        )
        consistent = bool(
            registered
            and error is not None
            and error <= float(contact_distance_consistency_tolerance_m)
        )
        every_registered = bool(every_registered and registered)
        every_consistent = bool(every_consistent and consistent)
        contact_records.append({
            "contact_index": int(contact_index), "body_name": body_name,
            "protected_geom_id": protected_geom,
            "protected_geom_name": _name(wrapper_model, "geom", protected_geom),
            "obstacle_geom_id": obstacle_geom,
            "obstacle_geom_name": _name(wrapper_model, "geom", obstacle_geom),
            "contact_distance_m": contact_distance,
            "pair_registered": registered,
            "queried_distance_m": None if query is None else query["signed_distance_m"],
            "absolute_distance_error_m": error,
            "distance_consistent": consistent,
        })
    link_minima = {
        body_name: min(
            item["margin_m"] for item in group_records
            if item["body_name"] == body_name
        ) for body_name in PROTECTED_BODY_NAMES
    }
    margins = [float(item["margin_m"]) for item in group_records]
    return {
        "group_records": group_records, "group_margins_m": margins,
        "link_minimum_margins_m": link_minima,
        "global_minimum_margin_m": float(min(margins)),
        "all_queries_finite_and_uncensored": all_uncensored,
        "raw_protected_contact_count": len(contact_records),
        "raw_protected_contact_records": contact_records,
        "every_raw_contact_pair_registered": every_registered,
        "every_raw_contact_distance_consistent": every_consistent,
    }


class NativeGeomSubstepProbe(SubstepEightConstraintProbe):
    """Add native compiled-geometry distances to each cloned OSC substep."""

    def __init__(
        self, *args: Any, native_inventory: Mapping[str, Any],
        native_measurement: Mapping[str, Any], **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.native_inventory = dict(native_inventory)
        self.native_measurement = dict(native_measurement)

    def _capture(self, env: Any, substep_index: int, phase: str) -> dict[str, Any]:
        record = super()._capture(env, substep_index, phase)
        record["native_geometry"] = measure_native_groups(
            env, self.native_inventory,
            distance_max_m=float(self.native_measurement["distance_max_m"]),
            distance_censor_tolerance_m=float(
                self.native_measurement["distance_censor_tolerance_m"]
            ),
            safe_distance_m=float(self.native_measurement["safe_distance_m"]),
            contact_distance_threshold_m=float(
                self.native_measurement["contact_distance_threshold_m"]
            ),
            contact_distance_consistency_tolerance_m=float(
                self.native_measurement[
                    "contact_distance_consistency_tolerance_m"
                ]
            ),
        )
        return record

    def _transition_from_current_probe(
        self, action: Sequence[float], *, synchronization: Mapping[str, Any]
    ) -> dict[str, Any]:
        import numpy as np

        output = super()._transition_from_current_probe(
            action, synchronization=synchronization
        )
        trace = output["substeps"]
        margins = np.asarray([
            item["native_geometry"]["group_margins_m"] for item in trace
        ], dtype=np.float64)
        output["native_geometry"] = {
            "initial_group_margins_m": margins[0].tolist(),
            "minimum_substep_group_margins_m": np.min(margins, axis=0).tolist(),
            "minimum_substep_group_indexes": np.argmin(margins, axis=0).tolist(),
            "initial_global_minimum_margin_m": float(np.min(margins[0])),
            "minimum_substep_global_margin_m": float(np.min(margins)),
            "all_queries_finite_and_uncensored": all(
                item["native_geometry"]["all_queries_finite_and_uncensored"]
                for item in trace
            ),
            "every_raw_contact_pair_registered": all(
                item["native_geometry"]["every_raw_contact_pair_registered"]
                for item in trace
            ),
            "every_raw_contact_distance_consistent": all(
                item["native_geometry"]["every_raw_contact_distance_consistent"]
                for item in trace
            ),
        }
        return output


def result_payload(value: Mapping[str, Any]) -> str:
    return _hash_without(value, "result_payload_sha256")
