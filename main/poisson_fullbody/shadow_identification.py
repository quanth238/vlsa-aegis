"""Read-only static-Poisson identification over an unchanged OSC rollout.

The observer in this module never writes simulator state and never constructs
or applies a control command.  It evaluates the registered link-5/6 surface
samples on a clone-forwarded MuJoCo state, records typed field-query failures,
and computes an *observed* CBF directional diagnostic from the simulator's
instantaneous arm velocity.  The diagnostic is not a hypothetical QP result.

The selected-obstacle field is static.  Translation, rotation, and collision-
surface drift are checked before every query.  Once any registered threshold
is crossed, the observer permanently stops querying the field while still
emitting one trace row for every physics callback.  This lets the surrounding
runner complete exact historical replay and contact measurement after the
static model has become inadmissible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from main.poisson_fullbody.geometry import OrientedBox
from main.poisson_fullbody.jacobians import point_translational_jacobian
from main.poisson_fullbody.measurement import clone_forwarded_state
from main.poisson_fullbody.robot_samples import BodySample


class ShadowIdentificationError(RuntimeError):
    """Raised when a read-only shadow trace is incomplete or inconsistent."""


def _modules() -> Tuple[Any, Any]:
    try:
        import mujoco
        import numpy as np
    except ImportError as error:  # pragma: no cover - allocation dependency
        raise RuntimeError("MuJoCo and NumPy are required for shadow identification") from error
    return mujoco, np


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _raw_model_data(sim: Any) -> Tuple[Any, Any]:
    if not hasattr(sim, "model") or not hasattr(sim, "data"):
        raise TypeError("sim must expose model and data")
    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    return model, data


def _rotation_angle(reference: Any, current: Any, np: Any) -> float:
    relative = reference.T @ current
    cosine = (float(np.trace(relative)) - 1.0) * 0.5
    return float(math.acos(max(-1.0, min(1.0, cosine))))


def _reason_text(query: Any) -> str:
    reason = getattr(query, "reason", None)
    value = getattr(reason, "value", reason)
    return "unknown_invalid_query" if value is None else str(value)


def _sample_record(sample: BodySample) -> Dict[str, Any]:
    return {
        "sample_id": int(sample.sample_id),
        "body_id": int(sample.body_id),
        "body_name": str(sample.body_name),
        "geom_id": int(sample.geom_id),
        "geom_name": str(sample.geom_name),
    }


@dataclass(frozen=True)
class StaticDriftThresholds:
    translation_m: float
    rotation_rad: float
    surface_m: float

    def __post_init__(self) -> None:
        for value, label in (
            (self.translation_m, "translation_m"),
            (self.rotation_rad, "rotation_rad"),
            (self.surface_m, "surface_m"),
        ):
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError("%s must be finite and nonnegative" % label)


def robot_root_body_ids(
    robot_body_ids: Iterable[int], body_parent_ids: Sequence[int]
) -> Tuple[int, ...]:
    """Return the minimal authoritative roots of a registered robot-body set."""

    bodies = tuple(sorted(set(int(value) for value in robot_body_ids)))
    if not bodies or any(value <= 0 or value >= len(body_parent_ids) for value in bodies):
        raise ValueError("robot body IDs must be nonempty, unique, non-world model IDs")
    body_set = set(bodies)
    roots = tuple(value for value in bodies if int(body_parent_ids[value]) not in body_set)
    if not roots:
        raise ValueError("robot body set has no root outside itself")
    return roots


class StaticPoissonShadowObserver:
    """Evaluate a certified field on every callback without mutating physics."""

    def __init__(
        self,
        *,
        field: Any,
        samples: Iterable[BodySample],
        settled_obstacle_boxes: Iterable[OrientedBox],
        arm_dof_indices: Sequence[int],
        alpha_gain_per_s: float,
        physics_timestep_s: float,
        drift_thresholds: StaticDriftThresholds,
        physics_substeps_per_high_level_action: int = 25,
        snapshot_provider: Optional[Callable[[Any], Tuple[Any, Any]]] = None,
        jacobian_provider: Optional[Callable[..., Tuple[Any, Any]]] = None,
    ) -> None:
        _, np = _modules()
        records = tuple(samples)
        boxes = tuple(settled_obstacle_boxes)
        dofs = tuple(int(value) for value in arm_dof_indices)
        if not records or any(not isinstance(value, BodySample) for value in records):
            raise ValueError("samples must contain at least one BodySample")
        if tuple(sample.sample_id for sample in records) != tuple(range(len(records))):
            raise ValueError("shadow sample IDs must be contiguous and deterministic")
        if not boxes or any(not isinstance(value, OrientedBox) for value in boxes):
            raise ValueError("settled_obstacle_boxes must contain OrientedBox records")
        geom_ids = tuple(int(box.geom_id) for box in boxes)
        if len(geom_ids) != len(set(geom_ids)):
            raise ValueError("settled obstacle geom IDs must be unique")
        if len(dofs) != 7 or len(set(dofs)) != 7 or any(value < 0 for value in dofs):
            raise ValueError("arm_dof_indices must contain seven unique nonnegative IDs")
        alpha = float(alpha_gain_per_s)
        timestep = float(physics_timestep_s)
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha_gain_per_s must be finite and positive")
        if not math.isfinite(timestep) or timestep <= 0.0:
            raise ValueError("physics_timestep_s must be finite and positive")
        if (
            isinstance(physics_substeps_per_high_level_action, bool)
            or int(physics_substeps_per_high_level_action)
            != physics_substeps_per_high_level_action
            or int(physics_substeps_per_high_level_action) <= 0
        ):
            raise ValueError("physics_substeps_per_high_level_action must be positive")
        self._field = field
        self._samples = records
        self._boxes = boxes
        self._settled_vertices = {
            int(box.geom_id): np.asarray(box.vertices(), dtype=np.float64)
            for box in boxes
        }
        self._dofs = dofs
        self._alpha = alpha
        self._timestep = timestep
        self._drift_thresholds = drift_thresholds
        self._substeps_per_high = int(physics_substeps_per_high_level_action)
        self._snapshot_provider = snapshot_provider
        self._forwarded_clone = None
        self._snapshot_model = None
        self._jacobian_provider = jacobian_provider or point_translational_jacobian
        self._trace = []
        self._static_valid = True
        self._first_invalidation: Optional[Dict[str, Any]] = None

    @property
    def trace(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(self._trace)

    @property
    def static_field_valid(self) -> bool:
        return self._static_valid

    def _snapshot(self, sim: Any) -> Tuple[Any, Any]:
        if self._snapshot_provider is not None:
            return self._snapshot_provider(sim)
        model, live_data = _raw_model_data(sim)
        if self._snapshot_model is None:
            self._snapshot_model = model
        elif model is not self._snapshot_model:
            raise ShadowIdentificationError(
                "shadow callback received a different MuJoCo model"
            )
        self._forwarded_clone = clone_forwarded_state(
            model, live_data, reusable_clone=self._forwarded_clone
        )
        return model, self._forwarded_clone

    def _drift(self, data: Any) -> Dict[str, Any]:
        _, np = _modules()
        geoms = []
        for settled in self._boxes:
            geom_id = int(settled.geom_id)
            position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
            rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
                raise ShadowIdentificationError("selected-obstacle pose is non-finite")
            translation = float(np.linalg.norm(position - settled.center))
            rotation_angle = _rotation_angle(settled.R, rotation, np)
            current = OrientedBox(
                center=position,
                R=rotation,
                half_extents=settled.half_extents,
                geom_id=geom_id,
            )
            surface = float(
                np.max(
                    np.linalg.norm(
                        np.asarray(current.vertices(), dtype=np.float64)
                        - self._settled_vertices[geom_id],
                        axis=1,
                    )
                )
            )
            reasons = []
            if translation > float(self._drift_thresholds.translation_m):
                reasons.append("translation_threshold_crossed")
            if rotation_angle > float(self._drift_thresholds.rotation_rad):
                reasons.append("rotation_threshold_crossed")
            if surface > float(self._drift_thresholds.surface_m):
                reasons.append("surface_threshold_crossed")
            geoms.append(
                {
                    "geom_id": geom_id,
                    "translation_m": translation,
                    "rotation_rad": rotation_angle,
                    "maximum_surface_point_displacement_m": surface,
                    "threshold_crossing_reasons": reasons,
                }
            )
        return {
            "maximum_translation_m": max(row["translation_m"] for row in geoms),
            "maximum_rotation_rad": max(row["rotation_rad"] for row in geoms),
            "maximum_surface_point_displacement_m": max(
                row["maximum_surface_point_displacement_m"] for row in geoms
            ),
            "geoms": geoms,
        }

    def _invalidate(
        self,
        *,
        observation_index: int,
        high_level_index: int,
        physics_substep_index: int,
        drift: Mapping[str, Any],
    ) -> None:
        if not self._static_valid:
            return
        reasons = sorted(
            {
                reason
                for geom in drift["geoms"]
                for reason in geom["threshold_crossing_reasons"]
            }
        )
        if not reasons:
            return
        self._static_valid = False
        self._first_invalidation = {
            "observation_index": int(observation_index),
            "high_level_index": int(high_level_index),
            "physics_substep_index": int(physics_substep_index),
            "reason": "static_selected_obstacle_drift",
            "threshold_crossing_reasons": reasons,
            "drift": dict(drift),
            "semantics": (
                "field queries stop before evaluating the first post-state that "
                "crosses any registered static-obstacle drift threshold"
            ),
        }

    def observe(
        self,
        sim: Any,
        *,
        high_level_index: int,
        physics_substep_index: int,
    ) -> Mapping[str, Any]:
        expected = len(self._trace)
        observed = (
            int(high_level_index) * self._substeps_per_high
            + int(physics_substep_index)
        )
        if observed != expected:
            raise ShadowIdentificationError(
                "shadow callback gap/duplicate: expected %d, got %d"
                % (expected, observed)
            )
        model, data = self._snapshot(sim)
        _, np = _modules()
        drift = self._drift(data)
        self._invalidate(
            observation_index=observed,
            high_level_index=int(high_level_index),
            physics_substep_index=int(physics_substep_index),
            drift=drift,
        )
        base = {
            "observation_index": observed,
            "high_level_index": int(high_level_index),
            "physics_substep_index": int(physics_substep_index),
            "time_from_first_callback_s": float(observed * self._timestep),
            "time_from_settled_state_s": float((observed + 1) * self._timestep),
            "static_field_admissible_for_this_query": bool(self._static_valid),
            "drift": drift,
        }
        if not self._static_valid:
            row = dict(base)
            row.update(
                {
                    "field_query_attempted": False,
                    "skip_reason": "static_selected_obstacle_drift_invalidated_field",
                    "sample_count": len(self._samples),
                    "valid_query_count": 0,
                    "invalid_query_count": 0,
                    "observed_cbf_lhs_evaluation_count": 0,
                    "invalid_reason_counts": {},
                    "minimum_h_m2": None,
                    "minimum_h_sample": None,
                    "minimum_observed_cbf_lhs_m2_per_s": None,
                    "minimum_observed_cbf_lhs_sample": None,
                    "per_geom": [],
                }
            )
            self._trace.append(row)
            return row

        qvel = np.asarray(data.qvel, dtype=np.float64)
        if any(value >= qvel.shape[0] for value in self._dofs):
            raise ShadowIdentificationError("arm DOF index is outside data.qvel")
        arm_velocity = qvel[np.asarray(self._dofs, dtype=np.int64)]
        if arm_velocity.shape != (7,) or not np.all(np.isfinite(arm_velocity)):
            raise ShadowIdentificationError("instantaneous arm velocity is invalid")

        by_geom: Dict[int, Dict[str, Any]] = {}
        all_invalid = Counter()
        for sample in self._samples:
            point = sample.world_point(data)
            query = self._field.query(point)
            group = by_geom.setdefault(
                int(sample.geom_id),
                {
                    "geom_id": int(sample.geom_id),
                    "geom_name": str(sample.geom_name),
                    "body_id": int(sample.body_id),
                    "body_name": str(sample.body_name),
                    "sample_count": 0,
                    "valid_query_count": 0,
                    "invalid_query_count": 0,
                    "observed_cbf_lhs_evaluation_count": 0,
                    "invalid_reason_counts": Counter(),
                    "first_invalid_sample": None,
                    "first_invalid_sample_by_reason": {},
                    "minimum_h_sample": None,
                    "minimum_observed_cbf_lhs_sample": None,
                },
            )
            group["sample_count"] += 1
            if not bool(getattr(query, "valid", False)):
                reason = _reason_text(query)
                group["invalid_query_count"] += 1
                group["invalid_reason_counts"][reason] += 1
                all_invalid[reason] += 1
                if group["first_invalid_sample"] is None:
                    group["first_invalid_sample"] = {
                        **_sample_record(sample),
                        "reason": reason,
                        "point_world_m": [float(value) for value in point],
                    }
                if reason not in group["first_invalid_sample_by_reason"]:
                    group["first_invalid_sample_by_reason"][reason] = {
                        **_sample_record(sample),
                        "reason": reason,
                        "point_world_m": [float(value) for value in point],
                    }
                continue
            value = getattr(query, "value", None)
            gradient = getattr(query, "gradient", None)
            if value is None or gradient is None:
                raise ShadowIdentificationError("valid field query omitted h or gradient")
            value = float(value)
            gradient_array = np.asarray(gradient, dtype=np.float64)
            if (
                not math.isfinite(value)
                or gradient_array.shape != (3,)
                or not np.all(np.isfinite(gradient_array))
            ):
                raise ShadowIdentificationError("valid field query is non-finite")
            group["valid_query_count"] += 1
            jacobian_point, jacobian = self._jacobian_provider(
                model, data, sample, self._dofs
            )
            jacobian_point = np.asarray(jacobian_point, dtype=np.float64)
            jacobian = np.asarray(jacobian, dtype=np.float64)
            if jacobian_point.shape != (3,) or jacobian.shape != (3, 7):
                raise ShadowIdentificationError("protected-sample Jacobian shape is invalid")
            if not np.all(np.isfinite(jacobian)) or not np.all(
                np.isfinite(jacobian_point)
            ):
                raise ShadowIdentificationError("protected-sample Jacobian is non-finite")
            if float(np.max(np.abs(jacobian_point - point))) > 1.0e-12:
                raise ShadowIdentificationError(
                    "field query and Jacobian use different world points"
                )
            directional = float(gradient_array @ jacobian @ arm_velocity)
            lhs = float(directional + self._alpha * value)
            if not math.isfinite(directional) or not math.isfinite(lhs):
                raise ShadowIdentificationError("observed CBF diagnostic is non-finite")
            group["observed_cbf_lhs_evaluation_count"] += 1
            diagnostic = {
                **_sample_record(sample),
                "point_world_m": [float(item) for item in point],
                "h_m2": value,
                "gradient_world_m": [float(item) for item in gradient_array],
                "gradient_norm_m": float(np.linalg.norm(gradient_array)),
                "observed_arm_qvel_rad_s": [float(item) for item in arm_velocity],
                "observed_grad_h_J_qdot_m2_per_s": directional,
                "alpha_h_m2_per_s": float(self._alpha * value),
                "observed_cbf_lhs_m2_per_s": lhs,
                "observed_cbf_diagnostic_semantics": (
                    "post-integration instantaneous MuJoCo qvel at this protected "
                    "surface sample; not a QP nominal"
                ),
            }
            previous_h = group["minimum_h_sample"]
            if previous_h is None or value < float(previous_h["h_m2"]):
                group["minimum_h_sample"] = diagnostic
            previous_lhs = group["minimum_observed_cbf_lhs_sample"]
            if previous_lhs is None or lhs < float(
                previous_lhs["observed_cbf_lhs_m2_per_s"]
            ):
                group["minimum_observed_cbf_lhs_sample"] = diagnostic

        per_geom = []
        for geom_id in sorted(by_geom):
            group = by_geom[geom_id]
            if int(group["observed_cbf_lhs_evaluation_count"]) != int(
                group["valid_query_count"]
            ):
                raise ShadowIdentificationError(
                    "not every valid field query received a CBF residual evaluation"
                )
            group["invalid_reason_counts"] = dict(
                sorted(group["invalid_reason_counts"].items())
            )
            group["first_invalid_sample_by_reason"] = dict(
                sorted(group["first_invalid_sample_by_reason"].items())
            )
            per_geom.append(group)

        minimum_h_samples = [
            row["minimum_h_sample"]
            for row in per_geom
            if row["minimum_h_sample"] is not None
        ]
        minimum_lhs_samples = [
            row["minimum_observed_cbf_lhs_sample"]
            for row in per_geom
            if row["minimum_observed_cbf_lhs_sample"] is not None
        ]
        overall_h = (
            min(minimum_h_samples, key=lambda row: float(row["h_m2"]))
            if minimum_h_samples
            else None
        )
        overall_lhs = (
            min(
                minimum_lhs_samples,
                key=lambda row: float(row["observed_cbf_lhs_m2_per_s"]),
            )
            if minimum_lhs_samples
            else None
        )
        row = dict(base)
        row.update(
            {
                "field_query_attempted": True,
                "sample_count": len(self._samples),
                "valid_query_count": sum(
                    int(group["valid_query_count"]) for group in per_geom
                ),
                "invalid_query_count": sum(
                    int(group["invalid_query_count"]) for group in per_geom
                ),
                "observed_cbf_lhs_evaluation_count": sum(
                    int(group["observed_cbf_lhs_evaluation_count"])
                    for group in per_geom
                ),
                "invalid_reason_counts": dict(sorted(all_invalid.items())),
                "minimum_h_m2": (
                    float(overall_h["h_m2"]) if overall_h is not None else None
                ),
                "minimum_h_sample": (
                    dict(overall_h) if overall_h is not None else None
                ),
                "minimum_observed_cbf_lhs_m2_per_s": (
                    float(overall_lhs["observed_cbf_lhs_m2_per_s"])
                    if overall_lhs is not None
                    else None
                ),
                "minimum_observed_cbf_lhs_sample": (
                    dict(overall_lhs) if overall_lhs is not None else None
                ),
                "per_geom": per_geom,
            }
        )
        self._trace.append(row)
        return row

    @staticmethod
    def _first_signal(
        trace: Sequence[Mapping[str, Any]],
        *,
        geom_id: Optional[int],
        kind: str,
    ) -> Optional[Dict[str, Any]]:
        for row in trace:
            if not row.get("field_query_attempted"):
                continue
            groups = row.get("per_geom", [])
            if geom_id is not None:
                groups = [
                    group
                    for group in groups
                    if int(group["geom_id"]) == int(geom_id)
                ]
            for group in groups:
                evidence: Optional[Mapping[str, Any]] = None
                if kind == "invalid_cell" and int(
                    group["invalid_reason_counts"].get("invalid_cell", 0)
                ) > 0:
                    evidence = group.get("first_invalid_sample_by_reason", {}).get(
                        "invalid_cell"
                    )
                elif kind == "any_invalid" and int(group["invalid_query_count"]) > 0:
                    evidence = group.get("first_invalid_sample")
                elif kind == "cbf_lhs_negative":
                    minimum_lhs = group.get("minimum_observed_cbf_lhs_sample")
                    if (
                        minimum_lhs is not None
                        and float(minimum_lhs["observed_cbf_lhs_m2_per_s"]) < 0.0
                    ):
                        evidence = minimum_lhs
                if evidence is not None:
                    return {
                        "observation_index": int(row["observation_index"]),
                        "high_level_index": int(row["high_level_index"]),
                        "physics_substep_index": int(row["physics_substep_index"]),
                        "signal_kind": kind,
                        "geom_id": int(group["geom_id"]),
                        "geom_name": str(group["geom_name"]),
                        "body_id": int(group["body_id"]),
                        "body_name": str(group["body_name"]),
                        "evidence": dict(evidence),
                    }
        return None

    def result(
        self,
        *,
        expected_callback_count: int,
        first_link56_contact: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if len(self._trace) != int(expected_callback_count):
            raise ShadowIdentificationError(
                "shadow trace is incomplete: expected %d callbacks, observed %d"
                % (int(expected_callback_count), len(self._trace))
            )
        contact_observation: Optional[int] = None
        contact_geom: Optional[int] = None
        contact_phase: Optional[str] = None
        if first_link56_contact is not None:
            raw_observation = first_link56_contact.get("observation_index")
            if raw_observation is not None:
                contact_observation = int(raw_observation)
                contact_phase = str(first_link56_contact.get("source_phase", ""))
                if contact_phase not in {
                    "live_solver_phase_preintegration_geometry",
                    "post_integration_recomputed",
                }:
                    raise ShadowIdentificationError(
                        "rollout contact lacks an authoritative temporal phase"
                    )
            contact_geom = int(first_link56_contact["robot_geom_id"])

        signals = {
            "first_any_fail_closed_query": self._first_signal(
                self._trace, geom_id=None, kind="any_invalid"
            ),
            "first_invalid_cell_query": self._first_signal(
                self._trace, geom_id=None, kind="invalid_cell"
            ),
            "first_observed_minimum_cbf_lhs_negative": self._first_signal(
                self._trace, geom_id=None, kind="cbf_lhs_negative"
            ),
            "first_any_fail_closed_query_on_contact_geom": (
                self._first_signal(
                    self._trace, geom_id=contact_geom, kind="any_invalid"
                )
                if contact_geom is not None
                else None
            ),
            "first_invalid_cell_query_on_contact_geom": (
                self._first_signal(
                    self._trace, geom_id=contact_geom, kind="invalid_cell"
                )
                if contact_geom is not None
                else None
            ),
            "first_observed_minimum_cbf_lhs_negative_on_contact_geom": (
                self._first_signal(
                    self._trace,
                    geom_id=contact_geom,
                    kind="cbf_lhs_negative",
                )
                if contact_geom is not None
                else None
            ),
        }
        candidates = [
            signals["first_any_fail_closed_query_on_contact_geom"],
            signals[
                "first_observed_minimum_cbf_lhs_negative_on_contact_geom"
            ],
        ]
        candidates = [value for value in candidates if value is not None]
        primary = (
            min(candidates, key=lambda value: int(value["observation_index"]))
            if candidates
            else None
        )
        if first_link56_contact is None:
            assessment = "no_link56_contact_outcome"
            lead_substeps = None
        elif contact_observation is None:
            assessment = "link56_contact_already_present_after_settling"
            lead_substeps = None
        elif primary is None:
            assessment = "no_registered_precontact_warning_on_contact_geom"
            lead_substeps = None
        else:
            lead_substeps = contact_observation - int(primary["observation_index"])
            if lead_substeps > 0:
                assessment = "registered_warning_preceded_link56_contact"
            elif (
                lead_substeps == 0
                and contact_phase == "post_integration_recomputed"
            ):
                assessment = "registered_warning_coincident_with_link56_contact"
            else:
                assessment = "registered_warning_followed_link56_contact"
        valid_h = [
            float(row["minimum_h_m2"])
            for row in self._trace
            if row.get("minimum_h_m2") is not None
        ]
        valid_lhs = [
            float(row["minimum_observed_cbf_lhs_m2_per_s"])
            for row in self._trace
            if row.get("minimum_observed_cbf_lhs_m2_per_s") is not None
        ]
        valid_query_count = sum(
            int(row["valid_query_count"])
            for row in self._trace
            if row.get("field_query_attempted")
        )
        lhs_evaluation_count = sum(
            int(row["observed_cbf_lhs_evaluation_count"])
            for row in self._trace
            if row.get("field_query_attempted")
        )
        if lhs_evaluation_count != valid_query_count:
            raise ShadowIdentificationError(
                "aggregate CBF residual coverage differs from valid-query coverage"
            )
        trace_hash = hashlib.sha256(_canonical(self._trace)).hexdigest()
        return {
            "observed_callback_count": len(self._trace),
            "expected_callback_count": int(expected_callback_count),
            "field_query_callback_count": sum(
                bool(row["field_query_attempted"]) for row in self._trace
            ),
            "field_query_skipped_after_static_invalidation_count": sum(
                not bool(row["field_query_attempted"]) for row in self._trace
            ),
            "first_static_field_invalidation": self._first_invalidation,
            "minimum_valid_h_m2": min(valid_h) if valid_h else None,
            "minimum_observed_cbf_lhs_m2_per_s": (
                min(valid_lhs) if valid_lhs else None
            ),
            "valid_field_query_count": valid_query_count,
            "observed_cbf_lhs_evaluation_count": lhs_evaluation_count,
            "every_valid_query_has_observed_cbf_lhs": True,
            "signal_definitions": {
                "any_fail_closed_query": (
                    "any typed query invalidity; the registered active policy must "
                    "fail closed for outside-grid, invalid-cell, nondifferentiable-"
                    "face, nonregular-zero-cell, and nonfinite queries, although "
                    "not all reasons indicate obstacle proximity"
                ),
                "invalid_cell_query": (
                    "sample left the certified connected free-cell domain; this "
                    "alone does not distinguish buffered-obstacle, outer-boundary, "
                    "or disconnected-component causes"
                ),
                "observed_minimum_cbf_lhs_negative": (
                    "minimum over every valid protected sample of grad(h)^T J "
                    "qdot + alpha*h < 0, using post-integration MuJoCo qvel; "
                    "this is exhaustive for the registered sampled constraints "
                    "but is not a hypothetical QP solve"
                ),
            },
            "signals": signals,
            "contact_prediction_assessment": {
                "assessment": assessment,
                "first_link56_contact": (
                    dict(first_link56_contact)
                    if first_link56_contact is not None
                    else None
                ),
                "primary_registered_warning": primary,
                "lead_physics_substeps": lead_substeps,
                "lead_time_s": (
                    float(lead_substeps * self._timestep)
                    if lead_substeps is not None
                    else None
                ),
                "contact_source_phase": contact_phase,
                "same_callback_phase_semantics": (
                    "a live_solver_phase_preintegration_geometry contact at index N "
                    "precedes the post-integration field query at N; a "
                    "post_integration_recomputed contact at N is coincident with "
                    "that query state"
                ),
                "interpretation_limit": (
                    "one outcome-conditioned canary tests temporal identification "
                    "only; it does not establish specificity or active safety efficacy"
                ),
            },
            "hypothetical_qp": {
                "computed": False,
                "reason": (
                    "the unchanged OSC replay does not expose the registered "
                    "joint-velocity adapter nominal; fabricating a QP nominal would "
                    "not be semantically paired"
                ),
            },
            "trace_sha256": trace_hash,
            "trace": list(self._trace),
        }


__all__ = [
    "ShadowIdentificationError",
    "StaticDriftThresholds",
    "StaticPoissonShadowObserver",
    "robot_root_body_ids",
]
