"""Task-independent MuJoCo contact monitor for the direct Poisson-CBF study.

The registered forbidden union is intentionally broader than the controller:

* every robot collision geom against the selected obstacle; and
* literal link-5/link-6 collision geoms against every external non-robot geom.

It is a measurement-only dependency.  It contains no controller, torque
conversion, safety filter, fallback, or task-conditioned link selection.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from main.poisson_fullbody.contracts import canonical_json_bytes, sha256_bytes
from main.poisson_fullbody.measurement import clone_forwarded_state


class RegisteredContactMonitorError(RuntimeError):
    """The registered contact scope or observation cadence is invalid."""


def _canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _mujoco_name(model: Any, object_type: Any, object_id: int, prefix: str) -> str:
    import mujoco

    value = mujoco.mj_id2name(model, object_type, int(object_id))
    return str(value) if value else "%s_%d" % (prefix, int(object_id))


def registered_contact_scope(model: Any, resolved: Any) -> Dict[str, Any]:
    """Freeze the exact union contact scope from authoritative MuJoCo IDs."""

    import mujoco

    robot = {int(value) for value in resolved.robot_geom_ids}
    robot_bodies = {int(value) for value in resolved.robot_body_ids}
    selected = {int(value) for value in resolved.obstacle_geom_ids}
    link56 = {int(value) for value in resolved.link56_geom_ids}
    all_geoms = set(range(int(model.ngeom)))
    robot_owned = {
        geom_id
        for geom_id in all_geoms
        if int(model.geom_bodyid[geom_id]) in robot_bodies
    }
    external = all_geoms - robot_owned
    if not robot or not robot_bodies or not robot_owned or not selected or not link56:
        raise RegisteredContactMonitorError(
            "registered contact scope contains an empty ID set"
        )
    if (
        not robot <= robot_owned
        or not selected <= external
        or not link56 <= robot
        or robot_owned & external
    ):
        raise RegisteredContactMonitorError(
            "registered contact scope roles overlap or differ"
        )
    identities = []
    for geom_id in sorted(all_geoms):
        body_id = int(model.geom_bodyid[geom_id])
        identities.append(
            {
                "geom_id": geom_id,
                "geom_name": _mujoco_name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom_id, "geom"
                ),
                "body_id": body_id,
                "body_name": _mujoco_name(
                    model, mujoco.mjtObj.mjOBJ_BODY, body_id, "body"
                ),
                "is_robot_geom": geom_id in robot,
                "is_robot_owned_geom": geom_id in robot_owned,
                "is_selected_obstacle_geom": geom_id in selected,
                "is_link56_geom": geom_id in link56,
                "is_external_nonrobot_geom": geom_id in external,
            }
        )
    scope = {
        "schema_version": "vlsa_poisson_registered_contact_scope.v1",
        "semantics": (
            "forbidden_union=(any_robot_geom_vs_selected_obstacle_geom)_or_"
            "(literal_link5_link6_geom_vs_any_external_nonrobot_geom)"
        ),
        "model_geom_count": int(model.ngeom),
        "robot_geom_ids": sorted(robot),
        "robot_owned_geom_ids": sorted(robot_owned),
        "selected_obstacle_geom_ids": sorted(selected),
        "link56_geom_ids": sorted(link56),
        "external_nonrobot_geom_ids": sorted(external),
        "geom_identities": identities,
    }
    scope["identity_sha256"] = _canonical_sha256(scope)
    return scope


def registered_contact_records(
    model: Any,
    data: Any,
    scope: Mapping[str, Any],
    *,
    source_phase: str,
    physical_boundary: Optional[int] = None,
    source_action_index: Optional[int] = None,
    physics_substep_index: Optional[int] = None,
    controller_update_index: Optional[int] = None,
    physics_substep_within_controller_update: Optional[int] = None,
    callback_endpoint_action_inner_substep: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Return typed records for nonpositive contacts in the registered union."""

    robot = {int(value) for value in scope["robot_geom_ids"]}
    selected = {int(value) for value in scope["selected_obstacle_geom_ids"]}
    link56 = {int(value) for value in scope["link56_geom_ids"]}
    external = {int(value) for value in scope["external_nonrobot_geom_ids"]}
    identity = {
        int(row["geom_id"]): row for row in scope["geom_identities"]
    }
    output: List[Dict[str, Any]] = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        distance = float(contact.dist)
        if distance > 0.0 or not math.isfinite(distance):
            continue
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        selected_contact = bool(
            (geom1 in robot and geom2 in selected)
            or (geom2 in robot and geom1 in selected)
        )
        link_external_contact = bool(
            (geom1 in link56 and geom2 in external)
            or (geom2 in link56 and geom1 in external)
        )
        if not (selected_contact or link_external_contact):
            continue
        robot_geom_id = geom1 if geom1 in robot else geom2
        external_geom_id = geom2 if robot_geom_id == geom1 else geom1
        categories = []
        if selected_contact:
            categories.append("any_robot_vs_selected_obstacle")
        if link_external_contact:
            categories.append("link56_vs_external_nonrobot")
        if physical_boundary is None:
            executed_transition_start_boundary = None
            observed_state_boundary = (
                0 if source_phase == "settled_forwarded" else None
            )
        else:
            executed_transition_start_boundary = int(physical_boundary)
            if source_phase == "live_solver_phase_preintegration_geometry":
                observed_state_boundary = int(physical_boundary)
            elif source_phase == "post_integration_recomputed":
                observed_state_boundary = int(physical_boundary) + 1
            else:
                raise RegisteredContactMonitorError(
                    "registered contact source phase is unknown"
                )
        record = {
            "schema_version": (
                "vlsa_poisson_registered_forbidden_contact.v2"
                if callback_endpoint_action_inner_substep is not None
                else "vlsa_poisson_registered_forbidden_contact.v1"
            ),
            "source_phase": str(source_phase),
            "contact_index": int(contact_index),
            "physical_boundary": physical_boundary,
            "executed_transition_start_boundary": (
                executed_transition_start_boundary
            ),
            "observed_state_boundary": observed_state_boundary,
            "source_action_index": source_action_index,
            "physics_substep_index": physics_substep_index,
            "geom1_id": geom1,
            "geom2_id": geom2,
            "contact_distance_m": distance,
            "robot_geom_id": int(robot_geom_id),
            "robot_geom_name": identity[robot_geom_id]["geom_name"],
            "robot_body_id": int(identity[robot_geom_id]["body_id"]),
            "robot_body_name": identity[robot_geom_id]["body_name"],
            "external_geom_id": int(external_geom_id),
            "external_geom_name": identity[external_geom_id]["geom_name"],
            "external_body_id": int(identity[external_geom_id]["body_id"]),
            "external_body_name": identity[external_geom_id]["body_name"],
            "contact_categories": categories,
            "any_robot_selected_obstacle_contact": selected_contact,
            "link56_external_nonrobot_contact": link_external_contact,
            "link56_nonselected_external_contact": bool(
                link_external_contact and external_geom_id not in selected
            ),
        }
        if callback_endpoint_action_inner_substep is not None:
            record.update(
                {
                    "controller_update_index": controller_update_index,
                    "physics_substep_within_controller_update": (
                        physics_substep_within_controller_update
                    ),
                    "callback_endpoint_action_inner_substep": [
                        int(value)
                        for value in callback_endpoint_action_inner_substep
                    ],
                }
            )
        record["record_sha256"] = _canonical_sha256(record)
        output.append(record)
    return output


class RegisteredContactMonitor:
    """Observe the registered union after every completed 2 ms transition."""

    def __init__(
        self,
        sim: Any,
        resolved: Any,
        *,
        physics_substeps_per_action: int = 25,
        start_physical_boundary: int = 0,
        controller_updates_per_action: Optional[int] = None,
        physics_substeps_per_controller_update: Optional[int] = None,
    ) -> None:
        self._model = getattr(sim.model, "_model", sim.model)
        if (
            isinstance(physics_substeps_per_action, bool)
            or not isinstance(physics_substeps_per_action, int)
            or physics_substeps_per_action <= 0
        ):
            raise RegisteredContactMonitorError(
                "registered contact-monitor cadence is invalid"
            )
        self._physics_substeps_per_action = int(physics_substeps_per_action)
        if (
            isinstance(start_physical_boundary, bool)
            or not isinstance(start_physical_boundary, int)
            or start_physical_boundary < 0
            or start_physical_boundary % self._physics_substeps_per_action != 0
        ):
            raise RegisteredContactMonitorError(
                "registered contact-monitor start boundary is invalid"
            )
        self._start_physical_boundary = int(start_physical_boundary)
        self._typed_callback_cadence = bool(
            controller_updates_per_action is not None
            or physics_substeps_per_controller_update is not None
        )
        if self._typed_callback_cadence:
            if (
                isinstance(controller_updates_per_action, bool)
                or not isinstance(controller_updates_per_action, int)
                or isinstance(physics_substeps_per_controller_update, bool)
                or not isinstance(physics_substeps_per_controller_update, int)
                or controller_updates_per_action <= 0
                or physics_substeps_per_controller_update <= 0
                or controller_updates_per_action
                * physics_substeps_per_controller_update
                != self._physics_substeps_per_action
            ):
                raise RegisteredContactMonitorError(
                    "registered contact-monitor typed cadence is invalid"
                )
            self._controller_updates_per_action = int(
                controller_updates_per_action
            )
            self._physics_substeps_per_controller_update = int(
                physics_substeps_per_controller_update
            )
        else:
            self._controller_updates_per_action = None
            self._physics_substeps_per_controller_update = None
        data = getattr(sim.data, "_data", sim.data)
        self.scope = registered_contact_scope(self._model, resolved)
        settled = clone_forwarded_state(self._model, data)
        self._settled_records = registered_contact_records(
            self._model,
            settled,
            self.scope,
            source_phase="settled_forwarded",
        )
        self._rollout_records: List[Dict[str, Any]] = []
        self._observations = 0
        self._last_snapshot: Dict[str, Any] = {
            "literal_contact": False,
            "contacts": [],
            "contact_categories": [],
        }

    @property
    def settled_contact(self) -> bool:
        return bool(self._settled_records)

    @property
    def last_snapshot(self) -> Mapping[str, Any]:
        return self._last_snapshot

    def observe_post_integration(
        self,
        sim: Any,
        *,
        source_action_index: int,
        physics_substep_index: int,
    ) -> Mapping[str, Any]:
        expected = (
            source_action_index * self._physics_substeps_per_action
            + physics_substep_index
        )
        if expected != self._start_physical_boundary + self._observations:
            raise RegisteredContactMonitorError(
                "registered contact monitor gap/duplicate at physical boundary %d"
                % expected
            )
        data = getattr(sim.data, "_data", sim.data)
        common = {
            "physical_boundary": expected,
            "source_action_index": source_action_index,
            "physics_substep_index": physics_substep_index,
        }
        if self._typed_callback_cadence:
            inner = int(physics_substep_index) // int(
                self._physics_substeps_per_controller_update
            )
            local_substep = int(physics_substep_index) % int(
                self._physics_substeps_per_controller_update
            )
            common.update(
                {
                    "controller_update_index": inner,
                    "physics_substep_within_controller_update": local_substep,
                    "callback_endpoint_action_inner_substep": [
                        int(source_action_index),
                        inner,
                        local_substep,
                    ],
                }
            )
        records = registered_contact_records(
            self._model,
            data,
            self.scope,
            source_phase="live_solver_phase_preintegration_geometry",
            **common,
        )
        forwarded = clone_forwarded_state(self._model, data)
        records.extend(
            registered_contact_records(
                self._model,
                forwarded,
                self.scope,
                source_phase="post_integration_recomputed",
                **common,
            )
        )
        self._rollout_records.extend(records)
        categories = sorted(
            {
                category
                for record in records
                for category in record["contact_categories"]
            }
        )
        self._last_snapshot = {
            "schema_version": (
                "vlsa_poisson_registered_contact_substep.v2"
                if self._typed_callback_cadence
                else "vlsa_poisson_registered_contact_substep.v1"
            ),
            "physical_boundary": expected,
            "executed_transition_start_boundary": expected,
            "observed_state_boundary": expected + 1,
            "source_action_index": source_action_index,
            "physics_substep_index": physics_substep_index,
            "literal_contact": bool(records),
            "contact_categories": categories,
            "contacts": records,
        }
        if self._typed_callback_cadence:
            self._last_snapshot.update(
                {
                    "controller_update_index": common[
                        "controller_update_index"
                    ],
                    "physics_substep_within_controller_update": common[
                        "physics_substep_within_controller_update"
                    ],
                    "callback_endpoint_action_inner_substep": common[
                        "callback_endpoint_action_inner_substep"
                    ],
                }
            )
        self._observations += 1
        return self._last_snapshot

    def result(self) -> Dict[str, Any]:
        records = list(self._settled_records) + list(self._rollout_records)
        rollout_categories = sorted(
            {
                category
                for record in self._rollout_records
                for category in record["contact_categories"]
            }
        )
        result = {
            "schema_version": (
                "vlsa_poisson_registered_contact_measurement.v2"
                if self._typed_callback_cadence
                else "vlsa_poisson_registered_contact_measurement.v1"
            ),
            "scope_identity_sha256": self.scope["identity_sha256"],
            "scope": self.scope,
            "observed_physics_substeps": self._observations,
            "settled_forbidden_contact": bool(self._settled_records),
            "settled_contact_records": list(self._settled_records),
            "rollout_forbidden_contact": bool(self._rollout_records),
            "rollout_contact_categories": rollout_categories,
            "rollout_contact_records": list(self._rollout_records),
            "any_registered_forbidden_contact": bool(records),
            "any_robot_selected_obstacle_contact": any(
                record["any_robot_selected_obstacle_contact"]
                for record in records
            ),
            "any_link56_external_nonrobot_contact": any(
                record["link56_external_nonrobot_contact"]
                for record in records
            ),
            "any_link56_nonselected_external_contact": any(
                record["link56_nonselected_external_contact"]
                for record in records
            ),
        }
        if self._typed_callback_cadence:
            result["callback_cadence"] = {
                "controller_updates_per_action": (
                    self._controller_updates_per_action
                ),
                "physics_substeps_per_controller_update": (
                    self._physics_substeps_per_controller_update
                ),
                "physics_substeps_per_action": self._physics_substeps_per_action,
            }
            if self._start_physical_boundary:
                result["callback_cadence"]["start_physical_boundary"] = (
                    self._start_physical_boundary
                )
        return result
