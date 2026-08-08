"""Exact cloned-OSC candidate search for the primary distal-link case.

This is intentionally a small oracle / heuristic experiment.  It keeps the
released AEGIS command as the nominal action, checks the accepted seven-part
L5--L7 geometry plus the unchanged released EE proxy after a complete cloned
``env.step``, and searches a finite Cartesian candidate set only when the
nominal next state misses a registered clearance target.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from .barrier import support_gap
from .qp import MultiConstraintQp
from .rollout import ClonedSimulatorStepProbe, _dynamic_state_vector
from .shadow import (
    DISTAL_SLABBED_SCHEMA,
    MultilinkEllipsoidShadow,
    _numpy,
    _released_aegis_end_effector_ellipsoid,
)


SITL_CANDIDATE_SCHEMA = "vlsa_distal_sitl_candidate_e05.v1"
SITL_CANDIDATE_STEP_SCHEMA = "vlsa_distal_sitl_candidate_step_e05.v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_sitl_candidate_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("SITL candidate config is invalid JSON") from error
    required = {
        "schema_version",
        "protocol_id",
        "case_ids",
        "claim_scope",
        "nominal_action_source",
        "protected_geometry",
        "finite_difference",
        "optimizer",
        "candidate_search",
        "verification",
        "success_definition",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("SITL candidate config keys differ")
    if config["schema_version"] != SITL_CANDIDATE_SCHEMA:
        raise ValueError("SITL candidate schema differs")
    if config["case_ids"] != ["vlsa-t1-goal-ii-t0-e05"]:
        raise ValueError("SITL candidate must select only the primary case")
    if config["nominal_action_source"] not in {
        "immutable_successful_released_aegis_env_step_input",
        "live_pi05_libero_then_released_aegis_ee_qp_replanned_every_five_steps",
        "immutable_released_aegis_until_first_sitl_intervention_then_"
        "live_pi05_libero_recovery_with_released_aegis_ee_qp",
    }:
        raise ValueError("SITL nominal action source differs")
    geometry = config["protected_geometry"]
    expected_geometry = {
        "distal_activation_clearance_m": geometry.get(
            "distal_activation_clearance_m"
        ),
        "distal_clearance_target_m": geometry.get("distal_clearance_target_m"),
        "distal_constraint_count": 7,
        "end_effector_constraint_count": 1,
        "end_effector_target": (
            "do_not_worsen_exact_next_clearance_of_released_aegis_nominal"
        ),
        "source": (
            "accepted_v4_seven_slab_bounds_plus_unchanged_released_aegis_ee_proxy"
        ),
    }
    if geometry != expected_geometry:
        raise ValueError("SITL protected geometry contract differs")
    activation_clearance = geometry["distal_activation_clearance_m"]
    hard_clearance = geometry["distal_clearance_target_m"]
    if activation_clearance != 0.015 or hard_clearance not in {0.001, 0.01}:
        raise ValueError("SITL protected geometry margins differ")
    finite_difference = config["finite_difference"]
    if finite_difference != {
        "action_dimensions": [0, 1, 2],
        "perturbation_action": 0.1,
        "scheme": "clipped_central_difference",
    }:
        raise ValueError("SITL finite-difference contract differs")
    optimizer = config["optimizer"]
    if set(optimizer) != {
        "action_limit",
        "bound_tolerance_action",
        "eps_abs",
        "eps_rel",
        "max_iter",
        "residual_tolerance",
    }:
        raise ValueError("SITL optimizer keys differ")
    for key in (
        "action_limit",
        "bound_tolerance_action",
        "eps_abs",
        "eps_rel",
        "residual_tolerance",
    ):
        value = optimizer[key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("optimizer.%s must be finite and positive" % key)
    if not isinstance(optimizer["max_iter"], int) or optimizer["max_iter"] < 1:
        raise ValueError("optimizer.max_iter must be a positive integer")
    search = config["candidate_search"]
    selection_objective = search.get("selection_objective")
    search_without_objective = dict(search)
    search_without_objective.pop("selection_objective", None)
    if search_without_objective != {
        "correction_extrapolation_factors": [1.0, 1.5, 2.0, 3.0],
        "include_global_lattice": True,
        "include_reverse_nominal": True,
        "include_stop": True,
        "lattice_values": [-1.0, 0.0, 1.0],
        "local_offset_magnitudes": [0.25, 0.5],
    } or selection_objective not in {
        (
            "lexicographic_maximum_minimum_distal_clearance_then_"
            "minimum_nominal_deviation"
        ),
        (
            "lexicographic_minimum_nominal_deviation_then_"
            "maximum_minimum_distal_clearance"
        ),
    }:
        raise ValueError("SITL candidate search contract differs")
    verification = config["verification"]
    if verification != {
        "clearance_tolerance_m": 1.0e-6,
        "clone_state_tolerance": 1.0e-10,
        "repeat_nominal_on_first_step": True,
    }:
        raise ValueError("SITL verification contract differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def finite_difference_rows(
    plus_clearances: Any,
    minus_clearances: Any,
    plus_xyz: Any,
    minus_xyz: Any,
) -> Any:
    """Return d h_i(next) / d XYZ for any fixed number of constraints."""

    np = _numpy()
    plus_h = np.asarray(plus_clearances, dtype=np.float64)
    minus_h = np.asarray(minus_clearances, dtype=np.float64)
    plus = np.asarray(plus_xyz, dtype=np.float64)
    minus = np.asarray(minus_xyz, dtype=np.float64)
    if plus_h.ndim != 2 or plus_h.shape != minus_h.shape or plus_h.shape[0] != 3:
        raise ValueError("SITL clearance probes must have shape (3, constraints)")
    if plus.shape != (3, 3) or minus.shape != (3, 3):
        raise ValueError("SITL XYZ probes must have shape (3, 3)")
    if not all(np.all(np.isfinite(value)) for value in (plus_h, minus_h, plus, minus)):
        raise ValueError("SITL finite-difference probes must be finite")
    rows = np.empty((plus_h.shape[1], 3), dtype=np.float64)
    for dimension in range(3):
        denominator = float(plus[dimension, dimension] - minus[dimension, dimension])
        if denominator <= 0.0:
            raise ValueError("SITL clipped finite-difference denominator is nonpositive")
        difference = plus[dimension] - minus[dimension]
        if any(index != dimension for index in np.flatnonzero(np.abs(difference) > 1e-15)):
            raise ValueError("SITL probe changed more than one XYZ dimension")
        rows[:, dimension] = (plus_h[dimension] - minus_h[dimension]) / denominator
    return rows


def candidate_xyz_values(
    nominal_xyz: Any,
    qp_xyz: Optional[Any],
    config: Mapping[str, Any],
) -> list[tuple[str, Any]]:
    """Build the registered finite search set without simulator access."""

    np = _numpy()
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
        raise ValueError("nominal XYZ is invalid")
    limit = float(config["optimizer"]["action_limit"])
    search = config["candidate_search"]
    candidates: list[tuple[str, Any]] = []
    if qp_xyz is not None:
        qp = np.asarray(qp_xyz, dtype=np.float64)
        correction = qp - nominal
        for factor in search["correction_extrapolation_factors"]:
            candidates.append(
                ("qp_correction_x%.3g" % float(factor), np.clip(nominal + float(factor) * correction, -limit, limit))
            )
    for magnitude in search["local_offset_magnitudes"]:
        for offset in itertools.product((-float(magnitude), 0.0, float(magnitude)), repeat=3):
            if offset == (0.0, 0.0, 0.0):
                continue
            candidates.append(("local_%.3g" % float(magnitude), np.clip(nominal + np.asarray(offset), -limit, limit)))
    if search["include_global_lattice"]:
        for values in itertools.product(search["lattice_values"], repeat=3):
            candidates.append(("global_lattice", np.asarray(values, dtype=np.float64)))
    if search["include_stop"]:
        candidates.append(("stop", np.zeros(3, dtype=np.float64)))
    if search["include_reverse_nominal"]:
        candidates.append(("reverse_nominal", np.clip(-nominal, -limit, limit)))
    seen = set()
    output = []
    for source, value in candidates:
        key = tuple(float(item) for item in np.asarray(value, dtype=np.float64))
        if key in seen or np.array_equal(np.asarray(value), nominal):
            continue
        seen.add(key)
        output.append((source, np.asarray(value, dtype=np.float64)))
    return output


class SlabbedEightConstraintProbe(ClonedSimulatorStepProbe):
    """Cloned step probe for seven distal parts and the released EE proxy."""

    def clearances(self, env: Any) -> Any:
        np = _numpy()
        links = self.geometry._slabbed_links(env)
        links.append(_released_aegis_end_effector_ellipsoid(env))
        values = np.asarray(
            [
                support_gap(link, self.geometry.obstacle, optimizer_clearance_m=0.0)
                for link in links
            ],
            dtype=np.float64,
        )
        if values.shape != (8,) or not np.all(np.isfinite(values)):
            raise ValueError("SITL probe must return seven distal and one EE clearance")
        return values


class DistalSitlCandidateFilter:
    """One-step exact candidate search around a released AEGIS command."""

    def __init__(
        self,
        config: Mapping[str, Any],
        geometry: MultilinkEllipsoidShadow,
        probe_env: Any,
    ) -> None:
        if config.get("schema_version") != SITL_CANDIDATE_SCHEMA:
            raise ValueError("SITL candidate configuration was not validated")
        if geometry.config.get("schema_version") != DISTAL_SLABBED_SCHEMA:
            raise ValueError("SITL candidate filter requires accepted v4 slab geometry")
        self.config = dict(config)
        self.geometry = geometry
        optimizer = config["optimizer"]
        self.qp = MultiConstraintQp(
            eps_abs=float(optimizer["eps_abs"]),
            eps_rel=float(optimizer["eps_rel"]),
            max_iter=int(optimizer["max_iter"]),
            residual_tolerance=float(optimizer["residual_tolerance"]),
            bound_tolerance=float(optimizer["bound_tolerance_action"]),
        )
        self.probe = SlabbedEightConstraintProbe(probe_env, geometry, clearance_m=0.0)
        self._pending: Optional[dict[str, Any]] = None

    def targets(self, nominal_next_clearance: Any) -> Any:
        np = _numpy()
        geometry = self.config["protected_geometry"]
        nominal = np.asarray(nominal_next_clearance, dtype=np.float64)
        if nominal.shape != (8,) or not np.all(np.isfinite(nominal)):
            raise ValueError("nominal eight-row clearance vector is invalid")
        return np.asarray(
            [float(geometry["distal_clearance_target_m"])] * 7
            + [float(nominal[-1])],
            dtype=np.float64,
        )

    def filter(
        self,
        env: Any,
        nominal_aegis_action: Sequence[float],
        *,
        step: int,
    ) -> tuple[Optional[list[float]], dict[str, Any]]:
        np = _numpy()
        if self._pending is not None:
            raise ValueError("previous SITL transition was not verified")
        started = time.perf_counter_ns()
        action = np.asarray(nominal_aegis_action, dtype=np.float64)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("nominal AEGIS action must have seven finite values")
        if not np.allclose(action[3:6], 0.0, rtol=0.0, atol=1e-12):
            raise ValueError("SITL demo requires the released translational protocol")
        limit = float(self.config["optimizer"]["action_limit"])
        nominal = action.copy()
        nominal[:3] = np.clip(nominal[:3], -limit, limit)
        nominal[3:6] = 0.0
        tolerance = float(self.config["verification"]["clearance_tolerance_m"])
        clone_tolerance = float(self.config["verification"]["clone_state_tolerance"])
        base = self.probe.transition(env, nominal)
        if base["synchronization"]["maximum_absolute_error"] > clone_tolerance:
            raise ValueError("SITL nominal clone synchronization failed")
        repeatability = None
        if int(step) == 0 and self.config["verification"]["repeat_nominal_on_first_step"]:
            repeated = self.probe.transition(env, nominal)
            error = float(
                np.max(
                    np.abs(
                        np.asarray(base["next_state_vector"], dtype=np.float64)
                        - np.asarray(repeated["next_state_vector"], dtype=np.float64)
                    )
                )
            )
            repeatability = {
                "maximum_absolute_next_state_error": error,
                "first_next_state_sha256": base["next_state_sha256"],
                "second_next_state_sha256": repeated["next_state_sha256"],
            }
            if error > clone_tolerance:
                raise ValueError("SITL nominal repeated clone is not deterministic")
        base_h = np.asarray(base["next_h_opt_m"], dtype=np.float64)
        target = self.targets(base_h)
        preferred_target = target.copy()
        preferred_target[:7] = float(
            self.config["protected_geometry"]["distal_activation_clearance_m"]
        )
        nominal_safe = bool(np.all(base_h >= target - tolerance))
        nominal_preferred = bool(np.all(base_h >= preferred_target - tolerance))
        activation = not nominal_preferred
        probe_records = []
        rows = None
        lower = None
        qp_result = None
        qp_xyz = None
        candidates = []
        if activation:
            perturbation = float(self.config["finite_difference"]["perturbation_action"])
            plus_xyz = []
            minus_xyz = []
            plus_h = []
            minus_h = []
            for dimension in range(3):
                positive = nominal.copy()
                negative = nominal.copy()
                positive[dimension] = min(limit, positive[dimension] + perturbation)
                negative[dimension] = max(-limit, negative[dimension] - perturbation)
                plus = self.probe.transition(env, positive)
                minus = self.probe.transition(env, negative)
                plus_xyz.append(positive[:3].copy())
                minus_xyz.append(negative[:3].copy())
                plus_h.append(plus["next_h_opt_m"])
                minus_h.append(minus["next_h_opt_m"])
                probe_records.append(
                    {
                        "dimension": dimension,
                        "plus_xyz": positive[:3].tolist(),
                        "minus_xyz": negative[:3].tolist(),
                        "plus_h_m": list(plus["next_h_opt_m"]),
                        "minus_h_m": list(minus["next_h_opt_m"]),
                    }
                )
            rows = finite_difference_rows(plus_h, minus_h, plus_xyz, minus_xyz)
            lower = preferred_target - base_h + rows @ nominal[:3]
            qp_result = self.qp.solve(
                nominal[:3],
                np.eye(3),
                rows,
                lower,
                -limit * np.ones(3),
                limit * np.ones(3),
            )
            if qp_result.valid and qp_result.qdot_safe is not None:
                qp_xyz = np.asarray(qp_result.qdot_safe, dtype=np.float64)
            candidates = candidate_xyz_values(nominal[:3], qp_xyz, self.config)

        attempts = [
            {
                "source": "nominal_released_aegis",
                "candidate_xyz": nominal[:3].tolist(),
                "verified_next_clearance_m": base_h.tolist(),
                "verified_safe": nominal_safe,
                "preferred_safe": nominal_preferred,
                "minimum_distal_clearance_m": float(np.min(base_h[:7])),
                "objective_l2_from_nominal": 0.0,
                "next_state_sha256": base["next_state_sha256"],
                "env_step_wall_seconds": float(base["env_step_wall_seconds"]),
            }
        ]
        safe_options = []
        preferred_options = []
        if nominal_safe:
            safe_options.append(
                (float(np.min(base_h[:7])), 0.0, "nominal_released_aegis", nominal, base)
            )
        if nominal_preferred:
            preferred_options.append(
                (float(np.min(base_h[:7])), 0.0, "nominal_released_aegis", nominal, base)
            )
        if activation:
            for source, xyz in candidates:
                candidate_action = nominal.copy()
                candidate_action[:3] = xyz
                transition = self.probe.transition(env, candidate_action)
                clearances = np.asarray(transition["next_h_opt_m"], dtype=np.float64)
                safe = bool(np.all(clearances >= target - tolerance))
                preferred = bool(
                    np.all(clearances >= preferred_target - tolerance)
                )
                objective = float(np.linalg.norm(xyz - nominal[:3]))
                minimum_distal = float(np.min(clearances[:7]))
                attempts.append(
                    {
                        "source": source,
                        "candidate_xyz": xyz.tolist(),
                        "verified_next_clearance_m": clearances.tolist(),
                        "verified_safe": safe,
                        "preferred_safe": preferred,
                        "minimum_distal_clearance_m": minimum_distal,
                        "objective_l2_from_nominal": objective,
                        "next_state_sha256": transition["next_state_sha256"],
                        "env_step_wall_seconds": float(transition["env_step_wall_seconds"]),
                    }
                )
                if safe:
                    safe_options.append(
                        (minimum_distal, objective, source, candidate_action, transition)
                    )
                if preferred:
                    preferred_options.append(
                        (minimum_distal, objective, source, candidate_action, transition)
                    )
        accepted = None
        accepted_transition = None
        accepted_source = None
        selectable = preferred_options if preferred_options else safe_options
        if selectable:
            if self.config["candidate_search"]["selection_objective"].startswith(
                "lexicographic_minimum_nominal_deviation"
            ):
                selectable.sort(key=lambda item: (item[1], -item[0], item[2]))
            else:
                selectable.sort(key=lambda item: (-item[0], item[1], item[2]))
            _, _, accepted_source, accepted, accepted_transition = selectable[0]
        record = {
            "schema_version": SITL_CANDIDATE_STEP_SCHEMA,
            "step": int(step),
            "constraint_count": 8,
            "constraint_order": [
                "L5_part_0",
                "L5_part_1",
                "L5_part_2",
                "L6_part_0",
                "L6_part_1",
                "L7_part_0",
                "L7_part_1",
                "released_AEGIS_EE_proxy",
            ],
            "clearance_target_m": target.tolist(),
            "activation_clearance_target_m": preferred_target.tolist(),
            "nominal_released_aegis_action": action.tolist(),
            "nominal_next_clearance_m": base_h.tolist(),
            "nominal_safe": nominal_safe,
            "nominal_preferred": nominal_preferred,
            "activation": activation,
            "selection_objective": self.config["candidate_search"][
                "selection_objective"
            ],
            "finite_difference": {
                "used": activation,
                "probe_count": 6 if activation else 0,
                "probes": probe_records,
                "rows_m_per_normalized_xyz": None if rows is None else rows.tolist(),
                "lower": None if lower is None else lower.tolist(),
            },
            "qp": {
                "used": activation,
                "valid": None if qp_result is None else bool(qp_result.valid),
                "reason": None if qp_result is None else qp_result.reason,
                "candidate_xyz": None if qp_xyz is None else qp_xyz.tolist(),
                "diagnostics": None if qp_result is None else dict(qp_result.diagnostics),
            },
            "verification": {
                "exact_cloned_env_step": True,
                "attempts": attempts,
                "accepted": accepted is not None,
                "accepted_source": accepted_source,
                "accepted_next_clearance_m": (
                    None
                    if accepted_transition is None
                    else list(accepted_transition["next_h_opt_m"])
                ),
                "first_step_nominal_repeatability": repeatability,
                "main_env_post_step_checked": False,
                "main_vs_probe_next_state_max_abs_error": None,
                "main_vs_probe_next_clearance_max_abs_error_m": None,
            },
            "executed_action": None if accepted is None else accepted.tolist(),
            "modified": bool(accepted is not None and not np.array_equal(accepted, nominal)),
            "correction_l2": (
                None
                if accepted is None
                else float(np.linalg.norm(accepted[:3] - nominal[:3]))
            ),
            "timing": {
                "candidate_env_step_wall_seconds": float(
                    sum(item["env_step_wall_seconds"] for item in attempts)
                ),
                "total_filter_wall_seconds": (time.perf_counter_ns() - started) * 1e-9,
            },
        }
        if accepted_transition is not None:
            self._pending = {
                "record": record,
                "state_vector": np.asarray(accepted_transition["next_state_vector"], dtype=np.float64).copy(),
                "clearances": np.asarray(accepted_transition["next_h_opt_m"], dtype=np.float64).copy(),
            }
        return None if accepted is None else accepted.tolist(), record

    def verify_executed_transition(self, env: Any, record: Mapping[str, Any]) -> None:
        np = _numpy()
        pending = self._pending
        if pending is None or pending["record"] is not record:
            raise ValueError("no matching SITL transition awaits verification")
        actual_state = _dynamic_state_vector(env)
        actual_clearance = self.probe.clearances(env)
        state_error = float(np.max(np.abs(actual_state - pending["state_vector"])))
        clearance_error = float(np.max(np.abs(actual_clearance - pending["clearances"])))
        verification = record["verification"]
        verification["main_env_post_step_checked"] = True
        verification["main_vs_probe_next_state_max_abs_error"] = state_error
        verification["main_vs_probe_next_clearance_max_abs_error_m"] = clearance_error
        self._pending = None
        tolerance = float(self.config["verification"]["clone_state_tolerance"])
        if state_error > tolerance or clearance_error > tolerance:
            raise ValueError("executed OSC step differs from accepted SITL clone")


def summarize_sitl_steps(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    np = _numpy()
    if not records:
        return {"status": "no_records", "step_count": 0}
    timings = np.asarray(
        [float(item["timing"]["total_filter_wall_seconds"]) for item in records],
        dtype=np.float64,
    )
    qps = [
        float(item["qp"]["diagnostics"]["timing"]["total_wall_seconds"])
        for item in records
        if item["qp"]["diagnostics"] is not None
    ]

    def stats(values: Any) -> Optional[dict[str, float]]:
        array = np.asarray(values, dtype=np.float64)
        if not array.size:
            return None
        return {
            "mean_seconds": float(np.mean(array)),
            "p95_seconds": float(np.quantile(array, 0.95)),
            "maximum_seconds": float(np.max(array)),
        }

    return {
        "status": (
            "complete"
            if all(item["verification"]["accepted"] for item in records)
            else "method_failure"
        ),
        "step_count": len(records),
        "constraint_count_per_step": sorted({int(item["constraint_count"]) for item in records}),
        "material_intervention_count": sum(bool(item["modified"]) for item in records),
        "first_material_intervention_step": next(
            (int(item["step"]) for item in records if item["modified"]), None
        ),
        "all_executed_transitions_match_clone": all(
            item["verification"]["main_env_post_step_checked"] for item in records
        ),
        "total_filter_timing": stats(timings),
        "active_qp_timing": stats(qps),
        "simulator_candidate_step_count": int(
            sum(
                len(item["verification"]["attempts"])
                + (1 if item["verification"]["first_step_nominal_repeatability"] else 0)
                + int(item["finite_difference"]["probe_count"])
                for item in records
            )
        ),
    }
