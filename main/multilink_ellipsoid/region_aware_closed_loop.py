"""Receding learned regional QP controller for the conditional E05 test."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from .region_aware_mlp import (
    guarded_targets, live_regions, solve_regional_qps,
)


class RegionAwareRecedingFilter:
    """Select actions from learned rows; simulator rollouts are measurement only."""

    def __init__(
        self, *, models: Sequence[Any], model_state: Mapping[str, Any],
        config: Mapping[str, Any], probe: Any,
    ) -> None:
        self.models = list(models)
        self.model_state = model_state
        self.config = config
        self.probe = probe

    def propose(
        self, main_env: Any, nominal_action: Sequence[float],
        second_action: Sequence[float], pair_state_features: Any,
        current_clearance_m: Sequence[float], *, step: int,
    ) -> tuple[Any, dict[str, Any]]:
        import numpy as np

        nominal = np.asarray(nominal_action, dtype=np.float64)
        second = np.asarray(second_action, dtype=np.float64)
        current = np.asarray(current_clearance_m, dtype=np.float64)
        if (
            nominal.shape != (7,) or second.shape != (7,)
            or current.shape != (7,)
            or not np.all(np.isfinite(nominal)) or not np.all(np.isfinite(second))
            or not np.all(np.isfinite(current))
        ):
            raise ValueError("region-aware receding actions differ")
        started = time.perf_counter_ns()
        activation = float(
            self.config["closed_loop"]["activation_current_distal_clearance_m"]
        )
        active = bool(float(np.min(current)) <= activation)
        if active:
            _, _, regions = live_regions(nominal[:3], self.config["partition"])
            prediction_started = time.perf_counter_ns()
            targets = guarded_targets(
                self.models, self.model_state, pair_state_features, regions,
                self.config["uncertainty"],
            )
            prediction_seconds = (
                time.perf_counter_ns() - prediction_started
            ) * 1.0e-9
            qp_started = time.perf_counter_ns()
            qp = solve_regional_qps(
                nominal[:3], regions, targets, self.config["projection"]
            )
            qp_seconds = (time.perf_counter_ns() - qp_started) * 1.0e-9
            selected = qp["selected"]
            if selected is None:
                return None, {
                    "step": int(step), "status": "method_failure",
                    "reason": "no_valid_learned_regional_QP", "active": True,
                    "current_minimum_distal_clearance_m": float(np.min(current)),
                    "prediction_wall_seconds": prediction_seconds,
                    "QP_wall_seconds": qp_seconds, "predicted_targets": targets,
                    "QP": qp, "exact_measurement": None,
                    "total_filter_wall_seconds": (
                        time.perf_counter_ns() - started
                    ) * 1.0e-9,
                }
            executed = nominal.copy()
            executed[:3] = np.asarray(selected["candidate_xyz"], dtype=np.float64)
            modified = bool(
                float(selected["correction_l2"])
                > float(self.config["projection"]["bound_tolerance_action"])
            )
        else:
            targets = None
            prediction_seconds = 0.0
            qp_seconds = 0.0
            selected = {
                "valid": True, "reason": "inactive_current_clearance_above_50mm",
                "candidate_xyz": nominal[:3].tolist(), "correction_l2": 0.0,
                "region_index": None,
            }
            qp = {
                "status": "inactive", "regional_proposals": [],
                "valid_proposal_count": 0, "selected": selected,
            }
            executed = nominal.copy()
            modified = False
        measurement_started = time.perf_counter_ns()
        chunk = self.probe.rollout_chunk(main_env, [executed, second])
        measurement_seconds = (
            time.perf_counter_ns() - measurement_started
        ) * 1.0e-9
        transitions = chunk["transitions"]
        minimum = np.min(np.asarray([
            item["minimum_substep_clearance_m"] for item in transitions
        ], dtype=np.float64), axis=0)
        raw_contact_count = sum(
            int(item["raw_protected_contact_count"]) for item in transitions
        )
        maximum_motion = max(
            float(item["maximum_within_step_obstacle_l1_displacement_m"])
            for item in transitions
        )
        exact = {
            "measurement_only": True,
            "minimum_distal_margin_m": minimum[:7].tolist(),
            "minimum_released_AEGIS_EE_margin_m": float(minimum[7]),
            "raw_protected_contact_count": raw_contact_count,
            "maximum_within_step_obstacle_l1_displacement_m": maximum_motion,
            "first_transition_next_state_sha256": transitions[0][
                "next_state_sha256"
            ],
            "second_transition_next_state_sha256": transitions[1][
                "next_state_sha256"
            ],
            "true_distal_safe": bool(
                np.all(minimum[:7] >= 0.0) and raw_contact_count == 0
                and maximum_motion
                <= float(self.config["closed_loop"][
                    "maximum_per_step_obstacle_l1_displacement_m"
                ])
            ),
            "rollout_wall_seconds": measurement_seconds,
        }
        return executed, {
            "step": int(step), "status": "complete", "reason": None,
            "active": active,
            "current_minimum_distal_clearance_m": float(np.min(current)),
            "modified": modified,
            "nominal_action": nominal.tolist(), "executed_action": executed.tolist(),
            "prediction_wall_seconds": prediction_seconds,
            "QP_wall_seconds": qp_seconds, "predicted_targets": targets,
            "QP": qp,
            "exact_measurement": exact,
            "total_filter_wall_seconds": (
                time.perf_counter_ns() - started
            ) * 1.0e-9,
        }
