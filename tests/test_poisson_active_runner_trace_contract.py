from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import unittest

from main.poisson_fullbody.contracts import attach_payload_hash
from main.poisson_fullbody.result_schema import validate_episode_result
from scripts.run_poisson_active_canary import (
    ActiveRunnerError,
    _canonical,
    _sha256,
    _validate_trace_payload_against_scientific,
)
from tests.test_poisson_result_schema import valid_partial_payload


ROOT = Path(__file__).resolve().parents[1]


def serialized_trace_fixture():
    """One completed 2 ms callback followed by a tracking fail-closed stop."""

    scientific = valid_partial_payload("controller_tracking_invalid")
    source_action = [0.0] * 7
    nominal = [0.1] + [0.0] * 6
    issued = list(nominal)
    measured = [0.16] + [0.0] * 6
    settled_eef = [0.0, 0.0, 0.0]
    forwarded_eef = [0.001, 0.0, 0.0]
    tracking_rmse = math.sqrt((0.06**2) / 7.0)
    crossing = {
        "high_level_index": 0,
        "inner_control_index": 0,
        "physics_substep_index": 0,
        "error_linf_rad_s": 0.06,
        "cumulative_rmse_rad_s": tracking_rmse,
        "linf_threshold_rad_s": 0.05,
        "rmse_threshold_rad_s": 0.02,
        "command_rad_s": issued,
        "measured_rad_s": measured,
    }
    nominal_ledger = [
        {
            "high_level_index": 0,
            "inner_control_index": 0,
            "qdot_rad_s": nominal,
        }
    ]
    executed_ledger = [
        {
            "high_level_index": 0,
            "inner_control_index": 0,
            "qdot_rad_s": issued,
        }
    ]
    entered_actions = [source_action]
    completed_actions = []
    scientific["execution"].update(
        {
            "entered_source_action_prefix_sha256": _sha256(
                _canonical(entered_actions)
            ),
            "completed_high_level_source_action_prefix_sha256": _sha256(
                _canonical(completed_actions)
            ),
            "nominal_joint_velocity_ledger_sha256": _sha256(
                _canonical(nominal_ledger)
            ),
            "executed_controller_action_ledger_sha256": _sha256(
                _canonical(executed_ledger)
            ),
        }
    )
    scientific["endpoints"]["usefulness"].update(
        {
            "nominal_joint_motion_integral_rad": 0.1 * 0.002,
            "safe_joint_motion_integral_rad": 0.1 * 0.002,
            "measured_joint_motion_integral_rad": 0.16 * 0.002,
            "eef_path_length_m": 0.001,
            "correction_integral_rad": 0.0,
            "motion_retention_ratio": 1.0,
            "zero_motion_fraction": 0.0,
            "all_issued_arm_joint_commands_zero": False,
            "safety_by_no_execution": False,
        }
    )
    scientific["endpoints"]["validity"].update(
        {
            "maximum_velocity_tracking_error_rad_s": 0.06,
            "velocity_tracking_rmse_rad_s": tracking_rmse,
            "velocity_tracking_observed_physics_substep_count": 1,
            "first_velocity_tracking_threshold_crossing": crossing,
        }
    )
    car_ledger = {
        "settled_active_obstacle_root_position_m": [0.0, 0.0, 0.0],
        "settled_active_obstacle_position_sha256": "1" * 64,
        "source_settled_active_obstacle_position_sha256": "1" * 64,
        "historical_settled_active_obstacle_position_sha256": "1" * 64,
        "completed_post_step_positions": [],
    }
    scientific["endpoints"]["safety"]["paper_car"].update(
        {
            "maximum_active_obstacle_l1_displacement_m": 0.0,
            "position_ledger_sha256": _sha256(_canonical(car_ledger)),
        }
    )

    validity = scientific["endpoints"]["validity"]
    usefulness = scientific["endpoints"]["usefulness"]
    paper_car = scientific["endpoints"]["safety"]["paper_car"]
    outcome = {
        "arm": scientific["arm"],
        "completion_class": scientific["completion_class"],
        "exposure_complete": False,
        "prefix": {
            "high_level_steps": 1,
            "inner_control_steps": 1,
            "physics_substeps": 1,
            "physics_exposure_seconds": 0.002,
        },
        "physics_clock": {
            "settled_time_s": 2.0,
            "terminal_time_s": 2.002,
            "observed_exposure_s": 0.002,
            "expected_from_completed_substeps_s": 0.002,
            "exact_count_consistent_within_abs_1e_10_s": True,
        },
        "entered_source_actions": entered_actions,
        "completed_source_actions": completed_actions,
        "nominal_ledger": nominal_ledger,
        "executed_ledger": executed_ledger,
        "inner_trace": [
            {
                "high_level_index": 0,
                "inner_control_index": 0,
                "D_opt_min_m": 0.03,
                "minimum_h_m2": 0.004,
                "minimum_nominal_cbf_residual_m2_per_s": -0.2,
                "minimum_safe_cbf_residual_m2_per_s": -1e-8,
            }
        ],
        "physics_trace": [
            {
                "high_level_index": 0,
                "inner_control_index": 0,
                "physics_substep_index": 0,
                "nominal_joint_velocity_command_rad_s": nominal,
                "issued_joint_velocity_command_rad_s": issued,
                "measured_arm_joint_velocity_rad_s": measured,
                "forwarded_eef_position_m": forwarded_eef,
            }
        ],
        "fail_closed_attempt_trace": [],
        "realized_cbf_trace": [
            {
                "high_level_index": 0,
                "inner_control_index": 0,
                "physics_substep_index": 0,
                "valid": True,
                "query_count": 1,
                "minimum_realized_cbf_residual_m2_per_s": 1e-6,
                "minimum_h_m2": 0.004,
                "D_opt_min_m": 0.03,
                "first_negative_crossing": None,
            }
        ],
        "monitor": {
            "D_sim_min_m": 0.01,
            "any_contact": False,
            "link56_contact": False,
            "total": 0,
            "settled": 0,
            "rollout": 0,
            "live": 0,
            "post": 0,
            "first": None,
            "first_settled": None,
            "first_live": None,
            "first_post": None,
            "record": {
                "observed_physics_substeps": 1,
                "total_physical_contact_point_record_count": 0,
                "rollout_phase_physical_contact_point_record_count": 0,
                "live_solver_nonpositive_contact_point_record_count": 0,
                "post_state_physical_contact_point_record_count": 0,
            },
        },
        "paper_car": paper_car,
        "paper_car_position_ledger": car_ledger,
        "task": scientific["endpoints"]["task"],
        "usefulness": usefulness,
        "validity": validity,
        "optimizer_counts": {
            "attempt_count": 1,
            "solved_count": 1,
            "infeasible_count": 0,
            "solver_failure_count": 0,
            "postcheck_failure_count": 0,
        },
        "optimizer_terminal_status": "solved",
        "minimums": {
            "h_m2": 0.004,
            "D_opt_m": 0.03,
            "nominal_raw_cbf_residual_m2_per_s": -0.2,
            "safe_raw_cbf_residual_m2_per_s": -1e-8,
            "safe_normalized_cbf_residual": -1e-9,
            "realized_raw_cbf_residual_m2_per_s": 1e-6,
        },
        "initial_protected_sample_audit": {
            "protected_sample_count": 1,
            "field_query_count": 1,
            "valid_field_query_count": 1,
            "minimum_h_m2": 0.004,
            "D_opt_min_m": 0.03,
            "strict_safe_start": True,
        },
        "realized_cbf_audit": {
            "semantics": "every_completed_2ms_post_state_grad_h_T_J_qvel_actual_plus_alpha_h",
            "observed_physics_substep_count": 1,
            "residual_evaluation_count": 1,
            "negative_residual_callback_count": 0,
            "first_negative_residual": None,
        },
        "eef_path_audit": {
            "semantics": "settled_forwarded_grip_site_plus_every_completed_2ms_post_state",
            "position_sample_count": 2,
            "settled_forwarded_eef_position_m": settled_eef,
        },
        "tracking": {
            "command_count": 1,
            "observed_physics_substep_count": 1,
            "maximum_linf_error_rad_s": 0.06,
            "cumulative_rmse_rad_s": tracking_rmse,
            "maximum_linf_threshold_rad_s": 0.05,
            "maximum_rmse_threshold_rad_s": 0.02,
            "first_threshold_crossing": crossing,
        },
        "terminal_simulator_state_sha256": scientific["execution"][
            "terminal_simulator_state_sha256"
        ],
        "terminal_observation_sha256": scientific["execution"][
            "terminal_observation_sha256"
        ],
    }
    trace = {
        "schema_version": "vlsa_poisson_active_arm_trace.v2",
        "scientific_result": False,
        "run_id": scientific["run_id"],
        "case_id": scientific["case_id"],
        "arm": scientific["arm"],
        "outcome": outcome,
    }
    # Exercise the same plain JSON representation that is read back from disk.
    return json.loads(json.dumps(trace)), json.loads(json.dumps(scientific))


class ActiveRunnerTraceContractTest(unittest.TestCase):
    def test_one_substep_tracking_failure_trace_matches_scientific_projection(self):
        trace, scientific = serialized_trace_fixture()
        validate_episode_result(attach_payload_hash(scientific))
        _validate_trace_payload_against_scientific(trace, scientific)

    def test_tampered_measured_qvel_is_rejected(self):
        trace, scientific = serialized_trace_fixture()
        tampered = copy.deepcopy(trace)
        tampered["outcome"]["physics_trace"][0][
            "measured_arm_joint_velocity_rad_s"
        ][0] = 0.17
        with self.assertRaisesRegex(ActiveRunnerError, "trace/scientific mismatch"):
            _validate_trace_payload_against_scientific(tampered, scientific)

    def test_tampered_forwarded_eef_position_is_rejected(self):
        trace, scientific = serialized_trace_fixture()
        tampered = copy.deepcopy(trace)
        tampered["outcome"]["physics_trace"][0][
            "forwarded_eef_position_m"
        ][0] = 0.002
        with self.assertRaisesRegex(ActiveRunnerError, "trace/scientific mismatch"):
            _validate_trace_payload_against_scientific(tampered, scientific)

    def test_publication_and_partial_resume_guards_are_structurally_ordered(self):
        source = (ROOT / "scripts/run_poisson_active_canary.py").read_text(
            encoding="utf-8"
        )
        validate = source.index(
            "validate_episode_result(attach_payload_hash(payload))"
        )
        publish = source.index("publish_episode_result(final_path, payload")
        self.assertLess(validate, publish)

        complete_receipt = source.index("if receipt_path.exists():")
        complete_return = source.index("            return 0", complete_receipt)
        partial_guard = source.index("                if final_path.exists():")
        arm_execution = source.index("                    outcome = _run_arm(", partial_guard)
        refusal = (
            "partial arm result exists without a complete immutable run "
            "\"\n                        \"receipt; choose a new run ID"
        )
        self.assertLess(complete_receipt, complete_return)
        self.assertLess(complete_return, partial_guard)
        self.assertLess(partial_guard, arm_execution)
        self.assertIn(refusal, source[partial_guard:arm_execution])


if __name__ == "__main__":
    unittest.main()
