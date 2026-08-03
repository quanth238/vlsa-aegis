import json
from pathlib import Path
import unittest


from scripts.validate_poisson_triggered_rescue_artifact import (
    _Audit,
    _classify,
    _post_motion,
    _trigger,
)
from main.poisson_fullbody.triggered_rescue import validate_triggered_rescue_protocol


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts/validate_poisson_triggered_rescue_artifact.py"
PROTOCOL = json.loads(
    (ROOT / "configs/vlsa_poisson_triggered_rescue.v1.json").read_text(
        encoding="utf-8"
    )
)


def positive_metrics():
    return {
        "trigger_scan_complete": True,
        "trigger_found": True,
        "native_prefix_exact": True,
        "exact_paired_trigger_state": True,
        "full_recorded_episode_complete": True,
        "baseline_exposure_complete": True,
        "psf_exposure_complete": True,
        "baseline_link56_contact_present": True,
        "psf_any_robot_selected_obstacle_contact_present": False,
        "psf_shifted_link56_external_contact_present": False,
        "psf_contact_terminated": False,
        "material_correction_before_baseline_contact": True,
        "psf_paper_car_avoided": True,
        "psf_useful_post_correction_motion": True,
        "psf_task_success_after_correction": True,
        "psf_terminal_task_success": True,
        "psf_method_stop_or_stall": False,
        "all_psf_field_queries_valid": True,
        "all_psf_qps_solved_and_postchecked": True,
    }


def trigger_row():
    row = {
        "source_action_index": 0,
        "physical_boundary": 0,
        "decision_variable": "bounded_physical_arm_joint_velocity_rad_s",
        "nominal_minimum_cbf_residual_m2_per_s": -0.1,
        "safe_minimum_cbf_residual_m2_per_s": 0.01,
        "nominal_threshold_m2_per_s": -5e-7,
        "filter_correction_l2_rad_s": 0.2,
        "material_correction_threshold_rad_s": 1e-4,
        "argmin_protected_sample": {
            "sample_id": 1,
            "body_id": 5,
            "body_name": "robot0_link5",
            "geom_id": 44,
            "geom_name": "link5_collision",
        },
        "triggered": True,
        "bounded_nominal_qdot_rad_s": [-0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "field_query_count": 20,
        "field_queries_all_valid_positive": True,
        "hard_qp_attempted": True,
        "hard_qp_valid": True,
    }
    return row


class TriggeredRescueIndependentValidatorTests(unittest.TestCase):
    def test_validator_does_not_import_producer_or_classifier(self):
        source = VALIDATOR.read_text(encoding="utf-8")
        self.assertNotIn("run_poisson_triggered_rescue", source)
        self.assertNotIn("classify_triggered_rescue", source)
        self.assertIn("load_historical_action_replay", source)

    def test_first_direct_qdot_trigger_is_reconstructed(self):
        row = trigger_row()
        selected = dict(
            row,
            native_goal_values_at_boundary=[False],
            simulator_state_sha256="1" * 64,
            official_state_raw_bytes_sha256="2" * 64,
        )
        audit = _Audit()
        result = _trigger(
            audit,
            {
                "trigger_scan": {
                    "complete": True,
                    "decision_variable": "bounded_physical_arm_joint_velocity_rad_s",
                    "fixed_trigger_action_index_used": False,
                    "case_identity_used_by_trigger": False,
                    "historical_contact_action_used_by_trigger": False,
                    "rows": [row],
                    "first_actionable_trigger": selected,
                }
            },
            validate_triggered_rescue_protocol(
                PROTOCOL, case_id="vlsa-t1-goal-ii-t0-e05"
            ),
            237,
        )
        self.assertEqual(audit.discrepancies, [])
        self.assertTrue(result["found"])
        self.assertEqual(result["action"], 0)

    def test_forged_trigger_flag_is_rejected(self):
        row = trigger_row()
        row["triggered"] = False
        audit = _Audit()
        _trigger(
            audit,
            {
                "trigger_scan": {
                    "complete": True,
                    "decision_variable": "bounded_physical_arm_joint_velocity_rad_s",
                    "rows": [row],
                    "first_actionable_trigger": None,
                }
            },
            validate_triggered_rescue_protocol(
                PROTOCOL, case_id="vlsa-t1-goal-ii-t0-e05"
            ),
            1,
        )
        self.assertIn("trigger_row_0_flag_differs", audit.discrepancies)

    def test_useful_motion_is_reconstructed_from_executed_traces(self):
        command = {
            "source_action_index": 10,
            "inner_control_index": 0,
            "physical_boundary": 250,
            "correction_l2_rad_s": 0.1,
            "executed_qdot_rad_s": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "eef_position_before_update_world_m": [0.0, 0.0, 0.0],
        }
        physics = [
            {
                "source_action_index": 10,
                "inner_control_index": 0,
                "physics_substep_index": index,
                "measured_qvel_rad_s": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "eef_position_world_m": [0.001 * (index + 1), 0.0, 0.0],
            }
            for index in range(5)
        ]
        motion = _post_motion(_Audit(), [command], physics, 250)
        self.assertAlmostEqual(motion["filter_correction_integral_rad"], 0.001)
        self.assertAlmostEqual(motion["measured_joint_motion_integral_rad"], 0.001)
        self.assertAlmostEqual(motion["cartesian_path_length_m"], 0.005)
        self.assertEqual(motion["zero_command_fraction"], 0.0)

    def test_method_stop_is_a_scientific_negative(self):
        metrics = positive_metrics()
        metrics["psf_method_stop_or_stall"] = True
        metrics["psf_exposure_complete"] = False
        metrics["full_recorded_episode_complete"] = False
        metrics["all_psf_field_queries_valid"] = False
        metrics["all_psf_qps_solved_and_postchecked"] = False
        classification, valid = _classify(metrics)
        self.assertTrue(valid)
        self.assertEqual(classification, "STOP_OR_METHOD_FAILURE")

    def test_shifted_contact_cannot_pass(self):
        metrics = positive_metrics()
        metrics["psf_shifted_link56_external_contact_present"] = True
        classification, valid = _classify(metrics)
        self.assertTrue(valid)
        self.assertEqual(classification, "CONTACT_REMAINS_OR_SHIFTED")


if __name__ == "__main__":
    unittest.main()
