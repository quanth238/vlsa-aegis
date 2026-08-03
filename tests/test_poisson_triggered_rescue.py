import copy
import json
from pathlib import Path
import unittest


from main.poisson_fullbody.triggered_rescue import (
    METRICS_SCHEMA,
    TriggeredRescueError,
    classify_triggered_rescue,
    evaluate_direct_qdot_trigger_candidate,
    select_first_actionable_trigger,
    validate_triggered_rescue_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/vlsa_poisson_triggered_rescue.v1.json"
E05 = "vlsa-t1-goal-ii-t0-e05"
E42 = "vlsa-t1-goal-ii-t3-e42"


def protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def positive_metrics():
    return {
        "schema_version": METRICS_SCHEMA,
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


class TriggeredRescueProtocolTests(unittest.TestCase):
    def test_protocol_registers_one_fixed_link_set_and_no_trigger_action(self):
        value = protocol()
        e05 = validate_triggered_rescue_protocol(value, case_id=E05)
        e42 = validate_triggered_rescue_protocol(value, case_id=E42)
        self.assertEqual(
            e05["protected_robot_body_names"],
            ("robot0_link5", "robot0_link6"),
        )
        self.assertEqual(
            e05["protected_robot_body_names"],
            e42["protected_robot_body_names"],
        )
        self.assertEqual(e05["horizon_action_count"], 237)
        self.assertEqual(e42["horizon_action_count"], 120)
        self.assertIsNone(value["trigger"]["expected_action_index"])
        serialized = json.dumps(value, sort_keys=True)
        self.assertNotIn("branch_action_index", serialized)
        self.assertNotIn("native_OSC_pose_to_qdot_predictor", serialized)

    def test_task_conditioned_link_selection_is_rejected(self):
        value = copy.deepcopy(protocol())
        value["method"]["protected_robot_body_names"] = ["robot0_link5"]
        with self.assertRaises(TriggeredRescueError):
            validate_triggered_rescue_protocol(value, case_id=E05)

    def test_expected_trigger_action_is_rejected(self):
        value = copy.deepcopy(protocol())
        value["trigger"]["expected_action_index"] = 180
        with self.assertRaises(TriggeredRescueError):
            validate_triggered_rescue_protocol(value, case_id=E05)

    def test_trigger_cannot_use_case_or_historical_contact_identity(self):
        for key in (
            "case_identity_used_by_trigger",
            "historical_contact_action_used_by_trigger",
        ):
            value = copy.deepcopy(protocol())
            value["trigger"][key] = True
            with self.assertRaises(TriggeredRescueError):
                validate_triggered_rescue_protocol(value, case_id=E05)


class DirectQdotTriggerTests(unittest.TestCase):
    def _candidate(self, nominal_x=-0.2, safe_x=0.0, action=12):
        return evaluate_direct_qdot_trigger_candidate(
            source_action_index=action,
            physical_boundary=action * 25,
            nominal_qdot=[nominal_x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            safe_qdot=[safe_x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            h=[0.01],
            gradients=[[1.0, 0.0, 0.0]],
            jacobians=[[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            alpha_per_s=5.0,
            nominal_threshold_m2_per_s=-5e-7,
            material_correction_threshold_rad_s=1e-4,
            sample_records=[{
                "sample_id": 7,
                "body_id": 5,
                "body_name": "robot0_link5",
                "geom_id": 44,
                "geom_name": "robot0_link5_collision",
            }],
        )

    def test_unsafe_bounded_direct_command_triggers(self):
        row = self._candidate()
        self.assertTrue(row["triggered"])
        self.assertAlmostEqual(
            row["nominal_minimum_cbf_residual_m2_per_s"], -0.15
        )
        self.assertEqual(
            row["decision_variable"],
            "bounded_physical_arm_joint_velocity_rad_s",
        )

    def test_safe_nominal_command_does_not_trigger(self):
        row = self._candidate(nominal_x=0.0, safe_x=0.0)
        self.assertFalse(row["triggered"])

    def test_unbounded_surrogate_command_is_rejected(self):
        with self.assertRaises(TriggeredRescueError):
            self._candidate(nominal_x=-5.0, safe_x=0.0)

    def test_first_actionable_trigger_is_latched(self):
        safe = self._candidate(nominal_x=0.0, safe_x=0.0, action=1)
        first = self._candidate(action=2)
        later = self._candidate(action=3)
        selected = select_first_actionable_trigger([safe, first, later])
        self.assertEqual(selected["source_action_index"], 2)


class TriggeredRescueClassificationTests(unittest.TestCase):
    def test_positive_requires_contact_motion_and_task_success(self):
        result = classify_triggered_rescue(
            positive_metrics(), protocol(), case_id=E05
        )
        self.assertEqual(
            result["classification"],
            "SAFE_TASK_SUCCESS_USEFUL_CORRECTION",
        )
        self.assertTrue(result["feasible"])

    def test_no_warning_is_a_scientific_negative(self):
        metrics = positive_metrics()
        metrics["trigger_found"] = False
        metrics["exact_paired_trigger_state"] = False
        metrics["full_recorded_episode_complete"] = False
        metrics["all_psf_field_queries_valid"] = False
        metrics["all_psf_qps_solved_and_postchecked"] = False
        result = classify_triggered_rescue(metrics, protocol(), case_id=E05)
        self.assertEqual(
            result["classification"],
            "NO_ACTIONABLE_DIRECT_QDOT_WARNING",
        )

    def test_task_failure_cannot_be_reported_positive(self):
        metrics = positive_metrics()
        metrics["psf_terminal_task_success"] = False
        result = classify_triggered_rescue(metrics, protocol(), case_id=E05)
        self.assertEqual(result["classification"], "CONTACT_PREVENTED_TASK_FAILED")

    def test_stopping_cannot_be_reported_positive(self):
        metrics = positive_metrics()
        metrics["psf_method_stop_or_stall"] = True
        result = classify_triggered_rescue(metrics, protocol(), case_id=E05)
        self.assertEqual(result["classification"], "STOP_OR_METHOD_FAILURE")

    def test_typed_method_refusal_is_a_valid_negative_not_apparatus_noise(self):
        metrics = positive_metrics()
        metrics["psf_method_stop_or_stall"] = True
        metrics["psf_exposure_complete"] = False
        metrics["full_recorded_episode_complete"] = False
        metrics["all_psf_field_queries_valid"] = False
        metrics["all_psf_qps_solved_and_postchecked"] = False
        result = classify_triggered_rescue(metrics, protocol(), case_id=E05)
        self.assertTrue(result["apparatus_valid"])
        self.assertEqual(result["classification"], "STOP_OR_METHOD_FAILURE")

    def test_shifted_contact_cannot_be_reported_positive(self):
        metrics = positive_metrics()
        metrics["psf_shifted_link56_external_contact_present"] = True
        result = classify_triggered_rescue(metrics, protocol(), case_id=E05)
        self.assertEqual(result["classification"], "CONTACT_REMAINS_OR_SHIFTED")


if __name__ == "__main__":
    unittest.main()
