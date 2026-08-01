import copy
import json
from pathlib import Path
import unittest

from main.poisson_fullbody.fast_feasibility import (
    FastFeasibilityError,
    classify_fast_feasibility,
    validate_fast_feasibility_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return json.loads(
        (ROOT / "configs/vlsa_poisson_fast_feasibility.v1.json").read_text(
            encoding="utf-8"
        )
    )


def valid_metrics():
    return {
        "exact_paired_start": True,
        "adapter_exposure_complete": True,
        "psf_exposure_complete": True,
        "adapter_physics_monitor_trace_counts_match": True,
        "psf_physics_monitor_trace_counts_match": True,
        "adapter_precontact_static_obstacle_admissible": True,
        "psf_static_obstacle_admissible": True,
        "psf_all_qp_solved": True,
        "psf_all_qp_postchecks_passed": True,
        "psf_all_joint_limit_postchecks_passed": True,
        "psf_all_post_state_field_queries_valid_and_positive": True,
        "psf_precontact_execution_valid": True,
        "psf_precontact_static_obstacle_admissible": True,
        "psf_precontact_field_queries_valid_and_positive": True,
        "psf_nominal_cbf_activation_with_material_correction_before_adapter_contact": True,
        "psf_active_interval_motion_complete": True,
        "all_issued_commands_within_physical_bounds": True,
        "adapter_all_nominal_commands_within_dynamic_joint_bounds": True,
        "psf_all_nominal_commands_within_dynamic_joint_bounds": True,
        "adapter_link56_contact_present": True,
        "adapter_first_selected_obstacle_contact_is_link56": True,
        "psf_link56_contact_present": False,
        "psf_any_robot_selected_obstacle_contact_present": False,
        "psf_conservative_full_robot_clearance_lower_bound_available": True,
        "adapter_filter_update_count": 40,
        "psf_filter_update_count": 40,
        "adapter_physics_substep_count": 200,
        "psf_physics_substep_count": 200,
        "adapter_monitor_observed_physics_substep_count": 200,
        "psf_monitor_observed_physics_substep_count": 200,
        "psf_qp_solve_count": 40,
        "psf_qp_postcheck_count": 40,
        "psf_joint_limit_postcheck_count": 40,
        "adapter_issued_command_bound_check_count": 40,
        "psf_issued_command_bound_check_count": 40,
        "adapter_precontact_tracking_observation_count": 190,
        "psf_tracking_observation_count": 200,
        "psf_pre_filter_field_observation_count": 40,
        "psf_pre_filter_field_query_count": 40 * 1531,
        "psf_post_state_field_observation_count": 1,
        "psf_post_state_field_query_count": 1531,
        "psf_nonpositive_post_state_field_query_count": 0,
        "psf_precontact_tracking_observation_count": 200,
        "psf_activation_update_count_before_adapter_contact": 2,
        "adapter_first_link56_contact_physical_boundary": 4697,
        "psf_first_activation_physical_boundary": 4650,
        "psf_invalid_field_query_count": 0,
        "adapter_nominal_dynamic_bound_check_count": 40,
        "psf_nominal_dynamic_bound_check_count": 40,
        "adapter_nominal_dynamic_bound_violation_count": 0,
        "psf_nominal_dynamic_bound_violation_count": 0,
        "psf_protected_sample_count": 1531,
        "psf_minimum_D_sim_m": -0.01,
        "psf_minimum_safe_cbf_residual_m2_per_s": -4.0e-7,
        "adapter_precontact_tracking_linf_rad_s": 0.04,
        "adapter_precontact_tracking_rmse_rad_s": 0.01,
        "psf_tracking_linf_rad_s": 0.04,
        "psf_tracking_rmse_rad_s": 0.01,
        "psf_precontact_tracking_linf_rad_s": 0.04,
        "psf_precontact_tracking_rmse_rad_s": 0.01,
        "maximum_active_filter_correction_norm_rad_s": 0.02,
        "maximum_active_safe_command_norm_rad_s": 0.2,
        "active_safe_to_nominal_command_motion_ratio": 0.7,
        "active_safe_measured_joint_motion_integral_rad": 0.02,
        "active_cartesian_path_length_m": 0.002,
        "psf_conservative_full_robot_surface_clearance_lower_bound_m": 0.001,
        "psf_minimum_nominal_cbf_residual_before_adapter_contact_m2_per_s": -1.0e-6,
        "psf_maximum_activation_correction_norm_before_adapter_contact_rad_s": 0.02,
    }


class FastFeasibilityProtocolTests(unittest.TestCase):
    def test_frozen_window_and_cadence(self):
        derived = validate_fast_feasibility_protocol(protocol())
        self.assertEqual(derived["start_boundary"], 4500)
        self.assertEqual(derived["end_boundary"], 4700)
        self.assertEqual(derived["start_action"], 180)
        self.assertEqual(derived["end_action"], 187)
        self.assertEqual(derived["expected_updates"], 40)
        self.assertEqual(derived["expected_substeps"], 200)
        self.assertEqual(derived["qp_max_iterations"], 50000)
        self.assertEqual(derived["expected_post_state_field_observations"], 1)

    def test_protocol_requires_executed_aegis_actions_and_settled_field(self):
        value = protocol()
        self.assertEqual(value["source"]["action_field"], "actions[*].executed")
        self.assertTrue(value["source"]["nominal_translational_field_prohibited"])
        self.assertIn("after_20_settling_actions", value["pairing"]["field"])

    def test_motion_preserving_feasible(self):
        result = classify_fast_feasibility(valid_metrics(), protocol())
        self.assertEqual(result["primary_outcome"], "CONTACT_PREVENTION_FEASIBLE")
        self.assertEqual(result["safety_mechanism"], "MOTION_PRESERVING_CORRECTION")

    def test_stop_only_keeps_prevention_outcome(self):
        metrics = valid_metrics()
        metrics["active_cartesian_path_length_m"] = 0.00001
        result = classify_fast_feasibility(metrics, protocol())
        self.assertEqual(result["primary_outcome"], "CONTACT_PREVENTION_FEASIBLE")
        self.assertEqual(result["safety_mechanism"], "STOP_ONLY")
        self.assertFalse(
            result["motion_preservation_checks"]["material_cartesian_motion"]
        )

    def test_reproduced_baseline_and_psf_literal_contact_is_failure(self):
        metrics = valid_metrics()
        metrics["psf_link56_contact_present"] = True
        metrics["psf_any_robot_selected_obstacle_contact_present"] = True
        result = classify_fast_feasibility(metrics, protocol())
        self.assertEqual(result["primary_outcome"], "CONTACT_PREVENTION_FAILED")
        self.assertFalse(result["contact_prevention_feasible"])

    def test_psf_contact_early_terminal_is_decisive_not_apparatus_failure(self):
        metrics = valid_metrics()
        metrics.update(
            {
                "psf_exposure_complete": False,
                "psf_static_obstacle_admissible": False,
                "psf_all_qp_solved": False,
                "psf_all_qp_postchecks_passed": False,
                "psf_all_joint_limit_postchecks_passed": False,
                "psf_all_post_state_field_queries_valid_and_positive": False,
                "psf_link56_contact_present": True,
                "psf_any_robot_selected_obstacle_contact_present": True,
                "psf_filter_update_count": 39,
                "psf_physics_substep_count": 193,
                "psf_monitor_observed_physics_substep_count": 193,
                "psf_qp_solve_count": 39,
                "psf_qp_postcheck_count": 39,
                "psf_joint_limit_postcheck_count": 39,
                "psf_issued_command_bound_check_count": 39,
                "psf_nominal_dynamic_bound_check_count": 39,
                "psf_tracking_observation_count": 193,
                "psf_pre_filter_field_observation_count": 39,
                "psf_pre_filter_field_query_count": 39 * 1531,
                "psf_post_state_field_observation_count": 0,
                "psf_post_state_field_query_count": 0,
                "psf_precontact_tracking_observation_count": 192,
            }
        )
        result = classify_fast_feasibility(metrics, protocol())
        self.assertEqual(result["primary_outcome"], "CONTACT_PREVENTION_FAILED")
        self.assertTrue(result["apparatus_valid"])

    def test_shifted_robot_contact_is_failure(self):
        metrics = valid_metrics()
        metrics["psf_any_robot_selected_obstacle_contact_present"] = True
        result = classify_fast_feasibility(metrics, protocol())
        self.assertEqual(result["primary_outcome"], "CONTACT_PREVENTION_FAILED")
        self.assertIn(
            "psf_shifted_or_other_robot_contact_present",
            result["classification_reasons"],
        )

    def test_noncontact_with_nonpositive_coverage_is_uncertified(self):
        metrics = valid_metrics()
        metrics["psf_conservative_full_robot_surface_clearance_lower_bound_m"] = 0.0
        result = classify_fast_feasibility(metrics, protocol())
        self.assertEqual(result["primary_outcome"], "UNCERTIFIED_CLEARANCE")
        self.assertNotEqual(result["primary_outcome"], "CONTACT_PREVENTION_FAILED")

    def test_coarse_full_robot_Dsim_is_diagnostic_only(self):
        metrics = valid_metrics()
        metrics["psf_minimum_D_sim_m"] = -10.0
        result = classify_fast_feasibility(metrics, protocol())
        self.assertEqual(result["primary_outcome"], "CONTACT_PREVENTION_FEASIBLE")

    def test_missing_adapter_contact_is_inconclusive(self):
        metrics = valid_metrics()
        metrics["adapter_link56_contact_present"] = False
        result = classify_fast_feasibility(metrics, protocol())
        self.assertEqual(result["primary_outcome"], "INCONCLUSIVE")

    def test_earlier_other_robot_contact_is_inconclusive(self):
        metrics = valid_metrics()
        metrics["adapter_first_selected_obstacle_contact_is_link56"] = False
        result = classify_fast_feasibility(metrics, protocol())
        self.assertEqual(result["primary_outcome"], "INCONCLUSIVE")
        self.assertFalse(result["contact_prevention_feasible"])

    def test_no_material_precontact_activation_is_inconclusive(self):
        metrics = valid_metrics()
        metrics["psf_nominal_cbf_activation_with_material_correction_before_adapter_contact"] = False
        metrics["psf_activation_update_count_before_adapter_contact"] = 0
        metrics["psf_first_activation_physical_boundary"] = 4701
        metrics["psf_minimum_nominal_cbf_residual_before_adapter_contact_m2_per_s"] = -1.0e-8
        metrics["psf_maximum_activation_correction_norm_before_adapter_contact_rad_s"] = 0.0
        result = classify_fast_feasibility(metrics, protocol())
        self.assertEqual(result["primary_outcome"], "INCONCLUSIVE")
        self.assertIn("not_materially_activated", result["classification_reasons"][0])

    def test_joint_bound_only_difference_is_rejected_as_confounded(self):
        metrics = valid_metrics()
        metrics["psf_all_nominal_commands_within_dynamic_joint_bounds"] = False
        metrics["psf_nominal_dynamic_bound_violation_count"] = 1
        result = classify_fast_feasibility(metrics, protocol())
        self.assertEqual(result["primary_outcome"], "APPARATUS_FAILURE")
        self.assertIn(
            "psf_nominal_dynamic_joint_bounds_confounded",
            result["apparatus_failure_reasons"],
        )

    def test_post_state_field_query_population_must_be_exhaustive(self):
        metrics = valid_metrics()
        metrics["psf_post_state_field_query_count"] -= 1
        result = classify_fast_feasibility(metrics, protocol())
        self.assertEqual(result["primary_outcome"], "APPARATUS_FAILURE")
        self.assertIn(
            "psf_post_state_field_query_population_differs",
            result["apparatus_failure_reasons"],
        )

    def test_pre_filter_field_query_population_must_be_exhaustive(self):
        metrics = valid_metrics()
        metrics["psf_pre_filter_field_query_count"] -= 1
        result = classify_fast_feasibility(metrics, protocol())
        self.assertEqual(result["primary_outcome"], "APPARATUS_FAILURE")
        self.assertIn(
            "psf_pre_filter_field_query_population_differs",
            result["apparatus_failure_reasons"],
        )

    def test_contact_prefix_monitor_count_is_reconstructed_not_trusted(self):
        metrics = valid_metrics()
        metrics.update(
            {
                "psf_exposure_complete": False,
                "psf_static_obstacle_admissible": False,
                "psf_all_qp_solved": False,
                "psf_all_qp_postchecks_passed": False,
                "psf_all_joint_limit_postchecks_passed": False,
                "psf_all_post_state_field_queries_valid_and_positive": False,
                "psf_link56_contact_present": True,
                "psf_any_robot_selected_obstacle_contact_present": True,
                "psf_filter_update_count": 39,
                "psf_physics_substep_count": 193,
                "psf_monitor_observed_physics_substep_count": 192,
                "psf_qp_solve_count": 39,
                "psf_qp_postcheck_count": 39,
                "psf_joint_limit_postcheck_count": 39,
                "psf_issued_command_bound_check_count": 39,
                "psf_nominal_dynamic_bound_check_count": 39,
                "psf_tracking_observation_count": 193,
                "psf_pre_filter_field_observation_count": 39,
                "psf_pre_filter_field_query_count": 39 * 1531,
                "psf_post_state_field_observation_count": 0,
                "psf_post_state_field_query_count": 0,
                "psf_precontact_tracking_observation_count": 192,
            }
        )
        result = classify_fast_feasibility(metrics, protocol())
        self.assertEqual(result["primary_outcome"], "APPARATUS_FAILURE")
        self.assertIn(
            "psf_monitor_observed_physics_substep_count_differs_from_trace",
            result["apparatus_failure_reasons"],
        )

    def test_contact_prefix_filter_physics_cadence_is_reconstructed(self):
        metrics = valid_metrics()
        metrics.update(
            {
                "psf_exposure_complete": False,
                "psf_static_obstacle_admissible": False,
                "psf_all_qp_solved": False,
                "psf_all_qp_postchecks_passed": False,
                "psf_all_joint_limit_postchecks_passed": False,
                "psf_all_post_state_field_queries_valid_and_positive": False,
                "psf_link56_contact_present": True,
                "psf_any_robot_selected_obstacle_contact_present": True,
                "psf_filter_update_count": 40,
                "psf_physics_substep_count": 1,
                "psf_monitor_observed_physics_substep_count": 1,
                "psf_qp_solve_count": 40,
                "psf_qp_postcheck_count": 40,
                "psf_joint_limit_postcheck_count": 40,
                "psf_issued_command_bound_check_count": 40,
                "psf_nominal_dynamic_bound_check_count": 40,
                "psf_tracking_observation_count": 1,
                "psf_pre_filter_field_observation_count": 40,
                "psf_pre_filter_field_query_count": 40 * 1531,
                "psf_post_state_field_observation_count": 0,
                "psf_post_state_field_query_count": 0,
                "psf_precontact_tracking_observation_count": 0,
            }
        )
        result = classify_fast_feasibility(metrics, protocol())
        self.assertEqual(result["primary_outcome"], "APPARATUS_FAILURE")
        self.assertIn(
            "psf_contact_prefix_filter_physics_cadence_differs",
            result["apparatus_failure_reasons"],
        )

    def test_postcontact_baseline_tracking_is_not_an_input(self):
        metrics = valid_metrics()
        self.assertNotIn("adapter_tracking_linf_rad_s", metrics)
        result = classify_fast_feasibility(metrics, protocol())
        self.assertTrue(result["apparatus_valid"])

    def test_tracking_is_diagnostic_for_exploratory_contact_outcome(self):
        metrics = valid_metrics()
        metrics["adapter_precontact_tracking_linf_rad_s"] = 2.0
        metrics["adapter_precontact_tracking_rmse_rad_s"] = 0.2
        metrics["psf_tracking_linf_rad_s"] = 2.0
        metrics["psf_tracking_rmse_rad_s"] = 0.2
        result = classify_fast_feasibility(metrics, protocol())
        self.assertEqual(result["primary_outcome"], "CONTACT_PREVENTION_FEASIBLE")
        self.assertTrue(result["tracking_diagnostic_only"])
        self.assertFalse(result["tracking_certified"])

    def test_explicit_qp_boolean_and_count_are_both_required(self):
        for field in (
            "psf_all_qp_solved",
            "psf_all_qp_postchecks_passed",
            "psf_all_joint_limit_postchecks_passed",
            "psf_all_post_state_field_queries_valid_and_positive",
            "all_issued_commands_within_physical_bounds",
        ):
            metrics = valid_metrics()
            metrics[field] = False
            result = classify_fast_feasibility(metrics, protocol())
            self.assertEqual(result["primary_outcome"], "APPARATUS_FAILURE")
        metrics = valid_metrics()
        metrics["psf_qp_postcheck_count"] = 39
        self.assertEqual(
            classify_fast_feasibility(metrics, protocol())["primary_outcome"],
            "APPARATUS_FAILURE",
        )

    def test_type_confusion_fails_closed(self):
        metrics = valid_metrics()
        metrics["psf_qp_solve_count"] = True
        with self.assertRaises(FastFeasibilityError):
            classify_fast_feasibility(metrics, protocol())

    def test_protocol_label_mutation_is_rejected(self):
        value = copy.deepcopy(protocol())
        value["classification"]["primary_outcomes"].remove(
            "CONTACT_PREVENTION_FAILED"
        )
        with self.assertRaises(FastFeasibilityError):
            validate_fast_feasibility_protocol(value)

    def test_exploratory_execution_mutation_is_rejected(self):
        value = copy.deepcopy(protocol())
        value["exploratory_execution"]["qp_max_iterations"] = 10000
        with self.assertRaises(FastFeasibilityError):
            validate_fast_feasibility_protocol(value)


if __name__ == "__main__":
    unittest.main()
