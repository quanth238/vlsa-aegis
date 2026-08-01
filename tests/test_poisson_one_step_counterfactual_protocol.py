from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from main.poisson_fullbody.feasibility_protocol import load_feasibility_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/vlsa_poisson_one_step_counterfactual.v1.json"


def _load_without_duplicate_keys(path: Path):
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key %r" % key)
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("non-finite JSON constant %r" % value)
        ),
    )


class OneStepCounterfactualProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = _load_without_duplicate_keys(PROTOCOL_PATH)

    def test_runtime_binding_is_current_and_semantically_valid(self):
        prerequisites = self.protocol["prerequisites"]
        selection_path = ROOT / prerequisites["selection_protocol_relative_path"]
        selection = _load_without_duplicate_keys(selection_path)
        self.assertEqual(
            hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            prerequisites["selection_raw_file_sha256"],
        )
        self.assertEqual(
            selection["schema_version"], prerequisites["selection_schema_version"]
        )
        self.assertEqual(
            selection["protocol_id"], prerequisites["selection_protocol_id"]
        )
        runtime_path = ROOT / prerequisites["runtime_protocol_relative_path"]
        runtime, hashes = load_feasibility_protocol(runtime_path)
        self.assertEqual(
            hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
            prerequisites["runtime_raw_file_sha256"],
        )
        self.assertEqual(
            hashes.protocol_sha256,
            prerequisites["runtime_semantic_protocol_sha256"],
        )
        self.assertEqual(
            hashes.parameter_block_sha256,
            prerequisites["runtime_parameter_block_sha256"],
        )
        self.assertEqual(runtime["schema_version"], prerequisites["runtime_schema_version"])
        self.assertEqual(runtime["protocol_id"], prerequisites["runtime_protocol_id"])
        boundary = self.protocol["counterfactual_boundary"]
        acceptance = self.protocol["acceptance"]
        estimator = self.protocol["nominal_velocity_estimator"]
        self.assertEqual(
            boundary["physics_timestep_s"], runtime["cadence"]["physics_timestep_s"]
        )
        self.assertEqual(
            boundary["physics_substeps_per_filter_update"],
            runtime["cadence"]["active"]["physics_substeps_per_filter_update"],
        )
        self.assertEqual(
            boundary["counterfactual_horizon_s"],
            boundary["physics_timestep_s"]
            * boundary["counterfactual_horizon_substeps"],
        )
        self.assertEqual(
            estimator["arm_velocity_lower_rad_s"],
            runtime["qp"]["velocity_lower_rad_s"],
        )
        self.assertEqual(
            estimator["arm_velocity_upper_rad_s"],
            runtime["qp"]["velocity_upper_rad_s"],
        )
        self.assertEqual(
            acceptance["qp_safe_residual_minimum"],
            -runtime["qp"]["postcheck_cbf_tolerance"],
        )
        self.assertEqual(
            acceptance["joint_velocity_tracking_linf_max_rad_s"],
            runtime["admissibility"]["max_joint_velocity_tracking_linf_rad_s"],
        )
        self.assertEqual(
            acceptance["joint_velocity_tracking_rmse_max_rad_s"],
            runtime["admissibility"]["max_joint_velocity_tracking_rmse_rad_s"],
        )
        self.assertEqual(
            self.protocol["qp_execution"]["joint_position_constraints"],
            "exact_runtime_v2_joint_limit_margin_and_gain",
        )
        self.assertIn(
            "physical_model_v3_compiled_identity_and_exact_state_layout_contract",
            prerequisites["required_shadow_construction_bindings"],
        )

    def test_warning_authority_is_phase_correct_and_cbf_specific(self):
        authority = self.protocol["prerequisites"]["identification_authorization"]
        self.assertEqual(
            authority["primary_registered_warning_signal_kind"], "cbf_lhs_negative"
        )
        self.assertTrue(authority["invalid_query_warning_cannot_authorize"])
        self.assertTrue(
            authority["primary_warning_robot_geom_must_equal_first_contact_robot_geom"]
        )
        self.assertTrue(authority["require_positive_phase_correct_lead"])
        self.assertTrue(authority["require_next_filter_boundary_strictly_before_C"])
        self.assertEqual(
            authority["next_filter_boundary_formula"],
            "ceil(warning_physical_boundary/5)*5",
        )

    def test_boundary_and_velocity_estimator_are_phase_correct(self):
        contact = self.protocol["contact_boundary"]
        boundary = self.protocol["counterfactual_boundary"]
        estimator = self.protocol["nominal_velocity_estimator"]
        self.assertEqual(
            contact["trace_callback_semantics"],
            "callback_index_k_is_post_integration_state_at_physical_boundary_k_plus_1",
        )
        self.assertEqual(boundary["warning_boundary_symbol"], "W")
        self.assertEqual(
            boundary["selection_rule"],
            "first_registered_filter_boundary_at_or_after_W_and_strictly_before_C",
        )
        self.assertEqual(
            boundary["formula"],
            "W=warning_observation_index+1;B=ceil(W/5)*5;B<C",
        )
        self.assertEqual(boundary["physics_substeps_per_filter_update"], 5)
        self.assertEqual(boundary["counterfactual_horizon_substeps"], 5)
        self.assertEqual(boundary["counterfactual_horizon_s"], 0.01)
        self.assertEqual(estimator["method"], "mujoco_mj_differentiatePos_full_nv")
        self.assertEqual(estimator["interval_s"], 0.01)
        self.assertEqual(estimator["expected_arm_qpos_indices"], list(range(7)))
        self.assertEqual(estimator["out_of_bounds_policy"], "inadmissible_no_hidden_clipping")
        self.assertEqual(estimator["arm_velocity_lower_rad_s"], [-0.5] * 7)
        self.assertEqual(estimator["arm_velocity_upper_rad_s"], [0.5] * 7)
        self.assertTrue(estimator["full_nv_producer_diagnostic_retained"])
        self.assertEqual(
            estimator["independent_claim_bearing_reconstruction"],
            "seven_scalar_hinge_arm_slice_only",
        )
        self.assertEqual(
            estimator["nonarm_values_validation_scope"],
            "producer_diagnostic_not_independently_reconstructed_no_claim",
        )

    def test_joint_velocity_controller_authority_is_frozen_to_h100_evidence(self):
        authority = self.protocol["joint_velocity_controller_authority"]
        contract = authority["expected_restore_controller_contract"]
        self.assertTrue(
            authority["independent_installed_source_rehash_required"]
        )
        self.assertEqual(contract["control_frequency_hz"], 100)
        self.assertEqual(contract["control_timestep_s"], 0.01)
        self.assertEqual(contract["physics_timestep_s"], 0.002)
        self.assertEqual(contract["physics_substeps_per_control"], 5)
        self.assertEqual(contract["normalized_input_lower"], [-1.0] * 7)
        self.assertEqual(contract["normalized_input_upper"], [1.0] * 7)
        self.assertEqual(contract["physical_output_lower_rad_s"], [-0.5] * 7)
        self.assertEqual(contract["physical_output_upper_rad_s"], [0.5] * 7)
        self.assertEqual(contract["arm_joint_names"], [
            "robot0_joint%d" % index for index in range(1, 8)
        ])
        self.assertEqual(contract["arm_actuator_names"], [
            "robot0_torq_j%d" % index for index in range(1, 8)
        ])
        for field in (
            "controller_implementation_file_sha256",
            "controller_configuration_file_sha256",
            "panda_robot_xml_file_sha256",
        ):
            self.assertRegex(contract[field], r"^[0-9a-f]{64}$")

    def test_gate_has_local_claim_scope_and_no_post_hoc_prediction_tolerance(self):
        acceptance = self.protocol["acceptance"]
        self.assertIsNone(acceptance["prediction_error_acceptance_threshold"])
        self.assertEqual(acceptance["maximum_safe_arm_invalid_field_queries"], 0)
        self.assertEqual(
            acceptance["prediction_error_policy"],
            "diagnostic_only_no_post_hoc_tolerance",
        )
        self.assertTrue(acceptance["require_nominal_arm_target_contact_for_strong_result"])
        self.assertTrue(
            acceptance["require_safe_arm_all_robot_to_selected_obstacle_contacts_absent"]
        )
        self.assertTrue(
            acceptance[
                "require_safe_arm_minimum_D_sim_strictly_positive_at_every_substep"
            ]
        )
        self.assertEqual(acceptance["minimum_filter_correction_norm_rad_s"], 1e-4)
        self.assertEqual(acceptance["minimum_safe_command_norm_rad_s"], 0.05)
        self.assertEqual(
            acceptance["minimum_safe_to_nominal_command_norm_ratio"], 0.25
        )
        self.assertEqual(
            acceptance["minimum_safe_measured_joint_motion_rad"], 1e-4
        )
        self.assertEqual(
            acceptance[
                "minimum_safe_measured_motion_to_command_integral_ratio"
            ],
            0.25,
        )
        self.assertEqual(
            acceptance["local_motion_interpretation"],
            "nonzero_local_joint_motion_not_nominal_direction_progress_or_task_success",
        )
        self.assertEqual(
            acceptance["static_field_admissibility_scopes"],
            {
                "poisson_filtered_velocity": (
                    "boundary_B_through_B_plus_5_inclusive"
                ),
                "nominal_with_selected_obstacle_contact": (
                    "boundary_B_through_strictly_before_phase_correct_C_any_nom_"
                    "excluding_contact_consequence"
                ),
                "nominal_without_selected_obstacle_contact": (
                    "boundary_B_through_B_plus_5_inclusive"
                ),
            },
        )
        self.assertEqual(
            acceptance["tracking_admissibility_scopes"],
            {
                "poisson_filtered_velocity": (
                    "physical_boundaries_B_plus_1_through_B_plus_5_inclusive"
                ),
                "nominal_with_selected_obstacle_contact": (
                    "physical_boundaries_B_plus_1_through_strictly_before_"
                    "phase_correct_C_any_nom_excluding_contact_consequence"
                ),
                "nominal_without_selected_obstacle_contact": (
                    "physical_boundaries_B_plus_1_through_B_plus_5_inclusive"
                ),
            },
        )
        self.assertEqual(
            acceptance["trend_only_gate_status"],
            "passes_stage_13_as_one_interval_local_feasibility_only_when_all_"
            "local_and_active_trend_conditions_pass",
        )
        self.assertEqual(
            acceptance["trend_only_definition"]["combination"],
            "all_registered_trend_conditions_and_no_nominal_target_contact",
        )
        self.assertEqual(
            acceptance["trend_only_quality_prerequisites"],
            [
                "complete_exposure",
                "exact_paired_start",
                "qp_valid",
                "both_tracking_valid",
                "static_field_admissible",
            ],
        )

    def test_safe_start_exact_samples_and_two_arm_exposure_are_mandatory(self):
        preflight = self.protocol["boundary_B_preflight"]
        self.assertTrue(preflight["require_query_count_equal_bound_sample_count"])
        self.assertTrue(preflight["require_all_queries_valid"])
        self.assertTrue(preflight["require_every_h_strictly_positive"])
        self.assertTrue(preflight["require_minimum_D_sim_strictly_positive"])
        self.assertTrue(
            preflight["require_zero_robot_to_selected_obstacle_physical_contacts"]
        )
        self.assertTrue(
            preflight[
                "require_selected_obstacle_body_linear_and_angular_speed_within_runtime_v2_thresholds"
            ]
        )
        self.assertEqual(
            preflight["failure_policy"],
            "apparatus_failure_no_QP_no_arm_physics_no_scientific_interpretation",
        )
        exposure = self.protocol["exposure_and_tracking"]
        self.assertEqual(
            exposure["applies_to_arms"],
            ["nominal_measured_velocity", "poisson_filtered_velocity"],
        )
        self.assertEqual(exposure["required_substep_count_per_arm"], 5)
        self.assertEqual(
            exposure["required_completed_physical_boundaries"],
            "B_plus_1_through_B_plus_5_inclusive",
        )
        self.assertTrue(
            exposure[
                "require_each_arm_linf_and_cumulative_rmse_within_registered_thresholds"
            ]
        )
        self.assertIn(
            "complete_fixed_counterfactual_exposure",
            self.protocol["arms"]["nominal_measured_velocity"][
                "post_B_invalid_field_query_policy"
            ],
        )
        self.assertIn(
            "full_robot_measurement_sampling_exact_sample_ledger_and_sha256",
            self.protocol["prerequisites"][
                "required_shadow_construction_bindings"
            ],
        )
        self.assertIn(
            "ordered_full_robot_sample_distance_m",
            self.protocol["required_evidence"]["per_physics_substep"],
        )
        self.assertEqual(
            self.protocol["required_evidence"][
                "at_boundary_B_simulator_clearance"
            ],
            [
                "ordered_full_robot_sample_distance_m",
                "registered_full_robot_sample_count",
                "registered_full_robot_sample_ledger_sha256",
                "certified_full_robot_coverage_radius_m",
                "minimum_D_sim_m",
            ],
        )

    def test_trend_acceptance_evidence_is_not_mislabeled_diagnostic(self):
        evidence = self.protocol["required_evidence"]
        trend = evidence["trend_reference_acceptance_evidence"]
        diagnostic = evidence["diagnostic_only"]
        self.assertEqual(
            trend,
            [
                "first_order_predicted_h_after_horizon_m2",
                "actual_h_after_each_physics_substep_m2",
                "nominal_D_sim_at_B_m",
                "nominal_minimum_D_sim_over_horizon_m",
                "psf_h_after_horizon_m2",
                "psf_minus_nominal_h_after_horizon_m2",
                "psf_warning_sample_h_strictly_greater_than_nominal_at_horizon",
            ],
        )
        self.assertEqual(
            diagnostic, ["prediction_error_m2_without_acceptance_threshold"]
        )
        self.assertFalse(set(trend).intersection(diagnostic))

    def test_qp_is_hard_exhaustive_and_gripper_is_source_bound(self):
        qp = self.protocol["qp_execution"]
        self.assertEqual(qp["solve_count"], 1)
        self.assertEqual(qp["hold_solution_physics_substeps"], 5)
        self.assertTrue(qp["hard_constraints"])
        self.assertFalse(qp["slack_enabled"])
        self.assertTrue(qp["require_solved_status"])
        self.assertTrue(qp["require_independent_CBF_residual_reconstruction"])
        reference = qp["independent_postproducer_reference_check"]
        self.assertEqual(reference["solver"], "cvxpy_osqp")
        self.assertEqual(reference["eps_abs"], 1e-9)
        self.assertEqual(reference["eps_rel"], 1e-9)
        self.assertEqual(reference["max_iterations"], 20000)
        self.assertEqual(reference["qdot_linf_tolerance_rad_s"], 2e-5)
        self.assertTrue(reference["required_before_scientific_interpretation"])
        self.assertEqual(
            qp["constraint_rows"],
            "exactly_one_CBF_row_per_bound_ordered_protected_sample",
        )
        self.assertEqual(
            qp["expected_arm_q_min_rad"],
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        )
        self.assertEqual(
            qp["expected_arm_q_max_rad"],
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        )
        self.assertEqual(qp["joint_limit_match_absolute_tolerance_rad"], 1e-12)
        gripper = self.protocol["arms"]["gripper_policy"]
        self.assertEqual(gripper["source_action_index_formula"], "floor(B/25)")
        self.assertIn("source_action_7d", gripper["required_evidence"])
        self.assertIn("source_action_sha256", gripper["required_evidence"])
        self.assertIn("exact_gripper_value_sha256", gripper["required_evidence"])

    def test_result_is_allocation_only_atomic_and_receipt_last(self):
        contract = self.protocol["result_contract"]
        self.assertEqual(
            contract["execution_environment"], "VinUni_H100_Slurm_allocation_only"
        )
        self.assertEqual(
            contract["write_policy"], "atomic_only_after_complete_core_validation"
        )
        self.assertEqual(contract["receipt_policy"], "write_validation_receipt_last")
        self.assertEqual(
            contract["resume_policy"],
            "existing_output_path_rejected_new_immutable_run_required",
        )
        self.assertEqual(contract["policy_server"], "forbidden")
        self.assertEqual(
            contract["independent_validator"],
            "separate_artifact_consumer_required_after_producer_exit",
        )
        self.assertIn("COMPLETED", contract["post_job_acceptance"])
        self.assertIn("slurm_job_id_host_and_single_H100_device", contract["required_provenance"])
        self.assertIn("not_task_success", contract["claim_scope"])
        self.assertIn("without_QP_or_paired_arm_physics", contract["preflight_inadmissible_terminal"])


if __name__ == "__main__":
    unittest.main()
