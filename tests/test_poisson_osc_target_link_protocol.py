import copy
import hashlib
import inspect
import json
from pathlib import Path
import unittest


from main.poisson_fullbody.osc_arm_link_canary import (
    OscArmLinkCanaryError,
    TARGET_LINK_ALLOWED_CASE_IDS,
    TARGET_LINK_ALLOWED_CASE_REGISTRY,
    TARGET_LINK_CLASSIFICATION_SCHEMA,
    TARGET_LINK_FROZEN_RUNTIME_PARAMETER_SHA256,
    TARGET_LINK_PROTECTED_LINK_BODY_NAMES,
    TARGET_LINK_PROTOCOL_ID,
    TARGET_LINK_PROTOCOL_SCHEMA,
    TARGET_LINK_REQUIRED_ACCEPTANCE,
    TARGET_LINK_REQUIRED_APPARATUS_METRICS,
    TARGET_LINK_TREATMENT_ARM,
    classify_osc_target_link_canary,
    classify_osc_target_link_pair,
    target_link_allowed_case_registry_payload,
    validate_osc_target_link_canary_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = {
    "vlsa-t1-goal-ii-t0-e05": (
        ROOT / "configs" / "vlsa_poisson_osc_target_link_e05.v4.json"
    ),
    "vlsa-t1-goal-ii-t3-e42": (
        ROOT / "configs" / "vlsa_poisson_osc_target_link_e42.v4.json"
    ),
}
CONFIG_SHA256 = {
    "vlsa-t1-goal-ii-t0-e05": (
        "f5569ee5f3e5d45ae7d18cb02309e1f962a9801646c70a0356c362c0b43d5cc1"
    ),
    "vlsa-t1-goal-ii-t3-e42": (
        "f6c6216758bee2c04ae57e727b46708377815bd9f59fecae9183a1b24f19996e"
    ),
}


class OscTargetLinkProtocolTests(unittest.TestCase):
    def protocol(self, case_id):
        registry = target_link_allowed_case_registry_payload()
        row = registry[case_id]
        return {
            "schema_version": TARGET_LINK_PROTOCOL_SCHEMA,
            "protocol_id": TARGET_LINK_PROTOCOL_ID,
            "allowed_case_registry": registry,
            "case": {
                "case_id": case_id,
                "selected_obstacle_name": row["selected_obstacle_name"],
                "selected_obstacle_root_body_name": row[
                    "selected_obstacle_root_body_name"
                ],
                "literal_link56_body_names": ["robot0_link5", "robot0_link6"],
                "target_link_body_names": row["target_link_body_names"],
                "historical_literal_link_contact_body_names": row[
                    "historical_literal_link_contact_body_names"
                ],
                "historical_first_target_link_contact_source_action": row[
                    "historical_first_target_link_contact_source_action"
                ],
                "historical_prior_completed_action_endpoint": row[
                    "historical_prior_completed_action_endpoint"
                ],
                "historical_car_first_crossing_source_action": row[
                    "historical_car_first_crossing_source_action"
                ],
                "historical_executed_action_count": row[
                    "historical_executed_action_count"
                ],
                "selection_split": row["selection_split"],
            },
            "historical_control": {
                "source_arm": "pi05_plus_aegis_translational",
                "case_id": case_id,
                "rerun_baseline": False,
                "complete_scientific_result_required": True,
                "causal_role": (
                    "immutable_archived_control_no_simultaneous_or_posthoc_"
                    "baseline_rerun"
                ),
                "archived_direct_target_link_contact": True,
                "archived_car_failure": True,
                "archived_task_success": True,
                "executed_action_count": row["historical_executed_action_count"],
                "result_file_sha256": row["historical_result_file_sha256"],
                "result_payload_sha256": row[
                    "historical_result_payload_sha256"
                ],
                "executed_action_sequence_sha256": row[
                    "historical_executed_action_sequence_sha256"
                ],
            },
            "execution": {
                "treatment_arm": TARGET_LINK_TREATMENT_ARM,
                "controller": "OSC_POSE",
                "high_level_frequency_hz": 20,
                "physics_frequency_hz": 500,
                "physics_substeps_per_action": 25,
                "shield_location": (
                    "after_nominal_osc_torque_before_every_mujoco_step"
                ),
                "pre_divergence_action_source": "archived_aegis_executed_actions",
                "post_divergence_current_chunk_source": (
                    "archived_pi05_raw_chunk_reprocessed_by_aegis_on_treatment_"
                    "observation"
                ),
                "post_chunk_source": (
                    "fresh_pi05_and_aegis_from_treatment_observation"
                ),
                "historical_contact_boundary_reference": (
                    "allowed_case_registry_historical_first_target_link_contact_"
                    "source_action"
                ),
                "historical_prior_endpoint_reference": (
                    "allowed_case_registry_historical_prior_completed_action_"
                    "endpoint"
                ),
                "same_source_action_contact_exception": (
                    "allow_only_at_exact_callback_action_0_0_after_exact_prior_"
                    "completed_action_endpoint_parity_and_contact_free_prior_endpoint"
                ),
                "stop_on_native_task_success": True,
                "stop_on_any_robot_selected_obstacle_contact": True,
                "stop_on_link56_external_nonrobot_contact": True,
                "contact_scope": (
                    "union_of_all_authoritative_robot_collision_geoms_vs_"
                    "selected_obstacle_and_literal_link5_link6_collision_geoms_"
                    "vs_every_external_nonrobot_collision_geom"
                ),
                "maximum_action_count": 300,
                "replan_steps": 5,
                "model_action_horizon": 10,
            },
            "paper_car_measurement": {
                "metric": (
                    "SafeLIBERO_Table1_active_obstacle_L1_displacement_at_"
                    "completed_20Hz_action_endpoints"
                ),
                "position_source": "selected_obstacle_pos_native_observation",
                "root_body_binding": (
                    "env_obj_body_id_equals_contact_authority_root_body_id"
                ),
                "native_observable_binding": (
                    "returned_observation_equals_observable_value_and_"
                    "observation_cache"
                ),
                "live_root_position_role": (
                    "phase_diagnostic_only_not_authority_or_paper_car_metric"
                ),
                "post_integration_forwarded_pose_role": (
                    "phase_diagnostic_only_not_paper_car_metric"
                ),
            },
            "shield": {
                "field_geometry": "static_simulator_oracle_selected_obstacle",
                "field_frame": (
                    "world_frame_frozen_at_post_settling_selected_obstacle_pose"
                ),
                "protected_link_body_names": list(
                    TARGET_LINK_PROTECTED_LINK_BODY_NAMES
                ),
                "protected_samples": (
                    "common_protected_link5_link6_collision_surface_samples"
                ),
                "field_bundle_samples": (
                    "common_protected_link5_link6_collision_surface_samples"
                ),
                "binding_scope": (
                    "common_protected_link5_link6_collision_surfaces_vs_"
                    "selected_obstacle"
                ),
                "shield_body_authority": (
                    "exact_protocol_protected_link_body_names_independent_of_"
                    "case_task_and_historical_target_attribution_no_descendants"
                ),
                "shield_geometry_selection": (
                    "collision_enabled_geoms_directly_attached_to_exact_common_"
                    "protected_link_bodies"
                ),
                "contact_monitor_population": (
                    "all_authoritative_robot_collision_surfaces_vs_selected_"
                    "obstacle_union_literal_link56_vs_every_external_nonrobot_"
                    "collision_surface"
                ),
                "point_velocity_scope": (
                    "seven_registered_panda_arm_qvels_structurally_verified_"
                    "complete_for_common_protected_link_samples"
                ),
                "decision_variable": (
                    "seven_registered_panda_arm_torque_deltas"
                ),
                "nonarm_control_policy": "nominal_nonarm_controls_unchanged",
                "sample_binding": (
                    "same_ordered_common_protected_link_surface_sample_ledger_"
                    "for_field_queries_jacobians_and_constraint_rows"
                ),
                "constraint": "grad_h_J_qvel_next_plus_alpha_h_ge_margin",
                "alpha_gain_per_s": 5,
                "margin_m2_per_s": 0,
                "hard_constraints": True,
                "slack": False,
                "failure_policy": "stop_before_physics_and_retain_artifact",
                "sensitivity_method": (
                    "official_mujoco_one_step_clone_two_resolution_central_"
                    "difference"
                ),
                "torque_epsilon_nm": 0.001,
                "sensitivity_agreement_atol": 1e-8,
                "sensitivity_agreement_rtol": 0.001,
                "maximum_sensitivity_condition_number": 100000000,
                "actual_cbf_residual_tolerance_m2_per_s": 0.0005,
                "solver_eps_abs": 1e-8,
                "solver_eps_rel": 1e-8,
                "solver_max_iterations": 50000,
                "no_clipping_fallback_or_cached_command": True,
            },
            "acceptance": {
                "material_torque_correction_l2_nm": 0.01,
                "minimum_first_material_exact_cbf_residual_improvement_m2_per_s": 1e-8,
                "minimum_first_material_exact_next_qvel_change_l2_rad_s": 1e-5,
                "minimum_post_correction_joint_motion_integral_rad": 0.02,
                "minimum_post_correction_eef_path_length_m": 0.02,
                "paper_car_l1_displacement_threshold_m": 0.001,
                "all_required": list(TARGET_LINK_REQUIRED_ACCEPTANCE),
            },
            "selection": {
                "study_protocol_relative_path": (
                    "configs/vlsa_poisson_arm_contact_study.v1.json"
                ),
                "study_protocol_file_sha256": (
                    "2c940ed62876c7133011de8a21c0b12b6469f6543d6535074da7d608a9730221"
                ),
                "manifest_relative_path": (
                    "manifests/vlsa_poisson_arm_contact_165.v1.jsonl"
                ),
                "manifest_file_sha256": (
                    "80678be7bbef6be8027c4d276e3a0114b04409388df6ee74399d2e9f301a2935"
                ),
                "manifest_receipt_relative_path": (
                    "manifests/vlsa_poisson_arm_contact_165.v1.receipt.json"
                ),
                "manifest_receipt_file_sha256": (
                    "e3795e9c29f9841aaa3265de2d45fcf39ce10bfb5a0b7d47cad2a238d9f42955"
                ),
                "remote_artifact_receipt_relative_path": row[
                    "remote_artifact_receipt_relative_path"
                ],
                "remote_artifact_receipt_file_sha256": row[
                    "remote_artifact_receipt_file_sha256"
                ],
                "case_row_sha256": row["selection_case_row_sha256"],
            },
            "online_policy": {
                "checkpoint": (
                    "/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/"
                    "pi05_libero"
                ),
                "checkpoint_receipt": (
                    "/mnt/data/quanth/experiments/vlsa-aegis-table1/checkpoint-"
                    "receipts/vlsa-table1-pi05-hash-contact-authority-"
                    "20260718a.json"
                ),
                "checkpoint_receipt_schema_version": (
                    "vlsa_table1_pi05_hash_receipt.v1"
                ),
                "checkpoint_receipt_sha256": (
                    "423c10b3435b84dfb694c309c7d2684cc9457c4616878043618847e9e75df7b9"
                ),
                "checkpoint_tree_sha256": (
                    "7c81971fafcdbc677b0e8fd25b3bffd3d624abe4635323784f1918abdcef6f15"
                ),
                "policy_config": "pi05_libero",
                "no_inference_before_divergence": True,
                "own_observation_after_divergence": True,
                "same_historical_noise_schedule_by_query_index": True,
            },
            "runtime": {
                "relative_path": (
                    "configs/vlsa_poisson_runtime_protocol.canary.v5.json"
                ),
                "schema_version": "vlsa_poisson_runtime_protocol.v5",
                "protocol_id": (
                    "vlsa-poisson-movable-manipulator-canary-parameters-v5"
                ),
                "active_runtime_role": (
                    "static_field_geometry_admissibility_and_numeric_"
                    "prerequisite_only_not_active_control_law"
                ),
                "active_control_source": (
                    "this_canary_execution_and_shield_sections_plus_original_osc"
                ),
                "consumed_runtime_sections": [
                    "workspace",
                    "occupancy",
                    "coverage",
                    "safety",
                    "poisson",
                    "admissibility",
                    "differential_audit_for_numeric_prerequisite_only",
                ],
                "non_authoritative_runtime_control_sections": [
                    "adapter",
                    "cbf",
                    "qp",
                    "cadence",
                ],
                "raw_file_sha256": (
                    "f0b13698b2e175cdd4da3be25d6aea19e130c24455361592c2a3a49acfd42d56"
                ),
                "semantic_sha256": (
                    "54811752920c503ab4d4a983d154a42d95cacd71d6209d6dea558583c37f927f"
                ),
                "parameter_block_sha256": (
                    TARGET_LINK_FROZEN_RUNTIME_PARAMETER_SHA256
                ),
                "same_parameter_block_required_for_all_allowed_cases": True,
                "parameter_freeze_after_case_id": TARGET_LINK_ALLOWED_CASE_IDS[0],
                "heldout_evaluation_case_id": TARGET_LINK_ALLOWED_CASE_IDS[1],
            },
            "numeric_prerequisite": {
                "schema_version": "vlsa_poisson_numeric_validation.v1",
                "evidence_tier": "allocation_backed_implementation_validation",
                "required_test_modules": [
                    "tests.test_poisson_geometry",
                    "tests.test_poisson_field",
                    "tests.test_poisson_field_bundle",
                    "tests.test_poisson_production_grid",
                    "tests.test_poisson_surface_sampling",
                    "tests.test_poisson_measurement",
                    "tests.test_poisson_jacobians",
                    "tests.test_poisson_cbf_qp",
                    "tests.test_poisson_prephysics_control_wrapper",
                    "tests.test_poisson_post_osc_torque_sensitivity",
                    "tests.test_poisson_post_osc_torque_shield",
                    "tests.test_poisson_osc_arm_link_canary",
                    "tests.test_poisson_osc_target_link_protocol",
                    "tests.test_poisson_osc_numeric_prerequisite",
                    "tests.test_poisson_osc_arm_link_slurm_contract",
                    "tests.test_poisson_numeric_validation_runner",
                ],
                "minimum_test_count": 120,
                "zero_skips_required": True,
                "same_clean_source_commit_required": True,
                "single_h100_allocation_required": True,
                "production_grid_validation_required": True,
                "scientific_result": False,
            },
            "video": {
                "required": True,
                "source": "real_agentview_simulation_frames",
                "fps": 30,
                "two_dimensional_safety_overlay": False,
            },
            "claim_scope": (
                "two_case_simulator_oracle_direct_target_link5_link6_contact_"
                "prevention_with_same_common_protected_link5_link6_poisson_cbf_"
                "rows_independent_of_case_task_or_historical_target_attribution_"
                "all_authoritative_robot_selected_obstacle_and_literal_link56_"
                "external_contacts_monitored_empirical_post_osc_treatment_not_"
                "formal_invariance_or_all_environment_safety"
            ),
        }

    def passing_metrics(self):
        metrics = {name: True for name in TARGET_LINK_REQUIRED_APPARATUS_METRICS}
        metrics.update(
            {
                "any_robot_selected_obstacle_contact": False,
                "any_link56_external_nonrobot_contact": False,
                "any_registered_forbidden_contact": False,
                "all_live_substeps_before_first_divergence_contact_free": True,
                "safety_method_stop_before_physics": False,
                "treatment_stalled_after_correction": False,
                "material_correction_present": True,
                "first_any_byte_different_torque_registered_as_divergence": True,
                "material_torque_correction_l2_nm": 0.02,
                "first_material_correction_protected_link_attributed": True,
                "first_material_nominal_exact_protected_link_cbf_residual_negative": True,
                "first_material_exact_cbf_residual_improvement_m2_per_s": 1e-6,
                "first_material_exact_next_qvel_change_l2_rad_s": 1e-4,
                "first_material_correction_before_historical_target_link_contact": True,
                "task_incomplete_at_first_material_correction": True,
                "treatment_paper_car_avoided": True,
                "nonstopping_motion_after_correction": True,
                "post_correction_joint_motion_integral_rad": 0.03,
                "post_correction_eef_path_length_m": 0.03,
                "native_task_success": True,
            }
        )
        return metrics

    def test_checked_in_case_protocols_match_the_exact_factory_and_hashes(self):
        for case_id, path in CONFIG_PATHS.items():
            with self.subTest(case_id=case_id):
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    CONFIG_SHA256[case_id],
                )
                value = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(value, self.protocol(case_id))
                derived = validate_osc_target_link_canary_protocol(value)
                self.assertEqual(derived["case_id"], case_id)
                self.assertEqual(
                    derived["protected_link_body_names"],
                    TARGET_LINK_PROTECTED_LINK_BODY_NAMES,
                )

    def test_checked_in_pair_freezes_every_active_parameter(self):
        e05 = json.loads(
            CONFIG_PATHS["vlsa-t1-goal-ii-t0-e05"].read_text(encoding="utf-8")
        )
        e42 = json.loads(
            CONFIG_PATHS["vlsa-t1-goal-ii-t3-e42"].read_text(encoding="utf-8")
        )
        for section in (
            "execution",
            "paper_car_measurement",
            "acceptance",
            "online_policy",
            "runtime",
            "numeric_prerequisite",
            "video",
        ):
            with self.subTest(section=section):
                self.assertEqual(e05[section], e42[section])
        self.assertEqual(e05["shield"], e42["shield"])
        self.assertEqual(
            e05["shield"]["protected_link_body_names"],
            list(TARGET_LINK_PROTECTED_LINK_BODY_NAMES),
        )
        self.assertEqual(e05["case"]["target_link_body_names"], ["robot0_link5"])
        self.assertEqual(e42["case"]["target_link_body_names"], ["robot0_link6"])

    def test_exact_e05_and_e42_registry_and_generic_prior_endpoint(self):
        expected = {
            "vlsa-t1-goal-ii-t0-e05": (["robot0_link5"], 187, 186, 188, 237),
            "vlsa-t1-goal-ii-t3-e42": (["robot0_link6"], 105, 104, 106, 120),
        }
        for case_id, values in expected.items():
            with self.subTest(case_id=case_id):
                protocol = self.protocol(case_id)
                derived = validate_osc_target_link_canary_protocol(protocol)
                self.assertEqual(list(derived["target_link_body_names"]), values[0])
                self.assertEqual(
                    protocol["shield"]["protected_link_body_names"],
                    list(TARGET_LINK_PROTECTED_LINK_BODY_NAMES),
                )
                self.assertEqual(
                    list(derived["protected_link_body_names"]),
                    list(TARGET_LINK_PROTECTED_LINK_BODY_NAMES),
                )
                self.assertEqual(derived["historical_contact_action"], values[1])
                self.assertEqual(
                    derived["historical_prior_endpoint_action"], values[2]
                )
                self.assertEqual(
                    derived["historical_car_first_crossing_action"], values[3]
                )
                self.assertEqual(
                    derived["historical_executed_action_count"], values[4]
                )
                self.assertEqual(
                    derived[
                        "same_source_action_contact_exception_callback_suffix"
                    ],
                    (0, 0),
                )
                self.assertEqual(
                    derived["frozen_runtime_parameter_sha256"],
                    TARGET_LINK_FROZEN_RUNTIME_PARAMETER_SHA256,
                )
                serialized = repr(protocol)
                self.assertNotIn("action62", serialized)
                self.assertNotIn("action61", serialized)

    def test_target_attribution_is_registry_data_not_control_selection(self):
        validator_source = inspect.getsource(
            validate_osc_target_link_canary_protocol
        )
        classifier_source = inspect.getsource(classify_osc_target_link_canary)
        for case_id in TARGET_LINK_ALLOWED_CASE_IDS:
            self.assertNotIn(case_id, validator_source)
            self.assertNotIn(case_id, classifier_source)
            protocol = self.protocol(case_id)
            self.assertEqual(
                protocol["shield"]["protected_link_body_names"],
                list(TARGET_LINK_PROTECTED_LINK_BODY_NAMES),
            )
            self.assertEqual(
                protocol["case"]["target_link_body_names"],
                list(TARGET_LINK_ALLOWED_CASE_REGISTRY[case_id]["target_link_body_names"]),
            )

    def test_runner_and_consumer_contain_no_frozen_case_id_control_branches(self):
        for relative_path in (
            "scripts/run_poisson_osc_arm_link_canary.py",
            "scripts/validate_poisson_osc_arm_link_canary_artifact.py",
        ):
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            for case_id in TARGET_LINK_ALLOWED_CASE_IDS:
                with self.subTest(path=relative_path, case_id=case_id):
                    self.assertNotIn(case_id, source)

    def test_protocol_rejects_case_target_archive_or_runtime_drift(self):
        mutations = (
            lambda value: value["case"].__setitem__(
                "case_id", "vlsa-t1-spatial-i-t3-e03"
            ),
            lambda value: value["case"].__setitem__(
                "target_link_body_names", ["robot0_link6"]
            ),
            lambda value: value["shield"].__setitem__(
                "protected_link_body_names", ["robot0_link6"]
            ),
            lambda value: value["allowed_case_registry"].pop(
                "vlsa-t1-goal-ii-t3-e42"
            ),
            lambda value: value["historical_control"].__setitem__(
                "rerun_baseline", True
            ),
            lambda value: value["execution"].__setitem__(
                "same_source_action_contact_exception",
                "allow_later_callback_in_historical_contact_action",
            ),
            lambda value: value["historical_control"].__setitem__(
                "result_file_sha256", "0" * 64
            ),
            lambda value: value["runtime"].__setitem__(
                "parameter_block_sha256", "0" * 64
            ),
            lambda value: value["shield"].__setitem__("alpha_gain_per_s", 4),
            lambda value: value["acceptance"].__setitem__(
                "minimum_post_correction_eef_path_length_m", 0.01
            ),
            lambda value: value["selection"].__setitem__(
                "remote_artifact_receipt_file_sha256", "0" * 64
            ),
            lambda value: value["online_policy"].__setitem__(
                "policy_config", "different_policy"
            ),
            lambda value: value["video"].__setitem__(
                "two_dimensional_safety_overlay", True
            ),
            lambda value: value["shield"].__setitem__(
                "protected_samples",
                "all_kinematically_movable_manipulator_collision_surface_samples",
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                candidate = copy.deepcopy(self.protocol(TARGET_LINK_ALLOWED_CASE_IDS[0]))
                mutate(candidate)
                with self.assertRaises(OscArmLinkCanaryError):
                    validate_osc_target_link_canary_protocol(candidate)

    def test_v4_positive_does_not_inherit_e03_whole_manipulator_gates(self):
        forbidden = {
            "all_structurally_movable_manipulator_collision_surfaces_shielded",
            "excluded_fixed_infrastructure_zero_qvel_influence_and_settled_contact_free",
            "robot_tree_qvel_scope_verified",
            "link56_bundle_samples_field_seed_only",
            "link56_shield_sampling_exact",
            "link56_arm_qvel_scope_verified",
            "every_executed_substep_link56_shielded",
            "target_link_shield_sampling_exact",
            "target_link_arm_qvel_scope_verified",
            "every_executed_substep_target_link_shielded",
            "action62_same_action_exception_has_action61_endpoint_parity_exact",
        }
        self.assertTrue(forbidden.isdisjoint(TARGET_LINK_REQUIRED_APPARATUS_METRICS))
        self.assertTrue(forbidden.isdisjoint(TARGET_LINK_REQUIRED_ACCEPTANCE))
        result = classify_osc_target_link_canary(
            self.passing_metrics(), self.protocol(TARGET_LINK_ALLOWED_CASE_IDS[0])
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(
            result["classification"],
            "SAFE_TASK_SUCCESS_USEFUL_PROTECTED_LINK_CORRECTION",
        )

    def test_each_case_requires_no_shift_car_motion_and_task_success(self):
        cases = (
            ("any_robot_selected_obstacle_contact", True, "CONTACT_REMAINS_OR_SHIFTED"),
            ("any_link56_external_nonrobot_contact", True, "CONTACT_REMAINS_OR_SHIFTED"),
            ("any_registered_forbidden_contact", True, "CONTACT_REMAINS_OR_SHIFTED"),
            ("safety_method_stop_before_physics", True, "STOP_ONLY"),
            ("treatment_stalled_after_correction", True, "STOP_ONLY"),
            ("material_correction_present", False, "NO_MATERIAL_CORRECTION"),
            ("material_torque_correction_l2_nm", 0.0, "NO_MATERIAL_CORRECTION"),
            (
                "first_material_correction_protected_link_attributed",
                False,
                "NO_PROTECTED_LINK_ATTRIBUTION",
            ),
            (
                "first_material_nominal_exact_protected_link_cbf_residual_negative",
                False,
                "NO_DIRECTED_SAFETY_IMPROVEMENT",
            ),
            (
                "first_material_exact_cbf_residual_improvement_m2_per_s",
                0.0,
                "NO_DIRECTED_SAFETY_IMPROVEMENT",
            ),
            (
                "first_material_correction_before_historical_target_link_contact",
                False,
                "TOO_LATE_FOR_HISTORICAL_TARGET_CONTACT",
            ),
            (
                "task_incomplete_at_first_material_correction",
                False,
                "CORRECTION_AFTER_TASK_COMPLETION",
            ),
            ("treatment_paper_car_avoided", False, "CAR_FAILURE_REMAINS"),
            ("nonstopping_motion_after_correction", False, "STOP_ONLY"),
            ("post_correction_joint_motion_integral_rad", 0.0, "STOP_ONLY"),
            ("post_correction_eef_path_length_m", 0.0, "STOP_ONLY"),
            ("native_task_success", False, "CONTACT_PREVENTED_TASK_FAILED"),
        )
        for case_id in TARGET_LINK_ALLOWED_CASE_IDS:
            for field, value, expected in cases:
                with self.subTest(case_id=case_id, field=field):
                    metrics = self.passing_metrics()
                    metrics[field] = value
                    result = classify_osc_target_link_canary(
                        metrics, self.protocol(case_id)
                    )
                    self.assertFalse(result["feasible"])
                    self.assertEqual(result["classification"], expected)

    def test_missing_or_malformed_evidence_is_inconclusive(self):
        for field in (
            "protected_link_shield_sampling_exact",
            "historical_contact_same_source_action_exception_requires_exact_callback_action_0_0_and_exact_contact_free_prior_completed_endpoint",
            "all_live_substeps_before_first_divergence_contact_free",
        ):
            metrics = self.passing_metrics()
            metrics.pop(field)
            result = classify_osc_target_link_canary(
                metrics, self.protocol(TARGET_LINK_ALLOWED_CASE_IDS[0])
            )
            self.assertEqual(result["classification"], "INCONCLUSIVE_APPARATUS")
            self.assertIn(field, result["apparatus_failures"])

        metrics = self.passing_metrics()
        metrics["post_correction_eef_path_length_m"] = "unknown"
        result = classify_osc_target_link_canary(
            metrics, self.protocol(TARGET_LINK_ALLOWED_CASE_IDS[0])
        )
        self.assertEqual(result["classification"], "INCONCLUSIVE_APPARATUS")
        self.assertIn(
            "post_correction_eef_path_length_m", result["apparatus_failures"]
        )

        # A real contact can outrank a well-typed False trace fact, but missing
        # or malformed evidence must still fail closed before classification.
        metrics = self.passing_metrics()
        metrics["any_registered_forbidden_contact"] = True
        metrics["all_live_substeps_before_first_divergence_contact_free"] = 0
        result = classify_osc_target_link_canary(
            metrics, self.protocol(TARGET_LINK_ALLOWED_CASE_IDS[0])
        )
        self.assertEqual(result["classification"], "INCONCLUSIVE_APPARATUS")
        self.assertIn(
            "all_live_substeps_before_first_divergence_contact_free",
            result["apparatus_failures"],
        )

    def test_pre_correction_qp_stop_is_stop_only_not_apparatus_failure(self):
        metrics = self.passing_metrics()
        metrics.update(
            {
                "safety_method_stop_before_physics": True,
                "material_correction_present": False,
                "material_torque_correction_l2_nm": 0.0,
                "first_any_byte_different_torque_registered_as_divergence": False,
                "first_material_correction_protected_link_attributed": False,
                "first_material_nominal_exact_protected_link_cbf_residual_negative": False,
                "first_material_correction_before_historical_target_link_contact": False,
                "task_incomplete_at_first_material_correction": False,
                "nonstopping_motion_after_correction": False,
                "native_task_success": False,
            }
        )
        result = classify_osc_target_link_canary(
            metrics, self.protocol(TARGET_LINK_ALLOWED_CASE_IDS[0])
        )
        self.assertEqual(result["classification"], "STOP_ONLY")
        self.assertFalse(result["feasible"])

        metrics = self.passing_metrics()
        metrics["first_any_byte_different_torque_registered_as_divergence"] = False
        result = classify_osc_target_link_canary(
            metrics, self.protocol(TARGET_LINK_ALLOWED_CASE_IDS[0])
        )
        self.assertEqual(result["classification"], "INCONCLUSIVE_APPARATUS")
        self.assertIn(
            "first_any_byte_different_torque_registered_as_divergence",
            result["apparatus_failures"],
        )

    def test_contact_before_torque_divergence_is_contact_not_apparatus(self):
        trace_metric = (
            "all_live_substeps_before_first_divergence_contact_free"
        )
        self.assertNotIn(trace_metric, TARGET_LINK_REQUIRED_APPARATUS_METRICS)
        self.assertIn(trace_metric, TARGET_LINK_REQUIRED_ACCEPTANCE)
        for contact_field in (
            "any_robot_selected_obstacle_contact",
            "any_link56_external_nonrobot_contact",
            "any_registered_forbidden_contact",
        ):
            with self.subTest(contact_field=contact_field):
                metrics = self.passing_metrics()
                metrics.update(
                    {
                        contact_field: True,
                        trace_metric: False,
                        "material_correction_present": False,
                        "material_torque_correction_l2_nm": 0.0,
                        "first_any_byte_different_torque_registered_as_divergence": False,
                    }
                )
                result = classify_osc_target_link_canary(
                    metrics, self.protocol(TARGET_LINK_ALLOWED_CASE_IDS[0])
                )
                self.assertEqual(
                    result["classification"], "CONTACT_REMAINS_OR_SHIFTED"
                )
                self.assertFalse(result["feasible"])
                self.assertNotIn(
                    trace_metric,
                    result["apparatus_failures"],
                )

    def test_false_predivergence_contact_free_cannot_support_positive_claim(self):
        metrics = self.passing_metrics()
        metrics["all_live_substeps_before_first_divergence_contact_free"] = False
        result = classify_osc_target_link_canary(
            metrics, self.protocol(TARGET_LINK_ALLOWED_CASE_IDS[0])
        )
        self.assertEqual(result["classification"], "INCONCLUSIVE_APPARATUS")
        self.assertFalse(result["feasible"])
        self.assertIn(
            "all_live_substeps_before_first_divergence_contact_free",
            result["apparatus_failures"],
        )

    def test_coherent_terminal_negatives_remain_scientific_outcomes(self):
        cases = (
            (
                {
                    "safety_method_stop_before_physics": True,
                    "material_correction_present": False,
                    "material_torque_correction_l2_nm": 0.0,
                    "first_any_byte_different_torque_registered_as_divergence": False,
                },
                "STOP_ONLY",
            ),
            (
                {
                    "material_correction_present": False,
                    "material_torque_correction_l2_nm": 0.0,
                    "first_any_byte_different_torque_registered_as_divergence": False,
                },
                "NO_MATERIAL_CORRECTION",
            ),
            (
                {
                    "first_material_correction_before_historical_target_link_contact": False,
                },
                "TOO_LATE_FOR_HISTORICAL_TARGET_CONTACT",
            ),
        )
        for updates, expected in cases:
            with self.subTest(expected=expected):
                metrics = self.passing_metrics()
                metrics.update(updates)
                result = classify_osc_target_link_canary(
                    metrics, self.protocol(TARGET_LINK_ALLOWED_CASE_IDS[0])
                )
                self.assertEqual(result["classification"], expected)
                self.assertFalse(result["feasible"])
                self.assertEqual(result["apparatus_failures"], [])

    def test_pair_claim_requires_both_cases_under_same_frozen_parameters(self):
        results = {
            case_id: classify_osc_target_link_canary(
                self.passing_metrics(), self.protocol(case_id)
            )
            for case_id in TARGET_LINK_ALLOWED_CASE_IDS
        }
        paired = classify_osc_target_link_pair(results)
        self.assertTrue(paired["pair_feasible"])
        self.assertEqual(paired["classification"], "LINK5_LINK6_PAIR_FEASIBLE")
        self.assertEqual(
            paired["protected_link_body_names"],
            list(TARGET_LINK_PROTECTED_LINK_BODY_NAMES),
        )

        partial = copy.deepcopy(results)
        failed_metrics = self.passing_metrics()
        failed_metrics["native_task_success"] = False
        partial[TARGET_LINK_ALLOWED_CASE_IDS[1]] = classify_osc_target_link_canary(
            failed_metrics, self.protocol(TARGET_LINK_ALLOWED_CASE_IDS[1])
        )
        paired = classify_osc_target_link_pair(partial)
        self.assertFalse(paired["pair_feasible"])
        self.assertEqual(paired["classification"], "CASE_SPECIFIC_ONLY")

        for field, replacement in (
            ("frozen_runtime_parameter_sha256", "0" * 64),
            ("protected_link_body_names", ["robot0_link6"]),
        ):
            malformed = copy.deepcopy(results)
            malformed[TARGET_LINK_ALLOWED_CASE_IDS[1]][field] = replacement
            with self.subTest(field=field), self.assertRaises(
                OscArmLinkCanaryError
            ):
                classify_osc_target_link_pair(malformed)


if __name__ == "__main__":
    unittest.main()
