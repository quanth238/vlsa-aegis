from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_poisson_one_step_counterfactual.py"
BATCH_PATH = ROOT / "slurm/poisson_one_step_counterfactual.sbatch"
CORE_PATH = ROOT / "main/poisson_fullbody/one_step_counterfactual.py"


class OneStepCounterfactualRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner_source = RUNNER_PATH.read_text(encoding="utf-8")
        cls.batch_source = BATCH_PATH.read_text(encoding="utf-8")
        cls.core_source = CORE_PATH.read_text(encoding="utf-8")
        cls.runner = ast.parse(cls.runner_source)

    @classmethod
    def _runner_function_source(cls, name):
        node = next(
            item
            for item in cls.runner.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == name
        )
        return ast.get_source_segment(cls.runner_source, node)

    def test_runner_is_allocation_only_exact_prefix_and_no_policy_server(self):
        self.assertIn("_allocation(", self.runner_source)
        self.assertIn("_require_numeric_prerequisite(", self.runner_source)
        self.assertIn("_require_upstream_parity(", self.runner_source)
        self.assertIn("_require_identification_prerequisite(", self.runner_source)
        self.assertIn("one_exact_historical_OSC_prefix_no_policy_query", self.runner_source)
        self.assertIn("observed OSC callback prefix is incomplete", self.runner_source)
        self.assertNotIn("websocket", self.runner_source.lower())
        self.assertIn('"policy_server_started": False', self.runner_source)

    def test_runner_uses_full_tangent_velocity_and_refuses_hidden_clipping(self):
        self.assertIn("mujoco.mj_differentiatePos(", self.runner_source)
        self.assertIn("qdot_nom_full_nv", self.runner_source)
        self.assertIn("inadmissible_no_hidden_clipping", self.runner_source)
        self.assertIn("hidden_clipping_applied", self.runner_source)
        self.assertIn("arm_qpos_indices", self.runner_source)
        self.assertIn("arm_joint_types", self.runner_source)
        self.assertIn("arm_joint_qpos_widths", self.runner_source)
        self.assertIn("mjJNT_HINGE", self.runner_source)
        self.assertNotIn("qpos_B5 - qpos_B", self.runner_source)
        self.assertNotIn("expected_nominal_derivation_sha256", self.runner_source)

    def test_integration_state_prefix_uses_exact_time_qpos_qvel_act_layout(self):
        layout = self._runner_function_source(
            "_require_integration_state_prefix_layout"
        )
        capture = self._runner_function_source("_capture_state")
        measurement = self._runner_function_source("_state_measurement")
        self.assertIn("exact common [time, qpos, qvel, act] prefix", layout)
        self.assertIn("np.asarray([float(time_value)]", layout)
        self.assertIn("qpos_array", layout)
        self.assertIn("qvel_array", layout)
        self.assertIn("act_array", layout)
        self.assertIn("official_array[: expected_prefix.size]", layout)
        self.assertIn("_require_integration_state_prefix_layout(", capture)
        self.assertIn("time_value=float(data.time)", capture)
        self.assertIn("flattened.shape == expected_flattened.shape", capture)
        self.assertIn("np.array_equal(flattened, expected_flattened)", capture)
        self.assertIn("_require_integration_state_prefix_layout(", measurement)
        self.assertIn(
            "forwarded measurement qpos/qvel differ from integration-state source",
            measurement,
        )

    def test_result_and_receipt_bind_protocol_result_schema_and_provenance(self):
        self.assertIn('"result_schema_version": EXPECTED_RESULT_SCHEMA', self.runner_source)
        self.assertIn(
            'payload["provenance_sha256"] = _sha256_bytes(_canonical(payload["provenance"]))',
            self.runner_source,
        )

    def test_selection_and_joint_index_authorities_are_explicit(self):
        for token in (
            'frozen["selection_protocol_relative_path"]',
            'frozen["selection_schema_version"]',
            'frozen["selection_protocol_id"]',
            'frozen["selection_raw_file_sha256"]',
            '"relative_path": frozen["selection_protocol_relative_path"]',
            '"schema_version": selection["schema_version"]',
            'list(expected_arm_qpos_indices) != nominal_velocity["arm_qpos_indices"]',
            '"expected_arm_qpos_indices"',
            "source robot arm qpos registration differs from frozen protocol",
            "fresh JV arm DOF/qpos registration differs from source",
        ):
            self.assertIn(token, self.runner_source)
        self.assertIn('"registered_parameters": {', self.runner_source)
        self.assertIn(
            '("admissibility", "coverage", "cbf", "qp", "cadence")',
            self.runner_source,
        )

    def test_final_stdout_exposes_external_consumer_identity_without_login_parsing(self):
        for token in (
            '"producer_slurm_job_id"',
            '"producer_host"',
            '"producer_device"',
            '"result_file_sha256"',
            '"result_payload_sha256"',
            '"receipt_file_sha256"',
            '"receipt_payload_sha256"',
        ):
            self.assertIn(token, self.runner_source)
        self.assertIn(
            '"provenance_sha256": loaded_result["provenance_sha256"]',
            self.runner_source,
        )

    def test_classification_has_exact_local_and_active_reference_scopes(self):
        classification = self._runner_function_source("_classify_complete_pair")
        for token in (
            '"boundary_B_preflight_admissible"',
            '"nominal_reference"',
            '"target_contact_within_horizon"',
            '"unsafe_trend_without_target_contact"',
            '"warning_sample_valid_in_both_arms_at_horizon"',
            '"safe_clearance_greater_at_nominal_target_contact_boundary"',
        ):
            self.assertIn(token, classification)
        self.assertNotIn("trend_only_quality_prerequisites", classification)

    def test_nonzero_local_correction_and_not_full_stop_use_registered_thresholds(self):
        for token in (
            '"command_norm_rad_s"',
            '"command_integral_over_horizon_rad"',
            '"measured_motion_to_command_integral_ratio"',
            '"local_motion_retention"',
            '"filter_correction_above_registered_minimum"',
            '"safe_command_above_registered_minimum"',
            '"safe_command_retains_registered_nominal_norm_fraction"',
            '"safe_measured_motion_above_registered_minimum"',
            '"safe_measured_motion_retains_registered_command_integral_fraction"',
        ):
            self.assertIn(token, self.runner_source)

    def test_float_noise_does_not_pass_registered_local_motion_thresholds(self):
        from scripts.run_poisson_one_step_counterfactual import (
            _classify_complete_pair,
        )

        thresholds = {
            "minimum_filter_correction_norm_rad_s": 1e-4,
            "minimum_safe_command_norm_rad_s": 0.05,
            "minimum_safe_to_nominal_command_norm_ratio": 0.25,
            "minimum_safe_measured_joint_motion_rad": 1e-4,
            "minimum_safe_measured_motion_to_command_integral_ratio": 0.25,
        }
        protocol = {
            "acceptance": dict(
                thresholds,
                qp_safe_residual_minimum=-5e-7,
                maximum_safe_arm_invalid_field_queries=0,
                trend_only_quality_prerequisites=[
                    "complete_exposure",
                    "exact_paired_start",
                    "qp_valid",
                    "both_tracking_valid",
                    "static_field_admissible",
                ],
                strong_result_label="strong_causal_prevention",
                trend_only_label="unsafe_trend_without_reproduced_nominal_target_contact",
                other_outcome_label="inconclusive_or_failed_with_typed_reason",
            )
        }
        nominal = {
            "held_command_physics_substeps": 5,
            "hidden_clipping_applied": False,
            "start_boundary": {"integration_state_sha256": "a" * 64},
            "restore": {"settled_state_sha256": "b" * 64, "target_state_sha256": "b" * 64},
            "tracking": {"within_registered_thresholds": True},
            # Collision-caused post-contact drift may make the honest full
            # interval flag false without invalidating the pre-contact window.
            "static_field_admissible_for_complete_interval": False,
            "any_target_contact": True,
        }
        psf = {
            "held_command_physics_substeps": 5,
            "hidden_clipping_applied": False,
            "start_boundary": {"integration_state_sha256": "a" * 64},
            "restore": {"target_state_sha256": "b" * 64},
            "filter_solve_count": 1,
            "tracking": {"within_registered_thresholds": True},
            "static_field_admissible_for_complete_interval": True,
            "invalid_field_query_count": 0,
            "minimum_h_m2": 0.1,
            "physics_substep_ledger": [{"minimum_D_sim_m": 0.1}] * 5,
            "any_target_contact": False,
            "any_robot_to_selected_obstacle_contact": False,
            "any_shifted_robot_to_selected_obstacle_contact": False,
        }
        diagnostics = {
            "counterfactual_nominal_first_target_contact_boundary_C_nom": 4,
            "counterfactual_nominal_first_any_robot_selected_obstacle_contact_boundary_C_any_nom": 4,
            "counterfactual_nominal_target_contact_within_horizon": True,
            "nominal_clearance_at_C_nom_m": -0.001,
            "psf_clearance_at_C_nom_m": 0.001,
            "actual_h_after_each_physics_substep_m2": [0.9],
            "first_order_predicted_h_after_horizon_m2": 0.9,
            "h_at_B_m2": 1.0,
            "nominal_minimum_D_sim_over_horizon_m": -0.001,
            "nominal_D_sim_at_B_m": 0.1,
            "psf_warning_sample_h_strictly_greater_than_nominal_at_horizon": True,
            "psf_h_after_horizon_m2": 0.95,
            "static_field_admissibility_windows": {
                "nominal": {"admissible": True},
                "psf": {"admissible": True},
            },
            "tracking_admissibility_windows": {
                "nominal": {"admissible": True},
                "psf": {"admissible": True},
            },
            "local_motion_retention": {
                "filter_correction_norm_rad_s": 1e-12,
                "safe_command_norm_rad_s": 0.1,
                "safe_to_nominal_command_norm_ratio": 0.5,
                "safe_measured_joint_motion_rad": 0.001,
                "safe_measured_motion_to_command_integral_ratio": 1.0,
                "thresholds": thresholds,
            },
        }
        boundary_filter = {
            "safe_CBF_residuals_m2_per_s": [0.0],
            "one_CBF_row_per_exact_bound_sample": {"no_slack": True},
            "QP_status": "solved",
        }
        classified = _classify_complete_pair(
            protocol=protocol,
            boundary_admissibility={"passed": True},
            boundary_filter=boundary_filter,
            nominal=nominal,
            psf=psf,
            diagnostics=diagnostics,
            exact_paired_start=True,
        )
        self.assertFalse(classified["stage_13_passed"])
        self.assertFalse(
            classified["local_feasibility_conditions"][
                "filter_correction_above_registered_minimum"
            ]
        )

        above_floor = copy.deepcopy(diagnostics)
        above_floor["local_motion_retention"][
            "filter_correction_norm_rad_s"
        ] = 1e-4
        classified = _classify_complete_pair(
            protocol=protocol,
            boundary_admissibility={"passed": True},
            boundary_filter=boundary_filter,
            nominal=nominal,
            psf=psf,
            diagnostics=above_floor,
            exact_paired_start=True,
        )
        self.assertTrue(classified["stage_13_passed"])

        shifted_earlier = copy.deepcopy(above_floor)
        shifted_earlier[
            "counterfactual_nominal_first_any_robot_selected_obstacle_contact_boundary_C_any_nom"
        ] = 3
        classified = _classify_complete_pair(
            protocol=protocol,
            boundary_admissibility={"passed": True},
            boundary_filter=boundary_filter,
            nominal=nominal,
            psf=psf,
            diagnostics=shifted_earlier,
            exact_paired_start=True,
        )
        self.assertFalse(classified["stage_13_passed"])
        self.assertFalse(
            classified["nominal_reference"]["conditions"][
                "nominal_first_selected_obstacle_contact_matches_target_boundary"
            ]
        )
        self.assertIn(
            "nominal_first_selected_obstacle_contact_matches_target_boundary",
            classified["typed_reasons"],
        )

        trend = copy.deepcopy(above_floor)
        trend.update(
            {
                "counterfactual_nominal_first_target_contact_boundary_C_nom": None,
                "counterfactual_nominal_first_any_robot_selected_obstacle_contact_boundary_C_any_nom": None,
                "counterfactual_nominal_target_contact_within_horizon": False,
                "nominal_clearance_at_C_nom_m": None,
                "psf_clearance_at_C_nom_m": None,
            }
        )
        trend_nominal = copy.deepcopy(nominal)
        trend_nominal["any_target_contact"] = False
        classified = _classify_complete_pair(
            protocol=protocol,
            boundary_admissibility={"passed": True},
            boundary_filter=boundary_filter,
            nominal=trend_nominal,
            psf=psf,
            diagnostics=trend,
            exact_paired_start=True,
        )
        self.assertTrue(classified["stage_13_passed"])
        self.assertFalse(classified["contact_prevention_observed"])
        self.assertEqual(
            classified["nominal_reference"]["kind"],
            "unsafe_trend_without_target_contact",
        )
        self.assertEqual(classified["typed_reasons"], [])

    def test_paired_start_requires_controller_and_physical_state_equality(self):
        from scripts.run_poisson_one_step_counterfactual import (
            OneStepRunnerError,
            _canonical,
            _require_exact_paired_arm_start,
            _sha256_bytes,
        )

        software = {
            "schema_version": "vlsa_poisson_joint_velocity_controller_software_state.v1",
            "fields": {"goal_vel": {"available": True, "value": [0.0] * 7}},
        }
        software["sha256"] = _sha256_bytes(_canonical(software))
        restore = {
            "model_topology_sha256": "a" * 64,
            "physical_model_sha256": "b" * 64,
            "compiled_mjb_sha256": "c" * 64,
            "controller": {"name": "JOINT_VELOCITY"},
            "controller_software_state": software,
            "pid_memory_reset": {"goal_velocity_zero": True},
        }
        boundary = {
            "integration_state_sha256": "d" * 64,
            "qpos": [0.0] * 7,
            "qvel": [0.0] * 7,
            "commanded_arm_qdot": [0.1] * 7,
            "normalized_joint_velocity_action_8d": [0.2] * 7 + [1.0],
        }
        nominal = {"restore": copy.deepcopy(restore), "start_boundary": copy.deepcopy(boundary)}
        psf = {"restore": copy.deepcopy(restore), "start_boundary": copy.deepcopy(boundary)}
        psf["start_boundary"]["commanded_arm_qdot"] = [0.05] * 7
        psf["start_boundary"]["normalized_joint_velocity_action_8d"] = [0.1] * 7 + [1.0]
        self.assertTrue(
            _require_exact_paired_arm_start(
                nominal=nominal, psf=psf, boundary_measurement=boundary
            )
        )

        controller_tamper = copy.deepcopy(psf)
        controller_tamper["restore"]["controller_software_state"]["fields"][
            "goal_vel"
        ]["value"][0] = 1.0
        with self.assertRaisesRegex(
            OneStepRunnerError, "controller_software_state differs"
        ):
            _require_exact_paired_arm_start(
                nominal=nominal,
                psf=controller_tamper,
                boundary_measurement=boundary,
            )

        physical_tamper = copy.deepcopy(psf)
        physical_tamper["start_boundary"]["qpos"][0] = 1e-9
        with self.assertRaisesRegex(OneStepRunnerError, "physical start differs"):
            _require_exact_paired_arm_start(
                nominal=nominal,
                psf=physical_tamper,
                boundary_measurement=boundary,
            )

    def test_boundary_capture_and_controller_restore_are_complete(self):
        for token in (
            "mjSTATE_INTEGRATION",
            "flattened_simulator_state",
            "wrapper_bookkeeping",
            "mj_setState_forward_mj_setState_plus_wrapper_bookkeeping",
            "restore_osc_settled_state_into_joint_velocity_env",
            "pid_memory_reset",
        ):
            self.assertIn(token, self.runner_source)

    def test_contact_records_are_phase_and_boundary_explicit(self):
        self.assertIn('"physical_boundary": int(physical_boundary)', self.runner_source)
        self.assertIn("live_solver_phase_preintegration_geometry", self.runner_source)
        self.assertIn("post_integration_recomputed", self.runner_source)
        self.assertIn(
            "exact_ordered_nonpositive_physical_contact_subset", self.runner_source
        )
        self.assertIn("include_live_solver=False", self.runner_source)

    def test_B_and_arm_start_are_forwarded_only_while_callbacks_union_phases(self):
        measurement = self._runner_function_source("_state_measurement")
        arm = self._runner_function_source("_run_counterfactual_arm")
        main = self._runner_function_source("main")
        self.assertIn("forwarded = clone_forwarded_state(model, data)", measurement)
        self.assertIn('"live_solver_phase_preintegration_geometry"', measurement)
        self.assertIn("int(physical_boundary) - 1", measurement)
        self.assertIn('"post_integration_recomputed"', measurement)
        self.assertIn("model,\n        forwarded", measurement)
        self.assertIn(
            "physical_boundary=boundary + int(substep_index) + 1", arm
        )
        # Exactly the B preflight and each arm's B start disable stale live-solver
        # contacts; callback rows use the default interval-union measurement.
        self.assertEqual(self.runner_source.count("include_live_solver=False"), 2)
        self.assertIn("boundary_measurement = _state_measurement(", main)
        self.assertIn("start = _state_measurement(", arm)

    def test_obstacle_body_velocity_population_is_complete_not_one_witness(self):
        velocity = self._runner_function_source("_selected_obstacle_body_motion")
        self.assertIn(
            "for body_id in sorted(set(int(value) for value in obstacle_body_ids))",
            velocity,
        )
        self.assertIn("selected_obstacle_body_velocity_records", velocity)
        self.assertIn(
            'body_ids != sorted(resolved["ids"]["obstacle_body_ids"])',
            self.core_source,
        )

    def test_D_sim_uses_the_exact_bound_full_robot_sample_population(self):
        for token in (
            '"sample_ledger": full_sample_ledger',
            '"full_robot_measurement_sampling": dict(full_robot_sampling)',
            '"ordered_full_robot_sample_distance_m"',
            '"registered_full_robot_sample_count"',
            '"registered_full_robot_sample_ledger_sha256"',
            'full_robot_sampling=full_robot_sampling',
        ):
            self.assertIn(token, self.runner_source)

    def test_qp_evidence_has_every_exact_row_and_joint_bound_arithmetic(self):
        for token in (
            "one_CBF_row_per_exact_bound_sample",
            "joint_velocity_bound_rows",
            "joint_position_constraint_rows",
            "no_slack",
            "reconstructed_lower",
            "reconstructed_upper",
        ):
            self.assertIn(token, self.runner_source)
        for token in (
            '"expected_arm_q_min_rad"',
            '"expected_arm_q_max_rad"',
            '"joint_limit_match_absolute_tolerance_rad"',
            "compiled Panda joint limits differ from frozen Stage-13 authority",
        ):
            self.assertIn(token, self.runner_source)

    def test_alternate_model_joint_limits_are_rejected(self):
        from scripts.run_poisson_one_step_counterfactual import (
            OneStepRunnerError,
            _require_frozen_joint_limits,
        )

        lower = [
            -2.8973,
            -1.7628,
            -2.8973,
            -3.0718,
            -2.8973,
            -0.0175,
            -2.8973,
        ]
        upper = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        protocol = {
            "qp_execution": {
                "expected_arm_q_min_rad": lower,
                "expected_arm_q_max_rad": upper,
                "joint_limit_match_absolute_tolerance_rad": 1e-12,
            }
        }
        self.assertTrue(
            _require_frozen_joint_limits(
                stage_protocol=protocol,
                observed_lower_rad=lower,
                observed_upper_rad=upper,
            )
        )
        alternate = list(upper)
        alternate[3] += 1e-6
        with self.assertRaisesRegex(
            OneStepRunnerError, "compiled Panda joint limits differ"
        ):
            _require_frozen_joint_limits(
                stage_protocol=protocol,
                observed_lower_rad=lower,
                observed_upper_rad=alternate,
            )

    def test_slurm_wrapper_is_one_h100_clean_immutable_and_has_all_prerequisites(self):
        for token in (
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=64G",
            "EXPECTED_GIT_COMMIT",
            "NUMERIC_VALIDATION_RESULT",
            "PARITY_RESULT",
            "SHADOW_IDENTIFICATION_RESULT",
            "one-step output already exists",
            "git status --short",
        ):
            self.assertIn(token, self.batch_source)
        self.assertNotIn("policy server", self.batch_source.lower())

    def test_existing_output_is_never_resumed_or_reused(self):
        from scripts.run_poisson_one_step_counterfactual import (
            OneStepRunnerError,
            _validated_new_output,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = _validated_new_output(root, "unused-run")
            self.assertTrue(output.is_dir())
            with self.assertRaisesRegex(OneStepRunnerError, "already exists"):
                _validated_new_output(root, "unused-run")

    def test_warning_authorizes_first_scheduled_boundary_not_latest_before_contact(self):
        from scripts.run_poisson_one_step_counterfactual import (
            _contact_from_identification,
        )

        contact = {
            "observation_index": 4697,
            "source_phase": "live_solver_phase_preintegration_geometry",
            "is_physical_nonpositive_distance_contact": True,
            "robot_geom_id": 10,
            "robot_geom_name": "robot0_link5_collision",
            "robot_body_id": 5,
            "robot_body_name": "robot0_link5",
            "obstacle_geom_id": 20,
            "obstacle_geom_name": "moka_pot_collision",
            "obstacle_body_id": 8,
            "obstacle_body_name": "moka_pot_obstacle_1",
        }
        identification = {
            "shadow_replay": {
                "poisson_identification": {
                    "contact_prediction_assessment": {
                        "assessment": "registered_warning_preceded_link56_contact",
                        "first_link56_contact": contact,
                        "primary_registered_warning": {"observation_index": 4507},
                    }
                }
            }
        }
        target, contact_boundary, boundary = _contact_from_identification(
            identification,
            {"counterfactual_boundary": {"physics_substeps_per_filter_update": 5}},
        )
        self.assertEqual(target, contact)
        self.assertEqual(contact_boundary, 4697)
        self.assertEqual(boundary, 4510)
        self.assertNotEqual(boundary, 4695)

    def test_preflight_inadmissible_terminal_is_complete_and_executes_no_qp(self):
        from scripts.run_poisson_one_step_counterfactual import (
            _preflight_inadmissible_outcome,
        )

        gripper = {
            "source_action_index": 180,
            "source_action_7d": [0.0] * 6 + [1.0],
            "source_action_sha256": "a" * 64,
            "exact_gripper_value": 1.0,
            "exact_gripper_value_sha256": "b" * 64,
        }
        terminal = _preflight_inadmissible_outcome(
            bound_sample_count=1531,
            source_prefix_callback_count=4515,
            gripper_evidence=gripper,
        )
        self.assertEqual(terminal["gripper_evidence"], gripper)
        self.assertIsNone(terminal["boundary_B_filter"])
        self.assertEqual(terminal["arms"], [])
        self.assertIsNone(terminal["diagnostics"])
        self.assertFalse(terminal["execution"]["qp_executed"])
        self.assertFalse(
            terminal["execution"]["paired_joint_velocity_physics_executed"]
        )
        self.assertEqual(terminal["counts"]["qp_solve_count"], 0)
        self.assertEqual(terminal["counts"]["nominal_physics_substep_count"], 0)
        self.assertEqual(
            terminal["classification"]["typed_reasons"],
            ["nominal_velocity_outside_registered_controller_envelope"],
        )
        helper = self._runner_function_source("_preflight_inadmissible_outcome")
        self.assertNotIn("_boundary_filter_evidence", helper)
        self.assertNotIn("_run_counterfactual_arm", helper)
        main = self._runner_function_source("main")
        self.assertLess(
            main.index("except _NominalVelocityPreflightInadmissible"),
            main.index("validate_one_step_counterfactual_result("),
        )
        self.assertLess(
            main.index("validate_one_step_counterfactual_result("),
            main.index("publish_hashed_json(result_path, candidate)"),
        )
        self.assertIn("publish_hashed_json(receipt_path, receipt)", main)

    def test_full_physical_model_contract_is_exact_and_bound_everywhere(self):
        from scripts.run_poisson_one_step_counterfactual import (
            OneStepRunnerError,
            PHYSICAL_MODEL_CONTRACT_KEYS,
            _require_exact_physical_model_contract,
        )

        contract = {
            "schema_version": "vlsa_poisson_physical_model.v3",
            "sha256": "a" * 64,
            "field_count": 100,
            "option_field_count": 10,
            "compiled_mjb_sha256": "b" * 64,
            "compiled_mjb_bytes": 1000,
            "nq": 7,
            "nv": 7,
            "na": 0,
            "mjstate_integration_size": 15,
            "robosuite_flattened_state_size": 15,
            "robosuite_flattened_state_layout": "time_qpos_qvel_act_no_udd_tail",
        }
        self.assertEqual(
            set(_require_exact_physical_model_contract(contract, "test")),
            set(PHYSICAL_MODEL_CONTRACT_KEYS),
        )
        with self.assertRaisesRegex(OneStepRunnerError, "fields differ"):
            _require_exact_physical_model_contract(
                dict(contract, unexpected=True), "test"
            )
        with self.assertRaisesRegex(OneStepRunnerError, "state layout"):
            _require_exact_physical_model_contract(
                dict(contract, robosuite_flattened_state_size=16), "test"
            )
        binding = self._runner_function_source("_require_exact_shadow_construction")
        self.assertIn('"physical_model": dict(fresh_physical_model)', binding)
        main = self._runner_function_source("main")
        self.assertIn('"physical_model": dict(physical_model)', main)
        self.assertIn(
            '"physical_model": identification_construction["physical_model"]',
            main,
        )

    def test_controller_authority_is_captured_before_oob_terminal_branch(self):
        helper = self._runner_function_source(
            "_capture_joint_velocity_controller_authority"
        )
        self.assertIn("joint_velocity_controller_contract", helper)
        self.assertIn("model_physics_contract", helper)
        main = self._runner_function_source("main")
        self.assertLess(
            main.index("_capture_joint_velocity_controller_authority("),
            main.index("_derive_nominal_velocity("),
        )
        self.assertIn(
            '"joint_velocity_controller": dict(joint_velocity_controller)',
            main,
        )

    def test_identification_callback_authority_is_independently_hashed(self):
        from scripts.run_poisson_one_step_counterfactual import (
            OneStepRunnerError,
            _identification_callback_authority,
        )

        def digest(value):
            raw = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            return hashlib.sha256(raw).hexdigest()

        after = ["a" * 64, "b" * 64]
        rows = [
            {"physical_boundary": 1, "after_sha256": after[0]},
            {"physical_boundary": 2, "after_sha256": after[1]},
        ]
        shadow = {
            "callback_count": 2,
            "callback_state_read_only_ledger": rows,
            "callback_state_read_only_ledger_sha256": digest(rows),
            "callback_state_sequence_sha256": digest(after),
        }
        projected = _identification_callback_authority(
            {"shadow_replay": shadow}
        )
        self.assertEqual(projected["callback_state_read_only_count"], 2)
        self.assertEqual(
            projected["callback_state_read_only_after_sha256_ledger"], after
        )
        self.assertEqual(projected["callback_state_sequence_sha256"], digest(after))

        tampered = dict(shadow)
        tampered["callback_state_sequence_sha256"] = "c" * 64
        with self.assertRaisesRegex(
            OneStepRunnerError, "callback after-state sequence hash differs"
        ):
            _identification_callback_authority({"shadow_replay": tampered})

    def test_counterfactual_nominal_contact_boundary_is_phase_correct_not_historical(self):
        from scripts.run_poisson_one_step_counterfactual import (
            _diagnostic_and_classification_inputs,
        )

        scopes = {
            "nominal_with_selected_obstacle_contact": (
                "boundary_B_through_strictly_before_phase_correct_C_any_nom_"
                "excluding_contact_consequence"
            ),
            "nominal_without_selected_obstacle_contact": (
                "boundary_B_through_B_plus_5_inclusive"
            ),
            "poisson_filtered_velocity": (
                "boundary_B_through_B_plus_5_inclusive"
            ),
        }
        tracking_scopes = {
            "nominal_with_selected_obstacle_contact": (
                "physical_boundaries_B_plus_1_through_strictly_before_"
                "phase_correct_C_any_nom_excluding_contact_consequence"
            ),
            "nominal_without_selected_obstacle_contact": (
                "physical_boundaries_B_plus_1_through_B_plus_5_inclusive"
            ),
            "poisson_filtered_velocity": (
                "physical_boundaries_B_plus_1_through_B_plus_5_inclusive"
            ),
        }
        static_thresholds = {
            "translation_m": 0.01,
            "rotation_rad": 0.01,
            "surface_m": 0.01,
            "linear_speed_m_s": 0.01,
            "angular_speed_rad_s": 0.01,
        }

        def row(boundary, h, d_sim, contacts=None, drift=0.0):
            records = [] if contacts is None else contacts
            return {
                "physical_boundary": boundary,
                "ordered_sample_h_m2": [h],
                "first_order_predicted_h_m2": [h - 0.1],
                "prediction_error_m2": [0.1],
                "minimum_D_sim_m": d_sim,
                "post_state_minimum_D_sim_m": d_sim,
                "target_physical_contact_records": records,
                "robot_to_selected_obstacle_physical_contact_records": records,
                "selected_obstacle_translation_drift_m": drift,
                "selected_obstacle_rotation_drift_rad": drift,
                "selected_obstacle_surface_drift_m": drift,
                "selected_obstacle_max_body_linear_speed_m_s": drift,
                "selected_obstacle_max_body_angular_speed_rad_s": drift,
            }

        nominal_rows = [row(value, 1.0 - 0.1 * value, 0.2) for value in range(11, 16)]
        # A live-solver contact observed in callback row 13 is physically at C_nom=12.
        contact_records = [
            {
                "source_phase": "live_solver_phase_preintegration_geometry",
                "physical_boundary": 12,
                "contact_distance_m": -0.002,
            }
        ]
        nominal_rows[2]["target_physical_contact_records"] = contact_records
        nominal_rows[2][
            "robot_to_selected_obstacle_physical_contact_records"
        ] = contact_records
        nominal_rows[1]["post_state_minimum_D_sim_m"] = -0.001
        # Collision-caused drift at C_nom is retained in the full ledger but is
        # outside the nominal pre-contact static-field window.
        for index in range(1, len(nominal_rows)):
            for field in (
                "selected_obstacle_translation_drift_m",
                "selected_obstacle_rotation_drift_rad",
                "selected_obstacle_surface_drift_m",
                "selected_obstacle_max_body_linear_speed_m_s",
                "selected_obstacle_max_body_angular_speed_rad_s",
            ):
                nominal_rows[index][field] = 0.1
        safe_rows = [row(value, 1.0, 0.3) for value in range(11, 16)]
        value = _diagnostic_and_classification_inputs(
            protocol={
                "acceptance": {
                    "static_field_admissibility_scopes": scopes,
                    "tracking_admissibility_scopes": tracking_scopes,
                    "joint_velocity_tracking_linf_max_rad_s": 0.05,
                    "joint_velocity_tracking_rmse_max_rad_s": 0.02,
                    "minimum_filter_correction_norm_rad_s": 1e-4,
                    "minimum_safe_command_norm_rad_s": 0.05,
                    "minimum_safe_to_nominal_command_norm_ratio": 0.25,
                    "minimum_safe_measured_joint_motion_rad": 1e-4,
                    "minimum_safe_measured_motion_to_command_integral_ratio": 0.25,
                    "local_motion_interpretation": (
                        "nonzero_local_joint_motion_not_nominal_direction_progress_or_task_success"
                    ),
                }
            },
            boundary=10,
            contact_boundary=14,
            boundary_filter={
                "trend_sample_index": 0,
                "trend_sample_id": 7,
                "ordered_sample_h_m2": [2.0],
                "correction_norm": 0.1,
            },
            nominal={
                "physics_substep_ledger": nominal_rows,
                "start_boundary": row(10, 2.0, 0.4),
                "minimum_D_sim_m": -0.001,
                "any_target_contact": True,
                "any_robot_to_selected_obstacle_contact": True,
                "tracking": {
                    "error_ledger_rad_s": [[0.0] * 7 for _ in range(5)],
                    "registered_linf_max_rad_s": 0.05,
                    "registered_rmse_max_rad_s": 0.02,
                },
                "motion": {"command_norm_rad_s": 0.2},
            },
            psf={
                "physics_substep_ledger": safe_rows,
                "start_boundary": row(10, 2.0, 0.4),
                "tracking": {
                    "error_ledger_rad_s": [[0.0] * 7 for _ in range(5)],
                    "registered_linf_max_rad_s": 0.05,
                    "registered_rmse_max_rad_s": 0.02,
                },
                "motion": {
                    "command_norm_rad_s": 0.1,
                    "measured_arm_motion_l2_rad": 0.001,
                    "command_integral_over_horizon_rad": 0.001,
                    "measured_motion_to_command_integral_ratio": 1.0,
                },
            },
            static_field_thresholds=static_thresholds,
        )
        self.assertEqual(
            value["counterfactual_nominal_first_target_contact_boundary_C_nom"], 12
        )
        self.assertEqual(
            value[
                "counterfactual_nominal_first_any_robot_selected_obstacle_contact_boundary_C_any_nom"
            ],
            12,
        )
        self.assertEqual(value["historical_identification_contact_boundary_C"], 14)
        self.assertEqual(value["nominal_clearance_at_C_nom_m"], -0.002)
        self.assertEqual(value["psf_clearance_at_C_nom_m"], 0.3)
        self.assertEqual(
            value["local_motion_retention"][
                "safe_to_nominal_command_norm_ratio"
            ],
            0.5,
        )
        nominal_window = value["static_field_admissibility_windows"]["nominal"]
        self.assertEqual(nominal_window["cutoff_physical_boundary_exclusive"], 12)
        self.assertEqual(nominal_window["included_physical_boundaries"], [10, 11])
        self.assertTrue(nominal_window["admissible"])

    def test_static_window_rejects_precontact_drift_but_not_collision_consequence(self):
        from scripts.run_poisson_one_step_counterfactual import (
            _static_field_admissibility_windows,
        )

        scopes = {
            "nominal_with_selected_obstacle_contact": (
                "nominal_strictly_before_C_any_nom"
            ),
            "nominal_without_selected_obstacle_contact": "nominal_full_interval",
            "poisson_filtered_velocity": "psf_full_interval",
        }
        protocol = {"acceptance": {"static_field_admissibility_scopes": scopes}}
        thresholds = {
            "translation_m": 0.01,
            "rotation_rad": 0.01,
            "surface_m": 0.01,
            "linear_speed_m_s": 0.01,
            "angular_speed_rad_s": 0.01,
        }

        def row(boundary, value):
            return {
                "physical_boundary": boundary,
                "selected_obstacle_translation_drift_m": value,
                "selected_obstacle_rotation_drift_rad": value,
                "selected_obstacle_surface_drift_m": value,
                "selected_obstacle_max_body_linear_speed_m_s": value,
                "selected_obstacle_max_body_angular_speed_rad_s": value,
            }

        nominal_rows = [row(boundary, 0.0 if boundary < 12 else 0.1) for boundary in range(10, 16)]
        psf_rows = [row(boundary, 0.0) for boundary in range(10, 16)]
        nominal = {
            "start_boundary": nominal_rows[0],
            "physics_substep_ledger": nominal_rows[1:],
            "any_target_contact": True,
            "any_robot_to_selected_obstacle_contact": True,
            "static_field_admissible_for_complete_interval": False,
        }
        psf = {
            "start_boundary": psf_rows[0],
            "physics_substep_ledger": psf_rows[1:],
            "static_field_admissible_for_complete_interval": True,
        }
        windows = _static_field_admissibility_windows(
            protocol=protocol,
            boundary=10,
            nominal_any_contact_boundary=12,
            nominal=nominal,
            psf=psf,
            thresholds=thresholds,
        )
        self.assertEqual(
            windows["nominal"]["included_physical_boundaries"], [10, 11]
        )
        self.assertTrue(windows["nominal"]["admissible"])
        self.assertTrue(windows["psf"]["admissible"])

        precontact_drift = copy.deepcopy(nominal)
        precontact_drift["physics_substep_ledger"][0][
            "selected_obstacle_translation_drift_m"
        ] = 0.1
        rejected = _static_field_admissibility_windows(
            protocol=protocol,
            boundary=10,
            nominal_any_contact_boundary=12,
            nominal=precontact_drift,
            psf=psf,
            thresholds=thresholds,
        )
        self.assertFalse(rejected["nominal"]["admissible"])

    def test_tracking_window_allows_postcontact_response_but_rejects_precontact_error(self):
        from scripts.run_poisson_one_step_counterfactual import (
            _tracking_admissibility_windows,
        )

        scopes = {
            "nominal_with_selected_obstacle_contact": (
                "nominal_strictly_before_C_any_nom"
            ),
            "nominal_without_selected_obstacle_contact": "nominal_full_interval",
            "poisson_filtered_velocity": "psf_full_interval",
        }
        protocol = {
            "acceptance": {
                "tracking_admissibility_scopes": scopes,
                "joint_velocity_tracking_linf_max_rad_s": 0.05,
                "joint_velocity_tracking_rmse_max_rad_s": 0.02,
            }
        }

        def arm(errors, contact):
            return {
                "physics_substep_ledger": [
                    {"physical_boundary": value} for value in range(11, 16)
                ],
                "tracking": {
                    "error_ledger_rad_s": errors,
                    "registered_linf_max_rad_s": 0.05,
                    "registered_rmse_max_rad_s": 0.02,
                    "within_registered_thresholds": False,
                },
                "any_robot_to_selected_obstacle_contact": contact,
            }

        nominal_errors = [[0.0] * 7] + [[1.0] * 7 for _ in range(4)]
        nominal = arm(nominal_errors, True)
        psf = arm([[0.0] * 7 for _ in range(5)], False)
        windows = _tracking_admissibility_windows(
            protocol=protocol,
            boundary=10,
            nominal_any_contact_boundary=12,
            nominal=nominal,
            psf=psf,
        )
        self.assertEqual(
            windows["nominal"]["included_physical_boundaries"], [11]
        )
        self.assertTrue(windows["nominal"]["admissible"])
        self.assertTrue(windows["psf"]["admissible"])

        precontact = copy.deepcopy(nominal)
        precontact["tracking"]["error_ledger_rad_s"][0][0] = 0.1
        rejected = _tracking_admissibility_windows(
            protocol=protocol,
            boundary=10,
            nominal_any_contact_boundary=12,
            nominal=precontact,
            psf=psf,
        )
        self.assertFalse(rejected["nominal"]["admissible"])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is allocation-only")
    def test_array_hash_exactly_matches_table1_domain_separated_authority(self):
        import numpy as np
        from main.evaluate_safelibero_aegis import array_sha256
        from scripts.run_poisson_one_step_counterfactual import _array_sha256

        values = (
            np.asarray([1.0, 2.0], dtype=np.float64),
            np.arange(6, dtype=np.int32).reshape(2, 3),
        )
        for value in values:
            with self.subTest(dtype=str(value.dtype), shape=value.shape):
                self.assertEqual(_array_sha256(value), array_sha256(value))


if __name__ == "__main__":
    unittest.main()
