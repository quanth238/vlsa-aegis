import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.run_poisson_fast_feasibility import (
    _contact_physical_boundary,
    _first_contact_boundary,
    _psf_static_assumption_crossed,
    _run_arm,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_poisson_fast_feasibility.py"
SLURM = ROOT / "slurm/poisson_fast_feasibility.sbatch"


class FastFeasibilityRunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.slurm = SLURM.read_text(encoding="utf-8")

    def test_runner_is_syntax_valid_and_has_no_policy_server(self):
        self.assertNotIn("policy_server", self.source.lower())
        self.assertIn('"online_policy_query_count": 0', self.source)

    def test_field_is_built_before_prefix_replay(self):
        build = self.source.index("bundle = build_static_field_bundle(")
        prefix = self.source.index(
            'for expected_step in replay.steps[: derived["start_action"]]:'
        )
        self.assertLess(build, prefix)
        self.assertIn("prefix_obstacle_rows", self.source)
        self.assertIn("prefix_action_boundary_speed_admissible", self.source)

    def test_exact_historical_boundary_and_window_are_bound(self):
        self.assertIn("post_action_179_flattened_state_sha256", self.source)
        self.assertIn("window_action_record_sha256", self.source)
        self.assertIn("actions[*].executed", (ROOT / "configs/vlsa_poisson_fast_feasibility.v1.json").read_text())

    def test_two_arms_use_grouped_100hz_500hz_path(self):
        self.assertIn("step_grouped_actions_with_substep_callback", self.source)
        self.assertIn("expected_inner_updates=5", self.source)
        self.assertIn("expected_substeps_per_inner=5", self.source)
        self.assertIn("joint_velocity_adapter_only", self.source)
        self.assertIn("joint_velocity_adapter_plus_link56_psf", self.source)
        self.assertIn("live_control_timestep", self.source)
        self.assertIn("live_model_timestep", self.source)
        self.assertIn("live_raw_model_timestep", self.source)
        self.assertIn("INNER_DT_S", self.source)
        self.assertIn("PHYSICS_DT_S", self.source)
        self.assertIn('"execution_cadence": execution_cadence', self.source)
        self.assertIn('"physics_monitor_trace_counts_match"', self.source)

    def test_static_field_admissibility_only_aborts_the_psf_arm(self):
        protocol = {
            "admissibility": {
                "max_selected_geom_translation_drift_m": 1.0e-6,
                "max_selected_geom_rotation_drift_rad": 1.0e-6,
                "max_selected_geom_surface_drift_m": 1.0e-6,
                "max_selected_body_linear_speed_m_s": 1.0e-6,
                "max_selected_body_angular_speed_rad_s": 1.0e-6,
            }
        }
        crossed = {
            "translation_drift_m": 2.0e-6,
            "rotation_drift_rad": 0.0,
            "surface_drift_m": 2.0e-6,
            "maximum_body_linear_speed_m_s": 0.0,
            "maximum_body_angular_speed_rad_s": 0.0,
        }
        self.assertFalse(
            _psf_static_assumption_crossed(
                psf_enabled=False,
                any_contact_seen=False,
                row=crossed,
                protocol=protocol,
            )
        )
        self.assertTrue(
            _psf_static_assumption_crossed(
                psf_enabled=True,
                any_contact_seen=False,
                row=crossed,
                protocol=protocol,
            )
        )
        self.assertFalse(
            _psf_static_assumption_crossed(
                psf_enabled=True,
                any_contact_seen=True,
                row=crossed,
                protocol=protocol,
            )
        )
        run_arm = self.source[
            self.source.index("def _run_arm(") : self.source.index("def _pair_exact(")
        ]
        append = run_arm.index("static_rows.append(static)")
        check = run_arm.index("if _psf_static_assumption_crossed(")
        self.assertLess(append, check)

    def test_exposure_counts_are_reusable_parameters_with_window_defaults(self):
        parameters = inspect.signature(_run_arm).parameters
        self.assertEqual(parameters["expected_filter_updates"].default, 40)
        self.assertEqual(parameters["expected_physics_substeps"].default, 200)
        self.assertIsNone(parameters["expected_boundary_goal_values"].default)
        run_arm = self.source[
            self.source.index("def _run_arm(") : self.source.index("def _pair_exact(")
        ]
        self.assertIn(
            "expected_filter_updates != expected_action_count * 5", run_arm
        )
        self.assertIn(
            "expected_physics_substeps != expected_filter_updates * 5", run_arm
        )
        self.assertIn(
            "len(command_rows) == expected_filter_updates", run_arm
        )
        self.assertIn(
            "physics_trace_row_count == expected_physics_substeps", run_arm
        )
        self.assertNotIn("len(command_rows) == 40", run_arm)
        self.assertNotIn("physics_trace_row_count == 200", run_arm)

    def test_full_episode_mode_binds_frozen_suffix_and_protocol_counts(self):
        self.assertIn(
            '"vlsa_poisson_full_episode_feasibility_protocol.v1"',
            self.source,
        )
        self.assertIn(
            "validate_full_episode_feasibility_protocol", self.source
        )
        self.assertIn("classify_full_episode_feasibility", self.source)
        self.assertIn('"suffix_action_record_sha256"', self.source)
        self.assertIn('"suffix_action_array_sha256"', self.source)
        self.assertIn(
            'expected_filter_updates=derived["expected_updates"]',
            self.source,
        )
        self.assertIn(
            'expected_physics_substeps=derived["expected_substeps"]',
            self.source,
        )
        self.assertIn(
            "expected_boundary_goal_values=expected_boundary_goal_values",
            self.source,
        )
        self.assertIn(
            'protocol["episode"]["expected_boundary_goal_values"]',
            self.source,
        )
        self.assertIn(
            'replay.steps[derived["start_action"] - 1].goal_values',
            self.source,
        )

    def test_full_episode_compact_metrics_match_classifier_contract(self):
        expected = {
            "schema_version",
            "exact_paired_start",
            "shared_prefix_complete",
            "full_recorded_episode_complete",
            "adapter_exposure_complete",
            "psf_exposure_complete",
            "adapter_physics_monitor_trace_counts_match",
            "psf_physics_monitor_trace_counts_match",
            "adapter_filter_update_count",
            "psf_filter_update_count",
            "adapter_physics_substep_count",
            "psf_physics_substep_count",
            "adapter_completed_suffix_action_count",
            "psf_completed_suffix_action_count",
            "psf_qp_count_complete",
            "psf_qp_postchecks_complete",
            "psf_joint_limit_postchecks_complete",
            "all_issued_commands_within_physical_bounds",
            "both_nominal_commands_within_dynamic_joint_bounds",
            "psf_invalid_field_query_count",
            "psf_all_post_state_field_queries_valid_and_positive",
            "static_selected_obstacle_admissible",
            "boundary_goal_unsatisfied",
            "adapter_link56_contact_present",
            "adapter_first_selected_obstacle_contact_is_link56",
            "psf_link56_contact_present",
            "psf_any_robot_selected_obstacle_contact_present",
            "psf_clearance_certified",
            "material_correction_before_adapter_contact",
            "first_material_correction_physical_boundary",
            "adapter_first_link56_contact_physical_boundary",
            "material_correction_update_count",
            "maximum_correction_norm_rad_s",
            "filter_correction_integral_rad",
            "post_correction_measured_joint_motion_integral_rad",
            "post_correction_cartesian_path_length_m",
            "post_correction_executed_command_integral_rad",
            "post_correction_zero_command_fraction",
            "psf_task_success_ever",
            "psf_terminal_task_success",
            "psf_first_task_success_source_action_index",
            "psf_task_success_after_material_correction",
            "adapter_task_success_ever",
            "adapter_terminal_task_success",
        }
        candidates = []
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Assign) or not isinstance(
                node.value, ast.Dict
            ):
                continue
            keys = {
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant)
                and isinstance(key.value, str)
            }
            if "full_recorded_episode_complete" in keys:
                candidates.append(keys)
        self.assertEqual(candidates, [expected])

    def test_full_episode_result_and_failure_use_selected_schema(self):
        self.assertIn('"protocol_id": selected_protocol_id', self.source)
        self.assertIn('"episode": episode', self.source)
        self.assertIn('"shared_osc_prefix"', self.source)
        self.assertIn('"paired_joint_velocity_suffix"', self.source)
        self.assertIn(
            '"historical_aegis_reference": historical_aegis_reference',
            self.source,
        )
        self.assertIn(
            '"released_aegis_barrier_h_values_positive"', self.source
        )
        self.assertIn('"terminal_recorded_action"', self.source)
        self.assertIn('"post_correction_motion": post_correction_motion', self.source)
        self.assertIn('"schema_version": selected_result_schema', self.source)
        self.assertNotIn(
            "from main.poisson_fullbody.fast_feasibility import RESULT_SCHEMA\n",
            self.source[self.source.index("except Exception as error:") :],
        )

    def test_native_goal_is_initialized_only_after_exact_restore(self):
        run_arm = self.source[
            self.source.index("def _run_arm(") : self.source.index("def _pair_exact(")
        ]
        restore = run_arm.index(
            "restore = restore_osc_settled_state_into_joint_velocity_env("
        )
        exact_restore_check = run_arm.index(
            'raise FastRunnerError("JV arm did not receive exact boundary B")'
        )
        goal_definition = run_arm.index(
            "goal_definition, goal_atoms = evaluator._goal_progress_definition(env)"
        )
        goal_snapshot = run_arm.index(
            "boundary_goal = evaluator._goal_progress_snapshot("
        )
        self.assertLess(restore, exact_restore_check)
        self.assertLess(exact_restore_check, goal_definition)
        self.assertLess(goal_definition, goal_snapshot)
        self.assertIn(
            "observed_boundary_values != expected_boundary_values", run_arm
        )
        self.assertIn(
            'boundary_goal["transition_metadata_available"] = False', run_arm
        )
        self.assertIn(
            "unavailable_without_preceding_action_boundary_goal_vector", run_arm
        )
        self.assertIn(
            'boundary_goal["newly_satisfied_indices"] = []', run_arm
        )
        self.assertIn(
            'boundary_goal["regressed_indices"] = []', run_arm
        )

    def test_grouped_native_task_result_is_measured_without_early_success_stop(self):
        run_arm = self.source[
            self.source.index("def _run_arm(") : self.source.index("def _pair_exact(")
        ]
        self.assertIn(
            "observation, reward, done, info = (", run_arm
        )
        self.assertIn(
            'failure_stage = "measure_native_task_goal"', run_arm
        )
        self.assertIn(
            'if bool(done) is not bool(goal["all_satisfied"]):', run_arm
        )
        self.assertIn(
            'snapshot_kind="completed_high_level_post_step"', run_arm
        )
        self.assertIn(
            "ever_task_success = bool(ever_task_success or terminal_task_success)",
            run_arm,
        )
        self.assertIn(
            '"continue_fixed_exposure_after_success": True', run_arm
        )
        self.assertNotIn("if done:\n                break", run_arm)
        terminal_sync = run_arm.index(
            'failure_stage = "synchronize_terminal_task_observation"'
        )
        partial_start = run_arm.index(
            "if contact_terminated_early:", terminal_sync
        )
        partial = run_arm[
            partial_start : run_arm.index("measurement = monitor.result()", partial_start)
        ]
        self.assertIn("ever_task_success = True", partial)
        self.assertIn(
            "first_task_success_source_action_index = source_index", partial
        )

    def test_task_evidence_binds_rewards_goals_and_terminal_hashes(self):
        run_arm = self.source[
            self.source.index("def _run_arm(") : self.source.index("def _pair_exact(")
        ]
        for field in (
            '"goal_definition": goal_definition',
            '"goal_progress_ledger": goal_ledger',
            '"reward_sum": float(reward_sum)',
            '"terminal_reward": terminal_reward',
            '"ever_task_success_at_or_after_branch"',
            '"first_task_success_source_action_index"',
            '"terminal_task_success"',
            '"terminal_goal_fraction"',
            '"terminal_simulator_state_sha256": terminal_state_hash',
            '"terminal_official_integration_state_raw_bytes_sha256"',
            '"terminal_observation_sha256": terminal_observation_hash',
        ):
            self.assertIn(field, run_arm)
        self.assertIn(
            "terminal_official_before_observables = _official_state(env.sim)",
            run_arm,
        )
        self.assertIn(
            "terminal observation synchronization changed integration state",
            run_arm,
        )

    def test_runner_does_not_preclip_filtered_command(self):
        run_arm = self.source[
            self.source.index("def _run_arm(") : self.source.index("def _pair_exact(")
        ]
        self.assertNotIn("np.clip", run_arm)
        self.assertIn("normalized_joint_velocity_action(executed", run_arm)

    def test_post_state_field_queries_use_world_points_not_jacobians(self):
        callback = self.source[
            self.source.index("def callback(") : self.source.index(
                "env.step_grouped_actions_with_substep_callback("
            )
        ]
        self.assertIn("evaluate_world_points", callback)
        self.assertNotIn("evaluate_point_jacobians", callback)
        self.assertIn("if psf_enabled and final_endpoint:", self.source)
        self.assertIn("pre_filter_field_observation_count += 1", self.source)
        self.assertIn("post_state_field_observation_count == 1", self.source)

    def test_exploratory_solver_budget_is_explicit_and_hash_bound(self):
        protocol = (
            ROOT / "configs/vlsa_poisson_fast_feasibility.v1.json"
        ).read_text(encoding="utf-8")
        self.assertIn('"qp_max_iterations": 50000', protocol)
        self.assertIn("max_iter=int(qp_max_iterations)", self.source)
        self.assertIn('"qp_max_iterations_effective"', self.source)
        self.assertIn('"exploratory_execution": dict(', self.source)

    def test_marker_is_synced_to_source_boundary_before_restore(self):
        build = self.source.index("marker_sync = _synchronize_visual_marker_model(source_env, env)")
        restore = self.source.index("restore = restore_osc_settled_state_into_joint_velocity_env(")
        self.assertLess(build, restore)
        self.assertIn("np.array_equal(source_pos, target_pos)", self.source)

    def test_adapter_target_is_initialized_after_wrapper_forward(self):
        run_arm = self.source[
            self.source.index("def _run_arm(") : self.source.index("def _pair_exact(")
        ]
        provider = run_arm[run_arm.index("def provider(") : run_arm.index("def callback(")]
        self.assertIn("if int(inner_index) == 0:", provider)
        self.assertIn("target = adapter.begin_high_level_action(", provider)
        before_provider = run_arm[: run_arm.index("def provider(")]
        self.assertNotIn("adapter.begin_high_level_action(", before_provider)

    def test_non_poisson_dynamic_bounds_are_measured_in_both_arms(self):
        run_arm = self.source[
            self.source.index("def _run_arm(") : self.source.index("def _pair_exact(")
        ]
        bounds = run_arm.index("lower, upper = joint_velocity_bounds(")
        psf_branch = run_arm.index("if psf_enabled:", bounds)
        self.assertLess(bounds, psf_branch)
        self.assertIn("nominal_within_dynamic_joint_bounds", run_arm)

    def test_activation_trace_has_sample_identity_and_strict_boundary(self):
        self.assertIn('"argmin_protected_sample": nominal_argmin_sample', self.source)
        self.assertIn('row["physical_boundary"] < adapter_contact_boundary', self.source)
        self.assertIn("_PsfContactObserved", self.source)
        self.assertIn("active_motion = _active_interval_motion(", self.source)
        self.assertIn('"psf_active_interval_motion_complete"', self.source)

    def test_contact_phases_are_normalized_to_physical_boundaries(self):
        live = SimpleNamespace(
            observation_index=7,
            source_phase="live_solver_phase_preintegration_geometry",
            is_physical_nonpositive_distance_contact=True,
            robot_geom_id=11,
        )
        post = SimpleNamespace(
            observation_index=7,
            source_phase="post_integration_recomputed",
            is_physical_nonpositive_distance_contact=True,
            robot_geom_id=11,
        )
        self.assertEqual(_contact_physical_boundary(live, 4500), 4507)
        self.assertEqual(_contact_physical_boundary(post, 4500), 4508)
        measurement = SimpleNamespace(
            live_solver_phase_contact_point_records=(live,),
            post_state_physical_contact_point_records=(post,),
        )
        self.assertEqual(
            _first_contact_boundary(
                measurement, start_boundary=4500, robot_geom_ids=(11,)
            ),
            4507,
        )

    def test_atomic_final_and_typed_failure(self):
        self.assertGreaterEqual(self.source.count("publish_hashed_json(result_path"), 2)
        self.assertIn('"partial_output_interpreted": False', self.source)
        self.assertIn('"primary_outcome": "APPARATUS_FAILURE"', self.source)
        self.assertIn('"arms": arm_evidence', self.source)

    def test_slurm_is_one_h100_and_has_no_upstream_scan_dependencies(self):
        self.assertIn("#SBATCH --gres=gpu:1", self.slurm)
        self.assertIn("#SBATCH --cpus-per-task=8", self.slurm)
        self.assertIn("#SBATCH --mem=64G", self.slurm)
        self.assertNotIn("PARITY_RESULT", self.slurm)
        self.assertNotIn("IDENTIFICATION_RESULT", self.slurm)
        self.assertNotIn("POLICY_PYTHON", self.slurm)
        self.assertIn("run_poisson_fast_feasibility.py", self.slurm)


if __name__ == "__main__":
    unittest.main()
