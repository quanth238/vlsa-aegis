import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "vlsa_poisson_osc_arm_link_canary.v3.json"


class OscArmLinkProtocolTests(unittest.TestCase):
    def protocol(self):
        return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def passing_metrics(self):
        return {
            "allocation_numeric_prerequisite_verified": True,
            "historical_control_verified": True,
            "historical_direct_link56_contact_verified": True,
            "historical_control_task_success": True,
            "direct_unit_gain_hinge_torque_actuators_verified": True,
            "all_structurally_movable_manipulator_collision_surfaces_shielded": True,
            "excluded_fixed_infrastructure_zero_qvel_influence_and_settled_contact_free": True,
            "all_authoritative_robot_collision_surfaces_contact_monitored": True,
            "robot_tree_qvel_scope_verified": True,
            "seven_arm_torque_decision_verified": True,
            "nonarm_controls_unchanged": True,
            "link56_bundle_samples_field_seed_only": True,
            "original_osc_controller_verified": True,
            "static_field_admissible": True,
            "every_executed_substep_shielded": True,
            "every_executed_substep_contact_monitored": True,
            "every_executed_substep_registered_contact_scope_monitored": True,
            "nominal_pass_through_bitwise_exact": True,
            "pre_correction_historical_parity_exact": True,
            "no_policy_query_before_divergence": True,
            "cached_current_chunk_then_fresh_own_observation_policy": True,
            "all_shield_decisions_valid": True,
            "safety_method_stop_before_physics": False,
            "all_actual_cbf_residual_postchecks_pass": True,
            "complete_terminal_condition_reached": True,
            "video_complete": True,
            "all_predivergence_torque_commands_byte_identical_nominal": True,
            "action62_same_action_exception_has_action61_endpoint_parity_exact": True,
            "all_live_substeps_before_first_divergence_contact_free": True,
            "any_robot_selected_obstacle_contact": False,
            "any_link56_external_nonrobot_contact": False,
            "any_link56_nonselected_external_contact": False,
            "any_registered_forbidden_contact": False,
            "material_correction_present": True,
            "first_torque_divergence_is_material": True,
            "first_divergence_nominal_exact_cbf_residual_negative": True,
            "first_material_correction_minimum_constraint_is_literal_link56": True,
            "first_material_exact_cbf_residual_improvement_m2_per_s": 1e-06,
            "first_material_exact_next_qvel_change_l2_rad_s": 1e-04,
            "material_correction_before_historical_contact": True,
            "task_incomplete_at_first_material_correction": True,
            "treatment_paper_car_avoided": True,
            "native_task_success": True,
            "post_correction_joint_motion_integral_rad": 0.03,
            "post_correction_eef_path_length_m": 0.03,
            "post_correction_zero_torque_delta_fraction": 0.5,
        }

    def test_checked_in_protocol_is_valid(self):
        from main.poisson_fullbody.osc_arm_link_canary import (
            validate_osc_arm_link_canary_protocol,
        )
        from main.poisson_fullbody.feasibility_protocol import (
            load_feasibility_protocol,
        )

        protocol = self.protocol()
        self.assertEqual(
            hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
            "6404650bbd215bd465da04e46d1b82f9917a6f5c61e7127ee80863bdb5d48d3f",
        )
        derived = validate_osc_arm_link_canary_protocol(protocol)
        self.assertEqual(derived["historical_contact_action"], 62)
        self.assertEqual(derived["maximum_action_count"], 300)
        self.assertEqual(derived["expected_substeps_per_action"], 25)
        runtime_path = ROOT / protocol["runtime"]["relative_path"]
        runtime, runtime_hashes = load_feasibility_protocol(runtime_path)
        self.assertEqual(runtime["schema_version"], protocol["runtime"]["schema_version"])
        self.assertEqual(runtime["protocol_id"], protocol["runtime"]["protocol_id"])
        self.assertEqual(
            hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
            protocol["runtime"]["raw_file_sha256"],
        )
        self.assertEqual(
            runtime_hashes.protocol_sha256,
            protocol["runtime"]["semantic_sha256"],
        )
        self.assertEqual(
            runtime_hashes.parameter_block_sha256,
            protocol["runtime"]["parameter_block_sha256"],
        )

    def test_protocol_rejects_controller_or_baseline_rerun_drift(self):
        from main.poisson_fullbody.osc_arm_link_canary import (
            OscArmLinkCanaryError,
            validate_osc_arm_link_canary_protocol,
        )

        for mutate in (
            lambda value: value["execution"].__setitem__("controller", "JOINT_VELOCITY"),
            lambda value: value["historical_control"].__setitem__("rerun_baseline", True),
            lambda value: value["execution"].__setitem__("physics_substeps_per_action", 5),
        ):
            protocol = copy.deepcopy(self.protocol())
            mutate(protocol)
            with self.assertRaises(OscArmLinkCanaryError):
                validate_osc_arm_link_canary_protocol(protocol)

    def test_protocol_freezes_movable_scope_fixed_partition_and_link56_seed_role(self):
        from main.poisson_fullbody.osc_arm_link_canary import (
            OscArmLinkCanaryError,
            validate_osc_arm_link_canary_protocol,
        )

        protocol = self.protocol()
        self.assertEqual(
            protocol["case"]["literal_link56_body_names"],
            ["robot0_link5", "robot0_link6"],
        )
        self.assertEqual(
            protocol["shield"]["protected_samples"],
            "all_kinematically_movable_manipulator_collision_surface_samples",
        )
        self.assertEqual(
            protocol["shield"]["shield_geometry_selection"],
            "collision_geom_world_pose_structurally_affected_by_any_"
            "authoritative_robot_tree_qvel_through_self_or_ancestor_joint",
        )
        self.assertEqual(
            protocol["shield"]["fixed_infrastructure_policy"],
            "exclude_only_empty_influencing_qvel_set_require_settled_"
            "contact_free_zero_authority_certificate_and_keep_contact_monitored",
        )
        self.assertEqual(
            protocol["shield"]["contact_monitor_population"],
            "all_collision_enabled_geoms_in_authoritative_robot_body_tree",
        )
        self.assertEqual(
            protocol["shield"]["point_velocity_scope"],
            "all_authoritative_robot_tree_qvels_structurally_affecting_shield_samples",
        )
        self.assertEqual(
            protocol["shield"]["decision_variable"],
            "seven_registered_panda_arm_torque_deltas",
        )
        self.assertEqual(
            protocol["shield"]["nonarm_control_policy"],
            "nominal_nonarm_controls_unchanged",
        )

        mutations = (
            lambda value: value["case"].__setitem__(
                "literal_link56_body_names", ["robot0_link5"]
            ),
            lambda value: value["shield"].__setitem__(
                "protected_samples", "robot0_link5_and_robot0_link6_surfaces"
            ),
            lambda value: value["shield"].__setitem__(
                "field_bundle_samples", "shield_constraint_population"
            ),
            lambda value: value["shield"].__setitem__(
                "shield_geometry_selection", "pose_specific_nonzero_jacobian"
            ),
            lambda value: value["shield"].__setitem__(
                "fixed_infrastructure_policy", "exclude_by_name"
            ),
            lambda value: value["shield"].__setitem__(
                "contact_monitor_population", "shield_population_only"
            ),
            lambda value: value["shield"].__setitem__(
                "point_velocity_scope", "seven_arm_qvel_only"
            ),
            lambda value: value["shield"].__setitem__(
                "decision_variable", "all_robot_controls"
            ),
            lambda value: value["shield"].__setitem__(
                "nonarm_control_policy", "zeroed"
            ),
            lambda value: value["historical_control"].__setitem__(
                "rerun_baseline", True
            ),
            lambda value: value["runtime"].__setitem__(
                "schema_version", "vlsa_poisson_runtime_protocol.v4"
            ),
        )
        for mutate in mutations:
            candidate = copy.deepcopy(protocol)
            mutate(candidate)
            with self.assertRaises(OscArmLinkCanaryError):
                validate_osc_arm_link_canary_protocol(candidate)

    def test_positive_requires_movable_coverage_fixed_certificate_and_monitor(self):
        from main.poisson_fullbody.osc_arm_link_canary import (
            classify_osc_arm_link_canary,
        )

        for field in (
            "all_structurally_movable_manipulator_collision_surfaces_shielded",
            "excluded_fixed_infrastructure_zero_qvel_influence_and_settled_contact_free",
            "all_authoritative_robot_collision_surfaces_contact_monitored",
            "robot_tree_qvel_scope_verified",
            "seven_arm_torque_decision_verified",
            "nonarm_controls_unchanged",
            "link56_bundle_samples_field_seed_only",
        ):
            metrics = self.passing_metrics()
            metrics[field] = False
            classified = classify_osc_arm_link_canary(metrics, self.protocol())
            self.assertEqual(
                classified["classification"], "INCONCLUSIVE_APPARATUS"
            )
            self.assertIn(field, classified["apparatus_failures"])

    def test_immutable_v2_canary_protocol_file_is_unchanged(self):
        path = ROOT / "configs" / "vlsa_poisson_osc_arm_link_canary.v2.json"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "b987bc277a78f431adae80de784c95931d1114638709bf87a2cffe0ace1a4aca",
        )

    def test_protocol_rejects_paper_car_phase_or_source_drift(self):
        from main.poisson_fullbody.osc_arm_link_canary import (
            OscArmLinkCanaryError,
            validate_osc_arm_link_canary_protocol,
        )

        for field, value in (
            ("position_source", "post_integration_forwarded_body_xpos"),
            ("root_body_binding", "name_only"),
            ("native_observable_binding", "live_body_xpos_tolerance"),
            ("live_root_position_role", "paper_car_authority"),
            ("post_integration_forwarded_pose_role", "paper_car_metric"),
        ):
            protocol = copy.deepcopy(self.protocol())
            protocol["paper_car_measurement"][field] = value
            with self.assertRaises(OscArmLinkCanaryError):
                validate_osc_arm_link_canary_protocol(protocol)

        protocol = copy.deepcopy(self.protocol())
        protocol["case"]["selected_obstacle_root_body_name"] = "wrong_body"
        with self.assertRaises(OscArmLinkCanaryError):
            validate_osc_arm_link_canary_protocol(protocol)

    def test_positive_requires_contact_avoidance_motion_and_task_success(self):
        from main.poisson_fullbody.osc_arm_link_canary import (
            classify_osc_arm_link_canary,
        )

        result = classify_osc_arm_link_canary(
            self.passing_metrics(), self.protocol()
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(
            result["classification"], "SAFE_TASK_SUCCESS_USEFUL_CORRECTION"
        )

        metrics = self.passing_metrics()
        metrics[
            "first_material_correction_minimum_constraint_is_literal_link56"
        ] = False
        classified = classify_osc_arm_link_canary(metrics, self.protocol())
        self.assertFalse(classified["feasible"])
        self.assertEqual(
            classified["classification"],
            "SAFE_TASK_SUCCESS_USEFUL_CORRECTION_NOT_LINK56_ATTRIBUTED",
        )

        cases = (
            ("any_robot_selected_obstacle_contact", True, "CONTACT_REMAINS_OR_SHIFTED"),
            ("any_link56_external_nonrobot_contact", True, "CONTACT_REMAINS_OR_SHIFTED"),
            ("any_registered_forbidden_contact", True, "CONTACT_REMAINS_OR_SHIFTED"),
            ("native_task_success", False, "CONTACT_PREVENTED_TASK_FAILED"),
            ("material_correction_present", False, "NO_MATERIAL_CORRECTION"),
            (
                "first_material_exact_cbf_residual_improvement_m2_per_s",
                0.0,
                "NO_DIRECTED_SAFETY_IMPROVEMENT",
            ),
            ("task_incomplete_at_first_material_correction", False, "CORRECTION_AFTER_TASK_COMPLETION"),
            ("treatment_paper_car_avoided", False, "CAR_FAILURE_REMAINS"),
            ("post_correction_eef_path_length_m", 0.0, "STOP_ONLY"),
            ("safety_method_stop_before_physics", True, "STOP_ONLY"),
        )
        for field, value, expected in cases:
            with self.subTest(field=field):
                metrics = self.passing_metrics()
                metrics[field] = value
                classified = classify_osc_arm_link_canary(metrics, self.protocol())
                self.assertFalse(classified["feasible"])
                self.assertEqual(classified["classification"], expected)

    def test_missing_substep_or_parity_evidence_is_inconclusive(self):
        from main.poisson_fullbody.osc_arm_link_canary import (
            classify_osc_arm_link_canary,
        )

        for field in (
            "every_executed_substep_shielded",
            "every_executed_substep_contact_monitored",
            "every_executed_substep_registered_contact_scope_monitored",
            "pre_correction_historical_parity_exact",
            "all_actual_cbf_residual_postchecks_pass",
        ):
            metrics = self.passing_metrics()
            metrics[field] = False
            result = classify_osc_arm_link_canary(metrics, self.protocol())
            self.assertEqual(result["classification"], "INCONCLUSIVE_APPARATUS")
            self.assertIn(field, result["apparatus_failures"])


class OscArmLinkRunnerStructuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (
            ROOT / "scripts" / "run_poisson_osc_arm_link_canary.py"
        ).read_text(encoding="utf-8")

    def test_runner_uses_original_osc_prephysics_hook(self):
        self.assertIn("step_with_arm_control_intervention", self.source)
        self.assertIn("after_nominal_osc_torque_before_every_mujoco_step", PROTOCOL_PATH.read_text())
        self.assertNotIn('controller="JOINT_VELOCITY"', self.source)
        self.assertNotIn("step_grouped_actions_with_substep_callback", self.source)

    def test_runner_uses_clone_only_sensitivity_and_hard_shield(self):
        self.assertIn("capture_post_osc_integration_state", self.source)
        self.assertIn("clone_one_substep_transition", self.source)
        self.assertIn("estimate_post_osc_torque_sensitivity", self.source)
        self.assertIn("SampledDataPostOscTorqueShield", self.source)
        self.assertIn("nominal_safe_exact_clone", self.source)
        self.assertIn("candidate_exact_clone_contact", self.source)
        self.assertIn("all_actual_cbf_residual_postchecks_pass", self.source)

    def test_runner_does_not_execute_a_live_baseline_arm(self):
        self.assertIn('"historical_control_rerun": False', self.source)
        self.assertNotIn("for arm_key", self.source)
        self.assertNotIn("adapter_only", self.source)

    def test_runner_binds_new_physical_contact_manifest_and_paper_car(self):
        self.assertIn("vlsa_poisson_arm_contact_165.v1.jsonl", self.source)
        self.assertIn("direct_link_active_obstacle_pairs", self.source)
        self.assertIn("actual_link_body_name", self.source)
        self.assertIn("paper_car_endpoint_ledger", self.source)
        self.assertIn("task_incomplete_at_first_material_correction", self.source)

    def test_independent_consumer_binds_distinct_slurm_jobs_and_commits(self):
        validator = (
            ROOT / "scripts" / "validate_poisson_osc_arm_link_canary_artifact.py"
        ).read_text(encoding="utf-8")
        batch = (
            ROOT / "slurm" / "poisson_osc_arm_link_canary_validate.sbatch"
        ).read_text(encoding="utf-8")
        for token in (
            "expected_numeric_job_id",
            "expected_producer_job_id",
            "expected_producer_commit",
            "expected_consumer_job_id",
            "expected_consumer_commit",
        ):
            self.assertIn(token, validator)
        self.assertIn('${EXPECTED_PRODUCER_JOB_ID}', batch)
        self.assertIn('${EXPECTED_NUMERIC_JOB_ID}', batch)
        self.assertIn('${SLURM_JOB_ID}', batch)

    def test_consumer_reconstructs_field_samples_qp_and_policy_chain(self):
        validator = (
            ROOT / "scripts" / "validate_poisson_osc_arm_link_canary_artifact.py"
        ).read_text(encoding="utf-8")
        for token in (
            "_validate_field_sample_static_evidence",
            "protected_samples_sha256",
            "static_obstacle_trace_sha256",
            "_validate_solved_qp_certificate",
            "finite_difference_column_stencils",
            "independent minimum-norm KKT certificate",
            "cached historical raw chunk differs from the bound result",
            "fresh policy response is not bound to the executed action chain",
            "independently_observed_checkpoint",
            "policy_server_identity_sha256",
            "force_limit_cannot_clip_control_range",
        ):
            self.assertIn(token, validator)

    def test_consumer_accepts_one_registered_partial_terminal_action(self):
        validator = (
            ROOT / "scripts" / "validate_poisson_osc_arm_link_canary_artifact.py"
        ).read_text(encoding="utf-8")
        self.assertIn("_validate_partial_action_ledger", validator)
        self.assertIn("exactly one extra planner action", validator)

    def test_runner_serializes_only_one_full_constraint_certificate(self):
        self.assertIn("_compact_float64_array_record", self.source)
        self.assertIn("retain_full_constraint_qp_certificate", self.source)
        self.assertIn("first_byte_different_torque_row_only", self.source)
        self.assertIn("partial_action_record", self.source)
        self.assertIn("_registered_contact_hook", self.source)
        self.assertIn("_RegisteredContactMonitor", self.source)
        self.assertIn("link56_vs_external_nonrobot", self.source)


class OscArmLinkPaperCarPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
        except ImportError as error:
            raise unittest.SkipTest("NumPy is required for paper CAR tests") from error
        cls.np = np

    def native_observable_fixture(self, *, sensor_object_name=None):
        obstacle_name = "wine_bottle_obstacle_1"
        bound_name = sensor_object_name or obstacle_name
        task_env = SimpleNamespace()
        task_env.value = self.np.asarray([1.0, 2.0, 3.0], dtype=self.np.float64)

        def make_sensor(self, obj_name):
            def obj_pos(obs_cache):
                del obs_cache
                return self.value if obj_name else self.value

            return obj_pos

        sensor = make_sensor(task_env, bound_name)
        sensor.__modality__ = "object"

        class FakeObservable:
            name = "%s_pos" % obstacle_name
            modality = "object"
            _sampling_timestep = 0.05
            _sensor = sensor
            obs = task_env.value.copy()

            @staticmethod
            def is_enabled():
                return True

            @staticmethod
            def is_active():
                return True

        key = "%s_pos" % obstacle_name
        task_env._observables = {key: FakeObservable()}
        task_env._obs_cache = {key: task_env.value.copy()}
        return task_env, key, obstacle_name

    def test_native_observable_identity_and_cache_are_bound(self):
        from scripts.run_poisson_osc_arm_link_canary import (
            _native_object_observable_values,
        )

        task_env, key, obstacle_name = self.native_observable_fixture()
        record, observable, cache = _native_object_observable_values(
            task_env,
            observation_key=key,
            obstacle_name=obstacle_name,
            np=self.np,
        )
        self.assertTrue(record["all_checks_passed"])
        self.assertTrue(record["checks"]["sensor_nonlocal_environment_matches"])
        self.assertTrue(self.np.array_equal(observable, task_env.value))
        self.assertTrue(self.np.array_equal(cache, task_env.value))

    def test_native_observable_wrong_closure_object_is_rejected(self):
        from scripts.run_poisson_osc_arm_link_canary import (
            OscCanaryRunnerError,
            _native_object_observable_values,
        )

        task_env, key, obstacle_name = self.native_observable_fixture(
            sensor_object_name="wrong_object"
        )
        with self.assertRaises(OscCanaryRunnerError):
            _native_object_observable_values(
                task_env,
                observation_key=key,
                obstacle_name=obstacle_name,
                np=self.np,
            )

    def row(self):
        from scripts.run_poisson_osc_arm_link_canary import (
            _paper_car_endpoint_row,
        )

        np = self.np
        observed = np.asarray([1.001, 2.0, 3.0], dtype=np.float64)
        return _paper_car_endpoint_row(
            source_action_index=0,
            snapshot_kind="completed_high_level_endpoint",
            observation_key="wine_bottle_obstacle_1_pos",
            obstacle_root_body_id=32,
            observation_position=observed,
            observable_value_position=observed.copy(),
            observation_cache_position=observed.copy(),
            live_body_position_diagnostic=observed.copy(),
            post_integration_forwarded_body_position=np.asarray(
                [1.002, 1.998, 3.003], dtype=np.float64
            ),
            settled_observation_position=np.asarray(
                [1.0, 2.0, 3.0], dtype=np.float64
            ),
            np=np,
        )

    def validate(self, row):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _validate_paper_car_endpoint_row,
        )

        return _validate_paper_car_endpoint_row(
            row,
            index=1,
            car_key="wine_bottle_obstacle_1_pos",
            obstacle_root_body_id=32,
            settled_car_position=[1.0, 2.0, 3.0],
            np=self.np,
        )

    def test_forwarded_cross_phase_offset_is_diagnostic_not_car(self):
        row = self.row()
        self.assertAlmostEqual(row["l1_displacement_from_settled_m"], 0.001)
        self.assertAlmostEqual(
            row["observation_post_integration_forwarded_l1_delta_m"], 0.006
        )
        self.assertAlmostEqual(self.validate(row), 0.001)

    def test_native_observable_cache_mismatch_is_rejected(self):
        from scripts.run_poisson_osc_arm_link_canary import (
            OscCanaryRunnerError,
            _paper_car_endpoint_row,
            _require_paper_car_native_cache_binding,
        )

        row = _paper_car_endpoint_row(
            source_action_index=0,
            snapshot_kind="completed_high_level_endpoint",
            observation_key="wine_bottle_obstacle_1_pos",
            obstacle_root_body_id=32,
            observation_position=[1.0, 2.0, 3.0],
            observable_value_position=[1.0, 2.0, 3.0],
            observation_cache_position=[1.0, 2.0, 3.000001],
            live_body_position_diagnostic=[1.0, 2.0, 3.0],
            post_integration_forwarded_body_position=[1.0, 2.0, 3.0],
            settled_observation_position=[1.0, 2.0, 3.0],
            np=self.np,
        )
        self.assertFalse(row["observation_cache_value_bitwise_equal"])
        self.assertFalse(row["native_observable_cache_binding_exact"])
        with self.assertRaises(OscCanaryRunnerError):
            _require_paper_car_native_cache_binding(row)

    def test_live_phase_offset_is_diagnostic_not_observable_authority(self):
        from scripts.run_poisson_osc_arm_link_canary import (
            _paper_car_endpoint_row,
            _require_paper_car_native_cache_binding,
        )
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _validate_paper_car_endpoint_row,
        )

        row = _paper_car_endpoint_row(
            source_action_index=-1,
            snapshot_kind="settled_pre_action",
            observation_key="wine_bottle_obstacle_1_pos",
            obstacle_root_body_id=32,
            observation_position=[0.0, 2.0, 3.0],
            observable_value_position=[0.0, 2.0, 3.0],
            observation_cache_position=[0.0, 2.0, 3.0],
            live_body_position_diagnostic=[
                -0.0,
                2.0,
                3.0 + 1.0e-13,
            ],
            post_integration_forwarded_body_position=[0.0, 2.0, 3.0],
            settled_observation_position=[0.0, 2.0, 3.0],
            np=self.np,
        )
        self.assertTrue(row["native_observable_cache_binding_exact"])
        self.assertNotEqual(
            row["active_obstacle_position_observation_array_sha256"],
            row["active_obstacle_root_position_array_sha256"],
        )
        _require_paper_car_native_cache_binding(row)
        self.assertAlmostEqual(
            _validate_paper_car_endpoint_row(
                row,
                index=0,
                car_key="wine_bottle_obstacle_1_pos",
                obstacle_root_body_id=32,
                settled_car_position=[0.0, 2.0, 3.0],
                np=self.np,
            ),
            0.0,
        )

    def test_runner_rejects_non_native_float64_positions(self):
        from scripts.run_poisson_osc_arm_link_canary import (
            OscCanaryRunnerError,
            _paper_car_endpoint_row,
        )

        positions = self.np.asarray([1.0, 2.0, 3.0], dtype=self.np.float64)
        with self.assertRaises(OscCanaryRunnerError):
            _paper_car_endpoint_row(
                source_action_index=0,
                snapshot_kind="completed_high_level_endpoint",
                observation_key="wine_bottle_obstacle_1_pos",
                obstacle_root_body_id=32,
                observation_position=positions.astype(self.np.float32),
                observable_value_position=positions.copy(),
                observation_cache_position=positions.copy(),
                live_body_position_diagnostic=positions.copy(),
                post_integration_forwarded_body_position=positions.copy(),
                settled_observation_position=positions.copy(),
                np=self.np,
            )

    def test_consumer_rejects_coercible_or_nonfloat_json_vectors(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        fields = (
            "active_obstacle_position_observation_world_m",
            "native_observable_value_world_m",
            "native_observation_cache_value_world_m",
            "active_obstacle_root_position_world_m",
            "active_obstacle_root_position_post_integration_forwarded_world_m",
            "observation_post_integration_forwarded_component_delta_m",
        )
        for field in fields:
            for replacement in ("1.0", True, 1):
                with self.subTest(field=field, replacement=replacement):
                    row = self.row()
                    row[field][0] = replacement
                    with self.assertRaises(OscCanaryValidationError):
                        self.validate(row)

    def test_consumer_rejects_nonfloat_settled_vector(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
            _validate_paper_car_endpoint_row,
        )

        with self.assertRaises(OscCanaryValidationError):
            _validate_paper_car_endpoint_row(
                self.row(),
                index=1,
                car_key="wine_bottle_obstacle_1_pos",
                obstacle_root_body_id=32,
                settled_car_position=["1.0", 2.0, 3.0],
                np=self.np,
            )

    def test_consumer_rejects_native_dtype_or_shape_metadata_tampering(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        mutations = (
            ("active_obstacle_position_observation_dtype", "<f4"),
            ("native_observable_value_dtype", "<f4"),
            ("native_observation_cache_value_dtype", "<f4"),
            ("active_obstacle_position_observation_shape", [1, 3]),
            ("native_observable_value_shape", [1, 3]),
            ("native_observation_cache_value_shape", [1, 3]),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                row = self.row()
                row[field] = value
                with self.assertRaises(OscCanaryValidationError):
                    self.validate(row)

    def test_consumer_rejects_phase_root_hash_or_delta_tampering(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        mutations = (
            lambda row: row.__setitem__("active_obstacle_root_body_id", 31),
            lambda row: row.__setitem__("active_obstacle_root_body_id", "32"),
            lambda row: row.__setitem__(
                "post_integration_forwarded_pose_role", "paper_car_metric"
            ),
            lambda row: row.__setitem__(
                "active_obstacle_position_observation_array_sha256", "0" * 64
            ),
            lambda row: row.__setitem__(
                "native_observable_cache_binding_exact", False
            ),
            lambda row: row.__setitem__(
                "native_observation_cache_value_array_sha256", "0" * 64
            ),
            lambda row: row.__setitem__(
                "observation_post_integration_forwarded_l1_delta_m", 0.0
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                row = self.row()
                mutate(row)
                with self.assertRaises(OscCanaryValidationError):
                    self.validate(row)

    def test_consumer_rejects_rehashed_signed_zero_live_position(self):
        from scripts.run_poisson_osc_arm_link_canary import (
            _paper_car_endpoint_row,
        )
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
            _float64_sha256,
        )

        row = _paper_car_endpoint_row(
            source_action_index=0,
            snapshot_kind="completed_high_level_endpoint",
            observation_key="wine_bottle_obstacle_1_pos",
            obstacle_root_body_id=32,
            observation_position=[0.0, 2.0, 3.0],
            observable_value_position=[0.0, 2.0, 3.0],
            observation_cache_position=[0.0, 2.0, 3.0],
            live_body_position_diagnostic=[0.0, 2.0, 3.0],
            post_integration_forwarded_body_position=[0.0, 2.0, 3.0],
            settled_observation_position=[0.0, 2.0, 3.0],
            np=self.np,
        )
        row["active_obstacle_root_position_world_m"][0] = -0.0
        row["active_obstacle_root_position_array_sha256"] = _float64_sha256(
            row["active_obstacle_root_position_world_m"], self.np
        )
        with self.assertRaises(OscCanaryValidationError):
            self.validate(row)

    def test_consumer_rejects_signed_zero_scalar_and_component_delta(self):
        from scripts.run_poisson_osc_arm_link_canary import (
            _paper_car_endpoint_row,
        )
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
            _float64_sha256,
            _validate_paper_car_endpoint_row,
        )

        scalar = self.row()
        scalar["observation_live_root_l1_delta_m"] = -0.0
        with self.assertRaises(OscCanaryValidationError):
            self.validate(scalar)

        component = _paper_car_endpoint_row(
            source_action_index=0,
            snapshot_kind="completed_high_level_endpoint",
            observation_key="wine_bottle_obstacle_1_pos",
            obstacle_root_body_id=32,
            observation_position=[1.0, 2.0, 3.0],
            observable_value_position=[1.0, 2.0, 3.0],
            observation_cache_position=[1.0, 2.0, 3.0],
            live_body_position_diagnostic=[1.0, 2.0, 3.0],
            post_integration_forwarded_body_position=[1.0, 2.0, 3.0],
            settled_observation_position=[1.0, 2.0, 3.0],
            np=self.np,
        )
        component[
            "observation_post_integration_forwarded_component_delta_m"
        ][0] = -0.0
        component[
            "observation_post_integration_forwarded_component_delta_array_sha256"
        ] = _float64_sha256(
            component[
                "observation_post_integration_forwarded_component_delta_m"
            ],
            self.np,
        )
        with self.assertRaises(OscCanaryValidationError):
            _validate_paper_car_endpoint_row(
                component,
                index=1,
                car_key="wine_bottle_obstacle_1_pos",
                obstacle_root_body_id=32,
                settled_car_position=[1.0, 2.0, 3.0],
                np=self.np,
            )

    def test_consumer_rejects_string_authority_root_id(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
            _strict_integer,
        )

        with self.assertRaises(OscCanaryValidationError):
            _strict_integer("32", "test root")

    def test_consumer_requires_exact_root_id_name_pairing(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
            _require_resolved_body_name,
        )

        resolved = {
            "obstacle_body_ids": [32, 33],
            "obstacle_body_names": [
                "wine_bottle_obstacle_1_main",
                "wine_bottle_obstacle_1_child",
            ],
        }
        mapping = _require_resolved_body_name(
            resolved,
            ids_field="obstacle_body_ids",
            names_field="obstacle_body_names",
            expected_body_id=32,
            expected_body_name="wine_bottle_obstacle_1_main",
            label="test",
        )
        self.assertEqual(mapping[32], "wine_bottle_obstacle_1_main")
        resolved["obstacle_body_names"].reverse()
        with self.assertRaises(OscCanaryValidationError):
            _require_resolved_body_name(
                resolved,
                ids_field="obstacle_body_ids",
                names_field="obstacle_body_names",
                expected_body_id=32,
                expected_body_name="wine_bottle_obstacle_1_main",
                label="test",
            )

    def test_car_cadence_historical_hash_and_scalars_are_exact(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
            _validate_paper_car_cadence_and_historical_binding,
            _validate_paper_car_endpoint_row,
        )

        settled = self.row()
        settled["source_action_index"] = -1
        settled["snapshot_kind"] = "settled_pre_action"
        settled["l1_displacement_from_settled_m"] = 0.0
        historical_hash = settled[
            "active_obstacle_position_observation_array_sha256"
        ]
        endpoint = self.row()
        ledger = [settled, endpoint]
        _validate_paper_car_cadence_and_historical_binding(
            ledger,
            action_count=1,
            historical_settled_position_sha256=historical_hash,
        )
        for mutate in (
            lambda rows: rows[0].__setitem__("snapshot_kind", "wrong"),
            lambda rows: rows[1].__setitem__("source_action_index", False),
            lambda rows: rows[0].__setitem__(
                "active_obstacle_position_observation_array_sha256", "0" * 64
            ),
            lambda rows: rows[0].__setitem__(
                "l1_displacement_from_settled_m", self.np.nextafter(0.0, 1.0)
            ),
        ):
            rows = copy.deepcopy(ledger)
            mutate(rows)
            with self.assertRaises(OscCanaryValidationError):
                _validate_paper_car_cadence_and_historical_binding(
                    rows,
                    action_count=1,
                    historical_settled_position_sha256=historical_hash,
                )

        tampered = self.row()
        tampered["l1_displacement_from_settled_m"] = self.np.nextafter(
            tampered["l1_displacement_from_settled_m"], self.np.inf
        )
        with self.assertRaises(OscCanaryValidationError):
            _validate_paper_car_endpoint_row(
                tampered,
                index=1,
                car_key="wine_bottle_obstacle_1_pos",
                obstacle_root_body_id=32,
                settled_car_position=[1.0, 2.0, 3.0],
                np=self.np,
            )


class OscArmLinkPartialActionLedgerTests(unittest.TestCase):
    def fixture(self, terminal_kind, completed_partial_substeps):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _float64_vector7_bytes_and_sha256,
        )

        complete_action = [float(value) for value in range(7)]
        partial_action = [value + 1.0 for value in complete_action]
        _, complete_hash = _float64_vector7_bytes_and_sha256(complete_action)
        _, partial_hash = _float64_vector7_bytes_and_sha256(partial_action)
        completed_observation = "b" * 64
        terminal_observation = "c" * 64
        actions = [
            {
                "source_action_index": 0,
                "executed_high_level_action": list(complete_action),
                "executed_high_level_action_sha256": complete_hash,
                "observation_sha256": completed_observation,
            }
        ]
        planner_actions = [
            {
                "source_action_index": 0,
                "executed": list(complete_action),
                "executed_action_sha256": complete_hash,
                "native_observation_sha256": "a" * 64,
            },
            {
                "source_action_index": 1,
                "executed": list(partial_action),
                "executed_action_sha256": partial_hash,
                "native_observation_sha256": completed_observation,
            },
        ]
        terminal_observation_kind = (
            "post_contact_observable_refresh"
            if terminal_kind == "literal_registered_forbidden_contact"
            else "post_method_stop_observable_refresh"
        )
        partial = {
            "schema_version": "vlsa_poisson_partial_action_record.v1",
            "terminal_kind": terminal_kind,
            "source_action_index": 1,
            "planner_action_trace_index": 1,
            "executed_high_level_action": list(partial_action),
            "executed_high_level_action_sha256": partial_hash,
            "planner_executed_action_sha256": partial_hash,
            "planner_native_observation_sha256": completed_observation,
            "pre_action_observation_sha256": completed_observation,
            "completed_physics_substeps": completed_partial_substeps,
            "physics_boundary_start": 25,
            "physics_boundary_end_exclusive": 25 + completed_partial_substeps,
            "terminal_observation_kind": terminal_observation_kind,
            "terminal_observation_sha256": terminal_observation,
            "terminal_forbidden_contact_categories": (
                ["link56_vs_external_nonrobot"]
                if terminal_kind == "literal_registered_forbidden_contact"
                else []
            ),
            "terminal_forbidden_contact_records_sha256": "d" * 64,
        }
        return actions, planner_actions, partial

    def validate(self, terminal_kind, completed_partial_substeps, mutate=None):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _validate_partial_action_ledger,
        )

        actions, planner, partial = self.fixture(
            terminal_kind, completed_partial_substeps
        )
        if mutate is not None:
            mutate(partial)
        return _validate_partial_action_ledger(
            terminal_kind=terminal_kind,
            actions=actions,
            planner_actions=planner,
            partial_action_record=partial,
            physics_substep_count=25 + completed_partial_substeps,
        )

    def test_contact_partial_action_is_executable_and_bound(self):
        audit = self.validate("literal_registered_forbidden_contact", 3)
        self.assertTrue(audit["present"])
        self.assertEqual(audit["completed_physics_substeps"], 3)

    def test_method_stop_partial_action_allows_zero_completed_substeps(self):
        audit = self.validate("safety_method_stop_before_physics", 0)
        self.assertTrue(audit["present"])
        self.assertEqual(audit["completed_physics_substeps"], 0)

    def test_partial_action_tampering_is_rejected(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        with self.assertRaises(OscCanaryValidationError):
            self.validate(
                "literal_registered_forbidden_contact",
                3,
                lambda row: row.__setitem__("planner_executed_action_sha256", "0" * 64),
            )

    def test_full_terminal_requires_no_partial_action(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _validate_partial_action_ledger,
        )

        audit = _validate_partial_action_ledger(
            terminal_kind="native_task_success",
            actions=[],
            planner_actions=[],
            partial_action_record=None,
            physics_substep_count=0,
        )
        self.assertFalse(audit["present"])


class OscArmLinkShieldSamplingBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
        except ImportError as error:
            raise unittest.SkipTest(
                "NumPy is required for shield-binding tests"
            ) from error
        cls.np = np

    def fixture(self):
        from scripts.run_poisson_osc_arm_link_canary import (
            _canonical_sha256,
            _compact_float64_array_record,
        )
        apparatus, resolved = (
            OscArmLinkStructuralGeomPartitionConsumerTests().fixture()
        )
        samples = apparatus["movable_manipulator_sampling"]["samples"]
        qvel_indices = list(range(9))
        qvel_records = [
            {
                "qvel_index": index,
                "joint_id": index,
                "joint_name": (
                    "robot0_joint%d" % (index + 1)
                    if index < 7
                    else "gripper0_finger_joint%d" % (index - 6)
                ),
                "joint_body_id": 2 if index < 7 else 3,
                "joint_body_name": (
                    "robot0_link5" if index < 7 else "gripper0_leftfinger"
                ),
            }
            for index in qvel_indices
        ]
        sample_hash = _canonical_sha256(samples)
        h = self.np.asarray([0.1, 0.2, 0.3], dtype=self.np.float64)
        world_points = self.np.asarray(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
            dtype=self.np.float64,
        )
        workspace_lower = self.np.asarray([-1.0, -1.0, -1.0])
        workspace_upper = self.np.asarray([1.0, 1.0, 1.0])
        outer_clearance = self.np.min(
            self.np.concatenate(
                (
                    world_points - workspace_lower[None, :],
                    workspace_upper[None, :] - world_points,
                ),
                axis=1,
            ),
            axis=1,
        )
        comparison_tolerance = (
            64.0 * float(self.np.finfo(self.np.float64).eps)
        )
        binding = {
            "schema_version": (
                "vlsa_poisson_movable_manipulator_shield_sampling_binding.v1"
            ),
            "scope": (
                "all_structurally_movable_manipulator_collision_surfaces_"
                "vs_selected_obstacle"
            ),
            "sample_source": "apparatus.movable_manipulator_sampling.samples",
            "sample_count": len(samples),
            "sample_ledger_sha256": sample_hash,
            "resolved_robot_geom_ids": [10, 20, 30],
            "resolved_robot_geom_ids_sha256": _canonical_sha256([10, 20, 30]),
            "shield_manipulator_geom_ids": [10, 20],
            "shield_manipulator_geom_ids_sha256": _canonical_sha256([10, 20]),
            "fixed_robot_infrastructure_geom_ids": [30],
            "fixed_robot_infrastructure_geom_ids_sha256": _canonical_sha256([30]),
            "robot_geom_influence_partition_sha256": apparatus[
                "robot_geom_influence_partition_sha256"
            ],
            "all_robot_contact_monitor_scope_identity_sha256": apparatus[
                "registered_contact_scope"
            ]["identity_sha256"],
            "robot_qvel_selection_rule": (
                "ascending_dof_index_whose_dof_joint_body_is_in_resolved_robot_body_ids"
            ),
            "robot_qvel_indices": qvel_indices,
            "robot_qvel_indices_sha256": _canonical_sha256(qvel_indices),
            "robot_qvel_records": qvel_records,
            "robot_qvel_records_sha256": _canonical_sha256(qvel_records),
            "velocity_dimension": len(qvel_indices),
            "arm_qvel_indices": list(range(7)),
            "arm_qvel_indices_included": True,
            "decision_arm_actuator_ids": list(range(30, 37)),
            "decision_dimension": 7,
            "nonarm_robot_qvel_included": True,
            "nonarm_ctrl_policy": (
                "unchanged_byte_exact_in_all_cloned_and_live_transitions"
            ),
            "model_actuator_count": 37,
            "nonarm_ctrl_selection_rule": (
                "ascending_actuator_index_excluding_decision_arm_actuator_ids"
            ),
            "nonarm_ctrl_indices": list(range(30)),
            "nonarm_ctrl_indices_sha256": _canonical_sha256(list(range(30))),
            "nonarm_ctrl_count": 30,
            "live_nonarm_ctrl_array_hash_format": (
                "sha256_vlsa-table1-array-v1_header_and_c_order_float64_bytes"
            ),
            "field_seed_sample_scope": "link56_only_not_shield_scope",
            "settled_field_query_certificate": {
                "schema_version": (
                    "vlsa_poisson_movable_manipulator_settled_field_query.v1"
                ),
                "sample_count": len(samples),
                "sample_ledger_sha256": sample_hash,
                "all_queries_valid": True,
                "all_h_strictly_positive": True,
                "minimum_h_m2": float(self.np.min(h)),
                "h_m2": h.tolist(),
                "h_array_record": _compact_float64_array_record(h, self.np),
                "world_points_m": world_points.tolist(),
                "world_points_array_record": _compact_float64_array_record(
                    world_points, self.np
                ),
                "outer_boundary_clearance_m": outer_clearance.tolist(),
                "outer_boundary_clearance_array_record": (
                    _compact_float64_array_record(outer_clearance, self.np)
                ),
                "required_outer_boundary_clearance_m": 0.5,
                "minimum_outer_boundary_clearance_m": float(
                    self.np.min(outer_clearance)
                ),
                "outer_boundary_clearance_comparison_tolerance_m": (
                    comparison_tolerance
                ),
                "all_outer_boundary_clearances_pass": True,
            },
            "settled_zero_jacobian_sample_count": 0,
            "settled_nonzero_jacobian_sample_count": len(samples),
        }
        apparatus["shield_sampling_binding"] = binding
        apparatus["shield_sampling_binding_sha256"] = _canonical_sha256(binding)
        controller = {
            "arm_qvel_indexes": list(range(7)),
            "arm_actuator_indexes": list(range(30, 37)),
        }
        runtime_protocol = {
            "workspace": {
                "minimum_m": workspace_lower.tolist(),
                "maximum_m": workspace_upper.tolist(),
            },
            "occupancy": {"outer_boundary_clearance_m": 0.5},
        }
        return apparatus, resolved, controller, runtime_protocol

    def validate(self, fixture):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _validate_shield_sampling_binding,
        )

        apparatus, resolved, controller, runtime_protocol = fixture
        return _validate_shield_sampling_binding(
            apparatus=apparatus,
            resolved_geometry=resolved,
            controller=controller,
            runtime_protocol=runtime_protocol,
            np=self.np,
        )

    @staticmethod
    def refresh_binding_hash(apparatus):
        from scripts.run_poisson_osc_arm_link_canary import _canonical_sha256

        apparatus["movable_manipulator_sampling_sha256"] = _canonical_sha256(
            apparatus["movable_manipulator_sampling"]
        )
        apparatus["shield_sampling_binding_sha256"] = _canonical_sha256(
            apparatus["shield_sampling_binding"]
        )

    def test_movable_binding_accepts_finger_samples_and_nonarm_qvels(self):
        audit = self.validate(self.fixture())
        self.assertEqual(audit["sample_count"], 3)
        self.assertEqual(audit["velocity_dimension"], 9)
        self.assertEqual(audit["arm_qvel_indices"], list(range(7)))

    def test_link_only_sample_substitution_is_rejected_even_when_rehashed(self):
        from scripts.run_poisson_osc_arm_link_canary import _canonical_sha256
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        fixture = self.fixture()
        apparatus = fixture[0]
        samples = apparatus["movable_manipulator_sampling"]["samples"][:2]
        sample_hash = _canonical_sha256(samples)
        apparatus["movable_manipulator_sampling"].update(
            samples=samples,
            sample_count=len(samples),
            sample_ledger_sha256=sample_hash,
        )
        apparatus["shield_sampling_binding"].update(
            sample_count=len(samples), sample_ledger_sha256=sample_hash
        )
        self.refresh_binding_hash(apparatus)
        with self.assertRaisesRegex(
            OscCanaryValidationError, "exact reindexed all-robot subsequence"
        ):
            self.validate(fixture)

    def test_reordered_finger_sample_rows_are_rejected_even_when_rehashed(self):
        from scripts.run_poisson_osc_arm_link_canary import _canonical_sha256
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        fixture = self.fixture()
        apparatus = fixture[0]
        samples = apparatus["movable_manipulator_sampling"]["samples"]
        reordered = [
            copy.deepcopy(samples[2]),
            copy.deepcopy(samples[0]),
            copy.deepcopy(samples[1]),
        ]
        for sample_id, sample in enumerate(reordered):
            sample["sample_id"] = sample_id
        sample_hash = _canonical_sha256(reordered)
        apparatus["movable_manipulator_sampling"].update(
            samples=reordered, sample_ledger_sha256=sample_hash
        )
        apparatus["shield_sampling_binding"].update(
            sample_ledger_sha256=sample_hash
        )
        self.refresh_binding_hash(apparatus)
        with self.assertRaisesRegex(
            OscCanaryValidationError, "exact reindexed all-robot subsequence"
        ):
            self.validate(fixture)

    def test_qvel_reorder_or_omitted_finger_record_is_rejected(self):
        from scripts.run_poisson_osc_arm_link_canary import _canonical_sha256
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        fixture = self.fixture()
        apparatus = fixture[0]
        binding = apparatus["shield_sampling_binding"]
        binding["robot_qvel_indices"][-2:] = [8, 7]
        binding["robot_qvel_records"][-2:] = list(
            reversed(binding["robot_qvel_records"][-2:])
        )
        binding["robot_qvel_indices_sha256"] = _canonical_sha256(
            binding["robot_qvel_indices"]
        )
        binding["robot_qvel_records_sha256"] = _canonical_sha256(
            binding["robot_qvel_records"]
        )
        self.refresh_binding_hash(apparatus)
        with self.assertRaises(OscCanaryValidationError):
            self.validate(fixture)

        fixture = self.fixture()
        apparatus = fixture[0]
        apparatus["shield_sampling_binding"]["robot_qvel_records"].pop()
        self.refresh_binding_hash(apparatus)
        with self.assertRaises(OscCanaryValidationError):
            self.validate(fixture)

    def test_schema_qvel_hash_dimension_and_settled_h_tampering_are_rejected(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        mutations = (
            lambda binding: binding.__setitem__("schema_version", "v0"),
            lambda binding: binding.__setitem__("robot_qvel_indices_sha256", "0" * 64),
            lambda binding: binding.__setitem__("velocity_dimension", 7),
            lambda binding: binding["settled_field_query_certificate"].__setitem__(
                "minimum_h_m2", 0.2
            ),
            lambda binding: binding["settled_field_query_certificate"][
                "h_array_record"
            ].__setitem__("sha256", "0" * 64),
            lambda binding: binding["settled_field_query_certificate"].__setitem__(
                "required_outer_boundary_clearance_m", 0.6
            ),
            lambda binding: binding["settled_field_query_certificate"][
                "outer_boundary_clearance_array_record"
            ].__setitem__("sha256", "0" * 64),
            lambda binding: binding["settled_field_query_certificate"].__setitem__(
                "all_outer_boundary_clearances_pass", False
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                fixture = self.fixture()
                apparatus = fixture[0]
                mutate(apparatus["shield_sampling_binding"])
                self.refresh_binding_hash(apparatus)
                with self.assertRaises(OscCanaryValidationError):
                    self.validate(fixture)

    def test_rehashed_outer_clearance_not_matching_world_points_is_rejected(self):
        from scripts.run_poisson_osc_arm_link_canary import (
            _compact_float64_array_record,
        )
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        fixture = self.fixture()
        apparatus = fixture[0]
        settled = apparatus["shield_sampling_binding"][
            "settled_field_query_certificate"
        ]
        fabricated = self.np.asarray(
            settled["outer_boundary_clearance_m"], dtype=self.np.float64
        )
        fabricated[0] += 0.1
        settled["outer_boundary_clearance_m"] = fabricated.tolist()
        settled["outer_boundary_clearance_array_record"] = (
            _compact_float64_array_record(fabricated, self.np)
        )
        settled["minimum_outer_boundary_clearance_m"] = float(
            self.np.min(fabricated)
        )
        self.refresh_binding_hash(apparatus)
        with self.assertRaisesRegex(
            OscCanaryValidationError, "runtime outer-boundary clearance"
        ):
            self.validate(fixture)


class OscArmLinkStructuralGeomPartitionConsumerTests(unittest.TestCase):
    """Adversarial checks for the v3 movable/fixed consumer boundary."""

    def fixture(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _canonical_sha256,
            _compact_zero_float64_array_record,
        )

        qvel_indices = list(range(9))
        qvel_joint_body_ids = [2] * 7 + [3, 3]
        qvel_records = [
            {
                "qvel_index": qvel_index,
                "joint_body_id": joint_body_id,
            }
            for qvel_index, joint_body_id in zip(
                qvel_indices, qvel_joint_body_ids
            )
        ]
        full_samples = [
            {
                "sample_id": 0,
                "body_id": 2,
                "body_name": "robot0_link5",
                "geom_id": 10,
                "geom_name": "robot0_link5_collision",
                "point_body_local_m": [0.0, 0.0, 0.0],
                "source": "collision_geom_surface",
            },
            {
                "sample_id": 1,
                "body_id": 2,
                "body_name": "robot0_link5",
                "geom_id": 10,
                "geom_name": "robot0_link5_collision",
                "point_body_local_m": [0.01, 0.0, 0.0],
                "source": "collision_geom_surface",
            },
            {
                "sample_id": 2,
                "body_id": 3,
                "body_name": "gripper0_leftfinger",
                "geom_id": 20,
                "geom_name": "gripper0_leftfinger_collision",
                "point_body_local_m": [0.0, 0.01, 0.0],
                "source": "collision_geom_surface",
            },
            {
                "sample_id": 3,
                "body_id": 1,
                "body_name": "mount0_controller_box",
                "geom_id": 30,
                "geom_name": "mount0_controller_box_col",
                "point_body_local_m": [-0.435, -0.2, -0.645],
                "source": "collision_geom_surface",
            },
        ]
        full_hash = _canonical_sha256(full_samples)
        movable_samples = copy.deepcopy(full_samples[:3])
        fixed_samples = [copy.deepcopy(full_samples[3])]
        fixed_samples[0]["sample_id"] = 0

        def surface_evidence(samples, geom_ids):
            return {
                "sample_count": len(samples),
                "sample_ledger_sha256": _canonical_sha256(samples),
                "samples": samples,
                "geom_records": [{"geom_id": geom_id} for geom_id in geom_ids],
                "epsilon_m": 0.05,
                "maximum_surface_cover_radius_m": 0.04,
                "coverage_semantics": "test_surface_coverage",
                "roundtrip": {"passed": True},
            }
        partition = {
            "schema_version": "vlsa_poisson_robot_geom_influence_partition.v1",
            "selection_rule": (
                "geom_body_self_or_ancestor_owns_at_least_one_authoritative_"
                "robot_tree_qvel"
            ),
            "contact_monitor_robot_geom_ids": [10, 20, 30],
            "shield_manipulator_geom_ids": [10, 20],
            "fixed_robot_infrastructure_geom_ids": [30],
            "robot_qvel_indices": qvel_indices,
            "robot_qvel_joint_body_ids": qvel_joint_body_ids,
            "robot_body_parent_records": [
                {
                    "body_id": 1,
                    "body_name": "mount0_controller_box",
                    "parent_body_id": 0,
                    "parent_body_name": "world",
                    "body_ancestry_ids": [1],
                },
                {
                    "body_id": 2,
                    "body_name": "robot0_link5",
                    "parent_body_id": 1,
                    "parent_body_name": "mount0_controller_box",
                    "body_ancestry_ids": [2, 1],
                },
                {
                    "body_id": 3,
                    "body_name": "gripper0_leftfinger",
                    "parent_body_id": 2,
                    "parent_body_name": "robot0_link5",
                    "body_ancestry_ids": [3, 2, 1],
                },
            ],
            "geom_records": [
                {
                    "geom_id": 10,
                    "geom_name": "robot0_link5_collision",
                    "body_id": 2,
                    "body_name": "robot0_link5",
                    "body_ancestry_ids": [2, 1],
                    "influencing_robot_qvel_indices": list(range(7)),
                    "classification": (
                        "kinematically_movable_manipulator_surface"
                    ),
                },
                {
                    "geom_id": 20,
                    "geom_name": "gripper0_leftfinger_collision",
                    "body_id": 3,
                    "body_name": "gripper0_leftfinger",
                    "body_ancestry_ids": [3, 2, 1],
                    "influencing_robot_qvel_indices": qvel_indices,
                    "classification": (
                        "kinematically_movable_manipulator_surface"
                    ),
                },
                {
                    "geom_id": 30,
                    "geom_name": "mount0_controller_box_col",
                    "body_id": 1,
                    "body_name": "mount0_controller_box",
                    "body_ancestry_ids": [1],
                    "influencing_robot_qvel_indices": [],
                    "classification": (
                        "kinematically_fixed_robot_infrastructure"
                    ),
                },
            ],
            "all_contact_geoms_partitioned": True,
            "all_fixed_geoms_have_zero_structural_qvel_influence": True,
        }
        partition_hash = _canonical_sha256(partition)
        all_robot = surface_evidence(full_samples, [10, 20, 30])
        movable = surface_evidence(movable_samples, [10, 20])
        fixed_sampling = surface_evidence(fixed_samples, [30])
        scope_without_hash = {"robot_geom_ids": [10, 20, 30]}
        scope = {
            **scope_without_hash,
            "identity_sha256": _canonical_sha256(scope_without_hash),
        }
        fixed = {
            "schema_version": (
                "vlsa_poisson_fixed_infrastructure_settled_certificate.v1"
            ),
            "structural_partition_sha256": partition_hash,
            "fixed_geom_ids": [30],
            "fixed_geom_ids_sha256": _canonical_sha256([30]),
            "fixed_sample_ids": [0],
            "fixed_sample_ids_sha256": _canonical_sha256([0]),
            "fixed_sample_count": 1,
            "fixed_sample_ledger_sha256": fixed_sampling["sample_ledger_sha256"],
            "fixed_world_points_array_record": (
                _compact_zero_float64_array_record((1, 3))
            ),
            "fixed_point_jacobian_array_record": (
                _compact_zero_float64_array_record((1, 3, 9))
            ),
            "all_fixed_geoms_zero_structural_qvel_influence": True,
            "all_fixed_sample_jacobians_exactly_zero": True,
            "maximum_abs_fixed_sample_jacobian": 0.0,
            "settled_registered_contact_count": 0,
            "settled_contact_records": [],
            "settled_contact_free": True,
            "all_fixed_geoms_contact_monitored": True,
            "contact_monitor_scope_identity_sha256": scope["identity_sha256"],
        }
        apparatus = {
            "all_robot_sampling": all_robot,
            "all_robot_sampling_sha256": _canonical_sha256(all_robot),
            "shield_sampling_binding": {
                "robot_qvel_indices": qvel_indices,
                "robot_qvel_records": qvel_records,
            },
            "robot_geom_influence_partition": partition,
            "robot_geom_influence_partition_sha256": partition_hash,
            "movable_manipulator_sampling": movable,
            "movable_manipulator_sampling_sha256": _canonical_sha256(movable),
            "fixed_infrastructure_sampling": fixed_sampling,
            "fixed_infrastructure_sampling_sha256": _canonical_sha256(
                fixed_sampling
            ),
            "fixed_infrastructure_settled_certificate": fixed,
            "fixed_infrastructure_settled_certificate_sha256": (
                _canonical_sha256(fixed)
            ),
            "registered_contact_scope": scope,
        }
        resolved = {
            "robot_root_body_ids": [1],
            "robot_geom_ids": [10, 20, 30],
            "robot_geom_names": [
                "robot0_link5_collision",
                "gripper0_leftfinger_collision",
                "mount0_controller_box_col",
            ],
            "robot_body_ids": [1, 2, 3],
            "robot_body_names": [
                "mount0_controller_box",
                "robot0_link5",
                "gripper0_leftfinger",
            ],
            "link56_geom_ids": [10],
            "collision_enabled_pairs": [[10, 100], [20, 100], [30, 100]],
        }
        return apparatus, resolved

    @staticmethod
    def refresh_hashes(apparatus):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _canonical_sha256,
        )

        partition = apparatus["robot_geom_influence_partition"]
        movable = apparatus["movable_manipulator_sampling"]
        fixed = apparatus["fixed_infrastructure_settled_certificate"]
        partition_hash = _canonical_sha256(partition)
        apparatus["robot_geom_influence_partition_sha256"] = partition_hash
        fixed["structural_partition_sha256"] = partition_hash
        apparatus["movable_manipulator_sampling_sha256"] = _canonical_sha256(
            movable
        )
        apparatus["fixed_infrastructure_settled_certificate_sha256"] = (
            _canonical_sha256(fixed)
        )

    @staticmethod
    def validate(fixture):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _validate_structural_robot_geom_partition,
        )

        apparatus, resolved = fixture
        return _validate_structural_robot_geom_partition(
            apparatus=apparatus, resolved_geometry=resolved
        )

    def test_complete_structural_partition_reconstructs_all_three_positive_gates(self):
        audit = self.validate(self.fixture())
        self.assertEqual(audit["movable_geom_ids"], [10, 20])
        self.assertEqual(audit["fixed_geom_ids"], [30])
        self.assertEqual(audit["movable_sample_count"], 3)
        self.assertEqual(audit["fixed_sample_count"], 1)
        for field in (
            "all_structurally_movable_manipulator_collision_surfaces_shielded",
            "excluded_fixed_infrastructure_zero_qvel_influence_and_settled_contact_free",
            "all_authoritative_robot_collision_surfaces_contact_monitored",
        ):
            self.assertIs(audit[field], True)

    def test_link_or_finger_reclassification_is_rejected_after_rehash(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        for geom_index, qvels in ((0, []), (1, list(range(7)))):
            with self.subTest(geom_index=geom_index):
                fixture = self.fixture()
                apparatus = fixture[0]
                record = apparatus["robot_geom_influence_partition"]["geom_records"][
                    geom_index
                ]
                record["influencing_robot_qvel_indices"] = qvels
                if not qvels:
                    record["classification"] = (
                        "kinematically_fixed_robot_infrastructure"
                    )
                self.refresh_hashes(apparatus)
                with self.assertRaisesRegex(
                    OscCanaryValidationError, "structural influence"
                ):
                    self.validate(fixture)

    def test_partition_overlap_omission_or_link56_exclusion_is_rejected(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        mutations = (
            lambda partition: partition["fixed_robot_infrastructure_geom_ids"].append(
                20
            ),
            lambda partition: partition["contact_monitor_robot_geom_ids"].pop(),
            lambda partition: (
                partition["shield_manipulator_geom_ids"].remove(10),
                partition["fixed_robot_infrastructure_geom_ids"].insert(0, 10),
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                fixture = self.fixture()
                apparatus = fixture[0]
                mutate(apparatus["robot_geom_influence_partition"])
                self.refresh_hashes(apparatus)
                with self.assertRaises(OscCanaryValidationError):
                    self.validate(fixture)

    def test_movable_samples_must_be_exact_reindexed_full_ledger_subsequence(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
            _canonical_sha256,
        )

        fixture = self.fixture()
        apparatus = fixture[0]
        movable = apparatus["movable_manipulator_sampling"]
        movable["samples"].pop(2)
        movable["geom_records"].pop(1)
        movable["sample_count"] = len(movable["samples"])
        movable["sample_ledger_sha256"] = _canonical_sha256(movable["samples"])
        self.refresh_hashes(apparatus)
        with self.assertRaisesRegex(
            OscCanaryValidationError, "exact reindexed all-robot subsequence"
        ):
            self.validate(fixture)

    def test_fixed_certificate_contact_jacobian_pair_and_monitor_mutations_fail(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        def contact_mutation(fixed):
            fixed["settled_contact_records"] = [{"robot_geom_id": 30}]
            fixed["settled_registered_contact_count"] = 1
            fixed["settled_contact_free"] = False

        mutations = (
            lambda fixed: fixed.__setitem__(
                "maximum_abs_fixed_sample_jacobian", 1e-12
            ),
            lambda fixed: fixed.__setitem__(
                "all_fixed_sample_jacobians_exactly_zero", False
            ),
            contact_mutation,
            lambda fixed: fixed.__setitem__(
                "all_fixed_geoms_contact_monitored", False
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                fixture = self.fixture()
                apparatus = fixture[0]
                mutate(apparatus["fixed_infrastructure_settled_certificate"])
                self.refresh_hashes(apparatus)
                with self.assertRaisesRegex(
                    OscCanaryValidationError,
                    "fixed infrastructure zero-influence or settled contact-free",
                ):
                    self.validate(fixture)

    def test_contact_scope_cannot_drop_fixed_infrastructure(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        fixture = self.fixture()
        apparatus = fixture[0]
        scope = apparatus["registered_contact_scope"]
        scope["robot_geom_ids"] = [10, 20]
        scope_without_hash = dict(scope)
        scope_without_hash.pop("identity_sha256")
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _canonical_sha256,
        )

        scope["identity_sha256"] = _canonical_sha256(scope_without_hash)
        apparatus["fixed_infrastructure_settled_certificate"][
            "contact_monitor_scope_identity_sha256"
        ] = scope["identity_sha256"]
        self.refresh_hashes(apparatus)
        with self.assertRaisesRegex(
            OscCanaryValidationError, "contact scope omits"
        ):
            self.validate(fixture)

    def test_collusive_qvel_owner_mutation_is_rejected(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        fixture = self.fixture()
        apparatus = fixture[0]
        apparatus["robot_geom_influence_partition"][
            "robot_qvel_joint_body_ids"
        ][0] = 1
        apparatus["shield_sampling_binding"]["robot_qvel_records"][0][
            "joint_body_id"
        ] = 1
        self.refresh_hashes(apparatus)
        with self.assertRaisesRegex(
            OscCanaryValidationError, "structural influence"
        ):
            self.validate(fixture)

    def test_robot_root_with_external_parent_is_rejected_after_rehash(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        fixture = self.fixture()
        apparatus = fixture[0]
        partition = apparatus["robot_geom_influence_partition"]
        for record in partition["robot_body_parent_records"]:
            record["body_ancestry_ids"].append(99)
            if record["body_id"] == 1:
                record["parent_body_id"] = 99
                record["parent_body_name"] = "external_mobile_parent"
        for record in partition["geom_records"]:
            record["body_ancestry_ids"].append(99)
        self.refresh_hashes(apparatus)
        with self.assertRaisesRegex(
            OscCanaryValidationError, "roots are not world-mounted"
        ):
            self.validate(fixture)


class OscArmLinkRobotTreeQvelAuthorityTests(unittest.TestCase):
    def fixture(self):
        model = SimpleNamespace(
            nv=10,
            dof_jntid=list(range(10)),
            jnt_bodyid=[1, 1, 1, 1, 1, 1, 1, 2, 3, 99],
        )
        resolved = SimpleNamespace(robot_body_ids=(1, 2, 3))
        fake_mujoco = SimpleNamespace(
            mjtObj=SimpleNamespace(mjOBJ_JOINT=0, mjOBJ_BODY=1),
            mj_id2name=lambda unused_model, object_type, object_id: (
                "joint_%d" % object_id
                if object_type == 0
                else "body_%d" % object_id
            ),
        )
        return model, resolved, fake_mujoco

    def bind(self, *, arm_qvel_indices=None, arm_actuator_ids=None):
        from unittest import mock

        from scripts.run_poisson_osc_arm_link_canary import (
            _robot_tree_qvel_binding,
        )

        model, resolved, fake_mujoco = self.fixture()
        with mock.patch.dict("sys.modules", {"mujoco": fake_mujoco}):
            return _robot_tree_qvel_binding(
                model,
                resolved,
                arm_qvel_indices=(
                    list(range(7))
                    if arm_qvel_indices is None
                    else arm_qvel_indices
                ),
                arm_actuator_ids=(
                    list(range(7))
                    if arm_actuator_ids is None
                    else arm_actuator_ids
                ),
            )

    def test_model_authority_selects_every_robot_dof_in_ascending_order(self):
        binding = self.bind()
        self.assertEqual(binding["robot_qvel_indices"], list(range(9)))
        self.assertEqual(
            [row["qvel_index"] for row in binding["robot_qvel_records"]],
            list(range(9)),
        )
        self.assertNotIn(9, binding["robot_qvel_indices"])
        self.assertEqual(binding["robot_qvel_records"][-1]["joint_body_id"], 3)

    def test_omitted_duplicated_or_nonrobot_arm_dof_cannot_bind(self):
        from scripts.run_poisson_osc_arm_link_canary import OscCanaryRunnerError

        adversarial_arm_qvel = (
            list(range(6)),
            [0, 1, 2, 3, 4, 5, 5],
            [0, 1, 2, 3, 4, 5, 9],
        )
        for arm_qvel_indices in adversarial_arm_qvel:
            with self.subTest(arm_qvel_indices=arm_qvel_indices):
                with self.assertRaises(OscCanaryRunnerError):
                    self.bind(arm_qvel_indices=arm_qvel_indices)

    def test_nonrobot_dof_is_excluded_and_serialized_reorder_is_rejected(self):
        from scripts.run_poisson_osc_arm_link_canary import _canonical_sha256
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        binding = self.bind()
        self.assertNotIn(9, binding["robot_qvel_indices"])
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy is unavailable in the local structural environment")

        binding_validator = OscArmLinkShieldSamplingBindingTests()
        binding_validator.np = np
        fixture = binding_validator.fixture()
        apparatus = fixture[0]
        serialized = apparatus["shield_sampling_binding"]
        serialized["robot_qvel_indices"][-2:] = [8, 7]
        serialized["robot_qvel_indices_sha256"] = _canonical_sha256(
            serialized["robot_qvel_indices"]
        )
        apparatus["shield_sampling_binding_sha256"] = _canonical_sha256(
            serialized
        )
        with self.assertRaises(OscCanaryValidationError):
            binding_validator.validate(fixture)

        fixture = binding_validator.fixture()
        apparatus = fixture[0]
        serialized = apparatus["shield_sampling_binding"]
        serialized["robot_qvel_indices"][-1] = 9
        serialized["robot_qvel_records"][-1].update(
            qvel_index=9,
            joint_id=9,
            joint_name="object_free_joint",
            joint_body_id=99,
            joint_body_name="object_body",
        )
        serialized["robot_qvel_indices_sha256"] = _canonical_sha256(
            serialized["robot_qvel_indices"]
        )
        serialized["robot_qvel_records_sha256"] = _canonical_sha256(
            serialized["robot_qvel_records"]
        )
        apparatus["shield_sampling_binding_sha256"] = _canonical_sha256(
            serialized
        )
        with self.assertRaises(OscCanaryValidationError):
            binding_validator.validate(fixture)


class OscArmLinkMethodStopBindingTests(unittest.TestCase):
    def fixture(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _canonical_sha256,
        )

        qvel_indices = list(range(9))
        record = {
            "schema_version": "vlsa_poisson_safety_method_stop.v2",
            "source_action_index": 2,
            "physics_substep_index": 3,
            "physics_boundary_before_unexecuted_step": 53,
            "unexecuted_post_integration_boundary": 54,
            "reason": "candidate_clone_cbf_residual_stop_before_physics",
            "sample_count": 3,
            "sample_ledger_sha256": "a" * 64,
            "robot_qvel_indices": qvel_indices,
            "robot_qvel_indices_sha256": _canonical_sha256(qvel_indices),
            "velocity_dimension": len(qvel_indices),
            "candidate_minimum_cbf_residual_m2_per_s": -0.01,
        }
        record["record_payload_sha256"] = _canonical_sha256(record)
        return record

    def validate(self, record, *, terminal_kind="safety_method_stop_before_physics"):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _canonical_sha256,
            _validate_safety_method_stop,
        )

        return _validate_safety_method_stop(
            record,
            terminal_kind=terminal_kind,
            action_count=2,
            physics_substep_count=53,
            sample_count=3,
            sample_ledger_sha256="a" * 64,
            robot_qvel_indices=list(range(9)),
            robot_qvel_indices_sha256=_canonical_sha256(list(range(9))),
            velocity_dimension=9,
        )

    def test_method_stop_binds_unexecuted_candidate_boundary_and_full_scope(self):
        self.assertTrue(self.validate(self.fixture()))

    def test_method_stop_rejects_schema_hash_boundary_or_qvel_tampering(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
            _canonical_sha256,
        )

        def rehash(record):
            record.pop("record_payload_sha256", None)
            record["record_payload_sha256"] = _canonical_sha256(record)

        mutations = (
            lambda record: record.__setitem__("schema_version", "v1"),
            lambda record: record.__setitem__(
                "physics_boundary_before_unexecuted_step", 52
            ),
            lambda record: record.__setitem__("sample_ledger_sha256", "b" * 64),
            lambda record: record.__setitem__(
                "robot_qvel_indices", list(range(7)) + [8, 7]
            ),
            lambda record: record.__setitem__("velocity_dimension", 7),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                record = self.fixture()
                mutate(record)
                if record["robot_qvel_indices"] != list(range(9)):
                    record["robot_qvel_indices_sha256"] = _canonical_sha256(
                        record["robot_qvel_indices"]
                    )
                rehash(record)
                with self.assertRaises(OscCanaryValidationError):
                    self.validate(record)

        stale_hash = self.fixture()
        stale_hash["reason"] = "different_stop_before_physics"
        with self.assertRaises(OscCanaryValidationError):
            self.validate(stale_hash)

    def test_nonmethod_terminal_rejects_stale_method_stop_record(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        with self.assertRaises(OscCanaryValidationError):
            self.validate(self.fixture(), terminal_kind="native_task_success")


class OscArmLinkLiveNonarmControlEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
        except ImportError as error:
            raise unittest.SkipTest(
                "NumPy is required for live non-arm control tests"
            ) from error
        cls.np = np

    def fixture(self):
        from scripts.run_poisson_osc_arm_link_canary import (
            _canonical_sha256,
            _compact_float64_array_record,
        )

        values = self.np.asarray([0.25, -0.5], dtype=self.np.float64)
        record = _compact_float64_array_record(values, self.np)
        indices = [7, 8]
        return {
            "live_nonarm_ctrl_count": 2,
            "live_nonarm_ctrl_indices_sha256": _canonical_sha256(indices),
            "live_nonarm_ctrl_before_array_record": copy.deepcopy(record),
            "live_nonarm_ctrl_after_array_record": copy.deepcopy(record),
            "live_nonarm_ctrl_byte_identical": True,
        }

    def validate(self, row):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _canonical_sha256,
            _validate_live_nonarm_ctrl_evidence,
        )

        return _validate_live_nonarm_ctrl_evidence(
            row,
            expected_nonarm_ctrl_indices=[7, 8],
            expected_nonarm_ctrl_indices_sha256=_canonical_sha256([7, 8]),
            np=self.np,
        )

    def test_live_nonarm_before_after_identity_passes(self):
        self.assertIsNone(self.validate(self.fixture()))

    def test_live_nonarm_changed_hash_flag_shape_or_authority_is_rejected(self):
        from scripts.run_poisson_osc_arm_link_canary import (
            _compact_float64_array_record,
        )
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        mutations = (
            lambda row: row.__setitem__(
                "live_nonarm_ctrl_after_array_record",
                _compact_float64_array_record(
                    self.np.asarray([0.25, -0.4], dtype=self.np.float64),
                    self.np,
                ),
            ),
            lambda row: row.__setitem__("live_nonarm_ctrl_byte_identical", False),
            lambda row: row.__setitem__(
                "live_nonarm_ctrl_indices_sha256", "0" * 64
            ),
            lambda row: row.__setitem__("live_nonarm_ctrl_count", 1),
            lambda row: row["live_nonarm_ctrl_after_array_record"].__setitem__(
                "shape", [1]
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                row = self.fixture()
                mutate(row)
                with self.assertRaises(OscCanaryValidationError):
                    self.validate(row)


class OscArmLinkCompactConstraintTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
        except ImportError as error:
            raise unittest.SkipTest("NumPy is required for compact trace tests") from error
        cls.np = np

    def trace_row(self):
        from scripts.run_poisson_osc_arm_link_canary import (
            _canonical_sha256,
            _compact_float64_array_record,
        )

        np = self.np
        count = 3
        robot_qvel_indices = list(range(9))
        sample_ledger_sha256 = "a" * 64
        values = {
            "poisson_h_m2": np.asarray([0.1, 0.2, 0.3]),
            "joint_gradient_rows_m2_per_rad": np.arange(27).reshape(3, 9),
            "actual_hdot_m2_per_s": np.asarray([-0.3, 0.0, 0.2]),
            "candidate_exact_clone_hdot_m2_per_s": np.asarray([-0.2, 0.1, 0.3]),
            "nominal_exact_clone_hdot_m2_per_s": np.asarray([-0.4, 0.0, 0.1]),
        }
        return {
            "constraint_trace": {
                "schema_version": "vlsa_poisson_compact_constraint_trace.v2",
                "array_hash_format": (
                    "sha256_vlsa-table1-array-v1_header_and_c_order_float64_bytes"
                ),
                "sample_count": count,
                "sample_ledger_sha256": sample_ledger_sha256,
                "robot_qvel_indices": robot_qvel_indices,
                "robot_qvel_indices_sha256": _canonical_sha256(
                    robot_qvel_indices
                ),
                "velocity_dimension": len(robot_qvel_indices),
                "arrays": {
                    name: _compact_float64_array_record(value, np)
                    for name, value in values.items()
                },
            }
        }

    def test_compact_trace_accepts_hashes_shapes_and_minima(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _canonical_sha256,
            _validate_compact_constraint_trace,
        )

        arrays = _validate_compact_constraint_trace(
            self.trace_row(),
            expected_sample_count=3,
            expected_sample_ledger_sha256="a" * 64,
            expected_robot_qvel_indices=list(range(9)),
            expected_robot_qvel_indices_sha256=_canonical_sha256(list(range(9))),
            expected_velocity_dimension=9,
            np=self.np,
        )
        self.assertEqual(arrays["joint_gradient_rows_m2_per_rad"]["shape"], [3, 9])

    def test_compact_trace_rejects_hash_tampering(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
            _canonical_sha256,
            _validate_compact_constraint_trace,
        )

        row = self.trace_row()
        row["constraint_trace"]["arrays"]["poisson_h_m2"]["sha256"] = "bad"
        with self.assertRaises(OscCanaryValidationError):
            _validate_compact_constraint_trace(
                row,
                expected_sample_count=3,
                expected_sample_ledger_sha256="a" * 64,
                expected_robot_qvel_indices=list(range(9)),
                expected_robot_qvel_indices_sha256=_canonical_sha256(
                    list(range(9))
                ),
                expected_velocity_dimension=9,
                np=self.np,
            )

    def test_compact_trace_rejects_stale_schema_qvel_reorder_or_dimension(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
            _canonical_sha256,
            _validate_compact_constraint_trace,
        )

        def validate(row):
            return _validate_compact_constraint_trace(
                row,
                expected_sample_count=3,
                expected_sample_ledger_sha256="a" * 64,
                expected_robot_qvel_indices=list(range(9)),
                expected_robot_qvel_indices_sha256=_canonical_sha256(
                    list(range(9))
                ),
                expected_velocity_dimension=9,
                np=self.np,
            )

        stale = self.trace_row()
        stale["constraint_trace"]["schema_version"] = (
            "vlsa_poisson_compact_constraint_trace.v1"
        )
        with self.assertRaises(OscCanaryValidationError):
            validate(stale)

        reordered = self.trace_row()
        reordered_indices = list(range(7)) + [8, 7]
        reordered["constraint_trace"]["robot_qvel_indices"] = reordered_indices
        reordered["constraint_trace"]["robot_qvel_indices_sha256"] = (
            _canonical_sha256(reordered_indices)
        )
        with self.assertRaises(OscCanaryValidationError):
            validate(reordered)

        downgraded = self.trace_row()
        downgraded["constraint_trace"]["velocity_dimension"] = 7
        with self.assertRaises(OscCanaryValidationError):
            validate(downgraded)

    def test_compact_trace_rejects_link_only_gradient_width(self):
        from scripts.run_poisson_osc_arm_link_canary import (
            _compact_float64_array_record,
        )
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
            _canonical_sha256,
            _validate_compact_constraint_trace,
        )

        row = self.trace_row()
        row["constraint_trace"]["arrays"][
            "joint_gradient_rows_m2_per_rad"
        ] = _compact_float64_array_record(self.np.zeros((3, 7)), self.np)
        with self.assertRaises(OscCanaryValidationError):
            _validate_compact_constraint_trace(
                row,
                expected_sample_count=3,
                expected_sample_ledger_sha256="a" * 64,
                expected_robot_qvel_indices=list(range(9)),
                expected_robot_qvel_indices_sha256=_canonical_sha256(
                    list(range(9))
                ),
                expected_velocity_dimension=9,
                np=self.np,
            )

    def test_first_divergence_constraint_body_is_independently_attributed(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _independent_constraint_attribution,
        )

        class Sample:
            def __init__(self, sample_id, body_name):
                self.sample_id = sample_id
                self.body_id = sample_id + 10
                self.body_name = body_name
                self.geom_id = sample_id + 20
                self.geom_name = body_name + "_collision"

            def to_dict(self):
                return {
                    "sample_id": self.sample_id,
                    "body_id": self.body_id,
                    "body_name": self.body_name,
                    "geom_id": self.geom_id,
                    "geom_name": self.geom_name,
                    "point_body_local_m": [0.0, 0.0, 0.0],
                    "source": "collision_geom_surface",
                }

        samples = [
            Sample(0, "robot0_link5"),
            Sample(1, "robot0_link6"),
            Sample(2, "gripper0_leftfinger"),
        ]
        residuals = self.np.asarray([-0.2, -0.1, -0.3], dtype=self.np.float64)
        consumer = _independent_constraint_attribution(
            residuals, [sample.to_dict() for sample in samples], np=self.np
        )
        self.assertEqual(
            consumer["schema_version"], "vlsa_poisson_constraint_attribution.v2"
        )
        self.assertEqual(
            consumer["sample_scope"],
            "all_structurally_movable_manipulator_collision_surfaces",
        )
        self.assertEqual(
            consumer["minimum_sample"]["body_name"], "gripper0_leftfinger"
        )
        self.assertEqual(
            consumer["negative_nominal_residual_body_names"],
            ["gripper0_leftfinger", "robot0_link5", "robot0_link6"],
        )


class OscArmLinkRegisteredContactScopeTests(unittest.TestCase):
    def fixture(self):
        from scripts.run_poisson_osc_arm_link_canary import _canonical_sha256

        scope = {
            "schema_version": "vlsa_poisson_registered_contact_scope.v1",
            "semantics": (
                "forbidden_union=(any_robot_geom_vs_selected_obstacle_geom)_or_"
                "(literal_link5_link6_geom_vs_any_external_nonrobot_geom)"
            ),
            "model_geom_count": 5,
            "robot_geom_ids": [0, 1],
            "robot_owned_geom_ids": [0, 1, 4],
            "selected_obstacle_geom_ids": [2],
            "link56_geom_ids": [1],
            "external_nonrobot_geom_ids": [2, 3],
            "geom_identities": [],
        }
        for geom_id in range(5):
            body_id = 1 if geom_id == 4 else geom_id
            scope["geom_identities"].append(
                {
                    "geom_id": geom_id,
                    "geom_name": "geom_%d" % geom_id,
                    "body_id": body_id,
                    "body_name": "body_%d" % body_id,
                    "is_robot_geom": geom_id in (0, 1),
                    "is_robot_owned_geom": geom_id in (0, 1, 4),
                    "is_selected_obstacle_geom": geom_id == 2,
                    "is_link56_geom": geom_id == 1,
                    "is_external_nonrobot_geom": geom_id in (2, 3),
                }
            )
        scope["identity_sha256"] = _canonical_sha256(scope)
        record = {
            "schema_version": "vlsa_poisson_registered_forbidden_contact.v1",
            "source_phase": "post_integration_recomputed",
            "contact_index": 0,
            "physical_boundary": 0,
            "executed_transition_start_boundary": 0,
            "observed_state_boundary": 1,
            "source_action_index": 0,
            "physics_substep_index": 0,
            "geom1_id": 1,
            "geom2_id": 3,
            "contact_distance_m": -1e-5,
            "robot_geom_id": 1,
            "robot_geom_name": "geom_1",
            "robot_body_id": 1,
            "robot_body_name": "body_1",
            "external_geom_id": 3,
            "external_geom_name": "geom_3",
            "external_body_id": 3,
            "external_body_name": "body_3",
            "contact_categories": ["link56_vs_external_nonrobot"],
            "any_robot_selected_obstacle_contact": False,
            "link56_external_nonrobot_contact": True,
            "link56_nonselected_external_contact": True,
        }
        record["record_sha256"] = _canonical_sha256(record)
        candidate = {
            "literal_contact": False,
            "literal_contact_count": 0,
            "contact_categories": [],
            "contacts": [],
        }
        physics = [
            {
                "physical_boundary": 0,
                "registered_contact_scope_checked": True,
                "registered_forbidden_contact_seen": True,
                "registered_contact_categories": [
                    "link56_vs_external_nonrobot"
                ],
                "registered_forbidden_contacts": [record],
                "candidate_exact_clone_contact": candidate,
            }
        ]
        measurement = {
            "schema_version": "vlsa_poisson_registered_contact_measurement.v1",
            "scope_identity_sha256": scope["identity_sha256"],
            "observed_physics_substeps": 1,
            "settled_forbidden_contact": False,
            "settled_contact_records": [],
            "rollout_forbidden_contact": True,
            "rollout_contact_categories": ["link56_vs_external_nonrobot"],
            "rollout_contact_records": [record],
            "any_registered_forbidden_contact": True,
            "any_robot_selected_obstacle_contact": False,
            "any_link56_external_nonrobot_contact": True,
            "any_link56_nonselected_external_contact": True,
        }
        resolved = {
            "robot_geom_ids": [0, 1],
            "robot_body_ids": [0, 1],
            "obstacle_geom_ids": [2],
            "link56_geom_ids": [1],
        }
        return scope, measurement, physics, resolved

    def test_shifted_link_contact_is_independently_reconstructed(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _validate_registered_contact_evidence,
        )

        scope, measurement, physics, resolved = self.fixture()
        audit = _validate_registered_contact_evidence(
            scope=scope,
            measurement=measurement,
            physics=physics,
            resolved_geometry=resolved,
        )
        self.assertTrue(audit["rollout_contact"])
        self.assertTrue(audit["any_link56_nonselected_external_contact"])
        self.assertFalse(audit["any_robot_selected_obstacle_contact"])

    def test_fabricated_shifted_category_is_rejected(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
            _validate_registered_contact_evidence,
        )

        scope, measurement, physics, resolved = self.fixture()
        physics[0]["registered_forbidden_contacts"][0][
            "contact_categories"
        ] = ["any_robot_vs_selected_obstacle"]
        with self.assertRaises(OscCanaryValidationError):
            _validate_registered_contact_evidence(
                scope=scope,
                measurement=measurement,
                physics=physics,
                resolved_geometry=resolved,
            )


class OscArmLinkIndependentQpMutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
            import scipy  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest(
                "allocation-only NumPy/SciPy are required for QP mutation tests"
            ) from error
        cls.np = np

    def certificate(self):
        np = self.np
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        controller = {
            "arm_actuator_indexes": list(range(7)),
            "arm_qpos_indexes": list(range(7)),
            "arm_qvel_indexes": list(range(7)),
        }
        nominal = np.zeros(7)
        velocity_dimension = 9
        robot_qvel_indices = list(range(velocity_dimension))
        nominal_next = np.zeros(velocity_dimension)
        nominal_next[0] = -1.0
        command = np.zeros(7)
        command[0] = 0.5
        stencil = []
        for column in range(7):
            stencil.append(
                {
                    "column_index": column,
                    "nominal_torque_nm": 0.0,
                    "lower_bound_nm": -10.0,
                    "upper_bound_nm": 10.0,
                    "requested_full_epsilon_nm": 0.001,
                    "available_negative_delta_nm": 10.0,
                    "available_positive_delta_nm": 10.0,
                    "bound_adapted": False,
                    "full_resolution": {
                        "stencil": "centered",
                        "sample_deltas_nm": [-0.001, 0.001],
                        "denominator_nm": 0.002,
                    },
                    "half_resolution": {
                        "stencil": "centered",
                        "sample_deltas_nm": [-0.0005, 0.0005],
                        "denominator_nm": 0.001,
                    },
                    "difference_formula": "(v_next(delta_1)-v_next(delta_0))/(delta_1-delta_0)",
                }
            )
        generic_sensitivity = np.vstack((np.eye(7), np.zeros((2, 7))))
        sensitivity = {
            "output_qvel_indices": robot_qvel_indices,
            "torque_to_next_output_qvel_sensitivity": generic_sensitivity.tolist(),
            "full_epsilon_sensitivity": generic_sensitivity.tolist(),
            "half_epsilon_sensitivity": generic_sensitivity.tolist(),
            "torque_epsilon_nm": [0.001] * 7,
            "maximum_epsilon_agreement_absolute_error": 0.0,
            "maximum_epsilon_agreement_scaled_error": 0.0,
            "agreement_atol": protocol["shield"]["sensitivity_agreement_atol"],
            "agreement_rtol": protocol["shield"]["sensitivity_agreement_rtol"],
            "nominal_next_output_qvel_rad_s": nominal_next.tolist(),
            "nominal_next_integration_state_sha256": "a" * 64,
            "nominal_next_time_seconds": 0.002,
            "non_arm_ctrl_preserved": True,
            "finite_difference_column_stencils": stencil,
            "snapshot": {
                "integration_state_sha256": "b" * 64,
                "all_ctrl_sha256": "c" * 64,
                "arm_actuator_ids": list(range(7)),
                "arm_qpos_indices": list(range(7)),
                "arm_qvel_indices": list(range(7)),
                "output_qvel_indices": robot_qvel_indices,
                "arm_torque_lower_nm": [-10.0] * 7,
                "arm_torque_upper_nm": [10.0] * 7,
                "nominal_arm_torque_nm": nominal.tolist(),
                "timestep_seconds": 0.002,
            },
        }
        row = {
            "sensitivity": sensitivity,
            "nominal_torque_nm": nominal.tolist(),
            "command_torque_nm": command.tolist(),
            "nominal_predicted_next_shield_qvel_rad_s": nominal_next.tolist(),
            "poisson_h_m2": [0.1],
            "joint_gradient_rows_m2_per_rad": [
                [1.0, 0, 0, 0, 0, 0, 0, 0, 0]
            ],
            "shield_diagnostics": {
                "schema": "vlsa_poisson_post_osc_torque_shield.v1",
                "constraint_equation": "a@(v_nom_next+S@delta_tau)+alpha*h>=margin",
                "sensitivity_already_includes_dt_and_contact_effects": True,
                "dt_seconds": 0.002,
                "alpha": protocol["shield"]["alpha_gain_per_s"],
                "margin": protocol["shield"]["margin_m2_per_s"],
                "sensitivity_max_absolute_error": 0.0,
                "nominal_torque_nm": nominal.tolist(),
                "torque_lower_nm": [-10.0] * 7,
                "torque_upper_nm": [10.0] * 7,
                "nominal_next_qvel_rad_s": nominal_next.tolist(),
                "velocity_dimension": velocity_dimension,
                "torque_dimension": 7,
            },
        }
        torque_actuators = [
            {"actuator_id": index, "control_range": [-10.0, 10.0]}
            for index in range(7)
        ]
        return (
            row,
            protocol,
            controller,
            torque_actuators,
            robot_qvel_indices,
            velocity_dimension,
        )

    def validate(
        self,
        row,
        protocol,
        controller,
        torque_actuators,
        robot_qvel_indices,
        velocity_dimension,
    ):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _validate_solved_qp_certificate,
        )

        return _validate_solved_qp_certificate(
            row,
            protocol=protocol,
            controller=controller,
            torque_actuators=torque_actuators,
            robot_qvel_indices=robot_qvel_indices,
            velocity_dimension=velocity_dimension,
            np=self.np,
        )

    def test_exact_minimum_norm_certificate_passes(self):
        values = self.certificate()
        audit = self.validate(*values)
        self.assertAlmostEqual(audit["correction_l2_nm"], 0.5)

    def test_feasible_but_nonminimum_command_is_rejected(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        values = self.certificate()
        row = values[0]
        row["command_torque_nm"][0] = 0.6
        with self.assertRaises(OscCanaryValidationError):
            self.validate(*values)

    def test_mutated_stencil_or_sensitivity_is_rejected(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        for mutation in ("stencil", "matrix"):
            with self.subTest(mutation=mutation):
                values = self.certificate()
                row = values[0]
                if mutation == "stencil":
                    row["sensitivity"]["finite_difference_column_stencils"][0][
                        "full_resolution"
                    ]["sample_deltas_nm"][1] = 11.0
                else:
                    row["sensitivity"]["half_epsilon_sensitivity"][0][0] = 2.0
                with self.assertRaises(OscCanaryValidationError):
                    self.validate(*values)


if __name__ == "__main__":
    unittest.main()
