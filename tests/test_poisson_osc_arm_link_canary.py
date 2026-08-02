import copy
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "vlsa_poisson_osc_arm_link_canary.v1.json"


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

        derived = validate_osc_arm_link_canary_protocol(self.protocol())
        self.assertEqual(derived["historical_contact_action"], 62)
        self.assertEqual(derived["maximum_action_count"], 300)
        self.assertEqual(derived["expected_substeps_per_action"], 25)

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

    def test_protocol_rejects_paper_car_phase_or_source_drift(self):
        from main.poisson_fullbody.osc_arm_link_canary import (
            OscArmLinkCanaryError,
            validate_osc_arm_link_canary_protocol,
        )

        for field, value in (
            ("position_source", "post_integration_forwarded_body_xpos"),
            ("root_body_binding", "name_only"),
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
            live_solver_phase_body_position=observed.copy(),
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
            settled_car_position=self.np.asarray(
                [1.0, 2.0, 3.0], dtype=self.np.float64
            ),
            np=self.np,
        )

    def test_forwarded_cross_phase_offset_is_diagnostic_not_car(self):
        row = self.row()
        self.assertAlmostEqual(row["l1_displacement_from_settled_m"], 0.001)
        self.assertAlmostEqual(
            row["observation_post_integration_forwarded_l1_delta_m"], 0.006
        )
        self.assertAlmostEqual(self.validate(row), 0.001)

    def test_same_phase_live_observable_mismatch_is_rejected(self):
        from scripts.run_poisson_osc_arm_link_canary import (
            OscCanaryRunnerError,
            _paper_car_endpoint_row,
            _require_paper_car_same_phase_binding,
        )

        row = _paper_car_endpoint_row(
            source_action_index=0,
            snapshot_kind="completed_high_level_endpoint",
            observation_key="wine_bottle_obstacle_1_pos",
            obstacle_root_body_id=32,
            observation_position=[1.0, 2.0, 3.0],
            live_solver_phase_body_position=[1.0, 2.0, 3.000001],
            post_integration_forwarded_body_position=[1.0, 2.0, 3.0],
            settled_observation_position=[1.0, 2.0, 3.0],
            np=self.np,
        )
        self.assertFalse(row["observation_body_xpos_bitwise_equal"])
        with self.assertRaises(OscCanaryRunnerError):
            _require_paper_car_same_phase_binding(row)

    def test_signed_zero_is_not_bitwise_same_phase(self):
        from scripts.run_poisson_osc_arm_link_canary import (
            OscCanaryRunnerError,
            _paper_car_endpoint_row,
            _require_paper_car_same_phase_binding,
        )

        row = _paper_car_endpoint_row(
            source_action_index=-1,
            snapshot_kind="settled_pre_action",
            observation_key="wine_bottle_obstacle_1_pos",
            obstacle_root_body_id=32,
            observation_position=[0.0, 2.0, 3.0],
            live_solver_phase_body_position=[-0.0, 2.0, 3.0],
            post_integration_forwarded_body_position=[0.0, 2.0, 3.0],
            settled_observation_position=[0.0, 2.0, 3.0],
            np=self.np,
        )
        self.assertFalse(row["observation_body_xpos_bitwise_equal"])
        self.assertNotEqual(
            row["active_obstacle_position_observation_array_sha256"],
            row["active_obstacle_root_position_array_sha256"],
        )
        with self.assertRaises(OscCanaryRunnerError):
            _require_paper_car_same_phase_binding(row)

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
            live_solver_phase_body_position=[0.0, 2.0, 3.0],
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
        scalar["observation_live_solver_phase_l1_delta_m"] = -0.0
        with self.assertRaises(OscCanaryValidationError):
            self.validate(scalar)

        component = _paper_car_endpoint_row(
            source_action_index=0,
            snapshot_kind="completed_high_level_endpoint",
            observation_key="wine_bottle_obstacle_1_pos",
            obstacle_root_body_id=32,
            observation_position=[1.0, 2.0, 3.0],
            live_solver_phase_body_position=[1.0, 2.0, 3.0],
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
                settled_car_position=self.np.asarray(
                    [1.0, 2.0, 3.0], dtype=self.np.float64
                ),
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
                settled_car_position=self.np.asarray(
                    [1.0, 2.0, 3.0], dtype=self.np.float64
                ),
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
            _compact_float64_array_record,
        )

        np = self.np
        count = 3
        values = {
            "poisson_h_m2": np.asarray([0.1, 0.2, 0.3]),
            "joint_gradient_rows_m2_per_rad": np.arange(21).reshape(3, 7),
            "actual_hdot_m2_per_s": np.asarray([-0.3, 0.0, 0.2]),
            "candidate_exact_clone_hdot_m2_per_s": np.asarray([-0.2, 0.1, 0.3]),
            "nominal_exact_clone_hdot_m2_per_s": np.asarray([-0.4, 0.0, 0.1]),
        }
        return {
            "constraint_trace": {
                "schema_version": "vlsa_poisson_compact_constraint_trace.v1",
                "array_hash_format": (
                    "sha256_vlsa-table1-array-v1_header_and_c_order_float64_bytes"
                ),
                "sample_count": count,
                "arrays": {
                    name: _compact_float64_array_record(value, np)
                    for name, value in values.items()
                },
            }
        }

    def test_compact_trace_accepts_hashes_shapes_and_minima(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _validate_compact_constraint_trace,
        )

        arrays = _validate_compact_constraint_trace(
            self.trace_row(), expected_sample_count=3, np=self.np
        )
        self.assertEqual(arrays["joint_gradient_rows_m2_per_rad"]["shape"], [3, 7])

    def test_compact_trace_rejects_hash_tampering(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
            _validate_compact_constraint_trace,
        )

        row = self.trace_row()
        row["constraint_trace"]["arrays"]["poisson_h_m2"]["sha256"] = "bad"
        with self.assertRaises(OscCanaryValidationError):
            _validate_compact_constraint_trace(
                row, expected_sample_count=3, np=self.np
            )

    def test_first_divergence_constraint_body_is_independently_attributed(self):
        from scripts.run_poisson_osc_arm_link_canary import (
            _minimum_constraint_attribution,
        )
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

        samples = [Sample(0, "robot0_link5"), Sample(1, "robot0_link6")]
        residuals = self.np.asarray([-0.2, -0.1], dtype=self.np.float64)
        producer = _minimum_constraint_attribution(residuals, samples, self.np)
        consumer = _independent_constraint_attribution(
            residuals, [sample.to_dict() for sample in samples], np=self.np
        )
        self.assertEqual(producer, consumer)
        self.assertEqual(
            producer["minimum_sample"]["body_name"], "robot0_link5"
        )
        self.assertEqual(
            producer["negative_nominal_residual_body_names"],
            ["robot0_link5", "robot0_link6"],
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
        nominal_next = np.zeros(7)
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
        sensitivity = {
            "torque_to_next_arm_qvel_sensitivity": np.eye(7).tolist(),
            "full_epsilon_sensitivity": np.eye(7).tolist(),
            "half_epsilon_sensitivity": np.eye(7).tolist(),
            "torque_epsilon_nm": [0.001] * 7,
            "maximum_epsilon_agreement_absolute_error": 0.0,
            "maximum_epsilon_agreement_scaled_error": 0.0,
            "agreement_atol": protocol["shield"]["sensitivity_agreement_atol"],
            "agreement_rtol": protocol["shield"]["sensitivity_agreement_rtol"],
            "nominal_next_arm_qvel_rad_s": nominal_next.tolist(),
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
            "nominal_predicted_next_qvel_rad_s": nominal_next.tolist(),
            "poisson_h_m2": [0.1],
            "joint_gradient_rows_m2_per_rad": [[1.0, 0, 0, 0, 0, 0, 0]],
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
            },
        }
        torque_actuators = [
            {"actuator_id": index, "control_range": [-10.0, 10.0]}
            for index in range(7)
        ]
        return row, protocol, controller, torque_actuators

    def validate(self, row, protocol, controller, torque_actuators):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            _validate_solved_qp_certificate,
        )

        return _validate_solved_qp_certificate(
            row,
            protocol=protocol,
            controller=controller,
            torque_actuators=torque_actuators,
            np=self.np,
        )

    def test_exact_minimum_norm_certificate_passes(self):
        row, protocol, controller, torque_actuators = self.certificate()
        audit = self.validate(row, protocol, controller, torque_actuators)
        self.assertAlmostEqual(audit["correction_l2_nm"], 0.5)

    def test_feasible_but_nonminimum_command_is_rejected(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        row, protocol, controller, torque_actuators = self.certificate()
        row["command_torque_nm"][0] = 0.6
        with self.assertRaises(OscCanaryValidationError):
            self.validate(row, protocol, controller, torque_actuators)

    def test_mutated_stencil_or_sensitivity_is_rejected(self):
        from scripts.validate_poisson_osc_arm_link_canary_artifact import (
            OscCanaryValidationError,
        )

        for mutation in ("stencil", "matrix"):
            with self.subTest(mutation=mutation):
                row, protocol, controller, torque_actuators = self.certificate()
                if mutation == "stencil":
                    row["sensitivity"]["finite_difference_column_stencils"][0][
                        "full_resolution"
                    ]["sample_deltas_nm"][1] = 11.0
                else:
                    row["sensitivity"]["half_epsilon_sensitivity"][0][0] = 2.0
                with self.assertRaises(OscCanaryValidationError):
                    self.validate(row, protocol, controller, torque_actuators)


if __name__ == "__main__":
    unittest.main()
