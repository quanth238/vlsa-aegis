"""Focused tests for the fail-closed Poisson runtime protocol."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from main.poisson_fullbody.feasibility_protocol import (
    FeasibilityProtocolError,
    LEGACY_SCHEMA_VERSION,
    PARAMETER_SECTIONS,
    SCHEMA_VERSION,
    bind_parameter_block,
    load_feasibility_protocol,
    validate_feasibility_protocol,
)


PLACEHOLDER_HASH = "0" * 64


def valid_protocol():
    value = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "link56-static-poisson-canary-v1",
        "workspace": {
            "minimum_m": [-1.0, -1.0, 0.0],
            "maximum_m": [1.0, 1.0, 2.0],
            "grid_shape_vertices": [101, 101, 101],
            "grid_spacing_m": [0.02, 0.02, 0.02],
            "boundary_condition": "homogeneous_dirichlet_h_zero",
            "outside_workspace_policy": "invalid_fail_closed",
        },
        "occupancy": {
            "geometry_source": "selected_mujoco_collision_enabled_geoms",
            "rasterization": "closed_cell_obb_intersection",
            "obstacle_clearance_m": 0.051001,
            "outer_boundary_clearance_m": 0.051,
            "buffer_application": "exactly_once",
            "voxel_discretization_policy": "closed_cells_no_extra_half_diagonal",
            "field_motion_model": "static_selected_obstacle",
            "unknown_geometry_policy": "invalid_fail_closed",
        },
        "coverage": {
            "epsilon_m": 0.05,
            "ball_semantics": "strict_open_ball",
            "certificate_method": "analytic_triangle_lattice_covering_bound",
            "certificate_relation": (
                "max_covering_radius_strictly_less_than_epsilon"
            ),
            "require_every_collision_surface_component": True,
        },
        "safety": {
            "contact_margin_m": 0.001,
            "physical_contact_definition": "mujoco_contact_dist_le_zero",
            "positive_margin_contact_policy": (
                "diagnostic_candidate_not_physical_contact"
            ),
            "distance_accounting": "D_opt_and_D_sim_are_distinct",
        },
        "poisson": {
            "equation": "minus_laplacian_h_equals_forcing",
            "forcing_value": 1.0,
            "boundary_value": 0.0,
            "solver": "red_black_sor",
            "relaxation_omega": 1.7,
            "max_iterations": 50000,
            "normalized_backward_error_tolerance": 1.0e-8,
            "residual_check_interval": 20,
            "query_interpolation": "trilinear_discrete_C0",
            "invalid_query_policy": "terminate_retain_case",
            "zero_cell_policy": "nonregular_zero_cell_invalid_fail_closed",
        },
        "cbf": {
            "alpha_gain_per_s": 5.0,
            "constraint_form": (
                "grad_h_J_qdot_plus_alpha_h_ge_issf_epsilon0_grad_norm_squared"
            ),
            "time_derivative_policy": "zero_only_while_static_field_admissible",
            "issf_mode": "disabled_initial_pilot",
            "issf_epsilon0": 0.0,
            "unsafe_initial_policy": "terminate_retain_case",
        },
        "adapter": {
            "dls_damping": 0.05,
            "position_gain_per_s": 20.0,
            "orientation_gain_per_s": 20.0,
            "translation_action_scale_m": 0.05,
            "input_clip_abs": 1.0,
            "orientation_target": "hold_pose_at_high_level_action_start",
            "physical_joint_velocity_scale_rad_s": 0.5,
            "gripper_policy": "unchanged_vla_command",
        },
        "qp": {
            "solver": "osqp",
            "objective": "minimize_weighted_squared_qdot_deviation",
            "weight_diagonal": [1.0] * 7,
            "hard_cbf_constraints": True,
            "slack_enabled": False,
            "velocity_lower_rad_s": [-0.5] * 7,
            "velocity_upper_rad_s": [0.5] * 7,
            "joint_limit_alpha_per_s": 5.0,
            "joint_position_margin_rad": 0.01,
            "eps_abs": 1.0e-7,
            "eps_rel": 1.0e-7,
            "max_iterations": 10000,
            "postcheck_cbf_tolerance": 5.0e-7,
            "postcheck_bound_tolerance_rad_s": 5.0e-8,
            "row_scaling": "enabled",
            "failure_policy": "terminate_retain_case_no_cached_command",
        },
        "cadence": {
            "physics_timestep_s": 0.002,
            "high_level_frequency_hz": 20.0,
            "shadow": {
                "mode": "monitor_only_no_action_mutation",
                "filter_updates_per_high_level_action": 1,
                "physics_substeps_per_filter_update": 25,
            },
            "active": {
                "mode": "recompute_joint_velocity_each_filter_update",
                "filter_frequency_hz": 100.0,
                "filter_updates_per_high_level_action": 5,
                "physics_substeps_per_filter_update": 5,
            },
        },
        "admissibility": {
            "max_state_restore_qpos_error_rad": 1.0e-10,
            "max_state_restore_qvel_error_rad_s": 1.0e-10,
            "joint_position_margin_rad": 0.01,
            "max_joint_velocity_tracking_linf_rad_s": 0.05,
            "max_joint_velocity_tracking_rmse_rad_s": 0.02,
            "max_selected_body_linear_speed_m_s": 1.0e-4,
            "max_selected_body_angular_speed_rad_s": 1.0e-4,
            "max_selected_geom_translation_drift_m": 1.0e-6,
            "max_selected_geom_rotation_drift_rad": 1.0e-5,
            "max_selected_geom_surface_drift_m": 1.0e-6,
            "maximum_invalid_field_queries": 0,
            "require_safe_initial_samples": True,
            "static_field_refresh_policy": "never_refresh_terminate_on_drift",
            "violation_policy": "terminate_retain_invalid_not_collision_free",
        },
        "differential_audit": {
            "state_source": "settled_mujoco_mjstate_integration_clone",
            "perturbation_integrator": "mujoco_mj_integratePos_full_nv_tangent",
            "point_jacobian_delta_rad": 1.0e-6,
            "point_jacobian_absolute_tolerance_m_per_rad": 2.0e-6,
            "point_jacobian_relative_tolerance": 1.0e-4,
            "point_jacobian_near_zero_frobenius_m_per_rad": 1.0e-10,
            "arm_tangent_roundtrip_criterion": (
                "scalar_hinge_two_stage_exact_fraction_binary64_roundoff"
            ),
            "binary64_unit_roundoff": 2.0 ** -53,
            "expected_mujoco_version": "3.2.3",
            "expected_arm_dof_indices": list(range(7)),
            "expected_arm_joint_ids": list(range(7)),
            "expected_arm_qpos_indices": list(range(7)),
            "expected_arm_joint_names": [
                "robot0_joint%d" % index for index in range(1, 8)
            ],
            "required_arm_joint_type": "hinge",
            "legacy_arm_tangent_reconstruction_tolerance_rad_s": 1.0e-10,
            "nonarm_tangent_leakage_tolerance_rad_s": 1.0e-12,
            "joint_velocity_directions_rad_s": [
                [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
                [
                    1.0 / 14.0,
                    2.0 / 14.0,
                    3.0 / 14.0,
                    4.0 / 14.0,
                    5.0 / 14.0,
                    6.0 / 14.0,
                    7.0 / 14.0,
                ],
            ],
            "coupled_eta_ladder_s": [
                2.0e-6,
                5.0e-7,
                1.25e-7,
                3.125e-8,
                7.8125e-9,
                1.953125e-9,
                4.8828125e-10,
            ],
            "coupled_absolute_tolerance_m2_per_s": 2.0e-7,
            "coupled_relative_tolerance": 2.0e-4,
            "coupled_near_zero_m2_per_s": 1.0e-10,
            "same_trilinear_cell_required": True,
            "required_direction_count_per_sample": 9,
            "stencil_selection_policy": (
                "largest_eta_with_valid_base_plus_minus_in_same_exact_cell"
            ),
            "finite_difference_resolution_policy": (
                "fail_on_no_certified_stencil_or_detected_cancellation"
            ),
            "failure_policy": "fail_before_active_physics_retain_artifact",
        },
        "claim_scope": {
            "obstacle_scope": "one_selected_obstacle_collision_geometry_only",
            "protected_robot_bodies": ["robot0_link5", "robot0_link6"],
            "protected_robot_bodies_role": (
                "field_bundle_sample_seed_only_not_shield_constraint_scope"
            ),
            "shield_robot_collision_surface_scope": (
                "all_collision_enabled_geoms_in_authoritative_robot_body_tree"
            ),
            "shield_robot_qvel_scope": (
                "all_qvel_dofs_in_authoritative_robot_body_tree_affecting_shield_samples"
            ),
            "shield_decision_scope": (
                "seven_registered_panda_arm_torque_controls_only"
            ),
            "nonarm_control_policy": (
                "nominal_nonarm_controls_unchanged_with_motion_included_in_exact_affine_dynamics"
            ),
            "unprotected_body_policy": (
                "no_authoritative_robot_collision_surface_excluded_against_selected_obstacle"
            ),
            "population_scope": "outcome_conditioned_109_case_targeted_feasibility",
            "claim_strength": (
                "empirical_discrete_C0_static_field_feasibility_not_formal_guarantee"
            ),
            "prohibited_generalization": (
                "not_all_obstacles_not_unbiased_safelibero"
            ),
        },
        "parameter_selection": {
            "selection_order": [
                "bringup_canary",
                "parameter_freeze",
                "heldout_evaluation",
            ],
            "allowed_tuning_splits": ["bringup_canary", "parameter_freeze"],
            "heldout_split": "heldout_evaluation",
            "blocking_unit": "whole_semantic_task_family",
            "freeze_before_heldout": True,
            "heldout_parameter_changes": "forbidden",
            "after_change_policy": "new_protocol_and_new_heldout_required",
            "parameter_sections": list(PARAMETER_SECTIONS),
            "parameter_block_sha256": PLACEHOLDER_HASH,
        },
    }
    return bind_parameter_block(value)


class FeasibilityProtocolTest(unittest.TestCase):
    def assert_invalid(self, mutation, message_fragment=None):
        value = valid_protocol()
        mutation(value)
        with self.assertRaises(FeasibilityProtocolError) as raised:
            validate_feasibility_protocol(value)
        if message_fragment is not None:
            self.assertIn(message_fragment, str(raised.exception))

    def test_valid_protocol_has_stable_distinct_hashes(self):
        value = valid_protocol()
        first = validate_feasibility_protocol(value)
        second = validate_feasibility_protocol(copy.deepcopy(value))
        self.assertEqual(first, second)
        self.assertRegex(first.protocol_sha256, r"^[0-9a-f]{64}$")
        self.assertRegex(first.parameter_block_sha256, r"^[0-9a-f]{64}$")
        self.assertNotEqual(first.protocol_sha256, first.parameter_block_sha256)
        self.assertEqual(
            first.parameter_block_sha256,
            value["parameter_selection"]["parameter_block_sha256"],
        )

    def test_unknown_and_missing_fields_fail_closed(self):
        self.assert_invalid(
            lambda value: value["poisson"].update({"magic": 1}), "unknown fields"
        )
        self.assert_invalid(
            lambda value: value["qp"].pop("eps_abs"), "missing fields"
        )
        self.assert_invalid(
            lambda value: value["qp"].update({1: "not-a-json-key"}),
            "keys must be strings",
        )

    def test_nonfinite_boolean_and_bad_ranges_are_rejected(self):
        self.assert_invalid(
            lambda value: value["poisson"].update({"relaxation_omega": float("nan")}),
            "finite number",
        )
        self.assert_invalid(
            lambda value: value["poisson"].update({"max_iterations": True}),
            "integer",
        )
        self.assert_invalid(
            lambda value: value["poisson"].update({"relaxation_omega": 2.0}),
            "must be <",
        )
        self.assert_invalid(
            lambda value: value["cbf"].update({"alpha_gain_per_s": 0.0}),
            "must be >",
        )

    def test_grid_dimensions_and_spacing_must_agree(self):
        self.assert_invalid(
            lambda value: value["workspace"].update(
                {"grid_spacing_m": [0.021, 0.02, 0.02]}
            ),
            "disagree",
        )

    def test_strict_open_coverage_contract_is_mandatory(self):
        self.assert_invalid(
            lambda value: value["coverage"].update({"ball_semantics": "closed_ball"}),
            "strict_open_ball",
        )
        self.assert_invalid(
            lambda value: value["coverage"].update(
                {"require_every_collision_surface_component": False}
            ),
            "every protected",
        )

    def test_clearance_covers_epsilon_margin_and_static_drift(self):
        self.assert_invalid(
            lambda value: value["occupancy"].update(
                {"obstacle_clearance_m": 0.0510009}
            ),
            "obstacle clearance",
        )
        self.assert_invalid(
            lambda value: value["occupancy"].update(
                {"outer_boundary_clearance_m": 0.0509}
            ),
            "outer-boundary clearance",
        )

    def test_settled_obstacle_speed_thresholds_are_finite_nonnegative(self):
        for field in (
            "max_selected_body_linear_speed_m_s",
            "max_selected_body_angular_speed_rad_s",
        ):
            self.assert_invalid(
                lambda value, field=field: value["admissibility"].update(
                    {field: -1.0e-6}
                ),
                "must be >=",
            )
            self.assert_invalid(
                lambda value, field=field: value["admissibility"].update(
                    {field: float("inf")}
                ),
                "finite number",
            )

    def test_poisson_sign_boundary_and_convergence_contract(self):
        self.assert_invalid(
            lambda value: value["poisson"].update({"forcing_value": -1.0}),
            "strictly positive",
        )
        self.assert_invalid(
            lambda value: value["poisson"].update({"boundary_value": 1.0e-9}),
            "exactly zero",
        )
        self.assert_invalid(
            lambda value: value["poisson"].update(
                {"residual_check_interval": 50001}
            ),
            "may not exceed",
        )

    def test_initial_protocol_cannot_silently_enable_issf(self):
        self.assert_invalid(
            lambda value: value["cbf"].update({"issf_epsilon0": 0.01}),
            "no validated ISSf runtime",
        )

    def test_adapter_preserves_released_action_units_and_orientation_hold(self):
        self.assert_invalid(
            lambda value: value["adapter"].update(
                {"translation_action_scale_m": 0.01}
            ),
            "released OSC scale",
        )
        self.assert_invalid(
            lambda value: value["adapter"].update(
                {"orientation_target": "integrate_policy_rotation"}
            ),
            "hold_pose_at_high_level_action_start",
        )

    def test_qp_is_hard_bounded_and_postchecked(self):
        self.assert_invalid(
            lambda value: value["qp"].update({"slack_enabled": True}),
            "does not permit safety slack",
        )
        self.assert_invalid(
            lambda value: value["qp"].update(
                {"velocity_upper_rad_s": [0.6] + [0.5] * 6}
            ),
            "exceed",
        )
        self.assert_invalid(
            lambda value: value["qp"].update(
                {"velocity_lower_rad_s": [0.1] + [-0.5] * 6}
            ),
            "contain zero",
        )

    def test_shadow_and_active_cadences_are_exact(self):
        self.assert_invalid(
            lambda value: value["cadence"]["shadow"].update(
                {"physics_substeps_per_filter_update": 5}
            ),
            "1 x 25",
        )
        self.assert_invalid(
            lambda value: value["cadence"]["active"].update(
                {"filter_updates_per_high_level_action": 1}
            ),
            "5 x 5",
        )

    def test_admissibility_thresholds_are_consistent(self):
        self.assert_invalid(
            lambda value: value["admissibility"].update(
                {"joint_position_margin_rad": 0.02}
            ),
            "margins must match",
        )
        self.assert_invalid(
            lambda value: value["admissibility"].update(
                {"max_joint_velocity_tracking_rmse_rad_s": 0.06}
            ),
            "RMSE threshold",
        )
        self.assert_invalid(
            lambda value: value["admissibility"].update(
                {"maximum_invalid_field_queries": 1}
            ),
            "zero invalid",
        )
        self.assert_invalid(
            lambda value: value["admissibility"].update(
                {"max_state_restore_qpos_error_rad": 0.02}
            ),
            "qpos restore tolerance",
        )

    def test_claim_scope_freezes_full_robot_surfaces_and_seven_torque_decision(self):
        self.assert_invalid(
            lambda value: value["claim_scope"].update(
                {"protected_robot_bodies": ["robot0_link1", "robot0_link6"]}
            ),
            "robot0_link5",
        )
        self.assert_invalid(
            lambda value: value["claim_scope"].update(
                {"obstacle_scope": "all_scene_obstacles"}
            ),
            "one_selected_obstacle",
        )
        for field, replacement in (
            ("protected_robot_bodies_role", "shield_rows"),
            ("shield_robot_collision_surface_scope", "link56_only"),
            ("shield_robot_qvel_scope", "seven_arm_qvel_only"),
            ("shield_decision_scope", "all_robot_controls"),
            ("nonarm_control_policy", "zero_nonarm_controls"),
            ("unprotected_body_policy", "exclude_gripper"),
        ):
            self.assert_invalid(
                lambda value, field=field, replacement=replacement: value[
                    "claim_scope"
                ].update({field: replacement}),
                field,
            )

    def test_immutable_v3_runtime_remains_loadable(self):
        legacy_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "vlsa_poisson_runtime_protocol.canary.v3.json"
        )
        legacy, hashes = load_feasibility_protocol(legacy_path)
        self.assertEqual(legacy["schema_version"], LEGACY_SCHEMA_VERSION)
        self.assertEqual(
            hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
            "396de850fb7f03b9ee038b2fec05f90ef1cb0230c5d40a81a43177162193726a",
        )
        self.assertEqual(
            hashes.protocol_sha256,
            "2125989269a2ffeeb8d3408d56e4aaa74e1dc5816256d8210bf9ee686c5f5a30",
        )
        self.assertEqual(
            hashes.parameter_block_sha256,
            "11de833ab8b1586948e7eb039c449a66a0c1a4107fa3232b236ad1ecb9ceeabe",
        )

    def test_parameter_selection_forbids_heldout_tuning(self):
        self.assert_invalid(
            lambda value: value["parameter_selection"].update(
                {"allowed_tuning_splits": ["heldout_evaluation"]}
            ),
            "bringup_canary",
        )
        self.assert_invalid(
            lambda value: value["parameter_selection"].update(
                {"freeze_before_heldout": False}
            ),
            "freeze",
        )

    def test_differential_audit_is_exhaustive_and_hash_bound(self):
        self.assert_invalid(
            lambda value: value["differential_audit"].update(
                {"required_direction_count_per_sample": 1}
            ),
            "every protected sample",
        )
        self.assert_invalid(
            lambda value: value["differential_audit"].update(
                {"same_trilinear_cell_required": False}
            ),
            "within one trilinear cell",
        )
        self.assert_invalid(
            lambda value: value["differential_audit"][
                "joint_velocity_directions_rad_s"
            ][7].__setitem__(0, 0.4),
            "directions differ",
        )
        self.assert_invalid(
            lambda value: value["differential_audit"][
                "coupled_eta_ladder_s"
            ].__setitem__(1, 4.0e-7),
            "eta ladder differs",
        )
        self.assert_invalid(
            lambda value: value["differential_audit"].update(
                {"legacy_arm_tangent_reconstruction_tolerance_rad_s": 0.0}
            ),
            "must be > 0.0",
        )
        self.assert_invalid(
            lambda value: value["differential_audit"].update(
                {"legacy_arm_tangent_reconstruction_tolerance_rad_s": 2.0e-10}
            ),
            "must remain 1e-10",
        )
        self.assert_invalid(
            lambda value: value["differential_audit"][
                "expected_arm_dof_indices"
            ].__setitem__(0, False),
            "registered Panda topology",
        )
        self.assert_invalid(
            lambda value: value["differential_audit"].update(
                {"arm_tangent_roundtrip_criterion": "empirical_tolerance"}
            ),
            "arm_tangent_roundtrip_criterion",
        )
        self.assert_invalid(
            lambda value: value["differential_audit"].update(
                {"stencil_selection_policy": "first_valid_eta"}
            ),
            "stencil_selection_policy",
        )
        self.assert_invalid(
            lambda value: value["differential_audit"].update(
                {"finite_difference_resolution_policy": "ignore_cancellation"}
            ),
            "finite_difference_resolution_policy",
        )

    def test_parameter_change_invalidates_declared_freeze_hash(self):
        value = valid_protocol()
        value["qp"]["eps_abs"] = 2.0e-7
        with self.assertRaisesRegex(FeasibilityProtocolError, "does not match"):
            validate_feasibility_protocol(value)
        rebound = bind_parameter_block(value)
        self.assertNotEqual(
            rebound["parameter_selection"]["parameter_block_sha256"],
            valid_protocol()["parameter_selection"]["parameter_block_sha256"],
        )
        validate_feasibility_protocol(rebound)

    def test_expected_full_protocol_hash_is_enforced(self):
        value = valid_protocol()
        identity = validate_feasibility_protocol(value)
        validate_feasibility_protocol(
            value, expected_protocol_sha256=identity.protocol_sha256
        )
        with self.assertRaisesRegex(FeasibilityProtocolError, "expected identity"):
            validate_feasibility_protocol(
                value, expected_protocol_sha256="f" * 64
            )

    def test_loader_rejects_duplicates_nonfinite_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid_path = root / "protocol.json"
            value = valid_protocol()
            valid_path.write_text(json.dumps(value), encoding="utf-8")
            loaded, hashes = load_feasibility_protocol(valid_path)
            self.assertEqual(loaded, value)
            self.assertEqual(
                hashes, validate_feasibility_protocol(value)
            )

            duplicate = root / "duplicate.json"
            duplicate.write_text('{"schema_version":"a","schema_version":"b"}')
            with self.assertRaisesRegex(FeasibilityProtocolError, "duplicate"):
                load_feasibility_protocol(duplicate)

            nonfinite = root / "nonfinite.json"
            nonfinite.write_text('{"value": NaN}')
            with self.assertRaisesRegex(FeasibilityProtocolError, "non-finite"):
                load_feasibility_protocol(nonfinite)

            symlink = root / "linked.json"
            symlink.symlink_to(valid_path)
            with self.assertRaisesRegex(FeasibilityProtocolError, "non-symlink"):
                load_feasibility_protocol(symlink)


if __name__ == "__main__":
    unittest.main()
