import ast
import json
from pathlib import Path
import tempfile
import unittest


try:
    import numpy as np
    import robosuite  # noqa: F401
except ImportError:  # pragma: no cover - allocation dependency
    np = None


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_poisson_osc_reference_governor.py"
PROTOCOL = ROOT / "configs" / "vlsa_poisson_osc_reference_governor.v2.json"


class OscReferenceGovernorRunnerTests(unittest.TestCase):
    def test_protocol_uses_one_shared_link_set_and_dynamic_rigid_field(self):
        value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual(
            value["allowed_case_ids"],
            ["vlsa-t1-goal-ii-t0-e05", "vlsa-t1-goal-ii-t3-e42"],
        )
        self.assertEqual(
            value["protected_robot_body_names"],
            ["robot0_link5", "robot0_link6"],
        )
        self.assertFalse(value["task_conditioned_link_selection"])
        self.assertEqual(
            value["field"]["frame"],
            "selected_obstacle_rigid_body_attached",
        )
        self.assertFalse(value["field"]["rebuild_or_inflate_field"])
        self.assertEqual(
            value["controller"]["zero_correction_behavior"],
            "retain_native_global_target_and_source_action_bytes",
        )
        self.assertEqual(
            value["controller"]["joint_velocity_predictor"],
            "measured_qdot_plus_DLS_response_to_OSC_target_change",
        )
        self.assertEqual(value["controller"]["safety_update_frequency_hz"], 100)
        self.assertEqual(
            value["controller"]["safety_update_physics_substeps"],
            [0, 5, 10, 15, 20],
        )
        self.assertEqual(value["controller"]["response_horizon_seconds"], 0.05)
        self.assertEqual(
            value["controller"]["native_pose_input_bounds"],
            [-1.0, 1.0],
        )
        self.assertEqual(
            value["controller"]["distant_installed_target_behavior"],
            "hold_exactly_without_clipping",
        )
        self.assertEqual(
            value["controller"]["policy_divergence_trigger"],
            "first_applied_reference_behavior_different_from_native_interval",
        )
        self.assertEqual(
            value["policy"]["before_first_byte_divergence"],
            "archived_successful_AEGIS_actions",
        )
        self.assertNotIn("before_first_material_correction", value["policy"])
        self.assertEqual(
            value["acceptance"]["minimum_post_correction_eef_path_m"],
            0.01,
        )
        self.assertEqual(
            value["acceptance"][
                "maximum_post_correction_zero_pose_reference_interval_fraction"
            ],
            0.75,
        )
        self.assertEqual(
            value["acceptance"]["historical_precontact_timing_rule"],
            "first_material_physical_boundary_strictly_before_start_of_first_contact_sampled_action",
        )

    def test_runner_keeps_native_osc_and_does_not_import_adapter(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("step_with_osc_reference_updates", source)
        self.assertIn("query_rigid_obstacle_field_batch", source)
        self.assertIn("dynamic_cbf_rows", source)
        self.assertIn("OscPoseReferenceGovernor", source)
        self.assertNotIn("joint_velocity_adapter", source)
        self.assertNotIn("JOINT_VELOCITY", source)
        self.assertNotIn("step_grouped_actions_with_substep_callback", source)
        self.assertNotIn("step_with_arm_control_intervention", source)
        self.assertIn("filter_reference_target", source)
        self.assertIn("first_byte_divergence_physical_boundary", source)
        self.assertIn("invalid_queries", source)
        self.assertIn("predivergence_parity_trace", source)
        self.assertIn("_check_step(", source)
        self.assertIn("behavior_differs_from_native", source)
        self.assertIn("reference_update_contract", source)
        self.assertIn(
            "first_material_physical_boundary\n"
            "                < conservative_precontact_physical_boundary",
            source,
        )
        self.assertIn(
            "pose_reference_interval_count_after_correction",
            source,
        )

    def test_runner_has_no_task_or_case_conditioned_link_branch(self):
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        self.assertNotIn("target_link", names)
        source = RUNNER.read_text(encoding="utf-8")
        self.assertEqual(source.count('"robot0_link5"'), 1)
        self.assertEqual(source.count('"robot0_link6"'), 1)

    def test_protocol_loader_rejects_task_conditioned_selection(self):
        from scripts.run_poisson_osc_reference_governor import (
            ReferenceGovernorRunnerError,
            _load_protocol,
        )

        value = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        value["task_conditioned_link_selection"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ReferenceGovernorRunnerError):
                _load_protocol(path, "vlsa-t1-goal-ii-t0-e05")

    def test_only_full_task_useful_correction_is_positive(self):
        from scripts.run_poisson_osc_reference_governor import _classification

        base = {
            "registered_contact_seen": False,
            "method_failure_seen": False,
            "material_correction_present": True,
            "material_correction_before_historical_contact": True,
            "paper_car_avoided": True,
            "useful_post_correction_motion": True,
            "native_task_success_after_correction": True,
        }
        self.assertEqual(
            _classification(base),
            ("SAFE_TASK_SUCCESS_USEFUL_CORRECTION", True),
        )
        for field in (
            "material_correction_present",
            "material_correction_before_historical_contact",
            "paper_car_avoided",
            "useful_post_correction_motion",
            "native_task_success_after_correction",
        ):
            negative = dict(base)
            negative[field] = False
            self.assertFalse(_classification(negative)[1], field)
        for field in ("registered_contact_seen", "method_failure_seen"):
            negative = dict(base)
            negative[field] = True
            self.assertFalse(_classification(negative)[1], field)


@unittest.skipIf(np is None, "NumPy and Robosuite are allocation dependencies")
class OscNativeTargetRoundTripTests(unittest.TestCase):
    class Controller:
        def __init__(self):
            self.ee_pos = np.asarray([0.1, -0.2, 0.7], dtype=np.float64)
            self.ee_ori_mat = np.eye(3, dtype=np.float64)
            self.goal_pos = self.ee_pos.copy()
            self.goal_ori = np.asarray(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            self.position_limits = None
            self.orientation_limits = None
            self.output_scale = np.asarray(
                [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                dtype=np.float64,
            )

        def update(self):
            return None

        def scale_action(self, action):
            return np.clip(np.asarray(action), -1.0, 1.0) * self.output_scale

    def test_nonzero_native_target_round_trips_to_relative_action(self):
        from scripts.run_poisson_osc_reference_governor import (
            _nominal_global_osc_target,
            _osc_target_as_action,
        )

        controller = self.Controller()
        source = np.asarray([0.2, -0.4, 0.1, 0.08, -0.06, 0.04, -1.003])
        target_position, target_orientation = _nominal_global_osc_target(
            controller,
            source,
        )
        encoded, diagnostics = _osc_target_as_action(
            controller,
            target_position,
            target_orientation,
            controller.output_scale,
        )
        np.testing.assert_allclose(encoded, source[:6], rtol=0.0, atol=2.0e-6)
        self.assertFalse(
            diagnostics["target_outside_one_native_action_range"]
        )

    def test_zero_rotation_retains_existing_native_orientation_target(self):
        from scripts.run_poisson_osc_reference_governor import (
            _nominal_global_osc_target,
        )

        controller = self.Controller()
        _, target_orientation = _nominal_global_osc_target(
            controller,
            np.asarray([0.1, 0.0, -0.2, 0.0, 0.0, 0.0, 1.0]),
        )
        np.testing.assert_array_equal(target_orientation, controller.goal_ori)

    def test_distant_installed_target_coordinate_is_not_clipped(self):
        from scripts.run_poisson_osc_reference_governor import (
            _osc_target_as_action,
        )

        controller = self.Controller()
        target_position = controller.ee_pos + np.asarray([0.06, 0.0, 0.0])
        encoded, diagnostics = _osc_target_as_action(
            controller,
            target_position,
            controller.ee_ori_mat,
            controller.output_scale,
        )
        self.assertAlmostEqual(encoded[0], 1.2)
        self.assertTrue(
            diagnostics["target_outside_one_native_action_range"]
        )


if __name__ == "__main__":
    unittest.main()
