import ast
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_poisson_osc_reference_governor.py"
PROTOCOL = ROOT / "configs" / "vlsa_poisson_osc_reference_governor.v1.json"


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
            "source_action_byte_exact_passthrough",
        )
        self.assertEqual(
            value["controller"]["joint_velocity_predictor"],
            "measured_qdot_plus_DLS_of_commanded_minus_measured_EEF_twist",
        )
        self.assertEqual(
            value["controller"]["native_pose_input_bounds"],
            [-1.0, 1.0],
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
                "maximum_post_correction_zero_pose_action_fraction"
            ],
            0.75,
        )

    def test_runner_keeps_native_osc_and_does_not_import_adapter(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("step_with_action_reference_intervention", source)
        self.assertIn("query_rigid_obstacle_field_batch", source)
        self.assertIn("dynamic_cbf_rows", source)
        self.assertIn("OscPoseReferenceGovernor", source)
        self.assertNotIn("joint_velocity_adapter", source)
        self.assertNotIn("JOINT_VELOCITY", source)
        self.assertNotIn("step_grouped_actions_with_substep_callback", source)
        self.assertNotIn("step_with_arm_control_intervention", source)
        self.assertIn("if not byte_identical and first_byte_divergence_action is None", source)
        self.assertIn("predivergence_parity_trace", source)
        self.assertIn("_check_step(", source)

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


if __name__ == "__main__":
    unittest.main()
