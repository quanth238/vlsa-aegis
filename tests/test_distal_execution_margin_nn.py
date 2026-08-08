from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class DistalExecutionMarginNnTests(unittest.TestCase):
    def test_config_freezes_state_groups_physics_residual_and_projection(self):
        from main.multilink_ellipsoid.execution_margin_nn import (
            EXECUTION_MARGIN_NN_SCHEMA,
            FEATURE_NAMES,
            load_execution_margin_config,
        )

        config = load_execution_margin_config(
            ROOT / "configs/vlsa_distal_execution_margin_nn_e05.v2.json"
        )
        self.assertEqual(config["schema_version"], EXECUTION_MARGIN_NN_SCHEMA)
        self.assertEqual(config["state_groups"]["collect_steps"], list(range(180, 193)))
        self.assertEqual(config["state_groups"]["primary_projection_step"], 188)
        self.assertNotIn(188, config["state_groups"]["train_steps"])
        self.assertNotIn(188, config["state_groups"]["validation_steps"])
        self.assertEqual(
            sum(config["candidate_set"]["expected_candidate_count_by_state"].values()),
            1125,
        )
        self.assertEqual(len(FEATURE_NAMES), 33)
        self.assertEqual(config["network"]["output_count"], 7)
        self.assertEqual(config["projection"]["activation_warning_m"], 0.008)
        self.assertEqual(config["projection"]["clearance_target_m"], 0.0)
        self.assertTrue(config["labels"]["distinct_D_opt_and_D_sim"])

    @unittest.skipUnless(HAS_NUMPY, "NumPy is allocation dependency")
    def test_feature_vector_has_registered_order_and_dimension(self):
        from main.multilink_ellipsoid.execution_margin_nn import (
            FEATURE_NAMES,
            feature_vector,
        )

        start = {
            "robot_joint_position_rad": list(range(7)),
            "robot_joint_velocity_rad_s": list(range(7, 14)),
            "controller_goal_position_m": [14, 15, 16],
            "obstacle_position_m": [17, 18, 19],
            "clearance_m": list(range(20, 28)),
        }
        result = feature_vector(start, [27, 28, 29], [30, 31, 32])
        self.assertEqual(result.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.array_equal(result, np.arange(33, dtype=np.float64)))

    def test_evaluator_is_exact_substep_state_grouped_and_opt_in(self):
        source = (
            ROOT / "scripts/evaluate_distal_execution_margin_nn_e05.py"
        ).read_text()
        self.assertIn("SubstepEightConstraintProbe", source)
        self.assertIn("ExactObstacleBoxUnion", source)
        self.assertIn("train_distal_execution_margin_nn_e05.py", source)
        self.assertIn("project_action_with_model", source)
        self.assertIn("D_opt_proxy_safe", source)
        self.assertIn("D_sim_raw_safe", source)
        self.assertIn(
            'dataset_summary["oracle_analysis_gate_pass"] is True', source
        )
        self.assertIn("openpi_python=args.openpi_python.absolute()", source)
        self.assertNotIn("openpi_python=args.openpi_python.resolve()", source)

    def test_pretraining_allocation_collects_only_and_validates_gate(self):
        source = (
            ROOT / "slurm/distal_execution_margin_dataset_e05.sbatch"
        ).read_text()
        self.assertIn("--gres=gpu:1", source)
        self.assertIn("H100", source)
        self.assertIn("collect_distal_execution_margin_dataset_e05.py", source)
        self.assertIn("validate_distal_execution_margin_dataset_e05.py", source)
        self.assertNotIn("train_distal_execution_margin_nn_e05.py", source)
        self.assertNotIn("OPENPI_PYTHON", source)

    def test_validator_requires_model_representation_and_projection_gates(self):
        source = (
            ROOT / "scripts/validate_distal_execution_margin_nn_e05.py"
        ).read_text()
        self.assertIn("expected_model_gate", source)
        self.assertIn("representation_gate", source)
        self.assertIn("expected_projection_gate", source)
        self.assertIn("exact_projected_substep_verification_when_valid", source)
        self.assertIn("execution-margin training artifact differs", source)

    def test_allocation_requires_clean_source_and_one_H100(self):
        source = (
            ROOT / "slurm/distal_execution_margin_nn_e05.sbatch"
        ).read_text()
        self.assertIn("--gres=gpu:1", source)
        self.assertIn("H100", source)
        self.assertIn("status --porcelain=v1 --untracked-files=all", source)
        self.assertIn("tests.test_distal_execution_margin_nn", source)
        self.assertIn("validate_distal_execution_margin_nn_e05.py", source)


if __name__ == "__main__":
    unittest.main()
