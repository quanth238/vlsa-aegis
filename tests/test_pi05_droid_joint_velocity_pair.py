from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Pi05DroidJointVelocityPairTests(unittest.TestCase):
    def test_preregistered_config_is_direct_joint_velocity(self) -> None:
        from main.multilink_ellipsoid.joint_velocity import (
            load_joint_velocity_pair_config,
        )

        config = load_joint_velocity_pair_config(
            ROOT / "configs/vlsa_pi05_droid_joint_velocity_pair_e05.v1.json"
        )
        self.assertEqual(config["controller"]["type"], "JOINT_VELOCITY")
        self.assertEqual(config["controller"]["input_max"], 1.0)
        self.assertEqual(config["controller"]["output_max"], 1.0)
        self.assertEqual(config["protected_body_names"], [
            "robot0_link5",
            "robot0_link6",
            "robot0_link7",
        ])
        self.assertEqual(config["pairing"]["action_horizon"], 225)
        self.assertEqual(config["pairing"]["control_frequency_hz"], 15)
        self.assertEqual(config["pairing"]["model_action_horizon"], 15)

    def test_safelibero_custom_controller_config_is_additive(self) -> None:
        source = (
            ROOT / "safelibero/libero/libero/envs/env_wrapper.py"
        ).read_text(encoding="utf-8")
        self.assertIn("controller_configs=None", source)
        self.assertIn("if controller_configs is None:", source)
        self.assertIn("suite.load_controller_config(default_controller=controller)", source)
        evaluator = (ROOT / "main/evaluate_safelibero_aegis.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("controller_configs: Mapping[str, Any] | None = None", evaluator)
        self.assertIn('env_args["controller_configs"] = dict(controller_configs)', evaluator)

    def test_preregistration_forbids_kkt_claim(self) -> None:
        source = (
            ROOT / "docs/pi05_droid_joint_velocity_pair_preregistration.md"
        ).read_text(encoding="utf-8")
        self.assertIn("not modify or reinterpret Table 1", source)
        self.assertIn("Interpretation is gated on baseline competence", source)
        self.assertIn("KKT/VI learning remains out of scope", source)

    def test_pair_runner_is_one_h100_and_uses_droid_checkpoint(self) -> None:
        source = (
            ROOT / "slurm/pi05_droid_joint_velocity_pair_e05.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("#SBATCH --cpus-per-task=8", source)
        self.assertIn("#SBATCH --mem=64G", source)
        self.assertIn("#SBATCH --exclude=worker-3", source)
        self.assertIn("--policy.config pi05_droid", source)
        self.assertIn("$PI05_DROID_CHECKPOINT", source)
        self.assertIn("evaluate_pi05_droid_joint_velocity_pair.py", source)
        self.assertIn("trap cleanup EXIT", source)

    def test_pair_evaluator_requires_15_by_8_droid_chunks(self) -> None:
        source = (
            ROOT / "scripts/evaluate_pi05_droid_joint_velocity_pair.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"observation/joint_position"', source)
        self.assertIn('"observation/gripper_position"', source)
        self.assertIn("expected_chunk_shape", source)
        self.assertIn("direct_l5_l6_l7_joint_velocity_multicbf", source)
        self.assertIn("first_policy_chunks_equal", source)


if __name__ == "__main__":
    unittest.main()
