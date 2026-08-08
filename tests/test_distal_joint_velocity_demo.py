import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DistalJointVelocityDemoTests(unittest.TestCase):
    def test_config_is_live_translational_joint_velocity_pair(self) -> None:
        from main.multilink_ellipsoid.distal_joint_velocity_demo import (
            load_demo_config,
        )

        config = load_demo_config(
            ROOT / "configs/vlsa_distal_joint_velocity_demo_e05.v1.json"
        )
        self.assertEqual(config["controller"]["type"], "JOINT_VELOCITY")
        self.assertEqual(
            config["arms"],
            [
                "joint_velocity_live_base",
                "joint_velocity_live_l5_l7_ee",
            ],
        )
        self.assertTrue(
            config["nominal_policy"]["query_on_live_joint_controller_observation"]
        )
        self.assertEqual(
            config["robot_geometry"]["part_counts"],
            {"robot0_link5": 3, "robot0_link6": 2, "robot0_link7": 2},
        )
        self.assertEqual(
            config["end_effector_geometry"]["semiaxes_m"], [0.06, 0.12, 0.11]
        )
        self.assertEqual(len(config["config_payload_sha256"]), 64)

    def test_runner_gates_active_on_live_baseline_competence(self) -> None:
        source = (
            ROOT / "scripts/evaluate_distal_joint_velocity_demo_e05.py"
        ).read_text(encoding="utf-8")
        self.assertIn("client.infer(policy_input)", source)
        self.assertIn("if baseline_competent:", source)
        self.assertIn('active_skipped_due_to_competence_gate', source)
        self.assertIn('constraint_count_per_step\": 8', source)
        self.assertNotIn("archived_actions", source)

    def test_config_is_plain_json(self) -> None:
        value = json.loads(
            (
                ROOT / "configs/vlsa_distal_joint_velocity_demo_e05.v1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            value["claim_scope"],
            "single_primary_case_live_pi05_libero_joint_velocity_distal_l5_l7_plus_released_ee_ability_demo_not_population_or_whole_arm_claim",
        )

    def test_h100_runner_starts_frozen_pi05_libero(self) -> None:
        source = (
            ROOT / "slurm/distal_joint_velocity_demo_e05.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("simulation requires exactly one visible H100", source)
        self.assertIn("policy.config pi05_libero", source)
        self.assertIn("evaluate_distal_joint_velocity_demo_e05.py", source)
        self.assertNotIn("#SBATCH --array", source)


if __name__ == "__main__":
    unittest.main()
