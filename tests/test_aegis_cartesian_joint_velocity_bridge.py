import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class AegisCartesianJointVelocityBridgeTests(unittest.TestCase):
    def test_preregistered_config_uses_direct_joint_execution(self) -> None:
        from main.multilink_ellipsoid.cartesian_bridge import (
            load_cartesian_bridge_config,
        )

        config = load_cartesian_bridge_config(
            ROOT / "configs/vlsa_aegis_cartesian_joint_velocity_bridge_pair_e05.v1.json"
        )
        self.assertEqual(config["controller"]["type"], "JOINT_VELOCITY")
        self.assertEqual(
            config["output_action_contract"]["arm"],
            "execute_physical_joint_velocity_qp_variable_directly",
        )
        self.assertEqual(config["protected_body_names"], [
            "robot0_link5",
            "robot0_link6",
            "robot0_link7",
        ])
        self.assertEqual(config["pairing"]["action_horizon"], 237)
        self.assertEqual(
            config["bridge"]["cartesian_velocity_m_per_s_per_action_unit"],
            0.2,
        )

    def test_internal_qp_config_preserves_frozen_optimizer(self) -> None:
        from main.multilink_ellipsoid.cartesian_bridge import (
            internal_joint_velocity_cbf_config,
            load_cartesian_bridge_config,
        )

        config = load_cartesian_bridge_config(
            ROOT / "configs/vlsa_aegis_cartesian_joint_velocity_bridge_pair_e05.v1.json"
        )
        internal = internal_joint_velocity_cbf_config(config)
        self.assertEqual(internal["optimizer"], config["optimizer"])
        self.assertEqual(internal["pairing"]["control_frequency_hz"], 20)
        self.assertEqual(internal["protected_body_names"], config["protected_body_names"])

    def test_preregistration_gates_active_claim_on_bridge_competence(self) -> None:
        text = (
            ROOT / "docs/aegis_cartesian_joint_velocity_bridge_pair_preregistration.md"
        ).read_text(encoding="utf-8")
        self.assertIn("must both satisfy the native goal and reproduce", text)
        self.assertIn("active arm cannot support a safety-efficacy conclusion", text)
        self.assertIn("does not authorize KKT/VI training", text)

    def test_config_is_canonical_json_object(self) -> None:
        path = ROOT / "configs/vlsa_aegis_cartesian_joint_velocity_bridge_pair_e05.v1.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsInstance(value, dict)
        self.assertEqual(
            value["nominal_action_source"],
            "immutable_archived_released_aegis_env_step_input",
        )

    def test_h100_runner_has_no_policy_server(self) -> None:
        text = (
            ROOT / "slurm/aegis_cartesian_joint_velocity_bridge_pair_e05.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertIn("#SBATCH --exclude=worker-3", text)
        self.assertIn("replay_aegis_cartesian_joint_velocity_bridge_pair.py", text)
        self.assertNotIn("serve_policy", text)


if __name__ == "__main__":
    unittest.main()
