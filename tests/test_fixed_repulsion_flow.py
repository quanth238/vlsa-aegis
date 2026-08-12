from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FixedRepulsionFlowTests(unittest.TestCase):
    def test_config_freezes_early_slots_and_final_flow_steps(self) -> None:
        from main.multilink_ellipsoid.repulsive_flow import (
            load_repulsive_flow_config,
        )

        config = load_repulsive_flow_config(
            ROOT / "configs/vlsa_fixed_repulsion_flow_e05.v1.json"
        )
        self.assertEqual(config["state_protocol"]["activation_query_step"], 180)
        self.assertEqual(config["flow_guidance"]["guided_action_slots"], [2, 3, 4])
        self.assertEqual(config["flow_guidance"]["guided_euler_steps"], [5, 6, 7, 8, 9])
        self.assertEqual(config["flow_guidance"]["total_nominal_guidance_budget_action"], 0.25)

    def test_websocket_envelope_is_reserved_and_strict(self) -> None:
        source = (
            ROOT / "openpi/src/openpi/serving/websocket_policy_server.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"repulsive_flow_guidance"', source)
        self.assertIn("_validate_repulsive_flow_guidance", source)
        self.assertIn("guided_action_slots must equal executed steps 182--184", source)

    def test_model_applies_repulsion_inside_euler_loop(self) -> None:
        source = (ROOT / "openpi/src/openpi/models/pi0.py").read_text(encoding="utf-8")
        start = source.index("def sample_actions_with_repulsive_flow_guidance")
        end = source.index("def sample_actions_flow_step", start)
        method = source[start:end]
        self.assertIn("x_t + dt * v_t", method)
        self.assertIn("euler_mask[step_index]", method)
        self.assertIn("slot_mask * direction", method)
        self.assertIn("jax.lax.while_loop", method)

    def test_runner_compares_ordinary_posthoc_and_flow_arms(self) -> None:
        source = (
            ROOT / "scripts/evaluate_fixed_repulsion_flow_e05.py"
        ).read_text(encoding="utf-8")
        for arm in (
            "ordinary_pi05_chunk",
            "posthoc_fixed_repulsion",
            "fixed_repulsion_inside_final_flow_steps",
        ):
            self.assertIn(arm, source)
        self.assertIn("direction_gate_pass", source)
        self.assertIn("if direction_gate_pass:", source)
        self.assertIn(
            '"diagnostic_only_no_late_query_equivalence_claim"', source
        )
        self.assertNotIn("live query-36 chunk exceeds pairing tolerance", source)


if __name__ == "__main__":
    unittest.main()
