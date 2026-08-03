import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_poisson_triggered_rescue.py"
FAST_RUNNER_PATH = ROOT / "scripts/run_poisson_fast_feasibility.py"


class TriggeredRescueRunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_runner_is_syntax_valid_and_contains_no_fixed_trigger_action(self):
        self.assertIsInstance(self.tree, ast.Module)
        for literal in ("action 19", "action 180", "boundary 4500", "boundary 4510"):
            self.assertNotIn(literal, self.source)
        self.assertNotIn("osc_reference_governor", self.source)
        self.assertNotIn("OSCReferenceGovernor", self.source)

    def test_trigger_uses_bounded_direct_qdot_and_solved_hard_qp(self):
        self.assertIn("TranslationalJointVelocityAdapter", self.source)
        self.assertIn("preview.qdot_physical", self.source)
        self.assertIn("joint_velocity_bounds", self.source)
        self.assertIn("qp.solve_from_field", self.source)
        self.assertIn("evaluate_direct_qdot_trigger_candidate", self.source)
        self.assertIn(
            '"fixed_trigger_action_index_used": False', self.source
        )

    def test_native_osc_step_occurs_only_after_nontrigger_scan(self):
        trigger_break = self.source.index("if candidate[\"triggered\"]:")
        native_step = self.source.index(
            "source_env.step(expected_step.action)", trigger_break
        )
        self.assertLess(trigger_break, native_step)
        self.assertIn("_check_step", self.source[native_step:])

    def test_pair_uses_exact_trigger_state_and_remaining_recorded_suffix(self):
        self.assertIn("trigger_state = np.asarray", self.source)
        self.assertIn("suffix_actions = replay.actions[trigger_action:]", self.source)
        self.assertEqual(self.source.count("state_at_B=trigger_state"), 2)
        self.assertEqual(self.source.count("actions=suffix_actions"), 2)
        self.assertEqual(
            self.source.count("monitor_registered_forbidden_contacts=True"),
            2,
        )

    def test_suffix_paper_car_uses_settled_reference_not_branch_position(self):
        self.assertIn("paper_car_reference_position = np.asarray", self.source)
        self.assertEqual(
            self.source.count(
                "paper_car_reference_position=paper_car_reference_position"
            ),
            2,
        )
        fast_source = FAST_RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("external_paper_car_reference", fast_source)
        self.assertIn(
            "start_physical_boundary=source_start_action * 25", fast_source
        )

    def test_positive_contract_keeps_task_motion_car_and_shifted_contact(self):
        for token in (
            "psf_shifted_link56_external_contact_present",
            "psf_paper_car_avoided",
            "psf_useful_post_correction_motion",
            "psf_task_success_after_correction",
            "psf_terminal_task_success",
            "psf_method_stop_or_stall",
        ):
            self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
