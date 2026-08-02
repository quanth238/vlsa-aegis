from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest

from scripts.run_poisson_fast_feasibility import _post_correction_motion


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_poisson_fast_feasibility.py"


class PoissonFullEpisodeRunnerContractTest(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("numpy") is not None,
        "allocation-only numerical motion test requires numpy",
    )
    def test_post_correction_motion_excludes_preceding_updates(self):
        commands = []
        physics = []
        for update, boundary in enumerate((4495, 4500, 4505)):
            commands.append(
                {
                    "source_action_index": 180,
                    "inner_control_index": update,
                    "physical_boundary": boundary,
                    "executed_qdot_rad_s": [float(update + 1)] + [0.0] * 6,
                    "correction_l2_rad_s": float(update),
                    "eef_position_before_update_world_m": [
                        0.01 * update,
                        0.0,
                        0.0,
                    ],
                }
            )
            for substep in range(5):
                physics.append(
                    {
                        "source_action_index": 180,
                        "inner_control_index": update,
                        "physics_substep_index": substep,
                        "measured_qvel_rad_s": [float(update + 1)] + [0.0] * 6,
                        "eef_position_world_m": [
                            0.01 * update + 0.001 * (substep + 1),
                            0.0,
                            0.0,
                        ],
                    }
                )
        evidence = _post_correction_motion(
            commands,
            physics,
            first_correction_physical_boundary=4500,
        )
        self.assertEqual(evidence["command_update_count"], 2)
        self.assertEqual(evidence["physics_substep_count"], 10)
        self.assertEqual(evidence["complete_command_interval_count"], 2)
        self.assertAlmostEqual(evidence["filter_correction_integral_rad"], 0.03)
        self.assertAlmostEqual(evidence["executed_command_integral_rad"], 0.05)
        self.assertAlmostEqual(evidence["measured_joint_motion_integral_rad"], 0.05)
        self.assertAlmostEqual(evidence["cartesian_path_length_m"], 0.01)
        self.assertEqual(evidence["zero_command_fraction"], 0.0)

    def test_runner_declares_hybrid_full_episode_mode(self):
        source = RUNNER.read_text(encoding="utf-8")
        # Filled by the main integration; this assertion prevents accidentally
        # submitting the original 0.4-second protocol under the new launcher.
        self.assertIn("vlsa_poisson_full_episode_feasibility_protocol.v1", source)
        self.assertIn("classify_full_episode_feasibility", source)
        self.assertIn("expected_boundary_goal_values", source)
        self.assertIn("_post_correction_motion", source)
        self.assertIn('"source_action": source_action.tolist()', source)
        self.assertIn('"controller_updates_per_arm": derived[', source)
        self.assertIn('"physics_substeps_per_arm": derived[', source)
        self.assertNotIn('"registered_controller_updates_per_arm"', source)
        self.assertNotIn('"registered_physics_substeps_per_arm"', source)

    def test_command_trace_rows_serialize_the_exact_source_action(self):
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
        command_appends = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if not (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "command_rows"
                and node.func.attr == "append"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Dict)
            ):
                continue
            command_appends.append(node.args[0])
        self.assertEqual(len(command_appends), 1)
        keys = {
            key.value
            for key in command_appends[0].keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        self.assertIn("source_action_index", keys)
        self.assertIn("source_action", keys)
        source_value = command_appends[0].values[
            [
                key.value if isinstance(key, ast.Constant) else None
                for key in command_appends[0].keys
            ].index("source_action")
        ]
        self.assertIsInstance(source_value, ast.Call)
        self.assertIsInstance(source_value.func, ast.Attribute)
        self.assertIsInstance(source_value.func.value, ast.Name)
        self.assertEqual(source_value.func.value.id, "source_action")
        self.assertEqual(source_value.func.attr, "tolist")


if __name__ == "__main__":
    unittest.main()
