import json
from pathlib import Path
import tempfile
import unittest

from main.multilink_ellipsoid.terminalized_full_episode import (
    load_config, scientific_view,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_terminalized_full_episode.v1.json"


class TerminalizedFullEpisodeTest(unittest.TestCase):
    def test_config_starts_at_zero_and_abstains(self):
        config = load_config(CONFIG)
        self.assertEqual(config["method"]["start_step"], 0)
        self.assertEqual(
            config["method"]["selection"],
            "least_modifying_predicted_safe_else_abstain",
        )
        self.assertFalse(config["forbidden"]["archived_action_prefix"])
        self.assertFalse(config["forbidden"]["released_AEGIS_EE_QP"])

    def test_config_rejects_archived_prefix(self):
        value = json.loads(CONFIG.read_text())
        value["forbidden"]["archived_action_prefix"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "forbidden"):
                load_config(path)

    def test_evaluator_uses_selector_from_first_query(self):
        source = (ROOT / "scripts/evaluate_terminalized_full_episode.py").read_text()
        self.assertIn("query_index = 0", source)
        self.assertIn("select_safe_or_abstain", source)
        self.assertNotIn("_archived_action", source)
        self.assertNotIn("env.step(action_rows", source)

    def test_scientific_view_separates_task_and_collision(self):
        result = {
            "case_id": "case", "pairing_sha256": "pair",
            "episode": {
                "complete_executed_actions_sha256": "a", "action_count": 5,
                "selection_count": 1, "selection_history_sha256": "b",
                "abstention_count": 0, "terminal_reason": "raw_robot_contact",
                "native_task_success": True, "native_task_success_step": 4,
                "timeout": False, "raw_robot_contact_pass": False,
                "first_raw_robot_contact": {"step": 4},
                "robot_contact_sample_count": 1,
                "robot_contact_sample_count_by_group": {"L5": 1},
                "robot_contact_events_sha256": "c", "paper_CAR_pass": True,
                "first_paper_CAR_step": None,
                "maximum_active_obstacle_l1_displacement_m": 0.0,
                "collision_free_task_success": False,
                "terminal_dynamic_state_sha256": "d",
                "goal_progress_summary_sha256": "e",
            },
        }
        view = scientific_view(result)
        self.assertTrue(view["native_task_success"])
        self.assertFalse(view["collision_free_task_success"])


if __name__ == "__main__":
    unittest.main()
