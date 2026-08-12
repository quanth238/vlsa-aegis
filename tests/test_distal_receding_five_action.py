import json
from pathlib import Path
import unittest

from main.multilink_ellipsoid.receding_five_action import (
    load_receding_five_action_config,
    receding_query_index,
    rollout_is_safe,
)


ROOT = Path(__file__).resolve().parents[1]


class RecedingFiveActionTest(unittest.TestCase):
    def test_config_and_query_schedule(self):
        config = load_receding_five_action_config(
            ROOT / "configs/vlsa_distal_receding_five_action_e05.v1.json"
        )
        self.assertEqual(config["state_protocol"]["execute_prefix_actions"], 1)
        self.assertEqual(receding_query_index(183), 37)
        self.assertEqual(receding_query_index(197), 51)
        with self.assertRaises(ValueError):
            receding_query_index(182)

    def test_strict_rollout_gate(self):
        rollout = {
            "minimum_clearance_m": 0.001,
            "protected_contacts": [],
            "robot_contacts": [],
            "active_obstacle_l1_displacement_m": [0.0] * 5,
        }
        self.assertTrue(
            rollout_is_safe(
                rollout,
                clearance_buffer_m=0.001,
                paper_car_threshold_m=0.001,
            )
        )
        rollout["minimum_clearance_m"] = 0.000999
        self.assertFalse(
            rollout_is_safe(
                rollout,
                clearance_buffer_m=0.001,
                paper_car_threshold_m=0.001,
            )
        )

    def test_config_rejects_execute_five(self):
        source = ROOT / "configs/vlsa_distal_receding_five_action_e05.v1.json"
        value = json.loads(source.read_text())
        value["state_protocol"]["execute_prefix_actions"] = 5
        temporary = ROOT / "tests" / ".temporary-receding-five-action.json"
        try:
            temporary.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_receding_five_action_config(temporary)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
