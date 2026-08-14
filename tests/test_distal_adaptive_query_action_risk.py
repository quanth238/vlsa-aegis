import unittest
import json
import tempfile
from pathlib import Path

from main.multilink_ellipsoid.adaptive_query_action_risk import (
    load_config, midpoint_definition, select_bracket,
)


def record(name, order, value, safe, unsafe, offset):
    actions = [[float(offset)] + [0.0] * 6 for _ in range(5)]
    return {
        "definition": {"name": name, "order": order, "actions": actions},
        "L5_prefix_risk_m": value,
        "screen_safe": safe,
        "screen_unsafe": unsafe,
    }


class AdaptiveBoundaryTests(unittest.TestCase):
    def test_registered_config_is_strict_and_loadable(self):
        path = Path("configs/vlsa_distal_adaptive_query_action_risk.v1.json")
        loaded = load_config(path)
        self.assertEqual(
            loaded["feature_audit"]["coverage_gate"][
                "minimum_train_known_candidates"
            ],
            40,
        )
        value = json.loads(path.read_text())
        value["screening"]["coarse_candidate_count"] = 8
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.json"
            changed.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_config(changed)

    def test_closest_action_pair_is_selected(self):
        rows = [
            record("safe_far", 0, -0.001, True, False, -1.0),
            record("safe_near", 1, -0.002, True, False, 0.0),
            record("unsafe_near", 2, 0.001, False, True, 0.1),
            record("unsafe_far", 3, 0.0001, False, True, 1.0),
        ]
        self.assertEqual(select_bracket(rows), (1, 2))

    def test_missing_side_has_no_bracket(self):
        rows = [record("safe", 0, -0.001, True, False, 0.0)]
        self.assertIsNone(select_bracket(rows))

    def test_midpoint_does_not_integrate_actions(self):
        left = record("safe", 0, -1.0, True, False, 0.0)["definition"]
        right = record("unsafe", 1, 1.0, False, True, 0.4)["definition"]
        middle = midpoint_definition(left, right, iteration=0, order=9)
        self.assertEqual([row[0] for row in middle["actions"]], [0.2] * 5)


if __name__ == "__main__":
    unittest.main()
