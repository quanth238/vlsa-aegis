import json
import unittest
from pathlib import Path

from main.multilink_ellipsoid.adaptive_query_action_risk_v2 import (
    choose_target_row,
    coarse_candidate_definitions,
    load_config,
    midpoint_definition,
    row_support,
    select_row_bracket,
)


def screening_record(name, order, risks, offset, physical=False):
    return {
        "definition": {
            "name": name,
            "order": order,
            "actions": [[float(offset)] + [0.0] * 6 for _ in range(5)],
        },
        "prefix_risk": list(risks) + [-0.1] * 4,
        "physical_veto": physical,
    }


class AdaptiveBoundaryV2Tests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(
            Path("configs/vlsa_distal_adaptive_query_action_risk.v2.json")
        )

    def test_fixed_bank_has_profiles_amplitudes_and_seeded_mixtures(self):
        nominal = [[0.0] * 7 for _ in range(5)]
        frame = {
            "normal": [1.0, 0.0, 0.0],
            "tangent_up": [0.0, 1.0, 0.0],
            "tangent_side": [0.0, 0.0, 1.0],
        }
        base = json.loads(Path(
            "configs/vlsa_distal_query_action_risk_e05.v1.json"
        ).read_text())
        bank = coarse_candidate_definitions(nominal, frame, base, self.config)
        self.assertEqual(len(bank), 12)
        self.assertEqual(bank[0]["name"], "nominal")
        self.assertEqual(
            sum(item["direction_source"] == "fixed_seed_mixture" for item in bank),
            3,
        )
        self.assertEqual(
            {item["temporal_profile"] for item in bank[1:]},
            {"constant", "front_loaded"},
        )
        self.assertEqual(
            {item["requested_correction_l2_action"] for item in bank[1:]},
            {1.0, 1.5, 2.0},
        )

    def test_target_row_requires_global_safe_and_row_positive_support(self):
        rows = [
            screening_record("nominal", 0, [0.004, 0.002, -0.003], 0.0),
            screening_record("safe", 1, [-0.002, -0.003, -0.004], 0.1),
            screening_record("row0_positive", 2, [0.003, -0.002, -0.003], 0.2),
            screening_record("row1_positive", 3, [-0.002, 0.003, -0.003], 0.3),
        ]
        support = row_support(rows, 0.0005)
        self.assertTrue(support[0]["controllable_with_global_safe_endpoint"])
        self.assertTrue(support[1]["controllable_with_global_safe_endpoint"])
        self.assertFalse(support[2]["controllable_with_global_safe_endpoint"])
        self.assertEqual(choose_target_row(rows, support), 0)
        self.assertEqual(
            select_row_bracket(rows, row=0, epsilon_m=0.0005), (1, 2)
        )

    def test_proxy_or_contact_invalid_action_is_not_safe_endpoint(self):
        rows = [
            screening_record("nominal", 0, [0.003, -0.003, -0.003], 0.0),
            screening_record(
                "negative_but_contact", 1, [-0.004, -0.004, -0.004],
                0.1, physical=True,
            ),
        ]
        support = row_support(rows, 0.0005)
        self.assertIsNone(choose_target_row(rows, support))

    def test_midpoint_is_direct_horizon_displacement(self):
        negative = screening_record(
            "negative", 1, [-0.002, -0.002, -0.002], 0.0
        )["definition"]
        positive = screening_record(
            "positive", 2, [0.002, -0.002, -0.002], 0.4
        )["definition"]
        middle = midpoint_definition(
            negative, positive, iteration=0, order=12
        )
        self.assertEqual([row[0] for row in middle["actions"]], [0.2] * 5)


if __name__ == "__main__":
    unittest.main()
