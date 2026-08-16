from __future__ import annotations

import unittest
from itertools import product
from pathlib import Path

from main.multilink_ellipsoid.whole_body_candidate_bank import (
    candidate_name, candidate_vector, load_bank_config, opposite_name,
    select_bank,
)
from main.multilink_ellipsoid.whole_body_support_audit import load_audit_config


ROOT = Path(__file__).resolve().parents[1]
AUDIT_CONFIG = ROOT / "configs/vlsa_distal_whole_body_support_audit.v1.json"
BANK_CONFIG = ROOT / "configs/vlsa_distal_whole_body_candidate_bank.v1.json"


def _candidate(name, vector, invert=False):
    q = float(vector[0]) * (-1.0 if invert else 1.0)
    row_slack = [0.2, 0.2, 0.2, 0.2, -q, 0.2, 0.2, 0.2, 0.2]
    groups = {
        "end_effector": row_slack[0], "palm": row_slack[1],
        "L5": min(row_slack[2:5]), "L6": min(row_slack[5:7]),
        "L7": min(row_slack[7:9]),
    }
    return {
        "name": name,
        "requested_alpha": 0.0 if name == "nominal" else 2.0,
        "source_effective_post_AEGIS_correction_l2_action": (
            0.0 if name == "nominal" else 2.0
        ),
        "raw_protected_contact_sample_count": 0,
        "replayed_maximum_CAR_m": 0.0,
        "replayed_physical_veto": False,
        "exact_group_target": {
            "known_outcome": True,
            "group_future_violation": {
                group: -slack for group, slack in groups.items()
            },
            "group_contact_sample_count": {group: 0 for group in groups},
            "trace": [{"row_normalized_radial_slack": row_slack}],
        },
    }


def _case(case_id, split, invert=False):
    candidates = [_candidate("nominal", (0, 0, 0), invert=invert)]
    candidates.extend(
        _candidate(candidate_name(vector), vector, invert=invert)
        for vector in product((-1, 0, 1), repeat=3)
        if vector != (0, 0, 0)
    )
    groups = ["end_effector", "palm", "L5", "L6", "L7"]
    return {
        "case_id": case_id,
        "selection": {
            "episode_group_id": case_id, "split": split, "target_group": "L5",
        },
        "exact_case": {
            "exact_group_target": {
                "initial_group_normalized_radial_slack": {
                    group: 0.2 for group in groups
                },
                "initial_group_contact_sample_count": {group: 0 for group in groups},
            },
            "candidates": candidates,
        },
    }


class WholeBodyCandidateBankTest(unittest.TestCase):
    def test_config_forbids_holdout_influence(self):
        value = load_bank_config(BANK_CONFIG)
        self.assertFalse(value["validation_test_outcomes_influence_selection"])

    def test_candidate_vector_round_trip(self):
        name = "grid_m1_z0_p1_front_loaded_r2.0"
        self.assertEqual(candidate_vector(name), (-1, 0, 1))
        self.assertEqual(candidate_name(candidate_vector(name)), name)

    def test_opposite_pair_is_involutive(self):
        name = "grid_m1_p1_z0_front_loaded_r2.0"
        opposite = "grid_p1_m1_z0_front_loaded_r2.0"
        self.assertEqual(opposite_name(name), opposite)
        self.assertEqual(opposite_name(opposite), name)

    def test_selection_is_symmetric_and_independent_of_holdout_labels(self):
        audit = load_audit_config(AUDIT_CONFIG)
        first = select_bank([
            _case("train", "train"), _case("holdout", "validation", invert=False),
        ], audit)
        second = select_bank([
            _case("train", "train"), _case("holdout", "validation", invert=True),
        ], audit)
        self.assertEqual(first["selected_candidate_names"], second["selected_candidate_names"])
        names = first["selected_candidate_names"]
        self.assertEqual(len(names), 13)
        self.assertEqual(names[0], "nominal")
        self.assertEqual(set(names[1:]), {opposite_name(name) for name in names[1:]})
        self.assertTrue(first["selected_development_summary"]["per_case"][0][
            "per_group"
        ]["L5"]["two_sided_support"])


if __name__ == "__main__":
    unittest.main()
