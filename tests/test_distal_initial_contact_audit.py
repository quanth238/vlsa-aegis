import unittest
from pathlib import Path
from unittest.mock import patch

from main.multilink_ellipsoid.initial_contact_audit import load_config, summarize


ROOT = Path(__file__).resolve().parents[1]


class InitialContactAuditTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(
            ROOT / "configs/vlsa_distal_initial_contact_audit_moka10.v1.json"
        )

    def test_positive_native_distance_is_forbidden(self):
        self.assertTrue(
            self.config["measurement"]["positive_native_distance_forbidden"]
        )
        self.assertEqual(self.config["measurement"]["distmax_m"], 0.0)

    def test_gate_runs_no_learning_or_control(self):
        gate = self.config["gate"]
        self.assertFalse(gate["training_in_this_gate"])
        self.assertFalse(gate["QP_in_this_gate"])
        self.assertFalse(gate["closed_loop_in_this_gate"])

    def test_summary_separates_prevention_and_recovery(self):
        records = []
        for index in range(85):
            split = "train" if index < 60 else "validation" if index < 70 else "test"
            recovery = index == 0
            records.append({
                "split": split, "cohort": "recovery" if recovery else "prevention",
                "negative_without_raw_pair_contact_count": 0,
                "raw_pair_contact_without_negative_count": 0,
                "unregistered_raw_pair_count": 0, "all_queries_finite": True,
            })
        output = summarize(records, self.config, semantic_hash_count=1)
        self.assertEqual(output["aggregates"]["prevention_state_count"], 84)
        self.assertEqual(output["aggregates"]["recovery_state_count"], 1)
        self.assertTrue(output["decision"]["initial_contact_audit_complete"])

    def test_live_inventory_uses_group_index(self):
        from main.multilink_ellipsoid.initial_contact_audit import (
            measure_initial_contact,
        )

        inventory = {
            "active_obstacle_root_body_id": 7,
            "groups": [{
                "group_index": 0, "body_name": "robot0_link5",
                "protected_geom_id": 2, "obstacle_geom_ids": [3],
            }],
        }
        query = {"distance_m": 0.0, "finite": True}
        with patch(
            "main.multilink_ellipsoid.initial_contact_audit._native_model_data",
            return_value=(object(), object()),
        ), patch(
            "main.multilink_ellipsoid.initial_contact_audit._raw_pair_contacts",
            return_value=(set(), []),
        ), patch(
            "main.multilink_ellipsoid.initial_contact_audit._pair_query",
            return_value=query,
        ):
            result = measure_initial_contact(
                object(), inventory, self.config["measurement"]
            )
        self.assertEqual(result["group_records"][0]["group_index"], 0)
        self.assertEqual(result["cohort"], "prevention")


if __name__ == "__main__":
    unittest.main()
