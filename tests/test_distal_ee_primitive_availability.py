import gzip
import json
from pathlib import Path
import tempfile
import unittest

from main.multilink_ellipsoid.ee_primitive_availability import (
    classify_robot_group, contact_evidence, summarize_records,
)


def config():
    return {
        "constraint_groups": [
            {"name": "palm", "body_names": ["hand"], "geom_names": ["hand_geom"]},
            {"name": "finger1", "body_names": [], "geom_names": ["finger1"]},
            {"name": "finger2", "body_names": [], "geom_names": ["finger2"]},
            {"name": "L5", "body_names": ["L5"], "geom_names": ["L5_geom"]},
            {"name": "L6", "body_names": ["L6"], "geom_names": ["L6_geom"]},
            {"name": "L7", "body_names": ["L7"], "geom_names": ["L7_geom"]},
        ],
        "geometry_audit_gate": {
            "minimum_clean_contact_episodes": 2,
            "minimum_distinct_task_level_groups": 2,
            "minimum_matched_contact_free_controls": 2,
        },
        "next_if_ready": "fit_geometry",
    }


class EEPrimitiveAvailabilityTest(unittest.TestCase):
    def test_exact_geom_maps_finger_without_body_alias(self):
        self.assertEqual(
            classify_robot_group(
                body_name="finger_body", geom_name="finger1", config=config(),
            ),
            "finger1",
        )
        self.assertEqual(
            classify_robot_group(
                body_name="hand", geom_name="finger1", config=config(),
            ),
            "finger1",
        )

    def test_contact_evidence_keeps_unmapped_and_dynamic_events(self):
        artifact = {
            "schema_version": "vlsa_table1_active_obstacle_contacts.v3",
            "snapshots": [{"events": [
                {"step": 7, "other": {
                    "classification": "robot", "body_name": "hand",
                    "geom_name": "hand_geom",
                }},
                {"step": 8, "other": {
                    "classification": "robot", "body_name": "unknown",
                    "geom_name": "unknown_geom",
                }},
                {"step": 8, "other": {
                    "classification": "dynamic_task_object", "body_name": "cup",
                    "geom_name": "cup_geom",
                }},
            ]}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contacts.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump(artifact, stream)
            result = contact_evidence(path, config())
        self.assertEqual(result["robot_contact_counts_by_constraint_group"], {"palm": 1})
        self.assertEqual(result["unmapped_robot_contact_counts"], {"unknown|unknown_geom": 1})
        self.assertEqual(result["event_counts_by_role"]["dynamic_task_object"], 1)

    def test_summary_requires_independent_cohorts_and_controls(self):
        records = [
            {
                "case_id": "contact-a", "classification": "clean",
                "task_level_group_id": "task-a", "contact_groups": ["palm"],
                "eligible_clean_contact_groups": ["palm"],
                "eligible_clean_contact_free_control": False,
                "contacts": {"robot_contact_counts_by_geom": {"hand_geom": 1}},
            },
            {
                "case_id": "contact-b", "classification": "clean",
                "task_level_group_id": "task-b", "contact_groups": ["palm"],
                "eligible_clean_contact_groups": ["palm"],
                "eligible_clean_contact_free_control": False,
                "contacts": {"robot_contact_counts_by_geom": {"hand_geom": 1}},
            },
            {
                "case_id": "control-a", "classification": "control",
                "task_level_group_id": "task-a", "contact_groups": [],
                "eligible_clean_contact_groups": [],
                "eligible_clean_contact_free_control": True, "contacts": {},
            },
            {
                "case_id": "control-b", "classification": "control",
                "task_level_group_id": "task-b", "contact_groups": [],
                "eligible_clean_contact_groups": [],
                "eligible_clean_contact_free_control": True, "contacts": {},
            },
        ]
        summary = summarize_records(records, config())
        palm = summary["constraint_group_coverage"][0]
        self.assertTrue(palm["tighter_geometry_audit_ready"])
        self.assertEqual(summary["tighter_geometry_audit_ready_groups"], ["palm"])
        self.assertEqual(summary["next_action"], "fit_geometry")


if __name__ == "__main__":
    unittest.main()
