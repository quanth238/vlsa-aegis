from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_poisson_arm_contact_manifest.py"
SPEC = importlib.util.spec_from_file_location("poisson_arm_contact_manifest", BUILDER)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class PhysicalArmContactExtractionTest(unittest.TestCase):
    def test_actual_link_body_is_required_not_an_ancestor(self) -> None:
        pairs = [
            {
                "geom1": "distal",
                "geom2": "bottle",
                "body_lineage1": [
                    "robot0_link7",
                    "robot0_link6",
                    "robot0_link5",
                    "robot0_base",
                    "world",
                ],
                "body_lineage2": ["wine_bottle_obstacle_1_main", "world"],
            }
        ]
        direct, robot = builder.extract_active_obstacle_robot_contacts(
            pairs, "wine_bottle_obstacle_1"
        )
        self.assertEqual(direct, [])
        self.assertEqual(robot, ("robot0_link7",))

    def test_direct_pairs_and_mixed_robot_bodies_are_separate(self) -> None:
        pairs = [
            {
                "geom1": "link5_collision",
                "geom2": "bottle_g1",
                "body_lineage1": ["robot0_link5", "robot0_base", "world"],
                "body_lineage2": ["wine_bottle_obstacle_1_main", "world"],
            },
            {
                "geom1": "finger_collision",
                "geom2": "bottle_g2",
                "body_lineage1": [
                    "gripper0_leftfinger",
                    "robot0_link7",
                    "robot0_base",
                    "world",
                ],
                "body_lineage2": ["wine_bottle_obstacle_1_main", "world"],
            },
        ]
        direct, robot = builder.extract_active_obstacle_robot_contacts(
            pairs, "wine_bottle_obstacle_1"
        )
        self.assertEqual(len(direct), 1)
        self.assertEqual(direct[0]["actual_link_body_name"], "robot0_link5")
        self.assertEqual(robot, ("gripper0_leftfinger", "robot0_link5"))


class CheckedInPhysicalManifestContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = ROOT / "manifests" / "vlsa_poisson_arm_contact_165.v1.jsonl"
        self.receipt = ROOT / "manifests" / "vlsa_poisson_arm_contact_165.v1.receipt.json"

    def test_checked_in_manifest_and_receipt_are_self_consistent(self) -> None:
        self.assertTrue(self.manifest.is_file(), "physical manifest is missing")
        self.assertFalse(self.manifest.is_symlink(), "physical manifest is symlinked")
        self.assertTrue(self.receipt.is_file(), "physical receipt is missing")
        self.assertFalse(self.receipt.is_symlink(), "physical receipt is symlinked")
        raw = self.manifest.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        receipt = json.loads(self.receipt.read_bytes())
        self.assertEqual(len(rows), 165)
        self.assertEqual(receipt["manifest_rows"], 165)
        self.assertEqual(receipt["manifest_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(
            receipt["manifest_sha256"],
            "80678be7bbef6be8027c4d276e3a0114b04409388df6ee74399d2e9f301a2935",
        )
        self.assertEqual(
            receipt["all_case_ids_sha256"],
            "e8a82d59d95dbaf915eb646d86e714183c1bf1d61664a0f526dc9f5445343717",
        )
        self.assertEqual(
            receipt["clean_case_ids_sha256"],
            "b2fc7242b9e76fd6075eb4ecf265891f49d3c7ea380ba02fff136dd99a33d71a",
        )
        self.assertEqual(sum(row["clean_task_success_canary_eligible"] for row in rows), 40)
        self.assertEqual(
            receipt["accepted_aegis_payload_bindings_sha256"],
            "e291b5c7e47cc48bc0a9c8cfee8629b0ce7eae4c158cfdc5255606680672fb4c",
        )
        self.assertEqual(
            {row["contact_vs_car_phase"] for row in rows},
            {"through_car", "post_car", "car_safe"},
        )
        self.assertTrue(
            all(row["sample_phase"] == "post_env_step_control_endpoint" for row in rows)
        )
        by_id = {row["case_id"]: row for row in rows}
        expected = {
            "vlsa-t1-spatial-i-t3-e03": (
                "through_car", 62, "robot0_link5", True, True
            ),
            "vlsa-t1-goal-ii-t3-e42": (
                "through_car", 105, "robot0_link6", True, True
            ),
            "vlsa-t1-goal-ii-t0-e15": (
                "through_car", 153, "robot0_link5", False, False
            ),
        }
        for case_id, (phase, step, body, task_success, clean_eligible) in expected.items():
            row = by_id[case_id]
            self.assertEqual(row["contact_vs_car_phase"], phase)
            self.assertEqual(row["first_sampled_link_contact_control_step"], step)
            self.assertIn(body, row["literal_link_contact_bodies"])
            self.assertIs(row["historical_outcome"]["task_success"], task_success)
            self.assertIs(row["clean_task_success_canary_eligible"], clean_eligible)
        self.assertEqual(
            by_id["vlsa-t1-goal-ii-t3-e42"]["literal_link_contact_pattern"],
            "link6_only",
        )


if __name__ == "__main__":
    unittest.main()
