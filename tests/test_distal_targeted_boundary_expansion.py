import inspect
import unittest
from pathlib import Path

from main.multilink_ellipsoid.affine_coefficient_model import (
    load_affine_coefficient_config,
)
from main.multilink_ellipsoid.targeted_boundary_expansion import (
    build_dataset, collection_config, load_config, load_selected_manifest,
    select_boundary_steps,
)
from scripts.collect_distal_affine_coefficient_moka10 import collect


ROOT = Path(__file__).resolve().parents[1]


class TargetedBoundaryExpansionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(
            ROOT
            / "configs/vlsa_distal_targeted_boundary_expansion_moka10.v1.json"
        )

    def test_complete_episode_split_has_no_test_leakage(self):
        selected = load_selected_manifest(
            ROOT
            / "manifests/vlsa_distal_targeted_same_task_screen_moka10.v1.jsonl",
            self.config,
        )
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(item["split"] == "train" for item in selected))
        self.assertTrue(all(
            item["first_robot_contact_step"] is None for item in selected
        ))
        self.assertTrue(
            set(self.config["episode_split"]["immutable_test_case_ids"])
            .isdisjoint(item["case_id"] for item in selected)
        )
        self.assertFalse(
            self.config["state_selection"][
                "uses_test_episode_features_or_labels"
            ]
        )

    def test_selection_uses_smallest_safe_own_episode_anchor(self):
        records = []
        for case_index, case_id in enumerate(
            self.config["episode_split"]["additional_train_case_ids"]
        ):
            for step in range(20, 31):
                clearance = 0.02 + 0.001 * case_index + abs(step - 27) * 0.01
                records.append({
                    "case_id": case_id, "state_step": step,
                    "minimum_current_clearance_m": clearance,
                })
        output = select_boundary_steps(records, self.config)
        self.assertTrue(output["all_episode_windows_valid"])
        self.assertFalse(output["selection_uses_future_rollout_margin"])
        self.assertFalse(output["selection_uses_test_episode_features_or_labels"])
        for item in output["episode_selections"]:
            self.assertEqual(item["anchor_step"], 27)
            self.assertEqual(item["registered_state_steps"], [23, 24, 25, 26, 27])

    def test_collection_adapter_and_fixed_step_hook_preserve_grid(self):
        affine = load_affine_coefficient_config(
            ROOT / "configs/vlsa_distal_affine_coefficient_moka10.v1.json"
        )
        adapted = collection_config(self.config, affine)
        self.assertEqual(adapted["state_sampling"], affine["state_sampling"])
        self.assertEqual(adapted["action_sampling"], affine["action_sampling"])
        self.assertEqual(adapted["coefficient_target"], affine["coefficient_target"])
        signature = inspect.signature(collect)
        self.assertIn("registered_state_steps_override", signature.parameters)
        self.assertIsNone(
            signature.parameters["registered_state_steps_override"].default
        )

    def test_combined_dataset_keeps_existing_test_states_unchanged(self):
        test_ids = self.config["episode_split"]["immutable_test_case_ids"]
        base_states = []
        for index in range(65):
            if index < 40:
                split, case_id = "train", "base-train-%02d" % (index // 5)
            elif index < 50:
                split, case_id = "validation", "base-validation-%02d" % (index // 5)
            else:
                split, case_id = "test", test_ids[(index - 50) // 5]
            base_states.append({
                "state_index": index, "split": split, "case_id": case_id,
            })
        additional_states = []
        for case_id in self.config["episode_split"]["additional_train_case_ids"]:
            for _ in range(5):
                additional_states.append({
                    "state_index": len(additional_states), "split": "train",
                    "case_id": case_id,
                })
        base = {
            "state_records": base_states, "dataset_payload_sha256": "base",
            "constraint_order": list(range(7)),
        }
        additional = {
            "state_records": additional_states,
            "dataset_payload_sha256": "additional",
        }
        output = build_dataset(base, additional, self.config)
        self.assertEqual(output["summary"]["state_count"], 85)
        self.assertEqual(
            output["summary"]["split_counts"],
            {"train": 60, "validation": 10, "test": 15},
        )
        self.assertEqual(output["state_records"][50:65], base_states[50:65])


if __name__ == "__main__":
    unittest.main()
