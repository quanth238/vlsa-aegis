import inspect
import unittest
from pathlib import Path

from main.multilink_ellipsoid.affine_coefficient_model import (
    load_affine_coefficient_config,
)
from main.multilink_ellipsoid.same_task_boundary_expansion import (
    build_expanded_dataset, collection_config, load_config,
    load_selected_manifest,
)
from main.multilink_ellipsoid.state_support_smoothness import context_arrays
from main.multilink_ellipsoid.two_step_margin import PAIR_FEATURE_NAMES
from scripts.collect_distal_affine_coefficient_moka10 import collect


ROOT = Path(__file__).resolve().parents[1]


class SameTaskBoundaryExpansionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(
            ROOT
            / "configs/vlsa_distal_same_task_boundary_expansion_moka10.v1.json"
        )

    def test_complete_episode_split_keeps_failure_episodes_test_only(self):
        selected = load_selected_manifest(
            ROOT
            / "manifests/vlsa_distal_same_task_boundary_expansion_moka10.v1.jsonl",
            self.config,
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(
            [item["case_id"] for item in selected if item["split"] == "train"],
            ["vlsa-t1-goal-ii-t0-e00", "vlsa-t1-goal-ii-t0-e20"],
        )
        self.assertEqual(
            [item["case_id"] for item in selected if item["split"] == "validation"],
            ["vlsa-t1-goal-ii-t0-e30"],
        )
        self.assertTrue(
            set(self.config["episode_split"]["immutable_test_case_ids"])
            .isdisjoint(item["case_id"] for item in selected)
        )

    def test_collection_adapter_preserves_frozen_sampling(self):
        affine = load_affine_coefficient_config(
            ROOT / "configs/vlsa_distal_affine_coefficient_moka10.v1.json"
        )
        adapted = collection_config(self.config, affine)
        self.assertEqual(adapted["state_sampling"], affine["state_sampling"])
        self.assertEqual(adapted["action_sampling"], affine["action_sampling"])
        self.assertEqual(adapted["coefficient_target"], affine["coefficient_target"])
        signature = inspect.signature(collect)
        self.assertIn("config_override", signature.parameters)
        self.assertIn("selected_override", signature.parameters)
        self.assertIsNone(signature.parameters["config_override"].default)

    def test_context_normalization_accepts_expanded_train_count(self):
        states = []
        for index in range(2):
            states.append({
                "state_index": index, "split": "train",
                "pair_state_feature_vectors": [
                    [float(index)] * len(PAIR_FEATURE_NAMES) for _ in range(7)
                ],
            })
        output = context_arrays(states, expected_train_state_count=2)
        self.assertEqual(len(output["standardized"]), 2)

    def test_expanded_dataset_preserves_original_test_records(self):
        test_ids = self.config["episode_split"]["immutable_test_case_ids"]
        original_states = []
        for index in range(50):
            if index < 30:
                split, case_id = "train", "old-train-%02d" % (index // 5)
            elif index < 35:
                split, case_id = "validation", "old-validation"
            else:
                split, case_id = "test", test_ids[(index - 35) // 5]
            original_states.append({
                "state_index": index, "split": split, "case_id": case_id,
            })
        additional_states = []
        cases = [
            ("vlsa-t1-goal-ii-t0-e00", "train"),
            ("vlsa-t1-goal-ii-t0-e20", "train"),
            ("vlsa-t1-goal-ii-t0-e30", "validation"),
        ]
        for index, (case_id, split) in enumerate(cases):
            for _ in range(5):
                additional_states.append({
                    "state_index": len(additional_states), "split": split,
                    "case_id": case_id,
                })
        original = {
            "state_records": original_states,
            "dataset_payload_sha256": "original", "constraint_order": list(range(7)),
        }
        additional = {
            "state_records": additional_states,
            "dataset_payload_sha256": "additional",
        }
        expanded = build_expanded_dataset(original, additional, self.config)
        self.assertEqual(expanded["summary"]["state_count"], 65)
        self.assertEqual(
            expanded["summary"]["split_counts"],
            {"train": 40, "validation": 10, "test": 15},
        )
        self.assertEqual(
            expanded["state_records"][35:50], original_states[35:50]
        )


if __name__ == "__main__":
    unittest.main()
