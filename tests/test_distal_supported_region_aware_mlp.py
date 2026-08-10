import unittest
from pathlib import Path

from main.multilink_ellipsoid.region_aware_mlp import (
    load_config as load_reference_config,
)
from main.multilink_ellipsoid.supported_region_aware_mlp import (
    fresh_payload, load_config, require_unchanged_model_sections,
)


ROOT = Path(__file__).resolve().parents[1]


class SupportedRegionAwareMlpTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(
            ROOT
            / "configs/vlsa_distal_supported_region_aware_mlp_moka10.v1.json"
        )
        self.reference = load_reference_config(
            ROOT / "configs/vlsa_distal_region_aware_mlp_moka10.v1.json"
        )

    def test_model_loss_uncertainty_and_gates_are_unchanged(self):
        require_unchanged_model_sections(self.config, self.reference)
        self.assertEqual(
            self.config["split"]["expected_state_counts"],
            {"train": 60, "validation": 10, "test": 15},
        )
        self.assertEqual(
            self.config["split"]["new_validation_state_indexes"],
            [60, 61, 62, 63, 64],
        )

    def test_validation_fresh_contract_is_validation_only(self):
        settings = self.config["validation_fresh_actions"]
        self.assertEqual(settings["count_per_state"], 96)
        self.assertEqual(settings["seed"], 20260810)
        self.assertEqual(
            settings["purpose"], "validation_only_one_sided_calibration"
        )
        self.assertTrue(
            self.config["split"][
                "held_out_test_never_used_for_training_calibration_or_early_stopping"
            ]
        )

    def test_validation_fresh_payload_excludes_its_own_hash(self):
        value = {"schema_version": "x", "validation_fresh_payload_sha256": "bad"}
        self.assertEqual(
            fresh_payload(value), fresh_payload({"schema_version": "x"})
        )

    def test_closed_loop_is_not_run_in_this_gate(self):
        self.assertFalse(self.config["decision"]["closed_loop_in_this_gate"])
        self.assertEqual(
            self.config["decision"]["if_supported_MLP_fails"],
            "replace_coefficient_output_with_action_conditioned_conservative_safety_value_model",
        )


if __name__ == "__main__":
    unittest.main()
