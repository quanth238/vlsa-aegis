from __future__ import annotations

from pathlib import Path
import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from main.multilink_ellipsoid.factorized_input_representation_audit import (
    audit_decision, feature_classes, load_audit_config, semantic_group,
    terminal_rmse,
)


class FactorizedInputRepresentationAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_audit_config(Path(
            "configs/vlsa_distal_factorized_input_representation_audit_moka10.v1.json"
        ))

    def test_frozen_protocol(self) -> None:
        self.assertEqual(self.config["population"]["model_input_dimension"], 2124)
        self.assertEqual(
            self.config["normalization_audit"]["explosive_absolute_z_threshold"],
            100.0,
        )
        self.assertTrue(all(self.config["forbidden_actions"].values()))

    def test_semantic_groups_keep_action_families_separate(self) -> None:
        self.assertEqual(
            semantic_group("action[0].rx"), "action[0].rotation"
        )
        self.assertEqual(
            semantic_group("action[1].z"), "action[1].translation"
        )
        self.assertEqual(
            semantic_group("complete_snapshot.auxiliary.ctrl[4].value"),
            "complete_snapshot.auxiliary.ctrl[]",
        )

    @unittest.skipIf(np is None, "numpy is unavailable")
    def test_feature_classes_distinguish_padding_and_constant(self) -> None:
        names = [
            "field.value", "field.present", "constant", "binary", "physical",
        ]
        features = np.asarray([
            [0.0, 0.0, 3.0, 0.0, 0.0],
            [2.0, 1.0, 3.0, 1.0, 1.0],
            [3.0, 1.0, 3.0, 0.0, 2.0],
        ])
        classes = feature_classes(
            names, features, np.ones(3, dtype=bool), 1.0e-6
        )
        self.assertEqual(classes, [
            "padded_value", "presence", "train_constant",
            "binary_or_categorical", "continuous_physical",
        ])

    def test_decision_requires_both_models_to_show_causal_reduction(self) -> None:
        masking = {"group_masks": [{
            "semantic_group": "complete_snapshot.padding",
            "strong_causal_reduction_both_models": True,
            "artifact_class": True,
            "maximum_validation_absolute_z": 1000.0,
            "maximum_feature_class": "padded_value",
        }]}
        decision = audit_decision(
            normalization={
                "maximum_absolute_z_by_split": {"validation": 1000.0}
            },
            activation={"first_exploding_module": {"flat": {}, "time": {}}},
            masking=masking, config=self.config,
        )
        self.assertEqual(decision["root_cause_classification"], "representation_artifact")
        self.assertFalse(decision["retraining_or_architecture_change_authorized"])

    @unittest.skipIf(np is None, "numpy is unavailable")
    def test_terminal_rmse_uses_only_last_substep(self) -> None:
        exact = np.zeros((2, 51, 7))
        predicted = exact.copy()
        predicted[:, 0] = 100.0
        predicted[:, -1] = 2.0
        self.assertEqual(terminal_rmse(predicted, exact), 2.0)


if __name__ == "__main__":
    unittest.main()
