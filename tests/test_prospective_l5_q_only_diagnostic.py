from __future__ import annotations

from pathlib import Path
import unittest

from main.multilink_ellipsoid.prospective_l5_q_only_diagnostic import (
    classify_diagnostic, load_config,
)


def _metrics(*, false_safes=0, recall=1.0, support=2, near=0.05):
    return {
        "false_safe_count": false_safes,
        "safe_recall": recall,
        "supported_state_count": support,
        "near_boundary_global_RMSE": near,
    }


class ProspectiveL5QOnlyDiagnosticTest(unittest.TestCase):
    def test_config_freezes_split_and_model(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "configs/vlsa_distal_prospective_l5_q_only_diagnostic.v1.json")
        self.assertEqual([len(config["dataset"]["split_case_ids"][name]) for name in ("train", "validation", "test")], [6, 2, 2])
        self.assertEqual(config["features"]["input_dimension"], 9)
        self.assertEqual(config["model"]["hidden_widths"], [32, 32])
        self.assertEqual(config["model"]["checkpoint"], "fixed_final_epoch_no_validation_model_selection")
        self.assertIn("V_targets_or_loss", config["forbidden"])
        self.assertIn("QP_or_gradient_correction", config["forbidden"])

    def test_diagnostic_gate_requires_both_splits(self) -> None:
        root = Path(__file__).resolve().parents[1]
        decision = load_config(root / "configs/vlsa_distal_prospective_l5_q_only_diagnostic.v1.json")["decision"]
        verdict, checks = classify_diagnostic(_metrics(), _metrics(), decision)
        self.assertEqual(verdict, "diagnostic_transfer_signal_pass")
        self.assertTrue(all(checks.values()))
        verdict, checks = classify_diagnostic(_metrics(), _metrics(false_safes=1), decision)
        self.assertEqual(verdict, "diagnostic_transfer_signal_no_go")
        self.assertFalse(checks["test_zero_false_safes"])


if __name__ == "__main__":
    unittest.main()
