from pathlib import Path
import unittest

from main.multilink_ellipsoid.active_boundary_audit import (
    audit_state_row, load_config, row_interpretation,
)


ROOT = Path(__file__).resolve().parents[1]


class ActiveBoundaryAuditTests(unittest.TestCase):
    @staticmethod
    def candidate(name, risk, *, status="SAFE_TERMINAL", exact=False, physical=False):
        return {
            "name": name,
            "terminal_status": status,
            "combined_risk": risk,
            "exact_safe": exact,
            "physical_veto": physical,
        }

    def test_config_preserves_no_learning_gate(self):
        config = load_config(
            ROOT / "configs/vlsa_distal_active_boundary_audit.v1.json"
        )
        self.assertIn("model_training", config["forbidden"])
        self.assertEqual(config["source"]["primary_splits"], ["train", "validation"])

    def test_useful_independent_boundary(self):
        safe = self.candidate("safe", [-0.003] * 7, exact=True)
        unsafe = self.candidate(
            "unsafe", [-0.002, 0.004] + [-0.002] * 5,
            status="UNSAFE_CONTACT_OR_CAR",
        )
        result = audit_state_row(
            [safe, unsafe], 1, robust_margin_m=0.001,
            minimum_variation_m=0.002,
        )
        self.assertTrue(result["meaningful_variation"])
        self.assertTrue(result["robust_crossing_observed"])
        self.assertTrue(result["useful_boundary_observed"])
        self.assertTrue(result["independent_violation_observed"])

    def test_crossing_is_not_useful_without_global_safe_side(self):
        negative_but_other_row_unsafe = self.candidate(
            "negative", [0.004, -0.003] + [-0.002] * 5,
            status="UNSAFE_CONTACT_OR_CAR",
        )
        positive = self.candidate(
            "positive", [-0.002, 0.003] + [-0.002] * 5,
            status="UNSAFE_CONTACT_OR_CAR",
        )
        result = audit_state_row(
            [negative_but_other_row_unsafe, positive], 1,
            robust_margin_m=0.001, minimum_variation_m=0.002,
        )
        self.assertTrue(result["robust_crossing_observed"])
        self.assertFalse(result["useful_boundary_observed"])
        self.assertEqual(
            row_interpretation([result]),
            "ROW_CROSSING_WITHOUT_GLOBAL_SAFE_SIDE_WITNESSED",
        )

    def test_unknown_does_not_enter_extrema(self):
        known = self.candidate("known", [-0.002] * 7, exact=True)
        unknown = self.candidate(
            "unknown", [1.0] * 7, status="UNKNOWN_TIMEOUT"
        )
        result = audit_state_row(
            [known, unknown], 0, robust_margin_m=0.001,
            minimum_variation_m=0.002,
        )
        self.assertEqual(result["known_candidate_count"], 1)
        self.assertEqual(result["unknown_timeout_candidate_count"], 1)
        self.assertEqual(result["witnessed_maximum_Q_m"], -0.002)


if __name__ == "__main__":
    unittest.main()
