from pathlib import Path
import unittest

from main.multilink_ellipsoid.grouped_query_action_risk import load_config
from main.multilink_ellipsoid.grouped_query_action_risk_freeze import (
    candidate_outcome, load_config as load_freeze, state_classification,
)
from main.multilink_ellipsoid.query_action_risk import load_config as load_base


ROOT = Path(__file__).resolve().parents[1]


class GroupedQueryActionRiskTests(unittest.TestCase):
    def test_grouped_protocol_reuses_frozen_method(self):
        config = load_config(
            ROOT / "configs/vlsa_distal_grouped_query_action_risk.v1.json"
        )
        base = load_base(ROOT / config["method"]["base_config"])
        self.assertEqual(
            base["config_payload_sha256"],
            config["method"]["base_config_payload_sha256"],
        )
        self.assertEqual(config["method"]["candidate_count_per_state"], 37)
        self.assertTrue(config["forbidden"]["model_training"])

    @staticmethod
    def candidate(status, risk, *, physical=False, exact=False):
        return {
            "terminal_status": status,
            "combined_risk": risk,
            "physical_veto": physical,
            "exact_safe": exact,
        }

    def test_freeze_excludes_unknown_and_classifies_proxy_invalid_first(self):
        config = load_freeze(
            ROOT / "configs/vlsa_distal_grouped_query_action_risk_freeze.v1.json"
        )
        self.assertEqual(config["adequacy"]["claimed_rows"], list(range(7)))
        safe = self.candidate("SAFE_TERMINAL", [-0.1] * 7, exact=True)
        unsafe = self.candidate("UNSAFE_CONTACT_OR_CAR", [0.1] + [-0.1] * 6)
        unknown = self.candidate("UNKNOWN_TIMEOUT", [0.1] + [-0.1] * 6)
        proxy_invalid = self.candidate(
            "UNSAFE_CONTACT_OR_CAR", [-0.1] * 7, physical=True
        )
        self.assertEqual(candidate_outcome(safe), "safe")
        self.assertEqual(candidate_outcome(unsafe), "unsafe")
        self.assertEqual(candidate_outcome(unknown), "unknown")
        report = state_classification([safe, unsafe, unknown, proxy_invalid])
        self.assertEqual(report["primary_category"], "proxy_invalid")
        self.assertEqual(report["known_safe_candidate_count"], 1)
        self.assertEqual(report["known_unsafe_candidate_count"], 2)
        self.assertEqual(report["unknown_timeout_candidate_count"], 1)

    def test_freeze_distinguishes_no_safe_and_no_known_unsafe(self):
        unsafe = self.candidate("UNSAFE_CONTACT_OR_CAR", [0.1] + [-0.1] * 6)
        safe = self.candidate("SAFE_TERMINAL", [-0.1] * 7, exact=True)
        unknown = self.candidate("UNKNOWN_TIMEOUT", [-0.1] * 7)
        self.assertEqual(
            state_classification([unsafe, unknown])["primary_category"],
            "no_safe_candidate",
        )
        self.assertEqual(
            state_classification([safe, unknown])["primary_category"],
            "no_known_unsafe_candidate",
        )
        self.assertEqual(
            state_classification([safe, unsafe])["primary_category"],
            "usable_mixed_support",
        )


if __name__ == "__main__":
    unittest.main()
