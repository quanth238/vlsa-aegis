import unittest

from main.multilink_ellipsoid.l5_aegis_grouped_summary import (
    candidate_outcome, row_coverage, state_classification,
)


def candidate(status, risks, *, exact=False, physical=False):
    return {
        "terminal_status": status,
        "combined_risk": risks,
        "exact_safe": exact,
        "physical_veto": physical,
    }


class L5AegisGroupedSummaryTests(unittest.TestCase):
    def test_timeouts_are_unknown_and_retained(self):
        value = candidate("UNKNOWN_TIMEOUT", [0.1] * 7)
        self.assertEqual(candidate_outcome(value), "unknown")
        self.assertEqual(state_classification([value]), "unknown_only")
        self.assertEqual(row_coverage([value])[0]["unknown_timeout_candidate_count"], 1)

    def test_support_categories_do_not_drop_failures(self):
        safe = candidate("SAFE_TERMINAL", [-0.1] * 7, exact=True)
        unsafe = candidate("UNSAFE_CONTACT_OR_CAR", [0.1] + [-0.1] * 6)
        self.assertEqual(state_classification([safe, unsafe]), "usable_mixed_support")
        self.assertEqual(state_classification([unsafe]), "no_safe_candidate")
        self.assertEqual(state_classification([safe]), "no_known_unsafe_candidate")

    def test_proxy_invalid_precedes_mixed_support(self):
        safe = candidate("SAFE_TERMINAL", [-0.1] * 7, exact=True)
        invalid = candidate(
            "UNSAFE_CONTACT_OR_CAR", [-0.1] * 7, physical=True
        )
        self.assertEqual(state_classification([safe, invalid]), "proxy_invalid")

    def test_row_counts_use_known_candidates_only(self):
        values = [
            candidate("SAFE_TERMINAL", [-0.001] * 7, exact=True),
            candidate("UNSAFE_CONTACT_OR_CAR", [0.002] + [-0.02] * 6),
            candidate("UNKNOWN_TIMEOUT", [0.003] * 7),
        ]
        rows = row_coverage(values)
        self.assertEqual(rows[0]["known_safe_candidate_count"], 1)
        self.assertEqual(rows[0]["known_unsafe_candidate_count"], 1)
        self.assertEqual(rows[0]["near_boundary_known_candidate_count"], 2)
        self.assertEqual(rows[0]["active_witness_known_candidate_count"], 2)
        self.assertEqual(rows[0]["unknown_timeout_candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
