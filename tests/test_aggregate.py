from __future__ import annotations

import unittest

from crfs_harness.aggregate import aggregate_results


def result(case_id: str, group: str, oracle: bool, random: bool) -> dict:
    trial = lambda safe: {"safe": safe, "clearance_m": 0.01 if safe else -0.01, "endpoint_error_m": 0.0}
    return {
        "schema_version": "1.0",
        "case_id": case_id,
        "run_id": "run",
        "status": "completed",
        "config_hash": "a" * 64,
        "provenance": {"group_id": group},
        "repair": {"feasible": True},
        "trials": {
            "nominal": trial(False),
            "direct_repair": trial(True),
            "random_residual": trial(random),
            "oracle_residual": trial(oracle),
            "bridge_edit": trial(False),
        },
    }


class AggregateTest(unittest.TestCase):
    def test_feasible_and_overall_populations_are_explicit(self) -> None:
        values = [result("a", "episode-1", True, False), result("b", "episode-2", False, False)]
        values.append(
            {
                "schema_version": "1.0",
                "case_id": "c",
                "run_id": "run",
                "status": "infeasible",
                "config_hash": "a" * 64,
                "provenance": {"group_id": "episode-3"},
                "repair": {"feasible": False},
                "trials": {},
            }
        )
        aggregate = aggregate_results(values, samples=100)
        self.assertEqual(aggregate["counts"]["feasible_colliding"], 2)
        self.assertEqual(aggregate["counts"]["all_colliding_including_infeasible"], 3)
        self.assertAlmostEqual(aggregate["feasible_conditioned"]["oracle_rescue_rate"], 0.5)
        self.assertAlmostEqual(aggregate["overall"]["oracle_rescue_rate_all_colliding"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
