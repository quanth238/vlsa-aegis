import copy
import unittest

from main.multilink_ellipsoid.candidate_parallel import (
    merge_candidate_curve_shards,
)


def _candidate(name, safe, risk):
    return {
        "name": name,
        "exact_safe": safe,
        "terminal_status": "SAFE_TERMINAL" if safe else "UNSAFE_CONTACT_OR_CAR",
        "combined_risk": [risk] * 7,
        "combined_row_minimum_clearance_m": [-risk] * 7,
        "physical_veto": not safe,
        "source_restore_maximum_error": 0.0,
        "source_snapshot_sha256": "state",
        "prefix": {"maximum_boundary_equivalence_error_m": 0.0},
    }


def _shard(target=None):
    rows = [_candidate("nominal", False, 0.2)]
    if target is not None:
        rows.append(_candidate(target, True, -0.1))
    return {
        "status": "complete",
        "base_method_config": {"risk_target": {"terminal_statuses": [
            "SAFE_TERMINAL", "UNSAFE_CONTACT_OR_CAR", "UNKNOWN_TIMEOUT",
        ]}},
        "state": {"source_snapshot_sha256": "state"},
        "candidate_count": len(rows),
        "candidates": rows,
        "summary": {},
        "gates": {
            "exact_snapshot_replay": True,
            "zero_L5_residual_reproduces_recomputed_released_aegis": True,
            "query_boundary_is_initially_safe": True,
            "nominal_future_is_unsafe": True,
            "safe_and_unsafe_candidate_support": target is not None,
            "at_least_one_safe_terminal": target is not None,
            "no_timeout_labeled_safe": True,
            "proxy_safe_physical_collision_count_zero": False,
            "all_terminal_statuses_registered": True,
            "all_replays_boundary_exact": True,
        },
        "interpretation": "shard",
        "wall_seconds": 1.0,
        "result_payload_sha256": "ignored",
    }


class CandidateParallelMergeTest(unittest.TestCase):
    def test_merges_in_frozen_order_and_recomputes_support(self):
        merged = merge_candidate_curve_shards(
            [_shard(), _shard("plus"), _shard("minus")],
            ["nominal", "plus", "minus"], worker_count=4, wall_seconds=2.5,
        )
        self.assertEqual(
            [row["name"] for row in merged["candidates"]],
            ["nominal", "plus", "minus"],
        )
        self.assertEqual(merged["summary"]["safe_candidate_count"], 2)
        self.assertEqual(merged["summary"]["unsafe_candidate_count"], 1)
        self.assertTrue(merged["gates"]["safe_and_unsafe_candidate_support"])
        self.assertEqual(merged["parallel_candidate_execution"]["worker_count"], 4)

    def test_rejects_context_mismatch(self):
        bad = _shard("plus")
        bad["state"] = {"source_snapshot_sha256": "different"}
        with self.assertRaisesRegex(ValueError, "context differs"):
            merge_candidate_curve_shards(
                [_shard(), bad], ["nominal", "plus"],
                worker_count=4, wall_seconds=2.5,
            )

    def test_rejects_nominal_mismatch(self):
        bad = _shard("plus")
        bad["candidates"][0] = copy.deepcopy(bad["candidates"][0])
        bad["candidates"][0]["combined_risk"] = [0.3] * 7
        with self.assertRaisesRegex(ValueError, "nominal candidate differs"):
            merge_candidate_curve_shards(
                [_shard(), bad], ["nominal", "plus"],
                worker_count=4, wall_seconds=2.5,
            )


if __name__ == "__main__":
    unittest.main()
