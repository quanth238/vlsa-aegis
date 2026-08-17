import copy
import unittest

from main.multilink_ellipsoid.candidate_worker_equivalence import compare


def _source(parallel=False):
    row = {
        "name": "nominal",
        "post_aegis_candidate_before_consistency": [[0.0] * 7],
        "proposed_actions": [[0.0] * 7],
        "actions": [[0.0] * 7],
        "source_snapshot_sha256": "source",
        "source_restore_maximum_error": 0.0,
        "terminal_state_sha256": "terminal",
        "terminal_status": "SAFE_TERMINAL",
        "terminal_reason": "safe",
        "prefix": {"maximum_active_obstacle_l1_displacement_m": 0.0},
        "backup": {"maximum_active_obstacle_l1_displacement_m": 0.0},
        "physical_veto": False,
    }
    value = {
        "state": {"source_snapshot_sha256": "source"},
        "determinism_replay": {"contacts_identical": True},
        "candidates": [row],
        "wall_seconds": 4.0 if not parallel else 2.0,
    }
    if parallel:
        value["parallel_candidate_execution"] = {
            "enabled": True, "worker_count": 4,
        }
    return value


def _case():
    return {
        "status": "complete",
        "case_id": "case",
        "state_step": 5,
        "selection": {"target_group": "L6"},
        "exact_case": {
            "replayed_snapshot_sha256": "source",
            "source_replay_exact": True,
            "candidates": [{
                "name": "nominal",
                "raw_protected_contacts": [],
                "exact_group_target": {
                    "known_outcome": True,
                    "group_future_violation": {"palm": -1.0, "L5": -1.0, "L6": -1.0},
                    "group_contact_events": {"palm": [], "L5": [], "L6": []},
                },
            }],
        },
    }


class CandidateWorkerEquivalenceTest(unittest.TestCase):
    def test_accepts_equal_scientific_views(self):
        result = compare(_case(), _source(), _case(), _source(parallel=True))
        self.assertTrue(result["strict_equivalence_pass"])
        self.assertEqual(result["observed_speedup"], 2.0)

    def test_rejects_risk_difference(self):
        parallel_case = copy.deepcopy(_case())
        parallel_case["exact_case"]["candidates"][0]["exact_group_target"][
            "group_future_violation"
        ]["L6"] = 0.1
        result = compare(
            _case(), _source(), parallel_case, _source(parallel=True),
        )
        self.assertFalse(result["checks"]["per_constraint_risks_equal"])
        self.assertFalse(result["strict_equivalence_pass"])


if __name__ == "__main__":
    unittest.main()
