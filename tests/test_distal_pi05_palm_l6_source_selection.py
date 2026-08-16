import json
import os
import tempfile
import unittest
from pathlib import Path

from main.multilink_ellipsoid.pi05_palm_l6_source_selection import (
    _raw_action_contract, _warning_step, excluded_case_ids, load_config,
    cpu_allocation_record, scientific_view,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_pi05_palm_l6_source_audit.v1.json"


class Pi05PalmL6SourceSelectionTest(unittest.TestCase):
    def test_config_freezes_raw_pi05_no_qp_and_no_candidate_outcomes(self):
        config = load_config(CONFIG, repo_root=ROOT)
        self.assertEqual(config["source"]["arm"], "pi05_translational")
        self.assertEqual(
            config["source"]["nominal_action_source"],
            "raw_pi05_nominal_translational",
        )
        self.assertFalse(config["source"]["released_AEGIS_EE_QP_enabled"])
        self.assertEqual(config["target_groups"], ["palm", "L6"])
        self.assertFalse(config["warning_state_rule"]["candidate_outcomes_accessed"])
        self.assertTrue(config["warning_state_rule"]["one_state_per_root_episode"])

    def test_warning_step_is_latest_complete_prefix_before_contact(self):
        self.assertEqual(_warning_step(27), 20)
        self.assertEqual(_warning_step(30), 25)
        self.assertEqual(_warning_step(10), 5)

    def test_raw_action_contract_rejects_postprocessed_action(self):
        rows = []
        for step in range(10):
            action = [float(step)] * 7
            rows.append({
                "step": step,
                "control_path": "pi05_translational_nominal",
                "qp": None,
                "modified": False,
                "correction_l2": 0.0,
                "executed": action,
                "nominal_translational": action,
                "env_step_input": action,
            })
        result = {"actions": rows, "policy_queries": [{"query_index": 0}, {"query_index": 1}]}
        self.assertTrue(_raw_action_contract(result, 5)["raw_pi05_action_invariant"])
        rows[7]["correction_l2"] = 0.1
        self.assertFalse(_raw_action_contract(result, 5)["raw_pi05_action_invariant"])

    def test_previous_candidate_cases_are_excluded(self):
        config = load_config(CONFIG, repo_root=ROOT)
        identities = excluded_case_ids(ROOT, config)
        self.assertIn("vlsa-t1-spatial-i-t3-e09", identities)
        self.assertGreater(len(identities), 10)

    def test_scientific_view_ignores_allocation_identity(self):
        common = {
            "schema_version": "schema", "status": "complete",
            "scientific_result": True, "claim_scope": "scope",
            "config": {"config_payload_sha256": "config"},
            "summary": {}, "excluded_case_count": 1, "records": [],
            "new_simulation_performed": False,
            "candidate_outcomes_accessed": False,
            "boundary_collection_authorized": False,
            "training_authorized": False, "correction_authorized": False,
        }
        left = {**common, "allocation": {"job": 1}}
        right = {**common, "allocation": {"job": 2}}
        self.assertEqual(scientific_view(left), scientific_view(right))

    def test_cpu_allocation_receipt_rejects_non_slurm(self):
        old = os.environ.pop("SLURM_JOB_ID", None)
        try:
            with self.assertRaisesRegex(ValueError, "requires a Slurm allocation"):
                cpu_allocation_record()
        finally:
            if old is not None:
                os.environ["SLURM_JOB_ID"] = old


if __name__ == "__main__":
    unittest.main()
