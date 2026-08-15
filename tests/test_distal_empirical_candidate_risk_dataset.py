import unittest
from pathlib import Path

from main.multilink_ellipsoid.empirical_candidate_risk_dataset import (
    classify,
    load_cases,
    load_config,
    warning_step,
)


ROOT = Path(__file__).resolve().parents[1]


class EmpiricalCandidateRiskDatasetTest(unittest.TestCase):
    def test_config_and_grouped_cases(self):
        config = load_config(
            ROOT / "configs/vlsa_distal_empirical_candidate_risk_dataset.v1.json"
        )
        cases = load_cases(ROOT / config["selection_manifest"], config)
        self.assertEqual(len(cases), 13)
        self.assertEqual(sum(row["split"] == "train" for row in cases), 10)
        self.assertEqual(sum(row["split"] == "validation" for row in cases), 3)
        self.assertTrue(all(warning_step(row, config) % 5 == 0 for row in cases))

    def test_strict_dataset_classification(self):
        config = load_config(
            ROOT / "configs/vlsa_distal_empirical_candidate_risk_dataset.v1.json"
        )

        def candidate(index):
            contact = index == 0
            return {
                "name": "c%d" % index,
                "source_terminal_status": (
                    "UNSAFE_CONTACT_OR_CAR" if contact else "SAFE_TERMINAL"
                ),
                "source_physical_veto": contact,
                "source_raw_protected_contact_count": int(contact),
                "compiled_box_any_exact_overlap": contact,
                "compiled_box_safe_terminal": not contact,
                "source_effective_post_AEGIS_correction_l2_action": float(index),
                "requested_alpha": float(index),
            }

        cases = []
        for index in range(13):
            cases.append({
                "case_id": "case-%02d" % index,
                "split": "train" if index < 10 else "validation",
                "state_step": index * 5 + 5,
                "empirical_case": {
                    "source_replay_exact": True,
                    "initial_compiled_box_any_exact_overlap": False,
                    "initial_raw_protected_contact_count": 0,
                    "candidates": [candidate(item) for item in range(9)],
                },
            })
        report = classify(cases, config)
        self.assertTrue(report["strict_gate_pass"])
        self.assertTrue(report["training_authorized"])
        self.assertEqual(report["counts"]["candidates"], 117)


if __name__ == "__main__":
    unittest.main()
