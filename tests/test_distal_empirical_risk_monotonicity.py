import unittest
from pathlib import Path

from main.multilink_ellipsoid.empirical_risk_monotonicity import (
    case_report,
    classify,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


def source_candidate(index):
    profile = [5 / 55 ** 0.5, 4 / 55 ** 0.5, 3 / 55 ** 0.5,
               2 / 55 ** 0.5, 1 / 55 ** 0.5]
    return {
        "name": "c%d" % index,
        "direction": None if index == 0 else [1.0, 0.0, 0.0],
        "temporal_profile": None if index == 0 else profile,
        "clipped": False,
    }


def actions(alpha):
    profile = [5 / 55 ** 0.5, 4 / 55 ** 0.5, 3 / 55 ** 0.5,
               2 / 55 ** 0.5, 1 / 55 ** 0.5]
    return [[alpha * profile[index], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for index in range(5)]


def result(case_id, split, risks):
    candidates = []
    for index, risk in enumerate(risks):
        candidates.append({
            "name": "c%d" % index,
            "requested_alpha": float(index),
            "source_executed_actions": actions(float(index)),
            "source_terminal_status": "SAFE_TERMINAL",
            "compiled_box_risk": float(risk),
        })
    return {
        "case_id": case_id,
        "split": split,
        "state_step": 25,
        "empirical_case": {"candidates": candidates},
    }


class EmpiricalRiskMonotonicityTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(
            ROOT / "configs/vlsa_distal_empirical_risk_monotonicity.v1.json"
        )

    def test_realized_path_and_monotonic_classification(self):
        source = {"candidates": [source_candidate(index) for index in range(3)]}
        reports = []
        for index in range(13):
            split = "train" if index < 10 else "validation"
            reports.append(case_report(
                result("case-%02d" % index, split, [2.0, 1.0, -1.0]),
                source,
                self.config,
            ))
        report = classify(reports, self.config, dataset_strict_gate_pass=True)
        self.assertTrue(report["monotonic_regularization_authorized"])
        self.assertEqual(report["splits"]["train"]["eligible_pair_count"], 20)
        self.assertAlmostEqual(
            reports[0]["candidate_records"][2]["effective_outward_alpha"], 2.0
        )

    def test_risk_reversal_blocks_regularization(self):
        source = {"candidates": [source_candidate(index) for index in range(3)]}
        reports = []
        for index in range(13):
            split = "train" if index < 10 else "validation"
            risks = [2.0, 3.0, -1.0] if index == 0 else [2.0, 1.0, -1.0]
            reports.append(case_report(
                result("case-%02d" % index, split, risks), source, self.config
            ))
        report = classify(reports, self.config, dataset_strict_gate_pass=True)
        self.assertFalse(report["monotonic_regularization_authorized"])
        self.assertEqual(report["splits"]["train"]["monotonicity_violation_count"], 1)


if __name__ == "__main__":
    unittest.main()
