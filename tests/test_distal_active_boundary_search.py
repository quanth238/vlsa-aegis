import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from main.multilink_ellipsoid.active_boundary_search import (
    aggregate_search_rows, candidate_definitions, load_config,
    summarize_target_rows, target_job,
)


class ActiveBoundarySearchTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(
            "configs/vlsa_distal_active_boundary_search.v1.json"
        ))

    def test_config_and_job_binding(self):
        self.assertEqual(target_job(self.config, 0)["case_index"], 10)
        self.assertEqual(target_job(self.config, 3)["case_index"], 14)
        with self.assertRaises(ValueError):
            target_job(self.config, 4)

    def test_direction_grid_is_complete_and_preserves_nontranslation(self):
        nominal = np.linspace(-0.4, 0.4, 35).reshape(5, 7)
        frame = {
            "normal": [1.0, 0.0, 0.0],
            "tangent_up": [0.0, 1.0, 0.0],
            "tangent_side": [0.0, 0.0, 1.0],
        }
        definitions = candidate_definitions(
            nominal, frame, self.config, "constant"
        )
        self.assertEqual(len(definitions), 27)
        coefficients = {
            tuple(item["spatial_coefficients"])
            for item in definitions[1:]
        }
        self.assertEqual(len(coefficients), 26)
        self.assertNotIn((0, 0, 0), coefficients)
        for item in definitions:
            actions = np.asarray(item["actions"])
            np.testing.assert_array_equal(actions[:, 3:], nominal[:, 3:])
            self.assertLessEqual(np.max(np.abs(actions)), 1.0)

    def test_target_summary_excludes_unknown(self):
        def record(name, status, risks, safe=False):
            return {
                "name": name, "terminal_status": status,
                "combined_risk": risks, "exact_safe": safe,
            }
        candidates = [
            record("safe", "SAFE_TERMINAL", [0, 0, 0, 0, -0.002, -0.003, -0.004], True),
            record("unsafe", "UNSAFE_CONTACT_OR_CAR", [0, 0, 0, 0, 0.002, 0.003, 0.004]),
            record("unknown", "UNKNOWN_TIMEOUT", [0, 0, 0, 0, 1.0, 1.0, 1.0]),
        ]
        summary = summarize_target_rows(candidates, [4, 5, 6], 0.001)
        self.assertTrue(all(item["known_candidate_count"] == 2 for item in summary))
        self.assertTrue(all(item["useful_boundary_observed_within_job"] for item in summary))

    def test_aggregate_reports_witnesses_not_continuous_extrema(self):
        def result(index, state, profile, maximum, useful):
            rows = []
            for row in (4, 5, 6):
                rows.append({
                    "row": row,
                    "witnessed_maximum_Q_m": maximum if row == 5 else -0.01,
                    "maximum_candidate_name": "candidate-%d" % index,
                    "robust_positive_side_observed": row == 5 and maximum >= 0.001,
                    "useful_boundary_observed_combined": row == 5 and useful,
                })
            return {
                "active_boundary_job": {
                    "state_id": state, "split": "train",
                    "temporal_profile": profile,
                },
                "target_row_summary": rows,
            }
        aggregate = aggregate_search_rows([
            result(0, "state-a", "constant", 0.002, True),
            result(1, "state-a", "front_loaded", 0.003, True),
            result(2, "state-b", "constant", -0.001, False),
        ], [4, 5, 6])
        row5 = aggregate[1]
        self.assertEqual(row5["useful_boundary_state_ids"], ["state-a"])
        self.assertEqual(row5["witnessed_maximum_Q_m"], 0.003)
        self.assertEqual(row5["maximum_witness"]["temporal_profile"], "front_loaded")


if __name__ == "__main__":
    unittest.main()
