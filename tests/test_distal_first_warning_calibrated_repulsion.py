import json
from pathlib import Path
import unittest

from main.multilink_ellipsoid.first_warning_calibrated_repulsion import (
    aggregate, load_config,
)


ROOT = Path(__file__).resolve().parents[1]


class FirstWarningCalibratedRepulsionTest(unittest.TestCase):
    def test_registered_config_and_aggregate(self):
        config = load_config(
            ROOT / "configs/vlsa_distal_first_warning_calibrated_repulsion.v1.json"
        )
        rows = []
        for case_id, radius in zip(config["case_ids"], config["requested_radius_by_case"]):
            rows.append({
                "case_id": case_id,
                "raw_L5_L7_contact_pass": True,
                "paper_car_pass": True,
                "native_task_success": True,
                "timeout": False,
                "intervention_count": 2,
                "interventions": [
                    {"proposal": {"requested_correction_l2_action": radius}},
                    {"proposal": {"requested_correction_l2_action": radius}},
                ],
            })
        result = aggregate(rows, config)
        self.assertTrue(result["strict_gate_pass"])
        self.assertEqual(result["strictly_smaller_requested_correction_count"], 3)


if __name__ == "__main__":
    unittest.main()
