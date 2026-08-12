import json
from pathlib import Path
import unittest

from main.multilink_ellipsoid.scale_feedback_audit import (
    load_scale_feedback_audit_config,
)


ROOT = Path(__file__).resolve().parents[1]


class ScaleFeedbackAuditTest(unittest.TestCase):
    def test_registered_protocol(self):
        config = load_scale_feedback_audit_config(
            ROOT / "configs/vlsa_distal_scale_feedback_audit_e05.v1.json"
        )
        self.assertEqual(config["state_protocol"]["prelude_last_step"], 205)
        self.assertEqual(config["state_protocol"]["live_query_start_step"], 206)
        self.assertEqual(config["state_protocol"]["live_query_first_index"], 41)
        self.assertEqual(config["state_protocol"]["live_query_action_horizon"], 5)
        self.assertEqual(
            config["internal_verification"][
                "live_execution_clearance_equivalence_tolerance"
            ],
            1e-10,
        )
        self.assertEqual(
            sorted(item["correction_scale"] for item in config["registered_inputs"].values()),
            [1.01, 1.02],
        )

    def test_rejects_shifted_feedback_start(self):
        source = ROOT / "configs/vlsa_distal_scale_feedback_audit_e05.v1.json"
        value = json.loads(source.read_text())
        value["state_protocol"]["live_query_start_step"] = 207
        temporary = ROOT / "tests" / ".temporary-scale-feedback-audit.json"
        try:
            temporary.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load_scale_feedback_audit_config(temporary)
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
