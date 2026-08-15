import unittest
from pathlib import Path

from main.multilink_ellipsoid.compiled_box_risk_target_audit import (
    candidate_action_sequence,
    classify,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


class CompiledBoxRiskTargetAuditTest(unittest.TestCase):
    def test_config(self):
        config = load_config(
            ROOT / "configs/vlsa_distal_compiled_box_risk_target_audit.v1.json"
        )
        self.assertEqual(len(config["cases"]), 2)

    def test_candidate_sequence_preserves_phases(self):
        candidate = {
            "actions": [[0.0] * 7, [1.0] * 7],
            "backup": {
                "decisions": [{"selected_action": [2.0] * 7}],
                "terminal_hold": {"executed_actions": [[0.0] * 7, [0.0] * 7]},
            },
        }
        actions, phases = candidate_action_sequence(candidate)
        self.assertEqual(len(actions), 5)
        self.assertEqual(phases, ["prefix", "prefix", "backup", "terminal_hold", "terminal_hold"])

    def test_classification_separates_timeout_and_rescued_safe(self):
        def row(name, status, veto, proxy_safe, overlap, contacts=None):
            return {
                "case_id": name.split("-")[0], "name": name,
                "source_terminal_status": status,
                "source_physical_veto": veto,
                "source_raw_protected_contact_count": (
                    int(veto) if contacts is None else int(contacts)
                ),
                "source_proxy_nonoverlap": proxy_safe,
                "source_proxy_buffer_safe": proxy_safe,
                "compiled_box_any_exact_overlap": overlap,
                "compiled_box_safe_terminal": status == "SAFE_TERMINAL" and not overlap,
            }
        cases = [
            {"source_replay_exact": True,
             "initial_compiled_box_any_exact_overlap": False,
             "initial_raw_protected_contact_count": 0,
             "candidates": [
                row("a-contact", "UNSAFE_CONTACT_OR_CAR", True, False, True),
                row("a-safe", "SAFE_TERMINAL", False, False, False),
            ]},
            {"source_replay_exact": True,
             "initial_compiled_box_any_exact_overlap": False,
             "initial_raw_protected_contact_count": 0,
             "candidates": [
                row("b-contact", "UNSAFE_CONTACT_OR_CAR", True, False, True),
                row("b-car-only", "UNSAFE_CONTACT_OR_CAR", True, False, False, 0),
                row("b-safe", "SAFE_TERMINAL", False, False, False),
                row("b-timeout", "UNKNOWN_TIMEOUT", False, False, False),
            ]},
        ]
        result = classify(cases, {
            "require_case_count": 2, "require_candidate_count": 6,
            "minimum_contact_controls": 2, "minimum_safe_terminal_controls": 2,
            "minimum_proxy_rejected_safe_rescued": 2,
        })
        self.assertTrue(result["strict_gate_pass"])
        self.assertEqual(result["contact_control_count"], 2)
        self.assertEqual(result["timeout_control_count"], 1)


if __name__ == "__main__":
    unittest.main()
