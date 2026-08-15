import unittest
from pathlib import Path

from main.multilink_ellipsoid.empirical_candidate_governor import classify, load_config


ROOT = Path(__file__).resolve().parents[1]


class EmpiricalCandidateGovernorTest(unittest.TestCase):
    def test_config(self):
        config = load_config(
            ROOT / "configs/vlsa_distal_empirical_candidate_governor.v1.json"
        )
        self.assertEqual(len(config["cases"]), 4)

    def test_classification_selects_smaller_safe_candidate(self):
        def candidate(name, effective, status, contact, overlap):
            return {
                "name": name,
                "requested_alpha": float(name.split("_")[-1]) if name != "nominal" else 0.0,
                "source_effective_post_AEGIS_correction_l2_action": effective,
                "source_terminal_status": status,
                "source_physical_veto": bool(contact),
                "source_raw_protected_contact_count": int(contact),
                "compiled_box_any_exact_overlap": bool(overlap),
                "compiled_box_safe_terminal": status == "SAFE_TERMINAL" and not overlap,
            }

        cases = []
        for index in range(4):
            cases.append(
                {
                    "case_id": "c%d" % index,
                    "state_step": index,
                    "source_replay_exact": True,
                    "initial_compiled_box_any_exact_overlap": False,
                    "initial_raw_protected_contact_count": 0,
                    "candidates": [
                        candidate("normal_alpha_1.00", 1.0, "UNSAFE_CONTACT_OR_CAR", 1, 1),
                        candidate("normal_alpha_1.50", 1.5, "SAFE_TERMINAL", 0, 0),
                        candidate("normal_alpha_2.00", 2.0, "SAFE_TERMINAL", 0, 0),
                    ],
                }
            )
        result = classify(
            cases,
            {"require_case_count": 4, "require_candidate_count": 12},
        )
        self.assertTrue(result["strict_gate_pass"])
        self.assertTrue(
            all(item["selected"]["name"] == "normal_alpha_1.50" for item in result["selections"])
        )


if __name__ == "__main__":
    unittest.main()
