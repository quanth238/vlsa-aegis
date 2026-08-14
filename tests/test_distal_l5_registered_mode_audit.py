import unittest
from pathlib import Path

from main.multilink_ellipsoid.l5_registered_mode_audit import (
    audit_states, load_config,
)


class RegisteredModeAuditTest(unittest.TestCase):
    def test_registered_config(self):
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs/vlsa_distal_l5_registered_mode_audit.v1.json"
        )
        self.assertEqual(
            config["comparisons"]["strongest_analytical_repulsion"],
            "normal_pos_front_loaded_r2.0",
        )
        self.assertFalse(config["decision"]["learned_selector_authorized"])

    def test_tangent_rescue_is_detected(self):
        config = load_config(
            Path(__file__).resolve().parents[1]
            / "configs/vlsa_distal_l5_registered_mode_audit.v1.json"
        )
        names = ["nominal"] + [item["name"] for item in __import__("json").loads(
            (Path(__file__).resolve().parents[1]
             / "configs/vlsa_distal_adaptive_query_action_risk.v3.json").read_text()
        )["screening"]["candidate_recipes"] if item["name"] != "nominal"]
        modes = []
        for order, name in enumerate(names):
            risk = 0.01
            if name == "tangent_up_pos_front_loaded_r1.0":
                risk = -0.002
            modes.append({
                "name": name, "order": order, "risk_m": risk,
                "physical_veto": False, "zero_margin_safe": risk <= 0.0,
                "screen_safe": risk < -0.0005, "clipped": False,
                "applied_norm": 0.0 if name == "nominal" else 1.0,
            })
        report = audit_states([{
            "state_id": "s", "case_id": "c", "split": "train",
            "target_row": 0, "modes": modes,
        }], config)
        self.assertTrue(report["state_dependent_prefix_modes_observed"])
        self.assertEqual(report["state_count_matched_tangent_rescues_radius1_normal"], 1)


if __name__ == "__main__":
    unittest.main()
