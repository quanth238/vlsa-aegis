import json
from pathlib import Path
import tempfile
import unittest


class PalmPrimitiveAuditTests(unittest.TestCase):
    def _config(self):
        root = Path(__file__).resolve().parents[1]
        return json.loads(
            (root / "configs/vlsa_distal_palm_primitive_audit.v1.json")
            .read_text(encoding="utf-8")
        )

    def test_frozen_cohort_and_fit_do_not_use_outcomes(self):
        from main.multilink_ellipsoid.palm_primitive_audit import load_config

        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/vlsa_distal_palm_primitive_audit.v1.json",
            repo_root=root,
        )
        self.assertEqual(len(config["cohort"]["contact_case_ids"]), 3)
        self.assertEqual(len(config["cohort"]["control_case_ids"]), 40)
        self.assertFalse(config["cohort"]["outcomes_used_for_primitive_fit"])
        self.assertEqual(
            config["primitive"]["fit_source"],
            "compiled_collision_mesh_vertices_only",
        )

    def test_compiled_obstacle_variant_preserves_frozen_cohort(self):
        from main.multilink_ellipsoid.palm_primitive_audit import load_config

        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/vlsa_distal_palm_primitive_compiled_obstacle_audit.v1.json",
            repo_root=root,
        )
        self.assertEqual(len(config["cohort"]["contact_case_ids"]), 3)
        self.assertEqual(len(config["cohort"]["control_case_ids"]), 40)
        self.assertFalse(config["compiled_obstacle"]["fit_uses_contact_outcomes"])
        self.assertIn("privileged simulation", config["claim_scope"])

    def test_summary_passes_only_with_zero_false_safes(self):
        from main.multilink_ellipsoid.palm_primitive_audit import (
            summarize_case_records,
        )

        config = self._config()
        contacts = set(config["cohort"]["contact_case_ids"])
        records = []
        for case_id in (
            config["cohort"]["contact_case_ids"]
            + config["cohort"]["control_case_ids"]
        ):
            positive = case_id in contacts
            records.append({
                "case_id": case_id,
                "replay": {"fidelity_pass": True},
                "primitive_fit": {"certificate_pass": True},
                "physical_contact": {
                    "internal_palm_contact_sample_count": 1 if positive else 0,
                },
                "tight_primitive": {
                    "physical_false_safe_sample_count": 0,
                    "episode_minimum_support_gap_m": -0.001 if positive else 0.001,
                },
                "released_proxy": {
                    "physical_false_safe_sample_count": 0,
                    "episode_minimum_support_gap_m": -0.001,
                },
            })
        summary = summarize_case_records(records, config)
        self.assertTrue(summary["geometry_gate_pass"])
        self.assertEqual(summary["contact_episodes_reproduced"], 3)
        self.assertEqual(summary["internal_contact_free_control_count"], 40)
        self.assertEqual(summary["tight_contact_free_episode_accept_count"], 40)
        self.assertEqual(summary["released_contact_free_episode_accept_count"], 0)
        records[0]["tight_primitive"]["physical_false_safe_sample_count"] = 1
        self.assertFalse(
            summarize_case_records(records, config)["geometry_gate_pass"]
        )

    def test_config_rejects_overlapping_cohort(self):
        from main.multilink_ellipsoid.palm_primitive_audit import load_config

        config = self._config()
        config["cohort"]["control_case_ids"][0] = config["cohort"][
            "contact_case_ids"
        ][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cohort"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
