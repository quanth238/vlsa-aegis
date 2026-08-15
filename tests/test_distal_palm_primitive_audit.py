import json
from pathlib import Path
import tempfile
import unittest


class PalmPrimitiveAuditTests(unittest.TestCase):
    def test_diagnostic_perception_rotation_is_canonicalized_without_changing_axes(self):
        import numpy as np

        from scripts.audit_distal_palm_primitive_case import (
            _canonicalize_perception_ellipsoid_rotation,
        )

        reflected = np.diag([1.0, 1.0, -1.0])
        rotation, record = _canonicalize_perception_ellipsoid_rotation(reflected)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-12)
        np.testing.assert_allclose(
            reflected @ reflected.T, rotation @ rotation.T, atol=1.0e-12,
        )
        self.assertTrue(record["canonicalized"])

    def test_diagnostic_perception_rotation_rejects_nonorthogonal_input(self):
        import numpy as np

        from scripts.audit_distal_palm_primitive_case import (
            _canonicalize_perception_ellipsoid_rotation,
        )

        with self.assertRaisesRegex(ValueError, "not an orthogonal basis"):
            _canonicalize_perception_ellipsoid_rotation(np.diag([1.0, 1.0, 0.9]))

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

    def test_tracked_obstacle_variant_changes_only_pose_authority_and_gate(self):
        from main.multilink_ellipsoid.palm_primitive_audit import load_config

        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/vlsa_distal_palm_primitive_tracked_obstacle_audit.v1.json",
            repo_root=root,
        )
        self.assertEqual(len(config["cohort"]["contact_case_ids"]), 3)
        self.assertFalse(
            config["tracked_obstacle"]["shape_or_threshold_tuning_from_contacts"]
        )
        self.assertEqual(config["gate"]["minimum_tight_safe_controls"], 1)
        self.assertTrue(
            config["gate"]["require_tight_control_acceptance_better_than_released"]
        )

    def test_exact_compiled_variant_uses_no_obstacle_ellipsoid(self):
        from main.multilink_ellipsoid.palm_primitive_audit import load_config

        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/vlsa_distal_exact_compiled_geometry_audit.v1.json",
            repo_root=root,
        )
        exact = config["exact_compiled_obstacle"]
        self.assertEqual(
            exact["representation"],
            "exact_bound_constrained_ellipsoid_box_radial_slack",
        )
        self.assertEqual(exact["supported_obstacle_geom_kinds"], ["box"])
        self.assertEqual(exact["units"], "dimensionless_not_metric_clearance")
        self.assertFalse(exact["fit_uses_contact_outcomes"])
        self.assertEqual(len(config["cohort"]["contact_case_ids"]), 31)
        self.assertEqual(len(config["cohort"]["control_case_ids"]), 3)
        self.assertEqual(
            config["comparators"]["primary_obstacle_proxy"],
            "exact_live_compiled_MuJoCo_collision_boxes_no_obstacle_MVEE",
        )

    def test_exact_compiled_summary_requires_each_group(self):
        from main.multilink_ellipsoid.palm_primitive_audit import (
            load_config,
            summarize_case_records,
        )

        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/vlsa_distal_exact_compiled_geometry_audit.v1.json",
            repo_root=root,
        )
        records = []
        contacts = set(config["cohort"]["contact_case_ids"])
        for case_id in (
            config["cohort"]["contact_case_ids"]
            + config["cohort"]["control_case_ids"]
        ):
            positive = case_id in contacts
            records.append({
                "case_id": case_id,
                "replay": {"fidelity_pass": True},
                "exact_compiled_geometry": {
                    "robot_primitive_certificate_pass": True,
                    "group_raw_contact_sample_count": {
                        "palm": int(positive),
                        "L5": int(positive),
                        "L6": int(positive),
                    },
                    "group_physical_false_safe_sample_count": {
                        "palm": 0, "L5": 0, "L6": 0,
                    },
                    "group_episode_minimum_normalized_radial_slack": {
                        "palm": -0.1 if positive else 0.1,
                        "L5": -0.1 if positive else 0.1,
                        "L6": -0.1 if positive else 0.1,
                    },
                },
            })
        self.assertTrue(
            summarize_case_records(records, config)["geometry_gate_pass"]
        )
        records[0]["exact_compiled_geometry"][
            "group_physical_false_safe_sample_count"
        ]["L6"] = 1
        self.assertFalse(
            summarize_case_records(records, config)["geometry_gate_pass"]
        )

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
