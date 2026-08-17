from pathlib import Path
import unittest
from unittest import mock


class TightEEGeometryTests(unittest.TestCase):
    def test_config_freezes_five_separate_contact_geoms(self):
        from main.multilink_ellipsoid.tight_ee_geometry import load_config

        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/vlsa_tight_ee_geometry_audit.v1.json",
            repo_root=root,
        )
        self.assertEqual(
            config["tight_ee_primitives"],
            {
                "palm": "gripper0_hand_collision",
                "finger1_base": "gripper0_finger1_collision",
                "finger1_pad": "gripper0_finger1_pad_collision",
                "finger2_base": "gripper0_finger2_collision",
                "finger2_pad": "gripper0_finger2_pad_collision",
            },
        )
        self.assertFalse(config["cohort"]["outcomes_used_for_primitive_fit"])
        self.assertEqual(len(config["cohort"]["cases"]), 4)

    def test_summary_requires_raw_contacts_zero_false_safes_and_control_clearance(self):
        from main.multilink_ellipsoid.tight_ee_geometry import (
            load_config, summarize_records,
        )

        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/vlsa_tight_ee_geometry_audit.v1.json",
            repo_root=root,
        )
        groups = list(config["tight_ee_primitives"])
        records = []
        for case in config["cohort"]["cases"]:
            expected = list(case["expected_contact_groups"])
            records.append({
                "case_id": case["case_id"],
                "replay": {"fidelity_pass": True},
                "raw_contact": {
                    "observed_groups": expected,
                    "sample_count_by_group": {
                        group: int(group in expected) for group in groups
                    },
                },
                "geometry": {
                    "primitive_certificate_pass": True,
                    "physical_false_safe_sample_count_by_group": {
                        group: 0 for group in groups
                    },
                    "episode_minimum_radial_slack_by_group": {
                        group: (-0.1 if group in expected else 0.1)
                        for group in groups
                    },
                },
            })
        summary = summarize_records(records, config)
        self.assertTrue(summary["tight_ee_geometry_gate_pass"])
        records[0]["geometry"]["physical_false_safe_sample_count_by_group"]["palm"] = 1
        self.assertFalse(
            summarize_records(records, config)["tight_ee_geometry_gate_pass"]
        )

    def test_contact_geom_dispatch_keeps_each_geom_independent(self):
        from main.multilink_ellipsoid import palm_primitive_audit as audit

        class Model:
            ngeom = 1
            geom_type = [7]
            geom_contype = [1]
            geom_conaffinity = [1]

        class SimModel:
            def geom_id2name(self, geom_id):
                return "gripper0_hand_collision"

        env = mock.Mock()
        env.sim.model = SimModel()
        sentinel = object()
        with (
            mock.patch.object(audit, "_raw_model_data", return_value=(Model(), object())),
            mock.patch.object(audit, "_geom_kind", return_value="mesh"),
            mock.patch.object(audit, "fit_compiled_mesh_geom", return_value=sentinel) as mesh,
            mock.patch.object(audit, "fit_compiled_primitive_geom") as primitive,
        ):
            value = audit.fit_compiled_contact_geom(
                env,
                "gripper0_hand_collision",
                relative_padding=1.0e-6,
                tolerance=1.0e-9,
                max_iterations=100,
            )
        self.assertIs(value, sentinel)
        mesh.assert_called_once()
        primitive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
