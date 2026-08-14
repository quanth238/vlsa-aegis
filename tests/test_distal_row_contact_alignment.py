import unittest

from main.multilink_ellipsoid.row_contact_alignment import (
    decision, segment_samples, summarize_samples,
)


def config():
    return {
        "geometry_contract": {
            "row_groups": {"L5": [0, 1, 2], "L6": [3, 4], "L7": [5, 6]},
            "protected_geom_to_group": {
                "robot0_link5_collision": "L5",
                "robot0_link6_collision": "L6",
                "robot0_link7_collision": "L7",
            },
            "proxy_overlap_threshold_m": 0.0,
            "certification_buffer_m": 0.001,
        }
    }


class RowContactAlignmentTest(unittest.TestCase):
    def test_segment_alignment(self):
        segment = {
            "clearance_trace_m": [[0.1] * 7, [0.2] * 7, [-0.2] + [0.2] * 6],
            "substep_counts": [2],
            "protected_contacts": [{
                "step": 10, "substep": 1,
                "protected_geom_name": "robot0_link5_collision",
            }],
        }
        samples = segment_samples(segment, start_step=10)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[1]["step"], 10)
        self.assertEqual(samples[1]["substep"], 1)
        self.assertEqual(len(samples[1]["contacts"]), 1)

    def test_repeated_action_boundary_contact_maps_to_previous_sample(self):
        segment = {
            "clearance_trace_m": [[0.1] * 7, [0.2] * 7, [-0.2] + [0.2] * 6],
            "substep_counts": [1, 1],
            "protected_contacts": [{
                "step": 11, "substep": -1,
                "protected_geom_name": "robot0_link5_collision",
            }],
        }
        samples = segment_samples(segment, start_step=10)
        self.assertEqual(len(samples[0]["contacts"]), 1)
        self.assertEqual(samples[0]["step"], 10)
        self.assertEqual(samples[0]["substep"], 0)

    def test_proxy_contact_alignment_and_active_row(self):
        records = [{
            "state_id": "state-a",
            "samples": [{
                "clearance_m": [0.01, -0.002, 0.02, 0.03, 0.04, 0.05, 0.06],
                "contacts": [{"protected_geom_name": "robot0_link5_collision"}],
            }],
        }]
        result = summarize_samples(records, config())
        l5 = result["groups"][0]
        self.assertEqual(l5["raw_contact_sample_count"], 1)
        self.assertEqual(l5["proxy_false_safe_contact_sample_count"], 0)
        self.assertEqual(l5["active_contact_witness_count_by_row"], [0, 1, 0])
        self.assertEqual(result["physical_active_contact_witness_rows"]["L5"], [1])
        self.assertEqual(
            decision(result)["primary_root_cause"],
            "population_coverage_not_ellipsoid_geometry",
        )

    def test_proxy_false_safe_requires_geometry_audit(self):
        records = [{
            "state_id": "state-b",
            "samples": [{
                "clearance_m": [0.01] * 7,
                "contacts": [{"protected_geom_name": "robot0_link5_collision"}],
            }],
        }]
        result = summarize_samples(records, config())
        self.assertTrue(result["geometry_revision_required"])
        self.assertTrue(decision(result)["ellipsoid_geometry_change_authorized"])

    def test_conservative_overlap_is_not_false_safe(self):
        records = [{
            "state_id": "state-c",
            "samples": [{"clearance_m": [-0.001] + [0.1] * 6, "contacts": []}],
        }]
        result = summarize_samples(records, config())
        l5 = result["groups"][0]
        self.assertEqual(l5["proxy_overlap_without_contact_sample_count"], 1)
        self.assertEqual(result["proxy_false_safe_contact_sample_count"], 0)


if __name__ == "__main__":
    unittest.main()
