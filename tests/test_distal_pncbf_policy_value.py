import json
import unittest

from main.multilink_ellipsoid.pncbf_policy_value import (
    exact_group_action_boundary_values,
    exact_suffix_policy_values,
    violation_trace,
)


class DistalPncbfPolicyValueTest(unittest.TestCase):
    def test_exact_group_values_sample_action_boundaries_not_substeps(self):
        boundaries = [
            {
                "action_offset": 0,
                "phase": "prefix",
                "group_normalized_radial_slack": {"L5": 0.3, "L6": 0.5},
            },
            {
                "action_offset": 1,
                "phase": "backup",
                "group_normalized_radial_slack": {"L5": 0.2, "L6": 0.4},
            },
        ]
        trace = [
            {
                "action_offset": 0,
                "phase": "prefix",
                "substep": 0,
                "group_normalized_radial_slack": {"L5": 0.1, "L6": 0.3},
            },
            {
                "action_offset": 0,
                "phase": "prefix",
                "substep": 1,
                "group_normalized_radial_slack": {"L5": -0.2, "L6": 0.2},
            },
            {
                "action_offset": 1,
                "phase": "backup",
                "substep": 0,
                "group_normalized_radial_slack": {"L5": 0.4, "L6": -0.1},
            },
        ]
        value = exact_group_action_boundary_values(
            boundaries, trace, group_order=("L5", "L6"),
        )
        json.dumps(value, allow_nan=False)
        self.assertEqual(value["action_boundary_count"], 2)
        self.assertEqual(value["maximum_bellman_residual"], 0.0)
        self.assertAlmostEqual(value["records"][0]["value"]["L5"], 0.2)
        self.assertAlmostEqual(value["records"][0]["value"]["L6"], 0.1)
        self.assertAlmostEqual(value["records"][1]["value"]["L5"], -0.2)
        self.assertAlmostEqual(value["records"][1]["value"]["L6"], 0.1)

    def test_accepts_array_like_trace_without_importing_numpy(self):
        class ArrayLike:
            def __init__(self, value):
                self.value = value

            def tolist(self):
                return self.value

        self.assertEqual(
            violation_trace(ArrayLike([[0.002] * 7]), 0.001),
            [[-0.001] * 7],
        )

    def test_positive_is_unsafe(self):
        self.assertEqual(
            violation_trace([[0.002, 0.001, 0.0, 0.003, 0.004, 0.005, 0.006]], 0.001),
            [[-0.001, 0.0, 0.001, -0.002, -0.003, -0.004, -0.005]],
        )

    def test_exact_suffix_values_obey_discrete_recursion(self):
        windows = [
            {
                "step": 10,
                "executed_action_count": 1,
                "selected_clearance_trace_m": [
                    [0.003] * 7,
                    [0.002] * 7,
                    [0.0025] * 7,
                ],
            },
            {
                "step": 11,
                "executed_action_count": 1,
                "selected_clearance_trace_m": [
                    [0.0025] * 7,
                    [0.004] * 7,
                    [0.005] * 7,
                ],
            },
        ]
        value = exact_suffix_policy_values(
            windows,
            terminal_tail_clearance_trace_m=[[0.005] * 7, [0.006] * 7],
            safety_buffer_m=0.001,
            expected_substeps_per_action=2,
            boundary_tolerance_m=1e-12,
        )
        self.assertTrue(value["all_decision_states_safe"])
        self.assertTrue(value["nonincreasing_along_backup"])
        self.assertEqual(value["maximum_bellman_residual"], 0.0)
        for item in value["records"][0]["value"]:
            self.assertAlmostEqual(item, -0.001)
        for item in value["records"][1]["value"]:
            self.assertAlmostEqual(item, -0.0015)

    def test_rejects_noncontiguous_successor_state(self):
        windows = [
            {"step": 0, "executed_action_count": 1, "selected_clearance_trace_m": [[0.002] * 7, [0.003] * 7]},
            {"step": 2, "executed_action_count": 1, "selected_clearance_trace_m": [[0.003] * 7, [0.004] * 7]},
        ]
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            exact_suffix_policy_values(
                windows,
                terminal_tail_clearance_trace_m=[[0.004] * 7],
                safety_buffer_m=0.001,
                expected_substeps_per_action=1,
                boundary_tolerance_m=1e-12,
            )

    def test_reports_terminal_successor_mismatch(self):
        windows = [
            {
                "step": 0,
                "executed_action_count": 1,
                "selected_clearance_trace_m": [[0.002] * 7, [0.003] * 7],
            },
        ]
        value = exact_suffix_policy_values(
            windows,
            terminal_tail_clearance_trace_m=[[0.004] * 7],
            safety_buffer_m=0.001,
            expected_substeps_per_action=1,
            boundary_tolerance_m=1e-12,
        )
        self.assertFalse(value["successor_boundaries_consistent"])
        self.assertAlmostEqual(value["maximum_successor_boundary_error_m"], 0.001)


if __name__ == "__main__":
    unittest.main()
