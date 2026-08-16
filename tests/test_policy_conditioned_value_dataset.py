import copy
import unittest

from main.multilink_ellipsoid.policy_conditioned_value_dataset import (
    action_chunk_feature, constraint_state_feature, coverage_summary,
    extract_case_samples, readiness_checks,
)


GROUPS = ["palm", "L5", "L6"]
ROWS = {"palm": [0], "L5": [1, 2, 3], "L6": [4, 5]}


def boundary(offset, phase, action, slacks):
    rows = []
    names = ["hand", "robot0_link5", "robot0_link5", "robot0_link5", "robot0_link6", "robot0_link6"]
    for index, name in enumerate(names):
        rows.append({
            "body_name": name,
            "center_m": [0.1 + 0.01 * index, 0.2, 0.3],
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "semiaxes_m": [0.02, 0.03, 0.04],
        })
    return {
        "action_offset": offset,
        "phase": phase,
        "executed_action": action,
        "arm_joint_position_rad": [0.1] * 7,
        "arm_joint_velocity_rad_s": [0.2] * 7,
        "eef_position_m": [0.4, 0.5, 0.6],
        "controller_snapshot": {"goal_pos": [0.41, 0.52, 0.63]},
        "exact_robot_rows": rows,
        "compiled_obstacle_boxes": [{
            "center_m": [0.0, 0.0, 0.0],
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "half_extents_m": [0.1, 0.2, 0.3],
        }],
        "row_normalized_radial_slack": slacks,
    }


def candidate(name, l5_target):
    phases = ["prefix"] * 5 + ["backup", "terminal_hold"]
    boundaries = [
        boundary(index, phase, [0.1, -0.2, 0.3, 0, 0, 0, 1], [0.3, 0.2, 0.1, 0.4, 0.5, 0.6])
        for index, phase in enumerate(phases)
    ]
    records = []
    for index, phase in enumerate(phases):
        records.append({
            "action_offset": index,
            "phase": phase,
            "value": {"palm": -0.1, "L5": l5_target, "L6": -0.2},
            "training_sample_eligible": phase in ("backup", "terminal_hold"),
        })
    return {
        "name": name,
        "exact_group_target": {
            "known_outcome": True,
            "action_boundaries": boundaries,
            "trajectory_policy_value": {"records": records},
        },
    }


def case(case_id, split):
    return {
        "case_id": case_id,
        "selection": {"split": split},
        "exact_case": {"candidates": [candidate("safe", -0.1), candidate("unsafe", 0.2)]},
    }


class PolicyConditionedValueDatasetTest(unittest.TestCase):
    def test_features_are_boundary_only_and_complete(self):
        row = candidate("safe", -0.1)["exact_group_target"]["action_boundaries"][0]
        feature = constraint_state_feature(
            row, group="L5", group_order=GROUPS, group_rows=ROWS,
        )
        self.assertEqual(len(feature), 38)
        prefix = [copy.deepcopy(row) for _ in range(5)]
        for index, item in enumerate(prefix):
            item["action_offset"] = index
        action = action_chunk_feature(
            prefix, prefix_action_count=5,
            translation_scale_m_per_action_unit=0.05,
        )
        self.assertEqual(len(action), 15)
        self.assertAlmostEqual(action[0], 0.005)

    def test_exact_first_bellman_record_is_query_target(self):
        extracted = extract_case_samples(
            case("E00", "train"), learned_groups=["L5"],
            group_order=GROUPS, group_rows=ROWS,
            eligible_value_phases=["backup", "terminal_hold"],
            prefix_action_count=5, translation_scale_m_per_action_unit=0.05,
        )
        self.assertEqual(len(extracted["query_samples"]), 2)
        self.assertEqual([row["target"] for row in extracted["query_samples"]], [-0.1, 0.2])
        self.assertEqual(len(extracted["value_samples"]), 4)
        self.assertTrue(all(row["phase"] != "prefix" for row in extracted["value_samples"]))

    def test_unknown_trajectory_is_censored(self):
        value = case("E00", "train")
        unknown = value["exact_case"]["candidates"][0]
        unknown["exact_group_target"]["known_outcome"] = False
        for record in unknown["exact_group_target"]["trajectory_policy_value"]["records"]:
            record["training_sample_eligible"] = False
        extracted = extract_case_samples(
            value, learned_groups=["L5"], group_order=GROUPS, group_rows=ROWS,
            eligible_value_phases=["backup", "terminal_hold"],
            prefix_action_count=5, translation_scale_m_per_action_unit=0.05,
        )
        self.assertEqual(extracted["unknown_candidate_count"], 1)
        self.assertEqual(len(extracted["query_samples"]), 1)
        self.assertEqual(len(extracted["value_samples"]), 2)

    def test_readiness_counts_independent_states_not_actions(self):
        extracted = [
            extract_case_samples(
                case("E00", "train"), learned_groups=["L5"],
                group_order=GROUPS, group_rows=ROWS,
                eligible_value_phases=["backup", "terminal_hold"],
                prefix_action_count=5, translation_scale_m_per_action_unit=0.05,
            ),
            extract_case_samples(
                case("E01", "validation"), learned_groups=["L5"],
                group_order=GROUPS, group_rows=ROWS,
                eligible_value_phases=["backup", "terminal_hold"],
                prefix_action_count=5, translation_scale_m_per_action_unit=0.05,
            ),
        ]
        coverage = coverage_summary(extracted, learned_groups=["L5"])
        checks = readiness_checks(
            coverage, learned_groups=["L5"],
            required_two_sided_states={"train": 1, "validation": 1},
            required_value_episode_groups={"train": 1, "validation": 1},
        )
        self.assertEqual(coverage["train"]["groups"]["L5"]["two_sided_state_count"], 1)
        self.assertTrue(checks["training_ready"])


if __name__ == "__main__":
    unittest.main()
