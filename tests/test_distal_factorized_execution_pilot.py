import copy
import unittest
from pathlib import Path

import numpy as np

from main.multilink_ellipsoid.factorized_execution_pilot import (
    candidate_chunks, dataset_arrays, factorized_decision,
    finite_difference_candidate_indexes, load_config, sensitivity_arrays,
    trace_arrays,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "vlsa_distal_factorized_execution_moka10.v1.json"


class FactorizedExecutionPilotTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def test_candidate_population_covers_both_complete_actions(self):
        nominal = np.zeros((2, 7), dtype=np.float64)
        nominal[0, 6] = 1.0
        nominal[1, 6] = -1.0
        first = candidate_chunks(nominal, 9, self.config)
        second = candidate_chunks(nominal, 9, self.config)
        self.assertEqual(len(first), 93)
        self.assertEqual(first, second)
        observed_dimensions = []
        for dimension in range(14):
            negative, positive = finite_difference_candidate_indexes(dimension)
            pair = np.asarray([
                first[negative]["full_two_action_commands"],
                first[positive]["full_two_action_commands"],
            ]).reshape(2, 14)
            changed = np.flatnonzero(np.abs(pair[1] - pair[0]) > 1.0e-15)
            self.assertEqual(changed.tolist(), [dimension])
            observed_dimensions.extend(changed.tolist())
        self.assertEqual(observed_dimensions, list(range(14)))
        self.assertTrue(any(
            not np.allclose(
                item["full_two_action_commands"][0][3:6],
                nominal[0, 3:6],
            ) for item in first
        ))
        self.assertTrue(any(
            not np.allclose(
                item["full_two_action_commands"][1][3:6],
                nominal[1, 3:6],
            ) for item in first
        ))

    def test_trace_removes_only_second_interval_start(self):
        def transition(offset):
            substeps = []
            for index in range(26):
                value = offset + index
                substeps.append({
                    "robot_joint_position_rad": [value] * 7,
                    "clearance_m": [0.001 * value] * 8,
                })
            return {
                "expected_internal_mujoco_step_count": 25,
                "captured_state_count": 26,
                "substeps": substeps,
                "raw_protected_contact_count": 0,
                "next_state_sha256": str(offset),
                "maximum_within_step_obstacle_l1_displacement_m": 0.0,
            }

        receipt = trace_arrays({"transitions": [transition(0), transition(25)]}, self.config)
        self.assertEqual(receipt["joint_position_rad"].shape, (51, 7))
        self.assertEqual(receipt["ellipsoid_clearance_m"].shape, (51, 7))
        self.assertEqual(receipt["joint_position_rad"][25, 0], 25.0)
        self.assertEqual(receipt["joint_position_rad"][26, 0], 26.0)

    def test_sensitivity_uses_all_fourteen_actual_clipped_denominators(self):
        rows = []
        for state_index, split_code in ((0, 0), (1, 2)):
            nominal = np.zeros((2, 7), dtype=np.float64)
            candidates = candidate_chunks(nominal, state_index, self.config)
            for item in candidates:
                action = np.asarray(item["full_two_action_commands"], dtype=np.float64)
                flat = action.reshape(-1)
                q = np.empty((3, 7), dtype=np.float64)
                for time in range(3):
                    q[time] = (time + 1) * np.sum(
                        flat[:, None] * (np.arange(14)[:, None] + 1), axis=0
                    )
                clearance = np.repeat(
                    (0.1 + flat[:7])[None, :], 3, axis=0
                )
                rows.append((state_index, split_code, item, action, q, clearance))
        archive = {
            "state_input_vector": np.asarray([[row[0], 1.0] for row in rows]),
            "action_chunk": np.asarray([row[3] for row in rows]),
            "joint_position_rad": np.asarray([row[4] for row in rows]),
            "ellipsoid_clearance_m": np.asarray([row[5] for row in rows]),
            "state_index": np.asarray([row[0] for row in rows]),
            "candidate_index": np.asarray([row[2]["candidate_index"] for row in rows]),
            "split_code": np.asarray([row[1] for row in rows]),
            "source_code": np.asarray([
                0 if row[2]["source"] == "nominal" else
                1 if row[2]["source"].startswith("finite_difference") else 2
                for row in rows
            ]),
            "raw_contact_count": np.zeros(len(rows), dtype=np.int64),
        }
        metadata = {"summary": {
            "rollout_count": len(rows), "joint_trace_state_count": 3,
            "complete_state_input_dimension": 2,
        }}
        arrays = dataset_arrays(metadata, archive)
        sensitivity = sensitivity_arrays(arrays)
        self.assertEqual(sensitivity["joint_sensitivity_rad_per_action"].shape, (28, 3, 7))
        self.assertEqual(sensitivity["margin_sensitivity_m_per_action"].shape, (28, 7))
        self.assertEqual(sorted(set(sensitivity["dimension_index"].tolist())), list(range(14)))
        self.assertTrue(np.all(np.isfinite(sensitivity["joint_sensitivity_rad_per_action"])))

    def test_decision_requires_factorized_to_beat_direct(self):
        direct = {
            "false_safe_action_count": 40,
        }
        factor = {
            "false_safe_action_count": 30,
            "exact_safe_action_recall": 0.8,
            "state_safe_support_count": 15,
            "near_boundary_RMSE_m": 0.002,
        }
        decision = factorized_decision(
            collection_pass=True, direct_metrics=direct,
            exact_geometry_metrics={
                "false_safe_action_count": 0,
                "exact_safe_action_recall": 1.0,
            },
            factorized_metrics=factor, joint_rmse_rad=0.005,
            link_center_rmse_m=0.002, joint_sensitivity_cosine=0.9,
            margin_sensitivity_cosine=0.9, config=self.config,
        )
        self.assertTrue(decision["factorization_GO"])
        worse = copy.deepcopy(factor)
        worse["false_safe_action_count"] = 40
        decision = factorized_decision(
            collection_pass=True, direct_metrics=direct,
            exact_geometry_metrics={
                "false_safe_action_count": 0,
                "exact_safe_action_recall": 1.0,
            },
            factorized_metrics=worse, joint_rmse_rad=0.005,
            link_center_rmse_m=0.002, joint_sensitivity_cosine=0.9,
            margin_sensitivity_cosine=0.9, config=self.config,
        )
        self.assertFalse(decision["factorization_GO"])


if __name__ == "__main__":
    unittest.main()
