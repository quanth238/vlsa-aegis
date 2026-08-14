import unittest
from pathlib import Path

from main.multilink_ellipsoid.l5_row01_input_ablation import (
    COMPLETE_DIMENSION, complete_feature_vector, load_config,
)


def _matrix(scale=1.0):
    return [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, scale]]


class L5Row01InputAblationTest(unittest.TestCase):
    def test_registered_config(self):
        config = load_config(Path(
            "configs/vlsa_distal_l5_row01_input_ablation.v1.json"
        ))
        self.assertEqual(
            config["arms"]["complete_physical_OSC_354D"]["input_dimension"],
            COMPLETE_DIMENSION,
        )
        self.assertIn("QP", config["forbidden"])

    def test_complete_feature_dimension_and_nullable_controller_memory(self):
        rows = []
        for index in range(7):
            rows.append({
                "body_name": (
                    "robot0_link5" if index < 3 else
                    "robot0_link6" if index < 5 else "robot0_link7"
                ),
                "center_m": [0.1, 0.2, 0.3],
                "rotation": _matrix(),
                "semiaxes_m": [0.1, 0.1, 0.1],
                "obstacle_relative_center_m": [0.2, 0.3, 0.4],
                "outward_normal": [1.0, 0.0, 0.0],
                "current_clearance_m": 0.01,
            })
        context = {
            "arm_joint_position_rad": [0.0] * 7,
            "arm_joint_velocity_rad_s": [0.0] * 7,
            "eef_position_m": [0.0] * 3,
            "eef_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "controller_snapshot": {
                "goal_ori": _matrix(), "goal_pos": [0.0] * 3,
                "gripper_current_action": [0.0, 0.0], "new_update": True,
                "ori_ref": None, "relative_ori": [0.0] * 3,
                "robot_torques": [0.0] * 7, "torques": [0.0] * 7,
            },
            "obstacle": {
                "center_m": [0.0] * 3, "rotation": _matrix(),
                "semiaxes_m": [0.1] * 3,
            },
            "geometry_rows": rows,
        }
        qps = [{
            "status": "solved", "solver_status": "optimal",
            "barrier_h": 0.1, "constraint_lhs": 0.0, "objective": 0.0,
            "u_solution": [0.0] * 6, "z_before": [0.0] * 3,
            "z_after": [0.0] * 3,
        } for _ in range(5)]
        aegis = {
            "enabled": True, "qp_records": qps,
            "proposed_actions": [[0.0] * 7 for _ in range(5)],
            "correction_l2_action": 0.0,
            "maximum_absolute_action_change": 0.0,
            "z_before": [0.0] * 3,
            "z_after_by_action": [[0.0] * 3 for _ in range(5)],
        }
        feature = complete_feature_vector(
            physical_context=context,
            nominal_actions=[[0.0] * 7 for _ in range(5)],
            candidate_actions=[[0.0] * 7 for _ in range(5)],
            aegis=aegis,
        )
        self.assertEqual(len(feature), COMPLETE_DIMENSION)
        self.assertEqual(feature[36], 0.0)  # ori_ref presence bit


if __name__ == "__main__":
    unittest.main()
