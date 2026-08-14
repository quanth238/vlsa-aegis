import json
import unittest
from pathlib import Path

import numpy as np

from main.multilink_ellipsoid.l5_row01_relative_root_cause import (
    RELATIVE_DIMENSION, STATE_INDICES, load_config, relative_feature_vector,
    witness_phase,
)


class RelativeRootCauseTest(unittest.TestCase):
    def test_registered_config(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(
            root / "configs/vlsa_distal_l5_row01_relative_root_cause.v1.json"
        )
        self.assertEqual(config["relative_input"]["input_dimension"], 134)
        self.assertEqual(len(STATE_INDICES), 64)

    def test_relative_feature_is_rigid_frame_invariant(self):
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        context = {
            "arm_joint_position_rad": np.arange(7).tolist(),
            "arm_joint_velocity_rad_s": (np.arange(7) * 0.1).tolist(),
            "eef_position_m": [0.1, 0.2, 0.3],
            "controller_snapshot": {
                "goal_pos": [0.2, 0.4, 0.6], "relative_ori": [0.1, -0.2, 0.3],
            },
            "obstacle": {
                "center_m": [0.4, 0.5, 0.6], "rotation": np.eye(3).tolist(),
                "semiaxes_m": [0.1, 0.2, 0.3],
            },
            "geometry_rows": [],
        }
        for index in range(7):
            context["geometry_rows"].append({
                "body_name": "robot0_link5" if index < 3 else "diagnostic",
                "center_m": [0.2 + index * 0.01, 0.1, 0.3],
                "rotation": np.eye(3).tolist(), "semiaxes_m": [0.03, 0.04, 0.05],
                "outward_normal": [1.0, 0.0, 0.0], "current_clearance_m": 0.01,
            })
        nominal = np.arange(35, dtype=np.float64).reshape(5, 7) / 100.0
        candidate = nominal + 0.01
        first = np.asarray(relative_feature_vector(
            physical_context=context, nominal_actions=nominal,
            candidate_actions=candidate,
        ))
        translated = np.asarray([0.8, -0.3, 0.2])
        changed = json.loads(json.dumps(context))
        for field in ("eef_position_m",):
            changed[field] = (rotation @ np.asarray(context[field]) + translated).tolist()
        changed["controller_snapshot"]["goal_pos"] = (
            rotation @ np.asarray(context["controller_snapshot"]["goal_pos"]) + translated
        ).tolist()
        changed["controller_snapshot"]["relative_ori"] = (
            rotation @ np.asarray(context["controller_snapshot"]["relative_ori"])
        ).tolist()
        changed["obstacle"]["center_m"] = (
            rotation @ np.asarray(context["obstacle"]["center_m"]) + translated
        ).tolist()
        changed["obstacle"]["rotation"] = rotation.tolist()
        for index, row in enumerate(context["geometry_rows"]):
            changed["geometry_rows"][index]["center_m"] = (
                rotation @ np.asarray(row["center_m"]) + translated
            ).tolist()
            changed["geometry_rows"][index]["rotation"] = rotation.tolist()
            changed["geometry_rows"][index]["outward_normal"] = (
                rotation @ np.asarray(row["outward_normal"])
            ).tolist()
        nominal_changed = nominal.copy()
        candidate_changed = candidate.copy()
        nominal_changed[:, :3] = nominal[:, :3] @ rotation.T
        nominal_changed[:, 3:6] = nominal[:, 3:6] @ rotation.T
        candidate_changed[:, :3] = candidate[:, :3] @ rotation.T
        candidate_changed[:, 3:6] = candidate[:, 3:6] @ rotation.T
        second = np.asarray(relative_feature_vector(
            physical_context=changed, nominal_actions=nominal_changed,
            candidate_actions=candidate_changed,
        ))
        self.assertEqual(first.shape, (RELATIVE_DIMENSION,))
        self.assertTrue(np.allclose(first, second, atol=1e-12))

    def test_witness_phase(self):
        candidate = {
            "candidate_prefix_risk": [0.1, -0.2, 0, 0, 0, 0, 0],
            "backup_risk": [0.0, 0.3, 0, 0, 0, 0, 0],
            "combined_risk": [0.1, 0.3, 0, 0, 0, 0, 0],
        }
        self.assertEqual(witness_phase(candidate), ["prefix", "backup"])


if __name__ == "__main__":
    unittest.main()
