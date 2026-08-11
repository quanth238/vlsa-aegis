import inspect
from pathlib import Path
import unittest

import numpy as np

from main.multilink_ellipsoid.factorized_direct_horizon_root_cause_audit import (
    _population_metrics, candidate_group, fit_diagnostic_geometry_jacobians,
    load_audit_config, sensitivity_audit,
)
from main.multilink_ellipsoid.factorized_execution_pilot import sensitivity_arrays
from scripts.evaluate_distal_direct_horizon_displacement_moka10 import (
    evaluate_reserved_geometry,
)
from scripts.evaluate_distal_factorized_execution_moka10 import (
    evaluate_predicted_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/vlsa_distal_factorized_direct_horizon_root_cause_audit_moka10.v2.json"


def fixture_arrays() -> dict:
    count = 93
    q = np.zeros((count, 51, 7), dtype=np.float64)
    actions = np.zeros((count, 2, 7), dtype=np.float64)
    for candidate in range(1, 29):
        dimension = (candidate - 1) // 2
        sign = -1.0 if candidate % 2 else 1.0
        q[candidate, :, dimension % 7] = sign * 0.01
        actions[candidate].reshape(-1)[dimension] = sign * 0.1
    generator = np.random.default_rng(7)
    q[29:] = generator.normal(scale=0.005, size=(64, 51, 7))
    coefficient = np.linspace(0.01, 0.07, 7)
    h = 0.01 + q * coefficient[None, None, :]
    return {
        "features": np.zeros((count, 20)),
        "action_chunk": actions,
        "joint_position_rad": q,
        "ellipsoid_clearance_m": h,
        "minimum_margin_m": np.min(h, axis=1),
        "current_margin_m": h[:, 0],
        "state_index": np.zeros(count, dtype=np.int64),
        "candidate_index": np.arange(count, dtype=np.int64),
        "split": np.asarray(["train"] * count, dtype=object),
        "source_code": np.r_[0, np.ones(28, dtype=np.int8), np.full(64, 2, dtype=np.int8)],
        "raw_contact_count": np.zeros(count, dtype=np.int64),
    }


class DirectHorizonRootCauseAuditTest(unittest.TestCase):
    def test_config_and_candidate_groups(self) -> None:
        config = load_audit_config(CONFIG)
        self.assertEqual(config["audit"]["terminal_window_start"], 45)
        self.assertIn("14_action_dimensions", config["audit"]["sensitivity_reporting"])
        self.assertEqual(candidate_group(0, 0), "nominal")
        self.assertEqual(candidate_group(1, 1), "translation_FD")
        self.assertEqual(candidate_group(7, 1), "rotation_FD")
        self.assertEqual(candidate_group(13, 1), "gripper_FD")
        self.assertEqual(candidate_group(29, 2), "mixed_random")

    def test_diagnostic_geometry_and_population_localization(self) -> None:
        arrays = fixture_arrays()
        fit = fit_diagnostic_geometry_jacobians(arrays, ridge=1.0e-12)
        self.assertLess(fit["fit_residual"]["RMSE"], 1.0e-8)
        predicted_h = arrays["ellipsoid_clearance_m"].copy()
        arrays["ellipsoid_clearance_m"][29, 50, 0] = -0.001
        predicted_h[29, 50, 0] = 0.001
        exact_q = arrays["joint_position_rad"]
        result = _population_metrics(
            name="train", arrays=arrays, predicted_q=exact_q,
            member_q=np.repeat(exact_q[None], 5, axis=0),
            predicted_h=predicted_h, exact_static_h=arrays["ellipsoid_clearance_m"],
            state_to_episode={0: "episode"},
            input_distance={0: {"nearest_RMS_z": 0.0}},
            config=load_audit_config(CONFIG),
        )
        self.assertEqual(result["safety"]["false_safe_count"], 1)
        row = result["safety"]["false_safe_localization"][0]
        self.assertEqual((row["active_link"], row["active_substep"]), ("L5", 50))

    def test_geometry_evaluators_offer_trace_receipts(self) -> None:
        self.assertIn("return_trace", inspect.signature(evaluate_predicted_geometry).parameters)
        self.assertIn("return_trace", inspect.signature(evaluate_reserved_geometry).parameters)
        source = inspect.getsource(evaluate_reserved_geometry)
        self.assertIn('clearance_traces[name][row] = trace', source)
        self.assertNotIn('traces[name][row] = trace', source)

    def test_sensitivity_reports_every_dimension_and_horizon(self) -> None:
        arrays = fixture_arrays()
        audit = sensitivity_audit(
            arrays=arrays, predicted_q=arrays["joint_position_rad"],
            predicted_h=arrays["ellipsoid_clearance_m"],
            sensitivities=sensitivity_arrays(arrays),
            state_to_split={0: "train"},
        )
        dimensions = audit["train"]["by_dimension_and_horizon"]
        self.assertEqual(len(dimensions), 14)
        self.assertEqual(len(dimensions["action_0_translation_x"]["by_horizon"]), 51)
        self.assertAlmostEqual(
            dimensions["action_0_translation_x"]["by_horizon"][50]
            ["joint"]["mean_cosine"], 1.0,
        )


if __name__ == "__main__":
    unittest.main()
