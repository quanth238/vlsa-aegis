from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest


try:
    import mujoco  # noqa: F401
    import numpy as np
except ImportError:  # pragma: no cover - local dependency gate
    mujoco = None
    np = None


if np is not None:
    from main.poisson_fullbody.geometry import OrientedBox
    from main.poisson_fullbody.robot_samples import BodySample
    from main.poisson_fullbody.shadow_identification import (
        StaticDriftThresholds,
        StaticPoissonShadowObserver,
        robot_root_body_ids,
    )


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(np is None, "MuJoCo/NumPy are unavailable")
class StaticPoissonShadowObserverTest(unittest.TestCase):
    def setUp(self):
        self.data = SimpleNamespace(
            geom_xpos=np.zeros((1, 3), dtype=np.float64),
            geom_xmat=np.eye(3, dtype=np.float64).reshape(1, 9),
            xpos=np.array([[0.20, 0.0, 0.40], [0.30, 0.0, 0.40]]),
            xmat=np.tile(np.eye(3, dtype=np.float64).reshape(1, 9), (2, 1)),
            qvel=np.array([-1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        self.model = SimpleNamespace()
        self.samples = (
            BodySample(0, 0, "robot0_link5", 10, "link5_collision", (0, 0, 0)),
            BodySample(1, 1, "robot0_link6", 11, "link6_collision", (0, 0, 0)),
        )
        self.box = OrientedBox(
            center=np.zeros(3),
            R=np.eye(3),
            half_extents=np.array([0.05, 0.05, 0.05]),
            geom_id=0,
        )

    def _snapshot(self, _sim):
        return self.model, self.data

    def _jacobian(self, _model, data, sample, _dofs):
        jacobian = np.zeros((3, 7), dtype=np.float64)
        jacobian[0, 0] = 1.0
        return sample.world_point(data), jacobian

    def test_registered_warning_lead_and_drift_do_not_truncate_trace(self):
        class Field:
            def query(_self, point):
                return SimpleNamespace(
                    valid=True,
                    value=float(point[0]),
                    gradient=(1.0, 0.0, 0.0),
                    reason=None,
                )

        observer = StaticPoissonShadowObserver(
            field=Field(),
            samples=self.samples,
            settled_obstacle_boxes=(self.box,),
            arm_dof_indices=range(7),
            alpha_gain_per_s=5.0,
            physics_timestep_s=0.002,
            drift_thresholds=StaticDriftThresholds(1e-6, 1e-5, 1e-6),
            physics_substeps_per_high_level_action=3,
            snapshot_provider=self._snapshot,
            jacobian_provider=self._jacobian,
        )
        first = observer.observe(object(), high_level_index=0, physics_substep_index=0)
        observer.observe(object(), high_level_index=0, physics_substep_index=1)
        self.data.geom_xpos[0, 0] = 2e-6
        last = observer.observe(object(), high_level_index=0, physics_substep_index=2)
        self.assertTrue(first["field_query_attempted"])
        self.assertFalse(last["field_query_attempted"])
        result = observer.result(
            expected_callback_count=3,
            first_link56_contact={
                "observation_index": 2,
                "robot_geom_id": 10,
                "robot_geom_name": "link5_collision",
                "robot_body_id": 0,
                "robot_body_name": "robot0_link5",
                "source_phase": "post_integration_recomputed",
            },
        )
        self.assertEqual(result["observed_callback_count"], 3)
        self.assertEqual(result["field_query_callback_count"], 2)
        self.assertAlmostEqual(observer.trace[0]["time_from_settled_state_s"], 0.002)
        self.assertEqual(
            result["first_static_field_invalidation"]["observation_index"], 2
        )
        assessment = result["contact_prediction_assessment"]
        self.assertEqual(
            assessment["assessment"],
            "registered_warning_preceded_link56_contact",
        )
        self.assertEqual(assessment["lead_physics_substeps"], 2)
        self.assertAlmostEqual(assessment["lead_time_s"], 0.004)
        self.assertFalse(result["hypothetical_qp"]["computed"])
        no_contact = observer.result(
            expected_callback_count=3, first_link56_contact=None
        )
        self.assertEqual(
            no_contact["contact_prediction_assessment"]["assessment"],
            "no_link56_contact_outcome",
        )
        self.assertIsNone(
            no_contact["signals"][
                "first_observed_minimum_cbf_lhs_negative_on_contact_geom"
            ]
        )
        live_same_index = observer.result(
            expected_callback_count=3,
            first_link56_contact={
                "observation_index": 0,
                "robot_geom_id": 10,
                "source_phase": "live_solver_phase_preintegration_geometry",
            },
        )
        self.assertEqual(
            live_same_index["contact_prediction_assessment"]["assessment"],
            "registered_warning_followed_link56_contact",
        )
        self.assertEqual(
            live_same_index["contact_prediction_assessment"][
                "lead_physics_substeps"
            ],
            0,
        )
        post_same_index = observer.result(
            expected_callback_count=3,
            first_link56_contact={
                "observation_index": 0,
                "robot_geom_id": 10,
                "source_phase": "post_integration_recomputed",
            },
        )
        self.assertEqual(
            post_same_index["contact_prediction_assessment"]["assessment"],
            "registered_warning_coincident_with_link56_contact",
        )

    def test_typed_invalid_cell_is_distinct_from_observed_cbf_diagnostic(self):
        class Reason:
            value = "invalid_cell"

        class Field:
            def query(_self, point):
                if point[0] < 0.25:
                    return SimpleNamespace(
                        valid=False, value=None, gradient=None, reason=Reason()
                    )
                return SimpleNamespace(
                    valid=True,
                    value=0.5,
                    gradient=(0.0, 1.0, 0.0),
                    reason=None,
                )

        observer = StaticPoissonShadowObserver(
            field=Field(),
            samples=self.samples,
            settled_obstacle_boxes=(self.box,),
            arm_dof_indices=range(7),
            alpha_gain_per_s=5.0,
            physics_timestep_s=0.002,
            drift_thresholds=StaticDriftThresholds(1e-6, 1e-5, 1e-6),
            physics_substeps_per_high_level_action=1,
            snapshot_provider=self._snapshot,
            jacobian_provider=self._jacobian,
        )
        row = observer.observe(object(), high_level_index=0, physics_substep_index=0)
        self.assertEqual(row["invalid_reason_counts"], {"invalid_cell": 1})
        result = observer.result(
            expected_callback_count=1,
            first_link56_contact={
                "observation_index": 0,
                "robot_geom_id": 11,
                "source_phase": "post_integration_recomputed",
            },
        )
        self.assertIsNotNone(result["signals"]["first_invalid_cell_query"])
        self.assertIsNone(
            result["signals"]["first_observed_minimum_cbf_lhs_negative"]
        )

    def test_all_samples_are_checked_when_minimum_h_sample_is_not_violating(self):
        class Field:
            def query(_self, point):
                if point[0] < 0.25:
                    return SimpleNamespace(
                        valid=True,
                        value=0.1,
                        gradient=(0.0, 0.0, 0.0),
                        reason=None,
                    )
                return SimpleNamespace(
                    valid=True,
                    value=0.2,
                    gradient=(1.0, 0.0, 0.0),
                    reason=None,
                )

        observer = StaticPoissonShadowObserver(
            field=Field(),
            samples=self.samples,
            settled_obstacle_boxes=(self.box,),
            arm_dof_indices=range(7),
            alpha_gain_per_s=5.0,
            physics_timestep_s=0.002,
            drift_thresholds=StaticDriftThresholds(1e-6, 1e-5, 1e-6),
            physics_substeps_per_high_level_action=1,
            snapshot_provider=self._snapshot,
            jacobian_provider=self._jacobian,
        )
        row = observer.observe(object(), high_level_index=0, physics_substep_index=0)
        self.assertEqual(row["minimum_h_sample"]["sample_id"], 0)
        self.assertGreater(
            row["minimum_h_sample"]["observed_cbf_lhs_m2_per_s"], 0.0
        )
        self.assertEqual(row["minimum_observed_cbf_lhs_sample"]["sample_id"], 1)
        self.assertLess(row["minimum_observed_cbf_lhs_m2_per_s"], 0.0)
        self.assertEqual(row["valid_query_count"], 2)
        self.assertEqual(row["observed_cbf_lhs_evaluation_count"], 2)
        result = observer.result(
            expected_callback_count=1,
            first_link56_contact={
                "observation_index": 0,
                "robot_geom_id": 11,
                "source_phase": "post_integration_recomputed",
            },
        )
        warning = result["signals"][
            "first_observed_minimum_cbf_lhs_negative_on_contact_geom"
        ]
        self.assertEqual(warning["evidence"]["sample_id"], 1)
        self.assertEqual(result["valid_field_query_count"], 2)
        self.assertEqual(result["observed_cbf_lhs_evaluation_count"], 2)
        self.assertTrue(result["every_valid_query_has_observed_cbf_lhs"])

    def test_invalid_cell_signal_uses_evidence_from_that_exact_reason(self):
        class Reason:
            def __init__(self, value):
                self.value = value

        class Field:
            def query(_self, point):
                reason = (
                    "nondifferentiable_internal_face"
                    if point[0] < 0.25
                    else "invalid_cell"
                )
                return SimpleNamespace(
                    valid=False,
                    value=None,
                    gradient=None,
                    reason=Reason(reason),
                )

        observer = StaticPoissonShadowObserver(
            field=Field(),
            samples=self.samples,
            settled_obstacle_boxes=(self.box,),
            arm_dof_indices=range(7),
            alpha_gain_per_s=5.0,
            physics_timestep_s=0.002,
            drift_thresholds=StaticDriftThresholds(1e-6, 1e-5, 1e-6),
            physics_substeps_per_high_level_action=1,
            snapshot_provider=self._snapshot,
            jacobian_provider=self._jacobian,
        )
        observer.observe(object(), high_level_index=0, physics_substep_index=0)
        result = observer.result(
            expected_callback_count=1,
            first_link56_contact={
                "observation_index": 0,
                "robot_geom_id": 10,
                "source_phase": "post_integration_recomputed",
            },
        )
        any_invalid = result["signals"]["first_any_fail_closed_query"]
        invalid_cell = result["signals"]["first_invalid_cell_query"]
        self.assertEqual(any_invalid["evidence"]["sample_id"], 0)
        self.assertEqual(
            any_invalid["evidence"]["reason"], "nondifferentiable_internal_face"
        )
        self.assertEqual(invalid_cell["evidence"]["sample_id"], 1)
        self.assertEqual(invalid_cell["evidence"]["reason"], "invalid_cell")
        primary = result["contact_prediction_assessment"][
            "primary_registered_warning"
        ]
        self.assertEqual(primary["signal_kind"], "any_invalid")
        self.assertEqual(
            primary["evidence"]["reason"], "nondifferentiable_internal_face"
        )

    def test_callback_index_gap_is_rejected(self):
        class Field:
            def query(_self, _point):
                return SimpleNamespace(
                    valid=True, value=1.0, gradient=(1.0, 0.0, 0.0), reason=None
                )

        observer = StaticPoissonShadowObserver(
            field=Field(),
            samples=self.samples,
            settled_obstacle_boxes=(self.box,),
            arm_dof_indices=range(7),
            alpha_gain_per_s=5.0,
            physics_timestep_s=0.002,
            drift_thresholds=StaticDriftThresholds(1e-6, 1e-5, 1e-6),
            physics_substeps_per_high_level_action=3,
            snapshot_provider=self._snapshot,
            jacobian_provider=self._jacobian,
        )
        with self.assertRaisesRegex(RuntimeError, "gap/duplicate"):
            observer.observe(object(), high_level_index=0, physics_substep_index=1)

    def test_robot_roots_are_derived_from_authoritative_parent_ids(self):
        parents = (0, 0, 1, 2, 0, 4)
        self.assertEqual(robot_root_body_ids((1, 2, 3, 4, 5), parents), (1, 4))


if __name__ == "__main__":
    unittest.main()
