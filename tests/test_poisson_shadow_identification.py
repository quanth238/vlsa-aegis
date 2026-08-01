from __future__ import annotations

import hashlib
import json
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


def physical_model_contract_evidence():
    return {
        "schema_version": "vlsa_poisson_physical_model.v3",
        "sha256": "7" * 64,
        "field_count": 128,
        "option_field_count": 16,
        "compiled_mjb_sha256": "8" * 64,
        "compiled_mjb_bytes": 4096,
        "nq": 7,
        "nv": 7,
        "na": 0,
        "mjstate_integration_size": 15,
        "robosuite_flattened_state_size": 15,
        "robosuite_flattened_state_layout": "time_qpos_qvel_act_no_udd_tail",
    }


class ProtectedSampleIdentity(dict):
    """JSON-native BodySample stand-in for dependency-free contract tests."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def to_dict(self):
        return {
            "sample_id": int(self["sample_id"]),
            "body_id": int(self["body_id"]),
            "body_name": self["body_name"],
            "geom_id": int(self["geom_id"]),
            "geom_name": self["geom_name"],
            "point_body_local_m": [
                float(value) for value in self["point_body_local_m"]
            ],
            "source": self["source"],
        }


def differential_audit_config():
    directions = [
        [0.5 if row == column else 0.0 for column in range(7)]
        for row in range(7)
    ]
    directions.extend(
        (
            [0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
            [(index + 1.0) / 14.0 for index in range(7)],
        )
    )
    return {
        "state_source": "settled_mujoco_mjstate_integration_clone",
        "perturbation_integrator": "mujoco_mj_integratePos_full_nv_tangent",
        "point_jacobian_delta_rad": 1.0e-6,
        "point_jacobian_absolute_tolerance_m_per_rad": 2.0e-6,
        "point_jacobian_relative_tolerance": 1.0e-4,
        "point_jacobian_near_zero_frobenius_m_per_rad": 1.0e-10,
        "arm_tangent_reconstruction_tolerance_rad_s": 1.0e-10,
        "nonarm_tangent_leakage_tolerance_rad_s": 1.0e-12,
        "joint_velocity_directions_rad_s": directions,
        "coupled_eta_ladder_s": [2.0e-6],
        "coupled_absolute_tolerance_m2_per_s": 2.0e-7,
        "coupled_relative_tolerance": 2.0e-4,
        "coupled_near_zero_m2_per_s": 1.0e-10,
        "same_trilinear_cell_required": True,
        "required_direction_count_per_sample": 9,
        "stencil_selection_policy": (
            "largest_eta_with_valid_base_plus_minus_in_same_exact_cell"
        ),
        "finite_difference_resolution_policy": (
            "fail_on_no_certified_stencil_or_detected_cancellation"
        ),
        "failure_policy": "fail_before_active_physics_retain_artifact",
    }


def protected_sample_identities(
    *,
    body_ids=(50, 60),
    body_names=("robot0_link5", "robot0_link6"),
    geom_ids=(5, 6),
    geom_names=("link5_collision", "link6_collision"),
):
    return tuple(
        ProtectedSampleIdentity(
            sample_id=index,
            body_id=body_id,
            body_name=body_name,
            geom_id=geom_id,
            geom_name=geom_name,
            point_body_local_m=[0.0, 0.0, 0.0],
            source="collision_geom_surface",
        )
        for index, (body_id, body_name, geom_id, geom_name) in enumerate(
            zip(body_ids, body_names, geom_ids, geom_names)
        )
    )


def protected_sampling_evidence(samples):
    identities = [sample.to_dict() for sample in samples]
    components = []
    for identity in identities:
        components.append(
            {
                "geom_id": identity["geom_id"],
                "geom_name": identity["geom_name"],
                "body_id": identity["body_id"],
                "body_name": identity["body_name"],
                "geometry_kind": "mesh",
                "certificate_kind": "triangle_barycentric_lattice",
                "surface_element_count": 1,
                "sample_count": 1,
                "certified_surface_cover_radius_m": 0.001,
            }
        )
    sampling = {
        "epsilon_m": 0.01,
        "maximum_surface_cover_radius_m": 0.001,
        "coverage_semantics": "strict_open_epsilon_surface_cover",
        "components": components,
        "samples": identities,
    }
    return {
        "protected_sample_count": len(identities),
        "surface_components": components,
        "protected_sampling_epsilon_m": sampling["epsilon_m"],
        "protected_sampling_maximum_surface_cover_radius_m": sampling[
            "maximum_surface_cover_radius_m"
        ],
        "protected_sampling_coverage_semantics": sampling[
            "coverage_semantics"
        ],
        "protected_samples": identities,
        "hashes": {
            "protected_samples_sha256": hashlib.sha256(
                json.dumps(
                    sampling,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        },
    }


def valid_differential_audit(samples, state_sha256, config=None):
    """Return a strict passing zero-motion audit and its validation receipt."""

    from main.poisson_fullbody.contracts import canonical_json_bytes, sha256_bytes
    from main.poisson_fullbody.jacobians import (
        DIFFERENTIAL_AUDIT_HASH_FIELD,
        PROTECTED_SAMPLE_DIFFERENTIAL_SCHEMA,
        _coupled_comparison,
        _point_metrics,
        _stable_audit_hashes,
        _tangent_reconstruction,
        validate_protected_sample_differential_audit,
    )

    registered = differential_audit_config() if config is None else config
    registered = json.loads(json.dumps(registered, allow_nan=False))
    identities = [
        sample.to_dict() if hasattr(sample, "to_dict") else dict(sample)
        for sample in samples
    ]
    arm_dofs = list(range(7))
    analytic = [[0.0] * 7 for _ in range(3)]
    numerical = [[0.0] * 7 for _ in range(3)]
    point_metrics = _point_metrics(analytic, numerical, registered)

    def tangent_record(requested):
        return _tangent_reconstruction(
            requested,
            requested,
            [-value for value in requested],
            arm_dofs,
            registered,
        )

    def query_record(point):
        return {
            "point_world_m": list(point),
            "valid": True,
            "value_m2": 1.0,
            "gradient_m": [0.0, 0.0, 0.0],
            "reason": None,
            "cell_index": [0, 0, 0],
            "local_coordinates": [0.5, 0.5, 0.5],
            "outer_boundary_clearance_m": 1.0,
        }

    point_tangents = []
    for column in range(7):
        requested = [0.0] * 7
        requested[column] = 1.0
        point_tangents.append(tangent_record(requested))

    records = []
    eta = registered["coupled_eta_ladder_s"][0]
    for identity in identities:
        point = [0.1, 0.2, 0.3]
        base_query = query_record(point)
        directions = []
        for direction_index, arm_direction in enumerate(
            registered["joint_velocity_directions_rad_s"]
        ):
            attempt = {
                "eta_s": eta,
                "plus_point_world_m": list(point),
                "minus_point_world_m": list(point),
                "plus_query": query_record(point),
                "minus_query": query_record(point),
                "tangent_reconstruction": tangent_record(arm_direction),
                "eligible_same_cell_stencil": True,
                "noneligible_reasons": [],
                "selected": True,
            }
            comparison = _coupled_comparison(
                attempt,
                analytic,
                base_query["gradient_m"],
                arm_direction,
                registered,
            )
            directions.append(
                {
                    "direction_index": direction_index,
                    "arm_joint_velocity_rad_s": list(arm_direction),
                    "attempts": [attempt],
                    "selected_attempt_index": 0,
                    "derived_comparison": comparison,
                    "passed": True,
                }
            )
        records.append(
            {
                "identity": identity,
                "base_point_world_m": point,
                "base_field_query": base_query,
                "analytic_point_jacobian_m_per_rad_3x7": analytic,
                "plus_points_world_m_by_arm_dof": [list(point) for _ in range(7)],
                "minus_points_world_m_by_arm_dof": [list(point) for _ in range(7)],
                "numerical_point_jacobian_m_per_rad_3x7": numerical,
                "point_jacobian_metrics": point_metrics,
                "point_perturbation_tangent_reconstruction_by_arm_dof": (
                    point_tangents
                ),
                "point_jacobian_passed": True,
                "coupled_directions": directions,
                "eligible_coupled_direction_count": 9,
                "coupled_directions_passed": True,
                "passed": True,
            }
        )
    counts = {
        "sample_count": len(records),
        "point_jacobian_passed_sample_count": len(records),
        "required_coupled_direction_count": 9 * len(records),
        "selected_coupled_direction_count": 9 * len(records),
        "passed_coupled_direction_count": 9 * len(records),
        "coupled_attempt_count": 9 * len(records),
        "noneligible_coupled_attempt_count": 0,
        "passed_sample_count": len(records),
    }
    stable = _stable_audit_hashes(
        state_sha256, arm_dofs, identities, registered, records
    )
    audit = {
        "schema_version": PROTECTED_SAMPLE_DIFFERENTIAL_SCHEMA,
        "integration_state": {
            "state_specification": "mjSTATE_INTEGRATION",
            "element_count": 7,
            "source_initial_sha256": state_sha256,
            "clone_base_sha256": state_sha256,
            "clone_final_sha256": state_sha256,
            "source_final_sha256": state_sha256,
            "source_state_unchanged": True,
            "clone_state_restored": True,
        },
        "arm_dof_indices": arm_dofs,
        "ordered_samples": identities,
        "ordered_sample_identity_sha256": sha256_bytes(
            canonical_json_bytes(identities)
        ),
        "specification_sha256": stable["specification_sha256"],
        "binding_sha256": stable["binding_sha256"],
        "classification_ledger_sha256": stable[
            "classification_ledger_sha256"
        ],
        "differential_audit_config": registered,
        "sample_records": records,
        "counts": counts,
        "passed": True,
    }
    audit[DIFFERENTIAL_AUDIT_HASH_FIELD] = sha256_bytes(
        canonical_json_bytes(audit)
    )
    audit = json.loads(json.dumps(audit, allow_nan=False))
    receipt = validate_protected_sample_differential_audit(
        audit,
        expected_samples=identities,
        expected_arm_dof_indices=arm_dofs,
        expected_integration_state_sha256=state_sha256,
        expected_differential_audit_config=registered,
    )
    return audit, receipt


class ShadowProtectedSamplingValidationTest(unittest.TestCase):
    @staticmethod
    def _digest(value):
        return hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    @classmethod
    def _rehash_protected_sampling(cls, field_bundle):
        payload = {
            "epsilon_m": field_bundle["protected_sampling_epsilon_m"],
            "maximum_surface_cover_radius_m": field_bundle[
                "protected_sampling_maximum_surface_cover_radius_m"
            ],
            "coverage_semantics": field_bundle[
                "protected_sampling_coverage_semantics"
            ],
            "components": field_bundle["surface_components"],
            "samples": field_bundle["protected_samples"],
        }
        field_bundle["hashes"]["protected_samples_sha256"] = cls._digest(
            payload
        )

    @classmethod
    def _validation_prefix(cls, field_bundle):
        action_states = ["a" * 64]
        callback_states = ["c" * 64]
        callback_ledger = [
            {
                "observation_index": 0,
                "high_level_index": 0,
                "inner_control_index": 0,
                "physics_substep_index": 0,
                "before_sha256": callback_states[0],
                "after_sha256": callback_states[0],
                "exact_array_equal": True,
            }
        ]
        return {
            "executed_action_count": 1,
            "callback_count": 1,
            "expected_callback_count": 1,
            "measurement": {
                "observed_physics_substeps": 1,
                "first_index": [0, 0, 0],
                "last_index": [0, 0, 0],
                "physical_contact_distance_semantics": (
                    "mujoco_contact_dist_le_0"
                ),
            },
            "action_boundary_state_sha256_ledger": action_states,
            "state_sequence_sha256": cls._digest(action_states),
            "terminal_simulator_state_sha256": action_states[-1],
            "observation_sequence_sha256": "b" * 64,
            "callback_state_read_only_ledger": callback_ledger,
            "callback_state_read_only_ledger_sha256": cls._digest(
                callback_ledger
            ),
            "callback_state_sequence_sha256": cls._digest(callback_states),
            "construction": {
                "physical_model": physical_model_contract_evidence(),
                "resolved_geometry": {
                    "link56_geom_ids": [5, 6],
                    "obstacle_geom_ids": [100],
                },
                "field_bundle": field_bundle,
            },
        }

    @staticmethod
    def _validation_arguments():
        return {
            "action_count": 1,
            "inner_updates_per_high_level_action": 1,
            "physics_substeps_per_inner_update": 1,
            "physics_timestep_s": 0.002,
            "contact_definition": "mujoco_contact_dist_le_0",
            "alpha_gain_per_s": 5.0,
            "static_drift_thresholds": {
                "translation_m": 1.0e-6,
                "rotation_rad": 1.0e-5,
                "surface_m": 1.0e-6,
            },
            "differential_audit_config": differential_audit_config(),
        }

    def _assert_sampling_rejected(self, field_bundle, message):
        from main.poisson_fullbody.shadow_identification import (
            ShadowIdentificationError,
            validate_shadow_replay_record,
        )

        self._rehash_protected_sampling(field_bundle)
        with self.assertRaisesRegex(ShadowIdentificationError, message):
            validate_shadow_replay_record(
                self._validation_prefix(field_bundle),
                **self._validation_arguments(),
            )

    def test_rehashed_component_radius_at_epsilon_is_rejected(self):
        field_bundle = protected_sampling_evidence(
            protected_sample_identities()
        )
        field_bundle["surface_components"][0][
            "certified_surface_cover_radius_m"
        ] = field_bundle["protected_sampling_epsilon_m"]
        self._assert_sampling_rejected(
            field_bundle,
            "protected sampling certificate is invalid",
        )

    def test_rehashed_global_maximum_must_equal_component_maximum(self):
        field_bundle = protected_sampling_evidence(
            protected_sample_identities()
        )
        field_bundle[
            "protected_sampling_maximum_surface_cover_radius_m"
        ] = 0.0005
        self._assert_sampling_rejected(
            field_bundle,
            "protected sampling certificate is invalid",
        )

    def test_rehashed_malformed_component_fields_are_rejected(self):
        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation):
                field_bundle = protected_sampling_evidence(
                    protected_sample_identities()
                )
                if mutation == "missing":
                    field_bundle["surface_components"][0].pop(
                        "certificate_kind"
                    )
                else:
                    field_bundle["surface_components"][0][
                        "unregistered_field"
                    ] = True
                self._assert_sampling_rejected(
                    field_bundle,
                    "surface component fields differ",
                )


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
        diagnostic = first["per_geom"][0][
            "minimum_observed_cbf_lhs_sample"
        ]
        self.assertEqual(
            diagnostic["point_translational_jacobian_arm_3x7"][0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(
            diagnostic["observed_grad_h_J_qdot_m2_per_s"], -1.1
        )
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
            -1,
        )
        live_next_index = observer.result(
            expected_callback_count=3,
            first_link56_contact={
                "observation_index": 1,
                "robot_geom_id": 10,
                "source_phase": "live_solver_phase_preintegration_geometry",
            },
        )
        self.assertEqual(
            live_next_index["contact_prediction_assessment"]["assessment"],
            "registered_warning_coincident_with_link56_contact",
        )
        self.assertEqual(
            live_next_index["contact_prediction_assessment"][
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

    def test_serialized_validator_accepts_actual_observer_invalidation_trace(self):
        from main.poisson_fullbody.shadow_identification import (
            validate_shadow_replay_record,
        )

        class Field:
            def query(_self, point):
                return SimpleNamespace(
                    valid=True,
                    value=float(point[0]),
                    gradient=(1.0, 0.0, 0.0),
                    reason=None,
                )

        thresholds = {
            "translation_m": 1e-6,
            "rotation_rad": 1e-5,
            "surface_m": 1e-6,
        }
        observer = StaticPoissonShadowObserver(
            field=Field(),
            samples=self.samples,
            settled_obstacle_boxes=(self.box,),
            arm_dof_indices=range(7),
            alpha_gain_per_s=5.0,
            physics_timestep_s=0.002,
            drift_thresholds=StaticDriftThresholds(
                thresholds["translation_m"],
                thresholds["rotation_rad"],
                thresholds["surface_m"],
            ),
            physics_substeps_per_high_level_action=3,
            snapshot_provider=self._snapshot,
            jacobian_provider=self._jacobian,
        )
        observer.observe(object(), high_level_index=0, physics_substep_index=0)
        observer.observe(object(), high_level_index=0, physics_substep_index=1)
        self.data.geom_xpos[0, 0] = 2e-6
        observer.observe(object(), high_level_index=0, physics_substep_index=2)
        identification = observer.result(
            expected_callback_count=3, first_link56_contact=None
        )
        settled = {
            "candidate_contact_point_record_count": 0,
            "candidate_contact_point_records": [],
            "physical_contact_point_record_count": 0,
            "physical_contact_point_records": [],
        }
        measurement = {
            "observed_physics_substeps": 3,
            "first_index": [0, 0, 0],
            "last_index": [0, 0, 2],
            "any_robot_obstacle_contact": False,
            "link56_obstacle_contact": False,
            "rollout_any_robot_obstacle_contact": False,
            "rollout_link56_obstacle_contact": False,
            "post_state_any_robot_obstacle_contact": False,
            "post_state_link56_obstacle_contact": False,
            "live_solver_any_robot_obstacle_contact": False,
            "live_solver_link56_obstacle_contact": False,
            "total_candidate_contact_point_record_count": 0,
            "total_physical_contact_point_record_count": 0,
            "rollout_phase_physical_contact_point_record_count": 0,
            "post_state_candidate_contact_point_record_count": 0,
            "post_state_physical_contact_point_record_count": 0,
            "live_solver_candidate_contact_point_record_count": 0,
            "live_solver_nonpositive_contact_point_record_count": 0,
            "settled_state": settled,
            "live_solver_phase_contact_point_records": [],
            "post_state_candidate_contact_point_records": [],
            "post_state_physical_contact_point_records": [],
            "physical_contact_distance_semantics": "mujoco_contact_dist_le_0",
        }
        action_state_hashes = ["a" * 64]
        callback_state_hashes = ["b" * 64] * 3
        callback_state_ledger = [
            {
                "observation_index": index,
                "high_level_index": 0,
                "inner_control_index": 0,
                "physics_substep_index": index,
                "before_sha256": callback_state_hashes[index],
                "after_sha256": callback_state_hashes[index],
                "exact_array_equal": True,
            }
            for index in range(3)
        ]
        audit_config = differential_audit_config()
        field_sampling = protected_sampling_evidence(self.samples)
        settled_state_sha256 = "d" * 64
        differential_audit, differential_validation = valid_differential_audit(
            self.samples,
            settled_state_sha256,
            audit_config,
        )

        def digest(value):
            return hashlib.sha256(
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()

        shadow = {
            "executed_action_count": 1,
            "callback_count": 3,
            "expected_callback_count": 3,
            "action_boundary_state_sha256_ledger": action_state_hashes,
            "state_sequence_sha256": digest(action_state_hashes),
            "observation_sequence_sha256": "c" * 64,
            "callback_state_read_only_ledger": callback_state_ledger,
            "callback_state_read_only_ledger_sha256": digest(
                callback_state_ledger
            ),
            "callback_state_sequence_sha256": digest(callback_state_hashes),
            "terminal_simulator_state_sha256": action_state_hashes[-1],
            "construction": {
                "physical_model": physical_model_contract_evidence(),
                "resolved_geometry": {
                    "robot_body_ids": [0, 1],
                    "robot_body_names": ["robot0_link5", "robot0_link6"],
                    "obstacle_body_ids": [2],
                    "obstacle_body_names": ["obstacle_body"],
                    "robot_geom_ids": [10, 11],
                    "robot_geom_names": [
                        "link5_collision",
                        "link6_collision",
                    ],
                    "link56_geom_ids": [10, 11],
                    "link56_geom_names": [
                        "link5_collision",
                        "link6_collision",
                    ],
                    "obstacle_geom_ids": [0],
                    "obstacle_geom_names": ["obstacle_geom"],
                    "collision_enabled_pairs": [[10, 0], [11, 0]],
                },
                "field_bundle": field_sampling,
                "static_field_drift_thresholds": thresholds,
                "cbf_alpha_gain_per_s": 5.0,
                "arm_dof_indices": list(range(7)),
                "settled_link56_differential_audit": differential_audit,
                "settled_link56_differential_audit_validation": (
                    differential_validation
                ),
                "settled_measurement": settled,
                "complete_integration_state_read_only_audit": {
                    "exact_array_equal": True,
                    "before_sha256": settled_state_sha256,
                    "after_sha256": settled_state_sha256,
                },
            },
            "measurement": measurement,
            "monitor_static_drift_exception_count": 0,
            "monitor_static_drift_exceptions": [],
            "poisson_identification": identification,
        }
        serialized = json.loads(json.dumps(shadow, allow_nan=False))
        validate_shadow_replay_record(
            serialized,
            action_count=1,
            inner_updates_per_high_level_action=1,
            physics_substeps_per_inner_update=3,
            physics_timestep_s=0.002,
            contact_definition="mujoco_contact_dist_le_0",
            alpha_gain_per_s=5.0,
            static_drift_thresholds=thresholds,
            differential_audit_config=audit_config,
        )

    def test_robot_roots_are_derived_from_authoritative_parent_ids(self):
        parents = (0, 0, 1, 2, 0, 4)
        self.assertEqual(robot_root_body_ids((1, 2, 3, 4, 5), parents), (1, 4))


if __name__ == "__main__":
    unittest.main()
