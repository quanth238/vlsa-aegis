from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from tests.test_poisson_shadow_identification import (
    ProtectedSampleIdentity,
    differential_audit_config,
    physical_model_contract_evidence,
    protected_sample_identities,
    protected_sampling_evidence,
    valid_differential_audit,
)


ROOT = Path(__file__).resolve().parents[1]


class ShadowIdentificationRunnerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = (
            ROOT / "scripts/run_poisson_shadow_identification.py"
        ).read_text(encoding="utf-8")
        cls.module = (
            ROOT / "main/poisson_fullbody/shadow_identification.py"
        ).read_text(encoding="utf-8")
        cls.batch = (
            ROOT / "slurm/poisson_shadow_identification.sbatch"
        ).read_text(encoding="utf-8")
        cls.active = (
            ROOT / "scripts/run_poisson_active_canary.py"
        ).read_text(encoding="utf-8")

    def test_sources_parse_and_stage_is_bound_to_exact_parity(self):
        ast.parse(self.runner)
        ast.parse(self.module)
        self.assertIn("--parity-result", self.runner)
        self.assertIn(
            "upstream exact-parity source is not this exact clean commit",
            self.runner,
        )
        self.assertIn("ordinary_and_callback_observations_exact", self.runner)
        self.assertIn("state sequence differs from upstream exact parity", self.runner)

    def test_failed_differential_audit_retains_deep_receipt_without_authorizing_success(self):
        from main.poisson_fullbody.jacobians import (
            inspect_protected_sample_differential_audit,
        )
        from scripts import run_poisson_shadow_identification as runner
        from tests.test_poisson_jacobians import (
            _strict_failed_differential_audit_fixture,
        )

        audit, samples, state_sha256, config = (
            _strict_failed_differential_audit_fixture()
        )
        receipt = inspect_protected_sample_differential_audit(
            audit,
            expected_samples=samples,
            expected_arm_dof_indices=list(range(7)),
            expected_integration_state_sha256=state_sha256,
            expected_differential_audit_config=config,
        )
        evidence = runner._differential_audit_failure_evidence(
            audit, receipt
        )
        self.assertEqual(
            set(evidence),
            {
                "schema_version",
                "phase",
                "failure_kind",
                "scientific_result",
                "active_physics_authorized",
                "settled_link56_differential_audit",
                "settled_link56_differential_audit_validation",
                "interpretation",
            },
        )
        self.assertEqual(
            evidence["schema_version"],
            runner.DIFFERENTIAL_AUDIT_FAILURE_EVIDENCE_SCHEMA,
        )
        self.assertEqual(evidence["phase"], runner.DIFFERENTIAL_AUDIT_FAILURE_PHASE)
        self.assertIs(evidence["scientific_result"], False)
        self.assertIs(evidence["active_physics_authorized"], False)
        self.assertEqual(evidence["settled_link56_differential_audit"], audit)
        self.assertEqual(
            evidence["settled_link56_differential_audit_validation"], receipt
        )
        self.assertIs(receipt["passed"], False)

        mismatched = dict(receipt)
        mismatched["audit_payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            runner.ShadowIdentificationRunnerError, "differs from audit"
        ):
            runner._differential_audit_failure_evidence(audit, mismatched)

        passing = dict(receipt)
        passing["passed"] = True
        with self.assertRaises(runner.ShadowIdentificationRunnerError):
            runner._differential_audit_failure_evidence(audit, passing)

    def test_failed_audit_authority_binding_requires_source_inputs_and_slurm(self):
        from scripts import run_poisson_shadow_identification as runner

        provenance = {
            "source": {
                "commit": "a" * 40,
                "branch": "codex/full-body-poisson-cbf-feasibility",
                "status_short": [],
            },
            "manifest_sha256": "b" * 64,
            "manifest_row_sha256": "c" * 64,
            "selection_config_sha256": "d" * 64,
            "runtime_protocol_raw_sha256": "e" * 64,
            "runtime_protocol_semantic_sha256": "f" * 64,
            "runtime_parameter_block_sha256": "0" * 64,
            "historical_result_file_sha256": "1" * 64,
            "historical_result_payload_sha256": "2" * 64,
            "upstream_parity_payload_sha256": "3" * 64,
            "host": "worker",
            "slurm_job_id": "33712",
            "slurm_job_name": "poisson-identification",
        }
        binding = runner._failure_authority_binding(provenance)
        self.assertEqual(binding["source"], provenance["source"])
        self.assertEqual(binding["slurm_job_id"], "33712")
        self.assertEqual(
            binding["historical_result_payload_sha256"], "2" * 64
        )
        missing = dict(provenance)
        missing.pop("upstream_parity_payload_sha256")
        with self.assertRaisesRegex(
            runner.ShadowIdentificationRunnerError,
            "invalid immutable input or Slurm provenance",
        ):
            runner._failure_authority_binding(missing)
        malformed_source = dict(provenance)
        malformed_source["source"] = {
            "commit": "not-a-commit",
            "branch": "codex/full-body-poisson-cbf-feasibility",
            "status_short": [],
        }
        with self.assertRaisesRegex(
            runner.ShadowIdentificationRunnerError,
            "lacks exact clean source provenance",
        ):
            runner._failure_authority_binding(malformed_source)
        malformed_slurm = dict(provenance)
        malformed_slurm["slurm_job_id"] = "33712.batch"
        with self.assertRaisesRegex(
            runner.ShadowIdentificationRunnerError,
            "invalid immutable input or Slurm provenance",
        ):
            runner._failure_authority_binding(malformed_slurm)

    def test_failed_audit_is_recorded_with_exact_phase_and_authority(self):
        from main.poisson_fullbody.jacobians import (
            inspect_protected_sample_differential_audit,
        )
        from scripts import run_poisson_shadow_identification as runner
        from tests.test_poisson_jacobians import (
            _strict_failed_differential_audit_fixture,
        )

        audit, samples, state_sha256, config = (
            _strict_failed_differential_audit_fixture()
        )
        receipt = inspect_protected_sample_differential_audit(
            audit,
            expected_samples=samples,
            expected_arm_dof_indices=list(range(7)),
            expected_integration_state_sha256=state_sha256,
            expected_differential_audit_config=config,
        )
        evidence = runner._differential_audit_failure_evidence(
            audit, receipt
        )
        provenance = {
            "source": {
                "commit": "a" * 40,
                "branch": "codex/full-body-poisson-cbf-feasibility",
                "status_short": [],
            },
            "manifest_sha256": "b" * 64,
            "manifest_row_sha256": "c" * 64,
            "selection_config_sha256": "d" * 64,
            "runtime_protocol_raw_sha256": "e" * 64,
            "runtime_protocol_semantic_sha256": "f" * 64,
            "runtime_parameter_block_sha256": "0" * 64,
            "historical_result_file_sha256": "1" * 64,
            "historical_result_payload_sha256": "2" * 64,
            "upstream_parity_payload_sha256": "3" * 64,
            "host": "worker",
            "slurm_job_id": "33712",
            "slurm_job_name": "poisson-identification",
        }
        payload = {
            "status": "passed",
            "scientific_result": True,
            "phase": "prepare_shadow_runtime_and_exact_replay",
            "provenance": provenance,
        }
        error = runner.ShadowIdentificationDifferentialAuditFailure(evidence)
        runner._record_failed_result(payload, error, "registered traceback")
        self.assertEqual(payload["status"], "failed")
        self.assertIs(payload["scientific_result"], False)
        self.assertEqual(payload["phase"], runner.DIFFERENTIAL_AUDIT_FAILURE_PHASE)
        self.assertEqual(
            payload["failure_evidence"][
                "settled_link56_differential_audit"
            ],
            audit,
        )
        self.assertEqual(
            payload["failure_evidence"]["authority_binding"]["slurm_job_id"],
            "33712",
        )
        self.assertIs(
            payload["failure_evidence"]["active_physics_authorized"], False
        )
        self.assertEqual(payload["failure"]["traceback"], "registered traceback")

    def test_generic_failure_never_serializes_differential_audit_evidence(self):
        from scripts import run_poisson_shadow_identification as runner

        payload = {
            "status": "failed",
            "scientific_result": False,
            "phase": "prepare_shadow_runtime_and_exact_replay",
        }
        error = runner.ShadowIdentificationRunnerError(
            "source or clone state was not preserved"
        )
        runner._record_failed_result(payload, error, "registered traceback")
        self.assertEqual(payload["status"], "failed")
        self.assertNotIn("failure_evidence", payload)
        self.assertEqual(
            payload["failure"]["type"], "ShadowIdentificationRunnerError"
        )

    def test_exact_historical_replay_and_complete_callback_exposure(self):
        self.assertIn("len(replay.steps) != 237", self.runner)
        self.assertIn("INNER_UPDATES_PER_HIGH_LEVEL_ACTION = 5", self.runner)
        self.assertIn("PHYSICS_SUBSTEPS_PER_INNER_UPDATE = 5", self.runner)
        self.assertIn(
            "expected_substeps=PHYSICS_SUBSTEPS_PER_HIGH_LEVEL_ACTION",
            self.runner,
        )
        self.assertIn("_require_complete_measurement_exposure(", self.runner)
        self.assertIn("measurement.observed_physics_substeps", self.runner)
        self.assertIn("measurement.first_index", self.runner)
        self.assertIn("measurement.last_index", self.runner)
        self.assertIn("_check_step", self.runner)
        self.assertIn("reward=reward", self.runner)
        self.assertIn("previous_goal_values=previous_goal", self.runner)

    def test_complete_measurement_exposure_uses_typed_cadence_tuples(self):
        from scripts.run_poisson_shadow_identification import (
            ShadowIdentificationRunnerError,
            _require_complete_measurement_exposure,
        )

        valid = SimpleNamespace(
            observed_physics_substeps=50,
            first_index=(0, 0, 0),
            last_index=(1, 4, 4),
        )
        self.assertEqual(
            _require_complete_measurement_exposure(valid, action_count=2), 50
        )

        invalid = (
            SimpleNamespace(
                observed_physics_substeps=49,
                first_index=(0, 0, 0),
                last_index=(1, 4, 4),
            ),
            # Regression for H100 job 33451: these fields are tuples and must
            # never be coerced with int().
            SimpleNamespace(
                observed_physics_substeps=50,
                first_index=0,
                last_index=49,
            ),
            SimpleNamespace(
                observed_physics_substeps=50,
                first_index=[0, 0, 0],
                last_index=(1, 4, 4),
            ),
            SimpleNamespace(
                observed_physics_substeps=50,
                first_index=(0, 0, 0),
                last_index=(1, 4, 3),
            ),
        )
        for measurement in invalid:
            with self.subTest(measurement=measurement):
                with self.assertRaisesRegex(
                    ShadowIdentificationRunnerError,
                    "did not cover every physics callback|cadence indices",
                ):
                    _require_complete_measurement_exposure(
                        measurement, action_count=2
                    )

        with self.assertRaisesRegex(
            ShadowIdentificationRunnerError, "callback count is not an integer"
        ):
            _require_complete_measurement_exposure(
                SimpleNamespace(
                    observed_physics_substeps=50.0,
                    first_index=(0, 0, 0),
                    last_index=(1, 4, 4),
                ),
                action_count=2,
            )
        for invalid_action_count in (True, 0, -1, 2.0):
            with self.subTest(invalid_action_count=invalid_action_count):
                with self.assertRaisesRegex(
                    ShadowIdentificationRunnerError,
                    "action_count must be a positive integer",
                ):
                    _require_complete_measurement_exposure(
                        valid, action_count=invalid_action_count
                    )

    def test_contact_lead_uses_physical_phase_boundary(self):
        from main.poisson_fullbody.shadow_identification import (
            registered_filter_update_available_before_contact,
            rollout_contact_boundary_index,
        )

        self.assertEqual(
            rollout_contact_boundary_index(
                21, "live_solver_phase_preintegration_geometry"
            ),
            20,
        )
        self.assertEqual(
            rollout_contact_boundary_index(20, "post_integration_recomputed"),
            20,
        )
        self.assertTrue(
            registered_filter_update_available_before_contact(
                warning_observation_index=10,
                contact_observation_index=21,
                contact_source_phase="live_solver_phase_preintegration_geometry",
                physics_substeps_per_filter_update=5,
            )
        )
        self.assertFalse(
            registered_filter_update_available_before_contact(
                warning_observation_index=18,
                contact_observation_index=20,
                contact_source_phase="live_solver_phase_preintegration_geometry",
                physics_substeps_per_filter_update=5,
            )
        )
        self.assertIn("rollout_contact_boundary_index(", self.active)
        self.assertIn(
            "registered_filter_update_available_before_contact(", self.active
        )

    def test_upstream_parity_binds_clean_official_callback_state_ledger(self):
        from scripts import run_poisson_shadow_identification as runner
        from scripts import run_poisson_shadow_parity as parity

        callback_hashes = ["a" * 64] * 50
        action_state_hashes = ["b" * 64, "d" * 64]
        state_sequence_sha256 = parity._sha256(
            parity._canonical(action_state_hashes)
        )
        substep_trace = [
            [
                {
                    "substep": substep_index,
                    "official_mjstate_integration_sha256": "a" * 64,
                }
                for substep_index in range(25)
            ]
            for _ in range(2)
        ]
        historical = {
            "identity": "historical",
            "terminal_simulator_state_sha256": action_state_hashes[-1],
        }
        record = {
            "schema_version": runner.EXPECTED_PARITY_SCHEMA,
            "status": "passed",
            "scientific_result": False,
            "case_id": runner.DEFAULT_CASE_ID,
            "provenance": {
                "source": {"commit": "source-commit", "status_short": []},
                "historical_result_payload_sha256": "historical-payload",
                "manifest_sha256": "manifest",
                "manifest_row_sha256": "row",
            },
            "historical": historical,
            "ordinary_replay": {
                "executed_action_count": 2,
                "expected_state_match_count": 2,
                "action_boundary_state_sha256_ledger": action_state_hashes,
                "state_sequence_sha256": state_sequence_sha256,
                "observation_sequence_sha256": "c" * 64,
                "terminal_simulator_state_sha256": "d" * 64,
            },
            "callback_replay": {
                "executed_action_count": 2,
                "callback_count": 50,
                "expected_callback_count": 50,
                "action_boundary_state_sha256_ledger": action_state_hashes,
                "state_sequence_sha256": state_sequence_sha256,
                "observation_sequence_sha256": "c" * 64,
                "terminal_simulator_state_sha256": "d" * 64,
                "substep_trace": substep_trace,
                "substep_trace_sha256": parity._sha256(
                    parity._canonical(substep_trace)
                ),
                "official_integration_state_count": 50,
                "official_integration_state_sha256_ledger": callback_hashes,
                "official_integration_state_sequence_sha256": parity._sha256(
                    parity._canonical(callback_hashes)
                ),
                "first_substep": substep_trace[0][0],
                "last_substep": substep_trace[-1][-1],
            },
            "acceptance": {
                "all_historical_post_step_states_exact": True,
                "ordinary_and_callback_states_exact": True,
                "ordinary_and_callback_observations_exact": True,
                "reward_done_goal_exact": True,
                "full_callback_exposure": True,
                "ordinary_step_path_unmodified": True,
            },
        }
        arguments = {
            "case_id": runner.DEFAULT_CASE_ID,
            "source_commit": "source-commit",
            "historical_payload_sha256": "historical-payload",
            "manifest_sha256": "manifest",
            "manifest_row_sha256": "row",
            "expected_callback_count": 50,
            "historical_provenance": historical,
            "expected_action_state_sha256_ledger": action_state_hashes,
        }
        with mock.patch(
            "main.poisson_fullbody.contracts.load_hashed_json",
            return_value=record,
        ):
            self.assertIs(
                runner._require_upstream_parity(
                    Path("unused-parity.json"), **arguments
                ),
                record,
            )

        tampered_records = []
        dirty = json.loads(json.dumps(record))
        dirty["provenance"]["source"]["status_short"] = [" M file.py"]
        tampered_records.append(dirty)
        wrong_ledger = json.loads(json.dumps(record))
        wrong_ledger["callback_replay"][
            "official_integration_state_sha256_ledger"
        ][0] = "f" * 64
        tampered_records.append(wrong_ledger)
        wrong_historical = json.loads(json.dumps(record))
        wrong_historical["historical"] = {"identity": "other"}
        tampered_records.append(wrong_historical)
        wrong_trace = json.loads(json.dumps(record))
        wrong_trace["callback_replay"]["substep_trace"][0][0][
            "substep"
        ] = 1
        tampered_records.append(wrong_trace)
        legacy_trace = json.loads(json.dumps(record))
        legacy_trace["callback_replay"]["substep_trace"][0][7][
            "simulator_state_sha256"
        ] = "f" * 64
        legacy_trace["callback_replay"][
            "substep_trace_sha256"
        ] = parity._sha256(
            parity._canonical(
                legacy_trace["callback_replay"]["substep_trace"]
            )
        )
        tampered_records.append(legacy_trace)
        wrong_action_ledger = json.loads(json.dumps(record))
        wrong_action_ledger["ordinary_replay"][
            "action_boundary_state_sha256_ledger"
        ][0] = "0" * 64
        wrong_action_ledger["ordinary_replay"][
            "state_sequence_sha256"
        ] = parity._sha256(
            parity._canonical(
                wrong_action_ledger["ordinary_replay"][
                    "action_boundary_state_sha256_ledger"
                ]
            )
        )
        tampered_records.append(wrong_action_ledger)
        collusive_middle = json.loads(json.dumps(record))
        for arm in ("ordinary_replay", "callback_replay"):
            collusive_middle[arm][
                "action_boundary_state_sha256_ledger"
            ][0] = "e" * 64
            collusive_middle[arm][
                "state_sequence_sha256"
            ] = parity._sha256(
                parity._canonical(
                    collusive_middle[arm][
                        "action_boundary_state_sha256_ledger"
                    ]
                )
            )
        tampered_records.append(collusive_middle)
        collusive_terminal = json.loads(json.dumps(record))
        for arm in ("ordinary_replay", "callback_replay"):
            collusive_terminal[arm][
                "action_boundary_state_sha256_ledger"
            ][-1] = "e" * 64
            collusive_terminal[arm][
                "state_sequence_sha256"
            ] = parity._sha256(
                parity._canonical(
                    collusive_terminal[arm][
                        "action_boundary_state_sha256_ledger"
                    ]
                )
            )
            collusive_terminal[arm][
                "terminal_simulator_state_sha256"
            ] = "e" * 64
        tampered_records.append(collusive_terminal)
        for tampered in tampered_records:
            with self.subTest(tampered=tampered), mock.patch(
                "main.poisson_fullbody.contracts.load_hashed_json",
                return_value=tampered,
            ):
                with self.assertRaises(runner.ShadowIdentificationRunnerError):
                    runner._require_upstream_parity(
                        Path("unused-parity.json"), **arguments
                    )

    @mock.patch(
        "main.poisson_fullbody.shadow_identification.BodySample",
        ProtectedSampleIdentity,
    )
    def test_fake_complete_replay_executes_the_postrun_path(self):
        from scripts import run_poisson_shadow_identification as runner
        from scripts import run_poisson_shadow_parity as parity

        settled_measurement = {
            "candidate_contact_point_record_count": 0,
            "candidate_contact_point_records": [],
            "physical_contact_point_record_count": 0,
            "physical_contact_point_records": [],
        }

        class FakeMonitor:
            def __init__(self):
                self.indices = []

            def observe_post_integration(
                self,
                sim,
                *,
                high_level_index,
                inner_control_index,
                physics_substep_index,
            ):
                self.indices.append(
                    (
                        high_level_index,
                        inner_control_index,
                        physics_substep_index,
                    )
                )

            def result(self):
                record = {
                    "observed_physics_substeps": len(self.indices),
                    "first_index": self.indices[0],
                    "last_index": self.indices[-1],
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
                    "settled_state": settled_measurement,
                    "live_solver_phase_contact_point_records": [],
                    "post_state_candidate_contact_point_records": [],
                    "post_state_physical_contact_point_records": [],
                    "physical_contact_distance_semantics": (
                        runner.CONTACT_DEFINITION
                    ),
                }
                return SimpleNamespace(
                    observed_physics_substeps=len(self.indices),
                    first_index=self.indices[0],
                    last_index=self.indices[-1],
                    physical_contact_distance_semantics=(
                        runner.CONTACT_DEFINITION
                    ),
                    settled_state=SimpleNamespace(
                        physical_contact_point_records=()
                    ),
                    live_solver_phase_contact_point_records=(),
                    post_state_physical_contact_point_records=(),
                    to_dict=lambda: record,
                )

        class FakeObserver:
            def __init__(self):
                self.indices = []

            def observe(
                self, sim, *, high_level_index, physics_substep_index
            ):
                self.indices.append((high_level_index, physics_substep_index))

            def result(
                self, *, expected_callback_count, first_link56_contact
            ):
                if len(self.indices) != expected_callback_count:
                    raise AssertionError("fake observer callback count differs")
                if first_link56_contact is not None:
                    raise AssertionError("fake replay unexpectedly found contact")
                trace = []
                for observation_index, (
                    high_level_index,
                    physics_substep_index,
                ) in enumerate(self.indices):
                    groups = []
                    for sample_id, geom_id, geom_name, body_id, body_name in (
                        (0, 5, "link5_collision", 50, "robot0_link5"),
                        (1, 6, "link6_collision", 60, "robot0_link6"),
                    ):
                        diagnostic = {
                            "sample_id": sample_id,
                            "body_id": body_id,
                            "body_name": body_name,
                            "geom_id": geom_id,
                            "geom_name": geom_name,
                            "point_world_m": [0.1, 0.2, 0.3],
                            "h_m2": 0.2,
                            "gradient_world_m": [1.0, 0.0, 0.0],
                            "gradient_norm_m": 1.0,
                            "observed_arm_qvel_rad_s": [0.0] * 7,
                            "point_translational_jacobian_arm_3x7": [
                                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            ],
                            "observed_grad_h_J_qdot_m2_per_s": 0.0,
                            "alpha_h_m2_per_s": 1.0,
                            "observed_cbf_lhs_m2_per_s": 1.0,
                            "observed_cbf_diagnostic_semantics": (
                                "post-integration instantaneous MuJoCo qvel"
                            ),
                        }
                        groups.append(
                            {
                                "geom_id": geom_id,
                                "geom_name": geom_name,
                                "body_id": body_id,
                                "body_name": body_name,
                                "sample_count": 1,
                                "valid_query_count": 1,
                                "invalid_query_count": 0,
                                "observed_cbf_lhs_evaluation_count": 1,
                                "invalid_reason_counts": {},
                                "first_invalid_sample": None,
                                "first_invalid_sample_by_reason": {},
                                "minimum_h_sample": dict(diagnostic),
                                "minimum_observed_cbf_lhs_sample": dict(
                                    diagnostic
                                ),
                            }
                        )
                    trace.append(
                        {
                            "observation_index": observation_index,
                            "high_level_index": high_level_index,
                            "physics_substep_index": physics_substep_index,
                            "time_from_first_callback_s": (
                                observation_index * 0.002
                            ),
                            "time_from_settled_state_s": (
                                (observation_index + 1) * 0.002
                            ),
                            "static_field_admissible_for_this_query": True,
                            "field_query_attempted": True,
                            "drift": {
                                "maximum_translation_m": 0.0,
                                "maximum_rotation_rad": 0.0,
                                "maximum_surface_point_displacement_m": 0.0,
                                "geoms": [
                                    {
                                        "geom_id": 100,
                                        "translation_m": 0.0,
                                        "rotation_rad": 0.0,
                                        "maximum_surface_point_displacement_m": 0.0,
                                        "threshold_crossing_reasons": [],
                                    }
                                ],
                            },
                            "sample_count": 2,
                            "valid_query_count": 2,
                            "invalid_query_count": 0,
                            "observed_cbf_lhs_evaluation_count": 2,
                            "invalid_reason_counts": {},
                            "minimum_h_m2": 0.2,
                            "minimum_h_sample": dict(
                                groups[0]["minimum_h_sample"]
                            ),
                            "minimum_observed_cbf_lhs_m2_per_s": 1.0,
                            "minimum_observed_cbf_lhs_sample": dict(
                                groups[0][
                                    "minimum_observed_cbf_lhs_sample"
                                ]
                            ),
                            "per_geom": groups,
                        }
                    )
                return {
                    "observed_callback_count": len(self.indices),
                    "expected_callback_count": expected_callback_count,
                    "field_query_callback_count": len(self.indices),
                    "field_query_skipped_after_static_invalidation_count": 0,
                    "first_static_field_invalidation": None,
                    "valid_field_query_count": 2 * len(self.indices),
                    "observed_cbf_lhs_evaluation_count": 2 * len(self.indices),
                    "every_valid_query_has_observed_cbf_lhs": True,
                    "trace_sha256": parity._sha256(parity._canonical(trace)),
                    "trace": trace,
                    "signals": {
                        "first_any_fail_closed_query_on_contact_geom": None,
                        "first_observed_minimum_cbf_lhs_negative_on_contact_geom": None,
                    },
                    "contact_prediction_assessment": {
                        "assessment": "no_link56_contact_outcome",
                        "first_link56_contact": None,
                        "primary_registered_warning": None,
                        "lead_physics_substeps": None,
                        "lead_time_s": None,
                    },
                }

        class FakeEnv:
            def __init__(self):
                self.sim = object()
                self.step_index = 0
                self.closed = False

            def step_with_substep_callback(
                self, action, callback, *, expected_substeps
            ):
                for substep in range(expected_substeps):
                    callback(self.sim, substep)
                current = self.step_index
                self.step_index += 1
                return {"step": current}, 0.0, current == 1, {}

            def close(self):
                self.closed = True

        env = FakeEnv()
        monitor = FakeMonitor()
        observer = FakeObserver()
        state_hashes = ["a" * 64, "b" * 64]
        replay_steps = (
            SimpleNamespace(
                step=0,
                action=(0.0,) * 7,
                simulator_state_sha256=state_hashes[0],
            ),
            SimpleNamespace(
                step=1,
                action=(0.0,) * 7,
                simulator_state_sha256=state_hashes[1],
            ),
        )
        replay_provenance = {"identity": "fake-historical-replay"}
        replay = SimpleNamespace(
            steps=replay_steps,
            actions=tuple(step.action for step in replay_steps),
            result_file_sha256="historical-file",
            result_payload_sha256="historical-payload",
            terminal_simulator_state_sha256=state_hashes[-1],
            provenance=lambda: replay_provenance,
        )
        observation_hashes = ["observation-0", "observation-1"]
        upstream = {
            "result_payload_sha256": "parity-payload",
            "callback_replay": {
                "action_boundary_state_sha256_ledger": state_hashes,
                "state_sequence_sha256": parity._sha256(
                    parity._canonical(state_hashes)
                ),
                "observation_sequence_sha256": parity._sha256(
                    parity._canonical(observation_hashes)
                ),
                "terminal_simulator_state_sha256": state_hashes[-1],
                "official_integration_state_sha256_ledger": [
                    "c" * 64
                ] * 50,
                "official_integration_state_sequence_sha256": parity._sha256(
                    parity._canonical(["c" * 64] * 50)
                ),
            }
        }
        evaluator = SimpleNamespace(
            array_sha256=lambda state: "c" * 64,
            _eef_proxy=lambda runtime, observation: None,
            _update_eef_marker=lambda environment, proxy: None,
        )
        runtime = {
            "np": SimpleNamespace(array_equal=lambda left, right: left == right)
        }

        def fake_check_step(**keywords):
            index = int(keywords["expected_step"].step)
            return (
                state_hashes[index],
                observation_hashes[index],
                keywords["previous_goal_values"],
            )

        audit_config = differential_audit_config()
        protected_samples = protected_sample_identities()
        field_sampling = protected_sampling_evidence(protected_samples)
        settled_state_sha256 = "c" * 64
        differential_audit, differential_validation = valid_differential_audit(
            protected_samples,
            settled_state_sha256,
            audit_config,
        )
        prepared = (
            env,
            object(),
            {"step": -1},
            (),
            (),
            monitor,
            observer,
            {
                "resolved_geometry": {
                    "robot_body_ids": [50, 60],
                    "robot_body_names": ["robot0_link5", "robot0_link6"],
                    "obstacle_body_ids": [1000],
                    "obstacle_body_names": ["moka_pot_body"],
                    "robot_geom_ids": [5, 6],
                    "robot_geom_names": [
                        "link5_collision",
                        "link6_collision",
                    ],
                    "link56_geom_ids": [5, 6],
                    "link56_geom_names": [
                        "link5_collision",
                        "link6_collision",
                    ],
                    "obstacle_geom_ids": [100],
                    "obstacle_geom_names": ["moka_pot_geom"],
                    "collision_enabled_pairs": [[5, 100], [6, 100]],
                },
                "physical_model": physical_model_contract_evidence(),
                "field_bundle": field_sampling,
                "static_field_drift_thresholds": {
                    "translation_m": 1e-6,
                    "rotation_rad": 1e-5,
                    "surface_m": 1e-6,
                },
                "cbf_alpha_gain_per_s": 5.0,
                "arm_dof_indices": list(range(7)),
                "settled_link56_differential_audit": differential_audit,
                "settled_link56_differential_audit_validation": (
                    differential_validation
                ),
                "settled_measurement": settled_measurement,
                "complete_integration_state_read_only_audit": {
                    "exact_array_equal": True,
                    "before_sha256": settled_state_sha256,
                    "after_sha256": settled_state_sha256,
                },
            },
        )
        with mock.patch.object(
            runner, "_prepare_shadow_runtime", return_value=prepared
        ), mock.patch.object(
            runner, "_official_integration_state", return_value=(1.0, 2.0)
        ), mock.patch.object(parity, "_check_step", side_effect=fake_check_step):
            result = runner._run_shadow(
                evaluator=evaluator,
                runtime=runtime,
                case={},
                replay=replay,
                protocol={
                    "cbf": {"alpha_gain_per_s": 5.0},
                    "admissibility": {
                        "max_selected_geom_translation_drift_m": 1e-6,
                        "max_selected_geom_rotation_drift_rad": 1e-5,
                        "max_selected_geom_surface_drift_m": 1e-6,
                    },
                    "differential_audit": audit_config,
                },
                protocol_hashes=SimpleNamespace(),
                upstream_parity=upstream,
            )

        self.assertTrue(env.closed)
        self.assertEqual(result["executed_action_count"], 2)
        self.assertEqual(result["callback_count"], 50)
        self.assertEqual(result["expected_callback_count"], 50)
        self.assertEqual(
            result["action_boundary_state_sha256_ledger"], state_hashes
        )
        self.assertEqual(len(result["callback_state_read_only_ledger"]), 50)
        self.assertTrue(
            all(
                row["before_sha256"] == row["after_sha256"] == "c" * 64
                and row["exact_array_equal"] is True
                for row in result["callback_state_read_only_ledger"]
            )
        )
        self.assertEqual(
            result["terminal_simulator_state_sha256"], state_hashes[-1]
        )
        self.assertEqual(result["measurement"]["first_index"], (0, 0, 0))
        self.assertEqual(result["measurement"]["last_index"], (1, 4, 4))
        self.assertEqual(monitor.indices[0], (0, 0, 0))
        self.assertEqual(monitor.indices[-1], (1, 4, 4))
        self.assertEqual(
            result["state_sequence_sha256"],
            upstream["callback_replay"]["state_sequence_sha256"],
        )
        self.assertEqual(
            result["observation_sequence_sha256"],
            upstream["callback_replay"]["observation_sequence_sha256"],
        )
        construction = result["construction"]
        self.assertEqual(
            construction["physical_model"], physical_model_contract_evidence()
        )
        self.assertEqual(
            construction["settled_link56_differential_audit"],
            differential_audit,
        )
        self.assertEqual(
            construction["settled_link56_differential_audit_validation"],
            differential_validation,
        )
        self.assertEqual(
            differential_validation["counts"]["passed_sample_count"], 2
        )

        # Exercise the consumer representation too: JSON converts the typed
        # cadence tuples to lists, and the independent validator must accept
        # only an otherwise identical complete artifact.
        serialized = json.loads(json.dumps(result, allow_nan=False))
        from main.poisson_fullbody.shadow_identification import (
            ShadowIdentificationError,
            validate_shadow_replay_record,
        )

        validation_arguments = {
            "action_count": 2,
            "inner_updates_per_high_level_action": 5,
            "physics_substeps_per_inner_update": 5,
            "physics_timestep_s": 0.002,
            "contact_definition": runner.CONTACT_DEFINITION,
            "alpha_gain_per_s": 5.0,
            "static_drift_thresholds": {
                "translation_m": 1e-6,
                "rotation_rad": 1e-5,
                "surface_m": 1e-6,
            },
            "differential_audit_config": audit_config,
        }
        validate_shadow_replay_record(serialized, **validation_arguments)

        contacted = json.loads(json.dumps(serialized))
        contact_record = {
            "source_phase": "post_integration_recomputed",
            "observation_index": 20,
            "high_level_index": 0,
            "inner_control_index": 4,
            "physics_substep_index": 0,
            "mujoco_contact_index": 0,
            "mujoco_geom1_id": 5,
            "mujoco_geom1_name": "link5_collision",
            "mujoco_geom2_id": 100,
            "mujoco_geom2_name": "moka_pot_geom",
            "robot_geom_id": 5,
            "robot_geom_name": "link5_collision",
            "robot_body_id": 50,
            "robot_body_name": "robot0_link5",
            "obstacle_geom_id": 100,
            "obstacle_geom_name": "moka_pot_geom",
            "obstacle_body_id": 1000,
            "obstacle_body_name": "moka_pot_body",
            "is_physical_nonpositive_distance_contact": True,
            "contact_distance_m": -0.001,
        }
        contact_measurement = contacted["measurement"]
        contact_measurement.update(
            {
                "any_robot_obstacle_contact": True,
                "link56_obstacle_contact": True,
                "rollout_any_robot_obstacle_contact": True,
                "rollout_link56_obstacle_contact": True,
                "post_state_any_robot_obstacle_contact": True,
                "post_state_link56_obstacle_contact": True,
                "total_candidate_contact_point_record_count": 1,
                "total_physical_contact_point_record_count": 1,
                "rollout_phase_physical_contact_point_record_count": 1,
                "post_state_candidate_contact_point_record_count": 1,
                "post_state_physical_contact_point_record_count": 1,
                "post_state_candidate_contact_point_records": [contact_record],
                "post_state_physical_contact_point_records": [contact_record],
            }
        )
        warning_row = contacted["poisson_identification"]["trace"][10]
        for callback_group in warning_row["per_geom"]:
            for diagnostic_field in (
                "minimum_h_sample",
                "minimum_observed_cbf_lhs_sample",
            ):
                callback_diagnostic = dict(callback_group[diagnostic_field])
                callback_diagnostic["observed_arm_qvel_rad_s"] = (
                    [-0.5] + [0.0] * 6
                )
                callback_diagnostic[
                    "observed_grad_h_J_qdot_m2_per_s"
                ] = -0.5
                callback_diagnostic["observed_cbf_lhs_m2_per_s"] = 0.5
                callback_group[diagnostic_field] = callback_diagnostic
        warning_group = warning_row["per_geom"][0]
        warning_evidence = dict(
            warning_group["minimum_observed_cbf_lhs_sample"]
        )
        warning_evidence["gradient_world_m"] = [3.0, 0.0, 0.0]
        warning_evidence["gradient_norm_m"] = 3.0
        warning_evidence["observed_arm_qvel_rad_s"] = [-0.5] + [0.0] * 6
        warning_evidence["observed_grad_h_J_qdot_m2_per_s"] = -1.5
        warning_evidence["observed_cbf_lhs_m2_per_s"] = -0.5
        warning_group["minimum_h_sample"] = warning_evidence
        warning_group["minimum_observed_cbf_lhs_sample"] = warning_evidence
        warning_row["minimum_h_sample"] = warning_evidence
        warning_row["minimum_observed_cbf_lhs_m2_per_s"] = -0.5
        warning_row["minimum_observed_cbf_lhs_sample"] = warning_evidence
        warning = {
            "observation_index": 10,
            "high_level_index": 0,
            "physics_substep_index": 10,
            "signal_kind": "cbf_lhs_negative",
            "geom_id": 5,
            "geom_name": "link5_collision",
            "body_id": 50,
            "body_name": "robot0_link5",
            "evidence": warning_evidence,
        }
        contact_identification = contacted["poisson_identification"]
        contact_identification["signals"][
            "first_observed_minimum_cbf_lhs_negative_on_contact_geom"
        ] = warning
        contacted["poisson_identification"]["contact_prediction_assessment"] = {
            "assessment": "registered_warning_preceded_link56_contact",
            "first_link56_contact": contact_record,
            "primary_registered_warning": warning,
            "lead_physics_substeps": 10,
            "lead_time_s": 0.02,
        }
        contact_identification["trace_sha256"] = parity._sha256(
            parity._canonical(contact_identification["trace"])
        )
        validate_shadow_replay_record(contacted, **validation_arguments)

        invalid_warning_contacted = json.loads(json.dumps(contacted))
        invalid_identification = invalid_warning_contacted[
            "poisson_identification"
        ]
        invalid_row = invalid_identification["trace"][5]
        invalid_group = invalid_row["per_geom"][0]
        invalid_evidence = {
            "sample_id": 0,
            "body_id": 50,
            "body_name": "robot0_link5",
            "geom_id": 5,
            "geom_name": "link5_collision",
            "reason": "invalid_cell",
            "point_world_m": [0.1, 0.2, 0.3],
        }
        invalid_group.update(
            {
                "valid_query_count": 0,
                "invalid_query_count": 1,
                "observed_cbf_lhs_evaluation_count": 0,
                "invalid_reason_counts": {"invalid_cell": 1},
                "first_invalid_sample": invalid_evidence,
                "first_invalid_sample_by_reason": {
                    "invalid_cell": invalid_evidence
                },
                "minimum_h_sample": None,
                "minimum_observed_cbf_lhs_sample": None,
            }
        )
        remaining_group = invalid_row["per_geom"][1]
        invalid_row.update(
            {
                "valid_query_count": 1,
                "invalid_query_count": 1,
                "observed_cbf_lhs_evaluation_count": 1,
                "invalid_reason_counts": {"invalid_cell": 1},
                "minimum_h_m2": 0.2,
                "minimum_h_sample": remaining_group["minimum_h_sample"],
                "minimum_observed_cbf_lhs_m2_per_s": 1.0,
                "minimum_observed_cbf_lhs_sample": remaining_group[
                    "minimum_observed_cbf_lhs_sample"
                ],
            }
        )
        invalid_identification["valid_field_query_count"] = 99
        invalid_identification["observed_cbf_lhs_evaluation_count"] = 99
        invalid_warning = {
            "observation_index": 5,
            "high_level_index": 0,
            "physics_substep_index": 5,
            "signal_kind": "any_invalid",
            "geom_id": 5,
            "geom_name": "link5_collision",
            "body_id": 50,
            "body_name": "robot0_link5",
            "evidence": invalid_evidence,
        }
        invalid_identification["signals"][
            "first_any_fail_closed_query_on_contact_geom"
        ] = invalid_warning
        invalid_assessment = invalid_identification[
            "contact_prediction_assessment"
        ]
        invalid_assessment["primary_registered_warning"] = invalid_warning
        invalid_assessment["lead_physics_substeps"] = 15
        invalid_assessment["lead_time_s"] = 0.03
        invalid_identification["trace_sha256"] = parity._sha256(
            parity._canonical(invalid_identification["trace"])
        )
        validate_shadow_replay_record(
            invalid_warning_contacted, **validation_arguments
        )

        unregistered_invalid_reason = json.loads(
            json.dumps(invalid_warning_contacted)
        )
        unregistered_invalid_reason["poisson_identification"]["trace"][5][
            "per_geom"
        ][0]["first_invalid_sample"]["reason"] = "fabricated_reason"
        with self.assertRaises(ShadowIdentificationError):
            validate_shadow_replay_record(
                unregistered_invalid_reason,
                **validation_arguments
            )

        live_contacted = json.loads(json.dumps(contacted))
        live_record = dict(contact_record)
        live_record.update(
            {
                "source_phase": "live_solver_phase_preintegration_geometry",
                "observation_index": 21,
                "inner_control_index": 4,
                "physics_substep_index": 1,
            }
        )
        live_measurement = live_contacted["measurement"]
        live_measurement.update(
            {
                "post_state_any_robot_obstacle_contact": False,
                "post_state_link56_obstacle_contact": False,
                "post_state_candidate_contact_point_record_count": 0,
                "post_state_physical_contact_point_record_count": 0,
                "post_state_candidate_contact_point_records": [],
                "post_state_physical_contact_point_records": [],
                "live_solver_any_robot_obstacle_contact": True,
                "live_solver_link56_obstacle_contact": True,
                "live_solver_candidate_contact_point_record_count": 1,
                "live_solver_nonpositive_contact_point_record_count": 1,
                "live_solver_phase_contact_point_records": [live_record],
            }
        )
        live_assessment = live_contacted["poisson_identification"][
            "contact_prediction_assessment"
        ]
        live_assessment["first_link56_contact"] = live_record
        # The live record at boundary 21*dt and warning row 10 at
        # boundary 11*dt have a ten-substep physical lead.
        live_assessment["lead_physics_substeps"] = 10
        live_assessment["lead_time_s"] = 0.02
        validate_shadow_replay_record(live_contacted, **validation_arguments)

        from scripts import run_poisson_active_canary as active_runner

        prerequisite = {
            "schema_version": active_runner.EXPECTED_IDENTIFICATION_SCHEMA,
            "status": "passed",
            "scientific_result": False,
            "case_id": "vlsa-t1-goal-ii-t0-e05",
            "provenance": {
                "source": {"commit": "source-commit", "status_short": []},
                "manifest_sha256": "manifest",
                "manifest_row_sha256": "row",
                "selection_config_sha256": "selection",
                "runtime_protocol_raw_sha256": "runtime-raw",
                "runtime_protocol_semantic_sha256": "runtime-semantic",
                "runtime_parameter_block_sha256": "runtime-parameters",
                "historical_result_file_sha256": "historical-file",
                "historical_result_payload_sha256": "historical-payload",
                "upstream_parity_payload_sha256": "parity-payload",
            },
            "historical": replay_provenance,
            "shadow_replay": contacted,
            "acceptance": {
                "upstream_exact_parity_same_clean_commit": True,
                "all_237_historical_actions_executed": True,
                "all_5925_callbacks_observed": True,
                "historical_state_reward_done_goal_exact": True,
                "upstream_observation_sequence_exact": True,
                "complete_mujoco_integration_state_unchanged_by_construction": True,
                "complete_mujoco_integration_state_unchanged_by_callback": True,
                "all_link56_protected_sample_point_jacobians_validated": True,
                "all_link56_protected_sample_field_chain_rules_validated": True,
                "static_queries_stop_at_first_registered_drift": True,
                "contact_authority_is_mujoco_nonpositive_distance": True,
                "no_action_or_control_mutation": True,
                "no_active_safety_efficacy_claim": True,
            },
        }
        with mock.patch(
            "main.poisson_fullbody.contracts.load_hashed_json",
            return_value=prerequisite,
        ), mock.patch.object(
            active_runner, "_require_full_robot_sampling_evidence"
        ):
            loaded = active_runner._require_identification_prerequisite(
                Path("unused-identification.json"),
                case={"case_id": "vlsa-t1-goal-ii-t0-e05"},
                case_row_hash="row",
                source_commit="source-commit",
                manifest_sha256="manifest",
                selection_sha256="selection",
                runtime_protocol_raw_sha256="runtime-raw",
                runtime_protocol_semantic_sha256="runtime-semantic",
                runtime_parameter_block_sha256="runtime-parameters",
                alpha_gain_per_s=5.0,
                static_drift_thresholds=validation_arguments[
                    "static_drift_thresholds"
                ],
                differential_audit_config=validation_arguments[
                    "differential_audit_config"
                ],
                replay=replay,
                parity=upstream,
            )
        self.assertIs(loaded, prerequisite)

        invalid_query_only_prerequisite = json.loads(json.dumps(prerequisite))
        invalid_query_only_prerequisite["shadow_replay"] = (
            invalid_warning_contacted
        )
        with mock.patch(
            "main.poisson_fullbody.contracts.load_hashed_json",
            return_value=invalid_query_only_prerequisite,
        ), mock.patch.object(
            active_runner, "_require_full_robot_sampling_evidence"
        ):
            with self.assertRaisesRegex(
                active_runner.ActiveRunnerError,
                "warning lead or registered contact-geom identity is invalid",
            ):
                active_runner._require_identification_prerequisite(
                    Path("unused-identification.json"),
                    case={"case_id": "vlsa-t1-goal-ii-t0-e05"},
                    case_row_hash="row",
                    source_commit="source-commit",
                    manifest_sha256="manifest",
                    selection_sha256="selection",
                    runtime_protocol_raw_sha256="runtime-raw",
                    runtime_protocol_semantic_sha256="runtime-semantic",
                    runtime_parameter_block_sha256="runtime-parameters",
                    alpha_gain_per_s=5.0,
                    static_drift_thresholds=validation_arguments[
                        "static_drift_thresholds"
                    ],
                    differential_audit_config=validation_arguments[
                        "differential_audit_config"
                    ],
                    replay=replay,
                    parity=upstream,
                )

        wrong_terminal = json.loads(json.dumps(prerequisite))
        wrong_terminal["shadow_replay"][
            "terminal_simulator_state_sha256"
        ] = "d" * 64
        with mock.patch(
            "main.poisson_fullbody.contracts.load_hashed_json",
            return_value=wrong_terminal,
        ), mock.patch.object(
            active_runner, "_require_full_robot_sampling_evidence"
        ):
            with self.assertRaisesRegex(
                active_runner.ActiveRunnerError,
                "action-boundary states differ from history",
            ):
                active_runner._require_identification_prerequisite(
                    Path("unused-identification.json"),
                    case={"case_id": "vlsa-t1-goal-ii-t0-e05"},
                    case_row_hash="row",
                    source_commit="source-commit",
                    manifest_sha256="manifest",
                    selection_sha256="selection",
                    runtime_protocol_raw_sha256="runtime-raw",
                    runtime_protocol_semantic_sha256="runtime-semantic",
                    runtime_parameter_block_sha256="runtime-parameters",
                    alpha_gain_per_s=5.0,
                    static_drift_thresholds=validation_arguments[
                        "static_drift_thresholds"
                    ],
                    differential_audit_config=validation_arguments[
                        "differential_audit_config"
                    ],
                    replay=replay,
                    parity=upstream,
                )

        for rewritten_index in (0, len(state_hashes) - 1):
            collusive_action_rewrite = json.loads(
                json.dumps(prerequisite)
            )
            collusive_parity_rewrite = json.loads(json.dumps(upstream))
            rewritten_states = list(state_hashes)
            rewritten_states[rewritten_index] = "e" * 64
            rewritten_sequence = parity._sha256(
                parity._canonical(rewritten_states)
            )
            rewritten_identification_shadow = collusive_action_rewrite[
                "shadow_replay"
            ]
            rewritten_identification_shadow[
                "action_boundary_state_sha256_ledger"
            ] = rewritten_states
            rewritten_identification_shadow[
                "state_sequence_sha256"
            ] = rewritten_sequence
            rewritten_parity_callback = collusive_parity_rewrite[
                "callback_replay"
            ]
            rewritten_parity_callback[
                "action_boundary_state_sha256_ledger"
            ] = rewritten_states
            rewritten_parity_callback[
                "state_sequence_sha256"
            ] = rewritten_sequence
            if rewritten_index == len(state_hashes) - 1:
                rewritten_identification_shadow[
                    "terminal_simulator_state_sha256"
                ] = rewritten_states[-1]
                rewritten_parity_callback[
                    "terminal_simulator_state_sha256"
                ] = rewritten_states[-1]
            with self.subTest(
                collusive_action_rewrite_index=rewritten_index
            ), mock.patch(
                "main.poisson_fullbody.contracts.load_hashed_json",
                return_value=collusive_action_rewrite,
            ), mock.patch.object(
                active_runner, "_require_full_robot_sampling_evidence"
            ):
                with self.assertRaisesRegex(
                    active_runner.ActiveRunnerError,
                    "action-boundary states differ from history",
                ):
                    active_runner._require_identification_prerequisite(
                        Path("unused-identification.json"),
                        case={"case_id": "vlsa-t1-goal-ii-t0-e05"},
                        case_row_hash="row",
                        source_commit="source-commit",
                        manifest_sha256="manifest",
                        selection_sha256="selection",
                        runtime_protocol_raw_sha256="runtime-raw",
                        runtime_protocol_semantic_sha256=(
                            "runtime-semantic"
                        ),
                        runtime_parameter_block_sha256=(
                            "runtime-parameters"
                        ),
                        alpha_gain_per_s=5.0,
                        static_drift_thresholds=validation_arguments[
                            "static_drift_thresholds"
                        ],
                        differential_audit_config=validation_arguments[
                            "differential_audit_config"
                        ],
                        replay=replay,
                        parity=collusive_parity_rewrite,
                    )

        collusive_callback_rewrite = json.loads(json.dumps(prerequisite))
        rewritten_shadow = collusive_callback_rewrite["shadow_replay"]
        rewritten_callback = rewritten_shadow[
            "callback_state_read_only_ledger"
        ][0]
        rewritten_callback["before_sha256"] = "d" * 64
        rewritten_callback["after_sha256"] = "d" * 64
        rewritten_shadow[
            "callback_state_read_only_ledger_sha256"
        ] = parity._sha256(
            parity._canonical(
                rewritten_shadow["callback_state_read_only_ledger"]
            )
        )
        rewritten_shadow[
            "callback_state_sequence_sha256"
        ] = parity._sha256(
            parity._canonical(
                [
                    row["after_sha256"]
                    for row in rewritten_shadow[
                        "callback_state_read_only_ledger"
                    ]
                ]
            )
        )
        with mock.patch(
            "main.poisson_fullbody.contracts.load_hashed_json",
            return_value=collusive_callback_rewrite,
        ), mock.patch.object(
            active_runner, "_require_full_robot_sampling_evidence"
        ):
            with self.assertRaisesRegex(
                active_runner.ActiveRunnerError,
                "callback integration-state ledger differs from exact parity",
            ):
                active_runner._require_identification_prerequisite(
                    Path("unused-identification.json"),
                    case={"case_id": "vlsa-t1-goal-ii-t0-e05"},
                    case_row_hash="row",
                    source_commit="source-commit",
                    manifest_sha256="manifest",
                    selection_sha256="selection",
                    runtime_protocol_raw_sha256="runtime-raw",
                    runtime_protocol_semantic_sha256="runtime-semantic",
                    runtime_parameter_block_sha256="runtime-parameters",
                    alpha_gain_per_s=5.0,
                    static_drift_thresholds=validation_arguments[
                        "static_drift_thresholds"
                    ],
                    differential_audit_config=validation_arguments[
                        "differential_audit_config"
                    ],
                    replay=replay,
                    parity=upstream,
                )

        invalidated = json.loads(json.dumps(serialized))
        invalidation_index = 17
        identification = invalidated["poisson_identification"]
        crossing_drift = identification["trace"][invalidation_index]["drift"]
        crossing_geom = crossing_drift["geoms"][0]
        crossing_geom["translation_m"] = 2e-6
        crossing_geom["maximum_surface_point_displacement_m"] = 2e-6
        crossing_geom["threshold_crossing_reasons"] = [
            "translation_threshold_crossed",
            "surface_threshold_crossed",
        ]
        crossing_drift["maximum_translation_m"] = 2e-6
        crossing_drift["maximum_surface_point_displacement_m"] = 2e-6
        identification["first_static_field_invalidation"] = {
            "observation_index": invalidation_index,
            "high_level_index": 0,
            "physics_substep_index": invalidation_index,
            "reason": "static_selected_obstacle_drift",
            "threshold_crossing_reasons": [
                "surface_threshold_crossed",
                "translation_threshold_crossed",
            ],
            "drift": crossing_drift,
        }
        for row in identification["trace"][invalidation_index:]:
            row["static_field_admissible_for_this_query"] = False
            row["field_query_attempted"] = False
            row["skip_reason"] = (
                "static_selected_obstacle_drift_invalidated_field"
            )
            row["valid_query_count"] = 0
            row["invalid_query_count"] = 0
            row["observed_cbf_lhs_evaluation_count"] = 0
            row["minimum_h_m2"] = None
            row["minimum_h_sample"] = None
            row["minimum_observed_cbf_lhs_m2_per_s"] = None
            row["minimum_observed_cbf_lhs_sample"] = None
            row["per_geom"] = []
        identification["field_query_callback_count"] = invalidation_index
        identification[
            "field_query_skipped_after_static_invalidation_count"
        ] = 50 - invalidation_index
        identification["valid_field_query_count"] = 2 * invalidation_index
        identification[
            "observed_cbf_lhs_evaluation_count"
        ] = 2 * invalidation_index
        identification["trace_sha256"] = parity._sha256(
            parity._canonical(identification["trace"])
        )
        validate_shadow_replay_record(invalidated, **validation_arguments)

        tamperers = (
            (
                "action-boundary state ledger",
                lambda value: value[
                    "action_boundary_state_sha256_ledger"
                ].__setitem__(0, "d" * 64),
            ),
            (
                "terminal state summary",
                lambda value: value.__setitem__(
                    "terminal_simulator_state_sha256", "d" * 64
                ),
            ),
            (
                "callback state sequence hash",
                lambda value: value.__setitem__(
                    "callback_state_sequence_sha256", "d" * 64
                ),
            ),
            (
                "callback state ledger hash",
                lambda value: value.__setitem__(
                    "callback_state_read_only_ledger_sha256", "d" * 64
                ),
            ),
            (
                "callback state equality",
                lambda value: value["callback_state_read_only_ledger"][0].__setitem__(
                    "after_sha256", "d" * 64
                ),
            ),
            (
                "callback state cadence",
                lambda value: value["callback_state_read_only_ledger"][0].__setitem__(
                    "observation_index", 1
                ),
            ),
            (
                "measurement endpoint",
                lambda value: value["measurement"].__setitem__(
                    "last_index", [1, 4, 3]
                ),
            ),
            (
                "trace cadence",
                lambda value: value["poisson_identification"]["trace"][17].__setitem__(
                    "observation_index", 18
                ),
            ),
            (
                "trace hash",
                lambda value: value["poisson_identification"].__setitem__(
                    "trace_sha256", "0" * 64
                ),
            ),
            (
                "query stopping",
                lambda value: value["poisson_identification"]["trace"][17].__setitem__(
                    "field_query_attempted", False
                ),
            ),
            (
                "construction state",
                lambda value: value["construction"][
                    "complete_integration_state_read_only_audit"
                ].__setitem__("after_sha256", "different-state"),
            ),
            (
                "physical-model digest",
                lambda value: value["construction"]["physical_model"].__setitem__(
                    "sha256", "not-a-sha256"
                ),
            ),
            (
                "compiled-model digest",
                lambda value: value["construction"]["physical_model"].__setitem__(
                    "compiled_mjb_sha256", "not-a-sha256"
                ),
            ),
            (
                "physical-model exact structure",
                lambda value: value["construction"]["physical_model"].__setitem__(
                    "unregistered", True
                ),
            ),
            (
                "physical-model field population",
                lambda value: value["construction"]["physical_model"].__setitem__(
                    "field_count", 0
                ),
            ),
            (
                "registered sample coverage",
                lambda value: value["construction"]["field_bundle"].__setitem__(
                    "protected_sample_count", 1
                ),
            ),
            (
                "bound drift threshold",
                lambda value: value["construction"][
                    "static_field_drift_thresholds"
                ].__setitem__("translation_m", 1e-3),
            ),
            (
                "bound CBF alpha",
                lambda value: value["construction"].__setitem__(
                    "cbf_alpha_gain_per_s", 6.0
                ),
            ),
            (
                "differential audit payload",
                lambda value: value["construction"][
                    "settled_link56_differential_audit"
                ]["sample_records"][0].__setitem__("passed", False),
            ),
            (
                "differential audit validation receipt",
                lambda value: value["construction"][
                    "settled_link56_differential_audit_validation"
                ]["counts"].__setitem__("passed_sample_count", 1),
            ),
            (
                "per-geom trace coverage",
                lambda value: value["poisson_identification"]["trace"][0].__setitem__(
                    "per_geom", value["poisson_identification"]["trace"][0]["per_geom"][:1]
                ),
            ),
            (
                "monitor exception ledger",
                lambda value: (
                    value.__setitem__("monitor_static_drift_exception_count", 1),
                    value.__setitem__(
                        "monitor_static_drift_exceptions",
                        [{"observation_index": 0, "message": "unexpected"}],
                    ),
                ),
            ),
            (
                "contact physical authority",
                lambda value: value["poisson_identification"].__setitem__(
                    "contact_prediction_assessment",
                    {
                        "assessment": (
                            "registered_warning_preceded_link56_contact"
                        ),
                        "first_link56_contact": {
                            "source_phase": "post_integration_recomputed",
                            "observation_index": 20,
                            "high_level_index": 0,
                            "inner_control_index": 4,
                            "physics_substep_index": 0,
                            "robot_geom_id": 5,
                            "is_physical_nonpositive_distance_contact": False,
                        },
                        "primary_registered_warning": {
                            "observation_index": 10,
                            "geom_id": 5,
                        },
                        "lead_physics_substeps": 10,
                        "lead_time_s": 0.02,
                    },
                ),
            ),
        )
        for label, tamper in tamperers:
            with self.subTest(tamper=label):
                corrupted = json.loads(json.dumps(serialized))
                tamper(corrupted)
                with self.assertRaises(ShadowIdentificationError):
                    validate_shadow_replay_record(
                        corrupted, **validation_arguments
                    )

        corrupted_invalidation = json.loads(json.dumps(invalidated))
        corrupted_invalidation["poisson_identification"][
            "first_static_field_invalidation"
        ]["drift"] = {}
        with self.assertRaises(ShadowIdentificationError):
            validate_shadow_replay_record(
                corrupted_invalidation, **validation_arguments
            )

        off_by_one_live_lead = json.loads(json.dumps(live_contacted))
        live_assessment = off_by_one_live_lead["poisson_identification"][
            "contact_prediction_assessment"
        ]
        live_assessment["lead_physics_substeps"] = 11
        live_assessment["lead_time_s"] = 0.022
        with self.assertRaises(ShadowIdentificationError):
            validate_shadow_replay_record(
                off_by_one_live_lead, **validation_arguments
            )

        unregistered_drift_geom = json.loads(json.dumps(serialized))
        for row in unregistered_drift_geom["poisson_identification"]["trace"]:
            row["drift"]["geoms"][0]["geom_id"] = 999999
        unregistered_drift_geom["poisson_identification"][
            "trace_sha256"
        ] = parity._sha256(
            parity._canonical(
                unregistered_drift_geom["poisson_identification"]["trace"]
            )
        )
        with self.assertRaises(ShadowIdentificationError):
            validate_shadow_replay_record(
                unregistered_drift_geom, **validation_arguments
            )

        collusive_bad_lhs = json.loads(json.dumps(contacted))
        bad_identification = collusive_bad_lhs["poisson_identification"]
        bad_row = bad_identification["trace"][10]
        bad_row["per_geom"][0]["minimum_observed_cbf_lhs_sample"][
            "observed_cbf_lhs_m2_per_s"
        ] = -2.0
        bad_row["minimum_observed_cbf_lhs_sample"][
            "observed_cbf_lhs_m2_per_s"
        ] = -2.0
        bad_row["minimum_observed_cbf_lhs_m2_per_s"] = -2.0
        bad_identification["signals"][
            "first_observed_minimum_cbf_lhs_negative_on_contact_geom"
        ]["evidence"]["observed_cbf_lhs_m2_per_s"] = -2.0
        bad_identification["contact_prediction_assessment"][
            "primary_registered_warning"
        ]["evidence"]["observed_cbf_lhs_m2_per_s"] = -2.0
        bad_identification["trace_sha256"] = parity._sha256(
            parity._canonical(bad_identification["trace"])
        )
        with self.assertRaises(ShadowIdentificationError):
            validate_shadow_replay_record(
                collusive_bad_lhs, **validation_arguments
            )

        impossible_zero_qvel_warning = json.loads(json.dumps(contacted))
        impossible_identification = impossible_zero_qvel_warning[
            "poisson_identification"
        ]
        impossible_row = impossible_identification["trace"][10]
        impossible_row["per_geom"][0][
            "minimum_observed_cbf_lhs_sample"
        ]["observed_arm_qvel_rad_s"] = [0.0] * 7
        impossible_row["minimum_observed_cbf_lhs_sample"][
            "observed_arm_qvel_rad_s"
        ] = [0.0] * 7
        impossible_identification["signals"][
            "first_observed_minimum_cbf_lhs_negative_on_contact_geom"
        ]["evidence"]["observed_arm_qvel_rad_s"] = [0.0] * 7
        impossible_identification["contact_prediction_assessment"][
            "primary_registered_warning"
        ]["evidence"]["observed_arm_qvel_rad_s"] = [0.0] * 7
        impossible_identification["trace_sha256"] = parity._sha256(
            parity._canonical(impossible_identification["trace"])
        )
        with self.assertRaises(ShadowIdentificationError):
            validate_shadow_replay_record(
                impossible_zero_qvel_warning,
                **validation_arguments
            )

        cross_geom_sample = json.loads(json.dumps(contacted))
        cross_identification = cross_geom_sample["poisson_identification"]
        cross_row = cross_identification["trace"][10]
        cross_row["per_geom"][0]["minimum_observed_cbf_lhs_sample"][
            "sample_id"
        ] = 1
        cross_row["minimum_observed_cbf_lhs_sample"]["sample_id"] = 1
        cross_identification["signals"][
            "first_observed_minimum_cbf_lhs_negative_on_contact_geom"
        ]["evidence"]["sample_id"] = 1
        cross_identification["contact_prediction_assessment"][
            "primary_registered_warning"
        ]["evidence"]["sample_id"] = 1
        cross_identification["trace_sha256"] = parity._sha256(
            parity._canonical(cross_identification["trace"])
        )
        with self.assertRaises(ShadowIdentificationError):
            validate_shadow_replay_record(
                cross_geom_sample, **validation_arguments
            )

        contact_without_obstacle_authority = json.loads(json.dumps(contacted))
        contact_without_obstacle_authority["measurement"][
            "post_state_physical_contact_point_records"
        ][0].pop("obstacle_geom_id")
        contact_without_obstacle_authority["poisson_identification"][
            "contact_prediction_assessment"
        ]["first_link56_contact"].pop("obstacle_geom_id")
        with self.assertRaises(ShadowIdentificationError):
            validate_shadow_replay_record(
                contact_without_obstacle_authority,
                **validation_arguments
            )

    def test_callback_is_read_only_and_drift_does_not_end_replay(self):
        self.assertIn("mjtState.mjSTATE_INTEGRATION", self.runner)
        self.assertIn("mj_getState", self.runner)
        self.assertIn("construction_state_before", self.runner)
        self.assertIn("construction_state_after", self.runner)
        self.assertIn('runtime["np"].array_equal', self.runner)
        self.assertNotIn("sim.get_state().flatten()", self.runner)
        self.assertIn(
            "mutated complete MuJoCo integration state", self.runner
        )
        self.assertIn("except StaticObstacleDriftInadmissible", self.runner)
        self.assertIn("terminate_on_static_drift=False", self.runner)
        self.assertIn("exact OSC replay and contact measurement must continue", self.runner)
        self.assertIn("permanently stops querying the field", self.module)
        self.assertNotIn("env.step(", self.module)

    def test_observed_cbf_residual_is_evaluated_for_every_valid_sample(self):
        self.assertIn("for sample in self._samples", self.module)
        self.assertIn("minimum_h_sample", self.module)
        self.assertIn("minimum_observed_cbf_lhs_sample", self.module)
        self.assertIn(
            "minimum over every valid protected sample", self.module
        )
        self.assertIn(
            'signals["first_any_fail_closed_query_on_contact_geom"]',
            self.module,
        )
        self.assertIn("live_solver_phase_preintegration_geometry", self.module)
        self.assertIn("time_from_settled_state_s", self.module)

    def test_official_state_reader_matches_mujoco_integration_vector(self):
        try:
            import mujoco
            import numpy as np
        except ImportError:
            self.skipTest("official MuJoCo/NumPy are unavailable")
        from scripts.run_poisson_shadow_identification import (
            _official_integration_state,
        )

        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body><joint type='slide'/><geom "
            "type='sphere' size='.01'/></body></worldbody></mujoco>"
        )
        data = mujoco.MjData(model)
        data.time = 0.125
        data.qpos[0] = 0.25
        data.qvel[0] = -0.5
        observed = _official_integration_state(
            SimpleNamespace(model=model, data=data), np
        )
        specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        expected = np.empty(
            int(mujoco.mj_stateSize(model, specification)), dtype=np.float64
        )
        mujoco.mj_getState(model, data, expected, specification)
        self.assertTrue(np.array_equal(observed, expected))

    def test_field_bundle_measurement_and_claim_limits_are_explicit(self):
        self.assertIn("build_static_field_bundle", self.runner)
        self.assertIn("FullRobotObstacleMonitor", self.runner)
        self.assertIn("mujoco_contact_dist_le_0", self.runner)
        self.assertIn('"computed": False', self.module)
        self.assertIn("no active safety or utility efficacy", self.runner)
        self.assertIn('"scientific_result": False', self.runner)

    def test_allocation_wrapper_is_bounded_clean_and_uses_validated_osmesa(self):
        self.assertIn("#SBATCH --partition=mig", self.batch)
        self.assertIn("#SBATCH --gres=gpu:1", self.batch)
        self.assertIn("#SBATCH --cpus-per-task=8", self.batch)
        self.assertIn("#SBATCH --mem=64G", self.batch)
        self.assertIn("#SBATCH --no-requeue", self.batch)
        self.assertIn('if [[ -n "$(git status --short)" ]]', self.batch)
        self.assertIn("EXPECTED_GIT_COMMIT", self.batch)
        self.assertIn("git rev-parse HEAD", self.batch)
        self.assertIn("MUJOCO_GL=osmesa", self.batch)
        self.assertIn("PYOPENGL_PLATFORM=osmesa", self.batch)
        self.assertIn("LIBERO_CONFIG_PATH", self.batch)
        self.assertIn("PYTHONPATH", self.batch)
        self.assertIn("PARITY_RESULT_PATH", self.batch)

    def test_result_is_published_once_only_after_shadow_finishes(self):
        run_index = self.runner.index("shadow = _run_shadow")
        publish_index = self.runner.index("publish_hashed_json(output, payload)")
        self.assertLess(run_index, publish_index)
        self.assertIn("requires a clean source tree", self.runner)
        self.assertIn("SLURM_JOB_ID", self.runner)
        self.assertIn("_gpu_inventory()", self.runner)
        main_source = self.runner[self.runner.index("def main()") :]
        provenance_index = main_source.index('"provenance": provenance')
        risky_audit_index = main_source.index("shadow = _run_shadow")
        self.assertLess(provenance_index, risky_audit_index)
        self.assertIn("_record_failed_result(payload, error", main_source)
        self.assertIn('payload["failure_evidence"] = failure_evidence', self.runner)
        self.assertIn(
            "settled_link56_protected_sample_differential_audit_validation",
            self.runner,
        )


if __name__ == "__main__":
    unittest.main()
