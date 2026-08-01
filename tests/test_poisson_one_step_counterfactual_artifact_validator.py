from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import copy
from types import SimpleNamespace
from unittest import mock

from scripts import validate_poisson_one_step_counterfactual_artifact as validator


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hashed(value):
    output = dict(value)
    output["result_payload_sha256"] = hashlib.sha256(_canonical(output)).hexdigest()
    return output


def _write(path, value):
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _consumer_fixture(root):
    commit = "a" * 40
    run_id = "stage13-run"
    runtime_raw = "1" * 64
    runtime_semantic = "2" * 64
    runtime_parameters = "3" * 64
    run_root = root / run_id
    run_root.mkdir()
    configs = root / "configs"
    configs.mkdir()
    selection = {
        "schema_version": "selection.v1",
        "protocol_id": "selection-protocol-v1",
    }
    selection_path = configs / "selection.json"
    _write(selection_path, selection)
    selection_raw = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    runtime = {
        "schema_version": "vlsa_poisson_runtime_protocol.v3",
        "protocol_id": "runtime-v3-fixture",
        "differential_audit": {"fixture": True},
        "admissibility": {"fixture": True},
        "coverage": {"fixture": True},
        "cbf": {"fixture": True},
        "qp": {"fixture": True, "weight_diagonal": [1.0] * 7},
        "cadence": {"fixture": True},
    }
    runtime_path = configs / "runtime.json"
    _write(runtime_path, runtime)
    runtime_raw = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    stage_protocol = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/vlsa_poisson_one_step_counterfactual.v2.json"
        ).read_text(encoding="utf-8")
    )
    controller_authority = stage_protocol["joint_velocity_controller_authority"]
    protocol = {
        "schema_version": validator.EXPECTED_PROTOCOL_SCHEMA,
        "protocol_id": "fixture-protocol",
        "case": {"case_id": "vlsa-t1-goal-ii-t0-e05"},
        "joint_velocity_controller_authority": controller_authority,
        "prerequisites": {
            "selection_protocol_relative_path": "configs/selection.json",
            "selection_schema_version": selection["schema_version"],
            "selection_protocol_id": selection["protocol_id"],
            "selection_raw_file_sha256": selection_raw,
            "runtime_protocol_relative_path": "configs/runtime.json",
            "numeric_validation_schema_version": validator.PREREQUISITE_SCHEMAS[
                "numeric"
            ],
            "exact_parity_schema_version": validator.PREREQUISITE_SCHEMAS[
                "parity"
            ],
            "shadow_identification_schema_version": validator.PREREQUISITE_SCHEMAS[
                "identification"
            ],
            "runtime_raw_file_sha256": runtime_raw,
            "runtime_semantic_protocol_sha256": runtime_semantic,
            "runtime_parameter_block_sha256": runtime_parameters,
            "runtime_schema_version": "vlsa_poisson_runtime_protocol.v3",
            "runtime_protocol_id": "runtime-v3-fixture",
        },
        "qp_execution": copy.deepcopy(stage_protocol["qp_execution"]),
        "counterfactual_boundary": {
            "physics_substeps_per_filter_update": 5,
            "counterfactual_horizon_substeps": 5,
            "physics_timestep_s": 0.002,
            "counterfactual_horizon_s": 0.01,
        },
        "nominal_velocity_estimator": stage_protocol[
            "nominal_velocity_estimator"
        ],
        "boundary_B_preflight": copy.deepcopy(
            stage_protocol["boundary_B_preflight"]
        ),
        "arms": {
            "gripper_policy": {
                "required_evidence": [
                    "source_action_index",
                    "source_action_7d",
                    "source_action_sha256",
                    "exact_gripper_value",
                    "exact_gripper_value_sha256",
                ],
            },
        },
        "required_evidence": {
            "trend_reference_acceptance_evidence": [
                "first_order_predicted_h_after_horizon_m2",
                "actual_h_after_each_physics_substep_m2",
                "nominal_D_sim_at_B_m",
                "nominal_minimum_D_sim_over_horizon_m",
                "psf_h_after_horizon_m2",
                "psf_minus_nominal_h_after_horizon_m2",
                "psf_warning_sample_h_strictly_greater_than_nominal_at_horizon",
            ],
            "diagnostic_only": [
                "prediction_error_m2_without_acceptance_threshold"
            ],
        },
        "acceptance": {
            "prediction_error_acceptance_threshold": None,
            "prediction_error_policy": (
                "diagnostic_only_no_post_hoc_tolerance"
            ),
            "trend_only_quality_prerequisites": [
                "complete_exposure",
                "exact_paired_start",
                "qp_valid",
                "both_tracking_valid",
                "static_field_admissible",
            ],
            "minimum_filter_correction_norm_rad_s": 1e-4,
            "minimum_safe_command_norm_rad_s": 0.05,
            "minimum_safe_to_nominal_command_norm_ratio": 0.25,
            "minimum_safe_measured_joint_motion_rad": 1e-4,
            "minimum_safe_measured_motion_to_command_integral_ratio": 0.25,
            "local_motion_interpretation": (
                "nonzero_local_joint_motion_not_nominal_direction_progress_or_task_success"
            ),
        },
        "result_contract": {
            "schema_version": validator.EXPECTED_RESULT_SCHEMA,
            "claim_scope": (
                "one_100hz_interval_local_geometry_field_jacobian_qp_and_"
                "tracking_feasibility_not_task_success_or_full_episode_safety"
            ),
        },
    }
    protocol_path = root / "protocol.json"
    _write(protocol_path, protocol)
    prerequisite_paths = {}
    parity_payload_sha256 = None
    historical_action = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    historical_action_sha256 = hashlib.sha256(
        validator._historical_canonical([historical_action])
    ).hexdigest()
    historical_payload = _hashed(
        {
            "schema_version": "vlsa_table1_episode_result.v1",
            "status": "complete",
            "scientific_result": True,
            "case_id": "vlsa-t1-goal-ii-t0-e05",
            "arm": "pi05_plus_aegis_translational",
            "actions": [
                {
                    "step": 0,
                    "executed": historical_action,
                    "env_step_input": historical_action,
                }
            ],
            "action_invariance_ledger": {
                "action_count": 1,
                "executed_sequence_sha256": historical_action_sha256,
            },
            "pairing": {"policy_noise_schedule_sha256": "9" * 64},
        }
    )
    historical_path = root / "historical-result.json"
    _write(historical_path, historical_payload)
    historical = {
        "historical_result_file_sha256": hashlib.sha256(
            historical_path.read_bytes()
        ).hexdigest(),
        "historical_result_payload_sha256": historical_payload[
            "result_payload_sha256"
        ],
        "action_count": 1,
        "executed_sequence_sha256": historical_action_sha256,
        "policy_noise_schedule_sha256": "9" * 64,
    }
    action_state_ledger = ["a" * 64]
    official_state_ledger = ["c" * 64]
    callback = {
        "settled_official_integration_state": {
            "physical_boundary": 0,
            "mujoco_state_specification": "mjSTATE_INTEGRATION",
            "state_vector_length": 15,
            "sha256": "0" * 64,
        },
        "executed_action_count": 1,
        "action_boundary_state_sha256_ledger": action_state_ledger,
        "state_sequence_sha256": hashlib.sha256(
            _canonical(action_state_ledger)
        ).hexdigest(),
        "official_integration_state_count": 1,
        "official_integration_state_sha256_ledger": official_state_ledger,
        "official_integration_state_sequence_sha256": hashlib.sha256(
            _canonical(official_state_ledger)
        ).hexdigest(),
    }
    for kind, schema in validator.PREREQUISITE_SCHEMAS.items():
        source = {"commit": commit, "status_short": []}
        payload = {
            "schema_version": schema,
            "status": "passed",
            "acceptance": (
                {
                    field: True
                    for field in validator.PARITY_ACCEPTANCE_FIELDS
                }
                if kind == "parity"
                else {
                    field: True
                    for field in validator.IDENTIFICATION_ACCEPTANCE_FIELDS
                }
                if kind == "identification"
                else {"fixture_acceptance": True}
            ),
        }
        if kind == "numeric":
            payload["source"] = source
        else:
            payload["provenance"] = {
                "source": source,
                "manifest_sha256": "4" * 64,
                "manifest_row_sha256": "5" * 64,
                "historical_result_file_sha256": historical[
                    "historical_result_file_sha256"
                ],
                "historical_result_payload_sha256": historical[
                    "historical_result_payload_sha256"
                ],
            }
            payload["case_id"] = "vlsa-t1-goal-ii-t0-e05"
            payload["historical"] = dict(historical)
        if kind == "parity":
            payload["ordinary_replay"] = {
                "settled_official_integration_state": copy.deepcopy(
                    callback["settled_official_integration_state"]
                )
            }
            payload["callback_replay"] = copy.deepcopy(callback)
        if kind == "identification":
            callback_rows = [
                {
                    "physical_boundary": 1,
                    "before_sha256": official_state_ledger[0],
                    "after_sha256": official_state_ledger[0],
                    "exact_array_equal": True,
                }
            ]
            payload["provenance"].update(
                {
                    "upstream_parity_payload_sha256": parity_payload_sha256,
                    "runtime_protocol_raw_sha256": runtime_raw,
                    "runtime_protocol_semantic_sha256": runtime_semantic,
                    "runtime_parameter_block_sha256": runtime_parameters,
                    "selection_config_sha256": selection_raw,
                    "selection_protocol_id": selection["protocol_id"],
                }
            )
            payload["shadow_replay"] = {
                "callback_count": len(callback_rows),
                "callback_state_read_only_ledger": callback_rows,
                "callback_state_read_only_ledger_sha256": hashlib.sha256(
                    _canonical(callback_rows)
                ).hexdigest(),
                "action_boundary_state_sha256_ledger": list(
                    callback["action_boundary_state_sha256_ledger"]
                ),
                "state_sequence_sha256": callback["state_sequence_sha256"],
                "callback_state_sequence_sha256": callback[
                    "official_integration_state_sequence_sha256"
                ],
                "construction": {
                    "physical_model": {
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
                    },
                    "contact_model_authority_sha256": "e" * 64,
                    "resolved_geometry": {
                        "link56_geom_ids": [5],
                        "robot_geom_ids": [5],
                        "robot_geom_names": ["link5_collision"],
                        "robot_body_ids": [1],
                    },
                    "arm_dof_indices": list(range(7)),
                    "field_bundle": {
                        "hashes": {"protected_samples_sha256": "f" * 64},
                        "protected_sample_count": 1,
                        "protected_samples": [
                            {
                                "sample_id": 0,
                                "body_id": 1,
                                "body_name": "robot0_link5",
                                "geom_id": 5,
                                "geom_name": "link5_collision",
                                "point_body_local_m": [0.0, 0.0, 0.0],
                                "source": "collision_geom_surface",
                            }
                        ],
                    },
                    "full_robot_measurement_sampling": {},
                    "complete_integration_state_read_only_audit": {
                        "mujoco_state_specification": "mjSTATE_INTEGRATION",
                        "state_vector_length": 15,
                        "before_sha256": "0" * 64,
                        "after_sha256": "0" * 64,
                        "exact_array_equal": True,
                        "semantics": "fixture complete official state remained unchanged",
                    },
                    "settled_link56_differential_audit": {
                        "binding_sha256": "1" * 64,
                        "classification_ledger_sha256": "2" * 64,
                    },
                    "settled_link56_differential_audit_validation": {
                        "schema_version": "differential.v1",
                        "audit_payload_sha256": "3" * 64,
                        "passed": True,
                    },
                },
                "poisson_identification": {
                    "contact_prediction_assessment": {
                        "first_link56_contact": {
                            "robot_geom_id": 5,
                            "source_phase": "post_integration_recomputed",
                            "observation_index": 9,
                        },
                        "primary_registered_warning": {
                            "geom_id": 5,
                            "signal_kind": "cbf_lhs_negative",
                            "observation_index": 4,
                        },
                    }
                },
            }
            full_sample_ledger = [
                {
                    "sample_id": 0,
                    "body_id": 1,
                    "body_name": "robot0_link5",
                    "geom_id": 5,
                    "geom_name": "link5_collision",
                    "point_body_local_m": [0.0, 0.0, 0.0],
                    "source": "collision_geom_surface",
                }
            ]
            payload["shadow_replay"]["construction"][
                "full_robot_measurement_sampling"
            ] = {
                "sample_count": 1,
                "sample_ledger_sha256": hashlib.sha256(
                    _canonical(full_sample_ledger)
                ).hexdigest(),
                "geom_records": [{}],
                "epsilon_m": 0.02,
                "maximum_surface_cover_radius_m": 0.01,
                "coverage_semantics": "fixture",
                "rigid_roundtrip": {
                    "sample_count": 1,
                    "maximum_roundtrip_error_m": 0.0,
                    "tolerance_m": 1e-12,
                    "passed": True,
                },
            }
        prerequisite_paths[kind] = root / (kind + ".json")
        final_payload = _hashed(payload)
        _write(prerequisite_paths[kind], final_payload)
        if kind == "parity":
            parity_payload_sha256 = final_payload["result_payload_sha256"]

    digest = "b" * 64
    result = {
        "schema_version": validator.EXPECTED_RESULT_SCHEMA,
        "status": "complete",
        "scientific_result": False,
        "run_id": run_id,
        "case_id": "vlsa-t1-goal-ii-t0-e05",
        "provenance": {
            "joint_velocity_controller": controller_authority[
                "expected_restore_controller_contract"
            ]
        },
        "execution": {
            "outcome_kind": "paired_counterfactual_complete",
            "source_prefix_complete": True,
            "qp_executed": True,
            "paired_joint_velocity_physics_executed": True,
            "no_hidden_clipping": True,
        },
        "source_boundary": {
            "filter_boundary_B": 0,
            "gripper_evidence": {
                "source_action_index": 0,
                "source_action_7d": historical_action,
            },
        },
        "classification": {"label": "strong_causal_prevention"},
        "nominal_velocity_estimate": {"qdot_nom_arm_slice": [0.0] * 7},
        "boundary_B_filter": {
            "one_CBF_row_per_exact_bound_sample": {
                "rows_m_per_rad": [[1.0] + [0.0] * 6],
                "lower_bounds_m2_per_s": [-1.0],
            },
            "joint_velocity_bound_rows": {
                "final_lower_rad_s": [-0.5] * 7,
                "final_upper_rad_s": [0.5] * 7,
            },
            "qdot_safe": [0.0] * 7,
        },
    }
    for field in validator.PROJECTION_HASH_FIELDS:
        result[field] = digest
    result = _hashed(result)
    result_path = run_root / "result.json"
    _write(result_path, result)

    core = {"schema_version": "fixture-core.v1", "passed": True}
    receipt = {
        "schema_version": validator.EXPECTED_RECEIPT_SCHEMA,
        "status": "validated",
        "scientific_result": False,
        "run_id": run_id,
        "case_id": result["case_id"],
        "result": {
            "relative_path": "result.json",
            "file_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "payload_sha256": result["result_payload_sha256"],
            "schema_version": validator.EXPECTED_RESULT_SCHEMA,
        },
        "core_validation": core,
        "producer": {
            "code_commit": commit,
            "slurm_job_id": "12345",
            "host": "worker-1",
            "device": "GPU-uuid",
        },
        "independent_consumer_required": True,
        "external_slurm_requirement": validator.EXTERNAL_SLURM_REQUIREMENT,
    }
    for field in validator.PROJECTION_HASH_FIELDS:
        receipt[field] = result[field]
    receipt_path = run_root / "validation_receipt.json"
    _write(receipt_path, _hashed(receipt))
    arguments = {
        "result_path": result_path,
        "receipt_path": receipt_path,
        "historical_path": historical_path,
        "protocol_path": protocol_path,
        "selection_path": selection_path,
        "runtime_path": runtime_path,
        "numeric_path": prerequisite_paths["numeric"],
        "parity_path": prerequisite_paths["parity"],
        "identification_path": prerequisite_paths["identification"],
        "expected_code_commit": commit,
        "expected_run_id": run_id,
        "expected_slurm_job_id": "12345",
        "expected_host": "worker-1",
        "expected_device": "GPU-uuid",
    }
    return arguments, core


class OneStepArtifactStrictLoaderTest(unittest.TestCase):
    def setUp(self):
        self.installed_source_patch = mock.patch.object(
            validator,
            "_validate_installed_controller_source_identity",
            return_value={"status": "fixture_installed_source_rehashed"},
        )
        self.installed_source_patch.start()
        self.addCleanup(self.installed_source_patch.stop)

    def test_independent_consumer_rehashes_installed_controller_sources(self):
        self.installed_source_patch.stop()
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "robosuite"
            controllers = package / "controllers"
            config = controllers / "config"
            panda = package / "models/assets/robots/panda"
            config.mkdir(parents=True)
            panda.mkdir(parents=True)
            source_path = controllers / "joint_vel.py"
            config_path = config / "joint_velocity.json"
            panda_path = panda / "robot.xml"
            source_path.write_bytes(b"authoritative controller\n")
            config_path.write_bytes(b'{"type":"JOINT_VELOCITY"}\n')
            panda_path.write_bytes(b"<mujoco/>\n")

            protocol = json.loads(
                (
                    Path(__file__).resolve().parents[1]
                    / "configs/vlsa_poisson_one_step_counterfactual.v2.json"
                ).read_text(encoding="utf-8")
            )
            expected = protocol["joint_velocity_controller_authority"][
                "expected_restore_controller_contract"
            ]
            expected["controller_implementation_file_sha256"] = (
                hashlib.sha256(source_path.read_bytes()).hexdigest()
            )
            expected["controller_configuration_file_sha256"] = (
                hashlib.sha256(config_path.read_bytes()).hexdigest()
            )
            expected["panda_robot_xml_file_sha256"] = hashlib.sha256(
                panda_path.read_bytes()
            ).hexdigest()
            result = {
                "provenance": {
                    "joint_velocity_controller": copy.deepcopy(expected)
                }
            }
            with mock.patch.object(
                validator.importlib.util,
                "find_spec",
                return_value=SimpleNamespace(origin=str(source_path)),
            ):
                observed = validator._validate_installed_controller_source_identity(
                    result=result, protocol=protocol
                )
                self.assertEqual(observed["status"], "installed_source_rehashed")
                config_path.write_bytes(b"forged\n")
                with self.assertRaisesRegex(
                    validator.OneStepArtifactValidationError,
                    "source/config hash differs",
                ):
                    validator._validate_installed_controller_source_identity(
                        result=result, protocol=protocol
                    )
                with mock.patch.object(
                    validator.importlib.util, "find_spec", return_value=None
                ), self.assertRaisesRegex(
                    validator.OneStepArtifactValidationError, "unavailable"
                ):
                    validator._validate_installed_controller_source_identity(
                        result=result, protocol=protocol
                    )

    def test_physical_model_contract_requires_exact_hashed_structure(self):
        contract = {
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
        self.assertEqual(
            validator._validate_physical_model_contract(
                contract, "fixture physical model"
            ),
            contract,
        )
        mutations = (
            lambda value: value.__setitem__("sha256", "invalid"),
            lambda value: value.__setitem__("compiled_mjb_sha256", "invalid"),
            lambda value: value.__setitem__("field_count", 0),
            lambda value: value.__setitem__("compiled_mjb_bytes", 0),
            lambda value: value.__setitem__(
                "robosuite_flattened_state_size", 16
            ),
            lambda value: value.__setitem__("extra", True),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                corrupted = copy.deepcopy(contract)
                mutate(corrupted)
                with self.assertRaises(validator.OneStepArtifactValidationError):
                    validator._validate_physical_model_contract(
                        corrupted, "fixture physical model"
                    )

    def test_duplicate_keys_and_nonfinite_constants_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaisesRegex(
                validator.OneStepArtifactValidationError, "duplicate-free"
            ):
                validator._load_strict_json(duplicate, "fixture")

            nonfinite = Path(directory) / "nonfinite.json"
            nonfinite.write_text('{"a":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(
                validator.OneStepArtifactValidationError, "finite"
            ):
                validator._load_strict_json(nonfinite, "fixture")

    def test_symlink_is_rejected_before_loading_or_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            link = root / "link.json"
            _write(target, {"value": 1})
            link.symlink_to(target)
            with self.assertRaisesRegex(
                validator.OneStepArtifactValidationError, "nonsymlink"
            ):
                validator._load_strict_json(link, "fixture")

    def test_rehashed_payload_contract_detects_unrehashed_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            payload = _hashed({"schema_version": "fixture.v1", "status": "passed"})
            _write(path, payload)
            _, loaded = validator._load_strict_json(path, "fixture")
            validator._verify_payload_hash(loaded, "fixture")
            loaded["status"] = "failed"
            with self.assertRaisesRegex(
                validator.OneStepArtifactValidationError, "differs"
            ):
                validator._verify_payload_hash(loaded, "fixture")

    def test_numeric_and_replay_source_shapes_both_require_clean_exact_commit(self):
        commit = "a" * 40
        numeric = {"source": {"commit": commit, "status_short": []}}
        replay = {
            "provenance": {"source": {"commit": commit, "status_short": []}}
        }
        validator._cross_validate_prerequisite_commits(
            {"numeric": numeric, "parity": replay, "identification": replay},
            commit,
        )
        dirty = {"source": {"commit": commit, "status_short": [" M file.py"]}}
        with self.assertRaisesRegex(
            validator.OneStepArtifactValidationError, "expected clean commit"
        ):
            validator._cross_validate_prerequisite_commits(
                {"numeric": dirty}, commit
            )

    def test_consumer_has_no_producer_import_or_artifact_write_path(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "validate_poisson_one_step_counterfactual_artifact.py"
        )
        source = source_path.read_text(encoding="utf-8")
        self.assertNotIn("from scripts.run_poisson_one_step_counterfactual", source)
        self.assertNotIn("import scripts.run_poisson_one_step_counterfactual", source)
        self.assertNotIn("publish_hashed_json", source)
        self.assertNotIn("write_text(", source)
        self.assertIn("external_not_checked_by_this_consumer", source)
        self.assertIn("COMPLETED", validator.EXTERNAL_SLURM_REQUIREMENT)

    def test_slurm_wrapper_is_separate_cpu_only_stdout_consumer(self):
        wrapper_path = (
            Path(__file__).resolve().parents[1]
            / "slurm"
            / "poisson_one_step_counterfactual_validate.sbatch"
        )
        source = wrapper_path.read_text(encoding="utf-8")
        self.assertIn("${SLURM_JOB_ID:?", source)
        self.assertIn(
            '[[ "${SLURM_JOB_ID}" == "${EXPECTED_PRODUCER_JOB_ID}" ]]',
            source,
        )
        self.assertIn(
            "scripts/validate_poisson_one_step_counterfactual_artifact.py",
            source,
        )
        self.assertNotIn("scripts/run_poisson_one_step_counterfactual.py", source)
        self.assertNotIn("#SBATCH --gres", source)
        self.assertNotIn("#SBATCH --partition", source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", source)
        self.assertNotIn("validation_receipt.json\" >", source)
        self.assertIn('--selection-protocol "${SELECTION_RELATIVE_PATH}"', source)
        self.assertIn('--runtime-protocol "${RUNTIME_RELATIVE_PATH}"', source)
        self.assertIn('--historical-result "${HISTORICAL_RESULT_PATH}"', source)
        self.assertNotIn("EXPECTED_NOMINAL_DERIVATION_SHA256", source)
        for name in (
            "HISTORICAL_RESULT_PATH",
            "NUMERIC_VALIDATION_RESULT",
            "PARITY_RESULT",
            "SHADOW_IDENTIFICATION_RESULT",
            "EXPECTED_GIT_COMMIT",
            "EXPECTED_PRODUCER_JOB_ID",
            "EXPECTED_PRODUCER_HOST",
            "EXPECTED_PRODUCER_DEVICE",
        ):
            self.assertIn('${%s:?' % name, source)

    def test_receipt_explicitly_projects_claim_bearing_provenance(self):
        self.assertIn("provenance_sha256", validator.RECEIPT_FIELDS)
        self.assertIn("provenance_sha256", validator.PROJECTION_HASH_FIELDS)

    def test_receipt_is_an_exact_projection_of_result_core_and_external_producer(self):
        digest = "b" * 64
        result = {
            "run_id": "stage13-run",
            "case_id": "vlsa-t1-goal-ii-t0-e05",
        }
        for field in validator.PROJECTION_HASH_FIELDS:
            result[field] = digest
        core = {"schema_version": "core-validation.v1", "passed": True}
        receipt = {
            "schema_version": validator.EXPECTED_RECEIPT_SCHEMA,
            "status": "validated",
            "scientific_result": False,
            "run_id": result["run_id"],
            "case_id": result["case_id"],
            "result": {
                "relative_path": "result.json",
                "file_sha256": "c" * 64,
                "payload_sha256": "d" * 64,
                "schema_version": validator.EXPECTED_RESULT_SCHEMA,
            },
            "core_validation": core,
            "producer": {
                "code_commit": "a" * 40,
                "slurm_job_id": "12345",
                "host": "worker-1",
                "device": "GPU-uuid",
            },
            "independent_consumer_required": True,
            "external_slurm_requirement": validator.EXTERNAL_SLURM_REQUIREMENT,
            "result_payload_sha256": "e" * 64,
        }
        for field in validator.PROJECTION_HASH_FIELDS:
            receipt[field] = digest

        arguments = {
            "receipt": receipt,
            "result": result,
            "result_file_sha256": "c" * 64,
            "result_payload_sha256": "d" * 64,
            "protocol_identity": {"semantic_sha256": "f" * 64},
            "authorities": {},
            "core": core,
            "expected_code_commit": "a" * 40,
            "expected_slurm_job_id": "12345",
            "expected_host": "worker-1",
            "expected_device": "GPU-uuid",
        }
        validator._validate_receipt_binding(**arguments)

        mutations = {
            "unknown_key": lambda value: value.__setitem__("extra", 1),
            "result_file": lambda value: value["result"].__setitem__(
                "file_sha256", "0" * 64
            ),
            "projection": lambda value: value.__setitem__(
                "arm_ledger_sha256", "0" * 64
            ),
            "core": lambda value: value["core_validation"].__setitem__(
                "passed", False
            ),
            "producer": lambda value: value["producer"].__setitem__(
                "host", "wrong-worker"
            ),
            "slurm_requirement": lambda value: value.__setitem__(
                "external_slurm_requirement", "producer_claims_completed"
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                invalid = copy.deepcopy(receipt)
                mutate(invalid)
                invalid_arguments = dict(arguments)
                invalid_arguments["receipt"] = invalid
                with self.assertRaises(validator.OneStepArtifactValidationError):
                    validator._validate_receipt_binding(**invalid_arguments)

    def test_complete_consumer_supplies_external_file_and_payload_authorities(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments, core = _consumer_fixture(Path(directory))
            differential_validation = {
                "schema_version": "differential.v1",
                "audit_payload_sha256": "3" * 64,
                "passed": True,
            }
            with mock.patch(
                "main.poisson_fullbody.feasibility_protocol."
                "validate_feasibility_protocol",
                return_value=SimpleNamespace(
                    protocol_sha256="2" * 64,
                    parameter_block_sha256="3" * 64,
                ),
            ) as runtime_pure, mock.patch(
                "main.poisson_fullbody.jacobians."
                "validate_protected_sample_differential_audit",
                return_value=differential_validation,
            ) as differential_pure, mock.patch(
                "main.poisson_fullbody.surface_sampling."
                "validate_robot_sample_evidence",
                return_value={"mesh": 1, "box": 0, "cylinder": 0},
            ) as sampling_pure, mock.patch(
                "main.poisson_fullbody.cbf_qp.solve_reference_cvxpy",
                return_value=[0.0] * 7,
            ) as qp_reference_pure, mock.patch(
                "main.poisson_fullbody.one_step_counterfactual."
                "validate_one_step_counterfactual_result",
                return_value=core,
            ) as pure:
                summary = validator.validate_one_step_artifacts(**arguments)
            self.assertEqual(summary["status"], "valid")
            self.assertEqual(
                summary["slurm_terminal_state_validation"],
                "external_not_checked_by_this_consumer",
            )
            call = pure.call_args
            self.assertEqual(call.kwargs["protocol_raw_sha256"], hashlib.sha256(
                arguments["protocol_path"].read_bytes()
            ).hexdigest())
            authority = call.kwargs["expected_authority"]
            for kind in ("numeric", "parity", "identification"):
                record = authority[kind + "_prerequisite"]
                self.assertRegex(record["file_sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(record["payload_sha256"], r"^[0-9a-f]{64}$")
            dynamic = authority["dynamic_authority"]
            self.assertEqual(
                dynamic["selection"],
                {
                    "relative_path": "configs/selection.json",
                    "schema_version": "selection.v1",
                    "file_sha256": hashlib.sha256(
                        arguments["selection_path"].read_bytes()
                    ).hexdigest(),
                    "protocol_id": "selection-protocol-v1",
                },
            )
            self.assertEqual(dynamic["historical"]["action_count"], 1)
            self.assertEqual(
                dynamic["runtime"]["registered_parameters"],
                {
                    "admissibility": {"fixture": True},
                    "coverage": {"fixture": True},
                    "cbf": {"fixture": True},
                    "qp": {"fixture": True, "weight_diagonal": [1.0] * 7},
                    "cadence": {"fixture": True},
                },
            )
            self.assertEqual(
                dynamic["parity"]["settled_official_integration_state"],
                {
                    "physical_boundary": 0,
                    "mujoco_state_specification": "mjSTATE_INTEGRATION",
                    "state_vector_length": 15,
                    "sha256": "0" * 64,
                },
            )
            self.assertEqual(
                dynamic["parity"]["official_integration_state_sha256_ledger"],
                ["c" * 64],
            )
            self.assertEqual(
                dynamic["identification"][
                    "callback_state_read_only_after_sha256_ledger"
                ],
                ["c" * 64],
            )
            self.assertEqual(
                dynamic["identification"]["full_robot_measurement_sampling"][
                    "sample_count"
                ],
                1,
            )
            self.assertEqual(
                dynamic["identification"]["primary_registered_warning"][
                    "signal_kind"
                ],
                "cbf_lhs_negative",
            )
            self.assertEqual(
                dynamic["identification"]["physical_model"],
                {
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
                },
            )
            self.assertNotIn("expected_nominal_derivation_sha256", call.kwargs)
            self.assertEqual(runtime_pure.call_count, 1)
            self.assertEqual(differential_pure.call_count, 1)
            self.assertEqual(sampling_pure.call_count, 1)
            self.assertEqual(qp_reference_pure.call_count, 1)
            self.assertTrue(summary["independent_qp_reference"]["passed"])
            self.assertEqual(
                summary["independent_qp_reference"]["qdot_linf_error_rad_s"],
                0.0,
            )
            differential_call = differential_pure.call_args
            self.assertEqual(
                differential_call.kwargs["expected_integration_state_sha256"],
                "0" * 64,
            )
            self.assertEqual(
                differential_call.kwargs["expected_differential_audit_config"],
                {"fixture": True},
            )

    def test_complete_consumer_rejects_a_rehashed_nonpassing_prerequisite(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments, core = _consumer_fixture(Path(directory))
            parity_path = arguments["parity_path"]
            _, parity = validator._load_strict_json(parity_path, "parity")
            parity.pop("result_payload_sha256")
            parity["status"] = "failed"
            _write(parity_path, _hashed(parity))
            with mock.patch(
                "main.poisson_fullbody.one_step_counterfactual."
                "validate_one_step_counterfactual_result",
                return_value=core,
            ), self.assertRaisesRegex(
                validator.OneStepArtifactValidationError, "not a complete passing"
            ):
                validator.validate_one_step_artifacts(**arguments)

    def test_complete_consumer_rejects_rehashed_false_or_empty_acceptance(self):
        for kind in ("numeric", "parity", "identification"):
            for acceptance in ({"forged_pass": False}, {}):
                with self.subTest(kind=kind, acceptance=acceptance):
                    with tempfile.TemporaryDirectory() as directory:
                        arguments, _ = _consumer_fixture(Path(directory))
                        prerequisite_path = arguments[kind + "_path"]
                        _, prerequisite = validator._load_strict_json(
                            prerequisite_path, kind
                        )
                        prerequisite.pop("result_payload_sha256")
                        prerequisite["acceptance"] = acceptance
                        _write(prerequisite_path, _hashed(prerequisite))
                        with self.assertRaisesRegex(
                            validator.OneStepArtifactValidationError,
                            "acceptance",
                        ):
                            validator.validate_one_step_artifacts(**arguments)

    def test_parity_v3_requires_complete_acceptance_and_matching_boundary_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments, _ = _consumer_fixture(Path(directory))
            parity_path = arguments["parity_path"]
            _, parity = validator._load_strict_json(parity_path, "parity")
            incomplete = {
                key: value
                for key, value in parity.items()
                if key != "result_payload_sha256"
            }
            incomplete = copy.deepcopy(incomplete)
            incomplete["acceptance"].pop(
                "ordinary_and_callback_boundary_0_mjstate_integration_exact"
            )
            _write(parity_path, _hashed(incomplete))
            with self.assertRaisesRegex(
                validator.OneStepArtifactValidationError,
                "acceptance fields differ",
            ):
                validator.validate_one_step_artifacts(**arguments)

        with tempfile.TemporaryDirectory() as directory:
            arguments, _ = _consumer_fixture(Path(directory))
            _, parity = validator._load_strict_json(
                arguments["parity_path"], "parity"
            )
            _, identification = validator._load_strict_json(
                arguments["identification_path"], "identification"
            )
            parity = copy.deepcopy(parity)
            parity["ordinary_replay"][
                "settled_official_integration_state"
            ]["sha256"] = "9" * 64
            with self.assertRaisesRegex(
                validator.OneStepArtifactValidationError,
                "ordinary/callback boundary-0 integration states differ",
            ):
                validator._dynamic_authority(
                    protocol={"case": {}},
                    selection_identity={},
                    runtime_protocol={},
                    parity=parity,
                    identification=identification,
                )

    def test_rehashed_identification_settled_state_remains_bound_to_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments, _ = _consumer_fixture(Path(directory))
            _, identification = validator._load_strict_json(
                arguments["identification_path"], "identification"
            )
            identification.pop("result_payload_sha256")
            read_only = identification["shadow_replay"]["construction"][
                "complete_integration_state_read_only_audit"
            ]
            read_only["before_sha256"] = "9" * 64
            read_only["after_sha256"] = "9" * 64
            _write(
                arguments["identification_path"],
                _hashed(identification),
            )
            with mock.patch(
                "main.poisson_fullbody.feasibility_protocol."
                "validate_feasibility_protocol",
                return_value=SimpleNamespace(
                    protocol_sha256="2" * 64,
                    parameter_block_sha256="3" * 64,
                ),
            ), self.assertRaisesRegex(
                validator.OneStepArtifactValidationError,
                "settled state authority differs from parity",
            ):
                validator.validate_one_step_artifacts(**arguments)

    def test_rehashed_dynamic_authority_tamper_reaches_real_core(self):
        """A self-consistent upstream rewrite must not inherit old result authority."""

        with tempfile.TemporaryDirectory() as directory:
            arguments, fixture_core = _consumer_fixture(Path(directory))
            differential_validation = {
                "schema_version": "differential.v1",
                "audit_payload_sha256": "3" * 64,
                "passed": True,
            }
            support_patches = (
                mock.patch(
                    "main.poisson_fullbody.feasibility_protocol."
                    "validate_feasibility_protocol",
                    return_value=SimpleNamespace(
                        protocol_sha256="2" * 64,
                        parameter_block_sha256="3" * 64,
                    ),
                ),
                mock.patch(
                    "main.poisson_fullbody.jacobians."
                    "validate_protected_sample_differential_audit",
                    return_value=differential_validation,
                ),
                mock.patch(
                    "main.poisson_fullbody.surface_sampling."
                    "validate_robot_sample_evidence",
                    return_value={"mesh": 1, "box": 0, "cylinder": 0},
                ),
                mock.patch(
                    "main.poisson_fullbody.cbf_qp.solve_reference_cvxpy",
                    return_value=[0.0] * 7,
                ),
            )
            with support_patches[0], support_patches[1], support_patches[2], support_patches[3], mock.patch(
                "main.poisson_fullbody.one_step_counterfactual."
                "validate_one_step_counterfactual_result",
                return_value=fixture_core,
            ) as captured_core:
                validator.validate_one_step_artifacts(**arguments)
            original_authority = copy.deepcopy(
                captured_core.call_args.kwargs["expected_authority"]
            )

            # Make the result structurally complete enough to enter the real
            # pure core.  Its claim-bearing authority remains the original
            # externally reconstructed mapping.
            from main.poisson_fullbody import one_step_counterfactual as core_module

            digest = "b" * 64
            result = {key: {} for key in core_module.TOP_LEVEL_KEYS}
            result.update(
                {
                    "schema_version": validator.EXPECTED_RESULT_SCHEMA,
                    "status": "complete",
                    "scientific_result": True,
                    "run_id": arguments["expected_run_id"],
                    "case_id": "vlsa-t1-goal-ii-t0-e05",
                    "authority": original_authority,
                    "source_boundary": {
                        "filter_boundary_B": 0,
                        "gripper_evidence": {
                            "source_action_index": 0,
                            "source_action_7d": [
                                0.1,
                                0.2,
                                0.3,
                                0.4,
                                0.5,
                                0.6,
                                0.7,
                            ],
                        },
                    },
                    "arms": [],
                    "passed": False,
                }
            )
            for field in validator.PROJECTION_HASH_FIELDS:
                result[field] = digest
            result.pop("result_payload_sha256", None)
            _write(arguments["result_path"], _hashed(result))

            # Rewrite parity and identification coherently, including every
            # direct cross-link and both payload hashes.  The consumer accepts
            # this as a new external authority; the already-final result must
            # still be rejected by the actual core authority comparison.
            _, parity = validator._load_strict_json(
                arguments["parity_path"], "parity"
            )
            parity.pop("result_payload_sha256")
            rewritten_action_states = ["d" * 64]
            parity["callback_replay"][
                "action_boundary_state_sha256_ledger"
            ] = rewritten_action_states
            parity["callback_replay"]["state_sequence_sha256"] = hashlib.sha256(
                _canonical(rewritten_action_states)
            ).hexdigest()
            rewritten_parity = _hashed(parity)
            _write(arguments["parity_path"], rewritten_parity)

            _, identification = validator._load_strict_json(
                arguments["identification_path"], "identification"
            )
            identification.pop("result_payload_sha256")
            identification["provenance"]["upstream_parity_payload_sha256"] = (
                rewritten_parity["result_payload_sha256"]
            )
            identification["shadow_replay"][
                "action_boundary_state_sha256_ledger"
            ] = rewritten_action_states
            identification["shadow_replay"]["state_sequence_sha256"] = (
                parity["callback_replay"]["state_sequence_sha256"]
            )
            _write(arguments["identification_path"], _hashed(identification))

            support_patches = (
                mock.patch(
                    "main.poisson_fullbody.feasibility_protocol."
                    "validate_feasibility_protocol",
                    return_value=SimpleNamespace(
                        protocol_sha256="2" * 64,
                        parameter_block_sha256="3" * 64,
                    ),
                ),
                mock.patch(
                    "main.poisson_fullbody.jacobians."
                    "validate_protected_sample_differential_audit",
                    return_value=differential_validation,
                ),
                mock.patch(
                    "main.poisson_fullbody.surface_sampling."
                    "validate_robot_sample_evidence",
                    return_value={"mesh": 1, "box": 0, "cylinder": 0},
                ),
            )
            with support_patches[0], support_patches[1], support_patches[2], self.assertRaisesRegex(
                core_module.OneStepCounterfactualError,
                "result authority and external authority differs",
            ):
                validator.validate_one_step_artifacts(**arguments)

    def test_collusively_rehashed_result_cannot_substitute_gripper_action(self):
        """Result-local rewrites cannot replace the immutable historical action."""

        with tempfile.TemporaryDirectory() as directory:
            arguments, fixture_core = _consumer_fixture(Path(directory))
            _, result = validator._load_strict_json(
                arguments["result_path"], "Stage-13 result"
            )
            result.pop("result_payload_sha256")
            result["source_boundary"]["gripper_evidence"][
                "source_action_7d"
            ][6] = -0.7
            _write(arguments["result_path"], _hashed(result))

            differential_validation = {
                "schema_version": "differential.v1",
                "audit_payload_sha256": "3" * 64,
                "passed": True,
            }
            with mock.patch(
                "main.poisson_fullbody.feasibility_protocol."
                "validate_feasibility_protocol",
                return_value=SimpleNamespace(
                    protocol_sha256="2" * 64,
                    parameter_block_sha256="3" * 64,
                ),
            ), mock.patch(
                "main.poisson_fullbody.jacobians."
                "validate_protected_sample_differential_audit",
                return_value=differential_validation,
            ), mock.patch(
                "main.poisson_fullbody.surface_sampling."
                "validate_robot_sample_evidence",
                return_value={"mesh": 1, "box": 0, "cylinder": 0},
            ), mock.patch(
                "main.poisson_fullbody.one_step_counterfactual."
                "validate_one_step_counterfactual_result",
                return_value=fixture_core,
            ) as core:
                with self.assertRaisesRegex(
                    validator.OneStepArtifactValidationError,
                    "gripper source action differs from exact historical action",
                ):
                    validator.validate_one_step_artifacts(**arguments)
            core.assert_not_called()

    def test_independent_qp_reference_rejects_an_arbitrary_safe_command(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments, _ = _consumer_fixture(Path(directory))
            _, result = validator._load_strict_json(
                arguments["result_path"], "result"
            )
            _, protocol = validator._load_strict_json(
                arguments["protocol_path"], "protocol"
            )
            _, runtime = validator._load_strict_json(
                arguments["runtime_path"], "runtime"
            )
            with mock.patch(
                "main.poisson_fullbody.cbf_qp.solve_reference_cvxpy",
                return_value=[0.1] + [0.0] * 6,
            ) as solve:
                with self.assertRaisesRegex(
                    validator.OneStepArtifactValidationError,
                    "differs from the independent QP optimum",
                ):
                    validator._validate_independent_qp_reference(
                        result=result,
                        protocol=protocol,
                        runtime_protocol=runtime,
                    )
            call = solve.call_args
            self.assertEqual(call.args[0], [0.0] * 7)
            self.assertEqual(call.args[1], [[1.0] + [0.0] * 6])
            self.assertEqual(call.args[2], [-1.0])
            self.assertEqual(call.args[3], [-0.5] * 7)
            self.assertEqual(call.args[4], [0.5] * 7)
            self.assertEqual(call.kwargs["weight_diagonal"], [1.0] * 7)

    def test_inadmissible_nominal_is_reconstructed_without_running_qp(self):
        lower = [-0.5] * 7
        upper = [0.5] * 7
        qpos_B = [0.0] * 7
        qpos_B5 = [0.002, 0.012, -0.003, 0.014, 0.0, 0.0, 0.0]
        qdot = [
            (right - left) / 0.01 for left, right in zip(qpos_B, qpos_B5)
        ]
        lower_excess = [
            max(lower[index] - value, 0.0)
            for index, value in enumerate(qdot)
        ]
        upper_excess = [
            max(value - upper[index], 0.0)
            for index, value in enumerate(qdot)
        ]
        result = {
            "execution": {
                "outcome_kind": "preflight_inadmissible_nominal_velocity",
                "source_prefix_complete": True,
                "qp_executed": False,
                "paired_joint_velocity_physics_executed": False,
                "no_hidden_clipping": True,
            },
            "source_boundary": {
                "exact_prefix": {
                    "state_at_B": {
                        "qpos": qpos_B,
                        "integration_state": [0.0] + qpos_B + [0.0] * 7,
                    },
                    "state_at_B_plus_5": {
                        "qpos": qpos_B5,
                        "integration_state": [0.01] + qpos_B5 + [0.0] * 7,
                    },
                }
            },
            "nominal_velocity_estimate": {
                "method": "mujoco_mj_differentiatePos_full_nv",
                "interval_s": 0.01,
                "arm_qpos_indices": list(range(7)),
                "arm_joint_types": ["hinge"] * 7,
                "arm_joint_qpos_widths": [1] * 7,
                "qdot_nom_arm_slice": qdot,
                "registered_lower_rad_s": lower,
                "registered_upper_rad_s": upper,
                "finite": True,
                "arm_velocity_within_registered_bounds": False,
                "bound_violation_indices": [1, 3],
                "lower_bound_excess_rad_s": lower_excess,
                "upper_bound_excess_rad_s": upper_excess,
                "maximum_bound_excess_rad_s": max(
                    lower_excess + upper_excess
                ),
                "out_of_bounds_policy": "inadmissible_no_hidden_clipping",
                "hidden_clipping_applied": False,
            },
            "boundary_B_filter": None,
            "arms": [],
            "diagnostics": None,
        }
        protocol = {
            "qp_execution": {
                "independent_postproducer_reference_check": {
                    "execution": "separate_artifact_consumer_after_producer_exit",
                    "solver": "cvxpy_osqp",
                    "eps_abs": 1e-9,
                    "eps_rel": 1e-9,
                    "max_iterations": 20000,
                    "qdot_linf_tolerance_rad_s": 2e-5,
                    "required_before_scientific_interpretation": True,
                }
            },
            "counterfactual_boundary": {"counterfactual_horizon_s": 0.01},
            "nominal_velocity_estimator": {
                "expected_arm_qpos_indices": list(range(7)),
                "arm_velocity_lower_rad_s": lower,
                "arm_velocity_upper_rad_s": upper,
            },
        }
        runtime = {
            "qp": {
                "velocity_lower_rad_s": lower,
                "velocity_upper_rad_s": upper,
            }
        }
        with mock.patch(
            "main.poisson_fullbody.cbf_qp.solve_reference_cvxpy"
        ) as solver:
            observed = validator._validate_independent_qp_reference(
                result=result,
                protocol=protocol,
                runtime_protocol=runtime,
            )
        solver.assert_not_called()
        self.assertEqual(
            observed["status"],
            "not_run_by_preregistered_inadmissibility",
        )
        self.assertEqual(observed["bound_violation_indices"], [1, 3])
        self.assertFalse(observed["solver_executed"])
        self.assertNotIn("passed", observed)

        for label, mutation in (
            (
                "endpoint mismatch",
                lambda value: value["nominal_velocity_estimate"][
                    "qdot_nom_arm_slice"
                ].__setitem__(1, 0.4),
            ),
            (
                "forbidden QP evidence",
                lambda value: value.__setitem__("boundary_B_filter", {}),
            ),
        ):
            with self.subTest(tamper=label):
                corrupted = copy.deepcopy(result)
                mutation(corrupted)
                with self.assertRaises(
                    validator.OneStepArtifactValidationError
                ):
                    validator._validate_independent_qp_reference(
                        result=corrupted,
                        protocol=protocol,
                        runtime_protocol=runtime,
                    )


if __name__ == "__main__":
    unittest.main()
