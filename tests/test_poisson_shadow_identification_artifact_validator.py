from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from scripts import validate_poisson_shadow_identification_artifact as validator


ROOT = Path(__file__).resolve().parents[1]


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
    output["result_payload_sha256"] = hashlib.sha256(
        _canonical(output)
    ).hexdigest()
    return output


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source(commit):
    return {
        "commit": commit,
        "branch": "codex/full-body-poisson-cbf-feasibility",
        "status_short": [],
    }


def _warning_contact_shadow():
    warning_observation = 4507
    contact_observation = 4696
    warning = {
        "observation_index": warning_observation,
        "high_level_index": warning_observation // 25,
        "physics_substep_index": warning_observation % 25,
        "signal_kind": "cbf_lhs_negative",
        "geom_id": 44,
        "geom_name": "robot0_link5_collision",
        "body_id": 5,
        "body_name": "robot0_link5",
        "evidence": {"observed_cbf_lhs_m2_per_s": -0.005},
    }
    contact = {
        "source_phase": "post_integration_recomputed",
        "observation_index": contact_observation,
        "robot_geom_id": 44,
        "robot_geom_name": "robot0_link5_collision",
        "is_physical_nonpositive_distance_contact": True,
    }
    identification = {
        "observed_callback_count": validator.CALLBACK_COUNT,
        "expected_callback_count": validator.CALLBACK_COUNT,
        "trace": [{}] * validator.CALLBACK_COUNT,
        "signals": {
            "first_observed_minimum_cbf_lhs_negative_on_contact_geom": warning,
        },
        "contact_prediction_assessment": {
            "assessment": "registered_warning_preceded_link56_contact",
            "first_link56_contact": contact,
            "primary_registered_warning": warning,
            "lead_physics_substeps": 189,
            "lead_time_s": 0.378,
        },
    }
    return identification


class _CompleteFixture:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.commit = "a" * 40
        self.job_id = "33712"
        self.numeric_job_id = "33708"
        self.parity_job_id = "33709"
        self.host = "worker-mig-3g40gb-0"
        self.device = "NVIDIA H100 80GB HBM3"
        self.case_id = validator.EXPECTED_CASE_ID

        configs = self.root / "configs"
        selection_path = configs / "selection.json"
        runtime_path = configs / "runtime.json"
        _write(runtime_path, {"fixture": "runtime"})
        self.runtime_raw = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
        self.runtime_semantic = "1" * 64
        self.runtime_parameters = "2" * 64
        selection = {
            "schema_version": "selection.v1",
            "protocol_id": "selection-protocol-v1",
            "runtime_protocol": {
                "relative_path": "configs/runtime.json",
                "raw_file_sha256": self.runtime_raw,
                "semantic_protocol_sha256": self.runtime_semantic,
                "parameter_block_sha256": self.runtime_parameters,
            },
        }
        _write(selection_path, selection)
        selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()

        historical_root = self.root / "historical"
        historical_path = historical_root / "result.json"
        historical_path.parent.mkdir()
        historical_path.write_bytes(b"historical-fixture\n")
        historical_file_sha = hashlib.sha256(
            historical_path.read_bytes()
        ).hexdigest()
        historical_payload_sha = "3" * 64
        manifest_row = {
            "case_id": self.case_id,
            "split": "bringup_canary",
            "study_partition": "development",
            "settle_actions": 20,
            "protocol_config_sha256": selection_sha,
            "historical_aegis_result": {
                "source_relative_path": "result.json",
                "raw_file_sha256": historical_file_sha,
                "result_payload_sha256": historical_payload_sha,
            },
        }
        manifest_path = self.root / "manifest.jsonl"
        manifest_line = _canonical(manifest_row)
        manifest_path.write_bytes(manifest_line + b"\n")
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        manifest_row_sha = hashlib.sha256(manifest_line).hexdigest()

        action_states = [
            hashlib.sha256(("state-%d" % index).encode("ascii")).hexdigest()
            for index in range(validator.ACTION_COUNT)
        ]
        callback_hash = "4" * 64
        official_states = [callback_hash] * validator.CALLBACK_COUNT
        state_sequence = hashlib.sha256(_canonical(action_states)).hexdigest()
        callback_sequence = hashlib.sha256(_canonical(official_states)).hexdigest()
        observation_sequence = "5" * 64
        replay_provenance = {
            "case_id": self.case_id,
            "action_count": validator.ACTION_COUNT,
            "historical_result_file_sha256": historical_file_sha,
            "historical_result_payload_sha256": historical_payload_sha,
            "terminal_simulator_state_sha256": action_states[-1],
        }
        self.replay = SimpleNamespace(
            steps=[
                SimpleNamespace(simulator_state_sha256=value)
                for value in action_states
            ],
            actions=[(0.0,) * 7] * validator.ACTION_COUNT,
            result_file_sha256=historical_file_sha,
            result_payload_sha256=historical_payload_sha,
            executed_sequence_sha256="6" * 64,
            terminal_simulator_state_sha256=action_states[-1],
            provenance=lambda: replay_provenance,
        )

        from scripts.run_poisson_numeric_validation import DEFAULT_TEST_MODULES

        numeric = _hashed(
            {
                "schema_version": validator.EXPECTED_NUMERIC_SCHEMA,
                "status": "passed",
                "scientific_result": False,
                "evidence_tier": validator.NUMERIC_EVIDENCE_TIER,
                "claim_limit": validator.NUMERIC_CLAIM_LIMIT,
                "created_at_unix": 10,
                "duration_seconds": 1.0,
                "source": _source(self.commit),
                "runtime": {
                    "python_executable": validator.REGISTERED_EVALUATION_PYTHON,
                    "python_version": "3.8.18",
                    "host": self.host,
                    "slurm_job_id": self.numeric_job_id,
                    "slurm_job_name": validator.EXPECTED_NUMERIC_JOB_NAME,
                    "slurm_partition": "mig",
                    "cuda_visible_devices": "0",
                    "production_grid_validation_enabled": True,
                    "allocated_gpu_inventory": {
                        "available": True,
                        "devices": [
                            {
                                "name": self.device,
                                "uuid": "MIG-GPU-numeric-fixture",
                                "driver_version": "555.42.06",
                            }
                        ],
                        "error": None,
                    },
                    "packages": {
                        "numpy": "1.24.4",
                        "scipy": "1.10.1",
                        "mujoco": "3.2.3",
                        "cvxpy": "1.5.3",
                        "osqp": "0.6.7",
                    },
                },
                "tests": {
                    "modules": list(DEFAULT_TEST_MODULES),
                    "minimum_tests": 30,
                    "tests_run": 250,
                    "failure_count": 0,
                    "error_count": 0,
                    "skip_count": 0,
                    "failures": [],
                    "errors": [],
                    "skipped": [],
                    "output": "fixture\nRan 250 tests in 1.000s\n\nOK\n",
                },
                "acceptance": {
                    field: True
                    for field in validator.NUMERIC_ACCEPTANCE_FIELDS
                },
            }
        )
        numeric_path = self.root / "numeric.json"
        _write(numeric_path, numeric)

        parity = _hashed(
            {
                "schema_version": validator.EXPECTED_PARITY_SCHEMA,
                "status": "passed",
                "scientific_result": False,
                "evidence_tier": validator.PARITY_EVIDENCE_TIER,
                "claim_limit": validator.PARITY_CLAIM_LIMIT,
                "case_id": self.case_id,
                "timing": {
                    "started_unix": 10.0,
                    "finished_unix": 12.0,
                    "elapsed_seconds": 2.0,
                },
                "provenance": {
                    "source": _source(self.commit),
                    "manifest_path": str(manifest_path.resolve()),
                    "manifest_sha256": manifest_sha,
                    "manifest_line_number": 1,
                    "manifest_row_sha256": manifest_row_sha,
                    "historical_result_path": str(historical_path.resolve()),
                    "historical_result_file_sha256": historical_file_sha,
                    "historical_result_payload_sha256": historical_payload_sha,
                    "host": self.host,
                    "slurm_job_id": self.parity_job_id,
                    "slurm_job_name": validator.EXPECTED_PARITY_JOB_NAME,
                    "python_executable": validator.REGISTERED_EVALUATION_PYTHON,
                    "python_version": "3.8.18",
                    "packages": {
                        "numpy": "1.24.4",
                        "mujoco": "3.2.3",
                        "robosuite": "1.4.1",
                        "scipy": "1.10.1",
                    },
                    "gpu": {
                        "devices": [
                            {
                                "name": self.device,
                                "uuid": "MIG-GPU-parity-fixture",
                                "driver_version": "555.42.06",
                            }
                        ]
                    },
                },
                "historical": replay_provenance,
                "ordinary_replay": {},
                "callback_replay": {
                    "state_sequence_sha256": state_sequence,
                    "observation_sequence_sha256": observation_sequence,
                    "terminal_simulator_state_sha256": action_states[-1],
                    "official_integration_state_sha256_ledger": official_states,
                    "official_integration_state_sequence_sha256": callback_sequence,
                },
                "acceptance": {
                    field: True
                    for field in validator.PARITY_ACCEPTANCE_FIELDS
                },
            }
        )
        parity_path = self.root / "parity.json"
        _write(parity_path, parity)

        callback_read_only = [
            {
                "before_sha256": callback_hash,
                "after_sha256": callback_hash,
                "exact_array_equal": True,
            }
        ] * validator.CALLBACK_COUNT
        callback_read_only_sha256 = hashlib.sha256(
            _canonical(callback_read_only)
        ).hexdigest()
        poisson_identification = _warning_contact_shadow()
        shadow = {
            "executed_action_count": validator.ACTION_COUNT,
            "callback_count": validator.CALLBACK_COUNT,
            "expected_callback_count": validator.CALLBACK_COUNT,
            "action_boundary_state_sha256_ledger": action_states,
            "state_sequence_sha256": state_sequence,
            "observation_sequence_sha256": observation_sequence,
            "callback_state_read_only_ledger": callback_read_only,
            "callback_state_read_only_ledger_sha256": (
                callback_read_only_sha256
            ),
            "callback_state_sequence_sha256": callback_sequence,
            "terminal_simulator_state_sha256": action_states[-1],
            "measurement": {
                "observed_physics_substeps": validator.CALLBACK_COUNT,
                "first_index": [0, 0, 0],
                "last_index": [236, 4, 4],
            },
            "construction": {
                "settled_link56_differential_audit_validation": {
                    "passed": True,
                    "counts": {
                        "sample_count": 1531,
                        "point_jacobian_passed_sample_count": 1531,
                        "passed_coupled_direction_count": 13779,
                        "required_coupled_direction_count": 13779,
                    },
                }
            },
            "monitor_static_drift_exception_count": 0,
            "monitor_static_drift_exceptions": [],
            "poisson_identification": poisson_identification,
        }
        provenance = {
            "source": _source(self.commit),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": manifest_sha,
            "manifest_line_number": 1,
            "manifest_row_sha256": manifest_row_sha,
            "selection_config_path": str(selection_path.resolve()),
            "selection_config_sha256": selection_sha,
            "selection_protocol_id": selection["protocol_id"],
            "runtime_protocol_path": str(runtime_path.resolve()),
            "runtime_protocol_raw_sha256": self.runtime_raw,
            "runtime_protocol_semantic_sha256": self.runtime_semantic,
            "runtime_parameter_block_sha256": self.runtime_parameters,
            "historical_result_path": str(historical_path.resolve()),
            "historical_result_file_sha256": historical_file_sha,
            "historical_result_payload_sha256": historical_payload_sha,
            "upstream_parity_path": str(parity_path.resolve()),
            "upstream_parity_payload_sha256": parity[
                "result_payload_sha256"
            ],
            "host": self.host,
            "slurm_job_id": self.job_id,
            "slurm_job_name": validator.EXPECTED_PRODUCER_JOB_NAME,
            "python_executable": "/mnt/data/quanth/venvs/safety_vla/main/bin/python",
            "python_version": "3.8.18",
            "packages": {
                "numpy": "1.24.4",
                "mujoco": "3.2.3",
                "robosuite": "1.4.1",
                "scipy": "1.10.1",
            },
            "gpu": {
                "devices": [
                    {
                        "name": self.device,
                        "uuid": "MIG-GPU-fixture",
                        "driver_version": "555.42.06",
                    }
                ]
            },
        }
        result = _hashed(
            {
                "schema_version": validator.EXPECTED_SCHEMA,
                "status": "passed",
                "phase": "complete",
                "scientific_result": False,
                "evidence_tier": validator.IDENTIFICATION_EVIDENCE_TIER,
                "claim_limit": validator.IDENTIFICATION_CLAIM_LIMIT,
                "case_id": self.case_id,
                "timing": {
                    "started_unix": 10.0,
                    "finished_unix": 12.0,
                    "elapsed_seconds": 2.0,
                },
                "provenance": provenance,
                "historical": replay_provenance,
                "shadow_replay": shadow,
                "acceptance": {
                    field: True
                    for field in validator.IDENTIFICATION_ACCEPTANCE_FIELDS
                },
            }
        )
        result_path = self.root / "identification.json"
        _write(result_path, result)

        self.paths = {
            "identification_path": result_path,
            "numeric_path": numeric_path,
            "parity_path": parity_path,
            "manifest_path": manifest_path,
            "selection_path": selection_path,
            "runtime_path": runtime_path,
            "historical_result_root": historical_root,
        }
        self.values = {
            "numeric": numeric,
            "parity": parity,
            "result": result,
            "runtime_protocol": {
                "schema_version": "vlsa_poisson_runtime_protocol.v2",
                "protocol_id": "runtime-v2",
                "cbf": {"alpha_gain_per_s": 5.0},
                "admissibility": {
                    "max_selected_geom_translation_drift_m": 1e-6,
                    "max_selected_geom_rotation_drift_rad": 1e-5,
                    "max_selected_geom_surface_drift_m": 1e-6,
                },
                "differential_audit": {"fixture": True},
                "claim_scope": {
                    "protected_robot_bodies": ["robot0_link5", "robot0_link6"]
                },
            },
        }

    def arguments(self):
        output = dict(self.paths)
        output.update(
            {
                "expected_code_commit": self.commit,
                "expected_case_id": self.case_id,
                "expected_producer_job_id": self.job_id,
                "expected_producer_host": self.host,
                "expected_producer_device": self.device,
                "expected_numeric_job_id": self.numeric_job_id,
                "expected_numeric_host": self.host,
                "expected_numeric_device": self.device,
                "expected_parity_job_id": self.parity_job_id,
                "expected_parity_host": self.host,
                "expected_parity_device": self.device,
            }
        )
        return output

    def patches(self):
        hashes = SimpleNamespace(
            protocol_sha256=self.runtime_semantic,
            parameter_block_sha256=self.runtime_parameters,
        )
        return (
            mock.patch(
                "main.poisson_fullbody.feasibility_protocol.load_feasibility_protocol",
                return_value=(self.values["runtime_protocol"], hashes),
            ),
            mock.patch(
                "main.poisson_fullbody.shadow_replay.load_historical_action_replay",
                return_value=self.replay,
            ),
            mock.patch(
                "scripts.run_poisson_shadow_identification._require_upstream_parity",
                return_value=self.values["parity"],
            ),
            mock.patch(
                "main.poisson_fullbody.shadow_identification.validate_shadow_replay_record"
            ),
        )


class ShadowIdentificationIndependentConsumerTest(unittest.TestCase):
    def test_strong_complete_pass_reconstructs_exact_phase_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CompleteFixture(directory)
            patches = fixture.patches()
            with patches[0], patches[1], patches[2], patches[3] as pure:
                report = validator.validate_shadow_identification_artifact(
                    **fixture.arguments()
                )
            self.assertEqual(report["status"], "validated")
            self.assertIs(report["stage_13_authorized"], True)
            self.assertIs(report["active_physics_authorized"], False)
            self.assertIs(report["active_physics_executed"], False)
            self.assertIs(report["full_active_canary_authorized"], False)
            self.assertEqual(report["exposure"]["action_count"], 237)
            self.assertEqual(report["exposure"]["callback_count"], 5925)
            self.assertEqual(
                report["exposure"]["measurement_first_index"], [0, 0, 0]
            )
            self.assertEqual(
                report["exposure"]["measurement_last_index"], [236, 4, 4]
            )
            timing = report["timing_reconstruction"]
            self.assertEqual(timing["warning_physical_boundary_W"], 4508)
            self.assertEqual(timing["contact_physical_boundary_C"], 4697)
            self.assertEqual(timing["lead_physics_substeps"], 189)
            self.assertEqual(timing["first_scheduled_filter_boundary_B"], 4510)
            self.assertEqual(timing["C_minus_B_physics_substeps"], 187)
            self.assertTrue(timing["B_strictly_before_C"])
            pure.assert_called_once()
            self.assertEqual(pure.call_args.kwargs["action_count"], 237)
            self.assertEqual(
                pure.call_args.kwargs["differential_audit_config"],
                {"fixture": True},
            )

    def test_arithmetic_geom_and_signal_tampering_are_rejected(self):
        shadow = {"poisson_identification": _warning_contact_shadow()}
        valid = validator._reconstruct_warning_contact(shadow)
        self.assertTrue(valid["authorization"])

        wrong_lead = copy.deepcopy(shadow)
        wrong_lead["poisson_identification"]["contact_prediction_assessment"][
            "lead_physics_substeps"
        ] = 188
        with self.assertRaisesRegex(
            validator.ShadowIdentificationArtifactError,
            "lead differs",
        ):
            validator._reconstruct_warning_contact(wrong_lead)

        wrong_geom = copy.deepcopy(shadow)
        wrong_geom["poisson_identification"]["contact_prediction_assessment"][
            "first_link56_contact"
        ]["robot_geom_id"] = 45
        with self.assertRaisesRegex(
            validator.ShadowIdentificationArtifactError,
            "geometry differ",
        ):
            validator._reconstruct_warning_contact(wrong_geom)

        invalid_signal = copy.deepcopy(shadow)
        warning = invalid_signal["poisson_identification"]["signals"][
            "first_observed_minimum_cbf_lhs_negative_on_contact_geom"
        ]
        warning["signal_kind"] = "any_invalid"
        invalid_signal["poisson_identification"][
            "contact_prediction_assessment"
        ]["primary_registered_warning"] = warning
        with self.assertRaisesRegex(
            validator.ShadowIdentificationArtifactError,
            "not the registered cbf_lhs_negative",
        ):
            validator._reconstruct_warning_contact(invalid_signal)

    def test_no_scheduled_update_is_a_typed_negative_not_authorization(self):
        identification = _warning_contact_shadow()
        warning = identification["signals"][
            "first_observed_minimum_cbf_lhs_negative_on_contact_geom"
        ]
        warning["observation_index"] = 10
        warning["high_level_index"] = 0
        warning["physics_substep_index"] = 10
        contact = identification["contact_prediction_assessment"][
            "first_link56_contact"
        ]
        contact["observation_index"] = 12
        assessment = identification["contact_prediction_assessment"]
        assessment["primary_registered_warning"] = warning
        assessment["lead_physics_substeps"] = 2
        assessment["lead_time_s"] = 0.004
        result = validator._reconstruct_warning_contact(
            {"poisson_identification": identification}
        )
        self.assertIs(result["authorization"], False)
        self.assertEqual(
            result["diagnostic_kind"],
            "no_scheduled_100hz_update_before_contact",
        )

    def test_false_acceptance_and_partial_pass_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CompleteFixture(directory)
            result_path = fixture.paths["identification_path"]
            false_result = copy.deepcopy(fixture.values["result"])
            false_result.pop("result_payload_sha256")
            false_result["acceptance"][
                "all_5925_callbacks_observed"
            ] = False
            false_result = _hashed(false_result)
            _write(result_path, false_result)
            patches = fixture.patches()
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(
                    validator.ShadowIdentificationArtifactError,
                    "false or non-boolean acceptance",
                ):
                    validator.validate_shadow_identification_artifact(
                        **fixture.arguments()
                    )

            partial = {
                key: value
                for key, value in false_result.items()
                if key != "result_payload_sha256"
            }
            partial["status"] = "running"
            partial = _hashed(partial)
            _write(result_path, partial)
            with self.assertRaisesRegex(
                validator.ShadowIdentificationArtifactError,
                "neither a complete pass nor a valid failure",
            ):
                validator.validate_shadow_identification_artifact(
                    **fixture.arguments()
                )

            unknown = {
                key: value
                for key, value in fixture.values["result"].items()
                if key != "result_payload_sha256"
            }
            unknown["post_hoc_claim"] = True
            _write(result_path, _hashed(unknown))
            with self.assertRaisesRegex(
                validator.ShadowIdentificationArtifactError,
                "neither a complete pass nor a valid failure",
            ):
                validator.validate_shadow_identification_artifact(
                    **fixture.arguments()
                )

    def test_complete_pass_requires_exact_external_h100_device(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CompleteFixture(directory)
            arguments = fixture.arguments()
            arguments["expected_producer_device"] = "NVIDIA A100"
            patches = fixture.patches()
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(
                    validator.ShadowIdentificationArtifactError,
                    "H100 device differs",
                ):
                    validator.validate_shadow_identification_artifact(
                        **arguments
                    )

    def test_collusively_rehashed_semantic_and_unknown_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CompleteFixture(directory)
            result_path = fixture.paths["identification_path"]
            mutations = (
                (
                    "claim_limit",
                    lambda payload: payload.__setitem__(
                        "claim_limit", "full active safety efficacy established"
                    ),
                ),
                (
                    "timing",
                    lambda payload: payload.__setitem__("timing", "complete"),
                ),
                (
                    "provenance_extra",
                    lambda payload: payload["provenance"].__setitem__(
                        "post_hoc_claim", True
                    ),
                ),
                (
                    "producer_job_name",
                    lambda payload: payload["provenance"].__setitem__(
                        "slurm_job_name", "unregistered-identification"
                    ),
                ),
                (
                    "producer_python_executable",
                    lambda payload: payload["provenance"].__setitem__(
                        "python_executable", "/tmp/unregistered-python"
                    ),
                ),
                (
                    "producer_python_version",
                    lambda payload: payload["provenance"].__setitem__(
                        "python_version", "9.9.9"
                    ),
                ),
                (
                    "shadow_extra",
                    lambda payload: payload["shadow_replay"].__setitem__(
                        "post_hoc_claim",
                        {"full_active_canary_authorized": True},
                    ),
                ),
            )
            for label, mutate in mutations:
                with self.subTest(label=label):
                    payload = {
                        key: value
                        for key, value in fixture.values["result"].items()
                        if key != "result_payload_sha256"
                    }
                    payload = copy.deepcopy(payload)
                    mutate(payload)
                    _write(result_path, _hashed(payload))
                    patches = fixture.patches()
                    with patches[0], patches[1], patches[2], patches[3]:
                        with self.assertRaises(
                            validator.ShadowIdentificationArtifactError
                        ):
                            validator.validate_shadow_identification_artifact(
                                **fixture.arguments()
                            )

    def test_collusively_rehashed_numeric_and_parity_forgery_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CompleteFixture(directory)

            forged_numeric = {
                key: value
                for key, value in fixture.values["numeric"].items()
                if key != "result_payload_sha256"
            }
            forged_numeric = copy.deepcopy(forged_numeric)
            forged_numeric["tests"].update(
                {
                    "tests_run": 0,
                    "failure_count": 999,
                    "error_count": 999,
                    "failures": ["forged"],
                    "errors": ["forged"],
                }
            )
            _write(fixture.paths["numeric_path"], _hashed(forged_numeric))
            patches = fixture.patches()
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(
                    validator.ShadowIdentificationArtifactError,
                    "numeric prerequisite test execution evidence is incomplete",
                ):
                    validator.validate_shadow_identification_artifact(
                        **fixture.arguments()
                    )

            _write(fixture.paths["numeric_path"], fixture.values["numeric"])
            forged_parity = {
                key: value
                for key, value in fixture.values["parity"].items()
                if key != "result_payload_sha256"
            }
            forged_parity = copy.deepcopy(forged_parity)
            forged_parity["evidence_tier"] = "active safety efficacy"
            forged_parity["full_active_canary_authorized"] = True
            forged_parity = _hashed(forged_parity)
            _write(fixture.paths["parity_path"], forged_parity)
            forged_result = {
                key: value
                for key, value in fixture.values["result"].items()
                if key != "result_payload_sha256"
            }
            forged_result = copy.deepcopy(forged_result)
            forged_result["provenance"][
                "upstream_parity_payload_sha256"
            ] = forged_parity["result_payload_sha256"]
            _write(
                fixture.paths["identification_path"],
                _hashed(forged_result),
            )
            patches = fixture.patches()
            with patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaisesRegex(
                    validator.ShadowIdentificationArtifactError,
                    "exact-parity prerequisite has wrong envelope",
                ):
                    validator.validate_shadow_identification_artifact(
                        **fixture.arguments()
                    )

    def test_file_authority_rejects_traversal_and_symlink_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                validator._require_real_file(target, "target"), target
            )
            with self.assertRaisesRegex(
                validator.ShadowIdentificationArtifactError,
                "parent traversal",
            ):
                validator._require_real_file(
                    root / "child" / ".." / "target.json", "target"
                )
            link = root / "linked"
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(
                validator.ShadowIdentificationArtifactError,
                "symbolic links",
            ):
                validator._require_real_file(link / "target.json", "target")

    def test_well_formed_early_failure_is_validated_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory).resolve() / "failed.json"
            failure = _hashed(
                {
                    "schema_version": validator.EXPECTED_SCHEMA,
                    "status": "failed",
                    "scientific_result": False,
                    "evidence_tier": validator.IDENTIFICATION_EVIDENCE_TIER,
                    "claim_limit": validator.IDENTIFICATION_CLAIM_LIMIT,
                    "case_id": validator.EXPECTED_CASE_ID,
                    "phase": "initial_authority_validation",
                    "timing": {
                        "started_unix": 10.0,
                        "finished_unix": 12.0,
                        "elapsed_seconds": 2.0,
                    },
                    "failure": {
                        "type": "ShadowIdentificationRunnerError",
                        "message": "fixture failure",
                        "traceback": "fixture traceback",
                    },
                }
            )
            _write(result_path, failure)
            arguments = {
                "identification_path": result_path,
                "numeric_path": Path(directory) / "absent-numeric.json",
                "parity_path": Path(directory) / "absent-parity.json",
                "manifest_path": Path(directory) / "absent-manifest.jsonl",
                "selection_path": Path(directory) / "absent-selection.json",
                "runtime_path": Path(directory) / "absent-runtime.json",
                "historical_result_root": Path(directory),
                "expected_code_commit": "a" * 40,
                "expected_case_id": validator.EXPECTED_CASE_ID,
                "expected_producer_job_id": "33712",
                "expected_producer_host": "worker",
                "expected_producer_device": "NVIDIA H100 80GB HBM3",
                "expected_numeric_job_id": "33708",
                "expected_numeric_host": "numeric-worker",
                "expected_numeric_device": "NVIDIA H100 80GB HBM3",
                "expected_parity_job_id": "33709",
                "expected_parity_host": "parity-worker",
                "expected_parity_device": "NVIDIA H100 80GB HBM3",
            }
            report = validator.validate_shadow_identification_artifact(
                **arguments
            )
            self.assertEqual(report["status"], "validated_negative")
            self.assertIs(report["stage_13_authorized"], False)
            self.assertEqual(
                report["diagnostic_kind"],
                "producer_failed_before_complete_shadow",
            )
            self.assertIs(report["partial_shadow_interpreted"], False)

            malformed = dict(failure)
            malformed.pop("result_payload_sha256")
            malformed["failure"] = dict(malformed["failure"])
            malformed["failure"].pop("traceback")
            _write(result_path, _hashed(malformed))
            with self.assertRaisesRegex(
                validator.ShadowIdentificationArtifactError,
                "malformed failure record",
            ):
                validator.validate_shadow_identification_artifact(**arguments)

            for missing in ("evidence_tier", "claim_limit"):
                with self.subTest(missing=missing):
                    incomplete = {
                        key: value
                        for key, value in failure.items()
                        if key not in {"result_payload_sha256", missing}
                    }
                    _write(result_path, _hashed(incomplete))
                    with self.assertRaises(
                        validator.ShadowIdentificationArtifactError
                    ):
                        validator.validate_shadow_identification_artifact(
                            **arguments
                        )

    def test_retained_differential_failure_is_deeply_inspected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _CompleteFixture(directory)
            result_path = fixture.paths["identification_path"]
            provenance = copy.deepcopy(
                fixture.values["result"]["provenance"]
            )
            authority_keys = (
                "manifest_sha256",
                "manifest_row_sha256",
                "selection_config_sha256",
                "runtime_protocol_raw_sha256",
                "runtime_protocol_semantic_sha256",
                "runtime_parameter_block_sha256",
                "historical_result_file_sha256",
                "historical_result_payload_sha256",
                "upstream_parity_payload_sha256",
                "host",
                "slurm_job_id",
                "slurm_job_name",
            )
            binding = {"source": provenance["source"]}
            binding.update({key: provenance[key] for key in authority_keys})
            ordered_samples = [
                {
                    "sample_id": index,
                    "body_id": 5 if index < 800 else 6,
                    "body_name": (
                        "robot0_link5" if index < 800 else "robot0_link6"
                    ),
                    "geom_id": 44 if index < 800 else 45,
                    "geom_name": (
                        "robot0_link5_collision"
                        if index < 800
                        else "robot0_link6_collision"
                    ),
                    "point_body_local_m": [0.0, 0.0, 0.0],
                    "source": "collision_geom_surface",
                }
                for index in range(1531)
            ]
            audit = {
                "integration_state": {"source_initial_sha256": "a" * 64},
                "ordered_samples": ordered_samples,
                "arm_dof_indices": list(range(7)),
                "differential_audit_config": {"fixture": True},
            }
            receipt = {"passed": False, "counts": {"sample_count": 1531}}
            evidence = {
                "schema_version": validator.DIFFERENTIAL_FAILURE_SCHEMA,
                "phase": validator.DIFFERENTIAL_FAILURE_PHASE,
                "failure_kind": "registered_protected_sample_differential_audit_failed",
                "scientific_result": False,
                "active_physics_authorized": False,
                "settled_link56_differential_audit": audit,
                "settled_link56_differential_audit_validation": receipt,
                "interpretation": "diagnostic only",
                "authority_binding": binding,
            }
            failure = _hashed(
                {
                    "schema_version": validator.EXPECTED_SCHEMA,
                    "status": "failed",
                    "scientific_result": False,
                    "evidence_tier": validator.IDENTIFICATION_EVIDENCE_TIER,
                    "claim_limit": validator.IDENTIFICATION_CLAIM_LIMIT,
                    "case_id": validator.EXPECTED_CASE_ID,
                    "phase": validator.DIFFERENTIAL_FAILURE_PHASE,
                    "timing": {
                        "started_unix": 10.0,
                        "finished_unix": 12.0,
                        "elapsed_seconds": 2.0,
                    },
                    "provenance": provenance,
                    "historical": fixture.replay.provenance(),
                    "failure_evidence": evidence,
                    "failure": {
                        "type": validator.DIFFERENTIAL_FAILURE_TYPE,
                        "message": validator.DIFFERENTIAL_FAILURE_MESSAGE,
                        "traceback": "fixture traceback",
                    },
                }
            )
            _write(result_path, failure)
            patches = fixture.patches()
            with patches[0], patches[1], patches[2], mock.patch(
                "main.poisson_fullbody.jacobians.inspect_protected_sample_differential_audit",
                return_value=receipt,
            ) as inspect:
                report = validator.validate_shadow_identification_artifact(
                    **fixture.arguments()
                )
            inspect.assert_called_once()
            self.assertEqual(
                report["diagnostic_kind"],
                "producer_reported_differential_audit_failure_with_"
                "unbound_sample_state_authority",
            )
            self.assertIs(report["stage_13_authorized"], False)
            self.assertIs(report["external_sample_state_authority_bound"], False)
            self.assertIs(report["mechanism_diagnosis_verified"], False)
            self.assertEqual(
                report["retained_differential_audit_validation"], receipt
            )

            for label, mutate in (
                (
                    "generic_failure_type",
                    lambda payload: payload["failure"].update(
                        {
                            "type": "ShadowIdentificationRunnerError",
                            "message": (
                                "read-only shadow callback mutated complete "
                                "MuJoCo integration state"
                            ),
                        }
                    ),
                ),
                (
                    "wrong_typed_message",
                    lambda payload: payload["failure"].__setitem__(
                        "message", "source or clone state was not preserved"
                    ),
                ),
                (
                    "impossible_partial_shadow",
                    lambda payload: payload.__setitem__("shadow_replay", {}),
                ),
                (
                    "impossible_acceptance",
                    lambda payload: payload.__setitem__(
                        "acceptance",
                        {
                            field: False
                            for field in validator.IDENTIFICATION_ACCEPTANCE_FIELDS
                        },
                    ),
                ),
            ):
                with self.subTest(causal_binding=label):
                    tampered = {
                        key: value
                        for key, value in failure.items()
                        if key != "result_payload_sha256"
                    }
                    tampered = copy.deepcopy(tampered)
                    mutate(tampered)
                    _write(result_path, _hashed(tampered))
                    tamper_patches = fixture.patches()
                    with tamper_patches[0], tamper_patches[1], tamper_patches[
                        2
                    ], mock.patch(
                        "main.poisson_fullbody.jacobians."
                        "inspect_protected_sample_differential_audit",
                        return_value=receipt,
                    ):
                        with self.assertRaises(
                            validator.ShadowIdentificationArtifactError
                        ):
                            validator.validate_shadow_identification_artifact(
                                **fixture.arguments()
                            )

            # A collusively rehashed artifact still cannot replace the frozen
            # runtime differential config or an externally bound input hash.
            for field, mutate in (
                (
                    "config",
                    lambda payload: payload["failure_evidence"][
                        "settled_link56_differential_audit"
                    ].__setitem__("differential_audit_config", {"other": True}),
                ),
                (
                    "sample",
                    lambda payload: payload["failure_evidence"][
                        "settled_link56_differential_audit"
                    ]["ordered_samples"][0].__setitem__(
                        "body_name", "robot0_link4"
                    ),
                ),
                (
                    "binding",
                    lambda payload: payload["failure_evidence"][
                        "authority_binding"
                    ].__setitem__("manifest_sha256", "f" * 64),
                ),
            ):
                with self.subTest(field=field):
                    tampered = {
                        key: value
                        for key, value in failure.items()
                        if key != "result_payload_sha256"
                    }
                    tampered = copy.deepcopy(tampered)
                    mutate(tampered)
                    _write(result_path, _hashed(tampered))
                    tamper_patches = fixture.patches()
                    with tamper_patches[0], tamper_patches[1], tamper_patches[2], mock.patch(
                        "main.poisson_fullbody.jacobians.inspect_protected_sample_differential_audit",
                        return_value=receipt,
                    ):
                        with self.assertRaises(
                            validator.ShadowIdentificationArtifactError
                        ):
                            validator.validate_shadow_identification_artifact(
                                **fixture.arguments()
                            )


class ShadowIdentificationConsumerWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper = (
            ROOT / "slurm/poisson_shadow_identification_validate.sbatch"
        ).read_text(encoding="utf-8")
        cls.consumer = (
            ROOT / "scripts/validate_poisson_shadow_identification_artifact.py"
        ).read_text(encoding="utf-8")

    def test_wrapper_is_bounded_cpu_only_clean_and_read_only(self):
        self.assertIn("#SBATCH --cpus-per-task=2", self.wrapper)
        self.assertIn("#SBATCH --mem=8G", self.wrapper)
        self.assertNotIn("#SBATCH --gres", self.wrapper)
        self.assertNotIn("#SBATCH --gpus", self.wrapper)
        self.assertIn("SLURM_JOB_ID", self.wrapper)
        self.assertIn(
            '"${SLURM_JOB_ID}" == "${EXPECTED_PRODUCER_JOB_ID}"',
            self.wrapper,
        )
        self.assertIn('git status --short', self.wrapper)
        self.assertIn('git rev-parse HEAD', self.wrapper)
        self.assertIn('realpath -e --', self.wrapper)
        self.assertIn('export CUDA_VISIBLE_DEVICES=""', self.wrapper)
        self.assertIn('export PYTHONDONTWRITEBYTECODE=1', self.wrapper)
        self.assertIn("REGISTERED_EVALUATION_PYTHON", self.wrapper)
        self.assertIn("EXPECTED_NUMERIC_JOB_ID", self.wrapper)
        self.assertIn("EXPECTED_PARITY_JOB_ID", self.wrapper)
        self.assertNotIn("mkdir ", self.wrapper)
        self.assertNotIn("--output \"${", self.wrapper)
        self.assertIn(
            "validate_poisson_shadow_identification_artifact.py",
            self.wrapper,
        )

    def test_consumer_has_no_artifact_output_option_or_publication_call(self):
        self.assertNotIn('add_argument("--output"', self.consumer)
        self.assertNotIn("publish_hashed_json", self.consumer)
        self.assertNotIn("atomic_json", self.consumer)
        self.assertIn("validate_shadow_replay_record(", self.consumer)
        self.assertIn("inspect_protected_sample_differential_audit(", self.consumer)


if __name__ == "__main__":
    unittest.main()
