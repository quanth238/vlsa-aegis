from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ModuleNotFoundError:  # dependency-free local gate
    np = None

try:
    import jsonschema
except ModuleNotFoundError:  # optional local schema validator
    jsonschema = None


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "main", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crfs_harness.artifacts import atomic_write_json, content_hash, load_json  # noqa: E402

if np is not None:
    import crfs_oracle.r04_baseline_reference as baseline_reference  # noqa: E402
    from crfs_oracle.r04_labels import _observation_identity, _trace_record  # noqa: E402
    from crfs_oracle.r04_resume import (  # noqa: E402
        APPARATUS_SCOPE,
        EXPECTED_CONFIG_FILE_SHA256,
        EXPECTED_MANIFEST_SHA256,
        EXPECTED_R04A_RAW_ARTIFACT_SHA256,
        EXPECTED_R04A_VALIDATION_SUMMARY_SHA256,
        NONZERO_EDIT_ROLE,
        R04ResumeConfig,
        TRACE_STEPS,
        _array_record,
        _arrays_byte_exact,
        _nonzero_edit,
        r04_resume_config_from_mapping,
        run_r04_resume_case,
        valid_r04_resume_completion,
        validate_r04_resume_result,
    )
    from crfs_oracle.runner import OracleConfig, _array_hash  # noqa: E402
    from run_r04_resume_parity import _validate_artifact  # noqa: E402


CHECKPOINT_SHA = "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed"
SHA = "a" * 64
if np is not None:
    REAL_R04A_GOLDEN_OBSERVATION_SHA256 = (
        baseline_reference.R04A_GOLDEN_OBSERVATION_SHA256
    )
    REAL_R04A_GOLDEN_FINAL_PHYSICAL_SHA256 = (
        baseline_reference.R04A_GOLDEN_FINAL_PHYSICAL_SHA256
    )


def _oracle(output_root: str) -> "OracleConfig":
    return OracleConfig(
        host="127.0.0.1",
        port=8000,
        resize_size=224,
        settle_steps=20,
        executed_prefix=5,
        action_horizon=10,
        action_dim=32,
        sampler_steps=10,
        intervention_step=5,
        safety_margin_m=0.005,
        distance_limit_m=1.0,
        eef_radius_m=0.06,
        measurement_repeats=2,
        stop_after_measurement=False,
        response_matrix_m_per_action=None,
        optimizer_max_iterations=1,
        checkpoint_id="converted-pi05-libero",
        checkpoint_sha256=CHECKPOINT_SHA,
        output_root=output_root,
        run_id="r04b-test",
    )


def _sources() -> dict:
    return {
        "r00_summary": {
            "path": "/frozen/r00-summary.json",
            "sha256": "90fe09e075b30e680e5cfbd81c802afa03c94b58ba7f0e95e66080c61af2096f",
            "status": "passed",
            "p_min_m": 0.029897349105658888,
        },
        "r03_summary": {
            "path": "/frozen/r03-summary.json",
            "sha256": "dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e",
            "status": "passed",
            "learned_probe_authorized": True,
        },
        "sampler_parity": {
            "path": "/frozen/parity.json",
            "sha256": "26083b0a71ec41cf67e0978b815a77bfe71e4b4a4a08a8ddcdaece608de9818a",
            "status": "passed",
            "checkpoint_sha256": CHECKPOINT_SHA,
            "normalization_asset_sha256": SHA,
        },
        "r04a_validation_summary": {
            "path": "/frozen/r04a-validation.json",
            "sha256": EXPECTED_R04A_VALIDATION_SUMMARY_SHA256,
            "status": "passed_apparatus_only",
            "raw_artifact_sha256": EXPECTED_R04A_RAW_ARTIFACT_SHA256,
            "independent_validation_error_count": 0,
        },
        "r04a_raw_artifact": {
            "path": "/frozen/r04-label-contract.json",
            "sha256": EXPECTED_R04A_RAW_ARTIFACT_SHA256,
            "status": "validated_apparatus_only",
            "config_hash": SHA,
            "git_commit": "b" * 40,
            "golden_baseline": baseline_reference.expected_r04a_golden_contract(),
        },
    }


def _config(output_root: str) -> "R04ResumeConfig":
    return R04ResumeConfig(
        oracle=_oracle(output_root),
        enabled=True,
        apparatus_scope=APPARATUS_SCOPE,
        reuse_role="apparatus_only_never_train_calibrate_validate_test_or_claim",
        smoke_case_index=0,
        trace_steps=(1, 2, 3, 4, 5),
        source_calls_per_step=2,
        zero_resume_calls_per_step=2,
        nonzero_resume_calls_per_step=2,
        nonzero_edit_index=(0, 0),
        nonzero_edit_value=0.03125,
        declared_manifest_sha256=EXPECTED_MANIFEST_SHA256,
        config_file_sha256=EXPECTED_CONFIG_FILE_SHA256,
        source_evidence=_sources(),
    )


def _case() -> dict:
    return {
        "schema_version": "1.0",
        "case_id": "crfs-93365b8b851365f2",
        "task_suite": "safelibero_spatial",
        "safety_level": "II",
        "task_index": 0,
        "episode_index": 2,
        "environment_seed": 1635240984,
        "policy_seed": 1250848483,
        "random_control_seed": 98711256,
        "group_id": "safelibero_spatial:II:0:2",
    }


def _policy_input() -> dict:
    return {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((224, 224, 3), dtype=np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
        "prompt": "reach for the bowl",
    }


def _step_time(step: int) -> np.ndarray:
    value = np.asarray(1.0, dtype=np.float32)
    delta = np.asarray(-0.1, dtype=np.float32)
    for _ in range(step):
        value = np.asarray(value + delta, dtype=np.float32)
    return value.reshape(())


class FakeClient:
    def __init__(self) -> None:
        self.requests = []
        self.source_normalized = np.arange(320, dtype=np.float32).reshape(10, 32) / 1000.0
        self.source_physical = np.arange(70, dtype=np.float32).reshape(10, 7) / 100.0
        self.source_normalized[0, 0] = np.float32(-0.0)
        self.source_physical[0, 0] = np.float32(-0.0)

    def infer(self, request):
        controls = request["__crfs__"]
        self.requests.append(copy.deepcopy(controls))
        if controls.get("return_trace") is False:
            assert set(controls) == {"noise", "intervention_mode", "return_trace"}
            assert controls["intervention_mode"] == "none"
            assert controls["noise"].shape == (10, 32)
            assert controls["noise"].dtype == np.float32
            return {
                "actions": np.array(self.source_physical, copy=True),
                "policy_timing": {"infer_ms": 0.25},
            }
        self._validate_common_controls(controls)
        step = int(controls["intervention_step"])
        if controls["intervention_mode"] == "none":
            return self._source_reply(step)
        return self._resume_reply(step, controls)

    @staticmethod
    def _validate_common_controls(controls) -> None:
        assert controls["noise"].shape == (10, 32)
        assert controls["noise"].dtype == np.float32
        assert controls["return_trace"] is True
        assert controls["return_normalized_final"] is True
        assert "correction" not in controls

    def _source_reply(self, step: int) -> dict:
        time = _step_time(step)
        latent = np.full((10, 32), step / 16.0, dtype=np.float32)
        latent[0, 0] = np.float32(-0.0)
        velocity = np.ascontiguousarray(latent + np.float32(0.125))
        predicted = np.ascontiguousarray(latent - time * velocity)
        predicted_physical = np.zeros((10, 7), dtype=np.float32)
        predicted_physical[0, 0] = np.float32(-0.0)
        return {
            "actions": np.array(self.source_physical, copy=True),
            "crfs_trace": {
                "step_index": np.asarray(step, dtype=np.int64),
                "time": time,
                "x_t": latent,
                "v_base": velocity,
                "predicted_clean": predicted,
                "predicted_clean_physical": predicted_physical,
                "final_normalized": np.array(self.source_normalized, copy=True),
            },
            "policy_timing": {"infer_ms": float(step)},
        }

    def _resume_reply(self, step: int, controls) -> dict:
        assert controls["latent_edit_space"] == "model"
        assert controls["resume_time"].shape == ()
        assert controls["resume_time"].dtype == np.float32
        assert controls["resume_latent"].shape == (10, 32)
        assert controls["resume_latent"].dtype == np.float32
        assert controls["latent_edit"].shape == (10, 32)
        assert controls["latent_edit"].dtype == np.float32
        pre = np.ascontiguousarray(controls["resume_latent"])
        edit = np.ascontiguousarray(controls["latent_edit"])
        edited = np.ascontiguousarray(pre + edit)
        post = np.ascontiguousarray(np.where(edit == 0, pre, edited))
        time = np.asarray(controls["resume_time"], dtype=np.float32).reshape(())
        zero_edit = not bool(np.count_nonzero(edit))
        velocity_offset = np.float32(0.125 if zero_edit else 0.25)
        velocity = np.ascontiguousarray(post + velocity_offset)
        predicted = np.ascontiguousarray(post - time * velocity)
        if zero_edit:
            normalized = np.array(self.source_normalized, copy=True)
            physical = np.array(self.source_physical, copy=True)
            predicted_physical = np.zeros((10, 7), dtype=np.float32)
            predicted_physical[0, 0] = np.float32(-0.0)
        else:
            normalized = np.ascontiguousarray(self.source_normalized + edit)
            physical = np.array(self.source_physical, copy=True)
            physical[0, 0] += np.asarray(edit[0, 0], dtype=physical.dtype)
            predicted_physical = np.zeros((10, 7), dtype=np.float32)
        return {
            "actions": physical,
            "crfs_trace": {
                "step_index": np.asarray(step, dtype=np.int64),
                "time": time,
                "x_t_pre_edit": pre,
                "latent_edit": edit,
                "x_t_post_edit": post,
                "v_post_edit": velocity,
                "predicted_clean_post_edit": predicted,
                "predicted_clean_post_edit_physical": predicted_physical,
                "final_normalized": normalized,
            },
            "policy_timing": {"infer_ms": float(step) + 0.5},
        }


class FakeEnvironment:
    def __init__(self) -> None:
        self.obstacle_name = "active_obstacle"
        self.prompt = "reach for the bowl"
        self.configured = []
        self.closed = False

    def configure_case(self, case) -> None:
        self.configured.append(dict(case))

    def reset_and_settle(self):
        return {"unused": True}

    def close(self) -> None:
        self.closed = True


class R04ResumeDependencyFreeStructuralTest(unittest.TestCase):
    def test_config_is_one_case_setup_only_no_efficacy_or_learning_apparatus(self) -> None:
        value = json.loads(
            (ROOT / "configs/experiments/r04_resume_parity.json").read_text(
                encoding="utf-8"
            )
        )
        settings = value["r04b"]
        self.assertTrue(value["ready_to_run"])
        self.assertEqual(settings["source_trace_steps"], [1, 2, 3, 4, 5])
        self.assertEqual(settings["compiled_default_calls"], 2)
        self.assertEqual(settings["total_policy_calls"], 32)
        self.assertEqual(
            settings["compiled_default_call_placement"],
            "one_before_and_one_after_eager_resume_sequence",
        )
        self.assertEqual(
            settings["compiled_default_physical_limits"]["physical_xyz5_max"],
            0.01,
        )
        self.assertEqual(settings["latent_edit_space"], "model")
        self.assertEqual(
            settings["simulator_setup"],
            "one_reset_plus_20_dummy_settle_control_steps",
        )
        self.assertEqual(settings["policy_generated_action_steps_executed"], 0)
        self.assertEqual(settings["efficacy_rollouts_executed"], 0)
        self.assertFalse(settings["simulator_efficacy_evaluated"])
        self.assertFalse(settings["training"])
        self.assertFalse(settings["guidance"])
        self.assertFalse(settings["learned_probe"])
        self.assertEqual(
            settings["exact_equality_contract"],
            "dtype_shape_and_c_contiguous_array_bytes_sha256",
        )
        self.assertEqual(settings["nonzero_edit"]["role"], NONZERO_EDIT_ROLE if np is not None else "implementation_sentinel_only_never_support_or_dose_evidence")

    def test_core_has_fail_closed_validator_before_atomic_write(self) -> None:
        source = (ROOT / "main/crfs_oracle/r04_resume.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def validate_r04_resume_result(", source)
        self.assertIn("def valid_r04_resume_completion(", source)
        self.assertIn('"latent_edit_space": "model"', source)
        self.assertIn('"return_normalized_final": True', source)
        self.assertNotIn("environment.rollout(", source)
        validation = source.index("errors = validate_r04_resume_result(result)")
        atomic_write = source.index("atomic_write_json(output, result)", validation)
        self.assertLess(validation, atomic_write)
        self.assertNotIn("import torch", source.lower())

    def test_schema_and_cpu_validation_cli_freeze_surface(self) -> None:
        schema = json.loads(
            (ROOT / "schemas/r04-resume-parity.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(schema["properties"]["gate"]["const"], "R04B")
        self.assertIn("r03_summary", schema["properties"]["source_evidence"]["required"])
        self.assertEqual(
            schema["properties"]["execution_boundaries"]["properties"][
                "nonzero_edit_role"
            ]["const"],
            "implementation_sentinel_only_never_support_or_dose_evidence",
        )
        baseline_schema = schema["$defs"]["baselineReference"]
        self.assertFalse(baseline_schema["additionalProperties"])
        self.assertFalse(schema["$defs"]["errorMetric"]["additionalProperties"])
        self.assertEqual(
            schema["properties"]["pairing"]["properties"][
                "total_policy_call_count"
            ]["const"],
            32,
        )
        cli = (ROOT / "main/run_r04_resume_parity.py").read_text(encoding="utf-8")
        self.assertIn("--validate-artifact", cli)
        self.assertIn("validation_error_count", cli)
        self.assertIn("expected-source-slurm-array-task-id", cli)


@unittest.skipIf(np is None, "R04B runtime checks need numpy")
class R04ResumeConfigTest(unittest.TestCase):
    def test_checked_in_config_loads_only_with_bound_predecessors(self) -> None:
        value = json.loads(
            (ROOT / "configs/experiments/r04_resume_parity.json").read_text(
                encoding="utf-8"
            )
        )
        with mock.patch(
            "crfs_oracle.r04_resume._validated_sources", return_value=_sources()
        ):
            config = r04_resume_config_from_mapping(
                value,
                _oracle("/tmp/r04b"),
                repo_root=ROOT,
                config_file_sha256=EXPECTED_CONFIG_FILE_SHA256,
            )
        self.assertEqual(config.trace_steps, TRACE_STEPS)
        self.assertEqual(config.nonzero_edit_value, 0.03125)

        unsafe = copy.deepcopy(value)
        unsafe["r04b"]["nonzero_edit"]["role"] = "dose_evidence"
        with mock.patch(
            "crfs_oracle.r04_resume._validated_sources", return_value=_sources()
        ):
            with self.assertRaisesRegex(ValueError, "edit construction"):
                r04_resume_config_from_mapping(
                    unsafe,
                    _oracle("/tmp/r04b"),
                    repo_root=ROOT,
                    config_file_sha256=EXPECTED_CONFIG_FILE_SHA256,
                )


@unittest.skipIf(np is None, "R04B runtime checks need numpy")
class R04ResumeRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.client = FakeClient()
        self.environment = FakeEnvironment()
        observation_sha = _observation_identity(_policy_input())["sha256"]
        noise = np.random.default_rng(_case()["policy_seed"]).normal(
            size=(10, 32)
        ).astype(np.float32)
        trace_hashes = tuple(
            _trace_record(self.client._source_reply(step)["crfs_trace"])["sha256"]
            for step in TRACE_STEPS
        )
        self.golden_patchers = (
            mock.patch.object(
                baseline_reference,
                "R04A_GOLDEN_OBSERVATION_SHA256",
                observation_sha,
            ),
            mock.patch.object(
                baseline_reference,
                "R04A_GOLDEN_NOISE_SHA256",
                _array_hash(noise),
            ),
            mock.patch.object(
                baseline_reference,
                "R04A_GOLDEN_FINAL_PHYSICAL_SHA256",
                _array_hash(self.client.source_physical),
            ),
            mock.patch.object(
                baseline_reference,
                "R04A_GOLDEN_TRACE_SHA256_BY_STEP",
                trace_hashes,
            ),
        )
        for patcher in self.golden_patchers:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self.golden_patchers):
            patcher.stop()
        self.temporary.cleanup()

    def _run(self):
        allocation = {
            "SLURM_JOB_ID": "321",
            "SLURM_ARRAY_JOB_ID": "321",
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_JOB_PARTITION": "worker-mig",
            "CUDA_VISIBLE_DEVICES": "MIG-0",
            "EXPECTED_GIT_COMMIT": "b" * 40,
        }
        with mock.patch.dict(os.environ, allocation, clear=False), mock.patch(
            "crfs_oracle.r04_resume._git_state", return_value=("b" * 40, False)
        ), mock.patch(
            "crfs_oracle.r04_resume.policy_observation", return_value=_policy_input()
        ):
            return run_r04_resume_case(
                _case(),
                _config(str(self.root)),
                repo_root=ROOT,
                input_manifest_sha256=EXPECTED_MANIFEST_SHA256,
                client=self.client,
                environment=self.environment,
            )

    def test_runner_executes_32_call_baseline_resume_contract_and_validates_artifact(self) -> None:
        output, status = self._run()
        self.assertEqual(status, "completed")
        self.assertEqual(len(self.client.requests), 32)
        self.assertEqual(
            set(self.client.requests[0]),
            {"noise", "intervention_mode", "return_trace"},
        )
        self.assertFalse(self.client.requests[0]["return_trace"])
        self.assertFalse(self.client.requests[-1]["return_trace"])
        self.assertEqual(
            [request["intervention_mode"] for request in self.client.requests[1:7]],
            ["none", "none", "latent_resume_edit", "latent_resume_edit", "latent_resume_edit", "latent_resume_edit"],
        )
        first_noise = self.client.requests[0]["noise"]
        self.assertTrue(
            all(
                _arrays_byte_exact(first_noise, request["noise"])
                for request in self.client.requests
            )
        )
        resume_requests = [
            item
            for item in self.client.requests
            if item["intervention_mode"] == "latent_resume_edit"
        ]
        self.assertTrue(all(item["resume_time"].shape == () for item in resume_requests))
        self.assertTrue(all(item["latent_edit_space"] == "model" for item in resume_requests))

        result = load_json(output)
        self.assertEqual(validate_r04_resume_result(result), [])
        baseline = result["pairing"]["baseline_reference"]
        self.assertTrue(baseline["current_eager_golden_binding"]["all_exact"])
        self.assertTrue(
            baseline["compiled_default"]
            ["exact_dtype_shape_and_canonical_bytes_across_sequence"]
        )
        self.assertTrue(
            baseline["compiled_vs_current_eager"]
            ["all_registered_physical_limits_pass"]
        )
        self.assertIn(
            "not_exposed_by_ordinary_compiled_physical_policy_reply",
            baseline["compiled_vs_current_eager"]
            ["normalized_model_comparison_status"],
        )
        self.assertEqual(
            result["source_evidence"]["r04a_validation_summary"]["sha256"],
            EXPECTED_R04A_VALIDATION_SUMMARY_SHA256,
        )
        self.assertEqual(
            result["execution_boundaries"]["nonzero_edit_role"], NONZERO_EDIT_ROLE
        )
        self.assertEqual(
            result["execution_boundaries"]["simulator_setup"],
            "one_reset_plus_20_dummy_settle_control_steps",
        )
        self.assertEqual(
            result["execution_boundaries"]["policy_generated_action_steps_executed"],
            0,
        )
        self.assertEqual(result["execution_boundaries"]["efficacy_rollouts_executed"], 0)
        self.assertFalse(
            result["execution_boundaries"]["simulator_efficacy_evaluated"]
        )
        for record in result["pairing"]["step_records"]:
            self.assertTrue(all(record["zero_resume"]["source_parity"].values()))
        fifth_time = result["pairing"]["step_records"][4]["captured_time"]
        self.assertEqual(fifth_time["dtype"], "float32")
        self.assertNotEqual(fifth_time["values"][0], 0.5)
        self.assertTrue(
            valid_r04_resume_completion(
                output,
                case_id=_case()["case_id"],
                run_id="r04b-test",
                config_hash=result["config_hash"],
                input_manifest_sha256=EXPECTED_MANIFEST_SHA256,
                config_file_sha256=EXPECTED_CONFIG_FILE_SHA256,
                checkpoint_sha256=CHECKPOINT_SHA,
            )
        )

    @unittest.skipIf(jsonschema is None, "jsonschema is not installed")
    def test_generated_fixture_artifact_satisfies_frozen_draft_2020_schema(self) -> None:
        output, _ = self._run()
        schema = json.loads(
            (ROOT / "schemas/r04-resume-parity.schema.json").read_text(
                encoding="utf-8"
            )
        )
        jsonschema.Draft202012Validator.check_schema(schema)
        frozen_golden = schema["$defs"]["baselineReference"]["properties"][
            "r04a_golden"
        ]["properties"]
        self.assertEqual(
            frozen_golden["observation_sha256"]["const"],
            REAL_R04A_GOLDEN_OBSERVATION_SHA256,
        )
        self.assertEqual(
            frozen_golden["final_physical_sha256"]["const"],
            REAL_R04A_GOLDEN_FINAL_PHYSICAL_SHA256,
        )

        # The runtime fixture deliberately uses tiny synthetic arrays. Keep the
        # checked-in schema frozen to real R04A; substitute only the two
        # fixture-specific golden hashes in an in-memory schema copy.
        fixture_schema = copy.deepcopy(schema)
        fixture_golden = fixture_schema["$defs"]["baselineReference"][
            "properties"
        ]["r04a_golden"]["properties"]
        fixture_golden["observation_sha256"]["const"] = (
            baseline_reference.R04A_GOLDEN_OBSERVATION_SHA256
        )
        fixture_golden["final_physical_sha256"]["const"] = (
            baseline_reference.R04A_GOLDEN_FINAL_PHYSICAL_SHA256
        )
        failures = sorted(
            jsonschema.Draft202012Validator(fixture_schema).iter_errors(
                load_json(output)
            ),
            key=lambda error: tuple(str(item) for item in error.absolute_path),
        )
        self.assertEqual([failure.message for failure in failures], [])

    def test_resume_skips_only_matching_semantically_valid_artifact(self) -> None:
        output, _ = self._run()
        calls = len(self.client.requests)
        second_output, status = self._run()
        self.assertEqual(second_output, output)
        self.assertEqual(status, "skipped_valid_completion")
        self.assertEqual(len(self.client.requests), calls)

        corrupt = load_json(output)
        corrupt["pairing"]["step_records"][0]["zero_resume"]["primary"][
            "trace"
        ]["leaves"]["x_t_post_edit"]["values"][0][0] += 1.0
        atomic_write_json(output, corrupt)
        _, rerun_status = self._run()
        self.assertEqual(rerun_status, "completed")
        self.assertEqual(len(self.client.requests), calls + 32)

    def test_validator_rejects_time_algebra_parity_and_claim_mutations(self) -> None:
        output, _ = self._run()

        time = load_json(output)
        time["pairing"]["step_records"][0]["zero_resume"]["primary"]["trace"][
            "leaves"
        ]["time"]["values"][0] += 0.01
        self.assertTrue(validate_r04_resume_result(time))

        algebra = load_json(output)
        algebra["pairing"]["step_records"][0]["nonzero_resume"]["primary"][
            "trace"
        ]["leaves"]["x_t_post_edit"]["values"][0][0] += 0.01
        self.assertTrue(validate_r04_resume_result(algebra))

        parity = load_json(output)
        parity["pairing"]["step_records"][0]["zero_resume"]["primary"][
            "final_physical"
        ]["values"][0][0] += 0.01
        self.assertTrue(validate_r04_resume_result(parity))

        claim = load_json(output)
        claim["execution_boundaries"]["scientific_claim_authorized"] = True
        self.assertTrue(validate_r04_resume_result(claim))

        inappropriate_nonzero = load_json(output)
        inappropriate_nonzero["pairing"]["step_records"][0]["nonzero_resume"][
            "source_parity"
        ] = copy.deepcopy(
            inappropriate_nonzero["pairing"]["step_records"][0]["zero_resume"][
                "source_parity"
            ]
        )
        self.assertTrue(validate_r04_resume_result(inappropriate_nonzero))

    def test_validator_reconstructs_compiled_baseline_and_r04a_golden_binding(self) -> None:
        output, _ = self._run()

        compiled = load_json(output)
        record = compiled["pairing"]["baseline_reference"]["compiled_default"][
            "pre_sequence"
        ]["final_physical"]
        array = np.asarray(record["values"], dtype=np.dtype(record["dtype"]))
        changed = np.array(array, copy=True)
        changed[0, 0] += np.asarray(0.02, dtype=changed.dtype)
        compiled["pairing"]["baseline_reference"]["compiled_default"][
            "pre_sequence"
        ]["final_physical"] = _array_record(changed)
        self.assertTrue(validate_r04_resume_result(compiled))

        metric = load_json(output)
        metric["pairing"]["baseline_reference"]["compiled_vs_current_eager"][
            "pre_sequence"
        ]["physical_first_five_xyz"]["rmse"] += 1.0e-4
        self.assertTrue(validate_r04_resume_result(metric))

        golden = load_json(output)
        golden["pairing"]["baseline_reference"]["current_eager_golden_binding"][
            "final_physical_sha256"
        ] = "0" * 64
        self.assertTrue(validate_r04_resume_result(golden))

        tolerance = load_json(output)
        tolerance["pairing"]["baseline_reference"]["compiled_vs_current_eager"][
            "registered_physical_limits"
        ]["physical_xyz5_max"] = 1.0
        self.assertTrue(validate_r04_resume_result(tolerance))

    def test_zero_resume_rejects_each_source_feature_or_final_mutation(self) -> None:
        output, _ = self._run()
        locations = (
            ("trace", "x_t_post_edit"),
            ("trace", "v_post_edit"),
            ("trace", "predicted_clean_post_edit"),
            ("trace", "predicted_clean_post_edit_physical"),
            ("trace", "final_normalized"),
            ("copy", "final_physical"),
        )
        for container, key in locations:
            with self.subTest(key=key):
                value = load_json(output)
                primary = value["pairing"]["step_records"][0]["zero_resume"][
                    "primary"
                ]
                if container == "trace":
                    record = primary["trace"]["leaves"][key]
                else:
                    record = primary[key]
                array = np.asarray(record["values"], dtype=np.dtype(record["dtype"]))
                array = np.array(array, copy=True)
                array.reshape(-1)[-1] += np.asarray(0.25, dtype=array.dtype)
                replacement = _array_record(array)
                if container == "trace":
                    primary["trace"]["leaves"][key] = replacement
                    primary["trace"]["sha256"] = content_hash(
                        primary["trace"]["leaves"]
                    )
                else:
                    primary[key] = replacement
                errors = validate_r04_resume_result(value)
                self.assertTrue(errors)
                if key in {
                    "predicted_clean_post_edit_physical",
                    "final_normalized",
                    "final_physical",
                }:
                    self.assertTrue(any("parity" in error for error in errors), errors)

    def test_signed_zero_is_part_of_exact_byte_identity_and_zero_parity(self) -> None:
        self.assertFalse(
            _arrays_byte_exact(
                np.asarray([-0.0], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
            )
        )
        output, _ = self._run()
        value = load_json(output)
        primary = value["pairing"]["step_records"][0]["zero_resume"]["primary"]
        record = primary["trace"]["leaves"]["x_t_post_edit"]
        array = np.asarray(record["values"], dtype=np.dtype(record["dtype"]))
        self.assertTrue(np.signbit(array[0, 0]))
        array = np.array(array, copy=True)
        array[0, 0] = np.float32(0.0)
        primary["trace"]["leaves"]["x_t_post_edit"] = _array_record(array)
        primary["trace"]["sha256"] = content_hash(primary["trace"]["leaves"])
        errors = validate_r04_resume_result(value)
        self.assertTrue(any("byte-exact" in error or "parity" in error for error in errors))

    def test_cpu_validator_checks_semantics_bindings_and_source_job(self) -> None:
        output, _ = self._run()
        arguments = SimpleNamespace(
            validate_artifact=str(output),
            expected_config_file_sha256=EXPECTED_CONFIG_FILE_SHA256,
            expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
            expected_checkpoint_sha256=CHECKPOINT_SHA,
            expected_r04a_validation_summary_sha256=EXPECTED_R04A_VALIDATION_SUMMARY_SHA256,
            expected_source_slurm_job_id="321",
            expected_source_slurm_array_job_id="321",
            expected_source_slurm_array_task_id="0",
        )
        with mock.patch("builtins.print") as printed:
            self.assertEqual(_validate_artifact(arguments), 0)
        report = json.loads(printed.call_args.args[0])
        self.assertEqual(report["validation_error_count"], 0)
        self.assertEqual(len(report["artifact_sha256"]), 64)

        arguments.expected_source_slurm_job_id = "wrong"
        with mock.patch("builtins.print"):
            self.assertEqual(_validate_artifact(arguments), 1)

        arguments.expected_source_slurm_job_id = None
        arguments.expected_source_slurm_array_job_id = None
        arguments.expected_source_slurm_array_task_id = None
        with mock.patch("builtins.print"):
            self.assertEqual(_validate_artifact(arguments), 1)


@unittest.skipIf(np is None, "R04B runtime checks need numpy")
class R04ResumeEditTest(unittest.TestCase):
    def test_nonzero_edit_is_one_exact_float32_implementation_sentinel(self) -> None:
        edit = _nonzero_edit()
        self.assertEqual(edit.dtype, np.float32)
        self.assertEqual(edit.shape, (10, 32))
        self.assertEqual(np.count_nonzero(edit), 1)
        self.assertEqual(edit[0, 0], np.float32(0.03125))


if __name__ == "__main__":
    unittest.main()
