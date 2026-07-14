from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ModuleNotFoundError:  # dependency-free local gate
    np = None


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "main", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crfs_harness.artifacts import atomic_write_json, content_hash, load_json  # noqa: E402

if np is not None:
    from crfs_oracle.r04_labels import (  # noqa: E402
        MAXIMUM_OBBS,
        EXPECTED_CONFIG_FILE_SHA256,
        EXPECTED_MANIFEST_SHA256,
        EXPECTED_PARITY_SHA256,
        EXPECTED_R00_SUMMARY_SHA256,
        EXPECTED_R03_SUMMARY_SHA256,
        R04LabelConfig,
        _array_record,
        _pad_world_obbs,
        _trace_record,
        _validate_trace_record,
        r04_label_config_from_mapping,
        run_r04_label_case,
        valid_r04_label_completion,
        validate_r04_label_result,
    )
    from crfs_oracle.runner import OracleConfig  # noqa: E402
    from crfs_oracle.reach_progress import (  # noqa: E402
        ReachSnapshot,
        annotate_reach_snapshots,
    )
else:
    MAXIMUM_OBBS = 21


SHA = "a" * 64
CHECKPOINT_SHA = "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed"
P_MIN = 0.029897349105658888
BOXES = [
    {
        "name": "z_box",
        "center_m": [0.4, 0.2, 0.1],
        "rotation_world": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "half_size_m": [0.1, 0.2, 0.3],
    },
    {
        "name": "a_box",
        "center_m": [0.1, 0.2, 0.3],
        "rotation_world": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "half_size_m": [0.05, 0.06, 0.07],
    },
]
EEF_CENTER = [0.2, 0.1, 0.4]


def _oracle(output_root: str) -> OracleConfig:
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
        run_id="r04a-test",
    )


def _sources() -> dict:
    return {
        "r00_summary": {
            "path": "/frozen/r00-summary.json",
            "sha256": EXPECTED_R00_SUMMARY_SHA256,
            "status": "passed",
            "p_min_m": P_MIN,
        },
        "r03_summary": {
            "path": "/frozen/r03-summary.json",
            "sha256": EXPECTED_R03_SUMMARY_SHA256,
            "status": "passed",
            "learned_probe_authorized": True,
        },
        "sampler_parity": {
            "path": "/frozen/parity.json",
            "sha256": EXPECTED_PARITY_SHA256,
            "status": "passed",
            "checkpoint_sha256": CHECKPOINT_SHA,
            "normalization_asset_sha256": SHA,
        },
    }


def _config(output_root: str) -> R04LabelConfig:
    return R04LabelConfig(
        oracle=_oracle(output_root),
        enabled=True,
        apparatus_scope="label_contract_smoke_only",
        reuse_role="apparatus_only_never_train_calibrate_validate_test_or_claim",
        smoke_case_index=0,
        target_name="akita_black_bowl_1",
        trace_steps=(1, 2, 3, 4, 5),
        trace_times=(0.9, 0.8, 0.7, 0.6, 0.5),
        duplicate_trace_requests=2,
        simulator_repeats=2,
        maximum_obbs=21,
        minimum_progress_m=P_MIN,
        maximum_target_displacement_m=0.001,
        maximum_obstacle_displacement_m=0.001,
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


def _rollout() -> tuple:
    branch = ReachSnapshot(
        target_object_name="akita_black_bowl_1",
        active_obstacle_name="active_obstacle",
        eef_world_m=(0.0, 0.0, 0.0),
        target_world_m=(1.0, 0.0, 0.0),
        active_obstacle_world_m=(0.0, 1.0, 0.0),
    )
    end = ReachSnapshot(
        target_object_name="akita_black_bowl_1",
        active_obstacle_name="active_obstacle",
        eef_world_m=(-0.004, 0.0, 0.0),
        target_world_m=(1.0, 0.0, 0.0),
        active_obstacle_world_m=(0.0, 1.0, 0.0),
    )
    reach = annotate_reach_snapshots(
        branch,
        end,
        executed_actions=5,
        maximum_target_displacement_m=0.0,
        maximum_active_obstacle_displacement_m=0.0,
    ).to_dict()
    rollout = {
        "clearance_m": -0.002,
        "raw_mujoco_clearance_m": -0.001,
        "contact": True,
        "measurement_samples": 126,
        "minimum_geom_pair": ["crfs_eef_sphere", "a_box"],
        "raw_mujoco_minimum_geom_pair": ["eef_geom", "a_box"],
        "measurement": {
            "conservative_clearance_m": -0.002,
            "min_clearance_m": -0.001,
            "contact": True,
            "samples": 126,
            "distance_limit_m": 1.0,
            "conservative_eef_center_m": [0.208, 0.2, 0.3],
            "conservative_eef_radius_m": 0.06,
            "conservative_obstacle_center_m": [0.1, 0.2, 0.3],
            "conservative_obstacle_rotation_world": [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ],
            "conservative_obstacle_half_size_m": [0.05, 0.06, 0.07],
            "conservative_obstacle_geom": "a_box",
            "min_pair": ["eef_geom", "a_box"],
        },
        "start_eef_center_m": list(EEF_CENTER),
        "branch_obstacle_boxes": copy.deepcopy(BOXES),
        "end_eef_m": [-0.004, 0.0, 0.0],
        "tracked_body_motion": {
            "substep_samples": 125,
            "branch_eef_world_m": [0.0, 0.0, 0.0],
            "end_eef_world_m": [-0.004, 0.0, 0.0],
            "bodies": {
                "akita_black_bowl_1": {
                    "branch_world_m": [1.0, 0.0, 0.0],
                    "end_world_m": [1.0, 0.0, 0.0],
                    "maximum_displacement_m": 0.0,
                    "endpoint_displacement_m": 0.0,
                },
                "active_obstacle": {
                    "branch_world_m": [0.0, 1.0, 0.0],
                    "end_world_m": [0.0, 1.0, 0.0],
                    "maximum_displacement_m": 0.0,
                    "endpoint_displacement_m": 0.0,
                },
            },
        },
    }
    return rollout, reach


class FakeClient:
    def __init__(self) -> None:
        self.requests = []
        self.actions = np.arange(70, dtype=np.float32).reshape(10, 7) / 100.0

    def infer(self, request):
        controls = request["__crfs__"]
        self.requests.append(copy.deepcopy(controls))
        step = int(controls["intervention_step"])
        trace_value = np.full((10, 32), step, dtype=np.float32)
        trace_time = np.asarray(1.0 - step / 10.0, dtype=np.float32)
        velocity = trace_value + 1.0
        return {
            "actions": np.array(self.actions, copy=True),
            "crfs_trace": {
                "step_index": np.asarray(step, dtype=np.int64),
                "time": trace_time,
                "x_t": trace_value,
                "v_base": velocity,
                "predicted_clean": trace_value - trace_time * velocity,
                "predicted_clean_physical": np.full(
                    (10, 7), step / 10.0, dtype=np.float32
                ),
            },
            "policy_timing": {"infer_ms": float(step)},
        }


class FakeDomain:
    def check_success(self) -> bool:
        return False


class FakeEnvironment:
    def __init__(self) -> None:
        self.obstacle_name = "active_obstacle"
        self.prompt = "reach for the bowl"
        self.env = FakeDomain()
        self.configured = []
        self.closed = False

    def configure_case(self, case) -> None:
        self.configured.append(dict(case))

    def reset_and_settle(self):
        return {"unused": True}

    def close(self) -> None:
        self.closed = True


class R04DependencyFreeStructuralTest(unittest.TestCase):
    def test_apparatus_is_opt_in_and_forbids_training_guidance_and_correction(self) -> None:
        value = json.loads(
            (ROOT / "configs/experiments/r04_continuation_labels.json").read_text(
                encoding="utf-8"
            )
        )
        settings = value["r04a"]
        self.assertTrue(value["ready_to_run"])
        self.assertTrue(settings["enabled"])
        self.assertFalse(settings["training"])
        self.assertFalse(settings["guidance"])
        self.assertIsNone(settings["correction"])
        self.assertEqual(settings["intervention_mode"], "none")
        self.assertEqual(
            settings["exact_final_action_requirement"],
            "physical_10x7_array_equal_across_all_trace_steps_and_duplicates",
        )
        self.assertEqual(
            settings["model_action_shape_role"],
            "normalized_intermediate_trace_tensor_not_exposed_final_action",
        )

    def test_runner_fail_closed_surfaces_are_structurally_present(self) -> None:
        source = (ROOT / "main/crfs_oracle/r04_labels.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def validate_r04_label_result(", source)
        self.assertIn("def valid_r04_label_completion(", source)
        self.assertIn("def _rollout_branch_geometry(", source)
        self.assertIn('"training": False', source)
        self.assertIn('"guidance": False', source)
        self.assertIn('"correction": None', source)
        self.assertIn('"intervention_mode": "none"', source)
        self.assertIn('"return_trace": True', source)
        self.assertIn("if index >= len(trace_hashes):", source)
        validation = source.index("errors = validate_r04_label_result(result)")
        atomic_write = source.index("atomic_write_json(output, result)", validation)
        self.assertLess(validation, atomic_write)
        self.assertNotIn("import torch", source.lower())

    def test_cli_and_schema_freeze_hash_bound_atomic_case_artifact(self) -> None:
        cli = (ROOT / "main/run_r04_label_contract.py").read_text(encoding="utf-8")
        for argument in (
            "--manifest",
            "--config",
            "--output-root",
            "--run-id",
            "--case-index",
            "--checkpoint-id",
            "--checkpoint-sha256",
        ):
            self.assertIn(argument, cli)
        self.assertIn('value.get("manifest_sha256")', cli)
        self.assertIn("args.case_index != config.smoke_case_index", cli)
        schema = json.loads(
            (ROOT / "schemas/r04-label-contract.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("source_evidence", schema["required"])
        self.assertIn("trace_to_label_bindings", schema["properties"]["pairing"]["required"])
        self.assertEqual(
            schema["properties"]["simulator"]["properties"]["exact_replay"]["const"],
            True,
        )


@unittest.skipIf(np is None, "R04 runtime checks need the allocation numpy environment")
class R04GeometryTest(unittest.TestCase):
    def test_world_obbs_are_sorted_and_zero_padded_to_21(self) -> None:
        geometry = _pad_world_obbs(BOXES, branch_eef_center_m=EEF_CENTER)
        self.assertEqual(geometry["geom_names"][:2], ["a_box", "z_box"])
        self.assertEqual(geometry["geom_names"][2:], [""] * 19)
        self.assertEqual(geometry["validity_mask"], [True, True] + [False] * 19)
        self.assertEqual(len(geometry["centers_m"]), MAXIMUM_OBBS)
        self.assertEqual(geometry["centers_m"][2:], [[0.0, 0.0, 0.0]] * 19)
        payload = {key: value for key, value in geometry.items() if key != "sha256"}
        self.assertEqual(geometry["sha256"], content_hash(payload))

    def test_geometry_rejects_overflow_and_duplicate_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeding width"):
            _pad_world_obbs(
                [
                    {
                        **BOXES[0],
                        "name": f"box-{index:02d}",
                    }
                    for index in range(22)
                ],
                branch_eef_center_m=EEF_CENTER,
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            _pad_world_obbs(
                [BOXES[0], BOXES[0]], branch_eef_center_m=EEF_CENTER
            )


@unittest.skipIf(np is None, "R04 runtime checks need the allocation numpy environment")
class R04TraceContractTest(unittest.TestCase):
    def test_trace_primary_shapes_and_physical_audit_shape(self) -> None:
        trace = _trace_record(
            {
                "step_index": np.asarray(3, dtype=np.int64),
                "time": np.asarray(0.7, dtype=np.float32),
                "x_t": np.zeros((10, 32), dtype=np.float32),
                "v_base": np.zeros((10, 32), dtype=np.float32),
                "predicted_clean": np.zeros((10, 32), dtype=np.float32),
                "predicted_clean_physical": np.zeros((10, 7), dtype=np.float32),
            }
        )
        self.assertEqual(
            _validate_trace_record(
                trace, name="trace", expected_step=3, expected_time=0.7
            ),
            [],
        )
        broken = copy.deepcopy(trace)
        broken["leaves"]["x_t"] = _array_record(np.zeros((10, 31)))
        self.assertTrue(
            any(
                "shape" in error
                for error in _validate_trace_record(
                    broken, name="trace", expected_step=3, expected_time=0.7
                )
            )
        )
        semantically_broken = copy.deepcopy(trace)
        semantically_broken["leaves"]["predicted_clean"] = _array_record(
            np.ones((10, 32), dtype=np.float32)
        )
        semantically_broken["sha256"] = content_hash(
            semantically_broken["leaves"]
        )
        self.assertTrue(
            any(
                "x_t - time * v_base" in error
                for error in _validate_trace_record(
                    semantically_broken,
                    name="trace",
                    expected_step=3,
                    expected_time=0.7,
                )
            )
        )


@unittest.skipIf(np is None, "R04 runtime checks need the allocation numpy environment")
class R04ConfigContractTest(unittest.TestCase):
    def test_checked_in_config_is_opt_in_and_apparatus_only(self) -> None:
        value = json.loads(
            (ROOT / "configs/experiments/r04_continuation_labels.json").read_text(
                encoding="utf-8"
            )
        )
        with mock.patch(
            "crfs_oracle.r04_labels._validated_sources", return_value=_sources()
        ):
            config = r04_label_config_from_mapping(
                value,
                _oracle("/tmp/r04a"),
                repo_root=ROOT,
                config_file_sha256=EXPECTED_CONFIG_FILE_SHA256,
            )
        self.assertTrue(config.enabled)
        self.assertEqual(config.trace_steps, (1, 2, 3, 4, 5))
        self.assertEqual(config.simulator_repeats, 2)

        unsafe = copy.deepcopy(value)
        unsafe["r04a"]["training"] = True
        with mock.patch(
            "crfs_oracle.r04_labels._validated_sources", return_value=_sources()
        ):
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                r04_label_config_from_mapping(
                    unsafe,
                    _oracle("/tmp/r04a"),
                    repo_root=ROOT,
                    config_file_sha256=EXPECTED_CONFIG_FILE_SHA256,
                )


@unittest.skipIf(np is None, "R04 runtime checks need the allocation numpy environment")
class R04RunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.client = FakeClient()
        self.environment = FakeEnvironment()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self):
        geometry = _pad_world_obbs(BOXES, branch_eef_center_m=EEF_CENTER)
        allocation = {
            "SLURM_JOB_ID": "123",
            "SLURM_ARRAY_JOB_ID": "123",
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_JOB_PARTITION": "worker-mig",
            "CUDA_VISIBLE_DEVICES": "MIG-0",
            "EXPECTED_GIT_COMMIT": "b" * 40,
        }
        with mock.patch.dict(os.environ, allocation, clear=False), mock.patch(
            "crfs_oracle.r04_labels._git_state",
            return_value=("b" * 40, False),
        ), mock.patch(
            "crfs_oracle.r04_labels.capture_reach_snapshot",
            return_value=object(),
        ), mock.patch(
            "crfs_oracle.r04_labels._target_contact_at_branch",
            return_value=False,
        ), mock.patch(
            "crfs_oracle.r04_labels._capture_branch_geometry",
            return_value=geometry,
        ), mock.patch(
            "crfs_oracle.r04_labels.policy_observation",
            return_value=_policy_input(),
        ), mock.patch(
            "crfs_oracle.r04_labels.annotate_reach_rollout",
            side_effect=lambda *args, **kwargs: copy.deepcopy(_rollout()),
        ):
            return run_r04_label_case(
                _case(),
                _config(str(self.root)),
                repo_root=ROOT,
                input_manifest_sha256=EXPECTED_MANIFEST_SHA256,
                client=self.client,
                environment=self.environment,
            )

    def test_runner_binds_ten_real_trace_calls_to_exact_action_and_label(self) -> None:
        output, status = self._run()
        self.assertEqual(status, "completed")
        self.assertEqual(len(self.client.requests), 10)
        self.assertEqual(
            [item["intervention_step"] for item in self.client.requests],
            [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        )
        self.assertTrue(all(item["intervention_mode"] == "none" for item in self.client.requests))
        self.assertTrue(all("correction" not in item for item in self.client.requests))

        result = load_json(output)
        self.assertEqual(validate_r04_label_result(result), [])
        self.assertEqual(result["label"]["D_sim_m"], -0.002)
        self.assertTrue(result["label"]["contact_any"])
        self.assertAlmostEqual(result["outcome"]["reach_progress_m"], -0.004)
        self.assertTrue(result["outcome"]["retained_regardless_of_safety_or_progress"])
        self.assertEqual(len(result["pairing"]["trace_to_label_bindings"]), 10)
        self.assertTrue(result["pairing"]["all_final_actions_exact"])
        self.assertEqual(
            result["geometry"]["sha256"],
            result["simulator"]["repeats"][0]["branch_geometry_sha256"],
        )
        self.assertTrue(
            valid_r04_label_completion(
                output,
                case_id=_case()["case_id"],
                run_id="r04a-test",
                config_hash=result["config_hash"],
                input_manifest_sha256=EXPECTED_MANIFEST_SHA256,
                config_file_sha256=EXPECTED_CONFIG_FILE_SHA256,
                checkpoint_sha256=CHECKPOINT_SHA,
            )
        )

    def test_resume_skips_only_matching_valid_artifact(self) -> None:
        output, _ = self._run()
        requests = len(self.client.requests)
        second_output, status = self._run()
        self.assertEqual(second_output, output)
        self.assertEqual(status, "skipped_valid_completion")
        self.assertEqual(len(self.client.requests), requests)

        stale = load_json(output)
        stale["provenance"]["git_commit"] = "c" * 40
        stale["provenance"]["reviewed_git_commit"] = "c" * 40
        atomic_write_json(output, stale)
        _, stale_status = self._run()
        self.assertEqual(stale_status, "completed")
        self.assertEqual(len(self.client.requests), requests + 10)

    def test_validator_rejects_corrupt_trace_and_geometry_without_throwing(self) -> None:
        output, _ = self._run()
        result = load_json(output)
        result["pairing"]["trace_requests"][0]["primary"]["trace"] = None
        result["simulator"]["repeats"][0]["rollout"]["start_eef_center_m"][0] += 0.1
        errors = validate_r04_label_result(result)
        self.assertTrue(errors)
        self.assertTrue(any("geometry" in item for item in errors))

        empty_scalar = load_json(output)
        empty_scalar["pairing"]["trace_requests"][0]["primary"]["trace"][
            "leaves"
        ]["step_index"] = _array_record(np.empty((0,), dtype=np.int64))
        self.assertTrue(validate_r04_label_result(empty_scalar))

        invalid_box = load_json(output)
        invalid_box["simulator"]["repeats"][0]["rollout"][
            "branch_obstacle_boxes"
        ] = [1]
        self.assertTrue(validate_r04_label_result(invalid_box))

        endpoint_only = load_json(output)
        endpoint_only["simulator"]["repeats"][0]["rollout"]["measurement"] = {}
        endpoint_errors = validate_r04_label_result(endpoint_only)
        self.assertTrue(any("raw evidence" in item for item in endpoint_errors))

    def test_validator_reconstructs_observation_action_label_and_outcome(self) -> None:
        output, _ = self._run()

        observation = load_json(output)
        observation["pairing"]["fixed_observation"]["sha256"] = SHA
        self.assertTrue(validate_r04_label_result(observation))

        prefix = load_json(output)
        prefix["pairing"]["executed_action_prefix_sha256"] = SHA
        self.assertTrue(validate_r04_label_result(prefix))

        label = load_json(output)
        label["label"]["reach_progress_m"] = 999.0
        label_payload = {
            key: item for key, item in label["label"].items() if key != "sha256"
        }
        label["label"]["sha256"] = content_hash(label_payload)
        for binding in label["pairing"]["trace_to_label_bindings"]:
            binding["simulator_label_sha256"] = label["label"]["sha256"]
        self.assertTrue(validate_r04_label_result(label))

        outcome = load_json(output)
        outcome["outcome"]["D_sim_m"] = 999.0
        self.assertTrue(validate_r04_label_result(outcome))


class R04SchemaTest(unittest.TestCase):
    def test_schema_freezes_apparatus_and_pairing_surface(self) -> None:
        schema = json.loads(
            (ROOT / "schemas/r04-label-contract.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(schema["properties"]["gate"]["const"], "R04A")
        self.assertEqual(
            schema["properties"]["usage_restriction"]["const"],
            "apparatus_only_never_train_calibrate_validate_test_or_claim",
        )
        self.assertEqual(
            schema["properties"]["geometry"]["properties"]["maximum_obbs"]["const"],
            21,
        )
        self.assertEqual(
            schema["properties"]["pairing"]["properties"]["trace_requests"]["minItems"],
            5,
        )


if __name__ == "__main__":
    unittest.main()
