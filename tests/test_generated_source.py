from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import jsonschema
except ModuleNotFoundError:
    jsonschema = None

try:
    import numpy as _numpy_test_runtime  # noqa: F401
except (ImportError, ModuleNotFoundError, OSError):
    NUMPY_RUNTIME_AVAILABLE = False
else:
    NUMPY_RUNTIME_AVAILABLE = True

try:
    import mujoco as _mujoco_test_runtime  # noqa: F401
except (ImportError, ModuleNotFoundError, OSError):
    MUJOCO_RUNTIME_AVAILABLE = False
else:
    MUJOCO_RUNTIME_AVAILABLE = NUMPY_RUNTIME_AVAILABLE


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "main", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crfs_harness.artifacts import content_hash, file_sha256  # noqa: E402
_SPEC = importlib.util.spec_from_file_location(
    "generated_source_under_test", ROOT / "main/crfs_oracle/generated_source.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ACTIVE_SCHEDULE = _MODULE.ACTIVE_SCHEDULE
EXPECTED_BDDL_SHA256 = _MODULE.EXPECTED_BDDL_SHA256
OBSTACLE_POSES = _MODULE.OBSTACLE_POSES
SOURCE_ESTIMAND = _MODULE.SOURCE_ESTIMAND
USAGE_RESTRICTION = _MODULE.USAGE_RESTRICTION
allocation_provenance = _MODULE.allocation_provenance
freeze_model_xml = _MODULE.freeze_model_xml
load_generated_source_config = _MODULE.load_generated_source_config
rejection_bundle = _MODULE.rejection_bundle
rehydrate_model_xml = _MODULE.rehydrate_model_xml
validate_generated_source_artifact = _MODULE.validate_generated_source_artifact
validate_generated_source_config = _MODULE.validate_generated_source_config
source_branch_identity = _MODULE.source_branch_identity


CONFIG_PATH = ROOT / "configs/experiments/task0_single_obstacle_generated_v1.json"
SCHEMA_PATH = ROOT / "schemas/generated-source-state.schema.json"


def _accepted_bundle_fixture() -> dict[str, object]:
    """Build a semantic-validator fixture without compiling a simulator model."""

    import numpy as np

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    group = config["source_groups"][0]
    placement_rng = np.random.Generator(np.random.PCG64(group["placement_seed"]))
    placement_before = copy.deepcopy(placement_rng.bit_generator.state)
    active_x = float(placement_rng.uniform(*config["active_x_uniform_m"]))
    placement_after = copy.deepcopy(placement_rng.bit_generator.state)
    reset_state = np.random.RandomState(group["reset_seed"]).get_state()

    def array_record(value):
        return _MODULE._array_record(np.asarray(value))

    def legacy_record(value):
        return {
            "algorithm": value[0],
            "keys": array_record(value[1]),
            "position": int(value[2]),
            "has_gauss": int(value[3]),
            "cached_gaussian": float(value[4]),
        }

    flat = array_record(np.asarray([0.0], dtype=np.float64))
    settle_history = []
    integration_history = []
    for index in range(_MODULE.SETTLE_STEPS):
        flat_record = copy.deepcopy(flat)
        flat_record["settle_step"] = index + 1
        settle_history.append(flat_record)
        integration_record = array_record(
            np.asarray([float(index + 1)], dtype=np.float64)
        )
        integration_record["settle_step"] = index + 1
        integration_history.append(integration_record)
    integration_hashes = [record["sha256"] for record in integration_history]
    integration = {
        "spec": {
            "name": _MODULE.INTEGRATION_STATE_SPEC_NAME,
            "value": _MODULE.INTEGRATION_STATE_SPEC_VALUE,
            "size": 1,
            "dtype": _MODULE.INTEGRATION_STATE_DTYPE,
        },
        "pre_settle": copy.deepcopy(flat),
        "settle_history": integration_history,
        "settle_sequence_sha256": content_hash(integration_hashes),
        "final": array_record(np.asarray([float(_MODULE.SETTLE_STEPS)], dtype=np.float64)),
    }
    states = {
        "raw_reset": copy.deepcopy(flat),
        "pre_settle": copy.deepcopy(flat),
        "settle_history": settle_history,
        "final": copy.deepcopy(flat),
        "integration": integration,
    }

    bddl_path = ROOT / config["bddl_path"]
    bddl_content = bddl_path.read_text(encoding="utf-8")
    portable_xml = (
        '<mujoco><asset><texture name="fixture" file="crfs-asset://0000" /></asset></mujoco>'
    )
    assets = [
        {
            "token": "crfs-asset://0000",
            "element_tag": "texture",
            "element_name": "fixture",
            "original_path": str(bddl_path.resolve()),
            "locator": {"kind": "repo_relative", "path": config["bddl_path"]},
            "size_bytes": bddl_path.stat().st_size,
            "sha256": file_sha256(bddl_path),
        }
    ]
    model = {
        "format": "portable_finalized_mujoco_xml_with_hashed_asset_tokens",
        "xml": portable_xml,
        "sha256": hashlib.sha256(portable_xml.encode("utf-8")).hexdigest(),
        "assets": assets,
        "assets_sha256": content_hash(_MODULE._semantic_asset_manifest(assets)),
    }

    zero_pose = array_record(np.zeros(7, dtype=np.float64))
    edits = []
    for object_name, (position, quaternion) in OBSTACLE_POSES.items():
        edits.append(
            {
                "operation": "park_declared_obstacle",
                "object_name": object_name,
                "joint_name": f"{object_name}_joint",
                "joint_type": "free",
                "addressing": "named_joint_no_hard_coded_qpos_offset",
                "before": copy.deepcopy(zero_pose),
                "after": array_record(np.asarray(position + quaternion, dtype=np.float64)),
            }
        )
    active_name = group["active_obstacle_name"]
    active_pose = [active_x, _MODULE.ACTIVE_Y_M, _MODULE.ACTIVE_Z_M]
    active_quaternion = OBSTACLE_POSES[active_name][1]
    edits.append(
        {
            "operation": "activate_assigned_obstacle",
            "object_name": active_name,
            "joint_name": f"{active_name}_joint",
            "joint_type": "free",
            "addressing": "named_joint_no_hard_coded_qpos_offset",
            "before": copy.deepcopy(zero_pose),
            "after": array_record(
                np.asarray(tuple(active_pose) + active_quaternion, dtype=np.float64)
            ),
        }
    )
    base_pose = [active_x, _MODULE.ACTIVE_Y_M, _MODULE.BOX_BASE_Z_M]
    edits.append(
        {
            "operation": "move_box_base_to_active_xy",
            "object_name": _MODULE.BOX_BASE_NAME,
            "joint_name": f"{_MODULE.BOX_BASE_NAME}_joint",
            "joint_type": "free",
            "addressing": "named_joint_no_hard_coded_qpos_offset",
            "before": copy.deepcopy(zero_pose),
            "after": array_record(
                np.asarray(tuple(base_pose) + _MODULE.BOX_BASE_QUATERNION, dtype=np.float64)
            ),
        }
    )

    observation_components = {}
    geometry_joint_poses = [
        {
            "object_name": object_name,
            "joint_name": f"{object_name}_joint",
            "qpos": array_record(np.zeros(7, dtype=np.float64)),
        }
        for object_name in list(OBSTACLE_POSES) + [_MODULE.BOX_BASE_NAME]
    ]
    geometry = {
        "active_obstacle_name": active_name,
        "joint_poses": geometry_joint_poses,
        "active_obstacle_geoms": [
            {
                "name": "fixture_geom",
                "body_name": "fixture_body",
                "type_id": 6,
                "contype": 1,
                "conaffinity": 1,
                "size": array_record(np.ones(3, dtype=np.float64)),
                "world_position": array_record(np.zeros(3, dtype=np.float64)),
                "world_rotation": array_record(np.eye(3, dtype=np.float64)),
            }
        ],
    }
    branch_identity = source_branch_identity(
        bddl_sha256=config["bddl_sha256"],
        portable_model_xml_sha256=model["sha256"],
        model_asset_manifest_sha256=model["assets_sha256"],
        final_flattened_state_sha256=flat["sha256"],
        observation_sha256=content_hash(observation_components),
        geometry_sha256=content_hash(geometry),
        integration_state_spec_name=integration["spec"]["name"],
        integration_state_spec_value=integration["spec"]["value"],
        integration_state_size=integration["spec"]["size"],
        integration_state_dtype=integration["spec"]["dtype"],
        pre_settle_integration_state_sha256=integration["pre_settle"]["sha256"],
        settle_integration_state_sequence_sha256=integration["settle_sequence_sha256"],
        final_integration_state_sha256=integration["final"]["sha256"],
    )
    branch = {
        "state_sha256": flat["sha256"],
        "observation_sha256": content_hash(observation_components),
        "observation_components": observation_components,
        "geometry_sha256": content_hash(geometry),
        "geometry": geometry,
        "active_obstacle_name": active_name,
        "source_branch_identity": branch_identity["payload"],
        "source_branch_sha256": branch_identity["sha256"],
    }
    proof_base = {
        "model_xml_sha256": model["sha256"],
        "pre_settle_state_sha256": flat["sha256"],
        "settle_state_sha256": [record["sha256"] for record in settle_history],
        "final_state_sha256": flat["sha256"],
        "integration_state_spec": copy.deepcopy(integration["spec"]),
        "pre_settle_integration_state_sha256": integration["pre_settle"]["sha256"],
        "settle_integration_state_sha256": integration_hashes,
        "settle_integration_state_sequence_sha256": integration["settle_sequence_sha256"],
        "final_integration_state_sha256": integration["final"]["sha256"],
        "observation_sha256": branch["observation_sha256"],
        "geometry_sha256": branch["geometry_sha256"],
        "exact_replay": True,
    }
    proofs = []
    for index in (1, 2):
        proof = copy.deepcopy(proof_base)
        proof["fresh_load_index"] = index
        proofs.append(proof)

    bundle = {
        "schema_version": _MODULE.SCHEMA_VERSION,
        "artifact_type": _MODULE.ARTIFACT_TYPE,
        "source_estimand": SOURCE_ESTIMAND,
        "status": _MODULE.ACCEPTED_STATUS,
        "run_id": "semantic-fixture",
        "source_state": {
            "source_state_id": f"gsrc-{branch_identity['sha256'][:16]}",
            "source_state_sha256": flat["sha256"],
            "source_branch_sha256": branch_identity["sha256"],
            "generation_request_id": group["generation_request_id"],
            "group_index": 0,
            "task_suite": _MODULE.TASK_SUITE,
            "safety_level": "generated",
            "task_index": _MODULE.TASK_INDEX,
            "task_name": _MODULE.TASK_NAME,
            "active_obstacle_name": active_name,
        },
        "config": {
            "file_sha256": file_sha256(CONFIG_PATH),
            "scientific_sha256": content_hash(config),
            "snapshot": config,
        },
        "bddl": {
            "repo_relative_path": config["bddl_path"],
            "sha256": config["bddl_sha256"],
            "content": bddl_content,
        },
        "model": model,
        "rng": {
            "reset_seed": group["reset_seed"],
            "reset_api": "env.seed_plus_numpy_legacy_seed_before_exactly_one_reset",
            "numpy_legacy_state_before_reset": legacy_record(reset_state),
            "numpy_legacy_state_after_reset": legacy_record(reset_state),
            "placement_seed": group["placement_seed"],
            "placement_algorithm": "numpy.random.PCG64",
            "placement_state_before": placement_before,
            "placement_state_after": placement_after,
            "active_x_uniform_draw_m": active_x,
            "active_x_uniform_range_m": list(_MODULE.ACTIVE_X_RANGE_M),
        },
        "edits": {
            "contract": "named_free_joints_only_no_hard_coded_qpos_offsets",
            "active_pose_xyz_m": active_pose,
            "box_base_pose_xyz_m": base_pose,
            "log": edits,
        },
        "states": states,
        "settle": {
            "step_count": _MODULE.SETTLE_STEPS,
            "action_semantics": "baseline_LIBERO_dummy_control_no_policy_action",
            "actions": array_record(
                np.repeat(
                    np.asarray(_MODULE.DUMMY_ACTION, dtype=np.float64)[None, :],
                    _MODULE.SETTLE_STEPS,
                    axis=0,
                )
            ),
        },
        "branch": branch,
        "replay_proofs": proofs,
        "selection": _MODULE._selection_record(),
        "rejection": {
            "rejected": False,
            "outcome_blind": True,
            "stage": None,
            "code": None,
            "message": None,
        },
        "provenance": {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "baseline_commit": "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b",
            "slurm_job_id": "1",
            "slurm_array_job_id": "1",
            "slurm_array_task_id": "0",
            "partition": "mig",
            "host": "fixture",
            "python": "fixture",
            "physics_device": "cpu",
            "render_device": "cpu_osmesa",
            "allocation_visible_gpu": "MIG-fixture",
            "mujoco_gl": "osmesa",
            "pyopengl_platform": "osmesa",
            "purpose": "source_state_generation_and_rendering_only",
            "policy_server_started": False,
            "policy_calls": 0,
            "model_loaded": False,
            "training": False,
            "created_at_utc": "fixture",
        },
        "usage_restriction": USAGE_RESTRICTION,
    }
    bundle["bundle_content_sha256"] = content_hash(bundle)
    return bundle


def _rebind_fixture_branch(bundle: dict[str, object]) -> None:
    """Recompute every portable identity downstream of branch geometry."""

    branch = bundle["branch"]
    geometry_sha256 = content_hash(branch["geometry"])
    branch["geometry_sha256"] = geometry_sha256
    integration = bundle["states"]["integration"]
    model = bundle["model"]
    identity = source_branch_identity(
        bddl_sha256=bundle["bddl"]["sha256"],
        portable_model_xml_sha256=model["sha256"],
        model_asset_manifest_sha256=model["assets_sha256"],
        final_flattened_state_sha256=branch["state_sha256"],
        observation_sha256=branch["observation_sha256"],
        geometry_sha256=geometry_sha256,
        integration_state_spec_name=integration["spec"]["name"],
        integration_state_spec_value=integration["spec"]["value"],
        integration_state_size=integration["spec"]["size"],
        integration_state_dtype=integration["spec"]["dtype"],
        pre_settle_integration_state_sha256=integration["pre_settle"]["sha256"],
        settle_integration_state_sequence_sha256=integration["settle_sequence_sha256"],
        final_integration_state_sha256=integration["final"]["sha256"],
    )
    branch["source_branch_identity"] = identity["payload"]
    branch["source_branch_sha256"] = identity["sha256"]
    bundle["source_state"]["source_branch_sha256"] = identity["sha256"]
    bundle["source_state"]["source_state_id"] = f"gsrc-{identity['sha256'][:16]}"
    for proof in bundle["replay_proofs"]:
        proof["geometry_sha256"] = geometry_sha256
    bundle.pop("bundle_content_sha256", None)
    bundle["bundle_content_sha256"] = content_hash(bundle)


class GeneratedSourceConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_checked_in_plan_is_balanced_unique_and_review_activated(self) -> None:
        self.assertEqual(validate_generated_source_config(self.config, repo_root=ROOT), [])
        self.assertEqual(self.config["source_estimand"], SOURCE_ESTIMAND)
        self.assertTrue(self.config["pilot_only"])
        self.assertTrue(self.config["ready_to_run"])
        self.assertEqual(self.config["blocked_on"], [])
        self.assertFalse(self.config["ready_for_training"])
        self.assertFalse(self.config["ready_for_claims"])
        self.assertFalse(self.config["official_safelibero_level_ii"])
        self.assertEqual(self.config["evidence_tier"], "new_custom_retired_design_pilot_only")
        self.assertEqual(
            self.config["usage_restriction"],
            "retired_design_pilot_only_never_probe_train_calibrate_validate_test_or_claim",
        )
        self.assertEqual(self.config["safety_level"], "generated")
        groups = self.config["source_groups"]
        self.assertEqual([row["active_obstacle_name"] for row in groups], list(ACTIVE_SCHEDULE) * 2)
        self.assertEqual(len({row["reset_seed"] for row in groups}), 10)
        self.assertEqual(len({row["placement_seed"] for row in groups}), 10)
        self.assertEqual(len({row["generation_request_id"] for row in groups}), 10)
        self.assertTrue(all("source_state_id" not in row for row in groups))

    def test_plan_freezes_named_pose_constants_and_exact_bddl(self) -> None:
        self.assertEqual(self.config["bddl_sha256"], EXPECTED_BDDL_SHA256)
        bddl = ROOT / self.config["bddl_path"]
        self.assertEqual(file_sha256(bddl), EXPECTED_BDDL_SHA256)
        self.assertEqual(set(self.config["obstacles"]), set(OBSTACLE_POSES))
        self.assertEqual(self.config["settle_steps"], 20)
        self.assertEqual(self.config["dummy_action"], [0.0] * 6 + [-1.0])
        self.assertEqual(self.config["active_x_uniform_m"], [-0.08, -0.07])
        self.assertEqual(self.config["active_y_m"], 0.03)
        self.assertEqual(self.config["active_z_m"], 1.55)
        self.assertEqual(self.config["box_base_z_m"], 1.05)
        self.assertEqual(self.config["render_backend"], "osmesa")

    def test_config_validation_rejects_seed_reuse_or_official_level_claim(self) -> None:
        duplicate = copy.deepcopy(self.config)
        duplicate["source_groups"][1]["reset_seed"] = duplicate["source_groups"][0]["reset_seed"]
        self.assertTrue(any("independent reset seed" in item for item in validate_generated_source_config(duplicate)))
        mislabeled = copy.deepcopy(self.config)
        mislabeled["official_safelibero_level_ii"] = True
        self.assertTrue(validate_generated_source_config(mislabeled))
        conditioned = copy.deepcopy(self.config)
        conditioned["source_groups"][0]["collision"] = True
        self.assertTrue(
            any("unexpected or missing fields" in item for item in validate_generated_source_config(conditioned))
        )

    def test_loader_content_binds_config_file(self) -> None:
        config = load_generated_source_config(
            CONFIG_PATH,
            repo_root=ROOT,
            expected_config_sha256=file_sha256(CONFIG_PATH),
        )
        self.assertEqual(config, self.config)
        with self.assertRaisesRegex(ValueError, "expected SHA-256"):
            load_generated_source_config(CONFIG_PATH, repo_root=ROOT, expected_config_sha256="0" * 64)


class GeneratedSourceArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.rejection = rejection_bundle(
            config=self.config,
            config_file_sha256=file_sha256(CONFIG_PATH),
            group_index=0,
            run_id="generated-source-test",
            stage="fresh_load_replay",
            code="ReplayMismatch",
            message="state bytes differ",
            provenance={},
        )

    def test_rejection_is_atomic_payload_outcome_blind_and_has_no_final_state_id(self) -> None:
        self.assertEqual(validate_generated_source_artifact(self.rejection), [])
        self.assertEqual(self.rejection["status"], "rejected")
        self.assertTrue(self.rejection["rejection"]["outcome_blind"])
        self.assertIn("generation_request_id", self.rejection["source_state"])
        self.assertNotIn("source_state_id", self.rejection["source_state"])
        self.assertEqual(self.rejection["usage_restriction"], USAGE_RESTRICTION)

    def test_rejection_tampering_fails_content_hash(self) -> None:
        tampered = copy.deepcopy(self.rejection)
        tampered["rejection"]["message"] = "changed"
        self.assertIn("bundle_content_sha256 differs", validate_generated_source_artifact(tampered))

    @unittest.skipUnless(NUMPY_RUNTIME_AVAILABLE, "accepted semantic fixture requires NumPy")
    def test_accepted_fixture_passes_semantic_validation(self) -> None:
        self.assertEqual(validate_generated_source_artifact(_accepted_bundle_fixture()), [])

    @unittest.skipUnless(NUMPY_RUNTIME_AVAILABLE, "accepted schema fixture requires NumPy")
    @unittest.skipIf(jsonschema is None, "jsonschema is optional in dependency-free local gate")
    def test_accepted_fixture_satisfies_draft_2020_schema(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(_accepted_bundle_fixture())

    def test_recomputed_hash_cannot_hide_outcome_conditioning_field(self) -> None:
        from crfs_harness.artifacts import content_hash

        conditioned = copy.deepcopy(self.rejection)
        conditioned["collision"] = True
        conditioned.pop("bundle_content_sha256")
        conditioned["bundle_content_sha256"] = content_hash(conditioned)
        self.assertIn(
            "rejected artifact has unexpected or missing top-level fields",
            validate_generated_source_artifact(conditioned),
        )

    @unittest.skipIf(jsonschema is None, "jsonschema is optional in dependency-free local gate")
    def test_rejection_satisfies_draft_2020_schema(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(self.rejection)

    @unittest.skipUnless(NUMPY_RUNTIME_AVAILABLE, "accepted semantic fixture requires NumPy")
    def test_recomputed_hash_cannot_hide_outcome_key_in_any_settle_record(self) -> None:
        baseline = _accepted_bundle_fixture()
        self.assertEqual(validate_generated_source_artifact(baseline), [])
        schema_validator = None
        if jsonschema is not None:
            schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
            schema_validator = jsonschema.Draft202012Validator(schema)
        targets = (
            ("flattened", baseline["states"]["settle_history"][0]),
            ("integration", baseline["states"]["integration"]["settle_history"][0]),
        )
        for label, _record in targets:
            with self.subTest(label=label):
                tampered = copy.deepcopy(baseline)
                if label == "flattened":
                    record = tampered["states"]["settle_history"][0]
                    expected_error = (
                        "states.settle_history[0] has unexpected or missing settle-state fields"
                    )
                else:
                    record = tampered["states"]["integration"]["settle_history"][0]
                    expected_error = (
                        "states.integration.settle_history[0] has unexpected or missing "
                        "settle-state fields"
                    )
                record["collision"] = False
                tampered.pop("bundle_content_sha256")
                tampered["bundle_content_sha256"] = content_hash(tampered)
                errors = validate_generated_source_artifact(tampered)
                self.assertNotIn("bundle_content_sha256 differs", errors)
                self.assertIn(expected_error, errors)
                if schema_validator is not None:
                    with self.assertRaises(jsonschema.ValidationError):
                        schema_validator.validate(tampered)

    @unittest.skipUnless(NUMPY_RUNTIME_AVAILABLE, "accepted semantic fixture requires NumPy")
    def test_recomputed_hash_cannot_hide_wrong_reset_api_or_bool_settle_step(self) -> None:
        baseline = _accepted_bundle_fixture()
        wrong_api = copy.deepcopy(baseline)
        wrong_api["rng"]["reset_api"] = "wrong"
        wrong_api.pop("bundle_content_sha256")
        wrong_api["bundle_content_sha256"] = content_hash(wrong_api)
        api_errors = validate_generated_source_artifact(wrong_api)
        self.assertNotIn("bundle_content_sha256 differs", api_errors)
        self.assertIn("rng reset API differs from the frozen one-reset contract", api_errors)

        bool_step = copy.deepcopy(baseline)
        bool_step["states"]["integration"]["settle_history"][0]["settle_step"] = True
        bool_step.pop("bundle_content_sha256")
        bool_step["bundle_content_sha256"] = content_hash(bool_step)
        step_errors = validate_generated_source_artifact(bool_step)
        self.assertNotIn("bundle_content_sha256 differs", step_errors)
        self.assertIn(
            "states.integration.settle_history[0].settle_step must equal 1",
            step_errors,
        )

    @unittest.skipUnless(NUMPY_RUNTIME_AVAILABLE, "accepted semantic fixture requires NumPy")
    def test_integration_identity_and_proof_tampers_fail_closed(self) -> None:
        baseline = _accepted_bundle_fixture()
        history_hashes = baseline["states"]["integration"]["settle_history"]
        cases = (
            (
                "pre_settle",
                lambda value: value["states"]["integration"]["pre_settle"]["values"].__setitem__(
                    0, 1.0
                ),
                "states.integration.pre_settle.sha256 differs from exact dtype/shape/bytes",
            ),
            (
                "history",
                lambda value: value["states"]["integration"]["settle_history"][0][
                    "values"
                ].__setitem__(0, -1.0),
                "states.integration.settle_history[0].sha256 differs from exact dtype/shape/bytes",
            ),
            (
                "final",
                lambda value: value["states"]["integration"]["final"]["values"].__setitem__(
                    0, -1.0
                ),
                "states.integration.final.sha256 differs from exact dtype/shape/bytes",
            ),
            (
                "ordered_sequence",
                lambda value: value["states"]["integration"].__setitem__(
                    "settle_sequence_sha256",
                    content_hash([record["sha256"] for record in reversed(history_hashes)]),
                ),
                "states.integration.settle_sequence_sha256 differs from the ordered history",
            ),
            (
                "spec",
                lambda value: value["states"]["integration"]["spec"].__setitem__("value", 1),
                "states.integration.spec.value differs from frozen mjSTATE_INTEGRATION",
            ),
            (
                "branch_identity",
                lambda value: value["branch"]["source_branch_identity"].__setitem__(
                    "integration_state_size", 2
                ),
                "branch.source_branch_identity differs from the immutable scientific payload",
            ),
            (
                "proof_1",
                lambda value: value["replay_proofs"][0].__setitem__(
                    "final_integration_state_sha256", "c" * 64
                ),
                "replay_proofs[0] differs from exact recorded history",
            ),
            (
                "proof_2",
                lambda value: value["replay_proofs"][1].__setitem__(
                    "final_integration_state_sha256", "c" * 64
                ),
                "replay_proofs[1] differs from exact recorded history",
            ),
        )
        for label, mutate, expected_error in cases:
            with self.subTest(label=label):
                tampered = copy.deepcopy(baseline)
                mutate(tampered)
                tampered.pop("bundle_content_sha256")
                tampered["bundle_content_sha256"] = content_hash(tampered)
                errors = validate_generated_source_artifact(tampered)
                self.assertNotIn("bundle_content_sha256 differs", errors)
                self.assertIn(expected_error, errors)

    @unittest.skipUnless(NUMPY_RUNTIME_AVAILABLE, "accepted semantic fixture requires NumPy")
    def test_rehashed_nested_geometry_outcome_key_is_rejected(self) -> None:
        tampered = _accepted_bundle_fixture()
        tampered["branch"]["geometry"]["joint_poses"][0]["qpos"]["collision"] = False
        _rebind_fixture_branch(tampered)
        errors = validate_generated_source_artifact(tampered)
        self.assertNotIn("bundle_content_sha256 differs", errors)
        self.assertEqual(
            errors,
            [
                "branch.geometry.joint_poses[0].qpos has unexpected or missing "
                "array-record fields"
            ],
        )
        if jsonschema is not None:
            schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.Draft202012Validator(schema).validate(tampered)

    @unittest.skipUnless(NUMPY_RUNTIME_AVAILABLE, "accepted geometry fixture requires NumPy")
    @unittest.skipIf(jsonschema is None, "geometry schema checks require jsonschema")
    def test_fully_rebound_geometry_contract_mutations_fail_semantic_and_schema(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        schema_validator = jsonschema.Draft202012Validator(schema)
        cases = (
            (
                "wrong_active_name",
                lambda value: value["branch"]["geometry"].__setitem__(
                    "active_obstacle_name", "collision"
                ),
                "branch geometry active obstacle differs from the branch identity",
            ),
            (
                "empty_joint_poses",
                lambda value: value["branch"]["geometry"].__setitem__("joint_poses", []),
                "branch geometry joint_poses must contain exactly seven ordered objects",
            ),
            (
                "empty_active_geoms",
                lambda value: value["branch"]["geometry"].__setitem__(
                    "active_obstacle_geoms", []
                ),
                "branch geometry active_obstacle_geoms must be non-empty",
            ),
            (
                "renamed_joint_object",
                lambda value: value["branch"]["geometry"]["joint_poses"][0].__setitem__(
                    "object_name", "collision"
                ),
                "branch geometry joint_poses object order differs",
            ),
            (
                "scalar_outcome_string",
                lambda value: value["branch"]["geometry"]["active_obstacle_geoms"][
                    0
                ].__setitem__("type_id", "collision"),
                "branch geometry active_obstacle_geoms[0].type_id must be a nonnegative integer",
            ),
        )
        for label, mutate, expected_error in cases:
            with self.subTest(label=label):
                tampered = _accepted_bundle_fixture()
                mutate(tampered)
                _rebind_fixture_branch(tampered)
                errors = validate_generated_source_artifact(tampered)
                self.assertNotIn("bundle_content_sha256 differs", errors)
                self.assertIn(expected_error, errors)
                with self.assertRaises(jsonschema.ValidationError):
                    schema_validator.validate(tampered)

    @unittest.skipUnless(NUMPY_RUNTIME_AVAILABLE, "accepted geometry fixture requires NumPy")
    def test_fully_rebound_joint_name_swap_is_rejected_relationally(self) -> None:
        tampered = _accepted_bundle_fixture()
        joint_poses = tampered["branch"]["geometry"]["joint_poses"]
        joint_poses[0]["joint_name"], joint_poses[1]["joint_name"] = (
            joint_poses[1]["joint_name"],
            joint_poses[0]["joint_name"],
        )
        _rebind_fixture_branch(tampered)
        errors = validate_generated_source_artifact(tampered)
        self.assertNotIn("bundle_content_sha256 differs", errors)
        self.assertIn("branch geometry joint names differ from the edit log", errors)
        if jsonschema is not None:
            schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
            jsonschema.Draft202012Validator(schema).validate(tampered)

    @unittest.skipUnless(NUMPY_RUNTIME_AVAILABLE, "accepted edit fixture requires NumPy")
    def test_active_edit_must_reuse_parked_obstacle_joint_name(self) -> None:
        tampered = _accepted_bundle_fixture()
        tampered["edits"]["log"][6]["joint_name"] = "renamed_active_joint"
        tampered.pop("bundle_content_sha256")
        tampered["bundle_content_sha256"] = content_hash(tampered)
        errors = validate_generated_source_artifact(tampered)
        self.assertNotIn("bundle_content_sha256 differs", errors)
        self.assertIn(
            "active edit joint name differs from the matching parked obstacle joint",
            errors,
        )


class GeneratedSourcePortabilityTest(unittest.TestCase):
    @unittest.skipUnless(NUMPY_RUNTIME_AVAILABLE, "integration call-order test requires NumPy")
    def test_integration_restore_precedes_step_in_exact_call_order(self) -> None:
        import numpy as np

        events = []
        expected = np.asarray([0.25], dtype=np.float64)
        spec = {
            "name": "mjSTATE_INTEGRATION",
            "value": 8191,
            "size": 1,
            "dtype": "float64",
        }

        class FakeMujoco:
            @staticmethod
            def mj_setState(_model, _data, _state, _spec):
                events.append("setState")

            @staticmethod
            def mj_forward(_model, _data):
                events.append("forward")

        class FakeEnv:
            def step_with_substep_callback(
                self,
                _action,
                _callback,
                *,
                update_observables,
                collect_observations,
            ):
                self.assert_flags = (update_observables, collect_observations)
                events.append("step")

        env = FakeEnv()

        def capture(_env, _spec):
            events.append("verify")
            return expected.copy()

        with mock.patch.object(
            _MODULE,
            "_assert_live_integration_spec",
            return_value=(FakeMujoco(), 8191, (object(), object())),
        ), mock.patch.object(_MODULE, "_integration_state", side_effect=capture):
            _MODULE._restore_integration_state(
                env,
                _MODULE._array_record(expected),
                name="test.integration",
                expected_spec=spec,
            )
            _MODULE._step_dummy(env, np.zeros(7, dtype=np.float64))
        self.assertEqual(events, ["setState", "forward", "setState", "verify", "step"])
        self.assertEqual(env.assert_flags, (False, False))

    @unittest.skipUnless(MUJOCO_RUNTIME_AVAILABLE, "MuJoCo runtime is allocation-backed")
    def test_full_integration_state_round_trips_byte_exactly(self) -> None:
        import mujoco
        import numpy as np

        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><joint/><geom size="0.1"/></body></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        env = SimpleNamespace(sim=SimpleNamespace(model=model, data=data))
        spec = _MODULE._integration_state_spec(env)
        self.assertEqual(spec["name"], "mjSTATE_INTEGRATION")
        self.assertEqual(spec["value"], 8191)
        data.time = 0.25
        data.qpos[:] = 0.125
        data.qvel[:] = -0.25
        data.qacc_warmstart[:] = 0.75
        data.qfrc_applied[:] = -0.5
        expected = _MODULE._integration_state(env, spec)
        record = _MODULE._array_record(expected)
        data.time = 9.0
        data.qpos[:] = 2.0
        data.qvel[:] = 3.0
        data.qacc_warmstart[:] = 4.0
        data.qfrc_applied[:] = 5.0
        _MODULE._restore_integration_state(
            env,
            record,
            name="test.integration",
            expected_spec=spec,
        )
        actual = _MODULE._integration_state(env, spec)
        self.assertEqual(actual.dtype, np.dtype("float64"))
        self.assertTrue(_MODULE._array_bytes_equal(actual, expected))

    def test_libero_asset_locator_resolves_concrete_package_below_namespace(self) -> None:
        package_root = ROOT / "safelibero/libero/libero"
        asset = package_root / "bddl_files/safelibero_spatial" / (
            "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
        )
        concrete_package = mock.Mock(__file__=str(package_root / "__init__.py"))
        with mock.patch.object(
            _MODULE.importlib, "import_module", return_value=concrete_package
        ) as import_module:
            locator = _MODULE._asset_locator(asset, ROOT / "not-the-repository-root")
            restored = _MODULE._locator_path(locator, ROOT / "not-the-repository-root")
        self.assertEqual(locator["kind"], "libero_package_relative")
        self.assertEqual(locator["path"], asset.relative_to(package_root).as_posix())
        self.assertEqual(restored, asset.resolve())
        self.assertEqual(import_module.call_args_list, [mock.call("libero.libero")] * 2)

    def test_concrete_package_without_file_fails_explicitly(self) -> None:
        namespace_package = mock.Mock(__file__=None)
        with mock.patch.object(
            _MODULE.importlib, "import_module", return_value=namespace_package
        ):
            with self.assertRaisesRegex(ValueError, "has no filesystem location"):
                _MODULE._package_root("libero")

    def test_robosuite_asset_falls_through_concrete_libero_and_round_trips(self) -> None:
        libero_root = ROOT / "safelibero/libero/libero"
        robosuite_root = ROOT / "fake-site-packages/robosuite"
        asset = robosuite_root / "models/assets/meshes/test.stl"
        packages = {
            "libero.libero": mock.Mock(__file__=str(libero_root / "__init__.py")),
            "robosuite": mock.Mock(__file__=str(robosuite_root / "__init__.py")),
        }
        with mock.patch.object(
            _MODULE.importlib, "import_module", side_effect=packages.__getitem__
        ) as import_module:
            locator = _MODULE._asset_locator(asset, ROOT / "not-the-repository-root")
            restored = _MODULE._locator_path(locator, ROOT / "not-the-repository-root")
        self.assertEqual(
            locator,
            {"kind": "robosuite_package_relative", "path": "models/assets/meshes/test.stl"},
        )
        self.assertEqual(restored, asset.resolve())
        self.assertEqual(
            import_module.call_args_list,
            [mock.call("libero.libero"), mock.call("robosuite"), mock.call("robosuite")],
        )

    def test_static_model_validator_never_raises_on_non_mapping_asset(self) -> None:
        malformed = {
            "format": "wrong",
            "xml": "<mujoco><asset /></mujoco>",
            "sha256": "0" * 64,
            "assets": [0],
            "assets_sha256": "0" * 64,
        }
        errors = _MODULE._validate_model_static(malformed)
        self.assertIn("model.format differs", errors)
        self.assertTrue(any("asset manifest item 0" in item for item in errors))

    def test_branch_identity_binds_every_scientific_component(self) -> None:
        fields = {
            "bddl_sha256": "1" * 64,
            "portable_model_xml_sha256": "2" * 64,
            "model_asset_manifest_sha256": "3" * 64,
            "final_flattened_state_sha256": "4" * 64,
            "observation_sha256": "5" * 64,
            "geometry_sha256": "6" * 64,
            "integration_state_spec_name": "mjSTATE_INTEGRATION",
            "integration_state_spec_value": 8191,
            "integration_state_size": 123,
            "integration_state_dtype": "float64",
            "pre_settle_integration_state_sha256": "7" * 64,
            "settle_integration_state_sequence_sha256": "8" * 64,
            "final_integration_state_sha256": "a" * 64,
        }
        identity = source_branch_identity(**fields)
        self.assertEqual(identity["payload"], fields)
        self.assertEqual(len(identity["sha256"]), 64)
        for key in (
            "bddl_sha256", "portable_model_xml_sha256", "model_asset_manifest_sha256",
            "final_flattened_state_sha256", "observation_sha256", "geometry_sha256",
            "pre_settle_integration_state_sha256",
            "settle_integration_state_sequence_sha256", "final_integration_state_sha256",
        ):
            mutated = dict(fields)
            mutated[key] = "b" * 64
            self.assertNotEqual(source_branch_identity(**mutated)["sha256"], identity["sha256"])
        for key in ("integration_state_spec_value", "integration_state_size"):
            mutated = dict(fields)
            mutated[key] += 1
            if key == "integration_state_spec_value":
                with self.assertRaisesRegex(ValueError, "spec value differs"):
                    source_branch_identity(**mutated)
            else:
                self.assertNotEqual(source_branch_identity(**mutated)["sha256"], identity["sha256"])

    def test_model_xml_assets_are_tokenized_hash_bound_and_rehydrated(self) -> None:
        asset = ROOT / "safelibero/libero/libero/bddl_files/safelibero_spatial" / (
            "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
        )
        xml = f'<mujoco><asset><texture name="fixture" file="{asset}" /></asset></mujoco>'
        frozen = freeze_model_xml(xml, repo_root=ROOT)
        self.assertIn("crfs-asset://0000", frozen["xml"])
        self.assertNotIn(str(asset), frozen["xml"])
        restored = rehydrate_model_xml(frozen, repo_root=ROOT)
        self.assertIn(str(asset), restored)
        corrupted = copy.deepcopy(frozen)
        corrupted["assets"][0]["sha256"] = "0" * 64
        from crfs_harness.artifacts import content_hash

        corrupted["assets_sha256"] = content_hash(_MODULE._semantic_asset_manifest(corrupted["assets"]))
        with self.assertRaisesRegex(ValueError, "asset bytes differ"):
            rehydrate_model_xml(corrupted, repo_root=ROOT)

    def test_semantic_asset_identity_excludes_runtime_absolute_path(self) -> None:
        asset = ROOT / "safelibero/libero/libero/bddl_files/safelibero_spatial" / (
            "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl"
        )
        xml = f'<mujoco><asset><texture name="fixture" file="{asset}" /></asset></mujoco>'
        frozen = freeze_model_xml(xml, repo_root=ROOT)
        changed = copy.deepcopy(frozen["assets"])
        changed[0]["original_path"] = "/different/runtime/root/fixture.bddl"
        from crfs_harness.artifacts import content_hash

        self.assertEqual(
            frozen["assets_sha256"], content_hash(_MODULE._semantic_asset_manifest(changed))
        )

    def test_real_generation_requires_array_allocation(self) -> None:
        environment = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
        with mock.patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Slurm-allocation-only"):
                allocation_provenance(ROOT)

    def test_allocation_provenance_freezes_baseline_osmesa_and_keeps_mig_identity(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "SLURM_JOB_ID": "123",
                "SLURM_ARRAY_JOB_ID": "123",
                "SLURM_ARRAY_TASK_ID": "0",
                "SLURM_JOB_PARTITION": "mig",
                "CUDA_VISIBLE_DEVICES": "MIG-test-uuid",
                "MUJOCO_GL": "osmesa",
                "PYOPENGL_PLATFORM": "osmesa",
            }
        )
        with mock.patch.dict(os.environ, environment, clear=True):
            provenance = allocation_provenance(ROOT)
        self.assertEqual(provenance["physics_device"], "cpu")
        self.assertEqual(provenance["render_device"], "cpu_osmesa")
        self.assertEqual(provenance["allocation_visible_gpu"], "MIG-test-uuid")
        self.assertEqual(provenance["mujoco_gl"], "osmesa")
        self.assertEqual(provenance["pyopengl_platform"], "osmesa")


class GeneratedSourceStructuralTest(unittest.TestCase):
    def test_core_replays_pre_settle_and_every_recorded_settle_state(self) -> None:
        source = (ROOT / "main/crfs_oracle/generated_source.py").read_text(encoding="utf-8")
        self.assertIn("named_joint_no_hard_coded_qpos_offset", source)
        self.assertIn("env.sim.data.set_joint_qpos(joint_name, qpos)", source)
        self.assertIn("env.env.reset()", source)
        self.assertNotIn("\n        env.reset()", source)
        self.assertIn("range(SETTLE_STEPS)", source)
        self.assertIn("states.settle_history", source)
        self.assertIn("states.integration.settle_history", source)
        self.assertIn("mjSTATE_INTEGRATION", source)
        self.assertIn("mujoco.mj_stateSize(model, spec)", source)
        self.assertIn("mujoco.mj_getState(model, data, state, spec)", source)
        self.assertIn("mujoco.mj_setState(model, data, expected, spec)", source)
        self.assertNotIn("_set_state(env, pre_settle)", source)
        self.assertIn("for index in (1, 2)", source)
        self.assertIn('f"gsrc-{branch_identity[\'sha256\'][:16]}"', source)
        self.assertNotIn("get_task_init_states", source)

    def test_cli_is_hash_bound_blocked_until_review_and_writes_atomically(self) -> None:
        source = (ROOT / "main/generate_task0_single_obstacle_source.py").read_text(encoding="utf-8")
        self.assertIn("--expected-config-sha256", source)
        self.assertIn('config.get("ready_to_run") is not True', source)
        self.assertIn('config.get("blocked_on") != []', source)
        self.assertIn("atomic_write_json(output, bundle)", source)
        self.assertIn('groups[args.group_index]["generation_request_id"]', source)

    def test_compiled_mjb_is_not_an_artifact_or_replay_gate(self) -> None:
        core = (ROOT / "main/crfs_oracle/generated_source.py").read_text(encoding="utf-8")
        schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
        artifact_contract = f"{core}\n{schema_text}"
        self.assertNotIn("mjb", artifact_contract.lower())
        self.assertNotIn("mj_saveModel", artifact_contract)
        self.assertNotIn("mj_sizeModel", artifact_contract)

        schema = json.loads(schema_text)
        self.assertEqual(
            schema["$defs"]["model"]["required"],
            ["format", "xml", "sha256", "assets", "assets_sha256"],
        )
        proof_fields = set(schema["$defs"]["proof"]["required"])
        self.assertTrue(
            {
                "model_xml_sha256",
                "pre_settle_state_sha256",
                "settle_state_sha256",
                "final_state_sha256",
                "integration_state_spec",
                "pre_settle_integration_state_sha256",
                "settle_integration_state_sha256",
                "settle_integration_state_sequence_sha256",
                "final_integration_state_sha256",
                "observation_sha256",
                "geometry_sha256",
                "exact_replay",
            }.issubset(proof_fields)
        )

    def test_schema_and_validator_bind_full_source_state_hash_and_source_job(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
        validator = (ROOT / "main/validate_task0_single_obstacle_source.py").read_text(encoding="utf-8")
        core = (ROOT / "main/crfs_oracle/generated_source.py").read_text(encoding="utf-8")
        self.assertIn("source_state_sha256", schema_text)
        self.assertIn("source_branch_sha256", schema_text)
        self.assertIn("mjSTATE_INTEGRATION", schema_text)
        self.assertIn("settle_integration_state_sequence_sha256", schema_text)
        self.assertEqual(
            schema["$defs"]["sourceBranch"]["properties"]["geometry"],
            {"$ref": "#/$defs/sourceGeometry"},
        )
        self.assertIn("source_state_sha256 differs between manifest and bundle", core)
        self.assertIn("source_branch_sha256 differs between manifest and bundle", core)
        self.assertIn("environment_seed differs between manifest and bundle reset seed", core)
        self.assertIn("--expected-source-slurm-job-id", validator)
        self.assertIn("--expected-source-slurm-array-job-id", validator)
        self.assertIn("--expected-source-slurm-array-task-id", validator)


if __name__ == "__main__":
    unittest.main()
