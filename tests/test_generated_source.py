from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

try:
    import jsonschema
except ModuleNotFoundError:
    jsonschema = None


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "main", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crfs_harness.artifacts import file_sha256  # noqa: E402
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


class GeneratedSourcePortabilityTest(unittest.TestCase):
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
        }
        identity = source_branch_identity(**fields)
        self.assertEqual(identity["payload"], fields)
        self.assertEqual(len(identity["sha256"]), 64)
        for key in fields:
            mutated = dict(fields)
            mutated[key] = "a" * 64
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

    def test_schema_and_validator_bind_full_source_state_hash_and_source_job(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
        validator = (ROOT / "main/validate_task0_single_obstacle_source.py").read_text(encoding="utf-8")
        core = (ROOT / "main/crfs_oracle/generated_source.py").read_text(encoding="utf-8")
        self.assertIn("source_state_sha256", schema_text)
        self.assertIn("source_branch_sha256", schema_text)
        self.assertIn("source_state_sha256 differs between manifest and bundle", core)
        self.assertIn("source_branch_sha256 differs between manifest and bundle", core)
        self.assertIn("environment_seed differs between manifest and bundle reset seed", core)
        self.assertIn("--expected-source-slurm-job-id", validator)
        self.assertIn("--expected-source-slurm-array-job-id", validator)
        self.assertIn("--expected-source-slurm-array-task-id", validator)


if __name__ == "__main__":
    unittest.main()
