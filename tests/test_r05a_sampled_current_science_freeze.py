from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / "r05a_sampled_current_canary_apparatus.json"
SCHEMA_PATH = ROOT / "schemas" / "r05a-sampled-current-canary-envelope.schema.json"

EXPECTED_FROZEN_FILES = {
    "configs/experiments/r05a_inverse_flow_canary.json": (
        "c31401867f3cdce2b3f443ad021c39dfb812f573b570e1e7434e1f149f79abfb"
    ),
    "manifests/r05a_inverse_flow_teacher_smoke.jsonl": (
        "bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633"
    ),
    "docs/decisions/0028-pivot-to-inverse-flow-transport.md": (
        "d2a00b1b049e92bb1ec8f11d60fa447e5e8bd5109cb8cf3f3fbb08f89498656f"
    ),
    "schemas/r05a-inverse-flow-canary.schema.json": (
        "e1681f865f81f2986945d10fb14073fe4cd9e78cbc5026f6749ab321b9cfb7a7"
    ),
    "configs/experiments/r02_oracle_flow.json": (
        "c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e"
    ),
    "evidence/r03/r03-summary.json": (
        "dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e"
    ),
    "main/crfs_oracle/r05a_canary.py": (
        "9d4402ccb92835bc1af2a3e04fe54c0b94839f41beb2971022216f73ec67fede"
    ),
    "main/run_crfs_r05a_canary.py": (
        "247bd20e48ffe2228d2633d2f667a2791371163cca1c43ad7d278f4fbfd7439a"
    ),
    "openpi/src/openpi/models_pytorch/crfs_inverse_control.py": (
        "965082822466774e0a86eeb6f2d178c9e5f3d6d02bca5090c4e1fcc77bc52aa8"
    ),
    "openpi/src/openpi/models_pytorch/pi0_pytorch.py": (
        "80366dcc7b2ddc598717d4c71c0e68e46312a1e5ffd3fc04479d5599433f4c55"
    ),
    "openpi/src/openpi/policies/policy.py": (
        "d16767ff2073d5c177cdfcc06dc05dcbf7cdb9ef2a2a0150023f7953b03508b9"
    ),
}

LEGACY_MEMORY_ONLY_ERRORS = [
    "host cgroup path provenance is missing",
    "memory record is missing",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk_dicts(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_dicts(item)


class R05ASampledCurrentScienceFreezeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def test_apparatus_config_is_either_fail_closed_or_exactly_released(self) -> None:
        self.assertEqual(self.config["schema_version"], "1.0")
        self.assertEqual(
            self.config["scientific_role"],
            "telemetry_only_wrapper_around_byte_frozen_ift00a_science",
        )
        self.assertIs(type(self.config["ready_to_run"]), bool)
        if self.config["ready_to_run"] is False:
            self.assertGreater(len(self.config["blocked_on"]), 0)
            self.assertNotIn("execution_release", self.config)
            for mapping in _walk_dicts(self.config):
                self.assertNotIn("run_id", mapping)
        else:
            self.assertEqual(self.config["blocked_on"], [])
            release = self.config["execution_release"]
            self.assertEqual(
                set(release),
                {
                    "schema_version",
                    "artifact_role",
                    "decision_artifact",
                    "accepted_implementation_commit",
                    "run_id",
                    "single_submission",
                    "source_host",
                    "resources",
                    "release_only_parent_required",
                    "allowed_release_diff_paths",
                    "automatic_resubmission_allowed",
                    "automatic_next_experiment_allowed",
                },
            )
            self.assertRegex(release["accepted_implementation_commit"], r"^[0-9a-f]{40}$")
            self.assertRegex(release["run_id"], r"^[A-Za-z0-9._-]+$")
            self.assertIs(release["single_submission"], True)
            self.assertEqual(release["source_host"], "worker-1")
            self.assertIs(release["automatic_resubmission_allowed"], False)
            self.assertIs(release["automatic_next_experiment_allowed"], False)
        self.assertEqual(
            self.config["envelope_schema_sha256"], _sha256(SCHEMA_PATH)
        )

    def test_every_local_science_input_remains_byte_frozen(self) -> None:
        for relative, expected in EXPECTED_FROZEN_FILES.items():
            with self.subTest(path=relative):
                path = ROOT / relative
                self.assertTrue(path.is_file(), relative)
                self.assertEqual(_sha256(path), expected)

        frozen = self.config["frozen_science_bindings"]
        self.assertEqual(
            frozen["scientific_config"]["normalized_projection_sha256"],
            "9e2ff74cda8ac3b5d4098942a82cc3352cebea3f4b8e1fc1d326c2dc24d2887d",
        )
        self.assertEqual(
            frozen["source_r02"]["sha256"],
            "055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593",
        )
        self.assertEqual(
            frozen["source_r03_summary"]["ordered_result_set_digest"],
            "fa79a132f2fecbedab5917111ef71f022e52c933d8e535aaca118add5f5b7895",
        )
        self.assertEqual(
            frozen["checkpoint"]["sha256"],
            "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",
        )
        self.assertEqual(
            frozen["normalization_asset"]["sha256"],
            "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
        )
        self.assertEqual(
            frozen["baseline_revision"],
            "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b",
        )
        registered_live = {
            item["path"]: item["sha256"] for item in frozen["live_science_files"]
        }
        for relative, expected in EXPECTED_FROZEN_FILES.items():
            if relative in registered_live:
                self.assertEqual(registered_live[relative], expected)

    def test_case_solver_target_budget_and_resources_are_unchanged(self) -> None:
        case = self.config["frozen_case"]
        self.assertEqual(
            case,
            {
                "case_index": 0,
                "case_id": "crfs-1069f29a8d76463a",
                "group_id": "safelibero_spatial:II:0:46",
                "environment_seed": 924805038,
                "policy_seed": 1179198633,
                "source_host": "worker-1",
                "task_suite": "safelibero_spatial",
                "safety_level": "II",
                "task_index": 0,
                "episode_index": 46,
            },
        )
        science = self.config["frozen_scientific_procedure"]
        self.assertEqual(
            science["target_definition"],
            "fresh_frozen_normalized_final_plus_immutable_r02_delta_star_model_first_five_xyz_only",
        )
        self.assertEqual(
            science["normalization_rule"],
            "physical_displacement_to_model_uses_checkpoint_scale_only_never_mean",
        )
        self.assertEqual(science["source_reported_budget_float64"], 3.6398398429065115)
        self.assertEqual(science["source_budget_float32"], 3.6398398876190186)
        self.assertEqual(
            science["solver"],
            {
                "algorithm": "deterministic_constrained_direct_shooting",
                "max_iterations": 128,
                "learning_rate": 0.02,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-08,
                "xyz_max_abs_tolerance": 0.01,
                "xyz_rms_tolerance": 0.005,
                "full_max_abs_tolerance": 0.05,
                "full_rms_tolerance": 0.015,
                "constraint_slack_ulps": 8,
                "stop_on_first_feasible": False,
            },
        )
        control = science["control_contract"]
        self.assertEqual(control["active_steps"], [5, 6, 7, 8, 9])
        self.assertEqual(control["inactive_steps"], [0, 1, 2, 3, 4])
        self.assertEqual(control["mask"], "first_five_xyz_only")
        self.assertEqual(control["clipping"], "forbidden")
        self.assertEqual(control["terminal_overwrite"], "forbidden")
        self.assertEqual(control["direct_action_delta_after_sampling"], "forbidden")
        self.assertEqual(science["teacher_generated_action_steps_executed"], 0)
        self.assertIs(science["simulator_efficacy_evaluated"], False)

        resources = self.config["resource_contract"]
        self.assertEqual(
            (
                resources["partition"],
                resources["source_host"],
                resources["gpus"],
                resources["cpus_per_task"],
                resources["host_memory_mib"],
                resources["time_limit"],
                resources["array"],
                resources["requeue"],
            ),
            ("main", "worker-1", 1, 8, 65536, "02:00:00", "0-0%1", False),
        )
        self.assertEqual(resources["validator_dependency"], "afterany")
        self.assertEqual(resources["validator_gpus"], 0)

        allocation_tests = self.config["allocation_test_contract"]
        self.assertEqual(
            allocation_tests["registry_sha256"],
            "89bf8a509dafefcce2bd18cd6cf8e0233a8728165b6f033adc7f39a45c7e1432",
        )
        self.assertEqual(
            allocation_tests["expected_counts"],
            {
                "test_inverse_flow_control.py": 17,
                "test_inverse_flow_sampler.py": 8,
                "test_inverse_flow_policy.py": 10,
                "test_r05a_canary.py": 12,
            },
        )
        self.assertIs(allocation_tests["zero_skips_required"], True)

    def test_sampled_current_is_a_lower_bound_and_native_peak_is_separate(self) -> None:
        telemetry = self.config["host_telemetry_contract"]
        self.assertEqual(telemetry["sample_interval_requested_ms"], 100)
        self.assertEqual(telemetry["maximum_adjacent_gap_ns"], 500_000_000)
        self.assertEqual(telemetry["scope"], "exact_job_boundary_only")
        self.assertIs(telemetry["shared_ancestor_read"], False)
        self.assertIs(telemetry["sibling_or_descendant_enumeration"], False)
        self.assertIs(telemetry["memory_localevents_allowed"], False)
        self.assertEqual(
            telemetry["high_water_field"],
            "host_cgroup_sampled_current_high_water_bytes",
        )
        self.assertIs(telemetry["high_water_is_lower_bound_not_peak"], True)
        self.assertEqual(
            telemetry["hierarchical_event_deltas_required_zero"],
            ["max", "oom", "oom_kill"],
        )
        self.assertIn("nullable_positive_value", telemetry["native_memory_peak"])
        self.assertEqual(
            telemetry["required_lifecycle_markers"],
            [
                "monitor_ready",
                "policy_server_launch",
                "policy_server_cleanup_complete",
                "gpu_monitor_cleanup_complete",
                "workload_cleanup_complete",
                "monitor_stop_observed",
            ],
        )

        schema_source = SCHEMA_PATH.read_text(encoding="utf-8")
        self.assertNotIn("host_cgroup_peak_bytes", schema_source)
        host = self.schema["$defs"]["hostTelemetry"]
        properties = host["properties"]
        self.assertIs(
            properties["host_cgroup_sampled_current_is_lower_bound_not_peak"]["const"],
            True,
        )
        self.assertEqual(properties["sample_interval_requested_ms"]["const"], 100)
        self.assertEqual(
            properties["maximum_adjacent_gap_ns"]["maximum"], 500_000_000
        )
        native = properties["native_memory_peak_bytes"]["oneOf"]
        self.assertEqual(native[0], {"type": "null"})
        self.assertEqual(native[1]["type"], "integer")
        self.assertEqual(native[1]["minimum"], 1)
        self.assertEqual(len(host["allOf"]), 2)

    def test_envelope_schema_internal_contract_is_self_consistent(self) -> None:
        definitions = self.schema["$defs"]
        for mapping in _walk_dicts(self.schema):
            reference = mapping.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                self.assertIn(reference.removeprefix("#/$defs/"), definitions)
            if mapping.get("type") == "object" and "required" in mapping:
                self.assertIn("properties", mapping)
                self.assertTrue(
                    set(mapping["required"]).issubset(mapping["properties"]),
                    set(mapping["required"]) - set(mapping["properties"]),
                )
        self.assertIs(self.schema["additionalProperties"], False)
        self.assertEqual(self.schema["properties"]["schema_version"]["const"], "2.0")
        self.assertEqual(
            self.schema["properties"]["artifact_type"]["const"],
            "r05a_sampled_current_canary_envelope",
        )

    def test_legacy_semantic_validator_exempts_only_two_memory_errors(self) -> None:
        semantic = self.config["semantic_validation_contract"]
        self.assertEqual(
            semantic["validator"],
            "crfs_oracle.r05a_canary.validate_r05a_canary_result",
        )
        self.assertEqual(
            semantic["validator_module_sha256"],
            EXPECTED_FROZEN_FILES["main/crfs_oracle/r05a_canary.py"],
        )
        self.assertIs(semantic["fabricate_legacy_host_memory"], False)
        self.assertEqual(
            semantic["allowed_legacy_errors_exact_order"],
            LEGACY_MEMORY_ONLY_ERRORS,
        )
        self.assertEqual(semantic["allowlist_match"], "exact_list_equality_never_subset")
        self.assertIs(
            semantic["pre_memory_apparatus_summary_must_match_registered_shape_and_values"],
            True,
        )
        self.assertEqual(
            semantic["transient_legacy_view_apparatus_adaptation"],
            {
                "memory_pending": False,
                "memory_passed": True,
                "passed": "copy_exact_boolean_from_passed_before_memory_finalization",
            },
        )
        self.assertIs(
            semantic["transient_adaptation_mutates_persisted_science_payload"], False
        )
        self.assertEqual(semantic["nonmemory_error_count"], 0)

        schema_semantic = self.schema["properties"]["semantic_validation"]
        semantic_properties = schema_semantic["properties"]
        errors = semantic_properties["returned_legacy_errors"]
        self.assertEqual(errors["minItems"], 2)
        self.assertEqual(errors["maxItems"], 2)
        self.assertEqual(
            [item["const"] for item in errors["prefixItems"]],
            LEGACY_MEMORY_ONLY_ERRORS,
        )
        self.assertIs(errors["items"], False)
        self.assertIs(
            semantic_properties["legacy_view_host_memory_fabricated"]["const"], False
        )
        self.assertIs(
            semantic_properties["legacy_view_apparatus_memory_fields_adapted"]["const"],
            True,
        )
        self.assertIs(
            semantic_properties[
                "legacy_view_apparatus_passed_copied_from_pre_memory_result"
            ]["const"],
            True,
        )

    def test_cpu_validator_is_the_only_publisher_and_no_next_gate_is_authorized(self) -> None:
        publication = self.config["publication_contract"]
        self.assertIs(publication["gpu_allocation_may_write_result_candidate"], False)
        self.assertIs(publication["gpu_allocation_may_publish_results_json"], False)
        self.assertIs(
            publication[
                "cpu_afterany_validator_is_sole_candidate_creator_and_results_json_publisher"
            ],
            True,
        )
        self.assertEqual(publication["cpu_validator_gpu_count"], 0)
        self.assertEqual(publication["source_job_required_exit_code"], "0:0")
        self.assertIn("expected_pre_cpu_source_contract_receipt_sha256", publication["external_receipt_binding"])
        self.assertIs(publication["cpu_validator_creates_atomic_hidden_candidate"], True)
        self.assertIs(publication["cpu_validator_validates_candidate_before_publication"], True)
        self.assertIs(publication["cpu_validator_revalidates_candidate_before_publication"], True)
        self.assertEqual(publication["raw_mutation_after_candidate_creation"], "reject")
        self.assertIs(publication["failure_or_nonzero_source_job_leaves_results_json"], False)

        interpretation = self.config["interpretation_contract"]
        for key in (
            "simulator_efficacy_claim_allowed",
            "student_learning_claim_allowed",
            "probe_training_authorized",
            "ift01_authorized",
            "automatic_next_gate_allowed",
        ):
            self.assertIs(interpretation[key], False, key)
        forbidden = set(self.config["forbidden_changes"])
        self.assertIn("direct_action_delta_after_sampling", forbidden)
        self.assertIn("ift01_launch", forbidden)
        self.assertIn("probe_or_mlp_training", forbidden)

        schema_publication = self.schema["properties"]["publication"]["properties"]
        self.assertNotIn("hidden_candidate_sha256", schema_publication)
        self.assertEqual(schema_publication["cpu_validator_gpu_count"]["const"], 0)
        self.assertIs(schema_publication["gpu_job_created_result_candidate"]["const"], False)
        self.assertIs(schema_publication["gpu_job_published_results_json"]["const"], False)
        self.assertIs(schema_publication["candidate_created_by_cpu_validator"]["const"], True)
        self.assertIs(schema_publication["candidate_validated_before_publication"]["const"], True)
        self.assertIs(schema_publication["candidate_revalidated_before_publication"]["const"], True)
        self.assertIs(schema_publication["cpu_validator_is_sole_publisher"]["const"], True)
        self.assertEqual(schema_publication["cpu_validator_partition"]["const"], "main")
        self.assertEqual(schema_publication["cpu_validator_cpus_per_task"]["const"], 2)
        self.assertEqual(schema_publication["cpu_validator_host_memory_mib"]["const"], 8192)
        self.assertEqual(schema_publication["cpu_validator_time_limit"]["const"], "00:15:00")
        schema_tests = self.schema["properties"]["allocation_tests"]["properties"]
        self.assertEqual(
            schema_tests["registry_sha256"]["const"],
            "89bf8a509dafefcce2bd18cd6cf8e0233a8728165b6f033adc7f39a45c7e1432",
        )
        zero_deltas = self.schema["$defs"]["requiredZeroMemoryEventDeltas"]
        for key in ("max", "oom", "oom_kill"):
            self.assertEqual(zero_deltas["properties"][key]["const"], 0)
        self.assertEqual(self.schema["properties"]["schema_version"]["const"], "2.0")
        self.assertIs(self.schema["properties"]["scientific_claim_allowed"]["const"], False)
        self.assertIs(self.schema["properties"]["probe_training_authorized"]["const"], False)
        self.assertIs(self.schema["properties"]["ift01_authorized"]["const"], False)


if __name__ == "__main__":
    unittest.main()
