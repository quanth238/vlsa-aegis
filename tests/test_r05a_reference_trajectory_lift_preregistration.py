from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs" / "experiments" / "r05a_reference_trajectory_lift_canary.json"
)
ADR_0059_PATH = (
    ROOT
    / "docs"
    / "decisions"
    / "0059-interpret-af00a-population-and-test-reference-lift.md"
)
ADR_0060_PATH = (
    ROOT
    / "docs"
    / "decisions"
    / "0060-preregister-reference-trajectory-lift-canary.md"
)
DIAGNOSTIC_PATH = (
    ROOT / "evidence" / "r05a" / "af00a-sealed-population-diagnostic.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _squash_whitespace(value: str) -> str:
    return " ".join(value.split())


def _assert_exact_contiguous_ledger(
    testcase: unittest.TestCase,
    phases: list[dict[str, object]],
    expected: list[tuple[str, int, int]],
    request_count: int,
) -> None:
    testcase.assertEqual(
        phases,
        [
            {"name": name, "start": start, "stop_inclusive": stop}
            for name, start, stop in expected
        ],
    )
    expanded: list[int] = []
    for item in phases:
        expanded.extend(
            range(int(item["start"]), int(item["stop_inclusive"]) + 1)
        )
    testcase.assertEqual(expanded, list(range(request_count)))


class ReferenceTrajectoryLiftPreregistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.adr_0059 = ADR_0059_PATH.read_text(encoding="utf-8")
        cls.adr_0060 = ADR_0060_PATH.read_text(encoding="utf-8")
        cls.diagnostic = json.loads(DIAGNOSTIC_PATH.read_text(encoding="utf-8"))

    def test_config_is_fail_closed_or_one_exact_release(self) -> None:
        self.assertEqual(self.config["experiment_identity"], "TRL-00A")
        release = self.config["execution_release"]
        preregistration = self.config["preregistration"]
        self.assertTrue(preregistration["implementation_authorized"])
        if self.config["ready_to_run"] is False:
            self.assertEqual(
                self.config["config_status"], "draft_preregistered_not_released"
            )
            self.assertEqual(
                self.config["blocked_on"],
                [
                    "reference_lift_implementation_not_complete",
                    "independent_scientific_code_hpc_review_not_complete",
                    "exact_execution_release_not_authorized",
                ],
            )
            self.assertIsNone(release)
            self.assertFalse(preregistration["h100_submission_authorized"])
        else:
            self.assertIs(self.config["ready_to_run"], True)
            self.assertEqual(
                self.config["config_status"], "released_exact_single_canary"
            )
            self.assertEqual(self.config["blocked_on"], [])
            self.assertTrue(preregistration["h100_submission_authorized"])
            self.assertIsInstance(release, dict)
            self.assertEqual(
                release["artifact_role"],
                "r05a_reference_trajectory_lift_canary_execution_release",
            )
            self.assertRegex(
                release["accepted_implementation_commit"], r"^[0-9a-f]{40}$"
            )
            self.assertRegex(release["run_id"], r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
            self.assertTrue(release["single_submission"])
            self.assertEqual(release["source_host"], "worker-1")
            self.assertTrue(release["release_only_parent_required"])
            self.assertRegex(
                release["decision_artifact"],
                r"^docs/decisions/[0-9]{4}-release-reference-trajectory-lift-canary\.md$",
            )
            self.assertEqual(
                release["allowed_release_diff_paths"],
                [
                    "configs/experiments/r05a_reference_trajectory_lift_canary.json",
                    release["decision_artifact"],
                ],
            )
            self.assertFalse(release["automatic_resubmission_allowed"])
            self.assertFalse(release["automatic_next_experiment_allowed"])
            self.assertEqual(
                release["resources"],
                {
                    "partition": "main",
                    "account": "normal",
                    "qos": "normal",
                    "gpus": 1,
                    "cpus_per_task": 8,
                    "host_memory_mib": 65536,
                    "time_limit": "00:30:00",
                    "array": "0-0%1",
                    "requeue": False,
                    "validator_partition": "main",
                    "validator_account": "normal",
                    "validator_qos": "normal",
                    "validator_cpus": 2,
                    "validator_host_memory_mib": 8192,
                    "validator_time_limit": "00:15:00",
                    "validator_gpus": 0,
                    "validator_dependency": "afterany",
                },
            )
        self.assertFalse(preregistration["automatic_next_gate_authorized"])
        self.assertTrue(self.config["new_experiment_not_retry"])
        self.assertTrue(self.config["historical_run_id_reuse_forbidden"])
        decision = _squash_whitespace(self.adr_0060)
        self.assertIn("does not authorize an H100 submission", decision)
        self.assertIn("separate direct-child execution release", decision)

    def test_source_case_target_and_authority_are_exactly_frozen(self) -> None:
        source = self.config["frozen_source_bindings"]
        self.assertEqual(
            source["manifest"],
            {
                "path": "manifests/r05a_inverse_flow_teacher_smoke.jsonl",
                "sha256": "bdb8ccbba01ebf500e0f1bd0fe4a4043054f922a273f9e90eb3860cfe753a633",
                "row_index_zero_based": 0,
            },
        )
        self.assertEqual(
            source["source_r02"],
            {
                "run_id": "r02-oracle-flow-population-20260714a",
                "path": "/mnt/data/quanth/experiments/crfs-oracle/r02-oracle-flow-population-20260714a/crfs-1069f29a8d76463a/r02-paired.json",
                "sha256": "055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593",
            },
        )
        self.assertEqual(
            source["source_r02_config"],
            {
                "path": "configs/experiments/r02_oracle_flow.json",
                "sha256": "c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e",
            },
        )
        for name in ("manifest", "source_r02_config"):
            binding = source[name]
            self.assertEqual(_sha256(ROOT / binding["path"]), binding["sha256"])
        self.assertEqual(
            source["checkpoint_model"]["sha256"],
            "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",
        )
        self.assertEqual(
            source["checkpoint_config"]["sha256"],
            "5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a",
        )
        self.assertEqual(
            source["normalization_asset"]["sha256"],
            "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
        )
        self.assertEqual(
            source["baseline_revision"], "57b1aef306f212aea3574b0a3b64aa1a3d8f5e4b"
        )
        af00a = source["af00a_source_run"]
        self.assertEqual(af00a["run_id"], "r05a-actual-forward-cem-canary-20260716c")
        self.assertEqual(
            af00a["tensor_archive_path"],
            "/mnt/data/quanth/experiments/crfs-oracle/"
            "r05a-actual-forward-cem-canary-20260716c/"
            "crfs-1069f29a8d76463a/af00a-tensors.npz",
        )
        self.assertEqual(
            af00a["results_json_path"],
            "/mnt/data/quanth/experiments/crfs-oracle/"
            "r05a-actual-forward-cem-canary-20260716c/results.json",
        )
        self.assertEqual(
            af00a["tensor_archive_sha256"],
            "ada53973653ba7c21dab33cdbc4b4498c7b2b1840f2fc284bd3d3c76baf1d383",
        )
        self.assertEqual(
            af00a["results_json_sha256"],
            "507f25bc381b7bfeaa30abe3a7df970681f3b175f39221034c6e768e72dae914",
        )
        self.assertFalse(af00a["use_as_control_input"])
        self.assertTrue(af00a["historical_arm_a_reproduction_only"])

        self.assertEqual(
            self.config["frozen_case"],
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
        target = self.config["target_contract"]
        self.assertEqual(target["source_reported_budget_float64"], 3.6398398429065115)
        self.assertEqual(target["source_budget_float32"], 3.6398398876190186)
        self.assertEqual(
            target["normalization"],
            "physical_displacement_uses_checkpoint_scale_only_never_subtracts_normalization_mean",
        )
        self.assertEqual(
            target["delta_definition"],
            "elementwise_float32_cast_of_immutable_r02_delta",
        )
        self.assertFalse(target["af00a_population_fit_may_construct_arm"])
        self.assertFalse(target["new_planner_or_geometry_query"])

        flow = self.config["flow_contract"]
        self.assertEqual(flow["sampler_steps"], 10)
        self.assertEqual(flow["dt_float32"], -0.1)
        self.assertEqual(flow["intervention_step"], 5)
        self.assertEqual(flow["active_flow_steps"], [5, 6, 7, 8, 9])
        self.assertEqual(flow["control_mask"], "first_five_xyz_only")
        self.assertEqual(flow["reference_active_state_shape"], [6, 10, 32])
        self.assertEqual(flow["delta_shape"], [10, 32])
        lift = self.config["reference_lift_contract"]
        self.assertEqual(lift["projection_radius_float64"], 0.7279679775238037)
        self.assertEqual(lift["budget_validation"]["budget_float32"], 3.6398398876190186)
        self.assertEqual(lift["budget_validation"]["radius_float32"], 0.7279679775238037)

    def test_reference_equation_preserves_signed_zero_and_uses_each_arm_state(self) -> None:
        flow = self.config["flow_contract"]
        self.assertEqual(
            flow["alpha_float32_definition"],
            "alpha_5=float32(+0.0); alpha_j=float32(float32(j-5)/float32(5)) for j=6..10",
        )
        self.assertEqual(
            flow["base_proposal_order"],
            "base_next32=float32(x32+float32(dt32*v_base32))",
        )
        self.assertEqual(
            flow["transport_order"],
            "c32_then_u32=float32(c32/dt32)_then_authoritative_executed_c32=float32(dt32*u32)",
        )
        lift = self.config["reference_lift_contract"]
        self.assertEqual(lift["mode"], "reference_trajectory_lift")
        self.assertEqual(
            lift["reference_construction"],
            "copy_xbar_then_update_only_masked_coordinates_whose_float32_weighted_delta_is_nonzero",
        )
        self.assertEqual(
            lift["reference_states_input_semantics"],
            "six_fully_constructed_desired_ref_5_through_ref_10_states_not_unshifted_xbar",
        )
        self.assertFalse(lift["sampler_adds_delta_to_supplied_reference"])
        self.assertTrue(
            lift["runner_and_cpu_validator_reconstruct_reference_from_fresh_zero_trace"]
        )
        self.assertTrue(lift["outside_mask_and_unchanged_signed_zero_bytes_preserved"])
        self.assertEqual(
            lift["raw_increment"],
            "positive_zero_outside_mask_and_float32(reference_next32-base_next32)_inside_mask",
        )
        self.assertTrue(lift["recomputed_from_each_arms_current_state"])
        for equation in (
            "alpha_5 = float32(+0.0)",
            "alpha_j = float32(float32(j - 5) / float32(5))",
            "weighted_j32 = float32(alpha_j32 * Delta32)",
            "ref_j32 = copy(xbar_j32)",
            "weighted_j32 != 0",
            "ref_5 == xbar_5",
            "ref_10[mask] == target32[mask]",
            "must not add `Delta32` a second time",
            "every unchanged signed-zero byte",
        ):
            with self.subTest(equation=equation):
                self.assertIn(equation, self.adr_0060)

    def test_projection_uses_strict_float64_reduction_and_exact_noop_rule(self) -> None:
        lift = self.config["reference_lift_contract"]
        self.assertEqual(lift["projection_norm_dtype"], "float64")
        self.assertEqual(
            lift["projection_reduction_order"],
            "cast_each_of_15_c_order_coordinates_to_float64_then_sequential_sum64=float64(sum64+float64(x64*x64))_then_one_float64_sqrt",
        )
        self.assertEqual(
            lift["projection_zero_rule"],
            "preserve_raw_float64_block_including_signed_zero_if_norm_is_zero",
        )
        self.assertEqual(
            lift["projection_scale_rule"],
            "min(1,float64_radius_div_float64_norm)",
        )
        self.assertFalse(lift["iterative_rounding_correction_allowed"])
        self.assertFalse(lift["remaining_budget_redistribution_allowed"])
        validation = lift["budget_validation"]
        self.assertEqual(validation["norm_dtype"], "float32_on_authoritative_executed_increments")
        self.assertEqual(validation["path_sum_dtype"], "float32")
        self.assertEqual(validation["slack_float32_ulps"], 8)
        decision = _squash_whitespace(self.adr_0060)
        for statement in (
            "strict C order",
            "sum64 = float64(sum64 + float64(x64 * x64))",
            "projected64 = raw64 if n64 <= float64(R32)",
            "The first branch includes `n64 == 0`",
            "preserves the raw float64 signed-zero bytes",
            "exact no-op condition",
            "slack-bearing executed- schedule validator alone does not imply projection was a byte no-op",
        ):
            with self.subTest(statement=statement):
                self.assertIn(statement, decision)

    def test_complete_finite_ledger_is_exact_and_contiguous(self) -> None:
        ledger = self.config["request_ledger"]
        self.assertTrue(ledger["global_policy_request_indices_zero_based"])
        self.assertEqual(ledger["complete_finite_exact_policy_request_count"], 18)
        _assert_exact_contiguous_ledger(
            self,
            ledger["complete_ordered_phases"],
            [
                ("compiled_frozen_pre", 0, 0),
                ("eager_source_trace_pre", 1, 1),
                ("eager_normalized_final_pre", 2, 2),
                ("zero_schedule_pre", 3, 3),
                ("arm_a_equal_split", 4, 5),
                ("budgeted_lift_generation", 6, 7),
                ("budgeted_lift_replay", 8, 9),
                ("raw_lift_generation", 10, 11),
                ("raw_lift_replay", 12, 13),
                ("zero_schedule_post", 14, 14),
                ("eager_normalized_final_post", 15, 15),
                ("eager_source_trace_post", 16, 16),
                ("compiled_frozen_post", 17, 17),
            ],
            18,
        )
        self.assertTrue(ledger["duplicate_generation_requires_exact_non_timing_scientific_output"])
        self.assertTrue(ledger["ordinary_replay_requires_exact_shared_recurrence_final_and_actions"])
        self.assertTrue(ledger["missing_extra_reordered_or_deduplicated_request_invalid"])
        self.assertIn("exactly 18 ordinary policy requests", self.adr_0060)
        self.assertIn("There is no shorter accepted ledger", self.adr_0060)

    def test_outcome_precedence_and_every_rule_are_frozen(self) -> None:
        outcomes = self.config["outcome_contract"]
        precedence = [
            "apparatus_inconclusive",
            "same_budget_lift_pass",
            "canonical_budget_bottleneck",
            "canonical_mask_coupling",
            "canonical_lift_negative",
        ]
        self.assertEqual(outcomes["precedence"], precedence)
        self.assertEqual(
            {name: outcomes[name] for name in precedence},
            {
                "apparatus_inconclusive": "any_source_pairing_arm_a_reference_dtype_equation_determinism_replay_constraint_test_telemetry_schema_or_publication_failure",
                "same_budget_lift_pass": "changed_budgeted_or_exact_B32_valid_raw_ordinary_witness_passes_all_four_gates",
                "canonical_budget_bottleneck": "budgeted_misses_raw_full_passes_and_exact_B32_validator_rejects_raw",
                "canonical_mask_coupling": "no_prior_full_pass_outcome_and_at_least_one_valid_finite_budgeted_or_raw_arm_passes_xyz_but_fails_full",
                "canonical_lift_negative": "valid_budgeted_and_valid_finite_raw_both_miss_xyz_and_no_prior_outcome_applies",
            },
        )
        self.assertEqual(
            set(outcomes),
            {
                "precedence",
                *precedence,
                "infeasibility_claim_allowed",
                "raw_authority_is_minimum_required_claim_allowed",
                "automatic_next_experiment_allowed",
            },
        )
        self.assertFalse(outcomes["infeasibility_claim_allowed"])
        self.assertFalse(outcomes["raw_authority_is_minimum_required_claim_allowed"])
        self.assertFalse(outcomes["automatic_next_experiment_allowed"])
        positions = [self.adr_0060.index(f"**`{name}`:**") for name in precedence[1:]]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("has highest\nprecedence and yields `apparatus_inconclusive`", self.adr_0060)

    def test_artifacts_and_provisional_resources_are_exact(self) -> None:
        self.assertEqual(
            self.config["artifact_contract"],
            {
                "raw_payload_name": "trl00a-raw-payload.json",
                "tensor_archive_name": "trl00a-tensors.npz",
                "request_ledger_name": "request-ledger.json",
                "published_result_name": "results.json",
                "publication_receipt_name": "cpu-afterany-validation.json",
                "gpu_may_publish_results_json": False,
                "cpu_afterany_is_sole_publisher": True,
                "npz_allow_pickle": False,
                "envelope_schema_path": "schemas/r05a-reference-trajectory-lift-canary-envelope.schema.json",
                "envelope_schema_sha256": "44e7fae723fd3e20585f2d618c2a8839a2b7fd940fdcb9d541bc793335bc162e",
                "publication_receipt_schema_path": "schemas/r05a-reference-trajectory-lift-canary-publication-receipt.schema.json",
                "publication_receipt_schema_sha256": "90d9dfe9fd75dc67e8ba23a2bad5f0049c6d64ccdf6b0ab524729f6bb36a6673",
            },
        )
        artifacts = self.config["artifact_contract"]
        for path_key, hash_key in (
            ("envelope_schema_path", "envelope_schema_sha256"),
            ("publication_receipt_schema_path", "publication_receipt_schema_sha256"),
        ):
            self.assertEqual(
                _sha256(ROOT / artifacts[path_key]), artifacts[hash_key]
            )
        self.assertEqual(
            self.config["provisional_resource_contract"],
            {
                "partition": "main",
                "account": "normal",
                "qos": "normal",
                "source_host": "worker-1",
                "gpus": 1,
                "cpus_per_task": 8,
                "host_memory_mib": 65536,
                "time_limit": "00:30:00",
                "array": "0-0%1",
                "requeue": False,
                "validator_cpus": 2,
                "validator_host_memory_mib": 8192,
                "validator_time_limit": "00:15:00",
                "validator_gpus": 0,
                "validator_dependency": "afterany",
            },
        )
        decision = _squash_whitespace(self.adr_0060)
        for phrase in (
            "zero-GPU CPU `afterany` job is the sole `results.json` publisher",
            "one H100 on `worker-1`, eight CPUs, 64 GiB host",
            "30 minutes, array `0-0%1`, and no requeue",
            "CPU publisher uses two CPUs, 8 GiB, 15 minutes, and zero GPUs",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, decision)

    def test_no_simulator_training_or_ift01_is_authorized(self) -> None:
        boundary = self.config["execution_boundary"]
        self.assertTrue(boundary["simulator_reset_and_dummy_settle_only"])
        self.assertEqual(boundary["policy_generated_action_steps_executed"], 0)
        self.assertEqual(boundary["teacher_generated_action_steps_executed"], 0)
        self.assertEqual(boundary["efficacy_rollouts_executed"], 0)
        self.assertFalse(boundary["simulator_efficacy_evaluated"])
        self.assertEqual(boundary["geometry_or_planner_queries"], 0)
        self.assertFalse(boundary["parameter_gradients"])
        self.assertFalse(boundary["training"])
        self.assertFalse(boundary["collision_or_progress_claim_allowed"])
        self.assertFalse(boundary["probe_or_mlp_training_authorized"])
        self.assertFalse(boundary["automatic_next_gate_authorized"])
        self.assertFalse(self.config["allow_test_tuning"])
        self.assertFalse(self.config["training"])
        for decision in (self.adr_0059, self.adr_0060):
            for phrase in ("simulator execution", "IFT-01", "probe training"):
                with self.subTest(decision=decision[:6], phrase=phrase):
                    self.assertIn(phrase, decision)
        self.assertIn("residual-field MLP training", _squash_whitespace(self.adr_0059))
        self.assertIn("residual-field MLP training", _squash_whitespace(self.adr_0060))

    def test_adr0059_binds_checked_in_diagnostic_hashes_and_claim_boundaries(self) -> None:
        relative_evidence_path = "evidence/r05a/af00a-sealed-population-diagnostic.json"
        self.assertIn(relative_evidence_path, self.adr_0059)
        for digest in self.diagnostic["sealed_input_sha256"].values():
            with self.subTest(digest=digest):
                self.assertIn(digest, self.adr_0059)
        analysis = self.diagnostic["reproducible_analysis"]
        for hash_name in (
            "module_sha256",
            "cli_sha256",
            "focused_test_sha256",
            "deterministic_full_report_sha256",
        ):
            with self.subTest(hash_name=hash_name):
                self.assertIn(analysis[hash_name], self.adr_0059)
        exact = self.diagnostic["exact_transport_decomposition"]
        for value in (
            exact["target_physical_xyz_l2"],
            exact["necessary_xyz_rms_pass_alpha_lower_bound"],
            exact["arm_a"]["direct_target_alpha"],
            exact["arm_a"]["field_feedback_target_alpha"],
            exact["arm_a"]["terminal_target_alpha"],
            exact["population"]["terminal_target_alpha_max"],
        ):
            with self.subTest(value=value):
                self.assertIn(str(value), self.adr_0059)
        interpretation = self.diagnostic["interpretation"]
        self.assertFalse(interpretation["constant_delta_over_time_conversion_supported"])
        self.assertTrue(interpretation["state_and_time_dependent_reference_lift_test_required"])
        self.assertFalse(interpretation["global_reachability_resolved"])
        self.assertIn("not global\n   infeasibility", self.adr_0059)
        self.assertIn("state- and time-dependent teacher field", self.adr_0059)
        for claim, allowed in self.diagnostic["claim_boundaries"].items():
            with self.subTest(claim=claim):
                self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
