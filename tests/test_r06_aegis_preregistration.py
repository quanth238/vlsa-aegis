from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / "r06_aegis_collision_conditioned.json"
DECISION_PATH = (
    ROOT
    / "docs"
    / "decisions"
    / "0064-preregister-aegis-collision-conditioned-baseline.md"
)
MANIFEST_PATH = ROOT / "manifests" / "oracle_h05_colliding.jsonl"
ELIGIBLE_MANIFEST_PATH = ROOT / "manifests" / "r03a_analytic_kill_test_eligible.jsonl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class AegisCollisionConditionedPreregistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.manifest = _jsonl(MANIFEST_PATH)
        cls.eligible = _jsonl(ELIGIBLE_MANIFEST_PATH)
        cls.decision = DECISION_PATH.read_text(encoding="utf-8")

    def test_implementation_is_authorized_and_execution_is_draft_or_exact_capture_release(self) -> None:
        value = self.config
        self.assertEqual(value["experiment_identity"], "AEGIS-00A")
        self.assertEqual(value["gate"], "R06")
        self.assertFalse(value["claims"]["probe_or_mlp_training_authorized"])
        if value["ready_to_run"] is False:
            self.assertEqual(
                value["config_status"], "draft_preregistered_not_released"
            )
            self.assertIsNone(value["execution_release"])
            label = value["codex_label_protocol"]["canary_label_manifest"]
            if label["sha256"] is None:
                self.assertIn("canary_branch_image_not_captured", value["blocked_on"])
                self.assertIn("codex_canary_label_not_frozen", value["blocked_on"])
                self.assertIn(
                    "groundingdino_allocation_runtime_preflight_not_complete",
                    value["blocked_on"],
                )
                self.assertIn("no H100 execution release yet", self.decision)
            else:
                self.assertEqual(
                    label,
                    {
                        "path": "manifests/r06_codex_obstacle_labels_canary.jsonl",
                        "sha256": (
                            "6a22b6d2f3705c008e338be6b217ce947fabfd4f56f70d3a8f"
                            "442467694d996f"
                        ),
                        "expected_rows": 1,
                        "status": "frozen_from_valid_capture_retry_b",
                    },
                )
                self.assertEqual(
                    value["blocked_on"],
                    [
                        "paired_canary_retry_release_not_authorized",
                        "paired_canary_integration_not_terminally_validated",
                        "population_label_manifest_not_authorized",
                    ],
                )
            return

        self.assertIs(value["ready_to_run"], True)
        self.assertEqual(value["blocked_on"], [])
        release = value["execution_release"]
        self.assertIsInstance(release, dict)
        if release["stage"] == "paired_codex_label_canary":
            self.assertEqual(
                value["config_status"],
                "released_exact_codex_label_paired_canary",
            )
            self.assertEqual(
                release["artifact_role"],
                "r06_aegis_paired_codex_label_canary_execution_release",
            )
            self.assertEqual(release["single_case_index"], 0)
            self.assertEqual(release["case_id"], "crfs-1069f29a8d76463a")
            self.assertTrue(release["aegis_execution_allowed"])
            self.assertTrue(release["semantic_label_required"])
            self.assertTrue(release["groundingdino_execution_allowed"])
            self.assertTrue(release["qp_execution_allowed"])
            self.assertFalse(release["original_glm_execution_allowed"])
            self.assertFalse(release["probe_or_mlp_training_authorized"])
            self.assertFalse(release["automatic_population_launch_authorized"])
            self.assertEqual(
                release["codex_label_manifest"]["sha256"],
                "6a22b6d2f3705c008e338be6b217ce947fabfd4f56f70d3a8f"
                "442467694d996f",
            )
            self.assertEqual(release["resources"]["validator_gpus"], 0)
            self.assertEqual(
                release["resources"]["validator_dependency"], "afterany"
            )
            decision = release["decision_artifact"]
            self.assertTrue(
                decision.startswith("docs/decisions/")
                and decision.endswith("-release-aegis-paired-canary.md")
            )
            self.assertEqual(
                release["allowed_release_diff_paths"],
                [
                    "configs/experiments/r06_aegis_collision_conditioned.json",
                    decision,
                ],
            )
            return

        self.assertEqual(
            value["config_status"], "released_exact_codex_label_capture_canary"
        )
        self.assertEqual(
            release["artifact_role"],
            "r06_aegis_codex_label_capture_execution_release",
        )
        self.assertEqual(release["stage"], "codex_label_capture")
        self.assertEqual(release["single_case_index"], 0)
        self.assertEqual(release["case_id"], "crfs-1069f29a8d76463a")
        self.assertFalse(release["aegis_execution_allowed"])
        self.assertFalse(release["semantic_label_required"])
        self.assertFalse(release["groundingdino_execution_allowed"])
        self.assertFalse(release["qp_execution_allowed"])
        self.assertFalse(release["probe_or_mlp_training_authorized"])
        self.assertFalse(release["automatic_population_launch_authorized"])
        self.assertTrue(release["release_only_parent_required"])
        self.assertEqual(release["robosuite_version"], "1.4.1")
        self.assertEqual(release["robosuite_image_convention"], "opengl")
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
                "source_host": "worker-1",
                "requeue": False,
            },
        )
        self.assertEqual(len(release["accepted_implementation_commit"]), 40)
        self.assertTrue(
            all(
                character in "0123456789abcdef"
                for character in release["accepted_implementation_commit"]
            )
        )
        decision = release["decision_artifact"]
        self.assertTrue(
            decision.startswith("docs/decisions/")
            and decision.endswith("-release-aegis-label-capture-canary.md")
        )
        self.assertEqual(
            release["allowed_release_diff_paths"],
            ["configs/experiments/r06_aegis_collision_conditioned.json", decision],
        )

    def test_complete_ordered_twenty_case_denominator_is_immutable(self) -> None:
        source = self.config["manifest"]
        self.assertEqual(source["path"], "manifests/oracle_h05_colliding.jsonl")
        self.assertEqual(source["sha256"], _sha256(MANIFEST_PATH))
        self.assertEqual(source["ordered_cases"], 20)
        self.assertEqual(len(self.manifest), 20)
        self.assertEqual(len({row["case_id"] for row in self.manifest}), 20)
        self.assertTrue(source["case_removal_forbidden"])
        self.assertEqual(self.config["canary"]["row_index_zero_based"], 0)
        self.assertEqual(
            self.config["canary"]["case_id"], self.manifest[0]["case_id"]
        )
        self.assertEqual(self.config["canary"]["case_id"], "crfs-1069f29a8d76463a")

    def test_frozen_seventeen_plus_three_strata_partition_population(self) -> None:
        strata = self.config["strata"]
        primary = strata["primary_standard_settled"]
        late = strata["pre_settle_late_intervention"]
        primary_ids = {row["case_id"] for row in self.eligible}
        late_ids = set(late["case_ids"])
        population_ids = {row["case_id"] for row in self.manifest}
        self.assertEqual(primary["count"], 17)
        self.assertEqual(primary["manifest_sha256"], _sha256(ELIGIBLE_MANIFEST_PATH))
        self.assertEqual(late["count"], 3)
        self.assertEqual(
            late_ids,
            {
                "crfs-3bd38b2879b8b0a9",
                "crfs-b22f5fccb666732f",
                "crfs-dbbf42a4f4614e0a",
            },
        )
        self.assertFalse(primary_ids & late_ids)
        self.assertEqual(primary_ids | late_ids, population_ids)
        self.assertFalse(late["pooled_with_primary"])

    def test_branch_selection_is_outcome_independent_and_control_realizable(self) -> None:
        branch = self.config["branch_selection"]
        self.assertEqual(
            branch["candidate_boundaries"],
            "completed_dummy_control_boundaries_0_through_20",
        )
        self.assertEqual(
            branch["selection"],
            "latest_boundary_with_no_forbidden_contact_and_inclusive_D_sim_gte_0.005m",
        )
        self.assertEqual(
            branch["hidden_physics_substep_branching"],
            "forbidden_measurement_only",
        )
        self.assertEqual(branch["primary_requires_boundary"], 20)
        self.assertTrue(branch["no_admissible_branch_is_retained"])
        self.assertTrue(branch["selection_must_not_examine_aegis_outcome"])

    def test_pairing_preserves_frozen_policy_and_five_action_horizon(self) -> None:
        pairing = self.config["pairing"]
        self.assertEqual(pairing["policy_render_size"], 224)
        self.assertEqual(pairing["aegis_perception_render_size"], 1024)
        self.assertEqual(pairing["policy_noise_shape"], [10, 32])
        self.assertEqual(pairing["policy_noise_dtype"], "float32")
        self.assertEqual(pairing["sampler_path"], "eager_explicit_noise_trace_only")
        self.assertTrue(pairing["duplicate_policy_inference_required"])
        self.assertTrue(pairing["returned_action_byte_equality_required"])
        self.assertEqual(pairing["executed_actions"], 5)
        self.assertEqual(pairing["physics_substeps_per_action"], 25)
        self.assertEqual(
            set(pairing["paired_fields"]),
            {
                "full_integration_state",
                "policy_observation",
                "instruction",
                "policy_noise",
                "nominal_actions",
                "executed_horizon",
                "active_obstacle",
            },
        )

    def test_public_aegis_arm_is_full_literal_and_has_no_silent_fallback(self) -> None:
        arm = self.config["arms"]["pi05_plus_aegis_codex_label"]
        self.assertEqual(_sha256(ROOT / arm["authoritative_entrypoint"]), arm["authoritative_entrypoint_sha256"])
        self.assertEqual(_sha256(ROOT / arm["authoritative_utils"]), arm["authoritative_utils_sha256"])
        self.assertEqual(
            arm["controller"],
            "canonical_public_full_9_variable_translation_rotation_cbf_qp",
        )
        self.assertEqual(
            arm["first_step_robot_geometry_initialization"],
            "literal_public_pre_settle_state",
        )
        self.assertEqual(arm["current_state_initialization_substitution"], "forbidden")
        self.assertEqual(
            arm["perception_mode"],
            "codex_frozen_semantic_label_plus_original_groundingdino",
        )
        self.assertEqual(arm["semantic_selector"], "codex_manual_review_before_aegis_outcome")
        self.assertFalse(arm["original_glm_selector_evaluated"])
        self.assertFalse(arm["reported_as_original_end_to_end_aegis"])
        self.assertEqual(arm["oracle_label_or_geometry_substitution"], "forbidden")
        self.assertEqual(arm["translational_only_substitution"], "forbidden")
        self.assertEqual(arm["qp_fallback"], "forbidden_record_qp_failure")

    def test_metrics_separate_contact_buffer_progress_and_stopping(self) -> None:
        metrics = self.config["metrics"]
        self.assertEqual(metrics["physical_contact_free"], "no_forbidden_physical_contact")
        self.assertEqual(
            metrics["collision_avoidance_success"],
            "no_forbidden_physical_contact_and_D_sim_gte_0m",
        )
        self.assertEqual(
            metrics["registered_buffer_safety"],
            "no_contact_and_D_sim_gte_0.005m",
        )
        self.assertEqual(metrics["minimum_progress_primary_m"], 0.029897349105658888)
        self.assertEqual(
            metrics["joint_primary"],
            "registered_buffer_safety_and_reach_progress_gte_minimum_progress_primary_m",
        )
        self.assertEqual(
            metrics["joint_late_descriptive"],
            "registered_buffer_safety_and_reach_progress_gt_0",
        )
        self.assertEqual(
            metrics["stop_like"],
            "realized_eef_path_length_lte_0.005m_and_abs_reach_progress_lte_0.001m",
        )
        self.assertEqual(
            metrics["realized_eef_path_sampling"],
            "branch_sample_plus_all_125_physics_substeps",
        )
        self.assertTrue(metrics["continuous_motion_retention_required"])

    def test_all_failures_are_retained_and_claims_stay_narrow(self) -> None:
        failures = self.config["failure_accounting"]
        claims = self.config["claims"]
        self.assertTrue(failures["retain_every_case"])
        self.assertEqual(failures["silent_fallback"], "forbidden")
        self.assertEqual(
            failures["baseline_collision_not_reproduced"],
            "retained_not_a_rescue",
        )
        self.assertTrue(claims["collision_conditioned_only"])
        self.assertFalse(claims["original_glm_selector_evaluated"])
        self.assertTrue(claims["original_end_to_end_aegis_claim_forbidden"])
        self.assertTrue(claims["general_benchmark_rate_forbidden"])
        self.assertTrue(claims["forward_invariance_claim_forbidden"])
        self.assertTrue(claims["superiority_claim_forbidden"])
        self.assertFalse(claims["aegis_solves_cases_before_validated_artifacts"])

    def test_every_content_bound_local_source_matches(self) -> None:
        source = self.config["source_evidence"]
        self.assertEqual(_sha256(ROOT / source["r01_summary"]["path"]), source["r01_summary"]["sha256"])
        self.assertEqual(_sha256(ROOT / source["r02_config"]["path"]), source["r02_config"]["sha256"])
        self.assertEqual(_sha256(ROOT / source["base_decision"]), source["base_decision_sha256"])
        self.assertEqual(_sha256(ROOT / source["current_decision"]), source["current_decision_sha256"])
        self.assertEqual(
            source["canary_r02_pair"],
            {
                "run_id": "r02-oracle-flow-population-20260714a",
                "path": "/mnt/data/quanth/experiments/crfs-oracle/r02-oracle-flow-population-20260714a/crfs-1069f29a8d76463a/r02-paired.json",
                "sha256": "055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593",
            },
        )
        self.assertEqual(
            source["checkpoint"]["sha256"],
            "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",
        )
        self.assertEqual(
            source["checkpoint_config"]["sha256"],
            "5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a",
        )
        self.assertEqual(
            source["normalization_asset"]["sha256"],
            source["normalization_asset_sha256"],
        )

    def test_groundingdino_assets_are_setup_job_and_content_bound(self) -> None:
        assets = self.config["groundingdino_assets"]
        evidence_path = ROOT / assets["setup_evidence"]
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        self.assertEqual(_sha256(evidence_path), assets["setup_evidence_sha256"])
        self.assertEqual(evidence["slurm_job_id"], "28391")
        self.assertEqual(evidence["state"], "COMPLETED")
        self.assertEqual(evidence["exit_code"], "0:0")
        self.assertEqual(
            evidence["groundingdino"]["config_sha256"],
            assets["config_sha256"],
        )
        self.assertEqual(
            evidence["groundingdino"]["checkpoint_sha256"],
            assets["checkpoint_sha256"],
        )
        self.assertFalse(assets["glm_api_key_required"])

    def test_codex_label_is_frozen_before_aegis_and_uses_no_simulator_identity(self) -> None:
        protocol = self.config["codex_label_protocol"]
        self.assertEqual(
            protocol["stage_order"],
            [
                "allocation_capture_and_baseline_reproduction",
                "codex_image_review",
                "immutable_label_freeze",
                "paired_aegis_execution",
            ],
        )
        self.assertEqual(protocol["reviewer"], "codex")
        self.assertFalse(protocol["simulator_object_name_or_geometry_allowed"])
        self.assertEqual(protocol["relabel_after_aegis_outcome"], "forbidden")
        self.assertEqual(protocol["canary_label_manifest"]["expected_rows"], 1)
        canary_label = protocol["canary_label_manifest"]
        if canary_label["status"] == "not_captured_or_labeled":
            self.assertIsNone(canary_label["sha256"])
        else:
            self.assertEqual(
                canary_label["status"], "frozen_from_valid_capture_retry_b"
            )
            self.assertEqual(
                canary_label["sha256"],
                (
                    "6a22b6d2f3705c008e338be6b217ce947fabfd4f56f70d3a8f"
                    "442467694d996f"
                ),
            )
        self.assertEqual(protocol["population_label_manifest"]["expected_rows"], 20)


if __name__ == "__main__":
    unittest.main()
