from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/r04_continuation_labels.json"
MANIFEST = ROOT / "manifests/reach_progress_calibration.jsonl"
EXPECTED_CONFIG_SHA256 = (
    "561128a5a05710e50b282582463127ee3f8cd87c9ebbaee822f8c4602fda1c25"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R04ProtocolContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.settings = self.config["r04a"]

    def test_apparatus_is_opt_in_pregrasp_trace_only_and_never_training(self) -> None:
        self.assertIs(self.config["ready_to_run"], True)
        self.assertEqual(self.config["blocked_on"], [])
        self.assertEqual(
            self.config["manifest"], "manifests/reach_progress_calibration.jsonl"
        )
        self.assertEqual(self.config["executed_prefix"], 5)
        self.assertEqual(self.config["action_horizon"], 10)
        self.assertEqual(self.config["sampler_steps"], 10)
        self.assertEqual(self.config["measurement_repeats"], 2)
        self.assertIs(self.settings["enabled"], True)
        self.assertEqual(self.settings["phase"], "pregrasp_reach")
        self.assertEqual(self.settings["apparatus_scope"], "label_contract_smoke_only")
        self.assertEqual(
            self.settings["reuse_role"],
            "apparatus_only_never_train_calibrate_validate_test_or_claim",
        )
        self.assertEqual(self.settings["trace_steps"], [1, 2, 3, 4, 5])
        self.assertEqual(self.settings["trace_times"], [0.9, 0.8, 0.7, 0.6, 0.5])
        self.assertEqual(self.settings["duplicate_trace_requests"], 2)
        self.assertEqual(
            self.settings["continuation"],
            "deterministic_frozen_eager_no_intervention",
        )
        self.assertEqual(self.settings["intervention_mode"], "none")
        self.assertIsNone(self.settings["correction"])
        self.assertIs(self.settings["training"], False)
        self.assertIs(self.settings["guidance"], False)
        self.assertEqual(self.settings["model_action_shape"], [10, 32])
        self.assertEqual(
            self.settings["model_action_shape_role"],
            "normalized_intermediate_trace_tensor_not_exposed_final_action",
        )
        self.assertEqual(self.settings["physical_action_shape"], [10, 7])
        self.assertEqual(self.settings["simulator_repeats"], 2)
        self.assertEqual(
            self.settings["exact_final_action_requirement"],
            "physical_10x7_array_equal_across_all_trace_steps_and_duplicates",
        )
        self.assertIs(self.settings["retain_all_outcomes"], True)

    def test_label_and_variable_geometry_are_frozen_to_actual_outcome(self) -> None:
        self.assertEqual(
            self.settings["primary_trace_feature"],
            "predicted_clean_normalized_model_coordinates",
        )
        self.assertEqual(
            self.settings["physical_trace_feature_role"],
            "inverse_transformed_audit_only",
        )
        self.assertEqual(
            self.settings["label"],
            "minimum_D_sim_over_two_exact_replays_of_the_bound_frozen_continuation_prefix",
        )
        self.assertIs(self.settings["bind_each_trace_to_final_action_sha256"], True)
        self.assertIs(self.settings["bind_each_trace_to_rollout_and_label_sha256"], True)
        self.assertEqual(self.settings["geometry_frame"], "world")
        self.assertEqual(
            self.settings["geometry_representation"], "geom_name_sorted_padded_obb"
        )
        self.assertEqual(self.settings["maximum_obbs"], 21)
        self.assertEqual(self.settings["geometry_padding_value"], 0.0)
        self.assertIs(self.settings["geometry_validity_mask"], True)
        self.assertEqual(self.config["eef_radius_m"], 0.06)
        self.assertEqual(self.config["safety_margin_m"], 0.005)
        self.assertEqual(
            self.settings["minimum_progress_m"], 0.029897349105658888
        )

    def test_source_artifacts_and_checkpoint_are_exactly_content_bound(self) -> None:
        self.assertEqual(_sha256(CONFIG), EXPECTED_CONFIG_SHA256)
        expected = {
            MANIFEST: "3180836991e64b774670218f5fe4edaa10775fbb33fcdb330cc781b8b8937dad",
            ROOT / "evidence/r00/r00-summary.json": "90fe09e075b30e680e5cfbd81c802afa03c94b58ba7f0e95e66080c61af2096f",
            ROOT / "evidence/r03/r03-summary.json": "dea3e66c854faa0b659777bfed4b651b19d8d4d0710696ff7bfe52c44d4ee76e",
        }
        for path, digest in expected.items():
            self.assertEqual(_sha256(path), digest)
        self.assertEqual(self.config["manifest_sha256"], expected[MANIFEST])
        self.assertEqual(
            self.config["checkpoint_sha256"],
            "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",
        )
        self.assertEqual(
            self.settings["r00_summary_sha256"],
            expected[ROOT / "evidence/r00/r00-summary.json"],
        )
        self.assertEqual(
            self.settings["r03_summary_sha256"],
            expected[ROOT / "evidence/r03/r03-summary.json"],
        )
        self.assertEqual(
            self.settings["sampler_parity_sha256"],
            "26083b0a71ec41cf67e0978b815a77bfe71e4b4a4a08a8ddcdaece608de9818a",
        )
        for script in (
            ROOT / "scripts/hpc/run_r04_label_case.sh",
            ROOT / "scripts/hpc/submit_r04_label_smoke.sh",
        ):
            self.assertIn(
                EXPECTED_CONFIG_SHA256,
                script.read_text(encoding="utf-8"),
            )

    def test_r00_reuse_limitation_is_recomputed_not_asserted(self) -> None:
        records = [
            json.loads(line)
            for line in MANIFEST.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        summary = json.loads(
            (ROOT / "evidence/r00/r00-summary.json").read_text(encoding="utf-8")
        )
        clearances = [float(row["clearance_m"]) for row in summary["case_records"]]
        group_sizes = {
            group_id: sum(row["group_id"] == group_id for row in records)
            for group_id in {row["group_id"] for row in records}
        }
        self.assertEqual(len(records), 120)
        self.assertEqual(len({row["group_id"] for row in records}), 30)
        self.assertEqual(set(group_sizes.values()), {4})
        self.assertEqual(sum(0.0 <= value <= 0.010 for value in clearances), 0)
        self.assertEqual(min(clearances), 0.010326895138387013)
        audit = self.settings["coverage_audit"]
        self.assertEqual(audit["manifest_rows"], 120)
        self.assertEqual(audit["state_groups"], 30)
        self.assertEqual(audit["rows_in_boundary_interval"], 0)
        self.assertEqual(audit["boundary_interval_m"], [0.0, 0.01])

    def test_claim_population_requires_new_presplit_groups_and_deferred_edits(self) -> None:
        claim = self.settings["claim_bearing_requirements"]
        self.assertIs(claim["new_immutable_state_groups"], True)
        self.assertEqual(claim["minimum_unique_training_state_hashes"], 200)
        self.assertIs(claim["split_complete_groups_before_label_generation"], True)
        self.assertEqual(
            claim["split_and_bootstrap_group_unit"],
            "immutable_source_initial_episode_or_state",
        )
        self.assertIs(claim["keep_all_replay_derived_branches_with_source_group"], True)
        self.assertIs(claim["bootstrap_complete_source_episode_groups"], True)
        self.assertIs(claim["boundary_coverage_required"], True)
        self.assertIs(claim["r02_claim_cases_excluded_from_all_splits"], True)
        self.assertIs(self.settings["r02_claim_cases_excluded"], True)
        future = self.settings["future_local_perturbation_gate"]
        self.assertIs(future["implemented"], False)
        self.assertIs(future["requires_post_edit_trace"], True)
        self.assertIs(future["existing_bridge_pre_edit_trace_allowed"], False)
        decision = (
            ROOT / "docs/decisions/0017-freeze-r04a-continuation-label-contract.md"
        ).read_text(encoding="utf-8")
        self.assertIn("does not separately resume from a saved latent", decision)
        self.assertIn("does not add a normalized-final-action", decision)
        self.assertIn("post-edit", decision)
        self.assertIn("invalid as an edited-feature label source", decision)
        self.assertIn("no learning,", decision)


class R04HpcContractTest(unittest.TestCase):
    def test_allocation_runner_is_opt_in_hash_bound_and_atomic_on_failure(self) -> None:
        source = (ROOT / "scripts/hpc/run_r04_label_case.sh").read_text(
            encoding="utf-8"
        )
        for required in (
            "SLURM_JOB_ID:?run_r04_label_case.sh must execute inside",
            "SLURM_ARRAY_TASK_ID:?R04A smoke must be a one-element Slurm array",
            'test "$CASE_INDEX" = "$SLURM_ARRAY_TASK_ID"',
            "main/run_r04_label_contract.py",
            "R04A is frozen to reused R00 case index 0",
            "crfs-93365b8b851365f2",
            "r04-label-contract.json",
            "trap cleanup EXIT",
            'kill "$SERVER_PID"',
            "launch-failure.json",
            "tempfile.NamedTemporaryFile",
            "os.replace(temporary, path)",
            'sha256sum "$MANIFEST"',
            'sha256sum "$R00_SUMMARY"',
            'sha256sum "$R03_SUMMARY"',
            'sha256sum "$PARITY_ARTIFACT"',
            'sha256sum "$MODEL"',
            'test "$GIT_DIRTY" = false',
            "--checkpoint-sha256",
        ):
            self.assertIn(required, source)
        self.assertNotIn("CRFS_RUNNER_MODE", source)
        self.assertNotIn("--correction", source)
        self.assertNotIn("--train", source)
        for forbidden in ("rm -rf", "scancel", "pkill", "rsync --delete"):
            self.assertNotIn(forbidden, source)

    def test_smoke_resources_and_live_submit_guard_match_vinuni_contract(self) -> None:
        slurm = (ROOT / "slurm/r04_label_mig.sbatch").read_text(encoding="utf-8")
        for required in (
            "#SBATCH --partition=mig",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=6",
            "#SBATCH --mem=80G",
            "#SBATCH --time=01:00:00",
            "#SBATCH --array=0-0%1",
            "scripts/hpc/run_r04_label_case.sh",
        ):
            self.assertIn(required, slurm)
        submit = (ROOT / "scripts/hpc/submit_r04_label_smoke.sh").read_text(
            encoding="utf-8"
        )
        for required in (
            "scripts/hpc/preflight.sh",
            "RUN_ID:?",
            "R04A source files must be committed before submission",
            "remote source is not synchronized",
            "remote source tree is dirty",
            "squeue -h -u",
            "AllocTRES=",
            "projected_gpus=$((allocated_gpus + 1))",
            "projected_cpus=$((allocated_cpus + 6))",
            "projected_mem_mb=$((allocated_mem_mb + 80 * 1024))",
            "exceed the 2-GPU-equivalent user ceiling",
            "exceed the 16-CPU user ceiling",
            "exceed the 256-GiB user ceiling",
            "sinfo -h -p mig -N",
            "scontrol show node",
            "FreeMem=",
            "minimum_free_mem_mb=$((80 * 1024))",
            "*down*|*drain*|*drng*|*fail*|*maint*|*not_resp*",
            "no healthy MIG node has a fresh 80 GiB",
            "--nodelist=",
            "--exclude=",
            "slurm/r04_label_mig.sbatch",
        ):
            self.assertIn(required, submit)
        remote = submit.split("<<'REMOTE'", 1)[1]
        self.assertNotIn("python3", remote)
        for forbidden in ("rm -rf", "scancel", "pkill", "rsync", "--delete"):
            self.assertNotIn(forbidden, submit)

    def test_shell_files_parse(self) -> None:
        for path in (
            ROOT / "scripts/hpc/run_r04_label_case.sh",
            ROOT / "scripts/hpc/submit_r04_label_smoke.sh",
            ROOT / "slurm/r04_label_mig.sbatch",
        ):
            result = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
