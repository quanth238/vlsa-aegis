from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class R02HpcContractTest(unittest.TestCase):
    def test_external_config_is_content_bound_to_passing_predecessors(self) -> None:
        value = json.loads(
            (ROOT / "configs/experiments/r02_oracle_flow.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIs(value["ready_to_run"], True)
        self.assertEqual(value["blocked_on"], [])
        self.assertEqual(value["manifest"], "manifests/oracle_h05_colliding.jsonl")
        settings = value["r02"]
        self.assertEqual(
            settings["r01_summary_sha256"],
            "715d9326f02df531e0c6d7aa7b94629569b320800b4283c36fff41335d9b8bc5",
        )
        self.assertEqual(
            settings["sampler_parity_sha256"],
            "26083b0a71ec41cf67e0978b815a77bfe71e4b4a4a08a8ddcdaece608de9818a",
        )
        self.assertEqual(settings["sampler_parity_slurm_job_id"], "27389")
        self.assertEqual(settings["clipping_policy"], "fail_without_clipping")
        self.assertEqual(settings["simulator_repeats"], 2)
        self.assertEqual(
            settings["direction_reference"],
            "immutable R01 witness translation minus fresh paired eager translation",
        )
        self.assertEqual(
            settings["direction_semantics_decision"],
            "docs/decisions/0013-reference-oracle-to-fresh-paired-baseline.md",
        )
        self.assertEqual(
            settings["direction_semantics_decision_sha256"],
            "9af853d339059d8bbfade06f7d43f5ffe3303d89d98cbfa24158016813251b0c",
        )

    def test_allocation_runner_is_standalone_opt_in_and_hash_bound(self) -> None:
        source = (ROOT / "scripts/hpc/run_r02_case.sh").read_text(encoding="utf-8")
        self.assertIn("SLURM_JOB_ID:?run_r02_case.sh must execute inside", source)
        self.assertIn("main/run_crfs_r02.py", source)
        self.assertNotIn("CRFS_RUNNER_MODE", source)
        self.assertIn("EXPERIMENT_CONFIG:?", source)
        self.assertNotIn("configs/experiments/", source)
        for variable in (
            "R01_RAW_ROOT",
            "R01_SUMMARY",
            "R01_SUMMARY_SHA256",
            "PARITY_ARTIFACT",
            "PARITY_ARTIFACT_SHA256",
        ):
            self.assertIn(f"${{{variable}", source)
        self.assertIn('sha256sum "$R01_SUMMARY"', source)
        self.assertIn('sha256sum "$PARITY_ARTIFACT"', source)
        self.assertIn('test "$OBSERVED_R01_SUMMARY_SHA256" = "$R01_SUMMARY_SHA256"', source)
        self.assertIn(
            'test "$OBSERVED_PARITY_ARTIFACT_SHA256" = "$PARITY_ARTIFACT_SHA256"',
            source,
        )

    def test_allocation_runner_owns_server_and_writes_atomic_failure(self) -> None:
        source = (ROOT / "scripts/hpc/run_r02_case.sh").read_text(encoding="utf-8")
        self.assertIn("pi05_libero_pytorch", source)
        self.assertIn("scripts/serve_policy.py", source)
        self.assertIn("SERVER_PID=$!", source)
        self.assertIn("trap cleanup EXIT", source)
        self.assertIn('kill "$SERVER_PID"', source)
        self.assertIn("launch-failure.json", source)
        self.assertIn('"case_id": os.environ["CASE_ID"]', source)
        self.assertIn('"policy_server_log"', source)
        self.assertIn('"client_log"', source)
        self.assertIn("tempfile.NamedTemporaryFile", source)
        self.assertIn("os.replace(temporary, path)", source)
        self.assertNotIn("rm -rf", source)
        for flag in (
            "--manifest",
            "--config",
            "--r01-raw-root",
            "--r01-summary",
            "--r01-summary-sha256",
            "--parity-artifact",
            "--parity-artifact-sha256",
            "--checkpoint-id",
            "--checkpoint-sha256",
            "--case-index",
        ):
            self.assertIn(flag, source)

    def test_slurm_templates_stay_within_user_ceiling(self) -> None:
        smoke = (ROOT / "slurm/r02_mig.sbatch").read_text(encoding="utf-8")
        full = (ROOT / "slurm/r02_main_array.sbatch").read_text(encoding="utf-8")
        for source in (smoke, full):
            self.assertIn("#SBATCH --gres=gpu:1", source)
            self.assertIn("scripts/hpc/run_r02_case.sh", source)
        self.assertIn("#SBATCH --array=0-0%1", smoke)
        self.assertIn("#SBATCH --cpus-per-task=6", smoke)
        self.assertIn("#SBATCH --mem=80G", smoke)
        self.assertIn("#SBATCH --cpus-per-task=8", full)
        self.assertIn("#SBATCH --mem=128G", full)
        self.assertNotIn("#SBATCH --array", full)

    def test_submitters_preflight_and_fail_closed_on_unfrozen_inputs(self) -> None:
        smoke = (ROOT / "scripts/hpc/submit_r02_smoke.sh").read_text(encoding="utf-8")
        array = (ROOT / "scripts/hpc/submit_r02_array.sh").read_text(encoding="utf-8")
        for source in (smoke, array):
            self.assertIn("scripts/hpc/preflight.sh", source)
            self.assertIn("RUN_ID:?", source)
            self.assertIn("CONFIG_LOCAL", source)
            self.assertIn("ready_to_run", source)
            self.assertIn("blocked_on", source)
            self.assertIn("frozen original 20-case manifest", source)
            self.assertIn(r"[0-9a-f]{64}", source)
            self.assertIn('sha256sum "$r01_summary"', source)
            self.assertIn('sha256sum "$parity_artifact"', source)
            self.assertNotIn("rm -rf", source)
            self.assertNotIn("--delete", source)
        self.assertIn("#SBATCH --array=0-0%1", (ROOT / "slurm/r02_mig.sbatch").read_text())
        self.assertIn("CONCURRENCY=${8:-1}", array)
        self.assertIn("1|2", array)
        self.assertIn('sbatch --array="0-$last%$concurrency"', array)

    def test_summary_submitter_requires_frozen_full_population_and_cpu_job(self) -> None:
        source = (ROOT / "scripts/hpc/submit_r02_summary.sh").read_text(
            encoding="utf-8"
        )
        for required in (
            "RUN_ID:?",
            "CONFIG_SHA256",
            "R01_SUMMARY_SHA256",
            "CHECKPOINT_SHA256",
            "scripts/hpc/preflight.sh",
            "slurm/r02_summary.sbatch",
            "r02-paired.json",
            'test "$manifest_count" -eq 20',
            'test "$result_count" -eq 20',
            'sbatch --parsable',
            'printf \'%s\\n\' "$job_id"',
        ):
            self.assertIn(required, source)
        self.assertIn('shasum -a 256 "$CONFIG_LOCAL"', source)
        self.assertIn('shasum -a 256 "$R01_SUMMARY_LOCAL"', source)
        self.assertIn('sha256sum "$experiment_config"', source)
        self.assertIn('sha256sum "$r01_summary"', source)
        self.assertIn('test -d "$results_root"', source)
        self.assertIn('test -f "$checkpoint_id/model.safetensors"', source)
        self.assertIn("CPU-only Slurm allocation", source)
        remote = source.split("<<'REMOTE'", 1)[1]
        self.assertNotIn("python3", remote)
        self.assertNotIn("LIBERO_PYTHON", remote)
        self.assertNotIn("--gres=gpu", source)
        self.assertNotIn("rm -rf", source)
        self.assertNotIn("--delete", source)


if __name__ == "__main__":
    unittest.main()
