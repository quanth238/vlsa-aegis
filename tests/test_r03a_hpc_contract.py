from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUBMIT_JOB = ROOT / "scripts/hpc/submit_r03a_job.sh"
SUBMIT_SMOKE = ROOT / "scripts/hpc/submit_r03a_smoke.sh"
SUBMIT_H100_SMOKE = ROOT / "scripts/hpc/submit_r03a_h100_smoke.sh"
SUBMIT_ARRAY = ROOT / "scripts/hpc/submit_r03a_array.sh"
SUBMIT_SUMMARY = ROOT / "scripts/hpc/submit_r03a_summary.sh"
WORKER = ROOT / "scripts/hpc/run_r03a_case.sh"
JSONSCHEMA_OVERLAY = ROOT / "scripts/hpc/prepare_jsonschema_overlay.sh"
SLURM_SMOKE = ROOT / "slurm/r03a_mig.sbatch"
SLURM_H100_SMOKE = ROOT / "slurm/r03a_h100_smoke.sbatch"
SLURM_ARRAY = ROOT / "slurm/r03a_main_array.sbatch"
SLURM_SUMMARY = ROOT / "slurm/r03a_summary.sbatch"
CONFIG = ROOT / "configs/experiments/r03a_analytic_kill_test.json"
MANIFEST = ROOT / "manifests/r03a_analytic_kill_test_eligible.jsonl"
SCHEMA = ROOT / "schemas/r03a-analytic-kill-test.schema.json"
DECISION = ROOT / "docs/decisions/0023-run-strong-analytic-kill-test.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class R03AHPCContractTest(unittest.TestCase):
    def test_every_shell_entry_point_has_valid_bash_syntax(self) -> None:
        paths = (
            SUBMIT_JOB,
            SUBMIT_SMOKE,
            SUBMIT_H100_SMOKE,
            SUBMIT_ARRAY,
            SUBMIT_SUMMARY,
            WORKER,
            JSONSCHEMA_OVERLAY,
            SLURM_SMOKE,
            SLURM_H100_SMOKE,
            SLURM_ARRAY,
            SLURM_SUMMARY,
        )
        for path in paths:
            with self.subTest(path=path.name):
                subprocess.run(["bash", "-n", str(path)], check=True)

    def test_gpu_slurm_profiles_call_only_the_in_allocation_worker(self) -> None:
        smoke = SLURM_SMOKE.read_text(encoding="utf-8")
        h100_smoke = SLURM_H100_SMOKE.read_text(encoding="utf-8")
        array = SLURM_ARRAY.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --partition=mig", smoke)
        self.assertIn("#SBATCH --array=0-0%1", smoke)
        self.assertIn("#SBATCH --gres=gpu:1", smoke)
        self.assertIn("#SBATCH --cpus-per-task=6", smoke)
        self.assertIn("#SBATCH --mem=80G", smoke)
        self.assertIn("scripts/hpc/run_r03a_case.sh", smoke)
        self.assertIn("#SBATCH --partition=main", h100_smoke)
        self.assertIn("#SBATCH --array=0-0%1", h100_smoke)
        self.assertIn("#SBATCH --gres=gpu:1", h100_smoke)
        self.assertIn("#SBATCH --cpus-per-task=8", h100_smoke)
        self.assertIn("#SBATCH --mem=128G", h100_smoke)
        self.assertIn("scripts/hpc/run_r03a_case.sh", h100_smoke)
        self.assertIn("#SBATCH --partition=main", array)
        self.assertNotRegex(array, r"(?m)^#SBATCH --array")
        self.assertIn("#SBATCH --gres=gpu:1", array)
        self.assertIn("#SBATCH --cpus-per-task=8", array)
        self.assertIn("#SBATCH --mem=128G", array)
        self.assertIn("scripts/hpc/run_r03a_case.sh", array)
        for value in (smoke, h100_smoke, array):
            self.assertNotIn("main/run_crfs_r03a.py", value)
            self.assertNotIn("serve_policy.py", value)

    def test_submitter_binds_frozen_inputs_reviewed_commit_and_worker_env(self) -> None:
        value = SUBMIT_JOB.read_text(encoding="utf-8")
        expected = {
            "EXPECTED_MANIFEST_SHA256": sha256(MANIFEST),
            "EXPECTED_CONFIG_SHA256": sha256(CONFIG),
            "EXPECTED_SCHEMA_SHA256": sha256(SCHEMA),
            "EXPECTED_DECISION_SHA256": sha256(DECISION),
        }
        for name, digest in expected.items():
            self.assertIn(f"{name}={digest}", value)
        self.assertIn("scripts/hpc/preflight.sh", value)
        self.assertIn("EXPECTED_GIT_COMMIT=$(git rev-parse HEAD)", value)
        self.assertIn("git -C \"$remote_repo\" status --porcelain", value)
        self.assertIn("/^\\?\\? tmp\\//", value)
        for binding in (
            'EXPERIMENT_CONFIG="$config"',
            'CHECKPOINT_DIR="$checkpoint_id"',
            'EXPERIMENT_ROOT="$output_root"',
            'R02_RAW_ROOT="$r02_raw_root"',
            'R03_SUMMARY_SHA256="$r03_summary_sha256"',
            'EXPECTED_GIT_COMMIT="$expected_commit"',
        ):
            self.assertIn(binding, value)
        self.assertNotRegex(value, r"(?m)^\s*(python|python3|uv run)\b")

    def test_worker_preflights_production_schema_dependency_before_server(self) -> None:
        value = WORKER.read_text(encoding="utf-8")
        self.assertIn("prepare_jsonschema_overlay.sh", value)
        dependency = value.index("import jsonschema")
        server = value.rindex("scripts/serve_policy.py")
        runner = value.rindex("main/run_crfs_r03a.py")
        self.assertLess(dependency, server)
        self.assertLess(dependency, runner)
        runner_path = value.index(
            "export PYTHONPATH=$JSONSCHEMA_OVERLAY:$REMOTE_REPO/src"
        )
        self.assertLess(dependency, runner_path)

    def test_jsonschema_overlay_is_allocation_only_exact_and_offline(self) -> None:
        value = JSONSCHEMA_OVERLAY.read_text(encoding="utf-8")
        self.assertIn("SLURM_JOB_ID", value)
        self.assertIn(
            "EXPECTED_SOURCE_BUNDLE_SHA256="
            "72ccff502fcffe6ab4515cff5f9e3de8fa70e0a2b54c235389df003d25880a6c",
            value,
        )
        for package in (
            "jsonschema-4.23.0.dist-info",
            "jsonschema_specifications-2023.12.1.dist-info",
            "pkgutil_resolve_name-1.3.10.dist-info",
            "referencing-0.35.1.dist-info",
            "rpds_py-0.20.1.dist-info",
        ):
            self.assertIn(package, value)
        self.assertIn("sys.version_info[:2] == (3, 8)", value)
        self.assertIn("flock -x", value)
        self.assertIn("expected_top_level=", value)
        self.assertIn("observed_top_level=", value)
        self.assertIn('test ! -L "$root/.source-bundle-sha256"', value)
        self.assertIn("! -type f ! -type d", value)
        self.assertIn("mv -T", value)
        self.assertNotRegex(value, r"\b(pip|uv|curl|wget)\b")
        worker = WORKER.read_text(encoding="utf-8")
        summary = SLURM_SUMMARY.read_text(encoding="utf-8")
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", worker)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", summary)

    def test_submission_caps_resources_and_excludes_unhealthy_nodes(self) -> None:
        value = SUBMIT_JOB.read_text(encoding="utf-8")
        self.assertIn('1|2)', value)
        self.assertIn('--array="0-16%$concurrency"', value)
        self.assertIn('-t PENDING', value)
        self.assertIn('ReqTRES=', value)
        self.assertIn('allocated_job_ids=$(squeue', value)
        self.assertIn('pending_job_ids=$(squeue', value)
        self.assertLess(
            value.index('pending_job_ids=$(squeue'),
            value.index('allocated_job_ids=$(squeue'),
        )
        self.assertIn('node_states=$(sinfo', value)
        self.assertNotIn('done < <(squeue', value)
        self.assertNotIn('done < <(sinfo', value)
        self.assertIn('pending_gpus + concurrency', value)
        self.assertIn('pending_cpus + cpus_per_task * concurrency', value)
        self.assertIn('pending_mem_mb + memory_per_task_mb * concurrency', value)
        self.assertIn('test "$projected_gpus" -le 2', value)
        self.assertIn('test "$projected_cpus" -le 16', value)
        self.assertIn('test "$projected_mem_mb" -le $((256 * 1024))', value)
        for state in ("*down*", "*drain*", "*not_resp*"):
            self.assertIn(state, value)
        self.assertIn(r"*\**", value)
        self.assertIn('sbatch_args+=(--exclude="$excluded_csv")', value)
        self.assertIn("0024-preserve-source-node-trace-pairing.md", value)
        self.assertIn("0025-bind-r03a-trace-gate-to-native-leaf-evidence.md", value)
        self.assertIn("0026-validate-r03a-scalars-in-recorded-dtype.md", value)
        self.assertIn('if [ "$mode" = h100_smoke ]', value)
        self.assertIn(".provenance.host // empty", value)
        self.assertIn("selected R02 source hash differs", value)
        self.assertIn('node" != "$required_source_node', value)
        self.assertIn("required source node $required_source_node", value)

    def test_adr0024_retires_cross_node_smoke_and_ungrouped_array(self) -> None:
        env = {**os.environ, "RUN_ID": "r03a-adr0024-rejection-test"}
        for wrapper, reason in (
            (SUBMIT_SMOKE, "retires the cross-node MIG smoke"),
            (SUBMIT_ARRAY, "blocks the ungrouped full array"),
        ):
            with self.subTest(wrapper=wrapper.name):
                completed = subprocess.run(
                    [str(wrapper), "unused", "unused", "unused", "unused", "unused"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(reason, completed.stderr)
                self.assertNotIn("Live Slurm state", completed.stdout + completed.stderr)

    def test_worker_rejects_integrity_failures_but_keeps_nonfinite_science(self) -> None:
        value = WORKER.read_text(encoding="utf-8")
        result_exists = value.index('test -f "$EXPECTED_RESULT"')
        acceptance = value.index("FAILURE_STAGE=r03a_apparatus_acceptance")
        complete = value.index("FAILURE_STAGE=complete")
        self.assertLess(result_exists, acceptance)
        self.assertLess(acceptance, complete)
        self.assertIn('value.get("status") != "completed"', value)
        self.assertIn('"fresh_nominal_collision_reproduced"', value)
        self.assertIn('"policy_failure"', value)
        self.assertIn(
            '"not_evaluated_after_nominal_collision_not_reconfirmed"', value
        )
        self.assertNotIn('"nonfinite_failure"', value)

    def test_summary_is_cpu_only_and_has_an_exact_afterok_submission_path(self) -> None:
        sbatch = SLURM_SUMMARY.read_text(encoding="utf-8")
        submit = SUBMIT_SUMMARY.read_text(encoding="utf-8")
        self.assertNotRegex(sbatch, r"(?m)^#SBATCH\s+--gres")
        self.assertIn("#SBATCH --cpus-per-task=2", sbatch)
        self.assertIn("#SBATCH --mem=16G", sbatch)
        self.assertIn("main/summarize_r03a.py", sbatch)
        self.assertIn("import jsonschema", sbatch)
        self.assertIn("prepare_jsonschema_overlay.sh", sbatch)
        self.assertIn("PYTHONPATH=$JSONSCHEMA_OVERLAY", sbatch)
        self.assertIn("scripts/hpc/prepare_jsonschema_overlay.sh", submit)
        for option in (
            "--manifest",
            "--config",
            "--r02-raw-root",
            "--r03-summary",
            "--results-root",
            "--source-slurm-array-job-id",
            "--expected-git-commit",
        ):
            self.assertIn(option, sbatch)
        self.assertIn(sha256(CONFIG), submit)
        self.assertIn('SOURCE_ARRAY_JOB_ID must be one exact numeric Slurm job id', submit)
        self.assertIn('--dependency="afterok:$source_array_job_id"', submit)
        self.assertIn('SOURCE_SLURM_ARRAY_JOB_ID="$source_array_job_id"', submit)
        self.assertIn('EXPECTED_GIT_COMMIT="$expected_commit"', submit)
        self.assertIn('CONFIG_SHA256="$config_sha256"', submit)
        self.assertIn('R03_SUMMARY_SHA256="$r03_summary_sha256"', submit)
        self.assertNotRegex(submit, r"(?m)^\s*(python|python3|uv run)\b")

    def test_wrappers_cannot_select_unregistered_modes(self) -> None:
        self.assertIn("R03A_SUBMISSION_MODE=smoke", SUBMIT_SMOKE.read_text(encoding="utf-8"))
        self.assertIn(
            "R03A_SUBMISSION_MODE=h100_smoke",
            SUBMIT_H100_SMOKE.read_text(encoding="utf-8"),
        )
        self.assertIn("R03A_SUBMISSION_MODE=array", SUBMIT_ARRAY.read_text(encoding="utf-8"))
        common = SUBMIT_JOB.read_text(encoding="utf-8")
        self.assertRegex(
            common,
            re.compile(r"case \"\$MODE\" in.*smoke\).*h100_smoke\).*array\)", re.S),
        )


if __name__ == "__main__":
    unittest.main()
