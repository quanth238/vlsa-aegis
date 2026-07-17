"""Dependency-light checks for the R06 capture-only release path."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "main/run_crfs_r06_aegis_capture.py"
WRAPPER = ROOT / "scripts/hpc/run_r06_aegis_capture_canary.sh"
WORKLOAD = ROOT / "scripts/hpc/run_r06_aegis_capture_workload.sh"
SUBMITTER = ROOT / "scripts/hpc/submit_r06_aegis_capture_canary.sh"
SBATCH = ROOT / "slurm/r06_aegis_capture_canary_h100.sbatch"
TERMINAL_EVIDENCE = (
    ROOT / "evidence/r06/aegis-label-capture-canary-20260717a.json"
)
TERMINAL_DECISION = (
    ROOT
    / "docs/decisions/0067-preserve-failed-aegis-capture-and-repair-controller-contract.md"
)
CONFIG = ROOT / "configs/experiments/r06_aegis_collision_conditioned.json"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_cli():
    spec = importlib.util.spec_from_file_location("r06_capture_cli", CLI)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load R06 capture CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R06AegisCaptureHpcContractTest(unittest.TestCase):
    def test_terminal_capture_failure_is_preserved_without_aegis_claim(self) -> None:
        evidence = json.loads(TERMINAL_EVIDENCE.read_text(encoding="utf-8"))
        decision = _text(TERMINAL_DECISION)
        config = json.loads(CONFIG.read_text(encoding="utf-8"))

        self.assertEqual(evidence["status"], "apparatus_inconclusive")
        self.assertEqual(evidence["slurm"]["exact_task_id"], "28409_0")
        self.assertEqual(evidence["slurm"]["state"], "FAILED")
        self.assertEqual(evidence["slurm"]["exit_code"], "1:0")
        self.assertEqual(
            evidence["root_cause"]["robot_identity"][
                "pinned_runtime_expected_live_value"
            ],
            "MountedPanda",
        )
        self.assertEqual(
            evidence["root_cause"]["gripper_current_action"][
                "pinned_runtime_expected_shape_after_boundary_20"
            ],
            [2],
        )
        self.assertFalse(
            evidence["root_cause"]["persisted_live_controller_record_exists"]
        )
        self.assertTrue(
            evidence["root_cause"]["fresh_capture_runtime_confirmation_required"]
        )
        self.assertFalse(evidence["claims"]["aegis_safety_result_exists"])
        self.assertFalse(
            evidence["partial_capture_assets"]["label_freeze_allowed"]
        )
        self.assertFalse(
            evidence["terminal_interpretation"]["aegis_executed"]
        )
        self.assertEqual(evidence["terminal_interpretation"]["qp_steps"], 0)
        self.assertIn("Permanently consume", decision)
        self.assertIn("Do not freeze a Codex label", decision)
        self.assertIs(config["ready_to_run"], False)
        self.assertIsNone(config["execution_release"])

    def test_shell_files_parse_and_entrypoints_are_executable(self) -> None:
        for path in (WRAPPER, WORKLOAD, SUBMITTER, SBATCH):
            result = subprocess.run(
                ["bash", "-n", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=f"{path}: {result.stderr}")
        for path in (WRAPPER, WORKLOAD, SUBMITTER):
            self.assertTrue(path.stat().st_mode & stat.S_IXUSR, path)

    def test_single_worker1_h100_resources_are_64g_not_128g(self) -> None:
        value = _text(SBATCH)
        for token in (
            "#SBATCH --partition=main",
            "#SBATCH --account=normal",
            "#SBATCH --qos=normal",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --mem=64G",
            "#SBATCH --time=00:30:00",
            "#SBATCH --array=0-0%1",
            "#SBATCH --nodelist=worker-1",
            "#SBATCH --no-requeue",
        ):
            self.assertIn(token, value)
        self.assertNotIn("128G", value)
        self.assertNotIn("131072", value)

    def test_capture_workload_runs_pi05_but_never_aegis_or_dino(self) -> None:
        value = _text(WORKLOAD)
        self.assertIn("scripts/serve_policy.py", value)
        self.assertIn("run_crfs_r06_aegis_capture.py", value)
        self.assertIn("--case-index 0", value)
        self.assertIn("validate_aegis_label_capture", value)
        self.assertIn("aegis_execution_allowed == false", value)
        self.assertIn("groundingdino_execution_allowed == false", value)
        self.assertIn("qp_execution_allowed == false", value)
        self.assertIn("robosuite.macros", value)
        self.assertIn("metadata.version('robosuite')", value)
        self.assertIn("R06_ROBOSUITE_IMAGE_CONVENTION", value)
        self.assertIn("R06_ROBOSUITE_VERSION", value)
        self.assertIn('= opengl', value)
        self.assertIn("exact_array_record", value)
        self.assertIn("codex_review_agentview_array_sha256", value)
        for forbidden in (
            "CodexFrozenLabelSafetyCoreProvider",
            "AegisSafetyCore(",
            "groundingdino.util.inference",
            "ZHIPUAI_API_KEY",
            "GLM",
        ):
            self.assertNotIn(forbidden, value)

    def test_exact_manifest_r02_checkpoint_and_normalization_are_bound(self) -> None:
        combined = _text(WRAPPER) + _text(WORKLOAD) + _text(SUBMITTER)
        for digest in (
            "b12319d3fed151bfee85e1615bd72254474a1480308389d747dce954c6131e41",
            "c5be1b4759487f4d6541e49110b90fcf5de404bdaed5f389358db0abe4388f4e",
            "055fcf18781071c6c3575b42a1b44c32a76b474ac1911ad8c5443f78b9c42593",
            "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed",
            "5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a",
            "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
        ):
            self.assertIn(digest, combined)
        self.assertIn("EXPECTED_GIT_COMMIT", combined)
        self.assertIn("EXPECTED_CONFIG_SHA256", combined)
        self.assertIn("SOURCE_CONTRACT", combined)
        self.assertIn("SUBMISSION_RECEIPT", combined)

    def test_submitter_holds_validates_receipts_then_releases_once(self) -> None:
        value = _text(SUBMITTER)
        ordered = (
            'mkdir "$run_root"',
            'mv "$source_tmp" "$source_contract"',
            "sbatch --parsable --hold",
            'mv "$submission_tmp" "$submission"',
            'scontrol release "$job_id"',
        )
        positions = [value.index(token) for token in ordered]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(value.count("sbatch --parsable"), 1)
        self.assertEqual(value.count("scontrol release"), 1)
        self.assertIn('scontrol show job "${job_id}_0"', value)
        self.assertIn('scancel "${job_id}_0"', value)
        self.assertNotIn('scancel "$job_id"', value)
        self.assertNotIn("scancel -u", value)
        self.assertNotIn("rsync --delete", value)
        self.assertIn("accepted_implementation_commit", value)
        self.assertIn("rev-list --parents -n 1", value)
        self.assertIn("allowed_release_diff_paths", value)
        self.assertIn("-release-aegis-label-capture-canary.md", value)

    def test_cli_is_dependency_light_and_requires_capture_only_release(self) -> None:
        module = _load_cli()
        release = {
            "artifact_role": "r06_aegis_codex_label_capture_execution_release",
            "stage": "codex_label_capture",
            "run_id": "r06-capture-test",
            "accepted_implementation_commit": "a" * 40,
            "decision_artifact": (
                "docs/decisions/0066-release-aegis-label-capture-canary.md"
            ),
            "allowed_release_diff_paths": [
                "configs/experiments/r06_aegis_collision_conditioned.json",
                "docs/decisions/0066-release-aegis-label-capture-canary.md",
            ],
            "single_case_index": 0,
            "case_id": "crfs-1069f29a8d76463a",
            "aegis_execution_allowed": False,
            "semantic_label_required": False,
            "groundingdino_execution_allowed": False,
            "qp_execution_allowed": False,
            "robosuite_image_convention": "opengl",
            "robosuite_version": "1.4.1",
            "probe_or_mlp_training_authorized": False,
            "automatic_population_launch_authorized": False,
            "release_only_parent_required": True,
            "resources": {
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
        }
        value = {"ready_to_run": True, "blocked_on": [], "execution_release": release}
        observed = module._require_capture_release(value, run_id="r06-capture-test")
        self.assertEqual(observed, release)
        for mutation in (
            ("stage", "paired_codex_label_canary"),
            ("aegis_execution_allowed", True),
            ("groundingdino_execution_allowed", True),
            ("qp_execution_allowed", True),
        ):
            changed = json.loads(json.dumps(value))
            changed["execution_release"][mutation[0]] = mutation[1]
            with self.assertRaises(ValueError):
                module._require_capture_release(changed, run_id="r06-capture-test")

    def test_cli_calls_exact_capture_api_and_closes_runtime(self) -> None:
        value = _text(CLI)
        self.assertIn("run_aegis_label_capture(", value)
        self.assertIn("SafeLiberoAegisRuntime(safe_case)", value)
        self.assertIn("WebsocketClientPolicy(args.host, args.port)", value)
        self.assertIn("finally:", value)
        self.assertIn("runtime.close()", value)
        self.assertIn("_validate_live_release(", value)
        self.assertIn('"rev-list", "--parents", "-n", "1"', value)
        self.assertIn('"diff", "--name-only"', value)
        self.assertNotIn("run_aegis_case(", value)
        self.assertNotIn("CodexFrozenLabelSafetyCoreProvider", value)


if __name__ == "__main__":
    unittest.main()
