from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm"


class AegisSlurmContractTest(unittest.TestCase):
    def test_all_shell_files_parse(self) -> None:
        paths = sorted(SLURM.glob("*.sh")) + sorted(SLURM.glob("*.sbatch"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path.name):
                subprocess.run(
                    ["bash", "-n", str(path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

    def test_population_arrays_are_grouped_and_throttled_to_two(self) -> None:
        for name in (
            "aegis_capture_population.sbatch",
            "aegis_population.sbatch",
        ):
            text = (SLURM / name).read_text(encoding="utf-8")
            with self.subTest(path=name):
                self.assertIn("#SBATCH --array=0-31%2", text)
                self.assertIn("#SBATCH --exclude=worker-3", text)
                self.assertNotIn("#SBATCH --nodelist=", text)
                self.assertLessEqual(
                    int(re.search(r"--cpus-per-task=(\d+)", text).group(1))
                    * 2,
                    16,
                )
                self.assertLessEqual(
                    int(re.search(r"--mem=(\d+)G", text).group(1)) * 2,
                    256,
                )

    def test_canaries_are_single_task_single_concurrency(self) -> None:
        for name in (
            "aegis_capture_canary.sbatch",
            "aegis_paired_canary.sbatch",
        ):
            text = (SLURM / name).read_text(encoding="utf-8")
            with self.subTest(path=name):
                self.assertIn("#SBATCH --array=0-0%1", text)
                self.assertIn("#SBATCH --exclude=worker-3", text)
                self.assertNotIn("#SBATCH --nodelist=", text)

    def test_orchestration_never_submits_or_cancels_jobs(self) -> None:
        for path in sorted(SLURM.iterdir()):
            if not path.is_file():
                continue
            commands = []
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    commands.append(stripped)
            joined = "\n".join(commands)
            with self.subTest(path=path.name):
                self.assertNotRegex(joined, r"(^|[;&|]\s*)sbatch(\s|$)")
                self.assertNotRegex(joined, r"(^|[;&|]\s*)scancel(\s|$)")

    def test_capture_runtime_has_no_policy_or_safety_execution(self) -> None:
        text = (SLURM / "run_aegis_capture.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("aegis_start_pi05_server", text)
        self.assertNotRegex(text, r"(^|\s)--(?:dino|arm|host|port)(?:\s|$)")
        self.assertIn("POLICY_MODEL_EXECUTION_ALLOWED=false", text)
        self.assertIn("GROUNDING_EXECUTION_ALLOWED=false", text)
        self.assertIn("SAFETY_QP_EXECUTION_ALLOWED=false", text)
        self.assertIn("aegis_run_preflight capture", text)
        self.assertIn('--output-dir "$CAPTURE_OUTPUT_ROOT"', text)
        self.assertIn('"${CASE_ARGUMENTS[@]}"', text)

    def test_evaluation_uses_one_jax_server_for_both_exact_arms(self) -> None:
        runtime = (SLURM / "run_aegis_evaluation.sh").read_text(
            encoding="utf-8"
        )
        common = (SLURM / "aegis_runtime_common.sh").read_text(
            encoding="utf-8"
        )
        self.assertEqual(runtime.count("aegis_start_pi05_server"), 1)
        canary_order = (
            '"diagnostics-off|pi05_translational"',
            '"diagnostics-on|pi05_translational"',
            '"diagnostics-off|pi05_plus_aegis_translational"',
            '"diagnostics-on|pi05_plus_aegis_translational"',
        )
        positions = [runtime.index(item) for item in canary_order]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('for RUN_SPEC in "${RUN_SPECS[@]}"', runtime)
        self.assertIn("--failure-diagnostics", runtime)
        self.assertIn('"${CASE_ARGUMENTS[@]}"', runtime)
        self.assertIn("--mode", runtime)
        self.assertGreaterEqual(
            runtime.count('--labels "$LABEL_MANIFEST_PATH"'), 2
        )
        self.assertIn("--profile", common)
        self.assertIn("evaluation", common)
        self.assertIn(
            "/mnt/data/quanth/cache/openpi/openpi-assets/checkpoints/pi05_libero",
            common,
        )
        self.assertIn("scripts/serve_policy.py", common)
        self.assertIn("/healthz", common)
        self.assertIn("ArrayTaskThrottle=", common)
        self.assertIn("SLURM_ARRAY_TASK_COUNT", common)
        self.assertIn("trap - EXIT INT TERM", common)
        self.assertIn("aegis_stop_pi05_server", common)
        self.assertIn("LIBERO_CONFIG_PATH", common)
        self.assertIn("export MUJOCO_GL=osmesa", common)
        self.assertIn("export PYOPENGL_PLATFORM=osmesa", common)
        self.assertIn("export IMAGEIO_FFMPEG_EXE=", common)
        self.assertIn(
            "700073daef5c23bbcb18c2eae60553a454a5221ec19b4a88c8c367a664671a7c",
            common,
        )
        self.assertNotIn("export MUJOCO_GL=egl", common)
        self.assertIn("paired-canary-validation.json", runtime)
        self.assertIn("--expected-pi05-tree-sha256", common)
        self.assertIn("--pi05-hash-receipt", common)

    def test_checkpoint_hash_and_population_publisher_are_allocation_only(
        self,
    ) -> None:
        hash_batch = (
            SLURM / "aegis_pi05_hash_receipt.sbatch"
        ).read_text(encoding="utf-8")
        hash_runner = (
            SLURM / "run_pi05_hash_receipt.sh"
        ).read_text(encoding="utf-8")
        publisher_batch = (
            SLURM / "aegis_population_publisher.sbatch"
        ).read_text(encoding="utf-8")
        publisher_retry_batch = (
            SLURM / "aegis_population_publisher_retry.sbatch"
        ).read_text(encoding="utf-8")
        publisher = (
            SLURM / "run_aegis_population_publisher.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-0%1", hash_batch)
        self.assertNotIn("#SBATCH --gres=gpu", hash_batch)
        self.assertIn("SLURM_JOB_ID", hash_runner)
        self.assertIn("compute_pi05_tree_receipt.py", hash_runner)
        self.assertNotIn("#SBATCH --gres=gpu", publisher_batch)
        self.assertNotIn("#SBATCH --gres=gpu", publisher_retry_batch)
        self.assertIn("#SBATCH --cpus-per-task=4", publisher_retry_batch)
        self.assertIn("#SBATCH --mem=32G", publisher_retry_batch)
        self.assertIn("#SBATCH --exclude=worker-3", publisher_retry_batch)
        self.assertIn("--dependency=afterany:", publisher)
        self.assertIn(
            ': "${GROUNDINGDINO_DEVICE:?set the exact paired-canary device',
            publisher,
        )
        self.assertIn("SLURM_JOB_DEPENDENCY", publisher)
        self.assertIn(
            'afterany:$POPULATION_ARRAY_JOB_ID',
            publisher,
        )
        self.assertIn("population-prepublish", publisher)
        self.assertIn("--format=JobID,JobIDRaw,State", publisher)
        self.assertIn("aggregate_safelibero_aegis.py", publisher)
        self.assertIn("build_aegis_failure_report.py", publisher)
        self.assertIn("--population-validation-receipt", publisher)
        self.assertIn("--failure-cases", publisher)
        self.assertIn("--failure-report", publisher)
        self.assertIn("--failure-markdown", publisher)
        self.assertIn("build_safelibero_video_gallery.py", publisher)
        self.assertIn("population-finalize", publisher)
        self.assertIn("POPULATION_RUNTIME_REPO", publisher)
        self.assertIn("PUBLISHER_RELEASE_REPO", publisher)
        self.assertIn("EXPECTED_PUBLISHER_GIT_COMMIT", publisher)
        self.assertIn("build_aegis_publisher_retry_authority.py", publisher)
        self.assertNotIn("scripts/serve_policy.py", publisher)
        self.assertNotIn("evaluate_safelibero_aegis.py", publisher)

    def test_population_contract_requires_a_validated_canary_receipt(
        self,
    ) -> None:
        preparer = (SLURM / "prepare_aegis_run_root.sh").read_text(
            encoding="utf-8"
        )
        common = (SLURM / "aegis_runtime_common.sh").read_text(
            encoding="utf-8"
        )
        for text in (preparer, common):
            self.assertIn("PI05_HASH_RECEIPT_PATH", text)
            self.assertIn("vlsa_table1_pi05_hash_receipt.v1", text)
            self.assertIn("LABEL_PUBLICATION_RECEIPT_PATH", text)
            self.assertIn(
                "e83611f46ce5fbb13c84f74db3825ab114bf7184db96b62be2965c7a0c5b9e20",
                text,
            )
            self.assertIn("PAIRED_CANARY_RECEIPT_PATH", text)
            self.assertIn(
                "vlsa_table1_action_invariant_paired_canary_validation.v2",
                text,
            )
            self.assertIn(
                "vlsa_table1_active_obstacle_contacts.v3",
                text,
            )
            self.assertIn(
                "vlsa_table1_contact_model_authority.v2",
                text,
            )
            self.assertIn("full_content_tree_sha256", text)
        self.assertIn("EXPECTED_PI05_HASH_RECEIPT_SHA256", preparer)
        self.assertIn(
            "EXPECTED_PAIRED_CANARY_RECEIPT_SHA256",
            preparer,
        )
        self.assertNotIn("structural-and-metadata-only", preparer)
        self.assertNotIn("structural-and-metadata-only", common)

    def test_evaluation_groundingdino_device_is_frozen_to_cpu(self) -> None:
        preparer = (SLURM / "prepare_aegis_run_root.sh").read_text(
            encoding="utf-8"
        )
        common = (SLURM / "aegis_runtime_common.sh").read_text(
            encoding="utf-8"
        )
        publisher = (
            SLURM / "run_aegis_population_publisher.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '[[ "$GROUNDINGDINO_DEVICE" == cpu ]]',
            preparer,
        )
        self.assertIn(
            '[[ "$GROUNDINGDINO_DEVICE" == cpu ]]',
            common,
        )
        self.assertIn(
            '[[ "$GROUNDINGDINO_DEVICE" == cpu ]]',
            publisher,
        )
        for text in (preparer, common, publisher):
            self.assertNotIn("cpu|cuda", text)
            self.assertNotIn("cpu or cuda", text)

    def test_shell_receipt_parser_ignores_nested_status_fields(self) -> None:
        common_path = SLURM / "aegis_runtime_common.sh"
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = Path(temporary) / "receipt.json"
            receipt_path.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "vlsa_table1_action_invariant_"
                            "paired_canary_validation.v2"
                        ),
                        "status": "validated",
                        "results": [
                            {"status": "collision"},
                            {"status": "success"},
                        ],
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    (
                        'source "$1"; '
                        'aegis_json_top_level_string_value "$2" status'
                    ),
                    "_",
                    str(common_path),
                    str(receipt_path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        self.assertEqual(completed.stdout.strip(), "validated")

    def test_outputs_are_confined_to_immutable_experiment_root(self) -> None:
        common = (SLURM / "aegis_runtime_common.sh").read_text(
            encoding="utf-8"
        )
        preparer = (SLURM / "prepare_aegis_run_root.sh").read_text(
            encoding="utf-8"
        )
        prefix = "/mnt/data/quanth/experiments/"
        self.assertIn(prefix, common)
        self.assertIn(prefix, preparer)
        self.assertIn('mkdir "$RUN_ROOT" ||', preparer)
        self.assertIn("case_ordinal", preparer)
        self.assertIn("CONTRACT_CASE_ORDINAL", common)
        self.assertIn("immutable run root already exists", preparer)
        self.assertIn('mkdir "$TASK_ROOT" ||', common)
        self.assertIn("immutable task root already exists", common)

    def test_wrappers_match_capture_and_evaluator_help_contracts(self) -> None:
        capture_help = subprocess.run(
            [
                sys.executable,
                str(ROOT / "main/capture_safelibero_labels.py"),
                "--help",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"PYTHONDONTWRITEBYTECODE": "1"},
        ).stdout
        evaluator_help = subprocess.run(
            [
                sys.executable,
                str(ROOT / "main/evaluate_safelibero_aegis.py"),
                "--help",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"PYTHONDONTWRITEBYTECODE": "1"},
        ).stdout
        for flag in ("--manifest", "--output-dir", "--case-ordinal", "--repo-root"):
            self.assertIn(flag, capture_help)
            self.assertIn(flag, evaluator_help)
        for flag in (
            "--mode",
            "--labels",
            "--host",
            "--port",
            "--groundingdino-config",
            "--groundingdino-checkpoint",
            "--groundingdino-device",
        ):
            self.assertIn(flag, evaluator_help)


if __name__ == "__main__":
    unittest.main()
