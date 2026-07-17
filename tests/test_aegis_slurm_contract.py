from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
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
        self.assertIn(
            "for ARM in pi05_translational pi05_plus_aegis_translational",
            runtime,
        )
        self.assertIn('"${CASE_ARGUMENTS[@]}"', runtime)
        self.assertIn("--mode", runtime)
        self.assertEqual(runtime.count('--labels "$LABEL_MANIFEST_PATH"'), 2)
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
        self.assertNotIn("export MUJOCO_GL=egl", common)

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
