from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = (
    ROOT
    / "scripts"
    / "submit_vlsa_postpublication_analysis_v2_28610.sh"
)
RUNNER = ROOT / "slurm" / "run_vlsa_postpublication_analysis_v2.sh"
SBATCH = ROOT / "slurm" / "vlsa_postpublication_analysis_v2.sbatch"
BUILDER = ROOT / "analysis" / "build_safelibero_postpublication_v2.py"

RUN_ID = "vlsa-table1-contact-authority-population-20260718a"
SOURCE_COMMIT = "1592aa59361f431ba96c6ddcbebcb596f6c20853"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


class LaunchFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="vlsa-analysis-v2-launch-"
        )
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.experiment_root = self.root / "experiments"
        self.output_root = self.root / "analysis-output"
        self.fake_bin = self.root / "fake-bin"
        self.fake_state = self.root / "fake-state"
        for path in (
            self.repo,
            self.experiment_root,
            self.output_root,
            self.fake_bin,
            self.fake_state,
        ):
            path.mkdir()

        self.run_root = self.experiment_root / RUN_ID
        self.results_root = self.run_root / "tasks"
        self.publisher_root = (
            self.run_root / "publication-attempts" / "job-28610"
        )
        self.results_root.mkdir(parents=True)
        self.publisher_root.mkdir(parents=True)

        self._install_release_files()
        self._install_immutable_inputs()
        self._commit_release()
        self._install_fake_slurm()
        self.environment = self._environment()

    def cleanup(self) -> None:
        self.temporary.cleanup()

    @property
    def submit_path(self) -> Path:
        return (
            self.repo
            / "scripts"
            / "submit_vlsa_postpublication_analysis_v2_28610.sh"
        )

    @property
    def output_dir(self) -> Path:
        return self.output_root / f"{RUN_ID}-publisher-28610"

    @property
    def control_dir(self) -> Path:
        return (
            self.output_root
            / ".analysis-v2-control"
            / f"{RUN_ID}-publisher-28610"
        )

    @property
    def command_log(self) -> Path:
        return self.fake_state / "commands.log"

    def _install_release_files(self) -> None:
        replacements = (
            (
                "/mnt/data/quanth/experiments/vlsa-aegis-table1-analysis-v2",
                str(self.output_root),
            ),
            (
                "/mnt/data/quanth/experiments/vlsa-aegis-table1",
                str(self.experiment_root),
            ),
            (
                "/home/quanth/working_space/vlsa-aegis-table-repro",
                str(self.repo),
            ),
            (
                "/mnt/data/quanth/venvs/safety_vla/main/bin/python",
                str(self.fake_bin / "analysis-python"),
            ),
        )
        submit_text = SUBMIT.read_text(encoding="utf-8")
        runner_text = RUNNER.read_text(encoding="utf-8")
        for source, target in replacements:
            submit_text = submit_text.replace(source, target)
            runner_text = runner_text.replace(source, target)

        files = {
            "scripts/submit_vlsa_postpublication_analysis_v2_28610.sh": (
                submit_text
            ),
            "slurm/run_vlsa_postpublication_analysis_v2.sh": (
                runner_text
            ),
            "slurm/vlsa_postpublication_analysis_v2.sbatch": (
                SBATCH.read_text(encoding="utf-8")
            ),
            "analysis/build_safelibero_postpublication_v2.py": (
                BUILDER.read_text(encoding="utf-8")
            ),
            "configs/vlsa_table1_translational.json": '{"config":true}\n',
            "manifests/vlsa_table1_population.jsonl": '{"case":0}\n',
            "manifests/vlsa_table1_population.receipt.json": (
                '{"receipt":true}\n'
            ),
        }
        for relative, text in files.items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.submit_path.chmod(0o755)
        (self.repo / "slurm/run_vlsa_postpublication_analysis_v2.sh").chmod(
            0o755
        )
        (
            self.repo / "slurm/vlsa_postpublication_analysis_v2.sbatch"
        ).chmod(0o755)

    def _install_immutable_inputs(self) -> None:
        self.publication = (
            self.run_root / "population-publication-receipt.json"
        )
        self.summary = self.publisher_root / "population-summary.json"
        self.prepublish = (
            self.publisher_root / "prepublish-validation.json"
        )
        self.publication.write_text('{"published":true}\n', encoding="utf-8")
        self.summary.write_text('{"summary":true}\n', encoding="utf-8")
        self.prepublish.write_text(
            '{"validated":true}\n', encoding="utf-8"
        )

    def _commit_release(self) -> None:
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "add", "."], check=True
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repo),
                "-c",
                "user.name=Analysis Launch Test",
                "-c",
                "user.email=analysis-launch@example.invalid",
                "commit",
                "-qm",
                "Freeze analysis launcher fixture",
            ],
            check=True,
        )
        self.commit = subprocess.run(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    def _install_fake_slurm(self) -> None:
        write_executable(
            self.fake_bin / "sacct",
            """#!/usr/bin/env bash
set -euo pipefail
publisher_id=${FAKE_PUBLISHER_ID:-28610}
if [[ " $* " == *" -j $publisher_id "* ]]; then
  printf '%s|%s|%s\\n' "$publisher_id" "${FAKE_PUBLISHER_STATE:-COMPLETED}" "${FAKE_PUBLISHER_EXIT:-0:0}"
else
  printf '9002|%s|0:0\\n' "${FAKE_JOB_STATE:-PENDING}"
fi
""",
        )
        write_executable(
            self.fake_bin / "squeue",
            """#!/usr/bin/env bash
set -euo pipefail
if [[ -f "$FAKE_STATE_DIR/recover-job" ]]; then
  printf '9002\\n'
fi
""",
        )
        write_executable(
            self.fake_bin / "sbatch",
            """#!/usr/bin/env bash
set -euo pipefail
printf 'sbatch %s\\n' "$*" >>"$FAKE_STATE_DIR/commands.log"
printf '9002\\n'
""",
        )
        write_executable(
            self.fake_bin / "scontrol",
            """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == show ]]; then
  if [[ -f "$FAKE_STATE_DIR/released" ]]; then
    state=PENDING
    reason=Dependency
  else
    state=PENDING
    reason=JobHeldUser
  fi
  job_name=${FAKE_JOB_NAME:-vlsa-a2-p28610}
  printf 'JobId=9002 JobName=%s JobState=%s Reason=%s Dependency=%s Partition=main Account=normal QOS=normal NumNodes=1 NumCPUs=%s NumTasks=1 CPUs/Task=%s MinMemoryNode=%s TimeLimit=%s Requeue=0 ExcNodeList=%s Command=%s WorkDir=%s StdOut=/mnt/data/quanth/slurm_logs/%s-9002.out ReqTRES=cpu=4,mem=32G,node=1,billing=4 AllocTRES=(null) TresPerNode=(null) TresPerTask=cpu=4 Gres=(null)\\n' \
    "$job_name" "$state" "$reason" "${FAKE_DEPENDENCY:-afterok:28610}" \
    "${FAKE_CPUS:-4}" "${FAKE_CPUS:-4}" "${FAKE_MEMORY:-32G}" \
    "${FAKE_TIME:-04:00:00}" "${FAKE_EXCLUDE:-worker-3}" \
    "$FAKE_SBATCH_PATH" "$FAKE_REMOTE_REPO" "$job_name"
elif [[ "$1" == release ]]; then
  printf 'release %s\\n' "$2" >>"$FAKE_STATE_DIR/commands.log"
  if [[ -f "$FAKE_STATE_DIR/fail-release-once" ]]; then
    rm -f "$FAKE_STATE_DIR/fail-release-once"
    exit 41
  fi
  : >"$FAKE_STATE_DIR/released"
else
  exit 2
fi
""",
        )
        # macOS has no flock; this no-op is sufficient for a single-process
        # fixture and leaves the production locking code exercised.
        write_executable(
            self.fake_bin / "flock",
            "#!/usr/bin/env bash\nexit 0\n",
        )
        write_executable(
            self.fake_bin / "analysis-python",
            """#!/usr/bin/env bash
set -euo pipefail
printf 'analysis-python %s\\n' "$*" >>"$FAKE_STATE_DIR/commands.log"
""",
        )

    def _environment(self) -> dict[str, str]:
        runner = (
            self.repo / "slurm/run_vlsa_postpublication_analysis_v2.sh"
        )
        sbatch = self.repo / "slurm/vlsa_postpublication_analysis_v2.sbatch"
        builder = (
            self.repo / "analysis/build_safelibero_postpublication_v2.py"
        )
        config = self.repo / "configs/vlsa_table1_translational.json"
        manifest = self.repo / "manifests/vlsa_table1_population.jsonl"
        manifest_receipt = (
            self.repo / "manifests/vlsa_table1_population.receipt.json"
        )
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{self.fake_bin}:{environment['PATH']}",
                "USER": "launch-test",
                "FAKE_STATE_DIR": str(self.fake_state),
                "FAKE_SBATCH_PATH": str(sbatch),
                "FAKE_REMOTE_REPO": str(self.repo),
                "RUN_ID": RUN_ID,
                "EXPECTED_SOURCE_GIT_COMMIT": SOURCE_COMMIT,
                "EXPECTED_PUBLICATION_RECEIPT_SHA256": digest(
                    self.publication
                ),
                "EXPECTED_ANALYSIS_GIT_COMMIT": self.commit,
                "EXPECTED_BUILDER_SHA256": digest(builder),
                "EXPECTED_RUNNER_SHA256": digest(runner),
                "EXPECTED_SBATCH_SHA256": digest(sbatch),
                "EXPECTED_SUBMIT_HELPER_SHA256": digest(self.submit_path),
                "EXPECTED_CONFIG_SHA256": digest(config),
                "EXPECTED_MANIFEST_SHA256": digest(manifest),
                "EXPECTED_MANIFEST_RECEIPT_SHA256": digest(
                    manifest_receipt
                ),
                "EXPECTED_V1_SUMMARY_SHA256": digest(self.summary),
                "EXPECTED_PREPUBLISH_RECEIPT_SHA256": digest(
                    self.prepublish
                ),
            }
        )
        return environment

    def run(
        self,
        *,
        overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = dict(self.environment)
        if overrides:
            environment.update(overrides)
        return subprocess.run(
            ["/bin/bash", str(self.submit_path)],
            cwd=self.repo,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    def run_allocation(
        self,
        *,
        overrides: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = dict(self.environment)
        environment.update(
            {
                "SLURM_JOB_ID": "9002",
                "SLURM_JOB_NAME": "vlsa-a2-p28610",
                "SLURM_JOB_DEPENDENCY": "afterok:28610",
                "SLURMD_NODENAME": "worker-1",
                "SLURM_JOB_PARTITION": "main",
                "SLURM_CPUS_PER_TASK": "4",
                "SLURM_MEM_PER_NODE": "32768",
                "CUDA_VISIBLE_DEVICES": "NoDevFiles",
            }
        )
        if overrides:
            environment.update(overrides)
        runner = (
            self.repo / "slurm/run_vlsa_postpublication_analysis_v2.sh"
        )
        return subprocess.run(
            ["/bin/bash", str(runner)],
            cwd=self.repo,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )


class PostpublicationAnalysisV2LaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = LaunchFixture()

    def tearDown(self) -> None:
        self.fixture.cleanup()

    def test_00_exact_static_contract_and_shell_only_control_plane(self) -> None:
        for path in (SUBMIT, RUNNER, SBATCH):
            subprocess.run(
                ["/bin/bash", "-n", str(path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        submit = SUBMIT.read_text(encoding="utf-8")
        runner = RUNNER.read_text(encoding="utf-8")
        sbatch = SBATCH.read_text(encoding="utf-8")

        for text in (submit, runner):
            self.assertIn("readonly POPULATION_ARRAY_JOB_ID=28609", text)
            self.assertIn(
                "readonly PUBLISHER_JOB_ID=${PUBLISHER_JOB_ID:-28610}",
                text,
            )
            self.assertIn("ARTIFACT_PUBLISHER_JOB_ID", text)
            self.assertIn("TIMEOUT_ARTIFACT_PUBLISHER_JOB_ID=28940", text)
            self.assertIn(RUN_ID, text)
            self.assertIn(SOURCE_COMMIT, text)
            self.assertIn("afterok:$PUBLISHER_JOB_ID", text)
        for directive in (
            "#SBATCH --partition=main",
            "#SBATCH --account=normal",
            "#SBATCH --qos=normal",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --cpus-per-task=4",
            "#SBATCH --mem=32G",
            "#SBATCH --time=04:00:00",
            "#SBATCH --exclude=worker-3",
            "#SBATCH --no-requeue",
        ):
            self.assertIn(directive, sbatch)
        self.assertNotIn("#SBATCH --gres", sbatch)
        self.assertNotIn("#SBATCH --gpus", sbatch)
        self.assertNotIn("#SBATCH --array", sbatch)
        self.assertIn("sbatch --parsable --hold", submit)
        self.assertIn('scontrol release "$job_id"', submit)
        self.assertIn("scontrol show job -dd -o", submit)
        self.assertIn("Reason JobHeldUser", submit)
        self.assertIn("sacct -n -X -j", submit)
        self.assertIn("COMPLETED|0:0", submit)
        self.assertNotIn("--dependency=afterany:", submit)
        self.assertNotIn("--export=ALL", submit)
        self.assertNotIn("python ", submit.lower())
        for forbidden in ("scancel", "pkill", "rm -rf"):
            self.assertNotIn(forbidden, submit)
        self.assertIn(
            "build_safelibero_postpublication_v2.py", runner
        )
        for argument in (
            "--config",
            "--manifest-receipt",
            "--manifest",
            "--results-root",
            "--v1-publication-receipt",
            "--v1-summary",
            "--prepublish-receipt",
            "--output-root",
        ):
            self.assertIn(argument, runner)
        self.assertIn("analysis-v2 output directory already exists", runner)

    def test_10_happy_path_is_held_inspected_receipted_then_released(
        self,
    ) -> None:
        result = self.fixture.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "9002")
        commands = self.fixture.command_log.read_text(encoding="utf-8")
        self.assertEqual(commands.count("sbatch "), 1)
        self.assertEqual(commands.count("release 9002"), 1)
        self.assertLess(commands.index("sbatch "), commands.index("release "))
        self.assertFalse(self.fixture.output_dir.exists())

        expected_pairs = (
            ("submission-intent.tsv", "submission-intent.sha256"),
            ("job-id.txt", "job-id.sha256"),
            ("held-job-scontrol.txt", "held-job-scontrol.sha256"),
            ("submission-receipt.tsv", "submission-receipt.sha256"),
            ("release-receipt.tsv", "release-receipt.sha256"),
        )
        for receipt_name, sidecar_name in expected_pairs:
            receipt = self.fixture.control_dir / receipt_name
            sidecar = self.fixture.control_dir / sidecar_name
            self.assertTrue(receipt.is_file())
            self.assertFalse(receipt.is_symlink())
            self.assertTrue(sidecar.is_file())
            line = sidecar.read_text(encoding="utf-8").strip()
            self.assertEqual(
                line,
                f"{digest(receipt)}  {receipt.name}",
            )
        submission = (
            self.fixture.control_dir / "submission-receipt.tsv"
        ).read_text(encoding="utf-8")
        self.assertIn("status\theld_validated\n", submission)
        self.assertIn("cpu_only\ttrue\n", submission)
        self.assertIn("output_unused\ttrue\n", submission)
        release = (
            self.fixture.control_dir / "release-receipt.tsv"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "release_action\treleased_exact_held_job\n", release
        )

    def test_20_nonterminal_publisher_cannot_create_a_job(self) -> None:
        result = self.fixture.run(
            overrides={"FAKE_PUBLISHER_STATE": "RUNNING"}
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not exactly COMPLETED", result.stderr)
        self.assertFalse(self.fixture.command_log.exists())

    def test_22_timeout_recovery_separates_terminal_and_artifact_jobs(
        self,
    ) -> None:
        timeout_root = (
            self.fixture.run_root / "publication-attempts" / "job-28940"
        )
        timeout_root.mkdir(parents=True)
        shutil.copy2(
            self.fixture.summary,
            timeout_root / "population-summary.json",
        )
        shutil.copy2(
            self.fixture.prepublish,
            timeout_root / "prepublish-validation.json",
        )
        overrides = {
            "PUBLISHER_JOB_ID": "9003",
            "ARTIFACT_PUBLISHER_JOB_ID": "28940",
            "FAKE_PUBLISHER_ID": "9003",
            "FAKE_JOB_NAME": "vlsa-a2-p9003",
            "FAKE_DEPENDENCY": "afterok:9003",
        }
        result = self.fixture.run(overrides=overrides)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.fixture.command_log.read_text(encoding="utf-8")
        self.assertIn("--dependency=afterok:9003", commands)
        control_dir = (
            self.fixture.output_root
            / ".analysis-v2-control"
            / f"{RUN_ID}-publisher-9003"
        )
        receipt = (control_dir / "submission-receipt.tsv").read_text(
            encoding="utf-8"
        )
        self.assertIn("publisher_job_id\t9003\n", receipt)
        self.assertIn("artifact_publisher_job_id\t28940\n", receipt)

        allocation = self.fixture.run_allocation(
            overrides={
                **overrides,
                "SLURM_JOB_NAME": "vlsa-a2-p9003",
                "SLURM_JOB_DEPENDENCY": "afterok:9003",
            }
        )
        self.assertEqual(allocation.returncode, 0, allocation.stderr)

        rejected = LaunchFixture()
        try:
            failed = rejected.run(
                overrides={
                    "PUBLISHER_JOB_ID": "9003",
                    "ARTIFACT_PUBLISHER_JOB_ID": "28941",
                }
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("timed-out artifact publisher 28940", failed.stderr)
            self.assertFalse(rejected.command_log.exists())
        finally:
            rejected.cleanup()

    def test_25_dependency_and_resource_mutations_fail_before_release(
        self,
    ) -> None:
        for field, value, message in (
            ("FAKE_DEPENDENCY", "afterany:28610", "dependency differs"),
            ("FAKE_CPUS", "8", "NumCPUs differs"),
            ("FAKE_MEMORY", "64G", "MinMemoryNode differs"),
            ("FAKE_EXCLUDE", "(null)", "ExcNodeList differs"),
        ):
            with self.subTest(field=field):
                fixture = LaunchFixture()
                try:
                    result = fixture.run(overrides={field: value})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    commands = fixture.command_log.read_text(
                        encoding="utf-8"
                    )
                    self.assertEqual(commands.count("sbatch "), 1)
                    self.assertNotIn("release ", commands)
                finally:
                    fixture.cleanup()

    def test_30_input_mutation_and_output_reuse_fail_closed(self) -> None:
        original = self.fixture.publication.read_bytes()
        self.fixture.publication.write_bytes(original + b"mutation")
        result = self.fixture.run()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("publication_receipt SHA-256 changed", result.stderr)
        self.assertFalse(self.fixture.command_log.exists())

        fixture = LaunchFixture()
        try:
            fixture.output_dir.mkdir()
            result = fixture.run()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "analysis-v2 output directory already exists", result.stderr
            )
            self.assertFalse(fixture.command_log.exists())
        finally:
            fixture.cleanup()

    def test_35_allocation_rechecks_dependency_hashes_and_unused_output(
        self,
    ) -> None:
        result = self.fixture.run_allocation()
        self.assertEqual(result.returncode, 0, result.stderr)
        command = self.fixture.command_log.read_text(encoding="utf-8")
        self.assertIn("analysis-python ", command)
        for argument in (
            "--config",
            "--manifest-receipt",
            "--manifest",
            "--results-root",
            "--v1-publication-receipt",
            "--v1-summary",
            "--prepublish-receipt",
            "--output-root",
        ):
            self.assertIn(argument, command)

        for override, message in (
            (
                {"SLURM_JOB_DEPENDENCY": "afterany:28610"},
                "dependency must be exactly",
            ),
            ({"SLURM_CPUS_PER_TASK": "8"}, "requires exactly 4 CPUs"),
            ({"SLURM_MEM_PER_NODE": "65536"}, "32768 MiB"),
            ({"CUDA_VISIBLE_DEVICES": "0"}, "must not receive CUDA"),
            ({"SLURMD_NODENAME": "worker-3"}, "cannot execute"),
        ):
            with self.subTest(override=override):
                fixture = LaunchFixture()
                try:
                    failed = fixture.run_allocation(overrides=override)
                    self.assertNotEqual(failed.returncode, 0)
                    self.assertIn(message, failed.stderr)
                    self.assertFalse(fixture.command_log.exists())
                finally:
                    fixture.cleanup()

        fixture = LaunchFixture()
        try:
            fixture.summary.write_bytes(
                fixture.summary.read_bytes() + b"mutation"
            )
            failed = fixture.run_allocation()
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("v1_summary SHA-256 changed", failed.stderr)
            self.assertFalse(fixture.command_log.exists())
        finally:
            fixture.cleanup()

        fixture = LaunchFixture()
        try:
            fixture.output_dir.mkdir()
            failed = fixture.run_allocation()
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn(
                "analysis-v2 output directory already exists",
                failed.stderr,
            )
            self.assertFalse(fixture.command_log.exists())
        finally:
            fixture.cleanup()

    def test_40_recovery_reuses_job_and_accepts_started_output(self) -> None:
        (self.fixture.fake_state / "fail-release-once").touch()
        first = self.fixture.run()
        self.assertNotEqual(first.returncode, 0)
        self.assertTrue(
            (self.fixture.control_dir / "submission-receipt.tsv").is_file()
        )
        self.assertFalse(
            (self.fixture.control_dir / "release-receipt.tsv").exists()
        )

        second = self.fixture.run()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(second.stdout.strip(), "9002")
        commands = self.fixture.command_log.read_text(encoding="utf-8")
        self.assertEqual(commands.count("sbatch "), 1)
        self.assertEqual(commands.count("release 9002"), 2)

        # Simulate the released allocation having started and created its
        # receipt-last output directory.  A control-plane rerun must validate
        # the immutable release receipt without submitting another job.
        self.fixture.output_dir.mkdir()
        third = self.fixture.run()
        self.assertEqual(third.returncode, 0, third.stderr)
        self.assertEqual(third.stdout.strip(), "9002")
        commands_after = self.fixture.command_log.read_text(
            encoding="utf-8"
        )
        self.assertEqual(commands_after, commands)

    def test_45_sidecar_mutation_breaks_recovery(self) -> None:
        first = self.fixture.run()
        self.assertEqual(first.returncode, 0, first.stderr)
        sidecar = self.fixture.control_dir / "submission-intent.sha256"
        sidecar.chmod(0o644)
        sidecar.write_text(
            f"{'0' * 64}  submission-intent.tsv\n", encoding="utf-8"
        )
        second = self.fixture.run()
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("immutable file differs", second.stderr)
        commands = self.fixture.command_log.read_text(encoding="utf-8")
        self.assertEqual(commands.count("sbatch "), 1)


if __name__ == "__main__":
    unittest.main()
