from __future__ import annotations

from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SLURM = ROOT / "slurm"


class OscArmLinkSlurmContractTests(unittest.TestCase):
    def source(self, name: str) -> str:
        return (SLURM / name).read_text(encoding="utf-8")

    def test_numeric_gate_is_one_bounded_h100_mig_allocation(self) -> None:
        source = self.source("poisson_osc_arm_link_numeric.sbatch")
        self.assertIn("#SBATCH --partition=mig", source)
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("#SBATCH --cpus-per-task=4", source)
        self.assertIn("#SBATCH --mem=32G", source)
        self.assertIn("#SBATCH --no-requeue", source)
        self.assertNotIn("#SBATCH --time", source)
        self.assertIn("CUDA_VISIBLE_DEVICES", source)
        self.assertIn("exactly one visible GPU", source)
        self.assertIn('[[ "${SLURM_JOB_PARTITION}" != "mig" ]]', source)

    def test_numeric_gate_freezes_source_runtime_and_artifact_roots(self) -> None:
        source = self.source("poisson_osc_arm_link_numeric.sbatch")
        self.assertIn(
            'REGISTERED_REMOTE_REPO="/home/quanth/working_space/'
            'vlsa-aegis-poisson-arm-link-study"',
            source,
        )
        self.assertIn(
            'REGISTERED_RESULT_ROOT="/mnt/data/quanth/experiments/'
            'vlsa-aegis-poisson-arm-link-study"',
            source,
        )
        self.assertIn(
            'REGISTERED_EVALUATION_PYTHON="/mnt/data/quanth/venvs/'
            'safety_vla/main/bin/python"',
            source,
        )
        self.assertNotIn("${EVALUATION_PYTHON:-", source)
        self.assertIn('[[ -n "$(git status --short)" ]]', source)
        self.assertIn('git rev-parse HEAD', source)
        self.assertIn('[[ -e "${RUN_ROOT}" || -L "${RUN_ROOT}" ]]', source)
        self.assertIn('require_canonical_real_directory "numeric run root"', source)
        self.assertIn('require_canonical_real_file "immutable numeric result"', source)

    def test_numeric_gate_uses_frozen_post_osc_module_set_with_zero_skip_runner(
        self,
    ) -> None:
        source = self.source("poisson_osc_arm_link_numeric.sbatch")
        modules = tuple(re.findall(r"--test-module ([A-Za-z0-9_.]+)", source))
        self.assertEqual(
            modules,
            (
                "tests.test_poisson_geometry",
                "tests.test_poisson_field",
                "tests.test_poisson_field_bundle",
                "tests.test_poisson_production_grid",
                "tests.test_poisson_surface_sampling",
                "tests.test_poisson_measurement",
                "tests.test_poisson_jacobians",
                "tests.test_poisson_cbf_qp",
                "tests.test_poisson_prephysics_control_wrapper",
                "tests.test_poisson_post_osc_torque_sensitivity",
                "tests.test_poisson_post_osc_torque_shield",
                "tests.test_poisson_osc_arm_link_canary",
                "tests.test_poisson_osc_target_link_protocol",
                "tests.test_poisson_osc_numeric_prerequisite",
                "tests.test_poisson_osc_arm_link_slurm_contract",
                "tests.test_poisson_numeric_validation_runner",
            ),
        )
        self.assertIn("scripts/run_poisson_numeric_validation.py", source)
        self.assertIn("--minimum-tests 120", source)
        self.assertIn("VLSA_POISSON_RUN_PRODUCTION_GRID_VALIDATION=1", source)
        numeric_runner = (ROOT / "scripts/run_poisson_numeric_validation.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("and not skipped", numeric_runner)
        self.assertIn('"zero_skips": not skipped', numeric_runner)

    def test_all_post_osc_jobs_isolate_python_and_reject_symlinked_paths(self) -> None:
        for name in (
            "poisson_osc_arm_link_numeric.sbatch",
            "poisson_osc_arm_link_canary.sbatch",
            "poisson_osc_arm_link_canary_validate.sbatch",
        ):
            with self.subTest(name=name):
                source = self.source(name)
                self.assertIn("unset PYTHONHOME PYTHONPATH PYTHONSTARTUP", source)
                self.assertIn("export PYTHONNOUSERSITE=1", source)
                self.assertNotIn("${PYTHONPATH:+", source)
                self.assertIn("realpath -e --", source)
                self.assertIn("require_canonical_real_directory", source)

    def test_canary_has_no_long_walltime_and_consumer_remains_bounded(self) -> None:
        producer = self.source("poisson_osc_arm_link_canary.sbatch")
        consumer = self.source("poisson_osc_arm_link_canary_validate.sbatch")
        self.assertNotIn("#SBATCH --time", producer)
        self.assertIn("#SBATCH --time=00:30:00", consumer)
        self.assertIn(
            'OPENPI_PYTHON="/mnt/data/quanth/venvs/openpi/bin/python"', producer
        )
        self.assertIn(
            'EVALUATION_PYTHON="/mnt/data/quanth/venvs/safety_vla/main/bin/python"',
            producer,
        )
        self.assertNotIn("${OPENPI_PYTHON:-", producer)
        self.assertNotIn("${EVALUATION_PYTHON:-", producer)
        self.assertNotIn("${EVALUATION_PYTHON:-", consumer)

    def test_producer_and_consumer_bind_distinct_numeric_allocation(self) -> None:
        for name in (
            "poisson_osc_arm_link_canary.sbatch",
            "poisson_osc_arm_link_canary_validate.sbatch",
        ):
            with self.subTest(name=name):
                source = self.source(name)
                self.assertIn("NUMERIC_VALIDATION_RESULT", source)
                self.assertIn("EXPECTED_NUMERIC_JOB_ID", source)
                self.assertIn("--numeric-validation-result", source)
                self.assertIn("--expected-numeric-job-id", source)
                self.assertIn(
                    "numeric prerequisite is not a direct registered numeric run result",
                    source,
                )

    def test_consumer_can_bind_immutable_producer_and_repaired_consumer_commits(self) -> None:
        source = self.source("poisson_osc_arm_link_canary_validate.sbatch")
        self.assertIn("EXPECTED_PRODUCER_GIT_COMMIT", source)
        self.assertIn("EXPECTED_CONSUMER_GIT_COMMIT", source)
        self.assertNotIn("EXPECTED_GIT_COMMIT", source)
        self.assertIn(
            '--expected-producer-commit "${EXPECTED_PRODUCER_GIT_COMMIT}"',
            source,
        )
        self.assertIn(
            '--expected-consumer-commit "${EXPECTED_CONSUMER_GIT_COMMIT}"',
            source,
        )
        self.assertIn(
            'OUTPUT="${RUN_ROOT}/validation_receipt_schema_v4_target_link.json"',
            source,
        )
        self.assertIn('--expected-case-id "${CASE_ID}"', source)
        self.assertIn(
            '--expected-protocol-relative-path "${PROTOCOL_RELATIVE_PATH}"',
            source,
        )

    def test_producer_and_consumer_require_explicit_case_protocol_binding(self) -> None:
        for name in (
            "poisson_osc_arm_link_canary.sbatch",
            "poisson_osc_arm_link_canary_validate.sbatch",
        ):
            with self.subTest(name=name):
                source = self.source(name)
                self.assertIn('"${PROTOCOL_RELATIVE_PATH}"', source)
                self.assertIn('"${CASE_ID}"', source)
                self.assertNotIn("vlsa_poisson_osc_arm_link_canary.v3.json", source)

    def test_post_osc_slurm_scripts_parse_as_bash(self) -> None:
        for name in (
            "poisson_osc_arm_link_numeric.sbatch",
            "poisson_osc_arm_link_canary.sbatch",
            "poisson_osc_arm_link_canary_validate.sbatch",
        ):
            with self.subTest(name=name):
                subprocess.run(
                    ["bash", "-n", str(SLURM / name)],
                    check=True,
                    cwd=str(ROOT),
                )


if __name__ == "__main__":
    unittest.main()
