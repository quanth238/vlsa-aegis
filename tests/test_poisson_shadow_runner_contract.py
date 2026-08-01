from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_poisson_shadow_parity.py"
SPEC = importlib.util.spec_from_file_location(
    "poisson_shadow_parity_under_test", RUNNER_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ShadowRunnerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = (ROOT / "scripts/run_poisson_shadow_parity.py").read_text(
            encoding="utf-8"
        )
        cls.batch = (ROOT / "slurm/poisson_shadow_parity.sbatch").read_text(
            encoding="utf-8"
        )

    def test_runner_parses_and_uses_exact_historical_oracles(self):
        ast.parse(self.runner)
        self.assertIn("load_historical_action_replay", self.runner)
        self.assertIn("simulator state differs at step", self.runner)
        self.assertIn("reward/done differs at step", self.runner)
        self.assertIn("goal vector differs at step", self.runner)
        self.assertIn("callback and ordinary observations differ", self.runner)

    def test_callback_is_read_only_and_requires_all_25_substeps(self):
        start = self.runner.index("            def callback(")
        end = self.runner.index("            observation, reward", start)
        callback = self.runner[start:end]
        self.assertNotIn("sim.get_state().flatten()", callback)
        self.assertIn("_official_integration_state", callback)
        self.assertNotIn("sim.step", callback)
        self.assertNotIn("sim.forward", callback)
        self.assertIn("expected_substeps=25", self.runner)
        self.assertIn("25 * len(replay.steps)", self.runner)
        self.assertIn(
            '"official_integration_state_sha256_ledger"', self.runner
        )
        self.assertIn(
            '"official_integration_state_sequence_sha256"', self.runner
        )

    def test_job_requires_clean_slurm_h100_allocation(self):
        self.assertIn("source[\"status_short\"]", self.runner)
        self.assertIn("SLURM_JOB_ID", self.runner)
        self.assertIn('"H100" not in devices[0]["name"]', self.runner)
        self.assertIn('"--query-gpu=name,uuid,driver_version"', self.runner)
        self.assertNotIn("memory.total", self.runner)
        self.assertIn("git status --short", self.batch)
        self.assertIn("EXPECTED_GIT_COMMIT", self.batch)
        self.assertIn("git rev-parse HEAD", self.batch)
        self.assertIn("#SBATCH --no-requeue", self.batch)
        self.assertIn("MUJOCO_GL=osmesa", self.batch)
        self.assertIn("PYOPENGL_PLATFORM=osmesa", self.batch)
        self.assertIn("LIBERO_CONFIG_PATH", self.batch)
        self.assertIn("PYTHONPATH", self.batch)

    def test_artifact_is_final_only_after_both_stages(self):
        callback_call = self.runner.index("callback = _run_callback")
        publish_call = self.runner.index("publish_hashed_json(output, payload)")
        self.assertLess(callback_call, publish_call)
        self.assertIn(
            'SCHEMA_VERSION = "vlsa_poisson_shadow_parity.v3"',
            self.runner,
        )
        self.assertIn("scientific_result", self.runner)
        self.assertIn("no Poisson correction or safety efficacy", self.runner)

    def test_boundary_zero_official_state_is_captured_before_replay_steps(self):
        ordinary_start = self.runner.index("def _run_ordinary(")
        callback_start = self.runner.index("def _run_callback(")
        main_start = self.runner.index("def main()")
        ordinary = self.runner[ordinary_start:callback_start]
        callback = self.runner[callback_start:main_start]
        for arm in (ordinary, callback):
            self.assertLess(
                arm.index("settled_official_state = _official_integration_state("),
                arm.index("for expected_step in replay.steps:"),
            )
            self.assertIn('"physical_boundary": 0', arm)
            self.assertIn('"mujoco_state_specification": "mjSTATE_INTEGRATION"', arm)
            self.assertIn('"state_vector_length"', arm)
            self.assertIn('"settled_official_integration_state"', arm)

    def test_mig_inventory_uses_exactly_one_h100_identity_row(self):
        with mock.patch.object(
            MODULE.subprocess,
            "check_output",
            return_value="NVIDIA H100 80GB HBM3, GPU-012345, 555.42.06\n",
        ):
            inventory = MODULE._gpu_inventory()
        self.assertEqual(
            inventory,
            {
                "devices": [
                    {
                        "name": "NVIDIA H100 80GB HBM3",
                        "uuid": "GPU-012345",
                        "driver_version": "555.42.06",
                    }
                ]
            },
        )

    def test_shadow_inventory_rejects_ambiguous_or_non_h100_visibility(self):
        outputs = (
            "NVIDIA A100, GPU-a100, 555.42.06\n",
            (
                "NVIDIA H100 80GB HBM3, GPU-a, 555.42.06\n"
                "NVIDIA H100 80GB HBM3, GPU-b, 555.42.06\n"
            ),
            "NVIDIA H100 80GB HBM3, GPU-a, [Insufficient Permissions]\n",
        )
        for output in outputs:
            with self.subTest(output=output), mock.patch.object(
                MODULE.subprocess, "check_output", return_value=output
            ):
                with self.assertRaises(MODULE.ShadowParityError):
                    MODULE._gpu_inventory()


if __name__ == "__main__":
    unittest.main()
