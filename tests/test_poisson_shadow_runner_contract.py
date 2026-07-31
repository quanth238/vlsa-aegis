from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


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
        self.assertIn("sim.get_state().flatten()", callback)
        self.assertNotIn("sim.step", callback)
        self.assertNotIn("sim.forward", callback)
        self.assertIn("expected_substeps=25", self.runner)
        self.assertIn("25 * len(replay.steps)", self.runner)

    def test_job_requires_clean_slurm_h100_allocation(self):
        self.assertIn("source[\"status_short\"]", self.runner)
        self.assertIn("SLURM_JOB_ID", self.runner)
        self.assertIn('"H100" in row["name"]', self.runner)
        self.assertIn("git status --short", self.batch)
        self.assertIn("EXPECTED_GIT_COMMIT", self.batch)
        self.assertIn("git rev-parse HEAD", self.batch)
        self.assertIn("MUJOCO_GL=osmesa", self.batch)
        self.assertIn("PYOPENGL_PLATFORM=osmesa", self.batch)
        self.assertIn("LIBERO_CONFIG_PATH", self.batch)
        self.assertIn("PYTHONPATH", self.batch)

    def test_artifact_is_final_only_after_both_stages(self):
        callback_call = self.runner.index("callback = _run_callback")
        publish_call = self.runner.index("publish_hashed_json(output, payload)")
        self.assertLess(callback_call, publish_call)
        self.assertIn("scientific_result", self.runner)
        self.assertIn("no Poisson correction or safety efficacy", self.runner)


if __name__ == "__main__":
    unittest.main()
