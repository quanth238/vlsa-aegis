from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HarnessContractTest(unittest.TestCase):
    def test_at_most_one_gate_is_active(self) -> None:
        features = json.loads((ROOT / "feature_list.json").read_text(encoding="utf-8"))["features"]
        self.assertLessEqual(sum(feature["status"] == "active" for feature in features), 1)
        self.assertEqual({feature["status"] for feature in features} - {"not_started", "active", "blocked", "passing"}, set())

    def test_substep_measurement_is_additive_to_baseline_step(self) -> None:
        source = (ROOT / "safelibero/libero/libero/envs/env_wrapper.py").read_text(encoding="utf-8")
        self.assertIn("def step(self, action):\n        return self.env.step(action)", source)
        self.assertIn("def step_with_substep_callback", source)
        self.assertIn("callback(self.env.sim, substep_index)", source)
        self.assertIn("if update_observables:", source)
        self.assertIn("if collect_observations:", source)

    def test_real_runner_discloses_preliminary_limitations(self) -> None:
        source = (ROOT / "main/crfs_oracle/runner.py").read_text(encoding="utf-8")
        self.assertIn('"evidence_tier": "real_safelibero_preliminary"', source)
        self.assertIn("frozen H04 response matrix plus static branch oriented-box geometry", source)
        self.assertIn("Fixed observation/noise policy replay is not exact", source)
        self.assertIn("hard_reset=False", source)
        self.assertIn('_progress("projection_started")', source)
        self.assertIn("H03 conservative proxy missed a physical contact; refusing projection", source)
        self.assertIn('"measurement-audit.json"', source)
        self.assertIn("Sensor evaluation has no effect on physics", source)

    def test_full_h100_array_has_no_long_run_time_limit(self) -> None:
        source = (ROOT / "slurm/oracle_main_array.sbatch").read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --time", source)
        wrapper = (ROOT / "scripts/hpc/submit_oracle_array.sh").read_text(encoding="utf-8")
        self.assertIn('case "$CONCURRENCY" in 1|2)', wrapper)

    def test_remote_case_preconfigures_safelibero_noninteractively(self) -> None:
        source = (ROOT / "scripts/hpc/run_oracle_case.sh").read_text(encoding="utf-8")
        self.assertIn('"$LIBERO_CONFIG_PATH/config.yaml"', source)
        self.assertIn('SAFELIBERO_ROOT=$REMOTE_REPO/safelibero/libero/libero', source)
        self.assertLess(source.index('"$LIBERO_CONFIG_PATH/config.yaml"'), source.index("scripts/serve_policy.py"))

    def test_launchers_forward_the_declared_experiment_config(self) -> None:
        for relative_path in (
            "scripts/hpc/submit_oracle_smoke.sh",
            "scripts/hpc/submit_oracle_array.sh",
        ):
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            self.assertIn("EXPERIMENT_CONFIG='$EXPERIMENT_CONFIG'", source)

    def test_array_tasks_use_isolated_policy_ports(self) -> None:
        source = (ROOT / "scripts/hpc/run_oracle_case.sh").read_text(encoding="utf-8")
        self.assertIn("8130 + ${SLURM_ARRAY_TASK_ID:-0}", source)


if __name__ == "__main__":
    unittest.main()
