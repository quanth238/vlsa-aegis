from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class HarnessContractTest(unittest.TestCase):
    def test_exactly_one_gate_is_active(self) -> None:
        features = json.loads((ROOT / "feature_list.json").read_text(encoding="utf-8"))["features"]
        self.assertEqual(sum(feature["status"] == "active" for feature in features), 1)
        self.assertEqual({feature["status"] for feature in features} - {"not_started", "active", "blocked", "passing"}, set())

    def test_substep_measurement_is_additive_to_baseline_step(self) -> None:
        source = (ROOT / "safelibero/libero/libero/envs/env_wrapper.py").read_text(encoding="utf-8")
        self.assertIn("def step(self, action):\n        return self.env.step(action)", source)
        self.assertIn("def step_with_substep_callback", source)
        self.assertIn("callback(self.env.sim, substep_index)", source)

    def test_real_runner_discloses_preliminary_limitations(self) -> None:
        source = (ROOT / "main/crfs_oracle/runner.py").read_text(encoding="utf-8")
        self.assertIn('"evidence_tier": "real_safelibero_preliminary"', source)
        self.assertIn("D_opt/D_sim independence is not yet established", source)
        self.assertIn("Fixed observation/noise policy replay is not exact", source)

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


if __name__ == "__main__":
    unittest.main()
