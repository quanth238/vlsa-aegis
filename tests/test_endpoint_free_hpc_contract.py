from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EndpointFreeHpcContractTest(unittest.TestCase):
    def test_config_is_endpoint_free_and_bound_to_passed_calibration(self) -> None:
        config = json.loads(
            (ROOT / "configs/experiments/endpoint_free_reach_h5.json").read_text(encoding="utf-8")
        )
        self.assertTrue(config["ready_to_run"])
        self.assertEqual(config["blocked_on"], [])
        self.assertEqual(config["executed_prefix"], 5)
        self.assertEqual(config["action_horizon"], 10)
        self.assertEqual(config["progress"]["phase"], "pregrasp_reach")
        self.assertEqual(config["progress"]["target_object"], "akita_black_bowl_1")
        self.assertEqual(
            config["progress"]["scene_motion_measurement"],
            "maximum_substep_displacement",
        )
        self.assertAlmostEqual(config["progress"]["minimum_progress_m"], 0.029897349105658888)
        calibration_path = ROOT / config["progress"]["calibration_artifact"]
        self.assertTrue(calibration_path.is_file())
        self.assertEqual(
            config["progress"]["calibration_sha256"],
            "90fe09e075b30e680e5cfbd81c802afa03c94b58ba7f0e95e66080c61af2096f",
        )
        self.assertFalse(config["planner"]["endpoint_preservation"])
        self.assertEqual(config["planner"]["translation_action_bounds"], [-1.0, 1.0])
        self.assertEqual(config["planner"]["method"], "deterministic_cem_multistart")
        self.assertEqual(config["planner"]["simulator_verification_candidates"], 12)
        self.assertEqual(config["planner"]["simulator_verification_repeats"], 2)
        self.assertEqual(config["planner"]["search"]["population_size"], 512)
        self.assertEqual(config["planner"]["search"]["generations"], 12)
        self.assertEqual(config["planner"]["optimizer_safety_margin_m"], 0.01)
        self.assertEqual(config["planner"]["simulator_safety_margin_m"], 0.005)
        serialized = json.dumps(config).lower()
        self.assertNotIn("learned_probe", serialized)
        self.assertNotIn("training", serialized)

    def test_slurm_templates_are_explicit_opt_in_and_within_resource_ceiling(self) -> None:
        smoke = (ROOT / "slurm/endpoint_free_mig.sbatch").read_text(encoding="utf-8")
        main = (ROOT / "slurm/endpoint_free_main_array.sbatch").read_text(encoding="utf-8")
        for source in (smoke, main):
            self.assertIn("export CRFS_RUNNER_MODE=endpoint_free", source)
            self.assertIn("grep -q 'CRFS_RUNNER_MODE'", source)
            self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("#SBATCH --array=0-0%1", smoke)
        self.assertIn("#SBATCH --cpus-per-task=6", smoke)
        self.assertIn("#SBATCH --mem=80G", smoke)
        self.assertIn("#SBATCH --cpus-per-task=8", main)
        self.assertIn("#SBATCH --mem=128G", main)
        self.assertNotIn("#SBATCH --time=", main)

        existing_oracle = (ROOT / "slurm/oracle_mig.sbatch").read_text(encoding="utf-8")
        self.assertNotIn("CRFS_RUNNER_MODE=endpoint_free", existing_oracle)

    def test_submit_wrappers_default_to_one_and_cap_at_two(self) -> None:
        smoke = (ROOT / "scripts/hpc/submit_endpoint_free_smoke.sh").read_text(encoding="utf-8")
        array = (ROOT / "scripts/hpc/submit_endpoint_free_array.sh").read_text(encoding="utf-8")
        self.assertIn("smoke requires the frozen 20-case evaluation manifest", smoke)
        self.assertIn("array requires the frozen 20-case evaluation manifest", array)
        self.assertIn("CONCURRENCY=${2:-1}", array)
        self.assertIn("1|2)", array)
        self.assertIn("--array='0-$LAST%$CONCURRENCY'", array)
        for source in (smoke, array):
            self.assertIn("scripts/hpc/preflight.sh", source)
            self.assertIn("ready_to_run", source)
            self.assertIn("grep -q 'CRFS_RUNNER_MODE'", source)
            self.assertIn("endpoint_free_reach_h5.json", source)


if __name__ == "__main__":
    unittest.main()
