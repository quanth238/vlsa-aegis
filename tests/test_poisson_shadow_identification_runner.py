from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ShadowIdentificationRunnerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = (
            ROOT / "scripts/run_poisson_shadow_identification.py"
        ).read_text(encoding="utf-8")
        cls.module = (
            ROOT / "main/poisson_fullbody/shadow_identification.py"
        ).read_text(encoding="utf-8")
        cls.batch = (
            ROOT / "slurm/poisson_shadow_identification.sbatch"
        ).read_text(encoding="utf-8")

    def test_sources_parse_and_stage_is_bound_to_exact_parity(self):
        ast.parse(self.runner)
        ast.parse(self.module)
        self.assertIn("--parity-result", self.runner)
        self.assertIn("upstream exact-parity commit differs", self.runner)
        self.assertIn("ordinary_and_callback_observations_exact", self.runner)
        self.assertIn("state sequence differs from upstream exact parity", self.runner)

    def test_exact_historical_replay_and_complete_callback_exposure(self):
        self.assertIn("len(replay.steps) != 237", self.runner)
        self.assertIn("expected_substeps=25", self.runner)
        self.assertIn("25 * len(replay.steps)", self.runner)
        self.assertIn("measurement.observed_physics_substeps", self.runner)
        self.assertIn("measurement.first_index", self.runner)
        self.assertIn("measurement.last_index", self.runner)
        self.assertIn("_check_step", self.runner)
        self.assertIn("reward=reward", self.runner)
        self.assertIn("previous_goal_values=previous_goal", self.runner)

    def test_callback_is_read_only_and_drift_does_not_end_replay(self):
        self.assertIn("mjtState.mjSTATE_INTEGRATION", self.runner)
        self.assertIn("mj_getState", self.runner)
        self.assertIn("construction_state_before", self.runner)
        self.assertIn("construction_state_after", self.runner)
        self.assertIn('runtime["np"].array_equal', self.runner)
        self.assertNotIn("sim.get_state().flatten()", self.runner)
        self.assertIn(
            "mutated complete MuJoCo integration state", self.runner
        )
        self.assertIn("except StaticObstacleDriftInadmissible", self.runner)
        self.assertIn("terminate_on_static_drift=False", self.runner)
        self.assertIn("exact OSC replay and contact measurement must continue", self.runner)
        self.assertIn("permanently stops querying the field", self.module)
        self.assertNotIn("env.step(", self.module)

    def test_observed_cbf_residual_is_evaluated_for_every_valid_sample(self):
        self.assertIn("for sample in self._samples", self.module)
        self.assertIn("minimum_h_sample", self.module)
        self.assertIn("minimum_observed_cbf_lhs_sample", self.module)
        self.assertIn(
            "minimum over every valid protected sample", self.module
        )
        self.assertIn(
            'signals["first_any_fail_closed_query_on_contact_geom"]',
            self.module,
        )
        self.assertIn("live_solver_phase_preintegration_geometry", self.module)
        self.assertIn("time_from_settled_state_s", self.module)

    def test_official_state_reader_matches_mujoco_integration_vector(self):
        try:
            import mujoco
            import numpy as np
        except ImportError:
            self.skipTest("official MuJoCo/NumPy are unavailable")
        from scripts.run_poisson_shadow_identification import (
            _official_integration_state,
        )

        model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body><joint type='slide'/><geom "
            "type='sphere' size='.01'/></body></worldbody></mujoco>"
        )
        data = mujoco.MjData(model)
        data.time = 0.125
        data.qpos[0] = 0.25
        data.qvel[0] = -0.5
        observed = _official_integration_state(
            SimpleNamespace(model=model, data=data), np
        )
        specification = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        expected = np.empty(
            int(mujoco.mj_stateSize(model, specification)), dtype=np.float64
        )
        mujoco.mj_getState(model, data, expected, specification)
        self.assertTrue(np.array_equal(observed, expected))

    def test_field_bundle_measurement_and_claim_limits_are_explicit(self):
        self.assertIn("build_static_field_bundle", self.runner)
        self.assertIn("FullRobotObstacleMonitor", self.runner)
        self.assertIn("mujoco_contact_dist_le_0", self.runner)
        self.assertIn('"computed": False', self.module)
        self.assertIn("no active safety or utility efficacy", self.runner)
        self.assertIn('"scientific_result": False', self.runner)

    def test_allocation_wrapper_is_bounded_clean_and_uses_validated_osmesa(self):
        self.assertIn("#SBATCH --partition=mig", self.batch)
        self.assertIn("#SBATCH --gres=gpu:1", self.batch)
        self.assertIn("#SBATCH --cpus-per-task=8", self.batch)
        self.assertIn("#SBATCH --mem=64G", self.batch)
        self.assertIn('if [[ -n "$(git status --short)" ]]', self.batch)
        self.assertIn("EXPECTED_GIT_COMMIT", self.batch)
        self.assertIn("git rev-parse HEAD", self.batch)
        self.assertIn("MUJOCO_GL=osmesa", self.batch)
        self.assertIn("PYOPENGL_PLATFORM=osmesa", self.batch)
        self.assertIn("LIBERO_CONFIG_PATH", self.batch)
        self.assertIn("PYTHONPATH", self.batch)
        self.assertIn("PARITY_RESULT_PATH", self.batch)

    def test_result_is_published_once_only_after_shadow_finishes(self):
        run_index = self.runner.index("shadow = _run_shadow")
        publish_index = self.runner.index("publish_hashed_json(output, payload)")
        self.assertLess(run_index, publish_index)
        self.assertIn("requires a clean source tree", self.runner)
        self.assertIn("SLURM_JOB_ID", self.runner)
        self.assertIn("_gpu_inventory()", self.runner)


if __name__ == "__main__":
    unittest.main()
