import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from scripts.run_poisson_fast_feasibility import (
    _contact_physical_boundary,
    _first_contact_boundary,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_poisson_fast_feasibility.py"
SLURM = ROOT / "slurm/poisson_fast_feasibility.sbatch"


class FastFeasibilityRunnerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = RUNNER.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.slurm = SLURM.read_text(encoding="utf-8")

    def test_runner_is_syntax_valid_and_has_no_policy_server(self):
        self.assertNotIn("policy_server", self.source.lower())
        self.assertIn('"online_policy_query_count": 0', self.source)

    def test_field_is_built_before_prefix_replay(self):
        build = self.source.index("bundle = build_static_field_bundle(")
        prefix = self.source.index(
            'for expected_step in replay.steps[: derived["start_action"]]:'
        )
        self.assertLess(build, prefix)
        self.assertIn("prefix_obstacle_rows", self.source)
        self.assertIn("prefix_action_boundary_speed_admissible", self.source)

    def test_exact_historical_boundary_and_window_are_bound(self):
        self.assertIn("post_action_179_flattened_state_sha256", self.source)
        self.assertIn("window_action_record_sha256", self.source)
        self.assertIn("actions[*].executed", (ROOT / "configs/vlsa_poisson_fast_feasibility.v1.json").read_text())

    def test_two_arms_use_grouped_100hz_500hz_path(self):
        self.assertIn("step_grouped_actions_with_substep_callback", self.source)
        self.assertIn("expected_inner_updates=5", self.source)
        self.assertIn("expected_substeps_per_inner=5", self.source)
        self.assertIn("joint_velocity_adapter_only", self.source)
        self.assertIn("joint_velocity_adapter_plus_link56_psf", self.source)

    def test_runner_does_not_preclip_filtered_command(self):
        run_arm = self.source[
            self.source.index("def _run_arm(") : self.source.index("def _pair_exact(")
        ]
        self.assertNotIn("np.clip", run_arm)
        self.assertIn("normalized_joint_velocity_action(executed", run_arm)

    def test_post_state_field_queries_use_world_points_not_jacobians(self):
        clearance = self.source[
            self.source.index("def _protected_clearance(") : self.source.index(
                "def _first_link_contact_observation("
            )
        ]
        self.assertIn("evaluate_world_points", clearance)
        self.assertNotIn("evaluate_point_jacobians", clearance)
        self.assertIn("post_state_field_observation_count == 200", self.source)

    def test_marker_is_synced_to_source_boundary_before_restore(self):
        build = self.source.index("marker_sync = _synchronize_visual_marker_model(source_env, env)")
        restore = self.source.index("restore = restore_osc_settled_state_into_joint_velocity_env(")
        self.assertLess(build, restore)
        self.assertIn("np.array_equal(source_pos, target_pos)", self.source)

    def test_adapter_target_is_initialized_after_wrapper_forward(self):
        run_arm = self.source[
            self.source.index("def _run_arm(") : self.source.index("def _pair_exact(")
        ]
        provider = run_arm[run_arm.index("def provider(") : run_arm.index("def callback(")]
        self.assertIn("if int(inner_index) == 0:", provider)
        self.assertIn("target = adapter.begin_high_level_action(", provider)
        before_provider = run_arm[: run_arm.index("def provider(")]
        self.assertNotIn("adapter.begin_high_level_action(", before_provider)

    def test_non_poisson_dynamic_bounds_are_measured_in_both_arms(self):
        run_arm = self.source[
            self.source.index("def _run_arm(") : self.source.index("def _pair_exact(")
        ]
        bounds = run_arm.index("lower, upper = joint_velocity_bounds(")
        psf_branch = run_arm.index("if psf_enabled:", bounds)
        self.assertLess(bounds, psf_branch)
        self.assertIn("nominal_within_dynamic_joint_bounds", run_arm)

    def test_activation_trace_has_sample_identity_and_strict_boundary(self):
        self.assertIn('"argmin_protected_sample": nominal_argmin_sample', self.source)
        self.assertIn('row["physical_boundary"] < adapter_contact_boundary', self.source)
        self.assertIn("_PsfContactObserved", self.source)
        self.assertIn("active_motion = _active_interval_motion(", self.source)
        self.assertIn('"psf_active_interval_motion_complete"', self.source)

    def test_contact_phases_are_normalized_to_physical_boundaries(self):
        live = SimpleNamespace(
            observation_index=7,
            source_phase="live_solver_phase_preintegration_geometry",
            is_physical_nonpositive_distance_contact=True,
            robot_geom_id=11,
        )
        post = SimpleNamespace(
            observation_index=7,
            source_phase="post_integration_recomputed",
            is_physical_nonpositive_distance_contact=True,
            robot_geom_id=11,
        )
        self.assertEqual(_contact_physical_boundary(live, 4500), 4507)
        self.assertEqual(_contact_physical_boundary(post, 4500), 4508)
        measurement = SimpleNamespace(
            live_solver_phase_contact_point_records=(live,),
            post_state_physical_contact_point_records=(post,),
        )
        self.assertEqual(
            _first_contact_boundary(
                measurement, start_boundary=4500, robot_geom_ids=(11,)
            ),
            4507,
        )

    def test_atomic_final_and_typed_failure(self):
        self.assertGreaterEqual(self.source.count("publish_hashed_json(result_path"), 2)
        self.assertIn('"partial_output_interpreted": False', self.source)
        self.assertIn('"primary_outcome": "APPARATUS_FAILURE"', self.source)
        self.assertIn('"arms": arm_evidence', self.source)

    def test_slurm_is_one_h100_and_has_no_upstream_scan_dependencies(self):
        self.assertIn("#SBATCH --gres=gpu:1", self.slurm)
        self.assertIn("#SBATCH --cpus-per-task=8", self.slurm)
        self.assertIn("#SBATCH --mem=64G", self.slurm)
        self.assertNotIn("PARITY_RESULT", self.slurm)
        self.assertNotIn("IDENTIFICATION_RESULT", self.slurm)
        self.assertNotIn("POLICY_PYTHON", self.slurm)
        self.assertIn("run_poisson_fast_feasibility.py", self.slurm)


if __name__ == "__main__":
    unittest.main()
