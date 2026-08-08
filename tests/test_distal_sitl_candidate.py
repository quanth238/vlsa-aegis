from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DistalSitlCandidateTests(unittest.TestCase):
    def test_raw_contact_veto_recognizes_link6_obstacle_contact(self) -> None:
        from main.multilink_ellipsoid.sitl_candidate import (
            _protected_contact_evidence,
        )

        body_ids = {
            "robot0_link5": 9,
            "robot0_link6": 10,
            "robot0_link7": 11,
            "moka_pot_obstacle_1_main": 28,
        }
        parent_ids = [0] * 29
        parent_ids[9] = 8
        parent_ids[10] = 9
        parent_ids[11] = 10
        parent_ids[28] = 0
        model = SimpleNamespace(
            body_parentid=parent_ids,
            geom_bodyid=[28, 10],
            body_name2id=lambda name: body_ids[name],
            geom_id2name=lambda geom_id: (
                "moka_pot_obstacle_1_g12"
                if geom_id == 0
                else "robot0_link6_collision"
            ),
        )
        data = SimpleNamespace(
            ncon=1,
            contact=[SimpleNamespace(geom1=0, geom2=1, dist=-0.0005)],
        )
        env = SimpleNamespace(sim=SimpleNamespace(model=model, data=data))
        evidence = _protected_contact_evidence(env, "moka_pot_obstacle_1")
        self.assertEqual(evidence["nonpositive_protected_contact_count"], 1)
        self.assertEqual(
            evidence["events"][0]["protected_geom_name"],
            "robot0_link6_collision",
        )

    def test_config_and_targets(self) -> None:
        from main.multilink_ellipsoid.sitl_candidate import load_sitl_candidate_config

        config = load_sitl_candidate_config(
            ROOT / "configs/vlsa_distal_sitl_candidate_e05.v1.json"
        )
        self.assertEqual(config["protected_geometry"]["distal_constraint_count"], 7)
        self.assertEqual(config["protected_geometry"]["end_effector_constraint_count"], 1)
        self.assertEqual(
            config["protected_geometry"]["distal_activation_clearance_m"], 0.015
        )
        self.assertEqual(config["protected_geometry"]["distal_clearance_target_m"], 0.01)
        self.assertEqual(
            config["protected_geometry"]["end_effector_target"],
            "do_not_worsen_exact_next_clearance_of_released_aegis_nominal",
        )
        self.assertEqual(
            config["candidate_search"]["selection_objective"],
            "lexicographic_maximum_minimum_distal_clearance_then_minimum_nominal_deviation",
        )

    def test_live_config_replans_released_aegis_nominal(self) -> None:
        from main.multilink_ellipsoid.sitl_candidate import load_sitl_candidate_config

        config = load_sitl_candidate_config(
            ROOT / "configs/vlsa_distal_sitl_live_e05.v1.json"
        )
        self.assertEqual(
            config["nominal_action_source"],
            "live_pi05_libero_then_released_aegis_ee_qp_replanned_every_five_steps",
        )
        self.assertIn("live_closed_loop", config["claim_scope"])

    def test_exact_box_closed_loop_applies_margin_only_to_distal_rows(self) -> None:
        import numpy as np

        from main.multilink_ellipsoid.sitl_candidate import (
            DistalSitlCandidateFilter,
            EXACT_BOX_CLOSED_LOOP_SCHEMA,
            load_sitl_candidate_config,
        )

        config = load_sitl_candidate_config(
            ROOT / "configs/vlsa_distal_exact_box_closed_loop_e05.v1.json"
        )
        self.assertEqual(config["schema_version"], EXACT_BOX_CLOSED_LOOP_SCHEMA)
        self.assertEqual(
            config["obstacle_geometry"]["distal_warning_margin_m"], 0.008
        )
        self.assertEqual(
            config["obstacle_geometry"]["end_effector_warning_margin_m"], 0.0
        )
        instance = object.__new__(DistalSitlCandidateFilter)
        instance.config = config
        target = instance.targets(np.asarray([0.1] * 7 + [0.023]))
        np.testing.assert_allclose(target[:7], 0.008)
        self.assertEqual(float(target[-1]), 0.0)

    def test_exact_box_closed_loop_uses_substep_clearance_and_raw_veto(self) -> None:
        source = (
            ROOT / "main/multilink_ellipsoid/sitl_candidate.py"
        ).read_text()
        self.assertIn('record["minimum_substep_clearance_m"]', source)
        self.assertIn('"obstacle_geometry" in self.config', source)

        runner = (
            ROOT / "scripts/evaluate_distal_sitl_candidate_e05.py"
        ).read_text()
        self.assertIn("ExactObstacleBoxUnion", runner)
        self.assertIn("--obstacle-config", runner)
        validator = (
            ROOT / "scripts/validate_distal_exact_box_closed_loop_e05.py"
        ).read_text()
        self.assertIn("distal_only_target_vector", validator)
        self.assertIn("accepted_exact_substep_clearance", validator)

    def test_v2_retains_original_aegis_obstacle_for_only_the_ee_row(self) -> None:
        from main.multilink_ellipsoid.sitl_candidate import (
            DISTAL_ONLY_CLOSED_LOOP_SCHEMA,
            load_sitl_candidate_config,
        )

        config = load_sitl_candidate_config(
            ROOT / "configs/vlsa_distal_exact_box_closed_loop_e05.v2.json"
        )
        self.assertEqual(config["schema_version"], DISTAL_ONLY_CLOSED_LOOP_SCHEMA)
        self.assertEqual(
            config["obstacle_geometry"]["end_effector_obstacle_source"],
            "frozen_released_aegis_perception_mvee",
        )
        source = (
            ROOT / "main/multilink_ellipsoid/sitl_candidate.py"
        ).read_text()
        self.assertIn("links[:7], obstacles", source)
        self.assertIn("links[7:], [geometry.obstacle]", source)

    def test_v3_adds_no_second_discrete_ee_target(self) -> None:
        from main.multilink_ellipsoid.sitl_candidate import (
            DISTAL_ONLY_RELEASED_EE_SCHEMA,
            load_sitl_candidate_config,
        )

        config = load_sitl_candidate_config(
            ROOT / "configs/vlsa_distal_exact_box_closed_loop_e05.v3.json"
        )
        self.assertEqual(config["schema_version"], DISTAL_ONLY_RELEASED_EE_SCHEMA)
        self.assertEqual(
            config["obstacle_geometry"]["end_effector_constraint_mode"],
            "unchanged_released_aegis_qp_without_second_discrete_EE_target",
        )
        self.assertEqual(
            config["protected_geometry"]["end_effector_target"],
            "released_aegis_nominal_qp_then_raw_simulator_contact_and_displacement_veto",
        )

    def test_v4_pairs_failure_prefix_before_live_recovery(self) -> None:
        from main.multilink_ellipsoid.sitl_candidate import (
            PAIRED_DISTAL_RECOVERY_SCHEMA,
            load_sitl_candidate_config,
        )

        config = load_sitl_candidate_config(
            ROOT / "configs/vlsa_distal_exact_box_closed_loop_e05.v4.json"
        )
        self.assertEqual(config["schema_version"], PAIRED_DISTAL_RECOVERY_SCHEMA)
        self.assertEqual(
            config["nominal_action_source"],
            "immutable_released_aegis_until_first_sitl_intervention_then_live_pi05_libero_recovery_with_released_aegis_ee_qp",
        )

    def test_hybrid_config_switches_only_after_distal_intervention(self) -> None:
        from main.multilink_ellipsoid.sitl_candidate import load_sitl_candidate_config

        config = load_sitl_candidate_config(
            ROOT / "configs/vlsa_distal_sitl_hybrid_recovery_e05.v1.json"
        )
        self.assertEqual(
            config["nominal_action_source"],
            "immutable_released_aegis_until_first_sitl_intervention_then_live_pi05_libero_recovery_with_released_aegis_ee_qp",
        )
        self.assertEqual(
            config["candidate_search"]["selection_objective"],
            "lexicographic_minimum_nominal_deviation_then_maximum_minimum_distal_clearance",
        )
        self.assertIn("successful_aegis_prefix", config["claim_scope"])

    def test_minimal_replay_config_preserves_action_plan(self) -> None:
        from main.multilink_ellipsoid.sitl_candidate import load_sitl_candidate_config

        config = load_sitl_candidate_config(
            ROOT / "configs/vlsa_distal_sitl_minimal_replay_e05.v1.json"
        )
        self.assertEqual(
            config["nominal_action_source"],
            "immutable_successful_released_aegis_env_step_input",
        )
        self.assertEqual(
            config["candidate_search"]["selection_objective"],
            "lexicographic_minimum_reference_eef_error_then_minimum_nominal_deviation",
        )
        self.assertEqual(
            config["protected_geometry"]["distal_clearance_target_m"], -1.0
        )
        self.assertEqual(
            config["protected_geometry"]["distal_activation_clearance_m"], 0.015
        )
        self.assertEqual(
            config["protected_geometry"]["end_effector_target"],
            "released_aegis_nominal_qp_then_raw_simulator_contact_and_displacement_veto",
        )
        self.assertIn("not_online_policy", config["claim_scope"])
        self.assertIn("oracle_reference_eef_tracking", config["claim_scope"])

    def test_reference_runner_records_contact_veto_and_rejoin_phases(self) -> None:
        source = (
            ROOT / "main/multilink_ellipsoid/sitl_candidate.py"
        ).read_text()
        self.assertIn("raw_contact_veto", source)
        self.assertIn("reference_rejoin", source)

    def test_targets_preserve_nominal_end_effector_clearance(self) -> None:
        import numpy as np

        from main.multilink_ellipsoid.sitl_candidate import DistalSitlCandidateFilter

        instance = object.__new__(DistalSitlCandidateFilter)
        instance.config = {
            "protected_geometry": {
                "distal_clearance_target_m": 0.01,
                "distal_activation_clearance_m": 0.015,
                "end_effector_target": "do_not_worsen_exact_next_clearance_of_released_aegis_nominal",
            }
        }
        nominal = np.asarray([0.1] * 7 + [-0.003])
        target = instance.targets(nominal)
        np.testing.assert_allclose(target[:7], 0.01)
        self.assertEqual(float(target[-1]), -0.003)

    def test_finite_difference_rows_support_eight_constraints(self) -> None:
        import numpy as np

        from main.multilink_ellipsoid.sitl_candidate import finite_difference_rows

        expected = np.arange(24, dtype=np.float64).reshape(8, 3) / 10.0
        plus_xyz = np.diag([0.2, 0.2, 0.2])
        minus_xyz = -plus_xyz
        plus_h = np.stack([0.4 * expected[:, index] for index in range(3)])
        minus_h = np.zeros((3, 8), dtype=np.float64)
        observed = finite_difference_rows(plus_h, minus_h, plus_xyz, minus_xyz)
        np.testing.assert_allclose(observed, expected)

    def test_candidate_set_preserves_bounds_and_contains_escape_actions(self) -> None:
        import numpy as np

        from main.multilink_ellipsoid.sitl_candidate import (
            candidate_xyz_values,
            load_sitl_candidate_config,
        )

        config = load_sitl_candidate_config(
            ROOT / "configs/vlsa_distal_sitl_candidate_e05.v1.json"
        )
        nominal = np.asarray([0.8, -0.9, 0.1])
        qp = np.asarray([0.2, -0.2, 0.3])
        candidates = candidate_xyz_values(nominal, qp, config)
        self.assertGreaterEqual(len(candidates), 27)
        self.assertTrue(any(np.array_equal(value, np.zeros(3)) for _, value in candidates))
        self.assertTrue(any(source == "reverse_nominal" for source, _ in candidates))
        self.assertTrue(all(np.max(np.abs(value)) <= 1.0 for _, value in candidates))
        self.assertEqual(
            len({tuple(float(item) for item in value) for _, value in candidates}),
            len(candidates),
        )

    def test_slurm_contract_requires_h100_and_separate_output(self) -> None:
        source = (ROOT / "slurm/distal_sitl_candidate_e05.sbatch").read_text()
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("H100", source)
        self.assertIn("vlsa-distal-sitl-candidate-e05", source)
        self.assertIn("HEURISTIC_CONFIG", source)
        self.assertNotIn("rsync --delete", source)

    def test_live_slurm_runs_pi05_libero_on_one_h100(self) -> None:
        source = (ROOT / "slurm/distal_sitl_live_e05.sbatch").read_text()
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("pi05_libero", source)
        self.assertIn("vlsa_distal_sitl_live_e05.v1.json", source)
        self.assertIn("HEURISTIC_CONFIG", source)
        self.assertIn("--host 127.0.0.1", source)
        self.assertIn("OBSTACLE_CONFIG", source)

    def test_hybrid_runner_switches_to_corresponding_live_query(self) -> None:
        source = (
            ROOT / "scripts/evaluate_distal_sitl_candidate_e05.py"
        ).read_text()
        self.assertIn("hybrid_recovery", source)
        self.assertIn("recovery_activation_step", source)
        self.assertIn("index // int(case.get(\"replan_steps\", 5))", source)
        self.assertIn("immutable_successful_released_aegis_env_step_input", source)
        self.assertIn("vlsa_distal_sitl_hybrid_recovery_e05_result.v1", source)

    def test_live_runner_uses_calibrated_initial_chunk_gate(self) -> None:
        source = (
            ROOT / "scripts/evaluate_distal_sitl_candidate_e05.py"
        ).read_text()
        self.assertIn("maximum_absolute_raw_action_difference", source)
        self.assertIn("allocation_job_37054_split_jit_equivalence_gate", source)
        self.assertIn("first_five_gripper_signs_equal", source)
        self.assertIn("compared_executed_prefix_length", source)
        self.assertIn("archived_initial_prefix", source)

    def test_success_video_replays_one_rendered_environment_and_fails_on_stripes(self) -> None:
        source = (
            ROOT / "scripts/render_distal_sitl_success_video.py"
        ).read_text()
        self.assertEqual(source.count("_build_environment("), 1)
        self.assertIn("executed_sitl_action", source)
        self.assertIn("simulator-state trace differs", source)
        self.assertIn("source frame has striped pixel corruption", source)
        self.assertIn("decoded video has striped pixel corruption", source)
        self.assertIn("native_task_success_step", source)

        slurm = (
            ROOT / "slurm/render_distal_sitl_success_video.sbatch"
        ).read_text()
        self.assertIn("#SBATCH --gres=gpu:1", slurm)
        self.assertIn("exactly one H100", slurm)
        self.assertIn("EXPECTED_RESULT_SHA256", slurm)
        self.assertIn("verified-replay.mp4", slurm)
        self.assertNotIn("rsync --delete", slurm)


if __name__ == "__main__":
    unittest.main()
