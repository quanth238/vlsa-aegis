from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crfs_harness.artifacts import atomic_write_json, content_hash, file_sha256

try:
    from crfs_oracle.endpoint_free_runner import (
        _nontranslation_preserved,
        _pooled_witness_outcome,
        _simulator_trial_pass,
        _array_hash,
        endpoint_free_config_from_mapping,
        valid_endpoint_free_completion,
        validate_endpoint_free_result,
    )
    from crfs_oracle.runner import OracleConfig
except ModuleNotFoundError:  # The local bootstrap intentionally has no NumPy/main runtime.
    _nontranslation_preserved = None
    _pooled_witness_outcome = None
    _simulator_trial_pass = None
    _array_hash = None
    endpoint_free_config_from_mapping = None
    valid_endpoint_free_completion = None
    validate_endpoint_free_result = None
    OracleConfig = None


ROOT = Path(__file__).resolve().parents[1]


def _oracle(checkpoint_sha256: str) -> OracleConfig:
    return OracleConfig(
        host="127.0.0.1",
        port=8000,
        resize_size=224,
        settle_steps=20,
        executed_prefix=5,
        action_horizon=10,
        action_dim=32,
        sampler_steps=10,
        intervention_step=5,
        safety_margin_m=0.005,
        distance_limit_m=1.0,
        eef_radius_m=0.06,
        measurement_repeats=2,
        stop_after_measurement=False,
        response_matrix_m_per_action=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)),
        optimizer_max_iterations=1,
        checkpoint_id="checkpoint",
        checkpoint_sha256=checkpoint_sha256,
        output_root="/tmp/output",
        run_id="run",
    )


@unittest.skipUnless(
    endpoint_free_config_from_mapping is not None,
    "R01 runtime checks execute inside the allocation dependency environment",
)
class EndpointFreeRunnerRuntimeTest(unittest.TestCase):
    def test_config_loads_only_a_matching_frozen_r00_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_sha = "a" * 64
            checkpoint_sha = "b" * 64
            summary = {
                "schema_version": "1.0",
                "gate": "R00",
                "status": "passed",
                "calibration_and_evaluation_groups_disjoint": True,
                "scene_motion_measurement": (
                    "maximum direct MuJoCo body displacement over 125 substeps"
                ),
                "evaluation_manifest_sha256": manifest_sha,
                "checkpoint_sha256": checkpoint_sha,
                "calibration": {
                    "p_min_m": 0.012,
                    "quantile": 0.25,
                    "quantile_method": "inverted_cdf",
                },
                "eligible_positive_examples": 50,
                "calibration_groups": 30,
                "evaluation_groups": 20,
            }
            path = root / "r00.json"
            atomic_write_json(path, summary)
            value = {
                "ready_to_run": True,
                "progress": {
                    "phase": "pregrasp_reach",
                    "target_object": "akita_black_bowl_1",
                    "minimum_progress_m": 0.012,
                    "calibration_artifact": str(path),
                    "calibration_sha256": file_sha256(path),
                    "reject_target_motion_above_m": 0.001,
                    "reject_obstacle_motion_above_m": 0.001,
                    "scene_motion_measurement": "maximum_substep_displacement",
                },
                "planner": {
                    "method": "deterministic_cem_multistart",
                    "endpoint_preservation": False,
                    "translation_action_bounds": [-1.0, 1.0],
                    "optimizer_safety_margin_m": 0.01,
                    "simulator_safety_margin_m": 0.005,
                    "simulator_verification_candidates": 3,
                    "planner_seed": 7,
                },
            }
            config = endpoint_free_config_from_mapping(
                value,
                _oracle(checkpoint_sha),
                repo_root=root,
                evaluation_manifest_sha256=manifest_sha,
            )
            self.assertEqual(config.p_min_m, 0.012)
            self.assertEqual(config.r00_summary_sha256, file_sha256(path))
            self.assertEqual(
                (config.translation_action_low, config.translation_action_high), (-1.0, 1.0)
            )
            value["progress"]["reject_obstacle_motion_above_m"] = 0.0011
            with self.assertRaisesRegex(ValueError, "cannot exceed 1 mm"):
                endpoint_free_config_from_mapping(
                    value,
                    _oracle(checkpoint_sha),
                    repo_root=root,
                    evaluation_manifest_sha256=manifest_sha,
                )
            value["progress"]["reject_obstacle_motion_above_m"] = 0.001
            value["planner"]["simulator_safety_margin_m"] = 0.004
            with self.assertRaisesRegex(ValueError, "fixed 5 mm margin"):
                endpoint_free_config_from_mapping(
                    value,
                    _oracle(checkpoint_sha),
                    repo_root=root,
                    evaluation_manifest_sha256=manifest_sha,
                )
            value["planner"]["simulator_safety_margin_m"] = 0.005
            value["progress"]["calibration_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                endpoint_free_config_from_mapping(
                    value,
                    _oracle(checkpoint_sha),
                    repo_root=root,
                    evaluation_manifest_sha256=manifest_sha,
                )

    def test_simulator_gate_requires_margin_progress_stationarity_and_no_contact(self) -> None:
        trial = {
            "clearance_m": 0.006,
            "contact": False,
            "measurement_samples": 126,
            "reach": {
                "reach_progress_m": 0.02,
                "target_displacement_m": 0.0002,
                "active_obstacle_displacement_m": 0.0003,
                "maximum_target_displacement_m": 0.0004,
                "maximum_active_obstacle_displacement_m": 0.0005,
            },
        }
        passed, reasons = _simulator_trial_pass(
            trial,
            minimum_progress_m=0.01,
            simulator_safety_margin_m=0.005,
            maximum_target_displacement_m=0.001,
            maximum_obstacle_displacement_m=0.001,
        )
        self.assertTrue(passed)
        self.assertFalse(reasons)
        trial["contact"] = True
        trial["reach"]["maximum_active_obstacle_displacement_m"] = 0.002
        passed, reasons = _simulator_trial_pass(
            trial,
            minimum_progress_m=0.01,
            simulator_safety_margin_m=0.005,
            maximum_target_displacement_m=0.001,
            maximum_obstacle_displacement_m=0.001,
        )
        self.assertFalse(passed)
        self.assertIn("physical_contact", reasons)
        self.assertIn("obstacle_moved", reasons)

    def test_nontranslation_channels_must_be_identical(self) -> None:
        import numpy as np

        nominal = np.zeros((5, 7), dtype=float)
        candidate = nominal.copy()
        candidate[:, :3] = 0.2
        self.assertTrue(_nontranslation_preserved(candidate, nominal))
        candidate[0, 6] = 1.0
        self.assertFalse(_nontranslation_preserved(candidate, nominal))

    def test_p_min_witness_from_p_zero_search_is_not_ignored(self) -> None:
        empty = {
            "verified_at_p_min": False,
            "verified_at_p_zero": False,
            "changed_witness_at_p_min": False,
            "changed_witness_at_p_zero": False,
            "attempts": [],
        }
        p_zero_attempt = {
            "candidate_index": 3,
            "source": "p_zero_control",
            "actions": [[0.1] * 7 for _ in range(5)],
            "verified_at_p_min": True,
            "verified_at_p_zero": True,
            "changed_witness_at_p_min": True,
            "changed_witness_at_p_zero": True,
        }
        verification = {
            "p_min": dict(empty),
            "p_zero": {
                **empty,
                "verified_at_p_min": True,
                "verified_at_p_zero": True,
                "changed_witness_at_p_min": True,
                "changed_witness_at_p_zero": True,
                "attempts": [p_zero_attempt],
            },
        }
        outcome = _pooled_witness_outcome(
            verification,
            nominal_collision_reproduced=True,
        )
        self.assertTrue(outcome["p_min_verified"])
        self.assertEqual(outcome["selected_p_min_changed_witness"]["search"], "p_zero")
        mismatch = _pooled_witness_outcome(
            verification,
            nominal_collision_reproduced=False,
        )
        self.assertFalse(mismatch["p_min_verified"])
        self.assertTrue(mismatch["p_min_changed_witness_verified"])

    def test_schema_valid_search_miss_is_resumable_without_nonexistence_claim(self) -> None:
        import numpy as np

        actions = [[0.0] * 7 for _ in range(5)]
        case_record = {
            "schema_version": "1.0",
            "case_id": "case",
            "task_suite": "safelibero_spatial",
            "safety_level": "II",
            "task_index": 0,
            "episode_index": 1,
            "group_id": "group",
            "environment_seed": 2,
            "policy_seed": 3,
            "random_control_seed": 4,
        }
        result = {
            "schema_version": "1.0",
            "gate": "R01",
            "case_id": "case",
            "run_id": "run",
            "status": "no_verified_safe_progress",
            "config_hash": "a" * 64,
            "provenance": {
                "git_commit": "commit",
                "git_dirty": False,
                "baseline_commit": "baseline",
                "task_suite": "safelibero_spatial",
                "safety_level": "II",
                "task_index": 0,
                "episode_index": 1,
                "group_id": "group",
                "environment_seed": 2,
                "policy_seed": 3,
                "random_control_seed": 4,
                "planner_seed_p_min": 5,
                "planner_seed_p_zero": 5,
                "case_record": case_record,
                "checkpoint_sha256": "b" * 64,
                "checkpoint_id": "checkpoint",
                "noise_sha256": "c" * 64,
                "input_manifest_sha256": "d" * 64,
                "sampler_steps": 10,
                "intervention_step": 5,
                "model_action_horizon": 10,
                "executed_action_horizon": 5,
                "model_action_dimension": 32,
                "policy_determinism": {"passed": True},
                "nominal_simulator_replay_exact": True,
                "slurm_job_id": "1",
                "slurm_array_task_id": "0",
                "partition": "main",
                "host": "host",
                "device": "0",
                "case_record_sha256": content_hash(case_record),
                "r00_summary_sha256": "f" * 64,
                "action_frame": "world-frame OSC translation delta",
                "normalization_space": "unnormalized",
                "translation_action_bounds": [-1.0, 1.0],
                "d_opt_model": "proxy",
                "d_sim_model": "simulator",
                "scene_motion_measurement": (
                    "maximum direct MuJoCo body displacement over 125 substeps"
                ),
                "optimizer_safety_margin_m": 0.01,
                "simulator_safety_margin_m": 0.005,
                "simulator_verification_repeats": 2,
                "measurement_samples_per_trial": 126,
                "search_registration": {"method": "deterministic_cem_multistart"},
            },
            "calibration": {
                "gate": "R00",
                "summary_sha256": "f" * 64,
                "p_min_m": 0.01,
            },
            "nominal": {
                "actions": actions,
                "actions_sha256": _array_hash(np.asarray(actions, dtype=np.float64)),
                "full_model_actions_sha256": "2" * 64,
                "branch_snapshot": {},
                "collision_reproduced": True,
                "fails_registered_margin": True,
                "repeats": [{}, {}],
            },
            "searches": {
                "p_min": {
                    "outcome": "not_found_within_budget",
                    "candidates": [],
                    "best_attempt": {},
                    "maximum_clearance_attempt": {},
                    "maximum_progress_attempt": {},
                    "evaluations": 1,
                    "evaluation_budget": 1,
                },
                "p_zero": {
                    "outcome": "not_found_within_budget",
                    "candidates": [],
                    "best_attempt": {},
                    "maximum_clearance_attempt": {},
                    "maximum_progress_attempt": {},
                    "evaluations": 1,
                    "evaluation_budget": 1,
                },
            },
            "verification": {
                "p_min": {
                    "minimum_progress_m": 0.01,
                    "attempted_candidates": 0,
                    "verified": False,
                    "verified_at_p_min": False,
                    "verified_at_p_zero": False,
                    "changed_witness_at_p_min": False,
                    "changed_witness_at_p_zero": False,
                    "selected": None,
                    "attempts": [],
                },
                "p_zero": {
                    "minimum_progress_m": 0.0,
                    "attempted_candidates": 0,
                    "verified": False,
                    "verified_at_p_min": False,
                    "verified_at_p_zero": False,
                    "changed_witness_at_p_min": False,
                    "changed_witness_at_p_zero": False,
                    "selected": None,
                    "attempts": [],
                },
            },
            "outcome": {
                "p_min_verified": False,
                "p_zero_verified": False,
                "nominal_collision_reproduced": True,
                "nominal_fails_registered_margin": True,
                "p_min_any_action_verified": False,
                "p_zero_any_action_verified": False,
                "p_min_changed_witness_verified": False,
                "p_zero_changed_witness_verified": False,
                "interpretation": "bounded searches found no verified witness",
            },
        }
        self.assertFalse(validate_endpoint_free_result(result))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "endpoint-free-feasibility.json"
            atomic_write_json(path, result)
            self.assertTrue(valid_endpoint_free_completion(path))
            self.assertTrue(
                valid_endpoint_free_completion(
                    path,
                    case_id="case",
                    run_id="run",
                    config_hash="a" * 64,
                    input_manifest_sha256="d" * 64,
                )
            )
            self.assertFalse(valid_endpoint_free_completion(path, config_hash="0" * 64))
            path.write_text('{"status":', encoding="utf-8")
            self.assertFalse(valid_endpoint_free_completion(path))


class EndpointFreeRunnerStructuralTest(unittest.TestCase):
    def test_runner_keeps_search_and_simulator_verification_separate(self) -> None:
        source = (ROOT / "main/crfs_oracle/endpoint_free_runner.py").read_text(encoding="utf-8")
        first_search = source.index("p_min_search = solve_endpoint_free_projection")
        second_search = source.index("p_zero_search = solve_endpoint_free_projection")
        verification = source.index('verification = {', second_search)
        self.assertLess(first_search, second_search)
        self.assertLess(second_search, verification)
        self.assertIn("minimum_progress=0.0", source)
        self.assertIn("_verification_shortlist", source)
        self.assertIn('"proxy_clearance_false_negative"', source)
        self.assertIn("atomic_write_json(output, result)", source)
        self.assertNotIn('status = "infeasible"', source)


if __name__ == "__main__":
    unittest.main()
