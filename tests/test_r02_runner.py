from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # dependency-free local gate
    np = None


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "main", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from crfs_harness.artifacts import content_hash, file_sha256  # noqa: E402
try:  # unittest discovery may import tests either as a package or flat modules.
    from tests.test_sampler_parity_contract import valid_parity_artifact  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - flat discovery fallback
    from test_sampler_parity_contract import valid_parity_artifact  # type: ignore[no-redef] # noqa: E402

if np is not None:
    from crfs_oracle.r02_runner import (  # noqa: E402
        ACCEPTED_R01_SUMMARY_SHA256,
        ACCEPTED_R01_ORDERED_RESULTS_SHA256,
        ARMS,
        DIRECT_RECONFIRMATION_FAILURE,
        DIRECTION_REFERENCE,
        DIRECTION_SEMANTICS_DECISION,
        DIRECTION_SEMANTICS_DECISION_SHA256,
        NOMINAL_RECONFIRMATION_FAILURE,
        NOT_EVALUATED_AFTER_RECONFIRMATION_FAILURE,
        NORMALIZATION_ASSET_SHA256,
        REGISTERED_RESPONSE_MATRIX_M_PER_ACTION,
        REGISTERED_TRANSLATION_ACTION_SCALE,
        _array_record,
        _bounds_check,
        _direction_bundle,
        _direction_records,
        _historical_r01_diagnostic,
        _json_compatible,
        _no_witness_result,
        _normalized_config,
        _path_diagnostic,
        _portable_array_reconstruction_equal,
        _portable_diagnostic_reconstruction_equal,
        _reconfirmation_failure_result,
        _trace_record,
        r02_config_from_mapping,
        validate_r02_result,
    )
    from crfs_oracle.reach_progress import (  # noqa: E402
        ReachSnapshot,
        annotate_reach_snapshots,
    )
    from crfs_oracle.runner import OracleConfig  # noqa: E402
else:
    ACCEPTED_R01_SUMMARY_SHA256 = (
        "715d9326f02df531e0c6d7aa7b94629569b320800b4283c36fff41335d9b8bc5"
    )
    ARMS = (
        "frozen",
        "direct_witness",
        "random_residual",
        "analytic_geometry_residual",
        "oracle_residual",
        "bridge_diagnostic",
    )
    NORMALIZATION_ASSET_SHA256 = (
        "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84"
    )
    REGISTERED_TRANSLATION_ACTION_SCALE = (0.8422505, 0.827813, 0.937313)
    DIRECTION_REFERENCE = (
        "immutable R01 witness translation minus fresh paired eager translation"
    )
    DIRECTION_SEMANTICS_DECISION = (
        "docs/decisions/0013-reference-oracle-to-fresh-paired-baseline.md"
    )
    DIRECTION_SEMANTICS_DECISION_SHA256 = (
        "9af853d339059d8bbfade06f7d43f5ffe3303d89d98cbfa24158016813251b0c"
    )


CHECKPOINT_SHA = "988055ccfd7032903c073a641f3c5f0f0541df444a315116a16f0bf4716d26ed"


def _oracle(output_root: str) -> OracleConfig:
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
        response_matrix_m_per_action=REGISTERED_RESPONSE_MATRIX_M_PER_ACTION,
        optimizer_max_iterations=1,
        checkpoint_id="converted-pi05-libero",
        checkpoint_sha256=CHECKPOINT_SHA,
        output_root=output_root,
        run_id="r02-test",
    )


def _parity() -> dict:
    return valid_parity_artifact(pytorch_model_sha256=CHECKPOINT_SHA)


def _config_value(summary: Path, raw_root: Path, parity: Path) -> dict:
    repo_root = summary.parents[2]
    decision = repo_root / DIRECTION_SEMANTICS_DECISION
    decision.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / DIRECTION_SEMANTICS_DECISION, decision)
    return {
        "ready_to_run": True,
        "r02": {
            "phase": "pregrasp_reach",
            "required_arms": list(ARMS),
            "clipping_policy": "fail_without_clipping",
            "direction_reference": DIRECTION_REFERENCE,
            "direction_semantics_decision": DIRECTION_SEMANTICS_DECISION,
            "direction_semantics_decision_sha256": (
                DIRECTION_SEMANTICS_DECISION_SHA256
            ),
            "r01_summary_artifact": str(summary),
            "r01_summary_sha256": ACCEPTED_R01_SUMMARY_SHA256,
            "r01_results_root": str(raw_root),
            "sampler_parity_artifact": str(parity),
            "sampler_parity_sha256": file_sha256(parity),
            "target_object": "akita_black_bowl_1",
            "minimum_progress_m": 0.029897349105658888,
            "simulator_safety_margin_m": 0.005,
            "maximum_target_displacement_m": 0.001,
            "maximum_obstacle_displacement_m": 0.001,
            "simulator_repeats": 2,
            "translation_action_bounds": [-1.0, 1.0],
            "normalization_action_scale": list(
                REGISTERED_TRANSLATION_ACTION_SCALE
            ),
            "normalization_asset_sha256": NORMALIZATION_ASSET_SHA256,
            "analytic_samples_per_segment": 26,
        },
    }


class R02RunnerStructuralTest(unittest.TestCase):
    def test_six_arms_and_no_clipping_are_explicit(self) -> None:
        source = (ROOT / "main/crfs_oracle/r02_runner.py").read_text(encoding="utf-8")
        for arm in ARMS:
            self.assertIn(f'"{arm}"', source)
        self.assertIn('"clipping_policy": "fail_without_clipping"', source)
        self.assertNotIn("np.clip", source)

    def test_every_flow_arm_has_duplicate_replay_and_order_check(self) -> None:
        source = (ROOT / "main/crfs_oracle/r02_runner.py").read_text(encoding="utf-8")
        self.assertIn("duplicate_reply = _infer(", source)
        self.assertIn("policy_replay_exact", source)
        self.assertIn("order_contamination_check_exact", source)
        self.assertIn("final_compiled = _infer(", source)

    def test_no_witness_path_still_executes_frozen_arm(self) -> None:
        source = (ROOT / "main/crfs_oracle/r02_runner.py").read_text(encoding="utf-8")
        self.assertIn("def _no_witness_arms(frozen_arm", source)
        self.assertIn("no-witness artifact must still evaluate the frozen arm", source)
        self.assertIn("no-witness artifact must reconfirm the nominal collision", source)

    def test_reconfirmation_failures_are_terminal_artifacts_before_flow_arms(self) -> None:
        source = (ROOT / "main/crfs_oracle/r02_runner.py").read_text(encoding="utf-8")
        self.assertIn('"nominal_collision_not_reconfirmed"', source)
        self.assertIn('"direct_witness_not_reconfirmed"', source)
        self.assertIn('"not_evaluated_after_reconfirmation_failure"', source)
        self.assertIn("atomic_write_json(output, result)", source)
        direct_failure = source.index("status=DIRECT_RECONFIRMATION_FAILURE")
        flow_loop = source.index("for name, (mechanism, mode, direction_key) in arm_specs.items()")
        self.assertLess(direct_failure, flow_loop)

    def test_historical_drift_gate_precedes_any_simulator_rollout(self) -> None:
        source = (ROOT / "main/crfs_oracle/r02_runner.py").read_text(encoding="utf-8")
        run_source = source[source.index("def run_r02_case(") :]
        drift_gate = run_source.index('"historical_r01_drift_within_frozen_limits"')
        first_rollout = run_source.index('frozen_arm = _evaluated_arm(')
        self.assertLess(drift_gate, first_rollout)

    def test_policy_timing_is_descriptive_and_arm_specific(self) -> None:
        source = (ROOT / "main/crfs_oracle/r02_runner.py").read_text(encoding="utf-8")
        self.assertIn('"descriptive_only": True', source)
        self.assertIn("primary_reply=eager", source)
        self.assertIn("duplicate_reply=duplicate_eager", source)
        self.assertIn("primary_reply=reply", source)
        self.assertIn("duplicate_reply=duplicate_reply", source)
        self.assertIn("direct arm executes the immutable R01 action witness", source)
        self.assertNotIn("policy_timing_gate", source)

    @unittest.skipIf(np is None, "JSON canonicalization test needs runtime imports")
    def test_reach_snapshot_json_canonicalization_preserves_exact_coordinates(self) -> None:
        live = ReachSnapshot(
            target_object_name="akita_black_bowl_1",
            active_obstacle_name="obstacle",
            eef_world_m=(0.1, 0.2, 0.3),
            target_world_m=(0.4, 0.5, 0.6),
            active_obstacle_world_m=(0.7, 0.8, 0.9),
        ).to_dict()
        stored = json.loads(json.dumps(live))
        self.assertNotEqual(live, stored)
        self.assertEqual(_json_compatible(live), stored)


@unittest.skipIf(np is None, "R02 runtime checks execute in the allocation dependency environment")
class R02ConfigBindingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.summary = self.root / "evidence/r01/r01-summary.json"
        self.summary.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "evidence/r01/r01-summary.json", self.summary)
        self.parity = self.root / "parity.json"
        self.parity.write_text(json.dumps(_parity()), encoding="utf-8")
        self.raw = self.root / "raw"
        self.value = _config_value(self.summary, self.raw, self.parity)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_binds_exact_summary_raw_hashes_and_passing_parity(self) -> None:
        config = r02_config_from_mapping(
            self.value, _oracle(str(self.root / "out")), repo_root=self.root
        )

        self.assertEqual(len(config.r01_result_hashes), 20)
        self.assertEqual(len(config.eligible_case_ids), 17)
        self.assertEqual(len(config.no_witness_case_ids), 3)
        normalized = _normalized_config(config)
        self.assertEqual(normalized["r01_summary_sha256"], ACCEPTED_R01_SUMMARY_SHA256)
        self.assertNotIn(str(self.raw), json.dumps(normalized))
        self.assertEqual(tuple(normalized["required_arms"]), ARMS)

    def test_refuses_failed_or_malformed_parity(self) -> None:
        failed = _parity()
        failed["status"] = "failed"
        failed["acceptance"]["passed"] = False
        failed["acceptance"]["checks"][
            "primary_eager_final_normalized_within_limits"
        ] = False
        self.parity.write_text(json.dumps(failed), encoding="utf-8")
        self.value["r02"]["sampler_parity_sha256"] = file_sha256(self.parity)

        with self.assertRaisesRegex(ValueError, "parity"):
            r02_config_from_mapping(
                self.value, _oracle(str(self.root / "out")), repo_root=self.root
            )

    def test_refuses_shallow_self_asserted_parity(self) -> None:
        shallow = {
            "schema_version": "1.0",
            "artifact_type": "sampler_parity",
            "gate": "R02",
            "status": "passed",
            "identity": {
                "r01_summary_sha256": ACCEPTED_R01_SUMMARY_SHA256,
                "num_steps": 10,
                "action_horizon": 10,
                "action_dim": 32,
            },
            "acceptance": {"passed": True, "checks": {"self_claim": True}},
            "comparison": {},
            "checkpoints": {},
        }
        self.parity.write_text(json.dumps(shallow), encoding="utf-8")
        self.value["r02"]["sampler_parity_sha256"] = file_sha256(self.parity)
        with self.assertRaisesRegex(ValueError, "authoritative parity"):
            r02_config_from_mapping(
                self.value, _oracle(str(self.root / "out")), repo_root=self.root
            )

    def test_freezes_dsim_radius_distance_limit_and_h04_response(self) -> None:
        baseline = _oracle(str(self.root / "out"))
        for changed in (
            replace(baseline, eef_radius_m=0.05),
            replace(baseline, distance_limit_m=0.5),
            replace(
                baseline,
                response_matrix_m_per_action=((0.01, 0.0, 0.0),) * 3,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "radius|distance|response"):
                r02_config_from_mapping(self.value, changed, repo_root=self.root)

    def test_arm_order_and_no_clip_policy_are_frozen(self) -> None:
        self.value["r02"]["required_arms"] = list(reversed(ARMS))
        with self.assertRaisesRegex(ValueError, "required_arms"):
            r02_config_from_mapping(
                self.value, _oracle(str(self.root / "out")), repo_root=self.root
            )

    def test_binds_fresh_direction_reference_and_adr_content(self) -> None:
        for field, replacement in (
            ("direction_reference", "stale R01 nominal reference"),
            ("direction_semantics_decision", "docs/decisions/other.md"),
            ("direction_semantics_decision_sha256", "0" * 64),
        ):
            changed = copy.deepcopy(self.value)
            changed["r02"][field] = replacement
            with self.assertRaisesRegex(ValueError, "direction|ADR-0013"):
                r02_config_from_mapping(
                    changed, _oracle(str(self.root / "out")), repo_root=self.root
                )


@unittest.skipIf(np is None, "R02 runtime checks execute in the allocation dependency environment")
class R02DirectionIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        summary = root / "evidence/r01/r01-summary.json"
        summary.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "evidence/r01/r01-summary.json", summary)
        parity = root / "parity.json"
        parity.write_text(json.dumps(_parity()), encoding="utf-8")
        self.config = r02_config_from_mapping(
            _config_value(summary, root / "raw", parity),
            _oracle(str(root / "out")),
            repo_root=root,
        )

    def test_runner_calls_midpoint_random_and_exact_analytic_helpers(self) -> None:
        nominal = [[0.0] * 6 + [1.0] for _ in range(5)]
        witness = copy.deepcopy(nominal)
        witness[0][0] = 0.05
        attempt = {
            "candidate_index": 0,
            "source": "test_witness",
            "actions": witness,
            "simulator_safety_pass": True,
            "scene_stationary": True,
            "action_changed_from_nominal": True,
            "nontranslation_preserved": True,
            "direct_replay_exact": True,
            "verified_at_p_min": True,
            "changed_witness_at_p_min": True,
        }
        r01 = {
            "schema_version": "1.0",
            "gate": "R01",
            "status": "verified_safe_progress",
            "case_id": "case",
            "provenance": {
                "executed_action_horizon": 5,
                "model_action_horizon": 10,
                "model_action_dimension": 32,
                "random_control_seed": 7,
                "translation_action_bounds": [-1.0, 1.0],
            },
            "calibration": {"p_min_m": 0.01},
            "nominal": {"actions": nominal, "collision_reproduced": True},
            "verification": {
                "p_min": {"attempts": [attempt]},
                "p_zero": {"attempts": []},
            },
            "outcome": {"p_min_verified": True},
        }
        predicted = np.zeros((10, 7), dtype=np.float64)
        predicted[:5, 0] = 0.5
        fresh_eager = np.zeros((10, 7), dtype=np.float64)
        fresh_eager[:5, 0] = 0.01
        branch = {
            "start_eef_center_m": [0.0, 0.0, 0.0],
            "branch_obstacle_boxes": [
                {
                    "name": "obstacle",
                    "center_m": [0.1, 0.0, 0.0],
                    "half_size_m": [0.01, 0.02, 0.02],
                    "rotation_world": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                }
            ],
        }

        _, pointer, directions, failures, diagnostics = _direction_bundle(
            r01, self.config, fresh_eager, predicted, branch
        )

        self.assertIsNone(failures["random_model"])
        self.assertIsNone(failures["analytic_geometry_model"])
        self.assertIsNotNone(diagnostics["analytic_geometry"])
        self.assertEqual(pointer["actions_sha256"], content_hash(witness))
        oracle_norm = np.linalg.norm(directions["delta_star_model"])
        self.assertAlmostEqual(np.linalg.norm(directions["random_model"]), oracle_norm)
        self.assertAlmostEqual(
            np.linalg.norm(directions["analytic_geometry_model"]), oracle_norm
        )
        self.assertAlmostEqual(directions["delta_star_physical"][0, 0], 0.04)
        self.assertNotEqual(
            directions["delta_star_physical"][0, 0],
            np.asarray(
                _historical_r01_diagnostic(
                    nominal,
                    fresh_eager[:5, :7],
                    witness_actions=witness,
                )["raw_r01_delta_star_physical"]["values"],
                dtype=np.float64,
            )[0, 0],
        )

    def test_historical_drift_uses_frozen_limits_without_exact_identity(self) -> None:
        nominal = np.zeros((5, 7), dtype=np.float64)
        within = np.array(nominal, copy=True)
        within[:, 0] = 0.005
        accepted = _historical_r01_diagnostic(nominal, within)
        self.assertTrue(accepted["passed"])
        self.assertFalse(accepted["fresh_eager_vs_raw_r01_nominal"]["array_equal"])
        self.assertEqual(
            accepted["first_five_translation_drift"][
                "maximum_absolute_error_limit"
            ],
            0.010,
        )
        outside = np.array(nominal, copy=True)
        outside[:, 0] = 0.020
        rejected = _historical_r01_diagnostic(nominal, outside)
        self.assertFalse(rejected["passed"])
        self.assertFalse(rejected["first_five_translation_drift"]["passed"])


@unittest.skipIf(np is None, "R02 runtime checks execute in the allocation dependency environment")
class R02NoWitnessArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        summary = root / "evidence/r01/r01-summary.json"
        summary.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "evidence/r01/r01-summary.json", summary)
        parity = root / "parity.json"
        parity.write_text(json.dumps(_parity()), encoding="utf-8")
        self.config = r02_config_from_mapping(
            _config_value(summary, root / "raw", parity),
            _oracle(str(root / "out")),
            repo_root=root,
        )
        self.case = {
            "case_id": "no-witness",
            "task_suite": "safelibero_spatial",
            "safety_level": "II",
            "task_index": 0,
            "episode_index": 1,
            "group_id": "episode-1",
            "environment_seed": 2,
            "policy_seed": 3,
            "random_control_seed": 4,
        }
        actions = np.zeros((10, 7), dtype=np.float64)
        trace = _trace_record(
            {
                "step_index": np.asarray(5, dtype=np.int64),
                "time": np.asarray(0.5, dtype=np.float32),
                "x_t": np.zeros((10, 32), dtype=np.float32),
                "v_base": np.zeros((10, 32), dtype=np.float32),
                "predicted_clean": np.zeros((10, 32), dtype=np.float32),
                "predicted_clean_physical": np.zeros((10, 7), dtype=np.float32),
            }
        )
        self.pairing = {
            "applicable": True,
            "passed": True,
            "policy_observation": {"sha256": "1" * 64, "leaves": []},
            "branch_snapshot": {},
            "r01_branch_snapshot": {},
            "r01_nominal_actions": _array_record(actions[:5, :7]),
            "historical_r01_diagnostic": _historical_r01_diagnostic(
                actions[:5, :7], actions[:5, :7]
            ),
            "compiled_actions": _array_record(actions),
            "eager_actions": _array_record(actions),
            "duplicate_eager_actions": _array_record(actions),
            "final_compiled_actions": _array_record(actions),
            "final_eager_actions": _array_record(actions),
            "eager_trace": trace,
            "duplicate_eager_trace": copy.deepcopy(trace),
            "final_eager_trace": copy.deepcopy(trace),
            "compiled_vs_eager_diagnostic": {
                "shape_equal": True,
                "array_equal": True,
                "maximum_absolute_error": 0.0,
                "rms_absolute_error": 0.0,
            },
            "duplicate_eager_actions_exact": True,
            "duplicate_eager_trace_exact": True,
            "branch_snapshot_equals_r01_exact": True,
            "historical_r01_drift_within_frozen_limits": True,
            "order_contamination_check_exact": True,
            "eager_order_contamination_check_exact": True,
        }
        branch_snapshot = ReachSnapshot(
            target_object_name="akita_black_bowl_1",
            active_obstacle_name="obstacle",
            eef_world_m=(0.0, 0.0, 0.0),
            target_world_m=(1.0, 0.0, 0.0),
            active_obstacle_world_m=(0.0, 1.0, 0.0),
        )
        end_snapshot = ReachSnapshot(
            target_object_name="akita_black_bowl_1",
            active_obstacle_name="obstacle",
            eef_world_m=(0.04, 0.0, 0.0),
            target_world_m=(1.0, 0.0, 0.0),
            active_obstacle_world_m=(0.0, 1.0, 0.0),
        )
        reach = annotate_reach_snapshots(
            branch_snapshot,
            end_snapshot,
            executed_actions=5,
            maximum_target_displacement_m=0.0,
            maximum_active_obstacle_displacement_m=0.0,
        ).to_dict()
        trial = {
            "clearance_m": -0.001,
            "raw_mujoco_clearance_m": -0.001,
            "contact": False,
            "measurement_samples": 126,
            "minimum_geom_pair": ["crfs_eef_sphere", "obstacle_geom"],
            "raw_mujoco_minimum_geom_pair": ["eef_geom", "obstacle_geom"],
            "measurement": {
                "conservative_clearance_m": -0.001,
                "min_clearance_m": -0.001,
                "contact": False,
                "samples": 126,
                "distance_limit_m": 1.0,
                "conservative_eef_center_m": [0.069, 0.0, 0.0],
                "conservative_eef_radius_m": 0.06,
                "conservative_obstacle_center_m": [0.0, 0.0, 0.0],
                "conservative_obstacle_rotation_world": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
                "conservative_obstacle_half_size_m": [0.01, 0.01, 0.01],
                "conservative_obstacle_geom": "obstacle_geom",
                "min_pair": ["eef_geom", "obstacle_geom"],
            },
            "tracked_body_motion": {
                "substep_samples": 125,
                "branch_eef_world_m": [0.0, 0.0, 0.0],
                "end_eef_world_m": [0.04, 0.0, 0.0],
                "bodies": {
                    "akita_black_bowl_1": {
                        "branch_world_m": [1.0, 0.0, 0.0],
                        "end_world_m": [1.0, 0.0, 0.0],
                        "maximum_displacement_m": 0.0,
                        "endpoint_displacement_m": 0.0,
                    },
                    "obstacle": {
                        "branch_world_m": [0.0, 1.0, 0.0],
                        "end_world_m": [0.0, 1.0, 0.0],
                        "maximum_displacement_m": 0.0,
                        "endpoint_displacement_m": 0.0,
                    },
                },
            },
            "reach": reach,
        }
        self.frozen = {
            "mechanism": "compiled explicit-noise frozen sampler",
            "applicable": True,
            "status": "failed_gate",
            "controls": {"intervention_mode": "none"},
            "correction": None,
            "full_actions": _array_record(actions),
            "executed_actions": _array_record(actions[:5, :7]),
            "bounds": {
                "checked": True,
                "passed": True,
                "low": -1.0,
                "high": 1.0,
                "violations": [],
                "clipped": False,
            },
            "trace_sha256": None,
            "preintervention_trace_exact_to_frozen": None,
            "policy_replay_exact": None,
            "policy_duplicate_actions": None,
            "policy_duplicate_trace": None,
            "replay_exact": True,
            "repeats": [copy.deepcopy(trial), copy.deepcopy(trial)],
            "gate": {"passed": False, "trial_checks": []},
        }
        noise = np.random.default_rng(3).normal(size=(10, 32)).astype(np.float32)
        self.provenance = {
            "git_commit": "g",
            "git_dirty": False,
            "baseline_commit": "b",
            "host": "worker",
            "device": "0",
            "slurm_job_id": "1",
            "slurm_array_task_id": "0",
            "partition": "main",
            "case_record": self.case,
            "case_record_sha256": content_hash(self.case),
            "input_manifest_sha256": "1" * 64,
            "environment_seed": 2,
            "policy_seed": 3,
            "random_control_seed": 4,
            "checkpoint_id": "checkpoint",
            "checkpoint_sha256": CHECKPOINT_SHA,
            "r01_summary_sha256": ACCEPTED_R01_SUMMARY_SHA256,
            "r01_case_result_sha256": "2" * 64,
            "sampler_parity_sha256": self.config.parity_artifact_sha256,
            "direction_reference": DIRECTION_REFERENCE,
            "direction_semantics_decision": DIRECTION_SEMANTICS_DECISION,
            "direction_semantics_decision_sha256": (
                DIRECTION_SEMANTICS_DECISION_SHA256
            ),
            "noise": _array_record(noise),
            "sampler_steps": 10,
            "intervention_step": 5,
            "model_action_horizon": 10,
            "executed_action_horizon": 5,
            "model_action_dimension": 32,
            "action_frame": "world",
            "normalization_space": "scale only",
            "normalization_asset_sha256": NORMALIZATION_ASSET_SHA256,
            "normalization_action_scale": list(REGISTERED_TRANSLATION_ACTION_SCALE),
            "translation_action_bounds": [-1.0, 1.0],
            "clipping_policy": "fail_without_clipping",
            "d_opt_model": "proxy",
            "d_sim_model": "simulator",
            "eef_radius_m": 0.06,
            "distance_limit_m": 1.0,
            "response_matrix_m_per_action": [
                list(row) for row in REGISTERED_RESPONSE_MATRIX_M_PER_ACTION
            ],
            "simulator_safety_margin_m": 0.005,
            "minimum_progress_m": 0.029897349105658888,
            "maximum_target_displacement_m": 0.001,
            "maximum_obstacle_displacement_m": 0.001,
            "simulator_repeats": 2,
            "measurement_samples_per_trial": 126,
            "required_arms": list(ARMS),
        }

    def _result(self) -> dict:
        return _no_witness_result(
            self.case,
            self.config,
            config_hash="3" * 64,
            provenance=self.provenance,
            raw_r01_path=Path("/raw/no-witness/endpoint-free-feasibility.json"),
            raw_r01={"status": "no_verified_safe_progress"},
            raw_r01_sha256="2" * 64,
            pairing=self.pairing,
            frozen_arm=self.frozen,
        )

    def _safe_trial(self) -> dict:
        trial = copy.deepcopy(self.frozen["repeats"][0])
        trial["clearance_m"] = 0.01
        trial["raw_mujoco_clearance_m"] = 0.01
        trial["contact"] = False
        measurement = trial["measurement"]
        measurement["conservative_clearance_m"] = 0.01
        measurement["min_clearance_m"] = 0.01
        measurement["contact"] = False
        measurement["conservative_eef_center_m"] = [0.08, 0.0, 0.0]
        trial["start_eef_center_m"] = [0.0, 0.0, 0.0]
        trial["branch_obstacle_boxes"] = [
            {
                "name": "obstacle",
                "center_m": [0.1, 0.0, 0.0],
                "half_size_m": [0.01, 0.02, 0.02],
                "rotation_world": [
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
            }
        ]
        return trial

    def _nominal_mismatch_result(self) -> dict:
        frozen = copy.deepcopy(self.frozen)
        safe = self._safe_trial()
        frozen["status"] = "passed_gate"
        frozen["repeats"] = [copy.deepcopy(safe), copy.deepcopy(safe)]
        frozen["gate"]["passed"] = True
        return _reconfirmation_failure_result(
            self.case,
            self.config,
            status=NOMINAL_RECONFIRMATION_FAILURE,
            eligible=False,
            config_hash="3" * 64,
            provenance=self.provenance,
            raw_r01_path=Path("/raw/no-witness/endpoint-free-feasibility.json"),
            raw_r01={"status": "no_verified_safe_progress"},
            raw_r01_sha256="2" * 64,
            pairing=self.pairing,
            frozen_arm=frozen,
            r01_pointer=None,
            witness_pointer=None,
        )

    def _eligible_result(self) -> dict:
        case = {**self.case, "case_id": "eligible"}
        provenance = {
            **self.provenance,
            "case_record": case,
            "case_record_sha256": content_hash(case),
        }
        frozen = copy.deepcopy(self.frozen)
        for repeat in frozen["repeats"]:
            repeat["start_eef_center_m"] = [0.0, 0.0, 0.0]
            repeat["branch_obstacle_boxes"] = [
                {
                    "name": "obstacle",
                    "center_m": [0.1, 0.0, 0.0],
                    "half_size_m": [0.01, 0.02, 0.02],
                    "rotation_world": [
                        1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0,
                    ],
                }
            ]
        nominal = [[0.0] * 7 for _ in range(5)]
        witness_actions = copy.deepcopy(nominal)
        witness_actions[0][0] = 0.05
        attempt = {
            "candidate_index": 0,
            "source": "eligible_fixture",
            "actions": witness_actions,
            "simulator_safety_pass": True,
            "scene_stationary": True,
            "action_changed_from_nominal": True,
            "nontranslation_preserved": True,
            "direct_replay_exact": True,
            "verified_at_p_min": True,
            "changed_witness_at_p_min": True,
        }
        raw_r01 = {
            "schema_version": "1.0",
            "gate": "R01",
            "status": "verified_safe_progress",
            "case_id": case["case_id"],
            "provenance": {
                "executed_action_horizon": 5,
                "model_action_horizon": 10,
                "model_action_dimension": 32,
                "random_control_seed": case["random_control_seed"],
                "translation_action_bounds": [-1.0, 1.0],
            },
            "calibration": {"p_min_m": 0.01},
            "nominal": {"actions": nominal, "collision_reproduced": True},
            "verification": {
                "p_min": {"attempts": [attempt]},
                "p_zero": {"attempts": []},
            },
            "outcome": {"p_min_verified": True},
        }
        predicted = np.zeros((10, 7), dtype=np.float64)
        predicted[:5, 0] = 0.5
        eligible_trace = _trace_record(
            {
                "step_index": np.asarray(5, dtype=np.int64),
                "time": np.asarray(0.5, dtype=np.float32),
                "x_t": np.zeros((10, 32), dtype=np.float32),
                "v_base": np.zeros((10, 32), dtype=np.float32),
                "predicted_clean": np.zeros((10, 32), dtype=np.float32),
                "predicted_clean_physical": predicted.astype(np.float32),
            }
        )
        eligible_pairing = copy.deepcopy(self.pairing)
        fresh_eager = np.zeros((10, 7), dtype=np.float64)
        fresh_eager[:5, 0] = 0.005
        for key in (
            "compiled_actions",
            "eager_actions",
            "duplicate_eager_actions",
            "final_compiled_actions",
            "final_eager_actions",
        ):
            eligible_pairing[key] = _array_record(fresh_eager)
        eligible_pairing["historical_r01_diagnostic"] = (
            _historical_r01_diagnostic(
                nominal,
                fresh_eager[:5, :7],
                witness_actions=witness_actions,
            )
        )
        eligible_pairing["compiled_vs_eager_diagnostic"] = _path_diagnostic(
            fresh_eager, fresh_eager
        )
        for key in ("eager_trace", "duplicate_eager_trace", "final_eager_trace"):
            eligible_pairing[key] = copy.deepcopy(eligible_trace)
        _, pointer, values, failures, diagnostics = _direction_bundle(
            raw_r01,
            self.config,
            fresh_eager,
            predicted,
            frozen["repeats"][0],
        )
        self.assertIsNotNone(values["random_model"])
        self.assertIsNotNone(values["analytic_geometry_model"])
        direction_records = _direction_records(
            values,
            failures,
            witness_pointer=pointer,
            config=self.config,
            diagnostics=diagnostics,
        )
        safe = self._safe_trial()
        frozen["full_actions"] = _array_record(fresh_eager)
        frozen["executed_actions"] = _array_record(fresh_eager[:5, :7])

        def evaluated_arm(
            name: str,
            actions: np.ndarray,
            *,
            correction: np.ndarray | None = None,
            flow: bool = False,
        ) -> dict:
            arm = {
                "mechanism": name,
                "applicable": True,
                "status": "passed_gate",
                "controls": {"intervention_mode": "residual" if flow else "direct"},
                "correction": (
                    _array_record(correction, dtype=np.float32)
                    if correction is not None
                    else None
                ),
                "full_actions": _array_record(actions),
                "executed_actions": _array_record(actions[:5, :7]),
                "bounds": {
                    "checked": True,
                    "passed": True,
                    "low": -1.0,
                    "high": 1.0,
                    "violations": [],
                    "clipped": False,
                },
                "trace_sha256": None,
                "preintervention_trace_exact_to_frozen": None,
                "policy_replay_exact": None,
                "policy_duplicate_actions": None,
                "policy_duplicate_trace": None,
                "replay_exact": True,
                "repeats": [copy.deepcopy(safe), copy.deepcopy(safe)],
                "gate": {"passed": True, "trial_checks": []},
            }
            if flow:
                arm.update(
                    {
                        "trace_sha256": eligible_pairing["eager_trace"]["sha256"],
                        "preintervention_trace_exact_to_frozen": True,
                        "policy_replay_exact": True,
                        "policy_duplicate_actions": _array_record(actions),
                        "policy_duplicate_trace": copy.deepcopy(
                            eligible_pairing["eager_trace"]
                        ),
                    }
                )
            return arm

        direct_actions = np.asarray(witness_actions, dtype=np.float64)
        arms = {
            "frozen": frozen,
            "direct_witness": evaluated_arm("direct_witness", direct_actions),
        }
        direction_for_arm = {
            "random_residual": "random_model",
            "analytic_geometry_residual": "analytic_geometry_model",
            "oracle_residual": "delta_star_model",
            "bridge_diagnostic": "delta_star_model",
        }
        flow_actions = np.zeros((10, 7), dtype=np.float64)
        for name, key in direction_for_arm.items():
            arms[name] = evaluated_arm(
                name,
                flow_actions,
                correction=np.asarray(values[key], dtype=np.float64),
                flow=True,
            )
        r01_pointer = {
            "search": pointer["search"],
            "candidate_index": pointer["candidate_index"],
            "source": pointer["source"],
            "actions_sha256": pointer["actions_sha256"],
        }
        gates = {name: bool(arms[name]["gate"]["passed"]) for name in ARMS}
        return {
            "schema_version": "1.0",
            "artifact_type": "paired_oracle_flow_case",
            "gate": "R02",
            "case_id": case["case_id"],
            "run_id": self.config.oracle.run_id,
            "status": "completed",
            "config_hash": "3" * 64,
            "source_evidence": {
                "r01_summary_sha256": self.config.r01_summary_sha256,
                "r01_ordered_result_set_digest": ACCEPTED_R01_ORDERED_RESULTS_SHA256,
                "r01_case_path": "/raw/eligible/endpoint-free-feasibility.json",
                "r01_case_sha256": "2" * 64,
                "r01_case_validator": "validate_endpoint_free_result:passed",
                "r01_case_status": "verified_safe_progress",
                "r01_selected_p_min_changed_witness": r01_pointer,
                "selected_witness_pointer": pointer,
                "sampler_parity_sha256": self.config.parity_artifact_sha256,
                "sampler_parity_status": "passed",
                "direction_reference": DIRECTION_REFERENCE,
                "direction_semantics_decision": DIRECTION_SEMANTICS_DECISION,
                "direction_semantics_decision_sha256": (
                    DIRECTION_SEMANTICS_DECISION_SHA256
                ),
            },
            "provenance": provenance,
            "pairing": eligible_pairing,
            "directions": direction_records,
            "arms": arms,
            "outcome": {
                "population": "r01_changed_action_p_min_witness_conditioned",
                "r01_feasible_conditioned": True,
                "nominal_collision_reproduced": True,
                "direct_witness_reconfirmed": True,
                "population_mismatch_reason": None,
                "arm_gate_pass": gates,
                "arm_status": {name: arms[name]["status"] for name in ARMS},
            },
        }

    def _direct_mismatch_result(self) -> dict:
        completed = self._eligible_result()
        direct = copy.deepcopy(completed["arms"]["direct_witness"])
        direct["status"] = "failed_gate"
        direct["repeats"] = copy.deepcopy(completed["arms"]["frozen"]["repeats"])
        direct["gate"]["passed"] = False
        return _reconfirmation_failure_result(
            completed["provenance"]["case_record"],
            self.config,
            status=DIRECT_RECONFIRMATION_FAILURE,
            eligible=True,
            config_hash=completed["config_hash"],
            provenance=completed["provenance"],
            raw_r01_path=Path(completed["source_evidence"]["r01_case_path"]),
            raw_r01={"status": "verified_safe_progress"},
            raw_r01_sha256=completed["source_evidence"]["r01_case_sha256"],
            pairing=completed["pairing"],
            frozen_arm=completed["arms"]["frozen"],
            r01_pointer=completed["source_evidence"][
                "r01_selected_p_min_changed_witness"
            ],
            witness_pointer=completed["source_evidence"][
                "selected_witness_pointer"
            ],
            direction_records=completed["directions"],
            direct_arm=direct,
        )

    def _eligible_nominal_mismatch_result(self) -> dict:
        completed = self._eligible_result()
        frozen = copy.deepcopy(completed["arms"]["frozen"])
        safe = self._safe_trial()
        frozen["status"] = "passed_gate"
        frozen["repeats"] = [copy.deepcopy(safe), copy.deepcopy(safe)]
        frozen["gate"]["passed"] = True
        return _reconfirmation_failure_result(
            completed["provenance"]["case_record"],
            self.config,
            status=NOMINAL_RECONFIRMATION_FAILURE,
            eligible=True,
            config_hash=completed["config_hash"],
            provenance=completed["provenance"],
            raw_r01_path=Path(completed["source_evidence"]["r01_case_path"]),
            raw_r01={"status": "verified_safe_progress"},
            raw_r01_sha256=completed["source_evidence"]["r01_case_sha256"],
            pairing=completed["pairing"],
            frozen_arm=frozen,
            r01_pointer=completed["source_evidence"][
                "r01_selected_p_min_changed_witness"
            ],
            witness_pointer=completed["source_evidence"][
                "selected_witness_pointer"
            ],
        )

    def test_no_witness_still_requires_exact_frozen_collision_replay(self) -> None:
        result = self._result()
        self.assertEqual(validate_r02_result(result), [])
        reloaded = json.loads(json.dumps(result, sort_keys=True))
        self.assertEqual(validate_r02_result(reloaded), [])
        self.assertEqual(result["arms"]["frozen"]["status"], "failed_gate")
        self.assertTrue(result["outcome"]["nominal_collision_reproduced"])
        self.assertIsNone(
            result["pairing"]["historical_r01_diagnostic"]["delta_comparison"]
        )
        for name in ARMS[1:]:
            self.assertEqual(
                result["arms"][name]["status"],
                "not_applicable_no_r01_witness",
            )

    def test_nominal_mismatch_is_valid_atomic_payload_with_all_later_arms_unrun(self) -> None:
        result = self._nominal_mismatch_result()
        self.assertEqual(validate_r02_result(result), [])
        self.assertEqual(
            validate_r02_result(json.loads(json.dumps(result, sort_keys=True))), []
        )
        self.assertEqual(result["status"], NOMINAL_RECONFIRMATION_FAILURE)
        self.assertFalse(result["outcome"]["nominal_collision_reproduced"])
        for name in ARMS[1:]:
            self.assertEqual(
                result["arms"][name]["status"],
                NOT_EVALUATED_AFTER_RECONFIRMATION_FAILURE,
            )
            self.assertEqual(result["arms"][name]["repeats"], [])

    def test_nominal_mismatch_rejects_false_claim_or_unrun_execution(self) -> None:
        result = self._nominal_mismatch_result()
        result["outcome"]["nominal_collision_reproduced"] = True
        errors = validate_r02_result(result)
        self.assertTrue(any("nominal mismatch" in error for error in errors), errors)

        result = self._nominal_mismatch_result()
        result["arms"]["oracle_residual"]["full_actions"] = _array_record(
            np.zeros((10, 7), dtype=np.float64)
        )
        errors = validate_r02_result(result)
        self.assertTrue(any("not-evaluated" in error for error in errors), errors)

    def test_eligible_nominal_mismatch_retains_historical_source_without_directions(self) -> None:
        result = self._eligible_nominal_mismatch_result()
        self.assertEqual(validate_r02_result(result), [])
        self.assertIsNotNone(
            result["pairing"]["historical_r01_diagnostic"][
                "raw_r01_delta_star_physical"
            ]
        )
        self.assertIsNone(result["directions"]["arrays"]["delta_star_physical"])

    def test_direct_mismatch_retains_direct_repeats_and_stops_flow_arms(self) -> None:
        result = self._direct_mismatch_result()
        self.assertEqual(validate_r02_result(result), [])
        self.assertEqual(
            validate_r02_result(json.loads(json.dumps(result, sort_keys=True))), []
        )
        self.assertEqual(result["status"], DIRECT_RECONFIRMATION_FAILURE)
        self.assertEqual(len(result["arms"]["direct_witness"]["repeats"]), 2)
        self.assertFalse(result["outcome"]["direct_witness_reconfirmed"])
        for name in ARMS[2:]:
            self.assertEqual(
                result["arms"][name]["status"],
                NOT_EVALUATED_AFTER_RECONFIRMATION_FAILURE,
            )

    def test_historical_nominal_drift_is_diagnostic_and_oracle_uses_fresh_eager(self) -> None:
        result = self._eligible_result()
        self.assertEqual(validate_r02_result(result), [])
        historical = result["pairing"]["historical_r01_diagnostic"]
        self.assertFalse(
            historical["fresh_eager_vs_raw_r01_nominal"]["array_equal"]
        )
        raw_delta = np.asarray(
            historical["raw_r01_delta_star_physical"]["values"],
            dtype=np.float64,
        )
        applied = np.asarray(
            result["directions"]["arrays"]["delta_star_physical"]["values"],
            dtype=np.float64,
        )
        self.assertAlmostEqual(raw_delta[0, 0], 0.05)
        self.assertAlmostEqual(applied[0, 0], 0.045)
        comparison = historical["delta_comparison"]
        self.assertAlmostEqual(
            comparison["raw_model_l2"],
            0.05 / REGISTERED_TRANSLATION_ACTION_SCALE[0],
        )
        self.assertGreater(comparison["fresh_model_l2"], 0.0)
        self.assertGreater(comparison["fresh_minus_raw_model_l2"], 0.0)
        self.assertGreaterEqual(comparison["cosine"], -1.0)
        self.assertLessEqual(comparison["cosine"], 1.0)
        self.assertGreater(comparison["angle_degrees"], 0.0)

    def test_historical_raw_delta_tampering_fails_closed(self) -> None:
        result = self._eligible_result()
        record = result["pairing"]["historical_r01_diagnostic"][
            "raw_r01_delta_star_physical"
        ]
        tampered = np.asarray(record["values"], dtype=np.float64)
        tampered[0, 0] += 0.001
        result["pairing"]["historical_r01_diagnostic"][
            "raw_r01_delta_star_physical"
        ] = _array_record(tampered)
        errors = validate_r02_result(result)
        self.assertTrue(any("historical R01 Delta_star" in item for item in errors), errors)

    def test_historical_drift_hash_or_error_tampering_fails_closed(self) -> None:
        result = self._eligible_result()
        historical = result["pairing"]["historical_r01_diagnostic"]
        historical["fresh_eager_actions_array_sha256"] = "0" * 64
        historical["first_five_translation_drift"][
            "maximum_absolute_error"
        ] = 0.0
        errors = validate_r02_result(result)
        self.assertTrue(any("fresh eager array hash" in item for item in errors), errors)
        self.assertTrue(any("translation drift diagnostic" in item for item in errors), errors)

    def test_historical_delta_norm_and_angle_tampering_fails_closed(self) -> None:
        result = self._eligible_result()
        comparison = result["pairing"]["historical_r01_diagnostic"][
            "delta_comparison"
        ]
        comparison["fresh_model_l2"] += 0.001
        comparison["angle_degrees"] += 1.0
        errors = validate_r02_result(result)
        self.assertTrue(any("fresh_model_l2" in item for item in errors), errors)
        self.assertTrue(any("angle_degrees" in item for item in errors), errors)

        missing = self._eligible_result()
        missing["pairing"]["historical_r01_diagnostic"]["delta_comparison"] = None
        errors = validate_r02_result(missing)
        self.assertTrue(any("Delta comparison" in item for item in errors), errors)

    def test_portable_reconstruction_accepts_only_float64_last_bit_drift(self) -> None:
        base = np.asarray([[1.0, -0.25], [0.0, 2.0]], dtype=np.float64)
        one_ulp = np.array(base, copy=True)
        one_ulp[0, 0] = np.nextafter(one_ulp[0, 0], np.inf)
        self.assertTrue(_portable_array_reconstruction_equal(base, one_ulp))

        meaningful = np.array(base, copy=True)
        meaningful[0, 0] += 2.0e-12
        self.assertFalse(_portable_array_reconstruction_equal(base, meaningful))

        diagnostic = {
            "selected_pair": {"segment_index": 4, "distance": 0.01},
            "direction": [[1.0, 0.0]],
        }
        portable = copy.deepcopy(diagnostic)
        portable["direction"][0][0] = float(
            np.nextafter(portable["direction"][0][0], np.inf)
        )
        self.assertTrue(
            _portable_diagnostic_reconstruction_equal(diagnostic, portable)
        )
        wrong_pair = copy.deepcopy(diagnostic)
        wrong_pair["selected_pair"]["segment_index"] = 3
        self.assertFalse(
            _portable_diagnostic_reconstruction_equal(diagnostic, wrong_pair)
        )

    def test_portable_validator_accepts_ulp_reconstruction_not_direction_change(self) -> None:
        portable = self._eligible_result()
        random_record = portable["directions"]["arrays"]["random_model"]
        random_direction = np.asarray(random_record["values"], dtype=np.float64)
        nonzero = np.argwhere(random_direction != 0.0)[0]
        row, column = (int(nonzero[0]), int(nonzero[1]))
        random_direction[row, column] = np.nextafter(
            random_direction[row, column], np.inf
        )
        portable["directions"]["arrays"]["random_model"] = _array_record(
            random_direction
        )
        portable["directions"]["l2_norms"]["random_model"] = float(
            np.linalg.norm(random_direction)
        )
        self.assertEqual(validate_r02_result(portable), [])

        changed = self._eligible_result()
        random_record = changed["directions"]["arrays"]["random_model"]
        random_direction = np.asarray(random_record["values"], dtype=np.float64)
        nonzero = np.argwhere(random_direction != 0.0)[0]
        row, column = (int(nonzero[0]), int(nonzero[1]))
        random_direction[row, column] += 2.0e-12
        changed["directions"]["arrays"]["random_model"] = _array_record(
            random_direction
        )
        changed["directions"]["l2_norms"]["random_model"] = float(
            np.linalg.norm(random_direction)
        )
        errors = validate_r02_result(changed)
        self.assertTrue(any("seeded equal-L2" in item for item in errors), errors)

    def test_completed_eligible_rejects_raw_and_constructed_direction_tampering(self) -> None:
        completed = self._eligible_result()
        self.assertEqual(validate_r02_result(completed), [])

        delta_tamper = copy.deepcopy(completed)
        delta = np.asarray(
            delta_tamper["directions"]["arrays"]["delta_star_physical"]["values"],
            dtype=np.float64,
        )
        delta[0, 0] += 0.001
        delta_tamper["directions"]["arrays"]["delta_star_physical"] = _array_record(delta)
        delta_tamper["directions"]["l2_norms"]["delta_star_physical"] = float(
            np.linalg.norm(delta)
        )
        errors = validate_r02_result(delta_tamper)
        self.assertTrue(
            any("immutable witness minus fresh paired eager" in error for error in errors),
            errors,
        )

        random_tamper = copy.deepcopy(completed)
        random_direction = -np.asarray(
            random_tamper["directions"]["arrays"]["random_model"]["values"],
            dtype=np.float64,
        )
        random_tamper["directions"]["arrays"]["random_model"] = _array_record(
            random_direction
        )
        random_tamper["arms"]["random_residual"]["correction"] = _array_record(
            random_direction, dtype=np.float32
        )
        errors = validate_r02_result(random_tamper)
        self.assertTrue(any("seeded equal-L2" in error for error in errors), errors)

        analytic_tamper = copy.deepcopy(completed)
        analytic_direction = -np.asarray(
            analytic_tamper["directions"]["arrays"]["analytic_geometry_model"][
                "values"
            ],
            dtype=np.float64,
        )
        analytic_tamper["directions"]["arrays"]["analytic_geometry_model"] = (
            _array_record(analytic_direction)
        )
        analytic_tamper["arms"]["analytic_geometry_residual"]["correction"] = (
            _array_record(analytic_direction, dtype=np.float32)
        )
        analytic_tamper["directions"]["diagnostics"]["analytic_geometry"][
            "direction_model"
        ] = analytic_direction.tolist()
        errors = validate_r02_result(analytic_tamper)
        self.assertTrue(any("exact D_opt" in error for error in errors), errors)

    def test_tampered_frozen_measurement_cannot_complete(self) -> None:
        result = self._result()
        result["arms"]["frozen"]["repeats"][0]["measurement_samples"] = 125
        errors = validate_r02_result(result)
        self.assertTrue(any("frozen" in error or "gate" in error for error in errors))

    def test_top_level_clearance_cannot_override_raw_measurement(self) -> None:
        result = self._result()
        for repeat in result["arms"]["frozen"]["repeats"]:
            repeat["clearance_m"] = 0.01
        errors = validate_r02_result(result)
        self.assertTrue(any("raw evidence" in error for error in errors), errors)

    def test_tampered_r01_branch_snapshot_cannot_complete(self) -> None:
        result = self._result()
        result["pairing"]["r01_branch_snapshot"] = {"eef_world_m": [1.0, 0.0, 0.0]}
        errors = validate_r02_result(result)
        self.assertTrue(any("R01 branch" in error for error in errors), errors)

    def test_compiled_eager_difference_is_diagnostic_not_pairing_failure(self) -> None:
        result = self._result()
        compiled = np.asarray(
            result["pairing"]["compiled_actions"]["values"], dtype=np.float64
        )
        eager = np.asarray(
            result["pairing"]["eager_actions"]["values"], dtype=np.float64
        )
        compiled[0, 0] = 0.001
        result["pairing"]["compiled_actions"] = _array_record(compiled)
        result["pairing"]["final_compiled_actions"] = _array_record(compiled)
        result["pairing"]["compiled_vs_eager_diagnostic"] = _path_diagnostic(
            compiled, eager
        )

        self.assertEqual(validate_r02_result(result), [])
        self.assertFalse(
            result["pairing"]["compiled_vs_eager_diagnostic"]["array_equal"]
        )

    def test_bounds_failure_is_recorded_without_clipping(self) -> None:
        actions = np.zeros((10, 7), dtype=np.float64)
        actions[0, 0] = 1.01
        check = _bounds_check(actions, self.config)
        self.assertFalse(check["passed"])
        self.assertFalse(check["clipped"])
        self.assertEqual(actions[0, 0], 1.01)


if __name__ == "__main__":
    unittest.main()
