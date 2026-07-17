import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main"


def _load_runner():
    """Load the dependency-light submodule without package __init__ side effects."""

    for name in ("crfs_oracle.aegis_runner", "crfs_oracle.aegis_pairing", "crfs_oracle"):
        sys.modules.pop(name, None)
    package = types.ModuleType("crfs_oracle")
    package.__path__ = [str(MAIN / "crfs_oracle")]
    sys.modules["crfs_oracle"] = package
    spec = importlib.util.spec_from_file_location(
        "crfs_oracle.aegis_runner", MAIN / "crfs_oracle" / "aegis_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AegisRunnerDependencyLightTest(unittest.TestCase):
    @staticmethod
    def _forged_valid_implementation(runner, *, capture):
        source_paths = {
            "runner": Path(runner.__file__).resolve(),
            "pairing": Path(runner.__file__).with_name("aegis_pairing.py"),
            "baseline_adapter": Path(runner.__file__).with_name(
                "aegis_baseline.py"
            ),
            "perception_adapter": Path(runner.__file__).with_name(
                "aegis_perception.py"
            ),
        }
        return {
            "source_git_commit": "b" * 40,
            "expected_git_commit": "b" * 40,
            "accepted_implementation_commit": "a" * 40,
            "release_direct_child_verified": True,
            "release_diff_paths": ["config.json", "decision.md"],
            "allowed_release_diff_paths": ["config.json", "decision.md"],
            "git_dirty": False,
            "runtime_type": (
                f"{runner.SafeLiberoAegisRuntime.__module__}."
                f"{runner.SafeLiberoAegisRuntime.__qualname__}"
            ),
            "provider_type": (
                None
                if capture
                else (
                    f"{runner.CodexFrozenLabelSafetyCoreProvider.__module__}."
                    f"{runner.CodexFrozenLabelSafetyCoreProvider.__qualname__}"
                )
            ),
            "source_sha256": {
                name: runner._sha256_path(path)
                for name, path in source_paths.items()
            },
        }

    @staticmethod
    def _forged_valid_model(runner):
        return {
            "checkpoint_model": {
                "path": str(runner.CHECKPOINT_MODEL_PATH),
                "sha256": runner.CHECKPOINT_SHA256,
            },
            "checkpoint_config": {
                "path": str(runner.CHECKPOINT_CONFIG_PATH),
                "sha256": runner.CHECKPOINT_CONFIG_SHA256,
            },
            "normalization_asset": {
                "path": str(runner.NORMALIZATION_ASSET_PATH),
                "sha256": runner.NORMALIZATION_ASSET_SHA256,
            },
        }

    def test_module_import_is_dependency_light_and_execution_is_slurm_guarded(self):
        aegis_runner = _load_runner()

        self.assertNotIn("mujoco", aegis_runner.__dict__)
        self.assertNotIn("cvxpy", aegis_runner.__dict__)
        self.assertNotIn("groundingdino", aegis_runner.__dict__)
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(aegis_runner.AegisApparatusError, "Slurm"):
                aegis_runner._assert_allocation()

    def test_draft_config_validates_sources_but_cannot_execute(self):
        runner = _load_runner()

        path = ROOT / "configs" / "experiments" / "r06_aegis_collision_conditioned.json"
        reviewed = runner.load_aegis_experiment_config(
            path, repo_root=ROOT, require_execution_release=False
        )
        self.assertEqual(reviewed.cases[0]["case_id"], runner.CANARY_CASE_ID)
        self.assertEqual(len(reviewed.cases), 20)
        self.assertFalse(reviewed.ready_to_run)
        self.assertIsNone(reviewed.execution_release)
        self.assertIn("base_decision", reviewed.raw["source_evidence"])
        self.assertIn("current_decision", reviewed.raw["source_evidence"])
        self.assertIn("pi05_plus_aegis_codex_label", reviewed.raw["arms"])
        with self.assertRaisesRegex(runner.AegisApparatusError, "not released"):
            runner.load_aegis_experiment_config(path, repo_root=ROOT)

    def test_loader_rejects_stratum_tampering_even_with_valid_json(self):
        runner = _load_runner()

        original = json.loads(
            (ROOT / "configs" / "experiments" / "r06_aegis_collision_conditioned.json").read_text()
        )
        original["strata"]["pre_settle_late_intervention"]["case_ids"][0] = original[
            "canary"
        ]["case_id"]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "tampered.json"
            path.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "late stratum IDs"):
                runner.load_aegis_experiment_config(
                    path, repo_root=ROOT, require_execution_release=False
                )

    def test_result_validator_retains_failures_and_rejects_silent_fallback(self):
        runner = _load_runner()

        base = {
            "schema_version": "1.0",
            "experiment_identity": "AEGIS-00A",
            "case_id": "crfs-1069f29a8d76463a",
            "stratum": "primary_eligible",
            "status": "method_failure",
            "failure": {
                "classification": "retained_method_failure",
                "kind": "qp_failure",
                "message": "no solution",
                "evidence": {"failed_action_index": 2},
                "silent_fallback_used": False,
            },
        }
        self.assertEqual(runner.validate_aegis_case_result(base), [])
        tampered = copy.deepcopy(base)
        tampered["failure"]["silent_fallback_used"] = True
        self.assertIn(
            "failure result must explicitly forbid fallback",
            runner.validate_aegis_case_result(tampered),
        )

    def test_source_contains_exact_pairing_render_and_measurement_contract(self):
        source = (MAIN / "crfs_oracle" / "aegis_runner.py").read_text()
        self.assertIn("select_latest_safe_settle_boundary", source)
        self.assertIn("fixed_policy_noise", source)
        self.assertIn("infer_duplicate_eager_actions", source)
        self.assertIn("render_aegis_views_1024", source)
        self.assertIn("render_advanced_physics", source)
        self.assertIn("literal_pre_settle_geometry", source)
        self.assertIn("MEASUREMENT_SAMPLES_PER_ARM = 1 +", source)
        self.assertIn("controller.observe_executed_state", source)
        self.assertIn("refresh_after_action", source)
        self.assertIn("resolve_original_perception_dependencies", source)
        self.assertIn("atomic_write_json", source)
        self.assertNotIn("env.sim.set_state", source)
        self.assertNotIn("oracle_geometry", source)
        self.assertNotIn("except TypeError:\n                        filtered", source)

        capture_source = source[
            source.index("def run_aegis_label_capture(") : source.index(
                "def run_aegis_case("
            )
        ]
        self.assertEqual(capture_source.count("_validate_branch_identity("), 2)

    def test_branch_identity_rejects_controller_fingerprint_drift(self):
        runner = _load_runner()
        common = {
            "boundary_index": 20,
            "integration_state": {},
            "camera_observation": {},
            "policy_observation": {},
            "prompt": "pick up the bowl",
            "active_obstacle": "akita_black_bowl_main",
            "literal_pre_settle_geometry": {"p1": [0.0, 0.0, 0.0]},
            "branch_robot_geometry": {},
            "branch_reach_snapshot": {},
        }
        reference = runner.BranchContext(
            controller_state={"fingerprint_sha256": "a" * 64}, **common
        )
        changed = runner.BranchContext(
            controller_state={"fingerprint_sha256": "b" * 64}, **common
        )

        with mock.patch.object(runner, "_json_value", side_effect=lambda value: value):
            self.assertEqual(
                runner._validate_branch_identity(reference, changed),
                ["canonical controller-state fingerprint differs"],
            )

    def test_forged_canary_boolean_cannot_bypass_independent_evidence(self):
        runner = _load_runner()
        repeat = {
            "measurement_samples": 126,
            "frozen_collision_definition": True,
            "controller_steps": [],
        }
        forged = {
            "schema_version": "1.0",
            "experiment_identity": "AEGIS-00A",
            "case_id": runner.CANARY_CASE_ID,
            "stratum": "primary_eligible",
            "status": "complete",
            "settle_selection": {
                "status": "selected",
                "selected_boundary_index": 20,
            },
            "policy": {
                "duplicate_eager_actions_exact": True,
                "instruction": "pick up the bowl",
            },
            "pi05_baseline": {"repeats": [repeat, repeat]},
            "pi05_plus_aegis_codex_label": {
                "repeats": [repeat, repeat],
                "controller_repeat_exact": True,
            },
            "pairing": {"errors": []},
            "canary_apparatus_valid": True,
        }
        errors = runner.validate_aegis_case_result(forged)
        self.assertIn("valid canary lacks selected-ledger branch binding", errors)
        self.assertIn("valid canary lacks accepted R02 byte/source binding", errors)
        self.assertIn("valid canary lacks exact live model/config/norm hashes", errors)
        self.assertIn(
            "valid canary lacks complete Codex/DINO/filter/MVEE evidence", errors
        )
        self.assertEqual(errors.count("valid canary lacks exactly five controller steps"), 2)

    def test_capture_validator_requires_no_aegis_and_four_hash_bound_arrays(self):
        runner = _load_runner()
        value = {
            "schema_version": "1.0",
            "experiment_identity": "AEGIS-00A",
            "status": "capture_complete",
            "case_id": runner.CANARY_CASE_ID,
            "aegis_executed": True,
            "qp_steps": 1,
        }
        errors = runner.validate_aegis_label_capture(value)
        self.assertIn("capture must not execute AEGIS or any QP", errors)
        self.assertIn("capture has no assets", errors)

    def test_shallow_fully_forged_canary_is_rejected(self):
        runner = _load_runner()
        source_hashes = {
            "main/main_aegis.py": runner.AEGIS_MAIN_SHA256,
            "main/main_aegis_translational.py": runner.AEGIS_TRANSLATIONAL_SHA256,
            "main/utils.py": runner.AEGIS_UTILS_SHA256,
        }
        qp = {
            "solver_status": "optimal",
            "qp_backend": "cvxpy_osqp_upstream",
            "coefficient_backend": "upstream_literal",
            "gripper_preserved_exactly": True,
            "initialization": {"literal_upstream_initialization": True},
            "upstream_source_sha256": source_hashes,
        }
        steps = [
            {"step_index": index, "qp": qp, "executed_action": [0.0] * 7}
            for index in range(5)
        ]
        base_repeat = {
            "measurement_samples": 126,
            "frozen_collision_definition": True,
        }
        forged = {
            "schema_version": runner.SCHEMA_VERSION,
            "experiment_identity": runner.EXPERIMENT_IDENTITY,
            "case_id": runner.CANARY_CASE_ID,
            "stratum": "primary_eligible",
            "status": "complete",
            "settle_selection": {
                "status": "selected",
                "selected_boundary_index": 20,
            },
            "selected_branch_binding": {
                "passed": True,
                "checks": {
                    "integration_state": True,
                    "active_obstacle": True,
                    "clearance": True,
                    "contact": True,
                },
            },
            "policy": {
                "duplicate_eager_actions_exact": True,
                "instruction": "pick up the bowl",
            },
            "pi05_baseline": {
                "repeats": [base_repeat, copy.deepcopy(base_repeat)],
                "repeat_exact": True,
                "collision_reproduced_both": True,
            },
            "pi05_plus_aegis_codex_label": {
                "repeats": [
                    {**base_repeat, "controller_steps": steps},
                    {**base_repeat, "controller_steps": copy.deepcopy(steps)},
                ],
                "controller_repeat_exact": True,
            },
            "pairing": {"errors": [], "passed": True},
            "prior_r02_pairing": {
                "sha256": runner.CANARY_R02_SHA256,
                "checkpoint_sha256": runner.CHECKPOINT_SHA256,
                "normalization_asset_sha256": (
                    runner.NORMALIZATION_ASSET_SHA256
                ),
                "checks": {
                    "case_id": True,
                    "checkpoint": True,
                    "normalization": True,
                    "observation": True,
                    "nominal_actions": True,
                },
            },
            "model_identity": self._forged_valid_model(runner),
            "implementation_identity": self._forged_valid_implementation(
                runner, capture=False
            ),
            "canary_apparatus_valid": True,
        }
        self.assertTrue(
            runner._implementation_identity_valid(
                forged["implementation_identity"], capture=False
            )
        )
        errors = runner.validate_aegis_case_result(forged)
        self.assertTrue(errors)
        self.assertIn("paired branch validation did not pass", errors)
        self.assertIn(
            "valid canary lacks complete Codex/DINO/filter/MVEE evidence",
            errors,
        )

    def test_shallow_fully_forged_capture_is_rejected(self):
        runner = _load_runner()
        forged = {
            "schema_version": runner.SCHEMA_VERSION,
            "experiment_identity": runner.EXPERIMENT_IDENTITY,
            "status": "capture_complete",
            "case_id": runner.CANARY_CASE_ID,
            "aegis_executed": False,
            "qp_steps": 0,
            "settle_selection": {"selected_boundary_index": 20},
            "selected_branch_binding": {"passed": True},
            "pi05_baseline": {
                "repeats": [{}, {}],
                "repeat_exact": True,
                "collision_reproduced_both": True,
            },
            "policy": {
                "duplicate_eager_actions_exact": True,
                "checkpoint_sha256": runner.CHECKPOINT_SHA256,
                "normalization_asset_sha256": (
                    runner.NORMALIZATION_ASSET_SHA256
                ),
            },
            "pairing": {"errors": [], "passed": True},
            "prior_r02_pairing": {
                "sha256": runner.CANARY_R02_SHA256,
                "checks": {
                    "case_id": True,
                    "checkpoint": True,
                    "normalization": True,
                    "observation": True,
                    "nominal_actions": True,
                },
            },
            "model_identity": self._forged_valid_model(runner),
            "implementation_identity": self._forged_valid_implementation(
                runner, capture=True
            ),
            "assets": {
                name: {"sha256": "a" * 64}
                for name in (
                    "agentview_image_npy",
                    "agentview_depth_npy",
                    "backview_image_npy",
                    "backview_depth_npy",
                )
            },
        }
        self.assertTrue(
            runner._implementation_identity_valid(
                forged["implementation_identity"], capture=True
            )
        )
        errors = runner.validate_aegis_label_capture(forged)
        self.assertTrue(errors)
        self.assertIn(
            "capture assets are not file/array/PNG-decode bound", errors
        )
        self.assertIn("capture policy/source binding is invalid", errors)


if __name__ == "__main__":
    unittest.main()
