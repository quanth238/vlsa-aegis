from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np
except ModuleNotFoundError:  # Keep the repository bootstrap dependency-free.
    np = None


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "main", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

if np is not None:
    from crfs_harness.manifest import build_official_cases
    from crfs_harness.official_state import file_sha256, init_state_row_identity
    from crfs_oracle.runner import OracleConfig, SafeLiberoCase


class GeneratedSourceRunnerStructuralTest(unittest.TestCase):
    def test_generated_loading_is_opt_in_and_released_loader_remains_present(self) -> None:
        source = (ROOT / "main/crfs_oracle/runner.py").read_text(encoding="utf-8")
        self.assertIn('self._generated_source = "source_estimand" in case', source)
        self.assertIn("suite.get_task_init_states(self._task_index)", source)
        self.assertIn("restore_generated_source_branch(self.env, self._generated_bundle)", source)
        self.assertIn("verify_generated_source_branch(", source)
        self.assertIn("Generated source states must never be labeled SafeLIBERO Level II", source)

    def test_official_saved_state_loading_is_opt_in_and_fail_closed(self) -> None:
        source = (ROOT / "main/crfs_oracle/runner.py").read_text(encoding="utf-8")
        self.assertIn('self._official_state_bound = case.get("schema_version") == "3.0"', source)
        self.assertIn("verify_official_state_bindings(", source)
        self.assertIn("Invalid official saved-state manifest record", source)
        self.assertIn("Official saved-state identity verification failed", source)
        self.assertIn("cannot mix legacy released and official-state-bound records", source)


def _config(*, settle_steps: int = 20) -> "OracleConfig":
    return OracleConfig(
        host="127.0.0.1",
        port=8000,
        resize_size=224,
        settle_steps=settle_steps,
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
        response_matrix_m_per_action=None,
        optimizer_max_iterations=1,
        checkpoint_id="checkpoint",
        checkpoint_sha256="a" * 64,
        output_root="/tmp/output",
        run_id="run",
    )


def _released_case() -> dict:
    return {
        "task_suite": "safelibero_spatial",
        "safety_level": "II",
        "task_index": 0,
        "episode_index": 1,
        "environment_seed": 11,
    }


def _generated_case(
    *,
    state_sha256: str = "1" * 64,
    branch_sha256: str | None = None,
    safety_level: str = "generated",
) -> dict:
    branch_sha256 = branch_sha256 or state_sha256
    state_id = f"gsrc-{branch_sha256[:16]}"
    value = {
        "task_suite": "safelibero_spatial",
        "safety_level": safety_level,
        "task_index": 0,
        "environment_seed": 17,
        "source_estimand": "task0_single_obstacle_generated_v1",
        "source_state_id": state_id,
        "source_branch_sha256": branch_sha256,
        "source_state_sha256": state_sha256,
        "source_bundle_path": f"/source/{state_id}/source.json",
        "source_bundle_sha256": "b" * 64,
    }
    return value


class _FakeSuite:
    def __init__(self) -> None:
        self.init_state_calls: list[int] = []
        self.states = np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64
        )

    def get_task(self, task_index: int):
        if task_index != 0:
            raise AssertionError("unexpected task")
        return SimpleNamespace(
            name="task0",
            problem_folder="safelibero_spatial",
            bddl_file="task0.bddl",
            init_states_file="task0.pruned_init",
            language="pick up the bowl",
        )

    def get_task_init_states(self, task_index: int):
        self.init_state_calls.append(task_index)
        return self.states


class _FakeSuiteFactory:
    def __init__(self, suite: _FakeSuite) -> None:
        self.suite = suite
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self.suite


class _FakeRenderEnv:
    instances: list["_FakeRenderEnv"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)
        self.seed_calls: list[int] = []
        self.reset_calls = 0
        self.init_states: list[np.ndarray] = []
        self.step_calls: list[list[float]] = []
        self.update_calls: list[bool] = []
        self.closed = False
        self.env = SimpleNamespace(
            objects_dict={"moka_pot_obstacle_1": object()},
            _get_observations=lambda: _observation(),
        )
        self.sim = object()
        self.__class__.instances.append(self)

    def seed(self, seed: int) -> None:
        self.seed_calls.append(seed)

    def reset(self) -> None:
        self.reset_calls += 1

    def set_init_state(self, state) -> None:
        self.init_states.append(np.asarray(state).copy())

    def step_with_substep_callback(self, action, callback, **kwargs) -> None:
        self.step_calls.append(list(action))

    def _update_observables(self, force: bool = False) -> None:
        self.update_calls.append(force)

    def close(self) -> None:
        self.closed = True


def _observation() -> dict:
    return {
        "moka_pot_obstacle_1_pos": np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        "robot0_eef_pos": np.asarray((0.0, 0.0, 1.1), dtype=np.float64),
    }


def _fake_libero_modules(
    factory: _FakeSuiteFactory,
    *,
    init_states_root: str = "/tmp/libero",
) -> dict[str, types.ModuleType]:
    libero = types.ModuleType("libero")
    libero_package = types.ModuleType("libero.libero")
    benchmark = SimpleNamespace(
        get_benchmark_dict=lambda: {"safelibero_spatial": factory}
    )
    libero_package.benchmark = benchmark
    libero_package.get_libero_path = lambda _name: init_states_root
    envs = types.ModuleType("libero.libero.envs")
    envs.OffScreenRenderEnv = _FakeRenderEnv
    return {
        "libero": libero,
        "libero.libero": libero_package,
        "libero.libero.envs": envs,
    }


def _fake_generated_source_module(*, verifier_errors=None):
    module = types.ModuleType("crfs_oracle.generated_source")

    def load(case, **_kwargs):
        expected_id = f"gsrc-{case['source_branch_sha256'][:16]}"
        if case["source_state_id"] != expected_id:
            raise ValueError("manifest did not bind the canonical final-state identity")
        return {"source_state": {"source_state_id": expected_id}}

    module.load_generated_source_bundle = mock.Mock(side_effect=load)
    module.restore_generated_source_branch = mock.Mock(
        side_effect=lambda _env, _bundle: _observation()
    )
    module.verify_generated_source_branch = mock.Mock(
        return_value=[] if verifier_errors is None else list(verifier_errors)
    )
    return module


@unittest.skipUnless(np is not None, "Generated-source runner tests require NumPy")
class GeneratedSourceRunnerRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeRenderEnv.instances.clear()
        self.suite = _FakeSuite()
        self.factory = _FakeSuiteFactory(self.suite)
        self.libero_modules = _fake_libero_modules(self.factory)

    def _patch_modules(self, generated_module):
        return mock.patch.dict(
            sys.modules,
            {
                **self.libero_modules,
                "crfs_oracle.generated_source": generated_module,
            },
        )

    def test_released_state_path_retains_existing_reset_and_settle_behavior(self) -> None:
        generated = _fake_generated_source_module()
        generated.load_generated_source_bundle.side_effect = AssertionError(
            "released path imported generated loader"
        )
        with self._patch_modules(generated), mock.patch(
            "crfs_oracle.runner.resolve_crfs_geom_groups",
            return_value=(("eef",), ("obstacle",)),
        ):
            environment = SafeLiberoCase(_released_case(), _config())
            observation = environment.reset_and_settle()

        self.assertEqual(self.factory.calls, [{"safety_level": "II"}])
        self.assertEqual(self.suite.init_state_calls, [0])
        np.testing.assert_array_equal(
            environment.env.init_states[0], self.suite.states[1]
        )
        self.assertEqual(environment.env.reset_calls, 1)
        self.assertEqual(len(environment.env.step_calls), 20)
        self.assertTrue(
            all(action == [0.0] * 6 + [-1.0] for action in environment.env.step_calls)
        )
        self.assertEqual(environment.env.update_calls, [True])
        self.assertEqual(observation["moka_pot_obstacle_1_pos"].dtype, np.dtype("float64"))
        self.assertEqual(environment.obstacle_name, "moka_pot_obstacle_1")

    def test_generated_path_loads_bundle_replays_and_verifies_before_return(self) -> None:
        generated = _fake_generated_source_module()
        case = _generated_case()
        with self._patch_modules(generated), mock.patch(
            "crfs_oracle.runner.resolve_crfs_geom_groups",
            return_value=(("eef",), ("obstacle",)),
        ):
            environment = SafeLiberoCase(case, _config())
            observation = environment.reset_and_settle()

        self.assertEqual(self.factory.calls, [{}])
        self.assertEqual(self.suite.init_state_calls, [])
        generated.load_generated_source_bundle.assert_called_once_with(case)
        generated.restore_generated_source_branch.assert_called_once_with(
            environment.env,
            {"source_state": {"source_state_id": "gsrc-1111111111111111"}},
        )
        generated.verify_generated_source_branch.assert_called_once_with(
            {"source_state": {"source_state_id": "gsrc-1111111111111111"}},
            env=environment.env,
            observation=observation,
            active_obstacle_name="moka_pot_obstacle_1",
        )
        self.assertEqual(environment.env.reset_calls, 0)
        self.assertEqual(environment.env.init_states, [])
        self.assertEqual(environment.obstacle_name, "moka_pot_obstacle_1")
        self.assertEqual(environment.eef_geoms, ("eef",))
        self.assertEqual(environment.obstacle_geoms, ("obstacle",))

    def test_generated_configure_case_reloads_only_same_task_and_estimand(self) -> None:
        generated = _fake_generated_source_module()
        first = _generated_case(state_sha256="1" * 64)
        second = _generated_case(state_sha256="2" * 64)
        with self._patch_modules(generated):
            environment = SafeLiberoCase(first, _config())
            environment.configure_case(second)

        self.assertEqual(generated.load_generated_source_bundle.call_count, 2)
        self.assertEqual(
            environment._generated_bundle,
            {"source_state": {"source_state_id": "gsrc-2222222222222222"}},
        )
        self.assertEqual(environment.env.seed_calls, [17, 17])

    def test_generated_state_cannot_claim_level_ii(self) -> None:
        generated = _fake_generated_source_module()
        with self._patch_modules(generated):
            with self.assertRaisesRegex(ValueError, "must never be labeled"):
                SafeLiberoCase(
                    _generated_case(safety_level="II"),
                    _config(),
                )
        generated.load_generated_source_bundle.assert_not_called()

    def test_generated_path_uses_canonical_final_state_id_not_request_id(self) -> None:
        generated = _fake_generated_source_module()
        case = _generated_case()
        case["source_state_id"] = "task0-single-obstacle-generated-v1-0000"
        with self._patch_modules(generated):
            with self.assertRaisesRegex(ValueError, "canonical final-state identity"):
                SafeLiberoCase(case, _config())

    def test_generated_restore_rejects_nonregistered_settle_length(self) -> None:
        generated = _fake_generated_source_module()
        with self._patch_modules(generated):
            environment = SafeLiberoCase(_generated_case(), _config(settle_steps=19))
            with self.assertRaisesRegex(ValueError, "20-step settle history"):
                environment.reset_and_settle()
        generated.restore_generated_source_branch.assert_not_called()

    def test_generated_branch_hash_failure_stops_before_inference_surface(self) -> None:
        generated = _fake_generated_source_module(verifier_errors=["observation hash mismatch"])
        with self._patch_modules(generated), mock.patch(
            "crfs_oracle.runner.resolve_crfs_geom_groups",
            return_value=(("eef",), ("obstacle",)),
        ):
            environment = SafeLiberoCase(_generated_case(), _config())
            with self.assertRaisesRegex(RuntimeError, "observation hash mismatch"):
                environment.reset_and_settle()
        self.assertIsNone(environment.obstacle_name)

    def test_shared_environment_cannot_mix_released_and_generated_sources(self) -> None:
        generated = _fake_generated_source_module()
        with self._patch_modules(generated):
            environment = SafeLiberoCase(_released_case(), _config())
            with self.assertRaisesRegex(ValueError, "cannot mix"):
                environment.configure_case(_generated_case())

    def test_official_state_record_verifies_before_environment_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative_path = "safelibero_spatial/task0_level_II.pruned_init"
            state_path = root / relative_path
            state_path.parent.mkdir(parents=True)
            state_path.write_bytes(b"official-state-file")
            case = build_official_cases(
                task_suite="safelibero_spatial",
                safety_level="II",
                task_index=0,
                task_name="task0",
                init_state_relative_path=relative_path,
                init_state_file_sha256=file_sha256(state_path),
                row_identities={1: init_state_row_identity(self.suite.states[1])},
                seeds_per_episode=1,
                seed_namespace="official-runner-test",
            )[0]
            modules = _fake_libero_modules(self.factory, init_states_root=directory)
            with mock.patch.dict(sys.modules, modules):
                environment = SafeLiberoCase(case, _config())

            np.testing.assert_array_equal(environment._init_state, self.suite.states[1])
            self.assertEqual(len(_FakeRenderEnv.instances), 1)

            wrong_file = build_official_cases(
                task_suite="safelibero_spatial",
                safety_level="II",
                task_index=0,
                task_name="task0",
                init_state_relative_path=relative_path,
                init_state_file_sha256="0" * 64,
                row_identities={1: init_state_row_identity(self.suite.states[1])},
                seeds_per_episode=1,
                seed_namespace="official-runner-test",
            )[0]
            before = len(_FakeRenderEnv.instances)
            with mock.patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(ValueError, "init_state_file_sha256"):
                    SafeLiberoCase(wrong_file, _config())
            self.assertEqual(len(_FakeRenderEnv.instances), before)

            wrong_row = build_official_cases(
                task_suite="safelibero_spatial",
                safety_level="II",
                task_index=0,
                task_name="task0",
                init_state_relative_path=relative_path,
                init_state_file_sha256=file_sha256(state_path),
                row_identities={
                    1: init_state_row_identity(
                        np.asarray([9.0, 9.0, 9.0], dtype=np.float64)
                    )
                },
                seeds_per_episode=1,
                seed_namespace="official-runner-test",
            )[0]
            with mock.patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(ValueError, "init_state_row_bytes_sha256"):
                    SafeLiberoCase(wrong_row, _config())

    def test_shared_environment_cannot_downgrade_official_record_to_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative_path = "safelibero_spatial/task0_level_II.pruned_init"
            state_path = root / relative_path
            state_path.parent.mkdir(parents=True)
            state_path.write_bytes(b"official-state-file")
            official = build_official_cases(
                task_suite="safelibero_spatial",
                safety_level="II",
                task_index=0,
                task_name="task0",
                init_state_relative_path=relative_path,
                init_state_file_sha256=file_sha256(state_path),
                row_identities={1: init_state_row_identity(self.suite.states[1])},
                seeds_per_episode=1,
                seed_namespace="official-runner-test",
            )[0]
            modules = _fake_libero_modules(self.factory, init_states_root=directory)
            with mock.patch.dict(sys.modules, modules):
                environment = SafeLiberoCase(official, _config())
                with self.assertRaisesRegex(ValueError, "cannot mix legacy"):
                    environment.configure_case(_released_case())


if __name__ == "__main__":
    unittest.main()
