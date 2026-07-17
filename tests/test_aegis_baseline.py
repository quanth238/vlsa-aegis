from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "main/crfs_oracle/aegis_baseline.py"
MODULE_SPEC = importlib.util.spec_from_file_location("crfs_aegis_baseline", MODULE_PATH)
if (
    MODULE_SPEC is None or MODULE_SPEC.loader is None
):  # pragma: no cover - import guard.
    raise RuntimeError("failed to load AEGIS baseline adapter")
aegis = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = aegis
MODULE_SPEC.loader.exec_module(aegis)


class AegisBaselineStructuralTest(unittest.TestCase):
    def test_module_is_dependency_free_at_import_and_contains_full_qp(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_level_imports = {
            alias.name.split(".")[0]
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("numpy", top_level_imports)
        self.assertNotIn("cvxpy", top_level_imports)
        for fragment in (
            "u = cp.Variable(9)",
            "qp.solve(solver=cp.OSQP)",
            "UPSTREAM_ACTION_TO_VELOCITY * v_reference",
            "UPSTREAM_VELOCITY_TO_ACTION * r1_before @ u_v",
            "filtered[6] = action[6]",
            "nominal fallback is forbidden",
        ):
            self.assertIn(fragment, source)

    def test_frozen_upstream_hashes_match_all_aegis_sources(self) -> None:
        actual = aegis.verify_upstream_sources(ROOT)
        self.assertEqual(actual, dict(aegis.UPSTREAM_SOURCE_SHA256))
        for path, digest in actual.items():
            self.assertEqual(
                hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest
            )

    def test_source_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative_path in aegis.UPSTREAM_SOURCE_SHA256:
                destination = root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative_path, destination)
            (root / "main/main_aegis.py").write_text("drift\n", encoding="utf-8")
            with self.assertRaises(aegis.AegisBaselineError) as caught:
                aegis.verify_upstream_sources(root)
        self.assertEqual(caught.exception.code, "upstream_source_drift")


class AegisPerceptionContractTest(unittest.TestCase):
    def test_frozen_label_core_is_explicit_and_needs_no_external_secret(self) -> None:
        contract = aegis.resolve_perception_mode(
            aegis.PERCEPTION_FROZEN_LABEL_CORE,
            obstacle_label="white storage box",
            environ={},
        )
        self.assertEqual(contract.mode, "frozen_label_aegis_core")
        self.assertEqual(
            contract.semantic_label_source, "frozen_manifest_or_case_label"
        )
        self.assertFalse(contract.zhipu_key_present)
        self.assertIsNone(contract.dino_checkpoint_path)

    def test_frozen_label_core_rejects_missing_label(self) -> None:
        with self.assertRaises(aegis.AegisBaselineError) as caught:
            aegis.resolve_perception_mode(aegis.PERCEPTION_FROZEN_LABEL_CORE)
        self.assertEqual(caught.exception.code, "perception_failure")

    def test_original_mode_requires_key_then_both_dino_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(aegis.AegisBaselineError) as no_key:
                aegis.resolve_perception_mode(
                    aegis.PERCEPTION_ORIGINAL_END_TO_END,
                    environ={},
                )
            self.assertEqual(no_key.exception.details["missing"], ["ZHIPUAI_API_KEY"])

            with self.assertRaises(aegis.AegisBaselineError) as no_assets:
                aegis.resolve_perception_mode(
                    aegis.PERCEPTION_ORIGINAL_END_TO_END,
                    environ={"ZHIPUAI_API_KEY": "not-recorded"},
                )
            self.assertEqual(
                no_assets.exception.details["missing"],
                [
                    "AEGIS_GROUNDING_DINO_CONFIG",
                    "AEGIS_GROUNDING_DINO_CHECKPOINT",
                ],
            )

            config = root / "GroundingDINO_SwinT_OGC.py"
            checkpoint = root / "groundingdino_swint_ogc.pth"
            config.write_text("# fixture\n")
            checkpoint.write_bytes(b"fixture")
            contract = aegis.resolve_perception_mode(
                aegis.PERCEPTION_ORIGINAL_END_TO_END,
                environ={
                    "ZHIPUAI_API_KEY": "not-recorded",
                    "AEGIS_GROUNDING_DINO_CONFIG": str(config),
                    "AEGIS_GROUNDING_DINO_CHECKPOINT": str(checkpoint),
                },
            )
            self.assertTrue(contract.zhipu_key_present)
            self.assertNotIn("not-recorded", repr(contract))

    def test_original_mode_rejects_frozen_label_override(self) -> None:
        with self.assertRaises(aegis.AegisBaselineError):
            aegis.resolve_perception_mode(
                aegis.PERCEPTION_ORIGINAL_END_TO_END,
                obstacle_label="white storage box",
                environ={"ZHIPUAI_API_KEY": "secret"},
            )


class AegisMetricTest(unittest.TestCase):
    def test_action_modification_reports_channel_and_path_ratios(self) -> None:
        nominal = [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 1.0],
        ]
        filtered = [
            [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, -1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        ]
        metrics = aegis.action_modification_metrics(nominal, filtered)
        self.assertEqual(metrics.action_steps, 2)
        self.assertEqual(metrics.modified_steps, 2)
        self.assertAlmostEqual(metrics.translation_delta_l2, math.sqrt(1.25))
        self.assertAlmostEqual(metrics.rotation_delta_l2, math.sqrt(1.25))
        self.assertEqual(metrics.gripper_delta_l2, 0.0)
        self.assertEqual(metrics.translation_command_path_ratio, 0.5)

    def test_stopping_separates_ratio_diagnostic_from_registered_binary(self) -> None:
        ratio_only = aegis.stopping_diagnostic(
            baseline_eef_path_m=0.1,
            aegis_eef_path_m=0.01,
            baseline_progress_m=0.03,
            aegis_progress_m=-0.005,
        )
        self.assertAlmostEqual(ratio_only.eef_path_ratio, 0.1)
        self.assertTrue(ratio_only.ratio_low_motion_diagnostic)
        self.assertFalse(ratio_only.stop_like)

        registered_boundary = aegis.stopping_diagnostic(
            baseline_eef_path_m=0.1,
            aegis_eef_path_m=0.005,
            baseline_progress_m=0.03,
            aegis_progress_m=-0.001,
        )
        self.assertTrue(registered_boundary.stop_like)

        moving = aegis.stopping_diagnostic(
            baseline_eef_path_m=0.1,
            aegis_eef_path_m=0.0050001,
            baseline_progress_m=0.03,
            aegis_progress_m=0.0,
        )
        progressing = aegis.stopping_diagnostic(
            baseline_eef_path_m=0.1,
            aegis_eef_path_m=0.0,
            baseline_progress_m=0.03,
            aegis_progress_m=0.0010001,
        )
        undefined = aegis.stopping_diagnostic(
            baseline_eef_path_m=0.0,
            aegis_eef_path_m=0.0,
            baseline_progress_m=0.0,
            aegis_progress_m=0.0,
        )
        self.assertFalse(moving.stop_like)
        self.assertFalse(progressing.stop_like)
        self.assertIsNone(undefined.eef_path_ratio)
        self.assertTrue(undefined.stop_like)

    def test_upstream_obstacle_proxy_is_strict_and_not_contact(self) -> None:
        boundary = aegis.upstream_obstacle_displacement_diagnostic(
            [0.0, 0.0, 0.0], [0.001, 0.0, 0.0]
        )
        above = aegis.upstream_obstacle_displacement_diagnostic(
            [0.0, 0.0, 0.0], [0.0010001, 0.0, 0.0]
        )
        self.assertFalse(boundary.upstream_collision_proxy)
        self.assertTrue(above.upstream_collision_proxy)
        self.assertFalse(above.physical_contact_measurement)
        self.assertFalse(above.simulator_clearance_measurement)

    def test_exact_upstream_q1_task_string_branch(self) -> None:
        self.assertEqual(
            aegis.upstream_eef_q_diag("put the milk in the basket"),
            aegis.UPSTREAM_EEF_Q_DIAG_TALL,
        )
        self.assertEqual(
            aegis.upstream_eef_q_diag("put the bowl on the plate"),
            aegis.UPSTREAM_EEF_Q_DIAG_STANDARD,
        )


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(NUMPY_AVAILABLE, "numeric AEGIS adapter checks require NumPy")
class AegisSafetyCoreRuntimeTest(unittest.TestCase):
    @staticmethod
    def _perception() -> aegis.PerceptionContract:
        return aegis.resolve_perception_mode(
            aegis.PERCEPTION_FROZEN_LABEL_CORE,
            obstacle_label="white storage box",
        )

    @staticmethod
    def _coefficients(np, p1, _q1, r1, _p2, _q2, _r2, _z):
        AegisSafetyCoreRuntimeTest.last_p1 = p1.copy()
        AegisSafetyCoreRuntimeTest.last_r1 = r1.copy()
        return (
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            1.0,
            np.asarray([0.1, 0.2, 0.3]),
        )

    @staticmethod
    def _identity_qp(_np, problem):
        return aegis._QpSolution(
            values=problem.u_reference.copy(), status="optimal", objective_value=0.0
        )

    def _core(
        self, *, initialization_source=aegis.INITIALIZATION_PRE_SETTLE_UPSTREAM, qp=None
    ):
        return aegis.AegisSafetyCore(
            pre_settle_p1=[0.0, 0.0, 0.0],
            pre_settle_r1=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            branch_p1=[0.1, 0.0, 0.0],
            branch_r1=[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            q1_diag=[0.06, 0.12, 0.11],
            p2=[1.0, 1.0, 0.0],
            q2_diag=[0.1, 0.1, 0.1],
            r2=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            perception=self._perception(),
            initialization_source=initialization_source,
            repo_root=ROOT,
            _coefficient_backend=self._coefficients,
            _qp_backend=self._identity_qp if qp is None else qp,
        )

    def test_default_preserves_stale_pre_settle_pose_and_gripper(self) -> None:
        core = self._core()
        before_z = core.z
        result = core.filter_action([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.7])
        self.assertEqual(tuple(self.last_p1), (0.0, 0.0, 0.0))
        self.assertEqual(
            result.action,
            (0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, -0.7),
        )
        self.assertTrue(result.telemetry["gripper_preserved_exactly"])
        self.assertEqual(
            result.telemetry["initialization"]["initialization_source"],
            "pre_settle_upstream",
        )
        self.assertAlmostEqual(
            result.telemetry["initialization"][
                "pre_settle_to_branch_translation_delta_m"
            ],
            0.1,
        )
        self.assertNotEqual(core.z, before_z)

    def test_cbf_coefficients_match_byte_preserved_upstream_functions(self) -> None:
        import numpy as np

        source_tree = ast.parse((ROOT / "main/utils.py").read_text(encoding="utf-8"))
        required = {
            "vector_hat",
            "project_matrix",
            "compute_h_ij",
            "compute_h_coeffs_3d",
        }
        functions = [
            node
            for node in source_tree.body
            if isinstance(node, ast.FunctionDef) and node.name in required
        ]
        self.assertEqual({node.name for node in functions}, required)
        namespace = {"np": np}
        executable = ast.fix_missing_locations(
            ast.Module(body=functions, type_ignores=[])
        )
        exec(compile(executable, "main/utils.py", "exec"), namespace)

        inputs = (
            np.asarray([0.1, -0.2, 0.9]),
            np.asarray([0.06, 0.12, 0.11]),
            np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.asarray([0.4, 0.2, 1.1]),
            np.asarray([0.08, 0.10, 0.07]),
            np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
            np.asarray([0.3, 0.4, 0.2]),
        )
        upstream = namespace["compute_h_coeffs_3d"](*inputs)
        adapted = aegis._compute_h_coeffs_3d(np, *inputs)
        for upstream_value, adapted_value in zip(upstream, adapted):
            np.testing.assert_array_equal(upstream_value, adapted_value)

    def test_corrected_current_state_is_separately_named_and_explicit(self) -> None:
        core = self._core(
            initialization_source=aegis.INITIALIZATION_CURRENT_STATE_DIAGNOSTIC
        )
        core.filter_action([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(tuple(self.last_p1), (0.1, 0.0, 0.0))
        self.assertFalse(
            core.initialization_telemetry["literal_upstream_initialization"]
        )

    def test_qp_no_value_is_failure_without_z_mutation_or_fallback(self) -> None:
        def no_value(_np, _problem):
            return aegis._QpSolution(
                values=None, status="infeasible", objective_value=None
            )

        core = self._core(qp=no_value)
        before = core.z
        with self.assertRaises(aegis.AegisBaselineError) as caught:
            core.filter_action([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(caught.exception.code, "qp_failure")
        self.assertEqual(core.z, before)

    def test_refresh_after_action_matches_upstream_proxy_offset(self) -> None:
        core = self._core()
        telemetry = core.refresh_after_action(
            eef_position=[1.0, 2.0, 3.0],
            eef_rotation_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        self.assertEqual(telemetry["p1"], [1.0, 2.0, 2.92])


if __name__ == "__main__":
    unittest.main()
