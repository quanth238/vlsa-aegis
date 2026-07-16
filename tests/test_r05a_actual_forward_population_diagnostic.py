from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
IMPORT_ERROR: Exception | None = None
try:
    import numpy as np

    sys.path.insert(0, str(ROOT / "main"))
    from crfs_oracle import r05a_actual_forward_population_diagnostic as diagnostic  # noqa: E402
except (ImportError, ModuleNotFoundError) as exc:
    IMPORT_ERROR = exc


@unittest.skipIf(
    IMPORT_ERROR is not None,
    f"AF-00A population diagnostic requires NumPy: {IMPORT_ERROR}",
)
class ActualForwardPopulationDiagnosticTest(unittest.TestCase):
    def test_group_projection_enforces_both_caps(self) -> None:
        value = np.arange(1, 76, dtype=np.float64)
        projected = diagnostic._project_group_ball(value, radius=0.4, budget=1.1)
        norms = np.linalg.norm(projected.reshape(5, 15), axis=1)
        self.assertTrue(np.all(norms <= 0.4 + 1.0e-14))
        self.assertLessEqual(float(np.sum(norms)), 1.1 + 1.0e-12)
        self.assertGreater(float(np.sum(norms)), 1.1 - 1.0e-10)

    @staticmethod
    def _surrogate_fixture() -> dict[str, np.ndarray]:
        rng = np.random.Generator(np.random.PCG64(1701))
        raw = rng.normal(size=(520, 5, 15)).astype(np.float64)
        controls = np.empty_like(raw, dtype=np.float32)
        for index, candidate in enumerate(raw):
            controls[index] = diagnostic._project_group_ball(
                candidate, radius=0.2, budget=1.0
            ).reshape(5, 15).astype(np.float32)
        matrix = rng.normal(scale=0.015, size=(75, 35))
        response_scaled = controls.astype(np.float64).reshape(520, 75) @ matrix
        response_physical = response_scaled * diagnostic.GATE_SCALES
        zero = np.zeros((2, 10, 7), dtype=np.float64)
        actions = np.zeros((520, 10, 7), dtype=np.float64)
        actions[:, :5, :7] = response_physical.reshape(520, 5, 7)
        target = np.zeros((10, 7), dtype=np.float64)
        target[:5, :3] = 0.01
        return {
            "cem_executed_c_f32": controls,
            "zero_returned_actions_f64": zero,
            "cem_returned_actions_f64": actions,
            "source_target_physical_f64": target,
            "source_radius_f32": np.asarray(0.2, dtype=np.float32),
            "source_budget_f32": np.asarray(1.0, dtype=np.float32),
        }

    def test_surrogate_is_zero_anchored_group_held_out_and_explicitly_noncausal(self) -> None:
        report = diagnostic.surrogate_diagnostic(self._surrogate_fixture())
        self.assertEqual(
            report["evidence_class"],
            "retrospective_exploratory_surrogate_not_sampler_evidence",
        )
        self.assertFalse(report["model"]["regularization_or_hyperparameter_tuning"])
        self.assertEqual(report["model"]["rank"], 75)
        self.assertEqual(len(report["leave_one_complete_generation_out"]["folds"]), 8)
        self.assertTrue(
            report["constrained_inverse_prediction"][
                "prediction_only_not_actual_forward_evaluation"
            ]
        )
        self.assertGreater(report["in_sample"]["zero_anchored_energy_r2"], 0.999999)

    def test_hash_mismatch_rejects_before_scientific_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename in diagnostic.EXPECTED_FILES:
                (root / filename).write_bytes(b"not the sealed artifact")
            with self.assertRaisesRegex(
                diagnostic.ActualForwardPopulationDiagnosticError,
                "sealed file hash changed",
            ):
                diagnostic.diagnose_bound_run(root)

    @unittest.skipUnless(
        (ROOT / ".harness" / "af00a-run-c" / "af00a-tensors.npz").is_file(),
        "machine-local sealed AF-00A run-C copy is unavailable",
    )
    def test_sealed_run_report_is_hash_bound_semantically_valid_and_deterministic(self) -> None:
        report = diagnostic.diagnose_bound_run(ROOT / ".harness" / "af00a-run-c")
        encoded_first = diagnostic.deterministic_json(report)
        encoded_second = diagnostic.deterministic_json(report)
        self.assertEqual(encoded_first, encoded_second)
        parsed = json.loads(encoded_first)
        self.assertTrue(parsed["integrity_validation"]["existing_af_semantic_validator_passed"])
        self.assertEqual(parsed["integrity_validation"]["request_count"], 534)
        self.assertEqual(parsed["integrity_validation"]["selected_cem_query_index"], 464)
        exact = parsed["exact_descriptive_diagnostics"]["transport_decomposition"]
        self.assertEqual(
            exact["population"]["feedback_target_alpha_strictly_negative_count"],
            520,
        )
        self.assertEqual(
            exact["population"]["terminal_alpha_meets_necessary_xyz_rms_bound_count"],
            0,
        )
        self.assertAlmostEqual(exact["arm_a"]["feedback_target_alpha"], -0.4001357184428276)
        search = parsed["exact_descriptive_diagnostics"]["frozen_search"]
        self.assertEqual(search["first_generation_better_than_arm_a"], 7)
        self.assertEqual(search["count_queries_better_than_arm_a"], 3)
        surrogate = parsed["surrogate_diagnostics"]
        self.assertAlmostEqual(
            surrogate["leave_one_complete_generation_out"]["pooled_centered_r2"],
            0.9545156041596754,
        )
        self.assertFalse(parsed["claim_boundaries"]["global_infeasibility_proven"])
        self.assertFalse(
            parsed["claim_boundaries"]["probe_or_residual_mlp_training_authorized"]
        )


if __name__ == "__main__":
    unittest.main()
