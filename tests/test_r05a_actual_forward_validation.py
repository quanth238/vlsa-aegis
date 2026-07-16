from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
IMPORT_ERROR: Exception | None = None
try:
    import numpy as np

    sys.path.insert(0, str(ROOT / "main"))
    from crfs_oracle import r05a_actual_forward_validation as validation  # noqa: E402
except (ImportError, ModuleNotFoundError) as exc:  # dependency-free local gate.
    IMPORT_ERROR = exc


@unittest.skipIf(
    IMPORT_ERROR is not None,
    f"actual-forward tensor validation requires NumPy: {IMPORT_ERROR}",
)
class ActualForwardTensorValidationTest(unittest.TestCase):
    @staticmethod
    def _ledger(arrays: dict[str, np.ndarray]) -> list[dict]:
        rows = []
        for ordinal in range(534):
            if ordinal == 0:
                phase, index, query, generation, population = (
                    "compiled_frozen_pre",
                    0,
                    None,
                    None,
                    None,
                )
            elif ordinal == 1:
                phase, index, query, generation, population = (
                    "eager_source_trace_pre",
                    0,
                    None,
                    None,
                    None,
                )
            elif ordinal == 2:
                phase, index, query, generation, population = (
                    "eager_normalized_final_pre",
                    0,
                    None,
                    None,
                    None,
                )
            elif ordinal == 3:
                phase, index, query, generation, population = (
                    "zero_schedule_pre",
                    0,
                    None,
                    None,
                    None,
                )
            elif ordinal < 6:
                phase, index, query, generation, population = (
                    "arm_a_equal_split",
                    ordinal - 4,
                    None,
                    None,
                    None,
                )
            elif ordinal < 526:
                query = ordinal - 6
                phase, index, generation, population = (
                    "cem_search",
                    query,
                    query // 65,
                    query % 65,
                )
            elif ordinal < 528:
                phase, index, query, generation, population = (
                    "arm_b_replay",
                    ordinal - 526,
                    None,
                    None,
                    None,
                )
            elif ordinal < 530:
                phase, index, query, generation, population = (
                    "arm_c_reverse_replay",
                    ordinal - 528,
                    None,
                    None,
                    None,
                )
            elif ordinal == 530:
                phase, index, query, generation, population = (
                    "zero_schedule_post",
                    0,
                    None,
                    None,
                    None,
                )
            elif ordinal == 531:
                phase, index, query, generation, population = (
                    "eager_normalized_final_post",
                    0,
                    None,
                    None,
                    None,
                )
            elif ordinal == 532:
                phase, index, query, generation, population = (
                    "eager_source_trace_post",
                    0,
                    None,
                    None,
                    None,
                )
            else:
                phase, index, query, generation, population = (
                    "compiled_frozen_post",
                    0,
                    None,
                    None,
                    None,
                )
            row = {
                    "ordinal": ordinal,
                    "phase": phase,
                    "phase_index": index,
                    "cem_query_index": query,
                    "generation": generation,
                    "population_index": population,
                    "request_sha256": f"{ordinal:064x}"[-64:],
                }
            row["actions"] = validation._bytes_digest(
                validation._ledger_action(arrays, ordinal)
            )
            velocity = validation._ledger_velocity(arrays, ordinal)
            row["trace_slot"] = ordinal - 3 if velocity is not None else None
            if velocity is not None:
                row["requested_schedule"] = validation._bytes_digest(
                    validation._full_schedule(velocity)
                )
            rows.append(row)
        return rows

    @staticmethod
    def _project(value: np.ndarray, radius: float) -> np.ndarray:
        result = np.empty_like(value, dtype=np.float64)
        for index, block in enumerate(np.asarray(value, dtype=np.float64)):
            norm = np.sqrt(np.sum(block * block, dtype=np.float64), dtype=np.float64)
            if norm == 0.0:
                result[index] = 0.0
            elif norm <= radius:
                result[index] = block
            else:
                result[index] = block * (radius / norm)
        return result

    @staticmethod
    def _objective(actions: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        error = np.asarray(actions[:5, :7] - target[:5, :7], dtype=np.float64)
        weighted = np.concatenate(
            ((error[:, :3] / 0.005).reshape(-1), (error[:, 3:7] / 0.015).reshape(-1))
        )
        metrics = np.asarray(
            (
                np.max(np.abs(error[:, :3])),
                np.sqrt(np.mean(error[:, :3] ** 2, dtype=np.float64), dtype=np.float64),
                np.max(np.abs(error)),
                np.sqrt(np.mean(error**2, dtype=np.float64), dtype=np.float64),
            ),
            dtype=np.float64,
        )
        return (
            float(np.mean(weighted**2, dtype=np.float64)),
            metrics,
            metrics <= np.asarray((0.010, 0.005, 0.050, 0.015)),
        )

    @staticmethod
    def _fill_recurrence(arrays: dict[str, np.ndarray], prefix: str) -> None:
        applied = arrays[f"{prefix}_applied_velocity_f32"]
        count = applied.shape[0]
        steps = np.broadcast_to(np.arange(10, dtype=np.int64), (count, 10)).copy()
        times_one = np.empty((10,), dtype=np.float32)
        current = np.float32(1.0)
        for index in range(10):
            times_one[index] = current
            current = np.asarray(current + np.float32(-0.1), dtype=np.float32)
        active = np.broadcast_to(
            np.asarray([False] * 5 + [True] * 5, dtype=np.bool_), (count, 10)
        ).copy()
        control = np.zeros((count, 10, 10, 32), dtype=np.float32)
        control[:, 5:10, :5, :3] = applied.reshape(count, 5, 5, 3)
        v_base = arrays[f"{prefix}_trace_v_base_f32"]
        total = np.where(control == np.float32(0.0), v_base, v_base + control)
        x_t = arrays[f"{prefix}_trace_x_t_f32"]
        x_next = arrays[f"{prefix}_trace_x_next_f32"]
        state = np.broadcast_to(
            arrays["source_noise_f32"], (count, 10, 32)
        ).copy()
        initial = state.copy()
        for step in range(10):
            x_t[:, step] = state
            state = np.asarray(
                state + np.float32(-0.1) * total[:, step], dtype=np.float32
            )
            x_next[:, step] = state
        arrays[f"{prefix}_trace_step_index_i64"] = steps
        arrays[f"{prefix}_trace_time_f32"] = np.broadcast_to(
            times_one, (count, 10)
        ).copy()
        arrays[f"{prefix}_trace_active_bool"] = active
        arrays[f"{prefix}_trace_total_velocity_f32"] = total
        arrays[f"{prefix}_trace_initial_noise_f32"] = initial
        arrays[f"{prefix}_final_normalized_f32"] = state.copy()

    def _fixture(self) -> tuple[dict[str, np.ndarray], list[dict]]:
        arrays = {
            name: np.zeros(shape, dtype=dtype)
            for name, (shape, dtype) in validation.ARRAY_SPECS.items()
        }
        budget = np.float32(3.6398398876190186)
        dt = np.float32(-0.1)
        radius = np.asarray(budget / np.float32(5.0), dtype=np.float32)
        arrays["source_budget_f32"] = np.asarray(budget, dtype=np.float32)
        arrays["source_dt_f32"] = np.asarray(dt, dtype=np.float32)
        arrays["source_radius_f32"] = np.asarray(radius, dtype=np.float32)
        arrays["source_model_to_physical_scale_f32"].fill(np.float32(1.0))

        delta = arrays["source_delta_model_f32"]
        delta[:5, :3] = np.float32(0.1)
        arrays["source_target_normalized_f32"] = np.asarray(delta, dtype=np.float32).copy()
        target_physical = arrays["source_target_physical_f64"]
        target_physical[:5, :3] = np.asarray(delta[:5, :3], dtype=np.float64)
        arrays["reference_eager_final_f32"][:] = arrays["source_frozen_final_f32"]

        for name in (
            "reference_compiled_actions_f64",
            "reference_source_actions_f64",
            "reference_eager_actions_f64",
        ):
            arrays[name].fill(0.0)
        arrays["zero_final_normalized_f32"][:] = arrays["source_frozen_final_f32"]
        arrays["zero_returned_actions_f64"][:] = arrays["reference_eager_actions_f64"][0]

        arm_a_row = np.asarray(delta[:5, :3].reshape(15) / np.float32(5.0), dtype=np.float32)
        arm_a_c = np.broadcast_to(arm_a_row, (2, 5, 15)).copy()
        arm_a_u = np.asarray(arm_a_c / dt, dtype=np.float32)
        arm_a_executed = np.asarray(dt * arm_a_u, dtype=np.float32)
        arrays["arm_a_requested_c_f32"] = arm_a_c
        arrays["arm_a_requested_velocity_f32"] = arm_a_u
        arrays["arm_a_applied_velocity_f32"] = arm_a_u.copy()
        arrays["arm_a_executed_c_f32"] = arm_a_executed
        arrays["arm_a_final_normalized_f32"][:] = arrays["source_target_normalized_f32"]
        arrays["arm_a_returned_actions_f64"][:] = target_physical

        sigma0 = np.float64(radius) / np.sqrt(np.float64(15.0))
        mean = np.asarray(arm_a_executed[0], dtype=np.float64)
        sigma = np.full((5, 15), sigma0, dtype=np.float64)
        arrays["cem_mean_state_f64"][0] = mean
        arrays["cem_sigma_state_f64"][0] = sigma
        arrays["cem_raw_normals_f64"] = np.random.Generator(
            np.random.PCG64(20260716)
        ).standard_normal((8, 32, 5, 15), dtype=np.float64)
        cem_actions = target_physical.copy()
        cem_actions[:5, :3] += 0.1
        for generation in range(8):
            start = generation * 65
            proposals_list = [mean.copy()]
            for pair in range(32):
                offset = sigma * arrays["cem_raw_normals_f64"][generation, pair]
                proposals_list.extend((mean + offset, mean - offset))
            proposals = np.stack(proposals_list, axis=0)
            projected = np.stack(
                [self._project(item, float(radius)).astype(np.float32) for item in proposals]
            )
            requested_u = np.asarray(projected / dt, dtype=np.float32)
            executed = np.asarray(dt * requested_u, dtype=np.float32)
            arrays["cem_raw_proposal_f64"][start : start + 65] = proposals
            arrays["cem_projected_c_f32"][start : start + 65] = projected
            arrays["cem_requested_velocity_f32"][start : start + 65] = requested_u
            arrays["cem_applied_velocity_f32"][start : start + 65] = requested_u
            arrays["cem_executed_c_f32"][start : start + 65] = executed
            arrays["cem_final_normalized_f32"][start : start + 65] = arrays[
                "source_target_normalized_f32"
            ]
            arrays["cem_returned_actions_f64"][start : start + 65] = cem_actions
            objective, metrics, gates = self._objective(cem_actions, target_physical)
            arrays["cem_objective_f64"][start : start + 65] = objective
            arrays["cem_fidelity_metrics_f64"][start : start + 65] = metrics
            arrays["cem_gate_pass_bool"][start : start + 65] = gates
            arrays["cem_energy_f64"][start : start + 65] = np.sum(
                executed.astype(np.float64) ** 2, axis=(1, 2), dtype=np.float64
            )
            elites = np.arange(start, start + 13, dtype=np.int64)
            arrays["cem_elite_query_index_i64"][generation] = elites
            elite_values = executed[:13].astype(np.float64)
            mean = self._project(np.mean(elite_values, axis=0, dtype=np.float64), float(radius))
            variance = np.mean((elite_values - mean) ** 2, axis=0, dtype=np.float64)
            sigma = np.clip(np.sqrt(variance), sigma0 / 16.0, sigma0)
            arrays["cem_variance_after_f64"][generation] = variance
            arrays["cem_mean_state_f64"][generation + 1] = mean
            arrays["cem_sigma_state_f64"][generation + 1] = sigma

        arrays["cem_generation_i16"] = np.repeat(np.arange(8, dtype=np.int16), 65)
        arrays["cem_population_index_i16"] = np.tile(np.arange(65, dtype=np.int16), 8)
        arrays["cem_pair_index_i16"].fill(-1)
        for generation in range(8):
            base = generation * 65
            for pair in range(32):
                arrays["cem_pair_index_i16"][base + 1 + 2 * pair : base + 3 + 2 * pair] = pair
                arrays["cem_sign_i8"][base + 1 + 2 * pair] = 1
                arrays["cem_sign_i8"][base + 2 + 2 * pair] = -1
        arrays["cem_query_elapsed_ns_u64"].fill(1)
        arrays["selected_pool_index_i64"] = np.asarray(0, dtype=np.int64)

        for prefix in ("arm_b", "arm_c"):
            arrays[f"{prefix}_requested_c_f32"] = arm_a_c.copy()
            arrays[f"{prefix}_requested_velocity_f32"] = arm_a_u.copy()
            arrays[f"{prefix}_applied_velocity_f32"] = arm_a_u.copy()
            arrays[f"{prefix}_executed_c_f32"] = arm_a_executed.copy()
            arrays[f"{prefix}_final_normalized_f32"][:] = arrays[
                "source_target_normalized_f32"
            ]
            arrays[f"{prefix}_returned_actions_f64"][:] = target_physical
        for prefix in validation.SCHEDULE_GROUP_SIZES:
            self._fill_recurrence(arrays, prefix)
            arrays[f"{prefix}_final_normalized_physical_f64"] = arrays[
                f"{prefix}_returned_actions_f64"
            ].copy()
        return arrays, self._ledger(arrays)

    def test_complete_archive_recomputes_baseline_sufficient_outcome(self) -> None:
        arrays, ledger = self._fixture()
        result = validation.validate_actual_forward_tensors(arrays, ledger)
        self.assertEqual(result["request_count"], 534)
        self.assertEqual(result["selected_pool_index"], 0)
        self.assertTrue(result["arm_a"]["passed"])
        self.assertFalse(result["arm_b"]["changed_from_arm_a"])
        self.assertEqual(
            result["outcome"], "baseline_sufficient_no_incremental_support"
        )

    def test_cem_proposal_objective_elite_and_selection_tampering_fail(self) -> None:
        for name, mutate in (
            (
                "proposal",
                lambda arrays: arrays["cem_raw_proposal_f64"].__setitem__(
                    (0, 0, 0), arrays["cem_raw_proposal_f64"][0, 0, 0] + 1.0
                ),
            ),
            (
                "objective",
                lambda arrays: arrays["cem_objective_f64"].__setitem__(0, 1.0),
            ),
            (
                "elite",
                lambda arrays: arrays["cem_elite_query_index_i64"].__setitem__(
                    (0, 0), 64
                ),
            ),
            (
                "selection",
                lambda arrays: arrays.__setitem__(
                    "selected_pool_index_i64", np.asarray(1, dtype=np.int64)
                ),
            ),
        ):
            with self.subTest(name=name):
                arrays, ledger = self._fixture()
                mutate(arrays)
                with self.assertRaises(validation.ActualForwardValidationError):
                    validation.validate_actual_forward_tensors(arrays, ledger)

    def test_transport_and_replay_tampering_fail(self) -> None:
        arrays, ledger = self._fixture()
        arrays["cem_applied_velocity_f32"][11, 0, 0] += np.float32(0.25)
        with self.assertRaisesRegex(
            validation.ActualForwardValidationError, "applied velocity"
        ):
            validation.validate_actual_forward_tensors(arrays, ledger)

        arrays, ledger = self._fixture()
        arrays["cem_trace_x_next_f32"][11, 0, 0, 0] += np.float32(0.001)
        with self.assertRaisesRegex(
            validation.ActualForwardValidationError, "Euler recurrence"
        ):
            validation.validate_actual_forward_tensors(arrays, ledger)

        arrays, ledger = self._fixture()
        arrays["arm_b_returned_actions_f64"][1, 0, 0] += 0.001
        arrays["arm_b_final_normalized_physical_f64"][1] = arrays[
            "arm_b_returned_actions_f64"
        ][1]
        ledger[527]["actions"] = validation._bytes_digest(
            arrays["arm_b_returned_actions_f64"][1]
        )
        with self.assertRaisesRegex(
            validation.ActualForwardValidationError, "Arm B replay"
        ):
            validation.validate_actual_forward_tensors(arrays, ledger)

    def test_ledger_order_shape_dtype_and_nonfinite_fail_closed(self) -> None:
        arrays, ledger = self._fixture()
        ledger[526]["phase"] = "arm_c_reverse_replay"
        with self.assertRaisesRegex(
            validation.ActualForwardValidationError, "ledger row 526.phase"
        ):
            validation.validate_actual_forward_tensors(arrays, ledger)

        arrays, ledger = self._fixture()
        ledger[6]["actions"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            validation.ActualForwardValidationError, "action digest"
        ):
            validation.validate_actual_forward_tensors(arrays, ledger)

        arrays, ledger = self._fixture()
        ledger[6]["requested_schedule"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            validation.ActualForwardValidationError, "schedule digest"
        ):
            validation.validate_actual_forward_tensors(arrays, ledger)

        arrays, ledger = self._fixture()
        ledger[6]["trace_slot"] = 0
        with self.assertRaisesRegex(
            validation.ActualForwardValidationError, "recurrence slot"
        ):
            validation.validate_actual_forward_tensors(arrays, ledger)

        arrays, ledger = self._fixture()
        arrays["cem_final_normalized_physical_f64"][0, 0, 0] += 1.0
        with self.assertRaisesRegex(
            validation.ActualForwardValidationError, "physical terminal trace"
        ):
            validation.validate_actual_forward_tensors(arrays, ledger)

        arrays, ledger = self._fixture()
        arrays["cem_energy_f64"] = arrays["cem_energy_f64"].astype(np.float32)
        with self.assertRaisesRegex(
            validation.ActualForwardValidationError, "must preserve"
        ):
            validation.validate_actual_forward_tensors(arrays, ledger)

        arrays, ledger = self._fixture()
        arrays["cem_raw_normals_f64"][0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(
            validation.ActualForwardValidationError, "nonfinite"
        ):
            validation.validate_actual_forward_tensors(arrays, ledger)

    def test_npz_loader_disables_pickle_and_requires_exact_members(self) -> None:
        arrays, ledger = self._fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tensors.npz"
            np.savez(path, **arrays)
            result = validation.validate_actual_forward_npz(path, ledger)
            self.assertEqual(
                result["outcome"], "baseline_sufficient_no_incremental_support"
            )

            bad = dict(arrays)
            bad["cem_raw_normals_f64"] = np.asarray([object()], dtype=object)
            np.savez(path, **bad)
            with self.assertRaises(validation.ActualForwardValidationError):
                validation.validate_actual_forward_npz(path, ledger)


if __name__ == "__main__":
    unittest.main()
