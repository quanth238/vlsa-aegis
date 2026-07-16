from __future__ import annotations

import ast
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
IMPORT_ERROR: Exception | None = None
try:
    import numpy as np

    sys.path.insert(0, str(ROOT / "main"))
    from crfs_oracle import r05a_reference_trajectory_lift_validation as validation  # noqa: E402
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    IMPORT_ERROR = exc


@unittest.skipIf(
    IMPORT_ERROR is not None,
    f"TRL independent validation requires NumPy: {IMPORT_ERROR}",
)
class ReferenceTrajectoryLiftValidationTest(unittest.TestCase):
    def test_validator_has_no_policy_or_torch_dependency(self) -> None:
        path = ROOT / "main/crfs_oracle/r05a_reference_trajectory_lift_validation.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("torch", imported)
        self.assertNotIn("openpi", imported)
        self.assertNotIn("safelibero", imported)

    @staticmethod
    def _blank_arrays() -> dict[str, np.ndarray]:
        return {
            name: np.zeros(shape, dtype=dtype)
            for name, (shape, dtype) in validation.FINITE_ARRAY_SPECS.items()
        }

    @staticmethod
    def _field(state: np.ndarray, *, gain: float, coupling: float) -> np.ndarray:
        value = np.asarray(state, dtype=np.float32)
        result = np.asarray(np.float32(gain) * value, dtype=np.float32)
        if coupling:
            result = result.copy()
            result[0, 3] = np.asarray(
                result[0, 3] + np.float32(coupling) * value[0, 0],
                dtype=np.float32,
            )
        return result

    @classmethod
    def _simulate_fixed(
        cls,
        arrays: dict[str, np.ndarray],
        prefix: str,
        compact_c: np.ndarray,
        *,
        gain: float,
        coupling: float,
    ) -> None:
        requested_c = np.asarray(compact_c, dtype=np.float32)
        requested_u = np.asarray(requested_c / validation.EXPECTED_DT32, dtype=np.float32)
        if prefix == "zero":
            requested_c = np.zeros_like(requested_c)
            requested_u = np.zeros_like(requested_u)
        for replicate in range(2):
            arrays[f"{prefix}_requested_c_f32"][replicate] = requested_c
            arrays[f"{prefix}_requested_velocity_f32"][replicate] = requested_u
            arrays[f"{prefix}_applied_velocity_f32"][replicate] = requested_u
            arrays[f"{prefix}_executed_c_f32"][replicate] = (
                np.zeros_like(requested_u)
                if prefix == "zero"
                else np.asarray(
                    validation.EXPECTED_DT32 * requested_u, dtype=np.float32
                )
            )

            full = validation._full_schedule(requested_u)
            state = arrays["source_noise_f32"].copy()
            arrays[f"{prefix}_trace_initial_noise_f32"][replicate] = state
            for step in range(10):
                velocity = cls._field(state, gain=gain, coupling=coupling)
                control = full[step]
                total = np.where(
                    control == np.float32(0.0), velocity, velocity + control
                )
                increment = np.asarray(
                    validation.EXPECTED_DT32 * control, dtype=np.float32
                )
                next_state = np.asarray(
                    state + validation.EXPECTED_DT32 * total, dtype=np.float32
                )
                arrays[f"{prefix}_trace_step_index_i64"][replicate, step] = step
                arrays[f"{prefix}_trace_time_f32"][replicate, step] = validation._expected_times()[step]
                arrays[f"{prefix}_trace_active_bool"][replicate, step] = step >= 5
                arrays[f"{prefix}_trace_x_t_f32"][replicate, step] = state
                arrays[f"{prefix}_trace_v_base_f32"][replicate, step] = velocity
                arrays[f"{prefix}_trace_control_velocity_f32"][replicate, step] = control
                arrays[f"{prefix}_trace_total_velocity_f32"][replicate, step] = total
                arrays[f"{prefix}_trace_control_increment_f32"][replicate, step] = increment
                arrays[f"{prefix}_trace_x_next_f32"][replicate, step] = next_state
                state = next_state
            arrays[f"{prefix}_final_normalized_f32"][replicate] = state
            physical = np.asarray(state[:, :7], dtype=np.float64)
            arrays[f"{prefix}_final_normalized_physical_f64"][replicate] = physical
            arrays[f"{prefix}_returned_actions_f64"][replicate] = physical

    @classmethod
    def _simulate_generation(
        cls,
        arrays: dict[str, np.ndarray],
        prefix: str,
        *,
        projection_mode: int,
        gain: float,
        coupling: float,
    ) -> None:
        reference = arrays["reference_states_f32"]
        requested_rows: list[np.ndarray] = []
        state = arrays["source_noise_f32"].copy()
        for step in range(10):
            velocity = cls._field(state, gain=gain, coupling=coupling)
            if step < 5:
                compact = np.zeros((15,), dtype=np.float32)
            else:
                desired = reference[step - 4]
                base_next = np.asarray(
                    state + validation.EXPECTED_DT32 * velocity, dtype=np.float32
                )
                raw = np.asarray(
                    desired[:5, :3] - base_next[:5, :3], dtype=np.float32
                ).reshape(15)
                if projection_mode == 1:
                    compact, _norm, _scale, _projected = validation._project_block(
                        raw, validation.EXPECTED_RADIUS32
                    )
                else:
                    compact = raw
                requested_rows.append(compact.copy())
            control = np.zeros((10, 32), dtype=np.float32)
            if step >= 5:
                requested_u = np.asarray(
                    compact / validation.EXPECTED_DT32, dtype=np.float32
                )
                control[:5, :3] = requested_u.reshape(5, 3)
            total = np.where(
                control == np.float32(0.0), velocity, velocity + control
            )
            state = np.asarray(
                state + validation.EXPECTED_DT32 * total, dtype=np.float32
            )
        compact_c = np.stack(requested_rows).astype(np.float32)
        cls._simulate_fixed(
            arrays,
            prefix,
            compact_c,
            gain=gain,
            coupling=coupling,
        )
        cls._fill_reference_trace(arrays, prefix, projection_mode=projection_mode)

    @staticmethod
    def _fill_reference_trace(
        arrays: dict[str, np.ndarray], prefix: str, *, projection_mode: int
    ) -> None:
        arrays[f"{prefix}_reference_active_states_f32"][:] = arrays[
            "reference_states_f32"
        ]
        arrays[f"{prefix}_reference_delta_f32"][:] = arrays[
            "source_delta_model_f32"
        ]
        arrays[f"{prefix}_reference_alpha_f32"][:] = arrays[
            "reference_alpha_f32"
        ]
        arrays[f"{prefix}_reference_anchor_exact_bool"][:] = True
        arrays[f"{prefix}_reference_projection_mode_i64"][:] = projection_mode
        arrays[f"{prefix}_reference_source_budget_f32"][:] = validation.EXPECTED_BUDGET32
        arrays[f"{prefix}_reference_per_step_cap_f32"][:] = validation.EXPECTED_RADIUS32
        arrays[f"{prefix}_reference_product_ball_constraint_applied_bool"][:] = projection_mode == 1
        arrays[f"{prefix}_reference_product_ball_valid_bool"][:] = projection_mode == 1
        mask = validation._mask()
        for replicate in range(2):
            raw_path = np.float64(0.0)
            requested_path = np.float64(0.0)
            executed_path = np.float64(0.0)
            for step in range(10):
                x_t = arrays[f"{prefix}_trace_x_t_f32"][replicate, step]
                v_base = arrays[f"{prefix}_trace_v_base_f32"][replicate, step]
                uncontrolled = np.asarray(
                    x_t + validation.EXPECTED_DT32 * v_base, dtype=np.float32
                )
                arrays[f"{prefix}_reference_uncontrolled_next_steps_f32"][
                    replicate, step
                ] = uncontrolled
                arrays[f"{prefix}_reference_projection_scale_f64_steps"][
                    replicate, step
                ] = np.float64(1.0)
                if step < 5:
                    continue
                desired = arrays["reference_states_f32"][step - 4]
                raw = np.zeros((10, 32), dtype=np.float32)
                difference = np.asarray(desired - uncontrolled, dtype=np.float32)
                raw[mask] = difference[mask]
                compact_raw = raw[:5, :3].reshape(15)
                if projection_mode == 1:
                    compact_requested, raw_norm, scale, projected = validation._project_block(
                        compact_raw, validation.EXPECTED_RADIUS32
                    )
                else:
                    compact_requested = compact_raw.copy()
                    raw_norm = validation._fixed_norm64(compact_raw)
                    scale = np.float64(1.0)
                    projected = False
                requested = np.zeros((10, 32), dtype=np.float32)
                requested[:5, :3] = compact_requested.reshape(5, 3)
                requested_u = np.zeros((10, 32), dtype=np.float32)
                requested_u[mask] = np.asarray(
                    requested[mask] / validation.EXPECTED_DT32,
                    dtype=np.float32,
                )
                executed = np.zeros((10, 32), dtype=np.float32)
                executed[mask] = np.asarray(
                    validation.EXPECTED_DT32 * requested_u[mask],
                    dtype=np.float32,
                )
                requested_norm = validation._fixed_norm64(requested[:5, :3])
                executed_norm = validation._fixed_norm64(executed[:5, :3])
                tracking = np.zeros((10, 32), dtype=np.float32)
                tracking[mask] = np.asarray(
                    arrays[f"{prefix}_trace_x_next_f32"][replicate, step] - desired,
                    dtype=np.float32,
                )[mask]
                arrays[f"{prefix}_reference_desired_next_steps_f32"][replicate, step] = desired
                arrays[f"{prefix}_reference_raw_increment_steps_f32"][replicate, step] = raw
                arrays[f"{prefix}_reference_requested_increment_steps_f32"][replicate, step] = requested
                arrays[f"{prefix}_reference_requested_velocity_steps_f32"][replicate, step] = requested_u
                arrays[f"{prefix}_reference_executed_increment_steps_f32"][replicate, step] = executed
                arrays[f"{prefix}_reference_raw_norm_f64_steps"][replicate, step] = raw_norm
                arrays[f"{prefix}_reference_requested_norm_f64_steps"][replicate, step] = requested_norm
                arrays[f"{prefix}_reference_executed_norm_f64_steps"][replicate, step] = executed_norm
                arrays[f"{prefix}_reference_projection_scale_f64_steps"][replicate, step] = scale
                arrays[f"{prefix}_reference_projected_steps_bool"][replicate, step] = projected
                arrays[f"{prefix}_reference_tracking_error_steps_f32"][replicate, step] = tracking
                raw_path = np.float64(raw_path + raw_norm)
                requested_path = np.float64(requested_path + requested_norm)
                executed_path = np.float64(executed_path + executed_norm)
            arrays[f"{prefix}_reference_raw_path_length_f64"][replicate] = raw_path
            arrays[f"{prefix}_reference_requested_path_length_f64"][replicate] = requested_path
            arrays[f"{prefix}_reference_executed_path_length_f64"][replicate] = executed_path

    @staticmethod
    def _copy_replay(
        arrays: dict[str, np.ndarray], generation: str, replay: str
    ) -> None:
        for suffix in validation.ORDINARY_TRACE_SUFFIXES:
            arrays[f"{replay}_{suffix}"][:] = arrays[f"{generation}_{suffix}"][0]

    @staticmethod
    def _ledger(arrays: dict[str, np.ndarray]) -> list[dict]:
        rows: list[dict] = []
        for ordinal in range(18):
            phase, phase_index, trace_group, trace_index = validation._ledger_phase(
                ordinal
            )
            row = {
                "ordinal": ordinal,
                "phase": phase,
                "phase_index": phase_index,
                "reply_kind": "action",
                "elapsed_seconds": 0.001,
                "trace_group": trace_group,
                "trace_index": trace_index,
                "actions": validation._bytes_digest(
                    validation._ledger_action(arrays, ordinal)
                ),
            }
            if phase in {"budgeted_lift_generation", "raw_lift_generation"}:
                row["supplied_reference"] = validation._bytes_digest(
                    arrays["reference_states_f32"]
                )
                row["generated_schedule"] = validation._bytes_digest(
                    validation._full_schedule(
                        arrays[f"{trace_group}_requested_velocity_f32"][trace_index]
                    )
                )
                row["projection_mode"] = (
                    "product_ball"
                    if phase == "budgeted_lift_generation"
                    else "raw"
                )
            elif trace_group is not None:
                row["requested_schedule"] = validation._bytes_digest(
                    validation._full_schedule(
                        arrays[f"{trace_group}_requested_velocity_f32"][trace_index]
                    )
                )
            if trace_group is not None:
                budget = (
                    np.asarray(np.float32(0.0), dtype=np.float32)
                    if trace_group == "zero"
                    else (
                        arrays["raw_replay_budget_f32"]
                        if trace_group == "raw_replay"
                        else arrays["source_budget_f32"]
                    )
                )
                row["model_l2_path_budget"] = validation._bytes_digest(
                    np.asarray(budget, dtype=np.float32)
                )
            rows.append(row)
        return rows

    @staticmethod
    def _array_bindings(arrays: dict[str, np.ndarray]) -> dict[str, str]:
        return {
            name: validation._array_sha256(arrays[name])
            for name in validation.FROZEN_ARRAY_BINDINGS
        }

    @classmethod
    def _fixture(
        cls,
        *,
        gain: float = 2.0,
        coupling: float = 0.0,
        delta_value: float = 1.0,
        force_negative_outputs: bool = False,
    ) -> tuple[dict[str, np.ndarray], list[dict], dict[str, str], np.float64, np.ndarray]:
        arrays = cls._blank_arrays()
        arrays["source_budget_f32"] = np.asarray(
            validation.EXPECTED_BUDGET32, dtype=np.float32
        )
        arrays["source_dt_f32"] = np.asarray(
            validation.EXPECTED_DT32, dtype=np.float32
        )
        arrays["source_radius_f32"] = np.asarray(
            validation.EXPECTED_RADIUS32, dtype=np.float32
        )
        arrays["source_model_to_physical_scale_f32"][:] = np.float32(1.0)
        arrays["source_delta_model_f32"][0, 0] = np.float32(delta_value)
        arrays["source_target_normalized_f32"][0, 0] = np.float32(delta_value)
        arrays["source_target_physical_f64"][0, 0] = np.float64(
            np.float32(delta_value)
        )
        alpha = validation._expected_alpha()
        arrays["reference_alpha_f32"] = alpha
        for index in range(6):
            arrays["reference_states_f32"][index, 0, 0] = np.asarray(
                alpha[index] * np.float32(delta_value), dtype=np.float32
            )

        zero_c = np.zeros(validation.COMPACT_SHAPE, dtype=np.float32)
        cls._simulate_fixed(
            arrays, "zero", zero_c, gain=gain, coupling=0.0
        )
        arm_a_row = np.asarray(
            arrays["source_delta_model_f32"][:5, :3].reshape(15)
            / np.float32(5.0),
            dtype=np.float32,
        )
        arm_a_c = np.broadcast_to(arm_a_row, validation.COMPACT_SHAPE).copy()
        cls._simulate_fixed(
            arrays, "arm_a", arm_a_c, gain=gain, coupling=coupling
        )
        cls._simulate_generation(
            arrays,
            "arm_b_generation",
            projection_mode=1,
            gain=gain,
            coupling=coupling,
        )
        cls._copy_replay(arrays, "arm_b_generation", "arm_b_replay")
        cls._simulate_generation(
            arrays,
            "raw_generation",
            projection_mode=0,
            gain=gain,
            coupling=coupling,
        )
        cls._copy_replay(arrays, "raw_generation", "raw_replay")

        arrays["reference_compiled_actions_f64"][:] = arrays[
            "zero_returned_actions_f64"
        ][0]
        arrays["reference_source_actions_f64"][:] = arrays[
            "zero_returned_actions_f64"
        ][0]
        arrays["reference_eager_actions_f64"][:] = arrays[
            "zero_returned_actions_f64"
        ][0]
        arrays["reference_eager_final_f32"][:] = arrays[
            "source_frozen_final_f32"
        ]

        executed = arrays["raw_generation_executed_c_f32"][0]
        norms, path = validation._constraint_values(executed)
        maximum = np.asarray(np.max(norms), dtype=np.float32).reshape(())[()]
        exact = validation._constraints_accept(executed, validation.EXPECTED_BUDGET32)
        seed = np.asarray(
            max(
                np.float64(path),
                np.float64(5.0) * np.float64(maximum),
                np.float64(validation.EXPECTED_BUDGET32),
            ),
            dtype=np.float64,
        )
        replay_budget = (
            validation.EXPECTED_BUDGET32
            if exact
            else validation._upward_float32(seed)
        )
        arrays["raw_exact_budget_accepted_bool"] = np.asarray(exact, dtype=np.bool_)
        arrays["raw_path_f32"] = np.asarray(path, dtype=np.float32)
        arrays["raw_max_step_norm_f32"] = np.asarray(maximum, dtype=np.float32)
        arrays["raw_replay_seed_f64"] = np.asarray(seed, dtype=np.float64)
        arrays["raw_replay_budget_f32"] = np.asarray(replay_budget, dtype=np.float32)

        if force_negative_outputs:
            for prefix in (
                "arm_b_generation",
                "arm_b_replay",
                "raw_generation",
                "raw_replay",
            ):
                arrays[f"{prefix}_final_normalized_physical_f64"][:] = 0.0
                arrays[f"{prefix}_returned_actions_f64"][:] = 0.0

        arm_a_objective, arm_a_metrics, _ = validation._objective_metrics(
            arrays["arm_a_returned_actions_f64"][0],
            arrays["source_target_physical_f64"],
        )
        bindings = cls._array_bindings(arrays)
        ledger = cls._ledger(arrays)
        return arrays, ledger, bindings, arm_a_objective, arm_a_metrics

    @staticmethod
    def _validate_fixture(
        arrays: dict[str, np.ndarray],
        ledger: list[dict],
        bindings: dict[str, str],
        arm_a_objective: np.float64,
        arm_a_metrics: np.ndarray,
    ) -> dict:
        with mock.patch.object(
            validation, "EXPECTED_ARM_A_OBJECTIVE64", arm_a_objective
        ):
            with mock.patch.object(
                validation, "EXPECTED_ARM_A_METRICS64", arm_a_metrics
            ):
                with mock.patch.object(
                    validation, "FROZEN_ARRAY_BINDINGS", bindings
                ):
                    return validation.validate_reference_trajectory_lift_tensors(
                        arrays,
                        ledger,
                        validation.EXPECTED_SOURCE_BINDINGS,
                    )

    def test_finite_same_budget_witness_and_npz_round_trip(self) -> None:
        fixture = self._fixture()
        result = self._validate_fixture(*fixture)
        self.assertEqual(result["branch"], "finite")
        self.assertEqual(result["request_count"], 18)
        self.assertEqual(result["outcome"], "same_budget_lift_pass")
        self.assertTrue(result["budgeted"]["full_pass"])
        self.assertTrue(result["raw_authority"]["exact_budget_accepted"])
        self.assertFalse(result["claims"]["probe_or_mlp_training_authorized"])

        arrays, ledger, bindings, objective, metrics = fixture
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trl00a-tensors.npz"
            np.savez(path, **arrays)
            with mock.patch.object(
                validation, "EXPECTED_ARM_A_OBJECTIVE64", objective
            ):
                with mock.patch.object(
                    validation, "EXPECTED_ARM_A_METRICS64", metrics
                ):
                    with mock.patch.object(
                        validation, "FROZEN_ARRAY_BINDINGS", bindings
                    ):
                        loaded = validation.validate_reference_trajectory_lift_npz(
                            path,
                            ledger,
                            validation.EXPECTED_SOURCE_BINDINGS,
                        )
        self.assertEqual(loaded["outcome"], "same_budget_lift_pass")

    def test_canonical_budget_bottleneck_uses_upward_raw_envelope(self) -> None:
        fixture = self._fixture(gain=50.0, delta_value=1.0)
        result = self._validate_fixture(*fixture)
        self.assertEqual(result["outcome"], "canonical_budget_bottleneck")
        self.assertFalse(result["raw_authority"]["exact_budget_accepted"])
        self.assertGreater(
            result["raw_authority"]["replay_budget"],
            float(validation.EXPECTED_BUDGET32),
        )
        self.assertTrue(result["raw"]["full_pass"])

    def test_canonical_mask_coupling_precedes_negative(self) -> None:
        fixture = self._fixture(gain=2.0, coupling=10.0, delta_value=1.0)
        result = self._validate_fixture(*fixture)
        self.assertEqual(result["outcome"], "canonical_mask_coupling")
        self.assertTrue(result["raw"]["xyz_pass"])
        self.assertFalse(result["raw"]["full_pass"])

    def test_canonical_lift_negative_is_exhaustive_finite_fallback(self) -> None:
        fixture = self._fixture(force_negative_outputs=True)
        result = self._validate_fixture(*fixture)
        self.assertEqual(result["outcome"], "canonical_lift_negative")
        self.assertFalse(result["budgeted"]["xyz_pass"])
        self.assertFalse(result["raw"]["xyz_pass"])

    def test_exact_members_dtypes_source_bindings_and_frozen_bytes_fail_closed(self) -> None:
        arrays, ledger, bindings, objective, metrics = self._fixture()
        cases = []
        missing = dict(arrays)
        missing.pop("reference_alpha_f32")
        cases.append((missing, ledger, bindings, "tensor members"))
        extra = dict(arrays)
        extra["gpu_passed_bool"] = np.asarray(True, dtype=np.bool_)
        cases.append((extra, ledger, bindings, "tensor members"))
        wrong_dtype = dict(arrays)
        wrong_dtype["raw_path_f32"] = arrays["raw_path_f32"].astype(np.float64)
        cases.append((wrong_dtype, ledger, bindings, "must preserve"))
        wrong_source = dict(validation.EXPECTED_SOURCE_BINDINGS)
        wrong_source["source_r02_sha256"] = "0" * 64
        cases.append((arrays, ledger, bindings, "source binding"))

        for index, (candidate, candidate_ledger, candidate_bindings, message) in enumerate(cases[:3]):
            with self.subTest(case=index), self.assertRaisesRegex(
                validation.ReferenceTrajectoryLiftValidationError, message
            ):
                self._validate_fixture(
                    candidate,
                    candidate_ledger,
                    candidate_bindings,
                    objective,
                    metrics,
                )
        with self.assertRaisesRegex(
            validation.ReferenceTrajectoryLiftValidationError, "source binding"
        ):
            with mock.patch.object(
                validation, "EXPECTED_ARM_A_OBJECTIVE64", objective
            ):
                with mock.patch.object(
                    validation, "EXPECTED_ARM_A_METRICS64", metrics
                ):
                    with mock.patch.object(
                        validation, "FROZEN_ARRAY_BINDINGS", bindings
                    ):
                        validation.validate_reference_trajectory_lift_tensors(
                            arrays,
                            ledger,
                            wrong_source,
                        )
        bad_bound = dict(bindings)
        bad_bound["arm_a_returned_actions_f64"] = "0" * 64
        with self.assertRaisesRegex(
            validation.ReferenceTrajectoryLiftValidationError,
            "frozen bound array arm_a_returned_actions_f64 changed",
        ):
            self._validate_fixture(
                arrays, ledger, bad_bound, objective, metrics
            )

    def test_signed_zero_reference_and_full_reference_no_double_add_are_checked(self) -> None:
        arrays, ledger, bindings, objective, metrics = self._fixture()
        signed = copy.deepcopy(arrays)
        signed["reference_states_f32"][0, 0, 1] = np.float32(-0.0)
        signed_ledger = self._ledger(signed)
        with self.assertRaisesRegex(
            validation.ReferenceTrajectoryLiftValidationError,
            "reference reconstruction",
        ):
            self._validate_fixture(
                signed, signed_ledger, bindings, objective, metrics
            )

        doubled = copy.deepcopy(arrays)
        doubled[
            "arm_b_generation_reference_desired_next_steps_f32"
        ][:, 5, 0, 0] += doubled["source_delta_model_f32"][0, 0]
        with self.assertRaisesRegex(
            validation.ReferenceTrajectoryLiftValidationError,
            "consume the full reference directly",
        ):
            self._validate_fixture(doubled, ledger, bindings, objective, metrics)

    def test_projection_transport_recurrence_and_replay_tampers_are_rejected(self) -> None:
        arrays, ledger, bindings, objective, metrics = self._fixture()
        mutations = (
            (
                "projection scale",
                lambda value: value[
                    "arm_b_generation_reference_projection_scale_f64_steps"
                ].__setitem__((slice(None), 6), np.float64(0.5)),
            ),
            (
                "authoritative executed c",
                lambda value: value["raw_generation_executed_c_f32"].__setitem__(
                    (slice(None), 1, 0),
                    np.nextafter(
                        value["raw_generation_executed_c_f32"][:, 1, 0],
                        np.float32(np.inf),
                        dtype=np.float32,
                    ),
                ),
            ),
            (
                "Euler recurrence",
                lambda value: value["raw_generation_trace_x_next_f32"].__setitem__(
                    (slice(None), 7, 0, 0),
                    np.nextafter(
                        value["raw_generation_trace_x_next_f32"][:, 7, 0, 0],
                        np.float32(np.inf),
                        dtype=np.float32,
                    ),
                ),
            ),
            (
                "physical terminal|fixed-schedule witness",
                lambda value: value["raw_replay_returned_actions_f64"].__setitem__(
                    (slice(None), 0, 0),
                    np.nextafter(
                        value["raw_replay_returned_actions_f64"][:, 0, 0],
                        np.float64(np.inf),
                    ),
                ),
            ),
        )
        for message, mutate in mutations:
            candidate = copy.deepcopy(arrays)
            mutate(candidate)
            candidate_ledger = self._ledger(candidate)
            with self.subTest(message=message), self.assertRaisesRegex(
                validation.ReferenceTrajectoryLiftValidationError, message
            ):
                self._validate_fixture(
                    candidate, candidate_ledger, bindings, objective, metrics
                )

    def test_ledger_is_exact_and_raw_terminal_members_are_not_a_scientific_branch(self) -> None:
        arrays, ledger, bindings, objective, metrics = self._fixture()
        reordered = copy.deepcopy(ledger)
        reordered[6]["phase"] = "raw_lift_generation"
        with self.assertRaisesRegex(
            validation.ReferenceTrajectoryLiftValidationError,
            "row 6.phase",
        ):
            self._validate_fixture(
                arrays, reordered, bindings, objective, metrics
            )

        terminal = dict(arrays)
        terminal["raw_terminal_step_i64"] = np.asarray([7, 7], dtype=np.int64)
        with self.assertRaisesRegex(
            validation.ReferenceTrajectoryLiftValidationError,
            "finite tensor members changed",
        ):
            self._validate_fixture(
                terminal, ledger, bindings, objective, metrics
            )

        authority_fixture = self._fixture(gain=50.0, delta_value=1.0)
        authority_arrays, authority_ledger, authority_bindings, authority_objective, authority_metrics = authority_fixture
        wrong_budget = copy.deepcopy(authority_ledger)
        wrong_budget[12]["model_l2_path_budget"] = validation._bytes_digest(
            authority_arrays["source_budget_f32"]
        )
        with self.assertRaisesRegex(
            validation.ReferenceTrajectoryLiftValidationError,
            "row 12 model-space budget changed",
        ):
            self._validate_fixture(
                authority_arrays,
                wrong_budget,
                authority_bindings,
                authority_objective,
                authority_metrics,
            )

        wrong_projection = copy.deepcopy(ledger)
        wrong_projection[6]["projection_mode"] = "raw"
        with self.assertRaisesRegex(
            validation.ReferenceTrajectoryLiftValidationError,
            "row 6 projection mode changed",
        ):
            self._validate_fixture(
                arrays, wrong_projection, bindings, objective, metrics
            )


if __name__ == "__main__":
    unittest.main()
