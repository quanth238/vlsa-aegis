"""Read-only diagnostics for the sealed AF-00A run-C population.

This module performs no policy, model, simulator, optimizer-service, or Slurm
work.  It accepts only the immutable AF-00A run-C files whose hashes are
registered below, reuses the independent AF tensor validator, and separates
algebraic/descriptive population facts from an explicitly exploratory linear
surrogate.  Surrogate predictions are never promoted to sampler evaluations or
feasibility certificates.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .r05a_actual_forward_validation import validate_actual_forward_npz


SCHEMA_VERSION = "1.0"
ARTIFACT_TYPE = "r05a_af00a_sealed_population_offline_diagnostic"
RUN_ID = "r05a-actual-forward-cem-canary-20260716c"
CASE_ID = "crfs-1069f29a8d76463a"
SOURCE_COMMIT = "bd14f97eeffafd20525454db4d7a52614e1146c4"
SOURCE_JOB_ID = "28281_0"
EXPECTED_FILES = {
    "af00a-raw-payload.json": "00433437470ca7678a236c4778d5cf68160c8d3299f128b64985b1aeb142d5d1",
    "af00a-tensors.npz": "ada53973653ba7c21dab33cdbc4b4498c7b2b1840f2fc284bd3d3c76baf1d383",
    "query-ledger.json": "619f15ad46390d6e55e74365dc2530e58b383592ec998842d4bbd898c7266b92",
    "results.json": "507f25bc381b7bfeaa30abe3a7df970681f3b175f39221034c6e768e72dae914",
    "cpu-republication-validation.json": "5462f875ffb41a84a06c4df715366776af3811f3229385d05be8f5b0dafcb632",
}
EXPECTED_SELECTION_QUERY = 464
GENERATIONS = 8
POPULATION_SIZE = 65
SEARCH_QUERIES = GENERATIONS * POPULATION_SIZE
COMPACT_SHAPE = (5, 15)
GATE_SCALES = np.tile(
    np.asarray((0.005, 0.005, 0.005, 0.015, 0.015, 0.015, 0.015)), 5
).astype(np.float64)
MAX_ABS_TOLERANCES = np.tile(
    np.asarray((0.010, 0.010, 0.010, 0.050, 0.050, 0.050, 0.050)), 5
).astype(np.float64)


class ActualForwardPopulationDiagnosticError(ValueError):
    """The sealed input or offline diagnostic contract was violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ActualForwardPopulationDiagnosticError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, *, name: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{name} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ActualForwardPopulationDiagnosticError(f"{name} is not valid JSON: {error}") from error
    _require(isinstance(value, dict), f"{name} must contain one JSON object")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    _require(math.isfinite(result), f"{name} is nonfinite")
    return result


def _quantiles(value: np.ndarray) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    _require(array.size > 0 and bool(np.isfinite(array).all()), "quantile input is empty or nonfinite")
    points = np.quantile(array, (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))
    return {
        name: float(item)
        for name, item in zip(("min", "p10", "p25", "median", "p75", "p90", "max"), points)
    }


def _metric_summary(error_physical: np.ndarray) -> dict[str, float]:
    error = np.asarray(error_physical, dtype=np.float64).reshape(5, 7)
    xyz = error[:, :3]
    scaled = error.reshape(35) / GATE_SCALES
    return {
        "objective": float(np.mean(np.square(scaled), dtype=np.float64)),
        "xyz_max_abs": float(np.max(np.abs(xyz))),
        "xyz_rms": float(np.sqrt(np.mean(np.square(xyz), dtype=np.float64))),
        "full_max_abs": float(np.max(np.abs(error))),
        "full_rms": float(np.sqrt(np.mean(np.square(error), dtype=np.float64))),
    }


def _physical_first_five(value: np.ndarray, scale: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.ascontiguousarray((array[..., :5, :7] * scale[:5, :7]).reshape(*array.shape[:-2], 35))


def _expand_compact_sum(compact: np.ndarray) -> np.ndarray:
    value = np.asarray(compact, dtype=np.float64)
    _require(value.shape[-2:] == COMPACT_SHAPE, "compact controls must end in shape (5,15)")
    expanded = np.zeros((*value.shape[:-2], 10, 32), dtype=np.float64)
    expanded[..., :5, :3] = np.sum(value, axis=-2).reshape(*value.shape[:-2], 5, 3)
    return expanded


def _alpha(value: np.ndarray, target: np.ndarray) -> np.ndarray:
    first = np.asarray(value, dtype=np.float64)
    wanted = np.asarray(target, dtype=np.float64).reshape(-1)
    denominator = float(np.dot(wanted, wanted))
    _require(denominator > 0.0, "target-direction norm must be positive")
    return np.asarray(first @ wanted / denominator, dtype=np.float64)


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    numerator = np.sum(first * second, axis=-1, dtype=np.float64)
    denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
    _require(bool(np.all(denominator > 0.0)), "cosine denominator must be positive")
    return np.asarray(numerator / denominator, dtype=np.float64)


def _single_decomposition(
    final_action: np.ndarray,
    compact: np.ndarray,
    *,
    zero_action: np.ndarray,
    target_shift_xyz: np.ndarray,
    scale: np.ndarray,
    feedback_from_trace: np.ndarray,
    budget: float,
) -> dict[str, Any]:
    direct_model = _expand_compact_sum(np.asarray(compact)[None])[0]
    direct_physical = direct_model[:5, :7] * scale[:5, :7]
    response_physical = (
        np.asarray(final_action, dtype=np.float64)[:5, :7]
        - np.asarray(zero_action, dtype=np.float64)[:5, :7]
    )
    feedback_physical = response_physical - direct_physical
    direct_xyz = direct_physical[:, :3].reshape(15)
    feedback_xyz = feedback_physical[:, :3].reshape(15)
    response_xyz = response_physical[:, :3].reshape(15)
    norms = np.linalg.norm(np.asarray(compact, dtype=np.float64), axis=1)
    path = float(np.sum(norms, dtype=np.float64))
    direct_sum_norm = float(np.linalg.norm(np.sum(compact, axis=0)))
    reconstructed_physical_xyz = direct_physical[:, :3] + (
        np.asarray(feedback_from_trace, dtype=np.float64)[:5, :3] * scale[:5, :3]
    )
    reconstruction_error_xyz = response_physical[:, :3] - reconstructed_physical_xyz
    return {
        "path_model_l2": path,
        "path_over_budget": path / budget,
        "per_flow_step_model_l2": [float(item) for item in norms],
        "direct_sum_model_l2": direct_sum_norm,
        "triangle_efficiency": direct_sum_norm / path,
        "direct_target_alpha": float(_alpha(direct_xyz[None], target_shift_xyz)[0]),
        "feedback_target_alpha": float(_alpha(feedback_xyz[None], target_shift_xyz)[0]),
        "terminal_target_alpha": float(_alpha(response_xyz[None], target_shift_xyz)[0]),
        "feedback_direct_cosine_physical_xyz": float(
            _cosine(feedback_xyz[None], direct_xyz[None])[0]
        ),
        "trace_reconstruction_max_abs_physical_xyz": float(
            np.max(np.abs(reconstruction_error_xyz))
        ),
        "trace_reconstruction_l2_physical_xyz": float(
            np.linalg.norm(reconstruction_error_xyz)
        ),
    }


def exact_descriptive_diagnostic(
    arrays: Mapping[str, np.ndarray], *, selected_query: int
) -> dict[str, Any]:
    """Compute algebraic facts directly from sealed controls and traces."""

    scale = np.asarray(arrays["source_model_to_physical_scale_f32"], dtype=np.float64)
    controls = np.asarray(arrays["cem_executed_c_f32"], dtype=np.float64)
    zero_action = np.asarray(arrays["zero_returned_actions_f64"][0], dtype=np.float64)
    target_action = np.asarray(arrays["source_target_physical_f64"], dtype=np.float64)
    actions = np.asarray(arrays["cem_returned_actions_f64"], dtype=np.float64)
    _require(controls.shape == (SEARCH_QUERIES, *COMPACT_SHAPE), "sealed CEM control shape changed")
    target_shift_xyz = (target_action[:5, :3] - zero_action[:5, :3]).reshape(15)
    target_norm = float(np.linalg.norm(target_shift_xyz))
    _require(target_norm > 0.0, "sealed target physical XYZ shift is zero")
    pass_alpha_delta = math.sqrt(15.0) * 0.005 / target_norm

    direct_model = _expand_compact_sum(controls)
    direct_physical = direct_model[:, :5, :7] * scale[:5, :7]
    response_physical = actions[:, :5, :7] - zero_action[:5, :7]
    feedback_physical = response_physical - direct_physical
    direct_xyz = direct_physical[:, :, :3].reshape(SEARCH_QUERIES, 15)
    feedback_xyz = feedback_physical[:, :, :3].reshape(SEARCH_QUERIES, 15)
    response_xyz = response_physical[:, :, :3].reshape(SEARCH_QUERIES, 15)
    direct_alpha = _alpha(direct_xyz, target_shift_xyz)
    feedback_alpha = _alpha(feedback_xyz, target_shift_xyz)
    terminal_alpha = _alpha(response_xyz, target_shift_xyz)
    feedback_cosine = _cosine(feedback_xyz, direct_xyz)

    dt = float(np.asarray(arrays["source_dt_f32"]).reshape(()))
    zero_vbase = np.asarray(arrays["zero_trace_v_base_f32"][0], dtype=np.float64)
    cem_vbase = np.asarray(arrays["cem_trace_v_base_f32"], dtype=np.float64)
    trace_feedback = dt * np.sum(cem_vbase - zero_vbase, axis=1, dtype=np.float64)
    population_reconstruction_physical_xyz = direct_physical[:, :, :3] + (
        trace_feedback[:, :5, :3] * scale[:5, :3]
    )
    population_reconstruction_error_xyz = (
        response_physical[:, :, :3] - population_reconstruction_physical_xyz
    )

    arm_a_control = np.asarray(arrays["arm_a_executed_c_f32"][0], dtype=np.float64)
    arm_a_vbase = np.asarray(arrays["arm_a_trace_v_base_f32"][0], dtype=np.float64)
    arm_a_trace_feedback = dt * np.sum(arm_a_vbase - zero_vbase, axis=0, dtype=np.float64)
    budget = float(np.asarray(arrays["source_budget_f32"]).reshape(()))
    arm_a = _single_decomposition(
        arrays["arm_a_returned_actions_f64"][0],
        arm_a_control,
        zero_action=zero_action,
        target_shift_xyz=target_shift_xyz,
        scale=scale,
        feedback_from_trace=arm_a_trace_feedback,
        budget=budget,
    )
    selected = _single_decomposition(
        actions[selected_query],
        controls[selected_query],
        zero_action=zero_action,
        target_shift_xyz=target_shift_xyz,
        scale=scale,
        feedback_from_trace=trace_feedback[selected_query],
        budget=budget,
    )

    path = np.sum(np.linalg.norm(controls, axis=2), axis=1, dtype=np.float64)
    direct_norm = np.linalg.norm(np.sum(controls, axis=1), axis=1)
    return {
        "evidence_class": "exact_algebraic_and_observed_population",
        "definitions": {
            "target_shift": "source_target_physical_f64-minus-zero_returned_actions_f64[0]",
            "direct_control": "sum_over_flow_steps_of_cem_executed_c_f32",
            "field_feedback": "terminal_response_minus_direct_control",
            "physical_displacement": "checkpoint_scale_only_no_mean_subtraction",
            "target_alpha": "physical_XYZ_dot_target_div_target_squared_norm",
        },
        "target_physical_xyz_l2": target_norm,
        "necessary_xyz_rms_pass_alpha_interval": [
            1.0 - pass_alpha_delta,
            1.0 + pass_alpha_delta,
        ],
        "arm_a": arm_a,
        "selected_b": {"cem_query_index": selected_query, **selected},
        "population": {
            "direct_target_alpha": _quantiles(direct_alpha),
            "feedback_target_alpha": _quantiles(feedback_alpha),
            "terminal_target_alpha": _quantiles(terminal_alpha),
            "feedback_direct_cosine_physical_xyz": _quantiles(feedback_cosine),
            "path_over_budget": _quantiles(path / budget),
            "triangle_efficiency": _quantiles(direct_norm / path),
            "trace_reconstruction_max_abs_physical_xyz": float(
                np.max(np.abs(population_reconstruction_error_xyz))
            ),
            "trace_reconstruction_l2_physical_xyz": _quantiles(
                np.linalg.norm(
                    population_reconstruction_error_xyz.reshape(SEARCH_QUERIES, 15),
                    axis=1,
                )
            ),
            "feedback_target_alpha_strictly_negative_count": int(np.count_nonzero(feedback_alpha < 0.0)),
            "terminal_alpha_meets_necessary_xyz_rms_bound_count": int(
                np.count_nonzero(terminal_alpha >= 1.0 - pass_alpha_delta)
            ),
        },
    }


def _rankdata_no_ties(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    ranks[order] = np.arange(array.size, dtype=np.float64)
    # Exact ties receive their average rank; the sealed objectives have none,
    # but this keeps generated tests and the definition complete.
    sorted_values = array[order]
    start = 0
    while start < array.size:
        stop = start + 1
        while stop < array.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        if stop - start > 1:
            ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    first = _rankdata_no_ties(left)
    second = _rankdata_no_ties(right)
    if np.std(first) == 0.0 or np.std(second) == 0.0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _matrix_coverage(value: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(value, dtype=np.float64)
    singular = np.linalg.svd(matrix, compute_uv=False)
    _require(singular.size > 0 and singular[0] > 0.0, "coverage matrix is degenerate")
    energy = np.square(singular)
    threshold = singular[0] * max(matrix.shape) * np.finfo(np.float64).eps
    cumulative = np.cumsum(energy) / np.sum(energy)
    return {
        "rank_at_numpy_default_threshold": int(np.count_nonzero(singular > threshold)),
        "condition_number_nonzero": float(singular[0] / singular[-1]),
        "participation_ratio": float(np.square(np.sum(energy)) / np.sum(np.square(energy))),
        "components_for_99_percent_energy": int(np.searchsorted(cumulative, 0.99) + 1),
    }


def search_diagnostic(arrays: Mapping[str, np.ndarray], arm_a_objective: float) -> dict[str, Any]:
    objective = np.asarray(arrays["cem_objective_f64"], dtype=np.float64)
    controls = np.asarray(arrays["cem_executed_c_f32"], dtype=np.float64)
    actions = np.asarray(arrays["cem_returned_actions_f64"], dtype=np.float64)[:, :5, :7]
    raw = np.asarray(arrays["cem_raw_proposal_f64"], dtype=np.float64)
    sigma = np.asarray(arrays["cem_sigma_state_f64"], dtype=np.float64)
    radius = float(np.asarray(arrays["source_radius_f32"]).reshape(()))
    budget = float(np.asarray(arrays["source_budget_f32"]).reshape(()))
    _require(objective.shape == (SEARCH_QUERIES,), "CEM objective count changed")
    generation_rows = []
    incumbent = arm_a_objective
    pair_input_asymmetry: list[float] = []
    pair_output_curvature: list[float] = []
    for generation in range(GENERATIONS):
        start = generation * POPULATION_SIZE
        stop = start + POPULATION_SIZE
        values = objective[start:stop]
        incumbent = min(incumbent, float(np.min(values)))
        generation_rows.append(
            {
                "generation": generation,
                "center_objective": float(values[0]),
                "minimum_objective": float(np.min(values)),
                "median_objective": float(np.median(values)),
                "maximum_objective": float(np.max(values)),
                "count_better_than_arm_a": int(np.count_nonzero(values < arm_a_objective)),
                "incumbent_objective": incumbent,
            }
        )
        center_c = controls[start]
        center_y = actions[start].reshape(35) / GATE_SCALES
        for pair in range(32):
            plus = start + 1 + 2 * pair
            minus = plus + 1
            input_half_chord = 0.5 * np.linalg.norm(controls[plus] - controls[minus])
            output_half_chord = 0.5 * np.linalg.norm(
                actions[plus].reshape(35) / GATE_SCALES
                - actions[minus].reshape(35) / GATE_SCALES
            )
            _require(input_half_chord > 0.0 and output_half_chord > 0.0, "antithetic chord collapsed")
            pair_input_asymmetry.append(
                float(np.linalg.norm(0.5 * (controls[plus] + controls[minus]) - center_c) / input_half_chord)
            )
            pair_output_curvature.append(
                float(
                    np.linalg.norm(
                        0.5
                        * (
                            actions[plus].reshape(35) / GATE_SCALES
                            + actions[minus].reshape(35) / GATE_SCALES
                        )
                        - center_y
                    )
                    / output_half_chord
                )
            )

    better = np.flatnonzero(objective < arm_a_objective)
    raw_block_norm = np.linalg.norm(raw, axis=2)
    block_norm = np.linalg.norm(controls, axis=2)
    path = np.sum(block_norm, axis=1, dtype=np.float64)
    sigma0 = float(sigma[0, 0, 0])
    sigma_floor = sigma0 / 16.0
    sigma_rows = []
    for generation in range(GENERATIONS + 1):
        state = sigma[generation].reshape(-1)
        sigma_rows.append(
            {
                "state_index": generation,
                "median_over_initial": float(np.median(state / sigma0)),
                "geometric_mean_over_initial": float(np.exp(np.mean(np.log(state / sigma0)))),
                "dimensions_at_or_below_floor": int(np.count_nonzero(state <= sigma_floor)),
                "dimensions_at_or_above_initial": int(np.count_nonzero(state >= sigma0)),
            }
        )
    return {
        "evidence_class": "exact_observed_frozen_search_behavior",
        "generation_table": generation_rows,
        "count_queries_better_than_arm_a": int(better.size),
        "first_query_better_than_arm_a": None if not better.size else int(better[0]),
        "first_generation_better_than_arm_a": None if not better.size else int(better[0] // POPULATION_SIZE),
        "raw_block_projection_fraction": float(np.mean(raw_block_norm > radius)),
        "candidate_any_raw_block_projected_fraction": float(np.mean(np.any(raw_block_norm > radius, axis=1))),
        "executed_block_norm_over_radius": _quantiles(block_norm / radius),
        "executed_path_over_budget": _quantiles(path / budget),
        "antithetic_input_midpoint_asymmetry": _quantiles(np.asarray(pair_input_asymmetry)),
        "antithetic_output_curvature_gate_scaled": _quantiles(np.asarray(pair_output_curvature)),
        "generation_zero_antithetic_input_midpoint_asymmetry": _quantiles(
            np.asarray(pair_input_asymmetry[:32])
        ),
        "generation_zero_antithetic_output_curvature_gate_scaled": _quantiles(
            np.asarray(pair_output_curvature[:32])
        ),
        "sigma_states": sigma_rows,
        "centered_input_coverage": _matrix_coverage(controls.reshape(SEARCH_QUERIES, 75) - np.mean(controls.reshape(SEARCH_QUERIES, 75), axis=0)),
    }


def observed_envelope_diagnostic(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    zero = np.asarray(arrays["zero_returned_actions_f64"][0, :5, :7], dtype=np.float64).reshape(35)
    target = np.asarray(arrays["source_target_physical_f64"][:5, :7], dtype=np.float64).reshape(35)
    arm_a = np.asarray(arrays["arm_a_returned_actions_f64"][0, :5, :7], dtype=np.float64).reshape(35)
    cem = np.asarray(arrays["cem_returned_actions_f64"][:, :5, :7], dtype=np.float64).reshape(SEARCH_QUERIES, 35)
    actions = np.vstack((arm_a, cem))
    response = actions - zero
    desired = target - zero
    low = np.min(response, axis=0)
    high = np.max(response, axis=0)
    inside = np.logical_and(desired >= low, desired <= high)
    expanded = np.logical_and(desired >= low - MAX_ABS_TOLERANCES, desired <= high + MAX_ABS_TOLERANCES)
    error = actions - target
    best_rows = np.argmin(np.abs(error), axis=0)
    coordinatewise = error[best_rows, np.arange(35)].reshape(5, 7)
    return {
        "evidence_class": "observed_population_marginals_not_a_reachable_combination",
        "candidate_count_including_arm_a": int(actions.shape[0]),
        "target_coordinates_inside_raw_response_envelope": int(np.count_nonzero(inside)),
        "target_xyz_coordinates_inside_raw_response_envelope": int(np.count_nonzero(inside.reshape(5, 7)[:, :3])),
        "target_coordinates_inside_max_tolerance_expanded_envelope": int(np.count_nonzero(expanded)),
        "target_xyz_coordinates_inside_max_tolerance_expanded_envelope": int(
            np.count_nonzero(expanded.reshape(5, 7)[:, :3])
        ),
        "coordinatewise_oracle_is_not_one_executed_schedule": True,
        "coordinatewise_oracle_metrics": _metric_summary(coordinatewise),
    }


def _project_group_ball(value: np.ndarray, *, radius: float, budget: float) -> np.ndarray:
    blocks = np.asarray(value, dtype=np.float64).reshape(COMPACT_SHAPE).copy()
    _require(radius > 0.0 and budget > 0.0, "projection limits must be positive")
    norms = np.linalg.norm(blocks, axis=1)
    capped = np.minimum(norms, radius)
    if float(np.sum(capped)) > budget:
        low = 0.0
        high = float(np.max(norms))
        for _ in range(96):
            midpoint = 0.5 * (low + high)
            projected_norms = np.minimum(radius, np.maximum(0.0, norms - midpoint))
            if float(np.sum(projected_norms)) > budget:
                low = midpoint
            else:
                high = midpoint
        capped = np.minimum(radius, np.maximum(0.0, norms - high))
    nonzero = norms > 0.0
    blocks[~nonzero] = 0.0
    blocks[nonzero] *= (capped[nonzero] / norms[nonzero])[:, None]
    return blocks.reshape(75)


def _solve_surrogate(
    response_map: np.ndarray,
    desired: np.ndarray,
    *,
    radius: float,
    budget: float,
) -> tuple[np.ndarray, int, float]:
    jacobian = np.asarray(response_map, dtype=np.float64)
    target = np.asarray(desired, dtype=np.float64)
    spectral = float(np.linalg.norm(jacobian, ord=2))
    _require(spectral > 0.0, "surrogate response map is zero")
    lipschitz = 2.0 * spectral * spectral / 35.0
    current = np.zeros((75,), dtype=np.float64)
    accelerated = current.copy()
    momentum = 1.0
    iterations = 0
    for iteration in range(1, 20001):
        gradient = 2.0 * ((accelerated @ jacobian - target) @ jacobian.T) / 35.0
        candidate = _project_group_ball(
            accelerated - gradient / lipschitz, radius=radius, budget=budget
        )
        iterations = iteration
        if np.linalg.norm(candidate - current) <= 1.0e-12 * max(1.0, np.linalg.norm(current)):
            current = candidate
            break
        next_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
        accelerated = candidate + (momentum - 1.0) / next_momentum * (candidate - current)
        current = candidate
        momentum = next_momentum
    gradient = 2.0 * ((current @ jacobian - target) @ jacobian.T) / 35.0
    projected = _project_group_ball(
        current - gradient / lipschitz, radius=radius, budget=budget
    )
    projected_gradient = float(np.linalg.norm(current - projected))
    return current, iterations, projected_gradient


def _surrogate_solution_summary(
    response_map: np.ndarray,
    desired: np.ndarray,
    *,
    radius: float,
    budget: float,
) -> dict[str, Any]:
    control, iterations, projected_gradient = _solve_surrogate(
        response_map, desired, radius=radius, budget=budget
    )
    scaled_error = control @ response_map - desired
    physical_error = scaled_error * GATE_SCALES
    block_norm = np.linalg.norm(control.reshape(COMPACT_SHAPE), axis=1)
    return {
        "prediction_only_not_actual_forward_evaluation": True,
        "metrics": _metric_summary(physical_error.reshape(5, 7)),
        "block_norm_over_radius": [float(item / radius) for item in block_norm],
        "path_over_budget": float(np.sum(block_norm) / budget),
        "iterations": iterations,
        "projected_fixed_point_residual_at_inverse_lipschitz_step": projected_gradient,
    }


def surrogate_diagnostic(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Fit a zero-anchored empirical secant model; never call the sampler."""

    controls = np.asarray(arrays["cem_executed_c_f32"], dtype=np.float64).reshape(SEARCH_QUERIES, 75)
    zero = np.asarray(arrays["zero_returned_actions_f64"][0, :5, :7], dtype=np.float64).reshape(35)
    actions = np.asarray(arrays["cem_returned_actions_f64"][:, :5, :7], dtype=np.float64).reshape(SEARCH_QUERIES, 35)
    target = np.asarray(arrays["source_target_physical_f64"][:5, :7], dtype=np.float64).reshape(35)
    response = (actions - zero) / GATE_SCALES
    desired = (target - zero) / GATE_SCALES
    response_map = np.linalg.lstsq(controls, response, rcond=None)[0]
    prediction = controls @ response_map
    residual = prediction - response
    centered_denominator = float(np.sum(np.square(response - np.mean(response, axis=0))))
    zero_denominator = float(np.sum(np.square(response)))
    _require(centered_denominator > 0.0 and zero_denominator > 0.0, "surrogate response is degenerate")

    held_out_prediction = np.empty_like(response)
    fold_rows = []
    radius = float(np.asarray(arrays["source_radius_f32"]).reshape(()))
    budget = float(np.asarray(arrays["source_budget_f32"]).reshape(()))
    fold_solutions = []
    for generation in range(GENERATIONS):
        held = np.arange(SEARCH_QUERIES) // POPULATION_SIZE == generation
        train = ~held
        fold_map = np.linalg.lstsq(controls[train], response[train], rcond=None)[0]
        fold_prediction = controls[held] @ fold_map
        held_out_prediction[held] = fold_prediction
        held_response = response[held]
        fold_centered = float(np.sum(np.square(held_response - np.mean(held_response, axis=0))))
        fold_zero = float(np.sum(np.square(held_response)))
        true_objective = np.mean(np.square(held_response - desired), axis=1)
        predicted_objective = np.mean(np.square(fold_prediction - desired), axis=1)
        solution = _surrogate_solution_summary(
            fold_map, desired, radius=radius, budget=budget
        )
        fold_solutions.append(solution)
        fold_rows.append(
            {
                "held_out_generation": generation,
                "centered_r2": 1.0 - float(np.sum(np.square(fold_prediction - held_response))) / fold_centered,
                "zero_anchored_energy_r2": 1.0 - float(np.sum(np.square(fold_prediction - held_response))) / fold_zero,
                "gate_scaled_rmse": float(np.sqrt(np.mean(np.square(fold_prediction - held_response)))),
                "objective_rank_spearman": _spearman(true_objective, predicted_objective),
                "constrained_inverse_prediction": solution,
            }
        )

    held_residual = held_out_prediction - response
    constrained = _surrogate_solution_summary(
        response_map, desired, radius=radius, budget=budget
    )
    unconstrained = np.linalg.lstsq(response_map.T, desired, rcond=None)[0]
    unconstrained_error = unconstrained @ response_map - desired
    unconstrained_blocks = np.linalg.norm(unconstrained.reshape(COMPACT_SHAPE), axis=1)
    fold_objectives = [row["metrics"]["objective"] for row in fold_solutions]
    fold_xyz_max = [row["metrics"]["xyz_max_abs"] for row in fold_solutions]
    fold_xyz_rms = [row["metrics"]["xyz_rms"] for row in fold_solutions]
    return {
        "evidence_class": "retrospective_exploratory_surrogate_not_sampler_evidence",
        "model": {
            "input": "flattened_cem_executed_c_f32_520_by_75",
            "output": "gate_scaled_returned_physical_action_minus_exact_zero_action_520_by_35",
            "fit": "ordinary_least_squares_without_intercept_zero_response_anchored",
            "regularization_or_hyperparameter_tuning": False,
            "rank": int(np.linalg.matrix_rank(controls)),
            "condition_number": float(np.linalg.cond(controls)),
        },
        "in_sample": {
            "centered_r2": 1.0 - float(np.sum(np.square(residual))) / centered_denominator,
            "zero_anchored_energy_r2": 1.0 - float(np.sum(np.square(residual))) / zero_denominator,
            "gate_scaled_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        },
        "leave_one_complete_generation_out": {
            "folds": fold_rows,
            "pooled_centered_r2": 1.0
            - float(np.sum(np.square(held_residual))) / centered_denominator,
            "pooled_zero_anchored_energy_r2": 1.0
            - float(np.sum(np.square(held_residual))) / zero_denominator,
            "pooled_gate_scaled_rmse": float(np.sqrt(np.mean(np.square(held_residual)))),
        },
        "constrained_inverse_prediction": constrained,
        "cross_fit_constrained_inverse_prediction_ranges": {
            "objective": [float(min(fold_objectives)), float(max(fold_objectives))],
            "xyz_max_abs": [float(min(fold_xyz_max)), float(max(fold_xyz_max))],
            "xyz_rms": [float(min(fold_xyz_rms)), float(max(fold_xyz_rms))],
            "every_fold_saturates_all_five_blocks": bool(
                all(
                    all(abs(item - 1.0) <= 1.0e-9 for item in row["block_norm_over_radius"])
                    for row in fold_solutions
                )
            ),
        },
        "unconstrained_minimum_norm_prediction": {
            "prediction_only_not_actual_forward_evaluation": True,
            "metrics": _metric_summary((unconstrained_error * GATE_SCALES).reshape(5, 7)),
            "block_norm_over_radius": [float(item / radius) for item in unconstrained_blocks],
            "path_over_budget": float(np.sum(unconstrained_blocks) / budget),
        },
    }


def _validate_bound_files(run_root: Path) -> tuple[dict[str, str], dict[str, Any], dict[str, Any], list[Any]]:
    _require(run_root.is_dir() and not run_root.is_symlink(), "run root must be one non-symlinked directory")
    hashes: dict[str, str] = {}
    for filename, expected in EXPECTED_FILES.items():
        path = run_root / filename
        _require(path.is_file() and not path.is_symlink(), f"missing sealed file: {filename}")
        observed = _file_sha256(path)
        _require(observed == expected, f"sealed file hash changed: {filename}")
        hashes[filename] = observed
    payload = _json_object(run_root / "af00a-raw-payload.json", name="raw payload")
    result = _json_object(run_root / "results.json", name="published result")
    receipt = _json_object(
        run_root / "cpu-republication-validation.json", name="republication receipt"
    )
    ledger_artifact = _json_object(run_root / "query-ledger.json", name="query ledger")
    ledger = ledger_artifact.get("rows")
    _require(isinstance(ledger, list), "query ledger rows must be a JSON array")
    _require(payload.get("run_id") == RUN_ID and payload.get("case_id") == CASE_ID, "payload identity changed")
    _require(payload.get("status") == "frozen_cem_negative", "payload outcome changed")
    _require(result.get("run_id") == RUN_ID and result.get("case_id") == CASE_ID, "result identity changed")
    _require(result.get("status") == "frozen_cem_negative", "published outcome changed")
    execution = result.get("execution_identity", {})
    _require(
        execution.get("git_commit") == SOURCE_COMMIT
        and execution.get("exact_gpu_task_id") == SOURCE_JOB_ID,
        "published source identity changed",
    )
    _require(receipt.get("published") is True and receipt.get("passed") is True, "republication receipt is not passing")
    _require(receipt.get("outcome") == "frozen_cem_negative", "receipt outcome changed")
    return hashes, payload, result, ledger


def diagnose_bound_run(run_root: str | Path) -> dict[str, Any]:
    """Validate and diagnose exactly the immutable AF-00A run-C population."""

    supplied_root = Path(run_root)
    _require(not supplied_root.is_symlink(), "run root must not be a symlink")
    root = supplied_root.resolve()
    hashes, payload, result, ledger = _validate_bound_files(root)
    semantic = validate_actual_forward_npz(root / "af00a-tensors.npz", ledger)
    _require(semantic.get("outcome") == "frozen_cem_negative", "independent AF outcome changed")
    _require(semantic.get("request_count") == 534, "independent AF request count changed")
    _require(
        semantic.get("selected_cem_query_index") == EXPECTED_SELECTION_QUERY,
        "independent selected CEM query changed",
    )
    with np.load(root / "af00a-tensors.npz", allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
        exact = exact_descriptive_diagnostic(arrays, selected_query=EXPECTED_SELECTION_QUERY)
        search = search_diagnostic(arrays, float(semantic["arm_a"]["objective"]))
        envelope = observed_envelope_diagnostic(arrays)
        surrogate = surrogate_diagnostic(arrays)

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "analysis_mode": "read_only_offline_no_model_no_simulator_no_slurm_no_training",
        "identity": {
            "run_id": RUN_ID,
            "case_id": CASE_ID,
            "source_commit": SOURCE_COMMIT,
            "source_job_id": SOURCE_JOB_ID,
            "sealed_file_sha256": hashes,
        },
        "integrity_validation": {
            "all_registered_hashes_match": True,
            "existing_af_semantic_validator_passed": True,
            "request_count": int(semantic["request_count"]),
            "search_query_count": SEARCH_QUERIES,
            "selected_cem_query_index": int(semantic["selected_cem_query_index"]),
            "published_outcome": result["status"],
            "payload_outcome": payload["status"],
            "independently_recomputed_outcome": semantic["outcome"],
        },
        "exact_descriptive_diagnostics": {
            "transport_decomposition": exact,
            "frozen_search": search,
            "observed_output_envelope": envelope,
        },
        "surrogate_diagnostics": surrogate,
        "interpretation": {
            "classification": "retrospective_evidence_favors_same_budget_control_authority_as_primary_observed_barrier_with_secondary_fixed_cem_inefficiency",
            "fixed_cem_search_inefficient": True,
            "same_budget_control_authority_primary_observed_barrier_favored": True,
            "reachability_versus_optimizer_proven": False,
            "reason": (
                "CEM first improves Arm A only in generation seven, while every observed "
                "field reaction opposes the target and no observed terminal response reaches "
                "the necessary target-direction alpha; every cross-fit surrogate inverse also "
                "saturates all caps and remains far outside fidelity gates."
            ),
        },
        "claim_boundaries": {
            "global_infeasibility_proven": False,
            "actual_surrogate_candidate_evaluated_by_sampler": False,
            "simulator_safety_or_progress_evaluated": False,
            "generalization_or_novelty_supported": False,
            "solver_budget_or_tolerance_tuning_authorized": False,
            "ift01_authorized": False,
            "probe_or_residual_mlp_training_authorized": False,
        },
    }


def deterministic_json(value: Mapping[str, Any]) -> str:
    """Return one stable, strict JSON representation."""

    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


__all__ = [
    "ActualForwardPopulationDiagnosticError",
    "EXPECTED_FILES",
    "diagnose_bound_run",
    "deterministic_json",
    "exact_descriptive_diagnostic",
    "observed_envelope_diagnostic",
    "search_diagnostic",
    "surrogate_diagnostic",
]
