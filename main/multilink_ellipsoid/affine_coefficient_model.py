"""State-conditioned affine safety-row targets and neural surrogate.

This module is opt-in.  It learns the quantities a bounded safety QP actually
uses: nominal margin, local action coefficient, and a nonnegative one-sided
error.  Exact cloned OSC rollouts remain the label and verification authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .execution_margin_nn import CONSTRAINT_ORDER, _canonical, _numpy, _torch
from .two_step_margin import PAIR_FEATURE_NAMES


AFFINE_COEFFICIENT_SCHEMA = "vlsa_distal_affine_coefficient_moka10.v1"
AFFINE_COEFFICIENT_DATASET_SCHEMA = (
    "vlsa_distal_affine_coefficient_moka10_dataset.v1"
)
AFFINE_COEFFICIENT_DATASET_RESULT_SCHEMA = (
    "vlsa_distal_affine_coefficient_moka10_dataset_result.v1"
)
AFFINE_COEFFICIENT_TRAINING_SCHEMA = (
    "vlsa_distal_affine_coefficient_moka10_training.v1"
)
AFFINE_COEFFICIENT_RESULT_SCHEMA = (
    "vlsa_distal_affine_coefficient_moka10_result.v1"
)
AFFINE_COEFFICIENT_VALIDATION_SCHEMA = (
    "vlsa_distal_affine_coefficient_moka10_validation.v1"
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_affine_coefficient_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("affine-coefficient config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope",
        "source_population_manifest_sha256", "selected_manifest_sha256",
        "geometry_config_file_sha256", "exact_box_config_file_sha256",
        "split", "state_sampling", "action_sampling", "coefficient_target",
        "model", "training", "calibration", "projection", "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("affine-coefficient config keys differ")
    if (
        config["schema_version"] != AFFINE_COEFFICIENT_SCHEMA
        or config["protocol_id"] != "vlsa-distal-affine-coefficient-moka10-v1"
    ):
        raise ValueError("affine-coefficient protocol differs")
    split = config["split"]
    groups = {
        name: set(split[name])
        for name in (
            "train_task_groups", "validation_task_groups", "test_task_groups"
        )
    }
    if (
        split.get("unit") != "complete_episode_and_task_level_group"
        or split.get("primary_e05_use")
        != "test_only_never_training_or_calibration"
        or not all(groups.values())
        or groups["train_task_groups"] & groups["validation_task_groups"]
        or groups["train_task_groups"] & groups["test_task_groups"]
        or groups["validation_task_groups"] & groups["test_task_groups"]
    ):
        raise ValueError("affine-coefficient grouped split differs")
    state = config["state_sampling"]
    if state != {
        "search_start_offset_actions": 20,
        "crossing": "first_two_step_exact_proxy_crossing_in_registered_window",
        "state_offsets_from_crossing": [-4, -3, -2, -1, 0],
        "expected_states_per_episode": 5,
    }:
        raise ValueError("affine-coefficient state sampling differs")
    sampling = config["action_sampling"]
    if sampling != {
        "action_limit": 1.0, "trust_region_linf_action": 0.5,
        "grid_points_per_dimension": 5,
        "expected_grid_action_count_per_state": 125,
        "second_action": "immutable_released_AEGIS_nominal",
    }:
        raise ValueError("affine-coefficient action sampling differs")
    target = config["coefficient_target"]
    if target != {
        "candidate_order": "exact_safe_then_l2_from_nominal_then_grid_index",
        "fit": "candidate_conditioned_minimum_l1_sampled_grid_lower_envelope",
        "one_sided_padding_m": 1.0e-6,
        "target_clearance_m": 0.0,
        "coefficient_postcheck_tolerance_m": 1.0e-8,
        "error_target": "exact_nominal_margin_minus_lower_envelope_intercept",
        "require_every_registered_state": True,
    }:
        raise ValueError("affine-coefficient target contract differs")
    if config["model"] != {
        "input": "factorized_state_nominal_chunk_and_relative_geometry_features",
        "shared_across_constraint_rows": True,
        "hidden_widths": [128, 128], "hidden_activation": "silu",
        "outputs": [
            "nominal_margin_mm", "gradient_x_mm_per_action",
            "gradient_y_mm_per_action", "gradient_z_mm_per_action",
            "nonnegative_error_mm",
        ],
    }:
        raise ValueError("affine-coefficient model differs")
    projection = config["projection"]
    for key in (
        "action_limit", "trust_region_linf_action", "eps_abs", "eps_rel",
        "residual_tolerance", "bound_tolerance_action",
    ):
        if not math.isfinite(float(projection[key])) or float(projection[key]) <= 0:
            raise ValueError("affine-coefficient projection is invalid")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def coefficient_grid_actions(
    nominal_xyz: Sequence[float], config: Mapping[str, Any]
) -> tuple[Any, Any, list[Any]]:
    np = _numpy()
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    settings = config["action_sampling"]
    limit = float(settings["action_limit"])
    trust = float(settings["trust_region_linf_action"])
    if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
        raise ValueError("affine-coefficient nominal action is invalid")
    lower = np.maximum(-limit, nominal - trust)
    upper = np.minimum(limit, nominal + trust)
    axes = [
        np.linspace(lower[index], upper[index], int(settings["grid_points_per_dimension"]))
        for index in range(3)
    ]
    import itertools
    candidates = [
        np.asarray(values, dtype=np.float64) for values in itertools.product(*axes)
    ]
    if len(candidates) != int(settings["expected_grid_action_count_per_state"]):
        raise ValueError("affine-coefficient grid count differs")
    return lower, upper, candidates


def coefficient_targets(
    exact_nominal_margin_m: Sequence[float], certificate: Mapping[str, Any]
) -> dict[str, Any]:
    np = _numpy()
    nominal = np.asarray(exact_nominal_margin_m, dtype=np.float64)
    intercept = np.asarray(certificate["intercept_at_nominal_m"], dtype=np.float64)
    gradient = np.asarray(certificate["gradients_m_per_action"], dtype=np.float64)
    error = nominal - intercept
    if (
        nominal.shape != (7,) or intercept.shape != (7,)
        or gradient.shape != (7, 3) or np.any(error < -1.0e-8)
        or not all(np.all(np.isfinite(value)) for value in (nominal, intercept, gradient))
    ):
        raise ValueError("affine-coefficient targets are invalid")
    error = np.maximum(error, 0.0)
    return {
        "nominal_margin_m": nominal.tolist(),
        "gradient_m_per_action": gradient.tolist(),
        "state_conditioned_error_m": error.tolist(),
        "lower_intercept_m": (nominal - error).tolist(),
    }


def affine_values(
    nominal_margin: Any, gradient: Any, error: Any,
    candidate_xyz: Any, nominal_xyz: Any,
) -> Any:
    np = _numpy()
    return (
        np.asarray(nominal_margin, dtype=np.float64)
        - np.asarray(error, dtype=np.float64)
        + np.asarray(gradient, dtype=np.float64)
        @ (
            np.asarray(candidate_xyz, dtype=np.float64)
            - np.asarray(nominal_xyz, dtype=np.float64)
        )
    )


def build_affine_coefficient_model(hidden_widths: Sequence[int]) -> Any:
    torch = _torch()
    nn = torch.nn

    class AffineCoefficientNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            previous = len(PAIR_FEATURE_NAMES)
            for width in hidden_widths:
                layers.extend((nn.Linear(previous, int(width)), nn.SiLU()))
                previous = int(width)
            layers.append(nn.Linear(previous, 5))
            self.network = nn.Sequential(*layers)

        def forward(self, values: Any) -> Any:
            return self.network(values)

    return AffineCoefficientNet()
