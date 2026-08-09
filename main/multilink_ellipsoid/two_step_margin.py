"""Grouped two-step execution-margin learning for distal L5--L7 safety.

The experiment is deliberately narrow.  It changes the first Cartesian
translation in a two-action chunk, retains the immutable second action, and
learns the minimum exact-box clearance reached over both complete OSC steps.
Two paired models use the same labels and grouped split:

* ``global`` extends the earlier monolithic execution-loss MLP; and
* ``factorized`` applies one shared MLP to each of the seven constraint rows
  using relative robot--obstacle geometry.

Both models predict a nonnegative clearance loss from the interval-start
clearance.  The exact simulator remains evaluation authority, never a learned
model input at execution time.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Optional, Sequence

from .execution_margin_nn import CONSTRAINT_ORDER, _canonical, _numpy, _torch
from .obstacle_primitives import minimum_union_support_gap_witnesses


TWO_STEP_SCHEMA = "vlsa_distal_two_step_margin_moka10.v1"
TWO_STEP_DATASET_SCHEMA = "vlsa_distal_two_step_margin_moka10_dataset.v1"
TWO_STEP_DATASET_RESULT_SCHEMA = (
    "vlsa_distal_two_step_margin_moka10_dataset_result.v1"
)
TWO_STEP_TRAINING_SCHEMA = "vlsa_distal_two_step_margin_moka10_training.v1"
TWO_STEP_RESULT_SCHEMA = "vlsa_distal_two_step_margin_moka10_result.v1"
TWO_STEP_VALIDATION_SCHEMA = "vlsa_distal_two_step_margin_moka10_validation.v1"

GLOBAL_FEATURE_NAMES = tuple(
    ["q_rad_%d" % index for index in range(7)]
    + ["qdot_rad_s_%d" % index for index in range(7)]
    + ["osc_goal_position_m_%d" % index for index in range(3)]
    + ["obstacle_position_m_%d" % index for index in range(3)]
    + ["current_clearance_m_%d" % index for index in range(7)]
    + ["nominal_first_xyz_%d" % index for index in range(3)]
    + ["candidate_first_xyz_%d" % index for index in range(3)]
    + ["nominal_second_xyz_%d" % index for index in range(3)]
)

PAIR_FEATURE_NAMES = tuple(
    ["q_rad_%d" % index for index in range(7)]
    + ["qdot_rad_s_%d" % index for index in range(7)]
    + ["goal_minus_robot_center_robot_frame_m_%d" % index for index in range(3)]
    + ["obstacle_minus_robot_center_robot_frame_m_%d" % index for index in range(3)]
    + ["relative_rotation_%d" % index for index in range(9)]
    + ["robot_semiaxis_m_%d" % index for index in range(3)]
    + ["obstacle_half_size_m_%d" % index for index in range(3)]
    + ["current_clearance_m", "center_distance_m"]
    + ["constraint_one_hot_%d" % index for index in range(7)]
    + ["nominal_first_xyz_%d" % index for index in range(3)]
    + ["candidate_first_xyz_%d" % index for index in range(3)]
    + ["nominal_second_xyz_%d" % index for index in range(3)]
)

GLOBAL_CANDIDATE_SLICE = slice(len(GLOBAL_FEATURE_NAMES) - 6, len(GLOBAL_FEATURE_NAMES) - 3)
PAIR_CANDIDATE_SLICE = slice(len(PAIR_FEATURE_NAMES) - 6, len(PAIR_FEATURE_NAMES) - 3)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_two_step_config(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    try:
        config = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("two-step config is invalid JSON") from error
    required = {
        "schema_version", "protocol_id", "claim_scope",
        "source_population_manifest_sha256", "selected_manifest_sha256",
        "table1_population_run_id", "geometry_config_file_sha256",
        "exact_box_config_file_sha256", "split", "state", "sampling",
        "features", "network", "training", "calibration", "projection",
        "decision_gate",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("two-step config keys differ")
    if (
        config["schema_version"] != TWO_STEP_SCHEMA
        or config["protocol_id"] != "vlsa-distal-two-step-margin-moka10-v1"
    ):
        raise ValueError("two-step protocol differs")
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
        raise ValueError("two-step grouped split differs")
    if config["state"] != {
        "search_start_offset_actions": 20,
        "selection": "first_two_step_exact_proxy_crossing_in_registered_window",
        "candidate_state_offsets_from_first_crossing": [0, -1, -2, -3],
        "boundary_state_selection": "latest_state_with_balanced_two_step_grid",
    }:
        raise ValueError("two-step state selection differs")
    if config["sampling"] != {
        "action_limit": 1.0,
        "trust_region_linf_action": 0.5,
        "grid_points_per_dimension": 8,
        "expected_grid_action_count": 512,
        "boundary_band_m": 0.005,
        "gradient_anchor_count": 8,
        "gradient_anchor_safe_count": 4,
        "gradient_anchor_unsafe_count": 4,
        "finite_difference_epsilon_action": 0.02,
        "expected_gradient_probe_count_per_episode": 48,
        "require_matching_two_step_witness": True,
        "second_action": "immutable_nominal",
    }:
        raise ValueError("two-step sampling differs")
    if config["features"] != {
        "global_names": list(GLOBAL_FEATURE_NAMES),
        "factorized_names": list(PAIR_FEATURE_NAMES),
        "factorized_geometry": (
            "shared_constraint_MLP_with_start_state_closest_exact_box_in_"
            "each_robot_proxy_frame"
        ),
        "normalization": "train_group_mean_and_standard_deviation_with_1e-6_floor",
    }:
        raise ValueError("two-step features differ")
    if config["network"] != {
        "hidden_widths": [128, 128],
        "hidden_activation": "silu",
        "output_activation": "softplus_nonnegative_clearance_loss_mm",
        "global_output_count": 7,
        "factorized_shared_output_count": 1,
    }:
        raise ValueError("two-step network differs")
    training = config["training"]
    if training != {
        "arms": ["global", "factorized"],
        "batching": "deterministic_full_batch",
        "device": "cuda",
        "epochs": 2000,
        "gradient_loss_weight": 0.25,
        "gradient_normalization": (
            "training_valid_gradient_global_rms_with_1mm_per_action_floor"
        ),
        "huber_delta_mm": 1.0,
        "learning_rate": 0.001,
        "patience": 200,
        "seed": 20260809,
        "weight_decay": 1.0e-6,
        "sample_weighting": "equal_episode_then_equal_boundary_sign",
    }:
        raise ValueError("two-step training differs")
    if config["calibration"] != {
        "method": "per_constraint_maximum_validation_overprediction_plus_fixed_padding",
        "fixed_padding_m": 0.001,
    }:
        raise ValueError("two-step calibration differs")
    if config["projection"] != {
        "action_limit": 1.0,
        "activation_warning_m": 0.008,
        "bound_tolerance_action": 5.0e-8,
        "clearance_target_m": 0.0,
        "eps_abs": 1.0e-7,
        "eps_rel": 1.0e-7,
        "max_iter": 10000,
        "maximum_linearization_iterations": 4,
        "residual_tolerance": 5.0e-7,
        "trust_region_linf_action": 0.5,
    }:
        raise ValueError("two-step projection differs")
    if config["decision_gate"] != {
        "held_out_active_boundary_rmse": "strictly_below_current_clearance_baseline",
        "held_out_conservative_false_safe_candidate_count": 0,
        "held_out_active_gradient_cosine_similarity_minimum": 0.8,
        "held_out_projection": "valid_seven_row_QP_and_exact_safe_two_step_chunk",
        "research_direction_go": (
            "factorized_arm_passes_every_test_episode_and_primary_e05_projection"
        ),
        "closed_loop": (
            "run_only_after_research_direction_go_recompute_after_every_action_"
            "without_online_cloned_oracle"
        ),
    }:
        raise ValueError("two-step decision gate differs")
    output = json.loads(_canonical(config).decode("utf-8"))
    output["config_file_sha256"] = _sha256(raw)
    output["config_payload_sha256"] = _sha256(_canonical(config))
    return output


def load_selected_manifest(
    path: Path, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    raw = Path(path).read_bytes()
    if _sha256(raw) != config["selected_manifest_sha256"]:
        raise ValueError("two-step selected manifest hash differs")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if len(rows) != 10 or len({item.get("case_id") for item in rows}) != 10:
        raise ValueError("two-step selected cases differ")
    expected = {
        group: split
        for split, key in (
            ("train", "train_task_groups"),
            ("validation", "validation_task_groups"),
            ("test", "test_task_groups"),
        )
        for group in config["split"][key]
    }
    observed: dict[str, str] = {}
    for row in rows:
        group = str(row.get("task_level_group_id"))
        split = str(row.get("split"))
        if group in observed and observed[group] != split:
            raise ValueError("two-step task group crosses splits")
        observed[group] = split
    if observed != expected:
        raise ValueError("two-step selected split differs")
    e05 = next(item for item in rows if item["case_id"] == "vlsa-t1-goal-ii-t0-e05")
    if e05["split"] != "test":
        raise ValueError("primary E05 leaked outside two-step test")
    return rows


def grid_actions(
    nominal_xyz: Sequence[float], config: Mapping[str, Any]
) -> tuple[Any, Any, list[Any]]:
    np = _numpy()
    nominal = np.asarray(nominal_xyz, dtype=np.float64)
    if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
        raise ValueError("two-step nominal XYZ is invalid")
    settings = config["sampling"]
    limit = float(settings["action_limit"])
    trust = float(settings["trust_region_linf_action"])
    lower = np.maximum(-limit, nominal - trust)
    upper = np.minimum(limit, nominal + trust)
    axes = [
        np.linspace(lower[index], upper[index], int(settings["grid_points_per_dimension"]))
        for index in range(3)
    ]
    actions = [
        np.asarray(values, dtype=np.float64) for values in itertools.product(*axes)
    ]
    if len(actions) != int(settings["expected_grid_action_count"]):
        raise ValueError("two-step grid count differs")
    return lower, upper, actions


def boundary_category(minimum_clearance_m: Sequence[float], band_m: float) -> str:
    value = float(min(minimum_clearance_m))
    if 0.0 <= value <= band_m:
        return "boundary_safe"
    if -band_m <= value < 0.0:
        return "boundary_unsafe"
    return "far_safe" if value > band_m else "far_unsafe"


def select_balanced_anchors(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> list[int]:
    np = _numpy()
    settings = config["sampling"]
    eligible = []
    for record in records:
        action = np.asarray(record["candidate_first_xyz"], dtype=np.float64)
        epsilon = float(settings["finite_difference_epsilon_action"])
        limit = float(settings["action_limit"])
        if np.any(action - epsilon < -limit) or np.any(action + epsilon > limit):
            continue
        margin = float(min(record["minimum_substep_clearance_m"]))
        if abs(margin) <= float(settings["boundary_band_m"]):
            eligible.append((abs(margin), int(record["grid_index"]), margin))
    safe = sorted(item for item in eligible if item[2] >= 0.0)
    unsafe = sorted(item for item in eligible if item[2] < 0.0)
    safe_count = int(settings["gradient_anchor_safe_count"])
    unsafe_count = int(settings["gradient_anchor_unsafe_count"])
    if len(safe) < safe_count or len(unsafe) < unsafe_count:
        raise ValueError(
            "two-step grid lacks balanced anchors: safe=%d unsafe=%d"
            % (len(safe), len(unsafe))
        )
    selected = [item[1] for item in safe[:safe_count]] + [
        item[1] for item in unsafe[:unsafe_count]
    ]
    if len(selected) != int(settings["gradient_anchor_count"]):
        raise ValueError("two-step anchor count differs")
    return sorted(selected)


def summarize_chunk(chunk: Mapping[str, Any]) -> dict[str, Any]:
    np = _numpy()
    transitions = list(chunk.get("transitions", []))
    if len(transitions) != 2:
        raise ValueError("two-step chunk must contain exactly two transitions")
    margins = np.asarray(
        [item["minimum_substep_clearance_m"][:7] for item in transitions],
        dtype=np.float64,
    )
    minimum = np.min(margins, axis=0)
    active_steps = np.argmin(margins, axis=0)
    witnesses = []
    for row in range(7):
        step = int(active_steps[row])
        witness = transitions[step]["minimum_substep_witnesses"][row]
        witnesses.append(
            {
                "chunk_step": step,
                "substep_index": int(witness["substep_index"]),
                "obstacle_primitive_index": int(witness["obstacle_primitive_index"]),
            }
        )
    return {
        "minimum_substep_clearance_m": minimum.tolist(),
        "minimum_substep_witnesses": witnesses,
        "D_opt_proxy_safe": bool(np.all(minimum >= 0.0)),
        "D_sim_raw_safe": bool(
            all(
                item["raw_protected_contact_count"] == 0
                and item["maximum_within_step_obstacle_l1_displacement_m"] <= 1.0e-4
                for item in transitions
            )
        ),
        "raw_protected_contact_count": int(
            sum(item["raw_protected_contact_count"] for item in transitions)
        ),
        "maximum_within_step_obstacle_l1_displacement_m": float(
            max(
                item["maximum_within_step_obstacle_l1_displacement_m"]
                for item in transitions
            )
        ),
        "next_state_sha256": str(transitions[-1]["next_state_sha256"]),
        "env_step_wall_seconds": float(
            sum(item["env_step_wall_seconds"] for item in transitions)
        ),
    }


def feature_context(env: Any, probe: Any) -> dict[str, Any]:
    """Capture action-independent state and relative pair geometry once."""

    np = _numpy()
    start = probe._capture(env, 0, "two_step_feature_start")
    links = probe._ellipsoids(env)[:7]
    obstacles = probe._obstacles(env)
    clearances, witnesses = minimum_union_support_gap_witnesses(links, obstacles)
    clearances = np.asarray(clearances, dtype=np.float64)
    witnesses = np.asarray(witnesses, dtype=np.int64)
    if clearances.shape != (7,) or witnesses.shape != (7,):
        raise ValueError("two-step feature geometry differs")
    q = np.asarray(start["robot_joint_position_rad"], dtype=np.float64)
    qdot = np.asarray(start["robot_joint_velocity_rad_s"], dtype=np.float64)
    goal = np.asarray(start["controller_goal_position_m"], dtype=np.float64)
    obstacle_position = np.asarray(start["obstacle_position_m"], dtype=np.float64)
    global_prefix = np.concatenate((q, qdot, goal, obstacle_position, clearances))
    pair_prefixes = []
    for row, link in enumerate(links):
        obstacle = obstacles[int(witnesses[row])]
        relative_center = link.rotation.T @ (obstacle.center - link.center)
        relative_rotation = link.rotation.T @ obstacle.rotation
        goal_local = link.rotation.T @ (goal - link.center)
        obstacle_size = np.asarray(
            getattr(obstacle, "half_size_m", getattr(obstacle, "semiaxes_m", None)),
            dtype=np.float64,
        )
        one_hot = np.zeros(7, dtype=np.float64)
        one_hot[row] = 1.0
        pair_prefixes.append(
            np.concatenate(
                (
                    q, qdot, goal_local, relative_center,
                    relative_rotation.reshape(-1), link.semiaxes_m,
                    obstacle_size,
                    np.asarray(
                        [clearances[row], np.linalg.norm(obstacle.center - link.center)],
                        dtype=np.float64,
                    ),
                    one_hot,
                )
            )
        )
    return {
        "start_substep": start,
        "current_clearance_m": clearances,
        "start_obstacle_witness_indexes": witnesses,
        "global_prefix": global_prefix,
        "pair_prefixes": np.asarray(pair_prefixes, dtype=np.float64),
    }


def feature_vectors(
    context: Mapping[str, Any], nominal_first_xyz: Sequence[float],
    candidate_first_xyz: Sequence[float], nominal_second_xyz: Sequence[float],
) -> tuple[Any, Any]:
    np = _numpy()
    actions = np.concatenate(
        [
            np.asarray(nominal_first_xyz, dtype=np.float64),
            np.asarray(candidate_first_xyz, dtype=np.float64),
            np.asarray(nominal_second_xyz, dtype=np.float64),
        ]
    )
    global_feature = np.concatenate(
        (np.asarray(context["global_prefix"], dtype=np.float64), actions)
    )
    pair_prefixes = np.asarray(context["pair_prefixes"], dtype=np.float64)
    pair_features = np.concatenate(
        (pair_prefixes, np.repeat(actions[None, :], 7, axis=0)), axis=1
    )
    if (
        global_feature.shape != (len(GLOBAL_FEATURE_NAMES),)
        or pair_features.shape != (7, len(PAIR_FEATURE_NAMES))
        or not np.all(np.isfinite(global_feature))
        or not np.all(np.isfinite(pair_features))
    ):
        raise ValueError("two-step feature vectors are invalid")
    return global_feature, pair_features


def _build_model(model_kind: str, input_count: int, hidden_widths: Sequence[int]) -> Any:
    torch = _torch()
    nn = torch.nn
    output_count = 7 if model_kind == "global" else 1

    class TwoStepLossNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = []
            previous = int(input_count)
            for width in hidden_widths:
                layers.extend((nn.Linear(previous, int(width)), nn.SiLU()))
                previous = int(width)
            layers.append(nn.Linear(previous, output_count))
            self.network = nn.Sequential(*layers)
            self.softplus = nn.Softplus()

        def forward(self, values: Any) -> Any:
            return self.softplus(self.network(values))

    if model_kind not in {"global", "factorized"}:
        raise ValueError("two-step model kind differs")
    return TwoStepLossNet()


def _records_to_arrays(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    np = _numpy()
    global_features = np.asarray(
        [item["global_feature_vector"] for item in records], dtype=np.float64
    )
    pair_features = np.asarray(
        [item["pair_feature_vectors"] for item in records], dtype=np.float64
    )
    current = np.asarray(
        [item["current_clearance_m"] for item in records], dtype=np.float64
    )
    minimum = np.asarray(
        [item["minimum_substep_clearance_m"] for item in records], dtype=np.float64
    )
    gradients = np.full((len(records), 7, 3), np.nan, dtype=np.float64)
    gradient_valid = np.zeros((len(records), 7), dtype=bool)
    for index, record in enumerate(records):
        if record.get("gradient_m_per_action") is not None:
            gradients[index] = np.asarray(
                record["gradient_m_per_action"], dtype=np.float64
            )
            gradient_valid[index] = np.asarray(
                record["gradient_valid_rows"], dtype=bool
            )
    if (
        global_features.shape != (len(records), len(GLOBAL_FEATURE_NAMES))
        or pair_features.shape != (len(records), 7, len(PAIR_FEATURE_NAMES))
        or current.shape != (len(records), 7)
        or minimum.shape != current.shape
        or not all(
            np.all(np.isfinite(value))
            for value in (global_features, pair_features, current, minimum)
        )
        or np.any(
            np.logical_and(gradient_valid[:, :, None], ~np.isfinite(gradients))
        )
    ):
        raise ValueError("two-step learning arrays are invalid")
    loss_mm = (current - minimum) * 1000.0
    if float(np.min(loss_mm)) < -1.0e-7:
        raise ValueError("two-step interval minimum exceeds its start")
    return {
        "global_features": global_features,
        "pair_features": pair_features,
        "current": current,
        "minimum": minimum,
        "loss_mm": np.maximum(loss_mm, 0.0),
        "split": np.asarray([str(item["split"]) for item in records]),
        "case_id": np.asarray([str(item["case_id"]) for item in records]),
        "category": np.asarray([str(item["category"]) for item in records]),
        "gradients": gradients,
        "gradient_valid": gradient_valid,
    }


def _sample_weights(arrays: Mapping[str, Any], train_mask: Any) -> Any:
    np = _numpy()
    cases = arrays["case_id"][train_mask]
    categories = arrays["category"][train_mask]
    weights = np.zeros(len(cases), dtype=np.float64)
    unique_cases = sorted(set(cases.tolist()))
    for case in unique_cases:
        case_mask = cases == case
        for category in ("boundary_safe", "boundary_unsafe"):
            mask = np.logical_and(case_mask, categories == category)
            count = int(np.count_nonzero(mask))
            if count == 0:
                raise ValueError("two-step train episode lacks one boundary sign")
            weights[mask] = 1.0 / (len(unique_cases) * 2.0 * count)
    if not math.isclose(float(np.sum(weights)), 1.0, abs_tol=1.0e-12):
        raise ValueError("two-step sample weights do not sum to one")
    return weights


def _loss_and_gradients(
    model: Any, model_kind: str, values: Any, candidate_scale: Any,
    *, create_graph: bool,
) -> tuple[Any, Any]:
    torch = _torch()
    if not values.requires_grad:
        values = values.clone().detach().requires_grad_(True)
    if model_kind == "global":
        predicted = model(values)
        rows = []
        for row in range(7):
            gradient = torch.autograd.grad(
                predicted[:, row].sum(), values, create_graph=create_graph,
                retain_graph=True,
            )[0]
            rows.append(
                -gradient[:, GLOBAL_CANDIDATE_SLICE]
                / candidate_scale[None, :]
            )
        return predicted, torch.stack(rows, dim=1)
    shape = values.shape
    flat = values.reshape(-1, shape[-1])
    predicted_flat = model(flat)[:, 0]
    gradient = torch.autograd.grad(
        predicted_flat.sum(), values, create_graph=create_graph, retain_graph=True
    )[0]
    predicted = predicted_flat.reshape(shape[0], 7)
    margin_gradient = (
        -gradient[:, :, PAIR_CANDIDATE_SLICE]
        / candidate_scale[None, None, :]
    )
    return predicted, margin_gradient


def train_model(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any], *, arm: str
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    np = _numpy()
    torch = _torch()
    if arm not in config["training"]["arms"]:
        raise ValueError("two-step training arm differs")
    arrays = _records_to_arrays(records)
    masks = {
        name: arrays["split"] == name for name in ("train", "validation", "test")
    }
    if any(int(np.count_nonzero(mask)) == 0 for mask in masks.values()):
        raise ValueError("two-step learning split is empty")
    features = (
        arrays["global_features"] if arm == "global" else arrays["pair_features"]
    )
    train_values = features[masks["train"]]
    normalization_values = (
        train_values if arm == "global" else train_values.reshape(-1, train_values.shape[-1])
    )
    mean = np.mean(normalization_values, axis=0)
    standard_deviation = np.maximum(np.std(normalization_values, axis=0), 1.0e-6)
    normalized = (features - mean) / standard_deviation
    settings = config["training"]
    seed = int(settings["seed"])
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("two-step model training requires CUDA on H100")
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    input_count = len(GLOBAL_FEATURE_NAMES) if arm == "global" else len(PAIR_FEATURE_NAMES)
    model = _build_model(arm, input_count, config["network"]["hidden_widths"]).to(
        device=device, dtype=torch.float64
    )
    feature_tensor = torch.as_tensor(normalized, dtype=torch.float64, device=device)
    target = torch.as_tensor(arrays["loss_mm"], dtype=torch.float64, device=device)
    gradient_target = torch.as_tensor(
        np.where(arrays["gradient_valid"][:, :, None], arrays["gradients"], 0.0)
        * 1000.0,
        dtype=torch.float64, device=device,
    )
    gradient_valid = torch.as_tensor(
        arrays["gradient_valid"], dtype=torch.bool, device=device
    )
    mask_tensors = {
        name: torch.as_tensor(mask, dtype=torch.bool, device=device)
        for name, mask in masks.items()
    }
    weights = torch.as_tensor(
        _sample_weights(arrays, masks["train"]), dtype=torch.float64, device=device
    )
    train_gradient_values = arrays["gradients"][masks["train"]] * 1000.0
    train_gradient_mask = arrays["gradient_valid"][masks["train"]]
    expanded = np.repeat(train_gradient_mask[:, :, None], 3, axis=2)
    if not np.any(expanded):
        raise ValueError("two-step training lacks valid gradient supervision")
    gradient_scale = max(
        1.0, float(np.sqrt(np.mean(train_gradient_values[expanded] ** 2)))
    )
    candidate_slice = GLOBAL_CANDIDATE_SLICE if arm == "global" else PAIR_CANDIDATE_SLICE
    candidate_scale = torch.as_tensor(
        standard_deviation[candidate_slice], dtype=torch.float64, device=device
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    huber = torch.nn.HuberLoss(
        reduction="none", delta=float(settings["huber_delta_mm"])
    )

    def loss_for(name: str, create_graph: bool) -> tuple[Any, Any, Any]:
        selected = mask_tensors[name]
        values = feature_tensor[selected].clone().detach().requires_grad_(True)
        predicted, gradients = _loss_and_gradients(
            model, arm, values, candidate_scale, create_graph=create_graph
        )
        per_record = huber(predicted, target[selected]).mean(dim=1)
        margin_loss = (
            torch.sum(per_record * weights) if name == "train" else torch.mean(per_record)
        )
        valid = gradient_valid[selected][:, :, None].expand(-1, -1, 3)
        difference = (gradients - gradient_target[selected]) / gradient_scale
        gradient_loss = (
            torch.mean(difference[valid] ** 2)
            if torch.any(valid)
            else torch.zeros((), dtype=torch.float64, device=device)
        )
        total = margin_loss + float(settings["gradient_loss_weight"]) * gradient_loss
        return total, margin_loss, gradient_loss

    best_state = None
    best_epoch: Optional[int] = None
    best_validation = math.inf
    stale = 0
    history = []
    started = time.perf_counter_ns()
    for epoch in range(int(settings["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total, train_margin, train_gradient = loss_for("train", True)
        train_total.backward()
        optimizer.step()
        model.eval()
        with torch.enable_grad():
            validation_total, validation_margin, validation_gradient = loss_for(
                "validation", False
            )
        value = float(validation_total.detach().cpu())
        if epoch == 0 or (epoch + 1) % 100 == 0:
            history.append(
                {
                    "epoch": int(epoch),
                    "train_total": float(train_total.detach().cpu()),
                    "train_margin": float(train_margin.detach().cpu()),
                    "train_gradient": float(train_gradient.detach().cpu()),
                    "validation_total": value,
                    "validation_margin": float(validation_margin.detach().cpu()),
                    "validation_gradient": float(validation_gradient.detach().cpu()),
                }
            )
        if value < best_validation - 1.0e-12:
            best_validation = value
            best_epoch = int(epoch)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(settings["patience"]):
            break
    if best_state is None or best_epoch is None:
        raise RuntimeError("two-step training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.enable_grad():
        predicted_loss_mm, predicted_gradient_mm = _loss_and_gradients(
            model, arm, feature_tensor.clone().detach().requires_grad_(True),
            candidate_scale, create_graph=False,
        )
    predicted_loss = predicted_loss_mm.detach().cpu().numpy()
    predicted_gradient = predicted_gradient_mm.detach().cpu().numpy() / 1000.0
    predicted_margin = arrays["current"] - predicted_loss / 1000.0
    validation_overprediction = (
        predicted_margin[masks["validation"]] - arrays["minimum"][masks["validation"]]
    )
    calibration = np.maximum(0.0, np.max(validation_overprediction, axis=0)) + float(
        config["calibration"]["fixed_padding_m"]
    )
    conservative = predicted_margin - calibration[None, :]
    band = float(config["sampling"]["boundary_band_m"])

    def episode_metrics(case_id: str) -> dict[str, Any]:
        mask = arrays["case_id"] == case_id
        actual = arrays["minimum"][mask]
        predicted = predicted_margin[mask]
        lower = conservative[mask]
        baseline = arrays["current"][mask]
        active = np.argmin(actual, axis=1)
        indexes = np.arange(len(actual))
        active_actual = actual[indexes, active]
        boundary = np.abs(active_actual) <= band
        if not np.any(boundary):
            raise ValueError("two-step episode lacks boundary metrics")
        selected_indexes = indexes[boundary]
        selected_rows = active[boundary]
        error = (
            predicted[selected_indexes, selected_rows]
            - actual[selected_indexes, selected_rows]
        )
        baseline_error = (
            baseline[selected_indexes, selected_rows]
            - actual[selected_indexes, selected_rows]
        )
        false_safe = np.logical_and(lower >= 0.0, actual < 0.0)
        target_gradients = arrays["gradients"][mask]
        valid_gradients = arrays["gradient_valid"][mask]
        estimated_gradients = predicted_gradient[mask]
        cosines = []
        for local_index in selected_indexes:
            row = int(active[local_index])
            if not bool(valid_gradients[local_index, row]):
                continue
            left = target_gradients[local_index, row]
            right = estimated_gradients[local_index, row]
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator > 1.0e-12:
                cosines.append(float(np.dot(left, right) / denominator))
        return {
            "record_count": int(np.count_nonzero(mask)),
            "active_boundary_record_count": int(np.count_nonzero(boundary)),
            "active_boundary_rmse_m": float(np.sqrt(np.mean(error ** 2))),
            "active_boundary_current_clearance_baseline_rmse_m": float(
                np.sqrt(np.mean(baseline_error ** 2))
            ),
            "conservative_false_safe_candidate_count": int(
                np.count_nonzero(np.any(false_safe, axis=1))
            ),
            "active_gradient_cosine_count": len(cosines),
            "active_gradient_cosine_mean": (
                None if not cosines else float(np.mean(cosines))
            ),
            "active_gradient_cosine_minimum": (
                None if not cosines else float(np.min(cosines))
            ),
        }

    cases_by_split = {
        name: sorted(set(arrays["case_id"][mask].tolist()))
        for name, mask in masks.items()
    }
    per_episode = {
        case: episode_metrics(case)
        for case in sorted(set(arrays["case_id"].tolist()))
    }
    threshold = float(
        config["decision_gate"]["held_out_active_gradient_cosine_similarity_minimum"]
    )
    test_gate = bool(
        all(
            per_episode[case]["active_boundary_rmse_m"]
            < per_episode[case]["active_boundary_current_clearance_baseline_rmse_m"]
            and per_episode[case]["conservative_false_safe_candidate_count"] == 0
            and per_episode[case]["active_gradient_cosine_count"] > 0
            and per_episode[case]["active_gradient_cosine_mean"] >= threshold
            for case in cases_by_split["test"]
        )
    )
    serializable = {
        "model_kind": arm,
        "state_dict": best_state,
        "feature_mean": mean,
        "feature_standard_deviation": standard_deviation,
        "calibration_m": calibration,
        "feature_names": list(
            GLOBAL_FEATURE_NAMES if arm == "global" else PAIR_FEATURE_NAMES
        ),
        "constraint_order": list(CONSTRAINT_ORDER),
        "hidden_widths": list(config["network"]["hidden_widths"]),
    }
    audit = {
        "arm": arm,
        "model_class": (
            "global_seven_output_two_step_execution_loss"
            if arm == "global"
            else "shared_constraint_factorized_two_step_execution_loss"
        ),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_device_name": str(torch.cuda.get_device_name(0)),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "best_epoch": best_epoch,
        "completed_epoch_count": int(epoch + 1),
        "best_validation_objective": best_validation,
        "gradient_scale_mm_per_action": gradient_scale,
        "training_valid_gradient_row_count": int(np.count_nonzero(train_gradient_mask)),
        "history": history,
        "calibration_m": calibration.tolist(),
        "cases_by_split": cases_by_split,
        "per_episode_metrics": per_episode,
        "held_out_model_gate_pass": test_gate,
        "training_wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
    return model, serializable, audit


def save_model(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    np = _numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.tmp" % path.name)
    parameter_names = list(state["state_dict"])
    metadata = {
        "schema_version": "vlsa_distal_two_step_margin_weights.v1",
        "model_kind": state["model_kind"],
        "feature_names": list(state["feature_names"]),
        "constraint_order": list(state["constraint_order"]),
        "hidden_widths": list(state["hidden_widths"]),
        "parameter_names": parameter_names,
    }
    arrays = {
        "feature_mean": np.asarray(state["feature_mean"], dtype=np.float64),
        "feature_standard_deviation": np.asarray(
            state["feature_standard_deviation"], dtype=np.float64
        ),
        "calibration_m": np.asarray(state["calibration_m"], dtype=np.float64),
        "metadata_utf8": np.frombuffer(_canonical(metadata), dtype=np.uint8),
    }
    for index, name in enumerate(parameter_names):
        arrays["parameter_%03d" % index] = (
            state["state_dict"][name].detach().cpu().numpy().astype(np.float64)
        )
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)
    raw = path.read_bytes()
    return {"path": str(path), "file_sha256": _sha256(raw), "size_bytes": len(raw)}


def load_model(path: Path, *, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    np = _numpy()
    torch = _torch()
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(bytes(archive["metadata_utf8"].tolist()).decode("utf-8"))
        if metadata.get("schema_version") != "vlsa_distal_two_step_margin_weights.v1":
            raise ValueError("two-step model artifact schema differs")
        kind = str(metadata.get("model_kind"))
        expected_features = list(
            GLOBAL_FEATURE_NAMES if kind == "global" else PAIR_FEATURE_NAMES
        )
        if (
            metadata.get("feature_names") != expected_features
            or metadata.get("constraint_order") != list(CONSTRAINT_ORDER)
        ):
            raise ValueError("two-step model artifact identity differs")
        model = _build_model(kind, len(expected_features), metadata["hidden_widths"]).to(
            device=torch.device(device), dtype=torch.float64
        )
        names = metadata.get("parameter_names")
        if names != list(model.state_dict()):
            raise ValueError("two-step model parameter names differ")
        state_dict = {}
        for index, name in enumerate(names):
            values = np.asarray(archive["parameter_%03d" % index], dtype=np.float64)
            if values.shape != tuple(model.state_dict()[name].shape):
                raise ValueError("two-step model parameter shape differs")
            state_dict[name] = torch.as_tensor(
                values, dtype=torch.float64, device=device
            )
        model.load_state_dict(state_dict)
        model.eval()
        state = {
            "model_kind": kind,
            "feature_mean": np.asarray(archive["feature_mean"], dtype=np.float64),
            "feature_standard_deviation": np.asarray(
                archive["feature_standard_deviation"], dtype=np.float64
            ),
            "calibration_m": np.asarray(archive["calibration_m"], dtype=np.float64),
            "feature_names": expected_features,
            "constraint_order": list(CONSTRAINT_ORDER),
            "hidden_widths": list(metadata["hidden_widths"]),
        }
    return model, state


def predict_margin_and_jacobian(
    model: Any, model_state: Mapping[str, Any], context: Mapping[str, Any],
    nominal_first_xyz: Sequence[float], candidate_first_xyz: Sequence[float],
    nominal_second_xyz: Sequence[float],
) -> tuple[Any, Any, float]:
    np = _numpy()
    torch = _torch()
    global_feature, pair_features = feature_vectors(
        context, nominal_first_xyz, candidate_first_xyz, nominal_second_xyz
    )
    kind = str(model_state["model_kind"])
    feature = global_feature if kind == "global" else pair_features
    mean = np.asarray(model_state["feature_mean"], dtype=np.float64)
    standard_deviation = np.asarray(
        model_state["feature_standard_deviation"], dtype=np.float64
    )
    normalized = (feature - mean) / standard_deviation
    tensor = torch.as_tensor(
        normalized, dtype=torch.float64, device=next(model.parameters()).device
    ).clone().detach().requires_grad_(True)
    candidate_slice = GLOBAL_CANDIDATE_SLICE if kind == "global" else PAIR_CANDIDATE_SLICE
    candidate_scale = torch.as_tensor(
        standard_deviation[candidate_slice], dtype=torch.float64,
        device=tensor.device,
    )
    started = time.perf_counter_ns()
    predicted_loss_mm, margin_gradient_mm = _loss_and_gradients(
        model, kind, tensor[None, ...], candidate_scale, create_graph=False
    )
    current = np.asarray(context["current_clearance_m"], dtype=np.float64)
    margin = current - predicted_loss_mm[0].detach().cpu().numpy() / 1000.0
    jacobian = margin_gradient_mm[0].detach().cpu().numpy() / 1000.0
    return margin, jacobian, (time.perf_counter_ns() - started) * 1.0e-9


def project_first_action(
    model: Any, model_state: Mapping[str, Any], context: Mapping[str, Any],
    nominal_first_xyz: Sequence[float], nominal_second_xyz: Sequence[float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    np = _numpy()
    from .qp import MultiConstraintQp

    settings = config["projection"]
    nominal = np.asarray(nominal_first_xyz, dtype=np.float64)
    limit = float(settings["action_limit"])
    trust = float(settings["trust_region_linf_action"])
    action_lower = np.maximum(-limit, nominal - trust)
    action_upper = np.minimum(limit, nominal + trust)
    calibration = np.asarray(model_state["calibration_m"], dtype=np.float64)
    solver = MultiConstraintQp(
        eps_abs=float(settings["eps_abs"]), eps_rel=float(settings["eps_rel"]),
        max_iter=int(settings["max_iter"]),
        residual_tolerance=float(settings["residual_tolerance"]),
        bound_tolerance=float(settings["bound_tolerance_action"]),
    )
    current = nominal.copy()
    iterations = []
    started = time.perf_counter_ns()
    valid = False
    reason = "maximum_linearization_iterations_reached"
    for iteration in range(int(settings["maximum_linearization_iterations"])):
        margin, jacobian, inference = predict_margin_and_jacobian(
            model, model_state, context, nominal, current, nominal_second_xyz
        )
        conservative = margin - calibration
        if np.all(conservative >= float(settings["clearance_target_m"])):
            valid = True
            reason = "predicted_conservative_safe"
            iterations.append(
                {
                    "iteration": iteration, "linearization_xyz": current.tolist(),
                    "predicted_margin_m": margin.tolist(),
                    "conservative_margin_m": conservative.tolist(),
                    "jacobian_m_per_action": jacobian.tolist(),
                    "inference_and_jacobian_wall_seconds": inference, "qp": None,
                }
            )
            break
        lower = (
            float(settings["clearance_target_m"]) - conservative
            + jacobian @ current
        )
        qp = solver.solve(
            nominal, np.eye(3), jacobian, lower, action_lower, action_upper
        )
        iterations.append(
            {
                "iteration": iteration, "linearization_xyz": current.tolist(),
                "predicted_margin_m": margin.tolist(),
                "conservative_margin_m": conservative.tolist(),
                "jacobian_m_per_action": jacobian.tolist(),
                "inference_and_jacobian_wall_seconds": inference,
                "qp": {
                    "valid": bool(qp.valid), "reason": qp.reason,
                    "solution_xyz": None if qp.qdot_safe is None else qp.qdot_safe.tolist(),
                    "lower": lower.tolist(), "diagnostics": dict(qp.diagnostics),
                },
            }
        )
        if not qp.valid or qp.qdot_safe is None:
            reason = "two_step_neural_qp_%s" % qp.reason
            break
        next_value = np.asarray(qp.qdot_safe, dtype=np.float64)
        if np.max(np.abs(next_value - current)) <= 1.0e-10:
            current = next_value
            reason = "two_step_neural_projection_stalled"
            break
        current = next_value
    final_margin, final_jacobian, final_inference = predict_margin_and_jacobian(
        model, model_state, context, nominal, current, nominal_second_xyz
    )
    final_conservative = final_margin - calibration
    if np.all(final_conservative >= float(settings["clearance_target_m"])):
        valid = True
        reason = "predicted_conservative_safe"
    return {
        "valid": valid, "reason": reason, "nominal_xyz": nominal.tolist(),
        "projected_xyz": current.tolist(),
        "correction_l2": float(np.linalg.norm(current - nominal)),
        "action_lower": action_lower.tolist(), "action_upper": action_upper.tolist(),
        "calibration_m": calibration.tolist(),
        "final_predicted_margin_m": final_margin.tolist(),
        "final_conservative_margin_m": final_conservative.tolist(),
        "final_jacobian_m_per_action": final_jacobian.tolist(),
        "final_inference_and_jacobian_wall_seconds": final_inference,
        "iterations": iterations,
        "total_wall_seconds": (time.perf_counter_ns() - started) * 1.0e-9,
    }
