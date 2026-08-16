"""Shared Q/V model and metrics for policy-conditioned future violation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


CONFIG_SCHEMA = "vlsa_distal_policy_conditioned_value_mlp.v1"
RESULT_SCHEMA = "vlsa_distal_policy_conditioned_value_mlp_result.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def payload_sha256(value: Mapping[str, Any], key: str) -> str:
    public = dict(value)
    public.pop(key, None)
    return hashlib.sha256(canonical(public)).hexdigest()


def load_config(path: Path) -> Dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw)
    required = {
        "schema_version", "protocol_id", "claim_scope", "source_dataset",
        "model", "metrics", "decision", "forbidden",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("policy-value MLP config keys differ")
    model = value["model"]
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["protocol_id"] != "vlsa-distal-policy-conditioned-value-mlp-v1"
        or model["arms"] != ["q_only", "q_plus_v"]
        or int(model["hidden_width"]) <= 0
        or int(model["epochs"]) <= 0
        or int(model["seed"]) < 0
        or float(model["value_loss_weight"]) < 0.0
        or float(model["weight_decay"]) < 0.0
    ):
        raise ValueError("policy-value MLP protocol differs")
    output = json.loads(canonical(value).decode("utf-8"))
    output["config_file_sha256"] = hashlib.sha256(raw).hexdigest()
    output["config_payload_sha256"] = hashlib.sha256(canonical(value)).hexdigest()
    return output


def build_model(torch: Any, state_dimension: int, action_dimension: int, hidden_width: int) -> Any:
    state_dim = int(state_dimension)
    action_dim = int(action_dimension)
    width = int(hidden_width)
    if state_dim <= 0 or action_dim <= 0 or width <= 0:
        raise ValueError("policy-value model dimensions differ")

    class SharedPolicyValue(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state_encoder = torch.nn.Sequential(
                torch.nn.Linear(state_dim, width), torch.nn.SiLU(),
                torch.nn.Linear(width, width), torch.nn.SiLU(),
            )
            self.q_head = torch.nn.Sequential(
                torch.nn.Linear(width + action_dim, width), torch.nn.SiLU(),
                torch.nn.Linear(width, 1),
            )
            self.v_head = torch.nn.Sequential(
                torch.nn.Linear(width, width), torch.nn.SiLU(),
                torch.nn.Linear(width, 1),
            )

        def encode(self, state: Any) -> Any:
            return self.state_encoder(state)

        def q(self, state: Any, action: Any) -> Any:
            return self.q_head(torch.cat((self.encode(state), action), dim=-1))

        def v(self, state: Any) -> Any:
            return self.v_head(self.encode(state))

    return SharedPolicyValue()


def state_balanced_weights(samples: Sequence[Mapping[str, Any]]) -> Sequence[float]:
    """Give every (episode state, constraint) pair equal total weight."""

    keys = [
        (str(sample["state_id"]), str(sample["constraint"]))
        for sample in samples
    ]
    counts = {key: keys.count(key) for key in set(keys)}
    weights = [1.0 / float(counts[key]) for key in keys]
    total = sum(weights)
    if not weights or total <= 0.0:
        raise ValueError("policy-value sample weights differ")
    return [weight * len(weights) / total for weight in weights]


def weighted_mean_scale(
    values: Any, weights: Any, minimum_scale: float, fallback_scale: float,
) -> Tuple[Any, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    raw_weights = np.asarray(weights, dtype=np.float64)
    normalized = raw_weights / np.sum(raw_weights)
    mean = np.sum(array * normalized[:, None], axis=0)
    variance = np.sum((array - mean) ** 2 * normalized[:, None], axis=0)
    scale = np.where(
        np.sqrt(variance) >= float(minimum_scale),
        np.sqrt(variance), float(fallback_scale),
    )
    return mean, scale


def risk_metrics(
    samples: Sequence[Mapping[str, Any]], predictions: Sequence[float],
    near_boundary_abs_risk: float,
) -> Dict[str, Any]:
    actual = [float(sample["target"]) for sample in samples]
    guessed = [float(item) for item in predictions]
    if not actual or len(actual) != len(guessed):
        raise ValueError("policy-value metric input differs")
    near = [
        index for index, target in enumerate(actual)
        if abs(target) <= float(near_boundary_abs_risk)
    ]
    safe = [target <= 0.0 for target in actual]
    predicted_safe = [target <= 0.0 for target in guessed]
    groups = {}
    for constraint in sorted({str(sample["constraint"]) for sample in samples}):
        indexes = [
            index for index, sample in enumerate(samples)
            if str(sample["constraint"]) == constraint
        ]
        groups[constraint] = {
            "sample_count": len(indexes),
            "false_safe_count": sum(
                predicted_safe[index] and not safe[index] for index in indexes
            ),
            "rmse": math.sqrt(sum(
                (guessed[index] - actual[index]) ** 2 for index in indexes
            ) / len(indexes)),
        }
    candidate_groups = {}
    for index, sample in enumerate(samples):
        candidate_key = (
            str(sample["state_id"]),
            str(sample.get("trajectory_id", sample.get("candidate_name", index))),
            int(sample.get("action_offset", -1)),
        )
        candidate_groups.setdefault(candidate_key, []).append(index)
    candidate_outcomes = {}
    for key, indexes in candidate_groups.items():
        exact_candidate_safe = all(safe[index] for index in indexes)
        predicted_candidate_safe = all(predicted_safe[index] for index in indexes)
        candidate_outcomes[key] = {
            "exact_safe": exact_candidate_safe,
            "predicted_safe": predicted_candidate_safe,
            "false_safe": predicted_candidate_safe and not exact_candidate_safe,
        }
    states = {}
    for state_id in sorted({str(sample["state_id"]) for sample in samples}):
        indexes = [
            index for index, sample in enumerate(samples)
            if str(sample["state_id"]) == state_id
        ]
        state_candidates = [
            outcome for key, outcome in candidate_outcomes.items()
            if key[0] == state_id
        ]
        exact_safe = any(outcome["exact_safe"] for outcome in state_candidates)
        jointly_safe_prediction = any(
            outcome["exact_safe"] and outcome["predicted_safe"]
            for outcome in state_candidates
        )
        states[state_id] = {
            "exact_safe_support": exact_safe,
            "correctly_predicted_safe_support": jointly_safe_prediction,
            "false_safe_count": sum(
                bool(outcome["false_safe"]) for outcome in state_candidates
            ),
        }
    squared = [
        (prediction - target) ** 2
        for prediction, target in zip(guessed, actual)
    ]
    near_squared = [
        (guessed[index] - actual[index]) ** 2 for index in near
    ]
    exact_safe_count = sum(safe)
    return {
        "sample_count": len(actual),
        "rmse": math.sqrt(sum(squared) / len(squared)),
        "near_boundary_sample_count": len(near),
        "near_boundary_rmse": None if not near else math.sqrt(
            sum(near_squared) / len(near_squared)
        ),
        "constraint_false_safe_count": sum(
            proposed and not truth for proposed, truth in zip(predicted_safe, safe)
        ),
        "constraint_false_unsafe_count": sum(
            not proposed and truth for proposed, truth in zip(predicted_safe, safe)
        ),
        "false_safe_count": sum(
            bool(outcome["false_safe"]) for outcome in candidate_outcomes.values()
        ),
        "false_unsafe_count": sum(
            outcome["exact_safe"] and not outcome["predicted_safe"]
            for outcome in candidate_outcomes.values()
        ),
        "safe_recall": None if not exact_safe_count else sum(
            proposed and truth
            for proposed, truth in zip(predicted_safe, safe)
        ) / exact_safe_count,
        "recoverable_state_count": sum(
            bool(item["exact_safe_support"]) for item in states.values()
        ),
        "safe_support_state_count": sum(
            bool(item["correctly_predicted_safe_support"])
            for item in states.values()
        ),
        "constraints": groups,
        "states": states,
    }
