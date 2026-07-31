"""Strict loading of exact historical AEGIS actions for shadow replay."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


class ReplayContractError(RuntimeError):
    pass


def _canonical_historical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReplayContractError("%s must be a lowercase SHA-256" % label)
    return value


@dataclass(frozen=True)
class HistoricalStep:
    """One immutable post-step oracle from the completed Table-1 rollout."""

    step: int
    action: Tuple[float, ...]
    simulator_state_sha256: str
    reward: float
    done: bool
    goal_values: Tuple[bool, ...]


@dataclass(frozen=True)
class HistoricalActionReplay:
    case_id: str
    arm: str
    actions: Tuple[Tuple[float, ...], ...]
    steps: Tuple[HistoricalStep, ...]
    result_payload_sha256: str
    result_file_sha256: str
    executed_sequence_sha256: str
    source_policy_query_count: int
    source_policy_query_schedule_sha256: str
    settled_simulator_state_sha256: str
    initial_observation_sha256: str
    settled_active_obstacle_position_sha256: str
    policy_noise_schedule_sha256: str
    historical_task_success: bool
    historical_car_collision: bool
    historical_collision_first_step: Any
    terminal_simulator_state_sha256: str

    def provenance(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source_arm": self.arm,
            "action_count": len(self.actions),
            "historical_result_payload_sha256": self.result_payload_sha256,
            "historical_result_file_sha256": self.result_file_sha256,
            "executed_sequence_sha256": self.executed_sequence_sha256,
            "source_policy_query_count": self.source_policy_query_count,
            "source_policy_query_schedule_sha256": (
                self.source_policy_query_schedule_sha256
            ),
            "settled_simulator_state_sha256": self.settled_simulator_state_sha256,
            "initial_observation_sha256": self.initial_observation_sha256,
            "settled_active_obstacle_position_sha256": (
                self.settled_active_obstacle_position_sha256
            ),
            "policy_noise_schedule_sha256": self.policy_noise_schedule_sha256,
            "historical_task_success": self.historical_task_success,
            "historical_car_collision": self.historical_car_collision,
            "historical_collision_first_step": self.historical_collision_first_step,
            "terminal_simulator_state_sha256": self.terminal_simulator_state_sha256,
            "replay_semantics": "exact_actions[*].executed_not_nominal_raw",
        }


def load_historical_action_replay(
    path: Path,
    *,
    expected_case_id: str,
    expected_arm: str = "pi05_plus_aegis_translational",
) -> HistoricalActionReplay:
    if path.is_symlink() or not path.is_file():
        raise ReplayContractError("historical result is missing or symlinked")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayContractError("historical result is invalid JSON") from error
    if not isinstance(result, dict):
        raise ReplayContractError("historical result must contain one object")
    if result.get("case_id") != expected_case_id:
        raise ReplayContractError("historical result case differs")
    if result.get("arm") != expected_arm:
        raise ReplayContractError("historical result arm differs")
    if result.get("status") != "complete" or result.get("scientific_result") is not True:
        raise ReplayContractError("historical result is not a complete scientific result")
    expected_payload_hash = _require_hash(
        result.get("result_payload_sha256"), "historical result payload hash"
    )
    payload = {
        key: value for key, value in result.items() if key != "result_payload_sha256"
    }
    if _sha256(_canonical_historical_bytes(payload)) != expected_payload_hash:
        raise ReplayContractError("historical result payload hash differs")

    action_records = result.get("actions")
    ledger = result.get("action_invariance_ledger")
    pairing = result.get("pairing")
    metrics = result.get("metrics")
    if not isinstance(action_records, list) or not action_records:
        raise ReplayContractError("historical result has no actions")
    if not isinstance(ledger, dict) or not isinstance(pairing, dict) or not isinstance(metrics, dict):
        raise ReplayContractError("historical ledger, pairing, or metrics is absent")
    if ledger.get("action_count") != len(action_records):
        raise ReplayContractError("historical action count differs from ledger")

    actions = []
    steps = []
    for index, record in enumerate(action_records):
        if not isinstance(record, dict):
            raise ReplayContractError("historical action %d is not an object" % index)
        executed = record.get("executed")
        if (
            not isinstance(executed, list)
            or len(executed) != 7
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in executed
            )
        ):
            raise ReplayContractError("historical executed action %d is invalid" % index)
        if record.get("env_step_input", executed) != executed:
            raise ReplayContractError("historical env-step input differs at action %d" % index)
        action = tuple(float(value) for value in executed)
        actions.append(action)
        if record.get("step") != index:
            raise ReplayContractError("historical step index differs at action %d" % index)
        reward = record.get("reward")
        done = record.get("done")
        goal = record.get("goal_progress")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
            or not isinstance(done, bool)
            or not isinstance(goal, dict)
        ):
            raise ReplayContractError("historical post-step oracle %d is invalid" % index)
        before_hash = _require_hash(
            goal.get("simulator_state_sha256_before"),
            "historical step %d state-before hash" % index,
        )
        after_hash = _require_hash(
            goal.get("simulator_state_sha256_after"),
            "historical step %d state-after hash" % index,
        )
        if before_hash != after_hash or goal.get("inert") is not True:
            raise ReplayContractError(
                "historical goal telemetry was not state-inert at action %d" % index
            )
        goal_values = goal.get("values")
        if (
            not isinstance(goal_values, list)
            or not goal_values
            or any(not isinstance(value, bool) for value in goal_values)
        ):
            raise ReplayContractError("historical goal vector %d is invalid" % index)
        steps.append(
            HistoricalStep(
                step=index,
                action=action,
                simulator_state_sha256=after_hash,
                reward=float(reward),
                done=done,
                goal_values=tuple(goal_values),
            )
        )
    expected_sequence_hash = _require_hash(
        ledger.get("executed_sequence_sha256"), "executed action sequence hash"
    )
    if _sha256(_canonical_historical_bytes([list(row) for row in actions])) != expected_sequence_hash:
        raise ReplayContractError("executed action sequence hash differs")
    source_policy_query_count = ledger.get("policy_query_count")
    if (
        isinstance(source_policy_query_count, bool)
        or not isinstance(source_policy_query_count, int)
        or source_policy_query_count < 1
    ):
        raise ReplayContractError("historical policy query count is invalid")
    expected_source_query_count = (len(actions) + 4) // 5
    if source_policy_query_count != expected_source_query_count:
        raise ReplayContractError(
            "historical policy query count differs from the five-action schedule"
        )
    source_policy_query_schedule_hash = _require_hash(
        ledger.get("policy_query_schedule_sha256"),
        "historical policy query schedule hash",
    )

    settled_hash = _require_hash(
        pairing.get("settled_simulator_state_sha256"), "settled simulator state hash"
    )
    observation_hash = _require_hash(
        pairing.get("initial_observation_sha256"), "initial observation hash"
    )
    obstacle_position_hash = _require_hash(
        pairing.get("settled_active_obstacle_position_sha256"),
        "settled active-obstacle position hash",
    )
    noise_hash = _require_hash(
        pairing.get("policy_noise_schedule_sha256"), "policy noise schedule hash"
    )
    task_success = metrics.get("task_success")
    car_collision = metrics.get("paper_collision")
    if not isinstance(task_success, bool) or not isinstance(car_collision, bool):
        raise ReplayContractError("historical task/CAR metrics are invalid")
    collision_first_step = metrics.get("collision_first_step")
    if collision_first_step is not None and (
        isinstance(collision_first_step, bool)
        or not isinstance(collision_first_step, int)
        or collision_first_step < 0
    ):
        raise ReplayContractError("historical collision_first_step is invalid")
    terminal = result.get("terminal_observation")
    if not isinstance(terminal, dict):
        raise ReplayContractError("historical terminal observation is absent")
    terminal_state_hash = _require_hash(
        terminal.get("simulator_state_sha256"),
        "historical terminal simulator-state hash",
    )
    if terminal.get("after_executed_action_count") != len(actions):
        raise ReplayContractError("historical terminal action count differs")
    if terminal_state_hash != steps[-1].simulator_state_sha256:
        raise ReplayContractError("historical terminal state differs from final step")
    return HistoricalActionReplay(
        case_id=expected_case_id,
        arm=expected_arm,
        actions=tuple(actions),
        steps=tuple(steps),
        result_payload_sha256=expected_payload_hash,
        result_file_sha256=_file_sha256(path),
        executed_sequence_sha256=expected_sequence_hash,
        source_policy_query_count=source_policy_query_count,
        source_policy_query_schedule_sha256=source_policy_query_schedule_hash,
        settled_simulator_state_sha256=settled_hash,
        initial_observation_sha256=observation_hash,
        settled_active_obstacle_position_sha256=obstacle_position_hash,
        policy_noise_schedule_sha256=noise_hash,
        historical_task_success=task_success,
        historical_car_collision=car_collision,
        historical_collision_first_step=collision_first_step,
        terminal_simulator_state_sha256=terminal_state_hash,
    )
