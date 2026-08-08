from __future__ import annotations

import asyncio
from collections.abc import Mapping
import http
import logging
import math
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                obs, crfs_control = _extract_crfs_control(obs)

                infer_time = time.monotonic()
                if crfs_control is None:
                    # Preserve the upstream path exactly when no opt-in control
                    # envelope is present.
                    action = self._policy.infer(obs)
                else:
                    if "embodisteer_guidance" in crfs_control:
                        action = self._policy.infer(
                            obs,
                            rng_seed=crfs_control["rng_seed"],
                            embodisteer_guidance=crfs_control[
                                "embodisteer_guidance"
                            ],
                        )
                    elif "flow_guidance" in crfs_control:
                        action = self._policy.infer(
                            obs,
                            rng_seed=crfs_control["rng_seed"],
                            flow_guidance=crfs_control["flow_guidance"],
                        )
                    else:
                        action = self._policy.infer(
                            obs, rng_seed=crfs_control["rng_seed"]
                        )
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _extract_crfs_control(obs):
    """Remove and validate the reserved CRFS experiment-control envelope.

    Ordinary clients never send ``__crfs__`` and therefore take the exact
    upstream inference path. Benchmark clients may opt in to a per-request JAX
    seed so paired policy arms receive the same flow-matching noise without
    exposing experiment controls to the normal input transforms. The optional
    predictive-flow payload is validated here and is likewise removed before
    any ordinary transform sees the observation.
    """

    if not isinstance(obs, dict):
        raise TypeError("Policy observation must be a dictionary")

    control = obs.pop("__crfs__", None)
    if control is None:
        return obs, None
    if not isinstance(control, Mapping):
        raise TypeError("__crfs__ must be a mapping")
    if "rng_seed" not in control or not set(control).issubset(
        {"rng_seed", "flow_guidance", "embodisteer_guidance"}
    ):
        raise ValueError(
            "__crfs__ requires rng_seed and one optional guidance payload"
        )
    if "flow_guidance" in control and "embodisteer_guidance" in control:
        raise ValueError("__crfs__ guidance payloads are mutually exclusive")

    rng_seed = control["rng_seed"]
    if isinstance(rng_seed, bool) or not isinstance(rng_seed, int):
        raise TypeError("__crfs__.rng_seed must be an integer")
    if not 0 <= rng_seed <= 0xFFFFFFFF:
        raise ValueError("__crfs__.rng_seed must be in [0, 2**32 - 1]")

    output = {"rng_seed": rng_seed}
    if "flow_guidance" in control:
        output["flow_guidance"] = _validate_flow_guidance(
            control["flow_guidance"]
        )
    if "embodisteer_guidance" in control:
        output["embodisteer_guidance"] = _validate_embodisteer_guidance(
            control["embodisteer_guidance"]
        )
    return obs, output


def _validate_flow_guidance(value):
    """Fail closed on the bounded, JSON-like flow-guidance envelope."""

    if not isinstance(value, Mapping):
        raise TypeError("__crfs__.flow_guidance must be a mapping")
    expected = {
        "schema_version",
        "action_horizon",
        "action_dimensions",
        "nominal_output_actions",
        "delta_rows",
        "delta_lower",
        "projection_sweeps",
        "projection_tolerance",
    }
    if set(value) != expected:
        raise ValueError("__crfs__.flow_guidance keys differ")
    if value["schema_version"] != "crfs_predictive_flow_guidance.v1":
        raise ValueError("__crfs__.flow_guidance schema differs")
    if value["action_horizon"] != 10 or isinstance(
        value["action_horizon"], bool
    ):
        raise ValueError("__crfs__.flow_guidance horizon must equal ten")
    if value["action_dimensions"] != [0, 1, 2]:
        raise ValueError("__crfs__.flow_guidance must act on exactly XYZ")
    if value["projection_sweeps"] != 64 or isinstance(
        value["projection_sweeps"], bool
    ):
        raise ValueError("__crfs__.flow_guidance projection sweeps differ")

    tolerance = value["projection_tolerance"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or not 0.0 < float(tolerance) <= 0.01
    ):
        raise ValueError("__crfs__.flow_guidance tolerance is invalid")

    nominal = value["nominal_output_actions"]
    if not isinstance(nominal, list) or len(nominal) != 10:
        raise ValueError("__crfs__.flow_guidance nominal chunk shape differs")
    _validate_numeric_matrix(nominal, width=7, label="nominal_output_actions")
    rows = value["delta_rows"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 128:
        raise ValueError("__crfs__.flow_guidance row count is invalid")
    _validate_numeric_matrix(rows, width=30, label="delta_rows", nonzero=True)
    lower = value["delta_lower"]
    if not isinstance(lower, list) or len(lower) != len(rows):
        raise ValueError("__crfs__.flow_guidance lower-bound shape differs")
    _validate_numeric_vector(lower, label="delta_lower")
    return {
        "schema_version": value["schema_version"],
        "action_horizon": 10,
        "action_dimensions": [0, 1, 2],
        "nominal_output_actions": nominal,
        "delta_rows": rows,
        "delta_lower": lower,
        "projection_sweeps": 64,
        "projection_tolerance": float(tolerance),
    }


def _validate_embodisteer_guidance(value):
    """Fail closed on the task-metric multi-CBF flow envelope."""

    if not isinstance(value, Mapping):
        raise TypeError("__crfs__.embodisteer_guidance must be a mapping")
    expected = {
        "schema_version",
        "action_horizon",
        "action_dimensions",
        "nominal_output_actions",
        "delta_rows",
        "delta_lower",
        "task_metric_directions",
        "task_metric_condition_number",
        "guidance_schedule",
        "projection_sweeps",
        "projection_tolerance",
    }
    if set(value) != expected:
        raise ValueError("__crfs__.embodisteer_guidance keys differ")
    if value["schema_version"] != "crfs_embodisteer_multicbf_guidance.v1":
        raise ValueError("__crfs__.embodisteer_guidance schema differs")
    if value["action_horizon"] != 10 or isinstance(
        value["action_horizon"], bool
    ):
        raise ValueError("__crfs__.embodisteer_guidance horizon differs")
    if value["action_dimensions"] != [0, 1, 2]:
        raise ValueError("__crfs__.embodisteer_guidance must act on XYZ")
    if value["projection_sweeps"] != 64 or isinstance(
        value["projection_sweeps"], bool
    ):
        raise ValueError("__crfs__.embodisteer_guidance sweeps differ")

    tolerance = value["projection_tolerance"]
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or not 0.0 < float(tolerance) <= 0.01
    ):
        raise ValueError("__crfs__.embodisteer_guidance tolerance is invalid")
    nominal = value["nominal_output_actions"]
    if not isinstance(nominal, list) or len(nominal) != 10:
        raise ValueError("__crfs__.embodisteer_guidance nominal shape differs")
    _validate_numeric_matrix(nominal, width=7, label="nominal_output_actions")
    rows = value["delta_rows"]
    directions = value["task_metric_directions"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= 128:
        raise ValueError("__crfs__.embodisteer_guidance row count differs")
    if not isinstance(directions, list) or len(directions) != len(rows):
        raise ValueError("__crfs__.embodisteer_guidance direction count differs")
    _validate_numeric_matrix(rows, width=30, label="delta_rows", nonzero=True)
    _validate_numeric_matrix(
        directions, width=30, label="task_metric_directions", nonzero=True
    )
    lower = value["delta_lower"]
    if not isinstance(lower, list) or len(lower) != len(rows):
        raise ValueError("__crfs__.embodisteer_guidance lower shape differs")
    _validate_numeric_vector(lower, label="delta_lower")
    condition = value["task_metric_condition_number"]
    if (
        isinstance(condition, bool)
        or not isinstance(condition, (int, float))
        or not math.isfinite(float(condition))
        or not 1.0 <= float(condition) <= 1.0e12
    ):
        raise ValueError("__crfs__.embodisteer_guidance condition is invalid")
    schedule = value["guidance_schedule"]
    if not isinstance(schedule, Mapping) or set(schedule) != {
        "base_strength",
        "beta",
        "transition",
    }:
        raise ValueError("__crfs__.embodisteer_guidance schedule differs")
    expected_schedule = {"base_strength": 1.0, "beta": 50.0, "transition": 0.7}
    if any(float(schedule[key]) != expected_schedule[key] for key in expected_schedule):
        raise ValueError("__crfs__.embodisteer_guidance schedule values differ")
    return {
        "schema_version": value["schema_version"],
        "action_horizon": 10,
        "action_dimensions": [0, 1, 2],
        "nominal_output_actions": nominal,
        "delta_rows": rows,
        "delta_lower": lower,
        "task_metric_directions": directions,
        "task_metric_condition_number": float(condition),
        "guidance_schedule": dict(expected_schedule),
        "projection_sweeps": 64,
        "projection_tolerance": float(tolerance),
    }


def _validate_numeric_matrix(value, *, width, label, nonzero=False):
    for row in value:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("__crfs__.flow_guidance %s shape differs" % label)
        _validate_numeric_vector(row, label=label)
        if nonzero and sum(float(item) * float(item) for item in row) <= 1.0e-20:
            raise ValueError("__crfs__.flow_guidance %s contains a zero row" % label)


def _validate_numeric_vector(value, *, label):
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or abs(float(item)) > 1.0e6
        ):
            raise ValueError(
                "__crfs__.flow_guidance %s contains an invalid number" % label
            )


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
